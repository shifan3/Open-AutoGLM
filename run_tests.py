#!/usr/bin/env python3
"""
Phone Agent Test Harness - Automated phone app testing with natural language test cases.

Reads .md test case files from a directory, runs each through the Phone Agent,
and reports pass/fail results based on VLM verification of expected outcomes.

Usage:
    python run_tests.py ./tests
    python run_tests.py ./tests --device-type ios --base-url http://10.0.0.5:8000/v1
    python run_tests.py ./tests --max-steps 30 --verbose
"""

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Ensure stdout supports Unicode (required on Windows with GBK terminals)
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import yaml

from openai import OpenAI

from phone_agent import PhoneAgent
from phone_agent.agent import AgentConfig
from phone_agent.agent_ios import IOSAgentConfig, IOSPhoneAgent
from phone_agent.device_factory import DeviceType, get_device_factory, set_device_type
from phone_agent.model import ModelConfig


# ── Test case representation ───────────────────────────────────────────────


@dataclass
class TestCase:
    """A single test case parsed from a .md file."""

    name: str
    file_path: Path
    task: str
    expected: str
    body: str = ""
    max_steps: int | None = None
    lang: str = "cn"


@dataclass
class TestResult:
    """Result of running a single test case."""

    test: TestCase
    passed: bool
    steps_taken: int
    duration_seconds: float
    agent_message: str = ""
    verification_reason: str = ""
    error: str = ""
    test_steps: list[str] | None = None
    artifacts: list[str] | None = None
    device_id: str | None = None
    device_name: str = ""


@dataclass
class DeviceWakeState:
    """Original device display settings to restore after tests."""

    device_id: str | None
    screen_off_timeout: str | None = None
    stay_on_while_plugged_in: str | None = None


@dataclass
class TestRunOptions:
    """Shared options for running tests on one device."""

    base_url: str
    api_key: str
    model_name: str
    lang: str
    max_steps: int
    wda_url: str
    verbose: bool
    quiet: bool
    total_runs: int
    run_offsets: dict[str | None, int] = field(default_factory=dict)


# ── Markdown frontmatter parser ────────────────────────────────────────────


def parse_test_file(file_path: Path) -> TestCase:
    """Parse a .md test case file and extract task + expected from YAML frontmatter."""
    content = file_path.read_text(encoding="utf-8")

    if not content.startswith("---"):
        raise ValueError(f"Missing YAML frontmatter in {file_path.name}")

    parts = content.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"Malformed frontmatter in {file_path.name}")

    metadata = yaml.safe_load(parts[1])

    if not metadata or "task" not in metadata:
        raise ValueError(f"Missing 'task' field in {file_path.name} frontmatter")
    if "expected" not in metadata:
        raise ValueError(f"Missing 'expected' field in {file_path.name} frontmatter")

    # Use the first heading as test name, or fall back to filename
    body = parts[2].strip()
    body_lines = body.split("\n")
    name = file_path.stem
    for line in body_lines:
        if line.startswith("# "):
            name = line.lstrip("# ").strip()
            break

    return TestCase(
        name=name,
        file_path=file_path,
        task=metadata["task"],
        expected=metadata["expected"],
        body=body,
        max_steps=metadata.get("max_steps"),
        lang=metadata.get("lang", "cn"),
    )


# ── Verification helper ────────────────────────────────────────────────────


def verify_result(
    client: OpenAI,
    model_name: str,
    screenshot_b64: str,
    screen_info: str,
    expected: str,
    lang: str,
    artifacts: dict | None = None,
) -> tuple[bool, str]:
    """
    Verify that the final screen matches the expected outcome by asking the VLM.

    Returns (passed, reason).
    """
    artifacts = artifacts or {}
    artifact_names = ", ".join(artifacts.keys()) or ("无" if lang == "cn" else "None")

    prompt = (
        (
            "请由你根据图片内容判断测试结果是否符合预期，不要依赖本地代码判断。\n\n"
            f"预期：{expected}\n\n"
            f"当前屏幕信息：{screen_info}\n\n"
            f"测试过程中保存的命名图片：{artifact_names}\n\n"
            "始终会提供最后一步的当前屏幕截图。"
            "如果测试过程中通过 save(@name) 保存了额外图片，也会在下方以对应标签提供。"
            "如果预期中引用了 {@name} 这样的命名图片，请使用同名标签图片进行判断；"
            "即使预期没有显式引用命名图片，也可以结合所有提供的图片进行判断。"
            "请只回答 YES 或 NO，并给出简短理由。"
        )
        if lang == "cn"
        else (
            "Judge whether the test result matches the expected outcome from the provided images. "
            "Do not rely on local code for the judgment.\n\n"
            f"Expected: {expected}\n\n"
            f"Screen info: {screen_info}\n\n"
            f"Named images saved during the test: {artifact_names}\n\n"
            "The final current-screen screenshot is always provided. "
            "If the test captured extra images using save(@name), they are also provided below with matching labels. "
            "If expected references names like {@name}, use the image with the same label; "
            "even when expected does not explicitly reference named images, you may use all provided images for judgment. "
            "Answer only YES or NO with a brief reason."
        )
    )

    content = [
        {"type": "text", "text": "Current final screen:"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
        },
    ]

    for name, artifact in artifacts.items():
        content.extend(
            [
                {"type": "text", "text": f"Saved image {name}:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{artifact.base64_data}"},
                },
            ]
        )

    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=200,
        temperature=0.0,
        stream=False,
    )

    content = response.choices[0].message.content.strip()

    passed = content.upper().startswith("YES")
    return passed, content


def plan_test_steps(
    client: OpenAI,
    model_name: str,
    test: TestCase,
) -> list[str]:
    """
    Ask the LLM to decompose a test case into structured execution steps.

    The returned array is injected into every later agent call so the model can
    report progress and useful state while it drives the phone.
    """
    body_steps = _extract_numbered_steps(test.body)
    if body_steps:
        return body_steps

    if test.lang == "cn":
        prompt = (
            "请把下面的手机自动化测试用例拆解成结构化测试步骤数组。\n"
            "要求:\n"
            "1. 只返回 JSON，不要 Markdown，不要解释。\n"
            "2. JSON 格式必须是: {\"steps\": [\"步骤1\", \"步骤2\"]}\n"
            "3. 每个步骤应该可观察、可执行，并覆盖预期结果验证前的主要流程。\n"
            "4. 尽量保留用户原文措辞，不要重新描述、扩写或总结步骤。\n"
            "5. 除非任务本身只有一步，否则至少拆成 5 个步骤。\n\n"
            f"测试名称: {test.name}\n"
            f"任务: {test.task}\n"
            f"预期结果: {test.expected}\n"
        )
    else:
        prompt = (
            "Decompose this phone automation test case into a structured array of test steps.\n"
            "Rules:\n"
            "1. Return JSON only, no Markdown, no explanation.\n"
            "2. The JSON format must be: {\"steps\": [\"step 1\", \"step 2\"]}\n"
            "3. Each step should be observable, executable, and cover the main flow before final verification.\n"
            "4. Preserve the user's original wording as much as possible. Do not rewrite, expand, or summarize steps.\n"
            "5. Unless the task is truly one-step, produce at least 5 steps.\n\n"
            f"Test name: {test.name}\n"
            f"Task: {test.task}\n"
            f"Expected outcome: {test.expected}\n"
        )

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200,
        temperature=0.0,
        stream=False,
    )
    content = response.choices[0].message.content.strip()
    payload = _load_json_object(content)
    steps = payload.get("steps") if payload else None

    if not isinstance(steps, list):
        return [test.task]

    parsed_steps = [step.strip() for step in steps if isinstance(step, str) and step.strip()]
    return parsed_steps or [test.task]


def strip_framework_step_markers(steps: list[str]) -> list[str]:
    """Remove framework-only markers before passing steps to the LLM."""
    cleaned_steps = []
    for step in steps:
        cleaned = re.sub(r"\s*save\(\s*@[^)]+\s*\)", "", step).strip()
        cleaned_steps.append(cleaned or step)
    return cleaned_steps


def _load_json_object(content: str) -> dict | None:
    """Load a JSON object, tolerating fenced or prefixed model output."""
    try:
        payload = json.loads(content)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fenced:
        try:
            payload = json.loads(fenced.group(1))
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            pass

    object_match = re.search(r"\{.*\}", content, re.DOTALL)
    if object_match:
        try:
            payload = json.loads(object_match.group(0))
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            return None

    return None


def _extract_numbered_steps(text: str) -> list[str]:
    """Extract numbered test steps from markdown body as a fallback."""
    if not text:
        return []

    task_section = _extract_markdown_section(text, ["任务", "Task"])
    source = task_section or text

    matches = list(re.finditer(r"(?:^|\n|\s)(\d+)[\.、]\s*", source))
    steps: list[str] = []

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        step = source[start:end].strip()
        step = step.strip("- \t\r\n")
        if step:
            steps.append(step)

    return steps


def _extract_markdown_section(text: str, headings: list[str]) -> str:
    """Extract the body under the first matching markdown heading."""
    heading_pattern = "|".join(re.escape(heading) for heading in headings)
    pattern = rf"(?im)^##+\s*({heading_pattern})\s*$"
    match = re.search(pattern, text)
    if not match:
        return ""

    start = match.end()
    next_heading = re.search(r"(?m)^##+\s+", text[start:])
    end = start + next_heading.start() if next_heading else len(text)
    return text[start:end].strip()


# ── Test runner ─────────────────────────────────────────────────────────────


def run_single_test(
    test: TestCase,
    client: OpenAI,
    model_config: ModelConfig,
    base_agent_config: AgentConfig | IOSAgentConfig,
    global_max_steps: int,
    device_type: DeviceType,
    verbose: bool,
    device_name: str = "",
) -> TestResult:
    """Run a single test case and return the result."""
    start_time = time.time()

    try:
        # Determine max steps: test-level override > global config
        max_steps = test.max_steps or global_max_steps
        try:
            test_steps = plan_test_steps(
                client=client,
                model_name=model_config.model_name,
                test=test,
            )
        except Exception as e:
            print(f"[WARN] Failed to plan test steps for {test.name}: {e}")
            test_steps = [test.task]
        agent_test_steps = strip_framework_step_markers(test_steps)

        if device_type == DeviceType.IOS:
            agent_config = IOSAgentConfig(
                max_steps=max_steps,
                wda_url=base_agent_config.wda_url,
                device_id=base_agent_config.device_id,
                lang=test.lang,
                verbose=verbose,
                test_steps=agent_test_steps,
                artifact_steps=test_steps,
            )
            agent = IOSPhoneAgent(
                model_config=model_config,
                agent_config=agent_config,
            )
        else:
            agent_config = AgentConfig(
                max_steps=max_steps,
                device_id=base_agent_config.device_id,
                lang=test.lang,
                verbose=verbose,
                test_steps=agent_test_steps,
                artifact_steps=test_steps,
            )
            agent = PhoneAgent(
                model_config=model_config,
                agent_config=agent_config,
            )

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  Running: {test.name}")
            print(f"  Device: {device_name or base_agent_config.device_id or 'default'}")
            print(f"  Task: {test.task}")
            print(f"  Expected: {test.expected}")
            print(f"  Max steps: {max_steps}")
            print("  Planned steps:")
            for i, step in enumerate(test_steps, 1):
                print(f"    {i}. {step}")
            print(f"{'=' * 60}")

        # Phase 1: Execute the task
        agent_message = agent.run(test.task)
        steps_taken = agent.step_count

        # Phase 2: Verify the screen that the agent used for its final decision.
        screenshot = agent.last_screenshot
        current_app = agent.last_current_app

        if screenshot is None:
            if device_type == DeviceType.IOS:
                from phone_agent.xctest import get_current_app, get_screenshot

                screenshot = get_screenshot(
                    wda_url=agent_config.wda_url,
                    session_id=agent_config.session_id,
                    device_id=agent_config.device_id,
                )
                current_app = get_current_app(
                    wda_url=agent_config.wda_url,
                    session_id=agent_config.session_id,
                )
            else:
                device_factory = get_device_factory()
                screenshot = device_factory.get_screenshot(base_agent_config.device_id)
                current_app = device_factory.get_current_app(base_agent_config.device_id)

        screen_info_parts = [f"current_app: {current_app}"]
        screen_info = "\n".join(screen_info_parts)
        artifacts = agent.saved_artifacts
        if verbose and artifacts:
            print("  Saved artifacts:")
            for name in artifacts:
                print(f"    {name}")

        passed, reason = verify_result(
            client=client,
            model_name=model_config.model_name,
            screenshot_b64=screenshot.base64_data,
            screen_info=screen_info,
            expected=test.expected,
            lang=test.lang,
            artifacts=artifacts,
        )

        elapsed = time.time() - start_time

        return TestResult(
            test=test,
            passed=passed,
            steps_taken=steps_taken,
            duration_seconds=elapsed,
            agent_message=agent_message,
            verification_reason=reason,
            test_steps=test_steps,
            artifacts=list(artifacts.keys()),
            device_id=agent_config.device_id,
            device_name=device_name,
        )

    except Exception as e:
        elapsed = time.time() - start_time
        return TestResult(
            test=test,
            passed=False,
            steps_taken=0,
            duration_seconds=elapsed,
            error=f"{type(e).__name__}: {e}",
            device_id=base_agent_config.device_id,
            device_name=device_name,
        )


# ── Report formatting ──────────────────────────────────────────────────────


def format_duration(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.0f}s"


def print_results(results: list[TestResult]) -> None:
    """Print formatted test results."""
    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count

    print("\n" + "=" * 70)
    print(f"  Test Results: {passed_count} passed, {failed_count} failed, "
          f"{len(results)} total")
    print("=" * 70)

    for i, result in enumerate(results, 1):
        status = "PASS" if result.passed else "FAIL"
        icon = "[PASS]" if result.passed else "[FAIL]"
        duration = format_duration(result.duration_seconds)

        device_label = result.device_name or result.device_id or "default"
        print(f"\n  [{i}] {icon} {result.test.name} | Device: {device_label}")

        if result.passed:
            print(f"      Steps: {result.steps_taken} | Duration: {duration}")
            if result.verification_reason:
                reason_oneline = result.verification_reason.replace("\n", " ")
                print(f"      Reason: {reason_oneline}")
        else:
            if result.error:
                print(f"      Error: {result.error}")
            else:
                print(f"      Steps: {result.steps_taken} | Duration: {duration}")
                print(f"      Agent: {result.agent_message[:120]}")
                if result.verification_reason:
                    reason_oneline = result.verification_reason.replace("\n", " ")
                    print(f"      Reason: {reason_oneline}")

    print("\n" + "=" * 70)
    if passed_count == len(results):
        print("  All tests passed!")
    else:
        print(f"  FAILURES: {failed_count} of {len(results)} tests failed")
        for r in results:
            if not r.passed:
                summary = r.error or r.verification_reason[:100]
                device_label = r.device_name or r.device_id or "default"
                print(f"    - [{device_label}] {r.test.name}: {summary}")
    print("=" * 70 + "\n")


def write_json_results(results: list[TestResult], output_path: str) -> None:
    """Write structured test results to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    passed_count = sum(1 for result in results if result.passed)
    payload = {
        "summary": {
            "total": len(results),
            "passed": passed_count,
            "failed": len(results) - passed_count,
        },
        "results": [
            {
                "test_name": result.test.name,
                "test_file": str(result.test.file_path),
                "task": result.test.task,
                "expected": result.test.expected,
                "device_id": result.device_id,
                "device_name": result.device_name,
                "passed": result.passed,
                "steps_taken": result.steps_taken,
                "duration_seconds": result.duration_seconds,
                "agent_message": result.agent_message,
                "verification_reason": result.verification_reason,
                "error": result.error,
                "test_steps": result.test_steps or [],
                "artifacts": result.artifacts or [],
            }
            for result in results
        ],
    }

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def discover_target_devices(
    device_type: DeviceType, explicit_device_id: str | None
) -> list[dict[str, str | None]]:
    """Resolve which devices should run the test suite."""
    if explicit_device_id:
        return [{"device_id": explicit_device_id, "device_name": explicit_device_id}]

    if device_type == DeviceType.IOS:
        return [{"device_id": None, "device_name": "iOS default"}]

    device_factory = get_device_factory()
    devices = [
        device
        for device in device_factory.list_devices()
        if getattr(device, "status", "") == "device"
    ]

    if not devices:
        return [{"device_id": None, "device_name": "default"}]

    targets = []
    for device in devices:
        model = getattr(device, "model", None)
        device_id = getattr(device, "device_id", None)
        label = f"{model or 'Android'} ({device_id})" if device_id else model or "Android"
        targets.append({"device_id": device_id, "device_name": label})

    return targets


def build_base_agent_config(
    args: argparse.Namespace, device_type: DeviceType, device_id: str | None
) -> AgentConfig | IOSAgentConfig:
    """Build base agent config for one target device."""
    if device_type == DeviceType.IOS:
        return IOSAgentConfig(
            max_steps=args.max_steps,
            wda_url=args.wda_url,
            device_id=device_id,
            lang=args.lang,
            verbose=False,
        )

    return AgentConfig(
        max_steps=args.max_steps,
        device_id=device_id,
        lang=args.lang,
        verbose=False,
    )


def run_tests_for_device(
    target: dict[str, str | None],
    tests: list[TestCase],
    device_type: DeviceType,
    options: TestRunOptions,
) -> list[TestResult]:
    """Run all tests sequentially on one device."""
    local_client = OpenAI(
        base_url=options.base_url,
        api_key=options.api_key,
        timeout=60.0,
    )
    model_config = ModelConfig(
        base_url=options.base_url,
        model_name=options.model_name,
        api_key=options.api_key,
        lang=options.lang,
        verbose=options.verbose,
    )

    base_agent_config = build_base_agent_config_from_options(
        options=options,
        device_type=device_type,
        device_id=target["device_id"],
    )

    device_results: list[TestResult] = []
    offset = options.run_offsets.get(target["device_id"], 0)
    for local_index, test in enumerate(tests, 1):
        run_index = offset + local_index
        if not options.quiet:
            print(
                f"\n[{run_index}/{options.total_runs}] Running: {test.name} "
                f"on {target['device_name']}"
            )

        result = run_single_test(
            test=test,
            client=local_client,
            model_config=model_config,
            base_agent_config=base_agent_config,
            global_max_steps=options.max_steps,
            device_type=device_type,
            verbose=options.verbose,
            device_name=target["device_name"] or "",
        )
        device_results.append(result)

        status = "PASS" if result.passed else "FAIL"
        if not options.verbose and not options.quiet:
            print(
                f"      {status} | Device: {target['device_name']} | "
                f"Steps: {result.steps_taken} | "
                f"Duration: {format_duration(result.duration_seconds)}"
            )

    return device_results


def build_base_agent_config_from_options(
    options: TestRunOptions, device_type: DeviceType, device_id: str | None
) -> AgentConfig | IOSAgentConfig:
    """Build base agent config for one target device from immutable options."""
    if device_type == DeviceType.IOS:
        return IOSAgentConfig(
            max_steps=options.max_steps,
            wda_url=options.wda_url,
            device_id=device_id,
            lang=options.lang,
            verbose=False,
        )

    return AgentConfig(
        max_steps=options.max_steps,
        device_id=device_id,
        lang=options.lang,
        verbose=False,
    )


def keep_device_awake(device_type: DeviceType, device_id: str | None) -> DeviceWakeState:
    """Keep an Android device screen awake during tests."""
    state = DeviceWakeState(device_id=device_id)
    if device_type != DeviceType.ADB:
        return state

    adb_prefix = ["adb", "-s", device_id] if device_id else ["adb"]
    state.screen_off_timeout = _adb_get_setting(
        adb_prefix, "system", "screen_off_timeout"
    )
    state.stay_on_while_plugged_in = _adb_get_setting(
        adb_prefix, "global", "stay_on_while_plugged_in"
    )

    subprocess.run(
        adb_prefix + ["shell", "settings", "put", "system", "screen_off_timeout", "86400000"],
        capture_output=True,
        text=True,
    )
    subprocess.run(
        adb_prefix + ["shell", "settings", "put", "global", "stay_on_while_plugged_in", "7"],
        capture_output=True,
        text=True,
    )
    wake_device_to_home(adb_prefix)
    return state


def wake_device_to_home(adb_prefix: list[str]) -> None:
    """Wake a device and make a best-effort attempt to reach the home screen."""
    subprocess.run(
        adb_prefix + ["shell", "input", "keyevent", "KEYCODE_WAKEUP"],
        capture_output=True,
        text=True,
    )
    time.sleep(0.5)
    subprocess.run(
        adb_prefix + ["shell", "wm", "dismiss-keyguard"],
        capture_output=True,
        text=True,
    )
    subprocess.run(
        adb_prefix + ["shell", "input", "keyevent", "KEYCODE_MENU"],
        capture_output=True,
        text=True,
    )
    subprocess.run(
        adb_prefix + ["shell", "input", "swipe", "500", "900", "500", "200", "300"],
        capture_output=True,
        text=True,
    )
    time.sleep(0.5)
    subprocess.run(
        adb_prefix + ["shell", "input", "keyevent", "KEYCODE_HOME"],
        capture_output=True,
        text=True,
    )
    time.sleep(0.5)


def restore_device_awake_state(device_type: DeviceType, state: DeviceWakeState) -> None:
    """Restore Android display settings changed for tests."""
    if device_type != DeviceType.ADB:
        return

    adb_prefix = ["adb", "-s", state.device_id] if state.device_id else ["adb"]
    _adb_restore_setting(
        adb_prefix, "system", "screen_off_timeout", state.screen_off_timeout
    )
    _adb_restore_setting(
        adb_prefix,
        "global",
        "stay_on_while_plugged_in",
        state.stay_on_while_plugged_in,
    )


def _adb_get_setting(adb_prefix: list[str], namespace: str, key: str) -> str | None:
    result = subprocess.run(
        adb_prefix + ["shell", "settings", "get", namespace, key],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return None if value == "null" else value


def _adb_restore_setting(
    adb_prefix: list[str], namespace: str, key: str, value: str | None
) -> None:
    if value is None:
        subprocess.run(
            adb_prefix + ["shell", "settings", "delete", namespace, key],
            capture_output=True,
            text=True,
        )
        return

    subprocess.run(
        adb_prefix + ["shell", "settings", "put", namespace, key, value],
        capture_output=True,
        text=True,
    )


# ── CLI entry point ────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phone Agent Test Harness - Run natural language test cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example test case format (.md file):
    ---
    task: "打开微信，搜索'文件传输助手'"
    expected: "搜索结果页面应显示文件传输助手联系人"
    max_steps: 10
    lang: cn
    ---

    # Test name (from first heading or filename)

    ## 任务
    Open WeChat and search for "filehelper"

    ## 预期结果
    Search results show the filehelper contact
        """,
    )

    parser.add_argument(
        "directory",
        type=str,
        help="Path to directory containing .md test case files",
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("PHONE_AGENT_BASE_URL", "http://10.33.10.102:8101/v1"),
        help="Model API base URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("PHONE_AGENT_MODEL", "autoglm-phone"),
        help="Model name",
    )
    parser.add_argument(
        "--apikey",
        type=str,
        default=os.getenv("PHONE_AGENT_API_KEY", "EMPTY"),
        help="API key for model authentication",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(os.getenv("PHONE_AGENT_MAX_STEPS", "100")),
        help="Default max steps per test (overridable per test case via frontmatter)",
    )
    parser.add_argument(
        "--device-id",
        "-d",
        type=str,
        default=os.getenv("PHONE_AGENT_DEVICE_ID"),
        help="Device ID",
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=["cn", "en"],
        default=os.getenv("PHONE_AGENT_LANG", "cn"),
        help="Language for prompts (default: cn)",
    )
    parser.add_argument(
        "--device-type",
        type=str,
        choices=["adb", "hdc", "ios"],
        default=os.getenv("PHONE_AGENT_DEVICE_TYPE", "adb"),
        help="Device type (default: adb)",
    )
    parser.add_argument(
        "--wda-url",
        type=str,
        default=os.getenv("PHONE_AGENT_WDA_URL", "http://localhost:8100"),
        help="WebDriverAgent URL for iOS",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed step-by-step output"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Only print final results table"
    )
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="Run tests on multiple connected devices in parallel",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Path to write structured test results as JSON",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate directory
    test_dir = Path(args.directory)
    if not test_dir.is_dir():
        print(f"Error: '{args.directory}' is not a valid directory")
        sys.exit(1)

    # Discover test files
    test_files = sorted(test_dir.glob("*.md"))
    if not test_files:
        print(f"No .md files found in '{args.directory}'")
        sys.exit(1)

    # Parse test cases
    tests: list[TestCase] = []
    parse_errors: list[str] = []
    for fp in test_files:
        try:
            tests.append(parse_test_file(fp))
        except (ValueError, yaml.YAMLError) as e:
            parse_errors.append(str(e))

    if parse_errors:
        print("Parse errors in test files:")
        for err in parse_errors:
            print(f"  - {err}")
        if not tests:
            sys.exit(1)
        print()

    if not args.quiet:
        print(f"Found {len(tests)} test case(s) in '{args.directory}':")
        for t in tests:
            print(f"  - {t.name} (max_steps={t.max_steps or args.max_steps})")

    # Determine device type
    device_type_map = {"adb": DeviceType.ADB, "hdc": DeviceType.HDC, "ios": DeviceType.IOS}
    device_type = device_type_map[args.device_type]

    if device_type != DeviceType.IOS:
        set_device_type(device_type)

    target_devices = discover_target_devices(device_type, args.device_id)
    run_offsets = {
        target["device_id"]: index * len(tests)
        for index, target in enumerate(target_devices)
    }
    run_options = TestRunOptions(
        base_url=args.base_url,
        api_key=args.apikey,
        model_name=args.model,
        lang=args.lang,
        max_steps=args.max_steps,
        wda_url=args.wda_url,
        verbose=args.verbose,
        quiet=args.quiet,
        total_runs=len(target_devices) * len(tests),
        run_offsets=run_offsets,
    )

    if not args.quiet:
        print("Target device(s):")
        for target in target_devices:
            print(f"  - {target['device_name']}")
        if args.concurrent and len(target_devices) > 1:
            print("Running devices concurrently.")

    # Run tests
    results: list[TestResult] = []
    wake_states = [
        keep_device_awake(device_type, target["device_id"])
        for target in target_devices
    ]
    try:
        if args.concurrent and len(target_devices) > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(target_devices)
            ) as executor:
                future_to_target = {
                    executor.submit(
                        run_tests_for_device,
                        target,
                        tests,
                        device_type,
                        run_options,
                    ): target
                    for target in target_devices
                }
                ordered_results: dict[str | None, list[TestResult]] = {}
                for future in concurrent.futures.as_completed(future_to_target):
                    target = future_to_target[future]
                    ordered_results[target["device_id"]] = future.result()

                for target in target_devices:
                    results.extend(ordered_results.get(target["device_id"], []))
        else:
            for target in target_devices:
                results.extend(
                    run_tests_for_device(
                        target=target,
                        tests=tests,
                        device_type=device_type,
                        options=run_options,
                    )
                )
    finally:
        for wake_state in reversed(wake_states):
            restore_device_awake_state(device_type, wake_state)

    # Print final report
    print_results(results)
    if args.json_output:
        write_json_results(results, args.json_output)
        if not args.quiet:
            print(f"JSON results written to: {args.json_output}")

    # Exit code: non-zero if any test failed
    if any(not r.passed for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
