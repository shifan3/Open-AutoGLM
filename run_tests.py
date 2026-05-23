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
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

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
    body_lines = parts[2].strip().split("\n")
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
) -> tuple[bool, str]:
    """
    Verify that the final screen matches the expected outcome by asking the VLM.

    Returns (passed, reason).
    """
    prompt = (
        f"请判断当前屏幕截图展示的结果是否符合以下预期：\n\n"
        f"预期：{expected}\n\n"
        f"当前屏幕信息：{screen_info}\n\n"
        f"请只回答 YES 或 NO，并给出简短理由。"
        if lang == "cn"
        else (
            f"Based on the screenshot, does the result match the expected outcome?\n\n"
            f"Expected: {expected}\n\n"
            f"Screen info: {screen_info}\n\n"
            f"Answer only YES or NO with a brief reason."
        )
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]

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


# ── Test runner ─────────────────────────────────────────────────────────────


def run_single_test(
    test: TestCase,
    client: OpenAI,
    model_config: ModelConfig,
    base_agent_config: AgentConfig | IOSAgentConfig,
    global_max_steps: int,
    device_type: DeviceType,
    verbose: bool,
) -> TestResult:
    """Run a single test case and return the result."""
    start_time = time.time()

    try:
        # Determine max steps: test-level override > global config
        max_steps = test.max_steps or global_max_steps

        if device_type == DeviceType.IOS:
            agent_config = IOSAgentConfig(
                max_steps=max_steps,
                wda_url=base_agent_config.wda_url,
                device_id=base_agent_config.device_id,
                lang=test.lang,
                verbose=verbose,
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
            )
            agent = PhoneAgent(
                model_config=model_config,
                agent_config=agent_config,
            )

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  Running: {test.name}")
            print(f"  Task: {test.task}")
            print(f"  Expected: {test.expected}")
            print(f"{'=' * 60}")

        # Phase 1: Execute the task
        agent_message = agent.run(test.task)
        steps_taken = agent.step_count

        # Phase 2: Verify result with a fresh screenshot
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

        passed, reason = verify_result(
            client=client,
            model_name=model_config.model_name,
            screenshot_b64=screenshot.base64_data,
            screen_info=screen_info,
            expected=test.expected,
            lang=test.lang,
        )

        elapsed = time.time() - start_time

        return TestResult(
            test=test,
            passed=passed,
            steps_taken=steps_taken,
            duration_seconds=elapsed,
            agent_message=agent_message,
            verification_reason=reason,
        )

    except Exception as e:
        elapsed = time.time() - start_time
        return TestResult(
            test=test,
            passed=False,
            steps_taken=0,
            duration_seconds=elapsed,
            error=f"{type(e).__name__}: {e}",
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
        icon = "✓" if result.passed else "✗"
        duration = format_duration(result.duration_seconds)

        print(f"\n  [{i}] {icon} {result.test.name}")

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
                print(f"    - {r.test.name}: {summary}")
    print("=" * 70 + "\n")


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

    # Build model config
    model_config = ModelConfig(
        base_url=args.base_url,
        model_name=args.model,
        api_key=args.apikey,
        lang=args.lang,
    )

    # Create OpenAI client for verification calls
    client = OpenAI(base_url=args.base_url, api_key=args.apikey, timeout=60.0)

    # Build base agent config (used as template, max_steps/lang overridden per test)
    if device_type == DeviceType.IOS:
        base_agent_config = IOSAgentConfig(
            max_steps=args.max_steps,
            wda_url=args.wda_url,
            device_id=args.device_id,
            lang=args.lang,
            verbose=False,
        )
    else:
        base_agent_config = AgentConfig(
            max_steps=args.max_steps,
            device_id=args.device_id,
            lang=args.lang,
            verbose=False,
        )

    # Run tests
    results: list[TestResult] = []
    for i, test in enumerate(tests, 1):
        if not args.quiet:
            print(f"\n[{i}/{len(tests)}] Running: {test.name}")

        result = run_single_test(
            test=test,
            client=client,
            model_config=model_config,
            base_agent_config=base_agent_config,
            global_max_steps=args.max_steps,
            device_type=device_type,
            verbose=args.verbose,
        )
        results.append(result)

        # Quick inline status
        status = "PASS" if result.passed else "FAIL"
        if not args.verbose and not args.quiet:
            print(f"      {status} | Steps: {result.steps_taken} | "
                  f"Duration: {format_duration(result.duration_seconds)}")

    # Print final report
    print_results(results)

    # Exit code: non-zero if any test failed
    if any(not r.passed for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
