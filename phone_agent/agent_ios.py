"""iOS PhoneAgent class for orchestrating iOS phone automation."""

import json
import re
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from phone_agent.actions.handler import do, finish, parse_action
from phone_agent.actions.handler_ios import IOSActionHandler
from phone_agent.config import get_messages, get_system_prompt
from phone_agent.model import ModelClient, ModelConfig
from phone_agent.model.client import MessageBuilder
from phone_agent.test_state import TestExecutionState
from phone_agent.xctest import XCTestConnection, get_current_app, get_screenshot


@dataclass
class IOSAgentConfig:
    """Configuration for the iOS PhoneAgent."""

    max_steps: int = 100
    wda_url: str = "http://localhost:8100"
    session_id: str | None = None
    device_id: str | None = None  # iOS device UDID
    lang: str = "cn"
    system_prompt: str | None = None
    verbose: bool = True
    test_steps: list[str] = field(default_factory=list)
    artifact_steps: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.system_prompt is None:
            self.system_prompt = get_system_prompt(self.lang)


@dataclass
class StepResult:
    """Result of a single agent step."""

    success: bool
    finished: bool
    action: dict[str, Any] | None
    thinking: str
    message: str | None = None


class IOSPhoneAgent:
    """
    AI-powered agent for automating iOS phone interactions.

    The agent uses a vision-language model to understand screen content
    and decide on actions to complete user tasks via WebDriverAgent.

    Args:
        model_config: Configuration for the AI model.
        agent_config: Configuration for the iOS agent behavior.
        confirmation_callback: Optional callback for sensitive action confirmation.
        takeover_callback: Optional callback for takeover requests.

    Example:
        >>> from phone_agent.agent_ios import IOSPhoneAgent, IOSAgentConfig
        >>> from phone_agent.model import ModelConfig
        >>>
        >>> model_config = ModelConfig(base_url="http://localhost:8000/v1")
        >>> agent_config = IOSAgentConfig(wda_url="http://localhost:8100")
        >>> agent = IOSPhoneAgent(model_config, agent_config)
        >>> agent.run("Open Safari and search for Apple")
    """

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        agent_config: IOSAgentConfig | None = None,
        confirmation_callback: Callable[[str], bool] | None = None,
        takeover_callback: Callable[[str], None] | None = None,
    ):
        self.model_config = model_config or ModelConfig()
        self.agent_config = agent_config or IOSAgentConfig()
        self.model_config.verbose = self.agent_config.verbose

        self.model_client = ModelClient(self.model_config)

        # Initialize WDA connection and create session if needed
        self.wda_connection = XCTestConnection(wda_url=self.agent_config.wda_url)

        # Auto-create session if not provided
        if self.agent_config.session_id is None:
            success, session_id = self.wda_connection.start_wda_session()
            if success and session_id != "session_started":
                self.agent_config.session_id = session_id
                if self.agent_config.verbose:
                    print(f"✅ Created WDA session: {session_id}")
            elif self.agent_config.verbose:
                print(f"⚠️  Using default WDA session (no explicit session ID)")

        self.action_handler = IOSActionHandler(
            wda_url=self.agent_config.wda_url,
            session_id=self.agent_config.session_id,
            verbose=self.agent_config.verbose,
            confirmation_callback=confirmation_callback,
            takeover_callback=takeover_callback,
        )

        self._context: list[dict[str, Any]] = []
        self._step_count = 0
        self._task: str = ""
        self._test_state = TestExecutionState(self.agent_config.test_steps)
        self._last_screenshot = None
        self._last_current_app = ""
        self._saved_artifacts: dict[str, Any] = {}
        self._last_action_feedback = ""

    def run(self, task: str) -> str:
        """
        Run the agent to complete a task.

        Args:
            task: Natural language description of the task.

        Returns:
            Final message from the agent.
        """
        self._context = []
        self._step_count = 0
        self._task = task
        self._test_state = TestExecutionState(self.agent_config.test_steps)
        self._saved_artifacts = {}
        self._last_action_feedback = ""

        # First step with user prompt
        result = self._execute_step(task, is_first=True)

        if result.finished:
            return result.message or "Task completed"

        # Continue until finished or max steps reached
        while self._step_count < self.agent_config.max_steps:
            result = self._execute_step(is_first=False)

            if result.finished:
                return result.message or "Task completed"

        return "Max steps reached"

    def step(self, task: str | None = None) -> StepResult:
        """
        Execute a single step of the agent.

        Useful for manual control or debugging.

        Args:
            task: Task description (only needed for first step).

        Returns:
            StepResult with step details.
        """
        is_first = len(self._context) == 0

        if is_first and not task:
            raise ValueError("Task is required for the first step")

        return self._execute_step(task, is_first)

    def reset(self) -> None:
        """Reset the agent state for a new task."""
        self._context = []
        self._step_count = 0
        self._test_state = TestExecutionState(self.agent_config.test_steps)
        self._saved_artifacts = {}
        self._last_action_feedback = ""

    def _execute_step(
        self, user_prompt: str | None = None, is_first: bool = False
    ) -> StepResult:
        """Execute a single step of the agent loop."""
        self._step_count += 1

        # Capture current screen state
        screenshot = get_screenshot(
            wda_url=self.agent_config.wda_url,
            session_id=self.agent_config.session_id,
            device_id=self.agent_config.device_id,
        )
        current_app = get_current_app(
            wda_url=self.agent_config.wda_url, session_id=self.agent_config.session_id
        )
        self._last_screenshot = screenshot
        self._last_current_app = current_app

        # Build messages
        if is_first:
            self._context.append(
                MessageBuilder.create_system_message(self.agent_config.system_prompt)
            )

            screen_info = MessageBuilder.build_screen_info(current_app)
            text_content = self._build_user_prompt(
                task_text=user_prompt or "",
                screen_info=screen_info,
                is_first=True,
            )

            self._context.append(
                MessageBuilder.create_user_message(
                    text=text_content, image_base64=screenshot.base64_data
                )
            )
        else:
            screen_info = MessageBuilder.build_screen_info(current_app)
            text_content = self._build_user_prompt(
                task_text=self._task,
                screen_info=screen_info,
                is_first=False,
            )

            self._context.append(
                MessageBuilder.create_user_message(
                    text=text_content, image_base64=screenshot.base64_data
                )
            )

        # Get model response (with retry for transient failures)
        max_retries = 3
        retry_delay = 2.0
        response = None
        last_error = None

        for attempt in range(max_retries):
            try:
                response = self.model_client.request(self._context)
                break
            except Exception as e:
                last_error = e
                if self.agent_config.verbose:
                    traceback.print_exc()
                if attempt < max_retries - 1:
                    print(f"[WARN] Model error (step {self._step_count}, attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"[WARN] Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue
                print(f"[ERROR] Model failed after {max_retries} attempts: {e}")

        if response is None:
            return StepResult(
                success=False,
                finished=False,
                action=None,
                thinking="",
                message=f"Model error after {max_retries} retries: {last_error}",
            )

        self._test_state.apply_status_text(response.test_status)
        self._save_step_artifacts(screenshot)

        # Parse action from response
        try:
            action = parse_action(response.action, verbose=self.agent_config.verbose)
        except ValueError:
            if self.agent_config.verbose:
                traceback.print_exc()
            return StepResult(
                success=False,
                finished=False,
                action=None,
                thinking=response.thinking,
                message=f"Parse error: {response.action}",
            )

        # Guard: if model calls finish() but thinking indicates task is still in progress,
        # override with a Wait action instead of ending the task prematurely.
        if action.get("_metadata") == "finish":
            progress_keywords = [
                "等待", "加载中", "处理中", "上传中", "转换中", "下载中", "生成中",
                "提交中", "会自动完成", "需要进一步", "正在", "进度",
                "waiting", "loading", "processing", "uploading", "converting",
                "downloading", "generating", "in progress", "should wait",
            ]
            thinking_lower = response.thinking.lower()
            if any(kw.lower() in thinking_lower for kw in progress_keywords):
                print(
                    "[WARN] Model called finish() but thinking indicates task still in progress. "
                    "Overriding with Wait(3s)."
                )
                action = {"_metadata": "do", "action": "Wait", "duration": "3 seconds"}
                response.action = 'do(action="Wait", duration="3 seconds")'

        if self.agent_config.verbose:
            # Print thinking process
            msgs = get_messages(self.agent_config.lang)
            print("\n" + "=" * 50)
            print(f"[THINK] {msgs['thinking']}:")
            print("-" * 50)
            print(response.thinking)
            print("-" * 50)
            print(f"[ACTION] {msgs['action']}:")
            print(json.dumps(action, ensure_ascii=False, indent=2))
            print("=" * 50 + "\n")

        # Remove image from context to save space
        self._context[-1] = MessageBuilder.remove_images_from_message(self._context[-1])

        # Execute action
        try:
            result = self.action_handler.execute(
                action, screenshot.width, screenshot.height
            )
        except Exception as e:
            if self.agent_config.verbose:
                traceback.print_exc()
            result = self.action_handler.execute(
                finish(message=str(e)), screenshot.width, screenshot.height
            )

        if not result.should_finish and action.get("_metadata") != "finish":
            time.sleep(3)

        if result.message or not result.success:
            self._last_action_feedback = (
                f"Previous action result: success={result.success}, "
                f"message={result.message or ''}"
            )
        else:
            self._last_action_feedback = ""

        # Add assistant response to context
        self._context.append(
            MessageBuilder.create_assistant_message(
                f"<think>{response.thinking}</think><answer>{response.action}</answer>"
            )
        )

        # Check if finished
        finished = action.get("_metadata") == "finish" or result.should_finish

        if finished and self.agent_config.verbose:
            msgs = get_messages(self.agent_config.lang)
            print("\n" + "=" * 50)
            print(
                f"[DONE] {msgs['task_completed']}: {result.message or action.get('message', msgs['done'])}"
            )
            print("=" * 50 + "\n")

        return StepResult(
            success=result.success,
            finished=finished,
            action=action,
            thinking=response.thinking,
            message=result.message or action.get("message"),
        )

    def _build_user_prompt(
        self, task_text: str, screen_info: str, is_first: bool
    ) -> str:
        """Build the per-step user prompt with optional test progress."""
        test_context = self._test_state.format_prompt_context(self.agent_config.lang)

        if is_first:
            parts = [task_text]
        elif self.agent_config.lang == "cn":
            parts = [
                f"任务: {task_text}",
                f"当前步数: {self._step_count}/{self.agent_config.max_steps}",
            ]
        else:
            parts = [
                f"Task: {task_text}",
                f"Step: {self._step_count}/{self.agent_config.max_steps}",
            ]

        if test_context:
            parts.append(test_context)

        if self._last_action_feedback:
            parts.append(self._last_action_feedback)

        parts.append(f"** Screen Info **\n\n{screen_info}")
        return "\n\n".join(parts)

    def _save_step_artifacts(self, screenshot) -> None:
        """Save screenshots requested by save(@name) markers in test steps."""
        artifact_steps = self.agent_config.artifact_steps or self.agent_config.test_steps
        if not artifact_steps:
            return

        step_index = self._test_state.current_step - 1
        if step_index < 0 or step_index >= len(artifact_steps):
            return

        step_text = artifact_steps[step_index]
        for artifact_name in re.findall(r"save\(\s*@([^)]+)\s*\)", step_text):
            artifact_name = "@" + artifact_name.strip()
            if artifact_name not in self._saved_artifacts:
                self._saved_artifacts[artifact_name] = screenshot
                if self.agent_config.verbose:
                    print(f"[ARTIFACT] Saved {artifact_name} from step {self._test_state.current_step}")

    @property
    def context(self) -> list[dict[str, Any]]:
        """Get the current conversation context."""
        return self._context.copy()

    @property
    def step_count(self) -> int:
        """Get the current step count."""
        return self._step_count

    @property
    def last_screenshot(self):
        """Get the screenshot from the most recent model observation."""
        return self._last_screenshot

    @property
    def last_current_app(self) -> str:
        """Get the app name from the most recent model observation."""
        return self._last_current_app

    @property
    def saved_artifacts(self) -> dict[str, Any]:
        """Get screenshots captured by save(@name) test-step markers."""
        return self._saved_artifacts.copy()
