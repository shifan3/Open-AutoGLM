"""Helpers for carrying structured test progress through agent prompts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TestExecutionState:
    """Mutable state shared across model calls while running one test."""

    steps: list[str] = field(default_factory=list)
    current_step: int = 1
    completed_steps: set[int] = field(default_factory=set)
    feedback: str = ""

    def format_progress(self) -> str:
        """Return a human-readable numbered progress list for the prompt."""
        if not self.steps:
            return ""

        current_step = self.current_step
        if current_step < 1:
            current_step = 1
        if current_step > len(self.steps):
            current_step = len(self.steps)

        lines = []
        for index, step in enumerate(self.steps, 1):
            suffix = ""
            if index in self.completed_steps:
                suffix = " [done]"
            elif index == current_step:
                suffix = " <== here"
            lines.append(f"{index}. {step}{suffix}")
        return "\n".join(lines)

    def format_prompt_context(self, lang: str) -> str:
        """Build the test-state section injected into each model call."""
        if not self.steps:
            return ""

        progress = self.format_progress()
        feedback = self.feedback.strip() or ("无" if lang == "cn" else "None")

        if lang == "cn":
            return (
                "测试执行状态:\n"
                f"{progress}\n\n"
                f"上一次模型反馈/有助于后续测试的信息: {feedback}\n\n"
                "每次回复都必须在动作之前输出测试状态，格式如下:\n"
                '<test_status>{"current_step": 1, "completed_steps": [], '
                '"feedback": "当前测试到了哪一步、观察到什么、有助于后续测试的信息"}</test_status>\n'
                "current_step 使用 1-based 编号，completed_steps 是已经确认完成的步骤编号数组。"
            )

        return (
            "Test execution state:\n"
            f"{progress}\n\n"
            f"Previous model feedback / useful context for later steps: {feedback}\n\n"
            "Every response must include test state before the action using this format:\n"
            '<test_status>{"current_step": 1, "completed_steps": [], '
            '"feedback": "where the test is now, observations, useful context"}</test_status>\n'
            "current_step is 1-based, and completed_steps is an array of confirmed completed step numbers."
        )

    def apply_status_text(self, status_text: str | None) -> None:
        """Update state from the model's <test_status> JSON payload."""
        if not status_text:
            return

        payload = _parse_status_payload(status_text)
        if not payload:
            self.feedback = status_text.strip()
            return

        previous_step = self.current_step
        current_step = payload.get("current_step")
        if isinstance(current_step, int):
            self.current_step = _clamp(current_step, 1, max(len(self.steps), 1))
            if self.current_step > previous_step:
                self.completed_steps.update(range(previous_step, self.current_step))

        completed_steps = payload.get("completed_steps")
        if isinstance(completed_steps, list):
            parsed_completed = {
                step
                for step in completed_steps
                if isinstance(step, int) and 1 <= step <= len(self.steps)
            }
            self.completed_steps.update(parsed_completed)

        feedback = payload.get("feedback")
        if isinstance(feedback, str) and feedback.strip():
            self.feedback = feedback.strip()


def extract_test_status(content: str) -> str | None:
    """Extract the first <test_status>...</test_status> block."""
    match = re.search(r"<test_status>(.*?)</test_status>", content, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def _parse_status_payload(status_text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(status_text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))
