# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Phone Agent is an AI-powered phone automation framework built on AutoGLM. It uses a vision-language model (VLM) to understand phone screenshots and generate touch/input actions executed via ADB (Android), HDC (HarmonyOS), or WebDriverAgent (iOS). The VLM sees screen images, reasons about the next step, and outputs structured actions like `do(action="Tap", element=[x,y])` or `finish(message="done")`.

## Commands

**Install**: `pip install -r requirements.txt` (core deps: `Pillow`, `openai`). Optional dev: `pytest`, `black`, `mypy`, `ruff`.

**Run**: `python main.py` (interactive mode), `python main.py "Open WeChat and send a message"` (single task).

**Model serving** (SGLang via Docker): `bash start_serving.sh` (wrapper that sets env vars and calls `start_sglang.sh`). `start_sglang.sh` launches SGLang serving the AutoGLM-Phone-9B model in Docker containers with health checks.

**Pre-commit hooks**: `pre-commit install` — runs ruff (import sorting + formatting), typos (spell check), pymarkdown.

**Package install**: `pip install -e .` installs the `phone-agent` CLI entry point.

## Architecture

### Agent Loop (the core)

Both `PhoneAgent` (Android/HarmonyOS) and `IOSPhoneAgent` follow the same loop:

1. **Capture**: Take screenshot via device-specific module, detect current foreground app
2. **Send to VLM**: Build OpenAI-format multimodal message (system prompt + screenshot image + screen info JSON)
3. **Stream response**: Parse streaming output, splitting on `do(action=` / `finish(message=` markers to separate thinking from action. Legacy `\<think\>/\<answer\>` XML fallback.
4. **Execute action**: `ActionHandler` / `IOSActionHandler` dispatches to typed handlers (Tap, Type, Swipe, Launch, Back, Home, Long Press, Double Tap, Wait, Take_over, etc.)
5. **Repeat** until `finish` action or `max_steps` reached

Key files: [phone_agent/agent.py](phone_agent/agent.py), [phone_agent/agent_ios.py](phone_agent/agent_ios.py), [phone_agent/model/client.py](phone_agent/model/client.py), [phone_agent/actions/handler.py](phone_agent/actions/handler.py)

### Device Abstraction (Factory Pattern)

[phone_agent/device_factory.py](phone_agent/device_factory.py) provides a global `DeviceFactory` singleton that dispatches to one of three backend modules:

- **`phone_agent/adb/`** — Android via ADB: screenshots with `screencap`, input via `adb shell input`, ADB Keyboard IME for text entry
- **`phone_agent/hdc/`** — HarmonyOS via HDC: same interface, different shell commands
- **`phone_agent/xctest/`** — iOS via WebDriverAgent (XCTest): HTTP-based screenshot, tap, swipe through WDA REST API

Each module exposes the same interface: `get_screenshot`, `tap`, `swipe`, `type_text`, `back`, `home`, `launch_app`, `get_current_app`, etc. The factory is set globally at startup via `set_device_type()` and accessed everywhere via `get_device_factory()`.

### Action Parsing

[phone_agent/actions/handler.py](phone_agent/actions/handler.py) `parse_action()` uses Python's `ast.parse` for safe evaluation (not `eval`). Actions use a DSL syntax:
- `do(action="Tap", element=[x,y])` — coordinates in 0-1000 relative system
- `do(action="Type", text="...")` — text input (auto-switches to ADB Keyboard IME)
- `do(action="Swipe", start=[x1,y1], end=[x2,y2])`
- `do(action="Launch", app="AppName")`
- `finish(message="done")`

Coordinate conversion: `x_pixel = element[0] / 1000 * screen_width` (relative 0-1000 → absolute pixels).

### Configuration

- [phone_agent/config/prompts_zh.py](phone_agent/config/prompts_zh.py) / `prompts_en.py` — System prompts defining the action DSL and rules for the VLM. Date-injected at import time.
- [phone_agent/config/apps.py](phone_agent/config/apps.py) — `APP_PACKAGES` dict mapping Chinese app names → Android package names (e.g., `"微信" → "com.tencent.mm"`). Excluded from pre-commit checks.
- [phone_agent/config/i18n.py](phone_agent/config/i18n.py) — Chinese/English UI strings (`MESSAGES_ZH`, `MESSAGES_EN`).
- [phone_agent/config/timing.py](phone_agent/config/timing.py) — All delays (tap, swipe, keyboard switch, etc.), overridable via `PHONE_AGENT_*` env vars.

### Key Environment Variables

`PHONE_AGENT_BASE_URL`, `PHONE_AGENT_MODEL`, `PHONE_AGENT_API_KEY`, `PHONE_AGENT_MAX_STEPS`, `PHONE_AGENT_DEVICE_ID`, `PHONE_AGENT_LANG` (cn/en), `PHONE_AGENT_DEVICE_TYPE` (adb/hdc/ios)

### Runtime Behavior

- Screenshots are base64-encoded PNGs sent inline to the VLM
- After each step, images are stripped from context (via `MessageBuilder.remove_images_from_message`) to save context space — only the assistant's text response (thinking + action) remains
- Text input uses ADB Keyboard IME: switches IME → types text → restores original IME
- Sensitive operations (Tap with `message=` field, payments, privacy) trigger a confirmation callback (console Y/N by default)
