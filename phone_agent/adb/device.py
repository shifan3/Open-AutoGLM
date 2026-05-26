"""Device control utilities for Android automation."""

import os
import re
import subprocess
import time
from typing import List, Optional, Tuple

from phone_agent.config.apps import APP_PACKAGES
from phone_agent.config.timing import TIMING_CONFIG


def get_current_app(device_id: str | None = None) -> str:
    """
    Get the currently focused app name.

    Args:
        device_id: Optional ADB device ID for multi-device setups.

    Returns:
        The app name if recognized, otherwise "System Home".
    """
    adb_prefix = _get_adb_prefix(device_id)

    result = subprocess.run(
        adb_prefix + ["shell", "dumpsys", "window"], capture_output=True, text=True, encoding="utf-8"
    )
    output = result.stdout
    if not output:
        raise ValueError("No output from dumpsys window")

    # Parse window focus info
    for line in output.split("\n"):
        if "mCurrentFocus" in line or "mFocusedApp" in line:
            for app_name, package in APP_PACKAGES.items():
                if package in line:
                    return app_name

    return "System Home"


def _get_focused_package(device_id: str | None = None) -> str | None:
    """Return the foreground package name when it can be parsed from dumpsys."""
    adb_prefix = _get_adb_prefix(device_id)

    result = subprocess.run(
        adb_prefix + ["shell", "dumpsys", "window"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    for line in result.stdout.split("\n"):
        if "mCurrentFocus" in line or "mFocusedApp" in line:
            match = re.search(r"\s([a-zA-Z0-9_.]+)/(?:[a-zA-Z0-9_.$]+)", line)
            if match:
                return match.group(1)
            match = re.search(r"\{[^}]*\s([a-zA-Z0-9_.]+)/", line)
            if match:
                return match.group(1)

    return None


def tap(
    x: int, y: int, device_id: str | None = None, delay: float | None = None
) -> None:
    """
    Tap at the specified coordinates.

    Args:
        x: X coordinate.
        y: Y coordinate.
        device_id: Optional ADB device ID.
        delay: Delay in seconds after tap. If None, uses configured default.
    """
    if delay is None:
        delay = TIMING_CONFIG.device.default_tap_delay

    adb_prefix = _get_adb_prefix(device_id)

    subprocess.run(
        adb_prefix + ["shell", "input", "tap", str(x), str(y)], capture_output=True
    )
    time.sleep(delay)


def double_tap(
    x: int, y: int, device_id: str | None = None, delay: float | None = None
) -> None:
    """
    Double tap at the specified coordinates.

    Args:
        x: X coordinate.
        y: Y coordinate.
        device_id: Optional ADB device ID.
        delay: Delay in seconds after double tap. If None, uses configured default.
    """
    if delay is None:
        delay = TIMING_CONFIG.device.default_double_tap_delay

    adb_prefix = _get_adb_prefix(device_id)

    subprocess.run(
        adb_prefix + ["shell", "input", "tap", str(x), str(y)], capture_output=True
    )
    time.sleep(TIMING_CONFIG.device.double_tap_interval)
    subprocess.run(
        adb_prefix + ["shell", "input", "tap", str(x), str(y)], capture_output=True
    )
    time.sleep(delay)


def long_press(
    x: int,
    y: int,
    duration_ms: int = 3000,
    device_id: str | None = None,
    delay: float | None = None,
) -> None:
    """
    Long press at the specified coordinates.

    Args:
        x: X coordinate.
        y: Y coordinate.
        duration_ms: Duration of press in milliseconds.
        device_id: Optional ADB device ID.
        delay: Delay in seconds after long press. If None, uses configured default.
    """
    if delay is None:
        delay = TIMING_CONFIG.device.default_long_press_delay

    adb_prefix = _get_adb_prefix(device_id)

    subprocess.run(
        adb_prefix
        + ["shell", "input", "swipe", str(x), str(y), str(x), str(y), str(duration_ms)],
        capture_output=True,
    )
    time.sleep(delay)


def swipe(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    duration_ms: int | None = None,
    device_id: str | None = None,
    delay: float | None = None,
) -> None:
    """
    Swipe from start to end coordinates.

    Args:
        start_x: Starting X coordinate.
        start_y: Starting Y coordinate.
        end_x: Ending X coordinate.
        end_y: Ending Y coordinate.
        duration_ms: Duration of swipe in milliseconds (auto-calculated if None).
        device_id: Optional ADB device ID.
        delay: Delay in seconds after swipe. If None, uses configured default.
    """
    if delay is None:
        delay = TIMING_CONFIG.device.default_swipe_delay

    adb_prefix = _get_adb_prefix(device_id)

    if duration_ms is None:
        # Calculate duration based on distance
        dist_sq = (start_x - end_x) ** 2 + (start_y - end_y) ** 2
        duration_ms = int(dist_sq / 1000)
        duration_ms = max(1000, min(duration_ms, 2000))  # Clamp between 1000-2000ms

    subprocess.run(
        adb_prefix
        + [
            "shell",
            "input",
            "swipe",
            str(start_x),
            str(start_y),
            str(end_x),
            str(end_y),
            str(duration_ms),
        ],
        capture_output=True,
    )
    time.sleep(delay)


def back(device_id: str | None = None, delay: float | None = None) -> None:
    """
    Press the back button.

    Args:
        device_id: Optional ADB device ID.
        delay: Delay in seconds after pressing back. If None, uses configured default.
    """
    if delay is None:
        delay = TIMING_CONFIG.device.default_back_delay

    adb_prefix = _get_adb_prefix(device_id)

    subprocess.run(
        adb_prefix + ["shell", "input", "keyevent", "4"], capture_output=True
    )
    time.sleep(delay)


def home(device_id: str | None = None, delay: float | None = None) -> None:
    """
    Press the home button.

    Args:
        device_id: Optional ADB device ID.
        delay: Delay in seconds after pressing home. If None, uses configured default.
    """
    if delay is None:
        delay = TIMING_CONFIG.device.default_home_delay

    adb_prefix = _get_adb_prefix(device_id)

    subprocess.run(
        adb_prefix + ["shell", "input", "keyevent", "KEYCODE_HOME"], capture_output=True
    )
    time.sleep(delay)


def launch_app(
    app_name: str, device_id: str | None = None, delay: float | None = None
) -> bool:
    """
    Launch an app by name.

    Args:
        app_name: The app name (must be in APP_PACKAGES).
        device_id: Optional ADB device ID.
        delay: Delay in seconds after launching. If None, uses configured default.

    Returns:
        True if app was launched, False if app not found.
    """
    if delay is None:
        delay = TIMING_CONFIG.device.default_launch_delay

    package = _resolve_package_name(app_name)
    if not package:
        return False

    adb_prefix = _get_adb_prefix(device_id)

    result = subprocess.run(
        adb_prefix
        + [
            "shell",
            "monkey",
            "-p",
            package,
            "-c",
            "android.intent.category.LAUNCHER",
            "1",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    time.sleep(delay)
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0 and "No activities found" not in output


def kill_app(
    app_name: str, device_id: str | None = None, delay: float | None = None
) -> bool:
    """
    Force stop an app by name.

    Args:
        app_name: The app name (must be in APP_PACKAGES).
        device_id: Optional ADB device ID.
        delay: Delay in seconds after stopping. If None, uses configured default.

    Returns:
        True if the stop command was sent, False if app not found.
    """
    if delay is None:
        delay = TIMING_CONFIG.device.default_home_delay

    package = _resolve_package_name(app_name)
    if not package:
        focused_package = _get_focused_package(device_id)
        if focused_package and not _is_system_or_launcher_package(focused_package):
            package = focused_package
        else:
            return False

    adb_prefix = _get_adb_prefix(device_id)

    result = subprocess.run(
        adb_prefix + ["shell", "am", "force-stop", package],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        return False

    subprocess.run(
        adb_prefix + ["shell", "input", "keyevent", "KEYCODE_HOME"],
        capture_output=True,
    )
    time.sleep(delay)
    return True


def _resolve_package_name(app_name: str | None) -> str | None:
    """Resolve a configured app name or raw Android package name."""
    if not app_name:
        return None

    app_name = app_name.strip()
    if _looks_like_package_name(app_name):
        return app_name

    if app_name in APP_PACKAGES:
        return APP_PACKAGES[app_name]

    normalized = _normalize_app_name(app_name)
    for known_name, package in APP_PACKAGES.items():
        if _normalize_app_name(known_name) == normalized:
            return package

    return None


def _looks_like_package_name(value: str) -> bool:
    return bool(re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z0-9_]+)+", value))


def _normalize_app_name(value: str) -> str:
    return re.sub(r"[\s_\-]+", "", value).casefold()


def _is_system_or_launcher_package(package: str) -> bool:
    launcher_packages = {
        "com.miui.home",
        "com.huawei.android.launcher",
        "com.google.android.apps.nexuslauncher",
        "com.sec.android.app.launcher",
        "com.oppo.launcher",
        "com.vivo.launcher",
        "com.android.launcher",
        "com.android.launcher3",
    }
    return package.startswith("com.android.") or package in launcher_packages


def _get_adb_prefix(device_id: str | None) -> list:
    """Get ADB command prefix with optional device specifier."""
    if device_id:
        return ["adb", "-s", device_id]
    return ["adb"]
