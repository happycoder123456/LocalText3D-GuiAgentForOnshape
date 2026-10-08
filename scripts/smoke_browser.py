"""Smoke test for the real Playwright browser driver (needs Playwright + a browser).

Launches Chromium (bundled, system Chrome, or Edge — whichever exists), opens
Onshape, takes a screenshot, executes a few actions, and checks the URL allowlist.
Run:  python scripts/smoke_browser.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.browser import BrowserDriver, BrowserError, is_allowed_url  # noqa: E402


def main() -> int:
    driver = BrowserDriver(headless=True, viewport=(1280, 800))
    print("launching browser (bundled -> chrome -> edge fallback)...")
    t0 = time.monotonic()
    driver.start()
    print(f"started in {time.monotonic() - t0:.1f}s, url={driver.url()}")

    # Navigation allowlist
    assert is_allowed_url(driver.url()) or "signin" in driver.url() or driver.url() == "about:blank", driver.url()
    try:
        driver.goto("https://evil.com/")
        raise AssertionError("evil URL was allowed")
    except BrowserError:
        print("allowlist: OK (evil URL refused)")

    # Screenshot
    png = driver.screenshot_png()
    assert png.startswith(b"\x89PNG") and len(png) > 1000, len(png)
    print(f"screenshot: {len(png)} bytes")

    # Action execution (harmless: click center of blank-ish page, then keys)
    r = driver.execute({"action": "click", "x": 640, "y": 400})
    assert r.get("done"), r
    r = driver.execute({"action": "key", "keys": ["escape"]})
    assert r.get("done"), r
    r = driver.execute({"action": "hotkey", "keys": ["ctrl", "l"]})
    assert r.get("done"), r
    r = driver.execute({"action": "type", "text": "test"})
    assert r.get("done"), r
    r = driver.execute({"action": "wait", "seconds": 0.2}, stop_check=lambda: False)
    assert r.get("done"), r
    print("actions: click/key/hotkey/type/wait OK")

    # Out-of-viewport coordinates are clamped, not raised
    r = driver.execute({"action": "click", "x": 99999, "y": -5})
    assert r.get("done"), r
    print("coord clamp: OK")

    # Input capture install (best-effort on the real page)
    try:
        driver.start_input_capture()
        captured = driver.take_captured_actions()
        assert isinstance(captured, list)
        print("input capture: installed")
    except BrowserError as exc:
        print(f"input capture: skipped ({exc})")

    driver.stop()
    assert not driver.available
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
