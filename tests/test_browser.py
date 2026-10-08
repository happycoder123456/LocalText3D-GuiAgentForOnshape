from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent.browser import (
    BrowserError,
    BrowserDriver,
    BrowserUnavailable,
    FakeBrowser,
    is_allowed_url,
    needs_login,
    release_profile,
    _playwright_key,
    _profile_lock_error,
)


class UrlAllowlistTests(unittest.TestCase):
    def test_onshape_allowed(self):
        self.assertTrue(is_allowed_url("https://cad.onshape.com/documents"))
        self.assertTrue(is_allowed_url("https://www.onshape.com/signin"))
        self.assertTrue(is_allowed_url("https://cad.onshape.com/foo/bar?x=1"))

    def test_non_onshape_refused(self):
        self.assertFalse(is_allowed_url("https://evil.com/cad"))
        self.assertFalse(is_allowed_url("https://onshape.com.evil.com/"))
        self.assertFalse(is_allowed_url("https://notonshape.com/"))

    def test_non_http_refused(self):
        self.assertFalse(is_allowed_url("javascript:alert(1)"))
        self.assertFalse(is_allowed_url("file:///etc/passwd"))
        self.assertFalse(is_allowed_url("about:blank"))
        self.assertFalse(is_allowed_url(""))

    def test_extra_hosts_allowed(self):
        self.assertTrue(is_allowed_url("http://localhost:3000/", extra_hosts=("localhost",)))


class NeedsLoginTests(unittest.TestCase):
    def test_signin_pages_detected(self):
        self.assertTrue(needs_login("https://cad.onshape.com/signin", "Sign in - Onshape"))
        self.assertTrue(needs_login("https://cad.onshape.com/documents", "Sign in"))
        self.assertTrue(needs_login("https://cad.onshape.com/login", ""))
        self.assertTrue(needs_login("", "Sign in to Onshape"))

    def test_normal_pages_pass(self):
        self.assertFalse(needs_login("https://cad.onshape.com/documents", "ONSHAPE TEST 1 | Part Studio 1"))
        self.assertFalse(needs_login("https://cad.onshape.com/documents/abc/w", "Part Studio 1"))
        self.assertFalse(needs_login("", ""))


class ProfileLockTests(unittest.TestCase):
    def test_lock_error_classification(self):
        lock = (
            "BrowserType.launch_persistent_context: Opening in existing browser session. "
            "This usually means that the profile is already in use by another instance of Chromium."
        )
        self.assertTrue(_profile_lock_error(lock))
        self.assertTrue(_profile_lock_error("The process cannot access the file because it is being used by another process"))
        self.assertFalse(_profile_lock_error("Chromium distribution 'chrome' is not found at C:\\..."))
        self.assertFalse(_profile_lock_error(""))

    def test_release_profile_calls_powershell_and_parses_count(self):
        with mock.patch("agent.browser.subprocess.run") as run, mock.patch("agent.browser.time.sleep"):
            run.return_value = mock.Mock(stdout="2\n", returncode=0)
            n = release_profile(r"C:\prof\browser_profile")
        self.assertEqual(n, 2)
        argv = run.call_args[0][0]
        self.assertIn("powershell", argv[0])
        self.assertIn(r"C:\prof\browser_profile", argv[-1])
        self.assertIn("Stop-Process", argv[-1])

    def test_release_profile_swallows_failures(self):
        with mock.patch("agent.browser.subprocess.run", side_effect=OSError("no powershell")):
            self.assertEqual(release_profile(r"C:\prof"), 0)
        self.assertEqual(release_profile(""), 0)


class _FakePage:
    def __init__(self, events: list):
        self.url = "about:blank"
        self._events = events

    def goto(self, url, **kwargs):
        self._events.append("goto")
        self.url = url

    def wait_for_load_state(self, *args, **kwargs):
        pass

    def title(self):
        return "Owned by me | Documents"


class _FakeCtx:
    def __init__(self, events: list):
        self.pages = [_FakePage(events)]
        self._events = events
        self.added = []

    def add_cookies(self, data):
        self._events.append("cookies")
        self.added.extend(data)

    def cookies(self):
        return []

    def close(self):
        pass


class _FakeChromium:
    LOCK_ERR = (
        "launch_persistent_context: Opening in existing browser session. "
        "This usually means that the profile is already in use by another instance of Chromium."
    )

    def __init__(self, ctx: _FakeCtx, sequence: list[str]):
        self.calls = 0
        self._ctx = ctx
        self._sequence = list(sequence)

    def launch_persistent_context(self, profile, **kwargs):
        self.calls += 1
        step = self._sequence.pop(0) if self._sequence else "ok"
        if step == "lock":
            raise RuntimeError(self.LOCK_ERR)
        return self._ctx


class _FakePW:
    def __init__(self, chromium: _FakeChromium):
        self.chromium = chromium

    def start(self):
        return self

    def stop(self):
        pass


class StartProfileLockTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.profile = tmp.name
        self.events: list[str] = []
        self.ctx = _FakeCtx(self.events)
        self.patch_rel = mock.patch("agent.browser.release_profile", return_value=1)
        self.addCleanup(self.patch_rel.stop)
        self.release_profile = self.patch_rel.start()

    def _start(self, sequence: list[str]):
        chromium = _FakeChromium(self.ctx, sequence)
        pw = _FakePW(chromium)
        patch_pw = mock.patch("agent.browser._require_playwright", return_value=lambda: pw)
        self.addCleanup(patch_pw.stop)
        patch_pw.start()
        driver = BrowserDriver(profile_dir=self.profile)
        driver.start()
        return driver, chromium

    def test_start_retries_once_after_profile_lock(self):
        driver, chromium = self._start(["lock", "ok"])
        self.assertTrue(driver.available)
        self.assertEqual(chromium.calls, 2)
        # pre-launch sweep + the retry after the lock error
        self.assertEqual(self.release_profile.call_count, 2)
        driver.stop()

    def test_start_hint_names_the_locked_window_not_chromium(self):
        driver = BrowserDriver(profile_dir=self.profile)
        chromium = _FakeChromium(self.ctx, ["lock"] * 6)  # 3 channels x 2 attempts
        pw = _FakePW(chromium)
        with mock.patch("agent.browser._require_playwright", return_value=lambda: pw):
            with self.assertRaises(BrowserUnavailable) as cm:
                driver.start()
        msg = str(cm.exception)
        self.assertIn("agent browser window", msg)
        self.assertIn("already in use", msg)
        self.assertNotIn("No Chromium found", msg)

    def test_cookies_applied_before_first_navigation(self):
        cookies_file = Path(self.profile) / "saved_cookies.json"
        cookies_file.write_text(
            json.dumps([{"name": "on", "value": "x", "domain": ".onshape.com", "path": "/"}]),
            encoding="utf-8",
        )
        driver, chromium = self._start(["ok"])
        self.assertEqual(self.events, ["cookies", "goto"])  # auth must precede first navigation
        self.assertEqual(self.ctx.added[0]["name"], "on")
        driver.stop()


class DeadPageRecoveryTests(unittest.TestCase):
    """A user-closed window must not brick the driver forever.

    Regression: start() returned early whenever _page was set, even after the
    page was closed — every subsequent run then died with
    "Page.screenshot: Target page, context or browser has been closed".
    """

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.profile = tmp.name

    def test_available_reflects_a_closed_page(self):
        page = _FakePage([])
        driver = BrowserDriver(profile_dir=self.profile)
        driver._page = page
        page.is_closed = lambda: True
        self.assertFalse(driver.available)
        page.is_closed = lambda: False
        self.assertTrue(driver.available)
        driver._page = None
        self.assertFalse(driver.available)

    def test_start_relaunches_after_the_window_was_closed(self):
        events: list = []
        ctx = _FakeCtx(events)
        dead = ctx.pages[0]
        dead.is_closed = lambda: True  # the user closed the window
        driver = BrowserDriver(profile_dir=self.profile)
        driver._page = dead
        driver._ctx = ctx
        driver._pw = _FakePW(_FakeChromium(ctx, ["ok"]))

        class _RebornChromium(_FakeChromium):
            def launch_persistent_context(self, profile, **kwargs):
                out = super().launch_persistent_context(profile, **kwargs)
                out.pages = [_FakePage(self._ctx._events)]  # fresh, open page
                return out

        chromium = _RebornChromium(ctx, ["ok"])
        pw = _FakePW(chromium)
        with mock.patch("agent.browser.release_profile", return_value=0), mock.patch(
            "agent.browser._require_playwright", return_value=lambda: pw
        ):
            driver.start()
        self.assertEqual(chromium.calls, 1)  # a real relaunch happened
        self.assertTrue(driver.available)

    def test_start_is_a_noop_while_the_page_is_alive(self):
        page = _FakePage([])
        page.is_closed = lambda: False
        driver = BrowserDriver(profile_dir=self.profile)
        driver._page = page
        with mock.patch(
            "agent.browser._require_playwright",
            side_effect=AssertionError("healthy page must not trigger a relaunch"),
        ):
            driver.start()
        self.assertIs(driver._page, page)


class KeyFocusGuardTests(unittest.TestCase):
    """Letter hotkeys must not be swallowed by a focused filter box.

    Regression: 'r' pressed while the feature filter had focus typed "rr"
    into it, which hid every tree entry — including the plane the model was
    trying to click.
    """

    class _Page:
        def __init__(self, focused: bool):
            self.focused = focused
            self.events: list[str] = []
            outer = self

            class _KB:
                def press(self, key):
                    outer.events.append(f"press:{key}")

            self.keyboard = _KB()

        def evaluate(self, script):
            if ".blur(" in script:
                self.events.append("blur")
                return None
            self.events.append("focus-check")
            return "edit" if self.focused else ""

    def _driver(self, focused: bool) -> BrowserDriver:
        driver = BrowserDriver(viewport=(800, 600))
        driver._page = self._Page(focused)
        return driver

    def test_letter_key_blurs_focused_input_first(self):
        driver = self._driver(focused=True)
        out = driver.execute({"action": "key", "keys": ["r"]})
        self.assertTrue(out.get("done"))
        events = driver._page.events
        self.assertIn("blur", events)
        press_i = next(i for i, e in enumerate(events) if e.startswith("press:"))
        self.assertLess(events.index("blur"), press_i)

    def test_no_blur_when_nothing_focused(self):
        driver = self._driver(focused=False)
        driver.execute({"action": "key", "keys": ["r"]})
        self.assertNotIn("blur", driver._page.events)
        self.assertTrue(any(e.startswith("press:") for e in driver._page.events))

    def test_non_letter_keys_skip_the_blur(self):
        driver = self._driver(focused=True)
        driver.execute({"action": "key", "keys": ["esc"]})
        self.assertNotIn("blur", driver._page.events)
        self.assertTrue(any(e.startswith("press:") for e in driver._page.events))


class KeyMappingTests(unittest.TestCase):
    def test_aliases(self):
        self.assertEqual(_playwright_key("esc"), "Escape")
        self.assertEqual(_playwright_key("enter"), "Enter")
        self.assertEqual(_playwright_key("ctrl"), "Control")
        self.assertEqual(_playwright_key("up"), "ArrowUp")

    def test_single_chars_pass_through(self):
        self.assertEqual(_playwright_key("s"), "s")
        self.assertEqual(_playwright_key("z"), "z")

    def test_named_keys(self):
        self.assertEqual(_playwright_key("pageup"), "Pageup")
        self.assertEqual(_playwright_key("F5"), "F5")


class ExecuteKeySafetyTests(unittest.TestCase):
    def test_unknown_key_skipped_not_fatal(self):
        d = BrowserDriver()
        page = mock.MagicMock()
        page.keyboard.press.side_effect = Exception('Unknown key: "Rectangle"')
        d._page = page
        out = d.execute({"action": "key", "keys": ["Rectangle"]})
        self.assertFalse(out["done"])
        self.assertIn("unknown key", out["error"])

    def test_valid_key_pressed(self):
        d = BrowserDriver()
        d._page = mock.MagicMock()
        out = d.execute({"action": "key", "keys": ["esc"]})
        self.assertTrue(out["done"])
        d._page.keyboard.press.assert_called_once_with("Escape")

    def test_unknown_hotkey_not_fatal(self):
        d = BrowserDriver()
        page = mock.MagicMock()
        page.keyboard.press.side_effect = Exception("Unknown key")
        d._page = page
        out = d.execute({"action": "hotkey", "keys": ["Ctrl", "Rectangle"]})
        self.assertFalse(out["done"])


class TypeRoutingTests(unittest.TestCase):
    def test_type_without_focus_routes_through_search(self):
        d = BrowserDriver()
        page = mock.MagicMock()
        page.evaluate.return_value = "CANVAS"  # nothing editable focused
        d._page = page
        with mock.patch("agent.browser.time.sleep"):
            out = d.execute({"action": "type", "text": "rectangle"})
        self.assertTrue(out["done"])
        self.assertEqual(out["via"], "shortcut-search")
        calls = [c.args[0] for c in page.keyboard.press.call_args_list]
        self.assertEqual(calls, ["s", "Enter"])
        page.keyboard.type.assert_called_once()

    def test_type_with_focused_field_types_directly(self):
        d = BrowserDriver()
        page = mock.MagicMock()
        page.evaluate.return_value = "edit"
        d._page = page
        out = d.execute({"action": "type", "text": "20"})
        self.assertTrue(out["done"])
        page.keyboard.type.assert_called_once()
        page.keyboard.press.assert_not_called()


class DriverValidationTests(unittest.TestCase):
    def test_viewport_bounds(self):
        with self.assertRaises(ValueError):
            BrowserDriver(viewport=(10, 10))
        with self.assertRaises(ValueError):
            BrowserDriver(viewport=(99999, 900))
        d = BrowserDriver(viewport=(1440, 900))
        self.assertEqual(d.viewport, (1440, 900))

    def test_not_started_raises(self):
        d = BrowserDriver()
        with self.assertRaises(BrowserError):
            d.screenshot_png()
        with self.assertRaises(BrowserError):
            d.execute({"action": "click", "x": 1, "y": 1})

    def test_goto_refuses_other_hosts(self):
        d = BrowserDriver()
        with self.assertRaises(BrowserError):
            d.goto("https://evil.com/")


class FakeBrowserTests(unittest.TestCase):
    def test_records_actions(self):
        b = FakeBrowser()
        b.start()
        self.assertTrue(b.available)
        b.execute({"action": "click", "x": 10, "y": 20})
        b.execute({"action": "key", "keys": ["s"]})
        self.assertEqual([a["action"] for a in b.actions], ["click", "key"])
        self.assertTrue(b.screenshot_png().startswith(b"\x89PNG"))
        b.stop()
        self.assertFalse(b.available)

    def test_stop_action(self):
        b = FakeBrowser()
        b.start()
        result = b.execute({"action": "stop"})
        self.assertTrue(result.get("stop"))

    def test_goto_allowlist(self):
        b = FakeBrowser()
        b.start()
        b.goto("https://cad.onshape.com/documents")
        self.assertEqual(b.url(), "https://cad.onshape.com/documents")
        with self.assertRaises(BrowserError):
            b.goto("https://evil.com/")

    def test_unavailable_without_playwright_message(self):
        # If playwright is missing, start() must raise BrowserUnavailable with guidance.
        try:
            import playwright  # noqa: F401

            self.skipTest("playwright installed; unavailable-path not reachable")
        except ImportError:
            pass
        from agent.browser import BrowserUnavailable, _require_playwright

        with self.assertRaises(BrowserUnavailable) as ctx:
            _require_playwright()
        self.assertIn("playwright install", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
