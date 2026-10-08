"""Playwright-driven Chromium for Onshape — screenshots in, GUI actions out.

All navigation is restricted to Onshape hosts. The Playwright dependency is
imported lazily so the rest of the package (server, tests) works without it.
Playwright's sync API is not thread-safe: create and use one driver per thread.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from agent.policy import PolicyError

# The agent may only ever drive Onshape pages (plus explicit test pages).
ONSHAPE_HOST_SUFFIXES = (
    "onshape.com",
    "cad.onshape.com",
    "www.onshape.com",
)
DEFAULT_START_URL = "https://cad.onshape.com/documents"

MAX_VIEWPORT = 7680  # 8K guard for silly model coordinates


class BrowserError(Exception):
    pass


class BrowserUnavailable(BrowserError):
    """Playwright or its browser binary is not installed."""


def needs_login(url: str = "", title: str = "") -> bool:
    """True on an Onshape sign-in page — the agent must refuse to drive it
    (typing into the login form is never the user's goal)."""
    text = (url or "").strip().lower()
    if any(marker in text for marker in ("/signin", "/login", "sign_in")):
        return True
    head = (title or "").strip().lower()
    return head.startswith("sign in") or head.startswith("log in")


def is_allowed_url(url: str, extra_hosts: tuple[str, ...] = ()) -> bool:
    """True only for Onshape (or explicitly allowed) http(s) pages."""
    text = (url or "").strip().lower()
    if not text.startswith(("http://", "https://")):
        return False
    try:
        from urllib.parse import urlparse

        host = (urlparse(text).hostname or "").strip()
    except ValueError:
        return False
    if not host:
        return False
    allowed = ONSHAPE_HOST_SUFFIXES + tuple(h.lower() for h in extra_hosts)
    return any(host == h or host.endswith("." + h) for h in allowed)


def _profile_lock_error(text: str) -> bool:
    """True when a browser launch failed because the profile is held open."""
    low = (text or "").lower()
    return any(
        marker in low
        for marker in (
            "profile is already in use",
            "already in use by another instance",
            "existing browser session",
            "being used by another process",
        )
    )


def release_profile(profile_dir: str) -> int:
    """Force-close leftover agent browser windows still holding the profile.

    Only Chrome/Edge processes whose command line contains THIS profile
    directory are touched — the user's own browser sessions are untouched.
    A login window left open would otherwise make every run fail with
    "profile is already in use".
    """
    marker = str(profile_dir or "").strip()
    if not marker:
        return 0
    quoted = marker.replace("'", "''")
    script = (
        "$m = @(Get-CimInstance Win32_Process | Where-Object { "
        "($_.Name -eq 'chrome.exe' -or $_.Name -eq 'msedge.exe') -and $_.CommandLine -and "
        "$_.CommandLine.Contains('" + quoted + "') }); "
        "$m | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; "
        "$m.Count"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=20,
        )
        lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
        count = int(lines[-1]) if lines else 0
    except Exception:
        return 0
    if count > 0:
        time.sleep(0.8)  # give Windows a moment to release the profile lock
    return count


def _blur_focused(page: Any) -> None:
    """Unfocus a search/filter input so app hotkeys reach the canvas.

    Live failure: "r" pressed while the feature filter had focus typed "rr"
    into it — the filter then hid every tree entry, including the plane the
    model was trying to click.
    """
    try:
        page.evaluate(
            "() => { const a = document.activeElement;"
            " if (a && (a.tagName === 'INPUT' || a.tagName === 'TEXTAREA'"
            " || a.isContentEditable)) { a.blur(); } }"
        )
    except Exception:
        pass  # focus state is only ever a best-effort fixup


def _typing_focused(page: Any) -> bool:
    """True when an editable field (input/textarea/contenteditable) has focus."""
    try:
        info = page.evaluate(
            "() => { const a = document.activeElement; if (!a) return '';"
            " const t = (a.tagName || '').toUpperCase();"
            " const ty = (a.getAttribute('type') || '').toLowerCase();"
            " if (a.isContentEditable || t === 'TEXTAREA') return 'edit';"
            " if (t === 'INPUT' && !['button','submit','checkbox','radio','range',"
            " 'file','image','reset','hidden'].includes(ty)) return 'edit';"
            " return t; }"
        )
        return str(info) == "edit"
    except Exception:
        return False


def _require_playwright():
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except ImportError as exc:
        raise BrowserUnavailable(
            "Playwright is not installed. Run: pip install playwright && playwright install chromium"
        ) from exc
    return sync_playwright


def _page_closed(page: Any) -> bool:
    """True when a page handle is dead (window closed / browser crashed).

    Playwright's Page.is_closed() is the only reliable signal; objects
    without the method (test fakes) count as open.
    """
    checker = getattr(page, "is_closed", None)
    if not callable(checker):
        return False
    try:
        return bool(checker())
    except Exception:
        return True


class BrowserDriver:
    """Owns one Chromium page on Onshape and executes parsed policy actions."""

    def __init__(
        self,
        *,
        headless: bool = False,
        viewport: tuple[int, int] = (1440, 900),
        profile_dir: str | None = None,
        start_url: str = DEFAULT_START_URL,
        extra_hosts: tuple[str, ...] = (),
        slow_mo_ms: int = 0,
        channel: str = "",
    ):
        w = int(viewport[0])
        h = int(viewport[1])
        if not (200 <= w <= MAX_VIEWPORT and 200 <= h <= MAX_VIEWPORT):
            raise ValueError("viewport must be between 200 and 7680 pixels per side")
        self.headless = bool(headless)
        self.viewport = (w, h)
        self.profile_dir = profile_dir
        self.start_url = start_url
        self.extra_hosts = tuple(extra_hosts)
        self.slow_mo_ms = max(0, int(slow_mo_ms))
        # "" = try bundled Chromium, then fall back to system Chrome/Edge.
        self.channel = str(channel or "").strip()
        self._pw = None
        self._ctx = None
        self._page = None
        self._captured: list[dict[str, Any]] = []
        self._capture_installed = False

    # -- lifecycle ---------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._page is not None and not _page_closed(self._page)

    def start(self) -> None:
        if self._page is not None:
            if not _page_closed(self._page):
                return
            # The window was closed under us (user quit it, or the browser
            # died). Old code returned here anyway, so EVERY later run failed
            # forever with "Target page ... has been closed" — drop the dead
            # handles and fall through to a fresh launch with the same profile
            # (saved cookies keep the Onshape login).
            self.stop()
        sync_playwright = _require_playwright()
        if self.profile_dir:
            # A leftover agent window (e.g. the login window left open) holds
            # the profile lock and makes every launch fail — close it first.
            closed = release_profile(self.profile_dir)
            if closed:
                print(f"Closed {closed} leftover agent browser process(es) holding the profile", flush=True)
        self._pw = sync_playwright().start()
        launch_kwargs: dict[str, Any] = {
            "headless": self.headless,
            "slow_mo": self.slow_mo_ms,
        }
        context_viewport = {"width": self.viewport[0], "height": self.viewport[1]}
        channels = [self.channel] if self.channel else ["", "chrome", "msedge"]
        last_exc: Exception | None = None
        for ch in channels:
            for attempt in (1, 2):
                kwargs = dict(launch_kwargs)
                if ch:
                    kwargs["channel"] = ch
                try:
                    if self.profile_dir:
                        # Persistent profile keeps the user's Onshape login between runs.
                        self._ctx = self._pw.chromium.launch_persistent_context(
                            self.profile_dir, viewport=context_viewport, **kwargs
                        )
                    else:
                        browser = self._pw.chromium.launch(**kwargs)
                        self._ctx = browser.new_context(viewport=context_viewport)
                    if ch and not self.channel:
                        print(f"Using system {ch} for Onshape (bundled Chromium not installed)", flush=True)
                    last_exc = None
                    break
                except Exception as exc:  # browser missing -> try the next channel
                    last_exc = exc
                    self._ctx = None
                    if attempt == 1 and self.profile_dir and _profile_lock_error(str(exc)):
                        # A leftover instance won the race; close it and retry once.
                        release_profile(self.profile_dir)
                        continue
                    break
            if last_exc is None:
                break
        if self._ctx is None:
            self.stop()
            err = str(last_exc or "")
            if _profile_lock_error(err):
                hint = (
                    "Another agent browser window is still using this profile — "
                    "close that Chrome window (usually the sign-in window) and retry"
                )
            else:
                hint = (
                    "No Chromium found. Run: playwright install chromium "
                    "(or install Google Chrome / Microsoft Edge)"
                )
            raise BrowserUnavailable(f"{hint}. Last error: {err[:200]}")
        pages = self._ctx.pages
        self._page = pages[0] if pages else self._ctx.new_page()
        # Cookies BEFORE navigation: the Onshape auth cookie is session-scoped
        # in Chrome, so saved_cookies.json is its only copy after a restart —
        # loading it later would let the first request hit /signin.
        self._load_saved_cookies()
        self._last_cookie_save = 0.0
        if is_allowed_url(self._page.url or "", self.extra_hosts):
            self._page.goto(self.start_url, wait_until="domcontentloaded", timeout=30_000)
        elif not self._page.url or self._page.url in ("about:blank", "chrome://newtab/"):
            self._page.goto(self.start_url, wait_until="domcontentloaded", timeout=30_000)
        # Give the SPA a beat to render, so the first screenshot isn't blank.
        try:
            self._page.wait_for_load_state("load", timeout=15_000)
        except Exception:
            pass

    def _maybe_save_cookies(self) -> None:
        """Time-gated cookie save, called from driver-thread code paths only.

        Playwright's sync API is not thread-safe (a background saver thread
        raises greenlet "cannot switch to a different thread"), so saving is
        piggybacked on screenshot/execute calls at most every ~3 seconds.
        """
        if not self.profile_dir:
            return
        now = time.monotonic()
        if now - getattr(self, "_last_cookie_save", 0.0) < 3.0:
            return
        self._last_cookie_save = now
        self._save_cookies()

    def _cookies_file(self) -> Path | None:
        if not self.profile_dir:
            return None
        return Path(self.profile_dir) / "saved_cookies.json"

    def _load_saved_cookies(self) -> None:
        path = self._cookies_file()
        if path is None or not path.is_file() or self._ctx is None:
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                self._ctx.add_cookies(data)
        except Exception:
            pass  # a corrupt cookie file must never block startup

    def _save_cookies(self) -> None:
        path = self._cookies_file()
        if path is None or self._ctx is None:
            return
        try:
            cookies = self._ctx.cookies()
            path.write_text(json.dumps(cookies, indent=0), encoding="utf-8")
        except Exception:
            pass  # best-effort; without cookies the next run just re-logs-in

    def stop(self) -> None:
        # Persist cookies BEFORE closing: session-scoped auth cookies are lost
        # otherwise (verified: Chrome drops them on restart in this setup).
        self._save_cookies()
        for closer in (
            lambda: self._ctx and self._ctx.close(),
            lambda: self._pw and self._pw.stop(),
        ):
            try:
                closer()
            except Exception:
                pass
        self._page = None
        self._ctx = None
        self._pw = None

    def __enter__(self) -> "BrowserDriver":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- observation -------------------------------------------------------

    def screenshot_png(self) -> bytes:
        page = self._require_page()
        self._maybe_save_cookies()
        return page.screenshot(type="png")

    def url(self) -> str:
        if self._page is None:
            return ""
        try:
            return str(self._page.url or "")
        except Exception:
            return ""

    def title(self) -> str:
        if self._page is None:
            return ""
        try:
            # Polled in a tight loop by `agent login`; piggyback the cookie save.
            self._maybe_save_cookies()
            return str(self._page.title() or "")
        except Exception:
            return ""

    def alive(self) -> bool:
        """True while the browser window is still open (used by `agent login`)."""
        if self._page is None:
            return False
        try:
            self._page.url
            self._page.title()
        except Exception:
            return False
        return True

    def _require_page(self):
        if self._page is None:
            raise BrowserError("Browser is not started")
        return self._page

    # -- actions -----------------------------------------------------------

    def execute(self, action: dict[str, Any], *, stop_check=None) -> dict[str, Any]:
        """Run one parsed action. Returns a small result dict."""
        page = self._require_page()
        self._maybe_save_cookies()
        kind = str(action.get("action") or "")
        x = int(action.get("x") or 0)
        y = int(action.get("y") or 0)
        x2 = int(action.get("x2") or 0)
        y2 = int(action.get("y2") or 0)
        if not (0 <= x < self.viewport[0] and 0 <= y < self.viewport[1]):
            # Out-of-viewport clicks are model misses; clamp instead of crashing.
            x = max(0, min(self.viewport[0] - 1, x))
            y = max(0, min(self.viewport[1] - 1, y))

        if kind == "click":
            page.mouse.click(x, y)
            return {"done": True, "clicked": [x, y]}
        if kind == "dblclick":
            page.mouse.dblclick(x, y)
            return {"done": True, "dblclicked": [x, y]}
        if kind == "drag":
            x2 = max(0, min(self.viewport[0] - 1, x2))
            y2 = max(0, min(self.viewport[1] - 1, y2))
            page.mouse.move(x, y)
            page.mouse.down()
            # Steps give the CAD viewport a chance to process the drag.
            page.mouse.move(x2, y2, steps=12)
            page.mouse.up()
            return {"done": True, "dragged": [x, y, x2, y2]}
        if kind == "scroll":
            dx = int(action.get("dx") or 0)
            dy = int(action.get("dy") or 0)
            page.mouse.move(x, y)
            page.mouse.wheel(dx, dy)
            return {"done": True, "scrolled": [dx, dy]}
        if kind == "key":
            keys = action.get("keys") or []
            if not keys:
                return {"done": False, "error": "no keys"}
            pressed = []
            for raw in keys[:3]:
                key = _playwright_key(str(raw))
                if len(key) == 1 and key.isalpha() and _typing_focused(page):
                    # Letter hotkeys must reach the app, not a focused
                    # search/filter box (which would swallow and keep them).
                    _blur_focused(page)
                try:
                    page.keyboard.press(key)
                except Exception:
                    # Model narration like "Rectangle" is not a key name —
                    # skip it instead of killing the whole run.
                    continue
                pressed.append(key)
            if not pressed:
                return {"done": False, "error": f"unknown key: {keys[0]}"}
            return {"done": True, "pressed": pressed}
        if kind == "hotkey":
            keys = action.get("keys") or []
            if not keys:
                return {"done": False, "error": "no keys"}
            # Playwright accepts "Control+z" style combos.
            combo = "+".join(_playwright_key(k) for k in keys[:3])
            try:
                page.keyboard.press(combo)
            except Exception:
                return {"done": False, "error": f"unknown hotkey: {combo}"}
            return {"done": True, "pressed": combo}
        if kind == "type":
            text = str(action.get("text") or "")
            if not text:
                return {"done": False, "error": "no text"}
            if _typing_focused(page):
                page.keyboard.type(text, delay=25)
                return {"done": True, "typed": len(text)}
            # Nothing focused: typing would silently vanish. Run it through the
            # shortcut search (S -> text -> Enter) — what the model MEANS when
            # it "types" a tool name at the canvas. This fixed the run that
            # typed "rectangle" 46 times without ever opening the search.
            page.keyboard.press("s")
            time.sleep(0.15)
            page.keyboard.type(text, delay=25)
            time.sleep(0.1)
            page.keyboard.press("Enter")
            return {"done": True, "typed": len(text), "via": "shortcut-search"}
        if kind == "wait":
            seconds = float(action.get("seconds") or 0.5)
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                if stop_check is not None and stop_check():
                    return {"done": True, "interrupted": True}
                time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
            return {"done": True, "waited": round(seconds, 2)}
        if kind == "stop":
            return {"done": True, "stop": True}
        raise PolicyError(f"Unknown action kind: {kind!r}")

    def goto(self, url: str) -> None:
        if not is_allowed_url(url, self.extra_hosts):
            raise BrowserError(f"Navigation refused (not an allowed Onshape URL): {url!r}")
        page = self._require_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)

    # -- demo recording -----------------------------------------------------

    def start_input_capture(self) -> None:
        """Listen for the human's clicks/keys inside our browser tab.

        Only events inside this Playwright page are seen — unlike a global keylogger,
        typing in other windows is never captured.
        """
        page = self._require_page()
        if self._capture_installed:
            return
        self._captured = []

        def _on_rec(evt: dict[str, Any]) -> None:
            try:
                kind = str(evt.get("kind") or "")
                if kind == "click":
                    self._captured.append(
                        {
                            "action": "click",
                            "x": int(evt.get("x") or 0),
                            "y": int(evt.get("y") or 0),
                            "target": str(evt.get("target") or "")[:64],
                            "reason": "recorded user click",
                        }
                    )
                elif kind == "key":
                    key = str(evt.get("key") or "")
                    if key and key not in {"Control", "Shift", "Alt", "Meta"}:
                        self._captured.append(
                            {"action": "key", "keys": [key], "reason": "recorded user key"}
                        )
            except Exception:
                pass

        try:
            page.expose_function("__agentRec", _on_rec)
            page.evaluate(
                """() => {
                    if (window.__agentRecInstalled) return;
                    window.__agentRecInstalled = true;
                    document.addEventListener('click', (e) => {
                        const t = e.target;
                        const name = t && (t.getAttribute('aria-label') || t.title || t.tagName || '');
                        window.__agentRec({kind: 'click', x: e.clientX, y: e.clientY, target: String(name).slice(0, 64)});
                    }, true);
                    document.addEventListener('keydown', (e) => {
                        window.__agentRec({kind: 'key', key: String(e.key || '').slice(0, 24)});
                    }, true);
                }"""
            )
            self._capture_installed = True
        except Exception as exc:
            raise BrowserError(f"Cannot install input capture: {str(exc)[:120]}") from exc

    def take_captured_actions(self) -> list[dict[str, Any]]:
        """Drain user actions recorded since the last call."""
        out = list(self._captured)
        self._captured = []
        return out


_KEY_ALIASES = {
    "esc": "Escape",
    "escape": "Escape",
    "enter": "Enter",
    "return": "Enter",
    "space": "Space",
    "tab": "Tab",
    "backspace": "Backspace",
    "delete": "Delete",
    "del": "Delete",
    "up": "ArrowUp",
    "down": "ArrowDown",
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "arrowup": "ArrowUp",
    "arrowdown": "ArrowDown",
    "arrowleft": "ArrowLeft",
    "arrowright": "ArrowRight",
    "ctrl": "Control",
    "control": "Control",
    "cmd": "Meta",
    "meta": "Meta",
    "win": "Meta",
    "alt": "Alt",
    "shift": "Shift",
    "plus": "+",
}


def _playwright_key(key: str) -> str:
    raw = str(key or "").strip()
    if not raw:
        return raw
    low = raw.lower()
    if low in _KEY_ALIASES:
        return _KEY_ALIASES[low]
    if len(raw) == 1:
        return raw
    # "PageDown", "F5", ... — Playwright uses these names directly.
    return raw[0].upper() + raw[1:]


class FakeBrowser:
    """Test double with the same surface as BrowserDriver (no Playwright needed)."""

    def __init__(self, viewport: tuple[int, int] = (1440, 900)):
        self.viewport = viewport
        self.actions: list[dict[str, Any]] = []
        self.captured: list[dict[str, Any]] = []
        self._started = False
        self.current_url = "https://cad.onshape.com/documents"

    @property
    def available(self) -> bool:
        return self._started

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def screenshot_png(self) -> bytes:
        # 1x1 transparent PNG; enough for pipeline tests.
        import base64

        return base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg=="
        )

    def url(self) -> str:
        return self.current_url

    def title(self) -> str:
        return "Fake Onshape"

    def execute(self, action: dict[str, Any], *, stop_check=None) -> dict[str, Any]:
        self.actions.append(dict(action))
        if str(action.get("action")) == "stop":
            return {"done": True, "stop": True}
        return {"done": True}

    def goto(self, url: str) -> None:
        if not is_allowed_url(url, ("localhost", "127.0.0.1")):
            raise BrowserError(f"Navigation refused: {url!r}")
        self.current_url = url

    def start_input_capture(self) -> None:
        pass

    def take_captured_actions(self) -> list[dict[str, Any]]:
        out = list(self.captured)
        self.captured = []
        return out
