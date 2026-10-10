"""Main agent loop: plan + vision steps driving the Onshape browser tab."""

from __future__ import annotations

import base64
import concurrent.futures
import json
import queue
import random
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from agent.browser import (
    DEFAULT_START_URL,
    BrowserDriver,
    BrowserError,
    FakeBrowser,
    is_allowed_url,
    needs_login,
)
from agent.images import encode_jpeg_b64
from agent.memory import AgentMemory, action_signature, goals_match
from agent.ollama import (
    OllamaError,
    chat_text_json,
    chat_vision_json,
    prefer_text_model,
    prefer_vision_model,
    resolve_planner_model,
)
from agent.paths import ensure_dataset, prune_old_files, screenshots_dir
from agent.policy import PLAN_SYSTEM_PROMPT, SYSTEM_PROMPT, PolicyError, parse_action, parse_plan

MAX_STEPS_CAP = 2500
DEFAULT_MAX_STEPS = 160
NEEDS_LOGIN_DETAIL = (
    "NEEDS_LOGIN: Onshape is not signed in — click “Sign in to Onshape” in this app, "
    "sign in once in the browser window, then run again"
)
LOGIN_WAIT_DETAIL = (
    "Onshape sign-in needed — sign in in the browser window; "
    "the agent waits for you and continues automatically"
)


def _clip(value: Any, limit: int) -> str:
    text = str(value or "")
    return text[:limit]


def _doc_workspace_url(url: Any) -> bool:
    """True for a document workspace URL (/documents/<id>/...), not the list."""
    text = str(url or "")
    marker = "/documents/"
    i = text.find(marker)
    if i < 0:
        return False
    seg = text[i + len(marker) :].split("/", 1)[0].split("?", 1)[0]
    return len(seg) >= 12 and seg.isalnum()


# Mean grayscale delta (0-255) under which two frames count as "unchanged".
# Hover ripples and JPEG noise measured ~1-4 on real runs; any real state
# change (dialog, selection, grid) lands far above this.
_STATIC_DIFF_MAX = 4.0

# Mean grayscale delta that counts as a REAL screen change (used to verify an
# expected overlay appeared). Measured on a live run: a menu opening/closing
# moves 32-95, hover/click shimmer only 0-0.7 — 8.0 separates them cleanly.
_VISIBLE_CHANGE_MIN = 8.0

# After this many Esc-unstick cycles with no visible change, reload the page
# instead — a live-but-dead page (handlers gone silent) ignores Esc forever;
# observed burning all 160 steps of a run. Reload restores the handlers.
_MAX_PAGE_RELOADS = 3


# A click target within this many pixels of the screenshot origin is
# degenerate: Onshape's top strip is OS/browser chrome; the model occasionally
# degenerates to 0..15-px 'clicks' which ARE harmless-noise (and, executed
# dozens of times, what made runs look 'random').
_DEGENERATE_CLICK_MAX_PX = 12


def _sanitize_action(action: dict[str, Any], vp: tuple[int, int]) -> tuple[dict[str, Any], str]:
    """Validate/fix one model action; returns (action, reject_reason).

    reject_reason is empty when the action is fine (possibly after clamping).
    """
    kind = str(action.get("action") or "")
    vw, vh = vp
    if kind in {"click", "dblclick"}:
        # DOM-label clicks carry no pixel coordinates: they're resolved in
        # the driver against the REAL page (tooltip/aria labels), so the
        # degenerate-coordinate checks below don't apply to them.
        if str(action.get("target") or "").strip():
            return action, ""
        x, y = int(action.get("x") or 0), int(action.get("y") or 0)
        # Degenerate: inside the chrome strip / ON the logo area. The most
        # reliable live failure mode was (0,0)-(0,24) repeated "clicks".
        if x < _DEGENERATE_CLICK_MAX_PX and y < _DEGENERATE_CLICK_MAX_PX:
            return action, f"rejected degenerate click at ({x},{y}) — top-left corner is not a target"
        # Demonstrated live: 46 % up-left offset when the model assumed a
        # different viewport — clamp so a *slightly* off target still lands.
        if not (0 <= x < vw and 0 <= y < vh):
            action["x"] = max(0, min(vw - 1, x))
            action["y"] = max(0, min(vh - 1, y))
        return action, ""
    if kind == "type":
        text = str(action.get("text") or "").strip()
        if not text:
            return action, "rejected empty type action"
    return action, ""


def _frame_signature(b64: str) -> list[float]:
    """Tiny grayscale thumbnail used for change detection."""
    try:
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L").resize((48, 30))
        return [float(v) for v in img.tobytes()]
    except Exception:
        return []


def _signature_diff(a: list[float], b: list[float]) -> float:
    if not a or len(a) != len(b):
        return float("inf")
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


class BrowserThreadProxy:
    """Run every browser call on ONE long-lived owner thread.

    Playwright's sync API is bound to the thread that created the driver —
    a call from any other thread raises greenlet's "cannot switch to a
    different thread". Each run gets a fresh thread while the driver (and its
    browser window) survives across runs, so without this the SECOND run died
    instantly with that cryptic error. A single owner thread also serializes
    the GUI's status polls with the run's actions instead of racing Playwright
    from two threads at once.
    """

    # Attribute reads that must return a VALUE (not a proxied method).
    _VALUE_ATTRS = frozenset({"available", "viewport"})

    def __init__(self, factory: Any):
        self._factory = factory
        self._driver: Any | None = None
        self._jobs: "queue.Queue[tuple[Any, concurrent.futures.Future[Any]]]" = queue.Queue()
        self._pending = 0
        self._pending_lock = threading.Lock()
        thread = threading.Thread(target=self._serve, name="browser-owner", daemon=True)
        thread.start()

    def _serve(self) -> None:
        while True:
            job, fut = self._jobs.get()
            if job is None:
                return
            try:
                if self._driver is None:
                    self._driver = self._factory()
                fut.set_result(job(self._driver))
            except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
                fut.set_exception(exc)

    def _submit(self, job: Any) -> concurrent.futures.Future[Any]:
        fut: concurrent.futures.Future[Any] = concurrent.futures.Future()
        with self._pending_lock:
            self._pending += 1
        self._jobs.put((job, fut))
        return fut

    def _release(self) -> None:
        with self._pending_lock:
            self._pending -= 1

    def call(self, fn: Any) -> Any:
        """Run fn(driver) on the owner thread and return its result."""
        fut = self._submit(fn)
        try:
            return fut.result()
        finally:
            self._release()

    def call_if_idle(self, fn: Any, default: Any = None) -> Any:
        """Like call(), but never blocks — returns default while busy.

        The GUI polls /status continuously; probing the browser mid-action
        must not hold up the poll (a >5s stall looks like a dead sidecar).
        """
        with self._pending_lock:
            if self._pending:
                return default
        return self.call(fn)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._VALUE_ATTRS:
            return self.call(lambda driver: getattr(driver, name))

        def _method(*args: Any, **kwargs: Any) -> Any:
            return self.call(lambda driver: getattr(driver, name)(*args, **kwargs))

        return _method


class AgentLoop:
    def __init__(
        self,
        dataset_root: Path | None = None,
        browser: Any | None = None,
        driver_factory: Any | None = None,
    ):
        self.root = ensure_dataset(dataset_root)
        self.memory = AgentMemory(self.root)
        # browser may be injected (tests use FakeBrowser); otherwise Playwright is
        # started lazily on the agent thread only. driver_factory overrides the
        # BrowserDriver construction (test seam for thread-affinity tests).
        self._browser_factory = browser
        self._driver_factory = driver_factory
        self._browser: Any | None = None
        self._last_browser_url = ""
        # Requested browser window state. "" = normal/maximized window;
        # "fullscreen" = hide browser chrome + OS chrome so the model gets the
        # whole screen and Onshape gets room for the real UI.
        self._window_state = ""
        self._thread: threading.Thread | None = None
        self._video_thread: threading.Thread | None = None
        self._lifecycle = threading.Lock()
        self._stop = threading.Event()
        self._running = False
        self._recording = False
        self._learning_video = False
        self._status = "Idle"
        self._detail = ""
        self._error = ""
        self._goal = ""
        self._step = 0
        self._max_steps = DEFAULT_MAX_STEPS
        self._model = ""
        self._planner_model = ""
        self._mode = "idle"  # idle | plan | vision | record | learn
        self._window_state_requested = False  # set True right after a run applies it
        self._plan: list[dict[str, str]] = []
        self._plan_i = 0
        # (plan index, frame signature before the action) awaiting visual
        # verification of an overlay the model failed to narrate.
        self._pending_appeared: tuple[int, list[float]] | None = None
        self._unstick_cycles = 0
        self._reloads_done = 0
        self._last_action: dict[str, Any] = {}
        self._last_reason = ""
        self._recent_signatures: list[str] = []
        self._record_actions: list[dict[str, Any]] = []
        self._record_shots: list[str] = []
        self._record_goal = ""
        self._learn_percent = 0.0
        self._learn_phase = ""
        self._learn_history: list[str] = []

    # ------------------------------------------------------------- status

    def status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "recording": self._recording,
            "learning_video": self._learning_video,
            "status": self._status,
            "detail": self._detail,
            "error": self._error,
            "goal": self._goal,
            "step": self._step,
            "max_steps": self._max_steps,
            "mode": self._mode,
            "model": self._model,
            "planner_model": self._planner_model,
            "plan_preview": self._plan[:40],
            "plan_index": self._plan_i,
            "last_action": self._last_action,
            "last_reason": self._last_reason,
            "learn_percent": self._learn_percent,
            "learn_phase": self._learn_phase,
            "learn_history": self._learn_history[-8:],
            # status() crosses the HTTP boundary (GUI polls it constantly):
            # never include the absolute dataset path there — the GUI derives
            # the folder locally via agent.paths.default_dataset_dir().
            "memory": {k: v for k, v in self.memory.stats().items() if k != "root"},
            "browser_url": self._safe_browser_url(),
            "window_state": self._window_state,
        }

    def _safe_browser_url(self) -> str:
        # /status is polled continuously by the GUI; a crashed/closed browser
        # must not turn every poll into an HTTP 500. The probe is also skipped
        # (last URL returned) while the owner thread is mid-action so the poll
        # never blocks on the browser.
        if self._browser is None:
            return ""
        try:
            if isinstance(self._browser, BrowserThreadProxy):
                url = self._browser.call_if_idle(lambda d: str(d.url() or ""), default=None)
                if url is None:
                    return self._last_browser_url
                self._last_browser_url = url
                return url
            return str(self._browser.url() or "")
        except Exception:
            return ""

    def busy(self) -> bool:
        return self._running or self._recording or self._learning_video

    def stop(self, timeout: float = 12.0) -> None:
        """Signal the agent/record thread to stop and wait for it."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        # The video-learn thread polls _stop between segments/cards (yt-dlp and
        # Ollama calls stay blocking, so it cannot be joined instantly); the
        # StopIteration handler above flips its status to "Stopped".
        self._running = False
        self._recording = False
        if not self._learning_video and self._status not in {"Stopped", "Error"}:
            self._status = "Idle"
            self._mode = "idle"

    # ------------------------------------------------------------- browser

    def set_window_state(self, state: str) -> dict[str, Any]:
        """Users control the agent browser's window state from the GUI.

        Accepted: "normal" (restore/maximized), "fullscreen" (hide OS +
        browser chrome). The state is remembered and RE-APPLIED automatically
        the next time a fresh browser window is launched (the browser is left
        open between runs; launching a new one resets window chrome).
        """
        wanted = str(state or "").strip().lower()
        if wanted in {"", "normal", "maximize", "windowed", "restore"}:
            wanted = ""
        elif wanted in {"fullscreen", "f11", "full"}:
            wanted = "fullscreen"
        else:
            raise ValueError("window_state must be 'normal' or 'fullscreen'")
        self._window_state = wanted
        applied = {"done": True, "window_state": wanted}
        try:
            browser = self._ensure_browser()
            if hasattr(browser, "toggle_window_state"):
                m = getattr(browser, "toggle_window_state")
                # MERGE: the driver's result carries viewport/fullscreen detail
                # but always keep our canonical "window_state" key.
                applied = {**applied, **(m(wanted or "normal") or {})}
                applied["window_state"] = wanted
        except Exception as exc:
            applied = {"done": False, "error": str(exc)[:200], "window_state": wanted}
        return applied

    def _ensure_browser(self):
        if self._browser is not None and getattr(self._browser, "available", False):
            return self._browser
        if self._browser_factory is not None:
            self._browser = self._browser_factory
            if not getattr(self._browser, "available", False):
                self._browser.start()
            self._reapply_window_state()
            return self._browser
        if self._browser is None:
            # One permanent owner thread for the driver (see BrowserThreadProxy):
            # runs each get their own thread, and Playwright's sync API refuses
            # to be called from a thread other than the one that created it.
            self._browser = BrowserThreadProxy(
                self._driver_factory
                or (lambda: BrowserDriver(headless=False, profile_dir=str(self.root / "browser_profile")))
            )
        self._browser.start()
        self._reapply_window_state()
        return self._browser

    def _reapply_window_state(self) -> None:
        """A freshly launched browser window loses any prior fullscreen —
        remember and re-apply the user's choice once, after one start."""
        if not self._window_state or self._window_state_requested:
            return
        self._window_state_requested = True
        try:
            applier = getattr(self._browser, "toggle_window_state", None)
            if callable(applier):
                applier(self._window_state)  # blocks until applied
        except Exception:
            # Smoke once more on the next fresh start (window may just have
            # raced our call); never break the run over chrome.
            self._window_state_requested = False

    def _shutdown_browser_if_owned(self) -> None:
        # Injected browsers (tests) are not closed by the loop.
        if self._browser_factory is not None:
            return
        if self._browser is not None:
            try:
                self._browser.stop()
            except Exception:
                pass
            self._browser = None

    def close_browser(self) -> None:
        """Explicitly close the agent's browser (used when the sidecar quits).

        Runs/records/replays no longer close it themselves — the user needs to
        see the finished model instead of the tab vanishing.
        """
        self._shutdown_browser_if_owned()

    def _note_browser_left_open(self) -> None:
        if self._browser is not None and not self._detail.endswith("(browser left open)"):
            try:
                if getattr(self._browser, "available", False):
                    self._detail = f"{self._detail} (browser left open)"
            except Exception:
                pass

    # ------------------------------------------------------------ planning

    # -- deterministic primitive plans -------------------------------------
    # A local 3B/27B planner cannot reliably plan even a cube: the last live
    # run replayed a stale "Create button at (109,66)" step inside an open
    # document and burned 30 steps on pixel guesses while never emitting ONE
    # target click. For goals that name a primitive solid, the plan is fully
    # authored: every named-UI step is a target click the driver resolves
    # against real DOM labels; only the two canvas corner clicks are pixel
    # actions and even those are corrected by the user text hints.
    _PRIMITIVE_GOAL_RE = re.compile(
        r"\b(cube|box|block|square|rectangular\s+prism)\b", re.I
    )

    @classmethod
    def _primitive_plan(cls, goal: str) -> list[dict[str, str]] | None:
        """Authored step plan for primitive solid goals, or None.

        Returns plan items in the same {step, expect} shape parse_plan emits,
        so the run loop, plan advance logic and status reporting need no
        changes. Only used when the page is ALREADY a workspace, documents-
        list open-document steps stay under the planner's control.
        """
        text = str(goal or "")
        if not cls._PRIMITIVE_GOAL_RE.search(text):
            return None
        # Size defaults: "10" in the text wins, else 10mm square x 10 deep.
        numbers = [int(n) for n in re.findall(r"\b(\d{1,4})(?:\.\d+)?\b", text) if 0 < int(n) <= 2000]
        size = numbers[0] if numbers else 10
        steps = [
            {
                # Works FROM EITHER PAGE: the open-document act measures where
                # we are (loop-side), the sketch act drives the workspace side.
                "step": "open the Part Studio (most recent document, if on the documents list)",
                "expect": "workspace URL /documents/<id>/w/<id>/e/<id>",
                "acts": [{"action": "dblclick", "target": "recent document card", "reason": "authored: open document"}],
                "only_if": "documents_list",
            },
            {
                "step": "start a new sketch",
                "expect": "sketch plane selection appears",
                "acts": [{"action": "click", "target": "Sketch", "reason": "authored: open Sketch"}],
            },
            {
                "step": "sketch on the Top plane",
                "expect": "sketch opens on the Top plane",
                "acts": [{"action": "click", "target": "Top", "reason": "authored: Top sketch plane"}],
            },
            {
                "step": "pick the Corner rectangle tool",
                "expect": "rectangle tool active",
                "acts": [{"action": "click", "target": "Corner rectangle", "reason": "authored: rectangle tool"}],
            },
            {
                "step": "place the first corner at the sketch origin (canvas center)",
                "expect": "first corner set",
                "acts": [{"action": "click", "origin": True, "reason": "authored: origin corner"}],
            },
            {
                "step": "place the opposite corner down-right of the origin",
                "expect": "rectangle drawn on the Top plane",
                "acts": [{"action": "click", "origin": True, "dx": 170, "dy": 170, "reason": "authored: opposite corner"}],
            },
            {
                "step": (
                    f"finish the sketch with the green check; if a dimension is pending, "
                    f"type {size} for it — the exact size comes right after via dimensions"
                ),
                "expect": "sketch finished, back in 3D view",
                "acts": [{"action": "click", "target": "check", "reason": "authored: finish sketch"}],
            },
            {"step": f"dimension the bottom edge of the rectangle to {size} mm (click Dimension, click the bottom edge, place the dimension, type {size}, Enter)", "expect": "dimension value applied"},
            {"step": f"dimension the side edge of the rectangle to {size} mm the same way", "expect": "both dimensions applied"},
            {
                "step": f"open Extrude and set the depth to {size} mm",
                "expect": "extrude dialog opens with the depth field",
                "acts": [
                    {"action": "click", "target": "Extrude", "reason": "authored: open Extrude"},
                    {"action": "click", "target": "Depth field", "reason": "authored: focus Depth"},
                    {"action": "type", "text": str(size), "reason": "authored: depth value"},
                    {"action": "key", "keys": ["enter"], "reason": "authored: commit depth"},
                ],
            },
            {
                "step": f"accept the extrude — depth {size} mm",
                "expect": f"solid ~{size}x{size}x{size} block appears; Parts (1) in the tree",
                "acts": [{"action": "click", "target": "check", "reason": "authored: accept extrude"}],
            },
            {
                # The extrude may auto-accept when Enter commits the depth;
                # an extra check click just confirms nothing new opens.
                "step": "verify the part exists (Parts row in the tree)",
                "expect": "Parts count is now 1",
            },
        ]
        return steps

    def make_plan(
        self,
        goal: str,
        *,
        model: str = "",
        planner_model: str = "",
        page_hint: str = "",
        memory_block: str = "",
    ) -> list[dict[str, str]]:
        """Ask the local text model for a step plan for this goal."""
        goal = _clip(goal, 2000).strip()
        if not goal:
            raise ValueError("Goal is required")
        # Deterministic authored plan for primitive solids (cube, box, square,
        # plate): the planner was REPLACED by this when a goal matches, because
        # the local model reused older plans with pixels from a different
        # session while never using target clicks the driver understands.
        page_l = (page_hint or "").lower()
        workspace = "/documents/" in page_l and re.search(r"/e/", page_l)
        prim = self._primitive_plan(goal)
        if prim:
            # The recipe now covers the documents list too (its first step
            # opens the most recent document). On a workspace the open-
            # document step is skipped at execution time via 'only_if'.
            self._planner_model = "authored-recipe"
            return prim
        text_model = resolve_planner_model(planner_model or "auto")
        if not text_model:
            raise OllamaError("No local Ollama model available for planning")
        page = page_hint.strip() or "unknown (Onshape documents list or Part Studio)"
        user = (
            f"Goal: {goal}\n"
            f"Current page: {page}\n"
            "If the page is the documents list, the plan must first open a document.\n"
            "If the page is ALREADY a document workspace (URL has /documents/<id>/.../e/...), "
            "do NOT plan any create-document or open-document step — go straight to modeling.\n"
            "Right after a sketch starts, Onshape asks for a plane: include an explicit "
            "step to click a plane (Top/Front/Right).\n"
        )
        if memory_block.strip():
            # Learned video techniques belong in the PLAN, not just the steps.
            user += (
                "Known techniques learned from video lessons "
                "(these are HINTS — use the wording where it fits, but every step must "
                "still be a concrete GUI action whose precondition is already met; do not "
                "copy raw keystrokes such as 's -> \"rectangle\" -> enter' into the plan):\n"
                f"{memory_block.strip()}\n"
            )
        user += "Produce the plan now."
        raw = chat_text_json(text_model, system=PLAN_SYSTEM_PROMPT, user_text=user, num_predict=1400)
        plan = parse_plan(raw)
        # NOTE: deliberately does NOT touch self._plan / self._plan_i — a plan
        # PREVIEW request while a run is active used to clobber the running
        # agent's plan pointer. Callers that own the state assign it themselves.
        self._planner_model = text_model
        return plan

    def preview_plan(self, *, goal: str, model: str = "", planner_model: str = "") -> dict[str, Any]:
        # Use the REAL browser page as the hint: /plan while the Part Studio
        # is open must show the same authored recipe a run would use — not an
        # LLM plan written for the documents list (the last live preview said
        # "open a document" while the document was already open).
        hint = ""
        try:
            browser = self._ensure_browser()
            hint = f"{browser.title()} ({browser.url()})"
        except Exception:
            hint = ""
        plan = self.make_plan(
            goal,
            model=model,
            planner_model=planner_model,
            page_hint=hint,
            memory_block=self.memory.prompt_block(goal),
        )
        return {"steps": plan, "count": len(plan), "planner_model": self._planner_model}

    # ------------------------------------------------------------ run loop

    def start_run(
        self,
        *,
        goal: str,
        model: str = "",
        planner_model: str = "",
        max_steps: int = DEFAULT_MAX_STEPS,
        use_plan: bool = True,
        run_mode: str = "learn",
    ) -> None:
        if self.busy():
            raise RuntimeError("Stop the current run/record/learn first")
        goal = _clip(goal, 2000).strip()
        if not goal:
            raise ValueError("Goal is required")
        max_steps = max(1, min(MAX_STEPS_CAP, int(max_steps or DEFAULT_MAX_STEPS)))

        with self._lifecycle:
            self._stop.clear()
            self._error = ""
            self._goal = goal
            self._step = 0
            self._max_steps = max_steps
            self._model = model or ""
            self._running = True
            self._mode = "plan" if use_plan else "vision"
            self._status = "Starting"
            self._detail = f"Preparing browser for: {goal}"
            self._plan = []
            self._plan_i = 0
            self._pending_appeared = None
            self._window_state_requested = False
            self._unstick_cycles = 0
            self._reloads_done = 0
            self._recent_signatures = []
            self._last_action = {}
            self._last_reason = ""
            try:
                (self.root / "run_log.jsonl").write_text("", encoding="utf-8")
            except OSError:
                pass

            def _run() -> None:
                try:
                    self._run_worker(
                        goal=goal,
                        model=model,
                        planner_model=planner_model,
                        max_steps=max_steps,
                        use_plan=use_plan,
                        run_mode=run_mode,
                    )
                except Exception as exc:
                    self._error = str(exc)[:300]
                    self._status = "Error"
                    self._detail = self._error
                finally:
                    self._running = False
                    self._mode = "idle" if not self._learning_video else "learn"
                    if self._status not in {"Error", "Stopped"}:
                        self._status = "Idle"
                    # Browser is deliberately LEFT OPEN: the user must see the
                    # result (closing their Onshape tab at run end was the
                    # #1 complaint — work looked lost even though Onshape
                    # autosaves). Close it only via close_browser()/quit.
                    self._note_browser_left_open()

            self._thread = threading.Thread(target=_run, name="agent-run", daemon=True)
            self._thread.start()

    def _resolve_model(self, model: str) -> str:
        from agent.ollama import list_ollama_models

        if model:
            return model
        try:
            models = list_ollama_models(timeout=3.0)
        except OllamaError:
            models = []
        return prefer_vision_model(models) if models else "qwen2.5vl"

    # Perceptual JPEG quality for frames sent to the vision model. 80 smoothed
    # away thin Onshape overlays (dimension labels, hover outlines); 92 keeps
    # them and is still far below the point where the VLM's context becomes
    # the bottleneck.
    VLM_JPEG_QUALITY = 92

    def _observe(self) -> tuple[str, dict[str, Any]]:
        browser = self._ensure_browser()
        png = browser.screenshot_png()
        # Send the screenshot at VIEWPORT resolution: the VLM answers in the
        # pixel space of the image it RECEIVES, so downscaling would silently
        # shrink every coordinate (verified live: clicks landed ~46% up-left).
        b64 = encode_jpeg_b64(png, max_side=max(browser.viewport), quality=self.VLM_JPEG_QUALITY)
        obs = {"url": browser.url(), "title": browser.title()}
        self._last_browser_url = str(obs["url"] or "")
        return b64, obs

    def _viewport_wh(self) -> tuple[int, int]:
        """Current browser viewport (best-effort (1440,900) before it exists)."""
        try:
            vp = self._ensure_browser().viewport
            return int(vp[0]), int(vp[1])
        except Exception:
            return 1440, 900

    def _user_text(self, goal: str, obs: dict[str, Any], memory_block: str, plan_line: str) -> str:
        recent = ", ".join(self._recent_signatures[-8:])
        # The model must answer in the SAME pixel space the screenshot uses.
        # After a viewport change (fullscreen on/off) the old hardcoded
        # 1440x900 in the prompt made it emit coordinates beyond the real
        # screen — clamped to (0,0)-ish garbage clicks.
        vw, vh = self._viewport_wh()
        parts = [
            f"GOAL: {goal}",
            plan_line,
            f"Page: {obs.get('title') or 'Onshape'} ({obs.get('url') or ''})",
            f"Screen: {vw}x{vh} pixels. Coordinates must be 0<x<{vw}, 0<y<{vh}.",
            f"Step: {self._step + 1}/{self._max_steps}.",
            f"Last action: {json.dumps(self._last_action)[:220] if self._last_action else '(none)'}",
            f"Last reason: {self._last_reason or '(none)'}",
        ]
        if recent:
            parts.append(f"Recent action signatures (do not repeat): {recent}")
        if len(self._recent_signatures) >= 3 and len(set(self._recent_signatures[-3:])) == 1:
            # The model kept re-typing the same command against an unchanged
            # screen (live: 104 of 200 steps were no-op repeats). Tell it
            # bluntly instead of hoping it notices.
            parts.append(
                "STUCK: your last identical actions had NO visible effect. "
                "Do NOT repeat them — change approach (click a visible control, "
                "pick a plane, select geometry) or, if truly finished, stop."
            )
        if memory_block:
            parts.append("MEMORY:\n" + memory_block)
        parts.append("Respond with ONE JSON action object.")
        return "\n".join(p for p in parts if p)

    def _execute(self, action: dict[str, Any]) -> dict[str, Any]:
        browser = self._ensure_browser()
        return browser.execute(action, stop_check=self._stop.is_set)

    # How long a run waits for the user to finish signing in to Onshape.
    LOGIN_WAIT_SEC = 120.0

    def _await_login(self, browser: Any) -> bool:
        """Wait for the user to sign in instead of killing the run.

        The browser is sitting on Onshape's sign-in page right after the user
        clicked Run, so failing instantly made a one-click run impossible.
        The status explains what is needed and the run continues as soon as
        the page is signed in. Returns False on timeout or stop.
        """
        deadline = time.monotonic() + max(0.0, float(self.LOGIN_WAIT_SEC))
        self._status = "Needs login"
        self._detail = LOGIN_WAIT_DETAIL
        while True:
            if self._stop.is_set():
                return False
            try:
                url, title = str(browser.url() or ""), str(browser.title() or "")
            except Exception:
                url, title = "", ""
            if not needs_login(url, title):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(1.0)

    def _run_worker(
        self,
        *,
        goal: str,
        model: str,
        planner_model: str,
        max_steps: int,
        use_plan: bool,
        run_mode: str,
    ) -> None:
        vision_model = self._resolve_model(model)
        self._model = vision_model

        # Browser + login guard FIRST: the planner must know the real page, and
        # a signed-out session must fail fast instead of after a full plan.
        browser = self._ensure_browser()
        if hasattr(browser, "url") and browser.url() and not is_allowed_url(browser.url()):
            # Never drive a non-Onshape page; navigate home first.
            from agent.browser import DEFAULT_START_URL

            browser.goto(DEFAULT_START_URL)

        # Refuse to drive the sign-in page (typed "your_username" garbage once).
        try:
            page_url, page_title = browser.url(), browser.title()
        except Exception:
            page_url, page_title = "", ""
        if needs_login(page_url, page_title):
            # The user just clicked Run while the browser shows the sign-in
            # page: wait for them to sign in (status says so) instead of
            # failing the whole run outright.
            if not self._await_login(browser):
                self._error = NEEDS_LOGIN_DETAIL
                self._status = "Error"
                self._detail = NEEDS_LOGIN_DETAIL
                return
            try:
                browser.goto(DEFAULT_START_URL)
                page_url, page_title = browser.url(), browser.title()
            except Exception:
                page_url, page_title = "", ""

        if use_plan:
            self._status = "Planning"
            self._detail = f"Planning: {goal}"
            try:
                plan = self.make_plan(
                    goal,
                    planner_model=planner_model,
                    page_hint=f"{page_title} ({page_url})",
                    memory_block=(
                        self.memory.prompt_block(goal) if run_mode == "learn" else ""
                    ),
                )
                self._plan = list(plan or [])
                self._plan_i = 0
                self._pending_appeared = None
                self._mode = "plan"
            except (OllamaError, PolicyError) as exc:
                # Planner unavailable — fall back to pure vision stepping.
                self._detail = f"No plan ({str(exc)[:80]}); running freeform vision"
                self._mode = "vision"
                self._plan = []

        memory_block = self.memory.prompt_block(goal) if run_mode == "learn" else ""
        consecutive_failures = 0
        # Steps the current plan step has consumed; an author-free model step
        # (canvas clicks/dimensions) that never advances must eventually be
        # skipped — the live run burned 70 of 80 steps on ONE dimension step.
        plan_step_started = 0
        last_url = ""
        last_ws = ""
        prev_sig: list[float] = []
        static_steps = 0

        while not self._stop.is_set() and self._step < max_steps:
            plan_line = ""
            if self._mode == "plan" and self._plan:
                if self._plan_i >= len(self._plan):
                    plan_line = "Plan complete — verify the result, then stop if it looks right."
                else:
                    item = self._plan[self._plan_i]
                    plan_line = (
                        f"PLAN STEP {self._plan_i + 1}/{len(self._plan)}: {item.get('step')}"
                        + (f" (expect: {item.get('expect')})" if item.get("expect") else "")
                    )

            self._status = "Running"
            self._detail = plan_line or f"Looking at the screen ({self._step + 1}/{max_steps})"

            b64, obs = self._observe()
            url_now = str(obs.get("url") or "")
            if needs_login(obs.get("url"), obs.get("title")):
                # Session expired mid-run: wait for the user to sign in again
                # and re-observe. (continue skips the step counter on purpose
                # — waiting for a human is not an agent step.)
                if not self._await_login(browser):
                    self._error = NEEDS_LOGIN_DETAIL
                    self._status = "Error"
                    self._detail = NEEDS_LOGIN_DETAIL
                    return
                continue
            if _doc_workspace_url(url_now):
                last_ws = url_now
            elif (
                self._mode == "plan"
                and self._plan_i > 0
                and last_ws
                and not _doc_workspace_url(url_now)
            ):
                # Stray click (Onshape logo, browser back...) kicked us out of
                # the document mid-plan — return to it before acting.
                try:
                    self._ensure_browser().goto(last_ws)
                    b64, obs = self._observe()
                    url_now = str(obs.get("url") or "")
                    if _doc_workspace_url(url_now):
                        last_url = url_now
                        self._detail = f"Recovered: returned to the document ({url_now[:80]})"
                except BrowserError:
                    pass

            # Deadlock breaker input: count consecutive frames that LOOK the
            # same. Byte equality was too strict — hover ripples/JPEG noise
            # made the screen "change" every step, so the auto-Esc never fired
            # while a modal swallowed every click (the run that burned 140 of
            # 160 steps clicking a plane it could not select).
            sig = _frame_signature(b64)
            if sig and prev_sig and _signature_diff(sig, prev_sig) <= _STATIC_DIFF_MAX:
                static_steps += 1
            else:
                static_steps = 0
                self._unstick_cycles = 0  # real change: the escalation ladder restarts
            prev_sig = sig
            # Frame BEFORE this step's action: if the plan expects an overlay
            # (menu/dialog), the NEXT frame verifies it really appeared even
            # when the vision model never narrates it (a 3B model kept saying
            # "Sketch button" while the Create menu was open on screen).
            frame_sig = sig
            if self._pending_appeared is not None:
                pending_i, pending_sig = self._pending_appeared
                self._pending_appeared = None
                if (
                    pending_i == self._plan_i
                    and pending_sig
                    and sig
                    and _signature_diff(sig, pending_sig) >= _VISIBLE_CHANGE_MIN
                ):
                    self._plan_i += 1
            user_text = self._user_text(goal, obs, memory_block, plan_line)
            # Authored recipe step: execute its acts DIRECTLY (target clicks the
            # driver resolves against real DOM labels), no vision model in the
            # loop — the model was the failure source. If all acts fail, fall
            # through to the model for one recovery attempt.
            authored = (
                item.get("acts")
                if self._mode == "plan" and self._plan and self._plan_i < len(self._plan) and isinstance(self._plan[self._plan_i].get("acts"), list)
                else None
            )
            if authored:
                # 'only_if': skip the (whole) step when its precondition is
                # already satisfied — e.g. the open-document act is pointless
                # (and HARMFUL: it would dblclick whatever is in the card
                # position, likely the canvas) when we are ALREADY inside a
                # Part Studio workspace.
                if item.get("only_if") == "documents_list" and _doc_workspace_url(url_now):
                    self._plan_i += 1
                    self._detail = "Authored recipe: already inside a document — skipped open step"
                    continue
                self._status = "Running"
                self._detail = f"Authored recipe: {item.get('step')}"
                failed = False
                for act in authored:
                    if self._stop.is_set():
                        return
                    a = dict(act)
                    if a.get("origin"):
                        # driver.execute(origin=True) RESOLVES AND CLICKS in
                        # one action — passing its result back as a pixel
                        # click double-placed the corner (live bug: rectangle
                        # started 170 px away from the origin).
                        a = {"action": "click", "origin": True, "dx": int(a.get("dx") or 0), "dy": int(a.get("dy") or 0), "reason": a.get("reason") or ""}
                    try:
                        result = self._execute(a)
                    except BrowserError as exc:
                        self._error = str(exc)[:200]
                        self._detail = self._error
                        break
                    self._last_action = a
                    self._last_reason = str(a.get("reason") or "")[:120]
                    self._step += 1
                    self._log_step(a, obs)
                    self._save_debug_shot(b64, self._step)
                    if isinstance(result, dict) and result.get("done") is False:
                        failed = True
                        self._detail = f"Authored step '{item.get('step')}' act failed: {str(result.get('error') or '')[:120]}"
                        break
                    time.sleep(0.5)
                if not failed:
                    self._plan_i += 1
                    self._pending_appeared = None
                    time.sleep(0.3)
                    continue
                # fall through to the model for recovery below
            try:
                raw = chat_vision_json(
                    vision_model,
                    system=SYSTEM_PROMPT,
                    user_text=user_text,
                    images_b64=[b64],
                    num_predict=260,
                )
                action = parse_action(raw)
                consecutive_failures = 0
            except (OllamaError, PolicyError) as exc:
                consecutive_failures += 1
                self._error = str(exc)[:200]
                self._detail = self._error
                if consecutive_failures >= 5:
                    # Model is unreachable or consistently broken — abort, don't spin.
                    self._status = "Error"
                    break
                time.sleep(0.6)
                continue

            # --- garbage-output gate -------------------------------------
            # Live failure: the model degenerated into clicks at (0,0)/(0,10)
            # (the corner, harmless-but-random) and repeated 'type' spam. A
            # real Onshape target is NEVER in the top-left 12x12 corner —
            # that whole strip is the browser/OS chrome in the screenshot.
            # Rejecting those actions costs one step; executing them cost a
            # hundred. Also clamp impossibly out-of-screen coords instead of
            # letting driver clamping turn them into corner clicks.
            action, rejected = _sanitize_action(action, self._viewport_wh())
            if rejected:
                self._last_action = action
                self._last_reason = rejected
                self._step += 1
                self._log_step({**action, "reason": rejected}, obs)
                self._detail = f"Model emitted invalid action — {rejected}"
                time.sleep(0.2)
                continue

            # Anti-repeat: identical signature 3x in a row -> inject a nudge.
            sig = action_signature(action)
            if sig:
                self._recent_signatures.append(sig)
                self._recent_signatures = self._recent_signatures[-40:]
                if len(self._recent_signatures) >= 3 and self._recent_signatures[-3:] == [sig] * 3:
                    action["reason"] = (action.get("reason") or "") + " [blocked repeat]"
                    action = self._break_repeat(action, sig, self._viewport_wh())

            # Models often narrate "double-click ..." while emitting a plain
            # click; Onshape cards only open on a real dblclick — honor intent.
            reason_l_pre = str(action.get("reason") or "").lower()
            if action.get("action") == "click" and (
                "double-click" in reason_l_pre or "double click" in reason_l_pre
            ):
                action["action"] = "dblclick"

            if static_steps >= 8 and action.get("action") != "stop":
                static_steps = 0
                self._unstick_cycles += 1
                # Escalation ladder (tuned after live run: reload was firing
                # every ~15 steps and blinking the whole session away):
                #   cycle 1: Esc (closes stray menu/dialog)
                #   cycle 2: reload — ONCE per run, dead-handlers fix
                #   cycle 3+: STOP waiting — a static screen is not progress
                #             and a mid-plan wait-spam burned 50 of 80 steps
                #             (live: the run did nothing for its final 55
                #             steps after the viewport clicked itself into a
                #             zoomed-out corner). Hand control back to the
                #             model with a blunt instruction instead.
                if self._unstick_cycles == 1:
                    action = {
                        "action": "key",
                        "keys": ["esc"],
                        "reason": "unstick: screen unchanged for 8 steps",
                    }
                elif self._unstick_cycles == 2 and self._reloads_done < _MAX_PAGE_RELOADS:
                    self._reloads_done += 1
                    action = {
                        "action": "hotkey",
                        "keys": ["ctrl", "r"],
                        "reason": "unstick: reload once after Esc made no change",
                    }
                else:
                    # Repeat-driver for a mid-plan stuck model: RE-EMIT the
                    # current plan step's next authored act instead of
                    # waiting forever. A pure wait-loop can never recover.
                    action = {
                        "action": "wait",
                        "seconds": 1.2,
                        "reason": "unstick: screen static — re-observe, then retry the current plan step with a DIFFERENT approach",
                    }

            try:
                result = self._execute(action)
            except BrowserError as exc:
                self._error = str(exc)[:200]
                self._detail = self._error
                break

            self._last_action = action
            self._last_reason = str(action.get("reason") or "")[:120]
            self._step += 1
            self._log_step(action, obs)
            self._save_debug_shot(b64, self._step)

            # Stuck-plan-step breaker: a single plan step may consume at most
            # _MAX_STEPS_PER_PLAN_MODEL steps of model-driven (non-authored)
            # actions. The live run spent 70 of 80 steps on one dimension
            # step and never reached Extrude. Skipping costs nothing when the
            # step was actually done (the plan advances on 'step done');
            # it unblocks the run when the model simply cannot click a
            # sketch edge that frame.
            if self._mode == "plan" and self._plan and self._plan_i < len(self._plan):
                if plan_step_started == 0:
                    plan_step_started = self._step
                elif self._step - plan_step_started > 12:
                    self._plan_i += 1
                    plan_step_started = self._step
                    self._detail = "Plan step stuck — skipping ahead (model could not complete it)"

            if action.get("action") == "stop" or (isinstance(result, dict) and result.get("stop")):
                self._detail = f"Goal reported complete: {action.get('reason') or goal}"
                if self._mode == "plan":
                    self._plan_i = len(self._plan)
                break

            # Advance the plan when the model's reason says the step worked.
            if self._mode == "plan" and self._plan and self._plan_i < len(self._plan):
                reason_l = self._last_reason.lower()
                expect_l = str(self._plan[self._plan_i].get("expect") or "").lower()
                step_l = str(self._plan[self._plan_i].get("step") or "").lower()
                entered_doc = _doc_workspace_url(url_now) and not _doc_workspace_url(last_url)
                exp_words = {w for w in re.findall(r"[a-z]+", expect_l) if len(w) > 3}
                overlap = len(exp_words & set(re.findall(r"[a-z]+", reason_l))) >= 2
                if (
                    (expect_l and expect_l[:24] in reason_l)
                    or "step done" in reason_l
                    or overlap
                ):
                    self._plan_i += 1
                    plan_step_started = self._step
                elif action.get("action") in {"wait"} and "next step" in reason_l:
                    self._plan_i += 1
                    plan_step_started = self._step
                elif entered_doc and "open" in (expect_l + step_l):
                    # Deterministic outcome: the list -> workspace transition
                    # means the document opened, whatever the model said.
                    self._plan_i += 1
                    plan_step_started = self._step
                elif any(w in expect_l for w in ("menu", "dropdown", "dialog", "overlay")):
                    # The expect says an overlay APPEARS. Defer to the next
                    # frame: a large pixel change right after this action
                    # proves it opened, so a hallucinating model can't stall
                    # the plan forever with reasons that never match.
                    self._pending_appeared = (self._plan_i, frame_sig)
            last_url = url_now

            # Let Onshape actually process + redraw the action before we
            # observe again. 0.12s was fast enough to race the SPA: the next
            # screenshot showed the PRE-action screen, the model saw "nothing
            # happened" and started its panic loop (spurious Esc/reloads).
            time.sleep(0.45)

        if not self._stop.is_set() and self._step >= max_steps:
            self._detail = f"Step budget exhausted ({max_steps})"
        if self._stop.is_set() and self._status != "Error":
            self._status = "Stopped"
            self._detail = "Stopped by user"

    def _break_repeat(self, action: dict[str, Any], sig: str, vp: tuple[int, int]) -> dict[str, Any]:
        """Nudge a stuck model WITHOUT corrupting coordinates.

        The old nudge displaced clicks ±24 px EVERY time the signature
        repeated — against a degenerate (0,10) click that just moved the
        corner-click around for 40 steps. Now: clicks are sent for a
        mid-viewport refocus only after a key/hotkey repeat, and a repeated
        'type' is turned into a real re-observe pause with a hint, not
        another blind type.
        """
        kind = action.get("action")
        if kind == "wait":
            return {"action": "key", "keys": ["escape"], "reason": "break repeat: close menu"}
        if kind == "type":
            # Repeated identical 'type' means the tool never activated (no
            # sketch open / nothing selected). Pause + explicit hint beat
            # typing the same string a 4th time.
            return {
                "action": "wait",
                "seconds": 0.6,
                "reason": "break repeat: same command re-sent with no effect — search for it or set the precondition first",
            }
        if kind in {"click", "dblclick"}:
            # DON'T displace coordinates (that created the random-walk).
            # Re-run the SAME click once — if it truly is the right target,
            # the repetition may have been a registration race; the signature
            # guard elsewhere prevents an endless loop of them.
            action["reason"] = "retry same click once (repeat guard)"
            return action
        if kind in {"key", "hotkey"}:
            vw, vh = vp
            return {
                "action": "click",
                "x": int(vw * 0.45),
                "y": int(vh * 0.5),
                "reason": "break repeat: refocus viewport center",
            }
        return action

    def _log_step(self, action: dict[str, Any], obs: dict[str, Any]) -> None:
        """Append one JSONL row per step so stuck runs can be diagnosed later."""
        try:
            row = json.dumps(
                {
                    "t": round(time.time(), 1),
                    "step": self._step,
                    "plan_i": self._plan_i,
                    "url": str(obs.get("url") or "")[:200],
                    "action": action.get("action"),
                    "xy": [action.get("x"), action.get("y")],
                    "keys": action.get("keys"),
                    "text": str(action.get("text") or "")[:60],
                    "reason": str(action.get("reason") or "")[:200],
                },
                ensure_ascii=False,
            )
            with (self.root / "run_log.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(row + "\n")
        except OSError:
            pass  # diagnostics only — never break the run

    def _save_debug_shot(self, b64: str, step: int) -> None:
        """Keep the last frames the model actually saw (post-run diagnosis)."""
        try:
            target_dir = self.root / "debug_shots"
            target_dir.mkdir(exist_ok=True)
            (target_dir / f"step_{int(step):04d}.jpg").write_bytes(base64.b64decode(b64))
            prune_old_files(target_dir, "*.jpg", keep=60)
        except Exception:
            pass  # diagnostics only — never break the run

    def _save_screenshot(self, b64: str, step: int) -> str:
        try:
            path = screenshots_dir(self.root) / f"shot_{int(time.time())}_{step:04d}.jpg"
            path.write_bytes(base64.b64decode(b64))
            prune_old_files(screenshots_dir(self.root), "*.jpg", keep=1200)
            return str(path)
        except OSError:
            return ""

    # ------------------------------------------------------------ recording

    def start_record(self, *, goal: str = "", interval_ms: int = 400) -> None:
        """Record the agent's own screen+actions (browser-based, no keyloggers)."""
        if self.busy():
            raise RuntimeError("Stop the current run/record/learn first")
        self._stop.clear()
        self._recording = True
        self._record_goal = _clip(goal, 400)
        self._record_actions = []
        self._record_shots = []
        self._status = "Recording"
        self._mode = "record"
        self._detail = "Recording the browser tab. Press Stop Recording to finish."
        interval = max(100, min(5000, int(interval_ms or 400)))

        def _worker() -> None:
            try:
                browser = self._ensure_browser()
                try:
                    browser.start_input_capture()
                except BrowserError:
                    pass  # screenshots-only recording still works
                n = 0
                while not self._stop.is_set():
                    try:
                        for captured in browser.take_captured_actions():
                            self._record_actions.append(captured)
                    except Exception:
                        pass
                    try:
                        png = browser.screenshot_png()
                        b64 = encode_jpeg_b64(png, max_side=672, quality=70)
                        self._record_shots.append(self._save_screenshot(b64, n))
                        n += 1
                    except Exception:
                        pass
                    self._stop.wait(interval / 1000.0)
            except Exception as exc:
                # Without this, a browser launch failure left _recording=True
                # forever and every later start_run raised "busy" until restart.
                self._error = str(exc)[:300]
                self._status = "Error"
                self._detail = self._error
                self._recording = False

        self._thread = threading.Thread(target=_worker, name="agent-record", daemon=True)
        self._thread.start()

    def stop_record(self, *, goal: str = "", success: bool = True) -> dict[str, Any]:
        if not self._recording:
            raise RuntimeError("Not recording")
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=8.0)
        self._recording = False
        # Pull any actions that arrived after the worker's last tick.
        try:
            for captured in self._ensure_browser().take_captured_actions():
                self._record_actions.append(captured)
        except Exception:
            pass
        self._status = "Idle"
        self._mode = "idle"
        goal_text = _clip(goal, 400) or self._record_goal or "recorded demo"
        episode = {
            "goal": goal_text,
            "actions": list(self._record_actions),
            "screenshots": [s for s in self._record_shots if s],
            "source": "record",
            "success": bool(success),
        }
        self.memory.add_episode(episode)
        if episode["actions"]:
            self.memory.add_skill(goal=goal_text, actions=episode["actions"], source="record")
        self._detail = (
            f"Saved recording '{goal_text}' "
            f"({len(episode['actions'])} actions, {len(episode['screenshots'])} screenshots)"
        )
        self._record_actions = []
        self._record_shots = []
        self._stop.clear()
        return {"ok": True, "detail": self._detail}

    # ------------------------------------------------------------ replay

    def replay_last(self, *, goal: str = "") -> None:
        """Replay the last recorded episode's actions in a fresh browser tab."""
        episode = self.memory.last_episode()
        if not episode:
            raise RuntimeError("No recorded episode to replay")
        actions = [a for a in (episode.get("actions") or []) if isinstance(a, dict)]
        if not actions:
            raise RuntimeError("Last episode has no actions")
        if self.busy():
            raise RuntimeError("Stop the current run/record/learn first")
        self._stop.clear()
        self._running = True
        self._mode = "vision"
        self._status = "Replaying"
        self._goal = _clip(goal, 400) or str(episode.get("goal") or "replay")
        self._step = 0
        self._max_steps = len(actions)

        def _worker() -> None:
            try:
                self._ensure_browser()
                for i, action in enumerate(actions):
                    if self._stop.is_set():
                        break
                    self._step = i + 1
                    self._detail = f"Replay {i + 1}/{len(actions)}: {action.get('action')}"
                    try:
                        result = self._execute(dict(action))
                    except BrowserError as exc:
                        self._error = str(exc)[:200]
                        break
                    if isinstance(result, dict) and result.get("stop"):
                        break
                    time.sleep(0.15)
                if self._status != "Error":
                    self._status = "Idle"
                    self._detail = f"Replayed {self._step} action(s)"
            except Exception as exc:
                self._error = str(exc)[:300]
                self._status = "Error"
                self._detail = self._error
            finally:
                self._running = False
                self._mode = "idle" if not self._learning_video else "learn"
                # Leave the browser open so the replayed result stays visible.
                self._note_browser_left_open()

        self._thread = threading.Thread(target=_worker, name="agent-replay", daemon=True)
        self._thread.start()

    # -------------------------------------------------------- video learning

    def learn_from_video(
        self,
        *,
        url: str = "",
        path: str = "",
        goal: str = "",
        model: str = "",
        max_minutes: float = 6.0,
    ) -> None:
        """Background: extract Onshape concepts from YouTube/local video into memory."""
        if self.busy():
            raise RuntimeError("Stop recording/agent before learning from video")
        goal = _clip(goal, 400).strip()
        if not (url or "").strip() and not (path or "").strip():
            raise ValueError("Paste a YouTube URL or choose a local video file")
        if (url or "").strip():
            from agent.video_teacher import _clean_video_url, _is_youtube_url

            # Validate the URL before the background thread starts so HTTP gets 400.
            if not _is_youtube_url(_clean_video_url(url)):
                raise ValueError("Provide a YouTube URL or a local video file path")
        if (path or "").strip():
            from agent.video_teacher import ensure_video_path_in_dataset

            # Reject arbitrary on-disk paths before the background thread starts.
            ensure_video_path_in_dataset(path, self.root)

        self._learning_video = True
        self._stop.clear()  # a prior stopped run must not cancel this lesson
        self._error = ""
        self._status = "Learning"
        self._mode = "learn"
        self._learn_percent = 2.0
        self._learn_phase = "starting"
        self._learn_history = []
        self._note_learn_progress("Starting video teacher…", percent=2.0, phase="starting")

        def _run() -> None:
            from agent.video_teacher import learn_concepts_from_video

            def on_progress(msg: str) -> None:
                self._note_learn_progress(msg)

            try:
                chosen_model = model or self._model or ""
                if not chosen_model:
                    from agent.ollama import list_ollama_models

                    try:
                        models = list_ollama_models(timeout=3.0)
                    except OllamaError:
                        models = []
                    chosen_model = prefer_vision_model(models) if models else "qwen2.5vl"
                result = learn_concepts_from_video(
                    url=url,
                    path=path,
                    goal=goal,
                    model=chosen_model,
                    max_minutes=max_minutes,
                    dataset_root=self.root,
                    on_progress=on_progress,
                    cancelled=self._stop.is_set,
                )
                concepts = result.get("concepts") or []
                self._note_learn_progress("Saving concepts into memory…", percent=96.0, phase="saving")
                written = 0
                for c in concepts:
                    if not isinstance(c, dict):
                        continue
                    row = self.memory.add_concept(
                        name=str(c.get("name") or ""),
                        summary=str(c.get("summary") or ""),
                        when_to_use=str(c.get("when_to_use") or ""),
                        preconditions=[str(p) for p in (c.get("preconditions") or [])],
                        steps=[s for s in (c.get("steps") or []) if isinstance(s, dict)],
                        source="video",
                        url=str(result.get("video") or url)[:512],
                        t_start=float(c.get("t_start") or 0.0),
                    )
                    if row is not None:
                        written += 1
                names = [str(c.get("name") or "") for c in concepts if isinstance(c, dict)]
                names = [n for n in names if n][:12]
                self._learn_phase = "done"
                self._learn_percent = 100.0
                self._status = "Idle"
                self._mode = "idle"
                self._detail = (
                    f"Saved {written} concept(s) from video"
                    + (f": {', '.join(names)}" if names else "")
                    + ". Type a Goal, then Run Agent."
                )
                self._learn_history.append(self._detail[:240])
            except StopIteration as exc:
                # Clean stop requested from the Stop button, not a failure.
                self._status = "Stopped"
                self._detail = str(exc) or "Video learning stopped by user"
                self._mode = "idle"
                self._learn_phase = "stopped"
                self._learn_history.append(self._detail[:240])
            except Exception as exc:
                self._error = str(exc)[:300]
                self._learn_phase = "error"
                self._status = "Error"
                self._detail = self._error
            finally:
                self._learning_video = False
                if not self.busy():
                    self._stop.clear()

        self._video_thread = threading.Thread(target=_run, name="video-teacher", daemon=True)
        self._video_thread.start()

    def _note_learn_progress(self, msg: str, *, percent: float | None = None, phase: str = "") -> None:
        if phase:
            self._learn_phase = phase
        if percent is not None:
            self._learn_percent = float(percent)
        self._detail = msg
        if not self._learn_history or self._learn_history[-1] != msg[:240]:
            self._learn_history.append(msg[:240])
            self._learn_history = self._learn_history[-40:]

    # ------------------------------------------------------------ skills

    def matching_skills(self, goal: str) -> list[dict[str, Any]]:
        return self.memory.matching_skills(goal)

    def goal_matches(self, a: str, b: str) -> bool:
        return goals_match(a, b)


def create_loop_for_tests(dataset_root: Path | None = None) -> AgentLoop:
    """Loop wired to a FakeBrowser for deterministic tests."""
    return AgentLoop(dataset_root=dataset_root, browser=FakeBrowser())
