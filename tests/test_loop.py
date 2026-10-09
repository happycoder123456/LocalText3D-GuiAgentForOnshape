from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from agent.browser import FakeBrowser
from agent.loop import AgentLoop, BrowserThreadProxy, create_loop_for_tests


def _wait_until(predicate, timeout: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _fake_vision_response(action: dict) -> str:
    return json.dumps(action)


class ThreadGuardBrowser(FakeBrowser):
    """Mimics Playwright's thread affinity.

    Its sync API is bound to the thread that started the driver; a call from
    any other thread raises greenlet's "cannot switch to a different thread".
    """

    def _check(self) -> None:
        if getattr(self, "_tid", None) != threading.get_ident():
            raise RuntimeError("cannot switch to a different thread")

    def start(self):
        super().start()
        self._tid = threading.get_ident()

    def url(self):
        self._check()
        return super().url()

    def title(self):
        self._check()
        return super().title()

    def goto(self, url):
        self._check()
        return super().goto(url)

    def screenshot_png(self):
        self._check()
        return super().screenshot_png()

    def execute(self, action, *, stop_check=None):
        self._check()
        return super().execute(action, stop_check=stop_check)


class BrowserThreadProxyTests(unittest.TestCase):
    def test_calls_run_on_one_owner_thread_across_caller_threads(self):
        seen: list[int] = []

        class Driver:
            def ping(self):
                seen.append(threading.get_ident())
                return "pong"

        proxy = BrowserThreadProxy(lambda: Driver())
        results: list[str] = []

        def caller():
            results.append(proxy.ping())

        for _ in range(3):  # each caller thread dies after its call
            t = threading.Thread(target=caller)
            t.start()
            t.join(5)
        self.assertEqual(results, ["pong"] * 3)
        self.assertEqual(len(set(seen)), 1)

    def test_call_if_idle_returns_default_instead_of_blocking(self):
        gate = threading.Event()

        class Driver:
            def slow(self):
                gate.wait(5)
                return "done"

        proxy = BrowserThreadProxy(lambda: Driver())
        out: list[str] = []
        t = threading.Thread(target=lambda: out.append(proxy.slow()))
        t.start()
        deadline = time.monotonic() + 3
        while proxy._pending == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIsNone(proxy.call_if_idle(lambda d: "x", default=None))
        gate.set()
        t.join(5)
        self.assertEqual(out, ["done"])
        self.assertEqual(proxy.call_if_idle(lambda d: "x"), "x")

    def test_exceptions_propagate_to_the_caller(self):
        class Driver:
            def boom(self):
                raise ValueError("kaboom")

        proxy = BrowserThreadProxy(lambda: Driver())
        with self.assertRaises(ValueError):
            proxy.boom()


class LoopLifecycleTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.loop = create_loop_for_tests(self.root)

    def tearDown(self):
        self.loop.stop(timeout=5.0)
        self._tmp.cleanup()

    def test_status_shape(self):
        st = self.loop.status()
        for key in ("running", "recording", "learning_video", "status", "goal", "step", "memory"):
            self.assertIn(key, st)
        self.assertFalse(self.loop.busy())

    def test_start_requires_goal(self):
        with self.assertRaises(ValueError):
            self.loop.start_run(goal="   ")

    def test_run_refuses_signin_page(self):
        # Must stop with NEEDS_LOGIN instead of driving the login form — and
        # the guard fires before planning, so no LLM work happens either.
        # LOGIN_WAIT_SEC=0 disables the sign-in wait (covered separately), so
        # this keeps asserting the fail-fast terminal behavior.
        self.loop._browser_factory.goto("https://cad.onshape.com/signin")
        self.loop.LOGIN_WAIT_SEC = 0.0
        with mock.patch("agent.loop.AgentLoop.make_plan") as mk, mock.patch(
            "agent.loop.chat_vision_json", side_effect=AssertionError("must not drive sign-in page")
        ) as vision:
            self.loop.start_run(goal="make a cube", model="test-model", use_plan=True, max_steps=5)
            self.assertTrue(_wait_until(lambda: not self.loop.busy()))
        vision.assert_not_called()
        mk.assert_not_called()
        st = self.loop.status()
        self.assertEqual(st["status"], "Error")
        self.assertIn("NEEDS_LOGIN", st["error"])
        self.assertEqual(self.loop._browser.actions, [])

    def test_await_login_times_out_on_signin_page(self):
        browser = self.loop._browser_factory
        browser.goto("https://cad.onshape.com/signin")
        self.loop.LOGIN_WAIT_SEC = 0.0
        self.assertFalse(self.loop._await_login(browser))

    def test_await_login_returns_once_signed_in(self):
        browser = self.loop._browser_factory
        browser.goto("https://cad.onshape.com/signin")
        self.loop.LOGIN_WAIT_SEC = 30.0
        threading.Timer(
            0.2,
            lambda: browser.goto("https://cad.onshape.com/documents"),
        ).start()
        self.assertTrue(self.loop._await_login(browser))
        self.assertEqual(self.loop.status()["status"], "Needs login")

    def test_run_waits_for_signin_then_continues(self):
        # Regression: a sign-in page right after Run must not kill the run.
        # The loop waits for the user to sign in, then plans and steps.
        browser = self.loop._browser_factory
        browser.goto("https://cad.onshape.com/signin")
        self.loop.LOGIN_WAIT_SEC = 30.0
        threading.Timer(
            0.4,
            lambda: browser.goto("https://cad.onshape.com/documents"),
        ).start()
        plan = [{"step": "click Sketch", "expect": "sketch mode activates"}]
        with mock.patch("agent.loop.AgentLoop.make_plan", return_value=plan) as mk, mock.patch(
            "agent.loop.chat_vision_json",
            return_value=_fake_vision_response({"action": "stop", "reason": "done"}),
        ):
            self.loop.start_run(goal="make a cube", model="test-model", use_plan=True, max_steps=5)
            self.assertTrue(_wait_until(lambda: not self.loop.busy(), timeout=15))
        mk.assert_called_once()
        st = self.loop.status()
        self.assertNotIn("NEEDS_LOGIN", str(st.get("error") or ""))

    def test_second_run_works_after_first_run_thread_exits(self):
        # Each run gets its own thread while the driver outlives it, and
        # Playwright's sync API refuses calls from a thread other than the one
        # that created it — without a permanent owner thread the 2nd run died
        # with "cannot switch to a different thread (which happens to have
        # exited)". Both runs must complete cleanly.
        loop = AgentLoop(dataset_root=self.root, driver_factory=ThreadGuardBrowser)
        plan = [{"step": "click Sketch", "expect": "sketch mode activates"}]
        try:
            for attempt in (1, 2):
                with mock.patch("agent.loop.AgentLoop.make_plan", return_value=plan), mock.patch(
                    "agent.loop.chat_vision_json",
                    return_value=_fake_vision_response({"action": "stop", "reason": "done"}),
                ):
                    loop.start_run(goal="make a cube", model="test-model", use_plan=True, max_steps=5)
                    self.assertTrue(
                        _wait_until(lambda: not loop.busy()), f"run {attempt} never finished"
                    )
                st = loop.status()
                self.assertEqual(
                    str(st.get("error") or ""), "", f"run {attempt} failed: {st.get('error')}"
                )
                self.assertGreaterEqual(int(st.get("step") or 0), 1, f"run {attempt} did nothing")
        finally:
            loop.close_browser()
            loop.stop(timeout=5.0)

    def test_plan_advances_on_step_done_marker(self):
        plan = [
            {"step": "click Sketch", "expect": "sketch mode activates"},
            {"step": "pick a plane", "expect": "plane picker shown"},
        ]
        responses = iter(
            [
                _fake_vision_response({"action": "click", "x": 10, "y": 20, "reason": "clicked Sketch, step done"}),
                _fake_vision_response({"action": "stop", "reason": "done"}),
            ]
        )
        seen: list[int] = []

        def _resp(*args, **kwargs):
            seen.append(self.loop.status()["plan_index"])
            return next(responses)

        with mock.patch("agent.loop.AgentLoop.make_plan", return_value=plan), mock.patch(
            "agent.loop.chat_vision_json", side_effect=_resp
        ):
            self.loop.start_run(goal="a part", model="test-model", use_plan=True, max_steps=10)
            self.assertTrue(_wait_until(lambda: not self.loop.busy()))
        # The "step done" marker advanced the plan before the second call.
        self.assertEqual(seen, [0, 1])

    def test_run_log_written(self):
        with mock.patch(
            "agent.loop.chat_vision_json",
            side_effect=lambda *a, **k: _fake_vision_response({"action": "stop", "reason": "goal done"}),
        ):
            self.loop.start_run(goal="make a cube", model="test-model", use_plan=False, max_steps=3)
            self.assertTrue(_wait_until(lambda: not self.loop.busy()))
        log = (self.root / "run_log.jsonl").read_text(encoding="utf-8")
        rows = [json.loads(line) for line in log.splitlines() if line.strip()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["action"], "stop")
        self.assertIn("/documents", rows[0]["url"])
        shots = list((self.root / "debug_shots").glob("step_*.jpg"))
        self.assertEqual(len(shots), 1)

    def test_make_plan_includes_page_hint(self):
        raw = json.dumps({"plan": [{"step": "open the most recent document", "expect": ""}]})
        with mock.patch("agent.loop.chat_text_json", return_value=raw) as chat:
            plan = self.loop.make_plan(
                "make a cube",
                planner_model="test-model",
                page_hint="Owned by me | Documents (https://cad.onshape.com/documents)",
            )
        self.assertEqual(len(plan), 1)
        user = chat.call_args.kwargs["user_text"]
        self.assertIn("Owned by me | Documents", user)
        self.assertIn("documents list", user)

    def test_make_plan_does_not_clobber_running_plan(self):
        # A plan PREVIEW while a run is active must not reset the running
        # agent's plan pointer (used to jump the plan back to step 1).
        self.loop._plan = [{"step": "running step", "expect": ""}]
        self.loop._plan_i = 1
        raw = json.dumps({"plan": [{"step": "preview step", "expect": ""}]})
        with mock.patch("agent.loop.chat_text_json", return_value=raw):
            plan = self.loop.make_plan("another goal", planner_model="test-model")
        self.assertEqual(plan[0]["step"], "preview step")
        self.assertEqual(self.loop._plan[0]["step"], "running step")
        self.assertEqual(self.loop._plan_i, 1)

    def test_record_worker_failure_clears_recording_flag(self):
        # A browser launch failure during record used to leave _recording=True
        # forever, so every later run/record/learn raised "busy" until restart.
        class BrokenBrowser(FakeBrowser):
            def start(self):
                raise RuntimeError("cannot launch browser")

        loop = AgentLoop(dataset_root=self.root, browser=BrokenBrowser())
        loop.start_record(goal="demo")
        self.assertTrue(_wait_until(lambda: not loop.busy()))
        st = loop.status()
        self.assertEqual(st["status"], "Error")
        self.assertIn("cannot launch", st["error"])

    def test_status_survives_dead_browser_url(self):
        # GUI polls /status continuously; a closed browser must not 500 it.
        class DeadBrowser(FakeBrowser):
            def url(self):
                raise RuntimeError("browser closed")

        self.loop._browser = DeadBrowser()
        st = self.loop.status()
        self.assertEqual(st["browser_url"], "")
        self.assertIn("status", st)

    def test_run_leaves_browser_open(self):
        # The user's biggest complaint: the browser (their Onshape window)
        # closed the moment a run ended, so the result vanished. When the
        # loop owns the browser it must still stay open after the run.
        loop = AgentLoop(dataset_root=self.root, browser=FakeBrowser())
        loop._ensure_browser()          # start via factory…
        loop._browser_factory = None    # …then hand ownership to the loop
        try:
            with mock.patch(
                "agent.loop.chat_vision_json",
                side_effect=lambda *a, **k: _fake_vision_response(
                    {"action": "stop", "reason": "done"}
                ),
            ):
                loop.start_run(goal="make a cube", model="test-model", use_plan=False, max_steps=3)
                self.assertTrue(_wait_until(lambda: not loop.busy()))
            self.assertIsNotNone(loop._browser)
            self.assertTrue(loop._browser.available, "browser was closed at run end")
            self.assertIn("browser left open", loop.status()["detail"])
        finally:
            loop.close_browser()

    def test_make_plan_marks_workspace_context(self):
        # Live bug: plan said "create document" while ALREADY inside one.
        raw = json.dumps({"plan": [{"step": "sketch a square", "expect": ""}]})
        with mock.patch("agent.loop.chat_text_json", return_value=raw) as chat:
            self.loop.make_plan(
                "make a cube",
                planner_model="test-model",
                page_hint="Part Studio (https://cad.onshape.com/documents/533/tw/x/e/y)",
            )
        user = chat.call_args.kwargs["user_text"]
        self.assertIn("do NOT plan any create-document", user)
        self.assertIn("click a plane", user)

    def test_user_text_flags_stuck_repeats(self):
        # Live bug: 104 of 200 steps were identical no-op repeats; the next
        # prompt must tell the model it is stuck.
        self.loop._recent_signatures = ["type:rectangle", "type:rectangle", "type:rectangle"]
        text = self.loop._user_text("goal", {"title": "t", "url": "u"}, "", "")
        self.assertIn("STUCK", text)
        self.assertIn("Do NOT repeat", text)
        # Different signatures -> no stuck warning.
        self.loop._recent_signatures = ["type:a", "click:1:1", "key:esc"]
        text2 = self.loop._user_text("goal", {"title": "t", "url": "u"}, "", "")
        self.assertNotIn("STUCK", text2)

    def test_make_plan_includes_memory_block(self):
        raw = json.dumps({"plan": [{"step": "sketch a square", "expect": ""}]})
        with mock.patch("agent.loop.chat_text_json", return_value=raw) as chat:
            plan = self.loop.make_plan(
                "make a cube",
                planner_model="test-model",
                memory_block="TECHNIQUE 'line': draw a line — do: s -> \"line\" -> Enter",
            )
        self.assertEqual(len(plan), 1)
        user = chat.call_args.kwargs["user_text"]
        self.assertIn("Known techniques", user)
        self.assertIn("TECHNIQUE 'line'", user)

    def test_click_narrating_double_click_becomes_dblclick(self):
        with mock.patch(
            "agent.loop.chat_vision_json",
            side_effect=lambda *a, **k: _fake_vision_response(
                {
                    "action": "click",
                    "x": 410,
                    "y": 345,
                    "reason": "Double-click the most recent document in the list to open it.",
                }
            ),
        ):
            self.loop.start_run(goal="make a cube", model="test-model", use_plan=False, max_steps=1)
            self.assertTrue(_wait_until(lambda: not self.loop.busy()))
        actions = [a["action"] for a in self.loop._browser.actions]
        self.assertEqual(actions, ["dblclick"])

    def test_plan_advances_when_document_workspace_opens(self):
        plan = [
            {
                "step": "Double-click the most recent document in the list",
                "expect": "The document opens in a Part Studio",
            }
        ]
        responses = iter(
            [
                _fake_vision_response({"action": "click", "x": 410, "y": 345, "reason": "clicking the card"}),
                _fake_vision_response({"action": "wait", "reason": ""}),
                _fake_vision_response({"action": "stop", "reason": "done"}),
            ]
        )
        seen: list[int] = []
        browser = self.loop._browser_factory

        def _resp(*args, **kwargs):
            seen.append(self.loop.status()["plan_index"])
            if len(seen) == 1:
                # Simulate the card opening between the first and second observe.
                browser.current_url = "https://cad.onshape.com/documents/533ff50996f4e88475f3945/tw/abc/e/def"
            return next(responses)

        with mock.patch("agent.loop.AgentLoop.make_plan", return_value=plan), mock.patch(
            "agent.loop.chat_vision_json", side_effect=_resp
        ):
            self.loop.start_run(goal="a part", model="test-model", use_plan=True, max_steps=5)
            self.assertTrue(_wait_until(lambda: not self.loop.busy()))
        # The list->workspace URL transition advanced the plan without any
        # expect-echo from the model (observed at the third call).
        self.assertEqual(seen, [0, 0, 1])

    def test_kicked_out_of_workspace_recovers(self):
        plan = [
            {"step": "Double-click the recent document", "expect": "The document opens"},
            {"step": "Click Sketch", "expect": "sketch mode"},
        ]
        ws = "https://cad.onshape.com/documents/533ff50996f4e88475f3945/tw/abc/e/def"
        browser = self.loop._browser_factory
        calls = {"n": 0}

        def _resp(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                browser.current_url = ws  # the document opens
                return json.dumps({"action": "click", "x": 100, "y": 100, "reason": "double-click the card"})
            if calls["n"] == 2:
                browser.current_url = "https://cad.onshape.com/documents"  # stray click kicked us out
                return json.dumps({"action": "click", "x": 500, "y": 400, "reason": "trying to sketch"})
            return json.dumps({"action": "stop", "reason": "done"})

        with mock.patch("agent.loop.AgentLoop.make_plan", return_value=plan), mock.patch(
            "agent.loop.chat_vision_json", side_effect=_resp
        ):
            self.loop.start_run(goal="x", model="test-model", use_plan=True, max_steps=6)
            self.assertTrue(_wait_until(lambda: not self.loop.busy()))
        # The loop navigated back to the workspace before the final step.
        self.assertEqual(calls["n"], 3)
        self.assertEqual(browser.current_url, ws)

    def test_repeat_click_displaced_near_original(self):
        responses = iter(
            [
                _fake_vision_response({"action": "click", "x": 400, "y": 300, "reason": "a"}),
                _fake_vision_response({"action": "click", "x": 400, "y": 300, "reason": "b"}),
                _fake_vision_response({"action": "click", "x": 400, "y": 300, "reason": "c"}),
                _fake_vision_response({"action": "stop", "reason": "done"}),
            ]
        )
        with mock.patch("agent.loop.chat_vision_json", side_effect=lambda *a, **k: next(responses)):
            self.loop.start_run(goal="stuck", model="test-model", use_plan=False, max_steps=8)
            self.assertTrue(_wait_until(lambda: not self.loop.busy()))
        clicks = [a for a in self.loop._browser.actions if a["action"] in {"click", "dblclick"}]
        self.assertEqual(len(clicks), 3)
        third = clicks[2]
        # Third identical click is nudged — but stays NEAR the original point
        # (a cumulative walk once dragged the cursor into the Onshape logo).
        self.assertNotEqual((third["x"], third["y"]), (400, 300))
        self.assertLessEqual(abs(third["x"] - 400), 70)
        self.assertLessEqual(abs(third["y"] - 300), 50)

    def test_repeated_type_nudged_to_refocus(self):
        responses = iter(
            [
                _fake_vision_response({"action": "type", "text": "rectangle", "reason": "a"}),
                _fake_vision_response({"action": "type", "text": "rectangle", "reason": "b"}),
                _fake_vision_response({"action": "type", "text": "rectangle", "reason": "c"}),
                _fake_vision_response({"action": "stop", "reason": "done"}),
            ]
        )
        with mock.patch("agent.loop.chat_vision_json", side_effect=lambda *a, **k: next(responses)):
            self.loop.start_run(goal="stuck typing", model="test-model", use_plan=False, max_steps=8)
            self.assertTrue(_wait_until(lambda: not self.loop.busy()))
        acts = self.loop._browser.actions
        types = [a for a in acts if a["action"] == "type"]
        # Third identical type must be redirected (no infinite typing loop).
        self.assertEqual(len(types), 2)
        # The nudge pauses instead of pressing keys that fight the model.
        waits = [a for a in acts if a["action"] == "wait" and "break repeat" in str(a.get("reason") or "")]
        self.assertTrue(waits)

    def test_static_screen_injects_escape(self):
        # FakeBrowser renders an identical frame every step (simulated modal
        # swallowing clicks) — the loop must auto-press Esc after 6 steps.
        with mock.patch(
            "agent.loop.chat_vision_json",
            side_effect=lambda *a, **k: _fake_vision_response(
                {"action": "click", "x": 300, "y": 300, "reason": "clicking through the dialog"}
            ),
        ):
            self.loop.start_run(goal="stuck", model="test-model", use_plan=False, max_steps=10)
            self.assertTrue(_wait_until(lambda: not self.loop.busy()))
        unstick = [a for a in self.loop._browser.actions if "unstick" in str(a.get("reason") or "")]
        self.assertTrue(unstick, "expected an auto-Esc after a static screen")
        self.assertEqual(unstick[0]["action"], "key")
        self.assertEqual(unstick[0]["keys"], ["esc"])

    def test_near_identical_frames_still_unstick(self):
        # Real screens shimmer (hover ripples, JPEG noise): byte-equality
        # never fired while a modal swallowed every click — a run burned 140
        # of its 160 steps that way. Near-identical frames must still count
        # as static so the auto-Esc can close the blocking dialog.
        import io

        from PIL import Image

        counter = {"n": 0}

        def noisy_png():
            counter["n"] += 1
            value = 100 + (counter["n"] % 2)  # alternates 100/101 gray
            img = Image.new("RGB", (80, 50), (value, value, value))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

        self.loop._browser_factory.screenshot_png = noisy_png
        with mock.patch(
            "agent.loop.chat_vision_json",
            side_effect=lambda *a, **k: _fake_vision_response(
                {"action": "click", "x": 300, "y": 300, "reason": "clicking through the dialog"}
            ),
        ):
            self.loop.start_run(goal="stuck", model="test-model", use_plan=False, max_steps=10)
            self.assertTrue(_wait_until(lambda: not self.loop.busy()))
        unstick = [a for a in self.loop._browser.actions if "unstick" in str(a.get("reason") or "")]
        self.assertTrue(unstick, "near-identical frames must still trigger the auto-Esc")
        self.assertEqual(unstick[0]["keys"], ["esc"])

    def test_plan_advances_on_expect_word_overlap(self):
        plan = [
            {"step": "click Sketch", "expect": "Sketch mode activates on screen"},
            {"step": "pick a plane", "expect": "plane picker shown"},
        ]
        responses = iter(
            [
                _fake_vision_response({"action": "click", "x": 145, "y": 59, "reason": "sketch mode is active now"}),
                _fake_vision_response({"action": "stop", "reason": "done"}),
            ]
        )
        seen: list[int] = []

        def _resp(*args, **kwargs):
            seen.append(self.loop.status()["plan_index"])
            return next(responses)

        with mock.patch("agent.loop.AgentLoop.make_plan", return_value=plan), mock.patch(
            "agent.loop.chat_vision_json", side_effect=_resp
        ):
            self.loop.start_run(goal="a part", model="test-model", use_plan=True, max_steps=6)
            self.assertTrue(_wait_until(lambda: not self.loop.busy()))
        # Two shared words ("sketch", "mode") advanced the plan between calls.
        self.assertEqual(seen, [0, 1])

    def test_plan_advances_when_overlay_visibly_appears(self):
        # A 3B vision model kept narrating "Sketch button" while the Create
        # menu was open on screen — its reason never matched the expect and
        # the plan stalled at step 1 for all 160 steps. The frame change
        # alone must verify the overlay and advance the plan.
        import io

        from PIL import Image

        plan = [
            {"step": "click Create", "expect": "A dropdown menu appears with options"},
            {"step": "click New document", "expect": "The new document workspace opens"},
        ]
        calls = {"n": 0}

        def frame_png():
            # First frame: plain page. Later frames: the menu overlay is up
            # (a huge grayscale delta, like the 32-95 measured live).
            value = 250 if calls["n"] == 0 else 20
            calls["n"] += 1
            img = Image.new("RGB", (80, 50), (value, value, value))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

        responses = iter(
            [
                _fake_vision_response(
                    {"action": "click", "x": 109, "y": 66, "reason": "pressing the blue button"}
                ),
                _fake_vision_response({"action": "stop", "reason": "done"}),
            ]
        )
        seen: list[int] = []

        def _resp(*args, **kwargs):
            seen.append(self.loop.status()["plan_index"])
            return next(responses)

        self.loop._browser_factory.screenshot_png = frame_png
        with mock.patch("agent.loop.AgentLoop.make_plan", return_value=plan), mock.patch(
            "agent.loop.chat_vision_json", side_effect=_resp
        ):
            self.loop.start_run(goal="a part", model="test-model", use_plan=True, max_steps=6)
            self.assertTrue(_wait_until(lambda: not self.loop.busy()))
        # The reason shared no expect words — the visible overlay did it.
        self.assertEqual(seen, [0, 1])

    def test_plan_does_not_advance_without_screen_change(self):
        # Same overlay expect, but the screen never changes: the menu never
        # opened, so the plan must stay put.
        plan = [
            {"step": "click Create", "expect": "A dropdown menu appears with options"},
            {"step": "click New document", "expect": "The new document workspace opens"},
        ]
        responses = iter(
            [
                _fake_vision_response(
                    {"action": "click", "x": 109, "y": 66, "reason": "pressing the blue button"}
                ),
                _fake_vision_response({"action": "click", "x": 300, "y": 300, "reason": "clicking elsewhere"}),
                _fake_vision_response({"action": "stop", "reason": "done"}),
            ]
        )
        seen: list[int] = []

        def _resp(*args, **kwargs):
            seen.append(self.loop.status()["plan_index"])
            return next(responses)

        with mock.patch("agent.loop.AgentLoop.make_plan", return_value=plan), mock.patch(
            "agent.loop.chat_vision_json", side_effect=_resp
        ):
            self.loop.start_run(goal="a part", model="test-model", use_plan=True, max_steps=6)
            self.assertTrue(_wait_until(lambda: not self.loop.busy()))
        self.assertEqual(seen, [0, 0, 0])

    def test_static_screen_escalates_to_reload_after_failed_esc(self):
        # A page can be alive-but-dead: hover works, every click no-ops
        # (observed live — a run burned all 160 steps that way). Esc alone
        # never cures it; after two failed Esc rounds the loop must reload,
        # and stop reloading after the cap.
        with mock.patch(
            "agent.loop.chat_vision_json",
            side_effect=lambda *a, **k: _fake_vision_response(
                {"action": "click", "x": 300, "y": 300, "reason": "clicking the button"}
            ),
        ):
            self.loop.start_run(goal="zombie page", model="test-model", use_plan=False, max_steps=40)
            self.assertTrue(_wait_until(lambda: not self.loop.busy(), timeout=25.0))
        acts = self.loop._browser.actions
        escs = [a for a in acts if a["action"] == "key" and a.get("keys") == ["esc"]]
        reloads = [
            a
            for a in acts
            if a["action"] == "hotkey" and a.get("keys") == ["ctrl", "r"]
        ]
        # Esc@7, reload@13, Esc@19, reload@25, Esc@31, reload@37 — then cap.
        self.assertTrue(escs, "the first unstick cycle must still press Esc")
        self.assertEqual(len(reloads), 3, f"expected 3 capped reloads, got {len(reloads)}")
        # Reload must come only AFTER a failed Esc round, never first.
        first_esc = next(i for i, a in enumerate(acts) if a["action"] == "key" and a.get("keys") == ["esc"])
        first_reload = next(
            i for i, a in enumerate(acts) if a["action"] == "hotkey" and a.get("keys") == ["ctrl", "r"]
        )
        self.assertLess(first_esc, first_reload)

    def test_click_displacement_stays_near_target(self):
        # The anti-repeat nudge used to jump ±70/±50 px — far enough to
        # leave a 30 px menu row (it knocked near-correct clicks off
        # "New document" for a whole run, and once hit the Onshape logo).
        # It must stay near the original point yet still break the repeat.
        responses = iter(
            [
                _fake_vision_response({"action": "click", "x": 104, "y": 63, "reason": "click the item"}),
                _fake_vision_response({"action": "click", "x": 104, "y": 63, "reason": "click the item"}),
                _fake_vision_response({"action": "click", "x": 104, "y": 63, "reason": "click the item"}),
                _fake_vision_response({"action": "stop", "reason": "done"}),
            ]
        )
        with mock.patch("agent.loop.chat_vision_json", side_effect=lambda *a, **k: next(responses)):
            self.loop.start_run(goal="menu click", model="test-model", use_plan=False, max_steps=8)
            self.assertTrue(_wait_until(lambda: not self.loop.busy()))
        clicks = [a for a in self.loop._browser.actions if a["action"] == "click"]
        self.assertEqual(len(clicks), 3)
        displaced = [a for a in clicks if "displaced" in str(a.get("reason") or "")]
        self.assertEqual(len(displaced), 1, "the 3rd identical click must be displaced")
        d = displaced[0]
        self.assertLessEqual(abs(d["x"] - 104), 24)
        self.assertLessEqual(abs(d["y"] - 63), 18)
        self.assertNotEqual((d["x"], d["y"]), (104, 63))

    def test_doc_workspace_url_detection(self):
        from agent.loop import _doc_workspace_url

        self.assertTrue(_doc_workspace_url("https://cad.onshape.com/documents/533ff50996f4e88475f3945/tw/x/e/y"))
        self.assertFalse(_doc_workspace_url("https://cad.onshape.com/documents"))
        self.assertFalse(_doc_workspace_url("https://cad.onshape.com/documents?resourceType=resourceuserowner"))
        self.assertFalse(_doc_workspace_url("https://cad.onshape.com/documents/"))
        self.assertFalse(_doc_workspace_url(""))

    def test_run_with_mocked_vision_stops_on_stop_action(self):
        responses = iter(
            [
                _fake_vision_response({"action": "click", "x": 10, "y": 20, "reason": "pick tool"}),
                _fake_vision_response({"action": "stop", "reason": "goal done"}),
            ]
        )
        with mock.patch("agent.loop.chat_vision_json", side_effect=lambda *a, **k: next(responses)), mock.patch(
            "agent.loop.AgentLoop.make_plan", side_effect=RuntimeError("no plan")
        ):
            self.loop.start_run(goal="make a cube", use_plan=False, max_steps=5)
            self.assertTrue(_wait_until(lambda: not self.loop.busy()))
        st = self.loop.status()
        self.assertEqual(st["step"], 2)
        browser = self.loop._browser
        self.assertIsInstance(browser, FakeBrowser)
        actions = [a["action"] for a in browser.actions]
        self.assertEqual(actions, ["click", "stop"])

    def test_stop_interrupts_run(self):
        # A vision model that always returns wait; stop() must break the loop.
        with mock.patch(
            "agent.loop.chat_vision_json",
            side_effect=lambda *a, **k: _fake_vision_response({"action": "wait", "seconds": 0.2}),
        ):
            self.loop.start_run(goal="long task", use_plan=False, max_steps=500)
            self.assertTrue(_wait_until(lambda: self.loop.status()["step"] >= 1))
            self.loop.stop(timeout=5.0)
            self.assertFalse(self.loop.busy())
            self.assertIn(self.loop.status()["status"], {"Stopped", "Idle", "Error"})

    def test_plan_mode_advance(self):
        plan = [{"step": "sketch a circle", "expect": "circle drawn"}, {"step": "extrude 10mm", "expect": ""}]
        responses = iter(
            [
                _fake_vision_response(
                    {"action": "click", "x": 5, "y": 5, "reason": "draw circle; circle drawn"}
                ),
                _fake_vision_response({"action": "stop", "reason": "done"}),
            ]
        )
        with mock.patch("agent.loop.AgentLoop.make_plan", return_value=plan) as mk, mock.patch(
            "agent.loop.chat_vision_json", side_effect=lambda *a, **k: next(responses)
        ):
            self.loop.start_run(goal="a part", use_plan=True, max_steps=10)
            self.assertTrue(_wait_until(lambda: not self.loop.busy()))
            mk.assert_called_once()
            # The planner is told the REAL current page, not a hardcoded one.
            self.assertIn("Fake Onshape", mk.call_args.kwargs.get("page_hint", ""))
        st = self.loop.status()
        # First step matched its "expect" text -> plan advanced past step 1.
        self.assertGreaterEqual(st["plan_index"], 1)

    def test_repeat_breaker_injected(self):
        responses = iter(
            [
                _fake_vision_response({"action": "wait", "seconds": 0.1, "reason": "x"}),
                _fake_vision_response({"action": "wait", "seconds": 0.1, "reason": "x"}),
                _fake_vision_response({"action": "wait", "seconds": 0.1, "reason": "x"}),
                _fake_vision_response({"action": "stop", "reason": "done"}),
            ]
        )
        with mock.patch("agent.loop.chat_vision_json", side_effect=lambda *a, **k: next(responses)):
            self.loop.start_run(goal="stuck", use_plan=False, max_steps=8)
            self.assertTrue(_wait_until(lambda: not self.loop.busy()))
        browser = self.loop._browser
        kinds = [a["action"] for a in browser.actions]
        # Third identical wait must be converted into a repeat-breaker (escape key).
        self.assertIn("key", kinds)

    def test_ollama_failure_sets_error_and_recovers(self):
        from agent.ollama import OllamaError

        with mock.patch(
            "agent.loop.chat_vision_json", side_effect=OllamaError("ollama down")
        ):
            self.loop.start_run(goal="x", use_plan=False, max_steps=2)
            self.assertTrue(_wait_until(lambda: not self.loop.busy()))
        st = self.loop.status()
        self.assertTrue(st["error"])
        self.assertEqual(st["status"], "Error")

    def test_busy_blocks_second_run(self):
        with mock.patch(
            "agent.loop.chat_vision_json",
            side_effect=lambda *a, **k: _fake_vision_response({"action": "wait", "seconds": 0.2}),
        ):
            self.loop.start_run(goal="one", use_plan=False, max_steps=50)
            with self.assertRaises(RuntimeError):
                self.loop.start_run(goal="two")
            self.loop.stop(timeout=5.0)


class RecordReplayTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.loop = create_loop_for_tests(self.root)

    def tearDown(self):
        self.loop.stop(timeout=5.0)
        self._tmp.cleanup()

    def test_record_stop_saves_episode(self):
        self.loop.start_record(goal="draw a rectangle", interval_ms=100)
        self.assertTrue(self.loop.status()["recording"])
        time.sleep(0.2)
        # Simulate a user click observed inside the browser tab.
        self.loop._browser.captured.append(
            {"action": "click", "x": 100, "y": 200, "reason": "recorded user click"}
        )
        time.sleep(0.25)
        result = self.loop.stop_record(goal="draw a rectangle")
        self.assertTrue(result["ok"])
        self.assertFalse(self.loop.status()["recording"])
        episodes = self.loop.memory.episodes()
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["goal"], "draw a rectangle")
        # Skill also stored for future matching.
        self.assertTrue(self.loop.matching_skills("draw rectangle"))

    def test_stop_record_without_record_raises(self):
        with self.assertRaises(RuntimeError):
            self.loop.stop_record()

    def test_replay_last_episode(self):
        self.loop.memory.add_episode(
            {
                "goal": "demo",
                "actions": [
                    {"action": "click", "x": 1, "y": 2},
                    {"action": "key", "keys": ["s"]},
                ],
                "screenshots": [],
                "source": "record",
            }
        )
        self.loop.replay_last()
        self.assertTrue(_wait_until(lambda: not self.loop.busy(), timeout=6.0))
        browser = self.loop._browser
        self.assertEqual([a["action"] for a in browser.actions], ["click", "key"])

    def test_replay_without_episode_raises(self):
        with self.assertRaises(RuntimeError):
            self.loop.replay_last()


class LearningGuardTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.loop = create_loop_for_tests(self.root)

    def tearDown(self):
        self.loop.stop(timeout=5.0)
        self._tmp.cleanup()

    def test_learn_requires_source(self):
        with self.assertRaises(ValueError):
            self.loop.learn_from_video()

    def test_learn_rejects_non_youtube_url(self):
        with self.assertRaises(ValueError):
            self.loop.learn_from_video(url="https://evil.com/video")
        self.assertFalse(self.loop.status()["learning_video"])

    def test_learn_accepts_youtube_url(self):
        # Starts the background thread; it fails fast on missing Ollama/network
        # but the call itself must not raise.
        self.loop.learn_from_video(url="https://youtu.be/abc123def45", max_minutes=1)
        self.assertTrue(self.loop.status()["learning_video"])
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and self.loop.status()["learning_video"]:
            time.sleep(0.1)
        # Thread must terminate (download error is fine — no network in CI).
        self.assertFalse(self.loop.status()["learning_video"])

    def test_learn_rejects_path_outside_dataset(self):
        with self.assertRaises(ValueError):
            self.loop.learn_from_video(path=str(self.root.parent / "outside.mp4"))

    def test_learn_blocks_while_running(self):
        self.loop._running = True
        try:
            with self.assertRaises(RuntimeError):
                self.loop.learn_from_video(url="https://youtu.be/abc123def45")
        finally:
            self.loop._running = False

    def test_learn_cancel_flag_reaches_the_pipeline(self):
        # Regression: Stop used to have no effect on a running video lesson —
        # learn_concepts_from_video now receives the loop's stop flag and the
        # thread ends with a "Stopped" status instead of an Error.
        started = threading.Event()

        def fake_learn(**kwargs):
            started.set()
            self.assertTrue(callable(kwargs.get("cancelled")))
            # Simulate the user pressing Stop mid-lesson.
            self.loop._stop.set()
            if kwargs["cancelled"]():
                raise StopIteration("Video learning stopped by user")
            return {"concepts": []}

        with mock.patch(
            "agent.video_teacher.learn_concepts_from_video", side_effect=fake_learn
        ):
            self.loop.learn_from_video(url="https://youtu.be/abc123def45", max_minutes=1)
            self.assertTrue(started.wait(5.0), "learn thread never started")
        self.assertTrue(
            _wait_until(lambda: not self.loop.status()["learning_video"]),
            "learn thread did not finish",
        )
        st = self.loop.status()
        self.assertEqual(st["status"], "Stopped")
        self.assertIn("stopped", st["detail"].lower())


class GoalHelperTests(unittest.TestCase):
    def test_goal_matches(self):
        loop = AgentLoop.__new__(AgentLoop)  # no filesystem needed
        self.assertTrue(AgentLoop.goal_matches(loop, "fillet edges", "fillet the edges"))
        self.assertFalse(AgentLoop.goal_matches(loop, "fillet edges", "revolve profile"))


if __name__ == "__main__":
    unittest.main()
