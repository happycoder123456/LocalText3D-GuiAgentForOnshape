from __future__ import annotations

import http.client
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from agent.loop import AgentLoop
from agent.server import make_server


def _wait_for_server(httpd, timeout: float = 5.0) -> None:
    # ThreadingHTTPServer binds in the constructor; serve_forever just dispatches.
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=0.5)
            conn.request("GET", "/status")
            conn.getresponse().read()
            conn.close()
            return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("server did not start")


class ServerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.loop = AgentLoop(dataset_root=self.root, browser=_FakeBrowserForServer())
        self.httpd = make_server(host="127.0.0.1", port=0, loop=self.loop)
        _wait_for_server(self.httpd)
        self.port = self.httpd.server_port

    def tearDown(self):
        try:
            self.loop.stop(timeout=4.0)
        finally:
            self.httpd.shutdown()
            self.httpd.server_close()
            self._tmp.cleanup()

    def _request(self, method: str, path: str, body: dict | None = None, headers: dict | None = None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5.0)
        payload = json.dumps(body).encode() if body is not None else None
        hdrs = {"Accept": "application/json"}
        if payload is not None:
            hdrs["Content-Type"] = "application/json"
        if headers:
            hdrs.update(headers)
        conn.request(method, path, body=payload, headers=hdrs)
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        data = json.loads(raw.decode()) if raw else {}
        return resp.status, data

    def test_status_endpoint(self):
        status, data = self._request("GET", "/status")
        self.assertEqual(status, 200)
        self.assertIn("running", data)
        self.assertIn("memory", data)

    def test_health_endpoint_reports_ollama_down_gracefully(self):
        # Ollama is not running in the test environment; health must still answer 200.
        status, data = self._request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(data["service"], "onshape-gui-agent")
        self.assertIn("ollama_ok", data)
        self.assertIn("models", data)

    def test_skills_endpoint(self):
        self.loop.memory.add_skill(goal="extrude a circle", actions=[])
        self.loop.memory.add_concept(name="shell", summary="hollow it")
        status, data = self._request("GET", "/skills")
        self.assertEqual(status, 200)
        self.assertEqual(len(data["skills"]), 1)
        self.assertEqual(len(data["concepts"]), 1)

    def test_unknown_path_404(self):
        status, _ = self._request("GET", "/nope")
        self.assertEqual(status, 404)

    def test_non_local_host_rejected(self):
        status, data = self._request("GET", "/status", headers={"Host": "evil.com"})
        self.assertEqual(status, 403)
        self.assertIn("localhost", data.get("error", ""))

    def test_agent_start_validation(self):
        status, data = self._request("POST", "/agent/start", {"goal": ""})
        self.assertEqual(status, 400)
        self.assertIn("Goal", data["error"])

    def test_agent_start_bad_model_name(self):
        status, data = self._request(
            "POST", "/agent/start", {"goal": "x", "model": "bad\nname"}
        )
        self.assertEqual(status, 400)

    def test_agent_start_and_stop(self):
        with mock.patch(
            "agent.loop.chat_vision_json",
            side_effect=lambda *a, **k: json.dumps({"action": "wait", "seconds": 0.2}),
        ):
            status, data = self._request(
                "POST", "/agent/start", {"goal": "test goal", "use_plan": False, "max_steps": 40}
            )
            self.assertEqual(status, 200)
            self.assertTrue(data["running"])
            # Stop it.
            status, data = self._request("POST", "/agent/stop", {})
            self.assertEqual(status, 200)
            deadline = time.monotonic() + 6.0
            while time.monotonic() < deadline and self.loop.busy():
                time.sleep(0.05)
            self.assertFalse(self.loop.busy())

    def test_video_learn_validates_path(self):
        status, data = self._request("POST", "/video/learn", {"path": "/etc/hosts.mp4"})
        self.assertEqual(status, 400)

    def test_video_learn_requires_source(self):
        status, data = self._request("POST", "/video/learn", {})
        self.assertEqual(status, 400)

    def test_record_requires_active(self):
        status, _ = self._request("POST", "/record/stop", {"goal": "x"})
        self.assertEqual(status, 400)

    def test_plan_requires_goal(self):
        status, _ = self._request("POST", "/plan", {"goal": ""})
        self.assertEqual(status, 400)

    def test_plan_preview_not_clobbered_by_status_key(self):
        # status() also carries a plan_preview key (running plan as a LIST);
        # it must not overwrite the preview dict in the /plan response.
        with mock.patch.object(
            self.loop,
            "preview_plan",
            return_value={"steps": [{"step": "s1", "expect": ""}], "count": 1, "planner_model": "m"},
        ):
            status, data = self._request("POST", "/plan", {"goal": "make a cube"})
        self.assertEqual(status, 200)
        self.assertIsInstance(data["plan_preview"], dict)
        self.assertEqual(data["plan_preview"]["count"], 1)
        self.assertEqual(len(data["plan_preview"]["steps"]), 1)

    def test_malformed_body_400(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5.0)
        conn.request("POST", "/agent/start", body=b"not json", headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        resp.read()
        conn.close()
        self.assertEqual(resp.status, 400)

    def test_oversized_body_rejected(self):
        import agent.server as server_mod

        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5.0)
        huge = b"x" * (server_mod.MAX_BODY + 10)
        conn.putrequest("POST", "/agent/start")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(len(huge)))
        conn.endheaders()
        try:
            conn.send(huge)
            resp = conn.getresponse()
            status = resp.status
        except OSError:
            status = 0  # server closed the connection on purpose
        conn.close()
        self.assertIn(status, (0, 400))


class _FakeBrowserForServer:
    """Minimal browser double for server tests (loop never actually drives it)."""

    def __init__(self):
        self._available = False

    @property
    def available(self):
        return self._available

    def start(self):
        self._available = True

    def stop(self):
        self._available = False

    def url(self):
        return "https://cad.onshape.com/documents"

    def title(self):
        return "Onshape"

    def screenshot_png(self):
        return b"\x89PNG\r\n\x1a\n"

    def execute(self, action, stop_check=None):
        return {"done": True}

    def goto(self, url):
        pass


class BindRefusalTests(unittest.TestCase):
    def test_non_loopback_bind_refused_by_make_server(self):
        from agent.loopback import require_loopback_bind

        with self.assertRaises(ValueError):
            require_loopback_bind("0.0.0.0")

    def test_serve_uses_port_8767_default(self):
        import inspect

        from agent.server import serve

        sig = inspect.signature(serve)
        self.assertEqual(sig.parameters["port"].default, 8767)

    def test_make_server_refuses_already_served_port(self):
        # Windows allows a second SO_REUSEADDR bind over a live listener
        # (http.server sets it), so a duplicate sidecar would silently SHARE
        # port 8767 and answer with a different AgentLoop — status, browser,
        # and run state then come from different processes. Refuse instead.
        import socket

        first = socket.create_server(("127.0.0.1", 0))
        port = first.getsockname()[1]
        try:
            with self.assertRaises(RuntimeError):
                make_server(host="127.0.0.1", port=port)
        finally:
            first.close()

    def test_make_server_still_binds_free_port(self):
        import socket

        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]
        probe.close()
        with tempfile.TemporaryDirectory() as tmp:
            loop = AgentLoop(dataset_root=Path(tmp))
            httpd = make_server(host="127.0.0.1", port=free_port, loop=loop)
        try:
            self.assertEqual(httpd.server_port, free_port)
        finally:
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
