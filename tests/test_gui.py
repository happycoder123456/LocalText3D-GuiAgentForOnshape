from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from agent.gui import SidecarClient, SidecarError, project_root
from agent.loop import AgentLoop
from agent.server import make_server


def _wait_for_server(httpd, timeout: float = 5.0) -> None:
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            import http.client

            conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=0.5)
            conn.request("GET", "/status")
            conn.getresponse().read()
            conn.close()
            return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("server did not start")


class SidecarClientTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        from agent.browser import FakeBrowser

        loop = AgentLoop(dataset_root=self.root, browser=FakeBrowser())
        self.loop = loop
        self.httpd = make_server(host="127.0.0.1", port=0, loop=loop)
        _wait_for_server(self.httpd)
        self.client = SidecarClient(
            base_url=f"http://127.0.0.1:{self.httpd.server_port}", timeout=3.0
        )

    def tearDown(self):
        try:
            self.httpd.shutdown()
            self.httpd.server_close()
        finally:
            self._tmp.cleanup()

    def test_get_status(self):
        data = self.client.get("/status")
        self.assertIn("running", data)
        self.assertIn("memory", data)

    def test_reachable_true_when_up(self):
        self.assertTrue(self.client.reachable())

    def test_unreachable_raises_sidecar_error(self):
        client = SidecarClient(base_url="http://127.0.0.1:1", timeout=0.4)
        with self.assertRaises(SidecarError):
            client.get("/status")
        self.assertFalse(client.reachable())

    def test_http_error_surfaces_message(self):
        with self.assertRaises(SidecarError) as ctx:
            self.client.post("/agent/start", {"goal": ""})
        self.assertIn("Goal", str(ctx.exception))

    def test_run_error_field_is_not_a_dead_sidecar(self):
        # Regression: status/health carry the agent's `error` field (e.g.
        # NEEDS_LOGIN after a failed run). The client used to raise on it, so
        # the GUI declared "sidecar stopped", spawned restart attempts, showed
        # "did not come back", and froze the run display — all while the
        # sidecar was healthy. Run errors must read as normal status data.
        self.loop._error = "NEEDS_LOGIN: sign in"
        st = self.client.get("/status")
        self.assertEqual(st["error"], "NEEDS_LOGIN: sign in")
        self.assertTrue(self.client.reachable())
        health = self.client.get("/health")
        self.assertTrue(health.get("ok"))
        out = self.client.post("/agent/stop", {})
        self.assertTrue(out.get("ok"))

    def test_skills_lists_learned_concepts(self):
        data = self.client.get("/skills")
        self.assertIn("skills", data)
        self.assertIn("concepts", data)


class GuiModuleTests(unittest.TestCase):
    def test_project_root_contains_agent_package(self):
        root = project_root()
        self.assertTrue((root / "agent" / "gui.py").is_file())

    def test_gui_import_does_not_create_tk_window(self):
        # Importing agent.gui must be side-effect free (CI has no display).
        import importlib

        mod = importlib.import_module("agent.gui")
        self.assertTrue(callable(mod.main))
        self.assertFalse(hasattr(mod, "app"))


if __name__ == "__main__":
    unittest.main()
