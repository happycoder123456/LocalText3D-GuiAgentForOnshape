"""HTTP sidecar for the Onshape GUI Agent (localhost:8767)."""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agent.loop import AgentLoop, MAX_STEPS_CAP
from agent.ollama import (
    OllamaError,
    list_ollama_models,
    prefer_vision_model,
    safe_ollama_name,
)
from agent.paths import default_dataset_dir
from agent.loopback import (
    require_loopback_bind,
    require_loopback_port,
    request_is_local,
    server_class_for,
)

MAX_BODY = 16_000_000
MAX_GOAL_CHARS = 8192

def _safe_error_text(exc: BaseException, limit: int = 300) -> str:
    """One-line error for the client. Never echo absolute local paths."""
    parts = []
    for part in str(exc).split("\\"):
        parts.append(part.split("/")[-1])
    return (" ".join(parts))[:limit]

_OLLAMA_CACHE_SEC = 4.0
_ollama_cache: dict[str, Any] = {"t": 0.0, "models": [], "ok": False, "err": ""}
_ollama_lock = threading.Lock()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _clip_text(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) > limit:
        raise ValueError(f"text is longer than {limit} characters")
    return text


def _ollama_models_cached() -> tuple[list[str], bool, str]:
    """Health is polled with a short timeout; never block on Ollama longer."""
    with _ollama_lock:
        now = time.time()
        if now - float(_ollama_cache["t"]) < _OLLAMA_CACHE_SEC:
            return list(_ollama_cache["models"]), bool(_ollama_cache["ok"]), str(_ollama_cache["err"])
        try:
            models = list_ollama_models(timeout=1.2)
            _ollama_cache.update({"t": now, "models": models, "ok": True, "err": ""})
        except OllamaError as exc:
            _ollama_cache.update({"t": now, "models": [], "ok": False, "err": str(exc)})
        except Exception as exc:
            _ollama_cache.update({"t": now, "models": [], "ok": False, "err": str(exc)})
        return list(_ollama_cache["models"]), bool(_ollama_cache["ok"]), str(_ollama_cache["err"])


class AgentHandler(BaseHTTPRequestHandler):
    loop: AgentLoop
    server_version = "OnshapeGuiAgent"
    sys_version = ""

    def log_message(self, format: str, *args) -> None:
        import sys

        line = format % args
        if "/status" in line:
            return  # high-frequency poll; keep the console readable
        sys.stderr.write("%s - %s\n" % (self.address_string(), line))

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = _json_bytes(payload)
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            return

    def _ensure_local(self) -> bool:
        if request_is_local(self.headers):
            return True
        self._send(403, {"error": "This sidecar only accepts localhost requests"})
        return False

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as exc:
            raise ValueError("Bad Content-Length") from exc
        if length < 0 or length > MAX_BODY:
            raise ValueError(f"Body length {length} out of range")
        raw = self.rfile.read(length) if length else b"{}"
        if not raw:
            return {}
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def do_GET(self) -> None:
        if not self._ensure_local():
            return
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/health":
                models, ollama_ok, err = _ollama_models_cached()
                preferred = prefer_vision_model(models) if ollama_ok else err
                st = self.loop.status()
                self._send(
                    200,
                    {
                        "ok": True,
                        "service": "onshape-gui-agent",
                        "ollama_ok": ollama_ok,
                        "models": models,
                        "preferred_model": preferred,
                        **st,
                    },
                )
                return
            if parsed.path == "/status":
                self._send(200, self.loop.status())
                return
            if parsed.path == "/skills":
                cards = self.loop.memory.list_skills(limit=240)
                concepts = self.loop.memory.list_concepts(limit=400)
                self._send(200, {"ok": True, "skills": cards, "concepts": concepts})
                return
        except Exception as exc:
            self._send(500, {"ok": False, "error": _safe_error_text(exc)})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._ensure_local():
            return
        parsed = urlparse(self.path)
        try:
            body = self._read_json()
        except Exception as exc:
            self._send(400, {"error": str(exc)})
            return

        try:
            if parsed.path == "/plan":
                planner = str(body.get("planner_model") or "")
                if planner:
                    planner = safe_ollama_name(planner, field="planner_model")
                preview = self.loop.preview_plan(
                    goal=_clip_text(body.get("goal") or "", MAX_GOAL_CHARS),
                    planner_model=planner,
                )
                # plan_preview LAST: status() also carries a plan_preview key
                # (the running plan) which would otherwise clobber the result.
                self._send(200, {"ok": True, **self.loop.status(), "plan_preview": preview})
                return
            if parsed.path == "/agent/start":
                goal = _clip_text(body.get("goal") or "", MAX_GOAL_CHARS)
                model = str(body.get("model") or "")
                if model:
                    model = safe_ollama_name(model, field="model")
                planner = str(body.get("planner_model") or "")
                if planner:
                    planner = safe_ollama_name(planner, field="planner_model")
                max_steps = int(body.get("max_steps") or 160)
                max_steps = max(1, min(MAX_STEPS_CAP, max_steps))
                use_plan = bool(body.get("use_plan", True))
                run_mode = str(body.get("run_mode") or "learn")
                if run_mode not in {"learn", "replay"}:
                    run_mode = "learn"
                self.loop.start_run(
                    goal=goal,
                    model=model,
                    planner_model=planner,
                    max_steps=max_steps,
                    use_plan=use_plan,
                    run_mode=run_mode,
                )
                self._send(200, {"ok": True, **self.loop.status()})
                return
            if parsed.path == "/agent/stop":
                self.loop.stop()
                self._send(200, {"ok": True, **self.loop.status()})
                return
            if parsed.path == "/record/start":
                interval = int(body.get("interval_ms") or 400)
                goal = _clip_text(body.get("goal") or "", MAX_GOAL_CHARS)
                self.loop.start_record(goal=goal, interval_ms=interval)
                self._send(200, {"ok": True, **self.loop.status()})
                return
            if parsed.path == "/record/stop":
                goal = _clip_text(body.get("goal") or "", MAX_GOAL_CHARS)
                success = bool(body.get("success", True))
                result = self.loop.stop_record(goal=goal, success=success)
                self._send(200, {"ok": True, "detail": result.get("detail", ""), **self.loop.status()})
                return
            if parsed.path == "/replay/start":
                goal = _clip_text(body.get("goal") or "", MAX_GOAL_CHARS)
                self.loop.replay_last(goal=goal)
                self._send(200, {"ok": True, **self.loop.status()})
                return
            if parsed.path == "/video/learn":
                url = str(body.get("url") or "")
                path = str(body.get("path") or "")
                goal = _clip_text(body.get("goal") or "", MAX_GOAL_CHARS)
                model = str(body.get("model") or "")
                if model:
                    model = safe_ollama_name(model, field="model")
                max_minutes = float(body.get("max_minutes") or body.get("learn_minutes") or 6.0)
                self.loop.learn_from_video(
                    url=url,
                    path=path,
                    goal=goal,
                    model=model,
                    max_minutes=max_minutes,
                )
                self._send(200, {"ok": True, **self.loop.status()})
                return
        except (ValueError, RuntimeError) as exc:
            self._send(400, {"error": str(exc)[:300]})
            return
        except OllamaError as exc:
            self._send(502, {"error": str(exc)[:300]})
            return
        except Exception as exc:
            self._send(500, {"error": _safe_error_text(exc), **self.loop.status()})
            return

        self._send(404, {"error": "not found"})


def _port_already_served(host: str, port: int) -> bool:
    """True when something already answers on this loopback port.

    Windows lets a second socket bind() over a live listener that used
    SO_REUSEADDR (http.server sets it), so a duplicate sidecar would silently
    SHARE port 8767 — two AgentLoop instances then answer requests with
    different state (one holds the browser and the run, the other is fresh),
    which shows up as random status/stop failures. Probe before binding.
    """
    try:
        with socket.create_connection((host, port), timeout=0.4):
            return True
    except OSError:
        return False


def make_server(
    host: str = "127.0.0.1",
    port: int = 8767,
    dataset_dir: Path | None = None,
    loop: AgentLoop | None = None,
) -> ThreadingHTTPServer:
    """Build (but do not start) the sidecar server. Tests use port 0."""
    host = require_loopback_bind(host)
    port = require_loopback_port(port)
    if port and _port_already_served(host, port):
        raise RuntimeError(
            f"Another Onshape Agent sidecar is already serving http://{host}:{port}"
        )
    if loop is None:
        loop = AgentLoop(dataset_root=dataset_dir or default_dataset_dir())
    AgentHandler.loop = loop
    server_cls = server_class_for(host, ThreadingHTTPServer)
    httpd = server_cls((host, port), AgentHandler)
    httpd.daemon_threads = True
    return httpd


def serve(host: str = "127.0.0.1", port: int = 8767, dataset_dir: Path | None = None) -> None:
    try:
        httpd = make_server(host=host, port=port, dataset_dir=dataset_dir)
    except RuntimeError as exc:
        # Duplicate sidecar (port already served): say why, exit non-zero.
        print(str(exc), flush=True)
        raise SystemExit(1) from exc
    loop = AgentHandler.loop
    shown = f"[{host}]" if ":" in host else host
    print(f"Onshape GUI Agent sidecar listening on http://{shown}:{httpd.server_port}", flush=True)
    print(f"Dataset: {loop.root}", flush=True)
    print("Ctrl+C here quits.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            loop.stop()
            if loop.busy():
                loop.stop(timeout=6.0)
        except Exception:
            pass
        # Runs leave the browser open for the user; quitting the sidecar is
        # the one place we still clean it up (avoids an orphan Chromium
        # holding the profile lock).
        try:
            loop.close_browser()
        except Exception:
            pass
        httpd.server_close()
