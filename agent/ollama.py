"""Local Ollama client — list models and chat (text + vision), loopback only."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

from agent.loopback import read_http_body

OLLAMA_URL = "http://127.0.0.1:11434"
_MAX_OLLAMA_BODY = 8_000_000


class OllamaError(Exception):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.URLError("Refusing HTTP redirect")


_OPENER = urllib.request.build_opener(_NoRedirect)

# Vision-capable families only — plain "llama3.2" is text-only and cannot see
# screenshots. Within one family the BIGGER variant grounds clicks far better
# (the 3B model missed click coordinates ~46% on live runs). Specific size
# tags come before generic family keys so an installed 7B wins over a 3B.
VISION_PREFER = (
    "qwen3-vl:32b",
    "qwen3-vl:14b",
    "qwen3-vl:8b",
    "qwen2.5vl:14b",
    "qwen2.5vl:7b",
    "qwen2.5vl:3b",
    "qwen2.5-vl:14b",
    "qwen2.5-vl:7b",
    "qwen2.5-vl:3b",
    "qwen3-vl",
    "qwen2.5vl",
    "qwen2.5-vl",
    "llama3.2-vision",
    "gemma3",
    "gemma4",
    "llava",
    "bakllava",
    "moondream",
    "minicpm-v",
)
_VISION_MARKERS = ("vision", "-vl", "llava", "moondream", "minicpm-v")

TEXT_PREFER = (
    "qwen3.8",
    "qwen3:32b",
    "qwen3:14b",
    "qwen3:8b",
    "qwen3",
    "qwen2.5:14b",
    "qwen2.5:7b",
    "qwen2.5",
    "llama3.1",
    "llama3.2-vision",
    "llama3.2",
    "gemma4",
    "gemma3",
    "gemma4:26b",
    "mistral",
    "phi",
)

_OLLAMA_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


def safe_ollama_name(name: str, *, field: str = "model") -> str:
    """Reject Modelfile/API injection via newlines or shell-metacharacter names."""
    text = (name or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    if not _OLLAMA_NAME_RE.fullmatch(text) or ".." in text or text.startswith("/"):
        raise ValueError(f"Invalid {field} name")
    return text


def list_ollama_models(timeout: float = 5.0) -> list[str]:
    url = OLLAMA_URL.rstrip("/") + "/api/tags"
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            payload = json.loads(read_http_body(resp, _MAX_OLLAMA_BODY).decode("utf-8"))
    except Exception as exc:
        raise OllamaError(f"Cannot reach Ollama at {OLLAMA_URL}: {exc}") from exc
    names: list[str] = []
    for item in payload.get("models") or []:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            if name:
                names.append(name)
    return names


def _is_vision_name(name_l: str) -> bool:
    return any(marker in name_l for marker in _VISION_MARKERS)


def prefer_vision_model(models: list[str]) -> str:
    lower_map = {m.lower(): m for m in models}
    for key in VISION_PREFER:
        for name_l, name in lower_map.items():
            if key in name_l:
                return name
    # Last resort: any installed model may still be asked to try.
    return models[0] if models else "qwen2.5vl"


def prefer_text_model(models: list[str], fallback: str = "") -> str:
    """Pick a fast local text model for planning (vision weights are slower and worse at JSON)."""
    lower_map = {m.lower(): m for m in models}
    for allow_vision in (False, True):
        for key in TEXT_PREFER:
            for name_l, name in lower_map.items():
                if key not in name_l or "coder" in name_l or "embed" in name_l:
                    continue
                if not allow_vision and _is_vision_name(name_l):
                    continue
                return name
    return fallback or (models[0] if models else "")


def resolve_planner_model(requested: str, vision_model: str = "") -> str:
    """'auto'/'' -> best installed text model; 'none' -> no LLM planner; else the requested name."""
    req = (requested or "").strip()
    if req.lower() == "none":
        return ""
    if req and req.lower() != "auto":
        return req
    try:
        models = list_ollama_models(timeout=3.0)
    except OllamaError:
        return vision_model or ""
    return prefer_text_model(models, fallback=vision_model)


def _chat(
    model: str,
    messages: list[dict[str, Any]],
    timeout: float,
    *,
    num_predict: int,
    temperature: float,
    num_ctx: int,
) -> str:
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        # Thinking models spend num_predict on hidden reasoning and hand back
        # truncated (unparseable) JSON. All installed models accept think:false.
        "think": False,
        "keep_alive": "10m",
        "options": {
            "temperature": float(temperature),
            "num_predict": int(num_predict),
            "num_ctx": int(num_ctx),
        },
    }
    data = json.dumps(body).encode("utf-8")
    err_msg = "Ollama returned an empty response"
    # One retry: transient failures (model unloading, brief overload) and rare
    # empty completions usually clear on the second attempt.
    for attempt in range(2):
        if attempt:
            # A zero-value reply ({model:"", done:false}) means the runner is
            # wedged: it keeps failing for EVERY later call, even single-image
            # ones, until the model is evicted. Reloading on the retry clears it.
            body["keep_alive"] = 0
            data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_URL.rstrip("/") + "/api/chat",
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with _OPENER.open(req, timeout=timeout) as resp:
                payload = json.loads(read_http_body(resp, _MAX_OLLAMA_BODY).decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                raw = read_http_body(exc, min(_MAX_OLLAMA_BODY, 1_000_000)).decode("utf-8", "replace")
            except Exception:
                raw = str(exc)
            err_msg = (raw or str(exc))[:300]
        except Exception as exc:
            err_msg = f"Ollama chat failed: {exc}"
        else:
            problem = str(payload.get("error") or "")[:300]
            if not problem:
                message = payload.get("message") or {}
                content = message.get("content") if isinstance(message, dict) else None
                if content and str(content).strip():
                    return str(content)
                problem = "Ollama returned an empty response"
            err_msg = problem
        if attempt == 0:
            continue
        break
    raise OllamaError(err_msg)


def _merge_json_payloads(parts: list[Any]) -> dict[str, Any]:
    """Union of several JSON objects; list-valued keys are concatenated."""
    merged: dict[str, Any] = {}
    for part in parts:
        if isinstance(part, list):
            part = {"items": part}
        if not isinstance(part, dict):
            continue
        for key, value in part.items():
            if isinstance(value, list):
                current = merged.get(key)
                if not isinstance(current, list):
                    merged[key] = list(value)
                else:
                    current.extend(v for v in value if v not in current)
            elif key not in merged or merged[key] in (None, "", [], {}):
                merged[key] = value
    return merged


def chat_vision_json(
    model: str,
    *,
    system: str,
    user_text: str,
    images_b64: list[str] | None = None,
    num_predict: int = 512,
    timeout: float = 180.0,
) -> str:
    """Multi-image JSON vision call (screenshots, video teacher).

    Ollama's qwen2.5vl runner answers ANY request carrying 2+ images with a
    zero-value reply (``{model:"", done:false}``, HTTP 500 for some bodies) and
    then stays broken for later calls until the model is reloaded — which made
    every segment of a video lesson yield zero concepts. One frame per request
    avoids that entirely; results are merged so callers still see all frames.
    """
    images = list(images_b64 or [])[:8]
    if len(images) <= 1:
        return _chat(
            model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text, "images": images},
            ],
            timeout=timeout,
            num_predict=num_predict,
            temperature=0.1,
            num_ctx=8192,
        )

    parts: list[Any] = []
    last_err: OllamaError | None = None
    for image in images:
        try:
            raw = _chat(
                model,
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_text, "images": [image]},
                ],
                timeout=timeout,
                num_predict=num_predict,
                temperature=0.1,
                num_ctx=8192,
            )
        except OllamaError as exc:
            last_err = exc
            continue
        try:
            parts.append(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            continue
    merged = _merge_json_payloads(parts)
    if merged:
        return json.dumps(merged)
    if last_err is not None:
        raise last_err
    return ""


def chat_text_json(
    model: str,
    *,
    system: str,
    user_text: str,
    num_predict: int = 600,
    temperature: float = 0.1,
    timeout: float = 180.0,
    num_ctx: int = 8192,
) -> str:
    """Text-only JSON chat (planner, captions fallback)."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]
    # Ollama silently drops the head (our system prompt) when the prompt exceeds num_ctx.
    approx_tokens = (len(system) + len(user_text)) // 4 + int(num_predict)
    ctx = int(num_ctx)
    while ctx < approx_tokens + 512 and ctx < 32768:
        ctx *= 2
    return _chat(
        model,
        messages,
        timeout=timeout,
        num_predict=num_predict,
        temperature=temperature,
        num_ctx=ctx,
    )
