from __future__ import annotations

import json
import unittest
from unittest import mock

from agent.ollama import (
    OllamaError,
    chat_vision_json,
    prefer_text_model,
    prefer_vision_model,
    resolve_planner_model,
    safe_ollama_name,
)


class SafeNameTests(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(safe_ollama_name("llama3.2-vision"), "llama3.2-vision")
        self.assertEqual(safe_ollama_name("qwen2.5vl:7b"), "qwen2.5vl:7b")

    def test_empty_refused(self):
        with self.assertRaises(ValueError):
            safe_ollama_name("")
        with self.assertRaises(ValueError):
            safe_ollama_name("   ")

    def test_injection_refused(self):
        for bad in ("bad\nname", "a;rm -rf /", "../etc/passwd", "/abs/path", "a b"):
            with self.assertRaises(ValueError):
                safe_ollama_name(bad)


class PreferModelTests(unittest.TestCase):
    def test_vision_prefers_known_families(self):
        self.assertEqual(prefer_vision_model(["llama3.2:8b", "llama3.2-vision"]), "llama3.2-vision")
        self.assertEqual(prefer_vision_model(["qwen2.5vl:7b", "llava"]), "qwen2.5vl:7b")

    def test_vision_falls_back_to_first(self):
        self.assertEqual(prefer_vision_model(["mystery-model"]), "mystery-model")
        self.assertEqual(prefer_vision_model([]), "qwen2.5vl")

    def test_text_prefers_non_vision(self):
        # A text model should win over a vision model for planning.
        self.assertEqual(prefer_text_model(["llama3.2-vision", "qwen2.5:7b"]), "qwen2.5:7b")

    def test_text_skips_coder_and_embed(self):
        self.assertEqual(prefer_text_model(["qwen2.5-coder:7b", "phi3:mini"]), "phi3:mini")

    def test_text_falls_back_to_vision(self):
        self.assertEqual(prefer_text_model(["llama3.2-vision"]), "llama3.2-vision")
        self.assertEqual(prefer_text_model([], fallback="x"), "x")


class VisionFanOutTests(unittest.TestCase):
    """Ollama's qwen2.5vl runner breaks on 2+ images in one request.

    It answers with a zero-value reply and then keeps failing for later calls
    until the model reloads, which is what made every segment of a video lesson
    yield zero concepts. One image per request is the workaround.
    """

    @staticmethod
    def _images(messages):
        return messages[1]["images"]

    def test_single_image_stays_a_single_request(self):
        with mock.patch("agent.ollama._chat", return_value='{"a": 1}') as chat:
            out = chat_vision_json("m", system="s", user_text="u", images_b64=["one"])
        self.assertEqual(chat.call_count, 1)
        self.assertEqual(out, '{"a": 1}')
        self.assertEqual(self._images(chat.call_args[0][1]), ["one"])

    def test_no_image_is_still_one_request(self):
        with mock.patch("agent.ollama._chat", return_value="{}") as chat:
            chat_vision_json("m", system="s", user_text="u", images_b64=[])
        self.assertEqual(chat.call_count, 1)

    def test_multi_image_fans_out_one_image_each_and_merges(self):
        replies = [
            '{"concepts":[{"name":"sketch"}]}',
            '{"concepts":[{"name":"extrude"},{"name":"sketch"}]}',
            '{"concepts":[{"name":"fillet"}]}',
        ]
        with mock.patch("agent.ollama._chat", side_effect=replies) as chat:
            out = chat_vision_json(
                "m", system="s", user_text="u", images_b64=["i1", "i2", "i3"]
            )
        self.assertEqual(chat.call_count, 3)
        for call in chat.call_args_list:
            self.assertEqual(len(self._images(call[0][1])), 1, "each request must carry one image")
        names = [c["name"] for c in json.loads(out)["concepts"]]
        self.assertEqual(names, ["sketch", "extrude", "fillet"])

    def test_one_failed_frame_does_not_discard_the_others(self):
        replies = [OllamaError("wedged"), '{"steps":[{"action":"key","keys":["s"]}]}']
        with mock.patch("agent.ollama._chat", side_effect=replies):
            out = chat_vision_json("m", system="s", user_text="u", images_b64=["i1", "i2"])
        self.assertEqual(json.loads(out)["steps"][0]["action"], "key")

    def test_every_frame_failing_raises(self):
        with mock.patch(
            "agent.ollama._chat", side_effect=OllamaError("wedged")
        ):
            with self.assertRaises(OllamaError):
                chat_vision_json("m", system="s", user_text="u", images_b64=["i1", "i2"])


class _FakeResponse:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self, _n=-1):
        data, self._data = self._data, b""
        return data

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class ZeroValueReplyTests(unittest.TestCase):
    """A {model:"", done:false} body means the runner is wedged."""

    _ZERO = {
        "model": "",
        "created_at": "0001-01-01T00:00:00Z",
        "message": {"role": "", "content": ""},
        "done": False,
    }
    _GOOD = {"message": {"role": "assistant", "content": '{"ok":1}'}, "done": True}

    @staticmethod
    def _messages():
        return [{"role": "user", "content": "hi"}]

    def test_retry_reloads_the_wedged_model(self):
        from agent.ollama import _chat

        with mock.patch(
            "agent.ollama._OPENER.open",
            side_effect=[_FakeResponse(self._ZERO), _FakeResponse(self._GOOD)],
        ) as opener:
            out = _chat("m", self._messages(), timeout=5, num_predict=10, temperature=0.1, num_ctx=1024)
        self.assertEqual(out, '{"ok":1}')
        self.assertEqual(opener.call_count, 2)
        first = json.loads(opener.call_args_list[0][0][0].data)
        second = json.loads(opener.call_args_list[1][0][0].data)
        self.assertEqual(first["keep_alive"], "10m")
        self.assertEqual(second["keep_alive"], 0, "retry must evict the wedged runner")

    def test_persistent_zero_value_reply_raises(self):
        from agent.ollama import _chat

        # A fresh response per attempt: a single instance would be drained by
        # the first read and the second attempt would see an empty body.
        with mock.patch(
            "agent.ollama._OPENER.open",
            side_effect=lambda *_a, **_k: _FakeResponse(self._ZERO),
        ):
            with self.assertRaises(OllamaError) as ctx:
                _chat("m", self._messages(), timeout=5, num_predict=10, temperature=0.1, num_ctx=1024)
        self.assertIn("empty response", str(ctx.exception))


class ResolvePlannerTests(unittest.TestCase):
    def test_none_means_no_planner(self):
        self.assertEqual(resolve_planner_model("none"), "")

    def test_explicit_name_passthrough(self):
        self.assertEqual(resolve_planner_model("qwen3:8b"), "qwen3:8b")

    def test_auto_without_ollama_falls_back(self):
        # Hermetic: simulate Ollama being down regardless of the host machine.
        with mock.patch("agent.ollama.list_ollama_models", side_effect=OllamaError("down")):
            self.assertEqual(resolve_planner_model("auto", vision_model="qwen2.5vl"), "qwen2.5vl")

    def test_auto_with_ollama_picks_text_model(self):
        with mock.patch(
            "agent.ollama.list_ollama_models",
            return_value=["llama3.2-vision", "qwen2.5:3b"],
        ):
            self.assertEqual(
                resolve_planner_model("auto", vision_model="llama3.2-vision"),
                "qwen2.5:3b",
            )


if __name__ == "__main__":
    unittest.main()
