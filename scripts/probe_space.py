"""Probe: does the VLM answer in received-image pixel space? (small vs full res)"""

from __future__ import annotations

import base64

from agent.images import encode_jpeg_b64
from agent.ollama import chat_vision_json

SYS = (
    "You control a browser. Coordinates x,y are pixels of the screenshot image "
    "you see, origin top-left. Answer with ONE JSON object only.\n"
    'Schema: {"x":0,"y":0,"reason":""}'
)
Q = (
    "Locate the CENTER of the blue 'Create' button in the top-left corner. "
    'Answer {"x":0,"y":0} with its center pixel.'
)


def ask(label: str, b64: str) -> None:
    for i in range(3):
        raw = chat_vision_json(
            "qwen2.5vl:3b",
            system=SYS,
            user_text=Q,
            images_b64=[b64],
            num_predict=120,
        )
        print(f"{label} #{i + 1}: {raw}")


def main() -> None:
    full = open("_probe_full.png", "rb").read()
    small = open("_probe_672.jpg", "rb").read()
    big_b64 = encode_jpeg_b64(full, max_side=1440, quality=85)
    small_b64 = base64.b64encode(small).decode("ascii")
    ask("small(672)", small_b64)
    ask("big(1440)", big_b64)


if __name__ == "__main__":
    main()
