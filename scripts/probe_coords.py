"""One-off calibration probe: what coordinate space does the VLM answer in?"""

from __future__ import annotations

import base64
import json

from agent.browser import BrowserDriver
from agent.images import encode_jpeg_b64
from agent.ollama import chat_vision_json
from agent.paths import browser_profile_dir


def main() -> None:
    driver = BrowserDriver(headless=False, profile_dir=str(browser_profile_dir(None)))
    driver.start()
    try:
        png = driver.screenshot_png()
        open("_probe_full.png", "wb").write(png)
        b64 = encode_jpeg_b64(png, max_side=672, quality=80)
        open("_probe_672.jpg", "wb").write(base64.b64decode(b64))
        print("url:", driver.url())
        print("title:", driver.title())

        q_size = chat_vision_json(
            "qwen2.5vl:3b",
            system='Answer with ONE JSON object only, no commentary.',
            user_text='What is the pixel width and height of the image you received? {"width":0,"height":0}',
            images_b64=[b64],
            num_predict=80,
        )
        print("size answer:", q_size)

        q_row = chat_vision_json(
            "qwen2.5vl:3b",
            system='Answer with ONE JSON object only, no commentary.',
            user_text=(
                "Locate the center of the table row whose first column is exactly 'Bench Vise'. "
                'Coordinates are pixels of THIS image. {"x":0,"y":0}'
            ),
            images_b64=[b64],
            num_predict=80,
        )
        print("row answer:", q_row)

        q_card = chat_vision_json(
            "qwen2.5vl:3b",
            system='Answer with ONE JSON object only, no commentary.',
            user_text=(
                "Locate the center of the 'Bench Vise' card under 'Last opened by me'. "
                'Coordinates are pixels of THIS image. {"x":0,"y":0}'
            ),
            images_b64=[b64],
            num_predict=80,
        )
        print("card answer:", q_card)
    finally:
        driver.stop()


if __name__ == "__main__":
    main()
