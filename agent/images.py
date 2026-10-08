"""Screenshot/frame encoding helpers for the vision model (Pillow-based)."""

from __future__ import annotations

import base64
import io
from typing import Any

MAX_VLM_SIDE = 672
MAX_KEYFRAME_SIDE = 640


def to_pil(image: Any):
    """Accept PIL images, PNG/JPEG bytes, or a cv2 BGR ndarray -> PIL RGB image."""
    from PIL import Image

    if Image.isImageType(image) if hasattr(Image, "isImageType") else isinstance(image, Image.Image):
        return image.convert("RGB") if image.mode != "RGB" else image
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(image))).convert("RGB")
    # cv2 frame: numpy BGR ndarray
    import numpy as np

    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[:, :, :3][:, :, ::-1]  # BGR -> RGB
        if arr.dtype != "uint8":
            arr = (np.clip(arr, 0, 255)).astype("uint8")
        return Image.fromarray(arr, "RGB")
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def downsample(image: Any, max_side: int = MAX_VLM_SIDE):
    """RGB PIL image scaled so the longest side is <= max_side."""
    from PIL import Image

    img = to_pil(image)
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    # Pillow 10+ removed string filters; the enum works on 9.x and 10.x+ alike.
    resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    return img.resize(new_size, resample)


def encode_jpeg_b64(image: Any, max_side: int = MAX_VLM_SIDE, quality: int = 80) -> str:
    """Downsample + JPEG-encode an image as base64 for the Ollama vision API."""
    img = downsample(image, max_side=max_side)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def encode_png_bytes(image: Any) -> bytes:
    """PNG bytes for on-disk episode screenshots."""
    img = to_pil(image)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
