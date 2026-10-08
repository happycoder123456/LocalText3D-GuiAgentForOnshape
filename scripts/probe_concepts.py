"""Diagnostic: why does the video teacher return few concepts / placeholder steps?"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from agent import video_teacher as vt
from agent.ollama import OllamaError

SHOTS = sorted((Path.home() / "OnshapeGuiAgent" / "agent_dataset" / "debug_shots").glob("*.jpg"))[:6]
CAPTIONS = (
    "So let's start by creating a new sketch on the top plane. "
    "I'll pick the rectangle tool and draw a rectangle, then dimension it to fifty millimeters. "
    "Now we extrude that profile ten millimeters. Next I'll add a fillet on the top edges "
    "and a hole in the middle. Finally we will mirror the feature across the right plane "
    "and use a circular pattern for the bolt holes."
)


def main() -> int:
    frames = []
    for p in SHOTS[:4]:
        try:
            frames.append((0.0, Image.open(p).convert("RGB")))
        except OSError as exc:
            print("skip", p, exc)
    print(f"frames={len(frames)}")

    t0 = time.time()
    raw = ""
    try:
        raw = vt.chat_vision_json(
            "qwen2.5vl:3b",
            system=vt.CONCEPT_LIST_SYSTEM,
            user_text=(
                "Student goal (optional focus): (general CAD techniques)\n\n"
                f"Captions / narration excerpt:\n{CAPTIONS}\n\n"
                "List EVERY distinct Onshape technique taught in this segment "
                "(not only sketch/extrude/fillet — include revolve, sweep, loft, shell, draft, "
                "pattern, mirror, boolean, mates, sheet metal, construction geometry…)."
            ),
            images_b64=[vt.encode_jpeg_b64(f, max_side=512, quality=70) for _, f in frames],
            num_predict=2400,
        )
    except OllamaError as exc:
        print("VISION ERROR:", exc)
    print(f"vision call {time.time()-t0:.1f}s, raw len={len(raw)}")
    print("RAW >>>", raw[:1200].replace("\n", " "))
    data = vt._parse_json_object(raw)
    print("parsed keys:", list(data) if isinstance(data, dict) else type(data))
    cands = vt.extract_concept_candidates(
        model="qwen2.5vl:3b",
        goal="",
        captions=CAPTIONS,
        frames=frames,
    )
    print(f"extract_concept_candidates -> {len(cands)} concepts")
    for c in cands:
        print("   -", c.get("name"))

    cap = vt.concepts_from_captions([(0.0, CAPTIONS)], t0=0.0, t1=60.0)
    print(f"caption fallback -> {len(cap)} concepts:", [c.get("name") for c in cap])

    if cands:
        t0 = time.time()
        filled = vt.fill_concept_steps(
            dict(cands[0]), model="qwen2.5vl:3b", frames=frames, cues=[(0.0, CAPTIONS)]
        )
        steps = filled.get("steps") or []
        print(f"fill_concept_steps {time.time()-t0:.1f}s -> {len(steps)} steps")
        print(json.dumps(steps, indent=1)[:900])
        fallback = all(
            str(s.get("reason", "")).startswith(("open shortcut search", "find ", "run "))
            for s in steps
        )
        print("USED_SSEARCH_FALLBACK =", fallback)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
