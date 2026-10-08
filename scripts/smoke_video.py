"""Smoke test for the video-teacher frame pipeline (no network, no Ollama).

Generates a 6-second synthetic video with ffmpeg, samples keyframes, and checks
caption parsing + concept merging. Run:  python scripts/smoke_video.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.paths import ensure_dataset, videos_dir  # noqa: E402
from agent.video_teacher import (  # noqa: E402
    _read_caption_file,
    ensure_video_path_in_dataset,
    merge_concepts,
    sample_keyframes,
    video_duration_sec,
)


def main() -> int:
    if not shutil.which("ffmpeg"):
        print("SKIP: ffmpeg not on PATH")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        root = ensure_dataset(Path(tmp))
        video = videos_dir(root) / "smoke.mp4"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24:duration=6",
                "-pix_fmt", "yuv420p", "-y", str(video),
            ],
            check=True,
            timeout=120,
        )
        print(f"video: {video.stat().st_size} bytes")

        # Path containment
        assert ensure_video_path_in_dataset(str(video), root) is not None
        try:
            ensure_video_path_in_dataset(str(Path.home() / "nope.mp4"), root)
            raise AssertionError("outside path was accepted")
        except ValueError:
            print("containment: OK (outside path refused)")

        dur = video_duration_sec(video, fallback=0.0)
        assert 5.0 <= dur <= 7.0, dur
        print(f"duration: {dur:.2f}s")

        frames = sample_keyframes(video, max_minutes=1, max_frames=12, dataset_root=root)
        assert frames, "no frames"
        assert frames[0][0] == 0.0
        w, h = frames[0][1].size
        assert w <= 960, w
        print(f"frames: {len(frames)} sampled, first {w}x{h}")

        # Monotonic timestamps
        times = [t for t, _ in frames]
        assert times == sorted(times), "timestamps not monotonic"

        # Caption file parsing
        vtt = video.with_suffix(".vtt")
        vtt.write_text(
            "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nWe extrude the profile 10mm\n",
            encoding="utf-8",
        )
        cues = _read_caption_file(vtt)
        assert cues and "extrude" in cues[0][1], cues
        print(f"captions: {len(cues)} cue(s) parsed")

        merged = merge_concepts(
            [{"name": "Extrude", "summary": "a"}, {"name": "extrude", "summary": "longer"}]
        )
        assert len(merged) == 1 and merged[0]["summary"] == "longer"
        print("concept merge: OK")

    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
