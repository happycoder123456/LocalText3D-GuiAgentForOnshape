"""Learn reusable Onshape CAD concepts from YouTube videos or local files.

Pipeline: yt-dlp download -> ffmpeg keyframes + captions -> VLM concept cards
-> steps for the agent. Everything stays on this machine.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from agent.images import encode_jpeg_b64
from agent.memory import is_cad_concept
from agent.ollama import OllamaError, chat_text_json, chat_vision_json
from agent.paths import ensure_dataset, frames_dir, videos_dir

ProgressFn = Callable[[str], None]

_VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv", ".avi", ".mov", ".m4v"}
MAX_LOCAL_VIDEO_BYTES = 4_000_000_000
MAX_CAPTION_BYTES = 8_000_000
MAX_FRAMES = 300
_YOUTUBE_HOSTS = {
    "youtu.be",
    "www.youtu.be",
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
}
_FILLER = re.compile(
    r"\b(subscribe|like and subscribe|patreon|sponsor|outro|intro music|thanks for watching)\b",
    re.I,
)

_KNOWN_TECHNIQUES = (
    "sketch, line, rectangle, circle, arc, spline, dimension, constraint, coincident, "
    "parallel, perpendicular, tangent, symmetric, extrude, revolve, sweep, loft, fillet, "
    "chamfer, shell, draft, hole, thread, rib, pattern, linear pattern, circular pattern, "
    "mirror, boolean, union, subtract, intersect, split, plane, offset plane, mate connector, "
    "mate, fastened mate, revolute mate, slider mate, rectangular pattern, variable pattern, "
    "face fillet, full round, wrap, project, use, trim, extend, join, convert to sheet metal"
)

CONCEPT_LIST_SYSTEM = f"""Onshape CAD teacher. Extract DISTINCT transferable CONCEPTS (techniques) from the lesson, not a mouse screenplay.
JSON only: {{"concepts":[{{"name":"extrude","summary":"...","when_to_use":"...","preconditions":["sketch complete"],"aliases":[],"params":{{}},"t_start":12.0,"t_end":40.0}}]}}
Rules:
- Prefer SPECIFIC lesson names (extrude blind, revolve profile, circular pattern of holes, fillet with setback…). Do NOT collapse everything into only sketch/extrude/fillet.
- Techniques you may use as names when accurate: {_KNOWN_TECHNIQUES}.
- Each concept must be a different technique or clearly different use. Max 32 concepts per segment.
- Plain-language summary; preconditions for the USER's part; numeric values go in params; skip subscribe/outro filler.
"""

CONCEPT_STEPS_SYSTEM = """Turn one Onshape CONCEPT into a procedure for the user's current part.
JSON only, ALWAYS an array of 3-8 steps:
{{"steps":[
 {{"action":"click","target":"Extrude","reason":"open the extrude tool"}},
 {{"action":"click","target":"distance field","reason":"enter the distance from the plan"}},
 {{"action":"key","keys":["enter"],"reason":"commit the value"}},
 {{"action":"click","target":"green check mark","reason":"accept the feature"}}
]}}
Every step names the REAL Onshape control involved (toolbar button, field, face, edge).
A typical procedure is: open the tool -> pick geometry -> set the value -> click the green check.
Only use the S shortcut search (key "s", then type the command, Enter) when the technique is ONE Onshape command name.
NEVER answer with a single step, and NEVER answer with just "press S and type <the concept name>".
NEVER put a placeholder word in "text" (no "distance" or "value" — click the field instead).
NEVER invent click x/y coordinates — only give a textual target. Max 10 steps. No stop.
If you cannot describe a real multi-step procedure, return {{"steps":[]}} instead of guessing.
"""

# Caption / narration hints -> seed concepts when the VLM only returns the basics.
_CAPTION_TECHNIQUES: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"\bextrud", re.I), "extrude", "Push a sketch region into 3D"),
    (re.compile(r"\brevolv", re.I), "revolve", "Spin a profile around an axis"),
    (re.compile(r"\bsweep\b", re.I), "sweep", "Sweep a profile along a path"),
    (re.compile(r"\bloft\b", re.I), "loft", "Blend between cross-sections"),
    (re.compile(r"\bfillet\b", re.I), "fillet", "Round an edge"),
    (re.compile(r"\bchamfer\b", re.I), "chamfer", "Bevel an edge"),
    (re.compile(r"\bshell\b", re.I), "shell", "Hollow a solid to a wall thickness"),
    (re.compile(r"\bdraft\b", re.I), "draft", "Add a draft angle to faces"),
    (re.compile(r"\b(pattern|patterning)\b", re.I), "pattern", "Repeat features in a pattern"),
    (re.compile(r"\bmirror\b", re.I), "mirror", "Mirror features across a plane"),
    (re.compile(r"\b(hole|holes)\b", re.I), "hole", "Place a standard or custom hole"),
    (re.compile(r"\b(rib|ribs)\b", re.I), "rib", "Add a supporting rib"),
    (re.compile(r"\b(sketch|sketching)\b", re.I), "sketch", "Draw a constrained 2D profile"),
    (re.compile(r"\b(dimension|dimensions|constrain|constraint)\b", re.I), "dimension", "Constrain sketch geometry with dimensions"),
    (re.compile(r"\b(boolean|union|subtract|intersect)\b", re.I), "boolean", "Combine solids with boolean operations"),
    (re.compile(r"\b(sheet metal)\b", re.I), "sheet metal", "Model sheet metal parts and flats"),
    (re.compile(r"\b(mate|mates|assembly)\b", re.I), "mate", "Assemble parts with mates"),
    (re.compile(r"\bfillet\b.*\bface\b", re.I), "face fillet", "Fillet between two faces"),
    (re.compile(r"\b(derive|insert derived)\b", re.I), "derive", "Insert geometry from another document"),
    (re.compile(r"\b(construction (line|geometry)|centerline)\b", re.I), "construction geometry", "Build reference geometry that is not part of the solid"),
    (re.compile(r"\b(offset|offset plane|offset surface)\b", re.I), "offset", "Offset a face, surface or plane"),
    (re.compile(r"\b(split|split body|trim)\b", re.I), "split", "Cut a body with a surface or plane"),
    (re.compile(r"\b(thicken)\b", re.I), "thicken", "Give a surface a wall thickness"),
    (re.compile(r"\b(thread|threads|cosmetic thread)\b", re.I), "thread", "Add a thread to a cylindrical face"),
    (re.compile(r"\b(configurations?|configurator)\b", re.I), "configuration", "Drive variants of the same part"),
    (re.compile(r"\b(variable studio|variable)\b", re.I), "variable", "Drive dimensions from a variable"),
    (re.compile(r"\b(appearance|colour|color)\b", re.I), "appearance", "Set the look of a face or part"),
    (re.compile(r"\b(section view|section view)\b", re.I), "section view", "Cut the view to see inside"),
    (re.compile(r"\b(measure|measurement)\b", re.I), "measure", "Check a distance or angle"),
    (re.compile(r"\b(translate|rotate|transform)\b", re.I), "transform", "Move or rotate existing geometry"),
    (re.compile(r"\b(automatic constraints?|redundant constraints?)\b", re.I), "constraint", "Constrain sketch geometry"),
    (re.compile(r"\b(project|project curve|use edge)\b", re.I), "project", "Project edges into the current sketch"),
    (re.compile(r"\b(assemble|assembly|bill of materials)\b", re.I), "assembly", "Put parts together in an assembly"),
]


def _progress(cb: ProgressFn | None, msg: str) -> None:
    if cb:
        try:
            cb(msg)
        except Exception:
            pass


def _is_url(raw: str) -> bool:
    return bool(re.match(r"^https?://", (raw or "").strip(), re.I))


def _is_youtube_url(link: str) -> bool:
    parsed = urlparse(link)
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.username or parsed.password:
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    if host in _YOUTUBE_HOSTS:
        return True
    return host.endswith(".youtube.com") and host.count(".") >= 2


def _clean_video_url(raw: str) -> str:
    """Normalize YouTube links (drop playlist/time junk that breaks some extractors)."""
    link = (raw or "").strip()
    if not link:
        return ""
    m = re.match(r"https?://(?:www\.)?youtu\.be/([\w-]{6,})", link, re.I)
    if m:
        return f"https://www.youtube.com/watch?v={m.group(1)}"
    m = re.search(r"[?&]v=([\w-]{6,})", link)
    if m:
        return f"https://www.youtube.com/watch?v={m.group(1)}"
    return link.split("&")[0].split("#")[0]


def ensure_video_path_in_dataset(path: str, dataset_root: Path | None) -> Path | None:
    """Return a video file under the agent dataset, or None if that path is missing.

    Paths that resolve outside the dataset raise ValueError before the file is
    opened. /video/learn must not read an arbitrary video on the machine.
    """
    text = str(path or "").strip()
    if not text:
        return None
    root = ensure_dataset(dataset_root)
    local = Path(text).expanduser()
    try:
        resolved = local.resolve()
        base = root.resolve()
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError("Local video must be inside the agent dataset folder") from exc
    except OSError as exc:
        raise ValueError("Video file path is not readable") from exc
    if not resolved.exists():
        return None
    if not resolved.is_file():
        raise ValueError("Video file path is not readable")
    if resolved.suffix.lower() not in _VIDEO_SUFFIXES:
        raise ValueError("Local video must be mp4, webm, mkv, avi, mov, or m4v")
    try:
        size = resolved.stat().st_size
    except OSError as exc:
        raise ValueError("Video file path is not readable") from exc
    if size > MAX_LOCAL_VIDEO_BYTES:
        raise ValueError("Local video is too large (over 4 GB)")
    return resolved


def _yt_dlp_cmd() -> list[str]:
    """Prefer the yt-dlp installed in this interpreter's environment."""
    try:
        import yt_dlp  # noqa: F401

        return [sys.executable, "-m", "yt_dlp"]
    except ImportError:
        pass
    exe = shutil.which("yt-dlp")
    if exe:
        return [exe]
    return [sys.executable, "-m", "yt_dlp"]


def _ffmpeg_bin() -> str | None:
    return shutil.which("ffmpeg")


def _ffprobe_bin() -> str | None:
    return shutil.which("ffprobe")


def fetch_video_source(
    *,
    url: str = "",
    path: str = "",
    dataset_root: Path | None = None,
    max_minutes: float = 6.0,
    on_progress: ProgressFn | None = None,
) -> tuple[Path, list[tuple[float, str]], str]:
    """Return (video_path, caption_cues, title). Downloads YouTube via yt-dlp when needed."""
    root = ensure_dataset(dataset_root)
    dest_dir = videos_dir(root) / "teacher"
    dest_dir.mkdir(parents=True, exist_ok=True)
    max_minutes = float(max(1.0, min(1200.0, max_minutes)))

    resolved = ensure_video_path_in_dataset(path, root)
    if resolved is not None:
        _progress(on_progress, f"Using local video {resolved.name}")
        cues: list[tuple[float, str]] = []
        for side in _caption_sidecars(resolved, root):
            cues = _read_caption_file(side)
            if cues:
                break
        return resolved, cues, resolved.stem

    link = _clean_video_url(url)
    if not link or not _is_url(link) or not _is_youtube_url(link):
        raise ValueError("Provide a YouTube URL or a local video file path")

    session = f"{int(time.time())}"
    out_tmpl = str(dest_dir / f"{session}.%(ext)s")
    ffmpeg = _ffmpeg_bin()
    _progress(
        on_progress,
        "Downloading video (yt-dlp)…" + ("" if ffmpeg else " [no ffmpeg: full file]"),
    )
    attempts: list[list[str]] = []
    base_cmd = _yt_dlp_cmd() + [
        "--no-playlist",
        "--newline",
        "--format",
        "bv*[height<=720]+ba/b[height<=720]/b",
        "--merge-output-format",
        "mp4",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs",
        "en.*",
        "--sub-format",
        "vtt/srt",
        "-o",
        out_tmpl,
    ]
    if ffmpeg:
        attempts.append(
            base_cmd
            + [
                "--download-sections",
                f"*0-{int(max_minutes * 60)}",
                "--force-keyframes-at-cuts",
                "--",
                link,
            ]
        )
    attempts.append(base_cmd + ["--", link])

    last_err = ""
    for i, cmd in enumerate(attempts):
        timeout = 900 if i == 0 else 420
        try:
            proc = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            del proc
            last_err = ""
            break
        except FileNotFoundError as exc:
            raise RuntimeError(
                "yt-dlp not found. Install it: pip install yt-dlp"
            ) from exc
        except subprocess.TimeoutExpired:
            last_err = "download timed out"
        except subprocess.CalledProcessError as exc:
            lines = (exc.stderr or exc.stdout or str(exc)).strip().splitlines()
            last_err = (lines[-1] if lines else str(exc))[:300]
        if i + 1 < len(attempts):
            _progress(on_progress, f"Retrying download ({last_err[:80]})…")
    if last_err:
        raise RuntimeError(_friendly_ytdlp_error(last_err))

    videos = sorted(
        dest_dir.glob(f"{session}.*"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    video_path = next(
        (p for p in videos if p.suffix.lower() in _VIDEO_SUFFIXES),
        None,
    )
    if video_path is None:
        raise RuntimeError("Download finished but no video file was found")

    cues = []
    for p in dest_dir.glob(f"{session}*"):
        if p.suffix.lower() in {".vtt", ".srt"}:
            cues = _read_caption_file(p)
            if cues:
                break
    return video_path, cues, video_path.stem


def _friendly_ytdlp_error(raw: str) -> str:
    low = (raw or "").lower()
    if "yt-dlp not found" in low or "no module named" in low:
        return "yt-dlp not found. Install it: pip install yt-dlp"
    if "unsupported url" in low:
        return "That URL is not a supported video page"
    if "private video" in low or "sign in" in low:
        return "The video is private or requires sign-in"
    if "video unavailable" in low:
        return "The video is unavailable (removed or region-locked)"
    if "timed out" in low or "timeout" in low:
        return "Download timed out — check your connection and retry"
    return (raw or "download failed")[:300]


def _caption_under_dataset(path: Path, root: Path) -> Path | None:
    try:
        resolved = path.resolve()
        resolved.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    if resolved.is_file():
        return resolved
    return None


def _caption_sidecars(video: Path, root: Path) -> list[Path]:
    """Caption files next to a local video, in preference order.

    yt-dlp names subtitle files ``<stem>.en.vtt`` (plus ``<stem>.en-orig.vtt``),
    while this only ever looked for ``<stem>.vtt``. Captions therefore never
    loaded for a downloaded lesson, so the narration half of concept
    extraction silently produced nothing. Language-suffixed files are found
    too, with a translation preferred over the auto-generated one.
    """
    wanted = {".vtt", ".srt", ".txt"}
    found: list[Path] = []
    for ext in (".vtt", ".srt", ".txt"):
        exact = _caption_under_dataset(video.with_suffix(ext), root)
        if exact is not None:
            found.append(exact)
    try:
        siblings = sorted(video.parent.glob(video.stem + ".*"))
    except OSError:
        siblings = []
    for path in siblings:
        if path.suffix.lower() in wanted and path not in found:
            if _caption_under_dataset(path, root) is not None:
                found.append(path)
    found.sort(
        key=lambda p: (
            0 if p.stem == video.stem else 1,
            1 if "-orig" in p.name else 0,
        )
    )
    return found


def _read_caption_file(path: Path) -> list[tuple[float, str]]:
    """Return [(t_seconds, text)] from vtt/srt/txt."""
    if not path.is_file():
        return []
    try:
        if path.stat().st_size > MAX_CAPTION_BYTES:
            return []
    except OSError:
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    cues: list[tuple[float, str]] = []
    ts_re = re.compile(
        r"(\d{1,2}:)?(\d{1,2}):(\d{2})[.,](\d{1,3})\s*-->\s*(\d{1,2}:)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
    )

    def _to_sec(h: str | None, m: str, s: str, ms: str) -> float:
        hours = int(h[:-1]) if h else 0
        return hours * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000.0

    i = 0
    full = ""
    while i < len(lines):
        line = lines[i].strip()
        m = ts_re.search(line)
        if m:
            t0 = _to_sec(m.group(1), m.group(2), m.group(3), m.group(4))
            body: list[str] = []
            i += 1
            while i < len(lines) and lines[i].strip():
                body.append(re.sub(r"<[^>]+>", "", lines[i]).strip())
                i += 1
            joined = " ".join(b for b in body if b)
            # Auto-captions roll a window: each cue repeats the tail of the
            # previous one and appends a few words ("…of Onshape CAD." then
            # "…of Onshape CAD. In this course…"). Storing every cue whole made
            # the transcript repeat itself 2–3x, so the character budget was
            # spent on the opening seconds and nothing taught later in a
            # segment ever reached the extractor. Keep only what is new.
            if joined and not _FILLER.search(joined):
                tail = full[-240:]
                overlap = 0
                for length in range(min(len(joined), 240), 0, -1):
                    if tail.endswith(joined[:length]):
                        overlap = length
                        break
                delta = joined[overlap:].strip()
                if delta:
                    cues.append((t0, delta))
                    full = f"{full} {delta}".strip()
            continue
        i += 1
    if cues:
        return cues
    # Plain transcript fallback.
    for i, line in enumerate(lines):
        line = line.strip()
        if line and not line.startswith("WEBVTT"):
            cues.append((float(i), line))
    return cues


def captions_window(
    cues: list[tuple[float, str]],
    t0: float,
    t1: float,
    *,
    limit: int = 40,
    max_chars: int = 8000,
) -> str:
    bits = [txt for t, txt in cues if t0 - 1.0 <= t <= t1 + 1.0]
    if not bits and cues:
        bits = [txt for _, txt in cues[:limit]]
    # Sample evenly instead of taking the head: a dense caption file would
    # otherwise describe only the first few seconds of the segment, so the
    # tools taught later in it never reached the extractor.
    if len(bits) > limit:
        step = len(bits) / float(limit)
        bits = [bits[int(i * step)] for i in range(limit)]
    return " ".join(bits)[:max_chars]


def _salvage_list_payload(text: str) -> dict[str, Any]:
    """Pull whole {...} entries out of a cut-off `"key": [ {...}, {...}, …` body.

    A 3B model asked for 32 concepts routinely stops mid-object; without this
    the whole reply used to be discarded and the segment yielded nothing.
    """
    m = re.search(r'"(\w+)"\s*:\s*\[', text)
    if not m:
        return {}
    key, body = m.group(1), text[m.end() :]
    found: list[dict[str, Any]] = []
    depth = 0
    start = -1
    in_str = False
    escaped = False
    for i, ch in enumerate(body):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    obj = json.loads(body[start : i + 1])
                except json.JSONDecodeError:
                    obj = None
                if isinstance(obj, dict) and obj.get("name"):
                    found.append(obj)
                start = -1
    return {key: found} if found else {}


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and data:
            return {"concepts": [c for c in data if isinstance(c, dict)]}
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        cand = m.group(0)
        if cand.count("{") > cand.count("}"):
            cand += "}" * (cand.count("{") - cand.count("}"))
        if cand.count("[") > cand.count("]"):
            cand += "]" * (cand.count("[") - cand.count("]"))
        cand = re.sub(r",\s*([}\]])", r"\1", cand)
        try:
            data = json.loads(cand)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return _salvage_list_payload(text)


def _concept_list(data: Any) -> list[dict[str, Any]]:
    """Accept {"concepts": [...]}, a bare list, or a single concept object."""
    if isinstance(data, list):
        raw = data
    elif isinstance(data, dict):
        raw = data.get("concepts")
        if isinstance(raw, dict):
            raw = [raw]
        elif not isinstance(raw, list):
            raw = [data] if data.get("name") else []
    else:
        return []
    return [c for c in raw if isinstance(c, dict) and str(c.get("name") or "").strip()]


def _concept_key(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def merge_concepts(concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedupe by normalized name; keep the richest card; preserve first-seen order."""
    out: list[dict[str, Any]] = []
    index: dict[str, int] = {}
    for raw in concepts:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        if not name or len(name) > 80:
            continue
        if not is_cad_concept(name, str(raw.get("summary") or "")):
            continue
        key = _concept_key(name)
        if key in index:
            keep = out[index[key]]
            if len(str(raw.get("summary") or "")) > len(str(keep.get("summary") or "")):
                keep["summary"] = raw.get("summary")
            if raw.get("steps") and not keep.get("steps"):
                keep["steps"] = raw.get("steps")
            if raw.get("when_to_use") and not keep.get("when_to_use"):
                keep["when_to_use"] = raw.get("when_to_use")
            continue
        index[key] = len(out)
        out.append(dict(raw))
    return out


def concepts_from_captions(
    cues: list[tuple[float, str]], *, t0: float, t1: float
) -> list[dict[str, Any]]:
    """Seed concepts from narration keywords when the vision model under-reports."""
    text = captions_window(cues, t0, t1, limit=400)
    if not text.strip():
        return []
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pattern, name, summary in _CAPTION_TECHNIQUES:
        if name.lower() in seen:
            continue
        if not pattern.search(text):
            continue
        seen.add(name.lower())
        found.append(
            {
                "name": name,
                "summary": summary,
                "when_to_use": f"When the lesson demonstrates {name}",
                "preconditions": ["sketch or part ready"],
                "aliases": [],
                "params": {},
                "t_start": float(t0),
                "t_end": float(t1),
                "steps": [],
            }
        )
    return found


def video_duration_sec(video_path: Path, *, fallback: float = 300.0) -> float:
    """Duration via ffprobe, else a conservative fallback."""
    ffprobe = _ffprobe_bin()
    if ffprobe:
        try:
            proc = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
            value = float((proc.stdout or "").strip().splitlines()[0])
            if value > 0:
                return value
        except (Exception, ValueError, IndexError):
            pass
    return float(fallback)


def sample_keyframes(
    video_path: Path,
    *,
    max_minutes: float = 6.0,
    max_frames: int | None = None,
    dataset_root: Path | None = None,
) -> list[tuple[float, Any]]:
    """Extract frames with ffmpeg at a sparse interval. Returns [(t_rel, PIL image)].

    Falls back to OpenCV when ffmpeg is missing.
    """
    root = ensure_dataset(dataset_root)
    out_dir = frames_dir(root) / video_path.stem
    if max_frames is None:
        max_frames = int(max(24, min(MAX_FRAMES, 24 + float(max_minutes) * 0.7)))
    duration = min(video_duration_sec(video_path), float(max_minutes) * 60.0)
    if duration <= 0:
        raise RuntimeError("Cannot read video duration")
    interval = max(1.0, duration / float(max_frames))

    ffmpeg = _ffmpeg_bin()
    if ffmpeg:
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("frame_*.jpg"):
            try:
                old.unlink()
            except OSError:
                pass
        pattern = str(out_dir / "frame_%04d.jpg")
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"fps=1/{interval:.4f},scale='min(960,iw)':-2",
            "-q:v",
            "4",
            "-frames:v",
            str(max_frames),
            "-y",
            pattern,
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                timeout=600,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"ffmpeg frame extraction failed: {str(exc)[:200]}") from exc
        frames: list[tuple[float, Any]] = []
        from PIL import Image

        for i, path in enumerate(sorted(out_dir.glob("frame_*.jpg"))):
            t = i * interval
            try:
                img = Image.open(path).convert("RGB")
            except OSError:
                continue
            frames.append((t, img))
        if not frames:
            raise RuntimeError("No frames could be read from the video")
        return frames

    # OpenCV fallback (no ffmpeg on PATH).
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Need ffmpeg or opencv-python to extract video frames") from exc
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        frames = []
        t = 0.0
        while t <= duration and len(frames) < max_frames:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                t += interval
                continue
            frames.append((t, frame))
            t += interval
        return frames
    finally:
        cap.release()


def _concept_budget(max_minutes: float) -> int:
    minutes = float(max(1.0, max_minutes))
    return int(max(24, min(400, round(minutes * 3.0))))


def _segment_length_sec(max_minutes: float) -> float:
    """Short segments = more vision passes = more concepts.

    This used to return the WHOLE video for anything <= 12 min, so the default
    6-minute lesson produced exactly one model call — a hard ceiling of ~8
    concepts no matter how long the tutorial was.
    """
    minutes = float(max(1.0, max_minutes))
    if minutes <= 12:
        return 2.0 * 60.0
    if minutes <= 60:
        return 5.0 * 60.0
    if minutes <= 180:
        return 8.0 * 60.0
    return 12.0 * 60.0


def extract_concept_candidates(
    *,
    model: str,
    goal: str,
    captions: str,
    frames: list[tuple[float, Any]],
    on_progress: ProgressFn | None = None,
) -> list[dict[str, Any]]:
    _progress(on_progress, "Finding concepts in the lesson…")
    images: list[str] = []
    # 6 evenly-spaced frames instead of 4: a dense tutorial shows a different
    # tool every few seconds and 4 shots skipped most of them.
    for _t, frame in frames[:: max(1, len(frames) // 6)][:6]:
        images.append(encode_jpeg_b64(frame, max_side=512, quality=70))
    user = (
        f"Student goal (optional focus): {goal or '(general CAD techniques)'}\n\n"
        f"Captions / narration excerpt:\n{captions[:6000]}\n\n"
        "List EVERY distinct Onshape technique taught in this segment "
        "(not only sketch/extrude/fillet — include revolve, sweep, loft, shell, draft, "
        "pattern, mirror, boolean, mates, sheet metal, construction geometry…)."
    )
    raw = ""
    try:
        raw = chat_vision_json(
            model,
            system=CONCEPT_LIST_SYSTEM,
            user_text=user,
            images_b64=images,
            num_predict=2400,
        )
    except OllamaError as exc:
        _progress(on_progress, f"Vision failed ({str(exc)[:60]}); learning from narration…")
        raw = ""
    concepts = _concept_list(_parse_json_object(raw))
    if not concepts:
        # The vision reply was prose, truncated, or malformed — the narration
        # still names the tools, so retry as a plain text call rather than
        # silently returning zero concepts for this segment.
        _progress(on_progress, "Vision reply unusable — re-reading the narration…")
        try:
            raw = chat_text_json(model, system=CONCEPT_LIST_SYSTEM, user_text=user, num_predict=2400)
        except OllamaError:
            return []
        concepts = _concept_list(_parse_json_object(raw))
    return merge_concepts(concepts)


def sanitize_concept_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep transferable steps; drop video-pixel coordinates."""
    out: list[dict[str, Any]] = []
    for raw in steps:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("action") or "").lower()
        if kind not in {"key", "hotkey", "type", "wait", "click"}:
            continue
        if kind == "click":
            # Coordinates from a video do not transfer; keep as targetless hint only.
            target = str(raw.get("target") or raw.get("reason") or "")
            if not target:
                continue
            out.append({"action": "click", "target": target[:64], "reason": target[:120]})
            continue
        if kind == "wait":
            sec = float(raw.get("seconds") or 0.4)
            out.append({"action": "wait", "seconds": float(max(0.15, min(2.0, sec)))})
            continue
        step: dict[str, Any] = {
            "action": kind,
            "keys": [str(k) for k in (raw.get("keys") or [])][:3],
            "text": str(raw.get("text") or "")[:64],
            "reason": str(raw.get("reason") or "concept step")[:120],
        }
        out.append(step)
        if kind == "type":
            out.append({"action": "wait", "seconds": 0.35, "reason": "wait for search results"})
    return out[:12]


def _ssearch_fallback(name: str) -> list[dict[str, Any]]:
    """Safe default: S shortcut search for the concept name."""
    return sanitize_concept_steps(
        [
            {"action": "key", "keys": ["s"], "reason": f"open shortcut search for {name}"},
            {"action": "type", "text": name[:48], "reason": f"find {name}"},
            {"action": "key", "keys": ["enter"], "reason": f"run {name}"},
        ]
    )


# --- vetted recipes --------------------------------------------------------
# The vision model's generic answer for EVERY concept used to be
# "press S, type the concept name, Enter" — which is meaningless for procedures
# ("create sketch", "sign in", "loading workspaces"). The planner then copied
# those keystroke strings verbatim into its plans. These are the real control
# sequences instead; they are hints, never invented screen coordinates.
def _c(target: str, reason: str) -> dict[str, Any]:
    return {"action": "click", "target": target, "reason": reason}


def _k(*keys: str, reason: str) -> dict[str, Any]:
    return {"action": "key", "keys": list(keys), "reason": reason}


def _t(field: str, reason: str) -> dict[str, Any]:
    """A value to enter — modelled as a click on the field, NEVER as text.

    "type distance" used to be rendered into plans as a literal keystroke,
    so the agent typed the word "distance" into Onshape. The real value comes
    from the goal's plan step ("extrude 20 mm").
    """
    return {"action": "click", "target": f"{field} field", "reason": reason}


def _finish(reason: str = "accept the feature") -> dict[str, Any]:
    return _c("green check mark", reason)


def _feature(name: str, extra: list[dict[str, Any]], *, select: str = "") -> list[dict[str, Any]]:
    steps = [_c(name, f"open the {name} tool from the top toolbar")]
    if select:
        steps.append(_c(select, select))
    steps.extend(extra)
    steps.append(_finish())
    return steps


_CANON_STEPS: dict[str, list[dict[str, Any]]] = {
    "sketch": [
        _c("Sketch button", "start a new sketch from the top toolbar"),
        _c("Top plane", "pick the sketch plane (Top/Front/Right)"),
    ],
    "create sketch": [
        _c("Sketch button", "start a new sketch from the top toolbar"),
        _c("Top plane", "pick the sketch plane (Top/Front/Right)"),
    ],
    "new sketch": [
        _c("Sketch button", "start a new sketch from the top toolbar"),
        _c("Top plane", "pick the sketch plane (Top/Front/Right)"),
    ],
    "line": [_k("l", reason="line tool inside a sketch"), _c("start point", "start the line"), _c("end point", "end the line")],
    "rectangle": [_k("r", reason="rectangle tool inside a sketch"), _c("first corner", "first corner"), _c("opposite corner", "opposite corner")],
    "circle": [_k("c", reason="circle tool inside a sketch"), _c("center point", "circle centre"), _c("radius point", "set the radius")],
    "arc": [_c("3 point arc", "pick the arc tool in the sketch toolbar"), _c("start point", "arc start"), _c("end point", "arc end"), _c("radius point", "arc radius")],
    "spline": [_c("Spline", "pick the spline tool in the sketch toolbar"), _c("control point", "place spline points")],
    "dimension": [_k("d", reason="dimension tool inside a sketch"), _c("sketch entity", "select what to dimension"), _c("dimension placement", "place the dimension"), _t("value", "type the value"), _k("enter", reason="commit the value")],
    "constraint": [_c("sketch entity", "select the geometry to constrain"), _c("constraint in the sketch toolbar", "apply the constraint")],
    "coincident": [_c("first point", "select the first point"), _c("second point", "select the second point"), _c("Coincident", "apply Coincident in the sketch toolbar")],
    "parallel": [_c("first line", "select the first line"), _c("second line", "select the second line"), _c("Parallel", "apply Parallel in the sketch toolbar")],
    "perpendicular": [_c("first line", "select the first line"), _c("second line", "select the second line"), _c("Perpendicular", "apply Perpendicular in the sketch toolbar")],
    "tangent": [_c("curve", "select the curve"), _c("line", "select the line"), _c("Tangent", "apply Tangent in the sketch toolbar")],
    "symmetric": [_c("first entity", "select the first entity"), _c("second entity", "select the second entity"), _c("symmetry line", "select the symmetry line"), _c("Symmetric", "apply Symmetric in the sketch toolbar")],
    "extrude": _feature("Extrude", [_t("distance", "type the extrude distance"), _k("enter", reason="commit the distance")]),
    "revolve": _feature("Revolve", [_c("sketch profile", "select the profile to revolve"), _c("revolution axis", "select the axis")]),
    "sweep": _feature("Sweep", [_c("sketch profile", "select the sweep profile"), _c("path curve", "select the sweep path")]),
    "loft": _feature("Loft", [_c("first section", "select the first section"), _c("second section", "select the second section")]),
    "fillet": _feature("Fillet", [_c("edge to round", "select the edges"), _t("radius", "type the fillet radius"), _k("enter", reason="commit the radius")]),
    "face fillet": _feature("Fillet", [_c("first face", "select the first face"), _c("second face", "select the second face")]),
    "chamfer": _feature("Chamfer", [_c("edge to break", "select the edges"), _t("distance", "type the chamfer distance"), _k("enter", reason="commit the distance")]),
    "shell": _feature("Shell", [_c("face to remove", "select the face to open"), _t("thickness", "type the wall thickness"), _k("enter", reason="commit the thickness")]),
    "draft": _feature("Draft", [_c("neutral plane", "select the neutral plane"), _t("angle", "type the draft angle"), _k("enter", reason="commit the angle")]),
    "hole": _feature("Hole", [_c("hole position", "select the sketch point for the hole"), _c("hole type options", "choose the hole type and size")]),
    "rib": _feature("Rib", [_c("rib profile", "select the rib profile"), _t("thickness", "type the rib thickness"), _k("enter", reason="commit the thickness")]),
    "thread": _feature("Thread", [_c("cylindrical face", "select the cylindrical face"), _c("thread size", "choose the thread size")]),
    "pattern": _feature("Linear pattern", [_c("feature to pattern", "select the features to repeat"), _t("count", "set the instance count and spacing"), _k("enter", reason="commit the pattern")]),
    "linear pattern": _feature("Linear pattern", [_c("feature to pattern", "select the features to repeat"), _t("count", "set the instance count and spacing"), _k("enter", reason="commit the pattern")]),
    "circular pattern": _feature("Circular pattern", [_c("feature to pattern", "select the features to repeat"), _c("pattern axis", "select the rotation axis"), _t("count", "set the instance count"), _k("enter", reason="commit the pattern")]),
    "rectangular pattern": _feature("Rectangular pattern", [_c("feature to pattern", "select the features to repeat"), _t("count", "set counts and spacing"), _k("enter", reason="commit the pattern")]),
    "mirror": _feature("Mirror", [_c("feature to mirror", "select the features to mirror"), _c("mirror plane", "select the mirror plane")]),
    "plane": _feature("Plane", [_c("reference geometry", "choose what the plane is based on"), _t("offset", "type the offset distance")]),
    "offset plane": _feature("Plane", [_c("reference face", "select the reference face"), _t("offset", "type the offset distance")]),
    "boolean": _feature("Boolean", [_c("target body", "select the target bodies"), _c("tool body", "select the tool body")]),
    "union": _feature("Boolean", [_c("target body", "select the target bodies"), _c("tool body", "select the tool body")]),
    "subtract": _feature("Boolean", [_c("target body", "select the target body"), _c("tool body", "select the body to subtract")]),
    "intersect": _feature("Boolean", [_c("target body", "select the first body"), _c("tool body", "select the second body")]),
    "split": _feature("Split", [_c("body to split", "select the body"), _c("split tool", "select the splitting surface or plane")]),
    "project": [_c("Use", "click Use/Project in the sketch toolbar"), _c("edge to project", "select the edges to project")],
    "trim": [_c("Trim", "click Trim in the sketch toolbar"), _c("segment to remove", "drag across the segments to trim")],
    "create document": [_c("Create", "click Create on the documents list"), _c("Document", "pick Document"), _t("part name", "type the document name"), _k("enter", reason="create it")],
    "mate": _feature("Fastened mate", [_c("first mate connector", "select the first mate connector"), _c("second mate connector", "select the second mate connector")]),
    "fastened mate": _feature("Fastened mate", [_c("first mate connector", "select the first mate connector"), _c("second mate connector", "select the second mate connector")]),
}


def _norm_name(name: str) -> str:
    n = re.sub(r"\s+", " ", (name or "").strip().lower())
    return n[:-1] if n.endswith("s") and len(n) > 3 else n


def canonical_steps(name: str) -> list[dict[str, Any]]:
    """Vetted control sequence for a technique, or [] when there is none.

    Handles lesson-specific spellings ("extrude blind" -> the extrude recipe).
    """
    n = _norm_name(name)
    if not n:
        return []
    for key in (n, n.replace(" tool", ""), n.replace(" tools", "")):
        hit = _CANON_STEPS.get(key)
        if hit:
            return sanitize_concept_steps(hit)
    best = ""
    for key in _CANON_STEPS:
        if len(key) > len(best) and re.search(rf"\b{re.escape(key)}\b", n):
            best = key
    return sanitize_concept_steps(_CANON_STEPS[best]) if best else []


def is_onshape_command(name: str) -> bool:
    """True when the name is a single Onshape command S-search can launch."""
    n = _norm_name(name).replace(" tool", "")
    if not n:
        return False
    if " " in n:
        return n in _COMMAND_NAMES
    return n in _COMMAND_NAMES


_COMMAND_NAMES = {c.strip().lower() for c in _KNOWN_TECHNIQUES.split(",") if c.strip()}


def default_steps(name: str) -> tuple[list[dict[str, Any]], str]:
    """Best non-LLM recipe: vetted sequence, else S-search for a real command.

    Returns (steps, source) — steps is [] when nothing trustworthy exists, so
    the planner never gets a made-up keystroke recipe to imitate.
    """
    canon = canonical_steps(name)
    if canon:
        return canon, "curated"
    if is_onshape_command(name):
        return _ssearch_fallback(_norm_name(name)), "search"
    return [], "none"


def _is_generic_search_recipe(steps: list[dict[str, Any]], name: str) -> bool:
    """True when the model only echoed "press S and type the concept name"."""
    if not steps:
        return True
    if any(str(s.get("action")) == "click" for s in steps):
        return False
    n = _norm_name(name)
    typed = [str(s.get("text") or "").strip().lower() for s in steps if s.get("action") == "type"]
    return bool(typed) and any(t == n or t == n.replace(" tool", "") for t in typed)


_STOP_WORDS = {
    "the", "and", "for", "with", "from", "that", "this", "use", "into", "when",
    "your", "make", "tool", "tools", "part", "parts", "step", "steps", "new",
}


def _recipe_mentions_concept(steps: list[dict[str, Any]], name: str) -> bool:
    """True when the recipe actually talks about the technique it is for.

    The vision model happily returned a 12-step "Mirror -> Pattern -> Boolean"
    script for an *assembly* concept. A recipe that never mentions the
    technique is a hallucination, not a lesson — drop it rather than let the
    planner copy it.
    """
    words = {
        w
        for w in re.findall(r"[a-z]+", _norm_name(name))
        if len(w) >= 3 and w not in _STOP_WORDS
    }
    if not words:
        return True
    said = " ".join(
        f"{s.get('target') or ''} {s.get('reason') or ''} {s.get('text') or ''} "
        f"{' '.join(str(k) for k in (s.get('keys') or []))}"
        for s in steps
    ).lower()
    return any(w in said for w in words)


def fill_concept_steps(
    concept: dict[str, Any],
    *,
    model: str,
    frames: list[tuple[float, Any]],
    cues: list[tuple[float, str]],
    on_progress: ProgressFn | None = None,
) -> dict[str, Any]:
    name = str(concept.get("name") or "concept")
    _progress(on_progress, f"Teaching steps for '{name}'…")
    # Vetted recipe first: it is known-good for this Onshape build, whereas the
    # 3B model answered with a single useless step for most concepts.
    canon = canonical_steps(name)
    if canon:
        out = dict(concept)
        out["steps"] = canon
        out["steps_source"] = "curated"
        return out
    t0 = float(concept.get("t_start") or 0.0)
    t1 = float(concept.get("t_end") or (t0 + 30.0))
    if t1 <= t0:
        t1 = t0 + 30.0
    window = [f for t, f in frames if t0 - 2.0 <= t <= t1 + 2.0]
    if not window:
        window = [f for _, f in frames[:2]]
    images = [encode_jpeg_b64(f, max_side=512, quality=70) for f in window[:1]]
    cap = captions_window(cues, t0, t1)
    user = (
        f"Concept name: {name}\n"
        f"Summary: {concept.get('summary') or ''}\n"
        f"When to use: {concept.get('when_to_use') or ''}\n"
        f"Preconditions: {concept.get('preconditions') or []}\n"
        f"Narration around this concept:\n{cap}\n\n"
        "Produce canonical GUI steps for applying this on the USER's current part."
    )
    steps: list[dict[str, Any]] = []
    for attempt in ("vision", "text"):
        try:
            if attempt == "vision":
                raw = chat_vision_json(
                    model,
                    system=CONCEPT_STEPS_SYSTEM,
                    user_text=user,
                    images_b64=images,
                    num_predict=400,
                )
            else:
                raw = chat_text_json(
                    model, system=CONCEPT_STEPS_SYSTEM, user_text=user, num_predict=400
                )
            data = _parse_json_object(raw)
            steps_raw = data.get("steps") if isinstance(data, dict) else None
            steps = sanitize_concept_steps(list(steps_raw) if isinstance(steps_raw, list) else [])
            if steps:
                break
        except OllamaError:
            steps = []
    source = "video"
    real = [s for s in steps if str(s.get("action")) != "wait"]
    if (
        len(real) < 2
        or _is_generic_search_recipe(steps, name)
        or not _recipe_mentions_concept(steps, name)
    ):
        # A one-step answer, a bare search stub, or a script about some other
        # feature is not a procedure — fall back to something usable.
        steps, source = default_steps(name)
    out = dict(concept)
    out["steps"] = steps
    out["steps_source"] = source
    return out


def learn_concepts_from_video(
    *,
    url: str = "",
    path: str = "",
    goal: str = "",
    model: str = "qwen2.5vl",
    max_minutes: float = 6.0,
    dataset_root: Path | None = None,
    on_progress: ProgressFn | None = None,
    cancelled: "callable[[], bool] | None" = None,
) -> dict[str, Any]:
    """Full pipeline: fetch -> keyframes -> segmented concept cards with steps.

    ``cancelled`` is polled between segments and between step-teaching cards;
    raising StopIteration lets the UI Stop button end a long lesson promptly
    instead of after every remaining Ollama call.
    """
    started = time.time()
    max_minutes = float(max(1.0, min(1200.0, max_minutes)))
    video_path, cues, title = fetch_video_source(
        url=url,
        path=path,
        dataset_root=dataset_root,
        max_minutes=max_minutes,
        on_progress=on_progress,
    )
    _progress(on_progress, "Sampling keyframes…")
    frames = sample_keyframes(
        video_path, max_minutes=max_minutes, dataset_root=dataset_root
    )
    if not frames:
        raise RuntimeError("No frames could be read from the video")

    duration = min(float(frames[-1][0]) + 1.0, float(max_minutes) * 60.0)
    seg_len = _segment_length_sec(max_minutes)
    segments: list[tuple[float, float]] = []
    t = 0.0
    while t < duration - 0.5:
        segments.append((t, min(t + seg_len, duration)))
        t += seg_len
    if not segments:
        segments = [(0.0, duration)]

    all_candidates: list[dict[str, Any]] = []
    for si, (t0, t1) in enumerate(segments):
        if cancelled is not None and cancelled():
            raise StopIteration("Video learning stopped by user")
        _progress(
            on_progress,
            f"Scanning segment {si + 1}/{len(segments)} ({t0 / 60:.0f}–{t1 / 60:.0f} min)…",
        )
        seg_frames = [(tt, f) for tt, f in frames if t0 - 0.5 <= tt <= t1 + 0.5]
        if not seg_frames:
            mid = 0.5 * (t0 + t1)
            seg_frames = sorted(frames, key=lambda row: abs(row[0] - mid))[:4]
        cap_text = captions_window(cues, t0, t1, limit=160, max_chars=6000)
        if not cap_text:
            focus = (goal or "").strip() or title
            cap_text = (
                f"(no captions) Video title: {title}. Segment {si + 1}/{len(segments)}. "
                f"Goal focus: {focus}. List EVERY distinct Onshape technique shown "
                f"(extrude, revolve, sweep, loft, fillet, chamfer, shell, draft, pattern, "
                f"mirror, hole, rib, boolean, mates, sheet metal…)."
            )
        cands = extract_concept_candidates(
            model=model,
            goal=goal,
            captions=cap_text,
            frames=seg_frames,
            on_progress=on_progress,
        )
        for c in cands:
            c = dict(c)
            if float(c.get("t_start") or 0) <= 0 and float(c.get("t_end") or 0) <= 0:
                c["t_start"] = t0
                c["t_end"] = t1
            all_candidates.append(c)
        for c in concepts_from_captions(cues, t0=t0, t1=t1):
            all_candidates.append(c)

    candidates = merge_concepts(all_candidates)
    if not candidates:
        candidates = [
            {
                "name": (title or "modeling technique")[:48],
                "summary": f"Technique inferred from {title}",
                "when_to_use": "When refining a part",
                "preconditions": ["part ready"],
                "aliases": [],
                "t_start": 0.0,
                "t_end": duration,
                "steps": [],
            }
        ]

    budget = _concept_budget(max_minutes)
    filled: list[dict[str, Any]] = []
    take = candidates[:budget]
    # LLM step-teaching is the slow part — cap it so a long lesson stays tractable.
    step_llm_cap = min(len(take), 64)
    for i, card in enumerate(take):
        if cancelled is not None and cancelled():
            raise StopIteration("Video learning stopped by user")
        card = dict(card)
        card["video"] = str(video_path.name)
        if i < step_llm_cap:
            _progress(on_progress, f"Teaching {i + 1}/{len(take)} · {card.get('name')}")
            filled.append(
                fill_concept_steps(
                    card,
                    model=model,
                    frames=frames,
                    cues=cues,
                    on_progress=on_progress,
                )
            )
        else:
            if not card.get("steps"):
                steps, source = default_steps(str(card.get("name") or "technique"))
                card["steps"] = steps
                card["steps_source"] = source
            filled.append(card)

    merged = merge_concepts(filled)
    names = ", ".join(str(c.get("name")) for c in merged[:24])
    _progress(on_progress, f"Learned {len(merged)} concept(s): {names}")
    _prune_teacher_videos(video_path, dataset_root)
    return {
        "ok": True,
        "video": str(video_path),
        "title": title,
        "concepts": merged,
        "elapsed_sec": time.time() - started,
        "segments": len(segments),
    }


def _prune_teacher_videos(current: Path, dataset_root: Path | None, keep: int = 2) -> None:
    """Downloaded tutorials are only needed for sampling — keep the newest `keep`.

    Extracted keyframes get the same treatment, otherwise the frames folder
    grows without bound after every lesson.
    """
    try:
        folder = current.parent
        files = [p for p in folder.iterdir() if p.is_file() and p != current]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for path in files[keep:]:
            path.unlink()
    except OSError:
        pass
    try:
        frames_root = frames_dir(dataset_root)
        dirs = [p for p in frames_root.iterdir() if p.is_dir()]
        dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for old in dirs[keep:]:
            for path in old.glob("frame_*.jpg"):
                try:
                    path.unlink()
                except OSError:
                    pass
            try:
                old.rmdir()
            except OSError:
                pass
    except OSError:
        pass
