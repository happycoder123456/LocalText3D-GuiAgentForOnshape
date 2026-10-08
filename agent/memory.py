"""Cross-session memory: recorded episodes, distilled skills, and video concepts.

Storage is JSONL in the local agent dataset — never uploaded anywhere.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from agent.paths import ensure_dataset, episodes_path, memory_path

_TOKEN_RE = re.compile(r"[a-z0-9]+")
MAX_JSONL_LINE_BYTES = 1_048_576
MAX_MEMORY_FILE_BYTES = 24_000_000
KEEP_EPISODES = 400
KEEP_MEMORY_ROWS = 6000


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def goals_match(a: str, b: str, *, min_overlap: float = 0.34) -> bool:
    """True if two goal strings refer to the same skill."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    if ta == tb:
        return True
    al = (a or "").strip().lower()
    bl = (b or "").strip().lower()
    if al and bl and (al in bl or bl in al) and min(len(al), len(bl)) >= 4:
        return True
    return _overlap(ta, tb) >= min_overlap


def action_signature(action: dict[str, Any] | None) -> str:
    """Compact fingerprint for anti-repeat / ban lists."""
    if not isinstance(action, dict):
        return ""
    kind = str(action.get("action") or "").lower()
    if kind in {"click", "dblclick", "drag"}:
        x = int(float(action.get("x") or 0))
        y = int(float(action.get("y") or 0))
        # NOTE: target text is deliberately excluded — models alternate labels
        # ("Sketch button" / "Top plane") for the SAME point, which used to
        # defeat anti-repeat and let the click loop run forever.
        return f"{kind}:{x // 8}:{y // 8}"
    if kind in {"key", "hotkey"}:
        keys = action.get("keys") or []
        if isinstance(keys, str):
            keys = [keys]
        norm = ",".join(str(k).lower() for k in keys[:3])
        return f"{kind}:{norm}"
    if kind == "type":
        return f"type:{(action.get('text') or '')[:32].lower()}"
    if kind == "wait":
        return "wait"
    if kind == "stop":
        return "stop"
    return kind


def concept_key(text: str) -> str:
    """Exact technique identity for video concepts (no fuzzy substring merge)."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


# Website-navigation / channel filler that a "CAD tutorial" VLM loves to emit
# ("create an account", "sign in", "manage projects", "loading workspaces").
# Those cards used to occupy the memory slots a real technique should have had.
_NON_CAD_RE = re.compile(
    r"\b("
    r"account|sign[ -]?in|log[ -]?in|sign[ -]?up|register|subscribe|channel|playlist|"
    r"intro|outro|welcome back|thanks for watching|like (and )?(this|comment)|bell icon|"
    r"manage projects?|loading workspaces?|dashboard|homepage|pricing|free plan|"
    r"education (account|license|plan)|password|username|browser|website|web page|"
    r"video|chapter|timestamps?|playlist|download the app|install"
    r")\b",
    re.I,
)
# Anything that names real CAD vocabulary is always kept.
_CAD_HINT_RE = re.compile(
    r"\b("
    r"sketch|extrude|revolv|sweep|loft|fillet|chamfer|shell|draft|pattern|mirror|"
    r"hole|rib|boolean|mate|assembly|plane|constraint|dimension|feature|part studio|"
    r"sheet metal|geometry|profile|solid|surface|radius|edge|face|body|rectangl|"
    r"circle|arc|spline|polygon|slot|line|point|text|trim|project|offset|split|"
    r"thicken|thread|revolve|workplane|tool"
    r")\b",
    re.I,
)


def is_cad_concept(name: str, summary: str = "") -> bool:
    """False for channel/website filler that must never reach a planner prompt.

    Deliberately conservative: only blocks a name that carries navigation
    vocabulary AND no CAD vocabulary, so real techniques always survive.
    """
    text = f"{name or ''} {summary or ''}".strip()
    if not text.strip():
        return False
    # A bare video id/timestamp ("1791174595") is what the learn pipeline fell
    # back on when extraction produced nothing — it is not a technique, and it
    # used to be stored and then served to the planner as one.
    if not re.search(r"[a-z]", str(name or ""), re.I):
        return False
    if _CAD_HINT_RE.search(str(name or "")):
        return True
    return not _NON_CAD_RE.search(text)


# "make me a simple cube" shares no vocabulary with "extrude", so pure lexical
# matching left the modeling techniques out of the prompt entirely (only the
# sketch primitives happened to surface, by file order). Map goal phrasing to
# the technique that actually delivers it.
_GOAL_HINTS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"\b(cube|box|block|bracket|plate|prism|base|solid|body)\b", re.I), ("sketch", "extrude")),
    (re.compile(r"\b(round|rounding|smooth|rounded|fillet|fillets|chamfer|edge|edges|corner|corners)\b", re.I), ("fillet", "chamfer")),
    (re.compile(r"\b(hollow|wall|walls|thin|enclosure|case|housing|cavity)\b", re.I), ("shell",)),
    (re.compile(r"\b(hole|holes|bolt|screw|drill|bore)\b", re.I), ("hole",)),
    (re.compile(r"\b(spin|wheel|cylinder|axle|knob|rotate|axis)\b", re.I), ("revolve",)),
    (re.compile(r"\b(tube|pipe|path|channel|duct|follow)\b", re.I), ("sweep",)),
    (re.compile(r"\b(loft|blend|taper|draft|angled|angle)\b", re.I), ("loft", "draft")),
    (re.compile(r"\b(repeat|array|copies|pattern|grid|row)\b", re.I), ("pattern", "circular pattern")),
    # "4 holes in a circle" shares no vocabulary with "pattern", so the
    # learned patterning recipe never surfaced for exactly that phrasing.
    (re.compile(r"\b(in a circle|circular|circle of|equally spaced|ring of|around)\b", re.I), ("circular pattern", "pattern")),
    (re.compile(r"\b(symmetr|mirror)\b", re.I), ("mirror",)),
    (re.compile(r"\b(constrain|constrained|dimension|constraint)\b", re.I), ("dimension", "constraint")),
    (re.compile(r"\b(assembly|assemble|mate|attach)\b", re.I), ("mate", "fastened mate")),
)


def goal_technique_hints(goal: str) -> list[str]:
    """Ordered technique names this goal most likely needs."""
    text = goal or ""
    out: list[str] = []
    for pattern, names in _GOAL_HINTS:
        if pattern.search(text):
            for name in names:
                if name not in out:
                    out.append(name)
    return out


_path_locks: dict[str, threading.RLock] = {}
_path_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.RLock:
    key = str(path)
    with _path_locks_guard:
        lock = _path_locks.get(key)
        if lock is None:
            lock = threading.RLock()
            _path_locks[key] = lock
        return lock


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return rows
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False)
    if len(line.encode("utf-8")) > MAX_JSONL_LINE_BYTES:
        raise ValueError("memory row too large")
    with _lock_for(path):
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _rewrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with _lock_for(path):
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(path)


def _prune_file(path: Path, keep: int) -> None:
    """Cap a JSONL file at `keep` rows (oldest dropped). Cheap enough at our scale."""
    try:
        size = path.stat().st_size
    except OSError:
        return
    if size <= MAX_MEMORY_FILE_BYTES // 2:
        return
    rows = _read_jsonl(path)
    if len(rows) > keep:
        _rewrite_jsonl(path, rows[-keep:])


def _steps_recipe(steps: Any, limit: int = 6) -> str:
    """Compact 's -> "line" -> enter' rendering of a technique's key recipe."""
    out: list[str] = []
    for s in (steps or [])[: limit * 4]:
        if not isinstance(s, dict) or len(out) >= limit:
            continue
        kind = str(s.get("action") or "")
        if kind == "key":
            keys = s.get("keys") or []
            if keys:
                out.append(str(keys[0]))
        elif kind == "hotkey":
            keys = s.get("keys") or []
            if keys:
                out.append("+".join(str(k) for k in keys[:3]))
        elif kind == "type":
            txt = str(s.get("text") or "").strip()
            if txt:
                out.append(f'"{txt}"')
        elif kind == "click":
            target = str(s.get("target") or "").strip()
            out.append(f"click({target})" if target else "click")
        elif kind == "drag":
            out.append("drag")
    return " -> ".join(out)


class AgentMemory:
    def __init__(self, dataset_root: Path | None = None):
        self.root = ensure_dataset(dataset_root)
        self._lock = threading.RLock()

    @staticmethod
    def effective_steps(concept: dict[str, Any]) -> list[dict[str, Any]]:
        """Steps to render for a concept, preferring a vetted recipe.

        Rows written before recipes existed carry a placeholder like
        `s -> "rectangle" -> enter`, which the planner copied verbatim and then
        repeated against an unchanged screen. A vetted recipe always wins when
        one exists. Imported lazily: video_teacher imports this module.
        """
        stored = [s for s in (concept.get("steps") or []) if isinstance(s, dict)]
        name = str(concept.get("name") or "")
        try:
            from agent.video_teacher import default_steps

            canon, _source = default_steps(name)
        except Exception:
            return stored
        if canon:
            return canon
        return stored

    # -- episodes ----------------------------------------------------------

    def add_episode(self, episode: dict[str, Any]) -> None:
        """Persist a recorded demo: {goal, actions, screenshots, source, t}."""
        with self._lock:
            row = {
                "t": time.time(),
                "goal": str(episode.get("goal") or "")[:400],
                "source": str(episode.get("source") or "record")[:32],
                "actions": [a for a in (episode.get("actions") or []) if isinstance(a, dict)][:2000],
                "screenshots": [str(s) for s in (episode.get("screenshots") or [])][:2000],
                "success": bool(episode.get("success", False)),
            }
            _append_jsonl(episodes_path(self.root), row)
            _prune_file(episodes_path(self.root), KEEP_EPISODES)

    def episodes(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = _read_jsonl(episodes_path(self.root))
        return rows[-max(1, int(limit)):]

    def last_episode(self) -> dict[str, Any] | None:
        rows = self.episodes(limit=1)
        return rows[0] if rows else None

    # -- memory rows ---------------------------------------------------------

    def add_skill(self, *, goal: str, actions: list[dict[str, Any]], source: str = "agent_plan") -> dict[str, Any]:
        with self._lock:
            row = {
                "kind": "skill",
                "source": source[:32],
                "goal": str(goal)[:400],
                "actions": [a for a in actions if isinstance(a, dict)][:500],
                "t": time.time(),
                "success": 1,
                "uses": 1,
            }
            _append_jsonl(memory_path(self.root), row)
            _prune_file(memory_path(self.root), KEEP_MEMORY_ROWS)
            return row

    def add_concept(
        self,
        *,
        name: str,
        summary: str,
        when_to_use: str = "",
        preconditions: list[str] | None = None,
        steps: list[dict[str, Any]] | None = None,
        source: str = "video",
        url: str = "",
        t_start: float = 0.0,
    ) -> dict[str, Any] | None:
        """Store one technique learned from a video. Dedupes by exact name."""
        key = concept_key(name)
        if not key:
            return None
        if not is_cad_concept(name, summary):
            # Channel/website filler is not a technique — keep it out of storage
            # so it cannot crowd out real ones later.
            return None
        with self._lock:
            rows = _read_jsonl(memory_path(self.root))
            for row in rows:
                if row.get("kind") == "concept" and concept_key(str(row.get("name") or "")) == key:
                    # Newer sighting refreshes the summary/steps but keeps identity.
                    row.update(
                        {
                            "summary": str(summary)[:600] or row.get("summary", ""),
                            "when_to_use": str(when_to_use)[:400] or row.get("when_to_use", ""),
                            "steps": steps or row.get("steps") or [],
                            "t": time.time(),
                            "hits": int(row.get("hits") or 0) + 1,
                        }
                    )
                    _rewrite_jsonl(memory_path(self.root), rows)
                    return row
            row = {
                "kind": "concept",
                "name": key,
                "summary": str(summary)[:600],
                "when_to_use": str(when_to_use)[:400],
                "preconditions": [str(p)[:120] for p in (preconditions or [])][:8],
                "steps": [s for s in (steps or []) if isinstance(s, dict)][:64],
                "source": source[:32],
                "url": str(url)[:512],
                "t_start": float(t_start or 0.0),
                "t": time.time(),
                "hits": 1,
            }
            _append_jsonl(memory_path(self.root), row)
            _prune_file(memory_path(self.root), KEEP_MEMORY_ROWS)
            return row

    # -- queries --------------------------------------------------------------

    def list_skills(self, limit: int = 60) -> list[dict[str, Any]]:
        rows = [r for r in _read_jsonl(memory_path(self.root)) if r.get("kind") == "skill"]
        rows.sort(key=lambda r: float(r.get("t") or 0), reverse=True)
        return rows[: max(1, int(limit))]

    def list_concepts(self, limit: int = 400) -> list[dict[str, Any]]:
        rows = [
            r
            for r in _read_jsonl(memory_path(self.root))
            if r.get("kind") == "concept"
            and is_cad_concept(str(r.get("name") or ""), str(r.get("summary") or ""))
        ]
        rows.sort(key=lambda r: int(r.get("hits") or 0), reverse=True)
        return rows[: max(1, int(limit))]

    def matching_skills(self, goal: str, limit: int = 8) -> list[dict[str, Any]]:
        return [s for s in self.list_skills() if goals_match(goal, str(s.get("goal") or ""))][:limit]

    def matching_concepts(self, goal: str, limit: int = 8) -> list[dict[str, Any]]:
        rows = self.list_concepts(limit=400)
        out: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add(row: dict[str, Any]) -> None:
            key = concept_key(str(row.get("name") or ""))
            if key in seen:
                return
            seen.add(key)
            out.append(row)

        # 1. Techniques the goal itself implies (ordered by relevance).
        by_key = {concept_key(str(r.get("name") or "")): r for r in rows}
        for name in goal_technique_hints(goal):
            hit = by_key.get(concept_key(name))
            if hit is not None:
                add(hit)
            if len(out) >= limit:
                return out
        # 2. Techniques that share vocabulary with the goal.
        for row in rows:
            if len(out) >= limit:
                break
            name = str(row.get("name") or "")
            # Short technique names ("shell") must match richer goals ("shell the box"),
            # so the name is checked on its own before the full haystack dilutes overlap.
            if name and goals_match(goal, name):
                add(row)
                continue
            haystack = " ".join(
                [name, str(row.get("summary") or ""), str(row.get("when_to_use") or "")]
            )
            if goals_match(goal, haystack, min_overlap=0.30):
                add(row)
        return out

    def prompt_block(self, goal: str, *, max_chars: int = 5000) -> str:
        """Render matching skills + techniques for injection into prompts.

        Goals rarely share vocabulary with generic technique names ("make me a
        cube" vs "extrude"), so when few concepts match, the best recent ones
        are included anyway — otherwise video learnings are silently invisible.
        """
        parts: list[str] = [
            "LEARNED FROM VIDEOS - use these when they fit the goal. A recipe is a HINT: "
            "run it only when its precondition is visible on screen, and prefer clicking "
            "the named control over typing a bare noun. Never repeat an action that had "
            "no visible effect."
        ]
        for skill in self.matching_skills(goal, limit=4):
            steps = skill.get("actions") or []
            compact = [
                str(a.get("action")) + (":" + str(a.get("target") or a.get("text") or "") if a.get("target") or a.get("text") else "")
                for a in steps[:14]
                if isinstance(a, dict)
            ]
            parts.append(f"SKILL '{skill.get('goal')}': " + " -> ".join(compact))
        concepts = self.matching_concepts(goal, limit=8)
        if len(concepts) < 4:
            # Fallback: surface the most-useful techniques even when the goal
            # doesn't lexically match them. Ones carrying a real recipe win,
            # because a technique without steps cannot be acted on.
            known = {concept_key(str(c.get("name") or "")) for c in concepts}
            pool = [
                c
                for c in self.list_concepts(limit=60)
                if concept_key(str(c.get("name") or "")) not in known
            ]
            pool.sort(
                key=lambda c: (
                    0 if _steps_recipe(self.effective_steps(c)) else 1,
                    -int(c.get("hits") or 0),
                )
            )
            for c in pool:
                concepts.append(c)
                if len(concepts) >= 10:
                    break
        for concept in concepts:
            line = f"TECHNIQUE '{concept.get('name')}': {concept.get('summary')}"
            if concept.get("when_to_use"):
                line += f" (use when: {concept['when_to_use']})"
            recipe = _steps_recipe(self.effective_steps(concept))
            if recipe:
                line += f" — do: {recipe}"
            parts.append(line)
        block = "\n".join(parts)
        if len(block) > max_chars:
            block = block[:max_chars]
        return block

    def stats(self) -> dict[str, Any]:
        return {
            "episodes": len(_read_jsonl(episodes_path(self.root))),
            "skills": len(self.list_skills(limit=10_000)),
            "concepts": len(self.list_concepts(limit=10_000)),
            "root": str(self.root),
        }
