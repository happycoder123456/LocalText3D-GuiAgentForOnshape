"""Action schema + parsing and Onshape-specific prompts for the local vision policy."""

from __future__ import annotations

import json
import re
from typing import Any

MAX_WAIT_SECONDS = 8.0

ACTIONS = frozenset({"click", "dblclick", "drag", "scroll", "key", "hotkey", "type", "wait", "stop"})

SYSTEM_PROMPT = """You control Onshape (browser CAD) with mouse and keyboard like a careful human. Output ONE small JSON object only.
Never invent long key lists. keys must have at most 3 items.

Schema:
{"action":"click|dblclick|drag|scroll|key|hotkey|type|wait|stop","target":"","x":0,"y":0,"x2":0,"y2":0,"dx":0,"dy":0,"keys":[],"text":"","reason":""}

Coordinates x,y are pixels in the browser viewport (origin top-left). Click the actual UI control or model target you see.

Onshape facts:
- The toolbar is on top (Sketch, Extrude, Revolve, Fillet...), the feature tree is on the left, the view cube is top-right.
- NO toolbar visible (you are on the documents list): double-click the most recent document card under "Last opened by me" to open a Part Studio FIRST, then continue.
- Documents list, other route: the blue "Create" button is top-LEFT at about (109,66) — there is NO Create button top-right. Clicking it opens a dark dropdown whose FIRST item is "New document" (text row roughly x 75-300, y 110-140): click that item's TEXT to open a Part Studio.
- While a dropdown/menu is open, click the TEXT ROW of the item you want (rows are ~30 px tall); clicking the menu header or empty space does nothing, and Esc closes the menu. If a menu visibly opened as the plan expected, include "step done" in your reason.
- The documents list has NO Sketch/Extrude toolbar at all — never click where a toolbar "would be"; open a document first.
- Toolbar visible = you are ALREADY inside a document: NEVER run "create document" or open another document — start modeling right away.
- Document cards open ONLY on a double-click: emit action "dblclick" for that — a single click merely selects the card.
- Never click the Onshape logo in the page's top-left corner — it exits the document. Esc cancels tools.
- Never click the small icons next to the document title (version, history, push) in the top bar.
- If a modal/dialog covers the page ('Create version', 'Create variable', 'Rename'...), dismiss it FIRST: keys ["esc"] or its Cancel/X button. Canvas clicks do nothing while it is open.
- Sketch entries in the feature tree are RESULTS, not buttons: to sketch, click Sketch in the TOP toolbar, then click a plane (Top/Front/Right).
- Do NOT click a plane unless the "Select a sketch plane" prompt or the sketch dialog is visible — plane clicks do nothing without it. To start: Sketch button ≈ (144,55).
- When selecting a plane, click the CENTER of the plane's shaded rectangle in the viewport; its edge or empty canvas does nothing.
- If a search/filter box contains text (the tree then hides entries), click its × to clear it before clicking anything in the tree.
- "type" a tool name (for example "rectangle") and it runs via the S shortcut search automatically; pressing S yourself, typing, Enter also works.
- Sketch mode swaps the top toolbar row in place: Line ≈ (133,57), Rectangle ≈ (170,57) in the 1440x900 viewport. After picking the tool, click the origin point to place the first corner, then click the opposite corner.
- key/hotkey actions MUST list the keys in "keys" (for example "keys":["esc"]); keys placed in "text" do nothing.
- The current PLAN STEP is shown to you. When its expected outcome is visible on screen, include "step done" in your reason; otherwise say what you actually observe.
- Press S to open the shortcut search, type the command name, Enter. That is the fastest reliable way to run a command.
- Select-then-act: click geometry or a sketch region FIRST, then click the tool.
- Sketch workflow: Sketch -> pick a plane/face -> draw (L line, C circle, R rectangle) -> D to dimension -> click the green check to finish.
- Extrude needs a closed sketch region: select the region, press S, type "extrude", Enter, type a distance, Enter.
- Double-click a feature or dimension in the tree to edit its value; type the new value and press Enter.
- Right-click opens a context menu; click an item to pick it. Esc closes menus and cancels tools.
- Undo: Ctrl+Z. Never type into a field that is not focused; click it first.
- One purposeful action per step. Never spam the same click. Never more than 3 keys.
- If the same "type" command produced NO visible change twice, that tool is unavailable right now (rectangle before a sketch exists, extrude with nothing selected). Do NOT type it again — fix the precondition first: click Sketch, pick a plane, or select geometry.
- Learn from MEMORY: copy SUCCESS / teacher patterns; never repeat AVOID or BANNED actions.
- A MEMORY "TECHNIQUE ... — do:" line is a HINT, not a script: run it only when its precondition is already visible (a sketch open, geometry selected). If a technique has no recipe, reach for the visible toolbar button instead.
- NEVER type a bare noun ("rectangle", "fillet") with nothing selected and no sketch open — that produces no change. Set the precondition first.
- The agent ALSO supports clicking by label: emit {"action":"click","target":"Sketch"} (also "Extrude", "Top", "Front", "Right", "Corner rectangle", "Dimension", "Origin") INSTEAD of x,y when the target is a named toolbar button, feature-tree item, dialog field, or plane. "target" clicks are resolved against the REAL page (aria/tooltip labels) and are far more reliable than pixel guesses — always prefer target for named UI, pixels only for canvas/geometric clicks.
- If last reward was negative, change click target AND approach — do not nudge the same miss.
- stop only when the goal is visibly done in the viewport (feature exists in the tree, shape looks right).
"""

PLAN_SYSTEM_PROMPT = """You are an Onshape CAD planner. Given a modeling goal, output a JSON plan only:
{"plan":[{"step":"short instruction","expect":"visible sign it worked"}],"notes":""}
Rules:
- 3 to 14 steps. Each step is ONE observable action a GUI agent can perform (click Sketch, extrude 10mm, fillet edge...).
- If the current page is the documents list (no toolbar), the FIRST step must open a document: either "double-click the most recent document card under Last opened by me" OR "click the blue Create button (top-LEFT, about 109,66 — never 'top right'), then click New document (first item of the dropdown)". Opening the document via EITHER route is fine; do not plan both.
- Order steps the way a CAD modeler would: sketch on a plane, constrain, finish sketch, feature, then refine.
- Prefer the S shortcut search over hunting through menus.
- Numeric values go in the step text (for example "extrude 20 mm").
- When a "Known techniques" block is supplied, reuse its wording for steps it clearly covers — but write each step as a GUI action with its precondition satisfied. Never paste a technique's raw keystroke recipe into the plan, and never plan a tool before the sketch/selection it needs.
- PREFER named steps the vision model can satisfy with a "target" click (Sketch, Extrude, Top plane, Origin, Corner rectangle, Dimension, green check) over pixel-only instructions.
- If a listed technique covers something you are about to repeat (patterning, mirroring, filleting several edges), USE that technique instead of re-drawing or re-feature-ing each copy by hand — that is why it was taught to you, and it keeps the plan inside 14 steps.
- No markdown, no commentary outside the JSON.
"""

_JSON_RE = re.compile(r"\{[\s\S]*\}")


class PolicyError(Exception):
    pass


def _close_truncated_json(text: str) -> str:
    """Close unclosed strings/brackets left behind by a truncated response."""
    stack: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
    if in_string:
        text += '"'
    text = text.rstrip().rstrip(",")
    closing = {"{": "}", "[": "]"}
    while stack:
        text += closing[stack.pop()]
    return text


def _repair_json(text: str) -> str:
    """Best-effort repair for truncated model JSON (validated before returning)."""
    text = text.strip()
    if not text:
        return text
    start = text.find("{")
    if start < 0:
        return text
    text = text[start:]
    text = re.sub(
        r'"keys"\s*:\s*\[(?:[^\[\]]|\[[^\]]*\]){0,800}',
        lambda m: m.group(0)[:80].rstrip(",") + '"]',
        text,
        count=1,
    )
    # Drop trailing partial tokens until the closed-up text parses.
    chunk = text
    for _ in range(10):
        cand = _close_truncated_json(chunk)
        try:
            json.loads(cand)
            return cand
        except json.JSONDecodeError:
            pass
        trimmed = re.sub(r',?\s*"[^":{}]*"\s*$', "", chunk)  # dangling key, no value
        if trimmed != chunk:
            chunk = trimmed
            continue
        cut = max(chunk.rfind(","), chunk.rfind(":"))
        if cut <= 0:
            break
        chunk = chunk[:cut]
    return _close_truncated_json(chunk)


def clamp_wait_seconds(value: Any, default: float = 0.5) -> float:
    try:
        sec = float(value)
    except (TypeError, ValueError):
        return default
    if sec != sec or sec in {float("inf"), float("-inf")}:
        return default
    return float(max(0.0, min(MAX_WAIT_SECONDS, sec)))


def _coord(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def parse_action(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise PolicyError("Empty model response")
    data: Any = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_RE.search(text)
        candidates = [match.group(0)] if match else []
        candidates.append(_repair_json(text))
        for cand in candidates:
            try:
                data = json.loads(cand)
                break
            except json.JSONDecodeError:
                continue
    if not isinstance(data, dict):
        action_m = re.search(r'"action"\s*:\s*"(\w+)"', text)
        if not action_m:
            raise PolicyError(f"No JSON in model response: {text[:160]}")
        data = {"action": action_m.group(1), "keys": [], "reason": "repaired"}

    action = str(data.get("action") or "wait").lower().strip()
    if action not in ACTIONS:
        action = "wait"
    keys = data.get("keys") or []
    if isinstance(keys, str):
        keys = [keys]
    cleaned: list[str] = []
    for k in keys:
        label = str(k).strip()
        if not label:
            continue
        label = label.replace("Key.", "").replace("key.", "")
        cleaned.append(label)
        if len(cleaned) >= 3:
            break
    if not cleaned and action in {"key", "hotkey"}:
        # Models sometimes put the key combo in "text" instead of "keys" —
        # honoring it turns a silent no-op into the intended keypress.
        fallback = str(data.get("text") or "")
        cleaned = [p[:24] for p in re.split(r"[,\s+]+", fallback) if p][:3]
    # Collapse spam like C,C,C -> single C
    if len(cleaned) >= 2 and len(set(x.lower() for x in cleaned)) == 1:
        cleaned = [cleaned[0]]

    return {
        "action": action,
        "target": str(data.get("target") or "")[:64],
        "x": _coord(data.get("x")),
        "y": _coord(data.get("y")),
        "x2": _coord(data.get("x2")),
        "y2": _coord(data.get("y2")),
        "dx": _coord(data.get("dx")),
        "dy": _coord(data.get("dy")),
        "keys": cleaned,
        "text": str(data.get("text") or "")[:64],
        "seconds": clamp_wait_seconds(data.get("seconds"), 0.5 if action == "wait" else 0.0),
        "reason": str(data.get("reason") or "")[:120],
    }


def parse_plan(raw: str) -> list[dict[str, str]]:
    """Parse the planner's JSON into [{step, expect}] items."""
    text = (raw or "").strip()
    if not text:
        raise PolicyError("Empty planner response")
    data: Any = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_RE.search(text)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None
        if data is None:
            data = _repair_json(text)
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                raise PolicyError(f"No JSON in planner response: {text[:160]}")
    if not isinstance(data, dict):
        raise PolicyError("Planner response must be a JSON object")
    steps = data.get("plan") or data.get("steps") or []
    out: list[dict[str, str]] = []
    if isinstance(steps, list):
        for item in steps[:20]:
            if isinstance(item, str):
                step = item.strip()
                expect = ""
            elif isinstance(item, dict):
                step = str(item.get("step") or item.get("instruction") or "").strip()
                expect = str(item.get("expect") or item.get("done_when") or "").strip()
            else:
                continue
            if step:
                out.append({"step": step[:400], "expect": expect[:300]})
    if not out:
        raise PolicyError("Planner returned no steps")
    return out
