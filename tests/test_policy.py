from __future__ import annotations

import json
import unittest

from agent.policy import (
    ACTIONS,
    PolicyError,
    clamp_wait_seconds,
    parse_action,
    parse_plan,
)


class ParseActionTests(unittest.TestCase):
    def test_basic_click(self):
        raw = json.dumps(
            {"action": "click", "x": 100, "y": 200, "target": "Sketch button", "reason": "start sketch"}
        )
        a = parse_action(raw)
        self.assertEqual(a["action"], "click")
        self.assertEqual(a["x"], 100)
        self.assertEqual(a["y"], 200)
        self.assertEqual(a["target"], "Sketch button")
        self.assertEqual(a["reason"], "start sketch")

    def test_all_actions_accepted(self):
        for kind in ACTIONS:
            a = parse_action(json.dumps({"action": kind}))
            self.assertEqual(a["action"], kind)

    def test_unknown_action_becomes_wait(self):
        a = parse_action(json.dumps({"action": "explode"}))
        self.assertEqual(a["action"], "wait")

    def test_empty_raises(self):
        with self.assertRaises(PolicyError):
            parse_action("")

    def test_json_embedded_in_prose(self):
        raw = 'Sure! Here you go: {"action":"click","x":5,"y":6} hope that helps'
        a = parse_action(raw)
        self.assertEqual(a["action"], "click")
        self.assertEqual(a["x"], 5)

    def test_truncated_json_repaired(self):
        raw = '{"action":"click","x":10,"y":20,"keys":["a"'
        a = parse_action(raw)
        self.assertEqual(a["action"], "click")

    def test_key_list_capped_at_three(self):
        raw = json.dumps({"action": "hotkey", "keys": ["ctrl", "shift", "alt", "z", "q"]})
        a = parse_action(raw)
        self.assertEqual(a["keys"], ["ctrl", "shift", "alt"])

    def test_duplicate_keys_collapsed(self):
        raw = json.dumps({"action": "hotkey", "keys": ["c", "c", "c"]})
        a = parse_action(raw)
        self.assertEqual(a["keys"], ["c"])

    def test_key_action_falls_back_to_text(self):
        # Models sometimes put the combo in "text" — must not become a no-op.
        out = parse_action(json.dumps({"action": "key", "text": "esc", "reason": ""}))
        self.assertEqual(out["keys"], ["esc"])
        out2 = parse_action(json.dumps({"action": "hotkey", "text": "ctrl+s", "reason": ""}))
        self.assertEqual(out2["keys"], ["ctrl", "s"])

    def test_string_keys_normalized_to_list(self):
        a = parse_action(json.dumps({"action": "key", "keys": "escape"}))
        self.assertEqual(a["keys"], ["escape"])

    def test_wait_seconds_clamped(self):
        a = parse_action(json.dumps({"action": "wait", "seconds": 999}))
        self.assertEqual(a["seconds"], 8.0)
        a = parse_action(json.dumps({"action": "wait", "seconds": -5}))
        self.assertEqual(a["seconds"], 0.0)

    def test_text_truncated(self):
        a = parse_action(json.dumps({"action": "type", "text": "x" * 500}))
        self.assertEqual(len(a["text"]), 64)

    def test_float_coords_coerced(self):
        a = parse_action(json.dumps({"action": "click", "x": 12.7, "y": "34"}))
        self.assertEqual(a["x"], 12)
        self.assertEqual(a["y"], 34)

    def test_bad_coords_become_zero(self):
        a = parse_action(json.dumps({"action": "click", "x": "abc", "y": None}))
        self.assertEqual(a["x"], 0)
        self.assertEqual(a["y"], 0)

    def test_key_prefixes_stripped(self):
        a = parse_action(json.dumps({"action": "key", "keys": ["Key.esc"]}))
        self.assertEqual(a["keys"], ["esc"])


class ClampWaitTests(unittest.TestCase):
    def test_clamp(self):
        self.assertEqual(clamp_wait_seconds(0.1), 0.1)
        self.assertEqual(clamp_wait_seconds(100), 8.0)
        self.assertEqual(clamp_wait_seconds("bad", default=0.5), 0.5)
        self.assertEqual(clamp_wait_seconds(float("nan")), 0.5)


class ParsePlanTests(unittest.TestCase):
    def test_basic_plan(self):
        raw = json.dumps(
            {
                "plan": [
                    {"step": "Start a sketch on the top plane", "expect": "sketch toolbar visible"},
                    {"step": "Draw a 40mm circle", "expect": "circle drawn"},
                ]
            }
        )
        plan = parse_plan(raw)
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0]["step"], "Start a sketch on the top plane")
        self.assertEqual(plan[0]["expect"], "sketch toolbar visible")

    def test_string_steps_allowed(self):
        raw = json.dumps({"plan": ["click Sketch", "pick plane"]})
        plan = parse_plan(raw)
        self.assertEqual([p["step"] for p in plan], ["click Sketch", "pick plane"])

    def test_embedded_json(self):
        raw = 'Here is the plan:\n```json\n{"plan":[{"step":"extrude 10mm"}]}\n```'
        plan = parse_plan(raw)
        self.assertEqual(plan[0]["step"], "extrude 10mm")

    def test_empty_raises(self):
        with self.assertRaises(PolicyError):
            parse_plan("")

    def test_no_steps_raises(self):
        with self.assertRaises(PolicyError):
            parse_plan(json.dumps({"plan": []}))

    def test_garbage_raises(self):
        with self.assertRaises(PolicyError):
            parse_plan("I cannot do that.")

    def test_plan_capped_at_twenty(self):
        raw = json.dumps({"plan": [{"step": f"step {i}"} for i in range(50)]})
        plan = parse_plan(raw)
        self.assertEqual(len(plan), 20)

    def test_step_text_truncated(self):
        raw = json.dumps({"plan": [{"step": "x" * 900}]})
        plan = parse_plan(raw)
        self.assertEqual(len(plan[0]["step"]), 400)

    def test_truncated_plan_repaired(self):
        # Real planner output when the response was cut mid-string.
        raw = '{"plan":[{"step":"Click the Front plane in the Feat'
        plan = parse_plan(raw)
        self.assertEqual(len(plan), 1)
        self.assertTrue(plan[0]["step"].startswith("Click the Front plane"))

    def test_plan_truncated_after_comma(self):
        # Cut right after a comma + opening quote (partial next key).
        raw = (
            '{"plan":[{"step":"Sketch on the Top plane","expect":"sketch open"},'
            '{"step":"Extrude 20 mm","'
        )
        plan = parse_plan(raw)
        self.assertEqual([p["step"] for p in plan], ["Sketch on the Top plane", "Extrude 20 mm"])
        self.assertEqual(plan[1]["expect"], "")


if __name__ == "__main__":
    unittest.main()
