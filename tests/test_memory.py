from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.memory import (
    AgentMemory,
    action_signature,
    concept_key,
    goal_technique_hints,
    goals_match,
    is_cad_concept,
)


class GoalMatchTests(unittest.TestCase):
    def test_exact_match(self):
        self.assertTrue(goals_match("make a mug", "make a mug"))

    def test_substring_match(self):
        self.assertTrue(goals_match("extrude", "extrude the selected face"))

    def test_overlap_match(self):
        self.assertTrue(goals_match("fillet the top edges", "fillet top edges of the box"))

    def test_unrelated(self):
        self.assertFalse(goals_match("make a mug", "revolve a sphere"))

    def test_empty(self):
        self.assertFalse(goals_match("", "anything"))


class SignatureTests(unittest.TestCase):
    def test_click_signature_buckets(self):
        a = action_signature({"action": "click", "x": 100, "y": 200})
        b = action_signature({"action": "click", "x": 103, "y": 205})
        c = action_signature({"action": "click", "x": 400, "y": 200})
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_target_text_ignored(self):
        # Same point, alternating free-text target labels must stay the same
        # signature — otherwise anti-repeat never fires (lives loop bug).
        a = action_signature({"action": "click", "x": 142, "y": 105, "target": "Sketch button"})
        b = action_signature({"action": "click", "x": 142, "y": 105, "target": "Top plane"})
        self.assertEqual(a, b)

    def test_hotkey_signature(self):
        a = action_signature({"action": "hotkey", "keys": ["Ctrl", "z"]})
        b = action_signature({"action": "hotkey", "keys": ["ctrl", "Z"]})
        self.assertEqual(a, b)

    def test_type_signature(self):
        self.assertEqual(
            action_signature({"action": "type", "text": "extrude"}),
            "type:extrude",
        )

    def test_invalid(self):
        self.assertEqual(action_signature(None), "")
        self.assertEqual(action_signature("nope"), "")


class ConceptKeyTests(unittest.TestCase):
    def test_normalizes(self):
        self.assertEqual(concept_key("  Circular   Pattern "), "circular pattern")


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.memory = AgentMemory(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_dataset_dirs_created(self):
        for sub in ("screenshots", "videos", "frames"):
            self.assertTrue((self.root / sub).is_dir())

    def test_add_and_list_skills(self):
        self.memory.add_skill(goal="extrude a circle", actions=[{"action": "click", "x": 1, "y": 2}])
        skills = self.memory.list_skills()
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0]["goal"], "extrude a circle")

    def test_matching_skills(self):
        self.memory.add_skill(goal="extrude a circle", actions=[])
        self.memory.add_skill(goal="revolve a profile", actions=[])
        matches = self.memory.matching_skills("extrude the circle")
        self.assertEqual(len(matches), 1)
        self.assertIn("extrude", matches[0]["goal"])

    def test_add_concept_and_dedupe(self):
        row1 = self.memory.add_concept(name="Circular Pattern", summary="repeat a feature in a circle")
        row2 = self.memory.add_concept(name="circular  pattern", summary="better summary here")
        self.assertIsNotNone(row1)
        self.assertIsNotNone(row2)
        concepts = self.memory.list_concepts()
        self.assertEqual(len(concepts), 1)
        self.assertEqual(concepts[0]["summary"], "better summary here")
        self.assertEqual(concepts[0]["hits"], 2)

    def test_empty_concept_rejected(self):
        self.assertIsNone(self.memory.add_concept(name="   ", summary="x"))

    def test_matching_concepts(self):
        self.memory.add_concept(name="shell", summary="hollow a solid to a wall thickness")
        self.memory.add_concept(name="draft", summary="add a draft angle")
        matches = self.memory.matching_concepts("shell the box")
        self.assertTrue(any(m["name"] == "shell" for m in matches))

    def test_episode_roundtrip(self):
        self.memory.add_episode(
            {
                "goal": "make a plate",
                "actions": [{"action": "click", "x": 5, "y": 6}],
                "screenshots": ["/tmp/a.jpg"],
                "source": "record",
            }
        )
        self.assertEqual(len(self.memory.episodes()), 1)
        last = self.memory.last_episode()
        self.assertEqual(last["goal"], "make a plate")
        self.assertEqual(last["actions"][0]["action"], "click")

    def test_corrupt_lines_skipped(self):
        path = self.root / "memory.jsonl"
        path.write_text('{"kind":"skill","goal":"a","actions":[]}\nnot json\n{"broken\n', encoding="utf-8")
        skills = self.memory.list_skills()
        self.assertEqual(len(skills), 1)

    def test_prompt_block_contains_matching_material(self):
        self.memory.add_skill(goal="fillet the edges", actions=[{"action": "key", "keys": ["s"]}])
        self.memory.add_concept(name="fillet", summary="round an edge")
        block = self.memory.prompt_block("fillet the top edges")
        self.assertIn("SKILL", block)
        self.assertIn("TECHNIQUE", block)

    def test_prompt_block_falls_back_when_goal_has_no_shared_vocab(self):
        # "make me a simple cube" shares no words with "extrude" — the video
        # learnings must still surface, or the planner never sees them.
        self.memory.add_concept(name="extrude", summary="pull a sketch into 3D")
        block = self.memory.prompt_block("make me a simple cube")
        self.assertIn("TECHNIQUE 'extrude'", block)

    def test_prompt_block_renders_step_recipe(self):
        # "pocket" has no vetted recipe, so the stored one is rendered as-is.
        self.memory.add_concept(
            name="pocket",
            summary="cut a recess into a face",
            steps=[
                {"action": "key", "keys": ["s"]},
                {"action": "type", "text": "pocket"},
                {"action": "key", "keys": ["Enter"]},
            ],
        )
        block = self.memory.prompt_block("make me a simple cube")
        self.assertIn("— do: s -> \"pocket\" -> Enter", block)

    def test_placeholder_search_recipe_is_replaced_by_vetted_one(self):
        # The old teacher stored "press S and type the name" for everything;
        # the planner copied it verbatim and typed nouns at an unchanged screen.
        self.memory.add_concept(
            name="rectangle",
            summary="draw a rectangle in a sketch",
            steps=[
                {"action": "key", "keys": ["s"]},
                {"action": "type", "text": "rectangle"},
                {"action": "key", "keys": ["enter"]},
            ],
        )
        block = self.memory.prompt_block("make me a simple cube")
        self.assertNotIn("s -> \"rectangle\"", block)
        self.assertIn("— do: r -> click(first corner) -> click(opposite corner)", block)

    def test_prompt_block_respects_cap(self):
        for i in range(50):
            self.memory.add_concept(name=f"technique {i}", summary="y" * 400)
        block = self.memory.prompt_block("technique", max_chars=500)
        self.assertLessEqual(len(block), 500)

    def test_prompt_block_marks_recipes_as_hints(self):
        self.memory.add_concept(name="extrude", summary="pull a sketch into 3D")
        block = self.memory.prompt_block("make me a simple cube")
        self.assertIn("HINT", block)

    def test_prompt_block_renders_click_targets(self):
        self.memory.add_concept(
            name="fillet",
            summary="round an edge",
            steps=[{"action": "click", "target": "Fillet", "reason": "open it"}],
        )
        self.assertIn("click(Fillet)", self.memory.prompt_block("fillet edges"))

    def test_channel_filler_never_reaches_prompts(self):
        self.memory.add_concept(name="sign in", summary="Log in to your account")
        self.memory.add_concept(name="extrude", summary="pull a sketch into 3D")
        self.assertEqual(self.memory.stats()["concepts"], 1)
        self.assertNotIn("sign in", self.memory.prompt_block("make me a simple cube"))

    def test_stats(self):
        stats = self.memory.stats()
        self.assertEqual(stats["episodes"], 0)
        self.assertEqual(stats["skills"], 0)
        self.assertEqual(stats["concepts"], 0)


class CadFilterTests(unittest.TestCase):
    def test_video_id_is_not_a_technique(self):
        # The learn pipeline used to fall back to a card named after the file
        # ("Technique inferred from 1791174595") and store it as memory.
        self.assertFalse(is_cad_concept("1791174595", "Technique inferred from 1791174595"))
        self.assertFalse(is_cad_concept("20261006", "clip from 20261006"))

    def test_real_techniques_survive(self):
        for name in ("extrude", "fillet", "shell", "circular pattern", "fastened mate", "Loft"):
            self.assertTrue(is_cad_concept(name, "a technique"), name)

    def test_channel_filler_rejected(self):
        for name in ("create an account", "sign in", "manage projects", "loading workspaces"):
            self.assertFalse(is_cad_concept(name, "website navigation"), name)


class GoalHintTests(unittest.TestCase):
    def test_cube_goal_hints_modeling_techniques(self):
        hints = goal_technique_hints("make me a simple cube")
        self.assertIn("extrude", hints)

    def test_holes_in_a_circle_hints_patterning(self):
        # "4 holes in a circle" shares no vocabulary with "pattern", so the
        # learned patterning recipe used to never surface for that phrasing.
        hints = goal_technique_hints("make a flange with 4 holes in a circle")
        self.assertIn("pattern", hints)

    def test_unrelated_goal_hints_nothing(self):
        self.assertEqual(goal_technique_hints("say hello"), [])


if __name__ == "__main__":
    unittest.main()
