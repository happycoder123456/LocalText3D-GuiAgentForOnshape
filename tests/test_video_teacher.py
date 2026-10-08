from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.paths import ensure_dataset
from agent.video_teacher import (
    captions_window,
    _clean_video_url,
    _caption_sidecars,
    _concept_budget,
    _is_youtube_url,
    _parse_json_object,
    _read_caption_file,
    _segment_length_sec,
    canonical_steps,
    concepts_from_captions,
    default_steps,
    ensure_video_path_in_dataset,
    is_onshape_command,
    merge_concepts,
    sanitize_concept_steps,
)


class UrlTests(unittest.TestCase):
    def test_youtube_hosts(self):
        for url in (
            "https://www.youtube.com/watch?v=abc123def45",
            "https://youtu.be/abc123def45",
            "https://m.youtube.com/watch?v=abc123def45",
            "https://music.youtube.com/watch?v=abc123def45",
        ):
            self.assertTrue(_is_youtube_url(url), url)

    def test_non_youtube_refused(self):
        for url in (
            "https://evil.com/watch?v=abc123def45",
            "https://youtube.com.evil.com/watch?v=abc123def45",
            "ftp://youtube.com/x",
            "not a url",
            "",
        ):
            self.assertFalse(_is_youtube_url(url), url)

    def test_userinfo_refused(self):
        self.assertFalse(_is_youtube_url("https://user@youtube.com/watch?v=abc123def45"))

    def test_clean_short_link(self):
        self.assertEqual(
            _clean_video_url("https://youtu.be/abc123def45?t=315"),
            "https://www.youtube.com/watch?v=abc123def45",
        )

    def test_clean_watch_link_drops_playlist(self):
        self.assertEqual(
            _clean_video_url("https://www.youtube.com/watch?v=abc123def45&list=PLxxx&index=2"),
            "https://www.youtube.com/watch?v=abc123def45",
        )

    def test_clean_passthrough(self):
        self.assertEqual(
            _clean_video_url("https://www.youtube.com/watch?v=abc123def45"),
            "https://www.youtube.com/watch?v=abc123def45",
        )


class LocalPathTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = ensure_dataset(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def _make_video(self, name: str = "demo.mp4") -> Path:
        path = self.root / "videos" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00" * 64)
        return path

    def test_inside_dataset_accepted(self):
        path = self._make_video()
        resolved = ensure_video_path_in_dataset(str(path), self.root)
        self.assertEqual(resolved, path.resolve())

    def test_outside_dataset_refused(self):
        with self.assertRaises(ValueError):
            ensure_video_path_in_dataset(str(Path(self._tmp.name).parent / "other.mp4"), self.root)

    def test_traversal_refused(self):
        sneaky = str(self.root / "videos" / ".." / ".." / ".." / "windows" / "system32" / "x.mp4")
        with self.assertRaises(ValueError):
            ensure_video_path_in_dataset(sneaky, self.root)

    def test_wrong_extension_refused(self):
        path = self.root / "videos" / "evil.exe"
        path.write_bytes(b"MZ")
        with self.assertRaises(ValueError):
            ensure_video_path_in_dataset(str(path), self.root)

    def test_missing_returns_none(self):
        self.assertIsNone(ensure_video_path_in_dataset(str(self.root / "videos" / "gone.mp4"), self.root))

    def test_empty_returns_none(self):
        self.assertIsNone(ensure_video_path_in_dataset("", self.root))


class CaptionTests(unittest.TestCase):
    def test_read_vtt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub.vtt"
            path.write_text(
                "WEBVTT\n\n"
                "00:00:01.000 --> 00:00:04.000\n"
                "Now we extrude the sketch 20 millimeters\n\n"
                "00:00:05.000 --> 00:00:08.000\n"
                "Don't forget to subscribe\n",
                encoding="utf-8",
            )
            cues = _read_caption_file(path)
        self.assertEqual(len(cues), 1)  # filler line dropped
        self.assertIn("extrude", cues[0][1])

    def test_window_filters_by_time(self):
        cues = [(1.0, "first"), (50.0, "second"), (100.0, "third")]
        self.assertEqual(captions_window(cues, 0.0, 10.0), "first")
        self.assertIn("second", captions_window(cues, 40.0, 60.0))

    def test_window_falls_back_to_head_when_empty_range(self):
        cues = [(1.0, "alpha"), (2.0, "beta")]
        self.assertIn("alpha", captions_window(cues, 999.0, 1000.0))

    def test_missing_file(self):
        self.assertEqual(_read_caption_file(Path("nope.vtt")), [])


class CaptionConceptTests(unittest.TestCase):
    def test_seeds_from_narration(self):
        cues = [(0.0, "we will revolve this profile around the axis")]
        found = concepts_from_captions(cues, t0=0.0, t1=10.0)
        names = [c["name"] for c in found]
        self.assertIn("revolve", names)

    def test_no_cues_no_concepts(self):
        self.assertEqual(concepts_from_captions([], t0=0.0, t1=10.0), [])


class MergeTests(unittest.TestCase):
    def test_dedupes_case_insensitive(self):
        out = merge_concepts(
            [
                {"name": "Fillet", "summary": "short"},
                {"name": "fillet", "summary": "a much longer and better summary"},
                {"name": "Shell", "summary": "hollow"},
            ]
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["summary"], "a much longer and better summary")

    def test_empty_names_dropped(self):
        self.assertEqual(len(merge_concepts([{"name": "  "}, {"summary": "no name"}])), 0)

    def test_overlong_name_dropped(self):
        self.assertEqual(len(merge_concepts([{"name": "x" * 200}])), 0)


class SanitizeStepsTests(unittest.TestCase):
    def test_drops_video_coordinates(self):
        steps = sanitize_concept_steps(
            [
                {"action": "click", "x": 100, "y": 200},
                {"action": "click", "target": "Extrude button", "reason": "press extrude"},
                {"action": "stop"},
                {"action": "key", "keys": ["s"]},
            ]
        )
        kinds = [s["action"] for s in steps]
        self.assertNotIn("stop", kinds)
        self.assertIn("key", kinds)
        # coordinate-only click dropped; targetless click kept only with a target/reason
        self.assertEqual(kinds.count("click"), 1)
        self.assertEqual(steps[0].get("target"), "Extrude button")

    def test_type_gets_wait(self):
        steps = sanitize_concept_steps([{"action": "type", "text": "extrude"}])
        self.assertEqual([s["action"] for s in steps], ["type", "wait"])

    def test_cap_at_twelve(self):
        steps = sanitize_concept_steps([{"action": "key", "keys": ["a"]} for i in range(30)])
        self.assertEqual(len(steps), 12)


class SegmentationTests(unittest.TestCase):
    def test_short_lessons_are_split_into_several_segments(self):
        # A 6 minute lesson used to be ONE segment = ONE model call.
        self.assertEqual(_segment_length_sec(6.0), 120.0)
        self.assertLess(_segment_length_sec(10.0), 10.0 * 60.0)

    def test_long_lessons_stay_coarse(self):
        self.assertLessEqual(_segment_length_sec(120.0), 12.0 * 60.0)

    def test_budget_allows_a_full_lesson(self):
        self.assertGreaterEqual(_concept_budget(6.0), 24)
        self.assertLessEqual(_concept_budget(600.0), 400)


class JsonSalvageTests(unittest.TestCase):
    def test_fenced_json(self):
        out = _parse_json_object('```json\n{"concepts":[{"name":"extrude"}]}\n```')
        self.assertEqual(out["concepts"][0]["name"], "extrude")

    def test_truncated_array_is_salvaged(self):
        raw = '{"concepts":[{"name":"extrude","summary":"push"},{"name":"fillet","sum'
        out = _parse_json_object(raw)
        # The half-written second card is unrecoverable; the first must survive.
        self.assertEqual([c.get("name") for c in out.get("concepts", [])], ["extrude"])

    def test_complete_cards_before_the_cut_are_all_kept(self):
        raw = '{"concepts":[{"name":"extrude"},{"name":"fillet"},{"name":"shell"'
        out = _parse_json_object(raw)
        self.assertEqual(
            [c.get("name") for c in out.get("concepts", [])], ["extrude", "fillet"]
        )

    def test_bare_list_accepted(self):
        out = _parse_json_object('[{"name":"loft"}]')
        self.assertEqual([c.get("name") for c in out.get("concepts", [])], ["loft"])

    def test_prose_returns_nothing(self):
        self.assertEqual(_parse_json_object("Sorry, I cannot do that."), {})


class RecipeTests(unittest.TestCase):
    def test_vetted_recipe_uses_real_controls(self):
        steps, source = default_steps("fillet")
        self.assertEqual(source, "curated")
        targets = [s.get("target") for s in steps if s.get("action") == "click"]
        self.assertIn("Fillet", targets)
        self.assertTrue(all(s.get("target") or s.get("keys") for s in steps))

    def test_lesson_specific_name_inherits_the_recipe(self):
        self.assertTrue(canonical_steps("extrude blind"))

    def test_non_command_gets_no_invented_recipe(self):
        steps, source = default_steps("sign in")
        self.assertEqual((steps, source), ([], "none"))

    def test_command_detection(self):
        self.assertTrue(is_onshape_command("extrude"))
        self.assertFalse(is_onshape_command("loading workspaces"))

    def test_recipe_never_types_a_placeholder_word(self):
        for name in ("extrude", "fillet", "shell", "draft", "dimension", "create document"):
            for step in canonical_steps(name):
                if step.get("action") == "type":
                    self.fail(f"{name} recipe types literal text: {step}")


class JunkConceptTests(unittest.TestCase):
    def test_channel_filler_dropped(self):
        out = merge_concepts(
            [
                {"name": "sign in", "summary": "Log in to your account"},
                {"name": "create an account", "summary": "Sign up for a free account"},
                {"name": "loading workspaces", "summary": "Wait for the workspace to load"},
                {"name": "extrude", "summary": "Push a sketch region into 3D"},
            ]
        )
        self.assertEqual([c["name"] for c in out], ["extrude"])

    def test_real_techniques_survive(self):
        for name in ("fillet", "shell", "rectangle", "circular pattern", "fastened mate"):
            self.assertTrue(
                merge_concepts([{"name": name, "summary": "a CAD technique"}]), name
            )


class CaptionSidecarTests(unittest.TestCase):
    """yt-dlp writes ``<stem>.en.vtt``, not ``<stem>.vtt``.

    Looking only for the latter meant captions never loaded for a downloaded
    lesson, so the narration half of concept extraction returned nothing.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = ensure_dataset(Path(self._tmp.name))
        self.videos = self.root / "videos" / "teacher"
        self.videos.mkdir(parents=True, exist_ok=True)
        self.video = self.videos / "1791174595.mp4"
        self.video.write_bytes(b"\x00\x00")

    def test_language_suffixed_vtt_is_found(self):
        (self.videos / "1791174595.en.vtt").write_text("WEBVTT\n", encoding="utf-8")
        found = [p.name for p in _caption_sidecars(self.video, self.root)]
        self.assertIn("1791174595.en.vtt", found)

    def test_plain_sidecar_preferred_over_language_one(self):
        (self.videos / "1791174595.vtt").write_text("WEBVTT\n", encoding="utf-8")
        (self.videos / "1791174595.en.vtt").write_text("WEBVTT\n", encoding="utf-8")
        found = [p.name for p in _caption_sidecars(self.video, self.root)]
        self.assertEqual(found[0], "1791174595.vtt")

    def test_auto_generated_not_preferred_over_translation(self):
        (self.videos / "1791174595.en.vtt").write_text("WEBVTT\n", encoding="utf-8")
        (self.videos / "1791174595.en-orig.vtt").write_text("WEBVTT\n", encoding="utf-8")
        found = [p.name for p in _caption_sidecars(self.video, self.root)]
        self.assertEqual(found[0], "1791174595.en.vtt")

    def test_unrelated_files_ignored(self):
        (self.videos / "1791174595.thumb.jpg").write_bytes(b"\x00")
        self.assertEqual(_caption_sidecars(self.video, self.root), [])

    def test_local_video_loads_those_captions(self):
        from agent.video_teacher import fetch_video_source

        (self.videos / "1791174595.en.vtt").write_text(
            "WEBVTT\n\n00:00:08.000 --> 00:00:12.000\nWe extrude the sketch.\n",
            encoding="utf-8",
        )
        _path, cues, _title = fetch_video_source(path=str(self.video), dataset_root=self.root)
        self.assertTrue(cues, "language-suffixed captions must load")
        self.assertIn("extrude", " ".join(t for _t, t in cues))


class RollingCaptionTests(unittest.TestCase):
    """Auto-captions repeat each utterance 2–3x via a rolling window."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "clip.vtt"
        self.path.write_text(
            "WEBVTT\n\n"
            "00:00:08.000 --> 00:00:11.790\nHello everybody to the fresh new course\n\n"
            "00:00:11.790 --> 00:00:14.000\nHello everybody to the fresh new course of Onshape CAD.\n\n"
            "00:00:14.000 --> 00:00:15.000\nof Onshape CAD.\n\n"
            "00:00:15.000 --> 00:00:18.000\nof Onshape CAD. In this course we cover sketch and extrude.\n\n"
            "00:00:18.000 --> 00:00:19.000\nIn this course we cover sketch and extrude.\n\n",
            encoding="utf-8",
        )

    def test_repeated_tail_is_stored_once(self):
        cues = _read_caption_file(self.path)
        joined = " ".join(text for _t, text in cues)
        self.assertEqual(joined.count("fresh new course"), 1)
        self.assertEqual(joined.count("In this course we cover"), 1)
        self.assertEqual(joined.count("of Onshape CAD."), 1)

    def test_technique_words_survive(self):
        joined = " ".join(text for _t, text in _read_caption_file(self.path))
        self.assertIn("sketch", joined)
        self.assertIn("extrude", joined)


class CaptionsWindowTests(unittest.TestCase):
    def test_samples_across_the_whole_segment(self):
        cues = [(float(i), f"cue{i}") for i in range(100)]
        text = captions_window(cues, 0.0, 200.0, limit=10)
        sampled = text.split()
        self.assertEqual(len(sampled), 10)
        # A head-only window would return cue0..cue9 and miss cue90 entirely.
        self.assertIn("cue90", sampled)
        self.assertIn("cue0", sampled)

    def test_short_window_untouched(self):
        cues = [(float(i), f"cue{i}") for i in range(5)]
        self.assertEqual(captions_window(cues, 0.0, 10.0, limit=10), "cue0 cue1 cue2 cue3 cue4")

    def test_empty_range_falls_back_to_the_lesson_opening(self):
        cues = [(float(i), f"cue{i}") for i in range(5)]
        self.assertTrue(captions_window(cues, 900.0, 950.0, limit=3))


if __name__ == "__main__":
    unittest.main()
