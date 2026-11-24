"""Checks of scene_sheet_prompts.py on test_scene_sheet's synthetic sheets (questions 1 and 2, scene s/1). CPU only.

  PYTHONPATH=src python -m unittest discover -s src/mvp/benchmarks/tests -v      (from the repository root)
"""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from mvp.benchmarks import scene_sheet as ss
from mvp.benchmarks import scene_sheet_prompts as sp
from mvp.benchmarks.tests import test_scene_sheet as tss  # not imported by name, so its tests don't run twice
from mvp.benchmarks.vsibench import REASONING


class Prompts(unittest.TestCase):
    def setUp(self):
        tss.Synthetic.setUp(self)

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, *args):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(sp.main(["--root", str(self.root), "--vggt-root", str(self.vggt), *args]), 0)
        return out.getvalue()

    def test_curated_set_covers_every_option(self):
        configs = [config for _, config in sp.curated_configs()]
        self.assertEqual(len(configs), 21)
        self.assertEqual(len({label for label, _ in sp.curated_configs()}), 21)
        visibility = [v for v in ss.VISIBILITY if v != "nearby"]  # nearby needs a dense sheet: auto shows it there
        for key, values in [("source", [None] + ss.SOURCES), ("fmt", ss.SHEET_FORMATS), ("cameras", [False, True]),
                            ("time_unit", ss.TIME_UNITS), ("reasoning", REASONING), ("visibility", visibility),
                            ("step_order", ss.STEP_ORDERS), ("object_directions", ss.OBJECT_DIRECTIONS)]:
            self.assertTrue(set(values) <= {c[key] for c in configs}, key)
        shown, skipped = sp.render(1, sp.curated_configs(), self.root, self.vggt)
        self.assertEqual((len(shown), skipped), (21, []))
        for number, label, config, prompt, _ in shown:
            self.assertEqual(prompt, ss.build_sheet_prompt(
                1, config["source"], config["fmt"], self.root, self.vggt, cameras=config["cameras"],
                time_unit=config["time_unit"], reasoning=config["reasoning"], visibility=config["visibility"],
                step_order=config["step_order"], object_directions=config["object_directions"]), label)
        # the synthetic cameras are level, so the camera's axes give the same narrative as the floor
        same = {label: same_as for _, label, _, _, same_as in shown if same_as}
        # and on a 16-frame sheet auto is bridged
        self.assertEqual(same, {"narrative, object directions in the camera's axes": 15,
                                "objects as csv, bridged visibility": 3})

    def test_option_lists_combine_and_skip_contradictions(self):
        configs = sp.product_configs({"source": ["pipeline", "gold"], "fmt": ["csv", ss.NARRATIVE],
                                      "cameras": [False, True]})
        shown, skipped = sp.render(1, configs, self.root, self.vggt)
        self.assertEqual([label for _, label, *_ in shown], [
            "objects: pipeline csv | time: seconds | visibility: auto | reasoning: direct",
            "objects: pipeline csv + cameras | time: seconds | visibility: auto | reasoning: direct",
            "objects: pipeline narrative | time: seconds | visibility: auto | steps: move-turn | object directions: "
            "floor | reasoning: direct",
            "objects: gold csv | time: seconds | reasoning: direct"])
        self.assertEqual(len(skipped), 4)  # the narrative with cameras, and gold with cameras or the narrative
        # an option that changes nothing here gives the same prompt, which is not printed twice
        shown, _ = sp.render(1, sp.product_configs({"step_order": ss.STEP_ORDERS}), self.root, self.vggt)
        self.assertEqual([same_as for *_, same_as in shown], [None, 1])

    def test_command_line(self):
        out = self.run_main()
        self.assertTrue(out.startswith("question 1, s/1 (object_counting)\nHow many?\nground truth: 2\n\n21 prompts"))
        self.assertIn("===== 3/21  objects as csv (the baseline)\nobjects: pipeline csv | time: seconds | visibility: "
                      "auto | reasoning: direct\n\n<|im_start|>system", out)
        self.assertIn("===== 18/21  narrative, object directions in the camera's axes\n", out)
        self.assertIn("(the same prompt as 15)", out)
        out = self.run_main("--question", "2", "--format", "csv", "narrative", "--source", "gold", "pipeline")
        self.assertIn("3 prompts, 1 contradictory combinations skipped", out)
        self.assertIn("===== skipped, contradictory:\nobjects: gold narrative", out)
        out = self.run_main("--format", "json", "narrative", "--cameras", "off", "on")
        self.assertIn("3 prompts, 1 contradictory combinations skipped", out)
        self.assertIn("===== 2/3  objects: pipeline json + cameras |", out)
        self.assertIn("===== skipped, contradictory:\nobjects: pipeline narrative + cameras", out)
        self.assertIn("<|video_pad|> x 64", self.run_main("--format", "csv", "--frame-tokens", "64"))
        self.assertEqual(self.run_main("--list"), "s/1  (2 questions)\n")
        self.assertEqual(self.run_main("--list", "--scene", "s/1"),
                         "1  object_counting: How many?\n2  room_size_estimation: Room?\n")

    def test_out_dir(self):
        with tempfile.TemporaryDirectory() as out_dir:
            out = self.run_main("--out-dir", out_dir)
            files = sorted(p.name for p in Path(out_dir).iterdir())
            self.assertEqual(len(files), 19)  # 20 prompts, one the same as an earlier one
            self.assertEqual(files[0], "01_video_only.txt")
            self.assertIn(f"written to {Path(out_dir) / '01_video_only.txt'}", out)
            self.assertIn("These are frames of a video.", (Path(out_dir) / files[2]).read_text(encoding="utf-8"))

    def test_pick_question(self):
        self.assertEqual(sp.pick_question(self.root), 1)  # 706 isn't in the synthetic index: its first question
        self.assertEqual(sp.pick_question(self.root, 2), 2)
        self.assertEqual(sp.pick_question(self.root, scene="s/1"), 1)
        for question, scene in [(7, None), (None, "s/9"), (2, "s/9")]:
            with self.assertRaises(SystemExit):
                sp.pick_question(self.root, question, scene)


if __name__ == "__main__":
    unittest.main()
