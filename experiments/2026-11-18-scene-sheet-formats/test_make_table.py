"""Checks of make_table.py on fake prediction files, and of cluster_checks.py's prompt check on a synthetic sheet.

  PYTHONPATH=src python -m unittest discover -s experiments/2026-11-18-scene-sheet-formats -v
"""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import cluster_checks
import make_table as mt
from mvp.benchmarks import scene_sheet as ss
from mvp.benchmarks.tests import test_scene_sheet as tss  # not imported by name, so its tests don't run twice

TYPES = ["object_counting", "object_abs_distance", "object_size_estimation", "room_size_estimation",
         "object_rel_distance", "object_rel_direction_easy", "object_rel_direction_medium",
         "object_rel_direction_hard", "route_planning", "obj_appearance_order"]
README = f"# x\n\n## Results\n\n{mt.START}\nPLACEHOLDER\n{mt.END}\n\n### Examples\n"


NUMERIC = {"object_counting", "object_abs_distance", "object_size_estimation", "room_size_estimation"}


def rec(i, qtype, score):
    numeric = qtype in NUMERIC
    return {"id": i, "dataset": "d", "scene_name": "s", "question_type": qtype, "task": qtype, "question": "q",
            "options": None if numeric else ["A. a", "B. b"], "ground_truth": "3" if numeric else "A",
            "prediction": "3" if numeric else "A", "score": score, "n_frames": 16, "n_input_tokens": 100}


class MakeTable(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.ids = list(range(20))  # two questions per type
        self.qtype = {i: TYPES[i // 2] for i in self.ids}

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, scores, skip=()):
        """scores: {condition: function id -> score}."""
        for cond, _, _, stem in mt.CONDITIONS:
            rows = [rec(i, self.qtype[i], scores.get(cond, lambda i: 0.0)(i)) for i in self.ids if (cond, i) not in skip]
            (self.dir / f"{stem}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def run_table(self):
        readme = self.dir / "README.md"
        readme.write_text(README)
        tokens = lambda ids: {c: 100.0 + k for k, (c, *_) in enumerate(mt.CONDITIONS[1:])}  # B 100, C 101, ...
        df, versus = mt.run(self.dir, self.ids, tokens, out_dir=self.dir, readme=readme)
        return df, versus, readme.read_text()

    def test_scores_tests_and_outputs(self):
        # A gets nothing right; B everything; G the first question of each type; the rest nothing
        self.write({"B": lambda i: 1.0, "G": lambda i: float(i % 2 == 0)})
        df, versus, readme = self.run_table()
        self.assertEqual(df.loc["A", "overall"], 0)
        self.assertEqual(df.loc["B", "overall"], 100)
        self.assertEqual(df.loc["G", "overall"], 50)
        self.assertEqual(df.loc["B", "object_rel_direction"], 100)  # the three difficulties as one task
        self.assertEqual(df.loc["B", "type:object_rel_direction_hard"], 100)
        self.assertEqual(tuple(df.loc["B", ["better_than_A", "worse_than_A", "same_as_A"]]), (20, 0, 0))
        self.assertAlmostEqual(df.loc["B", "p_vs_A"], 2 / 2 ** 20)
        self.assertAlmostEqual(df.loc["B", "holm_p_vs_A"], 10 * 2 / 2 ** 20)  # smallest of ten: times ten
        self.assertEqual(df.loc["C", "p_vs_A"], 1.0)
        self.assertEqual(df.loc["B", "sheet_tokens"], 100)
        self.assertTrue(pd.isna(df.loc["A", "sheet_tokens"]))
        self.assertEqual(df.loc["A", "parse_rate"], 100)
        csv_ = versus[0]
        self.assertEqual((csv_["format"], csv_["pipeline"], csv_["gold"]), ("csv", "B", "G"))
        self.assertEqual((csv_["better"], csv_["worse"], csv_["same"], csv_["overall_diff"]), (10, 0, 10, 50))

        self.assertNotIn("PLACEHOLDER", readme)
        self.assertIn("| B | pipeline | csv | 100.0 |", readme)
        self.assertIn("| csv | B − G | +50.0 | 10 / 0 / 10 |", readme)
        self.assertTrue(readme.endswith(f"{mt.END}\n\n### Examples\n"))
        self.assertEqual(len(pd.read_csv(self.dir / "results.csv")), 11)

        tex = (self.dir / "results_pipeline.tex").read_text()
        self.assertIn(r"\label{tab:sheet-format}", tex)
        self.assertIn("Format & Overall & Count & Abs. dist. & Obj. size & Room size & Rel. dist. & Rel. dir. & Route "
                      r"& Order & Tokens \\", tex)
        self.assertIn(r"CSV & \textbf{100.0} &", tex)  # best overall
        self.assertIn(r"& \textbf{100} \\", tex)  # fewest tokens: B's
        self.assertIn(r"Video only & 0.0 &", tex)
        self.assertIn("--", tex)  # A has no sheet tokens
        gold = (self.dir / "results_gold.tex").read_text()
        self.assertIn(r"\label{tab:sheet-format-gold}", gold)
        self.assertIn(r"CSV & \textbf{50.0} &", gold)
        self.assertIn("on the 20 VSI-Bench questions", gold)

    def test_ties_are_all_bold(self):
        self.write({})
        self.run_table()
        tex = (self.dir / "results_gold.tex").read_text()
        self.assertIn(r"Video only & \textbf{0.0} &", tex)
        self.assertIn(r"Text & \textbf{0.0} &", tex)

    def test_question_sets_must_match(self):
        self.write({}, skip={("D", 3)})
        with self.assertRaises(SystemExit) as e:
            self.run_table()
        self.assertIn("D: 1 questions missing", str(e.exception))
        (self.dir / f"{mt.CONDITIONS[4][3]}.jsonl").unlink()
        with self.assertRaises(SystemExit) as e:
            self.run_table()
        self.assertIn("is missing", str(e.exception))

    def test_duplicates_are_refused(self):
        self.write({})
        path = self.dir / f"{mt.CONDITIONS[0][3]}.jsonl"
        path.write_text(path.read_text() + json.dumps(rec(0, self.qtype[0], 1.0)) + "\n")
        with self.assertRaises(SystemExit) as e:
            self.run_table()
        self.assertIn("A: 0 questions missing, 0 not in the list, 1 answered twice", str(e.exception))


class PromptCheck(unittest.TestCase):
    """cluster_checks.check_prompts on a prompt built from test_scene_sheet's synthetic sheet (30 fps, frame t at
    second t, so the pair (2k, 2k + 1) shows 2k + 0.5 seconds)."""
    def setUp(self):
        tss.Synthetic.setUp(self)

    def tearDown(self):
        self.tmp.cleanup()

    def prompt(self, labels):
        sheet = ss.render_sheet(1, "pipeline", "csv", self.root, vggt_root=self.vggt)
        video = "".join(f"<{t} seconds><|vision_start|><|video_pad|> x 64<|vision_end|>" for t in labels)
        return (f"===== question 1 (object_counting), /x/s/1.mp4, video fps 30.0, frames {list(range(0, 480, 30))}\n"
                f"<|im_start|>user\n{video}These are frames of a video.\n{sheet}\nHow many?<|im_end|>\n")

    def test_good_and_bad_labels(self):
        good = [f"{2 * k + 0.5:.1f}" for k in range(8)]
        self.assertEqual(cluster_checks.check_prompts(self.prompt(good), self.root, self.vggt), [])
        bad = good[:1] + ["2.4"] + good[2:]
        problems = cluster_checks.check_prompts(self.prompt(bad), self.root, self.vggt)
        self.assertEqual(len(problems), 2)  # the labels differ, and door_2's first_seen 2.5 isn't among them
        self.assertIn("not the sheet's", problems[0])
        self.assertIn("2.5", problems[1])
        self.assertEqual(cluster_checks.check_prompts("nothing printed", self.root, self.vggt),
                         ["no '===== question' blocks in the output"])

    def test_sheet_must_be_in_prompt(self):
        good = [f"{2 * k + 0.5:.1f}" for k in range(8)]
        text = self.prompt(good).replace("coffee_table_1", "coffee_table_9")
        self.assertIn("the rendered sheet is not in the prompt",
                      cluster_checks.check_prompts(text, self.root, self.vggt)[0])


if __name__ == "__main__":
    unittest.main()
