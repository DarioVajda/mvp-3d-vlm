"""Checks of the paired comparison in mvp.reporting.compare_runs and of the parse rule in mvp.benchmarks.vsibench.

  PYTHONPATH=src python -m unittest discover -s src/mvp/reporting/tests -v      (from the repository root)
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from mvp.benchmarks.vsibench import parsed
from mvp.reporting import compare_runs as cr


class SignTest(unittest.TestCase):
    def test_exact_values(self):
        self.assertAlmostEqual(cr.sign_test(10, 0), 2 / 2 ** 10)  # both tails of an all-one-way split
        self.assertAlmostEqual(cr.sign_test(0, 10), 2 / 2 ** 10)
        self.assertAlmostEqual(cr.sign_test(3, 1), 2 * (1 + 4) / 2 ** 4)
        self.assertEqual(cr.sign_test(5, 5), 1.0)
        self.assertEqual(cr.sign_test(0, 0), 1.0)

    def test_large_counts(self):
        # README's space-aware vs uniform, 2B at 8 frames: 1102 better / 943 worse, p < 0.001
        self.assertLess(cr.sign_test(1102, 943), 0.001)
        self.assertGreater(cr.sign_test(941, 888), 0.2)  # 4B at 16 frames: p = 0.22

    def test_holm(self):
        self.assertEqual([round(p, 6) for p in cr.holm([0.01, 0.04, 0.03])], [0.03, 0.06, 0.06])
        self.assertEqual(cr.holm([0.5, 0.9]), [1.0, 1.0])

    def test_paired_ties_and_order(self):
        ref = pd.Series([0.0, 1.0, 0.5, 0.2], index=[4, 5, 6, 7])
        other = pd.Series([0.3, 1.0, 0.0, 0.0], index=[7, 5, 6, 4])  # same questions, other order: 7 up, 6 down
        self.assertEqual(cr.paired(ref, other), (1, 1, 2))


class Parsed(unittest.TestCase):
    def test_rules(self):
        mca = {"question_type": "object_rel_distance", "options": ["A. tv", "B. sofa", "C. bed"]}
        na = {"question_type": "object_counting", "options": None}
        self.assertTrue(parsed(mca, "B. sofa"))
        self.assertTrue(parsed(mca, "c"))
        self.assertFalse(parsed(mca, "D"))  # not one of the options
        self.assertFalse(parsed(mca, "The sofa"))
        self.assertTrue(parsed(na, "about three"))
        self.assertFalse(parsed(na, "I can't tell"))

    def test_final_rules(self):
        mca = {"question_type": "object_rel_distance", "options": ["A. tv", "B. sofa", "C. bed"]}
        na = {"question_type": "object_counting", "options": None}
        reply = "The sofa is nearer than the bed.\nThe answer is B."
        self.assertFalse(parsed(mca, reply))  # the first word is not a letter
        self.assertTrue(parsed(mca, reply, final=True))
        self.assertFalse(parsed(mca, "The answer is D.", final=True))  # not one of the options
        self.assertFalse(parsed(mca, "Let me look at the frames", final=True))  # cut off before the answer
        self.assertTrue(parsed(na, "I see 2 on the left and 3 more, so 5", final=True))
        self.assertFalse(parsed(na, "I can't tell", final=True))


class Main(unittest.TestCase):
    def test_report(self):
        def rec(i, score, prediction):
            return {"id": i, "dataset": "d", "scene_name": "s", "question_type": "object_rel_distance",
                    "task": "object_rel_distance", "question": "q", "options": ["A. a", "B. b"], "ground_truth": "A",
                    "prediction": prediction, "score": score, "n_frames": 16}
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for name, rows in [("ref", [rec(1, 0.0, "B"), rec(2, 0.0, "x"), rec(3, 1.0, "A")]),
                               ("new", [rec(1, 1.0, "A"), rec(2, 1.0, "A"), rec(3, 1.0, "A")])]:
                paths.append(Path(tmp) / f"{name}.jsonl")
                paths[-1].write_text("".join(json.dumps(r) + "\n" for r in rows))
            out = io.StringIO()
            with mock.patch.object(sys, "argv", ["compare_runs", *map(str, paths)]), contextlib.redirect_stdout(out):
                cr.main()
        text = out.getvalue()
        self.assertIn("parsed 66.7%", text)  # "x" isn't an option letter
        self.assertIn("new", text)
        self.assertIn("2 better / 0 worse / 1 same", text)
        self.assertIn("p = 0.5", text)


if __name__ == "__main__":
    unittest.main()
