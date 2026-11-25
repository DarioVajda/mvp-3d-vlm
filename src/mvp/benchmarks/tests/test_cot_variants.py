"""The follow-up CoT variants (mvp.benchmarks.cot_variants): what each condition's prompt holds, the worked examples'
arithmetic against the code that computes the precomputed relations (scene_sheet.box_gap), the unit conversion, and
the final-answer parser on replies in the variants' structure. They test the prompts, not whether the model follows
them."""
import csv
import io
import re
import unittest
from unittest import mock

import numpy as np

from mvp.benchmarks import cot_variants as cv
from mvp.benchmarks import scene_sheet as ss
from mvp.benchmarks import vsibench as vb

QUESTIONS = {
    "object_abs_distance": {"question_type": "object_abs_distance", "question": "Measuring from the closest point of "
                            "each object, what is the distance between the bed and the tv (in meters)?"},
    "object_rel_distance": {"question_type": "object_rel_distance", "question": "Measuring from the closest point of "
                            "each object, which of these objects (chair, tv) is the closest to the bed?",
                            "options": ["A. chair", "B. tv"]},
    "object_size_estimation": {"question_type": "object_size_estimation", "question": "What is the length of the "
                               "longest dimension (length, width, or height) of the tv, measured in centimeters?"},
}


def example_parts(task):
    """(rows by id as (centre, extent), the example's reply lines) of a worked example."""
    text = cv.EXAMPLES[task]
    sheet, rest = text.split("\nQuestion:", 1)
    rows = {r["id"]: (np.array([float(r[k]) for k in "xyz"]), np.array([float(r[f"size_{k}"]) for k in "xyz"]))
            for r in csv.DictReader(io.StringIO(sheet))}
    reply = rest[rest.index("Objects:"):]
    return rows, reply


class Conditions(unittest.TestCase):
    def prompt(self, task, variant):
        return vb.build_prompt(QUESTIONS[task], "SHEET", variant, "gold")

    def test_each_condition_has_only_its_intervention(self):
        for task in cv.COMPUTING_TYPES:
            f1, f2, f3 = (self.prompt(task, v) for v in cv.VARIANTS)
            hint = vb.COT_HINTS[vb.task_of(task)]
            for p in (f1, f2, f3):
                self.assertIn(cv.CONCISE, p)
                self.assertIn(hint, p)  # the task's hint, unchanged from condition D
                self.assertIn(cv.FORMAT, p)
                self.assertTrue(p.endswith(vb.COT_ANSWER["mca" if task in vb.MCA_QUESTION_TYPES else "na"]))
                self.assertNotIn("check that the result agrees with what the video shows", p)  # D's video check
            self.assertNotIn(cv.MATH[task], f1)
            self.assertIn(cv.MATH[task], f2)
            self.assertIn(cv.MATH[task], f3)
            for t in cv.COMPUTING_TYPES:  # examples in F3 only, and only the question's own task's
                self.assertNotIn(cv.EXAMPLES[t], f1)
                self.assertNotIn(cv.EXAMPLES[t], f2)
                self.assertEqual(cv.EXAMPLES[t] in f3, t == task)
            # F2 is F1 with the formulas added, F3 is F2 with the example added, nothing else changes
            self.assertEqual(f2.replace(cv.MATH[task] + "\n", ""), f1)
            self.assertEqual(f3.replace(cv.EXAMPLE_INTRO + "\n" + cv.EXAMPLES[task] + "\n" + cv.EXAMPLE_OUTRO + "\n", ""), f2)

    def test_task_specific_formulas(self):
        self.assertIn("closest points, not their centres", cv.MATH["object_abs_distance"])
        self.assertIn("smallest distance", cv.MATH["object_rel_distance"])
        self.assertIn("largest of size_x, size_y and size_z", cv.MATH["object_size_estimation"])
        self.assertIn("multiplying by 100", cv.MATH["object_size_estimation"])
        self.assertIn("do not convert", cv.MATH["object_abs_distance"])

    def test_only_computing_tasks_with_the_gold_sheet(self):
        with self.assertRaises(ValueError):
            vb.build_prompt({"question_type": "object_counting", "question": "How many chairs?"}, "SHEET", "cot_f1", "gold")
        with self.assertRaises(ValueError):
            vb.build_prompt(QUESTIONS["object_abs_distance"], "SHEET", "cot_f2", "pipeline")
        with self.assertRaises(ValueError):
            vb.build_prompt(QUESTIONS["object_abs_distance"], None, "cot_f1", None)

    def test_condition_d_is_unchanged(self):
        """D's instructions are still COT_STEPS with the sheet, the hint and the answer line (as cot_prompt.md
        records them), and cot is still an eval option."""
        self.assertEqual(vb.REASONING, ["direct", "cot"])  # the variants are not general reasoning modes
        for task, q in QUESTIONS.items():
            d = vb.build_prompt(q, "SHEET", "cot", "gold")
            self.assertTrue(d.endswith(vb.COT_STEPS.format(where=" and in the scene information above") + " "
                                       + vb.COT_HINTS[vb.task_of(task)] + "\n"
                                       + vb.COT_ANSWER["mca" if task in vb.MCA_QUESTION_TYPES else "na"]))
        self.assertIn("Before answering, check that the result agrees with what the video shows.", vb.COT_STEPS)

    def test_fingerprint(self):
        prints = {v: cv.fingerprint(v, vb.COT_HINTS, vb.COT_ANSWER) for v in cv.VARIANTS}
        self.assertEqual(len(set(prints.values())), 3)
        with mock.patch.dict(cv.MATH, {"object_abs_distance": cv.MATH["object_abs_distance"] + " Changed."}):
            self.assertEqual(cv.fingerprint("cot_f1", vb.COT_HINTS, vb.COT_ANSWER), prints["cot_f1"])  # F1 has no MATH
            self.assertNotEqual(cv.fingerprint("cot_f2", vb.COT_HINTS, vb.COT_ANSWER), prints["cot_f2"])
        with mock.patch.dict(cv.EXAMPLES, {"object_size_estimation": "changed"}):
            self.assertNotEqual(cv.fingerprint("cot_f3", vb.COT_HINTS, vb.COT_ANSWER), prints["cot_f3"])
            self.assertEqual(cv.fingerprint("cot_f2", vb.COT_HINTS, vb.COT_ANSWER), prints["cot_f2"])


class Geometry(unittest.TestCase):
    """The formula F2 states is the one the precomputed relations use (scene_sheet.box_gap)."""

    @staticmethod
    def stated_formula(a, b):
        gaps = [max(0.0, abs(a[0][k] - b[0][k]) - (a[1][k] + b[1][k]) / 2) for k in range(3)]
        return float(np.sqrt(sum(g * g for g in gaps)))

    def test_formula_equals_box_gap(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            a = (rng.uniform(-3, 3, 3), rng.uniform(0.05, 2, 3))
            b = (rng.uniform(-3, 3, 3), rng.uniform(0.05, 2, 3))
            self.assertAlmostEqual(self.stated_formula(a, b), ss.box_gap(a, b), places=12)

    def test_overlapping_and_touching_boxes(self):
        a = (np.array([0.0, 0.0, 0.0]), np.array([2.0, 2.0, 2.0]))
        self.assertEqual(ss.box_gap(a, (np.array([0.5, 0.5, 0.5]), np.array([1.0, 1.0, 1.0]))), 0.0)  # inside
        self.assertEqual(ss.box_gap(a, (np.array([2.0, 0.0, 0.0]), np.array([2.0, 1.0, 1.0]))), 0.0)  # touching
        # overlapping along two axes: the distance is the gap along the third
        self.assertAlmostEqual(ss.box_gap(a, (np.array([0.0, 0.0, 3.0]), np.array([1.0, 1.0, 1.0]))), 1.5)

    def test_worked_examples(self):
        rows, reply = example_parts("object_abs_distance")
        d = ss.box_gap(rows["sofa_1"], rows["lamp_1"])
        self.assertEqual(re.search(r"Answer: ([\d.]+)", reply).group(1), f"{d:.3f}")
        rows, reply = example_parts("object_rel_distance")
        target = rows["desk_1"]
        options = {"A": rows["bed_1"], "B": rows["shelf_1"], "C": rows["chair_1"]}
        closest = min(options, key=lambda o: ss.box_gap(options[o], target))
        by_centre = min(options, key=lambda o: np.linalg.norm(options[o][0] - target[0]))
        self.assertEqual(re.search(r"Answer: ([A-D])", reply).group(1), closest)
        self.assertNotEqual(closest, by_centre)  # the example separates the two rules on purpose
        for name, opt in (("bed", "A"), ("shelf", "B"), ("chair", "C")):
            self.assertIn(f"{name}: gaps", reply)
            self.assertIn(f"distance {ss.box_gap(options[opt], target):.3f}", reply)
        rows, reply = example_parts("object_size_estimation")
        longest = float(rows["cabinet_1"][1].max())
        self.assertEqual(f"{longest:.3f}", "1.830")
        self.assertEqual(re.search(r"Answer: (\d+)", reply).group(1), str(round(longest * 100)))  # metres to cm

    def test_examples_use_the_gold_sheet_csv(self):
        header = "id,label,x,y,z,size_x,size_y,size_z"
        self.assertEqual(ss.render_csv([], ["id", "label", *ss.POSITION, *ss.EXTENT]).splitlines()[0], header)
        for task in cv.COMPUTING_TYPES:
            self.assertTrue(cv.EXAMPLES[task].startswith(header + "\n"))
            self.assertEqual([l.split(":")[0] for l in example_parts(task)[1].splitlines()[-4:]],
                             ["Objects", "Attributes", "Calculation", "Answer"])


class Parser(unittest.TestCase):
    def test_final_answer_rule_reads_the_structure(self):
        for task in cv.COMPUTING_TYPES:
            _, reply = example_parts(task)
            truth = reply.strip().splitlines()[-1].split(": ", 1)[1]
            doc = {"question_type": task, "ground_truth": truth, "options": ["A. a", "B. b", "C. c"]}
            self.assertEqual(vb.score_final(doc, reply), 1.0, task)
            self.assertTrue(vb.parsed(doc, reply, True))
        # a centre distance written in the same structure is read as such and scores below the closest-point answer
        doc = {"question_type": "object_abs_distance", "ground_truth": "1.06"}
        self.assertLess(vb.score_final(doc, "Objects: a\\nCalculation: centre distance 2.43\\nAnswer: 2.43"), 0.5)
