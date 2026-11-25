"""CPU tests of the follow-up's runs and analysis (no GPU, no model: a stand-in answers). They test the code, not whether
the model can reason as the prompts ask.

  PYTHONPATH=src python -m unittest discover -s experiments/2026-11-25-concise-math-cot -p "test_*.py"
"""
import contextlib
import csv
import hashlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis  # noqa: E402
import followup  # noqa: E402

TYPES = followup.COMPUTING
# the structured-CoT pilot's predictions, which the analysis reuses as baselines: they must stay as they were
PILOT = followup.REPO / "results/zeroshot/vsibench/structured_cot/explore200"
PILOT_SHA256 = {"A_video.jsonl": "69f2bc2f994ae864", "gold_csv.jsonl": "b2216572001a8fec", "gold_cot.jsonl": "f9ce594e4ddbaa63",
                "gold_relations.jsonl": "9b11b3fa2bb61d6d"}


def synthetic_questions(n_scenes=6, per_type=4):
    """{id: question}: per_type questions of each computing task (and some counting) spread over n_scenes scenes."""
    qs, i = {}, 0
    for s in range(n_scenes):
        for t in TYPES + ("object_counting",):
            for _ in range(per_type // 2):
                q = {"id": i, "dataset": "scannet", "scene_name": f"s{s}", "question_type": t,
                     "question": f"q{i}", "ground_truth": "A" if t == "object_rel_distance" else "1.0",
                     "options": ["A. bed", "B. tv"] if t == "object_rel_distance" else None}
                qs[i] = q
                i += 1
    return qs


class Subsets(unittest.TestCase):
    def test_nested_balanced_deterministic(self):
        qs = synthetic_questions()
        pool = [q for q, d in qs.items() if d["question_type"] in TYPES]
        sets = {k: followup.nested_subset(pool, qs, k) for k in (1, 3, 6, 12)}
        for k, ids in sets.items():
            counts = pd.Series([qs[q]["question_type"] for q in ids]).value_counts()
            self.assertEqual(set(counts), {k})  # balanced
            self.assertEqual(ids, followup.nested_subset(pool, qs, k))  # deterministic
        for small, large in ((1, 3), (3, 6), (6, 12)):
            self.assertTrue(set(sets[small]) <= set(sets[large]))  # nested
        self.assertNotEqual(followup.nested_subset(pool, qs, 3, seed=1), sets[3])
        with self.assertRaises(ValueError):
            followup.nested_subset(pool, qs, 13)

    def test_heldout_excludes_development_scenes(self):
        qs = synthetic_questions()
        dev = [q for q, d in qs.items() if d["scene_name"] in ("s0", "s1") and d["question_type"] in TYPES]
        held = followup.heldout_pool(qs, dev)
        self.assertTrue(held)
        self.assertFalse({qs[q]["scene_name"] for q in held} & {"s0", "s1"})
        self.assertTrue(all(qs[q]["question_type"] in TYPES for q in held))
        self.assertEqual(followup.heldout_pool(qs, dev, available={held[0]}), [held[0]])

    def test_explore200_dev_pool(self):
        path = followup.QUESTIONS
        cache = Path.home() / ".cache/huggingface/hub/datasets--nyu-visionx--VSI-Bench/snapshots"
        found = [path] if path.is_file() else sorted(cache.glob("*/test.jsonl"))
        if not found:
            self.skipTest("VSI-Bench's test.jsonl is not here")
        qs = followup.load_questions(found[0])
        pool = followup.dev_pool(qs)
        self.assertEqual(len(pool), 75)
        smoke = followup.nested_subset(pool, qs, 1)
        self.assertEqual([qs[q]["question_type"] for q in smoke], list(TYPES))
        self.assertEqual(len(followup.nested_subset(pool, qs, 25)), 75)


class Guard(unittest.TestCase):
    def test_configs_are_never_mixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "cot_f1"
            config = {"variant": "cot_f1", "prompt_fingerprint": "abc", "max_new_tokens": 2048}
            followup.guard(run, config)
            followup.guard(run, config)  # the same: resumes
            with self.assertRaises(followup.ConfigMismatch) as e:
                followup.guard(run, {**config, "prompt_fingerprint": "changed"})
            self.assertIn("prompt_fingerprint", str(e.exception))
            other = Path(tmp) / "cot_f2"
            other.mkdir()
            (other / "predictions.jsonl").write_text('{"id": 1}\n')
            with self.assertRaises(followup.ConfigMismatch):  # answers of unknown configuration
                followup.guard(other, config)

    def test_prompt_version_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "index.csv").write_text("id\n")
            q = Path(tmp) / "test.jsonl"
            q.write_text("{}\n")
            c = followup.run_config("cot_f2", tmp, questions_path=q)
            self.assertEqual(c["max_new_tokens"], 2048)
            self.assertEqual(c["sheet_source"], "gold")
            with self.assertRaises(followup.ConfigMismatch):
                followup.run_config("cot_f2", tmp, questions_path=q, prompt_version="another")
            self.assertNotEqual(c["prompt_fingerprint"], followup.run_config("cot_f3", tmp, questions_path=q)["prompt_fingerprint"])


class Resume(unittest.TestCase):
    """followup.run with the eval replaced: only unanswered questions are asked, a larger k reuses a smaller one's."""

    def test_incremental(self):
        qs = synthetic_questions()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            qfile = tmp / "test.jsonl"
            qfile.write_text("".join(json.dumps(q) + "\n" for q in qs.values()))
            (tmp / "sheets").mkdir()
            (tmp / "sheets" / "index.csv").write_text("id\n")
            pool = [q for q, d in qs.items() if d["question_type"] in TYPES]
            asked = []

            def fake_eval(cmd, check, cwd, env):
                ids = [int(x) for x in Path(cmd[cmd.index("--ids") + 1]).read_text().split()]
                out = Path(cmd[cmd.index("--out") + 1])
                done = followup.answered(out.parent)
                new = [q for q in ids if q not in done]
                asked.append(new)
                with open(out, "a") as f:
                    for q in new:
                        f.write(json.dumps({"id": q, "prediction": "Answer: 1"}) + "\n")

            with mock.patch.object(followup.subprocess, "run", side_effect=fake_eval), \
                    mock.patch.object(followup, "write_prompts", side_effect=lambda run_dir, ids, *a: Path(run_dir, "prompts.jsonl").write_text(
                        "".join(json.dumps({"id": q}) + "\n" for q in ids))):
                followup.run("cot_f1", 1, tmp / "sheets", tmp / "out", pool, questions_path=qfile)
                followup.run("cot_f1", 2, tmp / "sheets", tmp / "out", pool, questions_path=qfile)
                _, _, todo = followup.run("cot_f1", 2, tmp / "sheets", tmp / "out", pool, questions_path=qfile)
            self.assertEqual([len(a) for a in asked], [3, 3])  # 3, then only the 3 new ones; the third call asks nothing
            self.assertEqual(todo, [])
            self.assertEqual(len(followup.answered(tmp / "out" / "cot_f1")), 6)
            timing = [json.loads(l) for l in (tmp / "out/cot_f1/timing.jsonl").read_text().splitlines()]
            self.assertEqual([t["new"] for t in timing], [3, 3])
            self.assertTrue((tmp / "out/cot_f1/config.json").is_file())
            # another seed is another configuration: the folder refuses it
            with self.assertRaises(followup.ConfigMismatch):
                followup.run("cot_f1", 2, tmp / "sheets", tmp / "out", pool, seed=1, questions_path=qfile, dry_run=True)

    def test_eval_resumes_with_a_variant(self):
        """The eval's main() with --reasoning cot_f2 and a stand-in model: the variant's prompt reaches the model, the
        final-answer rule scores it, and a second call with more ids asks only the new ones."""
        from mvp.eval import vsibench as ev
        calls = []

        class StandIn:
            def __init__(self, model_id):
                pass

            def load_video_frames(self, path, indices):
                import numpy as np
                return np.zeros((16, 3, 64, 64)), {"frames_indices": list(indices)}, {}

            def generate_video(self, video, meta, vkw, prompts, pres=None, system=None, max_new_tokens=16):
                calls.append(prompts)
                self.last_new_tokens = [(12, False)] * len(prompts)
                self.last_prompt_tokens = [100] * len(prompts)
                return ["Objects: a\nAttributes: b\nCalculation: c\nAnswer: 1.0"] * len(prompts), 100

        docs = pd.DataFrame([{"id": i, "dataset": "scannet", "scene_name": "s0", "question_type": "object_abs_distance",
                              "question": f"distance {i}?", "options": None, "ground_truth": "1.0",
                              "video_path": "/no/such/scannet/s0.mp4"} for i in range(4)])
        fake = types.ModuleType("mvp.models.qwen_vl")
        fake.QwenVL, fake.SYSTEM_PROMPT = StandIn, "You are a helpful assistant."
        with tempfile.TemporaryDirectory() as tmp:
            out = f"{tmp}/p.jsonl"
            for ids in ([0, 1], [0, 1, 2, 3]):
                Path(f"{tmp}/ids.txt").write_text("\n".join(map(str, ids)))
                argv = ["vsibench", "--model", "m", "--nframes", "16", "--sampling", "uniform128", "--reasoning",
                        "cot_f2", "--sheet-source", "gold", "--sheet-format", "csv", "--sheets-root", tmp, "--ids",
                        f"{tmp}/ids.txt", "--out", out]
                with mock.patch.dict(sys.modules, {"mvp.models.qwen_vl": fake}), mock.patch.object(sys, "argv", argv), \
                        mock.patch.object(ev, "load_vsibench", return_value=docs), \
                        mock.patch.object(ev, "load_index", return_value=set(range(4))), \
                        mock.patch.object(ev, "render_scene", return_value="SHEET"), \
                        mock.patch.object(ev, "frame_indices", return_value=list(range(0, 128, 8))), \
                        contextlib.redirect_stdout(io.StringIO()):
                    ev.main()
            rows = [json.loads(l) for l in Path(out).read_text().splitlines()]
            summary = json.loads(Path(out.replace(".jsonl", "_summary.json")).read_text())
        self.assertEqual([len(c) for c in calls], [2, 2])
        self.assertTrue(all("Each row is a box" in p for c in calls for p in c))  # F2's formulas in the prompt
        self.assertEqual(sorted(r["id"] for r in rows), [0, 1, 2, 3])
        self.assertTrue(all(r["score"] == 1.0 and "score_first" in r for r in rows))  # the final-answer rule
        self.assertEqual((summary["reasoning"], summary["max_new_tokens"]), ("cot_f2", 2048))


class Preservation(unittest.TestCase):
    def test_pilot_predictions_unchanged(self):
        if not PILOT.is_dir():
            self.skipTest("the pilot's predictions are not here")
        for name, prefix in PILOT_SHA256.items():
            digest = hashlib.sha256((PILOT / name).read_bytes()).hexdigest()
            self.assertTrue(digest.startswith(prefix), name)
        self.assertEqual(len((PILOT / "gold_cot.jsonl").read_text().splitlines()), 200)

    def test_pilot_conditions_named_as_before(self):
        self.assertEqual({c: s for c, (s, _) in analysis.BASELINE_FILES.items()},
                         {"A": "A_video", "C": "gold_csv", "D": "gold_cot", "E": "gold_relations"})


class Analysis(unittest.TestCase):
    """analysis.report on synthetic runs: measures, comparisons and the automatic error flags."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        # gold sheets: question 1 absolute distance (closest 1.0 m, centre 2.0 m), 2 relative distance (closest A,
        # centre B), 3 object size (longest 1.5 m)
        rows = {1: [("bed", 0, 0, 0, 2, 1, 1), ("tv", 2, 0, 0, 0, 1, 1)],
                2: [("bed", 0, 0, 0, 1, 1, 1), ("chair", 2.6, 0, 0, 3, 1, 1), ("tv", 1.8, 0, 0, 0.2, 0.2, 0.2)],
                3: [("cabinet", 0, 0, 0, 0.5, 1.5, 0.8)]}
        prompts = {1: "bed, tv", 2: "chair, tv, bed", 3: "cabinet"}
        with open(root / "index.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "gold_sheet", "prompts"])
            for q, objs in rows.items():
                folder = root / f"g{q}"
                folder.mkdir()
                with open(folder / "objects.csv", "w", newline="") as g:
                    gw = csv.writer(g)
                    gw.writerow(["id", "concept", "track_id", "x", "y", "z", "size_x", "size_y", "size_z"])
                    for i, (c, *v) in enumerate(objs):
                        gw.writerow([f"{c}_1", c, 1, *v])
                w.writerow([q, f"g{q}", prompts[q]])
        self.sheets = root
        types_ = {1: "object_abs_distance", 2: "object_rel_distance", 3: "object_size_estimation"}
        truth = {1: "1.0", 2: "A", 3: "150"}
        def write(path, answers, tokens=40, cut=False):
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                for q, a in answers.items():
                    doc = {"id": q, "question_type": types_[q], "ground_truth": truth[q],
                           "options": ["A. chair", "B. tv"] if q == 2 else None}
                    from mvp.benchmarks.vsibench import score_final
                    f.write(json.dumps({**doc, "prediction": a, "score": score_final(doc, a), "score_first": 0.0,
                                        "n_new_tokens": tokens, "hit_token_limit": cut}) + "\n")
        base = root / "base"
        write(base / "A_video.jsonl", {1: "3", 2: "B", 3: "90"}, tokens=1)
        write(base / "gold_csv.jsonl", {1: "2.0", 2: "B", 3: "80"}, tokens=1)
        write(base / "gold_cot.jsonl", {1: "Answer: 2.0", 2: "Answer: B", 3: "Answer: 1.5"}, tokens=900)
        write(base / "gold_relations.jsonl", {1: "1.0", 2: "A", 3: "150"}, tokens=1)
        out = root / "out"
        write(out / "cot_f1/predictions.jsonl", {1: "Answer: 2.0", 2: "Answer: B", 3: "Answer: 150"}, tokens=200)
        write(out / "cot_f2/predictions.jsonl", {1: "Answer: 1.0", 2: "Answer: A", 3: "Answer: 150"}, tokens=300)
        write(out / "cot_f3/predictions.jsonl", {1: "Answer: 1.0", 2: "Calculation: ...", 3: "Answer: 150"}, tokens=2048, cut=True)
        (out / "cot_f1/timing.jsonl").write_text(json.dumps({"new": 3, "seconds": 30.0}) + "\n")
        self.base, self.out = base, out

    def tearDown(self):
        self.tmp.cleanup()

    def test_report(self):
        text, runs, comps, errors = analysis.report(self.out, self.sheets, self.base)
        self.assertIn("**Exploratory: 3 questions (1 per task).**", text)
        m = {c: analysis.measures(runs[c]) for c in runs}
        self.assertAlmostEqual(m["F2"]["computing"], 100.0)
        self.assertAlmostEqual(m["F3"]["parse rate"], 100 * 2 / 3)
        self.assertAlmostEqual(m["F3"]["cut off"], 100.0)
        self.assertEqual((m["D"]["reply tokens median"], m["F1"]["reply tokens p90"]), (900, 200))
        self.assertIn("10.0 |", text)  # F1's seconds per question
        primary = [c for c in comps if c["family"].startswith("primary")]
        self.assertEqual([c["comparison"] for c in primary], ["F1 - D", "F2 - F1", "F3 - F2"])
        f2 = next(c for c in primary if c["comparison"] == "F2 - F1")
        self.assertEqual((f2["better"], f2["worse"]), (2, 0))
        self.assertGreaterEqual(f2["holm p"], f2["p"])
        by = errors.set_index(["condition", "id"])
        self.assertTrue(by.loc[("D", 1), "equals_centre_rule"])  # D's 2.0 m is the centre distance
        self.assertTrue(by.loc[("D", 2), "picks_centre_rule"])
        self.assertTrue(by.loc[("D", 3), "metres_not_cm"])  # 1.5 for 150 cm
        self.assertTrue(by.loc[("F2", 1), "equals_closest_rule"])
        self.assertTrue(by.loc[("F3", 2), "unread"] and by.loc[("F3", 2), "cut_off"])
        self.assertFalse(by["sheet_rule_disagrees_with_truth"].any())
        self.assertEqual(set(c for c in errors.columns if c.startswith("manual_")),
                         {"manual_objects", "manual_attributes", "manual_operation", "manual_arithmetic",
                          "manual_unit", "manual_notes"})
        self.assertTrue((self.out / "analysis.md").is_file() and (self.out / "errors.csv").is_file())

    def test_conditions_must_share_questions(self):
        (self.out / "cot_f3/predictions.jsonl").write_text("")  # an empty run leaves no common question
        with self.assertRaises(Exception):
            analysis.report(self.out, self.sheets, self.base)
