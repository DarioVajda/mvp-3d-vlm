"""Tests of v5's code that need no GPU: the tracker patch (on the revamped tracker, read from git), the dev run's
records, the ported post-processing, and the scorecard's box measures.

  PYTHONPATH=src python -m unittest discover -s experiments/2026-11-18-scene-sheet-formats -p "test_v5.py"

With V5_SMOKE_RUN (the smoke run's root, <dataset>__<scene>/all/), the post-processing also runs end to end on that
run (TestSmoke), with this repository's src/."""
import ast
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pipeline_scorecard as sc  # noqa: E402
import rerun_detection as rd  # noqa: E402
import v5_dev  # noqa: E402
import v5_postprocess as v5  # noqa: E402


def git_show(path, commit=v5_dev.CODE_COMMIT):
    r = subprocess.run(["git", "-C", str(HERE), "show", f"{commit}:{path}"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


class TrackerPatch(unittest.TestCase):
    def setUp(self):
        self.source = git_show(str(v5_dev.TRACKER))
        if self.source is None:
            self.skipTest(f"{v5_dev.CODE_COMMIT[:7]} is not in this clone (git fetch origin revamped-pipeline)")

    def test_pinned_file_is_the_branchs(self):
        import hashlib
        self.assertEqual(hashlib.sha256(self.source.encode()).hexdigest(), v5_dev.CODE_FILES[str(v5_dev.TRACKER)])

    def test_det_score_and_scores(self):
        patched = v5_dev.patch_tracker(self.source)
        ast.parse(patched)
        self.assertNotIn("0.55)", patched.replace('default=0.55, help="detection score kept by tracking")', ""))
        self.assertEqual(patched.count("ARGS.det_score"), 3)  # SavedMasks, the per-detection check, summary.json
        self.assertIn('p.add_argument("--det-score", type=float, default=0.55', patched)
        self.assertIn('.scores.json").write_text(json.dumps(', patched)
        self.assertIn('summary["det_score"] = ARGS.det_score', patched)
        # only the patched lines differ
        added = set(patched.splitlines()) - set(self.source.splitlines())
        self.assertEqual(len(added), 8, sorted(added))

    def test_a_changed_tracker_is_refused(self):
        with self.assertRaises(SystemExit):  # an anchor gone
            v5_dev.patch_tracker(self.source.replace("if scores[k] < 0.55:", "if scores[k] <= 0.55:"))
        with self.assertRaises(SystemExit):  # an anchor twice
            v5_dev.patch_tracker(self.source + "\n        saved = SavedMasks(ARGS.masks_dir, prompts, frames, 0.55)\n")


class DevRun(unittest.TestCase):
    def test_configs(self):
        self.assertEqual(v5_dev.CONFIGS, {"v5-his": (0.55, 50), "v5-det": (0.4, 50), "v5-pts": (0.55, 300),
                                          "v5-ours": (0.4, 300)})

    def test_check_code(self):
        v5_dev.check_code(HERE.parents[1])  # this branch has the revamped code since its merge
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SystemExit):
            v5_dev.check_code(tmp)

    def test_geometry_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self.assertFalse(v5_dev.geometry_ready(d, "scannet/s1"))
            (d / "frames").mkdir()
            for t in range(128):
                (d / "frames" / f"{t:03d}.jpg").write_bytes(b"")
            for f in ("cameras.npz", "depth.npz"):
                (d / f).write_bytes(b"")
            report = {"scene": "scannet/s1", "selected_frames": list(range(128)), "depth_npz": "written",
                      "depth_source": "dense"}
            (d / "adapter_report.json").write_text(json.dumps(report))
            self.assertTrue(v5_dev.geometry_ready(d, "scannet/s1"))
            self.assertFalse(v5_dev.geometry_ready(d, "scannet/s2"))  # another scene's
            (d / "adapter_report.json").write_text(json.dumps({**report, "depth_source": "patch_points"}))
            self.assertFalse(v5_dev.geometry_ready(d, "scannet/s1"))  # not the dense depth
            (d / "adapter_report.json").write_text(json.dumps({**report, "selected_frames": list(range(16))}))
            self.assertFalse(v5_dev.geometry_ready(d, "scannet/s1"))  # not the 128 frames

    def test_question_rows_and_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            qs, ps = Path(tmp) / "test.jsonl", Path(tmp) / "prompts.jsonl"
            text = "Measuring from the closest point of each object, which of these objects (table, sofa) is the closest to the trash bin?"
            qs.write_text(json.dumps({"id": 7, "dataset": "scannet", "scene_name": "s1", "question_type":
                                      "object_rel_distance", "question": text}) + "\n" + json.dumps(
                {"id": 8, "dataset": "scannet", "scene_name": "s2", "question_type": "object_counting",
                 "question": "How many chair(s) are in this room?"}) + "\n")
            ps.write_text(json.dumps({"id": 7, "prompts": ["table", "sofa", "trash", "bin"]}) + "\n"
                          + json.dumps({"id": 8, "prompts": ["chair", "room"]}) + "\n")
            rows = v5_dev.question_rows({"scannet/s1"}, qs, ps)
        self.assertEqual([r["id"] for r in rows], [7])
        self.assertEqual(rows[0]["prompts"], "table, sofa, trash, bin")
        names = rd.scene_prompts(rows, v5_dev.PROMPT_SET)
        # fixed_v2: "trash bin" kept together with its synonyms, "sofa" without the dropped "couch"
        self.assertEqual(names, {"table": ["table"], "sofa": ["sofa"],
                                 "trash bin": ["trash bin", "trash can", "garbage bin"]})
        record = v5_dev.question_record(rows[0], list(names), {"min_points": 50}, "/x")
        self.assertEqual((record["id"], record["axes"], record["frames"], record["size_trim"]), (7, "upright", "all", 0.02))
        self.assertEqual(json.loads(v5.build_key({**record, "tracking": {"conf_percentile": 55.0}})),
                         [list(names), 55.0, None, None, None, 0.02, "upright", False])


class Stages(unittest.TestCase):
    """The dev run in two stages (README, "v5 on the dev scenes"): stage 1 is the smoke scene and dev_scenes_15, stage 2
    the same command on dev_scenes.txt, which must skip stage 1's finished scenes."""

    def test_stage_lists(self):
        stage1 = v5_dev.read_scenes(HERE / "dev_scenes_stage1.txt")
        dev15, dev35 = (v5_dev.read_scenes(HERE / f) for f in ("dev_scenes_15.txt", "dev_scenes_35.txt"))
        dev50 = v5_dev.read_scenes(HERE / "dev_scenes.txt")
        self.assertEqual(stage1, ["scannet/scene0025_01"] + dev15)
        self.assertEqual(sorted(dev15 + dev35), sorted(dev50))
        self.assertEqual(len(set(dev50) - set(stage1)), 34)  # stage 2's new scenes

    def test_unfinished_scenes(self):
        with tempfile.TemporaryDirectory() as out:
            for c in v5_dev.CONFIGS:  # a: finished everywhere; b: in one config only
                v5_dev.done_path(out, c, "d/a").parent.mkdir(parents=True)
                v5_dev.done_path(out, c, "d/a").write_text("{}")
            v5_dev.done_path(out, "v5-his", "d/b").parent.mkdir(parents=True)
            v5_dev.done_path(out, "v5-his", "d/b").write_text("{}")
            self.assertEqual(v5_dev.unfinished_scenes(["d/a", "d/b", "d/c"], out), ["d/b", "d/c"])

    def test_a_later_stage_skips_finished_scenes(self):
        """main() with the GPU parts replaced: stage 1 runs its scenes, stage 2 (a longer list, the same --out) runs
        only the new ones and loads no detector when nothing is left."""
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "code" / v5_dev.TRACKER).parent.mkdir(parents=True)
            (tmp / "code" / v5_dev.TRACKER).write_text("tracker")
            (tmp / "stage1.txt").write_text("d/a\nd/b\n")
            (tmp / "stage2.txt").write_text("d/a\nd/b\nd/c\n")
            ran, detectors = [], []

            def fake_run(code, qss, sheet, upright, detector, rows, scene, work, out, *rest):
                ran.append(scene)
                for c in v5_dev.CONFIGS:
                    v5_dev.done_path(out, c, scene).parent.mkdir(parents=True, exist_ok=True)
                    v5_dev.done_path(out, c, scene).write_text("{}")

            def main(scenes):
                argv = ["v5_dev.py", "--code", str(tmp / "code"), "--scenes", str(scenes), "--work", str(tmp / "work"),
                        "--out", str(tmp / "out")]
                fakes = {m: mock.MagicMock() for m in ("question_scene_sheet", "scene_sheet_creator", "upright")}
                with mock.patch.object(sys, "argv", argv), mock.patch.dict(sys.modules, fakes), \
                        mock.patch.object(v5_dev, "check_code"), mock.patch.object(v5_dev, "patch_tracker", lambda s: s), \
                        mock.patch.object(v5_dev, "question_rows", lambda scenes: [{"scene": s} for s in scenes]), \
                        mock.patch.object(v5_dev.rd, "load_detector", lambda code: detectors.append(1) or "det"), \
                        mock.patch.object(v5_dev, "run_scene", fake_run), mock.patch("builtins.print"), \
                        self.assertRaises(SystemExit) as e:
                    v5_dev.main()
                return e.exception.code

            self.assertEqual(main(tmp / "stage1.txt"), 0)
            self.assertEqual(ran, ["d/a", "d/b"])
            self.assertEqual(main(tmp / "stage2.txt"), 0)
            self.assertEqual(ran, ["d/a", "d/b", "d/c"])  # stage 2 ran only the new scene
            self.assertEqual(main(tmp / "stage2.txt"), 0)
            self.assertEqual((ran, len(detectors)), (["d/a", "d/b", "d/c"], 2))  # nothing left: no detector loaded

    def test_expected_counts(self):
        """The counts the README and the agent prompt give for each stage."""
        cache = Path.home() / ".cache/huggingface/hub/datasets--nyu-visionx--VSI-Bench/snapshots"
        found = [v5_dev.QUESTIONS] if v5_dev.QUESTIONS.is_file() else sorted(cache.glob("*/test.jsonl"))
        if not found:
            self.skipTest("VSI-Bench's test.jsonl is not here")
        stage1 = v5_dev.read_scenes(HERE / "dev_scenes_stage1.txt")
        dev50 = v5_dev.read_scenes(HERE / "dev_scenes.txt")
        self.assertEqual(len(v5_dev.question_rows(set(stage1), found[0])), 593)
        self.assertEqual(len(v5_dev.question_rows({"scannet/scene0025_01"}, found[0])), 38)
        self.assertEqual(len(v5_dev.question_rows(set(dev50), found[0])), 1813)


class Stage1Analysis(unittest.TestCase):
    """v5_stage1_analysis.py's reading of the relation texts and its stage-2 decision rule (README's criteria)."""

    def test_relation_answers(self):
        import v5_stage1_analysis as a
        abs_q = {"question_type": "object_abs_distance", "ground_truth": "2.0"}
        text = ("Relations between ...:\nobject_a,object_b,distance\ndoor_1,fan_1,0.526\n"
                "Closest-point distance between the door and the fan: 0.526 m.")
        self.assertEqual(a.relation_answer(abs_q, text), 0.526)
        self.assertIsNone(a.relation_answer(abs_q, None))
        rel_q = {"question_type": "object_rel_distance", "ground_truth": "D",
                 "options": "A. chair | B. pillow | C. keyboard | D. door"}
        text = ("...\nDistance from each candidate to the nearest fan (closest points, metres): chair 2.261; door 0.525. "
                "Closest to the fan: door.")
        self.assertEqual(a.relation_answer(rel_q, text), "D")
        self.assertEqual(a.relation_score(rel_q, "D"), 1.0)
        self.assertEqual(a.relation_score(abs_q, 2.0), 1.0)
        self.assertEqual(a.relation_score(abs_q, 4.0), 0.0)  # twice the truth: no threshold met
        self.assertAlmostEqual(a.mra(2.2, 2.0), 0.8)

    def test_go_no_go(self):
        import v5_stage1_analysis as a
        def res(sim, inflated, ratio, found, abs_, rel):
            return {"simulated distance score": sim, "boxes over 1.5x": inflated, "box side / gold, median": ratio,
                    "named objects found": found, "_abs": abs_, "_rel": rel}
        v4 = res(40, 5, 0.8, 78, {1: 0.5, 2: 0.2}, {3: 1.0})
        good = res(50, 8, 1.05, 77, {1: 0.6, 2: 0.2, 9: 0.0}, {3: 1.0, 8: 0.0})  # 9, 8: v4 has no relation, not compared
        results = {"v4": v4, "v5-his": res(45, 30, 1.3, 90, {1: 0.9}, {3: 1.0}), "v5-det": good,
                   "v5-pts": res(30, 0, 1.0, 90, {1: 0.9}, {3: 1.0}), "v5-ours": res(10, 0, 1.0, 90, {}, {})}
        best, checks, go = a.go_no_go(results)
        self.assertEqual(best, "v5-det")  # the highest simulated score
        self.assertTrue(go, checks)
        results["v5-det"] = {**good, "named objects found": 75}  # loses more than 2 points of found objects
        self.assertFalse(a.go_no_go(results)[2])
        results["v5-det"] = {**good, "boxes over 1.5x": 10.5}  # more than 5 points over v4's inflated boxes
        self.assertFalse(a.go_no_go(results)[2])
        results["v5-det"] = {**good, "_rel": {8: 1.0}}  # no question in common with v4: the criterion fails
        self.assertFalse(a.go_no_go(results)[2])


class PostProcess(unittest.TestCase):
    def test_direction_on_the_upright_floor(self):
        """Upright axes: x right, y the way the first camera faces, z up. Standing at the chair (origin) facing the door
        straight ahead (+y), a table at +x is to the right; the height (z) plays no part."""
        rows = [{"concept": "chair", "x": "0", "y": "0", "z": "5"}, {"concept": "door", "x": "0", "y": "2", "z": "0"},
                {"concept": "table", "x": "1", "y": "2", "z": "-3"}]
        q = {"question": "If I am standing by the chair and facing the door, is the table to the left or the right of "
                         "the door?", "question_type": "object_rel_direction_easy"}
        self.assertIn("the table is to the right of the door", v5.direction_relation(q, rows))
        rows[2]["x"] = "-1"
        self.assertIn("the table is to the left of the door", v5.direction_relation(q, rows))

    def test_scaled(self):
        r = v5.scaled([{"x": "1", "y": "2", "z": "3", "size_x": "1", "size_y": "1", "size_z": "2", "size": "2"}], 2.0)[0]
        self.assertEqual([float(r[k]) for k in ("x", "y", "z", "size_x", "size_z", "size")], [2, 4, 6, 2, 4, 16])

    def test_question_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "test.jsonl").write_text(json.dumps({"id": 3, "dataset": "d", "scene_name": "s", "question": "Q?"}) + "\n")
            for name, record in (("q3", {"id": 3, "question": "other text"}), ("a+b", {"question": "Q?"})):
                (tmp / name).mkdir()
                (tmp / name / "question.json").write_text(json.dumps(record))
            ids = v5.question_ids(tmp / "test.jsonl", "d__s", [tmp / "q3", tmp / "a+b"])
        self.assertEqual({k: v["id"] for k, v in ids.items()}, {"q3": 3, "a+b": 3})


class ScorecardBoxes(unittest.TestCase):
    def write_root(self, tmp, metric):
        root = Path(tmp)
        frames = [f"frame_{t:03d}" for t in range(16)]
        fields = ["object_id", "concept", "track_id", "x", "y", "z", "size_x", "size_y", "size_z", "size", "num_frames",
                  *frames]
        # a 16-frame sheet: its frames are the VLM's (the scorecard counts seen frames through adapter_report.json)
        import numpy as np
        vlm = np.linspace(0, 127, 16, dtype=int).tolist()
        (root / "scene").mkdir(parents=True)
        (root / "scene" / "adapter_report.json").write_text(json.dumps({"selected_frames": vlm,
                                                                        "source_video_frames": vlm}))
        def sheet(folder, rows):
            (root / folder).mkdir(parents=True)
            with open(root / folder / "objects.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
        row = lambda i, c, x, sx, n=1: {"object_id": i, "concept": c, "track_id": i, "x": x, "y": 0, "z": 0,
                                        "size_x": sx, "size_y": 1, "size_z": 1, "size": sx, "num_frames": n,
                                        **{f: int(t < n) for t, f in enumerate(frames)}}
        # pipeline: the chair seen in 5 frames is 2 m long (inflated), a second chair row 1 m in 1 frame; a table 1 m
        sheet("s/p", [row(1, "chair", 0, 2.0, 5), row(2, "chair", 0, 1.0, 1), row(3, "table", 4, 1.0, 3)])
        sheet("s/g", [row(1, "chair", 0, 1.0), row(2, "table", 4, 1.0)])
        with open(root / "index.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "scene", "question_type", "question", "prompts", "ground_truth",
                                              "pipeline_sheet", "gold_sheet"])
            w.writeheader()
            w.writerow({"id": 1, "scene": "d/s", "question_type": "object_rel_direction_easy", "question": "?",
                        "prompts": "chair, table", "ground_truth": "A", "pipeline_sheet": "s/p", "gold_sheet": "s/g"})
        if metric:
            (root / "scales_v5.json").write_text("{}")
        return root

    def test_box_measures_in_metres(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = sc.score_root(self.write_root(tmp, metric=True), {"d/s"})[0]
        self.assertEqual(sorted(rec["box_ratio"]), [1.0, 2.0])  # the most-seen chair (2 m), the table (1 m)
        self.assertEqual(sorted(rec["box_inflated"]), [False, True])
        self.assertEqual(sorted(rec["box_volume_ratio"]), [1.0, 2.0])
        # gaps: gold 4 - 0.5 - 0.5 = 3 m; pipeline 4 - 1 - 0.5 = 2.5 m
        self.assertAlmostEqual(rec["gap_abs"][0], 0.5)

    def test_unscaled_sheets_keep_the_priors_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.write_root(tmp, metric=False)
            self.assertFalse(sc.metric(root))
            rec = sc.score_root(root, {"d/s"})[0]
        k = rec["box_ratio"][0] / (1.0 if rec["box_ratio"][0] < 1.5 else 2.0)
        self.assertNotAlmostEqual(k, 1.0)  # the priors scale of these rows, not 1


@unittest.skipUnless(os.environ.get("V5_SMOKE_RUN"), "V5_SMOKE_RUN not set")
class TestSmoke(unittest.TestCase):
    """scene0025_01's smoke run, question 3139 (relative distance, ground truth D: door)."""

    def test_question_3139(self):
        """v5_postprocess.py in its own process, with the branch's src/ first, as it runs."""
        questions = Path(os.environ.get("V5_QUESTIONS", HERE.parents[1] / "data/raw/VSI-Bench/test.jsonl"))
        gold = Path(os.environ.get("V5_GOLD_ROOT", HERE.parents[1] / "tasks/openvocab_adapter/out_full"))
        with tempfile.TemporaryDirectory() as tmp:
            r = subprocess.run([sys.executable, str(HERE / "v5_postprocess.py"), "--run", os.environ["V5_SMOKE_RUN"],
                                "--questions", str(questions), "--gold-root", str(gold), "--out", f"{tmp}/out"],
                               env={**os.environ, "PYTHONPATH": str(HERE.parents[1] / "src")},
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            result = json.loads(r.stdout)
            relations = json.loads((Path(tmp) / "out" / v5.RELATIONS_FILE).read_text())
        self.assertEqual(result["rows"]["3139"]["rows_shown"], 5)
        self.assertIn("Closest to the fan: door.", relations["3139"])
        self.assertAlmostEqual(result["scales"]["scannet/scene0025_01"]["scale"], 1.9116, places=3)
