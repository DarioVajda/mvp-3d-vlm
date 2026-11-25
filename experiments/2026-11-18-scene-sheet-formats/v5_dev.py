"""v5 on the dev scenes (GPU): one SAM 3 run per scene with pipeline_v4's prompts on all 128 frames, then the
revamped tracker four ways (CONFIGS) on the same masks. The sheets are made afterwards on a CPU (v5_postprocess.py).

v5 is the revamped pipeline (`revamped-pipeline`, CODE_COMMIT: SAM 3 on the 128 VGGT frames, tracks joined by 3D
overlap, the mask depth rule, tracks kept from 4 of 128 frames, sheets in upright axes) with pipeline_v4's prompts and
post-processing. Its code is CODE_COMMIT's, which this branch has had since the merge of 2026-11-25 (6ead380): --code
(default: this repository) must have those files (CODE_FILES, checked before anything runs). For each scene of
--scenes:

  geometry  the branch's question_scene_sheet.ensure_geometry with --frames all: the 128 frames, cameras and dense
            depth converted into <work>/<dataset>__<scene>/all/scene/, and its upright frame (upright.json)
  SAM 3     every SAM 3 prompt of the scene's questions under pipeline_v4's prompt set (rerun_detection, "fixed_v2":
            the questions' prompts with multi-word names kept together, plural names, and SYNONYMS_V2), on all 128
            frames, keeping each detection from rerun_detection.SAVE_MIN_SCORE (0.2) with its score: masks/sam3_raw/
            (rerun_detection.run_sam3, sam3_infer.py's format). A finished run is reused; prompts it lacks are added
            (rerun_detection.merge_sam3)
  pooled    one mask set per object name (rerun_detection.pool_masks: a name's SAM 3 prompts pooled, the lower of two
            detections from different prompts dropped at IoU >= 0.5): masks/fixed_v2/
  tracking  for each of CONFIGS, the branch's tracker (milestone4_video.py) on the pooled masks with its settings
            (association overlap, depth rule mask, the branch's OVERLAP_OPTIONS, re-identification on), the config's
            detection score and min_points, in a copy given --det-score and per-track scores (patch_tracker):
            <out>/<config>/<dataset>__<scene>/all/scene/{tracks/, cameras.npz, adapter_report.json, upright.json}
            and one sheet folder per question, sheets/q<id>/question.json (no objects.csv), then DONE.json

Each (config, scene) with a DONE.json is skipped, so the run continues after a stop. Masks, frames and depth stay in
--work; --out holds only what the sheets are made from (with the dense depth of data/vggt, which every dev scene has).

  PYTHONPATH=src python experiments/2026-11-18-scene-sheet-formats/v5_dev.py \
      --scenes experiments/2026-11-18-scene-sheet-formats/dev_scenes.txt --work <work> --out <out>

The patched tracker is written next to the branch's (PATCHED_TRACKER, ignored by git).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import rerun_detection as rd  # noqa: E402

CODE_COMMIT = "119e6035225f4cad9b82872f9ba900ea6ab0afc3"  # revamped-pipeline, the tree v5_postprocess.py was ported on
CODE_FILES = {  # sha256 of the files of CODE_COMMIT the run uses
    "openvocab-4D-main/trackA/milestone4_video.py": "7531d411bdc1c7d70a85f7a148cdb9cc2f7d4e0bf984fdb87cbb5b778df0ed10",
    "src/mvp/openvocab_adapter/scene_sheet_creator.py": "a11bb8d20df773ba739d7371b58fc5850b737846a8dc80a8a8d1acb1c37304cc",
    "src/mvp/openvocab_adapter/question_scene_sheet.py": "4d3cf5daca8bc53c783dcafc934349a3a12eefe66a9241495ba437cfde04346e",
    "src/mvp/openvocab_adapter/upright.py": "0a9a6dff7149d66c15ef3a812c1c53e01e33cf60e9286a0d7818b040887584de",
    "src/mvp/openvocab_adapter/sam3_infer.py": "5e5e4e80f7bb8a3985e4a579c39a0cf5f2750388d4ebfd00920a7acc774e32a1",
    "src/mvp/openvocab_adapter/adapt_vggt.py": "7342060cce42755e7b75effc9b576a8299ae724b0ff99370e5b1d8689f5b63ba",
}
TRACKER = Path("openvocab-4D-main/trackA/milestone4_video.py")
PATCHED_TRACKER = TRACKER.with_name("milestone4_video_v5.py")
PROMPT_SET = "fixed_v2"
LABEL = "all"
SIZE_TRIM = 0.02  # the branch's default (question_scene_sheet), every pipeline sheet's
# name -> (detection score the tracker keeps, pixels of usable depth a detection needs): the branch's values (0.55; 50
# under the mask rule) and pipeline_v4's (0.4; 300)
CONFIGS = {"v5-his": (0.55, 50), "v5-det": (0.4, 50), "v5-pts": (0.55, 300), "v5-ours": (0.4, 300)}
QUESTIONS = REPO / "data/raw/VSI-Bench/test.jsonl"
PROMPTS = REPO / "experiments/2026-11-19-gold-sheet-relations/prompts_full.jsonl"  # each question's extractor prompts


def check_code(code):
    """Stop unless --code has CODE_COMMIT's files."""
    wrong = [f for f, h in CODE_FILES.items()
             if not (Path(code) / f).is_file() or hashlib.sha256((Path(code) / f).read_bytes()).hexdigest() != h]
    if wrong:
        raise SystemExit(f"{code} is not a tree of {CODE_COMMIT[:7]}: {', '.join(wrong)} differ. Make it with "
                         f"`git archive {CODE_COMMIT} src openvocab-4D-main | tar -x -C {code}`")


def patch_tracker(source):
    """The branch's tracker with the detection score as an option (--det-score, default 0.55, its constant) and each
    kept track's best detection score saved, tracks/<prompt>.scores.json ({track id: score}; a track joined into another
    passes its score on), and the score recorded in tracks/summary.json. Nothing else changes."""
    edits = [
        ('    p.add_argument("--masks-dir", default=None,',
         '    p.add_argument("--det-score", type=float, default=0.55, help="detection score kept by tracking")\n'
         '    p.add_argument("--masks-dir", default=None,'),
        ("saved = SavedMasks(ARGS.masks_dir, prompts, frames, 0.55)",
         "saved = SavedMasks(ARGS.masks_dir, prompts, frames, ARGS.det_score)"),
        ("                    if scores[k] < 0.55:", "                    if scores[k] < ARGS.det_score:"),
        # the centroid rule keeps no score of its own; the overlap rule already does (assign_by_overlap)
        ('                    tr["frame_set"].add(f)\n                    label_vols[pi][f][m] = best\n',
         '                    tr["frame_set"].add(f)\n                    tr["best_score"] = max(tr.get("best_score", 0.0), score)\n'
         '                    label_vols[pi][f][m] = best\n'),
        ('                        dst["frame_set"] |= tr["frame_set"]\n',
         '                        dst["frame_set"] |= tr["frame_set"]\n'
         '                        dst["best_score"] = max(dst.get("best_score", 0.0), tr.get("best_score", 0.0))\n'),
        ('    summary["min_track_frames"] = min_frames\n',
         '    summary["min_track_frames"] = min_frames\n    summary["det_score"] = ARGS.det_score\n'),
        ("""        np.savez_compressed(tracks_dir / f"{prompt.replace(' ', '_')}.npz", labels=vol)\n""",
         """        np.savez_compressed(tracks_dir / f"{prompt.replace(' ', '_')}.npz", labels=vol)\n"""
         """        (tracks_dir / f"{prompt.replace(' ', '_')}.scores.json").write_text(json.dumps(\n"""
         """            {str(t): round(float(tr.get("best_score", 0.0)), 4) for t, tr in stable.items()}))\n"""),
    ]
    for old, new in edits:
        if source.count(old) != 1:
            raise SystemExit(f"{TRACKER} has changed: {old.strip()[:60]!r} found {source.count(old)} times")
        source = source.replace(old, new)
    return source


def question_rows(scenes, questions=QUESTIONS, prompts=PROMPTS):
    """The scenes' questions as rerun_detection's prompt functions read index.csv rows: id, scene, question_type,
    question and the extractor's prompts (prompts_full.jsonl, the gold sheets' prompts)."""
    by_id = {int(r["id"]): r["prompts"] for r in map(json.loads, Path(prompts).read_text().splitlines())}
    rows = []
    for q in map(json.loads, Path(questions).read_text().splitlines()):
        scene = f"{q['dataset']}/{q['scene_name']}"
        if scene in scenes:
            rows.append({"id": int(q["id"]), "scene": scene, "question_type": q["question_type"],
                         "question": q["question"], "prompts": ", ".join(by_id[int(q["id"])])})
    return rows


def track_settings(qss, config):
    """The tracking record of a config, as the branch writes it (question_scene_sheet.track_settings), with the
    detection score and the prompt set."""
    det, min_points = CONFIGS[config]
    s = qss.track_settings(True, "transformers", rd.SAVE_MIN_SCORE, 55.0, min_points, "overlap", "mask")
    return {**s, "det_score": det, "prompt_set": PROMPT_SET, "config": config}


def question_record(row, names, settings, scene_dir):
    """A question's sheet folder's question.json, the fields the branch's create_scene_sheet writes, with its id."""
    return {"id": row["id"], "scene": row["scene"], "question": row["question"], "question_type": row["question_type"],
            "prompts": names, "prompt_mode": PROMPT_SET, "frames": LABEL, "tracking": settings, "size": True,
            "relative_positions": False, "size_trim": SIZE_TRIM, "geometry": str(scene_dir), "depth": "dense",
            "axes": "upright"}


def run_scene(code, qss, sheet, upright, detector, rows, scene, work, out, data_root, dense_dir, log):
    """Geometry, SAM 3 and pooling once, then each config's tracks and sheet folders."""
    folder = scene.replace("/", "__", 1)
    base = Path(work) / folder / LABEL
    scene_dir = base / "scene"
    spec, label = qss.frame_selection(LABEL)
    qss.ensure_geometry(scene, data_root, spec, label, dense_dir, scene_dir)
    upright.scene_upright(scene_dir, log=lambda text: log.write(text + "\n"))
    names = rd.scene_prompts(rows, PROMPT_SET)  # {name: SAM 3 prompts}
    sam = list(dict.fromkeys(p for v in names.values() for p in v))
    frames = sorted(p for p in (scene_dir / "frames").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    raw = base / "masks" / "sam3_raw"
    missing = rd.missing_prompts(raw, sam)
    if missing:
        target = raw if missing == sam else base / "masks" / "sam3_extra"
        rd.run_sam3(*detector, frames, missing, target)
        if target != raw:
            merged = base / "masks" / "sam3_merged"
            rd.merge_sam3(raw, target, merged)
            shutil.rmtree(raw)
            shutil.rmtree(target)
            merged.rename(raw)
    pooled = base / "masks" / PROMPT_SET
    shutil.rmtree(pooled, ignore_errors=True)
    rd.pool_masks(raw, pooled, names)
    for config in CONFIGS:
        dest = Path(out) / config / folder / LABEL
        if (dest / "DONE.json").is_file():
            continue
        shutil.rmtree(dest, ignore_errors=True)
        cfg_scene = dest / "scene"
        cfg_scene.mkdir(parents=True)
        for f in ("cameras.npz", "adapter_report.json", "upright.json"):
            shutil.copy(scene_dir / f, cfg_scene / f)
        os.symlink((scene_dir / "depth.npz").resolve(), cfg_scene / "depth.npz")  # the tracker reads it; not kept
        settings = track_settings(qss, config)
        det, min_points = CONFIGS[config]
        started = time.time()
        sheet.PIPELINE = Path(code) / PATCHED_TRACKER
        sheet.run_pipeline(scene_dir / "frames", cfg_scene, list(names), "track", 1, no_reid=False, masks_dir=pooled,
                           conf_percentile=55.0, min_points=min_points, stdout=log, association="overlap",
                           depth_rule="mask", mask_conf_quantile=settings["mask_conf_quantile"],
                           tracker_options={**{k: settings[k] for k in qss.OVERLAP_OPTIONS}, "det_score": det})
        (cfg_scene / "depth.npz").unlink()
        for name in names:
            (cfg_scene / "tracks" / f"{name.replace(' ', '_')}.settings.json").write_text(json.dumps(
                {"prompt": name, "masks": PROMPT_SET, "sam3_prompts": names[name], "settings": settings}, indent=1))
        for row in rows:
            q_names = [n for n, _ in rd.question_prompts(row, PROMPT_SET)]
            (dest / "sheets" / f"q{row['id']}").mkdir(parents=True)
            (dest / "sheets" / f"q{row['id']}" / "question.json").write_text(
                json.dumps(question_record(row, q_names, settings, cfg_scene), indent=1))
        (dest / "DONE.json").write_text(json.dumps({
            "scene": scene, "config": config, "settings": settings, "names": names, "questions": len(rows),
            "code_commit": CODE_COMMIT, "tracking_seconds": round(time.time() - started, 1),
            "sam3": json.loads((raw / "manifest.json").read_text())["seconds"]}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", default=str(REPO), help=f"a tree with {CODE_COMMIT[:7]}'s files (default: this repository)")
    ap.add_argument("--scenes", default=str(HERE / "dev_scenes.txt"))
    ap.add_argument("--work", required=True, help="geometry, frames and masks (kept for later runs)")
    ap.add_argument("--out", required=True, help="the four configs' tracks and question records")
    ap.add_argument("--data-root", default=str(REPO / "tasks/vggt_data/out"), help="the exported VGGT scenes")
    ap.add_argument("--dense-dir", default=str(REPO / "data/vggt"), help="the dense VGGT depth")
    args = ap.parse_args()
    code = Path(args.code).resolve()
    check_code(code)
    (code / PATCHED_TRACKER).write_text(patch_tracker((code / TRACKER).read_text()))
    sys.path.insert(0, str(code / "src/mvp/openvocab_adapter"))
    import question_scene_sheet as qss
    import scene_sheet_creator as sheet
    import upright
    scenes = [s.strip() for s in Path(args.scenes).read_text().split() if s.strip()]
    rows = question_rows(set(scenes))
    detector = rd.load_detector(code)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    failed = []
    with open(Path(args.out) / "v5_dev.log", "a") as log:
        for i, scene in enumerate(scenes, 1):
            mine = [r for r in rows if r["scene"] == scene]
            started = time.time()
            try:
                run_scene(code, qss, sheet, upright, detector, mine, scene, args.work, args.out, args.data_root,
                          args.dense_dir, log)
                print(f"[{i}/{len(scenes)}] {scene}: {len(mine)} questions, {time.time() - started:.0f} s", flush=True)
            except (Exception, SystemExit) as error:  # noqa: BLE001 - one scene's failure is reported, the rest run
                failed.append(scene)
                print(f"[{i}/{len(scenes)}] {scene}: FAILED: {error}", flush=True)
                log.write(f"FAILED {scene}: {error!r}\n")
    done = {c: sum((Path(args.out) / c / s.replace("/", "__", 1) / LABEL / "DONE.json").is_file() for s in scenes)
            for c in CONFIGS}
    print(f"finished: {done} of {len(scenes)} scenes per config; failed: {failed or 'none'}")
    sys.exit(1 if failed or min(done.values()) < len(scenes) else 0)


if __name__ == "__main__":
    main()
