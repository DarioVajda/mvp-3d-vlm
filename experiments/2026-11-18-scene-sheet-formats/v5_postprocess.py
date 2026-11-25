"""v5's sheets: the revamped tracker's output (revamped-pipeline: SAM 3 on all 128 frames, overlap association, the
mask depth rule, tracks kept from 4 of 128 frames, sheets in upright axes) with pipeline_v4's post-processing, ported
from the 16-frame, first-camera sheets it was written for. Runs on a CPU from a finished run folder in that branch's
layout (<dataset>__<scene>/all/{scene, sheets/<prompts>/question.json}), with that branch's src/ on PYTHONPATH (this
branch's, since the merge of 2026-11-25):

  1. unmerged rows: scene_sheet_creator.build_scene_sheet on the run's tracks with each question.json's settings (the
     run's own objects.csv, which is checked; --min-points changes which detections count)
  2. the point merge (merge_points.merge_tracks: same-class tracks never seen in the same frame, boxes overlapping or
     centres within 0.5 m, metres by the priors scale of step 1's rows), around the branch's collect_objects. No
     one-frame score filter: the branch's tracker writes no track scores, and its tracks are seen in at least 4 frames
  3. trimmed boxes: step 2 again with size_trim 10 (the 10th-90th percentile of each object's points), for relations
  4. question-conditioned dedup (dedup_sheets.best_row: most frames, then nearness to the other named objects for
     distance questions, then the larger box; no track-score tie-break, as in step 2)
  5. the scene's scale, the median of: priors (scene_scale on step 1's rows), room height (ROOM_HEIGHT over the
     floor-to-ceiling extent of the VLM's 16 frames' confident depth, along the upright z) and door height
     (DOOR_HEIGHT over the largest merged door row's size_z); v4 measured the last two along the cameras' mean down
     axis, which the upright frame replaces
  6. relations (relations_v4.py's sentences): for distance questions the closest-point distances between the shown
     rows' trimmed boxes in metres, for relative direction the direction the rows imply on the floor (x, y here; v4's
     direction_word reads x, z of the first camera's axes), only when the sheet has a row for every named object

The output is a sheets root in the same layout: index.csv for the questions (VSI-Bench's ids, gold sheets from
--gold-root), each question's sheet in metres (positions and sizes times the scale), relations_v5.json, scales_v5.json.
render() writes a question's prompt with the branch's renderer (upright axes, its 128-frame "nearby" visibility), the
preamble in metres as v4's, and the relations after the sheet.

  PYTHONPATH=src python experiments/2026-11-18-scene-sheet-formats/v5_postprocess.py \
      --run <run root> --questions data/raw/VSI-Bench/test.jsonl --gold-root tasks/openvocab_adapter/out_full \
      --out <new root> [--min-points N] [--render QUESTION_ID --vggt-root data/vggt]
"""
import argparse
import contextlib
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dedup_sheets  # noqa: E402
import explore_sheets as es  # noqa: E402
import merge_points  # noqa: E402

LABEL = "all"  # the branch's frames folder for the 128 frames
MERGE_THRESHOLD = 0.5  # metres, pipeline_v3's (rerun_detection.FROZEN)
TRIM = 10.0  # percentile at each end for the relations' boxes (trimmed_sheets.SIZE_TRIM)
ROOM_HEIGHT, DOOR_HEIGHT, CONF_PERCENTILE, STEP = 2.5, 2.0, 55, 4  # relations_v4.py's
RELATIONS_FILE, SCALES_FILE = "relations_v5.json", "scales_v5.json"


def creator():
    """The branch's scene_sheet_creator (with its upright module), from PYTHONPATH."""
    import mvp.openvocab_adapter as pkg
    sys.path.insert(0, str(list(pkg.__path__)[0]))  # a namespace package: no __init__.py
    import scene_sheet_creator
    if "axes" not in scene_sheet_creator.build_scene_sheet.__code__.co_varnames:
        raise SystemExit("PYTHONPATH's scene_sheet_creator has no upright axes: use a src/ with revamped-pipeline's code")
    return scene_sheet_creator


def sheet_rows(sheet, scene_dir, settings, out_dir, size_trim, merge_k=None, min_points=None):
    """objects.csv of one sheet: build_scene_sheet with the question's settings, the point merge if merge_k is given
    (metres per unit, for its threshold)."""
    plain = sheet.collect_objects
    if merge_k is not None:
        os.environ["MERGE_POINTS_THRESHOLD"] = str(MERGE_THRESHOLD)
        merge_points.scene_scale_k = lambda _scene_dir: merge_k
        sheet.collect_objects = lambda scene, prompts, *a, **kw: merge_points.merge_tracks(
            plain(scene, prompts, *a, **kw), scene, prompts)
    try:
        tracking = settings["tracking"]
        sheet.build_scene_sheet(
            scene_dir, settings["prompts"], relative_positions=settings.get("relative_positions", False),
            include_size=True, out_dir=out_dir, conf_percentile=tracking["conf_percentile"],
            min_points=min_points if min_points is not None else tracking.get("min_points"), size_trim=size_trim,
            cameras=False, depth_rule=tracking.get("depth_rule", "mask"),
            mask_conf_quantile=tracking.get("mask_conf_quantile", sheet.MASK_CONF_QUANTILE),
            axes=settings.get("axes", "upright"))
    finally:
        sheet.collect_objects = plain
        os.environ.pop("MERGE_POINTS_THRESHOLD", None)
    return es.read_csv(Path(out_dir) / "objects.csv")


def priors_scale(unmerged):
    """scene_scale's priors scale (metres per unit) from the scene's unmerged rows, as pipeline_v3 / v4 take it."""
    from mvp.benchmarks import scene_scale
    longest = {}
    for rows in unmerged:
        for r in rows:
            longest[r["concept"]] = max(longest.get(r["concept"], 0.0),
                                        max(float(r["size_x"]), float(r["size_y"]), float(r["size_z"])))
    ratio, classes = scene_scale.prior_ratio(longest)
    return (scene_scale.CALIBRATION * ratio if classes else scene_scale.FALLBACK), longest


def room_height_units(sheet, scene_dir):
    """The floor-to-ceiling extent, in units, of the VLM's 16 frames' confident depth along the upright z."""
    import upright
    from mvp.benchmarks.scene_sheet import VLM_FRAMES
    cams = np.load(Path(scene_dir) / "cameras.npz")
    geo = np.load(Path(scene_dir) / "depth.npz")
    sel = list(VLM_FRAMES)
    depth, conf = geo["depth"][sel].astype(np.float32), geo["conf"][sel].astype(np.float32)
    threshold = np.percentile(conf, CONF_PERCENTILE)
    frame = upright.scene_upright(Path(scene_dir))
    ext, K = cams["extrinsic"][sel].astype(np.float64), cams["intrinsic"][sel].astype(np.float64)
    H, W = depth.shape[1:]
    v, u = np.mgrid[0:H:STEP, 0:W:STEP]
    heights = []
    for f in range(len(sel)):
        z = depth[f, ::STEP, ::STEP]
        ok = (conf[f, ::STEP, ::STEP] >= threshold) & (z > 0)
        cam = np.stack([(u - K[f, 0, 2]) * z / K[f, 0, 0], (v - K[f, 1, 2]) * z / K[f, 1, 1], z], -1)[ok]
        world = (cam - ext[f, :, 3]) @ ext[f, :, :3]
        heights.append(upright.to_upright(world, frame)[:, 2])
    h = np.concatenate(heights)
    return float(np.percentile(h, 99) - np.percentile(h, 1))


def direction_relation(row, rows):
    """explore_sheets.direction_relation on upright rows: the floor is (x, y), so each position is handed to its
    direction_word, which reads the first camera's floor (x, z), as (x, -, y)."""
    shown = [dict(r, z=r["y"], y="0") for r in rows]
    folder = Path("/nonexistent")
    es_read = es.read_csv
    es.read_csv = lambda path: (shown, list(shown[0]) if shown else [])
    try:
        return es.direction_relation(dict(row, pipeline_sheet=str(folder), gold_sheet=str(folder)), folder, "pipeline")
    finally:
        es.read_csv = es_read


def scaled(rows, k):
    out = []
    for r in rows:
        r = dict(r)
        for c in es.POS + es.EXT:
            r[c] = repr(float(r[c]) * k)
        r["size"] = repr(float(r["size"]) * k ** 3)
        out.append(r)
    return out


def question_ids(questions, run_scene, sheets):
    """{sheet folder name: VSI-Bench row}: by question.json's id (v5_dev.py writes one folder per question), else by
    its question text in the scene (the branch's own runs, one folder per prompt set)."""
    scene = run_scene.replace("__", "/", 1)
    all_q = list(map(json.loads, Path(questions).read_text().splitlines()))
    by_id = {int(q["id"]): q for q in all_q}
    by_text = {(f"{q['dataset']}/{q['scene_name']}", q["question"]): q for q in all_q}
    out = {}
    for s in sheets:
        record = json.loads((s / "question.json").read_text())
        out[s.name] = by_id[int(record["id"])] if "id" in record else by_text[(scene, record["question"])]
    return out


def build_key(settings):
    """What a sheet's rows depend on: its prompts and the settings they are computed with."""
    t = settings["tracking"]
    return json.dumps([settings["prompts"], t["conf_percentile"], t.get("min_points"), t.get("depth_rule"),
                       t.get("mask_conf_quantile"), settings["size_trim"], settings.get("axes", "upright"),
                       settings.get("relative_positions", False)])


def ensure_depth(scene_dir, dense_root):
    """scene/depth.npz from the dense VGGT depth when the run left it out (v5_dev.py does), checked to be the same
    reconstruction and the tracks' shape (the branch's rebuild_scene_sheets.ensure_depth and check_tracks)."""
    import rebuild_scene_sheets
    _, shape = rebuild_scene_sheets.ensure_depth(Path(scene_dir), dense_root)
    rebuild_scene_sheets.check_tracks(Path(scene_dir), shape)


def build(run, questions, gold_root, out, min_points=None, dense_root=None):
    sheet = creator()
    run, out, gold_root = Path(run), Path(out), Path(gold_root)
    if out.exists():
        shutil.rmtree(out)
    gold_index = {int(r["id"]): r for r in csv.DictReader(open(gold_root / "index.csv"))}
    work = out.parent / (out.name + ".work")
    shutil.rmtree(work, ignore_errors=True)
    index, relations, scales, report = [], {}, {}, {}
    for scene_root in sorted(p for p in run.iterdir() if (p / LABEL / "scene" / "adapter_report.json").is_file()):
        scene_dir = scene_root / LABEL / "scene"
        if not (scene_dir / "depth.npz").is_file():
            if dense_root is None:
                raise SystemExit(f"{scene_dir} has no depth.npz: pass --dense-root (data/vggt)")
            ensure_depth(scene_dir, dense_root)
        sheets = sorted((scene_root / LABEL / "sheets").iterdir())
        qs = question_ids(questions, scene_root.name, sheets)
        settings = {s.name: json.loads((s / "question.json").read_text()) for s in sheets}
        # every distinct prompt set and its settings is built once; its questions share the rows
        keys = {s.name: build_key(settings[s.name]) for s in sheets}
        first = {}
        for s in sheets:
            first.setdefault(keys[s.name], s)
        # 1. unmerged rows, checked against the run's own sheets where it has them
        built = {}
        for key, s in first.items():
            rows, _ = sheet_rows(sheet, scene_dir, settings[s.name], work / "unmerged" / s.name,
                                 settings[s.name]["size_trim"], min_points=min_points)
            built[key] = rows
        unmerged = {s.name: built[keys[s.name]] for s in sheets}
        for s in sheets:
            if min_points is None and (s / "objects.csv").is_file() and (
                    work / "unmerged" / first[keys[s.name]].name / "objects.csv").read_bytes() != (s / "objects.csv").read_bytes():
                raise SystemExit(f"{s}: rebuilding the unmerged rows does not give the run's objects.csv")
        k_priors, longest = priors_scale(built.values())
        # 2-3. the point merge, with full and trimmed boxes
        built_merged, built_trimmed = {}, {}
        for key, s in first.items():
            built_merged[key] = sheet_rows(sheet, scene_dir, settings[s.name], work / "merged" / s.name,
                                           settings[s.name]["size_trim"], k_priors, min_points)
            built_trimmed[key] = sheet_rows(sheet, scene_dir, settings[s.name], work / "trimmed" / s.name, TRIM,
                                            k_priors, min_points)[0]
        merged = {s.name: built_merged[keys[s.name]][0] for s in sheets}
        fields = {s.name: built_merged[keys[s.name]][1] for s in sheets}
        trimmed = {s.name: built_trimmed[keys[s.name]] for s in sheets}
        # 5. the scale
        height = room_height_units(sheet, scene_dir)
        doors = [float(r["size_z"]) for rows in built_merged.values() for r in rows[0] if r["concept"] == "door"]
        room = ROOM_HEIGHT / height if height > 0 else None
        door = DOOR_HEIGHT / max(doors) if doors else None
        k = float(np.median([x for x in (k_priors, room, door) if x]))
        scene = scene_root.name.replace("__", "/", 1)
        scales[scene] = {"scale": k, "priors": k_priors, "room_height": room, "door_height": door,
                         "room_height_units": height, "longest_unmerged": longest}
        # copy the scene folder the renderer reads (frames, the cameras' upright frame), not the depth
        dest = out / scene_root.name / LABEL
        shutil.copytree(scene_dir, dest / "scene", ignore=shutil.ignore_patterns("depth.npz", "frames", "tracks"))
        for s in sheets:
            q = qs[s.name]
            qtype = q["question_type"]
            rows, trim = merged[s.name], {(r["concept"], r["track_id"]): r for r in trimmed[s.name]}
            names = [p for p in settings[s.name]["prompts"] if p != "room"]
            # 4. the dedup, for the question types it is for
            if qtype in dedup_sheets.DEDUP_TYPES:
                cands = {n: dedup_sheets.candidates([r for r in rows if r["concept"] == n]) for n in names
                         if any(r["concept"] == n for r in rows)}
                keep = []
                for name in dict.fromkeys(r["concept"] for r in rows):
                    mine = [r for r in rows if r["concept"] == name]
                    if name in names and len(mine) > 1:
                        others = [c for n, c in cands.items() if n != name] if qtype in dedup_sheets.NEAR_TYPES else ()
                        keep.append(dedup_sheets.best_row(mine, {}, others))
                    else:
                        keep += mine
                rows = sorted(keep, key=rows.index)
            folder = dest / "sheets" / s.name
            folder.mkdir(parents=True, exist_ok=True)
            es.write_csv(folder / "objects.csv", scaled(rows, k), fields[s.name])
            (folder / "question.json").write_text(json.dumps(
                {**settings[s.name], "v5": {"merge_threshold_m": MERGE_THRESHOLD, "scale": k, "dedup": qtype in
                                            dedup_sheets.DEDUP_TYPES, "min_points_sheet": min_points}}, indent=1))
            # 6. the relations
            row = {"id": q["id"], "prompts": ", ".join(settings[s.name]["prompts"]), "question_type": qtype,
                   "question": q["question"]}
            text = ""
            if qtype in es.DIRECTION_TYPES:
                text = direction_relation(row, scaled(rows, k))
            elif qtype in es.DISTANCE_TYPES and set(names) <= {r["concept"] for r in rows}:
                tmp = work / "relations" / s.name
                tmp.mkdir(parents=True, exist_ok=True)
                es.write_csv(tmp / "objects.csv", scaled([trim[(r["concept"], r["track_id"])] for r in rows], k),
                             fields[s.name])
                text = es.relations(dict(row, pipeline_sheet=s.name), work / "relations", "pipeline", "closest")
            if text:
                relations[str(q["id"])] = text
            g = gold_index[int(q["id"])]
            shutil.copytree(gold_root / g["gold_sheet"], out / g["gold_sheet"], dirs_exist_ok=True)
            index.append({**g, "pipeline_sheet": f"{scene_root.name}/{LABEL}/sheets/{s.name}", "pipeline_ok": "True"})
            report[q["id"]] = {"rows_unmerged": len(unmerged[s.name]), "rows_merged": len(merged[s.name]),
                               "rows_shown": len(rows)}
    with open(out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0]))
        w.writeheader()
        w.writerows(index)
    (out / RELATIONS_FILE).write_text(json.dumps(relations, indent=1, sort_keys=True))
    (out / SCALES_FILE).write_text(json.dumps(scales, indent=1, sort_keys=True))
    shutil.rmtree(work)
    return scales, relations, report


def render(question_id, root, vggt_root):
    """The text after the video for a v5 question: the branch's renderer on the scaled sheet (upright axes, nearby
    visibility), its preamble in metres as pipeline_v4's, then the question's relations."""
    from mvp.benchmarks import scene_sheet as ss
    from mvp.benchmarks.vsibench import build_prompt_parts
    root = Path(root)
    text = ss.render_scene(question_id, "pipeline", "csv", False, root, Path(vggt_root))
    raw = ss.preamble("pipeline", visibility="nearby", axes="upright")
    metres = ss.preamble("pipeline", visibility="nearby", axes="upright", metric_scale="priors")
    if raw not in text:
        raise SystemExit("the rendered sheet does not start with the expected preamble")
    text = text.replace(raw, metres)
    rel = json.loads((root / RELATIONS_FILE).read_text()).get(str(question_id), "")
    row = ss.load_index(root)[int(question_id)]
    doc = {"question_type": row["question_type"], "question": row["question"],
           "options": row["options"].split(" | ") if row["options"] else None}
    return build_prompt_parts(doc, "sit", text + ("\n" + rel if rel else ""))[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the run root: <dataset>__<scene>/all/ folders")
    ap.add_argument("--questions", required=True, help="VSI-Bench's test.jsonl")
    ap.add_argument("--gold-root", required=True, help="the gold sheets (build_gold_sheets.py's out_full)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-points", type=int, default=None,
                    help="pixels a detection needs for an object to count in a frame when the rows are computed "
                         "(default: each question.json's, 50 under the mask rule; pipeline_v4 used 300). The tracking "
                         "itself is not redone")
    ap.add_argument("--dense-root", help="the dense VGGT depth (data/vggt), for runs without scene/depth.npz")
    ap.add_argument("--render", type=int, nargs="*", default=[], help="print these questions' prompts")
    ap.add_argument("--vggt-root", help="meta.json (video_fps) per scene, for --render")
    args = ap.parse_args()
    with contextlib.redirect_stdout(sys.stderr):  # the sheet builder's log; stdout carries the summary
        scales, relations, report = build(args.run, args.questions, args.gold_root, args.out, args.min_points,
                                          args.dense_root)
    print(json.dumps({"scales": {s: {k: v for k, v in d.items() if k != "longest_unmerged"} for s, d in scales.items()},
                      "rows": report, "relations": len(relations)}, indent=1))
    for q in args.render:
        print(f"\n===== question {q} =====\n" + render(q, args.out, args.vggt_root))


if __name__ == "__main__":
    main()
