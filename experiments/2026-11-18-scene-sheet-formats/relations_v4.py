"""pipeline_v4's sheets: pipeline_v3_dedup's sheets in metres with the median scale, and the relations of each question
in relations_v4.json (explore_sheets.relation_text, mode "v4", appends them).

The scale of a scene (metres per sheet unit) is the median of three estimates, none fitted to gold:
  priors        scene_scale's scale from typical object sizes of the unmerged rows (CALIBRATION x prior_ratio, or
                FALLBACK), as pipeline_v3 used it
  room height   ROOM_HEIGHT / the scene's floor-to-ceiling extent: the depth of the 16 frames (pixels with confidence at
                least the frames' 55th percentile, every 4th pixel), unprojected with the cameras, measured along
                gravity (the mean of the cameras' down axes), from the 1st to the 99th percentile
  door height   DOOR_HEIGHT / the extent along gravity of the scene's largest door row, if the scene has one
(with no door row, the median of two is their mean). Every pipeline sheet's positions and sizes are multiplied by it
(explore_sheets.write_sheets, "oracle"), for every question of the scene, object-size questions included.

The relations, the same sentences as pipeline_v3_dedup_relations (mode "v3"), only when the shown sheet has a row
for every object the question names:
  relative direction            the direction the shown sheet's rows imply (explore_sheets.direction_relation)
  absolute / relative distance  the closest-point distances (explore_sheets.relations, "closest") between the shown
                                sheet's rows with their boxes from --trimmed (trimmed_sheets.py): the same tracks,
                                frames and centres, boxes from the 10th-90th percentile of each object's points
                                instead of their full extent, scaled by the same scale
The sheet shown keeps the full-extent boxes. scales_v4.json records each scene's three estimates and the median.

  PYTHONPATH=src python experiments/2026-11-18-scene-sheet-formats/relations_v4.py --sheets <pipeline_v3_dedup sheets> \
      --trimmed <trimmed_sheets.py's sheets> --runs <run folder> [...] --scenes <scene list> --out <new folder>
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import explore_sheets as es  # noqa: E402

ROOM_HEIGHT = 2.5  # metres, floor to ceiling
DOOR_HEIGHT = 2.0  # metres
CONF_PERCENTILE = 55
STEP = 4
RELATIONS_FILE = "relations_v4.json"
SCALES_FILE = "scales_v4.json"


def scene_folder(runs, scene):
    """subset-uniform16/scene/ of the scene in the first run folder that has its cameras."""
    for run in runs:
        d = Path(run) / scene.replace("/", "__", 1) / "subset-uniform16" / "scene"
        if (d / "cameras.npz").is_file():
            return d
    raise FileNotFoundError(f"{scene}: no cameras.npz in {runs}")


def gravity_and_height(runs, dense_root, scene):
    """The unit down direction (world frame) and the floor-to-ceiling extent along it, in sheet units."""
    d = scene_folder(runs, scene)
    cams = np.load(d / "cameras.npz")
    sel = json.loads((d / "adapter_report.json").read_text())["selected_frames"]
    ext, K = cams["extrinsic"].astype(np.float64), cams["intrinsic"].astype(np.float64)
    depth = np.load(Path(dense_root) / scene / "depth.npy", mmap_mode="r")
    conf = np.load(Path(dense_root) / scene / "depth_conf.npy", mmap_mode="r")
    d16, c16 = np.asarray(depth[sel], np.float32), np.asarray(conf[sel], np.float32)
    threshold = np.percentile(c16, CONF_PERCENTILE)
    down = np.mean([e[1, :3] for e in ext], axis=0)
    g = down / np.linalg.norm(down)
    H, W = d16.shape[1:]
    v, u = np.mgrid[0:H:STEP, 0:W:STEP]
    heights = []
    for f in range(len(sel)):
        z = d16[f, ::STEP, ::STEP]
        ok = (c16[f, ::STEP, ::STEP] >= threshold) & (z > 0)
        cam = np.stack([(u - K[f, 0, 2]) * z / K[f, 0, 0], (v - K[f, 1, 2]) * z / K[f, 1, 1], z], -1)[ok]
        R, t = ext[f, :, :3], ext[f, :, 3]
        heights.append(((cam - t) @ R) @ g)
    h = np.concatenate(heights)
    return g, float(np.percentile(h, 99) - np.percentile(h, 1))


def door_extent(root, index, scene, g):
    """The largest door row's extent along gravity over the scene's pipeline sheets, or None."""
    best = 0.0
    for folder in sorted({r["pipeline_sheet"] for r in index.values() if r["scene"] == scene}):
        for r in es.read_csv(Path(root) / folder / "objects.csv")[0]:
            if r["concept"] == "door":
                best = max(best, float(np.abs(g) @ np.array([float(r[k]) for k in es.EXT])))
    return best or None


def scene_scales(root, runs, dense_root, scenes):
    from mvp.benchmarks import scene_scale
    index = {int(r["id"]): r for r in csv.DictReader(open(Path(root) / "index.csv"))}
    out = {}
    for s in scenes:
        ratio, classes = scene_scale.prior_ratio(scene_scale.scene_longest(str(root), s))
        priors = scene_scale.CALIBRATION * ratio if classes else scene_scale.FALLBACK
        g, height = gravity_and_height(runs, dense_root, s)
        room = ROOM_HEIGHT / height if height > 0 else None
        door_units = door_extent(root, index, s, g)
        door = DOOR_HEIGHT / door_units if door_units else None
        out[s] = {"scale": float(np.median([k for k in (priors, room, door) if k])), "priors": priors,
                  "priors_fallback": not classes, "room_height": room, "door_height": door}
    return out


def trimmed_rows(shown, trimmed, folder, k):
    """The rows of `shown` with their boxes from the trimmed sheets (trimmed_sheets.py, before dedup), scaled by k: each
    row is the same track (concept, track id), with the same frames and centre, in the sheet folder the dedup started
    from (`folder` without its "__q<id>" suffix)."""
    original = Path(folder).parent / Path(folder).name.split("__q")[0]
    by_track = {(r["concept"], r["track_id"]): r for r in es.read_csv(Path(trimmed) / original / "objects.csv")[0]}
    out = []
    for r in shown:
        t = dict(by_track[(r["concept"], r["track_id"])])
        same = all(t[c] == r[c] for c in r if c.startswith("frame_") or c == "num_frames")
        assert same and all(abs(float(t[c]) * k - float(r[c])) < 1e-9 * max(1.0, abs(float(r[c]))) for c in es.POS), \
            f"{folder}: {r['concept']} {r['track_id']} is not the same track in the trimmed sheets"
        for c in es.POS + es.EXT:
            t[c] = repr(float(t[c]) * k)
        t["size"] = repr(float(t["size"]) * k ** 3)
        out.append(t)
    return out


def build(sheets, trimmed, runs, dense_root, scenes, out):
    sheets, trimmed, out = Path(sheets), Path(trimmed), Path(out)
    scales = scene_scales(sheets, runs, dense_root, scenes)
    present = {p.name.replace("__", "/", 1) for p in sheets.iterdir() if p.is_dir() and "__" in p.name}
    assert present == set(scenes), f"{sheets} has scenes {sorted(present ^ set(scenes))[:5]} not in the list (or the reverse)"
    es.write_sheets(sheets, out, scales, "oracle")
    tmp = out.parent / (out.name + ".trimmed")
    if tmp.exists():
        shutil.rmtree(tmp)
    index = {int(r["id"]): r for r in csv.DictReader(open(out / "index.csv"))}
    relations = {}
    for q, row in index.items():
        if row["scene"] not in scales:
            continue
        if row["question_type"] in es.DIRECTION_TYPES:
            text = es.direction_relation(row, out, "pipeline")
        elif row["question_type"] in es.DISTANCE_TYPES:
            shown, fields = es.read_csv(out / row["pipeline_sheet"] / "objects.csv")
            if not set(es.named_objects(row)) <= {r["concept"] for r in shown}:
                continue
            (tmp / row["pipeline_sheet"]).mkdir(parents=True, exist_ok=True)
            es.write_csv(tmp / row["pipeline_sheet"] / "objects.csv",
                         trimmed_rows(shown, trimmed, row["pipeline_sheet"], scales[row["scene"]]["scale"]), fields)
            text = es.relations(row, tmp, "pipeline", "closest")
        else:
            continue
        if text:
            relations[str(q)] = text
    shutil.rmtree(tmp, ignore_errors=True)
    (out / RELATIONS_FILE).write_text(json.dumps(relations, indent=1, sort_keys=True))
    (out / SCALES_FILE).write_text(json.dumps(scales, indent=1, sort_keys=True))
    return scales, relations


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheets", required=True, help="pipeline_v3_dedup's sheets (dedup_sheets.py), with scale_rows.json")
    ap.add_argument("--trimmed", required=True, help="trimmed_sheets.py's sheets of the same run (not deduplicated)")
    ap.add_argument("--runs", nargs="+", required=True, help="run folders with each scene's cameras.npz, adapter_report.json")
    ap.add_argument("--dense-root", default=str(HERE.parents[1] / "data" / "vggt"))
    ap.add_argument("--scenes", required=True, help="a file with one <dataset>/<scene> per line")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    scenes = [s for s in Path(args.scenes).read_text().split() if s]
    scales, relations = build(args.sheets, args.trimmed, args.runs, args.dense_root, scenes, args.out)
    print(f"{len(scales)} scenes scaled (median scale {np.median([v['scale'] for v in scales.values()]):.3f}, "
          f"{sum(v['door_height'] is not None for v in scales.values())} with a door row); "
          f"{len(relations)} questions with relations -> {args.out}")


if __name__ == "__main__":
    main()
