"""Pipeline scorecard: two pipeline sheet folders (old and new) against gold, side by side, on a list of scenes.
Rerun it after every pipeline change; tune on dev_scenes.txt (the scenes with no question in explore200_ids.txt).

  PYTHONPATH=src python experiments/2026-11-18-scene-sheet-formats/pipeline_scorecard.py \
      --old data/scene_sheets_75 --new <new sheets folder> [--scenes dev_scenes.txt] [--out pipeline_scorecard.md]

Each folder is a run_sheets.sh output (index.csv, the pipeline and gold sheet folders) and is scored against its own
gold sheets. Measures, overall and by dataset:
  - named objects found: (question, named class) pairs with at least one pipeline row; named classes are the
    question's prompts without `room` (none for room-size questions), as sheet_quality.py counts them
  - questions with every named object found
  - empty sheets where gold has objects: questions whose pipeline sheet has no row while the gold sheet has one
  - count match: (question, named class) pairs whose row count equals the gold count, among pairs gold has
  - rows seen in only one of the VLM's 16 frames, over the rows of every distinct pipeline sheet folder (each folder
    is its own pipeline run, so one object appears once per folder that names its class). The frames are counted as
    the prompt counts them (mvp.benchmarks.scene_sheet, visibility "auto"), so a sheet made on all 128 frames and one
    made on the 16 are measured alike: on a 16-frame sheet it is the rows with num_frames 1
  - rows per unique object: for the (question, named class) pairs whose gold sheet has exactly one object and whose
    pipeline sheet has at least one row, the rows the pipeline sheet has (1 is right), and the share with two or more
  - unmatched rows: in each sheet folder, a class's rows beyond the count in the matching gold sheet, plus every row of
    a class that gold sheet does not have; a count-based stand-in, since positions can only be matched to gold with an
    alignment
  - box size (key measures: inflated boxes shorten the closest-point distances the relations state), for classes with
    one gold object, from the row seen in the most frames (the one the question-conditioned dedup keeps): its longest
    side over the gold object's (median), the share of such boxes more than BOX_INFLATED times gold's longest side,
    its volume over gold's (median), and the closest-point gap between two such classes' boxes against gold's (median
    absolute error in metres); in metres as the size errors below
  - size and distance errors with the priors scale (mvp.benchmarks.scene_scale, no gold), or with --scale oracle each
    scene's scale fitted to its gold sheet (explore_sheets.oracle_scales; scenes without one are left out of these
    measures): the longest side of a class's largest row against the gold object's, and the distance between class
    centres against gold's, for classes with one gold object; median absolute error in metres and median relative
    error. The priors scale's calibration was fitted on 16-frame sheets, whose objects are split into small rows, so
    sheets whose rows are whole objects (128 frames, overlap tracking) need --scale oracle, or the calibration refitted.
    Sheets already in metres (a folder with scales_v4.json or scales_v5.json: pipeline_v4, v5) are used as they are
    under the priors scale, which on their rows would rescale them a second time
  - fixed rules reading only the sheet, scaled the same way: counting (rows of the class), object size (largest row's
    longest side), absolute distance (smallest gap between the two classes' boxes), relative distance (the candidate
    whose boxes come closest to the target's); scored like VSI-Bench (MRA, exact match)
Room fragments (room, wall, floor, ceiling) are left out of the row measures: they are never gold objects.
"""
import argparse
import csv
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np

from mvp.benchmarks import scene_scale
from mvp.benchmarks import scene_sheet as ss

HERE = Path(__file__).parent
FRAGMENTS = {"room", "wall", "floor", "ceiling"}
DATASETS = {"arkitscenes": "ARKitScenes", "scannet": "ScanNet", "scannetpp": "ScanNet++"}
CONTEXT = {"object_counting", "room_size_estimation"}
MRA_THRESHOLDS = np.arange(0.5, 1.0, 0.05)
RULE_TYPES = {"object_counting": "counting", "object_size_estimation": "object size",
              "object_abs_distance": "absolute distance", "object_rel_distance": "relative distance"}


METRIC_FILES = ("scales_v4.json", "scales_v5.json")  # a sheets folder already in metres (relations_v4.py, v5_postprocess.py)
BOX_INFLATED = 1.5  # a box this many times gold's longest side counts as inflated


def metric(root):
    """Whether a sheets folder is already in metres."""
    return any((Path(root) / f).is_file() for f in METRIC_FILES)


def most_seen(rows):
    """The row seen in the most frames (the first of those tied)."""
    return max(rows, key=lambda r: int(r["num_frames"]) if r.get("num_frames") else 0)


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def box(r):
    return (np.array([float(r[k]) for k in "xyz"]), np.array([float(r[f"size_{k}"]) for k in "xyz"]))


def longest(r):
    return max(float(r["size_x"]), float(r["size_y"]), float(r["size_z"]))


def gap(a, b):
    return float(np.linalg.norm(np.maximum(0, np.abs(a[0] - b[0]) - (a[1] + b[1]) / 2)))


def mra(pred, gt):
    return float(np.mean([abs(pred - gt) / gt < 1 - t for t in MRA_THRESHOLDS]))


def named(q):
    if q["question_type"] == "room_size_estimation":
        return []
    return [p.strip() for p in q["prompts"].split(",") if p.strip() != "room"]


def rule_score(q, rows, k):
    """A fixed rule's score on the question, reading only its pipeline sheet scaled by k (0 without the objects)."""
    names = named(q)
    of = lambda c: [box(r) for r in rows if r["concept"] == c]
    t, gt = q["question_type"], q["ground_truth"]
    if t == "object_counting":
        n = len(of(names[0]))
        return mra(n, float(gt)) if n else 0.0
    if t == "object_size_estimation":
        b = of(names[0])
        return mra(max(float(e.max()) for _, e in b) * k * 100, float(gt)) if b else 0.0
    if t == "object_abs_distance":
        d = [gap(a, b) for a in of(names[0]) for b in of(names[1])] if len(names) > 1 else []
        return mra(min(d) * k, float(gt)) if d and min(d) > 0 else 0.0
    *candidates, target = names
    g = [min([gap(a, b) for a in of(c) for b in of(target)], default=np.inf) for c in candidates]
    return float(np.isfinite(min(g)) and "ABCD"[int(np.argmin(g))] == gt)


def scene_rows(root, questions, key):
    """The scene's distinct rows over its sheet folders (key: pipeline_sheet or gold_sheet), by class."""
    out, seen = defaultdict(list), set()
    for folder in sorted({q[key] for q in questions}):
        for r in read(Path(root) / folder / "objects.csv"):
            k = (r["concept"], *(round(float(r[a]), 4) for a in "xyz"))
            if k not in seen:
                seen.add(k)
                out[r["concept"]].append(r)
    return out


def vlm_frame_count(row, folder):
    """The VLM frames a pipeline row counts as seen in, as the prompt counts them (visibility "auto")."""
    frames = ss.folder_frames(folder)
    return len(ss.vlm_frames_seen(ss.seen_in(row, len(frames.selected)), frames, "auto"))


def oracle_scales(root, scenes):
    """{scene: metres per sheet unit fitted to its gold sheet}, for the scenes that have one (explore_sheets)."""
    import explore_sheets
    return {s: v["scale"] for s, v in explore_sheets.oracle_scales(Path(root), sorted(scenes)).items()}


def score_root(root, scenes, scale="priors"):
    """Per-scene and per-question tallies for one sheets folder, as a list of records, one per scene. scale: priors
    (scene_scale) or oracle (oracle_scales; a scene without one gets no size, distance or rule measures)."""
    root = Path(root)
    oracle = oracle_scales(root, scenes) if scale == "oracle" else None
    index = read(root / "index.csv")
    by_scene = defaultdict(list)
    for q in index:
        if q["scene"] in scenes:
            by_scene[q["scene"]].append(q)
    missing = set(scenes) - set(by_scene)
    assert not missing, f"{root}: scenes not in index.csv: {sorted(missing)[:5]}"
    records = []
    for scene, qs in sorted(by_scene.items()):
        rec = defaultdict(list)
        rec["scene"], rec["dataset"] = scene, scene.split("/")[0]
        for q in qs:
            prows, grows = read(root / q["pipeline_sheet"] / "objects.csv"), read(root / q["gold_sheet"] / "objects.csv")
            names = named(q)
            if names:
                found = [any(r["concept"] == c for r in prows) for c in names]
                rec["found"] += found
                rec["all_found"].append(all(found))
                for c in names:
                    g = sum(r["concept"] == c for r in grows)
                    n = sum(r["concept"] == c for r in prows)
                    if g:
                        rec["count_match"].append(n == g)
                    if g == 1 and n:
                        rec["unique_rows"].append(n)
            rec["empty_where_gold"].append(not prows and bool(grows))
            if q["question_type"] in RULE_TYPES:
                k = scene_scale.question_scale(int(q["id"]), str(root))["k"] if oracle is None else oracle.get(scene)
                if oracle is None and metric(root):
                    k = 1.0
                if k is not None:
                    rec[f"rule:{q['question_type']}"].append(rule_score(q, prows, k))
        rec["rows"] = rec["one_frame"] = rec["unmatched"] = 0
        for pfolder, gfolder in sorted({(q["pipeline_sheet"], q["gold_sheet"]) for q in qs}):  # what each sheet lists
            prows = [r for r in read(root / pfolder / "objects.csv") if r["concept"] not in FRAGMENTS]
            gcount = defaultdict(int)
            for r in read(root / gfolder / "objects.csv"):
                gcount[r["concept"]] += 1
            pcount = defaultdict(int)
            for r in prows:
                pcount[r["concept"]] += 1
            rec["rows"] += len(prows)
            rec["one_frame"] += sum(vlm_frame_count(r, root / pfolder) == 1 for r in prows)
            rec["unmatched"] += sum(max(0, n - gcount[c]) for c, n in pcount.items())
        pipe, gold = scene_rows(root, qs, "pipeline_sheet"), scene_rows(root, qs, "gold_sheet")
        objects = {c: rows for c, rows in pipe.items() if c not in FRAGMENTS}
        ratio, classes = scene_scale.prior_ratio(scene_scale.scene_longest(str(root), scene))  # the scene's scale
        k = scene_scale.CALIBRATION * ratio if classes else scene_scale.FALLBACK
        if oracle is not None:
            k = oracle.get(scene)
        elif metric(root):
            k = 1.0
        single = [c for c, v in gold.items() if len(v) == 1 and objects.get(c)] if k is not None else []
        shown = {c: most_seen(objects[c]) for c in single}
        for c in single:
            g = gold[c][0]
            side = k * longest(shown[c]) / longest(g)
            rec["box_ratio"].append(side)
            rec["box_inflated"].append(side > BOX_INFLATED)
            gv = float(np.prod(box(g)[1]))
            if gv > 0:
                rec["box_volume_ratio"].append(k ** 3 * float(np.prod(box(shown[c])[1])) / gv)
        for a, b in itertools.combinations(single, 2):
            pb = [(box(shown[c])[0] * k, box(shown[c])[1] * k) for c in (a, b)]
            rec["gap_abs"].append(abs(gap(*pb) - gap(box(gold[a][0]), box(gold[b][0]))))
        for c in single:
            p, g = k * max(longest(r) for r in objects[c]), longest(gold[c][0])
            rec["size_abs"].append(abs(p - g))
            rec["size_rel"].append(abs(p / g - 1))
        centre = {c: np.median([box(r)[0] for r in objects[c]], axis=0) for c in single}
        for a, b in itertools.combinations(single, 2):
            dg = float(np.linalg.norm(box(gold[a][0])[0] - box(gold[b][0])[0]))
            if dg > 0.1:
                dp = k * float(np.linalg.norm(centre[a] - centre[b]))
                rec["dist_abs"].append(abs(dp - dg))
                rec["dist_rel"].append(abs(dp / dg - 1))
        records.append(rec)
    return records


MEASURES = [  # (name, how to compute over a group's records, format, better when higher)
    ("**box longest side / gold, median**", lambda R: np.median([x for r in R for x in r["box_ratio"]]), "x", None),
    (f"**boxes over {BOX_INFLATED}x gold's longest side**",
     lambda R: np.mean([x for r in R for x in r["box_inflated"]]), "pct", False),
    ("box volume / gold, median", lambda R: np.median([x for r in R for x in r["box_volume_ratio"]]), "x", None),
    ("**closest-point gap error, median (m)**", lambda R: np.median([x for r in R for x in r["gap_abs"]]), "m", False),
    ("named objects found", lambda R: np.mean([x for r in R for x in r["found"]]), "pct", True),
    ("questions with every named object", lambda R: np.mean([x for r in R for x in r["all_found"]]), "pct", True),
    ("empty sheets where gold has objects", lambda R: int(sum(sum(r["empty_where_gold"]) for r in R)), "int", False),
    ("count = gold, per (question, class)", lambda R: np.mean([x for r in R for x in r["count_match"]]), "pct", True),
    ("rows over the sheet folders (room fragments left out)", lambda R: int(sum(r["rows"] for r in R)), "int", None),
    ("rows seen in only one of the VLM's 16 frames",
     lambda R: sum(r["one_frame"] for r in R) / max(1, sum(r["rows"] for r in R)), "pct", False),
    ("rows per unique object (gold has one)", lambda R: np.mean([x for r in R for x in r["unique_rows"]]), "float",
     False),
    ("unique objects with two or more rows", lambda R: np.mean([x > 1 for r in R for x in r["unique_rows"]]), "pct",
     False),
    ("unmatched rows (beyond the gold sheet's count)", lambda R: sum(r["unmatched"] for r in R) / max(1, sum(r["rows"] for r in R)),
     "pct", False),
    ("size error, median (m)", lambda R: np.median([x for r in R for x in r["size_abs"]]), "m", False),
    ("size error, median relative", lambda R: np.median([x for r in R for x in r["size_rel"]]), "pct", False),
    ("distance error, median (m)", lambda R: np.median([x for r in R for x in r["dist_abs"]]), "m", False),
    ("distance error, median relative", lambda R: np.median([x for r in R for x in r["dist_rel"]]), "pct", False),
] + [(f"rule: {name}", (lambda t: lambda R: 100 * np.mean([x for r in R for x in r[f"rule:{t}"]]))(t), "score", True)
     for t, name in RULE_TYPES.items()]


def fmt(v, kind):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return {"pct": f"{100 * v:.1f}%", "int": f"{v:,}", "m": f"{v:.2f}", "score": f"{v:.1f}", "float": f"{v:.2f}",
            "x": f"{v:.2f}x"}[kind]


def diff(a, b, kind, higher):
    if any(isinstance(x, float) and np.isnan(x) for x in (a, b)):
        return "–"
    d = b - a
    text = {"pct": f"{100 * d:+.1f} pts", "int": f"{d:+,}", "m": f"{d:+.2f}", "score": f"{d:+.1f}",
            "float": f"{d:+.2f}", "x": f"{d:+.2f}x"}[kind]
    if d == 0 or higher is None:
        return text
    return text + (" ✓" if (d > 0) == higher else " ✗")


def table(old, new, title):
    lines = [f"### {title}", "", "| Measure | Old | New | New − old |", "|---|---:|---:|---:|"]
    for name, f, kind, higher in MEASURES:
        a, b = f(old), f(new)
        lines.append(f"| {name} | {fmt(a, kind)} | {fmt(b, kind)} | {diff(a, b, kind, higher)} |")
    return lines


def report(old_root, new_root, scenes_file, scale="priors"):
    scenes = [s.strip() for s in Path(scenes_file).read_text().split() if s.strip()]
    old, new = score_root(old_root, set(scenes), scale), score_root(new_root, set(scenes), scale)
    nq = sum(len(r["empty_where_gold"]) for r in new)
    lines = ["# Pipeline scorecard", "", "<!-- Written by pipeline_scorecard.py in this folder; edit that, not this file. -->",
             "", f"Old: `{old_root}`. New: `{new_root}`. Scenes: `{Path(scenes_file).name}`, {len(scenes)} scenes, "
             f"{nq:,} questions. Each folder against its own gold sheets. ✓ / ✗: the new sheets are better / worse on "
             "that measure. Definitions are in `pipeline_scorecard.py`.", "",
             ("The size and distance errors and the fixed rules use the priors scale (`--metric-scale priors`, "
              "`mvp.benchmarks.scene_scale`; no gold). Its calibration constant was fitted on the reliable scenes "
              "outside explore200's scenes, which are among the dev scenes, so on dev scenes it is slightly "
              "optimistic." if scale == "priors" else
              "The size and distance errors and the fixed rules use each scene's oracle scale, fitted to its gold "
              "sheet (`explore_sheets.oracle_scales`, `--scale oracle`); scenes without one are left out of them, "
              "and each folder's scenes are fitted on its own rows."),
             "Bold: the key box-size measures (boxes larger than the objects shorten the closest-point distances the "
             "relations state), from each class's most-seen row. A folder already in metres (pipeline_v4, v5: "
             "`scales_v4.json`, `scales_v5.json`) is used as it is under the priors scale.", ""]
    lines += table(old, new, f"All {len(scenes)} scenes")
    for ds, name in DATASETS.items():
        o, n = [r for r in old if r["dataset"] == ds], [r for r in new if r["dataset"] == ds]
        if n:
            lines += [""] + table(o, n, f"{name} ({len(n)} scenes)")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True, help="the sheets folder before the change")
    ap.add_argument("--new", required=True, help="the sheets folder after it")
    ap.add_argument("--scenes", default=str(HERE / "dev_scenes.txt"))
    ap.add_argument("--out", default=str(HERE / "pipeline_scorecard.md"))
    ap.add_argument("--scale", choices=["priors", "oracle"], default="priors",
                    help="metres per sheet unit for the size, distance and rule measures: priors (no gold) or oracle "
                         "(fitted to the gold sheet; for sheets the priors' calibration was not fitted on)")
    args = ap.parse_args()
    text = report(args.old, args.new, args.scenes, args.scale)
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
