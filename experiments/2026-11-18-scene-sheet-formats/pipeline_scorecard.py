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
  - rows seen in only one frame, over the rows of every distinct pipeline sheet folder (each folder is its own
    pipeline run, so one object appears once per folder that names its class)
  - unmatched rows: in each sheet folder, a class's rows beyond the count in the matching gold sheet, plus every row of
    a class that gold sheet does not have; a count-based stand-in, since positions can only be matched to gold with an
    alignment
  - size and distance errors with the priors scale (mvp.benchmarks.scene_scale, no gold): the longest side of a class's
    largest row against the gold object's, and the distance between class centres against gold's, for classes with
    one gold object; median absolute error in metres and median relative error
  - fixed rules reading only the sheet, scaled as --metric-scale priors scales it: counting (rows of the class), object
    size (largest row's longest side), absolute distance (smallest gap between the two classes' boxes), relative
    distance (the candidate whose boxes come closest to the target's); scored like VSI-Bench (MRA, exact match)
Room fragments (room, wall, floor, ceiling) are left out of the row measures: they are never gold objects.
"""
import argparse
import csv
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np

from mvp.benchmarks import scene_scale

HERE = Path(__file__).parent
FRAGMENTS = {"room", "wall", "floor", "ceiling"}
DATASETS = {"arkitscenes": "ARKitScenes", "scannet": "ScanNet", "scannetpp": "ScanNet++"}
CONTEXT = {"object_counting", "room_size_estimation"}
MRA_THRESHOLDS = np.arange(0.5, 1.0, 0.05)
RULE_TYPES = {"object_counting": "counting", "object_size_estimation": "object size",
              "object_abs_distance": "absolute distance", "object_rel_distance": "relative distance"}


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


def score_root(root, scenes):
    """Per-scene and per-question tallies for one sheets folder, as a list of records, one per scene."""
    root = Path(root)
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
                    if g:
                        rec["count_match"].append(sum(r["concept"] == c for r in prows) == g)
            rec["empty_where_gold"].append(not prows and bool(grows))
            if q["question_type"] in RULE_TYPES:
                k = scene_scale.question_scale(int(q["id"]), str(root))["k"]
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
            rec["one_frame"] += sum(int(r["num_frames"]) == 1 for r in prows)
            rec["unmatched"] += sum(max(0, n - gcount[c]) for c, n in pcount.items())
        pipe, gold = scene_rows(root, qs, "pipeline_sheet"), scene_rows(root, qs, "gold_sheet")
        objects = {c: rows for c, rows in pipe.items() if c not in FRAGMENTS}
        ratio, classes = scene_scale.prior_ratio(scene_scale.scene_longest(str(root), scene))  # the scene's scale
        k = scene_scale.CALIBRATION * ratio if classes else scene_scale.FALLBACK
        single = [c for c, v in gold.items() if len(v) == 1 and objects.get(c)]
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
    ("named objects found", lambda R: np.mean([x for r in R for x in r["found"]]), "pct", True),
    ("questions with every named object", lambda R: np.mean([x for r in R for x in r["all_found"]]), "pct", True),
    ("empty sheets where gold has objects", lambda R: int(sum(sum(r["empty_where_gold"]) for r in R)), "int", False),
    ("count = gold, per (question, class)", lambda R: np.mean([x for r in R for x in r["count_match"]]), "pct", True),
    ("rows over the sheet folders (room fragments left out)", lambda R: int(sum(r["rows"] for r in R)), "int", None),
    ("rows seen in only one frame", lambda R: sum(r["one_frame"] for r in R) / max(1, sum(r["rows"] for r in R)),
     "pct", False),
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
    return {"pct": f"{100 * v:.1f}%", "int": f"{v:,}", "m": f"{v:.2f}", "score": f"{v:.1f}"}[kind]


def diff(a, b, kind, higher):
    if any(isinstance(x, float) and np.isnan(x) for x in (a, b)):
        return "–"
    d = b - a
    text = {"pct": f"{100 * d:+.1f} pts", "int": f"{d:+,}", "m": f"{d:+.2f}", "score": f"{d:+.1f}"}[kind]
    if d == 0 or higher is None:
        return text
    return text + (" ✓" if (d > 0) == higher else " ✗")


def table(old, new, title):
    lines = [f"### {title}", "", "| Measure | Old | New | New − old |", "|---|---:|---:|---:|"]
    for name, f, kind, higher in MEASURES:
        a, b = f(old), f(new)
        lines.append(f"| {name} | {fmt(a, kind)} | {fmt(b, kind)} | {diff(a, b, kind, higher)} |")
    return lines


def report(old_root, new_root, scenes_file):
    scenes = [s.strip() for s in Path(scenes_file).read_text().split() if s.strip()]
    old, new = score_root(old_root, set(scenes)), score_root(new_root, set(scenes))
    nq = sum(len(r["empty_where_gold"]) for r in new)
    lines = ["# Pipeline scorecard", "", "<!-- Written by pipeline_scorecard.py in this folder; edit that, not this file. -->",
             "", f"Old: `{old_root}`. New: `{new_root}`. Scenes: `{Path(scenes_file).name}`, {len(scenes)} scenes, "
             f"{nq:,} questions. Each folder against its own gold sheets. ✓ / ✗: the new sheets are better / worse on "
             "that measure. Definitions are in `pipeline_scorecard.py`.", "",
             "The size and distance errors and the fixed rules use the priors scale (`--metric-scale priors`, "
             "`mvp.benchmarks.scene_scale`; no gold). Its calibration constant was fitted on the reliable scenes "
             "outside explore200's scenes, which are among the dev scenes, so on dev scenes it is slightly "
             "optimistic.", ""]
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
    args = ap.parse_args()
    text = report(args.old, args.new, args.scenes)
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
