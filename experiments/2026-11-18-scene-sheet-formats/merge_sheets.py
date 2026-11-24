"""Merge split objects in pipeline sheets after the fact, on a CPU, into a new sheets folder that pipeline_scorecard.py
can score against the old one.

  PYTHONPATH=src python experiments/2026-11-18-scene-sheet-formats/merge_sheets.py \
      --root data/scene_sheets_75 --out <new folder> --scenes <scene list> [--threshold 0.5] [--cap 2] [--uncapped merge]

Tracking itself (associating SAM 3's detections across frames by their 3D centres) and box fitting (fit_box on a
track's 3D points) need the dense VGGT depth, which the sheets zip leaves out. This works on what the sheets keep: each
row's box and the frames it was seen in. In every pipeline sheet of the listed scenes, rows of one class are joined
repeatedly, the closest pair first, if their centres are within --threshold metres or their boxes overlap, and never
two rows seen in the same frame (two detections in one frame are two objects). Two rows whose boxes overlap become the
larger of the two boxes (by volume), seen in both rows' frames; two that don't become the box around both. With --cap,
a pair is joined only if that joined box's longest side stays within cap times the class's typical longest side
(scene_scale.PRIORS); a class with no typical size is then not merged at all, unless --uncapped merge. Without --cap,
every class is merged with the box around both, as explore_sheets.merge_rows does for pipeline_v2. Distances are in
metres with the priors scale (mvp.benchmarks.scene_scale, no gold); the merged sheet is written back in the sheet's own
units, so the old and new folders are scored alike. Unlike pipeline_v2, room fragments are kept, so the folder differs
from the old one by the merge alone. The new folder holds index.csv and the listed scenes; gold sheets are copied.
"""
import argparse
import itertools
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import explore_sheets as es  # noqa: E402
from mvp.benchmarks import scene_scale  # noqa: E402


def overlap(a, b):
    return bool(np.all(a["lo"] <= b["hi"]) and np.all(b["lo"] <= a["hi"]))


def joined(a, b):
    """Two rows' joined box: the larger one if they overlap, else the box around both; seen in both rows' frames."""
    if overlap(a, b):
        big = max(a, b, key=lambda x: float(np.prod(x["hi"] - x["lo"])))
        lo, hi = big["lo"], big["hi"]
    else:
        lo, hi = np.minimum(a["lo"], b["lo"]), np.maximum(a["hi"], b["hi"])
    return dict(concept=a["concept"], track_id=min(a["track_id"], b["track_id"]), lo=lo, hi=hi,
                frames=a["frames"] | b["frames"])


def merge_capped(rows, k, threshold, cap, uncapped="skip"):
    """Rows (objects.csv) of one sheet merged as the module docstring says, in metres (k metres per unit)."""
    out = []
    for concept in dict.fromkeys(r["concept"] for r in rows):
        items = [es.as_box(r, k) for r in rows if r["concept"] == concept]
        limit = cap * scene_scale.PRIORS[concept] if concept in scene_scale.PRIORS else None
        if limit is None and uncapped == "skip":
            out += items
            continue
        while True:
            best = None
            for i, j in itertools.combinations(range(len(items)), 2):
                a, b = items[i], items[j]
                if a["frames"] & b["frames"]:
                    continue
                d = float(np.linalg.norm((a["lo"] + a["hi"]) / 2 - (b["lo"] + b["hi"]) / 2))
                if not (d <= threshold or overlap(a, b)):
                    continue
                box = joined(a, b)
                if limit is not None and float((box["hi"] - box["lo"]).max()) > limit:
                    continue
                if best is None or d < best[0]:
                    best = (d, i, j, box)
            if best is None:
                break
            _, i, j, box = best
            items = [x for n, x in enumerate(items) if n not in (i, j)] + [box]
        out += items
    numbered = []
    for concept in dict.fromkeys(b["concept"] for b in out):
        mine = sorted((b for b in out if b["concept"] == concept),
                      key=lambda x: (min(x["frames"]) if x["frames"] else es.N_FRAMES, x["track_id"]))
        numbered += [{**b, "track_id": n + 1} for n, b in enumerate(mine)]
    return numbered


def merge_folder(root, out, folder, questions, threshold, cap=None, uncapped="skip"):
    """One pipeline sheet folder merged: returns (rows before, rows after)."""
    rows, fields = es.read_csv(root / folder / "objects.csv")
    if not rows:
        return 0, 0
    k = es.priors_scale(root)(folder, questions)  # metres per unit, the same for every question of the folder
    boxes = es.merge_rows(rows, k, threshold) if cap is None else merge_capped(rows, k, threshold, cap, uncapped)
    for b in boxes:  # back to the sheet's units
        b["lo"], b["hi"] = b["lo"] / k, b["hi"] / k
    es.write_csv(out / folder / "objects.csv", es.box_rows(boxes), fields)
    return len(rows), len(boxes)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", required=True, help="a file with one <dataset>/<scene> per line")
    ap.add_argument("--threshold", type=float, default=es.MERGE_THRESHOLD, help="metres between centres")
    ap.add_argument("--cap", type=float, default=None,
                    help="a joined box may be at most this many times the class's typical longest side")
    ap.add_argument("--uncapped", choices=["skip", "merge"], default="skip",
                    help="with --cap, classes with no typical size: not merged (skip) or merged without a cap")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    scenes = {s.strip() for s in Path(args.scenes).read_text().split() if s.strip()}
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy(root / "index.csv", out / "index.csv")
    for s in sorted(scenes):
        name = s.replace("/", "__")
        shutil.copytree(root / name, out / name, ignore=shutil.ignore_patterns("masks", "frames", "tracks"))
    before = after = 0
    for folder, questions in es.folder_questions(root).items():
        if questions[0]["scene"] in scenes:
            b, a = merge_folder(root, out, folder, questions, args.threshold, args.cap, args.uncapped)
            before, after = before + b, after + a
    print(f"{len(scenes)} scenes: {before:,} rows merged into {after:,} (threshold {args.threshold} m, priors scale, "
          f"cap {args.cap}, classes without a typical size: {args.uncapped if args.cap else 'merge'})")


if __name__ == "__main__":
    main()
