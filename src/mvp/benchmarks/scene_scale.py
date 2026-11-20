"""Metres per sheet unit for a pipeline scene sheet, without gold: typical object sizes times one calibration constant.

The pipeline sheets are in VGGT's arbitrary per-scene units. For a scene, k (metres per unit) is CALIBRATION times the
median, over the classes in PRIORS that the scene's pipeline sheets have, of the class's typical longest side divided by
the longest side of its largest row (over every pipeline sheet of the scene in index.csv). Split objects make rows
smaller than the objects, which makes this ratio too large on average; CALIBRATION corrects that. For an object-size
question, the classes the question names are left out, so that its answer does not come from the prior; when no prior
class is left, the scene gets FALLBACK.

Fitted and compared with other conversions in experiments/2026-11-18-scene-sheet-formats/metric_scale.md (conversion
d'): over the 34 scenes with a reliable oracle fit, in 5-fold cross-validation by scene, the scale is within 30% of the
true one in 82% of scenes (median error 16%).

  question_scale(question_id, root) -> {"k", "classes", "left_out", "fallback"}
"""
import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

# Typical longest side in metres of objects whose size varies little, from common real-world dimensions (not from the
# gold sheets). An assumption, stated in metric_scale.md.
PRIORS = {"door": 2.0, "bed": 2.0, "sofa": 2.0, "refrigerator": 1.75, "chair": 0.85, "toilet": 0.7, "tv": 1.1,
          "monitor": 0.55, "pillow": 0.55, "nightstand": 0.55, "washer": 0.85, "keyboard": 0.45, "microwave": 0.5,
          "laptop": 0.35}
# Fitted with `metric_scale.py --fit` on the 24 scenes that have a reliable oracle fit (explore_sheets.align against
# the gold sheet: residual under 0.5 m, cameras 0.8-2.0 m above the gold floor) and are not among the scenes of
# explore200_ids.txt. CALIBRATION is the median over them of true scale / prior scale (the median ratio above, no class
# left out), FALLBACK the median true scale.
CALIBRATION = 0.8417
FALLBACK = 2.293
SIZE_QUESTION = "object_size_estimation"


@lru_cache(maxsize=None)
def scene_longest(root, scene):
    """{class: longest side of its largest row} over every pipeline sheet of the scene in index.csv, sheet units."""
    with open(Path(root) / "index.csv", newline="") as f:
        folders = sorted({r["pipeline_sheet"] for r in csv.DictReader(f) if r["scene"] == scene})
    out = {}
    for folder in folders:
        with open(Path(root) / folder / "objects.csv", newline="") as f:
            for r in csv.DictReader(f):
                side = max(float(r["size_x"]), float(r["size_y"]), float(r["size_z"]))
                out[r["concept"]] = max(out.get(r["concept"], 0.0), side)
    return out


def prior_ratio(longest, left_out=()):
    """The median of typical / measured longest side over the prior classes present and not left out, and those
    classes; (nan, []) when there are none."""
    classes = sorted(c for c in longest if c in PRIORS and c not in left_out and longest[c] > 0)
    if not classes:
        return float("nan"), []
    return float(np.median([PRIORS[c] / longest[c] for c in classes])), classes


@lru_cache(maxsize=None)
def question_scale(question_id, root):
    """The scale for one question's pipeline sheet: CALIBRATION x prior_ratio over the scene, without the classes an
    object-size question names; FALLBACK when no prior class is left."""
    with open(Path(root) / "index.csv", newline="") as f:
        row = next(r for r in csv.DictReader(f) if int(r["id"]) == int(question_id))
    left_out = ({p.strip() for p in row["prompts"].split(",")} - {"room"}
                if row["question_type"] == SIZE_QUESTION else set())
    ratio, classes = prior_ratio(scene_longest(str(root), row["scene"]), left_out)
    fallback = not classes
    return {"k": FALLBACK if fallback else CALIBRATION * ratio, "classes": classes,
            "left_out": sorted(left_out & set(PRIORS)), "fallback": fallback}
