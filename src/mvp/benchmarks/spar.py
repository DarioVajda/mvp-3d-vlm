"""SPAR-Bench-Tiny (jasonzhango/SPAR-Bench-Tiny: 1,000 ScanNet questions, 50 per task) data, prompts and metrics.

Prompts and scoring mirror the official lmms-eval task (LogosIntelligenceLab/SPAR, lmms-eval/lmms_eval/tasks/sparbench/
utils.py), quirks included:
  * fill tasks: "\n{question}\nPlease answer the question using a single word or phrase.", scored by MRA on the first
    number of the answer (the last one for the multi-view tasks);
  * select tasks: "\n{question}\n{bbox note}\nAnswer with the option's letter ...", scored by its loose exact match;
  * view_change_infer: "\n{question}\n\n", the five signed motion components each scored by MRA with prediction and
    ground truth swapped, as in the official code;
  * overall = mean over the 20 tasks.
One deviation: an empty answer scores 0 (the official exact match counts "" as a substring of the answer).
"""
import io
import re

import numpy as np
from PIL import Image

REPO = "jasonzhango/SPAR-Bench-Tiny"
MAX_NEW_TOKENS = 100  # the official generation_kwargs

MCA = ["obj_spatial_relation_oo", "obj_spatial_relation_oc_mv", "obj_spatial_relation_oo_mv", "spatial_imagination_oc",
       "spatial_imagination_oo", "spatial_imagination_oc_mv", "spatial_imagination_oo_mv", "position_matching",
       "camera_motion_infer", "distance_infer_center_oo", "distance_infer_center_oo_mv"]
NA = ["depth_prediction_oc", "depth_prediction_oo", "distance_prediction_oc", "distance_prediction_oo",
      "depth_prediction_oc_mv", "depth_prediction_oo_mv", "distance_prediction_oo_mv", "distance_prediction_oc_mv"]
LEVELS = {"low": NA, "middle": ["view_change_infer", "position_matching", "camera_motion_infer"],
          "high": [t for t in MCA if t not in ("position_matching", "camera_motion_infer")]}
BBOX_NOTE = ("The values represent the bounding box coordinates normalized to a 0-1000 scale, with the top-left corner "
             "as the origin of the image.")


def load():
    import pandas as pd
    from huggingface_hub import snapshot_download
    root = snapshot_download(REPO, repo_type="dataset", local_files_only=True)
    df = pd.read_parquet(f"{root}/data/test-00000-of-00001.parquet")
    return [{"id": r["id"], "task": r["task"], "image_bytes": [x["bytes"] for x in r["image"]], "prompt": build_prompt(r),
             "question": r["question"], "answer": r["answer"]} for r in df.to_dict("records")]


def images(doc):
    return [Image.open(io.BytesIO(b)).convert("RGB") for b in doc["image_bytes"]]


def build_prompt(doc):
    if doc["task"] in NA:
        return "\n" + doc["question"] + "\n" + "Please answer the question using a single word or phrase."
    if doc["task"] in MCA:
        note = BBOX_NOTE if doc["task"] in ("position_matching", "camera_motion_infer") else ""
        return "\n" + doc["question"] + "\n" + note + "\n" + "Answer with the option's letter from the given choices directly."
    assert doc["task"] == "view_change_infer", doc["task"]
    return "\n" + doc["question"] + "\n" + "\n"


def mean_relative_accuracy(pred, target, start=.5, end=.95, interval=.05):
    err = abs(pred - target) if target == 0.0 else abs((pred - target) / target)
    return float((err <= 1 - np.linspace(start, end, int((end - start) / interval + 2))).mean())


def exact_match(pred, target):
    pred, target = pred.lower(), target.lower()
    return float(bool(pred) and (pred == target or pred in target or pred[0] == target))


def first_number(pred, task):
    nums = [float(n) for n in re.findall(r"(?<!\^)\d+\.\d+|(?<!\^)\d+", pred)]
    if task.endswith("_mv"):
        nums = [nums[-1] if nums else -1]
    return nums[0]


def vci(pred, answer):
    parse = lambda s: {k: float(v) for k, v in [item.split(":") for item in s.split(",")]}
    p, g = parse(pred), parse(answer)
    pairs = [("move_right", "move_left"), ("move_up", "move_down"), ("move_forward", "move_backward"),
             ("rotate_right", "rotate_left"), ("rotate_up", "rotate_down")]
    return float(np.mean([mean_relative_accuracy(g.get(a, 0) - g.get(b, 0), p.get(a, 0) - p.get(b, 0)) for a, b in pairs]))


def score(doc, pred):
    try:
        if doc["task"] in MCA:
            return exact_match(pred, doc["answer"])
        if doc["task"] in NA:
            return mean_relative_accuracy(first_number(pred, doc["task"]), float(doc["answer"]))
        return vci(pred, doc["answer"])
    except (ValueError, IndexError, TypeError):
        return 0.0


def summarize(df):
    per_task = (100 * df.groupby("task")["score"].mean()).to_dict()
    out = {"overall": float(np.mean(list(per_task.values())))}
    out.update({lvl: float(np.mean([per_task[t] for t in ts if t in per_task])) for lvl, ts in LEVELS.items()})
    return {**out, **per_task}
