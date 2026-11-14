"""VSTI-Bench (Journey9ni/vstibench: 5,736 questions on 88 ScanNet val videos, the erratum-corrected test.json) data,
prompts and metrics.

Prompts and scoring mirror the official lmms-eval task (VITA-Group/VLM-3R, thinking-in-space/lmms_eval/tasks/
vstibench/utils.py), which reuses VSI-Bench's prompts:
  * MCA: "These are frames of a video.\n{question}\nOptions:\n{options}\nAnswer with the option's letter ...", scored by
    exact match of the first word of the answer against mc_answer;
  * NA (camera_obj_abs_dist, camera_displacement): "...\n{question}\nPlease answer the question using a single word
    or phrase.", scored by MRA on float(first word), 0 when that is not a number;
  * overall = mean over the question types; 32 uniform frames (the questions name "frame i of 32").
scene0217_00 is excluded: 49 questions of our SFT mix (SPAR, RC) use that scene.
"""
import json

import numpy as np

from mvp.benchmarks.vsibench import MCA_POST_PROMPT, NA_POST_PROMPT, PRE_PROMPT, fuzzy_matching
from mvp.paths import BENCH_MEDIA

REPO = "Journey9ni/vstibench"
MAX_NEW_TOKENS = 16
NFRAMES = 32
NA = ["camera_obj_abs_dist", "camera_displacement"]
EXCLUDED_SCENES = {"scene0217_00"}


def load():
    from huggingface_hub import snapshot_download
    root = snapshot_download(REPO, repo_type="dataset", local_files_only=True)
    docs = []
    for d in json.load(open(f"{root}/test.json")):
        if d["video_path"].rsplit("/", 1)[-1].removesuffix(".mp4") in EXCLUDED_SCENES:
            continue
        docs.append({"id": d["id"], "task": d["question_type"], "video": str(BENCH_MEDIA / "vstibench" / d["video_path"]),
                     "prompt": build_prompt(d), "question": d["question"],
                     "answer": d["ground_truth"] if d["question_type"] in NA else d["mc_answer"]})
    return docs


def build_prompt(doc):
    if doc["question_type"] in NA:
        return PRE_PROMPT + "\n" + doc["question"] + "\n" + NA_POST_PROMPT
    return "\n".join([PRE_PROMPT, doc["question"], "Options:\n" + "\n".join(doc["options"]), MCA_POST_PROMPT])


def score(doc, pred):
    pred = fuzzy_matching(pred)
    if doc["task"] not in NA:
        return float(pred.lower() == doc["answer"].lower())
    try:
        pred, target = float(pred), float(doc["answer"])
    except ValueError:
        return 0.0
    return float((abs(pred - target) / target <= 1 - np.linspace(.5, .95, 10)).mean())


def summarize(df):
    per_task = (100 * df.groupby("task")["score"].mean()).to_dict()
    return {"overall": float(np.mean(list(per_task.values()))), **per_task}
