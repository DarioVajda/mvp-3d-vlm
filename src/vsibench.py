"""VSI-Bench data loading, prompts and metrics.

Prompts and scoring mirror lmms-eval's `vsibench` task (lmms_eval/tasks/vsibench/utils.py), which is also
the prompt format given in the Qwen3-VL technical report (Appendix B.6):
  * MCA: "These are frames of a video.\n{question}\nOptions:\n{options}\nAnswer with the option's letter ..."
  * NA:  "These are frames of a video.\n{question}\nPlease answer the question using a single word or phrase."
  * MCA scored by exact match on the first token of the answer, NA by Mean Relative Accuracy (MRA).
  * Overall = macro average over the 8 task types, with rel_direction easy/medium/hard averaged into one.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VSI_DIR = ROOT / "data" / "raw" / "VSI-Bench"

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]
# The 8 reported task types (column order of the VSI-Bench paper, Table 1)
TASKS = [
    "object_counting", "object_abs_distance", "object_size_estimation", "room_size_estimation",
    "object_rel_distance", "object_rel_direction", "route_planning", "obj_appearance_order",
]

PRE_PROMPT = "These are frames of a video."
MCA_POST_PROMPT = "Answer with the option's letter from the given choices directly."
NA_POST_PROMPT = "Please answer the question using a single word or phrase."


def load_vsibench(split="full"):
    """split: 'full' (5130 qs), 'debiased' (2362) or 'pruned' (2768)."""
    if split == "full":
        df = pd.read_json(VSI_DIR / "test.jsonl", lines=True)
    else:
        df = pd.read_parquet(VSI_DIR / f"test_{split}.parquet")
    df["video_path"] = [str(VSI_DIR / d / f"{s}.mp4") for d, s in zip(df["dataset"], df["scene_name"])]
    return df


def task_of(question_type):
    return "object_rel_direction" if question_type.startswith("object_rel_direction") else question_type


def build_prompt(doc):
    if doc["question_type"] in NA_QUESTION_TYPES:
        return PRE_PROMPT + "\n" + doc["question"] + "\n" + NA_POST_PROMPT
    options = "Options:\n" + "\n".join(doc["options"])
    return "\n".join([PRE_PROMPT, doc["question"], options, MCA_POST_PROMPT])


# ---- scoring (verbatim logic from lmms-eval) ----
_NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
_NUMBER_WORDS.update({"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
                      "ninety": 90, "hundred": 100, "thousand": 1000})
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def fuzzy_matching(pred):
    return pred.split(" ")[0].rstrip(".").strip()


def exact_match(pred, target):
    return 1.0 if pred.lower() == target.lower() else 0.0


def mean_relative_accuracy(pred, target, start=.5, end=.95, interval=.05):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs(pred - target) / target <= 1 - conf_intervs
    return accuracy.mean()


def extract_number(text):
    """First number (digits or English number words) in the response, else None."""
    if text is None:
        return None
    s = str(text).strip().lower()
    if not s:
        return None
    digit_match = _NUM_RE.search(s)
    word_run_start, word_run_tokens = None, []
    for m in re.finditer(r"[a-z]+", s):
        tok = m.group(0)
        if word_run_start is None:
            if tok in _NUMBER_WORDS:
                word_run_start = m.start()
                word_run_tokens.append(tok)
        elif tok == "and":
            continue
        elif tok in _NUMBER_WORDS:
            word_run_tokens.append(tok)
        else:
            break
    if digit_match is not None and (word_run_start is None or digit_match.start() < word_run_start):
        return float(digit_match.group(0))
    if word_run_start is None:
        return None
    total, current = 0, 0
    for tok in word_run_tokens:
        val = _NUMBER_WORDS[tok]
        if val == 100:
            current = max(current, 1) * 100
        elif val == 1000:
            total += max(current, 1) * 1000
            current = 0
        else:
            current += val
    return float(total + current)


def score(doc, prediction):
    if doc["question_type"] in MCA_QUESTION_TYPES:
        return exact_match(fuzzy_matching(prediction), doc["ground_truth"])
    pred = extract_number(prediction)
    if pred is None:
        return 0.0
    return float(mean_relative_accuracy(pred, float(doc["ground_truth"])))


def summarize(df):
    """df has columns question_type, score. Returns {task: score*100, 'overall': macro avg} like lmms-eval."""
    per_qtype = df.groupby("question_type")["score"].mean()
    out = {}
    for t in TASKS:
        if t == "object_rel_direction":
            vals = [per_qtype[q] for q in per_qtype.index if q.startswith("object_rel_direction")]
        else:
            vals = [per_qtype[t]] if t in per_qtype.index else []
        if vals:
            out[t] = 100 * float(np.mean(vals))
    out["overall"] = float(np.mean(list(out.values())))
    return out


def load_predictions(path):
    with open(path) as f:
        return pd.DataFrame([json.loads(l) for l in f])
