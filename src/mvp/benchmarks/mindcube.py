"""MindCube-Tiny (MindCube_tinybench.jsonl: 1,050 questions, settings among 600 / around 250 / rotation 200) data,
prompts and metrics.

The questions embed their options ("... A. Door B. Two chairs"). Prompt: the images, then "{question}\nAnswer with
the option's letter from the given choices directly." (MindCube's own raw-QA prompt asks for free text that its
judge parses; we ask for the letter so that the base model and the trained arms answer in one format). The letter is
extracted with MMSI-Bench's regex extractor; overall accuracy = mean over all questions, as in the MindCube paper.
No image or image set of the benchmark is in MindCube's training split (checked).
"""
import json

from PIL import Image

from mvp.benchmarks.mmsibench import extract_choice
from mvp.paths import TRAIN_MEDIA

ROOT = TRAIN_MEDIA / "mindcube" / "data"
MAX_NEW_TOKENS = 16
POST = "Answer with the option's letter from the given choices directly."


def load():
    return [{"id": d["id"], "task": d["id"].split("_")[0], "type": d["type"], "image_paths": d["images"],
             "prompt": d["question"].strip() + "\n" + POST, "question": d["question"], "answer": d["gt_answer"]}
            for d in map(json.loads, open(ROOT / "raw" / "MindCube_tinybench.jsonl"))]


def images(doc):
    return [Image.open(ROOT / p).convert("RGB") for p in doc["image_paths"]]


def score(doc, pred):
    return float(extract_choice(pred) == doc["answer"])


def summarize(df):
    out = {"overall": 100 * float(df["score"].mean())}
    out.update({k: 100 * float(v) for k, v in df.groupby("task")["score"].mean().items()})
    return out
