"""Video-MME (lmms-eval/Video-MME: 2,700 questions on 900 YouTube videos, 300 each short / medium / long) data,
prompts and metrics, without subtitles.

The prompt is lmms-eval's videomme default: "Select the best answer to the following multiple-choice question based on
the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.\n{question}\n{options}
\nAnswer with the option's letter from the given choices directly.", scored by exact letter match. The letter is
extracted as Video-MME's own eval script does (strip "The best answer is" etc., then the first A-D), which is simpler
than lmms-eval's current extract_mcq_answer; for answers that start with the letter, as ours do, they agree.
32 uniform frames.
"""
import re

import numpy as np

from mvp.paths import BENCH_MEDIA

REPO = "lmms-eval/Video-MME"
MAX_NEW_TOKENS = 16
NFRAMES = 32
PRE = ("Select the best answer to the following multiple-choice question based on the video and the subtitles. "
       "Respond with only the letter (A, B, C, or D) of the correct option.")
POST = "Answer with the option's letter from the given choices directly."
PREFIXES = ["The best answer is", "The correct answer is", "The answer is", "The answer", "The best option is",
            "The correct option is", "Best answer:", "Best option:", "Answer:", "Option:"]


def load():
    import pandas as pd
    from huggingface_hub import snapshot_download
    root = snapshot_download(REPO, repo_type="dataset", local_files_only=True, allow_patterns=["videomme/*.parquet"])
    df = pd.read_parquet(f"{root}/videomme/test-00000-of-00001.parquet")
    return [{"id": r["question_id"], "task": r["duration"], "task_type": r["task_type"], "domain": r["domain"],
             "video": str(BENCH_MEDIA / "videomme" / "data" / f"{r['videoID']}.mp4"),
             "prompt": PRE + "\n" + r["question"] + "\n" + "\n".join(r["options"]) + "\n" + "\n" + POST,
             "question": r["question"], "answer": r["answer"]} for r in df.to_dict("records")]


def extract_choice(pred):
    s = pred.strip()
    for p in PREFIXES:
        s = s.replace(p, "")
    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return None
    m = re.search(r"[ABCD]", s)
    return m[0] if m else None


def score(doc, pred):
    return float(extract_choice(pred) == doc["answer"])


def summarize(df):
    out = {"overall": 100 * float(df["score"].mean())}
    out.update({k: 100 * float(v) for k, v in df.groupby("task")["score"].mean().items()})
    return out
