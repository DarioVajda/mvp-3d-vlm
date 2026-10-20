"""Zero-shot MMSI-Bench evaluation of a Qwen3-VL model.

Prompt and answer extraction follow the official repo (InternRobotics/MMSI-Bench, vlmeval/dataset/mmsi_bench.py):
images, then "{question}\nAnswer with the option's letter from the given choices directly. Enclose the option's
letter within ``.", parsed with the regex extractor (``x`` -> `x` -> first standalone A-D). The official
pipeline falls back to an LLM judge for unparsed answers; we count them as wrong and report the parse rate.

  python src/eval_mmsi.py --model Qwen/Qwen3-VL-2B-Instruct --out results/mmsi/2b.jsonl [--per-type 5]
"""
import argparse
import io
import json
import os
import re
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MMSI_PATH = ROOT / "data" / "raw" / "MMSI-Bench" / "MMSI_Bench.parquet"
POST_PROMPT = "Answer with the option's letter from the given choices directly. Enclose the option's letter within ``."


def extract_choice(pred):
    m = re.search(r"``([^`]*)``", pred)
    if m:
        pred = m.group(1)
    m = re.search(r"`([^`]*)`", pred)
    if m:
        pred = m.group(1)
    m = re.search(r"\b[A-D]\b(?!\s[a-zA-Z])", pred)
    return m.group() if m else None


def to_pil(x):
    return Image.open(io.BytesIO(x["bytes"] if isinstance(x, dict) else x)).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--per-type", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(MMSI_PATH)
    if args.per_type:
        df = df.groupby("question_type", group_keys=False).apply(
            lambda g: g.sample(min(len(g), args.per_type), random_state=args.seed))
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(l)["id"] for l in f}
    todo = df[~df["id"].isin(done)]
    print(f"{len(df)} questions, {len(todo)} to go", flush=True)

    from qwen_vl import QwenVL
    model = QwenVL(args.model)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as fout:
        for i, d in enumerate(todo.to_dict("records")):
            images = model.load_images([to_pil(x) for x in d["images"]])
            (pred,), n_tok = model.generate_images(images, f"{d['question']}\n{POST_PROMPT}", args.max_new_tokens)
            choice = extract_choice(pred)
            rec = {"id": int(d["id"]), "question_type": d["question_type"], "difficulty": d["difficulty"],
                   "n_images": len(d["images"]), "question": d["question"], "answer": d["answer"],
                   "prediction": pred, "choice": choice, "score": float(choice == d["answer"]), "n_input_tokens": n_tok}
            fout.write(json.dumps(rec) + "\n")
            if i % 100 == 0:
                print(f"[{i}] tokens={n_tok} pred={pred!r} gt={d['answer']}", flush=True)
                fout.flush()

    res = pd.read_json(args.out, lines=True)
    res = res[res["id"].isin(df["id"])]
    summary = {k: round(100 * v, 1) for k, v in res.groupby("question_type")["score"].mean().items()}
    summary["overall"] = round(100 * res["score"].mean(), 1)
    summary["parse_rate"] = round(100 * res["choice"].notna().mean(), 1)
    print(json.dumps(summary, indent=1))
    with open(args.out.replace(".jsonl", "_summary.json"), "w") as f:
        json.dump({"model": args.model, "n": len(res), **summary}, f, indent=1)


if __name__ == "__main__":
    main()
