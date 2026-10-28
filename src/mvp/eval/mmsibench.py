"""Zero-shot MMSI-Bench evaluation of a Qwen3-VL model (prompt and parsing: mvp.benchmarks.mmsibench).

  python -m mvp.eval.mmsibench --model Qwen/Qwen3-VL-2B-Instruct --out results/zeroshot/mmsibench/2b.jsonl [--per-type 5]
  python -m mvp.eval.mmsibench --fusion-ckpt results/train/2B_xattn/final.pt --max-new-tokens 2048 --out ...  # trained arm
  python -m mvp.eval.mmsibench --model Qwen/Qwen3.5-4B --images ...  # base model through mvp.fusion (same inputs as the arms)
"""
import argparse
import json
import os

import pandas as pd

from mvp.benchmarks.mmsibench import POST_PROMPT, extract_choice, to_pil
from mvp.paths import MMSI_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="required unless --fusion-ckpt is given")
    ap.add_argument("--images", action="store_true", help="through mvp.fusion: SpatialStack's eval pixel limits, VGGT on "
                    "the question's images (the inputs the trained arms see)")
    ap.add_argument("--fusion-ckpt", help="trained arm from mvp.train.run (implies --images)")
    ap.add_argument("--per-type", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.images = args.images or args.fusion_ckpt is not None
    assert args.model or args.fusion_ckpt, "--model or --fusion-ckpt"

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

    if args.images:
        from mvp.fusion.model import FusionVL
        model = (FusionVL.load(args.fusion_ckpt) if args.fusion_ckpt else FusionVL(args.model, "base")).cuda().eval()
        args.model = model.config["model_id"]
    else:
        from mvp.models.qwen_vl import QwenVL
        model = QwenVL(args.model)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as fout:
        for i, d in enumerate(todo.to_dict("records")):
            prompt = f"{d['question']}\n{POST_PROMPT}"
            if args.images:
                batch = model.inputs.build([to_pil(x).convert("RGB") for x in d["images"]], [prompt], padding_side="left")
                (pred,) = model.generate({k: v.cuda() for k, v in batch.items()}, args.max_new_tokens)
                n_tok = int(batch["input_ids"].shape[1])
            else:
                images = model.load_images([to_pil(x) for x in d["images"]])
                (pred,), n_tok = model.generate_images(images, prompt, args.max_new_tokens)
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
        json.dump({"model": args.model, "protocol": "images" if args.images else "qwen_vl", "fusion_ckpt": args.fusion_ckpt,
                   "arm": model.arm if args.images else None, "n": len(res), **summary}, f, indent=1)


if __name__ == "__main__":
    main()
