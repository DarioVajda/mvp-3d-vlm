"""Evaluation of a trained arm or a base model (through mvp.fusion) on SPAR-Bench-Tiny, MindCube-Tiny, VSTI-Bench or
Video-MME (data, prompts and scoring: mvp.benchmarks.<bench>).

  python -m mvp.eval.bench --bench vstibench --fusion-ckpt results/train/2B_mix500k_xattn/branches/step54691/final.pt \
      --out results/train/2B_mix500k_xattn/branches/step54691/eval/vstibench.jsonl
  python -m mvp.eval.bench --bench spar --model Qwen/Qwen3.5-2B --out ...  # base model, same inputs as the arms
  ... --limit 5  # first 5 questions per task (smoke test)
  ... --max-new-tokens 2048  # room to reason first (base models)

Video benchmarks: the module's NFRAMES uniform frames, each video decoded once for all its questions. Image
benchmarks: one question at a time. Resumes from an existing --out file; writes <out>_summary.json.
"""
import argparse
import importlib
import json
import os
import time

import pandas as pd

BENCHES = ["spar", "mindcube", "vstibench", "videomme"]
DROP = {"image_bytes", "image_paths", "prompt"}  # not written to --out


def generate(model, vis, batch, max_new_tokens):
    """Greedy answers to a batch of questions about the same frames, and the padded input length; a batch that runs
    out of GPU memory is answered in two halves."""
    import torch
    try:
        batch_in = {**vis, **model.inputs.text([d["prompt"] for d in batch], vis["image_grid_thw"], padding_side="left")}
        return model.generate({k: v.cuda() for k, v in batch_in.items()}, max_new_tokens), int(batch_in["input_ids"].shape[1])
    except torch.cuda.OutOfMemoryError:
        if len(batch) == 1:
            raise
    torch.cuda.empty_cache()
    print(f"OOM at batch size {len(batch)}; splitting", flush=True)
    (a, na), (b, nb) = (generate(model, vis, half, max_new_tokens) for half in (batch[:len(batch) // 2], batch[len(batch) // 2:]))
    return a + b, max(na, nb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=BENCHES)
    ap.add_argument("--model", help="base model id; required unless --fusion-ckpt is given")
    ap.add_argument("--fusion-ckpt", help="trained arm from mvp.train.run")
    ap.add_argument("--limit", type=int, default=None, help="only the first this many questions per task")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--token-budget", type=int, default=120000, help="max total input tokens per batch (video benchmarks)")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="default: the benchmark's official limit (a base "
                    "model that reasons before answering needs more)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert args.model or args.fusion_ckpt, "--model or --fusion-ckpt"
    bench = importlib.import_module(f"mvp.benchmarks.{args.bench}")
    args.max_new_tokens = args.max_new_tokens or bench.MAX_NEW_TOKENS

    docs = bench.load()
    if args.limit:
        docs = pd.DataFrame(docs).groupby("task", group_keys=False).head(args.limit).to_dict("records")
    ids = {d["id"] for d in docs}
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(l)["id"] for l in f}
    todo = [d for d in docs if d["id"] not in done]
    video = "video" in docs[0]
    print(f"{args.bench}: {len(docs)} questions, {len(done & ids)} done, {len(todo)} to go", flush=True)

    from mvp.fusion.model import FusionVL
    from mvp.fusion.inputs import uniform_frames
    model = (FusionVL.load(args.fusion_ckpt) if args.fusion_ckpt else FusionVL(args.model, "base")).cuda().eval()
    args.model = model.config["model_id"]
    if video:  # one group per video, in order of first appearance
        groups = {}
        for d in todo:
            groups.setdefault(d["video"], []).append(d)
        groups = list(groups.values())
    else:
        groups = [[d] for d in todo]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0, n = time.time(), 0
    with open(args.out, "a") as fout:
        for gi, group in enumerate(groups):
            if video:
                frames, _ = uniform_frames(group[0]["video"], bench.NFRAMES)
            else:
                frames = bench.images(group[0])
            vis = model.inputs.visual(frames)
            n_vis = int((vis["image_grid_thw"].prod(-1) // 4).sum())
            bs = max(1, min(args.batch_size, args.token_budget // (n_vis + 300)))
            for i in range(0, len(group), bs):
                batch = group[i:i + bs]
                preds, n_tok = generate(model, vis, batch, args.max_new_tokens)
                for d, p in zip(batch, preds):
                    rec = {k: v for k, v in d.items() if k not in DROP}
                    rec.update(prediction=p, score=bench.score(d, p), n_frames=len(frames), n_input_tokens=n_tok)
                    fout.write(json.dumps(rec) + "\n")
                fout.flush()
                n += len(batch)
            if gi % 50 == 0:
                print(f"[{gi + 1}/{len(groups)} groups, {n} qs, {time.time() - t0:.0f}s] tokens={n_tok} "
                      f"pred={preds[0]!r} gt={batch[0]['answer']!r}", flush=True)

    res = pd.read_json(args.out, lines=True, dtype={"id": str} if not isinstance(docs[0]["id"], int) else None)
    res = res[res["id"].isin(ids)]
    summary = bench.summarize(res)
    print(json.dumps({k: round(v, 1) for k, v in summary.items()}, indent=1))
    with open(args.out.replace(".jsonl", "_summary.json"), "w") as f:
        json.dump({"bench": args.bench, "model": args.model, "fusion_ckpt": args.fusion_ckpt, "arm": model.arm,
                   "limit": args.limit, "max_new_tokens": args.max_new_tokens, "n": len(res), **summary}, f, indent=1)


if __name__ == "__main__":
    main()
