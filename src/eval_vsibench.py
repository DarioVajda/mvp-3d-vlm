"""Zero-shot VSI-Bench evaluation of a Qwen3-VL model.

  python src/eval_vsibench.py --model Qwen/Qwen3-VL-2B-Instruct --nframes 32 --out results/vsi/2b_f32.jsonl
  python src/eval_vsibench.py ... --per-type 5          # stratified subset: 5 questions per question type
  python src/eval_vsibench.py ... --nframes 16 --sampling sa   # Spatial-MLLM space-aware frames (needs data/vggt)
  python src/eval_vsibench.py ... --layout stit [--echo-system]  # prompt echoing (see vsibench.build_prompt_parts)

Questions are grouped by video so each video is decoded once; resumes from an existing --out file.
"""
import argparse
import json
import os
import time

import pandas as pd

from pathlib import Path

from vsibench import ECHO_SYSTEM_PROMPT, LAYOUTS, ROOT, build_prompt_parts, load_vsibench, score, summarize, task_of


def frame_indices(path, sampling, nframes, vggt_root):
    """Video frame indices for --sampling uniform128 / sa, chosen among VGGT's 128 candidate frames."""
    from sa_sampling import cached_selection, uniform_subset
    import numpy as np
    scene_dir = Path(vggt_root) / path.split("/")[-2] / path.split("/")[-1].rsplit(".", 1)[0]
    candidates = np.load(scene_dir / "frame_indices.npy")
    if sampling == "sa":
        picks = cached_selection(scene_dir, nframes)
        if len(picks) % 2:  # Qwen3-VL merges frames in pairs; pad with the last pick if the greedy stopped early
            picks.append(picks[-1])
    else:
        picks = uniform_subset(len(candidates), nframes)
    return candidates[picks].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--nframes", type=int, default=32)
    ap.add_argument("--fps", type=float, default=None, help="sample at this fps instead of a fixed --nframes")
    ap.add_argument("--sampling", default="qwen", choices=["qwen", "uniform128", "sa"],
                    help="qwen: qwen_vl_utils uniform sampling over the whole video (default); uniform128 / sa: "
                         "uniform or space-aware subset of the 128 VGGT candidate frames (Spatial-MLLM protocol)")
    ap.add_argument("--scenes", nargs="*", help="only these scene names")
    ap.add_argument("--layout", default="sit", choices=LAYOUTS, help="question position relative to the video")
    ap.add_argument("--echo-system", action="store_true", help="system prompt that mentions the repeated question")
    ap.add_argument("--vggt-root", default=str(ROOT / "data" / "vggt"),
                    help="VGGT outputs used by --sampling sa (data/vggt_portrait_rotated for Spatial-MLLM's rotated portrait frames)")
    ap.add_argument("--split", default="full", choices=["full", "debiased", "pruned"])
    ap.add_argument("--per-type", type=int, default=None, help="sample this many questions per question type")
    ap.add_argument("--ids", default=None, help="file with one question id per line: evaluate only these")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--token-budget", type=int, default=120000, help="max total input tokens per batch")
    ap.add_argument("--shard", type=int, default=0, help="process only videos with index %% num_shards == shard")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = load_vsibench(args.split)
    if args.per_type:
        df = df.groupby("question_type", group_keys=False).apply(
            lambda g: g.sample(min(len(g), args.per_type), random_state=args.seed))
    if args.scenes:
        df = df[df["scene_name"].isin(args.scenes)]
    if args.ids:
        df = df[df["id"].isin([int(l) for l in open(args.ids) if l.strip()])]
    if args.num_shards > 1:
        videos = sorted(df["video_path"].unique())
        df = df[df["video_path"].isin(videos[args.shard::args.num_shards])]
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(l)["id"] for l in f}
    todo = df[~df["id"].isin(done)]
    print(f"{len(df)} questions, {len(done)} done, {len(todo)} to go, {todo['video_path'].nunique()} videos", flush=True)

    from qwen_vl import QwenVL, SYSTEM_PROMPT
    model = QwenVL(args.model)
    system = ECHO_SYSTEM_PROMPT if args.echo_system else SYSTEM_PROMPT
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0, n = time.time(), 0
    with open(args.out, "a") as fout:
        for vi, (path, group) in enumerate(todo.groupby("video_path", sort=False)):
            if args.sampling == "qwen":
                video, meta, vkw = model.load_video(path, nframes=args.nframes, fps=args.fps)
            else:
                video, meta, vkw = model.load_video_frames(path, frame_indices(path, args.sampling, args.nframes, args.vggt_root))
            docs = group.to_dict("records")
            n_vis = video.shape[0] * video.shape[2] * video.shape[3] // (2 * 32 * 32)  # visual tokens
            bs = max(1, min(args.batch_size, args.token_budget // (n_vis + 200)))
            for i in range(0, len(docs), bs):
                batch = docs[i:i + bs]
                parts = [build_prompt_parts(d, args.layout) for d in batch]
                preds, n_tok = model.generate_video(video, meta, vkw, [p for _, p in parts],
                                                    pres=[q for q, _ in parts], system=system)
                for d, p in zip(batch, preds):
                    rec = {"id": int(d["id"]), "dataset": d["dataset"], "scene_name": d["scene_name"],
                           "question_type": d["question_type"], "task": task_of(d["question_type"]),
                           "question": d["question"], "options": d["options"] if isinstance(d["options"], list) else None,
                           "ground_truth": d["ground_truth"], "prediction": p, "score": score(d, p),
                           "n_frames": int(video.shape[0]), "frame_hw": list(video.shape[-2:]), "n_input_tokens": n_tok,
                           "frame_indices": list(meta["frames_indices"])}
                    fout.write(json.dumps(rec) + "\n")
                fout.flush()
                n += len(batch)
            if vi % 20 == 0:
                print(f"[{vi + 1} videos, {n} qs, {time.time() - t0:.0f}s] {path.split('/')[-1]} "
                      f"frames={tuple(video.shape)} tokens={n_tok}", flush=True)

    res = pd.read_json(args.out, lines=True)
    res = res[res["id"].isin(df["id"])]
    summary = summarize(res)
    print(json.dumps({k: round(v, 1) for k, v in summary.items()}, indent=1))
    with open(args.out.replace(".jsonl", "_summary.json"), "w") as f:
        json.dump({"model": args.model, "nframes": args.nframes, "sampling": args.sampling, "vggt_root": args.vggt_root,
                   "layout": args.layout, "system": system, "fps": args.fps, "split": args.split, "n": len(res), **summary}, f, indent=1)


if __name__ == "__main__":
    main()
