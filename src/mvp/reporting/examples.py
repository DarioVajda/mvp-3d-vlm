"""Collect a few VSI-Bench / MMSI-Bench examples with model predictions for inspection.

Writes results/figures/examples/{vsi_<id>.jpg, mmsi_<id>_<k>.jpg, examples.json}. Light enough for the login node
(decodes 8 frames from each of 8 videos with decord).

  python -m mvp.reporting.examples --vsi results/zeroshot/vsibench/qwen_sampling/{2B,4B}_f32.jsonl \
      --mmsi results/zeroshot/mmsibench/{2B,4B}.jsonl [--failures-of 4B_f32]
"""
import argparse
import json
from pathlib import Path

import decord
import numpy as np
import pandas as pd
from PIL import Image

from mvp.benchmarks.mmsibench import to_pil
from mvp.benchmarks.vsibench import TASKS, load_vsibench
from mvp.paths import FIGURES, MMSI_PATH

OUT = FIGURES / "examples"


def frame_grid(path, n=8, cols=4, width=320):
    vr = decord.VideoReader(path)
    idx = np.linspace(0, len(vr) - 1, n, dtype=int)
    frames = [Image.fromarray(f) for f in vr.get_batch(idx).asnumpy()]
    ims = [f.resize((width, int(f.height * width / f.width))) for f in frames]
    h = max(im.height for im in ims)
    grid = Image.new("RGB", (cols * width, (n + cols - 1) // cols * h), "white")
    for k, im in enumerate(ims):
        grid.paste(im, (k % cols * width, k // cols * h))
    return grid


def video_link(d):
    """Markdown link to the video, relative to the project root (where README.md lives)."""
    rel = f"data/raw/VSI-Bench/{d['dataset']}/{d['scene_name']}.mp4"
    return f"[`{d['dataset']}/{d['scene_name']}.mp4`]({rel})"


def fmt_pred(r, numeric):
    return f"`{r['prediction']}` " + (f"(MRA {r['score']:.1f})" if numeric else ("✓" if r["score"] else "✗"))


def write_failures(vsi, preds, model, n, rng):
    """Pick `n` debiased questions that `model` got fully wrong (score 0: wrong letter, or MRA 0, i.e. >50%
    relative error), spread round-robin over the 8 tasks; write frame grids, failures.json and failures.md."""
    p = preds[model]
    debiased = set(load_vsibench("debiased")["id"])
    wrong = [i for i in p.index if i in debiased and p.loc[i, "score"] == 0]
    by_task = {t: list(rng.permutation([i for i in wrong if vsi.loc[i, "question_type"].startswith(t)])) for t in TASKS}
    picked = []
    while len(picked) < n and any(by_task.values()):
        for t in TASKS:
            if by_task[t] and len(picked) < n:
                picked.append((t, int(by_task[t].pop())))
    recs, md = [], []
    for k, (task, i) in enumerate(picked, 1):
        d = vsi.loc[i]
        numeric = d["options"] is None
        frame_grid(d["video_path"]).save(OUT / f"vsi_{i}.jpg", quality=85)
        rec = {"id": i, "task": task, "question_type": d["question_type"], "dataset": d["dataset"],
               "scene_name": d["scene_name"], "question": d["question"],
               "options": None if numeric else list(d["options"]), "ground_truth": str(d["ground_truth"]),
               "predictions": {m: {"prediction": q.loc[i, "prediction"], "score": float(q.loc[i, "score"])}
                               for m, q in preds.items() if i in q.index}}
        recs.append(rec)
        others = " · ".join(f"{m.split('_')[0]}: {fmt_pred(r, numeric)}" for m, r in rec["predictions"].items() if m != model)
        md += [f"**{k}. {d['question_type']}** ({video_link(d)}, id {i})",
               "",
               "> " + d["question"].strip().replace("\n", "\n> "),
               ""]
        if not numeric:
            md += [" · ".join(rec["options"]), ""]
        md += [f"GT **{rec['ground_truth']}** · **{model.split('_')[0]}: {fmt_pred(rec['predictions'][model], numeric)}**"
               + (f" · {others}" if others else ""), "", f"![](results/figures/examples/vsi_{i}.jpg)", ""]
    json.dump(recs, open(OUT / "failures.json", "w"), indent=1, ensure_ascii=False)
    (OUT / "failures.md").write_text("\n".join(md))
    print(f"wrote {len(recs)} failures of {model} to {OUT}/failures.md")


def model_name(path):
    return Path(path).stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vsi", nargs="+", required=True)
    ap.add_argument("--mmsi", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--failures-of", default=None, help="also pick debiased failures of this run (e.g. 4B_f32)")
    ap.add_argument("--n-failures", type=int, default=10)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    vsi = load_vsibench("full").set_index("id")
    preds = {model_name(p): pd.read_json(p, lines=True).set_index("id") for p in args.vsi}
    common = sorted(set.intersection(*[set(p.index) for p in preds.values()]))
    examples = []
    for task in TASKS:
        ids = [i for i in common if vsi.loc[i, "question_type"].startswith(task)]
        i = int(rng.choice(ids))
        d = vsi.loc[i]
        frame_grid(d["video_path"]).save(OUT / f"vsi_{i}.jpg", quality=85)
        examples.append({"bench": "VSI-Bench", "id": i, "task": task, "question_type": d["question_type"],
                         "source": f"{d['dataset']}/{d['scene_name']}", "image": [f"vsi_{i}.jpg"],
                         "question": d["question"], "options": list(d["options"]) if d["options"] is not None else None,
                         "ground_truth": str(d["ground_truth"]),
                         "predictions": {m: {"prediction": p.loc[i, "prediction"], "score": float(p.loc[i, "score"])}
                                         for m, p in preds.items()}})

    mmsi = pd.read_parquet(MMSI_PATH).set_index("id")
    preds = {model_name(p): pd.read_json(p, lines=True).set_index("id") for p in args.mmsi}
    common = sorted(set.intersection(*[set(p.index) for p in preds.values()]))
    for qt in ["Positional Relationship (Cam.–Obj.)", "Motion (Cam.)", "Attribute (Meas.)", "MSR"]:
        i = int(rng.choice([j for j in common if mmsi.loc[j, "question_type"] == qt]))
        d = mmsi.loc[i]
        names = []
        for k, im in enumerate(d["images"]):
            im = to_pil(im)
            im.thumbnail((480, 480))
            im.save(OUT / f"mmsi_{i}_{k}.jpg", quality=85)
            names.append(f"mmsi_{i}_{k}.jpg")
        examples.append({"bench": "MMSI-Bench", "id": i, "task": qt, "question_type": qt, "source": d["difficulty"],
                         "image": names, "question": d["question"], "options": None, "ground_truth": d["answer"],
                         "predictions": {m: {"prediction": p.loc[i, "prediction"], "score": float(p.loc[i, "score"])}
                                         for m, p in preds.items()}})
    json.dump(examples, open(OUT / "examples.json", "w"), indent=1, ensure_ascii=False)
    print(f"wrote {len(examples)} examples to {OUT}")
    if args.failures_of:
        vsi_preds = {model_name(p): pd.read_json(p, lines=True).set_index("id") for p in args.vsi}
        write_failures(vsi, vsi_preds, args.failures_of, args.n_failures, np.random.default_rng(args.seed))


if __name__ == "__main__":
    main()
