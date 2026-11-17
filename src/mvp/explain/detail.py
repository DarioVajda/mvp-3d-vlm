"""Detailed per-question analysis of one checkpoint on a few VSI-Bench questions (32 frames). No training.

Per question:
  masks     SAM 3, one text prompt per object named in the question (one_question.object_masks_sam3), pooled to cells
  saliency  vis and dest_0..2 (one_question.saliency); enrichment per object and map
  attention cross-attention only, per layer: object -> object mass (learned, 3D bias alone, uniform), and for each object
            the mean attention row of its cells over all keys (where it reads from), learned and bias-only
  J-lens    the paper's text lens and our visual lens (jspace.py, unchanged), fitted on this checkpoint, read at layers
            jspace.SOURCES at: the model's own top-5 saliency cells ("top"), 5 random cells ("rand"), up to 3 cells per
            object ("own": the cells its SAM 3 mask covers most), and the last 12 text tokens; with the geometry delta
            (h with geometry - h without) at every visual cell read

  python -m mvp.explain.detail --run 2B_v10_xa3_qknu_s0 --ids 1221 1525 --out results/detail_v10
"""
import argparse, json, os, sys, time
sys.path.insert(0, "src")
import numpy as np
import torch
from mvp.benchmarks.vsibench import load_vsibench
from mvp.fusion.inputs import EVAL_PIXELS, InputBuilder, uniform_frames
from mvp.fusion.model import FusionVL
from mvp.explain import one_question as OQ
from mvp.explain.jspace import jspace as JS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, default=5750)
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--prompts", type=int, default=30, help="text-lens prompts (v7 used 30)")
    ap.add_argument("--videos", type=int, default=3, help="visual-lens videos (v7 used 3)")
    ap.add_argument("--out", default="results/detail_v10")
    args = ap.parse_args()
    out = f"{args.out}/{args.run}"
    os.makedirs(out, exist_ok=True)
    ckpt = f"results/train/{args.run}/branches/step{args.step}/final.pt"
    t0 = time.time()
    df = load_vsibench("full")
    by_id = df.set_index("id", drop=False)
    docs = {i: dict(by_id.loc[i]) for i in args.ids}
    from transformers import AutoProcessor
    model_id = json.load(open(f"results/train/{args.run}/args.json")).get("model", "Qwen/Qwen3.5-2B")
    builder = InputBuilder(AutoProcessor.from_pretrained(model_id, **EVAL_PIXELS))

    # ---- masks + saliency + attention, per question ----
    info = {}
    for qid, doc in docs.items():
        phrases = OQ.phrases_of(doc)
        frames, idx = uniform_frames(doc["video_path"], 32)
        grids = builder.visual(frames)["image_grid_thw"].tolist()
        cells, pix, det = OQ.object_masks_sam3(frames, phrases, grids, args.thr, 3, "cuda")
        n_per = [(g[1] // 2) * (g[2] // 2) for g in grids]
        fr = np.concatenate([np.full(n, t) for t, n in enumerate(n_per)])
        rec, maps, attn = OQ.saliency(ckpt.rsplit("/", 1)[0], doc, frames, idx)
        dest = sum(maps[k] for k in maps if k.startswith("dest_"))
        rec["enrichment"] = {name: {p: OQ.enrichment(v, cells[p], fr) for p in phrases} for name, v in {**maps, "dest_sum": dest}.items()}
        rows = {}
        if attn:
            rec["object_attention"] = {}
            for k, A in attn.items():
                tab = {}
                for px in phrases:
                    if cells[px].sum() == 0:
                        continue
                    w = cells[px] / cells[px].sum()
                    rows[f"{k}:{px}:learned"] = (w @ A["a"]).astype(np.float32)   # where X's cells read from
                    rows[f"{k}:{px}:bias"] = (w @ A["a0"]).astype(np.float32)
                    for py in phrases:
                        if px != py and cells[py].sum() > 0:
                            tab[f"{px} -> {py}"] = {"learned": round(OQ.obj_attention(A["a"], cells[px], cells[py]), 4),
                                                    "bias_only": round(OQ.obj_attention(A["a0"], cells[px], cells[py]), 4),
                                                    "area_y": round(float(cells[py].sum() / len(fr)), 4)}
                rec["object_attention"][k] = {"pairs": tab, "scale": A["scale"],
                                              "flow_mass": {s: round(A["flow"][f"mass_{s}"], 4) for s in ("self", "same_frame", "other_near", "other_far")}}
            del attn
        rec.update({"id": qid, "question_type": doc["question_type"], "question": doc["question"], "ground_truth": doc["ground_truth"],
                    "options": list(doc["options"]) if doc.get("options") is not None else None, "video": doc["video_path"],
                    "frame_idx": idx, "grids": grids, "phrases": phrases,
                    "masks": {p: {"role": phrases[p], "frames_detected": int(sum(m.any() for m in pix[p])),
                                  "mean_cells_covered": round(float(cells[p].sum() / max(1, sum(m.any() for m in pix[p]))), 2),
                                  "detections": {t: d for t, d in enumerate(det[p]) if d}} for p in phrases}})
        json.dump(rec, open(f"{out}/{qid}_report.json", "w"), indent=1)
        np.savez_compressed(f"{out}/{qid}_maps.npz", **maps)
        np.savez_compressed(f"{out}/{qid}_masks.npz", **cells, frame_of_cell=fr)
        if rows:
            np.savez_compressed(f"{out}/{qid}_attn_rows.npz", **rows)
        own = {p: np.argsort(cells[p])[::-1][:3][cells[p][np.argsort(cells[p])[::-1][:3]] > 0.5].tolist() for p in phrases}
        info[qid] = {"top": np.argsort(dest)[::-1][:5].tolist(), "own": own,
                     "rand": np.random.default_rng(qid).choice(len(dest), 5, replace=False).tolist()}
        print(f"{qid}: {rec['question_type']} answer {rec['top_token']!r} gt {rec['gt_token']!r} margin {rec['margin']:.2f} "
              f"masks {[(p, v['frames_detected']) for p, v in rec['masks'].items()]} {time.time() - t0:.0f}s", flush=True)
    json.dump(info, open(f"{out}/cells.json", "w"))

    # ---- J-lens, fitted on this checkpoint ----
    m = FusionVL.load(ckpt, grad_ckpt=False).cuda().eval().requires_grad_(False)
    R = JS.Reader(m)
    prompts = JS.wiki_prompts(args.prompts + 20)
    fit_p, eval_p = prompts[:args.prompts], prompts[args.prompts:]
    q_videos = {d["video_path"] for d in docs.values()}
    fit_docs = df[~df["video_path"].isin(q_videos)].drop_duplicates("video_path").sample(args.videos, random_state=0).to_dict("records")
    lenses = {"text": JS.fit_text(m, R.tok, fit_p, 32, f"{out}/lens_text.pt"),
              "visual": JS.fit_visual(m, fit_docs, 16, f"{out}/lens_visual.pt")}
    print(f"lenses fitted {time.time() - t0:.0f}s", flush=True)
    JS.sanity(R, lenses, f"{out}/sanity.json")
    q = {n: JS.lens_quality(R, J, eval_p, min(20, len(eval_p))) for n, J in lenses.items()}
    json.dump(q, open(f"{out}/lens_quality.json", "w"), indent=1)
    with open(f"{out}/readouts.jsonl", "w") as fout:
        for qid, doc in docs.items():
            c = info[qid]
            own = [x for p in c["own"].values() for x in p]
            ans = JS.readout(R, lenses, doc, c["top"], c["rand"], fout, args.run, own)
            print(f"  J-lens {qid}: answer {ans!r} {time.time() - t0:.0f}s", flush=True)
    print(f"done {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
