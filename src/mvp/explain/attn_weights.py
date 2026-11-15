"""Attention weights of the language model from the last prompt position t (where the answer letter is predicted) to
every key position, at each softmax attention layer, for one checkpoint on a few VSI-Bench questions (32 frames, the
evaluation protocol). No training; no change to the model's output.

Qwen3.5-2B has 24 decoder layers; only layers 3, 7, 11, 15, 19, 23 (from 0; config.full_attention_interval = 4) use
softmax attention, with 8 query heads and 2 key / value heads. The other layers are Gated DeltaNet and have no attention
weights. The weights are taken after the q / k norms and RoPE, as the model computes them:
    a_l,h = softmax(q_t,h K_h^T * head_dim^-1/2)      (K_h: the key / value head shared by query head h)
The forward pass itself still runs SDPA unchanged.

Per question, saved to <out>/<run>/:
  <id>_attn.npz     a_<layer>: [8 heads, T] attention weights; visual: [T] bool; frame_of_cell: [N_visual]
  <id>_attn.json    logit difference, answer, and per layer (per head and mean over heads): visual share, per-frame
                    mass, object enrichment (SAM 3 cells from the detail run), Spearman rank correlation with the saved
                    gradient-based saliency maps

  python -m mvp.explain.attn_weights --run 2B_v10_xa3_qknu_s0 --ids 3584 --device mps
"""
import argparse, json, os, sys, time
sys.path.insert(0, "src")
import numpy as np
import torch

_Autocast = torch.autocast


class _DeviceAutocast(_Autocast):
    """The fusion code hard-codes torch.autocast("cuda"); route it to the device in use (mps / cpu) instead."""
    device = "cuda"
    fp32 = False  # --fp32: autocast off everywhere, so the model runs in the float32 it was cast to

    def __init__(self, device_type, *args, **kwargs):
        if self.fp32:
            kwargs["enabled"] = False
        super().__init__(self.device if device_type == "cuda" else device_type, *args, **kwargs)


torch.autocast = _DeviceAutocast

from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from mvp.benchmarks.vsibench import build_prompt, load_vsibench
from mvp.fusion.inputs import uniform_frames
from mvp.fusion.model import FusionVL
from mvp.explain import ROOT
from mvp.explain.explain_saliency import targets
from mvp.explain.one_question import enrichment

CAPTURE = {}  # layer index -> [heads, T] attention weights of the last query position, filled during one forward


def sdpa_lastrow(module, query, key, value, attention_mask, scaling=None, **kwargs):
    out = sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling, **kwargs)
    if CAPTURE.get("on") and query.shape[2] > 1:
        q = query[:, :, -1:].float()                                   # [B, Hq, 1, d]
        k = key.float().repeat_interleave(query.shape[1] // key.shape[1], dim=1)  # as HF repeat_kv: head h uses kv head h // n_rep
        s = (q @ k.transpose(-1, -2))[:, :, 0] * (scaling if scaling is not None else q.shape[-1] ** -0.5)
        CAPTURE[module.layer_idx] = s.softmax(-1)[0].cpu()             # batch of 1, no padding: the last row sees every key
    return out


AttentionInterface.register("sdpa_lastrow", sdpa_lastrow)


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return round(float(np.corrcoef(rx, ry)[0, 1]), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, default=5750)
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--detail", default="results/detail_v10", help="where <run>/<id>_masks.npz and _maps.npz are")
    ap.add_argument("--out", default="results/attn_v10")
    ap.add_argument("--fp32", action="store_true", help="float32 throughout (numerical check against bf16)")
    args = ap.parse_args()
    _DeviceAutocast.device = args.device
    _DeviceAutocast.fp32 = args.fp32
    out = f"{ROOT}/{args.out}/{args.run}"
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    m = FusionVL.load(f"{ROOT}/results/train/{args.run}/branches/step{args.step}/final.pt", grad_ckpt=False)
    m = m.to(args.device).eval().requires_grad_(False)
    if args.fp32:
        m = m.float()
    m.qwen.model.language_model.config._attn_implementation = "sdpa_lastrow"  # text layers only; vision encoder unchanged
    print(f"loaded {args.run} step {args.step} on {args.device} {time.time() - t0:.0f}s", flush=True)

    by_id = load_vsibench("full").set_index("id", drop=False)
    for qid in args.ids:
        doc = dict(by_id.loc[qid])
        frames, idx = uniform_frames(doc["video_path"], 32)
        vis_in = m.inputs.visual(frames)
        text = m.inputs.text([build_prompt(doc)], vis_in["image_grid_thw"], padding_side="left")
        batch = {k: v.to(args.device) for k, v in {**vis_in, **text}.items()}
        assert bool(batch["attention_mask"].all()), "expects one unpadded question"
        gt, alts = targets(m, doc)

        t1 = time.time()
        CAPTURE.clear()
        CAPTURE["on"] = True
        with torch.no_grad():
            m._geo = None
            if m.vggt is not None:
                m.set_geometry(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hs = m.qwen.model(**m.llm_inputs(batch), use_cache=False).last_hidden_state
                lg = m.qwen.lm_head(hs[0, -1]).float()
            m._geo = None
        CAPTURE.pop("on")
        if args.device == "mps":
            torch.mps.synchronize()
        sec = time.time() - t1

        layers = sorted(CAPTURE)
        visual = (batch["input_ids"][0] == m.inputs.image_token_id).cpu().numpy()
        grids = vis_in["image_grid_thw"].tolist()
        n_per = [(h // 2) * (w // 2) for _, h, w in grids]
        fr = np.concatenate([np.full(n, t) for t, n in enumerate(n_per)])
        assert visual.sum() == len(fr), (visual.sum(), len(fr))
        tok = m.processor.tokenizer
        margin = (lg[gt] - torch.logsumexp(lg[alts], 0)).item()
        rec = {"id": qid, "run": args.run, "step": args.step, "device": args.device, "forward_seconds": round(sec, 2),
               "question": doc["question"], "options": list(doc["options"]), "ground_truth": doc["ground_truth"],
               "top_token": tok.decode(lg.argmax().item()), "gt_token": tok.decode(gt), "logit_difference": margin,
               "T": int(len(visual)), "N_visual": int(visual.sum()), "softmax_layers": layers, "layers": {}}

        masks_p = f"{ROOT}/{args.detail}/{args.run}/{qid}_masks.npz"
        maps_p = f"{ROOT}/{args.detail}/{args.run}/{qid}_maps.npz"
        masks = dict(np.load(masks_p)) if os.path.exists(masks_p) else {}
        maps = dict(np.load(maps_p)) if os.path.exists(maps_p) else {}
        if maps:
            maps["dest_sum"] = sum(v for k, v in maps.items() if k.startswith("dest_"))
        objects = [k for k in masks if k != "frame_of_cell"]

        arrays = {"visual": visual, "frame_of_cell": fr}
        for L in layers:
            A = CAPTURE[L].numpy()                         # [heads, T]
            arrays[f"a_{L}"] = A.astype(np.float32)
            heads = {f"h{h}": A[h] for h in range(A.shape[0])}
            heads["mean"] = A.mean(0)                      # mean over the head axis
            lay = {}
            for name, a in heads.items():
                v = a[visual]
                r = {"sum": round(float(a.sum()), 5), "visual_share": round(float(v.sum()), 4),
                     "frame_mass": [round(float(v[fr == t].sum()), 4) for t in range(len(n_per))]}
                if objects:
                    r["enrichment"] = {p: enrichment(v, masks[p], fr) for p in objects}
                if maps:
                    r["spearman_vs_saliency"] = {k: spearman(v, s) for k, s in maps.items()}
                lay[name] = r
            rec["layers"][str(L)] = lay
        np.savez_compressed(f"{out}/{qid}_attn.npz", **arrays)
        json.dump(rec, open(f"{out}/{qid}_attn.json", "w"), indent=1)
        mean_vis = {L: rec["layers"][str(L)]["mean"]["visual_share"] for L in layers}
        print(f"{qid}: answer {rec['top_token']!r} gt {rec['gt_token']!r} logit diff {margin:.3f} T {rec['T']} "
              f"visual {rec['N_visual']} layers {layers} visual share (mean over heads) {mean_vis} forward {sec:.1f}s",
              flush=True)
    print(f"done {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
