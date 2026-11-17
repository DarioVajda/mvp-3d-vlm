"""Attribution for one v7 checkpoint on a stratified VSI-Bench sample (32 frames, the evaluation protocol). No training.

Target m, the same for every arm: log-odds of the correct answer's first token among its alternatives (option letters
for multiple choice, the ten digits for numeric answers, so numeric targets only explain the leading digit).

Maps, one value per visual token (L2 norm of grad x input over the token's dimensions):
  dest[k]  injection Delta_n at decoder layer k: comparable across arms; the signed sum is the first-order effect of
           removing the injection (additive: Delta_n = g_n; xattn: Delta_n = W_o sum_m a_nm v_m)
  src[k]   merger input u_n (after RMSNorm, before the MLP). Not g_n or the VGGT tokens: LN and RMSNorm are scale
           invariant, so grad x input at those points sums to zero by construction
  vis      Qwen's merged visual embedding
xattn only, flow[k]: Delta_n is linear in the attention weights, so the signed first-order attribution splits exactly by
where the mass came from: own cell, rest of own frame, other frames within 1 cell in 3D, other frames further away.
Faithfulness: m after zeroing Delta at the top 5% of cells by dest, at a random 5%, and everywhere.

  explain_saliency.py <run name> <final.pt> [--per-type 30] [--limit N]
"""
import argparse, json, math, os, sys, time
sys.path.insert(0, "src")
import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling
from mvp.benchmarks.vsibench import NA_QUESTION_TYPES, build_prompt, load_vsibench
from mvp.fusion.inputs import uniform_frames
from mvp.fusion.model import FusionVL
from mvp.fusion.modules import AlignedXAttn

OUT = "results/explain_v7"
AC = lambda: torch.autocast("cuda", dtype=torch.bfloat16)


def sample_ids(per_type):
    df = load_vsibench("full")
    return df.groupby("question_type", group_keys=False).apply(lambda g: g.sample(min(len(g), per_type), random_state=0))


def targets(model, doc):
    """(token id of the correct answer's first token, ids of its alternatives)."""
    enc = lambda s: model.processor.tokenizer(s, add_special_tokens=False).input_ids[0]
    if doc["question_type"] in NA_QUESTION_TYPES:
        digits = [enc(str(i)) for i in range(10)]
        first = next(c for c in str(doc["ground_truth"]) if c.isdigit())
        gt = enc(first)
        assert gt in digits
    else:
        letters = [o.strip()[0] for o in doc["options"]]
        digits = [enc(c) for c in letters]
        gt = enc(str(doc["ground_truth"]).strip()[0])
        assert gt in digits, (doc["ground_truth"], letters)
    return gt, [t for t in digits if t != gt]


class Run:
    def __init__(self, path):
        self.model = FusionVL.load(path).cuda().eval().requires_grad_(False)
        self.arm = self.model.arm
        self.keys = [str(k) for k in self.model.pairs]
        self.keep, self.cap, self.delta = None, {}, {}
        if self.arm == "xattn_aligned":
            for k in self.keys:
                self._wrap(k)

    def _wrap(self, k):
        blk = self.model.aligned[k]
        orig = blk.forward
        def forward(y, g, pos, pts=None, kv=None):  # kv: precomputed keys / values (FusionVL passes them)
            out = orig(y, g, pos, pts, kv)
            if out.requires_grad:
                out.retain_grad()
                self.cap[k], self.delta[k] = (y, g, pos, pts), out
            return out * self.keep[k][None, :, None].to(out.dtype) if self.keep is not None else out
        blk.forward = forward

    def geometry(self, batch):
        """As FusionVL.set_geometry, with the merger input u as a leaf. Returns {k: u}."""
        m = self.model
        per_layer, pts = m.vggt_frames(batch["vggt_images"], batch["image_grid_thw"])
        m._pts, m._geo, U = pts, {}, {}
        with AC():
            for k, L in m.pairs.items():
                mg = m.mergers[str(k)]
                us = []
                for _, p in per_layer[L]:
                    h, w, c = p.shape
                    us.append(mg.ln(p).view(h // 2, 2, w // 2, 2, c).permute(0, 2, 1, 3, 4).reshape(-1, 4 * c))
                U[str(k)] = torch.cat(us).float().detach().requires_grad_(True)
                g = mg.mlp(U[str(k)])
                g.retain_grad()
                m._geo[str(k)] = g
        if self.arm == "xattn_aligned":
            m._pos = AlignedXAttn.positions(batch["image_grid_thw"], m.inputs.merge)
        return U

    def logits(self, batch, vis):
        m = self.model
        inp = m.llm_inputs(batch) if vis is None else {**m.llm_inputs(batch), "mm_encoder_outputs": {"image": BaseModelOutputWithPooling(pooler_output=vis)}}
        with AC():
            hs = m.qwen.model(**inp, use_cache=False).last_hidden_state
            return m.qwen.lm_head(hs[0, -1]).float()

    @staticmethod
    def margin(lg, gt, alts):
        return lg[gt] - torch.logsumexp(lg[alts], 0)

    @torch.no_grad()
    def flow(self, k, grad, N):
        """Exact split of Delta by source class, from the attention weights. Returns the signed attributions, the norm
        shares, the reconstruction error against the real Delta and the attention row of the most salient token."""
        y, g, pos, pts = self.cap[k]
        blk = self.model.aligned[k]
        with AC():
            q, kk, v = blk.project(y.detach(), g.detach(), pos, pts)
            lg = (q[0].float() @ kk[0].float().transpose(-1, -2)) / math.sqrt(blk.head_dim)
            if blk.local_bias:
                lg = lg + blk.bias(pos, torch.float, pts)[0]
            a = lg.softmax(-1)
            del lg, q, kk
            v0 = v[0].float()
            t = pos[:, 0]
            same = t[None, :] == t[:, None]
            eye = torch.eye(N, dtype=torch.bool, device=a.device)
            near = pts.dist <= 1
            masks = {"self": eye, "same_frame": same & ~eye, "other_near": ~same & near, "other_far": ~same & ~near}
            D = lambda A: blk.o((A @ v0).transpose(0, 1).reshape(N, -1).to(blk.o.weight.dtype)).float()
            full = D(a)
            real = self.delta[k].detach()[0].float()
            out = {"recon_err": ((full - real).norm() / real.norm().clamp_min(1e-9)).item(), "attr_total": (full * grad).sum().item()}
            for s, mk in masks.items():
                d = D(a * mk)
                out[f"attr_{s}"] = (d * grad).sum().item()
                out[f"norm_{s}"] = d.norm(dim=-1).mean().item()
                out[f"mass_{s}"] = (a * mk).sum(-1).mean().item()
            out["norm_total"] = full.norm(dim=-1).mean().item()
        return out, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("ckpt")
    ap.add_argument("--per-type", type=int, default=30); ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    out = f"{OUT}/sal/{args.name}"
    os.makedirs(out, exist_ok=True)
    df = sample_ids(args.per_type)
    if args.limit:
        df = df.iloc[:: max(1, len(df) // args.limit)]
    R = Run(args.ckpt)
    m = R.model
    print(f"{args.name} arm {R.arm}: {len(df)} questions", flush=True)
    t0, video, vis_in = time.time(), None, None
    fout = open(f"{out}/summary.jsonl", "a")
    for qi, doc in enumerate(df.sort_values("video_path").to_dict("records")):
        path = f"{out}/{doc['id']}.npz"
        if os.path.exists(path):
            continue
        if doc["video_path"] != video:
            video = doc["video_path"]
            frames, idx = uniform_frames(video, 32)
            vis_in = m.inputs.visual(frames)
        text = m.inputs.text([build_prompt(doc)], vis_in["image_grid_thw"], padding_side="left")
        batch = {k: v.cuda() for k, v in {**vis_in, **text}.items()}
        gt, alts = targets(m, doc)
        R.keep = None
        with torch.no_grad():
            feats = m.qwen.model.get_image_features(batch["pixel_values"], batch["image_grid_thw"], return_dict=True).pooler_output
        leaves = tuple(f.detach().requires_grad_(True) for f in feats)
        U = R.geometry(batch) if R.arm != "lora" else {}
        lg = R.logits(batch, leaves)
        mv = R.margin(lg, gt, alts)
        mv.backward()
        V = torch.cat(leaves); Vg = torch.cat([l.grad for l in leaves])
        N = len(V)
        rec = {"id": int(doc["id"]), "question_type": doc["question_type"], "video": video, "N": N, "T": len(batch["image_grid_thw"]),
               "m": mv.item(), "top_token": m.processor.tokenizer.decode(lg.argmax().item()), "gt_token": m.processor.tokenizer.decode(gt)}
        ga = (V.detach().float() * Vg.float())
        arrays = {"vis": ga.norm(dim=-1).float().cpu().numpy(), "grid": batch["image_grid_thw"].cpu().numpy(), "frame_idx": np.array(idx)}
        rec["vis_signed"] = ga.sum().item()
        rng = np.random.default_rng(int(doc["id"]))
        cells = rng.choice(N, min(64, N), replace=False)
        arrays["cells"] = cells
        if R.arm != "lora":
            dest_tot = torch.zeros(N, device=V.device)
            top_rows = {}
            rec["flow"] = {}
            for k in R.keys:
                if R.arm == "additive":
                    d = m._geo[k]; dg = d.grad
                    d, dg = d.detach().float(), dg.float()
                else:
                    d, dg = R.delta[k].detach()[0].float(), R.delta[k].grad[0].float()
                gi = d * dg
                dn = gi.norm(dim=-1)
                dest_tot += dn
                arrays[f"dest_{k}"] = dn.cpu().numpy()
                arrays[f"delta_{k}"] = d[cells].half().cpu().numpy()
                arrays[f"src_{k}"] = (U[k].detach() * U[k].grad).norm(dim=-1).float().cpu().numpy()
                rec[f"dest_signed_{k}"] = gi.sum().item()
                rec[f"delta_norm_{k}"] = d.norm(dim=-1).mean().item()
                if R.arm == "xattn_aligned":
                    fl, a = R.flow(k, dg, N)
                    rec["flow"][k] = fl
                    top = int(dn.argmax())
                    arrays[f"top_{k}"] = np.array(top)
                    arrays[f"attn_top_{k}"] = a[:, top].mean(0).float().cpu().numpy()
                    del a
            # ---- faithfulness: zero Delta at chosen cells, at all three injection points ----
            n_del = max(1, N // 20)
            top_cells = dest_tot.topk(n_del).indices
            rand_cells = torch.as_tensor(rng.choice(N, n_del, replace=False), device=V.device)
            geo0 = {k: v.detach() for k, v in m._geo.items()}
            rec["del"] = {}
            with torch.no_grad():
                for lab, c in (("top5", top_cells), ("rand5", rand_cells), ("all", torch.arange(N, device=V.device))):
                    keep = torch.ones(N, device=V.device); keep[c] = 0
                    if R.arm == "additive":
                        m._geo = {k: g * keep[:, None].to(g.dtype) for k, g in geo0.items()}
                    else:
                        m._geo = geo0
                        R.keep = {k: keep for k in R.keys}
                    rec["del"][lab] = R.margin(R.logits(batch, leaves), gt, alts).item()
                R.keep = None
        np.savez_compressed(path, **arrays)
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        m._geo = None; R.cap.clear(); R.delta.clear()
        if qi % 10 == 0:
            print(f"[{qi + 1}/{len(df)}] {time.time() - t0:.0f}s, N {N}, m {rec['m']:.2f}, mem {torch.cuda.max_memory_allocated() / 2**30:.0f} GB, "
                  f"flow {json.dumps(rec.get('flow', {}).get(R.keys[0], {}))[:300]}, del {rec.get('del')}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
