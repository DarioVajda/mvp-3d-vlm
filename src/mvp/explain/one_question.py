"""One-question test of the v10 object-saliency pipeline. No training.

For one VSI-Bench question at 32 frames (the evaluation protocol, the same frames as explain_saliency.py):
  1. object masks: the question's object phrases (objwords.objects) -> OWLv2 boxes -> SAM 2 masks, per frame, pooled to
     each frame's own merged-token grid (fraction of each cell covered; portrait frames handled by image_grid_thw)
  2. saliency for each checkpoint, as explain_saliency.py: grad x input at Qwen's visual embedding (vis) and at the
     geometry injection of layers 0..2 (dest_k); xattn also returns the attention weights
  3. metrics: object enrichment (share of a map's mass on an object's cells / share of cells it covers, over the frames
     where it is detected); xattn: attention mass from object X's cells to object Y's cells, against the same mass
     under the 3D bias alone, and the size of the learned q.k logits against the bias
  4. a figure: frames with masks, and the maps of each model

  python -m mvp.explain.one_question --id 3004 --runs 2B_v9_fmix_additive_s0 2B_v10_xa3_qknu_s0 --step 5750
"""
import argparse, json, math, os, sys, time
sys.path.insert(0, "src")
import numpy as np
import torch
import torch.nn.functional as F
from mvp.benchmarks.vsibench import build_prompt, load_vsibench
from mvp.fusion.inputs import uniform_frames
from mvp.explain import explain_saliency as ES
from mvp.explain.objwords import objwords

OWL = "google/owlv2-large-patch14-ensemble"
SAM = "facebook/sam2.1-hiera-large"


def phrases_of(doc):
    t = doc["question_type"]
    d = {**doc, "task": "object_rel_direction" if t.startswith("object_rel_direction") else t}
    roles, _ = objwords.objects(d)
    if roles is None:
        raise SystemExit(f"parser found no objects in {doc['question']!r}")
    out = {}
    for role, p in roles:  # one entry per phrase (route landmarks can repeat start / goal)
        out.setdefault(p, role)
    return out  # phrase -> first role


@torch.no_grad()
def object_masks(frames, phrases, grids, thr, max_inst, dev):
    from transformers import Owlv2ForObjectDetection, Owlv2Processor, Sam2Model, Sam2Processor
    op, om = Owlv2Processor.from_pretrained(OWL), Owlv2ForObjectDetection.from_pretrained(OWL).to(dev).eval()
    sp, sm = Sam2Processor.from_pretrained(SAM), Sam2Model.from_pretrained(SAM).to(dev).eval()
    P = list(phrases)
    cells = {p: [] for p in P}       # per frame: [h*w] covered fraction
    pix = {p: [] for p in P}         # per frame: full-res bool mask (for the figure)
    det = {p: [] for p in P}         # per frame: [(score, box)]
    for fi, im in enumerate(frames):
        W, H = im.size
        S = max(W, H)  # OWLv2 pads to a square at the bottom / right, so boxes are in the padded square's pixels
        inp = op(text=[[f"a photo of a {p}" for p in P]], images=im, return_tensors="pt").to(dev)
        r = op.post_process_grounded_object_detection(om(**inp), threshold=thr, target_sizes=[(S, S)])[0]
        _, h, w = grids[fi]
        for j, p in enumerate(P):
            sel = (r["labels"] == j).nonzero().flatten()
            sel = sel[r["scores"][sel].argsort(descending=True)][:max_inst]
            boxes = [[max(0., b[0]), max(0., b[1]), min(W, b[2]), min(H, b[3])] for b in r["boxes"][sel].tolist()]
            det[p].append([(round(s, 3), [round(x) for x in b]) for s, b in zip(r["scores"][sel].tolist(), boxes)])
            if not boxes:
                m = torch.zeros(H, W, dtype=torch.bool)
            else:
                si = sp(images=im, input_boxes=[boxes], return_tensors="pt").to(dev)
                so = sm(**si, multimask_output=False)
                m = sp.post_process_masks(so.pred_masks.cpu(), si["original_sizes"].cpu())[0][:, 0].bool().any(0)
            pix[p].append(m.numpy())
            cells[p].append(F.interpolate(m[None, None].float(), size=(h // 2, w // 2), mode="area").flatten().numpy())
    del om, sm
    torch.cuda.empty_cache()
    return {p: np.concatenate(cells[p]) for p in P}, pix, det


SAM3_REV = "3c879f39826c281e95690f02c7821c4de09afae7"  # the revision the reported results used
_snap = os.path.join(os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub")),
                     "models--facebook--sam3", "snapshots", SAM3_REV)
# facebook/sam3 is gated: load the cached snapshot when present (no Hub access needed), else download that revision
SAM3, SAM3_KW = (_snap, {}) if os.path.isdir(_snap) else ("facebook/sam3", {"revision": SAM3_REV})


@torch.no_grad()
def object_masks_sam3(frames, phrases, grids, thr, max_inst, dev):
    """SAM 3 promptable concept segmentation: one text prompt per object, masks of every matching instance; the image
    encoder runs once per frame. Pinned to the revision the reported results used."""
    from transformers import Sam3Model, Sam3Processor
    proc = Sam3Processor.from_pretrained(SAM3, **SAM3_KW)
    model = Sam3Model.from_pretrained(SAM3, **SAM3_KW).to(dev).eval()
    P = list(phrases)
    texts = {p: proc(text=p, return_tensors="pt").to(dev) for p in P}
    cells, pix, det = ({p: [] for p in P} for _ in range(3))
    for fi, im in enumerate(frames):
        W, H = im.size
        img = proc(images=im, return_tensors="pt").to(dev)
        vis = model.get_vision_features(pixel_values=img.pixel_values)
        _, h, w = grids[fi]
        for p in P:
            out = model(vision_embeds=vis, input_ids=texts[p].input_ids, attention_mask=texts[p].attention_mask)
            r = proc.post_process_instance_segmentation(out, threshold=thr, target_sizes=[(H, W)])[0]
            order = r["scores"].argsort(descending=True)[:max_inst]
            det[p].append([(round(s, 3), [round(x) for x in b]) for s, b in zip(r["scores"][order].tolist(), r["boxes"][order].tolist())])
            m = r["masks"][order].bool().any(0).cpu() if len(order) else torch.zeros(H, W, dtype=torch.bool)
            pix[p].append(m.numpy())
            cells[p].append(F.interpolate(m[None, None].float(), size=(h // 2, w // 2), mode="area").flatten().numpy())
    del model
    torch.cuda.empty_cache()
    return {p: np.concatenate(cells[p]) for p in P}, pix, det


def saliency(run_dir, doc, frames, idx):
    R = ES.Run(f"{run_dir}/final.pt")
    m = R.model
    vis_in = m.inputs.visual(frames)
    text = m.inputs.text([build_prompt(doc)], vis_in["image_grid_thw"], padding_side="left")
    batch = {k: v.cuda() for k, v in {**vis_in, **text}.items()}
    gt, alts = ES.targets(m, doc)
    with torch.no_grad():
        feats = m.qwen.model.get_image_features(batch["pixel_values"], batch["image_grid_thw"], return_dict=True).pooler_output
    leaves = tuple(f.detach().requires_grad_(True) for f in feats)
    U = R.geometry(batch) if R.arm != "lora" else {}
    lg = R.logits(batch, leaves)
    mv = R.margin(lg, gt, alts)
    mv.backward()
    V, Vg = torch.cat(leaves), torch.cat([l.grad for l in leaves])
    N = len(V)
    tok = m.processor.tokenizer
    rec = {"arm": R.arm, "N": N, "margin": mv.item(), "top_token": tok.decode(lg.argmax().item()), "gt_token": tok.decode(gt)}
    maps = {"vis": (V.detach().float() * Vg.float()).norm(dim=-1).cpu().numpy()}
    attn = {}
    for k in R.keys:
        if R.arm == "additive":
            d, dg = m._geo[k].detach().float(), m._geo[k].grad.float()
        else:
            d, dg = R.delta[k].detach()[0].float(), R.delta[k].grad[0].float()
        maps[f"dest_{k}"] = (d * dg).norm(dim=-1).cpu().numpy()
        if R.arm == "xattn_aligned":
            fl, a = R.flow(k, dg, N)
            blk = m.aligned[k]
            y, g, pos, pts = R.cap[k]
            with torch.no_grad(), ES.AC():
                b = blk.bias(pos, torch.float, pts)[0]                      # [H, N, N]
                a0 = b.softmax(-1)                                          # attention under the 3D bias alone
                q, kk, _ = blk.project(y.detach(), g.detach(), pos, pts)
                rows = torch.randperm(N, device=b.device)[:512]
                qk = (q[0][:, rows].float() @ kk[0].float().transpose(-1, -2)) / math.sqrt(blk.head_dim)
                br = b[:, rows]
                fin = torch.isfinite(br)
                scale = {"qk_logit_std": qk.std().item(), "bias_std": br[fin].std().item(),
                         "nu": [round(x, 3) for x in blk.nu.float().tolist()] if hasattr(blk, "nu") else None}
            attn[k] = {"a": a.mean(0).float().cpu().numpy(), "a0": a0.mean(0).float().cpu().numpy(), "flow": fl, "scale": scale}
            del a, a0, b
            torch.cuda.empty_cache()
    rec["grid"] = batch["image_grid_thw"].cpu().numpy().tolist()
    del R, m
    torch.cuda.empty_cache()
    return rec, maps, attn


def frames_of(T, n_per):
    return np.repeat(np.arange(T), n_per)


def enrichment(s, f, fr):
    """Share of map s on object cells (weights f), over the frames where the object is detected, / share of cells."""
    present = np.unique(fr[f > 0])
    if len(present) == 0:
        return None
    keep = np.isin(fr, present)
    s, f = s[keep], f[keep]
    share, area = (s * f).sum() / s.sum(), f.sum() / len(f)
    return {"enrichment": round(float(share / area), 3), "share": round(float(share), 4), "area": round(float(area), 4),
            "frames": int(len(present))}


def obj_attention(A, fx, fy):
    """Mean over X's cells (weighted) of the attention mass on Y's cells (weighted)."""
    wx = fx / fx.sum()
    return float(wx @ (A @ fy))


def figure(path, frames, pix, maps_by_run, grids, phrases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = list(phrases)
    T = len(frames)
    n_obj = np.array([sum(pix[p][t].any() for p in P) for t in range(T)])
    show = sorted(np.argsort(-n_obj, kind="stable")[:4].tolist())
    cols = 1 + sum(2 for _ in maps_by_run)
    fig, ax = plt.subplots(len(show), cols, figsize=(3.2 * cols, 2.6 * len(show)), squeeze=False)
    cmap = plt.get_cmap("tab10")
    offs = np.cumsum([0] + [(g[1] // 2) * (g[2] // 2) for g in grids])
    for r, t in enumerate(show):
        im = np.asarray(frames[t]).astype(float) / 255
        over = im.copy()
        for j, p in enumerate(P):
            mk = pix[p][t]
            over[mk] = 0.45 * over[mk] + 0.55 * np.array(cmap(j)[:3])
        ax[r, 0].imshow(over); ax[r, 0].set_title(f"frame {t}", fontsize=9)
        c = 1
        for name, maps in maps_by_run.items():
            h, w = grids[t][1] // 2, grids[t][2] // 2
            for key in ("vis", "dest"):
                v = maps["vis"] if key == "vis" else sum(maps[k] for k in maps if k.startswith("dest_"))
                v = v / v.max()
                ax[r, c].imshow(im, alpha=0.35)
                ax[r, c].imshow(v[offs[t]:offs[t + 1]].reshape(h, w), cmap="magma", alpha=0.75, vmin=0, vmax=0.5,
                                extent=(0, im.shape[1], im.shape[0], 0))
                ax[r, c].set_title(f"{name}\n{'Qwen visual (vis)' if key == 'vis' else 'geometry injection (dest)'}", fontsize=8)
                c += 1
    for a in ax.flat:
        a.set_xticks([]); a.set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(j)) for j in range(len(P))]
    fig.legend(handles, [f"{p} ({phrases[p]})" for p in P], loc="lower center", ncol=len(P), fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, default=3004)
    ap.add_argument("--runs", nargs="+", default=["2B_v9_fmix_additive_s0", "2B_v10_xa3_qknu_s0"])
    ap.add_argument("--step", type=int, default=5750)
    ap.add_argument("--thr", type=float, default=0.25, help="OWLv2 score threshold")
    ap.add_argument("--max-inst", type=int, default=3, help="boxes kept per phrase per frame")
    ap.add_argument("--masker", choices=["owl_sam2", "sam3"], default="owl_sam2")
    ap.add_argument("--out", default="results/objsal_test")
    args = ap.parse_args()
    out = f"{args.out}/{args.id}"
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    doc = dict(load_vsibench("full").set_index("id", drop=False).loc[args.id])
    phrases = phrases_of(doc)
    frames, idx = uniform_frames(doc["video_path"], 32)
    print(f"question {args.id} ({doc['question_type']}): {doc['question']}\nphrases {phrases}, gt {doc['ground_truth']}", flush=True)

    # the grid each frame gets from Qwen's processor (the same processor every checkpoint uses)
    from transformers import AutoProcessor
    from mvp.fusion.inputs import EVAL_PIXELS, InputBuilder
    model_id = json.load(open(f"results/train/{args.runs[0]}/args.json")).get("model", "Qwen/Qwen3.5-2B")
    grids = InputBuilder(AutoProcessor.from_pretrained(model_id, **EVAL_PIXELS)).visual(frames)["image_grid_thw"].tolist()

    masker = object_masks_sam3 if args.masker == "sam3" else object_masks
    cells, pix, det = masker(frames, phrases, grids, args.thr, args.max_inst, "cuda")
    np.savez_compressed(f"{out}/masks_pix.npz", **{p: np.stack(v)[:, ::4, ::4] for p, v in pix.items()})
    print(f"masks done {time.time() - t0:.0f}s", flush=True)
    n_per = [(g[1] // 2) * (g[2] // 2) for g in grids]
    fr = np.concatenate([np.full(n, t) for t, n in enumerate(n_per)])

    report = {"masker": args.masker, "thr": args.thr, "id": args.id, "question_type": doc["question_type"], "question": doc["question"], "ground_truth": doc["ground_truth"],
              "options": list(doc["options"]) if doc.get("options") is not None else None, "frame_idx": idx, "grids": grids,
              "masks": {p: {"role": phrases[p], "frames_detected": int(sum(m.any() for m in pix[p])),
                            "mean_cells_covered": round(float(cells[p].reshape(-1).sum() / max(1, sum(m.any() for m in pix[p]))), 2),
                            "detections": {t: d for t, d in enumerate(det[p]) if d}} for p in phrases},
              "models": {}}
    maps_by_run = {}
    for run in args.runs:
        rec, maps, attn = saliency(f"results/train/{run}/branches/step{args.step}", doc, frames, idx)
        print(f"{run}: {rec['arm']} margin {rec['margin']:.2f} top {rec['top_token']!r} gt {rec['gt_token']!r} {time.time() - t0:.0f}s", flush=True)
        dest = sum(maps[k] for k in maps if k.startswith("dest_"))
        rec["enrichment"] = {name: {p: enrichment(v, cells[p], fr) for p in phrases}
                             for name, v in {**maps, "dest_sum": dest}.items()}
        if attn:
            rec["object_attention"] = {}
            for k, A in attn.items():
                tab = {}
                for px in phrases:
                    for py in phrases:
                        if px == py or cells[px].sum() == 0 or cells[py].sum() == 0:
                            continue
                        tab[f"{px} -> {py}"] = {"learned": round(obj_attention(A["a"], cells[px], cells[py]), 4),
                                                "bias_only": round(obj_attention(A["a0"], cells[px], cells[py]), 4),
                                                "area_y": round(float(cells[py].sum() / len(fr)), 4)}
                rec["object_attention"][k] = {"pairs": tab, "scale": A["scale"],
                                              "flow_mass": {s: round(A["flow"][f"mass_{s}"], 4) for s in ("self", "same_frame", "other_near", "other_far")}}
        report["models"][run] = rec
        maps_by_run[run.replace("2B_", "")] = maps
        np.savez_compressed(f"{out}/{run}_maps.npz", **maps)
    np.savez_compressed(f"{out}/masks.npz", **{p: c for p, c in cells.items()}, frame_of_cell=fr)
    json.dump(report, open(f"{out}/report.json", "w"), indent=1)
    try:
        figure(f"{out}/overview.png", frames, pix, maps_by_run, grids, phrases)
    except ImportError as e:
        print(f"no figure ({e}); the arrays are saved", flush=True)
    print(f"done {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
