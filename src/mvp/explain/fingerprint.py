"""Fingerprints of every stage of one forward pass, to find where two machines diverge (Mac mps vs cluster cuda gave
different logit differences for the same checkpoint and question). No training.

Stages, in the order the forward pass computes them:
  env        package versions, Hugging Face snapshot revisions of Qwen3.5-2B and VGGT-1B
  frames     md5 of the decoded frames' bytes, mean pixel of each frame
  text       md5 of the prompt text and of the token ids, T
  pixels     Qwen pixel_values and the VGGT input images
  vision     Qwen vision-encoder output (image embeddings)
  vggt       the 3D points per visual token position (fusion arms with pe="3d"); fusion module outputs per layer
  layers     hidden state after each decoder layer: at the last prompt position, and mean |h| over visual positions
  logits     option-letter logits and the logit difference z_gt - logsumexp(z_alt)

  python -m mvp.explain.fingerprint --run 2B_v10_xa3_qknu_s0 --id 3584 --device cuda --out fp_cuda.json
  python -m mvp.explain.fingerprint --compare fp_mac.json fp_cuda.json
"""
import argparse, hashlib, importlib.metadata as md, json, re, sys, time
import numpy as np
import torch

import mvp.explain.attn_weights as AW  # routes the code's torch.autocast("cuda") to the device in use


def st(x):
    x = x.detach().float().cpu()
    return {"shape": list(x.shape), "sum": float(x.sum()), "absmean": float(x.abs().mean()),
            "head": [round(float(v), 5) for v in x.flatten()[:4]]}


def md5(b):
    return hashlib.md5(b).hexdigest()


def env():
    out = {}
    for p in ["torch", "torchvision", "transformers", "peft", "pillow", "numpy", "decord", "eva-decord",
              "flash-linear-attention", "causal-conv1d", "accelerate"]:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    from huggingface_hub import try_to_load_from_cache
    for repo in ["Qwen/Qwen3.5-2B", "facebook/VGGT-1B"]:
        p = try_to_load_from_cache(repo, "config.json")
        m = re.search(r"snapshots/([0-9a-f]+)/", str(p))
        out[f"rev {repo}"] = m.group(1) if m else str(p)
    out["device_name"] = (torch.cuda.get_device_name() if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    return out


def run(args):
    from mvp.benchmarks.vsibench import build_prompt, load_vsibench
    from mvp.fusion.inputs import uniform_frames
    from mvp.fusion.model import FusionVL
    from mvp.explain import ROOT
    from mvp.explain.explain_saliency import targets

    AW._DeviceAutocast.device = args.device
    fp = {"run": args.run, "id": args.id, "env": env()}
    m = FusionVL.load(f"{ROOT}/results/train/{args.run}/branches/step{args.step}/final.pt", grad_ckpt=False)
    m = m.to(args.device).eval().requires_grad_(False)
    doc = dict(load_vsibench("full").set_index("id", drop=False).loc[args.id])

    frames, idx = uniform_frames(doc["video_path"], 32)
    arr = [np.asarray(f) for f in frames]
    fp["frames"] = {"idx": idx, "size": list(frames[0].size), "md5": md5(b"".join(a.tobytes() for a in arr)),
                    "mean_pixel": [round(float(a.mean()), 3) for a in arr]}
    prompt = build_prompt(doc)
    vis_in = m.inputs.visual(frames)
    text = m.inputs.text([prompt], vis_in["image_grid_thw"], padding_side="left")
    fp["text"] = {"prompt_md5": md5(m.inputs.prompt_text(prompt, len(frames)).encode()),
                  "ids_md5": md5(text["input_ids"].numpy().astype(np.int64).tobytes()), "T": int(text["input_ids"].shape[1])}
    fp["pixels"] = {"pixel_values": st(vis_in["pixel_values"]), "grid": vis_in["image_grid_thw"][0].tolist(),
                    "vggt_images": st(vis_in["vggt_images"])}
    batch = {k: v.to(args.device) for k, v in {**vis_in, **text}.items()}
    visual = batch["input_ids"][0] == m.inputs.image_token_id

    layers = m.qwen.model.language_model.layers
    hs = {}

    def hook(i):
        def f(module, a, out):
            h = out[0] if isinstance(out, tuple) else out
            hs[i] = {"last": st(h[0, -1]), "visual_absmean": float(h[0][visual].float().abs().mean())}
        return f

    handles = [l.register_forward_hook(hook(i)) for i, l in enumerate(layers)]  # after FusionVL's own hooks: fused output
    gt, alts = targets(m, doc)
    t0 = time.time()
    with torch.no_grad():
        m._geo = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = m.qwen.model.get_image_features(batch["pixel_values"], batch["image_grid_thw"], return_dict=True).pooler_output
        fp["vision"] = st(torch.cat(list(feats)))
        if m.vggt is not None:
            m.set_geometry(batch)
            fp["vggt"] = {"geo": {k: st(v if torch.is_tensor(v) else v[0]) for k, v in m._geo.items()}}
            if m._pts is not None:
                p = m._pts
                fp["vggt"]["points"] = {"xyz": st(p.xyz), "low": int(p.low.sum()), "scale": float(p.scale),
                                        "dist": st(p.dist)}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = m.qwen.model(**m.llm_inputs(batch), use_cache=False).last_hidden_state
            lg = m.qwen.lm_head(h[0, -1]).float()
        m._geo = None
    for hd in handles:
        hd.remove()
    tok = m.processor.tokenizer
    fp["layers"] = {str(i): hs[i] for i in sorted(hs)}
    fp["logits"] = {"options": {tok.decode(t): round(float(lg[t]), 4) for t in [gt, *alts]},
                    "top": tok.decode(lg.argmax().item()), "logit_difference": float(lg[gt] - torch.logsumexp(lg[alts], 0)),
                    "seconds": round(time.time() - t0, 1)}
    json.dump(fp, open(args.out, "w"), indent=1)
    print(json.dumps({k: fp[k] for k in ("env", "text", "logits")}, indent=1))
    print("->", args.out)


def compare(a_path, b_path, tol=1e-2):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    print("env differences:")
    for k in sorted(set(a["env"]) | set(b["env"])):
        if a["env"].get(k) != b["env"].get(k):
            print(f"  {k}: {a['env'].get(k)}  vs  {b['env'].get(k)}")

    def walk(x, y, path):
        if isinstance(x, dict):
            for k in x:
                if k in y:
                    yield from walk(x[k], y[k], f"{path}/{k}")
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            yield path, x, y
        elif isinstance(x, str) or isinstance(x, list) and x and not isinstance(x[0], (int, float)):
            yield path, x, y

    first = None
    for path, x, y in walk({k: a[k] for k in a if k != "env"}, {k: b[k] for k in b if k != "env"}, ""):
        if isinstance(x, str):
            same = x == y
            note = "" if same else f"{x} vs {y}"
        else:
            rel = abs(x - y) / max(abs(x), abs(y), 1e-6)
            same = rel <= tol
            note = f"{x:.6g} vs {y:.6g} (rel {rel:.2e})"
        if not same:
            print(("FIRST DIVERGENCE " if first is None else "  ") + f"{path}: {note}")
            first = first or path
    print("no divergence above tolerance" if first is None else f"first divergence: {first}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
        raise SystemExit
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="2B_v10_xa3_qknu_s0")
    ap.add_argument("--step", type=int, default=5750)
    ap.add_argument("--id", type=int, default=3584)
    ap.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    ap.add_argument("--out", required=True)
    run(ap.parse_args())
