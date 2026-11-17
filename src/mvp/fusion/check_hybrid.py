"""GPU checks of the hybrid arm (additive + AlignedXAttn on a shared merger trunk) before training, with the v7 options
of the aligned arm (pe="3d", 3D RoPE, 3D-distance bias). One GPU, about 10 minutes; a real 8-frame micro-batch.

  1. Structure: one merger per injection point with the additive head (mlp.2, zero) and the key / value head (xa,
     non-zero) on the shared trunk; trainable parameters = the xattn_aligned arm's + one 4096 x D head per point.
  2. Init loss == base model's (both added terms are zero).
  3. Additive path: an additive-arm model with non-zero merger output layers and LoRA, its weights copied into the
     hybrid (shared trunk + mlp.2) with W_o = 0 -> the same per-question loss.
  4. Cross-attention path: an xattn_aligned model with perturbed W_o, its merger copied into the trunk + xa head and
     its AlignedXAttn copied, additive head zero -> the same loss.
  5. Both paths on: the update at each injection point equals the sum of the two single-path updates (hook outputs).
  6. Gradients: at init they reach LoRA, the additive head and W_o only; the shared trunk, xa and W_q / W_k / W_v get
     none (both heads that read the trunk are zero, as in the additive arm's merger; W_o = 0, as in xattn_aligned);
     with both non-zero every new parameter (and nu) gets one.
  7. Generation, save/load round trip, time and peak memory of 8 questions fwd + bwd.

  slurm/submit.sh -J check_hybrid -t 00:40:00 -- python -m mvp.fusion.check_hybrid
"""
import argparse
import os
import tempfile
import time

import torch

from mvp.fusion.check import micro_batch
from mvp.fusion.check_aligned import ok
from mvp.fusion.inputs import TRAIN_PIXELS
from mvp.fusion.model import FusionVL, count_params
from mvp.train.data import to_device

DEV = "cuda"
XA3 = dict(rope3=True, local_bias=True, pe="3d", bias_nu=1.5, pe_dmax=10.0, pixels=TRAIN_PIXELS, grad_ckpt=True)


def build(model_id, arm, seed, **kw):
    torch.manual_seed(seed)
    return FusionVL(model_id, arm, **({**XA3, **kw} if arm in ("xattn_aligned", "hybrid") else
                                      dict(pixels=TRAIN_PIXELS, grad_ckpt=True, **kw))).cuda().train()


def loss_of(model, batch):
    """Per-question loss in eval mode: LoRA dropout off, so models with the same weights give the same loss."""
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            return model(batch)
    finally:
        model.train(was)


def copy_(dst, src):
    with torch.no_grad():
        dst.copy_(src)


def grads(model, names):
    gn = {}
    for n, p in model.named_parameters():
        for key in names:
            if key in n and p.grad is not None:
                gn[key] = gn.get(key, 0) + p.grad.float().norm().item()
    model.zero_grad(set_to_none=True)
    return gn


def updates(model, batch):
    """{k: the hook's added update at injection point k}, from the layer outputs with and without the hook."""
    out = {}
    layers = model.qwen.model.language_model.layers
    hooks = [layers[k].register_forward_hook(lambda mod, a, o, k=k: out.__setitem__(str(k), o[0] if isinstance(o, tuple) else o))
             for k in model.pairs]
    try:
        loss_of(model, batch)
    finally:
        for h in hooks:
            h.remove()
    return out  # hooks run in registration order: the fusion hook (registered first) has already added its update


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    args = ap.parse_args()

    base = FusionVL(args.model, "base", pixels=TRAIN_PIXELS).cuda()
    batch = micro_batch(base, 3)
    big = micro_batch(base, 8)
    ref = loss_of(base, batch)
    del base
    torch.cuda.empty_cache()

    # 1, 2
    hyb = build(args.model, "hybrid", 0)
    cp = count_params(hyb)
    print("hybrid trainable", {k: f"{v / 1e6:.2f}M" for k, v in cp.items()}, flush=True)
    m0 = hyb.mergers["0"]
    D = hyb.qwen.config.text_config.hidden_size
    ok(m0.xa is not None and m0.xa.weight.shape == m0.mlp[2].weight.shape == (D, 4096)
       and m0.mlp[2].weight.abs().max() == 0 and m0.xa.weight.abs().max() > 0
       and all(b.o.weight.abs().max() == 0 for b in hyb.aligned.values()),
       "additive head zero, key / value head non-zero, W_o zero")
    loss = loss_of(hyb, batch)
    ok(torch.allclose(loss, ref, atol=1e-4), f"init loss == base {loss.tolist()} vs {ref.tolist()}")

    # 6, at init
    hyb(batch).sum().backward()
    gn = grads(hyb, ["lora_", "mergers.0.mlp.0.weight", "mergers.2.mlp.2.weight", "mergers.1.xa.weight",
                     "aligned.0.q.weight", "aligned.1.k.weight", "aligned.2.v.weight", "aligned.2.o.weight"])
    ok(all(gn.get(k, 0) > 0 for k in ["lora_", "mergers.2.mlp.2.weight", "aligned.2.o.weight"])
       and all(gn.get(k, 0) == 0 for k in ["mergers.0.mlp.0.weight", "mergers.1.xa.weight", "aligned.0.q.weight",
                                           "aligned.1.k.weight", "aligned.2.v.weight"]),
       "gradients at init: " + ", ".join(f"{k} {v:.2e}" for k, v in gn.items()))

    # 3. additive path
    add = build(args.model, "additive", 1)
    with torch.no_grad():
        for mg in add.mergers.values():
            torch.nn.init.normal_(mg.mlp[2].weight, std=0.02)
        for n, p in add.named_parameters():
            if "lora_B" in n:
                torch.nn.init.normal_(p, std=0.01)
    ref_add = loss_of(add, batch)
    state = {n: v for n, v in add.trainable_state().items()}
    missing, unexpected = hyb.load_state_dict(state, strict=False)
    ok(not unexpected and not any("mergers" in n and ".xa." not in n for n in missing),
       "additive state (LoRA, merger trunk and output layer) loads into the hybrid")
    loss = loss_of(hyb, batch)
    ok(torch.allclose(loss, ref_add, atol=2e-3) and (ref_add - ref).abs().max() > 1e-2,
       f"additive path == additive arm {loss.tolist()} vs {ref_add.tolist()} (base {ref.tolist()})")
    del add
    torch.cuda.empty_cache()

    # 4. cross-attention path
    xa = build(args.model, "xattn_aligned", 2)
    with torch.no_grad():
        for b in xa.aligned.values():
            torch.nn.init.normal_(b.o.weight, std=0.02)
    ref_xa = loss_of(xa, batch)
    for k in hyb.mergers:
        mh, mx = hyb.mergers[k], xa.mergers[k]
        copy_(mh.ln.weight, mx.ln.weight)
        copy_(mh.mlp[0].weight, mx.mlp[0].weight); copy_(mh.mlp[0].bias, mx.mlp[0].bias)
        copy_(mh.xa.weight, mx.mlp[2].weight); copy_(mh.xa.bias, mx.mlp[2].bias)
        mh.mlp[2].weight.data.zero_(); mh.mlp[2].bias.data.zero_()
        hyb.aligned[k].load_state_dict(xa.aligned[k].state_dict())
    lora = {n: v for n, v in xa.trainable_state().items() if "lora_" in n}
    hyb.load_state_dict(lora, strict=False)
    loss = loss_of(hyb, batch)
    ok(torch.allclose(loss, ref_xa, atol=2e-3) and (ref_xa - ref).abs().max() > 1e-3,
       f"cross-attention path == xattn_aligned arm {loss.tolist()} vs {ref_xa.tolist()} (base {ref.tolist()})")
    del xa
    torch.cuda.empty_cache()

    # 5. both on: at layer 0 the hook's input h is the same whatever the fusion weights, so with the layer-0 output
    #    of every combination of the two heads, (both) - (neither) must equal (add only - neither) + (xa only - neither)
    with torch.no_grad():
        for mg in hyb.mergers.values():
            torch.nn.init.normal_(mg.mlp[2].weight, std=0.02)
    heads = {k: (hyb.mergers[k].mlp[2].weight.detach().clone(), hyb.aligned[k].o.weight.detach().clone()) for k in hyb.mergers}

    def layer0(use_add, use_xa):
        with torch.no_grad():
            for k, (wa, wo) in heads.items():
                hyb.mergers[k].mlp[2].weight.copy_(wa if use_add else torch.zeros_like(wa))
                hyb.aligned[k].o.weight.copy_(wo if use_xa else torch.zeros_like(wo))
        return updates(hyb, batch)["0"].float()

    none, a, x, both = layer0(False, False), layer0(True, False), layer0(False, True), layer0(True, True)
    mv = hyb._visual_mask
    dn = lambda t: t[mv]
    rel = ((dn(both) - dn(none)) - (dn(a) - dn(none)) - (dn(x) - dn(none))).norm() / (dn(both) - dn(none)).norm()
    ok(rel.item() < 1e-2 and (dn(a) - dn(none)).norm() > 0 and (dn(x) - dn(none)).norm() > 0
       and (both - none)[~mv].abs().max() == 0,
       f"layer 0: update(both) == update(add) + update(xa), rel. diff {rel.item():.2e}; text tokens untouched")

    # 6. with W_o perturbed (now) every new parameter gets a gradient
    hyb(batch).sum().backward()
    names = ["lora_", "mergers.0.mlp.0.weight", "mergers.2.mlp.2.weight", "mergers.1.xa.weight", "aligned.0.q.weight",
             "aligned.1.k.weight", "aligned.2.v.weight", "aligned.2.o.weight", "aligned.0.nu"]
    gn = grads(hyb, names)
    ok(all(gn.get(k, 0) > 0 for k in names), "gradients reach " + ", ".join(f"{k} {gn.get(k, 0):.2e}" for k in names))
    ok(hyb.bias_slopes() == {k: {"nu": 1.5} for k in "012"}, f"bias_slopes {hyb.bias_slopes()}")

    # 7.
    hyb.eval()
    gb = {k: batch[k] for k in ("pixel_values", "image_grid_thw", "vggt_images")}
    qs = ["How many chairs are in this room?", "What is the size of the room in square meters?", "Is the door left of the bed?"]
    gb.update(to_device(hyb.inputs.text(qs, batch["image_grid_thw"], padding_side="left"), DEV))
    print(f"     generate: {hyb.generate(gb)}", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.pt")
        hyb.save(path)
        l1 = loss_of(hyb, batch)
        loaded = FusionVL.load(path, pixels=TRAIN_PIXELS).cuda().eval()
        ok(loaded.config == hyb.config and loaded.arm == "hybrid" and loaded.mergers["0"].xa is not None,
           f"loaded config {loaded.config}")
        l2 = loss_of(loaded, batch)
        ok(torch.allclose(l1, l2, atol=1e-4), f"save/load round trip: same loss {l1.tolist()}")
        del loaded
    hyb.train()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t = time.time()
    hyb(big).sum().backward()
    torch.cuda.synchronize()
    print(f"     8 questions fwd+bwd: {time.time() - t:.2f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
    print("ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
