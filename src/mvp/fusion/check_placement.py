"""GPU checks of the v8 injection placements of xattn_aligned (slurm/launch_v8_placement.sh) before training, with the v7
3D options (pe="3d", 3D RoPE, 3D-distance bias). One GPU, about 10 minutes; a real 8-frame micro-batch. Per placement:

  1. Structure: one merger + AlignedXAttn per injection point, keyed by the decoder layer, fed by the intended VGGT
     layer; hooks on exactly those layers; trainable parameters = LoRA + n_points x the v7 arm's per-point count.
  2. Init loss == base model's (W_o = 0 at every point).
  3. Location: W_o perturbed at one point k only -> the outputs of layers < k are bit-identical to the all-zero model,
     layer k changes on visual tokens only (text untouched there), the loss changes.
  4. Gradients: at init they reach LoRA and every point's W_o only; with every W_o perturbed, every point's merger,
     W_q / W_k / W_v, W_o and nu get one.
  5. Save/load round trip keeps the placement and the loss.

  slurm/submit.sh -J check_placement -t 00:40:00 -- python -m mvp.fusion.check_placement
"""
import argparse
import os
import tempfile

import torch

from mvp.fusion.check import micro_batch
from mvp.fusion.check_aligned import ok
from mvp.fusion.check_hybrid import XA3, grads, loss_of
from mvp.fusion.inputs import TRAIN_PIXELS
from mvp.fusion.model import FusionVL, count_params

PLACEMENTS = {"v7": ((0, 1, 2), (11, 17, 23)),
              "deep": ((0, 1, 2, 6, 10, 14), (11, 17, 23, 11, 17, 23)),
              "late": ((6, 10, 14), (11, 17, 23))}


def build(model_id, after, vggt):
    torch.manual_seed(0)
    return FusionVL(model_id, "xattn_aligned", fusion_after=after, vggt_layers=vggt, **XA3).cuda().train()


def layer_outputs(model, batch):
    """[output of every decoder layer] (after the fusion hook, which is registered first)."""
    out = {}
    layers = model.qwen.model.language_model.layers
    hooks = [l.register_forward_hook(lambda mod, a, o, i=i: out.__setitem__(i, (o[0] if isinstance(o, tuple) else o).float().cpu()))
             for i, l in enumerate(layers)]
    try:
        loss = loss_of(model, batch)
    finally:
        for h in hooks:
            h.remove()
    return [out[i] for i in range(len(layers))], loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    args = ap.parse_args()

    base = FusionVL(args.model, "base", pixels=TRAIN_PIXELS).cuda()
    batch = micro_batch(base, 3)
    ref = loss_of(base, batch)
    del base
    torch.cuda.empty_cache()

    per_point = None
    for name, (after, vggt) in PLACEMENTS.items():
        print(f"== {name}: after {after}, VGGT {vggt}", flush=True)
        m = build(args.model, after, vggt)
        keys = [str(k) for k in after]
        cp = count_params(m)
        print("   trainable", {k: f"{v / 1e6:.2f}M" for k, v in cp.items()}, flush=True)
        n_fusion = sum(p.numel() for n, p in m.named_parameters() if p.requires_grad and "lora_" not in n)
        if name == "v7":
            per_point = n_fusion / 3
        hooked = [i for i, l in enumerate(m.qwen.model.language_model.layers) if l._forward_hooks]
        ok(m.pairs == dict(zip(after, vggt)) and list(m.mergers) == keys == list(m.aligned) and hooked == list(after)
           and n_fusion == per_point * len(after) and m.config["fusion_after"] == list(after),
           f"structure: pairs {m.pairs}, hooked layers {hooked}, fusion params {n_fusion / 1e6:.2f}M "
           f"= {len(after)} x {per_point / 1e6:.2f}M")
        ok(m.vggt.cached_layer_indices >= set(vggt), f"VGGT layers read {sorted(m.vggt.cached_layer_indices)}")

        loss = loss_of(m, batch)
        ok(torch.allclose(loss, ref, atol=1e-4), f"init loss == base {loss.tolist()} vs {ref.tolist()}")

        # 4, at init
        m(batch).sum().backward()
        names = ["lora_"] + [f"aligned.{k}.o.weight" for k in keys]
        zero = [f"mergers.{k}.mlp.0.weight" for k in keys] + [f"aligned.{k}.{w}.weight" for k in keys for w in "qkv"]
        gn = grads(m, names + zero)
        ok(all(gn.get(k, 0) > 0 for k in names) and all(gn.get(k, 0) == 0 for k in zero),
           "gradients at init only to LoRA and every W_o: " + ", ".join(f"{k} {gn.get(k, 0):.1e}" for k in names))

        # 3. location, one point at a time
        zeros, _ = layer_outputs(m, batch)
        again, _ = layer_outputs(m, batch)  # run-to-run noise of the kernels (0 if deterministic)
        noise = max((a - b).abs().max().item() for a, b in zip(zeros, again))
        print(f"   run-to-run noise of the layer outputs: {noise:.2e}", flush=True)
        mv = m._visual_mask.cpu()
        for k in after:
            o = m.aligned[str(k)].o
            with torch.no_grad():
                torch.nn.init.normal_(o.weight, std=0.02)
            outs, lk = layer_outputs(m, batch)
            with torch.no_grad():
                o.weight.zero_()
            before = max([(outs[i] - zeros[i]).abs().max().item() for i in range(k)], default=0.0)
            d = outs[k] - zeros[k]
            ok(before <= noise and d[mv].abs().max() > 100 * max(noise, 1e-6) and d[~mv].abs().max() <= noise
               and (lk - ref).abs().max() > 1e-4,
               f"point {k}: layers < {k} unchanged (max diff {before:.1e}), layer {k} changes on visual tokens only "
               f"(visual {d[mv].abs().max():.2e}, text {d[~mv].abs().max():.1e}), loss {lk.tolist()}")

        # 4, all W_o perturbed
        with torch.no_grad():
            for b in m.aligned.values():
                torch.nn.init.normal_(b.o.weight, std=0.02)
        m(batch).sum().backward()
        names = ["lora_"] + [f"{p}.{k}.{w}" for k in keys for p, w in
                             [("mergers", "mlp.0.weight"), ("mergers", "mlp.2.weight"), ("aligned", "q.weight"),
                              ("aligned", "k.weight"), ("aligned", "v.weight"), ("aligned", "o.weight"), ("aligned", "nu")]]
        gn = grads(m, names)
        ok(all(gn.get(k, 0) > 0 for k in names), f"gradients reach all {len(names)} groups")

        # 5
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            m.save(path)
            l1 = loss_of(m, batch)
            loaded = FusionVL.load(path, pixels=TRAIN_PIXELS).cuda()
            ok(loaded.pairs == m.pairs and list(loaded.aligned) == keys, f"loaded placement {loaded.pairs}")
            l2 = loss_of(loaded, batch)
            ok(torch.allclose(l1, l2, atol=1e-4), f"save/load round trip: same loss {l1.tolist()}")
            del loaded
        del m
        torch.cuda.empty_cache()
    print("ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
