"""GPU checks of the xattn_aligned arm (AlignedXAttn) before training (one GPU, about 10 minutes):

  Module (random weights, no Qwen):
  1. RoPE3: q.k after rotation depends only on the (t~, r, c) offset, equal positions leave q.k unchanged, the
     rotation preserves norms and leaves the non-rotary half of each head untouched.
  2. Locality bias: mask entries equal -lam*(|dr|+|dc|) - mu*[t != t'].
  3. SDPA (bf16, with the bias mask) against an eager fp32 reference: outputs and gradients of lam, mu, W_q, W_k,
     W_v, W_o and the inputs.
  4. Special cases: zero W_o -> exactly zero update; W_q = W_k = 0 with a steep bias -> each token reads its own
     geometry token (the additive arm); W_q = W_k = 0 without bias -> every token reads the mean.
  5. Memory of a 32-frame forward (N = 9600) with the bias, 8 questions, no grad.
  FusionVL (Qwen3.5, a real 8-frame micro-batch of 3 questions), for the four rope3 x local_bias settings:
  6. Positions match the merger's token order and count; at init the loss equals the base model's.
  7. With W_o perturbed the loss changes and gradients reach LoRA, the mergers, W_q/W_k/W_v/W_o and lam/mu.
  8. Attention diagnostics, generation (left-padded prefill), and save/load round trip.
  9. Time and peak memory of one micro-batch of 8 questions (fwd + bwd).

  slurm/submit.sh -J check_aligned -t 00:40:00 -- python -m mvp.fusion.check_aligned
"""
import argparse
import math
import os
import tempfile
import time

import torch
import torch.nn.functional as F

from mvp.fusion.check import micro_batch
from mvp.fusion.inputs import TRAIN_PIXELS
from mvp.fusion.model import FusionVL, count_params
from mvp.fusion.modules import AlignedXAttn
from mvp.train.data import to_device

DEV = "cuda"


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        raise SystemExit(1)


def grid(T, h=30, w=40):  # image_grid_thw of T frames of h x w patches (640x480 -> 30 x 40)
    return torch.tensor([[1, h, w]] * T, device=DEV)


def reference(blk, y, g, pos):
    """Eager fp32 AlignedXAttn.forward."""
    B, N, D = y.shape
    H, d = blk.heads, blk.head_dim
    split = lambda t: t.view(t.shape[0], N, H, d).transpose(1, 2)
    gn = F.layer_norm(g.float(), (D,), blk.norm_kv.weight, blk.norm_kv.bias)[None]
    q = split(F.layer_norm(y.float(), (D,), blk.norm_q.weight, blk.norm_q.bias) @ blk.q.weight.T)
    k, v = split(gn @ blk.k.weight.T), split(gn @ blk.v.weight.T)
    if blk.rope:
        cos, sin = blk.rotary(pos)
        q, k = blk.apply_rotary(q, cos, sin), blk.apply_rotary(k, cos, sin)
    logits = q @ k.transpose(-1, -2) / math.sqrt(d)
    if blk.local_bias:
        logits = logits + blk.bias(pos, torch.float)
    out = logits.softmax(-1) @ v
    return out.transpose(1, 2).reshape(B, N, D) @ blk.o.weight.T


def module_checks():
    torch.manual_seed(0)
    D, H = 2048, 16
    blk = AlignedXAttn(D, H).to(DEV)
    d = blk.head_dim

    # 1. RoPE3
    pos = AlignedXAttn.positions(grid(8))
    ok(pos.shape == (8 * 15 * 20, 3) and pos[:, 1].max() == 14 and pos[:, 2].max() == 19, f"positions {tuple(pos.shape)}, r<15, c<20")
    ok(pos[87].tolist() == [0, 4, 7] and pos[600 + 87].tolist() == [2, 4, 7], "token 87 of frame 2 is n=687 at (2, 4, 7)")
    q, k = torch.randn(1, 1, 1, d, device=DEV), torch.randn(1, 1, 1, d, device=DEV)
    def dot(pq, pk):
        P = torch.tensor([pq, pk], device=DEV)
        cos, sin = blk.rotary(torch.cat([P, torch.tensor([[7, 0, 0]], device=DEV)]))  # T = 8 so t~ = t
        rq = blk.apply_rotary(q, cos[:1], sin[:1])
        rk = blk.apply_rotary(k, cos[1:2], sin[1:2])
        return (rq * rk).sum().item(), rq
    base, _ = dot([1, 2, 3], [4, 6, 9])
    shifted, _ = dot([2, 5, 1], [5, 9, 7])  # same offset (3, 4, 6)
    ok(abs(base - shifted) < 1e-3 * max(1, abs(base)), f"q.k depends only on the offset ({base:.4f} vs {shifted:.4f})")
    other, _ = dot([1, 2, 3], [4, 6, 10])
    ok(abs(base - other) > 1e-3, f"q.k changes with the offset ({base:.4f} vs {other:.4f})")
    same, rq = dot([3, 7, 11], [3, 7, 11])
    ok(abs(same - (q * k).sum().item()) < 1e-3, "equal positions leave q.k unchanged")
    ok(torch.allclose(rq.norm(), q.norm(), atol=1e-4) and torch.equal(rq[..., 64:], q[..., 64:]),
       "rotation preserves the norm and leaves dims 64-127 untouched")
    cos, _ = blk.rotary(AlignedXAttn.positions(grid(32)))
    t32 = blk.rotary(torch.tensor([[31, 0, 0]], device=DEV).repeat(2, 1))  # with T = 32, t = 31 -> t~ = 7
    t8 = blk.rotary(torch.tensor([[7, 0, 0]], device=DEV).repeat(2, 1))
    ok(torch.allclose(t32[0], t8[0]) and torch.allclose(t32[1], t8[1]), "t~ = 7 t / (T - 1): last of 32 frames = last of 8")

    # 2. bias
    m = blk.bias(pos, torch.float)[0]
    i, j = 600 + 87, 600 + 20 * 6 + 4  # (2,4,7) -> (2,6,4): distance 5, same frame
    k2 = 1200 + 87  # (4,4,7): other frame, same cell
    ok(abs(m[0, i, j].item() + 1.5 * 5) < 1e-5 and abs(m[0, i, k2].item() + 3.0) < 1e-5 and m[0, i, i].item() == 0,
       f"bias: same frame d=5 -> {m[0, i, j].item():.2f}, other frame same cell -> {m[0, i, k2].item():.2f}, own 0")

    # 3. SDPA vs eager reference
    for p in blk.o.parameters():
        torch.nn.init.normal_(p, std=0.02)
    with torch.no_grad():
        blk.lam.copy_(torch.linspace(0.5, 3, H, device=DEV))
        blk.mu.copy_(torch.linspace(1, 5, H, device=DEV))
    B, N = 2, pos.shape[0]
    y = torch.randn(B, N, D, device=DEV, requires_grad=True)
    g = torch.randn(N, D, device=DEV, requires_grad=True)
    ref = reference(blk, y, g, pos)
    gref = torch.autograd.grad(ref.square().sum(), [y, g, blk.lam, blk.mu, blk.q.weight, blk.k.weight, blk.v.weight, blk.o.weight])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = blk(y, g, pos)
    gout = torch.autograd.grad(out.float().square().sum(), [y, g, blk.lam, blk.mu, blk.q.weight, blk.k.weight, blk.v.weight, blk.o.weight])
    rel = lambda a, b: ((a.float() - b).norm() / b.norm()).item()
    ok(rel(out, ref) < 0.03, f"SDPA bf16 output vs eager fp32: rel err {rel(out, ref):.4f}")
    for name, a, b in zip(["y", "g", "lam", "mu", "W_q", "W_k", "W_v", "W_o"], gout, gref):
        ok(b.norm() > 0 and rel(a, b) < 0.05, f"grad {name}: rel err {rel(a, b):.4f}, |grad| {b.norm():.3e}")

    # 4. special cases
    blk0 = AlignedXAttn(D, H).to(DEV)
    with torch.no_grad():
        ok(blk0(y, g, pos).abs().max().item() == 0, "zero-initialised W_o: the update is exactly 0")
        for steep_bias in (True, False):
            b = AlignedXAttn(D, H, rope=True, local_bias=steep_bias).to(DEV)
            b.q.weight.zero_(); b.k.weight.zero_()
            b.v.weight.copy_(torch.eye(D)); b.o.weight.copy_(torch.eye(D))
            if steep_bias:
                b.lam.fill_(60.0); b.mu.fill_(60.0)
            o = b(y, g, pos)[0]
            gn = F.layer_norm(g, (D,))
            target = gn if steep_bias else gn.mean(0, keepdim=True).expand_as(gn)
            ok(rel(o, target) < 1e-3, ("steep bias, no content: token n reads geometry token n (additive arm)" if steep_bias
                                       else "no bias, no content: every token reads the mean geometry") + f", rel err {rel(o, target):.2e}")
            st = b.attn_stats(y, g, pos)
            if steep_bias:
                ok(st["self_mass"] > 0.999 and st["frame_mass"] > 0.999, f"attn_stats steep bias: {st}")
            else:
                ok(abs(st["self_mass"] - 1 / N) < 1e-6 and abs(st["frame_mass"] - 1 / 8) < 1e-4 and abs(st["entropy"] - math.log(N)) < 1e-3,
                   f"attn_stats uniform: self {st['self_mass']:.2e} frame {st['frame_mass']:.4f} entropy {st['entropy']:.3f} (log N {math.log(N):.3f})")
        st = AlignedXAttn(D, H).to(DEV).attn_stats(y, g, pos)
        print(f"     init lam 1.5, mu 3, random W_q, W_k: {st}", flush=True)

    # 5. 32 frames
    pos32 = AlignedXAttn.positions(grid(32))
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        before = torch.cuda.memory_allocated()
        blk(torch.randn(8, len(pos32), D, device=DEV, dtype=torch.bfloat16), torch.randn(len(pos32), D, device=DEV), pos32)
        torch.cuda.synchronize()
    print(f"     32 frames (N {len(pos32)}), 8 questions, one point: peak +{(torch.cuda.max_memory_allocated() - before) / 2**30:.1f} GB", flush=True)


def fusion_checks(model_id):
    torch.manual_seed(0)
    base = FusionVL(model_id, "base", pixels=TRAIN_PIXELS).cuda()
    batch = micro_batch(base, 3)
    with torch.no_grad():
        ref = base(batch)
    big = micro_batch(base, 8)
    del base
    torch.cuda.empty_cache()
    for rope3, bias in [(False, False), (True, False), (False, True), (True, True)]:
        tag = f"rope3={int(rope3)} bias={int(bias)}"
        torch.manual_seed(0)
        model = FusionVL(model_id, "xattn_aligned", rope3=rope3, local_bias=bias, pixels=TRAIN_PIXELS,
                         grad_ckpt=True).cuda()  # as in training
        model.train()
        new = {k: v for k, v in count_params(model).items() if k != "lora"}
        print(f"[{tag}] trainable {({k: f'{v / 1e6:.2f}M' for k, v in count_params(model).items()})}, new {sum(new.values()) / 1e6:.2f}M", flush=True)
        # 6.
        with torch.no_grad():
            loss = model(batch)
        n_vis = int((batch["input_ids"][0] == model.inputs.image_token_id).sum())
        ok(model._pos.shape == (n_vis, 3) and all(len(v) == n_vis for v in model._geo.values()),
           f"[{tag}] {n_vis} visual tokens = positions = geometry tokens per point")
        ok(torch.allclose(loss, ref, atol=1e-4), f"[{tag}] init loss == base {loss.tolist()} vs {ref.tolist()}")
        # 7.
        with torch.no_grad():
            for b in model.aligned.values():
                torch.nn.init.normal_(b.o.weight, std=0.02)
        loss = model(batch)
        ok((loss - ref).abs().max().item() > 1e-3, f"[{tag}] perturbed W_o changes the loss ({(loss - ref).abs().max().item():.4f})")
        loss.sum().backward()
        names = ["lora_", "mergers.0.mlp.0.weight", "mergers.2.mlp.2.weight", "aligned.0.q.weight", "aligned.1.k.weight",
                 "aligned.2.v.weight", "aligned.2.o.weight"] + (["aligned.0.lam", "aligned.2.mu"] if bias else [])
        gn = {}
        for n, p in model.named_parameters():
            for key in names:
                if key in n and p.grad is not None:
                    gn[key] = gn.get(key, 0) + p.grad.float().norm().item()
        ok(all(gn.get(k, 0) > 0 for k in names), f"[{tag}] gradients reach " + ", ".join(f"{k} {gn.get(k, 0):.2e}" for k in names))
        model.zero_grad(set_to_none=True)
        # 8.
        model.eval()
        model.collect_attn = True
        with torch.no_grad():
            model(batch)
        model.collect_attn = False
        ok(set(model.attn_stats) == {"0", "1", "2"}, f"[{tag}] attn stats {model.attn_stats['0']}")
        model.attn_stats = {}
        gb = {k: batch[k] for k in ("pixel_values", "image_grid_thw", "vggt_images")}
        qs = ["How many chairs are in this room?", "What is the size of the room in square meters?", "Is the door left of the bed?"]
        gb.update(to_device(model.inputs.text(qs, batch["image_grid_thw"], padding_side="left"), DEV))
        print(f"     [{tag}] generate: {model.generate(gb)}", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            model.save(path)
            with torch.no_grad():
                l1 = model(batch)
            loaded = FusionVL.load(path, pixels=TRAIN_PIXELS).cuda().eval()
            ok(loaded.config == model.config, f"[{tag}] loaded config {({k: loaded.config[k] for k in ('arm', 'rope3', 'local_bias', 'bias_lam', 'bias_mu', 'train_frames')})}")
            with torch.no_grad():
                l2 = loaded(batch)
            ok(torch.allclose(l1, l2, atol=1e-4), f"[{tag}] save/load round trip: same loss {l1.tolist()}")
            del loaded
        # 9.
        model.train()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t = time.time()
        model(big).sum().backward()
        torch.cuda.synchronize()
        print(f"     [{tag}] 8 questions fwd+bwd: {time.time() - t:.2f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        del model
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--module-only", action="store_true")
    args = ap.parse_args()
    module_checks()
    if not args.module_only:
        fusion_checks(args.model)
    print("ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
