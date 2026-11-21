"""Packed micro-batches (InputBuilder.pack) are the same computation as one micro-batch per media: per-question losses
and gradients of a packed batch of three media (two ScanNet videos at 8 frames, 3 and 1 questions, and a one-image
SPAR question at another resolution) match the three batches run one by one. Trainable weights get random values first,
so the geometry injection is far from its zero init and a row reading another video's geometry would show.

Each one-by-one reference holds its questions twice (losses and gradients from the first copy): a batch of one row runs
other GPU kernels, whose bf16 rounding moves a long video question's loss by up to a few percent once LoRA is non-zero
(it vanishes in fp32, and both are equally far from fp32), so a reference with one row would not be a like-for-like. For the same reason each reference is right-padded to the
packed batch's length (padding changes the kernels' shapes, not what a row attends to); how much the padding alone moves
a loss is printed for information. Gradients are summed in another order (one backward over all rows against one per
media, in batches of other shapes), so they are held to the rounding level of a change of shape alone: the references
again, padded 64 tokens longer.

  slurm/submit.sh -J check_pack -t 00:30:00 -- python -m mvp.fusion.check_pack
"""
import json

import torch
import torch.nn.functional as F

from mvp.fusion.inputs import TRAIN_PIXELS
from mvp.fusion.model import FusionVL
from mvp.paths import DATA
from mvp.train.data import decode, load_vlm3r, to_device

ARMS = {"lora": {}, "xattn_aligned": dict(pe="3d", rope3=True, local_bias=True, pe_dmax=10.0, bias_nu=(0.1, 6.0))}
failures = []


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        failures.append(msg)


def groups():
    rows = load_vlm3r(10**9, 0)
    by = {}
    for s in rows:
        by.setdefault(s["video"], []).append(s)
    a, b = sorted(by)[:2]
    spar = json.loads(open(DATA / "mix" / "sft_500k.jsonl").readline())
    assert spar["source"] == "spar" and len(spar["media"]) == 1, spar["id"]
    return [(8, by[a][:3]), (8, by[b][:1]), (8, [spar])]


def padded(model, part, L):
    pad = {"input_ids": model.inputs.tok.pad_token_id, "attention_mask": 0, "labels": -100, "mm_token_type_ids": 0}
    return {**part, **{k: F.pad(part[k], (0, L - part[k].shape[1]), value=v) for k, v in pad.items()}}


def run(model, batch, n=None):
    """Per-question losses of the first n rows (all for None), the gradient of their sum and the injection stats."""
    model.zero_grad(set_to_none=True)
    loss = model(batch)[:n]
    loss.sum().backward()
    grads = torch.cat([p.grad.flatten().float() for p in model.parameters() if p.requires_grad and p.grad is not None])
    return loss.detach().float(), grads, dict(model.stats)


def main():
    gs = groups()
    for arm, kw in ARMS.items():
        torch.manual_seed(0)
        model = FusionVL("Qwen/Qwen3.5-2B", arm, pixels=TRAIN_PIXELS, train_frames=8, **kw).cuda()
        model.eval()  # no LoRA dropout, so both runs compute the same function
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p) * max(p.float().std().item(), 1e-2))
        parts = [model.inputs.build(decode(qs[0], k)[0], [s["question"] for s in qs], [s["answer"] for s in qs])
                 for k, qs in gs]
        twice = [model.inputs.build(decode(qs[0], k)[0], [s["question"] for s in qs] * 2, [s["answer"] for s in qs] * 2)
                 for k, qs in gs]
        packed = model.inputs.pack(parts)
        L = packed["input_ids"].shape[1]
        sep = [run(model, to_device(padded(model, p, L), "cuda"), len(qs)) for p, (_, qs) in zip(twice, gs)]
        with torch.no_grad():
            unpadded = torch.cat([model(to_device(p, "cuda"))[:len(qs)].float() for p, (_, qs) in zip(twice, gs)])
        print(f"[{arm}] (info) padding alone moves the losses by up to "
              f"{((unpadded - torch.cat([l for l, _, _ in sep])).abs() / unpadded).max().item():.2e} (bf16 rounding)")
        ok(packed["rows"] == [3, 1, 1] and packed["n_images"] == [8, 8, 1], f"[{arm}] rows {packed['rows']}, images "
           f"{packed['n_images']}")
        loss_p, grad_p, stats = run(model, to_device(packed, "cuda"))
        loss_s = torch.cat([l for l, _, _ in sep])
        grad_s = sum(g for _, g, _ in sep)
        rel = ((loss_p - loss_s).abs() / loss_s.abs().clamp(min=1e-3)).max().item()
        ok(rel < 1e-3, f"[{arm}] per-question losses packed {loss_p.tolist()} vs one by one {loss_s.tolist()} "
           f"(max relative difference {rel:.2e})")
        grad_l = sum(run(model, to_device(padded(model, p, L + 64), "cuda"), len(qs))[1] for p, (_, qs) in zip(twice, gs))
        floor = ((grad_l - grad_s).norm() / grad_s.norm()).item()
        cos = torch.nn.functional.cosine_similarity(grad_p, grad_s, dim=0).item()
        nrel = ((grad_p - grad_s).norm() / grad_s.norm()).item()
        ok(cos > 0.9999 and nrel < max(2 * floor, 1e-3), f"[{arm}] gradients: cosine {cos:.6f}, relative difference "
           f"{nrel:.2e} (the references padded 64 tokens longer: {floor:.2e})")
        if arm != "lora":
            ok(min(stats.values()) > 0.05, f"[{arm}] the injection changes the hidden states (||added||/||h|| {stats})")
            # the test can see a mix-up: giving the two ScanNet videos each other's geometry (same frame grid) changes
            # their losses by more than the tolerance above
            swapped = {**packed, "vggt_images": [packed["vggt_images"][1], packed["vggt_images"][0], packed["vggt_images"][2]]}
            with torch.no_grad():
                loss_w = model(to_device(swapped, "cuda")).float()
            relw = ((loss_w - loss_p).abs() / loss_p.abs().clamp(min=1e-3))[:4].max().item()
            ok(relw > 1e-2, f"[{arm}] swapping the videos' geometry changes the losses (max relative change {relw:.2e})")
        del model
        torch.cuda.empty_cache()
    print("ALL PASSED" if not failures else f"{len(failures)} FAILED", flush=True)
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
