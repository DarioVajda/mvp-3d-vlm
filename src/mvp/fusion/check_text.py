"""GPU checks of text queries in AlignedXAttn (FusionVL(..., text_queries=True), run.py --text-queries) before
training, on the v10 qknu recipe (pe="3d", spread nu). One GPU, about 15 minutes; real 8-frame micro-batches.

  1. Module: AlignedXAttn.text == the formula written out in fp32 (W_q LN(x), rotated pairs zeroed, softmax over the
     keys with no bias, W_v, W_o); it does not depend on the keys' positions; the visual path with precomputed keys
     (kv) == without.
  2. Init: the same parameters (names and values, same seed) as without text queries, and the same loss.
  3. Rows (trained checkpoint, layer-0 update): text tokens get an update, padding none, and the visual tokens' update
     is the same as without text queries; gradients reach W_q, W_k, W_v, W_o.
  4. Generation, on a left-padded batch: the geometry keys / values are computed once per injection point (in the
     prefill), not per decoded token; with the cache off the logits are the same; the decoding steps do use the text
     queries (their logits differ from a run without them); and the logits of every decoded token match a
     teacher-forced forward over prompt + answer (the training computation) as closely as without text queries,
     while a run whose decoding steps skip the text queries (negative control) does not.
  5. Save / load round trip; time and peak memory of 8 questions fwd + bwd and of generation, with and without.

  slurm/submit.sh -J check_text -t 00:40:00 -- python -m mvp.fusion.check_text
"""
import argparse
import math
import os
import tempfile
import time

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, LogitsProcessorList

from mvp.fusion.check import micro_batch
from mvp.fusion.check_aligned import ok
from mvp.fusion.check_hybrid import grads, loss_of, updates
from mvp.fusion.inputs import TRAIN_PIXELS
from mvp.fusion.model import FusionVL, stop_tokens
from mvp.fusion.modules import AlignedXAttn
from mvp.fusion.points import ScenePoints
from mvp.train.data import to_device

DEV = "cuda"
QKNU = dict(rope3=True, local_bias=True, pe="3d", bias_nu=[0.1, 6.0], pe_dmax=10.0, pixels=TRAIN_PIXELS, grad_ckpt=True)
CKPT = "results/train/2B_v10_xa3_qknu_s0/branches/step5750/final.pt"
QS = ["How many chairs are in this room?", "What is the size of the room in square meters?",
      "Is the door to the left of the bed? Answer in one sentence."]


def module_checks():
    torch.manual_seed(0)
    D, H, N, M = 512, 4, 96, 7
    b = AlignedXAttn(D, H, pe="3d", nu=(0.1, 6.0)).to(DEV)
    torch.nn.init.normal_(b.o.weight, std=0.05)
    pos = AlignedXAttn.positions(torch.tensor([[1, 8, 12]] * 4), 2).to(DEV)  # 4 frames of 4 x 6 cells = 96
    pts = lambda: ScenePoints(torch.randn(N, 3, device=DEV) * 3, torch.zeros(N, dtype=torch.bool, device=DEV),
                              torch.rand(N, N, device=DEV) * 10, 1.0)
    g, x, y = torch.randn(N, D, device=DEV), torch.randn(M, D, device=DEV), torch.randn(2, N, D, device=DEV)
    pa, pb = pts(), pts()
    with torch.no_grad():
        out = b.text(x, b.keys(g, pos, pa))
        n_rot = 2 * len(b.rope_freq)
        q = b.q(b.norm_q(x)).view(M, H, -1).transpose(0, 1).clone()  # [H, M, d]
        q[..., :n_rot] = 0
        k, v = b.keys(g, pos, pa)
        att = (q @ k[0].transpose(-1, -2) / math.sqrt(b.head_dim)).softmax(-1)
        ref = b.o((att @ v[0]).transpose(0, 1).reshape(M, D))
        ok(torch.allclose(out, ref, atol=1e-5) and ref.abs().max() > 1e-3,
           f"text == formula (max diff {(out - ref).abs().max().item():.1e}), {n_rot} of {b.head_dim} dims zeroed")
        other = b.text(x, b.keys(g, pos.flip(0), pb))
        ok(torch.allclose(out, other, atol=1e-5), "text output does not depend on the keys' positions")
        k_pos = (b.keys(g, pos, pa)[0] - b.keys(g, pos, pb)[0]).abs().max()
        ok(k_pos > 1e-2, f"(while the keys themselves do: {k_pos.item():.2f})")
        a1, a2 = b(y, g, pos, pa), b(y, g, pos, pa, b.keys(g, pos, pa))
        ok(torch.equal(a1, a2), "visual path with precomputed kv == without")


def build(model_id, seed, **kw):
    torch.manual_seed(seed)
    return FusionVL(model_id, "xattn_aligned", **{**QKNU, **kw}).cuda().train()


def gen_batch(model, batch):
    gb = {k: batch[k] for k in ("pixel_values", "image_grid_thw", "vggt_images")}
    gb.update(to_device(model.inputs.text(QS, batch["image_grid_thw"], padding_side="left"), DEV))
    return gb


class Hook(LogitsProcessor):
    """Called once per generated token, after that step's forward: runs `fn` after the first (prefill) call."""
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def __call__(self, input_ids, scores):
        self.calls += 1
        if self.calls == 1:
            self.fn()
        return scores


def generate(model, gb, n_new, cache=True, after_prefill=None):
    """Greedy decoding of exactly n_new tokens: (tokens [B, n_new], logits [B, n_new, V] fp32), as FusionVL.generate."""
    model._geo = None
    model.set_geometry(gb)
    model._kv = {} if (model.text_queries and cache) else None
    procs = LogitsProcessorList([Hook(after_prefill)] if after_prefill else [])
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.qwen.generate(**model.llm_inputs(gb), max_new_tokens=n_new, min_new_tokens=n_new, do_sample=False,
                                      temperature=None, top_p=None, top_k=None, logits_processor=procs,
                                      return_dict_in_generate=True, output_logits=True,
                                      **stop_tokens(model.processor.tokenizer))
    finally:
        model._geo = model._kv = None
    return out.sequences[:, gb["input_ids"].shape[1]:], torch.stack(out.logits, 1).float()


def teacher_forced(model, gb, toks):
    """Logits of the decoded tokens from one forward over prompt + tokens (the training computation)."""
    B, P = gb["input_ids"].shape
    fb = dict(gb, input_ids=torch.cat([gb["input_ids"], toks], 1),
              attention_mask=torch.cat([gb["attention_mask"], torch.ones_like(toks)], 1),
              mm_token_type_ids=torch.cat([gb["mm_token_type_ids"], torch.zeros_like(toks).to(gb["mm_token_type_ids"])], 1))
    model._geo = None
    model.set_geometry(fb)
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hs = model.qwen.model(**model.llm_inputs(fb), use_cache=False).last_hidden_state
            return model.qwen.lm_head(hs[:, P - 1:-1]).float()
    finally:
        model._geo = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--ckpt", default=CKPT, help="a trained xattn_aligned (pe=3d) checkpoint, for non-zero updates")
    ap.add_argument("--new", type=int, default=12, help="decoded tokens per question")
    args = ap.parse_args()
    module_checks()

    # 2. init
    plain, text = build(args.model, 0), build(args.model, 0, text_queries=True)
    sp, st = plain.state_dict(), text.state_dict()
    ok(sp.keys() == st.keys() and all(torch.equal(sp[n], st[n]) for n in sp),
       f"same parameters at init ({len(sp)} tensors, same values)")
    ok(text.config == {**plain.config, "text_queries": True} and "text_queries" not in plain.config,
       "config: text_queries only when on")
    batch = micro_batch(plain, 3)
    big = micro_batch(plain, 8)
    lp, lt = loss_of(plain, batch), loss_of(text, batch)
    ok(torch.equal(lp, lt), f"same loss at init {lt.tolist()}")
    del plain, text, sp, st
    torch.cuda.empty_cache()

    # 3. rows and gradients, trained weights
    plain = FusionVL.load(args.ckpt, pixels=TRAIN_PIXELS, grad_ckpt=True).cuda().train()
    text = FusionVL.load(args.ckpt, pixels=TRAIN_PIXELS, grad_ckpt=True, text_queries=True).cuda().train()
    ok(text.text_queries and not plain.text_queries and text.config["text_queries"], "checkpoint loads with text_queries")
    pad = batch["attention_mask"] == 0
    ok(pad.any(), f"the micro-batch has padding ({int(pad.sum())} positions)")
    lp, lt = loss_of(plain, batch), loss_of(text, batch)
    ok((lp - lt).abs().max() > 1e-3, f"loss changes with text queries: {lp.tolist()} -> {lt.tolist()}")
    up, ut = updates(plain, batch)["0"].float(), updates(text, batch)["0"].float()
    mv = text._visual_mask
    # without the hook: layer 0's input (the embeddings) is the same, so the update is the hook's own
    base0 = updates_without_hook(text, batch)
    dp, dt = up - base0, ut - base0
    txt = ~mv & ~pad
    ok(dt[txt].norm(dim=-1).min() > 0 and dp[txt].abs().max() == 0,
       f"every text token gets an update (min norm {dt[txt].norm(dim=-1).min().item():.2e}); none without text queries")
    ok(dt[pad].abs().max() == 0, "padding gets no update")
    ok(torch.equal(dt[mv], dp[mv]), "visual tokens: the same update as without text queries")
    text(batch).sum().backward()
    names = ["aligned.0.q.weight", "aligned.1.k.weight", "aligned.2.v.weight", "aligned.0.o.weight", "aligned.1.nu"]
    gn = grads(text, names)
    ok(all(gn.get(k, 0) > 0 for k in names), "gradients reach " + ", ".join(f"{k} {gn.get(k, 0):.2e}" for k in names))

    # 4. generation
    plain.eval(); text.eval()
    gb = gen_batch(text, batch)
    ok(bool((gb["attention_mask"] == 0).any()), "generation batch is left-padded")
    calls = []
    orig = AlignedXAttn.keys
    AlignedXAttn.keys = lambda self, *a, **k: calls.append(1) or orig(self, *a, **k)
    try:
        tok_t, log_t = generate(text, gb, args.new)
        n_cached = len(calls); calls.clear()
        tok_n, log_n = generate(text, gb, args.new, cache=False)
        n_uncached = len(calls)
    finally:
        AlignedXAttn.keys = orig
    L = len(text.pairs)
    ok(n_cached == L and n_uncached == L * args.new,
       f"keys / values computed {n_cached}x with the cache (once per injection point), {n_uncached}x without")
    ok(torch.equal(tok_t, tok_n) and (log_t - log_n).abs().max() < 1e-3,
       f"cache on == off: same tokens, max logit diff {(log_t - log_n).abs().max().item():.1e}")
    print("     decoded:", [text.processor.decode(t) for t in tok_t], flush=True)

    # negative control: the same model, its decoding steps without text queries
    def off():
        text.text_queries = False
    try:
        tok_c, log_c = generate(text, gb, args.new, after_prefill=off)
    finally:
        text.text_queries = True
    tok_p, log_p = generate(plain, gb, args.new)

    def mismatch(model, toks, logits):  # decoded logits vs teacher-forced logits of the same tokens
        tf = teacher_forced(model, gb, toks)
        return (tf - logits).abs().amax(-1), (tf.argmax(-1) == logits.argmax(-1)).float().mean().item()

    e_t, a_t = mismatch(text, tok_t, log_t)
    e_p, a_p = mismatch(plain, tok_p, log_p)
    e_c, a_c = mismatch(text, tok_c, log_c)
    print(f"     prefill logits (step 0) equal in the control: {torch.equal(log_c[:, 0], log_t[:, 0])}", flush=True)
    print(f"     max |decoded - teacher-forced| logit per step (over questions), argmax agreement:\n"
          f"       text queries   {[round(v, 3) for v in e_t.amax(0).tolist()]}  {a_t:.2f}\n"
          f"       without        {[round(v, 3) for v in e_p.amax(0).tolist()]}  {a_p:.2f}\n"
          f"       control        {[round(v, 3) for v in e_c[:, 1:].amax(0).tolist()]} (steps 1+)  {a_c:.2f}", flush=True)
    effect = (log_t[:, 1:] - log_c[:, 1:]).abs().amax().item()
    ok(effect > 5 * max(e_t.max().item(), e_p.max().item()),
       f"decoding steps use the text queries: logits move by up to {effect:.2f} without them at decoding, "
       f"> 5x the decoding-vs-forward mismatch ({e_t.max().item():.3f}; {e_p.max().item():.3f} without text queries)")
    ok(e_t.max() <= 2 * e_p.max() + 0.05 and a_t >= a_p - 0.05,
       "decoded logits == teacher-forced (the training computation) as closely as without text queries")
    ok(e_c[:, 1:].max() > 5 * e_t.max(), "negative control (decoding skips text queries) does not match the forward")

    # 5.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.pt")
        text.save(path)
        l1 = loss_of(text, batch)
        loaded = FusionVL.load(path, pixels=TRAIN_PIXELS).cuda().eval()
        ok(loaded.text_queries and loaded.config == text.config, "loaded config keeps text_queries")
        ok(torch.allclose(l1, loss_of(loaded, batch), atol=1e-4), f"save/load round trip: same loss {l1.tolist()}")
        del loaded
    for name, m in (("without", plain), ("text queries", text)):
        m.train()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t = time.time()
        m(big).sum().backward()
        torch.cuda.synchronize()
        tr, mem = time.time() - t, torch.cuda.max_memory_allocated() / 2**30
        m.zero_grad(set_to_none=True)
        m.eval()
        generate(m, gb, 2)  # warm-up
        torch.cuda.synchronize()
        t = time.time()
        generate(m, gb, 32)
        torch.cuda.synchronize()
        print(f"     {name}: 8 questions fwd+bwd {tr:.2f}s, peak {mem:.1f} GB; generate 3 x 32 tokens {time.time() - t:.2f}s",
              flush=True)
    print("ALL CHECKS PASSED", flush=True)


def updates_without_hook(model, batch):
    """Layer-0 output with the fusion hooks off (no geometry): the reference the hooks add their update to."""
    out = {}
    layer = model.qwen.model.language_model.layers[0]
    h = layer.register_forward_hook(lambda mod, a, o: out.__setitem__("0", o[0] if isinstance(o, tuple) else o))
    vggt, model.vggt = model.vggt, None  # forward() then sets no geometry, and the hooks return at once
    try:
        loss_of(model, batch)
    finally:
        model.vggt = vggt
        h.remove()
    return out["0"].float()


if __name__ == "__main__":
    main()
