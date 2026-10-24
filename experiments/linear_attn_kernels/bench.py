"""Speed and numerics of Qwen3.5's linear-attention layers with and without the optimised kernels
(flash-linear-attention for the gated delta rule, causal-conv1d for the short convolution).

  python experiments/linear_attn_kernels/bench.py --model Qwen/Qwen3.5-4B --out <dir>/kernels.json
  python experiments/linear_attn_kernels/bench.py --model Qwen/Qwen3.5-4B --no-kernels --out <dir>/torch.json

On one real 32-frame VSI-Bench video it times: prefill (batch 1 and 8), greedy decoding of 256 tokens, and a
training step (forward + backward through the frozen model, gradient checkpointing on, as with adapters). It also
saves the prefill logits so two runs can be compared (compare.py).
"""
import argparse
import inspect
import json
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
ap.add_argument("--no-kernels", action="store_true", help="hide fla and causal_conv1d: reference PyTorch path")
ap.add_argument("--nframes", type=int, default=32)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--out", required=True)
args = ap.parse_args()
if args.no_kernels:
    sys.modules["fla"] = sys.modules["causal_conv1d"] = None  # makes their imports fail inside transformers

import numpy as np
import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as q35

from mvp.benchmarks.vsibench import load_vsibench
from mvp.models.qwen_vl import QwenVL


def implementation(fn):
    """Which function transformers' kernel wrapper resolved to (closure variable of the wrapper)."""
    impl = inspect.getclosurevars(fn).nonlocals.get("implementation", fn)
    return f"{impl.__module__}.{impl.__qualname__}"


def timed(f, repeats):
    f()  # warm-up (Triton autotuning / compilation)
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        out = f()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)), out


kernels = {n: implementation(getattr(q35, n)) for n in
           ["torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update"]}
print(json.dumps(kernels, indent=1), flush=True)

m = QwenVL(args.model)
df = load_vsibench("full")
doc = df.iloc[0]
video, meta, vkw = m.load_video(doc["video_path"], nframes=args.nframes)
prompt = f"These are frames of a video.\n{doc['question']}\nPlease answer the question using a single word or phrase."


def inputs(bs):
    texts = [m.processor.apply_chat_template(m._messages(prompt, "video"), tokenize=False, add_generation_prompt=True,
                                             **m.template_kwargs)] * bs
    return m.processor(text=texts, videos=[video] * bs, video_metadata=[meta] * bs, do_resize=False, padding=True,
                       return_tensors="pt", **vkw).to("cuda")


res = {"model": args.model, "gpu": torch.cuda.get_device_name(0), "kernels": kernels, "torch": torch.__version__,
       "frames": list(video.shape)}
x1, x8 = inputs(1), inputs(8)
res["input_tokens"] = int(x1["input_ids"].shape[1])

with torch.inference_mode():
    res["prefill_bs1_s"], out = timed(lambda: m.model(**x1, logits_to_keep=64).logits, args.repeats)
    res["prefill_bs8_s"], _ = timed(lambda: m.model(**x8, logits_to_keep=1).logits, args.repeats)
    np.save(args.out.replace(".json", "_logits.npy"), out[0].float().cpu().numpy())
    gen = lambda: m.model.generate(**x1, max_new_tokens=256, min_new_tokens=256, do_sample=False,
                                   temperature=None, top_p=None, top_k=None)
    res["generate256_bs1_s"], g = timed(gen, 1)
    res["generated_text"] = m.processor.decode(g[0, x1["input_ids"].shape[1]:], skip_special_tokens=True)[:300]
    res["greedy16"] = m.processor.decode(m.model.generate(**x1, max_new_tokens=16, do_sample=False, temperature=None,
                                                          top_p=None, top_k=None)[0, x1["input_ids"].shape[1]:],
                                         skip_special_tokens=True)

# training step: frozen weights, gradients flow to the inputs (as they would to adapters), loss on the last 32 tokens
m.model.train()
m.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
m.model.enable_input_require_grads()
for p in m.model.parameters():
    p.requires_grad_(False)
labels = torch.full_like(x1["input_ids"], -100)
labels[:, -32:] = x1["input_ids"][:, -32:]


def step():
    loss = m.model(**x1, labels=labels).loss
    loss.backward()
    return loss.detach()


torch.cuda.reset_peak_memory_stats()
res["train_step_bs1_s"], loss = timed(step, args.repeats)
res["train_loss"] = float(loss)
res["train_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30

print(json.dumps(res, indent=1), flush=True)
with open(args.out, "w") as f:
    json.dump(res, f, indent=1)
