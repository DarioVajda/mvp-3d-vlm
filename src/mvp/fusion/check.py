"""GPU checks of FusionVL before training (one GPU, a few minutes):

  1. At init every arm computes the same per-question loss as the base model (zero-init output projections / mergers,
     LoRA B zero-init).
  2. Vision encoder run once per video and its output repeated for B questions == the standard per-question path.
  3. With the output projections / mergers perturbed, the loss changes and gradients reach LoRA and every new module.
  4. Generation runs (base arm), with the additive grid assertion exercised in the prefill.
  5. Trainable parameter counts, and time + peak memory of one micro-batch of 8 questions (fwd + bwd).

  slurm/submit.sh -J fusion_check -t 00:40:00 -- python -m mvp.fusion.check --model Qwen/Qwen3.5-2B
"""
import argparse
import gc
import time

import torch

from mvp.fusion.inputs import TRAIN_PIXELS
from mvp.fusion.model import FusionVL, count_params
from mvp.train.data import MicroBatches, epoch_plan, load_vlm3r, to_device


def micro_batch(model, n_q):
    mbs = [m for m in epoch_plan(load_vlm3r(), 0) if len(m) >= n_q]
    return to_device(MicroBatches([mbs[0][:n_q]], model.inputs)[0], "cuda")


def timed_step(model, batch):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    model(batch).sum().backward()
    torch.cuda.synchronize()
    return time.time() - t, torch.cuda.max_memory_allocated() / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    args = ap.parse_args()
    torch.manual_seed(0)
    ref, batch = None, None
    for arm in ["base", "lora", "additive", "xattn"]:
        model = FusionVL(args.model, arm, pixels=TRAIN_PIXELS).cuda()
        model.train()
        if batch is None:
            batch = micro_batch(model, 3)
            print("batch", {k: tuple(v.shape) for k, v in batch.items() if torch.is_tensor(v)}, flush=True)
        with torch.no_grad():
            loss = model(batch)
        print(f"[{arm}] trainable {({k: f'{v / 1e6:.1f}M' for k, v in count_params(model).items()})}", flush=True)
        if ref is None:
            ref = loss
            # 2. shared vision features == standard path (pixel values repeated per question)
            B = batch["input_ids"].shape[0]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                std = model.qwen.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                       mm_token_type_ids=batch["mm_token_type_ids"],
                                       pixel_values=batch["pixel_values"].repeat(B, 1),
                                       image_grid_thw=batch["image_grid_thw"].repeat(B, 1)).last_hidden_state
                ours = model.qwen.model(**model.llm_inputs(batch)).last_hidden_state
            m = batch["attention_mask"].bool()
            print(f"[{arm}] shared vision vs standard: max |dh| {(std - ours)[m].abs().max().item():.3e}", flush=True)
            # 4. generation
            gen = {k: v for k, v in batch.items()}
            gen.update(model.inputs.text(["These are frames of a video.\nHow many chair(s) are in this room?\n"
                                          "Please answer the question using a single word or phrase."] * 2,
                                         batch["image_grid_thw"], padding_side="left"))
            print(f"[{arm}] generate: {model.generate(to_device(gen, 'cuda'))}", flush=True)
        else:
            d = (loss - ref).abs().max().item()
            print(f"[{arm}] per-question loss {loss.tolist()} vs base {ref.tolist()}: max diff {d:.2e} "
                  f"{'OK' if d < 1e-3 else 'MISMATCH'}", flush=True)
        if arm in ("additive", "xattn"):
            # 3. open the fusion path and check that the loss moves and gradients arrive
            with torch.no_grad():
                for mods in model.fusion_modules().values():
                    last = mods[-1]  # block (xattn) or merger (additive)
                    if arm == "xattn":
                        torch.nn.init.normal_(last.xattn.o.weight, std=1e-3)
                        torch.nn.init.normal_(last.ffw[2].weight, std=1e-3)
                    else:
                        torch.nn.init.normal_(last.mlp[2].weight, std=1e-3)
            for p in model.parameters():
                p.grad = None
            loss2 = model(batch)
            loss2.sum().backward()
            groups = {}
            for n, p in model.named_parameters():
                if p.requires_grad:
                    g = "lora" if "lora_" in n else n.split(".")[0]
                    groups.setdefault(g, []).append(0.0 if p.grad is None else p.grad.norm().item())
            print(f"[{arm}] opened: loss {loss2.tolist()} (moved {(loss2 - ref).abs().max().item():.3f}); grad norm per group "
                  f"{({g: round(sum(x * x for x in v) ** 0.5, 5) for g, v in groups.items()})}; "
                  f"params without grad: {sum(1 for v in groups.values() for x in v if x == 0.0)}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # 5. time and memory of a full micro-batch (8 questions, 8 frames) for the heaviest arm
    for ckpt in (True, False):
        model = FusionVL(args.model, "xattn", pixels=TRAIN_PIXELS, grad_ckpt=ckpt).cuda()
        model.train()
        b8 = micro_batch(model, 8)
        try:
            timed_step(model, b8)  # warm-up (Triton autotuning)
            t, mem = timed_step(model, b8)
            print(f"[xattn, grad_ckpt={ckpt}] 8 questions, {int(b8['attention_mask'].sum())} tokens: fwd+bwd {t:.2f}s, "
                  f"peak {mem:.1f} GB", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"[xattn, grad_ckpt={ckpt}] OOM", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
