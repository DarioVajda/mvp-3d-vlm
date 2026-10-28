"""Train one arm (docs/idea.md, "Arms") on VLM-3R ScanNet QA: LoRA on Qwen3.5 plus the arm's new modules.

  # one GPU
  slurm/submit.sh -J train_2B_xattn -t 12:00:00 -- python -m mvp.train.run --model Qwen/Qwen3.5-2B --arm xattn \
      --out results/train/2B_xattn
  # four GPUs: same steps as one GPU (each GPU runs mb_per_step / 4 micro-batches per step instead of all of them)
  slurm/submit.sh -J train_4B_xattn -t 12:00:00 --gres gpu:4 -c 64 --mem 512G -- \
      torchrun --standalone --nproc_per_node 4 -m mvp.train.run --model Qwen/Qwen3.5-4B --arm xattn --out ...
  # smoke test: --max-steps 50

Dev: --dev-videos whole videos are held out (never trained on); dev loss every --dev-every steps, dev loss and
VSI-style accuracy per task at step 0 and --evals-per-epoch times per epoch (each with a checkpoint), as rows with
dev_* keys in log.jsonl (no "loss" key). dev_acc (what select_best uses) is the mean over the six VLM-3R tasks only;
dev_acc/all also includes appearance order. --appearance adds VSI-590K's appearance-order questions to training
(data.load_appearance), with their own held-out dev videos.
Schedule: linear warmup over --warmup of all steps, then one cosine from --lr down to --min-lr * --lr at the last step.
Loss: each question's mean answer-token loss, averaged over the questions of the step (all GPUs), so every question
counts once whatever its answer length or micro-batch. Writes to --out: args.json, freeze.txt, log.jsonl (one line per
step: loss, lr, gates, grad norm, time, memory), ckpt_epoch<k>.pt and final.pt (config + trainable weights only).
The same per-step records go to Weights & Biases (rank 0 only; --wandb-entity, --wandb-project, --no-wandb).
The entity is the slolama team: the W&B account belongs only to that team and has no personal entity (wandb.init with
entity=vajdadario fails with "entity not found", tested 2026-10-26). wandb authenticates
from ~/.netrc (Slurm jobs forward HOME); no key is ever passed on the command line or written to the run directory.
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from mvp.fusion.inputs import TRAIN_PIXELS
from mvp.fusion.model import ARMS, FusionVL, count_params
from mvp.paths import ROOT
from mvp.train.data import MicroBatches, epoch_plan, load_appearance, load_vlm3r, rank_steps, score_answer, to_device


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True, choices=[a for a in ARMS if a != "base"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many steps (smoke tests)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--fusion-lr", type=float, default=None, help="lr of the new fusion modules (default: --lr)")
    ap.add_argument("--warmup", type=float, default=0.1, help="linear warmup, fraction of all steps")
    ap.add_argument("--min-lr", type=float, default=0.1, help="cosine floor, fraction of --lr")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--mb-per-step", type=int, default=4, help="micro-batches per optimizer step, over all GPUs")
    ap.add_argument("--max-q", type=int, default=8, help="max questions per micro-batch (all about one video)")
    ap.add_argument("--cap", type=int, default=3000, help="max questions per task")
    ap.add_argument("--nframes", type=int, default=8)
    ap.add_argument("--dev-videos", type=int, default=60, help="whole videos held out for dev (0: none)")
    ap.add_argument("--dev-per-task", type=int, default=100, help="max dev questions per task")
    ap.add_argument("--dev-every", type=int, default=200, help="dev loss every this many steps (and at every epoch end)")
    ap.add_argument("--evals-per-epoch", type=int, default=1, help="dev accuracy + checkpoint this many times per epoch")
    ap.add_argument("--appearance", action="store_true", help="add VSI-590K appearance-order questions (all 16k)")
    ap.add_argument("--ao-dev-videos", type=int, default=40, help="extra held-out appearance-order videos (non-ScanNet)")
    ap.add_argument("--ao-dev-q", type=int, default=150, help="max appearance-order dev questions")
    ap.add_argument("--ao-cap", type=int, default=None, help="max appearance-order train questions (smoke tests)")
    ap.add_argument("--resampler-layers", type=int, default=8, help="Perceiver resampler depth (xattn arm)")
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--vggt-layers", type=int, nargs="+", default=[11, 17, 23], help="VGGT layers, one per injection point")
    ap.add_argument("--fusion-after", type=int, nargs="+", default=[0, 1, 2], help="decoder layers after which to inject")
    ap.add_argument("--gate", choices=["none", "tanh"], default="none", help="xattn: zero-init outputs (none) or tanh gates")
    ap.add_argument("--special-tokens", action="store_true", help="xattn: also feed VGGT camera/register tokens")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb-entity", default="slolama", help="W&B team (the account has no personal entity)")
    ap.add_argument("--wandb-project", default="mvp-geometry-fusion")
    ap.add_argument("--wandb-name", default=None, help="default: the --out directory name")
    ap.add_argument("--no-wandb", action="store_true")
    return ap.parse_args()


def main():
    args = parse()
    W = int(os.environ.get("WORLD_SIZE", 1))
    rank, local = int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))
    if W > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        (out / "args.json").write_text(json.dumps({**vars(args), "world_size": W, "git_commit": commit,
                                                   "host": os.uname().nodename, "gpu": torch.cuda.get_device_name()}, indent=1))
        (out / "freeze.txt").write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)

    torch.manual_seed(args.seed)  # same init of the new modules on every rank (DDP also broadcasts rank 0's)
    model = FusionVL(args.model, args.arm, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                     vggt_layers=args.vggt_layers, fusion_after=args.fusion_after, gate=args.gate,
                     special_tokens=args.special_tokens, resampler_layers=args.resampler_layers, pixels=TRAIN_PIXELS,
                     grad_ckpt=not args.no_grad_ckpt).to(dev)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    if rank == 0:
        print("trainable parameters:", {k: f"{v / 1e6:.1f}M" for k, v in count_params(model).items()}, flush=True)
    ddp = DDP(model, device_ids=[local], broadcast_buffers=False) if W > 1 else model

    if args.dev_videos:
        samples, dev_set = load_vlm3r(args.cap, args.seed, args.dev_videos, args.dev_per_task)
    else:
        samples, dev_set = load_vlm3r(args.cap, args.seed), []
    if args.appearance:
        ao_train, ao_dev = load_appearance(args.seed, {s["video"] for s in dev_set}, args.ao_dev_videos, args.ao_dev_q)
        if args.ao_cap is not None:
            ao_train = random.Random(f"ao-cap-{args.seed}").sample(ao_train, min(args.ao_cap, len(ao_train)))
        samples, dev_set = samples + ao_train, dev_set + (ao_dev if args.dev_videos else [])
        if rank == 0:
            print(f"appearance order: {len(ao_train)} train, {len(ao_dev)} dev questions", flush=True)
    dev_mbs = epoch_plan(dev_set, 0, args.seed, args.max_q)[rank::W]  # fixed; each rank scores its share
    if rank == 0 and dev_set:
        (out / "dev_ids.txt").write_text("\n".join(str(s["id"]) for s in dev_set))
        print(f"dev: {len(dev_set)} questions on {len({s['video'] for s in dev_set})} held-out videos", flush=True)
    steps_per_epoch = len(rank_steps(epoch_plan(samples, 0, args.seed, args.max_q), args.mb_per_step, W, rank))
    total = args.epochs * steps_per_epoch if args.max_steps is None else min(args.max_steps, args.epochs * steps_per_epoch)
    warm = max(1, round(args.warmup * total))
    groups = []  # LoRA at lr, new modules at fusion_lr; no decay on biases and norm weights (1-d), as in HF's Trainer
    for is_lora, lr in ((True, args.lr), (False, args.fusion_lr or args.lr)):
        ps = [p for n, p in model.named_parameters() if p.requires_grad and ("lora_" in n) == is_lora]
        for decay in (True, False):
            sub = [p for p in ps if (p.ndim > 1) == decay]
            if sub:
                groups.append({"params": sub, "lr": lr, "weight_decay": args.weight_decay if decay else 0.0})
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    floor = args.min_lr
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else floor + (1 - floor) *
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    if rank == 0:
        print(f"{len(samples)} questions, {steps_per_epoch} steps/epoch, {total} steps, warmup {warm}, lr floor {args.lr * floor:.1e}, "
              f"{args.mb_per_step // W} micro-batches per GPU per step, world size {W}", flush=True)

    accum = args.mb_per_step // W
    step, t_start = 0, time.time()
    log = open(out / "log.jsonl", "a") if rank == 0 else None
    run = None
    if rank == 0 and not args.no_wandb:
        import wandb
        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=args.wandb_name or out.name, dir=str(out),
                         config={**json.loads((out / "args.json").read_text()), "params": count_params(model),
                                 "questions": len(samples), "steps_per_epoch": steps_per_epoch, "total_steps": total})
    def dev_eval(generate=False):
        """Mean per-question answer loss on dev (and VSI-style accuracy per task with generate=True), all ranks."""
        if not dev_mbs and W == 1:
            return {}
        model.eval()
        acc = {}
        tot = torch.zeros(2, device=dev)  # loss sum, questions
        with torch.no_grad():
            loader_dev = DataLoader(MicroBatches(dev_mbs, model.inputs, args.nframes), batch_size=None, num_workers=2)
            for qs, b in zip(dev_mbs, loader_dev):
                b = to_device(b, dev)
                per_q = model(b)
                tot += torch.tensor([per_q.sum().item(), len(per_q)], device=dev)
                if generate:
                    gb = {k: b[k] for k in ("pixel_values", "image_grid_thw", "vggt_images")}
                    gb.update(to_device(model.inputs.text([s["question"] for s in qs], b["image_grid_thw"],
                                                          padding_side="left"), dev))
                    for s, p in zip(qs, model.generate(gb)):
                        acc.setdefault(s["task"], []).append(score_answer(p, s["answer"]))
        model.train()
        if W > 1:
            dist.all_reduce(tot)
            gathered = [None] * W
            dist.all_gather_object(gathered, acc)
            acc = {}
            for g in gathered:
                for t, v in g.items():
                    acc.setdefault(t, []).extend(v)
        res = {"dev_loss": tot[0].item() / max(1, tot[1].item())}
        if generate:
            per = {t: 100 * sum(v) / len(v) for t, v in sorted(acc.items())}
            vlm3r = [v for t, v in per.items() if t != "appearance_order"]
            mean = sum(vlm3r) / len(vlm3r)  # macro average over the six VLM-3R (ScanNet) tasks: the selection metric
            # dev_acc is what select_best reads; dev_acc/mean puts the same number in wandb's dev_acc panel section
            res.update({f"dev_acc/{t}": v for t, v in per.items()}, **{"dev_acc/mean": mean,
                       "dev_acc/all": sum(per.values()) / len(per)}, dev_acc=mean)
        return res

    def log_dev(res, epoch, **extra):
        if rank == 0 and res:
            log.write(json.dumps({"step": step, "epoch": epoch, **extra, **res}) + "\n")
            log.flush()
            if run is not None:
                run.log(res, step=step)
            print(f"dev step {step}: " + " ".join(f"{k} {v:.4f}" for k, v in res.items()), flush=True)

    marks = [round(k * steps_per_epoch / args.evals_per_epoch) for k in range(args.evals_per_epoch)]

    def save_point(epoch, ep):
        name = f"ckpt_epoch{ep:g}.pt"
        if dev_set:
            log_dev(dev_eval(generate=True), epoch, epoch_frac=ep, ckpt=name)
        if rank == 0:
            model.save(out / name)

    if dev_set:
        log_dev(dev_eval(generate=True), 0)  # untrained reference
    for epoch in range(args.epochs):
        steps = rank_steps(epoch_plan(samples, epoch, args.seed, args.max_q), args.mb_per_step, W, rank)
        flat = [mb for mbs, _ in steps for mb in mbs]
        loader = iter(DataLoader(MicroBatches(flat, model.inputs, args.nframes), batch_size=None, shuffle=False,
                                 num_workers=args.workers, pin_memory=True, prefetch_factor=2 if args.workers else None))
        for _, n_q in steps:
            if step >= total:
                break
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats(dev)
            stats = torch.zeros(3, device=dev)  # loss sum, questions, tokens
            for j in range(accum):
                batch = to_device(next(loader), dev)
                with ddp.no_sync() if W > 1 and j < accum - 1 else nullcontext():
                    per_q = ddp(batch)
                    (per_q.sum() * W / n_q).backward()  # DDP averages over ranks, hence * W
                stats += torch.tensor([per_q.detach().sum().item(), len(per_q), batch["attention_mask"].sum().item()], device=dev)
            gnorm = torch.nn.utils.clip_grad_norm_(params, args.clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            if W > 1:
                dist.all_reduce(stats)
            step += 1
            if rank == 0:
                rec = {"step": step, "epoch": epoch, "loss": stats[0].item() / stats[1].item(), "lr": sched.get_last_lr()[0],
                       "grad_norm": gnorm.item(), "questions": int(stats[1].item()), "tokens": int(stats[2].item()),
                       "step_time": time.time() - t0, "peak_mem_gb": torch.cuda.max_memory_allocated(dev) / 2**30,
                       "elapsed": time.time() - t_start, "gates": model.gates(), "inject": dict(model.stats)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if run is not None:
                    flat = {k: v for k, v in rec.items() if k not in ("gates", "inject")}
                    flat.update({f"gates/{i}_{k}": v for i, g in rec["gates"].items() for k, v in g.items()})
                    flat.update({f"inject/{i}": v for i, v in rec["inject"].items()})
                    run.log(flat, step=step)
                if step % 10 == 0 or step <= 5:
                    g = rec["gates"]
                    gs = f" gates xattn {min(v['xattn'] for v in g.values()):+.4f}..{max(v['xattn'] for v in g.values()):+.4f}" if g else ""
                    gs += " inject " + " ".join(f"{i}:{v:.3f}" for i, v in rec["inject"].items()) if rec["inject"] else ""
                    print(f"step {step}/{total} ep {epoch} loss {rec['loss']:.4f} lr {rec['lr']:.2e} gnorm {rec['grad_norm']:.3f} "
                          f"q {rec['questions']} tok {rec['tokens']} {rec['step_time']:.2f}s mem {rec['peak_mem_gb']:.1f}G{gs}", flush=True)
            k = next((k for k in range(1, args.evals_per_epoch) if step == epoch * steps_per_epoch + marks[k]), None)
            if k is not None:  # mid-epoch accuracy point + checkpoint
                save_point(epoch, epoch + k / args.evals_per_epoch)
            elif dev_set and step % args.dev_every == 0 and step % steps_per_epoch != 0:
                log_dev(dev_eval(), epoch)
        if step >= (epoch + 1) * steps_per_epoch:
            save_point(epoch, epoch + 1)  # loss + accuracy at every epoch end
    if rank == 0:
        model.save(out / "final.pt")
        print(f"done: {step} steps in {(time.time() - t_start) / 3600:.2f} h", flush=True)
        if run is not None:
            run.finish()
    if W > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
