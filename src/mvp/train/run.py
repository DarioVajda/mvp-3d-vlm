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
--schedule wsd (warmup-stable-decay, for training-length studies): a trunk warms up linearly over --warmup-steps and
then stays at the peak lr for --stable-steps; it never decays. At every --branch-steps step it logs dev accuracy and
saves the full training state (state_step<N>.pt: trainable weights, optimizer, lr schedule, step, RNG). A branch,
  python -m mvp.train.run --branch-of results/train/<trunk> --branch-step N [--decay-steps 750]
reruns the trunk's arguments (its args.json) from that state, continuing the trunk's data order exactly, with the lr
decaying linearly to 0 over --decay-steps, then logs dev accuracy and saves final.pt into <trunk>/branches/step<N>/.
Branches never log to W&B (mvp.train.wsd_collect adds them to the trunk's run once everything has finished).
--vsi590k adds VSI-590K's questions of the six VSI-Bench tasks on all its video sources (data.load_vsi590k); --cap
then applies per task to the pooled VLM-3R + VSI-590K questions.
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
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from mvp.fusion.inputs import TRAIN_PIXELS
from mvp.fusion.model import ARMS, FusionVL, count_params
from mvp.paths import ROOT
from mvp.train.data import (MicroBatches, cap_per_task, epoch_plan, load_appearance, load_vlm3r, load_vsi590k,
                            rank_steps, score_answer, to_device)


def is_slope(name):
    return name.startswith("aligned.") and name.rsplit(".", 1)[-1] in ("lam", "mu", "nu")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--arm", choices=[a for a in ARMS if a != "base"])
    ap.add_argument("--out")
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
    ap.add_argument("--rope3", action=argparse.BooleanOptionalAction, default=True,
                    help="xattn_aligned: 3-axis RoPE on (frame, row, col) of queries and keys")
    ap.add_argument("--local-bias", action=argparse.BooleanOptionalAction, default=True,
                    help="xattn_aligned: learned locality bias -lam*(|dr|+|dc|) - mu*[other frame] on the logits")
    ap.add_argument("--bias-lam", type=float, default=1.5, help="xattn_aligned: initial lam (per head)")
    ap.add_argument("--bias-mu", type=float, default=3.0, help="xattn_aligned: initial mu (per head)")
    ap.add_argument("--pe", choices=["grid", "3d"], default="grid",
                    help="xattn_aligned: position signals on the merged grid (t, r, c), or 3D from VGGT's point head: RoPE "
                    "on (t, x, y, z) and a bias -nu*min(3D distance, --pe-dmax) in place of the grid/frame bias")
    ap.add_argument("--bias-nu", type=float, default=1.5, help="xattn_aligned, --pe 3d: initial nu per head, per cell")
    ap.add_argument("--pe-dmax", type=float, default=10.0, help="xattn_aligned, --pe 3d: distance clip, in cells")
    ap.add_argument("--bias-lr", type=float, default=1e-3, help="xattn_aligned: lr of the lam / mu / nu slopes (Adam moves a "
                    "parameter by about lr per step, so at the 1e-5 module lr they would stay at their init)")
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--vggt-layers", type=int, nargs="+", default=[11, 17, 23], help="VGGT layers, one per injection point")
    ap.add_argument("--fusion-after", type=int, nargs="+", default=[0, 1, 2], help="decoder layers after which to inject")
    ap.add_argument("--gate", choices=["none", "tanh"], default="none", help="xattn: zero-init outputs (none) or tanh gates")
    ap.add_argument("--special-tokens", action="store_true", help="xattn: also feed VGGT camera/register tokens")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0, help="data: train subsample, dev split; also init and order unless "
                    "--init-seed is given")
    ap.add_argument("--init-seed", type=int, default=None, help="init of the new modules and LoRA, dropout and the "
                    "training order, with the data and dev split of --seed fixed (seed-variance reruns; default: --seed)")
    ap.add_argument("--wandb-entity", default="slolama", help="W&B team (the account has no personal entity)")
    ap.add_argument("--wandb-project", default="mvp-geometry-fusion")
    ap.add_argument("--wandb-name", default=None, help="default: the --out directory name")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--vsi590k", action="store_true", help="add VSI-590K's six non-appearance VSI-Bench tasks (all "
                    "video sources); --cap then applies per task to VLM-3R + VSI-590K together")
    ap.add_argument("--schedule", choices=["cosine", "wsd"], default="cosine")
    ap.add_argument("--warmup-steps", type=int, default=750, help="wsd: linear warmup steps")
    ap.add_argument("--stable-steps", type=int, default=10000, help="wsd trunk: steps at the peak lr after warmup")
    ap.add_argument("--branch-steps", type=int, nargs="*", default=[], help="wsd trunk: global steps at which to log dev "
                    "accuracy and save the full training state for a decay branch")
    ap.add_argument("--eval-every", type=int, default=1000, help="wsd: dev accuracy every this many steps")
    ap.add_argument("--branch-of", default=None, help="wsd branch: trunk directory (its args.json is reused)")
    ap.add_argument("--branch-step", type=int, default=None, help="wsd branch: trunk step to branch from")
    ap.add_argument("--decay-steps", type=int, default=750, help="wsd branch: linear decay to 0 over this many steps")
    args = ap.parse_args()
    args.resume = None
    if args.branch_of:  # everything as in the trunk, except the output, the state to start from and the decay
        trunk = Path(args.branch_of)
        assert args.branch_step is not None, "--branch-of needs --branch-step"
        saved = json.loads((trunk / "args.json").read_text())
        for k in ("world_size", "git_commit", "host", "gpu"):
            saved.pop(k, None)
        branch = {"branch_of": str(trunk), "branch_step": args.branch_step, "decay_steps": args.decay_steps,
                  "out": str(trunk / "branches" / f"step{args.branch_step}"), "no_wandb": True,
                  "resume": str(trunk / f"state_step{args.branch_step}.pt")}
        args = argparse.Namespace(**{**vars(args), **saved, **branch})
        assert args.schedule == "wsd", "branches decay a wsd trunk"
    assert args.model and args.arm and args.out, "--model, --arm and --out (or --branch-of)"
    return args


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

    init_seed = args.seed if args.init_seed is None else args.init_seed
    torch.manual_seed(init_seed)  # same init of the new modules on every rank (DDP also broadcasts rank 0's)
    model = FusionVL(args.model, args.arm, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                     vggt_layers=args.vggt_layers, fusion_after=args.fusion_after, gate=args.gate,
                     special_tokens=args.special_tokens, resampler_layers=args.resampler_layers, rope3=args.rope3,
                     local_bias=args.local_bias, bias_lam=args.bias_lam, bias_mu=args.bias_mu, train_frames=args.nframes,
                     pe=args.pe, bias_nu=args.bias_nu, pe_dmax=args.pe_dmax,
                     pixels=TRAIN_PIXELS,
                     grad_ckpt=not args.no_grad_ckpt).to(dev)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    if rank == 0:
        print("trainable parameters:", {k: f"{v / 1e6:.1f}M" for k, v in count_params(model).items()}, flush=True)
    ddp = DDP(model, device_ids=[local], broadcast_buffers=False) if W > 1 else model

    uncapped = args.vsi590k  # the cap then applies to the pooled sources below
    if args.dev_videos:
        samples, dev_set = load_vlm3r(10**9 if uncapped else args.cap, args.seed, args.dev_videos, args.dev_per_task)
    else:
        samples, dev_set = load_vlm3r(10**9 if uncapped else args.cap, args.seed), []
    if args.vsi590k:
        assert args.appearance, "--vsi590k holds out the appearance-order dev videos too, so it needs --appearance"
        # every video held out for dev (VLM-3R dev scenes and the appearance-order dev videos) is held out here as well
        _, _, ao_held = load_appearance(args.seed, {s["video"] for s in dev_set}, args.ao_dev_videos, args.ao_dev_q,
                                        return_held=True)
        extra = load_vsi590k(ao_held | {s["video"] for s in dev_set})
        if rank == 0:
            print(f"VSI-590K: {len(extra)} questions on {len({s['video'] for s in extra})} videos added; pooled per task "
                  f"before the cap of {args.cap}: " + ", ".join(
                      f"{t} {n}" for t, n in sorted(Counter(s['task'] for s in samples + extra).items())), flush=True)
        samples = cap_per_task(samples + extra, args.cap, args.seed)
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
    steps_per_epoch = len(rank_steps(epoch_plan(samples, 0, init_seed, args.max_q), args.mb_per_step, W, rank))
    resume = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if args.schedule == "wsd":
        warm = args.warmup_steps
        decay_from = resume["step"] if resume else None
        total = decay_from + args.decay_steps if resume else warm + args.stable_steps
        if args.max_steps is not None:
            total = min(total, args.max_steps)
    else:
        total = args.epochs * steps_per_epoch if args.max_steps is None else min(args.max_steps, args.epochs * steps_per_epoch)
        warm = max(1, round(args.warmup * total))
    groups = []  # LoRA at lr, new modules at fusion_lr; no decay on biases and norm weights (1-d), as in HF's Trainer
    for is_lora, lr in ((True, args.lr), (False, args.fusion_lr or args.lr)):
        ps = [p for n, p in model.named_parameters() if p.requires_grad and ("lora_" in n) == is_lora and not is_slope(n)]
        for decay in (True, False):
            sub = [p for p in ps if (p.ndim > 1) == decay]
            if sub:
                groups.append({"params": sub, "lr": lr, "weight_decay": args.weight_decay if decay else 0.0})
    slopes = [p for n, p in model.named_parameters() if p.requires_grad and is_slope(n)]
    if slopes:  # xattn_aligned locality-bias slopes: own lr, no decay
        groups.append({"params": slopes, "lr": args.bias_lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    floor = args.min_lr
    if args.schedule == "wsd":  # the factor for optimizer update s (0-based); a branch's last update is at 1/decay_steps
        def factor(s):
            if s < warm:
                return (s + 1) / warm
            return 1.0 if decay_from is None or s < decay_from else max(0.0, 1 - (s - decay_from) / args.decay_steps)
    else:
        def factor(s):
            return (s + 1) / warm if s < warm else floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
    step = 0
    if resume:  # a wsd branch: the trunk's weights, optimizer moments, schedule position, step and RNG
        missing, unexpected = model.load_state_dict(resume["state"], strict=False)
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        assert not unexpected and not (trainable & set(missing)) and set(resume["state"]) == trainable
        opt.load_state_dict(resume["opt"])
        sched.load_state_dict(resume["sched"])
        step = resume["step"]
        torch.set_rng_state(resume["rng"]["cpu"])
        torch.cuda.set_rng_state(resume["rng"]["cuda"], dev)
        random.setstate(resume["rng"]["python"])
        assert sched.last_epoch == step, (sched.last_epoch, step)
        del resume
    if rank == 0:
        sched_s = (f"wsd: warmup {warm}, " + (f"decay from {decay_from} to 0 over {args.decay_steps}" if decay_from is not None
                   else f"stable to {total}, branch states at {args.branch_steps}")) if args.schedule == "wsd" else \
            f"warmup {warm}, lr floor {args.lr * floor:.1e}"
        print(f"{len(samples)} questions, {steps_per_epoch} steps/epoch, {total} steps, {sched_s}, "
              f"{args.mb_per_step // W} micro-batches per GPU per step, world size {W}", flush=True)

    accum = args.mb_per_step // W
    t_start = time.time()
    log = open(out / "log.jsonl", "a") if rank == 0 else None
    run = None
    if rank == 0 and not args.no_wandb:
        import wandb
        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=args.wandb_name or out.name, dir=str(out),
                         config={**json.loads((out / "args.json").read_text()), "params": count_params(model),
                                 "questions": len(samples), "steps_per_epoch": steps_per_epoch, "total_steps": total})
        (out / "wandb_run.json").write_text(json.dumps({"id": run.id, "entity": args.wandb_entity,
                                                        "project": args.wandb_project}))  # for mvp.train.wsd_collect
    def dev_eval(generate=False):
        """Mean per-question answer loss on dev (and VSI-style accuracy per task with generate=True), all ranks."""
        if not dev_mbs and W == 1:
            return {}
        model.eval()
        acc = {}
        tot = torch.zeros(2, device=dev)  # loss sum, questions
        with torch.no_grad():
            loader_dev = DataLoader(MicroBatches(dev_mbs, model.inputs, args.nframes), batch_size=None, num_workers=2)
            for i, (qs, b) in enumerate(zip(dev_mbs, loader_dev)):
                b = to_device(b, dev)
                model.collect_attn = generate and i == 0  # xattn_aligned: attention diagnostics on one micro-batch
                per_q = model(b)
                model.collect_attn = False
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
        if generate and model.attn_stats:  # rank 0's micro-batch
            res.update({f"attn/{k}_{s}": v for k, st in model.attn_stats.items() for s, v in st.items()})
            model.attn_stats = {}
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

    def save_state(path):
        """Everything a wsd branch needs to continue exactly where the trunk is; written atomically (the watcher
        submits a branch as soon as the file exists)."""
        if rank == 0:
            tmp = path.with_name(path.name + ".tmp")
            torch.save({"config": model.config, "state": model.trainable_state(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step,
                        "rng": {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(dev),
                                "python": random.getstate()}}, tmp)
            os.replace(tmp, path)

    wsd = args.schedule == "wsd"
    if dev_set and not args.resume:
        log_dev(dev_eval(generate=True), 0)  # untrained reference
    start_epoch, offset = divmod(step, steps_per_epoch)  # a branch continues the trunk's data order from its step
    for epoch in range(start_epoch, math.ceil(total / steps_per_epoch) if wsd else args.epochs):
        steps = rank_steps(epoch_plan(samples, epoch, init_seed, args.max_q), args.mb_per_step, W, rank)
        if epoch == start_epoch:
            steps = steps[offset:]
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
                if model.bias_slopes():
                    rec["bias"] = model.bias_slopes()
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if run is not None:
                    flat = {k: v for k, v in rec.items() if k not in ("gates", "inject", "bias")}
                    flat.update({f"bias/{i}_{k}": v for i, g in rec.get("bias", {}).items() for k, v in g.items()})
                    flat.update({f"gates/{i}_{k}": v for i, g in rec["gates"].items() for k, v in g.items()})
                    flat.update({f"inject/{i}": v for i, v in rec["inject"].items()})
                    run.log(flat, step=step)
                if step % 10 == 0 or step <= 5:
                    g = rec["gates"]
                    gs = f" gates xattn {min(v['xattn'] for v in g.values()):+.4f}..{max(v['xattn'] for v in g.values()):+.4f}" if g else ""
                    gs += " inject " + " ".join(f"{i}:{v:.3f}" for i, v in rec["inject"].items()) if rec["inject"] else ""
                    gs += " bias " + " ".join(f"{i}:" + "/".join(f"{x:.2f}" for x in v.values()) for i, v in rec["bias"].items()) if "bias" in rec else ""
                    print(f"step {step}/{total} ep {epoch} loss {rec['loss']:.4f} lr {rec['lr']:.2e} gnorm {rec['grad_norm']:.3f} "
                          f"q {rec['questions']} tok {rec['tokens']} {rec['step_time']:.2f}s mem {rec['peak_mem_gb']:.1f}G{gs}", flush=True)
            if wsd:
                if step in args.branch_steps and not args.resume:  # trunk: dev accuracy and a branch state
                    log_dev(dev_eval(generate=True), epoch, branch_state=f"state_step{step}.pt")
                    save_state(out / f"state_step{step}.pt")
                elif dev_set and step == total:  # end of a branch (or of a trunk without a state there)
                    log_dev(dev_eval(generate=True), epoch, **({"decayed_from": decay_from} if decay_from is not None else {}))
                elif dev_set and step % args.eval_every == 0:
                    log_dev(dev_eval(generate=True), epoch)
                elif dev_set and step % args.dev_every == 0:
                    log_dev(dev_eval(), epoch)
                continue
            k = next((k for k in range(1, args.evals_per_epoch) if step == epoch * steps_per_epoch + marks[k]), None)
            if k is not None:  # mid-epoch accuracy point + checkpoint
                save_point(epoch, epoch + k / args.evals_per_epoch)
            elif dev_set and step % args.dev_every == 0 and step % steps_per_epoch != 0:
                log_dev(dev_eval(), epoch)
        if wsd:
            if step >= total:
                break
        elif step >= (epoch + 1) * steps_per_epoch:
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
