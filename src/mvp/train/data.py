"""VLM-3R ScanNet QA training data, grouped so that every micro-batch holds questions about one video.

  * Samples: data/raw/SpatialStackData/merged_qa_scannet_train.json (51,779 VSI-Bench-template questions on 1,201
    ScanNet videos, none of them in VSI-Bench). The file has no task field; the question template gives it.
  * Cap: at most `cap` questions per task (seeded); room size has one question per video and keeps all 1,201.
  * Micro-batch: up to `max_q` questions of one video, so VGGT and Qwen's vision encoder run once per micro-batch.
    A video's questions are split into ceil(n / max_q) near-equal chunks, re-drawn every epoch.
  * Step: `mb_per_step` micro-batches in total, spread over the GPUs: each rank runs mb_per_step / world_size of them
    one after the other (gradient accumulation). Different ranks and accumulation steps get different videos.
    The plan is the same for any world size, so a run on 1 GPU and on 4 GPUs sees the same steps.
"""
import json
import math
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from mvp.fusion.inputs import uniform_frames
from mvp.paths import DATA, TRAIN_ANN, TRAIN_MEDIA
from mvp.reporting.train_examples import VLM3R_TASKS

VLM3R = TRAIN_ANN / "merged_qa_scannet_train.json"
VSI590K = DATA / "raw" / "VSI-590K" / "vsi_590k.jsonl"


def task_of(question):
    q = question.lower()
    return next((t for t, keys in VLM3R_TASKS if any(k in q for k in keys)), "other")


def load_vlm3r(cap=3000, seed=0, dev_videos=0, dev_per_task=100):
    """Capped sample list: dicts with id, video, task, question, answer.

    dev_videos > 0: hold out that many whole videos (seeded) and return (train, dev). The cap is applied to the questions
    of the remaining videos only; dev keeps up to dev_per_task questions per task from the held-out videos, so no dev
    scene is ever seen in training."""
    rows = json.load(open(VLM3R))
    rng = random.Random(seed)
    videos = sorted({r["video"] for r in rows})
    held = set(random.Random(f"dev-{seed}").sample(videos, dev_videos)) if dev_videos else set()
    by_task, dev_by_task = defaultdict(list), defaultdict(list)
    for r in rows:
        q, a = r["conversations"][0]["value"], r["conversations"][1]["value"]
        s = {"id": r["id"], "video": r["video"], "task": task_of(q), "question": q, "answer": a}
        (dev_by_task if r["video"] in held else by_task)[s["task"]].append(s)
    assert "other" not in by_task and "other" not in dev_by_task
    out = []
    for t in sorted(by_task):
        rs = by_task[t]
        out += rs if len(rs) <= cap else rng.sample(rs, cap)
    if not dev_videos:
        return out
    drng = random.Random(f"devq-{seed}")
    dev = []
    for t in sorted(dev_by_task):
        rs = dev_by_task[t]
        dev += rs if len(rs) <= dev_per_task else drng.sample(rs, dev_per_task)
    return out, dev


def load_appearance(seed=0, held_videos=(), dev_videos=40, dev_q=150):
    """VSI-590K appearance-order questions (all 16,162: there are no more): (train, dev) lists of samples.

    Dev holds out whole videos: every video in `held_videos` (the VLM-3R dev videos, so a ScanNet scene is held out
    for both tasks or neither) plus `dev_videos` more, split evenly over ScanNet++ v2 and ProcTHOR; dev keeps up to
    `dev_q` of their questions, split evenly over the three sources. ScanNet videos are the VLM-3R copies already on disk;
    ScanNet++ v2 and ProcTHOR come from the VSI-590K archives (data/train_media/vsi590k). None of the 1,077 videos
    is a VSI-Bench test scene. ProcTHOR questions are open-ended (an ordered list of names), the rest multiple choice.
    """
    rows = [json.loads(l) for l in open(VSI590K)]
    by_src = defaultdict(list)
    for i, r in enumerate(rows):
        if r["question_type"] != "appearance_order":
            continue
        src, rest = r["video"].split("/", 1)
        video = f"scannet/videos/{rest}" if src == "scannet" else f"vsi590k/{r['video']}"
        q = r["conversations"][0]["value"].replace("<image>", "").strip()
        by_src[src].append({"id": f"vsi590k-{i}", "video": video,
                            "task": "appearance_order", "question": q, "answer": r["conversations"][1]["value"].strip()})
    rng = random.Random(f"ao-dev-{seed}")
    held = set(held_videos)
    extra = sorted(src for src in by_src if src != "scannet")  # ScanNet dev scenes are exactly the VLM-3R dev scenes,
    for src in extra:                                            # whose VLM-3R questions are held out as well
        videos = sorted({s["video"] for s in by_src[src]} - held)
        held |= set(rng.sample(videos, math.ceil(dev_videos / len(extra))))
    train, dev = [], []
    for src in sorted(by_src):
        dev_src = [s for s in by_src[src] if s["video"] in held]
        train += [s for s in by_src[src] if s["video"] not in held]
        k = math.ceil(dev_q / len(by_src))
        dev += dev_src if len(dev_src) <= k else rng.sample(dev_src, k)
    return train, dev


def score_answer(pred, answer):
    """VSI-Bench scoring: exact letter match for multiple choice, mean relative accuracy for numbers, and normalised
    exact match for the open-ended answers (ProcTHOR appearance order: an ordered list of names)."""
    from mvp.benchmarks.vsibench import extract_number, fuzzy_matching, mean_relative_accuracy
    if answer.strip() in ("A", "B", "C", "D"):
        return float(fuzzy_matching(pred).upper() == answer.strip())
    try:
        target = float(answer)
    except ValueError:
        norm = lambda s: ", ".join(w.strip() for w in s.lower().strip().rstrip(".").split(","))
        return float(norm(pred) == norm(answer))
    p = extract_number(pred)
    return 0.0 if p is None else float(mean_relative_accuracy(p, target))


def epoch_plan(samples, epoch, seed=0, max_q=8):
    """Shuffled list of micro-batches (lists of samples, all from one video) for this epoch."""
    rng = random.Random(f"{seed}-{epoch}")
    by_video = defaultdict(list)
    for s in samples:
        by_video[s["video"]].append(s)
    mbs = []
    for v in sorted(by_video):
        qs = by_video[v][:]
        rng.shuffle(qs)
        k = math.ceil(len(qs) / max_q)
        mbs += [qs[i::k] for i in range(k)]
    rng.shuffle(mbs)
    return mbs


def rank_steps(mbs, mb_per_step, world_size, rank):
    """[(micro-batches of this rank for step i, questions in step i over all ranks)]; a partial last step is dropped."""
    assert mb_per_step % world_size == 0, (mb_per_step, world_size)
    accum = mb_per_step // world_size
    steps = []
    for i in range(len(mbs) // mb_per_step):
        step = mbs[i * mb_per_step:(i + 1) * mb_per_step]
        steps.append((step[rank * accum:(rank + 1) * accum], sum(map(len, step))))
    return steps


class MicroBatches(Dataset):
    """Model-ready micro-batches (InputBuilder.build output + metadata), decoded in DataLoader workers."""

    def __init__(self, mbs, builder, nframes=8):
        self.mbs, self.builder, self.nframes = mbs, builder, nframes

    def __len__(self):
        return len(self.mbs)

    def __getitem__(self, i):
        mb = self.mbs[i]
        frames, idx = uniform_frames(TRAIN_MEDIA / mb[0]["video"], self.nframes)
        batch = self.builder.build(frames, [s["question"] for s in mb], [s["answer"] for s in mb])
        batch["meta"] = {"video": mb[0]["video"], "ids": [s["id"] for s in mb], "tasks": [s["task"] for s in mb],
                         "frame_indices": idx}
        return batch


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
