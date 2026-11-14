"""VLM-3R ScanNet QA training data, grouped so that every micro-batch holds questions about one video.

  * Samples: data/raw/SpatialStackData/merged_qa_scannet_train.json (51,779 VSI-Bench-template questions on 1,201
    ScanNet videos, none of them in VSI-Bench). The file has no task field; the question template gives it.
  * Cap: at most `cap` questions per task (seeded); room size has one question per video and keeps all 1,201.
  * Micro-batch: up to `max_q` questions of one video, so VGGT and Qwen's vision encoder run once per micro-batch.
    A video's questions are split into ceil(n / max_q) near-equal chunks, re-drawn every epoch.
  * Step: `mb_per_step` micro-batches in total, spread over the GPUs: each rank runs mb_per_step / world_size of them
    one after the other (gradient accumulation). Different ranks and accumulation steps get different videos.
    The plan is the same for any world size, so a run on 1 GPU and on 4 GPUs sees the same steps.
  * Frames: one count per step (step_frames), so every micro-batch of a step, on every rank, decodes the same number
    of frames: no padding between sequences of different lengths and no rank waiting for a longer one.
"""
import json
import math
import os
import random
import time
from collections import defaultdict

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "40960")  # 4x decord's default: retries when seeking to the last frames

import torch
from torch.utils.data import Dataset

from mvp.fusion.inputs import media_frames
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


def load_appearance(seed=0, held_videos=(), dev_videos=40, dev_q=150, return_held=False):
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
    return (train, dev, held) if return_held else (train, dev)  # held: every held-out video, sampled into dev or not


# VSI-590K question types that are VSI-Bench tasks (appearance order comes from load_appearance), in dev's task names.
# Not used: relative_size_object, absolute_direction_object, relative_count, relative_*_camera (no VSI-Bench task) and
# the image rows (no video). VSI-590K has no route-planning questions.
VSI590K_TASKS = {"relative_direction_object": "rel_direction", "relative_distance_object": "rel_distance",
                 "absolute_distance_object": "abs_distance", "absolute_size_object": "obj_size",
                 "absolute_size_room": "room_size", "absolute_count": "counting"}


def load_vsi590k(held_videos=()):
    """VSI-590K's questions of the six VSI-Bench tasks other than appearance order, on every video source (ScanNet,
    ScanNet++ v2, ARKitScenes, ADT, S3DIS, ProcTHOR), as samples. Skipped: videos in `held_videos` (dev, so no dev
    scene is trained on), and videos that failed prep_vsi590k's decode check. None of the videos is a VSI-Bench test
    scene (checked by scene id for ScanNet, ScanNet++ and ARKitScenes; ADT, S3DIS and ProcTHOR are not in VSI-Bench).
    Templates are paraphrases of VSI-Bench's, some in other units (cm, inches); answers are letters or numbers."""
    from mvp.train.prep_vsi590k import MANIFEST, media_path
    ok = {p for p, r in json.loads(MANIFEST.read_text()).items() if r["ok"]}
    held = set(held_videos)
    out = []
    for i, r in enumerate(map(json.loads, open(VSI590K))):
        if "video" not in r or r["question_type"] not in VSI590K_TASKS:
            continue
        video = media_path(r["video"])
        if video in held or video not in ok:
            continue
        q = r["conversations"][0]["value"].replace("<image>", "").strip()
        out.append({"id": f"vsi590k-{i}", "video": video, "task": VSI590K_TASKS[r["question_type"]], "question": q,
                    "answer": r["conversations"][1]["value"].strip()})
    return out


def cap_per_task(samples, cap, seed=0):
    """At most `cap` samples per task (seeded, uniform over the pooled sources)."""
    rng = random.Random(f"cap-{seed}")
    by_task = defaultdict(list)
    for s in samples:
        by_task[s["task"]].append(s)
    return [s for t in sorted(by_task) for s in (by_task[t] if len(by_task[t]) <= cap else rng.sample(by_task[t], cap))]


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


def step_frames(n_steps, mix, seed, epoch):
    """Frame count of each of an epoch's n_steps steps. mix: {frames: percent}, percentages summing to 100; each count
    gets its share of the steps exactly (largest remainder), in a seeded order that depends only on (seed, epoch),
    so a wsd branch that resumes mid-epoch sees the trunk's counts."""
    ks = sorted(mix)
    exact = [mix[k] / 100 * n_steps for k in ks]
    counts = [math.floor(x) for x in exact]
    for i in sorted(range(len(ks)), key=lambda i: counts[i] - exact[i])[:n_steps - sum(counts)]:
        counts[i] += 1
    out = [k for k, c in zip(ks, counts) for _ in range(c)]
    random.Random(f"frames-{seed}-{epoch}").shuffle(out)
    return out


def load_mix(path):
    """Samples of a training mix built by mvp.train.mix (one JSON object per line)."""
    return [json.loads(l) for l in open(path)]


class MicroBatches(Dataset):
    """Model-ready micro-batches (InputBuilder.build output + metadata), decoded in DataLoader workers. nframes: one
    count for all micro-batches, or one per micro-batch. A sample's "media" (a path under TRAIN_MEDIA or a list of
    them; default: its "video") is what is decoded, and its "frames", if set, overrides the count (questions that name
    frames, e.g. "frame 3 of 32"); mvp.train.mix gives such samples a group key of their own, so all samples of a
    micro-batch share media and count."""

    def __init__(self, mbs, builder, nframes=8):
        self.mbs, self.builder = mbs, builder
        self.nframes = [nframes] * len(mbs) if isinstance(nframes, int) else nframes
        assert len(self.nframes) == len(mbs), (len(self.nframes), len(mbs))

    def __len__(self):
        return len(self.mbs)

    def __getitem__(self, i):
        mb = self.mbs[i]
        media = mb[0].get("media", mb[0]["video"])
        media = [TRAIN_MEDIA / p for p in media] if isinstance(media, list) else TRAIN_MEDIA / media
        k = mb[0].get("frames") or self.nframes[i]
        for attempt in range(6):  # decord occasionally fails to seek near the end of a video under load (EOF retry
            try:                  # limit) and succeeds on a fresh reader; a video that keeps failing still raises.
                frames, idx = media_frames(media, k)  # 3 attempts 2 s apart once failed (v9 lora_s0)
                break
            except Exception as e:
                if attempt == 5:
                    raise RuntimeError(f"cannot decode {media} ({k} frames)") from e
                time.sleep(5 * (attempt + 1))
        batch = self.builder.build(frames, [s["question"] for s in mb], [s["answer"] for s in mb])
        batch["meta"] = {"video": mb[0]["video"], "ids": [s["id"] for s in mb], "tasks": [s["task"] for s in mb],
                         "frame_indices": idx}
        return batch


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
