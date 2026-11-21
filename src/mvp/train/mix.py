"""The 500K-question SFT mix and the held-out RL pool (data/mix/), one sample per line in data.load_mix's format.

  python -m mvp.train.mix build   # login node: annotations only (writes sft_500k.jsonl, rl_pool.jsonl, stats.json)
  python -m mvp.train.mix check   # Slurm CPU job: decodes the media of every SFT sample not checked yet
Run build, check, build, ... until check finds no new failures; build skips samples whose media failed.

  python -m mvp.train.mix ext     # Slurm CPU job: the 450K extended mix (ext_450k.jsonl, see build_ext), with its
                                  # own build / check rounds and the token estimates that data.pack_plan reads

Budgets (questions): VSI-590K the rest (~188K), DK-VLM route (RC) data 50K, SPAR 50K, VLM-3R VSI-Bench-style all on ScanNet
videos (~12K; its ScanNet++/ARKitScenes videos are gated), VLM-3R VSTI-Bench 50K, MindCube 6K, LLaVA-Hound 48K, PhysGame PhysInstruct 48K, Cambrian-S (short-clip sources) 48K.

Held out:
  * rooms of VSI-Bench (test) and the dev rooms of run.py (load_vlm3r's dev videos and load_appearance's held-out
    videos): in neither the mix nor the RL pool;
  * RL rooms: 15% of all rooms (hashed, so one room is RL for every source; ScanNet re-scans _00, _01 are one room;
    ScanNet++ v2 = ScanNet++ ids). Videos without a room (ADT, S3DIS, ProcTHOR, general video) and MindCube image sets
    are their own room. Every question on an RL room goes to rl_pool.jsonl; the SFT budgets are sampled (seeded) from
    the other rooms, and their unused questions are dropped, so the RL pool's media are never seen in SFT.

Sample: {id, source, task, video (micro-batch group key), media (path under data/train_media, or a list of image
paths), question (no <image>/<video> tags, except SPAR's "Frame-i: <image>" labels, where InputBuilder puts each
image), answer, room, frames (only for questions that name frames: VSTI "frame 3 of 32", SPAR's 32 labelled frames)}.
Selection per source:
  * VSI-590K: every video question type (image rows skipped), uniform over its videos' questions;
  * RC: the 2- and 3-step chains only; every video question (VSI-590K videos), the rest from SPAR's image tuples,
    one question (heading or delta, seeded) per tuple;
  * Cambrian-S: single-turn conversations only; the budget split as evenly as possible over the 14 sources
    (a source with too few questions gives all of them, the rest is shared by the others).
"""
import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from mvp.paths import DATA, TRAIN_ANN, TRAIN_MEDIA

OUT = DATA / "mix"
CHECK = OUT / "media_check.json"
RAW = DATA / "raw"
VLM3R = RAW / "VLM-3R-DATA"
RC = Path("/shared/workspace/povejmo/huggingface_cache/hub/datasets--HITshan--DK-VLM-RCdata/snapshots/"
          "8b84d48a3a5474ff02f0b41a1198bf420fc3ef64")
CAMBRIAN = RAW / "Cambrian-S-3M" / "cambrian_s_3m.jsonl"
CAMBRIAN_SOURCES = ["ssv2", "textvr", "star", "clevrer", "EGTEA", "IndustReal", "ChardesEgo", "nextqa", "HoloAssist",
                    "tgif", "guiworld", "activitynet", "EgoTask", "tvqa"]
# None: vlm3r_vsi takes every question on SFT rooms (only its ScanNet videos are available, ~12K); vsi590k (built
# last) fills the rest of TOTAL. The VLM-3R ScanNet++/ARKitScenes videos are gated and were never granted, and
# VSI-590K has the same question types on the same ScanNet++ rooms.
TOTAL = 500_000
BUDGET = {"rc": 50_000, "spar": 50_000, "vlm3r_vsi": None, "vlm3r_vsti": 50_000, "mindcube": 6_000,
          "llava_hound": 48_000, "physgame": 48_000, "cambrian": 48_000, "vsi590k": None}
RL_FRACTION = 0.15
TAGS = re.compile(r"^(\s*<(image|video)>\s*)+")


def room_of(media):
    """Room key of a sample's media (see the module docstring)."""
    p = media[0] if isinstance(media, list) else media
    parts = p.split("/")
    if parts[0] == "scannet":  # scannet/videos/scene0191_00.mp4
        return "scannet/" + re.sub(r"_\d+$", "", Path(p).stem)
    if parts[0] == "vsi590k":  # vsi590k/<source>/<path>
        src, rest = parts[1], "/".join(parts[2:])
        src = {"scannetppv2": "scannetpp"}.get(src, src)
        return f"{src}/{Path(rest).with_suffix('')}"
    if parts[0] == "vlm3r":  # vlm3r/<scannetpp|arkitscenes>/videos/<id>.mp4
        return f"{parts[1]}/{Path(p).stem}"
    if parts[0] == "spar":  # spar/<source>/images/<scene>/...
        scene = re.sub(r"_\d+$", "", parts[3]) if parts[1] == "scannet" else parts[3]
        return f"{parts[1]}/{scene}"
    if parts[0] == "mindcube":  # one image set
        return "mindcube/" + str(Path(p).parent)
    return "video/" + p


def is_rl(room):
    return int(hashlib.sha256(f"rl-{room}".encode()).hexdigest(), 16) % 10_000 < RL_FRACTION * 10_000


def clean(q):
    """The question without its <image>/<video> media tags (the trainer adds the media itself), wherever they are."""
    q = TAGS.sub("", q).strip()
    if re.search(r"<(image|video)>", q):  # a tag later in the text (some Cambrian-S questions end with one)
        q = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+(?=\n)| (?= )", "", re.sub(r"<(image|video)>", "", q))).strip()
    return q


def sample(source, task, sid, media, question, answer, frames=None):
    key = json.dumps(media) if isinstance(media, list) else media
    s = {"id": sid, "source": source, "task": task, "video": key if frames is None else f"{key}@{frames}",
         "media": media, "question": question, "answer": answer.strip(), "room": room_of(media)}
    if frames is not None:
        s["frames"] = frames
    return s


# ---- sources: every candidate question, as samples ----

def vsi590k():
    from mvp.train.data import VSI590K
    from mvp.train.prep_vsi590k import MANIFEST, media_path
    ok = {p for p, r in json.loads(MANIFEST.read_text()).items() if r["ok"]}
    out = []
    for i, r in enumerate(map(json.loads, open(VSI590K))):
        if "video" in r and media_path(r["video"]) in ok:
            c = r["conversations"]
            out.append(sample("vsi590k", r["question_type"], f"vsi590k-{i}", media_path(r["video"]), clean(c[0]["value"]),
                              c[1]["value"]))
    return out


def rc():
    """(video questions, SPAR tuple questions), 2- and 3-step chains only."""
    from mvp.train.prep_vsi590k import MANIFEST, media_path
    ok = {p for p, r in json.loads(MANIFEST.read_text()).items() if r["ok"]}
    steps = lambda r: len(json.loads(r["conversations"][-1]["value"]))
    video = []
    for i, r in enumerate(map(json.loads, open(RC / "rc_vsi_sft_all_2to4.jsonl"))):
        if "video" in r and steps(r) in (2, 3) and media_path(r["video"]) in ok:
            c = r["conversations"]
            video.append(sample("rc", r["question_type"], f"rc_vsi-{i}", media_path(r["video"]), clean(c[0]["value"]),
                                c[1]["value"]))
    by_tuple = defaultdict(list)
    for part in (1, 2):
        for i, r in enumerate(map(json.loads, open(RC / f"rc_spar_sft_all_2to4.part{part}.jsonl"))):
            if steps(r) in (2, 3):
                c = r["conversations"]
                by_tuple[tuple(r["images"])].append(sample("rc", r["question_type"], f"rc_spar{part}-{i}", r["images"],
                                                           clean(c[0]["value"]), c[1]["value"]))
    rng = random.Random("rc-tuple")
    return video, [rng.choice(qs) for _, qs in sorted(by_tuple.items())]


def spar():
    out = []
    for i, r in enumerate(json.load(open(TRAIN_ANN / "spar_234k.json"))):
        q, a = r["conversations"][0]["value"], r["conversations"][1]["value"]
        n = len(r["images"])
        info = json.loads(r["spar_info"]) if isinstance(r["spar_info"], str) else r["spar_info"]
        task = info.get("type", "spar")
        if n == 32:  # "Frame-0: <image>\nFrame-1: <image>..." labels, referred to in the question: kept, all 32 shown
            assert q.count("<image>") == 32, r["id"]
            out.append(sample("spar", task, f"spar-{i}", r["images"], q.strip(), a, frames=32))
        else:
            out.append(sample("spar", task, f"spar-{i}", r["images"], clean(q), a))
    return out


def vlm3r_vsi():
    from mvp.train.data import task_of
    out = []
    for f in sorted((VLM3R / "vsibench_train").glob("merged_qa_*_train.json")):
        for i, r in enumerate(json.load(open(f))):
            v = r["video"]  # scannet/videos/<scene>.mp4 (on disk); scannetpp/..., arkitscenes/... (gated, skipped)
            if not v.startswith("scannet/"):
                continue
            media = v
            q, a = clean(r["conversations"][0]["value"]), r["conversations"][1]["value"]
            t = task_of(q)
            out.append(sample("vlm3r_vsi", "route_plan" if "route_plan" in f.name else t, f"vlm3r-{f.stem}-{i}", media, q, a))
    return out


def vlm3r_vsti():
    out = []
    for f in sorted((VLM3R / "vstibench_train").glob("qa_*.json")):
        for i, r in enumerate(json.load(open(f))):
            q, a = clean(r["conversations"][0]["value"]), r["conversations"][1]["value"]
            n = {int(x) for x in re.findall(r"\bof (\d+)\b", q)}
            assert len(n) == 1, q
            assert r["video"].startswith("ScanNet/videos/train/"), r["video"]
            out.append(sample("vlm3r_vsti", r["question_type"], f"vsti-{f.stem}-{i}",
                              f"scannet/videos/{Path(r['video']).name}", q, a, frames=n.pop()))
    return out


def mindcube():
    root = TRAIN_MEDIA / "mindcube" / "data"
    out = []
    for i, r in enumerate(map(json.loads, open(root / "raw" / "MindCube_train.jsonl"))):
        out.append(sample("mindcube", r["type"], f"mindcube-{i}", [f"mindcube/data/{p}" for p in r["images"]],
                          r["question"].strip(), r["gt_answer"]))
    return out


def llava_hound():
    out = []
    for i, r in enumerate(json.load(open(TRAIN_ANN / "llava_hound_64k.json"))):
        c = r["conversations"]
        out.append(sample("llava_hound", "caption_qa", f"hound-{i}", r["video"], clean(c[0]["value"]), c[1]["value"]))
    return out


def physgame():
    out = []
    for i, r in enumerate(json.load(open(RAW / "PhysGame-40K" / "PhysInstruct_anno_40k.json"))):
        for j, qa in enumerate(r["QA"]):
            assert not qa["i"]
            out.append(sample("physgame", "physinstruct", f"physgame-{i}-{j}",
                              f"physgame/PhysInstruct/{r['video']}", qa["q"].strip(), qa["a"]))
    return out


def cambrian():
    src = set(CAMBRIAN_SOURCES)
    out = []
    for i, r in enumerate(map(json.loads, open(CAMBRIAN))):
        v = r.get("video")
        # 2-turn conversations; not the ~2K "<speech>" questions, which need the audio
        if isinstance(v, str) and v.split("/")[0] in src and len(r["conversations"]) == 2 \
                and "<speech>" not in r["conversations"][0]["value"]:
            c = r["conversations"]
            out.append(sample("cambrian", v.split("/")[0], f"cambrian-{i}", f"cambrian/{v}", clean(c[0]["value"]),
                              c[1]["value"]))
    return out


# ---- held-out rooms ----

def excluded_rooms():
    """{room: reason} for VSI-Bench test rooms and run.py's dev rooms (seed 0, its default dev arguments)."""
    from mvp.benchmarks.vsibench import load_vsibench
    from mvp.train.data import load_appearance, load_vlm3r
    out = {}
    bench = load_vsibench("full")
    for d, s in set(zip(bench["dataset"], bench["scene_name"].astype(str))):
        out[{"scannet": "scannet/" + re.sub(r"_\d+$", "", s)}.get(d, f"{d}/{s}")] = "vsibench"
    _, dev = load_vlm3r(10**9, 0, 60, 100)
    _, _, held = load_appearance(0, {s["video"] for s in dev}, 40, 150, return_held=True)
    for v in held | {s["video"] for s in dev}:
        out.setdefault(room_of(v), "dev")
    return out


# The extended mix (2026-11-20): 250K VSI-Bench-style questions (the ablation's training set, run.py --vsi590k --cap
# 50000 --appearance, plus VSI-590K's other video question types), all of MindCube, and DK-VLM's other datasets in
# proportion to their table (SPAR 234K, VSTI 132K, Cambrian-S 430K, PhysGame 140K, LLaVA-Hound 64K, RC 150K: 16.5%
# of each, 190K in all). No RL rooms; VSI-Bench and dev rooms are held out as in build.
EXT_VSI = 250_000
EXT_VSI_EXTRA = ("relative_size_object", "absolute_direction_object", "relative_count")
EXT_BUDGET = {"rc": 24_783, "spar": 38_661, "vlm3r_vsti": 21_809, "mindcube": 10_000, "llava_hound": 10_574,
              "physgame": 23_130, "cambrian": 71_043}


def take(cands, n, bad, seed):
    """n candidates (seeded uniform order) whose media did not fail the check."""
    order = sorted(cands, key=lambda s: s["id"])
    random.Random(f"mix-{seed}").shuffle(order)
    return [s for s in order if s["video"].split("@")[0] not in bad][:n]


def select(src, pool, n, bad):
    """n questions of a source from its pool (all of it for n None), by the source's rule (module docstring)."""
    if src == "rc":  # every video question, then SPAR tuples up to the budget
        vid = [s for s in pool if isinstance(s["media"], str)]
        got = take(vid, n, bad, "rc-video")
        got += take([s for s in pool if isinstance(s["media"], list)], n - len(got), bad, "rc-spar")
    elif src == "cambrian":  # as even as possible over the sources
        by = defaultdict(list)
        for s in pool:
            by[s["task"]].append(s)
        left, got = n, []
        for i, t in enumerate(sorted(by, key=lambda t: len(by[t]))):
            g = take(by[t], left // (len(by) - i), bad, f"cambrian-{t}")
            got += g
            left -= len(g)
    else:
        got = take(pool, len(pool) if n is None else n, bad, src)
    assert n is None or len(got) == n, (src, len(got), n)
    return got


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    checked = json.loads(CHECK.read_text()) if CHECK.exists() else {}
    bad = {k for k, r in checked.items() if not r["ok"]}
    excl = excluded_rooms()
    cands = {"vsi590k": vsi590k(), "spar": spar(), "vlm3r_vsi": vlm3r_vsi(), "vlm3r_vsti": vlm3r_vsti(),
             "mindcube": mindcube(), "llava_hound": llava_hound(), "physgame": physgame(), "cambrian": cambrian()}
    rc_video, rc_spar = rc()
    cands["rc"] = rc_video + rc_spar
    stats = {"excluded_rooms": dict(Counter(excl.values())), "sources": {}}
    sft, rl = [], []
    for src in BUDGET:
        n = TOTAL - len(sft) if src == "vsi590k" else BUDGET[src]
        rows = cands[src]
        hit = Counter(excl[s["room"]] for s in rows if s["room"] in excl)
        rows = [s for s in rows if s["room"] not in excl]
        pool = [s for s in rows if not is_rl(s["room"])]
        rl_src = [s for s in rows if is_rl(s["room"])]
        got = select(src, pool, n, bad)
        sft += got
        rl += rl_src
        stats["sources"][src] = {
            "candidates": len(cands[src]), "on_excluded_rooms": dict(hit), "sft_room_questions": len(pool),
            "sft": len(got), "sft_rooms": len({s["room"] for s in got}), "rl_questions": len(rl_src),
            "rl_rooms": len({s["room"] for s in rl_src}), "sft_tasks": dict(Counter(s["task"] for s in got).most_common()),
            "fixed_frames": dict(Counter(s.get("frames") for s in got if s.get("frames")))}
    sft_rooms, rl_rooms = {s["room"] for s in sft}, {s["room"] for s in rl}
    assert not sft_rooms & rl_rooms and not (sft_rooms | rl_rooms) & set(excl)
    assert len({s["id"] for s in sft}) == len(sft)
    for s in sft + rl:  # "<image>" only as the placeholders of an image set (inputs.prompt_text interleaves them)
        n = s["question"].count("<image>")
        assert n == 0 or (isinstance(s["media"], list) and n == len(s["media"])), (s["id"], s["question"][:100])
    unchecked = {s["video"].split("@")[0] for s in sft if s["source"] != "vsi590k" and s["video"].split("@")[0] not in checked}
    stats.update(sft=len(sft), rl=len(rl), sft_rooms=len(sft_rooms), rl_rooms=len(rl_rooms),
                 media_failed=len(bad), media_unchecked=len(unchecked))
    random.Random("mix-order").shuffle(sft)  # the trainer re-plans every epoch; the file order does not matter
    with open(OUT / "sft_500k.jsonl", "w") as f:
        f.writelines(json.dumps(s) + "\n" for s in sft)
    with open(OUT / "rl_pool.jsonl", "w") as f:
        f.writelines(json.dumps(s) + "\n" for s in rl)
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


def ablation_vsi():
    """The ablation runs' training questions (run.py --appearance --vsi590k --cap 50000, data seed 0), as mix samples:
    VLM-3R ScanNet (source vlm3r_vsi) and VSI-590K (vsi590k), the six VSI-Bench tasks capped at 50K each over both,
    plus every appearance-order question off the dev videos."""
    from mvp.train.data import cap_per_task, load_appearance, load_vlm3r, load_vsi590k
    vlm3r, dev = load_vlm3r(10**9, 0, 60, 100)
    dev_videos = {s["video"] for s in dev}
    _, _, held = load_appearance(0, dev_videos, 40, 150, return_held=True)
    samples = cap_per_task(vlm3r + load_vsi590k(held | dev_videos), 50_000, 0)
    samples += load_appearance(0, dev_videos, 40, 150)[0]
    return [{**s, "source": "vsi590k" if str(s["id"]).startswith("vsi590k-") else "vlm3r_vsi", "media": s["video"],
             "room": room_of(s["video"])} for s in samples]


def annotate(rows, checked):
    """Token estimates for data.pack_plan: text_tokens (template, question, answer), frame_tokens (merged visual tokens
    of one frame at the training pixel limits, from the first frame's size) and media_frames (frames or images there
    are, up to 32). Returns the number of samples without a check record (left unannotated)."""
    from PIL import Image
    from transformers import AutoProcessor
    from mvp.fusion.inputs import TRAIN_PIXELS, InputBuilder
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-2B", **TRAIN_PIXELS)  # the same tokenizer and processor as 4B, 9B
    merge = proc.image_processor.merge_size ** 2
    frame_tokens = {}
    base = len(proc.tokenizer(InputBuilder(proc).prompt_text("", 0), add_special_tokens=False).input_ids) + 1  # + <|im_end|>
    tok = proc.tokenizer([s["question"] for s in rows], add_special_tokens=False).input_ids
    ans = proc.tokenizer([s["answer"] for s in rows], add_special_tokens=False).input_ids
    missing = 0
    for s, q, a in zip(rows, tok, ans):
        r = checked.get(s["video"].split("@")[0])
        if r is None or not r["ok"]:
            missing += 1
            continue
        hw = tuple(r["hw"])
        if hw not in frame_tokens:
            g = proc.image_processor(images=[Image.new("RGB", hw[::-1])], return_tensors="pt")["image_grid_thw"]
            frame_tokens[hw] = int(g.prod()) // merge
        s.update(text_tokens=base + len(q) + len(a), frame_tokens=frame_tokens[hw], media_frames=r["n"])
    return missing


def build_ext(annotated=False):
    """data/mix/ext_450k.jsonl and ext_450k_stats.json (see EXT_BUDGET). annotated: every media must have been
    checked; adds the token estimates. VSI-590K's videos (and the VLM-3R ScanNet copies) were checked by prep_vsi590k,
    whose manifest also has their frame size and count."""
    from mvp.train.prep_vsi590k import MANIFEST
    checked = json.loads(CHECK.read_text()) if CHECK.exists() else {}
    for k, r in json.loads(MANIFEST.read_text()).items():
        checked.setdefault(k, {"ok": r["ok"], "n": min(32, r.get("frames", 0)), "hw": r.get("hw")})
    bad = {k for k, r in checked.items() if not r["ok"]}
    excl = excluded_rooms()
    stats = {"excluded_rooms": dict(Counter(excl.values())), "sources": {}}
    vsi = ablation_vsi()
    n_abl = len(vsi)
    vsi = [s for s in vsi if s["room"] not in excl and s["video"] not in bad]
    stats["ablation_set"] = {"questions": n_abl, "dropped_on_held_out_rooms_or_failed_media": n_abl - len(vsi)}
    extra = [s for s in vsi590k() if s["task"] in EXT_VSI_EXTRA and s["room"] not in excl]
    vsi += take(extra, EXT_VSI - len(vsi), bad, "ext-vsi590k")
    assert len(vsi) == EXT_VSI, len(vsi)
    stats["sources"]["vsi"] = {"sft": len(vsi), "extra_pool": len(extra), "by_source": dict(Counter(s["source"] for s in vsi)),
                               "sft_tasks": dict(Counter(s["task"] for s in vsi).most_common())}
    cands = {"spar": spar(), "vlm3r_vsti": vlm3r_vsti(), "mindcube": mindcube(), "llava_hound": llava_hound(),
             "physgame": physgame(), "cambrian": cambrian()}
    rc_video, rc_spar = rc()
    cands["rc"] = rc_video + rc_spar
    sft = vsi
    for src, n in EXT_BUDGET.items():
        pool = [s for s in cands[src] if s["room"] not in excl]
        got = select(src, pool, n, bad)
        sft += got
        stats["sources"][src] = {"candidates": len(cands[src]), "off_held_out_rooms": len(pool), "sft": len(got),
                                 "sft_media": len({s["video"] for s in got}),
                                 "sft_tasks": dict(Counter(s["task"] for s in got).most_common())}
    assert len(sft) == EXT_VSI + sum(EXT_BUDGET.values()) and len({s["id"] for s in sft}) == len(sft)
    assert not {s["room"] for s in sft} & set(excl)
    for s in sft:  # as in build
        n = s["question"].count("<image>")
        assert n == 0 or (isinstance(s["media"], list) and n == len(s["media"])), (s["id"], s["question"][:100])
    unchecked = {s["video"].split("@")[0] for s in sft} - set(checked)
    stats.update(sft=len(sft), media=len({s["video"].split("@")[0] for s in sft}), media_unchecked=len(unchecked),
                 media_failed_skipped=len(bad))
    if annotated:
        assert not unchecked, f"{len(unchecked)} media not checked"
        assert annotate(sft, checked) == 0
    random.Random("ext-order").shuffle(sft)
    with open(OUT / "ext_450k.jsonl", "w") as f:
        f.writelines(json.dumps(s) + "\n" for s in sft)
    (OUT / "ext_450k_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "sources"}), flush=True)
    return unchecked


def check_one(key):
    """Decode a media item as the trainer does, at 32 frames (the largest count it uses)."""
    from mvp.fusion.inputs import media_frames
    try:
        media = [TRAIN_MEDIA / p for p in json.loads(key)] if key.startswith("[") else TRAIN_MEDIA / key
        frames, idx = media_frames(media, 32)
        assert frames, "no frames"
        return key, {"ok": True, "n": len(idx), "hw": [frames[0].height, frames[0].width]}
    except Exception as e:  # missing file, broken container, decord EOF errors
        return key, {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def check(workers, exclude=(), keys=None):
    """Decode check of the SFT mix's media not checked yet (VSI-590K's were checked by prep_vsi590k), or of `keys`."""
    from multiprocessing import Pool
    checked = json.loads(CHECK.read_text()) if CHECK.exists() else {}
    if keys is None:
        keys = {json.loads(l)["video"].split("@")[0] for l in open(OUT / "sft_500k.jsonl") if '"source": "vsi590k"' not in l}
    keys = sorted(set(keys) - set(checked))
    keys = [k for k in keys if not k.lstrip('["').startswith(tuple(exclude))]  # e.g. media not downloaded yet
    print(f"{len(keys)} media to check, {len(checked)} checked before", flush=True)
    with Pool(workers) as pool:
        for i, (k, r) in enumerate(pool.imap_unordered(check_one, keys, chunksize=16)):
            checked[k] = r
            if (i + 1) % 5000 == 0:
                print(f"{i + 1}/{len(keys)}, {sum(not v['ok'] for v in checked.values())} failed", flush=True)
                CHECK.write_text(json.dumps(checked))
    CHECK.write_text(json.dumps(checked))
    fails = [(k, r["error"]) for k, r in checked.items() if not r["ok"] and k in set(keys)]
    print(f"new failures: {len(fails)}", *fails[:20], sep="\n")
    return len(fails)


def ext(workers):
    """build_ext, then check its new media (the check also gives each media's frame size and count), rebuilding until
    every media of the mix is checked and none fails; then the annotated build."""
    for _ in range(5):
        unchecked = build_ext()
        if not unchecked:
            build_ext(annotated=True)
            return
        check(workers, keys=unchecked)
    sys.exit("media still unchecked after 5 rounds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check", "ext"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--exclude", nargs="*", default=[], help="check: skip media under these prefixes (e.g. vlm3r/)")
    a = ap.parse_args()
    if a.cmd == "ext":
        return ext(a.workers)
    build() if a.cmd == "build" else check(a.workers, a.exclude)


if __name__ == "__main__":
    sys.exit(main())
