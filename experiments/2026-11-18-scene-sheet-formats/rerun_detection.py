"""Rerun SAM 3 and tracking on the dev scenes with other detection settings, for rerun_detection.ipynb (Colab, GPU).

Scenes: dev_scenes.txt (50), or SUBSET_FILE (dev_scenes_15.txt), 5 per dataset drawn with SUBSET_SEED
(stratified_subset): ScanNet++ 3 from the lower half of its scenes by recall (the share of the questions' named objects
with a row in the pipeline sheets of data/scene_sheets_75) and 2 from the upper half, ARKitScenes and ScanNet 5 at
random. Recall is measured on dev scenes only. `python rerun_detection.py` writes the file again.

For each scene, on the 16 frames its sheets were made on:

  SAM 3     facebook/sam3 at SAM3_REV, transformers backend, as sam3_infer.py runs it (its own functions), once per
            scene for every SAM 3 prompt any run of RUNS needs (whether or not it is chosen), keeping every detection
            from SAVE_MIN_SCORE (0.2) with its score: sam3/<dataset>__<scene>/ (<frame>__masks.npz and manifest.json,
            sam3_infer.py's format). A scene's SAM 3 run is reused by every later run; if a run needs a prompt it does
            not have, SAM 3 runs for the missing prompts only and they are added to it (merge_sam3: SAM 3 segments each
            prompt on its own, with the frame's image features computed once, so adding prompts later gives the same
            detections as one run with all of them).
  runs      each run (RUNS) takes a prompt set (the original prompts, or the fixed ones) and tracking settings, and
            writes a sheets folder that the local scripts read like data/scene_sheets_75:
              runs/<run>/index.csv                              the dev scenes' questions, prompts as the run uses them
              runs/<run>/<dataset>__<scene>/subset-uniform16/
                  masks/rerun_<prompt set>/                     the run's masks (sam3_infer.py's format)
                  scene/cameras.npz, adapter_report.json        as saved by the sheet run
                  scene/tracks/<prompt>.npz, .settings.json     milestone4_video.py's track stage on those masks
                  sheets/<folder>/question.json                 the question's prompts and settings (no objects.csv)
              runs/<run>/<dataset>__<scene>/gold/sheets/<folder>/  question.json, and objects.csv and sheet.json where
                                                                the prompts are unchanged
              runs/<run>/<dataset>__<scene>/DONE.json           written last: the scene is finished
            The sheets themselves are made locally from the tracks (rebuild_scene_sheets.py, or merge_points.py
            --no-merge for a chosen sizing), with the dense depth in data/vggt.

Prompt sets (fix_prompts):
  original  each question's prompts as the sheet run used them
  fixed     the same with three fixes, from the question's text alone:
            - multi-word names kept together: two prompts that the question writes next to each other ("ceiling" and
              "light" in "... the ceiling light?") become one ("ceiling light"), if neither appears on its own
            - plural names: a prompt the question writes only in the plural ("shoe" for "shoes") is named as the
              question writes it, and SAM 3 is prompted with both forms
            - synonyms (SYNONYMS): SAM 3 is also prompted with other common names of the object, and the detections
              of all of an object's prompts are pooled under its name, a detection dropped if its mask overlaps one
              with a higher score from another prompt (IoU >= DEDUPE_IOU, so the tracker doesn't see one object twice
              in a frame); detections of a single prompt are kept as SAM 3 gives them
            Each object keeps one name, so a question's rows stay under the names it asks about.

Tracking: milestone4_video.py's track stage with --masks-dir, re-identification on. det_score is the detection score
the tracker keeps (0.55 in the pipeline, fixed in its code), near_frac the matching distance (5% of the scene's
diagonal, fixed in its code); reid_sim and reid_radius_frac are its options. A run with the pipeline's values runs
milestone4_video.py unchanged; any other det_score or near_frac runs a copy with those two constants made options
(patch_tracker), which is otherwise the same code. track_many.py runs the track stage for several scenes of one run
in one process (one start-up of Python, PyTorch and DINOv2), the same computation as one process per scene.
"""
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SAVE_MIN_SCORE = 0.2
DEDUPE_IOU = 0.5
LABEL = "subset-uniform16"
HERE = Path(__file__).resolve().parent
SUBSET_FILE, SUBSET_SEED = "dev_scenes_15.txt", 0
TRACKER = Path("openvocab-4D-main/trackA/milestone4_video.py")
PATCHED_TRACKER = TRACKER.with_name("milestone4_video_rerun.py")
TRACK_MANY = HERE / "track_many.py"
CURRENT = dict(det_score=0.55, near_frac=0.05, reid_sim=0.6, reid_radius_frac=0.15)  # the pipeline's tracking
RUNS = {
    "original_current": dict(prompts="original"),
    "original_s020": dict(prompts="original", det_score=0.2),
    "fixed_current": dict(prompts="fixed"),
    "fixed_s020": dict(prompts="fixed", det_score=0.2),
    "original_near10": dict(prompts="original", near_frac=0.10),
    "original_reid_loose": dict(prompts="original", reid_sim=0.5, reid_radius_frac=0.25),
    # the fixed prompts between the pipeline's score and 0.2 (the sheets are point-merged locally, merge_points.py);
    # 0.5 is the score SAM 3's masks were saved from in the pipeline, so it shows the prompt fixes on their own
    "fixed_s050": dict(prompts="fixed", det_score=0.5),
    "fixed_s040": dict(prompts="fixed", det_score=0.4),
    "fixed_s030": dict(prompts="fixed", det_score=0.3),
    # the fixed prompts without the synonyms that only added unmatched rows (SYNONYMS_DROPPED), and with track scores
    "fixed2_s050": dict(prompts="fixed_v2", det_score=0.5),
    "fixed2_s040": dict(prompts="fixed_v2", det_score=0.4),
    "pipeline_v3": dict(prompts="fixed_v2", det_score=0.4),  # frozen, see FROZEN
}
# configurations chosen on the dev scenes and not changed after: only these run on other scenes (explore200's), to
# evaluate, not to tune. Each one's tracking is its RUNS entry; its sheets are made locally by pipeline_v3.py with these
# settings: rows of tracks seen in one frame dropped below min_single_frame_score, then the point merge
# (merge_points.py, merge_threshold metres), the scale from the unmerged rows (scene_scale.SCALE_ROWS), sizes as
# scene_sheet_creator.py at sizing_commit. Chosen on the 14 scenes of dev_scenes_15.txt that every run has
# (pipeline_scorecard_rerun_fixed2_s040_min050_point_merge.md).
FROZEN = {"pipeline_v3": dict(min_single_frame_score=0.5, merge_threshold=0.5, sizing_commit="50fe543")}
# other common names of an object, chosen as plain English, never two names that are separate gold classes in the
# same dev scene ("trash bin" is ScanNet's name and "trash can" ScanNet++'s; no dev scene has both)
SYNONYMS = {
    "trash bin": ["trash can", "garbage bin"],
    "trash can": ["trash bin", "garbage bin"],
    "tv": ["television"],
    "sofa": ["couch"],
    "refrigerator": ["fridge"],
    "fridge": ["refrigerator"],
    "telephone": ["phone"],
    "nightstand": ["bedside table"],
    "bookshelf": ["bookcase"],
    "washer": ["washing machine"],
    "washing machine": ["washer"],
    "dish washer": ["dishwasher"],
    "ceiling light": ["ceiling lamp"],
    "coat rack": ["coat stand", "coat hanger"],
    "computer tower": ["desktop computer"],
    "closet": ["wardrobe"],
    "cup": ["mug"],
    "heater": ["radiator"],
    "plant": ["potted plant"],
    "headphones": ["headset"],
    "computer mouse": ["mouse"],
}
# synonyms the "fixed_v2" prompt set leaves out: on the 14 scenes of dev_scenes_15.txt that original_s020 and
# fixed_s020 both have (the same tracking, so their difference is the synonyms), each added only rows beyond the gold
# count and no named object found: "bookcase" 163 rows, "phone" 70, "bedside table" 33, "couch" 8. The others added no
# row, apart from "radiator" (14 rows, 3 beyond the gold count, 11 more named objects found), which is kept.
SYNONYMS_DROPPED = {"bookshelf": ["bookcase"], "telephone": ["phone"], "nightstand": ["bedside table"],
                    "sofa": ["couch"]}
SYNONYMS_V2 = {n: [p for p in s if p not in SYNONYMS_DROPPED.get(n, [])] for n, s in SYNONYMS.items()}
PROMPT_SETS = {"original": None, "fixed": SYNONYMS, "fixed_v2": SYNONYMS_V2}
KEEP_AS_IS = {"room"}  # the room-size questions' prompt


def settings(run):
    """A run's prompt set and tracking settings, the pipeline's where the run doesn't set them."""
    return {**CURRENT, **RUNS[run]} if isinstance(run, str) else {**CURRENT, **run}


def is_current(s):
    return all(s[k] == v for k, v in CURRENT.items() if k in ("det_score", "near_frac"))


# ---------------------------------------------------------------------------- prompts

def occurrences(phrase, text):
    return len(re.findall(r"(?<![\w-])" + re.escape(phrase) + r"(?![\w-])", text))


def fix_prompts(question, prompts, synonyms=SYNONYMS):
    """[(name, SAM 3 prompts)] for a question: its prompts with the fixes of the module docstring."""
    text = question.lower()
    names = list(prompts)
    changed = True
    while changed:  # multi-word names
        changed = False
        for a in names:
            for b in names:
                phrase = f"{a} {b}"
                if a == b or a in KEEP_AS_IS or b in KEEP_AS_IS or phrase in names:
                    continue
                n = occurrences(phrase, text)
                if n and occurrences(a, text) == n and occurrences(b, text) == n:
                    names[names.index(a)] = phrase
                    names.remove(b)
                    changed = True
                    break
            if changed:
                break
    out = []
    for name in names:
        sam = [name]
        if name not in KEEP_AS_IS and not occurrences(name, text):  # plural names
            for suffix in ("s", "es"):
                if occurrences(name + suffix, text):
                    name, sam = name + suffix, [name + suffix, name]
                    break
        if name not in KEEP_AS_IS:
            sam += synonyms.get(name, [])
        if name not in [n for n, _ in out]:
            out.append((name, list(dict.fromkeys(sam))))
    return out


def question_prompts(row, prompt_set):
    """[(name, SAM 3 prompts)] for an index.csv row."""
    prompts = [p.strip() for p in row["prompts"].split(",") if p.strip()]
    if prompt_set == "original":
        return [(p, [p]) for p in prompts]
    return fix_prompts(row["question"], prompts, PROMPT_SETS[prompt_set])


def scene_prompts(rows, prompt_set):
    """{name: SAM 3 prompts} over a scene's questions (a name gets the same SAM 3 prompts in every question)."""
    names = {}
    for row in rows:
        for name, sam in question_prompts(row, prompt_set):
            assert names.setdefault(name, sam) == sam, (name, names[name], sam)
    return names


def sam_prompts(rows, runs):
    """Every SAM 3 prompt the runs need for a scene, in a fixed order."""
    out = {}
    for prompt_set in sorted({settings(r)["prompts"] for r in runs}):
        for sam in scene_prompts(rows, prompt_set).values():
            out.update(dict.fromkeys(sam))
    return list(out)


# ---------------------------------------------------------------------------- scenes

def stratified_subset(recall, seed=SUBSET_SEED):
    """The subset of the module docstring, from {scene: recall}; sorted."""
    rng = np.random.default_rng(seed)
    out = []
    for dataset in ("arkitscenes", "scannet", "scannetpp"):
        scenes = sorted((s for s in recall if s.split("/")[0] == dataset), key=lambda s: (recall[s], s))
        if dataset == "scannetpp":
            half = len(scenes) // 2
            out += list(rng.choice(scenes[:half], 3, replace=False)) + list(rng.choice(scenes[half:], 2, replace=False))
        else:
            out += list(rng.choice(sorted(scenes), 5, replace=False))
    return sorted(str(s) for s in out)


def dev_recall(sheets):
    """{dev scene: share of its questions' named objects with a pipeline row}, by pipeline_scorecard.py."""
    sys.path.insert(0, str(HERE))
    import pipeline_scorecard
    dev = set((HERE / "dev_scenes.txt").read_text().split())
    return {r["scene"]: float(np.mean(r["found"])) for r in pipeline_scorecard.score_root(sheets, dev)}


# ---------------------------------------------------------------------------- inputs

def read_index(path, scenes):
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["scene"] in scenes]


def scene_dir(root, scene):
    return Path(root) / scene.replace("/", "__", 1)


def frames_of(scene_root):
    return sorted(p for p in (scene_root / LABEL / "scene" / "frames").iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


# ---------------------------------------------------------------------------- SAM 3

def load_detector(code_root):
    """sam3_infer.py's transformers detector: (detect(image, prompts, min_score), model name)."""
    import torch
    sys.path.insert(0, str(Path(code_root) / "src/mvp/openvocab_adapter"))
    import sam3_infer
    return sam3_infer.transformers_detector(torch)


def run_sam3(detect, model_name, frames, prompts, out):
    """sam3_infer.run for these frames and prompts from SAVE_MIN_SCORE, with an already loaded model."""
    import torch
    import common
    import sam3_infer
    from PIL import Image
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = []
    for path in frames:
        image = Image.open(path).convert("RGB")
        detections = detect(image, prompts, SAVE_MIN_SCORE)
        sam3_infer.save_frame(out, path, image, detections, prompts)
        found = {p: sum(1 for d in detections if d[3] == p) for p in prompts}
        records.append({"stem": path.stem, "file": f"{path.stem}__masks.npz", "image_sha256": common.sha256_file(path),
                        "image_hw": [image.size[1], image.size[0]], "detections": found})
    (out / "manifest.json").write_text(json.dumps({
        "format": sam3_infer.FORMAT, "model": model_name, "backend": "transformers", "prompts": prompts,
        "min_score": SAVE_MIN_SCORE, "frame_selection": "all", "gpu": torch.cuda.get_device_name(0),
        "seconds": round(time.perf_counter() - started, 1), "frames": records}, indent=1))


def missing_prompts(raw_dir, prompts):
    """The prompts a finished SAM 3 run in raw_dir lacks (all of them if there is no finished run)."""
    manifest = Path(raw_dir) / "manifest.json"
    have = set(json.loads(manifest.read_text())["prompts"]) if manifest.is_file() else set()
    return [p for p in prompts if p not in have]


def merge_sam3(base, extra, out):
    """base's SAM 3 run with extra's (other prompts, the same frames) added, in sam3_infer.py's format, into out."""
    base, extra, out = Path(base), Path(extra), Path(out)
    mb, me = (json.loads((d / "manifest.json").read_text()) for d in (base, extra))
    assert not set(mb["prompts"]) & set(me["prompts"]), "the runs share prompts"
    assert mb["min_score"] == me["min_score"] and mb["model"] == me["model"], "the runs differ in model or threshold"
    assert [(f["stem"], f["image_sha256"]) for f in mb["frames"]] == [(f["stem"], f["image_sha256"]) for f in me["frames"]]
    out.mkdir(parents=True, exist_ok=True)
    prompts = mb["prompts"] + me["prompts"]
    for rec in mb["frames"]:
        with np.load(base / rec["file"]) as a, np.load(extra / rec["file"]) as b:
            np.savez_compressed(out / rec["file"], masks=np.concatenate([a["masks"], b["masks"]]),
                                boxes=np.concatenate([a["boxes"], b["boxes"]]),
                                scores=np.concatenate([a["scores"], b["scores"]]),
                                prompt_ids=np.concatenate([a["prompt_ids"], b["prompt_ids"] + len(mb["prompts"])]),
                                prompts=np.array(prompts), source=str(a["source"]))
    frames = [{**fb, "detections": {**fb["detections"], **fe["detections"]}} for fb, fe in zip(mb["frames"], me["frames"])]
    (out / "manifest.json").write_text(json.dumps({
        **mb, "prompts": prompts, "seconds": round(mb["seconds"] + me["seconds"], 1),
        "added": mb.get("added", []) + [{"prompts": me["prompts"], "gpu": me["gpu"], "seconds": me["seconds"]}],
        "frames": frames}, indent=1))


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    return float(inter / max(1, np.logical_or(a, b).sum()))


def pool_masks(raw_dir, out_dir, names):
    """The run's masks from the SAM 3 run in raw_dir: one prompt per name ({name: SAM 3 prompts}), in sam3_infer.py's
    format. Of two detections from different SAM 3 prompts of a name that overlap (IoU >= DEDUPE_IOU), the one with
    the higher score is kept; detections of one SAM 3 prompt are kept as SAM 3 gives them, so a name with one SAM 3
    prompt has exactly its detections."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    manifest = json.loads((raw_dir / "manifest.json").read_text())
    missing = sorted({p for sam in names.values() for p in sam} - set(manifest["prompts"]))
    if missing:  # absent prompts would look like prompts SAM 3 found nothing for
        raise ValueError(f"{raw_dir} has no SAM 3 run of {missing}: delete it to run SAM 3 again with them")
    out_dir.mkdir(parents=True, exist_ok=True)
    order = list(names)
    records = []
    for rec in manifest["frames"]:
        with np.load(raw_dir / rec["file"]) as z:
            masks, boxes, scores = z["masks"], z["boxes"], z["scores"]
            raw_prompts = [str(p) for p in z["prompts"]]
            ids, source = z["prompt_ids"], str(z["source"])
        keep = []  # (index into the raw detections, name)
        for name in order:
            mine = [(i, raw_prompts[ids[i]]) for i in range(len(scores)) if raw_prompts[ids[i]] in names[name]]
            mine.sort(key=lambda x: -scores[x[0]])
            kept = []
            for i, prompt in mine:
                if any(p != prompt and iou(masks[i], masks[j]) >= DEDUPE_IOU for j, p in kept):
                    continue
                kept.append((i, prompt))
            keep += [(i, name) for i, _ in sorted(kept)]
        h, w = masks.shape[1:]
        np.savez_compressed(
            out_dir / rec["file"],
            masks=masks[[i for i, _ in keep]] if keep else np.zeros((0, h, w), bool),
            boxes=boxes[[i for i, _ in keep]] if keep else np.zeros((0, 4), np.float32),
            scores=np.array([scores[i] for i, _ in keep], np.float64),
            prompt_ids=np.array([order.index(n) for _, n in keep], np.int64),
            prompts=np.array(order), source=source)
        records.append({**rec, "detections": {n: sum(1 for _, m in keep if m == n) for n in order}})
    (out_dir / "manifest.json").write_text(json.dumps({
        **{k: v for k, v in manifest.items() if k not in ("frames", "prompts", "seconds")}, "prompts": order,
        "pooled_from": {"sam3_run": str(raw_dir.name), "prompts": names, "dedupe_iou": DEDUPE_IOU},
        "frames": records}, indent=1))


# ---------------------------------------------------------------------------- tracking

def patch_tracker(code_root):
    """milestone4_video_rerun.py next to milestone4_video.py: the same code with the detection score (0.55) and the
    matching distance (5% of the scene diagonal) as options, --det-score and --near-frac, defaulting to those values.
    It also saves each kept track's best detection score, tracks/<prompt>.scores.json ({track id: score}; two tracks
    consolidated into one keep the higher), for dropping low-score rows when the sheets are made
    (merge_points.py --min-single-frame-score); the tracks themselves are computed as before."""
    src = (Path(code_root) / TRACKER).read_text()
    edits = [
        ('    p.add_argument("--masks-dir", default=None,',
         '    p.add_argument("--det-score", type=float, default=0.55, help="detection score kept by tracking")\n'
         '    p.add_argument("--near-frac", type=float, default=0.05, help="tier-1 matching distance, of the scene diagonal")\n'
         '    p.add_argument("--masks-dir", default=None,'),
        ("saved = SavedMasks(ARGS.masks_dir, prompts, frames, 0.55)",
         "saved = SavedMasks(ARGS.masks_dir, prompts, frames, ARGS.det_score)"),
        ("if scores[k] < 0.55:", "if scores[k] < ARGS.det_score:"),
        ("eps_near = 0.05 * scene_diag", "eps_near = ARGS.near_frac * scene_diag"),
        ('                    tr["frame_set"].add(f)\n',
         '                    tr["frame_set"].add(f)\n'
         '                    tr["best_score"] = max(tr.get("best_score", 0.0), score)\n'),
        ('                        dst["frame_set"] |= tr["frame_set"]\n',
         '                        dst["frame_set"] |= tr["frame_set"]\n'
         '                        dst["best_score"] = max(dst.get("best_score", 0.0), tr.get("best_score", 0.0))\n'),
        ("""        np.savez_compressed(tracks_dir / f"{prompt.replace(' ', '_')}.npz", labels=vol)\n""",
         """        np.savez_compressed(tracks_dir / f"{prompt.replace(' ', '_')}.npz", labels=vol)\n"""
         """        (tracks_dir / f"{prompt.replace(' ', '_')}.scores.json").write_text(json.dumps(\n"""
         """            {str(t): round(float(tr.get("best_score", 0.0)), 4) for t, tr in stable.items()}))\n"""),
    ]
    for old, new in edits:
        assert src.count(old) == 1, f"{TRACKER} has changed: {old!r} found {src.count(old)} times"
        src = src.replace(old, new)
    (Path(code_root) / PATCHED_TRACKER).write_text(src)
    return Path(code_root) / PATCHED_TRACKER


def tracker_script(code_root, s):
    return Path(code_root) / (TRACKER if is_current(s) else PATCHED_TRACKER)


def tracker_args(frames_dir, scene_out, masks_dir, names, s):
    """The tracker's arguments for one scene: the track stage on the saved masks, with settings s."""
    args = ["--images", str(frames_dir), "--out", str(scene_out), "--prompts", ",".join(names), "--stages", "track",
            "--step", "1", "--masks-dir", str(masks_dir), "--reid-sim", str(s["reid_sim"]),
            "--reid-radius-frac", str(s["reid_radius_frac"])]
    if not is_current(s):
        args += ["--det-score", str(s["det_score"]), "--near-frac", str(s["near_frac"])]
    return args


def track_batch(code_root, jobs, s, work, log):
    """track_many.py: the tracker for several scenes (argument lists, tracker_args) with settings s in one process.
    Returns its results, one per job ({"out", "ok", "error", "seconds"}) and the process's wall time."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    jobs_file, results_file = work / "jobs.json", work / "results.json"
    jobs_file.write_text(json.dumps(jobs))
    results_file.unlink(missing_ok=True)
    start = time.time()
    with open(log, "a") as f:
        subprocess.run([sys.executable, "-u", str(TRACK_MANY), str(tracker_script(code_root, s)), str(jobs_file),
                        str(results_file)], stdout=f, stderr=subprocess.STDOUT)
    results = json.loads(results_file.read_text()) if results_file.is_file() else []
    results += [{"out": j[j.index("--out") + 1], "ok": False, "error": f"the process stopped, see {log}"}
                for j in jobs[len(results):]]
    return results, time.time() - start


def track(code_root, frames_dir, scene_out, masks_dir, names, s, log):
    """milestone4_video.py's track stage for these prompt names on the saved masks, with settings s, into
    scene_out/tracks/ (scene_out holds cameras.npz and depth.npz), in a process of its own."""
    cmd = [sys.executable, "-u", str(tracker_script(code_root, s)),
           *tracker_args(frames_dir, scene_out, masks_dir, names, s)]
    with open(log, "a") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        if subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode != 0:
            raise RuntimeError(f"tracking failed, see {log}")


def track_settings(s):
    """The tracking record of a run, as question.json and <prompt>.settings.json hold it."""
    return {"reid": True, "backend": "transformers", "min_score": SAVE_MIN_SCORE, "conf_percentile": 55.0,
            "min_points": 300, "min_track_frames": "linear: 1 at 16 frames, 4 at 128, at least 1",
            "det_score": s["det_score"], "near_frac": s["near_frac"], "reid_sim": s["reid_sim"],
            "reid_radius_frac": s["reid_radius_frac"]}


# ---------------------------------------------------------------------------- a run's scene

def prepare_run_scene(inputs, raw_dir, run, rows, work):
    """A run's scene folder in work/<dataset>__<scene>/ ready for tracking: cameras, depth and the run's masks.
    Returns the tracker's arguments for it (tracker_args)."""
    s = settings(run)
    scene = rows[0]["scene"]
    src, dst = scene_dir(inputs, scene), scene_dir(work, scene)
    if dst.exists():
        shutil.rmtree(dst)
    names = scene_prompts(rows, s["prompts"])
    scene_out = dst / LABEL / "scene"
    scene_out.mkdir(parents=True)
    for f in ("cameras.npz", "adapter_report.json", "depth.npz"):
        shutil.copy(src / LABEL / "scene" / f, scene_out / f)
    masks = dst / LABEL / "masks" / f"rerun_{s['prompts']}"
    pool_masks(raw_dir, masks, names)
    return tracker_args((src / LABEL / "scene" / "frames").resolve(), scene_out.resolve(), masks.resolve(), list(names),
                        s)


def write_run_scene(code_root, inputs, raw_dir, run, rows, work, log):
    """One run's outputs for one scene in work/<dataset>__<scene>/ (the layout of the module docstring), the tracker
    in a process of its own."""
    args = prepare_run_scene(inputs, raw_dir, run, rows, work)
    script = tracker_script(code_root, settings(run))
    with open(log, "a") as f:
        if subprocess.run([sys.executable, "-u", str(script), *args], stdout=f, stderr=subprocess.STDOUT).returncode:
            raise RuntimeError(f"tracking failed, see {log}")
    return finish_run_scene(inputs, run, rows, work)


def finish_run_scene(inputs, run, rows, work):
    """The rest of a run's scene folder once its tracks are made: the records of the module docstring and DONE.json.
    Returns the folder."""
    s = settings(run)
    scene = rows[0]["scene"]
    dst = scene_dir(work, scene)
    names = scene_prompts(rows, s["prompts"])
    scene_out = dst / LABEL / "scene"
    masks = dst / LABEL / "masks" / f"rerun_{s['prompts']}"
    missing = [n for n in names if not (scene_out / "tracks" / f"{n.replace(' ', '_')}.npz").is_file()]
    if missing:
        raise RuntimeError(f"{scene}: no tracks for {missing}")
    if not is_current(s):  # the tracker copy saves every kept track's best score
        for n in names:
            stem = scene_out / "tracks" / n.replace(" ", "_")
            scores = Path(f"{stem}.scores.json")
            if not scores.is_file():
                raise RuntimeError(f"{scene}: no track scores for {n!r}")
            with np.load(f"{stem}.npz") as z:
                ids = {int(t) for t in np.unique(z["labels"]) if t}
            if ids - {int(t) for t in json.loads(scores.read_text())}:
                raise RuntimeError(f"{scene}: tracks of {n!r} without a score: {sorted(ids - set(map(int, json.loads(scores.read_text()))))}")
    (scene_out / "depth.npz").unlink()  # the dense depth makes it again locally
    for name in names:
        (scene_out / "tracks" / f"{name.replace(' ', '_')}.settings.json").write_text(json.dumps(
            {"prompt": name, "masks": masks.name, "sam3_prompts": names[name], "settings": track_settings(s)},
            indent=1))
    by_folder = {}
    for row in rows:
        by_folder.setdefault(row["pipeline_sheet"], row)
        by_folder.setdefault(row["gold_sheet"], row)
    for folder, row in by_folder.items():
        old = json.loads((Path(inputs) / folder / "question.json").read_text())
        new_prompts = [n for n, _ in question_prompts(row, s["prompts"])]
        record = {**old, "prompts": new_prompts}
        if old["prompts"] != new_prompts:
            record["original_prompts"] = old["prompts"]
        out = Path(work) / folder
        out.mkdir(parents=True, exist_ok=True)
        if "/gold/" in folder:
            if old["prompts"] == new_prompts:  # the gold sheet is unchanged; otherwise it is made again locally
                for f in ("objects.csv", "sheet.json"):
                    if (Path(inputs) / folder / f).is_file():
                        shutil.copy(Path(inputs) / folder / f, out / f)
        else:
            record["tracking"] = track_settings(s)
        (out / "question.json").write_text(json.dumps(record, indent=1))
    (dst / "DONE.json").write_text(json.dumps({"run": run, "settings": s, "scene": scene, "names": names}, indent=1))
    return dst


def write_index(index_rows, run, path):
    """index.csv of a run: the dev scenes' rows, prompts as the run uses them."""
    s = settings(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index_rows[0]))
        w.writeheader()
        for row in index_rows:
            w.writerow({**row, "prompts": ", ".join(n for n, _ in question_prompts(row, s["prompts"]))})


def publish(local, final):
    """Copy a finished folder to Drive under a temporary name, then rename it, so a half-copied folder is never
    taken for a finished one."""
    final = Path(final)
    tmp = final.with_name(final.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    if final.exists():
        shutil.rmtree(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(local, tmp)
    os.replace(tmp, final)


# ---------------------------------------------------------------------------- checks

def compare_masks(saved_dir, raw_dir, min_score=0.5, match_iou=0.9):
    """The saved SAM 3 run (from min_score) against the new one's detections from min_score, for the saved run's
    prompts: per detection, whether a new one of the same prompt in the same frame matches it (IoU >= match_iou)."""
    saved_dir, raw_dir = Path(saved_dir), Path(raw_dir)
    prompts = json.loads((saved_dir / "manifest.json").read_text())["prompts"]
    n_saved = n_new = matched = 0
    score_diff = []
    for f in sorted(saved_dir.glob("*__masks.npz")):
        with np.load(f) as a, np.load(raw_dir / f.name) as b:
            pa, pb = [str(p) for p in a["prompts"]], [str(p) for p in b["prompts"]]
            for p in prompts:
                ia = [i for i in range(len(a["scores"])) if pa[a["prompt_ids"][i]] == p]
                ib = [i for i in range(len(b["scores"])) if pb[b["prompt_ids"][i]] == p and b["scores"][i] >= min_score]
                n_saved, n_new = n_saved + len(ia), n_new + len(ib)
                free = set(ib)
                for i in ia:
                    best = max(free, key=lambda j: iou(a["masks"][i], b["masks"][j]), default=None)
                    if best is not None and iou(a["masks"][i], b["masks"][best]) >= match_iou:
                        matched += 1
                        free.discard(best)
                        score_diff.append(abs(float(a["scores"][i]) - float(b["scores"][best])))
    return {"saved": n_saved, "new": n_new, "matched": matched,
            "max_score_diff": max(score_diff, default=0.0)}


def compare_tracks(saved_tracks, new_tracks, prompts):
    """Per prompt: whether the label volumes are identical, else the share of pixels labelled alike (up to the
    tracks' numbering: each saved track id is mapped to the new id it overlaps most)."""
    out = {}
    for p in prompts:
        f = f"{p.replace(' ', '_')}.npz"
        with np.load(Path(saved_tracks) / f) as a, np.load(Path(new_tracks) / f) as b:
            x, y = a["labels"], b["labels"]
        if np.array_equal(x, y):
            out[p] = "identical"
            continue
        fg = (x > 0) | (y > 0)
        agree = 0
        for t in np.unique(x[x > 0]):
            ids, counts = np.unique(y[x == t], return_counts=True)
            agree += counts[ids != 0].max() if (ids != 0).any() else 0
        out[p] = f"{agree / max(1, fg.sum()):.1%} of labelled pixels alike"
    return out


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


if __name__ == "__main__":  # SUBSET_FILE from the pipeline sheets: python rerun_detection.py [data/scene_sheets_75]
    sheets = sys.argv[1] if len(sys.argv) > 1 else str(HERE.parents[1] / "data" / "scene_sheets_75")
    recall = dev_recall(sheets)
    subset = stratified_subset(recall)
    (HERE / SUBSET_FILE).write_text("\n".join(subset) + "\n")
    for s in subset:
        print(f"{s}  recall {recall[s]:.2f}")
