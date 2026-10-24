"""A few examples from each of SpatialStack's training sets, to get a feel for the data.

Writes docs/training_data_examples.md and its images in docs/training_data_examples/. SPAR's points and boxes are
drawn with SpatialStack's own draw_marker.py, as its training loader does. Light enough for the login node.

  python -m mvp.reporting.train_examples
"""
import importlib.util
import json
import random
import re
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

from mvp.paths import ROOT, TRAIN_ANN, TRAIN_MEDIA
from mvp.reporting.examples import frame_grid

MD = ROOT / "docs" / "training_data_examples.md"
OUT = ROOT / "docs" / "training_data_examples"  # images

# VLM-3R ScanNet QA has no task field; its questions are VSI-Bench templates, so keywords identify the task
VLM3R_TASKS = [("route_plan", ["route", "navigate"]), ("room_size", ["size of this room"]),
               ("abs_distance", ["direct distance", "distance between"]),
               ("rel_distance", ["closest to", "farthest", "closest"]),
               ("rel_direction", ["to my left", "left or the right", "left, right", "front-left", "quadrant"]),
               ("counting", ["how many"]), ("obj_size", ["longest dimension"])]

SPAR_PICKS = [  # (source, task type)
    ("scannet", "distance_prediction_oc"), ("scannet", "obj_spatial_relation_oo_mv"),
    ("scannetpp", "depth_prediction_oo_mv"), ("scannetpp", "obj_count"), ("scannet", "distance_prediction_oo_video"),
    ("scannet", "spatial_imagination_oc_video"), ("scannetpp", "distance_infer_center_oo_video"),
    ("scannetpp", "spatial_imagination_oo_video_hard"),
    ("structured3d", "depth_prediction_oc"), ("structured3d", "distance_prediction_oo_mv"),
    ("structured3d", "position_matching"), ("structured3d", "view_change_infer"),
    ("structured3d", "spatial_imagination_oc_mv"), ("structured3d", "spatial_volume_infer"),
]


def load_draw_functions():
    path = ROOT / "third_party/SpatialStack/src/qwen_vl/data/draw_marker.py"
    spec = importlib.util.spec_from_file_location("draw_marker", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DRAW_FUNCTIONS


def text(s):
    """One line of markdown: the 32 'Frame-k: <image>' lines collapse to '[32 frames]', media tags are dropped."""
    s = re.sub(r"(Frame-\d+: <image>\n)+", lambda m: f"[{m.group(0).count(chr(10))} frames] ", s)
    return s.replace("<image>", "").replace("<video>", "").strip().replace("\n", "<br>")


def tiles(ims, cols, width, labels=None):
    """Paste images into a grid, each resized to `width`, with an optional label in the top-left corner."""
    ims = [im.convert("RGB").resize((width, int(im.height * width / im.width))) for im in ims]
    h = max(im.height for im in ims)
    grid = Image.new("RGB", (cols * width, (len(ims) + cols - 1) // cols * h), "white")
    for k, im in enumerate(ims):
        if labels:
            d = ImageDraw.Draw(im)
            d.rectangle((0, 0, 8 * len(labels[k]) + 6, 16), fill="black")
            d.text((3, 2), labels[k], fill="white")
        grid.paste(im, (k % cols * width, k // cols * h))
    return grid


def qa(r):
    return [f"**Q:** {text(r['conversations'][0]['value'])}", "", f"**A:** {text(r['conversations'][1]['value'])}", ""]


def scannet_section(rng):
    md = ["## VLM-3R ScanNet QA", "", "VSI-Bench templates on 1,201 ScanNet videos; one example per task. "
          "8 uniform frames shown.", ""]
    data = json.load(open(TRAIN_ANN / "merged_qa_scannet_train.json"))
    by_task = defaultdict(list)
    for r in data:
        q = r["conversations"][0]["value"].lower()
        by_task[next((t for t, keys in VLM3R_TASKS if any(k in q for k in keys)), "other")].append(r)
    for task, rs in sorted(by_task.items(), key=lambda kv: -len(kv[1])):
        r = rng.choice(rs)
        name = f"vlm3r_{task}.jpg"
        frame_grid(str(TRAIN_MEDIA / r["video"])).save(OUT / name, quality=85)
        md += [f"### {task} ({len(rs):,} samples): `{r['video']}`", "", f"![]({OUT.name}/{name})", "", *qa(r)]

    md += ["## VSI-590K appearance order", "", "Same ScanNet videos. The annotations' paths lack `videos/`, "
           "so they are rewritten here.", ""]
    data = json.load(open(TRAIN_ANN / "vsi_appearance_order_vsibench_scannet.json"))
    for k, r in enumerate(rng.sample(data, 2)):
        video = f"scannet/videos/{r['video'].split('/')[-1]}"
        name = f"appearance_order_{k}.jpg"
        frame_grid(str(TRAIN_MEDIA / video)).save(OUT / name, quality=85)
        md += [f"### `{video}`", "", f"![]({OUT.name}/{name})", "", *qa(r)]
    return md


def spar_section(rng):
    md = ["## SPAR-234k", "", "Points and boxes drawn as in training. The 32-frame samples only mark the frames the "
          "question refers to; those are shown (labelled `*`), padded with uniform frames up to 8.", ""]
    draw = load_draw_functions()
    by = defaultdict(list)
    for r in json.load(open(TRAIN_ANN / "spar_234k.json")):
        by[(r["images"][0].split("/")[1], json.loads(r["spar_info"])["type"])].append(r)
    for src, task in SPAR_PICKS:
        r = rng.choice(by[(src, task)])
        ims = [Image.open(TRAIN_MEDIA / p).convert("RGB") for p in r["images"]]
        before = [np.asarray(im).copy() for im in ims]
        fn = draw[task]
        fn(ims[0], json.loads(r["spar_info"])) if len(ims) == 1 else fn(ims, json.loads(r["spar_info"]))
        marked = [i for i, (a, im) in enumerate(zip(before, ims)) if not np.array_equal(a, np.asarray(im))]
        if len(ims) > 4:
            uniform = [i for i in np.linspace(0, len(ims) - 1, 8, dtype=int) if i not in marked]
            idx = sorted(marked + uniform[:max(0, 8 - len(marked))])
            grid = tiles([ims[i] for i in idx], 4, 320, [f"{i}{'*' * (i in marked)}" for i in idx])
        else:
            grid = tiles(ims, len(ims), 320 if len(ims) > 1 else 480)
        name = f"spar_{src}_{task}.jpg"
        grid.save(OUT / name, quality=85)
        note = ["*No markers: the `_hard` tasks name objects only by category.*", ""] if task.endswith("_hard") else []
        md += [f"### {task} ({src}, {len(ims)} image{'s' * (len(ims) > 1)})", "", f"![]({OUT.name}/{name})", "",
               *note, *qa(r)]
    return md


def hound_section(rng):
    md = ["## LLaVA-Hound-64k", "", "General video QA on web videos (no spatial content). 8 uniform frames shown.", ""]
    for k, r in enumerate(rng.sample(json.load(open(TRAIN_ANN / "llava_hound_64k.json")), 4)):
        d = TRAIN_MEDIA / r["video"]
        frames = sorted(d.iterdir())
        name = f"llava_hound_{k}.jpg"
        tiles([Image.open(frames[i]) for i in np.linspace(0, len(frames) - 1, 8, dtype=int)], 4, 320).save(
            OUT / name, quality=85)
        md += [f"### `{r['video']}` ({len(frames)} frames)", "", f"![]({OUT.name}/{name})", "", *qa(r)]
    return md


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    md = ["# Training data examples", "", "SpatialStack's training mix (annotations in `data/raw/SpatialStackData/`, "
          "media in `data/train_media/`). Written by `python -m mvp.reporting.train_examples`.", "",
          *scannet_section(rng), *spar_section(rng), *hound_section(rng)]
    MD.write_text("\n".join(md))
    print(f"wrote {MD}")


if __name__ == "__main__":
    main()
