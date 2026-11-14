"""Screen candidate questions for the detailed analysis: SAM 3 on each question's 32 frames, frames in which each named
object is detected (score >= thr, at a low threshold too), so the detailed examples have their objects in view.

  python -m mvp.explain.screen --ids 1076 1221 ... --out results/detail_v10/screen.json
"""
import argparse, json, sys
sys.path.insert(0, "src")
import numpy as np
from mvp.benchmarks.vsibench import load_vsibench
from mvp.fusion.inputs import EVAL_PIXELS, InputBuilder, uniform_frames
from mvp.explain import one_question as OQ

ap = argparse.ArgumentParser()
ap.add_argument("--ids", type=int, nargs="+", required=True)
ap.add_argument("--thr", type=float, default=0.3)
ap.add_argument("--out", required=True)
args = ap.parse_args()
from transformers import AutoProcessor
builder = InputBuilder(AutoProcessor.from_pretrained("Qwen/Qwen3.5-2B", **EVAL_PIXELS))
df = load_vsibench("full").set_index("id", drop=False)
res = {}
for i in args.ids:
    doc = dict(df.loc[i])
    try:
        phrases = OQ.phrases_of(doc)
    except SystemExit:
        continue
    frames, _ = uniform_frames(doc["video_path"], 32)
    grids = builder.visual(frames)["image_grid_thw"].tolist()
    _, pix, det = OQ.object_masks_sam3(frames, phrases, grids, args.thr, 3, "cuda")
    res[i] = {p: {"f30": sum(1 for d in det[p] if d), "f50": sum(1 for d in det[p] if d and d[0][0] >= 0.5),
                  "best": max((d[0][0] for d in det[p] if d), default=0)} for p in phrases}
    print(i, doc["question_type"], {p: (v["f50"], v["f30"]) for p, v in res[i].items()}, flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
