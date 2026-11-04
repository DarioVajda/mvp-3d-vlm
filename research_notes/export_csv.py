"""Export question-level VSI-Bench results to CSV: one row per question x model run x checkpoint x frame count, plus a per-run summary."""
import csv, glob, json, os, re
from collections import defaultdict
import numpy as np

ROOT = "/shared/workspace/povejmo/math/projects/mvp"
R = f"{ROOT}/results/train"
OUT = f"{ROOT}/.claude/worktrees/je-xattn-proposal/research_notes/exports"
os.makedirs(OUT, exist_ok=True)
TASKS = ["object_counting", "object_abs_distance", "object_size_estimation", "room_size_estimation", "object_rel_distance", "object_rel_direction", "route_planning", "obj_appearance_order"]
NUM = {"object_counting", "object_abs_distance", "object_size_estimation", "room_size_estimation"}
COLS = ["run", "version", "arm", "seed", "checkpoint", "eval_frames", "id", "dataset", "scene_name", "task", "question_type", "answer_type", "n_options", "question", "options", "ground_truth", "prediction", "score", "correct_mc", "gt_letter", "pred_letter"]
letter = lambda s: (str(s).strip()[0].upper() if str(s).strip() and str(s).strip()[0].upper() in "ABCD" else "")
ARM = {"lora": "lora_only", "additive": "additive", "xa3_both": "cross_attention", "hybrid": "hybrid", "xa3_late": "cross_attention_late_layers", "xa3_deep": "cross_attention_six_layers"}
files = []
for p in sorted(glob.glob(f"{R}/2B_v[789]_*/branches/*/eval/vsi_f*_full.jsonl")):
    m = re.search(r"train/(2B_(v\d)_(?:fmix_)?(.+?)_s(\d))/branches/(step\d+)/eval/vsi_f(\d+)_full", p)
    if not m or "crashed" in p:
        continue
    run, ver, arm, seed, step, fr = m.groups()
    if step not in ("step5750", "step10750"):
        continue
    n = sum(1 for _ in open(p))
    if n != 5130:
        print("skipped, incomplete:", run, step, f"f{fr}", n)
        continue
    files.append((p, run, ver, ARM.get(arm, arm), seed, step, fr))
rows_summary = []
with open(f"{OUT}/vsi_full_results.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, COLS); w.writeheader()
    for p, run, ver, arm, seed, step, fr in files:
        per, ds = defaultdict(list), defaultdict(lambda: defaultdict(list))
        for x in map(json.loads, open(p)):
            mc = x["task"] not in NUM
            w.writerow({"run": run, "version": ver, "arm": arm, "seed": seed, "checkpoint": step, "eval_frames": fr, "id": x["id"], "dataset": x["dataset"], "scene_name": x["scene_name"], "task": x["task"], "question_type": x["question_type"],
                        "answer_type": "multiple_choice" if mc else "numeric", "n_options": len(x["options"]) if mc else "", "question": x["question"], "options": " | ".join(x["options"]) if mc else "", "ground_truth": x["ground_truth"], "prediction": x["prediction"], "score": x["score"],
                        "correct_mc": int(x["score"] == 1) if mc else "", "gt_letter": letter(x["ground_truth"]) if mc else "", "pred_letter": letter(x["prediction"]) if mc else ""})
            per[x["task"]].append(x["score"]); ds[x["dataset"]][x["task"]].append(x["score"])
        rows_summary.append({"run": run, "version": ver, "arm": arm, "seed": seed, "checkpoint": step, "eval_frames": fr, "n": sum(len(v) for v in per.values()), "overall": round(100 * float(np.mean([np.mean(per[t]) for t in TASKS])), 2),
                             **{t: round(100 * float(np.mean(per[t])), 2) for t in TASKS}, **{f"overall_{d}": round(100 * float(np.mean([np.mean(v) for v in t.values()])), 2) for d, t in ds.items()}})
    # SpatialStack reproduction
    per, ds = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for r in map(json.loads, open(glob.glob(f"{ROOT}/results/baselines/spatialstack/full_qwen35/vsibench/*/*samples_vsibench.jsonl")[0])):
        s = r["vsibench_score"]; t = re.sub(r"_(easy|medium|hard)$", "", s["question_type"]); mc = t not in NUM
        sc = s.get("accuracy") if mc else s.get("MRA:.5:.95:.05")
        w.writerow({"run": "SpatialStack-Qwen3.5-4B", "version": "baseline", "arm": "spatialstack_4B", "seed": "", "checkpoint": "released", "eval_frames": 32, "id": s["id"], "dataset": s["dataset"], "scene_name": s["scene_name"], "task": t, "question_type": s["question_type"],
                    "answer_type": "multiple_choice" if mc else "numeric", "n_options": len(s["options"]) if mc and s.get("options") else "", "question": s["question"], "options": " | ".join(s["options"]) if mc and s.get("options") else "", "ground_truth": s["ground_truth"], "prediction": s["prediction"], "score": sc,
                    "correct_mc": int(sc == 1) if mc else "", "gt_letter": letter(s["ground_truth"]) if mc else "", "pred_letter": letter(s["prediction"]) if mc else ""})
        per[t].append(float(sc)); ds[s["dataset"]][t].append(float(sc))
    rows_summary.append({"run": "SpatialStack-Qwen3.5-4B", "version": "baseline", "arm": "spatialstack_4B", "seed": "", "checkpoint": "released", "eval_frames": 32, "n": sum(len(v) for v in per.values()), "overall": round(100 * float(np.mean([np.mean(per[t]) for t in TASKS])), 2),
                         **{t: round(100 * float(np.mean(per[t])), 2) for t in TASKS}, **{f"overall_{d}": round(100 * float(np.mean([np.mean(v) for v in t.values()])), 2) for d, t in ds.items()}})
keys = list(rows_summary[0].keys())
with open(f"{OUT}/vsi_summary_by_run.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows_summary)
for r in rows_summary:
    print(r["run"], r["checkpoint"], "f" + str(r["eval_frames"]), r["overall"])
print("rows", sum(r["n"] for r in rows_summary), "MB", round(os.path.getsize(f"{OUT}/vsi_full_results.csv") / 1e6, 1))
