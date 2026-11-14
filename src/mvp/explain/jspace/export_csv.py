"""Export the J-lens readouts (decayed step-10750 v7 branches) to CSV: one row per model x question x lens x layer x cell, plus the lens checks.
Ranks are converted from the files' 0-based values to 1-based (1 = top token), matching the J-Lens page."""
import csv, gzip, json, os
from collections import Counter
import numpy as np

from mvp.explain import ROOT
SRC = f"{ROOT}/results/jspace_v7"
OUT = f"{ROOT}/results/jspace_v7/exports"
os.makedirs(OUT, exist_ok=True)
MODELS = {"lora_s0": "lora_only", "additive_s0": "additive", "xa3_both_s0": "cross_attention"}
CELLS = {"top": "additive_salient", "own": "own_salient", "rand": "random", "text": "last_12_prompt_tokens"}
COLS = ["model", "arm", "checkpoint", "id", "dataset", "scene_name", "question_type", "question", "options", "ground_truth", "eval_full_answer", "eval_score", "lens_first_token",
        "lens", "layer", "cell_set", "cell", "frame", "row", "col", "text_pos_from_end", "token_at_position",
        "best_spatial_rank", "best_spatial_word", "best_question_rank", "best_question_word", "gt_token_rank",
        "raw_top5", "contrast_top5",
        "jspace_r2", "jspace_top5",
        "delta_norm", "delta_ratio", "delta_best_spatial_rank", "delta_best_question_rank", "delta_gt_rank", "delta_top5",
        "raw_json", "contrast_json", "ranks_json", "jspace_json", "delta_top_json", "delta_ranks_json"]


def best(ranks, prefix):
    v = {k[len(prefix):]: r + 1 for k, r in ranks.items() if k.startswith(prefix)}
    if not v:
        return "", ""
    w = min(v, key=v.get)
    return v[w], w


top5 = lambda pairs: " | ".join(p[0].strip() for p in pairs[:5])
dump = lambda o: json.dumps(o, ensure_ascii=False)
n, check = 0, []
kinds = Counter()
with gzip.open(f"{OUT}/jspace_readouts.csv.gz", "wt", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, COLS); w.writeheader()
    for m, arm in MODELS.items():
        ev = {x["id"]: x for x in map(json.loads, open(f"{ROOT}/results/train/2B_v7_{m}/branches/step10750/eval/vsi_f32_full.jsonl"))}
        recs = [json.loads(l) for l in open(f"{SRC}/{m}/readouts.jsonl")]
        # text-token records are stored in prompt order, 12 per question x lens x layer
        total = Counter((r["id"], r["lens"], r["layer"]) for r in recs if r["kind"] == "text"); seen = Counter()
        for r in recs:
            e = ev[r["id"]]; ranks = r["ranks"]; place = r.get("place") or ["", "", ""]
            sp, spw = best(ranks, "sp:"); q, qw = best(ranks, "q:")
            pos = ""
            if r["kind"] == "text":
                key = (r["id"], r["lens"], r["layer"]); seen[key] += 1; pos = total[key] - seen[key]  # 0 = final prompt token
            row = {"model": f"2B_v7_{m}", "arm": arm, "checkpoint": "step10750_decayed_branch", "id": r["id"], "dataset": e["dataset"], "scene_name": e["scene_name"], "question_type": r["type"],
                   "question": e["question"], "options": " | ".join(e["options"]) if e["options"] else "", "ground_truth": e["ground_truth"], "eval_full_answer": e["prediction"], "eval_score": e["score"], "lens_first_token": r["answer"],
                   "lens": r["lens"], "layer": r["layer"], "cell_set": CELLS[r["kind"]], "cell": r["cell"] if r["kind"] != "text" else "", "frame": place[0], "row": place[1], "col": place[2], "text_pos_from_end": pos, "token_at_position": r["tok"],
                   "best_spatial_rank": sp, "best_spatial_word": spw, "best_question_rank": q, "best_question_word": qw, "gt_token_rank": ranks["gt"] + 1 if "gt" in ranks else "",
                   "raw_top5": top5(r["raw"]), "contrast_top5": top5(r["contrast"]) if r.get("contrast") else "", "raw_json": dump(r["raw"]), "contrast_json": dump(r.get("contrast", [])), "ranks_json": dump({k: v + 1 for k, v in ranks.items()})}
            if "jspace" in r:
                row.update(jspace_r2=r["jspace"]["r2"], jspace_top5=top5(r["jspace"]["top"]), jspace_json=dump(r["jspace"]["top"]))
            if "delta" in r:
                d = r["delta"]; ds, _ = best(d["ranks"], "sp:"); dq, _ = best(d["ranks"], "q:")
                row.update(delta_norm=d["norm"], delta_ratio=d["ratio"], delta_best_spatial_rank=ds, delta_best_question_rank=dq, delta_gt_rank=d["ranks"]["gt"] + 1 if "gt" in d["ranks"] else "", delta_top5=top5(d["top"]), delta_top_json=dump(d["top"]), delta_ranks_json=dump({k: v + 1 for k, v in d["ranks"].items()}))
            w.writerow(row); n += 1; kinds[(m, r["kind"])] += 1
            if m == "additive_s0" and r["lens"] == "visual" and r["layer"] == 14 and r["kind"] == "top":
                check.append(q)
            if r["kind"] == "text" and pos == 0 and r["id"] == 542 and r["lens"] == "text" and r["layer"] == 20:
                print(m, "id 542: lens first token", repr(r["answer"]), "| eval answer", e["prediction"], "| truth", e["ground_truth"])
print("rows", n, dict(kinds))
print("check: additive, salient, L14 median best question rank", np.median(check), "(J-Lens page shows 44)")
with open(f"{OUT}/jspace_lens_checks.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["model", "check", "lens", "layer", "metric_or_prompt", "value"])
    for m in MODELS:
        for lens, layers in json.load(open(f"{SRC}/{m}/lens_quality.json")).items():
            for l, v in layers.items():
                w.writerow([m, "lens_quality", lens, l, "median_rank_of_next_token", dump(v)])
        for s in json.load(open(f"{SRC}/{m}/sanity.json")):
            w.writerow([m, "sanity_prompt", s["lens"], s["layer"], s["prompt"], top5(s["top"])])
print("MB", round(os.path.getsize(f"{OUT}/jspace_readouts.csv.gz") / 1e6, 1), "| checks rows", sum(1 for _ in open(f"{OUT}/jspace_lens_checks.csv")) - 1)
