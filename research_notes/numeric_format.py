"""Are full generated numeric answers single digits? Format and magnitude of predictions vs truth, per numeric task, from the evaluation files."""
import json, re
from collections import Counter
import numpy as np

R = "/shared/workspace/povejmo/math/projects/mvp/results/train"
NUM = ["object_counting", "object_abs_distance", "object_size_estimation", "room_size_estimation"]


def fmt(s):
    s = str(s).strip()
    if re.fullmatch(r"\d", s): return "single digit"
    if re.fullmatch(r"\d+", s): return "multi-digit integer"
    if re.fullmatch(r"\d+\.\d+", s): return "decimal"
    return "other"


for run in ("2B_v9_fmix_additive_s0", "2B_v9_fmix_lora_s1", "2B_v9_fmix_xa3_both_s0"):
    rows = [json.loads(l) for l in open(f"{R}/{run}/branches/step5750/eval/vsi_f32_full.jsonl")]
    print("==", run)
    for t in NUM:
        x = [r for r in rows if r["task"] == t]
        P, G = Counter(fmt(r["prediction"]) for r in x), Counter(fmt(r["ground_truth"]) for r in x)
        p = np.array([float(re.search(r"\d+(\.\d+)?", str(r["prediction"])).group()) if re.search(r"\d", str(r["prediction"])) else np.nan for r in x]); g = np.array([float(r["ground_truth"]) for r in x])
        print(f"{t:24} n{len(x)} pred {dict(P)} | truth {dict(G)} | median pred {np.nanmedian(p):.1f} truth {np.median(g):.1f} | exact {100 * np.mean(p == g):.1f}% | top {Counter(str(r['prediction']).strip() for r in x).most_common(5)}")
# the J-lens questions: first token vs full answer
ids = {1723, 32, 181, 542}
for r in map(json.loads, open(f"{R}/2B_v7_additive_s0/branches/step10750/eval/vsi_f32_full.jsonl")):
    if r["id"] in ids:
        print("v7 additive step10750 id", r["id"], r["task"], "truth", r["ground_truth"], "full answer", r["prediction"], "score", r["score"])
