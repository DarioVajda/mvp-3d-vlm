"""Per-task VSI of the local SpatialStack-Qwen3.5-4B reproduction (lmms-eval samples) and zero-shot images-protocol summaries."""
import glob
import json
from collections import defaultdict

R = "/shared/workspace/povejmo/math/projects/mvp/results"
p = f"{R}/baselines/spatialstack/full_qwen35/vsibench/snapshots__777b2289252f7c78628e0f8ac63ffddf50fe7f7b/20261022_065438_samples_vsibench.jsonl"
rows = [json.loads(l) for l in open(p)]
print("n samples", len(rows), "keys:", sorted(rows[0].keys())[:20])
doc = rows[0].get("doc", {})
print("doc keys:", sorted(doc.keys())[:20])
by = defaultdict(list)
for r in rows:
    d = r["doc"]
    t = d.get("question_type") or d.get("task")
    sc = r["vsibench_score"]
    s = sc.get("MRA:.5:.95:.05")
    if s is None:
        s = sc.get("accuracy")
    if s is None:
        s = next((v for k, v in sc.items() if k not in d and isinstance(v, (int, float))), None)
    if s is None:
        print("no score key in", sorted(sc.keys()))
        break
    by[t].append(float(s))
per = {t: 100 * sum(v) / len(v) for t, v in by.items()}
print("SpatialStack-Qwen3.5-4B per task:", {t: round(v, 1) for t, v in sorted(per.items())})
print("mean of tasks:", round(sum(per.values()) / len(per), 2), "counts:", {t: len(v) for t, v in by.items()})
for q in sorted(glob.glob(f"{R}/zeroshot/images/**/*summary.json", recursive=True)):
    d = json.load(open(q))
    keep = ("n", "nframes", "overall", "model", "object_counting", "object_abs_distance", "object_size_estimation", "room_size_estimation", "object_rel_distance", "object_rel_direction", "route_planning", "obj_appearance_order")
    print("==", q.replace(R + "/", ""), {k: (round(v, 1) if isinstance(v, float) else v) for k, v in d.items() if k in keep})
