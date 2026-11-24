"""The object-size questions of explore200 whose v3 sheet has the object: the size each sheet variant implies against
the truth, the video-only answer and the v3 answer, to separate the scale, the merge and the box shape."""
import csv, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, sys.argv[2])
import explore_sheets as es
from mvp.benchmarks import scene_scale
from mvp.benchmarks.vsibench import final_number

R, E, M = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
D = M / "data/rerun_sheets"
roots = {"v3": D / "pipeline_v3_explore200", "v3 unmerged": D / "pipeline_v3_explore200_unmerged",
         "oriented": D / "pipeline_v3_oriented_explore200", "oriented unmerged": D / "pipeline_v3_oriented_explore200_unmerged"}
idx = {int(r["id"]): r for r in csv.DictReader(open(roots["v3"] / "index.csv"))}
v = pd.read_json(R / "A_video.jsonl", lines=True).set_index("id")
p3 = pd.read_json(R / "pipeline_v3.jsonl", lines=True).set_index("id")
mra = lambda p, g: float(np.mean([abs(p - g) / g < 1 - t for t in np.arange(0.5, 1.0, 0.05)])) if p and np.isfinite(p) else 0.0

def rows(root, q, c):
    return [r for r in csv.DictReader(open(root / idx[q]["pipeline_sheet"] / "objects.csv")) if r["concept"] == c]
longest = lambda rs: max(max(float(r[k]) for k in ("size_x", "size_y", "size_z")) for r in rs) if rs else np.nan

qs = [q for q in v.index if v.loc[q, "question_type"] == "object_size_estimation"]
scenes = sorted({idx[q]["scene"] for q in qs})
oracle = es.oracle_scales(str(M / "data/scene_sheets_75"), scenes)
out = []
for q in qs:
    c = [p.strip() for p in idx[q]["prompts"].split(",") if p.strip() != "room"][0]
    if not rows(roots["v3"], q, c):
        continue
    g = float(idx[q]["ground_truth"])
    k3 = scene_scale.question_scale(q, str(roots["v3"]))["k"]
    lo, cls = scene_scale.prior_ratio(scene_scale.scene_longest(str(roots["oriented"]), idx[q]["scene"]), {c})
    k_or = 0.8756 * lo if cls else scene_scale.FALLBACK
    ko = oracle[idx[q]["scene"]]["scale"]
    L = {n: longest(rows(r, q, c)) for n, r in roots.items()}
    rec = {"id": q, "object": c, "truth": g, "video": final_number(v.loc[q, "prediction"]),
           "v3 answer": final_number(p3.loc[q, "prediction"]),
           "sheet v3": 100 * k3 * L["v3"], "v3 unmerged": 100 * k3 * L["v3 unmerged"],
           "v3, oracle scale": 100 * ko * L["v3"], "oriented 0.8756": 100 * k_or * L["oriented"],
           "oriented, oracle scale": 100 * ko * L["oriented"], "oriented unmerged 0.8756": 100 * k_or * L["oriented unmerged"],
           "rows": len(rows(roots["v3"], q, c)), "priors / oracle scale": k3 / ko,
           "oracle reliable": oracle[idx[q]["scene"]].get("reliable", None)}
    out.append(rec)
df = pd.DataFrame(out).set_index("id")
cands = ["video", "v3 answer", "sheet v3", "v3 unmerged", "v3, oracle scale", "oriented 0.8756", "oriented, oracle scale",
         "oriented unmerged 0.8756"]
print(f"{len(df)} object-size questions with the object in the v3 sheet\n")
print("| | " + " | ".join(cands) + " |\n|---|" + "---:|" * len(cands))
print("| mean MRA vs truth (x100) | " + " | ".join(f"{100 * np.mean([mra(df.loc[q, c], df.loc[q, 'truth']) for q in df.index]):.1f}" for c in cands) + " |")
print("| median ratio to truth | " + " | ".join(f"{np.nanmedian(df[c] / df['truth']):.2f}" for c in cands) + " |")
print("| over / under truth | " + " | ".join(f"{int((df[c] > df['truth']).sum())} / {int((df[c] < df['truth']).sum())}" for c in cands) + " |")
f = (df["v3 answer"] / df["sheet v3"] - 1).abs()
print(f"\nv3's answer within 10% of the sheet's size: {int((f < 0.1).sum())} of {len(df)}; within 25%: {int((f < 0.25).sum())}")
print(f"priors / oracle scale: median {df['priors / oracle scale'].median():.2f}, range {df['priors / oracle scale'].min():.2f}-{df['priors / oracle scale'].max():.2f}")
print(f"rows of the object in the v3 sheet: median {int(df['rows'].median())}, max {df['rows'].max()}")
pd.set_option("display.width", 250)
print("\n" + df[["object", "truth", "video", "v3 answer", "sheet v3", "v3 unmerged", "v3, oracle scale", "oriented 0.8756", "rows", "priors / oracle scale"]].round(2).to_string())
