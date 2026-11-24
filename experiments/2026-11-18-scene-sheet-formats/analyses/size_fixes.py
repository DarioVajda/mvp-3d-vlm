"""Do trimmed boxes make object sizes and room sizes more accurate? Dev scenes only (the 50 of dev_scenes.txt).
Object size: the named object's row seen in the most frames (then the larger box), its longest side in cm; room size:
the largest `room` row's floor area (the two box axes least aligned with gravity) in m^2. Scored by VSI-Bench's MRA
against the truth. Scales as relation_fixes: (1) priors, (6) median of priors / room height / door height, and the
oracle (diagnostic, uses gold); every scale is computed from the current (untrimmed) sheets."""
import csv, runpy, sys
from pathlib import Path
import numpy as np, pandas as pd

M, E, S = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
g = runpy.run_path(str(Path(__file__).with_name("box_fixes.py")), run_name="box_fixes_lib") if False else None
sys.argv = ["relation_fixes.py", str(M), str(E), str(S)]
rf = runpy.run_path(str(Path(__file__).with_name("relation_fixes.py")))
mra, read_index, scales, geo, scene_geometry = rf["mra"], rf["read_index"], rf["scales"], rf["geo"], rf["scene_geometry"]
D, B = M / "data/rerun_sheets", S / "boxfix"
dev = set((E / "dev_scenes.txt").read_text().split()); dev15 = set((E / "dev_scenes_15.txt").read_text().split())
base = D / "pipeline_v3_dedup2_dev50"
def root_for(variant, s):
    if variant == "base": return base
    return B / (f"dedup_fixed2_s040_t{variant[4:]}" if s in dev15 else f"dedup_pipeline_v3_t{variant[4:]}")
VARIANTS = [("current boxes (min-max)", "base"), ("trimmed 2nd-98th", "trim2"), ("trimmed 5th-95th", "trim5"), ("trimmed 10th-90th", "trim10")]
_ix = {}
def ix(root):
    if root not in _ix: _ix[root] = read_index(root)
    return _ix[root]
idx = ix(base)
qs = {t: [q for q, r in idx.items() if r["scene"] in dev and r["question_type"] == t] for t in ("object_size_estimation", "room_size_estimation")}
for s in dev:
    if s not in geo: geo[s] = scene_geometry(s)
K = {q: scales(dict(index=idx, root=base), q) for t in qs for q in qs[t]}
SC = [("(1) priors", "(1) priors"), ("(6) median", "(6) median of priors, room, door"), ("oracle [gold]", "(2) oracle [uses gold]")]

def rows(root, q, concept):
    p = root / ix(root)[q]["pipeline_sheet"] / "objects.csv"
    return [r for r in csv.DictReader(open(p)) if r["concept"] == concept] if p.is_file() else []
size3 = lambda r: np.array([float(r[a]) for a in ("size_x", "size_y", "size_z")])
def obj_value(root, q, k):
    name = [p.strip() for p in ix(root)[q]["prompts"].split(",") if p.strip() and p.strip() != "room"][0]
    rs = rows(root, q, name)
    if not rs: return None
    r = max(rs, key=lambda r: (int(r["num_frames"]), float(np.prod(size3(r)))))
    return 100 * k * float(size3(r).max())
def room_value(root, q, k):
    rs = rows(root, q, "room")
    if not rs: return None
    gv = np.abs(geo[idx[q]["scene"]][0])
    r = max(rs, key=lambda r: float(np.prod(size3(r))))
    e = size3(r); keep = np.argsort(gv)[:2]  # the two axes least aligned with gravity
    return k * k * float(e[keep[0]] * e[keep[1]])

vid = pd.read_json(E / "results/dev200/A_video.jsonl", lines=True).set_index("id")
out = []
for t, fn in (("object_size_estimation", obj_value), ("room_size_estimation", room_value)):
    for label, v in VARIANTS:
        for sl, sk in SC:
            vals = {q: fn(root_for(v, idx[q]["scene"]), q, K[q][sk]) for q in qs[t] if K[q][sk]}
            ok = [q for q in vals if vals[q] is not None]
            truth = {q: float(idx[q]["ground_truth"]) for q in ok}
            d200 = [q for q in ok if q in vid.index]
            out.append({"question type": t, "boxes": label, "scale": sl, "n": f"{len(ok)} of {len(qs[t])}",
                        "MRA": 100 * np.mean([mra(vals[q], truth[q]) for q in ok]),
                        "pred/truth": float(np.median([vals[q] / truth[q] for q in ok])),
                        "dev200: sheet MRA": 100 * np.mean([mra(vals[q], truth[q]) for q in d200]) if d200 else float("nan"),
                        "dev200: video MRA": 100 * vid.loc[d200, "score"].mean() if d200 else float("nan"), "dev200 n": len(d200)})
d = pd.DataFrame(out)
for t in qs:
    x = d[d["question type"] == t].drop(columns="question type")
    print(f"\n## {t}\n\n| " + " | ".join(x.columns) + " |\n|" + "---|" * len(x.columns))
    for r in x.itertuples(index=False):
        print("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) and c == "pred/truth" else f"{v:.1f}" if isinstance(v, float) else str(v) for c, v in zip(x.columns, r)) + " |")
