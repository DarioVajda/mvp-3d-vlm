"""Offline evaluation of box fixes for the distance relations, all with the median scale (relation_fixes fix 6), on every
distance question of the 50 dev scenes (primary basis), dev200 and the final test (secondary check)."""
import csv, math, runpy, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd

M, E, S = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.argv = ["relation_fixes.py", str(M), str(E), str(S)]
g = runpy.run_path(str(Path(__file__).with_name("relation_fixes.py")))
es, scene_scale, mra, relation, read_index = g["es"], g["scene_scale"], g["mra"], g["relation"], g["read_index"]
scales, geo, scene_geometry = g["scales"], g["geo"], g["scene_geometry"]
D, B = M / "data/rerun_sheets", S / "boxfix"
dev = set((E / "dev_scenes.txt").read_text().split()); dev15 = set((E / "dev_scenes_15.txt").read_text().split())
DIST = ("object_abs_distance", "object_rel_distance")
SCALE = "(6) median of priors, room, door"

def root_for(variant, s):
    d15, isdev = s in dev15, s in dev
    if variant == "base":
        return D / ("pipeline_v3_dedup2_dev50" if isdev else "pipeline_v3_dedup2_explore200")
    if variant.startswith("trim"):
        return B / (f"dedup_fixed2_s040_t{variant[4:]}" if d15 else f"dedup_pipeline_v3_t{variant[4:]}")
    if variant == "unmerged":
        return B / ("unmerged_fixed2_s040_min050" if d15 else "unmerged_pipeline_v3_dev35_unmerged" if isdev else "unmerged_pipeline_v3_explore200_unmerged")
_ix = {}
def ix(root):
    if root not in _ix: _ix[root] = read_index(root)
    return _ix[root]

VARIANTS = [("current boxes (min-max of the points, as built)", "base", "closest"),
            ("(a) trimmed 2nd-98th percentile", "trim2", "closest"),
            ("(a) trimmed 5th-95th percentile", "trim5", "closest"),
            ("(a) trimmed 10th-90th percentile", "trim10", "closest"),
            ("(b) unmerged boxes (best single row)", "unmerged", "closest"),
            ("(c) centre distance minus typical half-sizes", "base", "centre-prior"),
            ("(d) centre distance", "base", "centre")]

def named_rows(root, q):
    r = ix(root)[q]
    rows = list(csv.DictReader(open(root / r["pipeline_sheet"] / "objects.csv")))
    return {c: [x for x in rows if x["concept"] == c] for c in es.named_objects(r)}

def value(kind, root, q, k):
    """The relation in metres (absolute distance) or the closest candidate's name (relative distance), or None."""
    r = ix(root)[q]
    if kind == "closest":
        v = relation(ix(root), root, q)
        return v * k if isinstance(v, float) else v
    by = named_rows(root, q)
    if any(len(v) != 1 for v in by.values()):
        return None
    pos = {c: np.array([float(v[0][a]) for a in "xyz"]) for c, v in by.items()}
    half = {c: (scene_scale.PRIORS[c] / 2 if c in scene_scale.PRIORS else
                k * max(float(by[c][0][a]) for a in ("size_x", "size_y", "size_z")) / 2) for c in by}
    def dist(a, b):
        d = k * float(np.linalg.norm(pos[a] - pos[b]))
        return max(0.0, d - half[a] - half[b]) if kind == "centre-prior" else d
    names = es.named_objects(r)
    if r["question_type"] == "object_abs_distance":
        return dist(names[0], names[1])
    *cands, target = names
    return min(cands, key=lambda c: dist(c, target))

# the questions: every distance question on the 50 dev scenes, dev200's and the final test's; covered = the current sheet
# has every named object (the frozen rule), the same set for every variant
base_dev, base_fin = root_for("base", next(iter(dev))), root_for("base", "arkitscenes/42445429")
sets = {"all dev scenes": [q for q, r in ix(base_dev).items() if r["scene"] in dev and r["question_type"] in DIST],
        "dev200": [int(x) for x in (E / "dev200_ids.txt").read_text().split()],
        "final test": [int(x) for x in (E / "final_ids.txt").read_text().split()]}
sets["dev200"] = [q for q in sets["dev200"] if ix(base_dev)[q]["question_type"] in DIST]
sets["final test"] = [q for q in sets["final test"] if ix(base_fin)[q]["question_type"] in DIST]
video = {"dev200": pd.read_json(E / "results/dev200/A_video.jsonl", lines=True).set_index("id"),
         "final test": pd.read_json(E / "results/final/A_video.jsonl", lines=True).set_index("id")}
info = lambda q: (ix(base_dev) if q in ix(base_dev) and ix(base_dev)[q]["scene"] in dev else ix(base_fin))[q]
for qs in sets.values():
    for s in {info(q)["scene"] for q in qs}:
        if s not in geo: geo[s] = scene_geometry(s)
K = {}
for name, qs in sets.items():
    for q in qs:
        r = info(q); root = root_for("base", r["scene"])
        K[q] = scales(dict(index=ix(root), root=root), q)[SCALE]
covered = {q for qs in sets.values() for q in qs if es.relation_text(info(q), str(root_for("base", info(q)["scene"])), "pipeline", "v3")}
letter = lambda q, w: next((o.split(". ", 1)[0] for o in info(q)["options"].split(" | ") if o.split(". ", 1)[1].strip() == w), None)

out = {}
for label, variant, kind in VARIANTS:
    for name, qs in sets.items():
        absq = [q for q in qs if info(q)["question_type"] == "object_abs_distance"]
        rdq = [q for q in qs if info(q)["question_type"] == "object_rel_distance"]
        val = {q: value(kind, root_for(variant, info(q)["scene"]), q, K[q]) for q in qs if q in covered}
        ca = [q for q in absq if val.get(q) is not None]; cr = [q for q in rdq if val.get(q) is not None]
        t = lambda q: float(info(q)["ground_truth"])
        rec = {"abs n": len(ca), "abs MRA": 100 * np.mean([mra(val[q], t(q)) for q in ca]),
               "abs pred/truth": float(np.median([val[q] / t(q) for q in ca])),
               "rel n": len(cr), "rel acc": 100 * np.mean([letter(q, val[q]) == info(q)["ground_truth"] for q in cr])}
        if name in video:
            v = video[name]
            rec["abs simulated"] = 100 * np.mean([mra(val[q], t(q)) if q in ca else float(v.loc[q, "score"]) for q in absq])
            rec["rel simulated"] = 100 * np.mean([float(letter(q, val[q]) == info(q)["ground_truth"]) if q in cr else float(v.loc[q, "score"]) for q in rdq])
            rec["video abs"], rec["video rel"] = 100 * v.loc[absq, "score"].mean(), 100 * v.loc[rdq, "score"].mean()
        per = {}
        for ds in ("arkitscenes", "scannet", "scannetpp"):
            a_ = [q for q in ca if info(q)["scene"].startswith(ds + "/")]; r_ = [q for q in cr if info(q)["scene"].startswith(ds + "/")]
            per[ds] = (100 * np.mean([mra(val[q], t(q)) for q in a_]) if a_ else float("nan"), len(a_),
                       float(np.median([val[q] / t(q) for q in a_])) if a_ else float("nan"),
                       100 * np.mean([letter(q, val[q]) == info(q)["ground_truth"] for q in r_]) if r_ else float("nan"), len(r_))
        rec["per"] = per
        out[(label, name)] = rec

f = lambda v, d=1: "–" if isinstance(v, float) and math.isnan(v) else f"{v:.{d}f}"
print("## All 512 dev-scene distance questions (the basis for choosing)\n")
print("| fix | abs MRA (n) | abs pred/truth | rel acc (n) |\n|---|---:|---:|---:|")
for label, *_ in VARIANTS:
    r = out[(label, "all dev scenes")]
    print(f"| {label} | {f(r['abs MRA'])} ({r['abs n']}) | {f(r['abs pred/truth'], 2)} | {f(r['rel acc'])} ({r['rel n']}) |")
for name in ("dev200", "final test"):
    r0 = out[(VARIANTS[0][0], name)]
    print(f"\n## {name}: simulated scores (relation where there is one, video's answer otherwise); video only: abs {r0['video abs']:.1f}, rel {r0['video rel']:.1f}\n")
    print("| fix | abs MRA of relations (n) | abs pred/truth | abs simulated | rel acc (n) | rel simulated |\n|---|---:|---:|---:|---:|---:|")
    for label, *_ in VARIANTS:
        r = out[(label, name)]
        print(f"| {label} | {f(r['abs MRA'])} ({r['abs n']}) | {f(r['abs pred/truth'], 2)} | {f(r['abs simulated'])} | {f(r['rel acc'])} ({r['rel n']}) | {f(r['rel simulated'])} |")
for name in ("all dev scenes", "final test"):
    print(f"\n## Per dataset, {name}: abs MRA (n) / pred-truth / rel acc (n)\n")
    print("| fix | ARKitScenes | ScanNet | ScanNet++ |\n|---|---|---|---|")
    for label, *_ in VARIANTS:
        p = out[(label, name)]["per"]
        print(f"| {label} | " + " | ".join(f"{f(p[ds][0])} ({p[ds][1]}) / {f(p[ds][2], 2)} / {f(p[ds][3])} ({p[ds][4]})" for ds in ("arkitscenes", "scannet", "scannetpp")) + " |")
