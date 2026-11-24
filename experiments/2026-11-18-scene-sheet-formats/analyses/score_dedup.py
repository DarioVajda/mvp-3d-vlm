"""pipeline_v3 against its dedup and size-aware-merge variants on the 50 dev scenes: count = gold, the clean-sheet rate
and the answer the sheet's positions imply for relative-direction questions."""
import csv, itertools, math, re, sys
from pathlib import Path
import numpy as np
import pipeline_scorecard as pc

D, E = Path(sys.argv[1]), Path(sys.argv[2])
dev = set((E / "dev_scenes.txt").read_text().split())
variants = {"pipeline_v3": "pipeline_v3_dev50", "v3_dedup": "pipeline_v3_dedup_dev50",
            "v3_sizemerge": "pipeline_v3_sizemerge_dev50", "v3_sizemerge_dedup": "pipeline_v3_sizemerge_dedup_dev50"}
UNIQUE = {"object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard", "object_rel_distance", "object_abs_distance"}
named = lambda r: [] if r["question_type"] == "room_size_estimation" else [p.strip() for p in r["prompts"].split(",") if p.strip() and p.strip() != "room"]
def rows_of(root, r): return list(csv.DictReader(open(root / r["pipeline_sheet"] / "objects.csv")))
def parse(text):
    m = re.search(r"standing by the (.+?) and facing the (.+?), is the (.+?) (?:to (?:my|the)|in)", text)
    return m.groups() if m else None
def direction(t, X, Y, Z):  # camera frame: x right, y down, z forward; floor plane (x, z); right of facing f is (f_z, -f_x)
    P = lambda v: np.array([v[0], v[2]]); right = lambda f: np.array([f[1], -f[0]])
    f = P(Y) - P(X)
    if t.endswith("easy"):
        return "right" if (P(Z) - P(Y)) @ right(f) > 0 else "left"
    z = P(Z) - P(X)
    ang = math.degrees(math.acos(np.clip(z @ f / (np.linalg.norm(z) * np.linalg.norm(f) + 1e-12), -1, 1)))
    side = "right" if z @ right(f) > 0 else "left"
    return ("back" if ang >= 135 else side) if t.endswith("medium") else ("front-" if z @ f > 0 else "back-") + side
pos = lambda r: np.array([float(r[k]) for k in "xyz"])
def letter(r, word):
    for o in r["options"].split(" | "):
        k, v = o.split(". ", 1)
        if v.strip() == word: return k

out = {}
for name, folder in variants.items():
    root = D / folder
    index = [r for r in csv.DictReader(open(root / "index.csv")) if r["scene"] in dev]
    recs = pc.score_root(root, dev)
    clean_u = [all(sum(x["concept"] == c for x in rows_of(root, r)) == 1 for c in named(r)) for r in index if r["question_type"] in UNIQUE]
    clean_all = [all(sum(x["concept"] == c for x in rows_of(root, r)) == 1 for c in named(r)) for r in index if named(r)]
    best_ok = unamb_ok = present = n = 0
    for r in index:
        if not r["question_type"].startswith("object_rel_direction"):
            continue
        n += 1
        names = parse(r["question"])
        rs = rows_of(root, r)
        by = {c: [x for x in rs if x["concept"] == c] for c in names} if names else {}
        if not names or not all(by.values()):
            continue
        present += 1
        best = {c: max(v, key=lambda x: (int(x["num_frames"]), float(x["size_x"]) * float(x["size_y"]) * float(x["size_z"]))) for c, v in by.items()}
        best_ok += letter(r, direction(r["question_type"], *(pos(best[c]) for c in names))) == r["ground_truth"]
        combos = {letter(r, direction(r["question_type"], *(pos(x) for x in c))) for c in itertools.product(*(by[c] for c in names))}
        unamb_ok += combos == {r["ground_truth"]}
    m = {mm[0]: mm[1] for mm in pc.MEASURES}
    out[name] = {"count = gold": 100 * m["count = gold, per (question, class)"](recs),
                 "unmatched rows": 100 * m["unmatched rows (beyond the gold sheet's count)"](recs),
                 "named objects found": 100 * m["named objects found"](recs),
                 "clean sheets, the 3 dedup types": 100 * np.mean(clean_u), "clean sheets, all with named objects": 100 * np.mean(clean_all),
                 "rel. direction: every object present": f"{present} / {n}",
                 "rel. direction: implied answer right (most-seen rows)": 100 * best_ok / n,
                 "rel. direction: unambiguous and right": 100 * unamb_ok / n,
                 "rel. direction: implied right, of those present": 100 * best_ok / max(1, present)}
keys = list(next(iter(out.values())))
print("| measure (50 dev scenes) | " + " | ".join(out) + " |\n|---|" + "---:|" * len(out))
for k in keys:
    print(f"| {k} | " + " | ".join(v[k] if isinstance(v[k], str) else f"{v[k]:.1f}%" for v in out.values()) + " |")
