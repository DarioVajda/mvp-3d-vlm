"""Descriptive analysis of the confirmation run (relative direction, 177 questions). The pre-registered result is the
notebook's sign test; nothing here is a new test of the hypothesis."""
import csv, itertools, math, re, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
from mvp.benchmarks import scene_scale
from mvp.benchmarks.vsibench import summarize, final_letter

R, V3 = Path(sys.argv[1]), Path(sys.argv[2])
A = pd.read_json(R / "A_video.jsonl", lines=True).drop_duplicates("id").set_index("id")
B = pd.read_json(R / "pipeline_v3.jsonl", lines=True).drop_duplicates("id").set_index("id")
REL = ["object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard"]
rel = [q for q in A.index if A.loc[q, "question_type"] in REL]
idx = {int(r["id"]): r for r in csv.DictReader(open(V3 / "index.csv"))}
md = lambda df: "\n".join(["| " + " | ".join(map(str, df.reset_index().columns)) + " |", "|" + "---|" * len(df.reset_index().columns)] +
                          ["| " + " | ".join((f"{x:.1f}" if isinstance(x, float) else str(x)) for x in r) + " |" for r in df.reset_index().itertuples(index=False)])

# ---------------- 1. the reported numbers, from the predictions
d = B.loc[rel, "score"] - A.loc[rel, "score"]
h, l = int((d > 0).sum()), int((d < 0).sum())
p = min(1.0, 2 * sum(math.comb(h + l, i) for i in range(min(h, l) + 1)) / 2 ** (h + l))
sa, sb = summarize(A.loc[rel].reset_index()), summarize(B.loc[rel].reset_index())
rng = np.random.default_rng(0)
groups = {t: np.array([rel.index(q) for q in rel if A.loc[q, "question_type"] == t]) for t in REL}
a_s, b_s = A.loc[rel, "score"].to_numpy(), B.loc[rel, "score"].to_numpy()
draws = np.empty(10000)
for i in range(10000):
    s = {t: rng.choice(g, len(g)) for t, g in groups.items()}
    draws[i] = 100 * (np.mean([b_s[k].mean() for k in s.values()]) - np.mean([a_s[k].mean() for k in s.values()]))
print("## 1. The pre-registered result, recomputed from the predictions\n")
print(f"- relative-direction questions: {len(rel)}; helped {h}, hurt {l}, unchanged {len(rel) - h - l}")
print(f"- exact two-sided sign test: p = {p:.6f} -> {'CONFIRMED' if p < 0.05 and h > l else 'NOT CONFIRMED'}")
print(f"- relative-direction score: A_video {sa['object_rel_direction']:.2f}, pipeline_v3 {sb['object_rel_direction']:.2f}, "
      f"difference {sb['object_rel_direction'] - sa['object_rel_direction']:+.2f}, 95% bootstrap [{np.percentile(draws, 2.5):+.2f}, {np.percentile(draws, 97.5):+.2f}]")
for t in REL:
    g = groups[t]
    print(f"- {t.rsplit('_', 1)[1]}: n {len(g)}, A_video {100 * a_s[g].mean():.1f}, pipeline_v3 {100 * b_s[g].mean():.1f}, "
          f"helped/hurt {int((b_s[g] > a_s[g]).sum())}/{int((b_s[g] < a_s[g]).sum())}")
pr = pd.read_csv(R / "preregistered_test.csv").iloc[0]
print(f"- the notebook's preregistered_test.csv: helped {pr['helped']}, hurt {pr['hurt']}, p {pr['sign test p']:.6f}, "
      f"{pr['verdict']}, difference {pr['difference']:+.2f} [{pr['95% low']:+.2f}, {pr['95% high']:+.2f}]")

# ---------------- 2. answer distributions
letter = lambda run, q: final_letter(run.loc[q, "prediction"])
print("\n## 2. Answer letters (descriptive)\n")
for t in REL:
    qs = [q for q in rel if A.loc[q, "question_type"] == t]
    letters = "AB" if t.endswith("easy") else "ABC" if t.endswith("medium") else "ABCD"
    rows = []
    for name, f in (("truth", lambda q: A.loc[q, "ground_truth"]), ("A_video", lambda q: letter(A, q)), ("pipeline_v3", lambda q: letter(B, q))):
        c = Counter(f(q) for q in qs)
        rows.append({"": name, **{x: c.get(x, 0) for x in letters}, "other": sum(v for k, v in c.items() if k not in letters)})
    print(f"{t.rsplit('_', 1)[1]} ({len(qs)} questions):\n\n" + md(pd.DataFrame(rows).set_index("")) + "\n")
helped = [q for q in rel if B.loc[q, "score"] > A.loc[q, "score"]]
hurt = [q for q in rel if B.loc[q, "score"] < A.loc[q, "score"]]
va = Counter(letter(A, q) for q in helped)
print(f"helped ({len(helped)}): A_video's (wrong) answer was {dict(sorted(va.items()))}; 'A' in {va.get('A', 0)} of {len(helped)} "
      f"(A_video answers 'A' in {sum(letter(A, q) == 'A' for q in rel)} of {len(rel)} questions overall, the truth is 'A' in "
      f"{sum(A.loc[q, 'ground_truth'] == 'A' for q in rel)})")
vb = Counter(letter(B, q) for q in hurt)
print(f"hurt ({len(hurt)}): pipeline_v3's (wrong) answer was {dict(sorted(vb.items()))}")

# ---------------- 3. sheet quality and the sheet-implied answer
def rows_of(root, q, src="pipeline"):
    return list(csv.DictReader(open(root / idx[q][f"{src}_sheet"] / "objects.csv")))
def parse(q):
    """(standing by, facing, target) from the question's text."""
    text = A.loc[q, "question"]
    m = re.search(r"standing by the (.+?) and facing the (.+?), is the (.+?) (?:to (?:my|the)|in)", text)
    return m.groups() if m else None
def direction(t, X, Y, Z, up_axis):
    """The answer the positions imply. X: where I stand, Y: what I face, Z: the object asked about. Floor plane: the
    axes other than up_axis; for the camera frame (y down) the up axis is -y, for gold (z up) it is +z."""
    if up_axis == "y":    # pipeline: x right, y down, z forward; seen from above: (x, z), right of +z is +x
        P = lambda v: np.array([v[0], v[2]]); right = lambda f: np.array([f[1], -f[0]])
    else:                 # gold: z up; floor (x, y), right of facing f is (f_y, -f_x)
        P = lambda v: np.array([v[0], v[1]]); right = lambda f: np.array([f[1], -f[0]])
    f = P(Y) - P(X)
    if t.endswith("easy"):   # left or right of Y, seen from X facing Y
        z = P(Z) - P(Y)
        return "right" if z @ right(f) > 0 else "left"
    z = P(Z) - P(X)
    ang = math.degrees(math.acos(np.clip(z @ f / (np.linalg.norm(z) * np.linalg.norm(f) + 1e-12), -1, 1)))
    side = "right" if z @ right(f) > 0 else "left"
    if t.endswith("medium"):
        return "back" if ang >= 135 else side
    return ("front-" if z @ f > 0 else "back-") + side
def pos(r): return np.array([float(r[k]) for k in "xyz"])
def option_letter(q, word):
    for o in A.loc[q, "options"]:
        k, v = o.split(". ", 1)
        if v.strip() == word: return k
    return None
qual, implied, implied_all, gold_implied = {}, {}, {}, {}
for q in rel:
    names = parse(q)
    pr_ = rows_of(V3, q); gr = rows_of(V3, q, "gold")
    t = A.loc[q, "question_type"]
    by = {n: [r for r in pr_ if r["concept"] == n] for n in names} if names else {}
    gby = {n: [r for r in gr if r["concept"] == n] for n in names} if names else {}
    if not names:
        qual[q] = "unparsed"; continue
    if any(not v for v in by.values()):
        qual[q] = "some missing"
    elif any(len(v) > max(1, len(gby[n])) for n, v in by.items()):
        qual[q] = "all present, duplicated"
    else:
        qual[q] = "all present, each once"
    if all(by.values()):
        best = {n: max(v, key=lambda r: (int(r["num_frames"]), float(r["size_x"]) * float(r["size_y"]) * float(r["size_z"]))) for n, v in by.items()}
        implied[q] = option_letter(q, direction(t, *(pos(best[n]) for n in names), "y"))
        combos = {option_letter(q, direction(t, *(pos(r) for r in c), "y")) for c in itertools.product(*(by[n] for n in names))}
        implied_all[q] = "consistent" if len(combos) == 1 else "ambiguous"
    if all(len(v) == 1 for v in gby.values()):
        gold_implied[q] = option_letter(q, direction(t, *(pos(gby[n][0]) for n in names), "z"))
print("\n## 3. Sheet quality (descriptive)\n")
ok = [q for q in gold_implied]
print(f"geometry check: the gold sheet's positions imply the true answer in {sum(gold_implied[q] == A.loc[q, 'ground_truth'] for q in ok)} "
      f"of {len(ok)} questions (where gold has one row per named object)")
rows = []
for g in ("all present, each once", "all present, duplicated", "some missing", "unparsed"):
    qs = [q for q in rel if qual[q] == g]
    if not qs: continue
    dd = B.loc[qs, "score"] - A.loc[qs, "score"]
    rows.append({"sheet": g, "n": len(qs), "A_video": 100 * A.loc[qs, "score"].mean(), "pipeline_v3": 100 * B.loc[qs, "score"].mean(),
                 "helped / hurt": f"{int((dd > 0).sum())} / {int((dd < 0).sum())}",
                 "sheet implies the truth": f"{sum(implied.get(q) == A.loc[q, 'ground_truth'] for q in qs)} / {sum(q in implied for q in qs)}",
                 "v3 answers what the sheet implies": f"{sum(implied.get(q) is not None and letter(B, q) == implied[q] for q in qs)} / {sum(q in implied for q in qs)}"})
print(md(pd.DataFrame(rows).set_index("sheet")))
rows = []
for g, sel in (("sheet implies the truth", lambda q: q in implied and implied[q] == A.loc[q, "ground_truth"]),
               ("sheet implies a wrong answer", lambda q: q in implied and implied[q] != A.loc[q, "ground_truth"]),
               ("sheet cannot imply an answer (object missing)", lambda q: q not in implied)):
    qs = [q for q in rel if sel(q)]
    dd = B.loc[qs, "score"] - A.loc[qs, "score"]
    rows.append({"": g, "n": len(qs), "A_video": 100 * A.loc[qs, "score"].mean(), "pipeline_v3": 100 * B.loc[qs, "score"].mean(),
                 "helped / hurt": f"{int((dd > 0).sum())} / {int((dd < 0).sum())}"})
print("\nby what the sheet's positions imply:\n\n" + md(pd.DataFrame(rows).set_index("")))

# ---------------- 4. per dataset and sub-type
rows = []
for ds in ("arkitscenes", "scannet", "scannetpp"):
    for t in REL + ["all"]:
        qs = [q for q in rel if A.loc[q, "dataset"] == ds and (t == "all" or A.loc[q, "question_type"] == t)]
        dd = B.loc[qs, "score"] - A.loc[qs, "score"]
        rows.append({"dataset": ds, "sub-type": t.rsplit("_", 1)[-1], "n": len(qs), "A_video": 100 * A.loc[qs, "score"].mean(),
                     "pipeline_v3": 100 * B.loc[qs, "score"].mean(), "helped / hurt": f"{int((dd > 0).sum())} / {int((dd < 0).sum())}"})
print("\n## 4. Per dataset and sub-type (descriptive)\n\n" + md(pd.DataFrame(rows).set_index("dataset")))

# ---------------- 5. examples: the first 5 helped and the first 5 hurt by question id
def k_of(q):
    return scene_scale.question_scale(q, str(V3))["k"]
def show(q):
    names = parse(q); k = k_of(q); t = A.loc[q, "question_type"]
    pr_ = rows_of(V3, q); gr = rows_of(V3, q, "gold")
    print(f"\n### {q} ({A.loc[q, 'dataset']}/{A.loc[q, 'scene_name']}, {t.rsplit('_', 1)[1]}): {qual[q]}")
    print(f"{A.loc[q, 'question'].splitlines()[0]}")
    print(f"options {A.loc[q, 'options']}; truth {A.loc[q, 'ground_truth']}; A_video {letter(A, q)}; pipeline_v3 {letter(B, q)}; "
          f"sheet implies {implied.get(q, '-')} ({implied_all.get(q, 'no answer')}); gold implies {gold_implied.get(q, '-')}")
    for n in (names or []):
        rs = [r for r in pr_ if r["concept"] == n]
        print(f"  {n}: {len(rs)} sheet row(s), gold {sum(r['concept'] == n for r in gr)}" +
              "".join(f"\n    ({float(r['x']) * k:.2f}, {float(r['y']) * k:.2f}, {float(r['z']) * k:.2f}) m, size "
                      f"{float(r['size_x']) * k:.2f}x{float(r['size_y']) * k:.2f}x{float(r['size_z']) * k:.2f}, {r['num_frames']} frame(s)" for r in rs))
print("\n## 5. Examples: the first 5 helped and the first 5 hurt, by question id\n")
print("**helped**")
for q in sorted(helped)[:5]: show(q)
print("\n**hurt**")
for q in sorted(hurt)[:5]: show(q)
