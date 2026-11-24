"""The registered results of prereg_full_method.md (H1, H2), recomputed from the predictions, then exploratory analyses."""
import csv, math, re, sys
from pathlib import Path
import numpy as np, pandas as pd
R, E, ROOT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(E))
import explore_sheets as es
from mvp.benchmarks import scene_sheet as ss
from mvp.benchmarks.vsibench import summarize, task_of, final_letter, final_number

IDS = [int(x) for x in (E / "final_ids.txt").read_text().split()]
A = pd.read_json(R / "A_video.jsonl", lines=True).drop_duplicates("id").set_index("id").loc[IDS]
B = pd.read_json(R / "pipeline_v3_dedup_relations.jsonl", lines=True).drop_duplicates("id").set_index("id").loc[IDS]
def sign_p(h, l):
    n = h + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(h, l) + 1)) / 2 ** n)

# H1, as registered
DIST = [q for q in IDS if A.loc[q, "question_type"] in ("object_abs_distance", "object_rel_distance")]
d = B.loc[DIST, "score"] - A.loc[DIST, "score"]
h, l = int((d > 0).sum()), int((d < 0).sum()); p = sign_p(h, l)
print("## REGISTERED RESULTS\n")
print(f"H1 (primary): {len(DIST)} distance questions; helped {h}, hurt {l}, unchanged {len(DIST) - h - l}; exact two-sided sign test p = {p:.6f}")
print(f"   verdict: {'CONFIRMED' if p < 0.05 and h > l else 'NOT CONFIRMED'} (confirmed only if p < 0.05 and helped > hurt)")

# H2, as registered (the notebook's procedure: IDS order, groups by sorted question type, seed 0)
qtype = A["question_type"]
full = B["score"].where(qtype != "object_size_estimation", A["score"])
QT = sorted(set(qtype))
def overall_of(means):
    t = {}
    for q, m in means.items(): t.setdefault(task_of(q), []).append(m)
    return 100 * np.mean([np.mean(x) for x in t.values()])
groups = {t: np.array([IDS.index(q) for q in qtype.index[qtype == t]]) for t in QT}
f_s, a_s = full.to_numpy(), A["score"].to_numpy()
fo, ao = overall_of({t: f_s[g].mean() for t, g in groups.items()}), overall_of({t: a_s[g].mean() for t, g in groups.items()})
assert abs(ao - summarize(A.reset_index())["overall"]) < 1e-9
rng = np.random.default_rng(0); draws = np.empty(10000)
for i in range(10000):
    s = {t: rng.choice(g, len(g)) for t, g in groups.items()}
    draws[i] = overall_of({t: f_s[k].mean() for t, k in s.items()}) - overall_of({t: a_s[k].mean() for t, k in s.items()})
fd = full - A["score"]
print(f"\nH2 (secondary): all {len(IDS)} questions; VSI-Bench overall (tasks present): A_video {ao:.2f}, full method {fo:.2f}, difference {fo - ao:+.2f}, "
      f"95% bootstrap [{np.percentile(draws, 2.5):+.2f}, {np.percentile(draws, 97.5):+.2f}]; helped {int((fd > 0).sum())}, hurt {int((fd < 0).sum())}")
print(f"   {'supported' if np.percentile(draws, 2.5) > 0 else 'not supported'} (supported only if the interval excludes 0 and lies above it)")
nb = pd.read_csv(R / "preregistered_test.csv").iloc[0]
print(f"\nthe notebook's preregistered_test.csv: helped {nb['helped']}, hurt {nb['hurt']}, p {nb['sign test p']:.6f}, {nb['verdict']}; "
      f"secondary {nb['full method minus video (secondary)']:+.2f} [{nb['95% low']:+.2f}, {nb['95% high']:+.2f}]")

# ---------------- exploratory
print("\n## EXPLORATORY (not registered)\n")
def md(df):
    df = df.reset_index(); f = lambda x: f"{x:.1f}" if isinstance(x, float) else str(x)
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)] + ["| " + " | ".join(f(x) for x in r) + " |" for r in df.itertuples(index=False)])
rows = []
for t in QT:
    g = groups[t]; dt = B["score"].to_numpy()[g] - a_s[g]
    rows.append({"question type": t, "n": len(g), "A_video": 100 * a_s[g].mean(), "dedup_relations": 100 * B["score"].to_numpy()[g].mean(),
                 "full method": 100 * f_s[g].mean(), "sheet vs video helped/hurt": f"{int((dt > 0).sum())}/{int((dt < 0).sum())}",
                 "sign p (unadjusted)": f"{sign_p(int((dt > 0).sum()), int((dt < 0).sum())):.3f}"})
print("per question type:\n\n" + md(pd.DataFrame(rows).set_index("question type")))
idx = ss.load_index(ROOT)
withrel = [q for q in DIST if es.relation_text(idx[q], str(ROOT), "pipeline", "v3")]
assert len(withrel) == 114, len(withrel)
rows = []
for name, qs in (("with relations (every named object in the sheet)", withrel), ("without relations (a named object missing)", [q for q in DIST if q not in withrel])):
    for t in ("object_abs_distance", "object_rel_distance"):
        sel = [q for q in qs if qtype[q] == t]
        dt = B.loc[sel, "score"] - A.loc[sel, "score"]
        rows.append({"group": name, "type": t, "n": len(sel), "A_video": 100 * A.loc[sel, "score"].mean(), "dedup_relations": 100 * B.loc[sel, "score"].mean(),
                     "helped/hurt": f"{int((dt > 0).sum())}/{int((dt < 0).sum())}"})
print("\ndistance questions with and without relations:\n\n" + md(pd.DataFrame(rows).set_index("group")))
mra = lambda pr, g: float(np.mean([abs(pr - g) / g < 1 - t for t in np.arange(0.5, 1.0, 0.05)])) if pr is not None and np.isfinite(pr) else 0.0
abs_rel = [q for q in withrel if qtype[q] == "object_abs_distance"]
pre = {q: float(re.search(r"Closest-point distance between .*?: ([0-9.]+) m", es.relation_text(idx[q], str(ROOT), "pipeline", "v3")).group(1)) for q in abs_rel}
g_ = {q: float(A.loc[q, "ground_truth"]) for q in abs_rel}
copied = sum(final_number(B.loc[q, "prediction"]) is not None and abs(final_number(B.loc[q, "prediction"]) - pre[q]) <= 0.01 * pre[q] + 0.005 for q in abs_rel)
r = np.array([pre[q] / g_[q] for q in abs_rel])
print(f"\nabsolute distance with a precomputed distance ({len(abs_rel)}): precomputed vs truth MRA {100 * np.mean([mra(pre[q], g_[q]) for q in abs_rel]):.1f}, "
      f"median ratio {np.median(r):.2f} (quartiles {np.percentile(r, 25):.2f}-{np.percentile(r, 75):.2f}); within 25% of the truth {int((np.abs(r - 1) <= 0.25).sum())}; "
      f"the model repeats it in {copied}; video's MRA there {100 * A.loc[abs_rel, 'score'].mean():.1f}, dedup_relations' {100 * B.loc[abs_rel, 'score'].mean():.1f}")
rd_rel = [q for q in withrel if qtype[q] == "object_rel_distance"]
def closest(q):
    m = re.search(r"Closest to the .*?: (.+)\.$", es.relation_text(idx[q], str(ROOT), "pipeline", "v3").strip().splitlines()[-1])
    return next((o.split(". ", 1)[0] for o in A.loc[q, "options"] if o.split(". ", 1)[1].strip() == m.group(1)), None) if m else None
cl = {q: closest(q) for q in rd_rel}
print(f"relative distance with a precomputed closest object ({len(rd_rel)}): it is right in {sum(cl[q] == A.loc[q, 'ground_truth'] for q in rd_rel)}; "
      f"the model answers it in {sum(final_letter(B.loc[q, 'prediction']) == cl[q] for q in rd_rel)}; video right in {int(A.loc[rd_rel, 'score'].sum())}, dedup_relations right in {int(B.loc[rd_rel, 'score'].sum())}")
