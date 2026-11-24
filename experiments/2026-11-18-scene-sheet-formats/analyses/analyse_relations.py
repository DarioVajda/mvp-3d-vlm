"""dev200: pipeline_v3_dedup_relations against A_video, pipeline_v3 and pipeline_v3_dedup. Dev results: chosen and
analysed on dev, so descriptive."""
import csv, math, re, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, sys.argv[3])
import explore_sheets as es
from mvp.benchmarks import scene_sheet as ss
from mvp.benchmarks.vsibench import summarize, task_of, final_letter, final_number

R, WORK = Path(sys.argv[1]), Path(sys.argv[2])
names = ("A_video", "pipeline_v3", "pipeline_v3_dedup", "pipeline_v3_dedup_relations")
runs = {n: pd.read_json(R / f"{n}.jsonl", lines=True).drop_duplicates("id").set_index("id") for n in names}
ids = sorted(runs["A_video"].index)
V, P, D, X = (runs[n].loc[ids] for n in names)
qtype = V["question_type"]; task = qtype.map(task_of)
REL_ROOT, DD_ROOT, V3_ROOT = (WORK / f"data/{d}" for d in ("scene_sheets_pipeline_v3_dedup2_priors", "scene_sheets_pipeline_v3_dedup_priors", "scene_sheets_pipeline_v3_priors"))
irel, idd, iv3 = ss.load_index(REL_ROOT), ss.load_index(DD_ROOT), ss.load_index(V3_ROOT)
def md(df):
    df = df.reset_index(); f = lambda x: f"{x:.1f}" if isinstance(x, float) else str(x)
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)] + ["| " + " | ".join(f(x) for x in r) + " |" for r in df.itertuples(index=False)])
def sign_p(h, l):
    n = h + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(h, l) + 1)) / 2 ** n)
QT = sorted(set(qtype))
def overall_of(means):
    t = {}
    for q, m in means.items(): t.setdefault(task_of(q), []).append(m)
    return 100 * np.mean([np.mean(x) for x in t.values()])
def overall(s): return overall_of({t: s.loc[qtype.index[qtype == t]].mean() for t in QT})
def boot(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed); g = {t: np.array([ids.index(q) for q in qtype.index[qtype == t]]) for t in QT}
    A_, B_ = a.to_numpy(), b.to_numpy(); out = np.empty(n)
    for i in range(n):
        s = {t: rng.choice(x, len(x)) for t, x in g.items()}
        out[i] = overall_of({t: A_[k].mean() for t, k in s.items()}) - overall_of({t: B_[k].mean() for t, k in s.items()})
    return out
for df in (V, P, D, X):
    assert abs(overall(df["score"]) - summarize(df.reset_index())["overall"]) < 1e-9

print("## 1. Overall (dev200; dev results)\n")
print("VSI-Bench overall: " + ", ".join(f"{n} {overall(df['score']):.2f}" for n, df in zip(names, (V, P, D, X))) + "\n")
def compare(rows, label, a, b):
    d = a - b; h, l = int((d > 0).sum()), int((d < 0).sum()); bs = boot(a, b)
    rows.append({"comparison": label, "difference": overall(a) - overall(b), "95% CI": f"[{np.percentile(bs, 2.5):+.1f}, {np.percentile(bs, 97.5):+.1f}]",
                 "P(> 0)": f"{(bs > 0).mean():.2f}", "helped / hurt": f"{h} / {l}", "sign test p": f"{sign_p(h, l):.3f}"})
rows = []
compare(rows, "dedup_relations - video", X["score"], V["score"])
compare(rows, "dedup_relations - dedup", X["score"], D["score"])
print(md(pd.DataFrame(rows).set_index("comparison")))

rows = []
for t in ["object_abs_distance", "object_rel_distance", "object_rel_direction"] + [t for t in sorted(set(task)) if t not in ("object_abs_distance", "object_rel_distance", "object_rel_direction")]:
    qs = task.index[task == t]; sc = lambda df: summarize(df.loc[qs].reset_index())[t]
    dv, dd_ = X.loc[qs, "score"] - V.loc[qs, "score"], X.loc[qs, "score"] - D.loc[qs, "score"]
    rows.append({"task": t, "n": len(qs), "video": sc(V), "v3": sc(P), "dedup": sc(D), "dedup_relations": sc(X),
                 "rel. vs video helped/hurt": f"{int((dv > 0).sum())}/{int((dv < 0).sum())}",
                 "rel. vs dedup helped/hurt": f"{int((dd_ > 0).sum())}/{int((dd_ < 0).sum())}"})
print("\n## 2. Per task (descriptive)\n\n" + md(pd.DataFrame(rows).set_index("task")))

# 3. relative direction
REL = [q for q in ids if qtype[q].startswith("object_rel_direction")]
let = {n: {q: final_letter(df.loc[q, "prediction"]) for q in REL} for n, df in zip(("video", "v3", "dedup", "dedup_relations"), (V, P, D, X))}
stated = {}
for q in REL:
    line = es.direction_relation(irel[q], str(REL_ROOT))
    if line:
        w = re.search(r"is to (?:the|your) (\S+?)(?: of|\.)", line).group(1)
        opts = {o.split(". ", 1)[1].strip(): o.split(". ", 1)[0] for o in V.loc[q, "options"]}
        stated[q] = opts.get(w)
qs = list(stated)
print(f"\n## 3. Relative direction ({len(REL)} questions; descriptive)\n")
print(f"questions with a stated direction: {len(qs)}; the stated direction is right in {sum(stated[q] == V.loc[q, 'ground_truth'] for q in qs)}")
for n in let:
    print(f"- {n}: answers the stated direction in {sum(let[n][q] == stated[q] for q in qs)} of {len(qs)}; right in {sum(let[n][q] == V.loc[q, 'ground_truth'] for q in qs)} of {len(qs)}; "
          f"right on the other {len(REL) - len(qs)}: {sum(let[n][q] == V.loc[q, 'ground_truth'] for q in REL if q not in stated)}")
rows = []
for name, f in (("truth", lambda q: V.loc[q, "ground_truth"]),) + tuple((n, (lambda n: lambda q: let[n][q])(n)) for n in let):
    c = Counter(f(q) for q in REL); rows.append({"": name, **{x: c.get(x, 0) for x in "ABCD"}})
print("\nanswer letters, all 25:\n\n" + md(pd.DataFrame(rows).set_index("")))
c = Counter(let["dedup_relations"][q] for q in qs); ct = Counter(V.loc[q, "ground_truth"] for q in qs)
print(f"\non the {len(qs)} with a stated direction: truth {dict(sorted(ct.items()))}, dedup_relations {dict(sorted(c.items()))}, video {dict(sorted(Counter(let['video'][q] for q in qs).items()))}")

# 4. distance questions
def precomputed(q):
    r = es.relations(irel[q], str(REL_ROOT), "pipeline", "closest")
    if qtype[q] == "object_abs_distance":
        m = re.search(r"Closest-point distance between .*?: ([0-9.]+) m", r or ""); return float(m.group(1)) if m else None
    m = re.search(r"Closest to the .*?: (.+)\.$", (r or "").strip().splitlines()[-1]) if r else None
    if not m: return None
    return next((o.split(". ", 1)[0] for o in V.loc[q, "options"] if o.split(". ", 1)[1].strip() == m.group(1)), None)
mra = lambda p, g: float(np.mean([abs(p - g) / g < 1 - t for t in np.arange(0.5, 1.0, 0.05)])) if p is not None and np.isfinite(p) else 0.0
ABS = [q for q in ids if qtype[q] == "object_abs_distance"]; RD = [q for q in ids if qtype[q] == "object_rel_distance"]
pa = {q: precomputed(q) for q in ABS}; have = [q for q in ABS if pa[q] is not None]
num = lambda df, q: final_number(df.loc[q, "prediction"])
print(f"\n## 4. Distance questions (descriptive)\n")
print(f"absolute distance: {len(have)} of {len(ABS)} questions get a precomputed distance. On those:")
print(f"- precomputed vs truth: MRA {100 * np.mean([mra(pa[q], float(V.loc[q, 'ground_truth'])) for q in have]):.1f}, median ratio {np.median([pa[q] / float(V.loc[q, 'ground_truth']) for q in have]):.2f}")
for n, df in (("video", V), ("dedup", D), ("dedup_relations", X)):
    copied = sum(num(df, q) is not None and abs(num(df, q) - pa[q]) <= 0.01 * max(pa[q], 0.01) + 0.005 for q in have)
    print(f"- {n}: MRA {100 * np.mean([df.loc[q, 'score'] for q in have]):.1f}; answer equals the precomputed distance (within 1%) in {copied} of {len(have)}")
print(f"- on the {len(ABS) - len(have)} without one: video {100 * np.mean([V.loc[q, 'score'] for q in ABS if q not in have]):.1f}, dedup_relations {100 * np.mean([X.loc[q, 'score'] for q in ABS if q not in have]):.1f}")
pr = {q: precomputed(q) for q in RD}; hr = [q for q in RD if pr[q] is not None]
print(f"\nrelative distance: {len(hr)} of {len(RD)} get a precomputed closest object; it is right in {sum(pr[q] == V.loc[q, 'ground_truth'] for q in hr)}")
for n, df in (("video", V), ("dedup", D), ("dedup_relations", X)):
    print(f"- {n}: answers the precomputed object in {sum(final_letter(df.loc[q, 'prediction']) == pr[q] for q in hr)} of {len(hr)}; right in {int(sum(df.loc[q, 'score'] for q in hr))} of {len(hr)}")

# 5. 1282
q = 1282
print(f"\n## 5. Question 1282 (tie-break)\n\n{V.loc[q, 'question']}\noptions {V.loc[q, 'options']}; truth {V.loc[q, 'ground_truth']}; "
      + "; ".join(f"{n} {final_letter(df.loc[q, 'prediction'])}" for n, df in zip(("video", "v3", "dedup", "dedup_relations"), (V, P, D, X))))
rows_ = lambda root, ix: [(r["concept"], round(float(r["x"]), 2), round(float(r["z"]), 2), r["num_frames"]) for r in csv.DictReader(open(root / ix[q]["pipeline_sheet"] / "objects.csv"))]
print(f"dedup sheet: {rows_(DD_ROOT, idd)}\ndedup_relations sheet: {rows_(REL_ROOT, irel)}\nrelation line: {es.relation_text(irel[q], str(REL_ROOT), 'pipeline', 'v3') or '(none: an object is missing)'}")

# 6. the simulated full method
def full(q):
    t = qtype[q]
    if t in ("object_size_estimation", "room_size_estimation"): return V.loc[q, "score"]
    if t in es.DIRECTION_TYPES | es.DISTANCE_TYPES: return X.loc[q, "score"]
    return D.loc[q, "score"]
F = pd.Series({q: full(q) for q in ids})
print("\n## 6. Simulated full method (CHOSEN ON DEV: not a confirmed result)\n")
print("dedup_relations for distance and direction questions, video for object and room size, dedup for the rest")
rows = []
compare(rows, "full method - video", F, V["score"])
compare(rows, "full method - dedup", F, D["score"])
print(f"overall {overall(F):.2f}\n\n" + md(pd.DataFrame(rows).set_index("comparison")))

# 7. examples vs dedup
d = X["score"] - D["score"]
print("\n## 7. Examples: dedup_relations against dedup, the first 5 helped and the first 5 hurt by question id\n")
for label, sel in (("helped", sorted(q for q in ids if d[q] > 0)[:5]), ("hurt", sorted(q for q in ids if d[q] < 0)[:5])):
    print(f"**{label}**")
    for q in sel:
        a = (lambda df: final_letter(df.loc[q, "prediction"])) if V.loc[q, "options"] is not None and not isinstance(V.loc[q, "options"], float) else (lambda df: df.loc[q, "prediction"])
        rr = [(r["concept"], r["track_id"], round(float(r["x"]), 2), round(float(r["z"]), 2)) for r in csv.DictReader(open(REL_ROOT / irel[q]["pipeline_sheet"] / "objects.csv")) if r["concept"] in es.named_objects(irel[q])]
        line = es.relation_text(irel[q], str(REL_ROOT), "pipeline", "v3")
        print(f"- {q} ({qtype[q]}): {V.loc[q, 'question'].splitlines()[0][:150]}\n  truth {V.loc[q, 'ground_truth']}; video {a(V)} ({V.loc[q, 'score']:.2f}); dedup {a(D)} ({D.loc[q, 'score']:.2f}); dedup_relations {a(X)} ({X.loc[q, 'score']:.2f})"
              f"\n  named rows (metres): {rr}\n  relation: {line.splitlines()[-1] if line else '(none)'}")
    print()
