"""dev200: pipeline_v3_dedup against A_video and pipeline_v3 (the same 200 questions). Dev results: chosen and
analysed on dev, so descriptive; nothing here is a confirmatory test."""
import csv, itertools, math, re, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
from mvp.benchmarks import scene_scale
from mvp.benchmarks.vsibench import summarize, task_of, final_letter

R, D = Path(sys.argv[1]), Path(sys.argv[2])
runs = {n: pd.read_json(R / f"{n}.jsonl", lines=True).drop_duplicates("id").set_index("id") for n in ("A_video", "pipeline_v3", "pipeline_v3_dedup")}
ids = sorted(runs["A_video"].index)
V, P, Q = (runs[n].loc[ids] for n in ("A_video", "pipeline_v3", "pipeline_v3_dedup"))
qtype = V["question_type"]; task = qtype.map(task_of)
V3R, DDR = D / "pipeline_v3_dev50", D / "pipeline_v3_dedup_dev50"
i3 = {int(r["id"]): r for r in csv.DictReader(open(V3R / "index.csv"))}
idd = {int(r["id"]): r for r in csv.DictReader(open(DDR / "index.csv"))}
def md(df):
    df = df.reset_index()
    f = lambda x: f"{x:.1f}" if isinstance(x, float) else str(x)
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)] + ["| " + " | ".join(f(x) for x in r) + " |" for r in df.itertuples(index=False)])
def sign_p(h, l):
    n = h + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(h, l) + 1)) / 2 ** n)
def overall(df, s=None):
    df = df.copy()
    if s is not None: df["score"] = s
    return summarize(df.reset_index())["overall"]
QT = sorted(set(qtype))
def overall_of(means):
    t = {}
    for q, m in means.items(): t.setdefault(task_of(q), []).append(m)
    return 100 * np.mean([np.mean(x) for x in t.values()])
def boot(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    g = {t: np.array([ids.index(q) for q in qtype.index[qtype == t]]) for t in QT}
    A_, B_ = a.to_numpy(), b.to_numpy(); out = np.empty(n)
    for i in range(n):
        s = {t: rng.choice(x, len(x)) for t, x in g.items()}
        out[i] = overall_of({t: A_[k].mean() for t, k in s.items()}) - overall_of({t: B_[k].mean() for t, k in s.items()})
    return out
for df in (V, P, Q):
    assert abs(overall_of({t: df.loc[qtype.index[qtype == t], "score"].mean() for t in QT}) - overall(df)) < 1e-9

# 1. overall
print("## 1. Overall (dev200; dev results)\n")
print(f"VSI-Bench overall: A_video {overall(V):.2f}, pipeline_v3 {overall(P):.2f}, pipeline_v3_dedup {overall(Q):.2f}\n")
rows = []
for name, a, b in (("v3_dedup - video", Q, V), ("v3_dedup - v3", Q, P), ("v3 - video", P, V)):
    d = a["score"] - b["score"]; h, l = int((d > 0).sum()), int((d < 0).sum()); bs = boot(a["score"], b["score"])
    rows.append({"comparison": name, "difference": overall(a) - overall(b), "95% CI": f"[{np.percentile(bs, 2.5):+.1f}, {np.percentile(bs, 97.5):+.1f}]",
                 "P(> 0)": f"{(bs > 0).mean():.2f}", "helped / hurt": f"{h} / {l}", "sign test p": f"{sign_p(h, l):.3f}"})
print(md(pd.DataFrame(rows).set_index("comparison")))

# 2. per task
rows = []
for t in sorted(set(task)):
    qs = task.index[task == t]
    sc = lambda df: summarize(df.loc[qs].reset_index())[t]
    dv, d3 = Q.loc[qs, "score"] - V.loc[qs, "score"], Q.loc[qs, "score"] - P.loc[qs, "score"]
    rows.append({"task": t, "n": len(qs), "video": sc(V), "v3": sc(P), "v3_dedup": sc(Q),
                 "dedup vs video helped/hurt": f"{int((dv > 0).sum())}/{int((dv < 0).sum())}",
                 "dedup vs v3 helped/hurt": f"{int((d3 > 0).sum())}/{int((d3 < 0).sum())}"})
print("\n## 2. Per task (descriptive)\n\n" + md(pd.DataFrame(rows).set_index("task")))

# 3. did dedup do its job
TYPES = {"object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard", "object_rel_distance", "object_abs_distance"}
named = lambda r: [p.strip() for p in r["prompts"].split(",") if p.strip() and p.strip() != "room"]
rows_of = lambda root, ix, q: list(csv.DictReader(open(root / ix[q]["pipeline_sheet"] / "objects.csv")))
clean = lambda root, ix, q: all(sum(x["concept"] == c for x in rows_of(root, ix, q)) == 1 for c in named(ix[q]))
dq = [q for q in ids if qtype[q] in TYPES]
before = {q: clean(V3R, i3, q) for q in dq}; after = {q: clean(DDR, idd, q) for q in dq}
changed = {q: "__q" in idd[q]["pipeline_sheet"] for q in dq}
missing = {q: not all(any(x["concept"] == c for x in rows_of(DDR, idd, q)) for c in named(idd[q])) for q in dq}
print(f"\n## 3. Did dedup do its job? ({len(dq)} relative-direction, relative-distance and absolute-distance questions; descriptive)\n")
print(f"clean sheets (each named object exactly once): before {sum(before.values())} ({100 * np.mean(list(before.values())):.0f}%), "
      f"after {sum(after.values())} ({100 * np.mean(list(after.values())):.0f}%); sheets changed by dedup {sum(changed.values())}; "
      f"not clean after dedup because an object is missing: {sum(missing.values())}")
rows = []
for g, sel in (("clean after dedup, was duplicated (dedup changed the sheet)", lambda q: after[q] and changed[q]),
               ("clean before and after", lambda q: after[q] and not changed[q]),
               ("not clean: an object missing", lambda q: not after[q])):
    qs = [q for q in dq if sel(q)]
    if not qs: continue
    d3 = Q.loc[qs, "score"] - P.loc[qs, "score"]; dv = Q.loc[qs, "score"] - V.loc[qs, "score"]
    rows.append({"group": g, "n": len(qs), "video": 100 * V.loc[qs, "score"].mean(), "v3": 100 * P.loc[qs, "score"].mean(),
                 "v3_dedup": 100 * Q.loc[qs, "score"].mean(), "dedup vs v3 helped/hurt": f"{int((d3 > 0).sum())}/{int((d3 < 0).sum())}",
                 "dedup vs video helped/hurt": f"{int((dv > 0).sum())}/{int((dv < 0).sum())}"})
print("\n(mean score x100; relative distance and direction are 0/1, absolute distance MRA)\n\n" + md(pd.DataFrame(rows).set_index("group")))
print(f"\nanswers that differ between v3 and v3_dedup: on changed sheets {sum(final_letter(P.loc[q, 'prediction']) != final_letter(Q.loc[q, 'prediction']) if qtype[q] != 'object_abs_distance' else P.loc[q, 'prediction'] != Q.loc[q, 'prediction'] for q in dq if changed[q])} of {sum(changed.values())}; "
      f"on every other question {sum(P.loc[q, 'prediction'] != Q.loc[q, 'prediction'] for q in ids if not changed.get(q, False))} of {len(ids) - sum(changed.values())}")

# 4. the "A" bias and the sheet-implied direction
REL = [q for q in ids if qtype[q].startswith("object_rel_direction")]
L = {n: {q: final_letter(df.loc[q, "prediction"]) for q in REL} for n, df in (("video", V), ("v3", P), ("v3_dedup", Q))}
print(f"\n## 4. Relative direction ({len(REL)} questions; descriptive)\n")
rows = []
for name, f in (("truth", lambda q: V.loc[q, "ground_truth"]),) + tuple((n, (lambda n: lambda q: L[n][q])(n)) for n in L):
    c = Counter(f(q) for q in REL); rows.append({"": name, **{x: c.get(x, 0) for x in "ABCD"}})
print(md(pd.DataFrame(rows).set_index("")))
def parse(text):
    m = re.search(r"standing by the (.+?) and facing the (.+?), is the (.+?) (?:to (?:my|the)|in)", text)
    return m.groups() if m else None
def direction(t, X, Y, Z):
    Pp = lambda v: np.array([v[0], v[2]]); right = lambda f: np.array([f[1], -f[0]])
    f = Pp(Y) - Pp(X)
    if t.endswith("easy"): return "right" if (Pp(Z) - Pp(Y)) @ right(f) > 0 else "left"
    z = Pp(Z) - Pp(X); ang = math.degrees(math.acos(np.clip(z @ f / (np.linalg.norm(z) * np.linalg.norm(f) + 1e-12), -1, 1)))
    side = "right" if z @ right(f) > 0 else "left"
    return ("back" if ang >= 135 else side) if t.endswith("medium") else ("front-" if z @ f > 0 else "back-") + side
pos = lambda r: np.array([float(r[k]) for k in "xyz"])
def letter_of(q, word):
    for o in V.loc[q, "options"]:
        k, v = o.split(". ", 1)
        if v.strip() == word: return k
implied = {}
for q in REL:
    names = parse(V.loc[q, "question"]); rs = rows_of(DDR, idd, q)
    by = {c: [x for x in rs if x["concept"] == c] for c in names} if names else {}
    if names and all(len(v) == 1 for v in by.values()):
        implied[q] = letter_of(q, direction(qtype[q], *(pos(by[c][0]) for c in names)))
qi = list(implied)
print(f"\nquestions whose dedup sheet has each named object once (one implied direction): {len(qi)} of {len(REL)}; the implied direction is right in {sum(implied[q] == V.loc[q, 'ground_truth'] for q in qi)}")
for n in L:
    print(f"- {n}: answers the sheet-implied direction in {sum(L[n][q] == implied[q] for q in qi)} of {len(qi)}; right in {sum(L[n][q] == V.loc[q, 'ground_truth'] for q in qi)}")
chg = [q for q in REL if L["v3_dedup"][q] != L["video"][q]]
off_a = [q for q in chg if L["video"][q] == "A"]
print(f"\nanswers differing from video's: {len(chg)}; of them video answered 'A' in {len(off_a)}; v3_dedup's new answer is the sheet-implied one in "
      f"{sum(q in implied and L['v3_dedup'][q] == implied[q] for q in chg)} of the {sum(q in implied for q in chg)} with one implied direction")
chg3 = [q for q in REL if L["v3_dedup"][q] != L["v3"][q]]
print(f"answers differing from v3's: {len(chg3)}; v3_dedup's answer is the sheet-implied one in {sum(q in implied and L['v3_dedup'][q] == implied[q] for q in chg3)} "
      f"of the {sum(q in implied for q in chg3)} with one implied direction; v3's was in {sum(q in implied and L['v3'][q] == implied[q] for q in chg3)}")

# 5. examples: dedup vs v3, the first 5 helped and hurt by id
d3 = Q["score"] - P["score"]
helped, hurt = sorted(q for q in ids if d3[q] > 0)[:5], sorted(q for q in ids if d3[q] < 0)[:5]
def k_of(q): return scene_scale.question_scale(q, str(DDR))["k"]
def fmt_rows(root, ix, q):
    k = k_of(q); rs = rows_of(root, ix, q); out = []
    for c in named(ix[q]):
        mine = [x for x in rs if x["concept"] == c]
        out.append(f"{c} x{len(mine)}" + "".join(f" [({float(x['x']) * k:.2f},{float(x['y']) * k:.2f},{float(x['z']) * k:.2f}) m, {x['num_frames']} fr]" for x in mine))
    return "; ".join(out)
print(f"\n## 5. Examples: v3_dedup against pipeline_v3, the first 5 helped and the first 5 hurt by question id (descriptive)")
for label, qs in (("helped", helped), ("hurt", hurt)):
    print(f"\n**{label}**")
    for q in qs:
        ans = lambda df: final_letter(df.loc[q, "prediction"]) if V.loc[q, "options"] is not None and not (isinstance(V.loc[q, "options"], float)) else df.loc[q, "prediction"]
        print(f"\n- {q} ({qtype[q]}): {V.loc[q, 'question'].splitlines()[0][:170]}")
        print(f"  options {V.loc[q, 'options']}; truth {V.loc[q, 'ground_truth']}; video {ans(V)} ({V.loc[q, 'score']:.2f}); v3 {ans(P)} ({P.loc[q, 'score']:.2f}); v3_dedup {ans(Q)} ({Q.loc[q, 'score']:.2f}); "
              f"dedup sheet implies {implied.get(q, '-')}; sheet changed by dedup: {changed.get(q, False)}")
        print(f"  before: {fmt_rows(V3R, i3, q)}")
        if changed.get(q, False): print(f"  after:  {fmt_rows(DDR, idd, q)}")
