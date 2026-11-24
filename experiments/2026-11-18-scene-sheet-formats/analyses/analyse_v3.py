"""pipeline_v3 against video only on the 200 explore200 questions: per task, by named-object coverage, and selective
rules simulated from the existing predictions (chosen on these 200, so in-sample; plus a split-half estimate)."""
import csv, math, sys
from pathlib import Path
import numpy as np, pandas as pd
from mvp.benchmarks.vsibench import summarize, task_of


def md(df, index=True):
    df = df.reset_index() if index else df
    fmt = lambda x: f"{x:.1f}" if isinstance(x, float) else str(x)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(fmt(x) for x in r) + " |" for r in df.itertuples(index=False)])

R, V3 = Path(sys.argv[1]), Path(sys.argv[2])
extra = {n: Path(p) for n, p in (a.split("=", 1) for a in sys.argv[3:])}  # e.g. pipeline_v2=path.jsonl
runs = {"video": R / "A_video.jsonl", "pipeline_v3": R / "pipeline_v3.jsonl", **extra}
runs = {n: pd.read_json(p, lines=True).drop_duplicates("id").set_index("id") for n, p in runs.items()}
ids = sorted(runs["video"].index)
v, p3 = runs["video"].loc[ids], runs["pipeline_v3"].loc[ids]
idx = {int(r["id"]): r for r in csv.DictReader(open(V3 / "index.csv"))}
task = pd.Series({q: task_of(v.loc[q, "question_type"]) for q in ids})

def named(r):
    return [] if r["question_type"] == "room_size_estimation" else [p.strip() for p in r["prompts"].split(",") if p.strip() != "room"]
def coverage(q):
    r = idx[q]; n = named(r)
    if not n:
        return "no named objects"
    have = {x["concept"] for x in csv.DictReader(open(V3 / r["pipeline_sheet"] / "objects.csv"))}
    return "all present" if all(c in have for c in n) else "some missing"
cov = pd.Series({q: coverage(q) for q in ids})

def overall(scores):
    df = v.copy(); df["score"] = scores
    return summarize(df.reset_index())["overall"]
def sign_p(h, l):
    n = h + l
    if n == 0: return 1.0
    k = min(h, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0] * len(ps); m = len(ps); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); out[i] = run
    return out

# (1) per task
print(f"## (1) Per task ({len(ids)} questions)\n")
conds = list(runs)
rows = []
for t in sorted(set(task)):
    qs = task.index[task == t]
    row = {"task": t, "n": len(qs)}
    for c in conds:
        row[c] = 100 * runs[c].loc[qs, "score"].mean()
    for c in conds[1:]:
        d = runs[c].loc[qs, "score"] - v.loc[qs, "score"]
        row[f"{c} helped/hurt"] = f"{int((d > 0).sum())}/{int((d < 0).sum())}"
        row[f"_{c}_p"] = sign_p(int((d > 0).sum()), int((d < 0).sum()))
    rows.append(row)
for c in conds[1:]:
    adj = holm([r[f"_{c}_p"] for r in rows])
    for r, a in zip(rows, adj):
        r[f"{c} sign p (Holm)"] = f"{r.pop(f'_{c}_p'):.3f} ({a:.2f})"
tab = pd.DataFrame(rows).set_index("task")
tot = {"n": len(ids), **{c: overall(runs[c].loc[ids, "score"]) for c in conds}}
for c in conds[1:]:
    d = runs[c].loc[ids, "score"] - v["score"]
    h, l = int((d > 0).sum()), int((d < 0).sum())
    tot[f"{c} helped/hurt"] = f"{h}/{l}"; tot[f"{c} sign p (Holm)"] = f"{sign_p(h, l):.3f}"
tab.loc["overall (VSI-Bench)"] = tot
print(md(tab))

# (2) by coverage
print("\n## (2) By whether the v3 sheet has every object the question names (question text and sheet only)\n")
rows = []
for t in sorted(set(task)) + ["all tasks"]:
    for g in ("all present", "some missing", "no named objects"):
        qs = [q for q in ids if cov[q] == g and (t == "all tasks" or task[q] == t)]
        if not qs: continue
        d = p3.loc[qs, "score"] - v.loc[qs, "score"]
        rows.append({"task": t, "group": g, "n": len(qs), "video": 100 * v.loc[qs, "score"].mean(),
                     "pipeline_v3": 100 * p3.loc[qs, "score"].mean(), "v3 - video": 100 * d.mean(),
                     "helped/hurt": f"{int((d > 0).sum())}/{int((d < 0).sum())}",
                     "sign p": round(sign_p(int((d > 0).sum()), int((d < 0).sum())), 3)})
print(md(pd.DataFrame(rows), index=False))

# (3) selective rules
def rule_types(qs_fit):
    return {t for t in set(task) if p3.loc[[q for q in qs_fit if task[q] == t], "score"].mean() >
            v.loc[[q for q in qs_fit if task[q] == t], "score"].mean()}
def apply(rule, types=None):
    use = pd.Series(False, index=ids)
    for q in ids:
        if rule == "types": use[q] = task[q] in types
        elif rule == "present": use[q] = cov[q] == "all present"
        elif rule == "types and present": use[q] = task[q] in types and cov[q] == "all present"
        elif rule == "present, room size v3": use[q] = cov[q] in ("all present", "no named objects")
    return use
types_all = rule_types(ids)
print("\n## (3) Selective rules, from the existing predictions (IN-SAMPLE: chosen on these 200)\n")
print("types where v3 beats video on these 200:", sorted(types_all), "\n")
res = []
def add(name, use, note=""):
    s = p3["score"].where(use, v["score"])
    d = s - v["score"]
    res.append({"rule": name, "uses v3 on": int(use.sum()), "overall": overall(s), "minus video": overall(s) - tot["video"],
                "helped/hurt vs video": f"{int((d > 0).sum())}/{int((d < 0).sum())}", "note": note})
add("video only", pd.Series(False, index=ids))
add("pipeline_v3 always", pd.Series(True, index=ids))
add("v3 for the types where it helps", apply("types", types_all), "types chosen on these 200")
add("v3 when every named object is in its sheet", apply("present"), "fixed rule; room size -> video")
add("  same, room size -> v3", apply("present, room size v3"), "")
add("v3 for helping types AND all present", apply("types and present", types_all), "types chosen on these 200")
add("best of the two per question", (p3["score"] > v["score"]), "UPPER BOUND, needs the answer")
print(md(pd.DataFrame(res), index=False))

# split-half: choose on one half (stratified by task), score on the other
rng = np.random.default_rng(0); diffs = {"types": [], "present": [], "types and present": []}
for _ in range(2000):
    fit, test = [], []
    for t in set(task):
        qs = list(task.index[task == t]); rng.shuffle(qs); h = len(qs) // 2
        fit += qs[:h]; test += qs[h:]
    types = rule_types(fit)
    vt = v.loc[test].copy()
    for rule in diffs:
        use = apply(rule, types).loc[test]
        s = p3.loc[test, "score"].where(use, vt["score"])
        a = vt.copy(); a["score"] = s
        diffs[rule].append(summarize(a.reset_index())["overall"] - summarize(vt.reset_index())["overall"])
print("\nsplit-half estimate (rule chosen on a random half, scored on the other half, 2,000 splits): minus video, "
      "median [2.5%, 97.5%]")
for rule, d in diffs.items():
    print(f"  {rule:22s} {np.median(d):+.1f} [{np.percentile(d, 2.5):+.1f}, {np.percentile(d, 97.5):+.1f}]")
