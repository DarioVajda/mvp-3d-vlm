"""dev200: video, pipeline_v3 and pipeline_v3_fixed per task; the selective rule fixed on explore200, tested here
unchanged; and whether the Not detected line moved answers toward video's."""
import csv, math, sys
from pathlib import Path
import numpy as np, pandas as pd
from mvp.benchmarks.vsibench import summarize, task_of, final_letter, final_number, MCA_QUESTION_TYPES

R, V3 = Path(sys.argv[1]), Path(sys.argv[2])
runs = {n: pd.read_json(R / f"{n}.jsonl", lines=True).drop_duplicates("id").set_index("id")
        for n in ("A_video", "pipeline_v3", "pipeline_v3_fixed")}
ids = sorted(runs["A_video"].index)
assert all(set(r.index) == set(ids) for r in runs.values()) and len(ids) == 200
v, p3, pf = (runs[n].loc[ids] for n in ("A_video", "pipeline_v3", "pipeline_v3_fixed"))
task = pd.Series({q: task_of(v.loc[q, "question_type"]) for q in ids})
idx = {int(r["id"]): r for r in csv.DictReader(open(V3 / "index.csv"))}

def md(df, index=True):
    df = df.reset_index() if index else df
    fmt = lambda x: f"{x:.1f}" if isinstance(x, float) else str(x)
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)] +
                     ["| " + " | ".join(fmt(x) for x in r) + " |" for r in df.itertuples(index=False)])
def sign_p(h, l):
    n = h + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(h, l) + 1)) / 2 ** n)
def holm(ps):
    order, out, run, m = sorted(range(len(ps)), key=lambda i: ps[i]), [0] * len(ps), 0, len(ps)
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); out[i] = run
    return out
def overall(scores):
    df = v.copy(); df["score"] = scores
    return summarize(df.reset_index())["overall"]

# (1) per task
rows, tasks = [], sorted(set(task))
for t in tasks:
    qs = task.index[task == t]
    task_score = lambda run: summarize(run.loc[qs].reset_index())[t]  # relative direction: the mean of its sub-types
    row = {"task": t, "n": len(qs), "video": task_score(v), "v3": task_score(p3), "v3_fixed": task_score(pf)}
    for name, run in (("v3", p3), ("v3_fixed", pf)):
        d = run.loc[qs, "score"] - v.loc[qs, "score"]
        h, l = int((d > 0).sum()), int((d < 0).sum())
        row[f"{name} helped/hurt"], row[f"_{name}"] = f"{h}/{l}", sign_p(h, l)
    d = pf.loc[qs, "score"] - p3.loc[qs, "score"]
    row["fixed vs v3 helped/hurt"] = f"{int((d > 0).sum())}/{int((d < 0).sum())}"
    rows.append(row)
for name in ("v3", "v3_fixed"):
    for r, a in zip(rows, holm([r[f"_{name}"] for r in rows])):
        r[f"{name} sign p (Holm)"] = f"{r.pop(f'_{name}'):.3f} ({a:.2f})"
tab = pd.DataFrame(rows).set_index("task")[["n", "video", "v3", "v3_fixed", "v3 helped/hurt", "v3 sign p (Holm)",
                                            "v3_fixed helped/hurt", "v3_fixed sign p (Holm)", "fixed vs v3 helped/hurt"]]
tot = {"n": 200, "video": overall(v["score"]), "v3": overall(p3["score"]), "v3_fixed": overall(pf["score"])}
for name, run in (("v3", p3), ("v3_fixed", pf)):
    d = run["score"] - v["score"]; h, l = int((d > 0).sum()), int((d < 0).sum())
    tot[f"{name} helped/hurt"], tot[f"{name} sign p (Holm)"] = f"{h}/{l}", f"{sign_p(h, l):.3f}"
d = pf["score"] - p3["score"]; tot["fixed vs v3 helped/hurt"] = f"{int((d > 0).sum())}/{int((d < 0).sum())}"
tab.loc["overall (VSI-Bench)"] = tot
print("## (1) Per task, dev200\n\n" + md(tab))

# (2) the rule fixed on explore200, unchanged: v3 when every named object is in its v3 sheet, room size -> video
def named(r):
    return [] if r["question_type"] == "room_size_estimation" else [p.strip() for p in r["prompts"].split(",") if p.strip() != "room"]
def all_present(q):
    n = named(idx[q])
    if not n:
        return False  # room size -> video
    have = {x["concept"] for x in csv.DictReader(open(V3 / idx[q]["pipeline_sheet"] / "objects.csv"))}
    return all(c in have for c in n)
use = pd.Series({q: all_present(q) for q in ids})
rule = p3["score"].where(use, v["score"])
rule_fixed = pf["score"].where(use, v["score"])

qtype = v["question_type"]
QTYPES = sorted(set(qtype))
def overall_of(means):
    """VSI-Bench's overall from {question type: mean score}, as summarize: relative direction is the mean of its
    sub-types, overall the mean over the tasks."""
    tasks_ = {}
    for q, m in means.items():
        tasks_.setdefault(task_of(q), []).append(m)
    return 100 * np.mean([np.mean(x) for x in tasks_.values()])
def boot(a, b, n=10000, seed=0):
    """Paired bootstrap of overall(a) - overall(b), resampling questions within each question type."""
    rng = np.random.default_rng(seed)
    groups = {t: np.array([ids.index(q) for q in qtype.index[qtype == t]]) for t in QTYPES}
    A, B = a.loc[ids].to_numpy(), b.loc[ids].to_numpy()
    out = np.empty(n)
    for i in range(n):
        s = {t: rng.choice(g, len(g)) for t, g in groups.items()}
        out[i] = overall_of({t: A[k].mean() for t, k in s.items()}) - overall_of({t: B[k].mean() for t, k in s.items()})
    return out
# the bootstrap's statistic is VSI-Bench's overall exactly
for run in (v, p3, pf):
    assert abs(overall_of({t: run.loc[qtype.index[qtype == t], "score"].mean() for t in QTYPES}) - overall(run["score"])) < 1e-9
res = []
for name, s in (("video only", v["score"]), ("pipeline_v3 always", p3["score"]), ("pipeline_v3_fixed always", pf["score"]),
                ("RULE (frozen on explore200): v3 when all named objects present, else video", rule),
                ("same rule with v3_fixed's answers (not the frozen rule)", rule_fixed)):
    d = s - v["score"]
    b = boot(s, v["score"]) if name != "video only" else np.zeros(1)
    res.append({"condition": name, "uses sheet on": int(use.sum()) if "rule" in name.lower() else ("200" if "always" in name else "0"),
                "overall": overall(s), "minus video": overall(s) - tot["video"],
                "95% CI of minus video": "" if name == "video only" else f"[{np.percentile(b, 2.5):+.1f}, {np.percentile(b, 97.5):+.1f}]",
                "P(minus video > 0)": "" if name == "video only" else f"{(b > 0).mean():.2f}",
                "helped/hurt": f"{int((d > 0).sum())}/{int((d < 0).sum())}",
                "sign p": "" if name == "video only" else f"{sign_p(int((d > 0).sum()), int((d < 0).sum())):.3f}"})
print("\n## (2) The rule fixed on explore200, tested unchanged on dev200 (stratified paired bootstrap, 10,000 resamples)\n\n" + md(pd.DataFrame(res), index=False))
print(f"\nquestions where the rule uses v3: {int(use.sum())} of 200 (all named objects present); by task: "
      + ", ".join(f"{t} {int(use[task == t].sum())}" for t in tasks))
cov = pd.Series({q: ("no named objects" if not named(idx[q]) else "all present" if use[q] else "some missing") for q in ids})
rows = []
for g in ("all present", "some missing", "no named objects"):
    qs = cov.index[cov == g]
    d = p3.loc[qs, "score"] - v.loc[qs, "score"]
    rows.append({"group": g, "n": len(qs), "video": 100 * v.loc[qs, "score"].mean(), "v3": 100 * p3.loc[qs, "score"].mean(),
                 "v3_fixed": 100 * pf.loc[qs, "score"].mean(), "v3 helped/hurt": f"{int((d > 0).sum())}/{int((d < 0).sum())}"})
print("\nby coverage (as on explore200):\n\n" + md(pd.DataFrame(rows), index=False))

# (3) did the Not detected line move answers toward video's?
def answer(run, q):
    p, t = run.loc[q, "prediction"], run.loc[q, "question_type"]
    return final_letter(p) if t in MCA_QUESTION_TYPES else final_number(p)
def same(a, b):
    if a is None or b is None: return False
    if isinstance(a, str) or isinstance(b, str): return a == b
    return abs(a - b) <= 1e-9 * max(1, abs(b))
def closer(a3, af, av):
    if any(x is None for x in (a3, af, av)) or isinstance(av, str): return None
    return abs(af - av) < abs(a3 - av)
print("\n## (3) The Not detected line: v3_fixed's answers against v3's, measured against video's answer\n")
for label, sel in (("questions with the Not detected line, object-size questions left out (only this fix applies)",
                    lambda q: cov[q] == "some missing" and v.loc[q, "question_type"] != "object_size_estimation"),
                   ("object-size questions (no sizes; and the line where an object is missing)",
                    lambda q: v.loc[q, "question_type"] == "object_size_estimation"),
                   ("questions with every named object present (neither fix applies, apart from object size)",
                    lambda q: cov[q] == "all present" and v.loc[q, "question_type"] != "object_size_estimation")):
    qs = [q for q in ids if sel(q)]
    a3 = {q: answer(p3, q) for q in qs}; af = {q: answer(pf, q) for q in qs}; av = {q: answer(v, q) for q in qs}
    changed = [q for q in qs if not same(af[q], a3[q])]
    to_video = [q for q in changed if same(af[q], av[q])]
    away = [q for q in changed if same(a3[q], av[q])]
    nearer = [q for q in changed if closer(a3[q], af[q], av[q])]
    d = pf.loc[qs, "score"] - p3.loc[qs, "score"]
    print(f"- {label}: {len(qs)} questions; v3_fixed's answer differs from v3's in {len(changed)}: "
          f"now video's answer {len(to_video)}, was video's answer before {len(away)}, numeric and nearer video's {len(nearer)}; "
          f"answers equal to video's: v3 {sum(same(a3[q], av[q]) for q in qs)}, v3_fixed {sum(same(af[q], av[q]) for q in qs)}; "
          f"score v3 {100 * p3.loc[qs, 'score'].mean():.1f} -> v3_fixed {100 * pf.loc[qs, 'score'].mean():.1f} "
          f"(video {100 * v.loc[qs, 'score'].mean():.1f}); fixed vs v3 helped/hurt {int((d > 0).sum())}/{int((d < 0).sum())}")

# (4) the other rule chosen on explore200 (the tasks where v3 helped there), tested unchanged on dev
EXPLORE_TASKS = {"object_rel_direction", "route_planning"}
use_t = pd.Series({q: task[q] in EXPLORE_TASKS for q in ids})
rows = []
for name, run in (("v3", p3), ("v3_fixed", pf)):
    s = run["score"].where(use_t, v["score"])
    d = s - v["score"]; b = boot(s, v["score"])
    rows.append({"rule": f"{name} for relative direction and route planning (chosen on explore200), else video",
                 "uses sheet on": int(use_t.sum()), "overall": overall(s), "minus video": overall(s) - tot["video"],
                 "95% CI": f"[{np.percentile(b, 2.5):+.1f}, {np.percentile(b, 97.5):+.1f}]", "P(> 0)": f"{(b > 0).mean():.2f}",
                 "helped/hurt": f"{int((d > 0).sum())}/{int((d < 0).sum())}",
                 "sign p": f"{sign_p(int((d > 0).sum()), int((d < 0).sum())):.3f}"})
print("\n## (4) The task rule from explore200, tested unchanged on dev200\n\n" + md(pd.DataFrame(rows), index=False))
