"""prereg_batch2_A.md's registered results recomputed from the predictions, then exploratory analyses."""
import csv, hashlib, json, math, re, sys
from pathlib import Path
import numpy as np, pandas as pd
R, E, V4 = map(Path, sys.argv[1:4])
from mvp.benchmarks.vsibench import summarize, task_of, final_letter, final_number

ids = [int(x) for x in (E / "batch2_A_ids.txt").read_text().split()]
assert hashlib.sha256((E / "batch2_A_ids.txt").read_bytes()).hexdigest().startswith("ed82abf7")
print("## COMPLETENESS\n")
raw = {}
for n in ("A_video", "pipeline_v4"):
    d = pd.read_json(R / f"{n}.jsonl", lines=True); raw[n] = d
    s = json.loads((R / f"{n}_summary.json").read_text())
    print(f"{n}: {len(d)} lines, {d['id'].nunique()} distinct, = batch2_A_ids: {set(d['id']) == set(ids)}, missing scores {int(d['score'].isna().sum())}; "
          f"settings {[s.get(k) for k in ('model', 'nframes', 'sampling', 'max_new_tokens', 'final_answer', 'n')]}; sheet {s.get('scene_sheet')}")
A = raw["A_video"].drop_duplicates("id").set_index("id").loc[ids]; B = raw["pipeline_v4"].drop_duplicates("id").set_index("id").loc[ids]
print("same frames in both conditions:", all(A.loc[q, "frame_indices"] == B.loc[q, "frame_indices"] for q in ids))
rev = json.loads((R / "model_revision.json").read_text()); print("model:", rev["revision_served_before_runs"], "| snapshots after:", rev["snapshots_in_cache_after_runs"])

def sign_p(h, l):
    n = h + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(h, l) + 1)) / 2 ** n)
qtype = A["question_type"]
def st(qs):
    d = B.loc[qs, "score"] - A.loc[qs, "score"]; h, l = int((d > 0).sum()), int((d < 0).sum()); return h, l, sign_p(h, l)
DIST = [q for q in ids if qtype[q] in ("object_abs_distance", "object_rel_distance")]
DIR = [q for q in ids if qtype[q].startswith("object_rel_direction")]
h1, l1, p1 = st(DIST); h2, l2, p2 = st(DIR)
SHEET = {"object_abs_distance", "object_rel_distance", "object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard", "route_planning"}
full = B["score"].where(qtype.isin(SHEET), A["score"])
groups = {t: np.array([ids.index(q) for q in qtype.index[qtype == t]]) for t in sorted(set(qtype))}
def ov(m):
    t = {}
    for k, v in m.items(): t.setdefault(task_of(k), []).append(v)
    return 100 * np.mean([np.mean(x) for x in t.values()])
f_s, a_s = full.to_numpy(), A["score"].to_numpy()
fo, ao = ov({t: f_s[g].mean() for t, g in groups.items()}), ov({t: a_s[g].mean() for t, g in groups.items()})
assert abs(ao - summarize(A.reset_index())["overall"]) < 1e-9
rng = np.random.default_rng(0); dr = np.empty(10000)
for i in range(10000):
    s = {t: rng.choice(g, len(g)) for t, g in groups.items()}
    dr[i] = ov({t: f_s[k].mean() for t, k in s.items()}) - ov({t: a_s[k].mean() for t, k in s.items()})
p3 = min(1.0, 2 * min(np.mean(dr <= 0), np.mean(dr >= 0)))
order = sorted([("direction", p2), ("full", p3)], key=lambda x: x[1])
holm = {}; run = 0
for i, (k, p) in enumerate(order): run = max(run, min(1, (2 - i) * p)); holm[k] = run
print("\n## REGISTERED RESULTS (recomputed)\n")
print(f"H1 primary, 309 distance: helped {h1}, hurt {l1}, unchanged {len(DIST) - h1 - l1}; sign test p = {p1:.6f} -> {'CONFIRMED' if p1 < 0.05 and h1 > l1 else 'NOT CONFIRMED'}")
print(f"H2, 110 direction: helped {h2}, hurt {l2}, unchanged {len(DIR) - h2 - l2}; p = {p2:.6f}, Holm {holm['direction']:.6f} -> {'SUPPORTED' if holm['direction'] < 0.05 and h2 > l2 else 'NOT SUPPORTED'}")
print(f"H3, full method overall: video {ao:.2f}, full {fo:.2f}, diff {fo - ao:+.2f} [{np.percentile(dr, 2.5):+.2f}, {np.percentile(dr, 97.5):+.2f}], bootstrap p {p3:.4f}, Holm {holm['full']:.4f} "
      f"-> {'SUPPORTED' if holm['full'] < 0.05 and fo > ao else 'NOT SUPPORTED'}")
nb = pd.read_csv(R / "preregistered_test.csv"); print("\nnotebook's csv:\n" + nb.to_string(index=False))

print("\n## EXPLORATORY (not registered)\n")
rows = []
for t, g in groups.items():
    b = B["score"].to_numpy()[g]
    rows.append([t, len(g), round(100 * a_s[g].mean(), 1), round(100 * b.mean(), 1), round(100 * f_s[g].mean(), 1), "v4" if t in SHEET else "video", f"{int((b > a_s[g]).sum())}/{int((b < a_s[g]).sum())}"])
print(pd.DataFrame(rows, columns=["type", "n", "A_video", "pipeline_v4", "full method", "answer used", "v4 vs video h/h"]).to_string(index=False))
rel = {int(q): t for q, t in json.loads((V4 / "relations_v4.json").read_text()).items()}
idx = {int(r["id"]): r for r in csv.DictReader(open(V4 / "index.csv"))}
print()
for t in ("object_abs_distance", "object_rel_distance"):
    for has in (True, False):
        qs = [q for q in ids if qtype[q] == t and (q in rel) == has]
        d = B.loc[qs, "score"] - A.loc[qs, "score"]
        print(f"{t}, {'with' if has else 'without'} relations ({len(qs)}): video {100 * A.loc[qs, 'score'].mean():.1f}, v4 {100 * B.loc[qs, 'score'].mean():.1f}, helped/hurt {int((d > 0).sum())}/{int((d < 0).sum())}")
dq = [q for q in DIR]
for has in (True, False):
    qs = [q for q in dq if (q in rel) == has]; d = B.loc[qs, "score"] - A.loc[qs, "score"]
    print(f"relative direction, {'with' if has else 'without'} relations ({len(qs)}): video {100 * A.loc[qs, 'score'].mean():.1f}, v4 {100 * B.loc[qs, 'score'].mean():.1f}, helped/hurt {int((d > 0).sum())}/{int((d < 0).sum())}")
mra = lambda p, g: float(np.mean([abs(p - g) / g < 1 - t for t in np.arange(0.5, 1.0, 0.05)]))
aq = [q for q in ids if qtype[q] == "object_abs_distance" and q in rel]
pre = {q: float(re.search(r"Closest-point distance between .*?: ([0-9.]+) m", rel[q]).group(1)) for q in aq}
tr = {q: float(A.loc[q, "ground_truth"]) for q in aq}
r = np.array([pre[q] / tr[q] for q in aq])
cop = sum(final_number(B.loc[q, "prediction"]) is not None and abs(final_number(B.loc[q, "prediction"]) - pre[q]) <= 0.01 * pre[q] + 0.005 for q in aq)
print(f"\nabsolute distance relations ({len(aq)}): MRA vs truth {100 * np.mean([mra(pre[q], tr[q]) for q in aq]):.1f}; pred/truth median {np.median(r):.2f} (quartiles {np.percentile(r, 25):.2f}-{np.percentile(r, 75):.2f}); within 25%: {int((abs(r - 1) <= .25).sum())}; copied {cop}/{len(aq)}")
rq = [q for q in ids if qtype[q] == "object_rel_distance" and q in rel]
def closest(q):
    m = re.search(r"Closest to the .*?: (.+)\.$", rel[q].strip().splitlines()[-1])
    return next((o.split(". ", 1)[0] for o in A.loc[q, "options"] if o.split(". ", 1)[1].strip() == m.group(1)), None) if m else None
cl = {q: closest(q) for q in rq}
print(f"relative distance relations ({len(rq)}): closest right {sum(cl[q] == A.loc[q, 'ground_truth'] for q in rq)}/{len(rq)}; copied {sum(final_letter(B.loc[q, 'prediction']) == cl[q] for q in rq)}/{len(rq)}")
dq2 = [q for q in DIR if q in rel]
def dir_letter(q):
    w = re.search(r"is (?:to the|to your) (.+?)(?: of the .+)?\.$", rel[q]).group(1)
    return next((o.split(". ", 1)[0] for o in A.loc[q, "options"] if o.split(". ", 1)[1].strip().lower() == w.lower()), None)
dl = {q: dir_letter(q) for q in dq2}
print(f"direction relations ({len(dq2)}): mapped to an option {sum(v is not None for v in dl.values())}; right {sum(dl[q] == A.loc[q, 'ground_truth'] for q in dq2)}/{len(dq2)}; copied {sum(final_letter(B.loc[q, 'prediction']) == dl[q] for q in dq2)}/{len(dq2)}")
sc = json.loads((V4 / "scales_v4.json").read_text())
print("scale components (median over scenes): median", round(np.median([v['scale'] for v in sc.values()]), 3), "| priors/median", round(np.median([v['priors'] / v['scale'] for v in sc.values()]), 2),
      "| room/median", round(np.median([v['room_height'] / v['scale'] for v in sc.values()]), 2))
