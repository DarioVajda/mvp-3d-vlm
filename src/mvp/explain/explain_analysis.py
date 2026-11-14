"""Reads results/explain_v7 (explain_saliency.py, explain_ablate.py). Run from the main mvp directory."""
import glob, itertools, json, os, sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from mvp.benchmarks.vsibench import TASKS, summarize, task_of

SAL = "results/explain_v7/sal"
RUNS = ["additive_s0", "additive_s1", "xa3_both_s0", "xa3_both_s1", "lora_s0", "lora_s1"]
KEYS = ["0", "1", "2"]
branch = lambda r: f"results/train/2B_v7_{r}/branches/step10750/eval/vsi_f32_full.jsonl"
rec = {r: {d["id"]: d for d in map(json.loads, open(f"{SAL}/{r}/summary.jsonl"))} for r in RUNS}
ids = sorted(set.intersection(*(set(v) for v in rec.values())))
print(len(ids), "questions in all six runs;", {r: len(v) for r, v in rec.items()})
qt = {i: rec[RUNS[0]][i]["question_type"] for i in ids}
base = {r: {d["id"]: d for d in map(json.loads, open(branch(r)))} for r in RUNS}
print("\n=== sanity: top token of my forward vs first character of the stored evaluation prediction (multiple choice)")
for r in RUNS:
    x = [(rec[r][i]["top_token"].strip()[:1] == str(base[r][i]["prediction"]).strip()[:1]) for i in ids if base[r][i]["options"] is not None]
    print(f"  {r:12s} {np.mean(x):.3f} (n {len(x)})")
if any("flow" in rec[r][ids[0]] and rec[r][ids[0]]["flow"] for r in RUNS):
    e = [f["recon_err"] for r in RUNS[2:4] for i in ids for f in rec[r][i]["flow"].values()]
    print(f"  flow reconstruction error of Delta: median {np.median(e):.4f}, max {np.max(e):.4f}")
    c = [abs(sum(f[f'attr_{s}'] for s in ('self', 'same_frame', 'other_near', 'other_far')) - f['attr_total']) / max(abs(f['attr_total']), 1e-3) for r in RUNS[2:4] for i in ids for f in rec[r][i]["flow"].values()]
    print(f"  flow completeness (parts vs total, relative): median {np.median(c):.4f}")


def maps(r, i):
    z = np.load(f"{SAL}/{r}/{i}.npz")
    out = {"vis": z["vis"], "grid": z["grid"]}
    if "dest_0" in z:
        out["dest"] = sum(z[f"dest_{k}"] for k in KEYS)
        out["src"] = sum(z[f"src_{k}"] for k in KEYS)
        for k in KEYS:
            out[f"dest_{k}"] = z[f"dest_{k}"]
    return out


rank = lambda x: np.argsort(np.argsort(x)).astype(float)
spear = lambda a, b: np.corrcoef(rank(a), rank(b))[0, 1]
def overlap(a, b, f=.05):
    n = max(1, int(len(a) * f))
    return len(set(np.argpartition(-a, n)[:n]) & set(np.argpartition(-b, n)[:n])) / n
def frames(a, grid):
    c = (grid.prod(-1) // 4)
    s = np.split(a, np.cumsum(c)[:-1])
    return np.array([x.sum() for x in s])
M = {r: {i: maps(r, i) for i in ids} for r in RUNS}
PAIRS = {"additive s0-s1": [("additive_s0", "additive_s1")], "xattn s0-s1": [("xa3_both_s0", "xa3_both_s1")],
         "additive-xattn": [(a, b) for a in RUNS[:2] for b in RUNS[2:4]]}
print("\n=== map similarity per question, mean [question bootstrap 95%]; chance overlap 0.05")
rng = np.random.default_rng(0)
ci = lambda v: f"{np.mean(v):.3f} [{np.percentile([np.mean(rng.choice(v, len(v))) for _ in range(1000)], 2.5):.3f}, {np.percentile([np.mean(rng.choice(v, len(v))) for _ in range(1000)], 97.5):.3f}]"
SIM = {}
for field in ("dest", "src", "vis"):
    print(field)
    for lab, ps in PAIRS.items():
        sp = [np.mean([spear(M[a][i][field], M[b][i][field]) for a, b in ps]) for i in ids]
        ov = [np.mean([overlap(M[a][i][field], M[b][i][field]) for a, b in ps]) for i in ids]
        fr = [np.mean([spear(frames(M[a][i][field], M[a][i]["grid"]), frames(M[b][i][field], M[b][i]["grid"])) for a, b in ps]) for i in ids]
        SIM[(field, lab)] = (sp, ov, fr)
        print(f"  {lab:16s} rank corr {ci(sp)}  top5% overlap {ci(ov)}  frame-mass rank corr {ci(fr)}")
    if field == "vis":
        for lab, ps in {"LoRA s0-s1": [("lora_s0", "lora_s1")], "LoRA-additive": [(a, b) for a in RUNS[4:] for b in RUNS[:2]], "LoRA-xattn": [(a, b) for a in RUNS[4:] for b in RUNS[2:4]]}.items():
            sp = [np.mean([spear(M[a][i][field], M[b][i][field]) for a, b in ps]) for i in ids]
            ov = [np.mean([overlap(M[a][i][field], M[b][i][field]) for a, b in ps]) for i in ids]
            print(f"  {lab:16s} rank corr {ci(sp)}  top5% overlap {ci(ov)}")
# question specificity: same model, same video, two different questions
byv = {}
for i in ids:
    byv.setdefault(rec[RUNS[0]][i]["video"], []).append(i)
pairs = [p for v in byv.values() for p in itertools.combinations(v, 2)]
print(f"question specificity: {len(pairs)} question pairs on one video, same model")
for field, rs in (("dest", RUNS[:4]), ("vis", RUNS)):
    sp = [np.mean([spear(M[r][a][field], M[r][b][field]) for r in rs]) for a, b in pairs]
    ov = [np.mean([overlap(M[r][a][field], M[r][b][field]) for r in rs]) for a, b in pairs]
    print(f"  {field}: rank corr {ci(sp)}  top5% overlap {ci(ov)}")
print("\n=== by task: dest top5% overlap: additive s0-s1 | xattn s0-s1 | cross; rank corr cross")
for t in TASKS:
    sel = [n for n, i in enumerate(ids) if task_of(qt[i]) == t]
    g = lambda lab, j: np.mean([SIM[("dest", lab)][j][n] for n in sel])
    print(f"  {t:26s} {g('additive s0-s1', 1):.3f} {g('xattn s0-s1', 1):.3f} {g('additive-xattn', 1):.3f}  corr {g('additive-xattn', 0):.3f}")
print("\n=== concentration and pathway share per arm")
def conc(a):
    p = a / a.sum(); return np.exp(-(p * np.log(p + 1e-20)).sum()) / len(a), np.sort(p)[-len(a) // 20:].sum()
for r in RUNS[:4]:
    c = np.array([conc(M[r][i]["dest"]) for i in ids]); v = np.array([conc(M[r][i]["vis"]) for i in ids])
    geo = np.array([M[r][i]["dest"].sum() for i in ids]); vis = np.array([M[r][i]["vis"].sum() for i in ids])
    ks = np.array([[M[r][i][f"dest_{k}"].sum() for k in KEYS] for i in ids]); ks = ks / ks.sum(1, keepdims=True)
    print(f"  {r:12s} effective cells {c[:, 0].mean():.3f} of N, top 5% hold {c[:, 1].mean():.3f}; vis {v[:, 0].mean():.3f}, {v[:, 1].mean():.3f} | "
          f"|attr| geometry/(geometry+vision) median {np.median(geo / (geo + vis)):.3f} | share by layer {np.round(ks.mean(0), 3)} | Delta norm {np.mean([[rec[r][i][f'delta_norm_{k}'] for k in KEYS] for i in ids], 0).round(2)}")
print("\n=== faithfulness: |change in m| when zeroing Delta at cells (median, mean); top 5% by dest vs random 5% vs all cells")
for r in RUNS[:4]:
    d = {k: np.array([abs(rec[r][i]["del"][k] - rec[r][i]["m"]) for i in ids]) for k in ("top5", "rand5", "all")}
    print(f"  {r:12s} " + "  ".join(f"{k} {np.median(v):.2f}/{v.mean():.2f}" for k, v in d.items()) + f"  P(top5 > rand5) {(d['top5'] > d['rand5']).mean():.3f}")
    s = np.array([rec[r][i]["del"]["all"] - rec[r][i]["m"] for i in ids])
    print(f"     m with geometry {np.mean([rec[r][i]['m'] for i in ids]):+.2f}; removing all geometry changes m by {s.mean():+.2f}")
print("\n=== xattn flow: where the decision-relevant injection comes from, |signed attr| share, attention mass, Delta-norm share per class")
S = ("self", "same_frame", "other_near", "other_far")
for k in KEYS:
    rows = [rec[r][i]["flow"][k] for r in RUNS[2:4] for i in ids]
    A = np.array([[abs(f[f"attr_{s}"]) for s in S] for f in rows]); A = A / A.sum(1, keepdims=True)
    print(f"  layer {k}: " + "  ".join(f"{s}: attr {A[:, n].mean():.3f} mass {np.mean([f[f'mass_{s}'] for f in rows]):.3f} norm/total {np.mean([f[f'norm_{s}'] / f['norm_total'] for f in rows]):.3f}" for n, s in enumerate(S)))
for t in TASKS:
    rows = [f for r in RUNS[2:4] for i in ids if task_of(qt[i]) == t for f in rec[r][i]["flow"].values()]
    A = np.array([[abs(f[f"attr_{s}"]) for s in S] for f in rows]); A = A / A.sum(1, keepdims=True)
    print(f"  {t:26s} attr share " + " ".join(f"{A[:, n].mean():.3f}" for n in range(4)))
print("\n=== linear predictability of one model's Delta from another's, sampled cells, held-out videos (R^2)")
videos = sorted(byv); test_v = set(videos[::4])
def load_delta(r, k):
    tr, te = [], []
    for i in ids:
        x = np.load(f"{SAL}/{r}/{i}.npz")[f"delta_{k}"].astype(np.float32)
        (te if rec[r][i]["video"] in test_v else tr).append(x)
    return np.concatenate(tr), np.concatenate(te)
def r2(xr, xe, yr, ye, lam=1e-1):
    mx, sx = xr.mean(0), xr.std(0) + 1e-6; my = yr.mean(0)
    X, Xe = (xr - mx) / sx, (xe - mx) / sx
    w = np.linalg.solve(X.T @ X + lam * len(X) * np.eye(X.shape[1]), X.T @ (yr - my))
    return 1 - ((ye - my - Xe @ w) ** 2).sum() / ((ye - ye.mean(0)) ** 2).sum()
for k in KEYS:
    D = {r: load_delta(r, k) for r in RUNS[:4]}
    out = []
    for a, b in [("additive_s0", "additive_s1"), ("xa3_both_s0", "xa3_both_s1"), ("additive_s0", "xa3_both_s0"), ("additive_s1", "xa3_both_s1"), ("xa3_both_s0", "additive_s0"), ("xa3_both_s1", "additive_s1")]:
        out.append(f"{a[:3]}{a[-1]}->{b[:3]}{b[-1]} {r2(*D[a], *D[b]):.3f}")
    print(f"  layer {k} (train n {len(D[RUNS[0]][0])}): " + "  ".join(out))
print("\n=== ablations on the 1,994-question sample: official overall (mean of tasks)")
sample = [int(l) for l in open("results/explain_v7/ids_200.txt")]
def load(path):
    return pd.DataFrame([json.loads(l) for l in open(path)]).set_index("id").loc[sample].sort_index()
B = {r: pd.DataFrame(list(base[r].values())).set_index("id").loc[sample].sort_index() for r in RUNS}
done = [p for p in sorted(glob.glob("results/explain_v7/abl/*.jsonl")) if sum(1 for _ in open(p)) >= len(sample)]
print(f"{len(done)} finished ablation runs")
A = {(os.path.basename(p)[:-6].rsplit("_", 1)[0], os.path.basename(p)[:-6].rsplit("_", 1)[1]): load(p) for p in done}
d0 = B[RUNS[0]]
qs = d0["question_type"].values; T = sorted(set(qs)); idx = [np.where(qs == x)[0] for x in T]
to = {x: task_of(x) for x in T}; W = np.array([1 / sum(to[y] == to[x] for y in T) / len(TASKS) for x in T])
rb = np.random.default_rng(0); TB = [rb.integers(0, len(i), (4000, len(i))) for i in idx]
ov = lambda s: float(np.sum(W * [s[i].mean() for i in idx])) * 100
ovb = lambda s: np.sum([w * s[i][tb].mean(1) for w, i, tb in zip(W, idx, TB)], 0) * 100
sc = lambda d: d["score"].values.astype(float)
print("baseline (unedited, same questions): " + ", ".join(f"{r} {ov(sc(B[r])):.1f}" for r in RUNS))
assert all(abs(ov(sc(B[r])) - summarize(B[r])["overall"]) < 1e-6 for r in RUNS)
print(" per run (change from that run's baseline):")
for (r, mode), d in sorted(A.items()):
    dd = ovb(sc(d)) - ovb(sc(B[r]))
    print(f"   {r:12s} {mode:11s} {ov(sc(d)):5.1f}  {ov(sc(d)) - ov(sc(B[r])):+5.1f} [{np.percentile(dd, 2.5):+.1f}, {np.percentile(dd, 97.5):+.1f}]")
print(" arm       mode         s0    s1   seed-mean change from baseline [95%, question bootstrap]")
for arm, rs in (("additive", RUNS[:2]), ("xattn", RUNS[2:4])):
    for mode in ("zero", "shuffle", "self", "nonself", "frame", "otherframe"):
        if (rs[0], mode) not in A or (rs[1], mode) not in A:
            continue
        v = [ov(sc(A[(r, mode)])) for r in rs]
        diff = np.mean([ovb(sc(A[(r, mode)])) - ovb(sc(B[r])) for r in rs], 0)
        print(f" {arm:9s} {mode:11s} {v[0]:5.1f} {v[1]:5.1f}   {np.mean([ov(sc(A[(r, mode)])) - ov(sc(B[r])) for r in rs]):+5.1f} [{np.percentile(diff, 2.5):+.1f}, {np.percentile(diff, 97.5):+.1f}]")
lb = np.mean([ovb(sc(B[r])) for r in RUNS[4:]], 0)
print(f" LoRA-only seed mean on this sample: {np.mean([ov(sc(B[r])) for r in RUNS[4:]]):.1f}")
print("\nself vs baseline and additive baseline, xattn seed mean:")
sf = np.mean([ovb(sc(A[(r, 'self')])) for r in RUNS[2:4]], 0) if (RUNS[2], "self") in A and (RUNS[3], "self") in A else None
if sf is not None:
    ad = np.mean([ovb(sc(B[r])) for r in RUNS[:2]], 0); xb = np.mean([ovb(sc(B[r])) for r in RUNS[2:4]], 0)
    print(f"  xattn(self only) - additive: {np.mean(sf - ad):+.1f} [{np.percentile(sf - ad, 2.5):+.1f}, {np.percentile(sf - ad, 97.5):+.1f}]")
    print(f"  xattn(full) - xattn(self only): {np.mean(xb - sf):+.1f} [{np.percentile(xb - sf, 2.5):+.1f}, {np.percentile(xb - sf, 97.5):+.1f}]")
    print(f"  xattn(full) - additive: {np.mean(xb - ad):+.1f} [{np.percentile(xb - ad, 2.5):+.1f}, {np.percentile(xb - ad, 97.5):+.1f}]")
print("\nper task change from baseline, seed mean\n" + " " * 21 + " ".join(f"{t.split('_', 1)[1][:9]:>9s}" for t in TASKS))
tbase = {r: summarize(B[r]) for r in RUNS}
for arm, rs in (("additive", RUNS[:2]), ("xattn", RUNS[2:4])):
    print(f"{arm:9s} baseline   " + " ".join(f"{np.mean([tbase[r][t] for r in rs]):9.1f}" for t in TASKS))
    for mode in ("zero", "shuffle", "self", "nonself", "frame", "otherframe"):
        if (rs[0], mode) in A and (rs[1], mode) in A:
            print(f"{arm:9s} {mode:10s} " + " ".join(f"{np.mean([summarize(A[(r, mode)])[t] - tbase[r][t] for r in rs]):+9.1f}" for t in TASKS))
print(f"LoRA      baseline   " + " ".join(f"{np.mean([tbase[r][t] for r in RUNS[4:]]):9.1f}" for t in TASKS))
