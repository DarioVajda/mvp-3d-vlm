"""Log-based diagnostics of the v3 runs: correlations with performance, and anomalies / spikes during training."""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, "/shared/workspace/povejmo/math/projects/mvp/src")
R = "/shared/workspace/povejmo/math/projects/mvp/results/train"
RUNS = [f"{s}_v3_{a}" for s in ("2B", "4B") for a in ("lora", "additive", "xattn")]
VSI = {"2B_v3_lora": 52.7, "2B_v3_additive": 55.0, "2B_v3_xattn": 49.0, "4B_v3_lora": 58.7, "4B_v3_additive": 59.0, "4B_v3_xattn": 51.0}
logs = {r: [json.loads(l) for l in open(f"{R}/{r}/log.jsonl")] for r in RUNS}
steps = {r: [x for x in logs[r] if "loss" in x] for r in RUNS}
devs = {r: [x for x in logs[r] if "dev_acc" in x and x["step"] > 0] for r in RUNS}
SPE = 931


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")


print("=" * 80)
print("A. CORRELATIONS")
print("=" * 80)
# A1: across the six runs, run-level summaries vs final full-VSI
rows = []
for r in RUNS:
    st = steps[r]
    g = np.array([x["grad_norm"] for x in st])
    inj = np.mean([np.mean(list(x["inject"].values())) for x in st[-100:]]) if st[-1].get("inject") else 0.0
    rows.append({"run": r, "vsi": VSI[r], "dev_acc6": devs[r][-1]["dev_acc"], "dev_loss6": devs[r][-1]["dev_loss"],
                 "train_loss6": np.mean([x["loss"] for x in st if x["epoch"] == 5]), "spikes": int((g > 50).sum()),
                 "gnorm_median": float(np.median(g)), "inject_end": float(inj),
                 "min_dev_loss": min(x["dev_loss"] for x in logs[r] if "dev_loss" in x)})
print("\nA1. run-level summary")
print(f"{'run':16}{'VSI':>6}{'dev6':>7}{'devL6':>7}{'trL6':>7}{'spikes':>8}{'gmed':>7}{'inj':>7}{'minDevL':>9}")
for x in rows:
    print(f"{x['run']:16}{x['vsi']:6.1f}{x['dev_acc6']:7.1f}{x['dev_loss6']:7.3f}{x['train_loss6']:7.3f}{x['spikes']:8d}{x['gnorm_median']:7.2f}{x['inject_end']:7.3f}{x['min_dev_loss']:9.3f}")
for k in ("dev_acc6", "dev_loss6", "train_loss6", "spikes", "gnorm_median", "min_dev_loss"):
    print(f"  corr({k}, VSI) over 6 runs: {pearson([x[k] for x in rows], [x['vsi'] for x in rows]):+.2f}")
print("  (n=6; treat as descriptive)")

# A2: within each run, epoch-level: dev acc vs train loss, dev loss, inject (per point)
print("\nA2. epoch-level within-run correlations (6 epochs each)")
for r in RUNS:
    ep_loss = [np.mean([x["loss"] for x in steps[r] if x["epoch"] == e]) for e in range(6)]
    da = [d["dev_acc"] for d in devs[r]]
    dl = [d["dev_loss"] for d in devs[r]]
    line = f"{r:16} corr(dev_acc, train_loss) {pearson(da, ep_loss):+.2f}  corr(dev_acc, dev_loss) {pearson(da, dl):+.2f}"
    if steps[r][-1].get("inject"):
        for k in ("0", "1", "2"):
            inj = [np.mean([x["inject"][k] for x in steps[r] if x["epoch"] == e][-50:]) for e in range(6)]
            line += f"  corr(dev_acc, inject{k}) {pearson(da, inj):+.2f}"
    print(line)

# A3: pooled over the 4 geometry runs: per-epoch dev-acc *gain over the size-matched LoRA run* vs injection ratio
print("\nA3. geometry runs: dev-acc advantage over LoRA (same size, same epoch) vs injection ratio")
pts = []
for s in ("2B", "4B"):
    base = [d["dev_acc"] for d in devs[f"{s}_v3_lora"]]
    for a in ("additive", "xattn"):
        r = f"{s}_v3_{a}"
        for e in range(6):
            inj = np.mean([np.mean(list(x["inject"].values())) for x in steps[r] if x["epoch"] == e][-50:])
            adv = devs[r][e]["dev_acc"] - base[e]
            pts.append((r, e + 1, inj, adv))
            print(f"  {r:16} ep{e + 1} inject {inj:.3f}  dev advantage {adv:+5.1f}")
print(f"  pooled corr(inject, advantage): {pearson([p[2] for p in pts], [p[3] for p in pts]):+.2f}  (n={len(pts)})")
for a in ("additive", "xattn"):
    sel = [p for p in pts if a in p[0]]
    print(f"  {a:9} corr: {pearson([p[2] for p in sel], [p[3] for p in sel]):+.2f}")

# A4: per-task dev accuracy at epoch 6 vs injection ratio at each point (geometry runs)
print("\nA4. per-task dev accuracy (epoch 6), geometry arm minus LoRA, by task")
tasks = sorted(k[8:] for k in devs["2B_v3_lora"][-1] if k.startswith("dev_acc/"))
print(f"{'run':16}" + "".join(f"{t[:11]:>12}" for t in tasks))
for s in ("2B", "4B"):
    base = devs[f"{s}_v3_lora"][-1]
    for a in ("additive", "xattn"):
        d = devs[f"{s}_v3_{a}"][-1]
        print(f"{s + '_' + a:16}" + "".join(f"{d['dev_acc/' + t] - base['dev_acc/' + t]:+12.1f}" for t in tasks))

print("\n" + "=" * 80)
print("B. ANOMALIES AND SPIKES")
print("=" * 80)
# B1: grad-norm spikes: when, and shared across arms (same seed => same micro-batch order within a size)
spike_steps = {}
for r in RUNS:
    st = steps[r]
    spike_steps[r] = {x["step"] for x in st if x["grad_norm"] > 50}
    big = sorted(((x["grad_norm"], x["step"], x["loss"]) for x in st), reverse=True)[:5]
    by_epoch = Counter((x["step"] - 1) // SPE + 1 for x in st if x["grad_norm"] > 50)
    warm = sum(1 for x in st if x["grad_norm"] > 50 and x["step"] <= 559)
    print(f"\n{r}: {len(spike_steps[r])} spikes >50; by epoch {dict(sorted(by_epoch.items()))}; in warmup (<=559): {warm}")
    print("   top-5 grad norms (norm, step, loss):", [(round(g, 0), s, round(l, 2)) for g, s, l in big])
for s in ("2B", "4B"):
    L, A, X = (spike_steps[f"{s}_v3_{a}"] for a in ("lora", "additive", "xattn"))
    print(f"\n{s}: spike steps shared lora&additive {len(L & A)}, lora&xattn {len(L & X)}, additive&xattn {len(A & X)}, all three {len(L & A & X)}; xattn-only {len(X - L - A)}")
    common = sorted(L & A)
    print(f"   steps shared by lora&additive: {common[:20]}")
# B1b: xattn-only spike steps: do they recur at the same position within an epoch (same data)?
for s in ("2B", "4B"):
    X = spike_steps[f"{s}_v3_xattn"] - spike_steps[f"{s}_v3_lora"] - spike_steps[f"{s}_v3_additive"]
    pos = Counter((st - 1) % SPE for st in X)
    rep = {p: c for p, c in pos.items() if c > 1}
    print(f"{s} xattn-only spikes: {len(X)}; positions-in-epoch recurring across epochs: {rep}")

# B2: loss spikes (loss > 3x the 50-step rolling median) and their effect: next 50-step loss and next dev loss
print("\nB2. loss spikes (> 3x rolling 50-step median)")
for r in RUNS:
    st = steps[r]
    tl = np.array([x["loss"] for x in st])
    out = []
    for i in range(50, len(st)):
        med = np.median(tl[i - 50:i])
        if tl[i] > 3 * med and tl[i] > 1.0:
            after = tl[i + 1:i + 51].mean() if i + 1 < len(tl) else float("nan")
            out.append((st[i]["step"], round(float(tl[i]), 2), round(float(med), 2), round(float(after), 2), round(st[i]["grad_norm"], 0), st[i]["questions"]))
    print(f"{r}: {len(out)} loss spikes; (step, loss, median_before, mean_loss_next50, gnorm, n_q): {out[:12]}")

# B3: injection jumps: largest single-step change of the mean injection ratio
print("\nB3. injection-ratio jumps (largest step-to-step change, mean of 3 points)")
for r in RUNS:
    st = steps[r]
    if not st[-1].get("inject"):
        continue
    inj = np.array([np.mean(list(x["inject"].values())) for x in st])
    d = np.diff(inj)
    idx = np.argsort(-np.abs(d))[:5]
    print(f"{r}: ", [(int(st[i + 1]["step"]), round(float(inj[i]), 3), round(float(inj[i + 1]), 3), round(st[i + 1]["grad_norm"], 0)) for i in idx])
    # per point end-of-training and max
    for k in ("0", "1", "2"):
        v = np.array([x["inject"][k] for x in st])
        print(f"   point {k}: max {v.max():.3f} at step {st[int(v.argmax())]['step']}, end {v[-100:].mean():.3f}")

# B4: dev-loss trajectory anomalies: first step where dev loss rises above its minimum by >0.1, and dev-acc drops between consecutive evals
print("\nB4. dev loss turning points and dev-acc regressions")
for r in RUNS:
    dl = [(x["step"], x["dev_loss"]) for x in logs[r] if "dev_loss" in x]
    m = min(v for _, v in dl)
    first_up = next((s for s, v in dl if v > m + 0.1 and s > [s2 for s2, v2 in dl if v2 == m][0]), None)
    drops = [(devs[r][i]["step"] // SPE, round(devs[r][i]["dev_acc"] - devs[r][i - 1]["dev_acc"], 1)) for i in range(1, len(devs[r])) if devs[r][i]["dev_acc"] < devs[r][i - 1]["dev_acc"]]
    print(f"{r}: min dev loss {m:.3f} at step {[s for s, v in dl if v == m][0]}; first +0.1 above min at step {first_up} (epoch {first_up / SPE if first_up else float('nan'):.1f}); dev-acc regressions {drops}")

# B5: step-time / memory anomalies
print("\nB5. step time and memory")
for r in RUNS:
    t = np.array([x["step_time"] for x in steps[r]]); mem = np.array([x["peak_mem_gb"] for x in steps[r]])
    print(f"{r}: step time median {np.median(t):.2f}s p99 {np.percentile(t, 99):.2f}s max {t.max():.1f}s; peak mem median {np.median(mem):.1f} max {mem.max():.1f} GB")

# B6: which micro-batches sit at the shared spike steps (same seed => reconstruct the epoch plan)
print("\nB6. data behind spikes shared by all arms (reconstructed epoch plan)")
try:
    from mvp.train.data import epoch_plan, load_vlm3r, rank_steps
    samples, _ = load_vlm3r(6000, 0, 60, 100)
    for s in ("2B",):
        shared = sorted(spike_steps[f"{s}_v3_lora"] & spike_steps[f"{s}_v3_additive"] & spike_steps[f"{s}_v3_xattn"])
        for stp in shared[:8]:
            e, i = (stp - 1) // SPE, (stp - 1) % SPE
            plan = rank_steps(epoch_plan(samples, e, 0, 8), 4, 1, 0)
            mbs, n_q = plan[i]
            vids = [(mb[0]["video"], len(mb), Counter(x["task"] for x in mb).most_common(2)) for mb in mbs]
            print(f"   step {stp} (epoch {e + 1}, pos {i}): {n_q} questions; micro-batches {vids}")
except Exception as ex:
    print("   could not reconstruct:", type(ex).__name__, ex)
