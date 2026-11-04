"""Is there an answer-letter bias on VSI-Bench multiple choice? All letters treated alike. Read-only, per-question files; bootstrap over questions."""
import glob, json, re
from collections import Counter, defaultdict
import numpy as np

ROOT = "/shared/workspace/povejmo/math/projects/mvp"
R = f"{ROOT}/results/train"
RUNS = {"v9 LoRA s1": "2B_v9_fmix_lora_s1", "v9 add s0": "2B_v9_fmix_additive_s0", "v9 add s1": "2B_v9_fmix_additive_s1", "v9 xa s0": "2B_v9_fmix_xa3_both_s0", "v9 xa s1": "2B_v9_fmix_xa3_both_s1",
        "v7 LoRA s0": "2B_v7_lora_s0", "v7 LoRA s1": "2B_v7_lora_s1", "v7 add s0": "2B_v7_additive_s0", "v7 add s1": "2B_v7_additive_s1", "v7 xa s0": "2B_v7_xa3_both_s0", "v7 xa s1": "2B_v7_xa3_both_s1"}
L = "ABCD"
rng = np.random.default_rng(0)


def letter(s):
    s = str(s).strip()
    return s[0].upper() if s and s[0].upper() in L else "?"


def load(run):
    rows = []
    for x in map(json.loads, open(f"{R}/{run}/branches/step5750/eval/vsi_f32_full.jsonl")):
        if x.get("options") is not None and len(x["options"]):
            rows.append({"id": x["id"], "task": x["task"], "qt": x["question_type"], "gt": letter(x["ground_truth"]), "p": letter(x["prediction"]), "n": len(x["options"]), "ok": x["score"] == 1})
    return rows


def ss():
    rows = []
    for r in map(json.loads, open(glob.glob(f"{ROOT}/results/baselines/spatialstack/full_qwen35/vsibench/*/*samples_vsibench.jsonl")[0])):
        s = r["vsibench_score"]
        if s.get("options") is None or not len(s["options"]):
            continue
        rows.append({"id": s["id"], "task": re.sub(r"_(easy|medium|hard)$", "", s["question_type"]), "qt": s["question_type"], "gt": letter(s["ground_truth"]), "p": letter(s["prediction"]), "n": len(s["options"]), "ok": letter(s["prediction"]) == letter(s["ground_truth"])})
    return rows


D = {k: load(v) for k, v in RUNS.items()}
D["SpatialStack-4B"] = ss()
first = D["v9 add s0"]
n = len(first)
print("MC questions", n, "| options per question", Counter(r["n"] for r in first), "| by task", Counter(r["task"] for r in first))
gt = Counter(r["gt"] for r in first)
print("ground truth letters:", {k: f"{gt[k]} ({100 * gt[k] / n:.1f}%)" for k in L})
print("available in the options:", {k: sum(L.index(k) < r["n"] for r in first) for k in L})
print("GT share given the letter is available:", {k: round(gt[k] / sum(L.index(k) < r["n"] for r in first), 3) for k in L})
print("GT by option count:", {c: dict(sorted(Counter(r["gt"] for r in first if r["n"] == c).items())) for c in sorted({r["n"] for r in first})})

print("\n=== 1. Predicted vs true share per letter (pred − truth, percentage points, 95% bootstrap over questions) ===")
out = {"n": n, "gt": dict(gt), "runs": {}}
for name, rows in D.items():
    gtv = np.array([r["gt"] for r in rows]); pv = np.array([r["p"] for r in rows]); ok = np.array([r["ok"] for r in rows])
    line = {}
    for k in L:
        d = (pv == k).astype(float) - (gtv == k).astype(float)
        bs = [d[rng.integers(0, len(d), len(d))].mean() * 100 for _ in range(2000)]
        line[k] = (round(d.mean() * 100, 2), round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2))
    rec = {k: (round(float(ok[gtv == k].mean()) * 100, 1)) for k in L}  # recall: accuracy when this letter is correct
    prec = {k: (round(float((gtv[pv == k] == k).mean()) * 100, 1) if (pv == k).any() else None) for k in L}
    # among wrong answers, share of each letter vs what uniform-over-wrong-available-options predicts
    wrong = [r for r in rows if not r["ok"] and r["p"] in L]
    exp, obs = Counter(), Counter()
    for r in wrong:
        opts = [c for c in L[:r["n"]] if c != r["gt"]]
        for c in opts:
            exp[c] += 1 / len(opts)
        obs[r["p"]] += 1
    wr = {k: (obs[k], round(exp[k], 1)) for k in L}
    # bootstrap of obs-exp in the wrong answers
    z = {}
    for k in L:
        d = np.array([(r["p"] == k) - (1 / (r["n"] - 1) if k in L[:r["n"]] and k != r["gt"] else 0) for r in wrong], float)
        bs = [d[rng.integers(0, len(d), len(d))].mean() * 100 for _ in range(2000)]
        z[k] = (round(d.mean() * 100, 2), round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2))
    out["runs"][name] = {"share_diff": line, "recall": rec, "precision": prec, "wrong_obs_exp": wr, "wrong_excess": z, "pred": dict(Counter(pv.tolist())), "acc": round(float(ok.mean()) * 100, 1), "n_wrong": len(wrong), "invalid": int((pv == "?").sum())}
    print(f"{name:16} acc {ok.mean() * 100:5.1f} | pred−truth " + " ".join(f"{k} {line[k][0]:+.1f}[{line[k][1]:+.1f},{line[k][2]:+.1f}]" for k in L) + f" | invalid {int((pv == '?').sum())}")
print("\n=== 2. Accuracy when each letter is the correct answer (recall), and precision ===")
for name, v in out["runs"].items():
    print(f"{name:16} recall {v['recall']}  precision {v['precision']}")
print("\n=== 3. Among WRONG answers: letter chosen vs expectation if wrong answers were uniform over the available wrong options (excess, % of wrong answers, 95% CI) ===")
for name, v in out["runs"].items():
    print(f"{name:16} n {v['n_wrong']:4} " + " ".join(f"{k} {v['wrong_obs_exp'][k][0]}/{v['wrong_obs_exp'][k][1]} {v['wrong_excess'][k][0]:+.1f}[{v['wrong_excess'][k][1]:+.1f},{v['wrong_excess'][k][2]:+.1f}]" for k in L))
print("\n=== 4. By question type: pred share vs truth share, per letter (mean over v9 additive and xattn seeds) ===")
g = ["v9 add s0", "v9 add s1", "v9 xa s0", "v9 xa s1"]
types = sorted({r["qt"] for r in first})
out["by_type"] = {}
for t in types:
    rows0 = [r for r in first if r["qt"] == t]
    tr = {k: 100 * np.mean([r["gt"] == k for r in rows0]) for k in L}
    pr = {k: np.mean([100 * np.mean([r["p"] == k for r in D[x] if r["qt"] == t]) for x in g]) for k in L}
    lo = {k: 100 * np.mean([r["p"] == k for r in D["v9 LoRA s1"] if r["qt"] == t]) for k in L}
    out["by_type"][t] = {"n": len(rows0), "opts": dict(Counter(r["n"] for r in rows0)), "truth": {k: round(tr[k], 1) for k in L}, "geo": {k: round(pr[k], 1) for k in L}, "lora": {k: round(lo[k], 1) for k in L}}
    print(f"{t:30} n{len(rows0):4} opts {dict(Counter(r['n'] for r in rows0))} | " + " ".join(f"{k}: true {tr[k]:4.1f} geo {pr[k]:4.1f} lora {lo[k]:4.1f} |" for k in L))
print("\n=== 5. Same questions, 4-option only, so all letters are available ===")
out["four"] = {}
for name in ("v9 LoRA s1", "v9 add s0", "v9 add s1", "v9 xa s0", "v9 xa s1", "v7 add s0", "SpatialStack-4B"):
    rows = [r for r in D[name] if r["n"] == 4]
    gtv = np.array([r["gt"] for r in rows]); pv = np.array([r["p"] for r in rows])
    out["four"][name] = {"n": len(rows), "truth": {k: round(100 * float((gtv == k).mean()), 1) for k in L}, "pred": {k: round(100 * float((pv == k).mean()), 1) for k in L}, "recall": {k: round(100 * float((pv[gtv == k] == k).mean()), 1) for k in L}}
    print(f"{name:16} n {len(rows)} truth {out['four'][name]['truth']} pred {out['four'][name]['pred']} recall {out['four'][name]['recall']}")
print("\n=== 6. Lens at the final token vs the answer: which letter tops the readout at L17/L20/L22 (MC questions), J-lens text, additive & LoRA ===")
out["lens"] = []
for m in ("lora_s0", "additive_s0"):
    rs = [json.loads(l) for l in open(f"{ROOT}/results/jspace_v7/{m}/readouts.jsonl")]
    for l_ in (17, 20, 22):
        seen = {}
        for r in rs:
            if r["lens"] == "text" and r["layer"] == l_ and r["kind"] == "text":
                seen[r["id"]] = r  # the last text position
        rows = []
        for r in seen.values():
            if letter(r["gt"]) in L and not re.search(r"\d", r["gt"]):
                top = [re.sub(r"[^A-D]", "", n) for n, _ in r["raw"]]
                first_letter = next((t for t in top if len(t) == 1), None)
                rows.append((r["type"][:18], r["gt"], r["answer"], first_letter))
        out["lens"].append({"model": m, "layer": l_, "rows": rows})
        print(m, "L", l_, rows)
json.dump(out, open(f"{ROOT}/.claude/worktrees/je-xattn-proposal/research_notes/letter_bias.json", "w"))
