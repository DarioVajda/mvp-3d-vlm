"""Error structure of route planning and absolute distance on v9 (and SpatialStack), from the per-question files. No GPU."""
import glob, json, re, sys
from collections import Counter, defaultdict
import numpy as np

ROOT = "/shared/workspace/povejmo/math/projects/mvp"
R = f"{ROOT}/results/train"
RUNS = {"lora": ["2B_v9_fmix_lora_s1"], "add": ["2B_v9_fmix_additive_s0", "2B_v9_fmix_additive_s1"], "xa": ["2B_v9_fmix_xa3_both_s0", "2B_v9_fmix_xa3_both_s1"]}
load = lambda r: {x["id"]: x for x in map(json.loads, open(f"{R}/{r}/branches/step5750/eval/vsi_f32_full.jsonl"))}
D = {g: [load(r) for r in rs] for g, rs in RUNS.items()}
ss_file = glob.glob(f"{ROOT}/results/baselines/spatialstack/full_qwen35/vsibench/*/*samples_vsibench.jsonl")[0]
SS = {}
for r in map(json.loads, open(ss_file)):
    sc = r["vsibench_score"]; p = r["filtered_resps"]
    while isinstance(p, list): p = p[0]
    SS[sc["id"]] = {"prediction": p, "ground_truth": sc["ground_truth"], "options": sc.get("options"), "question": sc["question"], "task": re.sub(r"_(easy|medium|hard)$", "", sc["question_type"])}
ex = next(iter(D["add"][0].values()))
print("example keys:", sorted(ex.keys()))

# ---------------- route planning ----------------
print("\n================ ROUTE PLANNING ================")
rp = [i for i, x in D["add"][0].items() if x["task"] == "route_planning"]
q = D["add"][0][rp[0]]
print("example question:\n", q["question"][:700], "\noptions:", q["options"], "gt:", q["ground_truth"], "pred:", q["prediction"])
TURN = {"turn left": "L", "turn right": "R", "turn back": "B", "turn around": "B"}


def turns(opt):
    return [TURN[t.strip().lower()] for t in re.split(r",", opt) if t.strip().lower() in TURN]


def parse(x):
    opts = x["options"]
    letter = lambda s: str(s).strip()[0].upper()
    gt = letter(x["ground_truth"]); pr = letter(x["prediction"]) if str(x["prediction"]).strip() else None
    seqs = {letter(o): turns(o) for o in opts}
    return gt, pr, seqs


def analyse(name, recs):
    rows = []
    for i in rp:
        if i not in recs: continue
        x = recs[i]
        gt, pr, seqs = parse(x)
        if pr not in seqs: rows.append({"valid": False}); continue
        g, p = seqs[gt], seqs[pr]
        n = len(g)
        mirror = all((a == "B") == (b == "B") and (a == b or {a, b} == {"L", "R"}) for a, b in zip(g, p)) and p != g and all(a != b or a == "B" for a, b in zip(g, p))
        rows.append({"valid": True, "correct": pr == gt, "n": n, "per_turn": [a == b for a, b in zip(g, p)], "mirror": mirror, "pred_letter": pr, "gt_letter": gt,
                     "nstepsq": x["question"].count("Go forward until")})
    v = [r for r in rows if r["valid"]]
    acc = np.mean([r["correct"] for r in v])
    byn = defaultdict(list)
    for r in v: byn[r["n"]].append(r["correct"])
    pos = defaultdict(list)
    for r in v:
        for k, ok in enumerate(r["per_turn"]): pos[k].append(ok)
    first_vs_rest = (np.mean(pos[0]), np.mean([o for k in pos if k > 0 for o in pos[k]]))
    mirror_rate = np.mean([r["mirror"] for r in v if not r["correct"]])
    letters = Counter(r["pred_letter"] for r in v); gts = Counter(r["gt_letter"] for r in v)
    print(f"{name:6} n {len(v)} acc {acc:.3f} | by #turns {{{', '.join(f'{k}: {np.mean(vv):.2f} (n{len(vv)})' for k, vv in sorted(byn.items()))}}} | per-turn acc first {first_vs_rest[0]:.2f} later {first_vs_rest[1]:.2f} "
          f"| wrong answers that are the exact L/R mirror {mirror_rate:.2f} | pred letters {dict(letters)} gt {dict(gts)}")
    return {r["gt_letter"]: 0 for r in v}, v


res = {}
for g in RUNS:
    for s, recs in enumerate(D[g]):
        _, res[f"{g}{s}"] = analyse(f"{g} s{s}", recs)
_, res["ss"] = analyse("SS", {i: SS[i] for i in rp if i in SS})
# agreement: do all models fail the same questions?
ok = {k: np.array([r["correct"] for r in v]) for k, v in res.items() if len(v) == len(res["add0"])}
allk = list(ok)
print("questions all models get right:", int(np.all([ok[k] for k in allk], 0).sum()), "all wrong:", int(np.all([~ok[k] for k in allk], 0).sum()), "of", len(ok["add0"]))
# chance-level check: options per question
print("options per question:", Counter(len(D['add'][0][i]['options']) for i in rp))

# ---------------- absolute distance ----------------
print("\n================ ABSOLUTE DISTANCE ================")
ad = [i for i, x in D["add"][0].items() if x["task"] == "object_abs_distance"]
room = {}
for i, x in D["add"][0].items():
    if x["task"] == "room_size_estimation": room[x["scene_name"]] = float(x["ground_truth"])
print("scenes with a room-size label:", len(room), "of", len({D['add'][0][i]['scene_name'] for i in ad}))


def num(p):
    m = re.search(r"-?\d+(\.\d+)?", str(p)); return float(m.group()) if m else None


def ad_analyse(name, recs):
    rows = []
    for i in ad:
        if i not in recs: continue
        x = recs[i]; p = num(x["prediction"]); t = float(x["ground_truth"])
        if p is None or t <= 0: continue
        rows.append({"t": t, "p": p, "ratio": p / t, "mra": x.get("score", np.nan), "scene": x.get("scene_name"), "ds": x.get("dataset")})
    t = np.array([r["t"] for r in rows]); ratio = np.array([r["ratio"] for r in rows]); mra = np.array([r["mra"] for r in rows], float)
    bins = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 99)]
    by = {f"{a}-{b}m": (float(np.median(ratio[(t >= a) & (t < b)])), float(np.nanmean(mra[(t >= a) & (t < b)])), int(((t >= a) & (t < b)).sum())) for a, b in bins}
    logerr = np.log(ratio)
    print(f"{name:6} n {len(rows)} MRA {np.nanmean(mra):.3f} median ratio {np.median(ratio):.2f} | spread of log-ratio (IQR) {np.percentile(logerr, 75) - np.percentile(logerr, 25):.2f} "
          f"| corr(log pred, log true) {np.corrcoef(np.log([r['p'] for r in rows]), np.log(t))[0, 1]:.2f}")
    print("        by true distance: " + ", ".join(f"{k}: ratio {v[0]:.2f} MRA {v[1]:.2f} n{v[2]}" for k, v in by.items()))
    # scene scale: does the error correlate with room size?
    rs = np.array([room.get(r["scene"], np.nan) for r in rows]); m = np.isfinite(rs)
    print(f"        corr(log ratio, log room size) {np.corrcoef(logerr[m], np.log(rs[m]))[0, 1]:+.2f}; predictions rounded to 0.5m: {np.mean([abs(r['p'] * 2 - round(r['p'] * 2)) < 1e-6 for r in rows]):.2f}, integers: {np.mean([float(r['p']).is_integer() for r in rows]):.2f}")
    return rows


rows = {}
for g in RUNS:
    for s, recs in enumerate(D[g]): rows[f"{g}{s}"] = ad_analyse(f"{g} s{s}", recs)
rows["ss"] = ad_analyse("SS", {i: SS[i] for i in ad if i in SS})
# where does geometry help: paired by question, additive mean vs lora
L = {i: D["lora"][0][i] for i in ad}; A = [D["add"][0], D["add"][1]]
gain = defaultdict(list)
for i in ad:
    t = float(L[i]["ground_truth"]); b = next(k for k, (a, c) in enumerate([(0, 1), (1, 2), (2, 3), (3, 5), (5, 99)]) if a <= t < c)
    gain[b].append(np.mean([a[i]["score"] for a in A]) - L[i]["score"])
print("additive - lora MRA by true-distance bin:", {["0-1", "1-2", "2-3", "3-5", "5+"][k]: f"{np.mean(v):+.3f} (n{len(v)})" for k, v in sorted(gain.items())})
pv = Counter();
for i in ad: pv[str(D["add"][0][i]["prediction"]).strip()] += 1
print("most common additive predictions:", pv.most_common(8))
