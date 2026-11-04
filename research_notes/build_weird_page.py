"""Build docs/weird_cases.html: deep-dive EDA on question-level VSI-Bench results (v9 runs, step 5750 decayed branches, 32 frames) — failure patterns, biases, label problems, frame-count effects."""
import glob, json, os, re
from collections import Counter, defaultdict
import numpy as np

ROOT = "/shared/workspace/povejmo/math/projects/mvp"
W = f"{ROOT}/.claude/worktrees/je-xattn-proposal"
R = f"{ROOT}/results/train"
RUNS = {"lora0": "2B_v9_fmix_lora_s0", "lora1": "2B_v9_fmix_lora_s1", "add0": "2B_v9_fmix_additive_s0", "add1": "2B_v9_fmix_additive_s1", "xa0": "2B_v9_fmix_xa3_both_s0", "xa1": "2B_v9_fmix_xa3_both_s1"}
NAME = {"lora0": "LoRA s0", "lora1": "LoRA s1", "add0": "Additive s0", "add1": "Additive s1", "xa0": "Cross-attn s0", "xa1": "Cross-attn s1", "ss": "SpatialStack-4B"}
COL = {"lora": "var(--lora)", "add": "var(--add)", "xa": "var(--xa)", "ss": "var(--ss)"}
NUMT = ["object_counting", "object_abs_distance", "object_size_estimation", "room_size_estimation"]
load = lambda p: {x["id"]: x for x in map(json.loads, open(p))}
D = {k: load(f"{R}/{v}/branches/step5750/eval/vsi_f32_full.jsonl") for k, v in RUNS.items()}
F128 = {k: load(f"{R}/{v}/branches/step5750/eval/vsi_f128_full.jsonl") for k, v in RUNS.items() if k in ("add0", "add1", "lora1")}
ss = {}
for r in map(json.loads, open(glob.glob(f"{ROOT}/results/baselines/spatialstack/full_qwen35/vsibench/*/*samples_vsibench.jsonl")[0])):
    s = r["vsibench_score"]; p = r["filtered_resps"]
    while isinstance(p, list): p = p[0]
    ss[s["id"]] = {"prediction": p, "score": float(s.get("accuracy", s.get("MRA:.5:.95:.05")))}
D["ss"] = ss
ids = sorted(D["add0"]); Q = D["add0"]
num = lambda s: (float(re.search(r"-?\d+(\.\d+)?", str(s)).group()) if re.search(r"\d", str(s)) else np.nan)
letter = lambda s: str(s).strip()[0].upper()
opt_text = lambda x, L: next((o.split(".", 1)[1].strip() for o in x["options"] if o.strip()[0].upper() == L), None)
pred = lambda k, i: D[k][i]["prediction"]
sc = lambda k, i: float(D[k][i]["score"])
MIRROR = {"left": "right", "right": "left", "front-left": "front-right", "front-right": "front-left", "back-left": "back-right", "back-right": "back-left"}


def mra(p, g):
    return float(np.mean([abs(p - g) / g < 1 - th for th in np.arange(0.5, 1.0, 0.05)]))


def boot(a, b=None, n=3000, seed=0):
    rng = np.random.default_rng(seed); a = np.asarray(a, float); idx = rng.integers(0, len(a), (n, len(a)))
    d = a[idx].mean(1) if b is None else (a[idx] - np.asarray(b, float)[idx]).mean(1)
    return float(np.mean(a if b is None else a - np.asarray(b))), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


sgn = lambda v: f"{v:+.1f}"
pm = lambda m, lo, hi: f"{sgn(m)} [{sgn(lo)}, {sgn(hi)}]"
cls = lambda m, lo: "pos" if lo > 0 else ("neg" if m < 0 and lo < 0 else "")


# ---------- SVG helpers ----------
def grouped_bars(groups, series, title, xlab, ylab, ymax=100, w=900, h=360, fmt="{:.0f}", ref=None, nlab=None):
    """groups: list of labels; series: list of (name, color, values[, (lo, hi) per value])."""
    L, Rm, T, B = 56, 16, 44, 64; pw, ph = w - L - Rm, h - T - B
    gw = pw / len(groups); bw = min(28, gw * 0.8 / len(series)); out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{title}">']
    out.append(f'<text class="title" x="{L}" y="20">{title}</text>')
    for yv in range(0, ymax + 1, 20 if ymax >= 60 else 10):
        y = T + ph - ph * yv / ymax
        out.append(f'<line class="grid" x1="{L}" x2="{w - Rm}" y1="{y:.1f}" y2="{y:.1f}"/><text class="tick" x="{L - 6}" y="{y + 4:.1f}" text-anchor="end">{yv}</text>')
    if ref is not None:
        y = T + ph - ph * ref[0] / ymax; out.append(f'<line x1="{L}" x2="{w - Rm}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--ink3)" stroke-dasharray="4 4"/><text class="tick" x="{w - Rm}" y="{y - 4:.1f}" text-anchor="end">{ref[1]}</text>')
    for gi, g in enumerate(groups):
        x0 = L + gi * gw + (gw - bw * len(series)) / 2
        for si, s in enumerate(series):
            v = s[2][gi]; y = T + ph - ph * v / ymax; x = x0 + si * bw
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw - 3:.1f}" height="{ph - (y - T):.1f}" fill="{s[1]}" rx="2"/>')
            out.append(f'<text class="val" x="{x + (bw - 3) / 2:.1f}" y="{y - 4:.1f}" text-anchor="middle">{fmt.format(v)}</text>')
            if len(s) > 3 and s[3][gi] is not None:
                lo, hi = s[3][gi]; out.append(f'<line x1="{x + (bw - 3) / 2:.1f}" x2="{x + (bw - 3) / 2:.1f}" y1="{T + ph - ph * hi / ymax:.1f}" y2="{T + ph - ph * lo / ymax:.1f}" stroke="var(--ink)" stroke-width="1.2"/>')
        out.append(f'<text class="lab" x="{L + gi * gw + gw / 2:.1f}" y="{T + ph + 16}" text-anchor="middle">{g}</text>')
        if nlab: out.append(f'<text class="tick" x="{L + gi * gw + gw / 2:.1f}" y="{T + ph + 30}" text-anchor="middle">{nlab[gi]}</text>')
    lx = L
    for s in series:
        out.append(f'<rect x="{lx}" y="{T - 16}" width="10" height="10" fill="{s[1]}" rx="2"/><text class="lab" x="{lx + 14}" y="{T - 7}">{s[0]}</text>'); lx += 14 + 7 * len(s[0]) + 18
    out.append(f'<text class="axlab" x="{L + pw / 2:.0f}" y="{h - 8}" text-anchor="middle">{xlab}</text><text class="axlab" transform="translate(14 {T + ph / 2:.0f}) rotate(-90)" text-anchor="middle">{ylab}</text></svg>')
    return "".join(out)


def hist_pair(bins_lab, a, b, la, lb, ca, cb, title, xlab, ylab, w=900, h=330):
    return grouped_bars(bins_lab, [(la, ca, a), (lb, cb, b)], title, xlab, ylab, ymax=int(np.ceil(max(max(a), max(b)) / 10) * 10), w=w, h=h)


# ---------- 1. route planning ----------
RT = [Q[i] for i in ids if Q[i]["task"] == "route_planning"]
first = lambda s: s.split(",")[0].strip()
gt_first = lambda x: first(opt_text(x, letter(x["ground_truth"])))
tb = [x for x in RT if gt_first(x) == "Turn Back"]; ntb = [x for x in RT if gt_first(x) != "Turn Back"]
route_rows = ""; rt_tb, rt_ntb, rt_names, rt_cols = [], [], [], []
for k in list(RUNS) + ["ss"]:
    a = 100 * np.mean([sc(k, x["id"]) for x in tb]); b = 100 * np.mean([sc(k, x["id"]) for x in ntb]); said = sum(first(opt_text(x, letter(pred(k, x["id"])))) == "Turn Back" for x in RT if letter(pred(k, x["id"])) in "ABCD")
    route_rows += f"<tr><td>{NAME[k]}</td><td class='num'>{a:.0f}%</td><td class='num'>{b:.0f}%</td><td class='num'>{said}</td><td class='num'>{100 * np.mean([sc(k, x['id']) for x in RT]):.1f}</td></tr>"
    rt_tb.append(a); rt_ntb.append(b); rt_names.append(NAME[k])
opt_first = Counter(first(o.split(".", 1)[1]) for x in RT for o in x["options"])
n_tb_opts = Counter(sum(first(o.split(".", 1)[1]) == "Turn Back" for o in x["options"]) for x in tb)
turns = lambda s: len(s.split(","))
turn_rows = "".join(f"<tr><td class='num'>{n}</td><td class='num'>{sum(turns(opt_text(x, letter(x['ground_truth']))) == n for x in RT)}</td>" + "".join(f"<td class='num'>{100 * np.mean([sc(k, x['id']) for x in RT if turns(opt_text(x, letter(x['ground_truth']))) == n]):.0f}%</td>" for k in ("lora1", "add0", "xa0", "ss")) + f"<td class='num'>{100 / np.mean([len(x['options']) for x in RT if turns(opt_text(x, letter(x['ground_truth']))) == n]):.0f}%</td></tr>" for n in (1, 2, 3))
route_chart = grouped_bars(["Correct route starts with Turn Back", "Correct route starts with Turn Left / Right"], [("LoRA s1", COL["lora"], [rt_tb[1], rt_ntb[1]]), ("Additive s0", COL["add"], [rt_tb[2], rt_ntb[2]]), ("Cross-attn s0", COL["xa"], [rt_tb[4], rt_ntb[4]]), ("SpatialStack-4B", COL["ss"], [rt_tb[6], rt_ntb[6]])],
                           "Route planning accuracy, split by the first move of the correct route", "First move of the correct answer", "Accuracy (%)", ref=(100 / np.mean([len(x["options"]) for x in RT]), "chance (mean over 3–4 options)"), nlab=[f"n = {len(tb)}", f"n = {len(ntb)}"])
potential = 100 * (len(tb) * np.mean(rt_ntb[2:4]) / 100) / len(RT)

# ---------- 2. relative direction ----------
TRAIN_DIR = {"right": 35.4, "left": 34.5, "front-left": 10.1, "front-right": 9.5, "back": 4.7, "back-left": 3.0, "back-right": 2.8}  # research_notes/train_balance.py on VSI-590K
dir_groups, dir_n, dir_vals, dir_ci, dir_rows = [], [], {"lora": [], "add": [], "xa": []}, {"add": [], "xa": []}, ""
for qt, lab in (("object_rel_direction_easy", "easy"), ("object_rel_direction_medium", "medium"), ("object_rel_direction_hard", "hard")):
    xs = [Q[i] for i in ids if Q[i]["question_type"] == qt]
    for c in ("left", "right", "back", "front-left", "front-right", "back-left", "back-right"):
        sel = [x["id"] for x in xs if opt_text(x, letter(x["ground_truth"])) == c]
        if not sel: continue
        arm = {a: [np.mean([sc(f"{a}0", i), sc(f"{a}1", i)]) for i in sel] for a in ("lora", "add", "xa")}
        ma, la, ha = boot(arm["add"], arm["lora"]); mx, lx, hx = boot(arm["xa"], arm["lora"])
        dir_groups.append(f"{lab}: {c}"); dir_n.append(f"n = {len(sel)}")
        for a in arm: dir_vals[a].append(100 * np.mean(arm[a]))
        dir_ci["add"].append((100 * la, 100 * ha)); dir_ci["xa"].append((100 * lx, 100 * hx))
        share_bench = 100 * len(sel) / len(xs)
        dir_rows += f"<tr><td>{lab}</td><td>{c}</td><td class='num'>{len(sel)}</td><td class='num'>{share_bench:.0f}%</td><td class='num'>{TRAIN_DIR[c]:.1f}%</td><td class='num'>{100 * np.mean(arm['lora']):.0f}</td><td class='num'>{100 * np.mean(arm['add']):.0f}</td><td class='num'>{100 * np.mean(arm['xa']):.0f}</td><td class='num {cls(100 * ma, 100 * la)}'>{pm(100 * ma, 100 * la, 100 * ha)}</td><td class='num {cls(100 * mx, 100 * lx)}'>{pm(100 * mx, 100 * lx, 100 * hx)}</td></tr>"
dir_chart = grouped_bars(dir_groups, [("LoRA (2 seeds)", COL["lora"], dir_vals["lora"]), ("Additive (2 seeds)", COL["add"], dir_vals["add"]), ("Cross-attn (2 seeds)", COL["xa"], dir_vals["xa"])], "Relative direction accuracy by the true answer", "Question difficulty and true answer", "Accuracy (%)", w=1000, nlab=dir_n)
back_said = {k: (sum("back" in (opt_text(x, letter(pred(k, x["id"]))) or "") for x in Q.values() if x["question_type"] in ("object_rel_direction_medium", "object_rel_direction_hard")), sum("back" in opt_text(x, letter(x["ground_truth"])) for x in Q.values() if x["question_type"] in ("object_rel_direction_medium", "object_rel_direction_hard"))) for k in list(RUNS) + ["ss"]}
back_rows = "".join(f"<tr><td>{NAME[k]}</td><td class='num'>{s}</td><td class='num'>{t}</td><td class='num'>{s / t:.2f}</td></tr>" for k, (s, t) in back_said.items())

# ---------- 3. counting ----------
CT = [i for i in ids if Q[i]["task"] == "object_counting"]
g = np.array([num(Q[i]["ground_truth"]) for i in CT])
const2 = 100 * np.mean([mra(2, v) for v in g])
bins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; blab = [str(b) for b in bins[:-1]] + ["10+"]
hist = lambda vals: [int(sum(1 for v in vals if (v == b if b < 10 else v >= 10))) for b in bins]
count_chart = hist_pair(blab, hist(g), hist([num(pred("add0", i)) for i in CT]), "Ground truth", "Additive s0 prediction", "var(--ink3)", COL["add"], "Object counting: how many questions have each count, truth vs prediction", "Count", "Number of questions")
by_true = {}
for c in (2, 3, 4, 5, 6, 7):
    sel = [i for i in CT if (num(Q[i]["ground_truth"]) == c if c < 7 else num(Q[i]["ground_truth"]) >= 7)]
    by_true[c] = (len(sel), {k: (100 * np.mean([sc(k, i) for i in sel]), np.median([num(pred(k, i)) for i in sel])) for k in ("lora1", "add0", "xa0", "ss")})
count_rows = "".join(f"<tr><td class='num'>{c if c < 7 else '7+'}</td><td class='num'>{n}</td>" + "".join(f"<td class='num'>{v[0]:.0f} / {v[1]:.0f}</td>" for v in d.values()) + "</tr>" for c, (n, d) in by_true.items())
said2 = {k: (sum(num(pred(k, i)) == 2 for i in CT), sum(num(pred(k, i)) == 2 and num(Q[i]["ground_truth"]) == 2 for i in CT), 100 * np.mean([sc(k, i) for i in CT if num(pred(k, i)) != 2])) for k in ("lora1", "add0", "xa0", "ss")}
count_overall = {k: 100 * np.mean([sc(k, i) for i in CT]) for k in ("lora1", "add0", "xa0", "ss")}

# ---------- 4. absolute distance ----------
AD = [i for i in ids if Q[i]["task"] == "object_abs_distance"]
gad = np.array([num(Q[i]["ground_truth"]) for i in AD])
const_ad = 100 * np.mean([mra(np.median(gad), v) for v in gad])
ratio = {k: np.array([num(pred(k, i)) / num(Q[i]["ground_truth"]) for i in AD]) for k in ("lora1", "add0", "xa0", "ss")}
edges = [0, 1 / 3, 0.5, 0.7, 0.9, 1.1, 1.4, 2, 3, 1e9]; elab = ["<⅓", "⅓–½", "½–0.7", "0.7–0.9", "0.9–1.1", "1.1–1.4", "1.4–2", "2–3", ">3"]
rh = lambda r: [int(((r >= lo) & (r < hi)).sum()) for lo, hi in zip(edges[:-1], edges[1:])]
ad_chart = grouped_bars(elab, [("LoRA s1", COL["lora"], rh(ratio["lora1"])), ("Additive s0", COL["add"], rh(ratio["add0"])), ("SpatialStack-4B", COL["ss"], rh(ratio["ss"]))], "Absolute distance: predicted ÷ true distance", "Prediction as a multiple of the true distance (1 = exact; inside 0.5–2 scores partially)", "Number of questions", ymax=int(np.ceil(max(max(rh(ratio[k])) for k in ("lora1", "add0", "ss")) / 50) * 50), fmt="{:.0f}")
ad_rows = "".join(f"<tr><td>{NAME[k]}</td><td class='num'>{np.median(ratio[k]):.2f}</td><td class='num'>{int((ratio[k] < 1 / 3).sum())}</td><td class='num'>{int((ratio[k] > 3).sum())}</td><td class='num'>{100 * np.mean([sc(k, i) for i in AD]):.1f}</td></tr>" for k in ("lora1", "add0", "xa0", "ss"))
worst = sorted(AD, key=lambda i: -abs(np.log(num(pred("add0", i)) / num(Q[i]["ground_truth"]))))[:6]
objs = lambda i: re.search(r"between the (.+?) \(in", Q[i]["question"])[1]
worst_rows = "".join(f"<tr><td class='num'>{i}</td><td>{Q[i]['dataset']} {Q[i]['scene_name']}</td><td>{objs(i)}</td><td class='num'>{Q[i]['ground_truth']}</td>" + "".join(f"<td class='num'>{pred(k, i)}</td>" for k in ("lora1", "add0", "xa0", "ss")) + "</tr>" for i in worst)
dist_bins = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 99)]
dist_rows = ""
for lo, hi in dist_bins:
    sel = [i for i in AD if lo <= num(Q[i]["ground_truth"]) < hi]
    dist_rows += f"<tr><td>{lo}–{hi if hi < 99 else '∞'} m</td><td class='num'>{len(sel)}</td>" + "".join(f"<td class='num'>{100 * np.mean([sc(k, i) for i in sel]):.0f} / {np.median([num(pred(k, i)) / num(Q[i]['ground_truth']) for i in sel]):.2f}</td>" for k in ("lora1", "add0", "xa0", "ss")) + "</tr>"

# ---------- 5. mirror scene ----------
DR = [x for x in Q.values() if x["task"] == "object_rel_direction"]
allmir = lambda x: all(opt_text(x, letter(pred(k, x["id"]))) == MIRROR.get(opt_text(x, letter(x["ground_truth"]))) for k in ("lora1", "add0", "xa0", "ss"))
n_allmir = sum(allmir(x) for x in DR); scenes_mir = Counter(x["scene_name"] for x in DR if allmir(x))
scene = "47204578"; SC = sorted([x for x in Q.values() if x["scene_name"] == scene and x["task"] == "object_rel_direction"], key=lambda x: x["id"])
scene_rows = "".join(f"<tr><td class='num'>{x['id']}</td><td>{re.sub(r' An object is.*| The directions refer.*', '', x['question']).replace('If I am standing by the ', 'By the ').replace(' and facing the ', ', facing the ')[:90]}</td><td>{opt_text(x, letter(x['ground_truth']))}</td>" + "".join(f"<td class='{'neg' if letter(pred(k, x['id'])) != letter(x['ground_truth']) else 'pos'}'>{opt_text(x, letter(pred(k, x['id'])))}</td>" for k in ("lora1", "add0", "xa0", "ss")) + "</tr>" for x in SC)
orient_rows = ""
for key in sorted({(x["dataset"], tuple(x["frame_hw"])) for x in DR}):
    sel = [x for x in DR if (x["dataset"], tuple(x["frame_hw"])) == key]
    orient_rows += f"<tr><td>{key[0]}</td><td>{key[1][1]}×{key[1][0]} ({'portrait' if key[1][0] > key[1][1] else 'landscape'})</td><td class='num'>{len(sel)}</td>" + "".join(f"<td class='num'>{100 * np.mean([sc(k, x['id']) for x in sel]):.0f}%</td>" for k in ("lora1", "add0", "xa0", "ss")) + "</tr>"

# ---------- 6. frames ----------
fr_groups = ["Shortest third", "Middle third", "Longest third"]; fr_series = []; fr_task = defaultdict(dict)
nf = np.array([max(Q[i]["frame_indices"]) + 1 for i in ids]); qv = np.percentile(nf, [33, 66]); sels = [nf <= qv[0], (nf > qv[0]) & (nf <= qv[1]), nf > qv[1]]
for k, col in (("lora1", COL["lora"]), ("add0", COL["add"]), ("add1", COL["add"])):
    s32 = np.array([sc(k, i) for i in ids]); s128 = np.array([float(F128[k][i]["score"]) for i in ids]); vals, cis = [], []
    for s in sels:
        m, lo, hi = boot(s128[s], s32[s]); vals.append(100 * m); cis.append((100 * lo, 100 * hi))
    fr_series.append((NAME[k], col if k != "add1" else "#f0a07a", vals, cis))
    for t in sorted({Q[i]["task"] for i in ids}):
        s = np.array([Q[i]["task"] == t for i in ids]); m, lo, hi = boot(s128[s], s32[s]); fr_task[t][k] = (100 * m, 100 * lo, 100 * hi)
    mc = np.array([Q[i]["task"] not in NUMT for i in ids]); fr_task["flips"][k] = (int(((s32 == 1) & (s128 == 0) & mc).sum()), int(((s32 == 0) & (s128 == 1) & mc).sum()))
fr_chart = grouped_bars(fr_groups, fr_series, "Gain from 128 frames over 32, by video length", f"Video length (terciles: ≤{qv[0]:.0f}, ≤{qv[1]:.0f}, >{qv[1]:.0f} frames)", "Score change (points)", ymax=4, fmt="{:+.1f}", nlab=[f"n = {int(s.sum())}" for s in sels])
fr_rows = "".join(f"<tr><td>{t.replace('object_', '').replace('obj_', '').replace('_', ' ')}</td>" + "".join(f"<td class='num {cls(v[0], v[1])}'>{pm(*v)}</td>" for v in fr_task[t].values()) + "</tr>" for t in fr_task if t != "flips")

# ---------- 7. shared difficulty, seeds, label noise ----------
MC = [i for i in ids if Q[i]["task"] not in NUMT]
allc = np.array([[sc(k, i) for k in list(RUNS) + ["ss"]] for i in MC])
n_allright, n_allwrong, n_mixed = int((allc.min(1) == 1).sum()), int((allc.max(1) == 0).sum()), int(((allc.min(1) < 1) & (allc.max(1) > 0)).sum())
exp_allwrong = len(MC) * float(np.prod(1 - allc.mean(0)))
same_wrong = [i for i in MC if len({letter(pred(k, i)) for k in list(RUNS) + ["ss"]}) == 1 and sc("add0", i) == 0]
same_wrong_task = Counter(Q[i]["task"] for i in same_wrong)
seed_rows = ""
for arm, (a, b) in {"LoRA": ("lora0", "lora1"), "Additive": ("add0", "add1"), "Cross-attn": ("xa0", "xa1")}.items():
    seed_rows += f"<tr><td>{arm}</td>" + "".join(f"<td class='num'>{100 * np.mean([(num(pred(a, i)) == num(pred(b, i))) if t in NUMT else (letter(pred(a, i)) == letter(pred(b, i))) for i in ids if Q[i]['task'] == t]):.0f}%</td>" for t in ["object_counting", "object_abs_distance", "object_size_estimation", "room_size_estimation", "object_rel_distance", "object_rel_direction", "route_planning", "obj_appearance_order"]) + "</tr>"

# ---------- 8. prediction values vs training ----------
# from research_notes/deep_eda2.py section F (VSI-590K)
TRAIN = {"object_counting": ("none", "2.0 (IQR 2–4)", "0%", "99%"), "object_abs_distance": ("m 29%, cm 23%, inch 23%, feet 24%", "15.1 (IQR 3.3–120)", "69%", "32%"), "object_size_estimation": ("cm 33%, inch 23%, m 22%, feet 22%", "18.5 (IQR 2–79)", "63%", "0%"), "room_size_estimation": ("m² 60%, ft² 40%", "73 (IQR 19.5–191.5)", "49%", "5%")}
val_rows = ""
for t in NUMT:
    sel = [i for i in ids if Q[i]["task"] == t]; gt = np.array([num(Q[i]["ground_truth"]) for i in sel]); pv = Counter(num(pred("add0", i)) for i in sel)
    val_rows += f"<tr><td>{t.replace('object_', '').replace('_', ' ')}</td><td>{TRAIN[t][0]}</td><td class='num'>{TRAIN[t][1]}</td><td class='num'>{np.median(gt):.1f} (IQR {np.percentile(gt, 25):.1f}–{np.percentile(gt, 75):.1f})</td><td class='num'>{len(pv)} / {len(set(gt))}</td><td>{', '.join(f'{v:g} ×{c}' for v, c in pv.most_common(4))}</td><td class='num'>{TRAIN[t][3]}</td><td class='num'>{TRAIN[t][2]} / {100 * np.mean(gt != np.round(gt)):.0f}% / {100 * np.mean([v != round(v) for v in pv.elements()]):.0f}%</td></tr>"
RS = [i for i in ids if Q[i]["task"] == "room_size_estimation"]
rs_corr = {k: np.corrcoef(np.log([num(pred(k, i)) for i in RS]), np.log([num(Q[i]["ground_truth"]) for i in RS]))[0, 1] for k in ("lora1", "add0", "xa0", "ss")}

# ---------- 9. clean checks ----------
RD = [Q[i] for i in ids if Q[i]["task"] == "object_rel_distance"]
pos_pick = {}
for k in ("lora1", "add0", "xa0"):
    c = Counter()
    for x in RD:
        cands = [s.strip().lower() for s in re.search(r"\((.+?)\) is the closest", x["question"])[1].split(",")]; pt = opt_text(x, letter(pred(k, x["id"])))
        c[cands.index(pt.lower()) if pt and pt.lower() in cands else -1] += 1
    pos_pick[k] = [c[j] for j in range(4)]
AO = [Q[i] for i in ids if Q[i]["task"] == "obj_appearance_order"]
first_cat = {}
for k in ("lora1", "add0", "xa0"):
    first_cat[k] = sum((opt_text(x, letter(pred(k, x["id"]))) or "").lower().split(",")[0].strip() == x["question"].split("video:")[1].strip(" ?").split(",")[0].strip().lower() for x in AO)
truth_first = sum(opt_text(x, letter(x["ground_truth"])).lower().split(",")[0].strip() == x["question"].split("video:")[1].strip(" ?").split(",")[0].strip().lower() for x in AO)
unparse = sum((Q[i]["task"] not in NUMT and letter(pred(k, i)) not in "ABCD") or (Q[i]["task"] in NUMT and np.isnan(num(pred(k, i)))) for k in RUNS for i in ids)

CSS = open(f"{W}/research_notes/build_v9_page.py").read().split("<style>")[1].split("</style>")[0].replace("{{", "{").replace("}}", "}")
PAGE = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weird Cases and Patterns</title>
<meta name="description" content="Deep-dive EDA on question-level VSI-Bench results of the v9 runs: failure patterns, content biases, label problems and frame-count effects.">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{CSS}
.frames {{ width:100%; height:auto; border-radius:6px; display:block; }}
ol {{ margin:0; padding-left:22px; display:grid; gap:8px; }}
</style></head><body><main>
<header style="display:grid;gap:12px;padding-bottom:6px">
  <div class="eyebrow">MVP project · run v9 · question-level EDA · 5,130 questions · 32 frames · 2026-11-04</div>
  <h1>Weird Cases and Patterns</h1>
  <p class="sub">A deep dive into the per-question results of the six v9 models (LoRA only, additive, cross-attention; two seeds each; trunk step 5,750, decayed branch) and the SpatialStack-4B baseline, looking for anything the headline scores hide: answers the models never give, biases in content, scenes that fool everyone, and what more frames buy. Everything here is computed from the stored evaluation files, the same ones as the results CSV; no model was run.</p>
</header>

<section>
  <div class="tiles">
    <div class="tile"><b>{np.mean(rt_tb[:6]):.0f}%</b><span>Route planning accuracy when the correct route starts with "Turn Back" ({len(tb)} of {len(RT)} questions), versus {np.mean(rt_ntb[:6]):.0f}% otherwise. No model ever turns back first.</span></div>
    <div class="tile"><b>{sgn(np.mean([dir_vals['add'][i] - dir_vals['lora'][i] for i, g in enumerate(dir_groups) if 'back' in g]))}</b><span>Points that geometry adds on relative-direction questions whose answer is "back…", versus {sgn(np.mean([dir_vals['add'][i] - dir_vals['lora'][i] for i, g in enumerate(dir_groups) if 'back' not in g]))} on "left / right / front…". The gain is concentrated where the object is behind the viewer.</span></div>
    <div class="tile"><b>{const2:.0f}</b><span>Counting MRA of a model that always answers "2", against {count_overall['add0']:.0f} for additive. Most of the counting score needs no video.</span></div>
    <div class="tile"><b>{n_allwrong}</b><span>Multiple-choice questions all seven models get wrong (expected if errors were independent: {exp_allwrong:.0f}); {len(same_wrong)} of them with the same wrong letter from all seven.</span></div>
  </div>
</section>

<section id="route">
  <div class="eyebrow">Pattern 1 · route planning</div><h2>The models never turn back first</h2>
  <p class="lede">Route planning is the weakest task (about 40 points, chance about 30). Splitting the {len(RT)} questions by the first move of the correct route shows that the loss is almost entirely one case: when the correct route begins with "Turn Back", every model scores near zero. VSI-590K contains no route-planning questions at all, so this task is answered zero-shot, and "Turn Back" as an opening move seems to be something the models will not produce.</p>
  <figure><div class="scroll">{route_chart}</div><figcaption>Accuracy on route planning split by the first move of the correct option. The dashed line is chance for the mean number of options. Only seed 1 of LoRA and seed 0 of additive and cross-attention are drawn; the table has all runs.</figcaption></figure>
  <figure><div class="scroll"><table><thead><tr><th>Model</th><th class="num">Accuracy, truth starts Turn Back (n = {len(tb)})</th><th class="num">Accuracy, otherwise (n = {len(ntb)})</th><th class="num">Times the chosen option starts with Turn Back</th><th class="num">Route planning score</th></tr></thead><tbody>{route_rows}</tbody></table></div>
  <figcaption>"Turn Back" is not rare among the options: of the {sum(opt_first.values())} options across all route questions, {opt_first['Turn Back']} start with Turn Back, {opt_first['Turn Left']} with Turn Left and {opt_first['Turn Right']} with Turn Right. In the {len(tb)} questions whose answer starts with Turn Back, {n_tb_opts[1]} have exactly one such option and {n_tb_opts[2]} have two. The models pick a left or right option instead. If the models answered these questions at their usual {np.mean(rt_ntb[2:4]):.0f}% rate the task score would rise by about {potential:.0f} points, from ~{100 * np.mean([sc('add0', x['id']) for x in RT]):.0f} to ~{100 * np.mean([sc('add0', x['id']) for x in RT]) + potential:.0f}.</figcaption></figure>
  <figure><div class="scroll"><table><thead><tr><th class="num">Turns in the correct route</th><th class="num">Questions</th><th class="num">LoRA s1</th><th class="num">Additive s0</th><th class="num">Cross-attn s0</th><th class="num">SpatialStack-4B</th><th class="num">Chance</th></tr></thead><tbody>{turn_rows}</tbody></table></div>
  <figcaption>Accuracy by the number of turns in the correct route. One-turn routes are answered well above chance; routes with two or three turns are at or below chance for every model, including the 4B baseline. (Two-turn routes include most of the Turn Back cases.)</figcaption></figure>
</section>

<section id="direction">
  <div class="eyebrow">Pattern 2 · relative direction</div><h2>Geometry helps most when the object is behind you</h2>
  <p class="lede">Relative direction is where additive gains most over LoRA overall (+5 to +9 points). Splitting by the true answer shows the gain is not uniform: on "left", "right", "front-left" and "front-right" it is 2–9 points; on "back", "back-left" and "back-right" it is 15–33 points. These answers are rare in training (10.5% of VSI-590K direction answers are a "back" class) and all models under-predict them, LoRA most.</p>
  <figure><div class="scroll">{dir_chart}</div><figcaption>Accuracy by question difficulty and true answer, each arm pooled over its two seeds. n is the number of benchmark questions with that answer.</figcaption></figure>
  <figure><div class="scroll"><table><thead><tr><th>Difficulty</th><th>True answer</th><th class="num">n</th><th class="num">Share of benchmark</th><th class="num">Share of training answers</th><th class="num">LoRA</th><th class="num">Additive</th><th class="num">Cross-attn</th><th class="num">Additive − LoRA [95% CI]</th><th class="num">Cross-attn − LoRA [95% CI]</th></tr></thead><tbody>{dir_rows}</tbody></table></div>
  <figcaption>Paired bootstrap over questions, both seeds averaged per question. Green: interval excludes zero. Training shares are over the {176991:,} relative-direction questions of VSI-590K.</figcaption></figure>
  <figure><div class="scroll"><table><thead><tr><th>Model</th><th class="num">Answers containing "back"</th><th class="num">True answers containing "back"</th><th class="num">Ratio</th></tr></thead><tbody>{back_rows}</tbody></table></div>
  <figcaption>Medium and hard questions (the only ones with a "back" option). Every model says "back" less often than it is true; geometry closes about half of the gap.</figcaption></figure>
</section>

<section id="counting">
  <div class="eyebrow">Pattern 3 · object counting</div><h2>Most of the counting score comes from answering "2"</h2>
  <p class="lede">{100 * np.mean(g == 2):.0f}% of counting questions have the answer 2 (training: 41%). The models answer 2 on about {100 * np.mean([said2[k][0] for k in ('lora1', 'add0', 'xa0')]) / len(CT):.0f}% of questions. A model that always says 2 would score MRA {const2:.1f}; the real models score {min(count_overall.values()):.1f}–{max(count_overall.values()):.1f}, so the video is worth about {count_overall['add0'] - const2:.0f} points here.</p>
  <figure><div class="scroll">{count_chart}</div><figcaption>Number of questions per count, ground truth versus additive s0's predictions. The prediction distribution is squeezed toward 2 and 3; counts of 4–6 are under-produced, while 10+ is over-produced (the model says "10" for most large counts).</figcaption></figure>
  <figure><div class="scroll"><table><thead><tr><th class="num">True count</th><th class="num">Questions</th><th class="num">LoRA s1: MRA / median answer</th><th class="num">Additive s0</th><th class="num">Cross-attn s0</th><th class="num">SpatialStack-4B</th></tr></thead><tbody>{count_rows}</tbody></table></div>
  <figcaption>When the truth is 2, every model scores about 95. When the truth is 3 or 4 the median answer is still 2 and the score halves. Of the {said2['add0'][0]} questions where additive s0 answered 2, the truth was 2 in {said2['add0'][1]}; on the {len(CT) - said2['add0'][0]} questions where it answered something else its MRA is {said2['add0'][2]:.0f}, so those answers are informative but noisy.</figcaption></figure>
</section>

<section id="absdist">
  <div class="eyebrow">Pattern 4 · absolute distance</div><h2>LoRA under-estimates; geometry removes the tail</h2>
  <p class="lede">Absolute distance is the task with the second-largest geometry gain (+10 points). The gain comes from the far tail: LoRA answers less than a third of the true distance on {int((ratio['lora1'] < 1 / 3).sum())} questions, additive on {int((ratio['add0'] < 1 / 3).sum())}. A constant answer (the median distance, {np.median(gad):.1f} m) scores {const_ad:.0f}.</p>
  <figure><div class="scroll">{ad_chart}</div><figcaption>Distribution of prediction ÷ truth. MRA gives partial credit roughly inside 0.5×–2× and nothing outside; the two left-most bins are lost points.</figcaption></figure>
  <div class="two">
  <figure><div class="scroll"><table><thead><tr><th>Model</th><th class="num">Median pred ÷ truth</th><th class="num">Below ⅓×</th><th class="num">Above 3×</th><th class="num">Score</th></tr></thead><tbody>{ad_rows}</tbody></table></div><figcaption>All models are biased short (median ratio below 1), LoRA most.</figcaption></figure>
  <figure><div class="scroll"><table><thead><tr><th>True distance</th><th class="num">n</th><th class="num">LoRA s1: MRA / median ratio</th><th class="num">Additive s0</th><th class="num">Cross-attn s0</th><th class="num">SpatialStack-4B</th></tr></thead><tbody>{dist_rows}</tbody></table></div><figcaption>By true distance. Beyond 3 m LoRA's median answer is about half the truth; the geometry arms stay near 0.8–0.9.</figcaption></figure>
  </div>
  <figure><div class="scroll"><table><thead><tr><th class="num">id</th><th>Scene</th><th>Objects</th><th class="num">Truth (m)</th><th class="num">LoRA s1</th><th class="num">Additive s0</th><th class="num">Cross-attn s0</th><th class="num">SpatialStack-4B</th></tr></thead><tbody>{worst_rows}</tbody></table></div>
  <figcaption>The six questions additive s0 gets most wrong (by log ratio). Several are label ambiguities rather than model errors: in scene 47334362 (first row) the TV stands on a desk, so "the table" next to the TV is 0.2–0.4 m away while the labelled table is 4.4 m away; four models agree with each other and not with the label.</figcaption></figure>
  <figure><img class="frames" src="frames/47334362.jpg" alt="Eight frames of scene 47334362: a small bedroom with a TV on a desk and a large mirror" width="1440" height="720"><figcaption>Scene 47334362 (ARKitScenes), 8 of 3,536 frames with their frame index. The TV (frame 2525) sits on a desk; a mirror (frame 3030) reflects the bed.</figcaption></figure>
</section>

<section id="mirror">
  <div class="eyebrow">Pattern 5 · a scene that fools everyone</div><h2>A mirrored room produces mirrored answers</h2>
  <p class="lede">Among the {len(DR)} relative-direction questions, {n_allmir} are answered with the exact mirror of the truth (left for right, back-left for back-right) by all three of our models <em>and</em> by SpatialStack-4B, which uses a different frame pipeline. These are concentrated in {len(scenes_mir)} scenes; scene 47204578 alone has {scenes_mir[scene]} of them, every direction question asked about it. The video shows why.</p>
  <figure><img class="frames" src="frames/47204578.jpg" alt="Eight frames of scene 47204578: a living room with a large mirrored wall and mirror tiles" width="1440" height="720"><figcaption>Scene 47204578 (ARKitScenes), 8 of 1,646 frames. A floor-to-ceiling mirror (frame 470) reflects the sofa and the room; a second mirror (940) is tiled. The room also has two tables (a sideboard and a small white side table, frame 705), so "the table" in the questions is ambiguous.</figcaption></figure>
  <figure><div class="scroll"><table><thead><tr><th class="num">id</th><th>Question</th><th>Truth</th><th>LoRA s1</th><th>Additive s0</th><th>Cross-attn s0</th><th>SpatialStack-4B</th></tr></thead><tbody>{scene_rows}</tbody></table></div>
  <figcaption>Every direction question on this scene. All four models give the mirror image of the label on every one; either the reflection is being read as the room, or the label's "table" is not the one the models pick. Either way, these questions cannot be answered from the video as the label expects.</figcaption></figure>
  <figure><div class="scroll"><table><thead><tr><th>Dataset</th><th>Frame size</th><th class="num">n</th><th class="num">LoRA s1</th><th class="num">Additive s0</th><th class="num">Cross-attn s0</th><th class="num">SpatialStack-4B</th></tr></thead><tbody>{orient_rows}</tbody></table></div>
  <figcaption>Checked and ruled out: a left/right flip from portrait-video handling. ARKitScenes portrait videos score the same as landscape ones on relative direction, for every model.</figcaption></figure>
</section>

<section id="frames">
  <div class="eyebrow">Pattern 6 · frame count</div><h2>128 frames help on long videos and on relative direction, and hurt appearance order</h2>
  <p class="lede">Three models have a complete 128-frame evaluation. The gain over 32 frames is about +1 point overall, but it is not spread evenly: it comes from the longest third of the videos and from relative direction (+3 to +5), while appearance order gets worse for both additive seeds and room size for two of three models. About 8–13% of multiple-choice answers change when frames are added, in both directions.</p>
  <figure><div class="scroll">{fr_chart}</div><figcaption>Score change from 32 to 128 frames, paired by question, with 95% bootstrap intervals. Longer videos are sampled more sparsely at 32 frames, so they gain most.</figcaption></figure>
  <figure><div class="scroll"><table><thead><tr><th>Task</th><th class="num">LoRA s1</th><th class="num">Additive s0</th><th class="num">Additive s1</th></tr></thead><tbody>{fr_rows}</tbody></table></div>
  <figcaption>Change by task, points, with 95% CI. Multiple-choice answers that flipped (right→wrong / wrong→right): LoRA s1 {fr_task['flips']['lora1'][0]} / {fr_task['flips']['lora1'][1]}, additive s0 {fr_task['flips']['add0'][0]} / {fr_task['flips']['add0'][1]}, additive s1 {fr_task['flips']['add1'][0]} / {fr_task['flips']['add1'][1]}, of {len(MC)}.</figcaption></figure>
</section>

<section id="shared">
  <div class="eyebrow">Pattern 7 · shared difficulty and label noise</div><h2>The models fail together</h2>
  <p class="lede">Of the {len(MC)} multiple-choice questions, all seven models (six of ours and SpatialStack) are right on {n_allright} and wrong on {n_allwrong}. If each model's errors were independent at its own accuracy, about {exp_allwrong:.0f} questions would be missed by all seven. On {len(same_wrong)} questions all seven give the <em>same</em> wrong letter ({', '.join(f'{v} {k.replace("object_", "").replace("obj_", "").replace("_", " ")}' for k, v in same_wrong_task.most_common())}); these are the first candidates for label errors or ambiguous questions, like the mirror scene above.</p>
  <figure><div class="scroll"><table><thead><tr><th>Arm</th><th class="num">counting</th><th class="num">abs distance</th><th class="num">object size</th><th class="num">room size</th><th class="num">rel distance</th><th class="num">rel direction</th><th class="num">route planning</th><th class="num">appearance order</th></tr></thead><tbody>{seed_rows}</tbody></table></div>
  <figcaption>How often the two seeds of an arm give the identical answer. Letters agree 60–90% of the time (route planning least); exact numbers rarely, because numeric answers are continuous, but their scores agree (room-size log-correlation with truth is {rs_corr['lora1']:.2f} / {rs_corr['add0']:.2f} / {rs_corr['xa0']:.2f} for LoRA / additive / cross-attn, {rs_corr['ss']:.2f} for SpatialStack).</figcaption></figure>
</section>

<section id="values">
  <div class="eyebrow">Pattern 8 · the numbers the models say</div><h2>Predictions are quantised but not copied from training</h2>
  <p class="lede">Numeric predictions use far fewer distinct values than the truth, and some values recur oddly (additive s0 answers 10.7 m² to 21 room-size questions, 1.5 m to 100 distance questions). We checked whether these are the most common training answers: they are not. Training answers come in mixed units (cm, inches, feet, metres) that the model must convert from the question's unit, and the benchmark's most-asked values are not training modes.</p>
  <figure><div class="scroll"><table><thead><tr><th>Task</th><th>Training units</th><th class="num">Training answer median</th><th class="num">Benchmark truth median</th><th class="num">Distinct values: predicted / true</th><th>Most frequent predictions (additive s0)</th><th class="num">Predictions that are a top-20 training answer</th><th class="num">Has decimals: train / truth / pred</th></tr></thead><tbody>{val_rows}</tbody></table></div>
  <figcaption>Training statistics are over VSI-590K's absolute_count, absolute_distance_object, absolute_size_object and absolute_size_room questions ({12986 + 41095 + 24349 + 6403:,} with a video). Counting predictions are trivially "top-20 training answers" because counts are small integers. For the other tasks the overlap is at or near chance, so the recurring values are the model's own rounding, not memorised answers.</figcaption></figure>
</section>

<section id="clean">
  <div class="eyebrow">Checks that came back clean</div><h2>What is not going on</h2>
  <ul>
    <li><strong>No answer-letter bias.</strong> Predicted and true letter frequencies agree within ±1 point for every letter and model (full analysis on the J-Lens page). Note that in relative distance the options are always in the same order as the candidates in the question ({len(RD)} of {len(RD)}), so a letter bias and a list-position bias would be the same thing; there is neither: additive s0 picks positions 1–4 {pos_pick['add0']} times against a uniform truth.</li>
    <li><strong>No "first-named category" shortcut in appearance order.</strong> The chosen option starts with the first category named in the question {first_cat['lora1']} / {first_cat['add0']} / {first_cat['xa0']} times (LoRA / additive / cross-attn) against {truth_first} for the truth, out of {len(AO)}.</li>
    <li><strong>No left/right bias.</strong> On easy direction questions the models say left and right as often as the truth does (within 4 questions of 217).</li>
    <li><strong>No portrait-video problem.</strong> See the orientation table above.</li>
    <li><strong>No parsing failures.</strong> {unparse} unparseable answers in {len(ids) * len(RUNS):,} predictions; no answer longer than 6 characters; numeric answers are full decimals (1.4, 40.6, 81), not single digits.</li>
    <li><strong>No train/test scene overlap</strong> (checked earlier by scene id for ScanNet, ScanNet++ and ARKitScenes).</li>
    <li><strong>Video length is a mild, consistent effect:</strong> longer videos score lower on relative distance (correlation −0.19 with log length), absolute distance (−0.17) and counting (−0.12), not on the other tasks; 128 frames recover part of it.</li>
  </ul>
</section>

<section id="next">
  <div class="eyebrow">What this suggests</div><h2>Where the remaining points are</h2>
  <ol>
    <li><strong>Route planning, "Turn Back" (~{potential:.0f} points of task score, ~{potential / 8:.1f} overall).</strong> This is a behavioural gap, not a perception one: the models have the geometry to know the goal is behind them (they get "back" right 50–78% of the time in relative direction) but never emit a route starting with Turn Back. Cheapest tests, in order: (a) prompt-only, add one sentence explaining that Turn Back means a 180° turn, or a one-shot example; (b) look at the first-token logits over the three option letters on the {len(tb)} Turn Back questions to see whether Turn Back is a close second or never considered; (c) add synthetic route-planning questions to training, since VSI-590K has none. (a) and (b) are one evaluation job each.</li>
    <li><strong>Relative direction, "back" classes.</strong> Geometry already helps most here; the residual under-prediction tracks the training prior (10.5% back). Re-weighting or over-sampling "back" answers in training is a one-line change to the data loader.</li>
    <li><strong>Counting.</strong> The models default to 2 and under-produce 4–6. Worth checking the J-lens / object-word activations at the count token, and whether the frame mix (v9) moved anything here (it did not: v7 and v9 counting are within a point).</li>
    <li><strong>Label noise.</strong> The {len(same_wrong)} same-wrong-letter questions and the 29 all-mirror questions are a fixed cost shared by every model, about {100 * len(same_wrong) / len(MC):.1f}% of multiple-choice. They do not change any comparison between arms, but they cap the achievable score; report scores with and without them if the benchmark is used for a final claim.</li>
  </ol>
</section>

<footer>Built by <code>research_notes/build_weird_page.py</code> from <code>results/train/2B_v9_*/branches/step5750/eval/vsi_f{{32,128}}_full.jsonl</code>, the SpatialStack sample file, and VSI-590K (<code>research_notes/deep_eda.py</code>, <code>deep_eda2.py</code>, <code>mirror_check.py</code>, <code>train_balance.py</code>, <code>contact_sheet.py</code>). All intervals are 95% paired bootstraps over questions.</footer>
</main></body></html>'''
os.makedirs(f"{W}/docs", exist_ok=True)
open(f"{W}/docs/weird_cases.html", "w").write(PAGE)
print("written", len(PAGE), "chars")
