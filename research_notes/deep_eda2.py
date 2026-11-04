"""Follow-ups to deep_eda.py: constant-answer baselines, route planning 'Turn Back', where geometry helps in relative direction, 32 vs 128 frames by video length, suspicious scenes, modal predictions vs training-answer modes."""
import json, re
from collections import Counter, defaultdict
import numpy as np

ROOT = "/shared/workspace/povejmo/math/projects/mvp"
R = f"{ROOT}/results/train"
RUNS = {"lora0": "2B_v9_fmix_lora_s0", "lora1": "2B_v9_fmix_lora_s1", "add0": "2B_v9_fmix_additive_s0", "add1": "2B_v9_fmix_additive_s1", "xa0": "2B_v9_fmix_xa3_both_s0", "xa1": "2B_v9_fmix_xa3_both_s1"}
NUMT = ["object_counting", "object_abs_distance", "object_size_estimation", "room_size_estimation"]
load = lambda p: {x["id"]: x for x in map(json.loads, open(p))}
D = {k: load(f"{R}/{v}/branches/step5750/eval/vsi_f32_full.jsonl") for k, v in RUNS.items()}
F128 = {k: load(f"{R}/{v}/branches/step5750/eval/vsi_f128_full.jsonl") for k, v in RUNS.items() if k in ("add0", "add1", "lora1")}
ids = sorted(D["add0"]); Q = D["add0"]
num = lambda s: (float(re.search(r"-?\d+(\.\d+)?", str(s)).group()) if re.search(r"\d", str(s)) else np.nan)
letter = lambda s: str(s).strip()[0].upper()
opt_text = lambda x, L: next((o.split(".", 1)[1].strip() for o in x["options"] if o.strip()[0].upper() == L), None)


def mra(p, g):
    return float(np.mean([abs(p - g) / g < 1 - th for th in np.arange(0.5, 1.0, 0.05)]))


def boot(a, b=None, n=2000, seed=0):
    rng = np.random.default_rng(seed); a = np.asarray(a, float); idx = rng.integers(0, len(a), (n, len(a)))
    d = a[idx].mean(1) if b is None else (a[idx] - np.asarray(b, float)[idx]).mean(1)
    return float(np.mean(a if b is None else a - np.asarray(b))), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


print("########## A. CONSTANT-ANSWER BASELINES (what MRA does a model get by ignoring the video?) ##########")
for t in NUMT:
    g = np.array([num(Q[i]["ground_truth"]) for i in ids if Q[i]["task"] == t])
    best_c = max(set(np.round(g, 1)), key=lambda c: np.mean([mra(c, v) for v in g]))
    print(f"{t:24} always-median({np.median(g):.1f}) MRA {100 * np.mean([mra(np.median(g), v) for v in g]):.1f} | best constant {best_c} MRA {100 * np.mean([mra(best_c, v) for v in g]):.1f} | models: lora1 {100 * np.mean([D['lora1'][i]['score'] for i in ids if Q[i]['task'] == t]):.1f} add0 {100 * np.mean([D['add0'][i]['score'] for i in ids if Q[i]['task'] == t]):.1f}")
xs = [i for i in ids if Q[i]["task"] == "object_counting"]
for k in ("lora1", "add0", "xa0"):
    said2 = [i for i in xs if num(D[k][i]["prediction"]) == 2]; not2 = [i for i in xs if num(D[k][i]["prediction"]) != 2]
    print(f"counting {k}: said 2 on {len(said2)} q (truth really 2 on {sum(num(Q[i]['ground_truth']) == 2 for i in said2)}); on the {len(not2)} q where it said something else MRA {100 * np.mean([D[k][i]['score'] for i in not2]):.1f}, truth=2 among them {sum(num(Q[i]['ground_truth']) == 2 for i in not2)}")

print("\n########## B. ROUTE PLANNING: the missing 'Turn Back' ##########")
xs = [Q[i] for i in ids if Q[i]["task"] == "route_planning"]
first = lambda s: s.split(",")[0].strip()
tb = [x for x in xs if first(opt_text(x, letter(x["ground_truth"]))) == "Turn Back"]; ntb = [x for x in xs if x not in tb]
print(f"{len(tb)} of {len(xs)} route questions have a correct answer starting with 'Turn Back'")
for k in RUNS:
    print(f"  {k:5} accuracy when truth starts Turn Back {100 * np.mean([D[k][x['id']]['score'] for x in tb]):.0f}%  vs otherwise {100 * np.mean([D[k][x['id']]['score'] for x in ntb]):.0f}%")
nopt_tb = Counter(sum(first(o.split(".", 1)[1]) == "Turn Back" for o in x["options"]) for x in tb)
print("  among those, number of options that start with Turn Back:", dict(nopt_tb))
print("  what the models pick instead (first turn of chosen option), add0:", Counter(first(opt_text(x, letter(D['add0'][x['id']]['prediction']))) for x in tb))
print("  all options of one example:", tb[0]["id"], tb[0]["options"], "truth", tb[0]["ground_truth"])
print("  per-option first-turn distribution over ALL route options:", Counter(first(o.split(".", 1)[1]) for x in xs for o in x["options"]))
turns = lambda s: len(s.split(","))
print("  accuracy by number of turns in the correct answer (add0, lora1, n):", {n: (round(100 * np.mean([D['add0'][x['id']]['score'] for x in xs if turns(opt_text(x, letter(x['ground_truth']))) == n])), round(100 * np.mean([D['lora1'][x['id']]['score'] for x in xs if turns(opt_text(x, letter(x['ground_truth']))) == n])), sum(turns(opt_text(x, letter(x['ground_truth']))) == n for x in xs)) for n in (1, 2, 3)})
print("  in questions whose truth does NOT start with Turn Back, how often does an option start with Turn Back and get chosen (add0):", sum(first(opt_text(x, letter(D['add0'][x['id']]['prediction']))) == 'Turn Back' for x in ntb), "/ options available:", sum(any(first(o.split('.', 1)[1]) == 'Turn Back' for o in x['options']) for x in ntb))

print("\n########## C. RELATIVE DIRECTION: where does geometry help? (additive − LoRA, by true answer; both seeds pooled) ##########")
for qt in ("object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard"):
    xs = [Q[i] for i in ids if Q[i]["question_type"] == qt]
    for cls in sorted({opt_text(x, letter(x["ground_truth"])) for x in xs}):
        sel = [x["id"] for x in xs if opt_text(x, letter(x["ground_truth"])) == cls]
        a = [np.mean([D["add0"][i]["score"], D["add1"][i]["score"]]) for i in sel]; l = [np.mean([D["lora0"][i]["score"], D["lora1"][i]["score"]]) for i in sel]; c = [np.mean([D["xa0"][i]["score"], D["xa1"][i]["score"]]) for i in sel]
        m, lo, hi = boot(a, l); m2, lo2, hi2 = boot(c, l)
        print(f"{qt[21:]:7} {cls:12} n{len(sel):4} lora {100 * np.mean(l):.0f} add {100 * np.mean(a):.0f} xa {100 * np.mean(c):.0f} | add−lora {100 * m:+.1f} [{100 * lo:+.1f},{100 * hi:+.1f}] | xa−lora {100 * m2:+.1f} [{100 * lo2:+.1f},{100 * hi2:+.1f}]")
print("--- is 'back' predicted less often than it is true? (said/true ratio) ---")
for k in RUNS:
    xs = [Q[i] for i in ids if Q[i]["question_type"] in ("object_rel_direction_medium", "object_rel_direction_hard")]
    said = sum("back" in (opt_text(x, letter(D[k][x["id"]]["prediction"])) or "") for x in xs); tru = sum("back" in opt_text(x, letter(x["ground_truth"])) for x in xs)
    print(f"  {k}: said back {said} / true back {tru} = {said / tru:.2f}")

print("\n########## D. 32 vs 128 FRAMES, per question, by video length ##########")
for k in F128:
    rows = [(Q[i]["task"], max(Q[i]["frame_indices"]) + 1, D[k][i]["score"], F128[k][i]["score"]) for i in ids]
    nf = np.array([r[1] for r in rows]); s32 = np.array([r[2] for r in rows]); s128 = np.array([r[3] for r in rows])
    q = np.percentile(nf, [33, 66])
    for lab, sel in (("short", nf <= q[0]), ("mid", (nf > q[0]) & (nf <= q[1])), ("long", nf > q[1])):
        m, lo, hi = boot(s128[sel], s32[sel])
        print(f"  {k} {lab:5} videos (n{sel.sum()}): f32 {100 * s32[sel].mean():.1f} f128 {100 * s128[sel].mean():.1f} gain {100 * m:+.1f} [{100 * lo:+.1f},{100 * hi:+.1f}]")
    per_task = {}
    for t in sorted({r[0] for r in rows}):
        sel = np.array([r[0] == t for r in rows]); per_task[t[:14]] = f"{100 * (s128[sel] - s32[sel]).mean():+.1f}"
    print(f"  {k} gain by task:", per_task)
    mc = np.array([r[0] not in NUMT for r in rows])
    print(f"  {k} MC flips: right→wrong {int(((s32 == 1) & (s128 == 0) & mc).sum())}, wrong→right {int(((s32 == 0) & (s128 == 1) & mc).sum())} of {int(mc.sum())}")

print("\n########## E. SUSPICIOUS SCENES ##########")
for sid in (931, 1037):
    scene = Q[sid]["scene_name"]
    print(f"--- scene {scene} ({Q[sid]['dataset']}) ---")
    for i in ids:
        x = Q[i]
        if x["scene_name"] != scene or x["task"] not in ("object_abs_distance", "object_rel_direction", "room_size_estimation"):
            continue
        preds = " ".join(f"{k}={str(D[k][i]['prediction']):>4}" for k in ("lora1", "add0", "xa0"))
        if x["task"] == "object_rel_direction":
            preds += "  (" + ", ".join(opt_text(x, letter(D[k][i]["prediction"])) for k in ("lora1", "add0", "xa0")) + ")"
        print(f"  {i:5} {x['question_type'][7:28]:21} truth {x['ground_truth']:>6} {(opt_text(x, letter(x['ground_truth'])) if x['options'] else ''):12} {preds} | {re.sub(r'An object is.*|If multiple.*', '', x['question']).replace(chr(10), ' ')[:95].strip()}")

print("\n########## F. TRAINING ANSWER DISTRIBUTION vs MODAL PREDICTIONS ##########")
TMAP = {"absolute_distance_object": "object_abs_distance", "absolute_size_object": "object_size_estimation", "absolute_size_room": "room_size_estimation", "absolute_count": "object_counting"}
tr = defaultdict(Counter); units = defaultdict(Counter); n_tr = Counter()
for l in open(f"{ROOT}/data/raw/VSI-590K/vsi_590k.jsonl"):
    if '"absolute_' not in l:
        continue
    r = json.loads(l)
    t = TMAP.get(r.get("question_type"))
    if not t or "video" not in r:
        continue
    q = r["conversations"][0]["value"]; a = r["conversations"][1]["value"].strip()
    u = "cm" if re.search(r"\bcm\b|centimet", q) else "inch" if "inch" in q else "feet" if re.search(r"\bfeet\b|\bft\b|square foot|square feet", q) else "m" if re.search(r"\bmeter|\bm\b|m²|square met", q) else "none"
    units[t][u] += 1; n_tr[t] += 1
    try:
        tr[t][round(float(a), 1)] += 1
    except ValueError:
        tr[t]["non-numeric"] += 1
for t in NUMT:
    preds = Counter(num(D["add0"][i]["prediction"]) for i in ids if Q[i]["task"] == t)
    top20 = {a for a, _ in tr[t].most_common(20)}
    print(f"{t:24} train n{n_tr[t]} units {dict(units[t])}")
    print(f"   train top answers {[(v, round(100 * c / n_tr[t], 1)) for v, c in tr[t].most_common(8)]}")
    print(f"   model top preds   {preds.most_common(6)}  | share of model predictions that are a top-20 training answer: {100 * sum(c for v, c in preds.items() if v in top20) / sum(preds.values()):.0f}%")
    vals = np.array([v for v, c in tr[t].items() if v != "non-numeric" for _ in range(c)], float)
    g = np.array([num(Q[i]["ground_truth"]) for i in ids if Q[i]["task"] == t])
    print(f"   train answer median {np.median(vals):.1f} (IQR {np.percentile(vals, 25):.1f}–{np.percentile(vals, 75):.1f}) vs bench truth median {np.median(g):.1f} (IQR {np.percentile(g, 25):.1f}–{np.percentile(g, 75):.1f}); answers with a decimal part: train {100 * np.mean(vals != np.round(vals)):.0f}%, bench truth {100 * np.mean(g != np.round(g)):.0f}%, model preds {100 * np.mean([v != round(v) for v in preds.elements()]):.0f}%")
