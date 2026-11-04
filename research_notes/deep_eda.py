"""Deep EDA on question-level VSI-Bench results: mode collapse, content biases, scene-level copying, label noise, seed instability, length effects."""
import glob, json, re
from collections import Counter, defaultdict
import numpy as np

ROOT = "/shared/workspace/povejmo/math/projects/mvp"
R = f"{ROOT}/results/train"
RUNS = {"lora0": "2B_v9_fmix_lora_s0", "lora1": "2B_v9_fmix_lora_s1", "add0": "2B_v9_fmix_additive_s0", "add1": "2B_v9_fmix_additive_s1", "xa0": "2B_v9_fmix_xa3_both_s0", "xa1": "2B_v9_fmix_xa3_both_s1"}
NUMT = ["object_counting", "object_abs_distance", "object_size_estimation", "room_size_estimation"]
D = {k: {x["id"]: x for x in map(json.loads, open(f"{R}/{v}/branches/step5750/eval/vsi_f32_full.jsonl"))} for k, v in RUNS.items()}
ss = {}
for r in map(json.loads, open(glob.glob(f"{ROOT}/results/baselines/spatialstack/full_qwen35/vsibench/*/*samples_vsibench.jsonl")[0])):
    s = r["vsibench_score"]; p = r["filtered_resps"]
    while isinstance(p, list): p = p[0]
    ss[s["id"]] = {"prediction": p, "score": s.get("accuracy", s.get("MRA:.5:.95:.05"))}
ids = sorted(D["add0"])
Q = D["add0"]
num = lambda s: (float(re.search(r"-?\d+(\.\d+)?", str(s)).group()) if re.search(r"\d", str(s)) else np.nan)
letter = lambda s: (str(s).strip()[0].upper() if str(s).strip() else "?")
opt_text = lambda x, L: next((o.split(".", 1)[1].strip() for o in x["options"] if o.strip()[0].upper() == L), None)
print("runs:", list(RUNS), "| questions", len(ids))

print("\n########## 1. NUMERIC: prediction distribution vs truth distribution (mode collapse?) ##########")
for t in NUMT:
    xs = [Q[i] for i in ids if Q[i]["task"] == t]
    truth = [num(x["ground_truth"]) for x in xs]
    for k in ("lora1", "add0", "xa0"):
        preds = [num(D[k][x["id"]]["prediction"]) for x in xs]
        pc, tc = Counter(preds), Counter(truth)
        top = pc.most_common(3)
        print(f"{t:24} {k:5} distinct pred values {len(pc):4} vs truth {len(tc):4} | top preds {[(v, n, f'truth share {100 * tc[v] / len(xs):.1f}%') for v, n in top]} | pred std {np.nanstd(preds):.2f} truth std {np.std(truth):.2f}")
print("\n--- counting: accuracy by true count, and what the model says when truth is large ---")
xs = [Q[i] for i in ids if Q[i]["task"] == "object_counting"]
for k in ("lora1", "add0"):
    by = defaultdict(list); said = defaultdict(list)
    for x in xs:
        p = D[k][x["id"]]; by[min(int(num(x["ground_truth"])), 7)].append(p["score"]); said[min(int(num(x["ground_truth"])), 7)].append(num(p["prediction"]))
    print(k, {c: f"MRA {np.mean(v):.2f} n{len(v)} median said {np.median(said[c]):.0f}" for c, v in sorted(by.items())})
print("\n--- abs distance: relative error histogram and worst cases ---")
xs = [Q[i] for i in ids if Q[i]["task"] == "object_abs_distance"]
for k in ("lora1", "add0"):
    ratio = np.array([num(D[k][x["id"]]["prediction"]) / num(x["ground_truth"]) for x in xs])
    print(k, "ratio percentiles 5/25/50/75/95:", np.round(np.percentile(ratio, [5, 25, 50, 75, 95]), 2), "| >3x off:", int((ratio > 3).sum()), "| <1/3:", int((ratio < 1 / 3).sum()))
worst = sorted(xs, key=lambda x: -abs(np.log(num(D["add0"][x["id"]]["prediction"]) / num(x["ground_truth"]))))[:6]
for x in worst:
    print("   worst:", x["id"], x["dataset"], f"truth {x['ground_truth']} add0 {D['add0'][x['id']]['prediction']} lora1 {D['lora1'][x['id']]['prediction']} ss {ss[x['id']]['prediction']} |", x["question"][55:130])
print("\n--- room size: one question per scene; are predictions scene-specific? ---")
xs = [Q[i] for i in ids if Q[i]["task"] == "room_size_estimation"]
for k in ("lora1", "add0", "xa0"):
    preds = [num(D[k][x["id"]]["prediction"]) for x in xs]; truth = [num(x["ground_truth"]) for x in xs]
    print(k, f"corr(log pred, log truth) {np.corrcoef(np.log(preds), np.log(truth))[0, 1]:.2f} | distinct values {len(set(preds))}/288 | most common {Counter(preds).most_common(3)} | truth range {min(truth):.0f}-{max(truth):.0f} pred range {min(preds):.0f}-{max(preds):.0f}")

print("\n########## 2. MULTIPLE CHOICE: content biases (not letters) ##########")
print("--- relative direction easy (2 options left/right): what the model says vs truth ---")
for k in ("lora1", "add0", "xa0"):
    xs = [Q[i] for i in ids if Q[i]["question_type"] == "object_rel_direction_easy"]
    pc = Counter(opt_text(x, letter(D[k][x["id"]]["prediction"])) for x in xs); tc = Counter(opt_text(x, letter(x["ground_truth"])) for x in xs)
    print(k, "said", dict(pc), "truth", dict(tc))
print("--- relative direction medium/hard: content said vs truth ---")
for qt in ("object_rel_direction_medium", "object_rel_direction_hard"):
    xs = [Q[i] for i in ids if Q[i]["question_type"] == qt]
    tc = Counter(opt_text(x, letter(x["ground_truth"])) for x in xs)
    for k in ("lora1", "add0"):
        pc = Counter(opt_text(x, letter(D[k][x["id"]]["prediction"])) for x in xs)
        acc = {c: np.mean([D[k][x["id"]]["score"] for x in xs if opt_text(x, letter(x["ground_truth"])) == c]) for c in tc}
        print(f"{qt[21:]:7} {k:5} said {dict(pc)} | truth {dict(tc)} | accuracy by true answer {dict((c, round(v, 2)) for c, v in acc.items())}")
print("--- relative distance: does the model prefer the first-listed candidate? position of the chosen object among the 4 listed in the question ---")
xs = [Q[i] for i in ids if Q[i]["task"] == "object_rel_distance"]
for k in ("lora1", "add0", "xa0"):
    pos_pred, pos_true = Counter(), Counter()
    for x in xs:
        m = re.search(r"\((.+?)\) is the closest", x["question"]); cands = [c.strip().lower() for c in m[1].split(",")]
        pt = opt_text(x, letter(D[k][x["id"]]["prediction"])); tt = opt_text(x, letter(x["ground_truth"]))
        pos_pred[cands.index(pt.lower()) if pt and pt.lower() in cands else -1] += 1; pos_true[cands.index(tt.lower()) if tt and tt.lower() in cands else -1] += 1
    print(k, "chosen position in question list", dict(sorted(pos_pred.items())), "| truth position", dict(sorted(pos_true.items())))
print("--- relative distance: is the option order the same as the question's list order? (if yes, letter and list position are confounded) ---")
same = sum(all(opt_text(x, L).lower() == c.strip().lower() for L, c in zip("ABCD", re.search(r"\((.+?)\) is the closest", x["question"])[1].split(","))) for x in xs)
print(f"options in question order: {same}/{len(xs)}")
print("--- appearance order: does the model pick the option that starts with the first category named in the question? ---")
xs = [Q[i] for i in ids if Q[i]["task"] == "obj_appearance_order"]
for k in ("lora1", "add0", "xa0"):
    first_match, truth_first = 0, 0
    for x in xs:
        cats = [c.strip().lower() for c in x["question"].split("video:")[1].strip(" ?").split(",")]
        pt = opt_text(x, letter(D[k][x["id"]]["prediction"])); tt = opt_text(x, letter(x["ground_truth"]))
        first_match += pt is not None and pt.lower().split(",")[0].strip() == cats[0]; truth_first += tt.lower().split(",")[0].strip() == cats[0]
    print(k, f"chose option starting with first-named category: {first_match}/{len(xs)}; truth does: {truth_first}/{len(xs)}")
print("--- route planning: content of chosen vs true first turn ---")
xs = [Q[i] for i in ids if Q[i]["task"] == "route_planning"]
for k in ("lora1", "add0", "xa0"):
    said = Counter(); tru = Counter()
    for x in xs:
        pt = opt_text(x, letter(D[k][x["id"]]["prediction"])); tt = opt_text(x, letter(x["ground_truth"]))
        if pt: said[pt.split(",")[0].strip()] += 1
        tru[tt.split(",")[0].strip()] += 1
    print(k, "first turn said", dict(said), "| truth", dict(tru))

print("\n########## 3. SCENE-LEVEL: does the model give the same answer to different questions on one video? ##########")
for t in ("object_counting", "object_abs_distance", "object_size_estimation"):
    by_scene = defaultdict(list)
    for i in ids:
        if Q[i]["task"] == t: by_scene[Q[i]["scene_name"]].append(i)
    for k in ("lora1", "add0"):
        within = []; within_t = []
        for s, qs in by_scene.items():
            if len(qs) >= 3:
                p = [num(D[k][i]["prediction"]) for i in qs]; g = [num(Q[i]["ground_truth"]) for i in qs]
                within.append(len(set(p)) / len(qs)); within_t.append(len(set(g)) / len(qs))
        print(f"{t:24} {k:5} distinct answers per scene (share): pred {np.mean(within):.2f} truth {np.mean(within_t):.2f}  (scenes with >=3 q: {len(within)})")
print("--- scenes where every model fails most questions (possible bad video or labels) ---")
scene_acc = defaultdict(list)
for i in ids:
    scene_acc[(Q[i]["scene_name"], Q[i]["dataset"])].append(np.mean([D[k][i]["score"] for k in RUNS] + [float(ss[i]["score"])]))
bad = sorted(((np.mean(v), len(v), s) for s, v in scene_acc.items() if len(v) >= 8))[:8]
print("lowest mean score across all 7 models, scenes with >=8 questions:", [(round(a, 2), n, s[0], s[1]) for a, n, s in bad])
print("highest:", [(round(a, 2), n, s[0]) for a, n, s in sorted(((np.mean(v), len(v), s) for s, v in scene_acc.items() if len(v) >= 8), reverse=True)[:4]])

print("\n########## 4. LABEL NOISE candidates: all 7 models agree on the same wrong MC answer ##########")
mc = [i for i in ids if Q[i]["task"] not in NUMT]
agree_wrong = [i for i in mc if len({letter(D[k][i]["prediction"]) for k in RUNS} | {letter(ss[i]["prediction"])}) == 1 and D["add0"][i]["score"] == 0]
print(f"{len(agree_wrong)} of {len(mc)} MC questions: all 7 models give the same wrong letter. By task:", Counter(Q[i]["task"] for i in agree_wrong))
for i in agree_wrong[:5]:
    x = Q[i]; print("   ", i, x["question_type"], "truth", x["ground_truth"], opt_text(x, letter(x["ground_truth"])), "| all said", letter(D["add0"][i]["prediction"]), opt_text(x, letter(D["add0"][i]["prediction"])), "|", x["question"][:110].replace("\n", " "))
print("--- numeric: all 7 models off by >2x in the same direction ---")
for t in NUMT:
    xs = [Q[i] for i in ids if Q[i]["task"] == t]; n_hi = n_lo = 0
    for x in xs:
        rs = [num(D[k][x["id"]]["prediction"]) / num(x["ground_truth"]) for k in RUNS] + [num(ss[x["id"]]["prediction"]) / num(x["ground_truth"])]
        n_hi += all(r > 2 for r in rs); n_lo += all(r < 0.5 for r in rs)
    print(f"{t:24} all >2x too high: {n_hi}  all <0.5x too low: {n_lo}  of {len(xs)}")

print("\n########## 5. SEED INSTABILITY: how often do two seeds of one arm disagree? ##########")
for arm, (a, b) in {"lora": ("lora0", "lora1"), "add": ("add0", "add1"), "xa": ("xa0", "xa1")}.items():
    row = {}
    for t in sorted({Q[i]["task"] for i in ids}):
        xs = [i for i in ids if Q[i]["task"] == t]
        if t in NUMT:
            row[t[:12]] = f"same number {100 * np.mean([num(D[a][i]['prediction']) == num(D[b][i]['prediction']) for i in xs]):.0f}%"
        else:
            row[t[:12]] = f"same letter {100 * np.mean([letter(D[a][i]['prediction']) == letter(D[b][i]['prediction']) for i in xs]):.0f}%"
    print(arm, row)
print("--- questions every model gets right / wrong (difficulty is shared across models?) ---")
allc = np.array([[D[k][i]["score"] for k in RUNS] + [float(ss[i]["score"])] for i in mc])
print(f"MC: all 7 right {int((allc.min(1) == 1).sum())}, all 7 wrong {int((allc.max(1) == 0).sum())}, mixed {int(((allc.min(1) < 1) & (allc.max(1) > 0)).sum())} of {len(mc)}; expected 'all wrong' if independent at mean acc: {int(len(mc) * np.prod(1 - allc.mean(0))):d}")

print("\n########## 6. LENGTH EFFECTS ##########")
for t in sorted({Q[i]["task"] for i in ids}):
    xs = [Q[i] for i in ids if Q[i]["task"] == t]
    nf = np.array([max(x["frame_indices"]) + 1 for x in xs]); sc = np.array([np.mean([D[k][x["id"]]["score"] for k in ("add0", "add1")]) for x in xs])
    lo, hi = nf <= np.median(nf), nf > np.median(nf)
    print(f"{t:24} video length median {np.median(nf):.0f} frames | additive score short videos {sc[lo].mean():.2f} vs long {sc[hi].mean():.2f} | corr(score, log length) {np.corrcoef(sc, np.log(nf))[0, 1]:+.2f}")
print("--- frame height/width (portrait vs landscape) ---")
hw = Counter((tuple(Q[i]["frame_hw"]), Q[i]["dataset"]) for i in ids)
print(hw.most_common(6))
for shape in {tuple(Q[i]["frame_hw"]) for i in ids}:
    xs = [i for i in ids if tuple(Q[i]["frame_hw"]) == shape]
    print(shape, "n", len(xs), "additive macro-ish mean score", round(float(np.mean([D["add0"][i]["score"] for i in xs])), 3), "lora", round(float(np.mean([D["lora1"][i]["score"] for i in xs])), 3))

print("\n########## 7. INVALID / ODD OUTPUTS ##########")
for k in RUNS:
    odd = [i for i in ids if (Q[i]["task"] not in NUMT and letter(D[k][i]["prediction"]) not in "ABCD") or (Q[i]["task"] in NUMT and np.isnan(num(D[k][i]["prediction"])))]
    long_ = [i for i in ids if len(str(D[k][i]["prediction"])) > 6]
    print(k, "unparseable", len(odd), "| answers longer than 6 chars", len(long_), [str(D[k][i]["prediction"])[:30] for i in long_[:3]])
