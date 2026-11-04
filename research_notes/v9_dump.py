"""Run v9 (frame mix) against v7/v8 at the matched checkpoint (trunk step 5750, decayed branch) and SpatialStack: per-task, per-dataset,
paired bootstraps, agreement. Read-only; numpy on per-question files."""
import glob, json, os, sys
from collections import defaultdict
import numpy as np

ROOT = "/shared/workspace/povejmo/math/projects/mvp"
sys.path.insert(0, f"{ROOT}/src")
R = f"{ROOT}/results/train"
W = f"{ROOT}/.claude/worktrees/je-xattn-proposal/research_notes"
TASKS = ["object_counting", "object_abs_distance", "object_size_estimation", "room_size_estimation", "object_rel_distance", "object_rel_direction", "route_planning", "obj_appearance_order"]
GROUPS = {
    "v9_lora": ["2B_v9_fmix_lora_s1"], "v9_add": ["2B_v9_fmix_additive_s0", "2B_v9_fmix_additive_s1"], "v9_xa": ["2B_v9_fmix_xa3_both_s0", "2B_v9_fmix_xa3_both_s1"],
    "v7_lora": ["2B_v7_lora_s0", "2B_v7_lora_s1"], "v7_add": ["2B_v7_additive_s0", "2B_v7_additive_s1"], "v7_xa": ["2B_v7_xa3_both_s0", "2B_v7_xa3_both_s1"],
    "v7_hybrid": ["2B_v7_hybrid_s0", "2B_v7_hybrid_s1"], "v8_late": ["2B_v8_xa3_late_s0", "2B_v8_xa3_late_s1"], "v8_deep": ["2B_v8_xa3_deep_s0", "2B_v8_xa3_deep_s1"]}


def load(run, step="step5750"):
    return {r["id"]: r for r in map(json.loads, open(f"{R}/{run}/branches/{step}/eval/vsi_f32_full.jsonl"))}


data = {r: load(r) for g in GROUPS.values() for r in g}
ids = sorted(next(iter(data.values())))
assert all(sorted(v) == ids for v in data.values())
idx = {i: n for n, i in enumerate(ids)}
first = next(iter(data.values()))
task = np.array([first[i]["task"] for i in ids]); qtype = np.array([first[i]["question_type"] for i in ids]); ds = np.array([first[i]["dataset"] for i in ids])
print("record keys:", sorted(first[ids[0]].keys()))
S = {r: np.array([v[i]["score"] for i in ids], float) for r, v in data.items()}
GS = {g: np.mean([S[r] for r in rs], 0) for g, rs in GROUPS.items()}


def macro(s, mask=None):
    m = np.ones(len(s), bool) if mask is None else mask
    return 100 * np.mean([s[(task == t) & m].mean() for t in TASKS if ((task == t) & m).any()])


out = {"groups": GROUPS, "runs": {}, "agg": {}}
for r, s in S.items():
    out["runs"][r] = {"overall": round(macro(s), 2), **{t: round(100 * s[task == t].mean(), 1) for t in TASKS}}
for g, s in GS.items():
    out["agg"][g] = {"overall": round(macro(s), 2), **{t: round(100 * s[task == t].mean(), 1) for t in TASKS},
                     "by_ds": {d: round(macro(s, ds == d), 1) for d in sorted(set(ds))},
                     "dir_sub": {q.split("_")[-1]: round(100 * s[qtype == q].mean(), 1) for q in sorted(set(qtype)) if "rel_direction" in q}}
rng = np.random.default_rng(0)
strata = [np.where(task == t)[0] for t in TASKS]


def boot(a, b, n=2000, ts=None):
    d = GS[b] - GS[a] if isinstance(a, str) else b - a
    st = strata if ts is None else [np.where(task == ts)[0]]
    vals = [np.mean([d[rng.choice(x, len(x))].mean() for x in st]) * 100 for _ in range(n)]
    return round(float(np.mean([d[x].mean() for x in st]) * 100), 2), round(float(np.percentile(vals, 2.5)), 2), round(float(np.percentile(vals, 97.5)), 2)


pairs = [("v9_lora", "v9_add"), ("v9_lora", "v9_xa"), ("v9_add", "v9_xa"), ("v7_lora", "v7_add"), ("v7_add", "v7_xa"), ("v7_add", "v9_add"), ("v7_xa", "v9_xa"), ("v7_lora", "v9_lora")]
out["boot"] = {f"{b}-{a}": boot(a, b) for a, b in pairs}
out["boot_task"] = {f"{b}-{a}": {t: boot(a, b, 800, t) for t in TASKS} for a, b in [("v9_lora", "v9_add"), ("v9_add", "v9_xa"), ("v7_add", "v9_add"), ("v7_xa", "v9_xa"), ("v7_lora", "v9_lora")]}
# seed-to-seed on identical settings, as the noise floor
out["seed_gap"] = {g: round(abs(macro(S[rs[0]]) - macro(S[rs[1]])), 2) for g, rs in GROUPS.items() if len(rs) == 2}
out["seed_gap_task"] = {g: {t: round(abs(100 * (S[rs[0]][task == t].mean() - S[rs[1]][task == t].mean())), 1) for t in TASKS} for g, rs in GROUPS.items() if len(rs) == 2}

# answer overlap on multiple-choice, additive vs xattn (v9)
mc = np.isin(task, ["object_rel_distance", "object_rel_direction", "route_planning", "obj_appearance_order"])
A, X = GS["v9_add"], GS["v9_xa"]
L = S["2B_v9_fmix_lora_s1"]
out["mc_agree"] = {"n": int(mc.sum()), "both_right": int(((A == 1) & (X == 1) & mc).sum()), "both_wrong": int(((A == 0) & (X == 0) & mc).sum()), "add_only": int(((A == 1) & (X == 0) & mc).sum()), "xa_only": int(((A == 0) & (X == 1) & mc).sum()),
                   "seed_split": int((mc & ((A == 0.5) | (X == 0.5))).sum())}
out["lora_flip"] = {"geo_fixes": int(((A == 1) & (L == 0) & mc).sum()), "geo_breaks": int(((A == 0) & (L == 1) & mc).sum()), "n": int(mc.sum())}
# numeric bias: predicted / truth, per task, by arm
num = {}
for g in ("v9_lora", "v9_add", "v9_xa", "v7_add"):
    rs = GROUPS[g]
    for t in ["object_counting", "object_abs_distance", "object_size_estimation", "room_size_estimation"]:
        ratios = []
        for r in rs:
            for i in np.where(task == t)[0]:
                x = data[r][ids[i]]
                try:
                    p, a = float(str(x["prediction"]).split()[0]), float(x["ground_truth"])
                    if a > 0:
                        ratios.append(p / a)
                except Exception:
                    pass
        ratios = np.array(ratios)
        num.setdefault(g, {})[t] = {"median_ratio": round(float(np.median(ratios)), 3), "within10": round(float((np.abs(ratios - 1) < .1).mean()) * 100, 1), "within25": round(float((np.abs(ratios - 1) < .25).mean()) * 100, 1)}
out["numeric"] = num
# dev by task and overall from branches.json, trunk dev curve from log
out["dev"] = {}
for g, rs in GROUPS.items():
    row = []
    for r in rs:
        bj = json.load(open(f"{R}/{r}/branches.json"))["branches"]
        b = next(x for x in bj if x["trunk_step"] == 5750)
        row.append({"dev_acc": round(b["dev_acc"], 1), "dev_all": round(b.get("dev_acc_all", 0), 1), "dev_loss": round(b.get("dev_loss", 0), 3), "debiased": round(b.get("vsi_debiased", 0), 1), "before": round(b.get("dev_acc_before", 0), 1),
                    "tasks": {k[4:]: round(v, 1) for k, v in b.items() if k.startswith("dev/")}})
    out["dev"][g] = row
curves = {}
for g in ("v9_lora", "v9_add", "v9_xa", "v7_lora", "v7_add", "v7_xa"):
    r = GROUPS[g][0]
    rows = [json.loads(l) for l in open(f"{R}/{r}/log.jsonl")]
    curves[g] = [[x["step"], round(x["dev_acc"], 2)] for x in rows if "dev_acc" in x and x["step"] > 0]
out["curves"] = curves
# the v7 checkpoint sweep, as the noise floor over steps
out["v7_steps"] = {g: {st: round(np.mean([json.load(open(f"{R}/{r}/branches/{st}/eval/vsi_f32_full_summary.json"))["overall"] for r in GROUPS[g]]), 2)
                       for st in ("step5750", "step6750", "step7750", "step8750", "step9750", "step10750")} for g in ("v7_lora", "v7_add", "v7_xa", "v7_hybrid")}
# SpatialStack reproduction, same scoring check
from mvp.benchmarks.vsibench import fuzzy_matching, extract_number, mean_relative_accuracy
ss = glob.glob(f"{ROOT}/results/baselines/spatialstack/full_qwen35/vsibench/*/*samples_vsibench.jsonl")[0]
rows = [json.loads(l) for l in open(ss)]
print("spatialstack score keys:", {k: rows[0]["vsibench_score"][k] for k in rows[0]["vsibench_score"] if k not in ("question", "options")})
mc_row = next(r for r in rows if r["doc"]["options"] is not None)
print("spatialstack MC score keys:", {k: mc_row["vsibench_score"][k] for k in mc_row["vsibench_score"] if k not in ("question", "options")})
per_official, per_ours = defaultdict(list), defaultdict(list)
mapping = {"object_rel_direction_easy": "object_rel_direction", "object_rel_direction_medium": "object_rel_direction", "object_rel_direction_hard": "object_rel_direction"}
ssids = {}
for r in rows:
    sc = r["vsibench_score"]; t = mapping.get(sc["question_type"], sc["question_type"])
    pred = r["filtered_resps"][0] if isinstance(r["filtered_resps"], list) else r["filtered_resps"]
    pred = pred[0] if isinstance(pred, list) else pred
    if sc.get("options") is None or t in ("object_counting", "object_abs_distance", "object_size_estimation", "room_size_estimation"):
        p = extract_number(str(pred)); s = 0.0 if p is None else float(mean_relative_accuracy(p, float(sc["ground_truth"])))
    else:
        s = float(fuzzy_matching(str(pred)).upper() == str(sc["ground_truth"]).strip().upper())
    per_ours[t].append(s)
    ssids[sc["id"]] = (t, s)
    off = sc.get("MRA:.5:.95:.05", sc.get("accuracy"))
    per_official[t].append(float(off) if off is not None else np.nan)
out["spatialstack"] = {"ours_scoring": {t: round(100 * np.mean(per_ours[t]), 1) for t in TASKS}, "official_field": {t: round(100 * float(np.nanmean(per_official[t])), 1) for t in TASKS}}
out["spatialstack"]["ours_overall"] = round(np.mean([np.mean(per_ours[t]) for t in TASKS]) * 100, 2)
print("spatialstack:", out["spatialstack"], "lmms 67.53")
sv = np.array([ssids[i][1] if i in ssids else np.nan for i in ids])
print("id overlap with spatialstack", int(np.isfinite(sv).sum()))
if np.isfinite(sv).all():
    GS["ss"] = sv
    out["boot"]["v9_add-ss"] = boot("ss", "v9_add")
    out["boot"]["v9_xa-ss"] = boot("ss", "v9_xa")
    out["boot_task"]["v9_add-ss"] = {t: boot("ss", "v9_add", 800, t) for t in TASKS}
    out["agg"]["ss"] = {"overall": out["spatialstack"]["ours_overall"], **out["spatialstack"]["ours_scoring"], "by_ds": {d: round(macro(sv, ds == d), 1) for d in sorted(set(ds))},
                        "dir_sub": {q.split("_")[-1]: round(100 * sv[qtype == q].mean(), 1) for q in sorted(set(qtype)) if "rel_direction" in q}}
# zero-shot 2B on the full benchmark, if there is one
for f in glob.glob(f"{ROOT}/results/zeroshot/**/*2B*summary.json", recursive=True):
    j = json.load(open(f))
    if j.get("n") == 5130 and j.get("nframes") == 32:
        print("zeroshot", os.path.relpath(f, ROOT), round(j["overall"], 2))
# partial 128-frame progress
for g in ("v9_lora", "v9_add", "v9_xa"):
    for r in GROUPS[g]:
        p = f"{R}/{r}/branches/step5750/eval/vsi_f128_full.jsonl"
        print("f128", r, sum(1 for _ in open(p)) if os.path.exists(p) else None)
json.dump(out, open(f"{W}/v9_data.json", "w"))
for g, v in out["agg"].items():
    print(f"{g:10}", v["overall"], [v[t] for t in TASKS], v["by_ds"], v["dir_sub"])
print(json.dumps(out["boot"])); print(out["seed_gap"]); print(out["mc_agree"], out["lora_flip"]); print(json.dumps(out["numeric"])[:900]); print(json.dumps(out["dev"]["v9_add"])[:400]); print(out["v7_steps"]); print({k: v[-3:] for k, v in curves.items()})
print(json.dumps(out["boot_task"]["v9_add-v9_lora"])); print(json.dumps(out["boot_task"]["v9_xa-v9_add"])); print(json.dumps(out["boot_task"]["v9_add-v7_add"]))
