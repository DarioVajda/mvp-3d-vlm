"""Are left/right mirror errors tied to portrait ARKitScenes videos (rotation handling) or to particular scenes? Includes the SpatialStack baseline, which has its own frame pipeline."""
import glob, json, re
from collections import Counter, defaultdict
import numpy as np

ROOT = "/shared/workspace/povejmo/math/projects/mvp"
R = f"{ROOT}/results/train"
RUNS = {"lora1": "2B_v9_fmix_lora_s1", "add0": "2B_v9_fmix_additive_s0", "xa0": "2B_v9_fmix_xa3_both_s0"}
D = {k: {x["id"]: x for x in map(json.loads, open(f"{R}/{v}/branches/step5750/eval/vsi_f32_full.jsonl"))} for k, v in RUNS.items()}
ss = {}
for r in map(json.loads, open(glob.glob(f"{ROOT}/results/baselines/spatialstack/full_qwen35/vsibench/*/*samples_vsibench.jsonl")[0])):
    p = r["filtered_resps"]
    while isinstance(p, list): p = p[0]
    ss[r["vsibench_score"]["id"]] = p
Q = D["add0"]; letter = lambda s: str(s).strip()[0].upper()
opt_text = lambda x, L: next((o.split(".", 1)[1].strip() for o in x["options"] if o.strip()[0].upper() == L), None)
MIRROR = {"left": "right", "right": "left", "front-left": "front-right", "front-right": "front-left", "back-left": "back-right", "back-right": "back-left"}
xs = [x for x in Q.values() if x["task"] == "object_rel_direction"]
print("relative direction by dataset and orientation (accuracy / share of errors that are exact mirrors):")
for key in sorted({(x["dataset"], tuple(x["frame_hw"])) for x in xs}):
    sel = [x for x in xs if (x["dataset"], tuple(x["frame_hw"])) == key]
    row = f"{key[0]:12} {str(key[1]):10} n{len(sel):4}"
    for k in list(RUNS) + ["ss"]:
        pred = lambda x: letter(ss[x["id"]]) if k == "ss" else letter(D[k][x["id"]]["prediction"])
        acc = np.mean([pred(x) == letter(x["ground_truth"]) for x in sel])
        wrong = [x for x in sel if pred(x) != letter(x["ground_truth"])]
        mir = np.mean([opt_text(x, pred(x)) == MIRROR.get(opt_text(x, letter(x["ground_truth"]))) for x in wrong]) if wrong else float("nan")
        row += f" | {k} {100 * acc:.0f}% mirror {100 * mir:.0f}%"
    print(row)
print("\nscenes where the 3 models + SpatialStack ALL mirror on >= 4 direction questions:")
per_scene = defaultdict(list)
for x in xs:
    allmir = all(opt_text(x, letter(D[k][x["id"]]["prediction"])) == MIRROR.get(opt_text(x, letter(x["ground_truth"]))) for k in RUNS) and opt_text(x, letter(ss[x["id"]])) == MIRROR.get(opt_text(x, letter(x["ground_truth"])))
    per_scene[(x["scene_name"], x["dataset"], tuple(x["frame_hw"]))].append(allmir)
bad = [(s, sum(v), len(v)) for s, v in per_scene.items() if sum(v) >= 4]
for s, n, tot in sorted(bad, key=lambda t: -t[1]):
    print(f"  {s[0]:14} {s[1]:12} {s[2]} all-four-mirror on {n}/{tot} questions")
print("total direction questions where all four mirror:", sum(sum(v) for v in per_scene.values()), "of", len(xs), "| scenes with any:", sum(any(v) for v in per_scene.values()), "of", len(per_scene))
print("\nobject pairs in those all-mirror questions (standing-by object, facing object):")
pairs = Counter()
for x in xs:
    m = re.search(r"standing by the (.+?) and facing the (.+?),", x["question"])
    if m and all(opt_text(x, letter(D[k][x["id"]]["prediction"])) == MIRROR.get(opt_text(x, letter(x["ground_truth"]))) for k in RUNS):
        pairs[m[1]] += 1
print("  standing-by object counts:", pairs.most_common(10))
print("  baseline: standing-by object counts over all direction questions:", Counter(re.search(r"standing by the (.+?) and", x["question"])[1] for x in xs if re.search(r"standing by the (.+?) and", x["question"])).most_common(10))
