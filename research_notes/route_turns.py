"""Route planning: turns per correct option (prefix-stripped regex), accuracy by number of turns, across v9 arms."""
import json, re
from collections import Counter

R = "/shared/workspace/povejmo/math/projects/mvp/results/train"
TURN = {"left": "L", "right": "R", "back": "B", "around": "B"}
turns = lambda opt: [TURN[t] for t in re.findall(r"turn (left|right|back|around)", opt.lower())]
for run in ("2B_v9_fmix_lora_s1", "2B_v9_fmix_additive_s0", "2B_v9_fmix_additive_s1", "2B_v9_fmix_xa3_both_s0", "2B_v9_fmix_xa3_both_s1"):
    rows = [x for x in map(json.loads, open(f"{R}/{run}/branches/step5750/eval/vsi_f32_full.jsonl")) if x["task"] == "route_planning"]
    acc, tot, perturn, mirror, wrong = Counter(), Counter(), Counter(), 0, 0
    for x in rows:
        L = lambda s: str(s).strip()[0].upper()
        opts = {L(o): turns(o) for o in x["options"]}
        g = opts[L(x["ground_truth"])]; k = len(g)
        tot[k] += 1; acc[k] += x["score"]
        p = opts.get(L(x["prediction"])) if str(x["prediction"]).strip() else None
        if p is not None and len(p) == len(g):
            for a, b in zip(g, p):
                perturn[a == b] += 1
            if p != g:
                wrong += 1
                if all({a, b} == {"L", "R"} for a, b in zip(g, p) if a != b) and all(a == b or {a, b} == {"L", "R"} for a, b in zip(g, p)):
                    mirror += 1
    print(f"{run:26} acc by #turns {dict((k, (round(acc[k] / tot[k], 2), tot[k])) for k in sorted(tot))}  per-turn acc {perturn[True] / (perturn[True] + perturn[False]):.2f}  exact L/R mirror among wrong {mirror}/{wrong}")
rows = [x for x in map(json.loads, open(f"{R}/2B_v9_fmix_additive_s0/branches/step5750/eval/vsi_f32_full.jsonl")) if x["task"] == "route_planning"]
print("turns in correct option:", Counter(len(turns([o for o in x["options"] if o.strip()[0].upper() == str(x["ground_truth"]).strip()[0].upper()][0])) for x in rows))
print("options count:", Counter(len(x["options"]) for x in rows), "chance ≈", round(sum(1 / len(x["options"]) for x in rows) / len(rows), 3))
