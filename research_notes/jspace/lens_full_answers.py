"""Full stored evaluation answers (decayed step-10750 branches) for the ten J-lens questions, plus their question text."""
import json
ROOT = "/shared/workspace/povejmo/math/projects/mvp"
qs = json.load(open(f"{ROOT}/results/jspace_v7_undecayed_trunk/questions.json")); ids = [q[0] for q in qs]
for run in ("2B_v7_lora_s0", "2B_v7_additive_s0", "2B_v7_xa3_both_s0"):
    d = {x["id"]: x for x in map(json.loads, open(f"{ROOT}/results/train/{run}/branches/step10750/eval/vsi_f32_full.jsonl"))}
    print(run, {i: (d[i]["ground_truth"], d[i]["prediction"], d[i]["score"]) for i in ids})
d = {x["id"]: x for x in map(json.loads, open(f"{ROOT}/results/train/2B_v7_additive_s0/branches/step10750/eval/vsi_f32_full.jsonl"))}
for i in ids:
    print(i, d[i]["question_type"], "|", d[i]["question"].replace("\n", " ")[:170], "| options", d[i]["options"])
