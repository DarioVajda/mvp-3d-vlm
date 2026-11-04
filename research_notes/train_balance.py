"""Training-set (VSI-590K) class balance for relative direction answers and route planning presence."""
import json, re
from collections import Counter
ROOT = "/shared/workspace/povejmo/math/projects/mvp"
qt = Counter(); dir_ans = Counter(); n_back_opts = Counter(); route = 0
for l in open(f"{ROOT}/data/raw/VSI-590K/vsi_590k.jsonl"):
    r = json.loads(l); qt[r.get("question_type")] += 1
    q = r["conversations"][0]["value"]
    if "Turn Back" in q or "Turn Left" in q: route += 1
    if r.get("question_type") == "relative_direction_object":
        a = r["conversations"][1]["value"].strip()
        m = re.search(rf"\n{a}\. (.+)", q)
        if m: dir_ans[m[1].strip()] += 1
        n_back_opts[sum("back" in o for o in re.findall(r"\n[A-D]\. (.+)", q))] += 1
print("question types:", qt.most_common())
print("route-like prompts (Turn Left/Back in the question):", route)
tot = sum(dir_ans.values())
print("relative direction answer classes in training:", {k: f"{v} ({100 * v / tot:.1f}%)" for k, v in dir_ans.most_common()})
print("number of options containing 'back' per training direction question:", dict(n_back_opts))
