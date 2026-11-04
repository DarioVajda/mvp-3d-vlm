"""Do any training videos of v9 appear in VSI-Bench? Compares file stems. Also prints v9 vs v7 args and the 2B zero-shot per-task row."""
import json, os, re, sys
from collections import Counter

ROOT = "/shared/workspace/povejmo/math/projects/mvp"
sys.path.insert(0, f"{ROOT}/src")
os.chdir(ROOT)
from mvp.benchmarks.vsibench import load_vsibench
from mvp.train.data import VLM3R, VSI590K

R = f"{ROOT}/results/train"
a9, a7 = (json.load(open(f"{R}/{n}/args.json")) for n in ("2B_v9_fmix_additive_s0", "2B_v7_additive_s0"))
drop = ("out", "wandb_name", "git_commit", "host", "gpu")
print("v9 vs v7 differing args:", {k: (a7.get(k), a9.get(k)) for k in sorted(set(a7) | set(a9)) if a7.get(k) != a9.get(k) and k not in drop})
print("v9 data args:", {k: v for k, v in a9.items() if any(s in k for s in ("cap", "appear", "vsi", "590", "frame", "dev", "ao_", "epoch", "seed"))})
df = load_vsibench("full")
stem = lambda p: os.path.splitext(os.path.basename(str(p)))[0]
test = {stem(p) for p in df["video_path"]}
print("VSI videos", len(test), Counter(df.drop_duplicates("video_path")["dataset"]))
vl = {stem(r["video"]) for r in json.load(open(VLM3R))}
v590 = Counter()
stems590 = {}
for r in map(json.loads, open(VSI590K)):
    if "video" in r:
        s = stem(r["video"])
        stems590[s] = r["video"].split("/")[0] if "/" in r["video"] else "?"
        v590[r["question_type"]] += 1
print("VLM-3R train videos", len(vl), "overlap with VSI", len(vl & test), sorted(vl & test)[:5])
print("VSI-590K video stems", len(stems590), "overlap with VSI", len(set(stems590) & test), sorted(set(stems590) & test)[:5], Counter(stems590.values()).most_common(6))
print("VSI-590K question types", v590.most_common(12))
# scene-level: scannet scene ids without the scan suffix
base = lambda s: re.sub(r"_\d+$", "", s)
print("scene-level overlap (scannet sceneXXXX, any scan): VLM-3R", len({base(s) for s in vl if s.startswith("scene")} & {base(s) for s in test if s.startswith("scene")}),
      "VSI-590K", len({base(s) for s in stems590 if s.startswith("scene")} & {base(s) for s in test if s.startswith("scene")}))
j = json.load(open(f"{ROOT}/results/zeroshot/vsibench/qwen_sampling/2B_f32_summary.json"))
print("zero-shot 2B:", {k: (round(v, 1) if isinstance(v, float) else v) for k, v in j.items() if k not in ("system",)})
