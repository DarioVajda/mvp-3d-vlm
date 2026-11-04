"""Print per-task VSI numbers of the local SpatialStack reproduction and the zero-shot Qwen3.5 runs (read-only)."""
import glob
import json
import os

R = "/shared/workspace/povejmo/math/projects/mvp/results"
paths = sorted(glob.glob(f"{R}/baselines/spatialstack/full_qwen35/**/*", recursive=True))
print("files under full_qwen35:", [os.path.relpath(p, R) for p in paths][:30])
for p in paths + sorted(glob.glob(f"{R}/zeroshot/**/*4B*", recursive=True)) + sorted(glob.glob(f"{R}/zeroshot/**/*summary*", recursive=True)):
    if not os.path.isfile(p) or not p.endswith(".json"):
        continue
    try:
        d = json.load(open(p))
    except Exception:
        continue
    def show(tag, obj):
        if isinstance(obj, dict) and any(k in obj for k in ("overall", "object_counting", "vsibench")):
            print("==", tag)
            for k, v in obj.items():
                if isinstance(v, (int, float)):
                    print(f"   {k}: {v:.2f}" if isinstance(v, float) else f"   {k}: {v}")
                elif isinstance(v, dict):
                    show(f"{tag} / {k}", v)
    show(os.path.relpath(p, R), d)
