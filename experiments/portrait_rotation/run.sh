#!/bin/bash
# Upright vs rotated (Spatial-MLLM) VGGT inputs on the 57 portrait videos: frame picks, then Qwen3-VL-4B accuracy.
#   slurm/submit.sh -J portrait_cmp -t 01:00:00 -- bash experiments/portrait_rotation/run.sh
set -e
cd "$(dirname "$0")/../.."
OUT=results/zeroshot/vsibench/portrait
SCENES=$(sed 's#.*/##' data/vggt_portrait_rotated/videos.txt | tr '\n' ' ')
.venv/bin/python - <<'PY'
import json
from mvp.paths import DATA, ZEROSHOT
from mvp.reconstruction.sa_sampling import space_aware_sampling
out = {}
for v in open(DATA / "vggt_portrait_rotated/videos.txt").read().split():
    up = space_aware_sampling(DATA / "vggt" / v)["candidates"]
    rot = space_aware_sampling(DATA / "vggt_portrait_rotated" / v)["candidates"]
    out[v] = {"upright": up, "rotated": rot, "shared": len(set(up) & set(rot))}
    print(v, len(up), len(rot), "shared", out[v]["shared"], flush=True)
(ZEROSHOT / "vsibench/portrait").mkdir(parents=True, exist_ok=True)
(ZEROSHOT / "vsibench/portrait/sa_picks.json").write_text(json.dumps(out, indent=1))
PY
M=Qwen/Qwen3-VL-4B-Instruct
.venv/bin/python -m mvp.eval.vsibench --model $M --nframes 16 --sampling sa --scenes $SCENES --out $OUT/4B_f16_sa_upright.jsonl
.venv/bin/python -m mvp.eval.vsibench --model $M --nframes 16 --sampling sa --scenes $SCENES --vggt-root data/vggt_portrait_rotated --out $OUT/4B_f16_sa_rotated.jsonl
.venv/bin/python -m mvp.eval.vsibench --model $M --nframes 16 --sampling uniform128 --scenes $SCENES --out $OUT/4B_f16_uniform128.jsonl
