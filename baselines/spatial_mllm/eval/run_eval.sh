#!/usr/bin/env bash
# Run Spatial-MLLM's VSI-Bench evaluation exactly as scripts/evaluation/evaluate_vsibench_spatial_mllm{,_w_sa_sampling}.sh
# do: same python command and arguments, from the repo root, with the checkpoint as its pinned local snapshot.
#   run_eval.sh <135K|820K> <uniform|sa> <output dir> [<shard index> <number of shards>]
# With a shard, the annotation directory is a copy of the dataset card's "full" config (test*.parquet, in the
# order datasets loads it) cut to one contiguous slice, so separate jobs cover the benchmark exactly once;
# merge_shards.py joins their results and scores them with the authors' calculate_metrics.
# The repo's expected layout datasets/evaluation/vsibench{,/sa_sampling_16f} is made of symlinks to our copy of
# VSI-Bench (revision a9fde418 test.jsonl, plus the dataset card's parquet configs) and to the authors' pre-sampled
# frames (Diankun/Spatial-MLLM-Data, evaluation/vsibench/sa_sampling_16f.zip).
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)  # the method directory
MVP=$(cd "$HERE/../.." && pwd)
REPO=$MVP/third_party/Spatial-MLLM
SNAP=$HOME/.cache/huggingface/hub
case $1 in
  135K) CKPT=$SNAP/models--Diankun--Spatial-MLLM-v1.1-Instruct-135K/snapshots/635b63733ed24f76b2f1a533dd5bc32836ea6c53 ;;
  820K) CKPT=$SNAP/models--Diankun--Spatial-MLLM-v1.1-Instruct-820K/snapshots/89ac646b88effaa651d304a308bf6e543c305f0c ;;
  *) echo "unknown model $1" >&2; exit 1 ;;
esac
DS=$REPO/datasets/evaluation/vsibench
mkdir -p "$DS"
for f in "$MVP"/data/raw/VSI-Bench/*; do [[ $f == *.zip ]] || ln -sfn "$f" "$DS/"; done
ln -sfn "$MVP/data/raw/spatial_mllm_sa/sa_sampling_16f" "$DS/sa_sampling_16f"
case $2 in
  uniform) VIDEO_DIR=datasets/evaluation/vsibench ;;
  sa) VIDEO_DIR=datasets/evaluation/vsibench/sa_sampling_16f ;;
  *) echo "unknown sampling $2" >&2; exit 1 ;;
esac
OUT=$3
mkdir -p "$OUT"
ANN=$DS
if [[ $# -ge 5 ]]; then
  ANN=$OUT/annotations
  mkdir -p "$ANN"
  printf -- '---\nconfigs:\n  - config_name: full\n    data_files:\n      - split: test\n        path: "test*.parquet"\n---\n' > "$ANN/README.md"
  # Every shard directory has the same name, so the datasets cache would hand all shards the first one's rows;
  # give each its own cache and check that the authors' load_dataset call returns exactly this shard.
  export HF_DATASETS_CACHE=$OUT/hf_datasets_cache
  "$HERE/.venv/bin/python" -c "
import sys, glob, numpy as np, pandas as pd
from datasets import load_dataset
df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('$DS/test*.parquet'))], ignore_index=True)
part = np.array_split(np.arange(len(df)), $5)[$4]
df.iloc[part].to_parquet('$ANN/test_shard.parquet', index=False)
loaded = load_dataset('$ANN', 'full')['test']['id']
assert list(loaded) == df['id'].iloc[part].tolist(), 'load_dataset does not return this shard'
print(f'shard $4/$5: rows {part[0]}-{part[-1]} of {len(df)}, ids {loaded[0]}..{loaded[-1]} verified')"
fi
{ echo "date: $(date -u +%FT%TZ)"; echo "host: $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c
  echo "repo: $(git -C "$REPO" rev-parse HEAD)"; echo "checkpoint: $CKPT"; echo "video_dir: $VIDEO_DIR"
  echo "annotation_dir: $ANN"; echo "shard: ${4:-} ${5:-}"; } > "$OUT/manifest.txt"
cp "$HERE/env/freeze.txt" "$OUT/"
cd "$REPO"
"$HERE/.venv/bin/python" src/evaluation/vsibench/eval_vsibench.py \
    --model_path "$CKPT" \
    --model_type spatial-mllm \
    --nframes 16 \
    --annotation_dir "$ANN" \
    --video_dir "$VIDEO_DIR" \
    --batch_size 1 \
    --output_dir "$OUT" \
    --output_name eval_result 2>&1 | tee -a "$OUT/run.log"
