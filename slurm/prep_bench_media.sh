#!/usr/bin/env bash
# Unpack the videos of VSTI-Bench (Journey9ni/vstibench, ScanNet val) and Video-MME (lmms-eval/Video-MME) from the
# Hugging Face cache into data/bench_media/, as mvp.benchmarks.{vstibench,videomme} expect. Host-side (the container
# has no pigz), CPU only:
#   sbatch -p frida -A lkm -c 8 --mem 8G -t 04:00:00 -J prep_bench_media -o logs/%x_%j.out slurm/prep_bench_media.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"  # sbatch runs a spool copy of this script
hub=/shared/workspace/povejmo/huggingface_cache/hub
out=data/bench_media
mkdir -p "$out/vstibench" "$out/videomme"
tar -I pigz -xf "$hub"/datasets--Journey9ni--vstibench/snapshots/*/ScanNet.tar.gz -C "$out/vstibench" --no-same-owner
for z in "$hub"/datasets--lmms-eval--Video-MME/snapshots/*/videos_chunked_*.zip; do
    unzip -q -n "$z" -d "$out/videomme"
done
echo "vstibench: $(find "$out/vstibench" -name '*.mp4' | wc -l) videos; videomme: $(ls "$out/videomme/data" | wc -l) videos"
