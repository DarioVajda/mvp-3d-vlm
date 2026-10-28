#!/usr/bin/env bash
# Run v4 (2026-10-28): VLM-3R ScanNet (cap 6k) + all VSI-590K appearance-order questions, 5 epochs, dev accuracy and a
# checkpoint every half epoch, best-dev checkpoint (six ScanNet tasks) on full VSI-Bench at 32 frames.
# Seven single-seed runs on Qwen3.5-2B; LoRA r64 at 1e-4 everywhere, new-module lr 1e-5 or 3e-5.
set -euo pipefail
common=(--epochs 5 --cap 6000 --appearance --evals-per-epoch 2 --lr 1e-4 --weight-decay 0.01)
export TRAIN_SBATCH="-t 16:00:00 --constraint GPU_BRD:B300|GPU_BRD:B200"
jobs=results/train/jobs_2B_v4.txt
: > "$jobs"
slurm/train_eval.sh lora     2B_v4_lora              "${common[@]}"                                  | tee -a "$jobs"
for nm in 1e-5 3e-5; do
    slurm/train_eval.sh additive 2B_v4_additive_nm$nm    "${common[@]}" --fusion-lr $nm                        | tee -a "$jobs"
    slurm/train_eval.sh xattn    2B_v4_xattn2L_nm$nm     "${common[@]}" --fusion-lr $nm --resampler-layers 2   | tee -a "$jobs"
    slurm/train_eval.sh xattn    2B_v4_xattn8L_nm$nm     "${common[@]}" --fusion-lr $nm --resampler-layers 8   | tee -a "$jobs"
done
