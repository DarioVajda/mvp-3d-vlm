#!/usr/bin/env bash
# Full VSI-Bench at 128 frames (inference only, as the 128-frame ablations of GeoThinker and DK-VLM) of the step-5750
# decay branches: v9 (8/16/32-frame training mix) LoRA, additive and xa3_both, and v7 additive (8-frame training), for
# how far test-time frames extrapolate and whether the frame mix helps it. One B300 / B200 per checkpoint, lkm account.
# XA at 128 frames relies on AlignedXAttn's query-chunked inference path (the dense [H, N, N] 3D bias would not fit).
# Resumable: vsibench skips questions already in the output file.
#   slurm/launch_eval_f128.sh [run ...]        -> results/train/<run>/branches/step5750/eval/vsi_f128_full.jsonl
set -euo pipefail
runs=("$@")
(( ${#runs[@]} )) || runs=(2B_v9_fmix_lora_s0 2B_v9_fmix_lora_s1 2B_v9_fmix_additive_s0 2B_v9_fmix_additive_s1
                           2B_v9_fmix_xa3_both_s0 2B_v9_fmix_xa3_both_s1 2B_v7_additive_s0 2B_v7_additive_s1)
for run in "${runs[@]}"; do
    d=results/train/$run/branches/step5750
    [[ -f $d/final.pt ]] || { echo "$run: no $d/final.pt, skipped" >&2; continue; }
    id=$(LOGDIR=eval slurm/submit.sh -A lkm --constraint 'GPU_BRD:B300|GPU_BRD:B200' -c 8 -t ${TIME:-06:00:00} \
        -J "vsi_f128_$run" -- python -m mvp.eval.vsibench --fusion-ckpt "$d/final.pt" --nframes 128 --split full \
        --out "$d/eval/vsi_f128_full.jsonl")
    echo "$run vsi_f128 $id"
done
