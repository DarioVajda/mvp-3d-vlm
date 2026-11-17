#!/usr/bin/env bash
# The explainability runs behind explainability-analysis/ and results/{explain_v7,jspace_v7,objwords_v7,objsal_test*,
# detail_v10*,ablation_v10}, with the arguments they were run with. Each group submits its own jobs through
# slurm/submit.sh (logs in logs/explain/); job ids go to stdout.
#
#   slurm/explain.sh <group>
#
#   v7_saliency  grad x input attribution, v7 decay branches at step 10750 (30 questions per type)  6 jobs, ~1-3 h each
#   v7_ablate    geometry-path ablations on results/explain_v7/ids_200.txt                         16 jobs, ~1-3 h each
#   v7_analysis  tables and figures from the two groups above (after they finish)                   2 jobs
#   jspace_v7    J-lens fit and readouts (text and visual lens), v7 step 10750 branches             2 jobs, ~20-60 min
#   objwords_v7  object-word readouts and patching, v7 step 10750 branches                           1 job,  ~3-5 h
#   objsal       object saliency on question 3004: OWLv2 + SAM 2, then SAM 3 masks                   2 jobs, ~2 min
#   screen       SAM 3 screening of the direction questions the two v10 arms answer differently     1 job,  ~15 min
#   detail       detailed analysis (masks, saliency, attention, J-lens) of 1221/1525, 3584/3655, 3004 5 jobs, ~3-7 min
#   swap         attention-swap ablation, v10 xa3_qknu s0 step 5750 (bias / uniform / diag)        3 jobs, ~15-35 min
#
# Before the first run: scripts/fetch_jacobian_lens.sh (jspace_v7, detail), and the id lists
# (python -m mvp.explain.ids v7 | swap | screen; they reproduce the committed files exactly).
# Pages: python -m mvp.explain.pages.build all (login node is fine).
set -euo pipefail
cd "$(dirname "$0")/.."
export LOGDIR=explain
S=slurm/submit.sh
GPU='GPU_BRD:H100|GPU_BRD:B200|GPU_BRD:B300'
v7() { echo "results/train/2B_v7_$1/branches/step10750/final.pt"; }

case "${1:-}" in
v7_saliency)
    for r in additive_s0 additive_s1 xa3_both_s0 xa3_both_s1 lora_s0 lora_s1; do
        echo "sal $r $($S -J sal_$r -t 03:00:00 -c 8 -- python -m mvp.explain.explain_saliency $r "$(v7 $r)")"
    done ;;
v7_ablate)
    abl() { echo "abl $1 $2 $($S -J abl_$1_$2 -t 03:00:00 -c 8 -- python -m mvp.explain.explain_ablate $2 --fusion-ckpt "$(v7 $1)" \
        --nframes 32 --split full --ids results/explain_v7/ids_200.txt --out results/explain_v7/abl/$1_$2.jsonl)"; }
    mkdir -p results/explain_v7/abl
    for r in additive_s0 additive_s1; do for m in zero shuffle; do abl $r $m; done; done
    for r in xa3_both_s0 xa3_both_s1; do for m in zero shuffle self nonself frame otherframe; do abl $r $m; done; done ;;
v7_analysis)
    $S -J expl_analysis -t 00:40:00 -c 16 -- python -m mvp.explain.explain_analysis
    $S -J expl_fig -t 00:20:00 -c 8 -- python -m mvp.explain.explain_figures results/explain_v7/figures ;;
jspace_v7)
    $S -J jspace_branch -t 02:00:00 -- python -m mvp.explain.jspace.jspace --models lora_s0 additive_s0 --prompts 30 --videos 3 --per-type 1 --out results/jspace_v7
    $S -J jspace_xa3 -t 03:00:00 -- python -m mvp.explain.jspace.jspace --models xa3_both_s0 --prompts 30 --videos 3 --per-type 1 --out results/jspace_v7 ;;
objwords_v7)
    $S -J objwords_v7 -t 05:00:00 -- python -m mvp.explain.objwords.objwords --models lora_s0 additive_s0 xa3_both_s0 \
        --cap 200 --patch 50 --patch-layers 3 11 19 --out results/objwords_v7 ;;
objsal)
    $S -J objsal_test -t 00:40:00 -c 8 --mem 96G --constraint "$GPU" -- python -m mvp.explain.one_question --id 3004
    $S -J objsal_sam3 -t 00:40:00 -c 8 --mem 96G --constraint "$GPU" -- python -m mvp.explain.one_question --id 3004 --masker sam3 --thr 0.5 --out results/objsal_test_sam3 ;;
screen)
    ids=$(python3 -c "import json; d = json.load(open('results/detail_v10/screen_ids.json')); print(' '.join(map(str, d['xa'] + d['add'])))")
    $S -J screen_sam3 -t 01:00:00 -c 8 --mem 64G --constraint "$GPU|GPU_BRD:A100" -- python -m mvp.explain.screen --ids $ids --out results/detail_v10/screen.json ;;
detail)
    for r in 2B_v10_xa3_qknu_s0 2B_v9_fmix_additive_s0; do
        $S -J detail_$r -t 01:30:00 -c 8 --mem 128G --constraint "$GPU" -- python -m mvp.explain.detail --run $r --ids 1221 1525
        $S -J detail2_$r -t 00:40:00 -c 8 --mem 128G --constraint "$GPU" -- python -m mvp.explain.detail --run $r --ids 3584 3655
    done
    $S -J detail_3004 -t 00:30:00 -c 8 --mem 128G --constraint "$GPU" -- python -m mvp.explain.detail --run 2B_v10_xa3_qknu_s0 --ids 3004 --out results/detail_v10_3004 ;;
swap)
    R=2B_v10_xa3_qknu_s0; mkdir -p results/ablation_v10/$R
    for m in bias uniform diag; do
        echo "$m $($S -J swap_$m -t 01:30:00 -- python -m mvp.explain.swap_eval --run $R --step 5750 --mode $m \
            --ids results/ablation_v10/ids.txt --out results/ablation_v10/$R/step5750_$m.jsonl)"
    done
    echo "then: python -m mvp.explain.swap_summary --run $R --control 2B_v9_fmix_additive_s0 --step 5750" ;;
*)
    sed -n '2,20p' "$0"; exit 1 ;;
esac
