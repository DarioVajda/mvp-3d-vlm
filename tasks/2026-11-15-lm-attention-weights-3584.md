---
id: 2026-11-15-lm-attention-weights-3584
title: Language-model attention weights from the answer position on question 3584, v10 and v9, bf16 and float32
author: vtwoptwo
created: 2026-11-15
status: queued
priority: normal
kind: eda
training: false
downloads: []
gpu_hours: 0.5
disk_gb: 0.1
depends_on: []
papers: []
results: task-results/2026-11-15-lm-attention-weights-3584/
job_ids: []
---

# Language-model attention weights on question 3584, cluster vs Mac

## Question

At the last prompt position t, where the answer letter is predicted, how is the attention weight in the language
model's softmax attention layers spread over the 9,600 visual token positions and the text positions? Does it differ
between v10 (3D cross-attention fusion) and v9 (additive fusion)?

This task also checks the Mac run. On the M2 Max, v10's logit difference for 3584 is +0.81 (bf16) and +0.48
(float32), while the saved cluster result is −0.086. The cluster run should show whether that gap comes from the
hardware or from something else.

## Hypothesis and what would change our mind

- **Expected:** on the cluster, the attention weights match the Mac's closely: rank correlation above 0.9 over the
  visual token positions in each layer, most of the top-50 positions shared, per-frame weights within about 0.005. The
  logit difference lands near the earlier cluster value (−0.086 in bf16), so the Mac gap is numerical.
- **Proves it wrong:** the attention weights differ a lot between cluster and Mac (rank correlation below 0.8, or
  different top frames), or the cluster logit difference also comes out far from −0.086. Then the inputs differ
  between machines (frames, the 3D points from VGGT, package versions), not just the floating-point arithmetic.
- **Next, in each case:** if they agree, run the analysis on the Mac for the other 7 shortlisted questions and compare
  v9 with v10. If they don't, find the input that differs before running anything else.

## Grounding in previous work

- The question choice comes from `results/detail_v10/screen_ids.json` (v10 right, v9 wrong) and
  `results/detail_v10/screen.json`. In the latter, SAM 3 detects all three objects in 6 or more of the 32 frames: window
  12, backpack 11, towel 6.
- The cluster logit difference for v10 on 3584 (−0.0864, answer D) is from
  `results/detail_v10/2B_v10_xa3_qknu_s0/3584_report.json`, field `margin`.
- Layer types are from the Qwen3.5-2B `config.json`: 24 decoder layers and `full_attention_interval` 4. So only
  layers 3, 7, 11, 15, 19 and 23 (counting from 0) are softmax attention. The other 18 are Gated DeltaNet layers, which
  have no attention weights.
- The README already notes cross-GPU variation: "the answer margin for question 3004 differs in the second decimal
  between two runs of the same checkpoint" (`explainability-analysis/README.md`, last paragraph). The Mac gap is much
  larger than that.

## Method

New script: `src/mvp/explain/attn_weights.py`. It changes no existing code. It reuses `FusionVL.load`,
`InputBuilder`, `uniform_frames`, `build_prompt`, `explain_saliency.targets` and `one_question.enrichment`.

- **Capture.** It registers an attention function `sdpa_lastrow` with transformers' `AttentionInterface`, on the text
  layers only. That function runs the normal SDPA, so the model output is unchanged. It also computes
  softmax(q_t Kᵀ · head_dim^(−1/2)) for the last query position, after the q/k norms and RoPE. The 2 key/value heads are
  repeated to match the 8 query heads, as transformers' `repeat_kv` does.
- **Checked on the plain Qwen3.5-2B** (Mac, `mps`, question 3584):
  - logits identical with the capture on and off;
  - every row sums to 1;
  - the captured weights match the model's own eager attention weights to within 1.7×10⁻³ (bf16 rounding).
- **Logit difference:** z_gt − logsumexp(z_alt) over the option letters, the same quantity as `explain_saliency.Run.margin`.
- **Per layer,** per head and as the mean over heads, the script stores:
  - the visual share;
  - the weight on each of the 32 frames (the sum over its 300 visual token positions);
  - object enrichment from the saved SAM 3 masks;
  - Spearman correlation with the saved saliency maps in `results/detail_v10/<run>/3584_maps.npz`.
- **`--fp32`:** casts the model to float32 and turns autocast off.

### Inputs

- **Data:** VSI-Bench question 3584. Video `scannet/scene0357_01.mp4`, 32 uniform frames (indices 0 … 906), 480 × 640.
- **Models:** `results/train/2B_v10_xa3_qknu_s0/branches/step5750/final.pt` and
  `results/train/2B_v9_fmix_additive_s0/branches/step5750/final.pt`, both already on the cluster. Base model
  `Qwen/Qwen3.5-2B` and `facebook/VGGT-1B`, both already in the shared cache.
- **Settings:** 32 frames and `EVAL_PIXELS`, as in evaluation. bf16 (the default) and float32 (`--fp32`).
- **Masks and saliency maps:** read from `results/detail_v10/<run>/3584_{masks,maps}.npz`, already committed.

### Commands

`src/mvp/explain/attn_weights.py` must be pushed first.

```bash
T=task-results/2026-11-15-lm-attention-weights-3584
# smoke test: v10, bf16 (one question, about 5 min)
slurm/submit.sh -J attn3584_smoke -t 00:20:00 -c 8 --mem 128G -- python -m mvp.explain.attn_weights \
    --run 2B_v10_xa3_qknu_s0 --ids 3584 --device cuda --out $T/cuda_bf16
# full run: both models, bf16 and float32 (v10 bf16 is the smoke test above)
slurm/submit.sh -J attn3584_v9_bf16 -t 00:20:00 -c 8 --mem 128G -- python -m mvp.explain.attn_weights \
    --run 2B_v9_fmix_additive_s0 --ids 3584 --device cuda --out $T/cuda_bf16
for r in 2B_v10_xa3_qknu_s0 2B_v9_fmix_additive_s0; do
  slurm/submit.sh -J attn3584_${r}_fp32 -t 00:20:00 -c 8 --mem 128G -- python -m mvp.explain.attn_weights \
      --run $r --ids 3584 --device cuda --fp32 --out $T/cuda_fp32
done
```

### Comparing with the Mac

The Mac outputs for v10 are already in `task-results/2026-11-15-lm-attention-weights-3584/mac_bf16/` and `mac_fp32/`.

```bash
python - <<'EOF'
import json, numpy as np
T = "task-results/2026-11-15-lm-attention-weights-3584"
for prec in ["bf16", "fp32"]:
    p = lambda m, ext: f"{T}/{m}_{prec}/2B_v10_xa3_qknu_s0/3584_attn.{ext}"
    a, b = np.load(p("mac", "npz")), np.load(p("cuda", "npz"))
    ja, jb = json.load(open(p("mac", "json"))), json.load(open(p("cuda", "json")))
    print(prec, "logit difference: mac", round(ja["logit_difference"], 3), "cuda", round(jb["logit_difference"], 3))
    vis, fr = a["visual"], a["frame_of_cell"]
    for L in [3, 7, 11, 15, 19, 23]:
        x, y = a[f"a_{L}"].mean(0)[vis], b[f"a_{L}"].mean(0)[vis]
        rho = np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1]
        top = len(set(np.argsort(x)[-50:]) & set(np.argsort(y)[-50:]))
        dfr = np.abs(np.bincount(fr, x) - np.bincount(fr, y)).max()
        print(f"  layer {L}: Spearman {rho:.3f}  top-50 shared {top}/50  max per-frame difference {dfr:.4f}")
EOF
```

## Smoke test

The v10 bf16 run. Check the log line `3584: answer 'D' gt 'D' logit diff … T 9789 visual 9600 layers
[3, 7, 11, 15, 19, 23]`:

- the answer is D;
- T = 9789 and 9,600 visual positions;
- all six layers are listed;
- the logit difference is close to the earlier cluster value −0.086.

If any of these fail, stop and note it under "Notes from Dario".

## Expected outputs

In `task-results/2026-11-15-lm-attention-weights-3584/`:

```
mac_bf16/2B_v10_xa3_qknu_s0/3584_attn.{npz,json}     already there (Mac, mps)
mac_fp32/2B_v10_xa3_qknu_s0/3584_attn.{npz,json}     already there (Mac, mps)
cuda_bf16/<run>/3584_attn.{npz,json}                 2 runs, about 1 MB each
cuda_fp32/<run>/3584_attn.{npz,json}                 2 runs, about 1.7 MB each
```

- `3584_attn.npz`:
  - `a_<layer>` [8 heads, 9789]: the attention weights of the last prompt position;
  - `visual` [9789]: True at visual token positions;
  - `frame_of_cell` [9600]: the frame of each visual token position.
- `3584_attn.json`:
  - the answer and the logit difference;
  - the forward time;
  - per layer and per head (and the mean over heads): visual share, per-frame weight, object enrichment, and
    Spearman correlation with the saliency maps.

## Results so far on the Mac (v10, question 3584, `mps`)

| | bf16 | float32 | earlier cluster result |
|---|---|---|---|
| Answer | D | D | D |
| Logit difference z_gt − logsumexp(z_alt) | +0.813 | +0.477 | −0.086 |
| Forward time (VGGT included) | 67 s* | 58 s | |

\* while another job was using the CPU.

Visual share: the attention weight on visual token positions, as a mean over heads.

| Layer | 3 | 7 | 11 | 15 | 19 | 23 |
|---|---|---|---|---|---|---|
| bf16 | 0.178 | 0.051 | 0.168 | 0.098 | 0.051 | 0.055 |
| float32 | 0.177 | 0.050 | 0.164 | 0.112 | 0.048 | 0.054 |

Object enrichment in float32: the share of the visual attention weight on an object's positions, divided by the share
of positions the object covers, over the frames where it is detected. 1 means as much weight as the object's area
predicts.

| Layer | 3 | 7 | 11 | 15 | 19 | 23 |
|---|---|---|---|---|---|---|
| window (where the observer stands) | 1.14 | 0.72 | 0.75 | 0.29 | 0.91 | 0.77 |
| backpack (the facing direction) | 0.38 | 1.05 | 0.62 | 0.37 | 1.01 | 1.58 |
| towel (the object asked about) | 0.18 | 0.86 | 1.79 | **6.71** | 3.01 | 4.34 |

bf16 and float32 on the Mac agree closely, per layer:

| Layer | 3 | 7 | 11 | 15 | 19 | 23 |
|---|---|---|---|---|---|---|
| Spearman over visual token positions | 0.974 | 0.978 | 0.975 | 0.928 | 0.954 | 0.966 |
| Top-50 positions shared | 49/50 | 49/50 | 45/50 | 47/50 | 43/50 | 47/50 |

The largest difference in a frame's weight is 0.0022.

## Failure modes and checks

| Risk | How we'd notice | What to do |
| ---- | --------------- | ---------- |
| Capture misses a layer or changes the output | Fewer than 6 layers listed; or the logits differ from a plain `sdpa` run | Check `config._attn_implementation` on the text config; rerun the plain-model check |
| Not enough memory in float32 | CUDA out of memory | 128G RAM and one B200/B300 is plenty (Mac peak was 26 GB in bf16); otherwise run bf16 only |
| The cluster logit difference also disagrees with −0.086 | Smoke-test log line | Compare frames (`frame_idx` in the json) and package versions; report in Notes |
| Attention weights differ between cluster and Mac | Comparison script: Spearman < 0.8 or different top frames | Don't run more questions on the Mac until explained |

## Resources

One B200/B300 (H100 is fine), 8 CPUs, 128 GB RAM. Four jobs of about 5 min each. No downloads. Under 10 MB of outputs.

## Notes for Dario

- The script is new, so please run it from the pushed commit that contains it.
- If `--device cuda` with `--fp32` is slow because of the Gated DeltaNet reference kernels in float32, the bf16 runs
  alone are still useful.

## Notes from Dario

<!-- Dario writes here when status is needs-info, failed or done -->

## Grill log

- **Why question 3584?** v10 is right and v9 wrong on it. All three objects are clearly visible, and its saliency and
  J-lens readouts already exist.
- **Why only the last prompt position?** The answer letter is predicted there. Storing one row per layer keeps the
  output at about 1 MB instead of the full 9,789 × 9,789 matrices.
- **Why only six layers?** The other 18 decoder layers are Gated DeltaNet and have no attention weights.
- **Why both bf16 and float32?** The Mac logit difference changed by 0.34 between them, while the attention weights
  barely moved. Running both on the cluster separates precision effects from hardware effects.
