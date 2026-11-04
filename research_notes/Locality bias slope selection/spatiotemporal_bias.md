# Spatial vs temporal (cross-frame) attention biases and multi-axis RoPE

Scope: how video transformers, video LLMs and multi-view 3D models set spatial and temporal position terms, what their constants are, and what is known about frame-count and key-count extrapolation. Our layer for reference: logit bias −λ_h(|Δr|+|Δc|) − μ_h[t≠t′], learned per head, init λ = μ = 4, plus 3-axis RoPE with 16/24/24 pairs for t/r/c. The grid is 20×15 per frame, with T = 8 at training and 32 at evaluation.

Verification legend: unmarked items were read in the primary source (paper HTML or official code/config) during this session. **[UNVERIFIED]** marks items from memory or from secondary summaries whose primary text I could not open. The arXiv PDFs for ALiBi, Chiang & Cholak and MViTv2 came back as binary I could not parse, and no PDF tools were available.

## Q1. Video transformers with relative spatio-temporal biases: values and initialisation of temporal vs spatial terms

### Takeaway
None of the canonical video backbones uses a hand-set, scalar, monotone distance penalty, and none sets the temporal strength relative to the spatial one by hand. They use learned tables or embeddings initialised at or near zero, and the temporal term starts either at zero or as a copy of the spatial values. A fixed slope of 4 nats per grid step is far steeper than any prior in this family. The closest analogue to a "slope" is ALiBi's fixed geometric slopes (≤ 0.5 per token in language models).

### Cited Findings
- **Video Swin** (arXiv 2106.13230; official code `WindowAttention3D`) uses one joint table, not separate temporal and spatial terms. The table is `relative_position_bias_table` with shape `(2Wd−1)(2Wh−1)(2Ww−1) × num_heads`: one learned scalar per (Δt, Δh, Δw) triple and per head. It is created with `torch.zeros` and then `trunc_normal_(…, std=.02)`, so the spatial and temporal offsets get the same near-zero initialisation. The flat index scales Δt by (2Wh−1)(2Ww−1) and Δh by (2Ww−1). Default windows are (2,7,7) in `SwinTransformer3D`/`SwinTransformerBlock3D` and (1,7,7) in `BasicLayer` defaults. When the model is inflated from a 2D Swin, `inflate_weights` tiles the pretrained 2D table along the temporal axis (`repeat(2*wd-1,1)`), so every temporal offset starts with exactly the image-pretrained spatial bias and there is no extra temporal penalty — [Video Swin code](https://raw.githubusercontent.com/SwinTransformer/Video-Swin-Transformer/master/mmaction/models/backbones/swin_transformer.py)
- **MViTv2** (arXiv 2112.01526, official PySlowFast code) uses decomposed, content-dependent relative positions rather than a scalar bias. Separate tables `rel_pos_h`, `rel_pos_w` (shape `(2·max(q_size,kv_size)−1, head_dim)`) and `rel_pos_t` (shape `(2·T_in−1, head_dim)`) are dotted with the query (`einsum("bythwc,hkc->bythwk", r_q, Rh)`; for time, `matmul(r_q, Rt^T)`), and the three terms are added to the logits. Initialisation is zeros when `rel_pos_zero_init=True`, otherwise `trunc_normal_(std=0.02)`, with the same rule for spatial and temporal tables. When sizes differ, tables are resampled with `F.interpolate(mode="linear")`, and distance indices are rescaled by q/k pooling ratios — [PySlowFast attention.py](https://raw.githubusercontent.com/facebookresearch/SlowFast/main/slowfast/models/attention.py)
- MViTv2's motivation: the original MViT's "space-time modeling relies solely on the 'absolute' positional embedding", which ignores shift-invariance. The decomposed embedding is `R_p(i),p(j)` in the paper's Fig. 2 — [arXiv 2112.01526](https://arxiv.org/abs/2112.01526v2); [CVPR paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_MViTv2_Improved_Multiscale_Vision_Transformers_for_Classification_and_Detection_CVPR_2022_paper.pdf)
- **[UNVERIFIED, paper PDF unreadable]** MViTv2 §4.1 writes the term as R_p(i),p(j) = R^h + R^w + R^t, with the logit term Q·R^T added before softmax, and states parameter cost O(T+H+W) rather than O(THW) for a joint table. The ablation table compares absolute, joint relative and decomposed relative positions — [arXiv 2112.01526](https://arxiv.org/pdf/2112.01526)
- **ALiBi** (arXiv 2108.12409): the abstract says it biases query-key scores "with a penalty that is proportional to their distance", with no positional embeddings. A 1.3B model trained at length 1024 and evaluated at 2048 matches a sinusoidal model trained at 2048, while training 11% faster with 11% less memory — [ALiBi abstract](https://arxiv.org/abs/2108.12409)
- **[UNVERIFIED, from memory of §3]** ALiBi slopes are fixed, not learned: a geometric sequence starting at 2^(−8/n) with ratio 2^(−8/n) for n heads. For 8 heads that is 1/2, 1/4, …, 1/256. The paper reports that learned slopes did not give strong extrapolation — [ALiBi](https://arxiv.org/abs/2108.12409)
- **TimeSformer / ViViT / VideoMAE v2**: I found no relative-bias constants for these in this session. **[UNVERIFIED]** To my knowledge they use learned absolute (separate space and time) position embeddings and factorise attention (divided space-time attention, or a factorised encoder) instead of using distance biases, so there are no temporal-vs-spatial bias constants to report.

### Inferences
- With λ = 4 per Manhattan step, a key one cell away gets e^−4 ≈ 0.018 of the weight of the co-located key, before content and RoPE. Summed over a 20×15 grid (about 4d cells at distance d), the non-co-located spatial mass is Σ 4d·e^(−4d) ≈ 0.076, so about 93% of each frame's mass sits on the co-located cell at initialisation (computed). This is near-hard locality. Prior video work starts the position terms at about zero (std 0.02) and lets data learn them, which suggests a much smaller initial λ (around 0.5–1, the ALiBi-steepest-head range) if the aim is a soft prior rather than a hard mask.
- Video Swin's inflation trick (temporal offsets copy the spatial values at Δt = 0) is the only precedent I found for "relative to each other". It sets the initial temporal penalty to 0 relative to the same spatial offset. Nothing I found sets μ ≥ λ by hand.

### Gaps
- Exact MViTv2 ablation numbers (abs vs joint vs decomposed relative positions on K400/IN-1k) and the ALiBi slope text could not be read from the PDFs.
- Found no "temporal ALiBi" or exponential-decay variant for video backbones with published constants.

## Q2. Video LLMs and multi-axis RoPE: frequency splits and frame-count extrapolation

### Takeaway
Our 16/24/24 split is exactly Qwen2-VL's `mrope_section`. In that layout time takes the 16 highest-frequency pairs, which VideoRoPE and HoPE both identify as the weak point for long-video and more-frames extrapolation. The successors either move time to the lowest frequencies (VideoRoPE, Qwen3-VL's interleaved layout) or zero the temporal frequencies entirely (HoPE). They also compress or rescale the temporal index (VideoRoPE δ, HoPE γ). Every reported train-short/eval-long gain comes from these changes.

### Cited Findings
- **Qwen2-VL-7B** config: `rope_scaling: {type: "mrope", mrope_section: [16, 24, 24]}`, `rope_theta` 1,000,000, hidden 3584, 28 heads (head_dim 128 = 64 rotary pairs = 16+24+24), `max_position_embeddings` 32768 — [Qwen2-VL-7B-Instruct config](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/raw/main/config.json)
- **Qwen3-VL-8B** config: `mrope_section: [24, 20, 20]`, `mrope_interleaved: true`, `rope_theta` 5,000,000, head_dim 128. Time gets more pairs (24), and the axes are interleaved across the frequency spectrum instead of time holding a contiguous high-frequency block — [Qwen3-VL-8B-Instruct config](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/raw/main/config.json)
- **VideoRoPE** (arXiv 2502.05173, §3–4):
  - Its account of M-RoPE: of 128 dims, the first 32 are t, then 48 x and 48 y. Lower dims rotate faster, so t gets θ0–θ15, the high frequencies. High-frequency channels oscillate with short monotonic intervals, so distant frames can get near-identical embeddings ("hash collisions"). Fig. 3 shows M-RoPE's temporal channels attend mostly to local frames.
  - Its own layout: t on dims 96–127 (θ48–θ63, lowest frequencies), with x and y interleaved over dims 0–95 (48 each).
  - Interleaved x/y beats sequential x/y (App. A.1, Table 7).
  - Adjustable temporal spacing δ multiplies the frame index of visual tokens. Ablation (Table 6, LongVideoBench/MLVU/VideoMME average): δ = 0.5 → 56.34, 1.0 → 59.11, 2.0 → 60.92 (chosen), 3.0 → 59.18.
  - Training: 8k context, max 128 frames at 2 fps, Qwen2-VL-7B init, about 1.3M QA pairs. Evaluation at 8k, 16k, 32k and 64k.
  - — [VideoRoPE HTML](https://arxiv.org/html/2502.05173v3)
- VideoRoPE Table 2, LongVideoBench at 8k / 16k / 32k / 64k: M-RoPE 53.42 / 52.80 / 53.11 / 54.35 vs VideoRoPE 54.46 / 55.29 / 57.15 / 57.26. MLVU at 64k: 61.10 vs 65.56. Video-MME at 64k: 59.67 vs 61.33. V-NIAH: vanilla RoPE 31.78, M-RoPE 78.67, VideoRoPE 91.11. V-NIAH-D (periodic distractors): 30.22, 74.67, 87.11 — [VideoRoPE HTML](https://arxiv.org/html/2502.05173v3)
- **HoPE** (arXiv 2505.20444, §4):
  - Hybrid allocation for d = 128: x and y are interleaved in 96 dims (48 each), and the 32 temporal dims have frequency set to zero (identity rotation, i.e. NoPE on time). App. B.4 states the original allocation as 16/24/24 pairs t/x/y.
  - Dynamic temporal scaling: γ ∈ {0.5, 0.75, 1, 1.25, 1.5} is sampled per video in training. At inference γ is chosen per task: 0.75 for V-NIAH and 1.5 for LongVideoBench (App. B.3).
  - Training at 8k, evaluation at 8–64k.
  - V-NIAH average: vanilla 21.00, M-RoPE 47.11, VideoRoPE 52.00, HoPE 63.56.
  - Ablation, Video-MME 7B at 32k: vanilla 57.99 → +3D 58.37 → +HFA 59.33 → +DTS 59.44. At 64k the 3D-only (M-RoPE-like) variant falls to 23.43, below vanilla's 26.13.
  - 7B at 64k (MLVU/LVB/VMME): HoPE 50.01 / 51.22 / 27.34, VideoRoPE 45.31 / 47.37 / 26.52, M-RoPE 46.88 / 50.00 / 23.43.
  - — [HoPE HTML](https://arxiv.org/html/2505.20444v1)
- **RoPE for ViT** (Heo et al., arXiv 2403.13298, §3.2):
  - RoPE-Axial splits the rotary channels half to x and half to y (interleaved even/odd), with base θ = 100 (Eq. 13), the square root of the 1D base 10000, because each axis gets half the frequencies. Frequencies are fixed.
  - RoPE-Mixed learns a frequency pair (θx_t, θy_t) per rotary pair, per head and per layer (about 0.01% of ViT-B parameters), which allows diagonal directions.
  - ViT-B trained at 224, top-1 at 224 / 320 / 384 / 512: APE 83.4 / 83.5 / 82.8 / 80.5; RoPE-Axial 83.6 / 84.3 / 83.9 / 82.0; RoPE-Mixed 83.8 / 84.7 / 84.4 / 82.9 (Table A.4). Swin-B with relative position bias: 83.3 at 224 and 81.2 at 384 (Table A.8).
  - — [RoPE-ViT HTML](https://arxiv.org/html/2403.13298v2)
- **V2PE** (arXiv 2412.09616): visual tokens get smaller, variable position increments than text tokens. InternVL2 trained on 256K-token sequences handles multimodal sequences up to about 1M tokens. Exact increment values were not retrieved — [V2PE abstract](https://arxiv.org/abs/2412.09616)

### Inferences
- In our layer, RoPE on t at training only sees Δt ∈ [−7, 7]. At T = 32 it sees Δt up to ±31. With the 16 highest-frequency pairs on t (the Qwen2-VL convention, if our implementation follows its ordering), unseen Δt values alias onto seen phases: the "hash collision" failure from VideoRoPE §4. Three options are backed by the literature: (i) place the t pairs on the lowest frequencies (VideoRoPE, Qwen3-VL interleaving); (ii) zero the temporal frequencies and let μ carry same-frame vs other-frame (HoPE). This is natural for us, since μ already encodes the frame boundary, and geometry tokens from VGGT are in a shared world frame where frame order is weakly informative. (iii) rescale the frame index by (T_train−1)/(T_eval−1) = 7/31 at evaluation, so Δt stays in the training range (cf. VideoRoPE δ and HoPE γ, which show sensitivity of about ±2–4 points to the scale).
- Whether our 16/24/24 pairs follow Qwen2-VL's ordering (t on highest frequencies) should be checked in code. If they are copied from Qwen2-VL, they inherit the configuration that VideoRoPE and HoPE show extrapolates worst.

### Gaps
- VRoPE (arXiv 2502.11664) was not fetched; no constants to report.
- No paper found that evaluates M-RoPE-style RoPE inside a cross-attention adapter (as opposed to LLM self-attention) under a frame-count change.

## Q3. Multi-view 3D models: cross-view vs within-view attention biases

### Takeaway
Feed-forward multi-view reconstructors (VGGT) separate "same frame" from "other frame" architecturally, by alternating frame-wise and global attention, not with a learned scalar penalty. Generalisable-NeRF transformers use hard epipolar geometry rather than learned per-head cross-view penalties. I found no published learned per-head cross-view scalar like our μ_h.

### Cited Findings
- **VGGT** (arXiv 2503.11651):
  - L = 24 blocks, each with one frame-wise and one global self-attention layer (48 attention layers), and no cross-attention (§3.2, App. B).
  - There is no per-frame positional bias in global attention. Frames are distinguished only by giving the first frame its own camera and register tokens, and the model is otherwise permutation-equivariant (§3.1, §3.3).
  - Training samples 2–24 frames per scene (48 per batch). Inference claims "hundreds", with runtime and memory measured up to 200 frames (Table 9, about 40.6 GB).
  - ETH3D point-map ablation (Table 5, overall ↓): cross-attention 1.061, global-only 0.827, alternating 0.709.
  - — [VGGT HTML](https://arxiv.org/html/2503.11651v1)
- **GNT** ("Is Attention All That NeRF Needs?", arXiv 2207.13298): the view transformer aggregates source-view features under the epipolar constraint (features sampled along epipolar lines), so the cross-view prior is geometric and hard rather than a learned penalty — [GNT](https://arxiv.org/pdf/2207.13298)
- **EVE-NeRF** (CVPR 2024) entangles cross-view and along-epipolar aggregation — [EVE-NeRF](https://openaccess.thecvf.com/content/CVPR2024/papers/Min_Entangled_View-Epipolar_Information_Aggregation_for_Generalizable_Neural_Radiance_Fields_CVPR_2024_paper.pdf). Epipolar methods depend on accurate calibration — [review](https://liner.com/review/epipolar-transformers)

### Inferences
- VGGT itself generalises from ≤ 24 training frames to 100+ inference frames with plain global softmax attention and no frame penalty. This suggests its geometry tokens are already strongly cross-frame-consistent. Our queries attend to geometry tokens that already went through global attention, so other-frame keys carry largely redundant world-frame information. That argues for a sizeable μ, or for μ scaled with frame count as in Q4, rather than a small one.
- VGGT's Table 5 (cross-attention is worst, and slow as frames grow) is a caution for cross-attention designs over many frames, although its setting (backbone attention) differs from ours (an adapter).

### Gaps
- MVSplat, pixelSplat, OmniSpace, Fast3R, CUT3R and DUSt3R constants were not retrieved. **[UNVERIFIED]** To my knowledge none uses a learned scalar cross-view penalty: DUSt3R uses pairwise cross-attention with RoPE-2D per image, Fast3R uses global attention with image-index embeddings, CUT3R uses a recurrent state.
- Epipolar Transformers (He et al. 2020) bias values were not found.

## Q4. Key-count growth at evaluation: attention dilution, log-n temperature, normalising the bias by frame count

### Takeaway
Softmax mass on any fixed key set shrinks as the number of competing keys grows unless the logit gaps grow by about log n. SSMax (s·log n scaling of logits, s learned per head) and Chiang & Cholak's log-n scaling are the standard fixes. Applied to our layer, going from 8 to 32 frames at fixed μ = 4 cuts own-frame mass from about 89% to about 64%. Adding ln((T_eval−1)/(T_train−1)) = ln(31/7) ≈ 1.49 to μ at evaluation restores the training-time split exactly when content logits are neutral. I found no paper that normalises a cross-frame bias by frame count; this is an inference from the SSMax and log-n literature.

### Cited Findings
- **SSMax** (arXiv 2501.19399):
  - SSMax(z)_i = n^(s·z_i) / Σ_j n^(s·z_j), i.e. the query is scaled by s·log n before softmax (Eq. 2, Eq. 11).
  - s is a learnable scalar per layer and per head (144 parameters for 12×12), initialised to 1. For post-hoc conversion it is initialised to 1/mean(log n) ≈ 0.168 at training length 1024.
  - Under plain softmax the maximum output is at most 1/((n−1)/e^(z_max−z_min)+1) → 0 as n grows (Eq. 6). Under SSMax a fixed top-2 margin above about 1/s keeps the maximum near 1 (Eqs. 8–9).
  - A learnable bias variant (s·log n + b) speeds early loss slightly but hurts length generalisation and retrieval (§3.1–3.3).
  - SSMax keeps loss low up to about 10× training length (N = 1024, 162M model, RoPE θ raised 50× at test, §3.2). NIAH stays highly accurate at about 10× length where standard softmax fails (§3.3; exact percentages are not given in the text).
  - — [SSMax HTML](https://arxiv.org/html/2501.19399v1)
- **Chiang & Cholak** (arXiv 2202.12172) propose scaling attention logits by log n as a "simple remedy" that "also improves length generalization in machine translation" — [abstract](https://arxiv.org/abs/2202.12172). Secondary sources describe it as multiplying logits by log n — [Information Entropy Invariance, 2501.08570](https://arxiv.org/html/2501.08570v1); [2401.07004](https://arxiv.org/pdf/2401.07004). The base of the log (training length vs current length) is treated inconsistently across follow-ups — [Scaling Laws of RoPE-based Extrapolation, 2310.05209](https://arxiv.org/pdf/2310.05209). **[UNVERIFIED]** exact MT train and test lengths and BLEU numbers.
- Related temperatures: InfoScale and CosScale (entropy-invariance temperatures) — [2501.08570](https://arxiv.org/html/2501.08570v1). A log L/√d factor citing Su (2021) entropy invariance — [Softplus attention, 2501.13428](https://arxiv.org/pdf/2501.13428).
- In the video-LLM papers, frame-count extrapolation is handled by rescaling positions (VideoRoPE δ, HoPE γ ∈ {0.5…1.5} chosen per input length) rather than by logit temperature — [VideoRoPE](https://arxiv.org/html/2502.05173v3); [HoPE](https://arxiv.org/html/2505.20444v1)

### Inferences (computed for our layer; content and RoPE taken as neutral)
- Spatial mass per frame ≈ 1.076 relative to the co-located key at λ = 4. Each other frame contributes the same times e^−μ, so own-frame fraction = 1/(1 + (T−1)e^−μ):

  | T | μ | Own-frame fraction |
  |---|---|---|
  | 8 | 4 | 0.886 |
  | 32 | 4 | 0.638 |
  | 8 | 5.49 | 0.972 |
  | 32 | 5.49 | 0.886 |

  A fixed μ therefore moves about 25 points of attention mass to other frames at evaluation, which is a distribution shift the trained output projection never saw.
- Frame-count-aware μ, the SSMax analogue on the frame axis: μ_eff = μ_h + ln(T−1) − ln(T_train−1), or equivalently parametrise μ_h = μ̃_h + ln(T−1). This is scale-invariant in the sense of SSMax: a learned μ̃ fixes the same-vs-other-frame mass split regardless of T. An SSMax-style alternative multiplies all logits by s_h·log(N_keys); that also sharpens spatial locality, which we may not want.
- Other options:
  - (i) train with randomly sampled T ∈ {4…32} (the HoPE γ-sampling idea), so μ is learned across key counts;
  - (ii) report both the fixed-μ and the ln(T−1)-shifted variant at evaluation as an ablation;
  - (iii) because the spatial term is per frame, only the frame term needs this correction; λ needs no T-dependence.
- On relative strength (question a): the logit gap between the co-located same-frame key and the co-located other-frame key is μ. The gap between a same-frame neighbour at Manhattan distance d and an other-frame co-located key is μ − λd. With λ = μ = 4, a one-step spatial neighbour in the same frame ties with the co-located cell in every other frame. If the intent is "other frames matter less than nearby cells of this frame", μ should exceed λ (e.g. μ ≥ 2λ), plus the ln(T−1) correction above. No paper prescribes this ratio; it follows from the bias form.

### Gaps
- No paper found that normalises an additive cross-frame or cross-view bias by the number of frames, or reports a frame-count-dependent temperature in video or multi-view attention. This combination appears unpublished as of this search.
- Exact Chiang & Cholak MT numbers and the ALiBi extrapolation tables were not readable (PDF parsing unavailable).
