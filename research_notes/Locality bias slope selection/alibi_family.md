# ALiBi family: how distance-penalty slopes λ are chosen, initialised or learned, and how concentrated the resulting attention is

Scope: additive biases on attention logits of the form −λ_h·f(distance), in 1-D (language) and 2-D (images/grids), plus learned-table relatives (Swin, BEiT, iRPE, T5) for their initialisations. All constants below were read from the paper PDFs (text-extracted) or the official code unless marked UNVERIFIED.

Notation: logit_ij = q_i·k_j/√d − λ_h·dist(i,j). The ALiBi bias is **not** divided by √d ([ALiBi, §3 footnote 10](https://arxiv.org/abs/2108.12409)), so λ is in the same units as post-scaling logits. This matches our design.

**Reference calculation (my own numpy, not from a paper).** It shows how concentrated the attention prior is for the slopes reported below. Setup: a single 40×60 grid with the query at the centre and content logits = 0. Columns: softmax mass on the query's own cell, mass within distance ≤1, and mean attended distance E[d].

| λ | L1 self | L1 mass d≤1 | L1 mass d≤2 | L1 E[d] | L2 self | L2 mass d≤1 | L2 E[d] | Where this λ appears |
|---|---|---|---|---|---|---|---|---|
| 0.0039 | 0.000 | 0.002 | 0.006 | 24.6 | 0.000 | 0.002 | 19.1 | ALiBi weakest head (2⁻⁸); RMT weakest head |
| 0.0625 | 0.002 | 0.008 | 0.019 | 18.5 | 0.001 | 0.006 | 15.7 | ALiBi 8-head, head 4 |
| 0.25 | 0.016 | 0.064 | 0.140 | 7.8 | 0.010 | 0.042 | 7.7 | ALiBi 8-head, head 2 |
| 0.288 | 0.021 | 0.082 | 0.174 | 6.8 | 0.013 | 0.053 | 6.8 | RMT strongest head (−ln 0.75) |
| 0.5 | 0.060 | 0.206 | 0.382 | 3.8 | 0.040 | 0.136 | 4.0 | ALiBi 8-head strongest; ViTARC r_before |
| 0.707 | 0.115 | 0.343 | 0.567 | 2.6 | 0.079 | 0.234 | 2.8 | ALiBi/CROMA 16-head strongest (2^−0.5); ViTARC r_after |
| 1.0 | 0.214 | 0.528 | 0.759 | 1.7 | 0.154 | 0.380 | 1.9 | Sandwich "h=0" all-heads (bad); DINO-ALiBi m=1 on normalised distance |
| 2.0 | 0.580 | 0.894 | 0.979 | 0.55 | 0.502 | 0.773 | 0.70 | — |
| **4.0** | **0.929** | **0.997** | 1.000 | 0.07 | 0.918 | 0.985 | 0.09 | **our proposed init** (single frame, before the frame term μ) |
| 8.0 | 0.999 | 1.000 | 1.000 | 0.00 | 0.999 | 1.000 | 0.00 | — |

On the analytic side, an infinite 2-D L1 lattice gives self-mass = 1 / (1 + 4e^{−λ}/(1−e^{−λ})²), which equals 0.93 at λ=4 and 0.58 at λ=2. Our quoted 82% is lower because the μ term lets the same cell in other frames take some of the mass.

---

## Q1. ALiBi (Press, Smith, Lewis; arXiv 2108.12409, ICLR 2022): the slope sequence, why that range, the slope ablations, and whether learned slopes were tried

### Takeaway
ALiBi slopes are fixed, never learned: a geometric sequence 2^(−8i/H), i=1..H, spanning from 2^(−8/H) down to 2^(−8). The set was found by a "brief manual exploration of around ten slope sets". The authors found trainable slopes extrapolated poorly and ran 3% slower. Their stated rule: slopes should lie in (0,1) and be denser near 0. Even the sharpest ALiBi head (0.5 for 8 heads, 0.707 for 16) is about 6–8× weaker than our λ=4.

### Cited Findings
- Bias definition: softmax(q_i K^T + m·[−(i−1), …, −2, −1, 0]), "where scalar m is a head-specific slope fixed before training" (§3, Fig. 3: "m is a head-specific scalar that is set and not learned throughout training"). — [ALiBi](https://arxiv.org/abs/2108.12409)
- Exact slopes. For 8 heads: 1/2¹, 1/2², …, 1/2⁸. For 16 heads, "we interpolate those 8 slopes by geometrically averaging every consecutive pair", giving 1/2^0.5, 1/2¹, 1/2^1.5, …, 1/2⁸. "In general, for n heads, our set of slopes is the geometric sequence that starts at 2^(−8/n) and uses that same value as its ratio" (§3). — [ALiBi](https://arxiv.org/abs/2108.12409)
- How the set was chosen: "A brief manual exploration of around ten slope sets led us to discover the set of slopes that we finally picked. Our main insight from this exploration is that the slope sets that work best are those with slopes in the (0, 1) range, with the slopes' density increasing as we get closer to 0. We also found our method to be robust to slope choice. Even randomly sampling from the exponential distribution worked well in some cases (although that method had high variance)." (§3). The ten slope sets and their numbers are not reported. — [ALiBi](https://arxiv.org/abs/2108.12409)
- Learned slopes: "We initially experimented with making the slopes trainable, but this did not yield strong extrapolation results." Footnote 11: "trainable slopes also slowed down the training speed by 3%." (§3). — [ALiBi](https://arxiv.org/abs/2108.12409)
- The slopes were tuned once on WikiText-103 and reused unchanged on the Toronto BookCorpus and on a 1.3B CC100+RoBERTa model: "our set of slopes was not tuned on this dataset" (§4). The authors argue there is no need to re-tune per model or dataset, by analogy with the sinusoidal wavelengths. — [ALiBi](https://arxiv.org/abs/2108.12409)
- Multiplicative versus additive bias: "In our experiments (not presented), multiplying attention scores by the bias (instead of adding, as in ALiBi) degraded performance." (§5, discussion of the Distance Aware Transformer, which learns per-head parameters). — [ALiBi](https://arxiv.org/abs/2108.12409)
- Appendix B: ALiBi's apparent extrapolation is "largely explained by its improved avoidance of the early token curse". With sliding-window evaluation, "ALiBi might not be using contexts longer than the ones it was trained on". Example: L=512 sliding-window perplexity is flat from 17.98 to 18.3 between eval lengths 512 and 3072 (Table 15). — [ALiBi](https://arxiv.org/abs/2108.12409)
- Production code matches the paper. MPT/llm-foundry `gen_slopes(n_heads, alibi_bias_max=8)` computes m = arange(1..n)·8/n and slopes = 1/2^m. For non-power-of-2 head counts it interleaves `slopes[1::2], slopes[::2]`. The KERPLE/GPT-NeoX and CROMA repos use the same `get_slopes_power_of_2`: start = 2^(−2^−(log2 n − 3)), ratio = start. — [llm-foundry attention.py](https://github.com/mosaicml/llm-foundry/blob/main/llmfoundry/models/layers/attention.py); [KERPLE layers.py](https://github.com/chijames/KERPLE/blob/main/megatron/mpu/layers.py); [CROMA use_croma.py](https://github.com/antofuller/CROMA/blob/main/use_croma.py)

### Inferences
- The "8" in 2^(−8/H) is an empirical choice from about ten trials on WikiText-103. There is no principled derivation. The intended spread runs from strongly local heads (receptive field of a few tokens) to nearly position-agnostic heads (2⁻⁸ ≈ 0.004 per token, which still allows attention over ~hundreds of tokens).
- ALiBi's "learned slopes are bad" finding concerns **length extrapolation** in causal LMs. It does not say learned slopes hurt in-distribution quality. Our setting has a fixed grid size, so extrapolation is not the main concern and that negative result does not carry over directly.

### Gaps
- The specific alternative slope sets ALiBi tried, their numbers, and the learned slope values from the trainable-slope experiment are not reported anywhere I could find.

---

## Q2. 2-D extensions for images and grids: CROMA 2D-ALiBi/X-ALiBi, LookHere, RMT (Manhattan), ViTARC, DINO-ALiBi, Swin/BEiT/iRPE/CoAtNet. Which metric, and how initialised?

### Takeaway
Every 2-D linear-penalty method I found that uses fixed slopes copies the ALiBi geometric sequence (CROMA, ViTARC) or uses slopes ≤1.5 (LookHere). The only Manhattan-distance vision backbone (RMT) uses even weaker per-step penalties, 0.004–0.29. Euclidean (CROMA, LookHere, DINO-ALiBi) and Manhattan (RMT, ViTARC) are both used. Euclidean is usually justified by rotation invariance and Manhattan by analogy to the RetNet decay or raster/diagonal structure. LookHere found squared or square-root distance worse than linear. CROMA's X-ALiBi is the closest precedent to our design: a cross-attention between two spatially aligned modalities biased by grid distance with fixed ALiBi slopes. No 2-D paper initialises any head anywhere near λ=4.

### Cited Findings
**CROMA (Fuller, Millard, Green; arXiv 2311.00566, NeurIPS 2023)**
- 2D-ALiBi "bias[es] the self-attention matrix based on the Euclidean distance between query-key pairs in a ViT". X-ALiBi biases "the cross-attention matrix based on the Euclidean distance between cross-modal query-key pairs" (radar queries attending to optical keys). "For both 2D and X-ALiBi, we calculate our attention-head slopes with the same geometric sequence as ALiBi." With 16 heads, slopes are 1/2^0.5, 1/2¹, …, 1/2⁸. "Since our two modalities are aligned 2D sensor data, our 2D-ALiBi and X-ALiBi matrices are identical." (§3, Fig. 2 caption: "m is a fixed scalar"). — [CROMA](https://arxiv.org/abs/2311.00566)
- Code: `dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)`; bias = −dist·slope, in units of patches. The bias is a plain tensor, neither a Parameter nor a registered buffer, so it is not learned. — [CROMA use_croma.py](https://github.com/antofuller/CROMA/blob/main/use_croma.py)
- Stated rationale for Euclidean distance: "These position encoding techniques are rotation and translation invariant, which are desirable properties for the overhead imagery." — [CROMA §3](https://arxiv.org/abs/2311.00566)
- Ablation (Table 6): 2D-ALiBi alone beats PEG and 2D-sinusoidal (2D-sinusoidal averages −1.7 points). Adding X-ALiBi on top of 2D-ALiBi "further improves average performance by 0.2%… particularly improving multimodal performance". — [CROMA §4.1](https://arxiv.org/abs/2311.00566)
- Extrapolation (Table 7, mIoU, trained at 120 px): 2D-ALiBi scores 59.3 at 120 px and 58.8 at 504 px. 2D-sinusoidal drops from 59.2 to 19.5 (54.9 with interpolation). PEG drops from 58.3 to 57.1. — [CROMA §4.2](https://arxiv.org/abs/2311.00566)
- Appendix A.2: 2D-ALiBi gives rotation/flip-invariant image representations (cosine 0.992 versus 0.575 for 2D-sinusoidal, Table 11). It also limits patch-wise representational collapse: mean patch-patch cosine is 0.582 for 2D-ALiBi, 0.546 for 2D-ALiBi+X-ALiBi and 0.701 for PEG (Table 12). The paper attributes this to the fact that "At every ViT layer, 2D-ALiBi injects a local bias in each attention head". — [CROMA App. A.2](https://arxiv.org/abs/2311.00566)
- No slope ablation is reported in CROMA. — [CROMA](https://arxiv.org/abs/2311.00566)

**LookHere (Fuller, Kyrollos, Yassin, Green; arXiv 2405.13985, NeurIPS 2024)**
- Uses 2-D directional masks plus a Euclidean distance penalty: bias = m(l,h)·Distance(i,j) if j is visible to i, ∞ otherwise, with m(l,h) = s_l(l)·s_h(h)·s_g. Defaults: "s_l to linearly decrease from 1.5 to 0.5 with increasing depth", s_h = (1/2, 1/8, 1/32, 1/128) for the four undirected heads, and s_g = 1. The slopes are fixed and precomputed (A_FIX). — [LookHere §3](https://arxiv.org/abs/2405.13985)
- Ablation takeaways (§3, App. A.6): "LookHere is robust to the choice of slope function… We believe precisely tuning slopes is unnecessary because models can learn to scale attention logit magnitudes. Increasing penalties with the square or square root of the distance harms extrapolation. Removing all distance penalties harms extrapolation." — [LookHere](https://arxiv.org/abs/2405.13985)
- Table 12 (ViT-B, ImageNet, 150 epochs, 224² val top-1; LH-90 default ≈81.0):
  - Global slope s_g: 0.125 → 81.20, 0.25 → 81.08, 0.5 → 81.09, 4 → 80.91.
  - Distance shape: dist² → 80.98, √dist → 80.86, no distance → 80.19.
  - Learned layer slope (s_l fix→learn) → 81.35.

  In-distribution, slope magnitude barely matters (±0.3). — [LookHere App. A.6](https://arxiv.org/abs/2405.13985)
- Table 13 (trained at 224², tested at 1024², top-1). The global slope matters a lot here: s_g = 0.125 → 67.36, 0.25 → 70.46, 0.5 → 72.53, 4 → 55.16. **A 4× sharper slope costs about 14–17 points under extrapolation.** — [LookHere App. A.6](https://arxiv.org/abs/2405.13985)
- App. A.4.4: at 512² test resolution, tuning a single slope-scaling parameter on minival gains 2.1% for 2D-ALiBi and only 0.15% for LH-45. — [LookHere](https://arxiv.org/abs/2405.13985)

**RMT, Manhattan Self-Attention (Fan et al.; arXiv 2309.11523, CVPR 2024)**
- Spatial decay D_nm = γ^{|x_n−x_m|+|y_n−y_m|}, explicitly based on Manhattan distance and adapted from RetNet. In the paper it is written as Softmax(QKᵀ)⊙D. — [RMT §3.2](https://arxiv.org/abs/2309.11523)
- In the official code the decay is applied **additively to logits before softmax**: `decay = torch.log(1 - 2 ** (-initial_value - heads_range * arange(num_heads)/num_heads))`, `mask = |Δ|.sum(-1) * decay`, `qk_mat = qk_mat + mask`, then softmax. The decay is a `register_buffer`, so it is fixed and not learned. This makes it exactly an L1 ALiBi with per-head λ_h = −ln γ_h. — [RMT code](https://github.com/qhfan/RMT/blob/main/classfication_release/RMT.py)
- Schedule (App. D, Eq. 8): γ_i = 1 − 2^{−a−(b−a)i/N}. Table 12 lists a = 2 for all stages and b = 6–8; code init_values = 2 and heads_ranges = 4–6. The strongest head therefore has γ = 0.75, i.e. **λ ≈ 0.288 per Manhattan step**, and the weakest has γ ≈ 1−2⁻⁸, i.e. λ ≈ 0.004. — [RMT App. D](https://arxiv.org/abs/2309.11523); [RMT code](https://github.com/qhfan/RMT/blob/main/classfication_release/RMT.py)
- Ablation (Table 6): replacing MaSA with plain attention costs 0.8 top-1 (82.4 → 81.6), 2.5 box AP and 2.5 mIoU. Replacing softmax with RetNet's gate gives NaN (unstable). Applying MaSA to DeiT-S gives +1.9 top-1, and to Swin-T +2.3. — [RMT §4.4](https://arxiv.org/abs/2309.11523)

**ViTARC (Li et al.; arXiv 2410.06405, TMLR 2025)**
- 2D-RPE: bias = r·d_Manhattan, with separate slopes for "before" (above/left in raster order) and "after". "The slope values r_before and r_after are derived following the ALiBi setup, forming a geometric sequence of the form 2^(−8/n) for n heads. r_before starts at 1/2¹, while r_after starts at 1/2^0.5" (§4, App. D.1 says "initialized at"). — [ViTARC](https://arxiv.org/abs/2410.06405)
- App. D (50-task ARC subset): 4-diagonal-direction slopes score 79.74 mean and 4 cardinal directions score 78.10, against 79.51 for 2 directions. "Both Manhattan-based configurations generally outperform the strictly cardinal-based approach." — [ViTARC App. D](https://arxiv.org/abs/2410.06405)

**"What DINO saw" (Pawlowsky et al.; arXiv 2603.16840, 2026 preprint)**
- Fine-tunes DINOv2 with ALiBi using Euclidean distance with wrap-around boundaries. "The entries in the ALiBi distance matrix were normalised between 0-1 by dividing by the longest distance". They "used a fixed m = 1 for all head-specific scalars to avoid potential asymmetries compounding across layers". — [DINO-ALiBi §3](https://arxiv.org/abs/2603.16840)
- S2.2: with learned per-head m, PCA maps are less homogeneous than with fixed m = 1. "We hypothesize that this arises because heads learn distinct length scales". No learned values are reported. — [DINO-ALiBi Supp. S2.2](https://arxiv.org/abs/2603.16840)

**Learned relative-bias tables (no slope; initialisation only)**
- Swin: B̂ ∈ R^{(2M−1)×(2M−1)} per head, added after QKᵀ/√d (Eq. 4). Code: `nn.Parameter(torch.zeros(...))` followed by `trunc_normal_(…, std=.02)`, so it starts effectively flat. Table 4: relative bias adds +1.2/+0.8 top-1 over no position encoding or absolute position encoding. — [Swin](https://arxiv.org/abs/2103.14030); [Swin code](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py)
- BEiT: `relative_position_bias_table = nn.Parameter(torch.zeros(num_relative_distance, num_heads))` with num_relative_distance = (2Wh−1)(2Ww−1)+3 (three extra entries for CLS). It is **zero-initialised**; the `trunc_normal_` line is commented out. — [BEiT code](https://github.com/microsoft/unilm/blob/master/beit/modeling_finetune.py); [BEiT](https://arxiv.org/abs/2106.08254)
- iRPE (Wu et al., 2107.14222): four 2-D mappings to buckets: Euclidean, Quantization (undirected), Cross and Product (directed). A piecewise log index g(x) uses α:β:γ = 1:2:8. Table 1 (DeiT-S, 79.9 baseline), bias mode: Euclidean 80.1, Quantization 80.3, Cross 80.5, Product 80.5. Contextual mode: 80.4, 80.5, 80.8, 80.9. Directed mappings beat Euclidean distance. — [iRPE §3–4](https://arxiv.org/abs/2107.14222)
- CoAtNet (2106.04803): "relative attention" adds a scalar w_{i−j} (an input-independent, translation-equivariant static kernel) to the logits before softmax (Eq. 3). — [CoAtNet §2.1](https://arxiv.org/abs/2106.04803)

### Inferences
- In-distribution image classification is insensitive to slope magnitude (LookHere s_g 0.125–4 is within ±0.3 top-1). Strength matters when the grid seen at test time differs from training, and for keeping patch features local (CROMA's collapse result).
- Every fixed-slope 2-D method starts in a regime where the strongest head still spreads mass over several cells (λ ≤ 0.7, giving self-mass ≤12% on L1). λ=4 (93% on self) behaves more like a hard same-cell mask than any published prior. That may be what we want, since geometry tokens on the same cell are the intended match. If so, it is a design choice without published precedent, not an ALiBi convention.
- L1 versus L2 matters little in practice. The most direct evidence is ViTARC, where Manhattan-with-diagonals beats cardinal-only, and RMT, where L1 works well. CROMA and LookHere chose Euclidean distance for rotation invariance. In my calculation the self-mass difference between L1 and L2 at the same λ is small (0.929 vs 0.918 at λ=4).

### Gaps
- CROMA does not report any slope ablation, and the X-ALiBi gain (+0.2%) is not split by slope.
- The CoAtNet initialisation of w_{i−j} is not stated in the paper. The official code is not public; UNVERIFIED.
- iRPE's bias-table initialisation was not checked in code.

---

## Q3. Learned-slope and generalised variants: KERPLE, FIRE, Sandwich, T5, xPos, RetNet, CAPE

### Takeaway
The learned-parameter families use small random initialisations: KERPLE r1 ~ U(0,2), r2 ~ U(0,1), clamped ≥0.01. FIRE uses c=0.1 and L=512 with an MLP. Learned power/linear kernels tend to become very steep, while learned log kernels tend to flatten. FIRE learns both local and "anti-local" heads. No paper reports a table of converged per-head values; there are only plots.

### Cited Findings
**KERPLE (Chi, Fan, Ramadge, Rudnicky; arXiv 2205.09921, NeurIPS 2022)**
- Power variant: q·k + c − r1|m−n|^{r2}, with r1 > 0 and 0 < r2 ≤ 2. Log variant: q·k + c − r1·log(1 + r2|m−n|), with r1, r2 > 0. Both are learned per head (§4 "Practical Choice"). The triangle kernel c−|m−n| recovers ALiBi. — [KERPLE](https://arxiv.org/abs/2205.09921)
- Code initialisation (log variant, `ParallelKerpleLog`): `bias_p` (r1) = `torch.rand(heads)*2`, i.e. U(0,2). `bias_a` (r2) = `torch.rand(heads)*1`, i.e. U(0,1). Both are clamped to min eps = 1e-2 at every forward pass; bias = −bias_p·log(1 + bias_a·diff). The power variant draws its exponent bias_p ~ U(0,2) and coefficient bias_a ~ U(0,1). The `'original'` option loads ALiBi slopes with requires_grad=False. — [KERPLE layers.py](https://github.com/chijames/KERPLE/blob/main/megatron/mpu/layers.py)
- Learned kernels (§5.4, Figs. 2–4, GitHub dataset): "Both ALiBi and its generalized power variant quickly reach a very negative value. In contrast, the log variant successfully discovers several flat kernels, effectively extending the window attention." The Fig. 3 caption notes that the power-variant "y-axis should be multiplied by 1e8, which is a very negative value". — [KERPLE §5.4](https://arxiv.org/abs/2205.09921)
- Position-wise perplexity (§5.5, Fig. 5): "ALiBi lies above KERPLE-log-windowed@512 for almost all the positions, indicating that window attention might be a better choice than ALiBi". KERPLE-log adds "only 24 additional learnable parameters (2 para. for each head)". — [KERPLE §5.5](https://arxiv.org/abs/2205.09921)
- Gaussian-like and multiplicative ("wht") kernel variants were also tried (App., Table 6). §5.3 says the more complex variants give "only… lower training speeds and marginal improvement". — [KERPLE](https://arxiv.org/abs/2205.09921)

**Sandwich and receptive-field analysis (Chi, Fan, Rudnicky, Ramadge; arXiv 2212.10356, ACL 2023)**
- This is the only systematic published slope sweep for ALiBi (Table 1, ArXiv dataset, L_tr=512, slope = 1/2^h):
  - **Shift all h by Δ.** Δ=−3 (slopes 8× larger): perplexity 5.76 at 512 and 7.15 at 8192 (flat). Δ=0 (ALiBi): 5.57 → 5.95. Δ=+2: 5.50 → 5.92. Δ=+6/+8 (slopes 64–256× smaller): 5.70 → 96.0 (explodes).
  - **Same h for all heads**, h = 0, 2, 4, 6, 8 (slopes 1, 0.25, 0.0625, …): perplexity at 512 = 9.45, 6.65, 5.85, 5.60, 5.70; at 8192 = 9.21, 7.08, 8.66, 31.8, 96.0.

  "Only large slopes… allow a model to extrapolate well. Large slopes implicitly enforce a narrow windowed bias". "Slope diversity itself might not be the deciding factor, but… the actual slope value is more important." (§4.1–4.2). — [Sandwich](https://arxiv.org/abs/2212.10356)
- Why linear slopes are awkward (§6): "ALiBi's linear bias lacks the middle ground to learn the averaging behavior: It either decays so fast that distant tokens are masked out or so slow that the ERF becomes much greater than L_tr." Sandwich's fitted log curve equals KERPLE-log with r1 = 0.825, r2 = 1 (§6.3). — [Sandwich](https://arxiv.org/abs/2212.10356)

**FIRE (Li et al.; arXiv 2310.04418, ICLR 2024)**
- b(i,j) = f_θ(ψ(i−j)/ψ(max{L,i})), with ψ(x) = log(cx+1). c and L are learnable; f_θ is a small MLP (hidden size 32) with outputs per head (Eq. 4). — [FIRE §3](https://arxiv.org/abs/2310.04418)
- Reference code (App.): `init_c=0.1`, `init_L=512.` (fixed) times a learned `L_multiplier` initialised at 1.0, and eps = 1e-6. The listing shows a one-hidden-layer MLP, while the text says the default is two hidden layers of 32. This is a minor inconsistency inside the paper. — [FIRE App.](https://arxiv.org/abs/2310.04418)
- Theorem 3.1: FIRE can represent T5 RPE, ALiBi, KERPLE and Sandwich. For ALiBi, ψ is the identity and the threshold is L0. — [FIRE §3](https://arxiv.org/abs/2310.04418)
- Learned biases (§4.5, Fig. 4, C4, query at position 128): "FIRE learns both local and 'anti-local' attention patterns that emphasize far away keys more… as opposed to a fixed local inductive bias (Press et al., 2022; Chi et al., 2022; 2023)". — [FIRE §4.5](https://arxiv.org/abs/2310.04418)
- The FIRE authors' critique: "Alibi, Kerple, and Sandwich significantly bias towards local attention, making it harder to attend to more distant query-key pairs… [which] can prevent the model from capturing long-range dependencies" (§2). — [FIRE](https://arxiv.org/abs/2310.04418)
- Ablation (Table 3, C4 log perplexity at 512/8192): no log transform and no threshold 3.194/3.334; log transform only 3.161/3.085; both 3.149/3.056. — [FIRE App. B.1](https://arxiv.org/abs/2310.04418)

**T5 relative buckets**
- b(i,j) = r_{min(i−j,K)} with learnable scalars r (FIRE's summary of T5). — [FIRE §2](https://arxiv.org/abs/2310.04418)

**xPos (Sun et al.; arXiv 2212.10554)**
- RoPE times an exponential decay per dimension, ζ̂_i = (i/(d/2)+γ)/(1+γ), with γ = 0.4 (Alg. 1). "Numerical optimization methods are tried to find optimal values for ζ. However, the results rely on the initial value… Finally, we find a sub-optimal solution by manually setting γ". The decay is fixed, not learned. — [xPos §3](https://arxiv.org/abs/2212.10554)

**RetNet (Sun et al.; arXiv 2307.08621)**
- Per-head decay γ = 1 − 2^{−5−arange(0,h)}, fixed and identical across layers. This is equivalent to a linear log-bias slope λ_h = −ln γ_h ≈ 2^{−5−h}, so the strongest head has λ ≈ 0.031. For the size-scaling experiments it uses γ = 1 − exp(linspace(log 1/32, log 1/512, h)) (§2.2, §3). — [RetNet](https://arxiv.org/abs/2307.08621)

**Slope rescaling at inference (Al-Khateeb et al.; arXiv 2310.13017)**
- When test length L′ > train length L, the slopes are scaled as m′_j = m_j·(L/L′). The authors report that this improves ALiBi extrapolation. — [PI for ALiBi](https://arxiv.org/abs/2310.13017)

### Inferences
- Learned slopes are not unstable in general, but they **drift**. KERPLE's learned power/linear kernels become extremely steep (×1e8 on the y-axis). That is evidence that, given freedom, many heads push toward very sharp locality. It suggests a learned λ starting at 4 is unlikely to be pulled back toward ALiBi-scale values unless the content signal asks for it. The same KERPLE result also shows that other heads, in the log parameterisation, flatten out.
- KERPLE's U(0,2) and U(0,1) random initialisations across heads are effectively a randomised per-head schedule. FIRE starts with an essentially uninformative MLP. Neither imposes a sharp prior at initialisation.

### Gaps
- No paper I found tabulates converged per-head slope values. KERPLE Figs. 2–4 and FIRE Fig. 4 are plots only, and I could not read numbers off them from the text extraction.
- CAPE (Likhomanenko et al., 2106.03143) is an augmentation of absolute sinusoidal positions (global shift, local shift, scaling), not a distance slope. I did not check its constants.
- Exact T5 defaults (32 buckets, max_distance 128) and its bias initialisation were not verified from the original source in this pass. UNVERIFIED.

---

## Q4. Hybrids of ALiBi-style bias with RoPE

### Takeaway
I found no peer-reviewed, empirically validated RoPE+ALiBi hybrid with reported slopes. The closest relatives are xPos (rotation times exponential decay, which is a log-linear bias) and RMT (RoPE-like θ plus a Manhattan decay inside the retention formulation; the released code uses a 2-D decay mask). One theoretical preprint proposes damping ALiBi when it is combined with RoPE.

### Cited Findings
- "Factored" ALiBi+RoPE (arXiv 2410.18441, single-author theory preprint): "a simple combination of ALiBi and RoPE may lead to the signal from relative position information being too strong as both ALiBi and RoPE uses relative position information". It proposes keeping ALiBi's geometric slopes and adding a harmonic-series factor, but leaves the evaluation as "one of our future directions". — [2410.18441 §3.1.3](https://arxiv.org/abs/2410.18441)
- xPos: RoPE plus a fixed exponential decay (γ = 0.4 in ζ̂). Because ζ^{n−m} enters multiplicatively on q/k, it is not a pure additive slope. — [xPos](https://arxiv.org/abs/2212.10554)
- RMT's RetNet-derived formulation includes rotary θ (Eq. 1, γ^{n−m}(Q_n e^{inθ})(K_m e^{imθ})†). The vision MaSA uses softmax plus the Manhattan decay mask. — [RMT §3.1](https://arxiv.org/abs/2309.11523)
- ViTARC combines its 2-D Manhattan ALiBi (RPE) with a 2-D sinusoidal *absolute* PE and reports that "our results demonstrate a marked improvement", in contrast with Swin's finding that adding APE to RPE slightly hurts. — [ViTARC §4](https://arxiv.org/abs/2410.06405)
- A search snippet says HoPE (arXiv 2509.05218) uses ALiBi-style slopes "set to 96 times the original ALiBi slope values" in a segment-level BiPE-ALiBi variant. UNVERIFIED: I did not open that paper. — [HoPE](https://arxiv.org/abs/2509.05218)

### Inferences
- Our VLM keeps the base model's RoPE on the LLM side and adds the distance bias only in a new cross-attention. That is closer to CROMA's X-ALiBi (pure bias, no RoPE in that sublayer) than to a RoPE+ALiBi hybrid, so the "double relative signal" concern mostly does not apply.

### Gaps
- I found no hybrid paper reporting slope values tuned specifically for coexistence with RoPE.

---

## Q5. Evidence on how sharp the bias should be: too large versus too small

### Takeaway
The evidence is mixed and depends on the task. For 1-D LM extrapolation, larger slopes extrapolate better, but all-heads-equal slope=1 costs a lot in-distribution (perplexity 9.45 versus 5.57). For 2-D vision trained and tested at one resolution, slope magnitude barely matters. When test resolution differs, a 4× sharper global slope costs ~14–17 top-1 points (LookHere). Learned linear/power kernels drift sharper, and learned log kernels flatten. Nothing published starts as sharp as λ=4, which puts 93% of mass on self.

### Cited Findings
- Too sharp, 1-D: all heads at slope 1 (h=0) gives perplexity 9.45 at 512 versus 5.57 for ALiBi; slope 0.25 (h=2) gives 6.65. Uniformly sharp heads hurt in-distribution modelling. — [Sandwich Table 1](https://arxiv.org/abs/2212.10356)
- Too weak, 1-D: slopes 64–256× smaller than ALiBi give perplexity 5.70 at 512 but 96.0 at 8192. Weak slopes fail to extrapolate. — [Sandwich Table 1](https://arxiv.org/abs/2212.10356)
- Sharp is fine as long as other heads are diverse: Δ=−3 (all slopes ×8; strongest head ≈4 if 8 heads, ≈5 if 12 heads; I did not verify the head count of the Sandwich models) gives 5.76 at 512, only 0.19 worse than ALiBi, and is the flattest under extrapolation (7.15). **This is the closest published data point to our λ=4 strongest head**, but the other heads in that run were also ×8 (down to 2⁻⁵). — [Sandwich Table 1](https://arxiv.org/abs/2212.10356)
- Too sharp, 2-D, with a resolution change: LookHere s_g=4 gives 55.16 top-1 at 1024², against 72.53 for s_g=0.5 and roughly 69 for s_g=1. At the training resolution, s_g=4 gives 80.91 versus 81.2 for 0.125 (negligible). — [LookHere Tables 12–13](https://arxiv.org/abs/2405.13985)
- No bias at all is also worse in 2-D: removing distance penalties gives 80.19 versus ~81.0 in-distribution (LookHere). MaSA→plain attention gives −0.8 top-1 and −2.5 AP (RMT). 2D-ALiBi also limits patch representational collapse (CROMA). — [LookHere](https://arxiv.org/abs/2405.13985); [RMT](https://arxiv.org/abs/2309.11523); [CROMA](https://arxiv.org/abs/2311.00566)
- The linear shape is a compromise. Linear penalties "either decay so fast that distant tokens are masked out or so slow that the ERF becomes much greater than L_tr" (Sandwich §6). In 2-D, LookHere found dist² and √dist worse than linear for extrapolation. — [Sandwich](https://arxiv.org/abs/2212.10356); [LookHere](https://arxiv.org/abs/2405.13985)
- Models can compensate for slope scale: "models can learn to scale attention logit magnitudes" (LookHere §3). — [LookHere](https://arxiv.org/abs/2405.13985)

### Inferences
- **On λ=4 as an init.** The published fixed-slope schedules have strongest heads of 0.29 (RMT), 0.5–0.71 (ALiBi/CROMA/ViTARC) and up to 1.5 (LookHere layer 1). λ=4 is 3–14× sharper than all of them. In our setting the queries and keys are spatially co-registered, and same-cell is the intended match (as in CROMA's X-ALiBi), so a sharp prior is defensible. The risk is that softmax saturates at 93–99.7% on self or a 1-ring, which gives tiny gradients for content logits to overrule it. LookHere's argument that the model can rescale logits assumes the content logits can grow to ~λ·Δd, i.e. ~4 per cell for us.
- **Per-head schedule.** Every family surveyed uses per-head diversity: geometric (ALiBi, CROMA, ViTARC, RMT, RetNet) or random (KERPLE). Sandwich shows that absolute magnitude matters more than diversity per se, but all of their best runs were still diverse. A schedule such as λ_h = 4·2^{−k·h} (e.g. 4, 2, 1, 0.5, …), or a geometric span with 4 as the sharpest head, would match published practice better than a uniform λ=4.
- **Learn or fix?** For a fixed grid with no extrapolation, the learned-slope evidence is neutral to slightly positive: LookHere s_l learned gives 81.35 versus ~81.0; DINO-ALiBi says heads learn distinct length scales. ALiBi's negative result was only about length extrapolation. Parameterising λ = softplus(ρ) or exp(ρ) with a clamp (KERPLE clamps at ≥0.01) is standard practice.

### Gaps
- No paper reports converged learned slope values for a 2-D or cross-attention distance bias. Our run would be the first data point. Logging λ_h and μ_h per layer over training would answer this directly.
- No ablation varies the slope of a cross-modal (X-ALiBi-style) bias specifically.
