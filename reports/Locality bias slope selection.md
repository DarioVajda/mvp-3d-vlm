# Spread the slopes, then let them move

The draft plan has a more basic problem than its choice of 4: **at lr 1e-5, AdamW can move a scalar by at most about 0.05 in 5,000 steps, so λ and μ are frozen at their initial values in practice**. The run would test a fixed, uniform, unusually sharp prior, not a learned one. No published locality bias starts near λ = 4. The sharpest fixed slopes in 2-D vision are 0.29 to 1.5 per grid step (RMT, ALiBi/CROMA, LookHere), and every family that works (ALiBi, RetNet, RMT, CROMA, ViTARC, KERPLE) spreads heads geometrically or randomly instead of giving them all one value. ConViT shows that a strong additive positional init can trap attention in the local configuration even when its strength parameter is trainable. On the real 20×15 grid, λ = μ = 4 puts **82.7% of each query's attention on its own cell at T = 8, which falls to 59.5% at T = 32**, because 24 extra frames each take e⁻⁴ of the mass. The aligned grid does justify keeping one or two sharp "anchor" heads. The recommended configuration (§5) is therefore: a 4×4 factorial spread of λ ∈ {4, 1.41, 0.5, 0.18} by μ ∈ {6, 4, 2, 0}; λ learned in log space; both scalars in their own AdamW group at lr 1e-3 with no weight decay; a fixed +ln((T−1)/7) offset on μ, which adds 1.49 at T = 32; no rotary frequency on the time axis; and the bias folded exactly into extra Q/K channels so Flash or cuDNN kernels compute the λ/μ gradients. Most of these choices extend published evidence by derivation rather than copying a tested recipe. The final section separates the two.

## Published slopes stop at 1.5; λ = 4 is a hard mask in disguise

Every fixed-slope locality bias in the literature is far weaker than the draft's.

**ALiBi** uses the fixed geometric sequence 2^(−8i/H). For 16 heads that runs from 2^−0.5 ≈ 0.71 down to 2^−8 ≈ 0.004. The authors found it with "a brief manual exploration of around ten slope sets", concluding that the best sets lie in (0, 1) and get denser toward 0 ([ALiBi](https://arxiv.org/abs/2108.12409)). The 2-D methods copy this sequence:

- **CROMA's X-ALiBi**, the closest precedent to our layer (cross-attention between two spatially co-registered modalities, biased by grid distance), uses exactly the ALiBi 16-head slopes and never learns them ([CROMA](https://arxiv.org/abs/2311.00566)).
- **ViTARC** uses the same sequence for its Manhattan bias ([ViTARC](https://arxiv.org/abs/2410.06405)).
- **RMT**, the only Manhattan-distance vision backbone, applies a fixed additive decay with **λ ≈ 0.288 on its strongest head** and about 0.004 on its weakest ([RMT code](https://github.com/qhfan/RMT/blob/main/classfication_release/RMT.py)).
- **LookHere** goes up to 1.5 in its first layer ([LookHere](https://arxiv.org/abs/2405.13985)).

The softer priors are wider still. ConViT's default locality strength α = 1 on a squared distance puts roughly a third of the positional softmax on the centre cell ([ConViT](https://arxiv.org/abs/2103.10697)). SMCA chose its Gaussian bandwidth explicitly "to ensure G covering a large spatial range at the beginning of training so that the network can receive more informative gradients" ([SMCA](https://arxiv.org/abs/2101.07448)).

On our grid these slopes look like this. The table gives bias-only attention for one frame (content logits set to 0), averaged over all 300 query positions of the finite 20×15 grid (my computation):

| λ | Own-cell mass | Mean attended L1 distance | Published anchor |
|---|---|---|---|
| 4.0 | **0.933** | 0.07 | none (draft) |
| 2.83 | 0.801 | 0.22 | — |
| 2.0 | 0.601 | 0.51 | — |
| 1.41 | 0.396 | 0.96 | LookHere layer-1 scale ≈ 1.5 |
| 0.71 | 0.139 | 2.31 | ALiBi/CROMA 16-head strongest |
| 0.5 | 0.079 | 3.27 | ALiBi 8-head strongest |
| 0.29 | ≈0.035 | ≈5 | RMT strongest |
| 0.18 | 0.017 | 6.96 | — |
| 0.022 | 0.004 | 10.9 | ALiBi weakest (≈ uniform) |

At λ = 4 a one-step neighbour gets e⁻⁴ ≈ 1.8% of the own-cell weight, so the bias behaves like a same-cell mask. That is a defensible design choice, since the geometry token at the same (t, r, c) is the intended match. It is not an ALiBi convention, and the closest evidence on starting that sharp is cautionary:

- **Uniform sharp slopes hurt.** In Sandwich's slope sweep, giving all heads slope 1 raised perplexity from 5.57 to 9.45 at the training length. Shifting every ALiBi slope up 8× while keeping them diverse cost only 0.19 ([Sandwich](https://arxiv.org/abs/2212.10356)). The damage came from uniformity, not from sharpness.
- **A strong additive prior can trap content.** ConViT's appendix reports that an additive, pre-softmax positional prior with convolutional init "stays in the convolutional configuration and ignores the content information", and that DeiT overtakes it late in training. This happened even though its strength parameter was trainable ([ConViT](https://arxiv.org/abs/2103.10697)). ConViT's fix was to mix the positional softmax outside the content softmax through a sigmoid gate initialised at 1, "to avoid the ConViT staying stuck".

The mechanism follows from the softmax Jacobian (my derivation). The gradient to λ_h is −Cov_p(dist, u), where u is the upstream signal per key. At 93% own-cell mass the variance of distance under p is close to zero, so a saturated head barely learns to widen. Content-logit gradients for the neighbours are also scaled by p ≈ 0.018. Zhai et al. tie low attention entropy to training instability, but through large q·k logits, which produce *high* gradient norms ([Zhai et al.](https://ar5iv.labs.arxiv.org/html/2303.06296)). That is a different failure from the quiet lock-in a fixed bias produces.

Two pieces of evidence argue against overcorrecting toward flat priors:

- **Removing the bias hurts.** LookHere's no-distance variant loses about 0.8 top-1, RMT's plain-attention ablation costs 0.8 top-1 and 2.5 box AP, and CROMA attributes reduced patch representational collapse to the local bias injected "at every ViT layer" ([LookHere](https://arxiv.org/abs/2405.13985); [RMT](https://arxiv.org/abs/2309.11523); [CROMA](https://arxiv.org/abs/2311.00566)).
- **Magnitude matters little in-distribution.** LookHere's global-slope sweep from 0.125 to 4 moves 224² top-1 by only 0.3 points (81.20 to 80.91) ([LookHere](https://arxiv.org/abs/2405.13985)).

Taken together, the right move is to **keep λ = 4 on a minority of anchor heads and spread the rest geometrically**, as every successful family does. One spacing that does this is the ALiBi 16-head ratio of 2^−0.5 shifted up by 5.66×, which lands close to Sandwich's well-behaved 8× shift.

## At lr 1e-5 the 96 scalars are frozen, and decay would bias them

Adam's per-step update is bounded by roughly the step size regardless of gradient scale ([Adam](https://arxiv.org/abs/1412.6980)). The movement budget over 5,000 steps (my derivation) is therefore:

| lr | Maximum drift (consistent gradient sign) | Pure-noise random walk |
|---|---|---|
| 1e-5 | **0.05** | 0.0007 |
| 1e-4 | 0.5 | 0.007 |
| 1e-3 | **5** | 0.07 |
| 1e-2 | 50 | 0.7 |

The draft's lr can take λ from 4 to 3.95 at best. lr 1e-3 is the smallest value that allows a full move to "locality off" or to a much sharper head, while keeping noise-driven drift below 0.1. Published learned-bias models never face this problem because they train the scalars at a from-scratch base lr: KERPLE uses plain Adam at 6e-4 and 2e-4 with no special group ([KERPLE](https://arxiv.org/abs/2205.09921)). None reports a separate lr for bias scalars during low-lr fine-tuning, so the 1e-3 figure is derived, not copied.

**Parameterisation.**

- **λ in log space.** Use λ = exp(θ), as Mamba stores its decay in `A_log` ([Mamba](https://github.com/state-spaces/mamba)). Steps are then relative, positivity is free, and a single lr gives heads that differ 20× in scale the same mobility. This matches the geometric spacing of ALiBi and RetNet ([RetNet](https://arxiv.org/abs/2307.08621)).
- **μ as a raw scalar.** μ spans 0 to 6 and zero is meaningful, so a raw unconstrained scalar is the simpler choice. A slightly negative μ would just make a head prefer other frames; FIRE finds learned "anti-local" heads useful ([FIRE](https://arxiv.org/abs/2310.04418)).
- **No clamp on λ.** KERPLE clamps its parameters at ≥ 0.01 in the forward pass ([KERPLE code](https://github.com/chijames/KERPLE/blob/main/megatron/mpu/layers.py)), but a clamp zeroes the gradient at the boundary, and exp(θ) needs none.

**Weight decay must be zero for this group.** AdamW at lr 1e-3 and wd 0.1 multiplies a parameter by (1 − 10⁻⁴)^5000 ≈ 0.61 regardless of data, pulling raw μ toward 0 and log-λ toward 1 (my derivation). Swin excludes its relative-bias table from decay and the Mamba reference marks `A_log` no-decay for the same reason ([Swin](https://github.com/microsoft/Swin-Transformer); [Mamba](https://github.com/state-spaces/mamba)).

**Zero-initialised W_o adds a start-up delay.** With W_o = 0, every upstream parameter, including λ and μ, gets exactly zero gradient at step 0. From then on their gradients scale with ‖W_o‖, which grows by about the lr per step (my derivation, the same argument ControlNet makes for zero convolutions). Two consequences follow:

- **Hold λ and μ frozen for the first ~300 steps**, then warm their lr up over 200 steps, so the first noisy, W_o-starved gradients cannot push them around.
- **Lower Adam ε to 1e-10 for this group**, because gradients this small can sit near the default 1e-8 floor.

There is also a run-length risk for the whole module. ControlNet's zero-init branch at a similar lr showed a "sudden convergence" at **6,133 steps** ([ControlNet](https://ar5iv.labs.arxiv.org/html/2302.05543)), longer than this entire run. A flat dev curve at step 3,000 is therefore not yet evidence that geometry fails. Flamingo-style and LLaMA-Adapter-style gates (tanh(g) with g = 0 and W_o at its normal init) avoid the deadlock, because the gate itself gets gradient at step 0 ([Flamingo](https://arxiv.org/abs/2204.14198); [LLaMA-Adapter](https://ar5iv.labs.arxiv.org/html/2303.16199)).

**Learning the slopes is worth it, but the evidence for a gain is weak, so keep a fixed-slope control.**

- ALiBi found trainable slopes extrapolated worse and trained 3% slower. That result concerns 1-D length extrapolation, which is not our spatial setting ([ALiBi](https://arxiv.org/abs/2108.12409)).
- In-distribution, LookHere's learned layer slope gained about 0.35 top-1 ([LookHere](https://arxiv.org/abs/2405.13985)), and learned versus fixed windows in NMT differed by 0.07 to 0.23 BLEU ([Yang et al.](https://arxiv.org/abs/1810.10182)).
- Learned slopes drift in both directions. KERPLE's learned power kernels went extremely steep while its log kernels flattened ([KERPLE](https://arxiv.org/abs/2205.09921)), and DINO-ALiBi reports heads learning "distinct length scales" ([DINO-ALiBi](https://arxiv.org/abs/2603.16840)).

The init spread is where most of the published benefit sits. Learning is cheap insurance on top of it, not a substitute for it.

## Frame count, not grid size, is the axis that extrapolates

Training and evaluation use the same 20×15 grid, so λ never extrapolates. That removes the one setting where slope magnitude matters strongly in 2-D. In LookHere, a 4× sharper global slope cost **about 17 points at 1024² test resolution (55.16 vs 72.53)**, but only 0.3 points at the training resolution ([LookHere](https://arxiv.org/abs/2405.13985)). The quantity that changes between train and eval is the number of other-frame keys. That is the dilution problem SSMax addresses: under plain softmax the maximum attention weight decays toward 0 as n grows unless logit gaps grow by about log n ([SSMax](https://arxiv.org/html/2501.19399v1); [Chiang & Cholak](https://arxiv.org/abs/2202.12172)).

Because the frame term is a same/other-frame indicator rather than a distance in Δt, the fix is exact when content logits are neutral (my derivation). Each other frame contributes e^−μ times the own-frame spatial mass, so the own-frame fraction is 1/(1 + (T−1)e^−μ). Parameterising μ = m + ln((T−1)/7) keeps it identical for any T. The offset is 0 at T = 8 and **ln(31/7) = 1.49 at T = 32**. On the real grid at λ = μ = 4, own-frame mass is 0.886 at T = 8, drops to **0.638 at T = 32** without the offset, and returns to exactly 0.886 with it. λ needs no correction, since the spatial term acts within each frame. I found no paper that normalises a cross-frame or cross-view bias by frame count, so this correction is unpublished. Its justification is the SSMax log-n argument applied to one axis.

The other ways the video literature handles frame-count change are rescaling the temporal position index (VideoRoPE's δ, HoPE's γ ∈ {0.5…1.5}) and training on varied lengths ([VideoRoPE](https://arxiv.org/html/2502.05173v3); [HoPE](https://arxiv.org/html/2505.20444v1)). Sampling T between 4 and 16 during training would let m learn across key counts and is a reasonable ablation.

**The spatial-to-temporal ratio follows from the bias form, not from any paper** (my derivation). The logit gap between a same-frame neighbour at Manhattan distance d and the co-located cell in another frame is μ − λd. At λ = μ = 4, the cell one step away in the current frame ties with the same pixel in every other frame. Because the camera moves, the same pixel in another frame is generally a different 3-D point, so anchor heads should have **μ ≥ 1.5λ**. The literature gives no hand-set ratio to compare against:

- Video Swin inflates its 2-D bias table along time, so every temporal offset starts with the spatial value and **no extra temporal penalty** ([Video Swin code](https://raw.githubusercontent.com/SwinTransformer/Video-Swin-Transformer/master/mmaction/models/backbones/swin_transformer.py)).
- MViTv2 zero-initialises separate learned t/h/w tables ([PySlowFast](https://raw.githubusercontent.com/facebookresearch/SlowFast/main/slowfast/models/attention.py)).
- VGGT separates frames architecturally, alternating frame-wise and global attention. It has no frame penalty at all, yet generalises from at most 24 training frames to hundreds ([VGGT](https://arxiv.org/html/2503.11651v1)). Its geometry tokens are therefore already cross-frame consistent, which is why low-μ heads should exist alongside high-μ ones rather than one compromise value.

**The 16/24/24 rotary split is the weakest part of the temporal design.** It is Qwen2-VL's `mrope_section`, and in that layout time takes the 16 highest-frequency pairs ([Qwen2-VL config](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/raw/main/config.json)). VideoRoPE shows these channels alias distant frames ("hash collisions") and moves time to the lowest frequencies. It improves LongVideoBench at 64k from 54.35 to 57.26 and V-NIAH-D from 74.67 to 87.11 ([VideoRoPE](https://arxiv.org/html/2502.05173v3)). HoPE goes further and sets the temporal frequencies to zero, and beats both on V-NIAH (63.56 vs 52.00 vs 47.11) ([HoPE](https://arxiv.org/html/2505.20444v1)). Qwen3-VL abandons the contiguous layout for an interleaved [24, 20, 20] ([Qwen3-VL config](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/raw/main/config.json)).

In this adapter, μ already encodes same-frame versus other-frame, and VGGT keys are in a shared world frame. Following HoPE, the recommendation is **no rotary frequency on t inside the adapter**. That leaves frame order to the LLM's own M-RoPE.

A further problem appears when the Qwen2-VL frequencies are computed for this grid (my computation, assuming the sequential layout and Qwen's base of 10⁶):

- **Columns:** a maximum phase of **0.0034 rad** across all 20 columns, so they carry essentially no rotary signal.
- **Rows:** at most 0.44 rad.
- **Time:** the slowest pair moves from 0.28 rad at Δt = 7 to 1.22 rad at Δt = 31, a range training never sees.

With a base of 10⁴ the columns still reach only 0.06 rad. This is why RoPE-ViT uses **base 100** for axial 2-D RoPE ([RoPE-ViT](https://arxiv.org/html/2403.13298v2)). The adapter's base should be checked in code. Whatever it is, give r and c 32 pairs each on their own base-100 ladder (at base 100, all 24 row pairs and 11 of 24 column pairs already exceed 0.5 rad over the grid). The "double relative signal" concern about combining RoPE and ALiBi comes from an unevaluated theory preprint ([2410.18441](https://arxiv.org/abs/2410.18441)). Here the two encode different things: λ is an isotropic prior, and RoPE allows directional, content-gated offsets.

## Exact gradients at Flash speed by folding the bias into Q and K

There are four ways to get gradients to λ and μ in PyTorch 2.14.1. They differ mainly in memory and in which kernels can run.

**1. Dense float mask through SDPA.** This is the obvious route.

- SDPA adds a float `attn_mask` to the scores ([SDPA docs](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)).
- The installed binary's Flash backend rejects any non-null mask.
- The memory-efficient backend computes `attn_bias.grad` but needs aligned, contiguous strides (Hugging Face forces contiguity for [pytorch#127523](https://github.com/pytorch/pytorch/issues/127523)).
- cuDNN supports dBias, with restrictions below cuDNN 9.5.
- Cost (my arithmetic): the bias is 16 × 2,400 × 2,400 in bf16, **0.18 GB per sample per layer at T = 8**, plus an equal-sized gradient. At T = 32 evaluation, with no gradient, it is **2.95 GB per sample per layer**.

**2. FlexAttention `score_mod` with captured scalars.** This avoids materialising the bias entirely ([FlexAttention blog](https://pytorch.org/blog/flexattention/)). The installed torch 2.14.1 source supports gradients to tensors captured by `score_mod` (not `mask_mod`). However, it accumulates them with atomic adds (`torch/_inductor/kernel/flex/common.py`), and its flash/CuTe backend refuses grad-requiring buffers, so it falls back to Triton. Contention when 2,400² score elements all reduce into 16 scalars has not been benchmarked anywhere I found.

**3. Fold the bias into extra Q/K channels (recommended).** This option is exact and fast (my derivation; checked numerically on the full T = 8 grid). The L1 distance on a discrete grid is an inner product of one-hot codes: |r − r′| = onehot(r)ᵀ M_r onehot(r′), with M_r the 15×15 matrix of |i − j|, and likewise for columns with 20×20. The frame indicator satisfies −μ[t ≠ t′] = μ·onehot(t)·onehot(t′) − μ, and the constant cancels in softmax. Construct:

- q′ = [q_rope/√d, −λ_h M_r[r_q], −λ_h M_c[c_q], μ_h onehot(t_q)]
- k′ = [k_rope, onehot(r_k), onehot(c_k), onehot(t_k)]

and call SDPA with `scale=1.0` and **no mask**. The full logit matrix then differs from the target by exactly μ per row, which the softmax ignores. Properties:

- **Size:** 15 + 20 + T extra channels, i.e. 43 at T = 8 and 67 at T = 32, taking head_dim 128 to 176 or 200 after padding to a multiple of 8. That is within Flash's 256 limit, with V zero-padded if the backend insists on equal head dims.
- **Gradients:** λ and μ enter only through q′, so ordinary autograd gives their exact gradients at O(N) memory, through the fastest kernels.
- **Precision:** the worst bf16 error is about 0.25 nats on cells whose weight is already e⁻⁷⁶.
- **Generality:** the same construction covers any separable per-axis distance, not only L1. It is not specific to squared Euclidean distance, which is where the research notes first found the trick.

**4. Math backend as the reference.** Keep the dense-mask path as the correctness reference. Compare λ and μ gradients against the fp32 math backend on a small batch before the first real run, and pin the backend with `sdpa_kernel` so nothing falls back silently to O(N²) math.

## A concrete configuration, its controls, and what rests on what

**Recommended configuration** (16 heads; head h = 4i + j, with λ index i and μ index j; the same layout at all 3 injection points):

| Setting | Draft | Recommended |
|---|---|---|
| λ init per head | 4 everywhere | λ_i ∈ {**4, 1.41, 0.5, 0.177**} (ratio 2^−1.5) |
| μ init per head | 4 everywhere | μ_j ∈ {**6, 4, 2, 0**} |
| Bias-only start, T = 8 | own-cell 0.83 on all heads | own-cell from 0.92 (λ4/μ6) to 0.002 (λ0.18/μ0); own-frame 0.98 / 0.89 / 0.51 / 0.125 by μ column |
| λ parameterisation | raw | λ = exp(θ), θ₀ = ln λ₀, no clamp |
| μ parameterisation | raw | μ = m + ln((T−1)/7), m raw and unconstrained, m₀ = μ₀ |
| Optimiser group {θ, m} | new-module lr 1e-5 | AdamW lr **1e-3**, wd **0**, ε 1e-10, β (0.9, 0.999); frozen for steps 0–300, linear warmup to step 500, then the main cosine decay |
| Evaluation at T = 32 | μ unchanged | offset applied automatically (+1.49) |
| Temporal RoPE | 16 highest-frequency pairs | **zero frequency on t**; r and c get 32 pairs each, base 100 |
| Kernel | unspecified | augmented Q/K channels + Flash/cuDNN; dense-bias memory-efficient path as reference |
| Output | W_o = 0 | keep, but log ‖W_o‖; if the module has not switched on by ~3k steps, use a per-head tanh gate with W_o at normal init |

**Ablations, in priority order.** The first three share a budget and answer the main question. The fourth costs nothing because it reuses checkpoints.

| # | Arm | Question it answers |
|---|---|---|
| A0 | Draft: uniform λ = μ = 4 at lr 1e-5 | baseline; in effect fixed and uniform |
| A1 | Recommended | — |
| A2 | A1 spread, fixed (lr 0) | does learning add anything beyond the spread? |
| A3 | A1 evaluated at T = 32 with and without the ln(31/7) offset | is the frame-count correction right? (eval-only) |
| A4 | Temporal RoPE: none vs Qwen2-VL high-frequency vs VideoRoPE low-frequency | does the t-axis layout matter at 32 frames? |
| A5 | λ term applied only within frame (cross-frame keys get μ only) | is the "same pixel, other frame" prior harmful under camera motion? |
| A6 | Train with T sampled from {4, …, 16} | HoPE-style robustness to frame count |

**Diagnostics to log** (per layer and head, on a fixed dev batch, at T = 8 and at T = 32 with and without the offset):

- λ_h and μ_h trajectories.
- |∂L/∂θ|, |∂L/∂m| and ‖W_o‖ for the first 500 steps, then every 100 steps.
- Own-cell mass, own-frame mass, attention entropy, and ConViT's nonlocality metric (attention-weighted mean L1 distance) ([ConViT](https://arxiv.org/abs/2103.10697)).
- The fraction of queries with top-1 mass above 0.99, which detects saturation.
- The ratio of the content-logit standard deviation (q·k/√d) to the bias standard deviation across keys. This shows whether content can ever override the prior.
- Whether query i's top-1 key is its own cell.
- A one-time gradient check of the augmented-channel path against the fp32 math backend.

**What rests on evidence versus derivation:**

| Recommendation | Basis |
|---|---|
| Per-head geometric or diverse slopes instead of one value | **Published**: ALiBi, RetNet, RMT, CROMA, ViTARC, KERPLE; Sandwich shows uniform-sharp hurts |
| Keep the strongest head sharp (≈4) | **Partly published**: Sandwich's 8× shift was benign. The aligned-cell rationale and the exact value are derived |
| Strong additive priors can lock out content | **Published**: ConViT App. A |
| Fixed-slope control arm; learning gives small gains | **Published**: ALiBi, LookHere, Yang et al. |
| lr 1e-3 for the scalars, freeze then warm up, ε 1e-10 | **Derived** from Adam's step bound and the W_o = 0 gradient algebra; no paper reports this setting |
| Log-space λ, zero weight decay | **Practice** (Mamba `A_log`, Swin bias table) plus derivation |
| μ + ln((T−1)/7) at evaluation | **Derived** by analogy to SSMax log-n scaling; unpublished for cross-frame biases |
| μ ≥ 1.5λ on anchor heads | **Derived** from the bias form; no published ratio |
| Zero temporal RoPE frequency | **Published** for LLM self-attention (HoPE, VideoRoPE); untested inside a cross-attention adapter |
| Base-100 per-axis RoPE for r and c | **Published** (RoPE-ViT); the near-zero column phase under base 10⁶ is my computation |
| Q/K channel augmentation | **Derived** and numerically verified on the grid; not benchmarked on GPU |
| Kernel restrictions (Flash rejects masks, Flex atomics) | **Verified** in the installed torch 2.14.1 binary and source |

## Conclusion

The question "what should λ be?" turns out to be three separate decisions. The init spread carries most of the published signal. The optimiser group decides whether "learned" means anything at all. The frame-count correction is the only point where train/eval mismatch can quietly move about 25 points of attention mass, from 0.886 to 0.638 own-frame. The draft's single value of 4 is not wrong for an anchor head, but as a uniform setting at lr 1e-5 it turns an intended learned prior into an untested hard mask. That would also confound any null result with a module that never switched on, as ControlNet's 6,133-step switch-on suggests can happen.

Two findings are new rather than inherited. First, the L1 grid bias folds exactly into Q/K, which makes learning λ and μ essentially free at Flash speed. Second, Qwen2-VL's sequential rotary layout leaves this grid's column axis with essentially no positional signal, so in the current design λ is doing all the spatial work. Both should be checked numerically before the first run. The learned λ_h and μ_h trajectories from A1 would be the first published data on where cross-modal, co-registered locality slopes converge.
