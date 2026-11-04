# Learned additive attention biases: optimisation and implementation

Setting under study: cross-attention logits `z = q·k/√d − λ_h·dist − μ_h·[other frame]`, 96 scalars (16 heads × 3 layers × {λ, μ}) initialised to 4, AdamW lr 1e-5 (new-module lr; LoRA at 1e-4), ~5,000 steps, output projection W_o zero-initialised, N = 2,400 keys at training and 9,600 at evaluation, implemented with `F.scaled_dot_product_attention` (SDPA).

Provenance key: **[verified]** means checked this session against the paper/docs or the installed PyTorch 2.14.1 source (`mvp/.venv/lib/python3.10/site-packages/torch`). **[recalled, not re-verified]** means the claim comes from memory of the source and was not re-fetched this session. **[derivation]** means my own algebra, not taken from a source.

---

## Q1. Zero-initialised W_o: what gradient do λ, μ, W_q, W_k get early on, and do zero-init papers report slow starts or recommend a different init or lr for gate-like scalars?

### Takeaway
With `out = W_o · Attn(...)` and `W_o = 0`, every parameter upstream of W_o (λ, μ, W_q, W_k, W_v) gets **exactly zero** gradient at step 0. Only W_o gets a gradient. From step 1 onwards their gradients are proportional to ‖W_o‖, which Adam's first step makes about lr = 1e-5 per entry. Adam's per-parameter normalisation largely removes the *magnitude* problem: tiny but consistent gradients still produce updates of about lr per step. The *lr cap* (≤ about 1e-5 per step) is the binding constraint, not the zero-init. Zero-init papers (ControlNet, LLaMA-Adapter, ReZero, Flamingo) all report that this works. ControlNet, however, documents a "sudden convergence" only after about 6k steps, which is longer than our whole 5k-step run.

### Cited Findings
- ControlNet says zero convolutions "protect this backbone by eliminating random noise as gradients in the initial training steps". It puts the gradient derivation for W = 0 in the supplement (not read here). It reports a "sudden convergence phenomenon": the model "abruptly succeeds in following the input conditioning image; usually in less than 10K optimization steps", and Fig. 4 marks the switch at **6133 steps**. The mechanism is not explained. — [ControlNet, Zhang et al. 2023, arXiv:2302.05543](https://ar5iv.labs.arxiv.org/html/2302.05543) [verified]
- LLaMA-Adapter uses a learnable gate g_l "initialized by zero", applied as `tanh(g_l)` times the softmax over the adaption-prompt scores only (word-token softmax untouched). There is a separate g_l **per attention head**. In the ablation on ScienceQA val, zero-init gives 83.85% vs 40.77% for random init, and the random-init loss "slowly approaches 0.15". — [LLaMA-Adapter, Zhang et al. 2023, arXiv:2303.16199](https://ar5iv.labs.arxiv.org/html/2303.16199) [verified]. The adapter learning rate (I recall about 9e-3 for all adapter params) is [recalled, not re-verified].
- Flamingo's gated xattn-dense layers multiply the new cross-attention output by `tanh(α)` with α initialised at 0, so the frozen LM is unchanged at init. — [Flamingo, Alayrac et al. 2022, arXiv:2204.14198](https://arxiv.org/abs/2204.14198) [recalled, not re-verified]
- ReZero uses `x + α·F(x)` with α = 0 at init. It reports faster convergence than LayerNorm transformers (I recall about 56% faster on enwik8). — [ReZero, Bachlechner et al. 2020, arXiv:2003.04887](https://arxiv.org/abs/2003.04887) [recalled, not re-verified]
- LoRA+ analyses the same "one factor zero-initialised" structure as LoRA (B = 0, A random). It argues that equal learning rates for A and B are inefficient and recommends a larger lr for B (ratio around 16). — [LoRA+, Hayou et al. 2024, arXiv:2402.12354](https://arxiv.org/abs/2402.12354) [recalled, not re-verified]
- In Adam the effective step is approximately bounded by the step size α: `|Δθ| ≲ α` per step, independent of gradient scale. This is the "scale invariance" property. — [Adam, Kingma & Ba 2014, arXiv:1412.6980, §2.1](https://arxiv.org/abs/1412.6980) [recalled, not re-verified; pre-2019 but foundational]

### Inferences
- **[derivation] Exact zero at step 0.** Write `y = W_o h`, `h = Σ_j p_j v_j`, `p = softmax(z)`. Then `∂L/∂h = W_oᵀ ∂L/∂y = 0` when W_o = 0, so ∂L/∂z = 0 and ∂L/∂{λ, μ, W_q, W_k, W_v} = 0. The only non-zero gradient is `∂L/∂W_o = (∂L/∂y) hᵀ`, which is non-zero because h ≠ 0. This is the ControlNet argument.
- **[derivation] Step 1 onwards.** Adam's first bias-corrected update is `lr·g/(|g|+ε)`, which is about ±lr per element. So after step 1, every entry of W_o has magnitude about 1e-5, and it grows roughly linearly while its gradient sign is consistent. Upstream gradients scale with W_o, so they are tiny at first: roughly 1e-5·t times the usual magnitude. Two consequences:
  1. Adam rescales tiny gradients to about lr-sized steps **only if |g| ≫ ε** (default ε = 1e-8). λ/μ gradients are a product of a very small W_o, a small residual-stream gradient and a covariance (see Q2). In the first tens to hundreds of steps they may be comparable to ε, which damps updates further. **Recommendation: log |grad λ|, |grad μ| and ‖W_o‖ for the first ~200 steps.** If |g| is below about 1e-7, lower ε (for example 1e-10) for this param group.
  2. Early gradients for λ, μ, W_q, W_k are computed through a near-random W_o, so their direction is noisy. Adam's second moment v (β2 = 0.999, about 1000-step memory) is dominated by these early tiny values and then lags the growth of |g|. Combined with the bias correction this gives *larger* than lr steps for a while (m/√v > 1 when g is growing). That is mildly helpful.
- **Practical options** (none is standard for this exact configuration; pick by ablation):
  - (a) Keep W_o = 0 but give W_o a higher lr. This follows LoRA+ logic: the zero-init factor needs to grow fast so the rest gets signal.
  - (b) Init W_o with a small non-zero std (for example 1e-3 to 1e-4 times the default). This breaks the exact-zero deadlock while keeping the residual perturbation negligible.
  - (c) Use a Flamingo/LLaMA-Adapter-style scalar gate `tanh(g)` with g = 0 and W_o at its normal init. Then g is the only parameter that is exactly zero-grad-blocked. g itself gets a non-zero gradient at step 0 because `d tanh/dg = 1` at 0 and the attention output is non-zero, and gate-scalars are usually trained at the adapter lr.
- ControlNet's ~6k-step "sudden convergence" at lr 1e-5 (ControlNet's own lr; [recalled]) is a warning sign. A 5k-step run with zero-init output and lr 1e-5 may end **before** the new module has meaningfully switched on. Check this on dev curves before interpreting a null result.

### Gaps
- I did not read ControlNet's supplementary gradient derivation or ReZero's/Flamingo's exact text this session.
- I found no paper that specifically measures how slowly *attention-logit* parameters (as opposed to output-gate parameters) move when the output projection is zero-initialised.

---

## Q2. Saturation: if a bias puts >80% of softmax mass on one key, how do gradients to content logits (W_q, W_k) and to λ, μ behave? What do entropy-collapse analyses say?

### Takeaway
Under saturation the softmax Jacobian shrinks the gradient to every *non-dominant* key's logit by its tiny probability. The gradient to λ_h (or μ_h) is minus a **covariance under p** between the bias feature and the upstream signal, and it vanishes as attention becomes one-hot. A strong bias therefore partly self-locks. Zhai et al. show that *low-entropy attention from large logits* correlates with instability and **high** gradient norms, not vanishing ones. That is a different mechanism from a fixed additive bias.

### Cited Findings
- Zhai et al. define "attention entropy collapse" as "pathologically low attention entropy, corresponding to highly concentrated attention scores". They report that "low attention entropy is accompanied by high training instability" (an empirical correlation; causality is an open question). — [Zhai et al. 2023, "Stabilizing Transformer Training by Preventing Attention Entropy Collapse", arXiv:2303.06296](https://ar5iv.labs.arxiv.org/html/2303.06296) [verified]
- Their Theorem 3.1 gives a tight lower bound on attention entropy that "decreases exponentially fast with the spectral norm of the attention logits" (in terms of ‖W_K W_Qᵀ‖ and the input norm). — [same](https://ar5iv.labs.arxiv.org/html/2303.06296) [verified]
- σReparam reparameterises "all linear layers with spectral normalization and an additional learned scalar", `Ŵ = γ/σ(W) · W`, with γ initialised to 1. — [same](https://ar5iv.labs.arxiv.org/html/2303.06296) [verified]
- On gradients, the same paper says entropy collapse "tends to introduce high gradient norms rather than vanishing gradients". It attributes vanishing gradients to rank collapse, a separate failure mode. — [same](https://ar5iv.labs.arxiv.org/html/2303.06296) [verified]
- ALiBi chose fixed slopes in (0, 1) after a manual search. The best slope sets had "density increasing toward 0", meaning most heads are weakly biased. — [ALiBi, Press et al. 2022, arXiv:2108.12409](https://ar5iv.labs.arxiv.org/html/2108.12409) [verified]
- In KERPLE, the power-kernel variant (and ALiBi) "reach very negative values quickly", while the learned log-kernel produced several **flat** kernels, that is, heads that learned to *weaken* the locality prior to use distant context. — [KERPLE, Chi et al. 2022, arXiv:2205.09921](https://ar5iv.labs.arxiv.org/html/2205.09921) [verified]

### Inferences
- **[derivation] Softmax gradient.** Let `u_j = v_jᵀ (∂L/∂h)` and `ū = Σ_k p_k u_k`. Then `∂L/∂z_j = p_j (u_j − ū)`.
  - For a key with p_j ≈ 0.001, the content-logit gradient (and so its contribution to ∂L/∂W_q, ∂L/∂W_k) is scaled by about 1e-3.
  - For the dominant key with p ≈ 0.8+, `u_1 − ū = (1 − p_1)(u_1 − ū_rest)` is also shrunk, by (1 − p_1) ≤ 0.2.
  - So at 80% concentration, content gradients are at most about 0.2 of their unsaturated scale, and they → 0 as p → one-hot.
- **[derivation] Bias-scalar gradient.** Since `∂z_j/∂λ_h = −dist_j`:
  - `∂L/∂λ_h = −Σ_j p_j dist_j (u_j − ū) = −Cov_p(dist, u)`.
  - Likewise `∂L/∂μ_h = −Cov_p(1[other frame], u)`.
  - When the bias concentrates mass, Var_p(dist) → 0 and so does the gradient: λ cannot easily learn to *decrease* itself once it has collapsed attention onto the nearest key.
  - For μ: with μ = 4 each other-frame key is down-weighted by e⁻⁴ ≈ 0.018. If the two frames have equal key counts, other-frame mass is about 1.8%, and the gradient to μ is scaled by roughly p_other(1 − p_other) ≈ 0.018.
  - Adam normalises magnitude, so this mostly matters for gradient **signal-to-noise** and the ε floor (Q1), not step size.
- **Strength of the init.** What "λ = 4" means depends entirely on the units of dist. If dist is in metres (or normalised scene units) and typical neighbour spacing is around 0.1, then λ·Δdist ≈ 0.4 and attention stays broad. If dist is in patch or grid units, a 1-unit offset costs e⁻⁴ ≈ 55× and saturates almost immediately. **Recommendation: before training, compute the attention entropy (or the top-1 mass) produced by the bias alone (q·k = 0) on real batches at N = 2,400 and N = 9,600, and choose λ so that typical heads keep top-1 mass well below 0.8.**
- **Train/eval length shift.** Going from 2,400 to 9,600 keys at evaluation adds about 4× as many competing keys, but far keys get exponentially small weight under the bias. Saturated heads are length-robust; diffuse heads are not. This is the ALiBi extrapolation argument (ALiBi shows bias-dominated heads extrapolate). It is also a reason to keep at least some heads strongly biased.
- Zhai et al.'s mechanism (logit-norm growth in W_q/W_k) does not apply directly to a *fixed-magnitude* additive bias. It could matter if W_q/W_k grow to fight the bias. σReparam-style control is probably unnecessary at this scale (3 layers, 5k steps).

### Gaps
- I found no paper that explicitly analyses gradient flow to *learned additive bias slopes* under saturation. The covariance result above is my derivation; check it numerically with `torch.autograd.gradcheck` on a toy example.

---

## Q3. Parameterising learned slopes (raw vs softplus/exp), effective learning rate, and weight decay

### Takeaway
With Adam at lr 1e-5, a raw scalar moves at most about 0.05 in 5,000 steps (4 → 3.95 to 4.05). In practice it is **frozen at its init**. This is a real pitfall. A log-space parameterisation (`λ = exp(θ)`) makes steps relative, but at the same lr θ also moves ≤ 0.05, so λ can change by at most about ×e^{±0.05}, which is ±5%. Bias scalars need their own param group with a much higher lr (about 1e-3 to 1e-2) and **no weight decay**. Published learned-bias models simply train these scalars at the base lr of a from-scratch run (around 1e-4 to 1e-3), not at 1e-5.

### Cited Findings
- ALiBi: the authors "initially experimented with making the slopes trainable, but this did not yield strong extrapolation results", and "trainable slopes also slowed down the training speed by 3%". The fixed slopes form a geometric sequence starting at 2^(−8/n) with the same ratio (1/2, …, 1/2⁸ for 8 heads). For 16 heads they interpolate the 8-head slopes geometrically. — [ALiBi, arXiv:2108.12409](https://ar5iv.labs.arxiv.org/html/2108.12409) [verified]
- KERPLE: the power kernel is `q_mᵀk_n + c − r1|m−n|^{r2}` with r1 > 0 and 0 < r2 ≤ 2; the log kernel is `… − r1·log(1 + r2|m−n|)` with r1, r2 > 0. The paper **does not say how positivity is enforced** and gives no separate lr. Training is plain Adam at lr 6e-4 (162M) and 2e-4 (1.3B), with 2 params per head. — [KERPLE, arXiv:2205.09921](https://ar5iv.labs.arxiv.org/html/2205.09921) [verified]. That the official code enforces positivity by clamping in forward is [recalled, not re-verified]; I could not locate the file in the repo.
- ConViT: the gating parameter λ_h enters as σ(λ_h) and is initialised to 1 for all heads. This is deliberate: a large positive init would make σ(λ_h) ≈ 1 and "trap" the model in positional attention. The locality strength α^h > 0 acts like a softmax temperature and is learned as part of v_pos. The paper mentions no special lr or weight decay; training follows the DeiT hyperparameters. During training σ(λ_h) falls, reaching 0 in later layers, and heads "escape" locality. — [ConViT, d'Ascoli et al. 2021, arXiv:2103.10697](https://ar5iv.labs.arxiv.org/html/2103.10697) [verified]
- Mamba stores the decay as `A_log` and uses `A = −exp(A_log)` (log-space, sign-constrained). This is visible in HF transformers 5.20.1, `models/mamba/modeling_mamba.py` lines 326 and 373 (local install) [verified]. The reference implementation marks `A_log._no_weight_decay = True` — [state-spaces/mamba](https://github.com/state-spaces/mamba) [recalled, not re-verified]
- RetNet uses **fixed** (not learned) per-head decays `γ = 1 − 2^{−5−arange(0,h)}`, a geometric spread across heads. — [RetNet, Sun et al. 2023, arXiv:2307.08621](https://arxiv.org/abs/2307.08621) [recalled, not re-verified]
- Swin keeps a learned `relative_position_bias_table` (HF `modeling_swin.py` line 343, [verified] local). The official Swin repo excludes it from weight decay via `no_weight_decay_keywords` — [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer) [recalled, not re-verified]
- Adam's per-step update magnitude is ≲ lr regardless of gradient scale. — [Kingma & Ba, arXiv:1412.6980](https://arxiv.org/abs/1412.6980) [recalled]

### Inferences
- **[derivation] Movement budget.** The maximum total displacement is about lr × steps, and only if the gradient sign is consistent.
  - lr 1e-5 → 0.05.
  - lr 1e-4 → 0.5.
  - lr 1e-3 → 5.
  - lr 1e-2 → 50.
  - Getting λ from 4 to 0 (switch locality off) needs at least about 8e-4. Getting it to 0.4 or 40 in log-space (Δθ = ±2.3) needs at least about 5e-4. Allowing for noise (sign inconsistency), **lr ≈ 1e-3 to 1e-2 for the 96 scalars** is the reasonable range. Use warmup for this group so early noisy gradients (Q1) don't push the values around.
- **Parameterisation choice.**
  - `λ = exp(θ)`: steps are relative, positivity is guaranteed, and it matches the geometric spacing used by ALiBi and RetNet, so a uniform lr suits heads that differ by ~100× in scale. Downside: λ can never reach exactly 0, though exp(−5) ≈ 0.007 is effectively off.
  - `softplus(θ)` behaves linearly for large values and exponentially near 0. It is a reasonable compromise if you want "off" reachable.
  - Raw scalar with clamp: allowed, but clamping zeroes the gradient at the boundary.
  - With log-space λ, the gradient wrt θ is λ·∂L/∂λ, which compounds the small-covariance problem from Q2 for heads with small λ. That is harmless under Adam.
- **[derivation] Weight decay.** AdamW decoupled decay shrinks a parameter by a factor of (1 − lr·wd) per step.
  - At lr 1e-5 and wd 0.01 over 5,000 steps: (1 − 1e-7)^5000 ≈ 0.9995, which is negligible.
  - At lr 1e-2 and wd 0.1: (1 − 1e-3)^5000 ≈ e⁻⁵ ≈ 0.007. That drives raw λ toward 0 (locality off), or exp-param θ toward 0 (λ → 1), independent of the data.
  - So **set weight_decay = 0 for the λ/μ group**, matching common practice for biases, norm gains, `A_log` and Swin's bias table.
- The ALiBi "trainable slopes didn't help" result is for *extrapolation in from-scratch LM training*. It does not argue against learning here. It does suggest that a **good fixed per-head spread** may get most of the benefit. A fixed-slope arm is a sensible control.

### Gaps
- I could not verify how KERPLE's code enforces r1, r2 > 0.
- I found no paper that reports the lr used *specifically* for learned bias slopes when fine-tuning at a small base lr.

---

## Q4. PyTorch implementation: SDPA with a grad-requiring float mask, FlexAttention with learnable captured tensors, xFormers

### Takeaway
SDPA's **Flash backend rejects any non-null attn_mask**. A float mask that requires grad works on the **memory-efficient** backend (which computes `attn_bias.grad`), on **cuDNN** (which supports dBias, with restrictions below cuDNN 9.5), and on **math**. It costs a fully materialised B×H×Nq×Nk bias plus an equally large gradient. **FlexAttention** supports gradients to tensors captured by `score_mod` (but not `mask_mod`). Captured grads are accumulated with atomic adds, it needs a GPU for backward, and its flash/CuTe backend refuses grad-requiring buffers, so it falls back to the Triton kernel. For a squared-Euclidean distance there is an exact trick that folds the bias into augmented Q/K, which lets Flash run with gradients to λ, μ.

### Cited Findings
- SDPA docs: a float `attn_mask` is "added to the attention score". The mask must be broadcastable to (N, …, Hq, L, S). The listed implementations are FlashAttention-2, Memory-Efficient Attention, and a C++ math implementation. "Each of the fused kernels has specific input limitations", but these are not enumerated. The docs say nothing about gradients wrt attn_mask. — [PyTorch 2.14 SDPA docs](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html) [verified]
- Installed `libtorch_cuda.so` (torch 2.14.1+cu130) contains these error strings [verified via `strings`]:
  - Flash: "Flash Attention does not support non-null attn_mask."
  - Memory-efficient: "attn_bias.grad is not correctly aligned (strideB/strideH/strideM)". This implies a bias-gradient output that requires aligned strides.
  - cuDNN: "For cuDNN version below 9.5.0, dBias with variable sequence lengths is not supported", "For cuDNN version below 9.5.0, dBias not support s_q/s_kv which aren't multiple of 64", "bias_requires_grad is true but no bias was provided", "cuDNN SDPA got attn_bias of unsupported dtype", and "TORCH_CUDNN_SDPA_AVOID_RECOMPILE=1 only works without bias".
- Hugging Face Transformers forces the mask to be contiguous before the memory-efficient path, citing [pytorch/pytorch#127523](https://github.com/pytorch/pytorch/issues/127523). Fully masked rows give NaNs on the memory-efficient backend ([pytorch/pytorch#110213](https://github.com/pytorch/pytorch/issues/110213)). — via search results summarising the HF code [partially verified: issue numbers from secondary source]
- FlexAttention blog:
  - `score_mod(score, b, h, q_idx, kv_idx)` modifies scores pre-softmax.
  - It can read captured tensors, e.g. `alibi_bias[h] * (kv_idx - q_idx)`, without materialising an S×S bias.
  - Changing captured tensor values does not recompile.
  - "We also automatically generate the backwards pass".
  - Performance is about 90% of FA2 forward and 85% of FA2 backward on A100.
  - At the time of the blog it was "currently available in PyTorch nightly releases, we plan to release it as a prototype feature in 2.5.0", and it required sequence lengths to be multiples of 128 ("will be addressed soon").
  — [FlexAttention blog](https://pytorch.org/blog/flexattention/) [verified]
- Installed torch 2.14.1 source [verified]:
  - `torch/_higher_order_ops/flex_attention.py` raises "Captured buffers from mask mod that require grad are not supported." The comment at lines 874–875 reads "mask_mod_other_buffers do not require grad, but score_mod_other_buffers can require grad". The backward returns `*grad_score_mod_captured`.
  - Zero-dim (scalar) grad-requiring buffers are specially handled (lines 733–737).
  - `torch/_inductor/kernel/flex/common.py:157` uses `scatter_mode="atomic_add"` to accumulate captured-buffer gradients. `flex_attention.py:628` notes that this "will either require atomic updates to some grad values or to have a two pass kernel design".
  - `torch/_inductor/kernel/flex/flex_flash_attention.py:357–360, 603–608` makes the flash (CuTe/FA-based) flex backend reject such graphs ("Input buffers require gradients (not supported by flash attention)"), so it falls back to the Triton template.
  - `torch/nn/attention/flex_attention.py:2212, 2218`: "FlexAttention does not support backward on CPU" and "… on MPS".
- The PyTorch release in which grad-requiring captured buffers ("learnable biases") were first supported by FlexAttention: I believe 2.6 or 2.7. — [recalled, not re-verified]. In 2.14.1 the feature is present [verified].
- xFormers: an issue says PyTorch's mask support was added in [pytorch/pytorch#104042](https://github.com/pytorch/pytorch/pull/104042) but "does not work correctly" per comments — [xformers#893](https://github.com/facebookresearch/xformers/issues/893) [secondary, not read in full]. That xFormers `memory_efficient_attention` supports gradients for a dense tensor `attn_bias` is [recalled, not re-verified]. xFormers is not installed in the project venv.

### Inferences
- **[derivation] Memory cost of a dense bias (bf16).**
  - The bias is B × 16 heads × Nq × Nk × 2 bytes, and the backward needs a gradient of the same shape.
  - With Nq = 1,000 query tokens and Nk = 2,400 keys: about 77 MB per sample per layer for the bias, plus 77 MB for its gradient, × 3 layers.
  - At Nk = 9,600 (evaluation, no grad) it is 4× larger.
  - Broadcasting λ_h·D over heads still materialises the per-head tensor, because `λ[:,None,None] * D` creates a fresh tensor and mem-efficient wants aligned, contiguous strides.
  - This is acceptable on B200/B300 but not free. Masks whose key length is not a multiple of 8/16 may also need padding for alignment.
- **Recommended route (in order).**
  1. **Q/K augmentation with Flash/cuDNN and no mask** [derivation; verify numerically].
     - If `dist` is **squared** Euclidean distance between 3-D points p_q and p_k: `−λ‖p_q − p_k‖² = −λ‖p_q‖² + 2λ p_q·p_k − λ‖p_k‖²`. The first term is constant per query, so it cancels in softmax.
     - The frame term: `−μ·[f_q ≠ f_k] = −μ + μ·onehot(f_q)·onehot(f_k)`. The constant cancels.
     - So append channels: `q' = [q/√d, 2λ_h p_q, −λ_h, μ_h e_{f_q}]` and `k' = [k, p_k, ‖p_k‖², e_{f_k}]`. Call SDPA with `scale=1.0` and pad the head dim to a multiple of 8.
     - λ_h and μ_h enter Q, so autograd gives exact gradients through the fastest kernels. Memory is O(N), and the 9,600-key evaluation is cheap.
     - Caveats: this is exact only for *squared* distance (or any kernel that factorises as an inner product), not for plain |·| distance. In bf16, large ‖p‖² values cause cancellation error, so centre and scale the coordinates per scene and consider fp32 for the logits.
  2. **FlexAttention `score_mod`** capturing `lam` (shape [H]) and per-token positions and frame ids, for example `score - lam[h]*dist(pos_q[b,q_idx], pos_k[b,kv_idx]) - mu[h]*(fq[b,q_idx]!=fk[b,kv_idx])`.
     - Supports any distance function with no S×S materialisation.
     - Needs `torch.compile` and runs on the Triton backend because buffers require grad.
     - The 16-element λ gradient is accumulated with atomic adds from every score element, which is expected to be slow (heavy contention) and non-deterministic. Benchmark it. If it is slow, capture λ as a [B, H, …]-shaped or per-block buffer and reduce afterwards [unverified idea].
  3. **SDPA with a dense float mask** on the memory-efficient or cuDNN backends. This is the simplest and most robust route. Pin the backend with `torch.nn.attention.sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])` so it does not silently fall back to math, which is O(N²) in memory *and* slow.
  4. Math backend only as a correctness reference.
- Check correctness by comparing gradients for λ, μ between the chosen route and the math backend (fp32) on a small batch with `gradcheck` or `allclose`.

### Gaps
- The exact PyTorch version that introduced FlexAttention grad-requiring captured buffers is unconfirmed.
- I found no benchmark of atomic-add contention when gradients collapse to per-head scalars.
- xFormers attn_bias gradient support is not verified from docs.
- The SDPA docs do not document per-backend mask/grad support; the evidence above comes from error strings in the binary, not prose docs.

---

## Q5. Per-head initialisation schemes (geometric spread)

### Takeaway
Every established locality/decay scheme initialises (or fixes) heads on a **geometric spread**, not at a common value: ALiBi uses 2^(−8i/n), RetNet uses 1 − 2^(−5−i), and ConViT gives each head a different attention centre. A common init of 4 for all 16 heads throws away this multi-scale prior, and given the small movement budget (Q3) the heads will stay nearly identical in bias strength.

### Cited Findings
- ALiBi: slopes for n heads form the geometric sequence starting at 2^(−8/n) with ratio 2^(−8/n). The best slope sets lie in (0, 1) with density increasing toward 0. — [ALiBi, arXiv:2108.12409](https://ar5iv.labs.arxiv.org/html/2108.12409) [verified]
- ConViT: each head's attention centre Δ^h is set to a different offset (heads together mimic a k×k convolution, e.g. 9 heads → 3×3). The locality strength α^h > 0 controls focus, and higher locality strength in the ablation improved accuracy. — [ConViT, arXiv:2103.10697](https://ar5iv.labs.arxiv.org/html/2103.10697) [verified]
- RetNet: per-head decay `γ = 1 − 2^{−5−arange(0,h)}`, a geometric spread of effective windows (32 to about 32·2^{h−1} tokens). — [RetNet, arXiv:2307.08621](https://arxiv.org/abs/2307.08621) [recalled, not re-verified]
- LLaMA-Adapter learns per-head gates for "learning diversity of multi-head mechanisms". — [LLaMA-Adapter, arXiv:2303.16199](https://ar5iv.labs.arxiv.org/html/2303.16199) [verified]
- KERPLE's learned kernels diverge across heads and datasets: some become flat (global), others local. — [KERPLE, arXiv:2205.09921](https://ar5iv.labs.arxiv.org/html/2205.09921) [verified]

### Inferences
- Initialise λ_h log-uniformly across the 16 heads so that the bias-only top-1 mass ranges from diffuse to near-local. For example `λ_h = λ_max · r^h` with λ_max chosen from the dist-unit calibration in Q2 and r ≈ 2^(−0.5) (a 2^8 range over 16 heads, as in ALiBi). Include 1 to 2 heads with λ ≈ 0 so global content attention is available from the start.
- Spread μ_h similarly, for example 0 to 4. Heads with μ ≈ 0 can attend across frames immediately; heads with large μ stay within-frame.
- Use the same spread in every layer, or shift it per layer (ConViT shows later layers want less locality).
- Combined with the exp parameterisation (Q3), a geometric init means a single lr gives every head the same *relative* mobility.

### Gaps
- I found no paper that compares a uniform vs geometric init for *learned* (as opposed to fixed) bias slopes in a fine-tuning setting.
