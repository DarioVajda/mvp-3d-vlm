# Locality priors in attention: how strength/width is initialised and tuned

Context for the reader: our design adds a logit bias b = -λ·(|Δr|+|Δc|) - μ·[Δt≠0] (per head, learned) to cross-attention from visual tokens to geometry tokens on the same (t,r,c) grid. λ is set so ≈82% of mass starts on the query's own cell. Own computation (plain arithmetic, no source needed): on an unbounded 2D grid with an L1 bias of slope λ, own-cell mass within one frame is tanh(λ/2)^2. That gives λ=1→0.21, λ=2→0.58, λ=2.5→0.72, **λ=3→0.82**, λ=3.5→0.89, λ=4→0.93. With T frames and a time penalty μ, divide by (1+(T−1)e^{−μ}). This ignores content logits, which are ≈0 at init only if q·k/√d is small. All numbers below were extracted from the primary PDFs or repos unless marked UNVERIFIED.

## ConViT (d'Ascoli et al., arXiv 2103.10697): GPSA, locality strength α, gating λ

### Takeaway
ConViT is the closest analogue, with one important difference. Its locality prior is a **separate positional softmax mixed after the softmax** through a learned per-head sigmoid gate (init λ=1, so σ=0.73 positional). It is **not** an additive logit bias. The authors show that an additive positional bias with convolutional init (vanilla PSA) stays stuck local and ignores content, which hurts late-training accuracy. The default locality strength is α=1. Accuracy rose monotonically with α over the range tried (≈0.01–1). After training, positional reliance decays to ≈0 in deep layers and survives only in early layers.

### Cited Findings
- **Positional attention parameterisation (Eq. 5):** v_pos^h = −α_h(1, −2Δ^h_1, −2Δ^h_2), r_δ = (‖δ‖², δ1, δ2), so the positional logit is −α_h‖δ−Δ^h‖² + const, a Gaussian centred at offset Δ^h. At conv init W_qry = W_key = 0 and W_val = I. D_pos = 3. Only v_pos is trained and r_δ is fixed. The paper says this "relates to the adaptive attention span introduced in Sukhbaatar et al. (2019)". — [ConViT §2–3](https://arxiv.org/abs/2103.10697)
- **Meaning of α:** "The locality strength α_h > 0 determines how focused the attention is around its center Δ_h (it can also by understood as the 'temperature' of the softmax)". When α is large, attention goes only to the patch at Δ_h; when α is small, it spreads. Fig. 3 illustrates α=0.5 and α=2. — [ConViT §2, Fig. 3](https://arxiv.org/abs/2103.10697)
- **Centres:** with N_h heads, the centres tile a √N_h×√N_h kernel. Example for 4 heads: Δ = (−1,1), (−1,−1), (1,1), (1,−1). — [ConViT §2](https://arxiv.org/abs/2103.10697)
- **Code defaults:** `locality_strength=1.`, `use_local_init=True`, `local_up_to_layer=10` (GPSA in blocks 0–9 of 12, vanilla SA in the last 2). `gating_param = nn.Parameter(torch.ones(num_heads))`, i.e. λ_h=1. `local_init` sets `v.weight = I`, sets `pos_proj.weight[position,2] = -1` (the ‖δ‖² column), puts 2×offset in the other columns, then multiplies by `locality_strength`. `locality_distance` is hard-coded to 1. kernel_size = int(num_heads**0.5), so with 9 heads (ConViT-S) all 9 heads get a 3×3 offset. — [facebookresearch/convit convit.py](https://github.com/facebookresearch/convit/blob/main/convit.py)
- **Gated mixing (Eq. 7):** A^h = (1−σ(λ_h))·softmax(QKᵀ) + σ(λ_h)·softmax(v_posᵀ r_ij), followed by row renormalisation. Rationale: "the content and positional terms … are potentially of different magnitudes, in which case the softmax will ignore the smallest of the two. In particular, the convolutional initialization scheme … involves highly concentrated attention scores, i.e. high-magnitude values in the softmax." — [ConViT §3 "Positional gating"](https://arxiv.org/abs/2103.10697)
- **Gate init rationale:** "By setting the gating parameter λ_h to a large positive value at initialization, one has σ(λ_h) ≃ 1 … However, to avoid the ConViT staying stuck at λ_h ≫ 1, we initialize λ_h = 1 for all layers and all heads." σ(1) ≈ 0.73. — [ConViT §3](https://arxiv.org/abs/2103.10697)
- **Failure mode of an additive (pre-softmax) conv-init prior (App. A, Fig. 9):** with vanilla PSA, where content and position are summed *before* the softmax, "The convolutional initialization of the PSA still gives the ConViT a large advantage over the DeiT baseline early in training. However, the ConViT stays in the convolutional configuration and ignores the content information … Later in training, the DeiT catches up and surpasses the performance of the ConViT". The main text says it "degrades late-time performance as the attention mechanism lazily ignores the content information". The experiment used ConViT-B, 300 epochs, ImageNet. — [ConViT §3 and App. A](https://arxiv.org/abs/2103.10697)
- **α ablation (Fig. 8 left):** a heatmap of locality strength (log axis, ticks 0.01, 0.1, 1) against number of GPSA layers (0, 3, 6, 9), on ConViT-S+ (per the caption) trained on the first 100 ImageNet classes, with colour range ≈77–81 top-1. "Final test accuracy increases both with the locality strength and with the number of GPSA layers; in other words, the more convolutional, the better." Values of α above 1 were **not** shown. The exact grid values are read off the axis ticks (partially UNVERIFIED, since they are not tabulated). — [ConViT §4 "Strong locality is desirable", Fig. 8](https://arxiv.org/abs/2103.10697)
- **Head start:** "after 20 epochs, using 9 GPSA layers leads the test-accuracy to almost double". — [ConViT §4, Fig. 8 right](https://arxiv.org/abs/2103.10697)
- **Ablation, Table 3** (ConViT-S+, 300 epochs; full ImageNet / 10% data). Row values:

  | Row | Configuration | Full ImageNet | 10% data |
  |---|---|---|---|
  | (a) | full ConViT | 82.2 | 59.7 |
  | (b) | gates frozen at λ=0 (50/50 mix) | 82.0 | 57.4 |
  | (c) | no conv init | 81.4 | 56.9 |
  | (d) | no gating and no conv init | 81.6 | 54.6 |
  | (e) | DeiT | 79.1 | 47.8 |
  | (f) | GPSA frozen | 78.6 | 54.3 |
  | (g) | GPSA frozen | 73.7 | 44.8 |

  Rows f and g are both "GPSA frozen", one with random init and one with conv init. Which is which is ambiguous in the extracted table. The text says frozen random-init GPSA is −5.4 vs DeiT and frozen conv-init is −0.5, which suggests 78.6 = conv-init frozen and 73.7 = random-init frozen. In low data, "GPSA alone brings +6.8, with an extra +2.3 coming from gating, +2.8 from convolution initialization and +5.1 with the two together". — [ConViT Table 3, §4](https://arxiv.org/abs/2103.10697)
- **Learned locality after training (Fig. 5).** The nonlocality metric D_loc is the attention-weighted mean patch distance. DeiT layers first become *more* local in early epochs, then upper layers grow non-local. In ConViT, "this local configuration is escaped throughout training, as the nonlocality metric grows in all the GPSA layers", but it ends lower than DeiT. "The first layer and the final layers strongly escape locality, whereas the intermediate layers (particularly the second layer) stay more local." The y-axis range is ≈3–7 patches. — [ConViT §4, Fig. 5](https://arxiv.org/abs/2103.10697)
- **Gate trajectories (Fig. 6):** the mean E_h σ(λ_h) "decreases throughout training" in all layers. It "reaches 0 in layers 6-10, meaning that positional information is practically ignored. However, in layers 1-5, some of the attention heads keep a high value of σ(λ_h)". ConViT-Ti uses position up to layer 4, ConViT-S up to layer 5 and ConViT-B up to layer 6 ("the larger … the more layers make use of the convolutional prior"). The example layer-2 heads of ConViT-Ti have σ = 0.00, 0.09, 0.51, 0.73. "Some stay highly localized (head 4) whereas others broaden their attention span (head 3)." — [ConViT §4, Figs 6–7, App. D](https://arxiv.org/abs/2103.10697)
- **Test-time masking (Table 6, ConViT-Ti; no mask = 73.1):**

  | GPSA layers masked | Mask content | Mask position |
  |---|---|---|
  | 3 | 62.3 | 63.5 |
  | 5 | 35.0 | 53.1 |
  | 10 | 1.3 | 46.8 |

  Content becomes crucial in the deeper GPSA layers. — [ConViT App. F](https://arxiv.org/abs/2103.10697)

### Inferences
- Own arithmetic: ConViT's default α=1 with an L2² penalty puts only ≈32% of the *positional* softmax on the centre cell (head with Δ=0, 14×14 grid). After the 0.73 gate and a near-uniform content softmax, that is ≈23% of total attention. α=0.5 gives 16% and α=2 gives 62%. Our 82% own-cell target (λ≈3 with L1) is much more concentrated than ConViT's default, above even α=2. ConViT's ablation explored only α ≤ 1 and found "stronger is better" within that range. It therefore gives no evidence either way for priors as strong as ours.
- The ConViT App. A result is the most direct warning for our design. An additive pre-softmax locality bias with a strong init can trap attention in the local configuration because content logits cannot compete. ConViT's fix was to move the prior out of the softmax and gate it.
- In vanilla PSA, α (inside v_pos) was trainable, yet the model still stayed local. So "λ is learnable" is not by itself sufficient protection against getting stuck.

### Gaps
- The exact α grid values and per-cell accuracies of Fig. 8 are not tabulated. The final learned α values per layer and head are not reported: the paper reports only the nonlocality metric and the gates.
- No ConViT experiment uses α > 1, apart from the illustration of α=2 in Fig. 3.

## Localness modelling in NMT: Yang et al. 2018 (arXiv 1810.10182), CSAN (Yang et al. 2019, arXiv 1904.03107), Gaussian Transformer

### Takeaway
NLP localness priors either set the Gaussian width from Luong et al.'s window D=10 (σ=D/2), or predict D through a sigmoid scaled by sequence length. The prior helps most when applied to lower layers only. Learned windows grow with depth, except in layer 1. Gains are small (≈+0.5 BLEU), and fixed and learned windows differ by only 0.07–0.23 BLEU.

### Cited Findings
- **Yang et al. 2018:** additive bias ATT = softmax(energy + G), with G_ij = −(j−P_i)²/(2σ_i²) and **σ_i = D_i/2** ("empirically set"). Centre and window are predicted as [P_i; D_i] = I·sigmoid([p_i; z_i]), where I is the sequence length. — [Yang 2018 §3.1](https://arxiv.org/abs/1810.10182)
- **Fixed window:** "following the common practice (Luong et al., 2015), D is set to 10." The variants are layer-specific (z from the mean of the keys) and query-specific (z_i = U_dᵀ tanh(W_p Q_i)). — [Yang 2018 §3.3](https://arxiv.org/abs/1810.10182)
- **Table 1 (Zh→En dev BLEU, encoder self-attention only):** baseline 22.59, Fixed 23.07 (+0.48), Layer-Spec 23.13, Query-Spec 23.13. The query-specific window beats fixed by +0.07 (Zh-En) and +0.23 (En-De). — [Yang 2018 Table 1](https://arxiv.org/abs/1810.10182)
- **Which layers (Table 2):** all 6 layers 23.13, layer 1 only 23.20, layers 1–3 **23.29** (best), layers 4–6 23.22. "Lower-level layers benefit more from the local context." Table 3 compares attention types: Enc + Enc-Dec is best at 23.33, and Enc + Dec + Enc-Dec gives 23.19. — [Yang 2018 Tables 2–3](https://arxiv.org/abs/1810.10182)
- **Head-specific learned window:** D_m = N·sigmoid(z_m), with N=50 and z_m a free scalar per head. Learned windows differ across heads (mixes of local and global in the middle layers). "Except the first layer, the higher layers are more likely to pay attention to larger scopes". The first layer is assigned large scopes. — [Yang 2018 §5.1–5.2, Figs 2–3](https://arxiv.org/abs/1810.10182)
- **CSAN (hard local window):** after sweeping the window size, "the local size with 11 is superior to other settings. This is consistent with Luong et al. (2015) who found that 10 is the best window size". 2D-CSAN (window across neighbouring heads) improves with a few heads but degrades as the number of heads grows. — [Yang 2019 §5.1, Fig. 2](https://arxiv.org/abs/1904.03107)

### Inferences
- The NLP precedent is to start from a moderate window chosen by a dev sweep (D≈10–11 tokens) and then let it be learned. Gains were small and fixed ≈ learned, which suggests final quality depends only weakly on the exact width when the prior is mild.

### Gaps
- Gaussian Transformer (Guo, Zhang, Liu, AAAI 2019) was not retrieved: it is not on arXiv and I did not fetch the AAAI PDF. I recall that it uses a learnable Gaussian distance penalty of the form −|w·d²+b| with learned w and b, but this is UNVERIFIED.

## SMCA (Gao et al., arXiv 2101.07448): Gaussian spatial prior on DETR cross-attention

### Takeaway
SMCA adds log G to the cross-attention logits, where G = exp(−(i−c_w)²/(β s_w²) − (j−c_h)²/(β s_h²)). The centre and per-head scales are predicted from the query. β is a hand-set "bandwidth" chosen so that G covers a large area *at the start* of training for informative gradients. In the official code it appears as `smooth=8`. A fixed scale (=1) is clearly worse than predicted scales.

### Cited Findings
- Eq. 4 defines G as above. "β is a hyper-parameter to modulate the bandwidth of the Gaussian-like distribution … β can be manually tuned with a handcrafted scheme to ensure G covering a large spatial range at the beginning of training so that the network can receive more informative gradients." Eq. 5 is C_i = softmax(Kᵀ Q/√d + log G) V. — [SMCA §3.2](https://arxiv.org/abs/2101.07448)
- **Table 2 (COCO val, R50, 50 epochs):**

  | Variant | mAP |
  |---|---|
  | baseline DETR | 34.8 |
  | head-shared, independent h/w scales | 40.2 |
  | multi-head, **Fixed** scale (constant 1, only the centre predicted) | 38.5 |
  | multi-head, **Single** predicted scale | 40.4 |
  | multi-head, **Indep.** (separate h/w scales) | **41.0** |

  The text describes Fixed as "+3.7 gain over the baseline DETR-R50" and Single as "+1.9 compared with the above fixed-scale modulation". — [SMCA Table 2, §4](https://arxiv.org/abs/2101.07448)
- **Code:**
  - The bias is `gaussian = -(distance - 0).abs() / self.smooth`, with default `--smooth 8`.
  - distance = Σ(point − grid)² × scale, where scale = (linear output)² for type2/3/4, or 1 for type1.
  - Coordinates are in feature-map cells: the reference point is sigmoid × h_w/32.
  - The README training command uses `--dynamic_scale type3` (independent h/w scales).

  So the effective logit bias is −d²·s/8 in feature-cell units. With s=1 this puts ≈4% of mass on the centre cell (own arithmetic, unbounded grid), i.e. a wide prior. — [SMCA-DETR models/transformer.py](https://github.com/gaopengcuhk/SMCA-DETR/blob/main/models/transformer.py), [SMCA-DETR repo (main.py, README)](https://github.com/gaopengcuhk/SMCA-DETR)

### Inferences
- SMCA deliberately starts **broad**, with the stated reason of informative early gradients. That is the opposite philosophy to our "start sharp at identity". Starting broad suits SMCA, where the centre itself must be learned. In our case the centre is known exactly (the aligned cell), which argues for a sharper start. The gradient argument is still the same concern that drives the saturation risk below.

### Gaps
- The paper reports no sweep over β (`smooth`). The value 8 comes only from the code default.

## Conditional DETR, DAB-DETR, Anchor DETR: positional priors via sinusoidal-PE dot products

### Takeaway
These methods have no explicit distance penalty. Their spatial prior comes from dot products of sinusoidal embeddings of the reference point and the key positions. Its width is controlled by the PE temperature (DAB-DETR uses T=20 instead of 10000) and by learned per-query modulation (λ_q in Conditional DETR, w/h ratios in DAB-DETR).

### Cited Findings
- **Conditional DETR:** p_s = sinusoidal(sigmoid(s)) of the reference point, and the conditional spatial query is p_q = λ_q ⊙ p_s. λ_q is a transformation predicted from the decoder embedding; it scales the spatial query and so shapes the positional attention. — [Conditional DETR §3, Eqs 5–6](https://arxiv.org/abs/2108.06152)
- **DAB-DETR:** ModulateAttn = (PE(x)·PE(x_ref)·w_q,ref/w_q + PE(y)·PE(y_ref)·h_q,ref/h_q)/√D. "The temperature T in Eq. (8) influences the size of positional priors … A larger T results in a more flattened attention map." Fig. 7 visualises T = 1, 10 and 10000. "In this work, we empirically choose T = 20 in all our models." — [DAB-DETR §4, Eqs 6–8, Fig. 7](https://arxiv.org/abs/2201.12329)
- **DAB-DETR Table 3 (R50-DC5):** full model 45.7 AP; without wh-modulated attention 45.0; without temperature tuning 44.4. "Fine-grained tuning of positional attentions is of great importance". — [DAB-DETR Table 3](https://arxiv.org/abs/2201.12329)
- **Anchor DETR:** queries are encodings of anchor-point coordinates, with Row-Column Decoupled Attention. The PDF mentions no explicit width or temperature hyperparameter for the prior. — [Anchor DETR](https://arxiv.org/abs/2109.07107)

### Inferences
- The 1.3 AP effect from PE temperature alone shows that prior width matters measurably even when the prior is implicit.

### Gaps
- There is no sweep of T values beyond the visualisation, so sensitivity around T=20 is unknown.

## Guided attention (TTS) and diagonal/monotonic priors

### Takeaway
Tachibana et al. use a soft diagonal *loss*, not a logit bias: W_nt = 1 − exp(−(n/N − t/T)²/(2g²)) with g=0.2. The value of g is stated with no ablation.

### Cited Findings
- "Latt(A) = E_nt[A_nt W_nt], where W_nt = 1 − exp{−(n/N − t/T)²/2g²}. In this paper, we set g = 0.2." With this loss, the attention loss begins decreasing early and the attention quickly becomes diagonal. At synthesis they add a hard rule: if n_t − n_{t−1} ∉ [−1, 3], the attention is forced to n_{t−1}+1. — [Tachibana et al. 2017 §4](https://arxiv.org/abs/1710.08969)

### Inferences
- g=0.2 is in *normalised* coordinates, a band of about 20% of the sequence length. It is a wide tolerance, applied as a penalty rather than imposed on the logits, and an example of a prior chosen once without a sweep.

### Gaps
- I did not retrieve ASR monotonic or diagonal-prior papers (e.g. Gaussian-prior or windowed attention in ASR), so their width choices remain open.

## Learnable or fixed per-head decay: ALiBi, RetNet, Cordonnier et al., LSA

### Takeaway
Per-head linear (ALiBi) and exponential (RetNet) distance penalties use **fixed geometric multi-scale schedules across heads**, chosen by a small manual search and not learned. ALiBi tried learnable slopes and found them no better, as well as slower. Cordonnier et al. learn their width α and note that α≈46 already gives hard attention in float32.

### Cited Findings
- **ALiBi:** the bias is −m·|i−j|, where "m is a head-specific slope fixed before training". For 8 heads, m = 1/2^1 … 1/2^8. For n heads, the slopes form a geometric sequence starting at 2^(−8/n) with ratio 2^(−8/n). "We initially experimented with making the slopes trainable, but this did not yield strong extrapolation results" (trainable slopes were also 3% slower). "A brief manual exploration of around ten slope sets … the slope sets that work best are those with slopes in the (0, 1) range, with the slopes' density increasing as we get closer to 0. We also found our method to be robust to slope choice." — [ALiBi §3](https://arxiv.org/abs/2108.12409)
- **RetNet:** per-head decay γ = 1 − 2^(−5−arange(0,h)). "We set γ identical among different layers and keep them fixed". This is a multi-scale schedule, from γ = 1 − 1/32 (short range) to values very close to 1 (long range). — [RetNet §2.2](https://arxiv.org/abs/2307.08621)
- **Cordonnier, Loukas, Jaggi 2020 (quadratic encoding, the precursor of ConViT):** A = −α(‖δ−Δ‖² + c). "Since the smallest positive float32 scalar is approximately 10^−45, setting α = 46 would suffice to obtain hard attention". Centres are initialised as Δ^(h) ~ N(0, 2I) and learned; after training the heads form a grid around the query pixel. In the non-isotropic variant, Σ^(−1/2) = I + N(0, 0.01I) at init, which corresponds to an effective α ≈ 0.5 for a −½δᵀΣ⁻¹δ logit. Some heads degenerate into thin stripes or uniform attention. — [Cordonnier et al. 2020 §3–4 and appendix](https://arxiv.org/abs/1911.03584)
- **LSA (Lee et al. 2021, ViT for small datasets):** learnable softmax temperature plus diagonal masking. Table 1 with fixed temperatures, top-1 on CIFAR100:

  | Temperature | CIFAR100 top-1 |
  |---|---|
  | ¼√d_k | 73.70 |
  | ½√d_k | 74.54 |
  | √d_k | 73.81 |

  A sharper-than-default softmax helps on small data. — [Lee et al. 2021, Table 1](https://arxiv.org/abs/2112.13492)

### Inferences
- Precedent for our per-head λ: initialise the heads at *different* strengths with a geometric spread, as ALiBi and RetNet do, rather than all at the 82% value. Some heads would then start near identity and others wider, which hedges against all heads saturating.

### Gaps
- Mamba/S4D decay initialisation (A = −(1..N), Δ log-uniform in [0.001, 0.1]) is from memory and UNVERIFIED here. Transformer-XL has no decay parameter and was not reviewed.

## How much mass on the anchor at initialisation, and saturation risk

### Takeaway
No source gives a principled "right" fraction of anchor mass at init. The evidence is indirect. ConViT found stronger-is-better for α ≤ 1 (≈32% on the centre per positional head). ConViT also found that a strong *additive* positional init makes attention lazily ignore content (vanilla PSA), which motivated its gate. SMCA chose a broad init explicitly for informative early gradients.

### Cited Findings
- Strong additive prior: in vanilla PSA the model "stays in the convolutional configuration and ignores the content information", and DeiT catches up and surpasses it late in training. — [ConViT App. A, Fig. 9](https://arxiv.org/abs/2103.10697)
- SMCA chooses β "to ensure G covering a large spatial range at the beginning of training so that the network can receive more informative gradients". — [SMCA §3.2](https://arxiv.org/abs/2101.07448)
- ConViT initialises its gate at λ=1, not at a large value, explicitly "to avoid the ConViT staying stuck at λ_h ≫ 1". — [ConViT §3](https://arxiv.org/abs/2103.10697)

### Inferences
- **Own derivation (softmax Jacobian):** ∂p_j/∂s_k = p_j(δ_jk − p_k).
  - With p_anchor = 0.82, the gradient through the anchor's own logit scales as p(1−p) ≈ 0.15, against a maximum of 0.25 at p = 0.5.
  - For a direct neighbour with p ≈ 0.045, the factor is ≈ 0.043.
  - Content-projection gradients flow mainly through the 0.18 of mass that is off the anchor. True saturation would mean ≥ 0.99 on the anchor, i.e. λ ≳ 6 (tanh(λ/2)² ≥ 0.99).

  So 82% is moderate rather than saturated, but it is in the regime that ConViT's vanilla PSA showed can be "sticky".
- A further subtlety: if the geometry token at the query's own cell already carries what the model needs, the loss gives little incentive to widen, and the prior becomes self-confirming (the ConViT PSA failure mode). Mitigations seen in prior work:
  - gate the prior outside the softmax (ConViT);
  - spread the per-head init over a geometric range (ALiBi, RetNet);
  - start broader (SMCA).

  A cheap diagnostic, following ConViT Figs 5–6, is to log the attention-weighted mean |Δ| (the nonlocality metric) and the learned λ per head and layer during training.
- Both ConViT and Yang 2018 find that early layers keep or benefit from locality while later layers abandon it or prefer wider scopes. If our cross-attention is injected deep in the LLM, λ may shrink substantially during training.

### Gaps
- I found no paper that directly sweeps "initial anchor mass" for an additive logit bias with an identity-alignment motivation. The closest is ConViT's α sweep, which covers α ≤ 1 and uses a gated, not additive, prior.
- Work on attention entropy collapse (Zhai et al. 2023, arXiv 2303.06296) is relevant to saturation but was not retrieved here (UNVERIFIED).
