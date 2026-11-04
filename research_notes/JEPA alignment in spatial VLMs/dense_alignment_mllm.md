# Dense auxiliary alignment / feature-prediction losses on MLLM hidden states (2024–2026)

Scope: methods that add a non-autoregressive auxiliary loss to a VLM's LLM hidden states so they encode more visual or geometric information. Compiled 2026-10-26. All numbers come from the arXiv HTML/abs pages fetched (version noted where it matters). Where a page was truncated or a number was not visible, that is said explicitly. "Unverified" marks claims taken from a search snippet or a secondary summary rather than from the paper page itself.

Note on key questions: the first section is a per-method catalogue (requested record format), followed by the five key questions.

---

## Per-method catalogue (target encoder / layers / tokens / loss / weight / gains / ablations)

### Takeaway
The dominant recipe (VIRAL, VaLR, PRe, ANCHOR, Spatial Forcing, ROCKET, GLaD, PokeVLA, VEGA) is: a frozen teacher encoder, a 2–3 layer MLP projector on the LLM hidden states at visual-token positions at one intermediate layer, negative cosine similarity (occasionally MSE), weight 0.1–0.5 added to the next-token loss, teacher discarded at inference. ROSS/Ross3D instead use a diffusion denoiser on a frozen VAE latent; OLA-VLM/VisPer-LM uses Perceiver-resampler probes with smooth-L1 + InfoNCE at several layers; JARVIS uses an I-JEPA masked-prediction objective with the LLM's first quarter as the predictor and a frozen DINOv2 target (no EMA).

### Cited Findings

**ROSS — Reconstructive Visual Instruction Tuning** (arXiv 2410.09575, v2 Dec 2024; Wang, Zheng, Zhao, Wang, Ge, Zhang, Zhang; later ICLR 2025 — venue not on the abs page)
- Target: latents of a frozen KL-16 continuous VAE (Rombach et al.) — not SigLIP/DINO (Sec. 4.2, App. B). Alternatives tested: VQ-16, DINOv2, DEiT-III, EVA02-CLIP, raw pixels (Sec. 5.1, Fig. 6, App. C.1). — [ROSS HTML v2](https://arxiv.org/html/2410.09575v2)
- Tokens/layer: loss uses only LLM outputs at the visual-input positions x_{i<=N}; the text loss covers text tokens only (Sec. 3–4). The HTML text does not state which LLM layer feeds the denoiser (described only as "high-level visual outputs"); VIRAL's Table 6 compares "ROSS (Default)" vs "ROSS (Middle)", implying the default is the final layer (inference, see Gaps). — [ROSS HTML v2](https://arxiv.org/html/2410.09575v2); [VIRAL HTML v2](https://arxiv.org/html/2509.07979v2)
- Loss: diffusion-style noise-prediction MSE, E||J_pi(z_t; x_{i<=N}, t) − eps||^2 (Eq. 4); denoiser is a stack of Transformer-encoder blocks with projections for condition, noisy input and timestep (Fig. 4). Total loss L_text + L_visual with no explicit weight (Sec. 4). — [ROSS HTML v2](https://arxiv.org/html/2410.09575v2)
- Training: Stage I projector + denoiser trainable, LLM frozen; Stage II projector + LLM + denoiser trainable; SigLIP encoder frozen in both (Table 6). Overhead 1.03–1.11× per iteration (Table 10). — [ROSS HTML v2](https://arxiv.org/html/2410.09575v2)
- Resolution handling: SigLIP-384 gives 729 tokens; KL-16 downsamples by 16, so the VAE is fed a 432×432 image (sqrt(729)×16) to produce exactly 729 latent targets, i.e. one-to-one correspondence by construction (App. B). — [ROSS HTML v2](https://arxiv.org/html/2410.09575v2)
- Gains (Tables 3–4): MMVP 28.0→36.0 (Vicuna-7B/CLIP), 29.3→42.0 (Qwen2-7B/CLIP), 40.7→49.3 (Qwen2-7B/SigLIP); HallusionBench 57.3→58.2 (Qwen2-7B/SigLIP). CV-Bench not present on the page; RealWorldQA listed in Table 12 but its numbers were truncated in the fetch. — [ROSS HTML v2](https://arxiv.org/html/2410.09575v2)
- Ablations: denoising beats plain MSE regression on both KL-16 and DINOv2 targets (Fig. 7, Sec. 5.1); pixel regression "fails to provide meaningful supervision" (Fig. 5); KL-16 best among tokenizers (Fig. 6); generative (text-to-image) training on the same data gives HallusionBench 52.0 vs 58.0 for reconstruction (Table 2). Small drops: MMBench-CN 60.0→59.8, SEED-Img 66.7→66.4 (Table 12); AI2D −0.2 for Vicuna/CLIP (Table 14); OCRBench −5 at 3B in the scaling table. Adding a MiDaS-3.0 depth expert as an extra input "fails to help" and "sometimes harms comprehension" (Table 5, Fig. 16). Robust across all tested beta schedules (Table 11). — [ROSS HTML v2](https://arxiv.org/html/2410.09575v2)

**Ross3D — Reconstructive Visual Instruction Tuning with 3D-Awareness** (arXiv 2504.01901, Apr 2025; Wang, Zhao, Wang, Fan, Zhang, Zhang)
- Target: frozen continuous KL-regularised VAE from FLUX (Sec. 4.4). Cross-view objective: VAE latents of masked views only (mask ratio 25%, Eq. 5); global-view objective: latent of a rendered 432×432 BEV image (Eq. 7, Table S4). — [Ross3D HTML](https://arxiv.org/html/2504.01901)
- Tokens/layer: DiT-based denoiser conditioned on the LLM's visual-token outputs only (language tokens excluded because causal attention means they do not affect visual positions); layer not stated (Sec. 4.1–4.2, Fig. 3). — [Ross3D HTML](https://arxiv.org/html/2504.01901)
- Loss: denoising MSE (Eq. 6–9); cross-view loss applied every 4 steps (Δt=1,2 worse on SQA3D, Table S3). No numeric weight vs cross-entropy given. — [Ross3D HTML](https://arxiv.org/html/2504.01901)
- Base: LLaVA-Video-7B, vision encoder frozen, 32 frames at 384², ~223K samples, 1 epoch (Sec. 5, App. A.2). — [Ross3D HTML](https://arxiv.org/html/2504.01901)
- Gains vs the same model without reconstruction (Table 4): SQA3D EM 58.6→63.0, ScanQA CIDEr 102.1→107.0, ScanRefer Acc@0.25 58.1→61.1. Cross-view alone 60.0/103.6/60.3; global-view alone 61.6/105.6/58.8. Vanilla (non-3D-aware) reconstruction gives only +0.2/+1.4/+0.1. VSI-Bench not reported. Cost 1.12× training time (Table 7). — [Ross3D HTML](https://arxiv.org/html/2504.01901)
- Semi-supervised (Table 6): 50% text + 3D loss on the other unlabeled 50% gives ScanQA CIDEr 103.2 > 102.1 full-text baseline, but SQA3D 57.7 < 58.6 and ScanRefer 57.9 < 58.1. — [Ross3D HTML](https://arxiv.org/html/2504.01901)

**VIRAL — Visual Representation Alignment for MLLMs** (arXiv 2509.07979, v2 Oct 2025; Yoon, Jung, Kim, ... Seungryong Kim; ICLR 2026)
- Target: DINOv2 by default; CLIP, SAM, Depth Anything v2, RADIO compared (Table 2). Targets are treated as fixed; gradients do not flow into the VFM (per search snippet of the paper). — [VIRAL HTML v2](https://arxiv.org/html/2509.07979v2); [ICLR 2026 listing](https://iclr.cc/virtual/2026/10013230); [GitHub](https://github.com/cvlab-kaist/VIRAL)
- Layer/tokens: single intermediate layer, 16 of 32; N visual-token positions only (Sec. 4.2, 5.3). — [VIRAL HTML v2](https://arxiv.org/html/2509.07979v2)
- Loss: negative mean cosine similarity between a 3-layer SiLU MLP projection of the layer-16 visual hidden states and the VFM patch features (Eq. 6); L_total = L_LM + λ L_VRA, λ = 0.5 (Eq. 5, App. A). LoRA instruction tuning on LLaVA-665K, ~5.2K steps, ~3% time overhead (Sec. 5.1, 5.7). — [VIRAL HTML v2](https://arxiv.org/html/2509.07979v2)
- Resolution: 576 CLIP-ViT-L/14@336 tokens (24×24); patch-14 VFMs (DINOv2, CLIP, DAv2) are run at 336×336 so grids match; patch-16 models (RADIO, SAM) at 384×384; SAM features are interpolated, padded to 1024² and cropped back (App. A). — [VIRAL HTML v2](https://arxiv.org/html/2509.07979v2)
- Gains (Table 1, Vicuna-7B/CLIP): CV-Bench2D 56.82→59.67, MMVP 28.20→33.33, What's Up 40.13→48.55, POPE 85.70→88.32, MMStar 33.93→33.93, MME 1650→1695. Vicuna-13B: What's Up 44.44→62.26, MMVP 32.30→37.80. Qwen2.5-7B/CLIP: POPE drops 85.88→84.92. CV-Bench 3D and RealWorldQA not reported. — [VIRAL HTML v2](https://arxiv.org/html/2509.07979v2)
- Layer sweep (Table 2, DINOv2): layer 4: 58.55/30.67 (CV-Bench2D/MMVP); 8: 58.28/27.70; 12: 57.77/28.59; 16: 59.67/33.33; layers 20–32 mostly lower (e.g. 54.87 CV-Bench2D at layer 28). Multi-layer targets worse: layers 15–17 59.32/28.00; 14–18 49.62/22.55 (Table 5). — [VIRAL HTML v2](https://arxiv.org/html/2509.07979v2)
- VFM sweep at layer 16 (Table 2): DINOv2 59.67/33.33; CLIP 59.53/29.33; SAM 57.58 CV-Bench2D but best What's Up 49.84; DAv2 58.55/28.67; RADIO 57.59/31.80. — [VIRAL HTML v2](https://arxiv.org/html/2509.07979v2)
- Negative/other: relation (self-similarity MSE) loss underperforms cosine (Table 5); pre-projection residual injection is below baseline (MMVP 19.33 vs 28.20, App. B Table 4-d); MMVP gains "marginal or slightly negative" when aligning to the model's own encoder features (Sec. 4.3); token-shuffle permutation test hurts VIRAL more than baseline (−54/−83 vs −26/−21, Table 3), read as spatial sensitivity; no λ sweep reported; limitations say linear projectors needed to keep the grid, Resampler/Q-Former less compatible, token pruning may reduce benefit, mid-layer finding may not generalise (App. G). — [VIRAL HTML v2](https://arxiv.org/html/2509.07979v2)

**OLA-VLM / VisPer-LM** (arXiv 2412.09585; v1 Dec 2024 titled "OLA-VLM: Elevating Visual Perception in Multimodal LLMs with Auxiliary Embedding Distillation"; v3 Oct 2025 retitled "Elevating Visual Perception in Multimodal LLMs with Visual Embedding Distillation", method renamed VisPer-LM; Jain, Yang, Shi, Gao, Yang)
- Targets: depth = DINOv2-L backbone of Depth Anything v2 (1024-d, 576 tokens); segmentation = Swin-L from OneFormer (1536-d, 576 tokens); generation = CLIP-ViT-L of unCLIP-SD-2.1 (1 token). Freezing not stated explicitly, but during PT only the projector, embedding predictors and special tokens train (Sec. 5.1). — [VisPer-LM HTML v3](https://arxiv.org/html/2412.09585v3); [OLA-VLM v1 abs](https://arxiv.org/abs/2412.09585v1)
- Layers/tokens (32-layer Llama3-8B): depth at layers {8, 20}, seg at {10, 18}, gen at {12, 20}; N=8 learnable special tokens appended after image tokens; a single-layer Perceiver Resampler per (target, layer) with 576/576/1 learnable queries reads keys from system, image, special and text positions (Sec. 4.1, 5.1, Table III). — [VisPer-LM HTML v3](https://arxiv.org/html/2412.09585v3)
- Loss: smooth-L1 (λ=1) + InfoNCE contrastive (λ=0.3, learnable τ init 2.0); per-task weights λ_depth=λ_seg=λ_gen=0.5; added to NTP during PT only (LLaVA-558K). Adding it during IFT hurt (Table 6). — [VisPer-LM HTML v3](https://arxiv.org/html/2412.09585v3)
- Resolution: handled by the resampler's fixed query count (576), not by patch correspondence (Sec. 3, 4.1). — [VisPer-LM HTML v3](https://arxiv.org/html/2412.09585v3)
- Gains (Table 1, Llama3-8B/CLIP-ViT-L): CV-Bench Depth 73.3→74.2, Distance 48.7→54.3, Relation 64.9→69.4, Count 50.4→51.3; MMStar 38.8→39.5; RealWorldQA 57.8→57.9; OK-VQA 56.9→56.6; avg 55.1→57.6. The headline "+8.7 on Depth" is for CLIP-ConvNeXt-XXL (62.8→71.5). BLINK avg 55.6→58.8, Spatial Relation 72.0→75.5, Relative Depth flat (51.4→51.6) (Table 4). With VPT stage Depth is lower (75.0 vs 76.8, Table 3). — [VisPer-LM HTML v3](https://arxiv.org/html/2412.09585v3)
- Ablations: probing shows middle layers 12–24 best for depth/seg probe quality and probe quality correlates with CV-Bench depth accuracy at r=0.98 (Sec. 3, Fig. 3, App. D); chosen layer sets best (54.1 avg vs 51.7–52.4, Table 5); N=8 tokens best (N=0: 53.6, 16: 52.9, 24: 53.1, Table II); learnable tokens at IFT worse than frozen (52.3 vs 55.1, Table 7); POPE/GQA/MMMU/VizWiz not degraded (Table 2); same throughput as single-encoder (9.86 vs 9.92 samples/s, Table 8). Fixed seed, no error bars (checklist). — [VisPer-LM HTML v3](https://arxiv.org/html/2412.09585v3)

**VaLR — Vision-aligned Latent Reasoning** (arXiv 2602.04476, Feb 2026; Jeon, Jeong, Lee, Cho, Shin — KAIST/POSTECH)
- Targets: DINOv3 default (VaLR-S); DINOv2, CLIP, SigLIPv2 and π³ (a 3D feed-forward reconstruction model) tested; VaLR-M combines DINOv3 + SigLIPv2 + π³. Encoders frozen; only the MLLM decoder trained (Sec. 3.2–3.3, 4.5). Base Qwen2.5-VL-7B. — [VaLR HTML](https://arxiv.org/html/2602.04476v1)
- Layer/tokens: intermediate layer (middle = 12th best; front = 4th and last = 27th worse, Table 6); aligned states are only the K=16 latent reasoning tokens produced at each reasoning stage, not the original visual tokens or language tokens (Sec. 3.2, Table 7, Table 12). — [VaLR HTML](https://arxiv.org/html/2602.04476v1)
- Loss: "REPA loss" = negative mean patch-wise cosine between an MLP projection of the *upsampled* latent-token features and encoder patch features, averaged over stages and patches; L = L_CE + λ L_REPA with λ = 0.5 (λ ∈ [0,1] swept, 0.5 best on VSI-Bench/BLINK/V*, Table 11). Per-encoder heads averaged for multi-encoder. Alignment only in training. — [VaLR HTML](https://arxiv.org/html/2602.04476v1)
- Resolution: an "Upsample" op maps the K latent tokens to the encoder patch grid; the mechanism is not described in the visible text. — [VaLR HTML](https://arxiv.org/html/2602.04476v1)
- Gains (Tables 1–2): VSI-Bench Qwen2.5-VL-7B 33.0 → SFT 33.7 → VaLR-S 41.5 → VaLR-M 52.9; BLINK 55.7→63.1/64.7; MMVP 56.0→60.3; CVBench 74.5→83.1/87.6; V* 76.4→86.4/86.9; MMStar 67.1→70.8/72.3. Without visual alignment (same latent tokens): VSI 34.0, MMVP 56.7, CVBench 73.4 (below SFT) (Table 3). π³ as target mainly moves VSI-Bench (Table 4). Single encoders (Table 5): CLIP 79.1 CVBench, SigLIPv2 81.9, DINOv2 81.8, DINOv3 83.1. ~20× data efficiency vs SFT on V* (Fig. 3). Training 450K samples, 1 epoch, 4×A100 (App. B.1). No EMA; no language-only regression reported. — [VaLR HTML](https://arxiv.org/html/2602.04476v1)

**JARVIS — Seeing Beyond Words (I-JEPA inside LLaVA)** (arXiv 2512.15885, Dec 2025; Caffagni, Sarto, Cornia, Baraldi, Dovesi, Roohi, Granroth-Wilding, Cucchiara — UniMoRE / AMD Silo AI)
- Target: frozen DINOv2-L/14 as target encoder; context encoder = the MLLM's own frozen CLIP ViT-L/14@336; target projector = 2-layer GELU MLP. **No EMA/momentum teacher** (Sec. 3.2, 4). — [JARVIS HTML](https://arxiv.org/html/2512.15885)
- Predictor/layer: the LLM's first quarter of layers acts as the I-JEPA predictor; loss read at layer j = depth/4 (j=8 for Vicuna-7B). Input [I_ctx, z_tgt], z_tgt = learnable embedding + positional encoding of each masked patch; attention mask: context cannot attend targets, targets from different blocks cannot attend each other, captions causal and cannot influence visual tokens. Masking is on frozen visual embeddings (context blocks 85–100% of image, k=4 target blocks). — [JARVIS HTML](https://arxiv.org/html/2512.15885)
- Loss: negative cosine (smooth-L1 tested, worse: 49.1 vs 51.7 VC-avg); L_NTP + L_JEPA with no weight; the loss is skipped with probability λ=0.2 per step. Applied in the LLaVA alignment stage. — [JARVIS HTML](https://arxiv.org/html/2512.15885)
- Gains (Table 2; RWQA/MMVP/Blink/CVB2D/CVB3D/VC-avg): Vicuna-7B 54.9/31.3/46.8/57.6/63.7/50.9 → 55.6/30.7/48.3/60.4/63.8/51.7; Qwen2-7B/CLIP 58.0/36.7/52.7/62.5/66.8/55.3 → 55.8/38.0/49.3/63.9/73.0/56.0 (RWQA −2.2, Blink −3.4, CVB3D +6.2); Qwen2-7B/SigLIP2 57.5/48.5/48.0/65.8/68.1/57.6 → 59.6/53.1/50.0/67.4/66.8/59.4. Gemma2-2B 50.0→50.8, LLaMA-3.2-3B 52.1→53.5, Ministral-8B 51.5→53.1 (Table 3). General/Knowledge/OCR roughly stable. — [JARVIS HTML](https://arxiv.org/html/2512.15885)
- Layer ablation (Table 1, VC-avg): j=4 50.4, j=8 51.7, j=16 50.3, j=24 50.2, j=31 49.2, j=32 49.4 (baseline 50.9). Linear projector 49.6 (< baseline). DINOv2 best teacher vs CLIP/SigLIP2/DINOv3 at B/L/g/H scales (Fig. 5). VIRAL and a "w/o masking" variant (plain intermediate alignment) "do not improve over LLaVA on vision-centric tasks" in their setup (Table 2). No collapse/stability discussion. — [JARVIS HTML](https://arxiv.org/html/2512.15885)

**HeRA — Mind the Heads: Topological Representation Alignment** (arXiv 2606.23885, 2026; Caffagni, Compagnoni, Melis, Sarto, Dovesi, Granroth-Wilding, Cornia, Baraldi)
- Target: frozen DINOv2 ViT-L [CLS] token; student = per-head attention outputs (projected by W_O,h) of the m=5 least mutual-kNN-aligned heads, selected once on the base LLM with 1,000 GranD samples; text tokens average-pooled (Sec. 3.3, 4.1, App. A). — [HeRA HTML](https://arxiv.org/html/2606.23885)
- Loss: multi-positive InfoNCE over k=10 teacher-space neighbours, learnable temperature init 0.07; λ = 0.01; both LLaVA stages (Eq. 4, 8, 9). — [HeRA HTML](https://arxiv.org/html/2606.23885)
- Gains (Table 2 vision-centric avg): Qwen2.5-3B 50.5→52.9; Qwen3-4B 56.3→58.5; Vicuna-7B 49.7→52.0; Qwen3-8B 55.9→59.5; Qwen2.5-14B 54.9→58.3; Qwen3-14B 58.2→58.9. Table 4 (Qwen3-8B, Vision avg / CVBench / MMVP / RWQA): LLaVA 55.9/67.8/49.3/59.2; ROSS 56.2/69.7/50.3/59.8; **VIRAL 54.2 (−1.7)/69.3/46.7/57.6**; JARVIS 58.7/69.9/54.7/59.2; CMAR 56.9/70.9/50.0/58.8; HeRA 59.5/73.5/58.0/60.1. — [HeRA HTML](https://arxiv.org/html/2606.23885)
- Ablations (Table 1, Qwen2.5-3B "All" avg, baseline 54.2): VIRAL-style feature cosine at middle layer 53.8 (regression); contrastive at middle layer 53.8; Worst-10 heads 46.0 (large regression); Worst-5 55.7. Stage-2-only (49.7) and stage-1-only (55.5) each fall below their baselines on one model (Table 7). SigLIP2 as teacher "largely ineffective"; DINOv2-g (1B) no clear gain (Sec. 4.4). Hallucination regressions on Vicuna-7B/13B (CHAIRs up), Vicuna-13B RWQA 58.7→57.3 (Tables 2–3). — [HeRA HTML](https://arxiv.org/html/2606.23885)

**PRe — Predictive Regularization Against Visual Representation Degradation** (arXiv 2603.20808, Mar 2026; Wang et al., Tencent Youtu Lab / Nankai)
- Target: the MLLM's *own* projected visual features before entering the LLM ("Pre-LLM"), detached with stop-gradient; alternatives Pre-Proj, DINOv2, DINOv3, SAM (Tables 3, 5). Not EMA. — [PRe HTML](https://arxiv.org/html/2603.20808)
- Layer/tokens: visual tokens at middle layer (16 for Vicuna, 14 for Qwen); 2-layer MLP predictor; patch-level loss (Fig. 10). Last layer worse than mid layer, attributed via logit lens to deep-layer visual states collapsing toward high-frequency tokens (Table 2, Fig. 9). — [PRe HTML](https://arxiv.org/html/2603.20808)
- Loss: negative cosine, λ = 0.5; λ = 0.2 gives only marginal gains, λ = 2.0 degrades (Fig. 11–12). Overhead +0.045% training PFLOPs; head discarded at inference (Table 6). — [PRe HTML](https://arxiv.org/html/2603.20808)
- Gains (Table 1): Vicuna-7B/frozen CLIP RWQA 54.8→55.4, MMVP 20.0→22.0; Qwen2.5-7B/trainable SigLIP2 RWQA 60.3→61.9, MMVP 46.0→46.7; 3B results mixed with an OCRBench drop for frozen-CLIP (Table 4). CV-Bench/VSI-Bench not visible. Paper argues aligning to an external space such as DINOv2 "may introduce conflicting objectives" due to a representational gap. — [PRe HTML](https://arxiv.org/html/2603.20808)

**ANCHOR — Proactive Visual Anchoring (multilingual MLLMs)** (arXiv 2608.15085, 2026; Du, Liang, Lai, Li, Hu — CUHK-Shenzhen)
- Target: SigLIP-SO400M/14 features (external VFM; freezing not stated); loss at early layer L=10 on all visual tokens; 2-layer GELU MLP; negative mean cosine; α = 0.4 (Eq. 1–2, App. D). — [ANCHOR HTML](https://arxiv.org/html/2608.15085)
- Layer ablation (Table 9, LLaVA-1.5-7B, xMMMU/MaXM): L=6 32.8/28.5; L=8 34.2/30.7; L=10 35.6/32.4; L=12 34.7/31.8; L=14 33.5/30.2; L=16 32.1/28.9 — an inverted U peaking early. No α sweep. Benchmarks are multilingual (xMMMU, MaXM, CVQA), not spatial; on the 7B CVQA subset ANCHOR (42.7) is below base (43.4). — [ANCHOR HTML](https://arxiv.org/html/2608.15085)

**SSVAL — Spatial-Spectral Visual Anchor Learning** (arXiv 2608.01635, Aug 2026, ACM MM 2026; Yang et al.)
- Learnable "visual anchor" prompts injected into the LLM absorb knowledge from external VFMs via spatial- and frequency-domain alignment losses at intermediate LLM layers; abstract claims direct VFM alignment "improves visual semantics but does not fix" inference-time representation drift. No layers, weights or numbers on the abs page (unverified beyond abstract). — [SSVAL abs](https://arxiv.org/abs/2608.01635)

**Spatial Forcing (VLA)** (arXiv 2510.12276, v2 Oct 2025; Li, Song, Zhao, Wang, Ding, Wang, Zeng, Li)
- Target: latent representation of the frozen VGGT transformer backbone (not depth maps), with a positional embedding added to targets (Sec. 2.1, 2.3, Eq. 3). Backbones OpenVLA-OFT (LIBERO) and π0 (RoboTwin). — [Spatial Forcing HTML v2](https://arxiv.org/html/2510.12276v2)
- Layer/tokens: visual tokens at layer 24 of 32 ("relatively deep but not the deepest"); batch-norm + 2-layer MLP; negative cosine; L_action + α L_align with α = 0.5 (Sec. 3.3, Eq. 3–4, App. A). Each VLA visual token is matched to the VGGT feature "corresponding to the pixel location" of that token; no interpolation details. — [Spatial Forcing HTML v2](https://arxiv.org/html/2510.12276v2)
- Layer sweep (Table 2, LIBERO avg, baseline 92.7): layer 1 94.6, 8 95.7, 16 93.8, 24 96.9, 32 94.8. Long-horizon suite: baseline 86.2; layer 1 83.0, 16 83.2, 32 84.8, VGGT w/o PE 84.4, layer 24 + PE 94.2. — [Spatial Forcing HTML v2](https://arxiv.org/html/2510.12276v2)
- Target sweep (Table 2, layer 24): SigLIP 94.0 avg, DINOv2 94.1, VGGT w/o PE 94.7, VGGT + PE 96.9. Depth estimators not compared. — [Spatial Forcing HTML v2](https://arxiv.org/html/2510.12276v2)
- α sweep (App. A): α=0: 73.2; 0.02: 92.2; 0.1: 92.8; 0.5: 93.6; 2.5: 86.6; 12.5: 81.2 — large weight "may destabilize the VLA visual modality". — [Spatial Forcing HTML v2 (appendix)](https://arxiv.org/html/2510.12276v2)
- Main: LIBERO 98.5 avg vs OpenVLA-OFT 97.1; reaches baseline SR 3.8× faster; 1%/5% data 42.3/75.8 (Table 1, Fig. 5a, Table 2). Depth probing (DPT head on frozen VLA) is qualitative only (Fig. 1c, Fig. 3). — [Spatial Forcing HTML v2](https://arxiv.org/html/2510.12276v2)

**ROCKET — Residual-Oriented Multi-Layer Alignment (VLA)** (arXiv 2602.17951, Feb 2026; Sun et al.)
- Target: frozen 3D-aware VFM, VGGT main example, π³ and Depth Anything 3 also named (Sec. 3.1). Visual tokens at N=10 layer pairs (default E2M-Last1: early-to-middle + last), one shared 2-layer GELU projector with Matryoshka-style prefix masking (ρ 0.2→1.0); per-pair 1 − cos; λ = 0.5 (Eqs. 12–16, Sec. 4.3). — [ROCKET HTML](https://arxiv.org/html/2602.17951)
- Gradient interference: with independent per-layer projectors, cross-layer gradients become nearly orthogonal (Fig. 1, App. E) and multi-layer alignment *drops* LIBERO avg from 96.4 to 80.0 (Goal 97.6→42.2); shared projector 98.2; + Matryoshka 98.5 (Table 8). Layer count: 4 pairs 96.3, 8: 97.4, 16: 96.7, 24: 94.1 (Table 4). Uniform-8 (97.4) beats default E2M-Last1 (96.3) in Table 5. — [ROCKET HTML](https://arxiv.org/html/2602.17951)
- Results: LIBERO 98.5 (= Spatial Forcing 98.5) at claimed 1.0× vs 24.0× tuning cost (Tables 2, 7); PI0.5 full FT 93.0→95.3 (SF 94.0, Table 3); LIBERO-Plus 80.0→81.7. — [ROCKET HTML](https://arxiv.org/html/2602.17951)

**GLaD — Geometric Latent Distillation (VLA)** (arXiv 2512.09619, Dec 2025; Guo et al.)
- Target: frozen VGGT spatio-temporal features (d=2048), adaptive-pooled to the VLM's Np patch tokens, last-frame aggregation (Sec. III-A). Student: hidden states at image-token positions at the **final** LLM layer (32/32); 2-layer MLP; **MSE** loss; λ "tuned on validation" (Eqs. 1–4). — [GLaD HTML](https://arxiv.org/html/2512.09619)
- Results (Table I): LIBERO 94.1 vs UniVLA 92.5 (Spatial 95.0 vs 95.2 — not uniformly better). Ablation (Table III): π³ teacher 86.1 (Spatial 65.2); layer 24/32 92.6; early weighted fusion 84.0. Sec. V-C: explicit depth-map supervision caused "training divergence due to conflicting objectives"; implicit contrastive supervision did not beat baselines. — [GLaD HTML](https://arxiv.org/html/2512.09619)

**VEGA — Visual Encoder Grounding Alignment (VLA)** (arXiv 2605.10485, May 2026)
- Aligns the DINOv2 branch of OpenVLA-OFT's encoder (block L−2) to a frozen DINOv2-FiT3D teacher — i.e. encoder-level, not LLM-level — with per-patch cosine, λ = 0.1 (0.05/0.2 worse). Teacher ablation on one task: no teacher 0.70/0.34 (easy/hard), VGGT 0.76/0.04, FiT3D 0.77/0.43 — VGGT "collapse" on Hard attributed to architectural mismatch with ViT patch tokens (Table 2). The argument that LLM-level alignment entangles geometry with language is not tested. — [VEGA HTML](https://arxiv.org/html/2605.10485v1)

**PokeVLA** (arXiv 2604.20834, Apr 2026) — abstract mentions a "geometry alignment" stage; a search snippet states it aligns last-layer vision-token hidden states to VGGT features via a projector with a 1 − cosine loss, training only (unverified: not on the abs page). — [PokeVLA abs](https://arxiv.org/abs/2604.20834)

**3DThinker — Think with 3D** (arXiv 2510.18632, v4 Mar 2026; Chen et al.)
- Aligns the last-layer hidden states of a block of 12 special "3D latent" tokens (not visual tokens) to the last-layer VGGT aggregator features via a projector; loss = squared Frobenius norm, λ_3D = 0.1 vs λ_text = 1 (Eq. 5, 9); Stage 2 uses a cosine reward instead (Eq. 12). VGGT freezing not stated; grid reconciliation not described. VSI-Bench Qwen2.5-VL-7B 35.8 → 57.3 (S1) → 63.7 (S1+S2) (Table 2) — no with/without-alignment ablation on VSI-Bench visible; removing the Stage-2 3D reward: MindCube-Tiny 75.2→68.3 (Table 6); latent size 32/64 collapse to 25.1/15.5 (Table 5); placing the 3D tokens mid-sequence drops to 42.0 (Table 6). — [3DThinker HTML](https://arxiv.org/html/2510.18632)

**Render to Reason** (arXiv 2610.05417, Oct 2026; Wu, Xiao, Zou, Wang, Hoiem)
- 256 learnable `<SEM_i>` query tokens; LLM hidden states → 2-layer MLP → 9-class semantic logits for a 16×16 grid of an *unseen* view (labels from Mask2Former/InternImage-H); cross-entropy, λ_sem = 1; frozen VGGT supplies geometry (camera token of target view) (Sec. 3.2, Eq. 2–3). — [Render to Reason HTML](https://arxiv.org/html/2610.05417v1)
- VSI-Bench (Table 3a, InternVL-4B): finetuned 62.9; + VGGT features 63.0; + VGGT + Sem 64.6. Qwen3-VL-4B: 66.0→67.0→68.2. **Predicting VGGT latents at novel views reduced VSI-Bench by 1.5 vs the geometry-augmented baseline** (App. A.3). Sem without VGGT 61.6 (−1.3); RGB target 62.4; grid 8×8 64.9 / 16×16 64.6 / 32×32 64.3 (Tables 3d, 3f). — [Render to Reason HTML](https://arxiv.org/html/2610.05417v1)

**PointAlign** (arXiv 2603.00412, CVPR 2026; Su, Zhang, Jia, Fan) — adds a feature-level consistency loss constraining intermediate point-cloud tokens inside the LLM to align with the visual input tokens, next to NTP; only projector + LoRA train; +2.08 pts avg classification (ModelNet40/Objaverse), +7.50 open-vocab Objaverse, +4.88 captioning. Exact target/loss not on the abs page. — [PointAlign abs](https://arxiv.org/abs/2603.00412)

**Perception Tokens / AURORA** (arXiv 2412.03548, CVPR 2025; Bigverdi et al.) — not a hidden-state loss: depth maps are VQ-VAE-tokenised and *generated autoregressively* as intermediate reasoning tokens; +10.8 BLINK counting, +11.3 CV-Bench, +8.3 SEED-Bench, >+6 BLINK relative depth (abstract-level, from search). — [Perception Tokens abs](https://arxiv.org/abs/2412.03548v2); [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Bigverdi_Perception_Tokens_Enhance_Visual_Reasoning_in_Multimodal_Language_Models_CVPR_2025_paper.html)

**DepthLM** (arXiv 2509.25413, ICLR 2026; Cai et al., Meta/Princeton) — contrast case: metric depth via *text-only* SFT with visual prompting and intrinsic-conditioned augmentation, no dense head or regression loss; 3B model >2× δ1 over GPT-5 class VLMs, beats DepthPro/Metric3Dv2 (abstract-level). — [DepthLM HF page](https://huggingface.co/papers/2509.25413); [ICLR 2026 oral](https://iclr.cc/virtual/2026/oral/10009768)

**Reconstruction-flavoured relatives (brief):**
- ASVR (arXiv 2506.09040): autoregressive reconstruction of *discrete semantic* visual tokens; +~5% avg over LLaVA-1.5 on 14 benchmarks; reconstructing raw appearance "may even impair multimodal understanding". — [ASVR abs](https://arxiv.org/abs/2506.09040)
- LaVer (arXiv 2512.06281): masked image modelling in the LLM's latent space; claims gains on dense visual tasks; no numbers on abs page. — [LaVer abs](https://arxiv.org/abs/2512.06281)
- GAS (arXiv 2608.12209): Next-Embedding-Prediction generation branch in a decoupled Mixture-of-Transformers (shared lower trunk, separate upper layers), branch discarded at inference; gains "strongest on perception and spatial comprehension"; no numbers on abs page. — [GAS abs](https://arxiv.org/abs/2608.12209)
- Latent Denoising (arXiv 2604.21343, ACM MM 2026): corrupt projected visual tokens, reconstruct clean teacher patch features from an intermediate LLM layer, plus a similarity-structure-preservation term "to prevent representation collapse" and intra-image contrastive patch distillation. — [Latent Denoising abs](https://arxiv.org/abs/2604.21343)
- GeoScaffold (arXiv 2610.02697): learnable geometry query tokens whose hidden states reconstruct depth/connectivity/traversability via a frozen depth tokenizer, for VLN. — [GeoScaffold abs](https://arxiv.org/abs/2610.02697)
- Imagine3D-LLM (arXiv 2609.38177, NeurIPS 2026): summary tokens decoded into 3D Gaussian Splatting with a photometric loss, jointly with NTP. — [Imagine3D-LLM abs](https://arxiv.org/abs/2609.38177)
- REPA (arXiv 2410.06940): the origin of the "REPA loss" naming used by VaLR; aligns projections of noisy DiT/SiT hidden states with clean pretrained-encoder features; >17.5× SiT speedup (abstract). Layer-8/DINOv2/λ=0.5 details not on the abs page (known from the paper but unverified here). — [REPA abs](https://arxiv.org/abs/2410.06940)

### Inferences
- Across MLLM papers the effective hyper-parameter envelope is narrow: single intermediate layer, visual tokens only, cosine, λ ≈ 0.4–0.5, MLP projector. VLA papers converge on the same recipe with VGGT as teacher.
- The two reported MSE-based LLM-state regressions (GLaD at the final layer; ROSS's plain-regression ablation) are the weaker variants in their own papers, suggesting cosine or denoising is preferable when the target is a semantic/geometric latent.

### Gaps
- ROSS/Ross3D never state the LLM layer feeding the denoiser in the fetched text; the "final layer" reading is inferred from VIRAL's Table 6 labels.
- VIRAL's claimed "17.3%/9.4%" average gains do not appear as such in its tables.
- OLA-VLM v1 vs VisPer-LM v3 may differ in numbers; only v3 was read in full.
- No OpenReview reviewer discussion could be fetched (bot check).

---

## KQ1. Which works regress LLM hidden states at visual-token positions onto frozen encoder features of the same image region (one-to-one patch correspondence), and how did they handle the resolution/grid mismatch?

### Takeaway
VIRAL, PRe, ANCHOR, Spatial Forcing, ROCKET, GLaD and (per snippet) PokeVLA do exact per-position alignment at visual-token positions; grid mismatch is mostly avoided by running the teacher at an input size whose patch grid equals the VLM's token grid (VIRAL, ROSS), or removed by adaptive pooling / interpolation of teacher features to the student's token count (GLaD, VIRAL-SAM), while OLA-VLM/VisPer-LM and VaLR sidestep correspondence with a resampler or an upsampling projector.

### Cited Findings
- VIRAL: loss over the N visual-token positions only; CLIP-ViT-L/14@336 gives 576 tokens; patch-14 teachers (DINOv2, CLIP, DAv2) are run at 336×336 to match; patch-16 teachers (RADIO, SAM) at 384×384; SAM features are interpolated, padded to 1024² and cropped back (App. A). Limitations say linear projectors are needed to keep the grid and Resampler/Q-Former designs are less compatible (App. G). — [VIRAL HTML v2](https://arxiv.org/html/2509.07979v2)
- ROSS: builds one-to-one correspondence by feeding the VAE a 432×432 image so KL-16 produces exactly 729 latents for the 729 SigLIP tokens (App. B). — [ROSS HTML v2](https://arxiv.org/html/2410.09575v2)
- PRe: patch-level negative cosine between a 2-layer MLP on middle-layer visual tokens and the detached pre-LLM projected features (same token count by construction); patch-level beats global (Fig. 10). — [PRe HTML](https://arxiv.org/html/2603.20808)
- ANCHOR: negative cosine over all visual tokens at layer 10 against SigLIP-SO400M features (Eq. 1); no mismatch procedure described. — [ANCHOR HTML](https://arxiv.org/html/2608.15085)
- Spatial Forcing: each VLA visual token is matched to the VGGT feature at the "corresponding pixel location"; positional embedding added to targets (removing it drops Long-horizon from 94.2 to 84.4, Table 2). — [Spatial Forcing HTML v2](https://arxiv.org/html/2510.12276v2)
- GLaD: VGGT features (T×L×2048) adaptive-pooled to the Np image-token positions, last-frame aggregation, MSE at final layer (Sec. III-A). — [GLaD HTML](https://arxiv.org/html/2512.09619)
- ROCKET: per-layer 1 − cos between a shared Matryoshka MLP projection of visual tokens and teacher features at 10 layer pairs (Eq. 16). — [ROCKET HTML](https://arxiv.org/html/2602.17951)
- OLA-VLM/VisPer-LM: no patch correspondence; a 1-layer Perceiver Resampler with 576 queries per target reads the whole sequence (Sec. 4.1). — [VisPer-LM HTML v3](https://arxiv.org/html/2412.09585v3)
- VaLR: alignment is on K=16 latent reasoning tokens that are "upsampled" to the encoder patch grid before the patch-wise cosine; the operation is not detailed (Sec. 3.2). — [VaLR HTML](https://arxiv.org/html/2602.04476v1)
- Render to Reason: 16×16 grid of 256 query tokens, 8×8 slightly better on VSI-Bench (64.9 vs 64.6), 16×16 best on 3D-Point-QA (Table 3f). — [Render to Reason HTML](https://arxiv.org/html/2610.05417v1)
- JARVIS: masks are applied on the frozen CLIP embeddings and targets are DINOv2-L features for the masked patches (both at /14, so grids match); a linear projector hurts (49.6 vs 51.7, Table 1). — [JARVIS HTML](https://arxiv.org/html/2512.15885)

### Inferences
- Nobody reports a careful study of mismatched grids (e.g. 2×2-merged Qwen2-VL tokens vs 14-px DINO patches); the practical choices are (a) resize the teacher's input so grids coincide, (b) pool/interpolate teacher features to the student grid. For a Qwen2.5-VL-style 28-px token grid, adaptive-pooling a 14-px teacher grid 2×2 (as GLaD does with VGGT) is the closest precedent.
- VIRAL's own permutation test (Table 3) suggests the alignment increases reliance on token order, which argues for keeping positional correspondence exact (and Spatial Forcing's PE ablation points the same way).

### Gaps
- VaLR's upsampling and Spatial Forcing's token→pixel mapping are not described; GLaD does not say which VGGT tokens (aggregator layer) it pools.
- No work was found that aligns Qwen2-VL/Qwen2.5-VL *visual tokens* (merged 2×2) to a 14-px teacher grid with an explicit description of the mismatch; VaLR uses Qwen2.5-VL but aligns latent tokens instead.

---

## KQ2. Has anyone used a depth or 3D encoder (Depth Anything, VGGT, DUSt3R, π³, point-cloud encoders) as the regression target for LLM hidden states, and did spatial benchmarks improve?

### Takeaway
Yes, but mostly in VLA/robotics (Spatial Forcing, ROCKET, GLaD, PokeVLA — VGGT) where success-rate gains are clear, and in VaLR (π³ as one of three targets) where π³ specifically drives the VSI-Bench gain; in pure VLMs the evidence is mixed: Depth Anything v2 as a VIRAL target was *not* better than DINOv2 on CV-Bench2D/MMVP, Render to Reason found predicting VGGT latents at novel views *hurt* VSI-Bench, and VEGA found VGGT as a teacher collapsed on hard tasks while a DINOv2-FiT3D teacher did not.

### Cited Findings
- VaLR: π³ as alignment target "mainly improves VSI-Bench"; VaLR-M (DINOv3 + SigLIPv2 + π³) reaches 52.9 VSI-Bench vs 41.5 for DINOv3-only and 33.0 for base Qwen2.5-VL-7B (Tables 1, 4); aligning with all four encoders incl. Qwen's own is best overall (Table 9). — [VaLR HTML](https://arxiv.org/html/2602.04476v1)
- VIRAL (Table 2, layer 16): Depth Anything v2 target gives CV-Bench2D 58.55 / MMVP 28.67 / MME 1682 vs DINOv2 59.67 / 33.33 / 1695; no CV-Bench 3D (depth/distance) numbers reported. — [VIRAL HTML v2](https://arxiv.org/html/2509.07979v2)
- OLA-VLM/VisPer-LM: Depth Anything v2 (DINOv2-L backbone) as a depth target at layers {8,20}: CV-Bench Depth 73.3→74.2 and Distance 48.7→54.3 (CLIP-ViT-L); Depth 62.8→71.5 with CLIP-ConvNeXt-XXL (Table 1); BLINK Relative Depth flat 51.4→51.6 (Table 4); with the extra VPT stage, Depth is lower than baseline (75.0 vs 76.8, Table 3). — [VisPer-LM HTML v3](https://arxiv.org/html/2412.09585v3)
- Spatial Forcing: VGGT target 96.9 LIBERO avg vs DINOv2 94.1, SigLIP 94.0, baseline 92.7 (Table 2); real-world +47.5% on a glass-cup task (Fig. 6). — [Spatial Forcing HTML v2](https://arxiv.org/html/2510.12276v2)
- GLaD: VGGT target 94.1 vs π³ target 86.1 (Spatial suite collapses to 65.2) (Table III); explicit depth-map supervision diverged (Sec. V-C). — [GLaD HTML](https://arxiv.org/html/2512.09619)
- ROCKET: VGGT (also π³ / Depth Anything 3 named) multi-layer target, LIBERO 98.5, PI0.5 93.0→95.3 (Tables 2–3). — [ROCKET HTML](https://arxiv.org/html/2602.17951)
- VEGA (encoder-level): VGGT teacher 0.76/0.04 vs DINOv2-FiT3D 0.77/0.43 vs none 0.70/0.34 (easy/hard, one task, Table 2), the VGGT failure attributed to "architectural mismatch with ViT patch tokens". — [VEGA HTML](https://arxiv.org/html/2605.10485v1)
- Render to Reason: predicting VGGT latents at novel views cut VSI-Bench by 1.5 vs the geometry-augmented baseline, while predicting 9-class semantic labels of the novel view added +1.6 (InternVL-4B 63.0→64.6; Qwen3-VL-4B 67.0→68.2) (Table 3a, App. A.3); simply feeding VGGT features gave +0.1/+1.0. — [Render to Reason HTML](https://arxiv.org/html/2610.05417v1)
- 3DThinker: Frobenius alignment of 12 special tokens to last-layer VGGT aggregator features, λ=0.1; VSI-Bench 35.8→57.3/63.7 (but no alignment-off ablation on VSI-Bench visible) (Table 2). — [3DThinker HTML](https://arxiv.org/html/2510.18632)
- Ross3D: 3D-aware *reconstruction* (VAE latents of masked views / BEV) gives SQA3D +4.4, ScanQA CIDEr +4.9 over a Video-3D-LLM baseline, while vanilla reconstruction gives +0.2/+1.4 (Table 4); VSI-Bench not reported. — [Ross3D HTML](https://arxiv.org/html/2504.01901)
- ROSS: MiDaS-3.0 depth maps as an *input* expert "fail to help" and "sometimes harm" (Table 5, Fig. 16). — [ROSS HTML v2](https://arxiv.org/html/2410.09575v2)
- PointAlign: alignment between intermediate point-cloud tokens and visual input tokens inside a 3D-LLM; +2.08 classification, +7.50 open-vocab, +4.88 captioning (abstract). — [PointAlign abs](https://arxiv.org/abs/2603.00412)
- GeoScaffold: geometry query tokens' hidden states reconstruct depth/connectivity/traversability through a frozen depth tokenizer (VLN; no numbers on abs page). — [GeoScaffold abs](https://arxiv.org/abs/2610.02697)

### Inferences
- The only VLM result where a 3D encoder target measurably moves VSI-Bench is VaLR's π³ target (on latent tokens, not visual tokens). For visual-token alignment with a depth/3D teacher in a VLM, the evidence is thin and two independent groups (Render to Reason, VEGA) report that raw VGGT latents are a worse target than semantic or ViT-matched alternatives.
- VLA gains with VGGT may transfer poorly to VQA-style spatial benchmarks: the VLA papers measure manipulation success with few-view inputs, and the one VLM paper that tried novel-view VGGT latents saw a drop.

### Gaps
- No paper found that uses DUSt3R/MASt3R features or a point-cloud encoder (e.g. Uni3D/PointBERT) as a regression target for a 2D VLM's visual-token hidden states.
- No VLM paper reports CV-Bench 3D (depth/distance) for a VGGT- or DAv2-aligned variant; OLA-VLM is the only one with CV-Bench depth/distance numbers.
- VIRAL's DAv2 run lacks CV-Bench 3D, so the natural "depth teacher → depth benchmark" test is absent.

---

## KQ3. Do any methods use an EMA/momentum target or a predictor head (JEPA/BYOL style) rather than a fixed frozen target, and what did they report about stability or collapse?

### Takeaway
No method found uses an EMA/momentum teacher on the LLM side; JARVIS (I-JEPA in LLaVA) explicitly uses a *frozen* DINOv2 target and the LLM's early layers as predictor, PRe uses a stop-gradient copy of the model's own pre-LLM features with a 2-layer predictor, and Latent Denoising adds an explicit similarity-structure term "to prevent representation collapse"; stability discussion elsewhere is limited to loss-weight blow-ups (Spatial Forcing, PRe) and GLaD's report of divergence with explicit depth-map supervision.

### Cited Findings
- JARVIS: target encoder is frozen DINOv2-L/14, context encoder frozen CLIP; "no EMA/momentum teacher"; predictor = first quarter of the LLM; loss read at layer depth/4; cosine beats smooth-L1 (51.7 vs 49.1); no collapse or stability discussion; no EMA-vs-frozen comparison (Sec. 3.2, 4; Table 1). — [JARVIS HTML](https://arxiv.org/html/2512.15885)
- PRe: 2-layer MLP predictor from middle-layer visual states to the *detached* (stop-gradient) pre-LLM projected features of the same model — a self-target rather than an external one; "not described as an EMA or momentum encoder"; stop-gradient is the only stability device; λ=2.0 degrades (Sec. 4.2, Eq. 2, Fig. 11–12). — [PRe HTML](https://arxiv.org/html/2603.20808)
- Latent Denoising: reconstructs clean teacher patch features from a corrupted-input forward pass and adds preservation of the teacher's intra-image similarity structure "to prevent representation collapse" plus intra-image contrastive patch distillation (abstract). — [Latent Denoising abs](https://arxiv.org/abs/2604.21343)
- Spatial Forcing: α=2.5 → 86.6 and α=12.5 → 81.2 vs 93.6 at α=0.5; "excessively large weight may destabilize the VLA visual modality" (App. A). — [Spatial Forcing HTML v2](https://arxiv.org/html/2510.12276v2)
- GLaD: explicit depth-map supervision caused "training divergence due to conflicting objectives"; implicit contrastive supervision did not beat baselines (Sec. V-C). — [GLaD HTML](https://arxiv.org/html/2512.09619)
- ROCKET: independent per-layer projectors make cross-layer gradients nearly orthogonal and drop LIBERO avg 96.4→80.0; a shared projector fixes it (Table 8, App. E). — [ROCKET HTML](https://arxiv.org/html/2602.17951)
- ROSS: plain MSE regression to KL-16/DINOv2 targets is clearly worse than denoising (Fig. 7); gains hold under every beta schedule tested (Table 11). — [ROSS HTML v2](https://arxiv.org/html/2410.09575v2)
- OLA-VLM/VisPer-LM: the Perceiver-resampler probes are effectively a predictor head; smooth-L1 + InfoNCE; embedding loss during IFT hurt (Table 6). — [VisPer-LM HTML v3](https://arxiv.org/html/2412.09585v3)
- VaLR: optimizer AdamW, no EMA mentioned (App. B.1). — [VaLR HTML](https://arxiv.org/html/2602.04476v1)
- Teacher-trainability data point (encoder-level): VEGA's unfrozen FiT3D teacher scores 0.04 vs 0.22 frozen on one hard task (Table 4, App. C.1). — [VEGA HTML](https://arxiv.org/html/2605.10485v1)
- Background on EMA targets: I-JEPA's targets come from an EMA teacher of the student (per a 2026 self-distillation paper's description); no MLLM paper found adopting this. — [Self-Distillation of Hidden Layers abs/pdf](https://arxiv.org/pdf/2603.15553) (unverified detail from search snippet)

### Inferences
- The field has not yet tested the BYOL/JEPA "EMA of the LLM" target; the closest are PRe (stop-gradient self-target, no momentum) and JARVIS (frozen external target). An EMA-target variant would be novel but the frozen-target results suggest the external teacher is doing real work (VaLR: without alignment the latent tokens *lose* to SFT; JARVIS: DINOv2 > CLIP/SigLIP teachers).
- Predictor heads are already standard (every method has a 2–3 layer MLP); the JEPA distinction that matters is whether the target is masked/unseen (JARVIS, Ross3D cross-view, Render to Reason) versus visible (VIRAL, PRe).

### Gaps
- No stability curves, loss-trajectory plots or collapse diagnostics (e.g. effective rank of aligned states) were found in any of the MLLM papers.
- JARVIS's appendix (hyper-parameters, B.1 ablations) was truncated; a masking-ratio sweep may exist there.

---

## KQ4. Where in the LLM (early vs middle layers) does alignment help most, and what is known about the loss-weight schedule?

### Takeaway
Across independent papers the best single layer sits in the first half: 1/4 depth for JARVIS (j=8/32), layer 10 for ANCHOR, 12/27 for VaLR, 16/32 for VIRAL and PRe, 24/32 for Spatial Forcing in VLAs; last-layer alignment is consistently worse (VIRAL, JARVIS, PRe, Spatial Forcing, GLaD's 24-vs-32 is the exception), and OLA-VLM's probing shows depth/segmentation are most decodable from layers 12–24; loss weights cluster at 0.4–0.5 with sharp degradation above ~2 and no paper uses a schedule.

### Cited Findings
- VIRAL (Table 2): layer 16 best on CV-Bench2D/MMVP (59.67/33.33); layer 4 best on MME (1720); layers 20–32 worse (54.87 CV-Bench2D at 28); multi-layer 14–18 collapses to 49.62/22.55 (Table 5); λ=0.5, no sweep. — [VIRAL HTML v2](https://arxiv.org/html/2509.07979v2)
- JARVIS (Table 1, VC-avg): j=4 50.4, 8 51.7, 16 50.3, 24 50.2, 31 49.2, 32 49.4 (baseline 50.9) — only the quarter-depth layer beats baseline clearly. — [JARVIS HTML](https://arxiv.org/html/2512.15885)
- ANCHOR (Table 9): L=6 32.8, 8 34.2, 10 35.6, 12 34.7, 14 33.5, 16 32.1 (xMMMU) — inverted U peaking at layer 10; α=0.4 fixed. — [ANCHOR HTML](https://arxiv.org/html/2608.15085)
- VaLR (Table 6): middle (12th) > last (27th) > front (4th: BLINK 59.2, CVBench 78.6); λ sweep 0–1, 0.5 best (Table 11). — [VaLR HTML](https://arxiv.org/html/2602.04476v1)
- PRe: middle layer (16 Vicuna / 14 Qwen) > last layer, explained by deep visual states drifting to high-frequency token space under the logit lens (Table 2, Fig. 9); λ=0.2 marginal, 0.5 default, 2.0 degrades (Fig. 11–12). — [PRe HTML](https://arxiv.org/html/2603.20808)
- OLA-VLM/VisPer-LM: probing of a language-only-trained LLaVA shows best depth/seg probe quality in layers 12–24 (Fig. 3); chosen sets depth {8,20}, seg {10,18}, gen {12,20} beat other sets (54.1 vs 51.7–52.4, Table 5); loss in PT only, IFT application hurt (Table 6). — [VisPer-LM HTML v3](https://arxiv.org/html/2412.09585v3)
- HeRA: head-level alignment at layers 3–5 (Qwen2.5-3B worst-5 heads: L3H4, L5H9/11/14/15) with λ=0.01; a VIRAL-style middle-layer cosine loss *regressed* (53.8 vs 54.2) in their setting; too many heads (worst-10) regress to 46.0 (Tables 1, 5). — [HeRA HTML](https://arxiv.org/html/2606.23885)
- Spatial Forcing (Table 2): layer 24 of 32 best (96.9); 1: 94.6, 8: 95.7, 16: 93.8, 32: 94.8; α=0.5 best with 0.02–0.5 flat and ≥2.5 harmful (App. A). — [Spatial Forcing HTML v2](https://arxiv.org/html/2510.12276v2)
- GLaD: final layer 94.1 > layer 24 92.6 (Table III), MSE loss. — [GLaD HTML](https://arxiv.org/html/2512.09619)
- ROCKET: 8–16 aligned layer pairs best (97.4 at 8), 24 pairs worse (94.1); λ=0.5 (Table 4, Sec. 4.3). — [ROCKET HTML](https://arxiv.org/html/2602.17951)
- VEGA (encoder-level): λ=0.1 best of {0.05, 0.1, 0.2} (Fig. 5). — [VEGA HTML](https://arxiv.org/html/2605.10485v1)
- JARVIS applies the loss stochastically (skipped with p=0.2) rather than scheduling the weight. — [JARVIS HTML](https://arxiv.org/html/2512.15885)
- Ross3D applies the cross-view loss every Δt=4 steps (Δt=1,2 worse on SQA3D, Table S3). — [Ross3D HTML](https://arxiv.org/html/2504.01901)
- Training-stage placement conflicts: OLA-VLM finds PT-only best; HeRA finds both stages needed (stage-2-only below baseline on Qwen2.5-3B, Table 7); VIRAL applies it only in the SFT/LoRA stage. — [VisPer-LM HTML v3](https://arxiv.org/html/2412.09585v3); [HeRA HTML](https://arxiv.org/html/2606.23885); [VIRAL HTML v2](https://arxiv.org/html/2509.07979v2)

### Inferences
- The optimum depends on the predictor's job: masked-prediction (JARVIS) peaks earliest (1/4), same-position alignment peaks around 1/3–1/2 (ANCHOR, VaLR, VIRAL, PRe), and VLAs with action heads tolerate deeper layers (SF 3/4). A reasonable prior for a 28–36 layer LLM is layer index ≈ 0.35–0.5 × depth for visible-token alignment.
- No paper uses a weight schedule; all use a constant λ. Sensitivity curves (SF, PRe, VaLR) are flat on [0.1, 0.5] and fall off above ~1–2, so a constant 0.5 with cosine loss is the de-facto default.

### Gaps
- Layer sweeps exist only for 7B-class decoders with 27–32 layers; nothing for larger models or Qwen3-style architectures.
- No paper reports an annealed or warm-up λ, so the effect of a schedule is unknown.

---

## KQ5. Negative or null results: where dense alignment did not help or hurt language/general ability.

### Takeaway
Null and negative results are frequent and under-advertised: VIRAL regresses by −1.7 vision-avg on Qwen3-8B in HeRA's reproduction and did not beat LLaVA in JARVIS's reproduction; a VIRAL-style middle-layer cosine loss regressed in HeRA's own ablation; VGGT latents as a VLM target hurt VSI-Bench (Render to Reason) and collapsed on hard tasks (VEGA); multi-layer alignment can be catastrophic (VIRAL 14–18, ROCKET independent projectors, HeRA worst-10); and small general/OCR/hallucination regressions appear in ROSS, PRe, JARVIS and HeRA.

### Cited Findings
- HeRA Table 4 (Qwen3-8B): VIRAL 54.2 vision-avg vs 55.9 LLaVA (−1.7), MMVP 46.7 vs 49.3, RWQA 57.6 vs 59.2; ROSS +0.3 only; HeRA Table 1: VIRAL-style feature cosine at middle layer 53.8 vs 54.2 baseline; worst-10 heads 46.0; contrastive on worst-5 *layers* 54.6 vs 57.4 baseline on Qwen3-4B. — [HeRA HTML](https://arxiv.org/html/2606.23885)
- JARVIS Table 2: VIRAL and the "w/o masking" alignment variant "do not improve over LLaVA on vision-centric tasks"; JARVIS itself loses Blink −3.4 and RWQA −2.2 on Qwen2-7B/CLIP and CVB3D −1.3 on SigLIP2. — [JARVIS HTML](https://arxiv.org/html/2512.15885)
- VIRAL's own paper: MMStar unchanged (33.93) for Vicuna-7B/CLIP; POPE drops 85.88→84.92 on Qwen2.5-7B; multi-layer 14–18 falls to 49.62/22.55; pre-projection residual injection falls to MMVP 19.33 (Tables 1, 5, App. B). — [VIRAL HTML v2](https://arxiv.org/html/2509.07979v2)
- Render to Reason: predicting VGGT latents at novel views −1.5 VSI-Bench; semantic prediction without VGGT −1.3; 32×32 grid worse than 8×8 (App. A.3, Table 3d, 3f). — [Render to Reason HTML](https://arxiv.org/html/2610.05417v1)
- VEGA: VGGT as teacher 0.04 hard-task success vs 0.34 with no teacher (Table 2). — [VEGA HTML](https://arxiv.org/html/2605.10485v1)
- GLaD: explicit depth-map supervision diverged; contrastive supervision no better than baseline (Sec. V-C); π³ teacher 86.1 vs 94.1 (Table III). — [GLaD HTML](https://arxiv.org/html/2512.09619)
- ROCKET: naive multi-layer alignment with independent projectors 96.4→80.0 (Table 8). — [ROCKET HTML](https://arxiv.org/html/2602.17951)
- ROSS: MMBench-CN −0.2, SEED-Img −0.3, AI2D −0.2, OCRBench −5 at 3B, HellaSwag slightly down for some configs; pixel-space regression gives no useful signal; MiDaS depth-map input expert sometimes harms (Tables 5, 12, 14; Fig. 5, 16). — [ROSS HTML v2](https://arxiv.org/html/2410.09575v2)
- Ross3D: semi-supervised variant beats the full-text baseline on ScanQA but not SQA3D (57.7 vs 58.6) or ScanRefer (57.9 vs 58.1); vanilla (non-3D) reconstruction +0.1 to +1.4 only (Tables 4, 6). — [Ross3D HTML](https://arxiv.org/html/2504.01901)
- OLA-VLM/VisPer-LM: embedding loss during IFT hurts (Table 6); learnable special tokens at IFT hurt (52.3 vs 55.1, Table 7); OK-VQA 56.9→56.6; RWQA flat (57.8→57.9); BLINK relative depth flat; Depth lower with VPT stage (Tables 1, 3, 4). — [VisPer-LM HTML v3](https://arxiv.org/html/2412.09585v3)
- PRe: 3B results mixed, OCRBench drop for frozen-CLIP; λ=2.0 degrades (Table 4, Fig. 12); argues external DINOv2 targets can introduce "conflicting objectives". — [PRe HTML](https://arxiv.org/html/2603.20808)
- VaLR: latent reasoning *without* alignment is below plain SFT on MMVP (56.7 vs 58.7), V* (75.9 vs 78.0) and CVBench (73.4 vs 77.0) (Table 3); inference 1.28× slower (Table 10). — [VaLR HTML](https://arxiv.org/html/2602.04476v1)
- HeRA: hallucination regressions (Qwen3-4B HallusionBench qAcc 22.2→18.9; Vicuna-13B CHAIRs 43.0→47.4) and Vicuna-13B RWQA 58.7→57.3 (Tables 2–3); SigLIP2 teacher "largely ineffective"; stage-restricted application below baseline (Table 7). — [HeRA HTML](https://arxiv.org/html/2606.23885)
- ANCHOR: 7B CVQA displayed subset 42.7 < base 43.4 (Table 2). — [ANCHOR HTML](https://arxiv.org/html/2608.15085)
- SSVAL abstract: aligning directly with external VFMs "does not fix" inference-time representation drift (unverified beyond abstract). — [SSVAL abs](https://arxiv.org/abs/2608.01635)
- ASVR: autoregressively reconstructing raw visual appearance "may even impair multimodal understanding". — [ASVR abs](https://arxiv.org/abs/2506.09040)
- "Why MLLMs Struggle to Determine Object Orientations" (arXiv 2604.13321): orientation is linearly decodable from CLIP/SigLIP encoder features of LLaVA-OV, Qwen2.5-VL, LLaVA-1.5/1.6, so encoder limitation is *not* the cause; the paper does not test alignment remedies. — [Orientation abs](https://arxiv.org/abs/2604.13321)
- Dedicated language-only evaluations (e.g. MMLU/GSM8K after alignment) were not reported by any of the MLLM papers read; the closest are ROSS's HellaSwag (slight drops) and the General/Knowledge/OCR averages in JARVIS/HeRA (roughly stable). — [ROSS HTML v2](https://arxiv.org/html/2410.09575v2); [JARVIS HTML](https://arxiv.org/html/2512.15885); [HeRA HTML](https://arxiv.org/html/2606.23885)

### Inferences
- Reproductions by other groups (HeRA, JARVIS) show VIRAL's gains are fragile to the LLM backbone and training stack; both reproductions used newer LLMs (Qwen2/Qwen3) and full-stage training rather than VIRAL's LoRA-only SFT.
- The failure modes cluster into: (i) too many layers/heads aligned, (ii) geometric latents with mismatched token structure (VGGT) as targets, (iii) applying the loss at the wrong stage, (iv) weights above ~1. General-language regressions, where measured, are small (≤0.5 pt) but OCR and hallucination metrics can move by several points.

### Gaps
- No paper reports text-only benchmarks (MMLU, GSM8K, IFEval) before/after alignment, so "hurts language ability" is unmeasured rather than refuted.
- VIRAL's OpenReview discussion could not be fetched (bot check), so reviewer-reported negatives are unknown.
- No controlled study compares the same alignment loss on Qwen2.5-VL visual tokens vs latent tokens, so VaLR's latent-token design choice is not isolated from its encoder choice.
