# Mechanistic interpretability of vision-language models (VLMs): methods, papers, cost, and how they apply to the VGGT-fused Qwen3.5-2B project

Scope note. "BLMs" in the request is taken to mean VLMs (see the last section). Project shorthand used below: **ADD** = additive injection h_n += G_n at decoder layers 0–2; **XA** = 3D-aligned cross-attention with distance bias -nu_h ||x_i - x_j|| (16 heads) at layers 0–2; N_vis = 9,600 visual tokens (32 frames); answer letters emerge at layer ~17 in the existing Jacobian-lens readout.

Sourcing note. Items marked "(not re-fetched this session)" are standard references cited from the agent's background knowledge with their arXiv IDs; the content was not re-checked during this research pass, so the report writer should treat their detail as lower-confidence than the fetched items.

---

## 1. Logit lens / tuned lens / Jacobian lens and other vocabulary readouts on visual tokens

### Takeaway
Visual tokens inside the LM drift toward vocabulary-interpretable representations in late-middle layers (LLaVA-1.5 best layer ~26 of 33; Qwen2-VL-2B best layer ~25 of 29), but the plain logit lens badly understates how interpretable they are: on Qwen2-VL-2B only 6.5% of object patches decode to the right object word, while nearest-neighbour matching against *contextual* text embeddings (LatentLens) reaches about 70% interpretability across many VLMs. For this project a lens readout is cheap (one forward pass plus a projection) and should be paired with a contextual-embedding lens and a causal test, because Qwen-family 2B models give weak logit-lens signal at visual positions.

### Cited Findings
- Neo et al., "Towards Interpreting Visual Information Processing in Vision-Language Models" (ICLR 2025; Neo, Ong, Torr, Geva, Krueger, Barez) applies the logit lens to every layer's hidden state at every visual position and finds that visual-token representations are refined toward embeddings of interpretable vocabulary tokens although the LM was never trained for this — [arXiv 2410.07149](https://arxiv.org/html/2410.07149v2); [ICLR 2025 listing](https://mlanthology.org/iclr/2025/neo2025iclr-interpreting)
- Quantitatively (Neo et al.): LLaVA-1.5 (33 layers, 170 COCO images) — at the best layer an average of 23.7% of object-patch positions decode to the correct object class token, best layer on average 25.7. Qwen2-VL-2B (29 layers, 253 images) — only 6.5% decode correctly, best layer on average 25.1, with some non-zero correspondence already in early layers — [arXiv 2410.07149](https://arxiv.org/html/2410.07149v2)
- Qwen2-VL-2B was used in Neo et al. only for the logit-lens generalisation check; ablation and knockout used LLaVA-1.5 and LLaVA-Phi-3 — [arXiv 2410.07149](https://arxiv.org/html/2410.07149v2)
- Neo et al. report no hardware/runtime figures (compute acknowledged only to the Torr Vision Group) — [arXiv 2410.07149](https://arxiv.org/html/2410.07149v2)
- LatentLens (Krojer, Mosbach, Reddy, Elliott et al., ICLR 2026) compares each visual token's hidden state to a large database of *contextualised* text-token representations from intermediate LLM layers instead of to the (un)embedding matrix. v1: LatentLens 72% of visual tokens interpretable (VLM judge) vs EmbeddingLens 30% and LogitLens 23% over 10 VLMs; a later version reports 68% over 15 VLMs (EmbeddingLens 32%) — numbers changed between versions — [arXiv 2602.00462](https://arxiv.org/html/2602.00462); [ICLR 2026](https://iclr.cc/virtual/2026/10022271)
- LatentLens "Mid-Layer Leap": visual tokens at the input and early LM layers align best with contextual text representations from mid layers (e.g. layers 8–16), suggesting the projector targets semantic rather than lexical space; even linear projectors give interpretable visual tokens — [arXiv 2602.00462](https://arxiv.org/html/2602.00462)
- A related hallucination paper also edits vision-language representations via logit-lens readouts of image features ("Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations") — [arXiv 2410.02762](https://arxiv.org/pdf/2410.02762)
- Tuned lens (Belrose et al. 2023) trains one affine translator per layer to the final-layer basis, fixing logit-lens bias in early layers (not re-fetched this session) — [arXiv 2303.08112](https://arxiv.org/abs/2303.08112)

### Inferences
- The project's existing Jacobian lens already sits between logit lens and tuned lens (it linearises the rest of the network rather than assuming identity). The Neo et al. Qwen2-VL-2B number (6.5%) predicts that visual positions in Qwen3.5-2B will give sparse vocabulary hits under any unembedding-based lens; low hit rates at visual positions should not be read as "no information".
- Spatial content (depth, distance, "left of", room size) is unlikely to have clean single-token vocabulary names, so for VGGT geometry the most informative lens is probably (a) LatentLens-style nearest neighbours among contextual embeddings of spatial words ("near", "behind", "metres", numbers), and (b) linear probes for VGGT quantities (depth, 3D coordinates) per layer, compared ADD vs XA vs no-fusion baseline. Probes on 9,600 tokens × 29-ish layers are cheap once activations are cached (one forward pass per video; store a subsample of positions).
- A concrete experiment: track at which layer the 3D coordinate x_i becomes linearly decodable from visual positions, and whether XA makes it decodable earlier/stronger than ADD. Because injection is at layers 0–2, decodability should be high there and the question is whether it survives to layer ~17 where letters emerge.
- Tuned-lens translators would need training on the model's own activations (a few thousand sequences); with hybrid linear-attention layers this is unchanged since the lens only touches the residual stream.

### Gaps
- No paper found that applies logit/tuned/latent lenses to *geometry* or 3D tokens injected into a VLM.
- Whether LatentLens has released its package/database for Qwen3.5 was not verified.
- No published lens results on Qwen3.5 or any hybrid (linear-attention) VLM were found.

---

## 2. Attention knockout and information flow from image to question/answer positions

### Takeaway
Three independent lines (Zhang et al. CVPR 2025; Kaduri et al. CVPR 2025; Neo et al. ICLR 2025) converge: image information flows into question tokens in early-to-middle layers, then from question tokens to the last position in upper-middle layers; object-specific information reaches the last token mainly in middle-to-late layers; and a narrow middle band (~25% of layers) carries most cross-modal transfer. Video-LLM follow-ups (Map the Flow, ICLR 2026) find the same staged pattern for temporal questions and show that many attention edges can be cut without loss. Attention knockout is cheap (one forward per window × condition) and is the most direct way to localise *where* VGGT-derived information is consumed.

### Cited Findings
- Zhang et al., "Cross-modal Information Flow in Multimodal Large Language Models" (CVPR 2025) adapts Geva et al.'s attention knockout to block edges between visual and text tokens in VQA and measures change in answer probability — [arXiv 2411.18620](https://arxiv.org/pdf/2411.18620); [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Cross-modal_Information_Flow_in_Multimodal_Large_Language_Models_CVPR_2025_paper.pdf)
- Zhang et al. stages: (1) in lower layers general whole-image information flows into question tokens; (2) in middle layers information from the question-relevant image patches flows into the question tokens; (3) in upper-middle layers the multimodal question representation flows to the last position where the answer is produced — [CVPR 2025 slides](https://cvpr2023.thecvf.com/media/cvpr-2025/Slides/33685.pdf); [arXiv 2411.18620](https://arxiv.org/pdf/2411.18620). (Exact layer indices and effect sizes were not extracted: the PDF could not be parsed in this session.)
- Kaduri, Bagon, Dekel, "What's in the Image? A Deep-Dive into the Vision of VLMs" (CVPR 2025): query tokens store global image information — the model produces descriptive answers from those tokens alone without access to image tokens; cross-modal flow is dominated by the middle ~25% of layers, with early and late layers contributing marginally; object details are retrieved locally (attention peaks land on the object) — [arXiv 2411.17491](https://arxiv.org/html/2411.17491v1); [project page](https://vision-of-vlm.github.io)
- Kaduri et al. "image re-prompting": on MME, re-prompting with 5% of image tokens achieved 96% of full performance with ~20× fewer tokens — [arXiv 2411.17491](https://arxiv.org/html/2411.17491v1)
- Neo et al. knockout (LLaVA-1.5, 100 VQA images, windows L1–10, L5–14, L11–20, L15–24, L21–31; relative performance, 1.00 = no effect): blocking object tokens → last token hurts most in mid-late (L15–24: 0.88) and late (L21–31: 0.93); all layers 0.82. Object+1-buffer → last: mid-late 0.82. *Non-object* visual tokens → last token matter most in early layers (L1–10: 0.88). Blocking visual → last visual row has no effect (1.00) — [arXiv 2410.07149](https://arxiv.org/html/2410.07149v2)
- Neo et al. token ablation: object-token mean ablation strongly degrades object identification (abstract: drop of over 70%) while random-token ablation of the same budget barely matters; however, the table caption's "lower = greater impact" wording conflicts with the abstract's direction, so treat specific percentages cautiously — [arXiv 2410.07149](https://arxiv.org/html/2410.07149v2)
- Basu et al., "Understanding Information Storage and Transfer in Multi-modal LLMs" (NeurIPS 2024): a multimodal causal-tracing method on LLaVA and multimodal Phi-2 finds factual information is retrieved from *earlier* MLP and self-attention layers than in text-only LLMs, and that a small, consistent subset of visual tokens transfers image information into those blocks; introduces VQA-Constraints (9.7K) and the MULTEDIT editing method — [arXiv 2406.04236](https://arxiv.org/pdf/2406.04236); [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0dfe31d6e703e138d46a7d2fced38b7c-Abstract.html)
- "Map the Flow: Revealing Hidden Pathways of Information in VideoLLMs" (ICLR 2026): temporal reasoning starts with cross-frame interactions among video tokens in early-to-middle layers, then video-language integration onto temporal keywords in the question, then convergence on the last position; VideoQA performance can be kept while suppressing many attention edges (e.g. 58% in LLaVA-NeXT-7B-Video-FT) — [arXiv 2510.13251](https://arxiv.org/html/2510.13251v1)
- "An Empirical Study on How Video-LLMs Answer Video Questions" defines Video Temporal, Video Spatial and Language-to-Video knockouts and finds spatio-temporal modelling relies more on language-guided retrieval than on intra/inter-frame self-attention among video tokens — [arXiv 2508.15360](https://www.emergentmind.com/papers/2508.15360)
- CircuitProbe blocks attention from object-region tokens (or all non-object tokens) to the first answer token within a layer window to test where object evidence is consumed in video LMs — [arXiv 2507.19420](https://arxiv.org/pdf/2507.19420)
- Chen et al., "Why Is Spatial Reasoning Hard for VLMs? An Attention Mechanism Perspective on Focus Areas" (ICML 2025): image tokens are ~90% of the input but receive ~10% of attention; spatial-reasoning success correlates with attention aligning to the actual object locations; errors come from attention on irrelevant objects; the training-free AdaptVis (confidence-dependent attention temperature) gives up to 50 absolute points on WhatsUp/VSR — [arXiv 2503.01773](https://arxiv.org/html/2503.01773v2); [PMLR](https://proceedings.mlr.press/v267/chen25cr.html); [code](https://github.com/shiqichen17/AdaptVis)
- Qi et al., "Beyond Semantics: Rediscovering Spatial Awareness in VLMs" reportedly finds an embedding-norm skew between vision and text tokens that suppresses RoPE positional information, and that shuffling vision tokens leaves performance largely unchanged (only a third-party summary seen; unverified) — [alphaXiv 2503.17349](https://alphaxiv.org/overview/2503.17349v2)

### Inferences
- Direct project analogue: run windowed knockouts (window ~5 layers, sliding over the whole depth) on four edge sets: (a) visual → answer/last position, (b) visual → question tokens, (c) question → last, (d) object-mask visual tokens (from SAM3, already available) → last vs non-object visual → last. Compare ADD vs XA. Prediction from the literature: (b) matters early/mid, (c) and (a-object) matter mid/upper-mid, i.e. just below layer ~17 where letters emerge in the Jacobian lens.
- The "shuffling vision tokens barely matters" result (Qi et al., unverified) and the "~10% attention on images" result make a crucial control for this project: shuffle or spatially permute the VGGT geometry tokens (or the x_i coordinates fed into the XA distance bias) and test VSI-Bench accuracy. If shuffling coordinates does not hurt XA, the gain is not geometric.
- Knockout is only directly defined for softmax-attention layers. Qwen3.5's linear-attention (gated delta-net-style) layers mix tokens through a recurrent state, so per-edge masking is not available there; knockout must either be restricted to the full-attention layers, or linear-attention layers must be handled by blocking at the state level (e.g. zeroing the contribution of visual positions to the recurrent state, which removes *all* later reads of those tokens in that layer). This is a project-specific complication not addressed in any paper found.
- Compute: each knockout condition is one forward pass with a custom mask; at 9,600+ tokens the dense mask is 10k×10k per head, so it must be applied via a custom attention function (FlexAttention / eager on chosen layers) rather than SDPA with a full mask, or by hooking K/V for visual positions to be excluded from selected query rows. With ~30 windows × 4 edge sets × a few hundred VSI-Bench questions, this is roughly the cost of ~100 eval passes, i.e. hours on one B300 given the project's eval speed.

### Gaps
- Exact layer ranges and effect sizes of Zhang et al. could not be extracted (PDF unparsable here).
- No information-flow study on Qwen3.5 / any hybrid linear-attention VLM, or on VSI-Bench-style 3D spatial questions, was found.
- The Findings of EMNLP 2026 paper on video multiple-choice knockout ([ACL Anthology 2026.findings-emnlp.1887](https://aclanthology.org/2026.findings-emnlp.1887.pdf)) could not be fetched (file too large); title and findings unknown.

---

## 3. Activation patching / causal tracing, counterfactual image pairs, path patching, attribution patching, EAP; feasibility at ~10k tokens

### Takeaway
Activation patching in VLMs is established (BLIP causal tracing 2023; NOTICE 2024 with semantic counterfactual images; Basu et al. 2024 multimodal causal tracing), and semantic minimal-pair corruptions are preferred over Gaussian noise. Attribution patching (a gradient, first-order approximation) has been used on VLMs with integrated gradients (AP-IG, "Same Task, Different Circuits", 2025). At 9,600 visual tokens, exact patching per token is infeasible but patching *per group* (per frame, per object mask, per layer/head, per fusion head) is cheap, and attribution patching gives all-position estimates in one forward + one backward pass.

### Cited Findings
- Palit, Pandey, Arora, Liang, "Towards Vision-Language Mechanistic Interpretability: A Causal Tracing Tool for BLIP" (2023): open-source causal tracing on BLIP, corrupting image-patch embeddings with noise — [arXiv 2308.14179](https://arxiv.org/pdf/2308.14179)
- Golovanevsky, Rudman, Palit, Singh, Eickhoff, "What Do VLMs NOTICE?" (NAACL 2025): replaces Gaussian-noise corruption with semantic image changes (Semantic Image Pairs) and semantic text corruption; activation patching on BLIP and LLaVA identifies "universal" cross-attention heads in BLIP and self-attention heads in LLaVA — [arXiv 2406.16320](https://arxiv.org/html/2406.16320v3)
- V-SEAM builds a semantic-level visual editing + attention-modulation framework for causal interpretability of VLMs on top of Golovanevsky et al.'s activation patching — [arXiv 2509.14837](https://arxiv.org/pdf/2509.14837)
- Basu et al. adapt causal tracing to multimodal inputs (corrupting the constraint in the question) and locate early-layer MLP/attention storage plus a small set of transfer-carrying visual tokens — [arXiv 2406.04236](https://arxiv.org/pdf/2406.04236)
- "Same Task, Different Circuits: Disentangling Modality-Specific Mechanisms in VLMs" uses attribution patching with integrated gradients (AP-IG) to find circuits for analogous vision and text tasks; on average only 18% of components are shared between modalities; a training-free fix patches visual-token embeddings from later layers back into earlier layers — [arXiv 2506.09047](https://arxiv.org/html/2506.09047v3)
- RelP replaces local gradients in attribution patching with LRP propagation coefficients for more faithful, still-cheap circuit discovery (tested on GPT-2/Pythia, not VLMs) — [arXiv 2508.21258](https://arxiv.org/html/2508.21258v1)
- ICLR 2025 blogpost "Mechanistic Interpretability Meets Vision Language Models: Insights and Limitations" surveys these methods for VLMs — [ICLR blogposts](https://iclr-blogposts.github.io/2025/blog/vlm-understanding)
- Background method references (not re-fetched this session): attribution patching (Nanda 2023, blog "Attribution Patching: Activation Patching At Industrial Scale" — [neelnanda.io](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching)); AtP* (Kramár et al. 2024, fixes attention-saturation and cancellation failure modes of AtP — [arXiv 2403.00745](https://arxiv.org/abs/2403.00745)); Edge Attribution Patching (Syed, Rager, Conmy 2023 — [arXiv 2310.10348](https://arxiv.org/abs/2310.10348)); EAP-IG (Hanna, Pezzelle, Belinkov 2024 — [arXiv 2403.17806](https://arxiv.org/abs/2403.17806)); path patching (Wang et al. 2022 IOI / Goldowsky-Dill et al. 2023 — [arXiv 2304.05969](https://arxiv.org/abs/2304.05969)).

### Inferences
- **Counterfactual pairs for VSI-Bench.** Natural minimal pairs for a 3D-fusion model are *geometry-only* counterfactuals: keep RGB frames fixed and corrupt only the VGGT inputs (shuffle geometry tokens across cells, rescale depth, swap the 3D coordinates of two objects, or feed geometry from another scene). Patch clean → corrupted (or vice versa) at G_n (ADD) or at the XA output / distance bias (XA), per fusion layer 0/1/2, per XA head (16), and per frame. This isolates "how much of the answer flows through geometry" more cleanly than the existing attention-swap ablation, which changes the attention pattern but not the geometry content.
- **Head-level patching in XA.** 3 layers × 16 heads = 48 components; exact patching of each is 48 forwards per pair — trivial. Report which heads' nu_h (distance sensitivity) predict their causal effect.
- **Token-level effect at 9,600 tokens.** Exact per-token patching = 9,600 forwards per pair per layer: infeasible. Attribution patching gives the first-order estimate (clean − corrupt activation) · grad for every token and layer from one forward on each input plus one backward: about 3× one forward pass, i.e. comparable to the project's existing grad×input saliency (which is attribution patching against a zero baseline). Validate top-k AtP tokens with a few exact patches (AtP* recommends this because AtP has false negatives on saturated attention).
- **Memory.** Backward through ~10k tokens on a 2B model in bf16 fits on 80–288 GB GPUs if only needed activations are kept (gradient checkpointing, or detach everything upstream of the patched site). The existing grad×input runs show this is already feasible on the cluster.
- **EAP** over all edges is expensive in memory for 10k tokens (edge scores require storing per-component input gradients); restrict nodes to heads/MLPs aggregated over positions and treat "visual tokens of frame f" as grouped nodes.
- **Path patching** is the right tool to answer "does geometry reach the answer *via* the question tokens or directly via visual tokens": patch G only on paths that go through question-token positions in layers 3–17.

### Gaps
- No VLM paper found that runs EAP/EAP-IG specifically, or any patching at ~10k-token video scale; the feasibility estimates above are the agent's inference.
- No published counterfactual-pair benchmark for 3D/spatial VQA was found.

---

## 4. Circuit discovery, head attribution, and token pruning/redundancy evidence

### Takeaway
Circuit-level VLM work is early but growing: head-level patching (NOTICE), AP-IG circuits showing modality-specific components (Same Task, Different Circuits), and transcoder attribution graphs on Gemma-3-4B (Circuit Tracing in VLMs, CVPR 2026). Token-pruning results (FastV: 50% of image tokens removable after layer 2 in LLaVA-1.5-13B; image re-prompting at 5% of tokens; EAZY hallucination tokens ~1.5%) are strong behavioural evidence that most visual tokens are redundant for the answer after early layers — which is directly relevant to whether 9,600 geometry-fused tokens are used.

### Cited Findings
- FastV (Chen et al., ECCV 2024): in deep layers of LLaVA-1.5, image tokens receive very little attention (average ~0.21% of the score attributed to system prompts); pruning 50% of image tokens after layer 2 in LLaVA-1.5-13B does not hurt average performance; 45% FLOPs reduction on LLaVA-1.5-13B without performance loss across image and video tasks — [arXiv 2403.06764](https://arxiv.org/html/2403.06764v3); [code](https://github.com/pkunlp-icler/FastV)
- Kaduri et al.: image re-prompting with 5% of image tokens keeps 96% of MME performance — [arXiv 2411.17491](https://arxiv.org/html/2411.17491v1)
- Map the Flow: performance kept while suppressing many attention edges (58% example in LLaVA-NeXT-7B-Video-FT) — [arXiv 2510.13251](https://arxiv.org/html/2510.13251v1)
- EAZY (ICCV 2025): a small set of high-attention image tokens drives object hallucination; removing ~1.5% of image tokens mitigates it — [papers.cool 2503.07772](https://papers.cool/arxiv/2503.07772)
- "Image Tokens Matter" (NeurIPS 2025, Wang, Ding, Gu et al.): in discrete-tokenizer LVLMs, co-occurring image tokens become linked to object words and induce hallucination; suppressing visually absent tokens via latent editing reduces it — [arXiv 2505.21547](https://arxiv.org/pdf/2505.21547)
- NOTICE identifies universal cross-attention heads (BLIP) and self-attention heads (LLaVA) that carry image information — [arXiv 2406.16320](https://arxiv.org/html/2406.16320v3)
- Same Task, Different Circuits: ~18% component overlap between visual and textual versions of the same task — [arXiv 2506.09047](https://arxiv.org/html/2506.09047v3)
- "Circuit Tracing in Vision–Language Models" (CVPR 2026): transcoders decompose Gemma-3-4B representations into features; attribution-based circuit discovery; circuits validated by feature steering and circuit patching — [arXiv 2602.20330](https://arxiv.org/html/2602.20330)
- Neo et al.: object-token ablation is far more damaging than random-token ablation of equal size — [arXiv 2410.07149](https://arxiv.org/html/2410.07149v2)

### Inferences
- **Pruning as a diagnostic.** Apply FastV-style pruning (keep top-k visual tokens by attention from the last/question positions after layer L) to both arms. If XA genuinely makes geometry useful, the accuracy-vs-k curve should degrade differently from ADD, and the retained tokens should be enriched in SAM3 object masks (the project already measures mask enrichment, so this reuses infrastructure). This is cheap: one forward per k.
- **Head attribution downstream of fusion.** Compute per-head AtP scores on the clean-vs-geometry-corrupted pairs for all full-attention heads in layers 3–17 to find "geometry reader" heads; compare head sets between ADD and XA (do they recruit the same heads?). This mirrors the "Same Task, Different Circuits" component-overlap metric.
- Transcoder/SAE circuit tracing (as in the Gemma-3-4B CVPR 2026 paper) would require training transcoders on Qwen3.5-2B, which is a substantial project in itself; likely out of scope for an MPhil timeline unless pretrained SAEs/transcoders for Qwen3.5 exist (none found).

### Gaps
- No pretrained SAEs/transcoders for Qwen3.5 (or Qwen3-VL) were found in this pass.
- No pruning study specific to VSI-Bench or 3D-augmented tokens was found.

---

## 5. Tools for HF Qwen-VL models

### Takeaway
The most directly usable tool is Spectra (ACL 2026 demo, Neo et al., MIT license), a HF-Transformers-based mech-interp library for VLMs supporting Qwen2.5-VL, Qwen3-VL, LLaVA-1.5 and SmolVLM. Otherwise nnsight or plain PyTorch hooks on the HF model are the practical route; TransformerLens does not support Qwen-VL vision/hybrid architectures out of the box, and LVLM-Interpret is a Gradio attention/relevancy viewer demonstrated on LLaVA. For Qwen3.5 (hybrid linear attention + custom fusion modules) custom hooks will be needed whatever library is chosen.

### Cited Findings
- Spectra: "A Mechanistic Interpretability Library for Vision-Language Models" (Neo, Zheng, Lam, Ong; ACL 2026 System Demonstrations, pp. 794–803): built on HF Transformers with per-checkpoint configs; supports Qwen2.5-VL, Qwen3-VL, LLaVA-1.5, SmolVLM; demonstrated on a counting task; MIT; code at github.com/clemneo/vlm-spectra — [ACL Anthology](https://preview.aclanthology.org/ingest-acl/2026.acl-demo.78/); [PDF](https://aclanthology.org/2026.acl-demo.78.pdf)
- Spectra's authors note NNsight is in principle architecture-agnostic but users report significant difficulty applying its abstractions to multimodal architectures, and that existing VLM tools are limited to basic activation probing and saving — [ACL Anthology](https://preview.aclanthology.org/ingest-acl/2026.acl-demo.78/)
- LVLM-Interpret (Intel Labs, 2024): Gradio tool that stores attention weights during generation and builds relevancy maps and causal graphs; applicable to any LVLM with a transformer LLM front-end; case study on LLaVA — [arXiv 2404.03118](https://arxiv.org/html/2404.03118v3)
- Prisma: open-source toolkit for mechanistic interpretability in vision and video (ViT-side, not LM-side) — [arXiv 2504.19475](https://arxiv.org/abs/2504.19475)
- VLM-Lens: probing-oriented toolkit; does not directly support attention interpretation or circuit discovery; has been applied to Qwen2-VL among others — [arXiv 2510.02292](https://arxiv.org/pdf/2510.02292)
- Background (not re-fetched this session): nnsight / NDIF (Fiotto-Kaufman et al. 2024) — [arXiv 2407.14561](https://arxiv.org/abs/2407.14561); pyvene (Wu et al. 2024, intervention library supporting HF models) — [arXiv 2403.07809](https://arxiv.org/abs/2403.07809); TransformerLens — [GitHub](https://github.com/TransformerLensOrg/TransformerLens).

### Inferences
- Since the project already has custom fusion modules and its own saliency / Jacobian-lens / attention-swap code, the lowest-friction path is to keep using PyTorch forward hooks (or nnsight wrapping the same HF module tree) and borrow Spectra's patterns (e.g. its Qwen3-VL config) for module naming. Spectra's Qwen3-VL support is the closest architecture match, but Qwen3.5's linear-attention layers and the VGGT fusion layers would need to be added.
- TransformerLens would require re-implementing the model, which is not worthwhile for a hybrid architecture.
- For knockout at 10k tokens, use a custom attention mask path (FlexAttention or an eager implementation on selected layers) rather than any library's default, since full dense masks per head are memory-heavy.

### Gaps
- Whether Spectra or nnsight already handle Qwen3.5 hybrid layers was not checked (repository not inspected).
- Whether pyvene/TransformerLens added Qwen3-VL/Qwen3.5 support by late 2026 was not verified.

---

## 6. The term "BLM"

### Takeaway
No literature usage of "BLM" as "bimodal language model" was found; "BLMs" in the request almost certainly means VLMs. The only notable hit is BLM₁, a "Boundless Large Model" for embodied cross-space/cross-task control, which is unrelated to interpretability.

### Cited Findings
- BLM₁: "A Boundless Large Model for Cross-Space, Cross-Task, and Cross-Embodiment Learning" (arXiv, Oct 2025), a multimodal embodied model compared against MLLMs and vision-language-action models — [arXiv 2510.24161](https://arxiv.org/pdf/2510.24161)
- A search for "bimodal language model" returned no source defining BLM this way — [search results incl. VLM surveys, e.g. arXiv 2404.01322](https://arxiv.org/pdf/2404.01322)

### Inferences
- Treat "BLM" as a typo for VLM (B and V are adjacent on QWERTY). "Bayesian language model" did not surface as a relevant acronym in this pass.

### Gaps
- "Bayesian language model" as an acronym was not separately searched; only the "bimodal" reading was checked directly.

---

## Cross-cutting summary for the project (agent inference, ranked by cost-benefit on one GPU)

1. **Geometry-only counterfactual activation patching** at fusion sites (ADD G_n vs XA output, per layer 0–2, per XA head, per frame): tens of forwards per pair; cleanest causal answer to "does the answer use geometry, and through which heads".
2. **Windowed attention knockout** (visual→question, question→last, object-visual→last, non-object-visual→last), full-attention layers only, ADD vs XA; expected to show transfer just below layer ~17.
3. **Attribution patching (AtP/AtP*)** over all 9,600 positions and all heads with geometry-corrupted baselines; cost ≈ existing grad×input; validate top hits with exact patching.
4. **Coordinate-shuffle / geometry-swap controls** (motivated by Qi et al. and Chen et al.): if XA's gain survives shuffled x_i, it is not 3D-geometric.
5. **Probes + contextual-embedding lens** for depth/3D coordinates per layer, complementing the existing Jacobian lens (expect weak logit-lens signal at visual positions in a Qwen 2B model, per Neo et al.'s 6.5%).
6. **FastV-style pruning curves** with SAM3 enrichment of retained tokens.
7. Transcoder/SAE circuit tracing: high cost, likely out of scope.
