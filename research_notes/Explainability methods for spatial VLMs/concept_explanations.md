# Concept-based explanation methods for vision models and VLMs (emphasis on spatial concepts)

Project context used for the "how it applies" parts: VGGT geometry tokens (per-cell 3D point maps) fused into Qwen3.5-2B at decoder layers 0-2; additive arm vs 3D-aligned cross-attention arm (distance bias -nu_h * ||x_i - x_j||); 9,600 visual tokens per 32-frame video; VSI-Bench (relative direction, distance, route planning). Existing analyses: grad x input, SAM3 mask enrichment, object-to-object attention, J-lens vocabulary readout (geometry delta reads as noise tokens), attention-swap ablation.

Research scope note: about 16 search/fetch calls. Several 2026 preprints were only seen through abstracts and summaries, and are flagged where that matters.

## Q1. TCAV and successors (CAVs, regression CAVs, CRP, CRAFT, MCD, CBMs, post-hoc and label-free CBMs): what they measure, costs, weaknesses

### Takeaway
Supervised concept directions (CAV/TCAV) are cheap and map straight onto spatial labels. They become the "linear probe + directional derivative" recipe, which suits the project. Unsupervised concept-discovery methods (CRAFT, CRP) were built for CNN image classifiers and transfer awkwardly to a token-level decoder VLM. CBMs require retraining a bottleneck and answer a different question: they build an interpretable model rather than explaining the existing one.

### Cited Findings
- **CRAFT (Fel et al., CVPR 2023).** Discovers concepts automatically by applying non-negative matrix factorisation (NMF) to activations, recursively from top layers downward to find sub-concepts. It scores concept importance with Sobol indices and uses implicit differentiation to produce concept attribution maps. Demonstrated on ResNet50V2/ImageNet. Builds on ACE, which used segmentation plus clustering. Code: github.com/deel-ai/Craft. — [CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/html/Fel_CRAFT_Concept_Recursive_Activation_FacTorization_for_Explainability_CVPR_2023_paper.html); [arXiv 2211.10154](https://arxiv.org/pdf/2211.10154)
- **CRP (Achtibat, Dreyer et al., Nature Machine Intelligence 5:1006-1019, 2023).** Extends LRP so that a single explanation answers both "where" (heatmap) and "what" (which concept/channel). It is implemented as one backward pass with binary masking of relevance tensors. A companion method, Relevance Maximization (RelMax), selects reference examples for each concept. Applications include concept atlases and concept-composition analysis. The implementation is built on Zennit. — [Nature MI](https://www.nature.com/articles/s42256-023-00711-8.pdf); [arXiv 2206.03208](https://arxiv.org/abs/2206.03208v2)
- **Label-Free CBM (Oikarinen & Das, ICLR 2023).** Converts an arbitrary backbone into a concept-bottleneck model without labelled concept data: GPT-3 proposes the concept set and CLIP scores it. Accuracy stays close to the original network. The paper notes that the earlier post-hoc CBM still needed annotated concept data for TCAV, and that its CLIP route works only when the backbone is CLIP. — [paper text](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/ICLR/2023/labelfree_concept_bottleneck_models__81d641a6.txt); [Post-hoc CBM](https://deepai.org/publication/post-hoc-concept-bottleneck-models)
- **Weaknesses of CBMs.**
  - Label-Free CBM depends on GPT-3's parametric knowledge, so its concepts are not data-driven. — [arXiv 2502.11100](https://arxiv.org/pdf/2502.11100v2)
  - Leakage: label or other-concept information leaks into the concept units, and concept embedding models leak substantially under every hyperparameter setting tested. — [Leakage and Interpretability in Concept-Based Models](https://arxiv.org/html/2504.14094v1)
  - Post-hoc CBMs trained on auxiliary datasets suffer covariate shift that can invalidate the learned concepts in the target domain. — [On the Faithfulness of Post-Hoc CBMs, arXiv 2606.30498](https://arxiv.org/pdf/2606.30498)
  - SupCBM adds label supervision to reduce leakage. — [arXiv 2402.05945](https://ar5iv.labs.arxiv.org/html/2402.05945)
- **LEACE (Belrose et al., NeurIPS 2023).** Closed-form linear concept erasure that provably prevents every linear classifier from detecting the concept, with minimal change to the representation. "Concept scrubbing" applies it at every layer of an LLM. Code: github.com/EleutherAI/concept-erasure. LEACE is the modern successor to INLP for the causal side of CAV/probe work. — [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract.html); [arXiv 2306.03819](https://export.arxiv.org/abs/2306.03819)

### Inferences
- **What each method measures.**
  - TCAV measures how sensitive a class logit is to a direction separating concept examples from random examples (directional derivative of the logit along the CAV).
  - CRP measures how much relevance flows through a chosen channel or concept to the output, localised in input space.
  - CRAFT measures NMF components of activations and their Sobol importance.
  - CBMs force the prediction through named concepts.
  - For the project, the TCAV logic maps onto the answer-token logit (for example "front-left"): take the gradient of that logit along a learned "camera-relative x" or "object A left of B" direction.
- **Compute cost (rough estimate, not from a source).**
  - A CAV or probe needs one forward pass to cache activations plus a logistic or ridge fit. That is minutes on one GPU and is limited by caching the 9,600-token sequences, roughly at the cluster's reference timing of about 1.5 min per 4.7k texts on a B300, scaled up for longer sequences.
  - CRP needs LRP rules for Qwen's attention and RMSNorm. LRP for transformers exists (AttnLRP from the same group), but this was not verified here, so adapting CRP is engineering-heavy.
  - CRAFT's NMF needs non-negative activations. Residual streams are signed, so it would have to be applied after an MLP nonlinearity or to SAE codes.
- **Weaknesses of CAVs.** They depend on the choice of negative and random sets. They are correlational unless paired with erasure or steering. A spatial "concept" such as "left" is relational (A relative to B, in a given viewpoint frame), so a single per-token CAV is ill-posed. Pair-level features are better, for example the difference of two object-token activations, as in the "spatial ID" work under Q3.
- **CBMs are low priority.** A CBM would mean retraining Qwen with a spatial bottleneck. Interpretability is not the project's aim, so CBMs are low value here. A "post-hoc spatial CBM" head on frozen features adds little beyond a multi-output probe.

### Gaps
- Primary sources were not retrieved for the following, so details are omitted:
  - Regression Concept Vectors (Graziani et al.), the continuous-valued CAV best suited to metric depth and distance.
  - MCD (Multi-dimensional Concept Discovery, Vielhaben et al.; concepts as subspaces rather than single directions).
  - TCAV itself (Kim et al., 2018) and Network Dissection (Bau et al., 2017).
- No source applies CRP or CRAFT to a decoder-only VLM on video spatial QA.

## Q2. Sparse autoencoders, transcoders and crosscoders on vision and VLM representations; cost for a 2B model at one layer

### Takeaway
SAEs on VLM residual streams are established: LLaVA-Next, CLIP encoders, and Qwen3-VL/Gemma-3/LLaVA in 2026 work. Crosscoders exist specifically for model diffing and are the natural tool for additive vs cross-attention or base vs fused. However, 2025 evidence shows SAEs often underperform plain linear probes on downstream detection. Treat them as a discovery tool for non-verbal geometry features, not as the main test.

### Cited Findings
- **Residual-stream SAEs in VLMs.**
  - "Line of Sight" (arXiv 2506.04706) trains SAEs on the residual stream of LLaVA-Next's text decoder, following the Templeton et al. recipe and using ShareGPT4V image-caption data. It combines them with linear probes and steering vectors to test whether image concepts are linearly represented and causally editable. — [arXiv 2506.04706](https://arxiv.org/pdf/2506.04706)
  - Pach et al. (arXiv 2504.02821) train SAEs on the CLIP vision encoder and score neurons for monosemanticity. Intervening on a single SAE neuron in the encoder steers LLaVA's output while the LLM stays untouched. — [arXiv 2504.02821](https://arxiv.org/html/2504.02821v3)
  - VISTA (arXiv 2605.24946, May 2026) constrains the visual projector so that image tokens map into the space of a pretrained language SAE. Visual tokens in the residual stream can then be labelled with text-SAE latents. — [arXiv 2605.24946](https://arxiv.org/pdf/2605.24946)
  - Cascaded/hierarchical SAEs (Zhao et al., June 2026) are evaluated on Qwen3-VL, Gemma-3 and LLaVA. — [arXiv 2606.16193](https://arxiv.org/html/2606.16193v1)
  - Joint SAEs on LLaVA (ICLR 2026) find paired vision and language neurons. — [ICLR 2026](https://iclr.cc/virtual/2026/10019232)
- **Gemma Scope (DeepMind, 2024).** JumpReLU SAEs on every layer and sublayer of Gemma 2 2B and 9B (and selected 27B layers): more than 400 SAEs and more than 30M features. Each was trained on 4-16B tokens. The project used over 20% of GPT-3's training compute and stored about 20 PiB of activations. A secondary summary gives 4B tokens for 16.4K-latent SAEs and 16B for 1M-width SAEs. — [arXiv 2408.05147](https://arxiv.org/pdf/2408.05147v2.pdf); [summary](https://chatpaper.com/paper/48252)
- **EleutherAI sparsify.**
  - Uses TopK SAEs (the maintainers consider TopK a Pareto improvement over L1) and supports transcoders.
  - The Llama-3-8B all-layer example uses about 48 GB per GPU on an 8-GPU node.
  - The released Llama-3.1-8B SAEs were trained on about 8.5B tokens of RedPajama v2.
  — [PyPI eai-sparsify](https://pypi.org/project/eai-sparsify/); [HF sae-llama-3.1-8b-32x](https://huggingface.co/EleutherAI/sae-llama-3.1-8b-32x)
- **Overcomplete (Kempner Institute)** is a PyTorch dictionary-learning toolbox for concept extraction in large vision models. — [overcomplete](https://awesome.ecosyste.ms/projects/github.com%2Fkempnerinstitute%2Fovercomplete)
- **Archetypal SAE (Fel et al., ICML 2025)** constrains dictionary atoms to the convex hull of the data. The motivation is that standard SAEs trained on similar data produce very different dictionaries. A relaxed variant (RA-SAE) matches unconstrained reconstruction error. — [arXiv 2502.12892](https://arxiv.org/abs/2502.12892)
  - A later critique argues the stability gain may come from a shared deterministic k-means initialisation. — [arXiv 2510.08638](https://arxiv.org/pdf/2510.08638v1)
- **Crosscoders for model diffing.** Minder et al. (arXiv 2504.02922) compare Gemma-2-2B base and chat.
  - The L1 crosscoder has two artefacts: "complete shrinkage", where a shared concept loses its base-side decoder norm, and "latent decoupling".
  - Most "chat-only" latents identified by decoder norm in fact exist in both models at different intensities.
  - "Latent Scaling" measures how much each latent explains each model's activations.
  - The fix is a BatchTopK loss. A "diff-SAE" trained on chat-minus-base activations gave highly interpretable chat-specific latents.
  - Validation: patch latents into base activations, run the chat model's remaining layers, and measure KL divergence to true chat outputs.
  — [arXiv 2504.02922](https://arxiv.org/pdf/2504.02922v3); [Alignment Forum post](https://www.alignmentforum.org/posts/xmpauEXEerzYcJKNm/what-we-learned-trying-to-diff-base-and-chat-models-and-why)
  - Original crosscoder proposal (Anthropic, 2024). — [transformer-circuits.pub/2024/crosscoders](https://transformer-circuits.pub/2024/crosscoders/index.html)
- **Negative results.**
  - DeepMind's March 2025 post reports that SAEs lost to linear probes at out-of-distribution harmful-intent detection, and that the team deprioritised SAE research. — [summary of DeepMind post](https://www.longtermwiki.com/resources/244c1b93ef0a083c)
  - "Are Sparse Autoencoders Useful? A case study in sparse probing" (Kantamneni, Engels, Rajamanoharan, Tegmark, Nanda): SAE probes underperform logistic regression on average. — [arXiv 2502.16681](https://arxiv.org/pdf/2502.16681.pdf)

### Inferences
- **Cost of one SAE on Qwen3.5-2B at one layer (my estimate).** Assume a hidden size of about 2048 (check the config), 16x expansion (about 32k latents) and TopK.
  - Parameters: about 2 x 2048 x 32768, roughly 134M.
  - Activations: one 32-frame VSI-590K/ScanNet-style video yields about 9.6k visual tokens. 20-50k clips give 200-500M tokens, the order of magnitude used for small published SAEs, though below Gemma Scope's 4B.
  - Caching cost: storing 500M tokens x 2048 x bf16 takes about 2 TB, so stream activations or subsample tokens per video. Visual tokens within a frame are highly redundant.
  - Training time: the SAE itself trains in a few GPU-hours on one B200/B300. The dominant cost is the 2B forward passes, plus VGGT preprocessing if geometry is not cached.
  - No source gives GPU-hours for a 2B single-layer SAE; this is a gap.
- **Applying crosscoders to the project.**
  - Train a BatchTopK crosscoder, or better a diff-SAE, on paired activations of (base Qwen, fused model) or (additive arm, cross-attention arm) for the same input tokens at the same layer, for example layer 3 just after injection, or a mid layer.
  - Latents specific to the fused model are candidate "geometry features".
  - Validate each one by correlating its activation with VGGT xyz, depth, and the camera-relative direction to the queried object, rather than relying on automatic text naming. The J-lens shows these features are non-verbal, so LLM-based auto-interpretation of top-activating text will likely fail. Label latents against geometric ground truth instead.
  - Apply the Minder et al. pitfalls: use Latent Scaling, and do not trust decoder-norm ratios.
- **Probes before SAEs.** Given the negative SAE-vs-probe results, use supervised probes (Q3) for the hypothesis test. Use SAEs for exploratory discovery of what else the geometry injects, such as surface normals, room layout or floor plane.

### Gaps
- No public SAE suite for Qwen2.5-VL, Qwen3-VL or Qwen3.5 residual streams was confirmed in these searches.
- Transcoders and cross-layer transcoders (attribution graphs) on VLMs: no VLM-specific source was found here. The original transcoder paper (Dunefsky et al., 2024) was not retrieved.
- No SAE work was found that targets 3D or geometric features in VLMs specifically.

## Q3. Linear probes for spatial concepts and probing pitfalls

### Takeaway
2025-2026 work shows that spatial relations in VLMs are largely linear, and causally steerable, at object-word tokens in mid layers ("spatial IDs"). It also shows that decodability frequently diverges from use: depth/front-back is often decodable but "inverted" or unused. Every probe result in the project therefore needs a causal control: blank or ablated geometry, erasure, or steering.

### Cited Findings
- **Linear Mechanisms for Spatiotemporal Reasoning in VLMs (ICLR 2026).**
  - Mirror-swap activation patching shows that object-word tokens affect spatial beliefs only within a narrow band of intermediate layers.
  - Averaging activations over positions isolates a linear "spatial ID" for each object's location. It is roughly a fixed linear transform of position encodings (rank-3 fit, R^2 of about 0.85 or more).
  - Adding or subtracting IDs flips left/right and far/near judgements. Adversarial IDs reach a median 64.6% belief-flip rate, versus 29.5% for norm-matched noise.
  - In LLaVA-1.5-7B, manipulating height IDs also shifts front/behind judgements, so the model has no independent depth representation.
  - Adding a spatial-ID loss during Qwen2-2B fine-tuning raised COCO-Val spatial accuracy from 0.85 to 0.91 at 3.2k steps.
  - Video models show linear early/late temporal IDs.
  - Code: github.com/Raphoo/linear-mech-vlms.
  — [OpenReview](https://openreview.net/forum?id=2zXRGiorSu); [notes](https://en.papernotes.org/ICLR2026/interpretability/linear_mechanisms_for_spatiotemporal_reasoning_in_vision_language_models/)
- **Decodable Is Not Grounded (Liao et al., arXiv 2606.31257, June 2026).**
  - Across 14 VLMs (2B-27B, six LM families), the horizontal axis is "grounded", the vertical axis is answered from a "prior" (no dependence on vision), and depth is "inverted": decodable and causally controllable, but applied with the wrong sign, so accuracy is below chance.
  - Probes decode within-axis answers at 73-97%.
  - The proposed default control is to replace the image with a gray blank.
  - A training-free projection lifts a near-chance axis from 59% to 79%.
  - Only the abstract and summaries were read.
  — [arXiv 2606.31257](https://arxiv.org/pdf/2606.31257); [summary](https://www.emergentmind.com/papers/2606.31257)
- **Knowing Isn't Always Saying (arXiv 2608.22916, Aug 2026).**
  - Spatial-ID directions affect answer logits causally only at mid-to-deep layers.
  - Text chain-of-thought suppresses this transport in most models; visually grounded prompts do not.
  - Only the abstract was read.
  — [arXiv 2608.22916](https://arxiv.org/pdf/2608.22916); [HF page](https://huggingface.co/papers/2608.22916)
- **Related 2026 spatial-mechanism work.**
  - "Dual Mechanisms of Spatial Reasoning" (ICLR 2026) argues that the vision encoder is the main source of spatial information. — [arXiv 2603.22278](https://arxiv.org/abs/2603.22278)
  - "Seeing Isn't Knowing": models answer spatial questions even when visual evidence is incomplete. — [arXiv 2605.30557](https://arxiv.org/abs/2605.30557)
  - Other search snippets, not verified in full, report the following. Left/right appears as a clean, unentangled direction, while vertical and depth are entangled. One study reports roughly 77% probe decodability alongside roughly 31% (below-chance) front/back accuracy, with the inversion growing with scale. — [search results incl. arXiv 2605.30161](https://arxiv.org/pdf/2605.30161)
- **3D awareness of vision encoders.**
  - El Banani et al. (CVPR 2024) probe frozen features with a dense DPT-style probe for depth and surface normals (NYUv2) and test multiview correspondence (ScanNet). Models that are strong on general vision tasks differ widely in 3D awareness. Code: github.com/mbanani/probe3d. — [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Banani_Probing_the_3D_Awareness_of_Visual_Foundation_Models_CVPR_2024_paper.html)
  - A 2026 ViT study finds depth becomes linearly decodable by layers 5-8. — [arXiv 2604.23452](https://arxiv.org/pdf/2604.23452)
- **Causal tools.**
  - LEACE: closed-form linear erasure. — [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract.html)
  - Steering and adding spatial IDs, as in the ICLR 2026 paper above.
  - "Line of Sight" combines probes, SAEs and steering in LLaVA-Next. — [arXiv 2506.04706](https://arxiv.org/pdf/2506.04706)

### Inferences
- **Concrete probe suite for the project.** Cache residual streams at layers {3, 8, 14, 20, last} for the base model, the additive arm and the cross-attention arm, then fit:
  - (a) A per-visual-token ridge regression to camera-relative xyz, or depth, from the VGGT point map of that cell. This is the "is geometry present" probe. Because the geometry was injected from exactly this signal, expect near-perfect decoding at layers 0-3 in the fused arms. That makes it a trivial positive: it shows the information survives, not that the model uses it.
  - (b) A probe at the token level of the object mention, and at the final question token: object-pair relative direction in the VSI-Bench frame ("standing at A facing B, where is C"; 4-way classification), and metric distance between two objects. This is the non-trivial concept.
  - (c) An egocentric vs allocentric probe: regress C's position in the A-to-B frame vs in the camera frame, and compare R^2.
- **Controls.**
  - Hewitt-Liang control tasks: shuffled labels per scene, for selectivity.
  - The same probe on the base model: does fusion add decodability beyond what RGB-only Qwen already has? This is the key comparison.
  - Blank or zeroed geometry: the analogue of the gray-image arbiter. Zero the geometry tokens, or substitute a geometry map from another scene.
  - Scene-level train/test splits, to avoid memorising rooms.
- **Causal validation.**
  - LEACE-erase the direction probe's subspace at the layer after injection and measure the drop in VSI-Bench relative-direction accuracy.
  - Steer: add (spatial ID of C at position p') minus (spatial ID at p) and check whether the answer flips to the predicted quadrant.
  - Compare the two arms: is the answer more causally dependent on the probed subspace in the cross-attention arm?
- **This complements the J-lens.** Probes target exactly the non-verbal concepts that a vocabulary readout misses.
- **Depth caveat from the literature.** Base VLMs often lack an independent depth axis, or invert it. A clear test of whether VGGT injection fixes this is whether front/back is grounded (survives the blank-RGB-but-real-geometry control) in fused models but inverted or prior-driven in the base model.

### Gaps
- No paper found probes for camera-relative xyz per token inside a VLM after 3D-feature injection. This appears novel.
- Probing work specific to VSI-Bench (multi-frame, room-scale, allocentric) was not found. The 2026 papers mostly use COCO-Spatial or single images.
- Full methodology and numbers for arXiv 2606.31257 and 2608.22916 come from abstracts only.

## Q4. Concept discovery and automatic concept naming in VLMs; concept-based explanations of multimodal answers

### Takeaway
Automatic naming in VLMs is SAE-based. Each latent or neuron is matched to text concepts through CLIP or text-SAE similarity, or LLM-generated descriptions of top activations. This works for object and semantic concepts but is untested for geometric ones. For the project, "naming" should come from regression against geometric ground truth, not from text.

### Cited Findings
- Lanza et al. (2026) propose a human-interpretable concept for each SAE neuron, score it by cosine alignment with dataset samples on LLaVA-NeXT VQA, and report up to 45% better visual concept quality than prior SAE methods. — [Hamburg bib entry](https://wtmbib.informatik.uni-hamburg.de/Basilic/publis/1010/)
- VISTA labels visual-token patches by which text-SAE latents fire, and reports an LLM-verified matching rate following Venhoff et al. 2025. — [arXiv 2605.24946](https://arxiv.org/pdf/2605.24946)
- Label-Free CBM's GPT-3 + CLIP concept-set pipeline is the canonical "automatic concept naming" approach for vision backbones. — [paper text](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/ICLR/2023/labelfree_concept_bottleneck_models__81d641a6.txt)
- CRP/RelMax gives concept-level "where + what" explanations for individual predictions, with reference images per concept. — [arXiv 2206.03208](https://arxiv.org/abs/2206.03208v2)
- Monosemanticity-scored CLIP-encoder SAE neurons can steer LLaVA answers, giving concept-level explanations that are causal. — [arXiv 2504.02821](https://arxiv.org/html/2504.02821v3)

### Inferences
- **Text-based naming will not work for geometry latents.** CLIP- and text-based naming presupposes concepts that are verbal or object-like. That matches the project's J-lens finding: geometry reads as noise tokens.
- **Geometric naming instead.** For each SAE or crosscoder latent, compute its correlation and mutual information with a library of geometric variables:
  - VGGT depth, x, y, z
  - surface-normal proxies
  - distance to the queried object
  - the sign of the relative-direction quadrant
  - floor or wall membership (a SAM3 mask label)
  - frame index
  Name each latent after its best-correlated variable and report the fraction of latents left unexplained.
- **Concept explanation of an answer.**
  - Attribute the answer logit to SAE latents (gradient x latent activation, i.e. CRP-like relevance on the latent basis).
  - Then show which geometric concepts carry the attribution in the cross-attention arm versus the additive arm.

### Gaps
- No work was found that names 3D or geometric concepts in VLM latents automatically.

## Q5. Which methods test whether injected geometry adds a decodable 3D concept, and whether the answer causally uses it

### Takeaway
The most decisive and cheapest pipeline has three steps:
1. Supervised probes (per-token xyz; pair-level relative direction and distance), comparing base, additive and cross-attention arms with control tasks.
2. Causal tests on the probed subspace: LEACE erasure, steering with spatial IDs, and a geometry-blank or geometry-swap arbiter, measured on VSI-Bench accuracy.
3. Optionally, a diff-SAE or BatchTopK crosscoder to discover unanticipated geometry features.

### Cited Findings
- Spatial-ID steering demonstrably flips left/right and far/near answers and gives a model-agnostic causal protocol with released code. — [OpenReview ICLR 2026](https://openreview.net/forum?id=2zXRGiorSu)
- Blank-image arbitration exposes decodable-but-unused or inverted spatial axes that probing and steering both miss. — [arXiv 2606.31257](https://arxiv.org/pdf/2606.31257)
- The causal effect of spatial directions on answer logits arises only at mid-to-deep layers. — [arXiv 2608.22916](https://arxiv.org/pdf/2608.22916)
- LEACE gives provable linear erasure with minimal distortion. — [arXiv 2306.03819](https://export.arxiv.org/abs/2306.03819)
- Crosscoder and diff-SAE latents can be validated causally by patching them into the other model and measuring KL divergence. — [arXiv 2504.02922](https://arxiv.org/pdf/2504.02922v3)
- SAEs often lose to linear probes on supervised targets. — [arXiv 2502.16681](https://arxiv.org/pdf/2502.16681.pdf)

### Inferences
- **Decodability comparison (cheap: hours).** Probe the relative direction between two objects, and the distance, at object-mention tokens and the answer token, across layers, for base vs additive vs cross-attention. If decodability is higher in the fused arms at mid layers, the geometry has propagated as a relational concept and not only as per-token xyz.
- **Use (cheap: hours).**
  - LEACE-erase that subspace at layer L and measure the accuracy drop per VSI-Bench task. Compare against erasing a random subspace of the same rank.
  - Steer object-C spatial IDs and check for quadrant flips.
  - Arbiter: run with real RGB plus zeroed or shuffled geometry, and with real geometry plus blank RGB. Distance and direction accuracy should track the geometry if it is used.
- **Mechanism by arm.** Combine with the existing attention-swap ablation. If the cross-attention arm's distance bias -nu_h ||x_i - x_j|| produces a neighbourhood-structured representation, probes on attention outputs should show better metric-distance decodability than in the additive arm.
- **Discovery (moderate cost: GPU-days).** Train a BatchTopK diff-SAE on (fused minus base) at layer 3 and a mid layer. Name latents geometrically (Q4). Patch the top latents into base activations to test sufficiency.
- **What not to prioritise.** CBMs and label-free CBMs (retraining, wrong question). CRAFT/NMF on signed residuals. Text-based auto-naming.

### Gaps
- No published work applies these tests to a VLM with explicit 3D-geometry fusion such as VGGT; the project would be first to report them. This is based on the searches here, not an exhaustive review.
- Exact compute numbers for SAE training on a 2B VLM layer are not published; the estimate above is my own.
