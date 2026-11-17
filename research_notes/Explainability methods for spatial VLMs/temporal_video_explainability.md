# Temporal and video explainability methods for video LLMs / video VLMs

Scope: methods that explain how a multi-frame video model uses evidence across frames and time, and how they would apply to the MVP project (VGGT geometry fused into Qwen3.5-2B at decoder layers 0-2; additive vs 3D-aligned cross-attention arms; VSI-Bench; 32 frames x 300 cells = 9,600 visual tokens).

**Architecture fact that affects almost every method below.** The local Qwen3.5-2B config (`/shared/workspace/povejmo/huggingface_cache/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c.../config.json`) has 24 decoder layers. `layer_types` is 3x `linear_attention` then 1x `full_attention`, repeated (`full_attention_interval: 4`). Only layers 3, 7, 11, 15, 19 and 23 are softmax attention, with 8 query heads and 2 KV heads (GQA), and hidden size 2048. **Decoder layers 0-2, where the geometry is injected, are all linear-attention layers.** All the published video-LLM knockout and tracing work below uses models where every layer is softmax attention (Qwen2.5-VL, LLaVA-NeXT/OneVision/Video, VideoLLaMA3, InternVideo2.5, LongVA). An edge-level attention mask (token i cannot see token j) is well defined only in the 6 full-attention layers. In linear/recurrent layers, "blocking frame f" has to be done another way: zeroing or resetting the recurrent state at frame boundaries, or zeroing a token's key/value or input contribution. That is an approximation. (Source: local config file read on 2026-11-06. No published paper was found that does attention knockout on hybrid linear-attention VLMs.)

---

## Q1. Which methods attribute a video model's answer to frames or time steps?

### Takeaway
Perturbation-based methods still give the most trustworthy frame-level attribution: frame deletion/occlusion, sufficient-subset search, and sampled frame-level Shapley. Gradient methods (IG, grad x input) are cheap but spread credit across near-duplicate frames. For an MCQ video LLM the natural target is the log-probability of the chosen option letter. Frame deletion over 32 frames costs about 32-64 forward passes per question, which is cheap for a 2B model.

### Cited Findings
- **Play Fair / Element Shapley Value (ESV), ACCV 2020 (foundational).** Splits a video model's class score across frames as a Shapley value. A tractable approximation scales linearly with the number of frames. Tested on TRN/TSN on Something-Something and compared against Integrated Gradients and Grad-CAM. — [Price & Damen, Play Fair (arXiv 2011.12372)](https://arxiv.org/pdf/2011.12372); [ACCV open access](https://openaccess.thecvf.com/content/ACCV2020/papers/Price_Play_Fair_Frame_Contributions_in_Video_Models_ACCV_2020_paper.pdf)
- **Sufficient Frame Subsets (SFS), 2026.** Adapts Carter et al.'s "sufficient input subsets" to frames. A subset S is sufficient if the relative drop in the predicted-class probability, (Phi(V)_c - Phi(V_S)_c)/Phi(V)_c, is at most tau (tau = 0.1). Found by instance-wise backward selection and repeated to obtain disjoint subsets. Masked frames are filled by temporal forward-fill (nearest kept frame). — [Sufficient Frame Subsets for Video Understanding (arXiv 2610.25084)](https://arxiv.org/html/2610.25084)
  - Average SFS size at tau = 0.1: 1.1 frames (mc3-18), 3.4 (r3d-18), 3.4 (V-JEPA 2), 2.8 (TimeSformer-base), out of 16 (or 8) frames. Accuracy on the subset matches the full video, e.g. V-JEPA 2 0.68 vs 0.67. — [arXiv 2610.25084](https://arxiv.org/html/2610.25084)
  - Insertion AUC, SFS vs best baseline: V-JEPA 2 0.783 vs 0.591 (IG); TimeSformer 0.657 vs 0.477 (IG). Baselines included Grad-CAM, Integrated Gradients, Shapley sampling, STAA attention attribution and facility-location / DINOv2 diversity. — [arXiv 2610.25084](https://arxiv.org/html/2610.25084)
  - The paper argues that standard per-frame attribution "spreads evidence diffusely across near-duplicate frames". — [arXiv 2610.25084](https://arxiv.org/html/2610.25084)
  - Cost: worst case O(p^2 k) model evaluations, reducible to O(T k) with batching. The paper reports 135 evaluations for a 16-frame video. — [arXiv 2610.25084](https://arxiv.org/html/2610.25084)
  - Caveat: only CNN and video-transformer classifiers are tested, not video LLMs. — [arXiv 2610.25084](https://arxiv.org/html/2610.25084)
- **H-TCC (hierarchical causal explainability with MLLMs, 2026).** Ranks key moments by temporal saliency, then checks them with occlusion. Blurring peak-saliency frames flips the MLLM's decision more often than blurring random frames. This is the "saliency, then deletion check against random" protocol. — [When, Where, and Why (arXiv 2610.17791)](https://arxiv.org/html/2610.17791)
- **EUSIPCO 2023 comparison** of IG, SmoothGrad, KernelSHAP, GradientSHAP, LIME and Occlusion at clip and frame level for video anomaly detection. Occlusion with large perturbations outperformed small perturbations. — [Quantitative Evaluation of Video Explainability Methods (EUSIPCO 2023)](https://eurasip.org/Proceedings/Eusipco/Eusipco2023/pdfs/0001235.pdf)
- **Older perturbation work.** Extremal spatiotemporal perturbation masks for video networks. — [Li et al., Towards Visually Explaining Video Understanding Networks with Perturbation (arXiv 2005.00375)](https://arxiv.org/pdf/2005.00375)
- **Temporal-explainability evaluation (2026).** Frame-wise explanation metrics can reach different conclusions depending on whether intermediate or final-prediction representations are scored. The paper argues for temporally aware evaluation. It uses ConvLSTM + Grad-CAM, not LLMs. — [arXiv 2609.08043](https://arxiv.org/pdf/2609.08043)
- **"Query restricted to one frame".** A frame-attribution variant done with attention masks inside a video LLM (Qwen2.5-VL-7B). Letting the query read only Frame 1 dropped MCQ probability by over 8%; Frame 4 only, by under 2%. — [Causality Matters (arXiv 2508.11576)](https://arxiv.org/html/2508.11576)
- **A naming trap.** "Saliency" in many video-LLM papers (e.g. VideoExpert, highlight detection) means a score the model outputs, not a post-hoc attribution. — [VideoExpert (arXiv 2504.07519)](https://arxiv.org/html/2504.07519)

### Inferences (how it applies to MVP)
- **Frame deletion at the input/token level** is the most direct analogue of the existing attention-swap ablation, but on the time axis. For each question, replace frame f's 300 visual tokens (and, separately, its 300 geometry tokens) and measure Delta log p(correct option). The replacement can be mean-token, a forward-fill copy of the neighbouring frame (SFS style), or deletion of both the tokens and their positions.
  - Do it in 3 conditions: (a) visual + geometry removed, (b) geometry only, (c) visual only. Conditions (b)/(c) separate "which frames does the geometry help on" from "which frames does the model look at".
  - Cost: 32 frames x 3 conditions ~= 96 forward passes per question. With ~4.7k-text Llama-8B forward passes at ~1.5 min on a B300 (cluster reference timing), a 2B model with ~9.7k tokens per sample should manage roughly hundreds to a thousand questions per GPU-hour. This is an extrapolation, not a measurement.
- **Use forward-fill rather than zeroing for geometry tokens.** Zeros (or a missing point map) are out of distribution for the cross-attention arm's distance bias -nu_h * ||x_i - x_j||. A zero coordinate places every cell at the origin and changes the bias for all other tokens. The cleanest deletion for the XA arm is to mask frame f's geometry tokens out of the cross-attention keys, which is well defined because that is the project's own module, not the backbone.
- **Sufficient frame subsets** directly answer "how many of the 32 frames does relative direction need?" Hypothesis to test: relative-direction questions need the frames showing the observer/anchor object and the target object, typically 2-3 frames. If the XA arm's SFS for relative-direction items is smaller, or contains frames where the objects are *not* visible (because geometry from other frames is pulled in via cross-attention), that is evidence of cross-frame geometric reading. Cost ~= O(T) to O(T^2) passes, e.g. 32-500 per question. Run it on a subset (e.g. 200 relative-direction items x 2 arms x 2 seeds).
- **Frame-level Shapley (permutation sampling).** With 32 players, ~20-50 sampled permutations x 32 = 640-1,600 forward passes per question. That is affordable only on a few hundred items. Deletion/insertion curves plus SFS are probably enough.
- **Integrated gradients over frames.** Scale frame f's visual and/or geometry embeddings from a baseline (mean token, or zero injection, which is a natural baseline for the additive arm) to the real value, then sum the IG attributions over the frame's 300 tokens. Cost ~= 20-50 steps x (forward + backward), batched over interpolation steps. The existing grad x input saliency is the 1-step version. Summing it per frame gives a per-frame temporal saliency curve for free, but the SFS paper's result suggests checking it against deletion.
- **Faithfulness check to report:** deletion/insertion AUC of grad x input frame rankings vs random frame rankings, as in H-TCC's "peak vs random frame occlusion".

### Gaps
- No paper found that runs frame-level Shapley or SFS on a video *LLM* (Qwen-VL, LLaVA-Video) with an MCQ answer target. The methods are transferable but untested there.
- No published wall-clock numbers for frame-deletion sweeps on 2B video LLMs. The cost figures above are extrapolations.

---

## Q2. Which papers study how video LLMs actually use multi-frame evidence?

### Takeaway
Many video benchmarks can be solved from one frame, from shuffled frames, or with no video at all. For VSI-Bench specifically, the original paper reports that relative direction, route plan and absolute distance gain *little* from vision over blind models, and a debiased VSI-Bench now exists. Any explanation of the +2-5 point relative-direction gain should be checked against blind, single-frame and shuffled-frame controls, and against VSI-Bench-Debiased.

### Cited Findings
- **Single-frame bias (foundational, 2022).** A single-frame-trained model can match multi-frame models on many video-language benchmarks. — [Lei et al., Revealing Single Frame Bias (arXiv 2206.03428)](https://arxiv.org/pdf/2206.03428)
- **TVBench / "Lost in Time".** Questions should not be answerable from a single random frame or from shuffled frames. On TVBench, single-image MLLMs are at chance, and shuffling or reversing the video causes big drops for video models. Many MVBench tasks are solvable from a single frame. — [TVBench (arXiv 2410.07752)](https://arxiv.org/html/2410.07752v2)
- **Shuffle robustness of closed models.** On other benchmarks, GPT-4o and Gemini-1.5-Pro scores stay stable under frame shuffling, and shuffling sometimes improves them. — [TVBench / Lost in Time (arXiv 2410.07752)](https://arxiv.org/pdf/2410.07752)
- **VidHal.** Video LLMs relied on single-frame information for over half of the queries. — [VidHal (arXiv 2411.16771)](https://arxiv.org/pdf/2411.16771)
- **TimeBlind.** GPT-5 scores 52.2 accuracy with one random frame vs 46.3 with the full video. On the stricter per-question metric it scores only 4.5. — [TimeBlind (arXiv 2602.00288)](https://arxiv.org/pdf/2602.00288)
- **Benchmark decomposition.** Splits video-LLM benchmarks into knowledge, spatial perception and true temporal understanding. — [Breaking Down Video LLM Benchmarks (arXiv 2505.14321)](https://arxiv.org/pdf/2505.14321)
- **Thinking in Space / VSI-Bench**, 288 indoor videos and more than 5,000 QA pairs. — [Yang et al. (arXiv 2412.14171)](https://arxiv.org/pdf/2412.14171)
  - Overall, blind ("vision disabled") models perform below chance and video is essential.
  - **Absolute distance, route plan and relative direction show little gain from vision.** MLLMs "struggle to improve beyond chance level" on these whether vision is enabled or not.
  - Frame count only marginally affects performance.
  - In a self-explanation analysis of 163 errors, ~71% were attributed to spatial reasoning, e.g. following the egocentric view instead of building an allocentric one.
  - Cognitive maps: Gemini-1.5 Pro is ~64% correct on adjacent objects, falling to 0 for the farthest pairs ("local world models"). Relative distance goes 46.0% -> 56.0% with its own map -> 66.0% with the ground-truth map.
  - Source for these figures: [HTML v2](https://arxiv.org/html/2412.14171v2). The per-task blind gaps are in its Table 13 / Fig. 13, which were not extracted.
- **VSI-Bench-Debiased ("Benchmark Designers Should Train on the Test Set", Nov 2025).** The Test-set Stress-Test (TsT) fine-tunes on text-only inputs. This lifted LLaVA-Video-7B's blind VSI-Bench score from 25.9% to 44.7%. Iterative Bias Pruning removed ~30.7% of samples, after which the blind score fell to 32.0%. Available as the `debiased` config of `nyu-visionx/VSI-Bench`. — [arXiv 2511.04655](https://arxiv.org/html/2511.04655v1); [HF papers page](https://huggingface.co/papers/2511.04655). Figures are partly from summaries and should be verified against the paper.
- **SIMS-V** reports gains on the debiased split (LLaVA-Video +7.7, LLaVA-OneVision +5.4) as evidence of real visual reasoning. — [SIMS-V (arXiv 2511.04668)](https://arxiv.org/pdf/2511.04668)
- **Frame sampling on VSI-Bench.**
  - ViCA: adding time instructions or more frames did not help spatial reasoning. — [ViCA (arXiv 2505.12312)](https://arxiv.org/html/2505.12312v1)
  - Struct2D: sparse sampling misses critical evidence, so it selects frames by 3D projection coverage. — [Struct2D (arXiv 2506.04220)](https://arxiv.org/pdf/2506.04220)
  - Spatial-MLLM (VGGT-style geometry encoder + Qwen2.5-VL) gets strong VSI-Bench results with 16 frames. — [Spatial-MLLM (arXiv 2505.23747)](https://arxiv.org/pdf/2505.23747)
- **Temporal-order sensitivity is carried by attention, not position encoding** (Qwen2.5-VL-7B on TempCompass). — [Causality Matters (arXiv 2508.11576)](https://arxiv.org/html/2508.11576)
  - Removing position encodings from intermediate/deep layers changes ground-truth probability by about +/-2%; removing them from the first layer drops it by up to 60%.
  - Shuffling video position IDs has negligible effect (-0.02% / +0.07%), while shuffling query position IDs drops MCQ probability by 53%.
  - Reversing position IDs changes it by only -1.1 to -1.8%, but reversing the actual frame order drops MCQ by 15.45% (8.93% for the 3B model).

### Inferences (how it applies to MVP)
- **Run 4 cheap controls per arm, each a single forward pass per question:**
  1. Blind: no frames, no geometry.
  2. Single best frame / single random frame.
  3. Shuffled frame order, keeping each frame's geometry attached to its own frame.
  4. Geometry-shuffled: frames in order, but point maps permuted across frames (or across scenes).
  - Control 4 is the key test for the XA arm. If its relative-direction gain survives cross-scene geometry shuffling, it is not using geometry content. If it survives within-scene frame permutation of geometry, then frame identity of geometry does not matter, which is expected because VGGT point maps share one world frame.
- **Report relative-direction gains on VSI-Bench-Debiased** as well as full VSI-Bench. Thinking in Space says relative direction is near chance even with vision, so a +2-5 point gain could sit partly on exploitable non-visual priors. Seed variance also matters at this size.
- **Order sensitivity.** Because VGGT point maps are expressed in the first frame's coordinate system, reversing frame order changes the world frame for geometry but not the visual content. Comparing "reverse frames + recompute VGGT" vs "reverse frames + keep original geometry" separates temporal-order effects from coordinate-frame effects. Relative direction depends on the observer's facing, which is defined in the question, not by camera order.

### Gaps
- Per-task Vision-Enabled minus Vision-Disabled numbers for VSI-Bench (Table 13) were not extracted. Check the paper.
- No paper found that applies shuffle or single-frame controls specifically to geometry-augmented VLMs (Spatial-MLLM, VG-LLM, etc.).

---

## Q3. Causal tracing, activation patching and attention knockout over frames/temporal tokens; information flow from video tokens to answer

### Takeaway
A consistent picture has emerged from four 2025-26 papers, all on fully softmax-attention 7B-class models:
- Cross-frame (video-video) interaction happens in early-to-middle layers.
- Video information is moved into the question/query tokens in middle layers.
- The final/answer token reads mostly from the query tokens, not directly from video tokens.
- Large parts of the attention graph (~40-60% of edges, or all video tokens after ~60-70% depth) can be removed with little accuracy loss.

The standard protocol is the Geva et al. (2023) attention knockout: mask edges to -inf over a sliding window of k layers. For Qwen3.5-2B, it can be applied only at the 6 full-attention layers.

### Cited Findings
- **Map the Flow (ICLR 2026)**, Kim, Kim & Han. — [arXiv 2510.13251](https://arxiv.org/abs/2510.13251); [HTML](https://arxiv.org/html/2510.13251v2); [project + code](https://map-the-flow.github.io)
  - Method: attention knockout sets mask = -inf on chosen (source -> target) pairs and measures the relative change in p(predicted answer). Cross-frame blocking stops vision tokens from attending to preceding frames, within a window of k = 9 layers centred on layer l. Following Geva et al., narrower windows let information bypass the block through the residual stream. The paper also uses logit lens on video tokens, counting spatial/temporal keywords.
  - Models: LLaVA-NeXT-7B/13B-Video-FT, Mini-InternVL-4B-Video-FT, VideoLLaMA3-7B. **No Qwen-VL models.** Benchmarks: TVBench, TOMATO, VCGBench.
  - Blocking cross-frame attention in layers 1-16 drops accuracy: Action Antonym -24.1, Action Sequence -20.2, Scene Transition -18.0, **Moving Direction -44.8**, Object Count -60.8. The image-only LLaVA-NeXT-7B is barely affected, so cross-frame interaction is learned during video fine-tuning.
  - Video->question flow peaks early-to-middle. Question->last flow runs middle-to-late. True-option probability jumps around layer 20.
  - "Effective pathways": cross-frame edges kept in L6-15, video->question in L6-20, question->last in L16-25, and video->last disabled everywhere. This keeps 42% of edges (10.8M of 25.7M). LLaVA-NeXT-7B-Video-FT scores 51.5/30.2 (full) vs 51.2/29.2 (pathways) vs 40.1/23.1 (random edges with the same budget) on TVBench/TOMATO.
- **"An Empirical Study on How Video-LLMs Answer Video Questions" (Aug 2025)**, Gou et al. — [arXiv 2508.15360](https://arxiv.org/abs/2508.15360); [HTML](https://arxiv.org/html/2508.15360)
  - Three knockouts: Video Temporal Knockout (inter-frame attention blocked), Video Spatial Knockout (intra-frame attention blocked) and Language-to-Video Knockout (text cannot read video). Applied globally, or in a 4-layer sliding window.
  - Models: LongVA, InternVideo2.5, LLaVA-Video-7B, LLaVA-OneVision-7B, and a 34B model. Benchmarks: Video-MME, MVBench, EgoSchema.
  - Temporal and spatial knockouts across *all* layers cause minimal degradation, while language-to-video knockout causes a large drop. In other words, spatial-temporal modelling relies on language-guided retrieval more than on video self-attention.
  - Layers 12-16 are critical and layer 8 is an outlier. Blocking beyond ~60% depth barely matters.
  - Exit (drop video tokens after layer 18) plus spatial-only attention in layers 1-8 cuts attention FLOPs to 37.1%. LLaVA-Video Video-MME goes 62.4 -> 60.0 and MVBench 61.1 -> 60.8. Temporal attention costs ~31x spatial attention.
- **"Causality Matters" (AAAI 2026)**, Shi, Long, Wu & Wang. — [arXiv 2508.11576](https://arxiv.org/html/2508.11576)
  - Models: **Qwen2.5-VL-7B** (main), plus Qwen2.5-VL-3B and LLaVA-OneVision-7B. Benchmarks: TempCompass, NExT-QA, ActivityNet-QA. 5-layer sliding-window knockout.
  - Final token -> video frames: minimal impact. Final token -> query: drops above 40% (MCQ). Inter-frame blocking: peak around layer 10, 4-6% drop. Frame -> query: sharp decline around layer 15.
  - "Corresponding Area" test: letting each token attend only to the same spatial location in other frames keeps the drop under 1%. Blocking the "Previous Frame" hurts more.
  - Temporal cues accumulate in the final frame (under causal attention) and then flow to the query.
  - Disabling inter-frame attention in layers 20-28, or removing all frame tokens from the KV cache in layers 20-28, leaves accuracy unchanged, e.g. NExT-QA 75.1 -> 75.2 / 75.1.
- **CircuitProbe (2025, v2 Mar 2026)**, Zhang et al. — [arXiv 2507.19420](https://arxiv.org/abs/2507.19420)
  - "Visual auditing": localise object semantics in the video-token sequence, then test necessity by ablating or substituting them.
  - "Semantic tracing": logit lens over layers plus temporal frame interventions.
  - Finds a critical layer interval. Amplifying temporally specialised heads in that interval gives up to +2.4% on TempCompass. Object evidence is necessary but not sufficient.
- **Related 2026 work.**
  - "Tracing the Arrow of Time" reports that temporal information is often lost between the vision encoder and the LLM. — [arXiv 2605.07568](https://arxiv.org/pdf/2605.07568)
  - "MLLMs Know When Before Speaking": temporal grounding can be recovered from attention cues. — [arXiv 2605.21954](https://arxiv.org/pdf/2605.21954)
  - "From Senses to Decisions" applies knockout to audio-visual LLMs. — [arXiv 2606.10147](https://arxiv.org/html/2606.10147)
  - These were seen only as abstracts or snippets.

### Inferences (how it applies to MVP)
- **Where knockout is possible in Qwen3.5-2B:** only layers 3, 7, 11, 15, 19 and 23 (full attention). A Geva-style window of 5-9 layers covers just 1-2 softmax layers. Information can also move between frames through the recurrent state of the Gated-DeltaNet-style linear layers, so **blocking softmax edges alone will under-estimate cross-frame flow.**
  - Proposed workaround: reset the linear-attention recurrent state at each frame boundary (block-diagonal "per-frame" recurrence) in a chosen layer range. This is the linear-attention analogue of Video Temporal Knockout. It needs custom code in the linear-attention kernel; a chunked/recurrent reference implementation is easier to patch than the fused kernel.
  - Alternative: knock out at the residual-stream level, e.g. replace frame f's hidden states with a forward-fill copy after layer l. That is coarser but architecture-agnostic.
- **Experiments that map directly onto the project's questions:**
  1. *Video -> question vs video -> last token* knockout at layers 3/7/11/15 per arm. Question: does the XA arm move more spatial information into question tokens earlier than the additive arm?
  2. *Cross-frame knockout* (inter-frame softmax edges at full-attention layers plus per-frame state reset in linear layers) for relative-direction items. The XA arm already does cross-frame mixing through its geometry cross-attention, so the prediction is that it should be **less** hurt by backbone cross-frame knockout than the additive arm. That would be a clean mechanistic signature of "the geometry cross-attention replaces backbone cross-frame work".
  3. *Corresponding-area knockout* (Causality Matters): restrict cross-frame attention to cells whose VGGT 3D points are within r metres, instead of to the same 2D location. This is a 3D version of the test. If accuracy holds, the backbone's cross-frame flow is 3D-local, which is exactly what the XA bias encodes.
  4. *Activation patching across arms or conditions:* patch the injection-layer outputs (layers 0-2) of the XA arm from a geometry-shuffled run into a clean run, one frame at a time. This gives per-frame causal effects of geometry, done at the project's own module, so the linear-attention issue does not arise.
- **The existing Jacobian-lens analysis** corresponds to the logit-lens component of Map the Flow and CircuitProbe. Restricting it to tokens of specific frames (the frames where question objects appear, from the SAM3 masks) would add the temporal dimension.
- **The existing attention-swap ablation** (learned -> bias-only / own-cell / uniform) is itself a knockout on the project's module. Map the Flow's "random edges with the same budget" control is the right comparison to add, e.g. own-cell vs a random single cell.

### Gaps
- No interpretability paper found on hybrid linear/softmax attention VLMs (Qwen3.5, Qwen3-Next style). How to define "token j cannot influence token i" in Gated DeltaNet layers is an open methodological issue for this project.
- None of these papers studies spatial (VSI-Bench-type) tasks. All use action/temporal benchmarks. "Moving Direction" (TVBench) is the closest analogue to relative direction, with a -44.8% drop under cross-frame blocking in Map the Flow.

---

## Q4. Cross-frame correspondence: does the model link the same object across frames?

### Takeaway
Current VLMs are reported to be weak at cross-view instance correspondence. Evidence comes from attention-vs-mask overlap analyses (SpatialMosaic) and point-correspondence benchmarks. Dedicated "correspondence heads" have been probed in DUSt3R-style multi-view transformers and in multi-view diffusion models (Probe3D-style), but not systematically in video LLMs. That is a gap the XA arm is well placed to fill, because it has explicit 3D positions per token.

### Cited Findings
- **SpatialMosaic.** Analyses whether VLM attention lines up with the target object's mask across views. As view overlap increases, attention becomes more consistent with the target mask, so shared visual content helps cross-view correspondence. Same-category duplicates make instance correspondence ambiguous. Current VLMs often fail at reliable cross-view alignment, instance correspondence and multi-hop evidence aggregation. — [arXiv 2512.23365](https://arxiv.org/html/2512.23365v4)
- **Cross-view point correspondence.** Introduces a benchmark and finds that VLMs lack robust point-level cross-view correspondence. — [arXiv 2512.04686](https://arxiv.org/html/2512.04686)
- **Cross-view relation fine-tuning.** Fine-tuning on cross-view relations lets Qwen3-VL-2B-XVR exceed human performance on point correspondence. — [arXiv 2603.27967](https://arxiv.org/html/2603.27967)
- **S2-MLLM.** An MLLM cannot tell whether chairs seen from different viewpoints are the same physical object. — [arXiv 2512.01223](https://arxiv.org/pdf/2512.01223)
- **Understanding multi-view transformers.** Several DUSt3R cross-attention heads act as correspondence heads. — [arXiv 2510.24907](https://arxiv.org/html/2510.24907)
- **CAMEO.** Measures geometric correspondence in a multi-view diffusion model's attention maps, following Probe3D. A substantial gap remains vs a strong baseline (VGGT). — [arXiv 2512.03045](https://arxiv.org/pdf/2512.03045)
- **Causality Matters.** Its "Corresponding Area" knockout shows that same-location cross-frame attention alone keeps Qwen2.5-VL within 1% on TempCompass. — [arXiv 2508.11576](https://arxiv.org/html/2508.11576)

### Inferences (how it applies to MVP)
- **Correspondence score for the XA arm.** Use the SAM3 masks to get object instance IDs per cell per frame. Tracking IDs across frames are needed: SAM3 video tracking, or 3D clustering of VGGT points. For each head h at layers 0-2, compute the attention mass that a query cell of object o in frame t places on key cells of the *same* object o in other frames t' != t. Compare that against:
  - (i) cells of other objects at similar 3D distance;
  - (ii) the bias-only attention softmax(-nu_h * d), which already favours nearby 3D points.
  - "Learned minus bias-only same-instance mass" measures whether the content term adds instance linking beyond geometric proximity.
  - This extends the existing object-to-object vs bias-only analysis along the time axis. Cost: one forward pass with attention outputs saved per question; the 9,600 x 9,600 x 16 attention for 3 layers is ~4.4 GB in fp16 per question, so reduce it on the fly.
- **Probe3D-style correspondence probe for the backbone.** For pairs of cells in different frames that VGGT says lie within epsilon in 3D, test whether the hidden states at layer l are nearest neighbours (or a linear probe classifies "same 3D point / object"). Compare the base Qwen3.5-2B, the additive arm and the XA arm. Prediction: the additive arm makes same-3D-point tokens more similar, because the same geometry is added, even if the model does not use it. Hence a **control: shuffled-geometry runs**.
- **Causal version.** Knock out same-instance cross-frame cross-attention edges in the XA module (mask key cells with the same instance ID in other frames) and measure Delta accuracy on relative direction vs on distance tasks. If relative direction relies on linking the anchor/facing objects across frames, this should hurt more than a random edge mask with the same edge count.

### Gaps
- No paper found that directly probes whether video LLMs track the same object instance across frames through attention or hidden-state similarity. This appears to be an open gap in the literature (2026-11).
- The SpatialMosaic, S2-MLLM and DUSt3R-head findings were taken from search snippets and abstracts. Exact numbers were not verified.

---

## Q5. Compute cost for a 2B model with ~9,600 visual tokens per question

### Takeaway
Perturbation and knockout studies of the published size (thousands of questions x tens of conditions) are cheap for a 2B model. Costs scale as (questions x conditions x forward passes). The main practical limits are:
- materialising attention maps: 9.7k^2 entries per head per softmax layer;
- needing eager attention for custom masks, which blocks FlashAttention;
- the custom kernel work needed for linear-attention layers.

### Cited Findings
- **Published experiment sizes.**
  - Map the Flow used 8 frames x 144 tokens (~1.2k visual tokens) for LLaVA-NeXT-7B-Video-FT; the full model has ~25.7M attention edges. — [arXiv 2510.13251](https://arxiv.org/html/2510.13251v2)
  - Gou et al.: temporal attention costs ~31x spatial attention; exit + window reduces attention FLOPs to 37.1%. — [arXiv 2508.15360](https://arxiv.org/html/2508.15360)
  - Sufficient Frame Subsets: 135 model evaluations for a 16-frame video; O(T k) with batching. — [arXiv 2610.25084](https://arxiv.org/html/2610.25084)
  - Play Fair: its ESV approximation scales linearly in the number of frames. — [arXiv 2011.12372](https://arxiv.org/pdf/2011.12372)
- **Project-specific facts.** Qwen3.5-2B has 6 full-attention layers with 8 query heads / 2 KV heads, plus 18 linear-attention layers (local config). Cluster reference timing: Llama-3.1-8B forward passes over ~4.7k short texts take ~1.5 min on one B300.

### Inferences (estimates, not measurements)
- **Edge count.** With ~9.7k tokens, a full-attention layer has ~9.7k^2/2 ~= 47M causal edges per head. That is ~2x Map the Flow's whole-model edge count per head per layer. Knockout masks are dense boolean [T, T] tensors, ~94 MB each, reusable across heads.
- **Per-question forward-pass budgets:**

  | Method | Forward passes per question |
  |---|---|
  | Blind / single-frame / shuffle controls | 1 each |
  | Frame deletion | 32-96 |
  | IG over frames | 20-50 forward + backward (backward ~2x forward) |
  | Sliding-window knockout | (6 softmax layers + linear-layer variants) x 3-4 edge types ~= 20-40 |
  | SFS | 32 to a few hundred |
  | Frame Shapley | ~1,000 |
  | Per-frame activation patching at layers 0-2 | 32 |

- **Example study size.** 500 relative-direction items x 2 arms x 2 seeds x ~100 passes = 200k forward passes of a 2B model at ~10k tokens. At an assumed 5-10 passes/s on a B300 with batching, that is roughly 6-11 GPU-hours. The rate is an assumption; benchmark it with a 50-item smoke run first.
- **Attention capture.** Saving full 9.6k x 9.6k maps for 16 XA heads x 3 layers is ~4.4 GB/question in fp16. Compute reductions on the fly (per-frame, per-object attention mass) instead of storing them.
- **Eager attention.** Masks for knockout in softmax layers need eager attention or FlexAttention (block masks) instead of SDPA/FlashAttention. FlexAttention block masks are a good fit for "frame-block" knockouts because they keep memory at O(T) rather than O(T^2). VGGT outputs are fixed per video, so cache them once and reuse them across all perturbations except "recompute VGGT after reordering".

### Gaps
- No measured throughput for Qwen3.5-2B at ~10k tokens with eager vs flex attention on FRIDA nodes. A smoke run is needed.
- The cost of patching the linear-attention (Gated DeltaNet) kernel for per-frame state resets is unknown. No reference implementation was found.
