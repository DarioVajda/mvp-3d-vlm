# Interpretability of foundation models with respect to spatial, geospatial and 3D information

Project context, used in the "how it applies" notes: VGGT point-head 3D point maps are pooled to one geometry token per merged 32-px cell (9,600 visual tokens for 32 frames, camera-0 frame, in cell units). They are fused into Qwen3.5-2B at decoder layers 0–2 through one of two arms: additive injection, or 3D-aligned cross-attention with distance bias -nu_h*min(d,10) and 3D RoPE on (t,x,y,z). Evaluation is on VSI-Bench. Cross-attention gains only on relative direction. Existing analyses: grad×input, SAM3 masks, object-to-object attention, Jacobian lens, attention swap.

Source-quality note: items marked "(snippet)" were checked only through search-result summaries or third-party digests, not the full paper. Items marked "(prior knowledge)" are well-known papers I did not re-fetch in this session; verify their exact numbers before citing them in the thesis.

---

## 1. Do LLMs/VLMs linearly represent space (and time)? World-model probing, depth/3D/pose/frames in VLMs

### Takeaway
Text LLMs linearly encode real-world coordinates (Gurnee & Tegmark), and VLMs bind object locations to text tokens through linear "spatial IDs" that causally mediate answers at intermediate layers. But depth/distance is consistently the weakest and most entangled axis: VLMs conflate "higher in the image" with "farther". So the axis VSI-Bench distance tasks need is exactly the one that is poorly organised in the base model.

### Cited Findings
- **Gurnee & Tegmark, "Language Models Represent Space and Time" (ICLR 2024).** Llama-2 learns *linear* representations of spatial (world/US/NYC place) and temporal coordinates at multiple scales. Method: linear (ridge) probes on residual-stream activations of entity names, per layer. Compute: forward passes plus closed-form ridge, so cheap. — [arXiv 2310.02207](https://arxiv.org/abs/2310.02207)
- **Tehenan et al., "Linear Spatial World Models Emerge in LLMs" (2025).** Linear and nonlinear probes on a synthetic dataset of object positions, plus geometric-consistency checks. The authors caveat that generalisation to real data needs verification. — [arXiv 2506.02996](https://arxiv.org/pdf/2506.02996) (snippet)
- **Kang, Chen, Gkioxari, Perona, "Linear Mechanisms for Spatiotemporal Reasoning in VLMs" (Jan 2026).** VLMs encode object locations by *linearly binding "spatial IDs" to textual activations* and then reason via language tokens. Spatial IDs are ubiquitous and *causally mediate model beliefs at intermediate layers*. The same holds for video VLMs, which have an analogous linear *temporal ID*. Spatial IDs can diagnose VLM limitations and serve as a learning signal. Code: github.com/Raphoo/linear-mech-vlms. — [arXiv 2601.12626](https://arxiv.org/abs/2601.12626v1) (abstract only; the specific models and layers were not visible)
- **"Why Far Looks Up" (Min et al., ECCV 2026 Oral).** Contrastive probing with minimal pairs measures how horizontal, vertical and distance axes are organised in VLM embeddings. Two metrics: Coh_D (cosine coherence of sign-corrected delta vectors) and VD-EI (vertical–distance entanglement index).
  - **Distance is the weakest axis in every model and scale.** For RoboRefer-2B, Coh is 0.649 horizontal, 0.830 vertical and 0.182 distance.
  - **"Far" and "above" occupy the same region in weaker models.** These include Molmo-7B, NVILA-Lite-2B and Qwen2.5-VL-3B. Qwen3-VL-235B separates the axes cleanly.
  - **Coh_D correlates r=0.92 with benchmark accuracy across 8 VLMs.** Counter-heuristic accuracy drops by 3.5–34 points. Scaling data improves the consistent cases but does not close the gap. No causal intervention is reported. — [project page](https://cheolhong0916.github.io/whyfarlooksup/), [arXiv 2605.30161](https://arxiv.org/pdf/2605.30161)
- **Theodoridis & Mohandas (2026).** Linear probes on four VLMs' intermediate activations, using counterfactual image sets. Presence and count are well encoded, fine-grained spatial concepts less so. Linear separability *degrades quickly as the object's distance increases*. — [arXiv 2603.06054](https://arxiv.org/html/2603.06054) (snippet)
- **"Do Foundation Models Know Geometry?" (2026).** Linear probes on frozen features of 14 backbones show that hand pose, head pose, object pose and camera intrinsics are linearly readable. — [arXiv 2603.06459](https://arxiv.org/html/2603.06459) (snippet)
- **WM-ABench (Jun 2025).** A behavioural (not probing) test of VLMs as world models. Even advanced models score under 60% on spatial positioning, against 90–100% for humans. — [alphaxiv 2506.21876](https://www.alphaxiv.org/overview/2506.21876) (snippet)
- **VLM3 (2026).** Training standard VLMs to output depth and camera pose raises camera-pose AUC30 from 5% to 94% over the base VLM. The base VLM therefore barely encodes usable pose. — [arXiv 2605.30561](https://arxiv.org/html/2605.30561) (snippet)
- **SpaRRTa (2026).** A synthetic Unreal Engine 5 benchmark for relational spatial concepts in vision foundation models. — [arXiv 2601.11729](https://arxiv.org/pdf/2601.11729) (snippet)

### Inferences (how it applies to the project)
- **Probe B1: linear coordinate probes on Qwen3.5-2B residuals.** Gurnee-style ridge probes regress the cell's VGGT (x,y,z) in camera-0 frame and its depth from visual-token hidden states at every decoder layer. Run them for three models: the base model, the additive arm and the cross-attention arm. If injection works, R² for z/depth should jump at layers 0–3 and persist into mid layers. If R² decays back to the base-model level by about layer 8–12, the geometry is injected but "washed out" before the reasoning layers. Cost: one forward pass over about 1–2k VSI-Bench clips per arm, caching hidden states for visual tokens (subsample cells), then sklearn ridge. Minutes on one B300 plus a CPU fit.
- **Probe B2: entanglement metric.** Compute a VD-EI-style metric for the fused models: is the "distance" direction still coupled to the image-y direction? 3D geometry should *decouple* depth from image height. This is a clean, cheap, thesis-friendly mechanistic claim.
- **Egocentric vs allocentric probes.** The geometry is in camera-0 frame, which is allocentric-ish. Probe for both camera-0 coordinates and per-frame camera-centric coordinates (VGGT gives extrinsics). Relational answers depend on the frame. Relative direction on VSI-Bench is posed from a standing/facing viewpoint, so it needs an allocentric-to-egocentric transform. That could explain why cross-attention, with 3D RoPE in a shared frame, helps exactly that task.
- **Spatial IDs as the readout channel.** Kang et al.'s finding suggests checking whether object-name text tokens in the question carry linearly decodable 3D position after fusion. This is a probe on the *text* token positions. It tests whether geometry reaches the language tokens, not just the visual ones. The existing object-to-object attention and Jacobian-lens analyses can be reframed as tests of this pathway.

### Gaps
- I found no paper that probes a *video* VLM's hidden states for metric 3D world coordinates (as opposed to 2D image relations) before and after geometry fusion. This appears to be an open niche.
- I did not see the full Kang et al. paper, so the models, layers and intervention details are unknown here.
- Egocentric-vs-allocentric representational probing: I found no dedicated paper (only VSI-Bench's behavioural analysis, Q3).

---

## 2. Interpretability of 3D foundation models (VGGT, DUSt3R/MASt3R, CroCo, DINOv2): what layers encode, which to tap

### Takeaway
In VGGT, correspondence and epipolar geometry emerge in the middle global-attention layers (correspondence heads from about layer 10, a sharp change at layers 12–14 of 24), and knocking out middle-layer heads breaks it. Probes on DUSt3R show the pointmap is built progressively and refined with correspondences. Downstream VLM-fusion papers find that different VGGT layers favour different VSI-Bench tasks: mid around layers 11–12, later around layers 17–23. Relative direction and absolute distance prefer the later layers.

### Cited Findings
- **Bratulić et al., "On Geometric Understanding and Learned Data Priors in VGGT" (Dec 2025).**
  - **Architecture:** VGGT has 24 alternating frame/global attention blocks on a DINOv2-L backbone (patch 14, 518×518 input, 1369 patches plus 1 camera and 4 register tokens per view).
  - **Epipolar geometry:** The fundamental matrix is decodable from *camera tokens*. Sampson error drops sharply around layer 12, continues to improve to the end, and is most marked at layers 13–14.
  - **Correspondence heads:** Strong correspondence-matching heads span layers 10–16.
  - **Attention knockout:** Zeroing QK weights for selected layer-heads in the *middle* layers makes F recovery fail completely, while early and late knockouts barely matter. Knocking out just two strong matching heads in one layer causes measurable degradation.
  - **Data priors:** On occluded patches about 50% of matching heads remain, which the authors call hallucination. VGGT is brittle when only one view's lighting changes (neon, rainbow). It exploits shadow cues in repetitive scenes but fails in fully ambiguous ones.
  - Synthetic ShapeNet-based scenes. — [arXiv 2512.11508](https://arxiv.org/html/2512.11508v1)
- **Stary, Gaubil, Tewari, Sitzmann, "Understanding Multi-View Transformers" (ICCV 2025 E2E3D workshop).** A 5-layer MLP probe decodes pointmaps from each layer's patch features in a DUSt3R variant and visualises the 3D state block by block. Conclusion: it "estimates correspondences that are refined with reconstructed geometry". Code: github.com/JulienGaubil/und3rstand. — [arXiv 2510.24907](https://arxiv.org/pdf/2510.24907) (snippet)
- **El Banani et al., "Probing the 3D Awareness of Visual Foundation Models" (CVPR 2024).** Frozen-feature probes for depth and surface normals, plus zero-shot multiview correspondence. Models encode single-view depth and normals reasonably but are weak at *multiview consistency*. MiDaS, despite depth training, performs about as well as classification-trained models and worse than the best self-supervised or generative ones. Code: github.com/mbanani/probe3d. Compute: light, small DPT-style probes on frozen features. — [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Banani_Probing_the_3D_Awareness_of_Visual_Foundation_Models_CVPR_2024_paper.html), [arXiv 2404.08636](https://arxiv.org/pdf/2404.08636)
- **Man et al., "Lexicon3D" (NeurIPS 2024).** Seven encoders (image, video, diffusion, 3D point-cloud) are probed on scene VQA, grounding, segmentation and registration.
  - DINOv2 is best overall.
  - Video models excel at object-level tasks.
  - Diffusion models help geometric tasks.
  - Language-pretrained models (e.g. CLIP) show unexpected limits on language-related tasks.
  - The 3D-native Swin3D lags because of data scarcity and domain shift. — [project](https://yunzeman.github.io/lexicon3d), [arXiv 2409.03757](https://arxiv.org/html/2409.03757v3)
- **Chen et al., "Feat2GS" (CVPR 2025).** Reads 3D Gaussian attributes from frozen VFM features of unposed images and measures geometry vs texture awareness via novel-view synthesis, with no 3D ground truth. Geometry-strong models (e.g. RADIO) are often texture-weak, and concatenating RADIO and MAE helps. — [arXiv 2412.09606](https://arxiv.org/abs/2412.09606) (snippet)
- **SpatialStack (Zhang et al., CVPR 2026).** Taps frozen VGGT layers 11, 17 and 23 (local, mid and global geometry), each with its own projector, and injects them layer-by-layer into *different* LLM decoder layers.
  - **Single layers:** L11 scores 66.11 low-level / 64.48 high-level tasks. L23 scores 64.33 / 66.36.
  - **Naive vision-side multi-layer concatenation:** 64.69 / 65.15, worse than a single layer ("feature interference").
  - **LLM-side stacking:** 65.89 / 67.52. — [arXiv 2603.27437](https://arxiv.org/abs/2603.27437); numbers from [papernotes summary](https://en.papernotes.org/CVPR2026/vlm_reasoning/spatialstack_layered_geometry-language_fusion_for_3d_vlm_spatial_reasoning/) (snippet)
- **GeoAlign (2026).** VGGT layer 20 wins on *relative direction* and absolute distance, while layer 12 wins on appearance order and room size: "no single layer is sufficient". Dynamic routing beats single-layer and mean-pooled features. Injecting geometry into the LLM's "semantic stage" hurts compared with injecting at visual features. — [arXiv 2604.12630](https://arxiv.org/pdf/2604.12630) (snippet)
- **GeoWeaver (2026).** VGGT layers respond heterogeneously to spatial structure. Visual tokens are routed over a multi-level geometry bank. — [arXiv 2605.22558](https://arxiv.org/html/2605.22558v1) (snippet)
- **VG-LLM (Zheng et al., 2025).** Feeding VGGT *latent features* beats feeding VGGT-predicted explicit signals (camera, depth, point maps), which the authors call sparse and error-prone. — [arXiv 2505.24625](https://arxiv.org/html/2505.24625v3) (snippet)

### Inferences (how it applies to the project)
- **The project's input is the most compressed form of VGGT output.** It uses the *point-head output* (explicit xyz), pooled to one token per cell. VG-LLM's ablation argues explicit point maps are a weaker signal than latents. GeoAlign and SpatialStack show mid and late latent layers carry task-specific information.
- **A natural ablation follows:** replace or augment xyz with pooled VGGT layer-12 and layer-20/23 tokens.
- **Explicit xyz is also an interpretability asset.** It enables exact counterfactuals (Q5), which latents do not.
- **Point-map quality check.** Because VGGT's priors can hallucinate under occlusion and lighting shifts (Bratulić et al.), first measure point-map error per VSI-Bench scene against ScanNet/ARKitScenes ground truth where available. Then stratify VSI-Bench gains by point-map error. If gains do not depend on geometry quality, the model is not using the geometry.
- **Relative direction and the late-layer preference.** GeoAlign reports layer 20 helps relative direction. The project's cross-attention arm only improves relative direction. Both suggest relative direction is the task where precise global 3D layout matters most, and where appearance shortcuts fail.

### Gaps
- No interpretability study of the VGGT *point head* or *camera head* themselves (versus the trunk) was found.
- MASt3R and CroCo-specific layer probing: not found beyond the DUSt3R-variant paper.
- SpatialStack's own per-layer table was not verified; the numbers come from a third-party summary.

---

## 3. Interpretability and explainability of spatial-reasoning VLMs (SpatialVLM, SpatialRGPT, Spatial-MLLM, VG-LLM, SpatialStack, Thinking in Space, AdaptVis)

### Takeaway
The main mechanistic explanation of VLM spatial failure is that attention to image tokens is mislocalised and miscalibrated. Correct answers correlate with attention aligning with object locations, and re-tempering image-attention logits (AdaptVis) gives large gains on 2D relations. For video, VSI-Bench shows the bottleneck is global layout and egocentric-to-allocentric transformation: local awareness is fine and global maps are poor. Explicit cognitive maps help. Geometry-fusion papers mostly report benchmark ablations, not internal evidence that geometry is used.

### Cited Findings
- **Chen et al., "Why Is Spatial Reasoning Hard for VLMs? An Attention Mechanism Perspective on Focus Areas" (ICML 2025).**
  - Errors coincide with attention being *misdirected to irrelevant objects*.
  - Image tokens get little total attention despite their count.
  - Familiar relations (left/right) fare better than unfamiliar ones (front/behind).
  - Success correlates with attention-to-object-location alignment.
  - **AdaptVis** is training-free. It scales the temperature of attention logits on image tokens, sharpening when confident and broadening when not. It reports up to +50 absolute points on WhatsUp for LLaVA. Code: github.com/shiqichen17/AdaptVis. Compute: inference only. — [arXiv 2503.01773](https://arxiv.org/html/2503.01773v2), [ICML](https://icml.cc/virtual/2025/poster/44272)
- **Yang et al., "Thinking in Space" (VSI-Bench).**
  - About 5k QA over about 288 indoor videos, 8 tasks (configurational, measurement, spatiotemporal).
  - Best model about 49% vs human about 79%.
  - Error analysis: spatial reasoning, especially *egocentric-allocentric transformation* and relational reasoning, is the main bottleneck.
  - Linguistic prompting (CoT etc.) does not help.
  - Prompting the model to *generate a cognitive map* (objects on a grid) improves relative-distance accuracy. Maps show good local but poor global layout. — [arXiv 2412.14171](https://arxiv.org/html/2412.14171v2)
- **"Make Geometry Matter for Spatial Reasoning" / GeoSR (Zhang et al., NUS, 2026).**
  - **Setup:** Qwen2.5-VL-7B with VGGT (static) or π³ (dynamic). Additive fusion F = F_V + MLP(F_G) yields *only marginal gains*. Ablation: no geometry 49.8 vs naive fusion 50.2 on static, and 64.0 vs 62.8 on dynamic, where naive fusion is worse.
  - **Fix:** MAE-style masking of vision tokens during training (γ=0.8), which forces reliance on geometry, plus a token- and channel-wise sigmoid gate α=σ(W[V‖G]).
  - **VSI-Bench:** average 51.9 vs VG-LLM 50.7. Relative direction 44.4 vs 40.7, route plan 35.6 vs 32.4.
  - The "underuse" claim rests on accuracy comparisons, not internal diagnostics. — [arXiv 2603.26639](https://arxiv.org/html/2603.26639v1)
- **VG-LLM ablations:** see Q2 (latent features vs explicit maps). — [arXiv 2505.24625](https://arxiv.org/html/2505.24625v3)
- **SpatialStack / GeoAlign / GeoWeaver:** layer-placement ablations, see Q2. GeoAlign reports injection *position* matters: geometry into the LLM semantic stage underperforms. — [arXiv 2604.12630](https://arxiv.org/pdf/2604.12630) (snippet)
- **Spatial-MLLM (Wu et al., 2025).** Dual encoder (Qwen2.5-VL vision encoder plus VGGT-based spatial encoder) with space-aware frame sampling, evaluated on VSI-Bench. — [arXiv 2505.23747](https://arxiv.org/abs/2505.23747) (prior knowledge)
- **VLM-3R (2025).** Feeds 3D reconstructive tokens (CUT3R-based) into a VLM for video spatial reasoning. — [arXiv 2505.20279](https://arxiv.org/abs/2505.20279) (prior knowledge)
- **SpatialVLM (Chen et al., CVPR 2024).** Synthesises about 2B spatial VQA pairs from 3D-lifted internet images (depth, metric scale). Improvement comes from *data*, and there is no internal-mechanism analysis. — [arXiv 2401.12168](https://arxiv.org/abs/2401.12168) (prior knowledge)
- **SpatialRGPT (Cheng et al., NeurIPS 2024).** Region-level spatial reasoning, with a plugin depth encoder and region representations. Its ablations add depth input and measure gains on SpatialRGPT-Bench. — [arXiv 2406.01584](https://arxiv.org/abs/2406.01584) (prior knowledge)
- **SpaceMind++ (2026).** Toward allocentric cognitive maps for spatially grounded video MLLMs. — [arXiv 2605.09449](https://arxiv.org/pdf/2605.09449) (title/snippet only)

### Inferences (how it applies to the project)
- **GeoSR replicates the project's additive-arm situation.** GeoSR's finding that additive fusion gives near-zero gains is the closest published analogue. Their remedy, masking vision tokens so the model must use geometry, is a training-time intervention. It doubles as a diagnostic: evaluate the trained model with vision tokens partly masked but geometry intact. If accuracy collapses much more for the additive arm than the cross-attention arm, the additive arm relies on appearance.
- **AdaptVis-style attention-temperature sweep.** Sweep the temperature of the image-token or geometry-bias logits (e.g. scale nu_h by 0, 0.5, 1, 2) at inference and plot per-task VSI-Bench accuracy. If relative-direction accuracy is monotone in the bias strength while other tasks are flat, the 3D-distance bias is causally responsible for the gain. Cost: only evaluation passes.
- **Cognitive-map readout as an explanation test.** Prompt both arms to output an object-coordinate map (VSI-Bench's cognitive-map prompt) and score it against ground-truth layout. If the cross-attention arm produces more globally consistent maps, that links the relative-direction gain to improved global layout, which is VSI-Bench's identified bottleneck.
- **Attention-to-object alignment, measured by the right objects.** Chen et al.'s alignment metric (attention mass on the queried objects' tokens) fits the existing SAM3-mask and object-to-object attention analysis. Report it split by correct vs incorrect answers and by arm.

### Gaps
- I found no paper applying probing or causal mechanistic analysis to Spatial-MLLM, VG-LLM, VLM-3R or SpatialStack internals. These papers rely on accuracy ablations.
- AdaptVis results on *video* or VSI-Bench were not found.
- SpatialVLM, SpatialRGPT, Spatial-MLLM and VLM-3R details are from prior knowledge, not fetched in this session.

---

## 4. Geospatial foundation models (Prithvi, SatMAE, GeoCLIP, SatCLIP, LLM geolocation) — transferable methods only

### Takeaway
Geospatial explainability work is mostly (a) linear or ridge probes of coordinates from hidden states (Gurnee-style), (b) sparse autoencoders or concept decompositions of location embeddings, and (c) concept-bottleneck geolocators. Probes and SAE/concept decomposition transfer directly to indoor 3D. Remote-sensing-specific methods (spectral band attribution, Prithvi/SatMAE patch saliency) do not add much beyond grad×input.

### Cited Findings
- **"What's in an Earth Embedding? An Explainability Analysis of Location Encoders" (2026).** Applies *sparse autoencoders* and *SpLiCE* concept decomposition to SatCLIP, GeoCLIP, Climplicit and CSP-fMoW embeddings. Pretraining data: GeoCLIP uses MP-16 Flickr images and SatCLIP uses S2-100K Sentinel-2. — [arXiv 2606.24997](https://arxiv.org/pdf/2606.24997) (snippet)
- **"Textual Supervision Enhances Geospatial Representations in VLMs" (ICML 2026).** Layer-wise ridge or linear probes for location across vision-only, CLIP-style and multimodal models. Text-supervised models encode location far better than vision-only ones. — [arXiv 2606.07172](https://arxiv.org/html/2606.07172v1), [summary](https://en.papernotes.org/ICML2026/interpretability/textual_supervision_enhances_geospatial_representations_in_vision-language_model/) (snippet)
- **Concept-aware GeoCLIP (2025).** Inserts a concept bottleneck between image and GPS embeddings so predictions decompose into named geographic concepts. It claims to be the first interpretability work for geolocalisation. — [arXiv 2509.01910](https://arxiv.org/pdf/2509.01910) (snippet)
- **LLM geospatial knowledge probing (2023).** Probes LLMs for geo-coordinates and tests distance and preposition reasoning. — [arXiv 2310.13002](https://arxiv.org/pdf/2310.13002v1) (snippet)

### Inferences (how it applies to the project)
- **SAE on fused visual tokens.** Train a small SAE on layer-2/3 visual-token residuals of the fused model (post-injection), then check whether some latents correlate with metric xyz, height above floor or distance to camera. That would be a "3D-concept" dictionary analogous to Earth-embedding SAEs. Cost: tens of millions of tokens is overkill. With about 9,600 tokens per clip, about 1–2k clips give about 10–20M tokens, which trains an SAE in under an hour on one GPU.
- **Probe metric choice.** The geolocation probe literature uses ridge regression with held-out *regions*, to avoid spatial leakage. The analogue here is held-out *scenes* (not frames) for 3D-coordinate probes, otherwise probes memorise rooms.

### Gaps
- Prithvi and SatMAE explainability papers were not searched in depth (out of scope per the brief). I did not find transfer-relevant methods beyond saliency, probes and SAEs.
- No study directly compares LLM hidden states with SatCLIP/GeoCLIP embeddings (e.g. by RSA/CKA) was found.

---

## 5. Methods to test whether injected 3D geometry is actually used

### Takeaway
The literature mostly shows "geometry helps on average" through accuracy ablations. GeoSR shows naive additive fusion is barely used. Almost no paper runs causal counterfactuals on the geometry input itself. The closest is 3D-VCD, which corrupts 3D scene-graph geometry and contrasts the outputs. Because this project injects *explicit* xyz with a known frame, it can run unusually clean counterfactuals: mirror, shuffle, rotate, scale, swap objects. These are probably the strongest causal evidence available to the thesis.

### Cited Findings
- **3D-VCD (2026).** Perturbs 3D scene-graph object geometry (centroids, extents) and compares the MLLM's answers on original vs corrupted inputs. It uses the contrast to find and suppress tokens insensitive to 3D inconsistency (visual contrastive decoding). This is the closest published "counterfactual geometry" reliance test. — [arXiv 2604.08645](https://arxiv.org/pdf/2604.08645) (snippet)
- **GeoSR.** Accuracy with no geometry vs naive additive geometry differs by +0.4 (static) and -1.2 (dynamic). This is evidence that additive geometry tokens are underused. The remedy is vision-token masking plus gating. — [arXiv 2603.26639](https://arxiv.org/html/2603.26639v1)
- **Attention knockout as a causal tool (VGGT).** QK-zeroing of selected heads, with localised variants restricted to corresponding patches, compared against random-layer controls. — [arXiv 2512.11508](https://arxiv.org/html/2512.11508v1)
- **Spatial IDs causally mediate beliefs.** Interventions on linear spatial-ID directions at intermediate layers change model beliefs. This is a template for steering along a probed "3D position" direction. — [arXiv 2601.12626](https://arxiv.org/abs/2601.12626v1)
- **Representation geometry predicts robustness.** Coh_D vs accuracy, r=0.92, on counter-heuristic splits. — [Why Far Looks Up](https://cheolhong0916.github.io/whyfarlooksup/)
- **Injection site matters.** GeoAlign: geometry into the LLM semantic stage underperforms. SpatialStack: layer-wise LLM-side injection beats vision-side concatenation. — [arXiv 2604.12630](https://arxiv.org/pdf/2604.12630), [arXiv 2603.27437](https://arxiv.org/abs/2603.27437) (snippets)

### Inferences (concrete experiment menu for the project, cheapest first; all are eval-only)
1. **Geometry knockout and noise.** Zero the geometry tokens, replace them with the dataset-mean point map, or add Gaussian noise at σ ∈ {0.5, 1, 2, 5} cell units. Report per-task VSI-Bench Δ for each arm. If accuracy is flat in the additive arm, it is not using geometry, matching GeoSR. A dose-response curve on relative direction for the cross-attention arm is causal evidence. Cost: one eval pass per condition, about the existing eval cost.
2. **Mirror counterfactual (answer-flip test).** Negate the x-coordinate in camera-0 frame (equivalently, mirror the point map left/right) while keeping RGB unchanged. For relative-direction questions ("is X to the left of Y"), the geometry-consistent answer flips. Measure the *flip rate*: the fraction of answers that flip to the mirrored-geometry answer. The control is mirroring the *image* frames and the geometry together, in which case answers should flip in both arms. Measure flips under geometry-only, RGB-only and both. This separates "uses geometry" from "uses appearance" per arm.
3. **Object-swap counterfactual.** Using SAM3 masks, swap the geometry tokens of two queried objects (their cells' xyz) and leave RGB fixed. Distance and direction answers should follow the swapped geometry. Absolute-distance answers should follow the new inter-object distance. This is the cleanest object-level causal test and reuses the existing SAM3 pipeline.
4. **Scale counterfactual.** Multiply all coordinates by k ∈ {0.5, 2}. Room size and absolute distance should scale (by k² for area, k for distance) if the model reads metric geometry. Note that the cross-attention bias clips at distance 10, so large k saturates the bias, a built-in prediction to test. Relative tasks should be invariant. Invariance in absolute tasks means geometry is ignored for measurement. That is plausible given that only relative direction improved.
5. **Rigid-transform invariance.** Apply a random rotation and translation to all point maps. A geometry-using model should be invariant on relative and room-size tasks but sensitive on egocentric direction tasks, if the question frame is tied to camera-0. 3D RoPE on (x,y,z) is not rotation-invariant, so this also tests whether the model learned frame-specific shortcuts.
6. **Temporal and frame shuffling of geometry only.** Permute the geometry tokens across frames (or cells within a frame) while keeping RGB order. This breaks the cell↔geometry correspondence and tests whether the per-cell alignment, rather than the global geometry statistics, is used.
7. **Probing before and after injection.** Q1's B1/B2 probes at each layer, for base, additive and cross-attention arms. Add a *control task* (probe on shuffled labels, Hewitt & Liang selectivity) so that high R² is not just the probe reading the raw injected xyz at layer 0. The key quantity is R² at layers well after injection (e.g. 8, 14, 20), plus the R² of the *text/answer* tokens for queried-object positions.
8. **Activation patching and steering.** Patch visual-token residuals at layer 3 from a geometry-mirrored run into a clean run, layer by layer, and measure the answer flip. This localises where geometry is consumed. Alternatively, add a probed "+x direction" vector and check that relative-direction answers move accordingly, following Kang et al.'s spatial-ID causal test. Cost: one forward pass per patched layer per example, so use about 200 relative-direction examples × 24 layers, which is minutes to an hour on one GPU.
9. **Stratify by geometry quality.** Correlate per-question gain (fused minus baseline) with VGGT point-map error against ScanNet/ARKitScenes depth. Use VGGT's own confidence map as a proxy when ground truth is missing.
10. **Bias-strength sweep (AdaptVis-style).** Scale nu_h at inference (0 removes the 3D-distance bias, leaving 3D RoPE only). Also run 3D RoPE off with the bias on, which separates the two geometric mechanisms of the cross-attention arm.

Interpretation guide:
- **Cross-attention arm** passes tests 2/3 on relative direction but fails 4: geometry is used *relationally* but not *metrically*. This is consistent with gains only on relative direction.
- **Additive arm** fails 1–3: matches GeoSR's "underuse", which motivates masking or gating.

### Gaps
- I found no published paper that mirrors or perturbs dense point maps fed to a geometry-fused VLM and measures answer flips. The protocol above is my design, adapted from 3D-VCD and standard counterfactual-intervention practice, not an established benchmark.
- No source gives compute figures for these analyses on video VLMs. The estimates above are extrapolated from the cluster's reference timing (Llama-3.1-8B, about 4.7k short texts in about 1.5 min on one B300) and are not measured.
