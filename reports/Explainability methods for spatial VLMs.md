# Geometry counterfactuals expose what fusion actually uses

The methods that fit this project best are not new saliency or attention maps. They are interventions on the injected geometry itself. Because the project feeds **explicit per-cell xyz in a known frame**, it can mirror, swap, shuffle and rescale the geometry while leaving the RGB untouched, then measure whether answers follow. No latent-feature fusion paper can run that test, and almost none has ([GeoSR](https://arxiv.org/html/2603.26639v1); [3D-VCD](https://arxiv.org/pdf/2604.08645)). Of the four families you asked about:

- **Spatial/3D foundation-model explainability** supplies those counterfactuals and coordinate probes.
- **Mechanistic VLM interpretability** supplies activation patching, which locates the geometry use at the 48 cross-attention heads. Attention knockout is less useful here, because only 6 of Qwen3.5-2B's 24 layers use softmax attention.
- **Concept-based methods** reduce in practice to linear probes plus LEACE erasure. Text-named concepts and SAEs suit the J-lens finding that geometry is non-verbal less well.
- **Temporal/video explainability** contributes mainly controls (frame and geometry shuffles, VSI-Bench-Debiased), not a primary method.

One structural fact sharpens the whole programme. **The distance bias −ν‖xᵢ − xⱼ‖ does not change under reflection**, so on its own it cannot tell left from right. A mirror counterfactual therefore separates the bias pathway from the 3D-RoPE pathway in the cross-attention arm. On relative direction the v10 cross-attention arm leads the v9 additive control by **2.8 points on seed 0 (88.2 vs 85.4 at 32 frames, n = 968) and 1.1 on seed 1 (87.6 vs 86.5)**, against a binomial standard error of about 1.1 points. Every experiment below therefore uses paired, per-item measures (flip rates, Δ log-probability) rather than accuracy differences. Five experiments fit in about two GPU-hours on the step-5750 checkpoints. The top two alone fit in one hour.

## Today's analyses show where attention goes, not whether geometry is used

The existing analyses are grad×input at the injection layers, SAM3 mask enrichment, object-to-object versus bias-only attention, Jacobian-lens readouts and the attention-swap ablation. Each one either describes the model or changes the attention pattern. **None of them changes the geometry content while holding everything else fixed.** The attention swap comes closest, but it replaces how cells are mixed, not what the cells say. Grad×input is a one-step attribution against a zero baseline, which is an out-of-distribution input for the cross-attention arm. The notes on video attribution warn that gradient methods spread credit across near-duplicate frames and should be checked against deletion ([Sufficient Frame Subsets](https://arxiv.org/html/2610.25084)). The Jacobian lens finding that the geometry delta reads as noise tokens is expected rather than damning. Even ordinary object patches decode to the right word in only **6.5% of positions on Qwen2-VL-2B** under the logit lens ([Neo et al., ICLR 2025](https://arxiv.org/html/2410.07149v2)). A contextual-embedding lens recovers far more (about **70% of visual tokens** interpretable across VLMs in [LatentLens](https://arxiv.org/html/2602.00462)). Ge**Effect size.** At step 5750, 32 frames, relative-direction accuracy over the 968 items is 85.4 (additive, `2B_v9_fmix_additive_s0`) vs 88.2 (cross-attention, `2B_v10_xa3_qknu_s0`) on seed 0, and 86.5 vs 87.6 on seed 1. The binomial standard error is about 1.1 points, so the seed-0 gap is about 2.5 SE and the seed-1 gap about 1 SE. Only 38 cross-attention wins and 16 additive wins on object questions replicate across both seeds. A between-arm accuracy comparison on one seed cannot carry a mechanistic claim, but a within-item flip rate can. (Source: `results/train/{2B_v9_fmix_additive,2B_v10_xa3_qknu}_s{0,1}/branches/step5750/eval/vsi_f32_full.jsonl`.)ce: `results/train/2B_v7_{additive,xa3_both}_s0/branches/step5750/eval/vsi_f32_full_summary.json`.)

**Architecture.** Qwen3.5-2B interleaves three linear-attention layers with one full-attention layer. **Only layers 3, 7, 11, 15, 19 and 23 are softmax attention**, and the injection layers 0–2 are all linear attention (local `config.json`, per the temporal notes). Every published knockout study uses all-softmax models ([Map the Flow](https://arxiv.org/abs/2510.13251); [Gou et al.](https://arxiv.org/html/2508.15360); [Causality Matters](https://arxiv.org/html/2508.11576)). Edge-level knockout here would therefore miss information that moves through the recurrent state. The project's own fusion module is the one place where interventions are exactly defined.

**Benchmark bias.** The Thinking-in-Space paper reports that relative direction, route planning and absolute distance gain little from vision over blind models ([Yang et al.](https://arxiv.org/html/2412.14171v2)). Text-only fine-tuning lifted a blind LLaVA-Video-7B from 25.9% to 44.7% on VSI-Bench, which motivated a debiased split ([arXiv 2511.04655](https://arxiv.org/html/2511.04655v1)). The project's scores are far above those, but any explanation of a 1–2 point gain should also be reported on the `debiased` config.

## Spatial and 3D explainability supplies the decisive counterfactuals

The geometry-fusion literature mostly shows that geometry helps on average, through accuracy ablations. The closest analogue to the additive arm is GeoSR's naive fusion F = F_V + MLP(F_G) on Qwen2.5-VL-7B with VGGT. It gained **+0.4 on static scenes and lost 1.2 on dynamic ones** relative to no geometry. The authors had to mask 80% of vision tokens in training to force the model to rely on geometry ([GeoSR](https://arxiv.org/html/2603.26639v1)). Their "underuse" diagnosis rests on accuracy alone. The only published test that perturbs geometry inputs to check reliance is 3D-VCD, which corrupts scene-graph centroids and extents and contrasts the outputs ([arXiv 2604.08645](https://arxiv.org/pdf/2604.08645), seen as a snippet only). Explicit point maps allow a cleaner family of interventions, listed below.

| Intervention | What changes | Prediction if geometry is used | What it adds beyond the current analyses |
|---|---|---|---|
| **Mirror** (reflect across a vertical plane) | The scene's handedness | Left/right answers flip; front/back answers stay | Separates the RoPE and bias pathways, because ‖xᵢ − xⱼ‖ is reflection-invariant |
| **Cross-scene swap** | All geometry content | Accuracy falls to the RGB-only level | Shows whether the gain needs this scene's geometry or only geometry-shaped inputs |
| **Object swap** (exchange the xyz of two queried objects, using SAM3 masks) | One relation | Direction and distance answers follow the swap | An object-level causal test that reuses the existing pipeline |
| **Scale ×k** | Metric size | Absolute distance and room size scale by k (or k²); relative tasks do not change | Tests metric use. The bias is clipped at d = 10, so large k saturates it, which is itself a prediction to test |
| **Rigid rotation** | The world frame | Relative tasks do not change | Shows whether 3D RoPE learned frame-specific shortcuts |

The reflection argument deserves emphasis because it is a property of the design, not an empirical result. A reflection preserves every pairwise distance, so **the bias-only attention pattern is identical in the real and mirrored scenes**. Yet the egocentric answer to "standing at A, facing B, is C to my left?" swaps under any reflection across a vertical plane. If the cross-attention arm's answers flip when only the geometry is mirrored, the information must arrive through 3D RoPE on (t, x, y, z) or through geometry in keys and values, not through the bias. If they do not flip, the relative-direction gain more plausibly comes from better grouping of co-located cells, which then lets the RGB carry direction. The existing `xa3_bias`, `xa3_rope` and `xa3_both` training runs make this a natural three-way comparison. In the additive arm, mirroring changes the injected vector directly, so it should flip only if the arm reads geometry at all.

The probing literature motivates checking distance and depth specifically. "Why Far Looks Up" finds distance is **the weakest-organised axis in every VLM tested**. Its coherence score is 0.182 for distance against 0.649 for horizontal in RoboRefer-2B, and "far" and "above" share a direction in Qwen2.5-VL-3B. Coherence correlates with benchmark accuracy at r = 0.92 across 8 VLMs ([arXiv 2605.30161](https://arxiv.org/pdf/2605.30161)). Gurnee-style ridge probes are the cheap, standard tool for asking whether coordinates are linearly present ([Gurnee & Tegmark](https://arxiv.org/abs/2310.02207)). The project can ask a question nobody has published: does injected 3D geometry decouple depth from image height, and does it survive to the layers where answers form?

Inference-time dose-response is the other high-value import. AdaptVis rescales attention temperature on image tokens and reports up to +50 points on WhatsUp ([Chen et al., ICML 2025](https://arxiv.org/html/2503.01773v2)). Applied here, scaling ν_h by {0, 0.5, 2}, disabling 3D RoPE, or scaling the additive injection α produces a dose-response curve. A monotone relative-direction curve with flat other tasks would be causal evidence that the existing on/off attention swap cannot give.

VGGT's own interpretability matters less, because the project uses the point-head output rather than the trunk. It does flag two risks.

- **Point-map quality varies.** Correspondence heads in layers 10–16 drive VGGT's geometry, and they "hallucinate" matches on occluded patches ([Bratulić et al.](https://arxiv.org/html/2512.11508v1)). Stratifying per-item gains by VGGT confidence is a cheap check that the gains track geometry quality.
- **Explicit point maps may be the weaker signal.** VG-LLM and GeoAlign report that latent VGGT features beat explicit point maps, with GeoAlign favouring layer 20 for relative direction ([VG-LLM](https://arxiv.org/html/2505.24625v3); [GeoAlign](https://arxiv.org/pdf/2604.12630); both seen as snippets). That is a modelling ablation, not an explainability one, and it trades away the clean counterfactuals above.

Geospatial work (SatCLIP and GeoCLIP probes, Earth-embedding SAEs) adds two transferable habits and little else: ridge probes with held-out regions, which here means **held-out scenes**, and SAE decomposition of location embeddings ([arXiv 2606.24997](https://arxiv.org/pdf/2606.24997); [arXiv 2606.07172](https://arxiv.org/html/2606.07172v1)).

## Mechanistic patching locates geometry use at the 48 fusion heads; knockout reaches only six layers

The VLM information-flow literature agrees on a staged picture:

1. Image information flows into question tokens in early-to-middle layers.
2. The question representation flows to the last position in upper-middle layers.
3. A narrow middle band, about 25% of layers, carries most of the cross-modal transfer.

([Zhang et al., CVPR 2025](https://arxiv.org/pdf/2411.18620); [Kaduri et al., CVPR 2025](https://arxiv.org/html/2411.17491v1).) Video LLMs follow the same pattern. Map the Flow keeps 42% of attention edges and loses almost nothing on TVBench (51.5 → 51.2). Blocking cross-frame attention in layers 1–16 costs **44.8 points on Moving Direction**, the closest analogue to relative direction ([Map the Flow](https://arxiv.org/html/2510.13251v2)). In Qwen2.5-VL-7B, the final token reads video almost entirely through the query tokens. Blocking final→video has minimal effect, while blocking final→query costs more than 40% ([Causality Matters](https://arxiv.org/html/2508.11576)).

These results make a cheap prediction for this project. The Jacobian lens has answer letters emerging around layer 17, so geometry should pass into question tokens at full-attention layers 7–15. Testing that requires FlexAttention masks at 9.7k tokens. Blocking softmax edges alone also understates flow through the Gated-DeltaNet state, and no reference exists for state-level knockout. Knockout is therefore a second-tier experiment here.

Activation patching avoids both problems by intervening at the project's own module. The NOTICE line of work established that semantic minimal pairs beat Gaussian-noise corruption ([Golovanevsky et al., NAACL 2025](https://arxiv.org/html/2406.16320v3)). Geometry-only counterfactuals are exactly such minimal pairs. With 3 fusion layers × 16 heads, **exact per-head patching costs only 48 forward passes per item pair**. Patch mirrored-geometry cross-attention head outputs into the clean run one head at a time and measure how much of the left-versus-right logit difference moves. The payoff is a head-level causal map. Do the heads with the largest ν_h, which are the most distance-sensitive, carry the effect, or do the heads where RoPE dominates? The current object-to-object attention analysis describes where those heads look but not whether looking matters.

For position-level questions, such as which of the 9,600 cells or 32 frames matter, attribution patching gives (clean − corrupt) · gradient for every position in one forward pass on each input plus one backward. That is roughly the cost of the existing grad×input, but against a meaningful counterfactual baseline instead of zero. Validate the top hits with a few exact patches, because first-order attribution patching produces false negatives on saturated attention ([AtP\*](https://arxiv.org/abs/2403.00745); [EAP-IG](https://arxiv.org/abs/2403.17806)). "Same Task, Different Circuits" used attribution patching with integrated gradients (AP-IG) in VLMs and found only 18% component overlap between visual and textual versions of a task ([arXiv 2506.09047](https://arxiv.org/html/2506.09047v3)). The same overlap metric could compare which downstream heads the two arms recruit as "geometry readers".

Token pruning offers a behavioural cross-check. FastV removes 50% of image tokens after layer 2 in LLaVA-1.5-13B without loss ([FastV](https://arxiv.org/html/2403.06764v3)). Accuracy-versus-k curves per arm, with SAM3 enrichment of the retained tokens, reuse existing infrastructure.

Three tools are out of reach for this budget:

- **Transcoder circuit tracing.** The CVPR 2026 Gemma-3-4B study requires training transcoders first ([arXiv 2602.20330](https://arxiv.org/html/2602.20330)).
- **Full edge attribution patching.** EAP over all edges at 10k tokens is memory-heavy.
- **Hybrid-layer support in libraries.** No library handles Qwen3.5's hybrid layers out of the box. Spectra supports Qwen2.5-VL and Qwen3-VL and offers the closest naming conventions, but plain PyTorch hooks remain the practical route ([Spectra, ACL 2026 demo](https://aclanthology.org/2026.acl-demo.78.pdf)).

## Concept methods reduce to probes plus erasure

The TCAV lineage asks whether an answer is sensitive to a learned concept direction. For this project the useful parts are the supervised direction and a causal test on it. The classic image-classifier tools transfer poorly:

- **CRAFT** factorises activations with NMF, which needs non-negative activations, while residual streams are signed ([CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Fel_CRAFT_Concept_Recursive_Activation_FacTorization_for_Explainability_CVPR_2023_paper.html)).
- **CRP** needs LRP rules for RMSNorm and attention ([Nature MI 2023](https://www.nature.com/articles/s42256-023-00711-8.pdf)).
- **Concept bottleneck models**, including the label-free variant, require retraining and answer a different question, and they are prone to leakage ([Label-Free CBM](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/ICLR/2023/labelfree_concept_bottleneck_models__81d641a6.txt); [leakage study](https://arxiv.org/html/2504.14094v1)).

The 2026 spatial-mechanism papers point directly at a better design. Kang et al. find that VLMs bind object locations to **linear "spatial IDs" on object-word tokens** in a narrow band of intermediate layers. Adding adversarial IDs flips left/right and far/near beliefs 64.6% of the time, against 29.5% for norm-matched noise, and video VLMs carry analogous temporal IDs ([OpenReview](https://openreview.net/forum?id=2zXRGiorSu); [arXiv 2601.12626](https://arxiv.org/abs/2601.12626v1)). "Decodable Is Not Grounded" warns that depth is often decodable at 73–97% yet applied with the wrong sign. It recommends a blank-image control as the default arbiter ([arXiv 2606.31257](https://arxiv.org/pdf/2606.31257), abstract only).

Together these suggest a three-part concept test:

1. **Probe for the relation.** Probe the pair-level relation (left/right of C in the A→B frame) at object-mention and final tokens, across the six full-attention layers, for the base model and both arms.
2. **Add the controls.** Use cross-scene geometry as the analogue of a blank image, and use scene-held-out splits with Hewitt-Liang control tasks.
3. **Erase the subspace.** Remove the probed subspace with **LEACE**, which provably blinds every linear classifier with minimal distortion ([Belrose et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract.html)). Compare the accuracy drop against erasing a random subspace of the same rank.

A per-cell xyz probe at layer 3 is a trivial positive, since the geometry was just injected. The informative number is R² at layers 11–19, compared with the base model.

Sparse autoencoders and crosscoders are a discovery tool, not the test. Crosscoders and diff-SAEs are designed for model diffing, which fits base vs fused or additive vs cross-attention. They come with known artefacts that BatchTopK training and Latent Scaling correct ([Minder et al.](https://arxiv.org/pdf/2504.02922v3)). However, SAE probes **lose to plain logistic regression on average** ([Kantamneni et al.](https://arxiv.org/pdf/2502.16681.pdf)). No pretrained SAEs exist for Qwen3.5, and Gemma Scope-scale training is far beyond budget ([arXiv 2408.05147](https://arxiv.org/pdf/2408.05147v2.pdf)). If attempted later, name latents by correlating them with geometric variables (depth, height, distance to the queried object, quadrant sign) rather than through text auto-interpretation, which the J-lens result predicts will fail.

## Temporal methods mainly supply controls for the 32-frame input

Frame-level attribution is mature for classifiers and untested on video LLMs. Sufficient Frame Subsets find that **2.8–3.4 of 16 frames** suffice for transformer video classifiers, and that they beat integrated gradients on insertion AUC (0.783 vs 0.591 on V-JEPA 2) ([arXiv 2610.25084](https://arxiv.org/html/2610.25084)). Frame-level Shapley values are principled, but permutation sampling needs about 1,000 passes per question at 32 frames ([Play Fair](https://arxiv.org/pdf/2011.12372)). Both would answer "how many frames does relative direction need, and does the cross-attention arm draw on frames where the objects are not visible?" Both cost 32–1,000 passes per item and so fall outside a 2-hour budget except on a few dozen items.

The cheap and decisive temporal tools are the controls. Benchmarks are often solvable from one frame or from shuffled frames ([TVBench](https://arxiv.org/html/2410.07752v2)). Spatio-temporal reasoning in video LLMs relies more on language-guided retrieval than on video self-attention ([Gou et al.](https://arxiv.org/html/2508.15360)).

For this project, the most useful control is **permuting the geometry across frames while keeping RGB order**. This breaks the cell-to-geometry correspondence but keeps the global geometry statistics. It answers whether the per-cell alignment is used, which neither the attention swap nor the cross-scene swap isolates.

A second, novel measurement uses SAM3 instance tracks. Compute the attention mass that the cross-attention heads place on **the same object instance in other frames**, minus what bias-only attention would assign. This extends the existing object-to-object analysis along time. It would be a first measurement of cross-frame instance linking in a geometry-fused VLM: current VLMs are reported to be weak at it ([SpatialMosaic](https://arxiv.org/html/2512.23365v4)), and DUSt3R-style models have dedicated correspondence heads ([arXiv 2510.24907](https://arxiv.org/html/2510.24907)). The 3D analogue of Causality Matters' "corresponding area" knockout, which restricts cross-frame attention to cells within r metres, is elegant. It needs the FlexAttention and linear-layer engineering discussed above, so it belongs to a later phase.

## Five experiments fit in two GPU-hours on seed 0

The cost basis is measured locally:

- A full zero-shot 32-frame VSI-Bench eval of Qwen3.5-2B (5,130 items) took **24 minutes on a B300** (job 202347), about 0.28 s per item.
- The 128-frame fusion evals took 3.0 h for the additive arm and 4.5 h for the cross-attention arm (jobs 214485 and 214487). That suggests the cross-attention arm runs about 1.5× slower.

One pass over the 968 relative-direction items at 32 frames should therefore take about **5 minutes (additive) to 8 minutes (cross-attention)**. These are extrapolations, so run a 50-item smoke test first. Score by option-letter log-probability rather than by generation. That gives a continuous, paired Δ log p per item and avoids decoding.

| Rank | Experiment | Arms and conditions | Est. GPU time | What it settles that current analyses cannot |
|---|---|---|---|---|
| 1 | **Geometry counterfactual battery** | Both arms on 968 relative-direction items: mirror, cross-scene swap; plus scale ×2 on 834 absolute-distance items | ~40 min | Whether each arm reads geometry content. The mirror flip rate separates "uses geometry" from "uses appearance" |
| 2 | **Pathway dose-response** | Cross-attention: ν×0, ν×2, 3D RoPE off, and RoPE off + mirror. Additive: α×0, α×2 | ~35 min | Whether the relative-direction gain comes from the chirality-blind bias or from RoPE, as a curve rather than a single swap |
| 3 | **Head-level geometry patching** | 150 left/right items; patch mirrored → clean per fusion head (48) and per layer (3 per arm) | ~25–40 min | Which heads causally carry geometry, and whether ν_h predicts their effect |
| 4 | **Relational probes + LEACE** | Cache layers 3/7/11/15/19/23 for base and both arms on ~400 items (scene split); one LEACE eval per arm | ~20 min (+ CPU) | Whether geometry survives as a relational concept to the answer-forming layers, and whether the answer depends on it |
| 5 | **Geometry-frame permutation + instance-linking attention** | Both arms: geometry permuted across frames; cross-attention arm: same-instance cross-frame attention mass from SAM3 tracks | ~15–20 min | Whether the per-cell alignment is used and whether the cross-attention arm links objects across frames beyond the distance prior |

How to read the results:

- **The cross-attention arm flips on mirror, loses most of its gain when RoPE is off, and stays insensitive to scale.** That would say the arm uses geometry relationally and directionally through RoPE but not metrically. This fits a gain confined to relative direction.
- **The additive arm is flat under every counterfactual.** That would reproduce GeoSR's "underuse" with internal evidence that GeoSR lacked.
- **Neither arm flips on mirror.** Then the relative-direction gain is better explained by grouping or optimisation effects than by geometric reading. That is still a publishable negative, given a gap of 1–3 points that is only partly seed-robust.

Experiments 1 and 2 together fit the one-hour floor. Report every result on VSI-Bench-Debiased as well as on the full split. Repeat the winning conditions on seed 1 before writing them up.

Deferred beyond this budget, roughly in order of value:

- windowed attention knockout at the six softmax layers, with FlexAttention;
- attribution patching over all 9,600 positions, validated by exact patches;
- spatial-ID steering;
- Sufficient Frame Subsets on about 100 items;
- a diff-SAE at layer 3.

Not recommended: CBMs, CRAFT, CRP, frame Shapley and transcoder circuit tracing.

## Conclusion

The literature review changes the frame of the question. The real choice is not which explainability family to adopt. It is that **explicit geometry turns explanation into experiment**: every interesting hypothesis about the two arms becomes a counterfactual input with a predicted answer. The reflection-invariance of the distance bias is the sharpest such hypothesis. It predicts that the component most associated with "3D-aligned" attention cannot by itself produce the one gain the arm shows, so the gain is either carried by 3D RoPE or not geometric at all. Both outcomes are thesis-worthy, and the first two experiments decide between them in about an hour.

The second implication concerns the hybrid backbone. Qwen3.5's linear-attention layers make the standard knockout toolkit incomplete. Interventions defined at the project's own fusion module (patching, dose-response, counterfactual geometry) are therefore not just cheaper but more rigorous than borrowing the published video-LLM protocols. Those protocols assume softmax attention everywhere.
