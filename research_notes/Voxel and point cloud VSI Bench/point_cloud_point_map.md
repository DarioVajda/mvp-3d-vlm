# Point clouds, point maps and 3D-foundation-model features fused into VLMs, evaluated on VSI-Bench (2026 focus, late 2025 as context)

Compiled 2026-11-14. Numbers come from the arXiv HTML full text through a fetch-and-summarise tool, unless marked **[snippet-only]** (search-engine summary, not checked against the paper). Per-task columns use the standard VSI-Bench order: Obj Count / Abs Dist / Obj Size / Room Size / Rel Dist / Rel Dir / Route Plan / Appr Order. Some baseline numbers are re-quoted from a later paper's table, and the source is given each time.

User's reference point: Qwen3.5-2B with VGGT point-head point maps at decoder layers 0–2, additive vs 3D-aligned cross-attention (distance bias plus 3D RoPE on (t,x,y,z)). At 32 frames it gets about 70–71 overall and 86–88 Rel Dir. The two arms are tied.

## Q1. Which papers feed explicit point clouds vs pixel-aligned point maps vs latent VGGT/CUT3R/π3 features into a VLM and evaluate on VSI-Bench?

### Takeaway
No paper I found evaluates on VSI-Bench with a *point-cloud encoder* (PointNet, Point-BERT, Sonata, PTv3). That family (PointLLM, SpatialLM, Pts3D-LLM) is evaluated on 3D-LLM and scene-modelling benchmarks instead. Every VSI-Bench method in this line takes geometry from a feed-forward 3D foundation model (VGGT, π3, CUT3R, Point3R, AnySplat). The 2026 papers split into three groups:
- **Latent features only:** GeoAlign, GeoWeaver, SpaceMind, VG-LLM, Spatial-MLLM.
- **Latent features plus pixel-aligned point maps:** VLM², SpaceMind++, Cog3DMap, VLM-IE3D.
- **Point-map coordinates used to build a 3D voxel or point memory:** SpaceMind++, Cog3DMap.

The 2026 state of the art is about 69–75 avg, at 4B–10B parameters.

### Cited findings

#### Latent 3D-foundation-model features

**GeoAlign** (arXiv 2604.12630, Apr 2026)
- Base: Qwen2.5-VL-3B, branded "GeoAlign-4B".
- 3D source: latent VGGT features from the second half of the layers, chosen per patch by a top-K=2 softmax router.
- Fusion: projected and added residually to the visual tokens before the LLM (Q̂ = Q + W_out F̂).
- Training: about 460K samples, 1 epoch, dataset not named. Frame count not stated.
- No point maps are used.
- VSI-Bench: 71.4 avg; 71.2 / 59.8 / 74.1 / 75.0 / 72.0 / **87.1 Rel Dir** / 50.5 / 81.7.
- [GeoAlign](https://arxiv.org/html/2604.12630)

**GeoWeaver** (arXiv 2605.22558, May 2026)
- Base: **Qwen3.5**, so the closest backbone to the user's. Variants are GeoWeaver-5B (full fine-tune) and GeoWeaver-10B (LoRA). The text also says "4B- and 9B-scale backbones", which is inconsistent with the names.
- 3D source: frozen VGGT, layers 12–23, patch tokens only.
- Fusion: additive residual v' = v + W_o g with zero-init W_o, plus a per-token top-k=2 layer router.
- The Limitations section says point maps, depth and camera poses are *not* modelled.
- Training data not specified. The implementation table says 8 frames. ReVSI is evaluated at 16/32/64/all frames, and 32 is best.
- VSI-Bench, 5B: **72.8** avg; Rel Dist 72.8, Rel Dir 83.5, Route 56.3, Appr 84.7.
- VSI-Bench, 10B: **75.05** avg; Rel Dist 73.5, Rel Dir 92.0, Route 55.0, Appr 84.3.
- [GeoWeaver](https://arxiv.org/html/2605.22558v1)

**SpaceMind** (arXiv 2511.23075, Nov 2025, context)
- Base: InternVL3-8B.
- 3D source: spatial and camera tokens from the VGGT aggregator.
- Fusion: "Camera-Guided Modality Fusion", where visual queries cross-attend to VGGT spatial keys and values. A camera-conditioned bias is added to K/V, then a per-token weight, a camera-conditioned SwiGLU gate and a residual add.
- No 3D positional encoding is described.
- Training: VLM-3R-data, ViCA-322K and SQA3D. **32 frames.**
- VSI-Bench: 69.6 avg; Rel Dist 67.2, Rel Dir 88.4, Route 44.3.
- SpaceMind++ re-lists SpaceMind as 70.2 avg (Rel Dist 67.7, Rel Dir 88.6, Route 46.9). The two figures differ.
- [SpaceMind](https://arxiv.org/html/2511.23075v1); [SpaceMind++ table](https://arxiv.org/html/2605.09449v1)

**VG-LLM** (NeurIPS 2025, context)
- Base: Qwen2.5-VL. VGGT features are fused with the 2D features at patch level.
- Its own abstract reports 4B = 46.1 avg.
- Later tables list 4B = 47.3 and 8B = 50.7:
  - 4B: 66.0 / 37.8 / 55.2 / 59.2 / 44.6 / 45.6 / 33.5 / 36.4
  - 8B: 67.9 / 37.7 / 58.6 / 62.0 / 46.6 / 40.7 / 32.4 / 59.2
- The authors argue that VGGT-predicted point maps are "relatively sparse and may contain errors" and use the latent features instead. **[snippet-only for that quote]**
- [VG-LLM GitHub](https://github.com/LaVi-Lab/VG-LLM); [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/1d8dc55c1f6cf124af840ce1d92d1896-Paper-Conference.pdf); [Cog3DMap table](https://arxiv.org/html/2603.23023v1)

**Spatial-MLLM-4B** (2025, context)
- VGGT-based dual encoder.
- VSI-Bench: 48.4 avg; 65.3 / 34.8 / 63.1 / 45.1 / 41.3 / 46.2 / 33.5 / 46.3, as re-quoted in [GeoAlign](https://arxiv.org/html/2604.12630) and [Cog3DMap](https://arxiv.org/html/2603.23023v1).

**VLM-3R-7B** (2025, context)
- CUT3R geometry tokens plus camera tokens.
- VSI-Bench: 60.9 avg; 70.2 / 49.4 / 69.2 / 67.1 / 65.4 / 80.5 / 45.4 / 40.1.
- [VLM-3R](https://arxiv.org/pdf/2505.20279); numbers as quoted in [GeoAlign](https://arxiv.org/html/2604.12630)

**GeoSR** ("Make Geometry Matter for Spatial Reasoning", ECCV 2026 Oral)
- Geometry-Unleashing Masking of the 2D tokens, plus gated Geometry-Guided Fusion.
- VSI-Bench: 51.9 avg. **[snippet-only]**
- [GeoSR GitHub](https://github.com/SuhZhang/GeoSR)

#### Pixel-aligned point maps used explicitly (with or without latent features)

**VLM²** (arXiv 2511.20644; v2 Jul 2026; ECCV 2026)
- Base: LLaVA-Video-7B (Qwen2-7B).
- 3D source: π3, which supplies geometry tokens, view tokens and **per-pixel point maps**.
- Point-map injection ("A3PI"): point maps are pooled per patch to coordinates C_t, passed through a sinusoidal encoding and a 2-layer MLP, then added through a learnable per-patch gate: F' = F + α ⊙ φ(C), α ∈ [0,1].
- The position-aware visual tokens then cross-attend to π3 geometry tokens.
- Memory is dual: a sliding-window working memory plus an episodic memory.
- Frame count not visible.
- VSI-Bench, 7B: **68.8** avg; Rel Dist 69.0, **Rel Dir 87.8**, Route 52.6, Appr 68.3.
- [VLM²](https://arxiv.org/html/2511.20644v2)

**SpaceMind++** (arXiv 2605.09449, May 2026)
- Base: InternVL3-8B, with the ViT and VGGT frozen and LoRA on the LLM.
- 3D source: VGGT latent features **plus per-pixel 3D point maps plus confidence**.
- Point maps are average-pooled to visual-token resolution. High-confidence points are recentred and voxelised (D=100, at most 5,000 voxels), and each voxel becomes one "map token".
- Fusion (CDIF): alternates map self-attention with map→visual cross-attention. Coordinates go through an MLP and are added to the features, **and 3D RoPE on the coordinates is applied to Q and K in both attention steps**. A sigmoid gate is followed by a residual update of the visual tokens.
- Training: SpaceMind-900K (ViCA-322K, VLM-3R-data, SQA3D and VSI-590K-Video). Frame count not stated.
- VSI-Bench, 8B: **73.2** avg; 74.5 / 62.4 / 77.9 / 76.9 / **73.5** / **89.7** / 48.2 / 84.1.
- [SpaceMind++](https://arxiv.org/html/2605.09449v1)

**Cog3DMap** (arXiv 2603.23023, Mar 2026)
- Base: Qwen3-VL-8B.
- 3D source: a frozen Point3R recurrent point-map estimator, so the geometry is predicted.
- Per patch, the Qwen semantic features and the 3D coordinates are average-pooled. Memory tokens are v = f + Prj(g), where g is the geometric feature.
- A recurrent 3D memory merges tokens within a distance threshold δ (dynamic). Up to 8K tokens are kept in training; all are kept at evaluation.
- Training: VLM-3R data plus a 73K SPAR-7M subset, about 40 A100-hours. **32 frames** for training and evaluation.
- VSI-Bench, 8B: **65.1** avg; 69.6 / 54.8 / 67.8 / 67.1 / 64.8 / **85.6** / 43.0 / 67.9.
- [Cog3DMap](https://arxiv.org/html/2603.23023v1)

**VLM-IE3D** ("3D-Aware VLMs with Implicit and Explicit Geometries", arXiv 2607.21595, Jul 2026, ECCV 2026)
- Base: Qwen2.5-VL-3B, about 4B in total.
- 3D source: AnySplat by default, with VGGT and π3 ablated. Implicit Geometry Tokens (IGTs) come from the fusion-decoder features. Explicit Geometry Tokens (EGTs) come from depth by default, with point maps (c'=3) and 3D Gaussians as alternatives.
- Fusion: IGTs (queries) cross-attend to EGTs (keys and values), and the result is **added element-wise** to the 2D tokens. No positional encoding is described.
- Training: 234K from SPAR-7M plus 63K from LLaVA-Hound. Up to **32 frames**.
- VSI-Bench, 4B: 47.6 avg; Rel Dir 45.1, Rel Dist 47.7, Route 36.6.
- [VLM-IE3D](https://arxiv.org/html/2607.21595v1)

#### Explicit point-cloud encoders (none on VSI-Bench)
- Pts3D-LLM (Apple, 2506.05689) adds Sonata-PTv3 features to SigLIP tokens. SpatialLM (2506.07491) uses a Sonata encoder with Qwen2.5-0.5B. PointLLM uses Point-BERT/ULIP-2. In the search results, none of them reports VSI-Bench. — [Pts3D-LLM](https://arxiv.org/pdf/2506.05689); [SpatialLM](https://arxiv.org/html/2506.07491v2); [PointLLM](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03601.pdf)

#### Other 2026 candidates seen but not read
These were not verified (titles and snippets only):
- ViPS / "Beyond Single Expert" (2607.15054): VGGT, Depth Anything 3 and RADIO priors on Qwen2-VL-7B and Qwen3-VL-8B. It compares against Vega-3D. — [arXiv](https://arxiv.org/pdf/2607.15054)
- "Learning Geometric Representations from Videos for Spatial Intelligent MLLMs" (2606.05833) — [arXiv](https://arxiv.org/pdf/2606.05833)
- Q-GeoMem (2605.27318) — [arXiv](https://arxiv.org/pdf/2605.27318)
- Map2Thought (2601.11442) — [arXiv](https://arxiv.org/pdf/2601.11442)
- ViSRA (2605.10106): an agent with VGGT-based 3D detection — [arXiv](https://arxiv.org/pdf/2605.10106)
- OneCanvas (2606.19253) — [arXiv](https://arxiv.org/html/2606.19253v2)
- "Projector Is All You Train" (2608.19726) and "Self-supervised point cloud encoders for 3D LLMs" (2607.29136) — [arXiv](https://arxiv.org/pdf/2608.19726), [arXiv](https://arxiv.org/pdf/2607.29136)

#### Reference baselines, as quoted in the papers above
- Qwen3-VL-2B: 50.4 avg (Rel Dist 52.3, Rel Dir 42.0, Route 30.4) — [VLM²](https://arxiv.org/html/2511.20644v2)
- Qwen3-VL-8B: 57.9 avg — [VLM²](https://arxiv.org/html/2511.20644v2)
- Cambrian-S-3B: 57.3 avg (Rel Dir 61.9) — [GeoAlign](https://arxiv.org/html/2604.12630)
- SenseNova-SI-1.2: 69.6 avg — [SpaceMind++](https://arxiv.org/html/2605.09449v1)
- VST-7B: 60.6 or 61.2, depending on which paper's table — [SpaceMind++](https://arxiv.org/html/2605.09449v1), [Cog3DMap](https://arxiv.org/html/2603.23023v1)

### Inferences
- The user's 2B model at about 70–71 avg and 86–88 Rel Dir is in the same band as 4B–8B 2026 systems: GeoAlign-4B 71.4 / 87.1, VLM²-7B 68.8 / 87.8, SpaceMind-8B 69.6 / 88.4, SpaceMind++-8B 73.2 / 89.7. It sits only below GeoWeaver (Qwen3.5-5B 72.8, 10B 75.05). I found no other ≤3B model in this range.
- Rel Dir in the high 80s now appears to be the norm for geometry-fused models trained on VLM-3R-style data. Route Plan stays low (43–56) for every method, as it is likely to for the user's model too.
- Comparisons across papers are confounded by different training corpora (VLM-3R data, ViCA, SPAR, VSI-590K) and different frame counts. Most papers do not state the frame count.

### Gaps
- Frame count is unreported for GeoAlign, VLM², SpaceMind++ and VG-LLM, so the user's 32-frame result can only be matched exactly against SpaceMind and Cog3DMap.
- I could not confirm whether any paper with a point-cloud encoder (Sonata/PTv3) reports VSI-Bench.
- ViPS, Vega-3D, Q-GeoMem, Map2Thought and 2606.05833 were not read, so their designs and numbers are unknown.

## Q2. Has anyone used 3D-position-aware attention (distance bias or 3D RoPE on reconstructed xyz), and are there additive vs cross-attention ablations?

### Takeaway
**SpaceMind++** is the closest published match to the user's cross-attention arm: 3D RoPE on VGGT point-map coordinates inside map↔visual cross-attention, 73.2 avg at 8B. However, it voxelises tokens and gives no additive-vs-cross-attention ablation. I found no paper with a *distance bias* (an ALiBi-style -ν·dist term) on reconstructed xyz.

Head-to-head fusion ablations do exist, and they mostly favour simple additive injection or gated cross-attention over concatenation:
- GeoAlign: Shared+Add is best.
- GeoWeaver: the additive design ablates *where* the injection happens.
- VLM²: cross-attention far exceeds concat-MLP.
- Cog3DMap: adding RoPE or learned PE on xyz on top of the additive projection *hurts* slightly.

### Cited findings
- **SpaceMind++** applies 3D RoPE to Q and K in both the map self-attention and the map→visual cross-attention, using coordinates from VGGT point maps. Coordinates are also MLP-encoded and added. Its ablation covers spatial-encoder baseline, DIF, CogMap+DIF and full model, without an additive-vs-xattn comparison. The final model also changes data (900K), which confounds the comparison. — [SpaceMind++](https://arxiv.org/html/2605.09449v1)
- **Cog3DMap positional-encoding ablation** (Scan2Cap, not VSI-Bench; CIDEr@0.5):

  | Variant | CIDEr@0.5 |
  |---|---|
  | Qwen3-VL-8B baseline | 65.2 |
  | f + Learnable-PE(p) | 66.7 |
  | f + 4D-RoPE(p) | 64.0 |
  | f + HRoPE(p) | 71.7 |
  | f + Prj(g) (additive geometry features, default) | **79.4** |
  | f + Prj(g) + Learnable-PE(p) | 78.0 |
  | f + Prj(g) + 4D-RoPE(p) | 77.9 |
  | f + Prj(g) + HRoPE(p) | 76.6 |

  So once latent geometry features are added, RoPE or PE on xyz does not help. The paper is also internally inconsistent on whether point tokens use 1D-RoPE or 3D-RoPE. — [Cog3DMap](https://arxiv.org/html/2603.23023v1)
- **GeoAlign fusion ablation** (VSI-Bench avg, 4B): Split-Proj 69.4, FiLM 70.1, Gated(2D) 70.7, Gated(2D+3D) 70.2, **Shared+Add 71.4**. Cross-attention was not tested. — [GeoAlign](https://arxiv.org/html/2604.12630)
- **GeoWeaver placement ablation** (VSI-Bench avg, 5B, additive): input-level 68.10, decoder-side 70.45, **pre-reasoning grounding 72.80**. Layer allocation: uniform 70.35, global weights 71.28, token-adaptive 72.80. — [GeoWeaver](https://arxiv.org/html/2605.22558v1)
- **SpaceMind ablation** (VSI-Bench avg, InternVL3-8B):

  | Configuration | VSI-Bench avg |
  |---|---|
  | Fine-tuned baseline | 63.07 |
  | + VGGT shallow cross-attention | 66.77 |
  | + token-weight MLP | 67.17 |
  | + geometric MLP | 68.73 |
  | Full CGMF | 69.58 |

  — [SpaceMind](https://arxiv.org/html/2511.23075v1)
- **VLM² Table 7** (VSI-Bench avg; base LLaVA-NeXT-Video-SFT 55.2):

  | Configuration | VSI-Bench avg |
  |---|---|
  | Gated point-map injection (A3PI) | 61.6 |
  | Ungated injection | 58.9 |
  | Cross-attention fusion, VGGT | 60.2 |
  | Cross-attention fusion, CUT3R | 59.4 |
  | Cross-attention fusion, π3 | 61.0 |
  | Concat-MLP fusion (range) | 40.4–43.3 |

  — [VLM²](https://arxiv.org/html/2511.20644v2)

### Inferences
- The user's finding that additive and 3D-aligned cross-attention tie overall is consistent with the literature:
  - GeoAlign's best fusion is a shared projection plus add.
  - GeoWeaver reaches 72.8–75 with purely additive injection.
  - Cog3DMap finds that xyz positional encodings on top of additive geometry give no gain.
- The user's distance-bias plus (t,x,y,z) RoPE cross-attention is therefore a novel ablation axis: no published paper isolates it at matched data.
- SpaceMind++ is the paper to cite as prior art for 3D RoPE on reconstructed coordinates.

### Gaps
- I found no published ALiBi-style metric-distance attention bias over reconstructed points in a VSI-Bench VLM.
- No paper reports an additive-vs-cross-attention comparison at matched data, model and frames on VSI-Bench per-task numbers.

## Q3. What do papers report about explicit point maps vs latent 3D features?

### Takeaway
Most evidence says latent 3D-foundation-model features carry more useful signal than raw point maps alone, and that point maps help mainly as a complement (gated or positional), not as a replacement:
- VG-LLM's motivating argument.
- VLM-IE3D: IGT > EGT, and both together are best.
- Cog3DMap: Prj(g) >> PE(xyz).
- VLM²: gated point-map injection adds about 2.7 avg.

No paper shows point maps alone matching latent features on VSI-Bench.

### Cited findings
- VG-LLM argues that VGGT point maps help Qwen2.5-VL but are "relatively sparse and may contain errors", and prefers the latent features. **[snippet-only]** — [VG-LLM](https://www.alphaxiv.org/abs/2505.24625v1)
- **VLM-IE3D** (3D video detection F1@25, not VSI-Bench): Qwen2.5-VL-3B baseline 30.9; + EGTs only 34.7; + IGTs only 40.5; both 42.8. Depth is the default EGT, with point maps and Gaussians as alternatives. — [VLM-IE3D](https://arxiv.org/html/2607.21595v1)
- **Cog3DMap** (Scan2Cap): xyz positional encodings alone give C 64.0–71.7, while the additive latent-geometry projection gives 79.4, over a 65.2 baseline. — [Cog3DMap](https://arxiv.org/html/2603.23023v1)
- **VLM²** (VSI-Bench): gated additive point-map injection, pooled per patch through sinusoidal encoding and an MLP, gives 61.6 vs 58.9 ungated. π3 > VGGT > CUT3R as the geometry source (61.0 / 60.2 / 59.4). — [VLM²](https://arxiv.org/html/2511.20644v2)
- **GeoWeaver** explicitly does not model point maps, depth or poses, and lists this as a limitation, yet it is the highest VSI-Bench score found (75.05 at 10B). — [GeoWeaver](https://arxiv.org/html/2605.22558v1)
- **Point maps as structure:** SpaceMind++ and Cog3DMap use point maps to build voxel or point memories (with 3D RoPE in SpaceMind++). SpaceMind++ reaches 73.2 vs 70.2 for its latent-only predecessor SpaceMind, but with more data (900K) and a new fusion module. — [SpaceMind++](https://arxiv.org/html/2605.09449v1)

### Inferences
- The user's design feeds *only* point-head xyz, with no latent VGGT features. It is therefore unusual: almost all competitors inject latent VGGT, π3 or CUT3R tokens.
- Reaching about 70–71 at 2B with point maps alone is a notable data point against the prevailing view that latent features are needed. Confounds remain: Qwen3.5 base strength, training data, and the fact that the paired ablations above use other base models.
- A latent-VGGT-feature arm (GeoAlign or GeoWeaver-style, additive, top-k layer routing) would be the natural control.

### Gaps
- No VSI-Bench ablation of point maps only vs latent features only at matched model and data was found. The VLM-IE3D and Cog3DMap comparisons are on detection and captioning.
- The user's own public repo (github.com/DarioVajda/mvp-3d-vlm) appeared in the search results and was not treated as an external source.
