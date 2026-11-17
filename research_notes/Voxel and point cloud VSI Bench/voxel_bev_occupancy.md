# Voxel / occupancy / BEV 3D representations in VLMs evaluated on VSI-Bench (2026 focus, late 2025 context)

Research date: 2026-11-14. Read-status legend used throughout:
- **[FULL]** = arXiv HTML read via fetch (the fetch tool summarises the page, so tables were transcribed by a model; the numbers below are as returned, cross-checked for internal consistency where possible).
- **[PAGE]** = project page or abstract read, full paper not read.
- **[SNIPPET]** = seen only in a search-result snippet; numbers not verified.

User context for comparison: Qwen3.5-2B + VGGT point maps pooled to visual-token cells, additive vs 3D-aligned cross-attention (distance bias + 3D RoPE), about 70-71 overall on VSI-Bench at 32 frames, relative direction about 86-88.

## Q1. Which papers voxelise / grid the scene and evaluate on VSI-Bench?

### Takeaway
Only a handful of 2026 papers both build an explicit grid/voxel/BEV scene representation and report VSI-Bench numbers. The cleanest match is **Cambrian-BEV** (Oct 2026), which replaces frame tokens with a 27x27 BEV grid of pooled patch features (63.48 avg, 0.5B LLM, GT depth/poses). **Cog3DMap** (Mar 2026) keeps a 3D memory with one token per spatial location from Point3R (65.1 avg, 8B). The other methods build grids/maps outside the VLM and give it text or images: **GeoLift** (training-free, VGGT-Omega point cloud plus an occupancy grid for A* routing, 64.1), **GR3D** (pi3/VGGT voxel-grid object extraction rendered as annotated images and text, 59.8 with GPT-5), and **EagleVision** (Dec 2025, BEV image from SLAM plus pose-query tool calls under RL, 63.5). The voxel-token LLMs (LLaVA-3D, Video-3D LLM, Qwen-3D, Geo3DPruner) do **not** report VSI-Bench in what I could read.

### Cited Findings

**A. Grid/voxel tokens fed directly into the LLM (closest to the user's setup)**

- **Cambrian-BEV: "Thinking from Above: Efficient Spatial Reasoning with Bird's-Eye-View Visual Tokens"** (Siddiqui, Savaliya, Chada, Sachdeva; Honda Research Institute USA; arXiv 2610.22700, 15 Oct 2026) [FULL] — [arXiv HTML](https://arxiv.org/html/2610.22700v1)
  - Base: Cambrian-S with a **Qwen2.5-0.5B** LLM, a 2-layer MLP projector and a frozen DINOv3 ViT-L/16 encoder. — [arXiv](https://arxiv.org/html/2610.22700v1)
  - 3D source: **dataset sensor depth, intrinsics and GT poses** ("we perform no reconstruction or pose estimation"). No VGGT. — [arXiv](https://arxiv.org/html/2610.22700v1)
  - Construction: patch features are back-projected to world coordinates and hard-binned onto a **27x27 ground-plane grid spanning +/-5 m (0.37 m cells) = 729 tokens**, with per-cell averaging and zero vectors for empty cells. The grid's forward axis is aligned with the first camera ("first-pose"). — [arXiv](https://arxiv.org/html/2610.22700v1)
  - Fusion: the BEV tokens **replace** the frame tokens at the `<image>` placeholder. There are no new parameters and no BEV-specific positional embedding; only the LLM's own RoPE is used. — [arXiv](https://arxiv.org/html/2610.22700v1)
  - Training: VSI-288K-Depth (288,478 QA over 4,955 scenes from ScanNet, ScanNet++ and ARKitScenes). Projector and LLM are trained, 2 epochs, lr 2e-5. **Route planning is absent from training** (held out). 128 frames. — [arXiv](https://arxiv.org/html/2610.22700v1)
  - VSI-Bench, BEV vs. a frame-token control with identical fine-tuning (128 frames; 729 vs 9,216 tokens): Count 70.16 vs 69.79; Abs Dist **44.78** vs 37.49; Obj Size 67.36 vs 69.97; Room Size **85.00** vs 60.49; Rel Dist **56.20** vs 49.86; Rel Dir **82.67** vs 48.58; Route Plan **39.18** vs 26.80; Appearance Order **62.46** vs 76.86; **Avg 63.48 vs 54.98**. — [arXiv](https://arxiv.org/html/2610.22700v1)
  - References: official Cambrian-S-0.5B scores 51.73 and Cambrian-S-7B 66.69. The authors say their reproduced frame baselines are about 10 points below the official 0.5B, which they attribute to bf16 autocast. ReVSI: 44.68 vs 40.17. — [arXiv](https://arxiv.org/html/2610.22700v1)
  - Efficiency: 12.6x fewer visual tokens, 2.9x faster prefill and 3.6x lower peak memory on one H100, but **only 1.13x end-to-end** because decoding is unchanged. — [arXiv](https://arxiv.org/html/2610.22700v1)

- **Cog3DMap: "Multi-View Vision-Language Reasoning with 3D Cognitive Maps"** (Gwak, Jeong, Jeon, Lee, Shin, Cho; arXiv 2603.23023, 24 Mar 2026) [PAGE: abstract + project page; full paper not read] — [arXiv abs](https://arxiv.org/abs/2603.23023); [project page](https://cog3dmap.github.io/)
  - The method "recurrently constructs an explicit 3D memory from multi-view images". Each spatial coordinate carries a token with semantic features from the MLLM's vision encoder and geometry from **a pretrained Point3R model**. Updates "retain, update, and expand tokens", keeping "a single token per spatial location", which amounts to a sparse voxel-like token map. — [project page](https://cog3dmap.github.io/)
  - The base MLLM is not named on the page (sizes 8B and 4B). Training data, positional encoding and frame count are not stated on the page. — [project page](https://cog3dmap.github.io/)
  - **VSI-Bench, Cog3DMap-8B**: Avg **65.1**; Count 69.6, Abs Dist 54.8, Obj Size 67.8, Room Size 67.1, Rel Dist 64.8, **Rel Dir 85.6**, Route Plan 43.0, Appearance Order 67.9. The project page lists these comparisons: VST-7B 61.2, VLM-3R-7B 60.9, VG-LLM-8B 50.7, VG-LLM-4B 47.3, Spatial-MLLM-4B 48.4. — [project page](https://cog3dmap.github.io/)
  - Up to **90.2% fewer visual tokens** on long sequences (RoboFAC: 460.1 vs 4,684.8 tokens for Qwen3-VL-4B). Not reported for VSI-Bench. — [project page](https://cog3dmap.github.io/)

**B. Grid / occupancy / voxel built outside the VLM and passed as text, images or tools (training-free or RL agents)**

- **GeoLift: "Grounding Video Spatial Reasoning in Explicit 3D Maps"** (Li, Long, Zhou, Chen, Yang, Xie; arXiv 2610.33558, 24 Oct 2026) [FULL] — [arXiv HTML](https://arxiv.org/html/2610.33558)
  - Frozen **Qwen3-VL-4B**, training-free. **VGGT-Omega** supplies depth, intrinsics and poses, and **MoGe-2** a metric scale factor; no GT geometry. Builds a metric point cloud with superpoints and object anchors and footprints. **For route planning it builds an occupancy grid and runs A\***. The MLLM receives geometric "hints" and a tool box (exact format not specified). — [arXiv](https://arxiv.org/html/2610.33558)
  - VSI-Bench, direct vs GeoLift: Avg 55.4 -> **64.1**; Count 65.9 -> 68.0; Abs Dist 43.1 -> 49.4; Obj Size 73.9 -> 74.4; Room 61.0 -> 65.9; Rel Dist 53.1 -> 57.8; **Rel Dir 46.1 -> 74.4**; **Route 30.9 -> 53.6**; Appearance Order 69.1 -> 69.1. — [arXiv](https://arxiv.org/html/2610.33558)

- **GR3D: "Boosting MLLM Spatial Reasoning with Geometrically Referenced 3D Scene Representations"** (Yuan, Kumar, Wang; Zillow; arXiv 2603.08592, v2 26 Apr 2026) [FULL] — [arXiv HTML](https://arxiv.org/html/2603.08592v2)
  - Zero-shot. **pi3** reconstruction (VGGT as an ablation) runs on unposed images. Mask2Former labels are back-projected into a point cloud, **aggregated in a voxel grid by majority label** and grouped into objects. The VLM receives images annotated with object IDs plus text listing 3D primitives (boxes, cylinders) in global coordinates. The authors explicitly reject BEV input, noting GPT4Scene needs fine-tuning to use it. — [arXiv](https://arxiv.org/html/2603.08592v2)
  - VSI-Bench, GPT-5 (16 frames) -> GR3D+GPT-5: Avg 50.3 -> **59.8**; Abs Dist 28.8 -> 48.2; Room 50.7 -> 72.1; Rel Dist 42.6 -> 50.8; **Rel Dir 42.4 -> 69.5**; Route 50.0 -> 58.7. GR3D+GPT-4o: 34.0 -> 45.3. GR3D+InternVL2-8B (32 frames): 37.5 -> 42.5. — [arXiv](https://arxiv.org/html/2603.08592v2)

- **EagleVision: "A Dual-Stage Framework with BEV-grounding-based Chain-of-Thought for Spatial Intelligence"** (Wan et al., Beihang; arXiv 2512.15160, **17 Dec 2025, context**) [FULL] — [arXiv HTML](https://arxiv.org/html/2512.15160v1)
  - **Qwen3-VL-8B** trained with GRPO (Zero-RL, no SFT). The **Vipe SLAM** pipeline gives poses and depth; the point cloud is rasterised into a **BEV grid of occupancy/height statistics** rendered as an image. The model issues pose queries (x, y, r) on the BEV plane, and the nearest real frame is retrieved and fed back. — [arXiv](https://arxiv.org/html/2512.15160v1)
  - VSI-Bench: Avg **63.5**; Count 74.9, Abs Dist 41.3, Obj Size 72.3, Room 69.4, Rel Dist 70.1, **Rel Dir 84.7**, Route 46.3, Appearance Order 49.3. The Qwen3-VL-8B baseline scores 59.4. Which split was used is not clearly stated, since some table rows are marked "tiny". — [arXiv](https://arxiv.org/html/2512.15160v1)

**C. Relevant 3D-token methods that do NOT report VSI-Bench (as far as I could read)**

- **Qwen-3D** (Lin, Jain, Liu, Fragkiadaki; CMU; arXiv 2608.02980, v2 13 Oct 2026) [FULL, about 100% of text read] — Qwen2.5-VL 3B/7B; **5 cm voxel pooling** (mean features and coordinates per occupied voxel; ScanNet points drop from about 33K to about 14K); **M-RoPE extended to world xyz** (PE3D = [PE(t); PE(x); PE(y); PE(z)]) plus full attention over visual tokens. Uses sensor RGB-D poses or MoGe lifting. **No VSI-Bench numbers in the text.** Its ablation is on ScanRefer (3B, 50 scenes): 2D PE 53.2, "vanilla 3D" PE 49.9, "aligned 3D" PE 53.5. Without voxel pooling the model runs out of memory. — [arXiv HTML](https://arxiv.org/html/2608.02980)
- **Geo3DPruner: "Geometry-Guided 3D Visual Token Pruning for Video-Language Models"** (Li, Huang, Fu, Wang, Liu; arXiv 2604.18260, Apr 2026) [FULL] — Video-3D LLM (LLaVA-Video 7B / Qwen2-7B) with VGGT-1B features added to SigLIP features. VGGT-predicted depth and poses assign tokens to **0.1 m voxels**, then the method prunes within and across voxels. **VSI-Bench is not reported**; it is evaluated only on ScanRefer, Multi3DRefer, Scan2Cap, ScanQA and SQA3D. — [arXiv HTML](https://arxiv.org/html/2604.18260)
- **LLaVA-3D** (arXiv 2409.18125, 2024) uses "3D voxelization pooling to compress the 3D patch tokens" [SNIPPET]. **Video-3D LLM** (arXiv 2412.00493, 2024) adds sinusoidal 3D PE to video tokens [SNIPPET]. I found no VSI-Bench numbers reported by either paper's authors. — [LLaVA-3D](https://arxiv.org/pdf/2409.18125); [Video-3D LLM](https://arxiv.org/pdf/2412.00493)

**D. Checked and excluded or uncertain**

- **ViSRA** (arXiv 2605.10106, 11 May 2026) [FULL]: a search snippet claimed it feeds BEV renders to the MLLM, but the paper does **not**. It only says ground-plane estimation "enables" BEV rendering. Its pipeline is VGGT + Rex-Omni + SAM 2 producing text tool outputs. Qwen2.5-VL-7B goes from 37.6 to 52.2 avg (Rel Dir 39.6 -> 71.7) at 64 frames, training-free. It is a point-map/object pipeline, not a grid method. — [alphaXiv](https://www.alphaxiv.org/abs/2605.10106.md)
- **SpatialImaginer** (arXiv 2604.17385, Apr 2026) [SNIPPET]: BAGEL-7B-MoT, VSI-Bench 63.7 avg. A search summary described it as generating BEV maps, but I could not confirm that the "visual imagination" is a BEV. Treat it as unverified. — [arXiv](https://arxiv.org/pdf/2604.17385)
- **Video2Layout** (arXiv 2511.16160, Nov 2025) [SNIPPET]: "metric-grounded cognitive map composed of BEV bounding boxes". VSI-Bench numbers not verified. — [arXiv](https://arxiv.org/html/2511.16160v1)

### Inferences
- None of the grid/voxel methods beat the user's about 70-71 overall at 32 frames. The best grid-based numbers are about 63-65 (Cambrian-BEV 63.48 at 0.5B with GT geometry and 128 frames; Cog3DMap-8B 65.1; GeoLift 64.1; EagleVision 63.5). The protocols differ (model size, frames, training data), so this is a loose comparison, not a ranking.
- The user's relative-direction score (86-88) is at or above the grid methods: Cog3DMap 85.6, EagleVision 84.7, Cambrian-BEV 82.67. Explicit geometry of any kind seems to unlock rel-dir; the representation (grid vs per-pixel point map) appears to matter less.
- Cambrian-BEV's very high room size (85.00) and large rel-dir gain with no new parameters suggest that much of the "3D" gain on these tasks comes from consolidating views in a world-aligned frame. The user's per-pixel approach does that implicitly through xyz.

### Gaps
- Cog3DMap's base model, training data, frame count and fusion/PE details: only the project page was read.
- Whether LLaVA-3D, Video-3D LLM or Qwen-3D have been evaluated on VSI-Bench by third parties. Some leaderboards may list them, but I did not verify any number.
- SpatialImaginer and Video2Layout representation details and VSI-Bench per-task numbers were not read in full.
- My searches found no occupancy-network-based (semantic occupancy) VLM evaluated on VSI-Bench. Occupancy-LLM work (e.g. OccVLA, Occ-LLM) is in driving and does not report VSI-Bench ([OccVLA](https://arxiv.org/pdf/2509.05578)).

## Q2. Per-paper details: base VLM, 3D source, fusion, training data, per-task numbers

### Takeaway
Grid methods split into (i) **learned token-level** fusion (Cambrian-BEV: GT depth/poses, token replacement, no extra PE; Cog3DMap: Point3R geometry, one token per 3D location) and (ii) **training-free or RL agent** pipelines that use predicted geometry (VGGT-Omega, pi3, Vipe SLAM) and pass grids or maps as text, images or tools. The training-free pipelines give large gains in rel-dir and route planning but stay in the low-to-mid 60s overall.

### Cited Findings
Summary table (all VSI-Bench; "—" = not reported or not found):

| Method (date) | Base VLM | 3D source | Representation -> fusion | Train data | Frames | Avg | AbsDist | RoomSz | RelDist | RelDir | Route |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Cambrian-BEV (Oct 2026) [FULL] | Cambrian-S, Qwen2.5-0.5B | GT sensor depth + GT poses | 27x27 BEV feature grid, replaces frame tokens, LLM RoPE only | VSI-288K-Depth | 128 | 63.48 | 44.78 | 85.00 | 56.20 | 82.67 | 39.18 (held out) |
| Cog3DMap-8B (Mar 2026) [PAGE] | not named (8B) | Point3R | one token per 3D location (recurrent 3D memory) | — | — | 65.1 | 54.8 | 67.1 | 64.8 | 85.6 | 43.0 |
| GeoLift (Oct 2026) [FULL] | Qwen3-VL-4B frozen | VGGT-Omega + MoGe-2 | point cloud + superpoints; occupancy grid + A* for routes; hints/tools | none | — | 64.1 | 49.4 | 65.9 | 57.8 | 74.4 | 53.6 |
| GR3D + GPT-5 (Mar/Apr 2026) [FULL] | GPT-5 | pi3 (VGGT ablation) | voxel-grid object extraction -> annotated images + 3D text | none | 16 | 59.8 | 48.2 | 72.1 | 50.8 | 69.5 | 58.7 |
| EagleVision (Dec 2025, context) [FULL] | Qwen3-VL-8B | Vipe SLAM | BEV occupancy/height image + pose-query tool | GRPO (data unnamed; VLM-3R data in ablations) | — | 63.5 | 41.3 | 69.4 | 70.1 | 84.7 | 46.3 |

Sources: [Cambrian-BEV](https://arxiv.org/html/2610.22700v1); [Cog3DMap](https://cog3dmap.github.io/); [GeoLift](https://arxiv.org/html/2610.33558); [GR3D](https://arxiv.org/html/2603.08592v2); [EagleVision](https://arxiv.org/html/2512.15160v1)

Point-map / feature reference rows reported in these papers (for the user's comparison):
- VLM-3R-7B (DUSt3R/CUT3R-derived features): Avg 60.9; Abs Dist 49.4, Room 67.1, Rel Dist 65.4, Rel Dir 80.5, Route 45.4, Appearance Order 40.1 — [Cog3DMap page](https://cog3dmap.github.io/); [EagleVision](https://arxiv.org/html/2512.15160v1)
- VG-LLM-8B 50.7, VG-LLM-4B 47.3, Spatial-MLLM-4B 48.4 (VGGT-encoder methods) — [Cog3DMap page](https://cog3dmap.github.io/)
- GS-Reasoner 64.7 avg with Rel Dir 88.9, as reported in ViSRA's table — [ViSRA](https://www.alphaxiv.org/abs/2605.10106.md)

Data-quality notes found while reading:
- EagleVision claims a "3.6-point" gain over VLM-3R, but its own table gives 63.5 - 60.9 = 2.6. It also claims to exceed humans on fine-grained distance, but the table shows Abs Dist 41.3 vs human 47.0. — [EagleVision](https://arxiv.org/html/2512.15160v1)
- SpatialImaginer quotes VLM-3R-7B as both 60.9 and 60.7 in different places [SNIPPET]. — [SpatialImaginer](https://arxiv.org/pdf/2604.17385)
- Cambrian-BEV's own frame baselines are about 10 points below the official Cambrian-S-0.5B (attributed to bf16), so its +8.5 gain is measured against a weakened control. — [Cambrian-BEV](https://arxiv.org/html/2610.22700v1)

### Inferences
- Among learned grid methods, Cambrian-BEV is the only one with a clean same-recipe control; Cog3DMap compares against other published models. For the user's thesis, Cambrian-BEV is the most useful counterpoint: a pooled, world-aligned grid **replacing** frame tokens at tiny scale (0.5B) gives large rel-dir and room-size gains but loses appearance order (-14.4). The user's additive/cross-attention fusion keeps the frame tokens and therefore keeps temporal order.
- Training-free pipelines are strongest on **route planning** (GeoLift 53.6, GR3D 58.7), probably because they plan explicitly (A* on an occupancy grid) rather than letting the LLM reason over tokens. Route planning is likely where learned token fusion (the user's setup included) has the most headroom.

### Gaps
- Per-task numbers for Object Count, Object Size and Appearance Order are in the cited sources but left out of the summary table for brevity (see the per-paper bullets in Q1).
- Frame counts for Cog3DMap, GeoLift and EagleVision were not found in what I read.

## Q3. Did any paper directly compare voxel/BEV vs point-map vs pixel-aligned fusion on VSI-Bench?

### Takeaway
**No paper I found directly compares voxel/BEV against per-pixel point-map fusion on VSI-Bench under matched conditions.** The closest evidence: (a) Cambrian-BEV compares BEV-only, frames-only and frames+BEV hybrid tokens; (b) GR3D compares explicit 3D object references against camera parameters only (an implicit association) and against VGGT vs pi3 geometry; (c) Qwen-3D compares 2D vs "vanilla 3D" vs "aligned 3D" positional encodings, but on ScanRefer, not VSI-Bench.

### Cited Findings
- Cambrian-BEV (VSI-Bench avg, same fine-tuning): BEV only (729 tokens) **63.48**; frame tokens only **54.98**; **hybrid 128 frames + BEV 56.06** (13.6x tokens); 16 temporal chunks + global grid 54.44. — [arXiv](https://arxiv.org/html/2610.22700v1)
- Cambrian-BEV interface ablations: a 2-layer MLP adapter on BEV tokens **drops to 51.45 (-12.0)**; a grid self-attention encoder gives 44.69; a learned empty-cell token 60.07. Grid resolution 18x18 gives 55.70, 27x27 63.48, 36x36 57.90. Orientation: world-aligned 55.69, PCA 52.09, first-pose 63.48. Height filter 0.05-2.0 m: 57.55. Encoder: DINOv3 63.48 vs SigLIP2 55.06. — [arXiv](https://arxiv.org/html/2610.22700v1)
- Cambrian-BEV explicitly does **not** compare quantitatively against point-map or pixel-aligned 3D fusion. GPT4Scene, Struct2D and Cog3DMap appear only in related work. — [arXiv](https://arxiv.org/html/2610.22700v1)
- GR3D (GPT-5, VSI-Bench avg): full GR3D (pi3) 59.8; with VGGT reconstruction 58.3; **camera parameters only ("implicit association") 48.8**; textual scene description with ID citations 53.4; no object annotation 50.3. — [arXiv](https://arxiv.org/html/2603.08592v2)
- GeoLift ablation: without superpoint association 62.8 avg (Route 46.9, Rel Dir 72.1) vs full 64.1. No representation-type comparison. — [arXiv](https://arxiv.org/html/2610.33558)
- Qwen-3D (ScanRefer Top-1@0.5, 3B, 50 scenes, **not VSI-Bench**): 2D PE 53.2, "vanilla 3D" PE 49.9, **"aligned 3D" PE 53.5**. — [arXiv](https://arxiv.org/html/2608.02980)
- ViSRA reports that **ground-truth cognitive maps lowered** Qwen2.5-VL-7B on six question types, including relative direction (values in a figure only). The paper did not test grid tokens. — [ViSRA](https://www.alphaxiv.org/abs/2605.10106.md)

### Inferences
- Cambrian-BEV's finding that **hybrid frames+BEV (56.06) is far worse than BEV alone (63.48)** at 0.5B is relevant to the user. Naively concatenating 3D-structured tokens with frame tokens may dilute the signal, while additive per-token fusion (the user's design) avoids adding tokens. Whether this holds at 2B+ is untested.
- Cambrian-BEV's result that adding **learned** adapters on BEV tokens hurts (-12 points) suggests that small models over-fit new interfaces on about 288K QA. This is loosely consistent with any observation by the user that zero-initialised or minimal fusion modules work best (inference, not tested).
- Qwen-3D's "aligned 3D" vs "vanilla 3D" PE result parallels the user's 3D RoPE design choice: naive xyz RoPE hurt and aligned RoPE recovered or beat 2D PE. It is a different benchmark, though.
- A direct matched comparison of voxel/BEV pooling vs per-pixel point-map fusion on VSI-Bench appears to be **an open gap the user's project could fill** (e.g. pooling VGGT points into a BEV/voxel grid vs per-token xyz, same base model and data).

### Gaps
- No source found comparing BEV/voxel vs point-map fusion on VSI-Bench with the same model and data.
- Cog3DMap may contain ablations (e.g. its 3D map vs raw frames, or vs VLM-3R-style features), but only the project page was read.

## Q4. Reported weaknesses (cost, frame count, need for GT poses)

### Takeaway
The main weaknesses: **dependence on GT or calibrated geometry** (Cambrian-BEV uses GT depth and poses and defers estimated geometry to future work), **loss of temporal order** when collapsing to a grid (appearance order -14.4), **flattened vertical structure** for BEV, **error propagation and latency** in training-free pipelines (ViSRA about 40 s per question), and **modest end-to-end speedups** despite large token reductions (Cambrian-BEV 1.13x).

### Cited Findings
- Cambrian-BEV: the time-collapsed grid discards visitation order (appearance order loss); vertical structure is flattened, so thin or stacked objects share cells; it assumes calibrated depth and pose and suits static, room-scale indoor scenes; "robustness to estimated geometry is deferred to future work"; rel-dir gains depend on the egocentric grid frame; end-to-end speedup is only 1.13x because decoding is unchanged. — [arXiv](https://arxiv.org/html/2610.22700v1)
- GeoLift: performance depends on the MLLM's 2D grounding and on the reconstructed geometry, so errors propagate; fallbacks cannot recover missing evidence; gains on counting and object size are small. — [arXiv](https://arxiv.org/html/2610.33558)
- GR3D: voxel-connectivity instance separation struggles in clutter (hurting count and size); noisy point clouds cause missed objects or oversized boxes; the multi-stage pipeline propagates errors; VGGT's metric scale had to be estimated from typical object heights. — [arXiv](https://arxiv.org/html/2603.08592v2)
- EagleVision: depends on frozen Vipe SLAM and FG-CLIP; evaluated only on VSI-Bench; reward weights unspecified; orientation conventions are inconsistent between text and tool prompt. — [arXiv](https://arxiv.org/html/2512.15160v1)
- ViSRA (non-grid pipeline, for cost reference): about 40.1 s per question at 64 frames vs 26.7 s at 16 frames; accuracy 52.2 / 50.4 / 46.7 at 64 / 32 / 16 frames. — [ViSRA](https://www.alphaxiv.org/abs/2605.10106.md)
- Qwen-3D: without voxel pooling it runs out of memory even with fewer frames; it depends on externally estimated depth and poses; it assumes static scenes. — [arXiv](https://arxiv.org/html/2608.02980)
- Geo3DPruner: adds a VGGT-1B encoder whose cost is not quantified; reports no latency or FLOPs. — [arXiv](https://arxiv.org/html/2604.18260)

### Inferences
- The user's pipeline (predicted VGGT point maps, no GT poses, frame tokens kept) avoids the two biggest grid weaknesses: GT-geometry dependence and temporal-order loss. That is a defensible point of contrast in the write-up.
- The strongest argument for grids is token efficiency (Cambrian-BEV 12.6x; Cog3DMap up to 90.2% on long sequences). That matters for frame scaling (e.g. 128 frames) more than at 32 frames.

### Gaps
- I found no grid method that reports VSI-Bench at matched frame counts (e.g. 32 frames) against a per-pixel 3D fusion baseline.
- No grid paper reports VSI-Bench robustness to predicted vs GT geometry.
