# Explicit 3D scene representations (scene graphs, 3D boxes, cognitive maps, splats, agentic tools) on VSI-Bench, and the ≤8B VSI-Bench state of the art

Scope: research notes, compiled 2026-11-14. The research budget was about 20 tool calls. Numbers come from arXiv HTML full text (fetched) unless marked **[snippet-only]**, which means the number was seen only in a search snippet or abstract. Per-task column order used here is Obj.Count / Abs.Dist / Obj.Size / Room.Size / Rel.Dist / Rel.Dir / Route.Plan / Appr.Order / Avg. Many papers copy baseline rows from each other, so the same baseline (e.g. VLM-3R 60.9) appears repeatedly.

User context: Qwen3.5-2B fused with VGGT point maps, about 70–71 overall at 32 frames and about 86–88 on Rel.Dir.

## Q1. Which papers build a scene graph, 3D-box object list, Gaussian splat or cognitive map and feed it to a VLM, evaluated on VSI-Bench?

### Takeaway
2026 has a dense cluster of such methods. Training-free agent/tool pipelines (DeSpatial, Skill-3D/Think3D, SpatialClaw, RieMind) reach about 69 with an 8B agent on reconstructed 3D (DeSpatial). They reach far higher only with ground-truth 3D (RieMind about 85–90 static split; DeSpatial's GT-box oracle about 98). Trained models with explicit maps or graphs (SSR 73.9, SpaceMind++ 73.2, Cog3DMap 65.1) are on par with the best learned-fusion models. I found no Gaussian-splat or NeRF VLM evaluated on VSI-Bench.

### Cited Findings

**Agentic / tool-use pipelines with explicit 3D (boxes, instances, point maps)**
- **DeSpatial** (arXiv 2610.20990, Oct 2026). Training-free tool pipeline (no training procedure is described). Perception and reasoning are decoupled. A shared 3D perception tool builds "a unified, metric, gravity-aligned 3D scene" through segmentation–depth co-refinement and cross-view instance fusion, merging voxel-overlapping instances. The agent then calls reasoning and computation tools. Tools: Pi3 (default reconstruction; VGGT and DA3 also cited), Depth Anything 3 for metric rescaling, GeoCalib for gravity, SAM3 in the raw baseline. Agent: Qwen3-VL-8B. 32 frames in the oracle and raw runs; the frame count for the main result is not stated. — [arXiv abs](https://arxiv.org/abs/2610.20990), [HTML](https://arxiv.org/html/2610.20990)
  - VSI-Bench (Qwen3-VL-8B agent): 65.43 / 55.08 / 66.86 / 73.58 / 67.93 / 82.53 / 64.43 / 77.90 / **69.22**. — [HTML Table 2](https://arxiv.org/html/2610.20990)
  - Oracle on 5 object-centric tasks (4,030 Qs): raw SAM3+Pi3 gives Cnt 42.70, AbsD 32.54, Size 49.68, RelD 47.62, RelDir 79.37, avg 50.38. GT 3D boxes give 98.76 / 96.73 / 99.37 / 98.59 / 97.70, avg 98.23. The authors caution that VSI-Bench annotations are imperfect (citing ReVSI). — [HTML Table 1](https://arxiv.org/html/2610.20990)
  - Other agentic rows in the same table: ViSRA (InternVL3-8B) 49.88; See&Trek (InternVL3-8B) 43.2; ReRe (InternVL3-8B) 34.16; CaMo-3B 44.0; AlloSpatial (Gemini-3-Pro, subset) 61.0; Think3D (Gemini-2.5-Pro, 4 MC tasks only) 51.61. — [HTML Table 2](https://arxiv.org/html/2610.20990)
- **Skill-3D** (arXiv 2606.07436, Jun 2026). Agent with a scene memory and a skill library distilled from successful tool trajectories. Tools: Pi3, GroundingDINO, SAM3, Orient Anything v2, DA3 metric, SwinIR. Open models (Qwen3-VL-4B/8B) are post-trained with SFT and GRPO, so it is **not** training-free for open models. Follows Think3D with **7 frames**. No VSI-Bench average is given. — [HTML](https://arxiv.org/html/2606.07436)
  - Qwen3-VL-8B per task: w/o tools 32.5 / 24.3 / 30.8 / 41.6 / 40.2 / 41.6 / 35.5 / 47.3. Think3D-8B: 48.3 / 38.5 / 51.2 / 58.1 / 41.6 / 52.9 / 45.4 / 60.8. Skill-3D-8B: 56.5 / 48.6 / 59.8 / 67.9 / 52.0 / 60.1 / 58.4 / 66.8. Skill-3D-4B: 48.6 / 36.8 / 50.2 / 57.4 / 43.5 / 50.4 / 48.8 / 56.7. — [HTML Table 2](https://arxiv.org/html/2606.07436)
  - GPT-5.4 + Skill-3D: 66.2 / 61.5 / 74.9 / 77.6 / 62.7 / 67.0 / 71.4 / 78.1. — [HTML Table 1](https://arxiv.org/html/2606.07436)
  - Search snippet: "raises tool-use accuracy from 39% to 78% on VSI-Bench" **[snippet-only]**. — [arXiv abs](https://arxiv.org/abs/2606.07436)
- **SpatialClaw** (arXiv 2606.13673, Jun 2026). Training-free; code is the action interface (a persistent Python kernel). Tools: DA3 `Reconstruct` (depth, intrinsics, extrinsics, point maps, up to 64 frames) and SAM3; the VLM sees at most 32 frames. Backbones are 26B–397B, so none are ≤8B. It reports only "VSI-Bench-U" (Brown et al.) overall. Tools *hurt* the large Qwen models: Qwen3.5-397B-A17B 58.8 → 56.6, Qwen3.5-122B 55.7 → 50.6, Qwen3.6-35B-A3B 54.2 → 51.7. Gains are small for Qwen3.6-27B (55.4 → 57.9) and larger for Gemma4-26B-A4B (30.1 → 38.5). pySpatial and SpaceTools baselines are also listed, but the column alignment in the extracted table is uncertain. — [HTML](https://arxiv.org/html/2606.13673)
- **RieMind** (arXiv 2603.15386, Mar 2026). Training-free LLM agent over an explicit **3D scene graph built from GT annotations** (not reconstructed). Static split only: 6 tasks, 4,185 Qs, with route plan and appearance order dropped. These are upper bounds. — [HTML](https://arxiv.org/html/2603.15386)
  - Columns: Cnt / AbsD / Size / Room / RelD / RelDir / Avg. Qwen2.5-VL-7B agent: 89.7 / 90.3 / 93.6 / 31.9 / 44.5 / 34.7 / 64.1. GPT-4o: 85.1 / 93.2 / 96.5 / 83.5 / 85.6 / 67.5 / 85.2. GPT-4.1: 86.5 / 94.9 / 97.9 / 77.8 / 92.7 / 87.3 / 89.5. Same table, static avg: VLM-3R 67.0, ViCA 63.5, SpaceMind 73.6 (RelDir 88.4). — [HTML Table 4](https://arxiv.org/html/2603.15386)
- **ViSRA** (arXiv 2605.10106) is a video spatial reasoning agent, InternVL3-8B, 49.88 avg (per DeSpatial's table). Per a search snippet, its authors argue cognitive maps "were largely ineffective in practice across most task types", even when GT cognitive maps were given as extra input **[snippet-only]**. — [arXiv PDF](https://arxiv.org/pdf/2605.10106), [DeSpatial Table 2](https://arxiv.org/html/2610.20990)

**Trained models with explicit scene graphs / cognitive maps**
- **SSR: Structured Scene Reasoning** (arXiv 2603.00409, Mar 2026). openPangu-VL-7B with VGGT layer-23 features. The 3D input is fused by "cross-modal addition and token interleaving", which is very close to the user's additive design. A text scene graph ("LocalCogMap": a chain of triplets on a 10×10 grid relative to two anchor objects) is *generated by the model as text* during reasoning. 32 frames. Data: SPAR-7M subset, ScanQA, SQA3D and others, then ViCA, VLM-3R data and the real-world subset of VSI-590K. The authors state that VSI-Bench videos were held out. — [HTML](https://arxiv.org/html/2603.00409)
  - SSR-3D (7B): 70.5 / 67.1 / 76.0 / 79.5 / 71.3 / **93.4** / 48.5 / 85.0 / **73.9**. SSR-2D (no VGGT): 71.8 / 63.4 / 76.5 / 76.5 / 69.2 / 87.9 / 48.5 / 81.9 / 71.9. — [HTML table](https://arxiv.org/html/2603.00409)
- **SpaceMind++** (arXiv 2605.09449, May 2026). InternVL3-8B + VGGT. Builds a **voxelized allocentric cognitive map**: patch features are scattered into ≤5,000 voxels by VGGT 3D coordinates, keeping first-seen timestamps. LoRA training on SpaceMind-900K, which includes VSI-590K. Frame count not stated. — [HTML](https://arxiv.org/html/2605.09449)
  - SpaceMind++: 74.5 / 62.4 / 77.9 / 76.9 / 73.5 / **89.7** / 48.2 / 84.1 / **73.2**. SpaceMind: 73.9 / 61.5 / 77.6 / 74.8 / 67.7 / 88.6 / 46.9 / 70.7 / 70.2. SenseNova-SI-1.2: 72.7 / 56.0 / 77.1 / 75.7 / 70.4 / 81.7 / 42.8 / 79.9 / 69.6. — [HTML Table 1](https://arxiv.org/html/2605.09449)
- **Cog3DMap** (arXiv 2603.23023, Mar 2026). Qwen3-VL-8B. A frozen Point3R (recurrent point-map model) maintains an explicit 3D memory of tokens (position, semantic feature, geometric feature), updated per frame and fed to the decoder. 32 frames for train and eval. Training data: VLM-3R and SPAR training sets. — [HTML](https://arxiv.org/html/2603.23023), [project page](https://cog3dmap.github.io/)
  - Cog3DMap-8B: 69.6 / 54.8 / 67.8 / 67.1 / 64.8 / 85.6 / 43.0 / 67.9 / **65.1**. — [HTML Table 2](https://arxiv.org/html/2603.23023)
- **DR-MV3D** (arXiv 2606.23557). Aligns MLLM-generated cognitive maps with 3D layouts using dense rewards from a 3D foundation model. Reports VSI-Bench 37.1 avg for its 3B model **[snippet-only]**. — [arXiv PDF](https://arxiv.org/pdf/2606.23557)
- **Cambrian-P** (arXiv 2605.22819, May 2026). Not an explicit representation: learned pose queries are added to Cambrian-S-7B and trained on VSI-590K only, with pose tokens omitted at inference. It is the strongest ≤8B result found. 74.9 / 60.1 / 76.0 / 76.9 / 74.8 / **89.5** / 52.6 / 85.0 / **73.7** (128 frames per Table 4; ablations use 32 frames). — [HTML](https://arxiv.org/html/2605.22819)

**Gaussian splats / NeRF**
- No VSI-Bench result for a Gaussian-splat or NeRF-based VLM was found. 3DGSNav (arXiv 2602.12159) puts 3DGS into VLMs for object navigation, not VSI-Bench. SpatialQ (2607.26595) uses an MLLM to *assess* 3DGS quality. — [3DGSNav](https://arxiv.org/pdf/2602.12159), [SpatialQ](https://arxiv.org/pdf/2607.26595)
- "GS-Reasoner" appears in SSR's table: 69.1 / 61.9 / 70.0 / 65.7 / 65.4 / 88.9 / 44.3 / 52.3 / 64.7. I did not verify its method, and the name does not necessarily mean Gaussian splatting. — [SSR HTML](https://arxiv.org/html/2603.00409)

**Benchmark note**
- **ReVSI** (arXiv 2604.24300) rebuilds VSI-Bench with tighter boxes, corrected orientations and more complete object annotations **[snippet-only]**. — [arXiv PDF](https://arxiv.org/pdf/2604.24300)

### Inferences
- Reconstructed-3D agents with an 8B backbone (DeSpatial 69.2) score below trained explicit-3D models (SSR 73.9, SpaceMind++ 73.2, Cambrian-P 73.7). The GT-box oracle (≈98) shows that the bottleneck is perception and instance fusion, not reasoning.
- Skill-3D and Think3D numbers come from 7 frames and a weak no-tool baseline (Qwen3-VL-8B at 7 frames scores 41.6 Rel.Dir). They should not be compared directly with 32-frame numbers.

### Gaps
- SpatialClaw's "VSI-Bench-U" is not the standard VSI-Bench, and I could not confirm whether it is the Debiased set.
- Not verified: GUIDE (2604.05695, 72.1 **[snippet-only]**), SpatialStack (2603.27437), GS-Reasoner's method, and AlloSpatial's full results.
- DeSpatial's frame count for the main result, and SpaceMind++'s frame count, are not stated.

## Q2. Which approach families win which VSI-Bench tasks?

### Takeaway
Explicit or metric 3D (tool agents, voxel maps, GT graphs) helps most on **Abs.Dist** and **Room Size**, and tool agents also lift **Route Plan**. Rel.Dir is high (≈85–93) for every method that injects 3D geometry features, including learned fusion. **Object counting** and **appearance order** are where agents lag trained video models.

### Cited Findings
- **Abs.Dist.** DeSpatial 55.08 and SpaceMind++ 62.4 vs Cambrian-S-7B 50.5. SSR-3D 67.1 vs SSR-2D 63.4. RieMind with GT graph reaches 90–95. — [DeSpatial](https://arxiv.org/html/2610.20990), [SpaceMind++](https://arxiv.org/html/2605.09449), [SSR](https://arxiv.org/html/2603.00409), [RieMind](https://arxiv.org/html/2603.15386)
- **Route Plan.** DeSpatial 64.43 is far above every trained ≤8B model (Cambrian-P 52.6, SSR 48.5, SpaceMind++ 48.2, Cambrian-S 41.8). — same sources
- **Rel.Dir.** SSR-3D 93.4, SpaceMind++ 89.7, Cambrian-P 89.5, GS-Reasoner 88.9, SpaceMind 88.6, Cog3DMap 85.6, DeSpatial 82.53, VLM-3R 80.5, Cambrian-S 76.2. Video-only Qwen3-VL-8B scores 50.9 and VST-7B 55.6. — [SSR](https://arxiv.org/html/2603.00409), [SpaceMind++](https://arxiv.org/html/2605.09449), [Cambrian-P](https://arxiv.org/html/2605.22819)
- **Obj.Count.** Raw SAM3+Pi3 instance counting scores only 42.70; DeSpatial's fused pipeline reaches 65.43. Trained models score about 70–75. — [DeSpatial](https://arxiv.org/html/2610.20990)
- **Appearance order.** VLM-3R scores 40.1 and Cog3DMap 67.9, while Cambrian-P scores 85.0 and SSR-3D 85.0. — [Cog3DMap](https://arxiv.org/html/2603.23023), [SSR](https://arxiv.org/html/2603.00409)
- **Small agents are weak.** RieMind with a Qwen2.5-VL-7B agent on GT graphs scores only 34.7 Rel.Dir and 31.9 Room, against GPT-4.1 at 87.3 and 77.8. — [RieMind](https://arxiv.org/html/2603.15386)

### Inferences
- The user's 86–88 Rel.Dir with a 2B model sits within the band of the best 3D-feature models (85–93). It is above every reconstructed-3D agent that was checked.
- The user's architecture is closest to SSR, which adds VGGT features by cross-modal addition. SSR's 2D→3D gain (+2.0 avg, +5.5 Rel.Dir, +3.7 AbsD) is a useful reference point.
- Abs.Dist and Route Plan are the tasks where explicit or agentic metric 3D still shows headroom over learned fusion.

### Gaps
- No paper compares additive fusion against cross-attention fusion against explicit scene graphs on a single fixed backbone, frame count and training set.

## Q3. Top VSI-Bench numbers for open models ≤8B, and contamination concerns

### Takeaway
The best reported ≤8B numbers (late 2026) are about 73–74: Cambrian-P 73.7, SSR-3D 73.9, SpaceMind++ 73.2 and GeoThinker-8B 72.6. All of these use VSI-590K or similar ScanNet/ARKitScenes-derived training data. Untuned Qwen3-VL-8B scores about 57–60. No official Qwen3.5 VSI-Bench number was found.

### Cited Findings

Columns: Model, Size, Uses 3D input, Avg, Rel.Dir, In-domain data, Source.

| Model | Size | 3D input | Avg | Rel.Dir | In-domain data | Source |
|---|---|---|---|---|---|---|
| SSR-3D | 7B (openPangu-VL) | VGGT feats + text scene graph | 73.9 | 93.4 | VSI-590K real subset, ViCA, VLM-3R, SPAR; 32 fr | [SSR](https://arxiv.org/html/2603.00409) |
| Cambrian-P | 7B (Qwen2.5) | pose queries (train only) | 73.7 | 89.5 | VSI-590K only; 128 fr | [Cambrian-P](https://arxiv.org/html/2605.22819) |
| SpaceMind++ | 8B (InternVL3) | VGGT voxel cog. map | 73.2 | 89.7 | 900K incl. VSI-590K | [SpaceMind++](https://arxiv.org/html/2605.09449) |
| GeoThinker | 8B (Qwen3) | (unverified) | 72.6 | – | – | [Cambrian-P T1](https://arxiv.org/html/2605.22819) |
| GUIDE | ? | geometric priors | 72.1 | – | – | [snippet-only](https://arxiv.org/pdf/2604.05695) |
| SSR-2D | 7B | no 3D | 71.9 | 87.9 | as SSR | [SSR](https://arxiv.org/html/2603.00409) |
| SpaceMind | ? | VGGT | 70.2 | 88.6 | – | [SpaceMind++](https://arxiv.org/html/2605.09449) |
| SenseNova-SI-1.2 | 8B | no | 69.6 | 81.7 | – | [SpaceMind++](https://arxiv.org/html/2605.09449) |
| Cambrian-S† (VSI-590K-only FT) | 7B | no | 69.2 | 82.0 | VSI-590K | [Cambrian-P](https://arxiv.org/html/2605.22819) |
| DeSpatial (agent) | 8B (Qwen3-VL) | Pi3+SAM3 instances, training-free | 69.22 | 82.53 | none | [DeSpatial](https://arxiv.org/html/2610.20990) |
| SenseNova-SI | 8B | no | 68.7 | 80.7 | – | [SSR](https://arxiv.org/html/2603.00409) |
| GeoThinker | 7B | – | 68.5 | – | – | [Cambrian-P](https://arxiv.org/html/2605.22819) |
| Cambrian-S | 7B | no | 67.5 | 76.2 | VSI-590K | [Cambrian-S page](https://cambrian-mllm.github.io/cambrian-s/) |
| Cog3DMap | 8B (Qwen3-VL) | Point3R 3D memory | 65.1 | 85.6 | VLM-3R + SPAR train; 32 fr | [Cog3DMap](https://arxiv.org/html/2603.23023) |
| GS-Reasoner | ? | 3D | 64.7 | 88.9 | – | [SSR](https://arxiv.org/html/2603.00409) |
| VST-7B-RL | 7B | no | 61.2 | 55.6 | – | [Cog3DMap](https://arxiv.org/html/2603.23023) |
| VLM-3R | 7B | CUT3R tokens | 60.9 | 80.5 | VLM-3R data | [Cog3DMap](https://arxiv.org/html/2603.23023) |
| ViCA | 7B | no | 60.6 | 42.6 | ViCA-322K | [SSR](https://arxiv.org/html/2603.00409) |
| Qwen3-VL-8B-Instruct | 8B | no | 57.9 (56.6 in Cambrian-P T1) | 50.9 | none | [SSR](https://arxiv.org/html/2603.00409) |
| Cambrian-S-3B / 1.5B / 0.5B | 3B / 1.5B / 0.5B | no | 57.3 / 54.8 / 50.6 | – | VSI-590K | [Cambrian-S page](https://cambrian-mllm.github.io/cambrian-s/) |
| InternVL3.5-8B / 2B | 8B / 2B | no | 56.3 / 51.5 | – | – | [Cambrian-S page](https://cambrian-mllm.github.io/cambrian-s/) |
| VG-LLM-8B / 4B | 8B / 4B | VGGT | 50.7 / 47.3 | 40.7 / 45.6 | – | [Cog3DMap](https://arxiv.org/html/2603.23023) |
| Spatial-MLLM | 4B | VGGT | 48.4 | 46.2 | – | [Cog3DMap](https://arxiv.org/html/2603.23023) |
| SpaceR | 7B | no | 41.5 | 46.1 | – | [SSR](https://arxiv.org/html/2603.00409) |
| SpatialLadder | 3B | no | 44.9 | – | – | [snippet-only, search result citing Cambrian-P / VST tables](https://arxiv.org/pdf/2511.05491) |

- **Conflicting Cambrian-S-7B numbers.** 67.5 on the project page and in most tables; 62.9 on the SOTA2 leaderboard and in SpatialImaginer **[snippet-only]**. — [Cambrian-S](https://cambrian-mllm.github.io/cambrian-s/), [SOTA2](https://www.sota2.com/research/sota/video-spatial-reasoning-on-vsi-bench)
- **Conflicting VST-7B numbers.** 61.2 in the Cog3DMap/SSR tables, 60.6 in SpaceMind++. — [Cog3DMap](https://arxiv.org/html/2603.23023), [SpaceMind++](https://arxiv.org/html/2605.09449)
- **Qwen3.5.** The Qwen3.5-2B model card reports ERQA, EmbSpatialBench and RefSpatialBench but **no VSI-Bench**. Qwen3.5-2B scores 43.8/33.0 (thinking/non-thinking) on ERQA and 77.9/66.4 on EmbSpatial. — [HF model card](https://huggingface.co/Qwen/Qwen3.5-2B)
- **Contamination: VSI-590K sources.** VSI-590K draws on ScanNet, ScanNet++ V2 and ARKitScenes, the same source datasets as VSI-Bench's videos. The Cambrian-S page makes no statement about overlap. — [Cambrian-S page](https://cambrian-mllm.github.io/cambrian-s/)
- **Contamination: SSR.** SSR explicitly says it isolated VSI-Bench video sequences from training. — [SSR](https://arxiv.org/html/2603.00409)
- **VSI-Bench-Debiased** (Brown et al., arXiv 2511.04655). Iterative Bias Pruning removes the samples that a blind model can solve. It keeps about 70% of samples. Size estimation is dominated by a category-mean-size prior (Gini importance 0.968). A blind LLaVA-Video-7B fine-tuned on in-domain data rises from 25.9 to 44.7 on the original set and drops to 32.0 on Debiased. These figures come from summaries **[snippet-only; not checked against the PDF]**. — [arXiv HTML](https://arxiv.org/html/2511.04655v1), [Emergent Mind](https://www.emergentmind.com/topics/vsi-bench-debiased)
- None of the top papers checked (Cambrian-P, SpaceMind++, SSR, Cog3DMap, DeSpatial) report VSI-Bench-Debiased in the parts read. — [Cambrian-P](https://arxiv.org/html/2605.22819), [SpaceMind++](https://arxiv.org/html/2605.09449)

### Inferences
- At about 70–71 with a 2B model, the user's result is above every ≤4B number found (Cambrian-S-3B 57.3, Spatial-MLLM-4B 48.4, VG-LLM-4B 47.3). It is roughly level with Cambrian-S† (69.2), SenseNova-SI (68.7–69.6), SpaceMind (70.2) and the training-free DeSpatial agent (69.2). It is about 3 points under the ≤8B leaders (73–74).
- The user's 86–88 Rel.Dir is in the top band, alongside SpaceMind 88.6 and GS-Reasoner 88.9, and below SSR-3D 93.4.
- Because VSI-590K and similar data share scenes and question templates with VSI-Bench, reporting VSI-Bench-Debiased (and possibly ReVSI) would make comparisons against the leaders more convincing. That applies especially to Obj.Size and Room Size, which have strong priors.

### Gaps
- No official or third-party VSI-Bench score for Qwen3.5-0.8B/2B/4B/9B was found.
- Not verified from primary sources: GeoThinker method/data, SenseNova-SI data, and SpaceMind's model size.
- No VSI-Bench-Debiased numbers were found for the top ≤8B models.
- I did not verify whether the arXiv PDF matches the Emergent Mind summary of the Debiased numbers.
- No official VSI-Bench leaderboard page was fetched; only the third-party SOTA2 page appeared, as a snippet.
