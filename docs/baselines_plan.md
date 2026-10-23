# Baselines

Evaluate released 3D-cue VLM checkpoints on VSI-Bench and MMSI-Bench, to answer:

1. **Reproduction:** with the authors' own pipeline and inputs, do we get their reported numbers?
2. **Representativeness:** how much do those numbers depend on frames, sampling, prompt or in-domain training data?

Reproduction runs use the authors' code unchanged (only paths, launcher and, if unavoidable, the attention kernel
change). Protocol variations are separate, labelled runs.

## Layout

- `baselines/<method>/`: venvs (`.venv*`), `env/` (setup scripts, `freeze*.txt`), `eval/` (run and merge scripts,
  lmms-eval tasks), `checks/` (weight checks).
- `baselines/_shared/input_dump_hook/`: hook that dumps prompts, frame indices and tensor hashes for the first questions.
- `results/baselines/<method>/`: outputs, each with `manifest.txt`, `freeze.txt` and `inputs.jsonl`.
- Upstream repos in `third_party/` (gitignored): Spatial-MLLM `9fc4738` (+ branch `v1.0`), SpatialStack `b84ebaa`,
  VG-LLM `c77fd1d`, VLM-3R `206afd0`.

## Baselines

| Model | Checkpoint | Base | Authors' VSI protocol | Status |
|---|---|---|---|---|
| SpatialStack-Qwen3.5-4B | `Journey9ni/SpatialStack-Qwen3.5-4B` | Qwen3.5-4B | lmms-eval, 32 uniform frames as images | done |
| SpatialStack-Qwen2.5-4B | `Journey9ni/SpatialStack-Qwen2.5-4B` | Qwen2.5-VL-3B | same | done |
| Spatial-MLLM v1.1 | `Diankun/Spatial-MLLM-v1.1-Instruct-{135K,820K}` | Qwen2.5-VL-3B | own script, 16 frames (uniform / space-aware) | done |
| VG-LLM | `zd11024/vgllm-qa-vggt-{4b,8b}` | Qwen2.5-VL-3B / 7B | lmms-eval, 32 uniform frames as images | done |
| Spatial-MLLM v1.0 | `Diankun/Spatial-MLLM-subset-sft` | Qwen2.5-VL-3B | own script, 16 frames | not run (released model is trained on a subset, so no target) |
| VLM-3R | `Journey9ni/vlm-3r-llava-qwen2-lora` | LLaVA-NeXT-Video-7B | lmms-eval fork, 32 frames | not run (hardest: 2024 stack, CUT3R CUDA extension) |

Out of scope: 3DRS (no VSI/MMSI results), unofficial `rooty2020/*` checkpoints.

## Results

VSI-Bench (5130 questions; standard error about 0.6 overall):

| Model | Frames | Reported | Ours |
|---|---|---|---|
| SpatialStack-Qwen3.5-4B | 32 uniform | 67.5 | 67.53 |
| SpatialStack-Qwen2.5-4B | 32 uniform | 60.9 | 60.77 |
| Spatial-MLLM-v1.1-135K | 16 uniform | 50.48 | 50.42 |
| Spatial-MLLM-v1.1-135K | 16 space-aware | 51.81 | 51.34 |
| Spatial-MLLM-v1.1-820K | 16 uniform | 53.00 | 53.32 |
| Spatial-MLLM-v1.1-820K | 16 space-aware | 53.43 | 53.08 |
| VG-LLM-8B | 32 uniform | 50.7 | 50.69 |
| VG-LLM-4B | 32 uniform | 47.3 | 46.53 |

Everything reproduces within 0.5 except VG-LLM-4B (−0.8, mostly room size −6.5). The environment doesn't explain it;
most likely the released 4B differs from the one the paper's v3 numbers came from. The space-aware gain for 820K
(+0.4 reported) doesn't show up (−0.2).

VG-LLM image benchmarks also reproduce: CV-Bench 81.7 / 80.5 (reported 81.7 / 79.5, 8B / 4B), BLINK 50.9 / 48.8
(reported 51.5 / 48.9).

MMSI-Bench (no paper reports it; SpatialStack's `mmsibench` task; chance 25%):

| Model | Overall |
|---|---|
| SpatialStack-Qwen3.5-4B | 28.0 |
| SpatialStack-Qwen2.5-4B | 25.7 |
| VG-LLM-4B | 28.1 |
| VG-LLM-8B | 27.2 |
| Qwen3-VL-2B / 4B zero-shot (ours) | 29.0 / 27.5 |

None of the baselines beats plain Qwen3-VL on MMSI.

## Environment notes

- **SpatialStack-Qwen3.5:** README environment, but torch 2.10.0+cu130 (not cu129) and `MODEL_IMPL=qwen3_5`. The
  "UNEXPECTED" geometry weights in the load report are by design; all checkpoint tensors match the loaded model.
- **SpatialStack-Qwen2.5:** needs VG-LLM's transformers 4.50 stack (`env/setup_qwen25.sh`), H100/A100.
- **Spatial-MLLM:** README environment, unpinned packages as of 2026-01-05, H100/A100. Space-aware runs use the
  authors' pre-sampled frames. Runs as 4 shards, merged and scored by `eval/merge_shards.py`.
- **VG-LLM:** `setup.py` pins (torch 2.5.1). Most shards ran in a Blackwell environment (`env/setup_blackwell.sh`,
  torch 2.7.1), checked against the pinned one on one shard (49.70 vs 49.80). VSI runs as 6 shards; BLINK is pinned
  to revision `cbb5717e` because the current Hub version breaks the task.

## Remaining

1. Training-data overlap: compare training-set scene ids and question templates with VSI-Bench's 288 scenes.
2. Shared protocol re-runs (same 16/32 uniform frames, prompt, greedy decoding for every model).
3. VLM-3R, if worth the effort.
