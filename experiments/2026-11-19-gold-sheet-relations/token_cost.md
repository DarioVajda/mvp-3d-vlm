# Input tokens per condition

Written by `token_cost.py` from the predictions in the repository: the whole input (system prompt, video and text) as the eval sends it. All runs: Qwen3.5-2B, thinking off, 16 frames `uniform128`, direct answers. **Added by the sheet**: the per-question difference from video only on the same questions and frames. **These runs record only `n_input_tokens`, the padded length of each batch** (up to 8 questions of one video), so each count is the longest prompt in its batch: the numbers are upper bounds, and the sheet's cost is overstated more than the video's (about 2,500 tokens, the same in every prompt). The eval now also records each question's own length (`n_prompt_tokens`), which the coming runs (A–E of this experiment, and pipeline v4 without the video) will give. No sheet-only run exists yet.

| Questions | n | Condition | Input tokens, mean | Median | Min–max | Added by the sheet, mean | Median |
|---|---:|---|---:|---:|---:|---:|---:|
| explore200 | 200 | video only | 2619 | 2613 | 2584–2700 | – | – |
| explore200 | 200 | video + gold sheet | 2909 | 2883 | 2727–3135 | 290 | 275 |
| explore200 | 200 | video + gold sheet + relations | 3193 | 3161 | 2759–3701 | 574 | 541 |
| dev200_v4 | 50 | video only | 2612 | 2599 | 2540–2700 | – | – |
| dev200_v4 | 50 | video + pipeline v4 sheet | 2966 | 2895 | 2722–3204 | 354 | 311 |
| batch2_A (final2) | 798 | video only | 2583 | 2577 | 2542–2693 | – | – |
| batch2_A (final2) | 798 | video + pipeline v4 sheet | 3078 | 2913 | 2725–4538 | 496 | 358 |

Question sets:

- **explore200:** structured-cot's 200 explore200 questions (25 scenes; gold sheets); `results/zeroshot/vsibench/structured_cot/explore200`
- **dev200_v4:** dev200's 50 questions run with pipeline v4; `experiments/2026-11-18-scene-sheet-formats/results/dev200_v4`
- **batch2_A (final2):** the 798 questions of the registered pipeline v4 test; `experiments/2026-11-18-scene-sheet-formats/results/final2`

The format experiment's 2,728-question runs are not in the repository; its `results.csv` gives only the sheet's mean tokens per format (gold csv 215.4, pipeline csv 469).
