# pipeline_v4's sheet without the video on batch2_A (exploratory)

**Exploratory, not pre-registered.** `prereg_batch2_A.md` registered `A_video` and `pipeline_v4` on these 798 questions (`results/final2/`); `pipeline_v4_no_video` was added after their results were known, so nothing here tests a registered hypothesis, and the p-values are not corrected for multiple comparisons.

- **`pipeline_v4_no_video`**: pipeline_v4's sheet, as final2 gave it (`v4_eval.py`, b381205's renderer and the v4 relations), and the question, without the video (`--no-video`): final2's prompt less the video and its first line. Same model (Qwen3.5-2B, revision `15852e8`), thinking off, up to 2,048 new tokens, final answer scored. Code `e60f467` (the commit that fixed `--no-video`'s progress line), Colab, one Tesla T4; final2's GPU is not recorded. `results/v4_sheet_only/`, `v4_sheet_only.ipynb`.

## Scores

VSI-Bench's scores ×100 (`summarize`; relative direction's three types as one task; overall: the mean of the 8 tasks).

| Condition | Overall | Count | Abs. dist. | Obj. size | Room size | Rel. dist. | Rel. dir. | Route | Order |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| video only (`A_video`) | 41.2 | 30.0 | 31.8 | 55.5 | 43.2 | 45.3 | 47.5 | 32.1 | 44.1 |
| video + v4 sheet (`pipeline_v4`) | 36.7 | 30.2 | 25.7 | 31.3 | 34.7 | 44.0 | 53.5 | 35.7 | 38.2 |
| v4 sheet only (`pipeline_v4_no_video`) | 31.2 | 27.6 | 23.5 | 29.5 | 24.5 | 37.1 | 44.4 | 21.4 | 41.2 |

## Question by question

Better / worse: questions where the second condition scores higher / lower than the first (numeric answers by VSI-Bench's partial score); exact two-sided sign test on those, uncorrected.

| Comparison | Overall diff | Better / worse / same | p |
|---|---:|---:|---:|
| v4 sheet only − video + v4 sheet | -5.5 | 114 / 161 / 523 | 0.00544 |
| v4 sheet only − video only | -10.0 | 177 / 309 / 312 | 2.26e-09 |
| video + v4 sheet − video only | -4.5 | 145 / 247 / 406 | 2.91e-07 |

The last row is shown for context only: the registered analysis of final2 is its own write-up (`pipeline_tests.md`), not this sign test.

Per task, v4 sheet only − video + v4 sheet (what removing the video changes):

| Task | n | Video + v4 sheet | v4 sheet only | Diff | Better / worse / same | p |
|---|---:|---:|---:|---:|---:|---:|
| Count | 88 | 30.2 | 27.6 | -2.6 | 7 / 15 / 66 | 0.134 |
| Abs. dist. | 150 | 25.7 | 23.5 | -2.2 | 14 / 21 / 115 | 0.311 |
| Obj. size | 157 | 31.3 | 29.5 | -1.8 | 43 / 42 / 72 | 1 |
| Room size | 38 | 34.7 | 24.5 | -10.3 | 7 / 14 / 17 | 0.189 |
| Rel. dist. | 159 | 44.0 | 37.1 | -6.9 | 20 / 31 / 108 | 0.161 |
| Rel. dir. | 110 | 53.5 | 44.4 | -9.0 | 9 / 22 / 79 | 0.0294 |
| Route | 28 | 35.7 | 21.4 | -14.3 | 1 / 5 / 22 | 0.219 |
| Order | 68 | 38.2 | 41.2 | +2.9 | 13 / 11 / 44 | 0.839 |

## Input tokens

`pipeline_v4_no_video` records each question's own input length (`n_prompt_tokens`); final2's runs record only the padded length of each batch of up to 8 questions of one video (`n_input_tokens`), so their numbers are upper bounds.

| Condition | Tokens | Mean | Median | Min–max |
|---|---|---:|---:|---:|
| video only | `n_input_tokens` | 2583 | 2577 | 2542–2693 |
| video + v4 sheet | `n_input_tokens` | 3078 | 2913 | 2725–4538 |
| v4 sheet only | `n_prompt_tokens` | 396 | 358 | 199–2058 |

Replies cut off at 2,048 tokens: 1 of 798 (`pipeline_v4_no_video`).
