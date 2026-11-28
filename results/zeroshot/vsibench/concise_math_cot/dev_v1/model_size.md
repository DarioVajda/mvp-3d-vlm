# Model size: Qwen3.5-9B against Qwen3.5-2B

<!-- Written by analysis.py --model-size. -->

**Exploratory: 30 questions (10 per task), development questions (explore200), on which the prompts were designed.** Nothing here is an established difference.

Questions: 30 (object_size_estimation 10, object_abs_distance 10, object_rel_distance 10), the same for every row. The same prompts, frames, decoding, token limit, batches and answer rule for both models; the 2B's A, C, D and E are the structured-CoT pilot's predictions. Scores x100 (score_final).

| Model | Condition | What | Computing | Abs. dist. | Rel. dist. | Obj. size | Parsed | Reply tokens, median / p90 | Cut off | s / question |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2B | A | video only, direct | 57.0 | 43.0 | 70.0 | 58.0 | 100.0% | 3 / 3 | 0.0% | - |
| 2B | C | gold sheet, direct | 52.0 | 27.0 | 70.0 | 59.0 | 100.0% | 4 / 5 | 0.0% | - |
| 2B | D | gold sheet, structured CoT (pilot) | 66.7 | 40.0 | 80.0 | 80.0 | 100.0% | 536 / 1548 | 3.3% | - |
| 2B | E | gold sheet + relations, direct | 85.3 | 97.0 | 100.0 | 59.0 | 100.0% | 4 / 5 | 0.0% | - |
| 2B | F1 | gold sheet, concise CoT | 64.7 | 45.0 | 70.0 | 79.0 | 100.0% | 143 / 463 | 0.0% | 24.1 |
| 9B | A | video only, direct | 53.3 | 54.0 | 60.0 | 46.0 | 100.0% | 2 / 3 | 0.0% | 4.0 |
| 9B | C | gold sheet, direct | 80.0 | 71.0 | 70.0 | 99.0 | 100.0% | 4 / 4 | 0.0% | 4.2 |
| 9B | E | gold sheet + relations, direct | 98.7 | 97.0 | 100.0 | 99.0 | 100.0% | 4 / 5 | 0.0% | 4.2 |
| 9B | F1 | gold sheet, concise CoT | 76.3 | 49.0 | 80.0 | 100.0 | 100.0% | 174 / 952 | 0.0% | 27.7 |
| 9B | F2 | gold sheet, + geometric formulas | 79.3 | 98.0 | 40.0 | 100.0 | 83.3% | 406 / 2048 | 16.7% | 46.9 |

s / question includes loading the model once per call. Contrasts, question by question (better / worse: questions where the contrast is above / below 0; exact two-sided sign test; Holm over the family; paired bootstrap 95% interval, 10,000 resamples within question types):

| Comparison | Question | Computing | 95% interval | Abs. dist. | Rel. dist. | Obj. size | Better / worse / same | p | Holm p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 9B F1 - 9B C | does CoT help the 9B? | -3.7 | [-17.0, +9.0] | -22.0 | +10.0 | +1.0 | 4 / 9 / 17 | 0.267 | 0.534 |
| 2B F1 - 2B C | does CoT help the 2B? | +12.7 | [-5.3, +31.0] | +18.0 | +0.0 | +20.0 | 10 / 6 / 14 | 0.454 | 0.534 |
| (9B F1 - 9B C) - (2B F1 - 2B C) | does model size change the effect of CoT? | -16.3 | [-35.0, +1.3] | -40.0 | +10.0 | -19.0 | 6 / 13 / 11 | 0.167 | 0.501 |
| 9B C - 2B C | the sheet, direct | +28.0 | [+13.0, +42.7] | +44.0 | +0.0 | +40.0 | 15 / 4 / 11 | 0.0192 | 0.0961 |
| 9B E - 2B E | the sheet and relations, direct | +13.3 | [+5.7, +21.0] | +0.0 | +0.0 | +40.0 | 6 / 1 / 23 | 0.125 | 0.5 |
