# Concise and mathematical CoT: results

<!-- Written by analysis.py; edit that, not this file. -->

**Exploratory: 3 questions (1 per task).** At this size the intervals are wide and nothing here is an established difference; it is for checking the runs and the prompts.

Questions: 3 (object_abs_distance 1, object_size_estimation 1, object_rel_distance 1), the same for every condition. A, C, D, E: the structured-CoT pilot's predictions (2026-11-19-structured-cot), not run again. Scores x100 (score_final).

| Condition | What | Computing | Abs. dist. | Rel. dist. | Obj. size | Parsed | Reply tokens, median / p90 | Cut off | s / question |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | video only, direct | 76.7 | 100.0 | 100.0 | 30.0 | 100.0% | 3 / 3 | 0.0% | - |
| C | gold sheet, direct | 83.3 | 90.0 | 100.0 | 60.0 | 100.0% | 4 / 4 | 0.0% | - |
| D | gold sheet, structured CoT (pilot) | 6.7 | 20.0 | 0.0 | 0.0 | 100.0% | 1196 / 1878 | 33.3% | - |
| E | gold sheet + relations, direct | 86.7 | 100.0 | 100.0 | 60.0 | 100.0% | 4 / 5 | 0.0% | - |
| F1 | gold sheet, concise CoT | 26.7 | 20.0 | 0.0 | 60.0 | 100.0% | 183 / 825 | 0.0% | 158.5 |
| F2 | gold sheet, + geometric formulas | 33.3 | 0.0 | 0.0 | 100.0 | 100.0% | 404 / 1719 | 33.3% | 150.7 |
| F3 | gold sheet, + worked examples | 0.0 | 0.0 | 0.0 | 0.0 | 100.0% | 235 / 299 | 0.0% | 122.5 |

Comparisons, question by question (better / worse: questions where the first condition scores higher / lower; exact two-sided sign test; Holm within each family; paired bootstrap 95% interval of the computing score's difference, 10,000 resamples within question types):

**primary: each intervention**

| Comparison | Computing | 95% interval | Abs. dist. | Rel. dist. | Obj. size | Better / worse / same | p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F1 - D (concise reasoning) | +20.0 | [+20.0, +20.0] | +0.0 | +0.0 | +60.0 | 1 / 0 / 2 | 1 | 1 |
| F2 - F1 (explicit formulas) | +6.7 | [+6.7, +6.7] | -20.0 | +0.0 | +40.0 | 1 / 1 / 1 | 1 | 1 |
| F3 - F2 (worked examples) | -33.3 | [-33.3, -33.3] | +0.0 | +0.0 | -100.0 | 0 / 1 / 2 | 1 | 1 |

**against direct answering with the sheet (C)**

| Comparison | Computing | 95% interval | Abs. dist. | Rel. dist. | Obj. size | Better / worse / same | p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F1 - C | -56.7 | [-56.7, -56.7] | -70.0 | -100.0 | +0.0 | 0 / 2 / 1 | 0.5 | 1 |
| F2 - C | -50.0 | [-50.0, -50.0] | -90.0 | -100.0 | +40.0 | 1 / 2 / 0 | 1 | 1 |
| F3 - C | -83.3 | [-83.3, -83.3] | -90.0 | -100.0 | -60.0 | 0 / 3 / 0 | 0.25 | 0.75 |

**against precomputed relations (E)**

| Comparison | Computing | 95% interval | Abs. dist. | Rel. dist. | Obj. size | Better / worse / same | p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F1 - E | -60.0 | [-60.0, -60.0] | -80.0 | -100.0 | +0.0 | 0 / 2 / 1 | 0.5 | 1 |
| F2 - E | -53.3 | [-53.3, -53.3] | -100.0 | -100.0 | +40.0 | 1 / 2 / 0 | 1 | 1 |
| F3 - E | -86.7 | [-86.7, -86.7] | -100.0 | -100.0 | -60.0 | 0 / 3 / 0 | 0.25 | 0.75 |

**against video only (A)**

| Comparison | Computing | 95% interval | Abs. dist. | Rel. dist. | Obj. size | Better / worse / same | p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F1 - A | -50.0 | [-50.0, -50.0] | -80.0 | -100.0 | +30.0 | 1 / 2 / 0 | 1 | 1 |
| F2 - A | -43.3 | [-43.3, -43.3] | -100.0 | -100.0 | +70.0 | 1 / 2 / 0 | 1 | 1 |
| F3 - A | -76.7 | [-76.7, -76.7] | -100.0 | -100.0 | -30.0 | 0 / 3 / 0 | 0.25 | 0.75 |

Automatic error flags (share of each CoT condition's replies; errors.csv has them per reply, with empty columns for what needs reading: object identification, attribute extraction, operation, arithmetic, unit). A flag such as 'equals the centre rule' says what number the reply ended on, not how it got there.

| Flag | D | F1 | F2 | F3 |
|---|---:|---:|---:|---:|
| cut_off | 33% | 0% | 33% | 0% |
| no_answer_line | 33% | 0% | 33% | 0% |
| unread | 0% | 0% | 0% | 0% |
| sheet_rule_disagrees_with_truth | 0% | 0% | 0% | 0% |
| named_objects_missing_from_sheet | 0% | 0% | 0% | 0% |
