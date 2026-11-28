---
id: definitive-5-structured-cot
title: Does structured chain-of-thought help Qwen3.5-2B use a gold scene sheet on VSI-Bench, and does it recover what precomputed relations give
owner: Tamara Grujičić
created: 2026-11-19
status: done                 # draft | planned | running | analysing | done | abandoned
tags: [scene-sheet, gold-sheet, prompting, chain-of-thought, definitive, exploratory]
varies: [prompt.reasoning, scene_sheet.source]
depends_on: [2026-11-19-structured-cot (plan and pilot), explore200's gold sheets]
---

# Structured chain-of-thought

**Outcome:** on explore200's 200 questions, structured CoT does not help Qwen3.5-2B use the gold sheet (D − C −3.3,
Holm p 0.27) and hurts without it (B − A −13.3, Holm p 0.004); precomputed relations stay ahead (E − D +13.9, Holm
p 0.0002). With the sheet the model computes centre distances, where VSI-Bench measures closest points, and 12–28% of
the CoT replies loop to the token limit.

This file describes the structured-CoT pilot as it was run (2026-11-20; `experiments/2026-11-19-structured-cot/`,
write-up `explore_cot.md`). **Exploratory, not pre-registered:** the questions are explore200's, the format
experiment's exploration set, and this pilot's replies are what experiments 6 and 7's prompts were developed from. The
hypothesis and comparisons below were written in the structured-cot README on 2026-11-19, before the pilot; the
README's planned run on 2,728 questions was not made. The settings are the ones the pilot used, not this folder's
`README.md` section 2.

## Question

Does step-by-step reasoning help Qwen3.5-2B use a correct scene sheet, and does it recover what precomputed relations
give it?

## Hypothesis and what would change our mind

As written before the pilot (structured-cot README):

- **Expected:**
  - D − C > 0: the sheet supplies the numbers, and the model can copy two objects' coordinates and compute their
    distance in writing.
  - C < D < E: CoT recovers part of the gap to the relations, not all, because a 2B model computes less reliably than
    the relations state.
  - B − A ≈ 0 or below: without a sheet, reasoning adds little, as VSI-Bench's authors report for linguistic prompting.
- **Would prove it wrong:**
  - D − C ≤ 0, overall and on the computing tasks (absolute distance, relative distance, object size);
  - D ≥ E: computing from the sheet is as good as being given the relations;
  - B − A clearly positive (Holm p < 0.05 and above the smallest difference that matters), which would point to the
    prompt rather than the sheet.
- **Next:**
  - CoT helps and closes much of the gap to E: try it with the pipeline sheet;
  - it helps but leaves most of the gap: precomputed relations are the better way to give spatial facts;
  - it doesn't help: precomputed relations are the way forward; try a larger model only if relations alone are not
    enough.

## Background

- **The prompt:** `--reasoning cot` (`vsibench.cot_prompt`, text in the structured-cot folder's `cot_prompt.md`,
  generated at `6642c48`). It asks the model to say which objects the question is about, write down the facts it
  needs, work out the answer one step at a time showing the arithmetic, check the result against the video, and end
  on "Answer: X". Each task adds one hint (`vsibench.COT_HINTS`).
- **Prior work:** `experiments/2026-11-25-concise-math-cot/README.md`, Background.
- **The direct conditions on these questions:** C and E equal, byte for byte, B and C of the gold-sheet experiment's
  explore200 preview on the rebuilt gold sheets (`2026-11-19-gold-sheet-relations`, Log, 2026-11-24).

## Design

### Conditions

| ID | Condition | `prompt.reasoning` | `scene_sheet.source` | Role | Run |
|---|---|---|---|---|---|
| A | Video only, direct | direct | none | baseline | `A_video.jsonl` |
| B | Video only, CoT | cot | none | treatment | `video_cot.jsonl` |
| C | Video + gold sheet, direct | direct | gold | control | `gold_csv.jsonl` |
| D | Video + gold sheet, CoT | cot | gold | treatment | `gold_cot.jsonl` |
| E | Video + gold sheet + relations, direct | direct | gold_relations | upper reference | `gold_relations.jsonl` |

All in `results/zeroshot/vsibench/structured_cot/explore200/`, run at one commit; none reused.

### Comparisons

- **One Holm family:** D − C (primary), B − A and E − D.
- **Descriptive:** E − C, and (D − C) − (B − A).
- Each also without counting, and on the computing tasks alone (sign test, no Holm).

### Decisions

| Decision | Choice | Why |
|---|---|---|
| Questions | `explore200_ids.txt` (format experiment): 200 questions, 25 per task (relative direction 8 easy / 8 medium / 9 hard), 25 scenes | A pilot before the planned 2,728-question run, on the exploration set |
| CoT prompt | `--reasoning cot`, unchanged | It has the four steps and an "Answer: X" line that `score_final` reads |
| Sheet | gold, csv | Isolates reasoning from detection errors |
| Relations | closest-point distances (`gold_relations`) | VSI-Bench measures between closest points |
| Token limit | 2,048 new tokens in every condition | The eval's CoT default; the direct runs use the same limit with `--final-answer` |
| Thinking mode | off, no condition for it | dropped on 2026-11-20; the reasoning comes from the prompt |

## Configuration

```yaml
data:
  benchmark: vsibench
  split: full
  subset: experiments/2026-11-18-scene-sheet-formats/explore200_ids.txt
  scenes: explore200's 25
  question_types: all          # all 10
  n_questions: 200

visual:
  video: true
  model: Qwen/Qwen3.5-2B
  protocol: video
  nframes: 16
  sampling: uniform128
  layout: sit
  resolution: default

scene_sheet:
  source: varies               # none (A, B) | gold (C, D) | gold_relations (E)
  format: csv
  placement: between video and question

prompt:
  system: You are a helpful assistant.
  template: lmms-eval
  reasoning: varies            # direct (A, C, E) | cot (B, D)
  thinking: off
  max_new_tokens: 2048
  decoding: greedy
  answer_extraction: score_final
  batches: up to 8 questions of one video

training: none
```

## Metrics and analysis

`make_table.py --explore` (structured-cot folder): VSI-Bench overall, without counting, the computing score (the mean
of absolute distance, relative distance and object size), per task and question type, parse rate, input tokens, reply
length and the share at the limit; cut-off replies that loop (10 or more mm:ss times, or a last 2,000 characters that
compress below 25%); what fixed rules on the gold sheet answer; the reasoning check (D's replies that name a sheet row
or copy a sheet number). 20 of D's 30 wrong computing answers (score below 0.5), drawn with seed 0, were labelled by
reading each reply (`explore_cot_reasoning_labels.csv`; one labeller, Claude, no second). The notebook's paired
bootstrap intervals are reported beside the sign tests.

## Threats to validity and checks

| Risk | How we'd notice | What to do |
|---|---|---|
| 200 questions, 25 per task | intervals of 30 points or more per type | read per-task differences as descriptive |
| CoT replies loop until the limit and lose for that alone | the share cut off and looping | report it; the loops are the parse failures |
| The gold sheet contains the counting answer, and its largest side is the object-size answer | counting and object size near 100 | read without counting, and object size apart |
| The relations state the answer to most distance questions | E near its ceiling there | E is an upper reference for using given relations, not a test of reasoning |
| explore200 is the exploration set | — | these results shaped later prompts (experiments 6, 7); they are development results |

## Runs and cost

`explore_cot.ipynb` on Colab, one NVIDIA L4, 2026-11-20, at commit `7d39234` (`abde60f` plus `metric_scale=None` in
the notebook's `namespace()`). The five conditions ran in one session; `reused.json` is empty.

**Cost:** 53 ms per decoding step; the CoT runs took 63,773 (B) and 51,641 (D) decoding steps, about 1.7 hours
together on the L4.

## Results

From `explore_cot.md` (written by `make_table.py --explore`; its per-type, group, length, loop, rule and reasoning
tables are there):

| ID | Condition | Overall | Without counting | Computing | Count | Abs. dist. | Obj. size | Room size | Rel. dist. | Rel. dir. | Route | Order | Parsed | Reply tokens, median / p90 | At the limit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | video, direct | 48.6 | 48.4 | 61.5 | 49.6 | 39.6 | 68.8 | 39.2 | 76.0 | 47.2 | 24.0 | 44.0 | 100.0% | 1 / 3 | 0.0% |
| B | video, CoT | 35.2 | 36.7 | 36.9 | 25.2 | 23.6 | 39.2 | 41.6 | 48.0 | 40.3 | 20.0 | 44.0 | 88.5% | 444 / 2048 | 28.0% |
| C | + gold sheet, direct | 52.1 | 45.3 | 48.5 | 100.0 | 26.4 | 55.2 | 47.2 | 64.0 | 56.0 | 32.0 | 36.0 | 100.0% | 1 / 4 | 0.0% |
| D | + gold sheet, CoT | 48.8 | 44.9 | 58.0 | 76.0 | 30.0 | 76.0 | 36.4 | 68.0 | 39.8 | 28.0 | 36.0 | 93.5% | 521 / 2048 | 11.5% |
| E | + gold sheet + relations, direct | 62.7 | 57.4 | 82.0 | 100.0 | 94.8 | 55.2 | 48.0 | 96.0 | 47.7 | 28.0 | 32.0 | 100.0% | 1 / 5 | 0.0% |

| Comparison | Overall | Without counting | Computing | Better / worse / same | Holm p | 95% interval (overall) | Computing: better / worse, p |
|---|---:|---:|---:|---:|---:|---:|---:|
| D − C | −3.3 | −0.4 | +9.5 | 43 / 55 / 102 | 0.266 | [−10.7, +4.1] | 27 / 19, 0.30 |
| B − A | −13.3 | −11.7 | −24.5 | 40 / 74 / 86 | 0.0037 | [−20.5, −5.8] | 14 / 35, 0.0038 |
| E − D | +13.9 | +12.5 | +24.0 | 71 / 30 / 99 | 0.00017 | [+7.1, +20.8] | 36 / 11, 0.00035 |
| E − C (descriptive) | +10.6 | +12.1 | +33.5 | 40 / 11 / 149 | – | – | 33 / 1, 4e-9 |

### Examples

In `explore_cot.md`, Examples: D computing the centre distance and calling it the closest-point distance (question
837: 3.358 m against a truth of 1.9); C giving the table's height and D its longest side (question 266); D listing the
right chair rows, then "verifying with the video" in invented mm:ss frames until the limit (question 4410).

### What the results show

- **CoT without a sheet hurts:** B − A −13.3; 28% of B's replies stop at the limit, 51 of those 56 narrating frames or
  repeating.
- **With the sheet, CoT moves tasks in opposite directions:** D − C +20.8 on object size, −24.0 on counting, −16.2 on
  relative direction, −10.8 on room size; +9.5 on the computing tasks' mean, almost all from object size.
- **The model computes from the sheet, but the wrong distance:** on the computing tasks 99% of D's replies copy a sheet
  number and 89% name a row; 84% of D's absolute-distance answers are the centre distance and none the closest-point
  one. Of 20 labelled wrong answers: 11 wrong method, 4 wrong rows, 3 cut off, 2 misread, none wrong arithmetic.

## Conclusion

- **Answer to the question:** on these 200 questions, structured CoT does not improve how Qwen3.5-2B uses the gold
  sheet overall, and it hurts without one. The model computes from the coordinates with correct arithmetic, but
  computes centre distances, and the prompt's check against the video sends replies into loops.
- **Hypothesis:** D − C > 0 not supported overall (−3.3, Holm p 0.27), only on the computing mean (+9.5, p 0.30, from
  object size); D < E holds, C < D does not; B − A ≤ 0 supported, clearly below (−13.3).
- **Caveats:** exploratory: 200 development questions, one model, one seed, greedy decoding, on an L4; one labeller
  for the reasoning check; the gold sheet contains the counting and object-size answers.
- **Next:** the failure analysis (wrong distance, loops) led to the concise and mathematical variants (experiment 6)
  and the model-size check (experiment 7).

## Log

- 2026-11-19: plan, hypothesis and comparisons written (structured-cot README), for a run on 2,728 questions.
- 2026-11-20: thinking-mode condition dropped; the pilot ran A–E on explore200 at `7d39234` (structured-cot README,
  Log).
- 2026-11-28: this file rewritten to describe the pilot as run, with the 2B; the 4B plan and its open questions
  removed.
