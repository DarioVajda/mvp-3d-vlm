# Structured CoT pilot: the five conditions on explore200's 200 questions

<!-- The Results block is written by `make_table.py --explore` (in this folder) from the predictions; the rest by hand. -->

**Outcome:** structured chain of thought does not help Qwen3.5-2B with the gold sheet overall (D − C = −3.3; 43
questions better, 55 worse, Holm p = 0.27) and hurts without it (B − A = −13.3, Holm p = 0.004). Precomputed
relations stay ahead (E − D = +13.9, Holm p = 0.0002). With the sheet the model does compute from the coordinates, and
its arithmetic was right in every reply read. But it computes the distance between the objects' centres, where
VSI-Bench measures between their closest points. On top of that, the prompt's "check against the video" step sends 12%
(D) to 28% (B) of the replies into frame-by-frame loops that run to the 2,048-token limit.

## What was run

- **Notebook:** `explore_cot.ipynb` on Colab, one NVIDIA L4, 2026-11-20.
- **Code:** commit `7d39234`, made by `git stash create` on `abde60f`. It differs from `abde60f` only by
  `metric_scale=None` in the notebook's `namespace()`, which the eval's `--metric-scale` option (`109018f`) needs.
- **Questions:** `explore200_ids.txt` in the format experiment's folder: 200 questions, 25 per task (relative direction
  8 easy / 8 medium / 9 hard), from 25 scenes.
- **Settings:** the structured-CoT plan's conditions A–E (README, Conditions). Qwen3.5-2B with thinking off, 16
  frames (`uniform128`), the gold sheet as csv, greedy decoding, up to 2,048 new tokens, up to 8 questions of one video
  per batch, and the final answer read by `score_final` in every condition.
- **One commit for all five:** none of explore200's predictions was copied (`reused.json` is empty; the notebook
  printed why, but its output was not kept).
- **Predictions:** `results/zeroshot/vsibench/structured_cot/explore200/`, with the notebook's own tables
  (`answers.csv`, `by_question_type.csv`, `differences_bootstrap.csv`, `per_question.csv`) and `timing.json`. The
  file-to-condition mapping is A `A_video`, B `video_cot`, C `gold_csv`, D `gold_cot`, E `gold_relations`.
- **Time:** 53 ms per decoding step, timed on a batch of 5 that ran to 2,048 tokens in 108 s. The CoT runs took
  63,773 (B) and 51,641 (D) decoding steps, about 1.7 hours together on the L4.
- **Intervals:** the notebook's bootstrap 95% intervals agree with the sign tests below: D − C −3.3 [−10.7, +4.1],
  B − A −13.3 [−20.5, −5.8], E − D +13.9 [+7.1, +20.8].

## Results

<!-- make_table:start -->
Written by `make_table.py --explore` from `results/zeroshot/vsibench/structured_cot/explore200` (200 questions per condition; scores ×100; overall: the mean over the 8 tasks present; computing: the mean of absolute distance, relative distance and object size).

| ID | Condition | Overall | Without counting | Computing | Count | Abs. dist. | Obj. size | Room size | Rel. dist. | Rel. dir. | Route | Order | Parsed | Input tokens | Reply tokens, median / p90 | At the limit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | video, direct | 48.6 | 48.4 | 61.5 | 49.6 | 39.6 | 68.8 | 39.2 | 76.0 | 47.2 | 24.0 | 44.0 | 100.0% | 2619 | 1 / 3 | 0.0% |
| B | video, CoT | 35.2 | 36.7 | 36.9 | 25.2 | 23.6 | 39.2 | 41.6 | 48.0 | 40.3 | 20.0 | 44.0 | 88.5% | 2755 | 444 / 2048 | 28.0% |
| C | + gold sheet, direct | 52.1 | 45.3 | 48.5 | 100.0 | 26.4 | 55.2 | 47.2 | 64.0 | 56.0 | 32.0 | 36.0 | 100.0% | 2909 | 1 / 4 | 0.0% |
| D | + gold sheet, CoT | 48.8 | 44.9 | 58.0 | 76.0 | 30.0 | 76.0 | 36.4 | 68.0 | 39.8 | 28.0 | 36.0 | 93.5% | 3040 | 521 / 2048 | 11.5% |
| E | + gold sheet + relations, direct | 62.7 | 57.4 | 82.0 | 100.0 | 94.8 | 55.2 | 48.0 | 96.0 | 47.7 | 28.0 | 32.0 | 100.0% | 3193 | 1 / 5 | 0.0% |

Comparisons, question by question (better / worse: questions where the first condition scores higher / lower; exact two-sided sign test; Holm over D − C, B − A and E − D; on the computing tasks' questions alone, the sign test without Holm):

| Comparison | Overall | Without counting | Computing | Better / worse / same | p | Holm p | Computing: better / worse | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D − C | -3.3 | -0.4 | +9.5 | 43 / 55 / 102 | 0.266 | 0.266 | 27 / 19 | 0.302 |
| B − A | -13.3 | -11.7 | -24.5 | 40 / 74 / 86 | 0.00187 | 0.00373 | 14 / 35 | 0.0038 |
| E − D | +13.9 | +12.5 | +24.0 | 71 / 30 / 99 | 5.53e-05 | 0.000166 | 36 / 11 | 0.000346 |
| E − C | +10.6 | +12.1 | +33.5 | 40 / 11 / 149 | 5.7e-05 | – | 33 / 1 | 4.07e-09 |
| (D − C) − (B − A) | +10.0 | +11.4 | +34.0 | – | – | – | – | – |

Per question type:

| Question type | n | A | B | C | D | E | D − C | E − D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| obj_appearance_order | 25 | 44.0 | 44.0 | 36.0 | 36.0 | 32.0 | +0.0 | -4.0 |
| object_abs_distance | 25 | 39.6 | 23.6 | 26.4 | 30.0 | 94.8 | +3.6 | +64.8 |
| object_counting | 25 | 49.6 | 25.2 | 100.0 | 76.0 | 100.0 | -24.0 | +24.0 |
| object_rel_direction_easy | 8 | 62.5 | 37.5 | 62.5 | 37.5 | 50.0 | -25.0 | +12.5 |
| object_rel_direction_hard | 9 | 66.7 | 33.3 | 55.6 | 44.4 | 55.6 | -11.1 | +11.1 |
| object_rel_direction_medium | 8 | 12.5 | 50.0 | 50.0 | 37.5 | 37.5 | -12.5 | +0.0 |
| object_rel_distance | 25 | 76.0 | 48.0 | 64.0 | 68.0 | 96.0 | +4.0 | +28.0 |
| object_size_estimation | 25 | 68.8 | 39.2 | 55.2 | 76.0 | 55.2 | +20.8 | -20.8 |
| room_size_estimation | 25 | 39.2 | 41.6 | 47.2 | 36.4 | 48.0 | -10.8 | +11.6 |
| route_planning | 25 | 24.0 | 20.0 | 32.0 | 28.0 | 28.0 | -4.0 | +0.0 |

Groups of questions (mean over the group's questions, not over tasks):

| Questions | n | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|---:|
| computing tasks | 75 | 61.5 | 36.9 | 48.5 | 58.0 | 82.0 |
| the other tasks | 125 | 41.0 | 34.2 | 54.2 | 43.3 | 51.2 |
| multiple choice | 100 | 48.0 | 38.0 | 47.0 | 43.0 | 51.0 |
| numeric | 100 | 49.3 | 32.4 | 57.2 | 54.6 | 74.5 |
| arkitscenes | 66 | 49.7 | 38.5 | 55.2 | 47.3 | 62.7 |
| scannet | 71 | 41.8 | 32.1 | 44.9 | 45.8 | 57.7 |
| scannetpp | 63 | 55.2 | 35.2 | 57.0 | 53.8 | 68.4 |

Reply length per task, CoT runs (tokens; at the limit: stopped by --max-new-tokens, 2,048 in B and 2,048 in D):

| Task | n | B median | B at the limit | D median | D at the limit |
|---|---:|---:|---:|---:|---:|
| Count | 25 | 2048 | 56% | 259 | 24% |
| Abs. dist. | 25 | 396 | 20% | 526 | 0% |
| Obj. size | 25 | 482 | 44% | 423 | 8% |
| Room size | 25 | 516 | 16% | 618 | 8% |
| Rel. dist. | 25 | 311 | 16% | 1087 | 4% |
| Rel. dir. | 25 | 592 | 20% | 554 | 4% |
| Route | 25 | 421 | 20% | 626 | 24% |
| Order | 25 | 320 | 32% | 275 | 20% |

Replies cut off at the limit (narrate: 10 or more mm:ss times, which the video's "<t seconds>" labels never use; repeat: the last 2,000 characters compress below 25%; read anyway: the final-answer rule still reads a letter or number from the unfinished reply):

| Run | Replies | Cut off | Narrate frames | Repeat | Either | Read anyway | Narrate frames, all replies |
|---|---:|---:|---:|---:|---:|---:|---:|
| B | 200 | 56 | 38 | 38 | 51 | 34 | 48 |
| D | 200 | 23 | 15 | 15 | 20 | 10 | 21 |

What the gold sheet alone answers: a fixed rule's score on the questions it applies to (every named class in the sheet), and the share of each condition's answers that are the rule's (the same letter, or a number within 5%):

| Rule on the gold sheet | Task | n | Rule's score | A | B | C | D | E |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rows of the asked class | Count | 25 | 96.0 | 36% | 20% | 96% | 76% | 96% |
| longest side of the object, in cm | Obj. size | 25 | 100.0 | 16% | 16% | 40% | 76% | 40% |
| distance between the centres | Abs. dist. | 25 | 34.0 | 4% | 12% | 0% | 84% | 0% |
| distance between the closest points | Abs. dist. | 25 | 94.8 | 8% | 0% | 0% | 0% | 100% |
| nearest candidate by centres | Rel. dist. | 23 | 100.0 | 78% | 48% | 65% | 70% | 100% |
| nearest candidate by closest points | Rel. dist. | 23 | 100.0 | 78% | 48% | 65% | 70% | 100% |

Reasoning check, D's replies:

| Replies | n | Name a sheet row | Copy a sheet number | Have an "Answer:" line |
|---|---:|---:|---:|---:|
| all | 200 | 61% | 68% | 88% |
| computing tasks | 75 | 89% | 99% | 96% |
| the other tasks | 125 | 44% | 49% | 84% |

Why D got it wrong: 20 of its 30 wrong answers on the computing tasks (score below 0.5), drawn with seed 0 and labelled by reading each reply (explore_cot_reasoning_check.md, explore_cot_reasoning_labels.csv):

| Reason | Answers |
|---|---:|
| wrong row | 4 |
| coordinates misread | 2 |
| arithmetic wrong | 0 |
| sheet ignored | 0 |
| no answer line | 3 |
| wrong method | 11 |
<!-- make_table:end -->

### Examples

**D computes the centre distance and calls it the closest-point distance** (question 837, sofa and tv; truth 1.9 m,
closest points 1.82 m, centres 3.36 m):

```
The question asks for the distance measured as the straight-line distance between the closest points of each object.
...
d = sqrt((-5.050 - (-2.140))^2 + (2.766 - 1.602)^2 + (0.421 - (-0.786))^2)
d = sqrt((-2.910)^2 + (1.164)^2 + (1.207)^2)
...
So, the straight-line distance between the closest points of the sofa and the TV is approximately 3.358 meters.

Answer: 3.358
```

**Object size: C reports another side of the box, D the longest** (question 266, the table; truth 98 cm; the sheet's
row is `table_1,table,3.522,-4.709,-0.042,0.977,0.616,0.753`). C answers `75.3`, the height. D writes:

```
    *   Length: 0.977 meters * 100 = 97.7 cm
    *   Width: 0.616 meters * 100 = 61.6 cm
    *   Height: 0.753 meters * 100 = 75.3 cm
    The longest dimension is 97.7 cm.

Answer: 97.7
```

**Counting: D lists the right rows, then "verifies with the video" until the limit** (question 4410, chairs; truth 3;
the sheet has `chair_1` to `chair_3`). After the three rows, the reply goes on:

```
2.  **Verify with the video:**
    *   Frame at 00:00 shows a desk and a computer monitor.
    *   Frame at 00:03 shows a chair with a patterned seat cushion. This matches `chair_1`.
    *   Frame at 00:04 shows a white dresser.
    ...
    *   Frame at 00:54 shows a black bag and a chair with a patterned seat cushion. This
```

It stops there, at 2,048 tokens, and the rule reads 54 from the last time it wrote. The video's own labels are
`<12.3 seconds>`, one per pair of frames; the mm:ss times are made up. Question 3530 (object size) is the same: the
right 93.6 cm first, then the check against the video loops until the limit, and the rule reads 1.

### What the results show

- **CoT without a sheet hurts.** B − A is −13.3 (40 better, 74 worse, Holm p = 0.004), and B is lower on 6 of the 8
  tasks. The worst are counting (49.6 → 25.2), object size (68.8 → 39.2) and relative distance (76.0 → 48.0). 28% of
  B's replies stop at the limit (counting 56%, object size 44%), and 51 of those 56 narrate the video frame by frame
  or repeat themselves. This matches VSI-Bench's authors' finding that linguistic prompting does not help on the
  benchmark (README, Background).
- **With the sheet, CoT neither helps nor hurts overall, but it moves the tasks in opposite directions.** D − C is −3.3
  overall and −0.4 without counting. It is +20.8 on object size, −24.0 on counting, −16.2 on relative direction and
  −10.8 on room size. On the computing tasks' mean it is +9.5 (27 better, 19 worse, p = 0.30), almost all from object
  size.
- **The model reads the sheet and computes correctly, but the wrong distance.** On the computing tasks, 99% of D's
  replies copy a sheet number and 89% name a sheet row. On absolute distance, 84% of D's answers are the centre-to-centre
  distance (within 5%) and none is the closest-point distance. Of 20 wrong D answers on the computing tasks:
  - 11 used the wrong method: the centre distance, or the gap along one axis, called the closest-point distance;
  - 4 used the wrong rows: they left out the candidate that is the answer, used one of three tables, or measured from
    the wrong object;
  - 3 were cut off at the limit;
  - 2 misread the sheet;
  - none got the arithmetic wrong.

  Centre distances are a median 37% longer than VSI-Bench's closest-point distances, so even perfect centre distances
  score 34.0. The closest-point distance scores 94.8, and E, which is given it, uses it every time. E − D on absolute
  distance (+64.8) is about which distance is computed, not whether the model can compute one.
- **Where the sheet holds the answer in the form asked, CoT uses it.** In all 25 object-size questions, the gold
  sheet's longest side is within 0.5 cm of the true size, just as the rows of a class give its count. C's answer is
  that side in 40% of the questions; 10 of its 25 answers are another side of the box. D's is that side in 76%. D's
  misses include metres read as centimetres (1.245 for 124.5 cm) and two loops.
- **Counting is where the loops cost most.** C counts the rows (96% of its answers are the row count; score 100). Six
  of D's seven wrong counting replies go through the video frame by frame until the limit, some of them after listing
  the right rows (the seventh is the `shoe` question, whose gold sheet is empty). 24% of D's counting replies are cut off, and so are 24% of route-planning and 20% of appearance-order
  replies.
- **The cut-off replies are the parse failures.** The parse rates are 88.5% (B) and 93.5% (D), below the plan's
  95%. But 22 of B's 23 unparsed replies and all 13 of D's are cut off. Every finished reply has an "Answer:" line,
  and the rule reads all of them but one: B's "Answer: N/A" on a question whose objects it did not see. Cut-off replies score 7.5 (B) and 8.7 (D), against 46.0 and 54.0 for the others. A lower limit
  would not stop the loops and would cost answers: at 1,024 tokens, 15 more of D's replies would be cut off, and
  those score 80 on average.
- **Relative distance: D's errors are choosing rows, not measuring.** In all 23 questions whose objects are all in the
  sheet, the nearest candidate by centres is the right answer. D follows that rule in 70% of them, C in 65% and A,
  from the video, in 78%. Four of D's six labelled relative-distance errors left out or mixed up rows.
- **Relations stay the strongest condition.** E − C is +10.6 (40 better, 11 worse) and E − D +13.9. E is at 94.8 on
  absolute and 96.0 on relative distance. The order is that of the 40-question pilot (A 47.3, C 54.7, E 67.3; here
  48.6, 52.1, 62.7).
- **Some tasks the gold sheet cannot help.**
  - Room size: all 25 room-size gold sheets are empty, since those questions name only `room`, so C, D and E are video
    plus the empty-sheet line there.
  - Appearance order: gold sheets have no time information; C and D are both 36.0, below A's 44.0.
  - Relative direction: the sheet gives positions but not where the viewer faces. With 25 questions, D − C (−16.2)
    is uncertain: the interval of each of its three question types includes 0.

## Conclusion

- **Answer to the question:** on these 200 questions, structured CoT does not improve how Qwen3.5-2B uses the gold
  sheet overall, and it hurts without a sheet. It does make the model compute from the coordinates, with correct
  arithmetic. Two things outside the reasoning keep that from showing in the score:
  - the sheet gives box centres, and the model computes centre distances, while VSI-Bench measures between closest
    points;
  - the prompt's last step sends a share of the replies into frame-by-frame loops.
- **Hypothesis (README):**
  - D − C > 0: not supported overall (−3.3, Holm p = 0.27). It holds on the computing tasks' mean (+9.5, p = 0.30),
    and that comes from object size alone.
  - C < D < E: D < E holds (Holm p = 0.0002); C < D does not.
  - B − A ≈ 0 or below: supported, and clearly below (−13.3, Holm p = 0.004).
  - Of the README's next steps, the case is "CoT doesn't help even with the sheet", with the caveat that on distances
    the failure is the definition of distance, which a prompt or sheet change could address.
- **Caveats:**
  - 200 questions (25 per task, 8–9 per relative-direction type), one model, one seed, greedy decoding, on an L4
    rather than the cluster.
  - A question type's 95% interval spans 30 points or more (`differences_bootstrap.csv`).
  - The 20 reasoning labels come from one reading (by Claude, reading each reply in full) and no second labeller.
  - The gold sheet contains the counting and the object-size answers, so gains there show reading, not spatial
    reasoning.
- **Next, before the full run (README, Runs):**
  - **Decide the CoT prompt first.** As written, B and D on all 2,728 questions would mostly measure the loops and
    the centre-versus-closest-point mismatch. Options:
    - run it unchanged, as a test of this prompt;
    - drop or reword the "check that the result agrees with what the video shows" step;
    - tell D how VSI-Bench measures distance (the gap along each axis, `max(0, |Δx| − (size_a + size_b) / 2)`,
      then the straight-line length of the three gaps), or write the boxes as min/max corners. This tests whether
      the model can compute what E is given.

    README TODO 4 ties prompt changes to the reasoning check. The check does not show D ignoring the sheet: it shows
    the wrong method and the loops.
  - **Keep the 2,048-token limit.** It is the loops, not the limit, that cut replies off, and a lower limit cuts off
    correct long replies.
  - **Read every gold comparison without counting and without object size,** since the sheet contains both answers
    (not only counting, as the format experiment found).
  - **Cost:** at the pilot's decoding steps per batch, B and D take about 17 hours on an L4 for all 2,728 questions
    (373 batches each) and about 7 hours for the 933 of `subset_100_per_type.txt` (151 batches). Full batches of 8
    are more likely to hold a loop than the pilot's smaller ones, so these are lower bounds. Both are over the
    plan's 4 GPU hours on this GPU.
