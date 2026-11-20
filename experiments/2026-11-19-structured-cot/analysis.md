# Structured CoT pilot: analysis

<!-- Written by analysis.py in this folder; edit that, not this file. -->

The predictions of `explore_cot.ipynb` (`results/zeroshot/vsibench/structured_cot/explore200/`), analysed as the README's Metrics and analysis section specifies: 200 questions (`explore200_ids.txt`, 25 per task), Qwen3.5-2B, 16 frames (`uniform128`), 2,048 new tokens, final answer scored, gold sheets as csv, on one L4. Bootstrap intervals: questions resampled within each question type, the same resample for every condition (10,000 resamples, seed 0).

## Scores

| | Condition | Overall | w/o counting | Computing tasks | Count | Abs. dist. | Obj. size | Room size | Rel. dist. | Rel. dir. | Route | Order |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | video only, direct | 48.6 | 48.4 | 61.5 | 49.6 | 39.6 | 68.8 | 39.2 | 76.0 | 47.2 | 24.0 | 44.0 |
| B | video only, structured CoT | 35.2 | 36.7 | 36.9 | 25.2 | 23.6 | 39.2 | 41.6 | 48.0 | 40.3 | 20.0 | 44.0 |
| C | gold sheet, direct | 52.1 | 45.3 | 48.5 | 100.0 | 26.4 | 55.2 | 47.2 | 64.0 | 56.0 | 32.0 | 36.0 |
| D | gold sheet, structured CoT | 48.8 | 44.9 | 58.0 | 76.0 | 30.0 | 76.0 | 36.4 | 68.0 | 39.8 | 28.0 | 36.0 |
| E | gold sheet + relations, direct | 62.7 | 57.4 | 82.0 | 100.0 | 94.8 | 55.2 | 48.0 | 96.0 | 47.7 | 28.0 | 32.0 |

Computing tasks: the mean of absolute distance, relative distance and object size. Per question type (x100, 25 questions each; relative direction 8, 8 and 9):

| Question type | n | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|---:|
| object_counting | 25 | 49.6 | 25.2 | 100.0 | 76.0 | 100.0 |
| object_abs_distance | 25 | 39.6 | 23.6 | 26.4 | 30.0 | 94.8 |
| object_size_estimation | 25 | 68.8 | 39.2 | 55.2 | 76.0 | 55.2 |
| room_size_estimation | 25 | 39.2 | 41.6 | 47.2 | 36.4 | 48.0 |
| object_rel_distance | 25 | 76.0 | 48.0 | 64.0 | 68.0 | 96.0 |
| object_rel_direction_easy | 8 | 62.5 | 37.5 | 62.5 | 37.5 | 50.0 |
| object_rel_direction_medium | 8 | 12.5 | 50.0 | 50.0 | 37.5 | 37.5 |
| object_rel_direction_hard | 9 | 66.7 | 33.3 | 55.6 | 44.4 | 55.6 |
| route_planning | 25 | 24.0 | 20.0 | 32.0 | 28.0 | 28.0 |
| obj_appearance_order | 25 | 44.0 | 44.0 | 36.0 | 36.0 | 32.0 |

## The main comparisons

Question by question: better / worse / same, an exact two-sided sign test, Holm over the three; and the difference in overall, without counting and on the computing tasks, with bootstrap 95% intervals.

| Comparison | Question | Better / worse / same | p | Holm p | Overall | w/o counting | Computing |
|---|---|---:|---:|---:|---:|---:|---:|
| D − C | does CoT help with a sheet | 43 / 55 / 102 | 0.266 | 0.266 | -3.3 [-10.5, +4.0] | -0.4 [-8.4, +7.6] | +9.5 [-3.9, +22.8] |
| B − A | does CoT help without a sheet | 40 / 74 / 86 | 0.00187 | 0.00373 | -13.3 [-20.7, -5.8] | -11.7 [-19.7, -3.6] | -24.5 [-36.0, -13.1] |
| E − D | do relations beat computing them in writing | 71 / 30 / 99 | 5.53e-05 | 0.000166 | +13.9 [+7.0, +20.6] | +12.5 [+5.1, +20.0] | +24.0 [+14.0, +33.9] |

## Replies

Parse rate: the share of replies the final-answer rule reads (`parsed(..., final=True)`). Reply length in new tokens, as the eval records it.

| | Condition | Parsed | Median length | 90th percentile | Hit the 2,048-token limit |
|---|---|---:|---:|---:|---:|
| A | video only, direct | 100.0% | 1 | 3 | 0 (0%) |
| B | video only, structured CoT | 88.5% | 444 | 2048 | 56 (28%) |
| C | gold sheet, direct | 100.0% | 1 | 4 | 0 (0%) |
| D | gold sheet, structured CoT | 93.5% | 521 | 2048 | 23 (12%) |
| E | gold sheet + relations, direct | 100.0% | 1 | 5 | 0 (0%) |

## Reasoning check (D)

Over D's 172 replies whose gold sheet has rows: a reply **names a row** if it contains one of the sheet's ids (such as `chair_2`), and **copies a number** if it contains one of the sheet's positions or extents exactly as the sheet writes them (three decimals).

| Replies | n | Name a row | Copy a number | Right |
|---|---:|---:|---:|---:|
| all | 172 | 63% | 77% | 52% |
| computing tasks | 75 | 83% | 97% | 60% |
| computing tasks, not cut off | 72 | 82% | 99% | 62% |
| cut off at the limit | 19 | 58% | 53% | 0% |

(right: the exact option for multiple choice, a score of at least 0.5 (within 25% of the answer) for numbers.)

### Three of D's replies

The first by question id of each kind: a right and a wrong computing-task reply that copies the sheet's numbers, and the first reply cut off at the limit (shortened in the middle).

**right**, question 266 (object_size_estimation): What is the length of the longest dimension (length, width, or height) of the table, measured in centimeters?  
ground truth 98; score 1.00; 485 tokens

```
The question asks for the length of the longest dimension of the table, measured in centimeters.

1.  **Identify the object:** The object in question is the table, labeled as "table_1" in the scene sheet.
2.  **Find the object in the video:** The table is visible in several frames, particularly at the beginning (00:00-00:02) and again around 00:23. It is a small, gold-framed side table with a glass top.
3.  **Find the object in the scene information:** The scene sheet provides the following details for "table_1":
    *   Position (x, y, z): 3.522, -4.709, -0.042
    *   Size (x, y, z): 0.977, 0.616, 0.753
4.  **Interpret the size values:** The size values are given in the same units as the p
[... 398 characters ...]
100 = 61.6 cm
    *   Height: 0.753 meters * 100 = 75.3 cm
6.  **Determine the longest dimension:** Comparing the converted values:
    *   97.7 cm (Length)
    *   61.6 cm (Width)
    *   75.3 cm (Height)
    The longest dimension is 97.7 cm.

The table's longest dimension is 97.7 cm.

Answer: 97.7
```

**wrong**, question 289 (object_size_estimation): What is the length of the longest dimension (length, width, or height) of the sofa, measured in centimeters?  
ground truth 237; score 0.00; 384 tokens

```
The question asks for the length of the longest dimension of the sofa.

1.  **Identify the object:** The object is the "sofa_1" mentioned in the scene sheet.
2.  **Find the object in the video:** The sofa is visible in multiple frames, particularly from 00:23 to 00:30 and 01:13 to 01:16. It is a dark brown sofa located in the living room area.
3.  **Extract size information from the scene sheet:** The scene sheet provides the dimensions for "sofa_1" as:
    *   size_x: 2.152 metres
    *   size_y: 5.895 metres
    *   size_z: 0.980 metres
4.  **Convert to centimeters:** The question asks for the answer in centimeters.
    *   2.152 metres * 100 = 215.2 centimeters
    *   5.895 metres * 100 
[... 153 characters ...]
cm (length)
    *   589.5 cm (width)
    *   98.0 cm (height)
    The longest dimension is 589.5 cm.

The video shows a living room with a large sofa. The scene sheet provides the dimensions in metres. Converting these to centimeters and identifying the maximum value gives the answer.

Answer: 589.5
```

**cut off**, question 1500 (object_rel_distance): Measuring from the closest point of each object, which of these objects (chair, sofa, refrigerator, stove) is the closest to the tv?  
ground truth B; score 0.00; 2048 tokens; hit the limit

```
The question asks which object is closest to the TV.

First, let's identify the positions of the relevant objects from the scene sheet:
- **tv_1**: located at coordinates (0.607, -0.951, 0.441).
- **sofa_1**: located at coordinates (1.434, 0.874, -0.428).
- **sofa_2**: located at coordinates (0.334, 1.569, -0.418).
- **sofa_3**: located at coordinates (-1.521, 1.227, -0.425).
- **stove_1**: located at coordinates (-3.052, 3.781, 0.083).
- **chair_1**: located at coordinates (-0.213, 2.955, -0.248).
- **chair_2**: located at coordinates (0.288, 2.923, -0.265).
- **chair_3**: located at coordinates (0.679, 3.526, -0.263).
- **chair_4**: located at coordinates (-0.417, 3.478, -0.266).

Now, we 
[... 2,891 characters ...]
.
    -   The x-distance is |0.679 - 0.607| = 0.072 meters.
    -   The y-distance is |3.526 - (-0.951)| = 4.477 meters.
    -   The z-distance is |-0.263 - 0.441| = 0.704 meters.
    -   Using the Euclidean distance formula: sqrt(0.072^2 + 4.477^2 + 0.704^2) ≈ sqrt(0.005 + 19.045 + 0.496) ≈ sqrt(20
```

## Which of the README's next steps

D − C is -3.3 overall (Holm p = 0.27; interval [-10.5, +4.0]), against a gap from C to E of +10.6. The README's case is therefore **"CoT doesn't help even with the sheet"**, whose next step is: precomputed relations are the way forward; stop prompting the 2B to compute from coordinates, and try a larger model only if relations alone are not enough.
