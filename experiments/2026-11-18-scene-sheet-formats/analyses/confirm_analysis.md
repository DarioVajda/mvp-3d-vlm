## 1. The pre-registered result, recomputed from the predictions

- relative-direction questions: 177; helped 28, hurt 21, unchanged 128
- exact two-sided sign test: p = 0.391603 -> NOT CONFIRMED
- relative-direction score: A_video 42.32, pipeline_v3 46.76, difference +4.44, 95% bootstrap [-3.97, +12.89]
- easy: n 41, A_video 46.3, pipeline_v3 53.7, helped/hurt 10/7
- medium: n 69, A_video 49.3, pipeline_v3 47.8, helped/hurt 9/10
- hard: n 67, A_video 31.3, pipeline_v3 38.8, helped/hurt 9/4
- the notebook's preregistered_test.csv: helped 28, hurt 21, p 0.391603, NOT CONFIRMED, difference +4.44 [-4.04, +12.71]

## 2. Answer letters (descriptive)

easy (41 questions):

|  | A | B | other |
|---|---|---|---|
| truth | 22 | 19 | 0 |
| A_video | 26 | 15 | 0 |
| pipeline_v3 | 9 | 32 | 0 |

medium (69 questions):

|  | A | B | C | other |
|---|---|---|---|---|
| truth | 19 | 26 | 24 | 0 |
| A_video | 35 | 27 | 7 | 0 |
| pipeline_v3 | 23 | 36 | 10 | 0 |

hard (67 questions):

|  | A | B | C | D | other |
|---|---|---|---|---|---|
| truth | 18 | 12 | 16 | 21 | 0 |
| A_video | 26 | 18 | 18 | 5 | 0 |
| pipeline_v3 | 15 | 24 | 24 | 4 | 0 |

helped (28): A_video's (wrong) answer was {'A': 23, 'B': 1, 'C': 1, 'D': 3}; 'A' in 23 of 28 (A_video answers 'A' in 87 of 177 questions overall, the truth is 'A' in 59)
hurt (21): pipeline_v3's (wrong) answer was {'A': 2, 'B': 14, 'C': 4, 'D': 1}

## 3. Sheet quality (descriptive)

geometry check: the gold sheet's positions imply the true answer in 170 of 170 questions (where gold has one row per named object)
| sheet | n | A_video | pipeline_v3 | helped / hurt | sheet implies the truth | v3 answers what the sheet implies |
|---|---|---|---|---|---|---|
| all present, each once | 7 | 42.9 | 42.9 | 0 / 0 | 7 / 7 | 3 / 7 |
| all present, duplicated | 87 | 44.8 | 42.5 | 12 / 14 | 50 / 87 | 35 / 87 |
| some missing | 83 | 38.6 | 49.4 | 16 / 7 | 0 / 0 | 0 / 0 |

by what the sheet's positions imply:

|  | n | A_video | pipeline_v3 | helped / hurt |
|---|---|---|---|---|
| sheet implies the truth | 57 | 47.4 | 45.6 | 6 / 7 |
| sheet implies a wrong answer | 37 | 40.5 | 37.8 | 6 / 7 |
| sheet cannot imply an answer (object missing) | 83 | 38.6 | 49.4 | 16 / 7 |

## 4. Per dataset and sub-type (descriptive)

| dataset | sub-type | n | A_video | pipeline_v3 | helped / hurt |
|---|---|---|---|---|---|
| arkitscenes | easy | 17 | 52.9 | 58.8 | 3 / 2 |
| arkitscenes | medium | 26 | 34.6 | 57.7 | 6 / 0 |
| arkitscenes | hard | 30 | 40.0 | 43.3 | 2 / 1 |
| arkitscenes | all | 73 | 41.1 | 52.1 | 11 / 3 |
| scannet | easy | 16 | 43.8 | 56.2 | 5 / 3 |
| scannet | medium | 29 | 65.5 | 44.8 | 2 / 8 |
| scannet | hard | 17 | 29.4 | 35.3 | 2 / 1 |
| scannet | all | 62 | 50.0 | 45.2 | 9 / 12 |
| scannetpp | easy | 8 | 37.5 | 37.5 | 2 / 2 |
| scannetpp | medium | 14 | 42.9 | 35.7 | 1 / 2 |
| scannetpp | hard | 20 | 20.0 | 35.0 | 5 / 2 |
| scannetpp | all | 42 | 31.0 | 35.7 | 8 / 6 |

## 5. Examples: the first 5 helped and the first 5 hurt, by question id

**helped**

### 988 (arkitscenes/42446529, hard): some missing
If I am standing by the bed and facing the chair, is the table to my front-left, front-right, back-left, or back-right?
options ['A. front-right', 'B. back-left', 'C. back-right', 'D. front-left']; truth D; A_video A; pipeline_v3 D; sheet implies - (no answer); gold implies D
  bed: 3 sheet row(s), gold 1
    (-0.10, 0.23, 0.51) m, size 1.41x1.15x0.69, 5 frame(s)
    (0.07, -1.00, 0.59) m, size 1.07x0.66x0.32, 1 frame(s)
    (-0.44, 0.09, 0.50) m, size 0.54x0.65x0.76, 3 frame(s)
  chair: 0 sheet row(s), gold 1
  table: 1 sheet row(s), gold 1
    (-0.65, 0.46, 0.40) m, size 0.42x0.25x0.12, 1 frame(s)

### 1099 (arkitscenes/47430468, hard): all present, duplicated
If I am standing by the tv and facing the refrigerator, is the stool to my front-left, front-right, back-left, or back-right?
options ['A. back-left', 'B. front-right', 'C. back-right', 'D. front-left']; truth B; A_video D; pipeline_v3 B; sheet implies B (ambiguous); gold implies B
  tv: 1 sheet row(s), gold 1
    (1.09, -0.38, -0.30) m, size 0.05x0.93x0.37, 1 frame(s)
  refrigerator: 3 sheet row(s), gold 1
    (-1.32, 2.44, 0.25) m, size 0.08x0.66x1.20, 1 frame(s)
    (-1.49, 2.93, 0.32) m, size 0.59x0.11x0.48, 1 frame(s)
    (0.80, 0.39, 0.84) m, size 0.03x0.29x0.38, 1 frame(s)
  stool: 3 sheet row(s), gold 1
    (-1.74, 1.65, 0.73) m, size 0.32x0.44x0.30, 1 frame(s)
    (-0.39, -0.24, 0.49) m, size 0.88x0.90x0.49, 1 frame(s)
    (-1.84, 0.24, 0.12) m, size 0.10x0.13x0.05, 1 frame(s)

### 1140 (arkitscenes/42446529, medium): some missing
If I am standing by the bed and facing the chair, is the table to my left, right, or back?
options ['A. right', 'B. back', 'C. left']; truth C; A_video A; pipeline_v3 C; sheet implies - (no answer); gold implies C
  bed: 3 sheet row(s), gold 1
    (-0.10, 0.23, 0.51) m, size 1.41x1.15x0.69, 5 frame(s)
    (0.07, -1.00, 0.59) m, size 1.07x0.66x0.32, 1 frame(s)
    (-0.44, 0.09, 0.50) m, size 0.54x0.65x0.76, 3 frame(s)
  chair: 0 sheet row(s), gold 1
  table: 1 sheet row(s), gold 1
    (-0.65, 0.46, 0.40) m, size 0.42x0.25x0.12, 1 frame(s)

### 1141 (arkitscenes/42446529, medium): some missing
If I am standing by the bed and facing the table, is the chair to my left, right, or back?
options ['A. left', 'B. right', 'C. back']; truth B; A_video A; pipeline_v3 B; sheet implies - (no answer); gold implies B
  bed: 3 sheet row(s), gold 1
    (-0.10, 0.23, 0.51) m, size 1.41x1.15x0.69, 5 frame(s)
    (0.07, -1.00, 0.59) m, size 1.07x0.66x0.32, 1 frame(s)
    (-0.44, 0.09, 0.50) m, size 0.54x0.65x0.76, 3 frame(s)
  table: 1 sheet row(s), gold 1
    (-0.65, 0.46, 0.40) m, size 0.42x0.25x0.12, 1 frame(s)
  chair: 0 sheet row(s), gold 1

### 1147 (arkitscenes/42897688, medium): all present, duplicated
If I am standing by the bed and facing the table, is the stool to my left, right, or back?
options ['A. right', 'B. left', 'C. back']; truth B; A_video A; pipeline_v3 B; sheet implies C (ambiguous); gold implies B
  bed: 2 sheet row(s), gold 1
    (0.70, -3.40, -0.16) m, size 3.90x3.13x1.37, 5 frame(s)
    (5.06, -6.72, 0.16) m, size 0.95x0.97x0.88, 2 frame(s)
  table: 1 sheet row(s), gold 1
    (-0.23, -2.56, 0.10) m, size 0.74x0.99x0.90, 2 frame(s)
  stool: 2 sheet row(s), gold 1
    (5.05, -6.70, 0.05) m, size 0.95x0.92x0.86, 1 frame(s)
    (5.02, -3.10, -0.05) m, size 0.09x0.23x0.06, 1 frame(s)

**hurt**

### 1008 (arkitscenes/42899696, hard): all present, duplicated
If I am standing by the bed and facing the sofa, is the tv to my front-left, front-right, back-left, or back-right?
options ['A. back-left', 'B. front-left', 'C. back-right', 'D. front-right']; truth D; A_video D; pipeline_v3 B; sheet implies A (ambiguous); gold implies D
  bed: 8 sheet row(s), gold 1
    (-0.82, -0.47, 1.70) m, size 0.83x1.66x1.25, 1 frame(s)
    (-2.64, 1.92, 0.23) m, size 2.13x1.75x0.70, 3 frame(s)
    (-4.57, 1.67, 1.08) m, size 0.99x1.07x1.13, 1 frame(s)
    (-4.63, -0.32, 1.76) m, size 2.46x2.83x2.31, 2 frame(s)
    (-3.29, -2.06, 2.14) m, size 0.49x0.26x0.33, 1 frame(s)
    (-4.88, 0.37, 0.66) m, size 1.84x1.26x1.16, 2 frame(s)
    (-4.57, 4.15, 0.13) m, size 0.75x0.38x0.38, 1 frame(s)
    (-1.72, 1.06, 0.76) m, size 1.09x1.51x1.63, 1 frame(s)
  sofa: 2 sheet row(s), gold 1
    (-0.82, -0.44, 1.72) m, size 0.83x1.67x1.10, 1 frame(s)
    (-5.08, 0.41, 1.83) m, size 0.32x0.69x0.54, 1 frame(s)
  tv: 1 sheet row(s), gold 1
    (-7.69, 0.70, -0.60) m, size 0.51x0.64x0.61, 1 frame(s)

### 1255 (arkitscenes/42445429, easy): all present, duplicated
If I am standing by the sofa and facing the tv, is the table to the left or the right of the tv?
options ['A. left', 'B. right']; truth A; A_video A; pipeline_v3 B; sheet implies B (ambiguous); gold implies A
  sofa: 2 sheet row(s), gold 1
    (-0.38, -0.19, 3.22) m, size 0.55x0.47x0.48, 2 frame(s)
    (-1.04, -0.00, 2.84) m, size 0.77x1.14x1.67, 4 frame(s)
  tv: 2 sheet row(s), gold 1
    (1.19, -0.47, 2.69) m, size 0.05x0.71x0.94, 4 frame(s)
    (-1.55, -0.79, 3.80) m, size 0.17x0.17x0.18, 2 frame(s)
  table: 3 sheet row(s), gold 1
    (1.00, -0.13, 2.67) m, size 0.23x0.54x0.46, 1 frame(s)
    (-0.42, -0.43, 3.81) m, size 0.44x0.14x0.40, 1 frame(s)
    (-0.40, -0.22, 3.29) m, size 0.55x0.16x0.36, 1 frame(s)

### 1297 (arkitscenes/47333457, easy): some missing
If I am standing by the tv and facing the toilet, is the stove to the left or the right of the toilet?
options ['A. left', 'B. right']; truth A; A_video A; pipeline_v3 B; sheet implies - (no answer); gold implies A
  tv: 1 sheet row(s), gold 1
    (-2.31, -1.16, -0.41) m, size 0.66x0.16x0.45, 2 frame(s)
  toilet: 0 sheet row(s), gold 1
  stove: 0 sheet row(s), gold 1

### 1513 (scannetpp/cc5237fd77, medium): some missing
If I am standing by the suitcase and facing the ceiling light, is the keyboard to my left, right, or back?
options ['A. left', 'B. right', 'C. back']; truth A; A_video A; pipeline_v3 B; sheet implies - (no answer); gold implies A
  suitcase: 1 sheet row(s), gold 1
    (0.05, 0.92, 1.56) m, size 0.23x0.27x0.33, 2 frame(s)
  ceiling light: 0 sheet row(s), gold 1
  keyboard: 0 sheet row(s), gold 1

### 1595 (scannetpp/c49a8c6cff, medium): some missing
If I am standing by the chair and facing the heater, is the ceiling light to my left, right, or back?
options ['A. left', 'B. back', 'C. right']; truth A; A_video A; pipeline_v3 C; sheet implies - (no answer); gold implies A
  chair: 1 sheet row(s), gold 1
    (-1.83, 0.86, 0.84) m, size 0.47x0.88x0.35, 4 frame(s)
  heater: 0 sheet row(s), gold 1
  ceiling light: 0 sheet row(s), gold 1
