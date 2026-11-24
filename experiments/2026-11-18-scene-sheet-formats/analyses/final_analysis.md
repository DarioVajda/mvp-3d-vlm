## REGISTERED RESULTS

H1 (primary): 245 distance questions; helped 49, hurt 75, unchanged 121; exact two-sided sign test p = 0.024370
   verdict: NOT CONFIRMED (confirmed only if p < 0.05 and helped > hurt)

H2 (secondary): all 538 questions; VSI-Bench overall (tasks present): A_video 47.97, full method 44.27, difference -3.70, 95% bootstrap [-8.15, +0.71]; helped 72, hurt 103
   not supported (supported only if the interval excludes 0 and lies above it)

the notebook's preregistered_test.csv: helped 49, hurt 75, p 0.024370, NOT CONFIRMED; secondary -3.70 [-8.15, +0.71]

## EXPLORATORY (not registered)

per question type:

| question type | n | A_video | dedup_relations | full method | sheet vs video helped/hurt | sign p (unadjusted) |
|---|---|---|---|---|---|---|
| obj_appearance_order | 129 | 48.8 | 45.7 | 45.7 | 15/19 | 0.608 |
| object_abs_distance | 122 | 34.8 | 28.4 | 28.4 | 35/54 | 0.056 |
| object_counting | 36 | 46.9 | 43.6 | 43.6 | 8/9 | 1.000 |
| object_rel_distance | 123 | 46.3 | 40.7 | 40.7 | 14/21 | 0.311 |
| object_size_estimation | 128 | 62.9 | 35.6 | 62.9 | 22/77 | 0.000 |

distance questions with and without relations:

| group | type | n | A_video | dedup_relations | helped/hurt |
|---|---|---|---|---|---|
| with relations (every named object in the sheet) | object_abs_distance | 72 | 36.8 | 27.4 | 20/35 |
| with relations (every named object in the sheet) | object_rel_distance | 42 | 42.9 | 38.1 | 8/10 |
| without relations (a named object missing) | object_abs_distance | 50 | 32.0 | 30.0 | 15/19 |
| without relations (a named object missing) | object_rel_distance | 81 | 48.1 | 42.0 | 6/11 |

absolute distance with a precomputed distance (72): precomputed vs truth MRA 27.2, median ratio 0.64 (quartiles 0.33-0.82); within 25% of the truth 22; the model repeats it in 72; video's MRA there 36.8, dedup_relations' 27.4
relative distance with a precomputed closest object (42): it is right in 16; the model answers it in 42; video right in 18, dedup_relations right in 16
