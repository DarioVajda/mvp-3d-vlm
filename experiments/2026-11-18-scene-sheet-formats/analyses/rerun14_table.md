| Measure (14 scenes) | Current | original_s020 | original_s020 + merge | fixed_s020 | fixed_s020 + merge |
|---|---:|---:|---:|---:|---:|
| named objects found | 72.1% | 83.0% | 83.0% | 83.8% | 83.8% |
| questions with every named object | 46.6% | 65.5% | 65.5% | 66.8% | 66.8% |
| empty sheets where gold has objects | 41 | 22 | 22 | 20 | 20 |
| count = gold, per (question, class) | 23.7% | 12.9% | 24.6% | 13.7% | 25.4% |
| rows over the sheet folders (room fragments left out) | 2,276 | 4,565 | 3,612 | 4,854 | 3,836 |
| rows seen in only one frame | 67.3% | 73.9% | 54.2% | 74.2% | 55.9% |
| unmatched rows (beyond the gold sheet's count) | 51.5% | 72.5% | 65.3% | 73.9% | 67.0% |
| size error, median relative | 29.3% | 28.3% | 30.9% | 27.3% | 30.7% |
| distance error, median relative | 22.6% | 37.9% | 45.7% | 38.1% | 42.2% |
| rule: counting | 23.9 | 17.4 | 20.0 | 17.4 | 20.0 |
| rule: object size | 29.7 | 33.1 | 33.4 | 33.2 | 32.4 |
| rule: absolute distance | 13.8 | 12.4 | 10.4 | 14.2 | 12.3 |
| rule: relative distance | 35.2 | 29.6 | 29.6 | 22.2 | 24.1 |

n: 478 questions, 1299 (question, object) pairs, 1276 count pairs, size pairs 49, distance pairs 86, rules {'object_abs_distance': 78, 'object_counting': 38, 'object_rel_distance': 54, 'object_size_estimation': 71}

per scene, against Current (better / same / worse of 14):
  original_s020          found 10/4/0   unmatched rows 1/0/13   count = gold 3/5/6
  original_s020 + merge  found 10/4/0   unmatched rows 2/0/12   count = gold 5/4/5
  fixed_s020             found 11/3/0   unmatched rows 1/0/13   count = gold 4/3/7
  fixed_s020 + merge     found 11/3/0   unmatched rows 2/0/12   count = gold 6/2/6
