| Measure (14 scenes) | Current | fixed_s050 + merge | fixed_s040 + merge | fixed_s030 + merge | fixed_s020 + merge | fixed_s050 (no merge) |
|---|---:|---:|---:|---:|---:|---:|
| named objects found | 72.1% | 76.0% | 78.9% | 80.4% | 83.8% | 76.0% |
| questions with every named object | 46.6% | 54.7% | 62.9% | 63.4% | 66.8% | 54.7% |
| empty sheets where gold has objects | 41 | 36 | 34 | 26 | 20 | 36 |
| count = gold, per (question, class) | 23.7% | 36.8% | 32.3% | 24.4% | 25.4% | 22.5% |
| rows over the sheet folders (room fragments left out) | 2,276 | 1,912 | 2,233 | 2,867 | 3,836 | 2,578 |
| unmatched rows (beyond the gold sheet's count) | 51.5% | 40.5% | 47.6% | 58.2% | 67.0% | 55.4% |
| size error, median relative | 29.3% | 29.7% | 31.6% | 27.7% | 30.7% | 26.6% |
| distance error, median relative | 22.6% | 30.4% | 31.0% | 37.8% | 42.2% | 27.5% |
| rule: counting | 23.9 | 36.1 | 31.8 | 26.8 | 20.0 | 23.9 |
| rule: object size | 29.7 | 29.7 | 28.9 | 32.0 | 32.4 | 30.8 |
| rule: absolute distance | 13.8 | 12.6 | 16.3 | 10.3 | 12.3 | 16.0 |
| rule: relative distance | 35.2 | 24.1 | 22.2 | 16.7 | 24.1 | 31.5 |

per scene against Current, better/same/worse of 14:
  fixed_s050 + merge     found 4/10/0  every-object 4/10/0  unmatched 13/1/0  distance error 4/1/7 (of 12)
  fixed_s040 + merge     found 5/9/0  every-object 5/9/0  unmatched 8/2/4  distance error 5/1/6 (of 12)
  fixed_s030 + merge     found 6/8/0  every-object 6/8/0  unmatched 4/0/10  distance error 5/1/6 (of 12)
  fixed_s020 + merge     found 11/3/0  every-object 11/3/0  unmatched 2/0/12  distance error 2/1/9 (of 12)
  fixed_s050 (no merge)  found 4/10/0  every-object 4/10/0  unmatched 1/8/5  distance error 6/3/3 (of 12)
