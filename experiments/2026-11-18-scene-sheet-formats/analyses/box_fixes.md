## All 512 dev-scene distance questions (the basis for choosing)

| fix | abs MRA (n) | abs pred/truth | rel acc (n) |
|---|---:|---:|---:|
| current boxes (min-max of the points, as built) | 41.1 (167) | 0.80 | 65.8 (79) |
| (a) trimmed 2nd-98th percentile | 46.3 (167) | 0.91 | 63.3 (79) |
| (a) trimmed 5th-95th percentile | 47.3 (167) | 0.93 | 63.3 (79) |
| (a) trimmed 10th-90th percentile | 47.5 (167) | 0.99 | 63.3 (79) |
| (b) unmerged boxes (best single row) | 41.0 (160) | 0.78 | 65.3 (75) |
| (c) centre distance minus typical half-sizes | 35.9 (167) | 0.70 | 62.0 (79) |
| (d) centre distance | 35.2 (167) | 1.28 | 60.8 (79) |

## dev200: simulated scores (relation where there is one, video's answer otherwise); video only: abs 43.2, rel 48.0

| fix | abs MRA of relations (n) | abs pred/truth | abs simulated | rel acc (n) | rel simulated |
|---|---:|---:|---:|---:|---:|
| current boxes (min-max of the points, as built) | 59.2 (12) | 0.93 | 53.2 | 69.2 (13) | 48.0 |
| (a) trimmed 2nd-98th percentile | 55.8 (12) | 1.01 | 51.6 | 69.2 (13) | 48.0 |
| (a) trimmed 5th-95th percentile | 55.8 (12) | 1.04 | 51.6 | 69.2 (13) | 48.0 |
| (a) trimmed 10th-90th percentile | 56.7 (12) | 1.08 | 52.0 | 69.2 (13) | 48.0 |
| (b) unmerged boxes (best single row) | 44.5 (11) | 0.70 | 44.4 | 75.0 (12) | 52.0 |
| (c) centre distance minus typical half-sizes | 36.7 (12) | 0.65 | 42.4 | 69.2 (13) | 48.0 |
| (d) centre distance | 35.0 (12) | 1.41 | 41.6 | 53.8 (13) | 40.0 |

## final test: simulated scores (relation where there is one, video's answer otherwise); video only: abs 34.8, rel 46.3

| fix | abs MRA of relations (n) | abs pred/truth | abs simulated | rel acc (n) | rel simulated |
|---|---:|---:|---:|---:|---:|
| current boxes (min-max of the points, as built) | 28.6 (72) | 0.65 | 30.0 | 38.1 (42) | 44.7 |
| (a) trimmed 2nd-98th percentile | 36.7 (72) | 0.73 | 34.8 | 40.5 (42) | 45.5 |
| (a) trimmed 5th-95th percentile | 40.0 (72) | 0.80 | 36.7 | 40.5 (42) | 45.5 |
| (a) trimmed 10th-90th percentile | 43.2 (72) | 0.88 | 38.6 | 40.5 (42) | 45.5 |
| (b) unmerged boxes (best single row) | 28.1 (69) | 0.66 | 30.2 | 32.4 (34) | 46.3 |
| (c) centre distance minus typical half-sizes | 21.0 (72) | 0.47 | 25.5 | 45.2 (42) | 47.2 |
| (d) centre distance | 31.2 (72) | 1.16 | 31.6 | 38.1 (42) | 44.7 |

## Per dataset, all dev scenes: abs MRA (n) / pred-truth / rel acc (n)

| fix | ARKitScenes | ScanNet | ScanNet++ |
|---|---|---|---|
| current boxes (min-max of the points, as built) | 31.7 (41) / 0.69 / 87.5 (8) | 47.4 (89) / 0.87 / 63.3 (49) | 36.2 (37) / 0.83 / 63.6 (22) |
| (a) trimmed 2nd-98th percentile | 39.3 (41) / 0.77 / 87.5 (8) | 49.9 (89) / 0.95 / 61.2 (49) | 45.7 (37) / 0.87 / 59.1 (22) |
| (a) trimmed 5th-95th percentile | 43.9 (41) / 0.82 / 87.5 (8) | 48.3 (89) / 0.98 / 61.2 (49) | 48.6 (37) / 0.89 / 59.1 (22) |
| (a) trimmed 10th-90th percentile | 44.9 (41) / 0.88 / 87.5 (8) | 47.6 (89) / 1.02 / 61.2 (49) | 50.3 (37) / 0.93 / 59.1 (22) |
| (b) unmerged boxes (best single row) | 34.6 (41) / 0.70 / 87.5 (8) | 44.3 (84) / 0.84 / 65.2 (46) | 40.6 (35) / 0.86 / 57.1 (21) |
| (c) centre distance minus typical half-sizes | 19.3 (41) / 0.44 / 87.5 (8) | 43.1 (89) / 0.86 / 59.2 (49) | 37.0 (37) / 0.62 / 59.1 (22) |
| (d) centre distance | 36.6 (41) / 1.25 / 87.5 (8) | 31.2 (89) / 1.32 / 59.2 (49) | 43.2 (37) / 1.20 / 54.5 (22) |

## Per dataset, final test: abs MRA (n) / pred-truth / rel acc (n)

| fix | ARKitScenes | ScanNet | ScanNet++ |
|---|---|---|---|
| current boxes (min-max of the points, as built) | 27.5 (24) / 0.63 / 33.3 (3) | 28.6 (37) / 0.67 / 39.4 (33) | 30.9 (11) / 0.62 / 33.3 (6) |
| (a) trimmed 2nd-98th percentile | 33.3 (24) / 0.69 / 0.0 (3) | 35.7 (37) / 0.84 / 45.5 (33) | 47.3 (11) / 0.77 / 33.3 (6) |
| (a) trimmed 5th-95th percentile | 34.6 (24) / 0.73 / 0.0 (3) | 39.5 (37) / 0.85 / 45.5 (33) | 53.6 (11) / 0.87 / 33.3 (6) |
| (a) trimmed 10th-90th percentile | 40.8 (24) / 0.82 / 0.0 (3) | 41.4 (37) / 0.90 / 45.5 (33) | 54.5 (11) / 0.95 / 33.3 (6) |
| (b) unmerged boxes (best single row) | 32.9 (24) / 0.67 / 0.0 (2) | 24.1 (34) / 0.53 / 34.6 (26) | 30.0 (11) / 0.62 / 33.3 (6) |
| (c) centre distance minus typical half-sizes | 15.0 (24) / 0.43 / 0.0 (3) | 25.9 (37) / 0.54 / 48.5 (33) | 17.3 (11) / 0.39 / 50.0 (6) |
| (d) centre distance | 33.3 (24) / 1.09 / 0.0 (3) | 28.4 (37) / 1.18 / 42.4 (33) | 36.4 (11) / 1.21 / 33.3 (6) |
