per-dataset constants fitted on dev scenes (calibration of priors, fixed scale, scenes): {'arkitscenes': (0.806, 2.527, 3), 'scannet': (0.886, 2.025, 13), 'scannetpp': (0.79, 2.34, 8)}
reliable oracle fits: 10 of 25 final-test scenes, 19 of 32 dev200 scenes, 24 of 50 dev scenes

| fix | final: abs MRA (n) | final: abs pred/truth | final: abs simulated | final: rel acc (n) | final: rel simulated | dev: abs MRA (n) | dev: abs pred/truth | dev: abs simulated | dev: rel acc (n) | dev: rel simulated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| video only | – (0) | – | 34.8 | – (0) | 46.3 | – (0) | – | 43.2 | – (0) | 48.0 |
| (1) priors | 27.2 (72) | 0.64 | 29.2 | 38.1 (42) | 44.7 | 65.0 (12) | 0.89 | 56.0 | 69.2 (13) | 48.0 |
| (2) oracle [uses gold] | 36.9 (32) | 0.70 | 34.8 | 38.1 (42) | 44.7 | 55.6 (9) | 0.97 | 48.8 | 69.2 (13) | 48.0 |
| (3) room height | 29.2 (72) | 0.64 | 30.3 | 38.1 (42) | 44.7 | 40.0 (12) | 1.05 | 44.0 | 69.2 (13) | 48.0 |
| (4) door height (priors if no door) | 28.9 (72) | 0.64 | 30.2 | 38.1 (42) | 44.7 | 60.8 (12) | 0.86 | 54.0 | 69.2 (13) | 48.0 |
| (5a) per-dataset calibration [fit on dev] | 27.2 (72) | 0.64 | 29.2 | 38.1 (42) | 44.7 | 58.3 (12) | 0.90 | 52.8 | 69.2 (13) | 48.0 |
| (5b) per-dataset fixed scale [fit on dev] | 24.0 (72) | 0.46 | 27.3 | 38.1 (42) | 44.7 | 48.3 (12) | 0.94 | 48.0 | 69.2 (13) | 48.0 |
| (6) median of priors, room, door | 28.6 (72) | 0.65 | 30.0 | 38.1 (42) | 44.7 | 59.2 (12) | 0.93 | 53.2 | 69.2 (13) | 48.0 |
| (7) priors, relations only if every object in >=2 frames | 30.9 (32) | 0.58 | 34.1 | 30.0 (10) | 46.3 | 61.4 (7) | 0.80 | 50.0 | 100.0 (2) | 52.0 |

final: absolute distance with a relation and a reliable oracle scale (32): median relative error with priors 46%, with the oracle scale 36% (what positions and boxes leave), the scale alone 18%; priors / oracle scale median 0.95; room height / oracle 1.09

dev200: absolute distance with a relation and a reliable oracle scale (9): median relative error with priors 14%, with the oracle scale 25% (what positions and boxes leave), the scale alone 23%; priors / oracle scale median 0.90; room height / oracle 1.06
