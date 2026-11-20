# Metric scale for pipeline sheets without gold

<!-- Written by metric_scale.py in this folder; edit that, not this file. -->

Source: `scene_sheets_75`, 75 scenes, read only. Regenerate with `PYTHONPATH=src python experiments/2026-11-18-scene-sheet-formats/metric_scale.py`.

## 1. The true scale

A scene has a reliable oracle fit when `explore_sheets.align` (a similarity fit of the pipeline sheet to the gold sheet from matched object centres) exists, its residual is under 0.5 m and it puts the cameras 0.8–2.0 m above the gold floor: 34 of 75 scenes (45 have a fit at all).

| Scene | Dataset | Metres per unit | Fit residual (m) | Cameras above the floor (m) |
|---|---|---:|---:|---:|
| arkitscenes/47333899 | arkitscenes | 2.496 | 0.26 | 1.46 |
| arkitscenes/42446049 | arkitscenes | 2.527 | 0.25 | 1.47 |
| arkitscenes/47333457 | arkitscenes | 2.634 | 0.35 | 0.91 |
| arkitscenes/47332899 | arkitscenes | 2.882 | 0.28 | 1.43 |
| arkitscenes/42445429 | arkitscenes | 3.445 | 0.18 | 1.93 |
| arkitscenes/42446103 | arkitscenes | 4.578 | 0.26 | 1.38 |
| arkitscenes/45663164 | arkitscenes | 5.115 | 0.17 | 1.67 |
| scannet/scene0458_00 | scannet | 1.160 | 0.30 | 1.15 |
| scannet/scene0693_00 | scannet | 1.411 | 0.50 | 1.25 |
| scannet/scene0426_00 | scannet | 1.640 | 0.36 | 1.14 |
| scannet/scene0378_01 | scannet | 1.769 | 0.30 | 1.29 |
| scannet/scene0695_00 | scannet | 1.776 | 0.24 | 1.37 |
| scannet/scene0608_00 | scannet | 1.892 | 0.40 | 1.69 |
| scannet/scene0580_01 | scannet | 1.949 | 0.28 | 1.82 |
| scannet/scene0353_00 | scannet | 1.980 | 0.44 | 1.56 |
| scannet/scene0046_01 | scannet | 2.025 | 0.27 | 1.72 |
| scannet/scene0025_01 | scannet | 2.097 | 0.33 | 1.54 |
| scannet/scene0149_00 | scannet | 2.184 | 0.30 | 1.36 |
| scannet/scene0356_00 | scannet | 2.269 | 0.26 | 1.60 |
| scannet/scene0591_01 | scannet | 2.394 | 0.21 | 1.44 |
| scannet/scene0354_00 | scannet | 2.434 | 0.24 | 1.42 |
| scannet/scene0699_00 | scannet | 2.441 | 0.35 | 1.75 |
| scannet/scene0207_02 | scannet | 2.962 | 0.33 | 1.60 |
| scannet/scene0645_00 | scannet | 3.155 | 0.31 | 1.94 |
| scannetpp/f9f95681fd | scannetpp | 0.619 | 0.49 | 1.17 |
| scannetpp/bcd2436daf | scannetpp | 1.426 | 0.35 | 0.93 |
| scannetpp/45b0dac5e3 | scannetpp | 1.568 | 0.24 | 1.32 |
| scannetpp/c49a8c6cff | scannetpp | 2.166 | 0.26 | 1.24 |
| scannetpp/7b6477cb95 | scannetpp | 2.317 | 0.30 | 1.40 |
| scannetpp/acd95847c5 | scannetpp | 2.362 | 0.23 | 1.29 |
| scannetpp/d755b3d9d8 | scannetpp | 2.589 | 0.33 | 1.51 |
| scannetpp/25f3b7a318 | scannetpp | 2.780 | 0.24 | 1.62 |
| scannetpp/09c1414f1b | scannetpp | 3.099 | 0.29 | 1.34 |
| scannetpp/5942004064 | scannetpp | 3.382 | 0.46 | 1.54 |

Over the 34 scenes: mean 2.40, median 2.34, standard deviation 0.88 (coefficient of variation 0.37), range 0.62–5.12, IQR 1.91–2.74.

| Dataset | Scenes | Median | IQR | Mean |
|---|---:|---:|---:|---:|
| arkitscenes | 7 | 2.88 | 2.58–4.01 | 3.38 |
| scannet | 17 | 2.03 | 1.78–2.39 | 2.09 |
| scannetpp | 10 | 2.34 | 1.72–2.73 | 2.23 |

Do the datasets differ? Kruskal–Wallis on the scales: H = 10.52, p = 0.005 (3 datasets, 34 scenes).

## 2. What the scale goes with

Each feature is read from the pipeline sheets alone. A conversion multiplies sheet units by k, so a feature that measures a real length in sheet units should go as k ∝ 1 / feature: a slope near −1 on log k against log feature.

| Feature | What it is | Spearman ρ | Pearson r (logs) | Slope (logs) |
|---|---|---:|---:|---:|
| camera_height | camera height above the floor (10th percentile of the object boxes' bottoms along the up the cameras give) | -0.60 | -0.61 | -0.81 |
| depth | median distance from an object's centre to the nearest camera (the depth maps are not in the sheets) | -0.51 | -0.48 | -0.71 |
| path_length | camera path length over the 16 frames | -0.07 | -0.09 | -0.13 |
| extent | scene extent: diagonal of the 5-95% range of object centres and cameras | -0.14 | -0.22 | -0.36 |
| object_size | median longest side of the pipeline's rows (room fragments left out) | -0.65 | -0.63 | -0.75 |
| n_frames | frames in the video | +0.27 | +0.19 | +0.17 |

The size-prior estimate (d below) against the true scale: Pearson r +0.64 on the logs, in 34 of 34 scenes (the others have no prior class).

## 3. Conversions without gold at test time

5-fold cross-validation by scene (5 folds stratified by dataset, seed 0): every scene's scale comes from a conversion fitted on the other folds' scenes only. Errors are |estimate / gold − 1|, pooled over the 34 scenes: 165 object sizes and 365 centre distances, with bootstrap 95% intervals over scenes (2,000 resamples). The oracle row is the floor set by the pipeline's content (missing and split objects); its distance error is optimistic, because the oracle scale is fitted to these centres. **The size error is circular for d and d'**: they set the scale from the sizes of the same kinds of objects the size error checks, so a scale that is too large can make up for split objects' small boxes. Their distance and scale errors are not circular.

| Conversion | Sizes: median error (95%) | Distances: median error (95%) | Scale: median error | Scale within 20% | Within 30% (95%) |
|---|---:|---:|---:|---:|---:|
| no conversion | 65% (61–69) | 57% (54–62) | 58% | 3% | 9% (0–18) |
| oracle | 27% (20–30) | 14% (10–16) | 0% | 100% | 100% (100–100) |
| a. one constant | 29% (25–34) | 23% (16–27) | 19% | 50% | 68% (53–82) |
| b. one constant per dataset | 32% (27–36) | 21% (15–26) | 19% | 53% | 71% (56–85) |
| c. regression on the best feature | 29% (24–34) | 19% (16–24) | 15% | 65% | 76% (62–88) |
| d. size priors | 21% (18–27) | 18% (13–24) | 16% | 53% | 68% (53–82) |
| d'. size priors, calibrated | 27% (24–30) | 20% (17–23) | 16% | 68% | 82% (68–94) |

c. picks, in each fold, the feature whose regression fits the training scenes best: object_size (2), camera_height (3). Every feature's regression, cross-validated the same way:

| Regression on | Sizes | Distances | Scale: median error | Within 30% |
|---|---:|---:|---:|---:|
| camera_height | 29% | 19% | 18% | 79% |
| depth | 29% | 20% | 16% | 79% |
| path_length | 31% | 22% | 16% | 65% |
| extent | 31% | 22% | 15% | 71% |
| object_size | 29% | 22% | 17% | 82% |
| n_frames | 33% | 21% | 21% | 71% |

d. uses no gold at all: per scene, the median over the prior classes it has of typical size / the class's largest row. d'. multiplies it by one constant fitted on the training folds (the median of true / prior scale), which corrects the priors' bias from split objects. The typical sizes are an assumption (common real-world dimensions, longest side, metres): door 2.0, bed 2.0, sofa 2.0, refrigerator 1.75, chair 0.85, toilet 0.7, tv 1.1, monitor 0.55, pillow 0.55, nightstand 0.55, washer 0.85, keyboard 0.45, microwave 0.5, laptop 0.35. Scenes without any prior class fall back to the global constant.

## 4. Recommendation

Chosen by the two errors that are not circular, the scale's error against the true scale and the distance error, with the simplest conversion first:

| Conversion | Needs | Scale within 30% (95%) | Scale: median error | Distances, points above the oracle |
|---|---|---:|---:|---:|
| a. one constant | 1 number | 68% (53–82) | 19% | +9 |
| b. one constant per dataset | 3 numbers and the dataset's name | 71% (56–85) | 19% | +7 |
| d. size priors | a table of typical sizes, nothing fitted | 68% (53–82) | 16% | +5 |
| d'. size priors, calibrated | the table and 1 number | 82% (68–94) | 16% | +6 |
| c. regression on the best feature | a feature (floor and up estimated from the sheet) and 2 numbers | 76% (62–88) | 15% | +6 |

**d'. size priors, calibrated** puts the scale within 30% of the true one in the most scenes (82%, against 68% for one constant), with a median scale error of 16%. Its distances are 20% off against the oracle's 14% (an optimistic floor) and no conversion's 57%; its sizes 27% against the oracle's 27%. With 34 scenes the intervals of a–d' overlap, so the differences between them are a few points at most; the clear gap is between any of them and no conversion.
