"""Can pipeline sheets be put in metres without gold at test time? Writes metric_scale.md next to this file.

  PYTHONPATH=src python experiments/2026-11-18-scene-sheet-formats/metric_scale.py [--root data/scene_sheets_75]
  PYTHONPATH=src python experiments/2026-11-18-scene-sheet-formats/metric_scale.py --fit   # scene_scale's constants

1. The true scale (metres per sheet unit) of every scene with a reliable oracle fit: explore_sheets.align (gold ~ s R
   pipeline + t) with a residual under MAX_RMS and the cameras a plausible height above the gold floor.
2. How that scale relates to quantities read from the pipeline sheets alone (features).
3. Candidate conversions that use no gold at test time, scored with 5-fold cross-validation by scene (stratified by
   dataset, seed 0): fitted on the other folds' scenes, scored on this fold's against gold. Scores are the median
   relative error of object sizes (the longest side of the largest pipeline row of a class against the gold object's,
   for classes with one gold object) and of distances between object centres (pairs of classes with one gold object
   each), next to no conversion and the oracle scale.
Reads the sheets only (index.csv, objects.csv, cameras.csv, adapter_report.json). The size priors are
mvp.benchmarks.scene_scale's; --fit prints the constants scene_scale uses for conversion d' at render time, fitted on the
reliable scenes outside explore200_ids.txt's scenes, and checks them against the ones in the code.
"""
import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import explore_sheets as es  # noqa: E402
from mvp.benchmarks import scene_scale  # noqa: E402
from mvp.openvocab_adapter.camera_sheet_format_converter import estimate_up, read_cameras  # noqa: E402
from mvp.paths import SCENE_SHEETS  # noqa: E402

MAX_RMS = 0.5  # metres
CAMERA_HEIGHT = (0.8, 2.0)  # metres above the gold floor
FRAGMENTS = {"room", "wall", "floor", "ceiling"}
FOLDS, SEED = 5, 0
PRIORS = scene_scale.PRIORS  # typical longest sides, metres: an assumption, stated in the report
FEATURES = {
    "camera_height": "camera height above the floor (10th percentile of the object boxes' bottoms along the up the cameras give)",
    "depth": "median distance from an object's centre to the nearest camera (the depth maps are not in the sheets)",
    "path_length": "camera path length over the 16 frames",
    "extent": "scene extent: diagonal of the 5-95% range of object centres and cameras",
    "object_size": "median longest side of the pipeline's rows (room fragments left out)",
    "n_frames": "frames in the video",
}


def longest(e):
    return float(np.max(e))


def scene_data(root, scene, rows):
    """Everything about one scene: gold and pipeline classes, the oracle fit, the features."""
    gold = es.classes({root / r["gold_sheet"] for r in rows})
    pipe = es.classes({root / r["pipeline_sheet"] for r in rows})
    folder = root / rows[0]["pipeline_sheet"]
    centres, rotations = read_cameras(folder / "cameras.csv")
    fit, oracle, rms, cam_gold = es.align(gold, pipe), None, None, None
    if fit:
        k, R, t, rms, _ = fit
        floor = min(c[2] - e[2] / 2 for objs in gold.values() for c, e in objs)
        cam_gold = float(np.median((k * centres @ R.T + t)[:, 2]) - floor)
        if rms < MAX_RMS and CAMERA_HEIGHT[0] < cam_gold < CAMERA_HEIGHT[1]:
            oracle = k
    objs = [(c, e) for cls, items in pipe.items() if cls not in FRAGMENTS for c, e in items]
    allc = np.array([c for cls, items in pipe.items() for c, _ in items])
    up = estimate_up(rotations)
    bottoms = [c @ up - np.abs(up) @ e / 2 for items in pipe.values() for c, e in items]
    report = json.loads((folder.parents[1] / "scene" / "adapter_report.json").read_text())
    pts = np.vstack([allc, centres])
    lo, hi = np.percentile(pts, 5, axis=0), np.percentile(pts, 95, axis=0)
    features = {
        "camera_height": float(np.median(centres @ up) - np.percentile(bottoms, 10)),
        "depth": float(np.median([np.min(np.linalg.norm(centres - c, axis=1)) for c in allc])),
        "path_length": float(np.linalg.norm(np.diff(centres, axis=0), axis=1).sum()),
        "extent": float(np.linalg.norm(hi - lo)),
        "object_size": float(np.median([longest(e) for _, e in objs])) if objs else np.nan,
        "n_frames": float(max(report["source_video_frames"]) + 1),
    }
    return dict(scene=scene, dataset=scene.split("/")[0], gold=gold, pipe=pipe, oracle=oracle, rms=rms,
                cam_gold=cam_gold, prior=prior_scale(pipe), **features)


def prior_scale(pipe):
    """Per scene, no gold: the median over prior classes present of typical size / the class's largest row
    (scene_scale.prior_ratio, nothing left out)."""
    return scene_scale.prior_ratio({c: max(longest(e) for _, e in items) for c, items in pipe.items() if items})[0]


def errors(d, k):
    """Relative errors of sizes and centre distances against gold for scale k."""
    single = [c for c, items in d["gold"].items() if len(items) == 1 and d["pipe"].get(c) and c not in FRAGMENTS]
    sizes = [abs(k * max(longest(e) for _, e in d["pipe"][c]) / longest(d["gold"][c][0][1]) - 1) for c in single]
    centre = {c: np.median([p for p, _ in d["pipe"][c]], axis=0) for c in single}
    dists = []
    for a, b in itertools.combinations(single, 2):
        dg = np.linalg.norm(d["gold"][a][0][0] - d["gold"][b][0][0])
        if dg > 0.1:
            dists.append(abs(k * np.linalg.norm(centre[a] - centre[b]) / dg - 1))
    return sizes, dists


def folds(scenes):
    """Scene -> fold, 5 folds, stratified by dataset, seed 0."""
    rng = np.random.default_rng(SEED)
    out = {}
    for ds in sorted({s["dataset"] for s in scenes}):
        names = sorted(s["scene"] for s in scenes if s["dataset"] == ds)
        rng.shuffle(names)
        out.update({n: i % FOLDS for i, n in enumerate(names)})
    return out


def fit_regression(train, feature):
    """log k = a + b log feature on the training scenes; (a, b, R^2)."""
    x = np.log([s[feature] for s in train])
    y = np.log([s["oracle"] for s in train])
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    return a, b, 1 - resid.var() / y.var()


def cross_validate(scenes):
    """{method: {scene: k}} for every method, each test scene's k from the other folds; and the features chosen."""
    fold_of = folds(scenes)
    methods = defaultdict(dict)
    chosen = []
    for f in range(FOLDS):
        train = [s for s in scenes if fold_of[s["scene"]] != f]
        test = [s for s in scenes if fold_of[s["scene"]] == f]
        glob = float(np.median([s["oracle"] for s in train]))
        per_ds = {ds: float(np.median([s["oracle"] for s in train if s["dataset"] == ds]))
                  for ds in {s["dataset"] for s in train}}
        fits = {feat: fit_regression(train, feat) for feat in FEATURES
                if all(np.isfinite(s[feat]) and s[feat] > 0 for s in scenes)}
        best = max(fits, key=lambda feat: fits[feat][2])
        chosen.append(best)
        calib = float(np.median([s["oracle"] / s["prior"] for s in train if np.isfinite(s["prior"])]))
        for s in test:
            methods["no conversion"][s["scene"]] = 1.0
            methods["oracle"][s["scene"]] = s["oracle"]
            methods["a. one constant"][s["scene"]] = glob
            methods["b. one constant per dataset"][s["scene"]] = per_ds.get(s["dataset"], glob)
            a, b, _ = fits[best]
            methods["c. regression on the best feature"][s["scene"]] = float(np.exp(a + b * np.log(s[best])))
            for feat, (a, b, _) in fits.items():
                methods[f"regression on {feat}"][s["scene"]] = float(np.exp(a + b * np.log(s[feat])))
            prior = s["prior"] if np.isfinite(s["prior"]) else glob
            methods["d. size priors"][s["scene"]] = prior
            methods["d'. size priors, calibrated"][s["scene"]] = (calib * s["prior"] if np.isfinite(s["prior"])
                                                                   else glob)
    return methods, chosen


BOOT = 2000


def score(scenes, ks):
    """Pooled median errors, the scale's error against the oracle, and bootstrap 95% intervals over scenes (the scenes
    resampled with replacement, their errors pooled; seed 0, the same resamples for every conversion)."""
    by = {s["scene"]: s for s in scenes}
    names = sorted(ks)
    per = {n: errors(by[n], ks[n]) for n in names}
    scale = np.array([abs(ks[n] / by[n]["oracle"] - 1) for n in names])

    def pooled(idx, j):
        """Median of the errors of kind j (0 sizes, 1 distances) of the scenes idx, pooled."""
        return np.median([e for i in idx for e in per[names[i]][j]])
    draws = np.random.default_rng(SEED).integers(0, len(names), size=(BOOT, len(names)))
    ci = lambda j: tuple(np.percentile([pooled(d, j) for d in draws], [2.5, 97.5]))
    sizes = [e for n in names for e in per[n][0]]
    dists = [e for n in names for e in per[n][1]]
    s30 = [np.mean(scale[d] < 0.3) for d in draws]
    return dict(sizes=float(np.median(sizes)), dists=float(np.median(dists)), scale=float(np.median(scale)),
                scale20=float(np.mean(scale < 0.2)), scale30=float(np.mean(scale < 0.3)), n_sizes=len(sizes),
                n_dists=len(dists), sizes_ci=ci(0), dists_ci=ci(1), scale30_ci=tuple(np.percentile(s30, [2.5, 97.5])))


def ci_text(lo_hi):
    return f"{100 * lo_hi[0]:.0f}–{100 * lo_hi[1]:.0f}"


def pct(x):
    return f"{100 * x:.0f}%"


def report(root):
    index = es.index_rows(root)
    by_scene = defaultdict(list)
    for r in index:
        by_scene[r["scene"]].append(r)
    data = [scene_data(root, s, rows) for s, rows in sorted(by_scene.items())]
    rel = [d for d in data if d["oracle"]]
    k = np.array([d["oracle"] for d in rel])
    ds = pd.Series([d["dataset"] for d in rel])
    lines = ["# Metric scale for pipeline sheets without gold", "",
             "<!-- Written by metric_scale.py in this folder; edit that, not this file. -->", "",
             f"Source: `{root.name}`, {len(data)} scenes, read only. Regenerate with `PYTHONPATH=src python "
             "experiments/2026-11-18-scene-sheet-formats/metric_scale.py`.", "",
             "## 1. The true scale", "",
             f"A scene has a reliable oracle fit when `explore_sheets.align` (a similarity fit of the pipeline sheet to "
             f"the gold sheet from matched object centres) exists, its residual is under {MAX_RMS} m and it puts the "
             f"cameras {CAMERA_HEIGHT[0]}–{CAMERA_HEIGHT[1]} m above the gold floor: {len(rel)} of {len(data)} scenes "
             f"({sum(d['rms'] is not None for d in data)} have a fit at all).", "",
             "| Scene | Dataset | Metres per unit | Fit residual (m) | Cameras above the floor (m) |",
             "|---|---|---:|---:|---:|"]
    lines += [f"| {d['scene']} | {d['dataset']} | {d['oracle']:.3f} | {d['rms']:.2f} | {d['cam_gold']:.2f} |"
              for d in sorted(rel, key=lambda d: (d["dataset"], d["oracle"]))]
    lines += ["", f"Over the {len(rel)} scenes: mean {k.mean():.2f}, median {np.median(k):.2f}, standard deviation "
              f"{k.std(ddof=1):.2f} (coefficient of variation {k.std(ddof=1) / k.mean():.2f}), range {k.min():.2f}–"
              f"{k.max():.2f}, IQR {np.percentile(k, 25):.2f}–{np.percentile(k, 75):.2f}.", "",
              "| Dataset | Scenes | Median | IQR | Mean |", "|---|---:|---:|---:|---:|"]
    for name in sorted(set(ds)):
        v = k[(ds == name).to_numpy()]
        lines.append(f"| {name} | {len(v)} | {np.median(v):.2f} | {np.percentile(v, 25):.2f}–{np.percentile(v, 75):.2f}"
                     f" | {v.mean():.2f} |")
    groups = [np.log(k[(ds == name).to_numpy()]) for name in sorted(set(ds))]
    kw = stats.kruskal(*groups)
    lines += ["", f"Do the datasets differ? Kruskal–Wallis on the scales: H = {kw.statistic:.2f}, p = {kw.pvalue:.3f} "
              f"({len(groups)} datasets, {len(rel)} scenes)."]

    # 2. correlations
    lines += ["", "## 2. What the scale goes with", "",
              "Each feature is read from the pipeline sheets alone. A conversion multiplies sheet units by k, so a "
              "feature that measures a real length in sheet units should go as k ∝ 1 / feature: a slope near −1 on "
              "log k against log feature.", "",
              "| Feature | What it is | Spearman ρ | Pearson r (logs) | Slope (logs) |", "|---|---|---:|---:|---:|"]
    corr = {}
    for feat, what in FEATURES.items():
        x = np.array([d[feat] for d in rel])
        ok = np.isfinite(x) & (x > 0)
        rho = stats.spearmanr(x[ok], k[ok]).statistic
        r = stats.pearsonr(np.log(x[ok]), np.log(k[ok])).statistic
        slope = np.polyfit(np.log(x[ok]), np.log(k[ok]), 1)[0]
        corr[feat] = (rho, r)
        lines.append(f"| {feat} | {what} | {rho:+.2f} | {r:+.2f} | {slope:+.2f} |")
    pri = np.array([d["prior"] for d in rel])
    okp = np.isfinite(pri)
    rp = stats.pearsonr(np.log(pri[okp]), np.log(k[okp])).statistic
    lines += ["", f"The size-prior estimate (d below) against the true scale: Pearson r {rp:+.2f} on the logs, in "
              f"{okp.sum()} of {len(rel)} scenes (the others have no prior class)."]

    # 3. cross-validation
    methods, chosen = cross_validate(rel)
    order = ["no conversion", "oracle", "a. one constant", "b. one constant per dataset",
             "c. regression on the best feature", "d. size priors", "d'. size priors, calibrated"]
    scores = {m: score(rel, methods[m]) for m in methods}
    n = scores["oracle"]
    lines += ["", "## 3. Conversions without gold at test time", "",
              f"5-fold cross-validation by scene ({FOLDS} folds stratified by dataset, seed {SEED}): every scene's "
              "scale comes from a conversion fitted on the other folds' scenes only. Errors are |estimate / gold − 1|, "
              f"pooled over the {len(rel)} scenes: {n['n_sizes']} object sizes and {n['n_dists']} centre distances, "
              f"with bootstrap 95% intervals over scenes ({BOOT:,} resamples). The oracle row is the floor set by the "
              "pipeline's content (missing and split objects); its distance error is optimistic, because the oracle "
              "scale is fitted to these centres. **The size error is circular for d and d'**: they set the scale from "
              "the sizes of the same kinds of objects the size error checks, so a scale that is too large can make up "
              "for split objects' small boxes. Their distance and scale errors are not circular.", "",
              "| Conversion | Sizes: median error (95%) | Distances: median error (95%) | Scale: median error | "
              "Scale within 20% | Within 30% (95%) |", "|---|---:|---:|---:|---:|---:|"]
    for m in order:
        s = scores[m]
        lines.append(f"| {m} | {pct(s['sizes'])} ({ci_text(s['sizes_ci'])}) | {pct(s['dists'])} "
                     f"({ci_text(s['dists_ci'])}) | {pct(s['scale'])} | {pct(s['scale20'])} | {pct(s['scale30'])} "
                     f"({ci_text(s['scale30_ci'])}) |")
    lines += ["", f"c. picks, in each fold, the feature whose regression fits the training scenes best: "
              f"{', '.join(f'{f} ({chosen.count(f)})' for f in dict.fromkeys(chosen))}. Every feature's regression, "
              "cross-validated the same way:", "",
              "| Regression on | Sizes | Distances | Scale: median error | Within 30% |", "|---|---:|---:|---:|---:|"]
    for feat in FEATURES:
        m = f"regression on {feat}"
        if m in scores:
            s = scores[m]
            lines.append(f"| {feat} | {pct(s['sizes'])} | {pct(s['dists'])} | {pct(s['scale'])} | {pct(s['scale30'])} |")
    lines += ["", "d. uses no gold at all: per scene, the median over the prior classes it has of typical size / the "
              "class's largest row. d'. multiplies it by one constant fitted on the training folds (the median of "
              "true / prior scale), which corrects the priors' bias from split objects. The typical sizes are an "
              "assumption (common real-world dimensions, longest side, metres): "
              + ", ".join(f"{c} {v}" for c, v in PRIORS.items()) + ". Scenes without any prior class fall back to "
              "the global constant."]

    # 4. recommendation
    # simplest first: what a conversion needs besides the sheet
    cands = {"a. one constant": "1 number", "b. one constant per dataset": "3 numbers and the dataset's name",
             "d. size priors": "a table of typical sizes, nothing fitted",
             "d'. size priors, calibrated": "the table and 1 number",
             "c. regression on the best feature": "a feature (floor and up estimated from the sheet) and 2 numbers"}
    best = max(cands, key=lambda m: (round(scores[m]["scale30"], 2), -list(cands).index(m)))
    s, sa = scores[best], scores["a. one constant"]
    lines += ["", "## 4. Recommendation", "",
              "Chosen by the two errors that are not circular, the scale's error against the true scale and the "
              "distance error, with the simplest conversion first:", "",
              "| Conversion | Needs | Scale within 30% (95%) | Scale: median error | Distances, points above the oracle |",
              "|---|---|---:|---:|---:|"]
    lines += [f"| {m} | {need} | {pct(scores[m]['scale30'])} ({ci_text(scores[m]['scale30_ci'])}) | "
              f"{pct(scores[m]['scale'])} | {100 * (scores[m]['dists'] - n['dists']):+.0f} |" for m, need in cands.items()]
    lines += ["", f"**{best}** puts the scale within 30% of the true one in the most scenes ({pct(s['scale30'])}, "
              f"against {pct(sa['scale30'])} for one constant), with a median scale error of {pct(s['scale'])}. Its "
              f"distances are {pct(s['dists'])} off against the oracle's {pct(n['dists'])} (an optimistic floor) and "
              f"no conversion's {pct(scores['no conversion']['dists'])}; its sizes {pct(s['sizes'])} against the "
              f"oracle's {pct(n['sizes'])}. With {len(rel)} scenes the intervals of a–d' overlap, so the differences "
              "between them are a few points at most; the clear gap is between any of them and no conversion."]
    return "\n".join(lines) + "\n", dict(data=data, rel=rel, scores=scores, corr=corr, chosen=chosen, best=best)


def fit_constants(root, ids_file=HERE / "explore200_ids.txt"):
    """scene_scale's CALIBRATION and FALLBACK, fitted on the scenes with a reliable oracle fit outside the scenes of
    ids_file: the median of true / prior scale, and the median true scale."""
    index = es.index_rows(root)
    ids = {int(x) for x in Path(ids_file).read_text().split()}
    held = {r["scene"] for r in index if int(r["id"]) in ids}
    by_scene = defaultdict(list)
    for r in index:
        by_scene[r["scene"]].append(r)
    rel = [d for d in (scene_data(root, s, rows) for s, rows in sorted(by_scene.items()) if s not in held)
           if d["oracle"]]
    return (float(np.median([d["oracle"] / d["prior"] for d in rel])), float(np.median([d["oracle"] for d in rel])),
            len(rel))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(SCENE_SHEETS))
    ap.add_argument("--out", default=str(HERE / "metric_scale.md"))
    ap.add_argument("--fit", action="store_true", help="print scene_scale's constants, fitted, and check the code's")
    args = ap.parse_args()
    if args.fit:
        calibration, fallback, n = fit_constants(Path(args.root))
        print(f"fitted on {n} scenes: CALIBRATION = {calibration:.4f}, FALLBACK = {fallback:.3f}; in scene_scale.py: "
              f"{scene_scale.CALIBRATION}, {scene_scale.FALLBACK}")
        if (round(calibration, 4), round(fallback, 3)) != (scene_scale.CALIBRATION, scene_scale.FALLBACK):
            sys.exit("scene_scale.py's constants differ from the fit: update them")
        return
    text, _ = report(Path(args.root))
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
