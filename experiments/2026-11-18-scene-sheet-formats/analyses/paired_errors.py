"""Size and distance errors per (scene, class) and (scene, class pair), computed as pipeline_scorecard.score_root
computes them, so different sheets folders can be compared on the same pairs."""
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
import pipeline_scorecard as pc
from mvp.benchmarks import scene_scale


def pair_errors(root, scenes, k_from=None):
    """{(scene, class): size rel error}, {(scene, a, b): distance rel error}, {scene: k}; k_from: take each scene's
    scale from that sheets folder instead of root's own."""
    root = Path(root)
    by_scene = defaultdict(list)
    for q in pc.read(root / "index.csv"):
        if q["scene"] in scenes:
            by_scene[q["scene"]].append(q)
    size, dist, ks = {}, {}, {}
    for scene, qs in sorted(by_scene.items()):
        pipe, gold = pc.scene_rows(root, qs, "pipeline_sheet"), pc.scene_rows(root, qs, "gold_sheet")
        objects = {c: rows for c, rows in pipe.items() if c not in pc.FRAGMENTS}
        ratio, classes = scene_scale.prior_ratio(scene_scale.scene_longest(str(k_from or root), scene))
        k = scene_scale.CALIBRATION * ratio if classes else scene_scale.FALLBACK
        ks[scene] = k
        single = [c for c, v in gold.items() if len(v) == 1 and objects.get(c)]
        for c in single:
            size[(scene, c)] = abs(k * max(pc.longest(r) for r in objects[c]) / pc.longest(gold[c][0]) - 1)
        centre = {c: np.median([pc.box(r)[0] for r in objects[c]], axis=0) for c in single}
        for a, b in itertools.combinations(sorted(single), 2):
            dg = float(np.linalg.norm(pc.box(gold[a][0])[0] - pc.box(gold[b][0])[0]))
            if dg > 0.1:
                dist[(scene, a, b)] = abs(k * float(np.linalg.norm(centre[a] - centre[b])) / dg - 1)
    return size, dist, ks
