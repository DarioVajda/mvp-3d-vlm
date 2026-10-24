"""Spatial-MLLM's space-aware frame sampling, re-implemented on the saved VGGT outputs (data/vggt/).

Follows space_aware_frame_sampling() in third_party/Spatial-MLLM/src/sampling/sa_sampling.py:
  1. keep points whose world_points_conf is >= the median over all points of all frames and > 0.1;
  2. axis-aligned bounding box of the kept points; voxel size = shortest box side / 20;
  3. per frame, the set of voxels its kept points fall in;
  4. greedy maximum coverage: repeatedly add the frame covering the most not-yet-covered voxels (ties go to
     the earliest frame), up to K frames or until no frame adds anything; return the picks in temporal order.
Uniform sampling in their pipeline takes np.linspace(0, 127, K, dtype=int) of the same 128 candidates.
"""
import json
from pathlib import Path

import numpy as np

from mvp.paths import VGGT_DIR


def load_points(scene_dir):
    d = Path(scene_dir)
    return (np.load(d / "world_points.npy").astype(np.float32), np.load(d / "world_points_conf.npy").astype(np.float32),
            np.load(d / "frame_indices.npy"), json.loads((d / "meta.json").read_text()))


def voxel_sets(world_points, conf, voxels_per_side=20):
    """Per-frame voxel ids (np.int64 arrays) plus the grid used, as in Spatial-MLLM."""
    thr = np.percentile(conf.reshape(-1), 50)
    mask = (conf >= thr) & (conf > 0.1)
    pts = world_points[mask]
    lo, hi = pts.min(0), pts.max(0)
    voxel = (hi - lo).min() / voxels_per_side
    dims = np.floor((hi - lo) / voxel).astype(np.int64) + 1
    sets = []
    for t in range(len(world_points)):
        v = np.floor((world_points[t][mask[t]] - lo) / voxel).astype(np.int64)
        sets.append(np.unique((v[:, 0] * dims[1] + v[:, 1]) * dims[2] + v[:, 2]))
    return sets, {"origin": lo, "voxel_size": float(voxel), "dims": dims, "conf_threshold": float(thr)}


def max_coverage(sets, K):
    """Greedy maximum coverage; returns (picks in selection order, marginal gain of each pick)."""
    covered = np.zeros(0, np.int64)
    picks, gains = [], []
    remaining = list(range(len(sets)))
    for _ in range(K):
        best, best_gain = None, -1
        for t in remaining:  # ascending order + strict '>' = their tie-breaking (first frame wins)
            g = len(np.setdiff1d(sets[t], covered, assume_unique=True))
            if g > best_gain:
                best, best_gain = t, g
        if best is None or best_gain <= 0:
            break
        picks.append(best)
        gains.append(best_gain)
        covered = np.union1d(covered, sets[best])
        remaining.remove(best)
    return picks, gains


def coverage_curve(sets, order):
    """Fraction of all voxels covered after each frame in `order`."""
    total = len(np.unique(np.concatenate(sets)))
    covered, curve = np.zeros(0, np.int64), []
    for t in order:
        covered = np.union1d(covered, sets[t])
        curve.append(len(covered) / total)
    return curve


def uniform_subset(n_candidates, K):
    return np.linspace(0, n_candidates - 1, num=K, dtype=int).tolist()


def space_aware_sampling(scene_dir, K=16):
    wp, conf, frame_indices, _ = load_points(scene_dir)
    sets, grid = voxel_sets(wp, conf)
    picks, gains = max_coverage(sets, K)
    return {"candidates": sorted(picks), "selection_order": picks, "gains": gains,
            "video_frames": frame_indices[sorted(picks)].tolist(), "grid": grid, "sets": sets}


CACHE_K = 64


def cached_selection(scene_dir, K):
    """Space-aware picks (candidate indices, temporal order) for K <= CACHE_K. Greedy selection is incremental,
    so the first K picks of one long run equal a run with budget K; the order is cached in sa_order.json."""
    assert K <= CACHE_K
    f = Path(scene_dir) / "sa_order.json"
    if not f.exists():
        r = space_aware_sampling(scene_dir, CACHE_K)
        f.write_text(json.dumps({"selection_order": r["selection_order"], "gains": r["gains"],
                                 "voxel_size": r["grid"]["voxel_size"], "n_voxels": int(len(np.unique(np.concatenate(r["sets"]))))}))
    return sorted(json.loads(f.read_text())["selection_order"][:K])


def _cache(scene_dir):
    cached_selection(scene_dir, 1)
    return scene_dir.name


if __name__ == "__main__":
    import argparse
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", help="<dataset>/<scene>; omit with --cache-all")
    ap.add_argument("-K", type=int, default=16)
    ap.add_argument("--cache-all", action="store_true", help="write sa_order.json for every scene in data/vggt")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.cache_all:
        scenes = sorted(p.parent for p in VGGT_DIR.glob("*/*/meta.json"))
        with Pool(args.workers) as pool:
            for i, name in enumerate(pool.imap_unordered(_cache, scenes)):
                print(f"[{i + 1}/{len(scenes)}] {name}", flush=True)
    else:
        r = space_aware_sampling(VGGT_DIR / args.video, args.K)
        print("candidates", r["candidates"], "\nvideo frames", r["video_frames"], "\ngains", r["gains"])
