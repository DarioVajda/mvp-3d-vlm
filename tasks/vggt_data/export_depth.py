"""Per-pixel VGGT depth maps for the 75 scenes of vggt_vsibench_75 (an add-on to that export).

Per scene, out_depth/scenes/<dataset>/<scene>/depth.npz:
  depth  float16 [128, H, W]  VGGT depth, same frames, pixels, world frame and scale as cameras.npz of the main export
  conf   float16 [128, H, W]  VGGT depth confidence (>= 1; higher is better)

Checks each scene: unprojecting this depth with the shared cameras and taking the per-patch median reproduces
patch_points.npz of the main export. Usage (login node; numpy only, no model):
  .venv/bin/python tasks/vggt_data/export_depth.py [--out tasks/vggt_data/out_depth]
"""
import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

from mvp.paths import VGGT_DIR

HERE = Path(__file__).parent
PATCH = 14


def unproject(depth, extrinsic, intrinsic):
    T, H, W = depth.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    fx, fy = intrinsic[:, 0, 0, None, None], intrinsic[:, 1, 1, None, None]
    cx, cy = intrinsic[:, 0, 2, None, None], intrinsic[:, 1, 2, None, None]
    cam = np.stack([(u - cx) * depth / fx, (v - cy) * depth / fy, depth], -1)
    R, t = extrinsic[:, :, :3], extrinsic[:, :, 3]
    return np.einsum("tji,thwj->thwi", R, cam - t[:, None, None, :])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default=str(HERE / "out"), help="the main export (for the scene list and the check)")
    ap.add_argument("--out", default=str(HERE / "out_depth"))
    args = ap.parse_args()
    main_root, out = Path(args.main), Path(args.out)
    scenes = [(r["dataset"], r["scene"]) for r in csv.DictReader(open(main_root / "scenes.csv"))]
    worst = 0.0
    for i, (ds, sc) in enumerate(scenes):
        src, shared = VGGT_DIR / ds / sc, main_root / "scenes" / ds / sc
        depth, conf = np.load(src / "depth.npy"), np.load(src / "depth_conf.npy")
        cams, pp = np.load(shared / "cameras.npz"), np.load(shared / "patch_points.npz")
        assert (np.load(src / "frame_indices.npy") == cams["frame_indices"]).all(), f"{ds}/{sc}: frames differ"
        T, H, W = depth.shape
        gh, gw = pp["conf"].shape[1:]
        assert (H, W) == (gh * PATCH, gw * PATCH), f"{ds}/{sc}: size {H}x{W} vs patch grid {gh}x{gw}"
        pts = unproject(depth.astype(np.float32), cams["extrinsic"], cams["intrinsic"])
        xyz = np.median(pts.reshape(T, gh, PATCH, gw, PATCH, 3), axis=(2, 4))
        err = float(np.abs(xyz - pp["xyz"]).max())
        assert err < 1e-4, f"{ds}/{sc}: depth does not reproduce patch_points (max diff {err})"
        worst = max(worst, err)
        dst = out / "scenes" / ds / sc
        dst.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst / "depth.npz", depth=depth.astype(np.float16), conf=conf.astype(np.float16))
        print(f"[{i + 1}/{len(scenes)}] {ds}/{sc} {T}x{H}x{W}", flush=True)
    shutil.copy(main_root / "scenes.csv", out / "scenes.csv")
    print(f"done: {len(scenes)} scenes; max patch-point difference {worst:.2e}")


if __name__ == "__main__":
    main()
