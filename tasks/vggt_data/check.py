"""Consistency checks on the exported scenes (export.py), reading only the exported files.

  * frame 0's camera is the world origin;
  * patch points reproject into their own patch with their frame's camera;
  * cloud points reproject onto pixels of their source frame JPEG with matching colour (so the frames are the
    exact images VGGT saw);
  * every frame / question / subset index is in range.

  .venv/bin/python tasks/vggt_data/check.py [tasks/vggt_data/out]
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def project(xyz, ext, intr):
    cam = xyz @ ext[:3, :3].T + ext[:3, 3]
    uvw = cam @ intr.T
    return uvw[:, :2] / uvw[:, 2:], cam[:, 2]


def check_scene(d):
    meta = json.loads((d / "meta.json").read_text())
    cams, pp, cloud = np.load(d / "cameras.npz"), np.load(d / "patch_points.npz"), np.load(d / "point_cloud.npz")
    T, (gh, gw), P = meta["num_frames"], meta["patch_grid_hw"], meta["patch_size"]
    ext, intr = cams["extrinsic"], cams["intrinsic"]
    frames = sorted((d / "frames").glob("*.jpg"))
    assert len(frames) == T == len(ext) == len(pp["xyz"]), d
    assert pp["xyz"].shape == (T, gh, gw, 3) and np.isfinite(pp["xyz"]).all()
    assert np.abs(cams["camera_center"][0]).max() < 1e-3, cams["camera_center"][0]
    assert all(0 <= i < T for s in meta["frame_subsets"].values() for i in s)
    assert cloud["frame"].min() >= 0 and cloud["frame"].max() < T

    # patch points: the projected median lands inside its own 14x14 patch
    inside = []
    for t in range(0, T, 8):
        uv, z = project(pp["xyz"][t].reshape(-1, 3), ext[t], intr[t])
        col, row = np.meshgrid(np.arange(gw), np.arange(gh))
        ok = (np.floor(uv[:, 0] / P) == col.ravel()) & (np.floor(uv[:, 1] / P) == row.ravel()) & (z > 0)
        inside.append(ok.mean())

    # cloud colours vs the JPEG pixel they project to (depth unprojection uses integer pixel coordinates)
    errs = []
    for t in range(0, T, 16):
        sel = cloud["frame"] == t
        if not sel.any():
            continue
        img = np.asarray(Image.open(frames[t])).astype(np.float32)
        uv, _ = project(cloud["xyz"][sel], ext[t], intr[t])
        u, v = np.round(uv).astype(int).T
        ok = (u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < img.shape[0])
        errs.append(np.abs(img[v[ok], u[ok]] - cloud["rgb"][sel][ok]).mean())
    return {"scene": f"{meta['dataset']}/{meta['scene']}", "patch_inside": float(np.mean(inside)),
            "cloud_rgb_err": float(np.mean(errs)), "questions": sum(1 for _ in open(d / "questions.jsonl"))}


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "out")
    rows = [check_scene(d) for d in sorted((root / "scenes").glob("*/*")) if d.is_dir()]
    for r in rows:
        print(f"{r['scene']:28s} patch points in own patch {r['patch_inside']:.1%}  "
              f"cloud colour error {r['cloud_rgb_err']:.1f}/255  {r['questions']} questions")
    worst_patch, worst_rgb = min(r["patch_inside"] for r in rows), max(r["cloud_rgb_err"] for r in rows)
    print(f"{len(rows)} scenes; worst patch-inside {worst_patch:.1%}, worst colour error {worst_rgb:.1f}")


if __name__ == "__main__":
    main()
