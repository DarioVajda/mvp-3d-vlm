"""Convert the saved VGGT reconstruction of one scene into OpenVocab-4D's geometry files.

No model is run and the source data is only read. Output, in --out:

  frames/         byte-identical copies of the selected frames, under their original names
  cameras.npz     extrinsic [S,3,4] float32, intrinsic [S,3,3] float32, image_paths [S]
  depth.npz       depth [S,H,W] float16, conf [S,H,W] float16; only with --dense-dir
  adapter_report.json   selection, source frame indices, every validation result

Dense depth is never approximated from the patch points. Without --dense-dir, depth.npz is not written and the
pipeline's track and fuse stages cannot run yet. The patch points are used only as validation data.

  py src/mvp/openvocab_adapter/adapt_vggt.py --preset smoke5
  py src/mvp/openvocab_adapter/adapt_vggt.py --preset eval32 --dense-dir <folder with depth.npy, depth_conf.npy>
  py src/mvp/openvocab_adapter/adapt_vggt.py --frames list:0,40,80 --label three_frames
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

PATCH_INSIDE_MIN = 0.99  # share of patch points that must project into their own patch, every frame
CLOUD_RGB_MAX = 10.0     # mean colour error /255 (tasks/vggt_data/check.py measured at most 1.5)
DENSE_PATCH_TOL = 1e-3   # p99 difference to the exported patch points, as a fraction of the scene's size


class Checks:
    """Collects validation results so that one run reports every problem."""

    def __init__(self):
        self.results = []

    def add(self, name, ok, detail=""):
        self.results.append({"check": name, "ok": bool(ok), "detail": str(detail)})
        print(f"[adapter] {'ok  ' if ok else 'FAIL'} {name}{': ' + str(detail) if detail else ''}")
        return bool(ok)

    @property
    def failed(self):
        return [r["check"] for r in self.results if not r["ok"]]


def load_source(scene_dir):
    meta = json.loads((scene_dir / "meta.json").read_text())
    with np.load(scene_dir / "cameras.npz") as z:
        cams = {k: z[k] for k in z.files}
    with np.load(scene_dir / "patch_points.npz") as z:
        patches = {k: z[k] for k in z.files}
    with np.load(scene_dir / "point_cloud.npz") as z:
        cloud = {k: z[k] for k in z.files}
    return meta, cams, patches, cloud


def check_cameras(checks, cams, meta):
    """OpenCV camera-from-world convention. Returns False if the arrays are too malformed to use further."""
    T = meta["num_frames"]
    ext, intr = cams.get("extrinsic"), cams.get("intrinsic")
    if ext is None or intr is None:
        checks.add("cameras: extrinsic and intrinsic present", False, sorted(cams))
        return False
    good = checks.add("cameras: extrinsic shape", ext.shape == (T, 3, 4) and ext.dtype == np.float32,
                      f"{ext.shape} {ext.dtype}")
    good &= checks.add("cameras: intrinsic shape", intr.shape == (T, 3, 3) and intr.dtype == np.float32,
                       f"{intr.shape} {intr.dtype}")
    if not good:
        return False
    if not checks.add("cameras: finite", np.isfinite(ext).all() and np.isfinite(intr).all()):
        return False

    R, t = ext[:, :, :3].astype(np.float64), ext[:, :, 3].astype(np.float64)
    ortho = np.abs(R @ R.transpose(0, 2, 1) - np.eye(3)).max()
    det = np.abs(np.linalg.det(R) - 1).max()
    checks.add("cameras: rotations are orthonormal", ortho < 1e-3, f"max |RR^T-I| = {ortho:.2e}")
    checks.add("cameras: rotations have det +1", det < 1e-3, f"max |det-1| = {det:.2e}")
    origin = max(np.abs(R[0] - np.eye(3)).max(), np.abs(t[0]).max())
    checks.add("cameras: world frame is frame 0's camera", origin < 1e-3, f"max deviation {origin:.2e}")

    center = -np.einsum("tji,tj->ti", R, t)
    if "camera_center" in cams:
        err = np.abs(center - cams["camera_center"]).max()
        checks.add("cameras: camera_center = -R^T t", err < 1e-3, f"max error {err:.2e}")
    if "camera_forward" in cams:
        err = np.abs(R[:, 2] - cams["camera_forward"]).max()
        checks.add("cameras: camera_forward = R[2]", err < 1e-3, f"max error {err:.2e}")

    H, W = meta["image_hw"]
    fx, fy, cx, cy = intr[:, 0, 0], intr[:, 1, 1], intr[:, 0, 2], intr[:, 1, 2]
    bottom = np.abs(intr[:, 2] - np.array([0, 0, 1])).max()
    checks.add("intrinsics: pinhole in pixels, no skew",
               (fx > 0).all() and (fy > 0).all() and np.abs(intr[:, 0, 1]).max() < 1e-3 and bottom < 1e-6,
               f"fx {fx.min():.1f}-{fx.max():.1f}, fy {fy.min():.1f}-{fy.max():.1f}")
    off = max(np.abs(cx - W / 2).max() / W, np.abs(cy - H / 2).max() / H)
    checks.add("intrinsics: principal point near the image centre", off < 0.02, f"max offset {off:.2%} of image size")
    return True


def check_frames(checks, scene_dir, cams, meta):
    """Files, sizes and video indices. Returns the frame paths, or None if they cannot be trusted."""
    T = meta["num_frames"]
    H, W = meta["image_hw"]
    gh, gw = meta["patch_grid_hw"]
    P = meta["patch_size"]
    checks.add("meta: image size is the patch grid", [gh * P, gw * P] == [H, W], f"{gh}x{gw} patches, {H}x{W} pixels")

    frames = common.list_images(scene_dir / "frames")
    expected = [f"{i:03d}.jpg" for i in range(T)]
    if not checks.add("frames: one file per frame, named by index", [p.name for p in frames] == expected,
                      f"{len(frames)} files, expected {T}"):
        return None
    wrong = []
    for p in frames:
        with Image.open(p) as img:
            if img.size != (W, H):
                wrong.append(p.name)
    checks.add("frames: every image has VGGT's input size", not wrong, f"{W}x{H}" if not wrong else wrong[:5])

    idx = cams.get("frame_indices")
    if idx is not None:
        checks.add("frames: video indices increase, one per frame",
                   idx.shape == (T,) and (np.diff(idx) > 0).all(), f"{idx.min()}..{idx.max()}")
    return frames


def check_patch_reprojection(checks, cams, patches, meta):
    """The convention test: with x_cam = R x_world + t and K, each patch point must land in its own patch."""
    T = meta["num_frames"]
    gh, gw = meta["patch_grid_hw"]
    P = meta["patch_size"]
    xyz = patches.get("xyz")
    if xyz is None or xyz.shape != (T, gh, gw, 3):
        checks.add("patch points: shape", False, None if xyz is None else xyz.shape)
        return
    column, row = np.meshgrid(np.arange(gw), np.arange(gh))
    worst = 1.0
    for t in range(T):
        uv, z = common.project(xyz[t].reshape(-1, 3), cams["extrinsic"][t], cams["intrinsic"][t])
        inside = (np.floor(uv[:, 0] / P) == column.ravel()) & (np.floor(uv[:, 1] / P) == row.ravel()) & (z > 0)
        worst = min(worst, float(inside.mean()))
    checks.add("alignment: patch points project into their own patch", worst >= PATCH_INSIDE_MIN,
               f"worst frame {worst:.1%}")


def check_cloud_colours(checks, frames, cams, cloud, selected):
    """The selected JPEGs must be the images behind the reconstruction (and be in the right order)."""
    errors = {}
    for t in selected:
        chosen = cloud["frame"] == t
        if not chosen.any():
            continue
        with Image.open(frames[t]) as handle:
            image = np.asarray(handle.convert("RGB")).astype(np.float32)
        uv, _ = common.project(cloud["xyz"][chosen], cams["extrinsic"][t], cams["intrinsic"][t])
        u, v = np.round(uv).astype(int).T
        inside = (u >= 0) & (u < image.shape[1]) & (v >= 0) & (v < image.shape[0])
        if inside.any():
            errors[t] = float(np.abs(image[v[inside], u[inside]] - cloud["rgb"][chosen][inside]).mean())
    if not errors:
        checks.add("alignment: cloud colours match the selected frames", False, "no cloud points in these frames")
        return
    worst = max(errors, key=errors.get)
    checks.add("alignment: cloud colours match the selected frames", errors[worst] <= CLOUD_RGB_MAX,
               f"{len(errors)}/{len(selected)} frames have points, worst frame {worst}: {errors[worst]:.2f}/255")


def check_dense(checks, dense_dir, cams, patches, meta, selected):
    """Validate the dense arrays against everything exported. Returns (depth, conf) memory maps, or None."""
    for name in ("depth.npy", "depth_conf.npy"):
        if not checks.add(f"dense: {name} exists", (dense_dir / name).is_file(), dense_dir / name):
            return None
    depth = np.load(dense_dir / "depth.npy", mmap_mode="r")
    conf = np.load(dense_dir / "depth_conf.npy", mmap_mode="r")
    expected = (meta["num_frames"], *meta["image_hw"])
    good = checks.add("dense: depth shape", depth.shape == expected, f"{depth.shape}, expected {expected}")
    good &= checks.add("dense: conf shape", conf.shape == expected, f"{conf.shape}, expected {expected}")
    if not good:
        return None
    # A wider source type would lose precision in the fp16 file the pipeline expects, so it is not accepted silently.
    checks.add("dense: stored as float16 (no precision lost)",
               depth.dtype == np.float16 and conf.dtype == np.float16, f"{depth.dtype}, {conf.dtype}")

    gh, gw = meta["patch_grid_hw"]
    P = meta["patch_size"]
    sel = sorted(selected)
    finite = positive = confident = True
    diffs, conf_diffs, sizes = [], [], []
    for t in sel:
        d, c = np.asarray(depth[t], np.float32), np.asarray(conf[t], np.float32)
        finite &= bool(np.isfinite(d).all() and np.isfinite(c).all())
        positive &= bool((d > 0).any())
        confident &= bool(c.min() >= 1.0 - 1e-3)  # VGGT's confidence is 1 + exp(x)
        # exactly how tasks/vggt_data/export.py made the patch points, so a mismatch means wrong scene, frame or order
        world = common.unproject_frame(d, cams["extrinsic"][t], cams["intrinsic"][t])
        median = np.median(world.reshape(gh, P, gw, P, 3), axis=(1, 3))
        diffs.append(np.abs(median - patches["xyz"][t]).max(-1).ravel())
        sizes.append(np.linalg.norm(patches["xyz"][t].reshape(-1, 3), axis=1))
        patch_conf = c.reshape(gh, P, gw, P).mean(axis=(1, 3))
        conf_diffs.append(np.abs(patch_conf - patches["conf"][t]).ravel())
    checks.add("dense: finite", finite)
    checks.add("dense: depth has positive values in every selected frame", positive)
    checks.add("dense: confidence is at least 1", confident)
    scale = float(np.median(np.concatenate(sizes))) or 1.0
    error = float(np.percentile(np.concatenate(diffs), 99)) / scale
    checks.add("alignment: dense depth reproduces the exported patch points", error <= DENSE_PATCH_TOL,
               f"p99 error {error:.2e} of scene size, limit {DENSE_PATCH_TOL:.0e}")
    conf_error = float(np.percentile(np.concatenate(conf_diffs), 99))
    limit = max(1e-2, 1e-3 * float(np.median(patches["conf"])))
    checks.add("alignment: dense confidence reproduces the exported patch confidence", conf_error <= limit,
               f"p99 error {conf_error:.2e}, limit {limit:.2e}")

    # The full data/vggt/<dataset>/<scene>/ folder also has these; compare them when they came along.
    for name, source in (("extrinsic.npy", "extrinsic"), ("intrinsic.npy", "intrinsic"),
                         ("frame_indices.npy", "frame_indices")):
        if (dense_dir / name).is_file() and source in cams:
            other = np.load(dense_dir / name)
            checks.add(f"dense: {name} equals the exported {source}",
                       other.shape == cams[source].shape and np.array_equal(other, cams[source]))
    return depth, conf


def resolve_out(args, meta, label, scene_dir, data_root):
    out = Path(args.out) if args.out else common.OUT_ROOT / f"{meta['dataset']}__{meta['scene']}" / label / "adapted"
    out = out.resolve()
    for protected in (scene_dir.resolve(), data_root.resolve()):
        if out == protected or out.is_relative_to(protected):
            sys.exit(f"[adapter] refusing to write inside the source data: {out}")
    return out


def prepare_out(out, overwrite):
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            sys.exit(f"[adapter] {out} is not empty; pass --overwrite to replace the adapter's files")
        if (out / "tracks").exists() or (out / "labels.npz").exists():
            print(f"[adapter] warning: {out} has tracking or fusion results that may now be stale")
        for name in ("cameras.npz", "depth.npz", "adapter_report.json"):
            (out / name).unlink(missing_ok=True)
        shutil.rmtree(out / "frames", ignore_errors=True)
    (out / "frames").mkdir(parents=True, exist_ok=True)


def write_outputs(out, frames, cams, selected, dense):
    copies = []
    for t in selected:
        copy = out / "frames" / frames[t].name
        shutil.copyfile(frames[t], copy)
        copies.append(str(copy))
    # exactly what milestone 1 / milestone 4's "vggt" stage write
    np.savez(out / "cameras.npz", extrinsic=cams["extrinsic"][selected], intrinsic=cams["intrinsic"][selected],
             image_paths=np.array(copies))
    if dense is not None:
        depth, conf = dense
        np.savez_compressed(out / "depth.npz", depth=np.stack([depth[t] for t in selected]).astype(np.float16),
                            conf=np.stack([conf[t] for t in selected]).astype(np.float16))


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default=common.DEFAULT_SCENE, help="<dataset>/<scene>")
    p.add_argument("--data-root", default=str(common.DATA_ROOT), help="folder holding the exported scenes/")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--preset", choices=sorted(common.PRESETS))
    group.add_argument("--frames", help="selection: all | uniform:K | subset:NAME | window:START:COUNT[:STRIDE] | list:a,b")
    p.add_argument("--label", help="name of this selection in the output path (default: the preset or the spec)")
    p.add_argument("--dense-dir", help="folder with the saved depth.npy and depth_conf.npy for this scene")
    p.add_argument("--out", help="output folder (default tasks/openvocab_adapter/out/<dataset>__<scene>/<label>/adapted)")
    p.add_argument("--overwrite", action="store_true", help="replace an earlier run's adapter files")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_root = Path(args.data_root)
    scene_dir = data_root / "scenes" / args.scene
    if not (scene_dir / "meta.json").is_file():
        sys.exit(f"[adapter] no exported scene at {scene_dir}")
    spec = common.PRESETS[args.preset] if args.preset else args.frames
    label = args.label or args.preset or spec.replace(":", "-").replace(",", "_")

    meta, cams, patches, cloud = load_source(scene_dir)
    selected = common.select_frames(spec, meta["num_frames"], meta)
    out = resolve_out(args, meta, label, scene_dir, data_root)
    print(f"[adapter] {args.scene}: {len(selected)} of {meta['num_frames']} frames ({spec})")

    checks = Checks()
    usable = check_cameras(checks, cams, meta)
    frames = check_frames(checks, scene_dir, cams, meta)
    dense = None
    if usable:
        check_patch_reprojection(checks, cams, patches, meta)
        if frames is not None:
            check_cloud_colours(checks, frames, cams, cloud, selected)
        if args.dense_dir:
            dense = check_dense(checks, Path(args.dense_dir), cams, patches, meta, selected)
    if checks.failed or frames is None or not usable:
        print(f"[adapter] {len(checks.failed)} check(s) failed, nothing written: {checks.failed}")
        return 1

    prepare_out(out, args.overwrite)
    write_outputs(out, frames, cams, selected, dense)
    depth_status = "written" if dense is not None else "missing: pass --dense-dir with depth.npy and depth_conf.npy"
    source_indices = cams["frame_indices"][selected].tolist() if "frame_indices" in cams else None
    (out / "adapter_report.json").write_text(json.dumps({
        "scene": args.scene, "frame_selection": spec, "label": label,
        "selected_frames": selected, "source_video_frames": source_indices,
        "image_hw": meta["image_hw"], "depth_npz": depth_status,
        "world_frame": "camera of frame 0 of the full 128-frame reconstruction, which a selection may not include",
        "scale": "VGGT's arbitrary scale of the full 128-frame reconstruction",
        "checks": checks.results,
    }, indent=1))
    print(f"[adapter] wrote {len(selected)} frames and cameras.npz to {out}")
    if dense is None:
        print("[adapter] depth.npz NOT written: the dense depth files were not given, "
              "so track and fuse cannot run on this output yet")
    else:
        print("[adapter] wrote depth.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
