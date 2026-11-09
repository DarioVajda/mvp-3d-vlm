"""Scene sheet: a CSV of the objects found in an OpenVocab-4D scene, plus a CSV of camera poses.

Takes a list of prompts and either an existing scene (the --out folder of `ov4d`) or a folder of
frames to reconstruct. Missing pieces are computed first: geometry (VGGT) needs --images, and any
prompt without tracks/<prompt>.npz gets the pipeline's "track" stage (SAM 3 + 3D tracking).

Outputs (in --out-dir, default <scene>/scene_sheet/):

  objects.csv   one row per object; a prompt that found several objects gives several rows
                  object_id, concept, track_id     track_id matches tracks/<concept>.npz and the
                                                   Rerun entity world/objects/<concept>/id<track_id>
                  x, y, z                          global position (world frame)
                  size_x, size_y, size_z, size     only with --size
                  num_frames                       frames the object was seen in
                  frame_<t>                        1 if seen in frame t, else 0
                  or, with --relative-positions:
                  frame_<t>_x, _y, _z              position in frame t's camera coordinates,
                                                   empty when not seen
  cameras.csv   one row per frame
                  frame, image, x, y, z            camera centre (world frame)
                  qw, qx, qy, qz                   camera orientation: the rotation taking camera
                                                   axes to world axes, as a unit quaternion
                  forward_x, forward_y, forward_z  optical axis (camera +z) in world coordinates

Conventions
  - World frame = camera of frame 0. Camera coordinates are OpenCV: x right, y down, z forward.
  - Units are VGGT's arbitrary scene scale, not metres; sizes are in those units cubed.
  - "Seen in frame t": the object's tracked mask in frame t has at least --min-points pixels with
    confident depth. This is the rule the pipeline's fuse stage uses, so num_frames equals
    report.json's frames_seen.
  - Positions are per-axis medians of the object's 3D points, which is robust to the stray points
    that depth edges produce. Global: all frames, world coordinates. Per frame: that frame's
    points, in that camera's coordinates (where the object was seen, not the global position
    re-expressed).
  - Size: no metric size exists, so each object is approximated by the cuboid its points span
    along the world axes: size_x/y/z = max - min of its points on each axis (or the --size-trim
    and 100 - --size-trim percentiles, to ignore outliers), size = size_x * size_y * size_z.

Usage (from the repo root, with the project venv):
  python scene_sheet_creation/main.py --scene out/test_scene --prompts "table,dog,chair"
  python scene_sheet_creation/main.py --images path/to/frames --scene out/my_scene ^
      --prompts table dog chair --relative-positions --size
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "trackA" / "milestone4_video.py"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}  # same as the pipeline


def parse_prompts(values):
    """Accepts "table,dog,chair" and/or separate arguments; keeps multi-word prompts."""
    prompts = [p.strip() for v in values for p in v.split(",") if p.strip()]
    return list(dict.fromkeys(prompts))  # drop duplicates, keep order


def track_file(scene, prompt):
    return scene / "tracks" / f"{prompt.replace(' ', '_')}.npz"


# ---------------------------------------------------------------------------
# running the pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    images, scene, prompts, stages, step, chunk_size=40, overlap=8, no_reid=False
):
    cmd = [
        sys.executable,
        "-u",
        str(PIPELINE),
        "--images",
        str(images),
        "--out",
        str(scene),
        "--prompts",
        ",".join(prompts),
        "--stages",
        stages,
        "--step",
        str(step),
        "--chunk-size",
        str(chunk_size),
        "--overlap",
        str(overlap),
    ]
    if no_reid:
        cmd.append("--no-reid")
    print(f"[sheet] running pipeline stages '{stages}' for {prompts}")
    if subprocess.run(cmd, cwd=str(ROOT)).returncode != 0:
        sys.exit("[sheet] pipeline failed, see its log above")


def frames_used_by(scene):
    """Recover the frame folder and --step a scene's geometry was built from.

    The track stage re-lists the frames itself and must see exactly the frames VGGT saw.
    """
    paths = [Path(str(p)) for p in np.load(scene / "cameras.npz")["image_paths"]]
    folder = paths[0].parent
    if not folder.is_dir():
        sys.exit(
            f"[sheet] frame folder {folder} (recorded in cameras.npz) no longer exists"
        )
    names = [p.name for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    used = [p.name for p in paths]
    for step in range(1, len(names) + 1):
        if names[::step] == used:
            return folder, step
    sys.exit(f"[sheet] the frames in {folder} no longer match cameras.npz")


def ensure_scene(
    scene, prompts, images=None, step=1, chunk_size=40, overlap=8, no_reid=False
):
    """Make sure the scene has geometry and tracks for every prompt, running the pipeline if not."""
    if not ((scene / "cameras.npz").exists() and (scene / "depth.npz").exists()):
        if images is None:
            sys.exit(
                f"[sheet] {scene} has no geometry (cameras.npz, depth.npz): "
                "pass --images to reconstruct it"
            )
        run_pipeline(
            images,
            scene,
            prompts,
            "vggt,track,fuse",
            step,
            chunk_size,
            overlap,
            no_reid,
        )
        return
    missing = [p for p in prompts if not track_file(scene, p).exists()]
    if missing:
        # "track" only: rerunning "fuse" would overwrite objects.rrd / labels.npz with just these
        # prompts. The scene sheet only needs the tracks.
        folder, scene_step = frames_used_by(scene)
        run_pipeline(folder, scene, missing, "track", scene_step, no_reid=no_reid)


# ---------------------------------------------------------------------------
# building the sheet
# ---------------------------------------------------------------------------
def rotation_to_quaternion(m):
    """3x3 rotation matrix -> unit quaternion (w, x, y, z) with w >= 0."""
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        ]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        ]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        ]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [
            (m[1, 0] - m[0, 1]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
        ]
    q = np.array(q) / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return f"{v:.6g}"
    return v


def write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows([fmt(v) for v in row] for row in rows)


def collect_objects(
    scene,
    prompts,
    depth,
    conf,
    extrinsic,
    intrinsic,
    conf_percentile,
    min_points,
    max_frame_points,
):
    """Per-frame 3D observations of every tracked object, keyed by (prompt index, track id)."""
    S, H, W = depth.shape
    volumes = {}
    for pi, prompt in enumerate(prompts):
        labels = np.load(track_file(scene, prompt))["labels"]
        if labels.shape != depth.shape:
            sys.exit(
                f"[sheet] {track_file(scene, prompt)} has shape {labels.shape} but the "
                f"geometry is {depth.shape}: delete it to re-track"
            )
        volumes[pi] = labels

    # same confident-depth rule as the pipeline: one percentile over the whole sequence
    conf_thr = np.percentile(conf.astype(np.float32), conf_percentile)
    rng = np.random.default_rng(0)
    objects = {}
    for t in range(S):
        d = depth[t].astype(np.float32)
        keep = (conf[t].astype(np.float32) >= conf_thr) & (d > 0)
        R, tvec = extrinsic[t, :, :3], extrinsic[t, :, 3]
        fx, fy, cx, cy = (
            intrinsic[t, 0, 0],
            intrinsic[t, 1, 1],
            intrinsic[t, 0, 2],
            intrinsic[t, 1, 2],
        )
        for pi, labels in volumes.items():
            lab = labels[t]
            for tid in np.unique(lab):
                if tid == 0:
                    continue
                ys, xs = np.nonzero((lab == tid) & keep)
                if len(ys) < min_points:
                    continue
                z = d[ys, xs]
                cam = np.stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z], axis=1)
                world = (
                    cam - tvec
                ) @ R  # x_cam = R x_world + t  ->  x_world = R^T (x_cam - t)
                obj = objects.setdefault(
                    (pi, int(tid)),
                    {
                        "frames": {},
                        "sample": [],
                        "lo": np.full(3, np.inf),
                        "hi": np.full(3, -np.inf),
                    },
                )
                obj["frames"][t] = np.median(cam, axis=0)
                obj["lo"] = np.minimum(obj["lo"], world.min(axis=0))
                obj["hi"] = np.maximum(obj["hi"], world.max(axis=0))
                if len(world) > max_frame_points:
                    world = world[
                        rng.choice(len(world), max_frame_points, replace=False)
                    ]
                obj["sample"].append(world)
    return objects


def build_scene_sheet(
    scene,
    prompts,
    relative_positions=False,
    include_size=False,
    out_dir=None,
    conf_percentile=55.0,
    min_points=300,
    size_trim=0.0,
    max_frame_points=5000,
):
    """Write objects.csv and cameras.csv for a scene that already has tracks for every prompt.

    Returns the two CSV paths.
    """
    scene = Path(scene)
    out_dir = Path(out_dir) if out_dir else scene / "scene_sheet"
    out_dir.mkdir(parents=True, exist_ok=True)

    cams = np.load(scene / "cameras.npz")
    dep = np.load(scene / "depth.npz")
    extrinsic = cams["extrinsic"].astype(np.float64)
    intrinsic = cams["intrinsic"].astype(np.float64)
    image_paths = [str(p) for p in cams["image_paths"]]
    depth, conf = dep["depth"], dep["conf"]
    S = depth.shape[0]

    objects = collect_objects(
        scene,
        prompts,
        depth,
        conf,
        extrinsic,
        intrinsic,
        conf_percentile,
        min_points,
        max_frame_points,
    )

    width = max(3, len(str(S - 1)))
    frame_names = [f"frame_{t:0{width}d}" for t in range(S)]
    header = ["object_id", "concept", "track_id", "x", "y", "z"]
    if include_size:
        header += ["size_x", "size_y", "size_z", "size"]
    header.append("num_frames")
    if relative_positions:
        header += [f"{n}_{a}" for n in frame_names for a in "xyz"]
    else:
        header += frame_names

    rows = []
    for (pi, tid), obj in sorted(
        objects.items()
    ):  # same order as the pipeline's fuse stage
        sample = np.concatenate(obj["sample"])
        row = [len(rows), prompts[pi], tid, *np.median(sample, axis=0)]
        if include_size:
            if size_trim > 0:
                lo, hi = np.percentile(sample, [size_trim, 100 - size_trim], axis=0)
            else:
                lo, hi = obj["lo"], obj["hi"]
            extent = hi - lo
            row += [*extent, float(np.prod(extent))]
        row.append(len(obj["frames"]))
        for t in range(S):
            pos = obj["frames"].get(t)
            if relative_positions:
                row += [None] * 3 if pos is None else list(pos)
            else:
                row.append(0 if pos is None else 1)
        rows.append(row)
    objects_csv = out_dir / "objects.csv"
    write_csv(objects_csv, header, rows)

    cam_rows = []
    for t in range(S):
        R, tvec = extrinsic[t, :, :3], extrinsic[t, :, 3]
        centre = -R.T @ tvec
        cam_rows.append(
            [t, image_paths[t], *centre, *rotation_to_quaternion(R.T), *R[2]]
        )
    cameras_csv = out_dir / "cameras.csv"
    write_csv(
        cameras_csv,
        [
            "frame",
            "image",
            "x",
            "y",
            "z",
            "qw",
            "qx",
            "qy",
            "qz",
            "forward_x",
            "forward_y",
            "forward_z",
        ],
        cam_rows,
    )

    counts = {
        p: sum(1 for (pi, _) in objects if pi == i) for i, p in enumerate(prompts)
    }
    print(f"[sheet] {len(rows)} objects over {S} frames: {counts}")
    print(f"[sheet] wrote {objects_csv}")
    print(f"[sheet] wrote {cameras_csv}")
    return objects_csv, cameras_csv


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--prompts",
        nargs="+",
        required=True,
        help='concepts, e.g. "table,dog,chair" or table dog chair',
    )
    p.add_argument(
        "--scene",
        required=True,
        help="scene folder (the pipeline's --out); created if --images is given",
    )
    p.add_argument(
        "--images",
        default=None,
        help="folder of frames, needed only when the scene has no geometry yet",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="where to write the CSVs (default <scene>/scene_sheet)",
    )
    p.add_argument(
        "--relative-positions",
        action="store_true",
        help="per-frame camera-frame positions instead of 0/1 appearance flags",
    )
    p.add_argument(
        "--size", action="store_true", help="add the cuboid size of each object"
    )
    p.add_argument(
        "--size-trim",
        type=float,
        default=0.02,
        help="percent of points ignored at each end of every axis when measuring size "
        "(0 = farthest points)",
    )
    p.add_argument(
        "--min-points",
        type=int,
        default=300,
        help="confident pixels needed for an object to count as seen in a frame "
        "(the pipeline's default)",
    )
    p.add_argument(
        "--conf-percentile",
        type=float,
        default=55.0,
        help="depth-confidence percentile below which pixels are ignored (the pipeline's default)",
    )
    p.add_argument(
        "--max-frame-points",
        type=int,
        default=5000,
        help="points per object per frame used for the global median",
    )
    # passed to the pipeline when it has to build the geometry
    p.add_argument(
        "--step", type=int, default=1, help="use every Nth frame of --images"
    )
    p.add_argument("--chunk-size", type=int, default=40, help="VGGT frames per chunk")
    p.add_argument(
        "--overlap", type=int, default=8, help="frames shared between VGGT chunks"
    )
    p.add_argument(
        "--no-reid",
        action="store_true",
        help="track without DINOv2 re-identification (fewer duplicate objects, shorter tracks)",
    )
    args = p.parse_args()

    prompts = parse_prompts(args.prompts)
    scene = Path(args.scene)
    ensure_scene(
        scene,
        prompts,
        Path(args.images) if args.images else None,
        args.step,
        args.chunk_size,
        args.overlap,
        args.no_reid,
    )
    build_scene_sheet(
        scene,
        prompts,
        args.relative_positions,
        args.size,
        args.out_dir,
        args.conf_percentile,
        args.min_points,
        args.size_trim,
        args.max_frame_points,
    )


if __name__ == "__main__":
    main()
