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
    usable depth (--depth-rule, the tracker's rule: usable_pixels). This is the rule the pipeline's
    track and fuse stages use, so num_frames equals report.json's frames_seen.
      mask   (default) confidence at least the lower of the sequence's --conf-percentile and the
             mask's own --mask-conf-quantile; --min-points 50 by default
      global (the original rule) confidence at least the sequence's --conf-percentile; --min-points
             300 by default
  - Positions are per-axis medians of the object's 3D points, which is robust to the stray points
    that depth edges produce. Global: all frames, world coordinates. Per frame: that frame's
    points, in that camera's coordinates (where the object was seen, not the global position
    re-expressed). Global positions and sizes use at most --max-frame-points of each frame's points.
  - Size: no metric size exists, so each object is approximated by the smallest box around its
    points, turned to fit the object (fit_box), so an object standing at an angle to the world axes
    is not overestimated. A side's length is max - min of the points along it (or between the
    --size-trim and 100 - --size-trim percentiles, to ignore outliers). size_x/y/z are the sides
    nearest the world x, y and z axes, and size = size_x * size_y * size_z.

  --no-frames drops num_frames and the per-frame columns, and --no-cameras drops cameras.csv.

Gold labels (--source gold): the sheet comes from VSI-Bench's ground-truth boxes (GOLD_REPO's data/meta_info/,
downloaded to GOLD_DIR the first time) instead of SAM 3 and tracking. --scene is a VSI-Bench scene, e.g.
scannet/scene0025_01. Columns as above up to size, with box centres and sides, in metres, in the dataset's frame
(z up). The labels' boxes are turned to fit their objects too (axesLengths along normalizedAxes), and their sides are
matched to x, y and z the same way. Prompts match class names ignoring case and a plural "s" (plus GOLD_ALIASES);
without --prompts every class is listed. There are no per-frame columns and no cameras.csv, and asking for them is an
error: the metadata has nothing per frame, and the saved VGGT cameras are in a different frame and scale with nothing
to align them by.

Usage (with the OpenVocab-4D environment; paths are relative to where you run it):
  python src/mvp/openvocab_adapter/scene_sheet_creator.py --scene out/test_scene --prompts "table,dog,chair"
  python src/mvp/openvocab_adapter/scene_sheet_creator.py --images path/to/frames --scene out/my_scene ^
      --prompts table dog chair --relative-positions --size
  python src/mvp/openvocab_adapter/scene_sheet_creator.py --source gold --scene scannet/scene0025_01 ^
      --prompts "table,chair,fan" --size --out-dir out/gold_sheet

To go from a question about one of the saved VGGT scenes straight to its scene sheet, use
question_scene_sheet.py, which picks the prompts, runs SAM 3 and calls the functions below.
"""

import argparse
import csv
import hashlib
import itertools
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

PIPELINE = common.TRACK_A / "milestone4_video.py"
IMAGE_EXTS = common.IMAGE_EXTS  # same as the pipeline

# Which pixels of a mask have usable depth: the same rules as openvocab-4D-main/trackA/depth_rule.py, which the tracker
# uses (tests/test_openvocab_adapter.py checks that the two agree). They are written out here because this file also
# runs from copies of src/ outside the repository (experiments/2026-11-18-scene-sheet-formats/merge_points.py), where
# trackA/ is not there to import from.
DEPTH_RULES = ("mask", "global")  # the default first
MIN_POINTS = {"global": 300, "mask": 50}  # pixels a detection needs under each rule (trackA/depth_rule.py)
MASK_CONF_QUANTILE = 50.0
MAX_OBJECT_POINTS = 80_000  # points an object's centre and size are computed from (16 frames x 5000 at most before)


def usable_pixels(mask, depth, conf, global_thr, rule="mask", quantile=MASK_CONF_QUANTILE):
    """The pixels of mask whose depth counts: depth > 0 and confidence at least global_thr (global), or at least the
    lower of global_thr and the mask's own quantile (mask). conf: float array of the frame's depth confidence."""
    if rule not in DEPTH_RULES:
        raise ValueError(f"depth rule must be one of {DEPTH_RULES}, not {rule!r}")
    valid = mask & (depth > 0)
    if rule == "global":
        return valid & (conf >= global_thr)
    if not valid.any():
        return valid
    return valid & (conf >= min(global_thr, np.percentile(conf[valid], quantile)))


def min_points_for(rule, min_points=None):
    """min_points, or the rule's default (MIN_POINTS) when it is None."""
    if rule not in DEPTH_RULES:
        raise ValueError(f"depth rule must be one of {DEPTH_RULES}, not {rule!r}")
    return MIN_POINTS[rule] if min_points is None else int(min_points)

# Gold labels: VSI-Bench's metadata (Apache-2.0), at the commit that added it (2025-08-05), with the files' SHA-256
GOLD_DIR = common.REPO / "data" / "raw" / "VSI-Bench" / "meta_info"
GOLD_REPO = "vision-x-nyu/thinking-in-space"
GOLD_COMMIT = "ba7a6b2d2ea094ebbfcd2cd6e98e611f40e7a46f"
GOLD_FILES = {
    "scannet": ("scannet_meta_info_val.json", "569d91e0fcae23a83bb35dc390cb0077c4b4ba03011c34d265df7a361e24c43a"),
    "scannetpp": ("scannetpp_meta_info_val.json", "ae414913186a477a79d8b753d2ab8039d8912088dda062ac80a2b71f487f5bb1"),
    "arkitscenes": ("arkitscenes_meta_info_val.json", "77b1f90a33b4aaa032280cdfd6e43986a3e3d818c750b2e1059b1c664ecdc21b"),
}
# Names the questions use for a class the metadata names otherwise (scannetpp/1ada7a0617's questions say "coat rack")
GOLD_ALIASES = {"coat rack": "coat hanger"}


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
    images,
    scene,
    prompts,
    stages,
    step,
    chunk_size=40,
    overlap=8,
    no_reid=False,
    masks_dir=None,
    conf_percentile=None,
    min_points=None,
    stdout=None,
    association=None,
    depth_rule=None,
    mask_conf_quantile=None,
    tracker_options=None,
):
    """Run stages of the OpenVocab-4D pipeline (milestone 4) on a scene folder.

    masks_dir: saved SAM 3 masks (sam3_infer.py) for the track stage, instead of running SAM 3.
    conf_percentile, min_points: the pipeline's confident-depth rule; None keeps its defaults.
    association, depth_rule, mask_conf_quantile: milestone 4's --association, --depth-rule and --mask-conf-quantile;
    tracker_options: more of its options, {name: value} as in its --help without the dashes ("voxel_frac": 0.065).
    None leaves an option out, so milestone 4 uses its own default (the original behaviour).
    stdout: where the pipeline's log goes (default: this process's stdout).
    """
    cmd = [
        sys.executable,
        "-u",
        str(PIPELINE),
        "--images",
        str(Path(images).resolve()),
        "--out",
        str(Path(scene).resolve()),
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
    if masks_dir is not None:
        cmd += ["--masks-dir", str(Path(masks_dir).resolve())]
    if conf_percentile is not None:
        cmd += ["--conf-percentile", str(conf_percentile)]
    if min_points is not None:
        cmd += ["--min-points", str(min_points)]
    options = {"association": association, "depth_rule": depth_rule, "mask_conf_quantile": mask_conf_quantile,
               **(tracker_options or {})}
    for name, value in options.items():
        if value is not None:
            cmd += [f"--{name.replace('_', '-')}", str(value)]
    print(f"[sheet] running pipeline stages '{stages}' for {prompts}")
    if subprocess.run(cmd, stdout=stdout).returncode != 0:
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
    scene,
    prompts,
    images=None,
    step=1,
    chunk_size=40,
    overlap=8,
    no_reid=False,
    masks_dir=None,
    association="overlap",
    depth_rule="mask",
    mask_conf_quantile=MASK_CONF_QUANTILE,
    min_points=None,
):
    """Make sure the scene has geometry and tracks for every prompt, running the pipeline if not.

    With masks_dir, the track stage reads those saved SAM 3 masks instead of running SAM 3. association, depth_rule,
    mask_conf_quantile and min_points go to the track stage (and to fuse); see run_pipeline.
    """
    tracking = dict(association=association, depth_rule=depth_rule, mask_conf_quantile=mask_conf_quantile,
                    min_points=min_points_for(depth_rule, min_points))
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
            masks_dir,
            **tracking,
        )
        return
    missing = [p for p in prompts if not track_file(scene, p).exists()]
    if missing:
        # "track" only: rerunning "fuse" would overwrite objects.rrd / labels.npz with just these
        # prompts. The scene sheet only needs the tracks.
        folder, scene_step = frames_used_by(scene)
        run_pipeline(
            folder, scene, missing, "track", scene_step, no_reid=no_reid, masks_dir=masks_dir, **tracking
        )


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


# ---------------------------------------------------------------------------
# object sizes: the smallest box around an object's points, turned to fit it
# ---------------------------------------------------------------------------
BOX_SEARCH_POINTS = 5000  # points fit_box searches the orientation with; the sides are then measured on all of them
TURN_GRIDS = [  # turns tried about one axis, each grid around the best of the one before
    np.deg2rad(np.arange(-45.0, 45.0, 3.0)),  # a box is the same after a quarter turn, so this covers every turn
    np.deg2rad(np.linspace(-3.0, 3.0, 21)),
    np.deg2rad(np.linspace(-0.3, 0.3, 21)),
]
AXIS_ORDERS = list(itertools.permutations(range(3)))


def spans(values, trim=0.0):
    """Length of each row of values along its last axis: max - min, or between the trim and 100 - trim percentiles."""
    if trim > 0:
        lo, hi = np.percentile(values, [trim, 100 - trim], axis=-1)
        return hi - lo
    return values.max(axis=-1) - values.min(axis=-1)


def match_world_axes(axes, sides):
    """Reorder a box's axes (the columns of axes) and sides so that the i-th is the axis nearest world axis i, and
    flip each axis to point along its world axis. "Nearest" is for all three together: the order with the largest
    sum of |cos(angle)| between each axis and its world axis."""
    order = list(max(AXIS_ORDERS, key=lambda o: sum(abs(axes[i, o[i]]) for i in range(3))))
    axes = axes[:, order]
    return axes * np.where(np.diag(axes) < 0, -1.0, 1.0), np.asarray(sides, float)[order]


def turn_to_fit(points, axes, trim, eps, max_rounds=10):
    """fit_box's search from one starting orientation: turn the box about each of its axes in turn, to the angle that
    gives the smallest cross-section, until a round changes the volume by less than one part in 10^9.
    Returns (axes, volume); volume counts eps on every side."""
    axes = axes.copy()
    proj = (points @ axes).T  # 3 x n: each point's coordinate along each box axis
    sides = spans(proj, trim) + eps
    for _ in range(max_rounds):
        before = sides.prod()
        for k in range(3):
            i, j = (k + 1) % 3, (k + 2) % 3
            angle, area = 0.0, sides[i] * sides[j]
            for turns in TURN_GRIDS:
                candidates = angle + turns
                c, s = np.cos(candidates)[:, None], np.sin(candidates)[:, None]
                areas = (spans(c * proj[i] + s * proj[j], trim) + eps) * (spans(c * proj[j] - s * proj[i], trim) + eps)
                if areas.min() < area:
                    angle, area = candidates[np.argmin(areas)], areas.min()
            if angle == 0.0:
                continue
            c, s = np.cos(angle), np.sin(angle)
            axes[:, [i, j]] = axes[:, [i, j]] @ np.array([[c, -s], [s, c]])
            proj[[i, j]] = [c * proj[i] + s * proj[j], c * proj[j] - s * proj[i]]
            sides = spans(proj, trim) + eps
        if sides.prod() > before * (1 - 1e-9):
            break
    return axes, sides.prod()


def fit_box(points, trim=0.0, max_points=BOX_SEARCH_POINTS, seed=0):
    """The smallest-volume box around points (n x 3), in any orientation. Returns (axes, sides): the box's unit axes as
    the columns of a 3x3 matrix, ordered and signed by match_world_axes, and its side along each, measured by spans
    with trim (so trim also keeps outliers from turning the box).

    The orientation is a local search (turn_to_fit) from two starts, the world axes and the principal axes of the
    points. The search uses at most max_points of the points, a fixed random choice; the two boxes it finds and the
    axis-aligned box are then measured on all of them, and the smallest is kept, so the box is never bigger than the
    axis-aligned one.
    """
    centred = np.asarray(points, float) - np.median(points, axis=0)
    search = centred
    if len(search) > max_points:
        search = search[np.random.default_rng(seed).choice(len(search), max_points, replace=False)]
    candidates = [np.eye(3)]
    scale = spans(search.T, trim).max()
    eps = 1e-6 * scale  # a flat object's box would otherwise have no volume whichever way it was turned
    if len(search) >= 4 and np.isfinite(scale) and scale > 0:
        candidates += [turn_to_fit(search, start, trim, eps)[0]
                       for start in (np.eye(3), np.linalg.eigh(np.cov(search.T))[1])]
    measured = [spans((centred @ axes).T, trim) for axes in candidates]
    best = min(range(len(candidates)), key=lambda c: np.prod(measured[c] + eps))  # ties: the axis-aligned box
    return match_world_axes(candidates[best], measured[best])


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
    *,
    depth_rule="mask",
    mask_conf_quantile=MASK_CONF_QUANTILE,
):
    """Per-frame 3D observations of every tracked object, keyed by (prompt index, track id). An object counts in a
    frame where its mask has at least min_points pixels of usable depth (usable_pixels under depth_rule; min_points
    None: the rule's default)."""
    min_points = min_points_for(depth_rule, min_points)
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
        c = conf[t].astype(np.float32)
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
                ys, xs = np.nonzero(usable_pixels(lab == tid, d, c, conf_thr, depth_rule, mask_conf_quantile))
                if len(ys) < min_points:
                    continue
                z = d[ys, xs]
                cam = np.stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z], axis=1)
                world = (
                    cam - tvec
                ) @ R  # x_cam = R x_world + t  ->  x_world = R^T (x_cam - t)
                obj = objects.setdefault((pi, int(tid)), {"frames": {}, "sample": []})
                obj["frames"][t] = np.median(cam, axis=0)
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
    min_points=None,
    size_trim=0.0,
    max_frame_points=5000,
    frame_columns=True,
    cameras=True,
    *,
    depth_rule="mask",
    mask_conf_quantile=MASK_CONF_QUANTILE,
    max_object_points=MAX_OBJECT_POINTS,
):
    """Write objects.csv and cameras.csv for a scene that already has tracks for every prompt.

    frame_columns=False leaves out num_frames and the per-frame columns, cameras=False leaves out cameras.csv.
    depth_rule, mask_conf_quantile, min_points (None: the rule's default): which pixels count (usable_pixels). An
    object's centre and size come from at most max_frame_points points per frame and max_object_points in all (a fixed
    random choice; at 16 frames the first limit keeps every object under the second).
    Returns the two CSV paths (None for cameras.csv when it is left out).
    """
    if relative_positions and not frame_columns:
        raise ValueError("relative_positions are per-frame columns, which frame_columns=False leaves out")
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
        depth_rule=depth_rule,
        mask_conf_quantile=mask_conf_quantile,
    )

    width = max(3, len(str(S - 1)))
    frame_names = [f"frame_{t:0{width}d}" for t in range(S)]
    header = ["object_id", "concept", "track_id", "x", "y", "z"]
    if include_size:
        header += ["size_x", "size_y", "size_z", "size"]
    if frame_columns:
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
        if len(sample) > max_object_points:
            sample = sample[np.random.default_rng([pi, tid]).choice(len(sample), max_object_points, replace=False)]
        row = [len(rows), prompts[pi], tid, *np.median(sample, axis=0)]
        if include_size:
            _, sides = fit_box(sample, size_trim)
            row += [*sides, float(np.prod(sides))]
        if frame_columns:
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
    counts = {
        p: sum(1 for (pi, _) in objects if pi == i) for i, p in enumerate(prompts)
    }
    print(f"[sheet] {len(rows)} objects over {S} frames: {counts}")
    print(f"[sheet] wrote {objects_csv}")
    if not cameras:
        (out_dir / "cameras.csv").unlink(missing_ok=True)  # an earlier sheet's, which would no longer match
        return objects_csv, None

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
    print(f"[sheet] wrote {cameras_csv}")
    return objects_csv, cameras_csv


# ---------------------------------------------------------------------------
# the sheet from gold labels
# ---------------------------------------------------------------------------
def ensure_gold_metadata(gold_dir=None):
    """The folder with VSI-Bench's three metadata files, downloading any that is missing (checked against GOLD_FILES)."""
    gold_dir = Path(gold_dir) if gold_dir else GOLD_DIR
    for name, digest in GOLD_FILES.values():
        path = gold_dir / name
        if path.is_file():
            continue
        url = f"https://raw.githubusercontent.com/{GOLD_REPO}/{GOLD_COMMIT}/data/meta_info/{name}"
        print(f"[sheet] downloading {url}")
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
        except OSError as error:
            raise RuntimeError(f"could not download {url} ({error}); put the file in {gold_dir} by hand") from error
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f"{url} does not have the expected SHA-256 {digest}")
        gold_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return gold_dir


def load_gold_scene(scene, gold_dir=None):
    """("<dataset>/<scene>", that scene's metadata) for "<dataset>/<scene>" or only a scene's name."""
    gold_dir = ensure_gold_metadata(gold_dir)
    dataset, _, name = scene.rpartition("/")
    if dataset and dataset not in GOLD_FILES:
        raise ValueError(f"unknown dataset {dataset!r}; VSI-Bench has {sorted(GOLD_FILES)}")
    for ds in [dataset] if dataset else list(GOLD_FILES):
        scenes = json.loads((gold_dir / GOLD_FILES[ds][0]).read_text())
        if name in scenes:
            return f"{ds}/{name}", scenes[name]
    raise ValueError(f"{scene!r} is not a VSI-Bench scene (not in the metadata in {gold_dir})")


def match_gold_class(info, prompt):
    """The metadata class a prompt names, or None: exact up to case and spaces, a plural "s", or GOLD_ALIASES."""
    classes = info["object_bbox"]
    key = " ".join(prompt.lower().split())
    for candidate in (key, GOLD_ALIASES.get(key), key[:-1] if key.endswith("s") else None):
        if candidate in classes:
            return candidate
    return None


def build_gold_scene_sheet(
    scene,
    prompts=None,
    include_size=False,
    out_dir=None,
    gold_dir=None,
    relative_positions=False,
    frame_columns=False,
    cameras=False,
):
    """Write objects.csv and sheet.json for a VSI-Bench scene from its gold labels, in metres. Returns (objects.csv,
    None). prompts: the classes to list (None: all). Per-frame columns and cameras don't exist for gold labels, so
    asking for them raises ValueError.
    """
    if relative_positions or frame_columns:
        raise ValueError("a gold sheet has no per-frame columns: the metadata says nothing about single frames")
    if cameras:
        raise ValueError("a gold sheet has no cameras.csv: the only camera poses are VGGT's, in a different frame "
                         "and scale from the gold labels, with nothing to align them by")
    scene, info = load_gold_scene(scene, gold_dir)
    out_dir = Path(out_dir) if out_dir else common.OUT_ROOT / scene.replace("/", "__") / "gold_sheet"
    out_dir.mkdir(parents=True, exist_ok=True)

    asked = list(prompts) if prompts is not None else list(info["object_bbox"])
    matched = {p: match_gold_class(info, p) for p in asked}
    header = ["object_id", "concept", "track_id", "x", "y", "z"]
    if include_size:
        header += ["size_x", "size_y", "size_z", "size"]
    rows, listed = [], set()
    for prompt in asked:
        if matched[prompt] is None or matched[prompt] in listed:  # a class two prompts name is listed once
            continue
        listed.add(matched[prompt])
        for instance, box in enumerate(info["object_bbox"][matched[prompt]], start=1):
            row = [len(rows), prompt, instance, *np.asarray(box["centroid"], float)]
            if include_size:  # normalizedAxes is row-major with one box axis per row; min/max is the axis-aligned box
                axes = np.asarray(box["normalizedAxes"], float).reshape(3, 3).T
                _, sides = match_world_axes(axes, box["axesLengths"])
                row += [*sides, float(np.prod(sides))]
            rows.append(row)
    objects_csv = out_dir / "objects.csv"
    write_csv(objects_csv, header, rows)
    (out_dir / "cameras.csv").unlink(missing_ok=True)  # an earlier pipeline sheet's, which would not match
    (out_dir / "sheet.json").write_text(json.dumps({
        "source": "gold",
        "scene": scene,
        "labels": f"github.com/{GOLD_REPO}@{GOLD_COMMIT}: data/meta_info/{GOLD_FILES[scene.split('/')[0]][0]}",
        "frame": "the dataset's world frame, z up, metres",
        "prompts": asked,
        "matched_classes": matched,
        "size": include_size,
    }, indent=1))
    found = {p: len(info["object_bbox"][c]) if c else 0 for p, c in matched.items()}
    print(f"[sheet] gold labels for {scene}: {len(rows)} objects {found}")
    if missing := [p for p, c in matched.items() if c is None]:
        print(f"[sheet] no gold class for {missing}; the scene's classes are {sorted(info['object_bbox'])}")
    print(f"[sheet] wrote {objects_csv}")
    return objects_csv, None


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--source",
        choices=["pipeline", "gold"],
        default="pipeline",
        help="pipeline: SAM 3 and tracking on the scene's geometry; gold: VSI-Bench's ground-truth boxes",
    )
    p.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        help='concepts, e.g. "table,dog,chair" or table dog chair (needed for --source pipeline; '
        "with --source gold, every class when left out)",
    )
    p.add_argument(
        "--scene",
        required=True,
        help="scene folder (the pipeline's --out; created if --images is given), or with --source gold a VSI-Bench "
        "scene, e.g. scannet/scene0025_01",
    )
    p.add_argument(
        "--no-frames",
        action="store_true",
        help="leave out num_frames and the per-frame columns (always so with --source gold)",
    )
    p.add_argument(
        "--no-cameras",
        action="store_true",
        help="leave out cameras.csv (always so with --source gold)",
    )
    gold = p.add_argument_group("--source gold")
    gold.add_argument(
        "--gold-dir",
        default=None,
        help=f"folder with VSI-Bench's metadata files (default {GOLD_DIR.relative_to(common.REPO)}, "
        "downloaded when missing)",
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
        "--size", action="store_true", help="add the sides and volume of each object's box, turned to fit it"
    )
    p.add_argument(
        "--size-trim",
        type=float,
        default=0.02,
        help="percent of points ignored at each end of every box side when measuring size "
        "(0 = farthest points)",
    )
    p.add_argument(
        "--min-points",
        type=int,
        default=None,
        help="pixels of usable depth needed for an object to count as seen in a frame "
        f"(default: the depth rule's, {MIN_POINTS})",
    )
    p.add_argument(
        "--depth-rule",
        choices=DEPTH_RULES,
        default="mask",
        help="which pixels of a mask have usable depth: mask (default), confidence at least the lower of "
        "--conf-percentile over the sequence and the mask's own --mask-conf-quantile; global (the original rule), at "
        "least --conf-percentile",
    )
    p.add_argument(
        "--mask-conf-quantile",
        type=float,
        default=MASK_CONF_QUANTILE,
        help="--depth-rule mask: percentile of the mask's own confidence its pixels need",
    )
    p.add_argument(
        "--association",
        choices=["overlap", "centroid"],
        default="overlap",
        help="when the pipeline has to track: how detections join tracks (milestone4_video.py --association); "
        "overlap (default), by their 3D points; centroid, the original rule",
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
        help="points per object per frame used for the global median and the size",
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
    p.add_argument(
        "--masks-dir",
        default=None,
        help="saved SAM 3 masks (sam3_infer.py) to track from, instead of running SAM 3",
    )
    args = p.parse_args(argv)

    prompts = parse_prompts(args.prompts) if args.prompts else None
    if args.source == "gold":
        if args.images:
            p.error("--images is for --source pipeline: a gold sheet needs no frames")
        try:
            build_gold_scene_sheet(
                args.scene,
                prompts,
                args.size,
                args.out_dir,
                args.gold_dir,
                relative_positions=args.relative_positions,
            )
        except (ValueError, RuntimeError) as error:
            p.error(str(error))
        return
    if not prompts:
        p.error("--prompts is needed for --source pipeline")
    if args.relative_positions and args.no_frames:
        p.error("--relative-positions are per-frame columns, which --no-frames leaves out")
    scene = Path(args.scene)
    ensure_scene(
        scene,
        prompts,
        Path(args.images) if args.images else None,
        args.step,
        args.chunk_size,
        args.overlap,
        args.no_reid,
        args.masks_dir,
        association=args.association,
        depth_rule=args.depth_rule,
        mask_conf_quantile=args.mask_conf_quantile,
        min_points=args.min_points,
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
        frame_columns=not args.no_frames,
        cameras=not args.no_cameras,
        depth_rule=args.depth_rule,
        mask_conf_quantile=args.mask_conf_quantile,
    )


if __name__ == "__main__":
    main()
