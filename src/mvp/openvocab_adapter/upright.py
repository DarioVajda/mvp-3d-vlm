"""An upright frame for a scene's pipeline sheets, from a fit of the floor plane (numpy only).

VGGT's world frame is the first camera's (OpenCV axes: x right, y down, z forward), and that camera is usually
pitched, often towards the floor, so its "down" is not the room's. The upright frame is the room's:

  z  up, perpendicular to the floor, which is at z = 0
  y  the way the first camera faces, along the floor
  x  to its right (x = y cross z)
  origin  the point of the floor below the first camera

in the same units as the reconstruction (VGGT's per-scene scale). How it is found, from the scene's cameras and depth:

  1. A first up from the cameras: people hold the camera level (little roll, any pitch), so each camera's right axis
     is nearly horizontal and up is the direction most perpendicular to all of them
     (camera_sheet_format_converter.estimate_up).
  2. Points: every STRIDE-th pixel of every frame whose depth confidence is in the top (100 - CONF_PERCENTILE)%,
     unprojected, at most MAX_POINTS of them. The scene's length unit is its median depth (about 1.5 m).
  3. Two fits of the floor, each refined by least squares (a plane through the points within tau = TAU_FRAC units of
     it, three times over):
       seeded  along the first up, the lowest strong peak of the low points' heights (the lowest LOW_QUANTILE; a peak
               at least PEAK_FRACTION of the highest), then planes through the points near it, from a band of
               SEED_BAND tau (a floor the first up sees slightly tilted spreads over a few centimetres) down to tau
       ransac  RANSAC_ITERS planes through three random low points, with normals within MAX_TILT_DEG of the first up
               and at most BELOW_MAX of the points more than BELOW_TAU tau below them (nothing is under the floor:
               this rejects table and bed tops), scored by the points within tau
  4. Checks: at least MIN_INLIER_SHARE of the points (and MIN_INLIERS) on the plane, at most BELOW_MAX more than
     BELOW_TAU tau below it, its normal within ACCEPT_TILT_DEG of the first up, at least MIN_CAMERAS_ABOVE of the cameras above it at a median height within
     CAMERA_HEIGHT_RANGE units. The better-supported fit that passes is the floor. If neither passes, the floor is not
     found: up is the cameras' up, z = 0 is the FALLBACK_FLOOR_PERCENTILE-th percentile of the points' heights, and a
     WARNING says so.
  5. The frame: z the floor's normal; y the first camera's forward along the floor (its image's up when it looks
     nearly straight down or up, FORWARD_MIN); x = y cross z; the origin below the first camera.

On the 75 exported scenes (16 and 128 frames), the floor's normal is a median 1-2 degrees from the cameras' up, the two
frame selections of one reconstruction agree to a median 0.3 degrees, and the cameras are a median 1.4 m above the
floor (with the gold scale, where there is a reliable one).

scene_upright(scene_dir) does all of it for a scene folder (cameras.npz, depth.npz) and keeps the result in
scene_dir/upright.json, so that every sheet of the scene gets the same frame; it is computed again if the cameras,
the settings or this format change. to_upright(points, frame) moves points into the frame.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera_sheet_format_converter as camera_text  # noqa: E402  (estimate_up)

FORMAT = "openvocab4d-upright/v1"
FILE = "upright.json"
CONF_PERCENTILE = 50.0
STRIDE = 4
MAX_POINTS = 200_000
LOW_QUANTILE = 0.35
RANSAC_ITERS = 512
HYPOTHESIS_POINTS = 20_000
SCORE_POINTS = 40_000
MAX_TILT_DEG = 15.0
ACCEPT_TILT_DEG = 8.0
TAU_FRAC = 0.02
BELOW_MAX = 0.05
BELOW_TAU = 5.0
PEAK_FRACTION = 0.5
SEED_BAND = 5.0
MIN_INLIER_SHARE = 0.01
MIN_INLIERS = 500
MIN_CAMERAS_ABOVE = 0.9
CAMERA_HEIGHT_RANGE = (0.2, 3.0)
FALLBACK_FLOOR_PERCENTILE = 2.0
FORWARD_MIN = 0.3
SEED = 0
SETTINGS = {name: globals()[name] for name in (
    "CONF_PERCENTILE", "STRIDE", "MAX_POINTS", "LOW_QUANTILE", "RANSAC_ITERS", "HYPOTHESIS_POINTS", "SCORE_POINTS",
    "MAX_TILT_DEG", "ACCEPT_TILT_DEG", "TAU_FRAC", "BELOW_MAX", "BELOW_TAU", "PEAK_FRACTION", "SEED_BAND",
    "MIN_INLIER_SHARE", "MIN_INLIERS", "MIN_CAMERAS_ABOVE",
    "CAMERA_HEIGHT_RANGE", "FALLBACK_FLOOR_PERCENTILE", "FORWARD_MIN", "SEED")}


def camera_poses(extrinsic):
    """Camera centres (S, 3) and camera-to-world rotations (S, 3, 3) from OpenCV world-to-camera extrinsics."""
    R, t = extrinsic[:, :, :3].astype(np.float64), extrinsic[:, :, 3].astype(np.float64)
    return np.einsum("sji,sj->si", R, -t), np.transpose(R, (0, 2, 1))


def scene_unit(depth):
    """The scene's length unit: its median depth over every frame (every 4th pixel), as milestone4_video.scene_unit."""
    d = np.asarray(depth[:, ::4, ::4], np.float32)
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 1.0


def scene_points(depth, conf, extrinsic, intrinsic, conf_percentile=CONF_PERCENTILE, stride=STRIDE,
                 max_points=MAX_POINTS, seed=SEED):
    """World points of every stride-th pixel with depth > 0 and confidence at least the conf_percentile-th percentile
    (over those pixels of every frame), at most max_points of them (a fixed random choice)."""
    d_all = np.asarray(depth[:, ::stride, ::stride], np.float32)
    c_all = np.asarray(conf[:, ::stride, ::stride], np.float32)
    threshold = np.percentile(c_all, conf_percentile)
    H, W = depth.shape[1:]
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    points = []
    for t in range(len(d_all)):
        z = d_all[t]
        ok = (c_all[t] >= threshold) & (z > 0)
        K = intrinsic[t].astype(np.float64)
        cam = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], -1)[ok]
        R, tvec = extrinsic[t, :, :3].astype(np.float64), extrinsic[t, :, 3].astype(np.float64)
        points.append((cam - tvec) @ R)  # x_cam = R x_world + t
    points = np.concatenate(points) if points else np.zeros((0, 3))
    if len(points) > max_points:
        points = points[np.random.default_rng(seed).choice(len(points), max_points, replace=False)]
    return points


def angle_deg(a, b):
    """The angle between two directions, in degrees, ignoring their signs."""
    return float(np.degrees(np.arccos(np.clip(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


def refine_plane(points, n, offset, up0, tau, rounds=3):
    """A least-squares plane through the points within tau of {n . p = offset}, rounds times over (n kept pointing
    along up0)."""
    for _ in range(rounds):
        inliers = points[np.abs(points @ n - offset) < tau]
        if len(inliers) < 3:
            break
        mean = inliers.mean(axis=0)
        n = np.linalg.svd(inliers - mean, full_matrices=False)[2][2]
        n = n if n @ up0 >= 0 else -n
        offset = float(n @ mean)
    return n, offset


def seeded_plane(points, up0, tau):
    """The floor as seen along the cameras' up: the lowest strong peak of the low points' heights (at least
    PEAK_FRACTION of the highest peak among them), then planes through the points near it, from a wide band (SEED_BAND
    tau, which takes in a floor the cameras' up sees a little tilted) down to tau."""
    heights = points @ up0
    low = heights[heights <= np.quantile(heights, LOW_QUANTILE)]
    edges = np.arange(low.min(), low.max() + 2 * tau, tau)
    counts = np.convolve(np.histogram(low, edges)[0], np.ones(5), "same")  # points within about 2.5 tau
    peak = int(np.nonzero(counts >= PEAK_FRACTION * counts.max())[0][0])
    n, offset = up0, float((edges[peak] + edges[peak + 1]) / 2)
    for band in (SEED_BAND * tau, 2 * tau):
        n, offset = refine_plane(points, n, offset, up0, band)
    return refine_plane(points, n, offset, up0, tau)


def ransac_plane(points, up0, tau, rng):
    """The best of RANSAC_ITERS planes through three random low points (the lowest LOW_QUANTILE along the cameras'
    up), with normals within MAX_TILT_DEG of it and at most BELOW_MAX of the points more than BELOW_TAU tau below
    them: by the points within tau. None if no plane qualifies."""
    heights = points @ up0
    low = points[heights <= np.quantile(heights, LOW_QUANTILE)]
    hypotheses = low[rng.choice(len(low), min(HYPOTHESIS_POINTS, len(low)), replace=False)]
    scoring = points[rng.choice(len(points), min(SCORE_POINTS, len(points)), replace=False)]
    triples = rng.integers(0, len(hypotheses), (RANSAC_ITERS, 3))
    a, b, c = (hypotheses[triples[:, i]] for i in range(3))
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, None]
    normals[normals @ up0 < 0] *= -1
    valid &= normals @ up0 >= np.cos(np.radians(MAX_TILT_DEG))
    best, best_score = None, 0
    for i in np.nonzero(valid)[0]:
        offset = normals[i] @ a[i]
        if np.mean(scoring @ normals[i] - offset < -BELOW_TAU * tau) > BELOW_MAX:
            continue  # too much below it to be the floor
        score = int(np.sum(np.abs(hypotheses @ normals[i] - offset) < tau))
        if score > best_score:
            best, best_score = (normals[i], offset), score
    return None if best is None else refine_plane(points, *best, up0, tau)


def assess_plane(points, n, offset, up0, unit, centres, tau):
    """(diagnostics, problems) of a floor candidate: the step 4 checks of the module docstring."""
    heights = points @ n - offset
    inliers = int(np.sum(np.abs(heights) < tau))
    camera_heights = centres @ n - offset
    diagnostics = {"inlier_share": inliers / len(points), "inliers": inliers, "tilt_from_cameras_deg": angle_deg(n, up0),
                   "below": float(np.mean(heights < -BELOW_TAU * tau)),
                   "cameras_above": float(np.mean(camera_heights > 0)),
                   "camera_height_median": float(np.median(camera_heights) / unit)}
    problems = []
    if inliers < MIN_INLIERS or inliers / len(points) < MIN_INLIER_SHARE:
        problems.append(f"only {inliers / len(points):.1%} of the points on it")
    if diagnostics["below"] > BELOW_MAX:  # a table or bed top, with the floor (or walls) under it
        problems.append(f"{diagnostics['below']:.0%} of the points below it")
    if diagnostics["tilt_from_cameras_deg"] > ACCEPT_TILT_DEG:
        problems.append(f"{diagnostics['tilt_from_cameras_deg']:.0f} degrees from the cameras' up")
    if diagnostics["cameras_above"] < MIN_CAMERAS_ABOVE:
        problems.append(f"only {diagnostics['cameras_above']:.0%} of the cameras above it")
    low_h, high_h = CAMERA_HEIGHT_RANGE
    if not low_h <= diagnostics["camera_height_median"] <= high_h:
        problems.append(f"cameras {diagnostics['camera_height_median']:.2f} scene units above it")
    return diagnostics, problems


def fit_floor(points, up0, unit, centres, seed=SEED):
    """The floor plane {p: n . p = d} (n pointing up) from two fits (module docstring, steps 3-4), seeded_plane (from
    the cameras' up) and ransac_plane: the one with more points on it that passes the checks (assess_plane). Returns
    (n, d, diagnostics), or (None, None, diagnostics) with diagnostics["reason"] when neither passes."""
    up0 = up0 / np.linalg.norm(up0)
    tau = TAU_FRAC * unit
    diagnostics = {"points": int(len(points)), "unit": unit, "tau": tau}
    if len(points) < MIN_INLIERS:
        return None, None, {**diagnostics, "reason": f"only {len(points)} points with confident depth"}
    candidates = {"seeded": seeded_plane(points, up0, tau),
                  "ransac": ransac_plane(points, up0, tau, np.random.default_rng(seed))}
    assessed = sorted(((name, plane, *assess_plane(points, *plane, up0, unit, centres, tau))
                       for name, plane in candidates.items() if plane is not None),
                      key=lambda c: -c[2]["inliers"])
    if not assessed:
        return None, None, {**diagnostics, "reason": f"no plane within {MAX_TILT_DEG:g} degrees of the cameras' up"}
    for name, (n, offset), found, problems in assessed:
        if not problems:
            return n, offset, {**diagnostics, **found, "fit": name}
    name, _, found, problems = assessed[0]
    return None, None, {**diagnostics, **found, "fit": name, "reason": "the best plane has " + ", ".join(problems)}


def upright_frame(extrinsic, n, offset):
    """{"rotation": R (rows x, y, z of the frame, in world coordinates), "origin": o} for the floor {n . p = offset}:
    z = n, y the first camera's forward along the floor (its image-up when it looks nearly straight along n), x = y x z,
    o the point of the floor below the first camera. p' = R (p - o)."""
    centres, rotations = camera_poses(extrinsic)
    z = n / np.linalg.norm(n)
    along = []
    for direction in (rotations[0][:, 2], -rotations[0][:, 1]):  # forward, then the image's up
        flat = direction - (direction @ z) * z
        along.append((np.linalg.norm(flat), flat))
    use = 0 if along[0][0] >= FORWARD_MIN else int(np.argmax([a[0] for a in along]))
    y = along[use][1] / along[use][0]
    x = np.cross(y, z)
    origin = centres[0] - (centres[0] @ z - offset) * z
    return {"rotation": np.stack([x, y, z]), "origin": origin, "forward": ("camera", "image up")[use]}


def to_upright(points, frame):
    """Points (N, 3) in VGGT's world frame, in the upright frame."""
    return (np.asarray(points, np.float64) - frame["origin"]) @ frame["rotation"].T


def compute_upright(extrinsic, intrinsic, depth, conf, where="the scene"):
    """The upright frame of one scene's geometry: {"rotation", "origin", "method" (floor or cameras), "normal",
    "offset", "diagnostics", "warning"}."""
    centres, rotations = camera_poses(extrinsic)
    up0 = camera_text.estimate_up(rotations)
    unit = scene_unit(depth)
    points = scene_points(depth, conf, extrinsic, intrinsic)
    n, offset, diagnostics = fit_floor(points, up0, unit, centres)
    warning = None
    if n is None:
        n = up0
        offset = float(np.percentile(points @ n, FALLBACK_FLOOR_PERCENTILE)) if len(points) else float(
            np.min(centres @ n) - unit)
        warning = (f"WARNING: no floor plane found in {where} ({diagnostics['reason']}); z is the cameras' up and z = 0 "
                   f"the {FALLBACK_FLOOR_PERCENTILE:g}th percentile of the points' heights")
    frame = upright_frame(extrinsic, n, offset)
    return {**frame, "method": "floor" if warning is None else "cameras", "normal": n, "offset": offset,
            "cameras_up": up0, "diagnostics": diagnostics, "warning": warning}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def scene_upright(scene_dir, log=None):
    """The upright frame of a scene folder (cameras.npz, depth.npz), from scene_dir/upright.json when it was made from
    the same cameras with the same settings, else computed and saved there. log(text) gets the WARNING when no floor
    plane was found (default: stderr), each time the frame is used, so that every sheet's log says so."""
    scene_dir = Path(scene_dir)
    digest = hashlib.sha256((scene_dir / "cameras.npz").read_bytes()).hexdigest()
    path = scene_dir / FILE
    record = None
    if path.is_file():
        record = json.loads(path.read_text())
        if (record.get("format"), record.get("settings"), record.get("cameras_sha256")) != (
                FORMAT, _jsonable(SETTINGS), digest):
            record = None
    if record is None:
        with np.load(scene_dir / "cameras.npz") as cams, np.load(scene_dir / "depth.npz") as dep:
            frame = compute_upright(cams["extrinsic"], cams["intrinsic"], dep["depth"], dep["conf"],
                                    where=str(scene_dir))
        record = _jsonable({"format": FORMAT, "settings": SETTINGS, "cameras_sha256": digest, **frame})
        path.write_text(json.dumps(record, indent=1))
    frame = {**record, "rotation": np.array(record["rotation"]), "origin": np.array(record["origin"])}
    if frame["warning"]:
        (log or (lambda text: print(f"[sheet] {text}", file=sys.stderr)))(frame["warning"])
    return frame


def floor_summary(frame):
    """What question.json records about the frame: how up was found, and how much of the scene is on the floor."""
    return {"method": frame["method"], "inlier_share": frame["diagnostics"].get("inlier_share"),
            "tilt_from_cameras_deg": frame["diagnostics"].get("tilt_from_cameras_deg"),
            "warning": frame["warning"]}
