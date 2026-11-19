"""Compare OpenVocab-4D run on the saved VGGT outputs (adapted) with OpenVocab-4D's own VGGT, on the same frames and masks.

  py src/mvp/openvocab_adapter/compare.py --run-dir tasks/openvocab_adapter/out/scannet__scene0025_01/smoke5 \
      --masks tasks/openvocab_adapter/out/scannet__scene0025_01/masks

--run-dir holds adapted/ and native/ (each one is milestone 4's --out). Every comparison ends in one state:

  PASS / FAIL   a requirement that has a right answer (file formats, same frames, identical masks)
  MEASURED      a number with no pass mark: how far apart the two reconstructions are
  BLOCKED       could not run, with the missing input named

The saved poses come from one 128-frame reconstruction and the native ones from a reconstruction of only the selected
frames, so the two have different world frames and scales. Before positions are compared, a Sim(3) is fitted on
pixels that are confident in both depth maps. It uses no object information, so the object errors are not circular.

If both runs have a scene sheet (scene_sheet/objects.csv, from scene_sheet_creator.py), the sheets are compared too,
row by row through the same track matching.

The report ends with a verdict against the provisional pass marks in CRITERIA: MATCHES, DIFFERS, or INCOMPLETE when
a needed number could not be measured.

Exit code 1 means a FAIL. BLOCKED alone exits 0. The verdict does not change the exit code.
"""
import argparse
import csv
import json
import sys
from pathlib import Path, PureWindowsPath

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

sys.path.insert(0, str(common.TRACK_A))
from eval_colmap import umeyama_sim3  # noqa: E402  (import-safe: its arguments are parsed inside main)

SEED = 0
NEEDS = {
    "adapted_depth": "dense depth: get depth.npy and depth_conf.npy from data/vggt/<scene>/ and rerun adapt_vggt.py --dense-dir",
    "masks": "SAM 3 masks: run sam3_infer.py on a GPU machine (README.md)",
    "adapted_outputs": "adapted tracking and fusion: run milestone 4 --stages track,fuse --masks-dir on adapted/",
    "native_geometry": "native VGGT reference: run milestone 4 --stages vggt on a GPU machine",
    "native_outputs": "native tracking and fusion: run milestone 4 --stages track,fuse --masks-dir on native/",
    "scene_sheets": "scene sheets: run scene_sheet_creator.py --scene <run> on adapted/ and native/",
}
SHEET = Path("scene_sheet") / "objects.csv"

# Provisional pass marks for "OpenVocab-4D on the saved VGGT outputs gives the same result as on its own VGGT". The two
# reconstructions come from different VGGT passes, so they can never agree exactly. Nobody has agreed on these
# numbers yet: they are here so that a run ends in a clear answer, and should be revisited once real results are in.
# (comparison, value, direction, limit)
CRITERIA = {
    "tracks: share of pixels with the same identity": ("tracking consistency", "pixels_with_matching_identity", ">=", 0.90),
    "objects: median 3D position error, share of scene size": (
        "3D object positions after alignment", "median_error_of_scene", "<=", 0.05),
    "scene sheet: share of rows found in both sheets": ("scene sheet: objects", "matched_share", ">=", 0.80),
    "scene sheet: median position error, share of scene size": (
        "scene sheet: objects", "median_position_error_of_scene", "<=", 0.05),
}

ARRAY_SPECS = {
    "cameras.npz": {"extrinsic": ("f4", ("S", 3, 4)), "intrinsic": ("f4", ("S", 3, 3)), "image_paths": ("U", ("S",))},
    "depth.npz": {"depth": ("f2", ("S", "H", "W")), "conf": ("f2", ("S", "H", "W"))},
    "labels.npz": {"xyz": ("f4", ("N", 3)), "concept_id": ("i2", ("N",)), "object_id": ("i4", ("N",)),
                   "concepts": ("U", ("C",))},
}
TRACK_SPEC = {"labels": ("u2", ("S", "H", "W"))}


class Report:
    def __init__(self):
        self.rows = []

    def add(self, name, status, detail="", values=None, needs=()):
        self.rows.append({"comparison": name, "status": status, "detail": detail, "values": values or {},
                          "needs": [NEEDS[n] for n in needs]})

    def counts(self):
        return {s: sum(r["status"] == s for r in self.rows) for s in ("PASS", "FAIL", "MEASURED", "BLOCKED")}

    def has_fail(self):
        return any(r["status"] == "FAIL" for r in self.rows)


def load_npz(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def array_problems(path, spec, dims):
    """Differences between an .npz and its spec; dims binds S, H, W, N across the files of one run."""
    arrays = load_npz(path)
    problems = []
    if set(arrays) != set(spec):
        problems.append(f"{path.name}: keys {sorted(arrays)}, expected {sorted(spec)}")
    for key, (dtype, shape) in spec.items():
        if key not in arrays:
            continue
        array = arrays[key]
        if (array.dtype.kind != "U") if dtype == "U" else (array.dtype != np.dtype(dtype)):
            problems.append(f"{path.name}:{key} dtype {array.dtype}, expected {dtype}")
        if len(array.shape) != len(shape):
            problems.append(f"{path.name}:{key} shape {array.shape}, expected {shape}")
            continue
        for actual, wanted in zip(array.shape, shape):
            if isinstance(wanted, str):
                if dims.setdefault(wanted, actual) != actual:
                    problems.append(f"{path.name}:{key} has {wanted}={actual}, elsewhere {dims[wanted]}")
            elif actual != wanted:
                problems.append(f"{path.name}:{key} shape {array.shape}, expected {shape}")
    return problems


def ply_problems(path, expected_points):
    data = path.read_bytes()
    end = data.find(b"end_header\n")
    if not data.startswith(b"ply\n") or end < 0:
        return [f"{path.name}: not a PLY file"]
    header = data[:end].decode("ascii").splitlines()
    vertices = [int(line.split()[2]) for line in header if line.startswith("element vertex")]
    properties = [line.split()[-1] for line in header if line.startswith("property")]
    problems = []
    if properties != ["x", "y", "z", "red", "green", "blue"]:
        problems.append(f"{path.name}: properties {properties}")
    if vertices != [expected_points]:
        problems.append(f"{path.name}: {vertices} vertices, labels.npz has {expected_points}")
    elif len(data) - end - len(b"end_header\n") != 15 * expected_points:
        problems.append(f"{path.name}: payload size does not match its vertex count")
    return problems


def check_run_format(report, name, run, want_outputs):
    """Check whichever of a run's files exist against the pipeline's formats."""
    dims = {}
    for filename in ("cameras.npz", "depth.npz") + (("labels.npz",) if want_outputs else ()):
        path = run / filename
        if not path.is_file():
            continue
        problems = array_problems(path, ARRAY_SPECS[filename], dims)
        report.add(f"format: {name}/{filename}", "FAIL" if problems else "PASS", "; ".join(problems))
    if not want_outputs:
        return
    for track in sorted((run / "tracks").glob("*.npz")):
        problems = array_problems(track, TRACK_SPEC, dims)
        report.add(f"format: {name}/tracks/{track.name}", "FAIL" if problems else "PASS", "; ".join(problems))
    if (run / "objects.ply").is_file() and "N" in dims:
        problems = ply_problems(run / "objects.ply", dims["N"])
        report.add(f"format: {name}/objects.ply", "FAIL" if problems else "PASS", "; ".join(problems))
    missing = [f for f in ("objects.rrd", "objects.ply", "report.json", "tracks/summary.json") if not (run / f).exists()]
    if (run / "labels.npz").is_file():
        report.add(f"format: {name} has all output files", "FAIL" if missing else "PASS", f"missing {missing}" if missing else "")


def frame_names(run):
    # PureWindowsPath splits on both / and \, so runs made on different systems can be compared
    return [PureWindowsPath(str(p)).name for p in load_npz(run / "cameras.npz")["image_paths"]]


def check_masks(report, adapted, masks, has):
    if not has["masks"]:
        report.add("masks: cover the selected frames", "BLOCKED", needs=["masks"])
        return
    manifest = json.loads((masks / "manifest.json").read_text())
    shas = {f["stem"]: f["image_sha256"] for f in manifest["frames"]}
    if not (adapted / "cameras.npz").is_file():
        report.add("masks: cover the selected frames", "BLOCKED", "adapted/cameras.npz is missing")
        return
    frames = [Path(str(p)) for p in load_npz(adapted / "cameras.npz")["image_paths"]]
    absent = [p.stem for p in frames if p.stem not in shas]
    different = [p.stem for p in frames if p.stem in shas and p.is_file() and common.sha256_file(p) != shas[p.stem]]
    report.add("masks: cover the selected frames", "FAIL" if absent or different else "PASS",
               f"missing {absent}, different images {different}" if absent or different else f"{len(frames)} frames")


def fit_alignment(adapted, native):
    """Sim(3) taking adapted world coordinates to native ones, from pixels confident in both reconstructions."""
    cams_a, cams_n = load_npz(adapted / "cameras.npz"), load_npz(native / "cameras.npz")
    dep_a, dep_n = load_npz(adapted / "depth.npz"), load_npz(native / "depth.npz")
    rng = np.random.default_rng(SEED)
    source, target = [], []
    for t in range(len(dep_a["depth"])):
        da, db = dep_a["depth"][t].astype(np.float32), dep_n["depth"][t].astype(np.float32)
        ca, cb = dep_a["conf"][t].astype(np.float32), dep_n["conf"][t].astype(np.float32)
        ys, xs = np.nonzero((ca >= np.median(ca)) & (cb >= np.median(cb)) & (da > 0) & (db > 0))
        if len(ys) == 0:
            continue
        chosen = rng.choice(len(ys), min(5000, len(ys)), replace=False)
        ys, xs = ys[chosen], xs[chosen]
        source.append(common.unproject_frame(da, cams_a["extrinsic"][t], cams_a["intrinsic"][t])[ys, xs])
        target.append(common.unproject_frame(db, cams_n["extrinsic"][t], cams_n["intrinsic"][t])[ys, xs])
    source, target = np.concatenate(source).astype(np.float64), np.concatenate(target).astype(np.float64)
    scale, rotation, translation = umeyama_sim3(source, target)
    diag = float(np.linalg.norm(np.percentile(target, 99, axis=0) - np.percentile(target, 1, axis=0)))
    residual = float(np.sqrt(((apply_sim3(source, (scale, rotation, translation)) - target) ** 2).sum(1).mean()))
    return (scale, rotation, translation), diag, residual, len(source)


def apply_sim3(points, sim3):
    scale, rotation, translation = sim3
    return scale * points @ rotation.T + translation


def compare_cameras(report, adapted, native, sim3, diag):
    cams_a, cams_n = load_npz(adapted / "cameras.npz"), load_npz(native / "cameras.npz")
    centres = {}
    for name, cams in (("a", cams_a), ("n", cams_n)):
        R, t = cams["extrinsic"][:, :, :3].astype(np.float64), cams["extrinsic"][:, :, 3].astype(np.float64)
        centres[name] = -np.einsum("tji,tj->ti", R, t)
    position = np.linalg.norm(apply_sim3(centres["a"], sim3) - centres["n"], axis=1) / diag
    angles = []
    for Ra, Rn in zip(cams_a["extrinsic"][:, :, :3].astype(np.float64), cams_n["extrinsic"][:, :, :3].astype(np.float64)):
        relative = Rn @ (Ra @ sim3[1].T).T
        angles.append(np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))
    report.add("cameras after alignment", "MEASURED", values={
        "position_error_median_of_scene": round(float(np.median(position)), 5),
        "position_error_max_of_scene": round(float(position.max()), 5),
        "rotation_error_median_deg": round(float(np.median(angles)), 3),
        "rotation_error_max_deg": round(float(np.max(angles)), 3)})


def match_tracks(labels_a, labels_n):
    """Greedy one-to-one matching of two label volumes by pixel IoU over all frames: [(id_a, id_n, iou)]."""
    both = (labels_a > 0) & (labels_n > 0)
    pairs, overlaps = np.unique(labels_a[both].astype(np.int64) * 65536 + labels_n[both], return_counts=True)
    ids_a, areas_a = np.unique(labels_a[labels_a > 0], return_counts=True)
    ids_n, areas_n = np.unique(labels_n[labels_n > 0], return_counts=True)
    area_a, area_n = dict(zip(ids_a.tolist(), areas_a.tolist())), dict(zip(ids_n.tolist(), areas_n.tolist()))
    candidates = []
    for pair, overlap in zip(pairs.tolist(), overlaps.tolist()):
        ia, ib = pair // 65536, pair % 65536
        candidates.append((overlap / (area_a[ia] + area_n[ib] - overlap), ia, ib))
    used_a, used_n, matches = set(), set(), []
    for iou, ia, ib in sorted(candidates, reverse=True):
        if ia not in used_a and ib not in used_n:
            used_a.add(ia)
            used_n.add(ib)
            matches.append((ia, ib, iou))
    return matches, area_a, area_n


def compare_tracking(report, adapted, native, prompts):
    """Coordinate-free: the same masks, so tracks can be matched on pixels. Returns matches per prompt."""
    per_prompt, all_matches = {}, {}
    matched_pixels = union_pixels = matched = unmatched_a = unmatched_n = 0
    ious = []
    for prompt in prompts:
        name = f"{prompt.replace(' ', '_')}.npz"
        labels_a, labels_n = load_npz(adapted / "tracks" / name)["labels"], load_npz(native / "tracks" / name)["labels"]
        if labels_a.shape != labels_n.shape:
            report.add(f"tracking: {prompt}", "FAIL", f"volumes {labels_a.shape} and {labels_n.shape}")
            continue
        matches, area_a, area_n = match_tracks(labels_a, labels_n)
        all_matches[prompt] = matches
        matched += len(matches)
        unmatched_a += len(area_a) - len(matches)
        unmatched_n += len(area_n) - len(matches)
        ious += [m[2] for m in matches]
        for ia, ib, _ in matches:
            matched_pixels += int(((labels_a == ia) & (labels_n == ib)).sum())
        union_pixels += int(((labels_a > 0) | (labels_n > 0)).sum())
        per_prompt[prompt] = {"adapted_tracks": len(area_a), "native_tracks": len(area_n), "matched": len(matches),
                              "mean_iou": round(float(np.mean([m[2] for m in matches])), 4) if matches else None}
    report.add("tracking consistency", "MEASURED", values={
        "matched_tracks": matched, "adapted_only_tracks": unmatched_a, "native_only_tracks": unmatched_n,
        "mean_matched_iou": round(float(np.mean(ious)), 4) if ious else None,
        "pixels_with_matching_identity": round(matched_pixels / union_pixels, 4) if union_pixels else None,
        "per_prompt": per_prompt})
    return all_matches


def object_points(labels, objects):
    """(concept, track_id) -> that object's points, using report.json's objects in labels.npz's object_id order."""
    return {(o["concept"], o["track_id"]): labels["xyz"][labels["object_id"] == j].astype(np.float64)
            for j, o in enumerate(objects)}


def nearest_distances(points, others):
    out = []
    for start in range(0, len(points), 512):
        delta = points[start:start + 512][:, None] - others[None]
        out.append(np.sqrt((delta ** 2).sum(-1).min(axis=1)))
    return np.concatenate(out)


def chamfer(points_a, points_b, limit, rng):
    """Symmetric mean nearest-neighbour distance between two clouds, on at most `limit` points of each."""
    if len(points_a) > limit:
        points_a = points_a[rng.choice(len(points_a), limit, replace=False)]
    if len(points_b) > limit:
        points_b = points_b[rng.choice(len(points_b), limit, replace=False)]
    return float((nearest_distances(points_a, points_b).mean() + nearest_distances(points_b, points_a).mean()) / 2)


def compare_objects(report, adapted, native, matches, sim3, diag):
    labels_a, labels_n = load_npz(adapted / "labels.npz"), load_npz(native / "labels.npz")
    report_a = json.loads((adapted / "report.json").read_text())
    report_n = json.loads((native / "report.json").read_text())
    objects_a = object_points(labels_a, report_a["objects"])
    objects_n = object_points(labels_n, report_n["objects"])
    seen_a = {(o["concept"], o["track_id"]): o["frames_seen"] for o in report_a["objects"]}
    seen_n = {(o["concept"], o["track_id"]): o["frames_seen"] for o in report_n["objects"]}

    rng = np.random.default_rng(SEED)
    position_errors, object_chamfers, seen_differences, rows = [], [], [], []
    for prompt, pairs in matches.items():
        for ia, ib, iou in pairs:
            key_a, key_n = (prompt, ia), (prompt, ib)
            if key_a not in objects_a or key_n not in objects_n:
                continue  # a track whose points all fell under fusion's limits
            aligned = apply_sim3(objects_a[key_a], sim3)
            error = float(np.linalg.norm(np.median(aligned, axis=0) - np.median(objects_n[key_n], axis=0))) / diag
            distance = chamfer(aligned, objects_n[key_n], 2000, rng) / diag
            position_errors.append(error)
            object_chamfers.append(distance)
            seen_differences.append(abs(seen_a[key_a] - seen_n[key_n]))
            rows.append({"concept": prompt, "adapted_track": ia, "native_track": ib, "iou": round(iou, 4),
                         "position_error_of_scene": round(error, 5), "chamfer_of_scene": round(distance, 5)})
    if position_errors:
        report.add("3D object positions after alignment", "MEASURED", values={
            "objects": len(position_errors), "median_error_of_scene": round(float(np.median(position_errors)), 5),
            "max_error_of_scene": round(float(np.max(position_errors)), 5),
            "mean_frames_seen_difference": round(float(np.mean(seen_differences)), 3), "per_object": rows})
        report.add("object point clouds after alignment", "MEASURED", values={
            "median_object_chamfer_of_scene": round(float(np.median(object_chamfers)), 5),
            "all_objects_chamfer_of_scene": round(
                chamfer(apply_sim3(labels_a["xyz"].astype(np.float64), sim3), labels_n["xyz"].astype(np.float64),
                        5000, rng) / diag, 5),
            "points": {"adapted": int(len(labels_a["xyz"])), "native": int(len(labels_n["xyz"]))}})
    else:
        report.add("3D object positions after alignment", "MEASURED", "no matched objects have fused points")


def read_sheet(run):
    """(concept, track_id) -> objects.csv row"""
    with open(run / SHEET, newline="") as handle:
        return {(row["concept"], int(row["track_id"])): row for row in csv.DictReader(handle)}


def compare_sheets(report, adapted, native, matches, sim3, diag):
    """The scene sheets row by row, pairing rows through the track matching (the sheets share the tracks' ids)."""
    sheet_a, sheet_n = read_sheet(adapted), read_sheet(native)
    paired_a, paired_n, rows, errors, frame_differences = set(), set(), [], [], []
    for prompt, pairs in matches.items():
        for ia, ib, iou in pairs:
            row_a, row_n = sheet_a.get((prompt, ia)), sheet_n.get((prompt, ib))
            if row_a is None or row_n is None:
                continue  # a track the sheet left out because no frame had enough confident pixels
            paired_a.add((prompt, ia))
            paired_n.add((prompt, ib))
            position_a = apply_sim3(np.array([[float(row_a[k]) for k in "xyz"]]), sim3)[0]
            position_n = np.array([float(row_n[k]) for k in "xyz"])
            error = float(np.linalg.norm(position_a - position_n)) / diag
            errors.append(error)
            frame_differences.append(abs(int(row_a["num_frames"]) - int(row_n["num_frames"])))
            rows.append({"concept": prompt, "adapted_track": ia, "native_track": ib, "iou": round(iou, 4),
                         "position_error_of_scene": round(error, 5),
                         "num_frames": [int(row_a["num_frames"]), int(row_n["num_frames"])]})
    total = len(sheet_a) + len(sheet_n)
    values = {
        "adapted_rows": len(sheet_a), "native_rows": len(sheet_n), "paired_rows": len(rows),
        "matched_share": round(2 * len(rows) / total, 4) if total else None,
        "adapted_only": sorted(f"{c}#{t}" for c, t in set(sheet_a) - paired_a),
        "native_only": sorted(f"{c}#{t}" for c, t in set(sheet_n) - paired_n),
        "rows_per_concept": {"adapted": count_concepts(sheet_a), "native": count_concepts(sheet_n)},
    }
    if errors:
        values.update({"median_position_error_of_scene": round(float(np.median(errors)), 5),
                       "max_position_error_of_scene": round(float(np.max(errors)), 5),
                       "same_num_frames_share": round(float(np.mean(np.equal(frame_differences, 0))), 4),
                       "per_object": rows})
    # Sizes are not compared: size_x/y/z are the sides of each object's box nearest the world axes, and the two runs'
    # world axes differ by a rotation, so the same side can be filed under a different axis.
    report.add("scene sheet: objects", "MEASURED", "" if errors else "no rows could be paired", values=values)


def count_concepts(sheet):
    counts = {}
    for concept, _ in sheet:
        counts[concept] = counts.get(concept, 0) + 1
    return counts


def verdict(report):
    """MATCHES, DIFFERS or INCOMPLETE, with every criterion's value, against the provisional CRITERIA."""
    rows = {r["comparison"]: r for r in report.rows}
    checks, missing = [], []
    for name, (comparison, key, direction, limit) in CRITERIA.items():
        value = rows.get(comparison, {}).get("values", {}).get(key)
        if value is None:
            missing.append(name)
            checks.append({"criterion": name, "value": None, "limit": f"{direction} {limit}", "ok": None})
            continue
        ok = value >= limit if direction == ">=" else value <= limit
        checks.append({"criterion": name, "value": value, "limit": f"{direction} {limit}", "ok": bool(ok)})
    fails = [r["comparison"] for r in report.rows if r["status"] == "FAIL"]
    checks.append({"criterion": "no FAIL in the comparison", "value": fails, "limit": "[]", "ok": not fails})
    if fails or any(c["ok"] is False for c in checks):
        state = "DIFFERS"
    elif missing:
        state = "INCOMPLETE"
    else:
        state = "MATCHES"
    return {"verdict": state, "provisional": True, "criteria": checks}


def compare(adapted, native, masks):
    adapted, native, masks = Path(adapted), Path(native), Path(masks) if masks else None
    has = {
        "adapted_depth": (adapted / "depth.npz").is_file(),
        "masks": masks is not None and (masks / "manifest.json").is_file(),
        "adapted_outputs": (adapted / "labels.npz").is_file() and (adapted / "report.json").is_file(),
        "native_geometry": (native / "cameras.npz").is_file() and (native / "depth.npz").is_file(),
        "native_outputs": (native / "labels.npz").is_file() and (native / "report.json").is_file(),
    }
    report = Report()
    missing = lambda *keys: [k for k in keys if not has[k]]  # noqa: E731

    check_run_format(report, "adapted", adapted, has["adapted_outputs"])
    check_run_format(report, "native", native, has["native_outputs"])
    if not has["adapted_depth"]:
        report.add("format: adapted/depth.npz", "BLOCKED", needs=["adapted_depth"])
    for name in ("adapted_outputs", "native_geometry", "native_outputs"):
        if not has[name]:
            report.add(f"format: {name.replace('_', ' ')}", "BLOCKED", needs=[name])
    check_masks(report, adapted, masks, has)

    if has["native_geometry"] and (adapted / "cameras.npz").is_file():
        names_a, names_n = frame_names(adapted), frame_names(native)
        report.add("frames: same frames in the same order", "PASS" if names_a == names_n else "FAIL",
                   f"{len(names_a)} frames" if names_a == names_n else f"{names_a} vs {names_n}")
    else:
        report.add("frames: same frames in the same order", "BLOCKED", needs=missing("native_geometry"))

    needs = missing("adapted_outputs", "native_outputs")
    if needs:
        report.add("masks: both runs used identical masks", "BLOCKED", needs=needs)
    else:
        summaries = [json.loads((run / "tracks" / "summary.json").read_text()).get("saved_masks")
                     for run in (adapted, native)]
        same = all(summaries) and summaries[0]["fingerprint"] == summaries[1]["fingerprint"]
        report.add("masks: both runs used identical masks", "PASS" if same else "FAIL",
                   summaries[0]["fingerprint"][:12] if same else "fingerprints differ or a run did not use --masks-dir")

    needs = missing("adapted_depth", "native_geometry")
    alignment = None
    shapes = None if needs else (load_npz(adapted / "depth.npz")["depth"].shape, load_npz(native / "depth.npz")["depth"].shape)
    if needs:
        report.add("Sim(3) alignment of the reconstructions", "BLOCKED", needs=needs)
        report.add("cameras after alignment", "BLOCKED", needs=needs)
    elif shapes[0] != shapes[1]:
        report.add("Sim(3) alignment of the reconstructions", "FAIL",
                   f"depth maps differ: adapted {shapes[0]}, native {shapes[1]}")
    else:
        sim3, diag, residual, count = fit_alignment(adapted, native)
        alignment = sim3, diag
        report.add("Sim(3) alignment of the reconstructions", "MEASURED", values={
            "scale_adapted_to_native": round(sim3[0], 5), "residual_of_scene": round(residual / diag, 5),
            "scene_size_native_units": round(diag, 4), "pixel_pairs": count})
        compare_cameras(report, adapted, native, sim3, diag)

    needs = missing("adapted_outputs", "native_outputs")
    matches = None
    if needs:
        report.add("object labels", "BLOCKED", needs=needs)
        report.add("tracking consistency", "BLOCKED", needs=needs)
    else:
        report_a = json.loads((adapted / "report.json").read_text())
        report_n = json.loads((native / "report.json").read_text())
        same = report_a["prompts"] == report_n["prompts"]
        report.add("object labels: same concept list", "PASS" if same else "FAIL",
                   f"{report_a['prompts']} vs {report_n['prompts']}" if not same else ", ".join(report_a["prompts"]))
        report.add("object labels: objects per concept", "MEASURED", values={
            "adapted": report_a["objects_per_concept"], "native": report_n["objects_per_concept"]})
        if same:
            matches = compare_tracking(report, adapted, native, report_a["prompts"])

    needs = missing("adapted_depth", "native_geometry", "adapted_outputs", "native_outputs")
    if needs or matches is None or alignment is None:
        report.add("3D object positions after alignment", "BLOCKED", needs=needs)
        report.add("object point clouds after alignment", "BLOCKED", needs=needs)
    else:
        compare_objects(report, adapted, native, matches, *alignment)

    if not ((adapted / SHEET).is_file() and (native / SHEET).is_file()):
        report.add("scene sheet: objects", "BLOCKED", needs=["scene_sheets", *needs])
    elif needs or matches is None or alignment is None:
        report.add("scene sheet: objects", "BLOCKED", needs=needs)
    else:
        compare_sheets(report, adapted, native, matches, *alignment)
    return report


def show(report):
    for row in report.rows:
        print(f"{row['status']:9s}{row['comparison']}{'  ' + row['detail'] if row['detail'] else ''}")
        summary = {k: v for k, v in row["values"].items() if k not in ("per_object", "per_prompt")}
        if summary:
            print(f"         {summary}")
        for need in row["needs"]:
            print(f"         needs {need}")
    print(report.counts())
    result = verdict(report)
    print(f"verdict (provisional pass marks): {result['verdict']}")
    for c in result["criteria"]:
        mark = {True: "ok  ", False: "FAIL", None: "n/a "}[c["ok"]]
        print(f"  {mark} {c['criterion']}: {c['value']} (needs {c['limit']})")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", help="folder holding adapted/ and native/")
    p.add_argument("--adapted", help="adapted run, instead of --run-dir")
    p.add_argument("--native", help="native run, instead of --run-dir")
    p.add_argument("--masks", help="the SAM 3 masks folder both runs used")
    p.add_argument("--out", help="JSON report (default: comparison.json next to adapted/)")
    args = p.parse_args(argv)
    if not args.run_dir and not (args.adapted and args.native):
        p.error("give --run-dir, or both --adapted and --native")
    adapted = Path(args.adapted) if args.adapted else Path(args.run_dir) / "adapted"
    native = Path(args.native) if args.native else Path(args.run_dir) / "native"

    report = compare(adapted, native, args.masks)
    show(report)
    out = Path(args.out) if args.out else adapted.parent / "comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"counts": report.counts(), **verdict(report), "comparisons": report.rows}, indent=1))
    print(f"report -> {out}")
    return 1 if report.has_fail() else 0


if __name__ == "__main__":
    sys.exit(main())
