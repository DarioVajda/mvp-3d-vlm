"""Scene sheet for one question about one of the saved VGGT scenes.

Takes a scene of the VGGT export (tasks/vggt_data/scenes/<dataset>/<scene>/) and a question as one string, and writes
the scene sheet of the objects the question names:

  1. prompts    prompt_object_extractor.py picks the SAM 3 prompts from the question. If it finds none, its syntactic
                mode is tried (every noun phrase). If that finds none either, the sheet has no objects.
  2. geometry   adapt_vggt.py converts the saved VGGT cameras and dense depth to OpenVocab-4D's format, after
                validating them. The dense depth (depth.npy, depth_conf.npy) is only on the cluster, in
                data/vggt/<dataset>/<scene>/; without it the script stops, unless --patch-depth-fallback is given:
                the depth is then approximated from the exported patch points (one depth per 14x14 patch), with a
                warning, and question.json records "depth": "patch_points". Positions are coarser and sizes can
                take in some background, so use it where the dense depth cannot be had.
  3. SAM 3      sam3_infer.py segments each prompt in every selected frame.
  4. tracking   OpenVocab-4D's track stage (milestone4_video.py) links the masks into objects, using the saved VGGT
                cameras and depth. By default (--association overlap) a detection continues the track whose 3D
                points it overlaps; --association centroid is the original rule (3D centres within 5% of the first
                frame's diagonal). A mask's depth counts by --depth-rule: mask (default), its pixels with confidence
                at least the lower of the sequence's --conf-percentile and the mask's own median; global, the
                original rule (the sequence's percentile only). It keeps an object seen in enough of the selected
                frames, a minimum that scales linearly with their number: 1 of 16, 4 of 128. Its fuse stage is not
                run: the scene sheet only needs the tracks.
  5. sheet      scene_sheet_creator.py writes objects.csv and cameras.csv (their columns are described there).

On stdout it prints exactly two lines, the path of objects.csv and then of cameras.csv. Every log goes to stderr.
From Python, create_scene_sheet(scene, question) returns the same two paths.

  python src/mvp/openvocab_adapter/question_scene_sheet.py scannet/scene0025_01 \
      "Measuring from the closest point of each object, which of these objects (table, chair) is the closest to the fan?"

By default it uses all 128 VGGT frames (--frames all): neighbouring frames overlap, so tracking can follow an object,
and objects seen only briefly are found. The VLM sees 16 of them (--sampling uniform128); the sheet's 128 frame columns
are mapped onto those 16 when the prompt is written (mvp.benchmarks.scene_sheet, visibility "nearby").
--frames subset:uniform16 uses only the VLM's 16 frames, as every sheet before 2026-11-24 did, and
--association centroid --depth-rule global the original tracking with them.

Work is kept and reused, in tasks/openvocab_adapter/out/<dataset>__<scene>/<frames>/ (<frames> is "all" by default,
"subset-uniform16" for the 16 frames):

  scene/              the converted geometry (frames/, cameras.npz, depth.npz) and tracks/<prompt>.npz
  masks/<prompts>/    the SAM 3 masks of the prompts that were segmented together
  sheets/<prompts>/   objects.csv, cameras.csv and question.json; the default --out-dir

A later question about the same scene runs SAM 3 and tracking only for prompts that have no tracks yet (tracks are
independent between prompts), so a question that names known objects only rebuilds the sheet. Tracks made with other
settings (re-identification, SAM 3 backend or thresholds, association, depth rule) are made again. --fresh starts the
scene over.

Needs the OpenVocab-4D environment (hpc/setup_env.sh) with a GPU: SAM 3, and DINOv2 for re-identification unless
--no-reid is given.
"""
import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adapt_vggt  # noqa: E402
import common  # noqa: E402
import scene_sheet_creator as sheet  # noqa: E402

SAM3_INFER = common.HERE / "sam3_infer.py"
DENSE_FILES = ("depth.npy", "depth_conf.npy")
MAX_FOLDER_NAME = 80
# All 128 VGGT frames. The VLM's 16 (meta.json's uniform16, np.linspace(0, 127, 16, dtype=int)) are
# "subset:uniform16", the default before 2026-11-24.
DEFAULT_FRAMES = "all"
# milestone 4 keeps a track seen in at least this many frames: linear in the frame count (min_track_frames there)
MIN_TRACK_FRAMES_RULE = "linear: 1 at 16 frames, 4 at 128, at least 1"
# Tracking (milestone4_video.py's options). The defaults are the fixes of 2026-11-24; ORIGINAL is milestone 4's own
# behaviour, which every sheet made before used. OVERLAP_OPTIONS are --association overlap's settings: provisional
# values, to be tuned on the dev scenes.
DEFAULT_ASSOCIATION, DEFAULT_DEPTH_RULE = "overlap", "mask"
ORIGINAL = {"association": "centroid", "depth_rule": "global"}
OVERLAP_OPTIONS = {"voxel_frac": 0.065, "overlap_min": 0.10, "reid_radius_depth": 0.5, "merge_overlap": 0.2,
                   "duplicate_overlap": 0.6}


def log(message):
    print(f"[question] {message}", file=sys.stderr)


def log_stream():
    """Where the steps' own logs go: stderr, so that stdout carries only the two paths."""
    try:
        sys.stderr.fileno()
        return sys.stderr
    except (AttributeError, OSError, ValueError):  # stderr replaced by an object that is not a real file
        return sys.__stderr__


def extract_prompts(question, syntactic=False):
    """(SAM 3 prompts, how they were found) for a question.

    The extractor's full mode keeps the physical things the question names. When it keeps nothing, its syntactic mode
    (every noun phrase) is used instead; `syntactic` asks for that mode from the start.
    """
    import prompt_object_extractor as extractor  # needs spaCy and NLTK, so it is only imported when used

    if not syntactic:
        prompts = extractor.main(question)
        if prompts:
            return prompts, "full"
        log("no physical object found in the question; trying the syntactic mode (every noun phrase)")
    prompts = extractor.main(question, syntactic=True)
    return prompts, "syntactic" if syntactic else "syntactic fallback"


def folder_name(prompts):
    """Folder for a list of prompts: 'chair+trash_can'; a long list is cut and ends with a hash of the whole list."""
    if not prompts:
        return "no_prompts"
    name = "+".join(p.replace(" ", "_") for p in prompts)
    if len(name) > MAX_FOLDER_NAME:
        name = f"{name[:MAX_FOLDER_NAME]}~{hashlib.sha256(name.encode()).hexdigest()[:8]}"
    return name


def frame_selection(frames):
    """(selection spec, its name in the work folder) for a preset name or a spec of common.select_frames."""
    if frames in common.PRESETS:
        return common.PRESETS[frames], frames
    return frames, frames.replace(":", "-").replace(",", "_")


def find_dense(scene, dense_dir=None, patch_depth_fallback=False):
    """The folder with the scene's dense depth: --dense-dir, else $DENSE_ROOT/<scene>, else data/vggt/<scene>.

    Without it this stops, or with patch_depth_fallback warns and returns None (the depth is then approximated).
    """
    folder = Path(dense_dir) if dense_dir else Path(os.environ.get("DENSE_ROOT") or common.DENSE_ROOT) / scene
    missing = [name for name in DENSE_FILES if not (folder / name).is_file()]
    if not missing:
        return folder
    absent = f"no dense depth for {scene}: {folder} has no {' or '.join(missing)}"
    if not patch_depth_fallback:
        sys.exit(f"[question] {absent}. The scene sheet needs VGGT's per-pixel depth, which is saved on the cluster in "
                 "data/vggt/<dataset>/<scene>/ and is not part of the exported scenes. Run this where data/vggt/ is, "
                 "pass --dense-dir (or set DENSE_ROOT), or pass --patch-depth-fallback to approximate the depth from "
                 "the exported patch points.")
    log(f"WARNING: {absent}. --patch-depth-fallback: the depth is approximated from the exported patch points, one "
        "depth per 14x14 patch, so the sheet's positions are coarser and its sizes can take in some background")
    return None


def ensure_geometry(scene, data_root, spec, label, dense_dir, scene_out, patch_depth_fallback=False):
    """Convert the saved VGGT cameras and depth into scene_out, unless an earlier run already did.

    Returns where the depth came from: "dense", or "patch_points" when there is no dense depth and
    patch_depth_fallback allows the approximation. Approximated geometry is converted again once dense depth is found.
    """
    meta = json.loads((Path(data_root) / "scenes" / scene / "meta.json").read_text())
    selected = common.select_frames(spec, meta["num_frames"], meta)
    report_path = scene_out / "adapter_report.json"  # the adapter writes it last, so it marks a finished conversion
    report = {}
    if report_path.is_file() and (scene_out / "cameras.npz").is_file() and (scene_out / "depth.npz").is_file():
        report = json.loads(report_path.read_text())
        if (report.get("scene"), report.get("selected_frames")) != (scene, selected):
            report = {}
    if report.get("depth_npz") == "written":  # the dense depth
        log(f"reusing the converted geometry in {scene_out}")
        return "dense"
    dense = find_dense(scene, dense_dir, patch_depth_fallback)
    if dense is None and report.get("depth_source") == "patch_points":
        log(f"reusing the converted geometry in {scene_out} (depth approximated from the patch points)")
        return "patch_points"
    shutil.rmtree(scene_out / "tracks", ignore_errors=True)  # tracks of an earlier, different geometry
    depth_args = ["--dense-dir", str(dense)] if dense else ["--patch-depth"]
    code = adapt_vggt.main(["--scene", scene, "--data-root", str(data_root), "--frames", spec, "--label", label,
                            *depth_args, "--out", str(scene_out), "--overwrite"])
    if code != 0:
        sys.exit("[question] the saved VGGT outputs failed the adapter's checks (see above); nothing was converted")
    return "dense" if dense else "patch_points"


def settings_file(scene_out, prompt):
    return sheet.track_file(scene_out, prompt).with_suffix(".settings.json")


def has_tracks(scene_out, prompt, settings):
    """Whether the prompt was already tracked in this scene, with these settings."""
    path = settings_file(scene_out, prompt)
    return (sheet.track_file(scene_out, prompt).is_file() and path.is_file()
            and json.loads(path.read_text()).get("settings") == settings)


def find_masks(masks_root, prompt, settings, stems):
    """A finished SAM 3 run that covers this prompt and these frames with these SAM 3 settings, or None."""
    if not masks_root.is_dir():
        return None
    for folder in sorted(p for p in masks_root.iterdir() if (p / "manifest.json").is_file()):
        manifest = json.loads((folder / "manifest.json").read_text())
        if (prompt in manifest["prompts"] and manifest.get("backend") == settings["backend"]
                and manifest["min_score"] == settings["min_score"]
                and set(stems) <= {frame["stem"] for frame in manifest["frames"]}):
            return folder
    return None


def segment(frames_dir, prompts, masks_dir, backend, min_score, checkpoint=None, overlays=False):
    """SAM 3 on every frame of frames_dir for these prompts (sam3_infer.py, in its own process to free the GPU after)."""
    cmd = [sys.executable, "-u", str(SAM3_INFER), "--images", str(frames_dir), "--frames", "all",
           "--prompts", ",".join(prompts), "--out", str(masks_dir), "--backend", backend,
           "--min-score", str(min_score), "--overwrite"]
    if checkpoint:
        cmd += ["--checkpoint", str(checkpoint)]
    if overlays:
        cmd.append("--overlays")
    log(f"SAM 3 for {prompts}")
    if subprocess.run(cmd, stdout=log_stream()).returncode != 0:
        sys.exit("[question] SAM 3 failed, see its log above")


def ensure_tracks(scene_out, masks_root, prompts, settings, checkpoint=None, overlays=False):
    """Make sure every prompt has tracks in scene_out, running SAM 3 and the track stage for those that have none."""
    todo = [p for p in prompts if not has_tracks(scene_out, p, settings)]
    if len(todo) < len(prompts):
        log(f"reusing the tracks of {[p for p in prompts if p not in todo]}")
    if not todo:
        return
    frames_dir = scene_out / "frames"
    stems = [p.stem for p in common.list_images(frames_dir)]
    unsegmented = [p for p in todo if find_masks(masks_root, p, settings, stems) is None]
    if unsegmented:
        segment(frames_dir, unsegmented, masks_root / folder_name(unsegmented), settings["backend"],
                settings["min_score"], checkpoint, overlays)
    groups = {}  # the track stage reads one masks folder per run
    for prompt in todo:
        groups.setdefault(find_masks(masks_root, prompt, settings, stems), []).append(prompt)
    for masks_dir, group in groups.items():
        for prompt in group:  # so that a failed run cannot leave an old track that looks current
            sheet.track_file(scene_out, prompt).unlink(missing_ok=True)
            settings_file(scene_out, prompt).unlink(missing_ok=True)
        sheet.run_pipeline(frames_dir, scene_out, group, "track", 1, no_reid=not settings["reid"],
                           masks_dir=masks_dir, conf_percentile=settings["conf_percentile"],
                           min_points=settings["min_points"], stdout=log_stream(), **tracker_options(settings))
        for prompt in group:
            settings_file(scene_out, prompt).write_text(json.dumps(
                {"prompt": prompt, "masks": masks_dir.name, "settings": settings}, indent=1))


def track_settings(reid, backend, min_score, conf_percentile, min_points=None, association=DEFAULT_ASSOCIATION,
                   depth_rule=DEFAULT_DEPTH_RULE, mask_conf_quantile=sheet.MASK_CONF_QUANTILE, overlap_options=None):
    """The settings a prompt's masks and tracks are kept under; tracks made under other settings are made again.

    min_points None is the depth rule's default (scene_sheet_creator.MIN_POINTS). With the original tracking
    (ORIGINAL) the settings are exactly those every earlier sheet recorded, so their tracks are reused; otherwise they
    also name the association, the depth rule and their options (overlap_options: OVERLAP_OPTIONS by default)."""
    if association not in ("overlap", "centroid"):
        raise ValueError(f"association must be overlap or centroid, not {association!r}")
    # min_track_frames names milestone 4's rule, so that tracks made under its old fixed 3 are made again
    settings = {"reid": bool(reid), "backend": backend, "min_score": min_score, "conf_percentile": conf_percentile,
                "min_points": sheet.min_points_for(depth_rule, min_points), "min_track_frames": MIN_TRACK_FRAMES_RULE}
    if {"association": association, "depth_rule": depth_rule} == ORIGINAL:
        return settings
    settings.update(association=association, depth_rule=depth_rule)
    if depth_rule == "mask":
        settings["mask_conf_quantile"] = float(mask_conf_quantile)
    if association == "overlap":
        settings.update({**OVERLAP_OPTIONS, **(overlap_options or {})})
    return settings


def tracker_options(settings):
    """run_pipeline's keyword arguments for the tracking settings beyond the original ones (none for ORIGINAL, so
    milestone 4 runs exactly as before)."""
    options = {"association": settings.get("association"), "depth_rule": settings.get("depth_rule"),
               "mask_conf_quantile": settings.get("mask_conf_quantile")}
    extra = {k: settings[k] for k in OVERLAP_OPTIONS if k in settings}
    return {**options, "tracker_options": extra or None}


def sheet_depth_options(settings):
    """build_scene_sheet's depth rule for these tracking settings: the original rule when they name none."""
    rule = settings.get("depth_rule", "global")
    return {"depth_rule": rule, "mask_conf_quantile": settings.get("mask_conf_quantile", sheet.MASK_CONF_QUANTILE)}


def prepare_scene(scene, questions, frames=DEFAULT_FRAMES, dense_dir=None, data_root=None, work_dir=None,
                  syntactic=False, reid=True, conf_percentile=55.0, min_points=None, backend="transformers",
                  min_score=0.5, checkpoint=None, overlays=False, patch_depth_fallback=False,
                  association=DEFAULT_ASSOCIATION, depth_rule=DEFAULT_DEPTH_RULE,
                  mask_conf_quantile=sheet.MASK_CONF_QUANTILE, overlap_options=None):
    """Segment and track the prompts of several questions about one scene together. Returns the prompts.

    Without it, create_scene_sheet starts SAM 3 and the track stage (each in its own process, with its models loaded
    again) for every question that names an object not tracked yet. With it, they run once for all the questions, and
    create_scene_sheet then only builds each sheet. The sheets are the same: SAM 3 and the tracker handle each prompt
    on its own. The arguments are create_scene_sheet's.
    """
    with contextlib.redirect_stdout(sys.stderr):
        try:
            data_root, scene = common.find_scene(scene, data_root)
        except (FileNotFoundError, ValueError) as error:
            sys.exit(f"[question] {error}")
        spec, label = frame_selection(frames)
        work = (Path(work_dir) if work_dir else common.OUT_ROOT / scene.replace("/", "__") / label).resolve()
        prompts = []
        for question in questions:
            if question.strip():
                prompts += [p for p in extract_prompts(question, syntactic)[0] if p not in prompts]
        log(f"{scene}: {len(questions)} questions name {len(prompts)} prompts {prompts}")
        ensure_geometry(scene, data_root, spec, label, dense_dir, work / "scene", patch_depth_fallback)
        settings = track_settings(reid, backend, min_score, conf_percentile, min_points, association, depth_rule,
                                  mask_conf_quantile, overlap_options)
        ensure_tracks(work / "scene", work / "masks", prompts, settings, checkpoint, overlays)
    return prompts


def create_scene_sheet(scene, question, frames=DEFAULT_FRAMES, dense_dir=None, data_root=None, out_dir=None, work_dir=None,
                       syntactic=False, reid=True, include_size=True, relative_positions=False, size_trim=0.02,
                       conf_percentile=55.0, min_points=None, backend="transformers", min_score=0.5, checkpoint=None,
                       overlays=False, fresh=False, patch_depth_fallback=False, association=DEFAULT_ASSOCIATION,
                       depth_rule=DEFAULT_DEPTH_RULE, mask_conf_quantile=sheet.MASK_CONF_QUANTILE,
                       overlap_options=None):
    """Write the scene sheet for a question about a saved VGGT scene. Returns (objects.csv, cameras.csv) paths.

    scene       "<dataset>/<scene>", or only the scene's name
    question    the question, as one string; it does not have to be one of the scene's questions.jsonl
    frames      a preset of common.PRESETS or a spec of common.select_frames; the default, "all", is the 128 frames
                VGGT saw, "subset:uniform16" the 16 the VLM sees.
                frame_<t> in the sheet counts the selected frames, cameras.csv names each one's image
    association, depth_rule, mask_conf_quantile, overlap_options, min_points: the tracking (track_settings)
    dense_dir   folder with depth.npy and depth_conf.npy (default: $DENSE_ROOT or data/vggt, then /<dataset>/<scene>)
    patch_depth_fallback  without dense depth, approximate it from the patch points (with a warning) instead of
                stopping
    out_dir     where the two CSVs go (default: <work_dir>/sheets/<prompts>)
    work_dir    where geometry, masks and tracks are kept (default: tasks/openvocab_adapter/out/<dataset>__<scene>/<frames>)
    The other arguments are the command line's options; see --help.
    """
    with contextlib.redirect_stdout(sys.stderr):  # every step prints its log; stdout is kept for the two paths
        if not question.strip():
            sys.exit("[question] the question is empty")
        try:
            data_root, scene = common.find_scene(scene, data_root)
        except (FileNotFoundError, ValueError) as error:
            sys.exit(f"[question] {error}")
        spec, label = frame_selection(frames)
        work = Path(work_dir) if work_dir else common.OUT_ROOT / scene.replace("/", "__") / label
        work = work.resolve()
        scene_out, masks_root = work / "scene", work / "masks"

        prompts, mode = extract_prompts(question, syntactic)
        log(f"{scene}: prompts {prompts} ({mode})" if prompts else f"{scene}: the question names nothing to segment")

        if fresh:
            log(f"--fresh: removing {scene_out} and {masks_root}")
            shutil.rmtree(scene_out, ignore_errors=True)
            shutil.rmtree(masks_root, ignore_errors=True)
        depth_source = ensure_geometry(scene, data_root, spec, label, dense_dir, scene_out, patch_depth_fallback)
        settings = track_settings(reid, backend, min_score, conf_percentile, min_points, association, depth_rule,
                                  mask_conf_quantile, overlap_options)
        ensure_tracks(scene_out, masks_root, prompts, settings, checkpoint, overlays)

        out = Path(out_dir).resolve() if out_dir else work / "sheets" / folder_name(prompts)
        objects_csv, cameras_csv = sheet.build_scene_sheet(
            scene_out, prompts, relative_positions, include_size, out, conf_percentile, settings["min_points"],
            size_trim, **sheet_depth_options(settings))
        (out / "question.json").write_text(json.dumps({
            "scene": scene, "question": question, "prompts": prompts, "prompt_mode": mode, "frames": spec,
            "tracking": settings, "size": include_size, "relative_positions": relative_positions,
            "size_trim": size_trim, "geometry": str(scene_out), "depth": depth_source,
        }, indent=1))
        if depth_source == "patch_points":
            log(f"WARNING: {objects_csv} was made with depth approximated from the patch points, not the dense depth")
    return objects_csv, cameras_csv


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scene", help="<dataset>/<scene>, e.g. scannet/scene0025_01, or only the scene's name")
    p.add_argument("question", help="the question, as one string")
    p.add_argument("--frames", default=DEFAULT_FRAMES,
                   help=f"frames to use: a preset ({', '.join(sorted(common.PRESETS))}) or all | uniform:K | subset:NAME | "
                        f"window:START:COUNT[:STRIDE] | list:a,b (default: {DEFAULT_FRAMES}, the 128 VGGT frames; "
                        "subset:uniform16 is the VLM's 16)")
    p.add_argument("--dense-dir", help="folder with the scene's depth.npy and depth_conf.npy "
                                       "(default: $DENSE_ROOT or data/vggt, then /<dataset>/<scene>)")
    p.add_argument("--patch-depth-fallback", action="store_true",
                   help="if the dense depth is not found, approximate it from the exported patch points (one depth "
                        "per 14x14 patch, coarser; a warning is printed) instead of stopping")
    p.add_argument("--data-root", help="folder holding the exported scenes/ (default: tasks/vggt_data, then the "
                                       "other places hpc/run_all.sh looks in)")
    p.add_argument("--out-dir", help="where to write the two CSVs (default: <work-dir>/sheets/<prompts>)")
    p.add_argument("--work-dir", help="where geometry, masks and tracks are kept and reused "
                                      "(default: tasks/openvocab_adapter/out/<dataset>__<scene>/<frames>)")
    p.add_argument("--fresh", action="store_true", help="discard the scene's kept geometry, masks and tracks first")
    p.add_argument("--syntactic", action="store_true",
                   help="take every noun phrase of the question as a prompt (the extractor's syntactic mode)")
    p.add_argument("--no-reid", action="store_true",
                   help="track without DINOv2 re-identification, as the comparison in tasks/openvocab_adapter/results/ did")
    p.add_argument("--no-size", action="store_true", help="leave out the size columns")
    p.add_argument("--relative-positions", action="store_true",
                   help="per-frame positions in camera coordinates instead of the 0/1 seen flags")
    p.add_argument("--size-trim", type=float, default=0.02,
                   help="percent of points ignored at each end of every axis when measuring size")
    p.add_argument("--conf-percentile", type=float, default=55.0,
                   help="the sequence's depth-confidence percentile a pixel needs (--depth-rule global), or the most "
                        "it needs (mask), in tracking and in the sheet")
    p.add_argument("--min-points", type=int, default=None,
                   help="pixels of usable depth an object needs to count as seen in a frame, in tracking and in the "
                        f"sheet (default: the depth rule's, {sheet.MIN_POINTS})")
    p.add_argument("--depth-rule", choices=sheet.DEPTH_RULES, default=DEFAULT_DEPTH_RULE,
                   help="which pixels of a mask have usable depth: mask (default), confidence at least the lower of "
                        "--conf-percentile and the mask's own --mask-conf-quantile; global, the original rule")
    p.add_argument("--mask-conf-quantile", type=float, default=sheet.MASK_CONF_QUANTILE,
                   help="--depth-rule mask: percentile of the mask's own confidence its pixels need")
    p.add_argument("--association", choices=["overlap", "centroid"], default=DEFAULT_ASSOCIATION,
                   help="how detections join tracks (milestone4_video.py): overlap (default), by their 3D points; "
                        "centroid, the original rule")
    for name, value in OVERLAP_OPTIONS.items():
        p.add_argument(f"--{name.replace('_', '-')}", type=float, default=value,
                       help=f"--association overlap: milestone4_video.py's --{name.replace('_', '-')} (default {value})")
    p.add_argument("--backend", choices=["transformers", "sam3"], default="transformers",
                   help="how SAM 3 is run, see sam3_infer.py (transformers: the cached snapshot; sam3: Meta's package)")
    p.add_argument("--checkpoint", help="--backend sam3 only: a local sam3.pt")
    p.add_argument("--min-score", type=float, default=0.5,
                   help="SAM 3 detections are saved from this score up; tracking then applies its own 0.55")
    p.add_argument("--overlays", action="store_true", help="also save SAM 3's masks drawn on each frame")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    objects_csv, cameras_csv = create_scene_sheet(
        args.scene, args.question, frames=args.frames, dense_dir=args.dense_dir, data_root=args.data_root,
        out_dir=args.out_dir, work_dir=args.work_dir, syntactic=args.syntactic, reid=not args.no_reid,
        include_size=not args.no_size, relative_positions=args.relative_positions, size_trim=args.size_trim,
        conf_percentile=args.conf_percentile, min_points=args.min_points, backend=args.backend,
        min_score=args.min_score, checkpoint=args.checkpoint, overlays=args.overlays, fresh=args.fresh,
        patch_depth_fallback=args.patch_depth_fallback, association=args.association, depth_rule=args.depth_rule,
        mask_conf_quantile=args.mask_conf_quantile,
        overlap_options={name: getattr(args, name) for name in OVERLAP_OPTIONS})
    print(objects_csv)
    print(cameras_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
