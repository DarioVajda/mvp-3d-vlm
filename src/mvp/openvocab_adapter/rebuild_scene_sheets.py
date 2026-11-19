"""Rebuild the scene sheets of a finished sheet run (scene_sheets_75/) from its saved tracks, without SAM 3.

hpc/run_sheets.sh makes the sheets with SAM 3 and tracking (GPU). Its output keeps, per scene, everything the sheet
itself is computed from except the depth:

  <dataset>__<scene>/subset-uniform16/scene/   cameras.npz, adapter_report.json, tracks/<prompt>.npz
  <dataset>__<scene>/subset-uniform16/sheets/<prompts>/question.json   the prompts and settings of each sheet
  <dataset>__<scene>/gold/sheets/<prompts>/question.json               the prompts of each gold sheet

This script writes every sheet again with the current scene_sheet_creator.py, so a change to how sheets are computed
or written (sizes, columns) reaches all of them in minutes on a CPU:

  pipeline  scene_sheet_creator.build_scene_sheet on the saved tracks, with the settings in each question.json. The
            zip leaves out scene/depth.npz (the shared VGGT depth, cut to the 16 frames), so it is made again first
            from data/vggt/<dataset>/<scene>/depth.npy and depth_conf.npy, cut to adapter_report.json's
            selected_frames, exactly as adapt_vggt.py writes it. Checked: the dense depth's frame_indices.npy equals
            the report's source_video_frames, and every track has the depth's shape.
  gold      scene_sheet_creator.build_gold_scene_sheet with each question.json's prompts (VSI-Bench's labels).

objects.csv, cameras.csv (pipeline) and sheet.json (gold) are replaced; question.json, the tracks, the masks and
index.csv are left as they are (the sheet folders and their prompts don't change). Every sheet is then checked as
scene_sheet_report.py checks it. A sheet whose prompt has no saved track is not rebuilt and is reported: making that
track needs SAM 3 (scene_sheet_report.py on the cluster).

Wrong tracks are never made again here, unlike question_scene_sheet.py, which deletes a scene's tracks when its
depth.npz is missing. Needs numpy and Pillow only.

  .venv/Scripts/python src/mvp/openvocab_adapter/rebuild_scene_sheets.py
  .venv/Scripts/python src/mvp/openvocab_adapter/rebuild_scene_sheets.py --scenes arkitscenes__41069025 --source gold
  .venv/Scripts/python src/mvp/openvocab_adapter/rebuild_scene_sheets.py --workers 8 --drop-depth

The steps' logs go to stderr, one summary line per scene and the totals to stdout. Exit status 1 if any sheet failed.
"""
import argparse
import contextlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import scene_sheet_creator as sheet  # noqa: E402
import scene_sheet_report as checks  # noqa: E402

DEFAULT_ROOT = common.REPO / "data" / "scene_sheets_75"
PIPELINE_LABEL = "subset-uniform16"
DENSE_FILES = ("depth.npy", "depth_conf.npy")


def log(message):
    print(f"[rebuild] {message}", file=sys.stderr, flush=True)


def sheet_folders(parent):
    """The sheet folders under parent/sheets/, each one with its question.json."""
    folder = parent / "sheets"
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if (p / "question.json").is_file())


def ensure_depth(scene_out, dense_root):
    """Write scene_out/depth.npz from the dense VGGT depth unless it is there. Returns (scene, its shape)."""
    report = json.loads((scene_out / "adapter_report.json").read_text())
    scene, selected = report["scene"], report["selected_frames"]
    if report.get("depth_source") != "dense":
        raise RuntimeError(f"{scene_out} was made with depth from {report.get('depth_source')!r}, not the dense depth")
    path = scene_out / "depth.npz"
    if path.is_file():
        with np.load(path) as z:
            shape = z["depth"].shape
        if shape[0] == len(selected):
            return scene, shape
        log(f"{path} has {shape[0]} frames, not the {len(selected)} selected ones: making it again")
    dense = Path(dense_root) / scene
    missing = [n for n in DENSE_FILES if not (dense / n).is_file()]
    if missing:
        raise RuntimeError(f"no dense depth for {scene}: {dense} has no {' or '.join(missing)}")
    indices = dense / "frame_indices.npy"
    if indices.is_file() and report.get("source_video_frames") is not None:
        source = np.load(indices)[selected].tolist()
        if source != report["source_video_frames"]:
            raise RuntimeError(f"{dense} is another reconstruction: its frames {selected} are video frames {source}, "
                               f"adapter_report.json says {report['source_video_frames']}")
    depth = np.load(dense / "depth.npy", mmap_mode="r")
    conf = np.load(dense / "depth_conf.npy", mmap_mode="r")
    if depth.shape != conf.shape or depth.shape[1:] != tuple(report["image_hw"]):
        raise RuntimeError(f"{dense}: depth {depth.shape} and conf {conf.shape}, the frames are {report['image_hw']}")
    # as adapt_vggt.write_outputs
    np.savez_compressed(path, depth=np.asarray(depth[selected]).astype(np.float16),
                        conf=np.asarray(conf[selected]).astype(np.float16))
    log(f"{scene}: wrote {path} from {dense}")
    return scene, (len(selected), *depth.shape[1:])


def check_tracks(scene_out, shape):
    """Prompts whose saved tracks don't have the depth's shape (they belong to other geometry)."""
    wrong = []
    for path in sorted((scene_out / "tracks").glob("*.npz")):
        with np.load(path) as z:
            if "labels" in z.files and z["labels"].shape != tuple(shape):
                wrong.append(f"{path.name} {z['labels'].shape}")
    if wrong:
        raise RuntimeError(f"tracks that don't match the depth {tuple(shape)}: {', '.join(wrong)}")


def rebuild_pipeline(scene_dir, dense_root):
    """Rebuild every pipeline sheet of one scene folder. Returns (results, scene)."""
    work = scene_dir / PIPELINE_LABEL
    scene_out = work / "scene"
    folders = sheet_folders(work)
    if not folders:
        return [], None
    scene, shape = ensure_depth(scene_out, dense_root)
    check_tracks(scene_out, shape)
    results = []
    for out in folders:
        record = json.loads((out / "question.json").read_text())
        prompts = record["prompts"]
        tracking = record.get("tracking", {})
        result = {"sheet": str(out.relative_to(scene_dir.parent)), "problems": []}
        untracked = [p for p in prompts if not sheet.track_file(scene_out, p).is_file()]
        if untracked:
            result["problems"] = [f"not rebuilt: no saved tracks for {untracked} (needs SAM 3)"]
            results.append(result)
            continue
        try:
            with contextlib.redirect_stdout(sys.stderr):
                objects_csv, cameras_csv = sheet.build_scene_sheet(
                    scene_out, prompts, record.get("relative_positions", False), record.get("size", True), out,
                    tracking.get("conf_percentile", 55.0), tracking.get("min_points", 300),
                    record.get("size_trim", 0.02))
            if record.get("relative_positions"):  # the report's checks are for the 0/1 frame columns
                result["counts"] = {}
            else:
                result["problems"], result["counts"] = checks.check_pipeline_sheet(objects_csv, cameras_csv, prompts)
        except SystemExit as error:  # scene_sheet_creator stops with sys.exit("[sheet] ...")
            result["problems"] = [f"not rebuilt: {error.code}"]
        except Exception as error:  # noqa: BLE001 - one sheet's failure must not stop the scene
            result["problems"] = [f"not rebuilt: {type(error).__name__}: {error}"]
        results.append(result)
    return results, scene


def rebuild_gold(scene_dir, scene, gold_dir):
    """Rebuild every gold sheet of one scene folder. Returns the results."""
    folders = sheet_folders(scene_dir / "gold")
    if not folders:
        return []
    if scene is None:
        scene = json.loads((folders[0] / "question.json").read_text())["scene"]
    info = sheet.load_gold_scene(scene, gold_dir)[1]
    results = []
    for out in folders:
        prompts = json.loads((out / "question.json").read_text())["prompts"]
        result = {"sheet": str(out.relative_to(scene_dir.parent)), "problems": []}
        try:
            with contextlib.redirect_stdout(sys.stderr):
                objects_csv, _ = sheet.build_gold_scene_sheet(scene, prompts, include_size=True, out_dir=out,
                                                              gold_dir=gold_dir)
            result["problems"], result["counts"] = checks.check_gold_sheet(objects_csv, prompts, info)
        except Exception as error:  # noqa: BLE001
            result["problems"] = [f"not rebuilt: {type(error).__name__}: {error}"]
        results.append(result)
    return results


def rebuild_scene(scene_dir, source, dense_root, gold_dir, drop_depth):
    """Rebuild one scene's sheets. Returns {"scene", "pipeline", "gold", "error", "seconds"}."""
    start = time.time()
    scene_dir = Path(scene_dir)
    summary = {"scene": scene_dir.name, "pipeline": [], "gold": [], "error": None}
    scene = None
    try:
        if source in ("pipeline", "both"):
            summary["pipeline"], scene = rebuild_pipeline(scene_dir, dense_root)
            if drop_depth:
                (scene_dir / PIPELINE_LABEL / "scene" / "depth.npz").unlink(missing_ok=True)
        if source in ("gold", "both"):
            summary["gold"] = rebuild_gold(scene_dir, scene, gold_dir)
    except Exception as error:  # noqa: BLE001 - reported per scene; the other scenes go on
        summary["error"] = f"{type(error).__name__}: {error}"
    summary["seconds"] = time.time() - start
    return summary


def scene_line(summary):
    def part(name):
        results = summary[name]
        failed = sum(bool(r["problems"]) for r in results)
        return f"{name} {len(results) - failed}/{len(results)} OK"
    line = f"{summary['scene']}: {part('pipeline')}, {part('gold')} ({summary['seconds']:.0f} s)"
    if summary["error"]:
        line += f"  FAILED: {summary['error']}"
    for name in ("pipeline", "gold"):
        for r in summary[name]:
            if r["problems"]:
                line += f"\n  {r['sheet']}: {'; '.join(r['problems'])}"
    return line


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(DEFAULT_ROOT),
                   help="the sheet run's output folder, holding <dataset>__<scene>/ (default: data/scene_sheets_75)")
    p.add_argument("--source", choices=["pipeline", "gold", "both"], default="both", help="which sheets to rebuild")
    p.add_argument("--scenes", nargs="+", help="only these scene folders, e.g. arkitscenes__41069025")
    p.add_argument("--dense-root", default=os.environ.get("DENSE_ROOT") or str(common.DENSE_ROOT),
                   help="the dense VGGT depth, <dataset>/<scene>/depth.npy (default: $DENSE_ROOT or data/vggt)")
    p.add_argument("--gold-dir", help="VSI-Bench's metadata (default: as scene_sheet_creator.py)")
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)),
                   help="scenes rebuilt at a time (default: half the CPUs, at most 8)")
    p.add_argument("--drop-depth", action="store_true",
                   help="delete each scene's depth.npz once its sheets are rebuilt (it is made again next time)")
    args = p.parse_args(argv)

    root = Path(args.root)
    scene_dirs = sorted(d for d in root.iterdir() if d.is_dir() and ((d / PIPELINE_LABEL).is_dir() or (d / "gold").is_dir()))
    if args.scenes:
        unknown = set(args.scenes) - {d.name for d in scene_dirs}
        if unknown:
            p.error(f"no such scene folders in {root}: {sorted(unknown)}")
        scene_dirs = [d for d in scene_dirs if d.name in args.scenes]
    if not scene_dirs:
        p.error(f"no scene folders in {root}")
    print(f"Rebuilding the {args.source} sheets of {len(scene_dirs)} scenes in {root}, {args.workers} at a time",
          flush=True)

    start = time.time()
    summaries = []
    jobs = (args.source, args.dense_root, args.gold_dir, args.drop_depth)
    if args.workers == 1:
        for d in scene_dirs:
            summaries.append(rebuild_scene(d, *jobs))
            print(scene_line(summaries[-1]), flush=True)
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            futures = [pool.submit(rebuild_scene, d, *jobs) for d in scene_dirs]
            for future in as_completed(futures):
                summaries.append(future.result())
                print(scene_line(summaries[-1]), flush=True)

    total = {name: [r for s in summaries for r in s[name]] for name in ("pipeline", "gold")}
    failed = {name: sum(bool(r["problems"]) for r in results) for name, results in total.items()}
    broken = [s["scene"] for s in summaries if s["error"]]
    print(f"\n{len(summaries)} scenes in {time.time() - start:.0f} s: "
          + ", ".join(f"{name} {len(total[name]) - failed[name]}/{len(total[name])} sheets OK" for name in total)
          + (f"; {len(broken)} scenes failed: {broken}" if broken else ""))
    return 1 if broken or any(failed.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
