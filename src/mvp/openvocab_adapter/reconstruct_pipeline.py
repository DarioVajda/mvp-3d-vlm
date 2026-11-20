"""Make every scene sheet of a finished sheet run (data/scene_sheets_75/) again from scratch, from the files saved in it.

For every scene folder <dataset>__<scene>/ of the run, both of its sheets/ folders are deleted and written again with
the current scene_sheet_creator.py, one sheet folder for each one that was there:

  subset-uniform16/sheets/<prompts>/   pipeline: objects.csv, cameras.csv, question.json
  gold/sheets/<prompts>/               gold: objects.csv, sheet.json, question.json

Each sheet is made from what the run saved, without SAM 3 or a GPU:

  pipeline  scene_sheet_creator.build_scene_sheet on subset-uniform16/scene/ (tracks/<prompt>.npz, cameras.npz,
            depth.npz, adapter_report.json), with the prompts and settings of the sheet's question.json. If
            depth.npz is missing (the shared zip leaves it out), it is made again from the dense VGGT depth
            (--dense-root), as rebuild_scene_sheets.py does.
  gold      scene_sheet_creator.build_gold_scene_sheet with the prompts of the sheet's question.json and the size
            columns. VSI-Bench's gold labels are the one input outside the run's folder: data/raw/VSI-Bench/meta_info/
            (--gold-dir), downloaded and checked against their SHA-256 if missing.

question.json is the record of the sheet's question, prompts and settings, so it is copied into the new folder as it
was; every other file is computed again, and nothing else of the old folder is kept. Each new sheet is then checked as
scene_sheet_report.py checks it. index.csv is read first: every sheet folder it names must be there, so none can be
lost. The run's other files (masks/, scene/, reports/, index.csv) are left as they are.

rebuild_scene_sheets.py instead writes the sheets' files again in place, keeping whatever else their folders hold.

The new sheets/ is built next to the old one, as sheets.new/, and replaces it only once every sheet in it is made and
passes its checks. Otherwise the old sheets/ is kept as it was and the failures are reported, so neither a failure nor
an interrupted run loses a sheet (the next run cleans up after an interrupted one). A sheet whose prompt has no saved
track can't be made without SAM 3 (scene_sheet_report.py on the cluster), so it fails that way. Needs numpy only.

  .venv/bin/python src/mvp/openvocab_adapter/reconstruct_pipeline.py
  .venv/bin/python src/mvp/openvocab_adapter/reconstruct_pipeline.py path/to/scene_sheets_75 --workers 4
  .venv/bin/python src/mvp/openvocab_adapter/reconstruct_pipeline.py --scenes arkitscenes__41069025

The steps' logs go to stderr, one summary line per scene and the totals to stdout. Exit status 1 if any sheet failed.
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import rebuild_scene_sheets as rebuild  # noqa: E402
import scene_sheet_creator as sheet  # noqa: E402

DEFAULT_ROOT = common.REPO / "data" / "scene_sheets_75"
SOURCES = {"pipeline": rebuild.PIPELINE_LABEL, "gold": "gold"}  # each source's folder, holding its sheets/


def log(message):
    print(f"[reconstruct] {message}", file=sys.stderr, flush=True)


def indexed_sheets(root):
    """{scene folder: {source: names of the sheet folders index.csv names}}, or None without an index.csv."""
    path = root / "index.csv"
    if not path.is_file():
        return None
    named = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for source, label in SOURCES.items():
                if not row[f"{source}_sheet"]:  # a sheet that wasn't made
                    continue
                folder = Path(row[f"{source}_sheet"])
                if folder.parent != Path(folder.parts[0]) / label / "sheets":
                    raise ValueError(f"index.csv, question {row['id']}: {folder} is not a {source} sheet folder")
                named.setdefault(folder.parts[0], {s: set() for s in SOURCES})[source].add(folder.name)
    return named


def recover(sheets):
    """Clean up after an interrupted run: put sheets/ back from sheets.old/ if the swap stopped between its two
    renames, then remove any sheets.new/ (unfinished) and sheets.old/ (already replaced)."""
    new, old = sheets.with_name("sheets.new"), sheets.with_name("sheets.old")
    if not sheets.exists() and old.is_dir():
        old.rename(sheets)
        log(f"{sheets}: put back from {old.name}/, where an interrupted run had left it")
    for leftover in (new, old):
        if leftover.exists():
            shutil.rmtree(leftover)
            log(f"removed {leftover}, left by an interrupted run")


def reconstruct_sheets(scene_dir, source, expected, dense_root, gold_dir):
    """Make one scene's pipeline or gold sheets/ again in sheets.new/, and swap it in if every sheet is made and
    passes its checks. Returns (results, one per sheet, whether sheets/ was replaced)."""
    work = scene_dir / SOURCES[source]
    sheets = work / "sheets"
    recover(sheets)
    folders = sorted(p for p in sheets.iterdir() if p.is_dir()) if sheets.is_dir() else []
    if missing := sorted(expected - {f.name for f in folders}):
        raise RuntimeError(f"index.csv names {source} sheets that {sheets} doesn't have: {missing}")
    if not folders:
        return [], False
    if unrecorded := [f.name for f in folders if not (f / "question.json").is_file()]:
        raise RuntimeError(f"{source} sheets with no question.json, which can't be made again: {unrecorded}")

    if source == "pipeline":
        scene_out = work / "scene"
        _, shape = rebuild.ensure_depth(scene_out, dense_root)
        rebuild.check_tracks(scene_out, shape)

        def build(out):
            return rebuild.build_pipeline_sheet(out, scene_out)
    else:
        scene = json.loads((folders[0] / "question.json").read_text())["scene"]
        info = sheet.load_gold_scene(scene, gold_dir)[1]

        def build(out):
            return rebuild.build_gold_sheet(out, scene, info, gold_dir)

    new = sheets.with_name("sheets.new")
    new.mkdir()
    results = []
    for folder in folders:
        out = new / folder.name
        out.mkdir()
        shutil.copyfile(folder / "question.json", out / "question.json")
        problems, counts = build(out)
        results.append({"sheet": str(folder.relative_to(scene_dir.parent)), "problems": problems, "counts": counts})
    if failed := sum(bool(r["problems"]) for r in results):
        shutil.rmtree(new)
        log(f"{sheets}: kept as it was, because {failed} of its {len(results)} new sheets failed")
        return results, False
    old = sheets.with_name("sheets.old")
    sheets.rename(old)
    new.rename(sheets)
    shutil.rmtree(old)
    return results, True


def reconstruct_scene(scene_dir, expected, dense_root, gold_dir):
    """Make one scene's sheets again. Returns {"scene", "pipeline", "gold", "replaced", "error", "seconds"}."""
    start = time.time()
    scene_dir = Path(scene_dir)
    summary = {"scene": scene_dir.name, "pipeline": [], "gold": [], "replaced": [], "error": None}
    errors = []
    for source in SOURCES:  # the two are independent: one failing doesn't stop the other
        try:
            summary[source], replaced = reconstruct_sheets(scene_dir, source, expected[source], dense_root, gold_dir)
            if replaced:
                summary["replaced"].append(source)
        except Exception as error:  # noqa: BLE001 - reported per scene; the other scenes go on
            errors.append(f"{source}: {type(error).__name__}: {error}")
    summary["error"] = "; ".join(errors) or None
    summary["seconds"] = time.time() - start
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", nargs="?", default=str(DEFAULT_ROOT),
                   help="the sheet run's folder, holding index.csv and <dataset>__<scene>/ (default: "
                        "data/scene_sheets_75)")
    p.add_argument("--scenes", nargs="+", help="only these scene folders, e.g. arkitscenes__41069025")
    p.add_argument("--gold-dir", help="VSI-Bench's metadata (default: data/raw/VSI-Bench/meta_info, as "
                                      "scene_sheet_creator.py)")
    p.add_argument("--dense-root", default=os.environ.get("DENSE_ROOT") or str(common.DENSE_ROOT),
                   help="the dense VGGT depth, <dataset>/<scene>/depth.npy, only read for a scene with no depth.npz "
                        "(default: $DENSE_ROOT or data/vggt)")
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)),
                   help="scenes made at a time (default: half the CPUs, at most 8)")
    args = p.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        p.error(f"no such folder: {root}")
    scene_dirs = sorted(d for d in root.iterdir()
                        if d.is_dir() and any((d / label).is_dir() for label in SOURCES.values()))
    try:
        named = indexed_sheets(root)
    except (KeyError, ValueError) as error:
        p.error(f"{root / 'index.csv'} is not a sheet run's index: {error}")
    if named is None:
        log(f"no index.csv in {root}: making again the sheet folders that are there, without checking that none "
            "is missing")
    elif unknown := sorted(set(named) - {d.name for d in scene_dirs}):
        p.error(f"index.csv names scene folders that {root} doesn't have: {unknown}")
    if args.scenes:
        if unknown := sorted(set(args.scenes) - {d.name for d in scene_dirs}):
            p.error(f"no such scene folders in {root}: {unknown}")
        scene_dirs = [d for d in scene_dirs if d.name in args.scenes]
    if not scene_dirs:
        p.error(f"no scene folders in {root}")
    try:  # once here, rather than in every scene at the same time
        sheet.ensure_gold_metadata(args.gold_dir)
    except RuntimeError as error:
        p.error(str(error))
    print(f"Making the pipeline and gold sheets of {len(scene_dirs)} scenes in {root} again, {args.workers} at a time",
          flush=True)

    start = time.time()
    none = {s: set() for s in SOURCES}
    jobs = [(d, (named or {}).get(d.name, none), args.dense_root, args.gold_dir) for d in scene_dirs]
    summaries = []
    if args.workers == 1:
        for job in jobs:
            summaries.append(reconstruct_scene(*job))
            print(rebuild.scene_line(summaries[-1]), flush=True)
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            for future in as_completed([pool.submit(reconstruct_scene, *job) for job in jobs]):
                summaries.append(future.result())
                print(rebuild.scene_line(summaries[-1]), flush=True)

    total = {s: [r for summary in summaries for r in summary[s]] for s in SOURCES}
    failed = {s: sum(bool(r["problems"]) for r in results) for s, results in total.items()}
    kept = sorted(f"{summary['scene']} {s}" for summary in summaries for s in SOURCES
                  if summary[s] and s not in summary["replaced"])
    broken = sorted(summary["scene"] for summary in summaries if summary["error"])
    print(f"\n{len(summaries)} scenes in {time.time() - start:.0f} s: "
          + ", ".join(f"{s} {len(total[s]) - failed[s]}/{len(total[s])} sheets OK" for s in SOURCES)
          + (f"; old sheets kept for {kept}" if kept else "")
          + (f"; {len(broken)} scenes failed: {broken}" if broken else ""))
    return 1 if broken or kept else 0


if __name__ == "__main__":
    sys.exit(main())
