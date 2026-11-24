"""pipeline_v3's sheets again with each row's box from the 10th-90th percentile of its points (size_trim 10) instead
of their full extent, and nothing else changed: the same tracks, the same rows and centres. relations_v4.py measures
distances between these boxes.

  1. a copy of the run folder with size_trim 10 in every sheet's question.json
  2. merge_points.py --no-merge --min-single-frame-score, as pipeline_v3.py step 1
  3. each scene's scale_rows.json written from --unmerged (pipeline_v3's own unmerged sheets): the point merge converts
     its threshold to sheet units with the priors scale of the rows it is given, and trimmed rows would change it, and
     with it which tracks merge
  4. the point merge (merge_points.py --threshold), as pipeline_v3.py step 3, with --code's tree given this
     repository's mvp/benchmarks/scene_scale.py (the same constants; the sizing commit's predates SCALE_ROWS, so it
     would ignore step 3)

  PYTHONPATH=src python experiments/2026-11-18-scene-sheet-formats/trimmed_sheets.py --run <run folder> \
      --unmerged <pipeline_v3's _unmerged sheets> --scenes <scene list> --code <src tree at the sizing commit> --out <new folder>
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rerun_detection as rd  # noqa: E402

SETTINGS = rd.FROZEN["pipeline_v3"]
SIZE_TRIM = 10.0  # percent at each end, per axis (scene_sheet_creator.build_scene_sheet reads it as a percentile)


def run(cmd):
    print("$", " ".join(map(str, cmd)), flush=True)
    status = subprocess.run(list(map(str, cmd))).returncode
    if status:
        sys.exit(f"failed ({status}): {cmd[1]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the run folder with the tracks")
    ap.add_argument("--unmerged", required=True, help="pipeline_v3.py's <out>_unmerged sheets of the same run")
    ap.add_argument("--scenes", required=True, help="a file with one <dataset>/<scene> per line")
    ap.add_argument("--code", required=True, help="a src/ tree with scene_sheet_creator.py from the sizing commit")
    ap.add_argument("--dense-root", default=str(HERE.parents[1] / "data" / "vggt"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from mvp.benchmarks import scene_scale
    out = Path(args.out)
    trimmed_run, unmerged = out.parent / (out.name + "_run"), out.parent / (out.name + "_unmerged")
    scenes = [s for s in Path(args.scenes).read_text().split() if s]
    if trimmed_run.exists():
        shutil.rmtree(trimmed_run)
    shutil.copytree(args.run, trimmed_run)
    for f in trimmed_run.glob("*/subset-uniform16/sheets/*/question.json"):
        q = json.loads(f.read_text())
        q["size_trim"] = SIZE_TRIM
        f.write_text(json.dumps(q, indent=1))
    merge = [sys.executable, HERE / "merge_points.py", "--scenes", args.scenes, "--code", args.code,
             "--dense-root", args.dense_root]
    run(merge + ["--sheets", trimmed_run, "--out", unmerged, "--no-merge",
                 "--min-single-frame-score", SETTINGS["min_single_frame_score"]])
    for s in scenes:
        longest = scene_scale.scene_longest(str(args.unmerged), s)
        (unmerged / s.replace("/", "__", 1) / "subset-uniform16" / scene_scale.SCALE_ROWS).write_text(json.dumps(
            {"longest": longest, "from": "the untrimmed unmerged rows (trimmed_sheets.py --unmerged)"}, indent=1))
    code = out.parent / (out.name + ".src")
    if code.exists():
        shutil.rmtree(code)
    shutil.copytree(args.code, code)
    shutil.copy(HERE.parents[1] / "src/mvp/benchmarks/scene_scale.py", code / "mvp/benchmarks/scene_scale.py")
    merge[merge.index("--code") + 1] = code
    run(merge + ["--sheets", unmerged, "--out", out, "--threshold", SETTINGS["merge_threshold"]])
    for scratch in (trimmed_run, code, out.parent / (out.name + ".code"), unmerged.parent / (unmerged.name + ".code")):
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"trimmed sheets of {len(scenes)} scenes in {out}")


if __name__ == "__main__":
    main()
