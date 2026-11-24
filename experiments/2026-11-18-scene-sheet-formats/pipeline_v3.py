"""Make pipeline_v3's sheets from its tracks (a run folder of rerun_detection.ipynb, runs/pipeline_v3/), with the
settings frozen in rerun_detection.FROZEN["pipeline_v3"]:

  1. pipeline sheets from the tracks and the dense depth, sizes as scene_sheet_creator.py at the sizing commit,
     dropping rows of tracks seen in one frame whose best detection score is below the cutoff (merge_points.py
     --no-merge --min-single-frame-score)
  2. gold sheets made again (the fixed prompts rename some questions' objects)
  3. the point merge (merge_points.py), the scale from step 1's unmerged rows (scene_scale.SCALE_ROWS)

  PYTHONPATH=src python experiments/2026-11-18-scene-sheet-formats/pipeline_v3.py --run data/rerun/pipeline_v3 \
      --out data/rerun_sheets/pipeline_v3 --scenes <scene list> [--code <src tree at the sizing commit>]

--out gets the final sheets; <out>_unmerged keeps step 1's. Without --code, a src/ tree with scene_sheet_creator.py
from the sizing commit is made in a temporary folder (git archive of HEAD plus that one file). Needs the dense depth in
data/vggt and VSI-Bench's gold metadata (downloaded if missing).
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import rerun_detection as rd  # noqa: E402

SETTINGS = rd.FROZEN["pipeline_v3"]


def old_sizing_tree(tmp):
    """src/ at HEAD with scene_sheet_creator.py from the sizing commit, in tmp."""
    archive = subprocess.run(["git", "-C", str(REPO), "archive", "HEAD", "src"], capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(tmp)], input=archive, check=True)
    path = "src/mvp/openvocab_adapter/scene_sheet_creator.py"
    old = subprocess.run(["git", "-C", str(REPO), "show", f"{SETTINGS['sizing_commit']}:{path}"],
                         capture_output=True, check=True).stdout
    (Path(tmp) / path).write_bytes(old)
    return Path(tmp) / "src"


def run(cmd):
    print("$", " ".join(map(str, cmd)), flush=True)
    status = subprocess.run(list(map(str, cmd))).returncode
    if status:
        sys.exit(f"failed ({status}): {cmd[1] if len(cmd) > 1 else cmd[0]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the run folder with the tracks (runs/pipeline_v3/ from Drive)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", required=True, help="a file with one <dataset>/<scene> per line")
    ap.add_argument("--code", help="a src/ tree with scene_sheet_creator.py from the sizing commit")
    ap.add_argument("--dense-root", default=str(REPO / "data" / "vggt"))
    ap.add_argument("--size-merge", action="store_true",
                    help="not pipeline_v3: merge within each prompt's typical size (merge_points.py --size-threshold)")
    args = ap.parse_args()
    out, unmerged = Path(args.out), Path(args.out + "_unmerged")
    scenes = [s for s in Path(args.scenes).read_text().split() if s]
    with tempfile.TemporaryDirectory() as tmp:
        code = Path(args.code) if args.code else old_sizing_tree(tmp)
        merge = [sys.executable, HERE / "merge_points.py", "--scenes", args.scenes, "--code", code,
                 "--dense-root", args.dense_root]
        run(merge + ["--sheets", args.run, "--out", unmerged, "--no-merge",
                     "--min-single-frame-score", SETTINGS["min_single_frame_score"]])
        run([sys.executable, REPO / "src/mvp/openvocab_adapter/rebuild_scene_sheets.py", "--root", unmerged,
             "--source", "gold", "--scenes", *[s.replace("/", "__", 1) for s in scenes]])
        run(merge + ["--sheets", unmerged, "--out", out, "--threshold", SETTINGS["merge_threshold"]]
            + (["--size-threshold"] if args.size_merge else []))
    for scratch in (out.parent / (out.name + ".code"), unmerged.parent / (unmerged.name + ".code")):
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"pipeline_v3 sheets of {len(scenes)} scenes in {out} (unmerged: {unmerged})")


if __name__ == "__main__":
    main()
