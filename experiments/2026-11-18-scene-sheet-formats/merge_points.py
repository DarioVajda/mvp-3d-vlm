"""Rebuild pipeline sheets with split tracks merged at the point level: same-class tracks are joined before their rows
are computed, so a merged object's centre and box come from the union of its tracks' 3D points.

  PYTHONPATH=src python experiments/2026-11-18-scene-sheet-formats/merge_points.py --sheets data/scene_sheets_75 \
      --out <new folder> --scenes dev_scenes.txt --code <src tree> [--threshold 0.5] [--cap 3] [--no-merge]

--code is the src/ tree whose scene_sheet_creator.py computes the rows (today's, or one checked out at 50fe543 for the
sizing every experiment so far used). It is copied to a scratch folder, collect_objects in the copy is wrapped by
merge_tracks below, and the copy's rebuild_scene_sheets.py rebuilds the listed scenes of a copy of --sheets in --out
(pipeline sheets only), from their saved tracks and the dense depth (--dense-root). The repository's code is not
changed.

merge_tracks: within one prompt, two tracks are joined if they were never seen in the same frame (two detections in one
frame are two objects) and their boxes (the 2nd-98th percentile span of their points, axis-aligned) overlap or their
centres (the median of their points) are within --threshold metres, the closest pair first, until none is left. The
joined track keeps both tracks' frames and points; its row is then computed from all of them as for any track. Metres
use the scene's priors scale (mvp.benchmarks.scene_scale, no gold) on --sheets' rows. With --cap, a pair is joined only
if the joined box's longest side stays within cap times the prompt's typical size (scene_scale.PRIORS); prompts with
no typical size are then not merged.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
PATCH = '''

# ---- added by merge_points.py: tracks merged at the point level when MERGE_POINTS_THRESHOLD is set ----
import os as _os
_collect_objects_unmerged = collect_objects


def collect_objects(scene, prompts, *args, **kwargs):
    objects = _collect_objects_unmerged(scene, prompts, *args, **kwargs)
    if "MERGE_POINTS_THRESHOLD" not in _os.environ:
        return objects
    import merge_points
    return merge_points.merge_tracks(objects, scene, prompts)
'''


def box(obj):
    pts = np.concatenate(obj["sample"])
    lo, hi = np.percentile(pts, [2, 98], axis=0)
    return np.median(pts, axis=0), lo, hi


def scene_scale_k(scene_dir):
    """Metres per unit for the scene of a rebuild's scene/ folder: the priors scale on MERGE_POINTS_K_ROOT's sheets."""
    from mvp.benchmarks import scene_scale
    name = Path(scene_dir).parents[1].name.replace("__", "/", 1)
    ratio, classes = scene_scale.prior_ratio(scene_scale.scene_longest(os.environ["MERGE_POINTS_K_ROOT"], name))
    return scene_scale.CALIBRATION * ratio if classes else scene_scale.FALLBACK


def merge_tracks(objects, scene_dir, prompts):
    """collect_objects' output with same-prompt tracks joined as the module docstring says."""
    from mvp.benchmarks import scene_scale
    threshold = float(os.environ["MERGE_POINTS_THRESHOLD"])
    cap = float(os.environ["MERGE_POINTS_CAP"]) if os.environ.get("MERGE_POINTS_CAP") else None
    k = scene_scale_k(scene_dir)
    out = {}
    for pi in sorted({pi for pi, _ in objects}):
        items = {key: obj for key, obj in objects.items() if key[0] == pi}
        limit = cap * scene_scale.PRIORS[prompts[pi]] if cap and prompts[pi] in scene_scale.PRIORS else None
        if cap and limit is None:
            out.update(items)
            continue
        while True:
            boxes = {key: box(obj) for key, obj in items.items()}
            best = None
            keys = sorted(items)
            for i, a in enumerate(keys):
                for b in keys[i + 1:]:
                    if set(items[a]["frames"]) & set(items[b]["frames"]):
                        continue
                    (ca, la, ha), (cb, lb, hb) = boxes[a], boxes[b]
                    d = k * float(np.linalg.norm(ca - cb))
                    if not (d <= threshold or (np.all(la <= hb) and np.all(lb <= ha))):
                        continue
                    if limit is not None:
                        pts = np.concatenate(items[a]["sample"] + items[b]["sample"])
                        lo, hi = np.percentile(pts, [2, 98], axis=0)
                        if k * float((hi - lo).max()) > limit:
                            continue
                    if best is None or d < best[0]:
                        best = (d, a, b)
            if best is None:
                break
            _, a, b = best
            joined = {"frames": {**items[a]["frames"], **items[b]["frames"]},
                      "sample": items[a]["sample"] + items[b]["sample"]}
            if "lo" in items[a]:  # the old code also keeps each track's full min / max
                joined["lo"] = np.minimum(items[a]["lo"], items[b]["lo"])
                joined["hi"] = np.maximum(items[a]["hi"], items[b]["hi"])
            del items[a], items[b]
            items[min(a, b)] = joined
        out.update(items)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheets", required=True, help="the sheets folder to copy (and to take the priors scale from)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", required=True, help="a file with one <dataset>/<scene> per line")
    ap.add_argument("--code", required=True, help="the src/ tree whose scene_sheet_creator.py computes the rows")
    ap.add_argument("--dense-root", default=str(Path(__file__).resolve().parents[2] / "data" / "vggt"))
    ap.add_argument("--threshold", type=float, default=0.5, help="metres between centres")
    ap.add_argument("--cap", type=float, default=None)
    ap.add_argument("--no-merge", action="store_true", help="rebuild only, as rebuild_scene_sheets.py does")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    code = out.parent / (out.name + ".code")
    for p in (out, code):
        if p.exists():
            shutil.rmtree(p)
    shutil.copytree(args.code, code / "src")
    with open(code / "src/mvp/openvocab_adapter/scene_sheet_creator.py", "a") as f:
        f.write(PATCH)
    shutil.copy(Path(__file__), code / "merge_points.py")
    scenes = [s.replace("/", "__") for s in Path(args.scenes).read_text().split() if s.strip()]
    out.mkdir(parents=True)
    shutil.copy(Path(args.sheets) / "index.csv", out / "index.csv")
    for s in scenes:
        shutil.copytree(Path(args.sheets) / s, out / s)
    env = {**os.environ, "PYTHONPATH": f"{code / 'src'}{os.pathsep}{code}", "MERGE_POINTS_K_ROOT": str(args.sheets)}
    if not args.no_merge:
        env["MERGE_POINTS_THRESHOLD"] = str(args.threshold)
        if args.cap:
            env["MERGE_POINTS_CAP"] = str(args.cap)
    cmd = [sys.executable, str(code / "src/mvp/openvocab_adapter/rebuild_scene_sheets.py"), "--root", str(out),
           "--source", "pipeline", "--dense-root", args.dense_root, "--workers", str(args.workers), "--scenes", *scenes]
    sys.exit(subprocess.run(cmd, env=env).returncode)


if __name__ == "__main__":
    main()
