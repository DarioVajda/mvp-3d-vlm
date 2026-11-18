"""Collect every scene's results into four flat CSV tables for analysis: tasks/openvocab_adapter/results/data/*.csv.

  python src/mvp/openvocab_adapter/export_tables.py [--results tasks/openvocab_adapter/results]

One row per object, camera, comparison criterion and SAM 3 frame, each tagged with dataset, scene, frame selection
and run (adapted = the saved VGGT outputs, native = OpenVocab-4D's own VGGT). Needs only the standard library; the
column meanings are in tasks/openvocab_adapter/results/data/README.md.
"""
import argparse
import csv
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[3] / "tasks" / "openvocab_adapter" / "results"
RUNS = ("adapted", "native")


def scene_dirs(results):
    for path in sorted(results.iterdir()):
        if path.is_dir() and "__" in path.name:
            dataset, scene = path.name.split("__", 1)
            yield dataset, scene, path


def selections(scene_dir):
    return sorted(p for p in scene_dir.iterdir() if p.is_dir())


def objects(results):
    rows = []
    for dataset, scene, sd in scene_dirs(results):
        for sel in selections(sd):
            for run in RUNS:
                path = sel / run / "objects.csv"
                if not path.is_file():
                    continue
                with path.open() as f:
                    for row in csv.DictReader(f):
                        frames = [k for k, v in row.items() if k.startswith("frame_") and v == "1"]
                        out = {"dataset": dataset, "scene": scene, "selection": sel.name, "run": run}
                        out.update({k: v for k, v in row.items() if not k.startswith("frame_")})
                        out["frames_seen"] = ";".join(str(int(k[len("frame_"):])) for k in frames)
                        rows.append(out)
    return rows


def cameras(results):
    rows = []
    for dataset, scene, sd in scene_dirs(results):
        for sel in selections(sd):
            report = sel / "adapted" / "adapter_report.json"
            source = {}
            if report.is_file():  # export frame (0-127) -> frame index in the original video
                r = json.loads(report.read_text())
                source = dict(zip(r["selected_frames"], r["source_video_frames"]))
            for run in RUNS:
                path = sel / run / "cameras.csv"
                if not path.is_file():
                    continue
                with path.open() as f:
                    for row in csv.DictReader(f):
                        image = Path(row["image"]).name  # the frame file; the folder is a local work path
                        export = int(Path(image).stem)
                        rows.append({"dataset": dataset, "scene": scene, "selection": sel.name, "run": run,
                                     "frame": row["frame"], "export_frame": export, "video_frame": source.get(export, ""),
                                     **{k: v for k, v in row.items() if k not in ("frame", "image")}})
    return rows


def comparisons(results):
    rows = []
    for dataset, scene, sd in scene_dirs(results):
        for sel in selections(sd):
            path = sel / "comparison.json"
            if not path.is_file():
                continue
            report = json.loads(path.read_text())
            base = {"dataset": dataset, "scene": scene, "selection": sel.name, "verdict": report["verdict"]}
            for c in report["criteria"]:
                rows.append({**base, "kind": "criterion", "name": c["criterion"], "status": {True: "PASS", False: "FAIL", None: "n/a"}[c["ok"]],
                             "value": c["value"], "limit": c["limit"], "detail": ""})
            for c in report["comparisons"]:
                values = c.get("values") or {}
                if not values:
                    rows.append({**base, "kind": "check", "name": c["comparison"], "status": c["status"],
                                 "value": "", "limit": "", "detail": c.get("detail", "")})
                for key, value in values.items():
                    rows.append({**base, "kind": "measure", "name": f"{c['comparison']}: {key}", "status": c["status"],
                                 "value": json.dumps(value) if isinstance(value, (list, dict)) else value,
                                 "limit": "", "detail": c.get("detail", "")})
    return rows


def detections(results):
    rows = []
    for dataset, scene, sd in scene_dirs(results):
        path = sd / "masks_manifest.json"
        if not path.is_file():
            continue
        manifest = json.loads(path.read_text())
        for frame in manifest["frames"]:
            for prompt, count in frame["detections"].items():
                rows.append({"dataset": dataset, "scene": scene, "frame": int(frame["stem"]), "prompt": prompt,
                             "detections": count, "min_score": manifest["min_score"],
                             "model": manifest.get("model", "")})
    return rows


def write(path, rows):
    if not rows:
        print(f"[tables] {path.name}: no rows")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[tables] {path}: {len(rows)} rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=RESULTS)
    args = ap.parse_args()
    out = args.results / "data"
    out.mkdir(exist_ok=True)
    for name, build in (("objects", objects), ("cameras", cameras), ("comparisons", comparisons),
                        ("detections", detections)):
        write(out / f"{name}.csv", build(args.results))


if __name__ == "__main__":
    main()
