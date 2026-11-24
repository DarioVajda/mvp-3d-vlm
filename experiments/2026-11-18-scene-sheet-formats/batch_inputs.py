"""rerun_detection.ipynb's inputs for scenes of a new VGGT export (vggt_vsibench_batch2.zip and its depth add-on), made
as for the first 75 scenes, in two steps.

  geometry (here, numpy only)   per scene: the export's files and dense depth unpacked (the depth as
                                <dense-root>/<dataset>/<scene>/depth.npy, depth_conf.npy, frame_indices.npy, where
                                pipeline_v3.py and relations_v4.py read it), then adapt_vggt.py --frames
                                subset:uniform16 --label subset-uniform16 --dense-dir into
                                <out>/<dataset>__<scene>/subset-uniform16/scene/ (frames, cameras.npz, depth.npz,
                                adapter_report.json), and the scene's questions.jsonl next to it
  questions (spaCy, NLTK)       per question: the SAM 3 prompts (question_scene_sheet.extract_prompts, as
                                scene_sheet_report.py takes them), each sheet folder's question.json (pipeline and gold,
                                the fields question_scene_sheet.py and scene_sheet_report.py write) and index.csv
                                (sheet_index.py's columns); then dev_scenes.txt and the scene list

  PYTHONPATH=src python experiments/2026-11-18-scene-sheet-formats/batch_inputs.py geometry --export <main zip> \
      --depth <depth zip> --scenes <scene list> --dense-root data/vggt --out <inputs folder>
  python experiments/2026-11-18-scene-sheet-formats/batch_inputs.py questions --scenes <scene list> --out <inputs folder>
"""
import argparse
import csv
import io
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ADAPTER = HERE.parents[1] / "src/mvp/openvocab_adapter"
LABEL = "subset-uniform16"
FRAMES = "subset:uniform16"
INDEX_FIELDS = ["id", "scene", "question_type", "question", "options", "ground_truth", "prompts",
                "pipeline_sheet", "gold_sheet", "pipeline_ok", "gold_ok"]


def member(z, scene, name):
    """The zip member <top>/scenes/<dataset>/<scene>/<name>."""
    hits = [n for n in z.namelist() if n.endswith(f"/scenes/{scene}/{name}")]
    assert len(hits) == 1, f"{scene}/{name}: {len(hits)} members"
    return hits[0]


def geometry(export_zip, depth_zip, scenes, dense_root, out):
    with zipfile.ZipFile(export_zip) as ez, zipfile.ZipFile(depth_zip) as dz, tempfile.TemporaryDirectory() as tmp:
        for i, scene in enumerate(scenes, 1):
            src = Path(tmp) / "scenes" / scene
            prefix = member(ez, scene, "meta.json")[: -len("meta.json")]
            for name in ez.namelist():
                if name.startswith(prefix) and not name.endswith("/"):
                    dest = src / name[len(prefix):]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(ez.read(name))
            dense = Path(dense_root) / scene
            dense.mkdir(parents=True, exist_ok=True)
            with np.load(io.BytesIO(dz.read(member(dz, scene, "depth.npz")))) as d:
                np.save(dense / "depth.npy", d["depth"].astype(np.float16))
                np.save(dense / "depth_conf.npy", d["conf"].astype(np.float16))
            with np.load(src / "cameras.npz") as c:
                np.save(dense / "frame_indices.npy", c["frame_indices"])
            scene_out = Path(out) / scene.replace("/", "__", 1) / LABEL / "scene"
            status = subprocess.run([sys.executable, str(ADAPTER / "adapt_vggt.py"), "--scene", scene, "--data-root", tmp,
                                     "--frames", FRAMES, "--label", LABEL, "--dense-dir", str(dense),
                                     "--out", str(scene_out), "--overwrite"], capture_output=True, text=True)
            if status.returncode:
                sys.exit(f"{scene}: adapt_vggt.py failed\n{status.stdout[-2000:]}\n{status.stderr[-2000:]}")
            shutil.copy(src / "questions.jsonl", scene_out.parents[1] / "questions.jsonl")
            shutil.rmtree(src)
            print(f"[{i}/{len(scenes)}] {scene}", flush=True)


def questions(scenes, out):
    sys.path.insert(0, str(ADAPTER))
    import question_scene_sheet as qss
    out = Path(out)
    index = []
    for scene in scenes:
        name = scene.replace("/", "__", 1)
        rows = [json.loads(line) for line in open(out / name / "questions.jsonl")]
        for row in sorted(rows, key=lambda r: int(r["id"])):
            prompts, mode = qss.extract_prompts(row["question"])
            folder = qss.folder_name(prompts)
            pipeline, gold = f"{name}/{LABEL}/sheets/{folder}", f"{name}/gold/sheets/{folder}"
            (out / pipeline).mkdir(parents=True, exist_ok=True)
            (out / pipeline / "question.json").write_text(json.dumps({
                "scene": scene, "question": row["question"], "prompts": prompts, "prompt_mode": mode, "frames": FRAMES,
                "tracking": qss.track_settings(True, "transformers", 0.5, 55.0, 300), "size": True,
                "relative_positions": False, "size_trim": 0.02, "geometry": f"{name}/{LABEL}/scene", "depth": "dense",
            }, indent=1))
            (out / gold).mkdir(parents=True, exist_ok=True)
            (out / gold / "question.json").write_text(json.dumps(
                {"scene": scene, "question": row["question"], "prompts": prompts, "prompt_mode": mode, "source": "gold"},
                indent=1))
            index.append({"id": row["id"], "scene": scene, "question_type": row["question_type"],
                          "question": row["question"], "options": " | ".join(row.get("options") or []),
                          "ground_truth": row["ground_truth"], "prompts": ", ".join(prompts),
                          "pipeline_sheet": pipeline, "gold_sheet": gold, "pipeline_ok": bool(prompts),
                          "gold_ok": bool(prompts)})
        print(f"{scene}: {len(rows)} questions", flush=True)
    with open(out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows(index)
    shutil.copy(HERE / "dev_scenes.txt", out / "dev_scenes.txt")
    print(f"{len(index)} questions, {sum(not r['pipeline_ok'] for r in index)} with no prompt -> {out / 'index.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["geometry", "questions"])
    ap.add_argument("--scenes", required=True, help="a file with one <dataset>/<scene> per line")
    ap.add_argument("--out", required=True, help="the inputs folder (rerun_inputs/)")
    ap.add_argument("--export", help="geometry: vggt_vsibench_batch2.zip")
    ap.add_argument("--depth", help="geometry: depth_vsibench_batch2.zip")
    ap.add_argument("--dense-root", default=str(HERE.parents[1] / "data" / "vggt"))
    args = ap.parse_args()
    scenes = [s for s in Path(args.scenes).read_text().split() if s]
    if args.step == "geometry":
        geometry(args.export, args.depth, scenes, args.dense_root, args.out)
        shutil.copy(args.scenes, Path(args.out) / Path(args.scenes).name)
    else:
        questions(scenes, args.out)


if __name__ == "__main__":
    main()
