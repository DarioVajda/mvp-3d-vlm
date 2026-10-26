"""Package the saved VGGT-1B reconstructions of a subset of VSI-Bench scenes for sharing (task tasks/vggt_data.md).

Everything comes from data/vggt/ (made by mvp.reconstruction.extract) and the VSI-Bench videos; no model is run.
Per scene, out/scenes/<dataset>/<scene>/:
  frames/000.jpg ...   the 128 frames VGGT saw, at its input resolution (same resize / crop as extract.py)
  cameras.npz          extrinsic [T,3,4] (OpenCV camera-from-world), intrinsic [T,3,3], camera_center [T,3],
                       camera_forward [T,3], frame_indices [T], timestamps_s [T]
  patch_points.npz     xyz [T,gh,gw,3]: per 14x14 patch, median of the depth map unprojected with the predicted
                       cameras; conf [T,gh,gw]: mean depth confidence of the patch's pixels
  point_cloud.ply      fused cloud (the saved cloud.npz) as binary PLY with colours and source frame
  point_cloud.npz      the same cloud as arrays: xyz, rgb, frame
  questions.jsonl      every VSI-Bench question of the scene, with Qwen3-VL 2B / 4B zero-shot answers
  meta.json            video / preprocessing info, frame subsets (uniform and space-aware 16 / 32)
  preview.jpg          check figure from mvp.reconstruction.show

  .venv/bin/python tasks/vggt_data/export.py [--scenes arkitscenes/41069025 ...] [--out tasks/vggt_data/out]
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from decord import VideoReader, cpu
from PIL import Image

from mvp.benchmarks.vsibench import load_vsibench
from mvp.paths import ROOT, VGGT_DIR, ZEROSHOT
from mvp.reconstruction.sa_sampling import uniform_subset

TARGET, PATCH = 518, 14
# Scenes per dataset: ARKitScenes is split into landscape and portrait (37x28 patch grid instead of 28x37)
QUOTA = {("arkitscenes", False): 25, ("arkitscenes", True): 5, ("scannet", False): 25, ("scannetpp", False): 20}
PREDICTIONS = {  # zero-shot runs on the full benchmark, 32 frames sampled by qwen_vl_utils over the whole video
    "qwen3vl_2b_f32": ZEROSHOT / "vsibench/qwen_sampling/2B_f32.jsonl",
    "qwen3vl_4b_f32": ZEROSHOT / "vsibench/qwen_sampling/4B_f32.jsonl",
}


def scene_meta(dataset, scene):
    return json.loads((VGGT_DIR / dataset / scene / "meta.json").read_text())


def select_scenes(df):
    """Per dataset (and orientation), the scenes with the most question types, then the most questions."""
    g = df.groupby(["dataset", "scene_name"]).agg(n=("id", "size"), types=("question_type", "nunique")).reset_index()
    g["portrait"] = [(lambda hw: hw[0] > hw[1])(scene_meta(d, s)["input_hw"]) for d, s in zip(g.dataset, g.scene_name)]
    picks = []
    for (dataset, portrait), k in QUOTA.items():
        sub = g[(g.dataset == dataset) & (g.portrait == portrait)]
        picks += [f"{d}/{s}" for d, s in sub.sort_values(["types", "n", "scene_name"], ascending=[False, False, True])
                  .head(k)[["dataset", "scene_name"]].itertuples(index=False)]
    return picks


def resize_like_vggt(frame, input_hw):
    """extract.preprocess (portrait="keep") on one uint8 frame, kept as uint8 instead of a tensor."""
    img = Image.fromarray(frame).convert("RGB")
    w, h = img.size
    if h > w:
        new_w = round(w * (TARGET / h) / PATCH) * PATCH
        img = img.resize((new_w, TARGET), Image.Resampling.BICUBIC)
    else:
        new_h = round(h * (TARGET / w) / PATCH) * PATCH
        img = img.resize((TARGET, new_h), Image.Resampling.BICUBIC)
        if new_h > TARGET:
            y = (new_h - TARGET) // 2
            img = img.crop((0, y, TARGET, y + TARGET))
    assert [img.size[1], img.size[0]] == input_hw, (img.size, input_hw)
    return img


def unproject(depth, extrinsic, intrinsic):
    """VGGT's unproject_depth_map_to_point_map in numpy: pixel (u, v) at integer coordinates, world = R^T (x - t)."""
    T, H, W = depth.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    fx, fy = intrinsic[:, 0, 0, None, None], intrinsic[:, 1, 1, None, None]
    cx, cy = intrinsic[:, 0, 2, None, None], intrinsic[:, 1, 2, None, None]
    cam = np.stack([(u - cx) * depth / fx, (v - cy) * depth / fy, depth], -1)  # [T,H,W,3]
    R, t = extrinsic[:, :, :3], extrinsic[:, :, 3]
    return np.einsum("tji,thwj->thwi", R, cam - t[:, None, None, :])


def patch_points(src, meta):
    depth = np.load(src / "depth.npy").astype(np.float32)
    conf = np.load(src / "depth_conf.npy").astype(np.float32)
    ext, intr = np.load(src / "extrinsic.npy"), np.load(src / "intrinsic.npy")
    gh, gw = meta["patch_grid_hw"]
    pts = unproject(depth, ext, intr)
    T = len(depth)
    xyz = np.median(pts.reshape(T, gh, PATCH, gw, PATCH, 3), axis=(2, 4)).astype(np.float32)
    pconf = conf.reshape(T, gh, PATCH, gw, PATCH).mean(axis=(2, 4)).astype(np.float32)
    return xyz, pconf, ext, intr


def write_ply(path, xyz, rgb, frame):
    rec = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"),
                                    ("blue", "u1"), ("frame", "<i2")])
    rec["x"], rec["y"], rec["z"] = xyz.T
    rec["red"], rec["green"], rec["blue"] = rgb.T
    rec["frame"] = frame
    header = ("ply\nformat binary_little_endian 1.0\ncomment VGGT-1B fused cloud, world = first camera (OpenCV), "
              f"arbitrary scale\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty short frame\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(rec.tobytes())


def frame_subsets(src, T):
    sa = json.loads((src / "sa_order.json").read_text())["selection_order"]  # greedy order, may stop before 32
    out = {}
    for k in (16, 32):
        out[f"uniform{k}"] = [int(i) for i in uniform_subset(T, k)]
        out[f"space_aware{k}"] = sorted(int(i) for i in sa[:k])
    return out


def export_scene(name, questions, out_root):
    dataset, scene = name.split("/")
    src = VGGT_DIR / dataset / scene
    meta = scene_meta(dataset, scene)
    dst = out_root / "scenes" / dataset / scene
    tmp = dst.with_name(scene + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "frames").mkdir(parents=True)

    idx = np.load(src / "frame_indices.npy")
    vr = VideoReader(str(ROOT / meta["video"]), ctx=cpu(0))
    assert len(vr) == meta["num_video_frames"]
    for i, f in enumerate(vr.get_batch(idx.tolist()).asnumpy()):
        resize_like_vggt(f, meta["input_hw"]).save(tmp / "frames" / f"{i:03d}.jpg", quality=92)

    xyz, pconf, ext, intr = patch_points(src, meta)
    np.savez(tmp / "patch_points.npz", xyz=xyz, conf=pconf)
    R, t = ext[:, :, :3], ext[:, :, 3]
    np.savez(tmp / "cameras.npz", extrinsic=ext, intrinsic=intr,
             camera_center=np.einsum("tji,tj->ti", R, -t).astype(np.float32),
             camera_forward=R[:, 2, :].astype(np.float32),  # optical axis (+z of the camera) in world coordinates
             frame_indices=idx, timestamps_s=(idx / meta["video_fps"]).astype(np.float32))

    cloud = np.load(src / "cloud.npz")
    np.savez(tmp / "point_cloud.npz", xyz=cloud["xyz"], rgb=cloud["rgb"], frame=cloud["frame"])
    write_ply(tmp / "point_cloud.ply", cloud["xyz"], cloud["rgb"], cloud["frame"])

    with open(tmp / "questions.jsonl", "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")

    gh, gw = meta["patch_grid_hw"]
    info = {
        "dataset": dataset, "scene": scene,
        "source_video": f"VSI-Bench {dataset}/{scene}.mp4",
        "video_num_frames": meta["num_video_frames"], "video_fps": meta["video_fps"], "video_hw": meta["video_hw"],
        "num_frames": len(idx), "image_hw": meta["input_hw"], "patch_size": PATCH, "patch_grid_hw": [gh, gw],
        "portrait": meta["input_hw"][0] > meta["input_hw"][1],
        "model": "facebook/VGGT-1B, one joint forward pass over all frames, bf16 autocast",
        "world_frame": "camera of frame 0, OpenCV axes (x right, y down, z forward); scale arbitrary, not metres",
        "cloud_voxel_size": float(cloud["voxel_size"]), "cloud_points": int(len(cloud["xyz"])),
        "frame_subsets": frame_subsets(src, len(idx)),
        "num_questions": len(questions),
    }
    (tmp / "meta.json").write_text(json.dumps(info, indent=1))

    subprocess.run([sys.executable, "-m", "mvp.reconstruction.show", name, "--out", str(tmp / "preview.jpg")],
                   check=True, capture_output=True, cwd=ROOT)
    shutil.rmtree(dst, ignore_errors=True)
    tmp.rename(dst)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", help="<dataset>/<scene> (default: the selection from QUOTA)")
    ap.add_argument("--out", default=str(Path(__file__).parent / "out"))
    args = ap.parse_args()
    out_root = Path(args.out)

    df = load_vsibench("full")
    for col, path in PREDICTIONS.items():
        p = pd.read_json(path, lines=True).set_index("id")
        df[col] = [{"prediction": p.at[i, "prediction"], "score": float(p.at[i, "score"])} for i in df["id"]]
    scenes = args.scenes or select_scenes(df)

    rows = []
    for k, name in enumerate(scenes):
        dataset, scene = name.split("/")
        qs = df[(df.dataset == dataset) & (df.scene_name == scene)].drop(columns=["video_path"])
        info = export_scene(name, qs.to_dict(orient="records"), out_root)
        rows.append({"dataset": dataset, "scene": scene, "portrait": info["portrait"], "num_frames": info["num_frames"],
                     "num_questions": info["num_questions"],
                     "question_types": ";".join(sorted(qs.question_type.unique())),
                     "cloud_points": info["cloud_points"]})
        print(f"[{k + 1}/{len(scenes)}] {name}: {info['num_questions']} questions", flush=True)

    index = pd.DataFrame(rows)
    old = out_root / "scenes.csv"
    if args.scenes and old.exists():  # partial rerun: update those rows only
        prev = pd.read_csv(old, dtype={"scene": str})
        index = pd.concat([prev[~(prev.dataset + "/" + prev.scene).isin(scenes)], index])
    index.sort_values(["dataset", "scene"]).to_csv(old, index=False)


if __name__ == "__main__":
    main()
