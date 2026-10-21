"""Run VGGT-1B on every VSI-Bench video and save the 3D reconstruction and aggregator features.

Frame selection and preprocessing follow Spatial-MLLM's space-aware sampling script
(third_party/Spatial-MLLM/src/sampling/sa_sampling.py), so the saved world points / confidences are exactly
the inputs its greedy voxel-coverage sampler needs:
  * min(128, N) frames, uniformly spaced with np.linspace(0, N-1, dtype=int), decoded with decord;
  * frames resized to width 518 and height round(h*518/w/14)*14 (bicubic), centre-cropped to 518 if taller
    (640x480 videos -> 518x392, a 28x37 patch grid);
  * portrait frames: Spatial-MLLM rotates them by -90 degrees to landscape (--portrait rotate). The 57 portrait
    ARKitScenes videos are upright, so this turns them sideways and VGGT fails on them (median confidence at its
    floor of 1.0). The default (--portrait keep) leaves them upright and scales the long side to 518 instead
    (480x640 -> 392x518, 37x28 patches, no crop). The rotated outputs are kept in data/vggt_portrait_rotated/.
  * one joint VGGT forward pass over all frames under bf16 autocast.

Output per video, data/vggt/<dataset>/<scene>/ (one .npy per array, so large arrays can be memory-mapped):
  meta.json            video / preprocessing info and array shapes
  frame_indices.npy    int32 [T]          indices into the original video
  extrinsic.npy        float32 [T,3,4]    OpenCV camera-from-world, world = first frame's camera
  intrinsic.npy        float32 [T,3,3]    in pixels of the 518-wide input
  pose_enc.npy         float32 [T,9]      raw camera head output
  depth.npy            float16 [T,H,W]    depth (up to the scene's global scale)
  depth_conf.npy       float16 [T,H,W]
  world_points.npy     float16 [T,H,W,3]  point head output, world frame
  world_points_conf.npy float16 [T,H,W]
  tokens.npy           float16 [T,P,2048] last aggregator layer, patch tokens (frame || global attention);
                                          the layer Spatial-MLLM feeds its connector (spatial_embeds_layer_idx=-1)
  special_tokens.npy   float16 [T,5,2048] same layer, camera token + 4 register tokens
  cloud.npz            fused point cloud from depth + cameras: xyz float32, rgb uint8, frame int16,
                       confidence-filtered and voxel-downsampled for viewing

  slurm/submit.sh -J vggt -t 04:00:00 -- python src/extract_vggt.py --shard 0 --num-shards 4
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from torchvision import transforms as TF

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from vsibench import ROOT, load_vsibench

TARGET = 518
PATCH = 14


def load_frames(path, max_frames):
    vr = VideoReader(str(path), ctx=cpu(0))
    n = len(vr)
    idx = np.linspace(0, n - 1, num=min(max_frames, n), dtype=int)
    frames = vr.get_batch(idx.tolist()).asnumpy()  # [T,H,W,3] uint8
    return frames, idx, n, float(vr.get_avg_fps())


def preprocess(frames, portrait="keep"):
    """Spatial-MLLM's load_and_preprocess_images, applied to decoded frames instead of PNG files.

    portrait="rotate" is Spatial-MLLM's handling of portrait frames; "keep" leaves them upright (long side 518).
    """
    to_tensor = TF.ToTensor()
    out, rotated = [], False
    for f in frames:
        img = Image.fromarray(f).convert("RGB")
        w, h = img.size
        if h > w and portrait == "keep":
            new_w = round(w * (TARGET / h) / PATCH) * PATCH
            out.append(to_tensor(img.resize((new_w, TARGET), Image.Resampling.BICUBIC)))
            continue
        if h > w:
            img, rotated = img.rotate(-90, expand=True), True
            w, h = img.size
        new_h = round(h * (TARGET / w) / PATCH) * PATCH
        img = to_tensor(img.resize((TARGET, new_h), Image.Resampling.BICUBIC))
        if new_h > TARGET:
            y = (new_h - TARGET) // 2
            img = img[:, y:y + TARGET, :]
        out.append(img)
    # All frames of a video share one size, so Spatial-MLLM's pad-to-max step is a no-op here.
    assert len({t.shape for t in out}) == 1
    return torch.stack(out), rotated


@torch.no_grad()
def run_vggt(model, images, dtype=torch.bfloat16):
    captured = {}

    def hook(_module, _inputs, output):
        tokens_list, ps_idx = output
        captured["last"], captured["ps_idx"] = tokens_list[-1][0].float().cpu(), ps_idx

    handle = model.aggregator.register_forward_hook(hook)
    try:
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            pred = model(images)
    finally:
        handle.remove()
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    out = {k: pred[k][0].float().cpu().numpy() for k in ["pose_enc", "depth", "depth_conf", "world_points", "world_points_conf"]}
    out["depth"] = out["depth"][..., 0]
    out["extrinsic"] = extrinsic[0].float().cpu().numpy()
    out["intrinsic"] = intrinsic[0].float().cpu().numpy()
    ps = captured["ps_idx"]
    out["tokens"] = captured["last"][:, ps:].numpy()
    out["special_tokens"] = captured["last"][:, :ps].numpy()
    return out


def fused_cloud(out, images, conf_quantile=0.5, max_points=300_000):
    """Unproject depth with the predicted cameras (more accurate than the point head, per the VGGT README)."""
    pts = unproject_depth_map_to_point_map(out["depth"][..., None], out["extrinsic"], out["intrinsic"])  # [T,H,W,3]
    conf = out["depth_conf"]
    keep = (conf >= np.quantile(conf, conf_quantile)) & (conf > 0.1)
    rgb = (images.permute(0, 2, 3, 1).float().cpu().numpy() * 255).astype(np.uint8)
    frame = np.broadcast_to(np.arange(len(conf))[:, None, None], conf.shape)
    xyz, rgb, frame = pts[keep].astype(np.float32), rgb[keep], frame[keep].astype(np.int16)
    # Random subset first (millions of points), then voxel downsample to <= max_points, one point per voxel.
    sub = np.random.default_rng(0).permutation(len(xyz))[:20 * max_points]
    xyz, rgb, frame = xyz[sub], rgb[sub], frame[sub]
    lo = xyz.min(0)
    voxel = np.ptp(xyz, axis=0).max() / 400
    while True:
        v = np.floor((xyz - lo) / voxel).astype(np.int64)
        _, first = np.unique((v[:, 0] << 42) | (v[:, 1] << 21) | v[:, 2], return_index=True)
        if len(first) <= max_points:
            break
        voxel *= 1.25
    return {"xyz": xyz[first], "rgb": rgb[first], "frame": frame[first], "voxel_size": np.float32(voxel)}


def check_fp16(name, a):
    m = float(np.abs(a).max())
    if not np.isfinite(m) or m > 6e4:
        raise ValueError(f"{name} does not fit in float16 (max |x| = {m})")
    return a.astype(np.float16)


def extract(model, device, video_path, out_dir, max_frames, portrait):
    t0 = time.time()
    frames, idx, n, fps = load_frames(video_path, max_frames)
    images, rotated = preprocess(frames, portrait)
    images = images.to(device, dtype=torch.bfloat16)  # as in sa_sampling.py
    t1 = time.time()
    out = run_vggt(model, images)
    torch.cuda.synchronize()
    t2 = time.time()
    cloud = fused_cloud(out, images)

    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    arrays = {
        "frame_indices": idx.astype(np.int32),
        "extrinsic": out["extrinsic"].astype(np.float32),
        "intrinsic": out["intrinsic"].astype(np.float32),
        "pose_enc": out["pose_enc"].astype(np.float32),
    }
    for k in ["depth", "depth_conf", "world_points", "world_points_conf", "tokens", "special_tokens"]:
        arrays[k] = check_fp16(k, out[k])
    for k, a in arrays.items():
        np.save(tmp / f"{k}.npy", a)
    np.savez(tmp / "cloud.npz", **cloud)
    meta = {
        "video": str(Path(video_path).relative_to(ROOT)),
        "num_video_frames": int(n),
        "video_fps": fps,
        "video_hw": list(frames.shape[1:3]),
        "portrait": portrait,
        "rotated_to_landscape": rotated,
        "input_hw": list(images.shape[-2:]),
        "patch_grid_hw": [images.shape[-2] // PATCH, images.shape[-1] // PATCH],
        "num_frames": int(len(idx)),
        "model": "facebook/VGGT-1B",
        "dtype": "bfloat16 autocast",
        "conf_max": {k: float(out[k].max()) for k in ["depth_conf", "world_points_conf"]},
        "cloud_points": int(len(cloud["xyz"])),
        "shapes": {k: list(a.shape) for k, a in arrays.items()},
        "seconds": {"decode": round(t1 - t0, 1), "vggt": round(t2 - t1, 1), "total": round(time.time() - t0, 1)},
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
    }
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1))
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp.rename(out_dir)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=128)
    ap.add_argument("--out", default=str(ROOT / "data" / "vggt"))
    ap.add_argument("--videos", nargs="*", help="<dataset>/<scene> to process (default: all benchmark videos)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--portrait", choices=["keep", "rotate"], default="keep")
    args = ap.parse_args()

    df = load_vsibench("full")
    videos = sorted(set(zip(df["dataset"], df["scene_name"], df["video_path"])))
    if args.videos:
        videos = [v for v in videos if f"{v[0]}/{v[1]}" in set(args.videos)]
    videos = videos[args.shard::args.num_shards]
    print(f"shard {args.shard}/{args.num_shards}: {len(videos)} videos", flush=True)

    device = "cuda"
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    out_root = Path(args.out)
    for i, (dataset, scene, path) in enumerate(videos):
        out_dir = out_root / dataset / scene
        if (out_dir / "meta.json").exists() and not args.overwrite:
            continue
        torch.cuda.reset_peak_memory_stats()
        meta = extract(model, device, path, out_dir, args.max_frames, args.portrait)
        size = sum(f.stat().st_size for f in out_dir.iterdir()) / 2**20
        print(f"[{i + 1}/{len(videos)}] {dataset}/{scene}: {meta['num_frames']}/{meta['num_video_frames']} frames "
              f"{meta['input_hw']}, {meta['seconds']}, {meta['peak_gpu_gb']} GB, {size:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
