compute and send the following data to Nicolas and Tamara:
1. reconstructed per-patch + camera 3d positions from VGGT
2. reconstructed point clouds for the video scenes
3. the videos (only the sampled frames)
4. corresponding examples from the datasets

This should be done for say 50-100ish videos.

Upload all of this to Google Drive.

---

# VGGT reconstructions of 75 VSI-Bench scenes

**Status:** data exported, checked and shared on Google Drive (2026-10-26).

This covers 75 indoor videos from [VSI-Bench](https://arxiv.org/abs/2412.14171), a video spatial-reasoning
benchmark. For each video it gives:

- 128 frames;
- their camera poses and per-patch 3D points from [VGGT-1B](https://github.com/facebookresearch/vggt), a
  feed-forward 3D reconstruction model;
- a fused, coloured point cloud;
- every VSI-Bench question about the scene, with its answer and the zero-shot answers of Qwen3-VL-2B/4B.

No model was run for this export. VGGT was run once on all 288 VSI-Bench videos (`src/mvp/reconstruction/extract.py`),
and this task only repackages 75 of them.

## Download

**Link:** <https://drive.google.com/open?id=1H7k60Sw2N0nmwLkWsifWNcxgPbgAsG-i> (anyone with the link can view)

The download is one zip: `vggt_vsibench_75.zip`, 1.0 GB, 1.1 GB unzipped.

- **In a browser:** open the link and click Download. Drive says it can't scan files this large for viruses;
  click "Download anyway".
- **From a terminal**, for example on a remote machine:

  ```bash
  curl -L -o vggt_vsibench_75.zip \
      "https://drive.usercontent.google.com/download?id=1H7k60Sw2N0nmwLkWsifWNcxgPbgAsG-i&export=download&confirm=t"
  # or: pip install gdown && gdown 1H7k60Sw2N0nmwLkWsifWNcxgPbgAsG-i
  ```

Check the download and unzip it:

```bash
sha256sum vggt_vsibench_75.zip
# expected: e31ddb04b42d845361f373360cb4b81dde622c3fdecf83c31553b7b686ac41ab
unzip -q vggt_vsibench_75.zip          # creates vggt_vsibench_75/
cd vggt_vsibench_75
pip install numpy pillow matplotlib    # all that load_example.py needs
python load_example.py scenes/scannet/scene0025_01 --frame 40 --out example.png
```

The tree below starts at `vggt_vsibench_75/`, which is `out/` in this repo. The zip also has a `README.md`: a copy of
this file from before the download section was added.

## What's in it

```
out/                                      1.1 GB, 75 scenes
├── scenes.csv                            one row per scene: dataset, scene, portrait, num_questions, question types
├── load_example.py                       loads a scene and plots it (numpy, pillow, matplotlib)
└── scenes/<dataset>/<scene>/             ~15 MB each
    ├── frames/000.jpg … 127.jpg          the frames VGGT saw, at its input resolution
    ├── cameras.npz                       camera poses
    ├── patch_points.npz                  one 3D point per 14×14 patch per frame
    ├── point_cloud.ply / .npz            fused coloured point cloud (same data, two formats)
    ├── questions.jsonl                   VSI-Bench questions, answers and Qwen3-VL answers
    ├── meta.json                         video info, image size, patch grid, frame subsets
    └── preview.jpg                       check figure: frames, depth, confidence, top and front views
```

### Scenes

| Source | Scenes | Frame size (H×W) | Patch grid (rows×cols) |
|---|---|---|---|
| ARKitScenes, landscape | 25 | 392×518 | 28×37 |
| ARKitScenes, portrait | 5 | 518×392 | 37×28 |
| ScanNet | 25 | 392×518 | 28×37 |
| ScanNet++ | 20 | 392×518 | 28×37 |

**How they were chosen.** Within each group, the scenes with the most distinct question types come first, with ties
broken by number of questions. So the set leans towards larger scenes with many questions. All 10 VSI-Bench question
types are covered: every scene has `object_abs_distance`, `object_size_estimation` and `room_size_estimation`; the others are in 54–71 scenes, except `obj_appearance_order` (24 scenes: VSI-Bench has none for ARKitScenes). The 75 scenes have 2,728 questions in total.

### Frames

There are 128 frames per video, evenly spaced over the whole video (`np.linspace(0, N-1, 128, dtype=int)`).
They were resized exactly as VGGT's input:

- landscape: width 518, then a bicubic resize (640×480 videos become 518×392);
- portrait: kept upright, with the long side at 518.

Pixel (u, v) of `frames/<t>.jpg` is pixel (u, v) of VGGT's depth map and patch grid. The original videos are not
included.

### Coordinates (read this first)

- **World frame** = the camera of frame 0, OpenCV axes: x right, y down, z forward. So "up" is roughly −y if the first
  frame was held level, and the x–z plane is roughly the floor plane.
- **Scale is arbitrary, not metres.** VGGT normalises each scene (one unit is roughly a few metres) and the scale
  differs between scenes. VSI-Bench's metric answers (distances, sizes) cannot be read off directly without a
  per-scene scale.
- Everything in one scene (cameras, patch points, cloud) shares the same world frame and scale.

### `cameras.npz`

| Key | Shape | Meaning |
|---|---|---|
| `extrinsic` | [128, 3, 4] float32 | camera-from-world `[R \| t]` (OpenCV): `x_cam = R @ x_world + t` |
| `intrinsic` | [128, 3, 3] float32 | pinhole K in pixels of the frame JPEGs (no skew; principal point ≈ image centre) |
| `camera_center` | [128, 3] float32 | camera position in world, `−Rᵀ t` |
| `camera_forward` | [128, 3] float32 | optical axis (camera +z) in world, `R[2]` |
| `frame_indices` | [128] int | index of each frame in the original video |
| `timestamps_s` | [128] float32 | time of each frame in the original video, in seconds |

### `patch_points.npz`

| Key | Shape | Meaning |
|---|---|---|
| `xyz` | [128, gh, gw, 3] float32 | 3D point of each 14×14 patch, world frame |
| `conf` | [128, gh, gw] float32 | mean VGGT depth confidence over the patch's pixels (≥ 1; higher is better) |

Patch `(r, c)` of frame `t` covers pixels `[14r, 14r+14) × [14c, 14c+14)` of `frames/<t>.jpg`. This is the same
patch grid VGGT's transformer uses, so `xyz[t, r, c]` is the 3D position of VGGT token `r * gw + c` of frame `t`.

**How it is computed:**

1. Unproject VGGT's depth map with its predicted cameras. VGGT's authors recommend this over the point-map head.
2. Take the per-coordinate median over the patch's 196 pixels.

Every patch has a point, including sky through windows, mirrors and depth edges, so use `conf` to filter. Confidence
is relative within a scene, not comparable across scenes. Keeping patches above the scene's median is a reasonable
default.

### `point_cloud.ply` / `point_cloud.npz`

This is the fused cloud of the whole scene, about 280k points:

1. depth from all 128 frames, unprojected;
2. kept where depth confidence is in the scene's top half;
3. voxel-downsampled to one point per voxel (`meta.json: cloud_voxel_size`).

The `.ply` opens in MeshLab, CloudCompare or Open3D, and holds xyz, colour and a per-point `frame` property. The
`.npz` holds the same data: `xyz` [N, 3] float32, `rgb` [N, 3] uint8, `frame` [N] int16 (the source frame of each point).

### `questions.jsonl`

There is one JSON object per line: every VSI-Bench question about this scene, from the full 5,130-question set.

```json
{"id": 3137, "dataset": "scannet", "scene_name": "scene0025_01", "question_type": "object_rel_distance",
 "question": "Measuring from the closest point of each object, which of these objects (table, chair, door, telephone) is the closest to the fan?",
 "options": ["A. table", "B. chair", "C. door", "D. telephone"], "ground_truth": "C",
 "qwen3vl_2b_f32": {"prediction": "C. door", "score": 1.0}, "qwen3vl_4b_f32": {"prediction": "C. door", "score": 1.0}}
```

- **Question types:**
  - multiple choice (`options` is a list; scored by exact match on the letter): `object_rel_direction_{easy,medium,hard}`, `object_rel_distance`, `route_planning`, `obj_appearance_order`;
  - numeric (`options` is null; scored by Mean Relative Accuracy, 0–1): `object_counting`, `object_abs_distance` (m), `object_size_estimation` (cm), `room_size_estimation` (m²).
- **`qwen3vl_*_f32`:** zero-shot answers of Qwen3-VL-2B/4B-Instruct with 32 frames sampled uniformly over the whole
  video by `qwen_vl_utils`. These are *not* the 128 frames here. `score` is 1/0 for multiple choice and MRA for
  numeric questions. They are useful for picking questions the models get wrong.

### `meta.json`

`meta.json` holds:

- the source video's frame count, fps and resolution;
- `image_hw`, `patch_grid_hw` and `portrait`;
- the cloud's voxel size and point count;
- `frame_subsets`: indices into the 128 frames for the subsets our experiments feed to Qwen:
  - `uniform16`, `uniform32`: `np.linspace(0, 127, k, dtype=int)`;
  - `space_aware16`, `space_aware32`: Spatial-MLLM's greedy voxel-coverage selection, in temporal order. These can
    have fewer than k frames when coverage saturates.

## Quick start

```python
import json, numpy as np
from PIL import Image

d = "scenes/scannet/scene0025_01"
meta = json.load(open(f"{d}/meta.json"))
cams = np.load(f"{d}/cameras.npz")
pts = np.load(f"{d}/patch_points.npz")
cloud = np.load(f"{d}/point_cloud.npz")
qs = [json.loads(l) for l in open(f"{d}/questions.jsonl")]

t = 40
img = Image.open(f"{d}/frames/{t:03d}.jpg")              # 518×392
xyz, conf = pts["xyz"][t], pts["conf"][t]                # [28, 37, 3], [28, 37]
good = conf > np.median(pts["conf"])                     # drop low-confidence patches

# project a world point into frame t
R, tr, K = cams["extrinsic"][t, :, :3], cams["extrinsic"][t, :, 3], cams["intrinsic"][t]
uvw = K @ (R @ xyz[10, 20] + tr)
u, v = uvw[:2] / uvw[2]                                  # ≈ centre of patch (10, 20): (20*14+7, 10*14+7)
```

`python load_example.py scenes/scannet/scene0025_01 --frame 40` prints all the shapes and draws:

- the frame;
- a top-down view of the cloud with the camera path and that frame's patch points.

## Checks

`check.py` reads only the exported files and checks every scene. Results over all 75 scenes:

- frame 0's camera is at the origin;
- every frame, subset and cloud index is in range;
- the patch points of every 8th frame project back into their own 14×14 patch: 100% in every scene;
- cloud points project onto their source frame's JPEG with matching colour, mean error 0.8–1.5/255 per scene. This shows the
  frames are exactly the images VGGT saw.

## Caveats

- VGGT is a learned model, so the reconstructions are approximate:
  - the world scale is arbitrary;
  - the poses drift on long or textureless videos;
  - glass and mirrors give wrong depth.

  Look at `preview.jpg` before relying on a scene.
- The frames come from ScanNet, ScanNet++ and ARKitScenes, through VSI-Bench. Their terms of use apply; ScanNet and
  ScanNet++ need a signed agreement. Please don't redistribute the data further.

## Reproducing

```bash
# from mvp/; CPU only, 7 min for the 75 scenes on 8 CPUs
LOGDIR=vggt_data slurm/submit.sh -J vggt_export -t 01:00:00 --gres gpu:0 -c 8 --mem 32G -- python tasks/vggt_data/export.py
.venv/bin/python tasks/vggt_data/check.py            # consistency checks
```

The script is `tasks/vggt_data/export.py`, which reads `data/vggt/` and `data/raw/VSI-Bench/`. `--scenes
<dataset>/<scene> ...` re-exports only those scenes. The output is in `tasks/vggt_data/out/` (gitignored).
