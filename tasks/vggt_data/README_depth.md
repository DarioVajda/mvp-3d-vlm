# VGGT per-pixel depth for the shared VSI-Bench scenes

An add-on to a main export zip (`vggt_vsibench_75.zip`, or `vggt_vsibench_batch2.zip` for the second set): the same
75 scenes, with VGGT's full-resolution depth map and depth confidence for each of the 128 frames. Unzip it next to the
main export's folder; the folder layout matches.

```
depth_vsibench_75/ (or depth_vsibench_batch2/)
├── scenes.csv                         same as in the main export
└── scenes/<dataset>/<scene>/depth.npz
        depth  float16 [128, H, W]     depth along the camera's optical axis (camera z)
        conf   float16 [128, H, W]     VGGT depth confidence (>= 1; higher is better)
```

- **Same frames and pixels.** `depth[t, v, u]` is the depth of pixel (u, v) of `frames/<t>.jpg` in the main export.
  H × W is 392 × 518, or 518 × 392 for the 5 portrait scenes.
- **Same scale and world frame as `cameras.npz`.** The scale is arbitrary per scene, not metres.
- **To get a 3D point** in the shared world frame, use `extrinsic[t] = [R | t]` and `intrinsic[t] = K`:

  ```python
  import numpy as np
  d = np.load("scenes/scannet/scene0025_01/depth.npz")  # a scene of the first set
  cam = np.load("../vggt_vsibench_75/scenes/scannet/scene0025_01/cameras.npz")
  t, v, u = 40, 200, 300
  z = float(d["depth"][t, v, u])
  K, R, tr = cam["intrinsic"][t], cam["extrinsic"][t][:, :3], cam["extrinsic"][t][:, 3]
  x_cam = np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
  x_world = R.T @ (x_cam - tr)
  ```

- **The patch points in the main export come from this depth.** The median of these points over each 14 × 14 patch
  is `patch_points.npz["xyz"]`. The export script checked this for every scene.
- **Filter by confidence.** Sky through windows, mirrors, glass and depth edges get a depth too, but with low
  confidence. Keeping pixels in each scene's top half of `conf` works well; the main export's point cloud does that.

Loading needs only numpy: `np.load(path)["depth"].astype(np.float32)`.
