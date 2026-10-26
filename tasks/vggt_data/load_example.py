"""Load one exported scene and draw a top-down view: point cloud, camera path, and the patch points of one frame.

Needs only numpy, pillow and matplotlib.

  python load_example.py scenes/scannet/scene0025_01 [--frame 40] [--out example.png]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("scene", help="path to scenes/<dataset>/<scene>")
ap.add_argument("--frame", type=int, default=0, help="index 0..127 into the sampled frames")
ap.add_argument("--out", default="example.png")
args = ap.parse_args()
d = Path(args.scene)

meta = json.loads((d / "meta.json").read_text())
cams = np.load(d / "cameras.npz")
patches = np.load(d / "patch_points.npz")
cloud = np.load(d / "point_cloud.npz")
questions = [json.loads(line) for line in open(d / "questions.jsonl")]
t = args.frame
image = Image.open(d / "frames" / f"{t:03d}.jpg")

print(f"{meta['dataset']}/{meta['scene']}: {meta['num_frames']} frames of {meta['image_hw']} px, "
      f"patch grid {meta['patch_grid_hw']}, {len(cloud['xyz'])} cloud points, {len(questions)} questions")
for k in cams.files:
    print(f"  cameras.{k:15s} {cams[k].shape}")
for k in patches.files:
    print(f"  patch_points.{k:10s} {patches[k].shape}")
print("  first question:", questions[0]["question"], "->", questions[0]["ground_truth"])

# Patch (r, c) of frame t covers pixels [14r, 14r+14) x [14c, 14c+14) of frames/<t>.jpg; its 3D point is xyz[t, r, c].
xyz, conf = patches["xyz"][t], patches["conf"][t]
good = conf > np.median(conf)  # confidence is relative within a scene; low on windows, mirrors, edges

fig, (a0, a1) = plt.subplots(1, 2, figsize=(14, 6))
a0.imshow(image)
a0.set_title(f"frame {t} (video frame {cams['frame_indices'][t]}, t = {cams['timestamps_s'][t]:.1f} s)")
a0.axis("off")
# Top-down view: world frame = camera 0 in OpenCV axes (x right, y down, z forward), so x-z is the floor plane
# if the first camera was held level.
sub = np.random.default_rng(0).permutation(len(cloud["xyz"]))[:60_000]
a1.scatter(cloud["xyz"][sub, 0], cloud["xyz"][sub, 2], c=cloud["rgb"][sub] / 255, s=0.3)
c = cams["camera_center"]
a1.plot(c[:, 0], c[:, 2], "-", color="k", lw=0.8)
a1.scatter(c[:, 0], c[:, 2], c=np.arange(len(c)), cmap="plasma", s=8, zorder=3)
a1.scatter(xyz[good, 0], xyz[good, 2], color="red", s=2, zorder=4, label=f"patch points of frame {t}")
f = cams["camera_forward"][t]
a1.arrow(c[t, 0], c[t, 2], f[0] * 0.5, f[2] * 0.5, color="red", width=0.02, zorder=5)
a1.set_aspect("equal")
a1.set_xlabel("x")
a1.set_ylabel("z")
a1.legend(loc="upper right")
a1.set_title("top view (scale arbitrary); cameras purple -> yellow over time")
fig.tight_layout()
fig.savefig(args.out, dpi=110)
print("wrote", args.out)
