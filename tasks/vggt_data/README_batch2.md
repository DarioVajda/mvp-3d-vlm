# VGGT reconstructions of 75 more VSI-Bench scenes (batch 2)

A second set of 75 indoor videos from [VSI-Bench](https://arxiv.org/abs/2412.14171), none of them in the first set
(`vggt_vsibench_75.zip`). **The format is exactly the same as the first set**, so code written for it works
unchanged. `FORMAT.md` is the first set's README and describes every file; its examples use the scene
`scannet/scene0025_01`, which is in the first set; here, try `scannet/scene0435_02` instead:

```bash
pip install numpy pillow matplotlib
python load_example.py scenes/scannet/scene0435_02 --frame 40 --out example.png
```

For each video, as before:

- 128 frames, evenly spaced over the video, at VGGT's input resolution;
- camera poses and one 3D point per 14×14 patch from VGGT-1B (`cameras.npz`, `patch_points.npz`);
- a fused, coloured point cloud (`point_cloud.ply` / `.npz`);
- every VSI-Bench question about the scene, with its answer and the zero-shot answers of Qwen3-VL-2B/4B
  (`questions.jsonl`);
- `meta.json` and a `preview.jpg` check figure.

The per-pixel depth maps are in a separate zip, `depth_vsibench_batch2.zip`, as for the first set.

## Scenes

The same number per source as the first set: 25 ARKitScenes landscape, 5 ARKitScenes portrait, 25 ScanNet and 20
ScanNet++. They were chosen by the same rule: within each group, the scenes with the most distinct question types,
then the most questions, after leaving out the first 75. So these are the next scenes on that ranking, and they are
smaller: 1,447 questions in total (19 per scene, against 36 in the first set), and fewer question types per scene.
All 10 question types still appear; `obj_appearance_order` is in only 8 scenes. `scenes.csv` lists the scenes with
their question counts and types.

Together with the first set, this covers 150 of VSI-Bench's 288 videos and 4,175 of its 5,130 questions.

## Caveats

The same as for the first set: the world scale is arbitrary per scene (not metres), poses can drift on long or
textureless videos, and glass and mirrors get wrong depth; look at `preview.jpg` before relying on a scene. The
frames come from ScanNet, ScanNet++ and ARKitScenes through VSI-Bench; their terms of use apply. Please don't
redistribute the data further.
