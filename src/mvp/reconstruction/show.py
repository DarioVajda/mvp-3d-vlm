"""Render a quick check figure of saved VGGT reconstructions: input frames, depth, confidence and the fused cloud.

  python -m mvp.reconstruction.show arkitscenes/41069025 [--root data/vggt] [--out results/figures/examples/vggt_<scene>.jpg]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu
from PIL import Image, ImageDraw, ImageFont

from mvp.paths import FIGURES, ROOT, VGGT_DIR


def colormap(x, lo, hi):
    """Simple blue -> yellow ramp without matplotlib."""
    t = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)[..., None]
    return (np.array([30, 40, 120]) * (1 - t) + np.array([250, 230, 60]) * t).astype(np.uint8)


PATH_STOPS = np.array([[13, 8, 135], [126, 3, 168], [204, 71, 120], [248, 149, 64], [240, 249, 33]], float)  # plasma


def path_colour(t):
    """Colour along the camera path, t in [0, 1] from first to last frame."""
    x = t * (len(PATH_STOPS) - 1)
    i = min(int(x), len(PATH_STOPS) - 2)
    return tuple((PATH_STOPS[i] + (x - i) * (PATH_STOPS[i + 1] - PATH_STOPS[i])).astype(int))


def splat(xyz, rgb, cams, axes, size=420):
    """Orthographic splat of the cloud onto the plane spanned by `axes`, with the camera path drawn on top,
    coloured from purple (first frame) to yellow (last frame)."""
    p = xyz[:, axes]
    lo, hi = np.percentile(p, 1, axis=0), np.percentile(p, 99, axis=0)
    scale = (size - 20) / (hi - lo).max()
    to_px = lambda q: ((q - lo) * scale + 10).astype(int)
    img = np.full((size, size, 3), 255, np.uint8)
    uv = to_px(p)
    ok = (uv >= 0).all(1) & (uv < size).all(1)
    order = np.argsort(xyz[ok][:, 3 - sum(axes)])  # draw along the dropped axis
    img[uv[ok][order, 1], uv[ok][order, 0]] = rgb[ok][order]
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    path = [tuple(q) for q in to_px(cams[:, axes])]
    n = len(path)
    for i in range(n - 1):
        d.line([path[i], path[i + 1]], fill=path_colour(i / max(n - 2, 1)), width=2)
    for i, (u, v) in enumerate(path):
        r = 5 if i in (0, n - 1) else 3
        d.ellipse([u - r, v - r, u + r, v + r], fill=path_colour(i / max(n - 1, 1)), outline=(0, 0, 0), width=1)
    return im


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

HEADERS = [
    ("Input frame", "VGGT-1B input (1 of 128)"),
    ("Depth", "depth head: DPT, layers 4/11/17/23"),
    ("Depth confidence", "depth head, log scale"),
]
FOOTNOTE = [
    "VGGT-1B, one joint forward pass over 128 uniformly spaced frames; 4 of them are shown. The aggregator has 24 blocks of",
    "alternating frame / global attention; the camera head reads the camera token of the last block, the DPT heads read",
    "blocks 4, 11, 17 and 23.",
    "Depth: blue = near, yellow = far, rescaled per frame (2nd-98th percentile). Units are VGGT's normalised scene scale, not metres.",
    "Confidence: blue = 1 (the minimum of VGGT's 1 + exp(x) confidence), yellow = 20 or more. It is low on windows, mirrors,",
    "object edges and textureless walls, and is relative within a scene, not comparable across scenes.",
    "Point cloud: depth unprojected with the camera head's poses, kept if confidence is in the scene's top half, voxel-downsampled.",
    "World frame = first camera (OpenCV: x right, y down, z forward). Camera path: one dot per frame, purple = first, yellow = last.",
]


def label_bar(width, lines, height, font, bold, bg=(245, 245, 245)):
    im = Image.new("RGB", (width, height), bg)
    d = ImageDraw.Draw(im)
    y = 4
    for i, text in enumerate(lines):
        f = bold if i == 0 else font
        d.text((6, y), text, fill=(20, 20, 20), font=f)
        y += f.size + 3
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="<dataset>/<scene>")
    ap.add_argument("--root", default=str(VGGT_DIR))
    ap.add_argument("--out")
    args = ap.parse_args()
    d = Path(args.root) / args.video
    meta = json.loads((d / "meta.json").read_text())
    idx = np.load(d / "frame_indices.npy")
    depth = np.load(d / "depth.npy", mmap_mode="r")
    conf = np.load(d / "depth_conf.npy", mmap_mode="r")
    E = np.load(d / "extrinsic.npy")
    cams = -np.einsum("nji,nj->ni", E[:, :, :3], E[:, :, 3])
    cloud = np.load(d / "cloud.npz")
    font, bold = ImageFont.truetype(FONT, 12), ImageFont.truetype(FONT_BOLD, 13)
    small = ImageFont.truetype(FONT, 11)

    vr = VideoReader(str(ROOT / meta["video"]), ctx=cpu(0))
    picks = np.linspace(0, len(idx) - 1, 4, dtype=int)
    frames = vr.get_batch(idx[picks].tolist()).asnumpy()
    w = 260
    h = round(w * frames.shape[1] / frames.shape[2])
    rows = []
    for f, k in zip(frames, picks):
        dk, ck = np.asarray(depth[k], np.float32), np.asarray(conf[k], np.float32)
        tiles = [Image.fromarray(f), Image.fromarray(colormap(dk, *np.percentile(dk, [2, 98]))),
                 Image.fromarray(colormap(np.log(ck), 0, np.log(20)))]
        tiles = [t.resize((w, h)) for t in tiles]
        tag = f"#{k}: video frame {idx[k]}, t={idx[k] / meta['video_fps']:.0f}s"
        dr = ImageDraw.Draw(tiles[0])
        x0, y0, x1, y1 = dr.textbbox((4, h - 18), tag, font=small)
        dr.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], fill=(0, 0, 0))
        dr.text((4, h - 18), tag, fill=(255, 255, 255), font=small)
        rows.append(np.hstack([np.asarray(t) for t in tiles]))
    left = Image.fromarray(np.vstack(rows))
    bar_h = 36
    vh = left.height // 2 - bar_h
    view_titles = [("Point cloud, top view (x-z)", "depth + camera head poses; camera path on top"),
                   ("Point cloud, front view (x-y)", "as seen from the first camera")]
    views = []
    for (title, sub), axes in zip(view_titles, ([0, 2], [0, 1])):
        views.append(np.asarray(label_bar(vh, [title, sub], bar_h, small, bold, bg=(255, 255, 255))))
        views.append(np.asarray(splat(cloud["xyz"], cloud["rgb"], cams, axes, size=vh)))
    right = Image.fromarray(np.vstack(views))

    head_h = 40
    header = Image.new("RGB", (left.width + right.width, head_h), (245, 245, 245))
    for i, (title, sub) in enumerate(HEADERS):
        header.paste(label_bar(w, [title, sub], head_h, small, bold), (i * w, 0))
    header.paste(label_bar(right.width, ["3D reconstruction", f"{meta['num_frames']} cameras, "
                                         f"{len(cloud['xyz']) // 1000}k points"], head_h, small, bold), (left.width, 0))
    title = f"{args.video}  ({meta['num_video_frames']} video frames, {meta['num_video_frames'] / meta['video_fps']:.0f}s, " \
            f"VGGT input {meta['input_hw'][1]}x{meta['input_hw'][0]})"
    foot = label_bar(header.width, [title] + FOOTNOTE, 16 + 15 * (len(FOOTNOTE) + 1), small, bold, bg=(255, 255, 255))

    canvas = Image.new("RGB", (header.width, head_h + max(left.height, right.height) + foot.height), "white")
    canvas.paste(header, (0, 0))
    canvas.paste(left, (0, head_h))
    canvas.paste(right, (left.width, head_h))
    body_h = max(left.height, right.height)
    canvas.paste(foot, (0, head_h + body_h))
    ImageDraw.Draw(canvas).line([(0, head_h + body_h), (canvas.width, head_h + body_h)], fill=(180, 180, 180))
    out = Path(args.out or FIGURES / "examples" / f"vggt_{args.video.replace('/', '_')}.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=90)
    print(out, f"(frames {idx[picks].tolist()}, {len(cloud['xyz'])} cloud points)")


if __name__ == "__main__":
    main()
