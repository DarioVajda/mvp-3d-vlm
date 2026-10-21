"""Figures for docs/walkthrough.md: one VSI-Bench question followed through frame sampling, VGGT, and the
Spatial-MLLM-style model (training and inference).

  python src/walkthrough.py --video scannetpp/09c1414f1b      # light: reads data/vggt, decodes 128 small frames
"""
import argparse
import json

import numpy as np
from decord import VideoReader, cpu
from PIL import Image, ImageDraw, ImageFont

from sa_sampling import coverage_curve, load_points, max_coverage, uniform_subset, voxel_sets
from show_vggt import FONT, FONT_BOLD, path_colour
from vsibench import ROOT

OUT = ROOT / "results" / "walkthrough"
SA_COL, UNI_COL, GREY = (230, 110, 20), (40, 100, 200), (205, 205, 205)


def fonts():
    return {s: ImageFont.truetype(FONT, s) for s in (11, 13, 15)} | {f"b{s}": ImageFont.truetype(FONT_BOLD, s) for s in (13, 15, 18)}


def candidates_figure(frames, idx, fps, sa, uni, F):
    """All 128 candidate frames; space-aware picks outlined orange (with pick order), uniform picks marked blue."""
    tw, th, cols, pad = 96, 72, 16, 6
    rows = int(np.ceil(len(frames) / cols))
    W, H0 = cols * (tw + pad) + pad, 58
    im = Image.new("RGB", (W, H0 + rows * (th + 20 + pad) + 10), "white")
    d = ImageDraw.Draw(im)
    d.text((pad, 6), "Step 1-3: 128 candidate frames (uniform over the video), each run through VGGT; 16 are kept", font=F["b15"], fill=(0, 0, 0))
    d.rectangle([pad, 32, pad + 14, 44], outline=SA_COL, width=3)
    d.text((pad + 20, 31), "space-aware pick (number = greedy pick order)", font=F[13], fill=(0, 0, 0))
    d.rectangle([380, 38, 394, 42], fill=UNI_COL)
    d.text((400, 31), "uniform pick (linspace over the 128)", font=F[13], fill=(0, 0, 0))
    order = {c: i + 1 for i, c in enumerate(sa["selection_order"])}
    for k, f in enumerate(frames):
        r, c = divmod(k, cols)
        x, y = pad + c * (tw + pad), H0 + r * (th + 20 + pad)
        im.paste(Image.fromarray(f).resize((tw, th)), (x, y))
        if k in order:
            d.rectangle([x - 3, y - 3, x + tw + 2, y + th + 2], outline=SA_COL, width=4)
            d.rectangle([x, y, x + 18, y + 15], fill=SA_COL)
            d.text((x + 2, y), str(order[k]), font=F["b13"], fill=(255, 255, 255))
        if k in uni:
            d.rectangle([x, y + th + 4, x + tw, y + th + 8], fill=UNI_COL)
        d.text((x, y + th + 8), f"#{k}  {idx[k] / fps:.0f}s", font=F[11], fill=(90, 90, 90))
    return im


def coverage_map(sets, grid, picks, cams, size, title, F, colour_by_step):
    """Top view (x-z) of the voxel grid: voxel columns covered by `picks`, coloured by the pick that first
    covered them (space-aware) or in blue (uniform); grey = seen by some candidate frame but by no pick."""
    dims = grid["dims"]
    def xz(ids):
        x = ids // (dims[1] * dims[2])
        z = ids % dims[2]
        return x, z
    allv = np.unique(np.concatenate(sets))
    first = {}
    for step, t in enumerate(picks):
        for v in sets[t]:
            first.setdefault(int(v), step)
    nx, nz = dims[0], dims[2]
    cell = min((size - 40) // nx, (size - 60) // nz)
    im = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(im)
    d.text((6, 4), title, font=F["b13"], fill=(0, 0, 0))
    ox, oz = (size - nx * cell) // 2, 30
    best = {}  # per (x, z) column: covered step (min) or -1
    ax, az = xz(allv)
    for v, x, z in zip(allv, ax, az):
        s = first.get(int(v), 10**9)
        best[(x, z)] = min(best.get((x, z), 10**9), s)
    for (x, z), s in best.items():
        if s == 10**9:
            col = GREY
        else:
            col = path_colour(s / max(len(picks) - 1, 1)) if colour_by_step else UNI_COL
        d.rectangle([ox + x * cell, oz + z * cell, ox + (x + 1) * cell - 1, oz + (z + 1) * cell - 1], fill=col)
    # camera centres of the picks, in voxel units
    for step, t in enumerate(picks):
        u = (cams[t, 0] - grid["origin"][0]) / grid["voxel_size"]
        w = (cams[t, 2] - grid["origin"][2]) / grid["voxel_size"]
        px, pz = ox + u * cell, oz + w * cell
        d.ellipse([px - 4, pz - 4, px + 4, pz + 4], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    n_cov = sum(s != 10**9 for s in best.values())
    d.text((6, size - 22), f"{n_cov}/{len(best)} floor-plan cells covered", font=F[13], fill=(0, 0, 0))
    return im


def curve_plot(curves, size, F):
    im = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(im)
    d.text((6, 4), "Voxels covered vs. number of frames", font=F["b13"], fill=(0, 0, 0))
    x0, y0, x1, y1 = 50, 40, size - 15, size - 45
    d.rectangle([x0, y0, x1, y1], outline=(150, 150, 150))
    for f in (0.25, 0.5, 0.75, 1.0):
        y = y1 - f * (y1 - y0)
        d.line([x0, y, x1, y], fill=(230, 230, 230))
        d.text((x0 - 36, y - 7), f"{int(f * 100)}%", font=F[11], fill=(90, 90, 90))
    for k in (1, 4, 8, 12, 16):
        x = x0 + (k - 1) / 15 * (x1 - x0)
        d.text((x - 4, y1 + 4), str(k), font=F[11], fill=(90, 90, 90))
    d.text(((x0 + x1) // 2 - 40, y1 + 20), "frames kept (K)", font=F[11], fill=(60, 60, 60))
    for (name, ys, col), ly in zip(curves, (y1 - 40, y1 - 22)):
        pts = [(x0 + i / 15 * (x1 - x0), y1 - v * (y1 - y0)) for i, v in enumerate(ys)]
        d.line(pts, fill=col, width=3)
        for p in pts:
            d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=col)
        d.line([x1 - 150, ly + 7, x1 - 130, ly + 7], fill=col, width=3)
        d.text((x1 - 125, ly), f"{name}: {ys[-1] * 100:.0f}%", font=F[11], fill=(0, 0, 0))
    return im


def selected_frames_figure(frames, idx, fps, picks_by_name, F):
    tw, th, pad, lw = 112, 84, 4, 120
    W = lw + 16 * (tw + pad)
    im = Image.new("RGB", (W, 36 + len(picks_by_name) * (th + 22)), "white")
    d = ImageDraw.Draw(im)
    d.text((6, 6), "Step 4: the 16 frames the language model sees", font=F["b15"], fill=(0, 0, 0))
    for r, (name, (picks, col)) in enumerate(picks_by_name.items()):
        y = 36 + r * (th + 22)
        d.text((6, y + th // 2 - 8), name, font=F["b13"], fill=col)
        for c, k in enumerate(picks):
            x = lw + c * (tw + pad)
            im.paste(Image.fromarray(frames[k]).resize((tw, th)), (x, y))
            d.text((x, y + th + 2), f"#{k}  {idx[k] / fps:.0f}s", font=F[11], fill=(90, 90, 90))
    return im


def box(d, xy, text, F, fill, outline=(60, 60, 60), bold=None):
    d.rounded_rectangle(xy, radius=8, fill=fill, outline=outline, width=2)
    lines = text.split("\n")
    y = xy[1] + 6
    for i, line in enumerate(lines):
        f = F["b13"] if i == 0 else F[11]
        d.text((xy[0] + 8, y), line, font=f, fill=(0, 0, 0))
        y += 17 if i == 0 else 14


def arrow(d, p, q, col=(60, 60, 60)):
    d.line([p, q], fill=col, width=2)
    v = np.array(q, float) - np.array(p, float)
    v /= np.linalg.norm(v)
    n = np.array([-v[1], v[0]])
    tip = np.array(q, float)
    d.polygon([tuple(tip), tuple(tip - 10 * v + 5 * n), tuple(tip - 10 * v - 5 * n)], fill=col)


def architecture_figure(F, gt):
    """Spatial-MLLM data flow with the real tensor shapes for this example (Qwen2.5-VL-3B base, as published)."""
    W, H = 1500, 640
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    FROZEN, TRAIN, DATA = (225, 232, 245), (255, 222, 190), (240, 240, 240)
    d.text((10, 8), "Step 5: Spatial-MLLM (Qwen2.5-VL-3B + VGGT-1B), shapes for this example", font=F["b18"], fill=(0, 0, 0))
    for x, col, t in [(10, FROZEN, "frozen (pretrained, no gradient)"), (290, TRAIN, "trained in SFT"), (480, DATA, "data / tensor")]:
        d.rounded_rectangle([x, 40, x + 22, 56], radius=4, fill=col, outline=(60, 60, 60))
        d.text((x + 30, 40), t, font=F[13], fill=(0, 0, 0))

    box(d, (10, 80, 230, 180), "Video\n5450 frames, 640x480, 182 s\n128 uniform candidates\n-> VGGT -> space-aware\n   sampling (steps 1-3)", F, DATA)
    box(d, (270, 80, 470, 180), "16 frames\nresized to 644x476\n(multiple of 28, ~300k px,\nas in training)", F, DATA)
    arrow(d, (230, 130), (270, 130))
    # 2D path
    box(d, (520, 70, 760, 170), "Qwen2.5-VL ViT\n14px patches: 46x34 per frame\n2 frames x 2x2 patches merged\n-> 8 x 23x17 = 3128 tokens", F, FROZEN)
    box(d, (800, 80, 980, 160), "video_embeds\n[3128, 2048]", F, DATA)
    arrow(d, (470, 115), (520, 115))
    arrow(d, (760, 120), (800, 120))
    # 3D path
    box(d, (520, 220, 760, 330), "VGGT-1B aggregator\nsame 16 frames, 14px patches\n24 blocks frame/global attn.\nlast block (layer_idx=-1)", F, FROZEN)
    box(d, (520, 360, 760, 420), "spatial tokens\n[16, 1564, 2048] (46x34 patches)", F, DATA)
    box(d, (800, 300, 1060, 420), "Connector (mlp_add)\nmerge 2 frames x 2x2 patches\n-> [3128, 8*2048=16384]\nRMSNorm, Linear 16384->16384,\nGELU, Linear 16384->2048", F, TRAIN)
    arrow(d, (470, 150), (520, 275))
    arrow(d, (640, 330), (640, 360))
    arrow(d, (760, 390), (800, 380))
    # fuse
    d.ellipse([1095, 175, 1135, 215], fill=(255, 255, 255), outline=(60, 60, 60), width=2)
    d.text((1107, 180), "+", font=F["b18"], fill=(0, 0, 0))
    arrow(d, (980, 120), (1100, 185))
    arrow(d, (1060, 340), (1105, 212))
    box(d, (1170, 140, 1490, 250), "Qwen2.5-3B LLM\nvisual tokens = video_embeds\n  + connector(spatial tokens)\ntext tokens: system + question", F, TRAIN)
    arrow(d, (1135, 195), (1170, 195))

    # training / inference strip
    d.line([10, 460, W - 10, 460], fill=(200, 200, 200), width=1)
    box(d, (10, 480, 735, 625),
        "Training (SFT, 1 epoch)\n"
        "data: 133k QA on 1,201 ScanNet scenes (none in VSI-Bench), e.g. room size:\n"
        "   user: <video> These are frames of a video. What is the size of this room (in square meters)? ...\n"
        "   assistant: 21.2\n"
        "loss: cross-entropy on the answer tokens only; 16 frames per video\n"
        "updated: connector (lr 2e-5) + all LLM weights (lr 7e-6); ViT and VGGT frozen", F, (250, 246, 240))
    box(d, (765, 480, 1490, 625),
        "Inference (this example)\n"
        "same pipeline, no gradients; greedy decoding of a short answer\n"
        "   user: <video> These are frames of a video. What is the size of this room (in square meters)?\n"
        "         If multiple rooms are shown, estimate the size of the combined space. ...\n"
        f"   ground truth: {gt} m^2   (scored by mean relative accuracy)\n"
        "our Qwen3-VL variant: 16px patches, so VGGT gets 560x420 frames to align 40x30 patch grids", F, (240, 246, 250))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="scannetpp/09c1414f1b")
    ap.add_argument("--qid", type=int, default=2707)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    F = fonts()
    scene_dir = ROOT / "data" / "vggt" / args.video
    wp, conf, idx, meta = load_points(scene_dir)
    E = np.load(scene_dir / "extrinsic.npy")
    cams = -np.einsum("nji,nj->ni", E[:, :, :3], E[:, :, 3])
    fps = meta["video_fps"]

    sets, grid = voxel_sets(wp, conf)
    sa_order, gains = max_coverage(sets, 16)
    sa = {"selection_order": sa_order, "candidates": sorted(sa_order)}
    uni = uniform_subset(len(idx), 16)

    frames = VideoReader(str(ROOT / meta["video"]), ctx=cpu(0), width=160, height=120).get_batch(idx.tolist()).asnumpy()
    candidates_figure(frames, idx, fps, sa, set(uni), F).save(OUT / "1_candidates.jpg", quality=90)

    size = 420
    maps = [coverage_map(sets, grid, sa_order, cams, size, "Space-aware 16 (colour = pick order)", F, True),
            coverage_map(sets, grid, uni, cams, size, "Uniform 16", F, False),
            curve_plot([("space-aware", coverage_curve(sets, sa_order), SA_COL),
                        ("uniform", coverage_curve(sets, uni), UNI_COL)], size, F)]
    cov = Image.new("RGB", (3 * size, size + 40), "white")
    ImageDraw.Draw(cov).text((6, 8), "Step 3: greedy voxel coverage, top view of the voxel grid (white dots = cameras of the picks)",
                             font=F["b15"], fill=(0, 0, 0))
    for i, m in enumerate(maps):
        cov.paste(m, (i * size, 40))
    cov.save(OUT / "3_coverage.jpg", quality=90)

    selected_frames_figure(frames, idx, fps, {"uniform": (uni, UNI_COL), "space-aware": (sa["candidates"], SA_COL)},
                           F).save(OUT / "4_frames.jpg", quality=90)
    architecture_figure(F, 54.1).save(OUT / "5_architecture.png")

    info = {"video": args.video, "qid": args.qid, "candidates": idx.tolist(), "uniform": uni, "sa_order": sa_order,
            "sa_gains": gains, "sa_sorted": sa["candidates"], "voxel_size": grid["voxel_size"],
            "grid_dims": grid["dims"].tolist(), "conf_threshold": grid["conf_threshold"],
            "n_voxels": int(len(np.unique(np.concatenate(sets)))),
            "coverage_sa": coverage_curve(sets, sa_order)[-1], "coverage_uniform": coverage_curve(sets, uni)[-1]}
    (OUT / "walkthrough.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "candidates"}, indent=1))


if __name__ == "__main__":
    main()
