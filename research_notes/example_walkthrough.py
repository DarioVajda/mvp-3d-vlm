"""One training example for the walkthrough page: a real question, its 8 frames with the 20x15 grid drawn, and the locality-bias arithmetic."""
import base64
import io
import json
import math
import sys

import numpy as np
from decord import VideoReader, cpu
from PIL import Image, ImageDraw

sys.path.insert(0, "/shared/workspace/povejmo/math/projects/mvp/src")
from mvp.train.data import load_vlm3r  # noqa: E402

OUT = "/shared/workspace/povejmo/math/projects/mvp/.claude/worktrees/je-xattn-proposal/research_notes/example.json"
VIDEO = "scannet/videos/scene0126_01.mp4"
samples, _ = load_vlm3r(6000, 0, 60, 100)
qs = [s for s in samples if s["video"] == VIDEO]
pick = {}
for s in qs:
    pick.setdefault(s["task"], s)
ex = pick.get("rel_direction") or qs[0]
print("video", VIDEO, "questions", len(qs), "tasks", sorted(pick))
print("example:", ex["task"], "|", ex["question"][:400].replace("\n", " / "), "| answer", ex["answer"])

vr = VideoReader(f"/shared/workspace/povejmo/math/projects/mvp/data/train_media/{VIDEO}", ctx=cpu(0))
idx = np.linspace(0, len(vr) - 1, 8).round().astype(int)
frames = vr.get_batch(idx).asnumpy()
T_, R_, C_ = 2, 4, 7  # the highlighted cell: frame 2, row 4, col 7 (same as the proposal)


def to_uri(img, q=82):
    b = io.BytesIO(); img.save(b, "JPEG", quality=q); return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


# contact sheet: 8 frames at 240x180 with the 20x15 grid and the highlighted cell in frame 2
thumbs = []
for t in range(8):
    im = Image.fromarray(frames[t]).resize((320, 240))
    d = ImageDraw.Draw(im, "RGBA")
    for c in range(1, 20): d.line([(16 * c, 0), (16 * c, 240)], fill=(255, 255, 255, 70))
    for r in range(1, 15): d.line([(0, 16 * r), (320, 16 * r)], fill=(255, 255, 255, 70))
    if t == T_:
        d.rectangle([16 * C_, 16 * R_, 16 * C_ + 16, 16 * R_ + 16], outline=(255, 60, 40, 255), width=3)
    d.rectangle([0, 0, 44, 18], fill=(0, 0, 0, 150)); d.text((4, 3), f"t = {t}", fill=(255, 255, 255, 255))
    thumbs.append(im)
sheet = Image.new("RGB", (320 * 4, 240 * 2))
for t, im in enumerate(thumbs): sheet.paste(im, (320 * (t % 4), 240 * (t // 4)))
# zoom of frame 2 with the cell and its neighbours
z = Image.fromarray(frames[T_]).resize((640, 480)); d = ImageDraw.Draw(z, "RGBA")
for c in range(1, 20): d.line([(32 * c, 0), (32 * c, 480)], fill=(255, 255, 255, 90))
for r in range(1, 15): d.line([(0, 32 * r), (640, 32 * r)], fill=(255, 255, 255, 90))
d.rectangle([32 * C_, 32 * R_, 32 * C_ + 32, 32 * R_ + 32], outline=(255, 60, 40, 255), width=4)
for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
    d.rectangle([32 * (C_ + dc), 32 * (R_ + dr), 32 * (C_ + dc) + 32, 32 * (R_ + dr) + 32], outline=(255, 200, 40, 230), width=2)
d.rectangle([0, 0, 230, 20], fill=(0, 0, 0, 160)); d.text((4, 4), f"frame t=2, cell (r=4, c=7): n = 2*300 + 4*20 + 7 = 687", fill=(255, 255, 255, 255))
# same cell in frame 5 for comparison (what "same (r,c), other frame" looks like)
z5 = Image.fromarray(frames[5]).resize((640, 480)); d5 = ImageDraw.Draw(z5, "RGBA")
for c in range(1, 20): d5.line([(32 * c, 0), (32 * c, 480)], fill=(255, 255, 255, 90))
for r in range(1, 15): d5.line([(0, 32 * r), (640, 32 * r)], fill=(255, 255, 255, 90))
d5.rectangle([32 * C_, 32 * R_, 32 * C_ + 32, 32 * R_ + 32], outline=(255, 60, 40, 255), width=4)
d5.rectangle([0, 0, 230, 20], fill=(0, 0, 0, 160)); d5.text((4, 4), "frame t=5, the same cell (r=4, c=7): n = 1587", fill=(255, 255, 255, 255))


# locality-bias arithmetic on the real 20x15 grid, T frames, query at (r,c)
def bias_mass(lam, mu, T, r=R_, c=C_, content=0.0):
    rows = []
    for t in range(T):
        for rr in range(15):
            for cc in range(20):
                b = -lam * (abs(rr - r) + abs(cc - c)) - mu * (t != T_)
                rows.append((t, rr, cc, b))
    logit = np.array([b for *_, b in rows]); w = np.exp(logit - logit.max()); w /= w.sum()
    own = float(w[[i for i, (t, rr, cc, _) in enumerate(rows) if t == T_ and rr == r and cc == c][0]])
    frame = float(sum(w[i] for i, (t, *_r) in enumerate(rows) if t == T_))
    d1 = float(sum(w[i] for i, (t, rr, cc, _) in enumerate(rows) if t == T_ and abs(rr - r) + abs(cc - c) == 1))
    same_other = float(sum(w[i] for i, (t, rr, cc, _) in enumerate(rows) if t != T_ and rr == r and cc == c))
    ent = float(-(w * np.log(w + 1e-30)).sum())
    return {"own": own, "frame": frame, "dist1_total": d1, "same_cell_other_frames": same_other, "entropy": ent, "n_keys": len(rows)}


tbl = {}
for name, lam, mu, T in (("init_T8", 1.5, 3.0, 8), ("learned_T8", 1.71, 3.10, 8), ("init_T32", 1.5, 3.0, 32), ("learned_T32", 1.71, 3.10, 32),
                         ("proposal_T8", 4.0, 4.0, 8), ("report_sharp_T8", 4.0, 6.0, 8), ("mu_corrected_T32", 1.71, 3.10 + math.log(31 / 7), 32)):
    tbl[name] = {"lam": lam, "mu": mu, "T": T, **bias_mass(lam, mu, T)}
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in tbl[name].items()})

# RoPE3 phase budget: Qwen2-VL-style sequential split 16/24/24 pairs of a 64-pair head, base 1e6 vs base 100
def phase_range(pairs_before, pairs, base, max_offset):
    # pair j (global index) rotates with frequency base^(-2j/128) per unit position
    freqs = [base ** (-2 * j / 128) for j in range(pairs_before, pairs_before + pairs)]
    return max(freqs) * max_offset, min(freqs) * max_offset  # radians over the full offset


rope = {"t_base1e6": phase_range(0, 16, 1e6, 7), "r_base1e6": phase_range(16, 24, 1e6, 14), "c_base1e6": phase_range(40, 24, 1e6, 19),
        "r_base100": phase_range(16, 24, 100, 14), "c_base100": phase_range(40, 24, 100, 19), "c_base100_first": phase_range(0, 24, 100, 19)}
for k, v in rope.items(): print("rope", k, "max/min phase over full offset (rad):", round(v[0], 4), round(v[1], 6))

json.dump({"video": VIDEO, "n_questions": len(qs), "frame_idx": idx.tolist(), "n_frames_total": int(len(vr)), "example": ex,
           "sheet": to_uri(sheet, 78), "zoom": to_uri(z, 80), "zoom5": to_uri(z5, 80), "bias": tbl, "rope": rope}, open(OUT, "w"))
print("saved", OUT)
