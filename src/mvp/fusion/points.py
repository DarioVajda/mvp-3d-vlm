"""3D positions of the merged 2x2 cells from VGGT's point head, for AlignedXAttn's 3D position signals (pe="3d").

VGGT's point head gives a 3D point and a confidence per pixel, in the first camera's frame and at an arbitrary scale
(VGGT is trained on scenes normalised to mean point distance 1, which is a learned habit, not a unit). Per video:

  xyz_n  = mean of the points over the 28 x 28 px of merged cell n   (2 x 2 VGGT patches = one Qwen visual token)
  low_n  = cell confidence at VGGT's floor (conf = 1 + exp(x) saturates at exactly 1), or below the 10th percentile
           of the other cells' confidence                             (sky, glass, far field; floor cells are often
           a third or half of a video, and their points do not line up across frames)
  s      = median over frames of |xyz(r, c) - xyz(r, c + 1)| over horizontally adjacent pairs of confident cells
  xyz_n /= s                                                          (1 unit = one merged cell on a typical surface)

  dist_ij = min(|xyz_i - xyz_j|, d_max), set to d_max for low-confidence keys j except on the diagonal (dist_ii = 0)

In cell units the 3D-distance slope nu has the units of the grid slope lam ("per cell"), the unit does not depend on
how many frames are sampled, and the median over adjacent pairs ignores the junk at depth edges.
"""
from dataclasses import dataclass

import torch


@dataclass
class ScenePoints:
    xyz: torch.Tensor   # [N, 3] float32, cell units, VGGT camera-0 frame
    low: torch.Tensor   # [N] bool, low-confidence cells
    dist: torch.Tensor  # [N, N] float32, clipped 3D distance between cells, low-confidence keys at d_max
    scale: float        # s: VGGT units per cell


FLOOR = 1 + 1e-6  # cell confidence at or below this: every pixel of the cell at VGGT's confidence floor


def cell_points(points, conf, grid, d_max=10.0, patch=14, merge=2, low_q=0.1):
    """points [T, H, W, 3], conf [T, H, W] of one video (VGGT point head, pixel resolution); grid: image_grid_thw
    [T, 3] in patches. Returns ScenePoints over the N = sum_t (h_t / merge) (w_t / merge) merged cells in Qwen's token
    order (frame, then row-major), the order of AlignedXAttn.positions."""
    cell = patch * merge
    xyz, cconf, pairs = [], [], []
    for t, (_, h, w) in enumerate(grid.tolist()):
        hc, wc = h // merge, w // merge
        p = points[t, :hc * cell, :wc * cell].float().reshape(hc, cell, wc, cell, 3).mean((1, 3))  # [hc, wc, 3]
        c = conf[t, :hc * cell, :wc * cell].float().reshape(hc, cell, wc, cell).mean((1, 3))
        xyz.append(p.reshape(-1, 3))
        cconf.append(c.reshape(-1))
        pairs.append((p, c))
    xyz, cconf = torch.cat(xyz), torch.cat(cconf)
    floor = cconf <= FLOOR
    rest = cconf[~floor]
    thr = torch.quantile(rest, low_q) if len(rest) else cconf.new_tensor(float("inf"))
    low = floor | (cconf < thr)
    gaps = []
    for p, c in pairs:  # adjacent pairs of confident cells in the same row
        good = (c > FLOOR) & (c >= thr)
        ok = good[:, :-1] & good[:, 1:]
        gaps.append((p[:, 1:] - p[:, :-1]).norm(dim=-1)[ok])
    gaps = torch.cat(gaps)
    scale = (gaps.median() if len(gaps) else (xyz[1:] - xyz[:-1]).norm(dim=-1).median()).clamp_min(1e-6)
    xyz = xyz / scale
    dist = torch.cdist(xyz, xyz, compute_mode="donot_use_mm_for_euclid_dist").clamp_max(d_max)  # exact near 0
    dist[:, low] = d_max
    dist.fill_diagonal_(0.0)
    return ScenePoints(xyz=xyz, low=low, dist=dist, scale=scale.item())
