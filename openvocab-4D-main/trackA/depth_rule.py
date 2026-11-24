"""Which pixels of a detection's mask have depth good enough to place it in 3D.

Shared by the track and fuse stages of milestone4_video.py and by the scene sheet (src/mvp/openvocab_adapter/
scene_sheet_creator.py), so that "the object is seen in this frame" means the same everywhere.

  global  (milestone 4's original rule) a pixel counts if its VGGT depth confidence is at least the global threshold,
          one percentile (--conf-percentile, 55) of the confidence over every frame of the sequence. Screens, windows,
          mirrors, lamps and thin objects such as chairs have low confidence over most of their mask, so many of
          their detections have too few such pixels and are dropped.
  mask    a pixel counts if its confidence is at least the lower of the global threshold and the mask's own
          `quantile` percentile (its median by default). A mask with confident depth gets the global rule; a mask
          whose depth is uncertain throughout keeps its better half, instead of being dropped.

Both also need depth > 0. A detection counts in a frame when it has at least min_points such pixels: MIN_POINTS gives
each rule's default (300 is the original; the mask rule keeps at most half a mask, so it needs fewer).
"""
import numpy as np

RULES = ("global", "mask")
MIN_POINTS = {"global": 300, "mask": 50}
DEFAULT_QUANTILE = 50.0


def check_rule(rule):
    if rule not in RULES:
        raise ValueError(f"depth rule must be one of {RULES}, not {rule!r}")


def min_points_for(rule, min_points=None):
    """min_points, or the rule's default when it is None."""
    check_rule(rule)
    return MIN_POINTS[rule] if min_points is None else int(min_points)


def usable_pixels(mask, depth, conf, global_thr, rule="global", quantile=DEFAULT_QUANTILE):
    """Boolean array: the pixels of mask whose depth counts under the rule.

    mask, depth, conf: one frame's (H, W) arrays (conf as float); global_thr: the sequence's confidence threshold."""
    check_rule(rule)
    valid = mask & (depth > 0)
    if rule == "global":
        return valid & (conf >= global_thr)
    if not valid.any():
        return valid
    own = np.percentile(conf[valid], quantile)
    return valid & (conf >= min(global_thr, own))
