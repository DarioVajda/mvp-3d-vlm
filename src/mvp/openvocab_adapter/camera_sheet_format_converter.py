"""Camera sheet in words: one instruction per frame, saying where the camera moves and where it then looks.

Reads a scene sheet's cameras.csv (its columns are described in scene_sheet_creator.py) and gives a paragraph. The
first sentence is frame 0's view. Each later one says how the camera got to its frame from the frame before: first
where it moved, then where it looks. There are two modes:

  global    (the default) Moves are compass directions, and each frame gives its facing and tilt in full:
              Start facing north, looking 29° down. Go a little north-east and face 49° clockwise from north,
              looking 18° down. Go north-east and face 121° clockwise from north, looking 21° down. ...
  relative  Moves are relative to where the camera faced in the frame before, and so are the turns:
              Start looking 29° down. Move a little forward and to the right, and turn 49° to the right and 11° up.
              Move forward and turn 72° to the right and 3° down. ...

Conventions. The sheet has no gravity, no compass and no metric scale, so these are assumptions:
  - Up: the direction that makes the camera's horizon most level over the whole trajectory, i.e. the one most
    perpendicular to every frame's camera x axis (scans are filmed with little roll). It does not depend on frame 0,
    whose camera gives the sheet's world axes and often looks 20-30° down. If the camera barely turns, every direction
    perpendicular to its x axis fits equally well, and the frames' mean camera up (OpenCV -y) in that plane is taken.
  - North (global): where frame 0 looks, projected onto the floor (the plane perpendicular to up). East is 90° to its
    right. Moves are rounded to the 8 compass points, and facings are given the shortest way from north, in whole
    degrees ("face 70° counter-clockwise from north"; "face north", "face south").
  - Forward (relative): where the previous frame looks, projected onto the floor. Moves are rounded to 8 directions:
    forward, backward, "strafe to the left/right", and the diagonals ("forward and to the left").
  - Distances: the step from one camera centre to the next, compared with the trajectory's median step (the sheet's
    units are VGGT's arbitrary scale). Under 0.1 median steps, no move is mentioned. Under 0.5 it is "a little", over
    2 "far". A step climbing or dropping more than 30° adds "up" / "down", and over 70° it is only "up" / "down".
  - Tilt: the angle of the view above or below the floor plane. Under 2° it is "straight ahead".
  - Turns (relative): the change in heading (left/right) and in tilt (up/down), in whole degrees. Changes under 2° are
    left out, and a frame with no move and no turn is "Stay still."
  - Roll is never mentioned.

From Python, describe_cameras(path, mode) returns the paragraph and frame_instructions(path, mode) the list of
sentences, where path is a cameras.csv or the sheet folder that holds it. From the command line, the paragraph is
printed:

  python src/mvp/openvocab_adapter/camera_sheet_format_converter.py --mode relative \
      tasks/openvocab_adapter/out/scannet__scene0025_01/subset-uniform16/sheets/door/cameras.csv
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

MODES = ("global", "relative")
COMPASS = [
    "north",
    "north-east",
    "east",
    "south-east",
    "south",
    "south-west",
    "west",
    "north-west",
]
RELATIVE = [
    "forward",
    "forward and to the right",
    "to the right",
    "backward and to the right",
    "backward",
    "backward and to the left",
    "to the left",
    "forward and to the left",
]  # clockwise from forward
SIDEWAYS = ("to the left", "to the right")  # the moves that are "strafe"
NO_MOVE = 0.1  # in median steps: a shorter step is not mentioned
A_LITTLE = 0.5  # in median steps: a shorter step is "a little"
FAR = 2.0  # in median steps: a longer step is "far"
CLIMB_DEG = 30  # a step steeper than this adds "up" / "down"
VERTICAL_DEG = 70  # a step steeper than this is only "up" / "down"
MIN_TURN_DEG = 2  # turns and tilts that round below this are left out ("straight ahead" for a tilt)
LEVEL_RATIO = 0.25  # the horizon fit is used only if its best axis fits this much better than the next one


def quaternion_to_rotation(w, x, y, z):
    """Quaternion (w, x, y, z), normalised first -> 3x3 rotation matrix."""
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def read_cameras(path):
    """cameras.csv, or the sheet folder that holds it -> camera centres (N, 3) and camera-to-world rotations (N, 3, 3),
    in frame order."""
    path = Path(path)
    if path.is_dir():
        path = path / "cameras.csv"
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [
            c
            for c in ("frame", "x", "y", "z", "qw", "qx", "qy", "qz")
            if c not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(
                f"{path} is not a camera sheet: it has no {', '.join(missing)} column"
            )
        rows = sorted(reader, key=lambda r: int(r["frame"]))
    centres = np.array([[float(r[a]) for a in "xyz"] for r in rows]).reshape(-1, 3)
    rotations = np.array(
        [
            quaternion_to_rotation(*(float(r[q]) for q in ("qw", "qx", "qy", "qz")))
            for r in rows
        ]
    ).reshape(-1, 3, 3)
    return centres, rotations


def estimate_up(rotations):
    """World up (unit), assuming a level horizon: the direction most perpendicular to every camera's x axis, on the
    side of the cameras' up (-y)."""
    rights = rotations[:, :, 0]
    cameras_up = -rotations[:, :, 1].sum(axis=0)
    values, vectors = np.linalg.eigh(rights.T @ rights)
    if values[0] < LEVEL_RATIO * values[1]:
        up = vectors[:, 0]
    else:  # the camera barely turned: take its mean up, in the plane perpendicular to its x axis
        right = vectors[:, 2]
        up = cameras_up - (cameras_up @ right) * right
    up = up / np.linalg.norm(up)
    return up if up @ cameras_up >= 0 else -up


def facing(rotation, up):
    """The unit floor direction a camera faces: its view direction projected onto the floor, or, when it looks almost
    straight up or down, the direction 90° to the left of its x axis."""
    view = rotation[:, 2]
    direction = view - (view @ up) * up
    if np.linalg.norm(direction) < 0.05:
        direction = np.cross(up, rotation[:, 0])
    return direction / np.linalg.norm(direction)


def bearing(vector, ahead, right):
    """Degrees clockwise from ahead (in [-180, 180]) of a floor vector, where right is 90° clockwise from ahead."""
    return math.degrees(math.atan2(vector @ right, vector @ ahead))


def move_phrase(step, median_step, up, ahead, right, relative=False):
    """How the camera moved, or None when the step is too short to mention. ahead and right are the floor directions
    the step is measured against: north and east ('Go a little north-east and up'), or, with relative=True, the
    previous camera's facing and its right ('Move a little forward and to the left', 'Strafe to the right and up')."""
    length = float(np.linalg.norm(step))
    if median_step <= 0 or length < NO_MOVE * median_step:
        return None
    rise = float(step @ up)
    flat = step - rise * up
    climb = math.degrees(math.atan2(abs(rise), np.linalg.norm(flat)))
    vertical = "up" if rise > 0 else "down"
    ratio = length / median_step
    how_far = "a little " if ratio < A_LITTLE else "far " if ratio > FAR else ""
    if climb > VERTICAL_DEG:
        return f"{'Move' if relative else 'Go'} {how_far}{vertical}"
    where = (RELATIVE if relative else COMPASS)[
        round(bearing(flat, ahead, right) / 45) % 8
    ]
    verb = "Strafe" if where in SIDEWAYS else "Move" if relative else "Go"
    if climb <= CLIMB_DEG:
        return f"{verb} {how_far}{where}"
    if " and " in where:  # a diagonal
        return f"{verb} {how_far}{vertical}, {where}"
    return f"{verb} {how_far}{where} and {vertical}"


def looking_phrase(tilt):
    """'20° down', or 'straight ahead'."""
    degrees = round(abs(tilt))
    return (
        "straight ahead"
        if degrees < MIN_TURN_DEG
        else f"{degrees}° {'up' if tilt > 0 else 'down'}"
    )


def facing_phrase(heading, tilt):
    """'face 30° clockwise from north, looking 20° down'; heading is in degrees clockwise from north."""
    degrees = round(abs(heading))
    if degrees in (0, 180):
        where = "north" if degrees == 0 else "south"
    else:
        where = f"{degrees}° {'clockwise' if heading > 0 else 'counter-clockwise'} from north"
    return f"face {where}, looking {looking_phrase(tilt)}"


def turn_phrase(heading_change, tilt_change):
    """'turn 23° to the right and 5° down', or None when both changes are too small to mention."""
    heading_change = (
        heading_change + 180
    ) % 360 - 180  # positive: clockwise seen from above, to the right
    parts = []
    if round(abs(heading_change)) >= MIN_TURN_DEG:
        parts.append(
            f"{round(abs(heading_change))}° to the {'right' if heading_change > 0 else 'left'}"
        )
    if round(abs(tilt_change)) >= MIN_TURN_DEG:
        parts.append(
            f"{round(abs(tilt_change))}° {'up' if tilt_change > 0 else 'down'}"
        )
    return "turn " + " and ".join(parts) if parts else None


def sentence(move, then):
    """The move and what follows it (a facing or a turn) as one sentence; either may be None."""
    if move and then:
        # a comma keeps "Go north and up, and turn ..." readable
        return f"{move}{', and' if ' and ' in move else ' and'} {then}."
    phrase = move or then
    return phrase[0].upper() + phrase[1:] + "." if phrase else "Stay still."


def frame_instructions(path, mode="global"):
    """The camera sheet at path (a cameras.csv, or the sheet folder that holds it) as one sentence per frame. mode is
    'global' (compass moves, absolute facings) or 'relative' (moves and turns relative to the previous frame)."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {', '.join(MODES)}")
    centres, rotations = read_cameras(path)
    if not len(centres):
        return []
    up = estimate_up(rotations)
    north = facing(rotations[0], up)
    east = np.cross(north, up)
    headings = [bearing(facing(r, up), north, east) for r in rotations]
    tilts = [math.degrees(math.asin(np.clip(r[:, 2] @ up, -1, 1))) for r in rotations]
    steps = np.diff(centres, axis=0)
    median_step = float(np.median(np.linalg.norm(steps, axis=1))) if len(steps) else 0.0

    if mode == "global":
        sentences = [f"Start facing north, looking {looking_phrase(tilts[0])}."]
    else:
        sentences = [f"Start looking {looking_phrase(tilts[0])}."]
    for t in range(1, len(centres)):
        if mode == "global":
            move = move_phrase(steps[t - 1], median_step, up, north, east)
            then = facing_phrase(headings[t], tilts[t])
        else:
            ahead = facing(rotations[t - 1], up)
            move = move_phrase(
                steps[t - 1], median_step, up, ahead, np.cross(ahead, up), relative=True
            )
            then = turn_phrase(headings[t] - headings[t - 1], tilts[t] - tilts[t - 1])
        sentences.append(sentence(move, then))
    return sentences


def describe_cameras(path, mode="global"):
    """The camera sheet at path (a cameras.csv, or the sheet folder that holds it) as one paragraph of directions.
    mode is 'global' (compass moves, absolute facings) or 'relative' (moves and turns relative to the previous
    frame)."""
    return " ".join(frame_instructions(path, mode))


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "cameras", help="a scene sheet's cameras.csv, or the sheet folder that holds it"
    )
    p.add_argument(
        "--mode",
        choices=MODES,
        default="relative",
        help="global: compass moves and absolute facings; relative: moves and turns relative to the "
        "previous frame",
    )
    args = p.parse_args(argv)
    print(describe_cameras(args.cameras, args.mode))
    return 0


if __name__ == "__main__":
    sys.exit(main())
