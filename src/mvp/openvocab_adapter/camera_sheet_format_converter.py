"""Camera sheet in words: one instruction per frame, saying where the camera moves and how it turns.

Reads a scene sheet's cameras.csv (its columns are described in scene_sheet_creator.py) and gives a paragraph like

  Start facing north, looking 30° down. Go a little north-east and turn 47° to the right and 15° up.
  Go north-east and turn 70° to the right. ...

The first sentence is frame 0's view. Each later one says how the camera got to its frame from the frame before:
first where it moved, then how it turned (or, with turn_first, the other way round: the words for each change the same,
since compass directions don't depend on where the camera faces).

Conventions. The sheet has no gravity, no compass and no metric scale, so these are assumptions:
  - Up: the direction that makes the camera's horizon most level over the whole trajectory, i.e. the one most
    perpendicular to every frame's camera x axis (scans are filmed with little roll). It does not depend on frame 0,
    whose camera gives the sheet's world axes and often looks 20-30° down. If the camera barely turns, every direction
    perpendicular to its x axis fits equally well, and the frames' mean camera up (OpenCV -y) in that plane is taken.
  - North: where frame 0 looks, projected onto the floor (the plane perpendicular to up). East is 90° to its right.
    Directions are rounded to the 8 compass points.
  - Moves: the step from one camera centre to the next, compared with the trajectory's median step (the sheet's units
    are VGGT's arbitrary scale). Under 0.1 median steps, no move is mentioned. Under 0.5 it is "a little", over 2 "far";
    with units=True, the step's length in the sheet's units is given instead ("Go 0.812 units north-east").
    A step climbing or dropping more than 30° adds "and up" / "and down", and over 70° it is only "up" / "down".
  - Turns: the change in heading (left/right) and in pitch (up/down), both measured against up, in whole degrees.
    Changes under 2° are left out, and roll is never mentioned. A frame with no move and no turn is "Stay still."

From Python, describe_cameras(path) returns the paragraph and frame_instructions(path) the list of sentences, where
path is a cameras.csv or the sheet folder that holds it. From the command line, the paragraph is printed:

  python src/mvp/openvocab_adapter/camera_sheet_format_converter.py \
      tasks/openvocab_adapter/out/scannet__scene0025_01/subset-uniform16/sheets/door/cameras.csv
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

COMPASS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]
NO_MOVE = 0.1  # in median steps: a shorter step is not mentioned
A_LITTLE = 0.5  # in median steps: a shorter step is "a little"
FAR = 2.0  # in median steps: a longer step is "far"
CLIMB_DEG = 30  # a step steeper than this adds "and up" / "and down"
VERTICAL_DEG = 70  # a step steeper than this is only "up" / "down"
MIN_TURN_DEG = 2  # heading and pitch changes that round below this are left out
LEVEL_RATIO = 0.25  # the horizon fit is used only if its best axis fits this much better than the next one
UNIT_DECIMALS = 3  # move lengths with units=True, as mvp.benchmarks.scene_sheet writes numbers


def quaternion_to_rotation(w, x, y, z):
    """Quaternion (w, x, y, z), normalised first -> 3x3 rotation matrix."""
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def read_cameras(path, frames=None):
    """cameras.csv, or the sheet folder that holds it -> camera centres (N, 3) and camera-to-world rotations (N, 3, 3),
    in frame order. frames: only the rows with these frame numbers, in this order (every row when None), e.g. the
    VLM's 16 frames of a sheet made on all 128 (mvp.benchmarks.scene_sheet)."""
    path = Path(path)
    if path.is_dir():
        path = path / "cameras.csv"
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in ("frame", "x", "y", "z", "qw", "qx", "qy", "qz") if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is not a camera sheet: it has no {', '.join(missing)} column")
        rows = sorted(reader, key=lambda r: int(r["frame"]))
    if frames is not None:
        by_frame = {int(r["frame"]): r for r in rows}
        if missing := [t for t in frames if t not in by_frame]:
            raise ValueError(f"{path} has no row for frames {missing}")
        rows = [by_frame[t] for t in frames]
    centres = np.array([[float(r[a]) for a in "xyz"] for r in rows]).reshape(-1, 3)
    rotations = np.array(
        [quaternion_to_rotation(*(float(r[q]) for q in ("qw", "qx", "qy", "qz"))) for r in rows]).reshape(-1, 3, 3)
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


def bearing(vector, north, east):
    """Degrees clockwise from north of a floor vector."""
    return math.degrees(math.atan2(vector @ east, vector @ north))


def move_phrase(step, median_step, up, north, east, units=False, scale=None):
    """'Go a little north-east and up', or None when the step is too short to mention. units=True gives the step's
    length instead of "a little" / "far": 'Go 0.812 units north-east and up', or with scale (metres per sheet unit)
    the length in metres: 'Go 1.868 metres north-east and up'. Which steps are mentioned doesn't depend on scale."""
    length = float(np.linalg.norm(step))
    if median_step <= 0 or length < NO_MOVE * median_step:
        return None
    rise = float(step @ up)
    flat = step - rise * up
    climb = math.degrees(math.atan2(abs(rise), np.linalg.norm(flat)))
    vertical = "up" if rise > 0 else "down"
    if climb > VERTICAL_DEG:
        where = vertical
    else:
        where = COMPASS[round(bearing(flat, north, east) / 45) % 8]
        if climb > CLIMB_DEG:
            where += f" and {vertical}"
    if units:
        if scale is not None:
            return f"Go {length * scale:.{UNIT_DECIMALS}f} metres {where}"
        return f"Go {length:.{UNIT_DECIMALS}f} units {where}"
    ratio = length / median_step
    how_far = "a little " if ratio < A_LITTLE else "far " if ratio > FAR else ""
    return f"Go {how_far}{where}"


def turn_phrase(heading_change, pitch_change):
    """'turn 23° to the right and 5° down', or None when both changes are too small to mention."""
    heading_change = (heading_change + 180) % 360 - 180  # positive: clockwise seen from above, to the right
    parts = []
    if round(abs(heading_change)) >= MIN_TURN_DEG:
        parts.append(f"{round(abs(heading_change))}° to the {'right' if heading_change > 0 else 'left'}")
    if round(abs(pitch_change)) >= MIN_TURN_DEG:
        parts.append(f"{round(abs(pitch_change))}° {'up' if pitch_change > 0 else 'down'}")
    return "turn " + " and ".join(parts) if parts else None


def frame_instructions(path, units=False, turn_first=False, scale=None, frames=None):
    """The camera sheet at path (a cameras.csv, or the sheet folder that holds it) as one sentence per frame.
    units=True says how far each move goes, in the sheet's units, instead of "a little" / "far"; with scale (metres
    per sheet unit) as well, in metres. turn_first=True says the turn before the move ("Turn 30° to the right and go
    north."); compass directions don't depend on the order. frames: only these frames (read_cameras), whose moves and
    turns are then from one of them to the next."""
    centres, rotations = read_cameras(path, frames)
    if not len(centres):
        return []
    up = estimate_up(rotations)
    north = facing(rotations[0], up)
    east = np.cross(north, up)
    headings = [bearing(facing(r, up), north, east) for r in rotations]
    pitches = [math.degrees(math.asin(np.clip(r[:, 2] @ up, -1, 1))) for r in rotations]
    steps = np.diff(centres, axis=0)
    median_step = float(np.median(np.linalg.norm(steps, axis=1))) if len(steps) else 0.0

    pitch = round(abs(pitches[0]))
    looking = "straight ahead" if pitch < MIN_TURN_DEG else f"{pitch}° {'up' if pitches[0] > 0 else 'down'}"
    sentences = [f"Start facing north, looking {looking}."]
    for t in range(1, len(centres)):
        move = move_phrase(steps[t - 1], median_step, up, north, east, units, scale)
        turn = turn_phrase(headings[t] - headings[t - 1], pitches[t] - pitches[t - 1])
        if move and turn and turn_first:
            # a comma keeps "Turn 23° to the right and 5° down, and go ..." readable
            sentences.append(f"{turn[0].upper()}{turn[1:]}{', and' if ' and ' in turn else ' and'} "
                             f"{move[0].lower()}{move[1:]}.")
        elif move and turn:
            # a comma keeps "Go north and up, and turn ..." readable
            sentences.append(f"{move}{', and' if ' and ' in move else ' and'} {turn}.")
        elif move or turn:
            phrase = move or turn
            sentences.append(phrase[0].upper() + phrase[1:] + ".")
        else:
            sentences.append("Stay still.")
    return sentences


def describe_cameras(path):
    """The camera sheet at path (a cameras.csv, or the sheet folder that holds it) as one paragraph of directions."""
    return " ".join(frame_instructions(path))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cameras", help="a scene sheet's cameras.csv, or the sheet folder that holds it")
    args = p.parse_args(argv)
    print(describe_cameras(args.cameras))
    return 0


if __name__ == "__main__":
    sys.exit(main())
