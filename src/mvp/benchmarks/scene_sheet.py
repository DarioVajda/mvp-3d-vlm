"""Scene sheets as prompt text: one question's objects, and for pipeline sheets its cameras, written out for the model.

The sheets are the CSVs of hpc/run_sheets.sh (docs/scene_sheets_75.md): index.csv gives each VSI-Bench question a
pipeline sheet folder (VGGT + SAM 3 + tracking, 16 uniform frames: objects.csv and cameras.csv) and a gold sheet folder
(VSI-Bench's labelled boxes: objects.csv only). The table formats are for comparing how the sheet is written, so they
hold exactly the same information: one canonical object list per question (load_objects) and one canonical camera list
(load_cameras), the same preamble per source, and renderers that differ only in layout. cameras.csv's image column is
never written. Nothing here writes to the sheets folder.

  render_sheet(question_id, source, fmt)      the objects. source: pipeline | gold; fmt: one of SHEET_FORMATS, where
                                              "narrative" is objects and cameras told as one walk through the frames
                                              (pipeline only)
  render_cameras(question_id, camera_fmt)     the pipeline's cameras. camera_fmt: one of CAMERA_FORMATS, where "text" is
                                              the compass directions of camera_sheet_format_converter.py, frame by frame
  render_scene(question_id, source, fmt, camera_fmt)   what the prompt carries: objects, then cameras (None: neither)
  build_sheet_prompt(question_id, source, fmt)  the whole text input the eval sends with that sheet

time_unit, for all of them: "seconds" (the default) writes a frame as the video timestamp the model's video input shows
for it (Qwen's processor writes one per pair of frames); "frames" writes it as its number, 1 to 16. The video's own
labels are always in seconds. A pipeline object's `visible` field gives the stretches of consecutive frames it is seen
in, "[2, 4], [6, 8]" (frames) or "[3.0, 15.5], [30.3, 41.3]" (seconds); visibility "bridged" (the default) counts a
single missed frame between two sightings as seen, "exact" doesn't, and the narrative follows the same rule.
step_order, for the camera text and the narrative: "move-turn" (the default) says each step as "move, then turn",
"turn-move" as "turn, then move". In the narrative, moves are relative to where the camera faces when it moves, so the
order can change a move's words; the camera text's compass directions don't depend on it. object_directions, for the
narrative: "floor" (the default) says where an object is along the floor from where the camera faces, plus above or
below, as the moves are measured; "camera" says it in the camera's own tilted axes. check_options says which
combinations of options are contradictory.

  python -m mvp.benchmarks.scene_sheet --examples experiments/2026-11-18-scene-sheet-formats/examples --question 706
  python -m mvp.benchmarks.scene_sheet --tokens Qwen/Qwen3.5-2B     # needs the tokenizer (transformers)
  python -m mvp.benchmarks.scene_sheet --question 706 --source pipeline --format csv --prompt
  python -m mvp.benchmarks.scene_sheet --question 706 --format narrative --time-unit frames --reasoning cot --prompt
"""
import argparse
import csv
import io
import json
import math
import re
import statistics
from functools import lru_cache
from pathlib import Path

import numpy as np

from mvp.benchmarks.vsibench import REASONING, build_prompt_parts
from mvp.openvocab_adapter import camera_sheet_format_converter as directions
from mvp.paths import SCENE_SHEETS, VGGT_DIR

SOURCES = ["pipeline", "gold"]
FORMATS = ["csv", "json", "markdown_kv", "markdown_table", "text"]  # the object sheet's layouts, both sources
NARRATIVE = "narrative"  # objects and cameras as one walk through the frames: pipeline only
SHEET_FORMATS = FORMATS + [NARRATIVE]
CAMERA_FORMATS = ["csv", "json", "markdown_kv", "markdown_table", "text"]
TIME_UNITS = ["seconds", "frames"]
VISIBILITY = ["bridged", "exact"]  # bridged: a gap of up to MAX_GAP missed frames between two sightings counts as seen
MAX_GAP = 1
STEP_ORDERS = ["move-turn", "turn-move"]  # how the camera text and the narrative word each step between two frames
OBJECT_DIRECTIONS = ["floor", "camera"]  # how the narrative says where an object is (floor_side, view_side)
N_FRAMES = 16  # the pipeline's frames: 16 uniform over the 128 exported ones (docs/scene_sheets_75.md)
MERGE = 2  # Qwen3-VL / Qwen3.5 merge video frames in pairs and show one timestamp per pair
DECIMALS = 3
EMPTY = ("The detector did not find the objects named in the question. This does not mean they are absent from "
         "the room.")

POSITION = ["x", "y", "z"]
EXTENT = ["size_x", "size_y", "size_z"]
FRAME_FIELDS = ["visible"]  # pipeline only; gold sheets have no per-frame columns
CAMERA_COLUMNS = ["x", "y", "z", "qw", "qx", "qy", "qz", "forward_x", "forward_y", "forward_z"]  # minus frame, image
FRAME_KEY = {"seconds": "time", "frames": "frame"}  # the camera rows' first field


def fields(source, volume=False):
    """The canonical field order. `size` (the product of the three extents) is left out unless volume is set."""
    return (["id", "label"] + POSITION + EXTENT + (["size"] if volume else [])
            + (FRAME_FIELDS if source == "pipeline" else []))


def camera_fields(time_unit="seconds"):
    return [FRAME_KEY[time_unit]] + CAMERA_COLUMNS


# The axis convention, checked against the code and the data (2026-11-18):
#  - pipeline: the world frame is the camera of exported frame 0 (adapt_vggt.py check_cameras: "world frame is frame
#    0's camera"), in OpenCV axes (scene_sheet_creator.py: x right, y down, z forward). Frame 0 of the 16 is exported
#    frame 0, and its row in cameras.csv is the origin with the identity rotation in all 75 scenes (largest deviation
#    2.4e-3). Which way is up in the room can't be read from the sheet: y is down only as seen from that first camera,
#    which is often tilted (camera_sheet_format_converter.py estimates up from the camera path instead).
#  - gold: the dataset's frame in metres, z up (sheet.json; scene_sheet_creator.py). Doors and refrigerators are
#    tallest along z in 962 of their 1,021 gold rows.
PIPELINE_AXES = ("Positions x, y, z are object centres in the 3D reconstruction's relative units, not metres. The "
                 "origin is the camera of the first video frame: x points right, y down and z forward as seen from that "
                 "camera. size_x, size_y and size_z are each object's extent along x, y and z, in the same units.")
FRAME_NUMBERS = ("The 16 video frames are numbered 1 to 16 in the order the video shows them. The video shows them in "
                 "pairs under one label each: frames 1 and 2 under its first label, frames 3 and 4 under the next one, "
                 "and so on.")
VISIBLE = {
    "seconds": ("visible gives the stretches of consecutive video frames in which the object is seen, each from its "
                "first frame to its last, written as the video's timestamps for them, in seconds. The video shows one "
                "timestamp per pair of frames, so a stretch within one pair starts and ends at the same time."),
    "frames": ("visible gives the stretches of consecutive video frames in which the object is seen, each from its "
               "first frame to its last, written as frame numbers. " + FRAME_NUMBERS),
}
BRIDGED_NOTE = (" The detector sometimes misses an object for a single frame, so one missed frame between two "
                "sightings is counted as seen.")
PREAMBLES = {  # with time_unit "seconds" and visibility "bridged"; preamble() gives the others
    "pipeline": "Scene sheet. " + PIPELINE_AXES + " " + VISIBLE["seconds"] + BRIDGED_NOTE,
    "gold": ("Scene sheet. Positions x, y, z are object centres in metres, with the z axis pointing up. size_x, size_y "
             "and size_z are each object's extent along x, y and z, in metres."),
}
VOLUME_NOTE = " size is the product of the three extents."

CAMERA_PREAMBLE = (
    "Camera sheet: where the camera is in each of the 16 video frames, one entry per frame. x, y, z is the camera's "
    "position in the 3D reconstruction's relative units, not metres. The origin is the camera of the first video "
    "frame: x points right, y down and z forward as seen from that camera. qw, qx, qy, qz is the camera's orientation, "
    "the unit quaternion that turns the camera's own axes (x right, y down, z forward) into these axes, and forward_x, "
    "forward_y, forward_z is the direction the camera looks.")
CAMERA_KEY = {
    "seconds": ("time is the frame's video timestamp, in seconds, as the video's timestamps show it. The video shows "
                "one timestamp per pair of frames, so two entries share each time."),
    "frames": "frame is the frame's number. " + FRAME_NUMBERS,
}
DIRECTIONS = (
    "Camera path: how the camera moves through the 16 video frames. North is the direction the camera faces in the "
    "first frame, along the floor, and east is 90° to its right; up is estimated from the camera's path. Each sentence "
    "after the first says how the camera got to its frame from the frame before: how far it moved, in the 3D "
    "reconstruction's relative units (not metres), in which direction, and how many degrees it turned left or right "
    "and up or down.")
NARRATIVE_PREAMBLE = (
    "Scene walkthrough: the camera's path through the 16 video frames, told from the camera's point of view, with the "
    "objects the detector found in each frame.")
NARRATIVE_STEPS = {
    "move-turn": ("Each sentence after the first says how the camera got to its frame from the frame before: first how "
                  "it moved, relative to where it faced before moving, then how it turned."),
    "turn-move": ("Each sentence after the first says how the camera got to its frame from the frame before: first how "
                  "it turned, then how it moved, relative to where it faces after turning."),
}
NARRATIVE_DETAILS = (
    "Forward, back, left and right are relative to the camera's facing, up and down are estimated from the camera's "
    "path, and turns are in degrees. Each object the camera sees comes with where it is as seen from the camera and how "
    "far away it is; where is left out when the object's centre is behind the camera, which happens when only part of a "
    "large object is in view. Distances and sizes are in the 3D reconstruction's relative units, not metres: an "
    "object's distance is from the camera to the object's centre, and its size is its extent along three "
    "perpendicular axes.")
LINE_LABELS = {  # the line-per-frame texts: the camera directions and the narrative
    "seconds": ("Each line starts with a video timestamp, in seconds, as the video's timestamps show it, and holds the "
                "sentences of the two frames shown under it."),
    "frames": "Each line starts with its frame's number. " + FRAME_NUMBERS,
}


def check_time_unit(time_unit):
    if time_unit not in TIME_UNITS:
        raise ValueError(f"time_unit must be one of {TIME_UNITS}, not {time_unit!r}")


def check_visibility(visibility):
    if visibility not in VISIBILITY:
        raise ValueError(f"visibility must be one of {VISIBILITY}, not {visibility!r}")


def check_step_order(step_order):
    if step_order not in STEP_ORDERS:
        raise ValueError(f"step_order must be one of {STEP_ORDERS}, not {step_order!r}")


def check_object_directions(object_directions):
    if object_directions not in OBJECT_DIRECTIONS:
        raise ValueError(f"object_directions must be one of {OBJECT_DIRECTIONS}, not {object_directions!r}")


def bridged_note(visibility):
    check_visibility(visibility)
    return BRIDGED_NOTE if visibility == "bridged" else ""


def preamble(source, volume=False, time_unit="seconds", visibility="bridged"):
    check_time_unit(time_unit)
    if source == "pipeline":
        text = "Scene sheet. " + PIPELINE_AXES + " " + VISIBLE[time_unit] + bridged_note(visibility)
    else:
        text = PREAMBLES[source]
    return text + (VOLUME_NOTE if volume else "")


def camera_preamble(time_unit="seconds"):
    check_time_unit(time_unit)
    return CAMERA_PREAMBLE + " " + CAMERA_KEY[time_unit]


def fmt_float(value):
    """DECIMALS decimals, no scientific notation, no negative zero: -0.5 -> '-0.500', -0.0001 -> '0.000'."""
    s = f"{float(value):.{DECIMALS}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


# ---- step 1: the canonical object and camera lists ----

@lru_cache(maxsize=None)
def load_index(root=SCENE_SHEETS):
    """index.csv as {question id: row}."""
    with open(Path(root) / "index.csv", newline="") as f:
        return {int(row["id"]): row for row in csv.DictReader(f)}


def sheet_folders(question_id, source, root=SCENE_SHEETS):
    """The sheet folders index.csv lists for this question and source (one in every row of the 75-scene index)."""
    row = load_index(root)[int(question_id)]
    return [Path(root) / row[f"{source}_sheet"]]


def camera_folder(question_id, root=SCENE_SHEETS):
    """The folder of the question's cameras.csv: its pipeline sheet's (gold sheets have none)."""
    return sheet_folders(question_id, "pipeline", root)[0]


@lru_cache(maxsize=None)
def frame_timestamps(scene_dir, scene, vggt_root=VGGT_DIR):
    """The timestamp the model's video input shows for each of the 16 frames, as text ('6.4').

    Qwen's processor (transformers 5.20.1, Qwen3VLProcessor, also Qwen3.5's) writes "<{t:.1f} seconds>" once per pair
    of frames, t being the mean of the pair's video frame index / fps (_calculate_timestamps). The frames are the
    sheet's source_video_frames (scene/adapter_report.json), which are the eval's --sampling uniform128 frames; fps is
    the video's, from the VGGT export's meta.json (decord's average fps)."""
    report = json.loads((Path(scene_dir) / "adapter_report.json").read_text())
    indices = report["source_video_frames"]
    if report["selected_frames"] != np.linspace(0, 127, N_FRAMES, dtype=int).tolist() or len(indices) % MERGE:
        raise ValueError(f"{scene_dir}: not the 16 uniform frames of --sampling uniform128")
    fps = json.loads((Path(vggt_root) / scene / "meta.json").read_text())["video_fps"]
    seconds = [idx / fps for idx in indices]  # in the processor's order: divide, then average, so .x5 rounds the same
    pairs = [(seconds[i] + seconds[i + MERGE - 1]) / 2 for i in range(0, len(seconds), MERGE)]
    return tuple(f"{pairs[t // MERGE]:.1f}" for t in range(len(indices)))


def frame_labels(folder, time_unit="seconds", vggt_root=VGGT_DIR):
    """How a pipeline sheet folder's 16 frames are written: their video timestamps ('6.4'; the two frames of a pair
    share one), or their numbers ('1' to '16')."""
    check_time_unit(time_unit)
    if time_unit == "frames":
        return tuple(str(t + 1) for t in range(N_FRAMES))
    folder = Path(folder)
    scene = json.loads((folder / "question.json").read_text())["scene"]
    return frame_timestamps(folder.parents[1] / "scene", scene, Path(vggt_root))


def object_id(row):
    return f"{row['concept'].replace(' ', '_')}_{row['track_id']}"


def seen_in(row):
    """The frames (0 to 15) a pipeline objects.csv row was seen in, from its frame flags."""
    return [t for t in range(N_FRAMES) if row[f"frame_{t:03d}"] == "1"]


def visible_frames(seen, visibility="bridged"):
    """The frames an object counts as visible in: those it was seen in, plus, when bridged, every gap of up to MAX_GAP
    frames between two of them: {1, 3, 4, 7} -> {1, 2, 3, 4, 7}."""
    check_visibility(visibility)
    seen = sorted(set(seen))
    frames = set(seen)
    if visibility == "bridged":
        for a, b in zip(seen, seen[1:]):
            if b - a - 1 <= MAX_GAP:
                frames.update(range(a + 1, b))
    return frames


def runs(frames):
    """Consecutive frames as (first, last) pairs: {1, 2, 3, 6} -> [(1, 3), (6, 6)]."""
    out = []
    for t in sorted(frames):
        if out and t == out[-1][1] + 1:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [tuple(r) for r in out]


RANGE = re.compile(r"\[([^,\]]+), ([^\]]+)\]")


def ranges_of(visible):
    """A visible field back to its (first, last) labels: '[2, 4], [6, 6]' -> [('2', '4'), ('6', '6')]."""
    return RANGE.findall(visible)


def read_objects(folder, source, volume=False, vggt_root=VGGT_DIR, time_unit="seconds", visibility="bridged"):
    """One sheet folder's objects.csv as canonical objects: dicts of formatted strings in the order of fields()."""
    folder = Path(folder)
    with open(folder / "objects.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    if source == "pipeline" and rows:
        labels = frame_labels(folder, time_unit, vggt_root)
    objects = []
    for row in rows:
        obj = {"id": object_id(row), "label": row["concept"]}
        for key in POSITION + EXTENT + (["size"] if volume else []):
            obj[key] = fmt_float(row[key])
        if source == "pipeline":
            seen = seen_in(row)
            if not seen or len(seen) != int(row["num_frames"]):
                raise ValueError(f"{folder}: {obj['id']} has num_frames {row['num_frames']} but frame flags {seen}")
            obj["visible"] = ", ".join(f"[{labels[a]}, {labels[b]}]"
                                       for a, b in runs(visible_frames(seen, visibility)))
        objects.append(obj)
    return objects


def merge(object_lists):
    """Objects of several sheet folders as one list, exact duplicates dropped. Returns (objects, duplicates dropped).
    Two different objects with the same id are an error: there is no rule for which one to keep."""
    merged, by_id, dropped = [], {}, 0
    for obj in (o for objects in object_lists for o in objects):
        if obj["id"] in by_id:
            if by_id[obj["id"]] != obj:
                raise ValueError(f"two different objects named {obj['id']}")
            dropped += 1
            continue
        by_id[obj["id"]] = obj
        merged.append(obj)
    return merged, dropped


def sort_key(obj):
    order = (obj["label"], int(obj["id"].rsplit("_", 1)[1]))
    return (float(ranges_of(obj["visible"])[0][0]),) + order if "visible" in obj else order


def load_objects(question_id, source, root=SCENE_SHEETS, volume=False, stats=None, vggt_root=VGGT_DIR,
                 time_unit="seconds", visibility="bridged"):
    """The question's canonical object list: every sheet folder of this source merged, sorted by the frame it is first
    seen in, label and track_id (gold: label and track_id). stats, a dict, counts merged folders and dropped
    duplicates."""
    folders = sheet_folders(question_id, source, root)
    objects, dropped = merge([read_objects(folder, source, volume, vggt_root, time_unit, visibility)
                              for folder in folders])
    if stats is not None:
        stats["merged_questions"] = stats.get("merged_questions", 0) + (len(folders) > 1)
        stats["duplicates_dropped"] = stats.get("duplicates_dropped", 0) + dropped
    return sorted(objects, key=sort_key)


def frames_seen(question_id, root=SCENE_SHEETS, visibility="bridged"):
    """{object id: the set of frames (0 to 15) it counts as visible in (visible_frames)}, over the question's pipeline
    sheet folders."""
    seen = {}
    for folder in sheet_folders(question_id, "pipeline", root):
        with open(folder / "objects.csv", newline="") as f:
            for row in csv.DictReader(f):
                seen.setdefault(object_id(row), set()).update(seen_in(row))
    return {key: visible_frames(frames, visibility) for key, frames in seen.items()}


def load_cameras(question_id, root=SCENE_SHEETS, vggt_root=VGGT_DIR, time_unit="seconds"):
    """The question's canonical camera list: one dict of formatted strings per frame, in frame order, with the fields
    of camera_fields(time_unit). cameras.csv's frame becomes the frame's label and its image column is dropped."""
    folder = camera_folder(question_id, root)
    with open(folder / "cameras.csv", newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["frame"]))
    if [int(r["frame"]) for r in rows] != list(range(N_FRAMES)):
        raise ValueError(f"{folder}: cameras.csv does not have one row for each of the {N_FRAMES} frames")
    key = FRAME_KEY[time_unit]
    labels = frame_labels(folder, time_unit, vggt_root)
    return [{key: label, **{c: fmt_float(row[c]) for c in CAMERA_COLUMNS}} for label, row in zip(labels, rows)]


# ---- the renderers, each the body that follows the preamble ----

def render_csv(objects, keys):
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")  # quotes a field only if it needs it (a comma in a label)
    writer.writerow(keys)
    writer.writerows([obj[k] for k in keys] for obj in objects)
    return out.getvalue().rstrip("\n")


def render_json(objects, keys):
    def value(key, obj):
        if key == "visible":  # a list of [first, last] pairs
            return "[" + obj[key] + "]"
        return json.dumps(obj[key]) if key in ("id", "label") else obj[key]  # numbers stay numbers, as formatted
    lines = ["{" + ", ".join(f"{json.dumps(k)}: {value(k, obj)}" for k in keys) + "}" for obj in objects]
    return "[\n" + ",\n".join(lines) + "\n]"


def render_markdown_kv(objects, keys):
    """One block per row, headed by its first field: the id of an object ('## chair_1'), or the field and its value
    for a camera ('## frame 3', '## time 2.5')."""
    def title(obj):
        return obj[keys[0]] if keys[0] == "id" else f"{keys[0]} {obj[keys[0]]}"
    blocks = [f"## {title(obj)}\n```\n" + "\n".join(f"{k}: {obj[k]}" for k in keys[1:]) + "\n```"
              for obj in objects]
    return "\n\n".join(blocks)


def render_markdown_table(objects, keys):
    if any("|" in obj[k] for obj in objects for k in keys):
        raise ValueError("a value contains |, which would break the table")
    rows = [keys, ["---"] * len(keys)] + [[obj[k] for k in keys] for obj in objects]
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def article(label):
    return "an" if label[0].lower() in "aeiou" else "a"


def join_and(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def visibility_phrase(visible, time_unit="seconds"):
    """A visible field in words: 'in frames 2 to 4 and 6', 'in frame 3', 'from 3.0 to 15.5 seconds and at 30.3
    seconds'."""
    ranges = ranges_of(visible)
    if time_unit == "frames":
        word = "frame" if len(ranges) == 1 and ranges[0][0] == ranges[0][1] else "frames"
        return f"in {word} " + join_and([a if a == b else f"{a} to {b}" for a, b in ranges])
    return join_and([f"at {a} seconds" if a == b else f"from {a} to {b} seconds" for a, b in ranges])


def text_sentence(obj, time_unit="seconds"):
    """The fixed sentence template; TEXT_PATTERN in the tests parses it back."""
    volume = f" and volume {obj['size']}" if "size" in obj else ""
    s = (f"{obj['id']} is {article(obj['label'])} {obj['label']} centred at x={obj['x']}, y={obj['y']}, z={obj['z']}, "
         f"with extent {obj['size_x']} along x, {obj['size_y']} along y and {obj['size_z']} along z{volume}.")
    if "visible" in obj:
        s += f" It is visible {visibility_phrase(obj['visible'], time_unit)}."
    return s


def render_text(objects, keys, time_unit="seconds"):
    return "\n".join(text_sentence(obj, time_unit) for obj in objects)


RENDERERS = {"csv": render_csv, "json": render_json, "markdown_kv": render_markdown_kv,
             "markdown_table": render_markdown_table, "text": render_text}


def render_objects(objects, source, fmt, volume=False, time_unit="seconds", visibility="bridged"):
    """Preamble, then the objects in this format, or EMPTY (the same line in every format) if there are none."""
    if not objects:
        body = EMPTY
    elif fmt == "text":
        body = render_text(objects, fields(source, volume), time_unit)
    else:
        body = RENDERERS[fmt](objects, fields(source, volume))
    return preamble(source, volume, time_unit, visibility) + "\n" + body


def label_lines(sentences, labels, time_unit="seconds"):
    """One sentence per frame as lines: 'Frame 3: ...', or, in seconds, one line per timestamp holding the sentences
    of the frames shown under it: 'At 2.5 seconds: ... ...'."""
    if time_unit == "frames":
        return [f"Frame {label}: {s}" for label, s in zip(labels, sentences)]
    groups = []
    for label, s in zip(labels, sentences):
        if groups and groups[-1][0] == label:
            groups[-1][1].append(s)
        else:
            groups.append((label, [s]))
    return [f"At {label} seconds: {' '.join(group)}" for label, group in groups]


def render_cameras(question_id, fmt, root=SCENE_SHEETS, vggt_root=VGGT_DIR, time_unit="seconds",
                   step_order="move-turn"):
    """The question's camera sheet as prompt text: a table format of CAMERA_FORMATS (the canonical camera list), or
    "text", the compass directions of camera_sheet_format_converter.py with each move's length, one line per frame,
    each step said in step_order."""
    if fmt not in CAMERA_FORMATS:
        raise ValueError(f"camera format must be one of {CAMERA_FORMATS}, not {fmt!r}")
    check_time_unit(time_unit)
    check_step_order(step_order)
    if fmt == "text":
        folder = camera_folder(question_id, root)
        sentences = directions.frame_instructions(folder, units=True, turn_first=step_order == "turn-move")
        if len(sentences) != N_FRAMES:
            raise ValueError(f"{folder}: cameras.csv does not have one row for each of the {N_FRAMES} frames")
        lines = label_lines(sentences, frame_labels(folder, time_unit, vggt_root), time_unit)
        return DIRECTIONS + " " + LINE_LABELS[time_unit] + "\n" + "\n".join(lines)
    cameras = load_cameras(question_id, root, vggt_root, time_unit)
    return camera_preamble(time_unit) + "\n" + RENDERERS[fmt](cameras, camera_fields(time_unit))


# ---- the narrative: objects and cameras as one walk through the frames ----

AHEAD_DEG, SLIGHTLY_DEG, SIDE_DEG = 10, 35, 100  # where an object is: in front, slightly to a side, to a side
MOVES = ["forward", "forward and to the right", "to the right", "back and to the right", "back",
         "back and to the left", "to the left", "forward and to the left"]  # clockwise from where the camera faces


def side_phrase(angle):
    """Where a point is, from its angle in degrees clockwise from where the camera faces, or None when it is more than
    SIDE_DEG off: a camera can't see behind itself, so a seen object whose centre is there (part of a large object in
    view) gets no side."""
    side = "right" if angle > 0 else "left"
    if abs(angle) < AHEAD_DEG:
        return "in front of you"
    if abs(angle) < SLIGHTLY_DEG:
        return f"ahead, slightly to your {side}"
    if abs(angle) <= SIDE_DEG:
        return f"on your {side}"
    return None


def view_side(offset):
    """side_phrase in the camera's own axes, from a point's offset in them (x right, z forward): tilted with the
    camera, and blind to up and down."""
    return side_phrase(math.degrees(math.atan2(offset[0], offset[2])))


def floor_side(offset, ahead, up):
    """Where a point is for a camera along the floor, from its offset in world axes, the camera's facing along the
    floor (camera_sheet_format_converter.facing) and the estimated up: side_phrase of its bearing from that facing,
    with " and below" / " and above" when it is more than CLIMB_DEG below or above the camera, or only "below you" /
    "above you" when it is more than VERTICAL_DEG so, or has no side. The thresholds are the moves' (walk_move)."""
    rise = float(offset @ up)
    flat = offset - rise * up
    elevation = math.degrees(math.atan2(rise, np.linalg.norm(flat)))
    vertical = ("above" if rise > 0 else "below") if abs(elevation) > directions.CLIMB_DEG else None
    if abs(elevation) > directions.VERTICAL_DEG:
        return f"{vertical} you"
    side = side_phrase(directions.bearing(flat, ahead, np.cross(ahead, up)))
    if side and vertical:
        return f"{side} and {vertical}"
    return side or (f"{vertical} you" if vertical else None)


def object_side(offset, rotation, ahead, up, object_directions="floor"):
    """Where an object is for the camera, from its offset in world axes: floor_side ("floor"), or view_side in the
    camera's own axes ("camera"; rotation is the camera's, camera to world)."""
    check_object_directions(object_directions)
    if object_directions == "floor":
        return floor_side(offset, ahead, up)
    return view_side(rotation.T @ offset)


def walk_move(step, median_step, up, ahead):
    """'move 0.812 units forward and to the left', relative to where the camera faced before it moved, or None when
    the step is too short to mention. The thresholds are camera_sheet_format_converter.py's."""
    length = float(np.linalg.norm(step))
    if median_step <= 0 or length < directions.NO_MOVE * median_step:
        return None
    rise = float(step @ up)
    flat = step - rise * up
    climb = math.degrees(math.atan2(abs(rise), np.linalg.norm(flat)))
    vertical = "up" if rise > 0 else "down"
    if climb > directions.VERTICAL_DEG:
        where = vertical
    else:
        where = MOVES[round(directions.bearing(flat, ahead, np.cross(ahead, up)) / 45) % 8]
        if climb > directions.CLIMB_DEG:
            where += f", going {vertical}"
    return f"move {fmt_float(length)} units {where}"


def object_clauses(objects, seen, t, mentioned, centre, rotation, ahead, up, object_directions="floor"):
    """What the camera sees in frame t, from its centre, rotation and facing along the floor: each object seen in it,
    then each one seen in the frame before but not in this one. Sizes are given the first time an object is seen; where
    it is (object_side) is left out when there is nothing to say."""
    clauses = []
    for obj in objects:
        if t not in seen[obj["id"]]:
            continue
        offset = np.array([float(obj[k]) for k in POSITION]) - centre
        side = object_side(offset, rotation, ahead, up, object_directions)
        where = (f" {side}," if side else ",") + f" {fmt_float(np.linalg.norm(offset))} units away"
        if t - 1 in seen[obj["id"]]:
            clauses.append(f"you still see {obj['id']}{where}")
        elif obj["id"] in mentioned:
            clauses.append(f"you see {obj['id']} again{where}")
        else:
            size = " × ".join(obj[k] for k in EXTENT)
            clauses.append(f"you see {obj['id']} ({article(obj['label'])} {obj['label']}, {size} units){where}")
            mentioned.add(obj["id"])
    clauses += [f"{obj['id']} is no longer in view" for obj in objects
                if t - 1 in seen[obj["id"]] and t not in seen[obj["id"]]]
    return clauses


def narrative_step(move, turn, step_order="move-turn"):
    """One step's words from walk_move's and turn_phrase's: 'move 0.812 units forward and turn 30° to the right'
    (move-turn) or 'turn 30° to the right and move 0.812 units forward' (turn-move); a turn alone is 'don't move and
    turn ...' / 'turn ... without moving', and neither is 'stay still'."""
    check_step_order(step_order)
    if not (move or turn):
        return "stay still"
    if step_order == "turn-move":
        if move and turn:  # a comma keeps "turn 23° to the right and 5° down, and move ..." readable
            return f"{turn}{',' if ' and ' in turn else ''} and {move}"
        return move or f"{turn} without moving"
    if move and turn:  # a comma keeps "move 0.8 units forward, going up, and turn ..." readable
        return f"{move}{',' if ', going' in move else ''} and {turn}"
    return move or f"don't move and {turn}"


def narrative_preamble(time_unit="seconds", visibility="bridged", step_order="move-turn"):
    check_step_order(step_order)
    return (" ".join([NARRATIVE_PREAMBLE, NARRATIVE_STEPS[step_order], NARRATIVE_DETAILS]) + bridged_note(visibility)
            + " " + LINE_LABELS[time_unit])


def render_narrative(question_id, root=SCENE_SHEETS, vggt_root=VGGT_DIR, time_unit="seconds", visibility="bridged",
                     step_order="move-turn", object_directions="floor"):
    """The pipeline sheet's objects and cameras as one walk, a sentence per frame: how the camera moved and turned
    since the frame before (forward, left... relative to where it faced), then what it sees: "Move 0.812 units forward
    and turn 30° to the right; you see table_1 (a table, 1.200 × 0.800 × 0.700 units) ahead, slightly to your left,
    2.100 units away; lamp_1 is no longer in view." Up is camera_sheet_format_converter.py's estimate. An object is
    seen in the frames visible_frames(visibility) gives. step_order "turn-move" says the turn first, and the move
    relative to where the camera faces after it (the frame's own facing) instead of before it (the previous frame's).
    object_directions: where an object is, along the floor and above or below ("floor", like the moves) or in the
    camera's own tilted axes ("camera")."""
    check_time_unit(time_unit)
    check_step_order(step_order)
    check_object_directions(object_directions)
    folder = camera_folder(question_id, root)
    objects = load_objects(question_id, "pipeline", root, vggt_root=vggt_root, time_unit=time_unit,
                           visibility=visibility)
    seen = frames_seen(question_id, root, visibility)
    centres, rotations = directions.read_cameras(folder)
    if len(centres) != N_FRAMES:
        raise ValueError(f"{folder}: cameras.csv does not have one row for each of the {N_FRAMES} frames")
    up = directions.estimate_up(rotations)
    facings = [directions.facing(r, up) for r in rotations]
    north, east = facings[0], np.cross(facings[0], up)
    headings = [directions.bearing(f, north, east) for f in facings]
    pitches = [math.degrees(math.asin(np.clip(r[:, 2] @ up, -1, 1))) for r in rotations]
    steps = np.diff(centres, axis=0)
    median_step = float(np.median(np.linalg.norm(steps, axis=1)))

    sentences, mentioned = [], set()
    for t in range(N_FRAMES):
        if t == 0:
            pitch = round(abs(pitches[0]))
            looking = ("straight ahead" if pitch < directions.MIN_TURN_DEG
                       else f"{pitch}° {'up' if pitches[0] > 0 else 'down'}")
            action = f"start facing forward, looking {looking}"
        else:
            ahead = facings[t] if step_order == "turn-move" else facings[t - 1]  # where the camera faces as it moves
            move = walk_move(steps[t - 1], median_step, up, ahead)
            turn = directions.turn_phrase(headings[t] - headings[t - 1], pitches[t] - pitches[t - 1])
            action = narrative_step(move, turn, step_order)
        clauses = object_clauses(objects, seen, t, mentioned, centres[t], rotations[t], facings[t], up,
                                 object_directions)
        sentences.append(action[0].upper() + action[1:] + "".join("; " + c for c in clauses) + ".")
    lines = label_lines(sentences, frame_labels(folder, time_unit, vggt_root), time_unit)
    head = narrative_preamble(time_unit, visibility, step_order) + ("" if objects else "\n" + EMPTY)
    return head + "\n" + "\n".join(lines)


# ---- one question's scene information ----

def check_options(source, fmt, camera_fmt=None, time_unit="seconds", visibility="bridged", step_order="move-turn",
                  object_directions="floor"):
    """ValueError if these sheet options are unknown or contradict each other. fmt is ignored without a source,
    visibility without pipeline objects, step_order without the camera text or the narrative, and object_directions
    without the narrative.
      - gold objects with cameras, as tables or in the narrative: gold sheets are in metres in the dataset's frame, and
        the only cameras are VGGT's, in another frame and scale, with nothing to align the two by
      - the narrative with a camera format: the narrative already tells the cameras"""
    if source is not None and source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES} or None, not {source!r}")
    if source is not None and fmt not in SHEET_FORMATS:
        raise ValueError(f"the sheet format must be one of {SHEET_FORMATS}, not {fmt!r}")
    if camera_fmt is not None and camera_fmt not in CAMERA_FORMATS:
        raise ValueError(f"the camera format must be one of {CAMERA_FORMATS} or None, not {camera_fmt!r}")
    check_time_unit(time_unit)
    check_visibility(visibility)
    check_step_order(step_order)
    check_object_directions(object_directions)
    if source == "gold" and (camera_fmt or fmt == NARRATIVE):
        raise ValueError("gold sheets have no cameras: they are in metres in the dataset's frame, and the only cameras "
                         "are VGGT's, in another frame and scale (so no camera format, and no narrative)")
    if source == "pipeline" and fmt == NARRATIVE and camera_fmt:
        raise ValueError("the narrative already tells where the camera goes: give no camera format with it")


def render_sheet(question_id, source, fmt, root=SCENE_SHEETS, volume=False, vggt_root=VGGT_DIR, time_unit="seconds",
                 visibility="bridged", step_order="move-turn", object_directions="floor"):
    """One question's objects as prompt text. source: pipeline | gold; fmt: one of SHEET_FORMATS ("narrative":
    pipeline only, with its cameras). vggt_root holds each video's meta.json (its fps, for timestamps)."""
    if source not in SOURCES or fmt not in SHEET_FORMATS:
        raise ValueError(f"source must be one of {SOURCES} and fmt one of {SHEET_FORMATS}, not {source!r}, {fmt!r}")
    check_options(source, fmt, None, time_unit, visibility, step_order, object_directions)
    if fmt == NARRATIVE:
        return render_narrative(question_id, root, vggt_root, time_unit, visibility, step_order, object_directions)
    objects = load_objects(question_id, source, root, volume, vggt_root=vggt_root, time_unit=time_unit,
                           visibility=visibility)
    return render_objects(objects, source, fmt, volume, time_unit, visibility)


def render_scene(question_id, source=None, fmt=None, camera_fmt=None, root=SCENE_SHEETS, vggt_root=VGGT_DIR,
                 time_unit="seconds", visibility="bridged", step_order="move-turn", object_directions="floor"):
    """What the prompt carries about the scene: the objects (source and fmt), then the cameras (camera_fmt), or
    either alone. None when there is neither. Contradictory options raise ValueError (check_options)."""
    check_options(source, fmt, camera_fmt, time_unit, visibility, step_order, object_directions)
    parts = ([render_sheet(question_id, source, fmt, root, vggt_root=vggt_root, time_unit=time_unit,
                           visibility=visibility, step_order=step_order, object_directions=object_directions)]
             if source else [])
    if camera_fmt:
        parts.append(render_cameras(question_id, camera_fmt, root, vggt_root, time_unit, step_order))
    return "\n".join(parts) if parts else None


def sheet_kind(source, camera_fmt=None):
    """Where the scene information comes from, for the chain-of-thought instructions (vsibench.cot_prompt): 'gold'
    (labels, in metres), 'pipeline' (computed from the video, in relative units; cameras are always this) or None."""
    return source or ("pipeline" if camera_fmt else None)


# ---- the whole prompt ----

# Qwen3.5's chat template with thinking off (enable_thinking=False), as QwenVL.generate_video applies it, with
# mvp.models.qwen_vl.SYSTEM_PROMPT. The template writes the video as <|vision_start|><|video_pad|><|vision_end|>; the
# processor (Qwen3VLProcessor.replace_video_token) then replaces <|video_pad|> with, per pair of frames, "<t seconds>"
# and that pair's video tokens between <|vision_start|> and <|vision_end|>. Checked against the tokenizer's template
# (test_scene_sheet.PromptTemplate).
SYSTEM_PROMPT = "You are a helpful assistant."
CHAT = ("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{pre}<|vision_start|>{video}<|vision_end|>{post}"
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
VIDEO_PAD = "<|video_pad|>"


def build_sheet_prompt(question_id, source, fmt, root=SCENE_SHEETS, vggt_root=VGGT_DIR, frame_tokens=None,
                       camera_fmt=None, time_unit="seconds", reasoning="direct", visibility="bridged",
                       step_order="move-turn", object_directions="floor"):
    """The text input the eval (mvp.eval.vsibench, --sampling uniform128 --nframes 16, layout sit) sends for this
    question with this sheet: chat template, the 8 "<t seconds>" labels of the video's frame pairs, the scene
    information (render_scene) after "These are frames of a video." and the question. source None and camera_fmt None
    give the video-only prompt (condition A). reasoning: one of vsibench.REASONING ("cot": think step by step, end on
    "Answer: X").

    Each frame pair's video tokens are written as one <|video_pad|>, or as "<|video_pad|> x frame_tokens", which is
    how QwenVL.render_prompt (--print-prompt) writes them, so the two can be compared. The question comes from
    index.csv, which gives the same prompt as VSI-Bench's test.jsonl for all 2,728 questions (checked 2026-11-18)."""
    row = load_index(root)[int(question_id)]
    doc = {"question_type": row["question_type"], "question": row["question"],
           "options": row["options"].split(" | ") if row["options"] else None}
    sheet = render_scene(question_id, source, fmt, camera_fmt, root, vggt_root, time_unit, visibility, step_order,
                         object_directions)
    pre, post = build_prompt_parts(doc, "sit", sheet, reasoning, sheet_kind(source, camera_fmt))
    # the video's labels are the pipeline frames' timestamps, whichever sheet goes in, and always in seconds
    times = frame_timestamps((Path(root) / row["pipeline_sheet"]).parents[1] / "scene", row["scene"], Path(vggt_root))
    pad = VIDEO_PAD if frame_tokens is None else f"{VIDEO_PAD} x {frame_tokens}"
    video = "".join(f"<{t} seconds><|vision_start|>{pad}<|vision_end|>" for t in times[::MERGE])
    return CHAT.format(system=SYSTEM_PROMPT, pre=pre, video=video, post=post)


# ---- command line: examples, token counts and prompts ----

def write_examples(out_dir, question_id, root=SCENE_SHEETS, vggt_root=VGGT_DIR, time_unit="seconds",
                   visibility="bridged", step_order="move-turn", object_directions="floor"):
    """The question rendered in every format and source, its cameras in every camera format and its narrative, one
    file each, plus the question itself. Options other than the defaults go into the file names."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    row = load_index(root)[int(question_id)]
    (out_dir / f"{question_id}_question.txt").write_text(
        f"{row['scene']} ({row['question_type']})\n{row['question']}\n{row['options']}\nground truth: "
        f"{row['ground_truth']}\n")
    texts = {f"{source}_{fmt}": render_sheet(question_id, source, fmt, root, vggt_root=vggt_root, time_unit=time_unit,
                                             visibility=visibility, step_order=step_order,
                                             object_directions=object_directions)
             for source in SOURCES for fmt in SHEET_FORMATS if source == "pipeline" or fmt != NARRATIVE}
    texts.update({f"cameras_{fmt}": render_cameras(question_id, fmt, root, vggt_root, time_unit, step_order)
                  for fmt in CAMERA_FORMATS})
    defaults = {"seconds", "bridged", "move-turn", "floor"}
    suffix = "".join(f"_{option}" for option in (time_unit, visibility, step_order, object_directions)
                     if option not in defaults)
    for name, text in texts.items():
        (out_dir / f"{question_id}_{name}{suffix}.txt").write_text(text + "\n", encoding="utf-8")
    print(f"wrote {len(texts) + 1} files to {out_dir}")


def token_table(count, root=SCENE_SHEETS, vggt_root=VGGT_DIR, time_unit="seconds", visibility="bridged",
                step_order="move-turn", object_directions="floor"):
    """Tokens per rendered sheet (count: text -> number of tokens) over every question: a markdown table with the mean,
    median and max per source and format (and for the cameras), and the question with the most."""
    ids = sorted(load_index(root))
    renders = [(source, fmt, lambda q, s=source, f=fmt: render_sheet(q, s, f, root, vggt_root=vggt_root,
                                                                     time_unit=time_unit, visibility=visibility,
                                                                     step_order=step_order,
                                                                     object_directions=object_directions))
               for source in SOURCES for fmt in (SHEET_FORMATS if source == "pipeline" else FORMATS)]
    renders += [("cameras", fmt, lambda q, f=fmt: render_cameras(q, f, root, vggt_root, time_unit, step_order))
                for fmt in CAMERA_FORMATS]
    lines = ["| source | format | mean | median | max | question with max |", "|---|---|---|---|---|---|"]
    for source, fmt, render in renders:
        counts = {q: count(render(q)) for q in ids}
        worst = max(counts, key=counts.get)
        lines.append(f"| {source} | {fmt} | {statistics.mean(counts.values()):.0f} | "
                     f"{statistics.median(counts.values()):.0f} | {counts[worst]} | {worst} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(SCENE_SHEETS), help="the sheets folder, with index.csv")
    ap.add_argument("--vggt-root", default=str(VGGT_DIR), help="the VGGT outputs, for each video's meta.json (fps)")
    ap.add_argument("--examples", help="write --question's sheet in every format and source to this folder")
    ap.add_argument("--question", type=int, default=706)
    ap.add_argument("--tokens", metavar="MODEL", help="print the token table with this model's tokenizer")
    ap.add_argument("--prompt", action="store_true", help="print --question's whole prompt (build_sheet_prompt)")
    ap.add_argument("--source", choices=SOURCES + ["none"], default="pipeline", help="for --prompt; none: no objects")
    ap.add_argument("--format", choices=SHEET_FORMATS, default="csv", help="for --prompt")
    ap.add_argument("--camera-format", choices=CAMERA_FORMATS, help="for --prompt: add the cameras in this format")
    ap.add_argument("--reasoning", choices=REASONING, default="direct", help="for --prompt")
    ap.add_argument("--time-unit", choices=TIME_UNITS, default="seconds", help="how the sheets write frames")
    ap.add_argument("--visibility", choices=VISIBILITY, default="bridged",
                    help="bridged: one missed frame between two sightings counts as seen; exact: only detections")
    ap.add_argument("--step-order", choices=STEP_ORDERS, default="move-turn",
                    help="how the camera text and the narrative word each step: move then turn, or turn then move")
    ap.add_argument("--object-directions", choices=OBJECT_DIRECTIONS, default="floor",
                    help="how the narrative says where an object is: along the floor plus above/below, or in the "
                         "camera's own tilted axes")
    args = ap.parse_args()
    root, vggt_root = Path(args.root), Path(args.vggt_root)
    if args.prompt:
        source = None if args.source == "none" else args.source
        print(build_sheet_prompt(args.question, source, args.format, root, vggt_root, camera_fmt=args.camera_format,
                                 time_unit=args.time_unit, reasoning=args.reasoning, visibility=args.visibility,
                                 step_order=args.step_order, object_directions=args.object_directions))
        return
    if args.examples:
        write_examples(args.examples, args.question, root, vggt_root, args.time_unit, args.visibility,
                       args.step_order, args.object_directions)
    if args.tokens:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokens)
        print(token_table(lambda text: len(tokenizer(text)["input_ids"]), root, vggt_root, args.time_unit,
                          args.visibility, args.step_order, args.object_directions))
    if not (args.examples or args.tokens):
        stats, empty = {}, {s: 0 for s in SOURCES}
        for q in load_index(root):
            for source in SOURCES:
                empty[source] += not load_objects(q, source, root, stats=stats, vggt_root=vggt_root)
        print(f"{len(load_index(root))} questions; empty sheets: {empty}; {stats}")


if __name__ == "__main__":
    main()
