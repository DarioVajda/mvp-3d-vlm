"""Scene sheets as prompt text: one question's sheet rendered as csv, json, markdown_kv, markdown_table or text.

The sheets are the CSVs of hpc/run_sheets.sh (docs/scene_sheets_75.md): index.csv gives each VSI-Bench question a
pipeline sheet folder (VGGT + SAM 3 + tracking, 16 uniform frames) and a gold sheet folder (VSI-Bench's labelled boxes),
each with an objects.csv. The formats are for comparing how the sheet is written, so they hold exactly the same
information: one canonical object list per question (load_objects), the same preamble per source (PREAMBLES), and
renderers that differ only in layout. cameras.csv is not used. Nothing here writes to the sheets folder.

  render_sheet(question_id, source, fmt) -> str     source: pipeline | gold; fmt: one of FORMATS

  python -m mvp.benchmarks.scene_sheet --examples experiments/2026-11-18-scene-sheet-formats/examples --question 706
  python -m mvp.benchmarks.scene_sheet --tokens Qwen/Qwen3.5-2B     # needs the tokenizer (transformers)
"""
import argparse
import csv
import io
import json
import statistics
from functools import lru_cache
from pathlib import Path

import numpy as np

from mvp.paths import SCENE_SHEETS, VGGT_DIR

SOURCES = ["pipeline", "gold"]
FORMATS = ["csv", "json", "markdown_kv", "markdown_table", "text"]
N_FRAMES = 16  # the pipeline's frames: 16 uniform over the 128 exported ones (docs/scene_sheets_75.md)
MERGE = 2  # Qwen3-VL / Qwen3.5 merge video frames in pairs and show one timestamp per pair
DECIMALS = 3
EMPTY = ("The detector did not find the objects named in the question. This does not mean they are absent from "
         "the room.")

POSITION = ["x", "y", "z"]
EXTENT = ["size_x", "size_y", "size_z"]
FRAME_FIELDS = ["first_seen", "num_frames"]  # pipeline only; gold sheets have no per-frame columns


def fields(source, volume=False):
    """The canonical field order. `size` (the product of the three extents) is left out unless volume is set."""
    return (["id", "label"] + POSITION + EXTENT + (["size"] if volume else [])
            + (FRAME_FIELDS if source == "pipeline" else []))


# The axis convention, checked against the code and the data (2026-11-18):
#  - pipeline: the world frame is the camera of exported frame 0 (adapt_vggt.py check_cameras: "world frame is frame
#    0's camera"), in OpenCV axes (scene_sheet_creator.py: x right, y down, z forward). Frame 0 of the 16 is exported
#    frame 0, and its row in cameras.csv is the origin with the identity rotation in all 75 scenes (largest deviation
#    2.4e-3). Which way is up in the room can't be read from the sheet: y is down only as seen from that first camera,
#    which is often tilted (camera_sheet_format_converter.py estimates up from the camera path instead).
#  - gold: the dataset's frame in metres, z up (sheet.json; scene_sheet_creator.py). Doors and refrigerators are
#    tallest along z in 962 of their 1,021 gold rows.
PREAMBLES = {
    "pipeline": ("Scene sheet. Positions x, y, z are object centres in the 3D reconstruction's relative units, not "
                 "metres. The origin is the camera of the first video frame: x points right, y down and z forward as "
                 "seen from that camera. size_x, size_y and size_z are each object's extent along x, y and z, in the "
                 "same units. first_seen is the video timestamp, in seconds, at which the object first appears, as "
                 "the video's timestamps show it, and num_frames is how many of the 16 video frames it appears in."),
    "gold": ("Scene sheet. Positions x, y, z are object centres in metres, with the z axis pointing up. size_x, size_y "
             "and size_z are each object's extent along x, y and z, in metres."),
}
VOLUME_NOTE = " size is the product of the three extents."


def preamble(source, volume=False):
    return PREAMBLES[source] + (VOLUME_NOTE if volume else "")


def fmt_float(value):
    """DECIMALS decimals, no scientific notation, no negative zero: -0.5 -> '-0.500', -0.0001 -> '0.000'."""
    s = f"{float(value):.{DECIMALS}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


# ---- step 1: the canonical object list ----

@lru_cache(maxsize=None)
def load_index(root=SCENE_SHEETS):
    """index.csv as {question id: row}."""
    with open(Path(root) / "index.csv", newline="") as f:
        return {int(row["id"]): row for row in csv.DictReader(f)}


def sheet_folders(question_id, source, root=SCENE_SHEETS):
    """The sheet folders index.csv lists for this question and source (one in every row of the 75-scene index)."""
    row = load_index(root)[int(question_id)]
    return [Path(root) / row[f"{source}_sheet"]]


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


def read_objects(folder, source, volume=False, vggt_root=VGGT_DIR):
    """One sheet folder's objects.csv as canonical objects: dicts of formatted strings in the order of fields()."""
    folder = Path(folder)
    with open(folder / "objects.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    if source == "pipeline" and rows:
        scene = json.loads((folder / "question.json").read_text())["scene"]
        times = frame_timestamps(folder.parents[1] / "scene", scene, Path(vggt_root))
    objects = []
    for row in rows:
        obj = {"id": f"{row['concept'].replace(' ', '_')}_{row['track_id']}", "label": row["concept"]}
        for key in POSITION + EXTENT + (["size"] if volume else []):
            obj[key] = fmt_float(row[key])
        if source == "pipeline":
            seen = [t for t in range(N_FRAMES) if row[f"frame_{t:03d}"] == "1"]
            if not seen or len(seen) != int(row["num_frames"]):
                raise ValueError(f"{folder}: {obj['id']} has num_frames {row['num_frames']} but frame flags {seen}")
            obj["first_seen"], obj["num_frames"] = times[seen[0]], str(len(seen))
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
    return (float(obj["first_seen"]),) + order if "first_seen" in obj else order


def load_objects(question_id, source, root=SCENE_SHEETS, volume=False, stats=None, vggt_root=VGGT_DIR):
    """The question's canonical object list: every sheet folder of this source merged, sorted by first_seen, label
    and track_id (gold: label and track_id). stats, a dict, counts merged folders and dropped duplicates."""
    folders = sheet_folders(question_id, source, root)
    objects, dropped = merge([read_objects(folder, source, volume, vggt_root) for folder in folders])
    if stats is not None:
        stats["merged_questions"] = stats.get("merged_questions", 0) + (len(folders) > 1)
        stats["duplicates_dropped"] = stats.get("duplicates_dropped", 0) + dropped
    return sorted(objects, key=sort_key)


# ---- the renderers, each the body that follows the preamble ----

def render_csv(objects, keys):
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")  # quotes a field only if it needs it (a comma in a label)
    writer.writerow(keys)
    writer.writerows([obj[k] for k in keys] for obj in objects)
    return out.getvalue().rstrip("\n")


def render_json(objects, keys):
    def value(key, obj):
        return json.dumps(obj[key]) if key in ("id", "label") else obj[key]  # numbers stay numbers, as formatted
    lines = ["{" + ", ".join(f"{json.dumps(k)}: {value(k, obj)}" for k in keys) + "}" for obj in objects]
    return "[\n" + ",\n".join(lines) + "\n]"


def render_markdown_kv(objects, keys):
    blocks = [f"## {obj['id']}\n```\n" + "\n".join(f"{k}: {obj[k]}" for k in keys if k != "id") + "\n```"
              for obj in objects]
    return "\n\n".join(blocks)


def render_markdown_table(objects, keys):
    if any("|" in obj[k] for obj in objects for k in keys):
        raise ValueError("a value contains |, which would break the table")
    rows = [keys, ["---"] * len(keys)] + [[obj[k] for k in keys] for obj in objects]
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def text_sentence(obj):
    """The fixed sentence template; TEXT_PATTERN in the tests parses it back."""
    article = "an" if obj["label"][0].lower() in "aeiou" else "a"
    volume = f" and volume {obj['size']}" if "size" in obj else ""
    s = (f"{obj['id']} is {article} {obj['label']} centred at x={obj['x']}, y={obj['y']}, z={obj['z']}, with extent "
         f"{obj['size_x']} along x, {obj['size_y']} along y and {obj['size_z']} along z{volume}.")
    if "first_seen" in obj:
        s += (f" It is first seen at {obj['first_seen']} seconds and is visible in {obj['num_frames']} of the "
              f"{N_FRAMES} frames.")
    return s


def render_text(objects, keys):
    return "\n".join(text_sentence(obj) for obj in objects)


RENDERERS = {"csv": render_csv, "json": render_json, "markdown_kv": render_markdown_kv,
             "markdown_table": render_markdown_table, "text": render_text}


def render_objects(objects, source, fmt, volume=False):
    """Preamble, then the objects in this format, or EMPTY (the same line in every format) if there are none."""
    body = RENDERERS[fmt](objects, fields(source, volume)) if objects else EMPTY
    return preamble(source, volume) + "\n" + body


def render_sheet(question_id, source, fmt, root=SCENE_SHEETS, volume=False, vggt_root=VGGT_DIR):
    """One question's scene sheet as prompt text. source: pipeline | gold; fmt: one of FORMATS. vggt_root holds each
    video's meta.json (its fps, for the pipeline's timestamps)."""
    if source not in SOURCES or fmt not in FORMATS:
        raise ValueError(f"source must be one of {SOURCES} and fmt one of {FORMATS}, not {source!r}, {fmt!r}")
    return render_objects(load_objects(question_id, source, root, volume, vggt_root=vggt_root), source, fmt, volume)


# ---- command line: examples and token counts ----

def write_examples(out_dir, question_id, root=SCENE_SHEETS, vggt_root=VGGT_DIR):
    """The question rendered in every format and source, one file each, plus the question itself."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    row = load_index(root)[int(question_id)]
    (out_dir / f"{question_id}_question.txt").write_text(
        f"{row['scene']} ({row['question_type']})\n{row['question']}\n{row['options']}\nground truth: "
        f"{row['ground_truth']}\n")
    for source in SOURCES:
        for fmt in FORMATS:
            (out_dir / f"{question_id}_{source}_{fmt}.txt").write_text(
                render_sheet(question_id, source, fmt, root, vggt_root=vggt_root) + "\n")
    print(f"wrote {len(SOURCES) * len(FORMATS) + 1} files to {out_dir}")


def token_table(count, root=SCENE_SHEETS, vggt_root=VGGT_DIR):
    """Tokens per rendered sheet (count: text -> number of tokens) over every question: a markdown table with the mean,
    median and max per source and format, and the question with the most."""
    ids = sorted(load_index(root))
    lines = ["| source | format | mean | median | max | question with max |", "|---|---|---|---|---|---|"]
    for source in SOURCES:
        for fmt in FORMATS:
            counts = {q: count(render_sheet(q, source, fmt, root, vggt_root=vggt_root)) for q in ids}
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
    args = ap.parse_args()
    root, vggt_root = Path(args.root), Path(args.vggt_root)
    if args.examples:
        write_examples(args.examples, args.question, root, vggt_root)
    if args.tokens:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokens)
        print(token_table(lambda text: len(tokenizer(text)["input_ids"]), root, vggt_root))
    if not (args.examples or args.tokens):
        stats, empty = {}, {s: 0 for s in SOURCES}
        for q in load_index(root):
            for source in SOURCES:
                empty[source] += not load_objects(q, source, root, stats=stats, vggt_root=vggt_root)
        print(f"{len(load_index(root))} questions; empty sheets: {empty}; {stats}")


if __name__ == "__main__":
    main()
