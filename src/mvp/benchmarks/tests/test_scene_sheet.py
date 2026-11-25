"""Checks of the scene sheet renderers (mvp.benchmarks.scene_sheet). CPU only; no model is run.

Every format is parsed back and compared with the canonical object list, on small synthetic sheets and, when
data/scene_sheets_75 and the VGGT meta.json files are there, on every question of both sources. The meta.json files
are read from SCENE_SHEET_VGGT_ROOT if set, else from data/vggt. build_sheet_prompt's chat template is checked against
the Qwen3.5-2B tokenizer's when it is in the Hugging Face cache (and transformers and jinja2 are installed), and its
questions against VSI-Bench's test.jsonl (SCENE_SHEET_VSI_JSONL, else data/raw/VSI-Bench/test.jsonl) when present.

  PYTHONPATH=src python -m unittest discover -s src/mvp/benchmarks/tests -v      (from the repository root)
"""
import ast
import csv
import itertools
import json
import math
import os
import random
import re
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mvp.benchmarks import scene_sheet as ss
from mvp.paths import SCENE_SHEETS, VGGT_DIR, VSI_DIR

VGGT_ROOT = Path(os.environ.get("SCENE_SHEET_VGGT_ROOT", VGGT_DIR))
VSI_JSONL = Path(os.environ.get("SCENE_SHEET_VSI_JSONL", VSI_DIR / "test.jsonl"))
QWEN_VL = Path(__file__).parents[2] / "models" / "qwen_vl.py"
NUM = rf"-?\d+\.\d{{{ss.DECIMALS}}}"
TEXT_PATTERN = re.compile(
    rf"^(?P<id>\S+) is an? (?P<label>.+?) centred at x=(?P<x>{NUM}), y=(?P<y>{NUM}), z=(?P<z>{NUM}), with extent "
    rf"(?P<size_x>{NUM}) along x, (?P<size_y>{NUM}) along y and (?P<size_z>{NUM}) along z"
    rf"(?: and volume (?P<size>{NUM}))?\."
    rf"(?: It is visible (?P<visible>.+)\.)?$")
SECONDS_ITEM = re.compile(r"from (\S+) to (\S+) seconds|at (\S+) seconds")


def parse_visibility(phrase):
    """scene_sheet.visibility_phrase's words back to a visible field: 'in frames 2 to 4 and 6' -> '[2, 4], [6, 6]'."""
    if phrase.startswith("in frame"):
        items = re.split(r", | and ", phrase.split(" ", 2)[2])
        pairs = [item.split(" to ") if " to " in item else [item, item] for item in items]
    else:
        pairs = []
        for item in re.split(r", | and ", phrase):
            m = SECONDS_ITEM.fullmatch(item)
            assert m, item
            pairs.append([m.group(1), m.group(2)] if m.group(1) else [m.group(3)] * 2)
    return ", ".join(f"[{a}, {b}]" for a, b in pairs)


def parse(rendered, source, fmt, volume=False, time_unit="seconds", visibility="bridged"):
    """A rendered sheet back to the canonical list of {field: string}; the preamble must be the source's."""
    head, body = rendered.split("\n", 1)
    assert head == ss.preamble(source, volume, time_unit, visibility), head
    if body == ss.EMPTY:
        return []
    return parse_body(body, fmt, ss.fields(source, volume))


def parse_cameras(rendered, fmt, time_unit="seconds"):
    """A rendered camera table back to the canonical camera list; the preamble must be the camera sheet's."""
    head, body = rendered.split("\n", 1)
    assert head == ss.camera_preamble(time_unit), head
    return parse_body(body, fmt, ss.camera_fields(time_unit))


def parse_body(body, fmt, keys):
    if fmt == "csv":
        rows = list(csv.DictReader(body.splitlines()))
        assert all(list(row) == keys for row in rows), "the canonical fields, in order"
        return rows
    if fmt == "json":
        # numbers are read back as their text, so "-0.500" stays "-0.500"
        objects = json.loads(body, parse_float=str, parse_int=str)
        strings = sum(k in ("id", "label") for k in keys)
        assert body.count('": "') == strings * len(objects), "only id and label are strings; numbers are not quoted"
        for obj in objects:
            if "visible" in obj:  # a list of [first, last] number pairs
                assert all(len(pair) == 2 for pair in obj["visible"]), obj["visible"]
                obj["visible"] = ", ".join(f"[{a}, {b}]" for a, b in obj["visible"])
        return objects
    if fmt == "markdown_kv":
        objects = []
        for block in body.split("\n\n"):
            lines = block.splitlines()
            assert lines[0].startswith("## ") and lines[1] == "```" and lines[-1] == "```", block
            title = lines[0][3:]
            if keys[0] != "id":  # a camera: '## frame 3'
                assert title.startswith(keys[0] + " "), title
                title = title.split(" ", 1)[1]
            objects.append({keys[0]: title, **dict(line.split(": ", 1) for line in lines[2:-1])})
        return objects
    if fmt == "markdown_table":
        rows = [line[2:-2].split(" | ") for line in body.splitlines()]
        assert all(line.startswith("| ") and line.endswith(" |") for line in body.splitlines())
        assert rows[1] == ["---"] * len(rows[0]), rows[1]
        return [dict(zip(rows[0], row, strict=True)) for row in rows[2:]]
    if fmt == "text":
        objects = []
        for line in body.splitlines():
            m = TEXT_PATTERN.match(line)
            assert m, line
            obj = {k: v for k, v in m.groupdict().items() if v is not None}
            if "visible" in obj:
                obj["visible"] = parse_visibility(obj["visible"])
            objects.append(obj)
        return objects
    raise ValueError(fmt)


def camera_row(t, position, heading):
    """A cameras.csv row of a level camera at position (in frame 0's camera axes) that has turned heading degrees to
    the right of frame 0's camera: a rotation about the y (down) axis."""
    h = math.radians(heading)
    return [t, f"/cluster/frames/{t:03d}.jpg", *position, math.cos(h / 2), 0, math.sin(h / 2), 0,
            math.sin(h), 0, math.cos(h)]


# frame 0 three units behind the origin; forward twice, the second time turning 30° right; still; turn back left;
# then a unit to the right per frame. NARRATIVE is what render_narrative says about it with the sheet's objects.
TRAJECTORY = ([((0, 0, -3), 0), ((0, 0, -2), 0), ((0, 0, -1), 30), ((0, 0, -1), 30), ((0, 0, -1), 0)]
              + [((t - 4, 0, -1), 0) for t in range(5, 16)])
NARRATIVE = [
    "Frame 1: Start facing forward, looking straight ahead.",
    "Frame 2: Move 1.000 units forward; you see chair_1 (a chair, 0.400 × 0.900 × 0.450 units) ahead, slightly to "
    "your left, 2.077 units away; you see coffee_table_1 (a coffee table, 1.000 × 1.000 × 1.000 units) in front of "
    "you, 2.000 units away; you see door_1 (a door, 0.310 × 2.000 × 0.120 units) ahead, slightly to your left, 2.791 "
    "units away.",
    "Frame 3: Move 1.000 units forward and turn 30° to the right; you still see door_1 on your left, 1.946 units "
    "away; chair_1 is no longer in view; coffee_table_1 is no longer in view.",
    "Frame 4: Stay still; you see door_2 (a door, 0.300 × 1.900 × 0.100 units) ahead, slightly to your left, 3.162 "
    "units away; door_1 is no longer in view.",
    "Frame 5: Don't move and turn 30° to the left; you still see door_2 ahead, slightly to your right, 3.162 units "
    "away.",
    "Frame 6: Move 1.000 units to the right; you see door_1 again on your left, 2.694 units away; door_2 is no longer "
    "in view.",
    "Frame 7: Move 1.000 units to the right; door_1 is no longer in view.",
] + [f"Frame {t}: Move 1.000 units to the right." for t in range(8, 17)]
DIRECTIONS = (["Frame 1: Start facing north, looking straight ahead.", "Frame 2: Go 1.000 units north.",
               "Frame 3: Go 1.000 units north and turn 30° to the right.", "Frame 4: Stay still.",
               "Frame 5: Turn 30° to the left."] + [f"Frame {t}: Go 1.000 units east." for t in range(6, 17)])


def triples(objects):
    return {(obj["id"], k, v) for obj in objects for k, v in obj.items()}


def write_sheet(folder, rows, scene=None):
    """objects.csv (with frame columns if scene is given, as in a pipeline sheet) and question.json."""
    folder.mkdir(parents=True)
    header = ["object_id", "concept", "track_id", "x", "y", "z", "size_x", "size_y", "size_z", "size"]
    if scene:
        header += ["num_frames"] + [f"frame_{t:03d}" for t in range(16)]
        (folder / "question.json").write_text(json.dumps({"scene": scene}))
    with open(folder / "objects.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def flags(*seen):
    return [len(seen)] + [int(t in seen) for t in range(16)]


class Synthetic(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name) / "sheets"
        self.vggt = Path(self.tmp.name) / "vggt"
        pipe, gold = root / "s__1" / "subset-uniform16", root / "s__1" / "gold"
        write_sheet(pipe / "sheets" / "chair+door", [
            [0, "door", 2, "1.0004", "-0.0004", "2", "0.3", "1.9", "0.1", "0.057"] + flags(3, 4),
            [1, "chair", 1, "-0.5", "0.25", "1e-05", "0.4", "0.9", "0.45", "0.162"] + flags(1),
            [2, "door", 1, "-1.2364", "0.1", "0.5", "0.31", "2.0", "0.12", "0.0744"] + flags(1, 2, 5),
            [3, "coffee table", 1, "0", "0", "0", "1", "1", "1", "1"] + flags(1),
        ], scene="s/1")
        write_sheet(gold / "sheets" / "chair+door", [
            [0, "door", 1, "-4.4", "-1.0", "-0.5", "0.7", "0.7", "0.9", "0.441"],
            [1, "armchair", 1, "2", "3", "1", "0.8", "0.8", "0.9", "0.576"],
        ])
        write_sheet(pipe / "sheets" / "room", [], scene="s/1")
        write_sheet(gold / "sheets" / "room", [])
        for folder in ("chair+door", "room"):
            with open(pipe / "sheets" / folder / "cameras.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame", "image", "x", "y", "z", "qw", "qx", "qy", "qz", "forward_x", "forward_y",
                            "forward_z"])
                w.writerows(camera_row(t, *view) for t, view in enumerate(TRAJECTORY))
        # video frame t * 30 at 30 fps is second t, so the pair (2k, 2k + 1) shows (2k + 0.5) seconds
        (pipe / "scene").mkdir()
        (pipe / "scene" / "adapter_report.json").write_text(json.dumps({
            "selected_frames": [0, 8, 16, 25, 33, 42, 50, 59, 67, 76, 84, 93, 101, 110, 118, 127],
            "source_video_frames": [30 * t for t in range(16)]}))
        (self.vggt / "s" / "1").mkdir(parents=True)
        (self.vggt / "s" / "1" / "meta.json").write_text(json.dumps({"video_fps": 30.0}))
        with open(root / "index.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "scene", "question_type", "question", "options", "ground_truth", "prompts",
                        "pipeline_sheet", "gold_sheet", "pipeline_ok", "gold_ok"])
            w.writerow([1, "s/1", "object_counting", "How many?", "", "2", "chair, door",
                        "s__1/subset-uniform16/sheets/chair+door", "s__1/gold/sheets/chair+door", True, True])
            w.writerow([2, "s/1", "room_size_estimation", "Room?", "", "9", "room",
                        "s__1/subset-uniform16/sheets/room", "s__1/gold/sheets/room", True, True])
        self.root = root

    def tearDown(self):
        self.tmp.cleanup()

    def render(self, q, source, fmt, volume=False):
        return ss.render_sheet(q, source, fmt, self.root, volume, vggt_root=self.vggt)

    def test_canonical_list(self):
        objects = ss.load_objects(1, "pipeline", self.root, vggt_root=self.vggt)
        self.assertEqual([o["id"] for o in objects], ["chair_1", "coffee_table_1", "door_1", "door_2"])
        self.assertEqual(list(objects[0]), ss.fields("pipeline"))
        self.assertEqual(objects[0], {"id": "chair_1", "label": "chair", "x": "-0.500", "y": "0.250", "z": "0.000",
                                      "size_x": "0.400", "size_y": "0.900", "size_z": "0.450",
                                      "visible": "[0.5, 0.5]"})
        self.assertEqual(objects[1]["label"], "coffee table")  # the id has underscores, the label doesn't
        self.assertEqual((objects[2]["x"], objects[3]["y"]), ("-1.236", "0.000"))  # never -0.000
        self.assertEqual(objects[2]["visible"], "[0.5, 2.5], [4.5, 4.5]")  # frames 1-2 and 5: two missed, not bridged
        self.assertEqual(objects[3]["visible"], "[2.5, 4.5]")  # frame 3 is in the pair (2, 3), frame 4 in (4, 5)
        gold = ss.load_objects(1, "gold", self.root, vggt_root=self.vggt)
        self.assertEqual([o["id"] for o in gold], ["armchair_1", "door_1"])
        self.assertNotIn("visible", gold[0])

    def test_round_trip_and_equality(self):
        for q in (1, 2):
            for source in ss.SOURCES:
                for volume in (False, True):
                    canonical = ss.load_objects(q, source, self.root, volume, vggt_root=self.vggt)
                    for fmt in ss.FORMATS:
                        self.assertEqual(parse(self.render(q, source, fmt, volume), source, fmt, volume), canonical,
                                         (q, source, fmt, volume))

    def test_empty_sheet(self):
        for source in ss.SOURCES:
            for fmt in ss.FORMATS:
                self.assertEqual(self.render(2, source, fmt), ss.preamble(source) + "\n" + ss.EMPTY)
        # the same line in every format, worded so that an empty sheet doesn't read as "the objects aren't there"
        self.assertEqual(ss.EMPTY, "The detector did not find the objects named in the question. This does not mean "
                                   "they are absent from the room.")

    def test_formats(self):
        self.assertEqual(self.render(1, "gold", "csv").splitlines()[1:],
                         ["id,label,x,y,z,size_x,size_y,size_z",
                          "armchair_1,armchair,2.000,3.000,1.000,0.800,0.800,0.900",
                          "door_1,door,-4.400,-1.000,-0.500,0.700,0.700,0.900"])
        self.assertEqual(self.render(1, "gold", "json").splitlines()[3],
                         '{"id": "door_1", "label": "door", "x": -4.400, "y": -1.000, "z": -0.500, "size_x": 0.700, '
                         '"size_y": 0.700, "size_z": 0.900}')
        self.assertEqual(self.render(1, "gold", "markdown_table").splitlines()[1:4],
                         ["| id | label | x | y | z | size_x | size_y | size_z |",
                          "| --- | --- | --- | --- | --- | --- | --- | --- |",
                          "| armchair_1 | armchair | 2.000 | 3.000 | 1.000 | 0.800 | 0.800 | 0.900 |"])
        self.assertEqual(self.render(1, "pipeline", "text").splitlines()[2],
                         "coffee_table_1 is a coffee table centred at x=0.000, y=0.000, z=0.000, with extent 1.000 "
                         "along x, 1.000 along y and 1.000 along z. It is visible at 0.5 seconds.")
        self.assertIn("door_1 is a door centred at x=-1.236, y=0.100, z=0.500, with extent 0.310 along x, 2.000 along "
                      "y and 0.120 along z. It is visible from 0.5 to 2.5 seconds and at 4.5 seconds.",
                      self.render(1, "pipeline", "text"))
        self.assertEqual(self.render(1, "pipeline", "csv").splitlines()[4],
                         'door_1,door,-1.236,0.100,0.500,0.310,2.000,0.120,"[0.5, 2.5], [4.5, 4.5]"')
        self.assertIn('"size_z": 0.120, "visible": [[0.5, 2.5], [4.5, 4.5]]}', self.render(1, "pipeline", "json"))
        self.assertIn("visible: [0.5, 2.5], [4.5, 4.5]\n", self.render(1, "pipeline", "markdown_kv"))
        self.assertIn("| 0.120 | [0.5, 2.5], [4.5, 4.5] |", self.render(1, "pipeline", "markdown_table"))
        self.assertIn("armchair_1 is an armchair", self.render(1, "gold", "text"))

    def test_timestamps_follow_the_processor(self):
        """The same arithmetic as transformers' Qwen3VLProcessor._calculate_timestamps, on uneven indices."""
        report = self.root / "s__1" / "subset-uniform16" / "scene" / "adapter_report.json"
        indices = [0, 182, 364, 568, 750, 955, 1137, 1342, 1524, 1729, 1911, 2116, 2298, 2503, 2685, 2890]
        report.write_text(json.dumps({**json.loads(report.read_text()), "source_video_frames": indices}))
        ss.frame_timestamps.cache_clear()
        times = ss.frame_timestamps(report.parent, "s/1", self.vggt)
        self.assertEqual(times[:4], ("3.0", "3.0", "15.5", "15.5"))  # (0 + 182) / 2 / 30 = 3.03; (364 + 568) / 60
        ss.frame_timestamps.cache_clear()

    def test_merge(self):
        a = ss.read_objects(self.root / "s__1" / "gold" / "sheets" / "chair+door", "gold")
        merged, dropped = ss.merge([a, a])
        self.assertEqual((merged, dropped), (a, 2))
        with self.assertRaises(ValueError):
            ss.merge([a, [dict(a[0], x="9.000")]])

    def test_build_sheet_prompt(self):
        prompt = ss.build_sheet_prompt(1, "pipeline", "csv", self.root, self.vggt)
        video = "".join(f"<{2 * k}.5 seconds><|vision_start|><|video_pad|><|vision_end|>" for k in range(8))
        self.assertEqual(prompt, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                         f"<|vision_start|>{video}<|vision_end|>These are frames of a video.\n"
                         f"{self.render(1, 'pipeline', 'csv')}\nHow many?\n"
                         "Please answer the question using a single word or phrase.<|im_end|>\n"
                         "<|im_start|>assistant\n<think>\n\n</think>\n\n")
        # the gold sheet in the same place, the video's labels unchanged; no sheet for A
        gold = ss.build_sheet_prompt(1, "gold", "json", self.root, self.vggt)
        self.assertEqual(gold, prompt.replace(self.render(1, "pipeline", "csv"), self.render(1, "gold", "json")))
        self.assertEqual(ss.build_sheet_prompt(1, None, None, self.root, self.vggt),
                         prompt.replace(self.render(1, "pipeline", "csv") + "\n", ""))
        # with the frame's token count, as QwenVL.render_prompt (--print-prompt) writes it
        self.assertIn("<0.5 seconds><|vision_start|><|video_pad|> x 64<|vision_end|><2.5 seconds>",
                      ss.build_sheet_prompt(1, "pipeline", "csv", self.root, self.vggt, frame_tokens=64))
        with self.assertRaises(ValueError):
            ss.build_sheet_prompt(1, "pipeline", "yaml", self.root, self.vggt)

    def test_build_sheet_prompt_without_video(self):
        """--no-video: the sheet and the question, with neither the video nor its sentence."""
        with_video = ss.build_sheet_prompt(1, "gold", "csv", self.root, self.vggt)
        prompt = ss.build_sheet_prompt(1, "gold", "csv", self.root, self.vggt, video=False)
        self.assertEqual(prompt, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                         f"{self.render(1, 'gold', 'csv')}\nHow many?\n"
                         "Please answer the question using a single word or phrase.<|im_end|>\n"
                         "<|im_start|>assistant\n<think>\n\n</think>\n\n")
        self.assertNotIn("vision", prompt)
        self.assertIn(self.render(1, "gold", "csv") + "\nHow many?", with_video)  # the same text after the video
        relations = ss.build_sheet_prompt(1, "gold_relations", "csv", self.root, self.vggt, video=False)
        self.assertEqual(relations, prompt.replace(self.render(1, "gold", "csv"),
                                                   self.render(1, "gold_relations", "csv")))

    def test_special_characters(self):
        self.assertEqual(ss.render_csv([{"id": "a,b_1", "label": "a,b"}], ["id", "label"]), 'id,label\n"a,b_1","a,b"')
        with self.assertRaises(ValueError):
            ss.render_markdown_table([{"id": "a_1", "label": "a|b"}], ["id", "label"])

    # ---- frames as frame numbers ----

    def test_frames_time_unit(self):
        objects = ss.load_objects(1, "pipeline", self.root, vggt_root=self.vggt, time_unit="frames")
        self.assertEqual([(o["id"], o["visible"]) for o in objects],
                         [("chair_1", "[2, 2]"), ("coffee_table_1", "[2, 2]"), ("door_1", "[2, 3], [6, 6]"),
                          ("door_2", "[4, 5]")])
        for q in (1, 2):
            for source in ss.SOURCES:
                canonical = ss.load_objects(q, source, self.root, vggt_root=self.vggt, time_unit="frames")
                for fmt in ss.FORMATS:
                    rendered = ss.render_sheet(q, source, fmt, self.root, vggt_root=self.vggt, time_unit="frames")
                    self.assertEqual(parse(rendered, source, fmt, time_unit="frames"), canonical, (q, source, fmt))
                    self.assertNotIn("second", rendered)
        text = ss.render_sheet(1, "pipeline", "text", self.root, vggt_root=self.vggt, time_unit="frames")
        self.assertIn("It is visible in frame 2.", text)
        self.assertIn("It is visible in frames 2 to 3 and 6.", text)
        self.assertIn("It is visible in frames 4 to 5.", text)
        self.assertEqual(ss.preamble("pipeline"), ss.PREAMBLES["pipeline"])
        with self.assertRaises(ValueError):
            ss.render_sheet(1, "pipeline", "csv", self.root, vggt_root=self.vggt, time_unit="minutes")

    # ---- visibility ranges ----

    def test_visible_frames_and_runs(self):
        self.assertEqual(ss.visible_frames([1, 3, 4, 7]), {1, 2, 3, 4, 7})  # one missed frame bridged, three not
        self.assertEqual(ss.visible_frames([1, 3, 4, 7], "exact"), {1, 3, 4, 7})
        self.assertEqual(ss.runs({1, 2, 3, 6}), [(1, 3), (6, 6)])
        self.assertEqual(ss.ranges_of("[2.5, 4.5], [6.5, 6.5]"), [("2.5", "4.5"), ("6.5", "6.5")])
        self.assertEqual(ss.visibility_phrase("[1, 1], [3, 4], [8, 8]", "frames"), "in frames 1, 3 to 4 and 8")
        self.assertEqual(ss.visibility_phrase("[0.5, 2.5]"), "from 0.5 to 2.5 seconds")
        with self.assertRaises(ValueError):
            ss.visible_frames([1], "fuzzy")

    def test_bridging(self):
        """A lamp seen in frames 1, 3, 4 and 8 (0, 2, 3 and 7 from 0): bridged, frame 2 counts as seen."""
        sheet = self.root / "s__1" / "subset-uniform16" / "sheets" / "chair+door"
        with open(sheet / "objects.csv", "a", newline="") as f:
            csv.writer(f).writerow([4, "lamp", 1, "0", "0", "1", "0.2", "0.5", "0.2", "0.02"] + flags(0, 2, 3, 7))
        expected = {("frames", "bridged"): "[1, 4], [8, 8]", ("frames", "exact"): "[1, 1], [3, 4], [8, 8]",
                    ("seconds", "bridged"): "[0.5, 2.5], [6.5, 6.5]",
                    ("seconds", "exact"): "[0.5, 0.5], [2.5, 2.5], [6.5, 6.5]"}
        for (time_unit, visibility), visible in expected.items():
            objects = ss.load_objects(1, "pipeline", self.root, vggt_root=self.vggt, time_unit=time_unit,
                                      visibility=visibility)
            lamp = next(o for o in objects if o["id"] == "lamp_1")
            self.assertEqual(lamp["visible"], visible)
            # sorted by where the first range starts: in seconds, frames 1 and 2 share 0.5, and chair < lamp
            self.assertEqual(objects[0]["id"], "lamp_1" if time_unit == "frames" else "chair_1")
            for fmt in ss.FORMATS:
                rendered = ss.render_sheet(1, "pipeline", fmt, self.root, vggt_root=self.vggt, time_unit=time_unit,
                                           visibility=visibility)
                self.assertEqual(parse(rendered, "pipeline", fmt, time_unit=time_unit, visibility=visibility),
                                 objects, (fmt, time_unit, visibility))
                self.assertEqual(ss.BRIDGED_NOTE in rendered, visibility == "bridged")
        text = ss.render_sheet(1, "pipeline", "text", self.root, vggt_root=self.vggt, time_unit="frames")
        self.assertIn("lamp_1 is a lamp centred at x=0.000, y=0.000, z=1.000, with extent 0.200 along x, 0.500 along "
                      "y and 0.200 along z. It is visible in frames 1 to 4 and 8.", text)
        self.assertIn("It is visible from 0.5 to 2.5 seconds and at 6.5 seconds.", self.render(1, "pipeline", "text"))
        # the narrative counts the same frames as seen
        bridged = ss.render_sheet(1, "pipeline", ss.NARRATIVE, self.root, vggt_root=self.vggt, time_unit="frames")
        exact = ss.render_sheet(1, "pipeline", ss.NARRATIVE, self.root, vggt_root=self.vggt, time_unit="frames",
                                visibility="exact")
        self.assertIn(ss.BRIDGED_NOTE, bridged.splitlines()[0])
        self.assertNotIn(ss.BRIDGED_NOTE, exact.splitlines()[0])
        self.assertIn("you still see lamp_1", bridged.splitlines()[2])  # frame 2
        self.assertIn("lamp_1 is no longer in view", exact.splitlines()[2])
        self.assertIn("you see lamp_1 again", exact.splitlines()[3])
        self.assertIn("lamp_1 is no longer in view", bridged.splitlines()[5])  # frame 5, after frame 4

    # ---- the camera sheet ----

    def test_camera_round_trip(self):
        for time_unit in ss.TIME_UNITS:
            canonical = ss.load_cameras(1, self.root, self.vggt, time_unit)
            self.assertEqual(len(canonical), 16)
            self.assertEqual(list(canonical[0]), ss.camera_fields(time_unit))
            for fmt in ss.FORMATS:
                rendered = ss.render_cameras(1, fmt, self.root, self.vggt, time_unit)
                self.assertNotIn("image", rendered)
                self.assertNotIn(".jpg", rendered)
                if fmt != "text":
                    self.assertEqual(parse_cameras(rendered, fmt, time_unit), canonical, (fmt, time_unit))

    def test_camera_formats(self):
        frames = ss.render_cameras(1, "csv", self.root, self.vggt, "frames").splitlines()
        self.assertEqual(frames[1:5], [
            "frame,x,y,z,qw,qx,qy,qz,forward_x,forward_y,forward_z",
            "1,0.000,0.000,-3.000,1.000,0.000,0.000,0.000,0.000,0.000,1.000",
            "2,0.000,0.000,-2.000,1.000,0.000,0.000,0.000,0.000,0.000,1.000",
            "3,0.000,0.000,-1.000,0.966,0.000,0.259,0.000,0.500,0.000,0.866"])
        self.assertEqual(ss.render_cameras(1, "csv", self.root, self.vggt).splitlines()[2:4],
                         ["0.5,0.000,0.000,-3.000,1.000,0.000,0.000,0.000,0.000,0.000,1.000",
                          "0.5,0.000,0.000,-2.000,1.000,0.000,0.000,0.000,0.000,0.000,1.000"])  # one time per pair
        self.assertEqual(ss.render_cameras(1, "markdown_kv", self.root, self.vggt, "frames").splitlines()[1:4],
                         ["## frame 1", "```", "x: 0.000"])
        self.assertEqual(ss.render_cameras(1, "json", self.root, self.vggt, "frames").splitlines()[2],
                         '{"frame": 1, "x": 0.000, "y": 0.000, "z": -3.000, "qw": 1.000, "qx": 0.000, "qy": 0.000, '
                         '"qz": 0.000, "forward_x": 0.000, "forward_y": 0.000, "forward_z": 1.000},')
        with self.assertRaises(ValueError):
            ss.render_cameras(1, "narrative", self.root, self.vggt)

    def test_camera_directions(self):
        rendered = ss.render_cameras(1, "text", self.root, self.vggt, "frames").splitlines()
        self.assertEqual(rendered[0], ss.DIRECTIONS + " " + ss.LINE_LABELS["frames"])
        self.assertEqual(rendered[1:], DIRECTIONS)
        # in seconds, one line per timestamp, with the two frames under it
        seconds = ss.render_cameras(1, "text", self.root, self.vggt).splitlines()
        self.assertEqual(seconds[1:3], [
            "At 0.5 seconds: Start facing north, looking straight ahead. Go 1.000 units north.",
            "At 2.5 seconds: Go 1.000 units north and turn 30° to the right. Stay still."])
        self.assertEqual(len(seconds), 9)

    # ---- the narrative ----

    def test_narrative(self):
        rendered = ss.render_sheet(1, "pipeline", ss.NARRATIVE, self.root, vggt_root=self.vggt, time_unit="frames")
        lines = rendered.splitlines()
        self.assertEqual(lines[0], " ".join([ss.NARRATIVE_PREAMBLE, ss.NARRATIVE_STEPS["move-turn"],
                                             ss.NARRATIVE_DETAILS]) + ss.BRIDGED_NOTE + " " + ss.LINE_LABELS["frames"])
        self.assertEqual(lines[1:], NARRATIVE)
        seconds = ss.render_sheet(1, "pipeline", ss.NARRATIVE, self.root, vggt_root=self.vggt).splitlines()
        self.assertEqual(len(seconds), 9)
        self.assertEqual(seconds[1], "At 0.5 seconds: " + NARRATIVE[0][len("Frame 1: "):] + " "
                         + NARRATIVE[1][len("Frame 2: "):])
        # nothing found: the walk, with the same line as the other formats
        empty = ss.render_sheet(2, "pipeline", ss.NARRATIVE, self.root, vggt_root=self.vggt, time_unit="frames")
        self.assertEqual(empty.splitlines()[1], ss.EMPTY)
        self.assertEqual(empty.splitlines()[3], "Frame 2: Move 1.000 units forward.")
        with self.assertRaises(ValueError):
            ss.render_sheet(1, "gold", ss.NARRATIVE, self.root, vggt_root=self.vggt)

    def test_view_side(self):
        self.assertEqual(ss.view_side([0, 0, 1]), "in front of you")
        self.assertEqual(ss.view_side([0.3, 0, 1]), "ahead, slightly to your right")
        self.assertEqual(ss.view_side([-1, 0, 0.2]), "on your left")
        self.assertIsNone(ss.view_side([1, 0, -1]))  # behind the camera: no side

    def test_floor_side(self):
        """World axes as in the sheets (x right, y down, z forward); up is -y and the camera faces z along the floor."""
        up, ahead = np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])
        rug = np.array([0.8, 1.4, 0.5])  # 0.8 to the right, 1.4 below, 0.5 ahead: 58° right, 56° down
        self.assertEqual(ss.floor_side(rug, ahead, up), "on your right and below")
        # a camera looking 40° down sees it near the middle of its view: 32° right in its own axes, height ignored
        p = math.radians(40)
        down_40 = np.column_stack([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
        self.assertEqual(ss.view_side(down_40.T @ rug), "ahead, slightly to your right")
        self.assertEqual(ss.floor_side(np.array([0.0, 0.2, 2.0]), ahead, up), "in front of you")  # 6° down: level
        self.assertEqual(ss.floor_side(np.array([0.0, -1.5, 1.0]), ahead, up), "in front of you and above")
        self.assertEqual(ss.floor_side(np.array([0.1, 2.0, 0.1]), ahead, up), "below you")  # steeper than 70°
        self.assertEqual(ss.floor_side(np.array([0.0, 1.5, -1.0]), ahead, up), "below you")  # behind, so no side
        self.assertIsNone(ss.floor_side(np.array([0.0, 0.0, -2.0]), ahead, up))  # behind and level: nothing
        self.assertEqual(ss.floor_side(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), up),
                         "on your left")  # facing x, the point is 90° to the left
        self.assertIsNone(ss.floor_side(np.array([-1.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), up))  # 135°: behind
        with self.assertRaises(ValueError):
            ss.object_side(rug, down_40, ahead, up, "image")

    def test_object_directions_in_the_narrative(self):
        """Frame 2 looks 40° down at a rug on the floor: along the floor it is on the right and below; in the camera's
        tilted axes it is ahead, slightly to the right. The preamble is the same either way."""
        sheet = self.root / "s__1" / "subset-uniform16" / "sheets" / "chair+door"
        with open(sheet / "objects.csv", "a", newline="") as f:
            csv.writer(f).writerow([4, "rug", 1, "0.8", "1.4", "-1.5", "1", "0.02", "1", "0.02"] + flags(1))
        with open(sheet / "cameras.csv", newline="") as f:
            rows = list(csv.reader(f))
        a = math.radians(-20)  # a rotation of -40° about x: the camera looks 40° down
        rows[2][5:12] = [math.cos(a), math.sin(a), 0, 0, 0, math.sin(math.radians(40)), math.cos(math.radians(40))]
        with open(sheet / "cameras.csv", "w", newline="") as f:
            csv.writer(f).writerows(rows)
        floor, camera = (ss.render_sheet(1, "pipeline", ss.NARRATIVE, self.root, vggt_root=self.vggt,
                                         time_unit="frames", object_directions=d).splitlines()
                         for d in ("floor", "camera"))
        rug = "you see rug_1 (a rug, 1.000 × 0.020 × 1.000 units) {}, 1.688 units away"
        self.assertIn(rug.format("on your right and below"), floor[2])
        self.assertIn(rug.format("ahead, slightly to your right"), camera[2])
        self.assertTrue(floor[2].startswith("Frame 2: Move 1.000 units forward and turn 40° down; "), floor[2])
        self.assertEqual(floor[0], camera[0])
        self.assertEqual(floor[0], ss.narrative_preamble("frames"))

    def test_object_behind_the_camera(self):
        """A box seen in frames 7 and 8 whose centre is behind the camera both times: its distance, but no side."""
        sheet = self.root / "s__1" / "subset-uniform16" / "sheets" / "chair+door"
        with open(sheet / "objects.csv", "a", newline="") as f:
            csv.writer(f).writerow([4, "box", 1, "2", "0", "-3", "0.5", "0.5", "0.5", "0.125"] + flags(6, 7))
        lines = ss.render_sheet(1, "pipeline", ss.NARRATIVE, self.root, vggt_root=self.vggt,
                                time_unit="frames").splitlines()
        self.assertEqual(lines[7:10], [
            "Frame 7: Move 1.000 units to the right; you see box_1 (a box, 0.500 × 0.500 × 0.500 units), 2.000 units "
            "away; door_1 is no longer in view.",
            "Frame 8: Move 1.000 units to the right; you still see box_1, 2.236 units away.",
            "Frame 9: Move 1.000 units to the right; box_1 is no longer in view."])
        self.assertNotIn("behind you", "\n".join(lines[1:]))

    def test_step_order(self):
        """Turn first: the move is said relative to where the camera faces after turning, so in frame 3, after a 30°
        turn to the right, the same step forward is 'forward and to the left'."""
        lines = ss.render_sheet(1, "pipeline", ss.NARRATIVE, self.root, vggt_root=self.vggt, time_unit="frames",
                                step_order="turn-move").splitlines()
        expected = list(NARRATIVE)
        expected[2] = ("Frame 3: Turn 30° to the right and move 1.000 units forward and to the left; you still see "
                       "door_1 on your left, 1.946 units away; chair_1 is no longer in view; coffee_table_1 is no "
                       "longer in view.")
        expected[4] = ("Frame 5: Turn 30° to the left without moving; you still see door_2 ahead, slightly to your "
                       "right, 3.162 units away.")
        self.assertEqual(lines[0], ss.narrative_preamble("frames", step_order="turn-move"))
        self.assertIn(ss.NARRATIVE_STEPS["turn-move"], lines[0])
        self.assertEqual(lines[1:], expected)
        # the compass directions are the same either way; only the words' order changes
        compass = ss.render_cameras(1, "text", self.root, self.vggt, "frames", "turn-move").splitlines()
        self.assertEqual(compass[1:], DIRECTIONS[:2] + ["Frame 3: Turn 30° to the right and go 1.000 units north."]
                         + DIRECTIONS[3:])
        self.assertEqual(ss.narrative_step("move 1.000 units forward", "turn 23° to the right and 5° down",
                                           "turn-move"),
                         "turn 23° to the right and 5° down, and move 1.000 units forward")
        self.assertEqual(ss.narrative_step(None, None, "turn-move"), "stay still")
        with self.assertRaises(ValueError):
            ss.render_cameras(1, "text", self.root, self.vggt, step_order="sideways")

    # ---- every combination of options ----

    def test_scene_combines_objects_and_cameras(self):
        """The cameras come after the objects, in the objects' format; never alone, with gold or with the narrative."""
        for fmt in ss.FORMATS:
            self.assertEqual(ss.render_scene(1, "pipeline", fmt, True, self.root, self.vggt),
                             self.render(1, "pipeline", fmt) + "\n" + ss.render_cameras(1, fmt, self.root, self.vggt))
        self.assertEqual(ss.render_scene(1, "pipeline", "csv", False, self.root, self.vggt),
                         self.render(1, "pipeline", "csv"))
        self.assertIsNone(ss.render_scene(1, None, None, False, self.root, self.vggt))
        for source, fmt in [(None, "csv"), ("gold", "csv"), ("pipeline", ss.NARRATIVE)]:
            with self.assertRaises(ValueError):
                ss.render_scene(1, source, fmt, True, self.root, self.vggt)

    def test_every_combination(self):
        """Each combination of options gives a prompt, unless it is one of the contradictory ones, which raise."""
        from mvp.benchmarks import vsibench
        for q, source, fmt, cameras, time_unit, reasoning, visibility, step_order, object_directions in \
                itertools.product((1, 2), [None] + ss.SHEET_SOURCES, [None] + ss.SHEET_FORMATS, [False, True],
                                  ss.TIME_UNITS, vsibench.REASONING, ss.VISIBILITY, ss.STEP_ORDERS,
                                  ss.OBJECT_DIRECTIONS):
            if object_directions != "floor" and fmt != ss.NARRATIVE:
                continue  # only the narrative says where objects are
            options = (q, source, fmt, cameras, time_unit, reasoning, visibility, step_order, object_directions)
            contradictory = (cameras and source != "pipeline") or source is not None and (
                fmt is None or (source != "pipeline" and fmt == ss.NARRATIVE) or (fmt == ss.NARRATIVE and cameras))
            # nearby needs a sheet made on more frames than the VLM's 16; the synthetic sheets are 16-frame ones
            contradictory = contradictory or (visibility == "nearby" and source == "pipeline")
            if contradictory:
                with self.assertRaises(ValueError, msg=options):
                    ss.build_sheet_prompt(q, source, fmt, self.root, self.vggt, cameras=cameras,
                                          time_unit=time_unit, reasoning=reasoning, visibility=visibility,
                                          step_order=step_order, object_directions=object_directions)
                continue
            prompt = ss.build_sheet_prompt(q, source, fmt, self.root, self.vggt, cameras=cameras,
                                           time_unit=time_unit, reasoning=reasoning, visibility=visibility,
                                           step_order=step_order, object_directions=object_directions)
            scene = ss.render_scene(q, source, fmt, cameras, self.root, self.vggt, time_unit, visibility,
                                    step_order, object_directions)
            text = prompt.split("<|vision_end|><|vision_end|>", 1)[1]  # after the video
            self.assertTrue(text.startswith("These are frames of a video.\n" + (scene + "\n" if scene else "")),
                            options)
            self.assertNotIn("None", text, options)
            self.assertEqual("directly" in text or "single word" in text, reasoning == "direct", options)
            self.assertEqual('"Answer: X"' in text, reasoning == "cot", options)
            self.assertEqual("scene information" in text, reasoning == "cot" and scene is not None, options)
            if scene is not None:
                self.assertEqual("second" in scene, time_unit == "seconds" and source == "pipeline", options)
                self.assertEqual("Frame 1:" in scene, time_unit == "frames"
                                 and (fmt == ss.NARRATIVE or (cameras and fmt == "text")), options)
                self.assertEqual("Camera sheet" in scene, cameras and fmt != "text", options)  # the camera tables
                self.assertEqual("Camera path" in scene, cameras and fmt == "text", options)  # the compass text
                self.assertEqual(ss.BRIDGED_NOTE in scene, source == "pipeline" and visibility in ("bridged", "auto"),
                                 options)  # auto is bridged on a 16-frame sheet
                self.assertNotIn(ss.NEARBY_NOTE, scene, options)
                self.assertNotIn("first_seen", scene, options)
                self.assertNotIn("num_frames", scene, options)
                self.assertNotIn("behind you", scene, options)
                narrative = source == "pipeline" and fmt == ss.NARRATIVE  # frame 5 turns without moving
                self.assertEqual("Don't move and turn" in scene, narrative and step_order == "move-turn", options)
                self.assertEqual("without moving" in scene, narrative and step_order == "turn-move", options)


def dense_flags(*seen):
    """num_frames and the 128 frame flags of a dense sheet's row (frames of the 128)."""
    return [len(seen)] + [int(t in seen) for t in range(ss.N_CANDIDATES)]


def vlm(*frames):
    """The frames of the 128 that are the VLM's frames with these positions (0 to 15)."""
    return [ss.VLM_FRAMES[j] for j in frames]


class Dense(unittest.TestCase):
    """Sheets made on all 128 VGGT frames (a dense sheet): their frames mapped onto the VLM's 16 when the prompt is
    written. The fixture is Synthetic's scene as a dense sheet: the same objects, seen in the same VLM frames (and only
    there, so every rule gives the 16-frame sheet's frames), and 128 cameras whose 16 VLM rows are Synthetic's."""

    def setUp(self):
        Synthetic.setUp(self)  # the 16-frame fixture in self.root, to compare with
        self.legacy = self.root
        root = Path(self.tmp.name) / "dense"
        pipe = root / "s__1" / "all"
        shutil.copytree(self.legacy / "s__1" / "gold", root / "s__1" / "gold")
        rows = {
            "chair+door": [
                [0, "door", 2, "1.0004", "-0.0004", "2", "0.3", "1.9", "0.1", "0.057"] + dense_flags(*vlm(3, 4)),
                [1, "chair", 1, "-0.5", "0.25", "1e-05", "0.4", "0.9", "0.45", "0.162"] + dense_flags(*vlm(1)),
                [2, "door", 1, "-1.2364", "0.1", "0.5", "0.31", "2.0", "0.12", "0.0744"] + dense_flags(*vlm(1, 2, 5)),
                [3, "coffee table", 1, "0", "0", "0", "1", "1", "1", "1"] + dense_flags(*vlm(1)),
            ],
            "room": [],
        }
        for folder, folder_rows in rows.items():
            sheet = pipe / "sheets" / folder
            sheet.mkdir(parents=True)
            (sheet / "question.json").write_text(json.dumps({"scene": "s/1"}))
            with open(sheet / "objects.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["object_id", "concept", "track_id", "x", "y", "z", "size_x", "size_y", "size_z", "size",
                            "num_frames"] + [f"frame_{t:03d}" for t in range(ss.N_CANDIDATES)])
                w.writerows(folder_rows)
            with open(sheet / "cameras.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame", "image", "x", "y", "z", "qw", "qx", "qy", "qz", "forward_x", "forward_y",
                            "forward_z"])
                for t in range(ss.N_CANDIDATES):  # between the VLM's frames, a camera far off and turned
                    view = TRAJECTORY[ss.VLM_FRAMES.index(t)] if t in ss.VLM_FRAMES else ((5, -2, t), 3 * t)
                    w.writerow(camera_row(t, *view))
        # the VLM frames' video frames are the 16-frame fixture's, so the timestamps are the same; the others between
        source = np.interp(range(ss.N_CANDIDATES), ss.VLM_FRAMES, [30 * j for j in range(ss.N_FRAMES)])
        (pipe / "scene").mkdir()
        (pipe / "scene" / "adapter_report.json").write_text(json.dumps({
            "selected_frames": list(range(ss.N_CANDIDATES)), "source_video_frames": [int(v) for v in source]}))
        with open(self.legacy / "index.csv", newline="") as f:
            index = list(csv.reader(f))
        with open(root / "index.csv", "w", newline="") as f:
            csv.writer(f).writerows([[c.replace("subset-uniform16", "all") for c in r] for r in index])
        self.root = root
        self.frames = ss.SheetFrames(range(ss.N_CANDIDATES), [int(v) for v in source])

    def add_lamp(self, *seen):
        with open(self.root / "s__1" / "all" / "sheets" / "chair+door" / "objects.csv", "a", newline="") as f:
            csv.writer(f).writerow([4, "lamp", 1, "0", "0", "1", "0.2", "0.5", "0.2", "0.02"] + dense_flags(*seen))

    def lamp(self, **options):
        objects = ss.load_objects(1, "pipeline", self.root, vggt_root=self.vggt, **options)
        return next(o for o in objects if o["id"] == "lamp_1")["visible"]

    def test_the_vlm_frames(self):
        self.assertEqual(ss.VLM_FRAMES, (0, 8, 16, 25, 33, 42, 50, 59, 67, 76, 84, 93, 101, 110, 118, 127))
        from mvp.reconstruction.sa_sampling import uniform_subset
        self.assertEqual(list(ss.VLM_FRAMES), list(uniform_subset(ss.N_CANDIDATES, ss.N_FRAMES)))  # the eval's
        self.assertEqual((self.frames.vlm, self.frames.dense), (ss.VLM_FRAMES, True))
        legacy = ss.folder_frames(self.legacy / "s__1" / "subset-uniform16" / "sheets" / "room")
        self.assertEqual((legacy.vlm, legacy.dense), (tuple(range(16)), False))
        self.assertEqual(self.frames.vlm_video_frames, legacy.vlm_video_frames)
        with self.assertRaises(ValueError):  # a sheet without the VLM's frames can't be mapped onto them
            ss.SheetFrames(range(0, 128, 2), range(64))

    def test_nearby_rule(self):
        def seen(*frames, visibility="nearby"):
            return ss.vlm_frames_seen(list(frames), self.frames, visibility)
        self.assertEqual(seen(8), {1})  # seen in the frame itself
        self.assertEqual(seen(6, 7, 9, 10), {1})  # two of the four before and two of the four after
        self.assertEqual(seen(4, 5, 11, 12), {1})  # anywhere in the four
        self.assertEqual(seen(7, 9, 10, 16), {2})  # one before is not enough (frame 16 counts on its own)
        self.assertEqual(seen(1, 2), {0})  # frame 0 has nothing before it: only after counts
        self.assertEqual(seen(125, 126), {15})  # and frame 127 nothing after it
        self.assertEqual(seen(*range(0, 128)), set(range(16)))
        # seen only between two VLM frames: the nearest of them for each sighting (the earlier when as near)
        self.assertEqual(seen(11, 12, 13), {1, 2})
        self.assertEqual(seen(12), {1})
        # exact and bridged on a dense sheet: the VLM frames themselves, bridged across one missed VLM frame
        self.assertEqual(seen(6, 7, 9, 10, visibility="exact"), {1})  # no VLM frame: the nearest
        self.assertEqual(seen(8, 25, visibility="exact"), {1, 3})
        self.assertEqual(seen(8, 25, visibility="bridged"), {1, 2, 3})
        self.assertEqual(seen(8, 25, visibility="auto"), {1, 3})  # nearby
        self.assertEqual((ss.resolve_visibility("auto", True), ss.resolve_visibility("auto", False)),
                         ("nearby", "bridged"))
        with self.assertRaises(ValueError):
            ss.resolve_visibility("nearby", False)
        with self.assertRaises(ValueError):
            ss.bridged_note("auto")

    def test_visible_ranges(self):
        self.add_lamp(6, 7, 9, 10, 23, 24, 26, 27, 59)  # around VLM frames 1 and 3, and at 7
        self.assertEqual(self.lamp(time_unit="frames"), "[2, 2], [4, 4], [8, 8]")
        self.assertEqual(self.lamp(time_unit="frames", visibility="exact"), "[8, 8]")
        self.assertEqual(self.lamp(), "[0.5, 0.5], [2.5, 2.5], [6.5, 6.5]")  # the 16-frame sheet's labels

    def test_an_object_seen_only_between_vlm_frames_is_kept(self):
        self.add_lamp(11, 12, 13)
        self.assertEqual(self.lamp(time_unit="frames"), "[2, 3]")
        self.assertEqual(self.lamp(), "[0.5, 2.5]")
        narrative = ss.render_sheet(1, "pipeline", ss.NARRATIVE, self.root, vggt_root=self.vggt, time_unit="frames")
        self.assertIn("you see lamp_1", narrative.splitlines()[2])

    def test_exact_renders_as_the_16_frame_sheet(self):
        """With the same objects seen in the same VLM frames, a dense sheet reads exactly as the 16-frame one: objects,
        cameras (only their 16 rows count, not the others between), the camera text and the narrative; and so does
        auto (nearby here), but for its own sentence about visibility."""
        for visibility in ("exact", "bridged"):
            for time_unit in ss.TIME_UNITS:
                for fmt in ss.SHEET_FORMATS:
                    for cameras in ((False, True) if fmt != ss.NARRATIVE else (False,)):
                        a, b = (ss.render_scene(1, "pipeline", fmt, cameras, root, self.vggt, time_unit, visibility)
                                for root in (self.root, self.legacy))
                        self.assertEqual(a, b, (visibility, time_unit, fmt, cameras))
        for fmt in ss.SHEET_FORMATS:
            dense = ss.render_scene(1, "pipeline", fmt, False, self.root, self.vggt)
            self.assertIn(ss.NEARBY_NOTE, dense)
            self.assertNotIn(ss.BRIDGED_NOTE, dense)
            legacy = ss.render_scene(1, "pipeline", fmt, False, self.legacy, self.vggt, visibility="exact")
            note = ss.preamble("pipeline", visibility="nearby") if fmt != ss.NARRATIVE else ss.narrative_preamble(
                visibility="nearby")
            plain = ss.preamble("pipeline", visibility="exact") if fmt != ss.NARRATIVE else ss.narrative_preamble(
                visibility="exact")
            self.assertEqual(dense, legacy.replace(plain, note, 1), fmt)
        self.assertEqual(ss.frame_timestamps(self.root / "s__1" / "all" / "scene", "s/1", self.vggt),
                         ss.frame_timestamps(self.legacy / "s__1" / "subset-uniform16" / "scene", "s/1", self.vggt))
        self.assertEqual(ss.build_sheet_prompt(1, "pipeline", "csv", self.root, self.vggt, visibility="exact"),
                         ss.build_sheet_prompt(1, "pipeline", "csv", self.legacy, self.vggt, visibility="exact"))
        self.assertEqual(len(ss.load_cameras(1, self.root, self.vggt)), 16)

    def test_round_trip(self):
        self.add_lamp(6, 7, 9, 10)
        for visibility in ("auto", "exact", "bridged"):
            objects = ss.load_objects(1, "pipeline", self.root, vggt_root=self.vggt, visibility=visibility)
            for fmt in ss.FORMATS:
                rendered = ss.render_sheet(1, "pipeline", fmt, self.root, vggt_root=self.vggt, visibility=visibility)
                self.assertEqual(parse(rendered, "pipeline", fmt, visibility=ss.resolve_visibility(visibility, True)),
                                 objects, (fmt, visibility))

    def test_the_eval_checks_the_video_frames(self):
        from types import SimpleNamespace
        from mvp.eval import vsibench as eval_vsibench
        args = SimpleNamespace(sheets_root=str(self.root))
        eval_vsibench.check_sheet_frames([{"id": 1}, {"id": 2}], [30 * j for j in range(16)], args)
        with self.assertRaises(SystemExit):
            eval_vsibench.check_sheet_frames([{"id": 1}], [30 * j + 1 for j in range(16)], args)


# the first camera's axes (x right, y down, z forward) to an upright frame's (x right, y forward, z up)
TO_UPRIGHT = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def quaternion(R):
    """A rotation matrix as a unit quaternion (w, x, y, z), w >= 0."""
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return w, x, y, z


class Upright(unittest.TestCase):
    """Sheets in the upright frame (question.json "axes": "upright"): Synthetic's scene turned into it (z up, y the
    first camera's forward). The texts say so, and since up is then known (z), the camera text and the narrative are
    the 16-frame sheet's, but for their sentences about the axes and the order of each size's sides."""

    def setUp(self):
        Synthetic.setUp(self)
        self.legacy = self.root
        root = Path(self.tmp.name) / "upright"
        shutil.copytree(self.legacy, root)
        for folder in (root / "s__1" / "subset-uniform16" / "sheets").iterdir():
            record = json.loads((folder / "question.json").read_text())
            (folder / "question.json").write_text(json.dumps({**record, "axes": "upright"}))
            with open(folder / "objects.csv", newline="") as f:
                reader = csv.DictReader(f)
                fields, rows = reader.fieldnames, list(reader)
            for r in rows:
                x, y, z = TO_UPRIGHT @ [float(r[k]) for k in "xyz"]
                r.update(x=repr(float(x)), y=repr(float(y)), z=repr(float(z)), size_y=r["size_z"], size_z=r["size_y"])
            with open(folder / "objects.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fields)
                w.writeheader()
                w.writerows(rows)
            with open(folder / "cameras.csv", newline="") as f:
                reader = csv.DictReader(f)
                fields, rows = reader.fieldnames, list(reader)
            from mvp.openvocab_adapter import camera_sheet_format_converter as cs
            for r in rows:
                R = TO_UPRIGHT @ cs.quaternion_to_rotation(*(float(r[k]) for k in ("qw", "qx", "qy", "qz")))
                position = TO_UPRIGHT @ [float(r[k]) for k in "xyz"]
                r.update({k: repr(float(v)) for k, v in zip(("x", "y", "z"), position)})
                r.update({k: repr(float(v)) for k, v in zip(("qw", "qx", "qy", "qz"), quaternion(R))})
                r.update({f"forward_{k}": repr(float(v)) for k, v in zip("xyz", R[:, 2])})
            with open(folder / "cameras.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fields)
                w.writeheader()
                w.writerows(rows)
        self.root = root

    def test_the_texts_say_the_axes(self):
        sheet = ss.render_sheet(1, "pipeline", "csv", self.root, vggt_root=self.vggt)
        self.assertTrue(sheet.startswith("Scene sheet. " + ss.upright_text(ss.PIPELINE_AXES)))
        self.assertIn(ss.UPRIGHT_AXES, sheet)
        self.assertNotIn("y down", sheet.splitlines()[0])
        self.assertIn("door_2,door,1.000,2.000,0.000,", sheet)  # z = -y: 0.0004 rounds to 0.000, never -0.000
        self.assertEqual(ss.question_axes(1, self.root), "upright")
        self.assertEqual(ss.question_axes(1, self.legacy), "camera0")  # no "axes": every sheet before the option
        cameras = ss.render_cameras(1, "csv", self.root, self.vggt)
        self.assertIn(ss.UPRIGHT_AXES, cameras.splitlines()[0])
        self.assertIn("(x right, y down, z forward)", cameras.splitlines()[0])  # the camera's own axes stay OpenCV's
        priors = ss.preamble("pipeline", metric_scale="priors", axes="upright")
        self.assertIn("approximately in metres", priors)
        self.assertIn(ss.UPRIGHT_AXES, priors)
        self.assertEqual(ss.render_sheet(1, "gold", "csv", self.root, vggt_root=self.vggt),
                         ss.render_sheet(1, "gold", "csv", self.legacy, vggt_root=self.vggt))  # gold: unchanged
        with self.assertRaises(ValueError):
            ss.preamble("pipeline", axes="sideways")

    def test_with_the_floors_up_the_camera_text_and_narrative_are_the_same(self):
        for time_unit in ss.TIME_UNITS:
            a, b = (ss.render_cameras(1, "text", root, self.vggt, time_unit).splitlines()
                    for root in (self.root, self.legacy))
            self.assertEqual(a[0], ss.upright_text(b[0]))
            self.assertIn("up is perpendicular to the floor", a[0])
            self.assertEqual(a[1:], b[1:])
            sizes = re.compile(r"[\d.]+ × [\d.]+ × [\d.]+")  # the sides' order follows the axes
            a, b = (ss.render_sheet(1, "pipeline", ss.NARRATIVE, root, vggt_root=self.vggt, time_unit=time_unit)
                    .splitlines() for root in (self.root, self.legacy))
            self.assertEqual(a[0], ss.narrative_preamble(time_unit, "bridged", axes="upright"))
            self.assertIn("up and down are perpendicular to the floor", a[0])
            self.assertEqual([sizes.sub("S", line) for line in a[1:]], [sizes.sub("S", line) for line in b[1:]])


class MetricScale(unittest.TestCase):
    """metric_scale "priors": the pipeline's objects approximately in metres, the scale from typical object sizes
    (mvp.benchmarks.scene_scale). The synthetic scene's largest rows: door 2.0 units, chair 0.9, coffee table 1.0 (no
    prior)."""
    def setUp(self):
        Synthetic.setUp(self)  # its fixture, without running its tests again
        from mvp.benchmarks import scene_scale
        self.sc = scene_scale
        with open(self.root / "index.csv", "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([3, "s/1", "object_size_estimation", "How long is the door?", "", "200", "door",
                        "s__1/subset-uniform16/sheets/chair+door", "s__1/gold/sheets/chair+door", True, True])
            w.writerow([4, "s/1", "object_size_estimation", "How long is the coffee table?", "", "100", "coffee table",
                        "s__1/subset-uniform16/sheets/chair+door", "s__1/gold/sheets/chair+door", True, True])
        self.clear()

    def clear(self):
        for f in (ss.load_index, self.sc.scene_longest, self.sc.question_scale):
            f.cache_clear()

    def tearDown(self):
        self.clear()
        self.tmp.cleanup()

    def k(self, q):
        return self.sc.question_scale(q, str(self.root))

    def test_scale(self):
        both = self.sc.CALIBRATION * float(np.median([2.0 / 2.0, 0.85 / 0.9]))
        self.assertAlmostEqual(self.k(1)["k"], both)  # counting: door and chair
        self.assertEqual(self.k(1)["classes"], ["chair", "door"])
        door = self.k(3)  # an object-size question about the door: the door is left out
        self.assertEqual((door["classes"], door["left_out"]), (["chair"], ["door"]))
        self.assertAlmostEqual(door["k"], self.sc.CALIBRATION * 0.85 / 0.9)
        self.assertAlmostEqual(self.k(4)["k"], both)  # the coffee table has no prior: nothing to leave out
        rows = list(csv.DictReader(open(self.root / "s__1/subset-uniform16/sheets/chair+door/objects.csv")))
        with open(self.root / "s__1/subset-uniform16/sheets/chair+door/objects.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(r for r in rows if r["concept"] != "chair")
        self.clear()
        only_door = self.k(3)  # no prior class left: the fallback
        self.assertEqual((only_door["k"], only_door["fallback"]), (self.sc.FALLBACK, True))

    def test_scale_rows_of_unmerged_sheets(self):
        """Merged sheets keep the unmerged rows' sizes in SCALE_ROWS, and the scale comes from those."""
        (self.root / "s__1/subset-uniform16" / self.sc.SCALE_ROWS).write_text(
            json.dumps({"longest": {"door": 1.0, "chair": 0.85}}))
        self.clear()
        self.assertEqual(self.sc.scene_longest(str(self.root), "s/1"), {"door": 1.0, "chair": 0.85})
        self.assertAlmostEqual(self.k(1)["k"], self.sc.CALIBRATION * float(np.median([2.0 / 1.0, 0.85 / 0.85])))
        (self.root / "s__1/subset-uniform16" / self.sc.SCALE_ROWS).unlink()
        self.clear()
        self.assertEqual(self.sc.scene_longest(str(self.root), "s/1")["door"], 2.0)  # the rows again

    def test_render(self):
        plain = ss.load_objects(1, "pipeline", self.root, vggt_root=self.vggt)
        text = ss.render_sheet(1, "pipeline", "csv", self.root, vggt_root=self.vggt, metric_scale="priors")
        self.assertTrue(text.startswith("Scene sheet. Positions x, y, z are object centres, approximately in metres, "
                                        "estimated from typical object sizes."))
        self.assertIn("approximately in metres.", text.split("\n", 1)[0])
        self.assertNotIn("relative units", text.split("\n", 1)[0])
        rows = {r["id"]: r for r in csv.DictReader(text.split("\n", 1)[1].splitlines())}
        k = self.k(1)["k"]
        for o in plain:
            for key in ss.POSITION + ss.EXTENT:
                self.assertAlmostEqual(float(rows[o["id"]][key]), float(o[key]) * k, delta=0.002)
            self.assertEqual(rows[o["id"]]["visible"], o["visible"])
        # without the option, nothing changes
        self.assertEqual(ss.render_sheet(1, "pipeline", "csv", self.root, vggt_root=self.vggt),
                         ss.render_sheet(1, "pipeline", "csv", self.root, vggt_root=self.vggt, metric_scale=None))

    def test_cameras_in_metres(self):
        """The camera tables: positions times the objects' scale, orientations and view directions unchanged."""
        k = self.k(1)["k"]
        plain, metres = (ss.render_cameras(1, "csv", self.root, self.vggt, "frames", metric_scale=m).split("\n")
                         for m in (None, "priors"))
        self.assertEqual(metres[0], ss.camera_preamble("frames", "priors"))
        self.assertIn("approximately in metres, estimated from typical object sizes", metres[0])
        self.assertNotIn("relative units", metres[0])
        self.assertNotEqual(ss.CAMERA_PREAMBLE_PRIORS, ss.CAMERA_PREAMBLE)
        self.assertEqual(metres[1], plain[1])  # the header
        for a, b in zip(plain[2:], metres[2:]):
            a, b = a.split(","), b.split(",")
            self.assertEqual(a[0], b[0])  # the frame
            for x, y in zip(a[1:4], b[1:4]):  # x, y, z
                self.assertAlmostEqual(float(y), float(x) * k, delta=0.002)
            self.assertEqual(a[4:], b[4:])  # quaternion, forward
        scene = ss.render_scene(1, "pipeline", "csv", True, self.root, self.vggt, "frames", metric_scale="priors")
        self.assertEqual(scene, ss.render_sheet(1, "pipeline", "csv", self.root, vggt_root=self.vggt,
                                                time_unit="frames", metric_scale="priors") + "\n" + "\n".join(metres))

    def assert_scaled_text(self, plain, metres, k):
        """plain and metres the same text apart from 'units' -> 'metres' and every length (a number with three
        decimals) times k; degrees and frame numbers unchanged."""
        number = re.compile(r"\d+\.\d{3}")
        self.assertNotIn(" units", metres)
        self.assertEqual(number.sub("N", metres), number.sub("N", plain).replace(" units", " metres"))
        a, b = number.findall(plain), number.findall(metres)
        self.assertEqual(len(a), len(b))
        self.assertTrue(a)
        for x, y in zip(a, b):
            self.assertAlmostEqual(float(y), float(x) * k, delta=0.01)

    def test_camera_text_in_metres(self):
        k = self.k(1)["k"]
        plain, metres = (ss.render_cameras(1, "text", self.root, self.vggt, "frames", metric_scale=m).split("\n", 1)
                         for m in (None, "priors"))
        self.assertEqual(metres[0], ss.DIRECTIONS_PRIORS + " " + ss.LINE_LABELS["frames"])
        self.assertNotEqual(ss.DIRECTIONS_PRIORS, ss.DIRECTIONS)
        self.assert_scaled_text(plain[1], metres[1], k)
        self.assertIn("Go ", metres[1])

    def test_narrative_in_metres(self):
        k = self.k(1)["k"]
        for step_order in ss.STEP_ORDERS:
            for object_directions in ss.OBJECT_DIRECTIONS:
                plain, metres = (ss.render_sheet(1, "pipeline", ss.NARRATIVE, self.root, vggt_root=self.vggt,
                                                 time_unit="frames", step_order=step_order,
                                                 object_directions=object_directions, metric_scale=m).split("\n", 1)
                                 for m in (None, "priors"))
                self.assertEqual(metres[0], ss.narrative_preamble("frames", step_order=step_order,
                                                                  metric_scale="priors"))
                self.assertIn(ss.NARRATIVE_DETAILS_PRIORS, metres[0])
                self.assertNotIn("relative units", metres[0])
                self.assert_scaled_text(plain[1], metres[1], k)
        self.assertNotEqual(ss.NARRATIVE_DETAILS_PRIORS, ss.NARRATIVE_DETAILS)

    def test_options_and_prompt(self):
        for source, fmt, cameras in [("gold", "csv", False), ("gold_relations", "csv", False)]:
            with self.assertRaises(ValueError, msg=(source, fmt, cameras)):
                ss.check_options(source, fmt, cameras, metric_scale="priors")
        for fmt, cameras in [("csv", True), ("text", True), ("markdown_table", True), (ss.NARRATIVE, False)]:
            ss.check_options("pipeline", fmt, cameras, metric_scale="priors")  # the cameras and the narrative too
        with self.assertRaises(ValueError):
            ss.check_options("pipeline", "csv", metric_scale="metres")
        prompt = ss.build_sheet_prompt(3, "pipeline", "csv", self.root, self.vggt, reasoning="cot",
                                       metric_scale="priors")
        self.assertIn("computed automatically from the video and can be wrong", prompt)  # the pipeline's warning
        self.assertNotIn("not metres: to answer", prompt)  # but not "work out the scale yourself"
        unscaled = ss.build_sheet_prompt(3, "pipeline", "csv", self.root, self.vggt, reasoning="cot")
        self.assertIn("relative units", unscaled)


class Relations(unittest.TestCase):
    """gold_relations: the gold sheet, then closest-point distances between the objects the question names."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = root = Path(self.tmp.name) / "sheets"
        self.vggt = Path(self.tmp.name) / "vggt"
        # boxes in metres: the table spans x 0..2, the door x 3..3.2 (1 m from the table along x), the tv's box is
        # 2 m above the table's and 0.5 m beside it along y, the chairs overlap the table's corner or stand 4 m away
        write_sheet(root / "s__1" / "gold" / "sheets" / "all", [
            [0, "table", 1, "1", "0", "0.5", "2", "1", "1", "2"],
            [1, "door", 1, "3.1", "0", "1", "0.2", "1", "2", "0.4"],
            [2, "tv", 1, "1", "1.5", "3.5", "1", "1", "1", "1"],
            [3, "chair", 1, "2", "0.5", "0.5", "1", "1", "1", "1"],
            [4, "chair", 2, "-4.5", "0", "0.5", "1", "1", "1", "1"],
        ])
        pipe = root / "s__1" / "subset-uniform16"
        write_sheet(pipe / "sheets" / "all", [], scene="s/1")
        (pipe / "scene").mkdir()
        (pipe / "scene" / "adapter_report.json").write_text(json.dumps({
            "selected_frames": [0, 8, 16, 25, 33, 42, 50, 59, 67, 76, 84, 93, 101, 110, 118, 127],
            "source_video_frames": [30 * t for t in range(16)]}))
        (self.vggt / "s" / "1").mkdir(parents=True)
        (self.vggt / "s" / "1" / "meta.json").write_text(json.dumps({"video_fps": 30.0}))
        rows = [(1, "object_abs_distance", "What is the distance between the table and the door?", "", "1",
                 "table, door"),
                (2, "object_rel_distance", "Which of these (door, tv, sofa, chair) is closest to the table?",
                 "A. door | B. tv | C. sofa | D. chair", "D", "door, tv, sofa, chair, table"),
                (3, "object_counting", "How many chair(s) are in this room?", "", "2", "chair, room"),
                (4, "object_abs_distance", "What is the distance between the sofa and the door?", "", "2",
                 "sofa, door")]
        with open(root / "index.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "scene", "question_type", "question", "options", "ground_truth", "prompts",
                        "pipeline_sheet", "gold_sheet", "pipeline_ok", "gold_ok"])
            w.writerows([[i, "s/1", t, q, o, g, p, "s__1/subset-uniform16/sheets/all", "s__1/gold/sheets/all",
                          True, True] for i, t, q, o, g, p in rows])
        ss.load_index.cache_clear()

    def tearDown(self):
        ss.load_index.cache_clear()
        self.tmp.cleanup()

    def test_closest_point_distances(self):
        self.assertEqual(ss.relations(1, self.root), "\n".join([
            "Relations between the objects the question names, from the sheet above (distance between the closest "
            "points of their boxes, in metres):", "object_a,object_b,distance", "table_1,door_1,1.000",
            "Closest-point distance between the table and the door: 1.000 m."]))
        text = ss.relations(2, self.root)
        self.assertIn("table_1,tv_1,2.062\n", text)  # 0.5 m along y and 2 m along z: sqrt(4.25)
        self.assertIn("table_1,chair_1,0.000\n", text)  # overlapping boxes
        self.assertIn("table_1,chair_2,4.000\n", text)
        self.assertNotIn("chair_1,chair_2", text)  # only objects of different classes
        self.assertTrue(text.endswith("Distance from each candidate to the nearest table (closest points, metres): "
                                      "door 1.000; tv 2.062; sofa not in the sheet; chair 0.000. Closest to the "
                                      "table: chair."))

    def test_sheet_source(self):
        gold = ss.render_sheet(1, "gold", "csv", self.root, vggt_root=self.vggt)
        self.assertEqual(ss.render_sheet(1, "gold_relations", "csv", self.root, vggt_root=self.vggt),
                         gold + "\n" + ss.relations(1, self.root))
        # one named class (counting: `room` is context) or a named object missing: the gold sheet alone
        for q in (3, 4):
            self.assertEqual(ss.relations(q, self.root), "")
            self.assertEqual(ss.render_sheet(q, "gold_relations", "csv", self.root, vggt_root=self.vggt),
                             ss.render_sheet(q, "gold", "csv", self.root, vggt_root=self.vggt))
        for fmt in ss.FORMATS:  # the relations follow the gold sheet in any table format
            self.assertTrue(ss.render_sheet(2, "gold_relations", fmt, self.root, vggt_root=self.vggt).endswith(
                ss.relations(2, self.root)))
        with self.assertRaises(ValueError):
            ss.render_sheet(1, "gold_relations", "narrative", self.root, vggt_root=self.vggt)
        with self.assertRaises(ValueError):
            ss.check_options("gold_relations", "csv", cameras=True)
        self.assertEqual(ss.load_objects(2, "gold_relations", self.root), ss.load_objects(2, "gold", self.root))

    def test_prompt_and_eval(self):
        prompt = ss.build_sheet_prompt(1, "gold_relations", "csv", self.root, self.vggt, reasoning="cot")
        self.assertIn(ss.relations(1, self.root), prompt)
        self.assertIn("in the scene information above", prompt)
        self.assertNotIn("computed automatically", prompt)  # the pipeline's warning is not for gold sheets
        from mvp.eval import vsibench as ev
        import inspect
        self.assertIn("SHEET_SOURCES", inspect.getsource(ev.main))  # --sheet-source takes gold_relations
        self.assertIn("gold_relations", ss.SHEET_SOURCES)


class Prompt(unittest.TestCase):
    def test_prompt_unchanged_without_sheet(self):
        from mvp.benchmarks import vsibench
        na = {"question_type": "object_counting", "question": "How many chairs?"}
        mca = {"question_type": "object_rel_distance", "question": "Which is closest?", "options": ["A. a", "B. b"]}
        self.assertEqual(vsibench.build_prompt(na), "These are frames of a video.\nHow many chairs?\n"
                         "Please answer the question using a single word or phrase.")
        self.assertEqual(vsibench.build_prompt_parts(mca), ("", vsibench.build_prompt(mca)))
        self.assertEqual(vsibench.build_prompt(mca, "SHEET"), vsibench.build_prompt(mca).replace(
            "These are frames of a video.\n", "These are frames of a video.\nSHEET\n"))
        with self.assertRaises(ValueError):
            vsibench.build_prompt_parts(mca, "stit", "SHEET")

    def test_chain_of_thought(self):
        from mvp.benchmarks import vsibench
        mca = {"question_type": "object_rel_direction_hard", "question": "Left or right?", "options": ["A. left",
                                                                                                    "B. right"]}
        na = {"question_type": "object_size_estimation", "question": "How long is the sofa?"}
        prompt = vsibench.build_prompt(mca, reasoning="cot")
        self.assertEqual(prompt.splitlines()[:5], ["These are frames of a video.", "Left or right?", "Options:",
                                                   "A. left", "B. right"])
        self.assertIn(vsibench.COT_HINTS["object_rel_direction"], prompt)
        self.assertNotIn("scene information", prompt)  # no sheet, nothing said about one
        self.assertTrue(prompt.endswith(vsibench.COT_ANSWER["mca"]))
        self.assertNotIn("directly", prompt)
        # a pipeline sheet: what to watch for in it, and its scale for questions in metres
        sheet = vsibench.build_prompt(na, "SHEET", "cot", "pipeline")
        self.assertIn("These are frames of a video.\nSHEET\nHow long is the sofa?\n", sheet)
        self.assertIn("in the video and in the scene information above", sheet)
        self.assertIn(vsibench.COT_PIPELINE, sheet)
        self.assertIn(vsibench.COT_SCALE, sheet)
        self.assertTrue(sheet.endswith(vsibench.COT_ANSWER["na"]))
        gold = vsibench.build_prompt(na, "SHEET", "cot", "gold")
        self.assertNotIn(vsibench.COT_PIPELINE, gold)
        self.assertNotIn(vsibench.COT_SCALE, gold)
        self.assertNotIn(vsibench.COT_SCALE, vsibench.build_prompt(mca, "SHEET", "cot", "pipeline"))
        # every question type has a hint, and the answer line is the one score_final reads
        self.assertEqual(set(vsibench.COT_HINTS), set(vsibench.TASKS))
        self.assertEqual(vsibench.score_final({**mca, "ground_truth": "B"}, "It is on the right.\nAnswer: B"), 1.0)
        self.assertEqual(vsibench.build_prompt_parts(mca, "stit", reasoning="cot")[1], prompt)
        with self.assertRaises(ValueError):
            vsibench.build_prompt(mca, reasoning="think")


class NoVideo(unittest.TestCase):
    """mvp.eval.vsibench --no-video: the scene sheet and the question as text, without the video."""
    NA = {"id": 1, "question_type": "object_counting", "question": "How many chairs?"}
    MCA = {"id": 2, "question_type": "object_rel_distance", "question": "Which is closest?", "options": ["A. a", "B. b"]}

    def test_prompt(self):
        from mvp.benchmarks import vsibench
        self.assertEqual(vsibench.build_prompt(self.NA, "SHEET", video=False),
                         "SHEET\nHow many chairs?\nPlease answer the question using a single word or phrase.")
        for doc in (self.NA, self.MCA):  # the prompt with the video, less its first line
            self.assertEqual(vsibench.build_prompt(doc, "SHEET", video=False),
                             vsibench.build_prompt(doc, "SHEET").split("\n", 1)[1])
            self.assertEqual(vsibench.build_prompt_parts(doc, "sit", "SHEET", video=False),
                             ("", vsibench.build_prompt(doc, "SHEET", video=False)))
        with self.assertRaises(ValueError):
            vsibench.build_prompt(self.MCA, "SHEET", "cot", "gold", video=False)
        with self.assertRaises(ValueError):
            vsibench.build_prompt_parts(self.MCA, "stit", video=False)

    def test_eval_options(self):
        import contextlib
        import io
        import sys
        from unittest import mock
        from mvp.eval import vsibench as ev
        common = ["vsibench", "--model", "m", "--nframes", "16", "--sampling", "uniform128", "--out", "x.jsonl"]
        for extra in (["--no-video"], ["--no-video", "--sheet-source", "gold", "--reasoning", "cot"],
                      ["--no-video", "--sheet-source", "gold", "--images"]):
            with mock.patch.object(sys, "argv", common + extra), contextlib.redirect_stderr(io.StringIO()) as err:
                with self.assertRaises(SystemExit):
                    ev.main()
            self.assertIn("--no-video", err.getvalue())
        args = mock.Mock(no_video=True, sheets=True, sheet_source="gold", sheet_format="csv", layout="sit",
                         reasoning="direct", metric_scale=None, sheets_root="sheets", vggt_root="vggt")
        with mock.patch.object(ev, "render_scene", return_value="SHEET"):
            self.assertEqual(ev.prompt_parts(self.MCA, args),
                             ("", "SHEET\nWhich is closest?\nOptions:\nA. a\nB. b\n"
                                  "Answer with the option's letter from the given choices directly."))

    def test_text_generation_sends_no_visuals(self):
        try:
            from mvp.models import qwen_vl
        except Exception as e:  # no torch, transformers or decord here
            self.skipTest(f"mvp.models.qwen_vl not importable: {e}")
        from unittest import mock
        model = qwen_vl.QwenVL.__new__(qwen_vl.QwenVL)
        model.template_kwargs = {"enable_thinking": False}
        model.processor = mock.Mock()
        model.processor.apply_chat_template.side_effect = lambda messages, **kw: messages
        model.text_inputs(["POST"], pres=["PRE"])
        messages = model.processor.apply_chat_template.call_args.args[0]
        self.assertEqual(messages[1]["content"], [{"type": "text", "text": "PRE"}, {"type": "text", "text": "POST"}])
        call = model.processor.call_args
        self.assertNotIn("videos", call.kwargs)
        self.assertNotIn("images", call.kwargs)

    def test_run_without_video(self):
        """The eval's main() end to end with --no-video, a stand-in model and 25 videos' questions (so the progress
        line prints at videos 1 and 21): every question answered with no frames, the summary written, and a second
        run resumes without asking again. No weights, video or sheet files are read."""
        import contextlib
        import io
        import json
        import sys
        import types
        from unittest import mock
        import pandas as pd
        from mvp.eval import vsibench as ev

        class StandIn:  # the part of QwenVL the eval uses without the video
            calls = []

            def __init__(self, model_id):
                pass

            def generate_text(self, prompts, max_new_tokens=16, pres=None, system=None):
                StandIn.calls.append(prompts)
                self.last_new_tokens = [(1, False)] * len(prompts)
                self.last_prompt_tokens = [len(p.split()) for p in prompts]
                return ["A"] * len(prompts), max(self.last_prompt_tokens)

            def generate_video(self, *args, **kwargs):
                raise AssertionError("--no-video asked for the video")

            def load_video_frames(self, *args, **kwargs):
                raise AssertionError("--no-video loaded a video")

            load_video = load_video_frames

        docs = pd.DataFrame([{"id": i, "dataset": "scannet", "scene_name": f"s{i // 2}", "question_type": "object_rel_distance",
                              "question": f"Which is closest? ({i})", "options": ["A. a", "B. b"], "ground_truth": "A",
                              "video_path": f"/no/such/scannet/s{i // 2}.mp4"} for i in range(50)])
        fake = types.ModuleType("mvp.models.qwen_vl")
        fake.QwenVL, fake.SYSTEM_PROMPT = StandIn, "You are a helpful assistant."
        with tempfile.TemporaryDirectory() as tmp:
            out = f"{tmp}/E.jsonl"
            argv = ["vsibench", "--model", "m", "--nframes", "16", "--sampling", "uniform128", "--max-new-tokens", "16",
                    "--final-answer", "--sheet-source", "gold_relations", "--sheet-format", "csv", "--no-video",
                    "--sheets-root", tmp, "--out", out]
            with mock.patch.dict(sys.modules, {"mvp.models.qwen_vl": fake}), mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(ev, "load_vsibench", return_value=docs), \
                    mock.patch.object(ev, "load_index", return_value=set(range(50))), \
                    mock.patch.object(ev, "render_scene", return_value="SHEET"), \
                    contextlib.redirect_stdout(io.StringIO()) as printed:
                ev.main()
                first = len(StandIn.calls)
                ev.main()  # resumes: nothing left to ask
            rows = [json.loads(line) for line in Path(out).read_text().splitlines()]
            summary = json.loads(Path(out.replace(".jsonl", "_summary.json")).read_text())
        self.assertEqual(sorted(r["id"] for r in rows), list(range(50)))
        self.assertTrue(all(r["n_frames"] == 0 and r["frame_hw"] is None and r["frame_indices"] == [] for r in rows))
        self.assertTrue(all(r["n_prompt_tokens"] > 0 and r["score"] == 1.0 for r in rows))
        self.assertEqual((summary["protocol"], summary["n"]), ("text", 50))
        self.assertEqual(len(StandIn.calls), first)
        self.assertTrue(all(p.startswith("SHEET\nWhich is closest?") for call in StandIn.calls for p in call))
        self.assertIn("[1 videos, 2 qs", printed.getvalue())
        self.assertIn("[21 videos, 42 qs", printed.getvalue())
        self.assertIn("no video tokens=", printed.getvalue())

    def test_gold_sheets_with_the_video_on_a_gold_only_index(self):
        """B and C of the gold-sheet experiment: gold sheets with the video, from an index with no pipeline sheets
        (build_gold_sheets.py's). The frame check is for sheets that refer to frames, so it is not run; the eval's
        main() answers every question with the video (a stand-in model; no weights or videos are read)."""
        import contextlib
        import io
        import json
        import sys
        import types
        from unittest import mock
        import numpy as np
        import pandas as pd
        from mvp.eval import vsibench as ev

        class StandIn:
            def __init__(self, model_id):
                pass

            def load_video_frames(self, path, indices):
                return np.zeros((16, 3, 64, 64)), {"frames_indices": list(indices)}, {}

            def generate_video(self, video, meta, vkw, prompts, pres=None, system=None, max_new_tokens=16):
                self.last_new_tokens = [(1, False)] * len(prompts)
                self.last_prompt_tokens = [10] * len(prompts)
                return ["A"] * len(prompts), 10

        docs = pd.DataFrame([{"id": i, "dataset": "scannet", "scene_name": f"s{i // 2}", "question_type": "object_rel_distance",
                              "question": f"Which is closest? ({i})", "options": ["A. a", "B. b"], "ground_truth": "A",
                              "video_path": f"/no/such/scannet/s{i // 2}.mp4"} for i in range(4)])
        fake = types.ModuleType("mvp.models.qwen_vl")
        fake.QwenVL, fake.SYSTEM_PROMPT = StandIn, "You are a helpful assistant."
        for source in ("gold", "gold_relations"):
            with tempfile.TemporaryDirectory() as tmp:
                out = f"{tmp}/{source}.jsonl"
                argv = ["vsibench", "--model", "m", "--nframes", "16", "--sampling", "uniform128", "--max-new-tokens",
                        "16", "--final-answer", "--sheet-source", source, "--sheet-format", "csv", "--sheets-root", tmp,
                        "--out", out]
                with mock.patch.dict(sys.modules, {"mvp.models.qwen_vl": fake}), mock.patch.object(sys, "argv", argv), \
                        mock.patch.object(ev, "load_vsibench", return_value=docs), \
                        mock.patch.object(ev, "load_index", return_value=set(range(4))), \
                        mock.patch.object(ev, "render_scene", return_value="SHEET"), \
                        mock.patch.object(ev, "frame_indices", return_value=list(range(0, 128, 8))), \
                        mock.patch.object(ev, "check_sheet_frames", side_effect=AssertionError("checked gold sheets")), \
                        contextlib.redirect_stdout(io.StringIO()):
                    ev.main()
                rows = [json.loads(line) for line in Path(out).read_text().splitlines()]
            self.assertEqual(sorted(r["id"] for r in rows), [0, 1, 2, 3], source)
            self.assertTrue(all(r["n_frames"] == 16 for r in rows), source)
        args = mock.Mock(sheets=True, sheet_source="pipeline", cameras=False)
        self.assertTrue(ev.frames_in_sheet(args))  # pipeline sheets are still checked
        args.sheet_source, args.cameras = "gold", True
        self.assertTrue(ev.frames_in_sheet(args))

    def test_each_prompts_own_length_is_recorded(self):
        try:
            import torch
            from mvp.models import qwen_vl
        except Exception as e:  # no torch, transformers or decord here
            self.skipTest(f"mvp.models.qwen_vl not importable: {e}")
        from unittest import mock
        model = qwen_vl.QwenVL.__new__(qwen_vl.QwenVL)
        model.processor = mock.Mock()
        model.processor.tokenizer.convert_tokens_to_ids.return_value = 0
        model.processor.batch_decode.return_value = ["A", "B"]
        model.model = mock.Mock(device="cpu")
        inputs = mock.MagicMock()
        inputs.to.return_value = inputs
        ids, mask = torch.zeros(2, 5, dtype=torch.long), torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
        inputs.__getitem__.side_effect = {"input_ids": ids, "attention_mask": mask}.__getitem__
        model.model.generate.return_value = torch.cat([ids, torch.tensor([[7, 0], [8, 0]])], dim=1)
        with mock.patch.object(qwen_vl, "stop_tokens", return_value={"eos_token_id": [0]}):
            replies, padded = model._generate(inputs, 2)
        self.assertEqual((replies, padded, model.last_prompt_tokens), (["A", "B"], 5, [3, 5]))
        from mvp.eval import vsibench as ev
        import inspect
        self.assertIn('rec["n_prompt_tokens"]', inspect.getsource(ev.main))


class PromptTemplate(unittest.TestCase):
    def test_system_prompt_is_the_models(self):
        tree = ast.parse(QWEN_VL.read_text())
        value = next(node.value.value for node in tree.body if isinstance(node, ast.Assign)
                     and getattr(node.targets[0], "id", None) == "SYSTEM_PROMPT")
        self.assertEqual(ss.SYSTEM_PROMPT, value)

    def test_chat_template_is_the_tokenizers(self):
        """ss.CHAT with the video as the template writes it is apply_chat_template's output, as QwenVL builds it."""
        try:
            import jinja2  # noqa: F401
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", local_files_only=True)
        except Exception as e:  # no transformers, no jinja2, or no tokenizer in the cache
            self.skipTest(f"Qwen3.5-2B tokenizer not available: {e}")
        messages = [{"role": "system", "content": [{"type": "text", "text": ss.SYSTEM_PROMPT}]},
                    {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "POST"}]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        self.assertEqual(text, ss.CHAT.format(system=ss.SYSTEM_PROMPT, pre="", video=ss.VIDEO_PAD, post="POST"))
        # without the video (--no-video), as QwenVL.render_text_prompt builds it
        messages[1]["content"] = [{"type": "text", "text": "POST"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        self.assertEqual(text, ss.TEXT_CHAT.format(system=ss.SYSTEM_PROMPT, text="POST"))


@unittest.skipUnless((SCENE_SHEETS / "index.csv").exists() and VGGT_ROOT.exists(),
                     "data/scene_sheets_75 or the VGGT meta.json files are not here")
class RealSheets(unittest.TestCase):
    def test_round_trip_every_question(self):
        for q in ss.load_index(SCENE_SHEETS):
            for source in ss.SOURCES:
                canonical = ss.load_objects(q, source, vggt_root=VGGT_ROOT)
                for fmt in ss.FORMATS:
                    rendered = ss.render_sheet(q, source, fmt, vggt_root=VGGT_ROOT)
                    self.assertEqual(parse(rendered, source, fmt), canonical, (q, source, fmt))

    def test_prompt_questions_are_vsibench(self):
        """build_sheet_prompt takes the question from index.csv; the eval takes it from VSI-Bench's test.jsonl."""
        if not VSI_JSONL.exists():
            self.skipTest(f"no {VSI_JSONL}")
        from mvp.benchmarks.vsibench import build_prompt
        docs = {d["id"]: d for d in map(json.loads, open(VSI_JSONL))}
        for q in ss.load_index(SCENE_SHEETS):
            prompt = ss.build_sheet_prompt(q, "gold", "csv", vggt_root=VGGT_ROOT)
            expected = build_prompt(docs[q], ss.render_sheet(q, "gold", "csv", vggt_root=VGGT_ROOT))
            self.assertIn("<|vision_end|><|vision_end|>" + expected + "<|im_end|>", prompt, q)

    def test_same_triples_in_every_format(self):
        ids = random.Random(0).sample(sorted(ss.load_index(SCENE_SHEETS)), 50)
        for q in ids:
            for source in ss.SOURCES:
                found = [triples(parse(ss.render_sheet(q, source, fmt, vggt_root=VGGT_ROOT), source, fmt))
                         for fmt in ss.FORMATS]
                self.assertTrue(all(t == found[0] for t in found), (q, source))


if __name__ == "__main__":
    unittest.main()
