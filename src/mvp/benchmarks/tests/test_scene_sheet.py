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
import tempfile
import unittest
from pathlib import Path

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
    rf"(?: It is first seen (?:at (?P<first_seen>\d+\.\d) seconds|in frame (?P<first_frame>\d+)) and is visible in "
    rf"(?P<num_frames>\d+) of the 16 frames\.)?$")


def parse(rendered, source, fmt, volume=False, time_unit="seconds"):
    """A rendered sheet back to the canonical list of {field: string}; the preamble must be the source's."""
    head, body = rendered.split("\n", 1)
    assert head == ss.preamble(source, volume, time_unit), head
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
            if "first_frame" in obj:
                obj["first_seen"] = obj.pop("first_frame")
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
                                      "size_x": "0.400", "size_y": "0.900", "size_z": "0.450", "first_seen": "0.5",
                                      "num_frames": "1"})
        self.assertEqual(objects[1]["label"], "coffee table")  # the id has underscores, the label doesn't
        self.assertEqual((objects[2]["x"], objects[3]["y"]), ("-1.236", "0.000"))  # never -0.000
        self.assertEqual(objects[3]["first_seen"], "2.5")  # frame 3 is in the pair (2, 3)
        gold = ss.load_objects(1, "gold", self.root, vggt_root=self.vggt)
        self.assertEqual([o["id"] for o in gold], ["armchair_1", "door_1"])
        self.assertNotIn("first_seen", gold[0])

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
                         "along x, 1.000 along y and 1.000 along z. It is first seen at 0.5 seconds and is visible in "
                         "1 of the 16 frames.")
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

    def test_special_characters(self):
        self.assertEqual(ss.render_csv([{"id": "a,b_1", "label": "a,b"}], ["id", "label"]), 'id,label\n"a,b_1","a,b"')
        with self.assertRaises(ValueError):
            ss.render_markdown_table([{"id": "a_1", "label": "a|b"}], ["id", "label"])

    # ---- frames as frame numbers ----

    def test_frames_time_unit(self):
        objects = ss.load_objects(1, "pipeline", self.root, vggt_root=self.vggt, time_unit="frames")
        self.assertEqual([(o["id"], o["first_seen"]) for o in objects],
                         [("chair_1", "2"), ("coffee_table_1", "2"), ("door_1", "2"), ("door_2", "4")])
        for q in (1, 2):
            for source in ss.SOURCES:
                canonical = ss.load_objects(q, source, self.root, vggt_root=self.vggt, time_unit="frames")
                for fmt in ss.FORMATS:
                    rendered = ss.render_sheet(q, source, fmt, self.root, vggt_root=self.vggt, time_unit="frames")
                    self.assertEqual(parse(rendered, source, fmt, time_unit="frames"), canonical, (q, source, fmt))
                    self.assertNotIn("second", rendered)
        self.assertIn("It is first seen in frame 4 and is visible in 2 of the 16 frames.",
                      ss.render_sheet(1, "pipeline", "text", self.root, vggt_root=self.vggt, time_unit="frames"))
        # the seconds preamble is the one the 2026-11-18 experiment ran with
        self.assertEqual(ss.preamble("pipeline"), ss.PREAMBLES["pipeline"])
        self.assertTrue(ss.PREAMBLES["pipeline"].endswith("first_seen is the video timestamp, in seconds, at which "
                                                          "the object first appears, as the video's timestamps show "
                                                          "it, and num_frames is how many of the 16 video frames it "
                                                          "appears in."))
        with self.assertRaises(ValueError):
            ss.render_sheet(1, "pipeline", "csv", self.root, vggt_root=self.vggt, time_unit="minutes")

    # ---- the camera sheet ----

    def test_camera_round_trip(self):
        for time_unit in ss.TIME_UNITS:
            canonical = ss.load_cameras(1, self.root, self.vggt, time_unit)
            self.assertEqual(len(canonical), 16)
            self.assertEqual(list(canonical[0]), ss.camera_fields(time_unit))
            for fmt in ss.CAMERA_FORMATS:
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
        self.assertEqual(lines[0], ss.NARRATIVE_PREAMBLE + " " + ss.LINE_LABELS["frames"])
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
        self.assertEqual(ss.view_side([1, 0, -1]), "behind you on your right")

    # ---- every combination of options ----

    def test_scene_combines_objects_and_cameras(self):
        both = ss.render_scene(1, "pipeline", "csv", "json", self.root, self.vggt)
        self.assertEqual(both, self.render(1, "pipeline", "csv") + "\n"
                         + ss.render_cameras(1, "json", self.root, self.vggt))
        self.assertEqual(ss.render_scene(1, None, None, "text", self.root, self.vggt),
                         ss.render_cameras(1, "text", self.root, self.vggt))
        self.assertIsNone(ss.render_scene(1, None, None, None, self.root, self.vggt))
        self.assertEqual(ss.sheet_kind(None, "csv"), "pipeline")
        self.assertEqual(ss.sheet_kind("gold"), "gold")
        self.assertIsNone(ss.sheet_kind(None))

    def test_every_combination(self):
        """Each combination of options gives a prompt, unless it is one of the contradictory ones, which raise."""
        from mvp.benchmarks import vsibench
        for q, source, fmt, camera_fmt, time_unit, reasoning in itertools.product(
                (1, 2), [None] + ss.SOURCES, [None] + ss.SHEET_FORMATS, [None] + ss.CAMERA_FORMATS, ss.TIME_UNITS,
                vsibench.REASONING):
            options = (q, source, fmt, camera_fmt, time_unit, reasoning)
            contradictory = source is not None and (
                fmt is None or (source == "gold" and (camera_fmt or fmt == ss.NARRATIVE))
                or (fmt == ss.NARRATIVE and camera_fmt))
            if contradictory:
                with self.assertRaises(ValueError, msg=options):
                    ss.build_sheet_prompt(q, source, fmt, self.root, self.vggt, camera_fmt=camera_fmt,
                                          time_unit=time_unit, reasoning=reasoning)
                continue
            prompt = ss.build_sheet_prompt(q, source, fmt, self.root, self.vggt, camera_fmt=camera_fmt,
                                           time_unit=time_unit, reasoning=reasoning)
            scene = ss.render_scene(q, source, fmt, camera_fmt, self.root, self.vggt, time_unit)
            text = prompt.split("<|vision_end|><|vision_end|>", 1)[1]  # after the video
            self.assertTrue(text.startswith("These are frames of a video.\n" + (scene + "\n" if scene else "")),
                            options)
            self.assertNotIn("None", text, options)
            self.assertEqual("directly" in text or "single word" in text, reasoning == "direct", options)
            self.assertEqual('"Answer: X"' in text, reasoning == "cot", options)
            self.assertEqual("scene information" in text, reasoning == "cot" and scene is not None, options)
            if scene is not None:
                self.assertEqual("second" in scene,
                                 time_unit == "seconds" and (source == "pipeline" or camera_fmt is not None), options)
                self.assertEqual("Frame 1:" in scene, time_unit == "frames"
                                 and ((source and fmt == ss.NARRATIVE) or camera_fmt == "text"), options)


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
