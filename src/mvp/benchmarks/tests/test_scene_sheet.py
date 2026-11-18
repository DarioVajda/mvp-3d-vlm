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
import json
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
    rf"(?: It is first seen at (?P<first_seen>\d+\.\d) seconds and is visible in (?P<num_frames>\d+) of the 16 "
    rf"frames\.)?$")


def parse(rendered, source, fmt, volume=False):
    """A rendered sheet back to the canonical list of {field: string}; the preamble must be the source's."""
    head, body = rendered.split("\n", 1)
    assert head == ss.preamble(source, volume), head
    if body == ss.EMPTY:
        return []
    if fmt == "csv":
        return list(csv.DictReader(body.splitlines()))
    if fmt == "json":
        # numbers are read back as their text, so "-0.500" stays "-0.500"
        objects = json.loads(body, parse_float=str, parse_int=str)
        assert body.count('": "') == 2 * len(objects), "only id and label are strings; numbers are not quoted"
        return objects
    if fmt == "markdown_kv":
        objects = []
        for block in body.split("\n\n"):
            lines = block.splitlines()
            assert lines[0].startswith("## ") and lines[1] == "```" and lines[-1] == "```", block
            objects.append({"id": lines[0][3:], **dict(line.split(": ", 1) for line in lines[2:-1])})
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
            objects.append({k: v for k, v in m.groupdict().items() if v is not None})
        return objects
    raise ValueError(fmt)


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
