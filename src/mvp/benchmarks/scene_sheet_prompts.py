"""Many prompts for one VSI-Bench question: the same question with its scene information written in different ways, to
compare them side by side. Each prompt is scene_sheet.build_sheet_prompt's, the whole text input the eval sends
(mvp.eval.vsibench --sampling uniform128 --nframes 16), with each pair of frames' video tokens shown as one
<|video_pad|> (--frame-tokens N shows "<|video_pad|> x N", as --print-prompt does).

By default the CURATED configurations are printed: about 20, each differing from BASELINE in a way worth seeing, so that
together they show every option (every format, gold objects, the objects with their cameras as tables and as text, the
narrative and its settings, frame numbers, exact visibility, chain of thought). The cameras are always in the objects'
format, and only come with the pipeline's objects. Giving any option list below prints every combination
of the given values instead, the other options staying at BASELINE. Contradictory combinations (scene_sheet.
check_options) are skipped and listed at the end, and a prompt identical to an earlier one is not printed again.

The question is --question, or the first question of --scene in index.csv, or by default DEFAULT_QUESTION (the format
experiment's example question) when the index has it, else the index's first question.

  python -m mvp.benchmarks.scene_sheet_prompts                               # the curated set for question 706
  python -m mvp.benchmarks.scene_sheet_prompts --question 1203
  python -m mvp.benchmarks.scene_sheet_prompts --scene scannet/scene0025_01   # that scene's first question
  python -m mvp.benchmarks.scene_sheet_prompts --list [--scene ...]          # the scenes, or one scene's questions
  python -m mvp.benchmarks.scene_sheet_prompts --format csv text narrative --reasoning direct cot --cameras off on
  python -m mvp.benchmarks.scene_sheet_prompts --out-dir prompts_706          # one file per prompt, not printed
"""
import argparse
import itertools
import re
import sys
from collections import Counter
from pathlib import Path

from mvp.benchmarks import scene_sheet as ss
from mvp.benchmarks.vsibench import REASONING
from mvp.paths import SCENE_SHEETS, VGGT_DIR

DEFAULT_QUESTION = 706  # experiments/2026-11-18-scene-sheet-formats/examples
BASELINE = {"source": "pipeline", "fmt": "csv", "cameras": False, "time_unit": "seconds", "reasoning": "direct",
            "visibility": "auto", "step_order": "move-turn", "object_directions": "floor"}
VIDEO_ONLY = {"source": None, "fmt": None}
WALK = {"fmt": ss.NARRATIVE}
CURATED = [  # (label, how it differs from BASELINE)
    ("video only", VIDEO_ONLY),
    ("video only, chain of thought", {**VIDEO_ONLY, "reasoning": "cot"}),
    ("objects as csv (the baseline)", {}),
    *[(f"objects as {fmt}", {"fmt": fmt}) for fmt in ss.FORMATS if fmt != "csv"],
    ("gold objects as csv", {"source": "gold"}),
    ("objects as csv, frame numbers", {"time_unit": "frames"}),
    ("objects as csv, exact visibility", {"visibility": "exact"}),
    ("objects and cameras as csv", {"cameras": True}),
    ("objects and cameras as json", {"fmt": "json", "cameras": True}),
    ("objects and cameras as text", {"fmt": "text", "cameras": True}),
    ("objects and cameras as text, turn then move", {"fmt": "text", "cameras": True, "step_order": "turn-move"}),
    ("narrative", WALK),
    ("narrative, frame numbers", {**WALK, "time_unit": "frames"}),
    ("narrative, turn then move", {**WALK, "step_order": "turn-move"}),
    ("narrative, object directions in the camera's axes", {**WALK, "object_directions": "camera"}),
    ("objects as csv, chain of thought", {"reasoning": "cot"}),
    ("narrative, chain of thought", {**WALK, "reasoning": "cot"}),
    # auto is bridged on a 16-frame sheet and nearby on a dense one (a sheet made on all 128 frames)
    ("objects as csv, bridged visibility", {"visibility": "bridged"}),
]
# the command line's option lists: (flag, config key, values); WORDS gives the values that aren't strings
OPTION_LISTS = [
    ("--source", "source", ss.SHEET_SOURCES + ["none"]),
    ("--format", "fmt", ss.SHEET_FORMATS),
    ("--cameras", "cameras", ["off", "on"]),
    ("--time-unit", "time_unit", ss.TIME_UNITS),
    ("--reasoning", "reasoning", REASONING),
    ("--visibility", "visibility", ss.VISIBILITY),
    ("--step-order", "step_order", ss.STEP_ORDERS),
    ("--object-directions", "object_directions", ss.OBJECT_DIRECTIONS),
]
WORDS = {"none": None, "off": False, "on": True}


def curated_configs():
    """[(label, configuration)] of CURATED, each configuration complete (BASELINE with its changes)."""
    return [(label, {**BASELINE, **changes}) for label, changes in CURATED]


def product_configs(lists):
    """[(label, configuration)] for every combination of lists ({config key: [values]}), the other keys at BASELINE.
    Without a source, the object format means nothing and is dropped, so those combinations come out the same."""
    keys = list(lists)
    configs = []
    for values in itertools.product(*(lists[k] for k in keys)):
        config = {**BASELINE, **dict(zip(keys, values))}
        if config["source"] is None:
            config["fmt"] = None
        configs.append((describe(config), config))
    return configs


def describe(config):
    """The options that make a difference to this configuration, in one line."""
    source, cameras = config["source"], config["cameras"]
    narrative = source is not None and config["fmt"] == ss.NARRATIVE
    head = f"objects: {source} {config['fmt']}" if source else "video only"
    parts = [head + (" + cameras" if cameras else "")]
    if source:
        parts.append(f"time: {config['time_unit']}")
    if source == "pipeline":
        parts.append(f"visibility: {config['visibility']}")
    if narrative or (cameras and config["fmt"] == "text"):
        parts.append(f"steps: {config['step_order']}")
    if narrative:
        parts.append(f"object directions: {config['object_directions']}")
    parts.append(f"reasoning: {config['reasoning']}")
    return " | ".join(parts)


def contradiction(config):
    """Why scene_sheet.check_options refuses this configuration, or None."""
    try:
        ss.check_options(config["source"], config["fmt"], config["cameras"], config["time_unit"],
                         config["visibility"], config["step_order"], config["object_directions"])
    except ValueError as e:
        return str(e)
    return None


def prompt_of(question_id, config, root=SCENE_SHEETS, vggt_root=VGGT_DIR, frame_tokens=None):
    return ss.build_sheet_prompt(question_id, config["source"], config["fmt"], root, vggt_root, frame_tokens,
                                 cameras=config["cameras"], time_unit=config["time_unit"],
                                 reasoning=config["reasoning"], visibility=config["visibility"],
                                 step_order=config["step_order"], object_directions=config["object_directions"])


def render(question_id, configs, root=SCENE_SHEETS, vggt_root=VGGT_DIR, frame_tokens=None):
    """(shown, skipped). shown: [(number, label, configuration, prompt, number of the earlier identical prompt or
    None)] for the valid configurations; skipped: [(label, configuration, why)] for the contradictory ones."""
    shown, skipped, first = [], [], {}
    for label, config in configs:
        why = contradiction(config)
        if why:
            skipped.append((label, config, why))
            continue
        prompt = prompt_of(question_id, config, root, vggt_root, frame_tokens)
        number = len(shown) + 1
        shown.append((number, label, config, prompt, first.get(prompt)))
        first.setdefault(prompt, number)
    return shown, skipped


def pick_question(root=SCENE_SHEETS, question=None, scene=None):
    """The question to show: question, else scene's first in index.csv, else DEFAULT_QUESTION if the index has it,
    else the index's first. SystemExit if the question or scene isn't in the index, or the two don't match."""
    index = ss.load_index(root)
    if question is not None:
        if question not in index:
            raise SystemExit(f"question {question} is not in {Path(root) / 'index.csv'}")
        if scene and index[question]["scene"] != scene:
            raise SystemExit(f"question {question} is about {index[question]['scene']}, not {scene}")
        return question
    if scene:
        ids = sorted(q for q, row in index.items() if row["scene"] == scene)
        if not ids:
            raise SystemExit(f"no question about {scene} in {Path(root) / 'index.csv'} (--list shows the scenes)")
        return ids[0]
    return DEFAULT_QUESTION if DEFAULT_QUESTION in index else min(index)


def listing(root=SCENE_SHEETS, scene=None):
    """--list: the scenes with their number of questions, or one scene's questions."""
    index = ss.load_index(root)
    if scene is None:
        counts = Counter(row["scene"] for row in index.values())
        return "\n".join(f"{name}  ({n} questions)" for name, n in sorted(counts.items()))
    rows = [(q, row) for q, row in sorted(index.items()) if row["scene"] == scene]
    if not rows:
        raise SystemExit(f"no question about {scene} in {Path(root) / 'index.csv'}")
    return "\n".join(f"{q}  {row['question_type']}: {row['question']}" for q, row in rows)


def question_header(question_id, root=SCENE_SHEETS):
    row = ss.load_index(root)[question_id]
    options = f"\noptions: {row['options']}" if row["options"] else ""
    return (f"question {question_id}, {row['scene']} ({row['question_type']})\n{row['question']}{options}\n"
            f"ground truth: {row['ground_truth']}")


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:80]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(SCENE_SHEETS), help="the sheets folder, with index.csv")
    ap.add_argument("--vggt-root", default=str(VGGT_DIR), help="the VGGT outputs, for each video's meta.json (fps)")
    ap.add_argument("--question", type=int, help=f"the question id (default {DEFAULT_QUESTION})")
    ap.add_argument("--scene", help="a scene as index.csv names it, e.g. scannet/scene0025_01: its first question")
    ap.add_argument("--list", action="store_true", help="list the scenes (or --scene's questions) and stop")
    ap.add_argument("--frame-tokens", type=int, help="write each frame pair's video tokens as '<|video_pad|> x N'")
    ap.add_argument("--out-dir", help="write one file per prompt here instead of printing them")
    for flag, key, values in OPTION_LISTS:
        ap.add_argument(flag, dest=key, nargs="+", choices=values,
                        help="values to combine (default: the curated set; baseline "
                             f"{next((w for w, v in WORDS.items() if v is BASELINE[key]), BASELINE[key])})")
    args = ap.parse_args(argv)
    root, vggt_root = Path(args.root), Path(args.vggt_root)
    if args.list:
        print(listing(root, args.scene))
        return 0

    question_id = pick_question(root, args.question, args.scene)
    lists = {key: [WORDS.get(v, v) for v in getattr(args, key)]
             for _, key, _ in OPTION_LISTS if getattr(args, key)}
    configs = product_configs(lists) if lists else curated_configs()
    shown, skipped = render(question_id, configs, root, vggt_root, args.frame_tokens)

    out = [question_header(question_id, root), f"{len(shown)} prompts"
           + (f", {len(skipped)} contradictory combinations skipped" if skipped else "")]
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    for number, label, config, prompt, same_as in shown:
        head = f"===== {number}/{len(shown)}  {label}" + ("" if label == describe(config) else f"\n{describe(config)}")
        if same_as:
            out.append(f"{head}\n(the same prompt as {same_as})")
        elif args.out_dir:
            path = out_dir / f"{number:02d}_{slug(label)}.txt"
            path.write_text(f"{head}\n\n{prompt}\n", encoding="utf-8")
            out.append(f"{head}\nwritten to {path}")
        else:
            out.append(f"{head}\n\n{prompt}")
    if skipped:
        out.append("===== skipped, contradictory:\n" + "\n".join(f"{label}: {why}" for label, _, why in skipped))
    print("\n\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
