"""Run the SAM 3 prompt extractor on real VSI-Bench questions and print every decision.

Loads the questions of a few scenes from each dataset of the VGGT export (tasks/vggt_data/scenes/
<dataset>/<scene>/questions.jsonl) and prints, for each question, the question, every candidate
noun phrase with the reason it was kept or dropped, and the final SAM 3 prompts. A summary at the
end counts the kept prompts and the dropped phrases over all questions, which is the quickest way
to spot a word that is wrongly kept or dropped.

Scenes are taken in name order; --seed picks them at random instead.

Usage (from the repo root, in an environment with spacy and nltk, see prompt_object_extractor.py):
  python openvocab-4D-main/scene_sheet_creation/tests.py
  python openvocab-4D-main/scene_sheet_creation/tests.py --per-dataset 2 --types route_planning
  python openvocab-4D-main/scene_sheet_creation/tests.py --seed 0 > prompts_report.txt
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from prompt_object_extractor import SPACY_MODEL, analyse, kept_prompts

REPO = Path(__file__).resolve().parents[2]
SCENES_DIR = REPO / "tasks" / "vggt_data" / "scenes"


def pick_scenes(scenes_dir, per_dataset, seed=None):
    """[(dataset, scene folder)], `per_dataset` scenes from each dataset folder."""
    rng = random.Random(seed)
    picked = []
    for dataset in sorted(p for p in scenes_dir.iterdir() if p.is_dir()):
        scenes = sorted(
            p for p in dataset.iterdir() if (p / "questions.jsonl").exists()
        )
        if seed is not None:
            scenes = sorted(rng.sample(scenes, min(per_dataset, len(scenes))))
        picked += [(dataset.name, s) for s in scenes[:per_dataset]]
    return picked


def load_questions(scene_dir, types=None):
    with open(scene_dir / "questions.jsonl", encoding="utf-8") as fh:
        questions = [json.loads(line) for line in fh if line.strip()]
    return [q for q in questions if not types or q["question_type"] in types]


def indent(text, prefix):
    return text.replace("\n", "\n" + " " * len(prefix))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--scenes-dir",
        type=Path,
        default=SCENES_DIR,
        help=f"folder with <dataset>/<scene>/questions.jsonl (default {SCENES_DIR})",
    )
    p.add_argument(
        "--per-dataset", type=int, default=1, help="scenes per dataset (default 1)"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="pick the scenes at random with this seed (default: the first in name order)",
    )
    p.add_argument(
        "--types",
        nargs="+",
        default=None,
        help="only these question types, e.g. object_rel_direction_hard route_planning",
    )
    p.add_argument(
        "--max-per-scene",
        type=int,
        default=None,
        help="at most this many questions per scene",
    )
    p.add_argument(
        "--model", default=SPACY_MODEL, help=f"spaCy pipeline (default {SPACY_MODEL})"
    )
    p.add_argument(
        "--top", type=int, default=40, help="rows in each summary table (default 40)"
    )
    p.add_argument(
        "--syntactic",
        action="store_true",
        help="syntactic mode: every noun phrase is a prompt, nothing based on meaning",
    )
    args = p.parse_args()

    if not args.scenes_dir.is_dir():
        sys.exit(f"[tests] no scenes folder at {args.scenes_dir}: pass --scenes-dir")
    scenes = pick_scenes(args.scenes_dir, args.per_dataset, args.seed)
    if not scenes:
        sys.exit(
            f"[tests] no <dataset>/<scene>/questions.jsonl under {args.scenes_dir}"
        )

    kept, dropped = Counter(), Counter()
    num_questions, num_prompts, empty = 0, 0, []
    for dataset, scene_dir in scenes:
        questions = load_questions(scene_dir, args.types)[: args.max_per_scene]
        print(
            f"\n{'=' * 100}\n{dataset}/{scene_dir.name}: {len(questions)} questions\n{'=' * 100}"
        )
        for q in questions:
            candidates = analyse(q["question"], args.model, args.syntactic)
            prompts = kept_prompts(candidates)
            num_questions += 1
            num_prompts += len(prompts)
            kept.update(prompts)
            dropped.update(c.text for c in candidates if not c.keep)
            if not prompts:
                empty.append(f"{dataset}/{scene_dir.name} #{q['id']}")

            print(f"\n[{q['id']}] {q['question_type']}")
            print(f"Q: {indent(q['question'], 'Q: ')}")
            if q.get("options"):
                print(f"   options: {' | '.join(q['options'])}")
            for c in candidates:
                print(f"   {'keep' if c.keep else 'drop'}  {c.text:28s} {c.reason}")
            print(f"   -> {', '.join(prompts) if prompts else '(no prompts)'}")

    print(
        f"\n{'=' * 100}\nSummary ({'syntactic' if args.syntactic else 'full'} mode): "
        f"{num_questions} questions from {len(scenes)} scenes, "
        f"{num_prompts / max(num_questions, 1):.2f} prompts per question, "
        f"{len(kept)} distinct prompts"
    )
    print(f"\nMost frequent prompts (of {len(kept)}):")
    for text, n in kept.most_common(args.top):
        print(f"  {n:5d}  {text}")
    print(f"\nMost frequent dropped phrases (of {len(dropped)}):")
    for text, n in dropped.most_common(args.top):
        print(f"  {n:5d}  {text}")
    print(f"\nQuestions with no prompts: {len(empty)}")
    for name in empty:
        print(f"  {name}")


if __name__ == "__main__":
    main()
