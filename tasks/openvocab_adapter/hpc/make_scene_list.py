"""Pick the scenes for the multi-scene comparison and the SAM 3 prompts for each, and write hpc/scenes.tsv.

Prompts are the objects named in the scene's VSI-Bench questions, read from the questions' fixed templates (no NLP
model needed). Route-planning questions are skipped: their free text names few objects that the other types don't.

Scenes: scannet/scene0025_01 (the first test scene), then the first landscape scenes of each dataset in name order.
Portrait scenes are left out: OpenVocab-4D's own VGGT loader resizes portrait frames to a different shape than Dario's
export, so the two runs' depth maps could not be compared pixel by pixel.

  py tasks/openvocab_adapter/hpc/make_scene_list.py            (reads tasks/vggt_data/out/vggt_vsibench_75)
  py tasks/openvocab_adapter/hpc/make_scene_list.py --data-root <export> --per-dataset scannet=4 scannetpp=3 arkitscenes=3
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

OUT = Path(__file__).resolve().parent / "scenes.tsv"
FIRST_SCENE = "scannet/scene0025_01"
# question type -> regex whose groups are object names (a group may hold a comma-separated list)
TEMPLATES = {
    "object_counting": r"How many (.+?)\(s\) are in this room",
    "object_size_estimation": r"dimension \(length, width, or height\) of the (.+?), measured",
    "object_abs_distance": r"distance between the (.+?) and the (.+?) \(in meters\)",
    "object_rel_distance": r"which of these objects \((.+?)\) is the closest to the (.+?)\?",
    "object_rel_direction_easy": r"standing by the (.+?) and facing the (.+?), is the (.+?) to the left",
    "object_rel_direction_medium": r"standing by the (.+?) and facing the (.+?), is the (.+?) to my left",
    "object_rel_direction_hard": r"standing by the (.+?) and facing the (.+?), is the (.+?) to my front",
    "obj_appearance_order": r"following categories in the video: (.+?)\?",
}
SKIPPED = {"room_size_estimation", "route_planning"}


def objects_in(question):
    """Object names in one question, or None if its type has no template."""
    if question["question_type"] in SKIPPED:
        return []
    pattern = TEMPLATES.get(question["question_type"])
    match = re.search(pattern, question["question"]) if pattern else None
    if match is None:
        return None
    names = []
    for group in match.groups():
        names += [n.strip().lower() for n in group.split(",") if n.strip()]
    return names


def scene_prompts(scene_dir, limit):
    counts, unparsed = Counter(), 0
    for line in (scene_dir / "questions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            names = objects_in(json.loads(line))
            if names is None:
                unparsed += 1
            else:
                counts.update(names)
    if unparsed:
        print(f"[scenes] {scene_dir.parent.name}/{scene_dir.name}: {unparsed} question(s) matched no template")
    # most often asked about first; ties in name order, so the list is reproducible
    return [name for name, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))][:limit]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=str(common.DATA_ROOT), help="export holding scenes.csv and scenes/")
    p.add_argument("--per-dataset", nargs="+", default=["scannet=4", "scannetpp=3", "arkitscenes=3"],
                   help="scenes per dataset, including the first scene (default: 10 scenes)")
    p.add_argument("--max-prompts", type=int, default=12, help="SAM 3 prompts per scene (SAM 3's time grows with it)")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args(argv)

    root = Path(args.data_root)
    with open(root / "scenes.csv", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["portrait"] == "False"]
    quota = {k: int(v) for k, v in (item.split("=") for item in args.per_dataset)}
    chosen = [FIRST_SCENE]
    for dataset, count in quota.items():
        names = sorted(f"{r['dataset']}/{r['scene']}" for r in rows if r["dataset"] == dataset)
        already = sum(c.startswith(dataset + "/") for c in chosen)  # the first scene counts towards its dataset
        chosen += [n for n in names if n not in chosen][:max(count - already, 0)]

    lines = []
    for name in chosen:
        prompts = scene_prompts(root / "scenes" / name, args.max_prompts)
        lines.append(f"{name}\t{','.join(prompts)}")
        print(f"[scenes] {name}: {', '.join(prompts)}")
    Path(args.out).write_text("# scene<TAB>SAM 3 prompts. Made by make_scene_list.py; edit freely.\n"
                              + "\n".join(lines) + "\n", encoding="utf-8", newline="\n")  # read by bash
    print(f"[scenes] {len(lines)} scenes -> {args.out}")


if __name__ == "__main__":
    main()
