"""An index of the scene sheets made by hpc/run_sheets.sh, and a check of the pipeline's object counts.

Reads the per-scene reports in out/reports/ (scene_sheet_report.py's output) and writes out/index.csv, one row per
question: its VSI-Bench id, scene, type, text, options and answer, the prompts taken from it, and the folders of its
pipeline and gold sheets (relative to out/) with whether each was made.

Then, for each object-counting question ("How many <class>(s) are in this room?"), compares the rows of that class in
the pipeline sheet, and in the gold sheet, with the true answer. The gold sheets should almost always be exact, which
checks the comparison itself. Standard library only, so it runs anywhere:

  python src/mvp/openvocab_adapter/sheet_index.py --data-root <the exported scenes>
  python src/mvp/openvocab_adapter/sheet_index.py --gold-only --questions data/raw/VSI-Bench/test.jsonl --out <root>

--gold-only indexes gold reports alone (<dataset>__<scene>_gold.txt), for scenes with no pipeline sheets and no export:
the questions come from --questions (VSI-Bench's format), the prompts from the gold report, and the pipeline columns
are empty (pipeline_sheet "", pipeline_ok False). Every question of a reported scene gets a row.
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / "tasks/openvocab_adapter/out"
SOURCES = ("pipeline", "gold")
COUNT_QUESTION = re.compile(r"How many (.+?)\(s\)")


def read_report(path):
    """{question id: (prompts, ok, sheet folder or "")} from one scene_sheet_report.py report."""
    found = {}
    for block in re.split(r"\n\[\d+/\d+\] ", path.read_text())[1:]:
        qid = re.match(r"\S+, id (\S+)", block).group(1)
        prompts = re.search(r"prompts:  (.*?)  \(", block).group(1)
        sheets = re.findall(r"(\S+)/objects\.csv", block)
        found[qid] = prompts, "sheet:    OK" in block, sheets[0] if sheets else ""
    return found


def count(sheet, concept):
    with open(sheet / "objects.csv") as f:
        return sum(row["concept"] == concept for row in csv.DictReader(f))


def scene_questions(questions_jsonl):
    """{"<dataset>__<scene>": {question id (str): question}} from a .jsonl in VSI-Bench's format."""
    by_scene = defaultdict(dict)
    with open(questions_jsonl) as f:
        for q in map(json.loads, f):
            by_scene[f"{q['dataset']}__{q['scene_name']}"][str(q["id"])] = q
    return by_scene


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", help="the exported scenes (scenes/<dataset>/<scene>/questions.jsonl)")
    ap.add_argument("--out", default=str(OUT), help="the sheets' root, with reports/ in it")
    ap.add_argument("--gold-only", action="store_true", help="index the gold reports alone (see above)")
    ap.add_argument("--questions", help="--gold-only: the questions, a .jsonl in VSI-Bench's format")
    args = ap.parse_args(argv)
    if args.gold_only != bool(args.questions) or args.gold_only == bool(args.data_root):
        ap.error("give --data-root, or --gold-only with --questions")
    out = Path(args.out).resolve()
    first = "gold" if args.gold_only else "pipeline"
    by_scene = scene_questions(args.questions) if args.gold_only else None

    index, counting = [], defaultdict(list)
    for report in sorted((out / "reports").glob(f"*_{first}.txt")):
        name = report.name[:-len(f"_{first}.txt")]
        dataset, scene = name.split("__", 1)
        if args.gold_only:
            questions = by_scene[name]
            assert questions, f"{name}: no questions in {args.questions}"
        else:
            with open(Path(args.data_root) / "scenes" / dataset / scene / "questions.jsonl") as f:
                questions = {str(q["id"]): q for q in map(json.loads, f)}
        sources = ("gold",) if args.gold_only else SOURCES
        reports = {s: read_report(out / "reports" / f"{name}_{s}.txt") if s in sources else {} for s in SOURCES}
        for qid, q in questions.items():
            row = {"id": qid, "scene": f"{dataset}/{scene}", "question_type": q["question_type"],
                   "question": q["question"], "options": " | ".join(q.get("options") or []),
                   "ground_truth": q["ground_truth"], "prompts": reports[first].get(qid, ("",))[0]}
            for s in SOURCES:
                _, ok, sheet = reports[s].get(qid, ("", False, ""))
                row[f"{s}_sheet"] = str(Path(sheet).resolve().relative_to(out)) if sheet else ""
                row[f"{s}_ok"] = ok
            index.append(row)
            match = COUNT_QUESTION.match(q["question"])
            if q["question_type"] == "object_counting" and match and all(row[f"{s}_sheet"] for s in sources):
                truth = int(float(q["ground_truth"]))
                for s in sources:
                    counting[s].append((f"{dataset}/{scene}", match.group(1), truth,
                                        count(out / row[f"{s}_sheet"], match.group(1))))

    with open(out / "index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, ["id", "scene", "question_type", "question", "options", "ground_truth", "prompts"]
                                + [f"{s}_sheet" for s in SOURCES]
                                + [f"{s}_ok" for s in SOURCES])
        writer.writeheader()
        writer.writerows(index)
    made = {s: sum(r[f"{s}_ok"] for r in index) for s in SOURCES}
    print(f"{out / 'index.csv'}: {len(index)} questions, sheets made: pipeline {made['pipeline']}, gold {made['gold']}")
    no_floor = sorted(r.name[:-len("_pipeline.txt")] for r in (out / "reports").glob("*_pipeline.txt")
                      if "WARNING: no floor plane found" in r.read_text())
    if no_floor:  # upright sheets whose up is the cameras' (upright.py)
        print(f"no floor plane found in {len(no_floor)} scenes: {', '.join(no_floor)}")

    for s in SOURCES:
        res = counting[s]
        if not res:
            continue
        print(f"{s} rows of the asked class vs the true count, {len(res)} counting questions:")
        print(f"  exact {sum(n == t for *_, t, n in res)}, too many {sum(n > t for *_, t, n in res)}, "
              f"too few {sum(0 < n < t for *_, t, n in res)}, no rows {sum(n == 0 for *_, n in res)}; "
              f"{sum(n for *_, n in res)} rows for {sum(t for *_, t, _ in res)} objects")
        worst = sorted(res, key=lambda r: r[3] - r[2], reverse=True)[:5]
        print("  most over-counted: " + "; ".join(f"{sc} {c} {n} vs {t}" for sc, c, t, n in worst if n > t))


if __name__ == "__main__":
    main()
