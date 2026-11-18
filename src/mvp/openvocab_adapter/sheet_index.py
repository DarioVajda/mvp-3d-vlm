"""An index of the scene sheets made by hpc/run_sheets.sh, and a check of the pipeline's object counts.

Reads the per-scene reports in out/reports/ (scene_sheet_report.py's output) and writes out/index.csv, one row per
question: its VSI-Bench id, scene, type, text, options and answer, the prompts taken from it, and the folders of its
pipeline and gold sheets (relative to out/) with whether each was made.

Then, for each object-counting question ("How many <class>(s) are in this room?"), compares the rows of that class in
the pipeline sheet, and in the gold sheet, with the true answer. The gold sheets should almost always be exact, which
checks the comparison itself. Standard library only, so it runs anywhere:

  python src/mvp/openvocab_adapter/sheet_index.py --data-root <the exported scenes>
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True, help="the exported scenes (scenes/<dataset>/<scene>/questions.jsonl)")
    ap.add_argument("--out", default=str(OUT), help="the sheets' root, with reports/ in it")
    args = ap.parse_args()
    out, data_root = Path(args.out).resolve(), Path(args.data_root)

    index, counting = [], defaultdict(list)
    for report in sorted((out / "reports").glob("*_pipeline.txt")):
        name = report.name[:-len("_pipeline.txt")]
        dataset, scene = name.split("__", 1)
        with open(data_root / "scenes" / dataset / scene / "questions.jsonl") as f:
            questions = {str(q["id"]): q for q in map(json.loads, f)}
        reports = {s: read_report(out / "reports" / f"{name}_{s}.txt") for s in SOURCES}
        for qid, q in questions.items():
            row = {"id": qid, "scene": f"{dataset}/{scene}", "question_type": q["question_type"],
                   "question": q["question"], "options": " | ".join(q.get("options") or []),
                   "ground_truth": q["ground_truth"], "prompts": reports["pipeline"].get(qid, ("",))[0]}
            for s in SOURCES:
                _, ok, sheet = reports[s].get(qid, ("", False, ""))
                row[f"{s}_sheet"] = str(Path(sheet).resolve().relative_to(out)) if sheet else ""
                row[f"{s}_ok"] = ok
            index.append(row)
            match = COUNT_QUESTION.match(q["question"])
            if q["question_type"] == "object_counting" and match and row["pipeline_sheet"] and row["gold_sheet"]:
                truth = int(float(q["ground_truth"]))
                for s in SOURCES:
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
