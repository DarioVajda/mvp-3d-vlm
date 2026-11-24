"""Render conditions B (gold, csv) and C (gold_relations, csv) for every question of a sheets root, exactly as the eval
renders them (scene_sheet.render_sheet), without a model: it fails if any question has no index row or does not
render, and prints per question type how many questions get relations and the sheets' lengths in characters.

  PYTHONPATH=src python experiments/2026-11-19-gold-sheet-relations/check_sheets.py --sheets-root <root> \
      --questions data/raw/VSI-Bench/test.jsonl [--ids <question ids>]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from mvp.benchmarks import scene_sheet as ss


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheets-root", required=True)
    ap.add_argument("--questions", required=True, help="VSI-Bench's test.jsonl")
    ap.add_argument("--ids", help="only these question ids (one per line)")
    args = ap.parse_args()
    root = Path(args.sheets_root)
    with open(args.questions) as f:
        questions = {int(q["id"]): q for q in map(json.loads, f)}
    if args.ids:
        keep = {int(x) for x in Path(args.ids).read_text().split()}
        questions = {q: v for q, v in questions.items() if q in keep}
    index = ss.load_index(root)
    problems, total, related, length = [], Counter(), Counter(), defaultdict(list)
    for q, row in sorted(questions.items()):
        kind = row["question_type"]
        total[kind] += 1
        if q not in index:
            problems.append(f"{q}: no index row")
            continue
        if index[q]["gold_ok"] != "True":
            problems.append(f"{q}: the gold sheet was not made")
        try:
            plain = ss.render_sheet(q, "gold", "csv", root)
            with_relations = ss.render_sheet(q, ss.RELATIONS_SOURCE, "csv", root)
        except Exception as error:  # noqa: BLE001 - every failure is listed
            problems.append(f"{q}: {type(error).__name__}: {error}")
            continue
        if not with_relations.startswith(plain):
            problems.append(f"{q}: C's sheet does not start with B's")
        related[kind] += with_relations != plain
        length["B"].append(len(plain))
        length["C"].append(len(with_relations))
    print(f"{sum(total.values())} questions; with relations in C: {sum(related.values())}")
    for kind in sorted(total):
        print(f"  {kind}: {related[kind]}/{total[kind]}")
    for c, values in length.items():
        values = sorted(values)
        print(f"{c}: sheet length in characters, median {values[len(values) // 2]}, max {values[-1]}")
    for problem in problems[:20]:
        print("FAIL:", problem)
    print(f"{len(problems)} problems" if problems else "OK: every question renders in B and C")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
