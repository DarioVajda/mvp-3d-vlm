"""Each VSI-Bench question's SAM 3 prompts, as the scene-sheet reports take them (question_scene_sheet.extract_prompts:
prompt_object_extractor.py's full mode, the syntactic mode when it keeps nothing), into one .jsonl that
scene_sheet_report.py --prompts and build_gold_sheets.py read, so that building the gold sheets needs no spaCy.

Needs spaCy 3.8 with en_core_web_md 3.8.0 and NLTK WordNet, the versions hpc/setup_env.sh installs (CPU only; on
Colab, prompts_full.ipynb):

  python experiments/2026-11-19-gold-sheet-relations/extract_prompts.py --questions data/raw/VSI-Bench/test.jsonl \
      --out experiments/2026-11-19-gold-sheet-relations/prompts_full.jsonl
"""
import argparse
import contextlib
import json
import sys
from collections import Counter
from pathlib import Path

ADAPTER = Path(__file__).resolve().parents[2] / "src/mvp/openvocab_adapter"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", required=True, help="VSI-Bench's test.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(ADAPTER))
    import question_scene_sheet as qss
    with open(args.questions) as f:
        rows = sorted(map(json.loads, f), key=lambda r: int(r["id"]))
    modes = Counter()
    with open(args.out, "w") as out:
        for i, row in enumerate(rows, 1):
            with contextlib.redirect_stdout(sys.stderr):
                prompts, mode = qss.extract_prompts(row["question"])
            modes[mode] += 1
            out.write(json.dumps({"id": int(row["id"]), "prompts": prompts, "prompt_mode": mode}) + "\n")
            if i % 500 == 0:
                print(f"{i}/{len(rows)}", flush=True)
    print(f"{len(rows)} questions -> {args.out}; prompt modes {dict(modes)}")


if __name__ == "__main__":
    main()
