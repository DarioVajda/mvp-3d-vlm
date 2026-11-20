"""Check the eval's --print-prompt output against scene_sheet.build_sheet_prompt, for any condition of this experiment.

  python -m mvp.eval.vsibench ... [--sheet-source S] [--reasoning cot] --print-prompt N > prompts.txt
  PYTHONPATH=src python experiments/2026-11-19-structured-cot/check_prompts.py prompts.txt --source S --reasoning cot \
      [--sheets-root DIR] [--vggt-root DIR]

cluster_checks.check_prompts (2026-11-19-broad-exploration-of-approaches, now in 2026-11-18-scene-sheet-formats)
rebuilds every prompt with direct reasoning and needs a sheet, so for the CoT conditions it reports a mismatch on every
question and it cannot check the video-only ones. This passes --reasoning through to build_sheet_prompt and takes
--source none, so every condition (A–E) is checked whole: the chat template, the video's "<t seconds>" labels, the sheet
and the question with its instructions. Exits non-zero on any mismatch.
"""
import argparse
import re
import sys
from pathlib import Path

from mvp.benchmarks import scene_sheet as ss
from mvp.benchmarks.vsibench import REASONING
from mvp.paths import SCENE_SHEETS, VGGT_DIR

HEADER = re.compile(r"^===== question (\d+) \((\S+)\), (\S+), video fps (\S+), frames (\[.*\])$", re.M)
FRAME_TOKENS = re.compile(re.escape(ss.VIDEO_PAD) + r" x (\d+)")


def check(text, source, reasoning, sheets_root, vggt_root, fmt="csv"):
    """(number of prompts checked, problems as strings)."""
    heads = list(HEADER.finditer(text))
    if not heads:
        return 0, ["no '===== question' blocks in the output"]
    problems = []
    for head, nxt in zip(heads, heads[1:] + [None]):
        qid = int(head.group(1))
        block = text[head.end():nxt.start() if nxt else len(text)].strip("\n")
        tokens = FRAME_TOKENS.search(block)
        expected = ss.build_sheet_prompt(qid, source, fmt if source else None, sheets_root, vggt_root,
                                         frame_tokens=tokens.group(1) if tokens else None, reasoning=reasoning)
        if block != expected.strip("\n"):
            problems.append(f"question {qid}: the prompt is not build_sheet_prompt's (source {source}, {reasoning})")
    return len(heads), problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--source", default="none", choices=["none"] + ss.SHEET_SOURCES)
    ap.add_argument("--reasoning", default="direct", choices=REASONING)
    ap.add_argument("--sheets-root", default=str(SCENE_SHEETS))
    ap.add_argument("--vggt-root", default=str(VGGT_DIR))
    args = ap.parse_args()
    source = None if args.source == "none" else args.source
    n, problems = check(Path(args.log).read_text(), source, args.reasoning, Path(args.sheets_root), Path(args.vggt_root))
    for p in problems:
        print("FAIL:", p)
    print(f"{n} prompts checked, {len(problems)} problems")
    sys.exit(1 if problems or not n else 0)


if __name__ == "__main__":
    main()
