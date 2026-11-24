"""The gold sheets of every VSI-Bench question (288 scenes, 5,130 questions), and their index, on a CPU with numpy only.

For each scene of --questions, scene_sheet_report.py --source gold with the prompts of --prompts (extract_prompts.py's
output, so no spaCy here): VSI-Bench's labelled boxes, in metres, into <out>/<dataset>__<scene>/gold/sheets/<prompts>/,
the report into <out>/reports/<dataset>__<scene>_gold.txt. Then sheet_index.py --gold-only writes <out>/index.csv,
which mvp.eval.vsibench --sheets-root <out> reads. The layout is the 75-scene sheets' (hpc/run_sheets.sh) without
the pipeline sheets.

At the end it prints the counts, and two SHA-256: of index.csv and of every objects.csv (their relative paths and
contents, in sorted order), so that a build elsewhere can be checked against this one (the README gives the values).

  PYTHONPATH=src python experiments/2026-11-19-gold-sheet-relations/build_gold_sheets.py \
      --questions data/raw/VSI-Bench/test.jsonl --prompts experiments/2026-11-19-gold-sheet-relations/prompts_full.jsonl \
      --out tasks/openvocab_adapter/out_full [--gold-dir data/raw/VSI-Bench/meta_info]
"""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

ADAPTER = Path(__file__).resolve().parents[2] / "src/mvp/openvocab_adapter"
sys.path.insert(0, str(ADAPTER))
import scene_sheet_report  # noqa: E402
import sheet_index  # noqa: E402


def sheets_digest(out):
    """SHA-256 over every objects.csv under out: relative path and bytes, in sorted order."""
    h = hashlib.sha256()
    for path in sorted(Path(out).glob("*/gold/sheets/*/objects.csv")):
        h.update(str(path.relative_to(out)).encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", required=True, help="VSI-Bench's test.jsonl")
    ap.add_argument("--prompts", required=True, help="extract_prompts.py's .jsonl")
    ap.add_argument("--out", required=True, help="the new sheets root")
    ap.add_argument("--gold-dir", help="VSI-Bench's metadata (default: as scene_sheet_creator.py, downloaded if missing)")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    (out / "reports").mkdir(parents=True, exist_ok=True)
    prompts = scene_sheet_report.read_prompts(args.prompts)
    with open(args.questions) as f:
        scenes = sorted({f"{q['dataset']}/{q['scene_name']}" for q in map(json.loads, f)})
    failed = Counter()
    for i, scene in enumerate(scenes, 1):
        name = scene.replace("/", "__", 1)
        with open(out / "reports" / f"{name}_gold.txt", "w") as report, contextlib.redirect_stderr(io.StringIO()):
            results = scene_sheet_report.report(scene, source="gold", questions=args.questions,
                                                work_dir=out / name / "gold", gold_dir=args.gold_dir, out=report,
                                                prompts=prompts)
        failed[scene] = sum(not r["ok"] for r in results)
        if i % 25 == 0 or i == len(scenes):
            print(f"[{i}/{len(scenes)}] {scene}", flush=True)
    sheet_index.main(["--gold-only", "--questions", args.questions, "--out", str(out)])
    with open(out / "index.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"{len(scenes)} scenes, {len(rows)} questions; sheets failed: {sum(failed.values())}"
          + (f" ({', '.join(f'{s} {n}' for s, n in failed.items() if n)})" if sum(failed.values()) else ""))
    print(f"index.csv sha256 {hashlib.sha256((out / 'index.csv').read_bytes()).hexdigest()}")
    print(f"objects.csv sha256 {sheets_digest(out)}")
    sys.exit(1 if sum(failed.values()) else 0)


if __name__ == "__main__":
    main()
