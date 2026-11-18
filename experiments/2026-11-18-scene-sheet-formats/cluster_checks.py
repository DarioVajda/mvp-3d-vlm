"""Checks for the README's Runs checklist, run on the cluster before the 11 runs. Each exits non-zero on failure.

  python cluster_checks.py prompts LOG --sheets-root DIR [--vggt-root DIR]   # output of vsibench --print-prompt
  python cluster_checks.py smoke A.jsonl B.jsonl --n 20                       # smoke runs: video only, with a sheet
  python cluster_checks.py revision Qwen/Qwen3.5-2B OUT                       # the cached snapshot the runs load

prompts: for each printed question, the video's "<t seconds>" labels must be exactly the sheet's frame timestamps
(scene_sheet.frame_timestamps, one per pair of frames), every first_seen in the sheet must be one of them, the sheet
must appear in the prompt as render_sheet writes it, and the whole prompt must be build_sheet_prompt's.
"""
import argparse
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from mvp.benchmarks import scene_sheet as ss
from mvp.benchmarks.vsibench import parsed
from mvp.paths import VGGT_DIR

HEADER = re.compile(r"^===== question (\d+) \((\S+)\), (\S+), video fps (\S+), frames (\[.*\])$", re.M)
LABEL = re.compile(r"<(\d+(?:\.\d+)?) seconds>")
FRAME_TOKENS = re.compile(re.escape(ss.VIDEO_PAD) + r" x (\d+)")


def check_prompts(text, sheets_root, vggt_root, source="pipeline", fmt="csv"):
    """Problems found in --print-prompt output, as a list of strings (empty: all good)."""
    heads = list(HEADER.finditer(text))
    if not heads:
        return ["no '===== question' blocks in the output"]
    problems = []
    for head, nxt in zip(heads, heads[1:] + [None]):
        qid = int(head.group(1))
        block = text[head.end():nxt.start() if nxt else len(text)]
        labels = LABEL.findall(block)
        row = ss.load_index(sheets_root)[qid]
        folder = Path(sheets_root) / row["pipeline_sheet"]
        times = ss.frame_timestamps(folder.parents[1] / "scene", row["scene"], Path(vggt_root))
        expected = list(times[::ss.MERGE])
        objects = ss.load_objects(qid, source, sheets_root, vggt_root=vggt_root)
        seen = sorted({o["first_seen"] for o in objects if "first_seen" in o}, key=float)
        fps = json.loads((Path(vggt_root) / row["scene"] / "meta.json").read_text())["video_fps"]
        print(f"question {qid}: prompt labels {labels}; sheet timestamps {expected}; first_seen {seen}; "
              f"fps {head.group(4)} (meta.json {fps})")
        if labels != expected:
            problems.append(f"question {qid}: the prompt's labels {labels} are not the sheet's {expected}")
        if not set(seen) <= set(labels):
            problems.append(f"question {qid}: first_seen {sorted(set(seen) - set(labels))} not among the labels")
        if ss.render_sheet(qid, source, fmt, sheets_root, vggt_root=vggt_root) not in block:
            problems.append(f"question {qid}: the rendered sheet is not in the prompt")
        tokens = FRAME_TOKENS.search(block)
        expected_prompt = ss.build_sheet_prompt(qid, source, fmt, sheets_root, vggt_root,
                                                frame_tokens=tokens.group(1) if tokens else None)
        if block.strip("\n") != expected_prompt.strip("\n"):
            problems.append(f"question {qid}: the prompt is not build_sheet_prompt's")
    return problems


def check_smoke(video, sheet, n):
    """Problems in the two smoke runs: n predictions each, on the same questions, at most 10% unparsed, and the
    sheet run's prompts longer than the video run's."""
    problems, runs = [], {}
    for name, path in [("video", video), ("sheet", sheet)]:
        if not Path(path).exists():
            return [f"{name}: {path} was not written"]
        runs[name] = run = pd.read_json(path, lines=True)
        ok = sum(parsed(r, r["prediction"]) for _, r in run.iterrows())
        print(f"{name}: {len(run)} predictions, {ok} parsed, mean input tokens {run['n_input_tokens'].mean():.0f}")
        if len(run) != n or run["id"].nunique() != n:
            problems.append(f"{name}: {len(run)} predictions ({run['id'].nunique()} questions), expected {n}")
        if ok < 0.9 * len(run):
            problems.append(f"{name}: only {ok} of {len(run)} answers parsed")
    if set(runs["video"]["id"]) != set(runs["sheet"]["id"]):
        problems.append("the two runs answered different questions")
    extra = runs["sheet"]["n_input_tokens"].mean() - runs["video"]["n_input_tokens"].mean()
    print(f"the sheet adds {extra:.0f} input tokens on average")
    if extra <= 0:
        problems.append("the sheet run's prompts are not longer than the video run's: was the sheet added?")
    return problems


def record_revision(model, out):
    """Write the snapshot of `model` in the local Hugging Face cache (what from_pretrained loads) to out."""
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(model, local_files_only=True))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(f"model: {model}\nrevision: {path.name}\nsnapshot: {path}\ncode commit: {commit}\n"
                         f"recorded: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
    print(Path(out).read_text(), end="")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("log")
    p.add_argument("--sheets-root", required=True)
    p.add_argument("--vggt-root", default=str(VGGT_DIR))
    p.add_argument("--format", default="csv")
    s = sub.add_parser("smoke")
    s.add_argument("video")
    s.add_argument("sheet")
    s.add_argument("--n", type=int, default=20)
    r = sub.add_parser("revision")
    r.add_argument("model")
    r.add_argument("out")
    args = ap.parse_args()
    if args.cmd == "revision":
        return record_revision(args.model, args.out)
    if args.cmd == "prompts":
        problems = check_prompts(Path(args.log).read_text(), Path(args.sheets_root), Path(args.vggt_root),
                                 fmt=args.format)
    else:
        problems = check_smoke(args.video, args.sheet, args.n)
    for problem in problems:
        print("FAIL:", problem)
    if problems:
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
