"""One table of every scene's comparison: results/<dataset>__<scene>/<selection>/comparison.json -> results/SUMMARY.md.

  python tasks/openvocab_adapter/summarize.py [--results tasks/openvocab_adapter/results]

A scene whose job failed before comparing shows as "no result", with the end of its run.log to say why.
"""
import argparse
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
SELECTION_ORDER = ["smoke5", "eval32", "half", "all"]


def value(rows, comparison, key):
    return rows.get(comparison, {}).get("values", {}).get(key)


def fmt(number, percent=False):
    if number is None:
        return "-"
    return f"{100 * number:.1f}%" if percent else f"{number:g}"


def total(counts):
    return sum(counts.values()) if isinstance(counts, dict) else None


def scene_rows(scene_dir):
    """Table rows for one scene's selections; a failure row when there are none."""
    selections = sorted((d for d in scene_dir.iterdir() if (d / "comparison.json").is_file()),
                        key=lambda d: (SELECTION_ORDER.index(d.name) if d.name in SELECTION_ORDER else 99, d.name))
    scene = scene_dir.name.replace("__", "/")
    if not selections:
        log = scene_dir / "run.log"
        tail = log.read_text(errors="replace").strip().splitlines()[-1:] if log.is_file() else ["no run.log"]
        return [f"| {scene} | - | no result | | | | | | `{tail[0][:80] if tail else ''}` |"], ["no result"]
    lines, verdicts = [], []
    for selection in selections:
        report = json.loads((selection / "comparison.json").read_text())
        rows = {r["comparison"]: r for r in report["comparisons"]}
        counts = value(rows, "object labels: objects per concept", "adapted"), \
            value(rows, "object labels: objects per concept", "native")
        fails = [r["comparison"] for r in report["comparisons"] if r["status"] == "FAIL"]
        verdicts.append(report["verdict"])
        lines.append("| " + " | ".join([
            scene, selection.name, f"**{report['verdict']}**",
            f"{fmt(total(counts[0]))} / {fmt(total(counts[1]))}",
            fmt(value(rows, "tracking consistency", "pixels_with_matching_identity"), True),
            fmt(value(rows, "3D object positions after alignment", "median_error_of_scene"), True),
            fmt(value(rows, "scene sheet: objects", "matched_share"), True),
            fmt(value(rows, "scene sheet: objects", "median_position_error_of_scene"), True),
            ", ".join(fails) or "",
        ]) + " |")
    return lines, verdicts


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(RESULTS))
    args = p.parse_args(argv)
    results = Path(args.results)
    scenes = sorted(d for d in results.iterdir() if d.is_dir()) if results.is_dir() else []
    if not scenes:
        sys.exit(f"[summary] no scene folders in {results}")

    table, verdicts = [], []
    for scene_dir in scenes:
        lines, found = scene_rows(scene_dir)
        table += lines
        verdicts += found
    tally = {v: verdicts.count(v) for v in ("MATCHES", "DIFFERS", "INCOMPLETE", "no result") if verdicts.count(v)}
    text = "\n".join([
        "# OpenVocab-4D: Dario's VGGT (adapted) vs its own VGGT (native)",
        "",
        f"{len(scenes)} scenes, {len(verdicts)} comparisons: "
        + ", ".join(f"{n} {v}" for v, n in tally.items()) + ". Pass marks are provisional (`CRITERIA` in compare.py).",
        "",
        "Columns: objects found (adapted / native); share of tracked pixels with the same identity (pass >= 90%); "
        "median 3D object position error as a share of the scene size (pass <= 5%); share of scene-sheet rows in "
        "both sheets (pass >= 80%); median scene-sheet position error (pass <= 5%).",
        "",
        "| Scene | Frames | Verdict | Objects | Same track | Position error | Sheet rows | Sheet error | FAIL checks |",
        "|---|---|---|---|---|---|---|---|---|",
        *table,
        "",
    ])
    (results / "SUMMARY.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"[summary] -> {results / 'SUMMARY.md'}")
    print(f"commit with: git add {results} && git commit -m 'OpenVocab-4D comparison results' && git push")


if __name__ == "__main__":
    main()
