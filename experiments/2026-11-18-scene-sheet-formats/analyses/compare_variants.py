"""Score every sheets variant on the dev scenes with pipeline_scorecard.py's measures, in one table.

  PYTHONPATH=src python experiments/2026-11-18-scene-sheet-formats/analyses/compare_variants.py <work folder> <scene_sheets_75>
"""
import sys
from pathlib import Path

E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E))
import pipeline_scorecard as pc  # noqa: E402

S = Path(sys.argv[1])
M = Path(sys.argv[2])
scenes = {s for s in (E / "dev_scenes.txt").read_text().split() if s}
runs = {"Current (old sizing)": M,
        "Box merge 0.5 m": S / "merge_nocap",
        "Box merge overlap, 3×": S / "merge_cap3_overlap",
        "Point merge 0.5 m": S / "mp_old_merge",
        "Point merge 0.5 m, 3×": S / "mp_old_merge_cap3",
        "Oriented sizing": S / "rebuilt_dev",
        "Oriented + point merge": S / "mp_new_merge"}
runs = {n: r for n, r in runs.items() if (r / "index.csv").exists()}
recs = {n: pc.score_root(r, scenes) for n, r in runs.items()}
keep = ["count = gold, per (question, class)", "rows over the sheet folders (room fragments left out)",
        "rows seen in only one frame", "unmatched rows (beyond the gold sheet's count)", "size error, median (m)",
        "size error, median relative", "distance error, median (m)", "distance error, median relative",
        "rule: counting", "rule: object size", "rule: absolute distance", "rule: relative distance"]
lines = ["| Measure | " + " | ".join(runs) + " |", "|---|" + "---:|" * len(runs)]
for name, fn, kind, higher in pc.MEASURES:
    if name in keep:
        lines.append(f"| {name} | " + " | ".join(pc.fmt(fn(recs[n]), kind) for n in runs) + " |")
text = "\n".join(lines)
print(text)
(S / "variants_table.md").write_text(text + "\n")
