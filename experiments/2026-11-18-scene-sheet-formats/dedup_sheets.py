"""Question-conditioned dedup of a pipeline sheets folder (pipeline_v3_dedup).

Relative direction, relative distance and absolute distance questions ask about objects the scene has once. For each
of them, only the best row of each object the question names is kept: the one seen in the most frames; among rows
tied on that, for distance questions the one nearest the other named objects (best_row; on dev200 this picked the
right closest candidate more often, while for direction questions it picked the wrong instance), then the one whose
track has the highest detection score (tracks/<prompt>.scores.json of the run the sheets were made from; a merged row keeps the smallest track id of its
tracks, so its score is that track's), then the larger box. A question whose
sheet names an object more than once gets a sheet folder of its own (<sheet folder>__q<id>, next to the original) and
its index.csv row points to it; every other question, and every row of an object the question does not name, is
unchanged. Gold sheets and the scale (scene_scale.SCALE_ROWS) are copied as they are.

  python experiments/2026-11-18-scene-sheet-formats/dedup_sheets.py --sheets <pipeline_v3 sheets> \
      --runs <run folder with the tracks> [<run folder> ...] --out <new sheets folder>
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

DEDUP_TYPES = {"object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard",
               "object_rel_distance", "object_abs_distance"}
NEAR_TYPES = {"object_rel_distance", "object_abs_distance"}  # ties on frames broken by nearness to the other named objects


def named_objects(row):
    return [p.strip() for p in row["prompts"].split(",") if p.strip() and p.strip() != "room"]


def read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
        return rows, (rows[0].keys() if rows else csv.DictReader(open(path)).fieldnames)


def track_scores(runs, scene_dir, concept):
    """{track id: best detection score} of a prompt's tracks, from the first run folder that has them."""
    for run in runs:
        path = Path(run) / scene_dir / "subset-uniform16/scene/tracks" / f"{concept.replace(' ', '_')}.scores.json"
        if path.is_file():
            return {int(t): s for t, s in json.loads(path.read_text()).items()}
    return {}


def candidates(rows):
    """The rows seen in the most frames."""
    most = max(int(r["num_frames"]) for r in rows)
    return [r for r in rows if int(r["num_frames"]) == most]


def best_row(rows, scores, others=()):
    """The row seen in the most frames; among rows tied on that, the one nearest the other named objects (the sum,
    over the other named objects, of the distance to their nearest candidate row), then the highest track score,
    then the larger box."""
    pos = lambda r: [float(r[k]) for k in ("x", "y", "z")]
    near = lambda r: sum(min(sum((a - b) ** 2 for a, b in zip(pos(r), pos(o))) ** 0.5 for o in other)
                         for other in others if other)
    volume = lambda r: float(r["size_x"]) * float(r["size_y"]) * float(r["size_z"])
    return min(candidates(rows), key=lambda r: (near(r), -scores.get(int(r["track_id"]), 0.0), -volume(r)))


def dedup(root, runs, out):
    """Write the deduplicated copy of root in out. Returns (questions given a sheet of their own, rows dropped)."""
    root, out = Path(root), Path(out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(root, out, ignore=shutil.ignore_patterns("tracks", "masks", "frames", "depth.npz", "cameras.npz"))
    with open(root / "index.csv", newline="") as f:
        reader = csv.DictReader(f)
        fields, index = reader.fieldnames, list(reader)
    changed = dropped = 0
    for row in index:
        if row["question_type"] not in DEDUP_TYPES:
            continue
        folder = Path(row["pipeline_sheet"])
        src = root / folder / "objects.csv"
        if not src.is_file():
            continue
        rows, columns = read_csv(src)
        names = set(named_objects(row))
        cands = {n: candidates([r for r in rows if r["concept"] == n]) for n in names if any(r["concept"] == n for r in rows)}
        keep = []
        for name in dict.fromkeys(r["concept"] for r in rows):
            mine = [r for r in rows if r["concept"] == name]
            if name in names and len(mine) > 1:
                others = [c for n, c in cands.items() if n != name] if row["question_type"] in NEAR_TYPES else ()
                keep.append(best_row(mine, track_scores(runs, folder.parts[0], name), others))
            else:
                keep += mine
        if len(keep) == len(rows):
            continue
        keep.sort(key=rows.index)  # the sheet's own row order
        own = folder.parent / f"{folder.name}__q{row['id']}"
        shutil.copytree(root / folder, out / own)
        with open(out / own / "objects.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(columns))
            w.writeheader()
            w.writerows(keep)
        row["pipeline_sheet"] = str(own)
        changed += 1
        dropped += len(rows) - len(keep)
    with open(out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(index)
    return changed, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheets", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="run folders with tracks/<prompt>.scores.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    changed, dropped = dedup(args.sheets, args.runs, args.out)
    print(f"{changed} questions given a sheet of their own, {dropped} duplicate rows of named objects dropped -> {args.out}")


if __name__ == "__main__":
    main()
