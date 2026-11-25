"""v5's stage 1 (README, "v5 on the dev scenes"): pipeline_v4 and each v5 configuration on dev_scenes_15.txt, against
gold and VSI-Bench's answers. Exploratory: these are tuning scenes, the questions are few, and nothing here is a test.

  PYTHONPATH=src python experiments/2026-11-18-scene-sheet-formats/v5_stage1_analysis.py \
      --v4 data/rerun_sheets/pipeline_v4_dev50 --v5-root data/rerun_sheets \
      --out experiments/2026-11-18-scene-sheet-formats/analyses/v5_stage1.md

Each sheets folder is a v5_postprocess.py / relations_v4.py output (index.csv, pipeline and gold sheet folders, the
relations file), already in metres. Per configuration, over the scenes:

  sheet quality   named objects found, rows per real object (pipeline rows of a class with one gold object, mean),
                  count = gold, the box's longest side over gold's (median) and the share over 1.5x, the closest-point
                  gap error between such boxes (pipeline_scorecard.score_root), and the position error after one
                  similarity alignment per scene (explore_sheets.align: the aligned centre of each class with one gold
                  object against gold's, median metres; scenes where it finds no fit are counted)
  relations       on the scenes' distance questions, from the relation text the model is given (it copies it):
                  absolute distance, VSI-Bench's MRA and the median prediction / truth; relative distance, the share
                  where the stated closest object is the answer; each over the questions that have a relation, and the
                  share that have one
  simulated       the distance questions' score if the model copies the relation where there is one and otherwise
                  answers as video only does: on the questions with a video-only answer (--video, in order: the
                  first file that has a question is used)

The go / no-go criteria for stage 2 are in the README and evaluated by go_no_go() below; they were written before any
stage-1 result.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import explore_sheets as es  # noqa: E402
import pipeline_scorecard as sc  # noqa: E402

CONFIGS = ("v5-his", "v5-det", "v5-pts", "v5-ours")
DIST = ("object_abs_distance", "object_rel_distance")
VIDEO = (HERE / "results/dev200/A_video.jsonl",)  # video only, 16 frames uniform128, as every sheet condition
RELATIONS = ("relations_v5.json", "relations_v4.json")
# README, "v5 on the dev scenes", stage-2 criteria (fixed before stage 1's results)
FOUND_MARGIN = 2.0          # points of named objects found v5 may lose against v4
INFLATED_MARGIN = 5.0       # points of boxes over 1.5x v5 may add against v4


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mra(pred, truth):
    return float(np.mean([abs(pred - truth) / truth < 1 - t for t in np.arange(0.5, 1.0, 0.05)]))


def relation_answer(question, text):
    """What a relation text says the answer is: a distance in metres, or an option letter; None without one."""
    if not text:
        return None
    if question["question_type"] == "object_abs_distance":
        m = re.search(r"Closest-point distance between .*?: ([0-9.]+) m", text)
        return float(m.group(1)) if m else None
    m = re.search(r"Closest to the .*?: (.+)\.$", text.strip().splitlines()[-1])
    if not m:
        return None
    for option in question["options"].split(" | "):
        letter, word = option.split(". ", 1)
        if word.strip() == m.group(1).strip():
            return letter
    return None


def relation_score(question, answer):
    if question["question_type"] == "object_abs_distance":
        return mra(answer, float(question["ground_truth"]))
    return float(answer == question["ground_truth"])


def relations(root):
    for name in RELATIONS:
        if (Path(root) / name).is_file():
            return json.loads((Path(root) / name).read_text())
    raise SystemExit(f"{root}: no relations file ({', '.join(RELATIONS)})")


def aligned_errors(root, questions):
    """Per scene, the position errors of the classes with one gold object after one similarity alignment; None where
    the alignment finds no fit."""
    by_scene = {}
    for q in questions:
        by_scene.setdefault(q["scene"], []).append(q)
    out = {}
    for scene, qs in sorted(by_scene.items()):
        gold = es.classes(sorted({Path(root) / q["gold_sheet"] for q in qs}))
        pipe = es.classes(sorted({Path(root) / q["pipeline_sheet"] for q in qs}))
        fit = es.align(gold, pipe)
        if fit is None:
            out[scene] = None
            continue
        s, R, t, _, _ = fit
        out[scene] = [float(np.linalg.norm(s * R @ np.median([c for c, _ in pipe[k]], axis=0) + t - gold[k][0][0]))
                      for k in gold if len(gold[k]) == 1 and pipe.get(k) and k != "room"]
    return out


def video_answers(paths):
    scores = {}
    for path in paths:
        if Path(path).is_file():
            for r in map(json.loads, Path(path).read_text().splitlines()):
                scores.setdefault(int(r["id"]), float(r["score"]))
    return scores


def measures(root, scenes, video):
    index = [q for q in read(Path(root) / "index.csv") if q["scene"] in scenes]
    missing = set(scenes) - {q["scene"] for q in index}
    if missing:
        raise SystemExit(f"{root}: scenes not in index.csv: {sorted(missing)}")
    recs = sc.score_root(root, scenes)
    flat = lambda k: [x for r in recs for x in r[k]]
    align = aligned_errors(root, index)
    rel = relations(root)
    m = {"named objects found": 100 * np.mean(flat("found")),
         "rows per real object": float(np.mean(flat("unique_rows"))) if flat("unique_rows") else float("nan"),
         "count = gold": 100 * np.mean(flat("count_match")),
         "box side / gold, median": float(np.median(flat("box_ratio"))) if flat("box_ratio") else float("nan"),
         "boxes over 1.5x": 100 * np.mean(flat("box_inflated")) if flat("box_inflated") else float("nan"),
         "gap error, median (m)": float(np.median(flat("gap_abs"))) if flat("gap_abs") else float("nan"),
         "aligned position error, median (m)": float(np.median([e for v in align.values() if v for e in v]))
         if any(align.values()) else float("nan"),
         "scenes aligned": f"{sum(v is not None for v in align.values())}/{len(align)}"}
    sims, per_q = [], {}
    for task, short in (("object_abs_distance", "abs"), ("object_rel_distance", "rel")):
        qs = [q for q in index if q["question_type"] == task]
        answers = {int(q["id"]): relation_answer(q, rel.get(q["id"])) for q in qs}
        have = [q for q in qs if answers[int(q["id"])] is not None]
        scores = [relation_score(q, answers[int(q["id"])]) for q in have]
        m[f"_{short}"] = {int(q["id"]): sc_ for q, sc_ in zip(have, scores)}  # per question, for paired comparisons
        m[f"{short}: questions with a relation"] = f"{len(have)}/{len(qs)}"
        if task == "object_abs_distance":
            m["abs: MRA of the relation"] = 100 * np.mean(scores) if scores else float("nan")
            m["abs: relation / truth, median"] = float(np.median([answers[int(q["id"])] / float(q["ground_truth"])
                                                                  for q in have])) if have else float("nan")
        else:
            m["rel: closest object right"] = 100 * np.mean(scores) if scores else float("nan")
        for q in qs:
            i = int(q["id"])
            if i in video:
                s = relation_score(q, answers[i]) if answers[i] is not None else video[i]
                sims.append(s)
                per_q[i] = s
    m["simulated distance score"] = 100 * np.mean(sims) if sims else float("nan")
    m["video-only distance score"] = 100 * np.mean([video[i] for i in per_q]) if per_q else float("nan")
    m["simulated: questions"] = len(sims)
    return m


def paired(results, name, key):
    """(name's mean, v4's mean, n) of a per-question relation score over the questions both have a relation for."""
    a, b = results[name][key], results["v4"][key]
    common = sorted(set(a) & set(b))
    if not common:
        return float("nan"), float("nan"), 0
    return 100 * np.mean([a[q] for q in common]), 100 * np.mean([b[q] for q in common]), len(common)


def go_no_go(results):
    """The README's stage-2 criteria: the best v5 configuration (the highest simulated distance score; ties: fewer
    boxes over 1.5x, then the order of CONFIGS) against pipeline_v4. Relation accuracy is compared on the questions
    both have a relation for (a sheet that states fewer relations is not credited for it; the simulated score counts
    coverage). A criterion with no question to compare on fails. Returns (config, {criterion: passed}, go)."""
    v4 = results["v4"]
    order = [c for c in CONFIGS if c in results]
    best = max(order, key=lambda c: (results[c]["simulated distance score"], -results[c]["boxes over 1.5x"],
                                     -order.index(c)))
    b = results[best]
    ratio_gap = lambda r: abs(np.log(r["box side / gold, median"]))
    checks = {
        "simulated distance score >= v4's": b["simulated distance score"] >= v4["simulated distance score"],
        "abs: MRA of the relation >= v4's, on the questions both have a relation for":
            (lambda x: x[2] > 0 and x[0] >= x[1])(paired(results, best, "_abs")),
        "rel: closest object right >= v4's, on the questions both have a relation for":
            (lambda x: x[2] > 0 and x[0] >= x[1])(paired(results, best, "_rel")),
        "box side / gold at least as close to 1 as v4's": ratio_gap(b) <= ratio_gap(v4),
        f"boxes over 1.5x at most v4's + {INFLATED_MARGIN:g} points": b["boxes over 1.5x"] <= v4["boxes over 1.5x"] + INFLATED_MARGIN,
        f"named objects found at least v4's - {FOUND_MARGIN:g} points": b["named objects found"] >= v4["named objects found"] - FOUND_MARGIN,
    }
    return best, checks, all(checks.values())


def fmt(v):
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return str(v)
    return "-" if v != v else f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}"


def report(results, scenes, video_paths):
    names = list(results)
    keys = [k for k in next(iter(results.values())) if not k.startswith("_")]
    lines = ["# v5 stage 1: pipeline_v4 and the four configurations on dev_scenes_15", "",
             "<!-- Written by v5_stage1_analysis.py. -->", "",
             f"**Exploratory: {len(scenes)} tuning scenes (dev_scenes_15.txt), no test.** The configurations were made to "
             "be chosen on these scenes; the differences are not significance-tested, and the simulated score rests "
             "on the few distance questions with a video-only answer.", "",
             "| Measure | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(fmt(results[n][k]) for n in names) + " |")
    for key, label in (("_abs", "abs: MRA, paired with v4 (v5 / v4, n)"), ("_rel", "rel: closest right, paired with v4 (v5 / v4, n)")):
        cells = ["-"] + [(lambda x: f"{fmt(x[0])} / {fmt(x[1])}, {x[2]}")(paired(results, n, key)) for n in names[1:]]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    best, checks, go = go_no_go(results)
    lines += ["", f"## Stage 2: {'GO' if go else 'NO-GO'} (README's criteria, best v5 configuration: {best})", ""]
    lines += [f"- {'pass' if ok else 'FAIL'}: {k}" for k, ok in checks.items()]
    lines += ["", f"Video-only answers: {', '.join(str(p) for p in video_paths)}. Percentages are of questions or "
              "(question, object) pairs; sizes and errors in metres."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v4", required=True, help="pipeline_v4's sheets of the dev scenes (relations_v4.json)")
    ap.add_argument("--v5-root", required=True, help="a folder with one v5_postprocess.py output per configuration")
    ap.add_argument("--scenes", default=str(HERE / "dev_scenes_15.txt"))
    ap.add_argument("--video", nargs="+", default=[str(p) for p in VIDEO])
    ap.add_argument("--out", default=str(HERE / "analyses/v5_stage1.md"))
    args = ap.parse_args()
    scenes = set(Path(args.scenes).read_text().split())
    video = video_answers(args.video)
    roots = {"v4": Path(args.v4), **{c: Path(args.v5_root) / c for c in CONFIGS}}
    results = {n: measures(r, scenes, video) for n, r in roots.items()}
    text = report(results, scenes, args.video)
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
