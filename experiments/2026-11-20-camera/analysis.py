"""The camera pilot's predictions analysed as the README's Metrics and analysis section specifies, plus the answer-
letter check the results called for; writes analysis.md next to this file.

  PYTHONPATH=src python experiments/2026-11-20-camera/analysis.py

Reads results/zeroshot/vsibench/camera/explore200/ (explore_camera.ipynb's predictions and tables). The scores and
bootstrap intervals are recomputed from the predictions, as the notebook computes them (the same resampling and seed),
and checked against the notebook's scores.csv and differences_bootstrap.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mvp.benchmarks.vsibench import (MCA_QUESTION_TYPES, METRIC_TASKS, TASKS, final_letter, final_number, parsed,
                                     summarize, task_of)
from mvp.paths import ROOT

HERE = Path(__file__).parent
RESULTS = ROOT / "results/zeroshot/vsibench/camera/explore200"
CONDITIONS = ["video", "obj_csv", "cam_csv", "obj_text", "cam_text", "narrative", "obj_markdown_table",
              "cam_markdown_table", "cam_csv_frames", "cam_text_turn_move", "cam_text_frames", "narrative_turn_move",
              "narrative_camera_dirs", "narrative_frames", "narrative_exact"]
COMPARISONS = [("cam_csv", "obj_csv"), ("cam_markdown_table", "obj_markdown_table"), ("cam_text", "obj_text"),
               ("narrative", "cam_text"), ("narrative", "obj_text"), ("narrative", "obj_csv"),
               ("cam_csv_frames", "cam_csv"), ("cam_text_turn_move", "cam_text"), ("cam_text_frames", "cam_text"),
               ("narrative_turn_move", "narrative"), ("narrative_camera_dirs", "narrative"),
               ("narrative_frames", "narrative"), ("narrative_exact", "narrative")]
TYPE_ORDER = [t for k in TASKS for t in ([f"object_rel_direction_{x}" for x in ("easy", "medium", "hard")]
                                         if k == "object_rel_direction" else [k])]
DIR_ROUTE = {"object_rel_direction", "route_planning"}
GROUPS = {"overall": set(TASKS), "direction and route": DIR_ROUTE, "without metric tasks": set(TASKS) - set(METRIC_TASKS)}
LEAD = {"overall": 2, "direction and route": 5}
SHORT = {"object_counting": "Count", "object_abs_distance": "Abs. dist.", "object_size_estimation": "Obj. size",
         "room_size_estimation": "Room size", "object_rel_distance": "Rel. dist.", "object_rel_direction": "Rel. dir.",
         "route_planning": "Route", "obj_appearance_order": "Order"}
B, SEED = 10_000, 0


def load():
    runs = {n: pd.read_json(RESULTS / f"{n}.jsonl", lines=True).drop_duplicates("id").set_index("id").sort_index()
            for n in CONDITIONS}
    ids = list(runs["video"].index)
    assert len(ids) == 200 and all(list(r.index) == ids for r in runs.values()), "the conditions answered different questions"
    return runs


def group_score(run, tasks):
    keep = run["question_type"].map(task_of).isin(tasks)
    return summarize(run.loc[keep, ["question_type", "score"]])["overall"]


def scores(runs, sheet_tokens):
    rows = {}
    for n, r in runs.items():
        s = summarize(r[["question_type", "score"]])
        rows[n] = {**{g: group_score(r, ts) for g, ts in GROUPS.items()}, **{SHORT[k]: s[k] for k in TASKS},
                   "Sheet tokens": sheet_tokens[n]}
    return pd.DataFrame(rows).T


def bootstrap(runs):
    """The notebook's resampling: questions drawn with replacement within each question type (in TYPE_ORDER), the same
    draws for every condition. Returns (types, point: type x condition, draws: B x type x condition)."""
    types = runs["video"]["question_type"]
    qtypes = [t for t in TYPE_ORDER if t in set(types)]
    X = np.stack([runs[n]["score"].to_numpy(float) for n in CONDITIONS], axis=1)
    members = [np.flatnonzero(types.to_numpy() == t) for t in qtypes]
    rng = np.random.default_rng(SEED)
    point = np.stack([X[m].mean(0) for m in members])
    draws = np.stack([X[m][rng.integers(0, len(m), size=(B, len(m)))].mean(1) for m in members], axis=1)
    return qtypes, point, draws


def task_weights(qtypes, tasks):
    kept = [k for k in dict.fromkeys(task_of(t) for t in qtypes) if k in tasks]
    return np.mean([[(task_of(t) == k) / sum(task_of(u) == k for u in qtypes) for t in qtypes] for k in kept], axis=0)


def differences(qtypes, point, draws, pairs):
    """{'x − y': {row: (difference, low, high)}} for the groups and each question type."""
    W = {g: task_weights(qtypes, ts) for g, ts in GROUPS.items()}
    out = {}
    for x, y in pairs:
        i, j = CONDITIONS.index(x), CONDITIONS.index(y)
        dp, db = 100 * (point[:, i] - point[:, j]), 100 * (draws[:, :, i] - draws[:, :, j])
        rows = {**{g: (dp @ w, db @ w) for g, w in W.items()}, **{t: (dp[k], db[:, k]) for k, t in enumerate(qtypes)}}
        out[f"{x} − {y}"] = {r: (p, *np.percentile(b, [2.5, 97.5])) for r, (p, b) in rows.items()}
    return out


def fmt_interval(p, lo, hi):
    return f"{p:+.1f} [{lo:+.1f}, {hi:+.1f}]" + (" *" if lo > 0 or hi < 0 else "")


def lead(rows):
    return ", ".join(f"{g} {'+' if rows[g][0] > 0 else '−'}" for g, t in LEAD.items()
                     if abs(rows[g][0]) >= t and (rows[g][1] > 0 or rows[g][2] < 0))


def md_table(df, index_name=""):
    cols = [index_name] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" + "---:|" * len(df.columns)]
    lines += ["| " + " | ".join([f"`{i}`" if i in CONDITIONS or " − " in str(i) else str(i)] + [str(v) for v in row]) + " |"
              for i, row in zip(df.index, df.to_numpy())]
    return "\n".join(lines)


def letters(run, ids):
    return run.loc[ids, "prediction"].astype(str).map(final_letter)


def report():
    runs = load()
    v = runs["video"]
    ids = list(v.index)
    task = v["question_type"].map(task_of)
    info = json.loads((RESULTS / "run_info.json").read_text())
    reused = json.loads((RESULTS / "reused.json").read_text())
    timing = json.loads((RESULTS / "timing.json").read_text())
    lengths = pd.read_csv(RESULTS / "input_tokens.csv", index_col=0)
    sheet_tokens = lengths.sub(lengths["video"], axis=0).mean()

    # the scores, checked against the notebook's
    S = scores(runs, sheet_tokens)
    nb = pd.read_csv(RESULTS / "scores.csv", index_col=0)
    for g in GROUPS:
        assert np.allclose(S[g].astype(float), nb.loc[S.index, g]), f"scores differ from the notebook's: {g}"
    qtypes, point, draws = bootstrap(runs)
    W = task_weights(qtypes, GROUPS["overall"])
    assert np.allclose(100 * W @ point, S.loc[CONDITIONS, "overall"].astype(float))
    D = differences(qtypes, point, draws, COMPARISONS)
    VS = differences(qtypes, point, draws, [(n, "video") for n in CONDITIONS if n != "video"])
    nbd = pd.read_csv(RESULTS / "differences_bootstrap.csv").set_index(["comparison", "rows"])
    for c, rows in {**D, **VS}.items():
        for r, (p, lo, hi) in rows.items():
            q = nbd.loc[(c, r)]
            assert np.allclose([p, lo, hi], [q["difference"], q["95% low"], q["95% high"]]), f"{c}, {r}"

    replies = pd.DataFrame({n: {"parsed %": 100 * np.mean([parsed(d, p, final=True) for d, p in
                                                          zip(r.reset_index().to_dict("records"), r["prediction"].astype(str))]),
                                "longest reply (tokens)": int(r["n_new_tokens"].max())} for n, r in runs.items()}).T

    # answer letters on the multiple-choice questions
    mca = v.index[v["question_type"].isin(MCA_QUESTION_TYPES)]
    dr = v.index[task.isin(DIR_ROUTE)]
    truth = v["ground_truth"]

    def letter_row(n, qs):
        lt, t = letters(runs[n], qs), truth.loc[qs]
        nb_ = t != "B"
        return {"answers B %": 100 * (lt == "B").mean(), "right, truth B %": 100 * (lt[~nb_] == "B").mean(),
                "right, truth not B %": 100 * (lt[nb_] == t[nb_]).mean(), "right, all %": 100 * (lt == t).mean()}

    bias_mca = pd.DataFrame({n: letter_row(n, mca) for n in CONDITIONS}).T
    bias_dr = pd.DataFrame({n: letter_row(n, dr) for n in CONDITIONS}).T
    const = {}
    for L in "ABCD":
        df = v.loc[mca, ["question_type"]].assign(score=(truth.loc[mca] == L).astype(float))
        s = summarize(df)
        const[f"always {L}"] = {SHORT[k]: s[k] for k in TASKS if k in s} | {"direction and route": group_score(df, DIR_ROUTE)}
    const = pd.DataFrame(const).T

    # how many direction and route answers each comparison changes, and which way
    changes = {}
    for x, y in COMPARISONS + [("obj_csv", "video"), ("narrative", "video")]:
        lx, ly, t = letters(runs[x], dr), letters(runs[y], dr), truth.loc[dr]
        ch = lx != ly
        changes[f"{x} − {y}"] = {"answers changed": int(ch.sum()), "to right": int((ch & (lx == t)).sum()),
                                 "from right": int((ch & (ly == t)).sum()), "to B": int((ch & (lx == "B")).sum())}
    changes = pd.DataFrame(changes).T

    # the numeric answers: medians against the truth
    numeric = {}
    for qt in ["object_counting", "object_abs_distance", "object_size_estimation", "room_size_estimation"]:
        qs = v.index[v["question_type"] == qt]
        numeric[f"{SHORT[qt]} (truth {truth.loc[qs].astype(float).median():g})"] = {
            n: runs[n].loc[qs, "prediction"].astype(str).map(final_number).astype(float).median() for n in CONDITIONS}
    numeric = pd.DataFrame(numeric)

    r1 = lambda df: df.astype(float).round(1)
    out = ["# Camera pilot: analysis", "",
           "<!-- Written by analysis.py in this folder; edit that, not this file. -->", "",
           f"The predictions of `explore_camera.ipynb` (`results/zeroshot/vsibench/camera/explore200/`): 200 questions "
           f"(`explore200_ids.txt`, 25 per task), Qwen3.5-2B, 16 frames (`uniform128`), direct answers, up to 256 new "
           f"tokens, final answer scored, batches of 8, pipeline sheets in VGGT's relative units, on one "
           f"{info['gpu']} at commit `{info['code_commit'][:7]}`. `video` was copied from "
           f"`{Path(reused['video']).parent.name}/{Path(reused['video']).name}` (the structured-CoT pilot's video-only run), "
           f"which passed the reuse check: the same questions, frames, prompts and batches. Bootstrap intervals: "
           f"questions resampled within each question type, the same draws for every condition ({B:,} resamples, seed "
           f"{SEED}); they and the scores are recomputed here and match the notebook's tables.", "",
           f"Every reply in every condition was read by the answer rule (parse rate 100%), and none was longer than "
           f"{int(replies['longest reply (tokens)'].max())} tokens: the model answered with a letter or a number every "
           f"time, so the 256-token limit never mattered. The 14 runs and the timing took about 70 minutes on the L4, at "
           f"{min(1000 * t['batch_s'] / t['padded_tokens'] for t in timing.values()):.2f}–"
           f"{max(1000 * t['batch_s'] / t['padded_tokens'] for t in timing.values()):.2f} ms per input token.", "",
           "## Scores", "",
           "×100. **Direction and route**: the mean of relative direction (its three types as one task) and route "
           "planning, 50 questions. **Without metric tasks**: the 5 tasks other than absolute distance, object size and "
           "room size. Sheet tokens: what the scene information adds to the video-only prompt, on average.", "",
           md_table(r1(S).rename(columns=str.capitalize), "Condition"), "",
           "## The comparisons", "",
           "The README's comparisons, then every condition against `video`. `*`: the interval excludes 0. Lead: the "
           "README's reading rule (at least 5 points on direction and route or 2 on the overall, interval excluding 0).",
           "",
           md_table(pd.DataFrame({c: {**{g: fmt_interval(*rows[g]) for g in GROUPS}, "lead": lead(rows)}
                                  for c, rows in D.items()}).T, "Comparison"), "",
           md_table(pd.DataFrame({c: {**{g: fmt_interval(*rows[g]) for g in GROUPS}, "lead": lead(rows)}
                                  for c, rows in VS.items()}).T, "Against video"), "",
           "Per question type (25 questions each; relative direction 8, 8 and 9):", "",
           md_table(pd.DataFrame({c: {t: fmt_interval(*rows[t]) for t in qtypes} for c, rows in D.items()}).T,
                    "Comparison"), "",
           "## The answer letters", "",
           "On the 100 multiple-choice questions, B is the true answer of "
           f"{int((truth.loc[mca] == 'B').sum())}, and on the 50 direction and route questions of "
           f"{int((truth.loc[dr] == 'B').sum())}. The sheets shift the model's answers to B: the objects-only tables "
           "least, the camera tables, the sentences and the narrative most. A condition can gain on a task from that "
           "alone; the constant answers' scores show how much:", "",
           md_table(r1(const), "Answer"), "",
           "Each condition's answers on the 100 multiple-choice questions:", "",
           md_table(r1(bias_mca), "Condition"), "",
           "And on the 50 direction and route questions (31 whose answer is not B):", "",
           md_table(r1(bias_dr), "Condition"), "",
           "How many of the 50 direction and route answers each comparison changes: to the right answer, away from it, "
           "and to B:", "",
           md_table(changes, "Comparison"), "",
           "## The numeric answers", "",
           "Median answer per numeric question type (25 questions each), with the true median in the column's name. The "
           "sheets' relative units are read as metres: one sheet unit is about 2.3 m (`metric_scale.md` in the format "
           "experiment's folder).", "",
           md_table(numeric.round(2), "Condition"), ""]
    (HERE / "analysis.md").write_text("\n".join(out))
    print(f"wrote {HERE / 'analysis.md'}")


if __name__ == "__main__":
    report()
