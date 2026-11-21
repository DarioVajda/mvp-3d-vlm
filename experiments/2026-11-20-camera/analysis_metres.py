"""Run 2 of the camera pilot (explore_camera_metres.ipynb: every sheet in metres, plus three chain-of-thought
conditions) analysed as the README's Metrics and analysis section specifies; writes analysis_metres.md next to this
file.

  PYTHONPATH=src python experiments/2026-11-20-camera/analysis_metres.py

Reads results/zeroshot/vsibench/camera/explore200_metres/ (run 2) and, for metres − relative, run 1's predictions in
results/zeroshot/vsibench/camera/explore200/. The scores, the CoT reply counts and the bootstrap intervals are
recomputed from the predictions as the notebook computes them (the same resampling and seed) and checked against the
notebook's scores.csv, scores_cot.csv, answers.csv and differences_bootstrap.csv. The reasoning check is the
notebook's own (reasoning_check.csv). The checks on copied numbers and counts render the sheets again, so they need
data/scene_sheets_75 and data/vggt.
"""
import json
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import DIR_ROUTE, SHORT, TYPE_ORDER, fmt_interval, task_weights
from analysis import RESULTS as RUN1
from mvp.benchmarks import scene_sheet as ss
from mvp.benchmarks.vsibench import (MCA_QUESTION_TYPES, METRIC_TASKS, TASKS, final_letter, final_number, parsed,
                                     summarize, task_of)
from mvp.paths import ROOT

HERE = Path(__file__).parent
RESULTS = ROOT / "results/zeroshot/vsibench/camera/explore200_metres"
IDS = [int(x) for x in (ROOT / "experiments/2026-11-18-scene-sheet-formats/explore200_ids.txt").read_text().split()]
DIRECT = ["video", "obj_csv", "cam_csv", "obj_text", "cam_text", "narrative", "obj_markdown_table",
          "cam_markdown_table", "cam_csv_frames", "cam_text_turn_move", "cam_text_frames", "narrative_turn_move",
          "narrative_camera_dirs", "narrative_frames", "narrative_exact"]
MAIN = ["obj_csv", "cam_csv", "obj_markdown_table", "cam_markdown_table", "obj_text", "cam_text", "narrative"]
COT_BASES = {"cot_cam_csv": "cam_csv", "cot_cam_markdown_table": "cam_markdown_table", "cot_cam_text": "cam_text",
             "cot_narrative": "narrative"}
COMPARISONS = [("cam_csv", "obj_csv"), ("cam_markdown_table", "obj_markdown_table"), ("cam_text", "obj_text"),
               ("narrative", "cam_text"), ("narrative", "obj_text"), ("narrative", "obj_csv"),
               ("cam_csv_frames", "cam_csv"), ("cam_text_turn_move", "cam_text"), ("cam_text_frames", "cam_text"),
               ("narrative_turn_move", "narrative"), ("narrative_camera_dirs", "narrative"),
               ("narrative_frames", "narrative"), ("narrative_exact", "narrative")]
GROUPS = {"overall": set(TASKS), "metric tasks": set(METRIC_TASKS), "direction and route": DIR_ROUTE,
          "without metric tasks": set(TASKS) - set(METRIC_TASKS)}
LEAD = {"overall": 2, "direction and route": 5}
B, SEED = 10_000, 0
NUMERIC = ["object_counting", "object_abs_distance", "object_size_estimation", "room_size_estimation"]
# the structured-CoT pilot's reply checks, as the notebook applies them
ANSWER_LINE = re.compile(r"^[\s*#>_-]*answer\s*\**\s*:", re.I)
FRAME_TIME = re.compile(r"(?<![\d:])\d{1,2}:\d{2}(?![\d:])")
DECIMALS = re.compile(r"-?\d+\.\d{2,}")
# CoT replies quoted in full (head and tail): what a right, a wrong and a cut-off reply look like
EXCERPTS = [("cot_cam_markdown_table", 974, "right: it takes the three objects' coordinates from the sheet and "
             "compares them"),
            ("cot_cam_markdown_table", 4474, "wrong: it counts the pipeline's bed rows, split pieces of 2 beds, as 8 "
             "beds"),
            ("cot_cam_markdown_table", 2659, "cut off: a run of mm:ss times to the limit; the rule reads the last "
             "number")]


def load(folder, name):
    run = pd.read_json(folder / f"{name}.jsonl", lines=True).drop_duplicates("id").set_index("id")
    run["prediction"] = run["prediction"].astype(str)
    return run.loc[IDS]


def loops(reply):
    tail = reply[-2000:].encode()
    return len(FRAME_TIME.findall(reply)) >= 10 or len(zlib.compress(tail)) / max(1, len(tail)) < 0.25


def bootstrap(S, types):
    """The notebook's resampling: questions drawn with replacement within each question type (in TYPE_ORDER), the same
    draws for every column of S. Returns (types, point: type x column, draws: B x type x column, columns)."""
    qt = [t for t in TYPE_ORDER if t in set(types)]
    X = S.to_numpy(float)
    members = [np.flatnonzero(types.to_numpy() == t) for t in qt]
    rng = np.random.default_rng(SEED)
    point = np.stack([X[m].mean(0) for m in members])
    draws = np.stack([X[m][rng.integers(0, len(m), size=(B, len(m)))].mean(1) for m in members], axis=1)
    return qt, point, draws, list(S.columns)


def difference(boot, x, y):
    """{row: (difference, low, high)} for x − y: the groups, then each question type."""
    qt, point, draws, cols = boot
    i, j = cols.index(x), cols.index(y)
    dp, db = 100 * (point[:, i] - point[:, j]), 100 * (draws[:, :, i] - draws[:, :, j])
    rows = {**{g: (dp @ task_weights(qt, ts), db @ task_weights(qt, ts)) for g, ts in GROUPS.items()},
            **{t: (dp[k], db[:, k]) for k, t in enumerate(qt)}}
    return {r: (p, *np.percentile(b, [2.5, 97.5])) for r, (p, b) in rows.items()}


def lead(rows):
    return ", ".join(f"{g} {'+' if rows[g][0] > 0 else '−'}" for g, t in LEAD.items()
                     if abs(rows[g][0]) >= t and (rows[g][1] > 0 or rows[g][2] < 0))


def interval_table(diffs, rows):
    return pd.DataFrame({c: {**{r: fmt_interval(*d[r]) for r in rows}, "lead": lead(d)} for c, d in diffs.items()}).T


def md_table(df, index_name=""):
    names = set(DIRECT) | set(COT_BASES)
    cols = [index_name] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" + "---:|" * len(df.columns)]
    lines += ["| " + " | ".join([f"`{i}`" if i in names or " − " in str(i) else str(i)] + [str(v) for v in row]) + " |"
              for i, row in zip(df.index, df.astype(object).to_numpy())]
    return "\n".join(lines)


def group_score(types, scores, tasks):
    keep = types.map(task_of).isin(tasks)
    return summarize(pd.DataFrame({"question_type": types[keep], "score": scores[keep]}))["overall"]


def score_table(runs, names, sheet_tokens):
    out = {}
    for n in names:
        r = runs[n]
        s = summarize(r[["question_type", "score"]])
        out[n] = {**{g: group_score(r["question_type"], r["score"], ts) for g, ts in GROUPS.items()},
                  **{SHORT[k]: s[k] for k in TASKS}, "sheet tokens": sheet_tokens[n]}
    return pd.DataFrame(out).T.astype(float)


def sheet_of(question, options, metric_scale):
    o = dict(options)
    return ss.render_scene(question, "pipeline", o.pop("fmt"), metric_scale=metric_scale, **o) or ""


def copied(answer, values):
    """Whether a bare numeric answer with 2 or more decimals equals one of the sheet's values at its precision."""
    a = answer.strip()
    if not re.fullmatch(r"-?\d+\.\d{2,}", a):
        return False
    d = min(len(a.split(".")[1]), ss.DECIMALS)
    return any(round(v, d) == round(float(a), d) for v in values)


def letter_row(run, truth, qs):
    lt, t = run.loc[qs, "prediction"].map(final_letter), truth.loc[qs]
    not_b = t != "B"
    return {"answers B %": 100 * (lt == "B").mean(), "no letter %": 100 * lt.isna().mean(),
            "right, truth B %": 100 * (lt[~not_b] == "B").mean(),
            "right, truth not B %": 100 * (lt[not_b] == t[not_b]).mean(), "right, all %": 100 * (lt == t).mean()}


def excerpt(text, head=1400, tail=300):
    text = text.strip()
    return text if len(text) <= head + tail + 50 else f"{text[:head]}\n[… {len(text) - head - tail:,} characters …]\n{text[-tail:]}"


def report():
    info = json.loads((RESULTS / "run_info.json").read_text())
    options = info["conditions"]
    cot_table = json.loads((RESULTS / "cot_table.json").read_text())
    COT = [{"csv": "cot_cam_csv", "markdown_table": "cot_cam_markdown_table"}[cot_table["table"]],
           "cot_cam_text", "cot_narrative"]
    assert info["cot_questions"] == info["direct_questions"] == len(IDS) == 200, "the CoT runs used a subset"
    runs = {n: load(RESULTS, n) for n in DIRECT + COT}
    rel = {n: load(RUN1, n) for n in DIRECT[1:]}
    v = runs["video"]
    truth, types, task = v["ground_truth"], v["question_type"], v["question_type"].map(task_of)
    assert (load(RUN1, "video")["prediction"] == v["prediction"]).all(), "run 1 and run 2 used different video runs"
    lengths = pd.read_csv(RESULTS / "input_tokens.csv", index_col=0).loc[IDS]
    sheet_tokens = lengths.sub(lengths["video"], axis=0).mean()

    # the scores and intervals, checked against the notebook's
    S = score_table(runs, DIRECT + COT, sheet_tokens)
    nb = pd.concat([pd.read_csv(RESULTS / "scores.csv", index_col=0),
                    pd.read_csv(RESULTS / "scores_cot.csv", index_col=0).loc[COT]])
    for g in GROUPS:
        assert np.allclose(S[g], nb.loc[S.index, g]), f"scores differ from the notebook's: {g}"
    cols = pd.DataFrame({**{n: runs[n]["score"] for n in DIRECT + COT},
                         **{f"{n} (relative)": rel[n]["score"] for n in rel}})
    boot = bootstrap(cols, types)
    D = {f"{x} − {y}": difference(boot, x, y) for x, y in COMPARISONS}
    VS = {f"{n} − video": difference(boot, n, "video") for n in DIRECT[1:]}
    MR = {f"{n}: metres − relative": difference(boot, n, f"{n} (relative)") for n in rel}
    table_cot, text_cot, walk_cot = COT
    cot_pairs = [(c, COT_BASES[c]) for c in COT] + [(text_cot, table_cot), (walk_cot, text_cot), (walk_cot, table_cot)]
    CD = {f"{x} − {y}": difference(boot, x, y) for x, y in cot_pairs}
    CV = {f"{n} − video": difference(boot, n, "video") for n in COT}
    nbd = pd.read_csv(RESULTS / "differences_bootstrap.csv").set_index(["comparison", "rows"])
    for c, rows in {**D, **VS, **MR, **CD, **CV}.items():
        for r, (p, lo, hi) in rows.items():
            q = nbd.loc[(c, r)]
            assert np.allclose([p, lo, hi], [q["difference"], q["95% low"], q["95% high"]]), f"{c}, {r}"

    # the replies
    docs = {n: r.reset_index().to_dict("records") for n, r in runs.items()}
    reply = {}
    for n, r in runs.items():
        cut = r["hit_token_limit"].astype(bool)
        line = r["prediction"].map(lambda p: any(ANSWER_LINE.match(x) for x in p.splitlines()))
        ok = pd.Series([parsed(d, d["prediction"], final=True) for d in docs[n]], index=r.index)
        reply[n] = {"parsed %": 100 * ok.mean(), "'Answer:' line %": 100 * line.mean(),
                    "median tokens": r["n_new_tokens"].median(), "90th pct tokens": r["n_new_tokens"].quantile(0.9),
                    "hit the limit": int(cut.sum()), "of them before answering": int((cut & ~line).sum()),
                    "of those, read anyway": int((cut & ~line & ok).sum()),
                    "cut off and looping": int((cut & r["prediction"].map(loops)).sum()),
                    "score": 100 * r["score"].mean(), "score, cut-off replies wrong": 100 * r["score"].where(~(cut & ~line), 0).mean()}
    reply = pd.DataFrame(reply).T
    nba = pd.read_csv(RESULTS / "answers.csv", index_col=0)
    for c in ["parsed %", "'Answer:' line %", "hit the limit", "of them before answering", "of those, read anyway",
              "cut off and looping", "score, cut-off replies wrong"]:
        assert np.allclose(reply[c].astype(float), nba.loc[reply.index, c]), f"reply counts differ from the notebook's: {c}"
    long_direct = {n: r.index[r["hit_token_limit"].astype(bool)].tolist() for n, r in runs.items() if n in DIRECT}
    long_direct = {n: q for n, q in long_direct.items() if q}
    per_type = pd.concat({n: pd.DataFrame({
        "median tokens": runs[n].groupby("question_type")["n_new_tokens"].median(),
        "cut off": runs[n].groupby("question_type")["hit_token_limit"].sum().astype(int),
        "parsed %": pd.Series([parsed(d, d["prediction"], final=True) for d in docs[n]], index=runs[n].index)
        .groupby(types).mean() * 100}) for n in COT}, axis=1).reindex(TYPE_ORDER)
    per_type.index = [t.replace("object_", "").replace("obj_", "") for t in per_type.index]
    per_type.columns = [f"{n.removeprefix('cot_').removeprefix('cam_')}: {c}" for n, c in per_type.columns]

    # the answer letters
    mca = v.index[types.isin(MCA_QUESTION_TYPES)]
    dr = v.index[task.isin(DIR_ROUTE)]
    const = {}
    for L in "ABCD":
        sc = (truth.loc[mca] == L).astype(float)
        s = summarize(pd.DataFrame({"question_type": types.loc[mca], "score": sc}))
        const[f"always {L}"] = {SHORT[k]: s[k] for k in TASKS if k in s} | {
            "direction and route": group_score(types.loc[mca], sc, DIR_ROUTE)}
    const = pd.DataFrame(const).T
    bias_mca = pd.DataFrame({n: letter_row(runs[n], truth, mca) for n in DIRECT + COT}).T
    bias_dr = pd.DataFrame({n: letter_row(runs[n], truth, dr) for n in DIRECT + COT}).T
    bias_rel = pd.DataFrame({f"{n} (relative)": letter_row(rel[n], truth, dr) for n in MAIN}).T
    changes = {}
    for x, y in COMPARISONS[:4] + [("narrative_turn_move", "narrative")] + [(c, COT_BASES[c]) for c in COT]:
        lx, ly = (runs[n].loc[dr, "prediction"].map(final_letter) for n in (x, y))
        t, ch = truth.loc[dr], lx != ly
        changes[f"{x} − {y}"] = {"answers changed": int(ch.sum()), "to right": int((ch & (lx == t)).sum()),
                                 "from right": int((ch & (ly == t)).sum()), "to B": int((ch & (lx == "B")).sum()),
                                 "from B": int((ch & (ly == "B")).sum())}
    changes = pd.DataFrame(changes).T

    # metres against relative units: what changed
    changed = pd.DataFrame({n: {
        "answers changed, metric tasks (75)": int((runs[n].loc[task.isin(METRIC_TASKS)].prediction
                                                   != rel[n].loc[task.isin(METRIC_TASKS)].prediction).sum()),
        "answers changed, direction and route (50)": int((runs[n].loc[dr].prediction != rel[n].loc[dr].prediction).sum()),
        "answers changed, other (75)": int((runs[n].loc[~task.isin(METRIC_TASKS | DIR_ROUTE)].prediction
                                            != rel[n].loc[~task.isin(METRIC_TASKS | DIR_ROUTE)].prediction).sum())}
        for n in rel}).T

    # the numeric answers: medians, copied sheet numbers, counts
    def median(run, qt):
        qs = v.index[types == qt]
        return run.loc[qs, "prediction"].map(final_number).astype(float).median()

    numeric = pd.DataFrame({f"{SHORT[qt]} (truth {truth.loc[v.index[types == qt]].astype(float).median():g})": {
        n: (f"{median(runs[n], qt):g}" + (f" ({median(rel[n], qt):g})" if n in rel else "")) for n in DIRECT + COT}
        for qt in NUMERIC})
    copy_rows, room_rows = {}, {}
    for qt in ["object_abs_distance", "room_size_estimation"]:
        qs = v.index[types == qt]
        for n in MAIN:
            for tag, run, metric in (("metres", runs[n], "priors"), ("relative", rel[n], None)):
                hits = [copied(run.loc[q, "prediction"], {float(x) for x in DECIMALS.findall(sheet_of(q, options[n], metric))})
                        for q in qs]
                copy_rows.setdefault(n, {})[f"{SHORT[qt]}, {tag}"] = int(sum(hits))
    rooms = v.index[types == "room_size_estimation"]
    room_values, room_counts, room_extent = {}, [], []
    for q in rooms:
        scale = ss.scale_of(q, ss.SCENE_SHEETS, "priors")
        objs = [o for o in ss.load_objects(q, "pipeline", scale=scale) if o["label"] == "room"]
        room_values[q] = {float(o[k]) for o in objs for k in ss.POSITION + ss.EXTENT}
        room_counts.append(len(objs))
        room_extent += [max(float(o[k]) for k in ss.EXTENT) for o in objs]
    n_room_rows = sum(c > 0 for c in room_counts)
    extent_lo, extent_hi = np.percentile(room_extent, [10, 90])
    away = 0  # the narrative's copied absolute-distance answers that are a camera-to-object distance ("… metres away")
    for q in v.index[types == "object_abs_distance"]:
        sheet = sheet_of(q, options["narrative"], "priors")
        away += copied(runs["narrative"].loc[q, "prediction"],
                       {float(x) for x in re.findall(r"(\d+\.\d+) metres away", sheet)})
    for n in MAIN:
        room_rows[n] = int(sum(copied(runs[n].loc[q, "prediction"], room_values[q]) for q in rooms))
    copies = pd.DataFrame(copy_rows).T
    copies["room size: a room row's number, metres"] = pd.Series(room_rows)
    counting = v.index[types == "object_counting"]
    sheet_count = {}
    for q in counting:
        cls = re.search(r"How many (.+?)\(s\)", v.loc[q, "question"]).group(1).strip()
        sheet_count[q] = sum(o["label"] == cls for o in ss.load_objects(q, "pipeline"))
    sheet_count = pd.Series(sheet_count)
    t_count = truth.loc[counting].astype(float)
    count_rows = {}
    for n in DIRECT + COT:
        a = runs[n].loc[counting, "prediction"].map(final_number).astype(float)
        count_rows[n] = {"answer = the sheet's count": int((a == sheet_count).sum()), "answer = truth": int((a == t_count).sum()),
                         "right where the sheet is right": int(((a == t_count) & (sheet_count == t_count)).sum()),
                         "right where the sheet is wrong": int(((a == t_count) & (sheet_count != t_count)).sum()),
                         "score": 100 * runs[n].loc[counting, "score"].mean()}
    count_rows = pd.DataFrame(count_rows).T
    count_rows = count_rows.astype({c: int for c in count_rows.columns if c != "score"}).round(1)

    reasoning = pd.read_csv(RESULTS / "reasoning_check.csv")
    reasoning.index = reasoning["condition"].map(lambda n: f"`{n}`") + ", " + reasoning["replies"]
    reasoning = reasoning.drop(columns=["condition", "replies"]).astype({"n": int})

    # the text
    r1 = lambda df: df.astype(float).round(1)
    COUNTS = ["90th pct tokens", "hit the limit", "of them before answering", "of those, read anyway", "cut off and looping"]
    short_reply = reply.loc[DIRECT]
    cut = {n: int(reply.loc[n, "hit the limit"]) for n in COT}
    looping = {n: int(reply.loc[n, "cut off and looping"]) for n in COT}
    no_line = {n: int(reply.loc[n, "of them before answering"]) for n in COT}
    out = [
        "# Camera pilot, run 2 (metres, with chain of thought): analysis", "",
        "<!-- Written by analysis_metres.py in this folder; edit that, not this file. -->", "",
        f"The predictions of `explore_camera_metres.ipynb` (`results/zeroshot/vsibench/camera/explore200_metres/`), on "
        f"one {info['gpu']} at commit `{info['code_commit'][:7]}` (the code zip, uncommitted `src/` changes included):",
        "- 200 questions (`explore200_ids.txt`, 25 per task), Qwen3.5-2B, 16 frames (`uniform128`), final answer "
        "scored, batches of 8;",
        "- every pipeline sheet in metres (`--metric-scale priors`): the objects, the camera tables, the compass path, "
        "the narrative;",
        f"- 15 direct conditions (up to 256 new tokens). `video` was copied from "
        f"`{Path(info['reused']['video']).parent.name}/{Path(info['reused']['video']).name}` after the reuse check, "
        "as in run 1;",
        f"- 3 CoT conditions on all 200 questions (`--reasoning cot`, up to 2,048 new tokens). The table format was "
        f"chosen by the rule fixed before the run: the direct `cam_markdown_table` scored "
        f"{cot_table['direct overall']['markdown_table']:.1f} overall against `cam_csv`'s "
        f"{cot_table['direct overall']['csv']:.1f}, so the CoT table is `{table_cot}`. `cot_cam_csv.jsonl` holds "
        "only its 5 timing questions.", "",
        f"Bootstrap intervals: questions resampled within each question type, the same draws for every condition "
        f"({B:,} resamples, seed {SEED}). The scores, the reply counts and every interval are recomputed here from the "
        "predictions and match the notebook's tables. Run 1's predictions (relative units) come from "
        "`results/zeroshot/vsibench/camera/explore200/`; both runs share `video`.", "",
        "## Replies", "",
        f"**Direct:** every reply was parsed. Median 1 token, 90th percentile {short_reply['90th pct tokens'].min():g}–"
        f"{short_reply['90th pct tokens'].max():g}. {sum(len(q) for q in long_direct.values())} replies ran to the 256-token "
        f"limit: {', '.join(f'`{n}` on {q[0]}' for n, q in long_direct.items())}, all room-size questions, each a "
        "product of the sheet's sizes that never ends (\"1.874*2.190*4.504*…\").", "",
        f"**CoT:** {cut[table_cot]}, {cut[text_cot]} and {cut[walk_cot]} of 200 replies were cut off at 2,048 tokens "
        f"(the table, the compass path, the narrative); {no_line[table_cot]}, {no_line[text_cot]} and "
        f"{no_line[walk_cot]} of them before an \"Answer:\" line. {looping[table_cot]}, {looping[text_cot]} and "
        f"{looping[walk_cot]} loop by the structured-CoT pilot's rule (10 or more mm:ss times, or a tail that "
        "compresses below a quarter); the others run out while still working: distances over many pairs of rows, the "
        "route options tried one by one, or lists of frames and numbers that the rule does not count as a loop. The "
        "pilot had 12% of replies cut off with the gold sheet and 28% with video only.", "",
        md_table(r1(reply.loc[COT]).astype({c: int for c in COUNTS}), "Condition"), "",
        "Per question type (n = 25; relative direction 8, 8 and 9): median reply tokens, replies cut off, parse "
        "rate:", "",
        md_table(per_type.round(0).astype("Int64"), "Question type"), "",
        "## Scores", "",
        "×100, all 18 conditions on the same 200 questions. **Metric tasks**: the mean of absolute distance, object "
        "size and room size. **Direction and route**: the mean of relative direction (its three types as one task) and "
        "route planning, 50 questions. **Without metric tasks**: the other 5 tasks. **Sheet tokens**: what the prompt "
        "adds to the video-only one, on average (for CoT, the CoT instructions too).", "",
        md_table(r1(S).rename(columns=str.capitalize), "Condition"), "",
        "## The comparisons", "",
        "`*`: the interval excludes 0. **Lead**: the README's reading rule (at least 5 points on direction and route "
        "or 2 on the overall, interval excluding 0).", "",
        "### 1–3. Cameras, the narrative, the options (direct)", "",
        md_table(interval_table(D, GROUPS), "Comparison"), "",
        "Per question type:", "",
        md_table(pd.DataFrame({c: {t: fmt_interval(*d[t]) for t in boot[0]} for c, d in D.items()}).T, "Comparison"), "",
        "### 4. Against video only", "",
        md_table(interval_table(VS, GROUPS), "Comparison"), "",
        "### 5. Metres against relative units", "",
        "Each run-2 condition minus the same condition in run 1, on the same 200 questions. The metric tasks are the "
        "row that matters.", "",
        md_table(interval_table(MR, list(GROUPS) + [t for t in TYPE_ORDER if task_of(t) in METRIC_TASKS]), "Comparison"), "",
        "How many answers the metres change, out of the 75 metric-task questions, the 50 direction and route ones "
        "and the 75 others:", "",
        md_table(changed, "Condition"), "",
        "### 6. Chain of thought", "",
        md_table(interval_table({**CD, **CV}, GROUPS), "Comparison"), "",
        "Per question type:", "",
        md_table(pd.DataFrame({c: {t: fmt_interval(*d[t]) for t in boot[0]} for c, d in CD.items()}).T, "Comparison"), "",
        "## The answer letters", "",
        f"B is the true answer of {int((truth.loc[mca] == 'B').sum())} of the 100 multiple-choice questions, and of "
        f"{int((truth.loc[dr] == 'B').sum())} of the 50 direction and route ones. The score of always giving one "
        "letter:", "",
        md_table(r1(const), "Answer"), "",
        "Each condition on the 100 multiple-choice questions (**no letter**: the rule reads none, as in cut-off CoT "
        "replies):", "",
        md_table(r1(bias_mca), "Condition"), "",
        f"On the 50 direction and route questions ({int((truth.loc[dr] != 'B').sum())} whose answer is not B):", "",
        md_table(r1(bias_dr), "Condition"), "",
        "Run 1's main conditions on the same 50, for comparison:", "",
        md_table(r1(bias_rel), "Condition"), "",
        "How many of the 50 direction and route answers each comparison changes, and which way:", "",
        md_table(changes, "Comparison"), "",
        "## The numeric answers", "",
        "Median answer per numeric question type (25 questions each), with the true median in the column's name; run "
        "1's in brackets. Counts are whole objects, distances metres, object sizes centimetres, room sizes square "
        "metres.", "",
        md_table(numeric, "Condition"), "",
        "**Sheet numbers copied as the answer.** A bare answer with 2 or more decimals that equals a number in the "
        "condition's sheet, out of 25 questions. Object sizes are asked in centimetres, so a copied size would not "
        f"show; none has 2 decimals. All {n_room_rows} room-size sheets have `room` rows, the pipeline's segments for "
        f"\"room\": {min(room_counts)}–{max(room_counts)} per sheet, each a piece of the room "
        f"{extent_lo:.1f}–{extent_hi:.1f} m across at its longest (10th–90th percentile). The last column counts the "
        "room-size answers that are one of those rows' numbers.", "",
        md_table(copies, "Condition"), "",
        f"The narrative gives each object's distance from the camera (\"… 1.775 metres away\"), not between objects: "
        f"{away} of its {copies.loc['narrative', 'Abs. dist., metres']} copied absolute-distance answers are one of "
        "those.", "",
        f"**Counting and the sheet's count.** The pipeline sheet's count of the asked class equals the true count "
        f"on {int((sheet_count == t_count).sum())} of the 25 counting questions; it is higher on "
        f"{int((sheet_count > t_count).sum())} and lower on {int((sheet_count < t_count).sum())} (split, merged or missed "
        "objects). Each condition's answers against it (MRA score ×100 in the last column):", "",
        md_table(count_rows, "Condition"), "",
        "## Chain of thought: what the replies do", "",
        "The notebook's reasoning check (`reasoning_check.csv`), over the replies whose sheet has objects (192 of "
        "200): the share that name a sheet row (an id such as `chair_2`), copy an object number as the sheet writes "
        "it, or copy a number the sheet gives only for the camera (camera positions in the table; the move lengths in "
        "the compass path and the narrative). **Right**: multiple choice right, or a numeric score of at least 0.5.", "",
        md_table(reasoning.round(1), "Condition, replies"), "",
        "Three replies, the head and the tail:", ""]
    for n, q, why in EXCERPTS:
        r = runs[n].loc[q]
        out += [f"**`{n}`, question {q}** ({r['question_type']}, truth {r['ground_truth']}, "
                f"{int(r['n_new_tokens'])} tokens): {why}.", "",
                "```text", excerpt(r["prediction"]), "```", ""]
    (HERE / "analysis_metres.md").write_text("\n".join(out))
    print(f"wrote {HERE / 'analysis_metres.md'}")


if __name__ == "__main__":
    report()
