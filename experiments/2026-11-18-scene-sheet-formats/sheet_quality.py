"""Pipeline vs gold scene sheets: are the objects a question names in each sheet, how their counts and relative sizes
compare, which pipeline sheets are empty. Writes sheet_quality.md (next to this file) in full; reads the sheets only.

  PYTHONPATH=src python experiments/2026-11-18-scene-sheet-formats/sheet_quality.py \
      [--root data/scene_sheets_75] [--vggt-root data/vggt] [--out experiments/2026-11-18-scene-sheet-formats/sheet_quality.md]

--vggt-root is only needed for the example sheets' first_seen (each video's meta.json).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mvp.benchmarks import scene_sheet as ss
from mvp.paths import SCENE_SHEETS, VGGT_DIR

HERE = Path(__file__).parent
TYPES = {  # report order and names
    "obj_appearance_order": "Appearance order",
    "object_abs_distance": "Absolute distance",
    "object_counting": "Counting",
    "object_rel_direction_easy": "Relative direction, easy",
    "object_rel_direction_medium": "Relative direction, medium",
    "object_rel_direction_hard": "Relative direction, hard",
    "object_rel_distance": "Relative distance",
    "object_size_estimation": "Object size",
    "room_size_estimation": "Room size",
    "route_planning": "Route planning",
}
DATASETS = {"scannet": "ScanNet", "arkitscenes": "ARKitScenes", "scannetpp": "ScanNet++"}
EXAMPLE_TYPES = ["object_size_estimation", "obj_appearance_order", "object_counting"]


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def num(n):
    return f"{n:,}"


def longest(rows):
    return [max(float(r["size_x"]), float(r["size_y"]), float(r["size_z"])) for r in rows]


def load(root):
    """Q: one row per question; C: one row per (question, named object). Named objects are the question's prompts
    without `room` (none for room-size questions)."""
    Q, C = [], []
    with open(root / "index.csv", newline="") as f:
        index = list(csv.DictReader(f))
    for q in index:
        prompts = json.loads((root / q["pipeline_sheet"] / "question.json").read_text())["prompts"]
        matched = json.loads((root / q["gold_sheet"] / "sheet.json").read_text())["matched_classes"]
        with open(root / q["pipeline_sheet"] / "objects.csv", newline="") as f:
            p_rows = list(csv.DictReader(f))
        with open(root / q["gold_sheet"] / "objects.csv", newline="") as f:
            g_rows = list(csv.DictReader(f))
        named = [] if q["question_type"] == "room_size_estimation" else [p for p in prompts if p != "room"]
        Q.append(dict(id=int(q["id"]), scene=q["scene"], type=q["question_type"], question=q["question"],
                      gt=q["ground_truth"], prompts=prompts, n_named=len(named), n_pipe=len(p_rows),
                      n_gold=len(g_rows), n_pipe_room=sum(r["concept"] == "room" for r in p_rows),
                      unmatched=[k for k, v in matched.items() if v is None and k != "room"]))
        for c in named:
            pr = [r for r in p_rows if r["concept"] == c]
            gr = [r for r in g_rows if r["concept"] == c]
            C.append(dict(id=int(q["id"]), scene=q["scene"], type=q["question_type"], cls=c,
                          gold_class=matched.get(c), n_pipe=len(pr), n_gold=len(gr), pipe_longest=longest(pr),
                          gold_longest=longest(gr), pipe_frames=[int(r["num_frames"]) for r in pr]))
    Q, C = pd.DataFrame(Q), pd.DataFrame(C)
    C["fp"], C["fg"] = C.n_pipe > 0, C.n_gold > 0
    C["d"] = C.n_pipe - C.n_gold
    return Q, C


def coverage(Q, C):
    """Section 1."""
    per_q = C.groupby("id").agg(type=("type", "first"), allp=("fp", "all"), allg=("fg", "all"))
    out = {"pairs": len(C), "questions": len(per_q), "pipe": C.fp.mean(), "gold": C.fg.mean(),
           "allp": per_q.allp.mean(), "allg": per_q.allg.mean(),
           "gold_only": int((C.fg & ~C.fp).sum()), "pipe_only": int((~C.fg & C.fp).sum()),
           "neither": int((~C.fg & ~C.fp).sum())}
    rows = []
    for t, name in TYPES.items():
        c, a = C[C.type == t], per_q[per_q.type == t]
        if len(c):
            rows.append(f"| {name} | {num(len(c))} | {pct(c.fp.mean())} | {pct(c.fg.mean())} | "
                        f"{pct(a.allp.mean())} | {pct(a.allg.mean())} |")
    rows.append(f"| **All** | **{num(len(C))}** | **{pct(out['pipe'])}** | **{pct(out['gold'])}** | "
                f"**{pct(out['allp'])}** | **{pct(out['allg'])}** |")
    out["table"] = "\n".join(rows)
    rd = Q[Q.type == "object_rel_distance"]
    out["rd_named"] = int(rd.n_named.median())
    out["rd_all"] = per_q[per_q.type == "object_rel_distance"].allp.mean()
    ds = C.assign(ds=C.scene.str.split("/").str[0]).groupby("ds").fp.mean().sort_values(ascending=False)
    out["datasets"] = ", ".join(f"{pct(v)} on {DATASETS[k]}" for k, v in ds.items())
    sc = C.groupby("scene").fp.mean().sort_values()
    out["scene_median"], out["scene_q25"] = sc.median(), sc.quantile(0.25)
    out["worst_scenes"] = ", ".join(f"`{s}` ({pct(v, 0)})" for s, v in sc.head(5).items())
    u = C[C.fg].drop_duplicates(["scene", "cls"])
    by_cls = u.groupby("cls").agg(n=("fp", "size"), found=("fp", "sum"), share=("fp", "mean"))
    missed = by_cls[by_cls.n >= 5].sort_values(["share", "n"], ascending=[True, False]).head(9)
    out["missed"] = ", ".join(f"{c} {int(r.found)} of {int(r.n)}" for c, r in missed.iterrows())
    found = by_cls[(by_cls.n >= 8) & (by_cls.share >= 0.9)].sort_values("share", ascending=False)
    out["found"] = ", ".join(found.index)
    ng = C[~C.fg]
    out["gold_miss"], out["gold_miss_unmatched"] = len(ng), int(ng.gold_class.isna().sum())
    out["gold_miss_route"] = int((ng.type == "route_planning").sum())
    route = ng[ng.type == "route_planning"].cls.value_counts()
    out["route_examples"] = ", ".join(f'"{c}"' for c in route.index[:4])
    other = ng[ng.type != "route_planning"].cls.value_counts()
    out["other_unmatched"] = ", ".join(f"`{c}` ({n})" for c, n in other.items())
    return out


def empties(Q):
    """Section 2."""
    E = Q[Q.n_pipe == 0]
    out = {"n": len(E), "folders": None, "scenes": E.scene.nunique(), "share": len(E) / len(Q),
           "gold_nonempty": int((E.n_gold > 0).sum())}
    rows, none = [], []
    for t, name in TYPES.items():
        e, q = E[E.type == t], Q[Q.type == t]
        if not len(e):
            none.append(name.lower())
            continue
        rows.append((len(e), f"| {name} | {len(e)} | {pct(len(e) / len(q))} | {int((e.n_gold > 0).sum())} |"))
    rows = [r for _, r in sorted(rows, key=lambda r: -r[0])]
    rows.append(f"| **All** | **{len(E)}** | **{pct(len(E) / len(Q))}** | **{out['gold_nonempty']}** |")
    out["table"], out["no_empties"] = "\n".join(rows), none
    out["by_type"] = E.type.value_counts()
    eg = E[E.n_gold == 0]
    out["gold_empty"] = "; ".join(
        f"{r.id} ({TYPES[r.type].lower()}" + (f", `{'`, `'.join(r.unmatched)}` matched to no gold class)" if r.unmatched
                                              else ", whose gold sheet is always empty)") for r in eg.itertuples())
    k = E.prompts.str.len().value_counts().sort_index()
    out["n_prompts"] = ", ".join(f"{n} with {i}" for i, n in k.items())
    single = E[E.prompts.str.len() == 1].prompts.str[0].value_counts()
    out["top_prompts"] = ", ".join(f"{p} {n}" for p, n in single[single >= 3].items())
    multi = E[E.prompts.str.len() > 1].prompts.str.join("+").value_counts()
    out["multi_examples"] = ", ".join(f"`{p}`" for p in multi.index[:2])
    bad = E.scene.value_counts().head(3)
    per_scene = Q.scene.value_counts()
    out["bad_total"] = int(bad.sum())
    out["bad"] = ", ".join(f"`{s}` ({n}, {pct(n / per_scene[s], 0)} of its questions)" for s, n in bad.items())
    return out


def counts(Q, C):
    """Section 3."""
    K = C[C.fg]
    def row(name, d, bold=False):
        cells = [name, num(len(d)), pct((d.d == 0).mean()), pct((d.d > 0).mean()),
                 pct(((d.d < 0) & (d.n_pipe > 0)).mean()), pct((d.n_pipe == 0).mean()), f"{d.d.mean():+.1f}"]
        return "| " + " | ".join(f"**{c}**" if bold else c for c in cells) + " |"
    rows = [row(name, K[K.type == t]) for t, name in TYPES.items() if (K.type == t).any()]
    rows.append(row("All", K, bold=True))
    over, found = K[K.d > 0].d, K[K.n_pipe > 0]
    one = K[K.n_gold == 1].n_pipe
    cnt = Q[Q.type == "object_counting"]
    kc = K[K.type == "object_counting"]
    u = K.drop_duplicates(["scene", "cls"])
    by_cls = u.groupby("cls").agg(n=("d", "size"), mean_d=("d", "mean"), max_p=("n_pipe", "max")).query("n >= 8")
    top = by_cls.sort_values("mean_d", ascending=False).head(6)
    low = by_cls.sort_values("mean_d").head(5)
    frames = np.concatenate([x for x in C.drop_duplicates(["scene", "cls"]).pipe_frames if len(x)])
    return {"pairs": len(K), "table": "\n".join(rows), "match": (K.d == 0).mean(),
            "over_median": over.median(), "over_mean": over.mean(), "over_p90": over.quantile(0.9),
            "ratio_found": (found.n_pipe / found.n_gold).median(),
            "under_nonzero": ((K.d < 0) & (K.n_pipe > 0)).mean(),
            "one_n": len(one), "one_share": len(one) / len(K), "one0": (one == 0).sum(), "one1": (one == 1).sum(),
            "one2": (one == 2).sum(), "one3": (one == 3).sum(), "one4": (one >= 4).sum(),
            "count_exact": int((kc.d == 0).sum()), "count_n": len(kc),
            "room_rows": int(cnt.n_pipe_room.sum()), "count_rows": int(cnt.n_pipe.sum()),
            "over_cls": ", ".join(f"{c} {r.mean_d:+.1f}" + (f" (max {int(r.max_p)} rows)" if i == 0 else "")
                                  for i, (c, r) in enumerate(top.iterrows())),
            "under_cls": ", ".join(f"{c} {r.mean_d:+.1f}".replace("-", "−") for c, r in low.iterrows()),
            "top_cls": top.index[0], "top_cls_d": top.mean_d.iloc[0], "top_cls_max": int(top.max_p.iloc[0]),
            "frames_n": len(frames), "frames1": (frames == 1).mean(), "frames2": (frames <= 2).mean(),
            "frames_median": float(np.median(frames)),
            "over_cls_names": ", ".join(top.index[:5])}


def sizes(C):
    """Section 4: the longest side per class (median or max over its rows), compared within each question."""
    B = C[C.fp & C.fg]
    res = {}
    for agg, f in [("median", np.median), ("max", np.max)]:
        b = B.assign(p=B.pipe_longest.apply(f), g=B.gold_longest.apply(f))
        R = []
        for i, d in b.groupby("id"):
            if len(d) < 2:
                continue
            lp, lg = np.log2(d.p.max() / d.p.min()), np.log2(d.g.max() / d.g.min())
            rho = np.corrcoef(d.p.rank(), d.g.rank())[0, 1] if len(d) > 2 else np.nan
            R.append(dict(id=i, type=d.type.iloc[0], k=len(d), lp=lp, lg=lg, rho=rho,
                          same_largest=d.p.idxmax() == d.g.idxmax(), same_smallest=d.p.idxmin() == d.g.idxmin()))
        R = pd.DataFrame(R)
        err = R.lp - R.lg
        res[agg] = dict(n=len(R), pipe_ratio=2 ** R.lp.median(), gold_ratio=2 ** R.lg.median(),
                        abs_err=err.abs().median(), within15=(err.abs() < np.log2(1.5)).mean(),
                        within2=(err.abs() < 1).mean(), smaller=(err < 0).mean(), r=np.corrcoef(R.lp, R.lg)[0, 1],
                        largest=R.same_largest.mean(), smallest=R.same_smallest.mean(), chance=(1 / R.k).mean(),
                        rho=R.rho.median(), rho_n=int(R.rho.notna().sum()),
                        by_type=R.groupby("type").same_largest.mean().sort_values(ascending=False))
    b = B.assign(p=B.pipe_longest.apply(np.median), g=B.gold_longest.apply(np.median)).drop_duplicates(["scene", "cls"])
    b = b.assign(s=np.log2(b.p / b.g))
    sp = b.groupby("scene").s.agg(n="size", sd="std", spread=lambda x: 2 ** (x.max() - x.min())).query("n >= 3")
    res["scale"] = dict(scenes=len(sp), sd=sp.sd.median(), factor=2 ** sp.sd.median(), spread=sp.spread.median())
    return res


def examples(Q, C, root, vggt_root):
    """Section 5: per type in EXAMPLE_TYPES, the question with the largest total |pipeline − gold| over its named
    classes (then most classes missing), each from a different scene; both sheets as the eval renders them in csv."""
    K = C[C.fg].assign(ad=lambda d: d.d.abs(), miss=lambda d: d.n_pipe == 0)
    score = K.groupby("id").agg(type=("type", "first"), scene=("scene", "first"), ad=("ad", "sum"),
                                miss=("miss", "sum")).reset_index()
    score = score.sort_values(["ad", "miss", "id"], ascending=[False, False, True])
    out, used = [], set()
    for t in EXAMPLE_TYPES:
        pick = score[(score.type == t) & ~score.scene.isin(used)].iloc[0]
        used.add(pick.scene)
        q = Q.set_index("id").loc[pick.id]
        cells = []
        for source in ss.SOURCES:
            objects = ss.load_objects(int(pick.id), source, root, vggt_root=vggt_root)
            body = ss.render_csv(objects, ss.fields(source)) if objects else ss.EMPTY
            cells.append(body.replace("&", "&amp;").replace("<", "&lt;"))
        c = C[C.id == pick.id]
        notes = ["Pipeline / gold rows per named class: " +
                 ", ".join(f"{r.cls} {r.n_pipe} / {r.n_gold}" for r in c.itertuples()) + "."]
        missing = c[~c.fp & c.fg].cls.tolist()
        if missing:
            notes.append(f"No pipeline rows for {', '.join(missing)}, which gold has.")
        frames = [n for x in c.pipe_frames for n in x]
        if frames:
            notes.append(f"{sum(n == 1 for n in frames)} of the {len(frames)} pipeline rows of named classes are seen "
                         f"in a single frame.")
        if q.n_pipe_room:
            notes.append(f"The pipeline sheet also has {q.n_pipe_room} `room` rows, from the extra prompt.")
        question = q.question.replace("\n", " ")
        out.append(f"### {pick.id}, {TYPES[t].lower()}, `{pick.scene}`\n\n*{question}* Answer: {q["gt"]}.\n\n"
                   + " ".join(notes) + "\n\n<table>\n<tr><th>Pipeline</th><th>Gold</th></tr>\n<tr>\n"
                   + "".join(f'<td valign="top"><pre>\n{cell}\n</pre></td>\n' for cell in cells) + "</tr>\n</table>")
    return "\n\n".join(out)


def report(root, vggt_root):
    Q, C = load(root)
    cov, emp, cnt, sz = coverage(Q, C), empties(Q), counts(Q, C), sizes(C)
    m, x, sc = sz["median"], sz["max"], sz["scale"]
    n_room = int((Q.type == "room_size_estimation").sum())
    size_empty = emp["by_type"].get("object_size_estimation", 0)
    size_share = size_empty / (Q.type == "object_size_estimation").sum()
    order_missing = 1 - C[C.type == "obj_appearance_order"].groupby("id").fp.all().mean()
    count_room = cnt["room_rows"] / cnt["count_rows"]
    by_type = ", ".join(f"{TYPES[t].lower()} {pct(v, 0)}" for t, v in m["by_type"].items() if t != "route_planning")

    return f"""# Pipeline vs gold scene sheets

<!-- Written by sheet_quality.py in this folder; edit that, not this file. -->

Source: `{root.name}` (Dario's run of `run_sheets.sh`: {Q.scene.nunique()} scenes, {num(len(Q))} questions), read
only. Regenerate with `PYTHONPATH=src python experiments/2026-11-18-scene-sheet-formats/sheet_quality.py`.

**Named objects** are a question's prompts from `question.json`, without the prompt `room`. The extractor adds `room`
to every counting question ("in this room") and uses it as the only prompt for room-size questions. Room-size
questions ({n_room}) name no object, so sections 1, 3 and 4 leave them out. That leaves {num(cov['questions'])}
questions and {num(cov['pairs'])} (question, named object) pairs. An object is **found** if its prompt has at least one
row in the sheet. The gold sheet's rows are VSI-Bench's labelled boxes for the class the prompt was matched to
(`sheet.json`, `matched_classes`).

## Summary

- **Pipeline sheets miss about a quarter of the objects the questions ask about.** {pct(cov['pipe'])} of named
  objects have at least one pipeline row, against {pct(cov['gold'])} in gold. Only {pct(cov['allp'])} of questions
  have every named object in the pipeline sheet, against {pct(cov['allg'])} in gold.
- **One object usually becomes several rows.** {pct(cnt['frames1'], 0)} of pipeline rows are seen in only 1 of the 16
  frames, and {pct(cnt['frames2'], 0)} in at most 2. Re-identification rarely joins views of the same object, so large
  objects are over-counted ({cnt['over_cls_names']}). The worst, {cnt['top_cls']}, averages {cnt['top_cls_d']:+.1f}
  rows per scene, with up to {cnt['top_cls_max']} rows for one object.
- **Counts agree only {pct(cnt['match'], 0)} of the time.** For the {num(cnt['one_n'])} pairs where gold has exactly
  one object, which covers most VSI-Bench questions, the pipeline has exactly one row in
  {pct(cnt['one1'] / cnt['one_n'], 0)} of cases. It has none in {pct(cnt['one0'] / cnt['one_n'], 0)} and two or more
  in {pct(1 - (cnt['one0'] + cnt['one1']) / cnt['one_n'], 0)}, which leaves the model to guess which row the question
  means.
- **Empty sheets come from small or rare objects and a few bad scenes.** {emp['gold_nonempty']} of the {emp['n']}
  empty pipeline sheets have a non-empty gold sheet. {size_empty} of the {emp['n']} are object-size questions
  ({pct(size_share, 0)} of them), which usually name a single, often small, object.
- **Relative sizes are roughly right but noisy.** Across a question's objects, the pipeline's ratio of largest to
  smallest object is within 1.5× of gold's in {pct(m['within15'], 0)} of questions. It picks the same largest object
  in {pct(m['largest'], 0)} (chance: {pct(m['chance'], 0)}). Each scene's pipeline-to-metre scale varies by a median
  factor of {sc['factor']:.1f} between objects of the same scene.
- **Gold has its own gaps, all in prompt matching.** {cov['gold_miss_unmatched']} of the {cov['gold_miss']} gold
  misses are prompts matched to no class. {cov['gold_miss_route']} are route-planning landmarks
  ({cov['route_examples']}). The rest come from two extraction bugs: "ceiling light" split into `ceiling` + `light`
  when it ends the question, and "shoes" singularised to `shoe` while the class is `shoes`.

## 1. Are the named objects in the sheets?

Share of (question, named object) pairs with at least one row, and share of questions with every named object:

| Question type | Pairs | Pipeline | Gold | Questions, all found (pipeline) | Questions, all found (gold) |
|---|---:|---:|---:|---:|---:|
{cov['table']}

- Of the {num(cov['pairs'])} pairs, {num(cov['gold_only'])} are in gold but not the pipeline, {cov['pipe_only']} in
  the pipeline but not gold, and {cov['neither']} in neither.
- **Questions that name more objects lose more.** Relative distance names {cov['rd_named']} objects, and only
  {pct(cov['rd_all'], 0)} of those questions have all of them in the pipeline sheet.
- **By dataset**, pipeline coverage is {cov['datasets']}.
- **By scene**, the median is {pct(cov['scene_median'], 0)}, and a quarter of scenes are below
  {pct(cov['scene_q25'], 0)}. The worst scenes are {cov['worst_scenes']}.
- **Classes the pipeline misses most**, counting each (scene, class) once among those gold has, with at least five
  scenes: {cov['missed']}.
- **Classes it almost always finds** (90% or more, at least eight scenes): {cov['found']}.
- **Gold's misses** are the {cov['gold_miss']} unmatched prompts. {cov['gold_miss_route']} are route-planning
  landmarks, which are free text rather than dataset classes. That makes route planning the only type where gold
  covers less than the pipeline. The others are {cov['other_unmatched']}.

## 2. The {emp['n']} empty pipeline sheets

{emp['n']} questions ({emp['scenes']} scenes) have no pipeline rows at all.

| Question type | Empty | Share of type | Gold sheet has objects |
|---|---:|---:|---:|
{emp['table']}

- **Gold has objects for {emp['gold_nonempty']} of the {emp['n']}.** The exceptions: {emp['gold_empty']}.
- **Prompts per empty sheet:** {emp['n_prompts']} prompts. So these are mostly single-object questions where that one
  object was never segmented.
- **Most common single prompts:** {emp['top_prompts']}. The empty multi-prompt sheets (for example
  {emp['multi_examples']}) mostly come from the bad scenes below, where even common objects were missed.
- **Bad scenes:** {emp['bad_total']} of the {emp['n']} come from three scenes: {emp['bad']}.
- **No empty sheets in {' or '.join(emp['no_empties'])}.** Counting always has the `room` prompt, and appearance
  order has four prompts, so at least one usually has rows.
- **What the eval sees:** every format renders an empty sheet as "{ss.EMPTY}"

## 3. Counts of each named class

For the {num(cnt['pairs'])} (question, named class) pairs that have gold rows, pipeline rows minus gold rows:

| Question type | Pairs | Same count | Pipeline more | Pipeline fewer, not 0 | Pipeline 0 | Mean difference |
|---|---:|---:|---:|---:|---:|---:|
{cnt['table']}

- **The usual over-count is about double.** When the pipeline has more rows, it has a median of
  {cnt['over_median']:.0f} extra (mean {cnt['over_mean']:.1f}, 90th percentile {cnt['over_p90']:.0f}). When the
  pipeline finds the class at all, the median ratio of pipeline to gold rows is {cnt['ratio_found']:.1f}.
- **The usual under-count is the object missing entirely.** Only {pct(cnt['under_nonzero'])} of pairs have fewer rows
  than gold without being zero.
- **Unique objects:** gold has exactly one object in {num(cnt['one_n'])} pairs ({pct(cnt['one_share'], 0)}), as
  VSI-Bench intends for every type except counting. There the pipeline has 0 rows in {num(cnt['one0'])}
  ({pct(cnt['one0'] / cnt['one_n'], 0)}), 1 in {num(cnt['one1'])} ({pct(cnt['one1'] / cnt['one_n'], 0)}), 2 in
  {num(cnt['one2'])}, 3 in {num(cnt['one3'])} and 4 or more in {num(cnt['one4'])}.
- **Counting questions:** {cnt['count_exact']} of {cnt['count_n']} are exact. In counting sheets
  {num(cnt['room_rows'])} of the {num(cnt['count_rows'])} rows ({pct(count_room, 0)}) are `room` rows, pieces of the
  room found in single frames. That noise sits next to the rows that matter.
- **By class**, mean pipeline minus gold per (scene, class), classes in at least 8 scenes:
  - over: {cnt['over_cls']};
  - under: {cnt['under_cls']}.
- **Why:** of the {num(cnt['frames_n'])} pipeline rows for named classes (each scene and class once),
  {pct(cnt['frames1'], 0)} are seen in only 1 of the 16 frames and {pct(cnt['frames2'], 0)} in at most 2. The median
  is {cnt['frames_median']:.0f}. Each view of a large object becomes its own track.

## 4. Relative sizes within a question

Units differ (VGGT units vs metres) and so do axes, so this compares sizes within each sheet. Each class's size is
the longest side of its box (the `size_x/y/z` maximum), taken as the median over its rows. Using the largest row
instead gives the numbers after the slash. The sample is the {num(m['n'])} questions with at least two named classes
found in both sheets.

- **Largest-to-smallest ratio:** the median is {m['pipe_ratio']:.2f}× in the pipeline vs {m['gold_ratio']:.2f}× in
  gold ({x['pipe_ratio']:.2f}× vs {x['gold_ratio']:.2f}×). The pipeline compresses size differences, because big
  objects are split into fragments.
- **How close the ratio is:** the median |log₂(pipeline ratio / gold ratio)| is {m['abs_err']:.2f}, a factor of
  {2 ** m['abs_err']:.2f} ({x['abs_err']:.2f}, {2 ** x['abs_err']:.2f}×). It is within 1.5× of gold in
  {pct(m['within15'], 0)} of questions ({pct(x['within15'], 0)}) and within 2× in {pct(m['within2'], 0)}
  ({pct(x['within2'], 0)}). The pipeline's ratio is the smaller one in {pct(m['smaller'], 0)} of questions.
- **Correlation:** the log ratios correlate at r = {m['r']:.2f} ({x['r']:.2f}).
- **Same largest class** in {pct(m['largest'], 0)} of questions ({pct(x['largest'], 0)}), and **same smallest** in
  {pct(m['smallest'], 0)} ({pct(x['smallest'], 0)}). Picking at random would give {pct(m['chance'], 0)}.
- **Rank correlation** over {num(m['rho_n'])} questions with three or more classes: Spearman median {m['rho']:.2f}
  ({x['rho']:.2f}).
- **By type, same largest class:** {by_type}.
- **Scale within a scene:** in {sc['scenes']} scenes with three or more classes in both sheets, each class's
  pipeline-to-gold size ratio varies with a median standard deviation of {sc['sd']:.2f} in log₂, a factor of about
  {sc['factor']:.1f}. The largest and smallest ratios in a scene differ by a median of {sc['spread']:.1f}×. So one
  scale factor per scene, such as from a known object, would still leave sizes off by about {sc['factor']:.1f}×.

## 5. Questions where pipeline and gold disagree most

For each of three question types, this takes the question with the largest total |pipeline − gold| over its named
classes, each from a different scene. The sheets are shown as the eval renders them in `csv`, without the preamble.
Pipeline rows are in VGGT units with the camera axes; gold rows are in metres, z up.

{examples(Q, C, root, vggt_root)}

## What this means for the format experiment

- **The pipeline sheets are a noisy signal.** A format that helps the model use them may still hurt overall, because
  in {pct(1 - cov['allp'], 0)} of questions the sheet is missing a named object, and for {pct(1 - (cnt['one0'] +
  cnt['one1']) / cnt['one_n'], 0)} of unique objects it gives several candidates. The format comparison (B–F against
  each other) is still fair, since all five carry the same content. The comparison with A (video only) mixes format
  with sheet quality.
- **Gold sheets separate the two.** Conditions G–K give the same formats with gold content, which shows how much of
  any gap is the sheet's content rather than its format.
- **Types to read with care:**
  - object size: {pct(size_share, 0)} of sheets are empty, and sizes are in VGGT units;
  - appearance order: {pct(order_missing, 0)} of questions are missing at least one object;
  - counting: the right count in only {pct(cnt['count_exact'] / cnt['count_n'], 0)}, and {pct(count_room, 0)} of rows
    are `room` noise.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(SCENE_SHEETS))
    ap.add_argument("--vggt-root", default=str(VGGT_DIR))
    ap.add_argument("--out", default=str(HERE / "sheet_quality.md"))
    args = ap.parse_args()
    text = report(Path(args.root), Path(args.vggt_root))
    Path(args.out).write_text(text)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
