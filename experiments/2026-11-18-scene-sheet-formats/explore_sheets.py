"""Sheets for explore.ipynb's extra conditions, written next to the original sheets, which stay untouched.

  pipeline_oracle_scale  ORACLE UPPER BOUND: the pipeline sheet in metres, with each scene's scale fitted to its gold
                         sheet (oracle_scales). Everything else is the pipeline's.
  pipeline_v2            ALSO USES THE ORACLE SCALE: the pipeline sheet with split objects merged (merge_rows) and
                         room/wall/floor/ceiling fragments removed (keep_row), in metres with the same oracle scale, so
                         pipeline_v2 - pipeline_oracle_scale is the gain from merging and filtering alone.
  gold_relations         the gold sheet plus precomputed relations between the objects the question names
                         (relations): centre-to-centre distances in metres and, for relative-distance questions, which
                         candidate is closest to the target. The coordinates stay.
  gold_relations_v2      the same with closest-point (box-to-box) distances, as VSI-Bench's distance questions measure
                         them, and the two classes' distance for absolute-distance questions.
  pipeline_relations     pipeline_v2 (ORACLE scale) plus gold_relations_v2's relations, computed from its own rows.

As the eval's entry point, `python explore_sheets.py <mvp.eval.vsibench arguments>` runs mvp.eval.vsibench with the
pipeline preamble replaced by EXPLORE_PREAMBLE (if set) and, with EXPLORE_RELATIONS=1, the relations appended to every
rendered sheet (EXPLORE_RELATIONS=closest: closest-point distances). Nothing else in the eval changes.

  PYTHONPATH=src python experiments/2026-11-18-scene-sheet-formats/explore_sheets.py  # see sheet_quality.py --v2
"""
import csv
import itertools
import os
import shutil
from pathlib import Path

import numpy as np

POS, EXT = ["x", "y", "z"], ["size_x", "size_y", "size_z"]
N_FRAMES = 16
FRAGMENTS = {"room", "wall", "floor", "ceiling"}  # pieces of the room, not objects ("ceiling light" is an object)
CONTEXT_TYPES = {"object_counting", "room_size_estimation"}  # their `room` prompt is context, not a named object
# Merge threshold in metres between the centres of two same-class rows (rows never seen in the same frame), chosen
# with sheet_quality.py --v2 (section 6) on the scenes outside explore_ids.txt: the best mean score of fixed rules
# reading the merged sheet against VSI-Bench's answers. Count match alone favours ever larger thresholds.
MERGE_THRESHOLD = 0.5


def read_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def index_rows(root):
    return read_csv(Path(root) / "index.csv")[0]


# ---- the oracle scale ----

def classes(folders):
    """Distinct rows of several sheet folders, by class: {class: [(centre, extent)]}."""
    out, seen = {}, set()
    for folder in folders:
        for r in read_csv(Path(folder) / "objects.csv")[0]:
            c, e = np.array([float(r[k]) for k in POS]), np.array([float(r[k]) for k in EXT])
            key = (r["concept"], *np.round(c, 4))
            if key not in seen:
                seen.add(key)
                out.setdefault(r["concept"], []).append((c, e))
    return out


def umeyama(P, G):
    """s, R, t minimising |G - (s R P + t)|^2."""
    mp, mg = P.mean(0), G.mean(0)
    U, D, Vt = np.linalg.svd((G - mg).T @ (P - mp) / len(P))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / (P - mp).var(0).sum()
    return s, R, mg - s * R @ mp


def align(gold, pipe):
    """Similarity fit gold ~ s R pipeline + t from matched object centres, seeded on the classes with one gold object
    (RANSAC over triplets), then ICP over every pipeline row against the nearest gold object of its class (the
    metric-sheet experiment's calibrate.py). (s, R, t, residual RMS in metres, rows used), or None."""
    single = [c for c in gold if len(gold[c]) == 1 and pipe.get(c) and c != "room"]
    if len(single) < 3:
        return None
    P = np.array([np.median([c for c, _ in pipe[k]], axis=0) for k in single])
    G = np.array([gold[k][0][0] for k in single])
    best = None
    for tri in itertools.combinations(range(len(single)), 3):
        i = list(tri)
        if np.linalg.matrix_rank(P[i] - P[i].mean(0), tol=1e-3) < 2:
            continue
        s, R, t = umeyama(P[i], G[i])
        res = np.linalg.norm(G - (s * P @ R.T + t), axis=1)
        score = ((res < 0.6).sum(), -np.median(res))
        if best is None or score > best[0]:
            best = (score, res < 0.6)
    if best is None or best[1].sum() < 3:
        return None
    s, R, t = umeyama(P[best[1]], G[best[1]])
    rows = [(c, k) for k in pipe if k in gold and k != "room" for c, _ in pipe[k]]
    for _ in range(15):
        pairs = []
        for c, k in rows:
            q = s * R @ c + t
            d = [np.linalg.norm(g - q) for g, _ in gold[k]]
            j = int(np.argmin(d))
            pairs.append((c, gold[k][j][0], d[j], k))
        d = np.array([p[2] for p in pairs])
        keep = d < max(0.5, np.percentile(d, 60))
        if keep.sum() < 4 or len({p[3] for p, ok in zip(pairs, keep) if ok}) < 3:
            break
        s, R, t = umeyama(np.array([p[0] for p, ok in zip(pairs, keep) if ok]),
                          np.array([p[1] for p, ok in zip(pairs, keep) if ok]))
    res = np.array([p[2] for p, ok in zip(pairs, keep) if ok])
    return float(s), R, t, float(np.sqrt(np.mean(res ** 2))), int(keep.sum())


def centre_distance_scale(gold, pipe):
    """Median, over pairs of classes in both sheets, of gold / pipeline distance between class centres."""
    both = [k for k in gold if pipe.get(k) and k != "room"]
    g = {k: np.mean([c for c, _ in gold[k]], axis=0) for k in both}
    p = {k: np.median([c for c, _ in pipe[k]], axis=0) for k in both}
    r = [np.linalg.norm(g[a] - g[b]) / np.linalg.norm(p[a] - p[b]) for a, b in itertools.combinations(both, 2)
         if np.linalg.norm(p[a] - p[b]) > 1e-3]
    return (float(np.median(r)), len(r)) if r else (None, 0)


def oracle_scales(root, scenes):
    """ORACLE: {scene: {scale, method, detail}}, metres per sheet unit fitted to the scene's gold sheet. The alignment's
    scale when it puts the cameras 0.8-2.0 m above the gold floor, else the centre-distance scale (weaker); scenes with
    neither are left out."""
    root, rows = Path(root), index_rows(root)
    out = {}
    for s in scenes:
        mine = [r for r in rows if r["scene"] == s]
        gold = classes({root / r["gold_sheet"] for r in mine})
        pipe = classes({root / r["pipeline_sheet"] for r in mine})
        fit, cam_h = align(gold, pipe), None
        if fit:
            k, R, t, rms, n = fit
            cams = np.array([[float(c[a]) for a in POS] for c in read_csv(root / mine[0]["pipeline_sheet"] / "cameras.csv")[0]])
            floor = min(c[2] - e[2] / 2 for objs in gold.values() for c, e in objs)
            cam_h = float(np.median((k * cams @ R.T + t)[:, 2]) - floor)
        if fit and 0.8 < cam_h < 2.0:
            out[s] = dict(scale=k, method="alignment", detail=f"RMS {rms:.2f} m over {n} rows; cameras {cam_h:.2f} m "
                                                                 "above the floor")
            continue
        k, n = centre_distance_scale(gold, pipe)
        if k:
            why = ("no alignment (fewer than 3 classes with one gold object)" if not fit
                   else f"alignment put the cameras {cam_h:.2f} m up")
            out[s] = dict(scale=k, method="centre distances (weaker)",
                          detail=f"{why}; median of {n} class pair{'s' * (n != 1)}")
    return out


def oracle_preamble(pipeline_preamble):
    """The pipeline preamble with the units changed to metres; nothing else."""
    text = pipeline_preamble
    for old, new in [("in the 3D reconstruction's relative units, not metres", "in metres"), ("in the same units", "in metres")]:
        assert old in text, old
        text = text.replace(old, new)
    return text


# ---- pipeline_v2: fragments removed, split objects merged ----

def folder_questions(root):
    """{pipeline sheet folder: [index rows of the questions that use it]}."""
    out = {}
    for r in index_rows(root):
        out.setdefault(r["pipeline_sheet"], []).append(r)
    return out


def named_fragments(questions):
    """Fragment classes the folder's questions name as objects: in their prompts, outside counting and room size."""
    return {p.strip() for q in questions if q["question_type"] not in CONTEXT_TYPES
            for p in q["prompts"].split(",")} & FRAGMENTS


def keep_row(concept, questions):
    return concept not in FRAGMENTS or concept in named_fragments(questions)


def as_box(r, scale):
    c, e = np.array([float(r[k]) for k in POS]) * scale, np.array([float(r[k]) for k in EXT]) * scale
    frames = {t for t in range(N_FRAMES) if r.get(f"frame_{t:03d}") == "1"}
    return dict(concept=r["concept"], track_id=int(r["track_id"]), lo=c - e / 2, hi=c + e / 2, frames=frames)


def merge_rows(rows, scale, threshold=MERGE_THRESHOLD, disjoint_frames=True):
    """Join rows of one class that are pieces of one object: repeatedly the closest pair whose box centres are within
    threshold metres or whose boxes overlap, and (disjoint_frames) that were never seen in the same frame, since two
    detections in one frame are two objects. The joined row is the box around both, seen in both rows' frames.
    Returns boxes (dicts with concept, lo, hi, frames), numbered 1, 2, ... per class by first frame."""
    out = []
    for concept in dict.fromkeys(r["concept"] for r in rows):
        items = [as_box(r, scale) for r in rows if r["concept"] == concept]
        while True:
            best = None
            for i, j in itertools.combinations(range(len(items)), 2):
                a, b = items[i], items[j]
                if disjoint_frames and a["frames"] & b["frames"]:
                    continue
                d = float(np.linalg.norm((a["lo"] + a["hi"]) / 2 - (b["lo"] + b["hi"]) / 2))
                overlap = bool(np.all(a["lo"] <= b["hi"]) and np.all(b["lo"] <= a["hi"]))
                if (d <= threshold or overlap) and (best is None or d < best[0]):
                    best = (d, i, j)
            if best is None:
                break
            _, i, j = best
            a, b = items[i], items[j]
            joined = dict(concept=concept, track_id=min(a["track_id"], b["track_id"]), lo=np.minimum(a["lo"], b["lo"]),
                          hi=np.maximum(a["hi"], b["hi"]), frames=a["frames"] | b["frames"])
            items = [x for n, x in enumerate(items) if n not in (i, j)] + [joined]
        items.sort(key=lambda x: (min(x["frames"]) if x["frames"] else N_FRAMES, x["track_id"]))
        out += [{**x, "track_id": n + 1} for n, x in enumerate(items)]
    return out


def v2_boxes(rows, questions, scale, threshold=MERGE_THRESHOLD, disjoint_frames=True):
    """pipeline_v2's objects for one sheet folder: fragments removed, then split objects merged, in metres."""
    return merge_rows([r for r in rows if keep_row(r["concept"], questions)], scale, threshold, disjoint_frames)


def box_rows(boxes):
    """Boxes as objects.csv rows (the pipeline's columns)."""
    rows = []
    for n, b in enumerate(boxes):
        c, e = (b["lo"] + b["hi"]) / 2, b["hi"] - b["lo"]
        r = {"object_id": n, "concept": b["concept"], "track_id": b["track_id"],
             **{k: repr(float(v)) for k, v in zip(POS, c)}, **{k: repr(float(v)) for k, v in zip(EXT, e)},
             "size": repr(float(np.prod(e))), "num_frames": len(b["frames"])}
        r.update({f"frame_{t:03d}": int(t in b["frames"]) for t in range(N_FRAMES)})
        rows.append(r)
    return rows


# ---- writing the sheets ----

def scale_cameras(src, dest, scale):
    rows, fields = read_csv(src)
    for r in rows:
        for k in POS:
            r[k] = repr(float(r[k]) * scale)
    write_csv(dest, rows, fields)


def write_sheets(root, out_root, scales, variant, threshold=MERGE_THRESHOLD):
    """A copy of the sheets folder at out_root with every pipeline sheet of the scaled scenes rewritten: variant
    "oracle" multiplies positions and extents by the scene's scale (size by its cube); "v2" also removes fragments
    and merges split objects. Gold sheets are copied unchanged. Returns the number of pipeline sheets rewritten."""
    root, out_root = Path(root), Path(out_root)
    if out_root.exists():
        shutil.rmtree(out_root)
    shutil.copytree(root, out_root, ignore=shutil.ignore_patterns("masks", "frames", "depth.npz"))
    by_folder, n = folder_questions(root), 0
    for folder, questions in by_folder.items():
        scene = questions[0]["scene"]
        if scene not in scales:
            continue
        k = scales[scene]["scale"] if isinstance(scales[scene], dict) else scales[scene]
        rows, fields = read_csv(root / folder / "objects.csv")
        if variant == "oracle":
            for r in rows:
                for col in POS + EXT:
                    r[col] = repr(float(r[col]) * k)
                r["size"] = repr(float(r["size"]) * k ** 3)
        else:
            rows = box_rows(v2_boxes(rows, questions, k, threshold))
        write_csv(out_root / folder / "objects.csv", rows, fields)
        scale_cameras(root / folder / "cameras.csv", out_root / folder / "cameras.csv", k)
        n += 1
    return n


# ---- relations: gold_relations, gold_relations_v2, pipeline_relations ----

def box_gap(a, b):
    """Distance between the closest points of two axis-aligned boxes (centre c, extent e), 0 if they touch."""
    return float(np.linalg.norm(np.maximum(0, np.abs(a[0] - b[0]) - (a[1] + b[1]) / 2)))


def relations(question, root, source="gold", mode="centre"):
    """Relations between the objects a question names, from its sheet of this source in root, or "" when the sheet
    has fewer than two of the named classes. mode "centre" (gold_relations): distances between object centres.
    mode "closest" (gold_relations_v2, pipeline_relations): distances between the closest points of the objects'
    boxes, as VSI-Bench measures distance ("Measuring from the closest point of each object"). Then a csv of the
    distances in metres between objects of different named classes (ids as the sheet renders them), and for
    absolute-distance questions the smallest distance between the two classes (closest mode), for relative-distance
    questions each candidate's distance to the nearest target and the closest candidate."""
    names = [p.strip() for p in question["prompts"].split(",") if p.strip() != "room" or
             question["question_type"] not in CONTEXT_TYPES]
    folder = question["gold_sheet"] if source == "gold" else question["pipeline_sheet"]
    rows = read_csv(Path(root) / folder / "objects.csv")[0]
    objs = [(f"{r['concept'].replace(' ', '_')}_{r['track_id']}", r["concept"],
             (np.array([float(r[k]) for k in POS]), np.array([float(r[k]) for k in EXT])))
            for r in rows if r["concept"] in names]
    if len({c for _, c, _ in objs}) < 2:
        return ""
    if mode == "centre":
        dist, what = (lambda a, b: float(np.linalg.norm(a[0] - b[0]))), "distance between centres"
    else:
        dist, what = box_gap, "distance between the closest points of their boxes"
    lines = [f"Relations between the objects the question names, from the sheet above ({what}, in metres):",
             "object_a,object_b,distance"]
    lines += [f"{a},{b},{dist(pa, pb):.3f}" for (a, ca, pa), (b, cb, pb) in itertools.combinations(objs, 2) if ca != cb]

    def nearest(c, target):
        return min(dist(p, t) for _, cc, p in objs if cc == c for _, ct, t in objs if ct == target)
    have = {c for _, c, _ in objs}
    if mode == "closest" and question["question_type"] == "object_abs_distance" and len(names) >= 2:
        a, b = names[:2]
        if {a, b} <= have:
            lines.append(f"Closest-point distance between the {a} and the {b}: {nearest(a, b):.3f} m.")
    if question["question_type"] == "object_rel_distance":
        *candidates, target = names
        if target in have:
            d = {c: nearest(c, target) for c in candidates if c in have}
            parts = [f"{c} {d[c]:.3f}" if c in d else f"{c} not in the sheet" for c in candidates]
            if d:
                kind = "centres" if mode == "centre" else "closest points"
                lines.append(f"Distance from each candidate to the nearest {target} ({kind}, metres): "
                             f"{'; '.join(parts)}. Closest to the {target}: {min(d, key=d.get)}.")
    return "\n".join(lines)


def render(question_id, source, fmt, root, vggt_root, preamble=None, relations_mode=None):
    """The sheet as the eval will render it for an explore condition: scene_sheet.render_sheet, with the pipeline
    preamble replaced by `preamble` and, with relations_mode ("centre" or "closest"), the relations appended."""
    from mvp.benchmarks import scene_sheet as ss
    saved = ss.PREAMBLES["pipeline"]
    if preamble:
        ss.PREAMBLES["pipeline"] = preamble
    try:
        text = ss.render_sheet(question_id, source, fmt, Path(root), vggt_root=Path(vggt_root))
    finally:
        ss.PREAMBLES["pipeline"] = saved
    if relations_mode:
        rel = relations(ss.load_index(Path(root))[int(question_id)], root, source, relations_mode)
        text += "\n" + rel if rel else ""
    return text


def main():
    """mvp.eval.vsibench with EXPLORE_PREAMBLE as the pipeline preamble and, with EXPLORE_RELATIONS ("centre", or "1"
    as gold_relations ran it, or "closest"), the relations appended to every sheet."""
    from mvp.benchmarks import scene_sheet as ss
    if os.environ.get("EXPLORE_PREAMBLE"):
        ss.PREAMBLES["pipeline"] = os.environ["EXPLORE_PREAMBLE"]
    from mvp.eval import vsibench
    mode = {"1": "centre", "centre": "centre", "closest": "closest"}.get(os.environ.get("EXPLORE_RELATIONS", ""))
    if mode:
        plain = vsibench.render_sheet

        def with_relations(question_id, source, fmt, root, **kw):
            rel = relations(ss.load_index(Path(root))[int(question_id)], root, source, mode)
            return plain(question_id, source, fmt, root, **kw) + ("\n" + rel if rel else "")
        vsibench.render_sheet = with_relations
    vsibench.main()


if __name__ == "__main__":
    main()
