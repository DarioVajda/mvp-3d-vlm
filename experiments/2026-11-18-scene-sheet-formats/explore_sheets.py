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
  pipeline_v2_intervals  pipeline_v2 (ORACLE scale) with each object's visible time intervals (render_intervals) in
                         place of first_seen and num_frames, as scene_sheet.py writes them since 980cf00.
  gold_csv_shuffled      the gold sheet with its rows in a random order fixed per question (render_shuffled), instead
                         of sorted by label.
  pipeline_v2_priors     pipeline_v2 with the scale estimated from typical object sizes (mvp.benchmarks.scene_scale,
                         no gold) instead of the oracle's; pipeline_relations_priors adds the closest-point relations.

As the eval's entry point, `python explore_sheets.py <mvp.eval.vsibench arguments>` runs mvp.eval.vsibench with the
pipeline preamble replaced by EXPLORE_PREAMBLE (if set) and, with EXPLORE_RELATIONS=1, the relations appended to every
rendered sheet (EXPLORE_RELATIONS=closest: closest-point distances); EXPLORE_INTERVALS=1 writes pipeline sheets with
visible intervals and EXPLORE_SHUFFLE=1 shuffles gold sheets' rows. pipeline_v3's fixes (fix_sheet):
EXPLORE_UNDETECTED=1 adds a line naming the question's objects that have no row in the pipeline sheet, telling
the model to use the video for them; EXPLORE_NO_SIZE=size_questions leaves the size columns out of an
object-size question's pipeline sheet. EXPLORE_RELATIONS=v3 (pipeline_v3_dedup_relations, relation_text) appends,
for relative-direction questions, the direction the sheet's rows imply and, for absolute and relative distance
questions, the closest-point distances between the named objects. EXPLORE_RELATIONS=v4 (pipeline_v4) appends each
question's relations from relations_v4.json in the sheets folder (relations_v4.py). Nothing else in the eval changes.

  PYTHONPATH=src python experiments/2026-11-18-scene-sheet-formats/explore_sheets.py  # see sheet_quality.py --v2
"""
import csv
import itertools
import json
import os
import random
import re
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


def priors_preamble(pipeline_preamble):
    """The pipeline preamble with the units changed to metres estimated from typical object sizes, worded as
    scene_sheet.PIPELINE_AXES_PRIORS; nothing else."""
    text = pipeline_preamble
    for old, new in [("object centres in the 3D reconstruction's relative units, not metres",
                      "object centres, approximately in metres, estimated from typical object sizes"),
                     ("in the same units", "approximately in metres")]:
        assert old in text, old
        text = text.replace(old, new)
    return text


def priors_scale(root):
    """For write_sheets: a pipeline sheet folder's scale from typical object sizes (scene_scale.question_scale), the
    same for every question that uses the folder (an object-size question leaves its class out)."""
    from mvp.benchmarks import scene_scale

    def scale_of(folder, questions):
        ks = {scene_scale.question_scale(int(q["id"]), str(root))["k"] for q in questions}
        assert len(ks) == 1, f"{folder}: its questions need different scales {sorted(ks)}"
        return ks.pop()
    return scale_of


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
    "oracle" multiplies positions and extents by the scale (size by its cube); "v2" also removes fragments and merges
    split objects. scales: {scene: scale} (oracle_scales), or a function (folder, its questions) -> scale
    (priors_scale), for every scene of index.csv whose folders are present. Gold sheets are copied unchanged. Returns
    the number of pipeline sheets rewritten."""
    root, out_root = Path(root), Path(out_root)
    if out_root.exists():
        shutil.rmtree(out_root)
    shutil.copytree(root, out_root, ignore=shutil.ignore_patterns("masks", "frames", "depth.npz"))
    by_folder, n = folder_questions(root), 0
    for folder, questions in by_folder.items():
        scene = questions[0]["scene"]
        if callable(scales):
            if not (root / folder / "objects.csv").exists():
                continue
            k = scales(folder, questions)
        elif scene not in scales:
            continue
        else:
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


# ---- pipeline_v2_intervals: visible intervals instead of first_seen and num_frames ----

# The wording and rules of scene_sheet.py since 980cf00 (VISIBLE["seconds"], BRIDGED_NOTE, visible_frames with
# visibility "bridged" and MAX_GAP 1, runs), which the explore notebooks' code (b381205) predates.
FIRST_SEEN_SENTENCE = ("first_seen is the video timestamp, in seconds, at which the object first appears, as the video's "
                       "timestamps show it, and num_frames is how many of the 16 video frames it appears in.")
VISIBLE_SENTENCES = ("visible gives the stretches of consecutive video frames in which the object is seen, each from its "
                     "first frame to its last, written as the video's timestamps for them, in seconds. The video shows "
                     "one timestamp per pair of frames, so a stretch within one pair starts and ends at the same time. "
                     "The detector sometimes misses an object for a single frame, so one missed frame between two "
                     "sightings is counted as seen.")
MAX_GAP = 1


def intervals_preamble(pipeline_preamble):
    """A pipeline preamble with the first_seen / num_frames sentence replaced by the visible one; nothing else."""
    assert FIRST_SEEN_SENTENCE in pipeline_preamble
    return pipeline_preamble.replace(FIRST_SEEN_SENTENCE, VISIBLE_SENTENCES)


def visible_frames(seen):
    """The frames seen, plus every gap of up to MAX_GAP frames between two of them (visibility "bridged")."""
    seen = sorted(set(seen))
    frames = set(seen)
    for a, b in zip(seen, seen[1:]):
        if b - a - 1 <= MAX_GAP:
            frames.update(range(a + 1, b))
    return frames


def runs(frames):
    """Consecutive frames as (first, last) pairs: {1, 2, 3, 6} -> [(1, 3), (6, 6)]."""
    out = []
    for t in sorted(frames):
        if out and t == out[-1][1] + 1:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [tuple(r) for r in out]


def render_intervals(question_id, root, vggt_root, preamble):
    """The question's pipeline sheet (in root) as csv with id, label, the position and extent, and visible: the
    stretches of frames the object is seen in, as the video's timestamps. Rows sorted by the first visible time, label
    and track id, as both renderers sort them."""
    from mvp.benchmarks import scene_sheet as ss
    row = ss.load_index(Path(root))[int(question_id)]
    folder = Path(root) / row["pipeline_sheet"]
    rows = read_csv(folder / "objects.csv")[0]
    if not rows:
        return preamble + "\n" + ss.EMPTY
    scene = json.loads((folder / "question.json").read_text())["scene"]
    labels = ss.frame_timestamps(folder.parents[1] / "scene", scene, Path(vggt_root))
    objects = []
    for r in rows:
        seen = [t for t in range(N_FRAMES) if r[f"frame_{t:03d}"] == "1"]
        spans = runs(visible_frames(seen))
        objects.append((float(labels[spans[0][0]]), r["concept"], int(r["track_id"]),
                        [f"{r['concept'].replace(' ', '_')}_{r['track_id']}", r["concept"]]
                        + [ss.fmt_float(r[k]) for k in POS + EXT]
                        + [", ".join(f"[{labels[a]}, {labels[b]}]" for a, b in spans)]))
    keys = ["id", "label"] + POS + EXT + ["visible"]
    return preamble + "\n" + ss.render_csv([dict(zip(keys, o[3])) for o in sorted(objects)], keys)


# ---- gold_csv_shuffled: the gold sheet's rows in a random order ----

SHUFFLE_SEED = 0


def render_shuffled(question_id, root, vggt_root, seed=SHUFFLE_SEED):
    """The gold sheet as csv with its rows in a random order fixed by the seed and the question, instead of by label."""
    from mvp.benchmarks import scene_sheet as ss
    objects = ss.load_objects(question_id, "gold", Path(root), vggt_root=Path(vggt_root))
    random.Random(f"{seed}:{int(question_id)}").shuffle(objects)
    return ss.render_objects(objects, "gold", "csv")


# ---- the sheet fixes of pipeline_v3: objects not detected, and no sizes for object-size questions ----

UNDETECTED = ("Not detected: {}. The question names these objects but the scene sheet has no row for them; use the "
              "video for them.")
SIZE_COLUMNS = ("size_x", "size_y", "size_z", "size")


def named_objects(question):
    """The objects a question names: its prompts (from the question's text) without "room"; none for room size."""
    if question["question_type"] == "room_size_estimation":
        return []
    return [p.strip() for p in question["prompts"].split(",") if p.strip() and p.strip() != "room"]


def undetected(question_id, root):
    """The UNDETECTED line for the question's named objects that have no row in its pipeline sheet, or ""."""
    from mvp.benchmarks import scene_sheet as ss
    question = ss.load_index(Path(root))[int(question_id)]
    rows, _ = read_csv(Path(root) / question["pipeline_sheet"] / "objects.csv")
    have = {r["concept"] for r in rows}
    missing = [c for c in dict.fromkeys(named_objects(question)) if c not in have]
    return UNDETECTED.format(", ".join(missing)) if missing else ""


def without_sizes(text):
    """A rendered csv sheet without its size columns and the preamble's sentence about them. A sheet with no rows
    (scene_sheet writes a sentence instead of the table) loses the sentence only."""
    head, sep, table = text.partition("\nid,")
    head = re.sub(r"size_x, size_y and size_z are [^.]*\. ", "", head)
    if not sep:
        return head
    lines = [r for r in ("id," + table).split("\n") if r]
    keep = [i for i, c in enumerate(lines[0].split(",")) if c not in SIZE_COLUMNS]
    return head + "\n" + "\n".join(",".join(r.split(",")[i] for i in keep) for r in lines)


def fix_sheet(text, question_id, source, root, undetected_note=False, no_size=False):
    """A rendered pipeline sheet with pipeline_v3's fixes: for an object-size question no sizes (no_size), and the
    UNDETECTED line (undetected_note). Other sources are returned as they are."""
    if source != "pipeline" or not (undetected_note or no_size):
        return text
    from mvp.benchmarks import scene_sheet as ss
    question = ss.load_index(Path(root))[int(question_id)]
    if no_size and question["question_type"] == "object_size_estimation":
        text = without_sizes(text)
    note = undetected(question_id, root) if undetected_note else ""
    return text + ("\n" + note if note else "")


# ---- pipeline_v3_dedup_relations: the relation mode "v3" ----

DIRECTION_TYPES = {"object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard"}
DISTANCE_TYPES = {"object_abs_distance", "object_rel_distance"}


def direction_word(question_type, by, facing, target):
    """The direction a relative-direction question asks for, from three positions in a pipeline sheet's camera frame
    (x right, y down, z forward), on the floor plane (x, z) seen from above: standing at `by` and facing `facing`,
    where `target` is. Easy: left or right of `facing`; medium: left, right, or back (turning at least 135 degrees);
    hard: front-left, front-right, back-left or back-right."""
    flat = lambda v: np.array([v[0], v[2]])
    right = lambda f: np.array([f[1], -f[0]])
    f = flat(facing) - flat(by)
    if question_type.endswith("easy"):
        return "right" if (flat(target) - flat(facing)) @ right(f) > 0 else "left"
    t = flat(target) - flat(by)
    side = "right" if t @ right(f) > 0 else "left"
    cos = float(t @ f / (np.linalg.norm(t) * np.linalg.norm(f) + 1e-12))
    if question_type.endswith("medium"):
        return "back" if np.degrees(np.arccos(np.clip(cos, -1, 1))) >= 135 else side
    return ("front-" if t @ f > 0 else "back-") + side


def direction_relation(question, root, source="pipeline"):
    """For a relative-direction question whose sheet has each of its three objects exactly once: the direction those
    rows imply, as a sentence; otherwise ""."""
    m = re.search(r"standing by the (.+?) and facing the (.+?), is the (.+?) (?:to (?:my|the)|in)", question["question"])
    if not m:
        return ""
    by, facing, target = m.groups()
    folder = question["gold_sheet"] if source == "gold" else question["pipeline_sheet"]
    rows = read_csv(Path(root) / folder / "objects.csv")[0]
    pos = {}
    for name in (by, facing, target):
        mine = [r for r in rows if r["concept"] == name]
        if len(mine) != 1:
            return ""
        pos[name] = np.array([float(mine[0][k]) for k in POS])
    word = direction_word(question["question_type"], pos[by], pos[facing], pos[target])
    where = f"to the {word} of the {facing}" if question["question_type"].endswith("easy") else f"to your {word}"
    return (f"Direction from the sheet's positions (the floor seen from above): standing by the {by} and facing the "
            f"{facing}, the {target} is {where}.")


V4_RELATIONS = "relations_v4.json"
_v4_cache = {}


def v4_relations(root):
    """{question id: relations} that relations_v4.py wrote in the sheets folder."""
    path = Path(root) / V4_RELATIONS
    if path not in _v4_cache:
        _v4_cache[path] = {int(q): text for q, text in json.loads(path.read_text()).items()}
    return _v4_cache[path]


def relation_text(question, root, source, mode):
    """The relations appended to a sheet: relations() for "centre" and "closest"; for "v3", the direction for
    relative-direction questions (direction_relation) and the closest-point distances for absolute and relative
    distance questions (relations, "closest"), only when the sheet has a row for every object the question names;
    nothing for other questions. For "v4", the question's relations in relations_v4.json (relations_v4.py), or
    nothing."""
    if mode == "v4":
        return v4_relations(root).get(int(question["id"]), "") if source == "pipeline" else ""
    if mode != "v3":
        return relations(question, root, source, mode)
    if question["question_type"] in DIRECTION_TYPES:
        return direction_relation(question, root, source)
    if question["question_type"] in DISTANCE_TYPES:
        folder = question["gold_sheet"] if source == "gold" else question["pipeline_sheet"]
        have = {r["concept"] for r in read_csv(Path(root) / folder / "objects.csv")[0]}
        if not set(named_objects(question)) <= have:
            return ""
        return relations(question, root, source, "closest")
    return ""


def render(question_id, source, fmt, root, vggt_root, preamble=None, relations_mode=None, intervals=False,
           shuffle=False, undetected_note=False, no_size=False):
    """The sheet as the eval will render it for an explore condition: scene_sheet.render_sheet, with the pipeline
    preamble replaced by `preamble`; with intervals, the pipeline sheet with visible intervals (render_intervals, csv,
    `preamble` as given); with shuffle, the gold sheet's rows shuffled (render_shuffled, csv); with undetected_note and
    no_size, pipeline_v3's fixes (fix_sheet); and with relations_mode ("centre", "closest" or "v3", relation_text),
    the relations appended."""
    from mvp.benchmarks import scene_sheet as ss
    saved = ss.PREAMBLES["pipeline"]
    if preamble:
        ss.PREAMBLES["pipeline"] = preamble
    try:
        if intervals and source == "pipeline":
            text = render_intervals(question_id, root, vggt_root, ss.PREAMBLES["pipeline"])
        elif shuffle and source == "gold":
            text = render_shuffled(question_id, root, vggt_root)
        else:
            text = ss.render_sheet(question_id, source, fmt, Path(root), vggt_root=Path(vggt_root))
    finally:
        ss.PREAMBLES["pipeline"] = saved
    text = fix_sheet(text, question_id, source, root, undetected_note, no_size)
    if relations_mode:
        rel = relation_text(ss.load_index(Path(root))[int(question_id)], root, source, relations_mode)
        text += "\n" + rel if rel else ""
    return text


def main():
    """mvp.eval.vsibench with EXPLORE_PREAMBLE as the pipeline preamble and, with EXPLORE_RELATIONS ("centre", or "1"
    as gold_relations ran it, or "closest", or "v3", or "v4": relation_text), the relations appended to every sheet; EXPLORE_UNDETECTED=1 and
    EXPLORE_NO_SIZE=size_questions add pipeline_v3's fixes (fix_sheet)."""
    from mvp.benchmarks import scene_sheet as ss
    if os.environ.get("EXPLORE_PREAMBLE"):
        ss.PREAMBLES["pipeline"] = os.environ["EXPLORE_PREAMBLE"]
    from mvp.eval import vsibench
    mode = {"1": "centre", "centre": "centre", "closest": "closest", "v3": "v3", "v4": "v4"}.get(os.environ.get("EXPLORE_RELATIONS", ""))
    intervals, shuffle = os.environ.get("EXPLORE_INTERVALS") == "1", os.environ.get("EXPLORE_SHUFFLE") == "1"
    undetected_note = os.environ.get("EXPLORE_UNDETECTED") == "1"
    no_size = os.environ.get("EXPLORE_NO_SIZE") == "size_questions"
    if mode or intervals or shuffle or undetected_note or no_size:
        plain = vsibench.render_sheet

        def explore_sheet(question_id, source, fmt, root, vggt_root=None, **kw):
            if intervals and source == "pipeline":
                text = render_intervals(question_id, root, vggt_root, ss.PREAMBLES["pipeline"])
            elif shuffle and source == "gold":
                text = render_shuffled(question_id, root, vggt_root)
            else:
                text = plain(question_id, source, fmt, root, vggt_root=vggt_root, **kw)
            text = fix_sheet(text, question_id, source, root, undetected_note, no_size)
            rel = relation_text(ss.load_index(Path(root))[int(question_id)], root, source, mode) if mode else ""
            return text + ("\n" + rel if rel else "")
        vsibench.render_sheet = explore_sheet
    vsibench.main()


if __name__ == "__main__":
    main()
