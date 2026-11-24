"""Offline evaluation of relation fixes: since the model copies precomputed relations, each fix is scored by its
relation values against the truth (absolute distance: VSI-Bench MRA; relative distance: closest-object accuracy), on the
final test's distance questions and on dev200's. Scales are metres per sheet unit:
  priors        scene_scale (CALIBRATION 0.8417, the unmerged rows' typical sizes): the scale the final test used
  oracle        fitted to the gold sheet (explore_sheets.oracle_scales): a diagnostic upper bound, uses gold
  room height   2.5 m / the floor-to-ceiling extent of the scene's points (16 frames, confident depth), measured along
                gravity, estimated as the mean of the cameras' down axes
  door height   2.0 m / the largest door row's extent along gravity, where the scene has a door row; else priors
  per-dataset   (a) the priors scale with one calibration constant per dataset, (b) one fixed scale per dataset:
                both FITTED ON DEV SCENES (those with a reliable oracle fit); dev results are in-sample
  median        the median of priors, room height and door height (where there is a door)
  >=2 frames    relations only when every named object's row is seen in at least 2 frames (priors scale)
Nothing is fitted on the final test's scenes except the oracle (diagnostic)."""
import csv, json, math, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd

M, E, S = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path[:0] = [str(E)]
import explore_sheets as es
import metric_scale as ms
from mvp.benchmarks import scene_scale
from mvp.benchmarks.vsibench import final_letter

D = M / "data/rerun_sheets"
SETS = {"final": dict(ids=E / "final_ids.txt", root=D / "pipeline_v3_dedup2_explore200", video=E / "results/final/A_video.jsonl"),
        "dev200": dict(ids=E / "dev200_ids.txt", root=D / "pipeline_v3_dedup2_dev50", video=E / "results/dev200/A_video.jsonl")}
DIST = ("object_abs_distance", "object_rel_distance")
mra = lambda p, g: float(np.mean([abs(p - g) / g < 1 - t for t in np.arange(0.5, 1.0, 0.05)]))

def read_index(root):
    return {int(r["id"]): r for r in csv.DictReader(open(root / "index.csv"))}

# ---- per-scene geometry: gravity and room height, from the cameras and the dense depth of the 16 frames
def scene_geometry(scene):
    sd = M / "data/scene_sheets_75" / scene.replace("/", "__", 1) / "subset-uniform16/scene"
    cams = np.load(sd / "cameras.npz"); sel = json.loads((sd / "adapter_report.json").read_text())["selected_frames"]
    ext, K = cams["extrinsic"].astype(np.float64), cams["intrinsic"].astype(np.float64)
    depth = np.load(M / "data/vggt" / scene / "depth.npy", mmap_mode="r")
    conf = np.load(M / "data/vggt" / scene / "depth_conf.npy", mmap_mode="r")
    d16, c16 = np.asarray(depth[sel], np.float32), np.asarray(conf[sel], np.float32)
    thr = np.percentile(c16, 55)
    down = np.mean([e[1, :3] for e in ext], axis=0); g = down / np.linalg.norm(down)  # camera y (down) in world
    hs = []
    H, W = d16.shape[1:]
    v, u = np.mgrid[0:H:4, 0:W:4]
    for f in range(len(sel)):
        z = d16[f, ::4, ::4]; ok = (c16[f, ::4, ::4] >= thr) & (z > 0)
        cam = np.stack([(u - K[f, 0, 2]) * z / K[f, 0, 0], (v - K[f, 1, 2]) * z / K[f, 1, 1], z], -1)[ok]
        R, t = ext[f, :, :3], ext[f, :, 3]
        hs.append(((cam - t) @ R) @ g)
    h = np.concatenate(hs)
    return g, float(np.percentile(h, 99) - np.percentile(h, 1))

def door_extent(root, index, scene, g):
    best = 0.0
    for folder in {r["pipeline_sheet"] for r in index.values() if r["scene"] == scene}:
        for r in csv.DictReader(open(root / folder / "objects.csv")):
            if r["concept"] == "door":
                s = np.array([float(r["size_x"]), float(r["size_y"]), float(r["size_z"])])
                best = max(best, float(np.abs(g) @ s))  # a box's extent along gravity
    return best or None

# ---- relations in sheet units, per question
def relation(index, root, q):
    r = index[q]; text = es.relation_text(r, str(root), "pipeline", "v3")
    if not text:
        return None
    if r["question_type"] == "object_abs_distance":
        return float(re.search(r"Closest-point distance between .*?: ([0-9.]+) m", text).group(1))
    m = re.search(r"Closest to the .*?: (.+)\.$", text.strip().splitlines()[-1])
    return m.group(1) if m else None

def min_frames(index, root, q):
    rows = list(csv.DictReader(open(root / index[q]["pipeline_sheet"] / "objects.csv")))
    return min(int(x["num_frames"]) for x in rows if x["concept"] in es.named_objects(index[q]))

data = {}
for name, cfg in SETS.items():
    index = read_index(cfg["root"]); ids = [int(x) for x in cfg["ids"].read_text().split()]
    video = pd.read_json(cfg["video"], lines=True).set_index("id")
    qs = [q for q in ids if index[q]["question_type"] in DIST]
    data[name] = dict(index=index, root=cfg["root"], video=video, qs=qs, scenes=sorted({index[q]["scene"] for q in qs}))

# scales per scene
all_scenes = sorted({s for d in data.values() for s in d["scenes"]})
# the oracle: only scenes with a reliable fit (metric_scale.scene_data: residual under 0.5 m, cameras 0.8-2.0 m above the
# gold floor), as CALIBRATION was fitted
_rows = defaultdict(list)
for r in csv.DictReader(open(M / "data/scene_sheets_75/index.csv")):
    _rows[r["scene"]].append(r)
dev_scenes = set((E / "dev_scenes.txt").read_text().split())
oracle = {}
for s in sorted(set(all_scenes) | dev_scenes):
    k = ms.scene_data(M / "data/scene_sheets_75", s, _rows[s])["oracle"]
    if k:
        oracle[s] = {"scale": k}
geo = {s: scene_geometry(s) for s in all_scenes}
def priors_ratio(root, s):
    ratio, cls = scene_scale.prior_ratio(scene_scale.scene_longest(str(root), s)); return ratio if cls else None
# per-dataset constants, fitted on dev scenes with a reliable oracle fit
fit = defaultdict(lambda: {"c": [], "k": []})
dev_root = D / "pipeline_v3_dedup2_dev50"
for s in sorted(dev_scenes):
    o = oracle.get(s)
    if not o:
        continue
    ds = s.split("/")[0]; r = priors_ratio(dev_root, s)
    if r: fit[ds]["c"].append(o["scale"] / r)
    fit[ds]["k"].append(o["scale"])
consts = {ds: (float(np.median(v["c"])), float(np.median(v["k"])), len(v["k"])) for ds, v in fit.items()}
print("per-dataset constants fitted on dev scenes (calibration of priors, fixed scale, scenes):", {k: tuple(round(x, 3) if isinstance(x, float) else x for x in v) for k, v in consts.items()})
print(f"reliable oracle fits: {sum(s in oracle for s in data['final']['scenes'])} of {len(data['final']['scenes'])} final-test scenes, "
      f"{sum(s in oracle for s in data['dev200']['scenes'])} of {len(data['dev200']['scenes'])} dev200 scenes, {sum(s in oracle for s in dev_scenes)} of {len(dev_scenes)} dev scenes")

def scales(d, q):
    s = d["index"][q]["scene"]; ds = s.split("/")[0]
    pr = scene_scale.question_scale(q, str(d["root"]))["k"]
    g, H = geo[s]; room = 2.5 / H if H > 0 else None
    de = door_extent(d["root"], d["index"], s, g); door = 2.0 / de if de else None
    ratio = priors_ratio(d["root"], s)
    out = {"(1) priors": pr, "(2) oracle [uses gold]": oracle[s]["scale"] if s in oracle else None,
           "(3) room height": room, "(4) door height (priors if no door)": door if door else pr,
           "(5a) per-dataset calibration [fit on dev]": consts[ds][0] * ratio if ratio else scene_scale.FALLBACK,
           "(5b) per-dataset fixed scale [fit on dev]": consts[ds][1],
           "(6) median of priors, room, door": float(np.median([x for x in (pr, room, door) if x]))}
    return out

rows = []
FIXES = ["(1) priors", "(2) oracle [uses gold]", "(3) room height", "(4) door height (priors if no door)",
         "(5a) per-dataset calibration [fit on dev]", "(5b) per-dataset fixed scale [fit on dev]", "(6) median of priors, room, door",
         "(7) priors, relations only if every object in >=2 frames"]
res = {}
for name, d in data.items():
    idx, video = d["index"], d["video"]
    rel = {q: relation(idx, d["root"], q) for q in d["qs"]}
    sc = {q: scales(d, q) for q in d["qs"]}
    mf = {q: min_frames(idx, d["root"], q) for q in d["qs"] if rel[q] is not None}
    absq = [q for q in d["qs"] if idx[q]["question_type"] == "object_abs_distance"]
    rdq = [q for q in d["qs"] if idx[q]["question_type"] == "object_rel_distance"]
    letter = lambda q, word: next((o.split(". ", 1)[0] for o in idx[q]["options"].split(" | ") if o.split(". ", 1)[1].strip() == word), None)
    for fix in FIXES:
        key = "(1) priors" if fix.startswith("(7)") else fix
        cover = lambda q: rel[q] is not None and (not fix.startswith("(7)") or mf[q] >= 2)
        ca = [q for q in absq if cover(q) and sc[q][key] is not None]
        pred = {q: rel[q] * sc[q][key] for q in ca}
        g = {q: float(idx[q]["ground_truth"]) for q in absq}
        sim_abs = [mra(pred[q], g[q]) if q in pred else float(video.loc[q, "score"]) for q in absq]
        cr = [q for q in rdq if cover(q)]
        sim_rd = [float(letter(q, rel[q]) == idx[q]["ground_truth"]) if q in cr else float(video.loc[q, "score"]) for q in rdq]
        res[(fix, name)] = {"abs n": len(ca), "abs MRA (relations)": 100 * np.mean([mra(pred[q], g[q]) for q in ca]) if ca else float("nan"),
                            "abs median pred/truth": float(np.median([pred[q] / g[q] for q in ca])) if ca else float("nan"),
                            "abs simulated": 100 * np.mean(sim_abs),
                            "rel n": len(cr), "rel acc (relations)": 100 * np.mean([letter(q, rel[q]) == idx[q]["ground_truth"] for q in cr]) if cr else float("nan"),
                            "rel simulated": 100 * np.mean(sim_rd)}
    res[("video only", name)] = {"abs n": 0, "abs MRA (relations)": float("nan"), "abs median pred/truth": float("nan"),
                                 "abs simulated": 100 * np.mean([float(video.loc[q, "score"]) for q in absq]), "rel n": 0,
                                 "rel acc (relations)": float("nan"), "rel simulated": 100 * np.mean([float(video.loc[q, "score"]) for q in rdq])}
    # error decomposition on the questions with a relation and an oracle scale
    ok = [q for q in absq if rel[q] is not None and sc[q]["(2) oracle [uses gold]"]]
    re_ = lambda a, b: abs(a / b - 1)
    tot = np.median([re_(rel[q] * sc[q]["(1) priors"], float(idx[q]["ground_truth"])) for q in ok])
    pos = np.median([re_(rel[q] * sc[q]["(2) oracle [uses gold]"], float(idx[q]["ground_truth"])) for q in ok])
    scl = np.median([re_(sc[q]["(1) priors"], sc[q]["(2) oracle [uses gold]"]) for q in ok])
    kr = np.median([sc[q]["(1) priors"] / sc[q]["(2) oracle [uses gold]"] for q in ok])
    room_r = np.median([sc[q]["(3) room height"] / sc[q]["(2) oracle [uses gold]"] for q in ok if sc[q]["(3) room height"]])
    res[("decomp", name)] = dict(n=len(ok), total=tot, positions=pos, scale=scl, kr=kr, room_r=room_r)

def f(v, kind="pct"):
    return "–" if isinstance(v, float) and math.isnan(v) else (f"{v:.1f}" if kind == "pct" else f"{v:.2f}")
print("\n| fix | final: abs MRA (n) | final: abs pred/truth | final: abs simulated | final: rel acc (n) | final: rel simulated | dev: abs MRA (n) | dev: abs pred/truth | dev: abs simulated | dev: rel acc (n) | dev: rel simulated |")
print("|---|" + "---:|" * 10)
for fix in ["video only"] + FIXES:
    cells = []
    for name in ("final", "dev200"):
        r = res[(fix, name)]
        cells += [f"{f(r['abs MRA (relations)'])} ({r['abs n']})", f(r["abs median pred/truth"], "x"), f(r["abs simulated"]),
                  f"{f(r['rel acc (relations)'])} ({r['rel n']})", f(r["rel simulated"])]
    print(f"| {fix} | " + " | ".join(cells) + " |")
for name in ("final", "dev200"):
    d = res[("decomp", name)]
    print(f"\n{name}: absolute distance with a relation and a reliable oracle scale ({d['n']}): median relative error with priors {100 * d['total']:.0f}%, "
          f"with the oracle scale {100 * d['positions']:.0f}% (what positions and boxes leave), the scale alone {100 * d['scale']:.0f}%; "
          f"priors / oracle scale median {d['kr']:.2f}; room height / oracle {d['room_r']:.2f}")
