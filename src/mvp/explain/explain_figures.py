"""Saliency overlays for the explain_v7 artifact. Examples are chosen by rule, not by eye: per task, among questions all
four geometry checkpoints answer correctly, the one whose cross-arm top-5% overlap is closest to the median. Plus one
question the arms disagree on, and two questions on one video (near the median overlap of such pairs)."""
import json, os, sys, textwrap
sys.path.insert(0, "src")
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mvp.benchmarks.vsibench import load_vsibench, task_of
from mvp.fusion.inputs import uniform_frames

SAL = "results/explain_v7/sal"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
GEO = ["additive_s0", "additive_s1", "xa3_both_s0", "xa3_both_s1"]
HUE = np.array([255, 110, 20], float)
df = load_vsibench("full").set_index("id")
rec = {r: {d["id"]: d for d in map(json.loads, open(f"{SAL}/{r}/summary.jsonl"))} for r in GEO + ["lora_s0"]}
ids = sorted(rec[GEO[0]])
base = {r: {d["id"]: d for d in map(json.loads, open(f"results/train/2B_v7_{r}/branches/step10750/eval/vsi_f32_full.jsonl"))} for r in GEO + ["lora_s0"]}
Z = lambda r, i: np.load(f"{SAL}/{r}/{i}.npz")
dest = lambda r, i: sum(Z(r, i)[f"dest_{k}"] for k in "012")
def overlap(a, b, f=.05):
    n = max(1, int(len(a) * f))
    return len(set(np.argpartition(-a, n)[:n]) & set(np.argpartition(-b, n)[:n])) / n
D = {i: {r: dest(r, i) for r in GEO} for i in ids}
cross = {i: np.mean([overlap(D[i][a], D[i][b]) for a in GEO[:2] for b in GEO[2:]]) for i in ids}
med = np.median(list(cross.values()))
right = lambda r, i: float(base[r][i]["score"]) > .5
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 13); bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
except OSError:
    font = bold = ImageFont.load_default()


def panel(i, name, note=""):
    doc = df.loc[i]
    z = Z("xa3_both_s0", i)
    grid = z["grid"]; T = len(grid)
    h, w = grid[0][1] // 2, grid[0][2] // 2
    assert all(g[1] // 2 == h and g[2] // 2 == w for g in grid)
    frames, _ = uniform_frames(doc["video_path"], 32)
    mean = sum(D[i][r] / D[i][r].sum() for r in GEO).reshape(T, h, w)
    show = sorted(np.argsort(-mean.sum((1, 2)))[:4].tolist())
    # the most salient token of xattn s0 at layer 1, and where it reads from
    top = int(z["top_1"]); ft, cell = top // (h * w), top % (h * w)
    attn = z["attn_top_1"].reshape(T, h, w)
    if ft not in show:
        show[-1] = ft; show = sorted(show)
    rows = [("additive s0", D[i]["additive_s0"]), ("additive s1", D[i]["additive_s1"]), ("xattn s0", D[i]["xa3_both_s0"]),
            ("xattn s1", D[i]["xa3_both_s1"]), ("LoRA s0 (vision map)", Z("lora_s0", i)["vis"]), ("xattn s0: attention of the boxed token, layer 1", attn.ravel())]
    cw = 250; ch = int(cw * frames[0].height / frames[0].width); lw = 130; gap = 4
    opts = doc["options"] if isinstance(doc["options"], list) else []
    text = textwrap.wrap(f"[{task_of(doc['question_type'])}] {doc['question']}" + ("  " + "  ".join(opts) if opts else ""), 125)
    answers = f"correct: {doc['ground_truth']}   |   " + "   ".join(f"{r.replace('xa3_both', 'xattn')}: {str(base[r][i]['prediction']).strip()[:12]}" for r in GEO + ["lora_s0"])
    head = 18 * len(text) + 28 + (18 if note else 0)
    W, H = lw + 4 * (cw + gap), head + 22 + len(rows) * (ch + gap)
    im = Image.new("RGB", (W, H), (255, 255, 255)); dr = ImageDraw.Draw(im)
    y = 6
    for line in text:
        dr.text((6, y), line, font=bold, fill=(20, 20, 20)); y += 18
    dr.text((6, y), answers, font=font, fill=(60, 60, 60)); y += 18
    if note:
        dr.text((6, y), note, font=font, fill=(60, 60, 60)); y += 18
    y = head
    for c, t in enumerate(show):
        dr.text((lw + c * (cw + gap) + 4, y + 3), f"frame {t + 1} of {T}", font=font, fill=(90, 90, 90))
    y += 22
    for lab, m in rows:
        m = np.asarray(m, float).reshape(T, h, w)
        scale = np.percentile(m, 99.5) + 1e-12
        for ln, part in enumerate(textwrap.wrap(lab, 17)):
            dr.text((6, y + 6 + 16 * ln), part, font=font, fill=(30, 30, 30))
        for c, t in enumerate(show):
            f = np.asarray(frames[t].convert("L").resize((cw, ch)), float)[..., None].repeat(3, -1) * .75 + 50
            a = np.clip(m[t] / scale, 0, 1)
            a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((cw, ch), Image.Resampling.BILINEAR), float)[..., None] / 255
            out = f * (1 - .85 * a) + HUE * .85 * a
            x0 = lw + c * (cw + gap)
            im.paste(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), (x0, y))
            if "attention" in lab and t == ft:
                r, cc = cell // w, cell % w
                bx = (x0 + cc * cw / w, y + r * ch / h, x0 + (cc + 1) * cw / w, y + (r + 1) * ch / h)
                dr.rectangle(bx, outline=(0, 90, 255), width=3)
        y += ch + gap
    im.save(f"{OUT}/{name}.png", optimize=True)
    return {"name": name, "id": int(i), "task": task_of(doc["question_type"]), "cross_overlap": cross[i], "question": doc["question"],
            "note": note, "frames": [t + 1 for t in show]}


meta = []
tasks = ["object_rel_direction", "object_abs_distance", "room_size_estimation", "object_rel_distance", "obj_appearance_order", "object_counting"]
for t in tasks:
    c = [i for i in ids if task_of(df.loc[i]["question_type"]) == t and all(right(r, i) for r in GEO)]
    i = min(c, key=lambda i: abs(cross[i] - med))
    meta.append(panel(i, f"ex_{t}"))
    print(t, i, round(cross[i], 3), f"of {len(c)}", flush=True)
c = [i for i in ids if base[GEO[0]][i]["options"] is not None and right(GEO[0], i) and right(GEO[1], i) and not right(GEO[2], i) and not right(GEO[3], i)]
c2 = [i for i in ids if base[GEO[0]][i]["options"] is not None and not right(GEO[0], i) and not right(GEO[1], i) and right(GEO[2], i) and right(GEO[3], i)]
print("additive right, xattn wrong:", len(c), "| xattn right, additive wrong:", len(c2), "of", sum(base[GEO[0]][i]["options"] is not None for i in ids))
if c:
    i = min(c, key=lambda i: abs(cross[i] - med)); meta.append(panel(i, "dis_add_right", "both additive seeds right, both xattn seeds wrong"))
if c2:
    i = min(c2, key=lambda i: abs(cross[i] - med)); meta.append(panel(i, "dis_xa_right", "both xattn seeds right, both additive seeds wrong"))
byv = {}
for i in ids:
    byv.setdefault(rec[GEO[0]][i]["video"], []).append(i)
pairs = [(a, b) for v in byv.values() for x, a in enumerate(v) for b in v[x + 1:] if task_of(df.loc[a]["question_type"]) != task_of(df.loc[b]["question_type"])]
ov = {p: np.mean([overlap(D[p[0]][r], D[p[1]][r]) for r in GEO]) for p in pairs}
p = min(pairs, key=lambda p: abs(ov[p] - np.median(list(ov.values()))))
print("same-video pair", p, round(ov[p], 3))
meta.append(panel(p[0], "pair_a", f"same video as the next figure; top-5% overlap between the two questions {ov[p]:.2f}"))
meta.append(panel(p[1], "pair_b", f"same video as the previous figure; top-5% overlap between the two questions {ov[p]:.2f}"))
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
