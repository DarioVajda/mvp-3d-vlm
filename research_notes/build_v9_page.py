"""Build docs/v9_results.html from research_notes/v9_data.json: tables and static SVG charts of run v9 against v7, v8 and SpatialStack."""
import json

W = "/shared/workspace/povejmo/math/projects/mvp/.claude/worktrees/je-xattn-proposal"
D = json.load(open(f"{W}/research_notes/v9_data.json"))
A, RUNS, BOOT, BT = D["agg"], D["runs"], D["boot"], D["boot_task"]
TASKS = [("object_abs_distance", "Absolute distance"), ("object_rel_direction", "Relative direction"), ("room_size_estimation", "Room size"), ("object_rel_distance", "Relative distance"),
         ("route_planning", "Route planning"), ("obj_appearance_order", "Appearance order"), ("object_size_estimation", "Object size"), ("object_counting", "Counting")]
COL = {"v9_lora": "var(--lora)", "v9_add": "var(--add)", "v9_xa": "var(--xa)", "ss": "var(--ss)"}
NM = {"v9_lora": "LoRA only", "v9_add": "Additive", "v9_xa": "Cross-attention", "ss": "SpatialStack-4B"}


def sgn(x, n=1):
    return f"{'+' if x >= 0 else '−'}{abs(x):.{n}f}"


def ci(t):
    return f"{sgn(t[0])} [{sgn(t[1])}, {sgn(t[2])}]"


def cls(t):
    return "pos" if t[1] > 0 else "neg" if t[2] < 0 else ""


def seeds(g):
    return [RUNS[r]["overall"] for r in D["groups"].get(g, [])]


# ---------- chart 1: overall, dots for seeds ----------
def overall_chart():
    rows = [("v7_lora", "v7 LoRA only", "--lora"), ("v9_lora", "v9 LoRA only (1 seed)", "--lora"), ("ss", "SpatialStack-4B", "--ss"), ("v8_late", "v8 xattn, late layers", "--xa"), ("v7_hybrid", "v7 hybrid", "--xa"), ("v8_deep", "v8 xattn, 6 layers", "--xa"),
            ("v7_xa", "v7 cross-attention", "--xa"), ("v7_add", "v7 additive", "--add"), ("v9_xa", "v9 cross-attention", "--xa"), ("v9_add", "v9 additive", "--add")]
    W_, L, R, T, rh = 900, 200, 60, 34, 30
    H = T + rh * len(rows) + 50
    lo, hi = 64, 71
    x = lambda v: L + (v - lo) / (hi - lo) * (W_ - L - R)
    s = [f'<svg viewBox="0 0 {W_} {H}" role="img" aria-label="Overall full VSI-Bench score for ten models; v9 additive 69.9 and v9 cross-attention 69.5 are highest, SpatialStack-4B 67.5.">',
         f'<text class="title" x="0" y="18">Overall VSI-Bench score by model</text><text class="axlab" x="{(L + W_ - R) / 2}" y="{H - 6}" text-anchor="middle">VSI-Bench overall score (%; axis starts at 64)</text>',
         f'<text class="axlab" x="{L - 10}" y="{T - 2}" text-anchor="end">Model</text>']
    for t in range(lo, hi + 1):
        s.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{T}" y2="{H - 50}"/><text class="tick" x="{x(t):.1f}" y="{H - 32}" text-anchor="middle">{t}</text>')
    for i, (g, lab, c) in enumerate(rows):
        y = T + i * rh
        v = A[g]["overall"]
        bold = g in ("v9_add", "v9_xa")
        s.append(f'<text class="lab{" strong" if bold else ""}" x="{L - 10}" y="{y + rh / 2 + 4}" text-anchor="end">{lab}</text>')
        s.append(f'<rect x="{L}" y="{y + 7}" width="{x(v) - L:.1f}" height="{rh - 14}" rx="3" fill="var({c})" opacity="{1 if g.startswith("v9") or g == "ss" else 0.45}"><title>{lab}: {v:.2f}</title></rect>')
        for sv in seeds(g):
            s.append(f'<circle cx="{x(sv):.1f}" cy="{y + rh / 2}" r="4" fill="var(--panel)" stroke="var(--ink)" stroke-width="1.4"><title>seed: {sv:.2f}</title></circle>')
        s.append(f'<text class="lab strong" x="{x(max([v] + seeds(g))) + 9:.1f}" y="{y + rh / 2 + 4}">{v:.1f}</text>')
    s.append("</svg>")
    return "".join(s)


# ---------- chart 2: grouped bars per task ----------
def task_chart():
    gs = ["v9_lora", "v9_add", "v9_xa", "ss"]
    W_, H, L, T, B = 960, 400, 58, 50, 84
    gw = (W_ - L - 8) / len(TASKS)
    bw = gw / 5.2
    y = lambda v: T + (1 - v / 100) * (H - T - B)
    s = [f'<svg viewBox="0 0 {W_} {H}" role="img" aria-label="Scores per VSI-Bench task for LoRA, additive, cross-attention and SpatialStack.">',
         f'<text class="title" x="{L}" y="18">VSI-Bench score by task and model</text>',
         f'<text class="axlab" x="{(L + W_ - 8) / 2}" y="{H - B + 38}" text-anchor="middle">VSI-Bench task</text>',
         f'<text class="axlab" transform="translate(14 {(T + H - B) / 2}) rotate(-90)" text-anchor="middle">Score (%)</text>']
    for t in range(0, 101, 20):
        s.append(f'<line class="grid" x1="{L}" x2="{W_ - 8}" y1="{y(t):.1f}" y2="{y(t):.1f}"/><text class="tick" x="{L - 6}" y="{y(t) + 4:.1f}" text-anchor="end">{t}</text>')
    for i, (t, lab) in enumerate(TASKS):
        x0 = L + i * gw + bw * 0.5
        for j, g in enumerate(gs):
            v = A[g][t]
            s.append(f'<rect x="{x0 + j * bw:.1f}" y="{y(v):.1f}" width="{bw - 2:.1f}" height="{y(0) - y(v):.1f}" rx="2" fill="{COL[g]}"><title>{lab}, {NM[g]}: {v:.1f}</title></rect>')
            s.append(f'<text class="val" x="{x0 + j * bw + bw / 2 - 1:.1f}" y="{y(v) - 4:.1f}" text-anchor="middle">{v:.0f}</text>')
        s.append(f'<text class="lab" x="{L + i * gw + gw / 2:.1f}" y="{H - B + 18}" text-anchor="middle">{lab}</text>')
    for j, g in enumerate(gs):
        lx = L + j * 170
        s.append(f'<rect x="{lx}" y="27" width="11" height="11" rx="2" fill="{COL[g]}"/><text class="lab" x="{lx + 16}" y="37">{NM[g]}</text>')
    s.append("</svg>")
    return "".join(s)


# ---------- chart 3: forest ----------
def forest(key, title, lo, hi, color):
    rows = BT[key]
    W_, L, R, T, rh = 470, 130, 96, 26, 26
    H = T + rh * len(TASKS) + 46
    x = lambda v: L + (min(hi, max(lo, v)) - lo) / (hi - lo) * (W_ - L - R)
    s = [f'<svg viewBox="0 0 {W_} {H}" role="img" aria-label="{title}"><text class="title" x="0" y="14">{title}</text>',
         f'<text class="axlab" x="{L + (W_ - L - R) / 2}" y="{H - 4}" text-anchor="middle">Difference in score (points, 95% interval)</text>',
         f'<text class="axlab" x="{L - 8}" y="{T - 2}" text-anchor="end">Task</text>']
    step = 5 if hi - lo > 12 else 2
    t = lo
    while t <= hi:
        s.append(f'<line class="{"zero" if t == 0 else "grid"}" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{T}" y2="{H - 44}"/><text class="tick" x="{x(t):.1f}" y="{H - 28}" text-anchor="middle">{sgn(t, 0) if t else 0}</text>')
        t += step
    for i, (k, lab) in enumerate(TASKS):
        m, a, b = rows[k]
        y = T + i * rh + rh / 2
        sig = a > 0 or b < 0
        s.append(f'<text class="lab" x="{L - 8}" y="{y + 4}" text-anchor="end">{lab}</text>')
        s.append(f'<line x1="{x(a):.1f}" x2="{x(b):.1f}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="2.4" opacity="{1 if sig else 0.4}"/>')
        s.append(f'<circle cx="{x(m):.1f}" cy="{y}" r="5" fill="{color if sig else "var(--panel)"}" stroke="{color}" stroke-width="2"><title>{lab}: {ci((m, a, b))}</title></circle>')
        s.append(f'<text class="val" x="{W_ - R + 6}" y="{y + 4}">{sgn(m)}</text><text class="val dim" x="{W_ - R + 44}" y="{y + 4}">[{a:.0f},{b:.0f}]</text>')
    s.append("</svg>")
    return "".join(s)


# ---------- chart 4: multiple-choice agreement ----------
def stack():
    m = D["mc_agree"]
    parts = [("Both arms right, all seeds", m["both_right"], "var(--good)"), ("At least one arm's two seeds split", m["seed_split"], "var(--ss)"), ("Both arms wrong, all seeds", m["both_wrong"], "var(--bad)"),
             ("Only additive right", m["add_only"], "var(--add)"), ("Only cross-attention right", m["xa_only"], "var(--xa)")]
    W_, H = 900, 76
    tot = m["n"]
    x, s = 0.0, [f'<svg viewBox="0 0 {W_} {H}" style="min-width:0" role="img" aria-label="Of 2,490 multiple-choice questions, the two arms cleanly disagree on 29."><text class="title" x="0" y="16">Agreement between additive and cross-attention</text>'
                 f'<text class="axlab" x="{W_ / 2}" y="{H - 2}" text-anchor="middle">Share of the {tot:,} multiple-choice questions (full bar = 100%)</text>']
    for lab, v, c in parts:
        w = v / tot * W_
        s.append(f'<rect x="{x:.1f}" y="26" width="{max(w - 2, 3):.1f}" height="30" fill="{c}"><title>{lab}: {v}</title></rect>')
        x += w
    s.append("</svg>")
    rows = "".join(f'<tr><td><span style="display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:8px;background:{c}"></span>{lab}</td>'
                   f'<td class="num">{v:,}</td><td class="num">{100 * v / tot:.1f}%</td></tr>' for lab, v, c in parts)
    return "".join(s) + f'<table style="margin-top:10px"><thead><tr><th>Outcome</th><th class="num">Questions</th><th class="num">Share</th></tr></thead><tbody>{rows}</tbody></table>'


def task_table():
    rows = []
    for k, lab in TASKS:
        vals = [A[g][k] for g in ("ss", "v9_lora", "v9_add", "v9_xa")]
        best = max(vals)
        tds = "".join(f'<td class="num{" best" if v == best else ""}">{v:.1f}</td>' for v in vals)
        g1, g2, g3, g4 = BT["v9_add-v9_lora"][k], BT["v9_xa-v9_add"][k], BT["v9_add-v7_add"][k], BT["v9_add-ss"][k]
        rows.append(f'<tr><td>{lab}</td>{tds}<td class="num {cls(g1)}">{sgn(g1[0])}</td><td class="num {cls(g2)}">{sgn(g2[0])}</td><td class="num {cls(g3)}">{sgn(g3[0])}</td><td class="num {cls(g4)}">{sgn(g4[0])}</td></tr>')
    o = [A[g]["overall"] for g in ("ss", "v9_lora", "v9_add", "v9_xa")]
    b = max(o)
    rows.append('<tr class="total"><td>Overall (mean of 8 tasks)</td>' + "".join(f'<td class="num{" best" if v == b else ""}">{v:.1f}</td>' for v in o) +
                f'<td class="num pos">{sgn(BOOT["v9_add-v9_lora"][0])}</td><td class="num">{sgn(BOOT["v9_xa-v9_add"][0])}</td><td class="num pos">{sgn(BOOT["v9_add-v7_add"][0])}</td><td class="num pos">{sgn(BOOT["v9_add-ss"][0])}</td></tr>')
    return "".join(rows)


def run_table():
    spec = [("v9_add", "v9", "Additive", "frame mix"), ("v9_xa", "v9", "Cross-attention (RoPE + bias, 3D)", "frame mix"), ("v9_lora", "v9", "LoRA only", "frame mix"),
            ("v7_add", "v7", "Additive", "8 frames"), ("v7_xa", "v7", "Cross-attention (RoPE + bias, 3D)", "8 frames"), ("v8_deep", "v8", "Cross-attention at layers 0–2, 6, 10, 14", "8 frames"),
            ("v7_hybrid", "v7", "Hybrid (additive + attention)", "8 frames"), ("v8_late", "v8", "Cross-attention at layers 6, 10, 14", "8 frames"), ("v7_lora", "v7", "LoRA only", "8 frames")]
    out = []
    for g, run, arm, fr in spec:
        s = seeds(g)
        dv = D["dev"][g]
        de = sum(x["dev_acc"] for x in dv) / len(dv)
        deb = sum(x["debiased"] for x in dv) / len(dv)
        out.append(f'<tr><td>{run}</td><td>{arm}</td><td>{fr}</td><td class="num best">{A[g]["overall"]:.1f}</td><td class="num">{" · ".join(f"{x:.1f}" for x in s)}</td><td class="num">{deb:.1f}</td><td class="num">{de:.1f}</td></tr>')
    out.append(f'<tr class="ref"><td>Published</td><td>SpatialStack-Qwen3.5-4B (our reproduction)</td><td>4–8 frames</td><td class="num">{A["ss"]["overall"]:.1f}</td><td class="num">one checkpoint</td><td class="num">–</td><td class="num">–</td></tr>')
    return "".join(out)


def ds_table():
    names = {"arkitscenes": "ARKitScenes", "scannet": "ScanNet", "scannetpp": "ScanNet++"}
    out = ""
    for d, lab in names.items():
        vals = [A[g]["by_ds"][d] for g in ("ss", "v9_lora", "v9_add", "v9_xa")]
        b = max(vals)
        out += f"<tr><td>{lab}</td>" + "".join(f'<td class="num{" best" if v == b else ""}">{v:.1f}</td>' for v in vals) + "</tr>"
    return out


def dir_table():
    out = ""
    for d in ("easy", "medium", "hard"):
        vals = [A[g]["dir_sub"][d] for g in ("ss", "v9_lora", "v9_add", "v9_xa")]
        b = max(vals)
        out += f"<tr><td>{d.capitalize()}</td>" + "".join(f'<td class="num{" best" if v == b else ""}">{v:.1f}</td>' for v in vals) + "</tr>"
    return out


def num_table():
    names = {"object_abs_distance": "Absolute distance", "room_size_estimation": "Room size", "object_size_estimation": "Object size", "object_counting": "Counting"}
    out = ""
    for k, lab in names.items():
        n = D["numeric"]
        out += f"<tr><td>{lab}</td>" + "".join(f'<td class="num">{n[g][k]["within25"]:.0f}%</td><td class="num">{n[g][k]["median_ratio"]:.2f}</td>' for g in ("v9_lora", "v9_add", "v9_xa")) + "</tr>"
    return out


m = D["mc_agree"]; fl = D["lora_flip"]
gaps = D["seed_gap"]
ADD, XA, LO, SS = A["v9_add"], A["v9_xa"], A["v9_lora"], A["ss"]
PAGE = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Run v9 Results</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
/* Layout: answer first, then the main table, then one chart per question. Same colour per arm throughout. */
:root {{ --bg:#f6f7f9; --panel:#fff; --ink:#14171c; --ink2:#4a505b; --ink3:#737a86; --rule:#dde1e7; --rule2:#eceff3; --accent:#3d3a8f; --soft:#e9e8f6;
  --lora:#2a78d6; --add:#eb6834; --xa:#1baf7a; --ss:#9aa1ab; --good:#1d7a3a; --bad:#b3322f; --warn-bg:#fff6e0; --warn:#8a5a00;
  --disp:"IBM Plex Sans Condensed","Arial Narrow",system-ui,sans-serif; --body:"IBM Plex Sans",system-ui,sans-serif; --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#121418; --panel:#1a1d22; --ink:#eceef2; --ink2:#b4bac5; --ink3:#868d99; --rule:#2e333b; --rule2:#23272d; --accent:#a9a6f0; --soft:#26253f;
  --lora:#3987e5; --add:#d95926; --xa:#199e70; --ss:#7c8491; --good:#6cc68a; --bad:#ef7f7c; --warn-bg:#3a2f14; --warn:#f0c060; color-scheme:dark; }} }}
:root[data-theme="dark"] {{ --bg:#121418; --panel:#1a1d22; --ink:#eceef2; --ink2:#b4bac5; --ink3:#868d99; --rule:#2e333b; --rule2:#23272d; --accent:#a9a6f0; --soft:#26253f;
  --lora:#3987e5; --add:#d95926; --xa:#199e70; --ss:#7c8491; --good:#6cc68a; --bad:#ef7f7c; --warn-bg:#3a2f14; --warn:#f0c060; color-scheme:dark; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.6 var(--body); }}
main {{ max-width:1040px; margin:0 auto; padding:40px 16px 80px; }}
h1,h2,h3 {{ font-family:var(--disp); line-height:1.15; text-wrap:balance; margin:0; }}
h1 {{ font-size:clamp(30px,5vw,42px); font-weight:600; }} h2 {{ font-size:25px; font-weight:600; }} h3 {{ font-size:17px; font-weight:600; }}
p,ul {{ margin:0; }} ul {{ padding-left:20px; display:grid; gap:8px; }} code {{ font-family:var(--mono); font-size:.88em; }}
.eyebrow {{ font:500 12px/1.2 var(--mono); letter-spacing:.08em; text-transform:uppercase; color:var(--ink3); }}
.sub {{ color:var(--ink2); font-size:17px; max-width:72ch; }}
section {{ display:grid; gap:16px; padding-top:38px; }}
.prose {{ max-width:76ch; display:grid; gap:10px; }} .lede {{ color:var(--ink2); max-width:76ch; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,225px),1fr)); gap:12px; }}
.tile {{ background:var(--panel); border:1px solid var(--rule); border-radius:8px; padding:14px 16px; display:grid; gap:4px; align-content:start; }}
.tile b {{ font:600 28px/1.1 var(--disp); }} .tile span {{ font-size:13.5px; color:var(--ink2); }}
figure {{ margin:0; background:var(--panel); border:1px solid var(--rule); border-radius:8px; padding:16px 16px 12px; display:grid; gap:10px; min-width:0; }}
figcaption {{ font-size:13.5px; color:var(--ink2); max-width:92ch; }}
.scroll {{ overflow-x:auto; min-width:0; }}
.two {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr)); gap:14px; }}
svg {{ width:100%; height:auto; display:block; min-width:520px; }}
svg .grid {{ stroke:var(--rule2); }} svg .zero {{ stroke:var(--ink3); }}
svg .tick {{ font:11px var(--mono); fill:var(--ink3); }} svg .lab {{ font:12.5px var(--body); fill:var(--ink); }} svg .strong {{ font-weight:600; }}
svg .title {{ font:600 15px var(--disp); fill:var(--ink); }} svg .axlab {{ font:500 12px var(--body); fill:var(--ink2); }}
svg .val {{ font:11px var(--mono); fill:var(--ink); }} svg .dim {{ fill:var(--ink3); }}
table {{ border-collapse:collapse; width:100%; font-size:14px; }}
th,td {{ text-align:left; padding:7px 9px; border-bottom:1px solid var(--rule2); vertical-align:top; }}
thead th {{ font:500 11.5px var(--mono); letter-spacing:.03em; text-transform:uppercase; color:var(--ink3); border-bottom:1px solid var(--rule); }}
td.num,th.num {{ text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums; white-space:nowrap; }}
td.best {{ font-weight:600; }} tr.total td {{ font-weight:600; border-top:1px solid var(--rule); }} tr.ref td {{ color:var(--ink3); }}
.pos {{ color:var(--good); font-weight:500; }} .neg {{ color:var(--bad); font-weight:500; }}
.vert {{ border-left:1px solid var(--rule); }}
.note {{ background:var(--warn-bg); color:var(--warn); border-radius:6px; padding:10px 14px; font-size:14px; }}
footer {{ margin-top:40px; padding-top:16px; border-top:1px solid var(--rule); font-size:13px; color:var(--ink3); }}
</style></head><body><main>
<header style="display:grid;gap:12px;padding-bottom:6px">
  <div class="eyebrow">MVP project · run v9 · full VSI-Bench, 5,130 questions, 32 frames</div>
  <h1>Run v9 Results</h1>
  <p class="sub">v9 is v7 with one change: training videos now use a mix of frame counts (8 frames 60%, 16 frames 30%, 32 frames 10%). All checkpoints below are the same point in training (trunk step 5750, decayed branch), so runs compare directly.</p>
</header>

<section>
  <div class="tiles">
    <div class="tile"><b>{ADD["overall"]:.1f}</b><span>Best result: v9 additive, Qwen3.5-2B, mean of 2 seeds ({" · ".join(f"{x:.1f}" for x in seeds("v9_add"))})</span></div>
    <div class="tile"><b>{sgn(BOOT["v9_add-ss"][0])}</b><span>Additive vs SpatialStack-4B ({SS["overall"]:.1f}), interval [{sgn(BOOT["v9_add-ss"][1])}, {sgn(BOOT["v9_add-ss"][2])}]. A smaller model scores higher.</span></div>
    <div class="tile"><b>{sgn(BOOT["v9_add-v9_lora"][0])}</b><span>What geometry adds over LoRA only, interval [{sgn(BOOT["v9_add-v9_lora"][1])}, {sgn(BOOT["v9_add-v9_lora"][2])}]</span></div>
    <div class="tile"><b>{sgn(BOOT["v9_xa-v9_add"][0])}</b><span>Cross-attention vs additive, interval [{sgn(BOOT["v9_xa-v9_add"][1])}, {sgn(BOOT["v9_xa-v9_add"][2])}]. A tie.</span></div>
    <div class="tile"><b>{sgn(BOOT["v9_add-v7_add"][0])}</b><span>What the frame mix adds to additive (v9 vs v7), interval [{sgn(BOOT["v9_add-v7_add"][1])}, {sgn(BOOT["v9_add-v7_add"][2])}]</span></div>
  </div>
</section>

<section id="runs">
  <div class="sec"><div class="eyebrow">All runs, same checkpoint</div><h2>Overall scores</h2></div>
  <figure><div class="scroll">{overall_chart()}</div>
  <figcaption>Bars are means over seeds; circles are individual seeds. The axis starts at 64 to show the differences. v9 LoRA has only one seed so far, because seed 0 crashed and is being rerun.</figcaption></figure>
  <figure><div class="scroll"><table>
    <thead><tr><th>Run</th><th>Arm</th><th>Training frames</th><th class="num">VSI overall</th><th class="num">Each seed</th><th class="num">Debiased VSI</th><th class="num">Dev accuracy</th></tr></thead>
    <tbody>{run_table()}</tbody></table></div>
  <figcaption>Debiased is VSI-Bench's debiased subset. Dev accuracy is the mean of the VLM-3R task scores on held-out videos. Seed gaps on identical settings run from {min(gaps.values()):.2f} to {max(gaps.values()):.2f} points, so treat a difference under about 1 between single runs as noise.</figcaption></figure>
</section>

<section id="tasks">
  <div class="sec"><div class="eyebrow">Where the points come from</div><h2>By VSI task</h2></div>
  <figure><div class="scroll"><table>
    <thead><tr><th>Task</th><th class="num">SpatialStack-4B</th><th class="num">v9 LoRA</th><th class="num">v9 additive</th><th class="num">v9 xattn</th><th class="num vert">Additive − LoRA</th><th class="num">Xattn − additive</th><th class="num">v9 − v7 additive</th><th class="num">Additive − SpatialStack</th></tr></thead>
    <tbody>{task_table()}</tbody></table></div>
  <figcaption>Bold is the best score in the row. Green or red means the 95% interval excludes zero. These are eight tests per column with no correction, so a borderline task result should be read as a hint. Tasks are sorted by geometry gain.</figcaption></figure>
  <figure><div class="scroll">{task_chart()}</div><figcaption>Route planning is the weakest task for every model, including SpatialStack. Our training data has no route-planning questions at all.</figcaption></figure>
</section>

<section id="diffs">
  <div class="sec"><div class="eyebrow">Paired differences with 95% intervals</div><h2>Compare and contrast</h2>
  <p class="lede">Each dot is a difference on the same questions. A filled dot has an interval that excludes zero. Intervals cover question sampling only, not training variation.</p></div>
  <div class="two">
    <figure><div class="scroll">{forest("v9_add-v9_lora", "What geometry adds: additive − LoRA", -5, 15, "var(--add)")}</div><figcaption>The gain is concentrated: absolute distance and relative direction account for most of it. Counting, object size and appearance order gain nothing. LoRA here is one seed.</figcaption></figure>
    <figure><div class="scroll">{forest("v9_xa-v9_add", "Fusion design: cross-attention − additive", -6, 8, "var(--xa)")}</div><figcaption>No task clearly separates the designs. Relative distance and object size lean to additive; route planning leans to cross-attention with a wide interval.</figcaption></figure>
    <figure><div class="scroll">{forest("v9_add-v7_add", "Frame mix: v9 additive − v7 additive", -6, 8, "var(--accent)")}</div><figcaption>The mix helps absolute distance, relative distance and appearance order. These tests use 32 frames, which v7 never saw in training.</figcaption></figure>
    <figure><div class="scroll">{forest("v9_add-ss", "Against the baseline: additive − SpatialStack-4B", -10, 15, "var(--ss)")}</div><figcaption>Object size gives the largest lead, but LoRA alone also beats SpatialStack there ({LO["object_size_estimation"]:.1f} vs {SS["object_size_estimation"]:.1f}), so that lead comes from training data, not geometry.</figcaption></figure>
  </div>
</section>

<section id="eda">
  <div class="sec"><div class="eyebrow">Other cuts</div><h2>Further comparisons</h2></div>
  <div class="two">
    <figure><h3>By source dataset</h3><div class="scroll"><table><thead><tr><th></th><th class="num">SpatialStack</th><th class="num">LoRA</th><th class="num">Additive</th><th class="num">Xattn</th></tr></thead><tbody>{ds_table()}</tbody></table></div>
    <figcaption>ScanNet++ is the hardest source for all models. Scores are macro averages within each dataset.</figcaption></figure>
    <figure><h3>Relative direction by difficulty</h3><div class="scroll"><table><thead><tr><th></th><th class="num">SpatialStack</th><th class="num">LoRA</th><th class="num">Additive</th><th class="num">Xattn</th></tr></thead><tbody>{dir_table()}</tbody></table></div>
    <figcaption>Geometry helps most on hard questions: additive gains {ADD["dir_sub"]["hard"] - LO["dir_sub"]["hard"]:.1f} on hard and {ADD["dir_sub"]["easy"] - LO["dir_sub"]["easy"]:.1f} on easy.</figcaption></figure>
  </div>
  <figure><h3>Numeric answers: share within 25% of the truth, and median predicted ÷ true</h3><div class="scroll"><table>
    <thead><tr><th></th><th class="num">LoRA ≤25%</th><th class="num">ratio</th><th class="num">Additive ≤25%</th><th class="num">ratio</th><th class="num">Xattn ≤25%</th><th class="num">ratio</th></tr></thead><tbody>{num_table()}</tbody></table></div>
  <figcaption>LoRA underestimates absolute distance (median ratio 0.83). Geometry reduces that bias and moves answers within 25% from 46% to 58%, which is the main source of its gain on that task.</figcaption></figure>
  <figure><h3>Do the two designs answer differently? {m["n"]:,} multiple-choice questions</h3><div class="scroll">{stack()}</div>
  <figcaption>Clean disagreements between designs are rare: {m["add_only"] + m["xa_only"]} questions. Far more questions ({m["seed_split"]}) flip between seeds of one design, so seed noise is much larger than the design effect. Against LoRA, geometry (additive, both seeds) fixes {fl["geo_fixes"]} questions LoRA gets wrong and breaks {fl["geo_breaks"]} it gets right.</figcaption></figure>
  <figure><h3>Where to inject (v8, 8-frame training)</h3><div class="scroll"><table><thead><tr><th>Cross-attention placement</th><th class="num">VSI overall</th></tr></thead><tbody>
    <tr><td>After layers 0, 1, 2 (v7)</td><td class="num best">{A["v7_xa"]["overall"]:.1f}</td></tr><tr><td>After layers 0, 1, 2, 6, 10, 14</td><td class="num">{A["v8_deep"]["overall"]:.1f}</td></tr><tr><td>After layers 6, 10, 14</td><td class="num">{A["v8_late"]["overall"]:.1f}</td></tr></tbody></table></div>
  <figcaption>Three more injection points add nothing, and moving later costs about 1 point on geometry-heavy tasks and appearance order. Early injection stays the choice.</figcaption></figure>
</section>

<section id="read">
  <div class="sec"><div class="eyebrow">Interpretation</div><h2>What this means</h2></div>
  <div class="prose"><ul>
    <li><strong>Most of the lead over SpatialStack comes from data and training, not fusion.</strong> LoRA alone, with no geometry, reaches {LO["overall"]:.1f}. That is only {SS["overall"] - LO["overall"]:.1f} below SpatialStack-4B and above its published no-geometry fine-tune (64.8), on a smaller model. Geometry then adds {BOOT["v9_add-v9_lora"][0]:.1f}, close to SpatialStack's own +2.8.</li>
    <li><strong>Geometry does a specific job.</strong> Absolute distance {sgn(BT["v9_add-v9_lora"]["object_abs_distance"][0])} and relative direction {sgn(BT["v9_add-v9_lora"]["object_rel_direction"][0])} carry it, with room size {sgn(BT["v9_add-v9_lora"]["room_size_estimation"][0])} behind. Counting and object size don't benefit.</li>
    <li><strong>Additive and cross-attention are tied, again.</strong> The difference is {BOOT["v9_xa-v9_add"][0]:.2f} [{BOOT["v9_xa-v9_add"][1]:.2f}, {BOOT["v9_xa-v9_add"][2]:.2f}], and it was {BOOT["v7_xa-v7_add"][0]:.2f} in v7. Additive is simpler and has 50.4 M fewer trainable parameters, so nothing here justifies cross-attention yet.</li>
    <li><strong>The frame mix is a small general gain.</strong> Additive {sgn(BOOT["v9_add-v7_add"][0])}, cross-attention {sgn(BOOT["v9_xa-v7_xa"][0])}, LoRA {sgn(BOOT["v9_lora-v7_lora"][0])}. All three arms gain about the same, so it helps the model generally, not geometry in particular. The 128-frame evals will show whether it also helps at longer inputs.</li>
    <li><strong>The biggest remaining room is route planning ({ADD["route_planning"]:.0f}) and absolute distance ({ADD["object_abs_distance"]:.0f}).</strong> Route planning has no training data here, so added data looks like the cheapest test.</li>
  </ul></div>
  <p class="note"><strong>Limits.</strong> Two seeds per arm, and one for v9 LoRA until the seed 0 rerun finishes. Intervals cover question sampling, not seeds. SpatialStack is one released checkpoint evaluated through its own lmms-eval pipeline, rescored here with our scorer (67.45, against 67.53 in lmms-eval), and I did not verify its prompts match ours token for token. Only one checkpoint of each v9 run was evaluated, so it wasn't picked by test score. I checked that no training video appears in VSI-Bench: file names and ScanNet scene ids show zero overlap. There is no MMSI result for v9, and the 128-frame evals are still running, so neither is included.</p>
</section>
<footer>Sources: <code>results/train/2B_v{{7,8,9}}_*/branches/step5750/eval/vsi_f32_full.jsonl</code> and <code>branches.json</code>; <code>results/baselines/spatialstack/full_qwen35</code>. Paired bootstrap, stratified by task, 800–2,000 resamples.</footer>
</main></body></html>
'''
open(f"{W}/docs/v9_results.html", "w").write(PAGE)
print(len(PAGE))
