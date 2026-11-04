"""Build docs/jspace_results.html from the J-lens readouts: what the lens reads at salient cells, random cells, the geometry delta and the answer token."""
import json
from collections import Counter
import numpy as np

import os
OUT = os.environ.get("JSPACE_OUT", "/shared/workspace/povejmo/math/projects/mvp/results/jspace_v7")
BANNER = ('<p class="note"><strong>Numbers on this page are provisional.</strong> The readouts below came from the wrong checkpoint: the v7 training trunk at step 10,750, before its learning rate was decayed. '
          'The saliency maps and every reported VSI score use the decayed branch of that step, which is a different set of weights and answers some questions differently. The run is being repeated on the correct checkpoint, and this page will be updated with the new numbers. '
          'The letter-bias counts are not affected, because they come from the evaluation files.</p>') if "undecayed" in OUT else ""
W ="/shared/workspace/povejmo/math/projects/mvp/.claude/worktrees/je-xattn-proposal"
MODELS = [("lora_s0", "LoRA only"), ("additive_s0", "Additive")] + ([("xa3_both_s0", "Cross-attention")] if os.path.exists(f"{OUT}/xa3_both_s0/readouts.jsonl") else [])
HAS_XA = len(MODELS) == 3
LAYERS = [0, 1, 2, 5, 8, 11, 14, 17, 20, 22]
FUNC = set(" and , . is in on the a to of with or are as at by for it its an be this that - / ( ) : ; ' \" 's".split()) | {"", " "}
recs = {m: [json.loads(l) for l in open(f"{OUT}/{m}/readouts.jsonl")] for m, _ in MODELS}
qual = {m: json.load(open(f"{OUT}/{m}/lens_quality.json")) for m, _ in MODELS}
san = json.load(open(f"{OUT}/additive_s0/sanity.json"))
clean = lambda n: (n.split("(")[0] if "(" in n else n).strip()
import re as _re
FALLBACK = {"或": "or", "或小": "or small", "或以": "or by", "名词": "noun", "性或": "-ness or", "在床上": "on the bed", "床上": "on the bed", "枕头": "pillow", "污水": "sewage",
            "洗": "wash", "掀": "lift", "发展和": "development and", "上述事实": "the above facts", "再过": "after another", "真菌": "fungus", "カ月": "months (Japanese)", "靴": "boot", "铺": "spread / lay"}


def show(n):
    """Token for display: non-Latin tokens carry their English gloss (from the J-lens repo's Qwen gloss file, else a hand gloss)."""
    base = clean(n)
    if not _re.search(r"[^\x00-\x7f]", base):
        return base
    g = n[n.find("(") + 1:-1] if "(" in n and n.endswith(")") else FALLBACK.get(base)
    return f"{base} ({g})" if g else f"{base} (no gloss)"
path = lambda pts: " ".join(("M" if i == 0 else "L") + f"{px:.1f},{py:.1f}" for i, (px, py) in enumerate(pts))
def best(rk, p):
    """Best 1-based rank (1 = top token) of the words in this group; the files store 0-based ranks."""
    v = [r for k, r in rk.items() if k.startswith(p)]
    return min(v) + 1 if v else None


def axes(Wd, H, L, R, T, B, title, xlab, ylab):
    return (f'<text class="title" x="{L}" y="18">{title}</text>'
            f'<text class="axlab" x="{(L + Wd - R) / 2}" y="{H - 6}" text-anchor="middle">{xlab}</text>'
            f'<text class="axlab" transform="translate(14 {(T + H - B) / 2}) rotate(-90)" text-anchor="middle">{ylab}</text>')


def gt_ranks(m):
    out = []
    for l in LAYERS:
        rr = [r for r in recs[m] if r["lens"] == "text" and r["layer"] == l and r["kind"] == "text"]
        last = {r["id"]: r for r in rr}
        out.append(float(np.median([r["ranks"]["gt"] + 1 for r in last.values()])))
    return out


def med(m, lens, kind, layer, fn):
    rr = [r for r in recs[m] if r["lens"] == lens and r["layer"] == layer and r["kind"] == kind]
    v = [fn(r) for r in rr]; v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def words(m, lens, kind, layer, field, k=5, n=10):
    c = Counter()
    for r in recs[m]:
        if r["lens"] == lens and r["layer"] == layer and r["kind"] == kind and field in r:
            src = r[field]["top"] if isinstance(r[field], dict) else r[field]
            for name, _ in src[:k]:
                w = clean(name)
                if w.lower() not in FUNC and len(w) > 1:
                    c[w] += 1
    return c.most_common(n)


# ---- chart: best question-word rank, salient vs random, by layer (log scale) ----
def rank_chart():
    Wd, H, L, R, T, B = 900, 370, 76, 16, 62, 54
    x = lambda i: L + i / (len(LAYERS) - 1) * (Wd - L - R)
    y = lambda v: T + (1 - np.log10(max(v, 1)) / 5) * (H - T - B)
    s = [f'<svg viewBox="0 0 {Wd} {H}" role="img" aria-label="Median rank of the best question word in the visual-lens readout, salient cells versus random cells, by layer, for LoRA and additive.">',
         axes(Wd, H, L, R, T, B, "Influential cells name the question's objects; random cells don't", "Decoder layer", "Median rank of best question word (log scale)")]
    for t in range(6):
        yy = T + (1 - t / 5) * (H - T - B)
        s.append(f'<line class="grid" x1="{L}" x2="{Wd - R}" y1="{yy:.1f}" y2="{yy:.1f}"/><text class="tick" x="{L - 6}" y="{yy + 4:.1f}" text-anchor="end">{10 ** t:,}</text>')
    for i, l in enumerate(LAYERS):
        s.append(f'<text class="tick" x="{x(i):.1f}" y="{H - B + 16}" text-anchor="middle">{l}</text>')
    series = [("lora_s0", "top", "var(--lora)", "", "LoRA, salient cells"), ("lora_s0", "rand", "var(--lora)", "5 4", "LoRA, random cells"),
              ("additive_s0", "top", "var(--add)", "", "Additive, salient cells"), ("additive_s0", "rand", "var(--add)", "5 4", "Additive, random cells")]
    if HAS_XA:
        series += [("xa3_both_s0", "own", "var(--xa)", "", "Cross-attn, its own salient cells"), ("xa3_both_s0", "rand", "var(--xa)", "5 4", "Cross-attn, random cells")]
    for m, kind, c, dash, lab in series:
        vals = [med(m, "visual", kind, l, lambda r: best(r["ranks"], "q:")) for l in LAYERS]
        pts = [(x(i), y(v)) for i, v in enumerate(vals)]
        s.append(f'<path d="{path(pts)}" fill="none" stroke="{c}" stroke-width="2.2" stroke-dasharray="{dash}"/>')
        for (px, py), l, v in zip(pts, LAYERS, vals):
            s.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.2" fill="{c}"><title>{lab}, layer {l}: rank {v:,.0f}</title></circle>')
    for j, (m, kind, c, dash, lab) in enumerate(series):
        lx = L + (j % 2) * 230 + (j // 2) * 0 if False else L + (j // 2) * 265
        ly = 32 if j % 2 == 0 else 48
        s.append(f'<line x1="{lx}" x2="{lx + 22}" y1="{ly}" y2="{ly}" stroke="{c}" stroke-width="2.2" stroke-dasharray="{dash}"/><text class="lab" x="{lx + 28}" y="{ly + 4}">{lab}</text>')
    s.append("</svg>")
    return "".join(s)


def gt_chart():
    Wd, H, L, R, T, B = 900, 340, 76, 16, 62, 54
    x = lambda i: L + i / (len(LAYERS) - 1) * (Wd - L - R)
    y = lambda v: T + (1 - np.log10(max(v, 1)) / 4) * (H - T - B)
    s = [f'<svg viewBox="0 0 {Wd} {H}" role="img" aria-label="Rank of the correct answer token at the final prompt token under the text lens, by layer.">',
         axes(Wd, H, L, R, T, B, "When the correct answer reaches the top of the final token's readout", "Decoder layer", "Median rank of correct answer (log scale)")]
    for t, lab in ((0, "1"), (1, "10"), (2, "100"), (3, "1,000"), (4, "10,000")):
        yy = T + (1 - t / 4) * (H - T - B)
        s.append(f'<line class="grid" x1="{L}" x2="{Wd - R}" y1="{yy:.1f}" y2="{yy:.1f}"/><text class="tick" x="{L - 6}" y="{yy + 4:.1f}" text-anchor="end">{lab}</text>')
    for i, l in enumerate(LAYERS):
        s.append(f'<text class="tick" x="{x(i):.1f}" y="{H - B + 16}" text-anchor="middle">{l}</text>')
    lx = L
    for m, lab, c in [("lora_s0", "LoRA only", "var(--lora)"), ("additive_s0", "Additive", "var(--add)")] + ([("xa3_both_s0", "Cross-attention", "var(--xa)")] if HAS_XA else []):
        vals = gt_ranks(m)
        pts = [(x(i), y(v)) for i, v in enumerate(vals)]
        s.append(f'<path d="{path(pts)}" fill="none" stroke="{c}" stroke-width="2.2"/>')
        for (px, py), l, v in zip(pts, LAYERS, vals):
            s.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.2" fill="{c}"><title>{lab}, layer {l}: rank {v:,.0f}</title></circle>')
        s.append(f'<line x1="{lx}" x2="{lx + 22}" y1="38" y2="38" stroke="{c}" stroke-width="2.2"/><text class="lab" x="{lx + 28}" y="42">{lab}</text>')
        lx += 140
    s.append("</svg>")
    return "".join(s)


def rank_table():
    rows = ""
    for m, lab in MODELS:
        for kind, kl in (("top", "salient"), ("rand", "random")):
            cells = "".join(f'<td class="num">{med(m, "visual", kind, l, lambda r: best(r["ranks"], "q:")):,.0f}</td><td class="num">{med(m, "visual", kind, l, lambda r: best(r["ranks"], "sp:")):,.0f}</td>' for l in (2, 8, 14, 20))
            rows += f"<tr><td>{lab}</td><td>{kl}</td>{cells}</tr>"
    return rows


def words_table():
    rows = ""
    for m, lab in MODELS:
        for l in (5, 11, 17):
            rows += f"<tr><td>{lab}</td><td class='num'>{l}</td><td>{', '.join(f'{show(w)} ({n})' for w, n in words(m, 'visual', 'top', l, 'contrast', 5, 12))}</td></tr>"
    return rows


def delta_stats(m, kind, l):
    rr = [r for r in recs[m] if r["lens"] == "visual" and r["layer"] == l and r["kind"] == kind and "delta" in r]
    if not rr:
        return None
    return (float(np.median([r["delta"]["ratio"] for r in rr])), float(np.median([best(r["delta"]["ranks"], "sp:") for r in rr])), float(np.median([best(r["delta"]["ranks"], "q:") for r in rr])))


def delta_table():
    rows = ""
    for l in (0, 2, 5, 11, 17, 22):
        ratio, sp, q = delta_stats("additive_s0", "top", l)
        rows += f"<tr><td class='num'>{l}</td><td class='num'>{ratio:.2f}</td><td class='num'>{sp:,.0f}</td><td class='num'>{q:,.0f}</td><td>{', '.join(show(w) for w, _ in words('additive_s0', 'visual', 'top', l, 'delta', 5, 8))}</td></tr>"
    return rows


def xa_rank_rows():
    rows = ""
    for m, lab, kind, kl in (("additive_s0", "Additive", "top", "additive's salient cells"), ("additive_s0", "Additive", "rand", "random cells"), ("xa3_both_s0", "Cross-attention", "top", "additive's salient cells"), ("xa3_both_s0", "Cross-attention", "own", "its own salient cells"), ("xa3_both_s0", "Cross-attention", "rand", "random cells")):
        cells = "".join(f"<td class='num'>{med(m, 'visual', kind, l, lambda r: best(r['ranks'], 'q:')):,.0f}</td>" for l in (2, 8, 14, 20))
        rows += f"<tr><td>{lab}</td><td>{kl}</td>{cells}<td class='num'>{med(m, 'visual', kind, 14, lambda r: best(r['ranks'], 'sp:')):,.0f}</td><td class='num'>{med(m, 'visual', kind, 20, lambda r: best(r['ranks'], 'sp:')):,.0f}</td></tr>"
    return rows


def xa_words_rows():
    rows = ""
    for m, lab, kind in (("additive_s0", "Additive, its salient cells", "top"), ("xa3_both_s0", "Cross-attention, its own salient cells", "own")):
        for l in (5, 11, 17):
            rows += f"<tr><td>{lab}</td><td class='num'>{l}</td><td>{', '.join(f'{show(w)} ({n})' for w, n in words(m, 'visual', kind, l, 'contrast', 5, 12))}</td></tr>"
    return rows


def xa_delta_rows():
    rows = ""
    for l in (0, 2, 5, 11, 17, 22):
        a = delta_stats("additive_s0", "top", l); x = delta_stats("xa3_both_s0", "own", l)
        if a and x:
            rows += f"<tr><td class='num'>{l}</td><td class='num'>{a[0]:.2f}</td><td class='num'>{x[0]:.2f}</td><td class='num'>{a[1]:,.0f}</td><td class='num'>{x[1]:,.0f}</td><td class='num'>{a[2]:,.0f}</td><td class='num'>{x[2]:,.0f}</td><td>{', '.join(show(w) for w, _ in words('xa3_both_s0', 'visual', 'own', l, 'delta', 5, 8))}</td></tr>"
    return rows


def cell_overlap():
    per = []
    for i in dict.fromkeys(r["id"] for r in recs["xa3_both_s0"]):
        top = {r["cell"] for r in recs["xa3_both_s0"] if r["id"] == i and r["kind"] == "top" and r["lens"] == "visual" and r["layer"] == 14}
        own = {r["cell"] for r in recs["xa3_both_s0"] if r["id"] == i and r["kind"] == "own" and r["lens"] == "visual" and r["layer"] == 14}
        per.append((len(top & own), len(top), len(own)))
    return per


EVAL = {x["id"]: x for x in map(json.loads, open("/shared/workspace/povejmo/math/projects/mvp/results/train/2B_v7_additive_s0/branches/step10750/eval/vsi_f32_full.jsonl"))}


def final_table():
    rows = ""
    tail = lambda i, l: [x for x in recs["additive_s0"] if x["lens"] == "text" and x["kind"] == "text" and x["layer"] == l and x["id"] == i][-1]  # the last prompt token
    for i in dict.fromkeys(r["id"] for r in recs["additive_s0"]):
        r = tail(i, 14); e = EVAL[i]
        q = _re.split(r" An object is| The directions refer| If multiple| You will perform| Measuring from the closest point of each object, ", _re.sub(r"\s+", " ", e["question"]).strip())
        q = max(q, key=len) if q[0].startswith("Measuring") or len(q[0]) < 20 else q[0]
        opt = lambda L: next((o.split(".", 1)[1].strip() for o in (e["options"] or []) if o.strip()[0].upper() == str(L).strip()[:1].upper()), None)
        gt = e["ground_truth"] + (f" ({opt(e['ground_truth'])})" if e["options"] else "")
        full = e["prediction"] + (f" ({opt(e['prediction'])})" if e["options"] else "") + (" ✓" if e["score"] == 1 else f" (MRA {e['score']:.1f})" if e["score"] > 0 else " ✗")
        same = (str(r["answer"]).strip()[:1] == str(e["prediction"]).strip()[:1])
        neq = "" if same else " <span style='color:#b3261e'>≠</span>"
        cols = "".join(f"<td>{', '.join(show(n) for n, _ in tail(i, l)['raw'][:5])}</td>" for l in (14, 17, 20))
        rows += (f"<tr><td><strong>{r['type'].replace('object_', '').replace('obj_', '').replace('_', ' ')}</strong> · id {i}<br><span style='color:var(--muted,#666);font-size:.9em'>{q}</span></td><td class='num'>{gt}</td><td class='num'>{full}</td>"
                 f"<td class='num'>{r['answer']}{neq}</td>{cols}</tr>")
    return rows


def sanity_rows():
    out = ""
    for p in ("Fact: The currency used in the country shaped like a boot is", "The capital of the state containing Dallas is", "Q: What is the opposite of hot? A: The opposite of hot is"):
        for l in (8, 14, 20):
            r = next(x for x in san if x["prompt"] == p and x["lens"] == "text" and x["layer"] == l)
            out += f"<tr><td>{p[:52]}…</td><td class='num'>{l}</td><td>{', '.join(show(n) for n, _ in r['top'][:6])}</td></tr>"
    return out


r2 = {m: {l: float(np.median([r["jspace"]["r2"] for r in recs[m] if r["lens"] == "visual" and r["layer"] == l and r["kind"] == "top" and "jspace" in r])) for l in (5, 11, 17)} for m, _ in MODELS}
q14 = qual["additive_s0"]["text"]["14"]
Q14 = {m: med(m, "visual", "top", 14, lambda r: best(r["ranks"], "q:")) for m, _ in MODELS}
GT17 = {m: gt_ranks(m)[LAYERS.index(17)] for m, _ in MODELS}
XA_SECTION = ""
if HAS_XA:
    ov = cell_overlap(); n_ov = sum(o[0] for o in ov); n_tot = sum(o[2] for o in ov)
    qx14 = {k: med("xa3_both_s0", "visual", k, 14, lambda r: best(r["ranks"], "q:")) for k in ("top", "own", "rand")}
    XA_SECTION = f'''<section>
  <div class="eyebrow">Additive versus cross-attention</div><h2>Two ways of injecting geometry, read at the same cells and at each model's own cells</h2>
  <p class="lede">The additive arm adds VGGT features into the image tokens at layers 0–2; the cross-attention arm lets the image tokens attend to VGGT features instead. They score the same on the benchmark. Here the lens reads the cross-attention model at two sets of cells: the cells <em>additive's</em> saliency maps picked (so both models are read at identical positions) and the cells the cross-attention model's <em>own</em> saliency maps picked. Of the {n_tot} own cells (5 per question), {n_ov} are also among additive's top 5, so the two models mostly attend to different cells of the same objects.</p>
  <figure><div class="scroll"><table><thead><tr><th>Model</th><th>Cells read</th><th class="num">L2: question word</th><th class="num">L8</th><th class="num">L14</th><th class="num">L20</th><th class="num">L14: spatial word</th><th class="num">L20: spatial word</th></tr></thead><tbody>{xa_rank_rows()}</tbody></table></div>
  <figcaption>Median over cells of the best rank (1 = top of 151k tokens) of any of the question's object words, and of any spatial word, in the visual-lens readout. At layer 14 the cross-attention model's own salient cells read the question's objects at rank {qx14["own"]:,.0f}, additive's salient cells at {qx14["top"]:,.0f}, random cells at {qx14["rand"]:,.0f}: both models' saliency maps select object cells, and both models represent the object identity there equally well. Spatial words are no more present at salient cells than at random ones for either model.</figcaption></figure>
  <figure><div class="scroll"><table><thead><tr><th>Model and cells</th><th class="num">Layer</th><th>Cell-specific words (readout minus the video's average; count over 50 cells)</th></tr></thead><tbody>{xa_words_rows()}</tbody></table></div>
  <figcaption>The words the two models' salient cells carry are the same kind: object names at layers 5 and 11, and by layer 17 room- and relation-level words.</figcaption></figure>
  <figure><div class="scroll"><table><thead><tr><th class="num">Layer</th><th class="num">Size of geometry's contribution: additive</th><th class="num">cross-attention</th><th class="num">Best spatial-word rank: additive</th><th class="num">cross-attention</th><th class="num">Best question-word rank: additive</th><th class="num">cross-attention</th><th>What cross-attention's contribution reads as</th></tr></thead><tbody>{xa_delta_rows()}</tbody></table></div>
  <figcaption>The geometry contribution h<sub>on</sub> − h<sub>off</sub> for each model at its own salient cells (additive at its salient cells, cross-attention at its own), as in the next section. For both models the contribution is a large fraction of the activation and reads as fragments rather than spatial or object words; the mechanism differs, the picture under the lens does not.</figcaption></figure>
</section>'''
PAGE = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>J-Lens Readouts</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {{ --bg:#f6f7f9; --panel:#fff; --ink:#14171c; --ink2:#4a505b; --ink3:#737a86; --rule:#dde1e7; --rule2:#eceff3; --accent:#3d3a8f; --lora:#2a78d6; --add:#eb6834; --warn-bg:#fff6e0; --warn:#8a5a00;
  --disp:"IBM Plex Sans Condensed","Arial Narrow",system-ui,sans-serif; --body:"IBM Plex Sans",system-ui,sans-serif; --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#121418; --panel:#1a1d22; --ink:#eceef2; --ink2:#b4bac5; --ink3:#868d99; --rule:#2e333b; --rule2:#23272d; --accent:#a9a6f0; --lora:#3987e5; --add:#d95926; --warn-bg:#3a2f14; --warn:#f0c060; color-scheme:dark; }} }}
:root[data-theme="dark"] {{ --bg:#121418; --panel:#1a1d22; --ink:#eceef2; --ink2:#b4bac5; --ink3:#868d99; --rule:#2e333b; --rule2:#23272d; --accent:#a9a6f0; --lora:#3987e5; --add:#d95926; --warn-bg:#3a2f14; --warn:#f0c060; color-scheme:dark; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.6 var(--body); }}
main {{ max-width:1040px; margin:0 auto; padding:40px 16px 80px; }}
h1,h2,h3 {{ font-family:var(--disp); line-height:1.15; text-wrap:balance; margin:0; }}
h1 {{ font-size:clamp(30px,5vw,42px); font-weight:600; }} h2 {{ font-size:25px; font-weight:600; }} h3 {{ font-size:17px; font-weight:600; }}
p,ul {{ margin:0; }} ul {{ padding-left:20px; display:grid; gap:8px; }} code {{ font-family:var(--mono); font-size:.88em; }}
.eyebrow {{ font:500 12px/1.2 var(--mono); letter-spacing:.08em; text-transform:uppercase; color:var(--ink3); }}
.sub {{ color:var(--ink2); font-size:17px; max-width:72ch; }}
section {{ display:grid; gap:16px; padding-top:38px; }}
.prose {{ max-width:76ch; display:grid; gap:10px; }} .prose h3 {{ margin-top:10px; }} .lede {{ color:var(--ink2); max-width:76ch; }}
.key {{ background:var(--panel); border:1px solid var(--rule); border-radius:8px; padding:20px 22px; display:grid; gap:10px; }} .key p {{ font-size:16px; }}
figure {{ margin:0; background:var(--panel); border:1px solid var(--rule); border-radius:8px; padding:16px 16px 12px; display:grid; gap:10px; min-width:0; }}
figcaption {{ font-size:13.5px; color:var(--ink2); max-width:92ch; }}
.scroll {{ overflow-x:auto; min-width:0; }}
svg {{ width:100%; height:auto; display:block; min-width:520px; }} svg .grid {{ stroke:var(--rule2); }} svg .tick {{ font:11px var(--mono); fill:var(--ink3); }} svg .lab {{ font:12.5px var(--body); fill:var(--ink); }} svg .title {{ font:600 15px var(--disp); fill:var(--ink); }} svg .axlab {{ font:500 12px var(--body); fill:var(--ink2); }}
table {{ border-collapse:collapse; width:100%; font-size:14px; }}
th,td {{ text-align:left; padding:7px 9px; border-bottom:1px solid var(--rule2); vertical-align:top; }}
thead th {{ font:500 11.5px var(--mono); letter-spacing:.03em; text-transform:uppercase; color:var(--ink3); border-bottom:1px solid var(--rule); }}
td.num,th.num {{ text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums; white-space:nowrap; }}
.note {{ background:var(--warn-bg); color:var(--warn); border-radius:6px; padding:10px 14px; font-size:14px; }}
footer {{ margin-top:40px; padding-top:16px; border-top:1px solid var(--rule); font-size:13px; color:var(--ink3); }}
</style></head><body><main>
<header style="display:grid;gap:12px">
  <div class="eyebrow">MVP project · interpretability · Jacobian lens on v7 checkpoints · 2026-11-04</div>
  <h1>J-Lens Readouts</h1>
  <p class="sub">What the Jacobian lens (Anthropic, "A global workspace in language models") reads out of our fine-tuned Qwen3.5-2B: at the image cells the saliency maps marked as influential, at random cells, in the geometry injection itself, and at the answer position. LoRA-only, additive and cross-attention, seed 0; 10 questions, one per VSI question type.</p>
  {BANNER}
</header>

<section>
  <div class="key">
    <p><strong>Salient cells read as the objects named in the question; the geometry injection reads as nothing.</strong> At the cells the saliency maps picked, the question's own object words (bed, table, TV, sofa) sit near the top of the lens readout (median best rank {min(Q14.values()):.0f}–{max(Q14.values()):.0f} at layer 14), while at random cells they rank in the thousands. Spatial words are no more present at salient cells than at random ones. What geometry adds to a cell (the activation with the injection minus the activation without it) is large, 20–50% of the cell's activation at every layer, but when the lens tries to read it as a word it gets control tokens and sub-word fragments, not "left", "far" or an object name. Only 7–9% of a visual token's activation lies in the part of the space that reads as words at all. If spatial content is readable anywhere, it is at text tokens, which the object-word study now tests.</p>
  </div>
</section>

<section>
  <div class="eyebrow">Methodology, for someone new to this project</div><h2>Exactly what was done</h2>
  <div class="prose">
    <p><strong>The question.</strong> Our models answer spatial questions about videos. A saliency analysis had already marked which image regions most influence each answer. We wanted to know what the model holds at those regions, in words if possible, and what the added 3D information contributes. Nothing was trained for this; it only reads trained models.</p>

    <h3>1. The models</h3>
    <p>The base is Qwen3.5-2B, a small vision-language model with 24 decoder layers and a hidden size of 2,048. A video goes in as 32 separate frames, and each frame becomes 300 visual tokens on a 20×15 grid, so a question has 9,600 visual tokens followed by its text. We read two fine-tuned models from training run v7, both seed 0:</p>
    <ul>
      <li><strong>LoRA only:</strong> small trainable adapters on Qwen, with no 3D input. This is the control.</li>
      <li><strong>Additive:</strong> the same adapters, plus 3D features from VGGT (a frozen 3D reconstruction model). After decoder layers 0, 1 and 2, each visual token has a learned projection of VGGT's features for that same image cell added to it.</li>
    </ul>
    <p>The intended checkpoint is step 10,750 of v7, after its learning rate was decayed to zero over 750 steps. The same step's decayed checkpoint produced the saliency maps.</p>

    <h3>2. The method: a Jacobian lens</h3>
    <p>The lens is from Anthropic's paper "A global workspace in language models" and asks: what is this internal activation disposed to make the model say? For a chosen layer ℓ it fits one 2,048×2,048 matrix J<sub>ℓ</sub>, the average Jacobian of the final layer's hidden state with respect to layer ℓ's hidden state. To read an activation h, it computes J<sub>ℓ</sub>h, passes that through the final norm and the model's output layer, and ranks all 151k vocabulary tokens. We reimplemented the estimator from the paper's reference code: one backward pass per output dimension, with the output selected at every target position at once, then an average of the gradients over the source positions. A readout is a disposition, not proof that the model is "thinking" that word.</p>

    <h3>3. Fitting two lenses per model</h3>
    <ul>
      <li><strong>Text lens (the paper's approach, on far less data).</strong> 30 Wikipedia passages of at least 600 characters, cut to 128 tokens, skipping the first 16 tokens and the last. The paper used 1,000.</li>
      <li><strong>Visual lens (our addition).</strong> The paper reports no systematic vision experiments, so a text-fitted lens may not suit visual tokens. This lens is fitted on 3 VSI-Bench videos at 8 frames, one question each, averaging over visual tokens as the source positions only. For the additive model, geometry is on. None of these videos are used in the readouts.</li>
    </ul>
    <p>Each model gets its own lenses, because the adapters change the weights. Lenses cover layers 0, 1, 2, 5, 8, 11, 14, 17, 20 and 22, with layer 23 as the target.</p>

    <h3>4. Choosing what to read</h3>
    <ul>
      <li><strong>Questions.</strong> One from each of the 10 VSI question types (relative direction counts as three). We preferred questions where additive was right on both seeds and LoRA wrong on both, which gave 5 of the 10; the other 5 had additive right on both seeds and LoRA right on at least one. This deliberately favours questions where geometry helped, so the sample doesn't represent the benchmark.</li>
      <li><strong>Salient cells.</strong> The 5 cells with the highest influence in the additive model's saliency map for that question (gradient × input of the injected geometry, summed over the three injection points). Both models are read at these same cells, so for LoRA they are additive's cells, not its own.</li>
      <li><strong>Random cells, the control.</strong> 5 random cells per question. Anything also seen here isn't specific to influential cells.</li>
      <li><strong>Text tokens.</strong> The last 12 tokens of the prompt, ending at the token where the model starts its answer.</li>
    </ul>

    <h3>5. What was computed at each position and layer</h3>
    <ul>
      <li><strong>Raw readout:</strong> the top 15 tokens.</li>
      <li><strong>Cell-specific readout (contrast):</strong> the lens's log-probabilities minus their average over 64 random cells of that video, to remove what is common to the whole video.</li>
      <li><strong>Word ranks:</strong> the best rank of the question's own words (after removing common words), the best rank of a fixed list of 38 spatial words chosen in advance (left, near, distance, wall and so on), and the rank of the correct answer's first token. Each word counts in its case and leading-space variants, if it is one token.</li>
      <li><strong>Geometry's contribution (additive only):</strong> the same question is run twice with the same weights, once with injection and once without, and the difference h<sub>on</sub> − h<sub>off</sub> is read through the lens.</li>
      <li><strong>J-space decomposition (visual lens, salient cells and text tokens):</strong> the activation minus the video's mean visual token is approximated by at most 25 nonnegative vocabulary directions, chosen greedily and refitted each time. R² is the share of variance explained. The paper uses gradient pursuit, so ours is a similar method, not the same one.</li>
    </ul>

    <h3>6. Checks</h3>
    <ul>
      <li><strong>Sanity prompts:</strong> three text prompts with known associations, read at the token before last.</li>
      <li><strong>Lens against baseline:</strong> on 20 passages not used for fitting, the rank the lens gives the model's own next token, against the plain logit lens (the same readout with no Jacobian). This score can flatter a lens that favours common words.</li>
      <li><strong>Letter bias:</strong> a separate count over all 2,490 multiple-choice questions in the evaluation files. It uses no lens.</li>
    </ul>

    <h3>7. How to read the numbers</h3>
    <p>Rank 1 is the top token out of 151k, and lower is better. Medians are over 50 cells (5 per question), and cells from one question aren't independent. So treat the results as descriptive: one seed, 10 hand-ruled questions, and no significance tests on the readouts. Everything here is observational. To show the model uses something, you have to change it and watch the answer change, and that is what the separate object-word study does.</p>
  </div>
</section>

<section>
  <div class="eyebrow">Does the lens work on this model?</div><h2>Checks</h2>
  <figure><div class="scroll"><table><thead><tr><th>Prompt (text lens)</th><th class="num">Layer</th><th>Top readout</th></tr></thead><tbody>{sanity_rows()}</tbody></table></div>
  <figcaption>Readouts are sensible but weaker than the paper's on Claude: "Dallas → Texas" and "hot → cold" appear, "boot-shaped country → Italy" does not. At layer 14 the text lens ranks the model's own next token at median {q14["jlens_median_rank"] + 1:.0f} of 151k, against {q14["logit_median_rank"] + 1:,.0f} for the logit lens. The visual lens scores better on that metric but its readouts are dominated by function words (and, in, is), which are common next tokens in wikitext; its cell-level results below are judged against random cells, not that rank.</figcaption></figure>
</section>

<section>
  <div class="eyebrow">Salient cells versus random cells</div><h2>What the influential cells "say"</h2>
  <figure><div class="scroll">{rank_chart()}</div><figcaption>Median over cells of the best rank (log scale, lower is better) of any of the question's own object words in the visual-lens readout. Salient cells name the question's objects; random cells do not. The separation holds for both models and grows with depth.</figcaption></figure>
  <figure><div class="scroll"><table><thead><tr><th>Model</th><th>Cells</th><th class="num">L2 question word</th><th class="num">L2 spatial word</th><th class="num">L8 question</th><th class="num">L8 spatial</th><th class="num">L14 question</th><th class="num">L14 spatial</th><th class="num">L20 question</th><th class="num">L20 spatial</th></tr></thead><tbody>{rank_table()}</tbody></table></div>
  <figcaption>Median best rank. "Spatial word" is the best of 38 words such as left, right, near, far, distance, meter, wall, floor. Spatial words sit at similar ranks at layers 14–20 for salient and random cells alike: the lens sees spatial vocabulary everywhere in a room video, not specifically at the influential cells.</figcaption></figure>
  <figure><div class="scroll"><table><thead><tr><th>Model</th><th class="num">Layer</th><th>Cell-specific words at salient cells (readout minus the video's average; count over 50 cells)</th></tr></thead><tbody>{words_table()}</tbody></table></div>
  <figcaption>Object identity dominates; by layer 17 relational words (near, next, room, sitting, seat) appear for every model. The models read almost identically, so geometry does not change what the lens can verbalise at these cells.</figcaption></figure>
</section>

{XA_SECTION}

<section>
  <div class="eyebrow">The geometry injection</div><h2>Geometry changes the image tokens a lot, but not in words</h2>
  <p class="lede">The question here is simple: <em>what does VGGT geometry add to an image token, and can it be read as language?</em> The test has three steps.</p>
  <ol>
    <li><strong>Run the additive model twice on the same video and question</strong>: once with the geometry injected (h<sub>on</sub>), once with the injection switched off (h<sub>off</sub>). Everything else is identical, so at each image token the difference h<sub>on</sub> − h<sub>off</sub> is exactly what geometry contributed.</li>
    <li><strong>Measure how big that contribution is</strong>: the length of the difference divided by the length of the token's activation without geometry. The second column below shows it is 20–50% of the activation at every layer; geometry is not a small nudge.</li>
    <li><strong>Ask the lens to read the difference as a word.</strong> The lens turns any vector in activation space into a ranking of all 151k vocabulary tokens. If geometry wrote "left", "behind", "far" or "table" into the token, those words should rank near the top. They do not: the best spatial word ranks in the hundreds (third column), the best word from the question in the thousands (fourth), and the tokens that do rank first are control tokens, punctuation and sub-word fragments (last column).</li>
  </ol>
  <figure><div class="scroll"><table><thead><tr><th class="num">Layer</th><th class="num">Size of geometry's contribution<br><span style="font-weight:400">‖h<sub>on</sub> − h<sub>off</sub>‖ / ‖h<sub>off</sub>‖</span></th><th class="num">Best rank of any spatial word<br><span style="font-weight:400">(1 = top of 151k)</span></th><th class="num">Best rank of any question word</th><th>What the difference reads as (top 8 tokens)</th></tr></thead><tbody>{delta_table()}</tbody></table></div>
  <figcaption>Additive model, at the cells the saliency maps picked, visual lens; medians over the 10 questions. Spatial words tested: left, right, front, back, behind, near, far, close, between, above, below, next, distance, meter(s), and their variants; question words are the object names in each question.</figcaption></figure>
  <p>A second, independent check says the same thing from another angle. The "J-space" of the paper is a small set of directions along which activations <em>are</em> readable as words; we fit 25 such directions per layer and asked how much of a salient image token's activation they can rebuild. The answer is a median R² of {r2["additive_s0"][5]:.2f} / {r2["additive_s0"][11]:.2f} / {r2["additive_s0"][17]:.2f} at layers 5 / 11 / 17 for the additive model and {r2["lora_s0"][5]:.2f} / {r2["lora_s0"][11]:.2f} / {r2["lora_s0"][17]:.2f} for LoRA: about 7–9% of the activation lives in the word-readable part of the space, with or without geometry, and the 25 directions themselves have no clear meaning. The paper finds the same ceiling (under 10%) for text tokens in a language model.</p>
  <p><strong>What to take from it.</strong> Geometry is clearly used, since it changes the token substantially and the scores improve, but at the point where it is injected (layers 0–2, at image tokens) it is not stored in a form that maps onto vocabulary. The model does not "think in words" about the geometry there. Whether a readable spatial signal appears later, at the text tokens that name the objects, is what the object-word study tests.</p>
</section>

<section>
  <div class="eyebrow">The answer position</div><h2>The question's topic first, then the answer</h2>
  <p class="note"><strong>Corrected 2026-11-04.</strong> An earlier version of this table read the wrong token: 12 positions before the end, which is the word " choices" or " or" in the prompt, while labelled as the final token. Its readouts (choices, options, or, noun) mostly echoed that word, so the earlier claim that the model represents the answer format first was wrong and is withdrawn. The table below reads the true final token. The rank chart was already correct.</p>
  <figure><div class="scroll">{gt_chart()}</div><figcaption>Rank of the correct answer's first token at the final prompt token, text lens, median over the 10 questions. Both models commit by layer 20; additive's rank reaches {GT17["additive_s0"]:.0f} at layer 17 against {GT17["lora_s0"]:.0f} for LoRA, a hint that geometry lets the model commit earlier, but at n = 10 no more than a hint.</figcaption></figure>
  <figure><div class="scroll"><table><thead><tr><th>Task and question</th><th class="num">Truth</th><th class="num">Full answer from the evaluation run</th><th class="num">First token in the lens run</th><th>Layer 14: top five tokens</th><th>Layer 17: top five tokens</th><th>Layer 20: top five tokens</th></tr></thead><tbody>{final_table()}</tbody></table></div>
  <figcaption>Additive model, seed 0. <strong>Two separate runs are shown.</strong> "Full answer" is what the model generated in the full VSI-Bench evaluation of the decayed step-10,750 branch (the same file the results CSV comes from), with ✓ for a correct answer and the MRA score for a numeric one. "First token in the lens run" is the single most likely next token at the end of the prompt during the lens run, which does not generate further: Qwen writes numbers one digit per token, so an answer of 81 begins with the token 8. When both runs use the same weights the first token must agree with the first character of the full answer; a red ≠ marks a disagreement, which happens only when the lens run used a different checkpoint (see the banner, if present). A readout is a ranking of the whole 151k-token vocabulary; each cell shows the five tokens at the top of that ranking (case and leading-space variants are separate tokens and so appear separately). On numeric questions the final token reads the question's topic at layers 14–17 (distance; two, three; size, height; spacious), then digits by layer 20. On multiple-choice questions layer 14 is mostly uninterpretable, and an option letter appears by layer 17. This is 10 questions on one seed.</figcaption></figure>
</section>

<section>
  <div class="eyebrow">Checked because of the "C" readouts</div><h2>Is there an answer-letter bias?</h2>
  <p class="lede">In these readouts C often tops intermediate prompt tokens: it is the first letter at " choices" on all 6 multiple-choice questions at layer 14, in both models, whatever the answer. So we tested whether that reaches the answers, on all 2,490 multiple-choice questions of the full benchmark, treating every letter alike.</p>
  <figure><div class="scroll"><table><thead><tr><th>Run (v9)</th><th class="num">Correct</th><th class="num">Of those, truth is C</th><th class="num">Wrong</th><th class="num">Of those, chose C</th><th class="num">Expected C if wrong answers were uniform</th><th class="num">Chose non-C</th></tr></thead><tbody>
    <tr><td>LoRA s1</td><td class="num">1,827</td><td class="num">444 (24.3%)</td><td class="num">663</td><td class="num">172 (25.9%)</td><td class="num">167</td><td class="num">491</td></tr>
    <tr><td>Additive s0</td><td class="num">1,931</td><td class="num">465 (24.1%)</td><td class="num">559</td><td class="num">142 (25.4%)</td><td class="num">140</td><td class="num">417</td></tr>
    <tr><td>Additive s1</td><td class="num">1,926</td><td class="num">460 (23.9%)</td><td class="num">564</td><td class="num">138 (24.5%)</td><td class="num">141</td><td class="num">426</td></tr>
    <tr><td>Cross-attention s0</td><td class="num">1,910</td><td class="num">462 (24.2%)</td><td class="num">580</td><td class="num">156 (26.9%)</td><td class="num">148</td><td class="num">424</td></tr>
    <tr><td>Cross-attention s1</td><td class="num">1,923</td><td class="num">467 (24.3%)</td><td class="num">567</td><td class="num">142 (25.0%)</td><td class="num">143</td><td class="num">425</td></tr>
  </tbody></table></div>
  <figcaption>C is the correct answer on 24.2% of these questions (603 of 2,490), matching its share of correct answers. Wrong answers pick C about as often as chance predicts. Across the 12 models checked (v7, v9, SpatialStack), C's predicted share minus its true share runs from −1.0 to +1.0 points, and every 95% interval includes zero. Accuracy when C is correct (recall) is in line with the other letters.</figcaption></figure>
  <div class="prose"><ul>
    <li><strong>Answers show no C bias.</strong> The intermediate C lean does not reach the output on this evidence. At the true final token, the lens's top letter matches the model's answer on 11 of 12 model-question pairs at layer 20.</li>
    <li><strong>The one letter effect is in route planning, and it isn't C.</strong> Models under-choose A (−4.6 to −10.8 points; the interval excludes zero in 4 of 5 runs). With n = 194 and several letters tested, it is only a lead.</li>
    <li><strong>What this cannot rule out.</strong> Evaluation files hold only the chosen answer, so a small C lean in the logits that rarely flips an answer would not show. The intermediate C lean rests on 6 questions, and it could be an artefact of the lens. Rotating the option letters on the same questions would test this directly; that has not been run.</li>
  </ul></div>
</section>

<section>
  <div class="eyebrow">Reading</div><h2>What this does and does not show</h2>
  <div class="prose"><ul>
    <li><strong>The saliency maps pass a second, independent randomness check.</strong> The cells they select are the ones whose activations verbalise the question's objects; random cells do not. The first check was the deletion test on the saliency page (zeroing the top 5% drops confidence 0.62–0.78 vs 0.11–0.12 for a random 5%).</li>
    <li><strong>"Internal thoughts" at visual cells are object identities, not spatial relations.</strong> Nothing here reads as "the chair is left of the table".</li>
    <li><strong>Geometry's contribution is real but non-verbal at the point of injection.</strong> That is consistent with the paper's picture, where the verbalisable workspace is a small slice of activation space, and with our injection being below the paper's workspace layers.</li>
    <li><strong>Limits.</strong> One seed; 10 questions; a 2B model far smaller than those in the paper; lenses fitted on 30 prompts / 3 videos; everything observational, no interventions. The object-word study (recording, probing and patching at the question's object tokens) is the follow-up and is running.</li>
  </ul></div>
  <p class="note">Nothing was trained. Lens fitting and readouts took 19 minutes on one B300. Data: <code>results/jspace_v7/*/readouts.jsonl</code>, <code>sanity.json</code>, <code>lens_quality.json</code>.</p>
</section>
<footer>Method and code: Anthropic, "A global workspace in language models" (2026) and <code>github.com/anthropics/jacobian-lens</code>; our script <code>research_notes/jspace/jspace.py</code>.</footer>
</main></body></html>
'''
open(f"{W}/docs/jspace_results.html", "w").write(PAGE)
print(len(PAGE))
