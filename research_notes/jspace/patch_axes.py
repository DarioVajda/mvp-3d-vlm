P = "/shared/workspace/povejmo/math/projects/mvp/.claude/worktrees/je-xattn-proposal/research_notes/jspace/build_page.py"
s = open(P).read()
s = s.replace('''best = lambda rk, p: min([r for k, r in rk.items() if k.startswith(p)] or [None])''',
 '''def best(rk, p):
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
    return out''')
a = s.index("def rank_chart():"); b = s.index("def rank_table():")
s = s[:a] + '''def rank_chart():
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
    for m, kind, c, dash, lab in series:
        vals = [med(m, "visual", kind, l, lambda r: best(r["ranks"], "q:")) for l in LAYERS]
        pts = [(x(i), y(v)) for i, v in enumerate(vals)]
        s.append(f'<path d="{path(pts)}" fill="none" stroke="{c}" stroke-width="2.2" stroke-dasharray="{dash}"/>')
        for (px, py), l, v in zip(pts, LAYERS, vals):
            s.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.2" fill="{c}"><title>{lab}, layer {l}: rank {v:,.0f}</title></circle>')
    lx = L
    for m, kind, c, dash, lab in series:
        s.append(f'<line x1="{lx}" x2="{lx + 22}" y1="38" y2="38" stroke="{c}" stroke-width="2.2" stroke-dasharray="{dash}"/><text class="lab" x="{lx + 28}" y="42">{lab}</text>')
        lx += 200
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
    for m, lab, c in (("lora_s0", "LoRA only", "var(--lora)"), ("additive_s0", "Additive", "var(--add)")):
        vals = gt_ranks(m)
        pts = [(x(i), y(v)) for i, v in enumerate(vals)]
        s.append(f'<path d="{path(pts)}" fill="none" stroke="{c}" stroke-width="2.2"/>')
        for (px, py), l, v in zip(pts, LAYERS, vals):
            s.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.2" fill="{c}"><title>{lab}, layer {l}: rank {v:,.0f}</title></circle>')
        s.append(f'<line x1="{lx}" x2="{lx + 22}" y1="38" y2="38" stroke="{c}" stroke-width="2.2"/><text class="lab" x="{lx + 28}" y="42">{lab}</text>')
        lx += 140
    s.append("</svg>")
    return "".join(s)


''' + s[b:]
s = s.replace("svg .lab {{ font:12.5px var(--body); fill:var(--ink); }}", "svg .lab {{ font:12.5px var(--body); fill:var(--ink); }} svg .title {{ font:600 15px var(--disp); fill:var(--ink); }} svg .axlab {{ font:500 12px var(--body); fill:var(--ink2); }}")
s = s.replace('q14 = qual["additive_s0"]["text"]["14"]', '''q14 = qual["additive_s0"]["text"]["14"]
Q14 = {m: med(m, "visual", "top", 14, lambda r: best(r["ranks"], "q:")) for m, _ in MODELS}
GT17 = {m: gt_ranks(m)[LAYERS.index(17)] for m, _ in MODELS}''')
s = s.replace("(median best rank 27–50 at layer 14)", "(median best rank {min(Q14.values()):.0f}–{max(Q14.values()):.0f} at layer 14)")
s = s.replace("additive's rank reaches 2 at layer 17 against 27 for LoRA", "additive's rank reaches {GT17[\"additive_s0\"]:.0f} at layer 17 against {GT17[\"lora_s0\"]:.0f} for LoRA")
s = s.replace('at median {q14["jlens_median_rank"]:.0f} of 151k, against {q14["logit_median_rank"]:,.0f}', 'at median {q14["jlens_median_rank"] + 1:.0f} of 151k, against {q14["logit_median_rank"] + 1:,.0f}')
s = s.replace("Spatial words rank 26–100 at layers 14–20 for salient and random cells alike", "Spatial words sit at similar ranks at layers 14–20 for salient and random cells alike")
open(P, "w").write(s)
print("patched")
