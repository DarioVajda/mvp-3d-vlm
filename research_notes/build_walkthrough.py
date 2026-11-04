"""Build docs/one_example_two_fusions.html: one training example through the additive and aligned cross-attention arms, step by step."""
import json

W = "/shared/workspace/povejmo/math/projects/mvp/.claude/worktrees/je-xattn-proposal"
E = json.load(open(f"{W}/research_notes/example.json"))
b = E["bias"]
q = E["example"]["question"].replace("These are frames of a video.\n", "").split("\n")
question, options = q[0], " ".join(x for x in q[1:] if x.startswith(("A.", "B.", "C.", "D.")))


def f(x, n=3):
    return f"{x:.{n}f}"


PAGE = f'''<title>One Example, Two Fusions</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<style>
/* Layout: a numbered walkthrough; each step is prose + one figure (image, equation block or table). Same tokens as the notebook. */
:root {{
  --bg: #f6f7f9; --panel: #ffffff; --ink: #14171c; --ink-2: #4a505b; --ink-3: #737a86;
  --rule: #dde1e7; --rule-2: #eceff3; --accent: #3d3a8f; --accent-soft: #e9e8f6;
  --add: #eb6834; --add-soft: #fdeee6; --xa: #e34948; --xa-soft: #fbe9e9; --llm: #2a78d6; --llm-soft: #e6f0fb;
  --good: #1d7a3a; --bad: #b3322f; --warn-bg: #fff6e0; --warn: #8a5a00;
  --f-display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --f-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --f-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #121418; --panel: #1a1d22; --ink: #eceef2; --ink-2: #b4bac5; --ink-3: #868d99;
  --rule: #2e333b; --rule-2: #23272d; --accent: #a9a6f0; --accent-soft: #26253f;
  --add: #f0824f; --add-soft: #3a2519; --xa: #ef7f7c; --xa-soft: #3a1f1f; --llm: #4a93ec; --llm-soft: #16283d;
  --good: #6cc68a; --bad: #ef7f7c; --warn-bg: #3a2f14; --warn: #f0c060; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #121418; --panel: #1a1d22; --ink: #eceef2; --ink-2: #b4bac5; --ink-3: #868d99;
  --rule: #2e333b; --rule-2: #23272d; --accent: #a9a6f0; --accent-soft: #26253f;
  --add: #f0824f; --add-soft: #3a2519; --xa: #ef7f7c; --xa-soft: #3a1f1f; --llm: #4a93ec; --llm-soft: #16283d;
  --good: #6cc68a; --bad: #ef7f7c; --warn-bg: #3a2f14; --warn: #f0c060; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--ink); font: 15px/1.6 var(--f-body); }}
.wrap {{ max-width: 1040px; margin: 0 auto; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 40px 80px; }}
.prose {{ max-width: 76ch; display: grid; gap: 12px; }}
h1, h2, h3 {{ font-family: var(--f-display); text-wrap: balance; line-height: 1.15; margin: 0; }}
h1 {{ font-size: clamp(30px, 5vw, 44px); font-weight: 600; letter-spacing: -0.01em; }}
h2 {{ font-size: 26px; font-weight: 600; }} h3 {{ font-size: 18px; font-weight: 600; }}
p, ul, ol {{ margin: 0; }} ul, ol {{ padding-left: 20px; display: grid; gap: 8px; }}
a {{ color: var(--accent); }} code {{ font-family: var(--f-mono); font-size: 0.88em; }}
.eyebrow {{ font: 500 12px/1.2 var(--f-mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-3); }}
header.top {{ display: grid; gap: 14px; padding-bottom: 28px; border-bottom: 1px solid var(--rule); }}
header.top .q {{ font-size: 18px; color: var(--ink-2); max-width: 70ch; }}
.meta {{ display: flex; flex-wrap: wrap; gap: 8px 18px; font: 13px var(--f-mono); color: var(--ink-3); }}
nav.toc {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }}
nav.toc a {{ font: 500 13px var(--f-body); text-decoration: none; color: var(--ink-2); padding: 4px 10px; border: 1px solid var(--rule); border-radius: 999px; background: var(--panel); }}
nav.toc a:hover, nav.toc a:focus-visible {{ color: var(--accent); border-color: var(--accent); outline: none; }}
section {{ display: grid; gap: 18px; padding-block: 40px 8px; scroll-margin-top: 16px; }}
.sec-head {{ display: grid; gap: 6px; }} .lede {{ color: var(--ink-2); }}
.step {{ font: 600 13px var(--f-mono); color: var(--accent); letter-spacing: 0.06em; }}
figure {{ margin: 0; display: grid; gap: 10px; background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: 18px 18px 14px; min-width: 0; }}
figcaption {{ font-size: 13.5px; color: var(--ink-2); max-width: 90ch; }}
figure img {{ max-width: 100%; height: auto; border-radius: 4px; display: block; }}
.scroll {{ overflow-x: auto; min-width: 0; }}
.two {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 440px), 1fr)); gap: 16px; }}
.math {{ font-family: var(--f-mono); font-size: 13.5px; line-height: 1.75; background: var(--bg); border: 1px solid var(--rule-2); border-radius: 6px; padding: 12px 16px; overflow-x: auto; white-space: pre; }}
.math i {{ font-style: normal; color: var(--ink-3); }}
.math b {{ font-weight: 500; color: var(--accent); }}
.arm {{ border-left: 4px solid var(--rule); padding-left: 14px; }}
.arm.add {{ border-color: var(--add); }} .arm.xa {{ border-color: var(--xa); }}
.tag {{ display: inline-block; font: 500 11px var(--f-mono); padding: 1px 7px; border-radius: 4px; margin-right: 6px; vertical-align: 2px; }}
.tag.add {{ background: var(--add-soft); color: var(--add); }} .tag.xa {{ background: var(--xa-soft); color: var(--xa); }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ text-align: left; padding: 7px 9px; border-bottom: 1px solid var(--rule-2); vertical-align: top; }}
thead th {{ font: 500 12px var(--f-mono); letter-spacing: 0.04em; text-transform: uppercase; color: var(--ink-3); border-bottom: 1px solid var(--rule); white-space: nowrap; }}
td.num, th.num {{ text-align: right; font-family: var(--f-mono); font-variant-numeric: tabular-nums; white-space: nowrap; }}
.pos {{ color: var(--good); }} .neg {{ color: var(--bad); }}
.key {{ display: grid; gap: 10px; background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: 20px 22px; }}
.key p {{ font-size: 16px; }}
.status {{ background: var(--warn-bg); color: var(--warn); border-radius: 6px; padding: 10px 14px; font-size: 14px; }}
svg.diag {{ max-width: 100%; height: auto; display: block; color: var(--ink); }}
svg.diag text {{ font-family: var(--f-body); font-size: 12px; fill: currentColor; }}
svg.diag .t2 {{ font-size: 11px; fill: var(--ink-2); }} svg.diag .mono {{ font-family: var(--f-mono); font-size: 10.5px; fill: var(--ink-3); }}
svg.diag .hd {{ font-family: var(--f-display); font-size: 14px; font-weight: 600; }}
svg.diag .box {{ fill: var(--panel); stroke: var(--rule); stroke-width: 1.2; }}
svg.diag .llm {{ fill: var(--llm-soft); stroke: var(--llm); stroke-width: 1.2; }}
svg.diag .add {{ fill: var(--add-soft); stroke: var(--add); stroke-width: 1.2; }}
svg.diag .xa {{ fill: var(--xa-soft); stroke: var(--xa); stroke-width: 1.2; }}
svg.diag .wire {{ stroke: currentColor; stroke-width: 1.3; fill: none; }}
svg.diag .ah {{ fill: currentColor; }}
svg.diag .cell {{ fill: var(--rule-2); stroke: var(--rule); stroke-width: 0.6; }}
svg.diag .cell.self {{ fill: var(--xa); }} svg.diag .cell.n1 {{ fill: var(--add); }} svg.diag .cell.n2 {{ fill: var(--add-soft); }}
footer {{ margin-top: 48px; padding-top: 18px; border-top: 1px solid var(--rule); font-size: 13px; color: var(--ink-3); }}
</style>

<div class="wrap">
<header class="top">
  <div class="eyebrow">MPhil ACS · MVP project · worked example · 2026-10-29</div>
  <h1>One Example, Two Fusions</h1>
  <p class="q">One real training question walked through the additive arm and the aligned cross-attention arm (RoPE₃ + locality bias), with the actual frames, the token bookkeeping, the equations as implemented in <code>fusion/modules.py</code>, and the numbers the runs logged. The aim: see exactly where the two arms differ and why the difference costs the attention arm 1.8 points.</p>
  <div class="meta"><span>Video: {E["video"].split("/")[-1]}</span><span>Qwen3.5-2B · D = 2048 · 16 heads of 128</span><span>VGGT-1B frozen</span><span>Runs: 2B_v4_additive_nm1e-5 (63.5) · 2B_xa_both (61.7)</span></div>
  <nav class="toc" aria-label="Steps">
    <a href="#s0">The example</a><a href="#s1">1 Tokens</a><a href="#s2">2 Geometry token</a><a href="#s3">3 Additive</a><a href="#s4">4 Aligned: projections</a><a href="#s5">5 RoPE₃</a><a href="#s6">6 Bias and softmax</a><a href="#s7">7 Where the mass goes</a><a href="#s8">8 Content term</a><a href="#s9">9 At 32 frames</a><a href="#s10">10 Opening</a><a href="#why">Why additive won</a>
  </nav>
</header>

<section id="s0">
  <div class="sec-head"><div class="step">THE EXAMPLE</div><h2>A relative-direction question about a ScanNet office</h2></div>
  <figure>
    <img src="{E["sheet"]}" alt="Eight uniformly sampled frames of the training video with the 20 by 15 token grid drawn on each; the cell at row 4, column 7 of frame 2 is outlined in red.">
    <figcaption>The 8 training frames, sampled uniformly from the {E["n_frames_total"]:,}-frame video at raw indices {E["frame_idx"]}, each with Qwen's 20×15 grid of merged visual tokens. The red cell in frame t = 2 is the token we follow. This video has {E["n_questions"]} training questions; one micro-batch takes up to 8 of them and shares the frames and the geometry.</figcaption>
  </figure>
  <div class="key">
    <p><strong>Question:</strong> {question}</p>
    <p><strong>Options:</strong> {options} &nbsp;·&nbsp; <strong>Answer:</strong> {E["example"]["answer"]} &nbsp;·&nbsp; <strong>Task:</strong> {E["example"]["task"]}</p>
    <p>To answer it the model must locate three objects (keyboard, bookshelf, trash bin), recover their 3-D positions relative to each other, and reason about left/right from a viewpoint. The geometry that could help lives in VGGT's patch tokens; the question is how it reaches the LLM tokens that represent the keyboard, the bookshelf and the bin.</p>
  </div>
</section>

<section id="s1">
  <div class="sec-head"><div class="step">STEP 1</div><h2>From pixels to 2,400 visual tokens, and the index that pairs them with geometry</h2></div>
  <div class="two">
    <figure>
      <img src="{E["zoom"]}" alt="Frame 2 at full size with the grid; the followed cell is outlined in red and its four neighbours in yellow.">
      <figcaption>Frame t = 2. Qwen's vision encoder cuts the 640×480 frame into 40×30 patches of 16 px and merges each 2×2 block into one visual token of 32×32 px: 20 columns × 15 rows = 300 tokens. The red cell is (r = 4, c = 7); yellow are its four Manhattan-distance-1 neighbours.</figcaption>
    </figure>
    <figure>
<div class="math"><b>Sequence of one question</b> (frames as separate images, temporal order):
  system · user text · [⟨vision_start⟩ 300 tokens ⟨vision_end⟩] × 8 · question · answer

<b>Visual-token index</b> (row-major inside a frame, frames concatenated):
  n = 300·t + 20·r + c           0 ≤ n &lt; 2400
  our cell: n = 300·2 + 20·4 + 7 = <b>687</b>
  inverse:  t = n // 300,  r = (n % 300) // 20,  c = n % 20

<b>VGGT side</b> (same frame resized by 14/16 = 7/8 to 560×420):
  40×30 patches of 14 px → 2×2 windows → the same 20×15 grid
  window (t, r, c) covers pixels [32c, 32c+32) × [32r, 32r+32) of the frame
  geometry token index = the same n = 687
  <i>(frozen VGGT; its 5 camera/register tokens per frame are dropped)</i></div>
      <figcaption>Both streams read the same 20×15 grid in the same order, so "token n" and "geometry token n" are the same 32×32 px of the same frame. The additive arm relies on this pairing with no position logic at all; the aligned arm uses it to build (t, r, c) for every token from <code>image_grid_thw</code>.</figcaption>
    </figure>
  </div>
</section>

<section id="s2">
  <div class="sec-head"><div class="step">STEP 2</div><h2>The two vectors at token 687: the LLM's hidden state and the geometry token</h2></div>
  <div class="two">
    <figure>
<div class="math"><b>h[687]</b>  ∈ ℝ²⁰⁴⁸     hidden state after decoder layer k (k = 0, 1, 2)
  what Qwen knows about this 32×32 px: appearance, texture,
  object identity, plus context from the rest of the sequence
  (LoRA r64 on every linear layer is training this path)

<b>G_k[687]</b> ∈ ℝ²⁰⁴⁸    geometry token from VGGT layer L_k ∈ {{11, 17, 23}}
  p_a ∈ ℝ²⁰⁴⁸  the four VGGT patch tokens of the window, a = TL, TR, BL, BR
  w  = [RMSNorm(p_TL); RMSNorm(p_TR); RMSNorm(p_BL); RMSNorm(p_BR)]  ∈ ℝ⁸¹⁹²
  G  = W₂ · GELU(W₁ w + b₁) + b₂       W₁: 8192→4096, W₂: 4096→2048
  <i>one merger per injection point, 41.9 M parameters each, identical in both arms</i></div>
      <figcaption>Everything up to here is shared. The merger is trained by the answer loss in both arms. One difference already: the additive arm zero-initialises W₂ (so G = 0 at step 0); the aligned arm cannot, because G is also its keys and values, so it zero-initialises the attention's output projection instead.</figcaption>
    </figure>
    <figure>
      <svg class="diag" viewBox="0 0 460 250" role="img" aria-label="Diagram: four VGGT patch tokens of one window are concatenated and passed through an MLP to give one geometry token G, which is paired with the LLM hidden state h of the same cell.">
        <defs><marker id="a1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path class="ah" d="M0,0 L10,5 L0,10 z"/></marker></defs>
        <text x="16" y="24" class="hd">VGGT window (t=2, r=4, c=7)</text>
        <rect class="cell" x="16" y="40" width="40" height="40"/><rect class="cell" x="58" y="40" width="40" height="40"/><rect class="cell" x="16" y="82" width="40" height="40"/><rect class="cell" x="58" y="82" width="40" height="40"/>
        <text x="36" y="65" text-anchor="middle" class="mono">TL</text><text x="78" y="65" text-anchor="middle" class="mono">TR</text><text x="36" y="107" text-anchor="middle" class="mono">BL</text><text x="78" y="107" text-anchor="middle" class="mono">BR</text>
        <text x="16" y="142" class="t2">4 × 2048 = 8192</text>
        <path class="wire" d="M104,81 H150" marker-end="url(#a1)"/>
        <rect class="box" x="154" y="56" width="110" height="50" rx="6"/><text x="209" y="77" text-anchor="middle">merger MLP</text><text x="209" y="95" text-anchor="middle" class="mono">8192→4096→2048</text>
        <path class="wire" d="M264,81 H310" marker-end="url(#a1)"/>
        <rect class="add" x="314" y="56" width="120" height="50" rx="6"/><text x="374" y="77" text-anchor="middle" font-weight="600">G[687]</text><text x="374" y="95" text-anchor="middle" class="mono">ℝ²⁰⁴⁸</text>
        <text x="16" y="186" class="hd">LLM, after decoder layer k</text>
        <rect class="llm" x="314" y="196" width="120" height="44" rx="6"/><text x="374" y="214" text-anchor="middle" font-weight="600">h[687]</text><text x="374" y="231" text-anchor="middle" class="mono">ℝ²⁰⁴⁸</text>
        <text x="16" y="212" class="t2">visual token 687 of the question's</text><text x="16" y="228" class="t2">sequence, shared frames, own text</text>
        <path class="wire" d="M374,110 V190" stroke-dasharray="3 3"/><text x="382" y="152" class="t2">same cell</text>
      </svg>
      <figcaption>The pairing h[687] ↔ G[687] is the whole of the additive arm's "alignment": index equality. The aligned arm replaces the dashed line by a softmax over all 2,400 geometry tokens.</figcaption>
    </figure>
  </div>
</section>

<section id="s3">
  <div class="sec-head"><div class="step">STEP 3 · <span class="tag add">ADDITIVE</span></div><h2>The additive update: one line</h2></div>
  <figure class="arm add">
<div class="math">h[n] ← h[n] + G_k[n]        for every visual token n, at each of the three hooks (k = 0, 1, 2)
text tokens: unchanged

<b>What token 687 receives:</b> exactly the geometry of its own 32×32 px, in frame 2, nothing else.
<b>Logged size</b> (2B_v4_additive, end of training, ‖Σ added‖ / ‖h_visual‖):  0.21 (k=0)   0.21 (k=1)   0.15 (k=2)
<b>Trainable path:</b> merger only (W₁, W₂, RMSNorm): 41.9 M per point, lr 1e-5, W₂ zero at init</div>
    <figcaption>There is no mixing across tokens, no normalisation of G, no projection after the merger. The next decoder layer's own attention and FFW do all the mixing, on a hidden state that now carries a sharp per-cell depth/orientation signal. For the question above, the tokens on the keyboard, the bookshelf and the bin each get their own geometry, unblurred; the decoder's self-attention then relates them.</figcaption>
  </figure>
</section>

<section id="s4">
  <div class="sec-head"><div class="step">STEP 4 · <span class="tag xa">ALIGNED</span></div><h2>The aligned update, part 1: normalise and project</h2></div>
  <figure class="arm xa">
<div class="math"><b>Inputs at hook k</b>:  y = h[visual] ∈ ℝ^(B×2400×2048)  (B questions of the video),   G ∈ ℝ^(2400×2048) (shared)

q = W_q · LN_q(y)            ∈ ℝ^(B×2400×2048)  → 16 heads × 128
k = W_k · LN_kv(G)           ∈ ℝ^(2400×2048)    → 16 heads × 128   <i>(shared by the B questions)</i>
v = W_v · LN_kv(G)           ∈ ℝ^(2400×2048)    → 16 heads × 128
<i>W_q, W_k, W_v: 2048×2048, no bias, default init;  W_o: 2048×2048, zero at init</i>

<b>At token 687, head h:</b>   q₆₈₇ ∈ ℝ¹²⁸ is a learned view of what the LLM sees there;
                           k_m, v_m ∈ ℝ¹²⁸ are learned views of the geometry of every cell m of the video.</div>
    <figcaption>Two differences from additive are already in place: G is LayerNormed (its scale is discarded and must be re-learned by W_v and W_o), and the geometry reaches the residual stream only through W_v then W_o, two more 2048×2048 maps trained at lr 1e-5, instead of directly.</figcaption>
  </figure>
</section>

<section id="s5">
  <div class="sec-head"><div class="step">STEP 5 · <span class="tag xa">ALIGNED</span></div><h2>The aligned update, part 2: RoPE₃ as implemented</h2></div>
  <figure class="arm xa">
<div class="math"><b>Positions</b>:   pos[n] = (t, r, c) from image_grid_thw;   t̃ = t · (8−1)/(T−1)   <i>(T frames at this forward; at training T = 8 so t̃ = t)</i>

<b>Rotary pairs</b> (of the 64 pairs in a 128-d head; the remaining 32 pairs carry no position):
  pairs  0–9    rotated by t̃,  frequencies ω_j = 100^(−j/10),  j = 0..9      <i>(1, 0.63, 0.40, …, 0.0126)</i>
  pairs 10–20   rotated by r,   frequencies ω_j = 100^(−j/11),  j = 0..10     <i>(1, 0.66, 0.43, …, 0.0152)</i>
  pairs 21–31   rotated by c,   frequencies ω_j = 100^(−j/11)
  layout: NeoX half-split, pair j = (x_j, x_{{j+32}})

<b>Rotation</b> (applied identically to q and k):  for pair j on axis a with coordinate p_a:
  (x_j, x_{{j+32}}) ← (x_j cos(ω_j p_a) − x_{{j+32}} sin(ω_j p_a),  x_{{j+32}} cos(ω_j p_a) + x_j sin(ω_j p_a))

<b>Consequence</b>:  q₆₈₇ · k_m depends on (Δt̃, Δr, Δc) = pos[687] − pos[m] only, not on absolute position:
  same cell, same frame  Δ = (0, 0, 0)      → no rotation between q and k
  the cell to the right  Δ = (0, 0, −1)     → pairs 21–31 rotated by −ω_j, i.e. by 1 rad on the fastest pair, 0.015 on the slowest
  same cell, frame 5     Δ = (−3, 0, 0)     → pairs 0–9 rotated by 3 rad on the fastest pair

<b>Phase budget over the grid</b> (base 100 per axis, this implementation):
  rows,   |Δr| ≤ 14:  fastest pair {f(E["rope"]["r_base100"][0], 1)} rad, slowest {f(E["rope"]["r_base100"][1], 2)} rad
  columns,|Δc| ≤ 19:  fastest pair 19.0 rad,  slowest {f(E["rope"]["c_base100_first"][1], 2)} rad
  <i>(the λ/μ report's worry that the column axis is "nearly blind" applied to Qwen2-VL's base 10⁶ layout; it does not apply here)</i></div>
    <figcaption>RoPE₃ gives the content logits a way to express "how far and in which direction is key m from me". It adds no parameters and no logit by itself: with random W_q, W_k the rotated dot products are still noise around zero. It only matters once W_q and W_k have learned something, which is step 10's story.</figcaption>
  </figure>
</section>

<section id="s6">
  <div class="sec-head"><div class="step">STEP 6 · <span class="tag xa">ALIGNED</span></div><h2>The aligned update, part 3: locality bias, softmax, value mixture, output</h2></div>
  <figure class="arm xa">
<div class="math"><b>Logit</b> of query 687 (at (2, 4, 7)) against key m at (t′, r′, c′), head h:
  s_m = q₆₈₇ · k_m / √128  −  λ_h · (|4 − r′| + |7 − c′|)  −  μ_h · [t′ ≠ 2]

<b>Attention</b>:   a_m = exp(s_m) / Σ_{{m′=1}}^{{2400}} exp(s_{{m′}})

<b>Value mixture and output</b>:
  z₆₈₇ = Σ_m a_m v_m                       <i>(per head, concatenated to 2048)</i>
  Δh[687] = W_o z₆₈₇
  h[687] ← h[687] + Δh[687]

<b>Bias at the followed cell, learned values (k = 0, means over heads): λ = 1.71, μ = 3.10</b>
  key                                   bias      e^bias    count
  own cell (2, 4, 7)                     0        1.000       1
  neighbours at distance 1, same frame  −1.71     0.181       4
  distance 2, same frame                −3.42     0.033       8
  distance 3, same frame                −5.13     0.006      12
  same cell (r=4, c=7) in another frame −3.10     0.045       7
  distance 1 in another frame           −4.81     0.008      28
  everything else                       &lt; −5.1   &lt; 0.006   2340</div>
    <figcaption>With content logits near zero the softmax is just these e^bias values normalised. The additive arm is the limit λ, μ → ∞: a_m = [m = 687]. The learned λ, μ sit far from that limit.</figcaption>
  </figure>
</section>

<section id="s7">
  <div class="sec-head"><div class="step">STEP 7</div><h2>Where token 687's attention mass actually goes</h2></div>
  <div class="two">
    <figure>
      <div class="scroll"><table>
        <thead><tr><th>Share of a₆₈₇ (bias alone, 8 frames)</th><th class="num">init λ=1.5, μ=3.0</th><th class="num">learned λ=1.71, μ=3.10</th><th class="num">draft λ=μ=4</th></tr></thead>
        <tbody>
          <tr><td>own cell</td><td class="num">{f(b["init_T8"]["own"])}</td><td class="num"><strong>{f(b["learned_T8"]["own"])}</strong></td><td class="num">{f(b["proposal_T8"]["own"])}</td></tr>
          <tr><td>4 neighbours at distance 1</td><td class="num">{f(b["init_T8"]["dist1_total"])}</td><td class="num">{f(b["learned_T8"]["dist1_total"])}</td><td class="num">{f(b["proposal_T8"]["dist1_total"])}</td></tr>
          <tr><td>rest of own frame</td><td class="num">{f(b["init_T8"]["frame"] - b["init_T8"]["own"] - b["init_T8"]["dist1_total"])}</td><td class="num">{f(b["learned_T8"]["frame"] - b["learned_T8"]["own"] - b["learned_T8"]["dist1_total"])}</td><td class="num">{f(b["proposal_T8"]["frame"] - b["proposal_T8"]["own"] - b["proposal_T8"]["dist1_total"])}</td></tr>
          <tr><td>same (r, c) in the 7 other frames</td><td class="num">{f(b["init_T8"]["same_cell_other_frames"])}</td><td class="num">{f(b["learned_T8"]["same_cell_other_frames"])}</td><td class="num">{f(b["proposal_T8"]["same_cell_other_frames"])}</td></tr>
          <tr><td>other cells of other frames</td><td class="num">{f(1 - b["init_T8"]["frame"] - b["init_T8"]["same_cell_other_frames"])}</td><td class="num">{f(1 - b["learned_T8"]["frame"] - b["learned_T8"]["same_cell_other_frames"])}</td><td class="num">{f(1 - b["proposal_T8"]["frame"] - b["proposal_T8"]["same_cell_other_frames"])}</td></tr>
          <tr><td><strong>own frame, total</strong></td><td class="num">{f(b["init_T8"]["frame"])}</td><td class="num">{f(b["learned_T8"]["frame"])}</td><td class="num">{f(b["proposal_T8"]["frame"])}</td></tr>
          <tr><td>entropy of the row (nats)</td><td class="num">{f(b["init_T8"]["entropy"], 2)}</td><td class="num">{f(b["learned_T8"]["entropy"], 2)}</td><td class="num">{f(b["proposal_T8"]["entropy"], 2)}</td></tr>
        </tbody>
      </table></div>
      <figcaption>Computed on the real 20×15 grid for the cell (4, 7). Logged on dev at the end of training (layer 0, 2B_xa_both, content included): self mass 0.369, own-frame mass 0.746, entropy 2.80, almost identical to the bias-alone column. See step 8.</figcaption>
    </figure>
    <figure>
      <img src="{E["zoom5"]}" alt="Frame 5 with the same grid cell outlined in red: it shows a different part of the room than the same cell in frame 2.">
      <figcaption>The same cell (r = 4, c = 7) in frame t = 5. The camera has moved, so this is a different surface at a different depth from the cell in frame 2, yet the bias gives it 7 × 0.016 = 0.115 of token 687's attention (0.045 e^bias each against 1.0 for the own cell) purely because it has the same grid coordinates. Grid position is not 3-D position; the bias and RoPE₃ only know grid position.</figcaption>
    </figure>
  </div>
  <div class="prose">
    <p>So at the end of training token 687 receives, through W_o, a vector that is <strong>37% its own geometry, 26% its four neighbours' geometry, 11% the geometry of unrelated surfaces that happen to sit at (4, 7) in other frames, and 26% scattered over the rest of the video</strong>. The additive arm gives it 100% its own geometry. For "is the trash bin left or right of the bookshelf", what the bin token needs is the bin's position, not a 26% blur with the floor around it and an 11% dose of some other wall.</p>
  </div>
</section>

<section id="s8">
  <div class="sec-head"><div class="step">STEP 8</div><h2>Did the content term q·k reshape the prior? Barely.</h2></div>
  <figure>
    <div class="scroll"><table>
      <thead><tr><th>Layer 0, end of training</th><th class="num">bias alone (computed)</th><th class="num">bias + content (logged on dev)</th><th class="num">difference</th></tr></thead>
      <tbody>
        <tr><td>self mass</td><td class="num">{f(b["learned_T8"]["own"])}</td><td class="num">0.369</td><td class="num">+0.003</td></tr>
        <tr><td>own-frame mass</td><td class="num">{f(b["learned_T8"]["frame"])}</td><td class="num">0.746</td><td class="num">−0.014</td></tr>
        <tr><td>entropy (nats)</td><td class="num">{f(b["learned_T8"]["entropy"], 2)}</td><td class="num">2.80</td><td class="num">−0.22</td></tr>
      </tbody>
    </table></div>
    <figcaption>After 7,625 steps the attention pattern is the positional prior with a small content-driven sharpening. The learned W_q, W_k did not find a content-based correspondence (which would show as self mass well above the prior, or as mass moving to specific other-frame cells with lower own-frame mass). Two readings: the content logits stayed small because W_o opened slowly and their gradient with it (step 10), or Qwen's token features and VGGT's features are not linearly matchable at this stage. Either way, in this run the aligned arm is, to a good approximation, <em>additive geometry passed through a fixed spatial blur plus two extra linear maps</em>.</figcaption>
  </figure>
</section>

<section id="s9">
  <div class="sec-head"><div class="step">STEP 9</div><h2>At evaluation, 32 frames: the blur gets worse</h2></div>
  <figure>
    <div class="scroll"><table>
      <thead><tr><th>Share of a₆₈₇ (bias alone, learned λ, μ)</th><th class="num">training, T = 8</th><th class="num">evaluation, T = 32</th><th class="num">T = 32 with μ + ln(31/7)</th></tr></thead>
      <tbody>
        <tr><td>own cell</td><td class="num">{f(b["learned_T8"]["own"])}</td><td class="num neg">{f(b["learned_T32"]["own"])}</td><td class="num">{f(b["mu_corrected_T32"]["own"])}</td></tr>
        <tr><td>own frame</td><td class="num">{f(b["learned_T8"]["frame"])}</td><td class="num neg">{f(b["learned_T32"]["frame"])}</td><td class="num">{f(b["mu_corrected_T32"]["frame"])}</td></tr>
        <tr><td>same (r, c) in other frames</td><td class="num">{f(b["learned_T8"]["same_cell_other_frames"])}</td><td class="num neg">{f(b["learned_T32"]["same_cell_other_frames"])}</td><td class="num">{f(b["mu_corrected_T32"]["same_cell_other_frames"])}</td></tr>
        <tr><td>entropy (nats)</td><td class="num">{f(b["learned_T8"]["entropy"], 2)}</td><td class="num">{f(b["learned_T32"]["entropy"], 2)}</td><td class="num">{f(b["mu_corrected_T32"]["entropy"], 2)}</td></tr>
      </tbody>
    </table></div>
    <figcaption>The per-key penalty μ is fixed but there are 31 other frames instead of 7, so the other frames' total share rises from 0.24 to 0.58 and the own cell falls from 0.37 to 0.20. The model was trained on the left column and evaluated on the middle one. Adding ln((T−1)/7) to μ at evaluation restores the training shares exactly (right column). This is also the simplest explanation of why dev (8 frames, in-domain) ranked the bias-only run first while VSI (32 frames) ranked it fourth. It costs nothing to test: re-run the eval on the saved checkpoint.</figcaption>
  </figure>
</section>

<section id="s10">
  <div class="sec-head"><div class="step">STEP 10</div><h2>How each arm opens, and why the no-bias runs never localised</h2></div>
  <div class="two">
    <figure class="arm add">
<div class="math"><span class="tag add">ADDITIVE</span> <b>at step 0</b>:  W₂ = 0  ⇒  G = b₂ = 0  ⇒  h unchanged
<b>gradient to W₂</b>:  ∂L/∂W₂ = Σ_n (∂L/∂h[n]) · GELU(W₁ w_n)ᵀ
  non-zero from the first step, and already <i>cell-specific</i>:
  the gradient at token n pairs with that token's own window features.
<b>injection ratio</b> reached 0.13 by the end of epoch 1 in v3, 0.21 at the end of v4.</div>
      <figcaption>One zero-initialised map between a cell-specific feature and a cell-specific gradient: the fastest possible way to learn "add this cell's geometry here".</figcaption>
    </figure>
    <figure class="arm xa">
<div class="math"><span class="tag xa">ALIGNED</span> <b>at step 0</b>:  W_o = 0  ⇒  Δh = 0
<b>gradients</b>:  ∂L/∂W_o = Σ_n (∂L/∂h[n]) · z_nᵀ          non-zero
             ∂L/∂W_q, ∂L/∂W_k, ∂L/∂W_v, ∂L/∂λ, ∂L/∂μ ∝ W_o   = 0 at step 0, tiny after
<b>content logits at init</b> (LN inputs, default Linear init, d = 128):
  var(q_i) ≈ 1/3  ⇒  std(q·k/√128) ≈ 0.33   ⇒ softmax ≈ uniform over 2400 keys

<b>Without the bias</b>:  z_n ≈ (1/2400) Σ_m v_m = the same vector for every token n
  ⇒ Δh is a per-video constant added to every visual token: carries no per-cell information
  ⇒ ∂L/∂W_o has no reason to grow;  ∂L/∂(q·k) = a_m (g_m − ḡ) is scaled by a_m = 1/2400
  ⇒ a fixed point: logged entropy 7.56–7.71 of 7.78 for the whole run (both no-position and RoPE₃-only)

<b>With the bias</b>:  z_n ≈ 0.3·v_n + 0.27·neighbours + …  from step 0
  ⇒ Δh is cell-specific immediately, W_o gets the additive arm's kind of gradient and opens
  ⇒ then q·k receives gradient; RoPE₃ becomes usable (+2.5 points over bias alone)</div>
      <figcaption>The attention could not bootstrap its own alignment: localisation is only rewarded through W_o, W_o only opens if the output is informative, and the output is only informative if the attention is already localised. The bias supplies the first link. This is why "RoPE₃ only" (58.3) equals "no position" (57.5) equals LoRA (57.8), and why the λ/μ research found every working method starts with an explicit prior.</figcaption>
    </figure>
  </div>
</section>

<section id="why">
  <div class="sec-head"><div class="step">SYNTHESIS</div><h2>Why additive fusion outperformed the attention arms</h2></div>
  <div class="prose">
    <ol>
      <li><strong>Exact correspondence beats learned correspondence when the correspondence is already known.</strong> The grids are aligned by construction, so "token n ↔ geometry n" is the right answer. Additive hard-codes it. The aligned arm has to express it through a prior it was told (λ, μ) and a content term it never learned; the old resampler arms could not express it at all (256 location-free latents, −8 points).</li>
      <li><strong>The soft alignment is a spatial blur, and blur hurts the tasks that need sharp per-token geometry.</strong> At the learned λ, 63% of each token's geometry comes from other cells. Relative direction, route planning and absolute distance, where aligned trails additive by 4–7 points, need the geometry <em>of the object's own tokens</em>. Relative distance, where aligned wins by 2, compares two regions and tolerates (or benefits from) neighbourhood averaging.</li>
      <li><strong>Grid position is not 3-D position.</strong> 11% of the mass goes to the same (r, c) in other frames, which after camera motion is an unrelated surface. Neither the bias nor RoPE₃ can know this; only a content-based match could, and it was not learned. Additive never looks at other frames, so it never pays this cost.</li>
      <li><strong>The optimiser agreed.</strong> λ moved up (1.5 → 1.71 at two of three points) and self mass rose through training: the gradient pulled toward the one-hot limit, i.e. toward the additive arm. At lr 1e-3 the scalars could only travel about 0.2; a sharper start or per-head spread tests whether the arm converges to additive's score or passes it.</li>
      <li><strong>Train/test mismatch in frame count.</strong> With fixed μ the other frames' share rises from 0.24 to 0.58 at 32 frames, so the model is evaluated under a wider blur than it was trained with. Additive has no such dependence. The μ correction (free re-eval) measures how much of the 1.8-point gap this is.</li>
      <li><strong>A longer, slower learning path.</strong> Geometry reaches the stream through LN → W_v → attention → W_o instead of directly, two more 2048×2048 maps at lr 1e-5 and a zero-initialised W_o; aligned dev accuracy trails additive by 3 points at epoch 1 (55.1 vs 58.6) and is still 0.4 behind at epoch 5. Both arms were still improving.</li>
      <li><strong>What attention could add, it did not get to use.</strong> The one capability additive lacks is reading geometry from <em>another</em> cell when that is the right thing to do (the same object seen in another frame; VGGT's camera tokens). That requires content-based matching, which in this run did not develop. The arm is not yet a test of that capability; it is a test of a positional prior around the additive solution, and the prior lost to the exact version by 1.8 points.</li>
    </ol>
    <p class="status">Nothing on this page was run on a GPU. The attention shares are computed from the bias formula at the logged λ, μ; the "bias + content" column is the training log's dev-time statistic; frames are decoded on the CPU. A per-token activation study (which tokens, which heads, which other-frame cells) would turn step 8 from an inference into a measurement, and was proposed earlier.</p>
  </div>
</section>

<footer>Code: <code>src/mvp/fusion/modules.py</code> (AlignedXAttn, AdditiveMerger) and the fusion hook in <code>src/mvp/fusion/model.py</code>; logs of <code>results/train/2B_v4_additive_nm1e-5</code> and <code>results/train/2B_xa_both</code>. Companion pages: <a href="https://claude.ai/artifact/UPAfWs1sCJE2XYkJN8AKfr">Aligned Attention Results</a>, <a href="https://claude.ai/artifact/QrsAZhjPoWfrrLGijrEgWC">JE-xattn Proposal</a>, <a href="https://claude.ai/artifact/4FjcTJT64mjCgab7h95Xcz">Geometry Fusion Notebook</a>.</footer>
</div>
'''
open(f"{W}/docs/one_example_two_fusions.html", "w").write(PAGE)
print("written", len(PAGE))
