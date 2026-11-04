"""Build docs/v4_results.html: charts + interpretation of run v4 and the aligned cross-attention runs."""
import json

W = "/shared/workspace/povejmo/math/projects/mvp/.claude/worktrees/je-xattn-proposal"
data = json.load(open(f"{W}/research_notes/v4_data.json"))
DATA = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")

PAGE = r'''<title>Aligned Attention Results</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<style>
/* Layout: reading column for prose, full 1040px for charts; one chart per claim, interpretation beside it. Same tokens as the notebook. */
:root {
  --bg: #f6f7f9; --panel: #ffffff; --ink: #14171c; --ink-2: #4a505b; --ink-3: #737a86;
  --rule: #dde1e7; --rule-2: #eceff3; --accent: #3d3a8f; --accent-soft: #e9e8f6;
  --c-lora: #2a78d6; --c-add: #eb6834; --c-add2: #f0a57c; --c-x: #1baf7a; --c-x2: #7fd3b0;
  --c-n: #9aa1ab; --c-r: #eda100; --c-b: #6250d6; --c-both: #e34948;
  --good: #1d7a3a; --bad: #b3322f; --good-bg: #e6f3ea; --bad-bg: #f9e7e6; --warn-bg: #fff6e0; --warn: #8a5a00;
  --f-display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --f-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --f-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #121418; --panel: #1a1d22; --ink: #eceef2; --ink-2: #b4bac5; --ink-3: #868d99;
  --rule: #2e333b; --rule-2: #23272d; --accent: #a9a6f0; --accent-soft: #26253f;
  --c-lora: #3987e5; --c-add: #f0824f; --c-add2: #b8683f; --c-x: #2fc48c; --c-x2: #4e8f72;
  --c-n: #7c8491; --c-r: #d9a21b; --c-b: #9085e9; --c-both: #ef7f7c;
  --good: #6cc68a; --bad: #ef7f7c; --good-bg: #17291d; --bad-bg: #331b1b; --warn-bg: #3a2f14; --warn: #f0c060; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #121418; --panel: #1a1d22; --ink: #eceef2; --ink-2: #b4bac5; --ink-3: #868d99;
  --rule: #2e333b; --rule-2: #23272d; --accent: #a9a6f0; --accent-soft: #26253f;
  --c-lora: #3987e5; --c-add: #f0824f; --c-add2: #b8683f; --c-x: #2fc48c; --c-x2: #4e8f72;
  --c-n: #7c8491; --c-r: #d9a21b; --c-b: #9085e9; --c-both: #ef7f7c;
  --good: #6cc68a; --bad: #ef7f7c; --good-bg: #17291d; --bad-bg: #331b1b; --warn-bg: #3a2f14; --warn: #f0c060; color-scheme: dark; }
body { background: var(--bg); color: var(--ink); font: 15px/1.6 var(--f-body); }
.wrap { max-width: 1040px; margin: 0 auto; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 40px 80px; }
.prose { max-width: 74ch; display: grid; gap: 12px; }
h1, h2, h3 { font-family: var(--f-display); text-wrap: balance; line-height: 1.15; margin: 0; }
h1 { font-size: clamp(30px, 5vw, 44px); font-weight: 600; letter-spacing: -0.01em; }
h2 { font-size: 26px; font-weight: 600; } h3 { font-size: 18px; font-weight: 600; }
p, ul { margin: 0; } ul { padding-left: 20px; display: grid; gap: 8px; }
a { color: var(--accent); } code { font-family: var(--f-mono); font-size: 0.88em; }
.eyebrow { font: 500 12px/1.2 var(--f-mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-3); }
header.top { display: grid; gap: 14px; padding-bottom: 28px; border-bottom: 1px solid var(--rule); }
header.top .q { font-size: 18px; color: var(--ink-2); max-width: 70ch; }
.meta { display: flex; flex-wrap: wrap; gap: 8px 18px; font: 13px var(--f-mono); color: var(--ink-3); }
nav.toc { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
nav.toc a { font: 500 13px var(--f-body); text-decoration: none; color: var(--ink-2); padding: 4px 10px; border: 1px solid var(--rule); border-radius: 999px; background: var(--panel); }
nav.toc a:hover, nav.toc a:focus-visible { color: var(--accent); border-color: var(--accent); outline: none; }
section { display: grid; gap: 18px; padding-block: 40px 8px; scroll-margin-top: 16px; }
.sec-head { display: grid; gap: 6px; } .lede { color: var(--ink-2); }
.verdict { display: grid; gap: 12px; background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: 22px 24px; }
.verdict p { font-size: 16px; }
figure { margin: 0; display: grid; gap: 10px; background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: 18px 18px 14px; min-width: 0; }
figcaption { font-size: 13.5px; color: var(--ink-2); max-width: 90ch; }
.scroll { overflow-x: auto; min-width: 0; }
.two { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 440px), 1fr)); gap: 16px; }
.fig-top { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 10px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 13px; color: var(--ink-2); }
.chip { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; user-select: none; }
.chip.off { opacity: 0.35; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex: none; }
.chart text { font: 11.5px var(--f-mono); fill: var(--ink-3); }
.chart .grid { stroke: var(--rule-2); stroke-width: 1; } .chart .axis { stroke: var(--rule); stroke-width: 1; }
.chart .line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.chart .ref { stroke: var(--ink-3); stroke-dasharray: 4 4; stroke-width: 1.2; }
.chart .lab { font: 500 12px var(--f-body); fill: var(--ink); }
.chart .cross { stroke: var(--ink-3); stroke-width: 1; opacity: 0; } .chart.hover .cross { opacity: 0.6; }
.tip { position: absolute; pointer-events: none; background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 8px 10px; font-size: 12.5px; box-shadow: 0 4px 14px rgb(0 0 0 / 0.12); display: grid; gap: 3px; min-width: 170px; z-index: 2; }
.tip b { font: 600 12px var(--f-mono); color: var(--ink-2); }
.tip .row { display: flex; align-items: center; gap: 8px; justify-content: space-between; }
.tip .row span:first-child { display: inline-flex; align-items: center; gap: 6px; color: var(--ink-2); }
.tip .row span:last-child { font-family: var(--f-mono); font-variant-numeric: tabular-nums; color: var(--ink); }
.chart-box { position: relative; min-width: 0; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { text-align: left; padding: 7px 9px; border-bottom: 1px solid var(--rule-2); vertical-align: top; }
thead th { font: 500 12px var(--f-mono); letter-spacing: 0.04em; text-transform: uppercase; color: var(--ink-3); border-bottom: 1px solid var(--rule); white-space: nowrap; }
td.num, th.num { text-align: right; font-family: var(--f-mono); font-variant-numeric: tabular-nums; white-space: nowrap; }
td.best { font-weight: 600; }
.pos { color: var(--good); } .neg { color: var(--bad); }
td.heat { text-align: right; font-family: var(--f-mono); font-variant-numeric: tabular-nums; }
.status { background: var(--warn-bg); color: var(--warn); border-radius: 6px; padding: 10px 14px; font-size: 14px; }
.seg { display: inline-flex; border: 1px solid var(--rule); border-radius: 6px; overflow: hidden; width: fit-content; }
.seg button { font: 500 13px var(--f-body); color: var(--ink-2); background: transparent; border: 0; padding: 6px 12px; cursor: pointer; }
.seg button + button { border-left: 1px solid var(--rule); }
.seg button[aria-pressed="true"] { background: var(--accent-soft); color: var(--accent); }
.seg button:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 8px 16px; margin: 0; font-size: 14px; }
dt { font-weight: 600; } dd { margin: 0; color: var(--ink-2); }
@media (max-width: 560px) { dl { grid-template-columns: 1fr; } }
footer { margin-top: 48px; padding-top: 18px; border-top: 1px solid var(--rule); font-size: 13px; color: var(--ink-3); }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>

<div class="wrap">
<header class="top">
  <div class="eyebrow">MPhil ACS · MVP project · results analysis · 2026-10-29</div>
  <h1>Aligned Attention Results</h1>
  <p class="q">Run v4 (appearance-order data added, 5 epochs, half-epoch dev evals) and the four aligned cross-attention variants from the design proposal, on Qwen3.5-2B. Nine runs, one seed, full VSI-Bench at 32 frames on each run's best-dev checkpoint.</p>
  <div class="meta"><span>Base: Qwen3.5-2B</span><span>Geometry: VGGT-1B, frozen</span><span>Train: VLM-3R ScanNet cap 6k + VSI-590K appearance order</span><span>LoRA lr 1e-4 · new modules 1e-5 unless noted</span><span>Bias scalars lr 1e-3</span></div>
  <nav class="toc" aria-label="Sections">
    <a href="#verdict">Verdict</a><a href="#overall">Overall</a><a href="#tasks">Per task</a><a href="#curves">Dev curves</a><a href="#attn">Where the attention looks</a><a href="#bias">λ and μ</a><a href="#inject">Injection and loss</a><a href="#interp">Interpretation</a><a href="#next">What to try next</a>
  </nav>
</header>

<section id="verdict">
  <div class="sec-head"><div class="eyebrow">In one paragraph</div><h2>Verdict</h2></div>
  <div class="verdict">
    <p><strong>Additive still wins (63.5), but aligned cross-attention with both position signals is now within 1.8 points (61.7) and 3.9 above LoRA (57.8).</strong> The old resampler-based cross-attention arms sit at 54.7–55.5, below LoRA. The aligned design's gain comes entirely from position: without any position signal the attention is uniform over all 2,400 geometry tokens (entropy 7.6 of a maximum 7.8 nats, own-frame mass exactly 1/8) and the run scores at LoRA level. The locality bias alone lifts it to 59.2; RoPE₃ on top of the bias adds another 2.5. RoPE₃ alone does nothing (58.3, still uniform attention). The learned bias scalars drifted <em>sharper</em> (λ 1.5 → 1.7), toward the additive arm's one-hot alignment, and the aligned arm's remaining deficit is on relative direction, the task most sensitive to exact token-to-geometry correspondence.</p>
  </div>
</section>

<section id="overall">
  <div class="sec-head"><div class="eyebrow">Full VSI-Bench · 5,130 questions · 32 frames · best-dev checkpoint</div><h2>Overall score by run</h2></div>
  <figure>
    <div class="chart-box" id="c-overall"></div>
    <figcaption>Paired bootstrap intervals (95%, stratified by task) against LoRA: additive 1e-5 +5.7 [+4.3, +7.0]; aligned RoPE₃+bias +3.9 [+2.6, +5.4]; aligned bias-only +1.4 [+0.1, +2.7]; aligned RoPE₃-only +0.5 [−0.9, +1.8]; aligned no-position −0.3 [−1.6, +1.1]; additive 3e-5 +2.3; old xattn 2L −2.3, 8L −3.1. Against additive 1e-5: aligned RoPE₃+bias −1.8 [−3.1, −0.6]. Within the aligned family: both − none +4.3, both − bias +2.5, both − RoPE +3.5, all intervals excluding zero.</figcaption>
  </figure>
  <figure>
    <h3>By source dataset</h3>
    <div class="scroll"><table id="t-ds"></table></div>
    <figcaption>Training videos are ScanNet only. Every arm is 5–7 points weaker on ARKitScenes and ScanNet++, and the ranking is the same in all three, so the aligned arm's gain is not a ScanNet-specific effect. The one exception is additive 3e-5, which keeps its ScanNet score (65.5) but loses 6 points off-domain; the higher fusion lr overfits the training scenes.</figcaption>
  </figure>
</section>

<section id="tasks">
  <div class="sec-head"><div class="eyebrow">Where the points come from</div><h2>Per-task scores and gains over LoRA</h2></div>
  <figure>
    <div class="scroll"><table id="t-tasks"></table></div>
    <figcaption>Coloured cells are the difference from v4 LoRA; bold is the best of the nine runs per task. Standard error per task is about 1.5–3 points. Read the three largest columns first: relative direction, room size and absolute distance are where geometry pays, in every arm that localises it. Appearance order is now 75–83 for every arm (v3: 15–50) because the training mix includes it.</figcaption>
  </figure>
  <figure>
    <div class="chart-box" id="c-taskbars"></div>
    <figcaption>Gain over LoRA per task for the four runs that matter. Additive and the aligned arm agree on <em>where</em> geometry helps; the aligned arm captures about two thirds of additive's relative-direction gain (+10.6 vs +17.9) and less of absolute distance (+5.2 vs +8.4), but matches or beats it on relative distance and appearance order.</figcaption>
  </figure>
</section>

<section id="curves">
  <div class="sec-head"><div class="eyebrow">Dev accuracy every half epoch · 60 held-out ScanNet videos</div><h2>How the runs developed</h2></div>
  <figure>
    <div class="fig-top"><div class="legend" id="leg-dev"></div><span class="eyebrow">click a run to hide it</span></div>
    <div class="chart-box" id="c-dev"></div>
    <figcaption>Mean over the six ScanNet dev tasks. Every run but the 2-layer-resampler xattn peaked at its last evaluation, so none is converged at 5 epochs. The aligned bias-only run has the highest dev accuracy of all (73.9) yet ranks fourth on VSI: dev is in-domain ScanNet, and the bias-only arm's advantage is partly scene familiarity. The two old cross-attention arms start lowest and never catch up.</figcaption>
  </figure>
  <div class="two">
    <figure><div class="chart-box" id="c-devloss"></div><figcaption>Dev loss, every 200 steps. Same picture as v3: minimum around epoch 2, then rising while accuracy still improves. Loss-based selection would have been wrong again.</figcaption></figure>
    <figure><div class="chart-box" id="c-train"></div><figcaption>Training loss, 50-step moving average. The old 8-layer xattn arm (8L) fits worst; the aligned arms fit as well as additive.</figcaption></figure>
  </div>
</section>

<section id="attn">
  <div class="sec-head"><div class="eyebrow">Attention statistics on dev, per injection point, averaged over heads</div><h2>Where the aligned attention looks</h2>
    <p class="lede prose">Three numbers per dev evaluation: <strong>self mass</strong> (weight on the query's own cell), <strong>frame mass</strong> (weight on the query's own frame) and the <strong>entropy</strong> of the attention row. Uniform attention over 2,400 keys has entropy ln 2400 = 7.78 and frame mass 1/8 = 0.125.</p></div>
  <figure>
    <div class="fig-top">
      <div class="seg" role="group" aria-label="Injection point"><button type="button" data-pt="0" aria-pressed="true">after layer 0</button><button type="button" data-pt="1" aria-pressed="false">after layer 1</button><button type="button" data-pt="2" aria-pressed="false">after layer 2</button></div>
      <div class="legend"><span class="chip"><span class="dot" style="background:var(--c-n)"></span>no position</span><span class="chip"><span class="dot" style="background:var(--c-r)"></span>RoPE₃ only</span><span class="chip"><span class="dot" style="background:var(--c-b)"></span>bias only</span><span class="chip"><span class="dot" style="background:var(--c-both)"></span>RoPE₃ + bias</span></div>
    </div>
    <div class="two">
      <div class="chart-box" id="c-self"></div>
      <div class="chart-box" id="c-frame"></div>
    </div>
    <div class="chart-box" id="c-entropy"></div>
    <figcaption>Without the bias, both runs sit at the uniform limit for the whole of training: self mass 0.000, frame mass 0.125, entropy 7.6. RoPE₃ gives the content logits a way to express relative position, but with the output projection starting at zero the content logits never grew enough to localise (the column axis also gets almost no rotation under Qwen-style frequencies; see the λ/μ report). With the bias, the attention starts at self mass ≈ 0.32 and <em>sharpens</em> slowly to 0.36 at layer 0; frame mass 0.73. Deeper injection points are a little wider (layer 2: self 0.27, frame 0.59–0.63), which is where VGGT layer 23's more global features enter.</figcaption>
  </figure>
</section>

<section id="bias">
  <div class="sec-head"><div class="eyebrow">Learned bias scalars, lr 1e-3, init λ = 1.5, μ = 3.0</div><h2>What λ and μ did</h2></div>
  <figure>
    <div class="chart-box" id="c-bias"></div>
    <figcaption>Mean over the 16 heads at each injection point, for the RoPE₃+bias run (the bias-only run is within 0.05 of it everywhere). λ rose from 1.5 to 1.71 / 1.67 / 1.48; μ stayed at 3.10 / 2.98 / 2.93. The model moved toward a <em>sharper</em> spatial prior at the first two points, not a wider one, and left the frame penalty alone. The literature review had predicted that at lr 1e-5 these scalars would not move at all; at 1e-3 they moved by about 0.2, enough to read the direction but not enough to reach a new regime. The additive arm is the limit λ, μ → ∞, and it still wins, so the direction is consistent with the ranking.</figcaption>
  </figure>
</section>

<section id="inject">
  <div class="sec-head"><div class="eyebrow">‖added‖ / ‖h‖ at visual tokens, 50-step average, last micro-batch of each step</div><h2>Injection ratio and what it says</h2></div>
  <div class="two">
    <figure><div class="chart-box" id="c-inj0"></div><figcaption>Injection point after layer 0. The aligned arms inject <em>more</em> than additive (0.38–0.44 vs 0.21) while scoring lower: the additive arm's vector is exactly the token's own geometry, while the aligned arm's is a weighted mix, so a larger norm carries less specific information. The old resampler xattn arms are at 0.3–0.4 with the least specific content of all.</figcaption></figure>
    <figure><div class="chart-box" id="c-inj2"></div><figcaption>Injection point after layer 2. All geometry arms converge to 0.08–0.19 here; the third point matters least, in both norm and (from the v3 per-point correlations) in dev-accuracy association.</figcaption></figure>
  </div>
</section>

<section id="interp">
  <div class="sec-head"><div class="eyebrow">Reading the evidence together</div><h2>Interpretation</h2></div>
  <div class="prose">
    <dl>
      <dt>1. Correspondence, not attention, was the missing ingredient</dt>
      <dd>The v2–v4 cross-attention arms attended from every token to 256 location-free latents and lost 8–9 points to additive. Giving the keys a location (one per visual token) and starting the attention on the matching key recovers 6 of those points. The no-position aligned run, which has the same keys but no way to find the matching one, scores exactly like LoRA: a uniform average of the video's geometry is worth nothing to the LLM.</dd>
      <dt>2. The attention never discovered correspondence on its own</dt>
      <dd>Two runs had the information needed to localise (RoPE₃ gives relative position; the keys are on the same grid) and 7,600 steps to learn it, and stayed uniform. With W_o at zero the gradient to W_q and W_k is zero at step 0 and tiny after, and uniform attention is a flat point for the content logits. The bias breaks the symmetry; RoPE₃ then adds 2.5 points on top, so relative position is useful once something anchors it.</dd>
      <dt>3. Training wanted a sharper prior</dt>
      <dd>λ increased at two of three points, self mass rose through training, and additive (one-hot) still leads. The design question "does a soft alignment beat a fixed one?" currently answers "not yet, and the optimiser is pulling toward fixed". Two readings are possible: the soft alignment is genuinely worse, or the start was too wide (self mass 0.32) and the slow lr on λ could not take it where it wanted to go. The per-head spread from the λ/μ report (λ from 0.18 to 4, μ from 0 to 6) tests the second reading directly.</dd>
      <dt>4. The remaining gap is task-specific</dt>
      <dd>Aligned loses to additive on relative direction (77.6 vs 84.9), route planning (36.6 vs 42.8) and absolute distance (47.1 vs 50.3), and wins on relative distance (60.8 vs 58.7). Relative direction asks "from A facing B, where is C": it needs the geometry of specific tokens, undiluted. Relative distance compares two distances and may benefit from a token seeing its neighbours' geometry. This is the first sign of the two mechanisms having different strengths, which is the result the thesis is after; it needs a second seed before it is more than a sign.</dd>
      <dt>5. The fusion learning rate is a first-order choice for additive</dt>
      <dd>3e-5 instead of 1e-5 costs 3.4 points overall, and the loss is almost all off-domain (ScanNet 65.5 vs 65.3, ARKitScenes 55.2 vs 61.2). The mergers overfit the 1,141 training scenes when opened faster. v3's 1e-5 was the right call and SpatialStack's merger lr is 1e-5 too.</dd>
      <dt>6. Appearance-order data removed the confound, and additive now beats SpatialStack's own geometry increment</dt>
      <dd>With order data in the mix, additive − LoRA is +5.7 at the 2B, against SpatialStack's published +2.8 at the 4B with full fine-tuning. Part of that is the weaker control (our LoRA control is 57.8 vs their data-only 64.8), so the increment is easier to earn; the per-task pattern (relative direction +17.9, room size +13.8, absolute distance +8.4) is nonetheless the clearest geometry signal the project has produced.</dd>
      <dt>7. Dev is a good selector but a biased ranker</dt>
      <dd>Dev picked the last checkpoint in 8 of 9 runs, matching VSI in every case where it mattered. But dev ranks bias-only (73.9) above RoPE₃+bias (72.4) and additive (72.8), while VSI ranks it fourth. Dev is in-domain ScanNet, and the bias-only arm is the one whose off-domain scores fall most. A held-out ARKitScenes/ScanNet++ dev slice would fix the ranking bias.</dd>
    </dl>
  </div>
</section>

<section id="next">
  <div class="sec-head"><div class="eyebrow">Ordered by expected information per GPU-hour</div><h2>What to try next</h2></div>
  <div class="prose">
    <ul>
      <li><strong>Per-head spread of λ and μ</strong> (λ ∈ {4, 1.41, 0.5, 0.18} × μ ∈ {6, 4, 2, 0}, as in the λ/μ report), same recipe otherwise. Tests whether the aligned arm closes the last 1.8 points when some heads start near one-hot. One 7 h run.</li>
      <li><strong>Second seed for additive 1e-5 and aligned RoPE₃+bias.</strong> The per-task split (direction vs distance) is the thesis's most interesting claim and rests on one seed. Two 7 h runs.</li>
      <li><strong>Evaluation-time μ correction</strong> (μ + ln(31/7) at 32 frames). Free: re-run the full-VSI eval on the existing RoPE₃+bias checkpoint. If the arm was leaking attention to the 31 other frames at evaluation, this recovers points on no training.</li>
      <li><strong>Off-domain dev slice.</strong> Hold out some ARKitScenes or ScanNet++ videos for dev so checkpoint ranking stops favouring ScanNet-fitted arms. Data change only.</li>
      <li><strong>Longer training</strong> (8 epochs) for additive and the best aligned run; both still rising at epoch 5.</li>
      <li><strong>Camera/register tokens as extra keys</strong> for the aligned arm: the one input additive cannot use, and the natural route to the relative-direction gap.</li>
    </ul>
  </div>
  <p class="status">Nothing on this page has been launched. All numbers are from completed runs in <code>results/train/2B_v4_*</code> and <code>results/train/2B_xa_*</code>; the <code>2B_xa_none</code> eval job was still finalising when the page was built but its result file was complete.</p>
</section>

<footer>Sources: <code>log.jsonl</code>, <code>best_epoch.json</code> and <code>eval/best/vsi_f32_full.jsonl</code> of each run; bootstrap intervals computed from per-question scores (1,000 resamples). Design: <a href="https://claude.ai/artifact/QrsAZhjPoWfrrLGijrEgWC">JE-xattn Proposal</a>; notebook: <a href="https://claude.ai/artifact/4FjcTJT64mjCgab7h95Xcz">Geometry Fusion Notebook</a>.</footer>
</div>

<script id="data" type="application/json">__DATA__</script>
<script>
(() => {
  const D = JSON.parse(document.getElementById("data").textContent);
  const RUNS = [
    ["lora", "LoRA only", "var(--c-lora)"], ["add1e5", "Additive, fusion lr 1e-5", "var(--c-add)"], ["add3e5", "Additive, fusion lr 3e-5", "var(--c-add2)"],
    ["x2L", "Old xattn, 2-layer resampler", "var(--c-x)"], ["x8L", "Old xattn, 8-layer resampler", "var(--c-x2)"],
    ["xa_none", "Aligned, no position", "var(--c-n)"], ["xa_rope", "Aligned, RoPE₃ only", "var(--c-r)"], ["xa_bias", "Aligned, bias only", "var(--c-b)"], ["xa_both", "Aligned, RoPE₃ + bias", "var(--c-both)"],
  ];
  const NAME = Object.fromEntries(RUNS.map(r => [r[0], r[1]])), COL = Object.fromEntries(RUNS.map(r => [r[0], r[2]]));
  const TASKS = [["object_rel_direction", "Rel direction"], ["room_size_estimation", "Room size"], ["object_abs_distance", "Abs distance"], ["route_planning", "Route"], ["object_rel_distance", "Rel distance"], ["obj_appearance_order", "Appearance"], ["object_counting", "Counting"], ["object_size_estimation", "Object size"]];
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent && parent.appendChild(e); return e; };

  // ---------- horizontal bar chart: overall ----------
  function barChart(box, items, { xmin, xmax, ticks, title, ref }) {
    box.innerHTML = "";
    const W = 980, rowH = 30, L = 250, R = 60, T = 30, H = T + items.length * rowH + 20;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": title, style: "width:100%;height:auto;display:block" }, box);
    const x = v => L + (v - xmin) / (xmax - xmin) * (W - L - R);
    el("text", { x: L, y: 16, class: "lab" }, svg).textContent = title;
    for (const t of ticks) { el("line", { x1: x(t), x2: x(t), y1: T, y2: H - 20, class: "grid" }, svg); el("text", { x: x(t), y: H - 6, "text-anchor": "middle" }, svg).textContent = t; }
    if (ref != null) { el("line", { x1: x(ref), x2: x(ref), y1: T - 4, y2: H - 20, class: "ref" }, svg); }
    items.forEach((it, i) => {
      const y = T + i * rowH;
      el("rect", { x: x(xmin), y: y + 6, width: x(it.v) - x(xmin), height: rowH - 12, rx: 3, fill: it.color }, svg);
      el("text", { x: L - 10, y: y + rowH / 2 + 4, "text-anchor": "end", class: "lab", style: it.bold ? "font-weight:600" : "" }, svg).textContent = it.name;
      el("text", { x: x(it.v) + 6, y: y + rowH / 2 + 4, class: "lab" }, svg).textContent = it.v.toFixed(1);
    });
  }
  const items = RUNS.map(([k, n, c]) => ({ name: n, v: D[k].vsi.overall, color: c, bold: k === "add1e5" || k === "xa_both" })).sort((a, b) => b.v - a.v);
  barChart(document.getElementById("c-overall"), items, { xmin: 50, xmax: 66, ticks: [50, 52, 54, 56, 58, 60, 62, 64, 66], title: "Full VSI-Bench overall (%), 32 frames", ref: D.lora.vsi.overall });

  // ---------- tables ----------
  const tds = document.getElementById("t-ds");
  tds.innerHTML = `<thead><tr><th>Run</th><th class="num">Overall</th><th class="num">ScanNet</th><th class="num">ARKitScenes</th><th class="num">ScanNet++</th><th class="num">off-domain − ScanNet</th></tr></thead><tbody>` +
    RUNS.map(([k, n, c]) => { const d = D[k].vsi_by_dataset, off = (d.arkitscenes + d.scannetpp) / 2 - d.scannet; return `<tr><td><span class="chip"><span class="dot" style="background:${c}"></span>${n}</span></td><td class="num">${D[k].vsi.overall.toFixed(1)}</td><td class="num">${d.scannet.toFixed(1)}</td><td class="num">${d.arkitscenes.toFixed(1)}</td><td class="num">${d.scannetpp.toFixed(1)}</td><td class="num ${off < -6 ? "neg" : ""}">${off.toFixed(1)}</td></tr>`; }).join("") + "</tbody>";
  const tt = document.getElementById("t-tasks");
  const heat = v => { const a = Math.min(1, Math.abs(v) / 15); const c = v >= 0 ? `rgba(29,122,58,${0.08 + 0.5 * a})` : `rgba(179,50,47,${0.08 + 0.5 * a})`; return `background:${c}`; };
  const best = {}; for (const [t] of TASKS) best[t] = Math.max(...RUNS.map(([k]) => D[k].vsi[t])); best.overall = Math.max(...RUNS.map(([k]) => D[k].vsi.overall));
  tt.innerHTML = `<thead><tr><th>Run</th><th class="num">Overall</th>${TASKS.map(([, s]) => `<th class="num">${s}</th>`).join("")}</tr></thead><tbody>` +
    RUNS.map(([k, n, c]) => { const v = D[k].vsi, L0 = D.lora.vsi; const cell = (t) => { const d = v[t] - L0[t]; return k === "lora" ? `<td class="num${v[t] === best[t] ? " best" : ""}">${v[t].toFixed(1)}</td>` : `<td class="heat${v[t] === best[t] ? " best" : ""}" style="${heat(d)}">${v[t].toFixed(1)} <span style="color:var(--ink-3);font-size:11px">${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(1)}</span></td>`; };
      return `<tr><td><span class="chip"><span class="dot" style="background:${c}"></span>${n}</span></td>${cell("overall")}${TASKS.map(([t]) => cell(t)).join("")}</tr>`; }).join("") + "</tbody>";

  // ---------- grouped bars: gain over LoRA per task ----------
  function groupBars(box, groups, series, { ymin, ymax, ticks, title }) {
    box.innerHTML = "";
    const W = 980, H = 320, L = 46, R = 16, T = 30, B = 44;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": title, style: "width:100%;height:auto;display:block" }, box);
    const gw = (W - L - R) / groups.length, bw = gw / (series.length + 1);
    const y = v => T + (1 - (v - ymin) / (ymax - ymin)) * (H - T - B);
    el("text", { x: L, y: 16, class: "lab" }, svg).textContent = title;
    for (const t of ticks) { el("line", { x1: L, x2: W - R, y1: y(t), y2: y(t), class: t === 0 ? "axis" : "grid" }, svg); el("text", { x: L - 6, y: y(t) + 4, "text-anchor": "end" }, svg).textContent = (t > 0 ? "+" : "") + t; }
    groups.forEach((g, gi) => {
      el("text", { x: L + gw * gi + gw / 2, y: H - B + 18, "text-anchor": "middle" }, svg).textContent = g.label;
      series.forEach((s, si) => {
        const v = g.vals[si]; const x0 = L + gw * gi + bw * (si + 0.5);
        el("rect", { x: x0, y: Math.min(y(v), y(0)), width: bw - 2, height: Math.abs(y(v) - y(0)), fill: s.color, rx: 2 }, svg);
        const t = el("text", { x: x0 + (bw - 2) / 2, y: v >= 0 ? y(v) - 4 : y(v) + 12, "text-anchor": "middle", style: "font-size:10px" }, svg); t.textContent = (v >= 0 ? "+" : "") + v.toFixed(1);
      });
    });
    series.forEach((s, si) => { const lx = L + si * 230, ly = H - 8; el("rect", { x: lx, y: ly - 9, width: 10, height: 10, fill: s.color, rx: 2 }, svg); const t = el("text", { x: lx + 14, y: ly, class: "lab", style: "font-size:12px" }, svg); t.textContent = s.name; });
  }
  const SER = [["add1e5", "Additive 1e-5"], ["xa_both", "Aligned RoPE₃+bias"], ["xa_bias", "Aligned bias only"], ["x2L", "Old xattn 2L"]].map(([k, n]) => ({ k, name: n, color: COL[k] }));
  groupBars(document.getElementById("c-taskbars"), TASKS.map(([t, s]) => ({ label: s, vals: SER.map(x => D[x.k].vsi[t] - D.lora.vsi[t]) })), SER, { ymin: -24, ymax: 20, ticks: [-20, -10, 0, 10, 20], title: "Gain over LoRA per task (points)" });

  // ---------- generic xy line chart ----------
  function xyChart(box, { series, xmin, xmax, ymin, ymax, xticks, yticks, xlabel, title, fmt, markers, refs }) {
    box.innerHTML = "";
    const W = 980, H = 330, L = 50, R = 16, T = 28, B = 40;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": title, style: "width:100%;height:auto;display:block" }, box);
    const x = v => L + (v - xmin) / (xmax - xmin) * (W - L - R), y = v => T + (1 - (Math.min(ymax, Math.max(ymin, v)) - ymin) / (ymax - ymin)) * (H - T - B);
    el("text", { x: L, y: 14, class: "lab", style: "fill:var(--ink-2)" }, svg).textContent = title;
    for (const t of yticks) { el("line", { x1: L, x2: W - R, y1: y(t), y2: y(t), class: "grid" }, svg); el("text", { x: L - 8, y: y(t) + 4, "text-anchor": "end" }, svg).textContent = fmt(t); }
    el("line", { x1: L, x2: W - R, y1: H - B, y2: H - B, class: "axis" }, svg);
    for (const [v, lab] of xticks) el("text", { x: x(v), y: H - B + 18, "text-anchor": "middle" }, svg).textContent = lab;
    el("text", { x: (L + W - R) / 2, y: H - 4, "text-anchor": "middle" }, svg).textContent = xlabel;
    for (const r of refs || []) { el("line", { x1: L, x2: W - R, y1: y(r.v), y2: y(r.v), class: "ref" }, svg); el("text", { x: W - R - 4, y: y(r.v) - 5, "text-anchor": "end" }, svg).textContent = r.label; }
    const vis = series.filter(s => !s.hidden);
    for (const s of vis) {
      el("path", { d: s.points.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join(""), class: "line", stroke: s.color, "stroke-dasharray": s.dash || "none" }, svg);
      if (markers) for (const p of s.points) el("circle", { cx: x(p[0]), cy: y(p[1]), r: 2.8, fill: s.color, stroke: "var(--panel)", "stroke-width": 1.2 }, svg);
    }
    const cross = el("line", { y1: T, y2: H - B, class: "cross" }, svg);
    const dots = vis.map(s => el("circle", { r: 4.5, fill: s.color, stroke: "var(--panel)", "stroke-width": 2, opacity: 0 }, svg));
    const hit = el("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "transparent" }, svg);
    const tip = document.createElement("div"); tip.className = "tip"; tip.hidden = true; box.appendChild(tip);
    const nearest = (pts, xv) => pts.reduce((a, p) => Math.abs(p[0] - xv) < Math.abs(a[0] - xv) ? p : a, pts[0]);
    hit.addEventListener("pointermove", evt => {
      const pt = svg.createSVGPoint(); pt.x = evt.clientX; pt.y = evt.clientY; const p = pt.matrixTransform(svg.getScreenCTM().inverse());
      const xv = xmin + (p.x - L) / (W - L - R) * (xmax - xmin);
      const rows = vis.map((s, k) => { const q = nearest(s.points, xv); dots[k].setAttribute("cx", x(q[0])); dots[k].setAttribute("cy", y(q[1])); dots[k].setAttribute("opacity", 1); return { s, q }; });
      if (!rows.length) return;
      const xr = rows[0].q[0]; svg.classList.add("hover"); cross.setAttribute("x1", x(xr)); cross.setAttribute("x2", x(xr));
      tip.innerHTML = `<b>epoch ${xr.toFixed(2)}</b>` + rows.sort((a, b) => b.q[1] - a.q[1]).map(r => `<div class="row"><span><span class="dot" style="background:${r.s.color}"></span>${r.s.name}</span><span>${fmt(r.q[1], true)}</span></div>`).join("");
      tip.hidden = false; const bw = box.clientWidth, sx = bw / W, left = x(xr) * sx; tip.style.left = (left + 14 + 190 > bw ? left - 14 - tip.offsetWidth : left + 14) + "px"; tip.style.top = (T * sx + 4) + "px";
    });
    hit.addEventListener("pointerleave", () => { svg.classList.remove("hover"); dots.forEach(d => d.setAttribute("opacity", 0)); tip.hidden = true; });
  }
  const EP = [[0, "0"], [1, "1"], [2, "2"], [3, "3"], [4, "4"], [5, "5"]];
  const hidden = new Set(["add3e5", "x8L"]);
  const drawDev = () => xyChart(document.getElementById("c-dev"), { title: "Dev accuracy (%), mean of 6 ScanNet tasks", xlabel: "epoch", xmin: 0, xmax: 5, ymin: 40, ymax: 76, xticks: EP, yticks: [40, 48, 56, 64, 72], fmt: (v, p) => p ? v.toFixed(1) : v, markers: true,
    series: RUNS.map(([k, n, c]) => ({ name: n, color: c, points: D[k].dev_acc.filter(p => p[0] > 0), hidden: hidden.has(k) })) });
  const leg = document.getElementById("leg-dev");
  for (const [k, n, c] of RUNS) { const s = document.createElement("span"); s.className = "chip" + (hidden.has(k) ? " off" : ""); s.innerHTML = `<span class="dot" style="background:${c}"></span>${n}`; s.addEventListener("click", () => { hidden.has(k) ? hidden.delete(k) : hidden.add(k); s.classList.toggle("off"); drawDev(); }); leg.appendChild(s); }
  drawDev();
  xyChart(document.getElementById("c-devloss"), { title: "Dev loss", xlabel: "epoch", xmin: 0, xmax: 5, ymin: 0.6, ymax: 1.5, xticks: EP, yticks: [0.6, 0.8, 1.0, 1.2, 1.4], fmt: (v, p) => p ? v.toFixed(3) : v.toFixed(1),
    series: RUNS.filter(([k]) => !hidden.has(k)).map(([k, n, c]) => ({ name: n, color: c, points: D[k].dev_loss })) });
  xyChart(document.getElementById("c-train"), { title: "Training loss, 50-step average", xlabel: "epoch", xmin: 0, xmax: 5, ymin: 0, ymax: 1.2, xticks: EP, yticks: [0, 0.3, 0.6, 0.9, 1.2], fmt: (v, p) => p ? v.toFixed(3) : v.toFixed(1),
    series: RUNS.filter(([k]) => !hidden.has(k)).map(([k, n, c]) => ({ name: n, color: c, points: D[k].train_loss })) });

  // ---------- attention stats ----------
  const XA = [["xa_none", "no position", "var(--c-n)"], ["xa_rope", "RoPE₃ only", "var(--c-r)"], ["xa_bias", "bias only", "var(--c-b)"], ["xa_both", "RoPE₃ + bias", "var(--c-both)"]];
  function drawAttn(pt) {
    const ser = key => XA.map(([k, n, c]) => ({ name: n, color: c, points: D[k].attn[`${pt}_${key}`] }));
    xyChart(document.getElementById("c-self"), { title: `Self mass (weight on own cell), after layer ${pt}`, xlabel: "epoch", xmin: 0, xmax: 5, ymin: 0, ymax: 0.5, xticks: EP, yticks: [0, 0.1, 0.2, 0.3, 0.4, 0.5], fmt: (v, p) => p ? v.toFixed(3) : v.toFixed(1), markers: true, series: ser("self_mass") });
    xyChart(document.getElementById("c-frame"), { title: `Own-frame mass, after layer ${pt}`, xlabel: "epoch", xmin: 0, xmax: 5, ymin: 0, ymax: 1, xticks: EP, yticks: [0, 0.25, 0.5, 0.75, 1], fmt: (v, p) => p ? v.toFixed(3) : v.toFixed(2), markers: true, refs: [{ v: 0.125, label: "uniform 1/8" }], series: ser("frame_mass") });
    xyChart(document.getElementById("c-entropy"), { title: `Attention entropy (nats), after layer ${pt}`, xlabel: "epoch", xmin: 0, xmax: 5, ymin: 2, ymax: 8, xticks: EP, yticks: [2, 3, 4, 5, 6, 7, 8], fmt: (v, p) => p ? v.toFixed(2) : v, markers: true, refs: [{ v: Math.log(2400), label: "uniform ln 2400 = 7.78" }], series: ser("entropy") });
  }
  drawAttn(0);
  document.querySelectorAll("button[data-pt]").forEach(b => b.addEventListener("click", () => { document.querySelectorAll("button[data-pt]").forEach(o => o.setAttribute("aria-pressed", o === b)); drawAttn(b.dataset.pt); }));

  // ---------- bias scalars ----------
  const bb = D.xa_both.bias;
  xyChart(document.getElementById("c-bias"), { title: "Learned λ (solid) and μ (dashed), mean over heads, RoPE₃ + bias run", xlabel: "epoch", xmin: 0, xmax: 5, ymin: 1, ymax: 3.5, xticks: EP, yticks: [1, 1.5, 2, 2.5, 3, 3.5], fmt: (v, p) => p ? v.toFixed(3) : v.toFixed(1),
    series: [0, 1, 2].flatMap(i => [{ name: `λ after layer ${i}`, color: ["var(--c-both)", "var(--c-b)", "var(--c-r)"][i], points: bb.map(r => [r[0], r[1 + 2 * i]]) }, { name: `μ after layer ${i}`, color: ["var(--c-both)", "var(--c-b)", "var(--c-r)"][i], dash: "5 4", points: bb.map(r => [r[0], r[2 + 2 * i]]) }]) });

  // ---------- injection ----------
  const INJ = [["add1e5", "Additive 1e-5", "var(--c-add)"], ["add3e5", "Additive 3e-5", "var(--c-add2)"], ["x2L", "Old xattn 2L", "var(--c-x)"], ["xa_none", "Aligned, no position", "var(--c-n)"], ["xa_bias", "Aligned, bias only", "var(--c-b)"], ["xa_both", "Aligned, RoPE₃ + bias", "var(--c-both)"]];
  for (const [id, pt] of [["c-inj0", 0], ["c-inj2", 2]]) xyChart(document.getElementById(id), { title: `Injection ratio after layer ${pt}`, xlabel: "epoch", xmin: 0, xmax: 5, ymin: 0, ymax: 0.6, xticks: EP, yticks: [0, 0.15, 0.3, 0.45, 0.6], fmt: (v, p) => p ? v.toFixed(3) : v.toFixed(2),
    series: INJ.map(([k, n, c]) => ({ name: n, color: c, points: D[k].inject.map(r => [r[0], r[1 + pt]]) })) });
})();
</script>
'''
open(f"{W}/docs/v4_results.html", "w").write(PAGE.replace("__DATA__", DATA))
print("written", len(PAGE) + len(DATA))
