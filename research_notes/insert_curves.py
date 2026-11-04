"""Insert the 'Run v3 in charts' section (results table + dev accuracy / loss / injection charts) into the notebook page."""
import json

P = "/shared/workspace/povejmo/math/projects/mvp/.claude/worktrees/je-xattn-proposal/docs/geometry_fusion_notebook.html"
html = open(P).read()
assert "id=\"v3-curves\"" not in html, "already inserted"
curves = json.load(open("/shared/workspace/povejmo/math/projects/mvp/.claude/worktrees/je-xattn-proposal/research_notes/v3_curves.json"))

SECTION = r'''
<section id="v3-curves">
  <div class="sec-head"><div class="eyebrow">Run v3 · all six runs · 6 epochs · 931 steps per epoch</div><h2>Results at a glance, and how they developed</h2>
    <p class="lede prose">Every run: full VSI-Bench per task at the best-dev checkpoint, then the curves behind it. Switch between the 2B and 4B base with the toggle; hover any chart for exact values.</p></div>
  <figure>
    <h3>Full VSI-Bench at 32 frames, best-dev checkpoint (epoch 6 in every run)</h3>
    <div class="scroll"><table>
      <thead><tr><th>Run</th><th class="num">Overall</th><th class="num">Count</th><th class="num">Abs dist</th><th class="num">Size</th><th class="num">Room</th><th class="num">Rel dist</th><th class="num">Rel dir</th><th class="num">Route</th><th class="num">Appear</th><th class="num">Dev acc</th><th class="num">Train loss ep 6</th><th class="num">Grad spikes</th></tr></thead>
      <tbody>
        <tr><td><span class="chip"><span class="dot d1"></span>2B LoRA only</span></td><td class="num">52.7</td><td class="num">65.9</td><td class="num">44.1</td><td class="num">62.5</td><td class="num">54.4</td><td class="num">59.0</td><td class="num">70.7</td><td class="num">40.2</td><td class="num">24.8</td><td class="num">71.4</td><td class="num">0.222</td><td class="num">11</td></tr>
        <tr><td><span class="chip"><span class="dot d2"></span>2B additive</span></td><td class="num best">55.0</td><td class="num">65.0</td><td class="num best">52.5</td><td class="num">61.2</td><td class="num best">64.2</td><td class="num">59.3</td><td class="num best">82.3</td><td class="num">40.2</td><td class="num">15.7</td><td class="num best">73.5</td><td class="num">0.237</td><td class="num">10</td></tr>
        <tr><td><span class="chip"><span class="dot d3"></span>2B cross-attention</span></td><td class="num">49.0</td><td class="num">64.7</td><td class="num">41.4</td><td class="num">58.7</td><td class="num">58.5</td><td class="num">53.9</td><td class="num">61.7</td><td class="num">33.0</td><td class="num">20.4</td><td class="num">69.9</td><td class="num">0.285</td><td class="num">70</td></tr>
        <tr><td><span class="chip"><span class="dot d1"></span>4B LoRA only</span></td><td class="num">58.7</td><td class="num">65.9</td><td class="num">49.0</td><td class="num best">63.8</td><td class="num">55.7</td><td class="num best">66.2</td><td class="num">75.8</td><td class="num best">42.3</td><td class="num best">50.6</td><td class="num">73.9</td><td class="num">0.125</td><td class="num">12</td></tr>
        <tr><td><span class="chip"><span class="dot d2"></span>4B additive</span></td><td class="num best">59.0</td><td class="num best">67.8</td><td class="num best">54.5</td><td class="num">62.7</td><td class="num">63.9</td><td class="num">63.8</td><td class="num best">82.4</td><td class="num">41.2</td><td class="num">35.4</td><td class="num best">74.7</td><td class="num">0.158</td><td class="num">14</td></tr>
        <tr><td><span class="chip"><span class="dot d3"></span>4B cross-attention</span></td><td class="num">51.0</td><td class="num">63.9</td><td class="num">43.7</td><td class="num">58.2</td><td class="num">58.9</td><td class="num">57.3</td><td class="num">69.0</td><td class="num">38.1</td><td class="num">19.3</td><td class="num">72.9</td><td class="num">0.203</td><td class="num">78</td></tr>
        <tr><td style="color:var(--ink-3)">SpatialStack-4B, our reproduction</td><td class="num" style="color:var(--ink-3)">67.5</td><td class="num" style="color:var(--ink-3)">70.8</td><td class="num" style="color:var(--ink-3)">55.4</td><td class="num" style="color:var(--ink-3)">69.0</td><td class="num" style="color:var(--ink-3)">69.1</td><td class="num" style="color:var(--ink-3)">66.8</td><td class="num" style="color:var(--ink-3)">83.9</td><td class="num" style="color:var(--ink-3)">41.8</td><td class="num" style="color:var(--ink-3)">83.0</td><td class="num" style="color:var(--ink-3)">–</td><td class="num" style="color:var(--ink-3)">–</td><td class="num" style="color:var(--ink-3)">–</td></tr>
      </tbody>
    </table></div>
    <figcaption>Bold: best of the three arms within each base size. Dev acc is the mean over the six dev tasks at epoch 6 (560 questions on 60 held-out ScanNet videos). Train loss is the epoch-6 mean. Grad spikes counts optimizer steps with gradient norm above 50 before clipping; the cross-attention arms' maxima were 6,469 (2B) and 7,294 (4B) against about 250–350 for the others.</figcaption>
  </figure>
  <figure>
    <div class="fig-top">
      <div class="seg" role="group" aria-label="Base model for the curves">
        <button type="button" data-sz="2B" aria-pressed="true">Qwen3.5-2B</button>
        <button type="button" data-sz="4B" aria-pressed="false">Qwen3.5-4B</button>
      </div>
      <div class="legend"><span class="chip"><span class="dot d1"></span>LoRA only</span><span class="chip"><span class="dot d2"></span>Additive</span><span class="chip"><span class="dot d3"></span>Cross-attention</span></div>
    </div>
    <div class="two">
      <div class="chart-box" id="c-devacc"></div>
      <div class="chart-box" id="c-devloss"></div>
      <div class="chart-box" id="c-train"></div>
      <div class="chart-box" id="c-inject"></div>
    </div>
    <figcaption><strong>Dev accuracy</strong> rises for all arms through epoch 6 and ranks additive ≥ LoRA > cross-attention at both sizes from epoch 3 on; cross-attention starts 7–10 points behind. <strong>Dev loss</strong> (every 200 steps) bottoms out in epochs 2–4 and then climbs while accuracy keeps improving, so loss-based selection would have picked an earlier, worse checkpoint; the rise is overconfidence on the questions still answered wrong, not a drop in accuracy. <strong>Training loss</strong> (50-step moving average) falls the same way for every arm; the geometry arms are never ahead of LoRA on training loss, so the additive gain is generalisation, not fit. <strong>Injection ratio</strong> ‖added‖/‖h‖ (mean of the three points, 50-step average) grows slowly under the 1e-5 fusion lr and settles at 0.15–0.3, versus 0.4–0.6 (additive) and 1.5 (cross-attention) in v2.</figcaption>
  </figure>
</section>
'''

SCRIPT = r'''
<script id="v3-curves-data" type="application/json">__DATA__</script>
<script>
(() => {
  const DATA = JSON.parse(document.getElementById("v3-curves-data").textContent);
  const ARMS = [["lora", "LoRA only", "var(--s1)"], ["additive", "Additive", "var(--s2)"], ["xattn", "Cross-attention", "var(--s3)"]];
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent && parent.appendChild(e); return e; };
  function xyChart(box, { series, xmin, xmax, ymin, ymax, xticks, yticks, xlabel, title, fmt, vlines, markers }) {
    box.innerHTML = "";
    const W = 520, H = 290, L = 50, R = 16, T = 28, B = 40;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": title, style: "width:100%;height:auto;display:block" }, box);
    const x = v => L + (v - xmin) / (xmax - xmin) * (W - L - R), y = v => T + (1 - (v - ymin) / (ymax - ymin)) * (H - T - B);
    el("text", { x: L, y: 14, style: "font:500 12px var(--f-body);fill:var(--ink-2)" }, svg).textContent = title;
    for (const t of yticks) { el("line", { x1: L, x2: W - R, y1: y(t), y2: y(t), class: "grid" }, svg); el("text", { x: L - 8, y: y(t) + 4, "text-anchor": "end" }, svg).textContent = fmt(t); }
    for (const v of vlines || []) el("line", { x1: x(v), x2: x(v), y1: T, y2: H - B, class: "grid" }, svg);
    el("line", { x1: L, x2: W - R, y1: H - B, y2: H - B, class: "axis" }, svg);
    for (const [v, lab] of xticks) el("text", { x: x(v), y: H - B + 18, "text-anchor": "middle" }, svg).textContent = lab;
    el("text", { x: (L + W - R) / 2, y: H - 4, "text-anchor": "middle" }, svg).textContent = xlabel;
    for (const s of series) {
      const pts = s.points.filter(p => p[1] >= ymin && p[1] <= ymax || true);
      el("path", { d: pts.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(Math.min(ymax, Math.max(ymin, p[1]))).toFixed(1)}`).join(""), class: "line", stroke: s.color, "stroke-width": s.thin ? 1.4 : 2 }, svg);
      if (markers) for (const p of pts) el("circle", { cx: x(p[0]), cy: y(p[1]), r: 3.2, fill: s.color, stroke: "var(--panel)", "stroke-width": 1.5 }, svg);
    }
    const cross = el("line", { y1: T, y2: H - B, class: "cross" }, svg);
    const dots = series.map(s => el("circle", { r: 4.5, fill: s.color, stroke: "var(--panel)", "stroke-width": 2, opacity: 0 }, svg));
    const hit = el("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "transparent" }, svg);
    const tip = document.createElement("div"); tip.className = "tip"; tip.hidden = true; box.appendChild(tip);
    const nearest = (pts, xv) => pts.reduce((a, p) => Math.abs(p[0] - xv) < Math.abs(a[0] - xv) ? p : a, pts[0]);
    const show = evt => {
      const pt = svg.createSVGPoint(); pt.x = evt.clientX; pt.y = evt.clientY;
      const p = pt.matrixTransform(svg.getScreenCTM().inverse());
      const xv = xmin + (p.x - L) / (W - L - R) * (xmax - xmin);
      const rows = series.map((s, k) => { const q = nearest(s.points, xv); dots[k].setAttribute("cx", x(q[0])); dots[k].setAttribute("cy", y(q[1])); dots[k].setAttribute("opacity", 1); return { s, q }; });
      const xr = rows[0].q[0];
      svg.classList.add("hover"); cross.setAttribute("x1", x(xr)); cross.setAttribute("x2", x(xr));
      tip.innerHTML = `<b>${series[0].xlab ? series[0].xlab(xr) : xr}</b>` + rows.sort((a, b) => b.q[1] - a.q[1]).map(r => `<div class="row"><span><span class="dot" style="background:${r.s.color}"></span>${r.s.name}</span><span>${fmt(r.q[1], true)}</span></div>`).join("");
      tip.hidden = false;
      const bw = box.clientWidth, sx = bw / W, left = x(xr) * sx;
      tip.style.left = (left + 14 + 170 > bw ? left - 14 - tip.offsetWidth : left + 14) + "px";
      tip.style.top = (T * sx + 4) + "px";
    };
    const hide = () => { svg.classList.remove("hover"); dots.forEach(d => d.setAttribute("opacity", 0)); tip.hidden = true; };
    hit.addEventListener("pointermove", show); hit.addEventListener("pointerdown", show); hit.addEventListener("pointerleave", hide);
  }
  const SPE = 931, TOTAL = 5586;
  const epochTicks = [[0, "0"], [1, "1"], [2, "2"], [3, "3"], [4, "4"], [5, "5"], [6, "6"]];
  const stepTicks = [0, 1, 2, 3, 4, 5, 6].map(e => [e * SPE, String(e)]);
  const epochLines = [1, 2, 3, 4, 5, 6].map(e => e * SPE);
  const epLab = s => `Step ${s} (epoch ${(s / SPE).toFixed(2)})`;
  function draw(size) {
    const runs = ARMS.map(([k, name, color]) => ({ k, name, color, d: DATA[`${size}_v3_${k}`] }));
    xyChart(document.getElementById("c-devacc"), { title: `Dev accuracy (%) · ${size}`, xlabel: "epoch", xmin: 0, xmax: 6, ymin: 30, ymax: 80, xticks: epochTicks, yticks: [30, 40, 50, 60, 70, 80], fmt: (v, p) => p ? v.toFixed(1) : v, markers: true,
      series: runs.map(r => ({ name: r.name, color: r.color, points: r.d.dev_acc.map(([s, v]) => [s / SPE, v]), xlab: e => `Epoch ${Math.round(e)}` })) });
    xyChart(document.getElementById("c-devloss"), { title: `Dev loss · ${size}`, xlabel: "epoch", xmin: 0, xmax: TOTAL, ymin: 0.6, ymax: 1.6, xticks: stepTicks, yticks: [0.6, 0.8, 1.0, 1.2, 1.4, 1.6], fmt: (v, p) => p ? v.toFixed(3) : v.toFixed(1), vlines: epochLines, markers: false,
      series: runs.map(r => ({ name: r.name, color: r.color, points: r.d.dev_loss, xlab: epLab })) });
    xyChart(document.getElementById("c-train"), { title: `Training loss, 50-step average · ${size}`, xlabel: "epoch", xmin: 0, xmax: TOTAL, ymin: 0, ymax: 1.2, xticks: stepTicks, yticks: [0, 0.3, 0.6, 0.9, 1.2], fmt: (v, p) => p ? v.toFixed(3) : v.toFixed(1), vlines: epochLines,
      series: runs.map(r => ({ name: r.name, color: r.color, points: r.d.train_loss, xlab: epLab })) });
    xyChart(document.getElementById("c-inject"), { title: `Injection ratio ‖added‖/‖h‖, mean of 3 points · ${size}`, xlabel: "epoch", xmin: 0, xmax: TOTAL, ymin: 0, ymax: 0.4, xticks: stepTicks, yticks: [0, 0.1, 0.2, 0.3, 0.4], fmt: (v, p) => p ? v.toFixed(3) : v.toFixed(1), vlines: epochLines,
      series: runs.filter(r => r.d.inject.length).map(r => ({ name: r.name, color: r.color, points: r.d.inject.map(([s, a, b, c]) => [s, (a + b + c) / 3]), xlab: epLab })) });
  }
  draw("2B");
  document.querySelectorAll("button[data-sz]").forEach(b => b.addEventListener("click", () => {
    document.querySelectorAll("button[data-sz]").forEach(o => o.setAttribute("aria-pressed", o === b)); draw(b.dataset.sz);
  }));
})();
</script>
'''

anchor = '<section id="now">'
assert html.count(anchor) == 1
html = html.replace(anchor, SECTION.strip() + "\n\n" + anchor)
nav_old = '<a href="#v3">Where we stand (v3)</a>'
assert html.count(nav_old) == 1
html = html.replace(nav_old, nav_old + '<a href="#v3-curves">v3 charts</a>')
end = "</script>\n\n</body></html>"
assert html.count(end) == 1
data = json.dumps(curves, separators=(",", ":")).replace("</", "<\\/")
html = html.replace(end, "</script>\n" + SCRIPT.replace("__DATA__", data).strip() + "\n\n</body></html>")
open(P, "w").write(html)
print("inserted; size", len(html))
