"""The HTML page of `rg.observe.report`. Self-contained: the data is embedded and nothing loads."""

PAGE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rg report</title>
<style>
:root {
  color-scheme: light;
  --surface-0: #f4f3ef; --surface-1: #fcfcfb; --border: #e2e1dc; --grid: #e8e7e2;
  --text-1: #0b0b0b; --text-2: #52514e; --text-3: #75746f;
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100;
  --series-5: #e87ba4; --series-6: #008300; --series-7: #6250d6; --series-8: #e34948;
  --mark-ring: #fcfcfb; --critical: #b3261e;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface-0: #111110; --surface-1: #1a1a19; --border: #33332f; --grid: #2a2a27;
    --text-1: #f0efec; --text-2: #c3c2b7; --text-3: #9b9a90;
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
    --series-5: #d55181; --series-6: #008300; --series-7: #9085e9; --series-8: #e66767;
    --mark-ring: #1a1a19; --critical: #f2b8b5;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #111110; --surface-1: #1a1a19; --border: #33332f; --grid: #2a2a27;
  --text-1: #f0efec; --text-2: #c3c2b7; --text-3: #9b9a90;
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
  --series-5: #d55181; --series-6: #008300; --series-7: #9085e9; --series-8: #e66767;
  --mark-ring: #1a1a19; --critical: #f2b8b5;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface-0); color: var(--text-1); font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", "Apple SD Gothic Neo", "Noto Sans KR", sans-serif; }
main { max-width: 1120px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; font-weight: 600; }
h2 { font-size: 15px; margin: 0; font-weight: 600; }
.sub { color: var(--text-2); margin: 0 0 20px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 16px; }
.tile, .card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; }
.tile { padding: 12px 14px; }
.tile .label { color: var(--text-2); font-size: 12px; }
.tile .value { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 480px), 1fr)); gap: 16px; }
.card { padding: 14px 16px; min-width: 0; }
.card.wide { grid-column: 1 / -1; }
.card header { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; flex-wrap: wrap; margin-bottom: 8px; }
.note { color: var(--text-2); font-size: 12px; margin: 6px 0 0; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 12px; color: var(--text-2); margin-top: 6px; }
.legend i { display: inline-block; width: 12px; height: 3px; border-radius: 2px; margin-right: 5px; vertical-align: middle; }
.chart { position: relative; }
svg, canvas { display: block; width: 100%; touch-action: pan-y; }
button { font: inherit; font-size: 12px; color: var(--text-1); background: transparent; border: 1px solid var(--border); border-radius: 6px; padding: 2px 8px; cursor: pointer; min-height: 28px; }
button:hover { background: var(--grid); }
button:focus-visible, canvas:focus-visible { outline: 2px solid var(--series-1); outline-offset: 2px; }
#tip { position: fixed; pointer-events: none; z-index: 5; background: var(--surface-1); border: 1px solid var(--border); border-radius: 6px; padding: 6px 8px; font-size: 12px; box-shadow: 0 2px 8px rgba(0,0,0,.18); display: none; max-width: 300px; font-variant-numeric: tabular-nums; }
#tip b { font-weight: 600; }
.findings { margin: 0 0 16px; padding: 12px 14px; list-style: none; background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; }
.findings li { padding: 4px 0; }
.findings code { font-size: 12px; }
.tag { display: inline-block; font-size: 11px; color: var(--text-2); border: 1px solid var(--border); border-radius: 4px; padding: 0 5px; margin-right: 6px; }
.tag.critical { color: var(--critical); border-color: var(--critical); }
details { margin-top: 8px; font-size: 12px; color: var(--text-2); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; margin-top: 6px; }
th, td { padding: 2px 10px 2px 0; text-align: left; border-bottom: 1px solid var(--grid); white-space: nowrap; }
.empty { color: var(--text-2); }
.ramp { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text-2); margin-top: 6px; }
.ramp canvas { width: 120px; height: 8px; border-radius: 2px; }
</style>
</head>
<body>
<main>
  <h1 id="title"></h1>
  <p class="sub" id="subtitle"></p>
  <div class="tiles" id="tiles"></div>
  <ul class="findings" id="findings" hidden></ul>
  <div class="grid" id="grid"></div>
</main>
<div id="tip" role="status"></div>
<script id="payload" type="application/json">/*DATA*/null</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("payload").textContent);
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const isDark = () => css("--surface-1").toLowerCase() === "#1a1a19";
// Sequential: one hue. On a dark surface the ramp runs the other way, so low stays quiet.
const RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281"];
const hex = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
function rampColor(t) {
  const stops = isDark() ? [...RAMP].reverse() : RAMP;
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const [a, b] = [hex(stops[i]), hex(stops[i + 1])];
  return a.map((v, k) => Math.round(v + (b[k] - v) * (x - i)));
}
const fmt = v => {
  if (v === null || v === undefined) return "없음";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return (+v.toPrecision(4)).toString();
};
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
const tip = document.getElementById("tip");
function showTip(x, y, rows) {
  tip.replaceChildren(...rows.map((r, i) => { const d = el("div"); if (i === 0) d.append(el("b", "", r)); else d.textContent = r; return d; }));
  tip.style.display = "block";
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = Math.min(innerWidth - w - 8, x + 14) + "px";
  tip.style.top = Math.max(8, Math.min(innerHeight - h - 8, y + 14)) + "px";
}
const hideTip = () => { tip.style.display = "none"; };
function card(title, wide) {
  const c = el("section", "card" + (wide ? " wide" : ""));
  const head = el("header");
  head.append(el("h2", "", title));
  c.append(head);
  document.getElementById("grid").append(c);
  return { c, head };
}
function tableOf(summary, headers, rows) {
  const d = el("details");
  d.append(el("summary", "", summary));
  const wrap = el("div", "scroll"), t = el("table");
  const hr = el("tr");
  headers.forEach(h => hr.append(el("th", "", h)));
  t.append(hr);
  rows.forEach(r => { const tr = el("tr"); r.forEach(v => tr.append(el("td", "", typeof v === "number" ? fmt(v) : v ?? "없음"))); t.append(tr); });
  wrap.append(t);
  d.append(wrap);
  return d;
}

function lineChart(host, series, opts) {
  const NS = "http://www.w3.org/2000/svg";
  const W = 560, H = 220, m = { l: 56, r: 12, t: 8, b: 26 };
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", opts.label);
  let log = !!opts.log;
  const wrap = el("div", "chart");
  wrap.append(svg);
  host.append(wrap);
  function draw() {
    svg.replaceChildren();
    const tx = v => (log ? Math.log10(v) : v);
    const usable = v => v !== null && (!log || v > 0);
    const ys = series.flatMap(s => s.values.filter(usable).map(tx));
    const xs = series.flatMap(s => s.steps);
    if (!ys.length) return;
    let [y0, y1] = [Math.min(...ys), Math.max(...ys)];
    if (y0 === y1) { y0 -= 1; y1 += 1; }
    const [x0, x1] = [Math.min(...xs), Math.max(...xs, 1)];
    const px = x => m.l + (x - x0) / Math.max(1, x1 - x0) * (W - m.l - m.r);
    const py = y => H - m.b - (tx(y) - y0) / (y1 - y0) * (H - m.t - m.b);
    const add = (name, attrs, text) => { const e = document.createElementNS(NS, name); for (const k in attrs) e.setAttribute(k, attrs[k]); if (text !== undefined) e.textContent = text; svg.append(e); return e; };
    for (let i = 0; i <= 4; i++) {
      const value = y0 + (y1 - y0) * i / 4;
      const y = H - m.b - i / 4 * (H - m.t - m.b);
      add("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, stroke: css("--grid"), "stroke-width": 1 });
      add("text", { x: m.l - 6, y: y + 4, "text-anchor": "end", "font-size": 11, fill: css("--text-3") }, fmt(log ? 10 ** value : value));
    }
    add("text", { x: m.l, y: H - 8, "font-size": 11, fill: css("--text-3") }, `스텝 ${x0}`);
    add("text", { x: W - m.r, y: H - 8, "text-anchor": "end", "font-size": 11, fill: css("--text-3") }, `${x1}`);
    series.forEach((s, k) => {
      let d = "", pen = false;
      s.values.forEach((v, i) => {
        if (!usable(v)) { pen = false; return; }
        d += `${pen ? "L" : "M"}${px(s.steps[i]).toFixed(1)} ${py(v).toFixed(1)}`;
        pen = true;
      });
      add("path", { d, fill: "none", stroke: css(`--series-${k + 1}`), "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" });
    });
    const cross = add("line", { y1: m.t, y2: H - m.b, stroke: css("--text-3"), "stroke-width": 1, visibility: "hidden" });
    const hit = add("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" });
    hit.addEventListener("pointermove", e => {
      const r = svg.getBoundingClientRect();
      const sx = (e.clientX - r.left) / r.width * W;
      const rows = [];
      let at = null;
      series.forEach(s => {
        let best = 0;
        s.steps.forEach((t, i) => { if (Math.abs(px(t) - sx) < Math.abs(px(s.steps[best]) - sx)) best = i; });
        at = s.steps[best];
        rows.push(`${s.name}: ${fmt(s.values[best])}`);
      });
      cross.setAttribute("x1", px(at)); cross.setAttribute("x2", px(at)); cross.setAttribute("visibility", "visible");
      showTip(e.clientX, e.clientY, [`스텝 ${at}`, ...rows]);
    });
    hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  }
  draw();
  if (series.length > 1) {
    const lg = el("div", "legend");
    series.forEach((s, k) => { const i = el("span"); const sw = el("i"); sw.style.background = `var(--series-${k + 1})`; i.append(sw, s.name); lg.append(i); });
    host.append(lg);
  }
  return { toggleLog() { log = !log; draw(); return log; }, draw };
}

function rampLegend(host, lo, hi, unit) {
  const r = el("div", "ramp");
  const c = document.createElement("canvas");
  c.width = 120; c.height = 8;
  const paint = () => {
    const g = c.getContext("2d");
    for (let x = 0; x < 120; x++) { g.fillStyle = `rgb(${rampColor(x / 119)})`; g.fillRect(x, 0, 1, 8); }
  };
  paint();
  r.append(el("span", "", lo), c, el("span", "", hi));
  if (unit) r.append(el("span", "", unit));
  host.append(r);
  return paint;
}

// Rows are parameters and columns are time: one cell per parameter per sampled step.
function heatmap(host, table, opts) {
  const names = Object.keys(table);
  if (!names.length) return null;
  const cols = table[names[0]].steps.length;
  const cell = 18;
  const H = names.length * cell + 8;
  let W = 960, labelW = 190;
  // Names usually share a prefix like `params.`, which only takes room from the part that differs.
  let shared = names.length > 1 ? names[0].slice(0, names[0].lastIndexOf(".") + 1) : "";
  while (shared && !names.every(n => n.startsWith(shared))) shared = shared.slice(0, shared.slice(0, -1).lastIndexOf(".") + 1);
  const c = document.createElement("canvas");
  c.setAttribute("role", "img");
  c.setAttribute("aria-label", opts.label);
  const wrap = el("div", "chart");
  wrap.append(c);
  host.append(wrap);
  const tx = v => (opts.log ? Math.log10(Math.max(v, 1e-30)) : v);
  const all = names.flatMap(n => table[n].values.filter(v => v !== null && (!opts.log || v > 0))).map(tx).sort((a, b) => a - b);
  // Percentiles: one wild step, like the first update of a small parameter, must not wash out the rest.
  const lo = all.length ? all[Math.floor(0.02 * (all.length - 1))] : 0;
  const hi = all.length ? all[Math.ceil(0.98 * (all.length - 1))] : 1;
  const clipped = all.length && (all[0] < lo || all[all.length - 1] > hi);
  function paint() {
    // Drawn at the real width, so labels stay readable instead of shrinking with the page.
    W = Math.max(240, wrap.clientWidth || 960);
    labelW = Math.min(190, Math.round(W * 0.36));
    const dpr = devicePixelRatio || 1;
    c.width = W * dpr; c.height = H * dpr;
    c.style.aspectRatio = `${W} / ${H}`;
    const g = c.getContext("2d");
    g.scale(dpr, dpr);
    g.clearRect(0, 0, W, H);
    g.font = "11px sans-serif"; g.textBaseline = "middle";
    const cw = (W - labelW) / cols;
    names.forEach((n, j) => {
      g.fillStyle = css("--text-2");
      const short = n.slice(shared.length);
      const fit = Math.floor(labelW / 6.2);
      const shown = short.length > fit ? "…" + short.slice(1 - fit) : short;
      g.fillText(shown, 0, j * cell + cell / 2);
      table[n].values.forEach((v, i) => {
        if (v === null || (opts.log && v <= 0)) { g.fillStyle = css("--grid"); }
        else { g.fillStyle = `rgb(${rampColor(hi === lo ? 0.5 : (tx(v) - lo) / (hi - lo))})`; }
        g.fillRect(labelW + i * cw, j * cell + 1, cw + 0.5, cell - 2);
      });
    });
  }
  paint();
  let seenWidth = wrap.clientWidth;
  new ResizeObserver(() => { if (wrap.clientWidth !== seenWidth) { seenWidth = wrap.clientWidth; paint(); } }).observe(wrap);
  c.addEventListener("pointermove", e => {
    const r = c.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width * W, y = (e.clientY - r.top) / r.height * H;
    const j = Math.floor(y / cell), i = Math.floor((x - labelW) / ((W - labelW) / cols));
    if (j < 0 || j >= names.length || i < 0 || i >= cols) return hideTip();
    showTip(e.clientX, e.clientY, [names[j], `스텝 ${table[names[j]].steps[i]}`, `${opts.value}: ${fmt(table[names[j]].values[i])}`]);
  });
  c.addEventListener("pointerleave", hideTip);
  const repaintRamp = rampLegend(host, opts.log ? fmt(10 ** lo) : fmt(lo), opts.log ? fmt(10 ** hi) : fmt(hi), [opts.log ? "로그 척도" : "", clipped ? "범위 밖은 끝 색" : ""].filter(Boolean).join(", "));
  host.append(tableOf("마지막 값 표", ["파라미터", opts.value], names.map(n => { const v = table[n].values; return [n, v[v.length - 1]]; })));
  return () => { paint(); repaintRamp(); };
}

function landscapeScale(L) {
  const vals = L.values.flat().filter(v => v !== null).sort((a, b) => a - b);
  if (!vals.length) return null;
  // Steep walls would flatten everything else, so color saturates at the 95th percentile.
  return { lo: vals[0], hi: vals[Math.floor(0.95 * (vals.length - 1))], max: vals[vals.length - 1] };
}

function contour(host, L, scale) {
  const W = 480, H = 400, m = { l: 58, r: 10, t: 8, b: 34 };
  const c = document.createElement("canvas");
  c.setAttribute("role", "img");
  c.setAttribute("aria-label", "손실 지형 등고선");
  const wrap = el("div", "chart");
  wrap.append(c);
  host.append(wrap);
  const nx = L.alphas.length, ny = L.betas.length;
  const px = a => m.l + (a - L.alphas[0]) / (L.alphas[nx - 1] - L.alphas[0]) * (W - m.l - m.r);
  const py = b => H - m.b - (b - L.betas[0]) / (L.betas[ny - 1] - L.betas[0]) * (H - m.t - m.b);
  const norm = v => (scale.hi === scale.lo ? 0.5 : (v - scale.lo) / (scale.hi - scale.lo));
  function paint() {
    const dpr = devicePixelRatio || 1;
    c.width = W * dpr; c.height = H * dpr;
    c.style.aspectRatio = `${W} / ${H}`;
    const g = c.getContext("2d");
    g.scale(dpr, dpr);
    const cw = (W - m.l - m.r) / (nx - 1), ch = (H - m.t - m.b) / (ny - 1);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const v = L.values[j][i];
      g.fillStyle = v === null ? css("--grid") : `rgb(${rampColor(norm(v))})`;
      g.fillRect(px(L.alphas[i]) - cw / 2, py(L.betas[j]) - ch / 2, cw + 1, ch + 1);
    }
    // Iso-lines by marching squares, on the same scale as the color.
    g.strokeStyle = isDark() ? "rgba(255,255,255,.35)" : "rgba(0,0,0,.3)";
    g.lineWidth = 1;
    for (let level = 1; level < 8; level++) {
      const iso = scale.lo + (scale.hi - scale.lo) * level / 8;
      g.beginPath();
      for (let j = 0; j + 1 < ny; j++) for (let i = 0; i + 1 < nx; i++) {
        const q = [L.values[j][i], L.values[j][i + 1], L.values[j + 1][i + 1], L.values[j + 1][i]];
        if (q.some(v => v === null)) continue;
        const corners = [[L.alphas[i], L.betas[j]], [L.alphas[i + 1], L.betas[j]], [L.alphas[i + 1], L.betas[j + 1]], [L.alphas[i], L.betas[j + 1]]];
        const pts = [];
        for (let e = 0; e < 4; e++) {
          const [a, b] = [q[e], q[(e + 1) % 4]];
          if ((a < iso) !== (b < iso)) {
            const t = (iso - a) / (b - a);
            const [p, r] = [corners[e], corners[(e + 1) % 4]];
            pts.push([px(p[0] + (r[0] - p[0]) * t), py(p[1] + (r[1] - p[1]) * t)]);
          }
        }
        for (let k = 0; k + 1 < pts.length; k += 2) { g.moveTo(...pts[k]); g.lineTo(...pts[k + 1]); }
      }
      g.stroke();
    }
    g.fillStyle = css("--text-3"); g.font = "11px sans-serif"; g.textBaseline = "alphabetic";
    g.textAlign = "center";
    g.fillText(fmt(L.alphas[0]), m.l + 14, H - m.b + 15);
    g.fillText(fmt(L.alphas[nx - 1]), W - m.r - 14, H - m.b + 15);
    g.fillText("방향 1", (m.l + W - m.r) / 2, H - 6);
    g.textAlign = "right";
    g.fillText(fmt(L.betas[0]), m.l - 7, H - m.b);
    g.fillText(fmt(L.betas[ny - 1]), m.l - 7, m.t + 10);
    g.save(); g.translate(10, (m.t + H - m.b) / 2); g.rotate(-Math.PI / 2); g.textAlign = "center"; g.fillText("방향 2", 0, 0); g.restore();
    if (L.trajectory && L.trajectory.length) {
      const ring = css("--mark-ring"), line = css("--series-2");
      const path = () => { g.beginPath(); L.trajectory.forEach((p, i) => (i ? g.lineTo(px(p.x), py(p.y)) : g.moveTo(px(p.x), py(p.y)))); };
      path(); g.strokeStyle = ring; g.lineWidth = 4; g.lineJoin = "round"; g.stroke();
      path(); g.strokeStyle = line; g.lineWidth = 2; g.stroke();
      const dot = (p, r, label) => {
        g.beginPath(); g.arc(px(p.x), py(p.y), r + 2, 0, 7); g.fillStyle = ring; g.fill();
        g.beginPath(); g.arc(px(p.x), py(p.y), r, 0, 7); g.fillStyle = line; g.fill();
        g.font = "600 11px sans-serif"; g.textAlign = px(p.x) > W - 60 ? "right" : "left"; g.lineWidth = 3; g.strokeStyle = ring;
        const tx = px(p.x) + (px(p.x) > W - 60 ? -9 : 9), ty = py(p.y) - 7;
        g.strokeText(label, tx, ty); g.fillStyle = css("--text-1"); g.fillText(label, tx, ty);
      };
      dot(L.trajectory[0], 4, "시작");
      dot(L.trajectory[L.trajectory.length - 1], 4, "끝");
    }
  }
  paint();
  c.addEventListener("pointermove", e => {
    const r = c.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width * W, y = (e.clientY - r.top) / r.height * H;
    if (L.trajectory) {
      const near = L.trajectory.find(p => Math.hypot(px(p.x) - x, py(p.y) - y) < 7);
      if (near) return showTip(e.clientX, e.clientY, [`스텝 ${near.step}`, `실제 손실: ${fmt(near.loss)}`, "경로가 평면을 벗어나면 표면과 다를 수 있음"]);
    }
    const i = Math.round((x - m.l) / (W - m.l - m.r) * (nx - 1)), j = Math.round((H - m.b - y) / (H - m.t - m.b) * (ny - 1));
    if (i < 0 || j < 0 || i >= nx || j >= ny) return hideTip();
    showTip(e.clientX, e.clientY, [`손실: ${fmt(L.values[j][i])}`, `방향 1: ${fmt(L.alphas[i])}`, `방향 2: ${fmt(L.betas[j])}`]);
  });
  c.addEventListener("pointerleave", hideTip);
  return paint;
}

function surface(host, L, scale) {
  const W = 480, H = 400;
  const c = document.createElement("canvas");
  c.tabIndex = 0;
  c.setAttribute("role", "img");
  c.setAttribute("aria-label", "손실 지형 3D 표면. 끌거나 방향키로 회전");
  c.style.cursor = "grab";
  const wrap = el("div", "chart");
  wrap.append(c);
  host.append(wrap);
  let yaw = -0.75, pitch = 0.75;
  const nx = L.alphas.length, ny = L.betas.length;
  const hgt = v => { const t = scale.hi === scale.lo ? 0.5 : (Math.min(v, scale.hi) - scale.lo) / (scale.hi - scale.lo); return t; };
  function project(u, v, h) {
    const [x, y, z] = [u - 0.5, v - 0.5, (h - 0.5) * 0.6];
    const rx = x * Math.cos(yaw) - y * Math.sin(yaw), ry = x * Math.sin(yaw) + y * Math.cos(yaw);
    const sy = ry * Math.sin(pitch) - z * Math.cos(pitch);
    return [W / 2 + rx * W * 0.72, H / 2 + sy * H * 0.82, ry * Math.cos(pitch) + z * Math.sin(pitch)];
  }
  function paint() {
    const dpr = devicePixelRatio || 1;
    c.width = W * dpr; c.height = H * dpr;
    c.style.aspectRatio = `${W} / ${H}`;
    const g = c.getContext("2d");
    g.scale(dpr, dpr);
    const quads = [];
    for (let j = 0; j + 1 < ny; j++) for (let i = 0; i + 1 < nx; i++) {
      const q = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]].map(([a, b]) => L.values[b][a]);
      if (q.some(v => v === null)) continue;
      const pts = [[i, j, q[0]], [i + 1, j, q[1]], [i + 1, j + 1, q[2]], [i, j + 1, q[3]]].map(([a, b, v]) => project(a / (nx - 1), b / (ny - 1), hgt(v)));
      quads.push({ pts, depth: pts.reduce((s, p) => s + p[2], 0) / 4, t: q.reduce((s, v) => s + hgt(v), 0) / 4 });
    }
    quads.sort((a, b) => b.depth - a.depth);
    const edge = css("--surface-1");
    quads.forEach(q => {
      g.beginPath();
      q.pts.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1])));
      g.closePath();
      g.fillStyle = `rgb(${rampColor(q.t)})`;
      g.fill();
      g.strokeStyle = edge; g.globalAlpha = 0.35; g.lineWidth = 0.5; g.stroke(); g.globalAlpha = 1;
    });
    if (L.trajectory && L.trajectory.length) {
      const a0 = L.alphas[0], a1 = L.alphas[nx - 1], b0 = L.betas[0], b1 = L.betas[ny - 1];
      const pos = p => project((p.x - a0) / (a1 - a0), (p.y - b0) / (b1 - b0), hgt(p.loss ?? scale.hi));
      const path = () => { g.beginPath(); L.trajectory.forEach((p, i) => { const [x, y] = pos(p); return i ? g.lineTo(x, y) : g.moveTo(x, y); }); };
      path(); g.strokeStyle = css("--mark-ring"); g.lineWidth = 4; g.lineJoin = "round"; g.stroke();
      path(); g.strokeStyle = css("--series-2"); g.lineWidth = 2; g.stroke();
      const [ex, ey] = pos(L.trajectory[L.trajectory.length - 1]);
      g.beginPath(); g.arc(ex, ey, 6, 0, 7); g.fillStyle = css("--mark-ring"); g.fill();
      g.beginPath(); g.arc(ex, ey, 4, 0, 7); g.fillStyle = css("--series-2"); g.fill();
    }
  }
  paint();
  let drag = null;
  c.addEventListener("pointerdown", e => { drag = [e.clientX, e.clientY]; c.setPointerCapture(e.pointerId); c.style.cursor = "grabbing"; });
  c.addEventListener("pointermove", e => {
    if (!drag) return;
    yaw += (e.clientX - drag[0]) * 0.01;
    pitch = Math.max(0.15, Math.min(1.45, pitch + (e.clientY - drag[1]) * 0.01));
    drag = [e.clientX, e.clientY];
    paint();
  });
  c.addEventListener("pointerup", () => { drag = null; c.style.cursor = "grab"; });
  c.addEventListener("keydown", e => {
    const d = { ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, -0.1], ArrowDown: [0, 0.1] }[e.key];
    if (!d) return;
    e.preventDefault();
    yaw += d[0];
    pitch = Math.max(0.15, Math.min(1.45, pitch + d[1]));
    paint();
  });
  return paint;
}

const repaints = [];
function renderRun(run) {
  const tiles = document.getElementById("tiles");
  const lossKeys = Object.keys(run.losses);
  const first = lossKeys.length ? run.losses[lossKeys[0]].values.filter(v => v !== null) : [];
  const tile = (label, value) => { const t = el("div", "tile"); t.append(el("div", "label", label), el("div", "value", value)); tiles.append(t); };
  tile("상태", { completed: "완료", stopped_nonfinite: "NaN·Inf로 중단", stopped_by_callback: "콜백이 중단", running: "진행 중" }[run.status] ?? run.status);
  tile("스텝", run.steps.toLocaleString());
  if (first.length) { tile("마지막 손실", fmt(first[first.length - 1])); tile("최저 손실", fmt(Math.min(...first))); }
  if (run.global_norm) { const v = run.global_norm.values.filter(x => x !== null); if (v.length) tile("마지막 gradient 노름", fmt(v[v.length - 1])); }
  if (run.findings.length) {
    const ul = document.getElementById("findings");
    ul.hidden = false;
    run.findings.forEach(f => {
      const li = el("li");
      li.append(el("span", "tag" + (f.certainty === "verified" ? " critical" : ""), f.certainty));
      li.append(el("code", "", f.code));
      li.append(document.createTextNode(` ${f.parameter ?? ""} ${JSON.stringify(f.evidence)}`));
      ul.append(li);
    });
  }
  if (lossKeys.length) {
    const { c, head } = card("손실");
    const shown = lossKeys.slice(0, 8);
    const chart = lineChart(c, shown.map(k => ({ name: k, ...run.losses[k] })), { label: "스텝별 손실", log: false });
    const b = el("button", "", "로그 척도"); b.setAttribute("aria-pressed", "false");
    b.onclick = () => { b.setAttribute("aria-pressed", String(chart.toggleLog())); };
    head.append(b);
    repaints.push(chart.draw);
    const v = run.losses[shown[0]];
    c.append(tableOf("표로 보기", ["스텝", shown[0]], v.steps.slice(0, 200).map((s, i) => [String(s), v.values[i]])));
  }
  if (run.global_norm) {
    const { c } = card("전역 gradient 노름");
    const chart = lineChart(c, [{ name: "노름", ...run.global_norm }], { label: "스텝별 gradient 노름", log: true });
    repaints.push(chart.draw);
    c.append(el("p", "note", "로그 척도입니다."));
  }
  const heat = (title, table, opts, note) => {
    if (!Object.keys(table).length) return;
    const { c } = card(title, true);
    const paint = heatmap(c, table, opts);
    if (paint) repaints.push(paint);
    if (note) c.append(el("p", "note", note));
  };
  heat("파라미터별 gradient 노름", run.grad_norm, { label: "파라미터별 스텝별 gradient 노름", log: true, value: "노름" }, "행은 파라미터, 열은 시간입니다. 층을 따라 색이 옅어지면 gradient가 전달되지 않는 신호일 수 있습니다.");
  heat("파라미터별 업데이트 비율", run.update, { label: "파라미터별 스텝별 업데이트 비율", log: true, value: "비율" }, "‖변화량‖ / ‖파라미터‖입니다. 학습률이 파라미터 크기에 비해 적절한지 보여 줍니다.");
  if (Object.values(run.grad_zero).some(s => s.values.some(v => v))) {
    heat("gradient가 정확히 0인 비율", run.grad_zero, { label: "파라미터별 0 gradient 비율", log: false, value: "비율" }, "쓰이지 않는 토큰이나 죽은 유닛이 있으면 높아집니다.");
  }
}
function renderLandscape(L) {
  const scale = landscapeScale(L);
  if (!scale) return;
  const kind = { pca: "학습 궤적의 주성분 평면", random: "무작위 방향 평면" }[L.kind] ?? L.kind;
  const a = card(`손실 지형: ${kind}`);
  repaints.push(contour(a.c, L, scale));
  const paintRamp = rampLegend(a.c, fmt(scale.lo), fmt(scale.hi), scale.max > scale.hi ? `이상은 같은 색 (최대 ${fmt(scale.max)})` : "");
  repaints.push(paintRamp);
  if (L.trajectory) a.c.append(el("p", "note", `주황색 선은 학습 경로입니다. 이 평면이 움직임의 ${(100 * (L.explained[0] + L.explained[1])).toFixed(1)}%를 설명합니다(방향 1 ${(100 * L.explained[0]).toFixed(1)}%, 방향 2 ${(100 * L.explained[1]).toFixed(1)}%).`));
  else a.c.append(el("p", "note", "무작위 단면 하나이므로 전체 지형의 특성이라고 일반화할 수 없습니다."));
  const b = card("3D 표면");
  repaints.push(surface(b.c, L, scale));
  b.c.append(el("p", "note", "끌거나 방향키로 돌려 볼 수 있습니다."));
  if (L.trajectory) a.c.append(tableOf("경로를 표로 보기", ["스텝", "방향 1", "방향 2", "손실"], L.trajectory.map(p => [String(p.step), p.x, p.y, p.loss])));
}
document.getElementById("title").textContent = DATA.title;
document.title = DATA.title;
document.getElementById("subtitle").textContent = DATA.run && DATA.landscape ? "학습 기록과 손실 지형" : DATA.run ? "학습 기록" : "손실 지형";
if (DATA.run) renderRun(DATA.run);
if (DATA.landscape) renderLandscape(DATA.landscape);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => repaints.forEach(f => f()));
addEventListener("scroll", hideTip, { passive: true });
</script>
</body>
</html>
"""
