/* Iceberg site: renders every section from data/results.json (see site/build_data.py). */
"use strict";

const CFG = window.ICEBERG_CONFIG || {};

// Colours match demo/visualize.py. Trigger / boundary text follows the task's class conventions.
const CLASS_META = {
  accident: ["#E5484D", "Contact between road users, or with a fixed object.", "First contact to everyone involved stopping or leaving the frame."],
  near_miss: ["#F76B15", "Sharp braking or swerving to avoid a collision, with no contact.", "Onset of the evasive action to the users being clear of each other."],
  red_light: ["#D6409F", "A vehicle's front crosses the stop line while its signal is red.", "Front crosses the line to leaving the intersection or frame."],
  wrong_way: ["#8E4EC6", "Moving against the lane direction, including into the oncoming lane.", "Entering the opposing lane to returning or leaving."],
  illegal_u_turn: ["#3E63DD", "A U-turn where U-turns are not allowed.", "Start of the turn to its completion."],
  stopped_vehicle: ["#8D8D8D", "Stationary on the carriageway for 10 s or more, not queued at a signal.", "Vehicle stops to moving again or being removed."],
  jaywalking: ["#FFC53D", "A pedestrian on the carriageway outside a crossing.", "Steps onto the road to leaving it."],
  failure_to_yield: ["#BDEE63", "A vehicle drives through a crossing while a pedestrian is on it or stepping on.", "Vehicle enters the crossing to leaving it."],
  illegal_turn: ["#0090FF", "A turn from the wrong lane or in a prohibited direction.", "Start of the turn to its completion."],
  solid_line_crossing: ["#12A594", "A lane change across a solid line.", "Wheel crosses the line to being fully in the new lane."],
  stop_line: ["#46A758", "Stopping past the stop line on red without entering the intersection.", "Vehicle stops to the signal turning green."],
  congestion: ["#978365", "Standstill or crawling traffic across all lanes of one direction.", "Queue stops moving to the queue clearing."],
  road_obstacle: ["#A18072", "Debris, an animal or a fallen object on the carriageway.", "Object appears to its removal."],
  fire_smoke: ["#FF977D", "Visible fire or smoke from a vehicle or on the road.", "First smoke to it clearing or the video ending."],
};
const color = (c) => (CLASS_META[c] || ["#888"])[0];
const pretty = (c) => String(c).replace(/_/g, " ");
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const fmtT = (s) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
const fmtDur = (s) => (s >= 60 ? `${Math.floor(s / 60)} min ${Math.round(s % 60)} s` : `${s.toFixed(1)} s`);
const num = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x) ? null : Number(x).toFixed(d));
const $ = (id) => document.getElementById(id);
const STATUS_TEXT = { tp: "matches a label", boundary: "boundaries off", fp: "no matching label" };

/** Tiny DOM builder: h("a", {href}, "text", child). Text is never parsed as HTML. */
function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "style") el.style.cssText = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid !== null && kid !== undefined && kid !== false) el.append(kid);
  return el;
}
const SVG_NS = "http://www.w3.org/2000/svg";
function s(tag, attrs = {}, ...kids) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) el.setAttribute(k, v);
  for (const kid of kids.flat()) if (kid) el.append(kid);
  return el;
}
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/* ------------------------------------------------------------------ lanes timeline */
function niceStep(duration, px) {
  const target = duration / Math.max(2, px / 90);
  return [1, 2, 5, 10, 15, 20, 30, 60, 120, 300, 600].find((v) => v >= target) || 1200;
}

/**
 * Road-style timeline: one lane per class, lane markings between lanes.
 * Returns {setTime(t)} so a player can move the playhead.
 */
function renderLanes(el, video, opts = {}) {
  const { showGT = false, onSeek = null } = opts;
  el.replaceChildren();
  const width = Math.max(300, el.clientWidth);
  const narrow = width < 560;
  const labelW = narrow ? 104 : 158;
  const laneH = showGT ? 34 : narrow ? 24 : 28;
  const gt = showGT ? video.ground_truth || [] : [];
  const present = new Set([...video.events.map((e) => e.label), ...gt.map((g) => g[2])]);
  const classes = (DATA.classes || Object.keys(CLASS_META)).filter((c) => present.has(c));
  if (!classes.length) {
    el.append(h("p", { class: "empty", style: "padding:12px 16px;margin:0" }, "No events in this video."));
    return { setTime() {} };
  }
  const top = 6, axisH = 26;
  const height = top + classes.length * laneH + axisH;
  const x0 = labelW + 8, x1 = width - 14, dur = Math.max(1, video.duration);
  const X = (t) => x0 + (Math.min(Math.max(t, 0), dur) / dur) * (x1 - x0);
  const svg = s("svg", { viewBox: `0 0 ${width} ${height}`, width, height, role: "group" });

  svg.append(s("line", { class: "lane-edge", x1: x0, x2: x1, y1: top, y2: top }));
  classes.forEach((c, i) => {
    const y = top + i * laneH;
    const label = s("text", { class: "lane-label", x: labelW, y: y + laneH / 2 + 4, "text-anchor": "end" });
    label.textContent = narrow && c.length > 14 ? pretty(c).replace("crossing", "cross.") : pretty(c);
    svg.append(label);
    svg.append(s("line", { class: i === classes.length - 1 ? "lane-edge" : "lane-sep", x1: x0, x2: x1, y1: y + laneH, y2: y + laneH }));
  });

  // axis
  const axis = s("g", { class: "axis" });
  const step = niceStep(dur, x1 - x0);
  for (let t = 0; t <= dur + 1e-6; t += step) {
    const x = X(t), yb = top + classes.length * laneH;
    axis.append(s("line", { x1: x, x2: x, y1: yb, y2: yb + 5 }));
    const tx = s("text", { x, y: yb + 19, "text-anchor": t === 0 ? "start" : "middle" });
    tx.textContent = t >= 60 ? fmtT(t).replace(/\.0$/, "") : `${t}s`;
    axis.append(tx);
  }
  svg.append(axis);

  // click anywhere on the lanes to seek
  if (onSeek) {
    const hit = s("rect", { class: "hit", x: x0, y: top, width: x1 - x0, height: classes.length * laneH });
    hit.addEventListener("click", (ev) => {
      const r = svg.getBoundingClientRect();
      const x = ((ev.clientX - r.left) / r.width) * width;
      onSeek(((x - x0) / (x1 - x0)) * dur);
    });
    svg.append(hit);
  }

  const barH = showGT ? 13 : laneH - 10;
  gt.forEach(([st, en, lab]) => {
    const i = classes.indexOf(lab);
    const g = s("g", { class: "gt" });
    g.append(s("rect", { x: X(st), y: top + i * laneH + 5 + barH + 3, width: Math.max(3, X(en) - X(st)), height: 8, stroke: color(lab), rx: 2 }));
    const title = s("title"); title.textContent = `Our label: ${pretty(lab)} ${fmtT(st)} to ${fmtT(en)}`;
    g.append(title);
    svg.append(g);
  });
  video.events.forEach((e) => {
    const i = classes.indexOf(e.label);
    const st = showGT ? e.status : null;
    const g = s("g", { class: `ev${st ? " ev-" + st : ""}`, tabindex: onSeek ? 0 : null, role: onSeek ? "button" : null,
      "aria-label": `${pretty(e.label)} from ${fmtT(e.start)} to ${fmtT(e.end)}${st ? ", " + STATUS_TEXT[st] : ""}` });
    const w = Math.max(4, X(e.end) - X(e.start));
    const bar = s("rect", { class: "bar", x: X(e.start), y: top + i * laneH + 5, width: w, height: barH, fill: color(e.label),
      stroke: color(e.label), "fill-opacity": st === "boundary" ? 0.55 : null });
    g.append(bar);
    const title = s("title");
    title.textContent = `${cap(pretty(e.label))}: ${fmtT(e.start)} to ${fmtT(e.end)}${st ? " (" + STATUS_TEXT[st] + ")" : ""}`;
    g.append(title);
    if (onSeek) {
      g.addEventListener("click", () => onSeek(e.start, e));
      g.addEventListener("keydown", (k) => { if (k.key === "Enter" || k.key === " ") { k.preventDefault(); onSeek(e.start, e); } });
    }
    svg.append(g);
  });

  const head = s("line", { class: "playhead", x1: x0, x2: x0, y1: top - 4, y2: top + classes.length * laneH + 4, visibility: onSeek ? "visible" : "hidden" });
  svg.append(head);
  el.append(svg);
  return {
    setTime(t) { const x = X(t); head.setAttribute("x1", x); head.setAttribute("x2", x); },
  };
}

/* ------------------------------------------------------------------ Plotly helpers */
const CHARTS = []; // re-rendered on theme change and resize
function chart(el, build) {
  const node = typeof el === "string" ? $(el) : el;
  if (!node) return;
  const draw = () => {
    if (!window.Plotly) {
      node.replaceChildren(h("p", { class: "empty" }, "Charts need Plotly from cdn.jsdelivr.net, which did not load."));
      return;
    }
    const { traces, layout, onClick } = build();
    Plotly.react(node, traces, baseLayout(layout), { displayModeBar: false, responsive: true });
    if (onClick && !node._clickBound) {
      node.on("plotly_click", (ev) => onClick(ev));
      node._clickBound = true;
    }
  };
  node._draw = draw;
  CHARTS.push(draw);
  draw();
}
function baseLayout(extra = {}) {
  const ink2 = cssVar("--ink-2"), rule = cssVar("--rule");
  const axis = { gridcolor: rule, zerolinecolor: rule, linecolor: rule, tickfont: { color: ink2 }, title: { font: { color: ink2 } }, automargin: true };
  return {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Barlow, system-ui, sans-serif", size: 13, color: ink2 },
    margin: { l: 10, r: 10, t: 10, b: 40 },
    hoverlabel: { font: { family: "Barlow, system-ui, sans-serif" } },
    legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom", font: { color: ink2 } },
    ...extra,
    xaxis: { ...axis, ...(extra.xaxis || {}) },
    yaxis: { ...axis, ...(extra.yaxis || {}) },
  };
}
function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
}

/* ------------------------------------------------------------------ results section */
let DATA = null;
const R = { idx: 0, lanes: null, virtual: 0, playable: false, lastRisk: 0, pending: null };

function currentVideo() { return DATA.videos[R.idx]; }

function initResults() {
  const tabs = $("videoTabs");
  DATA.videos.forEach((v, i) => {
    tabs.append(h("button", { class: "tab", role: "tab", type: "button", "aria-selected": i === 0 ? "true" : "false", onclick: () => selectVideo(i) },
      v.id, h("small", {}, `${v.title && v.title !== v.id ? v.title + ", " : ""}${fmtDur(v.duration)}`)));
  });
  const video = $("player");
  video.addEventListener("error", () => {
    setPlayable(false);
    if (R.pending != null) { const t = R.pending; R.pending = null; seek(t); }  // fall back to the virtual clock
  });
  video.addEventListener("loadeddata", () => setPlayable(true));
  video.addEventListener("timeupdate", () => onTime(video.currentTime));
  video.addEventListener("seeked", () => onTime(video.currentTime));
  // smooth playhead while playing; no animation loop when paused
  const loop = () => { if (!video.paused && !video.ended) { onTime(video.currentTime); requestAnimationFrame(loop); } };
  video.addEventListener("play", () => requestAnimationFrame(loop));
  $("gtToggle").addEventListener("change", drawLanes);
  onWidthChange($("resultLanes"), drawLanes);
  selectVideo(0);
}

function setPlayable(ok) {
  R.playable = ok;
  $("videoMissing").hidden = ok;
}

function selectVideo(i) {
  R.idx = i;
  const v = currentVideo();
  [...$("videoTabs").children].forEach((b, j) => b.setAttribute("aria-selected", j === i ? "true" : "false"));
  const video = $("player");
  video.pause();
  if (v.media) {
    video.poster = v.poster || "";
    video.src = v.media;
    setPlayable(true);
    video.load();
  } else {
    video.removeAttribute("src");
    setPlayable(false);
  }
  const bits = [v.width && v.height ? `${v.width} x ${v.height}` : null, v.fps ? `${v.fps} fps` : null, fmtDur(v.duration),
    v.recorded_at ? `recorded ${v.recorded_at.replace("T", " ").slice(0, 16)}` : null].filter(Boolean);
  $("videoMeta").textContent = bits.join(", ");
  R.virtual = 0;
  drawEventList();
  drawLanes();
  drawRisk();
  drawStats();
  onTime(0);
}

function drawEventList() {
  const v = currentVideo();
  const list = $("eventList");
  list.replaceChildren();
  $("eventCount").textContent = `(${v.events.length})`;
  if (!v.events.length) list.append(h("li", { class: "empty", style: "padding:10px" }, "No events detected."));
  v.events.forEach((e, k) => {
    list.append(h("li", {}, h("button", { type: "button", "data-k": k, onclick: () => seek(e.start) },
      h("span", { class: "dot", style: `background:${color(e.label)}` }),
      h("span", {}, cap(pretty(e.label)), e.status ? h("span", { class: "st" }, STATUS_TEXT[e.status]) : null),
      h("span", { class: "time" }, `${fmtT(e.start)} to ${fmtT(e.end)}`))));
  });
}

function drawLanes() {
  if (!DATA) return;
  R.lanes = renderLanes($("resultLanes"), currentVideo(), { showGT: $("gtToggle").checked && hasGT(), onSeek: (t) => seek(t) });
  R.lanes.setTime(R.playable ? $("player").currentTime : R.virtual);
}
const hasGT = () => DATA.videos.some((v) => (v.ground_truth || []).length);

function drawRisk() {
  const v = currentVideo();
  const el = $("riskChart");
  if (!v.risk || !v.risk.length) {
    if (window.Plotly) Plotly.purge(el);
    el._clickBound = false;
    el.replaceChildren(h("p", { class: "empty", style: "padding:16px" }, "No risk curve for this video yet."));
    return;
  }
  if (el._draw) { el.replaceChildren(); el._draw(); return; }
  chart(el, () => {
    const vv = currentVideo();
    const spans = vv.events.filter((e) => e.label === "accident" || e.label === "near_miss");
    const shapes = spans.map((e) => ({ type: "rect", xref: "x", yref: "paper", x0: e.start, x1: e.end, y0: 0, y1: 1,
      fillcolor: hexA(color(e.label), 0.16), line: { width: 0 }, layer: "below" }));
    shapes.push({ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0.5, y1: 0.5, line: { color: "#E5484D", width: 1.2, dash: "dash" } });
    // playhead: always the last shape (moved by onTime)
    shapes.push({ type: "line", xref: "x", yref: "paper", x0: R.lastRisk, x1: R.lastRisk, y0: 0, y1: 1, line: { color: cssVar("--ink"), width: 1.5 } });
    const annotations = spans.map((e) => ({ x: e.start, y: 1, yref: "paper", xanchor: "left", yanchor: "top", text: pretty(e.label),
      showarrow: false, font: { size: 12, color: color(e.label) } }));
    return {
      traces: [{ x: vv.risk.map((p) => p[0]), y: vv.risk.map((p) => p[1]), type: "scatter", mode: "lines", line: { color: "#F2C230", width: 2 },
        fill: "tozeroy", fillcolor: "rgba(242,194,48,0.14)", hovertemplate: "%{x:.1f} s: %{y:.2f}<extra></extra>" }],
      layout: { height: 240, shapes, annotations, margin: { l: 10, r: 14, t: 14, b: 36 },
        xaxis: { title: { text: "time (s)" }, range: [0, vv.duration] }, yaxis: { title: { text: "risk" }, range: [0, 1.02], fixedrange: true } },
      onClick: (ev) => ev.points && ev.points[0] && seek(ev.points[0].x),
    };
  });
}

function drawStats() {
  const st = currentVideo().stats || {};
  const items = [["Tracked objects", st.tracks], ["Vehicles", st.vehicles], ["Pedestrians", st.pedestrians], ["Two-wheelers", st.two_wheelers],
    ["Mean speed", st.mean_speed_kmh != null ? `${st.mean_speed_kmh} km/h` : null], ["Processing time", st.runtime_x_realtime != null ? `${st.runtime_x_realtime}x video` : null]]
    .filter(([, v]) => v !== null && v !== undefined);
  const dl = $("videoStats");
  dl.replaceChildren(...items.map(([k, v]) => h("div", {}, h("dt", {}, k), h("dd", {}, String(v)))));
  dl.hidden = !items.length;
}

let riskTimer = 0;
function onTime(t) {
  if (!R.lanes) return;
  R.lanes.setTime(t);
  const v = currentVideo();
  [...$("eventList").querySelectorAll("button")].forEach((b) => {
    const e = v.events[+b.dataset.k];
    b.classList.toggle("active", t >= e.start && t <= e.end);
  });
  const now = performance.now();
  if (window.Plotly && now - riskTimer > 250 && $("riskChart").data) {
    riskTimer = now;
    R.lastRisk = t;
    const shapes = ($("riskChart").layout.shapes || []).length;
    if (shapes) Plotly.relayout($("riskChart"), { [`shapes[${shapes - 1}].x0`]: t, [`shapes[${shapes - 1}].x1`]: t });
  }
}

function seek(t, scroll = false) {
  t = Math.max(0, t);
  const video = $("player");
  if (R.playable) {
    const go = () => {
      R.pending = null;
      video.currentTime = t;
      const p = video.play();
      if (p && p.catch) p.catch(() => {});
    };
    // before metadata arrives a seek is dropped, so wait for it
    if (video.readyState >= 1) go();
    else { R.pending = t; video.addEventListener("loadedmetadata", go, { once: true }); }
  } else {
    R.virtual = t;
    $("virtualClock").textContent = `${fmtT(t)}`;
  }
  R.lastRisk = t;
  onTime(t);
  if (scroll) $("videoTabs").scrollIntoView({ behavior: "smooth", block: "start" });
}

function jumpTo(videoId, t) {
  const i = DATA.videos.findIndex((v) => v.id === videoId);
  if (i < 0) return;
  if (i !== R.idx) selectVideo(i);
  seek(t, true);
}

function drawExamples() {
  const box = $("examples");
  const ex = DATA.examples || [];
  if (!ex.length) { box.append(h("p", { class: "empty" }, "Example frames appear here after rendering.")); return; }
  ex.forEach((e) => {
    const thumb = h("div", { class: "thumb" });
    const placeholder = () => thumb.replaceChildren(h("span", { class: "ph" }, `${e.video}, ${fmtT(e.start)} to ${fmtT(e.end)}`));
    if (e.image) {
      const img = h("img", { src: e.image, alt: `${pretty(e.label)} in ${e.video} at ${fmtT(e.start)}`, loading: "lazy" });
      img.addEventListener("error", placeholder);
      thumb.append(img);
    } else placeholder();
    box.append(h("article", { class: "example" }, thumb, h("div", { class: "band", style: `background:${color(e.label)}` }),
      h("div", { class: "body" }, h("h4", {}, cap(pretty(e.label))), h("p", {}, e.caption || ""),
        h("button", { class: "linkbtn", type: "button", onclick: () => jumpTo(e.video, e.start) }, `Play ${e.video} at ${fmtT(e.start)}`))));
  });
}

function drawFailures() {
  const box = $("failures");
  const f = DATA.failures || [];
  if (!f.length) { box.append(h("p", { class: "empty" }, "Failure analysis is added once the samples are labelled.")); return; }
  f.forEach((e) => {
    box.append(h("article", { class: "failure", style: `border-left-color:${color(e.label)}` },
      h("span", { class: "kind" }, `${e.kind}, ${e.video}`), h("h4", {}, cap(pretty(e.label))), h("p", {}, e.why),
      h("button", { class: "linkbtn", type: "button", onclick: () => jumpTo(e.video, e.start) }, `Play at ${fmtT(e.start)}`)));
  });
}

/* ------------------------------------------------------------------ data (EDA) */
function drawEDA() {
  const eda = DATA.eda || {};
  const vids = DATA.videos;
  const gtAll = vids.flatMap((v) => v.ground_truth || []);
  const facts = Object.keys(eda.summary || {}).length ? eda.summary : {
    Videos: vids.length,
    "Total footage": fmtDur(vids.reduce((a, v) => a + v.duration, 0)),
    Resolution: [...new Set(vids.map((v) => v.width && `${v.width} x ${v.height}`).filter(Boolean))].join(", ") || "unknown",
    "Labelled events": gtAll.length || "not labelled yet",
  };
  $("edaFacts").replaceChildren(...Object.entries(facts).flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, String(v))]));

  const series = vids.filter((v) => (v.stats || {}).objects_over_time);
  if (series.length) {
    chart("edaTraffic", () => ({
      traces: series.flatMap((v, i) => {
        const pts = v.stats.objects_over_time;
        const shade = ["#F2C230", "#7CE2FE", "#B1A9FF", "#FFA057", "#46A758"][i % 5];
        return [
          { x: pts.map((p) => p[0]), y: pts.map((p) => p[1]), name: `${v.id} vehicles`, mode: "lines", line: { color: shade, width: 2 }, hovertemplate: "%{x}s: %{y} vehicles<extra>" + v.id + "</extra>" },
          { x: pts.map((p) => p[0]), y: pts.map((p) => p[2]), name: `${v.id} people`, mode: "lines", line: { color: shade, width: 1.5, dash: "dot" }, hovertemplate: "%{x}s: %{y} people<extra>" + v.id + "</extra>" },
        ];
      }),
      layout: { height: 330, margin: { l: 10, r: 10, t: 56, b: 40 }, xaxis: { title: { text: "time in video (s)" } }, yaxis: { title: { text: "objects in frame" } } },
    }));
  } else $("edaTraffic").replaceChildren(h("p", { class: "empty" }, "Per-video traffic counts come from scripts/render.py."));

  const src = gtAll.length ? gtAll : vids.flatMap((v) => v.events.map((e) => [e.start, e.end, e.label]));
  const which = gtAll.length ? "our labels" : "our detections (no labels yet)";
  const classes = DATA.classes.filter((c) => src.some((g) => g[2] === c));
  chart("edaCounts", () => ({
    traces: [{ type: "bar", orientation: "h", y: classes.map(pretty), x: classes.map((c) => src.filter((g) => g[2] === c).length),
      marker: { color: classes.map(color) }, hovertemplate: "%{y}: %{x} events<extra></extra>" }],
    layout: { height: 60 + 24 * classes.length, xaxis: { title: { text: `events (${which})` }, dtick: 1 }, yaxis: { autorange: "reversed" } },
  }));
  chart("edaDurations", () => ({
    traces: classes.map((c) => ({ type: "box", name: pretty(c), x: src.filter((g) => g[2] === c).map((g) => +(g[1] - g[0]).toFixed(2)),
      marker: { color: color(c) }, line: { color: color(c) }, boxpoints: "all", jitter: 0.3, pointpos: 0, orientation: "h", hovertemplate: "%{x} s<extra>" + pretty(c) + "</extra>" })),
    layout: { height: 60 + 24 * classes.length, showlegend: false, xaxis: { title: { text: "event length (s, log scale)" }, type: "log", tickvals: [1, 2, 5, 10, 20, 50, 100, 200, 500], ticktext: ["1", "2", "5", "10", "20", "50", "100", "200", "500"] }, yaxis: { autorange: "reversed" } },
  }));

  const figs = eda.figures || [];
  $("edaFigures").replaceChildren(...figs.map((f) => {
    const fig = h("figure", {});
    const img = h("img", { src: f.src, alt: f.caption || "", loading: "lazy" });
    img.addEventListener("error", () => img.replaceWith(h("div", { class: "missing" }, `Figure not generated yet: ${f.src}`)));
    fig.append(img, h("figcaption", {}, f.caption || ""));
    return fig;
  }));
}

/* ------------------------------------------------------------------ approach: rules table */
function drawRulesTable() {
  const body = $("rulesTable").querySelector("tbody");
  (DATA ? DATA.classes : Object.keys(CLASS_META)).forEach((c) => {
    const [col, what, when] = CLASS_META[c] || ["#888", "", ""];
    body.append(h("tr", {}, h("td", {}, h("span", { class: "dot", style: `background:${col}` }), cap(pretty(c))), h("td", {}, what), h("td", {}, when)));
  });
}

/* ------------------------------------------------------------------ evaluation */
function drawEvaluation() {
  const sm = DATA.summary || {};
  const b = sm.part_b || {};
  const items = [
    ["Score A, events", num(sm.score_a), "macro F1"],
    ["Score B, risk", num(sm.score_b), b.ap != null ? `AP ${num(b.ap)}` : ""],
    ["Model score", num(sm.model_score), "0.7 A + 0.3 B"],
    ["Run time", sm.runtime_x_realtime != null ? `${sm.runtime_x_realtime}x` : null, "of video length, T4"],
  ];
  $("scoreline").replaceChildren(...items.map(([k, v, small]) => h("div", {}, h("dt", {}, k), h("dd", {}, v ?? "n/a", small ? h("small", {}, small) : null))));

  const pc = sm.per_class || {};
  const cls = DATA.classes.filter((c) => pc[c]);
  if (cls.length) {
    chart("f1Chart", () => {
      const shades = [["f1_03", "tIoU 0.3", cssVar("--ink-2")], ["f1_05", "tIoU 0.5", "#C9A227"], ["f1_07", "tIoU 0.7", "#F2C230"]];
      return {
        traces: shades.map(([k, name, col]) => ({ type: "bar", orientation: "h", name, y: cls.map(pretty), x: cls.map((c) => pc[c][k]), marker: { color: col },
          customdata: cls.map((c) => [pc[c].gt, pc[c].pred]), hovertemplate: `%{y}, ${name}: F1 %{x:.2f}<br>%{customdata[0]} labelled, %{customdata[1]} predicted<extra></extra>` })),
        layout: { barmode: "group", height: 80 + 30 * cls.length, bargap: 0.25, xaxis: { range: [0, 1], title: { text: "F1" } }, yaxis: { autorange: "reversed" }, legend: { orientation: "h", y: 1.06, yanchor: "bottom", x: 0 }, margin: { l: 10, r: 10, t: 30, b: 40 } },
      };
    });
  } else $("f1Chart").replaceChildren(h("p", { class: "empty" }, "Per-class F1 appears when build_data.py is run with --gt."));

  const cm = DATA.confusion;
  if (cm && cm.labels) {
    chart("confusionChart", () => {
      const labels = cm.labels.map(pretty);
      return {
        traces: [{ type: "heatmap", z: cm.matrix, x: labels, y: labels, colorscale: [[0, cssVar("--surface")], [0.001, "#5A4C1C"], [1, "#F2C230"]], showscale: false,
          xgap: 2, ygap: 2, text: cm.matrix.map((r) => r.map((v) => (v ? String(v) : ""))), texttemplate: "%{text}", textfont: { color: cssVar("--ink") }, hovertemplate: "predicted %{y}, labelled %{x}: %{z}<extra></extra>" }],
        layout: { height: 120 + 26 * labels.length, xaxis: { title: { text: "our label" }, tickangle: -40, side: "bottom" }, yaxis: { title: { text: "predicted" }, autorange: "reversed" }, margin: { l: 10, r: 10, t: 10, b: 10 } },
      };
    });
    $("confusionNote").textContent = cm.note || "";
  } else $("confusionChart").replaceChildren(h("p", { class: "empty" }, "The confusion matrix needs labelled samples."));

  const rows = DATA.ablations || [];
  const tb = $("ablationTable").querySelector("tbody");
  if (!rows.length) tb.append(h("tr", {}, h("td", { colspan: 7, class: "empty" }, "Ablation runs are added from ablations.json.")));
  const base = rows[0] || {};
  const best = Math.max(...rows.map((r) => r.score_a ?? -1));
  rows.forEach((r, i) => {
    const d = i > 0 && r.score_a != null && base.score_a != null ? (r.score_a - base.score_a) : null;
    tb.append(h("tr", { class: r.score_a === best ? "best" : null },
      h("td", {}, r.variant, r.note ? h("div", { class: "note", style: "margin:2px 0 0" }, r.note) : null),
      h("td", {}, num(r.score_a) ?? "n/a", d !== null ? h("span", { class: "delta" }, `${d >= 0 ? "+" : ""}${d.toFixed(2)}`) : null),
      h("td", {}, num(r.f1_03) ?? "n/a"), h("td", {}, num(r.f1_05) ?? "n/a"), h("td", {}, num(r.f1_07) ?? "n/a"),
      h("td", {}, num(r.score_b) ?? "n/a"), h("td", {}, r.runtime_x != null ? `${r.runtime_x}x` : "n/a")));
  });
}

/* ------------------------------------------------------------------ dashboard */
const DASH = { on: new Set() };
function allEvents() {
  return DATA.videos.flatMap((v) => v.events.map((e) => {
    const base = v.recorded_at ? new Date(v.recorded_at) : null;
    const clock = base && !isNaN(base) ? new Date(base.getTime() + e.start * 1000) : null;
    return { ...e, video: v.id, clock };
  }));
}
function drawDashboard() {
  const evs = allEvents();
  const classes = DATA.classes.filter((c) => evs.some((e) => e.label === c));
  classes.forEach((c) => DASH.on.add(c));
  const box = $("classFilters");
  const chips = classes.map((c) => h("button", { class: "chip", type: "button", "aria-pressed": "true", "data-c": c, onclick: (ev) => {
    const b = ev.currentTarget;
    DASH.on.has(c) ? DASH.on.delete(c) : DASH.on.add(c);
    b.setAttribute("aria-pressed", DASH.on.has(c) ? "true" : "false");
    refresh();
  } }, h("i", { style: `background:${color(c)}` }), pretty(c)));
  const all = h("button", { class: "chip", type: "button", onclick: () => {
    const every = DASH.on.size === classes.length;
    DASH.on.clear();
    if (!every) classes.forEach((c) => DASH.on.add(c));
    chips.forEach((b) => b.setAttribute("aria-pressed", DASH.on.has(b.dataset.c) ? "true" : "false"));
    refresh();
  } }, "All or none");
  box.replaceChildren(all, ...chips);

  const clocked = evs.some((e) => e.clock);
  const sel = () => evs.filter((e) => DASH.on.has(e.label));
  const vids = DATA.videos.map((v) => v.id);
  const shades = ["#F2C230", "#7CE2FE", "#B1A9FF", "#FFA057", "#46A758", "#E5484D"];

  chart("dashClass", () => {
    const cs = classes.filter((c) => DASH.on.has(c));
    return {
      traces: vids.map((vid, i) => ({ type: "bar", orientation: "h", name: vid, y: cs.map(pretty), x: cs.map((c) => sel().filter((e) => e.label === c && e.video === vid).length),
        marker: { color: shades[i % shades.length] }, hovertemplate: `%{y}: %{x} in ${vid}<extra></extra>` })),
      layout: { barmode: "stack", height: 90 + 26 * Math.max(1, cs.length), xaxis: { title: { text: "events" }, dtick: 1 }, yaxis: { autorange: "reversed" }, legend: { orientation: "h", y: 1.02, yanchor: "bottom", x: 0, traceorder: "normal" }, margin: { l: 10, r: 10, t: 30, b: 40 } },
    };
  });
  chart("dashHour", () => {
    const cs = classes.filter((c) => DASH.on.has(c));
    const bins = clocked ? [...Array(24).keys()] : [...Array(Math.ceil(Math.max(...DATA.videos.map((v) => v.duration)) / 60)).keys()];
    const binOf = (e) => (clocked && e.clock ? e.clock.getHours() : Math.floor(e.start / 60));
    const z = cs.map((c) => bins.map((b) => sel().filter((e) => e.label === c && binOf(e) === b).length));
    return {
      traces: [{ type: "heatmap", z, x: bins.map((b) => (clocked ? `${String(b).padStart(2, "0")}:00` : `${b} min`)), y: cs.map(pretty),
        colorscale: [[0, cssVar("--surface")], [0.001, "#5A4C1C"], [1, "#F2C230"]], showscale: false, xgap: 1, ygap: 1,
        hovertemplate: "%{y}, %{x}: %{z}<extra></extra>" }],
      layout: { height: 90 + 26 * Math.max(1, cs.length), xaxis: { title: { text: clocked ? "hour of day (recording clock)" : "minute of footage" }, tickangle: 0, nticks: 8 }, yaxis: { autorange: "reversed" } },
    };
  });

  function drawLog() {
    const tb = $("eventLog").querySelector("tbody");
    const rows = sel().sort((a, b) => (a.clock && b.clock ? a.clock - b.clock : a.video.localeCompare(b.video) || a.start - b.start));
    tb.replaceChildren(...rows.map((e) => h("tr", {},
      h("td", {}, e.video),
      h("td", {}, e.clock ? e.clock.toTimeString().slice(0, 8) : `+${fmtT(e.start)}`),
      h("td", {}, h("span", { class: "dot", style: `background:${color(e.label)}` }), cap(pretty(e.label))),
      h("td", {}, fmtDur(e.end - e.start)),
      h("td", {}, h("button", { class: "linkbtn", type: "button", onclick: () => jumpTo(e.video, e.start) }, "Play")))));
    if (!rows.length) tb.append(h("tr", {}, h("td", { colspan: 5, class: "empty" }, "No events for the selected classes.")));
  }
  function refresh() { CHARTS.forEach((d) => d()); drawLog(); }
  drawLog();
}

/* ------------------------------------------------------------------ static-ish sections */
function drawDemo() {
  const box = $("demoFrame");
  const id = CFG.SPACE_ID || "";
  if (!id || id.includes("TODO")) {
    box.append(h("div", { class: "demo-placeholder" },
      h("h3", {}, "The demo Space is not linked yet"),
      h("ol", {},
        h("li", {}, "Deploy demo/ to a Hugging Face Space (steps in demo/README.md)."),
        h("li", {}, "Set SPACE_ID in site/assets/config.js to \"user/space-name\"."),
        h("li", {}, "Reload: the demo appears here, embedded."))));
    return;
  }
  const sub = id.toLowerCase().replace(/[/_.]/g, "-");
  box.append(
    h("iframe", { src: `https://${sub}.hf.space`, title: "Iceberg live demo on Hugging Face", loading: "lazy", allow: "clipboard-write", referrerpolicy: "no-referrer" }),
    h("p", { class: "demo-open" }, "If the frame stays blank, ", h("a", { href: `https://huggingface.co/spaces/${id}`, target: "_blank", rel: "noopener" }, "open the demo on Hugging Face"), "."));
}

function drawReportPoints() {
  const sm = DATA.summary || {};
  const pts = [
    [`${sm.events ?? 0} events`, ` detected in ${fmtDur(sm.footage_sec || 0)} of sample footage.`],
    sm.score_a != null ? [`Score A ${num(sm.score_a)}`, " on our labels, averaged over tIoU 0.3, 0.5 and 0.7."] : null,
    sm.score_b != null ? [`Score B ${num(sm.score_b)}`, " for accident anticipation."] : null,
    sm.runtime_x_realtime != null ? [`${sm.runtime_x_realtime}x video length`, " to process on a T4, inside the 3x budget."] : null,
  ].filter(Boolean);
  $("reportPoints").replaceChildren(...pts.map(([b, t]) => h("li", {}, h("strong", {}, b), t)));
}

function drawTeam() {
  $("teamGrid").replaceChildren(...(CFG.TEAM || []).map((m) => {
    const initials = m.todo ? "?" : m.name.split(/\s+/).map((w) => w[0]).join("").slice(0, 2);
    return h("article", { class: `member${m.todo ? " todo" : ""}` },
      m.todo ? h("span", { class: "todo-flag" }, "Placeholder") : null,
      h("div", { class: "avatar", "aria-hidden": "true" }, initials),
      h("h3", {}, m.name), h("p", { class: "role" }, m.role), h("p", {}, m.focus),
      m.past && m.past.length ? h("div", {}, h("p", { style: "margin-bottom:4px;color:var(--ink)" }, "Past projects"), h("ul", {}, m.past.map((p) => h("li", {}, p)))) : null,
      h("div", { class: "social" },
        m.github ? h("a", { href: m.github, target: "_blank", rel: "noopener" }, "GitHub") : null,
        m.linkedin ? h("a", { href: m.linkedin, target: "_blank", rel: "noopener" }, "LinkedIn") : null));
  }));
}

function drawLinks() {
  const space = CFG.SPACE_ID && !CFG.SPACE_ID.includes("TODO") ? `https://huggingface.co/spaces/${CFG.SPACE_ID}` : null;
  const links = [
    ["Source code", CFG.REPO_URL, "GitHub repository: pipeline, rules, demo and this site"],
    ["Live demo", space || "#demo", space ? "Hugging Face Space, CPU" : "Hugging Face Space (link after deploy)"],
    ["Model weights", CFG.WEIGHTS_URL, "Detector and fire and smoke checkpoints"],
    ["predictions_samples.json", DATA && !DATA.is_sample ? CFG.PREDICTIONS_URL : "#links", "Our predictions on the sample videos, harness format"],
    ["Report", CFG.REPORT_URL, "One printable page"],
    ["results.json", DATA && !DATA.is_sample ? "data/results.json" : "data/results.sample.json", "Everything this page draws"],
  ];
  $("linkList").replaceChildren(...links.map(([t, href, d]) => h("li", {},
    h("a", { href, class: String(href).includes("TODO") || href === "#links" ? "pending" : null, target: href.startsWith("http") ? "_blank" : null, rel: "noopener" }, h("strong", {}, t), h("span", {}, d)))));
}

/* ------------------------------------------------------------------ chrome */
function initTheme() {
  $("themeBtn").addEventListener("click", () => {
    const dark = cssVar("color-scheme") !== "light";
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("iceberg-theme", next); } catch (e) {}
    CHARTS.forEach((d) => d());
  });
  const mq = window.matchMedia("(prefers-color-scheme: light)");
  if (mq.addEventListener) mq.addEventListener("change", () => CHARTS.forEach((d) => d()));
}
function initNavSpy() {
  const links = [...document.querySelectorAll(".nav-links a")];
  const byId = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
  const io = new IntersectionObserver((entries) => {
    entries.forEach((en) => {
      if (!en.isIntersecting) return;
      links.forEach((a) => a.removeAttribute("aria-current"));
      const a = byId.get(en.target.id);
      if (a) { a.setAttribute("aria-current", "true"); a.scrollIntoView({ block: "nearest", inline: "nearest" }); }
    });
  }, { rootMargin: "-45% 0px -50% 0px" });
  document.querySelectorAll("main section[id]").forEach((sec) => io.observe(sec));
}
/** Re-render on width changes only (re-rendering changes height, which must not loop). */
function onWidthChange(el, fn) {
  let w = el.clientWidth;
  const run = debounce(() => { if (el.clientWidth !== w) { w = el.clientWidth; fn(); } }, 120);
  new ResizeObserver(run).observe(el);
}
function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

async function loadData() {
  for (const url of CFG.DATA_URLS || ["data/results.json", "data/results.sample.json"]) {
    try {
      const r = await fetch(url, { cache: "no-cache" });
      if (r.ok) return await r.json();
    } catch (e) { /* try the next source */ }
  }
  return null;
}

function drawHero() {
  const v = DATA.videos[0];
  if (!v) return;
  const draw = () => renderLanes($("heroLanes"), v, {});
  draw();
  onWidthChange($("heroLanes"), draw);
  $("heroCaption").textContent = `Our detections on ${v.id} (${fmtDur(v.duration)}): one lane per event class.`;
}

async function main() {
  initTheme();
  initNavSpy();
  drawDemo();
  drawTeam();
  DATA = await loadData();
  if (!DATA) {
    $("heroCaption").textContent = "Could not load data/results.json. Serve the site over HTTP (for example: python -m http.server -d site).";
    drawRulesTable();
    drawLinks();
    return;
  }
  $("sampleBanner").hidden = !DATA.is_sample;
  if (DATA.generated_at) $("generatedAt").textContent = `Data generated ${DATA.generated_at.slice(0, 10)}.`;
  const safe = (fn) => { try { fn(); } catch (e) { console.error(fn.name, e); } };
  [drawHero, drawRulesTable, drawLinks, drawReportPoints, initResults, drawExamples, drawFailures, drawEDA, drawEvaluation, drawDashboard].forEach(safe);
  window.addEventListener("resize", debounce(() => CHARTS.forEach((d) => d()), 200));
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", main);
else main();
