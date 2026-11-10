// v2 single-track labeling (Phase B + multi-file sequencing). ESM module.
// Multiple files are concatenated server-side into one combined waveform whose timeline is
// already the sequenced one, so labeling/transcription reuse the single-file flow.
// Calls window.renderEditor (app.js) to show the transcript.
import WaveSurfer from "./vendor/wavesurfer.esm.js";
import RegionsPlugin from "./vendor/regions.esm.js";

const $ = (s) => document.querySelector(s);
const el = (tag, props = {}, ...kids) => {
  const n = Object.assign(document.createElement(tag), props);
  for (const k of kids) n.append(k);
  return n;
};
// The busy line above the waveform: busy("Detecting speech") slides; busy("Loading", 40) shows 40%;
// busy(null) hides it.
function busy(msg, pct) {
  const b = $("#label-busy");
  if (msg == null) { b.style.display = "none"; return; }
  b.style.display = "";
  b.querySelector("span").textContent = pct == null ? `${msg}…` : `${msg} · ${Math.round(pct)}%`;
  const fill = b.querySelector(".bar > div");
  fill.classList.toggle("indeterminate", pct == null);
  fill.style.width = pct == null ? "" : `${pct}%`;
}

// Swap the Label play/pause button glyph (uses app.js's shared SVG icon set).
function setPlayIcon(name) {
  const b = $("#label-play");
  if (b && window.uiIcon) b.replaceChildren(window.uiIcon(name));
}

const PALETTE = ["#5b9dff", "#57b894", "#d9a441", "#e0625e", "#b07cff", "#46c6c6", "#e08a3c", "#9ac85b"];
const UNASSIGNED = "rgba(150,160,175,0.25)";
const BOUNDARY = "rgba(230,233,239,0.9)";
const alpha = (hex) => hex + "66";

const files = [];            // [{file_id, track_index, filename, duration, tracks}]
const speakers = [];         // [{name, color}]
const assign = new Map();    // region.id -> speaker name
const boundaryIds = new Set();
let ws = null, regions = null, anchorId = null, splitView = false;
let decoded = null;          // { data: Float32Array, sr } — for drawing split-lane mini-waveforms
const selection = new Set();   // region ids currently selected (bulk-assignable)
let activeFileId = null, activeTrack = 0, activeDuration = 0, boundaries = [0];
let buildToken = 0;
let lastSpans = null;        // the last speech detection, so a Region pad change can re-pad it
let stackView = false;       // show the file's other tracks under the waveform
const stackPeaks = new Map(); // "file_id:track" -> peaks

$("#label-choose").onclick = addFiles;
$("#label-choose-more").onclick = addFiles;
// Detection runs by itself when a recording loads; the button is the fallback to start over.
$("#label-vad").onclick = () => {
  if (assign.size && !confirm("Detect speech again? This replaces the current regions and their speaker labels.")) return;
  runVad();
};
// Region pad applies straight away while nothing is labeled yet (no need to detect again).
$("#label-pad").onchange = () => { if (lastSpans && !assign.size) applySpans(lastSpans); };
$("#label-stack-btn").onclick = () => {
  stackView = !stackView;
  $("#label-stack-btn").setAttribute("aria-pressed", stackView);   // the name stays; it lights up when on
  renderStack();
};
$("#label-play").onclick = () => ws && ws.playPause();
$("#label-run").onclick = runTranscribe;
$("#label-view").onclick = () => {
  splitView = !splitView;
  $("#label-view").setAttribute("aria-pressed", splitView);
  renderLanes();
};
$("#label-dup").onclick = duplicateSelection;
// zoom is relative to "fit the whole file in view": slider=1 → entire waveform visible
// start-to-finish; higher → that many times zoomed in.
function fitPxPerSec() {
  const w = $("#waveform").clientWidth || 600;
  return w / (activeDuration || 1);
}
function applyZoom() {
  if (ws) try { ws.zoom(fitPxPerSec() * (+$("#label-zoom").value || 1)); } catch (_) {}
  renderLanes();   // lanes scale with the same zoom factor
  renderStack();
}
$("#label-zoom").oninput = applyZoom;
$("#label-vzoom").oninput = () => {
  if (ws) try { ws.setOptions({ barHeight: +$("#label-vzoom").value }); } catch (_) {}
  renderLanes();   // mini-waveforms use the same amplitude gain
  renderStack();
};

// Duplicate the selected region(s) as new unassigned regions on the same span — for talk-over,
// assign the copy to the other speaker so two speakers overlap on that range (PRD §11.4 / OQ-7).
function duplicateSelection() {
  if (!regions) return;
  const dups = [];
  for (const r of regions.getRegions()) {
    if (!selection.has(r.id) || boundaryIds.has(r.id)) continue;
    dups.push(regions.addRegion({ start: r.start, end: r.end, color: UNASSIGNED, drag: true, resize: true }));
  }
  if (!dups.length) { $("#label-status").textContent = "Select a region first, then Duplicate."; return; }
  selection.clear();
  dups.forEach((d) => selection.add(d.id));
  anchorId = dups[dups.length - 1].id;
  highlight();   // re-renders lanes too
  $("#label-status").textContent = "duplicated — now pick the other speaker";
}

document.addEventListener("keydown", (e) => {
  if (!$("#tab-label").classList.contains("active")) return;
  if (/^[1-9]$/.test(e.key) && selection.size) {
    const sp = speakers[+e.key - 1];
    if (sp) assignSelectedTo(sp);
  }
});

// --- file list -------------------------------------------------------------------------

async function addFiles() {
  $("#label-err").textContent = "";
  const res = await fetch("/api/browse", { method: "POST" });
  if (!res.ok) { $("#label-err").textContent = "Picker failed: " + (await res.text()); return; }
  for (const it of await res.json()) {
    files.push({ file_id: it.file_id, track_index: 0, filename: it.source.filename,
                 duration: it.source.duration, tracks: it.source.tracks });
  }
  renderFiles();
  rebuild();
}

function renderFiles() {
  const has = files.length > 0;                      // empty → entry hero; loaded → files panel + waveform
  $("#label-entry").style.display = has ? "none" : "";
  $("#label-input-panel").style.display = has ? "" : "none";
  const root = $("#label-files");
  root.innerHTML = "";
  files.forEach((f, i) => {
    const swap = (a, b) => { [files[a], files[b]] = [files[b], files[a]]; renderFiles(); rebuild(); };
    const up = el("button", { className: "ghost icon-only", title: "Move earlier" }, window.uiIcon("up"));
    up.onclick = () => { if (i > 0) swap(i, i - 1); };
    const down = el("button", { className: "ghost icon-only", title: "Move later" }, window.uiIcon("down"));
    down.onclick = () => { if (i < files.length - 1) swap(i, i + 1); };
    const rm = el("button", { className: "ghost icon-only", title: "Remove" }, window.uiIcon("close"));
    rm.onclick = () => { release(f.file_id); files.splice(i, 1); renderFiles(); rebuild(); };

    const row = el("div", { className: "track-row" },
      el("span", { className: "ti", textContent: `${i + 1}.` }),
      el("span", { textContent: f.filename, style: "flex:1" }),
      el("span", { className: "dur muted", textContent: `${f.duration.toFixed(1)}s` }));
    if (f.tracks.length > 1) {
      const sel = el("select");
      f.tracks.forEach((t) => sel.append(el("option", { value: t.track_index, textContent: `track ${t.track_index + 1}` })));
      sel.value = f.track_index;
      sel.onchange = () => { f.track_index = +sel.value; rebuild(); };
      row.append(sel);
    }
    row.append(up, down, rm);
    root.append(row);
  });
  $("#label-work").style.display = files.length ? "" : "none";
  renderStack();
}

// --- build / load the (combined) waveform ----------------------------------------------

// Disk cleanup follows the file list: a file's full-track audio is deleted when it's removed,
// and a combined recording when a different one (or a single file) replaces it on screen.
// ponytail: a combine superseded mid-flight is left for the restart wipe; rare and harmless.
const release = (id) => fetch(`/api/release/${id}`, { method: "POST" }).catch(() => {});
let shownCombined = null;   // file_id of the combined recording on screen, if any

function showCombined(id) {
  if (shownCombined && shownCombined !== id) release(shownCombined);
  shownCombined = id;
}

async function rebuild() {
  const token = ++buildToken;
  if (!files.length) { showCombined(null); busy(null); if (ws) { ws.destroy(); ws = null; } return; }

  let fid, track, dur, bounds;
  if (files.length === 1) {
    ({ file_id: fid, track_index: track, duration: dur } = files[0]);
    bounds = [0];
  } else {
    $("#label-status").textContent = "";
    busy("Combining files");
    const res = await fetch("/api/combine", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ files: files.map((f) => ({ file_id: f.file_id, track_index: f.track_index })) }),
    });
    if (!res.ok) { busy(null); $("#label-err").textContent = "Combine failed: " + (await res.text()); return; }
    const r = await res.json();
    if (token !== buildToken) return;  // a newer change superseded this build
    fid = r.file_id; track = 0; dur = r.duration; bounds = r.boundaries;
    $("#label-status").textContent = "";
  }
  if (token !== buildToken) return;
  if (ws) { ws.destroy(); ws = null; }   // stop streaming the old audio before it's released
  showCombined(files.length > 1 ? fid : null);
  activeFileId = fid; activeTrack = track; activeDuration = dur; boundaries = bounds;
  initWave(fid, track);
}

function initWave(fid, track) {
  if (ws) { ws.destroy(); ws = null; }
  lastSpans = null;
  assign.clear();
  boundaryIds.clear();
  selection.clear();
  anchorId = null;
  decoded = null;
  setPlayIcon("play");                 // fresh waveform starts paused
  ws = WaveSurfer.create({
    container: "#waveform", height: 180,
    waveColor: "#566", progressColor: "#5b9dff", cursorColor: "#e6e9ef",
    barHeight: +$("#label-vzoom").value || 1,
    url: `/api/track-audio/${fid}/${track}`,
  });
  regions = ws.registerPlugin(RegionsPlugin.create());
  ws.on("decode", () => { regions.enableDragSelection({ color: UNASSIGNED }); captureDecoded(); });
  // The server extracts the whole track before the first byte arrives (slow on big files), then
  // the browser downloads and decodes it.
  busy("Reading the track's audio");
  ws.on("loading", (pct) => busy("Loading the waveform", pct));
  ws.on("decode", () => busy("Drawing the waveform"));
  ws.on("error", (e) => { busy(null); $("#label-err").textContent = "Couldn't load the audio: " + (e && e.message || e); });
  ws.on("ready", () => { captureDecoded(); applyZoom(); runVad(); });   // fit whole file, then auto-detect speech
  ws.on("scroll", (_s, _e, left) => { $("#label-stack").scrollLeft = left; drawStack(); });   // stacked tracks follow
  // Spacebar transport + button glyph both reflect the real play state.
  ws.on("play", () => {
    setPlayIcon("pause");
    window.wmTransport && window.wmTransport.set({ toggle: () => ws && ws.playPause() });
  });
  ws.on("pause", () => setPlayIcon("play"));
  ws.on("finish", () => setPlayIcon("play"));
  regions.on("region-created", (r) => { if (!assign.has(r.id) && !boundaryIds.has(r.id)) { r.setOptions({ color: UNASSIGNED }); renderLanes(); } });
  regions.on("region-updated", () => renderLanes());
  regions.on("region-clicked", (r, e) => {
    if (boundaryIds.has(r.id)) return;   // ignore file-join markers
    e.stopPropagation();
    if (e.shiftKey && anchorId) {
      selectRange(anchorId, r.id);       // shift-click: select the whole run
    } else if (e.ctrlKey || e.metaKey) {
      selection.has(r.id) ? selection.delete(r.id) : selection.add(r.id);  // toggle one
      anchorId = r.id;
    } else {
      selection.clear(); selection.add(r.id); anchorId = r.id; r.play();   // single
    }
    highlight();
  });
}

function speechRegions() {
  return regions.getRegions().filter((r) => !boundaryIds.has(r.id)).sort((a, b) => a.start - b.start);
}

function selectRange(aId, bId) {
  const list = speechRegions();
  const ia = list.findIndex((r) => r.id === aId);
  const ib = list.findIndex((r) => r.id === bId);
  if (ia < 0 || ib < 0) return;
  for (let i = Math.min(ia, ib); i <= Math.max(ia, ib); i++) selection.add(list[i].id);
}

function drawBoundaries() {
  boundaries.slice(1).forEach((b, i) => {
    const m = regions.addRegion({ start: b, end: b, color: BOUNDARY, drag: false, resize: false,
                                  content: `file ${i + 2}` });
    boundaryIds.add(m.id);
  });
}

function highlight() {
  for (const x of regions.getRegions()) {
    if (x.element) x.element.style.outline = selection.has(x.id) ? "2px solid #e6e9ef" : "none";
  }
  renderLanes();
}

function captureDecoded() {
  try {
    const b = ws && ws.getDecodedData();
    if (b) { decoded = { data: b.getChannelData(0), sr: b.sampleRate }; renderLanes(); }
  } catch (_) { /* not decoded yet */ }
}

// Draw one region's slice of the decoded audio into its lane canvas (max-abs envelope, same
// amplitude gain as the vertical-zoom slider). Falls back to a flat tint until audio is decoded.
function drawRegionWave({ canvas, start, end, un, color }) {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  canvas.width = w; canvas.height = h;
  if (!decoded) return;
  const ctx = canvas.getContext("2d");
  const { data, sr } = decoded;
  const i0 = Math.max(0, Math.floor(start * sr)), i1 = Math.min(data.length, Math.ceil(end * sr));
  if (i1 <= i0) return;
  const gain = +($("#label-vzoom").value) || 1;
  ctx.fillStyle = un ? "rgba(230,233,239,0.6)" : color;
  const per = (i1 - i0) / w;
  for (let x = 0; x < w; x++) {
    let mx = 0;
    const s = i0 + Math.floor(x * per), e = Math.max(s + 1, i0 + Math.floor((x + 1) * per));
    for (let j = s; j < e; j++) { const a = Math.abs(data[j]); if (a > mx) mx = a; }
    const ph = Math.min(h, Math.max(1, mx * h * gain));
    ctx.fillRect(x, (h - ph) / 2, 1, ph);
  }
}

// Split-lane view: one row per speaker (+ unassigned), regions as mini-waveforms on a shared timescale.
function renderLanes() {
  const root = $("#label-lanes");
  if (!root) return;
  if (!splitView || !regions) { root.style.display = "none"; return; }
  root.style.display = "";
  root.style.overflowX = "auto";
  root.innerHTML = "";
  const dur = activeDuration || 1;
  // Horizontal zoom: lane strip width = (fit width) * zoom factor; factor 1 fits the whole file,
  // higher scrolls — kept proportional to the main waveform zoom (shared #label-zoom slider).
  const factor = +($("#label-zoom").value) || 1;
  const labelW = 110, gap = 8;
  const trackW = Math.max(50, root.clientWidth - labelW - gap - 4) * factor;
  const inner = el("div", { style: `width:${labelW + gap + trackW}px` });
  inner.append(buildRuler(dur, labelW, trackW));
  const byName = new Map();
  const unassigned = [];
  for (const r of regions.getRegions()) {
    if (boundaryIds.has(r.id)) continue;
    const name = assign.get(r.id);
    if (name) { if (!byName.has(name)) byName.set(name, []); byName.get(name).push(r); }
    else unassigned.push(r);
  }
  const rows = speakers.map((s) => ({ name: s.name, color: s.color, regs: byName.get(s.name) || [] }));
  rows.push({ name: "(unassigned)", color: "#8a93a3", regs: unassigned, un: true });
  const pending = [];
  for (const row of rows) {
    const track = el("div", { className: "lane-track", style: `flex:none; width:${trackW}px` });
    for (const r of row.regs) {
      const blk = el("canvas", { className: "lane-block", title: `${r.start.toFixed(2)}–${r.end.toFixed(2)}s` });
      blk.style.left = (r.start / dur * 100) + "%";
      blk.style.width = Math.max(0.3, (r.end - r.start) / dur * 100) + "%";
      blk.style.background = row.un ? UNASSIGNED : alpha(row.color);
      blk.style.outline = selection.has(r.id) ? "2px solid #e6e9ef" : "none";
      blk.onclick = () => { selection.clear(); selection.add(r.id); anchorId = r.id; highlight(); r.play(); };
      track.append(blk);
      pending.push({ canvas: blk, start: r.start, end: r.end, un: row.un, color: row.un ? "#aab" : row.color });
    }
    const label = el("div", { className: "lane-label" + (row.un ? " un" : ""),
      style: `width:${labelW}px; flex:none; position:sticky; left:0; z-index:1` },
      el("span", { className: "lane-swatch", style: `background:${row.un ? UNASSIGNED : row.color}` }),
      el("span", { className: "lane-name", textContent: row.name, style: `color:${row.un ? "var(--muted)" : row.color}` }),
      el("span", { className: "lane-count", textContent: row.regs.length || "" }));
    inner.append(el("div", { className: "lane-row" }, label, track));
  }
  root.append(inner);
  requestAnimationFrame(() => pending.forEach(drawRegionWave));   // size+draw after layout
}

// "All tracks": the file's other tracks as read-only waveforms under the main one, same width
// (fit width x zoom) and scroll, so you can see who's talking when. One multi-track file only;
// several files are combined into one track. Each row's canvas is only as wide as the view and
// redrawn as it scrolls: from whole-track peaks straight away, then from a sharper window around
// what's on screen once zoomed in past what those can show.
let stackToken = 0, stackW = 1, stackRows = [], stackFetching = false;
async function renderStack() {
  const root = $("#label-stack");
  const f = files.length === 1 && files[0].tracks.length > 1 ? files[0] : null;
  $("#label-stack-btn").style.display = f ? "" : "none";
  const token = ++stackToken;
  root.innerHTML = "";
  stackRows = [];
  if (!stackView || !f || !ws) { root.style.display = "none"; return; }
  root.style.display = "";
  stackW = Math.round(($("#waveform").clientWidth || 600) * (+$("#label-zoom").value || 1));
  const vw = root.clientWidth || 600;
  const inner = el("div", { style: `width:${stackW}px` });
  root.append(inner);
  for (const t of f.tracks) {
    if (t.track_index === activeTrack) continue;           // that one is the main waveform
    const canvas = el("canvas", { width: vw, height: 44,
      style: `display:block; flex:none; position:sticky; left:0; width:${vw}px; height:44px` });
    inner.append(el("div", { className: "stack-row" },
      el("span", { className: "stack-label", textContent: `track ${t.track_index + 1} · ${t.speaker}` }), canvas));
    stackRows.push({ canvas, key: `${f.file_id}:${t.track_index}`,
                     url: `/api/peaks/${f.file_id}?track=${t.track_index}`, detail: null });
  }
  root.scrollLeft = ws.getScroll();
  for (const r of stackRows) {
    if (stackPeaks.has(r.key)) continue;
    try { stackPeaks.set(r.key, (await (await fetch(`${r.url}&buckets=4000`)).json()).peaks); } catch { /* row stays blank */ }
    if (token !== stackToken) return;                        // re-rendered meanwhile
  }
  drawStack();
}

// Redraw the stacked rows for the part of the file on screen.
function drawStack() {
  const root = $("#label-stack");
  if (!stackRows.length || !activeDuration) return;
  const vw = root.clientWidth, dur = activeDuration, gain = +$("#label-vzoom").value || 1;
  const from = root.scrollLeft / stackW * dur, to = (root.scrollLeft + vw) / stackW * dur;
  for (const r of stackRows) {
    const d = r.detail, all = stackPeaks.get(r.key);
    if (d && d.from <= from && d.to >= Math.min(to, dur)) drawPeaks(r.canvas, d.peaks, d.from, d.to, from, to, gain);
    else if (all) {
      drawPeaks(r.canvas, all, 0, dur, from, to, gain);
      if (all.length * (to - from) / dur < vw) fetchDetail(r, from, Math.min(to, dur), vw);   // too coarse here
    }
  }
}

// One sharper window (the view plus one view-width either side) at a time across all rows; when
// it lands, redraw, which fetches the next one if the view has moved or another row still needs it.
async function fetchDetail(r, from, to, vw) {
  if (stackFetching) return;
  stackFetching = true;
  const span = to - from, a = Math.max(0, from - span), b = Math.min(activeDuration, to + span);
  let ok = false;
  try {
    const n = Math.max(1, Math.round(vw * (b - a) / span));
    r.detail = { from: a, to: b, peaks: (await (await fetch(`${r.url}&buckets=${n}&start=${a}&end=${b}`)).json()).peaks };
    ok = true;
  } catch { /* keep the coarse drawing */ }
  stackFetching = false;
  if (ok) drawStack();
}

// Draw peaks covering pFrom..pTo seconds into the canvas for from..to seconds, one column per pixel.
function drawPeaks(canvas, peaks, pFrom, pTo, from, to, gain) {
  const ctx = canvas.getContext("2d"), w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#566";
  const per = peaks.length / ((pTo - pFrom) || 1);         // peaks per second
  const idx = (x) => Math.floor((from + (to - from) * x / w - pFrom) * per);
  for (let x = 0; x < w; x++) {
    const i0 = idx(x), i1 = Math.max(i0 + 1, idx(x + 1));
    if (i0 >= peaks.length || i1 <= 0) continue;           // past the end of the audio
    let p = 0;
    for (let i = Math.max(0, i0); i < Math.min(peaks.length, i1); i++) if (peaks[i] > p) p = peaks[i];
    const ph = Math.min(h, Math.max(1, p * h * 0.9 * gain));
    ctx.fillRect(x, (h - ph) / 2, 1, ph);
  }
}

// DAW-style time ruler aligned to the lane tracks: ticks + mm:ss labels at a readable spacing.
function buildRuler(dur, labelW, trackW) {
  const ruler = el("div", { className: "lane-ruler" });
  const spacer = el("div", { className: "lane-label", style: `width:${labelW}px; flex:none; position:sticky; left:0; z-index:1` });
  const axis = el("div", { className: "lane-axis", style: `flex:none; width:${trackW}px` });
  // aim for a tick roughly every ~90px; snap the interval to a "nice" number of seconds
  const targetTicks = Math.max(2, Math.floor(trackW / 90));
  const raw = dur / targetTicks;
  const nice = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600].find((n) => n >= raw) || raw;
  for (let t = 0; t <= dur + 0.001; t += nice) {
    const tick = el("div", { className: "tick", style: `left:${(t / dur) * 100}%` },
      el("span", { textContent: fmtTick(t) }));
    axis.append(tick);
  }
  ruler.append(spacer, axis);
  return ruler;
}
function fmtTick(s) {
  s = Math.round(s);
  const m = Math.floor(s / 60), sec = s % 60;
  return `${m}:${String(sec).padStart(2, "0")}`;
}

// --- speakers + assignment -------------------------------------------------------------

function renderSpeakers() {
  const root = $("#label-speakers");
  root.innerHTML = "";
  speakers.forEach((sp, i) => {
    const b = el("button", { className: "ghost icon-btn", title: `assign selected region (key ${i + 1})` },
      el("span", { className: "lane-swatch", style: `background:${sp.color}` }),
      el("span", { textContent: `${i + 1} ${sp.name}` }));
    b.onclick = () => assignSelectedTo(sp);
    root.append(b);
  });
  const add = el("button", { className: "ghost", textContent: "+ Add speaker" });
  add.onclick = () => {
    const name = (prompt("Speaker name?") || "").trim();
    if (!name) return;
    speakers.push({ name, color: PALETTE[speakers.length % PALETTE.length] });
    renderSpeakers();
    renderLanes();
  };
  root.append(add);
}
renderSpeakers();

function assignSelectedTo(sp) {
  for (const r of regions.getRegions()) {
    if (!selection.has(r.id) || boundaryIds.has(r.id)) continue;
    assign.set(r.id, sp.name);
    r.setOptions({ color: alpha(sp.color) });
  }
  renderLanes();
}

// --- VAD + transcribe ------------------------------------------------------------------

function setLabelBar(frac, msg) {
  const pct = Math.round((frac || 0) * 100);
  $("#label-bar").style.width = pct + "%";
  $("#label-pct").textContent = pct + "%";
  $("#label-progress-msg").textContent = msg || "";
}

async function runVad() {
  $("#label-status").textContent = "";
  busy("Detecting speech");
  const fid = activeFileId, track = activeTrack;
  let res;
  try { res = await fetch(`/api/vad/${fid}/${track}`, { method: "POST" }); }
  catch (e) { busy(null); $("#label-status").textContent = "couldn't reach the server."; return; }
  if (fid !== activeFileId || track !== activeTrack) return;   // user moved to another track meanwhile
  busy(null);
  if (!res.ok) {
    $("#label-status").textContent = "Speech detection failed: " + (await res.text()) + " · Detect again to retry";
    return;
  }
  const spans = await res.json();
  lastSpans = spans;
  applySpans(spans);
}

// Speech regions from a detection, padded by Region pad, replacing what's on the waveform.
function applySpans(spans) {
  regions.clearRegions();
  assign.clear();
  boundaryIds.clear();
  selection.clear();
  anchorId = null;
  const pad = Math.max(0, +($("#label-pad").value) || 0);
  spans.forEach((s) => regions.addRegion({
    start: Math.max(0, s.start - pad), end: Math.min(activeDuration || s.end, s.end + pad),
    color: UNASSIGNED, drag: true, resize: true }));
  drawBoundaries();
  renderLanes();
  $("#label-status").textContent = `${spans.length} speech regions — assign each to a speaker`;
}

function collectSpeakerRegions() {
  const map = {};
  for (const r of regions.getRegions()) {
    if (boundaryIds.has(r.id)) continue;
    const name = assign.get(r.id);
    if (!name) continue;
    (map[name] ||= []).push([r.start, r.end]);
  }
  return map;
}

async function runTranscribe() {
  const map = collectSpeakerRegions();
  if (!Object.keys(map).length) { $("#label-status").textContent = "Assign at least one region to a speaker first."; return; }
  const btn = $("#label-run");
  btn.disabled = true;
  $("#label-status").textContent = "starting…";
  $("#label-progress").classList.add("show");
  setLabelBar(0, "starting…");
  let res;
  try {
    res = await window.fetchTO("/api/diarize", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ file_id: activeFileId, track_index: activeTrack, speakers: map, options: { detect_conflicts: true },
                             language: $("#label-language").value || null }),
    }, 30000);
  } catch (e) { $("#label-status").textContent = "couldn't reach the server to start."; btn.disabled = false; return; }
  if (!res.ok) { $("#label-status").textContent = "Failed: " + (await res.text()); btn.disabled = false; return; }
  const { job_id } = await res.json();
  poll(job_id, btn);
}

function poll(jobId, btn) {
  let fails = 0;
  // Pause/Cancel mirror the Transcribe tab (app.js pollJob); both act at the next speaker boundary.
  const cancelBtn = $("#label-cancel"), pauseBtn = $("#label-pause");
  let paused = false;                                  // last-known server pause state
  cancelBtn.style.display = pauseBtn.style.display = "";
  cancelBtn.disabled = pauseBtn.disabled = false;
  cancelBtn.textContent = "Cancel"; pauseBtn.textContent = "Pause";
  cancelBtn.onclick = async () => {
    cancelBtn.disabled = true; cancelBtn.textContent = "Canceling…";
    try { await fetch(`/api/cancel/${jobId}`, { method: "POST" }); } catch {}
  };
  pauseBtn.onclick = async () => {
    pauseBtn.disabled = true;
    pauseBtn.textContent = paused ? "Resuming…" : "Pausing…";
    try { await fetch(`/api/${paused ? "resume" : "pause"}/${jobId}`, { method: "POST" }); } catch {}
  };
  const finish = () => {
    btn.disabled = false;
    cancelBtn.style.display = pauseBtn.style.display = "none";
  };
  const tick = async () => {
    let p;
    try {
      const r = await window.fetchTO(`/api/progress/${jobId}`);
      if (!r.ok) throw new Error(await r.text());
      p = await r.json();
      fails = 0;
    } catch (e) {
      // Server briefly unresponsive (first-run model load) — keep waiting, never freeze.
      $("#label-progress-msg").textContent = (++fails > 5)
        ? "still working… first-run model load can take a few minutes (watch the server window)"
        : "working… waiting for the server";
      setTimeout(tick, 1200); return;
    }
    paused = !!p.paused;
    pauseBtn.disabled = false;
    pauseBtn.textContent = paused ? "Resume" : "Pause";
    const msg = (p.paused && !p.gpu_freed) ? "pausing… (finishing current speaker)" : (p.message || p.status);
    const line = msg + (p.elapsed ? " · " + window.fmtElapsed(p.elapsed) : "");
    setLabelBar(p.progress, line);
    $("#label-status").textContent = line;
    if (p.status === "done") {
      try {
        const merged = await (await window.fetchTO(`/api/result/${jobId}`)).json();
        $("#label-result-panel").style.display = "";
        // Every speaker here was labeled on the one active track, so per-line playback plays that
        // stream (not a downmix). Proxy answers any name, so a global rename can't break the lookup.
        const t = activeTrack;
        window.renderEditor($("#editor-label"), merged,
          { file_id: activeFileId, duration: activeDuration, trackOf: new Proxy({}, { get: () => t }) });
        setLabelBar(1, "done");
        $("#label-status").textContent = "done";
      } catch (e) { $("#label-status").textContent = "done — but couldn't load the result"; }
      finish();
      return;
    }
    if (p.status === "cancelled") { setLabelBar(p.progress, "canceled"); $("#label-status").textContent = "canceled"; finish(); return; }
    if (p.status === "error") { setLabelBar(p.progress, "error"); $("#label-status").textContent = "error: " + p.error; finish(); return; }
    setTimeout(tick, 600);
  };
  tick();
}
