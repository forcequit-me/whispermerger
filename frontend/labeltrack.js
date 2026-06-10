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

const PALETTE = ["#5b9dff", "#57b894", "#d9a441", "#e0625e", "#b07cff", "#46c6c6", "#e08a3c", "#9ac85b"];
const UNASSIGNED = "rgba(150,160,175,0.25)";
const BOUNDARY = "rgba(230,233,239,0.9)";
const alpha = (hex) => hex + "66";

const files = [];            // [{file_id, track_index, filename, duration, tracks}]
const speakers = [];         // [{name, color}]
const assign = new Map();    // region.id -> speaker name
const boundaryIds = new Set();
let ws = null, regions = null, anchorId = null, splitView = false;
const selection = new Set();   // region ids currently selected (bulk-assignable)
let activeFileId = null, activeTrack = 0, activeDuration = 0, boundaries = [0];
let buildToken = 0;

$("#label-choose").onclick = addFiles;
$("#label-vad").onclick = runVad;
$("#label-play").onclick = () => ws && ws.playPause();
$("#label-run").onclick = runTranscribe;
$("#label-view").onclick = () => {
  splitView = !splitView;
  $("#label-view").textContent = splitView ? "Single track" : "Split lanes";
  renderLanes();
};
$("#label-dup").onclick = duplicateSelection;
$("#label-zoom").oninput = () => { if (ws) try { ws.zoom(+$("#label-zoom").value); } catch (_) {} };

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
  const root = $("#label-files");
  root.innerHTML = "";
  files.forEach((f, i) => {
    const swap = (a, b) => { [files[a], files[b]] = [files[b], files[a]]; renderFiles(); rebuild(); };
    const up = el("button", { className: "ghost", textContent: "↑" });
    up.onclick = () => { if (i > 0) swap(i, i - 1); };
    const down = el("button", { className: "ghost", textContent: "↓" });
    down.onclick = () => { if (i < files.length - 1) swap(i, i + 1); };
    const rm = el("button", { className: "ghost", textContent: "✕" });
    rm.onclick = () => { files.splice(i, 1); renderFiles(); rebuild(); };

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
}

// --- build / load the (combined) waveform ----------------------------------------------

async function rebuild() {
  const token = ++buildToken;
  if (!files.length) { if (ws) { ws.destroy(); ws = null; } return; }

  let fid, track, dur, bounds;
  if (files.length === 1) {
    ({ file_id: fid, track_index: track, duration: dur } = files[0]);
    bounds = [0];
  } else {
    $("#label-status").textContent = "combining files…";
    const res = await fetch("/api/combine", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ files: files.map((f) => ({ file_id: f.file_id, track_index: f.track_index })) }),
    });
    if (!res.ok) { $("#label-err").textContent = "Combine failed: " + (await res.text()); return; }
    const r = await res.json();
    if (token !== buildToken) return;  // a newer change superseded this build
    fid = r.file_id; track = 0; dur = r.duration; bounds = r.boundaries;
    $("#label-status").textContent = "";
  }
  if (token !== buildToken) return;
  activeFileId = fid; activeTrack = track; activeDuration = dur; boundaries = bounds;
  initWave(fid, track);
}

function initWave(fid, track) {
  if (ws) { ws.destroy(); ws = null; }
  assign.clear();
  boundaryIds.clear();
  selection.clear();
  anchorId = null;
  ws = WaveSurfer.create({
    container: "#waveform", height: 140,
    waveColor: "#566", progressColor: "#5b9dff", cursorColor: "#e6e9ef",
    url: `/api/track-audio/${fid}/${track}`,
  });
  regions = ws.registerPlugin(RegionsPlugin.create());
  ws.on("decode", () => regions.enableDragSelection({ color: UNASSIGNED }));
  ws.on("ready", () => runVad());   // auto-detect speech on load
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

// Split-lane view: one row per speaker (+ unassigned), regions as blocks on a shared timescale.
function renderLanes() {
  const root = $("#label-lanes");
  if (!root) return;
  if (!splitView || !regions) { root.style.display = "none"; return; }
  root.style.display = "";
  root.innerHTML = "";
  const dur = activeDuration || 1;
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
  for (const row of rows) {
    const track = el("div", { className: "lane-track" });
    for (const r of row.regs) {
      const blk = el("div", { className: "lane-block", title: `${r.start.toFixed(2)}–${r.end.toFixed(2)}s` });
      blk.style.left = (r.start / dur * 100) + "%";
      blk.style.width = Math.max(0.3, (r.end - r.start) / dur * 100) + "%";
      blk.style.background = row.un ? UNASSIGNED : alpha(row.color);
      blk.style.outline = selection.has(r.id) ? "2px solid #e6e9ef" : "none";
      blk.onclick = () => { selection.clear(); selection.add(r.id); anchorId = r.id; highlight(); r.play(); };
      track.append(blk);
    }
    root.append(el("div", { className: "lane-row" },
      el("div", { className: "lane-label", textContent: row.name, style: `color:${row.color}` }), track));
  }
}

// --- speakers + assignment -------------------------------------------------------------

function renderSpeakers() {
  const root = $("#label-speakers");
  root.innerHTML = "";
  speakers.forEach((sp, i) => {
    const b = el("button", { className: "ghost", title: `assign selected region (key ${i + 1})` },
      el("span", { textContent: "■ ", style: `color:${sp.color}` }),
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

async function runVad() {
  $("#label-status").textContent = "detecting speech…";
  const res = await fetch(`/api/vad/${activeFileId}/${activeTrack}`, { method: "POST" });
  if (!res.ok) { $("#label-status").textContent = "VAD failed: " + (await res.text()); return; }
  const spans = await res.json();
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
  const res = await fetch("/api/diarize", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file_id: activeFileId, track_index: activeTrack, speakers: map, options: { detect_conflicts: true } }),
  });
  if (!res.ok) { $("#label-status").textContent = "Failed: " + (await res.text()); btn.disabled = false; return; }
  const { job_id } = await res.json();
  poll(job_id, btn);
}

function poll(jobId, btn) {
  const tick = async () => {
    const p = await (await fetch(`/api/progress/${jobId}`)).json();
    $("#label-status").textContent = (p.message || p.status) + (p.elapsed ? " · " + window.fmtElapsed(p.elapsed) : "");
    if (p.status === "done") {
      const merged = await (await fetch(`/api/result/${jobId}`)).json();
      $("#label-result-panel").style.display = "";
      window.renderEditor($("#editor-label"), merged, { file_id: activeFileId, duration: activeDuration });
      $("#label-status").textContent = "done";
      btn.disabled = false;
      return;
    }
    if (p.status === "error") { $("#label-status").textContent = "error: " + p.error; btn.disabled = false; return; }
    setTimeout(tick, 600);
  };
  tick();
}
