"use strict";

// --- surface any uncaught JS error on the page (so it can be screenshotted, no DevTools) ---
function showFatal(msg) {
  let bar = document.getElementById("js-error-bar");
  if (!bar) {
    bar = document.createElement("div");
    bar.id = "js-error-bar";
    bar.style.cssText = "position:fixed;left:0;right:0;bottom:0;z-index:99999;background:#e0625e;" +
      "color:#0b1220;font:12px/1.4 ui-monospace,monospace;padding:10px 14px;white-space:pre-wrap;" +
      "max-height:45vh;overflow:auto;border-top:2px solid #0b1220";
    (document.body || document.documentElement).appendChild(bar);
  }
  bar.textContent = "⚠ JS error (please screenshot this):\n" + msg;
}
window.addEventListener("error", (e) => {
  const where = e.error && e.error.stack ? e.error.stack : `${e.filename}:${e.lineno}:${e.colno}`;
  showFatal((e.message || e.error) + "\n" + where);
});
window.addEventListener("unhandledrejection", (e) => {
  const r = e.reason || {};
  showFatal("unhandled promise rejection: " + (r.message || r) + "\n" + (r.stack || ""));
});

// --- tiny helpers ----------------------------------------------------------------------
const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, props = {}, ...kids) => {
  const n = Object.assign(document.createElement(tag), props);
  for (const k of kids) n.append(k);
  return n;
};

// Inline SVG icons — no emoji (Phosphor/Radix-style). currentColor inherits the button's color.
const ICONS = {
  play:  { fill: true, d: "M8 5v14l11-7z" },
  pause: { fill: true, d: "M7 5h3v14H7zM14 5h3v14h-3z" },
  close: { d: "M6 6l12 12M18 6 6 18" },
  up:    { d: "M6 14l6-6 6 6" },
  down:  { d: "M6 10l6 6 6-6" },
  warn:  { d: "M12 4l9 16H3zM12 10v4M12 17h.01" },
};
function icon(name, size = 15) {
  const ns = "http://www.w3.org/2000/svg";
  const spec = ICONS[name];
  const s = document.createElementNS(ns, "svg");
  s.setAttribute("viewBox", "0 0 24 24");
  s.setAttribute("width", size); s.setAttribute("height", size);
  s.setAttribute("aria-hidden", "true");
  if (spec.fill) s.setAttribute("fill", "currentColor");
  else Object.entries({ fill: "none", stroke: "currentColor", "stroke-width": "2",
    "stroke-linecap": "round", "stroke-linejoin": "round" }).forEach(([k, v]) => s.setAttribute(k, v));
  const p = document.createElementNS(ns, "path");
  p.setAttribute("d", spec.d);
  s.append(p);
  return s;
}
function setIcon(node, name, size) { node.replaceChildren(icon(name, size)); }
window.uiIcon = icon;   // labeltrack.js (module) reuses these

// --- spacebar transport ----------------------------------------------------------------
// Tracks the single "currently active" player (an <Audio> or the WaveSurfer instance). The
// most recently played thing wins. Spacebar toggles it — unless focus is in a text field or
// on a button, where space must keep its normal meaning.
const wmTransport = {
  active: null,
  set(player) { this.active = player; },
  toggle() { if (this.active) { this.active.toggle(); return true; } return false; },
};
window.wmTransport = wmTransport;                                  // labeltrack.js registers ws here
const htmlPlayer = (el) => ({ toggle: () => { el.paused ? el.play() : el.pause(); } });

document.addEventListener("keydown", (e) => {
  if (e.key !== " " && e.code !== "Space") return;
  const t = e.target, tag = (t.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea" || tag === "select" || t.isContentEditable) return;
  if (tag === "button" || (t.getAttribute && t.getAttribute("role") === "button")) return;
  if (wmTransport.toggle()) e.preventDefault();                   // we handled it — don't scroll the page
});
const fmtTime = (s) => {
  s = Math.max(0, s || 0);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = (s % 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${sec.toFixed(2).padStart(5, "0")}`;
};
// elapsed-time label, shared with the label-track module via window
function fmtElapsed(s) {
  s = Math.round(s || 0);
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}
window.fmtElapsed = fmtElapsed;

// fetch with an abort-timeout, so a wedged/slow request can never freeze a polling loop
// forever (the server goes briefly unresponsive while it loads models on the first run).
async function fetchTO(url, opts = {}, ms = 15000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), ms);
  try { return await fetch(url, { ...opts, signal: ctrl.signal }); }
  finally { clearTimeout(timer); }
}
window.fetchTO = fetchTO;   // labeltrack.js (module) reuses it

// --- shutdown --------------------------------------------------------------------------
$("#shutdown-btn").onclick = async () => {
  if (!confirm("Stop the WhisperMerger server?")) return;
  try { await fetch("/api/shutdown", { method: "POST" }); } catch (_) { /* server is going down */ }
  document.body.innerHTML =
    '<div style="display:flex;height:100vh;align-items:center;justify-content:center;color:#8a93a3;font:15px sans-serif">' +
    "Server stopped — you can close this tab.</div>";
};

// --- tab switching ---------------------------------------------------------------------
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.onclick = () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b === btn));
    document.querySelectorAll(".tab").forEach((t) =>
      t.classList.toggle("active", t.id === `tab-${btn.dataset.tab}`));
  };
});

// =======================================================================================
// Shared segment editor + export panel. Operates on `merged` (a {version,sources,segments})
// in place. Words ride underneath untouched; editing text flags the segment "edited".
// =======================================================================================
// Preview audio: jump-to-position via instant input-seek clips (works on huge files).
let previewAudio = null;
let previewBtn = null;   // the per-line ▶/⏸ button whose clip is currently loaded
function playClip(fileId, start, dur, track) {
  if (previewAudio) previewAudio.pause();
  let url = `/api/clip/${fileId}?start=${start}&dur=${Math.max(2, Math.min(dur, 30))}`;
  if (track != null) url += `&track=${track}`;   // isolate one speaker's stream; omit → downmix
  previewAudio = new Audio(url);
  previewAudio.play();
  wmTransport.set(htmlPlayer(previewAudio));
}

// Per-line preview button: ▶ plays this line's clip and flips to ⏸ while it's playing; click
// again to pause/resume. Only one clip plays at a time, so starting another line reverts the
// previous button to ▶. Icons follow the audio's real state, so spacebar stays in sync.
function toggleClip(btn, fileId, start, dur, track) {
  if (btn === previewBtn && previewAudio && !previewAudio.paused) { previewAudio.pause(); return; }
  playClip(fileId, start, dur, track);
  const audio = previewAudio;
  if (previewBtn && previewBtn !== btn) { setIcon(previewBtn, "play"); previewBtn.title = "Play from here"; }
  previewBtn = btn;
  const showPlay = () => { setIcon(btn, "play"); btn.title = "Play from here"; };
  const showPause = () => { setIcon(btn, "pause"); btn.title = "Pause"; };
  showPause();
  audio.onplay = () => { showPause(); wmTransport.set(htmlPlayer(audio)); };
  audio.onpause = showPlay;
  audio.onended = showPlay;
}
function fmtClock(s) {
  s = Math.floor(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function renderEditor(container, merged, preview) {
  if (preview !== undefined) container._preview = preview;  // persist across re-renders
  preview = container._preview || null;
  // Per-speaker export toggles, keyed to the loaded transcript: a new `merged` resets them,
  // in-place re-renders (same object) keep them.
  if (container._merged !== merged) { container._merged = merged; container._disabled = new Set(); }
  container.innerHTML = "";
  container.append(buildToolbar(container, merged), buildSpeakerBar(container, merged),
    buildExportPanel(container, merged));
  if (preview) { container.append(buildPreviewBar(preview)); container.append(buildWaveform(preview)); }
  container.append(buildTable(container, merged, preview));
}

// Lightweight canvas waveform fed by server-side peaks (/api/peaks). No full-audio download.
// Click to play from that point (input-seek clip); horizontal zoom for precision on long files.
function buildWaveform(preview) {
  const scroller = el("div", { style: "overflow-x:auto; background:var(--panel-2); border:1px solid var(--line); border-radius:8px" });
  const canvas = el("canvas", { height: 120, style: "display:block; height:120px; cursor:pointer" });
  scroller.append(canvas);
  const zoom = el("input", { type: "range", min: 1, max: 20, value: 1, step: 1, title: "horizontal zoom" });
  const vzoom = el("input", { type: "range", min: 1, max: 20, value: 1, step: 0.5, title: "vertical (amplitude) zoom — for quiet audio" });
  const status = el("span", { className: "muted", textContent: "loading waveform…" });
  let peaks = [], duration = preview.duration || 1, playhead = 0, raf = 0;

  const draw = () => {
    const base = Math.max(scroller.clientWidth || 600, 600);
    const w = Math.floor(base * +zoom.value), h = canvas.height, gain = +vzoom.value;
    canvas.width = w; canvas.style.width = w + "px";
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#566";
    const bw = w / (peaks.length || 1);
    peaks.forEach((p, i) => {
      const ph = Math.min(h, Math.max(1, p * h * 0.9 * gain));
      ctx.fillRect(i * bw, (h - ph) / 2, Math.max(1, bw - 0.5), ph);
    });
    ctx.fillStyle = "#5b9dff";
    ctx.fillRect((playhead / duration) * w, 0, 2, h);
  };

  fetch(`/api/peaks/${preview.file_id}`)
    .then((r) => r.json())
    .then((d) => { peaks = d.peaks || []; duration = d.duration || duration; status.textContent = "click to play · zoom for precision"; draw(); })
    .catch(() => { status.textContent = "waveform unavailable"; });

  canvas.onclick = (e) => {
    const rect = canvas.getBoundingClientRect();
    const t0 = (e.clientX - rect.left) / canvas.width * duration;
    playClip(preview.file_id, t0, 5);
    playhead = t0; draw();
    cancelAnimationFrame(raf);
    const tick = () => {
      if (previewAudio && !previewAudio.paused) {
        playhead = t0 + previewAudio.currentTime; draw();
        raf = requestAnimationFrame(tick);
      }
    };
    raf = requestAnimationFrame(tick);
  };
  zoom.oninput = draw;
  vzoom.oninput = draw;

  return el("div", {},
    el("div", { className: "row", style: "margin:10px 0 6px" },
      el("span", { className: "muted", textContent: "Waveform" }), status,
      el("div", { className: "spacer" }),
      el("span", { className: "muted", textContent: "zoom" }), zoom,
      el("span", { className: "muted", textContent: "amp" }), vzoom),
    scroller);
}

function buildPreviewBar(preview) {
  const slider = el("input", { type: "range", min: 0, max: Math.floor(preview.duration), value: 0, step: 1, style: "flex:1" });
  const label = el("span", { className: "muted", textContent: "00:00:00", style: "width:70px; text-align:right" });
  slider.oninput = () => { label.textContent = fmtClock(+slider.value); };
  slider.onchange = () => playClip(preview.file_id, +slider.value, 5);
  return el("div", { className: "row", style: "margin:10px 0" },
    el("span", { className: "muted", textContent: "Preview" }), slider, label,
    el("span", { className: "muted", textContent: "drag to jump · or ▶ a line" }));
}

function distinctSpeakers(merged) {
  return [...new Set(merged.segments.map((s) => s.speaker))];
}

function buildToolbar(container, merged) {
  const from = el("select");
  const refreshFrom = () => {
    from.innerHTML = "";
    distinctSpeakers(merged).forEach((s) => from.append(el("option", { value: s, textContent: s })));
  };
  refreshFrom();
  const to = el("input", { placeholder: "new name", size: 12 });
  const renameBtn = el("button", { className: "ghost", textContent: "Rename globally" });
  renameBtn.onclick = () => {
    if (!to.value.trim()) return;
    merged.segments.forEach((s) => { if (s.speaker === from.value) s.speaker = to.value.trim(); });
    to.value = "";
    renderEditor(container, merged);
  };

  const removeBtn = el("button", { className: "ghost", textContent: "Remove speaker" });
  removeBtn.onclick = () => {
    const spk = from.value;
    const count = merged.segments.filter((s) => s.speaker === spk).length;
    if (!count) return;
    if (!confirm(`Remove speaker "${spk}" and all ${count} of their line(s)? This can't be undone.`)) return;
    merged.segments = merged.segments.filter((s) => s.speaker !== spk);
    renderEditor(container, merged);
  };

  const addBtn = el("button", { className: "ghost", textContent: "+ Add line" });
  addBtn.onclick = () => {
    const last = merged.segments[merged.segments.length - 1];
    const start = last ? last.end : 0;
    merged.segments.push({ speaker: from.value || "Speaker", start, end: start + 1,
      text: "", words: [], flags: [] });
    renderEditor(container, merged);
  };

  return el("div", { className: "row", style: "margin-bottom:12px" },
    el("span", { className: "muted", textContent: "Speaker:" }), from, to, renameBtn, removeBtn,
    el("div", { className: "spacer" }), addBtn);
}

// Per-speaker include/exclude toggle. A disabled speaker is dropped from every export (all
// formats) and dimmed in the table — non-destructive, unlike "Remove speaker" above.
function buildSpeakerBar(container, merged) {
  const disabled = container._disabled;
  const counts = {};
  merged.segments.forEach((s) => { counts[s.speaker] = (counts[s.speaker] || 0) + 1; });
  const chips = distinctSpeakers(merged).map((spk) => {
    const off = disabled.has(spk);
    const chip = el("button", { className: "spk-toggle" + (off ? " off" : ""),
      title: off ? "Excluded from export — click to include" : "Click to exclude from export" },
      el("span", { className: "spk-dot" }),
      el("span", { textContent: spk }),
      el("span", { className: "spk-count", textContent: counts[spk] }));
    chip.onclick = () => { off ? disabled.delete(spk) : disabled.add(spk); renderEditor(container, merged); };
    return chip;
  });
  return el("div", { className: "row spk-bar", style: "margin-bottom:12px" },
    el("span", { className: "muted", textContent: "Include:" }), ...chips);
}

function buildTable(container, merged, preview) {
  const tbody = el("tbody");
  // "Notes" column only shows once a visible line carries a flag (conflict / edited).
  const hasNotes = merged.segments.some((s) => !container._disabled.has(s.speaker) && (s.flags || []).length);
  const table = el("table", { className: "editor" + (hasNotes ? " has-notes" : "") },
    el("thead", {}, el("tr", {},
      el("th", { textContent: "Speaker" }), el("th", { textContent: "Start" }),
      el("th", { textContent: "End" }), el("th", { textContent: "Text" }),
      el("th", { className: "notes-col", textContent: "Notes" }),
      el("th", { className: "act-col", textContent: "" }))),
    tbody);

  merged.segments.forEach((seg, i) => {
    if (container._disabled.has(seg.speaker)) return;  // hidden — toggle the speaker back on to show
    tbody.append(buildRow(container, merged, seg, i, preview));
  });
  return table;
}

function buildRow(container, merged, seg, i, preview) {
  const tr = el("tr");
  for (const f of seg.flags || []) tr.classList.add(`flag-${f}`);
  const snapshot = JSON.stringify(seg);

  const spk = el("input", { value: seg.speaker });
  spk.oninput = () => { seg.speaker = spk.value; };

  const start = el("input", { value: fmtTimeShort(seg.start) });
  const end = el("input", { value: fmtTimeShort(seg.end) });
  start.onchange = () => { seg.start = parseTime(start.value); };
  end.onchange = () => { seg.end = parseTime(end.value); };

  const txt = el("textarea", { value: seg.text, rows: 1 });
  txt.oninput = () => {
    seg.text = txt.value;
    autoGrow(txt);                        // height tracks wrapped/multi-line content
    if (seg.words && seg.words.length && !seg.flags.includes("edited")) {
      seg.flags.push("edited");           // text changed -> word timings may be stale
      renderFlags(flagsCell, seg);
      tr.classList.add("flag-edited");
      tr.closest("table.editor")?.classList.add("has-notes");   // reveal the Notes column now
    }
  };
  // Enter inserts a line break (natural textarea behavior). Escape reverts; Ctrl+↑/↓ moves rows.
  txt.onkeydown = (e) => {
    if (e.key === "Escape") { Object.assign(seg, JSON.parse(snapshot)); renderEditor(container, merged); }
    else if (e.key === "ArrowDown" && e.ctrlKey) { e.preventDefault(); focusSibling(tr, 1); }
    else if (e.key === "ArrowUp" && e.ctrlKey) { e.preventDefault(); focusSibling(tr, -1); }
  };
  requestAnimationFrame(() => autoGrow(txt));   // size to initial content once in the DOM

  const flagsCell = el("td", { className: "notes-col" });
  renderFlags(flagsCell, seg);

  const del = el("button", { className: "ghost icon-only", title: "Remove line" }, icon("close"));
  del.onclick = () => { merged.segments.splice(i, 1); renderEditor(container, merged); };

  const rowActions = el("div", { className: "row-actions" });
  if (preview) {
    const jump = el("button", { className: "ghost icon-only", title: "Play from here" }, icon("play"));
    // play only this speaker's stream when we know the mapping (multi-track source); else downmix
    const track = preview.trackOf ? preview.trackOf[seg.speaker] : undefined;
    jump.onclick = () => toggleClip(jump, preview.file_id, seg.start, seg.end - seg.start, track);
    rowActions.append(jump);
  }
  rowActions.append(del);
  const actions = el("td", { className: "act-col" }, rowActions);

  tr.append(
    el("td", { className: "spk" }, spk),
    el("td", { className: "tc" }, start),
    el("td", { className: "tc" }, end),
    el("td", { className: "txt" }, txt),
    flagsCell,
    actions);
  tr.dataset.rowText = "";
  tr._textInput = txt;
  return tr;
}

function renderFlags(cell, seg) {
  cell.innerHTML = "";
  for (const f of seg.flags || []) cell.append(el("span", { className: `chip ${f}`, textContent: f }));
}

// Move focus to the adjacent rendered row's text (skips hidden/filtered rows automatically).
function focusSibling(tr, dir) {
  const next = dir > 0 ? tr.nextElementSibling : tr.previousElementSibling;
  if (next && next._textInput) next._textInput.focus();
}

// Grow a text cell's textarea to fit its content (top stays put, bottom extends).
function autoGrow(ta) {
  if (!ta.scrollHeight) return;   // not laid out (e.g. hidden tab) — keep CSS min-height
  ta.style.height = "auto";
  ta.style.height = ta.scrollHeight + "px";
}

// timecode helpers for the editor inputs (HH:MM:SS.ss <-> seconds)
function fmtTimeShort(s) { return fmtTime(s); }
function parseTime(v) {
  if (v == null) return 0;
  v = String(v).trim();
  if (v.includes(":")) {
    const p = v.split(":").map(Number);
    return p.length === 3 ? p[0] * 3600 + p[1] * 60 + p[2] : p[0] * 60 + p[1];
  }
  return parseFloat(v) || 0;
}

// --- export panel ----------------------------------------------------------------------
// Default export name: "<base> Transcript (M-D-YYYY h-mm AMPM)" — mirrors the backend batch
// naming so an interactively-exported file looks like a batch-exported one. Dash (not colon) in
// the time keeps it a legal filename.
function transcriptName(base) {
  const d = new Date();
  let h = d.getHours() % 12; if (h === 0) h = 12;
  const ap = d.getHours() < 12 ? "AM" : "PM";
  const mm = String(d.getMinutes()).padStart(2, "0");
  const stamp = `${d.getMonth() + 1}-${d.getDate()}-${d.getFullYear()} ${h}-${mm} ${ap}`;
  return `${base || "transcript"} Transcript (${stamp})`;
}

// Base name from the loaded transcript's first source (stem), stripping any prior
// " Transcript (…)" so re-exporting an exported file doesn't compound the label.
function exportBase(merged) {
  const src = merged.sources && merged.sources[0] && merged.sources[0].filename;
  let base = src ? src.replace(/\.[^.]+$/, "") : "transcript";       // drop extension
  base = base.replace(/\s*Transcript \([^)]*\)\s*$/, "").trim();
  return base || "transcript";
}

function buildExportPanel(container, merged) {
  const format = el("select");
  ["srt", "txt", "csv", "json"].forEach((f) => format.append(el("option", { value: f, textContent: f.toUpperCase() })));
  const filename = el("input", { value: transcriptName(exportBase(merged)), size: 36 });
  const applyLimits = el("input", { type: "checkbox", checked: true });
  const cpl = el("input", { type: "number", value: 42 });
  const maxLines = el("input", { type: "number", value: 2 });
  const minDur = el("input", { type: "number", value: 1, step: 0.5 });
  const pad = el("input", { type: "number", value: 0, step: 0.1, min: 0, title: "extend each segment start/end by N seconds (video-edit breathing room)" });
  const showSpk = el("input", { type: "checkbox", checked: true });

  // Subtitle limits only apply to SRT; toggling off disables (and ignores) the limit fields.
  const syncLimits = () => [cpl, maxLines, minDur].forEach((n) => { n.disabled = !applyLimits.checked; });
  applyLimits.onchange = syncLimits;
  syncLimits();

  const btn = el("button", { className: "primary", textContent: "Export" });
  btn.onclick = async () => {
    const options = {
      apply_limits: applyLimits.checked,
      chars_per_line: +cpl.value, max_lines: +maxLines.value,
      min_duration: +minDur.value, show_speakers: showSpk.checked,
      pad_seconds: +pad.value,
    };
    // Drop toggled-off speakers (all formats) — the backend just renders what we send.
    const segs = merged.segments.filter((s) => !container._disabled.has(s.speaker));
    if (!segs.length) { alert("All speakers are toggled off — nothing to export."); return; }
    await downloadExport({ ...merged, segments: segs }, format.value, filename.value || "transcript", options);
  };

  const field = (label, node) => el("label", { className: "field" }, el("span", { textContent: label }), node);
  return el("div", { className: "row", style: "margin:14px 0; padding-top:14px; border-top:1px solid var(--line)" },
    field("Format", format), field("Filename", filename),
    el("label", { className: "opt" }, applyLimits, "Subtitle limits"),
    field("Chars/line", cpl), field("Max lines", maxLines), field("Min dur (s)", minDur),
    field("Pad (s)", pad),
    el("label", { className: "opt" }, showSpk, "Show speakers"),
    el("div", { className: "spacer" }), btn);
}

async function downloadExport(merged, format, filename, options) {
  const res = await fetch(`/api/export?format=${format}`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ merged, options, filename }),
  });
  if (!res.ok) { alert("Export failed: " + (await res.text())); return; }
  const blob = await res.blob();
  const a = el("a", { href: URL.createObjectURL(blob), download: `${filename}.${format}` });
  document.body.append(a); a.click(); a.remove();
}

// =======================================================================================
// Transcribe & Merge tab
// =======================================================================================
const uploaded = [];  // [{file_id, source}]

// Declared before initTranscribeTab() runs — it calls syncDetect() immediately at load.
const MERGE_THRESHOLD_IDS = ["opt-backchannel", "opt-bc-words", "opt-overlap", "opt-bleed-win", "opt-bleed-sim"];

(function initTranscribeTab() {
  $("#choose-btn").onclick = () => browseFiles($("#choose-btn"));
  $("#choose-more").onclick = () => browseFiles($("#choose-more"));
  $("#run-btn").onclick = runPipeline;
  const detect = $("#opt-detect");
  const syncDetect = () => MERGE_THRESHOLD_IDS.forEach((id) => { $(`#${id}`).disabled = !detect.checked; });
  detect.onchange = syncDetect;
  syncDetect();
})();

function registerItems(items) {
  items.forEach((it) => uploaded.push(it));
  renderSources();
  $("#transcribe-entry").style.display = "none";
  $("#input-panel").style.display = "";
  $("#options-panel").style.display = "";
}

async function browseFiles(btn) {
  btn = btn || $("#choose-btn");
  const err = $("#path-err");
  err.textContent = "";
  btn.disabled = true;
  const label = btn.textContent;
  btn.textContent = "Opening file picker…";
  try {
    const res = await fetch("/api/browse", { method: "POST" });
    if (!res.ok) { err.textContent = "Could not open picker: " + (await res.text()); return; }
    const items = await res.json();
    if (items.length) registerItems(items);
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
}

// FR-TK-3: per-track audition. One clip plays at a time.
let currentAudio = null;
function auditionBtn(fileId, trackIndex) {
  const btn = el("button", { className: "ghost icon-only", title: "Audition track" });
  setIcon(btn, "play");
  const audio = new Audio(`/api/audition/${fileId}/${trackIndex}`);
  // Icon follows the audio's real state, so spacebar pause/resume keeps it in sync.
  audio.onplay = () => { setIcon(btn, "pause"); wmTransport.set(htmlPlayer(audio)); };
  audio.onpause = () => setIcon(btn, "play");
  audio.onended = () => setIcon(btn, "play");
  btn.onclick = () => {
    if (currentAudio && currentAudio !== audio) { currentAudio.pause(); currentAudio.currentTime = 0; }
    audio.paused ? (currentAudio = audio, audio.play()) : audio.pause();
  };
  return btn;
}

function renderSources() {
  const root = $("#sources");
  root.innerHTML = "";
  uploaded.forEach((it, idx) => {
    const box = el("div", { className: "source" });
    box.append(el("div", { className: "name" },
      el("span", { textContent: it.source.filename }),
      el("span", { className: "dur", textContent: `${it.source.duration.toFixed(1)}s` })));

    // FR-TK-4: track selector only when more than one stream.
    if (!it.source.tracks.length) {
      box.append(el("div", { className: "err", textContent: "No audio tracks found in this file — it can't be transcribed." }));
    } else if (it.source.tracks.length > 1) {
      it.source.tracks.forEach((t) => {
        const sel = el("input", { type: "checkbox", checked: t.selected });
        sel.onchange = () => { t.selected = sel.checked; };
        const spk = el("input", { value: t.speaker });
        spk.oninput = () => { t.speaker = spk.value; };
        box.append(el("div", { className: "track-row" },
          sel, el("span", { className: "ti", textContent: `track ${t.track_index + 1}` }),
          spk, auditionBtn(it.file_id, t.track_index)));
      });
    } else {
      const t = it.source.tracks[0];
      const spk = el("input", { value: t.speaker });
      spk.oninput = () => { t.speaker = spk.value; };
      box.append(el("div", { className: "track-row" },
        el("span", { className: "ti", textContent: "speaker" }),
        spk, auditionBtn(it.file_id, t.track_index)));
    }

    // sequence order (UC2) — only shown with multiple files
    if (uploaded.length > 1) {
      const seq = el("input", { type: "number", value: idx, style: "width:60px" });
      seq.onchange = () => { it.source.sequence = +seq.value; };
      it.source.sequence = idx;
      box.append(el("div", { className: "track-row" },
        el("span", { className: "ti", textContent: "order" }), seq));
    }
    root.append(box);
  });
}

function gatherOptions() {
  return {
    detect_conflicts: $("#opt-detect").checked,
    backchannel_render: $("#opt-backchannel").value,
    backchannel_max_words: +$("#opt-bc-words").value,
    overlap_min_seconds: +$("#opt-overlap").value,
    bleed_window_seconds: +$("#opt-bleed-win").value,
    bleed_similarity: +$("#opt-bleed-sim").value,
  };
}

async function runPipeline() {
  const sources = uploaded.map((it) => ({
    file_id: it.file_id,
    sequence: it.source.sequence || 0,
    tracks: it.source.tracks.map((t) => ({
      track_index: t.track_index, speaker: t.speaker, selected: t.selected !== false,
    })),
  }));
  const runBtn = $("#run-btn");
  runBtn.disabled = true;
  $("#progress-wrap").classList.add("show");
  setProgress(0, "starting…");

  let res;
  try {
    res = await fetchTO("/api/transcribe", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sources, options: gatherOptions(), language: $("#opt-language").value || null }),
    }, 30000);
  } catch (e) {
    setProgress(0, "couldn't reach the server to start — is it still running?");
    runBtn.disabled = false; return;
  }
  if (!res.ok) { setProgress(0, "error: " + (await res.text())); runBtn.disabled = false; return; }
  const { job_id } = await res.json();
  pollJob(job_id, runBtn);
}

function pollJob(jobId, runBtn) {
  let fails = 0;
  const cancelBtn = $("#cancel-job");
  cancelBtn.style.display = ""; cancelBtn.disabled = false; cancelBtn.textContent = "Cancel";
  cancelBtn.onclick = async () => {
    cancelBtn.disabled = true; cancelBtn.textContent = "Canceling…";
    try { await fetch(`/api/cancel/${jobId}`, { method: "POST" }); } catch {}
  };
  const finish = () => { runBtn.disabled = false; cancelBtn.style.display = "none"; cancelBtn.textContent = "Cancel"; };
  const tick = async () => {
    let p;
    try {
      const r = await fetchTO(`/api/progress/${jobId}`);
      if (!r.ok) throw new Error(await r.text());
      p = await r.json();
      fails = 0;
    } catch (e) {
      // Server briefly unresponsive (first-run model load ties it up) — keep waiting, never freeze.
      $("#progress-msg").textContent = (++fails > 5)
        ? "still working… first-run model load can take a few minutes (watch the server window)"
        : "working… waiting for the server";
      setTimeout(tick, 1200);
      return;
    }
    setProgress(p.progress, `${p.message || p.status}${p.elapsed ? " · " + fmtElapsed(p.elapsed) : ""}`);
    if (p.status === "done") {
      try {
        const merged = await (await fetchTO(`/api/result/${jobId}`)).json();
        $("#result-panel").style.display = "";
        // preview audio only when one source (merged timeline == that file's timeline).
        // trackOf maps speaker → audio-stream index so a per-line preview plays just that mic.
        const preview = uploaded.length === 1
          ? { file_id: uploaded[0].file_id, duration: uploaded[0].source.duration,
              trackOf: Object.fromEntries(uploaded[0].source.tracks.map((t) => [t.speaker, t.track_index])) }
          : null;
        renderEditor($("#editor-transcribe"), merged, preview);
        setProgress(1, "done");
      } catch (e) { setProgress(1, "done — but couldn't load the result: " + e.message); }
      finish();
      return;
    }
    if (p.status === "cancelled") { setProgress(p.progress, "canceled"); finish(); return; }
    if (p.status === "error") { setProgress(p.progress, "error: " + p.error); finish(); return; }
    setTimeout(tick, 500);
  };
  tick();
}

// Map a progress message to its pipeline stage so the stepper can light up.
function stageOf(msg) {
  const m = (msg || "").toLowerCase();
  if (m.includes("merg")) return "merge";
  if (m.includes("transcrib")) return "transcribe";
  if (m.includes("extract")) return "extract";
  return "";
}
const STAGE_ORDER = ["extract", "transcribe", "merge"];

function setProgress(frac, msg) {
  const pct = Math.round((frac || 0) * 100);
  $("#progress-bar").style.width = `${pct}%`;
  $("#progress-pct").textContent = `${pct}%`;
  $("#progress-msg").textContent = msg || "";

  const done = /\bdone\b/i.test(msg || "");
  const stage = done ? "merge" : stageOf(msg);
  const active = STAGE_ORDER.indexOf(stage);
  $("#stage-steps").querySelectorAll("li").forEach((li, i) => {
    li.classList.toggle("active", !done && i === active);
    li.classList.toggle("done", done || (active >= 0 && i < active));
  });
}

// =======================================================================================
// Edit Transcript tab
// =======================================================================================
let editMerged = null, editPreview = null;
(function initEditTab() {
  const dz = $("#dropzone-edit"), input = $("#file-input-edit");
  dz.onclick = () => input.click();
  dz.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } };
  input.onchange = () => input.files[0] && readTranscriptFile(input.files[0]);
  dz.ondragover = (e) => { e.preventDefault(); dz.classList.add("over"); };
  dz.ondragleave = () => dz.classList.remove("over");
  dz.ondrop = (e) => { e.preventDefault(); dz.classList.remove("over"); e.dataTransfer.files[0] && readTranscriptFile(e.dataTransfer.files[0]); };
  $("#json-load").onclick = () => loadEdit($("#json-in").value);
  $("#edit-attach").onclick = attachEditAudio;
  $("#edit-load-another").onclick = resetEdit;
})();

// Collapse the editor back to the loader so another transcript can be dropped in.
function resetEdit() {
  editMerged = null; editPreview = null;
  $("#editor-edit").innerHTML = "";
  $("#json-err").textContent = "";
  $("#edit-panel").style.display = "none";
  $("#dropzone-edit").style.display = "";
  $("#paste-details").style.display = "";
}

async function attachEditAudio() {
  if (!editMerged) return;
  const res = await fetch("/api/browse", { method: "POST" });
  if (!res.ok) return;
  const items = await res.json();
  if (!items.length) return;
  const it = items[0];
  editPreview = { file_id: it.file_id, duration: it.source.duration };
  $("#edit-attach-name").textContent = it.source.filename;
  renderEditor($("#editor-edit"), editMerged, editPreview);
}

function readTranscriptFile(file) {
  const r = new FileReader();
  r.onload = () => loadEdit(r.result, file.name);
  r.readAsText(file);
}

function loadEdit(text, name) {
  const errSpan = $("#json-err");
  errSpan.textContent = "";
  let merged;
  try { merged = parseTranscript(text, name); }
  catch (e) { errSpan.textContent = e.message || "Couldn't read this file"; return; }
  merged.version = merged.version || "1.0";
  merged.sources = merged.sources || [];
  merged.segments.forEach((s) => { s.words = s.words || []; s.flags = s.flags || []; });
  editMerged = merged; editPreview = null;
  $("#edit-attach-name").textContent = "attach the source file to scrub the waveform & play lines";
  $("#dropzone-edit").style.display = "none";
  $("#paste-details").style.display = "none";
  $("#edit-panel").style.display = "";
  renderEditor($("#editor-edit"), merged, null);  // no source yet — Attach audio to enable preview
}

// --- transcript import: parse any format we export back into a {segments:[...]} object ---
// JSON keeps word-level timings; SRT/TXT/CSV are lossy (words:[]), mirroring backend/app/export.py.
function parseTranscript(text, name) {
  const ext = (name || "").toLowerCase().split(".").pop();
  const fmt = ["json", "srt", "csv", "txt"].includes(ext) ? ext : sniffFormat(text);
  let merged;
  if (fmt === "json") merged = parseJsonTranscript(text);
  else if (fmt === "csv") merged = { segments: parseCsv(text) };
  else if (fmt === "srt") merged = { segments: parseSrt(text) };
  else merged = { segments: parseTxt(text) };
  if (!merged || !Array.isArray(merged.segments) || !merged.segments.length)
    throw new Error("No transcript lines found in this file");
  return merged;
}

function sniffFormat(text) {
  const t = (text || "").trim();
  if (t.startsWith("{") || t.startsWith("[{")) return "json";
  if (/^\s*start\s*,\s*end\s*,\s*speaker/i.test(t)) return "csv";
  if (/^\s*\[\d\d?:\d\d:\d\d/.test(t)) return "txt";   // [hh:mm:ss,mmm --> ...] text
  if (/-->/.test(t)) return "srt";
  return "json";
}

function parseJsonTranscript(text) {
  let merged;
  try { merged = JSON.parse(text); } catch (e) { throw new Error("Invalid JSON"); }
  if (!merged || !Array.isArray(merged.segments)) throw new Error("Missing segments[]");
  return merged;
}

// "HH:MM:SS,mmm" -> seconds (tolerates "." as the millisecond separator)
function parseSrtTime(v) {
  const m = String(v).trim().match(/(\d+):(\d\d):(\d\d)[,.](\d{1,3})/);
  if (!m) return 0;
  return +m[1] * 3600 + +m[2] * 60 + +m[3] + +(`0.${m[4]}`);
}

// Pull a leading "Name: " speaker label off a line (as written by show_speakers exports).
function splitSpeaker(s) {
  const m = s.match(/^([^:]{1,40}):\s(.*)$/);
  return m ? { speaker: m[1].trim(), text: m[2] } : { speaker: "Speaker", text: s };
}

function parseSrt(text) {
  const segs = [];
  for (const block of text.replace(/\r/g, "").split(/\n\s*\n/)) {
    const lines = block.split("\n").filter((l) => l.trim() !== "");
    if (!lines.length) continue;
    let i = /^\d+$/.test(lines[0].trim()) ? 1 : 0;   // optional cue-index line
    const tc = lines[i] && lines[i].match(/(.+?)\s*-->\s*(.+)/);
    if (!tc) continue;
    const { speaker, text: t } = splitSpeaker(lines.slice(i + 1).join(" ").trim());
    segs.push({ speaker, start: parseSrtTime(tc[1]), end: parseSrtTime(tc[2]), text: t, words: [], flags: [] });
  }
  return segs;
}

function parseTxt(text) {
  const segs = [];
  for (const raw of text.replace(/\r/g, "").split("\n")) {
    const m = raw.trim().match(/^\[(.+?)\s*-->\s*(.+?)\]\s*(.*)$/);
    if (!m) continue;
    const { speaker, text: t } = splitSpeaker(m[3]);
    segs.push({ speaker, start: parseSrtTime(m[1]), end: parseSrtTime(m[2]), text: t, words: [], flags: [] });
  }
  return segs;
}

// Minimal RFC-4180 reader (handles quoted fields with commas/newlines/"" escapes).
function parseCsvRows(text) {
  const rows = [], s = text.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  let row = [], field = "", inQ = false;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (inQ) {
      if (c === '"') { if (s[i + 1] === '"') { field += '"'; i++; } else inQ = false; }
      else field += c;
    } else if (c === '"') inQ = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n") { row.push(field); rows.push(row); row = []; field = ""; }
    else field += c;
  }
  if (field !== "" || row.length) { row.push(field); rows.push(row); }
  return rows;
}

function parseCsv(text) {
  const rows = parseCsvRows(text).filter((r) => r.some((c) => c !== ""));
  if (!rows.length) return [];
  const head = rows[0].map((h) => h.trim().toLowerCase());
  const hasHeader = head[0] === "start" && head[1] === "end";
  const at = hasHeader
    ? { start: head.indexOf("start"), end: head.indexOf("end"), speaker: head.indexOf("speaker"),
        text: head.indexOf("text"), flags: head.indexOf("flags") }
    : { start: 0, end: 1, speaker: 2, text: 3, flags: 4 };
  return (hasHeader ? rows.slice(1) : rows).map((r) => ({
    speaker: (r[at.speaker] || "Speaker").trim(),
    start: parseFloat(r[at.start]) || 0,
    end: parseFloat(r[at.end]) || 0,
    text: r[at.text] || "",
    words: [],
    flags: (at.flags >= 0 ? (r[at.flags] || "") : "").split("|").map((f) => f.trim()).filter(Boolean),
  }));
}

// =======================================================================================
// Batch / Studio tab
// =======================================================================================
const projects = [];           // [{ name, files: [{file_id, source}] }]
let outputFolder = null;
let projectCounter = 0;

function addProject() {
  projectCounter += 1;
  projects.push({ name: `Project ${projectCounter}`, files: [] });
  renderProjects();
}

(function initBatchTab() {
  $("#add-project").onclick = addProject;
  $("#add-project-entry").onclick = addProject;
  $("#choose-folder").onclick = chooseFolder;
  $("#run-batch").onclick = runBatch;
})();

function renderProjects() {
  const has = projects.length > 0;                       // empty → show the entry hero, hide work panels
  $("#batch-entry").style.display = has ? "none" : "";
  $("#batch-projects-panel").style.display = has ? "" : "none";
  $("#batch-output-panel").style.display = has ? "" : "none";
  const root = $("#projects");
  root.innerHTML = "";
  projects.forEach((proj, pi) => root.append(buildProjectCard(proj, pi)));
}

// Global batch output defaults (read live from the Output panel checkboxes).
function globalFormats() { return [...document.querySelectorAll(".batch-fmt:checked")].map((c) => c.value); }
function globalPerFile() { return $("#batch-perfile").checked; }

// Per-project output settings: inherit the global defaults, or override for this project alone.
function buildProjectOutput(proj) {
  const ov = el("input", { type: "checkbox", checked: !!proj.override });
  ov.onchange = () => {
    proj.override = ov.checked;
    if (proj.override && !proj.formats) { proj.formats = globalFormats(); proj.perFile = globalPerFile(); }
    renderProjects();
  };
  const row = el("div", { className: "row", style: "margin-top:6px" },
    el("label", { className: "opt" }, ov, "Override output settings"));
  if (!proj.override) {
    row.append(el("span", { className: "muted", textContent: "using global defaults" }));
    return row;
  }
  const pf = el("input", { type: "checkbox", checked: !!proj.perFile });
  pf.onchange = () => { proj.perFile = pf.checked; };
  const fmtChecks = ["srt", "txt", "csv", "json"].map((f) =>
    el("input", { type: "checkbox", value: f, checked: (proj.formats || []).includes(f) }));
  fmtChecks.forEach((cb) => { cb.onchange = () => { proj.formats = fmtChecks.filter((c) => c.checked).map((c) => c.value); }; });
  row.append(
    el("label", { className: "opt" }, pf, "One transcript per file"),
    el("span", { className: "muted", textContent: "Formats:" }),
    ...fmtChecks.map((cb) => el("label", { className: "opt" }, cb, cb.value.toUpperCase())));
  return row;
}

function buildProjectCard(proj, pi) {
  const box = el("div", { className: "source" });
  const name = el("input", { value: proj.name, style: "font-weight:600; width:240px" });
  name.oninput = () => { proj.name = name.value; };
  const addFiles = el("button", { className: "ghost", textContent: "Add files / folder" });
  addFiles.onclick = () => addToProject(proj, addFiles);
  const removeProj = el("button", { className: "ghost icon-btn", title: "Remove project" },
    icon("close"), "Remove project");
  removeProj.onclick = () => { projects.splice(pi, 1); renderProjects(); };
  box.append(el("div", { className: "name" },
    name, el("div", { className: "spacer" }), addFiles, removeProj));
  box.append(buildProjectOutput(proj));

  proj.files.forEach((_, fi) => box.append(buildFileRow(proj, fi)));
  if (!proj.files.length) box.append(el("div", { className: "muted", textContent: "no files yet — click Add files" }));
  return box;
}

function buildFileRow(proj, fi) {
  const f = proj.files[fi];
  const wrap = el("div", { style: "border-top:1px solid var(--line); padding:6px 0; margin-top:6px" });
  const swap = (a, b) => { [proj.files[a], proj.files[b]] = [proj.files[b], proj.files[a]]; renderProjects(); };
  const up = el("button", { className: "ghost icon-only", title: "Move earlier" }, icon("up"));
  up.onclick = () => { if (fi > 0) swap(fi, fi - 1); };
  const down = el("button", { className: "ghost icon-only", title: "Move later" }, icon("down"));
  down.onclick = () => { if (fi < proj.files.length - 1) swap(fi, fi + 1); };
  const rm = el("button", { className: "ghost icon-only", title: "Remove file" }, icon("close"));
  rm.onclick = () => { proj.files.splice(fi, 1); renderProjects(); };

  wrap.append(el("div", { className: "track-row" },
    el("span", { className: "ti", textContent: `${fi + 1}.` }),
    el("span", { textContent: f.source.filename, style: "flex:1" }),
    el("span", { className: "dur muted", textContent: `${f.source.duration.toFixed(1)}s` }),
    up, down, rm));

  f.source.tracks.forEach((t) => {
    const spk = el("input", { value: t.speaker, size: 12 });
    spk.oninput = () => { t.speaker = spk.value; };
    if (f.source.tracks.length > 1) {
      const sel = el("input", { type: "checkbox", checked: t.selected !== false });
      sel.onchange = () => { t.selected = sel.checked; };
      wrap.append(el("div", { className: "track-row" },
        sel, el("span", { className: "ti", textContent: `track ${t.track_index + 1}` }), spk));
    } else {
      wrap.append(el("div", { className: "track-row" },
        el("span", { className: "ti", textContent: "speaker" }), spk));
    }
  });
  return wrap;
}

// The project add button offers Files or Folder — one native dialog each, since Windows
// can't pick both in one window. "Folder" pulls in every audio/video file underneath,
// recursively; per-file output then lands next to each source (nested tree preserved).
function addToProject(proj, btn) {
  const open = btn.parentElement.querySelector(".addmenu");
  if (open) { open.remove(); return; }                 // click again to dismiss
  const filesBtn = el("button", { className: "ghost", textContent: "Files…" });
  const folderBtn = el("button", { className: "ghost", textContent: "Folder…" });
  const menu = el("span", { className: "addmenu", style: "display:inline-flex; gap:6px; margin-left:6px" },
    filesBtn, folderBtn);
  filesBtn.onclick = () => { menu.remove(); pickFiles(proj); };
  folderBtn.onclick = () => { menu.remove(); pickFolder(proj); };
  btn.insertAdjacentElement("afterend", menu);
}

async function pickFiles(proj) {
  $("#batch-err").textContent = "";
  const res = await fetch("/api/browse", { method: "POST" });
  if (!res.ok) { $("#batch-err").textContent = "Picker failed: " + (await res.text()); return; }
  proj.files.push(...(await res.json()));
  renderProjects();
}

async function pickFolder(proj) {
  $("#batch-err").textContent = "Scanning folder for audio/video…";
  let res;
  try {
    res = await fetchTO("/api/add-folder", { method: "POST" }, 600000);
  } catch (e) { $("#batch-err").textContent = "Folder scan timed out."; return; }
  if (!res.ok) { $("#batch-err").textContent = "Folder scan failed: " + (await res.text()); return; }
  const found = await res.json();
  $("#batch-err").textContent = found.length ? "" : "No audio/video files found in that folder.";
  proj.files.push(...found);
  renderProjects();
}

async function chooseFolder() {
  const res = await fetch("/api/browse-folder", { method: "POST" });
  if (!res.ok) return;
  const { path } = await res.json();
  if (path) { outputFolder = path; $("#out-folder-path").textContent = path; }
}

function runBatch() {
  const err = $("#batch-err");
  err.textContent = "";
  const usable = projects.filter((p) => p.files.length);
  if (!usable.length) { err.textContent = "Add at least one project with files."; return; }
  const mode = document.querySelector("input[name=outmode]:checked").value;
  if (mode === "folder" && !outputFolder) { err.textContent = "Choose an output folder."; return; }

  // Resolve each project's output: its own settings if overriding, else the global defaults.
  const gFormats = globalFormats(), gPerFile = globalPerFile();
  const resolved = usable.map((p) => ({
    p,
    perFile: p.override ? !!p.perFile : gPerFile,
    formats: p.override ? (p.formats || []) : gFormats,
  }));
  if (resolved.some((r) => !r.formats.length)) { err.textContent = "Pick at least one output format."; return; }

  const body = {
    output_mode: mode,
    output_folder: outputFolder,
    options: { detect_conflicts: $("#batch-detect").checked },
    language: $("#batch-language").value || null,
    projects: resolved.map(({ p, perFile, formats }) => ({
      name: p.name,
      per_file: perFile,
      formats,
      sources: p.files.map((f, i) => ({
        file_id: f.file_id,
        sequence: i,
        tracks: f.source.tracks.map((t) => ({
          track_index: t.track_index, speaker: t.speaker, selected: t.selected !== false,
        })),
      })),
    })),
  };
  startBatch(body);
}

async function startBatch(body) {
  const btn = $("#run-batch");
  btn.disabled = true;
  let res;
  try {
    res = await fetchTO("/api/batch", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    }, 30000);
  } catch (e) {
    $("#batch-err").textContent = "couldn't reach the server to start the batch."; btn.disabled = false; return;
  }
  if (!res.ok) { $("#batch-err").textContent = "Batch failed: " + (await res.text()); btn.disabled = false; return; }
  const { batch_id } = await res.json();
  $("#batch-status-panel").style.display = "";
  pollBatch(batch_id, btn);
}

function pollBatch(id, btn) {
  const cancelBtn = $("#cancel-batch"), retryBtn = $("#retry-batch");
  cancelBtn.style.display = ""; cancelBtn.disabled = false; cancelBtn.textContent = "Cancel batch";
  retryBtn.style.display = "none";
  cancelBtn.onclick = async () => {
    cancelBtn.disabled = true; cancelBtn.textContent = "Canceling… (current file finishes)";
    try { await fetch(`/api/batch/${id}/cancel`, { method: "POST" }); } catch {}
  };
  const tick = async () => {
    let b;
    try {
      const r = await fetchTO(`/api/batch/${id}`);
      if (!r.ok) throw new Error(await r.text());
      b = await r.json();
    } catch (e) { setTimeout(tick, 1500); return; }   // transient — keep waiting, don't die
    renderBatchStatus(b, id, btn);
    if (b.status === "done" || b.status === "canceled") {
      btn.disabled = false;
      cancelBtn.style.display = "none";
      const failed = b.projects.filter((p) => p.status === "error" || p.status === "canceled").length;
      if (failed) {
        retryBtn.style.display = "";
        retryBtn.disabled = false;
        retryBtn.textContent = `Retry ${failed} failed/canceled`;
        retryBtn.onclick = async () => {
          retryBtn.disabled = true;
          try {
            const r = await fetchTO(`/api/batch/${id}/retry`,
              { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }, 30000);
            if (!r.ok) { $("#batch-err").textContent = "Retry failed: " + (await r.text()); retryBtn.disabled = false; return; }
            retryBtn.style.display = "none";
            pollBatch(id, btn);                        // resume polling the same batch
          } catch (e) { $("#batch-err").textContent = "Couldn't reach server for retry."; retryBtn.disabled = false; }
        };
      }
      return;
    }
    setTimeout(tick, 800);
  };
  tick();
}

// status word -> { display label, css class } for the per-item chip.
const BATCH_STATE = {
  queued:   { label: "Queued",   cls: "queued" },
  running:  { label: "Running",  cls: "running" },
  done:     { label: "Done",     cls: "done" },
  error:    { label: "Failed",   cls: "error" },
  canceled: { label: "Canceled", cls: "canceled" },
};

function renderBatchStatus(b, id, btn) {
  // Overall bar: weighted by each project's audio length × tracks, so a long project moves it more.
  const pct = Math.round((b.progress || 0) * 100);
  $("#batch-bar").style.width = pct + "%";
  $("#batch-pct").textContent = pct + "%";

  const counts = { queued: 0, running: 0, done: 0, error: 0, canceled: 0 };
  b.projects.forEach((p) => { counts[p.status] = (counts[p.status] || 0) + 1; });
  const total = b.projects.length;

  const root = $("#batch-status");
  root.innerHTML = "";

  // Plain-language summary so the numbers above aren't ambiguous.
  root.append(el("div", { className: "batch-summary" },
    el("span", { textContent: `${counts.done} of ${total} projects complete` }),
    el("span", { className: "muted",
      textContent: ` · overall ${pct}% (by audio length) · ${fmtElapsed(b.elapsed)}` })));
  const parts = [];
  for (const [k, label] of [["running", "running"], ["queued", "queued"], ["done", "done"],
                            ["error", "failed"], ["canceled", "canceled"]]) {
    if (counts[k]) parts.push(`${counts[k]} ${label}`);
  }
  root.append(el("div", { className: "batch-counts muted", textContent: parts.join("  ·  ") }));

  const ended = b.status === "done" || b.status === "canceled";
  b.projects.forEach((p, i) => root.append(buildBatchItem(p, i, id, btn, ended)));
}

// One project row: name + status chip + (cancel | restart), its own progress bar + detail.
function buildBatchItem(p, i, id, btn, ended) {
  const meta = BATCH_STATE[p.status] || { label: p.status, cls: "queued" };
  const pct = Math.round((p.progress || 0) * 100);
  const card = el("div", { className: `batch-item state-${meta.cls}` });

  const head = el("div", { className: "batch-item-head" },
    el("span", { className: "bi-name", textContent: p.name }),
    el("span", { className: `chip-status ${meta.cls}`, textContent: meta.label }),
    el("div", { className: "spacer" }),
    el("span", { className: "muted bi-elapsed", textContent: fmtElapsed(p.elapsed) }));

  // Active item (running/queued): cancel just this one — the queue continues. Finished item
  // (done/failed/canceled): restart it. Restart works mid-batch too — if the card is busy the
  // item is queued to re-run when it frees up; otherwise a fresh run starts.
  if (p.status === "running" || p.status === "queued") {
    const cancel = el("button", { className: "ghost icon-only bi-act", title: "Cancel this item" }, icon("close"));
    cancel.onclick = async () => {
      cancel.disabled = true; cancel.title = "Canceling…";
      try { await fetch(`/api/batch/${id}/project/${i}/cancel`, { method: "POST" }); } catch {}
    };
    head.append(cancel);
  } else {
    const restart = el("button", { className: "ghost bi-act", textContent: "Restart",
      title: ended ? "Run this item again" : "Run this item again (queues until the batch frees up)" });
    restart.onclick = async () => {
      restart.disabled = true; restart.textContent = "Restarting…";
      try {
        const r = await fetchTO(`/api/batch/${id}/retry`,
          { method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ projects: [i] }) }, 30000);
        if (!r.ok) {
          $("#batch-err").textContent = "Restart failed: " + (await r.text());
          restart.disabled = false; restart.textContent = "Restart"; return;
        }
        if (ended) pollBatch(id, btn);                    // polling had stopped — resume it
        // else: the live poller will show the item re-queued
      } catch (e) {
        $("#batch-err").textContent = "Couldn't reach server to restart.";
        restart.disabled = false; restart.textContent = "Restart";
      }
    };
    head.append(restart);
  }
  card.append(head);

  // Per-item bar — only once there's something to show (skip idle queued rows).
  if (p.status === "running" || (pct > 0 && pct < 100)) {
    card.append(el("div", { className: "bar bi-bar" }, el("div", { style: `width:${pct}%` })));
  }

  // Detail line: error / output location / live message.
  let detail;
  if (p.error) {
    detail = el("div", { className: "bi-msg bi-err" }, icon("warn", 13), " " + p.error);
  } else if (p.status === "done") {
    const txt = `${p.message || "done"}${p.output_dir ? " · " + p.output_dir : ""}`;
    detail = el("div", { className: "bi-msg muted", textContent: txt });
  } else if (p.status === "running") {
    detail = el("div", { className: "bi-msg muted", textContent: `${pct}% · ${p.message || ""}` });
  } else {
    detail = el("div", { className: "bi-msg muted", textContent: p.message || meta.label });
  }
  card.append(detail);

  // Per-file projects: a row per source file, each with its own status + cancel/restart.
  if (p.files && p.files.length) {
    const files = el("div", { className: "batch-files" });
    p.files.forEach((f, fi) => files.append(buildBatchFile(f, i, fi, id, btn, ended)));
    card.append(files);
  }
  return card;
}

// One file row inside a per-file project: name + status chip + cancel/restart + its own bar.
function buildBatchFile(f, pidx, fidx, id, btn, ended) {
  const meta = BATCH_STATE[f.status] || { label: f.status, cls: "queued" };
  const pct = Math.round((f.progress || 0) * 100);
  const row = el("div", { className: `batch-file state-${meta.cls}` });

  const detail = f.error ? f.error
    : (f.status === "done" ? (f.message || "done")
       : (f.status === "running" ? `${pct}% · ${f.message || ""}` : (f.message || meta.label)));
  const msg = el("span", { className: "muted bf-msg", textContent: detail, title: detail });
  if (f.error) msg.style.color = "var(--bleed)";

  const head = el("div", { className: "bf-head" },
    el("span", { className: "bf-name", textContent: f.name, title: f.name }),
    el("span", { className: `chip-status ${meta.cls}`, textContent: meta.label }),
    el("div", { className: "spacer" }),
    msg);

  if (f.status === "running" || f.status === "queued") {
    const cancel = el("button", { className: "ghost icon-only bi-act", title: "Cancel this file" }, icon("close"));
    cancel.onclick = async () => {
      cancel.disabled = true; cancel.title = "Canceling…";
      try { await fetch(`/api/batch/${id}/project/${pidx}/file/${fidx}/cancel`, { method: "POST" }); } catch {}
    };
    head.append(cancel);
  } else {
    const restart = el("button", { className: "ghost bi-act", textContent: "Restart",
      title: ended ? "Run this file again" : "Run this file again (queues until the card frees up)" });
    restart.onclick = async () => {
      restart.disabled = true; restart.textContent = "…";
      try {
        const r = await fetchTO(`/api/batch/${id}/project/${pidx}/file/${fidx}/restart`, { method: "POST" }, 30000);
        if (!r.ok) {
          $("#batch-err").textContent = "Restart failed: " + (await r.text());
          restart.disabled = false; restart.textContent = "Restart"; return;
        }
        if (ended) pollBatch(id, btn);                    // polling had stopped — resume it
      } catch (e) {
        $("#batch-err").textContent = "Couldn't reach server to restart.";
        restart.disabled = false; restart.textContent = "Restart";
      }
    };
    head.append(restart);
  }
  row.append(head);
  if (f.status === "running" || (pct > 0 && pct < 100)) {
    row.append(el("div", { className: "bar bf-bar" }, el("div", { style: `width:${pct}%` })));
  }
  return row;
}
