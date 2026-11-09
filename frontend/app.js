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
  if (r.name === "AbortError") return;   // play() cut short by pause()/new clip: harmless
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

// Hours-aware duration, for audio totals that can run to many hours (track-seconds across a batch).
function fmtAudio(s) {
  s = Math.round(s || 0);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  if (h) return `${h}h ${String(m).padStart(2, "0")}m`;
  if (m) return `${m}m ${String(s % 60).padStart(2, "0")}s`;
  return `${s}s`;
}

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

// --- clear cache -----------------------------------------------------------------------
// Wipes the WhisperX-output cache so the next run transcribes cold (e.g. to benchmark real
// GPU time). Confirm first — afterwards re-runs of the same files are no longer instant.
// The button shows what's cached ("42 · 1.2 GB") and greys out when there's nothing to clear.
async function refreshCacheStats() {
  const btn = $("#clear-cache-btn"), stat = $("#cache-stat");
  try {
    const s = await (await fetch("/api/cache/stats")).json();
    const size = s.bytes >= 1e9 ? `${(s.bytes / 1e9).toFixed(1)} GB` : `${Math.max(1, Math.round(s.bytes / 1e6))} MB`;
    stat.textContent = s.entries ? `${s.entries} · ${size}` : "empty";
    btn.disabled = !s.entries;
    btn.title = s.entries ? `${s.entries} cached transcription${s.entries === 1 ? "" : "s"} (${size}). `
      + "Delete them so the next run transcribes cold (for benchmarking)." : "Nothing cached";
  } catch { stat.textContent = ""; btn.disabled = false; }
}
refreshCacheStats();

$("#clear-cache-btn").onclick = async (e) => {
  const btn = e.currentTarget;
  if (!confirm("Delete all cached transcriptions? The next run will transcribe from scratch.")) return;
  const stat = $("#cache-stat");
  btn.disabled = true; stat.textContent = "clearing…";
  try {
    const r = await fetch("/api/cache/clear", { method: "POST" });
    if (!r.ok) throw new Error();
  } catch (_) {
    stat.textContent = "failed";
    setTimeout(refreshCacheStats, 2000);
    return;
  }
  refreshCacheStats();
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

// Line -> audio-stream index, stamped once when a result loads. Playback follows the line, not
// the speaker name, so Rename globally or a per-line speaker edit keeps playing the right mic.
const segTrack = new WeakMap();

function renderEditor(container, merged, preview) {
  if (preview !== undefined) container._preview = preview;  // persist across re-renders
  preview = container._preview || null;
  if (preview?.trackOf) merged.segments.forEach((s) => {
    if (!segTrack.has(s)) segTrack.set(s, preview.trackOf[s.speaker]);
  });
  // Per-speaker export toggles, keyed to the loaded transcript: a new `merged` resets them,
  // in-place re-renders (same object) keep them.
  if (container._merged !== merged) { container._merged = merged; container._disabled = new Set(); }
  container.innerHTML = "";
  container.append(buildToolbar(container, merged), buildSpeakerBar(container, merged),
    buildExportPanel(container, merged));
  if (preview) { container.append(buildPreviewBar(preview)); container.append(buildWaveform(preview)); }
  container.append(buildTable(container, merged, preview));
}

// Per-line ▶ on a multi-track file: false = that speaker's track only, true = all tracks mixed.
// Kept for the browser session.
let _lineAll = null;
function lineAllTracks() {
  if (_lineAll === null) {
    try { _lineAll = sessionStorage.getItem("wm-line-all") === "all"; } catch { _lineAll = false; }
  }
  return _lineAll;
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

  // Multi-track source: a line's ▶ can play just that speaker's mic, or every track mixed.
  const tracks = new Set(Object.values(preview.trackOf || {}).filter((t) => t != null));
  let lineMode = "";
  if (tracks.size > 1) {
    lineMode = el("select", { title: "What a line's ▶ plays" },
      el("option", { value: "", textContent: "▶ plays this speaker" }),
      el("option", { value: "all", textContent: "▶ plays all tracks" }));
    lineMode.value = lineAllTracks() ? "all" : "";
    lineMode.onchange = () => {
      try { sessionStorage.setItem("wm-line-all", lineMode.value); } catch {}
      _lineAll = lineMode.value === "all";
    };
  }

  return el("div", {},
    el("div", { className: "row", style: "margin:10px 0 6px" },
      el("span", { className: "muted", textContent: "Waveform" }), status,
      el("div", { className: "spacer" }), lineMode,
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
  // Speaker column fits the longest name (CSS caps it; longer names show in full on hover).
  const longest = merged.segments.reduce((n, s) => Math.max(n, (s.speaker || "").length), 0);
  const table = el("table", { className: "editor" + (hasNotes ? " has-notes" : ""), style: `--spk-ch:${longest}` },
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

  const spk = el("input", { value: seg.speaker, title: seg.speaker });
  spk.oninput = () => { seg.speaker = spk.title = spk.value; };

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
    // play only this line's stream when we know the mapping (multi-track source); else downmix
    const track = segTrack.get(seg);
    jump.onclick = () => toggleClip(jump, preview.file_id, seg.start, seg.end - seg.start,
      lineAllTracks() ? null : track);
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

// Speaker names that eat more than half a subtitle line ("Name: " counts toward Chars/line).
// Subtitlers shorten these by hand (Rename globally / the track's speaker name), so we flag them.
function longSpeakerNames(names, x) {
  if (!x.show_speakers || !x.apply_limits) return [];
  return [...new Set(names)].filter((n) => n && n.length + 2 > x.chars_per_line / 2);
}

const EDITED_TIMING = [["align", "keep word timing"], ["even", "spread evenly"], ["whole", "one subtitle per line"]];

function buildExportPanel(container, merged) {
  // One format downloads as that file; several as one zip.
  const fmtBoxes = ["srt", "txt", "csv", "json"].map((f) => el("input", { type: "checkbox", value: f, checked: f === "srt" }));
  const formats = () => fmtBoxes.filter((c) => c.checked).map((c) => c.value);
  const filename = el("input", { value: transcriptName(exportBase(merged)), size: 36 });
  const applyLimits = el("input", { type: "checkbox", checked: true });
  const cpl = el("input", { type: "number", value: 42 });
  const maxLines = el("input", { type: "number", value: 2 });
  const minDur = el("input", { type: "number", value: 1, step: 0.5 });
  const pad = el("input", { type: "number", value: 0, step: 0.1, min: 0, title: "extend each segment start/end by N seconds (video-edit breathing room)" });
  const showSpk = el("input", { type: "checkbox", checked: true });
  const onChange = el("input", { type: "checkbox", title: "SRT: name the speaker on the first subtitle after a speaker change only" });
  const warn = el("div", { className: "warn", style: "flex-basis:100%" });
  const checkNames = () => {
    const names = merged.segments.filter((s) => !container._disabled.has(s.speaker)).map((s) => s.speaker);
    const long = longSpeakerNames(names, { show_speakers: showSpk.checked, apply_limits: applyLimits.checked, chars_per_line: +cpl.value });
    warn.textContent = long.length ? `Long speaker name${long.length > 1 ? "s" : ""} for ${cpl.value} characters per line: ${long.map((n) => `"${n}" (${n.length + 2})`).join(", ")}. Consider Rename globally to shorten.` : "";
  };
  // How SRT times a line whose text you changed (its word timings no longer match the words).
  const editedTiming = el("select", { title: "SRT timing for lines whose text you edited", style: "min-width:180px" });
  EDITED_TIMING.forEach(([v, t]) => editedTiming.append(el("option", { value: v, textContent: t })));

  // Subtitle limits only apply to SRT; toggling off disables (and ignores) the limit fields.
  const syncLimits = () => [cpl, maxLines, minDur].forEach((n) => { n.disabled = !applyLimits.checked; });
  applyLimits.onchange = () => { syncLimits(); checkNames(); };
  syncLimits();
  cpl.oninput = showSpk.onchange = checkNames;
  checkNames();

  const btn = el("button", { className: "primary", textContent: "Export" });
  btn.onclick = async () => {
    const options = {
      apply_limits: applyLimits.checked,
      chars_per_line: +cpl.value, max_lines: +maxLines.value,
      min_duration: +minDur.value, show_speakers: showSpk.checked,
      pad_seconds: +pad.value, edited_timing: editedTiming.value, label_on_change: onChange.checked,
    };
    const fmts = formats();
    if (!fmts.length) { alert("Tick at least one format."); return; }
    // Drop toggled-off speakers (all formats) — the backend just renders what we send.
    const segs = merged.segments.filter((s) => !container._disabled.has(s.speaker));
    if (!segs.length) { alert("All speakers are toggled off — nothing to export."); return; }
    const hidden = distinctSpeakers(merged).filter((s) => container._disabled.has(s));
    const edited = segs.filter((s) => (s.flags || []).includes("edited")).length;
    if (!(await confirmExport({ fmts, options, hidden, edited, lines: segs.length }))) return;
    await downloadExport({ ...merged, segments: segs }, fmts.join(","), filename.value || "transcript", options);
  };

  const field = (label, node) => el("label", { className: "field" }, el("span", { textContent: label }), node);
  return el("div", { className: "row", style: "margin:14px 0; padding-top:14px; border-top:1px solid var(--line)" },
    field("Formats", el("div", { className: "row fmt-set" },
      ...fmtBoxes.map((c) => el("label", { className: "opt" }, c, c.value.toUpperCase())))),
    field("Filename", filename),
    el("label", { className: "opt" }, applyLimits, "Subtitle limits"),
    field("Chars/line", cpl), field("Max lines", maxLines), field("Min dur (s)", minDur),
    field("Pad (s)", pad), field("Edited lines", editedTiming),
    el("label", { className: "opt" }, showSpk, "Show speakers"),
    el("label", { className: "opt" }, onChange, "Name only on speaker change"),
    el("div", { className: "spacer" }), btn, warn);
}

// What will shape the export, shown before writing it. "Don't show again" is remembered in this
// browser only.
function confirmExport({ fmts, options: o, hidden, edited, lines }) {
  try { if (localStorage.getItem("wm-skip-export-summary") === "1") return Promise.resolve(true); } catch {}
  const srt = fmts.includes("srt");
  const rows = [
    ["Formats", fmts.map((f) => f.toUpperCase()).join(", ") + (fmts.length > 1 ? " (one zip)" : "")],
    ["Lines", String(lines)],
    ["Subtitle limits", !srt ? "SRT only (not exported)" : o.apply_limits
      ? `on · ${o.chars_per_line} chars/line · ${o.max_lines} lines max · ${o.min_duration} s min` : "off"],
    ["Padding", o.pad_seconds ? `${o.pad_seconds} s added to each line's start and end` : "none"],
    ["Edited lines (SRT)", !srt ? "SRT only (not exported)" : EDITED_TIMING.find(([v]) => v === o.edited_timing)[1]],
    ["Speaker names", !o.show_speakers ? "off" : srt && o.label_on_change ? "on, in SRT only when the speaker changes" : "on every line"],
    ["Hidden speakers", hidden.length ? `${hidden.join(", ")} (left out)` : "none"],
    ["Lines marked edited", String(edited)],
  ];
  const skip = el("input", { type: "checkbox" });
  const dlg = el("dialog", { className: "export-summary" },
    el("h2", { textContent: "Export" }),
    el("dl", {}, ...rows.flatMap(([k, v]) => [el("dt", { textContent: k }), el("dd", { textContent: v })])),
    el("form", { method: "dialog", className: "row" },
      el("label", { className: "opt" }, skip, "Don't show again"), el("div", { className: "spacer" }),
      el("button", { className: "ghost", value: "back", textContent: "Back" }),
      el("button", { className: "primary", value: "export", textContent: "Export" })));
  document.body.append(dlg);
  return new Promise((resolve) => {
    dlg.onclose = () => {
      const go = dlg.returnValue === "export";
      if (go && skip.checked) { try { localStorage.setItem("wm-skip-export-summary", "1"); } catch {} }
      dlg.remove();
      resolve(go);
    };
    dlg.showModal();
  });
}

// `format` is one format or several, comma-separated (the server zips several).
async function downloadExport(merged, format, filename, options) {
  const res = await fetch(`/api/export?format=${format}`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ merged, options, filename }),
  });
  if (!res.ok) { alert("Export failed: " + (await res.text())); return; }
  const blob = await res.blob();
  const a = el("a", { href: URL.createObjectURL(blob), download: `${filename}.${format.includes(",") ? "zip" : format}` });
  document.body.append(a); a.click(); a.remove();
}

// =======================================================================================
// Transcribe & Merge tab
// =======================================================================================
const uploaded = [];  // [{file_id, source}]

// Declared before initTranscribeTab() runs — it calls syncDetect() immediately at load.
const MERGE_THRESHOLD_IDS = ["opt-backchannel", "opt-bc-words", "opt-overlap", "opt-bleed-win", "opt-bleed-sim", "opt-bleed-ratio"];

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

// A track's speaker name box. Before a run it just sets the name; after a run, changing it also
// renames that speaker in the transcript below (like Rename globally).
function speakerInput(t) {
  const spk = el("input", { value: t.speaker });
  spk.oninput = () => { t.speaker = spk.value; };
  spk.onfocus = () => { spk._was = t.speaker; };
  spk.onchange = () => renameSpeakerInResult(spk._was, t.speaker.trim());
  return spk;
}

// Rename a speaker in the finished Transcribe & Merge result: every line, the Include toggles and
// the speaker dropdown (via re-render), merged.sources (exports), and the speaker -> track map that
// per-line playback stamps new lines from. Lines already stamped keep their track (segTrack).
function renameSpeakerInResult(oldName, newName) {
  const box = $("#editor-transcribe"), merged = box._merged;
  if (!merged || !newName || oldName === newName) return;
  let hit = false;
  merged.segments.forEach((s) => { if (s.speaker === oldName) { s.speaker = newName; hit = true; } });
  if (!hit) return;
  (merged.sources || []).forEach((src) => (src.tracks || []).forEach((t) => {
    if (t.speaker === oldName) t.speaker = newName;
  }));
  if (box._disabled.delete(oldName)) box._disabled.add(newName);
  const trackOf = box._preview && box._preview.trackOf;
  if (trackOf && oldName in trackOf) { trackOf[newName] = trackOf[oldName]; delete trackOf[oldName]; }
  renderEditor(box, merged);
}

function renderSources() {
  const root = $("#sources");
  root.innerHTML = "";
  uploaded.forEach((it, idx) => {
    const box = el("div", { className: "source" });
    const rm = el("button", { className: "ghost icon-only", title: "Remove file", style: "margin-left:auto" }, icon("close"));
    rm.onclick = () => {
      uploaded.splice(idx, 1);
      if (!uploaded.length) {                          // back to the empty start screen
        $("#transcribe-entry").style.display = "";
        $("#input-panel").style.display = "none";
        $("#options-panel").style.display = "none";
      }
      renderSources();
    };
    box.append(el("div", { className: "name" },
      el("span", { textContent: it.source.filename }),
      el("span", { className: "dur", textContent: `${it.source.duration.toFixed(1)}s` }), rm));

    // FR-TK-4: track selector only when more than one stream.
    if (!it.source.tracks.length) {
      box.append(el("div", { className: "err", textContent: "No audio tracks found in this file — it can't be transcribed." }));
    } else if (it.source.tracks.length > 1) {
      it.source.tracks.forEach((t) => {
        const sel = el("input", { type: "checkbox", checked: t.selected });
        sel.onchange = () => { t.selected = sel.checked; };
        box.append(el("div", { className: "track-row" },
          sel, el("span", { className: "ti", textContent: `track ${t.track_index + 1}` }),
          speakerInput(t), auditionBtn(it.file_id, t.track_index)));
      });
    } else {
      const t = it.source.tracks[0];
      box.append(el("div", { className: "track-row" },
        el("span", { className: "ti", textContent: "speaker" }),
        speakerInput(t), auditionBtn(it.file_id, t.track_index)));
    }

    // sequence order (UC2) — only shown with multiple files
    if (uploaded.length > 1) {
      if (!it.orderSet) it.source.sequence = idx;   // auto-number until the user types an order
      const seq = el("input", { type: "number", value: it.source.sequence, style: "width:60px" });
      seq.onchange = () => { it.source.sequence = +seq.value; it.orderSet = true; };
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
    bleed_prefer_ratio: +$("#opt-bleed-ratio").value,
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
  $("#progress-wrap").classList.remove("finished");
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
  const pauseBtn = $("#pause-job");
  let paused = false;                                  // last-known server pause state
  pauseBtn.style.display = ""; pauseBtn.disabled = false; pauseBtn.textContent = "Pause";
  pauseBtn.onclick = async () => {
    pauseBtn.disabled = true;
    const action = paused ? "resume" : "pause";        // toggle against the known state
    pauseBtn.textContent = paused ? "Resuming…" : "Pausing…";
    try { await fetch(`/api/${action}/${jobId}`, { method: "POST" }); } catch {}
  };
  const finish = () => {
    runBtn.disabled = false;
    cancelBtn.style.display = "none"; cancelBtn.textContent = "Cancel";
    pauseBtn.style.display = "none"; pauseBtn.textContent = "Pause"; pauseBtn.disabled = false;
  };
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
    paused = !!p.paused;                               // sync the toggle to the server
    pauseBtn.disabled = false;
    pauseBtn.textContent = p.paused ? "Resume" : "Pause";
    // Two-stage feedback: pause is requested but the current stage hasn't finished (GPU still busy)
    // vs fully parked (GPU released). The backend sends "paused · GPU released" once freed.
    const msg = (p.paused && !p.gpu_freed) ? "pausing… (finishing current stage)" : (p.message || p.status);
    setProgress(p.progress, `${msg}${p.elapsed ? " · " + fmtElapsed(p.elapsed) : ""}`);
    if (p.status === "done") {
      refreshCacheStats();
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
        // Collapse to one line: "Done in 4m 12s · 2 tracks".
        const tracks = uploaded.reduce((n, it) => n + it.source.tracks.filter((t) => t.selected !== false).length, 0);
        setProgress(1, `Done in ${fmtElapsed(p.elapsed)} · ${tracks} track${tracks === 1 ? "" : "s"}`);
        $("#progress-wrap").classList.add("finished");
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
const projects = [];           // [{ name, files: [{file_id, source}], settings?: custom merge/subtitle settings }]
let outputFolder = null;
let projectCounter = 0;

// Merge + subtitle settings, same fields and defaults as Transcribe & Merge and its Export panel.
// Batch-wide in batchSettings; a project can carry its own copy (proj.settings) that overrides it.
const DEFAULT_BATCH_SETTINGS = {
  options: { detect_conflicts: true, backchannel_render: "parenthetical", backchannel_max_words: 4,
             overlap_min_seconds: 2, bleed_window_seconds: 1, bleed_similarity: 0.7, bleed_prefer_ratio: 2 },
  export: { apply_limits: true, chars_per_line: 42, max_lines: 2, min_duration: 1, pad_seconds: 0,
            show_speakers: true, edited_timing: "align", label_on_change: false },
};
const batchSettings = structuredClone(DEFAULT_BATCH_SETTINGS);

// Editable fields bound to a settings object `s` (writes go straight into it, so re-renders keep them).
function buildSettingsBlock(s) {
  const field = (label, node, title) => el("label", { className: "field", title: title || "" }, el("span", { textContent: label }), node);
  const num = (obj, key, attrs = {}) => {
    const i = el("input", { type: "number", value: obj[key], style: "width:80px", ...attrs });
    i.onchange = () => { obj[key] = +i.value; };
    return i;
  };
  const check = (obj, key, label) => {
    const c = el("input", { type: "checkbox", checked: obj[key] });
    c.onchange = () => { obj[key] = c.checked; sync(); };
    return el("label", { className: "opt" }, c, label);
  };
  const o = s.options, x = s.export;
  const bc = el("select", {});
  ["parenthetical", "separate", "skip"].forEach((v) => bc.append(el("option", { value: v, textContent: v === "separate" ? "separate line" : v })));
  bc.value = o.backchannel_render;
  bc.onchange = () => { o.backchannel_render = bc.value; };
  const thresholds = [bc, num(o, "backchannel_max_words"), num(o, "overlap_min_seconds", { step: 0.5 }),
    num(o, "bleed_window_seconds", { step: 0.5 }), num(o, "bleed_similarity", { step: 0.05, min: 0, max: 1 }),
    num(o, "bleed_prefer_ratio", { step: 0.5, min: 1 })];
  const limits = [num(x, "chars_per_line"), num(x, "max_lines"), num(x, "min_duration", { step: 0.5 })];
  // Same choice as the editor's Export panel; in batch output it applies to "(yeah)"-style backchannels.
  const edited = el("select", { style: "min-width:180px", title: "SRT timing for lines whose text no longer matches its word timings" });
  [["align", "keep word timing"], ["even", "spread evenly"], ["whole", "one subtitle per line"]]
    .forEach(([v, t]) => edited.append(el("option", { value: v, textContent: t })));
  edited.value = x.edited_timing;
  edited.onchange = () => { x.edited_timing = edited.value; };
  function sync() {                       // thresholds only matter with detection on; limits only with limits on
    thresholds.forEach((n) => { n.disabled = !o.detect_conflicts; });
    limits.forEach((n) => { n.disabled = !x.apply_limits; });
  }
  sync();
  return el("div", {},
    el("div", { className: "row" }, check(o, "detect_conflicts", "Detect conflicts (backchannel / overlap / bleed)")),
    el("div", { className: "row", style: "margin-top:4px" },
      field("Backchannel", thresholds[0]), field("Backchannel max words", thresholds[1]),
      field("Overlap min (s)", thresholds[2]), field("Bleed window (s)", thresholds[3]),
      field("Bleed similarity", thresholds[4]),
      field("Bleed loudness ratio", thresholds[5], "On a suspected bleed, mark the louder mic's line 'preferred' only if it is at least this many times louder.")),
    el("div", { className: "row", style: "margin-top:8px" },
      check(x, "apply_limits", "Subtitle limits"), field("Chars/line", limits[0]), field("Max lines", limits[1]),
      field("Min dur (s)", limits[2]),
      field("Pad (s)", num(x, "pad_seconds", { step: 0.1, min: 0 }), "extend each segment start/end by N seconds (video-edit breathing room)"),
      field("Edited lines", edited), check(x, "show_speakers", "Show speakers"),
      check(x, "label_on_change", "Name only on speaker change")));
}

function addProject() {
  projectCounter += 1;
  projects.push({ name: `Project ${projectCounter}`, files: [] });
  renderProjects();
}

(function initBatchTab() {
  $("#batch-settings").append(el("details", {},
    el("summary", { className: "muted", textContent: "Merge & subtitle settings — apply to every project unless a project uses its own" }),
    buildSettingsBlock(batchSettings)));
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

// Per-project merge/subtitle settings: the batch's, or this project's own copy (starts as the batch's).
function buildProjectSettings(proj) {
  const own = el("input", { type: "checkbox", checked: !!proj.settings });
  own.onchange = () => { proj.settings = own.checked ? structuredClone(batchSettings) : null; renderProjects(); };
  const row = el("div", { className: "row", style: "margin-top:6px" },
    el("label", { className: "opt" }, own, "Own merge & subtitle settings"));
  if (!proj.settings) {
    row.append(el("span", { className: "muted", textContent: "using global defaults" }));
    return row;
  }
  return el("div", {}, row, el("div", { style: "margin:6px 0 0 22px" }, buildSettingsBlock(proj.settings)));
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
  box.append(buildProjectOutput(proj), buildProjectSettings(proj));

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

  // Long speaker names: warn once, and run only on a second click ("Run anyway"), so it isn't missed.
  const long = resolved.flatMap(({ p }) => longSpeakerNames(
    p.files.flatMap((f) => f.source.tracks.filter((t) => t.selected !== false).map((t) => t.speaker)),
    (p.settings || batchSettings).export).map((n) => `"${n}" (${p.name})`));
  const warning = long.length ? `Long speaker name${long.length > 1 ? "s" : ""} for the subtitle line limit: ${long.join(", ")}. Shorten in the project, or click Run anyway.` : "";
  if (warning && runBatch.acknowledged !== warning) {
    $("#batch-warn").textContent = warning;
    $("#run-batch").textContent = "Run anyway";
    runBatch.acknowledged = warning;
    return;
  }
  $("#batch-warn").textContent = "";
  $("#run-batch").textContent = "Run batch";
  runBatch.acknowledged = null;

  const body = {
    output_mode: mode,
    output_folder: outputFolder,
    options: batchSettings.options,
    export: batchSettings.export,
    language: $("#batch-language").value || null,
    projects: resolved.map(({ p, perFile, formats }) => ({
      name: p.name,
      per_file: perFile,
      formats,
      ...(p.settings ? { options: p.settings.options, export: p.settings.export } : {}),
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
  const cancelBtn = $("#cancel-batch"), retryBtn = $("#retry-batch"), pauseBtn = $("#pause-batch");
  cancelBtn.style.display = ""; cancelBtn.disabled = false; cancelBtn.textContent = "Cancel batch";
  retryBtn.style.display = "none";
  cancelBtn.onclick = async () => {
    cancelBtn.disabled = true; cancelBtn.textContent = "Canceling… (current file finishes)";
    try { await fetch(`/api/batch/${id}/cancel`, { method: "POST" }); } catch {}
  };
  let paused = false;                                  // last-known server pause state
  pauseBtn.style.display = ""; pauseBtn.disabled = false; pauseBtn.textContent = "Pause";
  pauseBtn.onclick = async () => {
    pauseBtn.disabled = true;
    const action = paused ? "resume" : "pause";        // toggle against the known state
    pauseBtn.textContent = paused ? "Resuming…" : "Pausing… (current file finishes)";
    try { await fetch(`/api/batch/${id}/${action}`, { method: "POST" }); } catch {}
  };
  const tick = async () => {
    let b;
    try {
      const r = await fetchTO(`/api/batch/${id}`);
      if (!r.ok) throw new Error(await r.text());
      b = await r.json();
    } catch (e) { setTimeout(tick, 1500); return; }   // transient — keep waiting, don't die
    renderBatchStatus(b, id, btn);
    paused = !!b.paused;                                // sync the toggle to the server
    pauseBtn.disabled = false;
    pauseBtn.textContent = b.paused ? "Resume" : "Pause";
    if (b.status === "done" || b.status === "canceled") {
      refreshCacheStats();
      btn.disabled = false;
      cancelBtn.style.display = "none";
      pauseBtn.style.display = "none"; pauseBtn.textContent = "Pause"; pauseBtn.disabled = false;
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
  const ended = b.status === "done" || b.status === "canceled";

  // Audio totals make the % concrete: it's measured by audio length × tracks, so show how much of
  // that total has been transcribed vs the whole (e.g. "2h 44m / 7h 24m audio").
  const totalWeight = b.projects.reduce((a, p) => a + (p.weight || 0), 0);
  const doneWeight = b.projects.reduce((a, p) => a + (p.weight || 0) * (p.progress || 0), 0);
  const audioText = totalWeight > 0
    ? ` · ${fmtAudio(doneWeight)} / ${fmtAudio(totalWeight)} audio` : "";

  const pauseText = (b.paused && !ended)
    ? (b.gpu_freed ? "  ·  ⏸ Paused · GPU released" : "  ·  ⏸ Pausing… (current file finishes)")
    : "";
  const summaryText = `${counts.done} of ${total} projects complete${pauseText}`;
  const overallText = ` · overall ${pct}%${audioText} · ${fmtElapsed(b.elapsed)} elapsed`;
  const parts = [];
  for (const [k, label] of [["running", "running"], ["queued", "queued"], ["done", "done"],
                            ["error", "failed"], ["canceled", "canceled"]]) {
    if (counts[k]) parts.push(`${counts[k]} ${label}`);
  }
  const countsText = parts.join("  ·  ");

  // Per-file extract-vs-transcribe tally: real read + ffmpeg time (which overlaps the prior file's
  // GPU run) vs actual GPU transcription — the read-bound vs GPU-bound answer, live.
  let tRead = 0, tGpu = 0;
  b.projects.forEach((p) => (p.files || []).forEach((f) => {
    if (f.extract_seconds > 0 || f.status === "done" || f.status === "running") {
      tRead += f.extract_seconds || 0;
      tGpu += f.elapsed || 0;            // elapsed is the GPU clock; extraction is tracked separately
    }
  }));
  const timingText = (tRead + tGpu) > 0
    ? `${fmtAudio(tRead)} reading/extracting · ${fmtAudio(tGpu)} on GPU so far`
    : "";

  // Structure signature — only a change here (a different batch, a changed project/file count, or
  // the batch ending) triggers a full rebuild. Status / % / message changes are applied in place,
  // so the scrollable file list is never torn out from under an active scroll (rebuilding it every
  // poll killed trackpad momentum and dropped wheel events — that was the "freeze").
  const root = $("#batch-status");
  const sig = `${b.id}|${ended ? 1 : 0}|` +
    b.projects.map((p) => (p.files ? p.files.length : 0)).join(",");

  if (root.dataset.sig === sig) {                          // steady state — surgical in-place update
    root.querySelector(".batch-summary > :first-child").textContent = summaryText;
    root.querySelector(".batch-summary > .muted").textContent = overallText;
    root.querySelector(".batch-counts").textContent = countsText;
    root.querySelector(".batch-timing").textContent = timingText;
    b.projects.forEach((p, i) =>
      updateBatchItem(root.querySelector(`.batch-item[data-pidx="${i}"]`), p, i, id, btn, ended));
    return;
  }

  // Structure changed (or first render) — full rebuild, preserving any file-list scroll positions.
  const scrolls = {};
  root.querySelectorAll(".batch-files").forEach((f) => {
    if (f.dataset.pidx != null) scrolls[f.dataset.pidx] = f.scrollTop;
  });
  root.dataset.sig = sig;
  root.innerHTML = "";
  root.append(el("div", { className: "batch-summary" },
    el("span", { textContent: summaryText }),
    el("span", { className: "muted", textContent: overallText })));
  root.append(el("div", { className: "batch-counts muted", textContent: countsText }));
  root.append(el("div", { className: "batch-timing muted", textContent: timingText }));
  b.projects.forEach((p, i) => root.append(buildBatchItem(p, i, id, btn, ended)));
  root.querySelectorAll(".batch-files").forEach((f) => {
    if (f.dataset.pidx != null && scrolls[f.dataset.pidx] != null)
      f.scrollTop = scrolls[f.dataset.pidx];
  });
}

// Project action control: Cancel while active (queued/running) — the queue continues; Restart once
// finished (re-runs mid-batch by queuing until the GPU frees, or immediately if the batch ended).
// data-kind lets the in-place updater swap it only when the active↔finished state actually flips.
function projectActionBtn(p, i, id, btn, ended) {
  if (p.status === "running" || p.status === "queued") {
    const cancel = el("button", { className: "ghost icon-only bi-act", title: "Cancel this item" }, icon("close"));
    cancel.dataset.kind = "cancel";
    cancel.onclick = async () => {
      cancel.disabled = true; cancel.title = "Canceling…";
      try { await fetch(`/api/batch/${id}/project/${i}/cancel`, { method: "POST" }); } catch {}
    };
    return cancel;
  }
  const restart = el("button", { className: "ghost bi-act", textContent: "Restart",
    title: ended ? "Run this item again" : "Run this item again (queues until the batch frees up)" });
  restart.dataset.kind = "restart";
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
      if (ended) pollBatch(id, btn);                      // polling had stopped — resume it
    } catch (e) {
      $("#batch-err").textContent = "Couldn't reach server to restart.";
      restart.disabled = false; restart.textContent = "Restart";
    }
  };
  return restart;
}

function projectHead(p, i, id, btn, ended, meta) {
  return el("div", { className: "batch-item-head" },
    el("span", { className: "bi-name", textContent: p.name }),
    el("span", { className: `chip-status ${meta.cls}`, textContent: meta.label }),
    el("div", { className: "spacer" }),
    el("span", { className: "muted bi-elapsed", textContent: fmtElapsed(p.elapsed) }),
    projectActionBtn(p, i, id, btn, ended));
}

function projectDetail(p, meta, pct) {
  if (p.error) return el("div", { className: "bi-msg bi-err" }, icon("warn", 13), " " + p.error);
  if (p.status === "done") {
    return el("div", { className: "bi-msg muted",
      textContent: `${p.message || "done"}${p.output_dir ? " · " + p.output_dir : ""}` });
  }
  if (p.status === "running") {
    return el("div", { className: "bi-msg muted", textContent: `${pct}% · ${p.message || ""}` });
  }
  return el("div", { className: "bi-msg muted", textContent: p.message || meta.label });
}

// One project row: name + status chip + (cancel | restart), its own progress bar + detail.
function buildBatchItem(p, i, id, btn, ended) {
  const meta = BATCH_STATE[p.status] || { label: p.status, cls: "queued" };
  const pct = Math.round((p.progress || 0) * 100);
  const card = el("div", { className: `batch-item state-${meta.cls}` });
  card.dataset.pidx = String(i);
  card.append(projectHead(p, i, id, btn, ended, meta));
  // Per-item bar — only once there's something to show (skip idle queued rows).
  if (p.status === "running" || (pct > 0 && pct < 100)) {
    card.append(el("div", { className: "bar bi-bar" }, el("div", { style: `width:${pct}%` })));
  }
  card.append(projectDetail(p, meta, pct));
  // Per-file projects: a row per source file, each with its own status + cancel/restart.
  if (p.files && p.files.length) {
    const files = el("div", { className: "batch-files" });
    files.dataset.pidx = String(i);          // lets renderBatchStatus preserve its scroll across rebuilds
    p.files.forEach((f, fi) => files.append(buildBatchFile(f, i, fi, id, btn, ended)));
    card.append(files);
  }
  return card;
}

// Update a project card in place: refresh the (non-scrolling) head / bar / detail, but keep the
// scrollable file-list node alive and update its rows in place — that's what stops the scroll freeze.
function updateBatchItem(card, p, i, id, btn, ended) {
  if (!card) return;
  const meta = BATCH_STATE[p.status] || { label: p.status, cls: "queued" };
  const pct = Math.round((p.progress || 0) * 100);
  card.className = `batch-item state-${meta.cls}`;

  const files = card.querySelector(".batch-files");        // the scroll container — must survive
  [...card.children].forEach((c) => { if (c !== files) c.remove(); });
  const frag = document.createDocumentFragment();
  frag.append(projectHead(p, i, id, btn, ended, meta));
  if (p.status === "running" || (pct > 0 && pct < 100)) {
    frag.append(el("div", { className: "bar bi-bar" }, el("div", { style: `width:${pct}%` })));
  }
  frag.append(projectDetail(p, meta, pct));
  if (files) card.insertBefore(frag, files); else card.append(frag);

  if (files && p.files) {
    p.files.forEach((f, fi) =>
      updateBatchFile(files.querySelector(`.batch-file[data-fidx="${fi}"]`), f, i, fi, id, btn, ended));
  }
}

// File action control (Cancel while active / Restart once finished), data-kind tagged for in-place swap.
function fileActionBtn(f, pidx, fidx, id, btn, ended) {
  if (f.status === "running" || f.status === "queued") {
    const cancel = el("button", { className: "ghost icon-only bi-act", title: "Cancel this file" }, icon("close"));
    cancel.dataset.kind = "cancel";
    cancel.onclick = async () => {
      cancel.disabled = true; cancel.title = "Canceling…";
      try { await fetch(`/api/batch/${id}/project/${pidx}/file/${fidx}/cancel`, { method: "POST" }); } catch {}
    };
    return cancel;
  }
  const restart = el("button", { className: "ghost bi-act", textContent: "Restart",
    title: ended ? "Run this file again" : "Run this file again (queues until the card frees up)" });
  restart.dataset.kind = "restart";
  restart.onclick = async () => {
    restart.disabled = true; restart.textContent = "…";
    try {
      const r = await fetchTO(`/api/batch/${id}/project/${pidx}/file/${fidx}/restart`, { method: "POST" }, 30000);
      if (!r.ok) {
        $("#batch-err").textContent = "Restart failed: " + (await r.text());
        restart.disabled = false; restart.textContent = "Restart"; return;
      }
      if (ended) pollBatch(id, btn);                      // polling had stopped — resume it
    } catch (e) {
      $("#batch-err").textContent = "Couldn't reach server to restart.";
      restart.disabled = false; restart.textContent = "Restart";
    }
  };
  return restart;
}

function fileDetailText(f, meta, pct) {
  return f.error ? f.error
    : (f.status === "done" ? (f.message || "done")
       : (f.status === "running" ? `${pct}% · ${f.message || ""}` : (f.message || meta.label)));
}

// Per-file read-vs-GPU split, shown once extraction has been measured (running or done): the real
// time spent reading + extracting the source (ffmpeg) vs the actual GPU transcription.
function fileTiming(f) {
  if (!(f.extract_seconds > 0) || (f.status !== "done" && f.status !== "running")) return null;
  const gpu = f.elapsed || 0;            // elapsed is the GPU clock; extraction is tracked separately
  return `read ${fmtAudio(f.extract_seconds)} · gpu ${fmtAudio(gpu)}`;
}

// One file row inside a per-file project: name + status chip + cancel/restart + its own bar.
function buildBatchFile(f, pidx, fidx, id, btn, ended) {
  const meta = BATCH_STATE[f.status] || { label: f.status, cls: "queued" };
  const pct = Math.round((f.progress || 0) * 100);
  const row = el("div", { className: `batch-file state-${meta.cls}` });
  row.dataset.fidx = String(fidx);

  const detail = fileDetailText(f, meta, pct);
  const msg = el("span", { className: "muted bf-msg", textContent: detail, title: detail });
  if (f.error) msg.style.color = "var(--bleed)";

  const head = el("div", { className: "bf-head" },
    el("span", { className: "bf-name", textContent: f.name, title: f.name }),
    el("span", { className: `chip-status ${meta.cls}`, textContent: meta.label }));
  // Multi-track files: show which track is transcribing (track 2/5) so progress is legible even
  // before the bar moves. Single-track files have nothing useful to show here.
  if (f.status === "running" && f.tracks > 1) {
    head.append(el("span", { className: "muted bf-trk",
      textContent: `track ${f.track || 1}/${f.tracks}` }));
  }
  head.append(el("div", { className: "spacer" }), msg, fileActionBtn(f, pidx, fidx, id, btn, ended));
  row.append(head);
  const tl = fileTiming(f);
  if (tl) row.append(el("div", { className: "bf-timing muted", textContent: tl }));
  if (f.status === "running" || (pct > 0 && pct < 100)) {
    row.append(el("div", { className: "bar bf-bar" }, el("div", { style: `width:${pct}%` })));
  }
  return row;
}

// Update a file row in place — mutate fields only, never replace the row node, so the list's scroll
// position and momentum survive every poll.
function updateBatchFile(row, f, pidx, fidx, id, btn, ended) {
  if (!row) return;
  const meta = BATCH_STATE[f.status] || { label: f.status, cls: "queued" };
  const pct = Math.round((f.progress || 0) * 100);
  row.className = `batch-file state-${meta.cls}`;
  const head = row.querySelector(".bf-head");

  const chip = head.querySelector(".chip-status");
  chip.className = `chip-status ${meta.cls}`;
  chip.textContent = meta.label;

  let trk = head.querySelector(".bf-trk");
  if (f.status === "running" && f.tracks > 1) {
    if (!trk) { trk = el("span", { className: "muted bf-trk" }); head.insertBefore(trk, head.querySelector(".spacer")); }
    trk.textContent = `track ${f.track || 1}/${f.tracks}`;
  } else if (trk) {
    trk.remove();
  }

  const detail = fileDetailText(f, meta, pct);
  const msg = head.querySelector(".bf-msg");
  msg.textContent = detail; msg.title = detail;
  msg.style.color = f.error ? "var(--bleed)" : "";

  const act = head.querySelector(".bi-act");
  const wantKind = (f.status === "running" || f.status === "queued") ? "cancel" : "restart";
  if (!act || act.dataset.kind !== wantKind) {             // only swap when active↔finished flips
    const fresh = fileActionBtn(f, pidx, fidx, id, btn, ended);
    if (act) act.replaceWith(fresh); else head.append(fresh);
  }

  let tl = row.querySelector(".bf-timing");
  const tlText = fileTiming(f);
  if (tlText) {
    if (!tl) { tl = el("div", { className: "bf-timing muted" }); head.after(tl); }
    tl.textContent = tlText;
  } else if (tl) {
    tl.remove();
  }

  let bar = row.querySelector(".bf-bar");
  if (f.status === "running" || (pct > 0 && pct < 100)) {
    if (!bar) { bar = el("div", { className: "bar bf-bar" }, el("div", {})); row.append(bar); }
    bar.firstChild.style.width = pct + "%";
  } else if (bar) {
    bar.remove();
  }
}

// =======================================================================================
// Models tab — what's on disk, downloads (progress + Cancel, resumable), removal
// =======================================================================================
const LANG_SELECTS = ["#opt-language", "#batch-language", "#label-language"];
let modelsPoll = null;

const fmtSize = (mb) => (mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${mb} MB`);
const isDownloading = (m) => !!(m.download && m.download.active);   // queued, connecting, downloading or retrying

async function refreshModels() {
  let data;
  // first call loads the speech library on the server (~10 s), hence the long timeout
  try { data = await (await fetchTO("/api/models", {}, 60000)).json(); } catch { return; }
  renderModels(data);
  renderGroup("whisper", data.models, data.disk_free);
  renderGroup("align", data.models, data.disk_free);
  annotateLanguages(data.models, data.variants);
  $("#auto-portuguese").value = data.auto_portuguese || "";
  $("#pt-punctuation").checked = !!data.pt_punctuation;
  clearTimeout(modelsPoll);
  if (data.models.some(isDownloading)) modelsPoll = setTimeout(refreshModels, 1000);   // live progress
}

function modelAction(m, active) {
  const call = async (path, method = "POST") => {
    $("#models-err").textContent = "";
    const r = await fetch(`/api/models/${encodeURIComponent(m.id)}${path}`, { method });
    if (!r.ok) $("#models-err").textContent = (await r.json().catch(() => ({}))).detail || "request failed";
    refreshModels();
  };
  const d = m.download;
  if (isDownloading(m)) {
    const pct = d.total ? (100 * d.done) / d.total : 0;
    const cancel = el("button", { className: "ghost", textContent: "Cancel" });
    cancel.onclick = () => call("/cancel");
    return el("span", { className: "dl" }, el("div", { className: "bar" }, el("div", { style: `width:${pct}%` })),
      el("span", { className: "meta", textContent: d.text }), cancel);
  }
  if (m.installed) {
    const inUse = m.id === active;
    const rm = el("button", { className: "ghost", textContent: "Remove", disabled: inUse,
      title: inUse ? "in use — choose another speech model first" : "delete from disk" });
    rm.onclick = () => call("", "DELETE");
    // A one-language model (CAMÕES) is used automatically for its language, not chosen app-wide.
    const use = m.info.for_language ? el("span", { className: "meta", textContent: `used for ${m.info.languages}` })
      : m.kind === "whisper" && !inUse ? el("button", { className: "ghost", textContent: "Use", title: "make this the speech model every run uses" }) : "";
    if (use.tagName === "BUTTON") use.onclick = () => call("/use");
    return el("span", { className: "dl" }, el("span", { className: "ok", textContent: inUse ? "In use" : "Downloaded" }), use, rm);
  }
  const partial = d && d.done > 0;
  const get = el("button", { className: "ghost", textContent: partial ? "Resume download" : "Download" });
  get.onclick = () => call("/download");
  const note = d && d.status === "error" ? el("span", { className: "err", textContent: d.error })
    : d && d.status === "canceled" ? el("span", { className: "meta", textContent: "canceled — partial download kept" }) : "";
  return el("span", { className: "dl" }, note, get);
}

// One row per section: Download all (what isn't on disk), Cancel all, and progress across the queue.
// The buttons are built once and their labels updated in place: the 1 s refresh would otherwise
// swap a button out between mouse-down and mouse-up and the click would be lost.
function renderGroup(kind, models, free) {
  const box = $(`#models-${kind}-all`);
  const group = models.filter((m) => m.kind === kind);
  const missing = group.filter((m) => !m.installed && !isDownloading(m));
  const active = group.filter(isDownloading);
  const post = (ms, path) => Promise.all(ms.map((m) =>
    fetch(`/api/models/${encodeURIComponent(m.id)}${path}`, { method: "POST" }))).then(refreshModels);
  if (!box.firstChild) {
    box.all = el("button", { className: "ghost" });
    box.cancel = el("button", { className: "ghost", textContent: "Cancel all" });
    box.meta = el("span", { className: "meta" });
    box.all.onclick = () => {
      const { missing, free } = box.state, mb = missing.reduce((s, m) => s + m.size_mb, 0);
      const freeTxt = fmtSize(Math.round(free / 1e6));
      $("#models-err").textContent = "";
      if (mb * 1e6 > free) {
        $("#models-err").textContent = `Not enough disk space: these need ${fmtSize(mb)}, ${freeTxt} free.`;
        return;
      }
      if (confirm(`Download ${missing.length} models, ${fmtSize(mb)} in total?\n${freeTxt} free on disk. They download two at a time; the rest wait in a queue.`))
        post(missing, "/download");
    };
    box.cancel.onclick = () => post(box.state.active, "/cancel");
    box.append(box.all, box.cancel, box.meta);
  }
  box.state = { missing, active, free };
  const mb = missing.reduce((s, m) => s + m.size_mb, 0);
  box.all.textContent = `Download all (${missing.length} · ${fmtSize(mb)})`;
  box.all.style.display = missing.length ? "" : "none";
  box.cancel.style.display = box.meta.style.display = active.length ? "" : "none";
  if (active.length) {
    const total = active.reduce((s, m) => s + (m.download.total || m.size_mb * 1e6), 0);
    const done = active.reduce((s, m) => s + m.download.done, 0);
    const speed = active.reduce((s, m) => s + m.download.speed, 0);
    const queued = active.filter((m) => m.download.status === "queued").length;
    const parts = [`${active.length - queued} downloading` + (queued ? `, ${queued} queued` : ""),
      `${fmtSize(Math.round(done / 1e6))} of ${fmtSize(Math.round(total / 1e6))}`];
    if (speed > 0) parts.push(`${(speed / 1e6).toFixed(1)} MB/s`, `about ${fmtAudio((total - done) / speed)} left`);
    box.meta.textContent = parts.join(" · ");
  }
  box.style.display = missing.length || active.length ? "" : "none";
}

const modelNames = {};   // short name -> display name, for the benchmark table

function renderModels(data) {
  data.models.forEach((m) => { if (m.kind === "whisper") modelNames[m.id.slice(8)] = m.name; });
  const w = $("#models-whisper"), a = $("#models-align");
  w.innerHTML = a.innerHTML = "";
  for (const m of data.models) {
    const size = el("td", { className: "size", textContent: fmtSize(m.size_mb) });
    const act = el("td", { className: "act" }, modelAction(m, data.active));
    if (m.kind === "whisper") {
      const meta = `${m.info.accuracy} accuracy · ${m.info.speed} · ${m.info.languages}`;
      w.append(el("tr", {}, el("td", {}, el("div", { textContent: m.name }), el("div", { className: "meta", textContent: meta })), size, act));
    } else {
      a.append(el("tr", {}, el("td", { textContent: m.name.replace(/ word timing$/, "") }), size, act));
    }
  }
}

// Language dropdowns say up front when picking a language means a download (and how big).
function annotateLanguages(models, variants = {}) {
  const byId = Object.fromEntries(models.map((m) => [m.id, m]));
  for (const sel of LANG_SELECTS) {
    for (const o of $(sel)?.options || []) {
      if (!o.value) continue;
      o.dataset.base ||= o.textContent;
      // a plain language needs its word-timing model; a variant (pt-PT) may also need its speech model
      const need = (variants[o.value] || [`align:${o.value}`]).map((id) => byId[id])
        .filter((m) => m && !m.installed && !isDownloading(m));
      const mb = need.reduce((n, m) => n + m.size_mb, 0);
      o.textContent = mb ? `${o.dataset.base} · ${fmtSize(mb)} download` : o.dataset.base;
    }
  }
}

$("#auto-portuguese").onchange = async (e) => {
  await fetch("/api/settings/auto-portuguese", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ value: e.target.value || null }) });
};

$("#pt-punctuation").onchange = async (e) => {
  await fetch("/api/settings/pt-punctuation", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ value: e.target.checked }) });
  refreshModels();   // pt-PT download size now includes large-v3
};

document.querySelectorAll(".tab-btn").forEach((b) => b.addEventListener("click", refreshModels));
document.querySelectorAll(".tab-btn").forEach((b) => b.addEventListener("click", refreshCacheStats));
refreshModels();

// Benchmark: every downloaded speech model, cold load + transcription of a spoken test sentence.
$("#bench-run").onclick = async () => {
  const run = $("#bench-run"), cancel = $("#bench-cancel"), status = $("#bench-status");
  const r = await fetch("/api/benchmark", { method: "POST" });
  if (!r.ok) { status.textContent = (await r.json().catch(() => ({}))).detail || "couldn't start"; return; }
  const { job_id } = await r.json();
  run.disabled = true; cancel.style.display = ""; cancel.disabled = false;
  cancel.onclick = () => { cancel.disabled = true; fetch(`/api/cancel/${job_id}`, { method: "POST" }); };
  let p;
  do {
    await new Promise((res) => setTimeout(res, 700));
    p = await (await fetchTO(`/api/progress/${job_id}`)).json();
    status.textContent = p.message || p.status;
  } while (p.status === "queued" || p.status === "running");
  run.disabled = false; cancel.style.display = "none";
  if (p.status === "error") { status.textContent = "error: " + p.error; return; }
  if (p.status === "cancelled") { status.textContent = "canceled"; return; }
  const { expected, results } = await (await fetch(`/api/benchmark/${job_id}`)).json();
  status.textContent = "";
  const t = $("#bench-results");
  t.style.display = "";
  t.innerHTML = "";
  // One group per test sentence (English; European Portuguese when a CAMÕES model is downloaded).
  for (const test of [...new Set(results.map((x) => x.test))]) {
    t.append(el("tr", {}, el("td", { colSpan: 6 }, el("strong", { textContent: test }),
      el("div", { className: "meta", textContent: `Said: "${expected[test]}"` }))));
    t.append(el("tr", {}, ...["Model", "Load", "Transcribe", "Speed", "Match", "What it heard"]
      .map((h) => el("td", { className: "meta", textContent: h }))));
    for (const x of results.filter((r) => r.test === test)) {
      const name = (modelNames[x.model] || `Whisper ${x.model}`);
      t.append(el("tr", {},
        el("td", { textContent: name }),
        el("td", { className: "size", textContent: `${x.load_s}s` }),
        el("td", { className: "size", textContent: `${x.transcribe_s}s` }),
        el("td", { className: "size", textContent: `${x.speed_x}× realtime`, title: `${x.audio_s}s of audio` }),
        el("td", { className: "size", textContent: `${Math.round(x.match * 100)}%` }),
        el("td", { className: "meta", textContent: x.heard })));
    }
  }
};
