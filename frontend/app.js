"use strict";

// --- tiny helpers ----------------------------------------------------------------------
const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, props = {}, ...kids) => {
  const n = Object.assign(document.createElement(tag), props);
  for (const k of kids) n.append(k);
  return n;
};
const fmtTime = (s) => {
  s = Math.max(0, s || 0);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = (s % 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${sec.toFixed(2).padStart(5, "0")}`;
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
function playClip(fileId, start, dur) {
  if (previewAudio) previewAudio.pause();
  previewAudio = new Audio(`/api/clip/${fileId}?start=${start}&dur=${Math.max(2, Math.min(dur, 30))}`);
  previewAudio.play();
}
function fmtClock(s) {
  s = Math.floor(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function renderEditor(container, merged, preview) {
  if (preview !== undefined) container._preview = preview;  // persist across re-renders
  preview = container._preview || null;
  container.innerHTML = "";
  container.append(buildToolbar(container, merged), buildExportPanel(merged));
  if (preview) container.append(buildPreviewBar(preview));
  container.append(buildTable(container, merged, preview));
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

function buildTable(container, merged, preview) {
  const tbody = el("tbody");
  const table = el("table", { className: "editor" },
    el("thead", {}, el("tr", {},
      el("th", { textContent: "Speaker" }), el("th", { textContent: "Start" }),
      el("th", { textContent: "End" }), el("th", { textContent: "Text" }),
      el("th", { textContent: "Flags" }), el("th", { textContent: "" }))),
    tbody);

  merged.segments.forEach((seg, i) => tbody.append(buildRow(container, merged, seg, i, preview)));
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

  const txt = el("input", { value: seg.text });
  txt.oninput = () => {
    seg.text = txt.value;
    if (seg.words && seg.words.length && !seg.flags.includes("edited")) {
      seg.flags.push("edited");           // text changed -> word timings may be stale
      renderFlags(flagsCell, seg);
      tr.classList.add("flag-edited");
    }
  };
  txt.onkeydown = (e) => {
    if (e.key === "Enter") { e.preventDefault(); focusRow(container, i + 1); }
    else if (e.key === "Escape") { Object.assign(seg, JSON.parse(snapshot)); renderEditor(container, merged); }
    else if (e.key === "ArrowDown" && e.ctrlKey) { e.preventDefault(); focusRow(container, i + 1); }
    else if (e.key === "ArrowUp" && e.ctrlKey) { e.preventDefault(); focusRow(container, i - 1); }
  };

  const flagsCell = el("td");
  renderFlags(flagsCell, seg);

  const del = el("button", { className: "ghost", textContent: "✕", title: "Remove line" });
  del.onclick = () => { merged.segments.splice(i, 1); renderEditor(container, merged); };

  const actions = el("td", {});
  if (preview) {
    const jump = el("button", { className: "ghost", textContent: "▶", title: "Play from here" });
    jump.onclick = () => playClip(preview.file_id, seg.start, seg.end - seg.start);
    actions.append(jump);
  }
  actions.append(del);

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

function focusRow(container, i) {
  const rows = container.querySelectorAll("table.editor tbody tr");
  if (i >= 0 && i < rows.length) rows[i]._textInput.focus();
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
function buildExportPanel(merged) {
  const format = el("select");
  ["srt", "txt", "csv", "json"].forEach((f) => format.append(el("option", { value: f, textContent: f.toUpperCase() })));
  const filename = el("input", { value: "transcript", size: 14 });
  const applyLimits = el("input", { type: "checkbox", checked: true });
  const cpl = el("input", { type: "number", value: 42 });
  const maxLines = el("input", { type: "number", value: 2 });
  const minDur = el("input", { type: "number", value: 1, step: 0.5 });
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
    };
    await downloadExport(merged, format.value, filename.value || "transcript", options);
  };

  const field = (label, node) => el("label", { className: "field" }, el("span", { textContent: label }), node);
  return el("div", { className: "row", style: "margin:14px 0; padding-top:14px; border-top:1px solid var(--line)" },
    field("Format", format), field("Filename", filename),
    el("label", { className: "opt" }, applyLimits, "Subtitle limits"),
    field("Chars/line", cpl), field("Max lines", maxLines), field("Min dur (s)", minDur),
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

(function initTranscribeTab() {
  $("#choose-btn").onclick = browseFiles;
  $("#run-btn").onclick = runPipeline;
  const detect = $("#opt-detect");
  const syncDetect = () => MERGE_THRESHOLD_IDS.forEach((id) => { $(`#${id}`).disabled = !detect.checked; });
  detect.onchange = syncDetect;
  syncDetect();
})();

function registerItems(items) {
  items.forEach((it) => uploaded.push(it));
  renderSources();
  $("#options-panel").style.display = "";
}

async function browseFiles() {
  const err = $("#path-err"), btn = $("#choose-btn");
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
  const btn = el("button", { className: "ghost", textContent: "▶", title: "Audition track" });
  const audio = new Audio(`/api/audition/${fileId}/${trackIndex}`);
  audio.onended = () => { btn.textContent = "▶"; };
  btn.onclick = () => {
    if (currentAudio && currentAudio !== audio) { currentAudio.pause(); currentAudio.currentTime = 0; }
    if (audio.paused) { currentAudio = audio; audio.play(); btn.textContent = "⏸"; }
    else { audio.pause(); btn.textContent = "▶"; }
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
    if (it.source.tracks.length > 1) {
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

const MERGE_THRESHOLD_IDS = ["opt-backchannel", "opt-bc-words", "opt-overlap", "opt-bleed-win", "opt-bleed-sim"];

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

  const res = await fetch("/api/transcribe", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sources, options: gatherOptions() }),
  });
  if (!res.ok) { setProgress(0, "error: " + (await res.text())); runBtn.disabled = false; return; }
  const { job_id } = await res.json();
  pollJob(job_id, runBtn);
}

async function pollJob(jobId, runBtn) {
  const tick = async () => {
    const p = await (await fetch(`/api/progress/${jobId}`)).json();
    setProgress(p.progress, p.message || p.status);
    if (p.status === "done") {
      const merged = await (await fetch(`/api/result/${jobId}`)).json();
      $("#result-panel").style.display = "";
      // preview audio only when one source (merged timeline == that file's timeline)
      const preview = uploaded.length === 1
        ? { file_id: uploaded[0].file_id, duration: uploaded[0].source.duration } : null;
      renderEditor($("#editor-transcribe"), merged, preview);
      setProgress(1, "done");
      runBtn.disabled = false;
      return;
    }
    if (p.status === "error") { setProgress(p.progress, "error: " + p.error); runBtn.disabled = false; return; }
    setTimeout(tick, 500);
  };
  tick();
}

function setProgress(frac, msg) {
  $("#progress-bar").style.width = `${Math.round((frac || 0) * 100)}%`;
  $("#progress-msg").textContent = msg || "";
}

// =======================================================================================
// Edit Transcript tab
// =======================================================================================
(function initEditTab() {
  const dz = $("#dropzone-edit"), input = $("#file-input-edit");
  dz.onclick = () => input.click();
  input.onchange = () => input.files[0] && readJsonFile(input.files[0]);
  dz.ondragover = (e) => { e.preventDefault(); dz.classList.add("over"); };
  dz.ondragleave = () => dz.classList.remove("over");
  dz.ondrop = (e) => { e.preventDefault(); dz.classList.remove("over"); e.dataTransfer.files[0] && readJsonFile(e.dataTransfer.files[0]); };
  $("#json-load").onclick = () => loadEdit($("#json-in").value);
})();

function readJsonFile(file) {
  const r = new FileReader();
  r.onload = () => loadEdit(r.result);
  r.readAsText(file);
}

function loadEdit(text) {
  const errSpan = $("#json-err");
  errSpan.textContent = "";
  let merged;
  try { merged = JSON.parse(text); } catch (e) { errSpan.textContent = "Invalid JSON"; return; }
  if (!merged || !Array.isArray(merged.segments)) { errSpan.textContent = "Missing segments[]"; return; }
  merged.version = merged.version || "1.0";
  merged.sources = merged.sources || [];
  merged.segments.forEach((s) => { s.words = s.words || []; s.flags = s.flags || []; });
  $("#edit-panel").style.display = "";
  renderEditor($("#editor-edit"), merged, null);  // loaded JSON has no source file to preview
}

// =======================================================================================
// Batch / Studio tab
// =======================================================================================
const projects = [];           // [{ name, files: [{file_id, source}] }]
let outputFolder = null;
let projectCounter = 0;

(function initBatchTab() {
  $("#add-project").onclick = () => {
    projectCounter += 1;
    projects.push({ name: `Project ${projectCounter}`, files: [] });
    renderProjects();
  };
  $("#choose-folder").onclick = chooseFolder;
  $("#run-batch").onclick = runBatch;
})();

function renderProjects() {
  const root = $("#projects");
  root.innerHTML = "";
  projects.forEach((proj, pi) => root.append(buildProjectCard(proj, pi)));
}

function buildProjectCard(proj, pi) {
  const box = el("div", { className: "source" });
  const name = el("input", { value: proj.name, style: "font-weight:600; width:240px" });
  name.oninput = () => { proj.name = name.value; };
  const addFiles = el("button", { className: "ghost", textContent: "Add files" });
  addFiles.onclick = () => addProjectFiles(proj);
  const removeProj = el("button", { className: "ghost", textContent: "✕ Remove project" });
  removeProj.onclick = () => { projects.splice(pi, 1); renderProjects(); };
  box.append(el("div", { className: "name" },
    name, el("div", { className: "spacer" }), addFiles, removeProj));

  proj.files.forEach((_, fi) => box.append(buildFileRow(proj, fi)));
  if (!proj.files.length) box.append(el("div", { className: "muted", textContent: "no files yet — click Add files" }));
  return box;
}

function buildFileRow(proj, fi) {
  const f = proj.files[fi];
  const wrap = el("div", { style: "border-top:1px solid var(--line); padding:6px 0; margin-top:6px" });
  const swap = (a, b) => { [proj.files[a], proj.files[b]] = [proj.files[b], proj.files[a]]; renderProjects(); };
  const up = el("button", { className: "ghost", textContent: "↑", title: "Move earlier" });
  up.onclick = () => { if (fi > 0) swap(fi, fi - 1); };
  const down = el("button", { className: "ghost", textContent: "↓", title: "Move later" });
  down.onclick = () => { if (fi < proj.files.length - 1) swap(fi, fi + 1); };
  const rm = el("button", { className: "ghost", textContent: "✕" });
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

async function addProjectFiles(proj) {
  $("#batch-err").textContent = "";
  const res = await fetch("/api/browse", { method: "POST" });
  if (!res.ok) { $("#batch-err").textContent = "Picker failed: " + (await res.text()); return; }
  proj.files.push(...(await res.json()));
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

  const body = {
    output_mode: mode,
    output_folder: outputFolder,
    options: { detect_conflicts: $("#batch-detect").checked },
    projects: usable.map((p) => ({
      name: p.name,
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
  const res = await fetch("/api/batch", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!res.ok) { $("#batch-err").textContent = "Batch failed: " + (await res.text()); btn.disabled = false; return; }
  const { batch_id } = await res.json();
  $("#batch-status-panel").style.display = "";
  pollBatch(batch_id, btn);
}

function pollBatch(id, btn) {
  const tick = async () => {
    const b = await (await fetch(`/api/batch/${id}`)).json();
    renderBatchStatus(b);
    if (b.status === "done") { btn.disabled = false; return; }
    setTimeout(tick, 800);
  };
  tick();
}

function renderBatchStatus(b) {
  const root = $("#batch-status");
  root.innerHTML = "";
  b.projects.forEach((p) => {
    const pct = Math.round((p.progress || 0) * 100);
    const state = p.status === "running" ? `running ${pct}%` : p.status;
    const detail = p.error ? "⚠ " + p.error : (p.status === "done" ? "→ " + (p.output_dir || "") : p.message || "");
    const line = el("div", { className: "track-row" },
      el("span", { textContent: p.name, style: "width:200px; font-weight:600" }),
      el("span", { className: "muted", textContent: state, style: "width:120px" }),
      el("span", { className: "muted", textContent: detail, style: "flex:1" }));
    if (p.status === "error") line.classList.add("flag-bleed_suspected");
    root.append(line);
  });
}
