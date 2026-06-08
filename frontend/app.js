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
function renderEditor(container, merged) {
  container.innerHTML = "";
  container.append(buildToolbar(container, merged), buildExportPanel(merged), buildTable(container, merged));
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

  const addBtn = el("button", { className: "ghost", textContent: "+ Add line" });
  addBtn.onclick = () => {
    const last = merged.segments[merged.segments.length - 1];
    const start = last ? last.end : 0;
    merged.segments.push({ speaker: from.value || "Speaker", start, end: start + 1,
      text: "", words: [], flags: [] });
    renderEditor(container, merged);
  };

  return el("div", { className: "row", style: "margin-bottom:12px" },
    el("span", { className: "muted", textContent: "Speaker:" }), from, to, renameBtn,
    el("div", { className: "spacer" }), addBtn);
}

function buildTable(container, merged) {
  const tbody = el("tbody");
  const table = el("table", { className: "editor" },
    el("thead", {}, el("tr", {},
      el("th", { textContent: "Speaker" }), el("th", { textContent: "Start" }),
      el("th", { textContent: "End" }), el("th", { textContent: "Text" }),
      el("th", { textContent: "Flags" }), el("th", { textContent: "" }))),
    tbody);

  merged.segments.forEach((seg, i) => tbody.append(buildRow(container, merged, seg, i)));
  return table;
}

function buildRow(container, merged, seg, i) {
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

  tr.append(
    el("td", { className: "spk" }, spk),
    el("td", { className: "tc" }, start),
    el("td", { className: "tc" }, end),
    el("td", { className: "txt" }, txt),
    flagsCell,
    el("td", {}, del));
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
  const cpl = el("input", { type: "number", value: 42 });
  const maxLines = el("input", { type: "number", value: 2 });
  const minDur = el("input", { type: "number", value: 1, step: 0.5 });
  const showSpk = el("input", { type: "checkbox", checked: true });

  const btn = el("button", { className: "primary", textContent: "Export" });
  btn.onclick = async () => {
    const options = {
      chars_per_line: +cpl.value, max_lines: +maxLines.value,
      min_duration: +minDur.value, show_speakers: showSpk.checked,
    };
    await downloadExport(merged, format.value, filename.value || "transcript", options);
  };

  const field = (label, node) => el("label", { className: "field" }, el("span", { textContent: label }), node);
  return el("div", { className: "row", style: "margin:14px 0; padding-top:14px; border-top:1px solid var(--line)" },
    field("Format", format), field("Filename", filename),
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
  const dz = $("#dropzone"), input = $("#file-input");
  dz.onclick = () => input.click();
  input.onchange = () => handleFiles(input.files);
  dz.ondragover = (e) => { e.preventDefault(); dz.classList.add("over"); };
  dz.ondragleave = () => dz.classList.remove("over");
  dz.ondrop = (e) => { e.preventDefault(); dz.classList.remove("over"); handleFiles(e.dataTransfer.files); };
  $("#run-btn").onclick = runPipeline;
})();

async function handleFiles(fileList) {
  if (!fileList.length) return;
  const fd = new FormData();
  for (const f of fileList) fd.append("files", f);
  const res = await fetch("/api/upload", { method: "POST", body: fd });
  if (!res.ok) { alert("Upload failed: " + (await res.text())); return; }
  const items = await res.json();
  items.forEach((it) => uploaded.push(it));
  renderSources();
  $("#options-panel").style.display = "";
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
          sel, el("span", { className: "ti", textContent: `track ${t.track_index}` }),
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
      renderEditor($("#editor-transcribe"), merged);
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
  renderEditor($("#editor-edit"), merged);
}
