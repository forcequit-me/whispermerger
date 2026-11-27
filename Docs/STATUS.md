# WhisperMerger — Build Status

Snapshot of what's built vs. outstanding, mapped against `whispermerger-prd.md` (FR-\* IDs)
and `whispermerger-build-plan.md` (steps 1–10). Last updated 2026-06-20.

## Summary

Both v1 (multi-track stems + batch studio) and v2 (single-track manual diarization) are
**built and running** on native Windows + RTX 3080 Ti (the build plan's env was validated;
WSL2 not needed). Full pipeline works end-to-end: pick files → transcribe on GPU (WhisperX
large-v3 + wav2vec2) → merge with conflict flags → edit → export. **Validated on a real 56 GB /
1h51m / 4-track recording** — full transcription in ~10 min (beats NFR-1). **103 GPU-independent
tests pass**; the GPU path passes the §11.3 smoke gate. The Edit tab now imports all of our
export formats (JSON/SRT/TXT/CSV); remaining work is optional polish (some `FR-UI-*` toggles)
and proprietary-format import (Descript/Premiere).

## Built ✅

**Build-plan steps 1–10:** all done (env + smoke gate, data model, merge engine, probe/extract,
cache, WhisperX wrapper, FastAPI, export, fresh frontend, polish).

**Input (§8.1):** FR-IN-1/2/3 audio + multi-track video + mixed in one batch · FR-IN-4 ffprobe
streams+duration · FR-IN-5 ffmpeg mono/16k/pcm_s16le extraction · FR-IN-6 temp deleted after use.

**Track ID (§8.2):** FR-TK-1 single InputSource model · FR-TK-3 per-track audition · FR-TK-4
selector only when >1 stream. _(FR-TK-2: track labels default to **Speaker 1/2/…** (1-based),
editable — changed from the originally-specced OBS convention.)_

**Transcription (§8.3):** FR-TR-1 large-v3 + wav2vec2 align · FR-TR-2 float16 · FR-TR-3 VAD ·
FR-TR-4 progress · FR-TR-5 auto-merge on completion.

**Merge (§8.4):** FR-MG-1 interleave by time · FR-MG-2 backchannel (flag + parenthetical/
separate/skip) · FR-MG-3 genuine-overlap flag · FR-MG-5 conflicts surfaced in editor. Conflict
detection is toggleable.

**Word data (§8.5):** FR-WD-1 words stored under segments · FR-WD-2 preserved in JSON ·
FR-WD-3 SRT line splits use real word timestamps.

**Export (§8.7):** FR-EX-1 SRT/TXT/CSV/JSON · FR-EX-2 word-level SRT splits. All four formats
carry start+end timing; subtitle limits (CPL / max-lines / min-duration) toggleable.

**Caching (§8.8):** FR-CA-1/2/3 SHA-256(file)+model+align+language key; re-runs skip transcription.

**Modes (§8.9):** FR-MD-1 Transcribe & Merge · FR-MD-2 Edit Transcript · FR-MD-3 sequence-only
retired (folded into the pipeline).

**Editor (§8.6, partial):** FR-UI-1 inline edit speaker/text/timing · FR-UI-2 global rename ·
FR-UI-8 custom export filename · FR-UI-10 conflict-flag highlighting. Plus **remove a speaker +
all their lines** (destructive) and a non-destructive **per-speaker show/hide toggle** (hides a
speaker's lines from the table and from every export) (new requests).

## Built beyond the PRD (this chat)

- **Batch "transcription studio"** — Method 1 (a project's session files auto-merge sequentially
  with cumulative offsets) + Method 2 (queue many projects, each exports to its own subfolder).
  Output is now selectable (see 2026-06-18 note): merge vs one-transcript-per-file and which
  formats. Per-project error isolation; cache-resumable. New `batch.py` + `/api/batch`.
- **In-place file reads via a native picker** (`/api/browse`, `/api/browse-folder`) — no upload/
  copy, so 100–200 GB files work. Replaced the byte-upload flow.
- Toggleable conflict detection and subtitle limits; timestamps added to TXT export.
- **Per-line audio preview** — ▶ a line or drag a jump bar; instant input-seek clips, works on
  100+ GB files (`/api/clip`).
- **Stage-aware progress + total time** — jobs/batches report the current stage (Extracting ·
  Transcribing · Merging, per track) and elapsed wall time across _all_ stages, not just
  transcription, so the UI never looks stalled.
- **Storage hygiene** — each extracted track wav is deleted immediately after use; the working
  dir (extracted tracks, combined wavs, preview clips, audition samples) is wiped on server
  startup. Only the small WhisperX output cache is kept.
- Documented torch-2.6/Windows dependency fixes (setuptools<81, transformers<5, weights_only,
  cuDNN 8 on PATH) — see `SETUP.md`.

## Editing & workflow polish (2026-06-10)

- **Waveform + scrubbing in every editor** — new `/api/peaks/{file_id}` serves downsampled
  max-abs peaks (`extract.compute_peaks`, downmix of all streams, cached); the editor draws a
  lightweight `<canvas>` waveform (`buildWaveform` in `app.js`) with click-to-play (input-seek
  clip), an animated playhead, **horizontal zoom** (min = whole file in view), and **vertical /
  amplitude zoom** (gain for quiet audio). No full-audio download — scales to 100+ GB.
- **Export padding** — `ExportOptions.pad_seconds` + `export.pad_segments`; `_render` pads once so
  all four formats carry the padded timecodes. One `Pad (s)` field in the export panel. Render-time
  only (source transcript untouched; words/SRT word-splitting unaffected). Covers FR-style
  "breathing room for video editing".
- **Label Track: zoom + region pad** — WaveSurfer horizontal zoom (relative to a fit-whole-file
  baseline, so slider min shows the entire track start-to-finish) + vertical/amplitude zoom
  (`barHeight`), and a `Region pad (s)` input that widens VAD-seeded regions (clamped to
  `[0, duration]`) so words at region edges aren't clipped before per-speaker transcription.
  Split-lane blocks now render the **actual region waveform** (drawn from WaveSurfer's decoded
  audio via `getDecodedData`), not flat boxes, and the lane strip honors the same horizontal +
  amplitude zoom (scrolls horizontally when zoomed, with pinned speaker labels).
- **Edit tab: attach audio** — a loaded-from-JSON transcript can now bind a source file
  (`/api/browse`) to enable the waveform + per-line ▶ for verification (previously preview-less).
- **One-click start/stop** — `Startup.bat` / `Shutdown.bat` at the repo root, plus a `⏻ Shutdown`
  header button → `POST /api/shutdown` (hard `os._exit` on a background thread after the response
  flushes; safe since temp is wiped on next startup).
- Tests: `test_export.py` (padding shift/clamp + padded export timecodes) and `test_extract.py`
  (`compute_peaks` shape/range). **70 GPU-independent tests pass.**

## UI redesign + reliability (2026-06-11)

- **Design system docs (new, repo root):** `PRODUCT.md` (register=product, users, brand
  personality, anti-references, design principles, a11y targets) and `DESIGN.md` + `DESIGN.json`
  (tokens + components, North Star **"The Workbench"**). Generated via the impeccable skill so
  future UI work stays on-brand.
- **Visual overhaul:** `favicon.svg` + header brand mark; **all emoji-as-icons replaced with
  inline SVG** (play/pause/close/up/down/warn + speaker swatch); **monospace numerics** for every
  readout (timecodes/durations/counts/%); darker tinted base (`--bg #0e1014`) for clearer panel
  separation; header reworked — tabs then a quiet right-aligned **Shutdown** (muted, red on hover)
  + a thin accent rule; working width widened (`main` 1400, Label tab uncapped for the waveform).
- **Entry states:** each tab now opens with a **full-bleed "loader" entry zone** (waveform mark,
  headline, primary action, `1 · 2 · 3` step hint) that swaps to the working panels the moment
  files load; the Edit drop zone collapses on load with a **"Load another…"** reset. Fixes the
  bare "button in an empty box" empty states.
- **Progress everywhere:** Transcribe keeps the **stage stepper** (Extract → Transcribe → Merge)
  + big %; **progress bars added to Batch (overall) and Label**, so every processing tab shows
  one (previously Transcribe only).
- **Split-lane:** DAW **time ruler** (m:ss ticks), taller lanes with a centre gridline, sticky
  **swatch + name + region-count** headers (identity survives without colour — colourblind-safe).
- **Spacebar transport:** play/pause the most-recently-played audio (track preview, per-line clip,
  or the Label waveform) unless focus is in a text field; play/pause glyphs reflect real state.
- **Accessibility (WCAG AA pass):** accent `:focus-visible` ring on every control; Edit drop zone
  made keyboard-reachable + `aria-label`'d; `prefers-reduced-motion` honoured; radio/range accent
  colours.
- **Reliability:** every job-start POST and **polling loop** (`pollJob` / `pollBatch` / label
  `poll`) now uses an **abort-timeout fetch** (`fetchTO`) and **survives transient server
  unresponsiveness** (first-run model load ties the server up) instead of silently freezing at
  "starting…"; clearer "still working… first-run model load" messaging. Cache-busting `?v=` on
  css/js so stale assets don't linger during iteration.

## Editor toggles + multi-format import + wrapping rows (2026-06-18)

All front-end, in the shared editor (`renderEditor` in `app.js`) so they land in **all three**
tabs (Transcribe & Merge, Label Track, Edit Transcript) at once. No backend changes.

- **Per-speaker show/hide toggle** — new `buildSpeakerBar` renders an "Include:" chip per distinct
  speaker (with line count); clicking toggles a `container._disabled` set. Disabled speakers are
  **omitted from the table entirely** (`buildTable` skips their segments, preserving true indices
  for delete/edit) **and filtered out of every export** (`buildExportPanel` posts only the kept
  segments to `/api/export`). Non-destructive — distinct from the existing "Remove speaker". An
  all-off export is blocked with a "nothing to export" alert. (Started as a dim/strike treatment;
  switched to full hide for a cleaner view — `row-muted` styling removed.)
- **Edit tab imports all export formats** — file picker/drop/paste now accept `.json/.srt/.txt/
  .csv`; `parseTranscript(text, name)` dispatches by extension (sniff fallback) to JSON / SRT /
  TXT / CSV parsers that mirror `export.py` (incl. a small RFC-4180 CSV reader and `Speaker:`
  prefix stripping). JSON keeps `words`; the rest import with `words:[]`.
- **Auto-growing, wrapping text cells** — the per-row text `<input>` became an auto-sizing
  `<textarea>` (`autoGrow` sets height from `scrollHeight`); long lines now wrap and the row grows
  **downward** (cells are `vertical-align: top`) instead of overflowing horizontally. **Enter now
  inserts a line break** (line breaks persist in `seg.text` and ride through JSON/CSV cleanly,
  fold into SRT wrapping); row navigation moved to sibling-based `focusSibling` for Ctrl+↑/↓ so it
  skips hidden rows.
- **Per-line preview ▶/⏸ toggle** — the editor's per-line play button (`toggleClip` in `app.js`)
  now flips to a pause glyph while its clip plays and reverts on pause/end; click again to
  pause/resume, and starting another line reverts the previous button. Matches the existing
  track-audition (`auditionBtn`) and Label-waveform buttons; icons follow the audio's real state
  so spacebar stays in sync.
- **Wider Start/End fields** — `table.editor td.tc input` widened 78px → 116px so the full
  `HH:MM:SS.ss` monospace timecode shows without clipping.
- **Editor column layout** — Text column now absorbs the slack (`td.txt { width:100% }`) so
  Speaker/Start/End group tightly on the left and the transcript is the widest column; tighter
  `td.spk`/`td.tc` padding; per-line play and delete buttons separated by a `.row-actions` flex gap.
- **"Flags" column → "Notes", auto-hidden when empty** — the column is hidden via
  `table.editor:not(.has-notes) .notes-col` until a visible line carries a flag (conflict or
  `edited`); editing a word-timed line reveals it live (`tr.closest("table.editor").add("has-notes")`)
  without losing focus.
- Asset cache-bust bumped to `?v=13` (css + js).

## Batch output settings — global + per-project (2026-06-18)

Lets a big library export the way the user wants (e.g. ~200h → one TXT per recording to feed an AI).

- **Format selection** — `export.export_all(..., formats=None)` filters which of srt/txt/csv/json
  to write (`None` = all four, unchanged default). Plumbed through `batch.run_batch` per project.
- **Per-file vs merge, per project** — `batch.ProjectInput` gains `per_file` + `formats`;
  `run_batch` either merges a project's files (default) or runs the pipeline per source file and
  writes one transcript each (basename = source file stem, de-duped) into the project folder.
  Per-file progress aggregates as "file X/N". `routes/api.py` `BatchProjectIn` carries `per_file`
  + `formats` (validated non-empty).
- **UI: global defaults + override** — Batch output panel has global "one transcript per file" +
  format checkboxes; each project card has an "Override output settings" toggle revealing its own
  per-file + format picks. The frontend resolves global-vs-override into concrete per-project
  values before POSTing, so the backend has no global/override logic. Timing UI unchanged
  (per-project + total elapsed already tracked).
- Tests: `test_export.py` (formats filter) + `test_batch.py` (per-file → one transcript per source;
  formats filter limits a merge project).

## Language selection + download reliability (2026-06-19)

- **Forceable transcription language** — a **Language** dropdown (Batch output panel +
  Transcribe & Merge options), default **Auto-detect**. `routes/api.py` `TranscribeRequest` /
  `BatchRequest` carry `language`; it threads to `transcribe.transcribe_track(..., language=)`,
  which passes it to `model.transcribe` and forces the matching wav2vec2 align model. Choosing a
  language skips Whisper's per-file detection and the first-use download of that language's align
  model.
- **Cache key includes language** — `cache.cache_key(..., language)` adds the requested language
  ("auto" when unset) so an auto-detect run and a forced-language run can't return each other's
  cached results. New `test_cache.py::test_key_depends_on_language`.
- **HF Xet downloader disabled** (`HF_HUB_DISABLE_XET=1` in `Startup.bat`) — the Xet transfer
  backend hung indefinitely mid-download on the model CDN (CloudFront), wedging a batch with the
  GPU idle; plain HTTPS is reliable. Related Windows gotcha worth knowing: **QuickEdit Mode** —
  clicking in the server console pauses the next stdout write and freezes uvicorn's event loop
  until a key is pressed.
- Tests: **74 GPU-independent tests pass.**

## Batch folder ingest + per-file nested output (2026-06-19)

- **"Add files / folder" per project** — the project add button now opens a tiny Files/Folder
  chooser (Windows can't pick both in one native dialog). **Files** → `/api/browse` (unchanged);
  **Folder** → new `/api/add-folder` opens the native folder picker and registers **every
  audio/video file underneath, recursively** (`_MEDIA_EXTS` filter), display name = path relative
  to the chosen folder so nested files stay distinguishable; unreadable / audio-less files are
  skipped, not fatal.
- **Per-file output preserves the tree** — `batch.run_batch`, in `per_file` + `output_mode="source"`,
  now writes each transcript **next to its own source file** (`Path(src.path).parent`) instead of
  collapsing them into the first file's folder; dedupe is per-destination. So a folder of nested
  recordings yields transcripts beside each original.
- Tests: `test_api.py` (recursive `/add-folder` + cancel) and `test_batch.py`
  (`test_per_file_source_writes_beside_each_source`). **77 GPU-independent tests pass.** Assets `?v=15`.

## Job & batch cancel + retry (2026-06-19)

- **Cancel is cooperative** (a thread can't safely be killed mid-WhisperX), so it lands at the next
  stage/file boundary — the in-flight file finishes. `jobs.Job.cancel()` sets a `threading.Event`
  the progress callback checks (raises `JobCancelled` → status `cancelled`); `batch` mirrors it with
  a `cancelled` flag checked between projects/files (raises `Cancelled` → `canceled`), remaining
  projects marked canceled. Endpoints: `POST /api/cancel/{job_id}`, `POST /api/batch/{id}/cancel`.
- **Retry failed/canceled projects** — `BatchRegistry` persists a `_RunSpec` (projects + transcribe
  fn + output settings) per batch; `POST /api/batch/{id}/retry` re-runs the errored/canceled projects
  (or an explicit index list) in the same BatchStatus. Cache makes already-done files instant. The
  per-project loop was extracted into `_process_project` / `_run_projects` so run and retry share it.
- **UI** — **Cancel** on the Transcribe progress panel; **Cancel batch** + **Retry N failed/canceled**
  on the batch panel.
- Tests: `test_jobs.py` (cooperative cancel) + `test_batch.py` (cancel-before-run, mid-batch cancel
  stops the queue, retry re-runs a repaired project). **81 GPU-independent tests pass.** Assets `?v=16`.

## GPU uptime, dated transcript names, per-item + per-file batch controls (2026-06-19)

- **Prefetch (depth 1)** — `extract.extract_ahead` (a generic prefetch iterator) overlaps the next
  file's ffmpeg extraction with the current file's GPU transcription, removing the per-file GPU idle
  (~¼ of wall time on the 56 GB / 4-track file). Drives both `pipeline.run_merge_pipeline` (single
  job + multi-file merge) and the rewritten `run_per_file_pipeline` (per-file batches). Each file
  extracts into its own temp subdir (so prefetched WAVs don't collide); cleanup via `shutil.rmtree`;
  cancel stops prefetch promptly via a threaded `is_cancelled`.
- **Tunable batch size** — `transcribe.BATCH_SIZE = int(os.environ.get("WHISPERMERGER_BATCH_SIZE", "16"))`,
  used as `transcribe_track`'s default. `Startup.bat` ships **24** (headroom on a 12 GB 3080 Ti whose
  desktop + OBS already hold ~1.5 GB). Not part of the cache key — output is identical.
- **Self-describing exports** — `batch.transcript_basename` names every exported transcript
  `<base> Transcript (M-D-YYYY h-mm AMPM)` (e.g. `Project 1 Transcript (6-19-2026 2-02 AM)`; dash not
  colon, so it's a legal Windows name), for batch auto-exports and as the pre-filled default in the
  Transcribe/Edit Export boxes (`transcriptName`/`exportBase` in `app.js`). All formats of one
  file/project share a single timestamp.
- **Duration-weighted progress** — every bar is weighted by **audio length × track count** instead of
  counting tracks/files, so a long recording moves it more than a short clip. `BatchStatus.progress`
  weights projects by `ProjectStatus.weight`; per-file projects weight files by `FileStatus.weight`
  (`_recalc_project`); `run_merge_pipeline` reports by track-seconds. Track message clarified to
  "track N/total". Captions under the Transcribe + Label bars explain what the % counts.
- **Per-project cancel/restart** — `cancel_project` cancels one project while the queue continues
  (vs whole-batch cancel). The batch worker drains a lock-guarded `rerun_pending` queue, so a Restart
  requested mid-batch re-runs the project when the GPU frees up (single worker, single GPU). New
  `POST /api/batch/{id}/project/{idx}/cancel`; per-project Restart reuses `/retry {projects:[idx]}`.
- **Per-file checklist** — a per-file project tracks each source file as its own row
  (`ProjectStatus.files: list[FileStatus]`, populated at `create`): status chip, own progress bar,
  "track 2/4 · Speaker" message, output dir, elapsed. `cancel_file` skips one file (the rest
  continue); `restart_file` re-queues one **finished** file to run when free (single-pass driver +
  the batch re-run queue avoid double-runs/races). Per-file error isolation — a bad file fails its
  own row, the project keeps going. New `POST /api/batch/{id}/project/{pidx}/file/{fidx}/cancel` and
  `.../restart`; `_batch_out` exposes `files[]`.
- **UI** — Batch panel redesigned: a plain-language summary + status-count breakdown, a card per
  project (status chip, own bar, file counter, elapsed), per-item Cancel/Restart, and for per-file
  projects a scrollable **row per file** each with its own status + Cancel/Restart.
- Tests: `test_extract.py` (extract_ahead order / prefetch-overlap / error-surfacing / cancel),
  `test_pipeline.py` (temp-dir cleanup), `test_batch.py` (per-project + per-file cancel/restart,
  restart-while-running queues, duration-weighted progress, transcript naming), `test_api.py`
  (per-project cancel endpoint + per-file in the flow). **98 GPU-independent tests pass.** Assets
  `?v=19` (js) / `?v=16` (css).

## Per-file timer accuracy, scroll retention, track badge (2026-06-19)

- **Per-file timer now counts extraction** — `extract.extract_ahead` gained an `on_wait(item)`
  callback fired the instant an item becomes the consumer's active work (right before we block on
  its extraction). `run_per_file_pipeline` forwards it; `batch._process_project.on_wait` stamps
  `FileStatus.started_at` (status→RUNNING, "extracting audio") there. Result: per-file elapsed /
  "done in Xs" includes the ffmpeg extraction, not just transcription. It's prefetch-aware — only
  extraction **not** hidden behind the previous file's transcription is counted, so per-file times
  stay non-overlapping and sum to wall clock. Fixes the cache-hit re-run case where every file read
  "done in 0s" while the project clock kept climbing (the extraction wasn't attributed anywhere).
- **Per-file track N/M badge** — `FileStatus` gained `track`/`tracks` (set in `on_track`, primed in
  `on_wait`/`on_start` from the selected-track count), exposed via `FileStatusOut` + `_batch_out`.
  The frontend renders a small "track 2/5" pill on a running multi-track file row, so track progress
  is legible before the bar moves (single-track files show nothing — there's nothing to show).
- **File list scroll — in-place updates (no more snap, no more freeze)** — `renderBatchStatus`
  used to wipe and rebuild the whole panel every poll (~1 s). The first pass added a
  snapshot/restore of `.batch-files` `scrollTop` to stop the **snap-to-top**, but rebuilding the
  scroll container each tick still **froze** trackpad momentum and dropped wheel events mid-scroll.
  Now it reconciles in place: a structure signature (`batchId | ended | per-project file counts`)
  gates a full rebuild; while that's stable (the normal case during a run) it updates the existing
  nodes in place — `updateBatchItem` refreshes a card's non-scrolling head/bar/detail but keeps the
  `.batch-files` node alive, and `updateBatchFile` mutates each row's chip/badge/message/bar/elapsed
  by `data-fidx` without replacing the row. The scroll container is never torn out, so momentum and
  wheel input are untouched. Build/update share extracted helpers (`projectHead`/`projectDetail`/
  `projectActionBtn`, `fileActionBtn`/`fileDetailText`); action buttons carry `data-kind` so they're
  swapped only when a row crosses active↔finished.
- **Clear-cache button** — a header **Clear cache** button (next to Shutdown) wipes the WhisperX
  output cache so the next run transcribes cold, for benchmarking real GPU time. `Cache.clear()`
  deletes every `*.json` entry and returns the count; `POST /api/cache/clear` exposes it; the button
  confirms first and shows "Cleared N".
- Tests: the touched modules (`test_extract`, `test_pipeline`, `test_batch`) stay green; new
  `on_wait` is optional so existing callers/tests are unaffected; `test_cache` covers `clear()`.
  Assets `?v=21` (js) / `?v=17` (css).

## Performance instrumentation + read-bound vs GPU-bound diagnosis (2026-06-20)

Context: a full batch ran *slower* after the prefetch/batch-size changes (~900 vs ~700 min). In this
case the source files were large (80–100 GB) on a slow medium, so the working theory is the batch is
**read-bound** (ffmpeg streams ~the whole file to pull the audio; the GPU then idles waiting). The
instrumentation is medium-agnostic — it measures the read/extract wait vs GPU time regardless of where
files live (local SSD, external HDD, network share). Rather than keep guessing, added measurement.
Language in code/docs kept general ("read-bound", "reading the source") rather than NAS-specific.

- **Batch size default reverted to 16** — `Startup.bat` shipped `WHISPERMERGER_BATCH_SIZE=24`, which
  spilled ~1–2 GB into Windows "shared GPU memory" (system RAM over PCIe, ~36× slower than VRAM) on
  the 12 GB 3080 Ti. Back to 16; comment warns to keep it there unless Task Manager shows 0 shared-
  memory use. (Likely a contributor, not the whole regression — see read-bound note.)
- **Per-file extract-vs-transcribe timing — in the UI and the console** — `FileStatus.extract_seconds`
  (set in `on_start` = `on_start − on_wait`, the extraction wait while the GPU is idle) is exposed via
  `FileStatusOut`. The Batch UI renders, per file row, `read 5m 48s · gpu 2m 34s` (`fileTiming`), and
  the overall stats tally it live: `1h 12m reading/extracting · 38m on GPU so far` — the bigger number
  is the bottleneck, no verbal tag needed. `batch.py on_done` still prints the same split (with a
  grep-able READ-/GPU-bound tag) to the console as a scrollback
  record (`_fmt_dur`). A run dominated by the read side means the source/read path is the wall; one
  dominated by GPU means the GPU (and batch size) actually matter.
- **Stats show what the % means** — `ProjectStatusOut` now exposes each project's `weight`
  (audio-seconds × tracks); the frontend sums it to render **done / total audio** next to the overall
  percentage (`fmtAudio`, hours-aware), so the bar has a concrete denominator instead of a bare %.
- **Prefetch confirmed correct (no code change)** — `extract_ahead` submits the next file's
  extraction *before* it yields the current file to the GPU, i.e. the instant the current extraction
  finishes — so a single read path runs back-to-back and is already saturated. Depth-1 is right for
  one read path: deeper prefetch would only split the shared bandwidth and pile up temp WAVs. GPU
  idle on huge source files is the read path being the bottleneck, not a scheduling gap. Assets
  `?v=23` (js) / `?v=18` (css).

## Pause & resume with GPU release + pause-aware timers (2026-06-20)

Lets a long batch (or a multi-file Transcribe & Merge job) be parked mid-run and the GPU handed back,
without corrupting the timing readouts.

- **Pause is cooperative, like cancel** — it lands at the next file/track boundary (the in-flight file
  finishes), then the worker parks. Same checkpoints that drive cancel now also gate pause: the
  `is_cancelled` callback became a `_gate` that, while paused, frees the GPU and blocks on a
  `threading.Event` until resume or cancel. No pipeline code changed — only what the gate does.
- **GPU is released on pause** — new `transcribe.unload_models()` clears the `lru_cache`'d ASR +
  align models, `gc.collect()`s, and `torch.cuda.empty_cache()`s, run **on the worker thread** that
  owns the GPU (so ctranslate2/torch free cleanly). The models reload lazily on the next transcribe
  after resume (the chosen trade: VRAM back to ~0 while paused, at the cost of a one-time reload).
  Wired as an injected `on_pause` hook (`batch.BatchStatus.on_pause`, `jobs.run(..., on_pause=)`), so
  `batch.py`/`jobs.py` stay GPU-free and the tests need no card.
- **Pause-aware timers** — every timed row (`BatchStatus`/`ProjectStatus`/`FileStatus`/`Job`) gained
  `paused_total` + `pause_started`; a shared `_live_elapsed` subtracts paused time so `elapsed`,
  "done in Xs", and the read-vs-GPU split all exclude time spent parked (the clocks visibly **freeze**
  while paused and resume on continue). `extract_seconds` subtracts `paused_total` too, so a pause
  mid-extraction doesn't inflate the read figure. The model-reload on resume counts as real time
  (it's active work, not idle).
- **Two-stage status** — `gpu_freed` (worker actually parked) is reported alongside `paused`, so the
  UI distinguishes "Pausing… (current file finishes)" from "Paused · GPU released" honestly.
- **Endpoints** — `POST /api/batch/{id}/pause` · `/resume`, and `POST /api/pause/{job_id}` ·
  `/resume/{job_id}`. `BatchStatusOut`/`ProgressResponse` expose `paused` + `gpu_freed`. Cancelling a
  paused run wakes the parked worker so it unwinds (every gate call short-circuits on `cancelled`).
- **UI** — a **Pause/Resume** toggle by **Cancel batch** and by the Transcribe **Cancel** button; it
  reflects the server's `paused` each poll (survives a refresh), and the batch header shows the
  paused stage line.
- Tests: `test_jobs.py` + `test_batch.py` each gain pause→resume (clock frozen, run completes, pause
  banked out of `elapsed`) and cancel-while-paused (wakes + unwinds). **103 GPU-independent tests
  pass.** Assets `?v=25` (js) / `?v=18` (css).

## Extraction timing → real pre-GPU time, not idle wait (2026-06-20)

Follow-up to the read-vs-GPU split above. The per-file "extraction time" was the GPU **idle wait**
(`on_start − on_wait`): when prefetch hid a file's audio behind the previous file's GPU run, that
wait was ~0, so a fully pipelined file showed **0s extraction** even on a cold (cache-cleared) run.
Correct by the old definition but misleading — it read like extraction was free or the cache clear
had failed (it hadn't; the cache only covers transcription, ffmpeg always runs, and a *cold* run is
exactly what hides the next extraction).

- **`extract_seconds` now measures the real read + ffmpeg pass**, timed with `time.perf_counter()`
  around the extraction itself in `run_per_file_pipeline.extract_one`, so it holds even when the work
  ran prefetched in the background. Passed through to `on_start(i, src, extract_s)`; `extract_ahead`
  stays a generic, untimed prefetch iterator (no signature change).
- **Per-file `elapsed` is now the GPU clock** — `on_start` (not `on_wait`) stamps `started_at`, so a
  row's `elapsed` / "done in Xs" is pure transcription and the real read figure sits beside it. UI +
  console use the two directly (`gpu = elapsed`), dropping the old `elapsed − extract_seconds`
  derivation. Console: `read + extracted in X (pre-GPU) · transcribed Y on GPU · READ/GPU-bound`.
- **Pause-immune** — timed around the ffmpeg call (which runs uninterrupted), so the read figure no
  longer subtracts `paused_total`; a pause can only land between work units, never inside a measured
  extraction.
- Tests: `test_batch.py` gains a prefetch case — a hidden file must still record real extraction, not
  ~0. **104 GPU-independent tests pass.** Assets `?v=26` (js).

## Second-pass easy wins (2026-11-09, #27–#35)

- **#27 Speaker column** sizes to the longest name (96–260 px, `--spk-ch` on the table); longer
  names show in full on hover.
- **#28 Per-line playback** on multi-track files: a switch by the waveform plays this speaker's
  track or all tracks mixed (`/api/clip` without `track`). Kept for the browser session.
- **#29 Clear cache** shows "N · size" from `GET /api/cache/stats` (`Cache.stats()`); greyed out
  when empty; refreshed on load, after clearing, on run/batch finish and tab switch.
- **#30 Transcribe & Merge progress** is one compact line under the run button; collapses to
  "Done in 4m 12s · 2 tracks". Canceled/failed runs stay expanded.
- **#31 Input list rename after a run** renames the speaker in the transcript like Rename
  globally, plus `merged.sources`, the Include toggle and the speaker→track map.
- **#32 Export** takes several formats at once (comma list → one zip, standard `zipfile`) and shows
  a summary dialog first (formats, limits, padding, edited-line timing, hidden speakers, edited
  count) with "Don't show again".
- **#33 Per-chunk progress:** `transcribe._transcribe_counted` wraps the WhisperX pipeline
  instance's `__call__` for one call and reports chunks done through `chunk_progress` (per
  thread). The bar moves within a track (transcription 90% of its weight, word timing the rest);
  updates arrive per batch of up to 16 chunks. Per-file batch rows move the same way
  (`run_per_file_pipeline(on_chunk=...)`).
- **#34 Batch → editor:** finished files/projects keep their transcript and source path
  (`FileStatus.result`, `ProjectStatus.result`; in memory until restart). "Open in editor" loads it
  into Edit Transcript with playback per track. Routes `/api/batch/{id}/project/{p}[/file/{f}]/result`.
  Follow-up: a project row's buttons (Cancel / Restart / Open in editor) were rebuilt on every
  0.8 s poll, so a click whose press and release straddled a poll was lost. The project head now
  updates in place like file rows do; a button is swapped only when its job changes.
- **#35 Models tab:** Speech / Word-timing sub-tabs with counts, grouped one-line speech rows, a
  word-timing grid by language. 3308 px of scrolling → about 1100 px per tab.

## CLI + MCP server (2026-11-09, #36)

- **One action list** (`backend/actions.py`): plain functions over the HTTP API (stdlib
  `urllib`), whose signatures and docstrings define both interfaces. They start the server if it
  isn't running (detached, log in `backend/tmp/server.log`) and leave it up so models stay loaded.
- **CLI** (`backend/cli.py`, `whispermerger.bat`): one argparse command per action, built from the
  signature (paths positional, lists comma-separated, `--no-x` for on-by-default flags); progress to
  stderr, `--json` for machine-readable output, UTF-8 output even through a pipe.
- **MCP server** (`backend/mcp_server.py`, stdio, `mcp==2.3.0`, where FastMCP is now `MCPServer`):
  the same actions as tools, read-only/destructive annotations, and refusals passed to the model as
  `ToolError` (the SDK hides any other exception's message).
- **Server additions:** `PUT /api/result/{job_id}` stores an edited transcript (`edit_lines`);
  `/api/add-folder` takes an optional `path` (the UI still opens the picker).
- Tested end to end: CLI autostart → transcribe two tracks with names → rename / edit a line →
  export SRT/TXT; folder batch with a subfolder; the same flow over a real MCP stdio client.
  `tests/test_actions.py` runs the actions and CLI against the app in-process.

## Second model in a process ran ~10x slower: fixed (2026-11-09)

Found while timing #23 on a free GPU: whichever speech model loaded *second* in a process ran about
10x slower (large-v3: 7 s first, 60-67 s second, same 7 min clip). CTranslate2 keeps a model's VRAM
after Python drops it, until `unload_model()`; `unload_models()` only cleared the Python caches.
This hit the Portugal punctuation merge (its large-v3 pass), resume after Pause, auto-detect
switching to CAMOES, and every Benchmark model after the first (so Benchmark times were wrong).

`transcribe._release_loaded()` now calls `unload_model()` on the pipelines `_load_model` built, when
the cache replaces its model and in `unload_models()`. Jobs share no GPU lock, so it only does so
when no other thread is mid-transcription (`gpu_user()` marks `transcribe_track` and Benchmark);
otherwise it drops the references as before. Portugal merge on that clip: 79.3 s → 28.5 s.

## Batch size fits the free GPU memory (2026-11-09, #37)

`WHISPERMERGER_BATCH_SIZE` (16) is now a maximum. Right after a speech model loads,
`transcribe._fit_batch` reads free VRAM once and caps the batch at `(free - 0.5 GB) / 0.4 GB`
(`pipe.fit_batch`, applied in `_transcribe_counted`). Measured on the 3080 Ti: on top of the loaded
model a batch needs at most ~0.4 GB per item. With 6.5 GB held by another
process (like LM Studio), 35.7 min of speech: auto batch 6 = 46.9 s vs fixed 16 = 81.2 s (spilling).
**Margin raised 0.3 → 0.9 GB (2026-11-09):** the speech model takes ~0.6 GB more on its first run,
after free VRAM is read. At 0.3 a free 12 GB card picked 16, and with the word-timing model in,
16 spilled 0.4–0.7 GB and ran ~2× slower (CAMÕES and large-v3 alike); a free 12 GB card now picks 14.
Two 7 min pt-PT tracks: 30.7 / 30.6 s (was 32.3 / 38.8), no spill, steady run to run.
Read once per load because CTranslate2 keeps memory it has used, so later readings would shrink.

**Clean sweep (2026-11-09):** large-v3, 20 min English clip, each size in a fresh process after a
warm-up, nothing else on the GPU but Windows + NVIDIA Broadcast (0.8 GB idle). Spill = Windows
shared GPU memory above idle.

| Batch | Time | Speed | Peak VRAM | Spill |
|---|---|---|---|---|
| 4 | 26.3 s | 46× | 6.7 GB | — |
| 8 | 18.1 s | 66× | 7.5 GB | — |
| 12 | 15.9 s | 76× | 8.8 GB | — |
| **16** | **15.1 s** | **80×** | 10.5 GB | — |
| 20 | 20.3 s | 59× | 11.5 GB | 0.3 GB |
| 24 | 39.2 s | 31× | 11.8 GB | 1.4 GB |

16 is the fastest size on a 12 GB card; 12→16 gains only 5%, while any spill costs far more (20 is
25% slower, 24 is 2.6× slower). Transcripts are the same at every size (3066–3073 words). So the cap
of 16 plus fitting to free memory is right, and a batch-size Benchmark isn't worth building: the
only risk is spilling, which the fit already prevents. ~0.32 GB per batch item measured here, under
the 0.37 GB the fit assumes (a little headroom).

## "Cannot copy out of meta tensor" on a first run in a new language (2026-11-09)

Reported: a CLI run in Korean failed right after the word-timing model downloaded; the same command
again worked. Not reproducible on demand (a fresh Korean download then load worked first time), so
the fix covers the confirmed hazard plus a safety net:

- **Cause class, confirmed:** transformers 4.57 always builds models empty on the meta device by
  patching `torch.nn.Module.register_parameter` for the whole process. While a word-timing model
  loaded, 75 modules built on another thread came out empty; two overlapping patches can also
  restore each other out of order. The server builds models on several threads.
- **Fix:** `transcribe.MODEL_LOAD_LOCK` serializes every model build (`_load_model`, `_load_align`).
- **Safety net:** `_load_align` reloads once on a "meta tensor" error (a second load worked when it
  happened) and logs it with the traceback, as `[models] ...`.
- Failed jobs and batch projects now print their traceback to the server console (they only kept
  the message before, so this one couldn't be traced).
- A finished model download reports "Downloaded <model> · loading it"; it used to stop at the last
  99% tick, which looked like the load started before the download finished. It didn't.

Repro note: move the model's folder out of `~/.cache/huggingface/hub` (Korean:
`models--kresnik--wav2vec2-large-xlsr-korean`), then `whispermerger transcribe <uncached Korean clip> --wait`.

## Two transcriptions at once crashed one of them: fixed (2026-11-09)

Found while testing batch buttons: a batch and a second run (another batch, Transcribe & Merge, or the
CLI/MCP) on the shared speech model at the same time failed with `'NoneType' object has no attribute
'sot_sequence'` then `'FasterWhisperPipeline' object has no attribute '__call__'`. WhisperX sets the
model's tokenizer for each run's language and clears it when the run ends, and the per-chunk progress
hook (#33) is put on the model per run, so one run pulled them from under the other.

- **Fix:** `transcribe.GPU_TURN`, one uncached transcription at a time process-wide; the second waits
  (its bar sits at 0% until then). Cache hits don't wait. The Benchmark takes the same turn and says
  "Waiting for the current transcription to finish" when it has to wait.
- Waiting runs don't count as GPU users, so the Portugal two-pass can still fully unload a model.
- Verified: a 2-project batch and a CLI job started together, twice; all finished, no errors.

## Language detection across the file, ask before a big download, word-timing model left the GPU full (2026-11-25, #38)

Reported: an English 2-hour stream that opens with music was guessed as Nynorsk (`nn`, 0.36) from
its first 30 s. That downloaded the 3.9 GB Nynorsk word-timing model mid-job, the GPU then sat at
11.7 of 12 GB, and every later file crawled until the server was restarted.

**Why the GPU stayed full (two causes, both measured).**

- *A leftover stack frame held the model.* `nltk` (imported by WhisperX's alignment module) keeps
  the error of an optional import that failed, and with it the whole call stack that first imported
  it. On a server started by the CLI / MCP with no browser open, that first import happens inside
  the first `transcribe_track` call, so that call's frame lived on with its locals: the run's audio
  and its word-timing model. `_load_align.cache_clear()` dropped the cache's reference, the frame
  kept its own, and the model stayed on the GPU until restart. (With the app open, the Models list
  imports it first, which is why this showed up through the CLI / MCP.) Fix: `transcribe_track`
  clears those locals in a `finally`.
- *A big model sat through the next file's batch.* The batch size leaves 1.3 GB for a word-timing
  model (`ALIGN_MODEL_GB`); the Norwegian ones are 3.9 GB, and the model stays loaded until the
  next file's alignment replaces it, which is after that file's transcription. Fix: a model
  bigger than that room is unloaded right after use (`_drop_align`).

Measured on the 3080 Ti (12 GB), 8 min English clip, large-v3, batch 14, nothing else on the card
(0.5 GB idle). The Nynorsk model was stood in for by the same architecture built from its config
with random weights (963 M parameters, same size on the GPU, no 3.9 GB download). First run in a
fresh process is the "Nynorsk" one, then the same English clip three times:

| | GPU used after the Nynorsk file | Next English files | Spill into system memory | GPU used after them |
|---|---|---|---|---|
| Before | 9.5 GB (4.0 GB of it the model) | 50.0 / 57.2 / 60.7 s | 1.5–1.9 GB each | 9.9 GB, never drops |
| After | 5.5 GB (model gone) | 15.3 / 14.3 / 14.3 s | none | 5.9 GB |

Not covered: with another app holding several GB, the speech model plus a 3.9 GB word-timing
model may not fit together, and that one file's alignment would spill (the files after it don't).

**Detection.** `transcribe.language_probs`: up to 4 spots spread over the file (the middle of each
quarter), 45 s searched at each with the silero VAD (`diarize.speech_spans`), up to 30 s of the
speech found goes to Whisper's language head, probabilities averaged. A spot with under 5 s of
speech doesn't vote; with no speech anywhere it falls back to the first 30 s. Cost is the same
whatever the length: about 2 s of VAD per file plus 4 encoder passes (0.5 s). On a clip
that opens with 45 s of tones and noise: first-30-s guess `en` at 0.29 (a coin toss), new `en` at 1.00.

**Languages I use** (`modelstore.my_languages`, Models tab, saved in `settings.json`, empty by
default; part of the cache key). `transcribe.choose_language`: a guess in the list is accepted; a
guess outside it needs 0.5 (`OUTSIDE_LIST_MIN`), else the best-scoring language of the list is
used. A Portuguese default (`auto_portuguese`) counts as using Portuguese. Empty list: the top
guess stands, as before. CLI / MCP: `list_models` shows the list, `set_my_languages` replaces it
(no languages = empty); unknown codes are refused by the same route the app uses.

**Ask before a big download.** Auto-detect, language not in the list, word-timing model over 1.5 GB
(`ASK_OVER_MB`) and not on disk: `transcribe_track` asks through a per-thread hook
(`transcribe.asking`) instead of downloading. The limit started at 1 GB and was raised the same
day: the usual word-timing models are about 1.26 GB, so nearly every language asked on first use;
now only the unusually big ones ask (the Norwegian ones, 3.9 GB). Answers: download it, continue without word timing
for this file, or use a different language. A language picked in a dropdown, or one in the list,
downloads as before. No hook (a script calling `transcribe_track` directly): no download, line
timings only.

- The wait reuses the pause machinery: `Job._park` / `batch._park` (the body of the old pause
  block), so the GPU is freed and the timers freeze; Cancel ends it (batch, project or the one
  file). The GPU turn isn't held while waiting, so other runs go ahead.
- One answer covers the rest of that file (a job, a batch file or a merged project); the next
  file asks again.
- App: a box under the progress line in Transcribe & Merge, Batch and Label Track (`showQuestion`).
  API: `question` in `/api/progress/{id}` and `/api/batch/{id}`, `POST /api/answer/{job}` and
  `/api/batch/{id}/answer`. CLI / MCP: new `answer` action; `get_progress` / `get_batch` carry the
  question and its choices and say "waiting for your answer"; a `--wait` returns with the question
  instead of hanging.

**Shown.** The progress line gets the choice once detection has run (`track 1/1 · Speaker 1 ·
English 97%`), the finished line too (`done in 18s · English 97%`), and results carry it per track
(`MergedOutput.languages`, in the JSON export).

Verified on the real GPU and a real server (its own port): the numbers above; detection on the
music-first clip and a 2 h 11 min file; and, with the detection result faked to `nn` 0.36 for
marked clips, the whole question flow from the CLI and in the app (a job and a batch): asked, GPU
released and clock stopped while waiting, a wrong language code refused, each answer except
"download" carried out, cancel while waiting. Not run for real: the 3.9 GB download itself (the
"download" answer goes through the same `ensure()` every other model download uses), and Label
Track's question box (same function as the other two tabs). (Both since run for real: see "CLI + MCP against a real server" below.)

## CLI + MCP parity with the app (2026-11-25)

Things the app could do that the CLI and MCP server could not. All are actions in
`backend/actions.py`, so each is a command and a tool.

- **Settings.** `set_auto_portuguese` (pt-PT / pt-BR / leave out for neither) and
  `set_pt_punctuation` (`--on`; without it, off). `list_models` shows both next to `my_languages`.
- **Export options.** `export` takes `min_duration` (shortest SRT subtitle, default 1.0 s) and
  `edited_timing` (`align` / `even` / `whole`, default `align`), the route's own names and values.
  `_export_options` takes them last with those defaults.
- **Transcript files.** `get_transcript` and `export` take an `.srt`, `.txt` or `.csv` as well as a
  job id or `.json`. The parsers (`_parse_srt`, `_parse_txt`, `_parse_csv` in `actions.py`,
  standard library only) follow `parseSrt` / `parseTxt` / `parseCsv` in `frontend/app.js`: the same
  `Name: ` speaker rule, `no_timing` for a TXT line without a timestamp, CSV columns by header name
  or in export order. Tested by round trip (what `export.py` writes reads back as the same lines)
  and compared against the JS parsers on the same awkward inputs. `export` refuses to write over
  the file it is reading (`export talk.srt` would otherwise replace it with a re-wrapped copy).
  `edit_lines` takes a job or a batch result, not a file.
- **Models.** `cancel_download`, `remove_model` (marked destructive), `run_benchmark` (`--wait`, or
  follow the job with `get_progress`) and `get_benchmark`; `list_models` adds `disk_free_mb` and
  each model's `partial` (a stopped download left on disk).
- **Run control.** `cancel`, `pause` and `resume` take a job id or a batch id (a job's route is
  tried first, the batch's on a 404, as `answer` already did; shared in `_run_post`). `cancel` also
  takes `project` / `file` for one part of a batch. `retry_batch` re-runs every failed or canceled
  project, chosen `projects`, or one finished `file`. `get_batch` now gives the `batch_id`, each
  item's `project` and `file` number (from 1; these are what every other action takes) and
  `paused`; `get_progress` shows `paused` too.
- **Batch.** `run_batch` takes `names`, `tracks`, every export option of `export` and `projects`:
  more projects for the same run, each an object with `paths` and whatever differs from the
  call's own settings. Objects were chosen because a project has a dozen settings of its own and
  no flat list holds that; MCP clients pass them as they are, the CLI reads JSON text or a `.json`
  file (the one new argument shape in `cli.py`; a required list such as `paths` may now be empty,
  so `run-batch --projects batch.json` works). Each project is sent with its own export and merge
  options, built from the action's parameters by name, so an option added to `_export_options` and
  to the signature reaches batches without more code.
- **Batch results.** `get_transcript`, `edit_lines` and `export` take `project` / `file` with a
  batch id. New on the server: `PUT /api/batch/{id}/project/{p}/result` and
  `.../file/{f}/result` store an edited batch transcript in memory, like `PUT /api/result/{job}`;
  the files the batch wrote are not rewritten.
- **Merge options.** `transcribe` and `run_batch` (and a batch project) take `detect_conflicts`,
  `backchannel_render`, `backchannel_max_words`, `overlap_min_seconds`, `bleed_window_seconds`,
  `bleed_similarity` and `bleed_prefer_ratio`: `MergeOptionsIn`'s own names, passed through.
- **Server.** `GET /api/benchmark/{job}` also returns the job's `status`. A model id outside the
  catalog is now unknown (404) for download, cancel, remove and use: `whisper:<anything>` used to
  resolve to a model-cache folder of that name, which `DELETE /api/models/...` would then delete,
  whoever it belonged to.
- **Fixed on the way.** `whispermerger --help` crashed on the `%` in a command's description
  (argparse reads it as a format); the CLI now escapes it, and a test prints every command's help.

Not run for real: the Benchmark, a model download and a removal (tests stub them; the actions call
the same routes the Models tab does). Run control, several projects, batch results and merge
options are tested in-process only (the app through its test client, transcription stubbed), not
against a running server. (All since run against a real server: see "CLI + MCP against a real server" below.)

## Option to skip word timing (2026-11-25, #39)

Word timing (alignment) takes as long as the transcription or longer, and a caller that only
needs the text and rough line times had no way to turn it off. Now `word_timing` (default on, so
nothing changes unless asked): `transcribe_track(..., word_timing=False)`, `word_timing` on
`POST /api/transcribe` and `POST /api/batch` (the batch's, and each project's own over it), and
`transcribe` / `run_batch` in `actions.py`, so `--no-word-timing` on the CLI and `word_timing:
false` over MCP, also inside a `run_batch` project. The app got its tick box afterwards (see "In
the app" below).

- **Off:** the existing "keep line timings only" branch. No word-timing model is ensured, loaded
  or downloaded, and the #38 question about a big download is not asked.
- **Cache:** its own tag, `no-words`, in place of `wav2vec2`. The existing `none` tag means "this
  chosen language has no word-timing model"; reusing it would have mixed the two kinds of run for
  such a language. With a separate tag a run with word timing and one without never get each
  other's saved result.
- **Shown:** the track's output has `"word_timing": False`, kept per track in
  `MergedOutput.languages` (new field, default true, so it is in the JSON export), and
  `language_note()` adds it to the finished line and the `language` of a job or batch file:
  `English 97% (no word timing)`, or `English (no word timing)` when the language was chosen. The
  "skip" answer from #38 is marked the same way. A language that simply has no word-timing model
  is not marked (unchanged).
- **Batch:** `ProjectInput.transcribe_fn` (None = the batch's) carries a project's own setting.

Measured on the real GPU (3080 Ti, large-v3, batch 14, 0.9 GB in use by other apps, none of them
computing), own server, a 10 min 9 s file of synthesized English speech, models already loaded,
language auto-detected: **10.1 s without word timing, 20.9 s with it**. Same 1,665 words in the
same order. Lines differ: 28 lines averaging 21.8 s without (one per stretch of speech, as Whisper
returns them) against 95 averaging 5.8 s with (alignment also splits into sentences), so a caller
that wants short paragraphs gets coarser stamps. The server console showed "no word-timing model
used ... keeping line timings only" for the runs without it; GPU memory in use was not a clear
signal either way (5.3 to 5.6 GB in both cases). Running each again returned its own saved result.

**In the app:** a **Word timing** tick box, on by default, with a one-line note of what off costs.
Transcribe & Merge has it under Merge options, above Language (`#opt-word-timing`); Batch has it
under Output with the other per-project defaults (`#batch-word-timing`), and each project's
"Override output settings" row has its own, which starts as the batch's and is sent as that
project's `word_timing` (a project that does not override sends none, so it gets the batch's).
Both boxes are remembered per tab in `wm-settings`, like the Language choice. No new result UI:
the finished line and a batch file's message already carry "(no word timing)". Checked in a
browser on an own server with a 25 s clip of synthesized English: both boxes survive a reload,
unticked they send `word_timing: false`, the Transcribe & Merge run ended "Done in 9s · 1 track ·
English 100% (no word timing)" with the whole clip as one line, and in a two-project batch the
project on the batch's setting ended "English 100% (no word timing)" while the one overriding it
back on ended "English 100%".

Not covered: Label Track (`/api/diarize`) has no such option, on purpose. It transcribes each
speaker's labeled regions joined into one clip and puts every line back on the timeline from its
word times. Without them a line is a stretch of up to about 30 s of that joined clip, so it can
start in one region and end several regions later, across the other speakers' turns: lines come
out in the wrong order and get flagged as overlaps. Passing the option through is a few lines;
making the result usable needs lines cut at region ends, which is not. (Read from the code, not
run.)

## CLI + MCP against a real server, and rough edges (2026-11-26)

Everything from the two sections above, run for real: an own server on its own port, driven by
the CLI and by an MCP client over stdio (the `mcp` package's stdio client starting
`mcp_server.py`). Test audio: two-speaker dialogues spoken by the Windows voices into a
two-track `.mkv` (one voice per track), 1 min 51 s and 14 min 45 s. GPU: large-v3, with other
apps holding about 1.1 GB; no timings are claimed here.

**Ran, both from the CLI and over MCP, and passed:** `transcribe --wait` on the two-track file,
then `get-transcript`, `edit-lines` (rename, line text, delete, a wrong line refused) and `export`
(all four formats, and reading an exported `.txt` back); `pause` (GPU released, clock stopped),
`resume` and `cancel` on a running job and on a running batch, `cancel --project`; `run-batch`
with three projects (one with word timing off, ending "(no word timing)", one per-file, one
merged), `get-batch`, `retry-batch` (all canceled, one file, chosen projects), batch results by
`--project` / `--file`; `run-benchmark` / `get-benchmark` (17 models); `list-models`,
`download-model`, `cancel-download` (leaves a partial), resuming it, `remove-model` (full and
partial), `use-model` and back; `set-my-languages`, `set-auto-portuguese`, `set-pt-punctuation`
(wrong values refused), `cache-stats`, `clear-cache` (on that server's own cache folder), `status`.

**#38 for real:** with only the language guess faked to Norwegian Nynorsk / Norwegian at 36% (no
Norwegian voice to speak a clip), a job asked before the 3.9 GB word-timing download and `answer
... download` from the CLI downloaded it and finished; a batch asked the same over MCP and the
`answer` tool did the same with the other Norwegian model. In a browser, Label Track showed the
same question box under its progress line; "Use this language instead" carried on in English and
the box went away. (The file was added by answering the page's file-picker request with the
add-path route; a page can't drive the server's own file dialog.) Both big models were removed
afterwards.

**Fixed:**

- A batch id given to `get-transcript`, `export` or `edit-lines` without `project` / `file`
  answered "404: unknown job". It now says it is a batch id and lists the batch's projects and
  their files by number (`_result` in `actions.py`); nothing is picked for you.
- A wrong `backchannel_render` came back as the server's raw 422. `transcribe` and `run-batch`
  (a batch project too) now answer "backchannel_render is one of parenthetical, separate, skip"
  before anything starts (`_merge_options`).
- Retrying a project that is still running (or queued) was accepted, and reset it under the run.
  `POST /api/batch/{id}/retry` refuses it with 409 and the project's number and name, so the app
  gets the same answer as the CLI and MCP.
- After a whole-batch cancel, the files of a per-file project that hadn't started stayed
  "queued" in a canceled batch (the project itself said canceled). They now end canceled too
  (`_mark_canceled` in `batch.py`, also used when a project is canceled before it runs).
- The Word timing note (Transcribe & Merge, Batch) wrapped below about 1280 px. At note size
  (12 px) it now sits beside the box down to about 1100 px and on one line of its own under it
  down to about 985 px.

**Seen, left as they are** (the first four are fixed since, see "Four oddities from the live
test" below)**:** a project canceled during its last track still writes that file, so
it ends "canceled" with its one file "done", and `get-batch` (which lists file rows) shows done;
`run-batch` counts source files where `get-batch` counts rows (a merged project of two files is 1
there); pausing a job that is waiting for the GPU takes effect once it gets it, so its clock runs
until then; `add-files` prints each file's tracks as a raw list. `edit-lines` still takes only a
job or batch result, not a transcript file: where the edited copy would go is a decision of its
own (an `--out` file, a copy next to the source, or a result held on the server).

## edit-lines on a transcript file (2026-11-26)

`edit_lines` (`edit-lines`) takes a transcript file as well as a job or batch id: a WhisperMerger
`.json`, or an `.srt`, `.txt` or `.csv`, read by the same `_merged` as `get_transcript` and
`export`, then edited by the same code as a job (rename, a line's text / speaker / start / end,
delete, the `edited` flag on a changed text). Its first parameter is now `source`, like theirs
(it was `job_id`). The edited copy goes to `out`, a `.json` file the caller names, written
straight from the transcript with `json.dumps`, so nothing is lost: flags, word timings and
anything else in a `.json`. It needs no server. Refused with a plain message, before any edit:
a file without `out`, `out` that is the source (`samefile`, as `export` checks, so a path that
differs only in case counts on Windows), `out` not ending in `.json`, and `out` with a job or
batch id. An existing file at `out` is overwritten, as `export` does.

The files it writes carry a top-level `"edit_lines_copy": true`, and such a file may be edited in
place (`out` left out or the same file), so edits in a row are one call each. The mark is safe
everywhere else: `_merged` reads `.json` with `json.loads` and keeps every key, and the server's
`MergedOutput` ignores keys it doesn't know, so `export` of a marked file writes the same bytes
as of the same file without the mark, and its own `.json` export comes out without it.
`get_transcript --full` shows the file as it is, mark included.

## Four oddities from the live test (2026-11-26)

- **Canceled during the last track.** A file of a one-transcript-per-file project whose cancel
  (of the file, its project or the whole batch) arrived while its last track was being
  transcribed was still merged and written, so it showed "done" in a project that said
  "canceled". It now ends "canceled" with nothing written, as a cancel between tracks always did,
  and as a joined project already behaved. The track that was running still finishes and is
  cached, so a retry is immediate. `on_done` in `batch.py` asks `cancel_pending`, the cancel
  question without the pause stop (`is_cancelled` parks on a pause, and a pause during the last
  track must still write the file first).
- **File counts.** `get_batch` counted its items, `run_batch` source files, so two files joined
  into one transcript were 2 in one and 1 in the other. Both count source files now: a joined
  project is one item that counts as every file in it, in `files` and in `done`. The server
  reports each project's `sources` (`ProjectStatus`, `ProjectStatusOut`); the app doesn't use it.
- **Pause while waiting for the GPU.** A run blocked on `GPU_TURN` had no checkpoint, so a pause
  (or a cancel) landed only once it got the GPU: a job's clock ran until then and it then parked
  holding the turn; a batch file transcribed its whole track first. `_gpu_turn` now waits in
  half-second steps and reports "waiting" on each one, and that report is the checkpoint: a job
  parks or stops in `_set_progress`, a joined batch project in its guard, and a per-file batch
  project in `on_chunk` (new check). Nothing is held while it waits. `unload_models` does nothing
  for a thread in that wait (`_chunk.waiting`): the models on the GPU are the running run's, and
  dropping them under it is what made the next load slow. So "paused · GPU released" there means
  the run has let go, not that the GPU is empty.
- **`add-files` on the command line.** `_human` in `cli.py` printed a list inside a list item as
  Python text (`tracks=[{'track': 1, ...}]`). An item's plain values now go on its line and any
  list or object in it below, indented: one line per track. `--json` and MCP are unchanged.

Tests: 209 pass, 4 new (each fails on the old code). `add-files` was run through the real CLI on
a server of its own; the other three were not run on the real GPU (about 8.9 of 12 GB was in use
by other apps at the time).

## One spelling for "canceled" (2026-11-27)

A single transcription job reported `cancelled` (two l) as its status and message, while
batches, model downloads and all the app's wording said `canceled`. A caller checking for a
canceled run had to know both. Everything a user or caller sees now says `canceled`: the job
status (`JobStatus.CANCELED`, renamed to match `Status.CANCELED` in `batch.py`) and message, the
`get_progress` help text (CLI and MCP), the check in `_wait_job` and the app's checks in `app.js`
and `labeltrack.js`. Internal names (`is_cancelled`, `batch.cancelled`, `Cancelled`,
`JobCancelled`) are unchanged. There is no fallback for the old spelling.

Tests: 210 pass, 1 new: `transcribe --wait` still exits with code 1 when its job is canceled.

## A pause waits for the end of the track (2026-11-27)

A single Transcribe job, a batch project joined into one transcript, and the Benchmark checked for
a pause while they held `GPU_TURN`, the one-run-at-a-time lock. A pause asked for mid-track then
parked inside the lock, so every other run said "waiting for another run to finish" for the whole
pause. Parking also unloaded the model under the track: when audio was left in that track, the job
failed after resume with "The model for this translator was unloaded".

Now a pause waits for the end of the track it is on (its transcription and word timing), as a
batch with one transcript per file and Label Track already did. Until then the run shows
"pausing" (`paused` true, `gpu_freed` false); then it parks outside the lock, the GPU is released,
the clock stops and other runs can take their turn. Cancel is unchanged: it still stops a track
part-way, at the next chunk, also while the run is pausing.

- `_gpu_turn` (`transcribe.py`) marks the thread while it holds the turn, and `holding_gpu_turn()`
  says so. `Job._set_progress` and `_gate` in `batch.py` don't park while it is true. That covers
  every check inside the turn: the per-chunk progress, a joined project's guard, and the progress
  of a model download mid-run (a pause there lands after the track; Cancel still stops the download).
- The Benchmark takes the turn once per model, through `_gpu_turn`, so its check between models is
  outside the lock. Another run can now take its turn between two models; each model still
  starts cold, so the timings compare.
- Waiting for the GPU is now a checkpoint for Label Track and the Benchmark too: they report
  "waiting" every half second like the other runs, so a pause or a cancel there takes effect at
  once instead of after they get the GPU. Label Track's line also shows what Auto-detect chose
  ("transcribing A (1/2) · English 97%").
- With Auto-detect, a pause asked for while the language is being detected lands before that
  track's transcription starts (a job, a joined project, Label Track): detection and
  transcription are two turns, and the report of the chosen language between them is outside the
  lock. Nothing of the track is lost; its model loads again on resume.
- Wording: `pause` and the "paused" line in `get_progress` / `get_batch` (CLI and MCP), the pause
  routes, the app's "Pausing… (current track finishes)" and the README now say the track finishes
  first, not the file.

Tests: 225 pass, 8 new, and the 4 that were expected to fail now pass with their marks removed.
Five tests written for "another run takes its turn the moment the job is paused mid-track" (a job,
a joined batch project, the Benchmark, and the last-batch and second-batch cases) now check the
chosen rule instead: the whole track runs before the pause lands, the lock is free while parked,
and the result is complete after resume. New: cancel still stops a job and a joined project
mid-track while pausing; a pause during a download mid-run lands after the track (job and batch);
cancel stops that download; Label Track and the Benchmark pause and cancel at once while waiting
for the GPU; "pausing" is seen during the track's second batch, then "paused · GPU released". Not
yet run on the real GPU.

## A canceled row says how far it got (2026-11-27)

A canceled file in a batch said only "canceled", even when some of its tracks had already been
transcribed. Nothing of it is written either way, but those tracks are in the cache (a server
run always passes it, and a track's result is saved as soon as that track finishes), so a retry
reuses them. The status stays "canceled"; the message now says how far it got, for example
`canceled · 1 of 2 tracks transcribed, kept for a retry`. A row canceled before any of its tracks
finished (before it started, or while waiting for the GPU on its first track) still says
`canceled`.

- Batch: `FileStatus.transcribed` and `ProjectStatus.transcribed` count the tracks whose
  transcription returned in this run (a wrapper around the project's transcribe function in
  `_process_project`); `ProjectStatus.tracks` is a joined project's track count. `_canceled` gives
  the message, used by `on_skip` (a file canceled between tracks, during its last track, or while
  waiting) and `_mark_canceled` (a project or whole-batch cancel). A file row and a joined project
  say it; a one-transcript-per-file project's own row still says "canceled". A retry resets the count.
- A single Transcribe & Merge job says the same: the `/api/transcribe` route counts its finished
  tracks and the cancel carries the message (`canceled_message` in `jobs.py`, shared with the
  batch). The app shows a canceled job's message instead of a fixed "canceled". Label Track and
  the Benchmark are unchanged.
- `get_progress` and `get_batch` pass the message through, so the CLI and MCP show it; their help
  text and the README give the example.

Tests: 231 pass, 6 new: a file canceled (by itself, its project or the whole batch) during its
first track or its last one; a joined project likewise; rows canceled before they started; rows
canceled while waiting for the GPU; a retried row counts again; and the job and a batch file
through `get_progress` / `get_batch`.

## Retry, all-canceled batches and paused rows (2026-11-27)

Three rough edges from a live test on the real GPU.

- **A plain retry picks up canceled or failed files.** In a one-transcript-per-file project, a
  canceled (or failed) file doesn't stop the project finishing "done", so a plain `retry-batch`
  found no failed or canceled project and was refused ("no failed or canceled projects to
  retry"), while `--projects 1` worked. The retry route (`/api/batch/{id}/retry` with no projects)
  now also takes every finished project that has a failed or canceled file; `_reset_ps` already
  re-queues only those files, so the done ones are left alone. It is refused only when nothing
  anywhere failed or was canceled ("nothing failed or canceled to retry"). The app's Retry button
  uses the same route and now counts those projects too.

- **Everything canceled reads "canceled".** A batch whose only project was canceled with
  `cancel --project` ended "done", while a whole-batch cancel ended "canceled". Now a finished
  batch is "canceled" when it was canceled or every project in it ended canceled
  (`_ended_canceled`, used where `BatchRegistry.run` finishes the batch). One level down, a
  one-transcript-per-file project whose files all ended canceled, each on its own, is "canceled"
  (its message is still the tally, e.g. "2 canceled"). A mix stays "done". Retry and restart
  work after it as before (they start the worker whenever the batch isn't running), the `--wait`
  loop already stopped on "canceled", and the app treats a canceled batch as ended, as it did.
- **Restart a file of a canceled project.** Found while checking the above, and older than it:
  restarting one file of a project canceled as a whole did nothing, because the project's cancel
  stayed set and canceled the file again at once. `restart_file` now clears it when the project
  is canceled.

- **The row in progress says "paused".** A batch paused while its file waited for the GPU said
  "GPU released", but the file's row kept saying "waiting for another run to finish". Now, while
  the batch is parked for a pause (`_gate` in `batch.py`), the row mid-run says "paused": the
  running file of a one-transcript-per-file project (and so that project's line, "a.mkv ·
  paused"), or a joined project. On resume it gets its previous line back, as `_asker` does for
  a question. This holds for every pause of a batch, between tracks as well as while waiting.
  `get_batch` passes it through (CLI and MCP), and the app already shows a row's message.

Tests: 238 pass, 7 new: a canceled file, then a failed one, each run again by a plain
`retry_batch` while the done file isn't; batches whose projects were all canceled, some
canceled, then retried; a per-file project with all its files canceled, then one restarted, and
a mix; a file of a project canceled as a whole restarted; a batch paused between tracks shows
"paused" on the file row and on a joined project, and a batch paused while waiting for the GPU on
the file row, each back to its line on resume; and the same through `get_batch`. Two older tests now expect "canceled"
for a batch whose only project, or only file, was canceled.

## A pause with another run waiting left a model on the GPU (2026-11-27)

Found in a live test on the real GPU (large-v3, 12 GB card). An idle server normally holds one
speech model plus the word-timing model (about 4.6 GB). After a run was paused mid-track while
another waited for the GPU, the idle server held about 8.2 GB until shutdown, and the next model
load had so little room that its batch size dropped from 14 to 2.

Cause: a run now parks outside its GPU turn, so the waiting run takes the turn the moment the
paused run's track ends and becomes a GPU user before the paused run unloads. `unload_models` then
took its fallback for "another thread is transcribing": it dropped the speech model from `_loaded`
and the model cache without `unload_model()`. CTranslate2 keeps a model's GPU memory until
`unload_model()`, so that model stayed on the GPU, unused and forgotten, and the next load made a
second copy.

- `unload_models` (`transcribe.py`) now frees the models only while it holds the turn: at once
  when the turn is free (or is its own: the pt-PT second pass, the Benchmark), otherwise it leaves
  everything loaded for the run on the GPU and `_gpu_turn` frees them when that run's turn ends.
  Asking for it and ending a turn are one step (`_unload_lock`), so the request can't be missed.
- So after the other run's track nobody uses them and they come off the GPU, and "paused · GPU
  released" holds. If that run has another track, it loads the model again for it (one load, a
  few seconds); what it loads stays cached after it ends, as after any run, and the paused run
  reuses it on resume.
- A pause while waiting for the GPU still leaves the running run's models alone (`_chunk.waiting`).
  Choosing another speech model (`use-model`) while a run is mid-track also goes through this, so
  it no longer leaves the old model behind either.

Live check on the real GPU, same sequence (a two-track job paused mid-track while a two-track job
waits, the second finishes, the first resumes and finishes), used memory with about 1.8 GB used by
other apps: before, 6.7 GB after one plain transcription and 10.3 GB idle after the sequence; now,
6.5 GB and 6.5 GB, batch size 14 for every load. A pause with no other run still drops to 2.1 GB.

Tests: 239 pass, 1 new: the paused run parks while the other run is already transcribing on the
same model; the other run finishes, nothing is left loaded while the first stays paused, and after
resume there is one copy, the cached one. The test for a run paused while waiting now also checks
that it asked for nothing to be unloaded.

## An ended run no longer says "pausing" (2026-11-27)

A run paused and then canceled before the pause landed (seen with a Benchmark) ended "canceled"
but `get-progress` still showed `paused: "pausing: ..."`, and the app's Transcribe and Label Track
lines would have shown "pausing…" instead of the canceled message. The pause request stays set on
the run; the routes now report `paused` only while the run is queued or running
(`/api/progress/{id}` for a job, `/api/batch/{id}` for a batch), so `get_progress`, `get_batch`
(CLI and MCP) and the app all stop showing it once the run is done, failed or canceled.

Wording: a job's "pausing" line said "the track being transcribed finishes first", which is wrong
for Label Track (the speaker finishes) and the Benchmark (the model it is timing finishes). It now
reads "pausing: the track being transcribed (Label Track: the speaker, Benchmark: the model)
finishes first"; a batch only runs tracks, so its line is unchanged. The app's lines were already
right (Transcribe: the track, Label Track: the speaker; the Benchmark has no Pause button there),
and the README says it as `pause` does.

Tests: 240 pass, 1 new: a job and a batch paused then canceled show no pause in `get_progress` /
`get_batch` or the routes, and the "pausing" wording of each.

## A canceled file stops part-way through its track (2026-11-27)

A single job and a joined batch project stopped at the next chunk of audio after a cancel, but a
project with one transcript per file only checked while the file waited for the GPU: otherwise
the cancel landed between tracks or after the last one, and the row went on saying "running ·
track 2/2" (about 22 s in a live test). Now a cancel of the file, its project or the whole batch
stops the track being transcribed at its next chunk there too, as for a job.

- `on_chunk` in `_process_project` (`batch.py`) checks on every chunk. Mid-track it asks
  `cancel_pending`, which never parks, so a pause still waits for the end of the track; while
  waiting for the GPU it still asks `is_cancelled`, which pauses or cancels at once, as before.
- The file ends "canceled" with nothing written. The interrupted track isn't counted or cached;
  the finished ones are, and the message says so (`canceled · 1 of 2 tracks transcribed, kept for
  a retry`). A file cancel lets the other files go on; a project or batch cancel stops as before.
- A file canceled while the batch is pausing now reads "canceled" at once: in
  `run_per_file_pipeline` (`pipeline.py`) a file cancel is looked at before the project / batch
  check, which parks on a pause. The batch then parks before the next file.
- Word timing has no per-chunk report, so a cancel during it still lands when the track ends (the
  check in `on_done` covers the last track), as for a job. Wording: `cancel` (CLI and MCP), the
  cancel routes and `BatchRegistry` docstrings, the README, and the app's batch Cancel button,
  which said "current file finishes" and now says "Canceling…", like the job's.

Tests: 245 pass, 5 new, in `test_pause_gpu_turn.py` with a fake model that reports every chunk: a
file, a project and a batch canceled in the second track's second chunk stop at that chunk (the
first three fail on the old code), a pause mid-track still waits for the end of the track, and a
file canceled while the batch is pausing. One older test in `test_actions.py` expected the running
track to finish and count ("2 of 2"); it now expects "1 of 2". Live on the real GPU (large-v3, a
14 min and a 2 min two-track file, own server and cache): `cancel --file 1` four seconds into
track 2 ended the file 2.5 s later, the polling included; the other file was written; the
interrupted track wasn't cached (the next batch transcribed it again); a whole-batch cancel took
the same 2.5 s.

## A canceled Benchmark keeps what it timed (2026-11-27)

A Benchmark canceled part-way returned no results, though some models had been timed: the list
was only handed over when the run finished. Now `benchmark.run` fills a list the route keeps
(`app.state.benchmarks`, by job id), so `/api/benchmark/{job_id}` (`get_benchmark`, CLI and MCP)
returns the models timed so far, with the run's status, "canceled". The same list keeps them when
the Benchmark fails part-way (status "error"), and it also shows the rows so far while it runs.

- The app's Benchmark screen shows that partial table under "canceled · the models timed before
  that:" (or "error: ..."); with nothing timed it says "canceled" or the error, as before.
- `run_benchmark` with `wait`: a run canceled or failed meanwhile is still an error (exit code 1,
  like `transcribe --wait`), and its message now gives the job id for `get_benchmark`. The
  docstrings (CLI help, MCP descriptions) and the README say so.
- A cancel still lands between models: the model being timed finishes and is kept.

Tests: 246 pass, 1 new (fails on the old code): the real `benchmark.run` with a fake model,
canceled after its first model, then failing on its second, then canceled during `run-benchmark
--wait`. One older test's stand-in for `benchmark.run` takes the new argument. Live on the real
GPU (own server and cache): canceled once two of the 17 models were timed, `get-benchmark` gave
`canceled` and both models (English and European Portuguese rows); `run-benchmark --wait`
canceled from outside exited 1 with the job id. The app's screen was not opened.

## A batch paused, then canceled, runs when retried (2026-11-27)

A batch paused and then canceled kept its pause request (`BatchStatus.paused`), which is only
reported while a run is going. A `retry-batch`, or a restart of one file, started a new run that
parked at its first checkpoint: "paused · GPU released", with nothing running until Resume was
pressed, although nothing showed a pause before the retry. `retry` and `restart_file` in
`BatchRegistry` now clear the pause along with the cancel when they start a new run, so it runs
to the end. A retry queued while the batch is still running keeps that run's pause, as before.

A single job paused and then canceled needs no change: it ends "canceled" with its thread gone,
the GPU turn free, nothing loaded and its clock stopped; there is nothing to retry.

Tests: 248 pass, 2 new: a batch paused, parked, then canceled, run again by `retry` and (another
one) by `restart_file`, each ends done (each half fails on the old code); a job paused, parked,
then canceled leaves nothing behind (passes on the old code too). Not run on the real GPU.

## A retry while a canceled batch is still stopping (2026-11-27)

A batch canceled mid-track needs a moment to stop: the track stops at its next chunk, or at the end
of its word timing, which can take much longer. A `retry-batch` or a restart of one file sent in
that moment saw the batch still running and was queued, but the canceled run then dropped its
queue. What the retry had reset was swept up as canceled (a failed project lost its error and
said "canceled") or, for a restarted file of a finished project, stayed "queued" for good.

- Now it is refused while the batch is stopping (canceled and still running, `stopping` in
  `batch.py`): `retry` and `restart_file` in `BatchRegistry` return False, and both routes answer
  409 "the batch is still stopping; try again in a moment" before any other check. So the CLI,
  the MCP tool and the app's per-row Restart buttons give that message; the app's batch Retry
  button only appears once the batch has ended. A retry once it has stopped runs as before, also
  after a pause then a cancel.
- Refusing was chosen over keeping the retry for later: while the batch stops, the rows being
  canceled still read "running", so a plain retry couldn't pick them yet and a retry of them is
  refused as still running; keeping it the same as a retry a moment later would mean postponing
  those choices until the run stops.
- Found while checking: a re-run queued while the batch ran, then a batch cancel, could leave rows
  "queued" too (a restarted file of a finished project, or a project waiting behind a re-run). The
  canceled run now marks what it drops canceled (`_cancel_queued`).

Tests: 252 pass, 4 new (each fails on the old code): a retry and a restart while the batch is
stopping are refused and run once it has stopped, with and without a pause first; a restart queued
before a batch cancel ends canceled; and the same refusal through the CLI, MCP and the routes. Not
run on the real GPU.

## Canceling one project or file while the batch is paused (2026-11-27)

A cancel of one project or one file of a batch that was paused, or still pausing, was answered but
did nothing until Resume: the row kept saying "Canceling…", and a canceled file could even start
its next track before it stopped. A whole-batch cancel already worked at once. The batch's
checkpoint (`_gate` in `batch.py`) parked for the pause before the project's or file's own cancel
was looked at, and the park only ended on Resume or a whole-batch cancel.

- `_gate` no longer parks while the project or file in progress is canceled, and a park ends when
  it is: the item stops at once ("canceled" with how many tracks were transcribed, nothing written,
  no further track), then the batch parks again before anything else starts.
- The project's own check comes after the checkpoint where it was before it (a file between
  tracks in `run_per_file_pipeline`, and while waiting for the GPU in `on_chunk`), so a park ended
  by its cancel is seen.
- With nothing left to do, the batch ends instead of parking: the end of a one-transcript-per-file
  project and the end of the queue check for a cancel without parking. So a pause asked for
  during the last track of such a project now lets the batch end "done"; a pause before the next
  project, or before a joined project's merge step, still parks.
- A project or file that has not started yet and is canceled while the batch is paused still
  shows "canceled" when the queue reaches it (after Resume). Marking it at once would let a
  Restart or Retry re-queue it while the current pass still holds it, so it could run twice.
- Wording: `cancel` (CLI and MCP), the README and the `BatchRegistry` docstrings say so.

Tests: 256 pass, 4 new in `test_pause_gpu_turn.py` (each fails on the old code): a file and a
project canceled while the batch is parked (the file: the batch stays paused and the next file runs
after Resume; the project: the batch ends), the last file canceled while the batch is pausing ends
the batch, and a joined project canceled while pausing after its first track stops at once.

## Partial / outstanding (v1 polish)

- **FR-MG-4 bleed (P1):** detection flags `bleed_suspected`; the louder mic's line is marked
  `preferred` from per-track loudness measured before each temp wav is deleted, only when it is at
  least *Bleed loudness ratio* (default 2×) louder. Verified on a synthetic 15% cross-bleed
  recording; the default is still untuned on real bleed footage (OQ-3).
- **Portuguese (Portugal) + large-v3 punctuation merge (#23), experimental:** built and off by
  default. Verified only on one 14 s synthetic European Portuguese clip (Windows Helia voice):
  CAMÕES 0% word errors / 1 punctuation mark, large-v3 2.9% (Brazilian *econômico*) / 5, merge
  0% / 5. Needs thorough testing on real recordings before it's trusted or turned on by default;
  the checklist is on issue #23. Tool: `backend/compare_pt.py`.
  First real test (2026-11-09): RTPSpeech, two RTP newsreaders, ~7 min each, clips joined so
  alignment runs across many 30 s windows. Word errors (ignoring accents) CAMÕES 18.5% / 25.0%,
  large-v3 12.8% / 16.7%, merge = CAMÕES. CAMÕES drops whole phrases more often and writes
  numbers as words; on this clean broadcast speech it does not hear better than large-v3. The
  merge works as designed (sentence ends found 89–94% vs CAMÕES's 47–55%) but can't restore missed
  words. Stays off by default. Caveats: two speakers, clean studio speech, transcripts slightly out of
  step with the audio. Timing not measured cleanly (another app held 6.5 GB of VRAM).
  **Clean timing (2026-11-09):** the app's real path (`transcribe_track`, pt-PT, no cache), two RTP
  tracks of ~7 min in a row after a warm-up, each setting in a fresh process, run twice; GPU had only
  Windows + NVIDIA Broadcast (1 GB). After both spill fixes (below, and the #37 margin): setting off
  30.7 / 30.6 s, on 65.1 / 65.7 s, so the merge costs about **2×** (roughly +17 s per 7 min track): each track also
  swaps models (unload CAMÕES, load large-v3, reload CAMÕES for the next). Before that fix it looked
  like 1.6× (off 47.0 / 61.9, on 78.1 / 93.9) only because "off" was slowed by the spill too. Words
  are identical either way; punctuation marks 61 → 133 and 73 → 120. Keeping both models loaded is
  much slower on 12 GB (165.6 / 248.6 s): only batch 5 fits and it still spills 1.3–1.4 GB.
  `compare_pt.py` times transcription only (no swaps, no warm-up), so use the app path for timing.
  Any GPU timing in this project needs a clean GPU, or it measures the other app.
- **Word timing left GPU memory behind (fixed 2026-11-09).** After `whisperx.align`, PyTorch kept
  1.1–1.7 GB of spare working memory on the card on top of the 1.2 GB word-timing model, so every
  track after the first in a run spilled its batch into system memory: 41–57 s instead of 6.7 s for a
  7 min track's transcription (A/B, twice each). `transcribe_track` now calls
  `torch.cuda.empty_cache()` after aligning. Any language, any multi-track file or batch.
  The small spill left after this (0.1–0.7 GB at batch 16) was the batch-size margin, now 0.9 GB
  (see #37 above).
- **FR-UI-5 output toggles:** only _Show Speakers_ is implemented. _Show Transcription_, _Show
  CC_ and _Strip Silence_ are **not applicable** (decided 2026-11-09): they came from Transcript
  Merger, which worked on DaVinci Resolve transcripts. There, "transcription" is dialogue, "CC" is
  closed-caption sound labels (a door slam), and long gaps are written as "silence" entries.
  Whisper's output has neither sound labels nor silence entries (its voice detection drops silence
  first), and Resolve compatibility is no longer a goal. Add them back only if Resolve transcripts
  are imported again.
- **FR-UI-3:** add/remove lines works; "smart closed-caption placement" is basic (new line
  starts at the previous line's end), not the full CC-placement logic.
- **FR-UI-7 keyboard:** done (2026-11-09). In a text cell: Enter inserts a line break; Ctrl+Enter
  goes to the next line; Escape reverts the line; Ctrl+↑/↓ jump rows; plain ↑/↓ move the caret
  through the text and, from its very start/end, on to the row above/below (landing at the end of
  the text going up, the start going down, so every line of a multi-line cell is visited).
- **FR-UI-6:** per-file framerate / timecode handling (for Edit mode) not implemented.
- **FR-UI-9:** done (2026-11-09). Switching tabs never lost anything (tabs are hidden, not rebuilt);
  now each tab's settings also survive a reload or restart, in this browser (`localStorage`
  `wm-settings`): Batch language, one-transcript-per-file and its merge + subtitle settings;
  Transcribe & Merge language, conflict detection and thresholds; Label Track region pad and
  language; and each tab's Export panel (formats, limits, pad, speaker options). Not remembered:
  files, filenames, zoom, per-project overrides. Models-tab settings are saved by the server.
- **FR-MD-2 import:** Edit tab now loads **all of our export formats** — JSON / SRT / TXT / CSV
  (auto-detected by extension, with a content sniff fallback; parsers in `app.js` mirror
  `export.py`). JSON round-trips word-level timing; SRT/TXT/CSV import as plain segments
  (`words:[]`). Generic third-party SRT loads too (the parser isn't WhisperMerger-specific);
  proprietary formats (Descript JSON, Premiere/Resolve XML) are still unsupported.

- **TXT without timestamps opens in Edit Transcript as lines with no timing (2026-11-25):** Export
  has a *Timestamps in TXT* option (on by default). With it off, TXT lines have no
  `[start --> end]`; with *Show speakers* off too, TXT is the text alone. Edit Transcript now loads
  such a file (`parseTxt` in `frontend/app.js`): every non-empty line becomes one transcript line,
  a leading `Name: ` is read as the speaker (same rule as timed TXT, otherwise "Speaker"), Start and
  End are 0, and the line carries the `no_timing` flag, shown as *no timing* in the Notes column.
  A file that mixes timed and untimed lines loads both, and pasted plain text loads the same way.
  Typing a Start and a later End on a line removes its mark. The flag is an ordinary entry in
  `flags`, so it survives a JSON or CSV export and reload with no model change. The export summary
  warns, with a count, when lines with no timing are about to go into a format that carries times
  (SRT, CSV, JSON, or TXT with timestamps on); they would be written at 00:00:00. It shows even
  with *Don't show again* ticked, and doesn't block the export. The CLI and MCP `export` have the
  switch too (`--no-txt-timestamps` / `txt_timestamps: false`); `run_batch` takes no export
  options, so it has none. Still true: timing isn't recovered, only typed back in by hand, so keep a
  JSON (or a timestamped TXT) for anything that may need editing later. A line break inside one
  line's text comes back as separate lines (the extra ones marked *no timing*). The text stays one
  line per transcript line (no flowing-paragraph mode).

## Validation — real-world stress test ✅

- **SM-1 / SM-2 / NFR-1 — validated** on a real 56 GB / 1h51m / 4-track OBS recording (cold):
  full file transcribed end-to-end in **~10 min** (≈7.4 h of track-audio across 4 tracks),
  **11,860 words**, full 6639s coverage, coherent text + per-word timestamps, deterministic on
  re-run. Beats NFR-1's 12–20 min / 5 h budget. Stage breakdown: ~2.6 min extraction (four
  full-file demuxes), ~7.5 min transcription+align (incl. one-time model load), merge negligible.
- **Extraction overlap — resolved:** extraction is single-pass (`extract_tracks` reads the source
  once for all tracks) and now **prefetched** — `extract.extract_ahead` extracts the next file while
  the GPU transcribes the current one, so the GPU no longer idles between files. GPU throughput is
  tunable via `WHISPERMERGER_BATCH_SIZE` (default 16, and `Startup.bat` ships 16: on a 12 GB card 24 and 32 spill into system memory and
  run slower, measured 2026-11-08, see #24). See the
  2026-06-19 "GPU uptime" entry below.
- **Bleed (FR-MG-4):** the near-duplicate-text flag works, but real "duplicate lines across
  speakers" on this footage turned out to be **clean-vs-processed captures of the same mic**, not
  bleed — solved by deselecting the redundant track, no algorithm needed. Model-based RMS
  resolution remains unbuilt/untuned (deferred).

## v2 — single-track manual diarization (PRD §11.4) — built

For recordings with no stems (one mixed track, multiple speakers). All phases done:

- **Phase A (engine) — DONE.** `diarize.py`: silero-VAD speech detection + per-speaker region
  transcription with timeline remapping → merge (`/api/vad`, `/api/diarize`). Validated on real
  footage (22 speech regions found in a 60s vlog slice; remapping unit-tested).
- **Phase B (labeling UI) — DONE.** "Label Track" tab: WaveSurfer.js waveform (vendored,
  offline), `/api/track-audio` feed, VAD-seeded regions, speaker buttons + colors, click/
  number-key assignment, drag-to-create + drag-to-resize regions (free via WaveSurfer),
  per-region playback → `/api/diarize` → result in the editor.
- **Multi-file sequencing — DONE.** Add several single-track files in the Label tab → they
  concatenate into one combined waveform (`/api/combine`, cached) whose timeline is the
  sequenced one, with file-join markers; reorder/track-change auto-rebuilds. Reuses the whole
  single-file flow (VAD/label/diarize/preview) unchanged.
- **Phase C (refinement) — DONE.** Boundary drag + free-span (WaveSurfer), **bulk-assign**
  (shift-click a run / ctrl-click multi-pick → one speaker), and the **single-track ↔
  split-lane view toggle** (one row per speaker + an unassigned row, blocks on a shared
  timescale, click to select).
- **Overlapping speech — supported.** Two overlapping regions assigned to different speakers
  represent talk-over ("Duplicate region" copies a span onto another speaker); each is
  transcribed and the merge flags `overlap`, and split-lane shows it cleanly. Per PRD §11.4,
  truly simultaneous speech still yields one (garbled) line per speaker — a known limit.
- **Silent server crash fixed (2026-11-08, #25).** Switching tracks fires a new `/api/vad` while the
  last one still runs, and the server handles each request on its own thread. silero-VAD keeps state
  between chunks, so two threads sharing it corrupted memory (Windows logged heap corruption,
  `0xc0000374`) and the server quit with no traceback. `detect_speech` now runs one at a time
  (`_vad_lock`), and the UI drops a detection result if you've moved to another track meanwhile.

## Model downloader: queue, honest status, Download all (2026-11-08, #26)

- **Queue** — `DownloadManager` runs `MAX_PARALLEL` = 2 at once (a semaphore); the rest show *queued*.
  A run's `ensure()` passes `skip_queue` so it never waits behind a Download all. A queued model that
  got installed meanwhile is marked done without fetching again.
- **Honest status** — `queued | connecting | downloading | retrying | done | error | canceled`.
  A stall reads "no data for 90 s, retrying (attempt N)"; any other fetch failure shows its real
  error while retrying, and errors no retry fixes (`_PERMANENT`: 404/401/403, not found, gated, disk
  full, access denied) fail at once.
- **Disk check** — before each download, `need > free` on the target drive fails with both sizes.
  `/api/models` returns `disk_free` for the Download all confirm.
- **Download all / Cancel all** per section (speech, word-timing) with done / total, speed and time
  left. The row updates in place, since rebuilding it every second swallowed clicks.
- **No orphan downloads** — `app.fetch` exits when its stdin (held by the server) closes, so
  Shutdown or a crash no longer leaves a download running to fight a restarted server's.

- **Feedback from #25 (2026-11-09).** *All tracks* toggle (multi-track file only): the other tracks
  as read-only waveforms under the main one, same width (fit width × zoom) and scroll (WaveSurfer's
  `scroll` event), amplitude from the Amp slider; peaks from `/api/peaks/{id}?track=N`, which reuses
  the full-track wav Label Track extracts (freed by `/release`). Speech detection already ran on
  load; the button is now *Detect again* at the end of the row, asks before replacing labeled
  regions, and Region pad re-pads the last detection straight away while nothing is labeled.
- **All tracks stays sharp when zoomed in (2026-11-09).** It was blurry past ~zoom 10: 4000
  whole-file peaks stretched over one canvas (capped at 16384 px). Now each row's canvas is only
  as wide as the view and redraws as it scrolls, from the whole-file peaks at first, then from a
  window around what's on screen (`/api/peaks/{id}?track=N&start=&end=`, one column per pixel,
  the view plus one view-width either side). Windows are scaled by the whole track's loudest point
  so they match. One window request at a time; when it lands, the next one follows the view.
- **Language dropdowns fit their longest name (2026-11-09).** They were 120 px like every field and
  cut "Portuguese (Brazil)" to "Portuguese (I"; the three Language selects now size to their options.

## Not started / deferred

- **Tauri packaging** (PRD §16): wrap FastAPI + browser into a `.exe`. Future.
- **Recording-to-Video skill** (PRD Appendix C): downstream product, its own PRD, out of scope.
