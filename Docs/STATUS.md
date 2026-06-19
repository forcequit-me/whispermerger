# WhisperMerger — Build Status

Snapshot of what's built vs. outstanding, mapped against `whispermerger-prd.md` (FR-\* IDs)
and `whispermerger-build-plan.md` (steps 1–10). Last updated 2026-06-19.

## Summary

Both v1 (multi-track stems + batch studio) and v2 (single-track manual diarization) are
**built and running** on native Windows + RTX 3080 Ti (the build plan's env was validated;
WSL2 not needed). Full pipeline works end-to-end: pick files → transcribe on GPU (WhisperX
large-v3 + wav2vec2) → merge with conflict flags → edit → export. **Validated on a real 56 GB /
1h51m / 4-track recording** — full transcription in ~10 min (beats NFR-1). **98 GPU-independent
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

## Partial / outstanding (v1 polish)

- **FR-MG-4 bleed (P1):** detection flags `bleed_suspected` and an `rms_fn` hook exists, but
  real RMS-from-source-audio is not yet wired in, and the threshold is untuned (OQ-3). Needs
  real recordings to calibrate.
- **FR-UI-5 output toggles:** only _Show Speakers_ is implemented. _Show Transcription_, _Show
  CC_, _Strip Silence_ are not (Strip Silence is a no-op on the audio path per the PRD anyway).
- **FR-UI-3:** add/remove lines works; "smart closed-caption placement" is basic (new line
  starts at the previous line's end), not the full CC-placement logic.
- **FR-UI-7 keyboard:** Escape (revert) and Ctrl+↑/↓ (move between rows) done. Enter now inserts
  a line break inside a text cell (text cells became auto-growing textareas), so it no longer
  advances rows; Ctrl+Enter and bare arrow-up/down not bound.
- **FR-UI-6:** per-file framerate / timecode handling (for Edit mode) not implemented.
- **FR-UI-9:** per-tab settings preservation not implemented.
- **FR-MD-2 import:** Edit tab now loads **all of our export formats** — JSON / SRT / TXT / CSV
  (auto-detected by extension, with a content sniff fallback; parsers in `app.js` mirror
  `export.py`). JSON round-trips word-level timing; SRT/TXT/CSV import as plain segments
  (`words:[]`). Generic third-party SRT loads too (the parser isn't WhisperMerger-specific);
  proprietary formats (Descript JSON, Premiere/Resolve XML) are still unsupported.

## Validation — real-world stress test ✅

- **SM-1 / SM-2 / NFR-1 — validated** on a real 56 GB / 1h51m / 4-track OBS recording (cold):
  full file transcribed end-to-end in **~10 min** (≈7.4 h of track-audio across 4 tracks),
  **11,860 words**, full 6639s coverage, coherent text + per-word timestamps, deterministic on
  re-run. Beats NFR-1's 12–20 min / 5 h budget. Stage breakdown: ~2.6 min extraction (four
  full-file demuxes), ~7.5 min transcription+align (incl. one-time model load), merge negligible.
- **Extraction overlap — resolved:** extraction is single-pass (`extract_tracks` reads the source
  once for all tracks) and now **prefetched** — `extract.extract_ahead` extracts the next file while
  the GPU transcribes the current one, so the GPU no longer idles between files. GPU throughput is
  tunable via `WHISPERMERGER_BATCH_SIZE` (default 16; `Startup.bat` ships 24 on the 3080 Ti). See the
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

## Not started / deferred

- **Tauri packaging** (PRD §16): wrap FastAPI + browser into a `.exe`. Future.
- **Recording-to-Video skill** (PRD Appendix C): downstream product, its own PRD, out of scope.
