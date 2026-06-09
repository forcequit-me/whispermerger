# WhisperMerger — Build Status

Snapshot of what's built vs. outstanding, mapped against `whispermerger-prd.md` (FR-* IDs)
and `whispermerger-build-plan.md` (steps 1–10). Last updated 2026-06-09.

## Summary

Both v1 (multi-track stems + batch studio) and v2 (single-track manual diarization) are
**built and running** on native Windows + RTX 3080 Ti (the build plan's env was validated;
WSL2 not needed). Full pipeline works end-to-end: pick files → transcribe on GPU (WhisperX
large-v3 + wav2vec2) → merge with conflict flags → edit → export. **Validated on a real 56 GB /
1h51m / 4-track recording** — full transcription in ~10 min (beats NFR-1). **67 GPU-independent
tests pass**; the GPU path passes the §11.3 smoke gate. Remaining work is optional polish (some
`FR-UI-*` toggles) and foreign-format import.

## Built ✅

**Build-plan steps 1–10:** all done (env + smoke gate, data model, merge engine, probe/extract,
cache, WhisperX wrapper, FastAPI, export, fresh frontend, polish).

**Input (§8.1):** FR-IN-1/2/3 audio + multi-track video + mixed in one batch · FR-IN-4 ffprobe
streams+duration · FR-IN-5 ffmpeg mono/16k/pcm_s16le extraction · FR-IN-6 temp deleted after use.

**Track ID (§8.2):** FR-TK-1 single InputSource model · FR-TK-3 per-track audition · FR-TK-4
selector only when >1 stream. *(FR-TK-2: track labels default to **Speaker 1/2/…** (1-based),
editable — changed from the originally-specced OBS convention.)*

**Transcription (§8.3):** FR-TR-1 large-v3 + wav2vec2 align · FR-TR-2 float16 · FR-TR-3 VAD ·
FR-TR-4 progress · FR-TR-5 auto-merge on completion.

**Merge (§8.4):** FR-MG-1 interleave by time · FR-MG-2 backchannel (flag + parenthetical/
separate/skip) · FR-MG-3 genuine-overlap flag · FR-MG-5 conflicts surfaced in editor. Conflict
detection is toggleable.

**Word data (§8.5):** FR-WD-1 words stored under segments · FR-WD-2 preserved in JSON ·
FR-WD-3 SRT line splits use real word timestamps.

**Export (§8.7):** FR-EX-1 SRT/TXT/CSV/JSON · FR-EX-2 word-level SRT splits. All four formats
carry start+end timing; subtitle limits (CPL / max-lines / min-duration) toggleable.

**Caching (§8.8):** FR-CA-1/2/3 SHA-256(file)+model+align key; re-runs skip transcription.

**Modes (§8.9):** FR-MD-1 Transcribe & Merge · FR-MD-2 Edit Transcript · FR-MD-3 sequence-only
retired (folded into the pipeline).

**Editor (§8.6, partial):** FR-UI-1 inline edit speaker/text/timing · FR-UI-2 global rename ·
FR-UI-8 custom export filename · FR-UI-10 conflict-flag highlighting. Plus **remove a speaker +
all their lines** (new request).

## Built beyond the PRD (this chat)

- **Batch "transcription studio"** — Method 1 (a project's session files auto-merge sequentially
  with cumulative offsets) + Method 2 (queue many projects, each auto-exports all formats to its
  own subfolder). Per-project error isolation; cache-resumable. New `batch.py` + `/api/batch`.
- **In-place file reads via a native picker** (`/api/browse`, `/api/browse-folder`) — no upload/
  copy, so 100–200 GB files work. Replaced the byte-upload flow.
- Toggleable conflict detection and subtitle limits; timestamps added to TXT export.
- **Per-line audio preview** — ▶ a line or drag a jump bar; instant input-seek clips, works on
  100+ GB files (`/api/clip`).
- **Stage-aware progress + total time** — jobs/batches report the current stage (Extracting ·
  Transcribing · Merging, per track) and elapsed wall time across *all* stages, not just
  transcription, so the UI never looks stalled.
- **Storage hygiene** — each extracted track wav is deleted immediately after use; the working
  dir (extracted tracks, combined wavs, preview clips, audition samples) is wiped on server
  startup. Only the small WhisperX output cache is kept.
- Documented torch-2.6/Windows dependency fixes (setuptools<81, transformers<5, weights_only,
  cuDNN 8 on PATH) — see `SETUP.md`.

## Partial / outstanding (v1 polish)

- **FR-MG-4 bleed (P1):** detection flags `bleed_suspected` and an `rms_fn` hook exists, but
  real RMS-from-source-audio is not yet wired in, and the threshold is untuned (OQ-3). Needs
  real recordings to calibrate.
- **FR-UI-5 output toggles:** only *Show Speakers* is implemented. *Show Transcription*, *Show
  CC*, *Strip Silence* are not (Strip Silence is a no-op on the audio path per the PRD anyway).
- **FR-UI-3:** add/remove lines works; "smart closed-caption placement" is basic (new line
  starts at the previous line's end), not the full CC-placement logic.
- **FR-UI-7 keyboard:** Enter (next), Escape (revert), Ctrl+↑/↓ (move) done; Ctrl+Enter and bare
  arrow-up/down not bound.
- **FR-UI-6:** per-file framerate / timecode handling (for Edit mode) not implemented.
- **FR-UI-9:** per-tab settings preservation not implemented.
- **FR-MD-2 import:** Edit tab loads WhisperMerger JSON only; importing foreign transcripts
  (Resolve / Premiere / Descript SRT etc.) is not built yet.

## Validation — real-world stress test ✅

- **SM-1 / SM-2 / NFR-1 — validated** on a real 56 GB / 1h51m / 4-track OBS recording (cold):
  full file transcribed end-to-end in **~10 min** (≈7.4 h of track-audio across 4 tracks),
  **11,860 words**, full 6639s coverage, coherent text + per-word timestamps, deterministic on
  re-run. Beats NFR-1's 12–20 min / 5 h budget. Stage breakdown: ~2.6 min extraction (four
  full-file demuxes), ~7.5 min transcription+align (incl. one-time model load), merge negligible.
- **Known optimization (not blocking):** extraction re-reads the source once per track
  (~2.6 min on the 56 GB file); a single-pass extract (read once → all tracks) would shave ~2 min.
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
