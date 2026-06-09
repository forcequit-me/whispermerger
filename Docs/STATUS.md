# WhisperMerger — Build Status

Snapshot of what's built vs. outstanding, mapped against `whispermerger-prd.md` (FR-* IDs)
and `whispermerger-build-plan.md` (steps 1–10). Last updated 2026-06-09.

## Summary

v1 is **functionally complete and running** on native Windows + RTX 3080 Ti (the build plan's
env was validated; WSL2 not needed). Full pipeline works end-to-end: pick files → transcribe
on GPU (WhisperX large-v3 + wav2vec2) → merge with conflict flags → edit → export. A **batch
"studio"** (beyond the original PRD) was added on top. 56 GPU-independent tests pass; the GPU
path passes the §11.3 smoke gate. Remaining work is mostly P1 editor polish, bleed-threshold
tuning, foreign-format import, and real-footage validation.

## Built ✅

**Build-plan steps 1–10:** all done (env + smoke gate, data model, merge engine, probe/extract,
cache, WhisperX wrapper, FastAPI, export, fresh frontend, polish).

**Input (§8.1):** FR-IN-1/2/3 audio + multi-track video + mixed in one batch · FR-IN-4 ffprobe
streams+duration · FR-IN-5 ffmpeg mono/16k/pcm_s16le extraction · FR-IN-6 temp deleted after use.

**Track ID (§8.2):** FR-TK-1 single InputSource model · FR-TK-3 per-track audition · FR-TK-4
selector only when >1 stream. *(FR-TK-2 changed by request: defaults are **Speaker 1/2/…**, not
OBS Mic/Desktop; track labels shown 1-based.)*

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

## Validation outstanding

- **SM-1 / NFR-1:** real word-level accuracy and 5-hour-stem timing (12–20 min) not yet measured
  on real footage — the smoke test used a synthetic tone (0 words). Needs a real recording run.
- **SM-5 / OQ-3:** bleed flagging + RMS preference tuned on real multi-track recordings.

## Not started / deferred

- **v2 — single-track manual diarization** (PRD §11.4, all FR-V2-*): WaveSurfer.js waveform
  labeling, VAD-seeded regions, single-track vs split-lane views. Explicitly P2. The data model
  already generalizes to "one virtual track per speaker," so no engine change needed later.
- **Tauri packaging** (PRD §16): wrap FastAPI + browser into a `.exe`. Future.
- **Recording-to-Video skill** (PRD Appendix C): downstream product, its own PRD, out of scope.
