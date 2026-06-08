# WhisperMerger v1 — Build Plan

## Context

Goal: build WhisperMerger v1 per `Docs/whispermerger-prd.md`. Local desktop app that turns multi-track recordings (OBS MKV w/ mic+desktop streams, or several audio files) into speaker-labeled transcripts with **word-level** timestamps, via WhisperX on a local GPU.

Current state: repo holds only the PRD. Nothing built. The "Transcript Merger" web tool (`github.com/forcequit-me/Transcript-Merger`, a single 5381-line `index.html`) is the **feature reference, not a source to copy**. Per user direction, WhisperMerger gets its **own fresh frontend** that reproduces TM's *capabilities* — multi-format export (TXT/SRT/CSV/JSON), subtitle limits (CPL / max-lines / min-duration), output toggles (Speakers/Transcription/CC/Strip-Silence), inline editing, global speaker rename, and merge + sequential-ordering — without porting its markup or JS. (This relaxes PRD G3/NFR-5 "reuse TM code" to "reproduce TM behavior".)

User delegated all decisions ("what do you suggest", "build and source everything") — environment setup, sourcing, and build order are mine to drive.

**Verified runtime on this machine (changes the PRD's env premise):** RTX 3080 Ti 12GB visible via `nvidia-smi`; `torch 2.6.0+cu124` already installed with `torch.cuda.is_available() == True`; `ffmpeg`/`ffprobe` already on PATH (`C:\Program Files\FFmpeg`); native Python 3.12.7. The PRD chose WSL2+conda only to dodge native-Windows WhisperX dependency hell — but CUDA torch already runs natively here, so that pain is already solved. **v1 runs natively on Windows.** WSL2+Ubuntu+conda stays documented as a fallback only if installing whisperx fights the existing torch.

Why engine-first (PRD §11.1): a UI-first build risks discovering the merge logic needs a different data shape after the frontend is wired. The merge engine + data model lock the contract (§9.2) first, against which everything else is built. The merge engine and ffmpeg/ffprobe layer are also GPU-independent (no model load), so they're fast to test; only WhisperX transcription touches the GPU.

## Decisions made (open questions resolved for v1)

- **OQ-1 input scope:** v1 targets well-formed OBS MKV + clean audio files. No sync-drift/sample-rate robustness yet (flag, don't fix).
- **OQ-2 backchannel default:** user-configurable; default = `parenthetical`.
- **OQ-4 cache key:** SHA-256 of the **file** (fast; per PRD lean). Re-encode busts it — acceptable v1.
- **Frontend:** built **fresh** (own vanilla HTML/JS, no framework per PRD §10.2), not ported from TM. Feature parity only.
- **Exports server-side:** TXT/SRT/CSV/JSON generated in Python (`export.py`) from the merged `MergedOutput`, so word-level SRT splitting (FR-WD-3/FR-EX-2) uses the word data directly and exports stay single-sourced. Frontend just requests a format and downloads.
- **Job model:** in-process async job + poll (no Celery/Redis). Single-user local app.
- **Env:** **native Windows** (Python 3.12, existing `torch 2.6.0+cu124`, existing ffmpeg). `pip install whisperx` into a venv, pinned so it does **not** downgrade/replace the working torch. `Docs/SETUP.md` documents this; WSL2+Ubuntu+conda kept only as a fallback if the whisperx install fights torch.

## Proposed repo layout

```
backend/
  app/
    main.py          # FastAPI app + static frontend mount
    models.py        # pydantic: Word, Segment, Track, InputSource, MergedOutput (§9.1/§9.2)
    probe.py         # ffprobe: enumerate audio streams + duration (FR-IN-4)
    extract.py       # ffmpeg: stream -> mono/16k/pcm_s16le temp wav (FR-IN-5/6, Appendix A)
    transcribe.py    # WhisperX large-v3 + wav2vec2 align wrapper (FR-TR-*) [GPU]
    merge.py         # pure-python merge engine, conflict hierarchy (§8.4)
    cache.py         # SHA-256-keyed WhisperX-JSON cache (§8.8)
    export.py        # MergedOutput -> TXT/SRT/CSV/JSON; word-level SRT splits (FR-EX-*, FR-WD-3)
    jobs.py          # in-process job registry + progress
    routes/          # upload, tracks, transcribe, progress, result, export
  tests/
    test_merge.py    # synthetic WhisperX JSON -> assert merged output + flags
    test_probe.py    # ffprobe parse on a generated multi-track file
    test_extract.py
  requirements.txt   # fastapi, uvicorn, whisperx, pydantic, numpy, pytest (torch already present)
frontend/
  index.html         # fresh UI: "Transcribe & Merge" + "Edit Transcript" tabs (own design)
  app.js
  styles.css
Docs/
  SETUP.md           # native-Windows venv + whisperx install + GPU smoke test (§11.3); WSL fallback
README.md
```

## Build order (execute top-down)

1. **Scaffold + env** — repo layout, `requirements.txt` (fastapi, uvicorn, whisperx, pydantic, numpy, pytest; reuse existing torch+CUDA), venv, `Docs/SETUP.md`. Install whisperx pinned so it keeps the working `torch 2.6.0+cu124`, then a one-stem smoke test to confirm whisperx imports + runs on the GPU (the §11.3 gate, adapted to native Windows since `nvidia-smi`/torch.cuda already pass).
2. **Data model** (`models.py`) — pydantic types matching §9.2 exactly: `Word{word,start,end}`, `Segment{speaker,start,end,text,words[],flags[]}`, `Track{track_index,speaker,selected}`, `InputSource{filename,duration,tracks[],sequence}`, `MergedOutput{version,sources[],segments[]}`. This is the locked contract.
3. **Merge engine** (`merge.py`) — pure python, the core. Signature `merge(whisperx_outputs, labels, order) -> MergedOutput`. Conflict hierarchy in order (§8.4): (a) FR-MG-1 no-overlap interleave by start; (b) FR-MG-2 backchannel (B fully inside A, < N words) → flag `backchannel`, render per pref; (c) FR-MG-3 genuine overlap (multi-sec simultaneous) → word-boundary interleave + flag `overlap`; (d) FR-MG-4 bleed (similar text both streams in small window) → RMS compare from source audio, flag `bleed_suspected`/`preferred`. Sequential offsets applied from `order`. **Full pytest coverage with synthetic WhisperX JSON** — this is the verification anchor.
4. **Probe + extract** (`probe.py`, `extract.py`) — ffprobe stream/duration enumeration; ffmpeg extraction (Appendix A invocation) to temp dir, deleted after use (FR-IN-6/NFR-3). OBS-convention label pre-fill: Track 1→"Mic", Track 2→"Desktop" (FR-TK-2). Tested against an ffmpeg-generated multi-stream file.
5. **Cache** (`cache.py`) — SHA-256(file) + model-version + align-model key → cached WhisperX JSON; skip re-transcription on hit (§8.8).
6. **WhisperX wrapper** (`transcribe.py`) — large-v3 + wav2vec2 align, float16, VAD on (FR-TR-1/2/3). Cache-aware. Runs on this machine's GPU once whisperx is installed (step 1).
7. **FastAPI** (`main.py`, `routes/`) — endpoints per §11.1.5: upload/select files, get track info, start transcription job, poll progress, get merged result. Serves `frontend/`.
8. **Export module** (`export.py`) — TXT/SRT/CSV/JSON from `MergedOutput`; SRT line splits use word timestamps (FR-WD-3/FR-EX-2); honors subtitle limits (CPL/max-lines/min-duration) and output toggles. Unit-tested.
9. **Fresh frontend** (`frontend/`) — own vanilla HTML/JS/CSS, two tabs: **"Transcribe & Merge"** (FR-MD-1: drop files → label tracks → run → merged transcript; folds in sequential ordering UC2, retires standalone Sequence FR-MD-3) and **"Edit Transcript"** (FR-MD-2: load existing transcript). Capabilities: drag-drop, track selector shown only when >1 stream (FR-TK-4), inline edit (FR-UI-1), global speaker rename (FR-UI-2), add/remove entries (FR-UI-3), subtitle-limit + output-toggle controls (FR-UI-4/5), keyboard shortcuts (FR-UI-7), custom export filename (FR-UI-8). Talks to FastAPI via `fetch()`; downloads come from the export endpoint.
10. **Polish** — per-track 5–10s audition (FR-TK-3), progress UI (FR-TR-4), conflict-range highlighting from `flags` (FR-UI-10).

v2 (single-track manual diarization, WaveSurfer.js, §11.4) and Tauri packaging are out of scope.

## Key integration point

Since the frontend is fresh, we design its editor model **around** the engine contract (§9.2: nested segments with `words[]` + `flags[]`) instead of bridging to TM's flat-entry shape. The editor's display unit stays the segment (NG5/FR-WD-1); words ride underneath, untouched in the UI but preserved through edit → export. The one seam to keep honest: edits to a segment's text/timing must not silently invalidate its `words[]` (on text edit, either re-derive nothing and drop word data for that segment, or keep words and accept staleness — pick one explicitly and mark it in `flags`).

## Verification

- **Engine/probe/extract (GPU-independent — fast, no model load):** `pytest backend/tests/` — synthetic WhisperX JSON drives every merge conflict branch and asserts emitted `flags`; ffprobe/ffmpeg tested on an ffmpeg-generated multi-track file. (These don't need the GPU; the machine has one regardless.)
- **Transcription (native Windows GPU — already on this machine):** `nvidia-smi`/`torch.cuda` already pass; remaining gate is `import whisperx` + one real stem through large-v3 + align, inspect JSON word timestamps. NFR-1: 5-hr stem in 12–20 min.
- **End-to-end:** start FastAPI, open `localhost` frontend, drop an OBS MKV, label tracks, run → merged transcript with word timestamps; export clean SRT with word-boundary line splits; re-run unchanged session → cache hit skips transcription (SM-4).
