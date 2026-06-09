# WhisperMerger

Local desktop app that turns raw recordings into accurate, speaker-labeled transcripts with
**word-level** timestamps, via WhisperX on a local GPU. A FastAPI backend drives WhisperX
(large-v3 + wav2vec2 forced alignment) and a pure-Python merge engine; a vanilla HTML/JS
frontend handles input, labeling, editing, and export. Runs fully offline.

- **Setup:** [`Docs/SETUP.md`](Docs/SETUP.md)
- **Product spec:** [`Docs/whispermerger-prd.md`](Docs/whispermerger-prd.md)
- **Build plan:** [`Docs/whispermerger-build-plan.md`](Docs/whispermerger-build-plan.md)

## Features

- **Word-level timestamps** — WhisperX + wav2vec2 alignment, accurate enough to navigate
  multi-hour footage (segments stay the editing unit; words ride underneath and export).
- **Reads files in place** — a native file picker selects recordings; nothing is uploaded or
  copied, so 100–200 GB OBS files work (ffmpeg pulls only the audio streams it needs).
- **Any file type / any track count** — `.mkv/.mov/.mp4/.wav/.mp3/.flac`, with 1, 2, or N
  audio streams per file, mixable in one project. Tracks default to Speaker 1, 2, … (editable).
- **Merge engine** — interleaves speakers by time and flags conflicts: backchannel
  ("yeah"/"uh-huh"), genuine overlap, and mic bleed. Conflict detection is toggleable.
- **Batch "studio"** — queue multiple projects at once; each project's session files merge
  sequentially into one transcript and auto-export to its own subfolder. No babysitting.
- **Single-track labeling (v2)** — for one mixed track with multiple speakers (interview,
  podcast, collab): silero-VAD auto-detects speech on a WaveSurfer waveform, you assign regions
  to speakers (click / shift-click a run / ctrl-click; number keys; bulk-assign; duplicate a
  region onto another speaker for talk-over), then transcribe per speaker. Add several files to
  sequence them into one combined waveform. Single-track and split-lane (one row per speaker)
  views. Overlapping speech is supported and flagged.
- **Editor** — inline edit speaker/text/timing, global speaker rename, remove a speaker and
  all their lines, add/remove lines, keyboard shortcuts, conflict highlighting, per-line audio
  preview (▶ a line or drag a jump bar — instant input-seek clips, works on huge files).
- **Export** — SRT / TXT / CSV / JSON, all carrying start+end timing; SRT splits lines on real
  word timestamps (toggleable subtitle limits).
- **Caching** — WhisperX output is cached by file hash + model, so re-runs skip transcription.
- **Stage-aware progress + timing** — jobs and batches report the current stage (Extracting ·
  Transcribing · Merging, per track) and total elapsed wall time, so a long run never looks stuck.
- **Storage-clean** — extracted audio is deleted right after use, and the working dir is wiped on
  startup; only the small transcription cache persists. Nothing piles up.

## Requirements

Native Windows + an NVIDIA GPU (built/tested on an RTX 3080 Ti, 12 GB), Python 3.12, and
ffmpeg/ffprobe on PATH. See [`Docs/SETUP.md`](Docs/SETUP.md) for the exact venv + pinned
dependencies (torch 2.6.0+cu124, whisperx 3.4.2, and the torch-2.6 compatibility pins).

## Quickstart

```powershell
# one-time: create venv + install (see Docs/SETUP.md for details)
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r backend/requirements.txt -c backend/constraints.txt

# run the app
cd backend
python -m uvicorn app.main:app --port 8756
```

Then open **http://localhost:8756**. First transcription downloads the models (~3 GB) and
loads them into VRAM (~30–60 s) before running.

**Performance (real-world):** a 56 GB / 1h51m OBS recording with **4 audio tracks** transcribed
end-to-end (extract → transcribe+align → merge) in **~10 minutes** on an RTX 3080 Ti — ~7.4 h of
track-audio, ~11,860 words. Comfortably beats the 12–20 min / 5 h target.

## Using it

- **Batch** (default tab) — *New project* → name it → *Add files* (native picker) → order
  with ↑/↓ → optionally name speakers → repeat for more projects → choose a single output
  folder or "next to source files" → **Run batch**. Each project writes
  `<output>/<project>/<project>.{srt,txt,csv,json}`. A failed project is skipped, not fatal.
- **Transcribe & Merge** — one-off: pick files, label tracks, run, edit, export.
- **Label Track** — single-track files: add one (or several to sequence), auto-detected speech
  appears on a waveform, label regions per speaker, then transcribe into one transcript.
- **Edit Transcript** — load a WhisperMerger `.json` to tweak and re-export (e.g. a batch result).

## Architecture

```
backend/app/
  models.py     # pydantic contract: Word/Segment/Track/InputSource/MergedOutput
  probe.py      # ffprobe: enumerate audio streams + duration
  extract.py    # ffmpeg: stream -> mono/16k/pcm_s16le (+ audition samples)
  transcribe.py # WhisperX large-v3 + wav2vec2 align (GPU), cache-aware
  merge.py      # pure-python merge + conflict hierarchy
  cache.py      # SHA-256-keyed WhisperX-JSON cache
  export.py     # MergedOutput -> SRT/TXT/CSV/JSON (+ export_all for batch)
  pipeline.py   # extract -> transcribe -> merge (one project)
  jobs.py       # single-run job registry + progress
  batch.py      # sequential queue over projects (the studio)
  diarize.py    # v2: silero-VAD + per-speaker region transcription + timeline remap
  main.py       # FastAPI app + static frontend mount
  routes/api.py # HTTP API
frontend/       # vanilla HTML/JS/CSS: Batch, Transcribe & Merge, Label Track, Edit Transcript
  labeltrack.js #   v2 waveform labeling (ESM); vendor/ holds WaveSurfer
backend/tests/  # pytest suite (GPU-independent: stubbed transcribe + real ffmpeg)
```

## Tests

```powershell
cd backend; python -m pytest -q
```

The suite is GPU-independent — it stubs transcription and exercises the merge conflict
hierarchy, probe/extract, cache, export (incl. word-level SRT), pipeline, batch, and the API.
The GPU path is verified separately with `python backend/smoke_test.py`.
