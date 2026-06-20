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
- **Batch "studio"** — queue multiple projects at once and run them unattended. Set **global
  output defaults** (and override per project): **merge a project's files into one transcript** or
  emit **one transcript per file**, and tick **which formats** to write (SRT/TXT/CSV/JSON). Each
  project exports to its own subfolder — or, with *one transcript per file* + *next to source*,
  each transcript lands **beside its own source file** (nested folders preserved). Add files
  individually **or point at a folder** to pull in every audio/video file underneath (subfolders
  included). **Extraction is prefetched** — the next file's audio is pulled while the GPU transcribes
  the current one, so the card never idles between files. A per-project card shows live status +
  progress + elapsed; one-transcript-per-file projects expand to **a row per file** with its own
  status, **live track N/M progress** (for multi-track files), per-file elapsed (which counts the
  ffmpeg **extraction**, not just transcription, so a cache-hit re-run still reflects real work),
  and Cancel/Restart. The file list **keeps its scroll position** as it refreshes, so a long queue
  doesn't snap back to the top every second. No babysitting.
- **Language** — transcription **auto-detects per file** by default; a **Language** dropdown (in
  Batch and Transcribe & Merge) can force a specific language, which skips detection and avoids the
  first-use download of that language's wav2vec2 alignment model.
- **Cancel & restart, per item** — cancel the **whole batch**, a **single project** (the queue
  continues to the next), or a **single file** inside a one-transcript-per-file project (the other
  files keep going) — all cooperative, stopping at the next boundary so the in-flight file finishes.
  **Restart** any finished project or file: while the batch is still running it re-queues to run when
  the GPU frees up; once the batch is done it starts immediately. Caching makes unchanged files
  instant. Single Transcribe/Label jobs are cancelable too.
- **Pause & resume** — pause the **whole batch** or a **single Transcribe & Merge job** mid-run. The
  file currently transcribing finishes, then the model is **unloaded from the GPU** (VRAM returns to
  ~0, so the card is free for other work while you're paused) and the queue halts. **Resume** reloads
  the model and continues exactly where it left off. **The clocks freeze while paused** — elapsed,
  per-file "done in Xs", and the read-vs-GPU split all exclude paused time, so the timings stay
  honest no matter how long you step away. The status line spells out which stage you're in
  ("Pausing… (current file finishes)" → "Paused · GPU released"). The trade for freeing the card is
  a one-time model reload (a few seconds up to ~30 s) on resume.
- **Single-track labeling (v2)** — for one mixed track with multiple speakers (interview,
  podcast, collab): silero-VAD auto-detects speech on a WaveSurfer waveform, you assign regions
  to speakers (click / shift-click a run / ctrl-click; number keys; bulk-assign; duplicate a
  region onto another speaker for talk-over), then transcribe per speaker. Add several files to
  sequence them into one combined waveform. Single-track and split-lane (one row per speaker)
  views. **Horizontal zoom** (out = whole track in view) + **vertical/amplitude zoom** (for quiet
  audio) + optional **region padding** (widen detected speech a touch so words aren't clipped).
  Overlapping speech is supported and flagged.
- **Editor** — inline edit speaker/text/timing (text cells **wrap and auto-grow** so long lines
  stay fully visible instead of running off the edge), global speaker rename, **per-speaker
  show/hide toggle** (hides a speaker's lines from the table _and_ from every export, non-
  destructively — toggle them back anytime) alongside the destructive **remove a speaker and all
  their lines**, add/remove lines, keyboard shortcuts, conflict highlighting (review tags in an
  auto-hiding **Notes** column — present only when a line is flagged), per-line audio
  preview with a **▶/⏸ toggle** (play a line, click again to pause/resume — one clip at a time)
  or drag a jump bar; instant input-seek clips, works on huge files.
  **Spacebar** toggles play/pause on whatever you last played (unless you're typing in a field).
- **Waveform + scrubbing in every editor** — drawn from server-computed downsampled peaks
  (`/api/peaks`), so the browser never downloads full audio (scales to 100+ GB files). Click to
  play from any point; **horizontal zoom** for precision (zoomed all the way out shows the whole
  file) and **vertical/amplitude zoom** to read quiet audio. In *Edit Transcript*, *Attach audio*
  to a loaded transcript to enable the waveform + ▶ for verification.
- **Edit Transcript loads any of our formats** — drop/choose/paste a WhisperMerger **JSON, SRT,
  TXT, or CSV** (format auto-detected); JSON round-trips word-level timing, the others import as
  plain segments. Standard SRT from other tools loads too (the parser is generic).
- **Export** — SRT / TXT / CSV / JSON, all carrying start+end timing; SRT splits lines on real
  word timestamps (toggleable subtitle limits). **Pad (s)** extends each segment's start/end by N
  seconds (render-time only — source stays clean) so cut points aren't abrupt in a video editor.
  Speakers toggled off in the editor are dropped from whichever format you export. Files are named
  **`<name> Transcript (M-D-YYYY h-mm AMPM)`** so they're self-describing (batch auto-exports use it;
  the interactive Export filename is pre-filled with it and editable).
- **Caching** — WhisperX output is cached by file hash + model + alignment + language, so re-runs
  skip transcription (changing the language re-runs instead of returning a stale result). A **Clear
  cache** button in the header wipes it on demand, so you can force a cold-GPU run for benchmarking.
- **Progress that means something** — every processing tab shows a progress bar with the current
  stage (Transcribe & Merge has an Extract → Transcribe → Merge stepper; Batch shows an overall bar
  plus a card per project; Label shows job progress) and total elapsed wall time. Progress is
  **weighted by audio length × track count**, not a naive file/track count — so a long recording
  moves the bar more than a short clip and it no longer lurches. The batch's overall line spells out
  what the % means — **done / total audio** (e.g. `2h 44m / 7h 24m audio`) — so the bar has a
  concrete denominator. Polling is hang-proofed: if the server goes briefly unresponsive (e.g.
  loading ~3 GB of models on the first run) the UI keeps waiting and recovers instead of freezing.
- **Storage-clean** — extracted audio is deleted right after use, and the working dir is wiped on
  startup; only the small transcription cache persists. Nothing piles up.
- **One-click start/stop** — `Startup.bat` launches the server and opens the UI; `Shutdown.bat`
  (or the **Shutdown** button at the top-right of the header) stops it.
- **DAW-style UI** — dark "workbench" theme, full-width waveforms, monospace timecodes, inline-SVG
  icons (no emoji), big drop/choose entry zones per tab, and a keyboard-accessible (WCAG AA focus
  rings, reduced-motion, colourblind-safe lanes) layout. Design system captured in
  [`PRODUCT.md`](PRODUCT.md) + [`DESIGN.md`](DESIGN.md).

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

# run the app (or just double-click Startup.bat, which does this + opens the browser)
cd backend
python -m uvicorn app.main:app --port 8756
```

Then open **http://localhost:8756** (Startup.bat opens it for you). Stop with `Shutdown.bat` or
the **Shutdown** button at the top-right of the header. First transcription downloads the models
(~3 GB) and loads them into VRAM (~30–60 s) before running — the progress bar keeps waiting through
this one-time load rather than looking frozen.

**Performance (real-world):** a 56 GB / 1h51m OBS recording with **4 audio tracks** transcribed
end-to-end (extract → transcribe+align → merge) in **~10 minutes** on an RTX 3080 Ti — ~7.4 h of
track-audio, ~11,860 words. Comfortably beats the 12–20 min / 5 h target. In a batch the next file's
audio is **prefetched** while the GPU works, so it doesn't idle between files; GPU throughput is
tunable via **`WHISPERMERGER_BATCH_SIZE`** (default 16, which `Startup.bat` also ships — on a 12 GB
card, 24 spilled ~1–2 GB into shared system memory, which runs at PCIe speed and *slowed* the run;
keep 16 unless Task Manager shows 0 "Shared GPU memory" in use).

**Where the time actually goes:** prefetch only hides extraction when reading the source is *faster*
than transcribing it. Pulling audio out of an 80–100 GB video means ffmpeg streams ~the whole file,
so if your source files live on a slow medium — an external HDD, a network share / NAS — reading them
can **dominate** and the GPU idles waiting for the next file (a batch can be **read-bound**, not
GPU-bound; on a fast local SSD it's usually the GPU). The Batch UI shows the split so you don't have
to guess: each file row reads `read 5m 48s · gpu 2m 34s`, and the overall stats tally it live
(`1h 12m reading/extracting · 38m on GPU so far`) — the bigger number is the bottleneck. The same
split also prints to the server console as a scrollback record (`[timing] <file>: waited … · transcribed …`).

## Using it

- **Batch** (default tab) — *New project* → name it → *Add files / folder* (native picker — pick
  files, or a whole folder scanned recursively for audio/video) → order
  with ↑/↓ → optionally name speakers → repeat for more projects → choose a single output
  folder or "next to source files" → set the **output defaults** (merge vs one-transcript-per-file,
  and which formats; override per project as needed) → **Run batch**. Each project writes to
  `<output>/<project>/` — either `<project>.{formats}` (merge) or one `<recording>.{formats}` per
  file. A failed project is skipped, not fatal.
- **Transcribe & Merge** — one-off: pick files, label tracks, run, edit, export.
- **Label Track** — single-track files: add one (or several to sequence), auto-detected speech
  appears on a waveform, label regions per speaker, then transcribe into one transcript.
- **Edit Transcript** — load a WhisperMerger transcript in any export format (`.json` / `.srt` /
  `.txt` / `.csv`, or paste it) to tweak and re-export (e.g. a batch result).

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
  pipeline.py   # extract -> transcribe -> merge, prefetched (merge + per-file drivers)
  jobs.py       # single-run job registry + progress
  batch.py      # sequential queue over projects (the studio); per-project + per-file status/cancel/restart
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
