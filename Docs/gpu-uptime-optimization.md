# Handoff — GPU uptime / batch speed optimization

**Status:** #1 (prefetch) + #2 (batch_size knob) **implemented** 2026-06-19. #3 skipped as planned.

**What shipped:**
- Prefetch depth-1 via `extract.extract_ahead` (generic iterator). `pipeline.run_merge_pipeline`
  consumes it (single-job + multi-file merge projects); new `pipeline.run_per_file_pipeline` drives
  it across a project's files for the per-file batch case (`batch._process_project`). Each file
  extracts into its own temp subdir so prefetched WAVs don't collide; cleanup via `shutil.rmtree`.
- `transcribe.BATCH_SIZE` = `int(os.environ.get("WHISPERMERGER_BATCH_SIZE", "16"))`. **Still default
  16** — tune on the real 3080 Ti (try 24/32, largest with no OOM) and record the value here.
- Cancel: prefetch stops promptly via `is_cancelled` threaded through both pipelines.

Original analysis below (kept for reference).

**Goal:** keep the GPU busy a higher fraction of batch wall time. Today the GPU idles while
each file's audio is extracted; we want to overlap that CPU/disk work with transcription.

---

## What's already good (don't re-do these)

- **Models stay resident** across the whole batch. `transcribe._load_model` (lru_cache size 1)
  and `_load_align` (lru_cache size 4) load large-v3 + the wav2vec2 align models once and reuse
  them — no reload between files or projects.
- **Single-pass extraction.** `extract.extract_tracks` pulls *all* of a file's selected audio
  tracks in one ffmpeg read (one `-i`, multiple `-map`). The old "re-reads per track" note in
  `STATUS.md` (validation section) is stale — current code is single-pass.
- **Cache short-circuits the GPU.** `cache_key(file, model, align, language)` → a hit returns the
  stored transcript with zero GPU work. Re-runs are effectively free for unchanged files.

## The problem — serial pipeline, GPU idle during extraction

`pipeline.run_merge_pipeline` (and the batch per-file loop in `batch._process_project`) runs
**strictly serially per file**:

```
extract audio (ffmpeg, CPU/disk)  →  transcribe + align (GPU)  →  merge/export (CPU)
```

While a file is being extracted, the GPU does nothing. On the validated 56 GB / 4-track file:
~2.6 min extraction vs ~7.5 min GPU → GPU idle ≈ a quarter of wall time. In a folder batch of
many files, that gap repeats **once per file**.

**Hard constraint:** one GPU → only one transcription at a time. We cannot parallelize the GPU
itself. The only lever is removing the CPU/disk stalls *around* GPU work.

---

## Plan (priority order)

### 1. Prefetch / overlap extraction with transcription  ← the real win
While the GPU transcribes file N, extract file N+1's audio in a background thread so it's ready
the instant N finishes. Converts "extract → wait → transcribe → extract → wait …" into a
near-continuous GPU stream.

- **Effort:** moderate. **Risk:** medium (concurrency + temp-file lifecycle + cancel interplay).
- **Expected:** recovers most of the per-file extraction idle on extraction-heavy batches.
- **Disk cost:** up to two files' extracted WAVs on disk at once (mono/16k PCM ≈ 115 MB per
  track-hour). Use **prefetch depth 1** — don't extract more than one ahead.

**Implementation sketch:**
- Add a small reusable "extract-ahead" iterator: given ordered sources + an extract fn, submit
  the *next* file's extraction to a `ThreadPoolExecutor(max_workers=1)` before yielding the
  current `(src, wavs)`. Consumer transcribes, then `unlink`s that file's WAVs (as today).
- Two call sites benefit:
  - `pipeline.run_merge_pipeline` — prefetch the next *file* while transcribing the current
    file's tracks (helps multi-file merge projects).
  - `batch._process_project` per-file branch — each file is its own `run_merge_pipeline([src])`,
    so prefetch across files at the batch level (helps folder / one-transcript-per-file runs,
    the common case).
- **Cancel integration (already built):** the producer must check `is_cancelled()` before
  extracting the next file; on cancel, stop the producer and clean its temp dir. Consumer cancel
  checkpoints already exist via `batch._guard` / `jobs.Job._set_progress`.
- **Error isolation:** if a file's extraction throws, surface it as *that file's* error and keep
  the queue going (matches existing per-project error isolation) — don't let one bad file kill the
  prefetch thread silently.
- Keep all temp WAVs in the existing per-run tmpdir so cleanup stays centralized.

### 2. Tune transcription batch size  ← cheap, needs one real GPU test
`transcribe.transcribe_track` calls `model.transcribe(audio, batch_size=16, ...)`. A 12 GB
3080 Ti with fp16 large-v3 can likely go higher (try 24, 32) for more GPU throughput per pass.

- **Effort:** tiny (make it `BATCH_SIZE = int(os.environ.get("WHISPERMERGER_BATCH_SIZE", "16"))`).
- **Risk:** OOM — **must** be validated on a real run; pick the largest value that stays
  comfortably under 12 GB VRAM. Cannot be determined without the GPU.

### 3. Trim small CPU gaps  ← minor, probably skip
`whisperx.load_audio` re-decodes the extracted WAV once more (a second, small ffmpeg pass) before
transcription. Foldable, but low payoff. Defer.

**Recommendation when picking this up:** do **#1 + #2**. Skip #3.

---

## How to measure (define success before/after)

- **Quick:** run `nvidia-smi dmon -s u` during a multi-file batch and watch for util dropping to
  ~0% at file boundaries (those drops are the idle we're removing).
- **Precise:** sum per-stage wall time. `run_merge_pipeline` already emits stage messages
  (Extracting / Transcribing / Merging) — log their durations and compare extract-vs-GPU totals
  before and after.
- **Success:** GPU util stays high across file boundaries; total batch wall time drops on
  extraction-heavy workloads. #2 success: lower transcription time with no OOM.

## Files in scope
- `backend/app/pipeline.py` — `run_merge_pipeline` (prefetch across files).
- `backend/app/batch.py` — `_process_project` per-file branch (prefetch across files); respect
  the existing `Cancelled` / `is_cancelled` plumbing.
- `backend/app/transcribe.py` — `batch_size` knob (#2).
- `backend/app/extract.py` — `extract_tracks` already single-pass; reuse as the extract fn.
- Tests: extend `test_pipeline.py` / `test_batch.py` (stubs are GPU-free) to assert prefetch
  produces identical output + cleans up temp files; cancel still stops promptly.

## Open questions
- Optimal `batch_size` for 12 GB / large-v3 fp16 (real-GPU test).
- Is prefetch depth >1 ever worth it for many tiny files? (Likely not — disk + diminishing
  returns. Start at depth 1.)
