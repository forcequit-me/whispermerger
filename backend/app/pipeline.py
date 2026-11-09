"""Pipeline — extract -> transcribe -> merge (PRD §11.1, FR-TR-5 auto-merge).

Ties the engine modules into one call. `transcribe_fn` is injected (in production it's
`transcribe.transcribe_track`; in tests a synthetic stub) so the whole flow is GPU-independent
to test. Offsets: tracks of one file share that file's base offset (concurrent, UC1); each
sequential file is offset by the cumulative duration of files before it (UC2).
"""

from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .extract import extract_ahead, extract_tracks, loudness_profile, span_rms
from .merge import MergeOptions, merge
from .models import InputSource, MergedOutput
from .transcribe import chunk_progress

# transcribe_fn(audio_path) -> WhisperX output dict ({"segments": [...]})
TranscribeFn = Callable[[str], dict]
ProgressCb = Callable[[float, str], None]
IsCancelled = Callable[[], bool]


class _FileCancelled(Exception):
    """Internal: one file in a per-file run was canceled — skip it, keep the rest going."""


@dataclass
class PreparedSource:
    path: str               # real path to the source file on disk
    input: InputSource      # filename, duration, tracks (with selected + speaker), sequence


def _rms_from(profiles: list):
    """merge()'s loudness hook over per-track profiles; None when none were measured."""
    return (lambda i, s, e: span_rms(profiles[i], s, e)) if profiles else None


def _extract_into(tmpdir: Path, p: "PreparedSource") -> dict:
    """Extract a source's selected tracks (one ffmpeg pass) into a fresh subdir of `tmpdir`.
    A unique subdir per file keeps prefetched files (which coexist on disk) from colliding on
    the shared `track_<n>.wav` names."""
    selected = [t for t in p.input.tracks if t.selected]
    dest = Path(tempfile.mkdtemp(prefix="src_", dir=tmpdir))
    return extract_tracks(p.path, [t.track_index for t in selected], dest)


def run_merge_pipeline(
    prepared: list[PreparedSource],
    transcribe_fn: TranscribeFn,
    *,
    options: Optional[MergeOptions] = None,
    progress_cb: Optional[ProgressCb] = None,
    is_cancelled: Optional[IsCancelled] = None,
) -> MergedOutput:
    prepared = sorted(prepared, key=lambda p: p.input.sequence)
    total = sum(1 for p in prepared for t in p.input.tracks if t.selected) or 1
    # Progress is weighted by audio length × tracks (a track-pass takes ~the file's duration), so
    # the bar tracks real time remaining instead of just counting tracks.
    total_weight = sum(p.input.duration * sum(1 for t in p.input.tracks if t.selected)
                       for p in prepared) or 1.0

    outputs: list[dict] = []
    labels: list[str] = []
    offsets: list[float] = []
    sources: list[InputSource] = []

    def report(frac: float, msg: str) -> None:
        if progress_cb:
            progress_cb(frac, msg)

    # Prefetch: while the GPU transcribes file N, file N+1's audio extracts in the background, so
    # the GPU never waits on ffmpeg/disk between files (the main idle we're removing). Each file's
    # tracks come out in one ffmpeg pass; each temp wav is deleted right after it transcribes.
    base = 0.0
    done = 0
    done_weight = 0.0
    profiles: list = []                        # per-track loudness, for the bleed `preferred` pick
    detect = (options or MergeOptions()).detect_conflicts
    tmpdir = Path(tempfile.mkdtemp(prefix="wm_pipe_"))
    try:
        if prepared:
            report(0.0, f"Extracting audio · {prepared[0].input.filename}")
        for p, wavs in extract_ahead(prepared, lambda src: _extract_into(tmpdir, src),
                                     is_cancelled=is_cancelled):
            sources.append(p.input)
            for track in (t for t in p.input.tracks if t.selected):
                n = done + 1
                label = f"Transcribing track {n}/{total} · {track.speaker}"
                report(done_weight / total_weight, label)

                # ponytail: chunks fill 90% of the track's share; word timing (no progress hook) the rest
                def on_chunk(f: float, w=p.input.duration, d=done_weight, label=label) -> None:
                    report((d + 0.9 * f * w) / total_weight, f"{label} · {int(f * 100)}%")

                with chunk_progress(on_chunk):     # moves the bar while this track transcribes
                    outputs.append(transcribe_fn(str(wavs[track.track_index])))
                if detect:                                         # measure before the wav goes
                    profiles.append(loudness_profile(wavs[track.track_index]))
                wavs[track.track_index].unlink(missing_ok=True)   # free it right away
                labels.append(track.speaker)
                offsets.append(base)
                done += 1
                done_weight += p.input.duration
            base += p.input.duration

        report(0.99, "Merging tracks…")
        return merge(outputs, labels, offsets, options=options, sources=sources,
                     rms_fn=_rms_from(profiles))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_per_file_pipeline(
    prepared: list[PreparedSource],
    transcribe_fn: TranscribeFn,
    indices: list[int],
    *,
    options: Optional[MergeOptions] = None,
    is_cancelled: Optional[IsCancelled] = None,
    on_start: Callable[[int, "PreparedSource", float], None],
    on_track: Callable[[int, int, int, str], None],
    on_done: Callable[[int, "PreparedSource", MergedOutput], None],
    on_error: Callable[[int, "PreparedSource", Exception], None],
    on_skip: Callable[[int, "PreparedSource"], None],
    is_file_cancelled: Callable[[int], bool],
    on_wait: Optional[Callable[[int, "PreparedSource"], None]] = None,
) -> None:
    """Transcribe the given file positions (one transcript each) with depth-1 prefetch: while one
    file transcribes, the next extracts in the background. Per-file lifecycle is reported through
    callbacks (start / per-track / done / error / skip) so the caller tracks and exports each file.
    A per-file cancel (`is_file_cancelled`) skips just that file; a project/batch cancel
    (`is_cancelled`) stops the whole pass. This runs ONE pass over `indices`; a single-file restart
    is just a later call with that file back in `indices`."""
    def cancelled() -> bool:
        return bool(is_cancelled and is_cancelled())

    detect = (options or MergeOptions()).detect_conflicts
    tmpdir = Path(tempfile.mkdtemp(prefix="wm_perfile_"))
    extract_secs: dict[int, float] = {}

    def extract_one(it):
        # Time the real read + ffmpeg pass for this file — even when it runs prefetched in the
        # background behind the previous file's GPU work. This is the file's true pre-GPU cost,
        # not the (near-zero) idle wait the consumer sees once the audio is already prepped.
        t0 = time.perf_counter()
        try:
            wavs = _extract_into(tmpdir, it[1])
        except Exception as exc:     # hand it to the consumer so only this file fails
            return exc
        extract_secs[it[0]] = time.perf_counter() - t0
        return wavs

    try:
        items = [(i, prepared[i]) for i in indices]
        for (i, p), wavs in extract_ahead(items, extract_one,
                                          is_cancelled=cancelled,
                                          on_wait=(lambda it: on_wait(it[0], it[1])) if on_wait else None):
            if cancelled():
                break
            if is_file_cancelled(i):
                on_skip(i, p)
                continue
            if isinstance(wavs, Exception):          # extraction failed — isolate like a transcribe error
                on_error(i, p, wavs)
                continue
            on_start(i, p, extract_secs.get(i, 0.0))
            try:
                selected = [t for t in p.input.tracks if t.selected]
                outputs: list[dict] = []
                labels: list[str] = []
                profiles: list = []
                completed = True
                for j, track in enumerate(selected):
                    if is_file_cancelled(i):
                        raise _FileCancelled()
                    if cancelled():
                        completed = False
                        break
                    on_track(i, j, len(selected), track.speaker)
                    outputs.append(transcribe_fn(str(wavs[track.track_index])))
                    if detect:
                        profiles.append(loudness_profile(wavs[track.track_index]))
                    wavs[track.track_index].unlink(missing_ok=True)
                    labels.append(track.speaker)
                if completed:
                    merged = merge(outputs, labels, [0.0] * len(outputs),
                                   options=options, sources=[p.input], rms_fn=_rms_from(profiles))
                    on_done(i, p, merged)
            except _FileCancelled:
                on_skip(i, p)
            except Exception as exc:                 # per-file isolation — keep the rest going
                if cancelled():                      # e.g. a model download stopped by Cancel
                    raise
                on_error(i, p, exc)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
