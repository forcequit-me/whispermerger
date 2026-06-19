"""Pipeline — extract -> transcribe -> merge (PRD §11.1, FR-TR-5 auto-merge).

Ties the engine modules into one call. `transcribe_fn` is injected (in production it's
`transcribe.transcribe_track`; in tests a synthetic stub) so the whole flow is GPU-independent
to test. Offsets: tracks of one file share that file's base offset (concurrent, UC1); each
sequential file is offset by the cumulative duration of files before it (UC2).
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .extract import extract_ahead, extract_tracks
from .merge import MergeOptions, merge
from .models import InputSource, MergedOutput

# transcribe_fn(audio_path) -> WhisperX output dict ({"segments": [...]})
TranscribeFn = Callable[[str], dict]
ProgressCb = Callable[[float, str], None]
IsCancelled = Callable[[], bool]
# on_file(index, source, merged) — called once per finished file in run_per_file_pipeline so the
# caller can export it immediately (the next file is already extracting in the background).
OnFile = Callable[[int, "PreparedSource", MergedOutput], None]


@dataclass
class PreparedSource:
    path: str               # real path to the source file on disk
    input: InputSource      # filename, duration, tracks (with selected + speaker), sequence


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
    tmpdir = Path(tempfile.mkdtemp(prefix="wm_pipe_"))
    try:
        if prepared:
            report(0.0, f"Extracting audio · {prepared[0].input.filename}")
        for p, wavs in extract_ahead(prepared, lambda src: _extract_into(tmpdir, src),
                                     is_cancelled=is_cancelled):
            sources.append(p.input)
            for track in (t for t in p.input.tracks if t.selected):
                n = done + 1
                report(done / total, f"Transcribing · {track.speaker} ({n}/{total})")
                outputs.append(transcribe_fn(str(wavs[track.track_index])))
                wavs[track.track_index].unlink(missing_ok=True)   # free it right away
                labels.append(track.speaker)
                offsets.append(base)
                done += 1
            base += p.input.duration

        report(0.99, "Merging tracks…")
        return merge(outputs, labels, offsets, options=options, sources=sources)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_per_file_pipeline(
    prepared: list[PreparedSource],
    transcribe_fn: TranscribeFn,
    on_file: OnFile,
    *,
    options: Optional[MergeOptions] = None,
    progress_cb: Optional[ProgressCb] = None,
    is_cancelled: Optional[IsCancelled] = None,
) -> None:
    """One transcript per source file, with the same cross-file prefetch as run_merge_pipeline:
    while file N transcribes, file N+1 extracts in the background. Each file is merged on its own
    (offset 0) and handed to `on_file(index, source, merged)` so the caller exports it. progress_cb
    reports an overall fraction across all n files."""
    prepared = sorted(prepared, key=lambda p: p.input.sequence)
    n = len(prepared) or 1

    def report(frac: float, msg: str) -> None:
        if progress_cb:
            progress_cb(frac, msg)

    tmpdir = Path(tempfile.mkdtemp(prefix="wm_perfile_"))
    try:
        if prepared:
            report(0.0, f"file 1/{n}: {prepared[0].input.filename} · Extracting audio")
        for i, (p, wavs) in enumerate(extract_ahead(prepared, lambda src: _extract_into(tmpdir, src),
                                                     is_cancelled=is_cancelled)):
            selected = [t for t in p.input.tracks if t.selected]
            outputs: list[dict] = []
            labels: list[str] = []
            for j, track in enumerate(selected):
                report((i + j / max(len(selected), 1)) / n,
                       f"file {i + 1}/{n}: {p.input.filename} · Transcribing {track.speaker}")
                outputs.append(transcribe_fn(str(wavs[track.track_index])))
                wavs[track.track_index].unlink(missing_ok=True)
                labels.append(track.speaker)
            merged = merge(outputs, labels, [0.0] * len(outputs), options=options, sources=[p.input])
            report((i + 1) / n, f"file {i + 1}/{n}: {p.input.filename} · done")
            on_file(i, p, merged)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
