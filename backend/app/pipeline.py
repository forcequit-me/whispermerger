"""Pipeline — extract -> transcribe -> merge (PRD §11.1, FR-TR-5 auto-merge).

Ties the engine modules into one call. `transcribe_fn` is injected (in production it's
`transcribe.transcribe_track`; in tests a synthetic stub) so the whole flow is GPU-independent
to test. Offsets: tracks of one file share that file's base offset (concurrent, UC1); each
sequential file is offset by the cumulative duration of files before it (UC2).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .extract import extract_tracks
from .merge import MergeOptions, merge
from .models import InputSource, MergedOutput

# transcribe_fn(audio_path) -> WhisperX output dict ({"segments": [...]})
TranscribeFn = Callable[[str], dict]
ProgressCb = Callable[[float, str], None]


@dataclass
class PreparedSource:
    path: str               # real path to the source file on disk
    input: InputSource      # filename, duration, tracks (with selected + speaker), sequence


def run_merge_pipeline(
    prepared: list[PreparedSource],
    transcribe_fn: TranscribeFn,
    *,
    options: Optional[MergeOptions] = None,
    progress_cb: Optional[ProgressCb] = None,
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

    # Per file: extract all its selected tracks in ONE pass (one read of the source), then
    # transcribe each and delete its temp wav immediately (storage hygiene). Report each
    # stage so the UI never looks stalled.
    base = 0.0
    done = 0
    tmpdir = Path(tempfile.mkdtemp(prefix="wm_pipe_"))
    try:
        for p in prepared:
            sources.append(p.input)
            selected = [t for t in p.input.tracks if t.selected]
            report(done / total, f"Extracting audio · {p.input.filename} ({len(selected)} track(s))")
            wavs = extract_tracks(p.path, [t.track_index for t in selected], tmpdir)
            for track in selected:
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
        for f in tmpdir.glob("*"):
            f.unlink(missing_ok=True)
        try:
            tmpdir.rmdir()
        except OSError:
            pass
