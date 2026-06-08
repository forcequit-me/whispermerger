"""Pipeline — extract -> transcribe -> merge (PRD §11.1, FR-TR-5 auto-merge).

Ties the engine modules into one call. `transcribe_fn` is injected (in production it's
`transcribe.transcribe_track`; in tests a synthetic stub) so the whole flow is GPU-independent
to test. Offsets: tracks of one file share that file's base offset (concurrent, UC1); each
sequential file is offset by the cumulative duration of files before it (UC2).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .extract import extracted_audio
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

    base = 0.0
    done = 0
    for p in prepared:
        sources.append(p.input)
        selected = [t for t in p.input.tracks if t.selected]
        with extracted_audio(p.path, [t.track_index for t in selected]) as wavs:
            for track, wav in zip(selected, wavs):
                if progress_cb:
                    progress_cb(done / total, f"transcribing {p.input.filename} [{track.speaker}]")
                outputs.append(transcribe_fn(str(wav)))
                labels.append(track.speaker)
                offsets.append(base)
                done += 1
        base += p.input.duration

    if progress_cb:
        progress_cb(0.99, "merging")
    return merge(outputs, labels, offsets, options=options, sources=sources)
