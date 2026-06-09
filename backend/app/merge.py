"""Merge engine — pure Python, the core of WhisperMerger (PRD §8.4).

Takes per-track WhisperX outputs + speaker labels + per-track time offsets and returns a
single merged, conflict-flagged `MergedOutput`. GPU-independent: no model load, fully
testable with synthetic WhisperX JSON.

Conflict hierarchy applied per pair of time-overlapping segments from *different* speakers:
  FR-MG-1  no overlap          -> interleave by start time (the default; nothing flagged)
  FR-MG-4  bleed               -> near-duplicate text in a small window (mic catching the
                                  other stream); flag `bleed_suspected`, prefer the louder
                                  source via an optional RMS hook (`preferred`)
  FR-MG-2  backchannel         -> short segment fully inside another ("yeah", "right");
                                  flag `backchannel`, render per `BackchannelRender`
  FR-MG-3  genuine overlap     -> multi-second simultaneous speech; flag `overlap`

Precedence note: bleed is checked before backchannel/overlap because near-duplicate text
masquerades as simultaneous speech and must not be mistaken for it.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Sequence

from .models import InputSource, MergedOutput, Segment, Word

# Optional hook: given (speaker, start, end) return RMS loudness of that source's audio in
# that window. Used only to pick the `preferred` side of a suspected bleed (FR-MG-4, P1).
RmsFn = Callable[[str, float, float], float]


class BackchannelRender(str, Enum):
    PARENTHETICAL = "parenthetical"  # keep as its own line, text wrapped in ( )
    SEPARATE = "separate"            # keep as its own line, text unchanged
    SKIP = "skip"                    # drop the backchannel entirely


@dataclass
class MergeOptions:
    detect_conflicts: bool = True           # off -> just interleave by time, no flags
    backchannel_max_words: int = 4          # FR-MG-2: "under N words"
    backchannel_render: BackchannelRender = BackchannelRender.PARENTHETICAL  # OQ-2 default
    overlap_min_seconds: float = 2.0        # FR-MG-3: "several seconds"
    bleed_window_seconds: float = 1.0       # FR-MG-4: "small time window"
    bleed_similarity: float = 0.7           # difflib ratio threshold for "similar text"


def _to_segments(output: dict, speaker: str, offset: float) -> list[Segment]:
    """Parse one WhisperX result dict into Segments, applying the time offset.

    Accepts WhisperX align output where each word uses key `word` (falls back to `text`).
    """
    segments: list[Segment] = []
    for raw in output.get("segments", []):
        words = [
            Word(
                word=w.get("word", w.get("text", "")),
                start=float(w["start"]) + offset,
                end=float(w["end"]) + offset,
            )
            for w in raw.get("words", [])
            if w.get("start") is not None and w.get("end") is not None
        ]
        segments.append(
            Segment(
                speaker=speaker,
                start=float(raw["start"]) + offset,
                end=float(raw["end"]) + offset,
                text=raw.get("text", "").strip(),
                words=words,
            )
        )
    return segments


def _word_count(seg: Segment) -> int:
    return len(seg.words) if seg.words else len(seg.text.split())


def _overlap_seconds(a: Segment, b: Segment) -> float:
    return min(a.end, b.end) - max(a.start, b.start)


def _contains(outer: Segment, inner: Segment) -> bool:
    return outer.start <= inner.start and inner.end <= outer.end


def _similar(a: Segment, b: Segment, threshold: float) -> bool:
    ratio = difflib.SequenceMatcher(None, a.text.lower(), b.text.lower()).ratio()
    return ratio >= threshold


def _add_flag(seg: Segment, flag: str) -> None:
    if flag not in seg.flags:
        seg.flags.append(flag)


def _classify_pair(a: Segment, b: Segment, opts: MergeOptions, rms_fn: Optional[RmsFn]) -> None:
    """Flag a pair of overlapping, different-speaker segments in place per the hierarchy."""
    # FR-MG-4 bleed: near-duplicate text within a small window.
    if (
        abs(a.start - b.start) <= opts.bleed_window_seconds
        and _similar(a, b, opts.bleed_similarity)
    ):
        _add_flag(a, "bleed_suspected")
        _add_flag(b, "bleed_suspected")
        if rms_fn is not None:
            louder = a if rms_fn(a.speaker, a.start, a.end) >= rms_fn(b.speaker, b.start, b.end) else b
            _add_flag(louder, "preferred")
        return

    # FR-MG-2 backchannel: one short segment fully inside the other.
    inner = outer = None
    if _contains(a, b):
        outer, inner = a, b
    elif _contains(b, a):
        outer, inner = b, a
    if inner is not None and _word_count(inner) < opts.backchannel_max_words:
        _add_flag(inner, "backchannel")
        return

    # FR-MG-3 genuine overlap: multi-second simultaneous speech.
    if _overlap_seconds(a, b) >= opts.overlap_min_seconds:
        _add_flag(a, "overlap")
        _add_flag(b, "overlap")


def _apply_backchannel_render(segments: list[Segment], render: BackchannelRender) -> list[Segment]:
    out: list[Segment] = []
    for seg in segments:
        if "backchannel" not in seg.flags:
            out.append(seg)
            continue
        if render is BackchannelRender.SKIP:
            continue
        if render is BackchannelRender.PARENTHETICAL and seg.text:
            seg.text = f"({seg.text})"
        out.append(seg)
    return out


def merge(
    outputs: Sequence[dict],
    labels: Sequence[str],
    offsets: Optional[Sequence[float]] = None,
    *,
    options: Optional[MergeOptions] = None,
    sources: Optional[Sequence[InputSource]] = None,
    rms_fn: Optional[RmsFn] = None,
) -> MergedOutput:
    """Merge per-track WhisperX outputs into one conflict-flagged transcript.

    outputs  : WhisperX result dicts (each with a "segments" list), one per track.
    labels   : speaker name per output (same length as `outputs`).
    offsets  : absolute start offset (s) per output. None -> all zero (concurrent
               multi-track, UC1). For sequential files (UC2) pass cumulative durations so
               each file lands after the previous on the merged timeline.
    options  : conflict thresholds + backchannel render preference.
    sources  : optional InputSource list embedded in the result's `sources`.
    rms_fn   : optional loudness hook used only to pick the `preferred` side of a bleed.
    """
    if len(outputs) != len(labels):
        raise ValueError("outputs and labels must be the same length")
    opts = options or MergeOptions()
    offs = list(offsets) if offsets is not None else [0.0] * len(outputs)
    if len(offs) != len(outputs):
        raise ValueError("offsets must match outputs length")

    # Parse + offset every track, tagging each segment with its source index so we never
    # compare two segments from the same track against each other.
    tagged: list[tuple[int, Segment]] = []
    for i, (out, label, off) in enumerate(zip(outputs, labels, offs)):
        for seg in _to_segments(out, label, off):
            tagged.append((i, seg))

    # Pairwise conflict detection across different tracks only.
    if opts.detect_conflicts:
        for x in range(len(tagged)):
            ix, a = tagged[x]
            for y in range(x + 1, len(tagged)):
                iy, b = tagged[y]
                if ix == iy:
                    continue
                if _overlap_seconds(a, b) > 0:
                    _classify_pair(a, b, opts, rms_fn)

    segments = [seg for _, seg in tagged]
    segments = _apply_backchannel_render(segments, opts.backchannel_render)
    # FR-MG-1: interleave by start time (stable on end for deterministic ties).
    segments.sort(key=lambda s: (s.start, s.end))

    return MergedOutput(
        version="1.0",
        sources=list(sources) if sources is not None else [],
        segments=segments,
    )
