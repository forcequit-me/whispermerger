"""Export — MergedOutput -> TXT / SRT / CSV / JSON (FR-EX-*).

Server-side and single-sourced from MergedOutput, so SRT line splitting uses the actual
word timestamps (FR-WD-3 / FR-EX-2) rather than estimating from segment boundaries. The
frontend just requests a format and downloads.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

from .models import MergedOutput, Segment, Word


@dataclass
class ExportOptions:
    apply_limits: bool = True     # off -> one SRT cue per segment, no line splitting
    chars_per_line: int = 42      # FR-UI-4 CPL
    max_lines: int = 2            # FR-UI-4 max lines per subtitle
    min_duration: float = 1.0     # FR-UI-4 minimum cue duration (s)
    show_speakers: bool = True    # FR-UI-5 Show Speakers
    pad_seconds: float = 0.0      # breathing room: extend each segment start/end (video editing)


def pad_segments(merged: MergedOutput, pad: float) -> MergedOutput:
    """Return a copy of `merged` with every segment start pulled `pad`s earlier and end pushed
    `pad`s later (start clamped >= 0). Render-time only — words are left untouched, so the
    source transcript stays clean and SRT word-splitting still uses real word timings."""
    if not pad:
        return merged
    segs = [seg.model_copy(update={"start": max(0.0, seg.start - pad), "end": seg.end + pad})
            for seg in merged.segments]
    return merged.model_copy(update={"segments": segs})


# --- JSON / TXT / CSV ------------------------------------------------------------------

def to_json(merged: MergedOutput) -> str:
    return merged.model_dump_json(indent=2)


def to_txt(merged: MergedOutput, opts: ExportOptions | None = None) -> str:
    opts = opts or ExportOptions()
    lines = []
    for seg in merged.segments:
        body = f"{seg.speaker}: {seg.text}" if opts.show_speakers else seg.text
        # start + end timing in the transcript
        lines.append(f"[{_timecode(seg.start)} --> {_timecode(seg.end)}] {body}")
    return "\n".join(lines)


def to_csv(merged: MergedOutput) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["start", "end", "speaker", "text", "flags"])
    for seg in merged.segments:
        w.writerow([seg.start, seg.end, seg.speaker, seg.text, "|".join(seg.flags)])
    return buf.getvalue()


def export_all(merged: MergedOutput, out_dir: Path, basename: str,
               opts: ExportOptions | None = None,
               formats: list[str] | None = None) -> list[Path]:
    """Write the chosen formats as <basename>.{ext} into out_dir (batch export).

    `formats` selects which of srt/txt/csv/json to write; None (default) writes all four.
    """
    opts = opts or ExportOptions()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    renderers = {
        "srt": lambda: to_srt(merged, opts),
        "txt": lambda: to_txt(merged, opts),
        "csv": lambda: to_csv(merged),
        "json": lambda: to_json(merged),
    }
    chosen = [f for f in (formats or renderers) if f in renderers]
    written = []
    for ext in chosen:
        path = out_dir / f"{basename}.{ext}"
        path.write_text(renderers[ext](), encoding="utf-8")
        written.append(path)
    return written


# --- SRT with word-level line splitting ------------------------------------------------

def to_srt(merged: MergedOutput, opts: ExportOptions | None = None) -> str:
    opts = opts or ExportOptions()
    blocks: list[str] = []
    index = 1
    for seg in merged.segments:
        for start, end, text in _segment_cues(seg, opts):
            blocks.append(
                f"{index}\n{_timecode(start)} --> {_timecode(end)}\n{text}\n"
            )
            index += 1
    return "\n".join(blocks)


def _segment_cues(seg: Segment, opts: ExportOptions):
    """Yield (start, end, text) cues for a segment, splitting on real word timing.

    With word data, words are packed into lines (<= CPL) and grouped into cues of <=
    max_lines lines; each cue's timing comes from its first/last word. Without word data,
    the text is wrapped and time is interpolated across the segment span.
    """
    speaker = f"{seg.speaker}: " if opts.show_speakers else ""

    # Limits off: one cue per segment, whole text, segment timing — no splitting.
    if not opts.apply_limits:
        yield seg.start, seg.end, _prefix(speaker, seg.text)
        return

    if seg.words:
        line_groups = _pack_words(seg.words, opts.chars_per_line)
        cues = _group_lines(line_groups, opts.max_lines)
        for words_per_cue in cues:
            flat = [w for line in words_per_cue for w in line]
            start, end = flat[0].start, flat[-1].end
            text = "\n".join(" ".join(w.word.strip() for w in line) for line in words_per_cue)
            yield start, _enforce_min(start, end, opts.min_duration), _prefix(speaker, text)
        return

    # Fallback: no word timings — wrap text, interpolate time evenly.
    text_lines = _wrap_text(seg.text, opts.chars_per_line)
    cues = _chunk(text_lines, opts.max_lines)
    span = (seg.end - seg.start) / max(len(cues), 1)
    for i, cue_lines in enumerate(cues):
        start = seg.start + i * span
        end = _enforce_min(start, start + span, opts.min_duration)
        yield start, end, _prefix(speaker, "\n".join(cue_lines))


def _prefix(speaker: str, text: str) -> str:
    """Attach the speaker label to the first line only."""
    if not speaker:
        return text
    first, _, rest = text.partition("\n")
    return f"{speaker}{first}" + (f"\n{rest}" if rest else "")


def _pack_words(words: list[Word], cpl: int) -> list[list[Word]]:
    """Greedily pack words into lines no wider than `cpl` characters."""
    lines: list[list[Word]] = []
    cur: list[Word] = []
    cur_len = 0
    for w in words:
        token = w.word.strip()
        add = len(token) + (1 if cur else 0)
        if cur and cur_len + add > cpl:
            lines.append(cur)
            cur, cur_len = [w], len(token)
        else:
            cur.append(w)
            cur_len += add
    if cur:
        lines.append(cur)
    return lines


def _group_lines(lines: list, max_lines: int) -> list[list]:
    return _chunk(lines, max_lines)


def _chunk(items: list, size: int) -> list[list]:
    if not items:
        return [[]]
    return [items[i:i + size] for i in range(0, len(items), max(size, 1))]


def _wrap_text(text: str, cpl: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        candidate = f"{cur} {word}".strip()
        if cur and len(candidate) > cpl:
            lines.append(cur)
            cur = word
        else:
            cur = candidate
    if cur:
        lines.append(cur)
    return lines or [""]


def _enforce_min(start: float, end: float, min_duration: float) -> float:
    return end if (end - start) >= min_duration else start + min_duration


def _timecode(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
