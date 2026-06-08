"""Data model — the locked contract (PRD §9.1 / §9.2).

Everything downstream (merge engine, FastAPI, export, frontend editor) is built against
these types. The segment is the editing unit (NG5 / FR-WD-1); word-level data rides
underneath in `Segment.words` and is preserved through edit -> export, never edited directly.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Word(BaseModel):
    """A single word with acoustic-aligned timestamps (wav2vec2 forced alignment)."""

    word: str
    start: float
    end: float


class Segment(BaseModel):
    """One transcript line. The display + editing unit; words ride underneath untouched."""

    speaker: str
    start: float
    end: float
    text: str
    words: list[Word] = Field(default_factory=list)
    # Conflict markers read by the editor for highlighting/filtering, e.g.
    # ["overlap"], ["backchannel"], ["bleed_suspected", "preferred"].
    flags: list[str] = Field(default_factory=list)


class Track(BaseModel):
    """One audio stream within an input source. Audio files have exactly one track;
    video files expose one track per audio stream (§9.1)."""

    track_index: int
    speaker: str
    selected: bool = True


class InputSource(BaseModel):
    """One input file. The same structure drives audio files and multi-track video (§9.1)."""

    filename: str
    duration: float
    tracks: list[Track] = Field(default_factory=list)
    sequence: int = 0  # position when several files are merged in order (UC2)


class MergedOutput(BaseModel):
    """The merge engine's output and the export module's input (§9.2)."""

    version: str = "1.0"
    sources: list[InputSource] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)
