"""ffmpeg extraction — audio stream -> mono / 16 kHz / pcm_s16le WAV (FR-IN-5/6, Appendix A).

WhisperX expects mono 16 kHz PCM. Each extracted file lands in a temp dir and is deleted
after transcription (NFR-3). Use `extract_track` for one stream, or `extracted_audio` as a
context manager that cleans up automatically.
"""

from __future__ import annotations

import contextlib
import subprocess
import tempfile
from pathlib import Path
from typing import Iterator


def extract_track(src: str | Path, track_index: int, dest: str | Path) -> Path:
    """Extract audio stream `track_index` (0-based among audio streams) to `dest` as
    mono/16k/pcm_s16le. Mirrors Appendix A: `-map 0:a:N -ac 1 -ar 16000 -c:a pcm_s16le`."""
    dest = Path(dest)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-map", f"0:a:{track_index}",
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
         str(dest)],
        check=True, capture_output=True,
    )
    return dest


def extract_sample(src: str | Path, track_index: int, dest: str | Path,
                   start: float = 0.0, duration: float = 8.0) -> Path:
    """Extract a short audition clip (FR-TK-3) so the user can confirm a track's label.
    Input-side seek (`-ss` before `-i`) keeps it fast even on multi-hour files."""
    dest = Path(dest)
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start), "-t", str(duration), "-i", str(src),
         "-map", f"0:a:{track_index}",
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
         str(dest)],
        check=True, capture_output=True,
    )
    return dest


@contextlib.contextmanager
def extracted_audio(src: str | Path, track_indices: list[int]) -> Iterator[list[Path]]:
    """Extract the given tracks into a temp dir, yield their paths, delete the dir on exit.

    Honors FR-IN-6 / NFR-3: temp audio is removed once the caller (transcription) is done.
    """
    src = Path(src)
    tmpdir = Path(tempfile.mkdtemp(prefix="whispermerger_"))
    try:
        paths = [
            extract_track(src, ti, tmpdir / f"{src.stem}_track{ti}.wav")
            for ti in track_indices
        ]
        yield paths
    finally:
        for p in tmpdir.glob("*"):
            p.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            tmpdir.rmdir()
