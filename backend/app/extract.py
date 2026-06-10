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


def extract_tracks(src: str | Path, track_indices: list[int], dest_dir: str | Path) -> dict[int, Path]:
    """Extract several audio tracks in ONE ffmpeg pass (reads the source file once) to
    mono/16k/pcm_s16le. Returns {track_index: wav_path}. Much faster than one read per track
    on large multi-track files."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    args = ["ffmpeg", "-y", "-i", str(src)]
    outs: dict[int, Path] = {}
    for ti in track_indices:
        dest = dest_dir / f"track_{ti}.wav"
        args += ["-map", f"0:a:{ti}", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dest)]
        outs[ti] = dest
    subprocess.run(args, check=True, capture_output=True)
    return outs


def concat_wavs(wav_paths: list[str | Path], dest: str | Path) -> tuple[float, list[float]]:
    """Concatenate mono/16k wavs (in order) into one combined wav. Returns
    (total_duration, boundaries) where boundaries[i] is the start time of file i."""
    import numpy as np
    import soundfile as sf

    parts, boundaries, total, sr = [], [], 0.0, 16000
    for p in wav_paths:
        data, sr = sf.read(str(p), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        boundaries.append(total)
        total += len(data) / sr
        parts.append(data)
    combined = np.concatenate(parts) if parts else np.zeros(0, dtype="float32")
    sf.write(str(dest), combined, sr, subtype="PCM_16")
    return total, boundaries


def extract_clip(src: str | Path, dest: str | Path, start: float, duration: float,
                 num_streams: int) -> Path:
    """Extract a short preview clip starting at `start` (a downmix of all audio streams so
    you hear the whole moment). Input-side seek (`-ss` before `-i`) is instant even on a
    50+ GB / multi-hour file — it never reads the whole container."""
    dest = Path(dest)
    if num_streams <= 1:
        amap = ["-map", "0:a:0"]
    else:
        inputs = "".join(f"[0:a:{i}]" for i in range(num_streams))
        amap = ["-filter_complex", f"{inputs}amix=inputs={num_streams}:duration=longest[a]",
                "-map", "[a]"]
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start), "-t", str(duration), "-i", str(src),
         *amap, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dest)],
        check=True, capture_output=True,
    )
    return dest


def compute_peaks(wav_path: str | Path, buckets: int = 2000) -> list[float]:
    """Downsample a mono/16k wav into `buckets` per-bucket max-abs amplitudes, normalized to
    [0, 1]. Read bucket-by-bucket so a multi-hour track never lands fully in memory. Feeds the
    frontend waveform — the browser draws this instead of downloading the audio."""
    import numpy as np
    import soundfile as sf

    with sf.SoundFile(str(wav_path)) as f:
        total = len(f)
        if total == 0:
            return []
        buckets = max(1, min(buckets, total))
        step = total / buckets
        peaks: list[float] = []
        for i in range(buckets):
            start = int(i * step)
            end = total if i == buckets - 1 else int((i + 1) * step)
            f.seek(start)
            block = f.read(max(1, end - start), dtype="float32", always_2d=False)
            if getattr(block, "ndim", 1) > 1:
                block = block.mean(axis=1)
            peaks.append(float(np.abs(block).max()) if len(block) else 0.0)
    scale = max(peaks) or 1.0
    return [round(p / scale, 4) for p in peaks]


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
