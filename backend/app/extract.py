"""ffmpeg extraction — audio stream -> mono / 16 kHz / pcm_s16le WAV (FR-IN-5/6, Appendix A).

WhisperX expects mono 16 kHz PCM. Each extracted file lands in a temp dir and is deleted
after transcription (NFR-3). Use `extract_track` for one stream, or `extracted_audio` as a
context manager that cleans up automatically.
"""

from __future__ import annotations

import contextlib
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterator, Optional


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


def extract_ahead(items: list, extract_one: Callable, *,
                  is_cancelled: Optional[Callable[[], bool]] = None,
                  on_wait: Optional[Callable[[object], None]] = None) -> Iterator[tuple]:
    """Yield ``(item, extract_one(item))`` in order, running the NEXT item's extraction in a
    background thread (prefetch depth 1) while the consumer handles the current one. A single
    worker means at most one item is extracted ahead, so the GPU stays fed without piling up
    disk usage. Stops prefetching once ``is_cancelled()`` is true; an extraction error surfaces
    when its item is reached (per-item isolation — it is not swallowed in the worker thread).

    ``on_wait(item)`` fires the instant an item becomes the consumer's active work — right before
    we block on its extraction. Anything still extracting at that point was not hidden behind the
    previous item, so a per-item timer started here counts the un-prefetched extraction wait plus
    the work that follows, without double-counting the overlap (FR timer accuracy)."""
    if not items:
        return
    is_cancelled = is_cancelled or (lambda: False)
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(extract_one, items[0])
        for i, item in enumerate(items):
            if on_wait:
                on_wait(item)
            result = fut.result()                       # block until this item is extracted
            fut = (pool.submit(extract_one, items[i + 1])
                   if i + 1 < len(items) and not is_cancelled() else None)
            yield item, result


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
                 num_streams: int, track: int | None = None) -> Path:
    """Extract a short preview clip starting at `start`. By default a downmix of all audio
    streams (hear the whole moment); pass `track` to isolate one stream (`-map 0:a:track`) so a
    per-line preview plays only that speaker's mic. Input-side seek (`-ss` before `-i`) is
    instant even on a 50+ GB / multi-hour file — it never reads the whole container."""
    dest = Path(dest)
    if track is not None:
        amap = ["-map", f"0:a:{track}"]
    elif num_streams <= 1:
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


def compute_peaks(wav_path: str | Path, buckets: int = 2000, start: float | None = None,
                  end: float | None = None, normalize: bool = True) -> list[float]:
    """Downsample a mono/16k wav into `buckets` per-bucket max-abs amplitudes, normalized to
    [0, 1]. Read bucket-by-bucket so a multi-hour track never lands fully in memory. Feeds the
    frontend waveform — the browser draws this instead of downloading the audio.
    `start`/`end` (seconds) limit it to a window; `normalize=False` returns raw amplitudes."""
    import numpy as np
    import soundfile as sf

    with sf.SoundFile(str(wav_path)) as f:
        total = len(f)
        lo = max(0, min(total, int((start or 0) * f.samplerate)))
        hi = total if end is None else max(lo, min(total, int(end * f.samplerate)))
        if hi <= lo:
            return []
        buckets = max(1, min(buckets, hi - lo))
        step = (hi - lo) / buckets
        peaks: list[float] = []
        for i in range(buckets):
            s = lo + int(i * step)
            e = hi if i == buckets - 1 else lo + int((i + 1) * step)
            f.seek(s)
            block = f.read(max(1, e - s), dtype="float32", always_2d=False)
            if getattr(block, "ndim", 1) > 1:
                block = block.mean(axis=1)
            peaks.append(float(np.abs(block).max()) if len(block) else 0.0)
    if not normalize:
        return peaks
    scale = max(peaks) or 1.0
    return [round(p / scale, 4) for p in peaks]


LOUDNESS_HOP = 0.05   # seconds per loudness frame


def loudness_profile(wav_path: str | Path, hop: float = LOUDNESS_HOP):
    """Mean-square amplitude per `hop` seconds of a mono/16k wav, read in ~10 s blocks so a
    multi-hour track never lands fully in memory. 20 numbers per second (~0.6 MB per hour), so
    it's kept after the temp wav is deleted and used to compare mics on a suspected bleed."""
    import numpy as np
    import soundfile as sf

    parts = []
    with sf.SoundFile(str(wav_path)) as f:
        n = max(1, int(f.samplerate * hop))
        for block in f.blocks(blocksize=n * 200, dtype="float32", always_2d=False):
            if block.ndim > 1:
                block = block.mean(axis=1)
            starts = np.arange(0, len(block), n)
            sizes = np.diff(np.append(starts, len(block)))
            parts.append(np.add.reduceat(block.astype(np.float64) ** 2, starts) / sizes)
    return np.concatenate(parts) if parts else np.zeros(0)


def span_rms(profile, start: float, end: float, hop: float = LOUDNESS_HOP) -> float:
    """RMS loudness of a loudness_profile between `start` and `end` seconds (0 outside the audio)."""
    import math
    i = max(0, int(start / hop))
    j = max(i + 1, math.ceil(end / hop))
    frames = profile[i:j]
    return float(math.sqrt(frames.mean())) if len(frames) else 0.0


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
