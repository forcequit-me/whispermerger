"""v2 single-track manual diarization — headless engine (PRD §11.4, Phase A).

For recordings with no stems (one mixed track, multiple speakers). The user labels speech
regions per speaker (in the UI, Phase B); here we:
  1. detect_speech()      — VAD over the track → candidate speech regions (seed the UI)
  2. transcribe_regions() — per speaker, stitch only their regions into one audio, transcribe,
     then remap word timestamps back onto the ORIGINAL timeline, and merge speakers.

The remap is the crux: a speaker's regions are concatenated, so a word at time t in the
concatenated audio must be mapped back to where that slice actually sits in the source.
GPU-independent except the injected transcribe_fn; fully testable with a stub.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .merge import MergeOptions, merge
from .models import MergedOutput

Region = tuple[float, float]            # (start, end) seconds on the original timeline
TranscribeFn = Callable[[str], dict]    # wav path -> WhisperX output dict

SR = 16000  # tracks are extracted to mono/16k


# --- VAD -------------------------------------------------------------------------------

_vad_model = None


def detect_speech(wav_path: str | Path) -> list[Region]:
    """Run silero-VAD over a mono/16k wav and return speech regions (seconds)."""
    global _vad_model
    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
    if _vad_model is None:
        _vad_model = load_silero_vad()
    wav = read_audio(str(wav_path), sampling_rate=SR)
    spans = get_speech_timestamps(wav, _vad_model, sampling_rate=SR, return_seconds=True)
    return [(float(s["start"]), float(s["end"])) for s in spans]


# --- timeline remapping (pure, GPU-independent) ----------------------------------------

def _concat_regions(audio: np.ndarray, regions: list[Region], sr: int = SR):
    """Concatenate a speaker's region slices. Returns (concat_audio, mapping) where mapping
    is a list of (concat_start, orig_start, duration) — one piece per region, in order."""
    pieces, mapping, cursor = [], [], 0.0
    for start, end in sorted(regions):
        i0, i1 = max(0, int(round(start * sr))), min(len(audio), int(round(end * sr)))
        if i1 <= i0:
            continue
        pieces.append(audio[i0:i1])
        dur = (i1 - i0) / sr
        mapping.append((cursor, i0 / sr, dur))
        cursor += dur
    concat = np.concatenate(pieces) if pieces else np.zeros(0, dtype=audio.dtype)
    return concat, mapping


def _map_time(t: float, mapping: list[tuple[float, float, float]]) -> float:
    """Map a time in the concatenated audio back to the original timeline. Each piece owns
    the half-open span [concat_start, concat_start+dur), so a time exactly on a seam belongs
    to the later piece (where the next word actually starts)."""
    for concat_start, orig_start, dur in mapping:
        if t < concat_start + dur:
            return orig_start + max(0.0, t - concat_start)
    if mapping:  # at/past the end -> clamp to the last region's end
        concat_start, orig_start, dur = mapping[-1]
        return orig_start + dur
    return t


def _remap_output(wx: dict, mapping: list[tuple[float, float, float]]) -> dict:
    """Remap a WhisperX output (on concat timeline) onto the original timeline. Segment
    bounds are recomputed from remapped words so region seams don't distort them."""
    out_segments = []
    for seg in wx.get("segments", []):
        words = []
        for w in seg.get("words", []):
            if w.get("start") is None or w.get("end") is None:
                continue
            words.append({**w,
                          "start": _map_time(float(w["start"]), mapping),
                          "end": _map_time(float(w["end"]), mapping)})
        if words:
            start, end = min(x["start"] for x in words), max(x["end"] for x in words)
        else:
            start = _map_time(float(seg.get("start", 0.0)), mapping)
            end = _map_time(float(seg.get("end", 0.0)), mapping)
        out_segments.append({"start": start, "end": end,
                             "text": seg.get("text", ""), "words": words})
    return {"segments": out_segments}


def _safe(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip() or "spk"


# --- per-speaker transcription ---------------------------------------------------------

def transcribe_regions(
    track_wav: str | Path,
    speaker_regions: dict[str, list[Region]],
    transcribe_fn: TranscribeFn,
    *,
    options: Optional[MergeOptions] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> MergedOutput:
    """Transcribe each speaker on only their labeled regions, remap onto the original
    timeline, and merge. `track_wav` is the single track extracted to mono/16k.
    `progress(frac, msg)` fires before each speaker, weighted by labeled seconds; it is also the
    job's pause/cancel checkpoint, so those take effect between speakers."""
    import soundfile as sf

    audio, sr = sf.read(str(track_wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    outputs: list[dict] = []
    labels: list[str] = []
    secs = {spk: sum(e - s for s, e in regs) for spk, regs in speaker_regions.items()}
    total, done = sum(secs.values()) or 1.0, 0.0
    tmpdir = Path(tempfile.mkdtemp(prefix="wm_diar_"))
    try:
        for k, (speaker, regions) in enumerate(speaker_regions.items(), 1):
            if progress:
                progress(done / total, f"transcribing {speaker} ({k}/{len(speaker_regions)})")
            done += secs[speaker]
            concat, mapping = _concat_regions(audio, regions, sr)
            if len(concat) == 0:
                continue
            clip = tmpdir / f"{_safe(speaker)}.wav"
            sf.write(str(clip), concat, sr, subtype="PCM_16")
            wx = transcribe_fn(str(clip))
            outputs.append(_remap_output(wx, mapping))
            labels.append(speaker)
    finally:
        for p in tmpdir.glob("*"):
            p.unlink(missing_ok=True)
        tmpdir.rmdir()

    merged = merge(outputs, labels, [0.0] * len(outputs), options=options)
    # Where two speakers' labeled regions overlap, the user marked deliberate talk-over (e.g. a
    # duplicated region), so both copies transcribe the same audio — that's overlap, not bleed.
    # ponytail: O(regions²) pairwise scan; fine for hand-labeled counts, sweep-line if it ever isn't.
    spans = list(speaker_regions.items())
    shared = [(max(a0, b0), min(a1, b1))
              for x, (_, ra) in enumerate(spans) for _, rb in spans[x + 1:]
              for a0, a1 in ra for b0, b1 in rb if min(a1, b1) > max(a0, b0)]
    for seg in merged.segments:
        if "bleed_suspected" in seg.flags and any(seg.start < e and seg.end > s for s, e in shared):
            seg.flags = [f for f in seg.flags if f != "bleed_suspected"]
            if "overlap" not in seg.flags:
                seg.flags.append("overlap")
    return merged
