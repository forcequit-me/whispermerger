"""ffprobe wrapper — enumerate audio streams + duration (FR-IN-4).

Builds an `InputSource` (§9.1) from any audio or video file. `Track.track_index` is the
0-based index *among audio streams*, i.e. the N in ffmpeg's `-map 0:a:N` used by extract.py.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .models import InputSource, Track

# OBS multi-track convention (FR-TK-2): first two tracks default to Mic / Desktop. Labels
# stay editable in the UI.
_OBS_DEFAULTS = ["Mic", "Desktop"]


def _default_speaker(track_index: int, num_audio: int, filename: str) -> str:
    if num_audio == 1:
        return Path(filename).stem  # single source -> name it after the file
    if track_index < len(_OBS_DEFAULTS):
        return _OBS_DEFAULTS[track_index]
    return f"Track {track_index + 1}"


def probe(path: str | Path) -> InputSource:
    """Run ffprobe on `path` and return its InputSource with one Track per audio stream."""
    path = Path(path)
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        check=True, capture_output=True, text=True,
    )
    data = json.loads(result.stdout)

    audio_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    num_audio = len(audio_streams)

    duration = _read_duration(data, audio_streams)

    tracks = [
        Track(
            track_index=i,
            speaker=_default_speaker(i, num_audio, path.name),
        )
        for i in range(num_audio)
    ]
    return InputSource(filename=path.name, duration=duration, tracks=tracks)


def _read_duration(data: dict, audio_streams: list[dict]) -> float:
    """Prefer the container duration; fall back to the longest audio stream."""
    fmt = data.get("format", {})
    if fmt.get("duration") is not None:
        return float(fmt["duration"])
    stream_durations = [
        float(s["duration"]) for s in audio_streams if s.get("duration") is not None
    ]
    return max(stream_durations) if stream_durations else 0.0
