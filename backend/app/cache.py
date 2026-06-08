"""WhisperX-output cache (PRD §8.8, FR-CA-*).

Keyed by SHA-256 of the audio file (OQ-4 lean: file hash, re-encode busts it — acceptable
v1) plus model version plus alignment model, so a cached entry is only reused when the audio
*and* the models match. Skips 30+ minutes of recompute when merge logic is tweaked and the
same session is reprocessed (SM-4).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_CHUNK = 1 << 20  # 1 MiB


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_key(audio_path: str | Path, model: str, align_model: str) -> str:
    """Stable key for one transcription: audio content + model + alignment model."""
    digest = file_sha256(audio_path)
    return hashlib.sha256(f"{digest}|{model}|{align_model}".encode()).hexdigest()


class Cache:
    """Filesystem JSON cache of WhisperX outputs."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self._path(key)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def put(self, key: str, whisperx_output: dict) -> None:
        self._path(key).write_text(
            json.dumps(whisperx_output, ensure_ascii=False), encoding="utf-8"
        )
