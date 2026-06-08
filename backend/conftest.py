"""Shared test fixtures. Lives at backend/ so `app` imports resolve when running pytest
from the backend directory."""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))  # ensure `app` is importable


@pytest.fixture(scope="session")
def two_stream_file(tmp_path_factory) -> Path:
    """A 2-second .mkv with two audio streams (no video) — an OBS-like multi-track input."""
    out = tmp_path_factory.mktemp("media") / "two_stream.mkv"
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-f", "lavfi", "-i", "sine=frequency=880:duration=2",
         "-map", "0:a", "-map", "1:a",
         "-c:a", "pcm_s16le", str(out)],
        check=True, capture_output=True,
    )
    return out
