"""ffmpeg extraction tests — verify format (mono/16k/pcm_s16le) and temp cleanup."""

import json
import subprocess

from app.extract import extract_track, extracted_audio


def _probe_audio(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", "-select_streams", "a", str(path)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(out.stdout)["streams"][0]


def test_extract_track_format(two_stream_file, tmp_path):
    dest = tmp_path / "track0.wav"
    extract_track(two_stream_file, 0, dest)
    assert dest.exists()
    s = _probe_audio(dest)
    assert s["channels"] == 1
    assert s["sample_rate"] == "16000"
    assert s["codec_name"] == "pcm_s16le"


def test_extract_maps_correct_stream(two_stream_file, tmp_path):
    # Stream 0 is 440 Hz, stream 1 is 880 Hz — just confirm both extract independently.
    d0 = extract_track(two_stream_file, 0, tmp_path / "a.wav")
    d1 = extract_track(two_stream_file, 1, tmp_path / "b.wav")
    assert d0.exists() and d1.exists()


def test_extracted_audio_context_cleans_up(two_stream_file):
    with extracted_audio(two_stream_file, [0, 1]) as paths:
        assert len(paths) == 2
        assert all(p.exists() for p in paths)
        tmpdir = paths[0].parent
    assert not any(p.exists() for p in paths)  # files removed
    assert not tmpdir.exists()                  # temp dir removed
