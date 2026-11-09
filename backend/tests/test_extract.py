"""ffmpeg extraction tests — verify format (mono/16k/pcm_s16le) and temp cleanup."""

import json
import subprocess

import pytest

from app.extract import extract_ahead, extract_track, extracted_audio


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


def test_concat_wavs(tmp_path):
    import numpy as np
    import soundfile as sf
    from app.extract import concat_wavs
    sf.write(str(tmp_path / "a.wav"), np.zeros(16000 * 2, dtype="float32"), 16000, subtype="PCM_16")
    sf.write(str(tmp_path / "b.wav"), np.zeros(16000 * 3, dtype="float32"), 16000, subtype="PCM_16")
    total, boundaries = concat_wavs([tmp_path / "a.wav", tmp_path / "b.wav"], tmp_path / "c.wav")
    assert abs(total - 5.0) < 0.01
    assert boundaries[0] == 0.0 and abs(boundaries[1] - 2.0) < 0.01
    assert (tmp_path / "c.wav").exists()


def test_compute_peaks_shape_and_range(tmp_path):
    import numpy as np
    import soundfile as sf
    from app.extract import compute_peaks
    # 4s ramp 0->1 in amplitude so peaks should rise monotonically across buckets
    t = np.linspace(0, 1, 16000 * 4, dtype="float32")
    sf.write(str(tmp_path / "r.wav"), t * np.sin(2 * np.pi * 200 * t), 16000, subtype="PCM_16")
    peaks = compute_peaks(tmp_path / "r.wav", buckets=100)
    assert len(peaks) == 100
    assert all(0.0 <= p <= 1.0 for p in peaks)
    assert max(peaks) == 1.0                 # normalized to peak
    assert peaks[-1] > peaks[0]              # ramp -> louder at the end
    # a window (zoomed in): its own bucket count, raw amplitudes, only that stretch of audio
    first = compute_peaks(tmp_path / "r.wav", buckets=50, start=0.0, end=2.0, normalize=False)
    assert len(first) == 50 and max(first) < 0.6   # quiet first half of the ramp


def test_extracted_audio_context_cleans_up(two_stream_file):
    with extracted_audio(two_stream_file, [0, 1]) as paths:
        assert len(paths) == 2
        assert all(p.exists() for p in paths)
        tmpdir = paths[0].parent
    assert not any(p.exists() for p in paths)  # files removed
    assert not tmpdir.exists()                  # temp dir removed


# --- extract_ahead (prefetch iterator) — pure-Python, no ffmpeg ---------------------------

def test_extract_ahead_preserves_order_and_results():
    items = [1, 2, 3, 4]
    out = list(extract_ahead(items, lambda x: x * 10))
    assert out == [(1, 10), (2, 20), (3, 30), (4, 40)]


def test_extract_ahead_prefetches_one_ahead():
    # While the consumer holds item 1, item 2's extraction has already STARTED in the worker
    # (that overlap is the whole point — extraction runs during transcription).
    import threading
    started = {1: threading.Event(), 2: threading.Event(), 3: threading.Event()}
    yielded = []

    def extract_one(x):
        started[x].set()
        return x

    for item, _ in extract_ahead([1, 2, 3], extract_one):
        yielded.append(item)
        if item == 1:
            assert started[2].wait(timeout=2)   # next item is being extracted while we hold this one
            assert not started[3].is_set()       # but only one ahead (depth 1)

    assert yielded == [1, 2, 3]                   # each yielded exactly once, in order


def test_extract_ahead_error_surfaces_at_its_item():
    seen = []

    def extract_one(x):
        if x == 2:
            raise RuntimeError("boom on 2")
        return x

    gen = extract_ahead([1, 2, 3], extract_one)
    seen.append(next(gen))                 # item 1 yields fine
    with pytest.raises(RuntimeError, match="boom on 2"):
        next(gen)                          # item 2's error surfaces here, not swallowed
    assert seen == [(1, 1)]


def test_extract_ahead_stops_prefetch_when_cancelled():
    extracted = []
    cancelled = {"v": False}

    def extract_one(x):
        extracted.append(x)
        return x

    out = []
    for item, _ in extract_ahead([1, 2, 3, 4], extract_one,
                                 is_cancelled=lambda: cancelled["v"]):
        out.append(item)
        if item == 1:
            cancelled["v"] = True          # request cancel after the first item
            break

    # Item 2 may have been prefetched before cancel; items 3/4 must never be extracted.
    assert out == [1]
    assert 3 not in extracted and 4 not in extracted


def test_loudness_profile_and_span_rms(tmp_path):
    import numpy as np
    import soundfile as sf
    from app.extract import loudness_profile, span_rms
    sr = 16000
    loud = 0.8 * np.ones(sr * 2, dtype=np.float32)     # 0-2 s at 0.8
    quiet = 0.1 * np.ones(sr * 2, dtype=np.float32)    # 2-4 s at 0.1
    sf.write(str(tmp_path / "l.wav"), np.concatenate([loud, quiet]), sr, subtype="FLOAT")
    prof = loudness_profile(tmp_path / "l.wav")
    assert len(prof) == 80                              # 4 s at 20 frames/s
    assert abs(span_rms(prof, 0.0, 2.0) - 0.8) < 1e-3
    assert abs(span_rms(prof, 2.0, 4.0) - 0.1) < 1e-3
    assert span_rms(prof, 10.0, 11.0) == 0.0            # past the end
