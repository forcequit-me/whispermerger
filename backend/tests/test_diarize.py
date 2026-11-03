"""v2 diarization engine tests — timeline remapping (the risky core) + region transcription
with a stub. GPU-independent. detect_speech runs silero on CPU (no download, bundled model)."""

import numpy as np
import soundfile as sf

from app.diarize import _concat_regions, _map_time, _remap_output, detect_speech, transcribe_regions


def test_concat_regions_builds_mapping():
    audio = np.arange(16000 * 10, dtype=np.float32)  # 10s
    concat, mapping = _concat_regions(audio, [(1, 3), (5, 8)], sr=16000)
    assert len(concat) == (2 + 3) * 16000
    assert mapping[0] == (0.0, 1.0, 2.0)   # concat_start, orig_start, dur
    assert mapping[1] == (2.0, 5.0, 3.0)


def test_map_time_back_to_original():
    mapping = [(0.0, 1.0, 2.0), (2.0, 5.0, 3.0)]
    assert _map_time(0.0, mapping) == 1.0           # region 1 start -> orig 1
    assert abs(_map_time(1.5, mapping) - 2.5) < 1e-6  # mid region 1
    assert abs(_map_time(2.0, mapping) - 5.0) < 1e-6  # region 2 start -> orig 5 (jumps the gap)
    assert abs(_map_time(4.0, mapping) - 7.0) < 1e-6  # mid region 2
    assert abs(_map_time(99.0, mapping) - 8.0) < 1e-6  # past end -> clamp to last region end


def test_remap_output_uses_words_and_jumps_seam():
    mapping = [(0.0, 1.0, 2.0), (2.0, 5.0, 3.0)]
    wx = {"segments": [
        {"start": 0, "end": 1, "text": "a", "words": [{"word": "a", "start": 0, "end": 1}]},
        {"start": 2, "end": 3, "text": "b", "words": [{"word": "b", "start": 2, "end": 2.5}]}]}
    out = _remap_output(wx, mapping)
    assert abs(out["segments"][0]["words"][0]["start"] - 1.0) < 1e-6
    assert abs(out["segments"][1]["words"][0]["start"] - 5.0) < 1e-6  # crossed the seam
    assert abs(out["segments"][1]["start"] - 5.0) < 1e-6              # bounds from words


def test_transcribe_regions_remaps_and_merges(tmp_path):
    sr = 16000
    sf.write(str(tmp_path / "t.wav"), np.zeros(sr * 10, dtype=np.float32), sr, subtype="PCM_16")
    stub = lambda _w: {"segments": [{"start": 0.0, "end": 1.0, "text": "hi",
                                     "words": [{"word": "hi", "start": 0.0, "end": 1.0}]}]}
    out = transcribe_regions(str(tmp_path / "t.wav"), {"A": [(1, 3)], "B": [(5, 8)]}, stub)
    starts = {s.speaker: s.start for s in out.segments}
    assert abs(starts["A"] - 1.0) < 1e-6   # A's word lands at its region's real time
    assert abs(starts["B"] - 5.0) < 1e-6


def test_detect_speech_runs(tmp_path):
    sr = 16000
    sf.write(str(tmp_path / "s.wav"), np.zeros(sr * 2, dtype=np.float32), sr, subtype="PCM_16")
    regions = detect_speech(str(tmp_path / "s.wav"))
    assert isinstance(regions, list)  # silence -> no regions


def test_duplicated_region_is_overlap_not_bleed(tmp_path):
    # the same span labeled for two speakers transcribes identically — deliberate talk-over
    sr = 16000
    sf.write(str(tmp_path / "t.wav"), np.zeros(sr * 10, dtype=np.float32), sr, subtype="PCM_16")
    stub = lambda _w: {"segments": [{"start": 0.0, "end": 1.5, "text": "we both said this",
                                     "words": []}]}
    out = transcribe_regions(str(tmp_path / "t.wav"), {"A": [(1, 3)], "B": [(1, 3)]}, stub)
    assert [s.flags for s in out.segments] == [["overlap"], ["overlap"]]
