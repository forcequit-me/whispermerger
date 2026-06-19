"""Pipeline tests — real ffmpeg extraction + a synthetic transcribe stub (no GPU)."""

from app.models import InputSource, Track
from app.pipeline import PreparedSource, run_merge_pipeline


def _stub_transcribe(_audio_path: str) -> dict:
    # Canned WhisperX-shaped output; same for every track (path ignored).
    return {"segments": [{"start": 0.0, "end": 1.0, "text": "hello",
                          "words": [{"word": "hello", "start": 0.0, "end": 1.0}]}]}


def test_pipeline_mixed_track_counts_in_one_project(two_stream_file):
    # One project, files with differing track counts: a 2-track source then a 1-track source.
    two_track = InputSource(filename="multi.mkv", duration=4.0, sequence=0,
                            tracks=[Track(track_index=0, speaker="A"), Track(track_index=1, speaker="B")])
    one_track = InputSource(filename="solo.wav", duration=3.0, sequence=1,
                            tracks=[Track(track_index=0, speaker="C")])
    out = run_merge_pipeline(
        [PreparedSource(str(two_stream_file), two_track),
         PreparedSource(str(two_stream_file), one_track)],
        _stub_transcribe,
    )
    # 3 transcriptions (2 + 1); the 1-track file is offset by the first file's 4s duration
    assert sorted((s.speaker, s.start) for s in out.segments) == [("A", 0.0), ("B", 0.0), ("C", 4.0)]


def test_pipeline_extracts_transcribes_merges(two_stream_file):
    src = InputSource(
        filename=two_stream_file.name,
        duration=2.0,
        tracks=[Track(track_index=0, speaker="Speaker 1"), Track(track_index=1, speaker="Speaker 2")],
    )
    out = run_merge_pipeline([PreparedSource(str(two_stream_file), src)], _stub_transcribe)
    assert {s.speaker for s in out.segments} == {"Speaker 1", "Speaker 2"}
    assert len(out.sources) == 1


def test_pipeline_skips_unselected_tracks(two_stream_file):
    src = InputSource(
        filename=two_stream_file.name,
        duration=2.0,
        tracks=[Track(track_index=0, speaker="Speaker 1"),
                Track(track_index=1, speaker="Speaker 2", selected=False)],
    )
    out = run_merge_pipeline([PreparedSource(str(two_stream_file), src)], _stub_transcribe)
    assert {s.speaker for s in out.segments} == {"Speaker 1"}


def test_pipeline_cleans_temp_dirs(two_stream_file):
    # Prefetch uses a per-run tmpdir with per-file subdirs — it must leave nothing behind.
    import glob
    import os
    import tempfile
    root = tempfile.gettempdir()
    before = set(glob.glob(os.path.join(root, "wm_pipe_*")))
    src = InputSource(filename="x.mkv", duration=2.0,
                      tracks=[Track(track_index=0, speaker="A"), Track(track_index=1, speaker="B")])
    run_merge_pipeline([PreparedSource(str(two_stream_file), src)], _stub_transcribe)
    after = set(glob.glob(os.path.join(root, "wm_pipe_*")))
    assert after <= before                      # no run tmpdir left behind


def test_pipeline_sequential_offsets(two_stream_file):
    # Two files, one selected track each, sequence 0 then 1 -> second offset by first's duration.
    a = InputSource(filename="a.mkv", duration=4.0, sequence=0,
                    tracks=[Track(track_index=0, speaker="A")])
    b = InputSource(filename="b.mkv", duration=3.0, sequence=1,
                    tracks=[Track(track_index=0, speaker="B")])
    out = run_merge_pipeline(
        [PreparedSource(str(two_stream_file), b),   # deliberately out of order
         PreparedSource(str(two_stream_file), a)],
        _stub_transcribe,
    )
    by_speaker = {s.speaker: s.start for s in out.segments}
    assert by_speaker["A"] == 0.0
    assert by_speaker["B"] == 4.0  # offset by A's duration
