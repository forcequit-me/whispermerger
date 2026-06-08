"""Pipeline tests — real ffmpeg extraction + a synthetic transcribe stub (no GPU)."""

from app.models import InputSource, Track
from app.pipeline import PreparedSource, run_merge_pipeline


def _stub_transcribe(_audio_path: str) -> dict:
    # Canned WhisperX-shaped output; same for every track (path ignored).
    return {"segments": [{"start": 0.0, "end": 1.0, "text": "hello",
                          "words": [{"word": "hello", "start": 0.0, "end": 1.0}]}]}


def test_pipeline_extracts_transcribes_merges(two_stream_file):
    src = InputSource(
        filename=two_stream_file.name,
        duration=2.0,
        tracks=[Track(track_index=0, speaker="Mic"), Track(track_index=1, speaker="Desktop")],
    )
    out = run_merge_pipeline([PreparedSource(str(two_stream_file), src)], _stub_transcribe)
    assert {s.speaker for s in out.segments} == {"Mic", "Desktop"}
    assert len(out.sources) == 1


def test_pipeline_skips_unselected_tracks(two_stream_file):
    src = InputSource(
        filename=two_stream_file.name,
        duration=2.0,
        tracks=[Track(track_index=0, speaker="Mic"),
                Track(track_index=1, speaker="Desktop", selected=False)],
    )
    out = run_merge_pipeline([PreparedSource(str(two_stream_file), src)], _stub_transcribe)
    assert {s.speaker for s in out.segments} == {"Mic"}


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
