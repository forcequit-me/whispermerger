"""ffprobe wrapper tests — run against an ffmpeg-generated multi-stream file."""

from app.probe import probe


def test_enumerates_audio_streams(two_stream_file):
    src = probe(two_stream_file)
    assert len(src.tracks) == 2
    assert [t.track_index for t in src.tracks] == [0, 1]


def test_reads_duration(two_stream_file):
    src = probe(two_stream_file)
    assert src.duration == 2.0 or abs(src.duration - 2.0) < 0.2


def test_default_speaker_labels(two_stream_file):
    src = probe(two_stream_file)
    assert [t.speaker for t in src.tracks] == ["Speaker 1", "Speaker 2"]
    assert all(t.selected for t in src.tracks)


def test_filename_recorded(two_stream_file):
    src = probe(two_stream_file)
    assert src.filename == "two_stream.mkv"
