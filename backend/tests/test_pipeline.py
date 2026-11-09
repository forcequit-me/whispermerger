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


def test_pipeline_marks_louder_mic_preferred_on_bleed(tmp_path):
    import subprocess
    # track 1 is the speaker's own mic; track 2 hears the same audio at 10% (bleed)
    f = tmp_path / "bleed.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-y",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=2,volume=0.1",
                    "-map", "0:a", "-map", "1:a", "-c:a", "pcm_s16le", str(f)], check=True)
    src = InputSource(filename="bleed.mkv", duration=2.0,
                      tracks=[Track(track_index=0, speaker="Alice"), Track(track_index=1, speaker="Bob")])
    out = run_merge_pipeline([PreparedSource(str(f), src)], _stub_transcribe)   # same text on both
    flags = {s.speaker: s.flags for s in out.segments}
    assert flags == {"Alice": ["bleed_suspected", "preferred"], "Bob": ["bleed_suspected"]}


def test_chunk_progress_reports_each_chunk_and_restores_the_model():
    """#33: a wrapper on the WhisperX pipeline instance reports chunk progress, then goes away."""
    from app.transcribe import _transcribe_counted, chunk_progress

    class FakePipe:                                   # mimics WhisperX: transcribe() calls self.__call__
        def __call__(self, inputs, batch_size=None):
            assert not isinstance(inputs, list)       # must stay lazy, or progress lands all at once
            for x in inputs:
                yield {"text": x}

        def transcribe(self, audio, batch_size=None, language=None):
            return {"segments": [o["text"] for o in self.__call__(iter(audio), batch_size=batch_size)]}

    pipe, seen = FakePipe(), []
    with chunk_progress(seen.append):
        out = _transcribe_counted(pipe, ["a", "b", "c", "d"], 0.5, 1.0, batch_size=2)
    assert out == {"segments": ["a", "b", "c", "d"]}
    assert seen == [0.625, 0.75, 0.875, 1.0]
    assert "__call__" not in vars(pipe)               # instance wrapper removed
    assert _transcribe_counted(pipe, ["x"]) == {"segments": ["x"]}   # no callback: plain call


def test_pipeline_progress_moves_during_a_track(tmp_path):
    """The bar moves inside a track (weighted like whole tracks), and the message shows its percent."""
    from app.transcribe import _chunk
    seen = []

    def stub(path):
        _chunk.fn(0.5)                                # what transcribe_track reports mid-track
        return {"segments": []}

    import numpy as np, soundfile as sf
    from app.models import InputSource, Track
    from app.pipeline import PreparedSource, run_merge_pipeline
    wav = tmp_path / "a.wav"
    sf.write(wav, np.zeros(16000, dtype=np.float32), 16000)
    src = InputSource(filename="a.wav", duration=1.0, tracks=[Track(track_index=0, speaker="S1")])
    run_merge_pipeline([PreparedSource(input=src, path=str(wav))], stub,
                       progress_cb=lambda f, m: seen.append((round(f, 2), m)))
    assert (0.45, "Transcribing track 1/1 · S1 · 50%") in seen   # 90% of the track is transcription


def test_per_file_pipeline_reports_chunks_per_track(tmp_path):
    """Batch file rows move within a track too: on_chunk(file, track, tracks, share done)."""
    import numpy as np, soundfile as sf
    from app.models import InputSource, Track
    from app.pipeline import PreparedSource, run_per_file_pipeline
    from app.transcribe import _chunk

    def stub(path):
        _chunk.fn(0.5)
        return {"segments": []}

    wav = tmp_path / "a.wav"
    sf.write(wav, np.zeros(16000, dtype=np.float32), 16000)
    src = InputSource(filename="a.wav", duration=1.0,
                      tracks=[Track(track_index=0, speaker="S1"), Track(track_index=0, speaker="S2")])
    chunks, noop = [], (lambda *a: None)
    run_per_file_pipeline([PreparedSource(input=src, path=str(wav))], stub, [0],
                          on_start=noop, on_track=noop, on_done=noop, on_error=noop, on_skip=noop,
                          is_file_cancelled=lambda i: False, on_chunk=lambda *a: chunks.append(a))
    assert chunks == [(0, 0, 2, 0.5), (0, 1, 2, 0.5)]


def test_models_unload_for_real_unless_another_thread_is_transcribing():
    """CTranslate2 keeps VRAM until unload_model(); without it the next model ran ~10x slower. But
    jobs share no GPU lock, so never unload under another thread's transcription."""
    import threading
    from app import transcribe as t

    class Pipe:                                        # whisperx pipeline -> .model (faster-whisper) -> .model (ct2)
        def __init__(self):
            self.unloaded = False
            self.model = type("FW", (), {})()
            self.model.model = self

        def unload_model(self):
            self.unloaded = True

    a = Pipe()
    t._loaded[:] = [a]
    t.unload_models()
    assert a.unloaded and not t._loaded

    b, busy, done = Pipe(), threading.Event(), threading.Event()

    def other_job():
        with t.gpu_user():
            busy.set()
            done.wait(5)

    th = threading.Thread(target=other_job)
    th.start()
    busy.wait(5)
    t._loaded[:] = [b]
    t.unload_models()                                  # another thread is mid-transcription
    assert not b.unloaded and not t._loaded            # references dropped, model left working
    done.set()
    th.join()


def test_batch_size_fits_free_vram(monkeypatch):
    """#37: the batch shrinks to what fits the VRAM left after the model loads, never above the cap."""
    import torch
    from app import transcribe as t
    gb = lambda free: monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (free * 1e9, 12.9e9))
    gb(7.6); assert t._fit_batch(16) == 16          # free 12 GB card with large-v3 loaded
    gb(7.6); assert t._fit_batch(24) == 17          # 24 would spill: capped by memory
    gb(2.1); assert t._fit_batch(16) == 4           # another app holds ~5.5 GB
    gb(0.3); assert t._fit_batch(16) == 1           # never below 1

    class Pipe:                                     # the cap is applied to every transcribe call
        fit_batch = 4
        def transcribe(self, audio, batch_size=None, language=None):
            return {"batch_size": batch_size}
    assert t._transcribe_counted(Pipe(), [], batch_size=16) == {"batch_size": 4}
