"""Pausing a run mid-track: does it give up the GPU turn, and does the track survive it?

The real transcribe_track, JobRegistry, BatchRegistry, GPU_TURN, unload_models and the lru-cached
_load_model; only the speech model is fake (no GPU, no model files). The fake works like a WhisperX
pipeline: transcribe() walks self.__call__, which runs the model one batch of chunks at a time and
hands the results out one by one, so _transcribe_counted's progress hook fires after every chunk.
Under it sits a stand-in for the CTranslate2 model with the one behaviour that matters here: once
unload_model() has run, the next call raises "The model for this translator was unloaded", as
ctranslate2 4.4.0 does (checked on a tiny model on the CPU, 2026-11-27).

Each test asserts what a user would expect. The rule they hold every kind of run to: a pause asked
for mid-track waits for the end of that track ("pausing"), then the run parks outside its GPU turn
("GPU released") and another run can take its turn; the track's result is intact after resume.
Cancel still stops a track part-way, and a run waiting for the GPU pauses and cancels at once."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import whisperx

from app import benchmark, transcribe
from app.batch import BatchRegistry, ProjectInput, Status, run_batch
from app.diarize import transcribe_regions
from app.jobs import JobRegistry, JobStatus
from app.models import InputSource, Track
from app.pipeline import WAITING, PreparedSource, run_merge_pipeline

DONE_STATES = (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELED)


class _Ct2Model:
    """The CTranslate2 model under the pipeline: unusable once unloaded, until loaded again."""

    def __init__(self):
        self.loaded = True

    def unload_model(self, to_cpu=False):
        self.loaded = False

    def run(self, batch):
        if not self.loaded:
            raise RuntimeError("The model for this translator was unloaded")
        return [f" chunk {c}" for c in batch]


class _Gate:
    """Called by the fake model before it hands out each chunk. At chunk `hold_at` (counted over
    every run in the test) it stops until the test says go, so the test can pause a run while
    that chunk is in flight. `n` is how many chunks the model has run so far."""

    def __init__(self, hold_at=1):
        self.n, self.hold_at = 0, hold_at
        self.reached, self.go = threading.Event(), threading.Event()

    def __call__(self):
        self.n += 1
        if self.n == self.hold_at:
            self.reached.set()
            self.go.wait(30)


class _Look(_Gate):
    """A _Gate that also calls `look()` at chunk `look_at` and keeps what it returned in `seen`."""

    def __init__(self, hold_at, look_at, look):
        super().__init__(hold_at)
        self.look_at, self.look, self.seen = look_at, look, None

    def __call__(self):
        super().__call__()
        if self.n == self.look_at:
            self.seen = self.look()


class _Pipe:
    """A WhisperX pipeline in miniature (see asr.FasterWhisperPipeline.transcribe)."""

    def __init__(self, chunks, gate, batch):
        self.model = SimpleNamespace(model=_Ct2Model())
        self.chunks, self.gate = chunks, gate
        self.fit_batch = batch                   # _transcribe_counted caps the batch size at this

    def __call__(self, inputs, batch_size):
        items = list(inputs)
        for k in range(0, len(items), batch_size):           # one model call per batch of chunks
            for text in self.model.model.run(items[k:k + batch_size]):
                self.gate()
                yield {"text": text}

    def transcribe(self, audio, batch_size, language):
        outs = self.__call__((c for c in range(self.chunks)), batch_size=batch_size)
        return {"language": language,
                "segments": [{"start": float(i), "end": i + 1.0, "text": o["text"]} for i, o in enumerate(outs)]}


@pytest.fixture
def rig(monkeypatch, tmp_path):
    """Fake speech model behind the real model cache; nothing downloads, nothing touches CUDA."""
    monkeypatch.setattr(transcribe.modelstore, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(transcribe.modelstore, "ensure", lambda e: None)
    monkeypatch.setattr(whisperx, "load_audio", lambda p: np.zeros(16000, dtype=np.float32))
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)   # unload_models: no empty_cache
    r = SimpleNamespace(gate=_Gate(), chunks=3, batch=16, pipes=[], tmp=tmp_path)

    def build(model_name, compute_type, device):           # what _load_model builds, kept in _loaded
        pipe = _Pipe(r.chunks, r.gate, r.batch)
        transcribe._loaded.append(pipe)
        r.pipes.append(pipe)
        return pipe

    monkeypatch.setattr(transcribe, "_build_model", build)
    transcribe._load_model.cache_clear()
    transcribe._loaded.clear()
    yield r
    r.gate.go.set()
    transcribe._load_model.cache_clear()
    transcribe._loaded.clear()


def _until(check, timeout=10.0) -> bool:
    deadline = time.time() + timeout
    while not check() and time.time() < deadline:
        time.sleep(0.01)
    return check()


def _transcribe_fn(wav):
    """What the routes pass the pipelines (language given, so no Auto-detect)."""
    return transcribe.transcribe_track(wav, language="is", word_timing=False)


def _job_run(work, on_pause=transcribe.unload_models):
    job = JobRegistry().create()
    JobRegistry().run(job, work, on_pause=on_pause)    # as the routes do
    return job


def _transcribe_job(two_stream_file, on_pause=transcribe.unload_models):
    """POST /api/transcribe: run_merge_pipeline on one track of a file, in a job."""
    src = InputSource(filename="a.mkv", duration=2.0, tracks=[Track(track_index=0, speaker="A")])
    return _job_run(lambda progress: run_merge_pipeline(
        [PreparedSource(str(two_stream_file), src)], _transcribe_fn, progress_cb=progress), on_pause)


def _other_run(tmp_path):
    """Another run wanting the GPU (any kind; here a bare transcription with the waiting report)."""
    def work(progress):
        hook = lambda f, waiting=False, detected=None: progress(f, WAITING if waiting else f"{int(f * 100)}%")
        with transcribe.chunk_progress(hook):
            return transcribe.transcribe_track(tmp_path / "other.wav", language="is", word_timing=False)
    return _job_run(work)


def _settle(*jobs):
    for j in jobs:
        if j is not None:
            _until(lambda: j.status in DONE_STATES, 6)


def _pause_job_mid_chunk(rig, job):
    """Pause `job` while its model is on chunk `rig.gate.hold_at`, then let it go on until it parks.
    Returns how many chunks the model had run when it parked."""
    assert rig.gate.reached.wait(20), "the run never reached the chunk to pause on"
    job.pause()
    rig.gate.go.set()
    assert _until(lambda: job.gpu_freed), "the paused run never parked"
    return rig.gate.n


def _the_other_run_finishes_while_paused(rig, paused_where: str):
    other = _other_run(rig.tmp)
    ok = _until(lambda: other.status is JobStatus.DONE)
    return other, ok, (f"while the first run is paused ({paused_where}, GPU_TURN held: "
                       f"{transcribe.GPU_TURN.locked()}), the other run still says {other.message!r}")


# --- 1. a single Transcribe job --------------------------------------------------------------

def test_a_job_paused_mid_track_lets_another_run_take_its_turn(rig, two_stream_file):
    """The pause lands when the track ends, outside the GPU turn; then the other run goes."""
    job = _transcribe_job(two_stream_file)
    other = None
    try:
        landed = _pause_job_mid_chunk(rig, job)
        assert landed == rig.chunks                          # the whole track ran first
        assert not transcribe.GPU_TURN.locked()
        assert job.message == "paused · GPU released"
        other, ok, why = _the_other_run_finishes_while_paused(rig, f"after chunk {landed} of {rig.chunks}")
        assert ok, why
        job.resume()
        assert _until(lambda: job.status is JobStatus.DONE), job.error
        assert len(job.result.segments) == rig.chunks        # its track is complete
    finally:
        job.resume()
        _settle(job, other)


def test_a_pause_in_the_last_batch_of_a_track_still_finishes(rig, two_stream_file):
    """The whole track runs before the pause lands; only then is its model unloaded, and the next
    run loads it again."""
    job = _transcribe_job(two_stream_file)        # 3 chunks, one batch
    try:
        assert _pause_job_mid_chunk(rig, job) == rig.chunks
        assert not rig.pipes[0].model.model.loaded      # unloaded once the track was done with it
        job.resume()
        assert _until(lambda: job.status is JobStatus.DONE), job.error
        assert len(job.result.segments) == 3
    finally:
        job.resume()
        _settle(job)


class _TwoHolds(_Gate):
    """A _Gate that also stops at chunk `then_at` (the next run's first chunk) until `go2`."""

    def __init__(self, hold_at, then_at):
        super().__init__(hold_at)
        self.then_at = then_at
        self.reached2, self.go2 = threading.Event(), threading.Event()

    def __call__(self):
        super().__call__()
        if self.n == self.then_at:
            self.reached2.set()
            self.go2.wait(30)


def test_a_pause_never_leaves_a_model_behind_on_the_gpu(rig, two_stream_file):
    """The run waiting for the GPU takes its turn the moment the paused run's track ends, and is
    already transcribing on the same cached model when the paused run parks and frees the GPU.
    That model must stay usable for the other run, and come off the GPU for real once nobody uses
    it, so "GPU released" holds; after resume there is one copy of it, never a second one."""
    rig.gate = _TwoHolds(1, rig.chunks + 1)
    src = InputSource(filename="a.mkv", duration=2.0,
                      tracks=[Track(track_index=0, speaker="A"), Track(track_index=1, speaker="B")])
    # The paused run parks only once the other run is mid-track (what the live test hit by timing).
    job = _job_run(lambda progress: run_merge_pipeline(
        [PreparedSource(str(two_stream_file), src)], _transcribe_fn, progress_cb=progress),
        on_pause=lambda: (rig.gate.reached2.wait(10), transcribe.unload_models()))
    other = None
    try:
        assert rig.gate.reached.wait(20)
        other = _other_run(rig.tmp)
        assert _until(lambda: WAITING in other.message), other.message
        job.pause()
        rig.gate.go.set()
        assert _until(lambda: job.gpu_freed), "the paused run never parked"
        assert rig.gate.reached2.is_set() and len(rig.pipes) == 1      # the other run is on the same model
        rig.gate.go2.set()
        assert _until(lambda: other.status is JobStatus.DONE), other.error   # not unloaded under it
        assert not [p for p in rig.pipes if p.model.model.loaded], "a model was left on the GPU"
        job.resume()
        assert _until(lambda: job.status is JobStatus.DONE), job.error
        loaded = [p for p in rig.pipes if p.model.model.loaded]
        assert len(loaded) == 1 and transcribe._loaded == loaded      # one copy, the cached one
    finally:
        rig.gate.go2.set()
        job.resume()
        _settle(job, other)


# --- 3. the model, unloaded under the running track ---------------------------------------------

def test_a_track_paused_before_its_last_batch_finishes_after_resume(rig, two_stream_file):
    """The pause is asked for in the track's first batch. The track goes on ("pausing": paused, the
    GPU not freed, the turn held) through its second batch, which needs the model, then parks
    ("paused · GPU released") and finishes after resume with every chunk."""
    rig.chunks, rig.batch = 4, 2                   # two batches: the pause is asked for in the first
    rig.gate = _Look(1, 3, None)                   # before the job builds its model
    job = _transcribe_job(two_stream_file)
    rig.gate.look = lambda: (job.paused, job.gpu_freed, transcribe.GPU_TURN.locked())
    try:
        assert _pause_job_mid_chunk(rig, job) == 4
        assert rig.gate.seen == (True, False, True)        # pausing, during the second batch
        assert job.message == "paused · GPU released"
        job.resume()
        _until(lambda: job.status in DONE_STATES)
        assert job.status is JobStatus.DONE, f"ended {job.status.value}: {job.error}"
        assert len(job.result.segments) == 4
    finally:
        job.resume()
        _settle(job)


# --- 2. the other kinds of run ---------------------------------------------------------------

def _batch(two_stream_file, tmp_path, per_file):
    reg = BatchRegistry()
    tracks = [Track(track_index=0, speaker="A"), Track(track_index=1, speaker="B")] if per_file \
        else [Track(track_index=0, speaker="A")]
    src = InputSource(filename="a.mkv", duration=2.0, tracks=tracks)
    proj = ProjectInput("P", [PreparedSource(str(two_stream_file), src)], per_file=per_file)
    batch = reg.create([proj])
    batch.on_pause = transcribe.unload_models                    # as the /api/batch route does
    reg.run(batch, lambda: run_batch(batch, [proj], _transcribe_fn,
                                     output_mode="folder", output_folder=str(tmp_path / "out")))
    return reg, batch


def _pause_batch_mid_chunk(rig, reg, batch):
    assert rig.gate.reached.wait(20), "the batch never reached the chunk to pause on"
    reg.pause(batch.id)
    rig.gate.go.set()
    assert _until(lambda: batch.gpu_freed), "the paused batch never parked"
    return rig.gate.n


def _finish_batch(reg, batch):
    reg.resume(batch.id)
    _until(lambda: batch.status in (Status.DONE, Status.CANCELED, Status.ERROR), 6)


def test_a_joined_batch_project_paused_mid_track_lets_another_run_take_its_turn(rig, two_stream_file, tmp_path):
    """A project joined into one transcript: the pause lands when the track ends, outside the turn."""
    reg, batch = _batch(two_stream_file, tmp_path, per_file=False)
    other = None
    try:
        landed = _pause_batch_mid_chunk(rig, reg, batch)
        assert landed == rig.chunks                          # the whole track ran first
        assert not transcribe.GPU_TURN.locked()
        other, ok, why = _the_other_run_finishes_while_paused(rig, f"after chunk {landed} of {rig.chunks}")
        assert ok, why
        _finish_batch(reg, batch)
        assert batch.status is Status.DONE and batch.projects[0].status is Status.DONE
        assert len(batch.projects[0].result.segments) == rig.chunks
    finally:
        _finish_batch(reg, batch)
        _settle(other)


def test_a_per_file_batch_project_paused_mid_track_lets_another_run_take_its_turn(rig, two_stream_file, tmp_path):
    """One transcript per file: a pause asked mid-track lands when the track ends, before the next."""
    reg, batch = _batch(two_stream_file, tmp_path, per_file=True)
    other = None
    try:
        landed = _pause_batch_mid_chunk(rig, reg, batch)
        assert landed == rig.chunks                          # the whole first track ran first
        assert batch.projects[0].files[0].track == 1         # and the second hasn't started
        assert not transcribe.GPU_TURN.locked()
        other, ok, why = _the_other_run_finishes_while_paused(rig, "between tracks")
        assert ok, why
        _finish_batch(reg, batch)
        assert batch.status is Status.DONE and batch.projects[0].files[0].status is Status.DONE
    finally:
        _finish_batch(reg, batch)
        _settle(other)


def test_label_track_paused_mid_speaker_lets_another_run_take_its_turn(rig, tmp_path):
    """POST /api/diarize: a pause lands at the report before the next speaker, outside the turn."""
    wav = tmp_path / "track.wav"
    sf.write(str(wav), np.full(32000, 0.1, dtype=np.float32), 16000)
    job = _job_run(lambda progress: transcribe_regions(
        str(wav), {"A": [(0.0, 1.0)], "B": [(1.0, 2.0)]}, _transcribe_fn, progress=progress))
    other = None
    try:
        landed = _pause_job_mid_chunk(rig, job)
        assert landed == rig.chunks                          # speaker A's clip ran to the end first
        assert not transcribe.GPU_TURN.locked()
        other, ok, why = _the_other_run_finishes_while_paused(rig, "between speakers")
        assert ok, why
        job.resume()
        assert _until(lambda: job.status is JobStatus.DONE), job.error
    finally:
        job.resume()
        _settle(job, other)


def test_a_benchmark_paused_between_models_lets_another_run_take_its_turn(rig, monkeypatch):
    monkeypatch.setattr(benchmark, "_tests", lambda tmp, names: [
        ("English", benchmark.SENTENCE, np.zeros(16000, dtype=np.float32), "en", names)])
    job = _job_run(lambda progress: benchmark.run(progress, names=["small", "medium"]))
    other = None
    try:
        landed = _pause_job_mid_chunk(rig, job)
        assert landed == rig.chunks                          # the first model's test ran to the end
        assert not transcribe.GPU_TURN.locked()
        other, ok, why = _the_other_run_finishes_while_paused(rig, f"after {landed} chunks, before the second model")
        assert ok, why
        job.resume()
        assert _until(lambda: job.status is JobStatus.DONE), job.error
        assert [r["model"] for r in job.result] == ["small", "medium"]
    finally:
        job.resume()
        _settle(job, other)


# --- 4. cancel mid-track, a download inside the turn, waiting for the GPU --------------------------

def test_cancel_still_stops_a_job_mid_track_even_while_it_is_pausing(rig, two_stream_file):
    rig.chunks, rig.batch = 6, 1                   # a model call per chunk
    job = _transcribe_job(two_stream_file)
    try:
        assert rig.gate.reached.wait(20)
        job.pause()
        job.cancel()
        rig.gate.go.set()
        assert _until(lambda: job.status is JobStatus.CANCELED), job.status
        assert rig.gate.n < rig.chunks                       # stopped part-way, at the next chunk
        assert not job.gpu_freed and not transcribe.GPU_TURN.locked()
    finally:
        _settle(job)


def test_cancel_still_stops_a_joined_batch_project_mid_track(rig, two_stream_file, tmp_path):
    rig.chunks, rig.batch = 6, 1
    reg, batch = _batch(two_stream_file, tmp_path, per_file=False)
    try:
        assert rig.gate.reached.wait(20)
        reg.pause(batch.id)
        reg.cancel(batch.id)
        rig.gate.go.set()
        assert _until(lambda: batch.status is Status.CANCELED), batch.status
        assert rig.gate.n < rig.chunks
        assert batch.projects[0].status is Status.CANCELED
    finally:
        _finish_batch(reg, batch)


class _Download:
    """A model download in the middle of a run (modelstore.ensure, inside the GPU turn): it reports
    its progress, which is the run's checkpoint, `reports` times (None: until the run is canceled)."""

    def __init__(self, reports=3):
        self.reports, self.n = reports, 0
        self.reached, self.go = threading.Event(), threading.Event()

    def __call__(self, entry):
        while self.reports is None or self.n < self.reports:
            self.n += 1
            transcribe.modelstore.report(f"Downloading {self.n}")
            if self.n == 1:
                self.reached.set()
                self.go.wait(30)
            time.sleep(0.01)


def test_a_pause_during_a_download_in_the_turn_lands_after_the_track(rig, two_stream_file, monkeypatch):
    dl = _Download()
    rig.gate.go.set()                              # no hold on the chunks: only the download stops
    monkeypatch.setattr(transcribe.modelstore, "ensure", dl)
    job = _transcribe_job(two_stream_file)
    try:
        assert dl.reached.wait(20)
        job.pause()
        dl.go.set()
        assert _until(lambda: job.gpu_freed)
        assert dl.n == 3 and rig.gate.n == rig.chunks      # the download, then the whole track, then parked
        assert not transcribe.GPU_TURN.locked()
        job.resume()
        assert _until(lambda: job.status is JobStatus.DONE), job.error
    finally:
        job.resume()
        dl.go.set()
        _settle(job)


def test_cancel_stops_a_download_in_the_turn(rig, two_stream_file, monkeypatch):
    dl = _Download(reports=None)                   # never finishes on its own
    monkeypatch.setattr(transcribe.modelstore, "ensure", dl)
    job = _transcribe_job(two_stream_file)
    try:
        assert dl.reached.wait(20)
        job.pause()
        job.cancel()
        dl.go.set()
        assert _until(lambda: job.status is JobStatus.CANCELED), job.status
        assert rig.gate.n == 0                               # the track never started
    finally:
        dl.go.set()
        _settle(job)


def test_a_batch_paused_during_a_download_in_the_turn_parks_after_the_track(rig, two_stream_file, tmp_path,
                                                                          monkeypatch):
    dl = _Download()
    rig.gate.go.set()                              # no hold on the chunks: only the download stops
    monkeypatch.setattr(transcribe.modelstore, "ensure", dl)
    reg, batch = _batch(two_stream_file, tmp_path, per_file=True)
    try:
        assert dl.reached.wait(20)
        reg.pause(batch.id)
        dl.go.set()
        assert _until(lambda: batch.gpu_freed)
        assert dl.n == 3 and rig.gate.n == rig.chunks
        assert not transcribe.GPU_TURN.locked()
        _finish_batch(reg, batch)
        assert batch.status is Status.DONE
    finally:
        dl.go.set()
        _finish_batch(reg, batch)


def _waiting_run_pauses_and_cancels_at_once(rig, job, waiting_text):
    """With another run holding the GPU: `job` says it waits, pauses at once (nothing of the other
    run's unloaded), resumes to waiting, and cancels at once. Never runs a chunk."""
    assert _until(lambda: waiting_text in job.message), job.message
    job.pause()
    assert _until(lambda: job.gpu_freed), "a run waiting for the GPU did not pause at once"
    assert rig.pipes == [] and rig.gate.n == 0
    assert not transcribe._unload_after_turn       # nor freed when the other run's turn ends
    job.resume()
    assert _until(lambda: not job.gpu_freed and waiting_text in job.message)
    job.cancel()
    assert _until(lambda: job.status is JobStatus.CANCELED), "a run waiting for the GPU did not cancel at once"
    assert rig.gate.n == 0


def test_label_track_waiting_for_the_gpu_pauses_and_cancels_at_once(rig, tmp_path):
    wav = tmp_path / "track.wav"
    sf.write(str(wav), np.full(32000, 0.1, dtype=np.float32), 16000)
    transcribe.GPU_TURN.acquire()                  # another run has the GPU, to the end of this test
    job = None
    try:
        job = _job_run(lambda progress: transcribe_regions(
            str(wav), {"A": [(0.0, 1.0)], "B": [(1.0, 2.0)]}, _transcribe_fn, progress=progress))
        _waiting_run_pauses_and_cancels_at_once(rig, job, WAITING)
    finally:
        transcribe.GPU_TURN.release()
        _settle(job)


def test_a_benchmark_waiting_for_the_gpu_pauses_and_cancels_at_once(rig, monkeypatch):
    monkeypatch.setattr(benchmark, "_tests", lambda tmp, names: [
        ("English", benchmark.SENTENCE, np.zeros(16000, dtype=np.float32), "en", names)])
    transcribe.GPU_TURN.acquire()
    job = None
    try:
        job = _job_run(lambda progress: benchmark.run(progress, names=["small", "medium"]))
        _waiting_run_pauses_and_cancels_at_once(rig, job, "Waiting for the current transcription to finish")
    finally:
        transcribe.GPU_TURN.release()
        _settle(job)


def test_label_track_waiting_for_the_gpu_goes_on_when_it_gets_it(rig, tmp_path):
    """The wait report hands back to the speaker's line once the turn comes, and the speaker runs."""
    wav = tmp_path / "track.wav"
    sf.write(str(wav), np.full(32000, 0.1, dtype=np.float32), 16000)
    rig.gate.go.set()                              # no hold: just run
    transcribe.GPU_TURN.acquire()
    job = None
    try:
        job = _job_run(lambda progress: transcribe_regions(
            str(wav), {"A": [(0.0, 1.0)]}, _transcribe_fn, progress=progress))
        assert _until(lambda: WAITING in job.message), job.message
    finally:
        transcribe.GPU_TURN.release()
    assert _until(lambda: job.status is JobStatus.DONE), job.error
    assert len(job.result.segments) == rig.chunks


# --- 5. a one-transcript-per-file project canceled mid-track --------------------------------------

def _per_file_batch(rig, two_stream_file, files=2):
    """A per-file project of `files` two-track files (the same audio, so a later file reuses what an
    earlier one cached), run with a cache as the /api/batch route does. Six chunks a track, a model
    call per chunk: chunks 1-6 are the first file's track 1, 7-12 its track 2."""
    from app.cache import Cache
    rig.chunks, rig.batch = 6, 1
    cache = Cache(rig.tmp / "cache")
    fn = lambda wav: transcribe.transcribe_track(wav, language="is", word_timing=False, cache=cache)
    tracks = [Track(track_index=0, speaker="A"), Track(track_index=1, speaker="B")]
    proj = ProjectInput("P", [PreparedSource(str(two_stream_file), InputSource(
        filename=f"f{k}.mkv", duration=2.0, tracks=tracks)) for k in range(files)], per_file=True)
    reg = BatchRegistry()
    batch = reg.create([proj])
    batch.on_pause = transcribe.unload_models
    reg.run(batch, lambda: run_batch(batch, [proj], fn, output_mode="folder",
                                     output_folder=str(rig.tmp / "out")))
    return reg, batch, cache


HALF_WAY = "canceled · 1 of 2 tracks transcribed, kept for a retry"


def test_a_file_canceled_mid_track_stops_at_the_next_chunk_and_the_others_go_on(rig, two_stream_file):
    """Canceled during its second track's second chunk: it stops at that chunk, writes nothing, keeps
    the finished track in the cache (not the interrupted one), and the next file runs."""
    rig.gate = _TwoHolds(8, 9)                     # chunk 9 is only reached by the next file
    reg, batch, cache = _per_file_batch(rig, two_stream_file)
    ps = batch.projects[0]
    try:
        assert rig.gate.reached.wait(20)
        assert reg.cancel_file(batch.id, 0, 0)
        rig.gate.go.set()
        assert rig.gate.reached2.wait(5), "the next file never started"
        f0 = ps.files[0]
        assert f0.status is Status.CANCELED and f0.message == HALF_WAY, (f0.status, f0.message)
        assert f0.output_dir is None and f0.result is None             # nothing written
        assert cache.stats()["entries"] == 1                           # track 1 only
        assert ps.files[1].status is Status.RUNNING and ps.files[1].track == 2   # its track 1 reused
        rig.gate.go2.set()
        assert _until(lambda: batch.status is Status.DONE), batch.status
        assert ps.files[1].status is Status.DONE and ps.message == "1 done · 1 canceled"
        assert not list((rig.tmp / "out").rglob("f0*"))
    finally:
        rig.gate.go2.set()
        _finish_batch(reg, batch)


def test_a_project_canceled_mid_track_stops_at_the_next_chunk(rig, two_stream_file):
    rig.gate = _Gate(8)
    reg, batch, cache = _per_file_batch(rig, two_stream_file)
    ps = batch.projects[0]
    try:
        assert rig.gate.reached.wait(20)
        assert reg.cancel_project(batch.id, 0)
        rig.gate.go.set()
        assert _until(lambda: batch.status is Status.CANCELED), batch.status
        assert rig.gate.n == 8                                         # stopped at that chunk
        assert ps.status is Status.CANCELED
        assert [(f.status, f.message) for f in ps.files] == [(Status.CANCELED, HALF_WAY),
                                                             (Status.CANCELED, "canceled")]
        assert cache.stats()["entries"] == 1 and not (rig.tmp / "out").exists()
    finally:
        _finish_batch(reg, batch)


def test_a_batch_canceled_mid_track_stops_at_the_next_chunk(rig, two_stream_file):
    rig.gate = _Gate(8)
    reg, batch, cache = _per_file_batch(rig, two_stream_file)
    ps = batch.projects[0]
    try:
        assert rig.gate.reached.wait(20)
        assert reg.cancel(batch.id)
        rig.gate.go.set()
        assert _until(lambda: batch.status is Status.CANCELED), batch.status
        assert rig.gate.n == 8
        assert ps.status is Status.CANCELED
        assert [(f.status, f.message) for f in ps.files] == [(Status.CANCELED, HALF_WAY),
                                                             (Status.CANCELED, "canceled")]
        assert cache.stats()["entries"] == 1 and not (rig.tmp / "out").exists()
        assert not transcribe.GPU_TURN.locked()
    finally:
        _finish_batch(reg, batch)


def test_a_per_file_project_paused_mid_track_still_waits_for_the_end_of_the_track(rig, two_stream_file):
    """The cancel check on every chunk doesn't park: a pause asked for at chunk 2 goes on through
    the track ("pausing", the turn held) and lands only once its sixth chunk is done."""
    rig.gate = _Look(2, 4, None)
    reg, batch, cache = _per_file_batch(rig, two_stream_file, files=1)
    rig.gate.look = lambda: (batch.paused, batch.gpu_freed, transcribe.GPU_TURN.locked())
    try:
        assert _pause_batch_mid_chunk(rig, reg, batch) == 6
        assert rig.gate.seen == (True, False, True)                    # pausing, mid-track
        assert batch.projects[0].files[0].track == 1                   # track 2 hasn't started
        assert not transcribe.GPU_TURN.locked() and cache.stats()["entries"] == 1
        _finish_batch(reg, batch)
        assert batch.status is Status.DONE and batch.projects[0].files[0].status is Status.DONE
        assert rig.gate.n == 12
    finally:
        _finish_batch(reg, batch)


def test_a_file_canceled_mid_track_while_the_batch_is_pausing_reads_canceled_at_once(rig, two_stream_file):
    """The file stops at the next chunk and says so; then the batch parks for its pause, before the
    next file, and that file runs after resume."""
    rig.gate = _Gate(8)
    reg, batch, cache = _per_file_batch(rig, two_stream_file)
    ps = batch.projects[0]
    try:
        assert rig.gate.reached.wait(20)
        reg.pause(batch.id)
        reg.cancel_file(batch.id, 0, 0)
        rig.gate.go.set()
        assert _until(lambda: batch.gpu_freed), "the paused batch never parked"
        assert (ps.files[0].status, ps.files[0].message) == (Status.CANCELED, HALF_WAY)
        assert ps.files[1].status is not Status.DONE
        _finish_batch(reg, batch)
        assert batch.status is Status.DONE and ps.files[1].status is Status.DONE
    finally:
        _finish_batch(reg, batch)


def test_a_job_paused_then_canceled_leaves_nothing_hanging(rig, two_stream_file):
    """Parked for its pause ("GPU released"), then canceled: the job ends canceled, its thread is
    gone, the turn is free, no model is left loaded and its clock isn't counting paused time."""
    job = _transcribe_job(two_stream_file)
    try:
        _pause_job_mid_chunk(rig, job)
        job.cancel()
        assert _until(lambda: job.status is JobStatus.CANCELED), job.status
        job._thread.join(3)
        assert not job._thread.is_alive()
        assert not job.gpu_freed and not job.pause_started and not transcribe.GPU_TURN.locked()
        assert not [p for p in rig.pipes if p.model.model.loaded] and transcribe._loaded == []
    finally:
        job.resume()
        _settle(job)


# --- 6. one project or one file canceled while the batch is paused or pausing ------------------

def test_a_file_canceled_while_the_batch_is_parked_stops_at_once(rig, two_stream_file):
    """Seen live: the canceled file waited for Resume, then even started its next track. Now it
    ends canceled at once, and the batch stays paused: the next file waits for Resume."""
    rig.gate = _Gate(3)
    reg, batch, cache = _per_file_batch(rig, two_stream_file)
    ps = batch.projects[0]
    try:
        assert _pause_batch_mid_chunk(rig, reg, batch) == 6           # parked after track 1
        assert reg.cancel_file(batch.id, 0, 0)
        assert _until(lambda: ps.files[0].status is Status.CANCELED, 1.5), ps.files[0].message
        assert ps.files[0].message == HALF_WAY and ps.files[0].result is None
        assert _until(lambda: batch.gpu_freed)                         # parked again, before file 2
        time.sleep(0.2)
        assert rig.gate.n == 6 and ps.files[1].status is not Status.DONE   # nothing else ran
        assert batch.status is Status.RUNNING and batch.paused
        _finish_batch(reg, batch)
        assert batch.status is Status.DONE and ps.files[1].status is Status.DONE
        assert not list((rig.tmp / "out").rglob("f0*"))
    finally:
        _finish_batch(reg, batch)


def test_a_project_canceled_while_the_batch_is_parked_stops_at_once(rig, two_stream_file):
    """It ends canceled at once, with how far it got; it was the last thing to do, so the batch
    ends too, without a Resume."""
    rig.gate = _Gate(3)
    reg, batch, cache = _per_file_batch(rig, two_stream_file)
    ps = batch.projects[0]
    try:
        assert _pause_batch_mid_chunk(rig, reg, batch) == 6
        assert reg.cancel_project(batch.id, 0)
        assert _until(lambda: batch.status is Status.CANCELED, 1.5), (batch.status, ps.message)
        assert ps.status is Status.CANCELED and rig.gate.n == 6
        assert [(f.status, f.message) for f in ps.files] == [(Status.CANCELED, HALF_WAY),
                                                             (Status.CANCELED, "canceled")]
        assert not (rig.tmp / "out").exists() and not transcribe.GPU_TURN.locked()
    finally:
        _finish_batch(reg, batch)


def test_the_last_file_canceled_while_the_batch_is_pausing_ends_the_batch(rig, two_stream_file):
    """Canceled mid-track while "pausing": it stops at the next chunk, and with nothing left to do
    the batch ends instead of parking until Resume."""
    rig.gate = _Gate(8)
    reg, batch, cache = _per_file_batch(rig, two_stream_file, files=1)
    f = batch.projects[0].files[0]
    try:
        assert rig.gate.reached.wait(20)
        reg.pause(batch.id)
        reg.cancel_file(batch.id, 0, 0)
        rig.gate.go.set()
        assert _until(lambda: batch.status is Status.CANCELED, 1.5), (batch.status, batch.gpu_freed)
        assert (f.status, f.message) == (Status.CANCELED, HALF_WAY) and rig.gate.n == 8
    finally:
        _finish_batch(reg, batch)


def test_a_joined_project_canceled_while_the_batch_is_pausing_stops_at_once(rig, two_stream_file, tmp_path):
    """Pause, then cancel the project, after its first track's last chunk (as during word timing):
    the project used to park for the pause first and stay "running" until Resume."""
    reg = BatchRegistry()
    tracks = [Track(track_index=0, speaker="A"), Track(track_index=1, speaker="B")]
    proj = ProjectInput("P", [PreparedSource(str(two_stream_file), InputSource(
        filename="a.mkv", duration=2.0, tracks=tracks))])
    batch = reg.create([proj])
    batch.on_pause = transcribe.unload_models
    calls = []

    def fn(wav):
        out = _transcribe_fn(wav)
        calls.append(wav)
        if len(calls) == 1:                      # the track is done; the next checkpoint is next
            reg.pause(batch.id)
            reg.cancel_project(batch.id, 0)
        return out

    rig.gate.go.set()
    reg.run(batch, lambda: run_batch(batch, [proj], fn, output_mode="folder",
                                     output_folder=str(tmp_path / "out")))
    ps = batch.projects[0]
    try:
        assert _until(lambda: batch.status is Status.CANCELED, 10), (batch.status, batch.gpu_freed, ps.message)
        assert (ps.status, ps.message) == (Status.CANCELED, HALF_WAY) and len(calls) == 1
        assert not (tmp_path / "out").exists()
    finally:
        _finish_batch(reg, batch)
