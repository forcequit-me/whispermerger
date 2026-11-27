"""Pausing a run mid-track: does it give up the GPU turn, and does the track survive it?

The real transcribe_track, JobRegistry, BatchRegistry, GPU_TURN, unload_models and the lru-cached
_load_model; only the speech model is fake (no GPU, no model files). The fake works like a WhisperX
pipeline: transcribe() walks self.__call__, which runs the model one batch of chunks at a time and
hands the results out one by one, so _transcribe_counted's progress hook fires after every chunk.
Under it sits a stand-in for the CTranslate2 model with the one behaviour that matters here: once
unload_model() has run, the next call raises "The model for this translator was unloaded", as
ctranslate2 4.4.0 does (checked on a tiny model on the CPU, 2026-11-27).

Each test asserts what a user would expect. The ones marked xfail fail on today's code."""

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

DONE_STATES = (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED)


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


def _until(check, timeout=3.0) -> bool:
    deadline = time.time() + timeout
    while not check() and time.time() < deadline:
        time.sleep(0.01)
    return check()


def _transcribe_fn(wav):
    """What the routes pass the pipelines (language given, so no Auto-detect)."""
    return transcribe.transcribe_track(wav, language="is", word_timing=False)


def _job_run(work):
    job = JobRegistry().create()
    JobRegistry().run(job, work, on_pause=transcribe.unload_models)    # as the routes do
    return job


def _transcribe_job(two_stream_file):
    """POST /api/transcribe: run_merge_pipeline on one track of a file, in a job."""
    src = InputSource(filename="a.mkv", duration=2.0, tracks=[Track(track_index=0, speaker="A")])
    return _job_run(lambda progress: run_merge_pipeline(
        [PreparedSource(str(two_stream_file), src)], _transcribe_fn, progress_cb=progress))


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

@pytest.mark.xfail(strict=True, reason="a job paused mid-track parks inside its GPU turn (transcribe.py "
                   "_gpu_turn), so every other run waits for the whole pause")
def test_a_job_paused_mid_track_lets_another_run_take_its_turn(rig, two_stream_file):
    job = _transcribe_job(two_stream_file)
    other = None
    try:
        landed = _pause_job_mid_chunk(rig, job)
        assert job.message == "paused · GPU released"
        other, ok, why = _the_other_run_finishes_while_paused(rig, f"after chunk {landed} of {rig.chunks}")
        assert ok, why
        job.resume()
        assert _until(lambda: job.status is JobStatus.DONE), job.error
    finally:
        job.resume()
        _settle(job, other)


def test_a_pause_in_the_last_batch_of_a_track_still_finishes(rig, two_stream_file):
    """Why the live test (2026-11-26) saw pause and resume work on a running job: when every chunk
    left in the track is in the batch the model has already run, nothing calls the unloaded model
    after resume. Its track then finishes, and the next track loads the model again."""
    job = _transcribe_job(two_stream_file)        # 3 chunks, one batch
    try:
        _pause_job_mid_chunk(rig, job)
        assert not rig.pipes[0].model.model.loaded      # the running track's own model was unloaded
        job.resume()
        assert _until(lambda: job.status is JobStatus.DONE), job.error
        assert len(job.result.segments) == 3
    finally:
        job.resume()
        _settle(job)


# --- 3. the model, unloaded under the running track ---------------------------------------------

@pytest.mark.xfail(strict=True, reason="the pause unloads the model pipe.transcribe is using; after resume "
                   "its next batch raises 'The model for this translator was unloaded'")
def test_a_track_paused_before_its_last_batch_finishes_after_resume(rig, two_stream_file):
    rig.chunks, rig.batch = 4, 2                   # two batches: the pause lands in the first
    job = _transcribe_job(two_stream_file)
    try:
        _pause_job_mid_chunk(rig, job)
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


@pytest.mark.xfail(strict=True, reason="a joined project's progress callback (_guard in batch.py) parks on "
                   "every chunk report, inside the GPU turn")
def test_a_joined_batch_project_paused_mid_track_lets_another_run_take_its_turn(rig, two_stream_file, tmp_path):
    reg, batch = _batch(two_stream_file, tmp_path, per_file=False)
    other = None
    try:
        landed = _pause_batch_mid_chunk(rig, reg, batch)
        other, ok, why = _the_other_run_finishes_while_paused(rig, f"after chunk {landed} of {rig.chunks}")
        assert ok, why
        _finish_batch(reg, batch)
        assert batch.status is Status.DONE and batch.projects[0].status is Status.DONE
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
    """POST /api/diarize: its only checkpoint is the report before each speaker, outside the turn."""
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


@pytest.mark.xfail(strict=True, reason="the benchmark holds GPU_TURN for all its models and its checkpoint "
                   "(progress before each model) is inside it, so a pause parks holding the turn")
def test_a_benchmark_paused_between_models_lets_another_run_take_its_turn(rig, monkeypatch):
    monkeypatch.setattr(benchmark, "_tests", lambda tmp, names: [
        ("English", benchmark.SENTENCE, np.zeros(16000, dtype=np.float32), "en", names)])
    job = _job_run(lambda progress: benchmark.run(progress, names=["small", "medium"]))
    other = None
    try:
        landed = _pause_job_mid_chunk(rig, job)
        other, ok, why = _the_other_run_finishes_while_paused(rig, f"after {landed} chunks, before the second model")
        assert ok, why
        job.resume()
        assert _until(lambda: job.status is JobStatus.DONE), job.error
    finally:
        job.resume()
        _settle(job, other)
