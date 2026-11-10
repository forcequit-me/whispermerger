"""Job registry tests — lifecycle, progress, error capture (pure Python, no GPU)."""

import time

from app.jobs import JobRegistry, JobStatus
from app.models import MergedOutput, Segment


def _wait(job, timeout=2.0):
    deadline = time.time() + timeout
    while job.status in (JobStatus.PENDING, JobStatus.RUNNING) and time.time() < deadline:
        time.sleep(0.01)


def test_successful_job_sets_result_and_progress():
    reg = JobRegistry()
    job = reg.create()

    def work(progress):
        progress(0.5, "halfway")
        return MergedOutput(segments=[Segment(speaker="Max", start=0, end=1, text="hi")])

    reg.run(job, work)
    _wait(job)
    assert job.status is JobStatus.DONE
    assert job.progress == 1.0
    assert job.result.segments[0].text == "hi"


def test_job_says_when_every_track_was_reused():
    from app import transcribe
    reg = JobRegistry()
    job = reg.create()

    def work(progress):                                # what transcribe_track does on a cache hit
        transcribe._chunk.hits = transcribe.cache_tally()[0] + 1
        return MergedOutput(segments=[])

    reg.run(job, work)
    _wait(job)
    assert job.reused and job.message == "reused saved result"


def test_failed_job_captures_error():
    reg = JobRegistry()
    job = reg.create()

    def work(progress):
        raise RuntimeError("boom")

    reg.run(job, work)
    _wait(job)
    assert job.status is JobStatus.ERROR
    assert "boom" in job.error
    assert job.result is None


def test_get_unknown_job_returns_none():
    assert JobRegistry().get("nope") is None


def test_cancel_sets_cancelled_status():
    reg = JobRegistry()
    job = reg.create()

    def work(progress):
        progress(0.1, "stage one")      # checkpoint before cancel — proceeds
        job.cancel()                     # request cancellation
        progress(0.2, "stage two")      # next checkpoint raises -> cancelled
        return MergedOutput()

    reg.run(job, work)
    _wait(job)
    assert job.status is JobStatus.CANCELLED
    assert job.result is None


def _wait_freed(job, timeout=3.0):
    deadline = time.time() + timeout
    while not job.gpu_freed and time.time() < deadline:
        time.sleep(0.01)


def test_pause_frees_gpu_freezes_clock_then_resumes():
    reg = JobRegistry()
    job = reg.create()
    freed = []

    def work(progress):
        progress(0.1, "one")             # parks here (pause already requested)
        progress(0.9, "two")
        return MergedOutput(segments=[Segment(speaker="A", start=0, end=1, text="hi")])

    job.pause()                          # pause up front so the first checkpoint parks
    reg.run(job, work, on_pause=lambda: freed.append(True))
    _wait_freed(job)
    assert job.gpu_freed is True
    assert freed == [True]               # on_pause ran -> GPU released

    e1 = job.elapsed
    time.sleep(0.2)                      # paused time must not count
    assert job.elapsed - e1 < 0.1        # clock frozen while paused

    job.resume()
    _wait(job)
    assert job.status is JobStatus.DONE
    assert job.result.segments[0].text == "hi"
    assert job.paused_total >= 0.15      # the pause was banked out of elapsed


def test_cancel_while_paused_unwinds():
    reg = JobRegistry()
    job = reg.create()

    def work(progress):
        progress(0.1, "one")             # parks here
        return MergedOutput()

    job.pause()
    reg.run(job, work, on_pause=lambda: None)
    _wait_freed(job)
    assert job.gpu_freed is True
    job.cancel()                         # cancel while paused -> wakes and unwinds
    _wait(job)
    assert job.status is JobStatus.CANCELLED
