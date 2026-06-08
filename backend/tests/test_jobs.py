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
