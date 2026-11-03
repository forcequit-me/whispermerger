"""In-process job registry + progress (PRD §11.1.5: start a job, poll progress, get result).

Single-user local app, so no Celery/Redis — one background thread per job, polled over HTTP.
WhisperX transcription is blocking/GPU-bound, so a thread (not asyncio) is the right fit.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from . import modelstore
from .models import MergedOutput


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


class JobCancelled(Exception):
    """Raised through the progress callback when a job cancel is requested."""


# A pipeline reports progress through this callback: fraction in [0,1] + a status message.
ProgressCb = Callable[[float, str], None]


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    message: str = ""
    result: Optional[MergedOutput] = None
    error: Optional[str] = None
    started_at: float = 0.0
    finished_at: float = 0.0
    paused: bool = False
    gpu_freed: bool = False                       # the GPU has actually been released (worker parked)
    paused_total: float = 0.0                     # total wall time spent paused (excluded from elapsed)
    pause_started: float = 0.0                    # start of the current pause, 0 when running
    _thread: Optional[threading.Thread] = field(default=None, repr=False)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _wake: threading.Event = field(default_factory=threading.Event, repr=False)
    _on_pause: Optional[Callable[[], None]] = field(default=None, repr=False)

    def cancel(self) -> None:
        """Request cancellation; takes effect at the next progress checkpoint (stage boundary)."""
        self._cancel.set()
        self._wake.set()                          # wake a paused job so it can unwind

    def pause(self) -> None:
        """Request a pause; takes effect at the next progress checkpoint (the in-flight stage
        finishes first), then the GPU is freed and the clock stops until resume()."""
        self.paused = True

    def resume(self) -> None:
        self.paused = False
        self._wake.set()

    def _set_progress(self, fraction: float, message: str) -> None:
        if self.paused and not self._cancel.is_set():
            if self._on_pause:
                self._on_pause()                  # free the GPU (runs on the worker thread)
            self.gpu_freed = True
            self.pause_started = time.time()
            self.message = "paused · GPU released"
            while self.paused and not self._cancel.is_set():
                self._wake.wait()
                self._wake.clear()
            if self.pause_started:                # bank the paused gap so elapsed excludes it
                self.paused_total += time.time() - self.pause_started
                self.pause_started = 0.0
            self.gpu_freed = False
        if self._cancel.is_set():
            raise JobCancelled()
        self.progress = max(0.0, min(1.0, fraction))
        self.message = message

    @property
    def elapsed(self) -> float:
        """Total wall time across all stages (extraction + transcription + merge), minus any time
        spent paused (the GPU is idle then, so it would otherwise inflate the reported duration)."""
        if not self.started_at:
            return 0.0
        end = self.finished_at or time.time()
        ongoing = (time.time() - self.pause_started) if self.pause_started else 0.0
        return max(0.0, end - self.started_at - self.paused_total - ongoing)


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        job = Job(id=uuid.uuid4().hex)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def run(self, job: Job, work: Callable[[ProgressCb], MergedOutput],
            *, on_pause: Optional[Callable[[], None]] = None) -> None:
        """Run `work` in a background thread, streaming progress into the job. `on_pause` (if given)
        runs on the worker thread when the job parks on a pause — used to free the GPU."""
        job._on_pause = on_pause

        def _target() -> None:
            job.started_at = time.time()
            job.status = JobStatus.RUNNING
            try:
                # A model download inside the work reports on the job's progress line; that same
                # call is the pause/cancel checkpoint, so Cancel also stops the download.
                with modelstore.reporting(lambda msg: job._set_progress(job.progress, msg)):
                    job.result = work(job._set_progress)
                job.finished_at = time.time()
                job.progress = 1.0
                job.message = f"done in {job.elapsed:.0f}s"
                job.status = JobStatus.DONE
            except JobCancelled:  # cooperative cancel at a stage boundary
                job.finished_at = time.time()
                job.message = "cancelled"
                job.status = JobStatus.CANCELLED
            except Exception as exc:  # surface failure via the job, not a crashed thread
                job.finished_at = time.time()
                job.error = str(exc)
                job.status = JobStatus.ERROR

        job._thread = threading.Thread(target=_target, daemon=True)
        job._thread.start()
