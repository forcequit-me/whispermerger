"""In-process job registry + progress (PRD §11.1.5: start a job, poll progress, get result).

Single-user local app, so no Celery/Redis — one background thread per job, polled over HTTP.
WhisperX transcription is blocking/GPU-bound, so a thread (not asyncio) is the right fit.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from .models import MergedOutput


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


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
    _thread: Optional[threading.Thread] = field(default=None, repr=False)

    def _set_progress(self, fraction: float, message: str) -> None:
        self.progress = max(0.0, min(1.0, fraction))
        self.message = message


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

    def run(self, job: Job, work: Callable[[ProgressCb], MergedOutput]) -> None:
        """Run `work` in a background thread, streaming progress into the job."""
        def _target() -> None:
            job.status = JobStatus.RUNNING
            try:
                job.result = work(job._set_progress)
                job._set_progress(1.0, "done")
                job.status = JobStatus.DONE
            except Exception as exc:  # surface failure via the job, not a crashed thread
                job.error = str(exc)
                job.status = JobStatus.ERROR

        job._thread = threading.Thread(target=_target, daemon=True)
        job._thread.start()
