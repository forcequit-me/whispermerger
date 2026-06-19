"""Batch orchestration — the "transcription studio" (Method 2).

A batch is a sequential queue of projects. Each project is an ordered list of source files
that merge into one transcript (Method 1, via `run_merge_pipeline`), then auto-export all
formats to its own subfolder. Single GPU, so projects run one at a time; a failing project
is recorded and the queue continues. Caching makes a re-run skip already-transcribed audio.
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from .export import export_all
from .merge import MergeOptions
from .pipeline import PreparedSource, run_merge_pipeline, run_per_file_pipeline

TranscribeFn = Callable[[str], dict]


class Status(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELED = "canceled"


class Cancelled(Exception):
    """Raised inside the pipeline (via the progress callback) when a cancel is requested."""


@dataclass
class ProjectInput:
    name: str
    prepared: list[PreparedSource]
    per_file: bool = False                       # one transcript per source file (vs merge/sequence)
    formats: Optional[list[str]] = None          # which exports to write; None = all four


@dataclass
class ProjectStatus:
    name: str
    status: Status = Status.QUEUED
    progress: float = 0.0
    message: str = ""
    output_dir: Optional[str] = None
    error: Optional[str] = None
    started_at: float = 0.0
    finished_at: float = 0.0
    cancel_requested: bool = False               # cancel just this project (queue continues)

    @property
    def elapsed(self) -> float:
        if not self.started_at:
            return 0.0
        return (self.finished_at or time.time()) - self.started_at


@dataclass
class BatchStatus:
    id: str
    status: Status = Status.QUEUED
    projects: list[ProjectStatus] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0
    cancelled: bool = False
    # Project indices requested for re-run while the worker is busy — the worker drains these
    # when it frees up (one GPU, so re-runs queue instead of starting a competing thread).
    rerun_pending: list[int] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def progress(self) -> float:
        """Overall fraction: completed projects + the running one's internal progress."""
        if not self.projects:
            return 0.0
        return sum(p.progress for p in self.projects) / len(self.projects)

    @property
    def elapsed(self) -> float:
        if not self.started_at:
            return 0.0
        return (self.finished_at or time.time()) - self.started_at


def _safe_name(name: str) -> str:
    """Filesystem-safe folder/file base name."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip().strip(".")
    return cleaned or "project"


def transcript_basename(base: str, when: Optional[float] = None) -> str:
    """Self-describing export name: `<base> Transcript (M-D-YYYY h-mm AMPM)`, e.g.
    `Project 1 Transcript (6-19-2026 2-02 AM)`. Dash (not colon) in the time so it's a legal
    Windows filename. `when` defaults to now — pass it once per file so all formats share a stamp."""
    t = time.localtime(when)
    h = t.tm_hour % 12 or 12
    ap = "AM" if t.tm_hour < 12 else "PM"
    return f"{_safe_name(base)} Transcript ({t.tm_mon}-{t.tm_mday}-{t.tm_year} {h}-{t.tm_min:02d} {ap})"


def _project_output_parent(project: ProjectInput, output_mode: str, output_folder: Optional[str]) -> Path:
    if output_mode == "folder":
        if not output_folder:
            raise ValueError("output_folder required when output_mode='folder'")
        return Path(output_folder)
    # "source": alongside the project's first source file
    return Path(project.prepared[0].path).parent


IsCancelled = Callable[[], bool]


def _guard(cb, is_cancelled: IsCancelled):
    """Wrap a progress callback so every stage boundary is a cooperative cancel checkpoint."""
    def wrapped(frac: float, msg: str) -> None:
        if is_cancelled():
            raise Cancelled()
        cb(frac, msg)
    return wrapped


def _process_project(
    project: ProjectInput,
    ps: ProjectStatus,
    transcribe_fn: TranscribeFn,
    *,
    output_mode: str,
    output_folder: Optional[str],
    options: Optional[MergeOptions],
    is_cancelled: IsCancelled,
) -> None:
    """Run one project to disk, mutating its ProjectStatus. Raises Cancelled if a cancel is
    requested at a file/stage boundary; any other failure is recorded on the status."""
    ps.started_at = time.time()
    ps.finished_at = 0.0
    ps.error = None
    ps.progress = 0.0
    ps.status = Status.RUNNING
    try:
        parent = _project_output_parent(project, output_mode, output_folder)
        base = _safe_name(project.name)
        out_dir = parent / base
        if project.per_file:
            # One transcript per source file. In "source" mode each lands next to its own
            # source file (so a nested folder tree is preserved); otherwise all go to the
            # project's output folder. Prefetch overlaps the next file's extraction with the
            # current file's transcription (run_per_file_pipeline); on_file exports each result.
            seen: dict[tuple[str, str], int] = {}

            def on_file(i: int, src: PreparedSource, merged, seen=seen) -> None:
                dest = Path(src.path).parent if output_mode == "source" else out_dir
                stem = _safe_name(Path(src.input.filename).stem)
                seen[(str(dest), stem)] = seen.get((str(dest), stem), 0) + 1
                if seen[(str(dest), stem)] > 1:
                    stem = f"{stem}_{seen[(str(dest), stem)]}"
                export_all(merged, dest, transcript_basename(stem), formats=project.formats)

            run_per_file_pipeline(
                project.prepared, transcribe_fn, on_file, options=options,
                progress_cb=_guard(lambda frac, msg: _set(ps, frac, msg), is_cancelled),
                is_cancelled=is_cancelled)
            ps.output_dir = str(parent if output_mode == "source" else out_dir)
        else:
            merged = run_merge_pipeline(
                project.prepared, transcribe_fn, options=options,
                progress_cb=_guard(lambda frac, msg: _set(ps, frac, msg), is_cancelled),
                is_cancelled=is_cancelled)
            export_all(merged, out_dir, transcript_basename(base), formats=project.formats)
            ps.output_dir = str(out_dir)
        ps.finished_at = time.time()
        ps.progress = 1.0
        ps.message = f"done in {ps.elapsed:.0f}s"
        ps.status = Status.DONE
    except Cancelled:
        ps.finished_at = time.time()
        ps.message = "canceled"
        ps.status = Status.CANCELED
        raise
    except Exception as exc:  # record + let the queue continue to the next project
        ps.finished_at = time.time()
        ps.error = str(exc)
        ps.status = Status.ERROR


def _run_projects(
    pairs: list[tuple[ProjectInput, ProjectStatus]],
    transcribe_fn: TranscribeFn,
    *,
    output_mode: str,
    output_folder: Optional[str],
    options: Optional[MergeOptions],
    is_cancelled: IsCancelled,
) -> None:
    """Process (project, status) pairs in order; stop at the next boundary on cancel and
    mark any not-yet-finished ones CANCELED."""
    for project, ps in pairs:
        if is_cancelled():
            break
        if ps.cancel_requested:                       # this item canceled before it ran — skip it
            ps.status = Status.CANCELED
            ps.message = "canceled"
            continue

        def proj_cancelled(ps=ps) -> bool:            # batch-wide OR this one project
            return is_cancelled() or ps.cancel_requested

        try:
            _process_project(project, ps, transcribe_fn, output_mode=output_mode,
                             output_folder=output_folder, options=options,
                             is_cancelled=proj_cancelled)
        except Cancelled:
            if is_cancelled():                        # whole-batch cancel stops the queue
                break
            # single-item cancel: this project is marked CANCELED, the queue continues
    if is_cancelled():
        for _, ps in pairs:
            if ps.status in (Status.QUEUED, Status.RUNNING):
                ps.status = Status.CANCELED
                ps.message = "canceled"


def run_batch(
    batch: BatchStatus,
    projects: list[ProjectInput],
    transcribe_fn: TranscribeFn,
    *,
    output_mode: str,
    output_folder: Optional[str] = None,
    options: Optional[MergeOptions] = None,
) -> None:
    """Process every project in order, exporting each to disk. Mutates `batch` in place;
    stops at the next file/project boundary if `batch.cancelled` is set."""
    _run_projects(list(zip(projects, batch.projects)), transcribe_fn,
                  output_mode=output_mode, output_folder=output_folder,
                  options=options, is_cancelled=lambda: batch.cancelled)


def _set(ps: ProjectStatus, frac: float, msg: str) -> None:
    ps.progress = max(0.0, min(1.0, frac))
    ps.message = msg


@dataclass
class _RunSpec:
    """Everything needed to re-run a batch's projects (for retry)."""
    projects: list[ProjectInput]
    transcribe_fn: TranscribeFn
    output_mode: str
    output_folder: Optional[str]
    options: Optional[MergeOptions]


class BatchRegistry:
    def __init__(self) -> None:
        self._batches: dict[str, BatchStatus] = {}
        self._specs: dict[str, _RunSpec] = {}
        self._lock = threading.Lock()

    def create(self, project_names: list[str]) -> BatchStatus:
        batch = BatchStatus(id=uuid.uuid4().hex,
                            projects=[ProjectStatus(name=n) for n in project_names])
        with self._lock:
            self._batches[batch.id] = batch
        return batch

    def get(self, batch_id: str) -> Optional[BatchStatus]:
        with self._lock:
            return self._batches.get(batch_id)

    def set_spec(self, batch_id: str, spec: _RunSpec) -> None:
        with self._lock:
            self._specs[batch_id] = spec

    def cancel(self, batch_id: str) -> bool:
        """Request cancellation; the worker stops at the next file/project boundary."""
        batch = self.get(batch_id)
        if batch is None:
            return False
        batch.cancelled = True
        return True

    def cancel_project(self, batch_id: str, index: int) -> bool:
        """Cancel ONE project — the queue continues to the next. Takes effect at that project's
        next stage boundary (or immediately if it hasn't started yet)."""
        batch = self.get(batch_id)
        if batch is None or not (0 <= index < len(batch.projects)):
            return False
        batch.projects[index].cancel_requested = True
        return True

    @staticmethod
    def _reset_ps(ps: ProjectStatus) -> None:
        """Put a project back to a fresh QUEUED state so it can run (again)."""
        ps.status = Status.QUEUED
        ps.error = None
        ps.progress = 0.0
        ps.message = ""
        ps.started_at = 0.0
        ps.finished_at = 0.0
        ps.cancel_requested = False

    def _run_indices(self, batch: BatchStatus, spec: _RunSpec, indices: list[int]) -> None:
        pairs = [(spec.projects[i], batch.projects[i]) for i in indices]
        _run_projects(pairs, spec.transcribe_fn, output_mode=spec.output_mode,
                      output_folder=spec.output_folder, options=spec.options,
                      is_cancelled=lambda: batch.cancelled)

    def run(self, batch: BatchStatus, work: Callable[[], None]) -> None:
        """Run `work` on a background thread, then keep draining any re-runs requested while it
        ran (so a single worker, single GPU, processes everything in sequence)."""
        def _target() -> None:
            batch.started_at = time.time()
            batch.status = Status.RUNNING
            try:
                work()
                while True:
                    spec = self._specs.get(batch.id)
                    with batch.lock:
                        pending = [] if (spec is None or batch.cancelled) else batch.rerun_pending
                        batch.rerun_pending = []
                        if not pending:
                            # Finalize under the lock so a re-run request can't slip in after we've
                            # decided to stop (it would see RUNNING, enqueue, and never get drained).
                            batch.finished_at = time.time()
                            batch.status = Status.CANCELED if batch.cancelled else Status.DONE
                            return
                    self._run_indices(batch, spec, pending)
            finally:
                if batch.status is Status.RUNNING:   # safety net if work() raised unexpectedly
                    batch.finished_at = time.time()
                    batch.status = Status.CANCELED if batch.cancelled else Status.DONE
        threading.Thread(target=_target, daemon=True).start()

    def retry(self, batch_id: str, indices: list[int]) -> bool:
        """Re-run the given project indices, reusing the same BatchStatus so the UI keeps polling
        one batch. If the batch is still running, the indices are queued and run when the GPU frees
        up; if idle, a worker starts immediately. False only on unknown batch / bad indices."""
        batch = self.get(batch_id)
        spec = self._specs.get(batch_id)
        if batch is None or spec is None:
            return False
        if not all(0 <= i < len(batch.projects) for i in indices):
            return False
        for i in indices:
            self._reset_ps(batch.projects[i])
        start = False
        with batch.lock:
            for i in indices:
                if i not in batch.rerun_pending:
                    batch.rerun_pending.append(i)
            if batch.status is not Status.RUNNING:   # claim it so a second caller won't also start
                batch.status = Status.RUNNING
                batch.cancelled = False
                start = True
        if start:
            self.run(batch, lambda: None)            # worker drains rerun_pending
        return True
