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
from .pipeline import PreparedSource, run_merge_pipeline

TranscribeFn = Callable[[str], dict]


class Status(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


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


def _project_output_parent(project: ProjectInput, output_mode: str, output_folder: Optional[str]) -> Path:
    if output_mode == "folder":
        if not output_folder:
            raise ValueError("output_folder required when output_mode='folder'")
        return Path(output_folder)
    # "source": alongside the project's first source file
    return Path(project.prepared[0].path).parent


def run_batch(
    batch: BatchStatus,
    projects: list[ProjectInput],
    transcribe_fn: TranscribeFn,
    *,
    output_mode: str,
    output_folder: Optional[str] = None,
    options: Optional[MergeOptions] = None,
) -> None:
    """Process every project in order, exporting each to disk. Mutates `batch` in place."""
    for project, ps in zip(projects, batch.projects):
        ps.started_at = time.time()
        ps.status = Status.RUNNING
        try:
            parent = _project_output_parent(project, output_mode, output_folder)
            base = _safe_name(project.name)
            out_dir = parent / base
            if project.per_file:
                # One transcript per source file, all into the project folder.
                n = len(project.prepared) or 1
                seen: dict[str, int] = {}
                for i, src in enumerate(project.prepared):
                    name = src.input.filename

                    def cb(frac: float, msg: str, i=i, name=name) -> None:
                        _set(ps, (i + frac) / n, f"file {i + 1}/{n}: {name} · {msg}")

                    merged = run_merge_pipeline(
                        [src], transcribe_fn, options=options, progress_cb=cb)
                    stem = _safe_name(Path(name).stem)
                    seen[stem] = seen.get(stem, 0) + 1
                    if seen[stem] > 1:
                        stem = f"{stem}_{seen[stem]}"
                    export_all(merged, out_dir, stem, formats=project.formats)
            else:
                merged = run_merge_pipeline(
                    project.prepared, transcribe_fn,
                    options=options,
                    progress_cb=lambda frac, msg: _set(ps, frac, msg),
                )
                export_all(merged, out_dir, base, formats=project.formats)
            ps.output_dir = str(out_dir)
            ps.finished_at = time.time()
            ps.progress = 1.0
            ps.message = f"done in {ps.elapsed:.0f}s"
            ps.status = Status.DONE
        except Exception as exc:  # record + continue to the next project
            ps.finished_at = time.time()
            ps.error = str(exc)
            ps.status = Status.ERROR


def _set(ps: ProjectStatus, frac: float, msg: str) -> None:
    ps.progress = max(0.0, min(1.0, frac))
    ps.message = msg


class BatchRegistry:
    def __init__(self) -> None:
        self._batches: dict[str, BatchStatus] = {}
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

    def run(self, batch: BatchStatus, work: Callable[[], None]) -> None:
        def _target() -> None:
            batch.started_at = time.time()
            batch.status = Status.RUNNING
            try:
                work()
            finally:
                batch.finished_at = time.time()
                batch.status = Status.DONE  # per-project errors are tracked individually
        threading.Thread(target=_target, daemon=True).start()
