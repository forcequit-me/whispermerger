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


def _live_elapsed(started_at: float, finished_at: float,
                  paused_total: float, pause_started: float) -> float:
    """Wall time between start and finish (or now), minus time spent paused. Every timed row uses
    this so a paused run reports real working time, not clock time (the GPU sits idle while paused)."""
    if not started_at:
        return 0.0
    end = finished_at or time.time()
    ongoing = (time.time() - pause_started) if pause_started else 0.0
    return max(0.0, end - started_at - paused_total - ongoing)


@dataclass
class ProjectInput:
    name: str
    prepared: list[PreparedSource]
    per_file: bool = False                       # one transcript per source file (vs merge/sequence)
    formats: Optional[list[str]] = None          # which exports to write; None = all four


@dataclass
class FileStatus:
    """One source file inside a per-file project — its own queue row, status and controls."""
    name: str
    status: Status = Status.QUEUED
    progress: float = 0.0
    message: str = ""
    weight: float = 0.0                          # audio seconds × selected tracks (progress weight)
    track: int = 0                               # 1-based index of the track currently transcribing
    tracks: int = 0                              # total selected tracks in this file
    extract_seconds: float = 0.0                 # wall time waited on extraction (reading source + ffmpeg)
    output_dir: Optional[str] = None
    error: Optional[str] = None
    started_at: float = 0.0
    finished_at: float = 0.0
    cancel_requested: bool = False
    paused_total: float = 0.0                     # total wall time spent paused (excluded from elapsed)
    pause_started: float = 0.0                    # start of the current pause, 0 when running

    @property
    def elapsed(self) -> float:
        return _live_elapsed(self.started_at, self.finished_at, self.paused_total, self.pause_started)


@dataclass
class ProjectStatus:
    name: str
    status: Status = Status.QUEUED
    progress: float = 0.0
    message: str = ""
    weight: float = 0.0                          # total audio seconds × tracks (for batch weighting)
    output_dir: Optional[str] = None
    error: Optional[str] = None
    started_at: float = 0.0
    finished_at: float = 0.0
    cancel_requested: bool = False               # cancel just this project (queue continues)
    paused_total: float = 0.0                     # total wall time spent paused (excluded from elapsed)
    pause_started: float = 0.0                    # start of the current pause, 0 when running
    files: list[FileStatus] = field(default_factory=list)   # per-file rows (per-file projects only)

    @property
    def elapsed(self) -> float:
        return _live_elapsed(self.started_at, self.finished_at, self.paused_total, self.pause_started)


@dataclass
class BatchStatus:
    id: str
    status: Status = Status.QUEUED
    projects: list[ProjectStatus] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0
    cancelled: bool = False
    paused: bool = False                          # pause requested (queue halts at next boundary)
    gpu_freed: bool = False                       # the GPU has actually been released (worker parked)
    paused_total: float = 0.0                     # total wall time spent paused (excluded from elapsed)
    pause_started: float = 0.0                    # start of the current pause, 0 when running
    on_pause: Optional[Callable[[], None]] = field(default=None, repr=False)  # frees the GPU on park
    # Project indices requested for re-run while the worker is busy — the worker drains these
    # when it frees up (one GPU, so re-runs queue instead of starting a competing thread).
    rerun_pending: list[int] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _wake: threading.Event = field(default_factory=threading.Event, repr=False)  # unblocks a paused worker

    @property
    def progress(self) -> float:
        """Overall fraction, weighted by each project's audio length × tracks so a long project
        moves the bar more than a short one. Falls back to a plain average if weights are absent."""
        if not self.projects:
            return 0.0
        total = sum(p.weight for p in self.projects)
        if total > 0:
            return sum(p.progress * p.weight for p in self.projects) / total
        return sum(p.progress for p in self.projects) / len(self.projects)

    @property
    def elapsed(self) -> float:
        return _live_elapsed(self.started_at, self.finished_at, self.paused_total, self.pause_started)


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


def _selected_count(prep: PreparedSource) -> int:
    return sum(1 for t in prep.input.tracks if t.selected)


def _source_weight(prep: PreparedSource) -> float:
    """Rough work estimate for one file: audio length × its selected track count (each track is a
    separate transcription pass of about the file's duration)."""
    return prep.input.duration * max(_selected_count(prep), 1)


def _project_weight(project: ProjectInput) -> float:
    return sum(_source_weight(p) for p in project.prepared)


def _file_statuses(project: ProjectInput) -> list[FileStatus]:
    """A per-file project shows one row per source file; merge projects have no per-file rows."""
    if not project.per_file:
        return []
    return [FileStatus(name=p.input.filename, weight=_source_weight(p)) for p in project.prepared]


def _fmt_dur(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s}s" if s < 60 else f"{s // 60}m{s % 60:02d}s"


def _project_output_parent(project: ProjectInput, output_mode: str, output_folder: Optional[str]) -> Path:
    if output_mode == "folder":
        if not output_folder:
            raise ValueError("output_folder required when output_mode='folder'")
        return Path(output_folder)
    # "source": alongside the project's first source file
    return Path(project.prepared[0].path).parent


IsCancelled = Callable[[], bool]


def _open_timed(batch: "BatchStatus"):
    """The batch plus whatever project/file is mid-run — the timers that should freeze while paused.
    Nothing else is running (one GPU, one worker), so this is the full set of open clocks."""
    yield batch
    for p in batch.projects:
        if p.started_at and not p.finished_at:
            yield p
        for f in p.files:
            if f.started_at and not f.finished_at:
                yield f


def _gate(batch: "BatchStatus") -> bool:
    """Cooperative checkpoint, called wherever a cancel would be checked. While paused it frees the
    GPU, freezes the open clocks, and blocks here until resume or cancel. Returns True if the run
    should stop (a cancel is pending). Zero cost when not paused."""
    if batch.paused and not batch.cancelled:
        if batch.on_pause:
            batch.on_pause()                     # free the GPU (runs on the worker thread)
        batch.gpu_freed = True
        now = time.time()
        for obj in _open_timed(batch):
            obj.pause_started = now
        while batch.paused and not batch.cancelled:
            batch._wake.wait()
            batch._wake.clear()
        now = time.time()
        for obj in _open_timed(batch):           # bank the paused gap so elapsed excludes it
            if obj.pause_started:
                obj.paused_total += now - obj.pause_started
                obj.pause_started = 0.0
        batch.gpu_freed = False
    return batch.cancelled


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
            # One transcript per source file, each its own tracked queue row (ps.files). In
            # "source" mode each lands next to its own source file (nested tree preserved);
            # otherwise all go to the project's output folder. Prefetch overlaps the next file's
            # extraction with the current file's transcription. Only files still QUEUED are
            # processed, so a single-file restart (which re-queues one file) re-runs just it.
            files = ps.files
            seen: dict[tuple[str, str], int] = {}

            def on_wait(i: int, src: PreparedSource) -> None:
                # Extraction for this file is about to be waited on — start its clock here so
                # "done in Xs" counts the ffmpeg extraction, not just transcription.
                fs = files[i]
                fs.status = Status.RUNNING
                fs.started_at = fs.started_at or time.time()
                fs.finished_at = 0.0
                fs.error = None
                fs.tracks = _selected_count(src)
                fs.message = "extracting audio"
                _recalc_project(ps)

            def on_start(i: int, src: PreparedSource) -> None:
                fs = files[i]
                fs.status = Status.RUNNING
                fs.started_at = fs.started_at or time.time()
                fs.finished_at = 0.0
                fs.error = None
                fs.tracks = _selected_count(src)
                fs.message = "transcribing"
                # Extraction done, GPU work begins — record how long the extraction wait was
                # (reading the source + ffmpeg, during which the GPU was idle waiting for this file).
                # Subtract any pause so a mid-extraction pause doesn't inflate the wait.
                fs.extract_seconds = (max(0.0, time.time() - fs.started_at - fs.paused_total)
                                      if fs.started_at else 0.0)
                _recalc_project(ps)

            def on_track(i: int, j: int, total_tracks: int, speaker: str) -> None:
                fs = files[i]
                fs.track = j + 1
                fs.tracks = total_tracks
                fs.progress = j / max(total_tracks, 1)
                fs.message = f"track {j + 1}/{total_tracks} · {speaker}"
                _recalc_project(ps)

            def on_done(i: int, src: PreparedSource, merged, seen=seen) -> None:
                fs = files[i]
                dest = Path(src.path).parent if output_mode == "source" else out_dir
                stem = _safe_name(Path(src.input.filename).stem)
                seen[(str(dest), stem)] = seen.get((str(dest), stem), 0) + 1
                if seen[(str(dest), stem)] > 1:
                    stem = f"{stem}_{seen[(str(dest), stem)]}"
                export_all(merged, dest, transcript_basename(stem), formats=project.formats)
                fs.status = Status.DONE
                fs.progress = 1.0
                fs.finished_at = time.time()
                fs.output_dir = str(dest)
                fs.message = f"done in {fs.elapsed:.0f}s"
                # Per-file timing split (also shown per-row in the UI): the extraction wait (reading
                # the source + ffmpeg, GPU idle) vs the transcription itself, so a run reveals read-
                # vs GPU-bound files. Still printed to the console as a scrollback record.
                extract_s = fs.extract_seconds
                transcribe_s = max(0.0, fs.elapsed - extract_s)
                bound = "READ-bound" if extract_s > transcribe_s else "GPU-bound"
                print(f"[timing] {fs.name}: waited {_fmt_dur(extract_s)} for audio "
                      f"(read + extract, GPU idle) · transcribed {_fmt_dur(transcribe_s)} on GPU · {bound}",
                      flush=True)
                _recalc_project(ps)

            def on_error(i: int, src: PreparedSource, exc: Exception) -> None:
                fs = files[i]
                fs.status = Status.ERROR
                fs.error = str(exc)
                fs.finished_at = time.time()
                fs.message = "error"
                _recalc_project(ps)

            def on_skip(i: int, src: PreparedSource) -> None:
                fs = files[i]
                fs.status = Status.CANCELED
                fs.finished_at = time.time()
                fs.message = "canceled"
                _recalc_project(ps)

            pending = [k for k, fs in enumerate(files) if fs.status is Status.QUEUED]
            run_per_file_pipeline(
                project.prepared, transcribe_fn, pending, options=options, is_cancelled=is_cancelled,
                on_start=on_start, on_track=on_track, on_done=on_done, on_error=on_error,
                on_skip=on_skip, is_file_cancelled=lambda i: files[i].cancel_requested,
                on_wait=on_wait)
            if is_cancelled():
                raise Cancelled()
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
        ps.message = _file_summary(ps) if ps.files else f"done in {ps.elapsed:.0f}s"
        ps.status = Status.DONE
    except Cancelled:
        ps.finished_at = time.time()
        ps.message = "canceled"
        ps.status = Status.CANCELED
        for fs in ps.files:                          # mark any unfinished files canceled too
            if fs.status in (Status.QUEUED, Status.RUNNING):
                fs.status = Status.CANCELED
                fs.finished_at = time.time()
                fs.message = "canceled"
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
                  options=options, is_cancelled=lambda: _gate(batch))


def _set(ps: ProjectStatus, frac: float, msg: str) -> None:
    ps.progress = max(0.0, min(1.0, frac))
    ps.message = msg


def _file_summary(ps: ProjectStatus) -> str:
    """One-line tally of a per-file project's files, e.g. '28 done · 2 failed'."""
    c: dict[Status, int] = {}
    for f in ps.files:
        c[f.status] = c.get(f.status, 0) + 1
    order = [(Status.DONE, "done"), (Status.RUNNING, "running"), (Status.QUEUED, "queued"),
             (Status.ERROR, "failed"), (Status.CANCELED, "canceled")]
    parts = [f"{c[s]} {label}" for s, label in order if c.get(s)]
    return " · ".join(parts) or f"{len(ps.files)} files"


def _recalc_project(ps: ProjectStatus) -> None:
    """Roll a per-file project's file statuses up into its overall progress (length-weighted) and
    a short summary/current-file message."""
    if not ps.files:
        return
    total = sum(f.weight for f in ps.files)
    if total > 0:
        ps.progress = sum(f.progress * f.weight for f in ps.files) / total
    else:
        ps.progress = sum(f.progress for f in ps.files) / len(ps.files)
    running = next((f for f in ps.files if f.status is Status.RUNNING), None)
    ps.message = f"{running.name} · {running.message}" if running else _file_summary(ps)


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

    def create(self, projects: list[ProjectInput]) -> BatchStatus:
        batch = BatchStatus(
            id=uuid.uuid4().hex,
            projects=[ProjectStatus(name=p.name, weight=_project_weight(p), files=_file_statuses(p))
                      for p in projects])
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
        batch._wake.set()                         # wake a paused worker so it can unwind
        return True

    def pause(self, batch_id: str) -> bool:
        """Request a pause; the worker stops taking new work at the next file/project boundary,
        frees the GPU, and the clocks freeze until resume()."""
        batch = self.get(batch_id)
        if batch is None:
            return False
        batch.paused = True
        return True

    def resume(self, batch_id: str) -> bool:
        """Lift a pause; the parked worker wakes, reloads the model, and continues."""
        batch = self.get(batch_id)
        if batch is None:
            return False
        batch.paused = False
        batch._wake.set()
        return True

    def cancel_project(self, batch_id: str, index: int) -> bool:
        """Cancel ONE project — the queue continues to the next. Takes effect at that project's
        next stage boundary (or immediately if it hasn't started yet)."""
        batch = self.get(batch_id)
        if batch is None or not (0 <= index < len(batch.projects)):
            return False
        batch.projects[index].cancel_requested = True
        return True

    def cancel_file(self, batch_id: str, pindex: int, findex: int) -> bool:
        """Cancel ONE file inside a per-file project — the other files keep going. Stops at that
        file's next track boundary, or immediately if it hasn't started."""
        batch = self.get(batch_id)
        if batch is None or not (0 <= pindex < len(batch.projects)):
            return False
        ps = batch.projects[pindex]
        if not (0 <= findex < len(ps.files)):
            return False
        ps.files[findex].cancel_requested = True
        return True

    def restart_file(self, batch_id: str, pindex: int, findex: int) -> bool:
        """Re-run ONE finished file. The file goes back to QUEUED and its project is queued for a
        re-run that processes only the still-QUEUED files; if the batch is idle a worker starts.
        Only a finished file (done/failed/canceled) can be restarted — that avoids racing the
        in-flight pass and double-exporting."""
        batch = self.get(batch_id)
        spec = self._specs.get(batch_id)
        if batch is None or spec is None or not (0 <= pindex < len(batch.projects)):
            return False
        ps = batch.projects[pindex]
        if not (0 <= findex < len(ps.files)):
            return False
        if ps.files[findex].status not in (Status.DONE, Status.ERROR, Status.CANCELED):
            return False
        self._reset_fs(ps.files[findex])
        start = False
        with batch.lock:
            if pindex not in batch.rerun_pending:
                batch.rerun_pending.append(pindex)
            if batch.status is not Status.RUNNING:
                batch.status = Status.RUNNING
                batch.cancelled = False
                start = True
        if start:
            self.run(batch, lambda: None)            # worker drains rerun_pending
        return True

    @staticmethod
    def _reset_ps(ps: ProjectStatus) -> None:
        """Put a project back to a fresh QUEUED state so it can run (again). For per-file projects,
        also re-queue its failed/canceled files (done files stay done) so a project retry re-runs
        only what didn't succeed."""
        ps.status = Status.QUEUED
        ps.error = None
        ps.progress = 0.0
        ps.message = ""
        ps.started_at = 0.0
        ps.finished_at = 0.0
        ps.cancel_requested = False
        ps.paused_total = 0.0
        ps.pause_started = 0.0
        for fs in ps.files:
            if fs.status in (Status.ERROR, Status.CANCELED):
                BatchRegistry._reset_fs(fs)

    @staticmethod
    def _reset_fs(fs: FileStatus) -> None:
        fs.status = Status.QUEUED
        fs.error = None
        fs.progress = 0.0
        fs.message = ""
        fs.track = 0
        fs.tracks = 0
        fs.extract_seconds = 0.0
        fs.started_at = 0.0
        fs.finished_at = 0.0
        fs.cancel_requested = False
        fs.paused_total = 0.0
        fs.pause_started = 0.0

    def _run_indices(self, batch: BatchStatus, spec: _RunSpec, indices: list[int]) -> None:
        pairs = [(spec.projects[i], batch.projects[i]) for i in indices]
        _run_projects(pairs, spec.transcribe_fn, output_mode=spec.output_mode,
                      output_folder=spec.output_folder, options=spec.options,
                      is_cancelled=lambda: _gate(batch))

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
