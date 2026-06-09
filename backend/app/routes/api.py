"""HTTP API (PRD §11.1.5): upload+probe, start transcription job, poll progress,
get merged result, export. Routes stay thin — they wire request data into the engine
modules and the job registry held on app.state.
"""

from __future__ import annotations

import subprocess
import sys
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from ..batch import ProjectInput, run_batch
from ..export import ExportOptions, to_csv, to_json, to_srt, to_txt
from ..extract import extract_sample
from ..merge import BackchannelRender, MergeOptions
from ..models import InputSource, MergedOutput, Track
from ..pipeline import PreparedSource, run_merge_pipeline
from ..probe import probe
from ..transcribe import transcribe_track

router = APIRouter()


# --- request/response schemas ----------------------------------------------------------

class UploadItem(BaseModel):
    file_id: str
    source: InputSource


class TrackSelection(BaseModel):
    track_index: int
    speaker: str
    selected: bool = True


class SourceSelection(BaseModel):
    file_id: str
    sequence: int = 0
    tracks: list[TrackSelection]


class MergeOptionsIn(BaseModel):
    detect_conflicts: bool = True
    backchannel_max_words: int = 4
    backchannel_render: BackchannelRender = BackchannelRender.PARENTHETICAL
    overlap_min_seconds: float = 2.0
    bleed_window_seconds: float = 1.0
    bleed_similarity: float = 0.7

    def to_options(self) -> MergeOptions:
        return MergeOptions(**self.model_dump())


class TranscribeRequest(BaseModel):
    sources: list[SourceSelection]
    options: MergeOptionsIn = MergeOptionsIn()


class JobCreated(BaseModel):
    job_id: str


class ProgressResponse(BaseModel):
    status: str
    progress: float
    message: str
    error: str | None = None


class ExportOptionsIn(BaseModel):
    apply_limits: bool = True
    chars_per_line: int = 42
    max_lines: int = 2
    min_duration: float = 1.0
    show_speakers: bool = True

    def to_options(self) -> ExportOptions:
        return ExportOptions(**self.model_dump())


class ExportRequest(BaseModel):
    """Used by the Edit Transcript tab to export an edited transcript directly."""
    merged: MergedOutput
    options: ExportOptionsIn = ExportOptionsIn()
    filename: str = "transcript"


class BatchProjectIn(BaseModel):
    name: str
    sources: list[SourceSelection]


class BatchRequest(BaseModel):
    projects: list[BatchProjectIn]
    output_mode: Literal["folder", "source"] = "folder"
    output_folder: str | None = None
    options: MergeOptionsIn = MergeOptionsIn()


class BatchCreated(BaseModel):
    batch_id: str


class ProjectStatusOut(BaseModel):
    name: str
    status: str
    progress: float
    message: str
    output_dir: str | None = None
    error: str | None = None


class BatchStatusOut(BaseModel):
    id: str
    status: str
    progress: float
    projects: list[ProjectStatusOut]


# --- input: local path (FR-IN, large files) + upload (small files) ---------------------

class AddPathRequest(BaseModel):
    paths: list[str]


def _register(app_state, file_path: Path, display_name: str) -> UploadItem:
    """Probe a file already on disk and register it in place (no copy)."""
    file_id = uuid.uuid4().hex
    source = probe(file_path)
    source.filename = display_name
    app_state.uploads[file_id] = {"path": str(file_path), "source": source}
    return UploadItem(file_id=file_id, source=source)


@router.post("/add-path", response_model=list[UploadItem])
def add_path(request: Request, body: AddPathRequest) -> list[UploadItem]:
    """Register local files by path (programmatic / scriptable). The UI uses /browse."""
    items: list[UploadItem] = []
    for raw in body.paths:
        p = Path(raw.strip().strip('"'))
        if not p.is_file():
            raise HTTPException(404, f"not a file: {raw}")
        items.append(_register(request.app.state, p, p.name))
    return items


# Native OS file picker, run in a subprocess so tkinter stays on its own main thread
# (FastAPI handlers run in a worker thread, where Tk is unsafe).
_PICKER = (
    "import tkinter as tk; from tkinter import filedialog\n"
    "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
    "paths = filedialog.askopenfilenames(title='Select recordings')\n"
    "r.destroy()\n"
    "import sys; sys.stdout.write('\\n'.join(paths))\n"
)


def _native_pick() -> list[str]:
    out = subprocess.run([sys.executable, "-c", _PICKER],
                         capture_output=True, text=True, timeout=600)
    return [p for p in out.stdout.splitlines() if p.strip()]


@router.post("/browse", response_model=list[UploadItem])
def browse(request: Request) -> list[UploadItem]:
    """Open a native file picker on the server (same machine) and register the chosen files
    in place. No upload, no copy — ffmpeg reads the originals and pulls only the audio streams.
    The only way to get real local paths, since browsers hide them from the web file picker."""
    items: list[UploadItem] = []
    for raw in _native_pick():
        p = Path(raw)
        if p.is_file():
            items.append(_register(request.app.state, p, p.name))
    return items


_DIR_PICKER = (
    "import tkinter as tk; from tkinter import filedialog\n"
    "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
    "d = filedialog.askdirectory(title='Select output folder')\n"
    "r.destroy()\n"
    "import sys; sys.stdout.write(d or '')\n"
)


def _native_pick_dir() -> str:
    out = subprocess.run([sys.executable, "-c", _DIR_PICKER],
                         capture_output=True, text=True, timeout=600)
    return out.stdout.strip()


@router.post("/browse-folder")
def browse_folder() -> dict:
    """Native folder picker — used to choose the batch output directory."""
    return {"path": _native_pick_dir()}


# --- audition (FR-TK-3) ----------------------------------------------------------------

@router.get("/audition/{file_id}/{track_index}")
def audition(request: Request, file_id: str, track_index: int) -> FileResponse:
    """Serve a short sample of one track so the user can confirm its label."""
    entry = request.app.state.uploads.get(file_id)
    if entry is None:
        raise HTTPException(404, f"unknown file_id {file_id}")
    samples = request.app.state.upload_dir / "samples"
    samples.mkdir(exist_ok=True)
    dest = samples / f"{file_id}_t{track_index}.wav"
    if not dest.exists():
        extract_sample(entry["path"], track_index, dest)
    return FileResponse(dest, media_type="audio/wav")


# --- transcribe + merge (FR-TR-*, FR-MD-1) ---------------------------------------------

def _prepare_sources(app_state, sources: list[SourceSelection]) -> list[PreparedSource]:
    """Resolve a list of file_id selections into PreparedSources for the pipeline."""
    prepared: list[PreparedSource] = []
    for sel in sources:
        entry = app_state.uploads.get(sel.file_id)
        if entry is None:
            raise HTTPException(404, f"unknown file_id {sel.file_id}")
        base: InputSource = entry["source"]
        source = InputSource(
            filename=base.filename,
            duration=base.duration,
            sequence=sel.sequence,
            tracks=[Track(track_index=t.track_index, speaker=t.speaker, selected=t.selected)
                    for t in sel.tracks],
        )
        prepared.append(PreparedSource(path=entry["path"], input=source))
    return prepared


@router.post("/transcribe", response_model=JobCreated)
def transcribe(request: Request, body: TranscribeRequest) -> JobCreated:
    cache = request.app.state.cache
    prepared = _prepare_sources(request.app.state, body.sources)
    options = body.options.to_options()
    job = request.app.state.jobs.create()

    def work(progress) -> MergedOutput:
        return run_merge_pipeline(
            prepared,
            lambda wav: transcribe_track(wav, cache=cache),
            options=options,
            progress_cb=progress,
        )

    request.app.state.jobs.run(job, work)
    return JobCreated(job_id=job.id)


# --- progress + result (FR-TR-4) -------------------------------------------------------

@router.get("/progress/{job_id}", response_model=ProgressResponse)
def progress(request: Request, job_id: str) -> ProgressResponse:
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return ProgressResponse(status=job.status.value, progress=job.progress,
                            message=job.message, error=job.error)


@router.get("/result/{job_id}", response_model=MergedOutput)
def result(request: Request, job_id: str) -> MergedOutput:
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    if job.result is None:
        raise HTTPException(409, f"job not done (status={job.status.value})")
    return job.result


# --- export (FR-EX-*) ------------------------------------------------------------------

_FORMATS = {
    "srt": (to_srt, "application/x-subrip", "srt"),
    "txt": (to_txt, "text/plain", "txt"),
    "csv": (to_csv, "text/csv", "csv"),
    "json": (to_json, "application/json", "json"),
}


def _render(merged: MergedOutput, fmt: str, opts: ExportOptions) -> tuple[Response, str]:
    if fmt not in _FORMATS:
        raise HTTPException(400, f"unsupported format {fmt}")
    fn, media_type, ext = _FORMATS[fmt]
    body = fn(merged, opts) if fmt in ("srt", "txt") else fn(merged)
    return Response(content=body, media_type=media_type), ext


@router.post("/export/{job_id}")
def export_job(request: Request, job_id: str, format: str = "srt",
               filename: str = "transcript", options: ExportOptionsIn | None = None) -> Response:
    job = request.app.state.jobs.get(job_id)
    if job is None or job.result is None:
        raise HTTPException(404, "no result for job")
    resp, ext = _render(job.result, format, (options or ExportOptionsIn()).to_options())
    resp.headers["Content-Disposition"] = f'attachment; filename="{filename}.{ext}"'
    return resp


@router.post("/export")
def export_inline(body: ExportRequest, format: str = "srt") -> Response:
    """Export an edited transcript posted from the Edit Transcript tab."""
    resp, ext = _render(body.merged, format, body.options.to_options())
    resp.headers["Content-Disposition"] = f'attachment; filename="{body.filename}.{ext}"'
    return resp


# --- batch / studio (Method 1 + 2) -----------------------------------------------------

@router.post("/batch", response_model=BatchCreated)
def start_batch(request: Request, body: BatchRequest) -> BatchCreated:
    if body.output_mode == "folder" and not body.output_folder:
        raise HTTPException(400, "output_folder required when output_mode='folder'")
    cache = request.app.state.cache

    projects = [
        ProjectInput(name=p.name, prepared=_prepare_sources(request.app.state, p.sources))
        for p in body.projects
    ]
    options = body.options.to_options()
    batch = request.app.state.batches.create([p.name for p in body.projects])

    def work() -> None:
        run_batch(
            batch, projects,
            lambda wav: transcribe_track(wav, cache=cache),
            output_mode=body.output_mode,
            output_folder=body.output_folder,
            options=options,
        )

    request.app.state.batches.run(batch, work)
    return BatchCreated(batch_id=batch.id)


@router.get("/batch/{batch_id}", response_model=BatchStatusOut)
def batch_status(request: Request, batch_id: str) -> BatchStatusOut:
    batch = request.app.state.batches.get(batch_id)
    if batch is None:
        raise HTTPException(404, "unknown batch")
    return BatchStatusOut(
        id=batch.id, status=batch.status.value, progress=batch.progress,
        projects=[ProjectStatusOut(name=p.name, status=p.status.value, progress=p.progress,
                                   message=p.message, output_dir=p.output_dir, error=p.error)
                  for p in batch.projects],
    )
