"""HTTP API (PRD §11.1.5): upload+probe, start transcription job, poll progress,
get merged result, export. Routes stay thin — they wire request data into the engine
modules and the job registry held on app.state.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

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


# --- upload + probe (FR-IN-1/2/3/4) ----------------------------------------------------

@router.post("/upload", response_model=list[UploadItem])
async def upload(request: Request, files: list[UploadFile] = File(...)) -> list[UploadItem]:
    items: list[UploadItem] = []
    for f in files:
        file_id = uuid.uuid4().hex
        suffix = Path(f.filename or "").suffix
        dest = request.app.state.upload_dir / f"{file_id}{suffix}"
        dest.write_bytes(await f.read())

        source = probe(dest)
        source.filename = f.filename or dest.name  # keep the user-facing name
        request.app.state.uploads[file_id] = {"path": str(dest), "source": source}
        items.append(UploadItem(file_id=file_id, source=source))
    return items


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

@router.post("/transcribe", response_model=JobCreated)
def transcribe(request: Request, body: TranscribeRequest) -> JobCreated:
    uploads = request.app.state.uploads
    cache = request.app.state.cache

    prepared: list[PreparedSource] = []
    for sel in body.sources:
        entry = uploads.get(sel.file_id)
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
