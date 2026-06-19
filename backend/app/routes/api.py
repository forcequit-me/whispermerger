"""HTTP API (PRD §11.1.5): upload+probe, start transcription job, poll progress,
get merged result, export. Routes stay thin — they wire request data into the engine
modules and the job registry held on app.state.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from ..batch import ProjectInput, Status, _RunSpec, run_batch
from ..diarize import detect_speech, transcribe_regions
from ..export import ExportOptions, pad_segments, to_csv, to_json, to_srt, to_txt
from ..extract import compute_peaks, concat_wavs, extract_clip, extract_sample, extract_track
from ..merge import BackchannelRender, MergeOptions
from ..models import InputSource, MergedOutput, Track
from ..pipeline import PreparedSource, run_merge_pipeline
from ..probe import probe
from ..transcribe import transcribe_track

router = APIRouter()


# --- server lifecycle ------------------------------------------------------------------

@router.post("/shutdown")
def shutdown() -> dict:
    """Stop the local server from the web UI / Shutdown.bat. Hard exit on a background thread
    after the response flushes — safe here because temp working files are wiped on next startup."""
    def _die() -> None:
        time.sleep(0.3)
        os._exit(0)
    threading.Thread(target=_die, daemon=True).start()
    return {"stopping": True}


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
    language: str | None = None              # force a language code; None = auto-detect


class JobCreated(BaseModel):
    job_id: str


class ProgressResponse(BaseModel):
    status: str
    progress: float
    message: str
    elapsed: float = 0.0
    error: str | None = None


class ExportOptionsIn(BaseModel):
    apply_limits: bool = True
    chars_per_line: int = 42
    max_lines: int = 2
    min_duration: float = 1.0
    show_speakers: bool = True
    pad_seconds: float = 0.0

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
    per_file: bool = False                  # one transcript per file (vs merge/sequence)
    formats: list[str] | None = None        # which exports to write; None = all four


class BatchRequest(BaseModel):
    projects: list[BatchProjectIn]
    output_mode: Literal["folder", "source"] = "folder"
    output_folder: str | None = None
    options: MergeOptionsIn = MergeOptionsIn()
    language: str | None = None              # force a language code; None = auto-detect


class BatchCreated(BaseModel):
    batch_id: str


class RetryRequest(BaseModel):
    projects: list[int] | None = None        # project indices to retry; None = all failed/canceled


class FileStatusOut(BaseModel):
    name: str
    status: str
    progress: float
    message: str
    track: int = 0                           # 1-based index of the track currently transcribing
    tracks: int = 0                          # total selected tracks in this file
    elapsed: float = 0.0
    output_dir: str | None = None
    error: str | None = None


class ProjectStatusOut(BaseModel):
    name: str
    status: str
    progress: float
    message: str
    elapsed: float = 0.0
    output_dir: str | None = None
    error: str | None = None
    files: list[FileStatusOut] = []          # per-file rows (per-file projects only)


class BatchStatusOut(BaseModel):
    id: str
    status: str
    progress: float
    elapsed: float = 0.0
    projects: list[ProjectStatusOut]


class RegionOut(BaseModel):
    start: float
    end: float


class DiarizeRequest(BaseModel):
    file_id: str
    track_index: int = 0
    speakers: dict[str, list[list[float]]]   # speaker name -> [[start, end], ...]
    options: MergeOptionsIn = MergeOptionsIn()


class CombineFile(BaseModel):
    file_id: str
    track_index: int = 0


class CombineRequest(BaseModel):
    files: list[CombineFile]                 # in sequence order


class CombineResult(BaseModel):
    file_id: str
    duration: float
    boundaries: list[float]                  # start time of each file in the combined timeline


# --- input: local path (FR-IN, large files) + upload (small files) ---------------------

class AddPathRequest(BaseModel):
    paths: list[str]


# Audio/video extensions the batch "Folder" picker ingests (recursively).
_MEDIA_EXTS = {
    ".mkv", ".mov", ".mp4", ".m4v", ".avi", ".webm", ".flv", ".ts", ".m2ts", ".mpg", ".mpeg",
    ".wmv", ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wma", ".aiff", ".aif",
}


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


@router.post("/add-folder", response_model=list[UploadItem])
def add_folder(request: Request) -> list[UploadItem]:
    """Native folder picker → register every audio/video file under it (recursively), in
    place. The 'Folder' counterpart to /browse for a batch project's add button. Display
    name is the path relative to the chosen folder so nested files stay distinguishable;
    unreadable / audio-less files are skipped rather than failing the whole scan."""
    folder = _native_pick_dir()
    if not folder:
        return []
    root = Path(folder)
    if not root.is_dir():
        raise HTTPException(404, f"not a folder: {folder}")
    items: list[UploadItem] = []
    for p in sorted(root.rglob("*")):
        if not (p.is_file() and p.suffix.lower() in _MEDIA_EXTS):
            continue
        try:
            item = _register(request.app.state, p, str(p.relative_to(root)))
        except Exception:
            continue
        if not item.source.tracks:
            continue
        items.append(item)
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


@router.get("/clip/{file_id}")
def clip(request: Request, file_id: str, start: float = 0.0, dur: float = 5.0,
         track: int | None = None) -> FileResponse:
    """Preview clip starting at `start`. Downmix of all tracks by default; pass `track` to
    isolate one speaker's stream (per-line preview in the multi-track editor). Powers
    jump-to-line and jump-to-position — input-seek makes it instant even on huge files."""
    entry = request.app.state.uploads.get(file_id)
    if entry is None:
        raise HTTPException(404, f"unknown file_id {file_id}")
    dur = max(0.5, min(dur, 120.0))  # cap clip length
    clips = request.app.state.upload_dir / "clips"
    clips.mkdir(exist_ok=True)
    dest = clips / f"{file_id}_{start:.2f}_{dur:.2f}_{track}.wav"
    if not dest.exists():
        extract_clip(entry["path"], dest, start, dur, len(entry["source"].tracks), track=track)
    return FileResponse(dest, media_type="audio/wav")


@router.get("/peaks/{file_id}")
def peaks(request: Request, file_id: str, buckets: int = 2000) -> dict:
    """Downsampled waveform peaks for the editor (downmix of all streams, matching /clip).
    Browser draws these instead of downloading audio — scales to huge files. Cached per file."""
    entry = request.app.state.uploads.get(file_id)
    if entry is None:
        raise HTTPException(404, f"unknown file_id {file_id}")
    cached = entry.get("peaks")
    if cached and cached["buckets"] == buckets:
        return {"duration": entry["source"].duration, "peaks": cached["peaks"]}
    out_dir = request.app.state.upload_dir / "peaks"
    out_dir.mkdir(exist_ok=True)
    dest = out_dir / f"{file_id}.wav"
    if not dest.exists():
        extract_clip(entry["path"], dest, 0.0, entry["source"].duration,
                     len(entry["source"].tracks))
    vals = compute_peaks(dest, buckets)
    entry["peaks"] = {"buckets": buckets, "peaks": vals}
    return {"duration": entry["source"].duration, "peaks": vals}


# --- v2 single-track diarization (PRD §11.4) -------------------------------------------

def _track_wav(app_state, file_id: str, track_index: int) -> Path:
    """Extract one track to a cached mono/16k wav (the whole track — VAD/diarize need it)."""
    entry = app_state.uploads.get(file_id)
    if entry is None:
        raise HTTPException(404, f"unknown file_id {file_id}")
    out_dir = app_state.upload_dir / "diar"
    out_dir.mkdir(exist_ok=True)
    dest = out_dir / f"{file_id}_t{track_index}.wav"
    if not dest.exists():
        extract_track(entry["path"], track_index, dest)
    return dest


@router.get("/track-audio/{file_id}/{track_index}")
def track_audio(request: Request, file_id: str, track_index: int) -> FileResponse:
    """Serve the full track as mono/16k wav (range-supported) for the WaveSurfer waveform."""
    wav = _track_wav(request.app.state, file_id, track_index)
    return FileResponse(wav, media_type="audio/wav")


@router.post("/vad/{file_id}/{track_index}", response_model=list[RegionOut])
def vad(request: Request, file_id: str, track_index: int) -> list[RegionOut]:
    """Detect speech regions on a single track — seeds the manual-labeling UI."""
    wav = _track_wav(request.app.state, file_id, track_index)
    return [RegionOut(start=s, end=e) for s, e in detect_speech(wav)]


@router.post("/combine", response_model=CombineResult)
def combine(request: Request, body: CombineRequest) -> CombineResult:
    """Concatenate several files' tracks into one combined single-track recording (sequence
    order) and register it as a normal source. The Label Track UI then labels/transcribes it
    like any single file — the combined timeline is already the sequenced one."""
    if not body.files:
        raise HTTPException(400, "no files")
    state = request.app.state
    combo_key = "|".join(f"{f.file_id}:{f.track_index}" for f in body.files)
    cached_id = state.combined.get(combo_key)
    if cached_id and cached_id in state.uploads:
        e = state.uploads[cached_id]
        return CombineResult(file_id=cached_id, duration=e["source"].duration,
                             boundaries=e["boundaries"])

    wavs = [_track_wav(state, f.file_id, f.track_index) for f in body.files]
    out_dir = state.upload_dir / "combined"
    out_dir.mkdir(exist_ok=True)
    dest = out_dir / f"{hashlib.sha256(combo_key.encode()).hexdigest()[:16]}.wav"
    total, boundaries = concat_wavs(wavs, dest)

    file_id = uuid.uuid4().hex
    source = InputSource(filename=f"Combined ({len(body.files)} files)", duration=total,
                         tracks=[Track(track_index=0, speaker="Speaker 1")])
    state.uploads[file_id] = {"path": str(dest), "source": source, "boundaries": boundaries}
    state.combined[combo_key] = file_id
    return CombineResult(file_id=file_id, duration=total, boundaries=boundaries)


@router.post("/diarize", response_model=JobCreated)
def diarize(request: Request, body: DiarizeRequest) -> JobCreated:
    """Transcribe each speaker on only their labeled regions, remapped + merged. Job-based."""
    wav = _track_wav(request.app.state, body.file_id, body.track_index)
    cache = request.app.state.cache
    speaker_regions = {name: [(r[0], r[1]) for r in regions]
                       for name, regions in body.speakers.items()}
    options = body.options.to_options()
    job = request.app.state.jobs.create()

    def work(progress) -> MergedOutput:
        progress(0.1, "transcribing labeled regions")
        return transcribe_regions(
            str(wav), speaker_regions,
            lambda w: transcribe_track(w, cache=cache),
            options=options,
        )

    request.app.state.jobs.run(job, work)
    return JobCreated(job_id=job.id)


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
            lambda wav: transcribe_track(wav, cache=cache, language=body.language),
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
                            message=job.message, elapsed=job.elapsed, error=job.error)


@router.get("/result/{job_id}", response_model=MergedOutput)
def result(request: Request, job_id: str) -> MergedOutput:
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    if job.result is None:
        raise HTTPException(409, f"job not done (status={job.status.value})")
    return job.result


@router.post("/cancel/{job_id}")
def cancel_job(request: Request, job_id: str) -> dict:
    """Request job cancellation — takes effect at the next stage boundary (the in-flight
    extract/transcribe/merge stage finishes first)."""
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    job.cancel()
    return {"canceling": True}


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
    merged = pad_segments(merged, opts.pad_seconds)  # all formats carry padded timecodes
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

    projects = []
    for p in body.projects:
        if p.formats is not None and not [f for f in p.formats if f in _FORMATS]:
            raise HTTPException(400, "choose at least one format")
        projects.append(ProjectInput(
            name=p.name,
            prepared=_prepare_sources(request.app.state, p.sources),
            per_file=p.per_file,
            formats=p.formats,
        ))
    options = body.options.to_options()
    batch = request.app.state.batches.create(projects)

    def transcribe_fn(wav: str) -> dict:
        return transcribe_track(wav, cache=cache, language=body.language)

    request.app.state.batches.set_spec(batch.id, _RunSpec(
        projects=projects, transcribe_fn=transcribe_fn,
        output_mode=body.output_mode, output_folder=body.output_folder, options=options))
    request.app.state.batches.run(batch, lambda: run_batch(
        batch, projects, transcribe_fn,
        output_mode=body.output_mode, output_folder=body.output_folder, options=options))
    return BatchCreated(batch_id=batch.id)


def _batch_out(batch) -> BatchStatusOut:
    return BatchStatusOut(
        id=batch.id, status=batch.status.value, progress=batch.progress, elapsed=batch.elapsed,
        projects=[ProjectStatusOut(name=p.name, status=p.status.value, progress=p.progress,
                                   message=p.message, elapsed=p.elapsed,
                                   output_dir=p.output_dir, error=p.error,
                                   files=[FileStatusOut(name=f.name, status=f.status.value,
                                                        progress=f.progress, message=f.message,
                                                        track=f.track, tracks=f.tracks,
                                                        elapsed=f.elapsed, output_dir=f.output_dir,
                                                        error=f.error)
                                          for f in p.files])
                  for p in batch.projects],
    )


@router.get("/batch/{batch_id}", response_model=BatchStatusOut)
def batch_status(request: Request, batch_id: str) -> BatchStatusOut:
    batch = request.app.state.batches.get(batch_id)
    if batch is None:
        raise HTTPException(404, "unknown batch")
    return _batch_out(batch)


@router.post("/batch/{batch_id}/cancel")
def cancel_batch(request: Request, batch_id: str) -> dict:
    """Request batch cancellation — stops at the next file/project boundary (the file
    currently transcribing finishes); remaining projects are marked canceled."""
    if not request.app.state.batches.cancel(batch_id):
        raise HTTPException(404, "unknown batch")
    return {"canceling": True}


@router.post("/batch/{batch_id}/project/{idx}/cancel")
def cancel_batch_project(request: Request, batch_id: str, idx: int) -> dict:
    """Cancel one project in a batch — the file currently transcribing finishes, that project is
    marked canceled, and the queue continues to the next project (unlike whole-batch cancel)."""
    if not request.app.state.batches.cancel_project(batch_id, idx):
        raise HTTPException(404, "unknown batch or project")
    return {"canceling": True}


@router.post("/batch/{batch_id}/project/{pidx}/file/{fidx}/cancel")
def cancel_batch_file(request: Request, batch_id: str, pidx: int, fidx: int) -> dict:
    """Cancel one file inside a per-file project — the other files keep going."""
    if not request.app.state.batches.cancel_file(batch_id, pidx, fidx):
        raise HTTPException(404, "unknown batch / project / file")
    return {"canceling": True}


@router.post("/batch/{batch_id}/project/{pidx}/file/{fidx}/restart")
def restart_batch_file(request: Request, batch_id: str, pidx: int, fidx: int) -> dict:
    """Re-run one finished file. Queues it to run when the GPU frees up (or starts a worker if
    the batch is idle); only files that already finished can be restarted."""
    if not request.app.state.batches.restart_file(batch_id, pidx, fidx):
        raise HTTPException(409, "cannot restart that file")
    return {"restarting": True}


@router.post("/batch/{batch_id}/retry", response_model=BatchStatusOut)
def retry_batch(request: Request, batch_id: str,
                body: RetryRequest | None = None) -> BatchStatusOut:
    """Re-run projects in a batch (default: all errored/canceled), reusing the same batch +
    its persisted run spec. Caching makes already-transcribed files instant."""
    batch = request.app.state.batches.get(batch_id)
    if batch is None:
        raise HTTPException(404, "unknown batch")
    if body and body.projects:
        indices = [i for i in body.projects if 0 <= i < len(batch.projects)]
    else:
        indices = [i for i, p in enumerate(batch.projects)
                   if p.status in (Status.ERROR, Status.CANCELED)]
    if not indices:
        raise HTTPException(400, "no failed or canceled projects to retry")
    if not request.app.state.batches.retry(batch_id, indices):
        raise HTTPException(409, "batch cannot be retried")
    return _batch_out(batch)
