"""WhisperMerger actions for scripts and AI: the one list the CLI (cli.py) and the MCP server
(mcp_server.py) are both built from, so the two can't drift apart (#36).

Each action is a plain function over the local server's HTTP API (backend/app/routes/api.py). Its
signature and docstring ARE the interface: cli.py turns them into commands and flags, mcp_server.py
into tools. The server is started on first use if it isn't running, and keeps running afterwards
so the speech model stays loaded between calls. Files are read in place; nothing is uploaded.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Optional

BACKEND = Path(__file__).resolve().parent
PORT = int(os.environ.get("WHISPERMERGER_PORT", "8756"))
BASE = f"http://127.0.0.1:{PORT}"
FORMATS = ("srt", "txt", "csv", "json")

# Where long-running actions report progress (the CLI prints it to stderr; MCP leaves it off,
# since stdout carries the protocol).
progress_out: Optional[Callable[[str], None]] = None


class ActionError(Exception):
    """A request the server refused, with its reason."""


# --- talking to the server ---------------------------------------------------------------

def _request(method: str, path: str, body=None, timeout: float = 120, raw: bool = False):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = r.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except ValueError:
            pass
        raise ActionError(f"{e.code}: {detail}") from None
    return payload if raw else json.loads(payload or b"null")


def _up() -> bool:
    try:
        _request("GET", "/api/cache/stats", timeout=2)
        return True
    except (OSError, ActionError):
        return False


def ensure_server() -> None:
    """Start the WhisperMerger server in the background if it isn't running (log: backend/tmp/server.log)."""
    if _up():
        return
    _say(f"starting the WhisperMerger server on port {PORT}...")
    log = BACKEND / "tmp" / "server.log"
    log.parent.mkdir(exist_ok=True)
    env = {**os.environ, "HF_HUB_DISABLE_XET": "1"}       # same as Startup.bat
    flags = 0
    if os.name == "nt":                                    # outlive this process, no console window
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT)],
                     cwd=str(BACKEND), env=env, stdin=subprocess.DEVNULL,
                     stdout=open(log, "ab"), stderr=subprocess.STDOUT, creationflags=flags)
    for _ in range(120):
        if _up():
            return
        time.sleep(0.5)
    raise ActionError(f"the server didn't start within 60 s; see {log}")


def _say(msg: str) -> None:
    if progress_out:
        progress_out(msg)


def _api(method: str, path: str, body=None, **kw):
    ensure_server()
    return _request(method, path, body, **kw)


def _register(paths: list[str]) -> list[dict]:
    """Register files (and every media file under folders) with the server, in place."""
    items = []
    for p in paths:
        p = str(Path(p).expanduser().resolve())
        if Path(p).is_dir():
            items += _api("POST", "/api/add-folder", {"path": p}, timeout=600)
        else:
            items += _api("POST", "/api/add-path", {"paths": [p]}, timeout=120)
    if not items:
        raise ActionError("no audio or video files found in those paths")
    return items


def _selection(item: dict, names: Optional[list[str]], tracks: Optional[list[int]], seq: int) -> dict:
    sel = []
    for k, t in enumerate(item["source"]["tracks"]):
        sel.append({"track_index": t["track_index"],
                    "speaker": names[k] if names and k < len(names) and names[k] else t["speaker"],
                    "selected": tracks is None or (t["track_index"] + 1) in tracks})
    return {"file_id": item["file_id"], "sequence": seq, "tracks": sel}


def _asked(run_id: str, question: dict) -> dict:
    """What a run that stopped to ask returns: the question, its choices and how to answer."""
    return {"status": "waiting for your answer", "question": question,
            "next": f"answer it (answer {run_id} download | skip | language, the last with --language en); "
                    "the run then carries on, follow it with get_progress or get_batch"}


def _wait_job(job_id: str) -> dict:
    """Follow a job until it's done, or until it stops to ask something (then p["question"] is set)."""
    last = None
    while True:
        p = _request("GET", f"/api/progress/{job_id}")
        msg = f"{round(p['progress'] * 100)}% · {p['message']}"
        if msg != last and p["message"]:
            _say(msg)
            last = msg
        if p["status"] == "done" or p.get("question"):
            return p
        if p["status"] in ("error", "cancelled"):
            raise ActionError(f"job {p['status']}: {p.get('error') or p['message']}")
        time.sleep(1)


def _merged(source: str) -> tuple[dict, str]:
    """A transcript from a job id or a WhisperMerger .json file, plus a default export name."""
    path = Path(source).expanduser()
    if path.suffix.lower() == ".json" and path.is_file():
        return json.loads(path.read_text("utf-8")), path.stem
    merged = _api("GET", f"/api/result/{source}")
    name = Path(merged["sources"][0]["filename"]).stem if merged.get("sources") else "transcript"
    return merged, f"{name} Transcript"


def _write(merged: dict, out: str, filename: str, formats: list[str], options: dict) -> list[str]:
    bad = [f for f in formats if f not in FORMATS]
    if bad:
        raise ActionError(f"unknown format {', '.join(bad)} (use {', '.join(FORMATS)})")
    folder = Path(out).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        data = _api("POST", f"/api/export?format={fmt}",
                    {"merged": merged, "options": options, "filename": filename}, raw=True)
        dest = folder / f"{filename}.{fmt}"
        dest.write_bytes(data)
        written.append(str(dest))
    return written


def _export_options(max_chars: int, max_lines: int, pad: float, subtitle_limits: bool,
                    show_speakers: bool, name_on_change: bool, txt_timestamps: bool) -> dict:
    return {"apply_limits": subtitle_limits, "chars_per_line": max_chars, "max_lines": max_lines,
            "pad_seconds": pad, "show_speakers": show_speakers, "label_on_change": name_on_change,
            "txt_timestamps": txt_timestamps}


def _fmt_time(s: float) -> str:
    return f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{s % 60:05.2f}"


# --- the actions ---------------------------------------------------------------------------

def add_files(paths: list[str]) -> list[dict]:
    """Look at audio/video files (or every one under a folder) without transcribing: each file's
    id, duration and audio tracks (one per microphone in a multi-track recording)."""
    return [{"file_id": it["file_id"], "filename": it["source"]["filename"],
             "duration": it["source"]["duration"],
             "tracks": [{"track": t["track_index"] + 1, "speaker": t["speaker"]} for t in it["source"]["tracks"]]}
            for it in _register(paths)]


def transcribe(paths: list[str], names: Optional[list[str]] = None, tracks: Optional[list[int]] = None,
               language: Optional[str] = None, out: Optional[str] = None,
               formats: list[str] = ["srt", "json"], wait: bool = False) -> dict:
    """Transcribe audio/video files into one merged, speaker-labeled transcript.

    paths: files to transcribe; several are joined in that order on one timeline.
    names: speaker name for each audio track, in track order (default "Speaker 1", "Speaker 2", ...).
    tracks: which tracks to use, numbered from 1 (default: all).
    language: language code such as "en", "pt-PT" or "pt-BR" (default: detect it).
    out: folder to write the transcript files to; implies waiting for the result.
    formats: what to write to `out`: srt, txt, csv, json.
    wait: wait until it's done (long files take minutes) and return {job_id, status, elapsed}.
    Returns {job_id} at once otherwise (plus "files" when `out` is given); follow it with
    get_progress, then read it with get_transcript. A file transcribed before with the same model
    and language comes from the cache and is done immediately.
    With no language given, a run can stop to ask before a big word-timing download for a language
    it detected: waiting then ends early with the question (see answer), and nothing is written
    to `out` yet (export it once the job is done)."""
    items = _register(paths)
    sources = [_selection(it, names, tracks, k) for k, it in enumerate(items)]
    job_id = _api("POST", "/api/transcribe", {"sources": sources, "language": language})["job_id"]
    result = {"job_id": job_id}
    if out or wait:
        p = _wait_job(job_id)
        if p.get("question"):
            return {**result, **_asked(job_id, p["question"])}
        result.update(status="done", elapsed=round(p["elapsed"], 1))
        if p.get("language"):
            result["language"] = p["language"]
        if out:
            merged, name = _merged(job_id)
            result["files"] = _write(merged, out, name, formats, {})
    return result


def get_progress(job_id: str) -> dict:
    """Progress of a transcription job: status (queued, running, done, error, cancelled), percent and
    stage. Poll every few seconds until status is done, then call get_transcript. One transcription
    runs on the GPU at a time: while another is running (in the app, a batch or another job), the
    stage says "waiting for another run to finish" and the percent stays put; that's not stuck.
    When the result has "question", the job has stopped to ask (its message says "waiting for your
    answer") and stays put until you call answer. Once done, "language" is what Auto-detect chose
    and how sure it was, e.g. "English 97%"."""
    p = _api("GET", f"/api/progress/{job_id}")
    out = {"status": p["status"], "percent": round(p["progress"] * 100), "message": p["message"],
           "elapsed": round(p["elapsed"], 1), "error": p["error"]}
    if p.get("language"):
        out["language"] = p["language"]
    if p.get("question"):
        out.update(_asked(job_id, p["question"]), status=p["status"])
    return out


def answer(run_id: str, choice: str, language: Optional[str] = None) -> dict:
    """Answer the question a job or batch stopped to ask (get_progress / get_batch show it under
    "question"). Auto-detect asks before a big word-timing download for a language that isn't in
    "Languages I use" (Models tab).

    run_id: the job id or batch id that is waiting.
    choice: "download" (get the model and continue), "skip" (continue without word timing for this file: line timings only) or "language" (use a different language instead).
    language: for choice "language": the code to use, such as "en" or "pt-PT"."""
    body = {"choice": choice, "language": language}
    try:
        return _api("POST", f"/api/answer/{run_id}", body)
    except ActionError as e:
        if not str(e).startswith("404"):
            raise
    return _api("POST", f"/api/batch/{run_id}/answer", body)


def get_transcript(source: str, full: bool = False) -> dict:
    """A finished transcript, from a job id or a WhisperMerger .json file: lines numbered from 0
    (the numbers edit_lines takes), with start and end (seconds), speaker, text and flags (overlap,
    backchannel, edited...). full: the complete transcript including word-level timings."""
    merged, _ = _merged(source)
    if full:
        return merged
    return {"speakers": sorted({s["speaker"] for s in merged["segments"]}),
            "lines": [{"line": i, "start": round(s["start"], 2), "end": round(s["end"], 2),
                       "speaker": s["speaker"], "text": s["text"], "flags": s.get("flags", [])}
                      for i, s in enumerate(merged["segments"])]}


def edit_lines(job_id: str, rename_from: Optional[str] = None, rename_to: Optional[str] = None,
               line: Optional[int] = None, text: Optional[str] = None, speaker: Optional[str] = None,
               start: Optional[float] = None, end: Optional[float] = None,
               delete: bool = False) -> dict:
    """Edit a finished job's transcript; later reads and exports use the edits. Give a rename, or a
    line with at least one change, or both.

    rename_from: a speaker name to rename on every line, to rename_to.
    rename_to: the new name for rename_from.
    line: line number from get_transcript (counted from 0) to change with text / speaker / start / end, or delete.
    A changed text is marked edited, so subtitle timing treats it like an edit made in the app."""
    merged = _api("GET", f"/api/result/{job_id}")
    segs = merged["segments"]
    changed = 0
    if rename_from is not None or rename_to is not None:
        if not (rename_from and rename_to):
            raise ActionError("a rename needs both rename_from and rename_to")
        names = sorted({s["speaker"] for s in segs})
        if rename_from not in names:
            raise ActionError(f"no speaker {rename_from!r} (speakers: {', '.join(names)})")
        for s in segs:
            if s["speaker"] == rename_from:
                s["speaker"] = rename_to
                changed += 1
        for src in merged.get("sources", []):
            for t in src.get("tracks", []):
                if t["speaker"] == rename_from:
                    t["speaker"] = rename_to
    if line is not None:
        if not 0 <= line < len(segs):
            raise ActionError(f"no line {line}: lines are numbered 0 to {len(segs) - 1}")
        if not delete and all(v is None for v in (text, speaker, start, end)):
            raise ActionError(f"line {line}: say what to change (text, speaker, start, end) or delete")
        if delete:
            segs.pop(line)
        else:
            s = segs[line]
            if text is not None and text != s["text"]:
                s["text"] = text
                if "edited" not in s.setdefault("flags", []):
                    s["flags"].append("edited")
            if speaker is not None:
                s["speaker"] = speaker
            if start is not None:
                s["start"] = start
            if end is not None:
                s["end"] = end
        changed += 1
    if not changed:
        raise ActionError("nothing to change: give rename_from + rename_to, or line with text/speaker/start/end/delete")
    _api("PUT", f"/api/result/{job_id}", merged)
    return {"changed_lines": changed, "lines": len(segs)}


def export(source: str, out: Optional[str] = None, formats: list[str] = ["srt"], filename: Optional[str] = None,
           max_chars: int = 42, max_lines: int = 2, pad: float = 0.0, subtitle_limits: bool = True,
           show_speakers: bool = True, name_on_change: bool = False, txt_timestamps: bool = True) -> dict:
    """Write a transcript (a job id or a WhisperMerger .json file) to files.

    out: folder to write to (default: the .json file's folder, or the current folder for a job).

    formats: srt, txt, csv, json. max_chars / max_lines: SRT subtitle limits (subtitle_limits turns
    them on or off). pad: seconds added to each line's start and end. name_on_change: in SRT, name
    the speaker only when it changes.
    txt_timestamps: start each TXT line with its start and end time (off, and show_speakers off: text alone).
    A TXT without timestamps opens in Edit Transcript as lines with no timing; keep a JSON to edit later."""
    merged, name = _merged(source)
    opts = _export_options(max_chars, max_lines, pad, subtitle_limits, show_speakers, name_on_change,
                           txt_timestamps)
    if out is None:
        src = Path(source).expanduser()
        out = str(src.parent) if src.is_file() else os.getcwd()
    return {"files": _write(merged, out, filename or name, formats, opts)}


def run_batch(paths: list[str], out: Optional[str] = None, per_file: bool = True,
              formats: list[str] = ["srt", "txt", "csv", "json"], language: Optional[str] = None,
              name: str = "Batch", wait: bool = False) -> dict:
    """Transcribe many files (or whole folders) in the background, writing transcripts as each finishes.

    out: folder for the transcripts (default: next to each source file).
    per_file: one transcript per file; off joins everything into one transcript.
    Returns the batch id; use get_batch to follow it. With no language given, the batch can stop to
    ask before a big word-timing download (see answer); waiting then ends early with the question."""
    items = _register(paths)
    project = {"name": name, "per_file": per_file, "formats": formats,
               "sources": [_selection(it, None, None, k) for k, it in enumerate(items)]}
    body = {"projects": [project], "language": language,
            "output_mode": "folder" if out else "source",
            "output_folder": str(Path(out).expanduser().resolve()) if out else None}
    batch_id = _api("POST", "/api/batch", body, timeout=120)["batch_id"]
    if not wait:
        return {"batch_id": batch_id, "files": len(items)}
    last = None
    while True:
        b = get_batch(batch_id)
        msg = f"{b['percent']}% · {b['done']}/{b['files']} files"
        if msg != last:
            _say(msg)
            last = msg
        if b["status"] in ("done", "canceled") or b.get("question"):
            return b
        time.sleep(2)


def get_batch(batch_id: str) -> dict:
    """Progress of a batch: overall percent, and each file's status, message and where its transcript
    went. When the result has "question", the batch has stopped to ask about the file named in it
    and stays put until you call answer."""
    b = _api("GET", f"/api/batch/{batch_id}")
    files = []
    for p in b["projects"]:
        rows = p.get("files") or [p]
        files += [{"name": f["name"], "status": f["status"], "percent": round(f["progress"] * 100),
                   "message": f["message"], "output_dir": f.get("output_dir"), "error": f.get("error")}
                  for f in rows]
    out = {"status": b["status"], "percent": round(b["progress"] * 100), "files": len(files),
           "done": sum(f["status"] == "done" for f in files), "items": files}
    if b.get("question"):
        out.update(_asked(batch_id, b["question"]), status=b["status"])
    return out


def list_models(kind: str = "speech", installed_only: bool = False) -> dict:
    """Models: which are downloaded, which speech model runs (active), and download progress.

    kind: "speech" (speech-to-text, default), "word-timing" (one per language, for word-level
    timing; about 40, mostly not downloaded) or "all".
    installed_only: only models already on disk."""
    want = {"speech": "whisper", "word-timing": "align", "all": None}
    if kind not in want:
        raise ActionError('kind is "speech", "word-timing" or "all"')
    d = _api("GET", "/api/models", timeout=120)
    models = [m for m in d["models"] if want[kind] in (None, m["kind"]) and (m["installed"] or not installed_only)]
    return {"active": d["active"],
            "downloaded": f"{sum(m['installed'] for m in d['models'] if want[kind] in (None, m['kind']))} "
                          f"of {sum(1 for m in d['models'] if want[kind] in (None, m['kind']))}",
            "models": [{"id": m["id"], "name": m["name"], "size_mb": m["size_mb"], "installed": m["installed"],
                        "download": (m["download"] or {}).get("text") if (m["download"] or {}).get("active") else None}
                       for m in models]}


def download_model(model_id: str) -> dict:
    """Start downloading a model by id (from list_models, e.g. "whisper:medium" or "align:nl"). Several queue up."""
    return _api("POST", f"/api/models/{urllib.parse.quote(model_id, safe='')}/download")


def use_model(model_id: str) -> dict:
    """Make a downloaded speech model (e.g. "whisper:large-v3") the one every run uses."""
    return _api("POST", f"/api/models/{urllib.parse.quote(model_id, safe='')}/use")


def cache_stats() -> dict:
    """How many transcriptions are cached (re-running a cached file is instant) and their size."""
    return _api("GET", "/api/cache/stats")


def clear_cache() -> dict:
    """DESTRUCTIVE: delete every cached transcription, so files are transcribed again from scratch."""
    return _api("POST", "/api/cache/clear")


def status() -> dict:
    """Whether the WhisperMerger server is running (it starts on first use), and its address."""
    return {"running": _up(), "url": BASE}


def shutdown() -> dict:
    """Stop the WhisperMerger server (unloads the models from the GPU). Running jobs are lost."""
    if not _up():
        return {"running": False}
    _request("POST", "/api/shutdown")
    return {"stopped": True}


# The one list. Destructive ones are marked for MCP clients.
ACTIONS = [add_files, transcribe, get_progress, answer, get_transcript, edit_lines, export, run_batch,
           get_batch, list_models, download_model, use_model, cache_stats, clear_cache, status, shutdown]
DESTRUCTIVE = {clear_cache, shutdown}
