"""WhisperMerger actions for scripts and AI: the one list the CLI (cli.py) and the MCP server
(mcp_server.py) are both built from, so the two can't drift apart (#36).

Each action is a plain function over the local server's HTTP API (backend/app/routes/api.py). Its
signature and docstring ARE the interface: cli.py turns them into commands and flags, mcp_server.py
into tools. The server is started on first use if it isn't running, and keeps running afterwards
so the speech model stays loaded between calls. Files are read in place; nothing is uploaded.
"""

from __future__ import annotations

import csv
import inspect
import io
import json
import os
import re
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
EDITED_TIMING = ("align", "even", "whole")
BACKCHANNEL_RENDER = ("parenthetical", "separate", "skip")

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


def _run_post(run_id: str, verb: str, body=None) -> dict:
    """POST to a job's route, or to the batch's of the same name when the id isn't a job's."""
    try:
        return _api("POST", f"/api/{verb}/{run_id}", body)
    except ActionError as e:
        if not str(e).startswith("404"):
            raise
    return _api("POST", f"/api/batch/{run_id}/{verb}", body)


def _batch_path(batch_id: str, project: Optional[int], file: Optional[int]) -> str:
    """The route of one project, or one of its files, in a batch (both numbered from 1, as get_batch shows)."""
    return f"/api/batch/{batch_id}/project/{(project or 1) - 1}" + (f"/file/{file - 1}" if file else "")


def _paused(p: dict) -> str:
    return "GPU released" if p["gpu_freed"] else "pausing: the track being transcribed finishes first"


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
        if p["status"] in ("error", "canceled"):
            raise ActionError(f"job {p['status']}: {p.get('error') or p['message']}")
        time.sleep(1)


def _result(run_id: str, project: Optional[int] = None, file: Optional[int] = None) -> tuple[dict, str]:
    """A finished transcript on the server and its route: a job's, or with a project / file number
    one of a batch's."""
    if project is None and file is None:
        path = f"/api/result/{run_id}"
        try:
            return _api("GET", path), path
        except ActionError as e:
            if not str(e).startswith("404"):
                raise
            try:                                 # a batch id without its project / file number?
                b = _api("GET", f"/api/batch/{run_id}")
            except ActionError:
                raise e from None
        parts = [f'project {n} "{p["name"]}"' + (" (files: " + ", ".join(
                     f"{m} {f['name']}" for m, f in enumerate(p["files"], 1)) + ")" if p.get("files") else "")
                 for n, p in enumerate(b["projects"], 1)]
        raise ActionError(f"{run_id} is a batch id: add project, and file for a project with one transcript "
                          f"per file (numbers as get_batch shows them). Its projects: {'; '.join(parts)}")
    path = _batch_path(run_id, project, file) + "/result"
    return _api("GET", path)["merged"], path


# Reading .srt / .txt / .csv transcripts: the same rules as the Edit Transcript tab (parseSrt /
# parseTxt / parseCsv in frontend/app.js), which mirror what backend/app/export.py writes. These
# formats carry no word timings.

def _srt_time(v: str) -> float:
    """"HH:MM:SS,mmm" (or with ".") -> seconds; 0 when it isn't a time."""
    m = re.search(r"(\d+):(\d\d):(\d\d)[,.](\d{1,3})", v)
    return int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + float("0." + m[4]) if m else 0.0


def _line(start: float, end: float, body: str, flags: tuple = ()) -> dict:
    """One transcript line; a leading "Name: " (as exports with speakers write it) is the speaker."""
    m = re.fullmatch(r"([^:]{1,40}):\s(.*)", body)
    return {"speaker": m[1].strip() if m else "Speaker", "start": start, "end": end,
            "text": m[2] if m else body, "words": [], "flags": list(flags)}


def _parse_srt(text: str) -> list[dict]:
    segs = []
    for block in re.split(r"\n\s*\n", text):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        i = 1 if lines and re.fullmatch(r"\d+", lines[0].strip()) else 0     # optional cue number
        tc = re.search(r"(.+?)\s*-->\s*(.+)", lines[i]) if i < len(lines) else None
        if tc:
            segs.append(_line(_srt_time(tc[1]), _srt_time(tc[2]), " ".join(lines[i + 1:]).strip()))
    return segs


def _parse_txt(text: str) -> list[dict]:
    """A line without "[start --> end]" (TXT exported with timestamps off) loads at 0, flagged no_timing."""
    segs = []
    for line in filter(None, map(str.strip, text.split("\n"))):
        m = re.fullmatch(r"\[(.+?)\s*-->\s*(.+?)\]\s*(.*)", line)
        segs.append(_line(_srt_time(m[1]), _srt_time(m[2]), m[3]) if m else _line(0.0, 0.0, line, ("no_timing",)))
    return segs


def _parse_csv(text: str) -> list[dict]:
    """Columns start, end, speaker, text, flags: by name when the first row names them, else in that order."""
    rows = [r for r in csv.reader(io.StringIO(text)) if any(r)]
    cols = ["start", "end", "speaker", "text", "flags"]
    head = [h.strip().lower() for h in rows[0]] if rows else []
    if head[:2] == cols[:2]:
        at, rows = {c: head.index(c) if c in head else -1 for c in cols}, rows[1:]
    else:
        at = {c: i for i, c in enumerate(cols)}

    def cell(r, c):
        return r[at[c]] if 0 <= at[c] < len(r) else ""

    def num(v):
        try:
            return float(v)
        except ValueError:
            return 0.0

    return [{"speaker": (cell(r, "speaker") or "Speaker").strip(), "start": num(cell(r, "start")),
             "end": num(cell(r, "end")), "text": cell(r, "text"), "words": [],
             "flags": [f.strip() for f in cell(r, "flags").split("|") if f.strip()]} for r in rows]


_PARSERS = {".srt": _parse_srt, ".txt": _parse_txt, ".csv": _parse_csv}


def _merged(source: str, project: Optional[int] = None, file: Optional[int] = None) -> tuple[dict, str]:
    """A transcript from a job id, a batch id with a project / file number, or a transcript file (a
    WhisperMerger .json, or an .srt, .txt or .csv), plus a default export name."""
    path = Path(source).expanduser()
    ext = path.suffix.lower()
    if ext == ".json" and path.is_file():
        return json.loads(path.read_text("utf-8")), path.stem
    if ext in _PARSERS and path.is_file():
        segs = _PARSERS[ext](path.read_text("utf-8-sig", errors="replace"))
        if not segs:
            raise ActionError(f"no transcript lines found in {path.name}")
        return {"segments": segs}, path.stem
    merged, _ = _result(source, project, file)
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
                    show_speakers: bool, name_on_change: bool, txt_timestamps: bool,
                    min_duration: float = 1.0, edited_timing: str = "align") -> dict:
    if edited_timing not in EDITED_TIMING:
        raise ActionError(f"edited_timing is one of {', '.join(EDITED_TIMING)}")
    return {"apply_limits": subtitle_limits, "chars_per_line": max_chars, "max_lines": max_lines,
            "pad_seconds": pad, "show_speakers": show_speakers, "label_on_change": name_on_change,
            "txt_timestamps": txt_timestamps, "min_duration": min_duration, "edited_timing": edited_timing}


_EXPORT_KEYS = tuple(inspect.signature(_export_options).parameters)
# How the tracks are merged: the server's own names (MergeOptionsIn), so they pass straight through.
_MERGE_KEYS = ("detect_conflicts", "backchannel_render", "backchannel_max_words", "overlap_min_seconds",
               "bleed_window_seconds", "bleed_similarity", "bleed_prefer_ratio")


def _merge_options(values: dict) -> dict:
    """The merge options among `values`, checked here so a wrong value gets a plain answer, not the server's 422."""
    if values["backchannel_render"] not in BACKCHANNEL_RENDER:
        raise ActionError(f"backchannel_render is one of {', '.join(BACKCHANNEL_RENDER)}")
    return {k: values[k] for k in _MERGE_KEYS}


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
               formats: list[str] = ["srt", "json"], wait: bool = False,
               detect_conflicts: bool = True, backchannel_render: str = "parenthetical",
               backchannel_max_words: int = 4, overlap_min_seconds: float = 2.0,
               bleed_window_seconds: float = 1.0, bleed_similarity: float = 0.7,
               bleed_prefer_ratio: float = 2.0, word_timing: bool = True) -> dict:
    """Transcribe audio/video files into one merged, speaker-labeled transcript.

    paths: files to transcribe; several are joined in that order on one timeline.
    names: speaker name for each audio track, in track order (default "Speaker 1", "Speaker 2", ...).
    tracks: which tracks to use, numbered from 1 (default: all).
    language: language code such as "en", "pt-PT" or "pt-BR" (default: detect it).
    out: folder to write the transcript files to; implies waiting for the result.
    formats: what to write to `out`: srt, txt, csv, json.
    wait: wait until it's done (long files take minutes) and return {job_id, status, elapsed}.
    detect_conflicts: flag backchannels, overlaps and microphone bleed between speakers; off only orders the lines by time.
    backchannel_render: a short reaction ("mm-hm") inside someone else's line is written "parenthetical" (its own line, in brackets), "separate" (its own line, as said) or "skip" (left out).
    backchannel_max_words: a reaction counts as a backchannel when it has fewer words than this.
    overlap_min_seconds: two speakers talking at once for at least this long are flagged as an overlap.
    bleed_window_seconds: bleed is one voice picked up by two microphones; lines that start within this many seconds of each other are compared for it.
    bleed_similarity: how alike two such lines must read to count as bleed, from 0 to 1.
    bleed_prefer_ratio: the louder microphone's line is marked preferred only when it is at least this many times louder.
    word_timing: time every word (the default). Off takes roughly half the time per file and needs no per-language word-timing model, so none is downloaded or asked about. The words are the same, but there are no word-level times and the lines are longer: one per stretch of speech (up to about 30 seconds) instead of one per sentence, so line start and end times are coarser and SRT splits inside a line are estimated. The result's "language" then says "(no word timing)".
    Returns {job_id} at once otherwise (plus "files" when `out` is given); follow it with
    get_progress, then read it with get_transcript. A file transcribed before with the same model
    and language comes from the cache and is done immediately.
    With no language given, a run can stop to ask before a big word-timing download for a language
    it detected: waiting then ends early with the question (see answer), and nothing is written
    to `out` yet (export it once the job is done)."""
    options = _merge_options(locals())
    items = _register(paths)
    sources = [_selection(it, names, tracks, k) for k, it in enumerate(items)]
    job_id = _api("POST", "/api/transcribe",
                  {"sources": sources, "language": language, "options": options,
                   "word_timing": word_timing})["job_id"]
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
    """Progress of a transcription job: status (queued, running, done, error, canceled), percent and
    stage. Poll every few seconds until status is done, then call get_transcript. One transcription
    runs on the GPU at a time: while another is running (in the app, a batch or another job), the
    stage says "waiting for another run to finish" and the percent stays put; that's not stuck.
    When the result has "question", the job has stopped to ask (its message says "waiting for your
    answer") and stays put until you call answer. Once done, "language" is what Auto-detect chose
    and how sure it was, e.g. "English 97%", with "(no word timing)" when the run went without it.
    A canceled job that had finished some of its tracks says how many, e.g. "canceled · 1 of 2
    tracks transcribed, kept for a retry": nothing is written, but running it again reuses them."""
    p = _api("GET", f"/api/progress/{job_id}")
    out = {"status": p["status"], "percent": round(p["progress"] * 100), "message": p["message"],
           "elapsed": round(p["elapsed"], 1), "error": p["error"]}
    if p.get("language"):
        out["language"] = p["language"]
    if p["paused"]:
        out["paused"] = _paused(p)
    if p.get("question"):
        out.update(_asked(job_id, p["question"]), status=p["status"])
    return out


def answer(run_id: str, choice: str, language: Optional[str] = None) -> dict:
    """Answer the question a job or batch stopped to ask (get_progress / get_batch show it under
    "question"). Auto-detect asks before a big word-timing download for a language that isn't in
    "Languages I use" (Models tab, or set_my_languages).

    run_id: the job id or batch id that is waiting.
    choice: "download" (get the model and continue), "skip" (continue without word timing for this file: line timings only) or "language" (use a different language instead).
    language: for choice "language": the code to use, such as "en" or "pt-PT"."""
    return _run_post(run_id, "answer", {"choice": choice, "language": language})


def cancel(run_id: str, project: Optional[int] = None, file: Optional[int] = None) -> dict:
    """Cancel a job or a whole batch, or only one project or one file of a batch (the rest of the
    batch carries on). The step in progress finishes first; what was canceled can be run again
    with retry_batch.

    run_id: the job id or batch id.
    project: batch only: cancel just this project, by its number in get_batch.
    file: batch only: cancel just this file of the project, by its number in get_batch (project defaults to 1)."""
    if project is None and file is None:
        return _run_post(run_id, "cancel")
    return _api("POST", _batch_path(run_id, project, file) + "/cancel")


def pause(run_id: str) -> dict:
    """Pause a job or a batch. The track being transcribed finishes first (its transcription and
    word timing; Label Track: the speaker, Benchmark: the model), then the models leave the GPU, the
    clock stops and other runs can use the GPU until resume. get_progress / get_batch show "paused":
    "pausing" until that track ends, then "GPU released". A run that is still waiting for another
    run to finish pauses at once. Cancel doesn't wait for the track.

    run_id: the job id or batch id."""
    return _run_post(run_id, "pause")


def resume(run_id: str) -> dict:
    """Resume a paused job or batch: the model loads again and the run continues where it stopped.

    run_id: the job id or batch id."""
    return _run_post(run_id, "resume")


def retry_batch(batch_id: str, projects: Optional[list[int]] = None, file: Optional[int] = None) -> dict:
    """Run parts of a batch again, in the same batch: by default every failed or canceled project.
    While the batch is still running they queue up behind it. Files transcribed before come from
    the cache. Returns the batch's progress, as get_batch.

    projects: project numbers from get_batch to run again, finished ones included (default: all failed or canceled).
    file: run only this finished file again, by its number in get_batch; give its one project in projects (default 1)."""
    if file is not None:
        if projects and len(projects) > 1:
            raise ActionError("file goes with one project: give a single number in projects")
        _api("POST", _batch_path(batch_id, projects[0] if projects else 1, file) + "/restart")
    else:
        _api("POST", f"/api/batch/{batch_id}/retry", {"projects": [p - 1 for p in projects or []]})
    return get_batch(batch_id)


def get_transcript(source: str, full: bool = False, project: Optional[int] = None,
                   file: Optional[int] = None) -> dict:
    """A finished transcript, from a job id, a batch id (with project or file) or a transcript file
    (a WhisperMerger .json, or an .srt, .txt or .csv): lines numbered from 0 (the numbers edit_lines
    takes), with start and end (seconds), speaker, text and flags (overlap, backchannel, edited...).
    Only a job, a batch result or a .json has word-level timings; a TXT line without a timestamp
    starts at 0 with the flag no_timing.

    full: the complete transcript including word-level timings.
    project: with a batch id: the project's number in get_batch.
    file: with a batch id: the file's number in get_batch, when the project wrote one transcript per file (project defaults to 1)."""
    merged, _ = _merged(source, project, file)
    if full:
        return merged
    return {"speakers": sorted({s["speaker"] for s in merged["segments"]}),
            "lines": [{"line": i, "start": round(s["start"], 2), "end": round(s["end"], 2),
                       "speaker": s["speaker"], "text": s["text"], "flags": s.get("flags", [])}
                      for i, s in enumerate(merged["segments"])]}


# Top-level key edit_lines puts in the .json files it writes: such a file may be edited in place.
_EDITED_COPY = "edit_lines_copy"


def edit_lines(source: str, rename_from: Optional[str] = None, rename_to: Optional[str] = None,
               line: Optional[int] = None, text: Optional[str] = None, speaker: Optional[str] = None,
               start: Optional[float] = None, end: Optional[float] = None,
               delete: bool = False, out: Optional[str] = None,
               project: Optional[int] = None, file: Optional[int] = None) -> dict:
    """Edit a finished transcript of a job, of a batch (with project or file), or a transcript file
    (a WhisperMerger .json, or an .srt, .txt or .csv). A job's or batch's transcript is edited in
    place: later reads and exports of it use the edits, the files a batch already wrote don't
    change. A transcript file is never written over: the edited copy goes to out, always a
    WhisperMerger .json (export turns it into SRT, TXT or CSV). A .json that edit_lines wrote can
    be edited again in place, with out left out. Give a rename, or a line with at least one
    change, or both.

    source: the job id, a batch id, or a transcript file.
    out: with a transcript file: the .json file to write the edited copy to (an existing file there is overwritten, as export does).
    project: with a batch id: the project's number in get_batch.
    file: with a batch id: the file's number in get_batch (project defaults to 1).
    rename_from: a speaker name to rename on every line, to rename_to.
    rename_to: the new name for rename_from.
    line: line number from get_transcript (counted from 0) to change with text / speaker / start / end, or delete.
    A changed text is marked edited, so subtitle timing treats it like an edit made in the app."""
    src = Path(source).expanduser()
    dest = where = None
    if src.suffix.lower() in (".json", *_PARSERS) and src.is_file():
        merged, _ = _merged(source)
        mine = merged.get(_EDITED_COPY) is True
        if out is None and not mine:
            raise ActionError(f"{src.name} is a transcript file: add out, the .json file to write the edited copy to")
        dest = Path(out).expanduser() if out is not None else src
        if dest.suffix.lower() != ".json":
            raise ActionError("out must end in .json: the edited copy is a WhisperMerger .json, "
                              "and export turns it into SRT, TXT or CSV")
        if not mine and dest.exists() and dest.samefile(src):
            raise ActionError(f"that would write over {src.name}, the original; give out another file name")
    else:
        if out is not None:
            raise ActionError("out is only for a transcript file; a job's or batch's transcript is edited in place")
        merged, where = _result(source, project, file)
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
    if dest is None:
        _api("PUT", where, merged)
        return {"changed_lines": changed, "lines": len(segs)}
    merged[_EDITED_COPY] = True
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(merged, ensure_ascii=False, indent=2), "utf-8")
    return {"changed_lines": changed, "lines": len(segs), "file": str(dest.resolve())}


def export(source: str, out: Optional[str] = None, formats: list[str] = ["srt"], filename: Optional[str] = None,
           max_chars: int = 42, max_lines: int = 2, pad: float = 0.0, subtitle_limits: bool = True,
           show_speakers: bool = True, name_on_change: bool = False, txt_timestamps: bool = True,
           min_duration: float = 1.0, edited_timing: str = "align",
           project: Optional[int] = None, file: Optional[int] = None) -> dict:
    """Write a transcript (a job id, a batch id with project or file, or a transcript file: a
    WhisperMerger .json, or an .srt, .txt or .csv) to files. It never writes over the file it is reading.

    out: folder to write to (default: the transcript file's folder, or the current folder for a job or batch).
    project: with a batch id: the project's number in get_batch.
    file: with a batch id: the file's number in get_batch (project defaults to 1).

    formats: srt, txt, csv, json. max_chars / max_lines: SRT subtitle limits (subtitle_limits turns
    them on or off). pad: seconds added to each line's start and end. name_on_change: in SRT, name
    the speaker only when it changes.
    txt_timestamps: start each TXT line with its start and end time (off, and show_speakers off: text alone).
    A TXT without timestamps opens in Edit Transcript as lines with no timing; keep a JSON to edit later.
    min_duration: shortest time an SRT subtitle stays on screen, in seconds (part of the subtitle limits).
    edited_timing: SRT timing for a line whose text was edited: "align" (keep word timing for unchanged words), "even" (spread the text evenly over the line) or "whole" (one subtitle per line)."""
    merged, name = _merged(source, project, file)
    opts = _export_options(max_chars, max_lines, pad, subtitle_limits, show_speakers, name_on_change,
                           txt_timestamps, min_duration, edited_timing)
    src, name = Path(source).expanduser(), filename or name
    if out is None:
        out = str(src.parent) if src.is_file() else os.getcwd()
    for dest in (Path(out).expanduser() / f"{name}.{f}" for f in formats):
        if src.is_file() and dest.exists() and dest.samefile(src):
            raise ActionError(f"that would write over {src.name}, the transcript being read; "
                              "give another out folder or filename")
    return {"files": _write(merged, out, name, formats, opts)}


def run_batch(paths: list[str], out: Optional[str] = None, per_file: bool = True,
              formats: list[str] = ["srt", "txt", "csv", "json"], language: Optional[str] = None,
              name: str = "Batch", wait: bool = False, names: Optional[list[str]] = None,
              tracks: Optional[list[int]] = None, projects: Optional[list[dict]] = None,
              max_chars: int = 42, max_lines: int = 2, pad: float = 0.0, subtitle_limits: bool = True,
              show_speakers: bool = True, name_on_change: bool = False, txt_timestamps: bool = True,
              min_duration: float = 1.0, edited_timing: str = "align",
              detect_conflicts: bool = True, backchannel_render: str = "parenthetical",
              backchannel_max_words: int = 4, overlap_min_seconds: float = 2.0,
              bleed_window_seconds: float = 1.0, bleed_similarity: float = 0.7,
              bleed_prefer_ratio: float = 2.0, word_timing: bool = True) -> dict:
    """Transcribe many files (or whole folders) in the background, writing transcripts as each finishes.

    paths: files or folders, which make one project called `name`; none when `projects` lists everything.
    out: folder for the transcripts (default: next to each source file).
    per_file: one transcript per file; off joins everything into one transcript.
    names: speaker name for each audio track, in track order, used for every file.
    tracks: which tracks to use, numbered from 1 (default: all).
    projects: more projects for the same run. Each is an object with "paths" and, where it differs from what is given here, "name" (default "Project 2", ...), "names", "tracks", "per_file", "formats" or any of the options below. On the command line: JSON text, or the path of a .json file holding the list.
    max_chars, max_lines, pad, subtitle_limits, show_speakers, name_on_change, txt_timestamps,
    min_duration, edited_timing: how the files are written, as in export.
    detect_conflicts, backchannel_render, backchannel_max_words, overlap_min_seconds,
    bleed_window_seconds, bleed_similarity, bleed_prefer_ratio: how the tracks are merged, as in transcribe.
    word_timing: time every word (the default). Off takes roughly half the time per file and needs no per-language word-timing model; the words are the same, but there are no word-level times and the lines are longer, so line times are coarser and SRT splits inside a line are estimated (see transcribe). A file's message then ends with "(no word timing)".
    Returns the batch id; use get_batch to follow it. With no language given, the batch can stop to
    ask before a big word-timing download (see answer); waiting then ends early with the question."""
    given = dict(locals())                     # every setting by name; a project's own values go over these
    own = set(given) - {"out", "language", "wait", "projects"}
    batch, files = [], 0
    for n, spec in enumerate(([{"paths": paths, "name": name}] if paths else []) + (projects or []), 1):
        if not isinstance(spec, dict) or not spec.get("paths") or set(spec) - own:
            raise ActionError(f'project {n}: give an object with "paths" and any of {", ".join(sorted(own - {"paths"}))}')
        p = {**given, "name": f"Project {n}", **spec}
        items = _register(p["paths"])
        files += len(items)
        batch.append({"name": p["name"], "per_file": p["per_file"], "formats": p["formats"],
                      "sources": [_selection(it, p["names"], p["tracks"], k) for k, it in enumerate(items)],
                      "export": _export_options(**{k: p[k] for k in _EXPORT_KEYS}),
                      "options": _merge_options(p), "word_timing": p["word_timing"]})
    if not batch:
        raise ActionError("nothing to run: give paths, or projects")
    body = {"projects": batch, "language": language,
            "output_mode": "folder" if out else "source",
            "output_folder": str(Path(out).expanduser().resolve()) if out else None}
    batch_id = _api("POST", "/api/batch", body, timeout=120)["batch_id"]
    if not wait:
        return {"batch_id": batch_id, "files": files}
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
    went. Each item carries its project number and, in a one-transcript-per-file project, its file
    number: the numbers cancel, retry_batch, get_transcript, edit_lines and export take. "files" and
    "done" count source files, as run_batch does: a project joined into one transcript is one item
    that counts as every file in it. A canceled item that had finished some of its tracks says how
    many, e.g. "canceled · 1 of 2 tracks transcribed, kept for a retry": nothing of it is written,
    but retry_batch reuses them. When the result has "question", the batch has stopped to ask
    about the file named in it and stays put until you call answer."""
    b = _api("GET", f"/api/batch/{batch_id}")
    files, total, done = [], 0, 0
    for n, p in enumerate(b["projects"], 1):
        rows = [{"project": n, "file": m, **f} for m, f in enumerate(p.get("files") or [], 1)] or [{"project": n, **p}]
        files += [{**{k: f[k] for k in ("project", "file") if k in f},
                   "name": f["name"], "status": f["status"], "percent": round(f["progress"] * 100),
                   "message": f["message"], "output_dir": f.get("output_dir"), "error": f.get("error")}
                  for f in rows]
        each = 1 if p.get("files") else p["sources"]      # source files behind each of its rows
        total += each * len(rows)
        done += each * sum(f["status"] == "done" for f in rows)
    out = {"batch_id": batch_id, "status": b["status"], "percent": round(b["progress"] * 100), "files": total,
           "done": done, "items": files}
    if b["paused"]:
        out["paused"] = _paused(b)
    if b.get("question"):
        out.update(_asked(batch_id, b["question"]), status=b["status"])
    return out


def list_models(kind: str = "speech", installed_only: bool = False) -> dict:
    """Models: which are downloaded, which speech model runs (active), and download progress.
    Also the settings: "my_languages" ("Languages I use", see set_my_languages; empty = not set),
    "auto_portuguese" (see set_auto_portuguese) and "pt_punctuation" (see set_pt_punctuation).
    "disk_free_mb" is the free space on the drive models download to. A model with "partial" has
    a stopped download on disk: download_model resumes it, remove_model deletes it.

    kind: "speech" (speech-to-text, default), "word-timing" (one per language, for word-level
    timing; about 40, mostly not downloaded) or "all".
    installed_only: only models already on disk."""
    want = {"speech": "whisper", "word-timing": "align", "all": None}
    if kind not in want:
        raise ActionError('kind is "speech", "word-timing" or "all"')
    d = _api("GET", "/api/models", timeout=120)
    models = [m for m in d["models"] if want[kind] in (None, m["kind"]) and (m["installed"] or not installed_only)]
    return {"active": d["active"], "my_languages": d["my_languages"],
            "auto_portuguese": d["auto_portuguese"], "pt_punctuation": d["pt_punctuation"],
            "disk_free_mb": round(d["disk_free"] / 1e6),
            "downloaded": f"{sum(m['installed'] for m in d['models'] if want[kind] in (None, m['kind']))} "
                          f"of {sum(1 for m in d['models'] if want[kind] in (None, m['kind']))}",
            "models": [{"id": m["id"], "name": m["name"], "size_mb": m["size_mb"], "installed": m["installed"],
                        "partial": m["partial"],
                        "download": (m["download"] or {}).get("text") if (m["download"] or {}).get("active") else None}
                       for m in models]}


def download_model(model_id: str) -> dict:
    """Start downloading a model by id (from list_models, e.g. "whisper:medium" or "align:nl"). Several queue up."""
    return _api("POST", f"/api/models/{urllib.parse.quote(model_id, safe='')}/download")


def use_model(model_id: str) -> dict:
    """Make a downloaded speech model (e.g. "whisper:large-v3") the one every run uses."""
    return _api("POST", f"/api/models/{urllib.parse.quote(model_id, safe='')}/use")


def cancel_download(model_id: str) -> dict:
    """Stop a model download that is queued or running. What was downloaded stays on disk, so
    download_model carries on from there; remove_model deletes it."""
    return _api("POST", f"/api/models/{urllib.parse.quote(model_id, safe='')}/cancel")


def remove_model(model_id: str) -> dict:
    """DESTRUCTIVE: delete a model's files from disk, a partial download included. Refused for the
    speech model in use (choose another with use_model first) and for a model that is downloading
    (cancel_download first)."""
    return _api("DELETE", f"/api/models/{urllib.parse.quote(model_id, safe='')}")


def run_benchmark(wait: bool = False) -> dict:
    """Time every downloaded speech model on this machine: each loads from cold and transcribes a
    test sentence spoken by a Windows voice (a European Portuguese one too, when a model and a
    voice for it are installed). It uses the GPU, so it waits for a running transcription first.

    wait: wait until it's done and return the results.
    Returns {job_id} at once otherwise; follow it with get_progress, then read get_benchmark."""
    job_id = _api("POST", "/api/benchmark")["job_id"]
    if not wait:
        return {"job_id": job_id}
    _wait_job(job_id)
    return get_benchmark(job_id)


def get_benchmark(job_id: str) -> dict:
    """Results of a benchmark started with run_benchmark, complete once "status" is done. For each
    test sentence and model: load_s (seconds to load the model), transcribe_s, speed_x (times
    faster than real time), match (0 to 1, how much of the sentence it got right) and heard (the
    text it wrote). "expected" is what was said."""
    return _api("GET", f"/api/benchmark/{job_id}")


def set_my_languages(languages: list[str] = []) -> dict:
    """Set "Languages I use": the languages your recordings are in (list_models shows the current
    list). Auto-detect accepts a guess among them; a guess outside them must be at least 50% sure,
    and a word-timing download over 1.5 GB for a language outside them makes the run stop and ask.

    languages: language codes such as en,pt; this replaces the whole list, and leaving it out empties it (no restriction)."""
    return _api("POST", "/api/settings/my-languages", {"value": languages})


def set_auto_portuguese(variant: Optional[str] = None) -> dict:
    """Set what Auto-detect treats detected Portuguese as (list_models shows the current choice).

    variant: "pt-PT" (Portugal) or "pt-BR" (Brazil); leaving it out goes back to plain Portuguese with no variant."""
    if variant not in (None, "pt-PT", "pt-BR"):
        raise ActionError('variant is "pt-PT" or "pt-BR" (leave it out for none)')
    return _api("POST", "/api/settings/auto-portuguese", {"value": variant})


def set_pt_punctuation(on: bool = False) -> dict:
    """Turn the Portugal punctuation option on or off (list_models shows it): pt-PT runs then also
    run large-v3 and take its capitals and punctuation, at about twice the GPU time.

    on: turn it on; leaving it out turns it off."""
    return _api("POST", "/api/settings/pt-punctuation", {"value": on})


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
ACTIONS = [add_files, transcribe, get_progress, answer, cancel, pause, resume, get_transcript, edit_lines,
           export, run_batch, get_batch, retry_batch, list_models, download_model, cancel_download,
           remove_model, use_model, run_benchmark, get_benchmark, set_my_languages, set_auto_portuguese,
           set_pt_punctuation, cache_stats, clear_cache, status, shutdown]
DESTRUCTIVE = {remove_model, clear_cache, shutdown}
