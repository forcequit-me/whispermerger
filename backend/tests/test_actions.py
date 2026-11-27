"""CLI + MCP actions (#36), run against the real app through FastAPI's TestClient (no port, no GPU)."""

import json
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import routes
from app.main import create_app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import actions  # noqa: E402
import cli  # noqa: E402


def _stub_transcribe(_wav, **_kw):
    return {"segments": [{"start": 0.0, "end": 1.0, "text": "hello",
                          "words": [{"word": "hello", "start": 0.0, "end": 1.0}]}]}


@pytest.fixture
def api(monkeypatch):
    """Point the actions at an in-process app instead of the local server."""
    monkeypatch.setattr(routes.api, "transcribe_track", _stub_transcribe)
    client = TestClient(create_app())

    def request(method, path, body=None, timeout=120, raw=False):
        r = client.request(method, path, json=body)
        if r.status_code >= 400:
            raise actions.ActionError(f"{r.status_code}: {r.json().get('detail')}")
        return r.content if raw else r.json()

    monkeypatch.setattr(actions, "_request", request)
    monkeypatch.setattr(actions, "ensure_server", lambda: None)
    monkeypatch.setattr(actions.time, "sleep", lambda s: None)
    return client


def _held(monkeypatch) -> threading.Event:
    """Transcription waits until the test sets this, so a run can be controlled while it is running."""
    gate = threading.Event()

    def stub(_wav, **_kw):
        gate.wait(5)
        return _stub_transcribe(_wav)

    monkeypatch.setattr(routes.api, "transcribe_track", stub)
    monkeypatch.setattr(routes.api, "unload_models", lambda: None)
    return gate


def _until(check, seconds: float = 5) -> None:
    deadline = time.time() + seconds
    while not check() and time.time() < deadline:
        pass
    assert check()


def test_transcribe_edit_export(api, two_stream_file, tmp_path):
    job = actions.transcribe([str(two_stream_file)], names=["Host", "Guest"], out=str(tmp_path),
                             formats=["srt", "json"])
    assert job["status"] == "done" and len(job["files"]) == 2
    t = actions.get_transcript(job["job_id"])
    assert t["speakers"] == ["Guest", "Host"]

    actions.edit_lines(job["job_id"], rename_from="Host", rename_to="Presenter")
    actions.edit_lines(job["job_id"], line=0, text="fixed")
    t = actions.get_transcript(job["job_id"])
    assert "Presenter" in t["speakers"] and t["lines"][0]["text"] == "fixed"
    assert "edited" in t["lines"][0]["flags"]

    out = actions.export(job["job_id"], out=str(tmp_path), formats=["srt"], filename="cut")
    assert "Presenter: " in Path(out["files"][0]).read_text("utf-8")
    # a saved .json transcript exports too, next to itself by default
    saved = next(f for f in job["files"] if f.endswith(".json"))
    assert actions.export(saved, formats=["txt"])["files"][0].startswith(str(tmp_path))
    # TXT keeps "[start --> end]" by default; the switch drops it (and the names, with show_speakers off).
    # The saved file was written before the rename above, so it still says Host.
    txt = lambda **kw: Path(actions.export(saved, formats=["txt"], **kw)["files"][0]).read_text("utf-8")
    assert txt().startswith("[00:00:00,000 --> ")
    assert txt(txt_timestamps=False) == "Host: hello\nGuest: hello"
    assert txt(txt_timestamps=False, show_speakers=False) == "hello\nhello"

    with pytest.raises(actions.ActionError, match="no line 99: lines are numbered 0 to"):
        actions.edit_lines(job["job_id"], line=99, text="x")
    with pytest.raises(actions.ActionError, match=r"no speaker 'Nobody' \(speakers: Guest, Presenter\)"):
        actions.edit_lines(job["job_id"], rename_from="Nobody", rename_to="X")
    with pytest.raises(actions.ActionError, match="say what to change"):
        actions.edit_lines(job["job_id"], line=0)
    assert {m["id"].split(":")[0] for m in actions.list_models()["models"]} == {"whisper"}


def test_export_min_duration_and_edited_timing(api, tmp_path):
    """A line whose last word was edited: the three SRT timings for it, and the shortest subtitle."""
    words = [{"word": w, "start": s, "end": s + 1} for w, s in (("aaaa", 0), ("bbbb", 1), ("cccc", 8), ("dddd", 9))]
    src = tmp_path / "talk.json"
    src.write_text(json.dumps({"segments": [{"speaker": "A", "start": 0, "end": 10, "text": "aaaa bbbb cccc dddx",
                                             "words": words, "flags": ["edited"]}]}), "utf-8")

    def cues(**kw):
        out = actions.export(str(src), formats=["srt"], max_chars=9, max_lines=1, show_speakers=False, **kw)
        return [ln for ln in Path(out["files"][0]).read_text("utf-8").splitlines() if "-->" in ln]

    assert cues() == ["00:00:00,000 --> 00:00:02,000", "00:00:08,000 --> 00:00:10,000"]    # align: the default
    assert cues(edited_timing="even") == ["00:00:00,000 --> 00:00:05,000", "00:00:05,000 --> 00:00:10,000"]
    assert cues(edited_timing="whole") == ["00:00:00,000 --> 00:00:10,000"]
    assert cues(min_duration=3)[0] == "00:00:00,000 --> 00:00:03,000"
    with pytest.raises(actions.ActionError, match="edited_timing is one of align, even, whole"):
        cues(edited_timing="guess")
    a = cli.build_parser().parse_args(["export", "x.json", "--min-duration", "2.5", "--edited-timing", "whole"])
    assert a.min_duration == 2.5 and a.edited_timing == "whole"
    assert cli.build_parser().parse_args(["export", "x.json"]).edited_timing == "align"


def test_srt_txt_csv_transcripts_round_trip(api, tmp_path):
    """What export writes for each format reads back as the same lines (as Edit Transcript opens them)."""
    segs = [{"speaker": "Ana", "start": 0.0, "end": 1.5, "text": 'Hello, "world": it\'s 10:30',
             "words": [], "flags": ["overlap", "edited"]},
            {"speaker": "Speaker 2", "start": 3723.25, "end": 3725.0, "text": "Olá, então", "words": [], "flags": []}]
    src = tmp_path / "talk.json"
    src.write_text(json.dumps({"segments": segs}), "utf-8")
    plain = [{**s, "flags": []} for s in segs]                    # only CSV and JSON keep the flags

    def back(fmt, **kw):
        path = actions.export(str(src), out=str(tmp_path / "out"), formats=[fmt], **kw)["files"][0]
        return actions.get_transcript(path, full=True)["segments"]

    assert back("csv") == segs
    assert back("srt") == plain
    assert back("txt") == plain
    assert back("txt", txt_timestamps=False) == [{**s, "start": 0.0, "end": 0.0, "flags": ["no_timing"]} for s in segs]
    assert back("txt", txt_timestamps=False, show_speakers=False)[1]["speaker"] == "Speaker"   # no name written

    # the numbered view, and exporting on from such a file (next to it by default)
    srt = tmp_path / "out" / "talk.srt"
    t = actions.get_transcript(str(srt))
    assert t["speakers"] == ["Ana", "Speaker 2"] and t["lines"][1] == {
        "line": 1, "start": 3723.25, "end": 3725.0, "speaker": "Speaker 2", "text": "Olá, então", "flags": []}
    made = actions.export(str(srt), formats=["csv"], filename="again")["files"][0]
    assert Path(made) == tmp_path / "out" / "again.csv" and actions.get_transcript(made, full=True)["segments"] == plain
    before = srt.read_bytes()
    with pytest.raises(actions.ActionError, match="would write over talk.srt"):
        actions.export(str(srt))                                  # srt -> the same .srt
    assert srt.read_bytes() == before
    (tmp_path / "empty.txt").write_text(" \n", "utf-8")
    with pytest.raises(actions.ActionError, match="no transcript lines found in empty.txt"):
        actions.get_transcript(str(tmp_path / "empty.txt"))


def test_edit_lines_on_a_transcript_file(api, tmp_path):
    """A file is edited into a new .json, never over itself; a .json edit_lines wrote is edited in place."""
    words = [{"word": "hello", "start": 0.0, "end": 0.5}, {"word": "there", "start": 0.5, "end": 1.0}]
    src = tmp_path / "talk.json"
    src.write_text(json.dumps({"segments": [
        {"speaker": "Host", "start": 0.0, "end": 1.0, "text": "hello there", "words": words, "flags": ["overlap"]},
        {"speaker": "Ana", "start": 2.0, "end": 3.0, "text": "hi", "words": [], "flags": []}]}), "utf-8")
    actions.export(str(src), formats=["srt", "txt", "csv"])
    for fmt in ("json", "srt", "txt", "csv"):
        f = tmp_path / f"talk.{fmt}"
        before, copy = f.read_bytes(), tmp_path / "edits" / f"{fmt}.json"
        assert actions.edit_lines(str(f), line=1, text="hi all", out=str(copy)) == {
            "changed_lines": 1, "lines": 2, "file": str(copy.resolve())}
        assert f.read_bytes() == before
        t = actions.get_transcript(str(copy))
        assert t["speakers"] == ["Ana", "Host"] and t["lines"][1]["text"] == "hi all"
        assert t["lines"][1]["flags"] == ["edited"]                 # as on a job's line
    kept = actions.get_transcript(str(tmp_path / "edits" / "json.json"), full=True)["segments"][0]
    assert kept["words"] == words and kept["flags"] == ["overlap"]

    srt = tmp_path / "talk.srt"
    with pytest.raises(actions.ActionError, match="talk.srt is a transcript file: add out"):
        actions.edit_lines(str(srt), line=0, text="x")
    for same in (src, tmp_path / "TALK.JSON"):                    # paths compare case-insensitively on Windows
        with pytest.raises(actions.ActionError, match="would write over talk.json, the original"):
            actions.edit_lines(str(src), line=0, text="x", out=str(same))
    with pytest.raises(actions.ActionError, match="out must end in .json"):
        actions.edit_lines(str(srt), line=0, text="x", out=str(tmp_path / "talk2.srt"))
    with pytest.raises(actions.ActionError, match="out is only for a transcript file"):
        actions.edit_lines("some-job-id", line=0, text="x", out=str(tmp_path / "x.json"))
    assert not (tmp_path / "talk2.srt").exists() and not (tmp_path / "x.json").exists()

    # the copy carries the mark, so edits in a row go back into it: with out left out, or out the same file
    copy = tmp_path / "edits" / "srt.json"
    assert json.loads(copy.read_text("utf-8"))["edit_lines_copy"] is True
    actions.edit_lines(str(copy), rename_from="Host", rename_to="Presenter")
    actions.edit_lines(str(copy), line=1, delete=True, out=str(copy))
    t = actions.get_transcript(str(copy))
    assert t["speakers"] == ["Presenter"] and len(t["lines"]) == 1
    # and it reads and exports exactly like the same .json without the mark
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps({k: v for k, v in json.loads(copy.read_text("utf-8")).items()
                                 if k != "edit_lines_copy"}), "utf-8")
    assert actions.get_transcript(str(copy)) == actions.get_transcript(str(plain))
    a = actions.export(str(copy), out=str(tmp_path / "a"), filename="t", formats=["srt", "json"])["files"]
    b = actions.export(str(plain), out=str(tmp_path / "b"), filename="t", formats=["srt", "json"])["files"]
    assert [Path(f).read_bytes() for f in a] == [Path(f).read_bytes() for f in b]
    assert "Presenter: hello there" in Path(a[0]).read_text("utf-8")


def test_hand_written_srt_txt_csv_parse_like_the_app():
    # no cue number, "." before the milliseconds, a two-line cue, a cue with no name, junk between cues
    srt = "00:00:01.5 --> 00:00:02,250\nAna: one\ntwo\n\n\n7\n01:00:00,000 --> 01:00:01,000\nno name here\n\nnot a cue\n"
    assert [(s["speaker"], s["start"], s["end"], s["text"]) for s in actions._parse_srt(srt)] == [
        ("Ana", 1.5, 2.25, "one two"), ("Speaker", 3600.0, 3601.0, "no name here")]
    txt = actions._parse_txt("[00:00:01,000 --> 00:00:02,000] Ana: hi\n\n  just text\nBo: named, no time\n")
    assert [(s["speaker"], s["start"], s["text"], s["flags"]) for s in txt] == [
        ("Ana", 1.0, "hi", []), ("Speaker", 0.0, "just text", ["no_timing"]), ("Bo", 0.0, "named, no time", ["no_timing"])]
    # CSV: columns by header name in any order; without a header, in the export's order; blank rows skipped
    by_name = actions._parse_csv('start,end,text,speaker\n1,2.5,"a, ""b""\nc",Ana\n\n,,only text,\n')
    assert by_name == [
        {"speaker": "Ana", "start": 1.0, "end": 2.5, "text": 'a, "b"\nc', "words": [], "flags": []},
        {"speaker": "Speaker", "start": 0.0, "end": 0.0, "text": "only text", "words": [], "flags": []}]
    assert actions._parse_csv("0,1,Ana,hi,overlap| edited\n")[0] == {
        "speaker": "Ana", "start": 0.0, "end": 1.0, "text": "hi", "words": [], "flags": ["overlap", "edited"]}


def test_batch_on_a_folder(api, two_stream_file, tmp_path):
    b = actions.run_batch([str(two_stream_file.parent)], out=str(tmp_path / "out"), formats=["txt"], wait=True)
    assert b["status"] == "done" and b["done"] == b["files"] >= 1
    assert list((tmp_path / "out").rglob("*.txt"))


def test_run_batch_and_get_batch_count_the_same_files(api, two_stream_file, tmp_path):
    """Both count source files. get_batch used to count its items, so two files joined into one
    transcript were 2 for run_batch and 1 for get_batch."""
    f = str(two_stream_file)
    started = actions.run_batch([f, f], out=str(tmp_path), formats=["txt"], per_file=False)
    assert started["files"] == 2
    _until(lambda: actions.get_batch(started["batch_id"])["status"] == "done")
    b = actions.get_batch(started["batch_id"])
    assert (b["files"], b["done"], len(b["items"])) == (2, 2, 1)


def test_cli_has_a_command_per_action_and_parses_lists(api, two_stream_file, capsys):
    parser = cli.build_parser()
    a = parser.parse_args(["transcribe", "a.mov", "b.mov", "--names", "Host,Guest", "--tracks", "1,2", "--wait"])
    assert a.paths == ["a.mov", "b.mov"] and a.names == ["Host", "Guest"] and a.tracks == [1, 2] and a.wait
    assert parser.parse_args(["export", "x.json", "--no-show-speakers"]).show_speakers is False
    assert parser.parse_args(["export", "x.json"]).txt_timestamps is True
    assert parser.parse_args(["export", "x.json", "--no-txt-timestamps"]).txt_timestamps is False
    names = {fn.__name__.replace("_", "-") for fn in actions.ACTIONS}
    assert names <= set(parser._subparsers._group_actions[0].choices)
    # --help prints for the whole CLI and every command (a "%" in a docstring used to crash it)
    assert "get-progress" in parser.format_help()
    for sub in parser._subparsers._group_actions[0].choices.values():
        sub.format_help()

    assert cli.main(["--json", "add-files", str(two_stream_file)]) == 0
    assert json.loads(capsys.readouterr().out)[0]["tracks"][1]["track"] == 2
    assert cli.main(["add-files", str(two_stream_file)]) == 0        # without --json: a line per track
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[1] == "  tracks:" and lines[2].startswith("    - track=1, speaker=") and "[" not in out
    assert lines[3].startswith("    - track=2, speaker=")
    assert cli.main(["edit-lines", "nope"]) == 1                     # refused: nothing to change


def test_languages_i_use_is_read_and_set(api, monkeypatch, tmp_path, capsys):
    from app import modelstore
    monkeypatch.setattr(modelstore, "SETTINGS_FILE", tmp_path / "settings.json")
    assert actions.list_models()["my_languages"] == []                        # default: not set
    assert actions.set_my_languages(["pt", "en"]) == {"my_languages": ["en", "pt"]}
    assert actions.list_models()["my_languages"] == ["en", "pt"]
    with pytest.raises(actions.ActionError, match="unknown language code xx"):
        actions.set_my_languages(["en", "xx"])
    assert actions.list_models()["my_languages"] == ["en", "pt"]              # a refused list changes nothing
    assert actions.set_my_languages([]) == {"my_languages": []}               # empty: no restriction

    parser = cli.build_parser()
    assert parser.parse_args(["set-my-languages", "--languages", "en, pt"]).languages == ["en", "pt"]
    assert parser.parse_args(["set-my-languages"]).languages == []
    assert cli.main(["--json", "set-my-languages", "--languages", "nl"]) == 0
    assert json.loads(capsys.readouterr().out) == {"my_languages": ["nl"]}
    assert cli.main(["set-my-languages", "--languages", "xx"]) == 1           # refused, with the reason
    assert "unknown language code xx" in capsys.readouterr().err
    assert cli.main(["set-my-languages"]) == 0 and actions.list_models()["my_languages"] == []


def test_portuguese_settings_are_read_and_set(api, monkeypatch, tmp_path, capsys):
    from app import modelstore
    monkeypatch.setattr(modelstore, "SETTINGS_FILE", tmp_path / "settings.json")
    m = actions.list_models()
    assert m["auto_portuguese"] is None and m["pt_punctuation"] is False       # defaults
    assert actions.set_auto_portuguese("pt-PT") == {"auto_portuguese": "pt-PT"}
    assert actions.set_pt_punctuation(True) == {"pt_punctuation": True}
    m = actions.list_models()
    assert m["auto_portuguese"] == "pt-PT" and m["pt_punctuation"] is True
    with pytest.raises(actions.ActionError, match='variant is "pt-PT" or "pt-BR"'):
        actions.set_auto_portuguese("pt")
    assert actions.list_models()["auto_portuguese"] == "pt-PT"                # a refused value changes nothing

    assert cli.main(["--json", "set-auto-portuguese", "--variant", "pt-BR"]) == 0
    assert json.loads(capsys.readouterr().out) == {"auto_portuguese": "pt-BR"}
    assert cli.main(["set-auto-portuguese"]) == 0 and actions.list_models()["auto_portuguese"] is None
    assert cli.main(["set-pt-punctuation", "--on"]) == 0 and actions.list_models()["pt_punctuation"] is True
    assert cli.main(["set-pt-punctuation"]) == 0 and actions.list_models()["pt_punctuation"] is False


def test_model_downloads_are_cancelled_and_models_removed(api, monkeypatch, tmp_path):
    from app import modelstore
    monkeypatch.setattr(modelstore, "SETTINGS_FILE", tmp_path / "settings.json")   # speech model in use: large-v3
    monkeypatch.setattr(modelstore, "is_installed", lambda e: e.id == "whisper:large-v3")
    monkeypatch.setattr(modelstore, "has_partial", lambda e: e.id == "align:nl")
    monkeypatch.setattr(modelstore, "disk_free", lambda: 5_000_000_000)
    removed = []
    monkeypatch.setattr(modelstore, "remove", removed.append)                      # nothing leaves the disk here
    d = modelstore.Download("align:nl", status="downloading")
    monkeypatch.setitem(modelstore.downloads.downloads, "align:nl", d)

    m = actions.list_models(kind="all")
    assert m["disk_free_mb"] == 5000 and m["downloaded"].startswith("1 of ")
    assert [x["id"] for x in m["models"] if x["partial"]] == ["align:nl"]
    assert not any(x["partial"] for x in actions.list_models(installed_only=True)["models"])

    with pytest.raises(actions.ActionError, match="409: cancel the download first"):
        actions.remove_model("align:nl")
    assert actions.cancel_download("align:nl") == {"canceling": True} and d._cancel.is_set()
    d.status = "canceled"                                         # what the download thread does next
    assert actions.cancel_download("align:nl") == {"canceling": False}             # nothing running any more
    assert actions.remove_model("align:nl") == {"removed": True} and [e.id for e in removed] == ["align:nl"]
    with pytest.raises(actions.ActionError, match="409: that's the speech model in use"):
        actions.remove_model("whisper:large-v3")
    # an id outside the catalog names nothing to delete, whatever it looks like
    for bad in ("whisper:nope", "whisper:..", "align:zz", "nope"):
        for fn in (actions.cancel_download, actions.remove_model):
            with pytest.raises(actions.ActionError, match="404: unknown model"):
                fn(bad)
    assert len(removed) == 1
    assert cli.main(["remove-model", "align:nl"]) == 0 and len(removed) == 2
    assert cli.main(["cancel-download", "align:nl"]) == 0


def test_benchmark_runs_and_reports(api, monkeypatch, capsys):
    from app import modelstore
    row = {"test": "English", "model": "small", "load_s": 1.2, "transcribe_s": 0.5, "audio_s": 11.0,
           "speed_x": 22.0, "match": 1.0, "heard": "Let's move the quarterly review"}
    monkeypatch.setattr(routes.api.benchmark, "run", lambda progress: [row])       # no GPU, no voice
    monkeypatch.setattr(modelstore, "is_installed", lambda e: False)
    with pytest.raises(actions.ActionError, match="409: download a speech model first"):
        actions.run_benchmark()
    monkeypatch.setattr(modelstore, "is_installed", lambda e: True)

    done = actions.run_benchmark(wait=True)
    assert done["status"] == "done" and done["results"] == [row] and "quarterly review" in done["expected"]["English"]
    job_id = actions.run_benchmark()["job_id"]                    # not waiting: follow it, then read it
    deadline = time.time() + 5
    while actions.get_progress(job_id)["status"] != "done" and time.time() < deadline:
        pass
    assert actions.get_benchmark(job_id)["results"] == [row]
    with pytest.raises(actions.ActionError, match="404: unknown job"):
        actions.get_benchmark("nope")
    assert cli.main(["--json", "run-benchmark", "--wait"]) == 0
    assert json.loads(capsys.readouterr().out)["results"][0]["speed_x"] == 22.0


def test_mcp_tools_match_the_actions_and_their_marks():
    import asyncio
    import mcp_server
    tools = {t.name: t.annotations for t in asyncio.run(mcp_server.server.list_tools())}
    assert set(tools) == {fn.__name__ for fn in actions.ACTIONS}
    assert {n for n, a in tools.items() if a.destructive_hint} == {"remove_model", "clear_cache", "shutdown"}
    assert tools["get_benchmark"].read_only_hint and not tools["run_benchmark"].read_only_hint
    assert not tools["cancel_download"].read_only_hint and not tools["set_auto_portuguese"].read_only_hint


def test_save_result_needs_a_finished_job(api):
    assert api.put("/api/result/unknown", json={"segments": []}).status_code == 404


def test_a_waiting_job_says_so_and_takes_an_answer(api, two_stream_file, monkeypatch, tmp_path):
    """A run that stops to ask doesn't hang a waiting CLI / MCP call: it returns the question and
    how to answer; `answer` lets it carry on."""
    from app.transcribe import ASK_CHOICES, _asker

    def asking_stub(_wav, **_kw):
        _asker.fn({"language": "nn", "text": "big download?", "choices": ASK_CHOICES})
        return _stub_transcribe(_wav)

    monkeypatch.setattr(routes.api, "transcribe_track", asking_stub)
    monkeypatch.setattr(routes.api, "unload_models", lambda: None)
    job = actions.transcribe([str(two_stream_file)], tracks=[1], wait=True)
    assert job["status"] == "waiting for your answer" and job["question"]["text"] == "big download?"
    assert set(job["question"]["choices"]) == {"download", "skip", "language"} and "answer" in job["next"]
    p = actions.get_progress(job["job_id"])
    assert p["status"] == "running" and p["question"] and p["message"].startswith("waiting for your answer")

    with pytest.raises(actions.ActionError, match="unknown language code"):
        actions.answer(job["job_id"], "language", "xx")
    assert actions.answer(job["job_id"], "skip") == {"answered": True}
    deadline = time.time() + 5
    while actions.get_progress(job["job_id"])["status"] != "done" and time.time() < deadline:
        pass
    assert "question" not in actions.get_progress(job["job_id"])

    # a batch: the same, through the same action
    b = actions.run_batch([str(two_stream_file)], out=str(tmp_path / "out"), formats=["txt"], wait=True)
    assert b["status"] == "running" and b["question"]["item"] == two_stream_file.name
    batch_id = next(iter(api.app.state.batches._batches))
    assert actions.answer(batch_id, "download") == {"answered": True}
    deadline = time.time() + 5
    while actions.get_batch(batch_id)["status"] != "done" and time.time() < deadline:
        pass
    assert actions.get_batch(batch_id)["done"] == 1
    assert "answer" in {a.__name__ for a in actions.ACTIONS}


def test_a_job_pauses_resumes_and_cancels(api, two_stream_file, monkeypatch):
    gate = _held(monkeypatch)
    job_id = actions.transcribe([str(two_stream_file)], tracks=[1])["job_id"]
    assert actions.pause(job_id) == {"pausing": True}
    assert actions.get_progress(job_id)["paused"]                    # "pausing: ..." then "GPU released"
    assert actions.resume(job_id) == {"resuming": True}
    assert "paused" not in actions.get_progress(job_id)
    assert actions.cancel(job_id) == {"canceling": True}
    gate.set()
    _until(lambda: actions.get_progress(job_id)["status"] == "canceled")
    for act in (actions.pause, actions.resume, actions.cancel):
        with pytest.raises(actions.ActionError, match="404"):
            act("nope")


def test_transcribe_wait_fails_when_its_job_is_canceled(api, two_stream_file, monkeypatch, capsys):
    gate, rc = _held(monkeypatch), []
    run = threading.Thread(target=lambda: rc.append(
        cli.main(["transcribe", str(two_stream_file), "--tracks", "1", "--wait"])))
    run.start()
    jobs = api.app.state.jobs._jobs
    _until(lambda: jobs)
    assert actions.cancel(next(iter(jobs))) == {"canceling": True}
    gate.set()
    run.join(10)
    assert rc == [1] and "error: job canceled" in capsys.readouterr().err


def test_a_batch_is_controlled_by_project_and_file_number(api, two_stream_file, monkeypatch, tmp_path):
    gate, f = _held(monkeypatch), str(two_stream_file)
    done = lambda bid, status="done": _until(lambda: actions.get_batch(bid)["status"] == status)
    rows = lambda bid: [(i["project"], i.get("file"), i["status"]) for i in actions.get_batch(bid)["items"]]

    # one project, a transcript per file: cancel file 2 while file 1 runs, pause and resume the batch
    bid = actions.run_batch([f, f], out=str(tmp_path / "a"), formats=["txt"])["batch_id"]
    assert actions.get_batch(bid)["batch_id"] == bid
    assert actions.cancel(bid, file=2) == {"canceling": True}
    assert actions.pause(bid) == {"pausing": True} and actions.get_batch(bid)["paused"]
    assert actions.resume(bid) == {"resuming": True} and "paused" not in actions.get_batch(bid)
    gate.set()
    done(bid)
    assert rows(bid) == [(1, 1, "done"), (1, 2, "canceled")]

    # that file again, then the whole (finished) project; with nothing failed, a plain retry is refused
    assert actions.retry_batch(bid, file=2)["batch_id"] == bid
    done(bid)
    assert rows(bid) == [(1, 1, "done"), (1, 2, "done")]
    with pytest.raises(actions.ActionError, match="nothing failed or canceled"):
        actions.retry_batch(bid)
    with pytest.raises(actions.ActionError, match="file goes with one project"):
        actions.retry_batch(bid, projects=[1, 2], file=1)
    with pytest.raises(actions.ActionError, match="404"):
        actions.cancel(bid, project=9)
    actions.retry_batch(bid, projects=[1])
    done(bid)

    # one merged project: cancel it alone (its only project canceled, so the batch ends "canceled"),
    # retry what was canceled
    gate.clear()
    bid = actions.run_batch([f], out=str(tmp_path / "b"), formats=["txt"], per_file=False)["batch_id"]
    assert actions.cancel(bid, project=1) == {"canceling": True}
    gate.set()
    done(bid, "canceled")
    assert rows(bid) == [(1, None, "canceled")]
    actions.retry_batch(bid)
    done(bid)
    assert rows(bid) == [(1, None, "done")]

    # the whole batch
    gate.clear()
    bid = actions.run_batch([f], out=str(tmp_path / "c"), formats=["txt"])["batch_id"]
    assert actions.cancel(bid) == {"canceling": True}
    gate.set()
    done(bid, "canceled")

    parser = cli.build_parser()
    a = parser.parse_args(["cancel", "x", "--project", "2", "--file", "3"])
    assert (a.run_id, a.project, a.file) == ("x", 2, 3)
    a = parser.parse_args(["retry-batch", "x", "--projects", "1,2"])
    assert (a.batch_id, a.projects, a.file) == ("x", [1, 2], None)
    assert parser.parse_args(["pause", "x"]).run_id == parser.parse_args(["resume", "x"]).run_id == "x"


def test_batch_projects_keep_their_own_settings_and_open_by_number(api, two_stream_file, tmp_path):
    f = str(two_stream_file)
    b = actions.run_batch(
        [f], out=str(tmp_path), formats=["txt"], name="One", names=["Host", "Guest"], tracks=[1],
        txt_timestamps=False, wait=True,
        projects=[{"name": "Two", "paths": [f, f], "per_file": False, "names": ["Ana", "Bo"], "tracks": None,
                   "formats": ["srt"], "show_speakers": False},
                  {"paths": [f]}])
    bid = b["batch_id"]
    assert b["status"] == "done" and b["files"] == b["done"] == 4    # source files: "Two" joins two into one item
    assert [(i["project"], i.get("file"), i["name"]) for i in b["items"]] == [
        (1, 1, two_stream_file.name), (2, None, "Two"), (3, 1, two_stream_file.name)]
    written = lambda folder, ext: next((tmp_path / folder).glob(f"*.{ext}")).read_text("utf-8")
    assert written("One", "txt") == "Host: hello"                   # the batch's own settings
    assert "hello" in written("Two", "srt") and "Ana" not in written("Two", "srt")   # its own
    assert written("Project 3", "txt") == "Host: hello"             # nothing of its own: the batch's

    # read, edit and export a batch result by its numbers
    assert actions.get_transcript(bid, file=1)["speakers"] == ["Host"]
    assert actions.get_transcript(bid, project=2)["speakers"] == ["Ana", "Bo"]
    with pytest.raises(actions.ActionError, match="one transcript per file"):
        actions.get_transcript(bid, project=1)
    with pytest.raises(actions.ActionError, match="404"):
        actions.get_transcript(bid, project=2, file=1)
    actions.edit_lines(bid, project=2, rename_from="Ana", rename_to="Anna")
    actions.edit_lines(bid, file=1, line=0, text="fixed")
    assert actions.get_transcript(bid, project=2)["speakers"] == ["Anna", "Bo"]
    out = actions.export(bid, out=str(tmp_path / "x"), formats=["txt"], project=3, file=1)
    assert Path(out["files"][0]).read_text("utf-8").endswith("Host: hello")
    out = actions.export(bid, out=str(tmp_path / "x"), formats=["txt"], file=1, txt_timestamps=False, filename="cut")
    assert Path(out["files"][0]).read_text("utf-8") == "Host: fixed"
    assert api.put("/api/batch/nope/project/0/result", json={"segments": []}).status_code == 404

    # only `projects`, and what is refused
    b = actions.run_batch([], out=str(tmp_path / "only"), formats=["txt"], projects=[{"paths": [f]}], wait=True)
    assert b["status"] == "done" and list((tmp_path / "only" / "Project 1").glob("*.txt"))
    for bad in ([{"paths": [f], "nmae": "x"}], [{"name": "no files"}], ["a.mov"]):
        with pytest.raises(actions.ActionError, match='project 2: give an object with "paths"'):
            actions.run_batch([f], projects=bad)
    with pytest.raises(actions.ActionError, match="nothing to run"):
        actions.run_batch([])


def test_cli_takes_projects_as_json_and_numbers_for_batch_results(tmp_path):
    parser = cli.build_parser()
    a = parser.parse_args(["run-batch", "D:\\rec", "--out", "D:\\subs", "--formats", "srt,txt", "--wait"])
    assert a.paths == ["D:\\rec"] and a.formats == ["srt", "txt"] and a.wait and a.projects is None
    a = parser.parse_args(["run-batch", "--projects", '[{"paths": ["a.mov"], "names": ["Host"]}]',
                           "--names", "A,B", "--tracks", "2", "--max-chars", "30", "--no-show-speakers"])
    assert a.paths == [] and a.projects == [{"paths": ["a.mov"], "names": ["Host"]}]
    assert (a.names, a.tracks, a.max_chars, a.show_speakers) == (["A", "B"], [2], 30, False)
    saved = tmp_path / "batch.json"
    saved.write_text('[{"name": "Ep 1", "paths": ["a.mov"]}]', "utf-8-sig")       # a file, with or without a BOM
    assert parser.parse_args(["run-batch", "--projects", str(saved)]).projects == [{"name": "Ep 1", "paths": ["a.mov"]}]
    a = parser.parse_args(["transcript", "x", "--project", "2", "--file", "3"])
    assert (a.source, a.project, a.file) == ("x", 2, 3)
    assert parser.parse_args(["export", "x", "--file", "3"]).file == 3
    assert parser.parse_args(["edit-lines", "x", "--project", "2", "--line", "0", "--delete"]).project == 2


def test_merge_options_reach_a_job_and_each_batch_project(api, two_stream_file, tmp_path, monkeypatch):
    """Both tracks say "hello" at the same moment, which the merge flags as bleed unless told otherwise."""
    f = str(two_stream_file)
    flags = lambda t: [ln["flags"] for ln in t["lines"]]
    run = lambda **kw: flags(actions.get_transcript(actions.transcribe([f], wait=True, **kw)["job_id"]))
    assert all("bleed_suspected" in fl for fl in run())
    assert run(detect_conflicts=False) == [[], []]
    assert run(bleed_window_seconds=-1.0) == run(bleed_similarity=1.1) == [[], ["backchannel"]]   # no bleed: a backchannel
    assert run(bleed_similarity=1.1, backchannel_render="skip") == [[]]                           # ... left out
    assert run(bleed_similarity=1.1, backchannel_max_words=1) == [[], []]                         # too long for one
    assert run(bleed_similarity=1.1, backchannel_max_words=1, overlap_min_seconds=0.5) == [["overlap"], ["overlap"]]

    sent = []
    with monkeypatch.context() as m:           # its own scope: undoing the fixture's patches would reach a real server
        m.setattr(routes.api, "run_merge_pipeline",
                  lambda *a, options, **k: sent.append(options) or routes.api.MergedOutput(segments=[]))
        actions.transcribe([f], wait=True, backchannel_render="skip", backchannel_max_words=2,
                           overlap_min_seconds=3.5, bleed_prefer_ratio=4.0)
    o = sent[0]
    assert (o.backchannel_render.value, o.backchannel_max_words, o.overlap_min_seconds, o.bleed_prefer_ratio,
            o.detect_conflicts, o.bleed_window_seconds, o.bleed_similarity) == ("skip", 2, 3.5, 4.0, True, 1.0, 0.7)
    with pytest.raises(actions.ActionError, match="backchannel_render is one of"):
        actions.transcribe([f], backchannel_render="loud")

    b = actions.run_batch([f], out=str(tmp_path), formats=["json"], per_file=False, detect_conflicts=False,
                          projects=[{"paths": [f], "per_file": False, "detect_conflicts": True}], wait=True)
    assert flags(actions.get_transcript(b["batch_id"], project=1)) == [[], []]
    assert all("bleed_suspected" in fl for fl in flags(actions.get_transcript(b["batch_id"], project=2)))

    a = cli.build_parser().parse_args(["transcribe", "a.mov", "--no-detect-conflicts", "--backchannel-render", "skip",
                                       "--backchannel-max-words", "2", "--overlap-min-seconds", "3",
                                       "--bleed-window-seconds", "0.5", "--bleed-similarity", "0.9",
                                       "--bleed-prefer-ratio", "3"])
    assert (a.detect_conflicts, a.backchannel_render, a.backchannel_max_words, a.overlap_min_seconds,
            a.bleed_window_seconds, a.bleed_similarity, a.bleed_prefer_ratio) == (False, "skip", 2, 3.0, 0.5, 0.9, 3.0)
    assert cli.build_parser().parse_args(["run-batch", "d", "--backchannel-render", "separate"]).backchannel_render == "separate"


def test_a_batch_carries_every_export_option(api, two_stream_file, tmp_path, monkeypatch):
    """The two newest export options too: for the batch, and for one project."""
    f, sent, real = str(two_stream_file), [], actions._api
    monkeypatch.setattr(actions, "_api", lambda method, path, body=None, **kw: (
        sent.append(body) if path == "/api/batch" else None, real(method, path, body, **kw))[1])
    b = actions.run_batch([f], out=str(tmp_path), formats=["srt"], min_duration=2.0, edited_timing="whole",
                          projects=[{"paths": [f], "edited_timing": "even"}], wait=True)
    assert b["status"] == "done"
    one, two = (p["export"] for p in sent[0]["projects"])
    assert (one["min_duration"], one["edited_timing"]) == (2.0, "whole")
    assert (two["min_duration"], two["edited_timing"]) == (2.0, "even")
    assert set(one) == set(routes.api.ExportOptionsIn.model_fields)             # nothing the route has is left out
    assert set(sent[0]["projects"][0]["options"]) == set(routes.api.MergeOptionsIn.model_fields)
    with pytest.raises(actions.ActionError, match="edited_timing is one of"):
        actions.run_batch([f], projects=[{"paths": [f], "edited_timing": "nope"}])
    a = cli.build_parser().parse_args(["run-batch", "d", "--min-duration", "2", "--edited-timing", "whole"])
    assert (a.min_duration, a.edited_timing) == (2.0, "whole")


def test_word_timing_option_reaches_a_job_and_each_batch_project(api, two_stream_file, tmp_path, monkeypatch):
    """#39: on unless asked otherwise; off is passed down and the result says so."""
    f, seen = str(two_stream_file), []

    def stub(_wav, word_timing=True, **_kw):
        seen.append(word_timing)
        return {**_stub_transcribe(_wav), "language": "en", **({} if word_timing else {"word_timing": False})}

    with monkeypatch.context() as m:           # its own scope: undoing the fixture's patches would reach a real server
        m.setattr(routes.api, "transcribe_track", stub)
        done = actions.transcribe([f], wait=True)
        assert seen == [True, True] and "language" not in done          # default: as before
        seen.clear()
        done = actions.transcribe([f], wait=True, word_timing=False)
        assert seen == [False, False] and done["language"] == "English (no word timing)"
        assert actions.get_progress(done["job_id"])["message"].endswith("English (no word timing)")
        full = actions.get_transcript(done["job_id"], full=True)
        assert [t["word_timing"] for t in full["languages"]] == [False, False]

        seen.clear()                           # a batch: the call's setting, and a project's own over it
        b = actions.run_batch([f], out=str(tmp_path), formats=["json"], per_file=False, word_timing=False,
                              projects=[{"paths": [f], "per_file": False, "word_timing": True},
                                        {"paths": [f], "per_file": False}], wait=True)
        assert seen == [False, False, True, True, False, False]
        assert [i["message"].endswith("(no word timing)") for i in b["items"]] == [True, False, True]
        seen.clear()
        actions.run_batch([f], out=str(tmp_path), formats=["json"], wait=True)
        assert seen == [True, True]

    parser = cli.build_parser()
    assert parser.parse_args(["transcribe", "a.mov"]).word_timing is True
    assert parser.parse_args(["transcribe", "a.mov", "--no-word-timing"]).word_timing is False
    assert parser.parse_args(["run-batch", "d"]).word_timing is True
    assert parser.parse_args(["run-batch", "d", "--no-word-timing"]).word_timing is False


def _mcp(name: str, args: dict):
    """A tool called the way an MCP client calls it; a refusal comes back as ToolError."""
    import asyncio
    import mcp_server
    return asyncio.run(mcp_server.server.call_tool(name, args))


def test_a_batch_id_alone_says_what_to_add(api, two_stream_file, tmp_path, capsys):
    from mcp.server.mcpserver.exceptions import ToolError
    f = str(two_stream_file)
    b = actions.run_batch([f, f], out=str(tmp_path), formats=["txt"], name="Talks", wait=True,
                          projects=[{"paths": [f], "name": "Joined", "per_file": False}])
    hint = (f'is a batch id: add project, and file for a project with one transcript per file '
            f'(numbers as get_batch shows them). Its projects: project 1 "Talks" (files: 1 two_stream.mkv, '
            f'2 two_stream.mkv); project 2 "Joined"')
    for act in (lambda: actions.get_transcript(b["batch_id"]), lambda: actions.export(b["batch_id"], out=str(tmp_path)),
                lambda: actions.edit_lines(b["batch_id"], line=0, delete=True)):
        with pytest.raises(actions.ActionError) as e:
            act()
        assert hint in str(e.value)
    with pytest.raises(ToolError, match="is a batch id"):
        _mcp("get_transcript", {"source": b["batch_id"]})
    assert cli.main(["get-transcript", b["batch_id"]]) == 1 and hint in capsys.readouterr().err
    with pytest.raises(actions.ActionError, match="404: unknown job"):   # neither a job nor a batch
        actions.get_transcript("nope")
    assert actions.get_transcript(b["batch_id"], project=2)["lines"]


def test_a_wrong_backchannel_render_names_the_valid_ones(api, two_stream_file, capsys):
    from mcp.server.mcpserver.exceptions import ToolError
    f, msg = str(two_stream_file), "backchannel_render is one of parenthetical, separate, skip"
    with pytest.raises(actions.ActionError, match=msg):
        actions.transcribe([f], backchannel_render="inline")
    with pytest.raises(actions.ActionError, match=msg):
        actions.run_batch([f], backchannel_render="inline")
    with pytest.raises(actions.ActionError, match=msg):
        actions.run_batch([f], projects=[{"paths": [f], "backchannel_render": "inline"}])
    for tool, args in (("transcribe", {"paths": [f], "backchannel_render": "inline"}),
                       ("run_batch", {"paths": [f], "backchannel_render": "inline"})):
        with pytest.raises(ToolError, match=msg):
            _mcp(tool, args)
    assert cli.main(["transcribe", f, "--backchannel-render", "inline"]) == 1 and msg in capsys.readouterr().err
    assert not api.app.state.batches._batches and not api.app.state.jobs._jobs   # nothing was started


def test_retry_refuses_a_project_still_running_or_queued(api, two_stream_file, monkeypatch, tmp_path):
    from mcp.server.mcpserver.exceptions import ToolError
    gate, f = _held(monkeypatch), str(two_stream_file)
    bid = actions.run_batch([f], out=str(tmp_path), formats=["txt"], per_file=False, name="First",
                            projects=[{"paths": [f], "per_file": False, "name": "Second"}])["batch_id"]
    _until(lambda: actions.get_batch(bid)["items"][0]["status"] == "running")
    with pytest.raises(actions.ActionError, match='409: project 1 "First" is still running: wait for it'):
        actions.retry_batch(bid, projects=[1])
    with pytest.raises(ToolError, match='project 2 "Second" is still queued'):
        _mcp("retry_batch", {"batch_id": bid, "projects": [2]})
    r = api.post(f"/api/batch/{bid}/retry", json={"projects": [0, 1]})       # the app's route: the same
    assert r.status_code == 409 and "still running" in r.json()["detail"]
    gate.set()
    _until(lambda: actions.get_batch(bid)["status"] == "done")
    assert actions.retry_batch(bid, projects=[1, 2])["batch_id"] == bid       # finished: runs again
    _until(lambda: actions.get_batch(bid)["status"] == "done")


def test_a_canceled_batch_leaves_no_file_queued(api, two_stream_file, monkeypatch, tmp_path):
    """The files of a project that never started end canceled too, not queued."""
    turns, f = threading.Semaphore(0), str(two_stream_file)

    def stub(_wav, **_kw):                     # each transcription waits for its own turn
        turns.acquire(timeout=5)
        return _stub_transcribe(_wav)

    with monkeypatch.context() as m:
        m.setattr(routes.api, "transcribe_track", stub)
        m.setattr(routes.api, "unload_models", lambda: None)
        rows = lambda bid: [(i["project"], i["status"]) for i in actions.get_batch(bid)["items"]]

        bid = actions.run_batch([f], out=str(tmp_path / "a"), formats=["txt"], tracks=[1],
                                projects=[{"paths": [f, f]}])["batch_id"]   # project 2: a transcript per file
        _until(lambda: rows(bid)[0][1] == "running")
        actions.cancel(bid)
        turns.release()
        _until(lambda: actions.get_batch(bid)["status"] == "canceled")
        assert rows(bid) == [(1, "canceled"), (2, "canceled"), (2, "canceled")]


def test_a_run_canceled_part_way_says_how_far_it_got(api, two_stream_file, monkeypatch, tmp_path):
    """get_progress and get_batch (so the CLI and MCP) carry the canceled message: how many tracks
    were transcribed, and kept for a retry. A job canceled before any track finished stays "canceled"."""
    from app import transcribe
    gate, calls, hold = threading.Event(), [], [2]

    def stub(_wav, **_kw):                         # call `hold` waits for the test, then a chunk
        calls.append(_wav)                         # report (the job's checkpoint) comes in
        if len(calls) == hold[0]:
            gate.wait(5)
            if transcribe._chunk.fn:
                transcribe._chunk.fn(0.5)
        return _stub_transcribe(_wav)

    monkeypatch.setattr(routes.api, "transcribe_track", stub)
    monkeypatch.setattr(routes.api, "unload_models", lambda: None)
    kept = "canceled · {} of 2 tracks transcribed, kept for a retry"

    # a job, canceled during its second track: that track stops at the next chunk
    job_id = actions.transcribe([str(two_stream_file)], tracks=[1, 2])["job_id"]
    _until(lambda: len(calls) == 2)
    actions.cancel(job_id)
    gate.set()
    _until(lambda: actions.get_progress(job_id)["status"] == "canceled")
    assert actions.get_progress(job_id)["message"] == kept.format(1)

    # canceled during its first track: nothing finished
    gate.clear()
    calls.clear()
    hold[0] = 1
    job_id = actions.transcribe([str(two_stream_file)], tracks=[1, 2])["job_id"]
    _until(lambda: len(calls) == 1)
    actions.cancel(job_id)
    gate.set()
    _until(lambda: actions.get_progress(job_id)["status"] == "canceled")
    assert actions.get_progress(job_id)["message"] == "canceled"

    # a batch file, canceled during its last track: that track stops at the next chunk, nothing is written
    gate.clear()
    calls.clear()
    hold[0] = 2
    bid = actions.run_batch([str(two_stream_file)], out=str(tmp_path), formats=["txt"])["batch_id"]
    _until(lambda: len(calls) == 2)
    actions.cancel(bid, file=1)
    gate.set()
    _until(lambda: actions.get_batch(bid)["status"] == "canceled")      # its only file was canceled
    item = actions.get_batch(bid)["items"][0]
    assert (item["status"], item["message"]) == ("canceled", kept.format(1))


def test_a_plain_retry_reruns_the_canceled_or_failed_files_of_finished_projects(api, two_stream_file, monkeypatch,
                                                                                 tmp_path):
    """One transcript per file: file 1 canceled (or failed), file 2 done, so the project and the batch
    finished "done". A plain retry_batch runs file 1 again and leaves file 2 alone."""
    f, calls, gate, fail = str(two_stream_file), [], threading.Event(), []

    def stub(_wav, **_kw):
        calls.append(_wav)
        gate.wait(5)
        if fail and len(calls) == 1:
            raise RuntimeError("boom")
        return _stub_transcribe(_wav)

    monkeypatch.setattr(routes.api, "transcribe_track", stub)
    monkeypatch.setattr(routes.api, "unload_models", lambda: None)
    done = lambda bid: _until(lambda: actions.get_batch(bid)["status"] == "done")
    rows = lambda bid: [(i.get("file"), i["status"]) for i in actions.get_batch(bid)["items"]]

    bid = actions.run_batch([f, f], out=str(tmp_path / "a"), formats=["txt"], tracks=[1])["batch_id"]
    _until(lambda: calls)
    actions.cancel(bid, file=1)                    # while file 1 runs
    gate.set()
    done(bid)
    assert rows(bid) == [(1, "canceled"), (2, "done")] and len(calls) == 2
    actions.retry_batch(bid)                       # no projects, no file
    done(bid)
    assert rows(bid) == [(1, "done"), (2, "done")] and len(calls) == 3     # only file 1 ran again
    with pytest.raises(actions.ActionError, match="400: nothing failed or canceled to retry"):
        actions.retry_batch(bid)

    calls.clear()
    fail.append(True)                              # file 1 fails, file 2 is done
    bid = actions.run_batch([f, f], out=str(tmp_path / "b"), formats=["txt"], tracks=[1])["batch_id"]
    done(bid)
    assert rows(bid) == [(1, "error"), (2, "done")]
    fail.clear()
    actions.retry_batch(bid)
    done(bid)
    assert rows(bid) == [(1, "done"), (2, "done")] and len(calls) == 3


def test_get_batch_shows_paused_on_the_item_in_progress(api, two_stream_file, monkeypatch, tmp_path):
    gate = _held(monkeypatch)
    bid = actions.run_batch([str(two_stream_file)], out=str(tmp_path), formats=["txt"])["batch_id"]
    actions.pause(bid)                             # lands after the track that is running
    gate.set()
    _until(lambda: actions.get_batch(bid).get("paused") == "GPU released")
    assert actions.get_batch(bid)["items"][0]["message"] == "paused"
    actions.resume(bid)
    _until(lambda: actions.get_batch(bid)["status"] == "done")
    assert actions.get_batch(bid)["items"][0]["message"] != "paused"


def test_an_ended_run_shows_no_pause_and_the_wording_fits_every_run(api, two_stream_file, monkeypatch, tmp_path):
    """Paused, then canceled before the pause landed: canceled, and no "pausing" left behind, in
    get_progress / get_batch (CLI and MCP) and in the routes the app reads."""
    gate = _held(monkeypatch)
    job_id = actions.transcribe([str(two_stream_file)], tracks=[1])["job_id"]
    actions.pause(job_id)
    assert actions.get_progress(job_id)["paused"] == ("pausing: the track being transcribed (Label Track: "
                                                      "the speaker, Benchmark: the model) finishes first")
    actions.cancel(job_id)
    gate.set()
    _until(lambda: actions.get_progress(job_id)["status"] == "canceled")
    assert "paused" not in actions.get_progress(job_id)
    assert api.get(f"/api/progress/{job_id}").json()["paused"] is False

    gate.clear()
    bid = actions.run_batch([str(two_stream_file)], out=str(tmp_path), formats=["txt"])["batch_id"]
    actions.pause(bid)
    assert actions.get_batch(bid)["paused"] == "pausing: the track being transcribed finishes first"
    actions.cancel(bid)
    gate.set()
    _until(lambda: actions.get_batch(bid)["status"] == "canceled")
    assert "paused" not in actions.get_batch(bid)
    assert api.get(f"/api/batch/{bid}").json()["paused"] is False
