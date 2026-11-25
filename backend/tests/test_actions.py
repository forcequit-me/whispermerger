"""CLI + MCP actions (#36), run against the real app through FastAPI's TestClient (no port, no GPU)."""

import json
import sys
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
