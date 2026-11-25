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
    _until(lambda: actions.get_progress(job_id)["status"] == "cancelled")
    for act in (actions.pause, actions.resume, actions.cancel):
        with pytest.raises(actions.ActionError, match="404"):
            act("nope")


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
    with pytest.raises(actions.ActionError, match="no failed or canceled projects"):
        actions.retry_batch(bid)
    with pytest.raises(actions.ActionError, match="file goes with one project"):
        actions.retry_batch(bid, projects=[1, 2], file=1)
    with pytest.raises(actions.ActionError, match="404"):
        actions.cancel(bid, project=9)
    actions.retry_batch(bid, projects=[1])
    done(bid)

    # one merged project: cancel it alone (the batch itself still ends "done"), retry what was canceled
    gate.clear()
    bid = actions.run_batch([f], out=str(tmp_path / "b"), formats=["txt"], per_file=False)["batch_id"]
    assert actions.cancel(bid, project=1) == {"canceling": True}
    gate.set()
    done(bid)
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
