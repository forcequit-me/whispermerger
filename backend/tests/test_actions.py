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
    names = {fn.__name__.replace("_", "-") for fn in actions.ACTIONS}
    assert names <= set(parser._subparsers._group_actions[0].choices)

    assert cli.main(["--json", "add-files", str(two_stream_file)]) == 0
    assert json.loads(capsys.readouterr().out)[0]["tracks"][1]["track"] == 2
    assert cli.main(["edit-lines", "nope"]) == 1                     # refused: nothing to change


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
