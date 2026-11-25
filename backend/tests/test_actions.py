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


def test_batch_projects_keep_their_own_settings_and_open_by_number(api, two_stream_file, tmp_path):
    f = str(two_stream_file)
    b = actions.run_batch(
        [f], out=str(tmp_path), formats=["txt"], name="One", names=["Host", "Guest"], tracks=[1],
        txt_timestamps=False, wait=True,
        projects=[{"name": "Two", "paths": [f, f], "per_file": False, "names": ["Ana", "Bo"], "tracks": None,
                   "formats": ["srt"], "show_speakers": False},
                  {"paths": [f]}])
    bid = b["batch_id"]
    assert b["status"] == "done" and b["files"] == b["done"] == 3
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
    with pytest.raises(actions.ActionError, match="422"):
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
