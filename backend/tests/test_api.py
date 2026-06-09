"""API tests via FastAPI TestClient. Files are registered by path (/add-path); the GUI
picker (/browse) is tested with a monkeypatched picker. transcribe is stubbed (no GPU)."""

import time

import pytest
from fastapi.testclient import TestClient

from app import routes
from app.main import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def _stub_transcribe(_wav, **_kw):
    return {"segments": [{"start": 0.0, "end": 1.0, "text": "hello",
                          "words": [{"word": "hello", "start": 0.0, "end": 1.0}]}]}


def _add(client, path):
    return client.post("/api/add-path", json={"paths": [str(path)]}).json()


def test_add_path_registers_without_copy(client, two_stream_file):
    item = _add(client, two_stream_file)[0]
    assert item["source"]["filename"] == "two_stream.mkv"
    assert [t["speaker"] for t in item["source"]["tracks"]] == ["Speaker 1", "Speaker 2"]
    assert item["file_id"]


def test_add_path_missing_file_404(client):
    assert client.post("/api/add-path", json={"paths": ["C:/nope/missing.mkv"]}).status_code == 404


def test_browse_uses_native_picker(client, two_stream_file, monkeypatch):
    monkeypatch.setattr(routes.api, "_native_pick", lambda: [str(two_stream_file)])
    r = client.post("/api/browse")
    assert r.status_code == 200
    assert r.json()[0]["source"]["filename"] == "two_stream.mkv"


def test_browse_cancelled_returns_empty(client, monkeypatch):
    monkeypatch.setattr(routes.api, "_native_pick", lambda: [])
    assert client.post("/api/browse").json() == []


def test_export_inline_srt(client):
    merged = {"version": "1.0", "sources": [], "segments": [
        {"speaker": "Max", "start": 0.0, "end": 1.0, "text": "hello",
         "words": [{"word": "hello", "start": 0.0, "end": 1.0}], "flags": []}]}
    r = client.post("/api/export?format=srt", json={"merged": merged, "filename": "mycut"})
    assert r.status_code == 200
    assert "Max: hello" in r.text
    assert r.headers["content-disposition"] == 'attachment; filename="mycut.srt"'


def test_transcribe_flow(client, two_stream_file, monkeypatch):
    monkeypatch.setattr(routes.api, "transcribe_track", _stub_transcribe)
    file_id = _add(client, two_stream_file)[0]["file_id"]

    r = client.post("/api/transcribe", json={
        "sources": [{"file_id": file_id, "sequence": 0, "tracks": [
            {"track_index": 0, "speaker": "Mic", "selected": True},
            {"track_index": 1, "speaker": "Desktop", "selected": True}]}]})
    job_id = r.json()["job_id"]

    deadline = time.time() + 5
    status = "pending"
    while time.time() < deadline:
        status = client.get(f"/api/progress/{job_id}").json()["status"]
        if status in ("done", "error"):
            break
        time.sleep(0.02)
    assert status == "done"

    res = client.get(f"/api/result/{job_id}").json()
    assert {s["speaker"] for s in res["segments"]} == {"Mic", "Desktop"}


def test_audition_returns_audio(client, two_stream_file):
    file_id = _add(client, two_stream_file)[0]["file_id"]
    r = client.get(f"/api/audition/{file_id}/0")
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert len(r.content) > 0
    assert client.get("/api/audition/nope/0").status_code == 404


def test_clip_returns_audio(client, two_stream_file):
    file_id = _add(client, two_stream_file)[0]["file_id"]
    r = client.get(f"/api/clip/{file_id}?start=0&dur=1")
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert len(r.content) > 0
    assert client.get("/api/clip/nope?start=0&dur=1").status_code == 404


def test_track_audio_serves_wav(client, two_stream_file):
    file_id = _add(client, two_stream_file)[0]["file_id"]
    r = client.get(f"/api/track-audio/{file_id}/0")
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert len(r.content) > 0


def test_vad_endpoint_returns_regions(client, two_stream_file):
    file_id = _add(client, two_stream_file)[0]["file_id"]
    r = client.post(f"/api/vad/{file_id}/0")
    assert r.status_code == 200
    assert isinstance(r.json(), list)  # tone -> no speech, but endpoint works


def test_diarize_flow(client, two_stream_file, monkeypatch):
    monkeypatch.setattr(routes.api, "transcribe_track", _stub_transcribe)
    file_id = _add(client, two_stream_file)[0]["file_id"]
    r = client.post("/api/diarize", json={
        "file_id": file_id, "track_index": 0, "speakers": {"Alice": [[0.0, 1.0]]}})
    job_id = r.json()["job_id"]
    deadline = time.time() + 5
    status = "queued"
    while time.time() < deadline:
        status = client.get(f"/api/progress/{job_id}").json()["status"]
        if status in ("done", "error"):
            break
        time.sleep(0.02)
    assert status == "done"
    res = client.get(f"/api/result/{job_id}").json()
    assert any(s["speaker"] == "Alice" for s in res["segments"])


def test_combine_registers_sequenced_source(client, two_stream_file):
    f1 = _add(client, two_stream_file)[0]["file_id"]
    f2 = _add(client, two_stream_file)[0]["file_id"]
    r = client.post("/api/combine", json={"files": [
        {"file_id": f1, "track_index": 0}, {"file_id": f2, "track_index": 0}]})
    assert r.status_code == 200
    d = r.json()
    assert d["file_id"]
    assert abs(d["duration"] - 4.0) < 0.3          # two ~2s files
    assert d["boundaries"][0] == 0.0 and abs(d["boundaries"][1] - 2.0) < 0.3
    assert client.get(f"/api/track-audio/{d['file_id']}/0").status_code == 200  # usable source


def test_unknown_job_404(client):
    assert client.get("/api/progress/nope").status_code == 404
    assert client.get("/api/result/nope").status_code == 404


def test_browse_folder_uses_native_dir_picker(client, monkeypatch):
    monkeypatch.setattr(routes.api, "_native_pick_dir", lambda: "C:/out")
    assert client.post("/api/browse-folder").json() == {"path": "C:/out"}


def test_batch_flow_writes_outputs(client, two_stream_file, tmp_path, monkeypatch):
    monkeypatch.setattr(routes.api, "transcribe_track", _stub_transcribe)
    file_id = _add(client, two_stream_file)[0]["file_id"]

    r = client.post("/api/batch", json={
        "output_mode": "folder", "output_folder": str(tmp_path),
        "projects": [{"name": "Perf Poser", "sources": [
            {"file_id": file_id, "sequence": 0,
             "tracks": [{"track_index": 0, "speaker": "Speaker 1", "selected": True}]}]}]})
    batch_id = r.json()["batch_id"]

    deadline = time.time() + 5
    status = "queued"
    while time.time() < deadline:
        status = client.get(f"/api/batch/{batch_id}").json()["status"]
        if status == "done":
            break
        time.sleep(0.02)
    body = client.get(f"/api/batch/{batch_id}").json()
    assert body["status"] == "done"
    assert body["projects"][0]["status"] == "done"
    assert (tmp_path / "Perf Poser" / "Perf Poser.srt").exists()


def test_batch_folder_mode_requires_folder(client, two_stream_file):
    file_id = _add(client, two_stream_file)[0]["file_id"]
    r = client.post("/api/batch", json={
        "output_mode": "folder",
        "projects": [{"name": "P", "sources": [
            {"file_id": file_id, "tracks": [{"track_index": 0, "speaker": "Speaker 1"}]}]}]})
    assert r.status_code == 400
