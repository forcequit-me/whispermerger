"""API tests via FastAPI TestClient. Real upload/probe + a monkeypatched transcribe stub
so the full job flow runs without a GPU."""

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


def test_upload_returns_tracks(client, two_stream_file):
    with open(two_stream_file, "rb") as f:
        r = client.post("/api/upload", files={"files": ("two_stream.mkv", f, "video/x-matroska")})
    assert r.status_code == 200
    item = r.json()[0]
    assert item["source"]["filename"] == "two_stream.mkv"
    assert [t["speaker"] for t in item["source"]["tracks"]] == ["Mic", "Desktop"]


def test_export_inline_srt(client):
    merged = {"version": "1.0", "sources": [], "segments": [
        {"speaker": "Max", "start": 0.0, "end": 1.0, "text": "hello",
         "words": [{"word": "hello", "start": 0.0, "end": 1.0}], "flags": []}]}
    r = client.post("/api/export?format=srt",
                    json={"merged": merged, "filename": "mycut"})
    assert r.status_code == 200
    assert "Max: hello" in r.text
    assert r.headers["content-disposition"] == 'attachment; filename="mycut.srt"'


def test_transcribe_flow(client, two_stream_file, monkeypatch):
    monkeypatch.setattr(routes.api, "transcribe_track", _stub_transcribe)

    with open(two_stream_file, "rb") as f:
        up = client.post("/api/upload",
                         files={"files": ("two_stream.mkv", f, "video/x-matroska")}).json()
    file_id = up[0]["file_id"]

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
    with open(two_stream_file, "rb") as f:
        up = client.post("/api/upload",
                         files={"files": ("two_stream.mkv", f, "video/x-matroska")}).json()
    file_id = up[0]["file_id"]
    r = client.get(f"/api/audition/{file_id}/0")
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert len(r.content) > 0
    assert client.get("/api/audition/nope/0").status_code == 404


def test_unknown_job_404(client):
    assert client.get("/api/progress/nope").status_code == 404
    assert client.get("/api/result/nope").status_code == 404
