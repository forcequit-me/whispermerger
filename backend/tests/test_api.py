"""API tests via FastAPI TestClient. Files are registered by path (/add-path); the GUI
picker (/browse) is tested with a monkeypatched picker. transcribe is stubbed (no GPU)."""

import subprocess
import time
from pathlib import Path

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


def test_browse_skips_unreadable_file(client, two_stream_file, monkeypatch, tmp_path):
    junk = tmp_path / "junk.mkv"
    junk.write_bytes(b"not media")
    real_probe = routes.api.probe

    def probe(p):                                # ffprobe exits non-zero on the bad file
        if Path(p) == junk:
            raise subprocess.CalledProcessError(1, "ffprobe")
        return real_probe(p)

    monkeypatch.setattr(routes.api, "probe", probe)
    monkeypatch.setattr(routes.api, "_native_pick", lambda: [str(junk), str(two_stream_file)])
    r = client.post("/api/browse")
    assert r.status_code == 200
    assert [i["source"]["filename"] for i in r.json()] == ["two_stream.mkv"]


def test_add_folder_recursive_registers_media(client, two_stream_file, monkeypatch, tmp_path):
    import shutil
    (tmp_path / "sub").mkdir()
    shutil.copy(two_stream_file, tmp_path / "a.mkv")
    shutil.copy(two_stream_file, tmp_path / "sub" / "b.mkv")
    (tmp_path / "notes.txt").write_text("not media")          # skipped by extension filter
    monkeypatch.setattr(routes.api, "_native_pick_dir", lambda: str(tmp_path))
    r = client.post("/api/add-folder")
    assert r.status_code == 200
    names = sorted(it["source"]["filename"].replace("\\", "/") for it in r.json())
    assert names == ["a.mkv", "sub/b.mkv"]                    # recursive, relative display names


def test_add_folder_cancelled_returns_empty(client, monkeypatch):
    monkeypatch.setattr(routes.api, "_native_pick_dir", lambda: "")
    assert client.post("/api/add-folder").json() == []


def test_export_inline_srt(client):
    merged = {"version": "1.0", "sources": [], "segments": [
        {"speaker": "Max", "start": 0.0, "end": 1.0, "text": "hello",
         "words": [{"word": "hello", "start": 0.0, "end": 1.0}], "flags": []}]}
    r = client.post("/api/export?format=srt", json={"merged": merged, "filename": "mycut"})
    assert r.status_code == 200
    assert "Max: hello" in r.text
    assert r.headers["content-disposition"] == 'attachment; filename="mycut.srt"'


def test_export_several_formats_as_one_zip(client):
    import io, zipfile
    merged = {"version": "1.0", "sources": [], "segments": [
        {"speaker": "Max", "start": 0.0, "end": 1.0, "text": "hello",
         "words": [{"word": "hello", "start": 0.0, "end": 1.0}], "flags": []}]}
    r = client.post("/api/export?format=srt,json", json={"merged": merged, "filename": "mycut"})
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert r.headers["content-disposition"] == 'attachment; filename="mycut.zip"'
    z = zipfile.ZipFile(io.BytesIO(r.content))
    assert sorted(z.namelist()) == ["mycut.json", "mycut.srt"]
    assert "Max: hello" in z.read("mycut.srt").decode()
    assert client.post("/api/export?format=srt,docx", json={"merged": merged}).status_code == 400


def test_transcribe_flow(client, two_stream_file, monkeypatch):
    monkeypatch.setattr(routes.api, "transcribe_track", _stub_transcribe)
    file_id = _add(client, two_stream_file)[0]["file_id"]

    r = client.post("/api/transcribe", json={
        "sources": [{"file_id": file_id, "sequence": 0, "tracks": [
            {"track_index": 0, "speaker": "Speaker 1", "selected": True},
            {"track_index": 1, "speaker": "Speaker 2", "selected": True}]}]})
    job_id = r.json()["job_id"]

    deadline = time.time() + 5
    status = "pending"
    while time.time() < deadline:
        status = client.get(f"/api/progress/{job_id}").json()["status"]
        if status in ("done", "error"):
            break
        time.sleep(0.02)
    assert status == "done"
    final = client.get(f"/api/progress/{job_id}").json()
    assert final["elapsed"] >= 0 and "done in" in final["message"]

    res = client.get(f"/api/result/{job_id}").json()
    assert {s["speaker"] for s in res["segments"]} == {"Speaker 1", "Speaker 2"}


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


def test_clip_folder_is_capped(client, two_stream_file, monkeypatch):
    monkeypatch.setattr(routes.api, "_CLIP_KEEP", 2)
    file_id = _add(client, two_stream_file)[0]["file_id"]
    for start in (0, 0.2, 0.4, 0.6):
        assert client.get(f"/api/clip/{file_id}?start={start}&dur=1").status_code == 200
    assert len(list((client.app.state.upload_dir / "clips").iterdir())) == 2


def test_peaks_drops_its_temp_wav(client, two_stream_file):
    file_id = _add(client, two_stream_file)[0]["file_id"]
    assert client.get(f"/api/peaks/{file_id}?buckets=50").status_code == 200
    assert not list((client.app.state.upload_dir / "peaks").glob("*.wav"))


def test_peaks_for_one_track(client, two_stream_file):
    """Label Track's stacked view: peaks of each stream on its own (440 Hz vs 880 Hz sines)."""
    file_id = _add(client, two_stream_file)[0]["file_id"]
    a = client.get(f"/api/peaks/{file_id}?track=0&buckets=20").json()
    b = client.get(f"/api/peaks/{file_id}?track=1&buckets=20").json()
    assert len(a["peaks"]) == len(b["peaks"]) == 20 and a["duration"] > 0
    assert client.get(f"/api/peaks/{file_id}?track=1&buckets=20").json() == b      # cached
    # zoomed-in window: asked-for bucket count, same scale as the whole track
    w = client.get(f"/api/peaks/{file_id}?track=1&buckets=30&start=0&end=0.5").json()
    assert len(w["peaks"]) == 30 and 0 < max(w["peaks"]) <= 1.0
    assert list((client.app.state.upload_dir / "diar").glob(f"{file_id}_t1.wav"))  # freed by /release
    client.post(f"/api/release/{file_id}")
    assert not list((client.app.state.upload_dir / "diar").glob(f"{file_id}_t*.wav"))


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


def test_diarize_passes_language(client, two_stream_file, monkeypatch):
    seen = []
    monkeypatch.setattr(routes.api, "transcribe_track",
                        lambda w, **kw: seen.append(kw.get("language")) or _stub_transcribe(w))
    file_id = _add(client, two_stream_file)[0]["file_id"]
    job_id = client.post("/api/diarize", json={
        "file_id": file_id, "speakers": {"A": [[0.0, 1.0]]}, "language": "no"}).json()["job_id"]
    deadline = time.time() + 5
    while client.get(f"/api/progress/{job_id}").json()["status"] not in ("done", "error")             and time.time() < deadline:
        time.sleep(0.02)
    assert seen == ["no"]


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


def test_release_deletes_only_that_files_track_copies(client, two_stream_file):
    keep, drop = (_add(client, two_stream_file)[0]["file_id"] for _ in range(2))
    for fid in (keep, drop):
        for t in (0, 1):
            assert client.get(f"/api/track-audio/{fid}/{t}").status_code == 200
    assert client.post(f"/api/release/{drop}").json() == {"released": 2}
    diar = client.app.state.upload_dir / "diar"
    assert not list(diar.glob(f"{drop}_t*.wav"))
    assert len(list(diar.glob(f"{keep}_t*.wav"))) == 2           # other file untouched
    assert client.post("/api/release/nope").json() == {"released": 0}


def test_combined_is_used_directly_and_released(client, two_stream_file):
    ids = [_add(client, two_stream_file)[0]["file_id"] for _ in range(2)]
    body = {"files": [{"file_id": f, "track_index": 0} for f in ids]}
    cid = client.post("/api/combine", json=body).json()["file_id"]
    assert client.get(f"/api/track-audio/{cid}/0").status_code == 200
    diar = client.app.state.upload_dir / "diar"
    assert not list(diar.glob(f"{cid}_t*.wav"))                  # no second copy of the combined wav
    path = Path(client.app.state.uploads[cid]["path"])
    client.post(f"/api/release/{cid}")
    assert not path.exists() and cid not in client.app.state.uploads
    again = client.post("/api/combine", json=body)                # same order later: rebuilt
    assert again.status_code == 200 and again.json()["file_id"] != cid
    assert client.get(f"/api/track-audio/{again.json()['file_id']}/0").status_code == 200


def test_use_model_endpoint(client, monkeypatch, tmp_path):
    from app import modelstore
    monkeypatch.setattr(modelstore, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(routes.api, "unload_models", lambda: None)
    assert client.post("/api/models/align:en/use").status_code == 400            # not a speech model
    monkeypatch.setattr(modelstore, "is_installed", lambda e: False)
    assert client.post("/api/models/whisper:small/use").status_code == 409       # not downloaded
    monkeypatch.setattr(modelstore, "is_installed", lambda e: True)
    assert client.post("/api/models/whisper:small/use").json() == {"active": "whisper:small"}
    assert client.get("/api/models").json()["active"] == "whisper:small"
    assert client.delete("/api/models/whisper:small").status_code == 409         # in use: can't remove


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
    assert body["elapsed"] >= 0 and body["projects"][0]["elapsed"] >= 0
    assert list((tmp_path / "Perf Poser").glob("Perf Poser Transcript (*).srt"))

    # #34: the finished transcript can be opened in the editor, with its source for playback
    res = client.get(f"/api/batch/{batch_id}/project/0/result").json()
    assert res["merged"]["segments"] and res["source_path"] == str(two_stream_file)
    assert client.get(f"/api/batch/{batch_id}/project/9/result").status_code == 404
    assert client.get(f"/api/batch/{batch_id}/project/0/file/0/result").status_code == 404


def test_batch_per_file_result_opens_each_file(client, two_stream_file, tmp_path, monkeypatch):
    monkeypatch.setattr(routes.api, "transcribe_track", _stub_transcribe)
    file_id = _add(client, two_stream_file)[0]["file_id"]
    r = client.post("/api/batch", json={
        "output_mode": "folder", "output_folder": str(tmp_path),
        "projects": [{"name": "Each", "per_file": True, "sources": [
            {"file_id": file_id, "sequence": 0,
             "tracks": [{"track_index": 0, "speaker": "Speaker 1", "selected": True}]}]}]})
    batch_id = r.json()["batch_id"]
    deadline = time.time() + 5
    while client.get(f"/api/batch/{batch_id}").json()["status"] != "done" and time.time() < deadline:
        time.sleep(0.02)
    res = client.get(f"/api/batch/{batch_id}/project/0/file/0/result")
    assert res.status_code == 200 and res.json()["source_path"] == str(two_stream_file)
    assert res.json()["merged"]["segments"][0]["speaker"] == "Speaker 1"


def test_batch_settings_global_and_per_project_override(client, two_stream_file, tmp_path, monkeypatch):
    import json as _json
    monkeypatch.setattr(routes.api, "transcribe_track", _stub_transcribe)   # same words on both tracks
    file_id = _add(client, two_stream_file)[0]["file_id"]
    src = [{"file_id": file_id, "sequence": 0, "tracks": [
        {"track_index": 0, "speaker": "Speaker 1", "selected": True},
        {"track_index": 1, "speaker": "Speaker 2", "selected": True}]}]
    r = client.post("/api/batch", json={
        "output_mode": "folder", "output_folder": str(tmp_path),
        "export": {"show_speakers": False, "pad_seconds": 0.5},          # batch-wide
        "projects": [
            {"name": "Global", "sources": src, "formats": ["srt", "txt", "json"]},
            {"name": "Custom", "sources": src, "formats": ["srt", "txt", "json"],
             "options": {"detect_conflicts": False},
             "export": {"show_speakers": True}}]})
    batch_id = r.json()["batch_id"]
    deadline = time.time() + 5
    while client.get(f"/api/batch/{batch_id}").json()["status"] != "done" and time.time() < deadline:
        time.sleep(0.02)
    out = lambda name, ext: next((tmp_path / name).glob(f"{name} Transcript (*).{ext}")).read_text("utf-8")

    assert "Speaker 1:" not in out("Global", "srt")                    # batch: no names
    assert "Speaker 1: hello" in out("Custom", "srt")                  # override: names
    assert "00:00:01,500]" in out("Global", "txt")                     # batch: padded 0.5 s
    assert "00:00:01,000]" in out("Custom", "txt")                     # override: no padding
    g_flags = [seg["flags"] for seg in _json.loads(out("Global", "json"))["segments"]]
    c_flags = [seg["flags"] for seg in _json.loads(out("Custom", "json"))["segments"]]
    assert all("bleed_suspected" in f for f in g_flags)              # batch merge settings
    assert all(f == [] for f in c_flags)                             # override: detection off


def test_batch_edited_lines_setting_reaches_each_project(client, two_stream_file, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(routes.api, "run_batch", lambda batch, projects, *a, **kw: seen.extend(projects))
    file_id = _add(client, two_stream_file)[0]["file_id"]
    src = [{"file_id": file_id, "sequence": 0,
            "tracks": [{"track_index": 0, "speaker": "Speaker 1", "selected": True}]}]
    client.post("/api/batch", json={
        "output_mode": "folder", "output_folder": str(tmp_path), "export": {"edited_timing": "whole"},
        "projects": [{"name": "A", "sources": src},
                     {"name": "B", "sources": src, "export": {"edited_timing": "even"}}]})
    deadline = time.time() + 5
    while len(seen) < 2 and time.time() < deadline:
        time.sleep(0.02)
    assert [p.export.edited_timing for p in seen] == ["whole", "even"]     # batch-wide, then override


def test_batch_project_cancel_endpoint(client, two_stream_file, tmp_path, monkeypatch):
    monkeypatch.setattr(routes.api, "transcribe_track", _stub_transcribe)
    file_id = _add(client, two_stream_file)[0]["file_id"]
    r = client.post("/api/batch", json={
        "output_mode": "folder", "output_folder": str(tmp_path),
        "projects": [{"name": "P1", "sources": [
            {"file_id": file_id, "sequence": 0,
             "tracks": [{"track_index": 0, "speaker": "Speaker 1", "selected": True}]}]}]})
    batch_id = r.json()["batch_id"]
    # valid per-project cancel is accepted; bad project index / unknown batch -> 404
    assert client.post(f"/api/batch/{batch_id}/project/0/cancel").json()["canceling"]
    assert client.post(f"/api/batch/{batch_id}/project/99/cancel").status_code == 404
    assert client.post("/api/batch/nope/project/0/cancel").status_code == 404
    deadline = time.time() + 5
    while time.time() < deadline:
        if client.get(f"/api/batch/{batch_id}").json()["status"] in ("done", "canceled"):
            break
        time.sleep(0.02)


def test_batch_folder_mode_requires_folder(client, two_stream_file):
    file_id = _add(client, two_stream_file)[0]["file_id"]
    r = client.post("/api/batch", json={
        "output_mode": "folder",
        "projects": [{"name": "P", "sources": [
            {"file_id": file_id, "tracks": [{"track_index": 0, "speaker": "Speaker 1"}]}]}]})
    assert r.status_code == 400
