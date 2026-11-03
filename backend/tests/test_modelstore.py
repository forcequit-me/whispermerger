"""Model downloads — cancel, resume, and run integration against a slow local server (no internet)."""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import torch.hub

from app import modelstore as ms

PAYLOAD = bytes(range(256)) * 8192          # 2 MB


class _SlowRanged(BaseHTTPRequestHandler):
    """Serves PAYLOAD in 64 KB steps (~1.3 MB/s), honoring `Range: bytes=N-` like the real servers."""

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(PAYLOAD)))
        self.end_headers()

    def do_GET(self):
        start = int(self.headers["Range"].split("=")[1].rstrip("-")) if self.headers.get("Range") else 0
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(PAYLOAD) - start))
        self.end_headers()
        try:
            for i in range(start, len(PAYLOAD), 65536):
                self.wfile.write(PAYLOAD[i:i + 65536])
                time.sleep(0.05)
        except (ConnectionError, OSError):
            pass

    def log_message(self, *a):
        pass


@pytest.fixture
def entry(tmp_path, monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowRanged)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))      # where URL models land
    yield ms.ModelEntry("align:test", "align", "Test word timing", 2,
                        url=f"http://127.0.0.1:{server.server_port}/model.pt")
    server.shutdown()


def _wait(pred, timeout=20):
    end = time.time() + timeout
    while not pred() and time.time() < end:
        time.sleep(0.05)
    assert pred()


def test_torchaudio_files_match_the_bundles():
    import torchaudio
    from whisperx.alignment import DEFAULT_ALIGN_MODELS_TORCH
    for lang, bundle in DEFAULT_ALIGN_MODELS_TORCH.items():
        assert ms._TORCHAUDIO_FILES[lang] == getattr(torchaudio.pipelines, bundle)._path


def test_cancel_keeps_partial_and_restart_resumes(entry):
    mgr = ms.DownloadManager()
    d = mgr.start(entry)
    _wait(lambda: d.done > 300_000)
    assert mgr.cancel(entry.id)
    _wait(lambda: d.status == "canceled")
    part = entry.dest.with_name(entry.dest.name + ".part")
    kept = part.stat().st_size
    assert 0 < kept < len(PAYLOAD) and not ms.is_installed(entry)

    d2 = mgr.start(entry)
    _wait(lambda: d2.done > 0)
    assert d2.done >= kept                                # picked up the partial, didn't restart
    _wait(lambda: d2.status == "done")
    assert entry.dest.read_bytes() == PAYLOAD and not part.exists()


def test_ensure_reports_progress_and_cancel_stops_the_download(entry, monkeypatch):
    mgr = ms.DownloadManager()
    monkeypatch.setattr(ms, "downloads", mgr)
    seen = []

    def report(msg):                                      # a run's progress line; cancel on the 3rd call
        seen.append(msg)
        if len(seen) == 3:
            raise KeyboardInterrupt("run canceled")

    with ms.reporting(report), pytest.raises(KeyboardInterrupt):
        ms.ensure(entry)
    assert seen[0].startswith("Downloading Test word timing")
    _wait(lambda: mgr.downloads[entry.id].status == "canceled")

    with ms.reporting(lambda msg: None):                  # next run: resumes and completes
        ms.ensure(entry)
    assert ms.is_installed(entry) and entry.dest.read_bytes() == PAYLOAD


class _HangsOnce(_SlowRanged):
    """First request sends 512 KB then goes silent (a dropped connection that never closes)."""
    hung = False

    def do_GET(self):
        if type(self).hung:
            return super().do_GET()
        type(self).hung = True
        self.send_response(200)
        self.send_header("Content-Length", str(len(PAYLOAD)))
        self.end_headers()
        self.wfile.write(PAYLOAD[:524288])
        self.wfile.flush()
        time.sleep(30)


def test_stalled_connection_is_restarted_and_resumed(tmp_path, monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HangsOnce)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    monkeypatch.setattr(ms, "STALL_SECONDS", 2)
    e = ms.ModelEntry("align:stall", "align", "Stall test", 2, url=f"http://127.0.0.1:{server.server_port}/m.pt")
    d = ms.DownloadManager().start(e)
    _wait(lambda: d.note == "connection dropped, resuming", timeout=15)
    _wait(lambda: d.status == "done", timeout=30)
    assert e.dest.read_bytes() == PAYLOAD
    server.shutdown()
