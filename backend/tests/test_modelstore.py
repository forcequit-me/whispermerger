"""Model downloads — cancel, resume, and run integration against a slow local server (no internet)."""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import torch.hub

from app import modelstore as ms

PAYLOAD = bytes(range(256)) * 8192          # 2 MB


class _SlowRanged(BaseHTTPRequestHandler):
    """Serves PAYLOAD in 64 KB steps (~0.65 MB/s), honoring `Range: bytes=N-` like the real servers."""

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
                time.sleep(0.1)    # ~3 s per download: well over the 1 s progress tick, so tests can act mid-download
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
    assert ms.has_partial(entry)                          # what the Models tab offers Resume / Remove on

    d2 = mgr.start(entry)
    _wait(lambda: d2.done > 0)
    assert d2.done >= kept                                # picked up the partial, didn't restart
    _wait(lambda: d2.status == "done")
    assert entry.dest.read_bytes() == PAYLOAD and not part.exists()
    assert not ms.has_partial(entry)


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

    after = []
    with ms.reporting(after.append):                      # next run: resumes and completes
        ms.ensure(entry)
    assert ms.is_installed(entry) and entry.dest.read_bytes() == PAYLOAD
    assert after[-1] == "Downloaded Test word timing · loading it"   # not stuck at the last 99% tick


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
    _wait(lambda: d.status == "retrying" and d.note.startswith("no data for 2 s"), timeout=15)
    _wait(lambda: d.status == "done", timeout=30)
    assert e.dest.read_bytes() == PAYLOAD
    server.shutdown()


def test_downloads_queue_and_a_run_skips_the_queue(tmp_path, monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowRanged)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    monkeypatch.setattr(ms, "MAX_PARALLEL", 1)
    url = f"http://127.0.0.1:{server.server_port}/"
    a, b, c = (ms.ModelEntry(f"align:{n}", "align", n, 2, url=url + f"{n}.pt") for n in "abc")
    mgr = ms.DownloadManager()
    da, db = mgr.start(a), mgr.start(b)
    _wait(lambda: da.status == "downloading")
    assert db.status == "queued" and db.active and db.describe() == "queued"
    dc = mgr.start(c, skip_queue=True)                    # a run needs this one now: starts beside a
    _wait(lambda: dc.status != "queued", timeout=2)
    assert da.status == "downloading" and db.status == "queued"
    _wait(lambda: all(d.status == "done" for d in (da, db, dc)), timeout=30)
    server.shutdown()


class _NotFound(_SlowRanged):
    def do_GET(self):
        self.send_error(404)


def test_permanent_error_fails_at_once_with_the_real_reason(tmp_path, monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _NotFound)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    e = ms.ModelEntry("align:gone", "align", "Gone", 2, url=f"http://127.0.0.1:{server.server_port}/m.pt")
    d = ms.DownloadManager().start(e)
    _wait(lambda: d.status == "error", timeout=15)        # no minutes of "retrying"
    assert "404" in d.error
    server.shutdown()


def test_not_enough_disk_space_fails_before_downloading(entry, monkeypatch):
    monkeypatch.setattr(ms, "_free_bytes", lambda p: 1000)
    d = ms.DownloadManager().start(entry)
    _wait(lambda: d.status == "error")
    assert d.error.startswith("not enough disk space") and d.done == 0


def test_already_on_disk_is_not_downloaded_again(entry):
    entry.dest.parent.mkdir(parents=True, exist_ok=True)
    entry.dest.write_bytes(b"x")
    d = ms.DownloadManager().start(entry)
    _wait(lambda: d.status == "done")
    assert entry.dest.read_bytes() == b"x"


def test_fetch_quits_when_the_server_goes_away(tmp_path):
    """A fetch left running after the server stops would fight a restarted server's download."""
    import json, subprocess, sys
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HangsOnce)   # never finishes on its own
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _HangsOnce.hung = False
    spec = {"kind": "url", "url": f"http://127.0.0.1:{server.server_port}/m.pt", "dest": str(tmp_path / "m.pt")}
    proc = subprocess.Popen([sys.executable, "-m", "app.fetch", json.dumps(spec)], cwd=str(ms.BACKEND_DIR),
                            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)
    assert proc.poll() is None                     # still downloading while the server is there
    proc.stdin.close()                             # the server process ends
    assert proc.wait(timeout=10) == 1
    server.shutdown()
