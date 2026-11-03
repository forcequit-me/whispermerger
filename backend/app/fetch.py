"""Download one model's files, run as a subprocess by modelstore.DownloadManager.

A separate process so a cancel or a stalled connection can be ended at once (kill), with no
thread left blocked on a socket. Both kinds resume: Hugging Face keeps `.incomplete` blobs in its
cache, and URL downloads keep a `.part` file and continue with an HTTP Range request.

Usage: python -m app.fetch '<json spec>'
  {"kind": "hf", "repo": "...", "files": ["config.json", ...]}
  {"kind": "url", "url": "...", "dest": "C:/.../file.pt"}
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path


def fetch_hf(repo: str, files: list[str]) -> None:
    from huggingface_hub import hf_hub_download
    for name in files:
        hf_hub_download(repo, name)


def fetch_url(url: str, dest: str) -> None:
    dest_p = Path(dest)
    part = dest_p.with_name(dest_p.name + ".part")
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=60) as resp:
        mode = "ab" if have and resp.status == 206 else "wb"   # server ignored Range -> restart
        with open(part, mode) as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
    part.replace(dest_p)


if __name__ == "__main__":
    spec = json.loads(sys.argv[1])
    if spec["kind"] == "hf":
        fetch_hf(spec["repo"], spec["files"])
    else:
        fetch_url(spec["url"], spec["dest"])
