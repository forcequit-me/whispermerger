"""Model catalog + downloads (Models tab, FR model management).

Two kinds of model:
  whisper:<name>  the speech-to-text model (faster-whisper, from Hugging Face)
  align:<lang>    the per-language word-timing model (wav2vec2, from Hugging Face or, for five
                  languages, torchaudio's server)

Files land in the same places the libraries already look (the Hugging Face cache, torch hub's
checkpoints folder), so models downloaded before this existed count as installed and loading code
is unchanged. Each download runs as a subprocess (app.fetch) that this module can kill at once for
Cancel, restarts when it stalls, and resumes from the partial file either way.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

BACKEND_DIR = Path(__file__).resolve().parent.parent
STALL_SECONDS = 90         # no new bytes for this long = dropped connection; restart and resume
MAX_STALLS = 4             # give up after this many restarts in a row that gained nothing
MAX_PARALLEL = 2           # downloads at once; the rest wait as "queued" so each gets real bandwidth
# Errors no retry will fix: fail at once with the real reason instead of "retrying" for minutes.
_PERMANENT = ("404", "401", "403", "Not Found", "RepositoryNotFound", "EntryNotFound", "GatedRepo",
              "No space left", "Errno 28", "PermissionError", "Access is denied")

# Word-timing models for languages WhisperX has no default for (wav2vec2 CTC, character vocab).
EXTRA_ALIGN_MODELS = {
    "sv": "KBLab/wav2vec2-large-voxrex-swedish",   # National Library of Sweden, CC0
}

# The speech models offered in the Models tab. Sizes are the download in MB (measured 2026-11);
# speed/accuracy are rough tiers, the Benchmark gives real numbers on this machine.
WHISPER_MODELS = {
    # multilingual
    "large-v3":          dict(size_mb=3091, accuracy="best", speed="slowest", languages="99 languages"),
    "large-v2":          dict(size_mb=3090, accuracy="best (previous)", speed="slowest", languages="99 languages"),
    "large-v3-turbo":    dict(size_mb=1622, accuracy="near large-v3", speed="fast", languages="99 languages"),
    "medium":            dict(size_mb=1531, accuracy="good", speed="medium", languages="99 languages"),
    "small":             dict(size_mb=486, accuracy="fair", speed="fast", languages="99 languages"),
    "base":              dict(size_mb=148, accuracy="rough", speed="very fast", languages="99 languages"),
    "tiny":              dict(size_mb=78, accuracy="roughest", speed="fastest", languages="99 languages"),
    # English only
    "distil-large-v3.5": dict(size_mb=1516, accuracy="near large-v3", speed="fastest", languages="English only"),
    "distil-large-v3":   dict(size_mb=1516, accuracy="near large-v3", speed="fastest", languages="English only"),
    "medium.en":         dict(size_mb=1530, accuracy="good", speed="medium", languages="English only"),
    "distil-medium.en":  dict(size_mb=792, accuracy="fair", speed="very fast", languages="English only"),
    "small.en":          dict(size_mb=486, accuracy="fair", speed="fast", languages="English only"),
    "distil-small.en":   dict(size_mb=336, accuracy="rough", speed="very fast", languages="English only"),
    "base.en":           dict(size_mb=148, accuracy="rough", speed="very fast", languages="English only"),
    "tiny.en":           dict(size_mb=78, accuracy="roughest", speed="fastest", languages="English only"),
    # one language: used automatically for that language (not chosen app-wide). CAMOES is INESC-ID's
    # large-v3 fine-tune on 425 h of European Portuguese (arXiv 2508.19721: 12.5% vs 19.2% WER);
    # these are community CTranslate2 conversions of it. License unstated by the authors. It writes
    # lowercase with almost no punctuation (so do the official weights; tested 2026-11).
    "camoes":            dict(size_mb=3092, accuracy="best for European Portuguese, no punctuation", speed="slowest",
                              languages="Portuguese (Portugal)", for_language="pt-PT",
                              label="CAMÕES (European Portuguese)", repo="tigopt/CAMOES-whisper-asr-CT2"),
    "camoes-int8":       dict(size_mb=1564, accuracy="near CAMÕES, no punctuation", speed="slow",
                              languages="Portuguese (Portugal)", for_language="pt-PT",
                              label="CAMÕES compressed (European Portuguese)", compute_type="int8_float16",
                              repo="marquesafonso/camoes-faster-whisper-asr-int8"),
}
DEFAULT_WHISPER = "large-v3"
# App-wide settings that survive restarts (tmp/uploads is wiped at startup; this file isn't).
SETTINGS_FILE = BACKEND_DIR / "tmp" / "settings.json"


def _settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text("utf-8"))
    except (OSError, ValueError):
        return {}


def active_whisper() -> str:
    """The speech model every run uses (chosen in the Models tab)."""
    name = _settings().get("whisper_model")
    return name if name in WHISPER_MODELS and "for_language" not in WHISPER_MODELS[name] else DEFAULT_WHISPER


def set_active_whisper(name: str) -> None:
    if name not in WHISPER_MODELS or "for_language" in WHISPER_MODELS[name]:
        raise ValueError(f"{name} can't be the app-wide speech model")
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps({**_settings(), "whisper_model": name}, indent=2), "utf-8")


# Dropdown language codes that Whisper itself doesn't have: variant -> Whisper's code. Results are
# labeled with the variant; European Portuguese is transcribed with CAMOES.
LANGUAGE_VARIANTS = {"pt-PT": "pt", "pt-BR": "pt"}


def portugal_model() -> str:
    """CAMOES for European Portuguese: the full model if it's on disk, else the compressed one if
    that is, else the full one (downloaded on first use)."""
    for name in ("camoes", "camoes-int8"):
        if is_installed(whisper_entry(name)):
            return name
    return "camoes"


def pt_default() -> Optional[str]:
    """What Auto-detect treats detected Portuguese as: "pt-PT", "pt-BR", or None (plain "pt")."""
    v = _settings().get("auto_portuguese")
    return v if v in LANGUAGE_VARIANTS else None


def set_pt_default(value: Optional[str]) -> None:
    if value is not None and value not in LANGUAGE_VARIANTS:
        raise ValueError(f"unknown Portuguese variant {value}")
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps({**_settings(), "auto_portuguese": value}, indent=2), "utf-8")


def pt_punctuation() -> bool:
    """Portugal runs also transcribe with large-v3 and copy its capitalization and punctuation onto
    CAMOES's words (about 2x GPU time). Off = CAMOES alone."""
    return bool(_settings().get("pt_punctuation", False))


def set_pt_punctuation(on: bool) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps({**_settings(), "pt_punctuation": bool(on)}, indent=2), "utf-8")


def english_only(name: str) -> bool:
    return WHISPER_MODELS.get(name, {}).get("languages") == "English only"


# faster-whisper's own download filter, so we fetch exactly what it would.
_WHISPER_FILES = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"]
_ALIGN_FILES = ["config.json", "preprocessor_config.json", "vocab.json", "tokenizer_config.json",
                "special_tokens_map.json", "added_tokens.json"]

# torchaudio's five default alignment models: checkpoint file on download.pytorch.org.
_TORCHAUDIO_FILES = {
    "en": "wav2vec2_fairseq_base_ls960_asr_ls960.pth",
    "fr": "wav2vec2_voxpopuli_base_10k_asr_fr.pt",
    "de": "wav2vec2_voxpopuli_base_10k_asr_de.pt",
    "es": "wav2vec2_voxpopuli_base_10k_asr_es.pt",
    "it": "wav2vec2_voxpopuli_base_10k_asr_it.pt",
}
_TORCHAUDIO_URL = "https://download.pytorch.org/torchaudio/models/"

_ALIGN_SIZES_MB = {"vi": 380, "no": 3850, "nn": 3850, "ko": 1267, "zh": 1276, "ja": 1272}  # rest ~1262

LANG_NAMES = {
    "ar": "Arabic", "ca": "Catalan", "cs": "Czech", "da": "Danish", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "eu": "Basque", "fa": "Persian", "fi": "Finnish", "fr": "French",
    "gl": "Galician", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian", "hu": "Hungarian",
    "it": "Italian", "ja": "Japanese", "ka": "Georgian", "ko": "Korean", "lv": "Latvian",
    "ml": "Malayalam", "nl": "Dutch", "nn": "Norwegian Nynorsk", "no": "Norwegian", "pl": "Polish",
    "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sk": "Slovak", "sl": "Slovenian",
    "sv": "Swedish", "te": "Telugu", "tl": "Tagalog", "tr": "Turkish", "uk": "Ukrainian",
    "ur": "Urdu", "vi": "Vietnamese", "zh": "Chinese",
}


@dataclass
class ModelEntry:
    id: str                      # "whisper:large-v3" / "align:nl"
    kind: str                    # "whisper" | "align"
    name: str                    # shown in the UI
    size_mb: int
    repo: str = ""               # Hugging Face repo, or "" for a URL download
    url: str = ""
    info: dict = field(default_factory=dict)   # accuracy / speed / languages for the UI

    @property
    def dest(self) -> Path:      # URL downloads: where torchaudio looks for the checkpoint
        import torch.hub
        return Path(torch.hub.get_dir()) / "checkpoints" / self.url.rsplit("/", 1)[1]


def _whisper_repo(name: str) -> str:
    from faster_whisper.utils import _MODELS
    return _MODELS.get(name, name)


def whisper_entry(name: str) -> ModelEntry:
    meta = WHISPER_MODELS.get(name, dict(size_mb=0, accuracy="?", speed="?", languages="?"))
    return ModelEntry(f"whisper:{name}", "whisper", meta.get("label", f"Whisper {name}"), meta["size_mb"],
                      repo=meta.get("repo") or _whisper_repo(name), info=meta)


def model_path(name: str) -> str:
    """What to hand faster-whisper: its own short name, or our repo for a model it doesn't know."""
    return WHISPER_MODELS.get(name, {}).get("repo") or name


def align_entry(lang: str) -> Optional[ModelEntry]:
    """The word-timing model for a language code, or None if there isn't one."""
    from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF
    name = f"{LANG_NAMES.get(lang, lang)} word timing"
    if lang in _TORCHAUDIO_FILES:
        return ModelEntry(f"align:{lang}", "align", name, 378, url=_TORCHAUDIO_URL + _TORCHAUDIO_FILES[lang],
                          info={"language": lang})
    repo = EXTRA_ALIGN_MODELS.get(lang) or DEFAULT_ALIGN_MODELS_HF.get(lang)
    if not repo:
        return None
    return ModelEntry(f"align:{lang}", "align", name, _ALIGN_SIZES_MB.get(lang, 1262), repo=repo,
                      info={"language": lang})


def entries() -> list[ModelEntry]:
    from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF
    langs = sorted(set(_TORCHAUDIO_FILES) | set(DEFAULT_ALIGN_MODELS_HF) | set(EXTRA_ALIGN_MODELS),
                   key=lambda c: LANG_NAMES.get(c, c))
    return [whisper_entry(n) for n in WHISPER_MODELS] + [align_entry(c) for c in langs]


def get(model_id: str) -> Optional[ModelEntry]:
    kind, _, key = model_id.partition(":")
    if kind == "whisper":
        return whisper_entry(key)
    if kind == "align":
        return align_entry(key)
    return None


# --- on disk ---------------------------------------------------------------------------

def _cached(repo: str, filename: str) -> bool:
    from huggingface_hub import try_to_load_from_cache
    return isinstance(try_to_load_from_cache(repo, filename), str)


def is_installed(e: ModelEntry) -> bool:
    if e.url:
        return e.dest.exists()
    if e.kind == "whisper":
        return _cached(e.repo, "config.json") and _cached(e.repo, "model.bin")
    return (_cached(e.repo, "config.json") and _cached(e.repo, "vocab.json")
            and (_cached(e.repo, "model.safetensors") or _cached(e.repo, "pytorch_model.bin")))


def _repo_dir(repo: str) -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE
    return Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}"


def remove(e: ModelEntry) -> None:
    """Delete a model's files, including any partial download."""
    if e.url:
        for p in (e.dest, e.dest.with_name(e.dest.name + ".part")):
            p.unlink(missing_ok=True)
    else:
        shutil.rmtree(_repo_dir(e.repo), ignore_errors=True)


# --- downloads -------------------------------------------------------------------------

def _fmt_mb(n: float) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


@dataclass
class Download:
    id: str
    # queued | connecting | downloading | retrying | done | error | canceled
    status: str = "queued"
    done: int = 0                # bytes on disk so far
    total: int = 0
    speed: float = 0.0           # bytes/s
    note: str = ""               # why it's retrying, e.g. "no data for 90 s, retrying (attempt 2)"
    error: str = ""
    skip_queue: bool = False     # a run is waiting on this model: don't wait for a free slot
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def active(self) -> bool:
        return self.status in ("queued", "connecting", "downloading", "retrying")

    def describe(self) -> str:
        if self.status == "queued":
            return "queued"
        if not self.total:
            return "connecting…" if self.active else self.status
        pct = int(100 * self.done / self.total)
        head = f"{pct}% of {_fmt_mb(self.total)}"
        if self.status == "connecting":
            return f"{head} · connecting…"
        if self.status == "retrying":
            return f"{head} · {self.note}"
        return head + (f" · {_fmt_mb(self.speed)}/s" if self.speed > 0 else "")

    def to_dict(self) -> dict:
        return {"status": self.status, "active": self.active, "done": self.done, "total": self.total,
                "speed": self.speed, "text": self.describe(), "error": self.error}


def _free_bytes(path: Path) -> int:
    """Free space on the drive `path` is (or will be) on."""
    while not path.exists() and path.parent != path:
        path = path.parent
    return shutil.disk_usage(path).free


def _target_dir(e: ModelEntry) -> Path:
    return e.dest.parent if e.url else _repo_dir(e.repo)


def disk_free() -> int:
    """Free space where Hugging Face models land (the Models tab warns before Download all)."""
    from huggingface_hub.constants import HF_HUB_CACHE
    return _free_bytes(Path(HF_HUB_CACHE))


def _plan(e: ModelEntry) -> tuple[dict, Callable[[], int], int]:
    """What the fetch subprocess should get, how to measure bytes on disk, and the total size."""
    if e.url:
        import urllib.request
        with urllib.request.urlopen(urllib.request.Request(e.url, method="HEAD"), timeout=30) as r:
            total = int(r.headers.get("Content-Length") or 0)
        part = e.dest.with_name(e.dest.name + ".part")
        measure = lambda: (e.dest.stat().st_size if e.dest.exists()
                           else part.stat().st_size if part.exists() else 0)
        return {"kind": "url", "url": e.url, "dest": str(e.dest)}, measure, total

    import fnmatch
    from huggingface_hub import HfApi
    sib = HfApi().model_info(e.repo, files_metadata=True).siblings
    names = {s.rfilename for s in sib}
    if e.kind == "whisper":
        want = {n for n in names if any(fnmatch.fnmatch(n, p) for p in _WHISPER_FILES)}
    else:
        want = {n for n in names if n in _ALIGN_FILES}
        want.add("model.safetensors" if "model.safetensors" in names else "pytorch_model.bin")
    files = [s for s in sib if s.rfilename in want]
    blobs = _repo_dir(e.repo) / "blobs"

    def blob_name(s) -> str:
        lfs = s.lfs
        sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
        return sha or s.blob_id

    sized = [(blob_name(s), s.size or 0) for s in files]

    def measure() -> int:
        got = 0
        for name, size in sized:
            for p in (blobs / name, blobs / f"{name}.incomplete"):
                if p.exists():
                    got += min(size, p.stat().st_size)
                    break
        return got

    return ({"kind": "hf", "repo": e.repo, "files": sorted(want)}, measure,
            sum(size for _, size in sized))


class DownloadManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._slots = threading.Semaphore(MAX_PARALLEL)
        self.downloads: dict[str, Download] = {}

    def start(self, e: ModelEntry, skip_queue: bool = False) -> Download:
        """Queue a download of `e` (or return the one already queued or running for it).
        `skip_queue`: a run needs this model now, so it starts without waiting for a slot."""
        with self._lock:
            d = self.downloads.get(e.id)
            if d and d.active:
                d.skip_queue = d.skip_queue or skip_queue
                return d
            d = self.downloads[e.id] = Download(e.id, skip_queue=skip_queue)
        threading.Thread(target=self._queued, args=(e, d), daemon=True).start()
        return d

    def cancel(self, model_id: str) -> bool:
        d = self.downloads.get(model_id)
        if not d or not d.active:
            return False
        d._cancel.set()
        return True

    def _queued(self, e: ModelEntry, d: Download) -> None:
        """Wait for a free slot (or for a run to need this model), then download."""
        got = False
        while not d.skip_queue and not (got := self._slots.acquire(timeout=0.5)):
            if d._cancel.is_set():
                d.status = "canceled"
                return
        try:
            self._run(e, d)
        finally:
            if got:
                self._slots.release()

    def _run(self, e: ModelEntry, d: Download) -> None:
        if is_installed(e):              # e.g. a run fetched it while this one sat in the queue
            d.status = "done"
            return
        d.status = "connecting"
        try:
            spec, measure, d.total = _plan(e)
        except Exception as exc:
            d.status, d.error = "error", f"couldn't reach the download server: {exc}"
            return
        need, free = d.total - measure(), _free_bytes(_target_dir(e))
        if need > free:
            d.status = "error"
            d.error = f"not enough disk space: needs {_fmt_mb(need)}, {_fmt_mb(free)} free"
            return
        # Classic HTTP download, not Xet: it grows a .incomplete file we can measure and resume,
        # where Xet assembles the file elsewhere and only drops it in place at the end.
        env = {**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_HUB_DISABLE_XET": "1"}
        stalls, attempt = 0, 0
        while True:
            attempt += 1
            before = measure()
            log = tempfile.TemporaryFile()
            proc = subprocess.Popen([sys.executable, "-m", "app.fetch", json.dumps(spec)],
                                    cwd=str(BACKEND_DIR), env=env, stdout=subprocess.DEVNULL, stderr=log,
                                    stdin=subprocess.PIPE)   # closes when the server dies: fetch quits
            last_bytes, last_t, prev = before, time.time(), (before, time.time())
            stalled = False
            while proc.poll() is None:
                if d._cancel.wait(1.0):
                    proc.kill()
                    proc.wait()
                    d.status = "canceled"
                    return
                now_b, now_t = measure(), time.time()
                d.done = now_b
                d.speed = max(0.0, (now_b - prev[0]) / max(now_t - prev[1], 1e-6))
                prev = (now_b, now_t)
                if now_b > last_bytes:
                    last_bytes, last_t, d.note, d.status = now_b, now_t, "", "downloading"
                elif now_t - last_t > STALL_SECONDS:      # dropped connection: restart, resume
                    proc.kill()
                    proc.wait()
                    stalled = True
                    break
            if proc.returncode == 0 and is_installed(e):
                d.done, d.speed, d.status = d.total, 0.0, "done"
                return
            stalls = 0 if measure() > before else stalls + 1
            log.seek(0)
            err = log.read().decode("utf-8", "replace").strip().splitlines()
            reason = err[-1].strip() if err else ""
            if (reason and any(p in reason for p in _PERMANENT)) or stalls >= MAX_STALLS:
                d.status = "error"
                d.error = reason or "the download stopped making progress"
                return
            why = f"no data for {STALL_SECONDS} s" if stalled or not reason else f"error: {reason[:120]}"
            d.status, d.speed = "retrying", 0.0
            d.note = f"{why}, retrying (attempt {attempt + 1})"
            if d._cancel.wait(min(2 * attempt, 10)):
                d.status = "canceled"
                return


downloads = DownloadManager()


# --- making sure a run has what it needs -----------------------------------------------

_reporter = threading.local()


@contextmanager
def reporting(fn: Callable[[str], None]):
    """Route download progress for this thread to `fn(message)` (a job's progress line). `fn` is
    also the run's pause/cancel checkpoint: if it raises, the download it was waiting on stops."""
    prev = getattr(_reporter, "fn", None)
    _reporter.fn = fn
    try:
        yield
    finally:
        _reporter.fn = prev


def ensure(e: ModelEntry) -> None:
    """Download `e` if it isn't on disk yet, reporting progress to this thread's reporter."""
    if is_installed(e):
        return
    d = downloads.start(e, skip_queue=True)
    fn = getattr(_reporter, "fn", None)
    try:
        while d.active:
            if fn:
                fn(f"Downloading {e.name} · {d.describe()}")
            time.sleep(0.5)
    except BaseException:                 # the run was canceled: stop what it was waiting on
        downloads.cancel(e.id)
        raise
    if d.status == "done" and fn:         # the last tick may have shown 99%: say it finished
        fn(f"Downloaded {e.name} · loading it")
    if d.status != "done":
        raise RuntimeError(f"Download of {e.name} {d.status}" + (f": {d.error}" if d.error else ""))
