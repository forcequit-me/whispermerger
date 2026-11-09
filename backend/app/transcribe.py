"""WhisperX wrapper — the chosen Whisper model (large-v3 by default) + wav2vec2 forced alignment (FR-TR-1/2/3).

float16, VAD on (WhisperX strips silence at transcription), cache-aware (§8.8). This is the
only GPU-touching module. whisperx is imported lazily so the rest of the package imports
without it installed (e.g. for the GPU-independent tests).

Output shape matches what merge.py consumes:
    {"language": "en", "segments": [{"start","end","text","words":[{word,start,end}]}]}
"""

from __future__ import annotations

import importlib.util
import os
import threading
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

from .cache import Cache, cache_key
from . import modelstore
from .modelstore import EXTRA_ALIGN_MODELS   # extra word-timing models (e.g. Swedish)
from .punctmerge import merge_punctuation


def _ensure_cuda_dlls() -> None:
    """ctranslate2 4.4.0 needs the cuDNN 8 / cuBLAS / nvRTC DLLs on the Windows search
    path. torch bundles cuDNN 9, so we install the cu12 cuDNN-8 wheels and register their
    bin dirs here. No-op off Windows or if the packages aren't present (no-GPU test imports)."""
    if not hasattr(os, "add_dll_directory"):
        return
    for pkg in ("nvidia.cudnn", "nvidia.cublas", "nvidia.cuda_nvrtc"):
        spec = importlib.util.find_spec(pkg)
        if spec and spec.submodule_search_locations:
            bin_dir = Path(spec.submodule_search_locations[0]) / "bin"
            if bin_dir.is_dir():
                os.add_dll_directory(str(bin_dir))
                # ctranslate2's native loader resolves cuDNN transitively via PATH,
                # which add_dll_directory alone doesn't cover.
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


_ensure_cuda_dlls()

COMPUTE_TYPE = "float16"
DEVICE = "cuda"
# Transcription batch size (WhisperX batched inference). Higher = more GPU throughput per pass,
# at the cost of VRAM. 16 is safe on 12 GB / large-v3 fp16; bump (24, 32) and validate no OOM on
# a real run. Not part of cache_key — output is identical regardless, so changing it never busts
# the cache.
BATCH_SIZE = int(os.environ.get("WHISPERMERGER_BATCH_SIZE", "16"))
# The batch size actually used is BATCH_SIZE or less: whatever fits the VRAM left once the model is
# loaded (#37). Measured on a 3080 Ti, 2026-11-09: on top of the loaded model a batch needs at most
# ~0.4 GB per item (large-v3 at 16: 5.8 GB, at 4: 0.6 GB). A batch that doesn't fit spills into
# system memory and runs 4x+ slower, while a smaller one costs far less (8: +22%, 4: +77%). On a free
# 12 GB card this still picks 16; with another app holding VRAM it shrinks instead of spilling.
GB_PER_BATCH_ITEM = 0.4
VRAM_MARGIN_GB = 0.5               # left for the desktop and other apps' small changes


def _fit_batch(cap: int) -> int:
    """The largest batch up to `cap` that fits the free VRAM now (with the model already loaded)."""
    try:
        import torch
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
    except Exception:                  # no CUDA: nothing to fit
        return cap
    return max(1, min(cap, int((free_gb - VRAM_MARGIN_GB) / GB_PER_BATCH_ITEM)))
ALIGN_TAG = "wav2vec2"  # alignment method, part of the cache key (FR-CA-2)
# Optional language override: set WHISPERMERGER_LANGUAGE=en to skip auto-detect (and the
# per-language align-model download it can trigger). Unset = auto-detect (original behavior).
LANGUAGE = os.environ.get("WHISPERMERGER_LANGUAGE") or None

_torch_load_patched = False


def _allow_full_torch_load() -> None:
    """torch 2.6 flipped torch.load to weights_only=True, which rejects the omegaconf
    globals in whisperx's pyannote VAD / wav2vec2 checkpoints. These come from trusted HF
    repos, so load them in full. Idempotent; applied before any model load."""
    global _torch_load_patched
    if _torch_load_patched:
        return
    import torch
    _orig = torch.load

    def _full_load(*args, **kwargs):
        # Force, not setdefault: lightning's cloud_io passes weights_only=True explicitly.
        kwargs["weights_only"] = False
        return _orig(*args, **kwargs)

    torch.load = _full_load
    _torch_load_patched = True


# CTranslate2 keeps a model's VRAM after Python lets go of it, until unload_model() is called. So
# without an explicit unload, every model loaded after the first in a process (the Portugal merge's
# large-v3 pass, resume after Pause, each Benchmark model) ran ~10x slower: 66.6 s vs 6.5 s for the
# same 7 min clip on a free 3080 Ti, 2026-11-09. Jobs share no GPU lock, so the hard unload only
# happens when no other thread is mid-transcription; otherwise the old behaviour (drop references).
_loaded: list = []                 # pipelines _load_model built and hasn't unloaded yet
_gpu_users: dict[int, int] = {}    # thread id -> transcriptions it has in progress
_gpu_users_lock = threading.Lock()


@contextmanager
def gpu_user():
    """Mark this thread as using the loaded models, so another thread won't unload them under it."""
    tid = threading.get_ident()
    with _gpu_users_lock:
        _gpu_users[tid] = _gpu_users.get(tid, 0) + 1
    try:
        yield
    finally:
        with _gpu_users_lock:
            _gpu_users[tid] -= 1
            if not _gpu_users[tid]:
                del _gpu_users[tid]


def _release_loaded() -> None:
    """Unload the speech models built so far, for real, if no other thread is using them."""
    with _gpu_users_lock:
        others = any(tid != threading.get_ident() for tid in _gpu_users)
    while _loaded:
        pipe = _loaded.pop()
        if not others:
            try:
                pipe.model.model.unload_model()
            except Exception:
                pass


# One model built at a time, process-wide. transformers (4.57) builds every model empty on the "meta"
# device by patching torch.nn.Module.register_parameter for the WHOLE process, then fills the
# weights in. A model built on another thread meanwhile comes out with empty (meta) layers, and two
# overlapping patches can restore each other out of order. The server builds models on several
# threads (jobs, batch, Benchmark), and a first run in a new language failed with "Cannot copy out
# of meta tensor" while a retry worked (2026-11-09). So every model load takes this lock.
MODEL_LOAD_LOCK = threading.RLock()


@lru_cache(maxsize=1)
def _load_model(model_name: str, compute_type: str, device: str):
    with MODEL_LOAD_LOCK:
        return _build_model(model_name, compute_type, device)


def _build_model(model_name: str, compute_type: str, device: str):
    import whisperx
    _release_loaded()                  # the cache is replacing its model: free the old one's VRAM
    _allow_full_torch_load()
    # A catalog model may name its own repo and precision (e.g. an int8 conversion).
    compute_type = modelstore.WHISPER_MODELS.get(model_name, {}).get("compute_type", compute_type)
    pipe = whisperx.load_model(modelstore.model_path(model_name), device, compute_type=compute_type)
    # A conversion without preprocessor_config.json gets faster-whisper's 80-band default; large-v3
    # derivatives (e.g. CAMOES int8) need 128. Take the band count from the model itself.
    wm = pipe.model
    if wm.feat_kwargs.get("feature_size", 80) != wm.model.n_mels:
        from faster_whisper.feature_extractor import FeatureExtractor
        wm.feat_kwargs["feature_size"] = wm.model.n_mels
        wm.feature_extractor = FeatureExtractor(**wm.feat_kwargs)
    _loaded.append(pipe)
    # Measured once, right after loading: CTranslate2 keeps memory it has used, so free VRAM read
    # later would shrink after every run. ponytail: an app grabbing VRAM mid-run isn't seen until the
    # model reloads (next run after a Pause, a model switch or a restart).
    pipe.fit_batch = _fit_batch(BATCH_SIZE)
    if pipe.fit_batch < BATCH_SIZE:
        print(f"[gpu] {model_name}: batch size {pipe.fit_batch} (not {BATCH_SIZE}) to fit the free GPU "
              "memory; close other GPU apps for full speed", flush=True)
    return pipe


@lru_cache(maxsize=4)
def _load_align(language: str, device: str):
    import whisperx
    with MODEL_LOAD_LOCK:
        _allow_full_torch_load()
        try:
            return whisperx.load_align_model(language_code=language, device=device,
                                             model_name=EXTRA_ALIGN_MODELS.get(language))
        except (RuntimeError, NotImplementedError) as exc:
            if "meta tensor" not in str(exc):
                raise
            # Safety net for the case above: the model came out with empty layers. A second load
            # worked when this happened, so try once more, and say so in the server console.
            import traceback
            print(f"[models] loading the {language} word-timing model left empty (meta) weights; "
                  "loading it again", flush=True)
            traceback.print_exc()
            return whisperx.load_align_model(language_code=language, device=device,
                                             model_name=EXTRA_ALIGN_MODELS.get(language))


# Per-chunk progress while a track transcribes (#33). The pipeline sets a callback for its thread;
# transcribe_track reports the share of VAD chunks done (0..1) through it.
_chunk = threading.local()


@contextmanager
def chunk_progress(fn: Optional[Callable[[float], None]]):
    """Report this thread's transcription progress (0..1, per VAD chunk) to `fn` while inside (None: off)."""
    prev = getattr(_chunk, "fn", None)
    _chunk.fn = fn
    try:
        yield
    finally:
        _chunk.fn = prev


def _transcribe_counted(pipe, audio, lo: float = 0.0, hi: float = 1.0, **kw) -> dict:
    """pipe.transcribe(audio, **kw), reporting chunk progress scaled into [lo, hi]. WhisperX's loop
    calls `self.__call__(chunks)`, so a wrapper set on the instance sees each chunk as it finishes,
    without copying WhisperX's loop or parsing its printed "Progress: X%"."""
    if "batch_size" in kw:             # never more than fits the VRAM (#37)
        kw["batch_size"] = min(kw["batch_size"], getattr(pipe, "fit_batch", kw["batch_size"]))
    fn = getattr(_chunk, "fn", None)
    if fn is None:
        return pipe.transcribe(audio, **kw)
    orig = pipe.__call__

    def counted(inputs, *a, **k):
        items = list(inputs)                       # the VAD chunks (views into the audio, no copy)
        n = max(len(items), 1)
        # Hand them over as a generator: given a list, the transformers pipeline computes every
        # chunk before returning anything, and the progress would all land at the end.
        for i, out in enumerate(orig((x for x in items), *a, **k), 1):
            fn(lo + (hi - lo) * i / n)
            yield out

    pipe.__call__ = counted
    try:
        return pipe.transcribe(audio, **kw)
    finally:
        del pipe.__call__


def has_align_model(language: str) -> bool:
    """Whether word-level timing is available for this language code."""
    from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH
    return (language in EXTRA_ALIGN_MODELS or language in DEFAULT_ALIGN_MODELS_HF
            or language in DEFAULT_ALIGN_MODELS_TORCH)


def unload_models() -> None:
    """Drop the cached ASR + alignment models and hand their VRAM back to CUDA. Called when a run
    is paused so a paused job stops occupying the GPU; the models reload lazily on the next
    transcribe. Must run on the worker thread that owns the GPU (where the models were created)."""
    _release_loaded()
    _load_model.cache_clear()
    _load_align.cache_clear()
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def transcribe_track(audio_path: str | Path, **kw) -> dict:
    """Transcribe + align one mono/16k WAV (see _transcribe_track), counted as a GPU user meanwhile
    so another thread can't unload the models under it."""
    with gpu_user():
        return _transcribe_track(audio_path, **kw)


def _transcribe_track(
    audio_path: str | Path,
    *,
    model_name: Optional[str] = None,
    compute_type: str = COMPUTE_TYPE,
    device: str = DEVICE,
    batch_size: int = BATCH_SIZE,
    cache: Optional[Cache] = None,
    language: Optional[str] = LANGUAGE,
) -> dict:
    """Transcribe + align one mono/16k WAV. Returns the WhisperX output dict (cache-aware).
    `language` (e.g. "en") forces both transcription and alignment to that code; None
    auto-detects (and may fetch that language's align model on first use). `model_name` None =
    the speech model chosen in the Models tab."""
    import whisperx

    model_name = model_name or modelstore.active_whisper()
    # Auto-detect with a Portuguese default: the default is part of the key, since it decides the model.
    pt_default = modelstore.pt_default() if language is None else None

    # A forced language with no alignment model caches under its own tag, so adding a model for
    # it later re-aligns instead of reusing the unaligned result.
    # ponytail: auto-detect can't know the language up front, so an unaligned auto-detect result
    # stays cached under the normal tag; Clear cache re-runs it if a model is added later.
    base_lang = modelstore.LANGUAGE_VARIANTS.get(language, language)
    tag = ALIGN_TAG if (base_lang is None or has_align_model(base_lang)) else "none"
    punct = modelstore.pt_punctuation() and "pt-PT" in (language, pt_default)
    if punct:                                    # a merged result isn't a CAMOES-only one
        tag += "+punct"
    key_lang = language or (f"auto:{pt_default}" if pt_default else None)
    if language == "pt-PT":                      # European Portuguese runs on CAMOES, whatever is chosen
        model_name = modelstore.portugal_model()
    key = cache_key(audio_path, model_name, tag, key_lang) if cache else None
    if cache and (hit := cache.get(key)) is not None:
        return hit

    audio = whisperx.load_audio(str(audio_path))
    if pt_default:                                # find out first whether it's Portuguese at all
        modelstore.ensure(modelstore.whisper_entry(model_name))
        if _load_model(model_name, compute_type, device).detect_language(audio) == "pt":
            language = pt_default
    if language == "pt-PT":
        model_name = modelstore.portugal_model()
    whisper_lang = modelstore.LANGUAGE_VARIANTS.get(language, language)
    if whisper_lang not in (None, "en") and modelstore.english_only(model_name):
        raise ValueError(f"Whisper {model_name} only transcribes English. Choose another speech model "
                         f"in the Models tab, or set Language to English or Auto-detect.")

    modelstore.ensure(modelstore.whisper_entry(model_name))   # downloads visibly if missing
    model = _load_model(model_name, compute_type, device)
    two_pass = punct and language == "pt-PT"
    result = _transcribe_counted(model, audio, 0.0, 0.5 if two_pass else 1.0,
                                 batch_size=batch_size, language=whisper_lang)  # VAD on by default
    if two_pass:                                 # #23: large-v3's capitals and punctuation on CAMOES's words
        unload_models()                          # one model in VRAM at a time (12 GB cards)
        modelstore.ensure(modelstore.whisper_entry("large-v3"))
        v3 = _transcribe_counted(_load_model("large-v3", compute_type, device), audio, 0.5, 1.0,
                                 batch_size=batch_size, language="pt")
        texts = merge_punctuation([s["text"] for s in result["segments"]],
                                  " ".join(s["text"] for s in v3["segments"]))
        for s, t in zip(result["segments"], texts):
            s["text"] = t

    detected = result.get("language", "en")
    if has_align_model(detected):
        modelstore.ensure(modelstore.align_entry(detected))
        align_model, metadata = _load_align(detected, device)
        segments = whisperx.align(
            result["segments"], align_model, metadata, audio, device,
            return_char_alignments=False,
        )["segments"]
    else:
        # No word-timing model for this language: keep Whisper's line timings rather than fail
        # the run. Lines have no words, so SRT splits inside a long line are estimated.
        print(f"[align] no word-timing model for language '{detected}'; keeping line timings only",
              flush=True)
        segments = [{"start": s["start"], "end": s["end"], "text": s["text"], "words": []}
                    for s in result["segments"]]

    # Label results with the variant ("pt-PT" / "pt-BR") when one was chosen or defaulted to.
    output = {"language": language if language in modelstore.LANGUAGE_VARIANTS else detected,
              "segments": segments}
    if cache:
        cache.put(key, output)
    return output
