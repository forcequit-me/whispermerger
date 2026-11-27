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
# 12 GB card this picks 14; with another app holding VRAM it shrinks instead of spilling.
GB_PER_BATCH_ITEM = 0.37          # measured 0.36 at 16 (less per item at smaller batches)
# Left for the desktop and other apps' small changes (0.3 GB), plus ~0.6 GB the speech model takes
# on its first run, after free VRAM was read. At 0.3 a free 12 GB card picked 16, and with the
# word-timing model in, batch 16 spilled 0.4-0.7 GB and ran ~2x slower (CAMOES and large-v3 alike;
# 14 didn't), 2026-11-09.
VRAM_MARGIN_GB = 0.9
# The word-timing model loads after the speech model and stays on the GPU (one at a time, see
# _load_align): leave room for it, or a batch sized before it loaded spills once it's there.
# Measured on a free 3080 Ti: speech model + batch 16 + one word-timing model leaves ~0.5 GB.
ALIGN_MODEL_GB = 1.3


def _fit_batch(cap: int) -> int:
    """The largest batch up to `cap` that fits the free VRAM now (with the speech model loaded),
    keeping room for one word-timing model."""
    try:
        import torch
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        # a word-timing model already loaded is torch memory, already out of free_gb
        reserve = max(0.0, ALIGN_MODEL_GB - torch.cuda.memory_allocated() / 1e9)
    except Exception:                  # no CUDA: nothing to fit
        return cap
    return max(1, min(cap, int((free_gb - VRAM_MARGIN_GB - reserve) / GB_PER_BATCH_ITEM)))
ALIGN_TAG = "wav2vec2"  # alignment method, part of the cache key (FR-CA-2)
NO_WORDS_TAG = "no-words"  # in its place for a run asked to go without word timing (#39)
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

# One transcription at a time, process-wide. The speech model is shared, and a run changes it while
# it works: WhisperX sets its tokenizer for the run's language and clears it at the end, and
# _transcribe_counted puts a progress hook on it. Two runs at once (a batch plus a Transcribe job,
# or the CLI/MCP while the app is busy) crashed one of them ("'NoneType' object has no attribute
# 'sot_sequence'", then "no attribute '__call__'"), 2026-11-09. One card can't run two side by side
# anyway, so the second waits for its turn. Cache hits don't wait.
GPU_TURN = threading.Lock()


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


# One word-timing model on the GPU at a time (~1.3 GB each): several cached together pushed a
# 12 GB card into system memory mid-batch, several times slower (2026-11-09). Another language
# reloads from disk in a few seconds.
@lru_cache(maxsize=1)
def _load_align(language: str, device: str):
    import whisperx
    with MODEL_LOAD_LOCK:
        _drop_align()                      # drop the previous language's model before loading this one
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


def _drop_align() -> None:
    """Take the word-timing model off the GPU."""
    _load_align.cache_clear()
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# Per-chunk progress while a track transcribes (#33). The pipeline sets a callback for its thread;
# transcribe_track reports the share of VAD chunks done (0..1) through it, fn(0.0, waiting=True)
# while it waits for another run to finish with the GPU (GPU_TURN), and fn(0.0, detected="English 97%")
# once Auto-detect has chosen the language.
_chunk = threading.local()


@contextmanager
def chunk_progress(fn: Optional[Callable[..., None]]):
    """Report this thread's transcription progress (0..1, per VAD chunk) to `fn` while inside (None: off)."""
    prev = getattr(_chunk, "fn", None)
    _chunk.fn = fn
    try:
        yield
    finally:
        _chunk.fn = prev


def cache_tally() -> tuple[int, int]:
    """(saved results reused, tracks transcribed fresh) by transcribe_track on this thread so far."""
    return getattr(_chunk, "hits", 0), getattr(_chunk, "fresh", 0)


def reused_since(tally: tuple[int, int]) -> bool:
    """Whether everything transcribe_track did on this thread since `tally` came from the cache,
    so a run can say "reused saved result" where "done in 0s" would read as if nothing happened."""
    hits, fresh = cache_tally()
    return hits > tally[0] and fresh == tally[1]


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


@contextmanager
def _gpu_turn():
    """This thread's turn on the shared models (GPU_TURN), counted as a GPU user meanwhile so
    another thread can't unload them under it. While another run has the GPU, say so, every half
    second: that report is also where a waiting run is paused (it blocks) or canceled (it raises),
    while it holds nothing yet."""
    fn = getattr(_chunk, "fn", None)
    waited = not GPU_TURN.acquire(blocking=not fn)
    if waited:
        _chunk.waiting = True                         # unload_models leaves the other run's models alone
        try:
            while True:
                fn(0.0, waiting=True)
                if GPU_TURN.acquire(timeout=0.5):
                    break
        finally:
            _chunk.waiting = False
    try:
        with gpu_user():
            if waited:
                fn(0.0)                               # our turn: back to the normal label
            yield
    finally:
        GPU_TURN.release()


# Auto-detect listens to speech from across the file. It used to take the first 30 s, so a
# recording that opens with music or silence got a random guess: an English 2-hour stream came out
# as Nynorsk at 0.36, which pulled a 3.9 GB word-timing model mid-job (#38).
DETECT_WINDOWS = 4                 # spots spread over the file, up to 30 s of speech from each
DETECT_PROBE_S = 45                # audio searched for speech at each spot: the VAD costs ~0.5 s per spot,
                                   # whatever the file's length (2 s in all for a 2-hour file, 3080 Ti PC)
DETECT_MIN_SPEECH_S = 5            # a spot with less speech than this doesn't vote
# A guess outside the languages the user says they use (Models tab) must be at least this sure;
# under it, the best-scoring language of theirs is used instead.
OUTSIDE_LIST_MIN = 0.5
# Auto-detect never downloads a word-timing model bigger than this for a language outside that
# list on its own: the run stops and asks (see asking()). The usual ones are about 1.26 GB and
# download without asking (at 1 GB nearly every language asked on first use); the unusually big
# ones ask (the Norwegian ones are 3.9 GB).
ASK_OVER_MB = 1500
ASK_CHOICES = {"download": "download the model and continue",
               "skip": "continue without word timing for this file (line timings only)",
               "language": "use a different language instead (give its code, e.g. en)"}


def speech_windows(audio) -> list:
    """Up to DETECT_WINDOWS clips of speech (30 s at most each) from spots spread evenly over
    `audio` (mono/16k samples). Empty when no spot has enough speech."""
    import numpy as np
    import torch
    from .diarize import SR, speech_spans
    probe = DETECT_PROBE_S * SR
    n = max(1, min(DETECT_WINDOWS, len(audio) // probe))
    part = len(audio) // n
    out = []
    for i in range(n):
        start = i * part + max(0, part - probe) // 2          # the middle of each part
        chunk = audio[start:start + min(part, probe)]
        speech = [chunk[int(s * SR):int(e * SR)] for s, e in speech_spans(torch.from_numpy(chunk))]
        if sum(len(x) for x in speech) >= DETECT_MIN_SPEECH_S * SR:
            out.append(np.concatenate(speech)[:30 * SR])
    return out


def language_probs(pipe, audio) -> dict[str, float]:
    """Each language's probability for `audio`, averaged over its speech windows (its first 30 s
    when no speech was found, as before)."""
    from whisperx.audio import N_SAMPLES, log_mel_spectrogram
    windows = speech_windows(audio) or [audio[:N_SAMPLES]]
    n_mels = pipe.model.feat_kwargs.get("feature_size") or 80
    probs: dict[str, float] = {}
    for w in windows:
        mel = log_mel_spectrogram(w, n_mels=n_mels, padding=N_SAMPLES - len(w))
        for token, p in pipe.model.model.detect_language(pipe.model.encode(mel))[0]:
            probs[token[2:-2]] = probs.get(token[2:-2], 0.0) + p / len(windows)   # "<|en|>" -> "en"
    return probs


def choose_language(probs: dict[str, float], mine: list[str]) -> str:
    """The language to run with: the top guess, unless the user listed their languages (`mine`)
    and the guess is outside them without being sure; then the best-scoring one of theirs."""
    top = max(probs, key=probs.get)
    if not mine or top in mine or probs[top] >= OUTSIDE_LIST_MIN:
        return top
    return max(mine, key=lambda c: probs.get(c, 0.0))


# A question for the user mid-run. Whoever runs the work (a job, a batch) sets the way to ask for
# its thread; transcribe_track asks through it before a big download nobody chose.
_asker = threading.local()


@contextmanager
def asking(fn: Optional[Callable[[dict], tuple]]):
    """Route this thread's questions to `fn(question) -> (choice, language)` while inside. `fn`
    blocks until the user answers (choice is one of ASK_CHOICES; language is set for "language")."""
    prev = getattr(_asker, "fn", None)
    _asker.fn = fn
    try:
        yield
    finally:
        _asker.fn = prev


def _ask_download(language: str, confidence: float, entry, probs: dict[str, float]) -> tuple:
    """Ask before downloading `entry` for an auto-detected language. Nobody to ask (a script calling
    transcribe_track directly): no download, line timings only."""
    fn = getattr(_asker, "fn", None)
    if fn is None:
        return "skip", None
    name = modelstore.LANG_NAMES.get(language, language)
    size = f"{entry.size_mb / 1000:.1f} GB"
    others = sorted((c for c in probs if c != language), key=probs.get, reverse=True)[:3]
    return fn({
        "text": f"This sounds like {name} ({confidence:.0%} sure). Its word-timing model is a {size} "
                "download that isn't on this computer yet.",
        "language": language, "confidence": round(confidence, 3), "size_mb": entry.size_mb,
        "others": [{"language": c, "confidence": round(probs[c], 3)} for c in others],
        "choices": ASK_CHOICES})


def _note_detected(language: str, confidence: float) -> None:
    """Tell the run's progress line what Auto-detect chose, and the server console."""
    text = f"{modelstore.LANG_NAMES.get(language, language)} {confidence:.0%}"
    print(f"[language] detected {language} ({confidence:.2f}) from speech across the file", flush=True)
    fn = getattr(_chunk, "fn", None)
    if fn:
        fn(0.0, detected=text)
    else:                                             # no per-chunk progress (Label Track)
        modelstore.report(f"detected {text}")


def unload_models() -> None:
    """Drop the cached ASR + alignment models and hand their VRAM back to CUDA. Called when a run
    is paused so a paused job stops occupying the GPU; the models reload lazily on the next
    transcribe. Must run on the worker thread that owns the GPU (where the models were created)."""
    if getattr(_chunk, "waiting", False):  # paused while waiting for its turn: nothing on the GPU is
        return                             # this run's, the models there belong to the run using them
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


def transcribe_track(
    audio_path: str | Path,
    *,
    model_name: Optional[str] = None,
    compute_type: str = COMPUTE_TYPE,
    device: str = DEVICE,
    batch_size: int = BATCH_SIZE,
    cache: Optional[Cache] = None,
    language: Optional[str] = LANGUAGE,
    word_timing: bool = True,
) -> dict:
    """Transcribe + align one mono/16k WAV. Returns the WhisperX output dict (cache-aware).
    `language` (e.g. "en") forces both transcription and alignment to that code; None
    auto-detects from speech across the file (the output then also has "language_probability").
    A chosen language fetches its align model on first use; an auto-detected one asks first when
    that download is big and the language isn't one the user listed (see asking()). `model_name`
    None = the speech model chosen in the Models tab.
    `word_timing` False skips alignment: no word-timing model is loaded, downloaded or asked
    about, lines keep Whisper's own timings without words (one line per stretch of speech, up to
    30 s, not per sentence), and the output has "word_timing": False (#39; alignment takes about
    as long as the transcription itself, or longer)."""
    import whisperx

    model_name = model_name or modelstore.active_whisper()
    auto = language is None
    # Auto-detect: what steers the guess is part of the key, since it decides the language and model.
    pt_default = modelstore.pt_default() if auto else None
    mine = modelstore.my_languages() if auto else []

    # A forced language with no alignment model caches under its own tag, so adding a model for
    # it later re-aligns instead of reusing the unaligned result.
    # ponytail: auto-detect can't know the language up front, so an unaligned auto-detect result
    # stays cached under the normal tag; Clear cache re-runs it if a model is added later.
    base_lang = modelstore.LANGUAGE_VARIANTS.get(language, language)
    tag = ALIGN_TAG if (base_lang is None or has_align_model(base_lang)) else "none"
    if not word_timing:
        # Its own tag, not "none": a run without word timing and one with it never get each
        # other's result, whatever the language and whether it has a model.
        tag = NO_WORDS_TAG
    punct = modelstore.pt_punctuation() and "pt-PT" in (language, pt_default)
    if punct:                                    # a merged result isn't a CAMOES-only one
        tag += "+punct"
    steer = "+".join(x for x in (pt_default, ",".join(mine)) if x)
    key_lang = language or (f"auto:{steer}" if steer else None)
    if language == "pt-PT":                      # European Portuguese runs on CAMOES, whatever is chosen
        model_name = modelstore.portugal_model()
    key = cache_key(audio_path, model_name, tag, key_lang) if cache else None
    if cache and (hit := cache.get(key)) is not None:
        _chunk.hits = cache_tally()[0] + 1
        return hit
    _chunk.fresh = cache_tally()[1] + 1

    audio = model = align_model = metadata = None
    try:
        audio = whisperx.load_audio(str(audio_path))
        confidence = None
        if auto and modelstore.english_only(model_name):
            language = "en"
        elif auto:
            with _gpu_turn():
                modelstore.ensure(modelstore.whisper_entry(model_name))
                probs = language_probs(_load_model(model_name, compute_type, device), audio)
            used = set(mine) | ({"pt"} if pt_default else set())      # a Portuguese default counts as using it
            language = choose_language(probs, sorted(used) if mine else [])
            confidence = probs[language]
            _note_detected(language, confidence)
            entry = modelstore.align_entry(language) if has_align_model(language) else None
            choice = None
            if (word_timing and entry and language not in used and entry.size_mb > ASK_OVER_MB
                    and not modelstore.is_installed(entry)):
                choice, other = _ask_download(language, confidence, entry, probs)   # no turn held: others run meanwhile
            if choice == "skip":
                word_timing = False
            elif choice == "language":           # the user's pick, as if chosen in a Language dropdown
                language, confidence = other, None
            elif language == "pt" and pt_default:
                language = pt_default

        # One run at a time on the shared models; while another run has the GPU, say so.
        with _gpu_turn():
            if language == "pt-PT":
                model_name = modelstore.portugal_model()
            whisper_lang = modelstore.LANGUAGE_VARIANTS.get(language, language)
            if whisper_lang != "en" and modelstore.english_only(model_name):
                raise ValueError(f"Whisper {model_name} only transcribes English. Choose another speech model "
                                 f"in the Models tab, or set Language to English or Auto-detect.")

            modelstore.ensure(modelstore.whisper_entry(model_name))   # downloads visibly if missing
            model = _load_model(model_name, compute_type, device)
            two_pass = punct and language == "pt-PT"
            result = _transcribe_counted(model, audio, 0.0, 0.5 if two_pass else 1.0,
                                         batch_size=batch_size, language=whisper_lang)  # VAD on by default
            if two_pass:                                 # #23: large-v3's capitals and punctuation on CAMOES's words
                model = None
                unload_models()                          # one model in VRAM at a time (12 GB cards)
                modelstore.ensure(modelstore.whisper_entry("large-v3"))
                v3 = _transcribe_counted(_load_model("large-v3", compute_type, device), audio, 0.5, 1.0,
                                         batch_size=batch_size, language="pt")
                texts = merge_punctuation([s["text"] for s in result["segments"]],
                                          " ".join(s["text"] for s in v3["segments"]))
                for s, t in zip(result["segments"], texts):
                    s["text"] = t

            detected = result.get("language", "en")
            entry = modelstore.align_entry(detected) if word_timing and has_align_model(detected) else None
            if entry:
                modelstore.ensure(entry)
                align_model, metadata = _load_align(detected, device)
                segments = whisperx.align(
                    result["segments"], align_model, metadata, audio, device,
                    return_char_alignments=False,
                )["segments"]
                align_model = metadata = None
                # A word-timing model bigger than the room the batch size leaves for one (the two
                # Norwegian ones are 3.9 GB) goes straight back: left in, the next file's batch
                # spilled into system memory, 63 s instead of 16 s for 8 min on a 12 GB card (#38).
                if entry.size_mb > ALIGN_MODEL_GB * 1000:
                    _drop_align()
                # Word timing leaves 1.1-1.7 GB of PyTorch's spare working memory on the GPU, on top of
                # its model. The next track's batch then spilled into system memory: 41-57 s instead of
                # 6.7 s for a 7 min track on a 12 GB card, 2026-11-09. Hand it back.
                import torch
                torch.cuda.empty_cache()
            else:
                # No word-timing model for this language (or the user chose to go without): keep
                # Whisper's line timings rather than fail the run. Lines have no words, so SRT splits
                # inside a long line are estimated.
                print(f"[align] no word-timing model {'for' if word_timing else 'used for'} language "
                      f"'{detected}'; keeping line timings only", flush=True)
                segments = [{"start": s["start"], "end": s["end"], "text": s["text"], "words": []}
                            for s in result["segments"]]

            # Label results with the variant ("pt-PT" / "pt-BR") when one was chosen or defaulted to.
            output = {"language": language if language in modelstore.LANGUAGE_VARIANTS else detected,
                      "segments": segments}
            if confidence is not None:                   # Auto-detect chose it: how sure it was
                output["language_probability"] = round(confidence, 3)
            if not word_timing:                          # skipped by choice: the option, or the "skip" answer
                output["word_timing"] = False
            if cache:
                cache.put(key, output)
            return output
    finally:
        # This call's frame can outlive it: a library that failed an optional import keeps that
        # error, and with it the whole stack that first imported the library, which is whichever
        # run came first when nothing else had (a CLI / MCP run on a fresh server). Its locals then
        # held that run's audio and word-timing model, the model on the GPU, until the server
        # restarted, whatever the caches dropped (#38). So let go of them here.
        audio = model = align_model = metadata = None
