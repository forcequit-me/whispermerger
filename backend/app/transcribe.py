"""WhisperX wrapper — large-v3 + wav2vec2 forced alignment (FR-TR-1/2/3).

float16, VAD on (WhisperX strips silence at transcription), cache-aware (§8.8). This is the
only GPU-touching module. whisperx is imported lazily so the rest of the package imports
without it installed (e.g. for the GPU-independent tests).

Output shape matches what merge.py consumes:
    {"language": "en", "segments": [{"start","end","text","words":[{word,start,end}]}]}
"""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .cache import Cache, cache_key


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

MODEL_NAME = "large-v3"
COMPUTE_TYPE = "float16"
DEVICE = "cuda"
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


@lru_cache(maxsize=1)
def _load_model(model_name: str, compute_type: str, device: str):
    import whisperx
    _allow_full_torch_load()
    return whisperx.load_model(model_name, device, compute_type=compute_type)


@lru_cache(maxsize=4)
def _load_align(language: str, device: str):
    import whisperx
    _allow_full_torch_load()
    return whisperx.load_align_model(language_code=language, device=device)


def transcribe_track(
    audio_path: str | Path,
    *,
    model_name: str = MODEL_NAME,
    compute_type: str = COMPUTE_TYPE,
    device: str = DEVICE,
    batch_size: int = 16,
    cache: Optional[Cache] = None,
    language: Optional[str] = LANGUAGE,
) -> dict:
    """Transcribe + align one mono/16k WAV. Returns the WhisperX output dict (cache-aware).
    `language` (e.g. "en") forces both transcription and alignment to that code; None
    auto-detects (and may fetch that language's align model on first use)."""
    import whisperx

    key = cache_key(audio_path, model_name, ALIGN_TAG, language) if cache else None
    if cache and (hit := cache.get(key)) is not None:
        return hit

    audio = whisperx.load_audio(str(audio_path))
    model = _load_model(model_name, compute_type, device)
    result = model.transcribe(audio, batch_size=batch_size, language=language)  # VAD on by default

    detected = result.get("language", "en")
    align_model, metadata = _load_align(detected, device)
    aligned = whisperx.align(
        result["segments"], align_model, metadata, audio, device,
        return_char_alignments=False,
    )

    output = {"language": detected, "segments": aligned["segments"]}
    if cache:
        cache.put(key, output)
    return output
