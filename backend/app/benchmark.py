"""Speech-model benchmark (Models tab): time each downloaded model on this machine.

A fixed sentence is spoken by the Windows voice, so models can be compared with no recording at
hand. It measures load time (reading the model into VRAM) and transcription time, and how closely
the text matches what was said. Not a full word-error-rate test, just "how fast, and does it read
right" on this GPU.
"""

from __future__ import annotations

import difflib
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

from . import modelstore
from .transcribe import BATCH_SIZE, COMPUTE_TYPE, DEVICE, _load_model, unload_models

SENTENCE = ("Let's move the quarterly review to Thursday afternoon, and make sure the budget numbers "
            "are final before then. If anything changes, send me a message and I'll update the calendar.")

_SPEAK = (
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, "
    "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono); "
    "$s.SetOutputToWaveFile($env:WM_BENCH_WAV, $f); $s.Speak($env:WM_BENCH_TEXT); $s.Dispose()"
)


def synthesize(dest: Path, text: str = SENTENCE) -> Path:
    """Speak `text` with the default Windows voice into a 16 kHz mono wav."""
    import os
    subprocess.run(["powershell", "-NoProfile", "-Command", _SPEAK], check=True, capture_output=True,
                   timeout=60, env={**os.environ, "WM_BENCH_WAV": str(dest), "WM_BENCH_TEXT": text})
    return dest


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def match_score(heard: str, expected: str = SENTENCE) -> float:
    """Share of the expected words heard, in order (0-1), ignoring case and punctuation."""
    return difflib.SequenceMatcher(None, _words(expected), _words(heard)).ratio()


def run(progress: Callable[[float, str], None], names: Optional[list[str]] = None) -> list[dict]:
    """Benchmark each downloaded speech model (or `names`). `progress` is the job's checkpoint, so
    Pause/Cancel take effect between models."""
    import whisperx

    names = names or [n for n in modelstore.WHISPER_MODELS
                      if modelstore.is_installed(modelstore.whisper_entry(n))]
    with tempfile.TemporaryDirectory(prefix="wm_bench_") as tmp:
        progress(0.0, "Speaking the test sentence")
        audio = whisperx.load_audio(str(synthesize(Path(tmp) / "sentence.wav")))
    audio_s = len(audio) / 16000
    results = []
    try:
        for i, name in enumerate(names):
            progress(i / len(names), f"Benchmarking Whisper {name} ({i + 1}/{len(names)})")
            unload_models()                       # every model starts cold, so load times compare
            t0 = time.perf_counter()
            model = _load_model(name, COMPUTE_TYPE, DEVICE)
            t1 = time.perf_counter()
            out = model.transcribe(audio, batch_size=BATCH_SIZE, language="en")
            t2 = time.perf_counter()
            heard = " ".join(s["text"].strip() for s in out["segments"])
            results.append({"model": name, "load_s": round(t1 - t0, 2), "transcribe_s": round(t2 - t1, 2),
                            "audio_s": round(audio_s, 1), "speed_x": round(audio_s / max(t2 - t1, 1e-6), 1),
                            "match": round(match_score(heard), 3), "heard": heard})
    finally:
        unload_models()                           # leave VRAM free; the chosen model loads on the next run
    return results
