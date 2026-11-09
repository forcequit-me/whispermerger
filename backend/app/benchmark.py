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
from .transcribe import BATCH_SIZE, COMPUTE_TYPE, DEVICE, _load_model, gpu_user, unload_models

SENTENCE = ("Let's move the quarterly review to Thursday afternoon, and make sure the budget numbers "
            "are final before then. If anything changes, send me a message and I'll update the calendar.")

_SPEAK = (
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, "
    "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono); "
    "$s.SetOutputToWaveFile($env:WM_BENCH_WAV, $f); $s.Speak($env:WM_BENCH_TEXT); $s.Dispose()"
)


# European Portuguese, with words Portugal and Brazil spell differently (económico/econômico,
# factos/fatos, contactos/contatos, António/Antônio), so the match shows which spelling a model writes.
PT_SENTENCE = ("Bom dia a todos. A reunião do conselho foi adiada para quinta-feira à tarde, por causa do "
               "relatório económico. O António vai apresentar os factos, e a equipa deve enviar os "
               "contactos até amanhã.")

# Windows' newer (OneCore) voices, e.g. "Microsoft Helia" for pt-PT, only exist through WinRT.
_SPEAK_ONECORE = r"""
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media.SpeechSynthesis, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.DataReader, Windows.Storage.Streams, ContentType = WindowsRuntime]
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and
  $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($op, [Type]$t) { $k = $asTask.MakeGenericMethod($t).Invoke($null, @($op)); $k.Wait(-1) | Out-Null; $k.Result }
$voice = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices | Where-Object { $_.Language -eq $env:WM_BENCH_LANG } | Select-Object -First 1
if (-not $voice) { exit 3 }
$s = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer; $s.Voice = $voice
$st = Await ($s.SynthesizeTextToStreamAsync($env:WM_BENCH_TEXT)) ([Windows.Media.SpeechSynthesis.SpeechSynthesisStream])
$r = New-Object Windows.Storage.Streams.DataReader($st.GetInputStreamAt(0)); $n = [uint32]$st.Size
$null = Await ($r.LoadAsync($n)) ([uint32]); $b = New-Object byte[] $n; $r.ReadBytes($b)
[IO.File]::WriteAllBytes($env:WM_BENCH_WAV, $b)
"""


def synthesize(dest: Path, text: str = SENTENCE) -> Path:
    """Speak `text` with the default Windows voice into a 16 kHz mono wav."""
    import os
    subprocess.run(["powershell", "-NoProfile", "-Command", _SPEAK], check=True, capture_output=True,
                   timeout=60, env={**os.environ, "WM_BENCH_WAV": str(dest), "WM_BENCH_TEXT": text})
    return dest


def synthesize_onecore(dest: Path, text: str, lang: str) -> Optional[Path]:
    """Speak `text` with an installed Windows voice for `lang` (e.g. "pt-PT"); None if there isn't one."""
    import os
    r = subprocess.run(["powershell", "-NoProfile", "-Command", _SPEAK_ONECORE], capture_output=True, timeout=60,
                       env={**os.environ, "WM_BENCH_WAV": str(dest), "WM_BENCH_TEXT": text, "WM_BENCH_LANG": lang})
    return dest if r.returncode == 0 and dest.exists() else None


def _words(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower())          # \w keeps accents, so spelling counts


def match_score(heard: str, expected: str = SENTENCE) -> float:
    """Share of the expected words heard, in order (0-1), ignoring case and punctuation."""
    return difflib.SequenceMatcher(None, _words(expected), _words(heard)).ratio()


def _tests(tmp: Path, names: list[str]) -> list[tuple]:
    """(label, expected text, audio, Whisper language, models) for each test sentence. European
    Portuguese runs only when a European Portuguese model is downloaded and a pt-PT voice exists."""
    import whisperx
    W = modelstore.WHISPER_MODELS
    tests = [("English", SENTENCE, whisperx.load_audio(str(synthesize(tmp / "en.wav"))), "en",
              [n for n in names if "for_language" not in W.get(n, {})])]
    if any(W.get(n, {}).get("for_language") == "pt-PT" for n in names):
        wav = synthesize_onecore(tmp / "pt.wav", PT_SENTENCE, "pt-PT")
        if wav:
            tests.append(("European Portuguese", PT_SENTENCE, whisperx.load_audio(str(wav)), "pt",
                          [n for n in names if not modelstore.english_only(n)]))
    return tests


def run(progress: Callable[[float, str], None], names: Optional[list[str]] = None) -> list[dict]:
    """Benchmark each downloaded speech model (or `names`) on each test sentence it can handle.
    `progress` is the job's checkpoint, so Pause/Cancel take effect between models."""
    names = names or [n for n in modelstore.WHISPER_MODELS
                      if modelstore.is_installed(modelstore.whisper_entry(n))]
    with tempfile.TemporaryDirectory(prefix="wm_bench_") as tmp:
        progress(0.0, "Speaking the test sentences")
        tests = _tests(Path(tmp), names)
    results = []
    busy = gpu_user()                             # so another job won't unload a model mid-benchmark
    busy.__enter__()
    try:
        for i, name in enumerate(names):
            progress(i / len(names), f"Benchmarking {modelstore.whisper_entry(name).name} ({i + 1}/{len(names)})")
            unload_models()                       # every model starts cold, so load times compare
            t0 = time.perf_counter()
            model = _load_model(name, COMPUTE_TYPE, DEVICE)
            load_s = time.perf_counter() - t0
            for label, expected, audio, lang, models in tests:
                if name not in models:
                    continue
                t1 = time.perf_counter()
                out = model.transcribe(audio, batch_size=BATCH_SIZE, language=lang)
                t2 = time.perf_counter()
                heard = " ".join(s["text"].strip() for s in out["segments"])
                audio_s = len(audio) / 16000
                results.append({"test": label, "model": name, "load_s": round(load_s, 2),
                                "transcribe_s": round(t2 - t1, 2), "audio_s": round(audio_s, 1),
                                "speed_x": round(audio_s / max(t2 - t1, 1e-6), 1),
                                "match": round(match_score(heard, expected), 3), "heard": heard})
    finally:
        unload_models()                           # leave VRAM free; the chosen model loads on the next run
        busy.__exit__(None, None, None)
    return results
