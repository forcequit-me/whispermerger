"""WhisperMerger build-plan step-1 GPU smoke test (PRD §11.3 gate).

Proves the native-Windows WhisperX environment works end to end:
  1. whisperx imports and torch is still the cu124 build on GPU
  2. large-v3 loads into VRAM (cuda / float16) without OOM
  3. the wav2vec2 forced-alignment pass runs and emits per-word timestamps

Usage:
    python backend/smoke_test.py [path/to/clip.wav]

With a short real speech clip you get real word timestamps. With no argument a few
seconds of tone is generated via ffmpeg — that confirms the pipeline executes without
crashing, but produces no words (word accuracy is validated later on a real stem).
"""

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make `app` importable
from app.transcribe import _allow_full_torch_load

DEVICE = "cuda"
COMPUTE_TYPE = "float16"
MODEL = "large-v3"


def make_tone_clip() -> str:
    """Generate a 4s tone wav (mono/16k/pcm_s16le) so the pipeline has audio to run on."""
    path = Path(tempfile.gettempdir()) / "whispermerger_smoke_tone.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(path)],
        check=True, capture_output=True,
    )
    return str(path)


def main() -> int:
    import torch
    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        return 1
    print(f"gpu: {torch.cuda.get_device_name(0)}")

    import whisperx
    print(f"whisperx imported: {getattr(whisperx, '__version__', 'unknown')}")

    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        synthetic = False
    else:
        audio_path = make_tone_clip()
        synthetic = True
        print(f"no clip given -> generated tone at {audio_path} (expect 0 words)")

    print(f"loading {MODEL} on {DEVICE}/{COMPUTE_TYPE} ...")
    _allow_full_torch_load()  # torch 2.6 weights_only fix for whisperx checkpoints
    model = whisperx.load_model(MODEL, DEVICE, compute_type=COMPUTE_TYPE)

    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)
    lang = result.get("language", "en")
    print(f"transcribed: language={lang}, segments={len(result['segments'])}")

    align_model, metadata = whisperx.load_align_model(language_code=lang, device=DEVICE)
    aligned = whisperx.align(
        result["segments"], align_model, metadata, audio, DEVICE,
        return_char_alignments=False,
    )

    words = [w for seg in aligned["segments"] for w in seg.get("words", [])]
    print(f"aligned words: {len(words)}")
    for w in words[:20]:
        print(f"  {w.get('start'):>8} {w.get('end'):>8}  {w.get('word')}")

    if synthetic:
        print("\nPASS (synthetic): import + GPU load + transcribe + align ran without error.")
        print("Run again with a real speech clip to see actual word timestamps.")
    elif words:
        print("\nPASS: word-level timestamps produced.")
    else:
        print("\nWARN: ran clean but produced 0 words — was the clip actual speech?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
