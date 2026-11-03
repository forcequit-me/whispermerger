"""Compare Portugal-run output: CAMOES alone, large-v3 alone, and the #23 merge.

    python compare_pt.py recording.wav [reference.txt] [--cpu]

Runs each model once (the merge reuses both outputs), then prints word errors against the
reference (when given), twice: counting accents (so Brazilian "econômico" for European
"económico" is an error) and ignoring them (hearing only). Hyphenated words count as their
parts, so "quinta feira" vs "quinta-feira" isn't an error. Also punctuation counts and time. --cpu runs int8 on the CPU (slow; for
machines without CUDA).
"""

import re
import sys
import time

from app import modelstore, transcribe
from app.punctmerge import _key, merge_punctuation


def _words(text: str, accents: bool) -> list[str]:
    # \w+ splits hyphenated words into their parts; _key drops case, punctuation and accents
    toks = re.findall(r"[\w']+", text.lower())
    return toks if accents else [k for k in map(_key, toks) if k]


def word_errors(hyp: str, ref: str, accents: bool = False) -> float:
    """Word error rate over words (Levenshtein), ignoring case and punctuation; accents optional."""
    h, r = _words(hyp, accents), _words(ref, accents)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / max(len(r), 1)


def main(argv: list[str]) -> None:
    cpu = "--cpu" in argv
    args = [a for a in argv if a != "--cpu"]
    audio_path, ref = args[0], (open(args[1], encoding="utf-8-sig").read() if len(args) > 1 else None)
    device, compute = ("cpu", "int8") if cpu else (transcribe.DEVICE, transcribe.COMPUTE_TYPE)
    if cpu:   # catalog precisions like int8_float16 need a GPU
        for m in modelstore.WHISPER_MODELS.values():
            m.pop("compute_type", None)
    import whisperx
    audio = whisperx.load_audio(audio_path)

    runs = {}
    for name in (modelstore.portugal_model(), "large-v3"):
        modelstore.ensure(modelstore.whisper_entry(name))
        t = time.perf_counter()
        out = transcribe._load_model(name, compute, device).transcribe(
            audio, batch_size=transcribe.BATCH_SIZE, language="pt")
        runs[name] = ([s["text"].strip() for s in out["segments"]], time.perf_counter() - t)
        transcribe.unload_models()
    (camoes, t_c), (v3, t_v) = runs.values()
    merged = merge_punctuation(camoes, " ".join(v3))

    rows = [("CAMOES alone (setting off)", camoes, t_c), ("large-v3 alone", v3, t_v),
            ("CAMOES + large-v3 punctuation (setting on)", merged, t_c + t_v)]
    for label, segs, secs in rows:
        text = " ".join(segs)
        wer = (f"{word_errors(text, ref, accents=True):.1%} counting accents, "
               f"{word_errors(text, ref):.1%} ignoring accents") if ref else "n/a (no reference)"
        punct = len(re.findall(r"[.,;:!?]", text))
        print(f"## {label}\n- time: {secs:.1f} s\n- word errors: {wer}\n- punctuation marks: {punct}\n\n{text}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
