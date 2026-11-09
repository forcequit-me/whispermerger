"""Compare Portugal-run output: CAMOES alone, large-v3 alone, and the #23 merge.

    python compare_pt.py recording.wav [reference.txt] [--cpu]

Runs each model once (the merge reuses both outputs), then prints word errors against the
reference (when given), twice: counting accents (so Brazilian "econômico" for European
"económico" is an error) and ignoring them (hearing only). Hyphenated words count as their
parts, so "quinta feira" vs "quinta-feira" isn't an error. Also punctuation counts, whether sentence
ends and commas land where the reference has them, and time. --cpu runs int8 on the CPU (slow; for
machines without CUDA).
"""

import difflib
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


_MARK = {".": "end", "!": "end", "?": "end", ",": "comma", ";": "comma", ":": "comma"}


def punct_placement(hyp: str, ref: str) -> dict[str, tuple[float, float]]:
    """Does punctuation land where the reference has it? For words both texts share (aligned like
    the merge, ignoring case and accents), compare the mark after each word: sentence end (.!?) or
    comma (,;:). Returns {class: (precision, recall)}: precision = share of the marks written that
    are right, recall = share of the reference's marks found."""
    def marks(text):
        out = []
        for tok in text.split():
            k = _key(tok)
            if k:
                m = re.search(r"[.!?,;:]+$", tok)
                out.append((k, _MARK.get(m.group(0)[-1]) if m else None))
        return out
    h, r = marks(hyp), marks(ref)
    tp, fp, fn = ({"end": 0, "comma": 0} for _ in range(3))
    sm = difflib.SequenceMatcher(a=[k for k, _ in h], b=[k for k, _ in r], autojunk=False)
    for a, b, n in sm.get_matching_blocks():
        for (_, hm), (_, rm) in zip(h[a:a + n], r[b:b + n]):
            if hm and hm == rm:
                tp[hm] += 1
                continue
            if hm:
                fp[hm] += 1
            if rm:
                fn[rm] += 1
    return {c: (tp[c] / max(tp[c] + fp[c], 1), tp[c] / max(tp[c] + fn[c], 1)) for c in tp}


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
        place = "".join(f"\n- {c} marks in the right place: {p:.0%} of those written, {r:.0%} of the reference's"
                        for c, (p, r) in punct_placement(text, ref).items()) if ref else ""
        print(f"## {label}\n- time: {secs:.1f} s\n- word errors: {wer}\n- punctuation marks: {punct}{place}\n\n{text}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
