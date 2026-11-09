"""transcribe_track alignment routing — model loads stubbed (no GPU)."""

import numpy as np
import pytest
import whisperx

from app import transcribe


class _FakeModel:
    def __init__(self, lang):
        self.lang = lang

    def transcribe(self, audio, batch_size, language):
        return {"language": language or self.lang,
                "segments": [{"start": 0.5, "end": 2.0, "text": " halló heimur"}]}


def test_has_align_model():
    assert transcribe.has_align_model("en")          # torchaudio default
    assert transcribe.has_align_model("nl")          # Hugging Face default
    assert transcribe.has_align_model("sv")          # our extra model
    assert not transcribe.has_align_model("is")      # Icelandic: none


def test_language_without_align_model_keeps_line_timings(monkeypatch, tmp_path):
    monkeypatch.setattr(whisperx, "load_audio", lambda p: np.zeros(16000, dtype=np.float32))
    monkeypatch.setattr(transcribe, "_load_model", lambda *a: _FakeModel("is"))
    monkeypatch.setattr(transcribe.modelstore, "ensure", lambda e: None)    # never download in tests

    def no_align(*a):
        raise AssertionError("alignment must be skipped for a language without a model")

    monkeypatch.setattr(transcribe, "_load_align", no_align)
    for language in ("is", None):                    # forced, and auto-detected as Icelandic
        out = transcribe.transcribe_track(tmp_path / "x.wav", language=language)
        assert out == {"language": "is",
                       "segments": [{"start": 0.5, "end": 2.0, "text": " halló heimur", "words": []}]}


def test_swedish_loads_the_extra_align_model(monkeypatch):
    seen = {}
    monkeypatch.setattr(whisperx, "load_align_model",
                        lambda language_code, device, model_name=None: seen.update(model=model_name))
    transcribe._load_align.cache_clear()
    transcribe._load_align("sv", "cpu")
    transcribe._load_align.cache_clear()
    assert seen["model"] == "KBLab/wav2vec2-large-voxrex-swedish"


def test_runs_use_the_chosen_model_and_english_only_guard(monkeypatch, tmp_path):
    monkeypatch.setattr(transcribe.modelstore, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(transcribe.modelstore, "ensure", lambda e: None)
    monkeypatch.setattr(whisperx, "load_audio", lambda p: np.zeros(16000, dtype=np.float32))
    loaded = []
    monkeypatch.setattr(transcribe, "_load_model", lambda name, *a: loaded.append(name) or _FakeModel("is"))
    monkeypatch.setattr(transcribe, "_load_align", lambda *a: (_ for _ in ()).throw(AssertionError))

    assert transcribe.modelstore.active_whisper() == "large-v3"          # default
    transcribe.modelstore.set_active_whisper("small")
    transcribe.transcribe_track(tmp_path / "x.wav", language="is")
    assert loaded == ["small"]

    transcribe.modelstore.set_active_whisper("distil-large-v3.5")      # English only
    with pytest.raises(ValueError, match="only transcribes English"):
        transcribe.transcribe_track(tmp_path / "x.wav", language="nl")
    with pytest.raises(ValueError):
        transcribe.modelstore.set_active_whisper("nope")


class _DetectingModel(_FakeModel):
    def __init__(self, lang, calls):
        super().__init__(lang)
        self.calls = calls

    def detect_language(self, audio):
        return self.lang

    def transcribe(self, audio, batch_size, language):
        self.calls.append(language)
        return {"language": language or self.lang, "segments": [{"start": 0.0, "end": 1.0, "text": " olá"}]}


def test_portuguese_variants_route_model_language_and_label(monkeypatch, tmp_path):
    ms = transcribe.modelstore
    monkeypatch.setattr(ms, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(ms, "ensure", lambda e: None)
    monkeypatch.setattr(ms, "is_installed", lambda e: e.id == "whisper:camoes-int8")   # only compressed on disk
    monkeypatch.setattr(whisperx, "load_audio", lambda p: np.zeros(16000, dtype=np.float32))
    monkeypatch.setattr(transcribe, "has_align_model", lambda lang: False)            # skip word timing
    loaded, sent = [], []

    def load(name, *a):
        loaded.append(name)
        return _DetectingModel(detect, sent)

    monkeypatch.setattr(transcribe, "_load_model", load)
    detect = "pt"

    assert transcribe.transcribe_track(tmp_path / "x.wav", language="pt-PT")["language"] == "pt-PT"
    assert loaded[-1] == "camoes-int8" and sent[-1] == "pt"          # CAMOES (what's on disk), Whisper "pt"

    assert transcribe.transcribe_track(tmp_path / "x.wav", language="pt-BR")["language"] == "pt-BR"
    assert loaded[-1] == "large-v3" and sent[-1] == "pt"             # the app-wide model

    ms.set_pt_default("pt-PT")                                       # Auto-detect + Portugal default
    assert transcribe.transcribe_track(tmp_path / "x.wav")["language"] == "pt-PT"
    assert loaded[-2:] == ["large-v3", "camoes-int8"]               # detect with chosen, transcribe with CAMOES
    detect = "es"                                                    # not Portuguese: default doesn't apply
    assert transcribe.transcribe_track(tmp_path / "x.wav")["language"] == "es"
    assert loaded[-1] == "large-v3" and sent[-1] is None


class _PunctModel:
    def __init__(self, name):
        self.name = name

    def transcribe(self, audio, batch_size, language):
        text = " bom dia o crescimento económico" if self.name.startswith("camoes") else " Bom dia. O crescimento econômico."
        return {"language": language, "segments": [{"start": 0.0, "end": 2.0, "text": text}]}


def test_portugal_punctuation_is_optional(monkeypatch, tmp_path):
    ms = transcribe.modelstore
    monkeypatch.setattr(ms, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(ms, "ensure", lambda e: None)
    monkeypatch.setattr(whisperx, "load_audio", lambda p: np.zeros(16000, dtype=np.float32))
    monkeypatch.setattr(transcribe, "has_align_model", lambda lang: False)
    loaded = []
    monkeypatch.setattr(transcribe, "_load_model", lambda name, *a: loaded.append(name) or _PunctModel(name))
    monkeypatch.setattr(transcribe, "unload_models", lambda: loaded.append("unload"))

    off = transcribe.transcribe_track(tmp_path / "x.wav", language="pt-PT")       # default: CAMOES alone
    assert loaded == ["camoes"] and off["segments"][0]["text"] == " bom dia o crescimento económico"

    ms.set_pt_punctuation(True)
    loaded.clear()
    on = transcribe.transcribe_track(tmp_path / "x.wav", language="pt-PT")
    assert loaded == ["camoes", "unload", "large-v3"]      # frees CAMOES before loading large-v3
    assert on["segments"][0]["text"] == "Bom dia. O crescimento económico."     # European spelling kept

    loaded.clear()
    transcribe.transcribe_track(tmp_path / "x.wav", language="pt-BR")            # other languages untouched
    assert loaded == ["large-v3"]


def test_runs_take_turns_on_the_shared_model(monkeypatch, tmp_path):
    """Two runs at once on the shared model crashed one of them (WhisperX clears the model's
    tokenizer when a run ends). Now the second waits for its turn; a cached result doesn't wait."""
    import threading
    monkeypatch.setattr(whisperx, "load_audio", lambda p: np.zeros(16000, dtype=np.float32))
    monkeypatch.setattr(transcribe.modelstore, "ensure", lambda e: None)
    active, most, release = [0], [0], threading.Event()

    class Slow(_FakeModel):
        def transcribe(self, audio, batch_size, language):
            active[0] += 1
            most[0] = max(most[0], active[0])
            release.wait(5)
            active[0] -= 1
            return super().transcribe(audio, batch_size, language)

    monkeypatch.setattr(transcribe, "_load_model", lambda *a: Slow("is"))
    runs = [threading.Thread(target=transcribe.transcribe_track, args=(tmp_path / "x.wav",),
                             kwargs={"language": "is"}) for _ in range(2)]
    for r in runs:
        r.start()
    import time
    time.sleep(0.3)                                    # both started; one is in, one is waiting

    class Hit:                                         # a cached result returns while the GPU is busy
        def get(self, key): return {"cached": True}
    monkeypatch.setattr(transcribe, "cache_key", lambda *a: "k")
    assert transcribe.transcribe_track(tmp_path / "y.wav", language="is", cache=Hit()) == {"cached": True}

    release.set()
    for r in runs:
        r.join(5)
    assert most[0] == 1


def test_a_waiting_run_says_so(monkeypatch, tmp_path):
    """While another run has the GPU, the progress hook hears waiting=True, then a normal 0.0 when
    this run's turn starts."""
    import threading
    import time
    monkeypatch.setattr(whisperx, "load_audio", lambda p: np.zeros(16000, dtype=np.float32))
    monkeypatch.setattr(transcribe.modelstore, "ensure", lambda e: None)
    class Hooked(_FakeModel):                          # the progress hook wraps __call__
        def __call__(self, inputs, *a, **k):
            return iter(())
    monkeypatch.setattr(transcribe, "_load_model", lambda *a: Hooked("is"))
    seen = []

    def run():
        with transcribe.chunk_progress(lambda f, waiting=False: seen.append((f, waiting))):
            transcribe.transcribe_track(tmp_path / "x.wav", language="is")

    transcribe.GPU_TURN.acquire()                      # another run holds the GPU
    th = threading.Thread(target=run)
    try:
        th.start()
        time.sleep(0.3)
        waiting_seen = list(seen)
    finally:
        transcribe.GPU_TURN.release()
    th.join(5)
    assert waiting_seen == [(0.0, True)]
    assert seen[:2] == [(0.0, True), (0.0, False)]
