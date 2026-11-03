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
