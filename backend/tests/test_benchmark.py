"""Benchmark runner — voice and model loads stubbed (no GPU)."""

import numpy as np
import whisperx

from app import benchmark


class _Model:
    def __init__(self, text):
        self.text = text

    def transcribe(self, audio, batch_size, language):
        return {"segments": [{"text": " " + self.text}]}


def test_benchmark_times_each_model_and_scores_the_text(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "synthesize", lambda dest, text=benchmark.SENTENCE: dest)
    monkeypatch.setattr(whisperx, "load_audio", lambda p: np.zeros(16000 * 10, dtype=np.float32))
    monkeypatch.setattr(benchmark, "unload_models", lambda: None)
    heard = {"small": benchmark.SENTENCE, "medium": "lets move the quarterly review"}
    monkeypatch.setattr(benchmark, "_load_model", lambda name, *a: _Model(heard[name]))
    steps = []
    out = benchmark.run(lambda f, m: steps.append(m), names=["small", "medium"])
    assert [r["model"] for r in out] == ["small", "medium"]
    assert out[0]["match"] == 1.0 and out[1]["match"] < 0.5
    assert all(r["audio_s"] == 10.0 and r["load_s"] >= 0 and r["speed_x"] > 0 for r in out)
    assert steps[1:] == ["Benchmarking Whisper small (1/2)", "Benchmarking Whisper medium (2/2)"]


def test_match_score_ignores_case_and_punctuation():
    assert benchmark.match_score(benchmark.SENTENCE.upper().replace(",", "")) == 1.0
