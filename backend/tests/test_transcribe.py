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
    monkeypatch.setattr(transcribe, "language_probs", lambda pipe, audio: {"is": 0.9, "en": 0.1})
    monkeypatch.setattr(transcribe.modelstore, "SETTINGS_FILE", tmp_path / "settings.json")
    segments = [{"start": 0.5, "end": 2.0, "text": " halló heimur", "words": []}]
    assert transcribe.transcribe_track(tmp_path / "x.wav", language="is") == {"language": "is", "segments": segments}
    # auto-detected as Icelandic: the same, plus how sure the guess was
    assert transcribe.transcribe_track(tmp_path / "x.wav") == {
        "language": "is", "segments": segments, "language_probability": 0.9}


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
    monkeypatch.setattr(transcribe, "language_probs", lambda pipe, audio: {detect: 0.9})
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
    assert loaded[-1] == "large-v3" and sent[-1] == "es"


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
    before = transcribe.cache_tally()
    assert transcribe.transcribe_track(tmp_path / "y.wav", language="is", cache=Hit()) == {"cached": True}
    assert transcribe.reused_since(before)             # a run says "reused saved result" from this

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


def test_a_run_waiting_for_the_gpu_pauses_and_cancels_at_once(monkeypatch, tmp_path):
    """Waiting for the GPU reports every half second, and that report is the run's checkpoint: a
    paused job parks there (clock stopped, the running run's models left alone) and a canceled one
    stops there. Neither waits for its turn first."""
    import time
    from app.jobs import JobRegistry, JobStatus
    monkeypatch.setattr(whisperx, "load_audio", lambda p: np.zeros(16000, dtype=np.float32))
    monkeypatch.setattr(transcribe.modelstore, "ensure", lambda e: None)
    monkeypatch.setattr(transcribe, "_load_model", lambda *a: _FakeModel("is"))

    def work(progress):
        with transcribe.chunk_progress(lambda f, waiting=False: progress(f, "waiting" if waiting else "running")):
            transcribe.transcribe_track(tmp_path / "x.wav", language="is")

    def until(check):
        deadline = time.time() + 3
        while not check() and time.time() < deadline:
            time.sleep(0.01)
        assert check()

    theirs = object()                                  # the model of the run that has the GPU
    transcribe._loaded[:] = [theirs]
    job = JobRegistry().create()
    transcribe.GPU_TURN.acquire()                      # another run holds the GPU, to the end of this test
    try:
        JobRegistry().run(job, work, on_pause=transcribe.unload_models)
        until(lambda: job.message == "waiting")
        job.pause()
        until(lambda: job.gpu_freed)                   # parked, without its turn
        assert transcribe._loaded == [theirs]          # nothing dropped under the run using it
        e1 = job.elapsed
        time.sleep(0.2)
        assert job.elapsed - e1 < 0.1                  # its clock stopped
        job.resume()
        until(lambda: job.message == "waiting")        # back to waiting
        job.cancel()
        until(lambda: job.status is JobStatus.CANCELLED)
    finally:
        transcribe.GPU_TURN.release()
        transcribe._loaded.clear()


# --- Auto-detect: speech across the file, the user's languages, asking before a big download (#38) ---

def test_speech_windows_come_from_across_the_file_and_skip_what_has_no_speech(monkeypatch):
    from app import diarize
    SR = diarize.SR

    def fake_spans(chunk):                             # "speech" = the non-zero samples
        on = np.flatnonzero(chunk.numpy())
        return [(on[0] / SR, (on[-1] + 1) / SR)] if len(on) else []

    monkeypatch.setattr(diarize, "speech_spans", fake_spans)
    audio = np.ones(600 * SR, dtype=np.float32)
    audio[:80 * SR] = 0                                # opens with 80 s of no speech
    audio[330 * SR:420 * SR] = 0                       # and a silent stretch later on
    audio[80 * SR:120 * SR] = 0.25                     # each spot's speech is told apart by its level
    audio[180 * SR:270 * SR] = 0.5
    wins = transcribe.speech_windows(audio)
    # 4 spots of 45 s in the middle of each quarter: 52.5-97.5 s (17.5 s of speech), 202.5-247.5 s,
    # the silent 352.5-397.5 s (no vote), 502.5-547.5 s
    assert [(len(w) // SR, float(w[0])) for w in wins] == [(17, 0.25), (30, 0.5), (30, 1.0)]

    assert transcribe.speech_windows(np.zeros(600 * SR, dtype=np.float32)) == []      # no speech anywhere
    short = transcribe.speech_windows(np.ones(60 * SR, dtype=np.float32))             # a short file: one window
    assert [len(w) // SR for w in short] == [30]


def test_language_probs_average_the_windows(monkeypatch):
    votes = iter([[("<|nn|>", 0.6), ("<|en|>", 0.4)], [("<|en|>", 0.9), ("<|nn|>", 0.1)]])

    class Pipe:
        class model:
            feat_kwargs = {"feature_size": 80}
            encode = staticmethod(lambda mel: next(votes))
            class model:
                detect_language = staticmethod(lambda encoded: [encoded])

    monkeypatch.setattr(transcribe, "speech_windows", lambda audio: [np.zeros(16000, dtype=np.float32)] * 2)
    probs = transcribe.language_probs(Pipe, np.zeros(16000, dtype=np.float32))
    assert probs == pytest.approx({"nn": 0.35, "en": 0.65})          # one odd window is outvoted


def test_choose_language_trusts_the_users_languages():
    guess = {"nn": 0.36, "en": 0.30, "pt": 0.05}
    assert transcribe.choose_language(guess, []) == "nn"              # no list: the top guess stands
    assert transcribe.choose_language(guess, ["en", "pt"]) == "en"    # outside the list and unsure: best of theirs
    assert transcribe.choose_language(guess, ["nn"]) == "nn"          # in the list: accepted, however unsure
    assert transcribe.choose_language({"ja": 0.97, "en": 0.01}, ["en", "pt"]) == "ja"   # outside but sure


def _auto_detect(monkeypatch, tmp_path, probs, installed=()):
    """transcribe_track with every model stubbed; returns what was downloaded, aligned, dropped."""
    ms = transcribe.modelstore
    monkeypatch.setattr(ms, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(whisperx, "load_audio", lambda p: np.zeros(16000, dtype=np.float32))
    monkeypatch.setattr(transcribe, "_load_model", lambda *a: _FakeModel("?"))
    monkeypatch.setattr(transcribe, "language_probs", lambda pipe, audio: probs)
    monkeypatch.setattr(ms, "is_installed", lambda e: e.id in installed or e.kind == "whisper")
    seen = {"downloaded": [], "aligned": [], "dropped": []}
    monkeypatch.setattr(ms, "ensure", lambda e: seen["downloaded"].append(e.id))
    monkeypatch.setattr(transcribe, "_load_align", lambda lang, device: (seen["aligned"].append(lang), {}))
    monkeypatch.setattr(transcribe, "_drop_align", lambda: seen["dropped"].append(seen["aligned"][-1]))
    monkeypatch.setattr(whisperx, "align", lambda segs, *a, **k: {"segments": [{**s, "words": ["w"]} for s in segs]})
    import torch
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return seen


def test_auto_detect_asks_before_a_big_download(monkeypatch, tmp_path):
    seen = _auto_detect(monkeypatch, tmp_path, {"nn": 0.36, "en": 0.30})
    asked, reply = [], [("download", None)]

    def ask(question):
        asked.append(question)
        return reply[0]

    # nobody to ask (a script): no download, line timings only
    out = transcribe.transcribe_track(tmp_path / "x.wav")
    assert out["language"] == "nn" and out["language_probability"] == 0.36
    assert out["segments"][0]["words"] == [] and "align:nn" not in seen["downloaded"]

    with transcribe.asking(ask):
        out = transcribe.transcribe_track(tmp_path / "x.wav")           # "download": fetch it, align
        q = asked[0]
        assert "Norwegian Nynorsk (36% sure)" in q["text"] and "3.9 GB" in q["text"]
        assert q["language"] == "nn" and q["size_mb"] == 3850 and set(q["choices"]) == {"download", "skip", "language"}
        assert q["others"] == [{"language": "en", "confidence": 0.3}]
        assert seen["downloaded"][-1] == "align:nn" and out["segments"][0]["words"] == ["w"]
        assert seen["dropped"] == ["nn"]                                # a 3.9 GB model doesn't stay on the GPU

        reply[0] = ("skip", None)                                       # go without word timing
        seen["downloaded"].clear()
        out = transcribe.transcribe_track(tmp_path / "x.wav")
        assert out["language"] == "nn" and out["segments"][0]["words"] == [] and out["word_timing"] is False
        assert "align:nn" not in seen["downloaded"]

        reply[0] = ("language", "en")                                   # use another language instead
        out = transcribe.transcribe_track(tmp_path / "x.wav")
        assert out["language"] == "en" and "language_probability" not in out
        assert seen["downloaded"][-1] == "align:en" and seen["aligned"][-1] == "en"
        assert seen["dropped"] == ["nn"]                                # a small model stays loaded
        assert len(asked) == 3

        # no question when the language was picked, is one the user uses, or its model is small
        transcribe.transcribe_track(tmp_path / "x.wav", language="nn")
        assert seen["downloaded"][-1] == "align:nn"
        transcribe.modelstore.set_my_languages(["nn"])
        transcribe.transcribe_track(tmp_path / "x.wav")
        assert seen["downloaded"][-1] == "align:nn"
        transcribe.modelstore.set_my_languages(["en", "pt"])            # nn at 0.36 isn't trusted: English
        assert transcribe.transcribe_track(tmp_path / "x.wav")["language"] == "en"
        assert len(asked) == 3


def test_auto_detect_does_not_ask_when_the_model_is_on_disk(monkeypatch, tmp_path):
    seen = _auto_detect(monkeypatch, tmp_path, {"nn": 0.9}, installed={"align:nn"})
    with transcribe.asking(lambda q: (_ for _ in ()).throw(AssertionError("asked"))):
        assert transcribe.transcribe_track(tmp_path / "x.wav")["language"] == "nn"
    assert seen["aligned"] == ["nn"] and seen["dropped"] == ["nn"]


def test_auto_detect_does_not_ask_for_a_usual_size_model(monkeypatch, tmp_path):
    seen = _auto_detect(monkeypatch, tmp_path, {"nl": 0.9})            # 1.26 GB, not on disk
    with transcribe.asking(lambda q: (_ for _ in ()).throw(AssertionError("asked"))):
        assert transcribe.transcribe_track(tmp_path / "x.wav")["language"] == "nl"
    assert "align:nl" in seen["downloaded"] and seen["aligned"] == ["nl"] and seen["dropped"] == []


def test_a_finished_run_lets_go_of_its_audio_and_models(monkeypatch, tmp_path):
    """A frame of transcribe_track can outlive the call (a library holding on to a traceback), and
    kept the run's word-timing model on the GPU until the server restarted. Its locals are cleared."""
    import sys
    _auto_detect(monkeypatch, tmp_path, {"en": 0.9})
    frames = []
    monkeypatch.setattr(whisperx, "align", lambda segs, *a, **k: (frames.append(sys._getframe(1)), {"segments": []})[1])
    transcribe.transcribe_track(tmp_path / "x.wav")
    held = frames[0].f_locals
    assert held["audio"] is None and held["model"] is None and held["align_model"] is None and held["metadata"] is None


def test_word_timing_off_skips_alignment_and_has_its_own_cache_entry(monkeypatch, tmp_path):
    """#39: without word timing nothing is aligned, downloaded or asked; the words are the same;
    and neither kind of run is handed the other's saved result."""
    from app.cache import Cache
    seen = _auto_detect(monkeypatch, tmp_path, {"nn": 0.36, "en": 0.30})   # nn: the 3.9 GB model that asks
    wav, cache = tmp_path / "x.wav", Cache(tmp_path / "cache")
    wav.write_bytes(b"audio")                                           # the cache key reads the file
    texts = lambda out: [s["text"] for s in out["segments"]]
    aligned = lambda: seen["aligned"] + [m for m in seen["downloaded"] if m.startswith("align:")]

    with transcribe.asking(lambda q: (_ for _ in ()).throw(AssertionError("asked"))):
        auto_off = transcribe.transcribe_track(wav, cache=cache, word_timing=False)
        off = transcribe.transcribe_track(wav, cache=cache, language="en", word_timing=False)
    assert aligned() == []                                              # no word-timing model ensured or loaded
    assert auto_off["language"] == "nn" and auto_off["word_timing"] is False
    assert off["word_timing"] is False and off["segments"][0]["words"] == []

    on = transcribe.transcribe_track(wav, cache=cache, language="en")   # default: aligned, not the saved one
    assert on["segments"][0]["words"] == ["w"] and "word_timing" not in on
    assert texts(on) == texts(off)
    assert aligned() == ["en", "align:en"]

    ran = len(seen["downloaded"])                                       # each comes back from its own entry
    assert transcribe.transcribe_track(wav, cache=cache, language="en", word_timing=False) == off
    assert transcribe.transcribe_track(wav, cache=cache, language="en") == on
    assert transcribe.transcribe_track(wav, cache=cache, word_timing=False) == auto_off
    assert len(seen["downloaded"]) == ran

    # a language with no word-timing model has no words either way, and still keeps the two apart
    plain = transcribe.transcribe_track(wav, cache=cache, language="is")
    assert "word_timing" not in plain and plain["segments"][0]["words"] == []
    assert transcribe.transcribe_track(wav, cache=cache, language="is", word_timing=False)["word_timing"] is False
    assert transcribe.transcribe_track(wav, cache=cache, language="is") == plain
