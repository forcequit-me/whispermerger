"""Cache tests — key stability and put/get roundtrip (§8.8)."""

from app.cache import Cache, cache_key, file_sha256


def _write(path, data=b"some audio bytes"):
    path.write_bytes(data)
    return path


def test_file_sha256_stable(tmp_path):
    f = _write(tmp_path / "a.wav")
    assert file_sha256(f) == file_sha256(f)


def test_key_depends_on_model_and_align(tmp_path):
    f = _write(tmp_path / "a.wav")
    k1 = cache_key(f, "large-v3", "wav2vec2")
    k2 = cache_key(f, "large-v2", "wav2vec2")
    k3 = cache_key(f, "large-v3", "other-align")
    assert k1 != k2 != k3 and k1 != k3


def test_key_depends_on_language(tmp_path):
    f = _write(tmp_path / "a.wav")
    auto = cache_key(f, "large-v3", "wav2vec2")            # None -> "auto"
    en = cache_key(f, "large-v3", "wav2vec2", "en")
    es = cache_key(f, "large-v3", "wav2vec2", "es")
    assert auto != en != es and auto != es


def test_key_depends_on_content(tmp_path):
    a = _write(tmp_path / "a.wav", b"aaa")
    b = _write(tmp_path / "b.wav", b"bbb")
    assert cache_key(a, "large-v3", "wav2vec2") != cache_key(b, "large-v3", "wav2vec2")


def test_put_get_roundtrip(tmp_path):
    cache = Cache(tmp_path / "cache")
    payload = {"segments": [{"start": 0, "end": 1, "text": "hi", "words": []}]}
    cache.put("k1", payload)
    assert cache.get("k1") == payload


def test_get_miss_returns_none(tmp_path):
    cache = Cache(tmp_path / "cache")
    assert cache.get("nope") is None


def test_clear_removes_all_entries(tmp_path):
    cache = Cache(tmp_path / "cache")
    cache.put("k1", {"segments": []})
    cache.put("k2", {"segments": []})
    assert cache.clear() == 2
    assert cache.get("k1") is None and cache.get("k2") is None
    assert cache.clear() == 0                               # idempotent on an empty cache


def test_stats_counts_entries_and_bytes(tmp_path):
    cache = Cache(tmp_path / "cache")
    assert cache.stats() == {"entries": 0, "bytes": 0}
    cache.put("k1", {"segments": []})
    cache.put("k2", {"segments": [{"text": "hi"}]})
    s = cache.stats()
    assert s["entries"] == 2 and s["bytes"] == sum(p.stat().st_size for p in (tmp_path / "cache").glob("*.json"))
    cache.clear()
    assert cache.stats() == {"entries": 0, "bytes": 0}
