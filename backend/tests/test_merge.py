"""Merge-engine tests — synthetic WhisperX JSON exercises every conflict branch (§8.4).

GPU-independent: no model load. This is the verification anchor for the merge contract.
"""

from app.merge import BackchannelRender, MergeOptions, merge


def wx(*segments: dict) -> dict:
    """Build a minimal WhisperX result dict."""
    return {"segments": list(segments)}


def seg(start: float, end: float, text: str, words: list | None = None) -> dict:
    if words is None:
        # one synthetic word per token, evenly spread across the segment span
        toks = text.split()
        span = (end - start) / max(len(toks), 1)
        words = [
            {"word": t, "start": round(start + i * span, 3), "end": round(start + (i + 1) * span, 3)}
            for i, t in enumerate(toks)
        ]
    return {"start": start, "end": end, "text": text, "words": words}


# FR-MG-1 -------------------------------------------------------------------------------
def test_no_overlap_interleaves_by_start():
    a = wx(seg(0.0, 2.0, "first from max"), seg(6.0, 8.0, "third from max"))
    b = wx(seg(3.0, 5.0, "second from sam"))
    out = merge([a, b], ["Max", "Sam"])
    assert [s.start for s in out.segments] == [0.0, 3.0, 6.0]
    assert [s.speaker for s in out.segments] == ["Max", "Sam", "Max"]
    assert all(s.flags == [] for s in out.segments)


def test_words_preserved_and_offset_applied():
    a = wx(seg(0.0, 2.0, "hello world"))
    out = merge([a], ["Max"], offsets=[10.0])
    s = out.segments[0]
    assert s.start == 10.0 and s.end == 12.0
    assert [w.word for w in s.words] == ["hello", "world"]
    assert s.words[0].start == 10.0  # offset propagated to words


# FR-MG-2 -------------------------------------------------------------------------------
def test_backchannel_flagged_and_parenthetical_by_default():
    a = wx(seg(0.0, 6.0, "I was explaining the whole thing in detail"))
    b = wx(seg(2.0, 2.5, "yeah"))  # short, fully inside A
    out = merge([a, b], ["Max", "Sam"])
    bc = [s for s in out.segments if s.speaker == "Sam"][0]
    assert "backchannel" in bc.flags
    assert bc.text == "(yeah)"


def test_backchannel_skip_drops_it():
    a = wx(seg(0.0, 6.0, "I was explaining the whole thing in detail"))
    b = wx(seg(2.0, 2.5, "right"))
    out = merge([a, b], ["Max", "Sam"],
                options=MergeOptions(backchannel_render=BackchannelRender.SKIP))
    assert all(s.speaker != "Sam" for s in out.segments)


def test_backchannel_separate_keeps_text():
    a = wx(seg(0.0, 6.0, "I was explaining the whole thing in detail"))
    b = wx(seg(2.0, 2.5, "uh huh"))
    out = merge([a, b], ["Max", "Sam"],
                options=MergeOptions(backchannel_render=BackchannelRender.SEPARATE))
    bc = [s for s in out.segments if s.speaker == "Sam"][0]
    assert bc.text == "uh huh" and "backchannel" in bc.flags


def test_long_contained_segment_is_not_backchannel():
    a = wx(seg(0.0, 10.0, "I was explaining the whole thing in great detail today"))
    b = wx(seg(2.0, 8.0, "actually I completely disagree with that point you made"))  # long
    out = merge([a, b], ["Max", "Sam"])
    sam = [s for s in out.segments if s.speaker == "Sam"][0]
    assert "backchannel" not in sam.flags
    assert "overlap" in sam.flags  # multi-second simultaneous -> genuine overlap


# FR-MG-3 -------------------------------------------------------------------------------
def test_genuine_overlap_flags_both():
    a = wx(seg(0.0, 5.0, "so the plan here is to ship it"))
    b = wx(seg(2.0, 7.0, "but I think we should wait a bit"))  # ~3s simultaneous
    out = merge([a, b], ["Max", "Sam"])
    assert all("overlap" in s.flags for s in out.segments)


def test_brief_overlap_below_threshold_not_flagged():
    a = wx(seg(0.0, 3.0, "okay sounds good to me"))
    b = wx(seg(2.5, 5.0, "great lets do it then"))  # 0.5s overlap, not similar text
    out = merge([a, b], ["Max", "Sam"])
    assert all(s.flags == [] for s in out.segments)


# FR-MG-4 -------------------------------------------------------------------------------
def test_bleed_flagged_on_similar_text():
    text = "lets go ahead and merge this branch now"
    a = wx(seg(0.0, 3.0, text))
    b = wx(seg(0.1, 3.1, text))  # near-identical, tiny window -> bleed
    out = merge([a, b], ["Mic", "Desktop"])
    assert all("bleed_suspected" in s.flags for s in out.segments)
    assert all("overlap" not in s.flags for s in out.segments)  # bleed wins precedence


def test_bleed_preferred_uses_rms_hook():
    text = "lets go ahead and merge this branch now"
    a = wx(seg(0.0, 3.0, text))
    b = wx(seg(0.1, 3.1, text))
    # Mic is louder than Desktop -> Mic gets "preferred"
    rms = {"Mic": 0.9, "Desktop": 0.2}
    out = merge([a, b], ["Mic", "Desktop"], rms_fn=lambda spk, s, e: rms[spk])
    mic = [s for s in out.segments if s.speaker == "Mic"][0]
    desk = [s for s in out.segments if s.speaker == "Desktop"][0]
    assert "preferred" in mic.flags
    assert "preferred" not in desk.flags


# sequential ----------------------------------------------------------------------------
def test_sequential_offsets_concatenate():
    f1 = wx(seg(0.0, 4.0, "first file speaker a"))
    f2 = wx(seg(0.0, 3.0, "second file speaker b"))
    # f2 starts after f1's 4.0s duration
    out = merge([f1, f2], ["A", "B"], offsets=[0.0, 4.0])
    assert [s.start for s in out.segments] == [0.0, 4.0]
    assert all(s.flags == [] for s in out.segments)  # disjoint in time -> no conflicts
