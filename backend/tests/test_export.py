"""Export tests — TXT / CSV / JSON / SRT incl. word-level line splitting (FR-EX-*)."""

import json

from app.export import ExportOptions, export_all, pad_segments, to_csv, to_json, to_srt, to_txt, _timecode
from app.models import MergedOutput, Segment, Word


def _words(text, start, step=0.4):
    toks = text.split()
    return [Word(word=t, start=round(start + i * step, 3), end=round(start + (i + 1) * step, 3))
            for i, t in enumerate(toks)]


def _merged(*segments):
    return MergedOutput(segments=list(segments))


def test_export_all_writes_four_files(tmp_path):
    m = _merged(Segment(speaker="Max", start=0, end=1, text="hello", words=_words("hello", 0)))
    written = export_all(m, tmp_path / "proj", "proj")
    assert {p.name for p in written} == {"proj.srt", "proj.txt", "proj.csv", "proj.json"}
    assert all(p.exists() and p.read_text(encoding="utf-8") for p in written)


def test_export_all_formats_filter(tmp_path):
    m = _merged(Segment(speaker="Max", start=0, end=1, text="hello", words=_words("hello", 0)))
    written = export_all(m, tmp_path / "proj", "proj", formats=["txt"])
    assert {p.name for p in written} == {"proj.txt"}
    assert not (tmp_path / "proj" / "proj.srt").exists()
    assert not (tmp_path / "proj" / "proj.json").exists()


def test_timecode_format():
    assert _timecode(0) == "00:00:00,000"
    assert _timecode(3661.5) == "01:01:01,500"
    assert _timecode(12.34) == "00:00:12,340"


def test_to_json_roundtrips():
    m = _merged(Segment(speaker="Max", start=0, end=1, text="hello", words=_words("hello", 0)))
    data = json.loads(to_json(m))
    assert data["version"] == "1.0"
    assert data["segments"][0]["speaker"] == "Max"
    assert data["segments"][0]["words"][0]["word"] == "hello"


def test_to_txt_has_start_end_and_speaker_toggle():
    m = _merged(Segment(speaker="Max", start=0, end=1, text="hello there"))
    assert to_txt(m) == "[00:00:00,000 --> 00:00:01,000] Max: hello there"
    assert to_txt(m, ExportOptions(show_speakers=False)) == "[00:00:00,000 --> 00:00:01,000] hello there"


def test_to_csv_has_header_and_rows():
    m = _merged(Segment(speaker="Max", start=0.0, end=1.0, text="hi", flags=["overlap"]))
    rows = to_csv(m).splitlines()
    assert rows[0] == "start,end,speaker,text,flags"
    assert "Max" in rows[1] and "overlap" in rows[1]


def test_srt_basic_cue():
    m = _merged(Segment(speaker="Max", start=0.0, end=2.0, text="hello world",
                        words=_words("hello world", 0.0)))
    srt = to_srt(m)
    assert srt.startswith("1\n")
    assert "00:00:00,000 --> " in srt
    assert "Max: hello world" in srt


def test_srt_word_level_split_long_segment():
    # Long line; CPL forces multiple lines, max_lines forces multiple cues.
    text = "one two three four five six seven eight nine ten eleven twelve"
    seg = Segment(speaker="Max", start=0.0, end=6.0, text=text, words=_words(text, 0.0))
    srt = to_srt(_merged(seg), ExportOptions(chars_per_line=12, max_lines=1))
    # multiple cues indexed
    assert "\n1\n" in ("\n" + srt) and "2\n" in srt
    # cue timing comes from word timestamps, not the segment span
    assert "00:00:00,000 --> " in srt


def test_srt_apply_limits_off_one_cue_per_segment():
    text = "one two three four five six seven eight nine ten eleven twelve"
    seg = Segment(speaker="Max", start=0.0, end=6.0, text=text, words=_words(text, 0.0))
    srt = to_srt(_merged(seg), ExportOptions(apply_limits=False, chars_per_line=12, max_lines=1))
    blocks = [b for b in srt.split("\n\n") if b.strip()]
    assert len(blocks) == 1                              # no splitting
    assert "00:00:00,000 --> 00:00:06,000" in srt        # full segment timing
    assert f"Max: {text}" in srt                          # whole text on one cue


def test_srt_min_duration_enforced():
    seg = Segment(speaker="Max", start=0.0, end=0.3, text="hi",
                  words=[Word(word="hi", start=0.0, end=0.3)])
    srt = to_srt(_merged(seg), ExportOptions(min_duration=1.0))
    assert "00:00:00,000 --> 00:00:01,000" in srt


def test_pad_segments_shifts_and_clamps():
    m = _merged(Segment(speaker="Max", start=0.2, end=1.0, text="hi"),
                Segment(speaker="Sam", start=3.0, end=4.0, text="yo"))
    padded = pad_segments(m, 0.5)
    assert padded.segments[0].start == 0.0          # clamped at zero (0.2 - 0.5)
    assert padded.segments[0].end == 1.5
    assert padded.segments[1].start == 2.5 and padded.segments[1].end == 4.5
    # source untouched + zero pad is a no-op
    assert m.segments[0].start == 0.2
    assert pad_segments(m, 0.0) is m


def test_pad_seconds_in_export_timecodes():
    seg = Segment(speaker="Max", start=1.0, end=2.0, text="hello",
                  words=_words("hello", 1.0))
    m = _merged(seg)
    txt = to_txt(pad_segments(m, 0.5))
    assert "[00:00:00,500 --> 00:00:02,500]" in txt
    csv = to_csv(pad_segments(m, 0.5))
    assert ",0.5,2.5," in csv.replace(" ", "") or "0.5,2.5" in csv


def test_srt_pad_moves_only_outer_edges_of_word_timed_cues():
    text = "alpha beta gamma delta epsilon zeta"            # words 1.0 .. 3.4 s
    seg = Segment(speaker="Max", start=1.0, end=3.4, text=text, words=_words(text, 1.0))
    opts = dict(chars_per_line=12, max_lines=1, min_duration=0.0, show_speakers=False)
    times = lambda srt: [b.split("\n")[1] for b in srt.split("\n\n") if b.strip()]
    plain = times(to_srt(_merged(seg), ExportOptions(**opts)))
    padded = times(to_srt(pad_segments(_merged(seg), 0.5), ExportOptions(pad_seconds=0.5, **opts)))
    assert len(plain) == len(padded) >= 3
    assert padded[0].startswith("00:00:00,500") and padded[-1].endswith("00:00:03,900")
    assert padded[0].split(" --> ")[1] == plain[0].split(" --> ")[1]       # inner split unchanged
    assert padded[1:-1] == plain[1:-1]


def test_srt_speaker_prefix_first_line_only():
    text = "alpha beta gamma delta epsilon zeta"
    seg = Segment(speaker="Max", start=0.0, end=3.0, text=text, words=_words(text, 0.0))
    srt = to_srt(_merged(seg), ExportOptions(chars_per_line=10, max_lines=2))
    blocks = [b for b in srt.split("\n\n") if b.strip()]
    assert len(blocks) >= 2  # split into multiple cues
    for block in blocks:
        text_lines = block.split("\n")[2:]  # skip index + timecode lines
        assert text_lines[0].startswith("Max:")              # speaker on first line
        assert all("Max:" not in line for line in text_lines[1:])  # not on later lines


def test_srt_speaker_label_counts_toward_chars_per_line():
    # label "Speaker 2: " is 11 chars; every cue's first line must still fit within 20
    text = "Alice speaking on the first microphone and then some more words here"
    opts = ExportOptions(chars_per_line=20, max_lines=2, min_duration=0.0)
    for words in (_words(text, 0.0), []):                  # word-timed and plain-text paths
        seg = Segment(speaker="Speaker 2", start=0.0, end=10.0, text=text, words=words)
        srt = to_srt(_merged(seg), opts)
        lines = [ln for b in srt.split("\n\n") if b.strip() for ln in b.split("\n")[2:]]
        assert all(len(ln) <= 20 for ln in lines), [ln for ln in lines if len(ln) > 20]
        assert sum(ln.startswith("Speaker 2: ") for ln in lines) >= 2   # label on each cue


def _times(srt):
    return [b.split("\n")[1] for b in srt.split("\n\n") if b.strip()]


def _texts(srt):
    return ["\n".join(b.split("\n")[2:]) for b in srt.split("\n\n") if b.strip()]


_ORIG = "alpha beta gamma delta epsilon zeta eta theta"           # words at 0.0, 0.4, ... 2.8-3.2 s
_LIMITS = dict(chars_per_line=12, max_lines=1, min_duration=0.0, show_speakers=False)


def _edited(text):
    return _merged(Segment(speaker="Max", start=0.0, end=3.2, text=text, words=_words(_ORIG, 0.0),
                           flags=["edited"]))


def test_srt_uses_edited_text_and_keeps_word_timing_for_unchanged_words():
    plain = to_srt(_merged(Segment(speaker="Max", start=0.0, end=3.2, text=_ORIG, words=_words(_ORIG, 0.0))),
                   ExportOptions(**_LIMITS))
    srt = to_srt(_edited("alpha Beta gamma delta epsilon Zeta eta theta"), ExportOptions(**_LIMITS))
    assert "Beta" in srt and "Zeta" in srt                         # the edit is what's written
    assert _times(srt) == _times(plain)                              # 1:1 fixes keep exact timing


def test_srt_added_words_share_the_gap_and_rewrite_falls_back_to_even():
    srt = to_srt(_edited("alpha beta gamma NEW delta epsilon zeta eta theta"), ExportOptions(**_LIMITS))
    assert "NEW" in srt and _times(srt)[0].startswith("00:00:00,000")
    rewrite = _edited("completely different words were said here instead today")
    assert (to_srt(rewrite, ExportOptions(**_LIMITS)) ==
            to_srt(rewrite, ExportOptions(edited_timing="even", **_LIMITS)))


def test_srt_backchannel_keeps_parentheses():
    seg = Segment(speaker="Bob", start=2.0, end=2.4, text="(yeah)", words=_words("yeah", 2.0),
                  flags=["backchannel"])
    assert "Bob: (yeah)" in to_srt(_merged(seg))


def test_srt_edited_line_modes_even_and_whole():
    seg = _edited("alpha Beta gamma delta epsilon Zeta eta theta")
    even = to_srt(seg, ExportOptions(edited_timing="even", **_LIMITS))
    assert len(_times(even)) > 1 and _times(even)[-1].endswith("00:00:03,200")
    whole = to_srt(seg, ExportOptions(edited_timing="whole", **_LIMITS))
    assert _times(whole) == ["00:00:00,000 --> 00:00:03,200"]        # one subtitle, exact span
    assert _texts(whole)[0].count("\n") >= 2                         # all lines kept, past max_lines
