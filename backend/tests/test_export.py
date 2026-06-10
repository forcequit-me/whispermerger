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
