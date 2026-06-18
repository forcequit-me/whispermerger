"""Batch orchestration tests — real ffmpeg files + a stub transcribe (no GPU)."""

import json

from app.batch import BatchRegistry, ProjectInput, Status, run_batch
from app.models import InputSource, Track
from app.pipeline import PreparedSource


def _stub(_wav):
    return {"segments": [{"start": 0.0, "end": 1.0, "text": "hi",
                          "words": [{"word": "hi", "start": 0.0, "end": 1.0}]}]}


def _prepared(path, name, duration, sequence=0, speaker="Speaker 1"):
    src = InputSource(filename=name, duration=duration, sequence=sequence,
                      tracks=[Track(track_index=0, speaker=speaker)])
    return PreparedSource(path=str(path), input=src)


def _run(projects, **kw):
    batch = BatchRegistry().create([p.name for p in projects])
    run_batch(batch, projects, _stub, **kw)
    return batch


def test_writes_all_formats_per_project(two_stream_file, tmp_path):
    projects = [
        ProjectInput("Performance Poser", [_prepared(two_stream_file, "pp.mkv", 2.0)]),
        ProjectInput("Whiteout", [_prepared(two_stream_file, "wo.mkv", 2.0)]),
    ]
    batch = _run(projects, output_mode="folder", output_folder=str(tmp_path))
    assert all(p.status is Status.DONE for p in batch.projects)
    for name in ("Performance Poser", "Whiteout"):
        for ext in ("srt", "txt", "csv", "json"):
            assert (tmp_path / name / f"{name}.{ext}").exists()


def test_sequential_offsets_within_project(two_stream_file, tmp_path):
    # two session files, second offset by the first's 4s duration (Method 1)
    proj = ProjectInput("Multi", [
        _prepared(two_stream_file, "s1.mkv", 4.0, sequence=0),
        _prepared(two_stream_file, "s2.mkv", 3.0, sequence=1),
    ])
    _run([proj], output_mode="folder", output_folder=str(tmp_path))
    data = json.loads((tmp_path / "Multi" / "Multi.json").read_text(encoding="utf-8"))
    assert [s["start"] for s in data["segments"]] == [0.0, 4.0]


def test_error_isolation_continues_queue(two_stream_file, tmp_path):
    projects = [
        ProjectInput("Bad", [_prepared("C:/nope/missing.mkv", "x.mkv", 2.0)]),
        ProjectInput("Good", [_prepared(two_stream_file, "ok.mkv", 2.0)]),
    ]
    batch = _run(projects, output_mode="folder", output_folder=str(tmp_path))
    assert batch.projects[0].status is Status.ERROR
    assert batch.projects[0].error
    assert batch.projects[1].status is Status.DONE  # queue continued


def test_per_file_writes_one_transcript_per_source(two_stream_file, tmp_path):
    proj = ProjectInput("Library", [
        _prepared(two_stream_file, "intro.mkv", 2.0, sequence=0),
        _prepared(two_stream_file, "outro.mkv", 2.0, sequence=1),
    ], per_file=True, formats=["txt"])
    batch = _run([proj], output_mode="folder", output_folder=str(tmp_path))
    assert batch.projects[0].status is Status.DONE
    proj_dir = tmp_path / "Library"
    assert (proj_dir / "intro.txt").exists()
    assert (proj_dir / "outro.txt").exists()
    # only the chosen format, and no merged <project>.* file
    assert not (proj_dir / "intro.srt").exists()
    assert not (proj_dir / "Library.txt").exists()


def test_formats_filter_limits_merge_output(two_stream_file, tmp_path):
    proj = ProjectInput("Solo", [_prepared(two_stream_file, "a.mkv", 2.0)], formats=["txt"])
    _run([proj], output_mode="folder", output_folder=str(tmp_path))
    assert (tmp_path / "Solo" / "Solo.txt").exists()
    for ext in ("srt", "csv", "json"):
        assert not (tmp_path / "Solo" / f"Solo.{ext}").exists()


def test_output_mode_source_writes_next_to_file(two_stream_file, tmp_path):
    import shutil
    local = tmp_path / "session.mkv"
    shutil.copy(two_stream_file, local)
    proj = ProjectInput("NextTo", [_prepared(local, "session.mkv", 2.0)])
    batch = _run([proj], output_mode="source")
    assert batch.projects[0].output_dir == str(tmp_path / "NextTo")
    assert (tmp_path / "NextTo" / "NextTo.srt").exists()
