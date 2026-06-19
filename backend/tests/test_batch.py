"""Batch orchestration tests — real ffmpeg files + a stub transcribe (no GPU)."""

import json

from app.batch import BatchRegistry, ProjectInput, Status, _RunSpec, run_batch
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


def test_per_file_source_writes_beside_each_source(two_stream_file, tmp_path):
    # files in different nested folders -> per_file + "source" writes each transcript beside
    # its own source, preserving the tree (not collapsed into one folder).
    import shutil
    a = tmp_path / "showA" / "ep1.mkv"; a.parent.mkdir(parents=True)
    b = tmp_path / "showB" / "sub" / "ep2.mkv"; b.parent.mkdir(parents=True)
    shutil.copy(two_stream_file, a); shutil.copy(two_stream_file, b)
    proj = ProjectInput("Folder", [
        _prepared(a, "ep1.mkv", 2.0, sequence=0),
        _prepared(b, "ep2.mkv", 2.0, sequence=1),
    ], per_file=True, formats=["txt"])
    batch = _run([proj], output_mode="source")
    assert batch.projects[0].status is Status.DONE
    assert (a.parent / "ep1.txt").exists()           # beside its own source
    assert (b.parent / "ep2.txt").exists()           # nested location preserved
    assert not (a.parent / "ep2.txt").exists()       # not collapsed together


def test_formats_filter_limits_merge_output(two_stream_file, tmp_path):
    proj = ProjectInput("Solo", [_prepared(two_stream_file, "a.mkv", 2.0)], formats=["txt"])
    _run([proj], output_mode="folder", output_folder=str(tmp_path))
    assert (tmp_path / "Solo" / "Solo.txt").exists()
    for ext in ("srt", "csv", "json"):
        assert not (tmp_path / "Solo" / f"Solo.{ext}").exists()


def test_cancel_before_run_marks_all_canceled(two_stream_file, tmp_path):
    projects = [ProjectInput("A", [_prepared(two_stream_file, "a.mkv", 2.0)]),
                ProjectInput("B", [_prepared(two_stream_file, "b.mkv", 2.0)])]
    batch = BatchRegistry().create([p.name for p in projects])
    batch.cancelled = True
    run_batch(batch, projects, _stub, output_mode="folder", output_folder=str(tmp_path))
    assert all(p.status is Status.CANCELED for p in batch.projects)
    assert not (tmp_path / "A" / "A.srt").exists()


def test_cancel_mid_batch_stops_queue(two_stream_file, tmp_path):
    batch = BatchRegistry().create(["A", "B"])
    projects = [ProjectInput("A", [_prepared(two_stream_file, "a.mkv", 2.0)]),
                ProjectInput("B", [_prepared(two_stream_file, "b.mkv", 2.0)])]

    def stub(_wav):                       # request cancel during the first project's run
        batch.cancelled = True
        return _stub(_wav)

    run_batch(batch, projects, stub, output_mode="folder", output_folder=str(tmp_path))
    assert batch.projects[1].status is Status.CANCELED        # B never started
    assert not (tmp_path / "B" / "B.srt").exists()


def test_retry_reruns_selected_project(two_stream_file, tmp_path):
    import time as _t
    reg = BatchRegistry()
    projects = [ProjectInput("A", [_prepared(two_stream_file, "a.mkv", 2.0)]),
                ProjectInput("B", [_prepared("C:/nope/missing.mkv", "b.mkv", 2.0)])]
    batch = reg.create(["A", "B"])
    reg.set_spec(batch.id, _RunSpec(projects, _stub, "folder", str(tmp_path), None))
    run_batch(batch, projects, _stub, output_mode="folder", output_folder=str(tmp_path))
    assert batch.projects[1].status is Status.ERROR
    # repair B in the spec and retry just it; A must be left untouched
    projects[1] = ProjectInput("B", [_prepared(two_stream_file, "b.mkv", 2.0)])
    reg.set_spec(batch.id, _RunSpec(projects, _stub, "folder", str(tmp_path), None))
    assert reg.retry(batch.id, [1])
    deadline = _t.time() + 5
    while batch.status not in (Status.DONE, Status.CANCELED) and _t.time() < deadline:
        _t.sleep(0.02)
    assert batch.projects[1].status is Status.DONE
    assert (tmp_path / "B" / "B.srt").exists()
    assert batch.projects[0].status is Status.DONE


def test_output_mode_source_writes_next_to_file(two_stream_file, tmp_path):
    import shutil
    local = tmp_path / "session.mkv"
    shutil.copy(two_stream_file, local)
    proj = ProjectInput("NextTo", [_prepared(local, "session.mkv", 2.0)])
    batch = _run([proj], output_mode="source")
    assert batch.projects[0].output_dir == str(tmp_path / "NextTo")
    assert (tmp_path / "NextTo" / "NextTo.srt").exists()
