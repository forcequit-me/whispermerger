"""Batch orchestration tests — real ffmpeg files + a stub transcribe (no GPU)."""

import json
from pathlib import Path

from app.batch import BatchRegistry, ProjectInput, Status, _RunSpec, _recalc_project, run_batch
from app.models import InputSource, Track
from app.pipeline import PreparedSource


def _stub(_wav):
    return {"segments": [{"start": 0.0, "end": 1.0, "text": "hi",
                          "words": [{"word": "hi", "start": 0.0, "end": 1.0}]}]}


def _find(directory, stem, ext):
    """The exported transcript for `stem` in `directory` under the naming scheme
    `<stem> Transcript (M-D-YYYY h-mm AMPM).<ext>`, or None if absent."""
    matches = list(Path(directory).glob(f"{stem} Transcript (*).{ext}"))
    return matches[0] if matches else None


def _prepared(path, name, duration, sequence=0, speaker="Speaker 1"):
    src = InputSource(filename=name, duration=duration, sequence=sequence,
                      tracks=[Track(track_index=0, speaker=speaker)])
    return PreparedSource(path=str(path), input=src)


def _run(projects, **kw):
    batch = BatchRegistry().create(projects)
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
            assert _find(tmp_path / name, name, ext), f"{name}.{ext}"


def test_sequential_offsets_within_project(two_stream_file, tmp_path):
    # two session files, second offset by the first's 4s duration (Method 1)
    proj = ProjectInput("Multi", [
        _prepared(two_stream_file, "s1.mkv", 4.0, sequence=0),
        _prepared(two_stream_file, "s2.mkv", 3.0, sequence=1),
    ])
    _run([proj], output_mode="folder", output_folder=str(tmp_path))
    f = _find(tmp_path / "Multi", "Multi", "json")
    data = json.loads(f.read_text(encoding="utf-8"))
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
    assert _find(proj_dir, "intro", "txt")
    assert _find(proj_dir, "outro", "txt")
    # only the chosen format, and no merged <project>.* file
    assert not _find(proj_dir, "intro", "srt")
    assert not _find(proj_dir, "Library", "txt")


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
    assert _find(a.parent, "ep1", "txt")             # beside its own source
    assert _find(b.parent, "ep2", "txt")             # nested location preserved
    assert not _find(a.parent, "ep2", "txt")         # not collapsed together


def test_formats_filter_limits_merge_output(two_stream_file, tmp_path):
    proj = ProjectInput("Solo", [_prepared(two_stream_file, "a.mkv", 2.0)], formats=["txt"])
    _run([proj], output_mode="folder", output_folder=str(tmp_path))
    assert _find(tmp_path / "Solo", "Solo", "txt")
    for ext in ("srt", "csv", "json"):
        assert not _find(tmp_path / "Solo", "Solo", ext)


def test_cancel_before_run_marks_all_canceled(two_stream_file, tmp_path):
    projects = [ProjectInput("A", [_prepared(two_stream_file, "a.mkv", 2.0)]),
                ProjectInput("B", [_prepared(two_stream_file, "b.mkv", 2.0)])]
    batch = BatchRegistry().create(projects)
    batch.cancelled = True
    run_batch(batch, projects, _stub, output_mode="folder", output_folder=str(tmp_path))
    assert all(p.status is Status.CANCELED for p in batch.projects)
    assert not _find(tmp_path / "A", "A", "srt")


def test_cancel_mid_batch_stops_queue(two_stream_file, tmp_path):
    projects = [ProjectInput("A", [_prepared(two_stream_file, "a.mkv", 2.0)]),
                ProjectInput("B", [_prepared(two_stream_file, "b.mkv", 2.0)])]
    batch = BatchRegistry().create(projects)

    def stub(_wav):                       # request cancel during the first project's run
        batch.cancelled = True
        return _stub(_wav)

    run_batch(batch, projects, stub, output_mode="folder", output_folder=str(tmp_path))
    assert batch.projects[1].status is Status.CANCELED        # B never started
    assert not _find(tmp_path / "B", "B", "srt")


def test_retry_reruns_selected_project(two_stream_file, tmp_path):
    import time as _t
    reg = BatchRegistry()
    projects = [ProjectInput("A", [_prepared(two_stream_file, "a.mkv", 2.0)]),
                ProjectInput("B", [_prepared("C:/nope/missing.mkv", "b.mkv", 2.0)])]
    batch = reg.create(projects)
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
    assert _find(tmp_path / "B", "B", "srt")
    assert batch.projects[0].status is Status.DONE


def test_transcript_basename_format():
    import time
    from app.batch import transcript_basename
    when = time.mktime((2026, 6, 19, 2, 2, 0, 0, 0, -1))      # 2:02 AM, Jun 19 2026
    assert transcript_basename("My Project", when) == "My Project Transcript (6-19-2026 2-02 AM)"
    pm = time.mktime((2026, 12, 5, 14, 5, 0, 0, 0, -1))       # 2:05 PM, Dec 5
    assert transcript_basename("X", pm) == "X Transcript (12-5-2026 2-05 PM)"


def test_export_names_share_one_timestamp_across_formats(two_stream_file, tmp_path):
    import re
    proj = ProjectInput("Solo", [_prepared(two_stream_file, "a.mkv", 2.0)], formats=["srt", "txt"])
    _run([proj], output_mode="folder", output_folder=str(tmp_path))
    srt = _find(tmp_path / "Solo", "Solo", "srt")
    txt = _find(tmp_path / "Solo", "Solo", "txt")
    assert srt and txt
    assert re.fullmatch(r"Solo Transcript \(\d{1,2}-\d{1,2}-\d{4} \d{1,2}-\d{2} [AP]M\)", srt.stem)
    assert srt.stem == txt.stem                  # all formats of one project share the stamp


def test_cancel_queued_project_skips_it_continues_queue(two_stream_file, tmp_path):
    reg = BatchRegistry()
    projects = [ProjectInput("A", [_prepared(two_stream_file, "a.mkv", 2.0)]),
                ProjectInput("B", [_prepared(two_stream_file, "b.mkv", 2.0)])]
    batch = reg.create(projects)
    assert reg.cancel_project(batch.id, 0)               # cancel A before the run
    run_batch(batch, projects, _stub, output_mode="folder", output_folder=str(tmp_path))
    assert batch.projects[0].status is Status.CANCELED   # A skipped
    assert batch.projects[1].status is Status.DONE       # queue continued to B
    assert not _find(tmp_path / "A", "A", "srt")
    assert _find(tmp_path / "B", "B", "srt")


def test_cancel_running_project_stops_it_not_queue(two_stream_file, tmp_path):
    reg = BatchRegistry()
    projects = [ProjectInput("A", [_prepared(two_stream_file, "a.mkv", 2.0)]),
                ProjectInput("B", [_prepared(two_stream_file, "b.mkv", 2.0)])]
    batch = reg.create(projects)

    def stub(_wav):                                       # cancel A while it transcribes
        reg.cancel_project(batch.id, 0)
        return _stub(_wav)

    run_batch(batch, projects, stub, output_mode="folder", output_folder=str(tmp_path))
    assert batch.projects[0].status is Status.CANCELED   # A stopped at its next boundary
    assert batch.projects[1].status is Status.DONE       # B still ran
    assert _find(tmp_path / "B", "B", "srt")


def test_retry_clears_project_cancel(two_stream_file, tmp_path):
    import time as _t
    reg = BatchRegistry()
    projects = [ProjectInput("A", [_prepared(two_stream_file, "a.mkv", 2.0)])]
    batch = reg.create(projects)
    reg.set_spec(batch.id, _RunSpec(projects, _stub, "folder", str(tmp_path), None))
    assert reg.cancel_project(batch.id, 0)
    run_batch(batch, projects, _stub, output_mode="folder", output_folder=str(tmp_path))
    assert batch.projects[0].status is Status.CANCELED
    assert reg.retry(batch.id, [0])                       # restart just this item
    deadline = _t.time() + 5
    while batch.status not in (Status.DONE, Status.CANCELED) and _t.time() < deadline:
        _t.sleep(0.02)
    assert batch.projects[0].status is Status.DONE        # cancel_requested was cleared
    assert _find(tmp_path / "A", "A", "srt")


def test_retry_while_running_queues_instead_of_starting(two_stream_file, tmp_path):
    reg = BatchRegistry()
    projects = [ProjectInput("A", [_prepared(two_stream_file, "a.mkv", 2.0)])]
    batch = reg.create(projects)
    reg.set_spec(batch.id, _RunSpec(projects, _stub, "folder", str(tmp_path), None))
    batch.status = Status.RUNNING                # pretend the worker is mid-batch
    assert reg.retry(batch.id, [0])
    assert batch.rerun_pending == [0]            # queued for the busy worker to drain
    assert batch.projects[0].status is Status.QUEUED
    assert batch.status is Status.RUNNING        # did NOT start a second worker / finalize


def _perfile_project(two_stream_file):
    return ProjectInput("Lib", [
        _prepared(two_stream_file, "a.mkv", 2.0, sequence=0),
        _prepared(two_stream_file, "b.mkv", 2.0, sequence=1),
    ], per_file=True, formats=["txt"])


def test_per_file_tracks_each_file_status(two_stream_file, tmp_path):
    proj = _perfile_project(two_stream_file)
    batch = BatchRegistry().create([proj])
    assert [f.name for f in batch.projects[0].files] == ["a.mkv", "b.mkv"]   # a row per file
    run_batch(batch, [proj], _stub, output_mode="folder", output_folder=str(tmp_path))
    files = batch.projects[0].files
    assert all(f.status is Status.DONE for f in files)
    assert all(f.output_dir for f in files)


def test_per_file_cancel_one_file_continues_rest(two_stream_file, tmp_path):
    reg = BatchRegistry()
    proj = _perfile_project(two_stream_file)
    batch = reg.create([proj])
    assert reg.cancel_file(batch.id, 0, 0)                    # cancel just file a
    run_batch(batch, [proj], _stub, output_mode="folder", output_folder=str(tmp_path))
    files = batch.projects[0].files
    assert files[0].status is Status.CANCELED
    assert files[1].status is Status.DONE                    # the other file still ran
    assert not _find(tmp_path / "Lib", "a", "txt")
    assert _find(tmp_path / "Lib", "b", "txt")


def test_per_file_restart_reruns_only_that_file(two_stream_file, tmp_path):
    import time as _t
    reg = BatchRegistry()
    proj = _perfile_project(two_stream_file)
    batch = reg.create([proj])
    reg.set_spec(batch.id, _RunSpec([proj], _stub, "folder", str(tmp_path), None))
    run_batch(batch, [proj], _stub, output_mode="folder", output_folder=str(tmp_path))
    files = batch.projects[0].files
    assert all(f.status is Status.DONE for f in files)
    files[1].status = Status.ERROR                           # pretend file b failed
    assert reg.restart_file(batch.id, 0, 1)
    deadline = _t.time() + 5
    while batch.status not in (Status.DONE, Status.CANCELED) and _t.time() < deadline:
        _t.sleep(0.02)
    assert files[1].status is Status.DONE                    # b re-ran and recovered
    assert files[0].status is Status.DONE                    # a was left alone


def test_restart_running_file_is_rejected(two_stream_file, tmp_path):
    reg = BatchRegistry()
    proj = _perfile_project(two_stream_file)
    batch = reg.create([proj])
    reg.set_spec(batch.id, _RunSpec([proj], _stub, "folder", str(tmp_path), None))
    batch.projects[0].files[0].status = Status.RUNNING       # in-flight file
    assert reg.restart_file(batch.id, 0, 0) is False         # only finished files can restart


def test_per_file_progress_is_duration_weighted(two_stream_file, tmp_path):
    # A long file should dominate the project's progress over a short one.
    proj = ProjectInput("W", [
        _prepared(two_stream_file, "long.mkv", 100.0, sequence=0),
        _prepared(two_stream_file, "short.mkv", 1.0, sequence=1),
    ], per_file=True, formats=["txt"])
    batch = BatchRegistry().create([proj])
    files = batch.projects[0].files
    assert files[0].weight == 100.0 and files[1].weight == 1.0
    files[0].progress, files[0].status = 1.0, Status.DONE    # long done, short not started
    files[1].progress = 0.0
    _recalc_project(batch.projects[0])
    assert batch.projects[0].progress > 0.98                 # ~100/101, not 50%


def test_output_mode_source_writes_next_to_file(two_stream_file, tmp_path):
    import shutil
    local = tmp_path / "session.mkv"
    shutil.copy(two_stream_file, local)
    proj = ProjectInput("NextTo", [_prepared(local, "session.mkv", 2.0)])
    batch = _run([proj], output_mode="source")
    assert batch.projects[0].output_dir == str(tmp_path / "NextTo")
    assert _find(tmp_path / "NextTo", "NextTo", "srt")
