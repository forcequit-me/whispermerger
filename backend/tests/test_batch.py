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


def test_per_file_says_when_a_saved_result_was_reused(two_stream_file, tmp_path):
    from app import transcribe

    def cached(wav):                                   # what transcribe_track does on a cache hit
        transcribe._chunk.hits = transcribe.cache_tally()[0] + 1
        return _stub(wav)

    proj = _perfile_project(two_stream_file)
    batch = BatchRegistry().create([proj])
    run_batch(batch, [proj], cached, output_mode="folder", output_folder=str(tmp_path))
    assert [f.message for f in batch.projects[0].files] == ["reused saved result"] * 2

    merged = ProjectInput("Merged", [_prepared(two_stream_file, "a.mkv", 2.0)])   # one transcript
    batch = BatchRegistry().create([merged])
    run_batch(batch, [merged], cached, output_mode="folder", output_folder=str(tmp_path))
    assert batch.projects[0].message == "reused saved result"


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


def test_per_file_extract_seconds_is_real_ffmpeg_even_when_prefetched(two_stream_file, tmp_path):
    # The second file's audio is extracted in the background while the first file is on the GPU.
    # Its extract_seconds must still be the REAL read+ffmpeg time (the user's "everything pre-GPU"),
    # not the ~0 idle wait the consumer sees once the audio is already prepped. A slow stub holds
    # the GPU long enough to guarantee file b is fully prefetched/hidden.
    import time as _t
    proj = _perfile_project(two_stream_file)                  # two rows, same source file
    def slow_stub(_wav):
        _t.sleep(0.3)
        return _stub(_wav)
    batch = BatchRegistry().create([proj])
    run_batch(batch, [proj], slow_stub, output_mode="folder", output_folder=str(tmp_path))
    files = batch.projects[0].files
    assert all(f.status is Status.DONE for f in files)
    assert files[0].extract_seconds > 0
    # b was hidden behind a's GPU run yet still records real extraction, comparable to a's —
    # the old idle-wait metric reported ~0 here.
    assert files[1].extract_seconds >= files[0].extract_seconds * 0.3


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


def _wait_status(batch, timeout=6.0):
    import time as _t
    deadline = _t.time() + timeout
    while batch.status not in (Status.DONE, Status.CANCELED) and _t.time() < deadline:
        _t.sleep(0.02)


def test_pause_frees_gpu_freezes_clock_then_resumes(two_stream_file, tmp_path):
    import time as _t
    reg = BatchRegistry()
    proj = ProjectInput("A", [_prepared(two_stream_file, "a.mkv", 2.0)])
    batch = reg.create([proj])
    freed = []
    batch.on_pause = lambda: freed.append(True)
    reg.set_spec(batch.id, _RunSpec([proj], _stub, "folder", str(tmp_path), None))
    reg.pause(batch.id)                                   # pause up front -> worker parks at first boundary
    reg.run(batch, lambda: run_batch(batch, [proj], _stub,
            output_mode="folder", output_folder=str(tmp_path)))

    deadline = _t.time() + 3
    while not batch.gpu_freed and _t.time() < deadline:
        _t.sleep(0.01)
    assert batch.gpu_freed is True
    assert freed == [True]                                # GPU-release hook ran on park

    e1 = batch.elapsed
    _t.sleep(0.2)                                         # paused time must not count
    assert batch.elapsed - e1 < 0.1                       # clock frozen while paused

    reg.resume(batch.id)
    _wait_status(batch)
    assert batch.status is Status.DONE
    assert batch.paused_total >= 0.15                     # the pause was banked out of elapsed
    assert _find(tmp_path / "A", "A", "srt")              # batch finished correctly after resume


def test_cancel_while_paused_unwinds(two_stream_file, tmp_path):
    import time as _t
    reg = BatchRegistry()
    proj = ProjectInput("A", [_prepared(two_stream_file, "a.mkv", 2.0)])
    batch = reg.create([proj])
    reg.set_spec(batch.id, _RunSpec([proj], _stub, "folder", str(tmp_path), None))
    reg.pause(batch.id)
    reg.run(batch, lambda: run_batch(batch, [proj], _stub,
            output_mode="folder", output_folder=str(tmp_path)))

    deadline = _t.time() + 3
    while not batch.gpu_freed and _t.time() < deadline:
        _t.sleep(0.01)
    assert batch.gpu_freed is True
    reg.cancel(batch.id)                                  # cancel while paused -> wakes + unwinds
    _wait_status(batch)
    assert batch.status is Status.CANCELED
    assert batch.projects[0].status is Status.CANCELED


def test_output_mode_source_writes_next_to_file(two_stream_file, tmp_path):
    import shutil
    local = tmp_path / "session.mkv"
    shutil.copy(two_stream_file, local)
    proj = ProjectInput("NextTo", [_prepared(local, "session.mkv", 2.0)])
    batch = _run([proj], output_mode="source")
    assert batch.projects[0].output_dir == str(tmp_path / "NextTo")
    assert _find(tmp_path / "NextTo", "NextTo", "srt")


def test_per_file_extraction_failure_isolated(two_stream_file, tmp_path):
    # an unreadable file errors only its own row; the rest of the project still runs
    proj = ProjectInput("Lib", [
        _prepared("C:/nope/missing.mkv", "bad.mkv", 2.0, sequence=0),
        _prepared(two_stream_file, "good.mkv", 2.0, sequence=1),
    ], per_file=True, formats=["txt"])
    batch = _run([proj], output_mode="folder", output_folder=str(tmp_path))
    files = batch.projects[0].files
    assert files[0].status is Status.ERROR and files[0].error
    assert files[1].status is Status.DONE
    assert _find(tmp_path / "Lib", "good", "txt")


def test_restart_finished_per_file_project_reruns_all(two_stream_file, tmp_path):
    import time as _t
    reg = BatchRegistry()
    proj = _perfile_project(two_stream_file)
    batch = reg.create([proj])
    reg.set_spec(batch.id, _RunSpec([proj], _stub, "folder", str(tmp_path), None))
    run_batch(batch, [proj], _stub, output_mode="folder", output_folder=str(tmp_path))
    files = batch.projects[0].files
    first = [f.finished_at for f in files]
    _t.sleep(0.01)
    assert reg.retry(batch.id, [0])
    deadline = _t.time() + 5
    while batch.status not in (Status.DONE, Status.CANCELED) and _t.time() < deadline:
        _t.sleep(0.02)
    assert all(f.status is Status.DONE for f in files)
    assert all(f.finished_at > t for f, t in zip(files, first))   # every file actually re-ran


def test_same_name_merge_projects_do_not_overwrite(two_stream_file, tmp_path):
    projects = [ProjectInput("Dup", [_prepared(two_stream_file, "a.mkv", 2.0)], formats=["txt"]),
                ProjectInput("Dup", [_prepared(two_stream_file, "b.mkv", 2.0)], formats=["txt"])]
    _run(projects, output_mode="folder", output_folder=str(tmp_path))
    assert _find(tmp_path / "Dup", "Dup", "txt")
    assert _find(tmp_path / "Dup", "Dup_2", "txt")


def _asking_stub(_wav):
    """A transcribe stub that stops to ask, as transcribe_track does before a big download."""
    from app.transcribe import _asker
    choice, language = _asker.fn({"language": "nn", "text": "big download?"})
    return {"language": language or "nn", "segments": []}


def _start_asking(tmp_path, proj):
    import time as _t
    reg = BatchRegistry()
    batch = reg.create([proj])
    reg.set_spec(batch.id, _RunSpec([proj], _asking_stub, "folder", str(tmp_path), None))
    reg.run(batch, lambda: run_batch(batch, [proj], _asking_stub,
            output_mode="folder", output_folder=str(tmp_path)))
    deadline = _t.time() + 3
    while batch.question is None and _t.time() < deadline:
        _t.sleep(0.01)
    return reg, batch


def test_batch_waits_for_an_answer_like_a_pause(two_stream_file, tmp_path):
    import time as _t
    proj = ProjectInput("A", [_prepared(two_stream_file, "a.mkv", 2.0), _prepared(two_stream_file, "b.mkv", 2.0)],
                        per_file=True)
    reg, batch = _start_asking(tmp_path, proj)
    a = batch.projects[0].files[0]
    assert batch.question["item"] == "a.mkv" and batch.question["text"] == "big download?"
    assert a.message == "waiting for your answer: big download?"       # on the file's row
    assert batch.projects[0].message == "a.mkv · waiting for your answer: big download?"
    assert batch.gpu_freed
    e1 = (batch.elapsed, a.elapsed)
    _t.sleep(0.2)
    assert batch.elapsed - e1[0] < 0.1 and a.elapsed - e1[1] < 0.1    # clocks frozen, as when paused

    assert reg.answer(batch.id, "language", "en")
    deadline = _t.time() + 3
    while (batch.question or {}).get("item") != "b.mkv" and _t.time() < deadline:
        _t.sleep(0.01)
    assert batch.question["item"] == "b.mkv"           # an answer is for one file: the next one asks again
    assert reg.answer(batch.id, "skip")
    _wait_status(batch)
    assert batch.status is Status.DONE and batch.question is None
    assert [f.result.languages[0].language for f in batch.projects[0].files] == ["en", "nn"]
    assert reg.answer(batch.id, "skip") is False       # nothing is waiting any more


def test_cancel_while_batch_waits_for_an_answer(two_stream_file, tmp_path):
    proj = ProjectInput("A", [_prepared(two_stream_file, "a.mkv", 2.0), _prepared(two_stream_file, "b.mkv", 2.0)],
                        per_file=True)
    reg, batch = _start_asking(tmp_path, proj)
    reg.cancel_file(batch.id, 0, 0)                    # cancel just the file that is asking
    import time as _t
    deadline = _t.time() + 3
    while (batch.question or {}).get("item") != "b.mkv" and _t.time() < deadline:
        _t.sleep(0.01)
    assert batch.projects[0].files[0].status is Status.CANCELED       # skipped, not failed
    assert batch.question["item"] == "b.mkv"           # the queue moved on to the next file
    reg.cancel(batch.id)                               # cancel the whole batch while it waits
    _wait_status(batch)
    assert batch.status is Status.CANCELED and batch.question is None


def test_cancel_during_the_last_track_writes_nothing(two_stream_file, tmp_path):
    """A cancel that arrives while a file's last track is being transcribed leaves that file
    canceled with nothing written, like a cancel between tracks: of the file, its project or the
    whole batch. (It used to be written and shown "done", in a project that said "canceled".)"""
    def run(name, cancel):
        reg, proj, calls = BatchRegistry(), _perfile_project(two_stream_file), []
        batch = reg.create([proj])

        def stub(wav):                                     # the cancel arrives while b.mkv's only track runs
            calls.append(wav)
            if len(calls) == 2:
                cancel(reg, batch.id)
            return _stub(wav)

        run_batch(batch, [proj], stub, output_mode="folder", output_folder=str(tmp_path / name))
        assert _find(tmp_path / name / "Lib", "a", "txt") and not _find(tmp_path / name / "Lib", "b", "txt")
        return batch.projects[0].status, [f.status for f in batch.projects[0].files]

    done, canceled = Status.DONE, Status.CANCELED
    assert run("file", lambda reg, bid: reg.cancel_file(bid, 0, 1)) == (done, [done, canceled])
    assert run("project", lambda reg, bid: reg.cancel_project(bid, 0)) == (canceled, [done, canceled])
    assert run("batch", lambda reg, bid: reg.cancel(bid)) == (canceled, [done, canceled])


def test_a_batch_waiting_for_the_gpu_pauses_and_cancels_at_once(two_stream_file, tmp_path):
    """While another run has the GPU, transcribe_track reports "waiting" every half second. In a
    batch that report is a checkpoint: a pause parks there and a cancel stops there, instead of
    after the other run has finished and this track has been transcribed."""
    import threading
    import time as _t
    from app.transcribe import _chunk
    turn, transcribed = threading.Event(), []

    def waiting_stub(wav):                                 # never gets its turn in this test
        while not turn.is_set():
            _chunk.fn(0.0, waiting=True)
            _t.sleep(0.01)
        transcribed.append(wav)
        return _stub(wav)

    def until(check):
        deadline = _t.time() + 3
        while not check() and _t.time() < deadline:
            _t.sleep(0.01)
        assert check()

    reg, proj = BatchRegistry(), _perfile_project(two_stream_file)
    batch = reg.create([proj])
    reg.run(batch, lambda: run_batch(batch, [proj], waiting_stub,
            output_mode="folder", output_folder=str(tmp_path)))
    a = batch.projects[0].files[0]
    until(lambda: a.message.endswith("waiting for another run to finish"))
    reg.pause(batch.id)
    until(lambda: batch.gpu_freed)                         # parked while still waiting
    e1 = a.elapsed
    _t.sleep(0.2)
    assert a.elapsed - e1 < 0.1                            # its clock stopped
    reg.resume(batch.id)
    until(lambda: not batch.gpu_freed)
    reg.cancel_project(batch.id, 0)
    _wait_status(batch)
    turn.set()
    assert batch.projects[0].status is Status.CANCELED and not transcribed
    assert all(f.status is Status.CANCELED for f in batch.projects[0].files)
