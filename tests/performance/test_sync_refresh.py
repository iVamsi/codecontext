import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from codecontext import InitRequest, Memory, SyncRequest
from codecontext.git import git_blob_oid
from tests.fixtures.builders import (
    GitCommitSpec,
    commit_fixture_spec,
    init_fixture_repo,
    run_git_fixture,
)


def test_populated_sync_refresh(tmp_path: Path) -> None:
    """Print local timings; assert reused parses rather than machine-dependent speed."""
    root = tmp_path
    repo = init_fixture_repo(root / "repo")
    commit_fixture_spec(
        repo,
        GitCommitSpec(
            message="Base",
            files={f"f{i}.py": f"def call{i}(): pass\n".encode() for i in range(120)},
        ),
    )
    for i in range(12):
        commit_fixture_spec(
            repo,
            GitCommitSpec(
                message=f"History {i}",
                files={"history.py": f"def history(): return {i}\n".encode()},
            ),
        )
    with Memory.initialize(root / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True, max_seconds=120))
        results = {}
        for case in ["unchanged", "edit", "rename", "delete"]:
            if case == "edit":
                (repo / "f0.py").write_text("def edited(): call1()\n")
            if case == "rename":
                run_git_fixture(repo, ["mv", "f0.py", "moved.py"])
            if case == "delete":
                (repo / "moved.py").unlink()
            start = time.monotonic()
            report = mem.sync(SyncRequest(offline=True, max_seconds=120))
            results[case] = {
                "seconds": round(time.monotonic() - start, 3),
                "blobs_parsed": report.blobs_parsed,
                "commits_added": report.commits_added,
            }
        print(json.dumps(results, indent=2))
        assert results["unchanged"]["blobs_parsed"] == 0
        assert results["unchanged"]["commits_added"] == 0
        assert results["edit"]["blobs_parsed"] == 1
        assert results["rename"]["blobs_parsed"] == results["delete"]["blobs_parsed"] == 0
        assert mem.search_graph({"name_pattern": "^call1$"}).nodes
        assert not mem.search_graph({"name_pattern": "^edited$"}).nodes


def _record_git_spawns(monkeypatch: pytest.MonkeyPatch) -> tuple[list[list[str]], dict[str, bool]]:
    spawned: list[list[str]] = []
    state = {"on": True}

    class RecordingPopen(subprocess.Popen):  # type: ignore[type-arg]
        def __init__(self, args, *rest, **kwargs):  # type: ignore[no-untyped-def]
            if state["on"] and args and args[0] == "git":
                spawned.append([str(a) for a in args])
            super().__init__(args, *rest, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", RecordingPopen)
    return spawned, state


def test_32_commits_use_at_most_two_git_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codecontext import sync

    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(repo, GitCommitSpec(message="Base", files={"keep.txt": b"keep\n"}))
    for i in range(31):
        files: dict[str | bytes, bytes | None] = {f"f{i}.txt": f"{i}\n".encode()}
        if i:
            files[f"f{i - 1}.txt"] = None
        commit_fixture_spec(repo, GitCommitSpec(message=f"C{i}", files=files))
    spawned, state = _record_git_spawns(monkeypatch)
    state["on"] = False
    original = sync._drain_commit_diff_jobs

    def measured(*args, **kwargs):  # type: ignore[no-untyped-def]
        state["on"] = True
        try:
            return original(*args, **kwargs)
        finally:
            state["on"] = False

    monkeypatch.setattr(sync, "_drain_commit_diff_jobs", measured)
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        report = mem.sync(SyncRequest(offline=True, max_seconds=120))
        assert report.coverage.git.state == "complete"
        with mem._store.connect_read() as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE kind = 'git_commit_diff' AND state = 'complete'"
            ).fetchone()[0] == 32
            assert conn.execute(
                "SELECT COUNT(*) FROM changes WHERE change_kind = 'renamed'"
            ).fetchone()[0] == 0
    assert len(spawned) <= 2, spawned


def test_2000_untracked_files_use_few_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(repo, GitCommitSpec(message="Base", files={"base.txt": b"base\n"}))
    names = [f"u/{i}.txt".encode() for i in range(2000)] + [b"new\nline.txt", b'"quoted.txt']
    for name in names:
        target = Path(os.fsdecode(bytes(repo) + b"/" + name))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(name + b" body\n")
    spawned, _ = _record_git_spawns(monkeypatch)
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        report = mem.sync(SyncRequest(offline=True, include_untracked=True, max_seconds=120))
        tree = {e.path: e.oid for e in mem._mirror.list_tree_entries(report.revision)}
    for name in names:
        assert name in tree
    assert tree[b"u/7.txt"] == git_blob_oid(b"u/7.txt body\n")
    # Typed writes (`-t`) are the empty objects and the snapshot commit, not workspace content.
    blob_writes = [args for args in spawned if "hash-object" in args and "-t" not in args]
    assert len(blob_writes) <= 1, len(blob_writes)
