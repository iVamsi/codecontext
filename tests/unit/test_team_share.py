import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from codecontext.errors import CodeContextError
from codecontext.git import GitMirror
from codecontext.memory import Memory
from codecontext.models import InitRequest, SearchRequest, SyncRequest
from codecontext.store import Store
from codecontext.team import project_remote_key
from tests.fixtures.builders import (
    GitCommitSpec,
    commit_fixture_spec,
    init_fixture_repo,
    run_git_fixture,
)


def test_project_remote_key_drops_credentials_and_matches_transports() -> None:
    assert project_remote_key("https://github.com/acme/widget.git") == "github.com/acme/widget"
    assert project_remote_key("git@github.com:acme/widget.git") == "github.com/acme/widget"
    assert project_remote_key("https://oauth2:secret@github.com/acme/widget.git") == "github.com/acme/widget"
    assert (
        project_remote_key("ssh://git@gitlab.example:2222/team/app.git")
        == "gitlab.example:2222/team/app"
    )


def _repo(path, origin: str, body: bytes) -> None:
    init_fixture_repo(path)
    commit_fixture_spec(path, GitCommitSpec(message="init", files={"src/app.py": body}))
    run_git_fixture(path, ["remote", "add", "origin", origin])


def test_publish_join_and_newer_generation(tmp_path) -> None:
    origin = "https://github.com/acme/widget.git"
    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    _repo(repo_a, origin, b"def hello():\n    return 1\n")
    _repo(repo_b, "git@github.com:acme/widget.git", b"def hello():\n    return 1\n")
    share = tmp_path / "share"
    store_a = tmp_path / "store-a"

    with Memory.initialize(store_a, InitRequest(repo_path=repo_a)) as mem:
        mem.sync(SyncRequest(revision="HEAD", offline=True))
        first = mem.publish(share)
        second = mem.publish(share)
    assert first.generation == second.generation
    assert "already at this generation" in second.warnings[0]
    team = json.loads((share / "team.json").read_text(encoding="utf-8"))
    assert team["project_remote"] == "github.com/acme/widget"
    assert json.loads((share / "store.json").read_text(encoding="utf-8"))["repo_path"] is None
    with sqlite3.connect(share / "memory.sqlite3") as conn:
        row = conn.execute(
            "SELECT local_repository_path, principal_id FROM store_meta WHERE singleton = 1"
        ).fetchone()
    assert row == (None, None)

    hook = share / "tmp" / "_empty_hooks" / "reference-transaction"
    hook.parent.mkdir(parents=True)
    hook.write_text("#!/bin/sh\n", encoding="utf-8")
    hook.chmod(0o755)
    (share / "repository.git" / "hooks").mkdir()
    (share / "repository.git" / "hooks" / "post-checkout").write_text("#!/bin/sh\n", encoding="utf-8")

    store_b = repo_b / ".codecontext"
    sibling = store_b.parent / f".{store_b.name}.joining"
    sibling.mkdir()
    marker = sibling / "unrelated.txt"
    marker.write_text("keep", encoding="utf-8")
    status = Memory.join(store_b, share, repo_b)
    assert marker.read_text(encoding="utf-8") == "keep"
    assert (store_b / ".gitignore").read_text(encoding="utf-8") == "*\n"
    assert not (store_b / "tmp" / "_empty_hooks" / "reference-transaction").exists()
    assert not (store_b / "repository.git" / "hooks").exists()
    planted = store_b / "tmp" / "_empty_hooks" / "reference-transaction"
    planted.parent.mkdir(parents=True, exist_ok=True)
    hook_marker = tmp_path / "hook-ran"
    planted.write_text(f'#!/bin/sh\ntouch "{hook_marker}"\n', encoding="utf-8")
    planted.chmod(0o755)
    opened = Store.open(store_b)
    try:
        GitMirror(opened).prepare_snapshot("HEAD")
    finally:
        opened.close()
    assert not hook_marker.exists()
    status_out = run_git_fixture(repo_b, ["status", "--porcelain", "--untracked-files=all"]).decode()
    assert "memory.sqlite3" not in status_out
    assert status.read_only is False
    assert status.snapshot_file_count >= 1
    with Memory.open(store_b) as mem:
        found = mem.search(SearchRequest(query="hello"))
        assert found.hits
        mem.sync(SyncRequest(revision="HEAD", offline=True))
        updated = mem.publish(share)
    assert updated.generation > first.generation

    with Memory.open(store_a, read_only=True) as mem:
        with pytest.raises(CodeContextError) as stale:
            mem.publish(share)
    assert stale.value.code == "INVALID_REQUEST"

    junk = tmp_path / "junk"
    junk.mkdir()
    (junk / "keep.txt").write_text("safe", encoding="utf-8")
    with pytest.raises(CodeContextError) as foreign:
        Memory.join(junk, share, repo_b)
    assert foreign.value.code == "INVALID_REQUEST"
    assert (junk / "keep.txt").read_text(encoding="utf-8") == "safe"

    repo_c = tmp_path / "repo-c"
    _repo(repo_c, "https://github.com/other/widget.git", b"def hello():\n    return 1\n")
    with pytest.raises(CodeContextError) as mismatch:
        Memory.join(tmp_path / "store-c", share, repo_c)
    assert mismatch.value.code == "REPOSITORY_MISMATCH"


def test_concurrent_readers_leave_runtime_directory_untouched(tmp_path) -> None:
    repo = tmp_path / "repo"
    _repo(repo, "https://github.com/acme/widget.git", b"def hello():\n    return 1\n")
    store = tmp_path / "store"
    with Memory.initialize(store, InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True))
    hooks = store / "tmp" / "_empty_hooks"
    hooks.mkdir(exist_ok=True)
    marker = hooks / "unrelated.txt"
    marker.write_text("keep", encoding="utf-8")

    def read_store(_index):
        for _ in range(20):
            with Memory.open(store, read_only=True) as mem:
                mem.configure_workspace(repo)
                mem.context({"query": "hello"})

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(read_store, range(4)))
    assert marker.read_text(encoding="utf-8") == "keep"
