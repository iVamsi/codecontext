import json
import sqlite3

import pytest

from codecontext.errors import CodeContextError
from codecontext.memory import Memory
from codecontext.models import InitRequest, SearchRequest, SyncRequest
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

    store_b = repo_b / ".codecontext"
    status = Memory.join(store_b, share, repo_b)
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

    repo_c = tmp_path / "repo-c"
    _repo(repo_c, "https://github.com/other/widget.git", b"def hello():\n    return 1\n")
    with pytest.raises(CodeContextError) as mismatch:
        Memory.join(tmp_path / "store-c", share, repo_c)
    assert mismatch.value.code == "REPOSITORY_MISMATCH"
