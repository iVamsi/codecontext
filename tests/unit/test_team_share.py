import hashlib
import json
import os
import shutil
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from codecontext import team
from codecontext.errors import CodeContextError
from codecontext.git import GitMirror
from codecontext.memory import Memory
from codecontext.models import InitRequest, SearchRequest, SyncRequest
from codecontext.store import Store, _acquire_file_lock
from codecontext.team import execute_join, execute_team_merge, project_remote_key
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


def _merge_setup(tmp_path):
    origin = "https://github.com/acme/widget.git"
    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    _repo(repo_a, origin, b"def hello():\n    return 1\n")
    _repo(repo_b, origin, b"def hello():\n    return 1\n")
    share = tmp_path / "share"
    with Memory.initialize(tmp_path / "store-a", InitRequest(repo_path=repo_a)) as mem:
        mem.sync(SyncRequest(revision="HEAD", offline=True))
        mem.publish(share)
    store_b = repo_b / ".codecontext"
    Memory.join(store_b, share, repo_b)
    return share, store_b, repo_b


def _resign(share) -> None:
    """Recompute manifest hashes so a test can edit the share on purpose."""
    path = share / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for rel in manifest["files"]:
        manifest["files"][rel] = hashlib.sha256((share / rel).read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _object_file(objs):
    """A loose object or pack file, the kinds merge can check on their own."""
    return next(
        p.relative_to(objs)
        for p in sorted(objs.rglob("*"))
        if p.is_file() and (p.suffix == ".pack" or len(p.parent.name) == 2)
    )


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _add_decision(share, claim_id: str, text: str) -> None:
    with sqlite3.connect(share / "memory.sqlite3") as conn:
        conn.execute(
            "INSERT INTO model_runs VALUES ('mr1','fp','rq',1,'completed',1,1,1,1,NULL,NULL,1,1)"
        )
        conn.execute("INSERT INTO episodes VALUES ('ep1','commit',NULL,NULL,'fp','complete',1)")
        conn.execute(
            "INSERT INTO claims (id, episode_id, model_run_id, statement, rationale, support,"
            " lifecycle, state, created_at_us) VALUES (?, 'ep1', 'mr1', ?, 'because', 'direct',"
            " 'active', 'ready', 1)",
            (claim_id, text),
        )


def test_merge_claim_with_new_episode_succeeds_and_is_searchable(tmp_path) -> None:
    share, store_b, repo_b = _merge_setup(tmp_path)
    _add_decision(share, "claim_" + "a" * 64, "adopt zebra caching strategy")
    _resign(share)
    report = execute_team_merge(store_b, share, repo_b)
    assert report.claims_merged == 1
    with Memory.open(store_b) as mem:
        hits = mem.search(SearchRequest(query="zebra caching", kind="decision")).hits
    assert any(h.id == "claim_" + "a" * 64 for h in hits)


def test_merge_ignores_injected_share_column(tmp_path) -> None:
    share, store_b, repo_b = _merge_setup(tmp_path)
    with sqlite3.connect(share / "memory.sqlite3") as conn:
        conn.execute('ALTER TABLE claims ADD COLUMN "x) --" TEXT')
    _add_decision(share, "claim_" + "b" * 64, "ignore injected columns")
    _resign(share)
    execute_team_merge(store_b, share, repo_b)
    with sqlite3.connect(store_b / "memory.sqlite3") as conn:
        assert "x) --" not in [r[1] for r in conn.execute("PRAGMA table_info(claims)")]
        assert conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 1


def test_merge_rejects_share_with_newer_schema(tmp_path) -> None:
    share, store_b, repo_b = _merge_setup(tmp_path)
    with sqlite3.connect(share / "memory.sqlite3") as conn:
        conn.execute("UPDATE store_meta SET schema_version = 99")
    _resign(share)
    objs = store_b / "repository.git" / "objects"
    before = sorted(str(p) for p in objs.rglob("*"))
    with pytest.raises(CodeContextError) as exc:
        execute_team_merge(store_b, share, repo_b)
    assert exc.value.code == "INVALID_REQUEST"
    assert sorted(str(p) for p in objs.rglob("*")) == before


def test_merge_does_not_copy_alternates(tmp_path) -> None:
    share, store_b, repo_b = _merge_setup(tmp_path)
    info = share / "repository.git" / "objects" / "info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "alternates").write_text("/elsewhere/objects\n", encoding="utf-8")
    execute_team_merge(store_b, share, repo_b)
    assert not (store_b / "repository.git" / "objects" / "info" / "alternates").exists()


def test_merge_evidence_pointing_at_new_change_and_hunk_has_no_dangling_fk(tmp_path) -> None:
    share, store_b, repo_b = _merge_setup(tmp_path)
    _add_decision(share, "claim_" + "c" * 64, "evidence needs its change")
    oid = "d" * 40
    with sqlite3.connect(share / "memory.sqlite3") as conn:
        conn.execute(
            "INSERT INTO commits (oid, tree_oid, message, first_seen_at_us) VALUES (?, ?, 'm', 1)",
            (oid, "e" * 40),
        )
        conn.execute(
            "INSERT INTO changes (id, commit_oid, parent_index, change_kind) VALUES ('chg_x', ?, 0, 'M')",
            (oid,),
        )
        conn.execute(
            "INSERT INTO hunks (id, change_id, ordinal, old_start, old_count, new_start, new_count)"
            " VALUES ('hunk_x', 'chg_x', 0, 1, 1, 1, 1)"
        )
        conn.execute(
            "INSERT INTO evidence (id, kind, anchor_json, content_sha256)"
            " VALUES ('ev_x', 'hunk', '{\"hunk_id\": \"hunk_x\"}', 'h')"
        )
        conn.execute(
            "INSERT INTO claim_evidence (claim_id, evidence_id, role) VALUES (?, 'ev_x', 'support')",
            ("claim_" + "c" * 64,),
        )
    _resign(share)
    execute_team_merge(store_b, share, repo_b)
    with sqlite3.connect(store_b / "memory.sqlite3") as conn:
        assert conn.execute("SELECT COUNT(*) FROM hunks WHERE id = 'hunk_x'").fetchone()[0] == 1
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def _join_setup(tmp_path):
    origin = "https://github.com/acme/widget.git"
    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    _repo(repo_a, origin, b"def hello():\n    return 1\n")
    _repo(repo_b, origin, b"def hello():\n    return 1\n")
    share = tmp_path / "share"
    with Memory.initialize(tmp_path / "store-a", InitRequest(repo_path=repo_a)) as mem:
        mem.sync(SyncRequest(revision="HEAD", offline=True))
        mem.publish(share)
    return share, repo_b / ".codecontext", repo_b


def test_join_rejects_tampered_share(tmp_path) -> None:
    share, store_b, repo_b = _join_setup(tmp_path)
    db = share / "memory.sqlite3"
    data = bytearray(db.read_bytes())
    data[-1] ^= 0xFF
    db.write_bytes(bytes(data))
    with pytest.raises(CodeContextError) as exc:
        execute_join(store_b, share, repo_b)
    assert exc.value.code == "INVALID_REQUEST"
    assert not store_b.exists()


def test_join_rejects_unlisted_object(tmp_path) -> None:
    share, store_b, repo_b = _join_setup(tmp_path)
    planted = share / "repository.git" / "objects" / "ab" / ("c" * 38)
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.write_bytes(b"planted")
    with pytest.raises(CodeContextError) as exc:
        execute_join(store_b, share, repo_b)
    assert exc.value.code == "INVALID_REQUEST"


def test_join_rejects_share_with_triggers(tmp_path) -> None:
    share, store_b, repo_b = _join_setup(tmp_path)
    with sqlite3.connect(share / "memory.sqlite3") as conn:
        conn.execute(
            "CREATE TRIGGER evil AFTER UPDATE ON store_meta BEGIN"
            " UPDATE store_meta SET principal_id = 'owned'; END"
        )
    _resign(share)
    with pytest.raises(CodeContextError) as exc:
        execute_join(store_b, share, repo_b)
    assert exc.value.code == "INVALID_REQUEST"
    assert not store_b.exists()


def test_join_rejects_share_with_newer_schema(tmp_path) -> None:
    share, store_b, repo_b = _join_setup(tmp_path)
    with sqlite3.connect(share / "memory.sqlite3") as conn:
        conn.execute("UPDATE store_meta SET schema_version = 99")
    _resign(share)
    with pytest.raises(CodeContextError) as exc:
        execute_join(store_b, share, repo_b)
    assert exc.value.code == "INVALID_REQUEST"
    assert not store_b.exists()


def test_join_while_writer_lock_held_is_store_busy(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(team, "SWAP_LOCK_TIMEOUT_SECONDS", 0.1)
    share, store_b, repo_b = _join_setup(tmp_path)
    execute_join(store_b, share, repo_b)
    # Without a local team.json the store has not seen this publish, so join swaps it.
    (store_b / "team.json").unlink()
    holder = Store.open(store_b)
    try:
        with holder.acquire_writer_lock(timeout_seconds=1):
            with pytest.raises(CodeContextError) as exc:
                execute_join(store_b, share, repo_b)
    finally:
        holder.close()
    assert exc.value.code == "STORE_BUSY"
    assert not (store_b / "team.json").exists()


def test_join_does_not_serve_the_replaced_stores_cached_graph(tmp_path) -> None:
    from codecontext import graph
    from codecontext.retrieval import load_ancestor_commits

    share, store_b, repo_b = _join_setup(tmp_path)
    execute_join(store_b, share, repo_b)
    # Diverge the local copy without touching its generation or revision.
    with sqlite3.connect(store_b / "memory.sqlite3") as conn:
        conn.execute("UPDATE symbols SET name = 'local_only'")
        conn.execute("DELETE FROM snapshot_commits")

    def names_and_ancestors() -> tuple[set[str], frozenset[str]]:
        store = Store.open(store_b, read_only=True)
        try:
            nodes, _edges, _epoch = graph._load_active_graph(store)
            with store.connect_read() as conn:
                ancestors = load_ancestor_commits(conn, store)
        finally:
            store.close()
        return {n.name for n in nodes.values()}, ancestors

    assert names_and_ancestors() == ({"local_only"}, frozenset())
    (store_b / "team.json").unlink()
    execute_join(store_b, share, repo_b)
    names, ancestors = names_and_ancestors()
    assert "hello" in names
    assert ancestors


def test_join_notices_a_write_that_does_not_bump_generation(tmp_path, monkeypatch) -> None:
    share, store_b, repo_b = _join_setup(tmp_path)
    execute_join(store_b, share, repo_b)
    (store_b / "team.json").unlink()
    real_import_share = team._import_share

    def import_then_write_locally(*args, **kwargs):
        real_import_share(*args, **kwargs)
        local = Store.open(store_b)
        try:
            with local.write_transaction(bump_generation=False) as conn:
                conn.execute(
                    "INSERT INTO checkpoints (stream_key, cursor_json, last_success_at_us)"
                    " VALUES ('test:stream', '{}', 1)"
                )
        finally:
            local.close()

    monkeypatch.setattr(team, "_import_share", import_then_write_locally)
    with pytest.raises(CodeContextError) as exc:
        execute_join(store_b, share, repo_b)
    assert exc.value.code == "STORE_BUSY"
    with sqlite3.connect(store_b / "memory.sqlite3") as conn:
        assert conn.execute(
            "SELECT 1 FROM checkpoints WHERE stream_key = 'test:stream'"
        ).fetchone() is not None


def test_join_refuses_when_replaced_backup_exists(tmp_path) -> None:
    share, store_b, repo_b = _join_setup(tmp_path)
    execute_join(store_b, share, repo_b)
    (store_b / "team.json").unlink()
    backup = store_b.parent / f".{store_b.name}.replaced"
    backup.mkdir()
    (backup / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(CodeContextError) as exc:
        execute_join(store_b, share, repo_b)
    assert exc.value.code == "INVALID_REQUEST"
    assert str(backup) in exc.value.message
    assert (backup / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_sync_during_join_keeps_committed_writes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(team, "SWAP_LOCK_TIMEOUT_SECONDS", 30.0)
    share, store_b, repo_b = _join_setup(tmp_path)
    execute_join(store_b, share, repo_b)
    (store_b / "team.json").unlink()
    share_rev = json.loads((share / "team.json").read_text(encoding="utf-8"))["active_revision"]
    new_rev = commit_fixture_spec(
        repo_b, GitCommitSpec(message="more", files={"src/b.py": b"x = 1\n"})
    )

    sync_holds_lock = threading.Event()
    join_read_generation = threading.Event()
    real_store_generation = team._store_generation

    def store_generation(store):
        result = real_store_generation(store)
        join_read_generation.set()
        return result

    monkeypatch.setattr(team, "_store_generation", store_generation)
    errors: list[BaseException] = []
    join_errors: list[CodeContextError] = []

    def sync_with_lock() -> None:
        try:
            with Memory.open(store_b) as mem:
                with mem._store.acquire_writer_lock(timeout_seconds=5):
                    sync_holds_lock.set()
                    # Commit only after join has decided to swap, so join must notice the change.
                    assert join_read_generation.wait(timeout=30)
                    mem.sync(SyncRequest(revision="HEAD", offline=True))
        except BaseException as exc:
            errors.append(exc)
        finally:
            sync_holds_lock.set()

    def join_after_sync_starts() -> None:
        sync_holds_lock.wait(timeout=30)
        try:
            execute_join(store_b, share, repo_b)
        except CodeContextError as exc:
            join_errors.append(exc)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=sync_with_lock), threading.Thread(target=join_after_sync_starts)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=90)
    assert errors == []
    assert [e.code for e in join_errors] == ["STORE_BUSY"]
    with Memory.open(store_b, read_only=True) as mem:
        final_status = mem.status()
    assert final_status.revision == new_rev != share_rev
    with sqlite3.connect(store_b / "memory.sqlite3") as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_writer_waiting_during_join_locks_the_new_store(tmp_path, monkeypatch) -> None:
    share, store_b, repo_b = _join_setup(tmp_path)
    execute_join(store_b, share, repo_b)
    (store_b / "team.json").unlink()
    lock_path = store_b / "tmp" / "writer.lock"
    join_holds_lock = threading.Event()
    real_replace_tree = team._replace_tree

    def replace_tree(source, destination):
        join_holds_lock.set()
        # Give the waiter time to start polling the old lock file.
        threading.Event().wait(0.3)
        real_replace_tree(source, destination)

    monkeypatch.setattr(team, "_replace_tree", replace_tree)
    seen: list[tuple[int, int]] = []
    errors: list[BaseException] = []

    def waiter() -> None:
        join_holds_lock.wait(timeout=30)
        try:
            with _acquire_file_lock(lock_path, timeout_seconds=30) as fd:
                held = os.fstat(fd)
                current = os.stat(lock_path)
                seen.append((held.st_ino, current.st_ino))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=waiter)
    thread.start()
    execute_join(store_b, share, repo_b)
    thread.join(timeout=60)
    assert errors == []
    assert len(seen) == 1 and seen[0][0] == seen[0][1]


def test_truncated_object_is_repaired_on_merge(tmp_path) -> None:
    share, store_b, repo_b = _merge_setup(tmp_path)
    share_objs = share / "repository.git" / "objects"
    rel = _object_file(share_objs)
    local_obj = store_b / "repository.git" / "objects" / rel
    local_obj.write_bytes(b"x")
    execute_team_merge(store_b, share, repo_b)
    assert _sha256(local_obj) == _sha256(share_objs / rel)


def test_merge_rejects_tampered_share_before_copying_objects(tmp_path) -> None:
    share, store_b, repo_b = _merge_setup(tmp_path)
    share_objs = share / "repository.git" / "objects"
    rel = _object_file(share_objs)
    local_obj = store_b / "repository.git" / "objects" / rel
    local_obj.write_bytes(b"x")
    _add_decision(share, "claim_" + "f" * 64, "unsigned change")
    with pytest.raises(CodeContextError) as exc:
        execute_team_merge(store_b, share, repo_b)
    assert exc.value.code == "INVALID_REQUEST"
    assert local_obj.read_bytes() == b"x"


def test_merge_does_not_copy_http_alternates(tmp_path) -> None:
    share, store_b, repo_b = _merge_setup(tmp_path)
    info = share / "repository.git" / "objects" / "info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "http-alternates").write_text("https://evil.example/objects\n", encoding="utf-8")
    execute_team_merge(store_b, share, repo_b)
    assert not (store_b / "repository.git" / "objects" / "info" / "http-alternates").exists()


def test_merge_rejects_malformed_claim_json(tmp_path) -> None:
    share, store_b, repo_b = _merge_setup(tmp_path)
    _add_decision(share, "claim_" + "1" * 64, "bad constraints")
    with sqlite3.connect(share / "memory.sqlite3") as conn:
        conn.execute("UPDATE claims SET constraints_json = 'not json'")
    _resign(share)
    with pytest.raises(CodeContextError) as exc:
        execute_team_merge(store_b, share, repo_b)
    assert exc.value.code == "INVALID_REQUEST"


def test_merge_rejects_non_string_alternatives(tmp_path) -> None:
    share, store_b, repo_b = _merge_setup(tmp_path)
    _add_decision(share, "claim_" + "2" * 64, "bad alternatives")
    with sqlite3.connect(share / "memory.sqlite3") as conn:
        conn.execute("UPDATE claims SET alternatives_json = '[1, 2]'")
    _resign(share)
    with pytest.raises(CodeContextError) as exc:
        execute_team_merge(store_b, share, repo_b)
    assert exc.value.code == "INVALID_REQUEST"


def test_merge_rejects_dangling_foreign_key_and_rolls_back(tmp_path) -> None:
    share, store_b, repo_b = _merge_setup(tmp_path)
    _add_decision(share, "claim_" + "3" * 64, "dangling hunk")
    with sqlite3.connect(share / "memory.sqlite3") as conn:
        conn.execute(
            "INSERT INTO hunks (id, change_id, ordinal, old_start, old_count, new_start, new_count)"
            " VALUES ('hunk_dangling', 'chg_missing', 0, 1, 1, 1, 1)"
        )
    _resign(share)
    with pytest.raises(CodeContextError) as exc:
        execute_team_merge(store_b, share, repo_b)
    assert exc.value.code == "INVALID_REQUEST"
    with sqlite3.connect(store_b / "memory.sqlite3") as conn:
        assert conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 0


def _corrupt_db_copies_from(share, monkeypatch) -> None:
    """Simulate the share db changing after it was opened: the copy gets different bytes."""
    real_copyfile = shutil.copyfile

    def copyfile(src, dst, *args, **kwargs):
        result = real_copyfile(src, dst, *args, **kwargs)
        if str(src) == str((share / "memory.sqlite3").resolve()):
            with open(dst, "ab") as f:
                f.write(b"swapped")
        return result

    monkeypatch.setattr(shutil, "copyfile", copyfile)


def test_join_rejects_share_db_that_changes_during_copy(tmp_path, monkeypatch) -> None:
    share, store_b, repo_b = _join_setup(tmp_path)
    _corrupt_db_copies_from(share, monkeypatch)
    with pytest.raises(CodeContextError) as exc:
        execute_join(store_b, share, repo_b)
    assert exc.value.code == "INVALID_REQUEST"
    assert not store_b.exists()
    assert not any(p.name.startswith(".codecontext.joining-") for p in store_b.parent.iterdir())


def test_merge_rejects_share_db_that_changes_during_copy(tmp_path, monkeypatch) -> None:
    share, store_b, repo_b = _merge_setup(tmp_path)
    _add_decision(share, "claim_" + "4" * 64, "swapped after signing")
    _resign(share)
    db_before = _sha256(store_b / "memory.sqlite3")
    objs = store_b / "repository.git" / "objects"
    objs_before = sorted(str(p) for p in objs.rglob("*"))
    _corrupt_db_copies_from(share, monkeypatch)
    with pytest.raises(CodeContextError) as exc:
        execute_team_merge(store_b, share, repo_b)
    assert exc.value.code == "INVALID_REQUEST"
    assert _sha256(store_b / "memory.sqlite3") == db_before
    assert sorted(str(p) for p in objs.rglob("*")) == objs_before
    assert list((store_b / "tmp").glob(".share-merge-*")) == []


def test_merge_rejects_object_that_changes_during_copy(tmp_path, monkeypatch) -> None:
    share, store_b, repo_b = _merge_setup(tmp_path)
    share_objs = share / "repository.git" / "objects"
    rel = _object_file(share_objs)
    local_obj = store_b / "repository.git" / "objects" / rel
    local_obj.write_bytes(b"x")
    real_copyfile = shutil.copyfile

    def copyfile(src, dst, *args, **kwargs):
        result = real_copyfile(src, dst, *args, **kwargs)
        if str(src) == str((share_objs / rel).resolve()):
            with open(dst, "ab") as f:
                f.write(b"swapped")
        return result

    monkeypatch.setattr(shutil, "copyfile", copyfile)
    with pytest.raises(CodeContextError) as exc:
        execute_team_merge(store_b, share, repo_b)
    assert exc.value.code == "INVALID_REQUEST"
    assert local_obj.read_bytes() == b"x"


def test_merge_keeps_valid_local_object_the_share_disagrees_with(tmp_path) -> None:
    share, store_b, repo_b = _merge_setup(tmp_path)
    local_git = store_b / "repository.git"
    blob = tmp_path / "valid.txt"
    blob.write_bytes(b"a valid local blob\n")
    oid = run_git_fixture(
        repo_b, ["--git-dir", str(local_git), "hash-object", "-w", str(blob)]
    ).decode().strip()
    rel = f"{oid[:2]}/{oid[2:]}"
    local_obj = local_git / "objects" / rel
    before = local_obj.read_bytes()
    shared_obj = share / "repository.git" / "objects" / rel
    shared_obj.parent.mkdir(parents=True, exist_ok=True)
    shared_obj.write_bytes(b"crafted bytes the share signed itself")
    manifest = json.loads((share / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"][f"repository.git/objects/{rel}"] = _sha256(shared_obj)
    (share / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    execute_team_merge(store_b, share, repo_b)
    assert local_obj.read_bytes() == before


def test_join_rejects_share_missing_a_known_trigger(tmp_path) -> None:
    share, store_b, repo_b = _join_setup(tmp_path)
    with sqlite3.connect(share / "memory.sqlite3") as conn:
        conn.execute("DROP TRIGGER search_documents_ai")
    _resign(share)
    with pytest.raises(CodeContextError) as exc:
        execute_join(store_b, share, repo_b)
    assert exc.value.code == "INVALID_REQUEST"
    assert not store_b.exists()


def test_merge_rejects_share_with_unknown_trigger(tmp_path) -> None:
    share, store_b, repo_b = _merge_setup(tmp_path)
    _add_decision(share, "claim_" + "5" * 64, "trigger rides along")
    with sqlite3.connect(share / "memory.sqlite3") as conn:
        conn.execute(
            "CREATE TRIGGER evil AFTER INSERT ON claims BEGIN"
            " UPDATE store_meta SET principal_id = 'owned'; END"
        )
    _resign(share)
    db_before = _sha256(store_b / "memory.sqlite3")
    with pytest.raises(CodeContextError) as exc:
        execute_team_merge(store_b, share, repo_b)
    assert exc.value.code == "INVALID_REQUEST"
    assert _sha256(store_b / "memory.sqlite3") == db_before
