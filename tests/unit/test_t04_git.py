import shutil
from pathlib import Path

import pytest

from codecontext.errors import CodeContextError
from codecontext.git import GitMirror, parse_raw_commit_object, parse_unified_zero_diff
from codecontext.models import InitRequest
from codecontext.store import Store
from tests.fixtures.builders import (
    GitCommitSpec,
    build_f01_fixture,
    build_f03_fixture,
    commit_fixture_spec,
    init_fixture_repo,
    run_git_fixture,
)


def test_git_mirror_prepare_snapshot_and_survives_source_deletion(tmp_path: Path) -> None:
    """I01 & T04: Import requested commit and declared refs without changing developer checkout;
    all selected source bytes remain readable after detaching/deleting source fixture.
    Also verify executable hooks/fsmonitor in source repo never run.
    """
    f01 = build_f01_fixture(tmp_path / "f01")
    marker_file = tmp_path / "hook_ran_marker"

    # Configure a hostile fsmonitor and hook in the source repository
    hook_script = tmp_path / "evil_hook.sh"
    hook_script.write_text(f"#!/bin/sh\ntouch {marker_file}\nexit 1\n", encoding="utf-8")
    hook_script.chmod(0o755)
    run_git_fixture(f01.repo_dir, ["config", "core.fsmonitor", str(hook_script)])

    # Record developer checkout state before initialization & sync
    before_head = run_git_fixture(f01.repo_dir, ["rev-parse", "HEAD"])
    before_refs = run_git_fixture(f01.repo_dir, ["for-each-ref"])
    before_status = run_git_fixture(f01.repo_dir, ["status", "--porcelain"])

    store_dir = tmp_path / "store"
    store = Store.initialize(store_dir, InitRequest(repo_path=f01.repo_dir))
    mirror = GitMirror(store)

    active_rev = mirror.prepare_snapshot("HEAD")
    assert active_rev == f01.c2_oid
    assert not marker_file.exists(), "Configured fsmonitor/hook must never execute!"

    # Developer checkout is completely untouched
    assert run_git_fixture(f01.repo_dir, ["rev-parse", "HEAD"]) == before_head
    assert run_git_fixture(f01.repo_dir, ["for-each-ref"]) == before_refs
    assert run_git_fixture(f01.repo_dir, ["status", "--porcelain"]) == before_status

    # Verify idempotent repeat prepare_snapshot does not duplicate commits or files
    mirror.prepare_snapshot("HEAD")
    st = store.status()
    assert st.revision == f01.c2_oid
    assert st.commit_count == 3
    assert st.snapshot_file_count == 2

    # Delete the developer's source repository completely!
    shutil.rmtree(f01.repo_dir)
    assert not f01.repo_dir.exists()

    # Owned bare mirror still serves all historical blobs and commits via cat-file --batch
    with mirror.open_batch_session() as batch:
        c1_obj = batch.read_object(f01.c1_oid)
        assert c1_obj is not None and c1_obj[0] == "commit"
        pc = parse_raw_commit_object(f01.c1_oid, c1_obj[2])
        assert "Bound send retries to 3 attempts" in pc.message_text

        entries = mirror.list_tree_entries(f01.c1_oid)
        by_path = {e.path: e for e in entries}
        assert b"src/transport.py" in by_path
        blob_obj = batch.read_object(by_path[b"src/transport.py"].oid)
        assert blob_obj is not None and blob_obj[0] == "blob"
        assert blob_obj[2] == f01.c1_code

    store.close()


def test_f03_i05_branch_switching_and_Merge_diffs(tmp_path: Path) -> None:
    """F03 & I05: Prepare one branch, reject unprepared branch in resolve_prepared_revision,
    switch branches atomically, and verify parent-specific merge diffs and root empty-tree diffs.
    """
    f03 = build_f03_fixture(tmp_path / "f03")
    store_dir = tmp_path / "store"
    store = Store.initialize(store_dir, InitRequest(repo_path=f03.repo_dir))
    mirror = GitMirror(store)

    # Before first snapshot preparation, resolve_prepared_revision raises REVISION_NOT_PREPARED
    with store.connect_read() as conn:
        with pytest.raises(CodeContextError) as exc_info:
            mirror.resolve_prepared_revision(conn, None)
        assert exc_info.value.code == "REVISION_NOT_PREPARED"

    # Prepare main branch
    rev_main = mirror.prepare_snapshot("main")
    assert rev_main == f03.backdated_oid

    with store.connect_read() as conn:
        assert mirror.resolve_prepared_revision(conn, None) == f03.backdated_oid
        assert mirror.resolve_prepared_revision(conn, "main") == f03.backdated_oid
        assert mirror.resolve_prepared_revision(conn, "HEAD") == f03.backdated_oid

        # Requesting unprepared branch `divergent` raises REVISION_NOT_PREPARED
        with pytest.raises(CodeContextError) as exc_info:
            mirror.resolve_prepared_revision(conn, "divergent")
        assert exc_info.value.code == "REVISION_NOT_PREPARED"
        assert exc_info.value.details["active_revision"] == f03.backdated_oid

    # Root commit diff against empty tree (parent_index=-1, parent_oid=None)
    root_changes = mirror.diff_tree_parent(None, f03.root_oid, parent_index=-1)
    assert len(root_changes) == 2
    assert {c.new_path for c in root_changes} == {b"a.py", b"b.py"}
    assert all(c.change_kind == "added" and c.old_blob_oid is None for c in root_changes)

    # Two-parent merge commit has distinct changes against each parent
    p0_changes = mirror.diff_tree_parent(f03.branch_a_oid, f03.merge_oid, parent_index=0)
    p1_changes = mirror.diff_tree_parent(f03.branch_b_oid, f03.merge_oid, parent_index=1)
    assert [c.new_path for c in p0_changes] == [b"b.py"]
    assert [c.new_path for c in p1_changes] == [b"a.py"]

    # Unified-zero hunk parsing on a.py change
    hunks = mirror.diff_blobs_unified_zero(p1_changes[0].old_blob_oid, p1_changes[0].new_blob_oid)
    assert len(hunks) == 1
    assert hunks[0].old_start == 2 and hunks[0].old_count == 1
    assert hunks[0].new_start == 2 and hunks[0].new_count == 1

    # Switch active snapshot to `divergent` branch atomically
    rev_div = mirror.prepare_snapshot("divergent")
    assert rev_div == f03.divergent_oid
    with store.connect_read() as conn:
        assert mirror.resolve_prepared_revision(conn, "divergent") == f03.divergent_oid
        with pytest.raises(CodeContextError) as exc_info:
            mirror.resolve_prepared_revision(conn, "main")
        assert exc_info.value.code == "REVISION_NOT_PREPARED"

        # Reachable commits in snapshot_commits reflect `divergent` (root + divergent = 2 commits)
        snap_commits = {
            str(r["commit_oid"])
            for r in conn.execute("SELECT commit_oid FROM snapshot_commits").fetchall()
        }
        assert snap_commits == {f03.root_oid, f03.divergent_oid}
        assert f03.merge_oid not in snap_commits

    store.close()


def test_f09_unusual_paths_and_folded_commit_headers(tmp_path: Path) -> None:
    """Verify space/tab/special paths in ls-tree and folded headers in commit objects."""
    repo_dir = init_fixture_repo(tmp_path / "weird_repo")
    c_oid = commit_fixture_spec(
        repo_dir,
        GitCommitSpec(
            message="Commit with unusual filenames",
            files={
                "dir with spaces/file\twith\ttab.py": b"x = 1\n",
                "normal.py": b"y = 2\n",
            },
        ),
    )
    store_dir = tmp_path / "store"
    store = Store.initialize(store_dir, InitRequest(repo_path=repo_dir))
    mirror = GitMirror(store)
    mirror.prepare_snapshot("HEAD")

    entries = mirror.list_tree_entries(c_oid)
    paths = {e.path for e in entries}
    assert b"dir with spaces/file\twith\ttab.py" in paths

    # Test folded commit header (e.g., gpgsig with blank folded lines)
    raw_commit = (
        b"tree " + (b"a" * 40) + b"\n"
        b"parent " + (b"b" * 40) + b"\n"
        b"author Alice <alice@example.test> 1700000000 +0000\n"
        b"committer Alice <alice@example.test> 1700000000 +0000\n"
        b"gpgsig -----BEGIN PGP SIGNATURE-----\n"
        b" \n"
        b" iQIzBAABCAAdFiEE...\n"
        b" -----END PGP SIGNATURE-----\n"
        b"\n"
        b"Real commit subject line\n\nBody paragraph.\n"
    )
    pc = parse_raw_commit_object("c" * 40, raw_commit)
    assert pc.tree_oid == "a" * 40
    assert pc.parent_oids == ["b" * 40]
    assert pc.message_text == "Real commit subject line\n\nBody paragraph.\n"
    assert raw_commit[pc.header_byte_len :] == pc.message_bytes

    # Test inconsistent hunk line count rejection
    bad_diff = b"@@ -1,2 +1,1 @@\n-only_one_old_line\n+one_new_line\n"
    with pytest.raises(CodeContextError) as exc_info:
        parse_unified_zero_diff(bad_diff)
    assert exc_info.value.code == "IO_ERROR"

    store.close()
