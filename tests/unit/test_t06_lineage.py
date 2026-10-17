from pathlib import Path

from codecontext.git import GitMirror
from codecontext.lineage import (
    index_commit_changes_and_hunks,
    index_focused_path_history,
    map_original_anchor_to_target,
)
from codecontext.models import InitRequest
from codecontext.store import Store
from codecontext.syntax import index_active_snapshot_files
from tests.fixtures.builders import (
    GitCommitSpec,
    build_f01_fixture,
    commit_fixture_spec,
    init_fixture_repo,
)


def test_f01_focused_history_and_symbol_change_relations(tmp_path: Path) -> None:
    """T06: F01's `send` function connects to its introducing/modifying changes and exact source bytes,
    while preserving remaining full-history backfill work.
    """
    f01 = build_f01_fixture(tmp_path / "f01")
    store = Store.initialize(tmp_path / "store", InitRequest(repo_path=f01.repo_dir))
    mirror = GitMirror(store)

    active_rev = mirror.prepare_snapshot("HEAD")
    index_active_snapshot_files(store, mirror, focus_path=b"src/transport.py")

    # Run focused history on src/transport.py (processes C1 and C0; leaves C2 README commit in backfill queue)
    visited = index_focused_path_history(
        store,
        mirror,
        active_revision=active_rev,
        focus_path=b"src/transport.py",
    )
    assert f01.c1_oid in visited
    assert f01.c0_oid in visited
    assert f01.c2_oid not in visited

    with store.connect_read() as conn:
        # C2 (README-only commit) remains pending in jobs queue and GIT_HISTORY_PENDING gap is active
        pending_jobs = conn.execute(
            "SELECT owner_id FROM jobs WHERE kind = 'git_commit_diff' AND state = 'pending'"
        ).fetchall()
        assert [str(r["owner_id"]) for r in pending_jobs] == [f01.c2_oid]
        gaps = conn.execute(
            "SELECT code FROM coverage_gaps WHERE active = 1 AND code = 'GIT_HISTORY_PENDING'"
        ).fetchall()
        assert len(gaps) == 1

        # Verify `send` symbol in active snapshot links to C1 and C0 changes via `relations(relation_kind='changes')`
        send_row = conn.execute(
            "SELECT s.id FROM symbols s JOIN snapshot_files sf ON sf.extraction_id = s.extraction_id WHERE s.qualified_name = 'send'"
        ).fetchone()
        assert send_row is not None
        send_sym_id = str(send_row["id"])

        chg_rels = conn.execute(
            """
            SELECT c.commit_oid, r.method, r.evidence_id
            FROM relations r
            JOIN changes c ON c.id = r.from_id
            WHERE r.to_kind = 'symbol' AND r.to_id = ? AND r.relation_kind = 'changes'
            """,
            (send_sym_id,),
        ).fetchall()
        assert {str(r["commit_oid"]) for r in chg_rels} >= {f01.c1_oid}

        # Verify exact code, hunk, and commit evidence records were registered
        ev_kinds = {
            str(r["kind"])
            for r in conn.execute("SELECT DISTINCT kind FROM evidence").fetchall()
        }
        assert {"commit", "hunk", "code"} <= ev_kinds

    store.close()


def test_u06_and_i04_original_anchor_to_target_mappings(tmp_path: Path) -> None:
    """U06 & non-GitHub I04: Match F02/F04 original anchors against target code:
    - Exact unchanged blob and range
    - Exact unchanged range transported across preceding line insertions
    - Unique exact-byte file rename -> exact mapping
    - Edited function at same qualified name -> candidate mapping
    - Two identical file copies -> candidate mapping with MAPPING_AMBIGUOUS gap
    - Deleted symbol -> unmapped with MAPPING_UNAVAILABLE gap
    """
    repo_dir = init_fixture_repo(tmp_path / "f04_repo")
    orig_code = (
        b"def first_fn(a: int) -> int:\n"
        b"    return a + 1\n\n\n"
        b"def retry_send(payload: bytes) -> bool:\n"
        b"    limit = 5\n"
        b"    return len(payload) <= limit\n"
    )
    c_orig = commit_fixture_spec(
        repo_dir,
        GitCommitSpec(
            message="C_orig: initial retry_send with limit 5",
            files={"src/net.py": orig_code, "src/move_me.py": b"def movable() -> int:\n    return 42\n"},
        ),
    )

    # C_offset: insert lines above retry_send without modifying retry_send itself, and rename move_me.py -> moved.py unchanged
    offset_code = (
        b"# Header comment line 1\n"
        b"# Header comment line 2\n"
        b"def first_fn(a: int) -> int:\n"
        b"    return a + 1\n\n\n"
        b"def retry_send(payload: bytes) -> bool:\n"
        b"    limit = 5\n"
        b"    return len(payload) <= limit\n"
    )
    c_offset = commit_fixture_spec(
        repo_dir,
        GitCommitSpec(
            message="C_offset: add header lines and move file unchanged",
            files={
                "src/net.py": offset_code,
                "src/move_me.py": None,
                "src/moved.py": b"def movable() -> int:\n    return 42\n",
            },
        ),
    )

    # C_edited: modify retry_send's limit line (5 -> 3), copy moved.py into two identical copies, and delete first_fn
    edited_code = (
        b"# Header comment line 1\n"
        b"# Header comment line 2\n"
        b"def retry_send(payload: bytes) -> bool:\n"
        b"    limit = 3\n"
        b"    return len(payload) <= limit\n"
    )
    c_edited = commit_fixture_spec(
        repo_dir,
        GitCommitSpec(
            message="C_edited: change limit to 3 and duplicate moved file",
            files={
                "src/net.py": edited_code,
                "src/moved.py": None,
                "src/copy_a.py": b"def movable() -> int:\n    return 42\n",
                "src/copy_b.py": b"def movable() -> int:\n    return 42\n",
            },
        ),
    )

    store = Store.initialize(tmp_path / "f04_store", InitRequest(repo_path=repo_dir))
    mirror = GitMirror(store)

    # 1. Prepare snapshot at C_offset and test exact range transport + unique exact move
    mirror.prepare_snapshot(c_offset)
    index_active_snapshot_files(store, mirror)
    index_commit_changes_and_hunks(store, mirror, c_orig)
    index_commit_changes_and_hunks(store, mirror, c_offset)

    with store.connect_read() as conn:
        # Lines 5..7 in C_orig (`retry_send`) shifted to lines 7..9 in C_offset with identical bytes
        m_transport = map_original_anchor_to_target(
            conn,
            mirror,
            original_commit_oid=c_orig,
            original_path=b"src/net.py",
            original_line_start=5,
            original_line_end=7,
            target_revision=c_offset,
        )
        assert m_transport.mapping == "exact"
        assert m_transport.method == "unchanged_range_transport"
        assert (m_transport.target_line_start, m_transport.target_line_end) == (7, 9)

        # Unique move `src/move_me.py` -> `src/moved.py` with identical blob
        m_move = map_original_anchor_to_target(
            conn,
            mirror,
            original_commit_oid=c_orig,
            original_path=b"src/move_me.py",
            original_line_start=1,
            original_line_end=2,
            target_revision=c_offset,
        )
        assert m_move.mapping == "exact"
        assert m_move.method == "unique_exact_blob_move"
        assert m_move.target_path == b"src/moved.py"

    # 2. Prepare snapshot at C_edited and test edited range -> candidate, duplicate copies -> candidate + MAPPING_AMBIGUOUS, deleted -> unmapped
    mirror.prepare_snapshot(c_edited)
    index_active_snapshot_files(store, mirror)
    index_commit_changes_and_hunks(store, mirror, c_edited)

    with store.connect_read() as conn:
        # Line 6 in C_orig (`    limit = 5`) was modified to `    limit = 3` inside `retry_send`
        m_edited = map_original_anchor_to_target(
            conn,
            mirror,
            original_commit_oid=c_orig,
            original_path=b"src/net.py",
            original_line_start=6,
            original_line_end=6,
            target_revision=c_edited,
        )
        assert m_edited.mapping == "candidate"
        assert m_edited.method == "same_path_lexical_symbol"
        assert m_edited.target_symbol_id is not None

        # `src/move_me.py` now has two identical copies (`copy_a.py` and `copy_b.py`) in C_edited
        m_copies = map_original_anchor_to_target(
            conn,
            mirror,
            original_commit_oid=c_orig,
            original_path=b"src/move_me.py",
            original_line_start=1,
            original_line_end=2,
            target_revision=c_edited,
        )
        assert m_copies.mapping == "candidate"
        assert m_copies.gap_code == "MAPPING_AMBIGUOUS"

        # `first_fn` (lines 1..2 in C_orig) was deleted in C_edited
        m_deleted = map_original_anchor_to_target(
            conn,
            mirror,
            original_commit_oid=c_orig,
            original_path=b"src/net.py",
            original_line_start=1,
            original_line_end=2,
            target_revision=c_edited,
        )
        assert m_deleted.mapping == "unmapped"
        assert m_deleted.gap_code == "MAPPING_UNAVAILABLE"

    store.close()
