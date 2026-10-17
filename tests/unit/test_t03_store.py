import sqlite3
import threading
from pathlib import Path

import pytest

from codecontext.errors import CodeContextError
from codecontext.identity import (
    make_claim_id,
    make_episode_id,
    make_extraction_id,
    make_src_id,
    make_sum_id,
    make_sym_id,
    make_ver_id,
)
from codecontext.models import InitRequest
from codecontext.store import (
    Store,
    decode_and_validate_cursor,
    encode_cursor,
    get_migration_sql_and_hash,
)
from tests.fixtures.builders import build_f01_fixture, build_f03_fixture

NOW_US_VAL = 1700000000000000


def test_migration_in_memory_fts_and_foreign_keys() -> None:
    """Apply migration to in-memory SQLite; verify foreign keys and FTS5 triggers."""
    sql, _ = get_migration_sql_and_hash()
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(sql)

    # Foreign key constraint check: inserting symbol with non-existent extraction fails
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO symbols (
                id, extraction_id, parent_symbol_id, kind, name, qualified_name,
                byte_start, byte_end, body_byte_start, body_byte_end, line_start, line_end, signature
            ) VALUES ('sym_1', 'ext_missing', NULL, 'function', 'f', 'f', 0, 10, 0, 10, 1, 1, 'def f()')
            """
        )

    # FTS5 trigger insertion, update, and deletion
    conn.execute(
        """
        INSERT INTO search_documents (record_kind, record_id, chunk_ordinal, title, identifiers, body, metadata_json)
        VALUES ('code', 'sym_1', 0, 'send retry function', 'send MAX_RETRIES', 'retry three attempts on timeout', '{}')
        """
    )
    hits = conn.execute(
        "SELECT rowid FROM search_fts WHERE search_fts MATCH '\"retry\"'"
    ).fetchall()
    assert len(hits) == 1

    # Update document
    conn.execute(
        "UPDATE search_documents SET title = 'send function', body = 'updated body text exponential' WHERE record_id = 'sym_1'"
    )
    assert (
        len(
            conn.execute(
                "SELECT rowid FROM search_fts WHERE search_fts MATCH '\"retry\"'"
            ).fetchall()
        )
        == 0
    )
    assert (
        len(
            conn.execute(
                "SELECT rowid FROM search_fts WHERE search_fts MATCH '\"exponential\"'"
            ).fetchall()
        )
        == 1
    )

    # Delete document
    conn.execute("DELETE FROM search_documents WHERE record_id = 'sym_1'")
    assert (
        len(
            conn.execute(
                "SELECT rowid FROM search_fts WHERE search_fts MATCH '\"exponential\"'"
            ).fetchall()
        )
        == 0
    )
    conn.close()


def test_store_init_open_idempotency_and_mismatch(tmp_path: Path) -> None:
    """Verify Store.initialize, Store.open, missing store error, idempotent re-init, and REPOSITORY_MISMATCH."""
    f01 = build_f01_fixture(tmp_path / "r1")
    f03 = build_f03_fixture(tmp_path / "r2")
    store_dir = tmp_path / "store"

    # Missing store does not auto-create on open
    with pytest.raises(CodeContextError) as exc_info:
        Store.open(store_dir)
    assert exc_info.value.code == "STORE_NOT_INITIALIZED"
    assert not store_dir.exists()

    # Arbitrary subdirectory inside repository checkout (other than `.codecontext`) is rejected,
    # while `<repo>/.codecontext` is permitted and auto-creates `.gitignore`.
    with pytest.raises(CodeContextError) as exc_info:
        Store.initialize(f01.repo_dir / "src" / "bad_store", InitRequest(repo_path=f01.repo_dir))
    assert exc_info.value.code == "INVALID_REQUEST"

    dot_store = Store.initialize(f01.repo_dir / ".codecontext", InitRequest(repo_path=f01.repo_dir))
    assert (f01.repo_dir / ".codecontext" / ".gitignore").read_text(encoding="utf-8") == "*\n"
    dot_store.close()

    # Initialize store
    req = InitRequest(
        repo_path=f01.repo_dir,
        github_repository="example/transport",
        allowed_issue_repositories=["example/shared-issues"],
    )
    store = Store.initialize(store_dir, req)
    st = store.status()
    assert st.revision is None
    assert st.generation == 0
    assert st.access_epoch == 0
    assert st.coverage.git.state == "not_started"
    store.close()

    # Re-initializing with same binding and normalized config succeeds idempotently
    store_again = Store.initialize(store_dir, req)
    assert store_again.config.repository_id == st.repository_id
    store_again.close()

    # Re-initializing with different repository or config raises REPOSITORY_MISMATCH
    with pytest.raises(CodeContextError) as exc_info:
        Store.initialize(store_dir, InitRequest(repo_path=f03.repo_dir))
    assert exc_info.value.code == "REPOSITORY_MISMATCH"

    with pytest.raises(CodeContextError) as exc_info:
        Store.initialize(
            store_dir,
            InitRequest(repo_path=f01.repo_dir, github_repository="other/repo"),
        )
    assert exc_info.value.code == "REPOSITORY_MISMATCH"


def test_store_schema_checksum_and_version_validation(tmp_path: Path) -> None:
    """Reject modified migration checksums and unsupported future schema versions."""
    f01 = build_f01_fixture(tmp_path / "r1")
    store_dir = tmp_path / "store"
    store = Store.initialize(store_dir, InitRequest(repo_path=f01.repo_dir))
    store.close()

    # Tamper with migration checksum
    conn = sqlite3.connect(str(store_dir / "memory.sqlite3"))
    conn.execute("UPDATE schema_migrations SET migration_sha256 = 'bad_hash'")
    conn.commit()
    conn.close()

    with pytest.raises(CodeContextError) as exc_info:
        Store.open(store_dir)
    assert exc_info.value.code == "STORE_VERSION_UNSUPPORTED"

    # Restore checksum, tamper with schema_version to 99
    _, valid_hash = get_migration_sql_and_hash()
    conn = sqlite3.connect(str(store_dir / "memory.sqlite3"))
    conn.execute("UPDATE schema_migrations SET migration_sha256 = ?", (valid_hash,))
    conn.execute("UPDATE store_meta SET schema_version = 99")
    conn.commit()
    conn.close()

    with pytest.raises(CodeContextError) as exc_info:
        Store.open(store_dir)
    assert exc_info.value.code == "STORE_VERSION_UNSUPPORTED"


def test_blob_reuse_and_recursive_invalidation(tmp_path: Path) -> None:
    """Insert one blob reused at 100 paths; test recursive invalidation of dependent claims and summaries."""
    f01 = build_f01_fixture(tmp_path / "r1")
    store_dir = tmp_path / "store"
    store = Store.initialize(store_dir, InitRequest(repo_path=f01.repo_dir))

    blob_oid = "a" * 40
    ext_id = make_extraction_id(blob_oid=blob_oid, language="python", extractor_fingerprint="fp1")
    sym_id = make_sym_id(
        blob_oid=blob_oid,
        language="python",
        extractor_fingerprint="fp1",
        byte_start=0,
        byte_end=25,
        kind="function",
        qualified_name="send",
    )

    with store.write_transaction() as conn:
        conn.execute(
            "INSERT INTO extractions (id, blob_oid, language, extractor_fingerprint, state) VALUES (?, ?, 'python', 'fp1', 'complete')",
            (ext_id, blob_oid),
        )
        conn.execute(
            """
            INSERT INTO symbols (
                id, extraction_id, parent_symbol_id, kind, name, qualified_name,
                byte_start, byte_end, body_byte_start, body_byte_end, line_start, line_end, signature
            ) VALUES (?, ?, NULL, 'function', 'send', 'send', 0, 25, 10, 25, 1, 2, 'def send()')
            """,
            (sym_id, ext_id),
        )
        for i in range(100):
            conn.execute(
                "INSERT INTO snapshot_files (path, blob_oid, mode, extraction_id, content_state) VALUES (?, ?, 33188, ?, 'indexed')",
                (f"pkg/mod_{i}.py".encode(), blob_oid, ext_id),
            )

        # Verify only 1 extraction and 1 symbol row exist for 100 paths
        assert int(conn.execute("SELECT COUNT(*) FROM extractions").fetchone()[0]) == 1
        assert int(conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]) == 1
        assert int(conn.execute("SELECT COUNT(*) FROM snapshot_files").fetchone()[0]) == 100

        # Insert source, source_version, episode, model_run, claim, summary, and dependencies
        src_id = make_src_id(provider="github", repository_provider_id="1", kind="pull_request", provider_id="10")
        ver_id = make_ver_id(source_id=src_id, content_sha256="b" * 64)
        ep_id = make_episode_id(repository_id=store.config.repository_id, pr_source_id=src_id)
        claim_id = make_claim_id(
            repository_id=store.config.repository_id,
            episode_id=ep_id,
            request_fingerprint="rfp1",
            statement="Limit retries to 3",
            rationale="Avoid pool exhaustion",
            alternatives=[],
            constraints=[],
            support="explicit",
            lifecycle="adopted",
            evidence_roles=[],
        )
        sum_id = make_sum_id(
            repository_id=store.config.repository_id,
            scope_kind="repository",
            scope_id="scope_root",
            revision=blob_oid,
            input_fingerprint="ifp1",
            model_fingerprint="mfp1",
            prompt_version="v1",
        )

        conn.execute(
            "INSERT INTO sources (id, provider, repository_provider_id, provider_id, kind, current_version_id, availability, last_seen_at_us) VALUES (?, 'github', '1', '10', 'pull_request', ?, 'available', ?)",
            (src_id, ver_id, NOW_US_VAL),
        )
        conn.execute(
            "INSERT INTO source_versions (id, source_id, content_sha256, title, body, raw_payload_json, first_observed_at_us, last_observed_at_us) VALUES (?, ?, ?, 'Title', 'Body', '{}', ?, ?)",
            (ver_id, src_id, "b" * 64, NOW_US_VAL, NOW_US_VAL),
        )
        conn.execute(
            "INSERT INTO episodes (id, kind, primary_source_id, commit_oid, input_fingerprint, state) VALUES (?, 'pr', ?, NULL, 'ifp1', 'inspected')",
            (ep_id, src_id),
        )
        conn.execute(
            "INSERT INTO model_runs (id, model_fingerprint, request_fingerprint, schema_version, state, reserved_input_tokens, reserved_output_tokens, created_at_us) VALUES ('run_1', 'mfp1', 'rfp1', 1, 'complete', 100, 100, ?)",
            (NOW_US_VAL,),
        )
        conn.execute(
            "INSERT INTO claims (id, episode_id, model_run_id, statement, rationale, support, lifecycle, state, created_at_us) VALUES (?, ?, 'run_1', 'Limit retries to 3', 'Avoid pool exhaustion', 'explicit', 'adopted', 'ready', ?)",
            (claim_id, ep_id, NOW_US_VAL),
        )
        conn.execute(
            "INSERT INTO summaries (id, scope_kind, scope_id, revision, model_run_id, text, statements_json, coverage_json, state, input_fingerprint) VALUES (?, 'repository', 'scope_root', ?, 'run_1', 'Summary text', '[]', '{}', 'ready', 'ifp1')",
            (sum_id, blob_oid),
        )
        conn.execute(
            "INSERT INTO search_documents (record_kind, record_id, chunk_ordinal, title, identifiers, body, metadata_json) VALUES ('decision', ?, 0, 'Limit retries', '', 'Avoid pool exhaustion', '{}')",
            (claim_id,),
        )
        conn.execute(
            "INSERT INTO search_documents (record_kind, record_id, chunk_ordinal, title, identifiers, body, metadata_json) VALUES ('summary', ?, 0, 'Summary', '', 'Summary text', '{}')",
            (sum_id,),
        )
        conn.execute(
            "INSERT INTO dependencies (dependent_kind, dependent_id, input_kind, input_id) VALUES ('claim', ?, 'source_version', ?)",
            (claim_id, ver_id),
        )
        conn.execute(
            "INSERT INTO dependencies (dependent_kind, dependent_id, input_kind, input_id) VALUES ('summary', ?, 'claim', ?)",
            (sum_id, claim_id),
        )

    # Now invalidate ver_id in a new generation
    with store.write_transaction() as conn:
        invalidated = Store.invalidate_dependents_recursive(conn, [("source_version", ver_id)])
        assert ("claim", claim_id) in invalidated
        assert ("summary", sum_id) in invalidated

    with store.connect_read() as conn:
        c_state = conn.execute("SELECT state FROM claims WHERE id = ?", (claim_id,)).fetchone()["state"]
        s_state = conn.execute("SELECT state FROM summaries WHERE id = ?", (sum_id,)).fetchone()["state"]
        fts_cnt = int(
            conn.execute(
                "SELECT COUNT(*) FROM search_documents WHERE record_kind IN ('decision', 'summary')"
            ).fetchone()[0]
        )
        assert c_state == "stale"
        assert s_state == "stale"
        assert fts_cnt == 0
    store.close()


def test_u14_u15_and_i09_generation_consistency_and_cursors(tmp_path: Path) -> None:
    """Verify U14 (cursor validation), U15 (gap aggregation & omission), and I09 (reader/writer generation isolation)."""
    f01 = build_f01_fixture(tmp_path / "r1")
    store_dir = tmp_path / "store"
    store = Store.initialize(store_dir, InitRequest(repo_path=f01.repo_dir))

    # U14: Cursor encoding & stale checks
    cur = encode_cursor(query_fingerprint="q_fp_1", generation=1, access_epoch=0, offset=20)
    offset, extra = decode_and_validate_cursor(
        cur,
        expected_query_fingerprint="q_fp_1",
        current_generation=1,
        current_access_epoch=0,
    )
    assert offset == 20 and extra == {}

    # Generation mismatch -> CURSOR_STALE
    with pytest.raises(CodeContextError) as exc_info:
        decode_and_validate_cursor(
            cur,
            expected_query_fingerprint="q_fp_1",
            current_generation=2,
            current_access_epoch=0,
        )
    assert exc_info.value.code == "CURSOR_STALE"

    # Access epoch mismatch -> CURSOR_STALE
    with pytest.raises(CodeContextError) as exc_info:
        decode_and_validate_cursor(
            cur,
            expected_query_fingerprint="q_fp_1",
            current_generation=1,
            current_access_epoch=1,
        )
    assert exc_info.value.code == "CURSOR_STALE"

    # Query fingerprint mismatch -> CURSOR_STALE
    with pytest.raises(CodeContextError) as exc_info:
        decode_and_validate_cursor(
            cur,
            expected_query_fingerprint="q_fp_other",
            current_generation=1,
            current_access_epoch=0,
        )
    assert exc_info.value.code == "CURSOR_STALE"

    # U15: Insert 15 active gaps; verify status caps detailed gaps at 10, preserves aggregate count 15, omits 5, and provides first_gap_id
    with store.write_transaction() as conn:
        for i in range(15):
            Store.upsert_gap(
                conn,
                repository_id=store.config.repository_id,
                code="GIT_HISTORY_PENDING",
                scope=f"git:ref:{i:02d}",
                message=f"Pending ref {i}",
                retryable=True,
            )

    st = store.status()
    assert st.generation == 1
    assert len(st.gaps) == 10
    assert st.coverage.active_gap_count == 15
    assert st.budget.truncated is True
    assert st.budget.omitted_counts.gaps == 5
    assert st.coverage.first_gap_id is not None
    assert len(st.expand) == 1
    assert st.expand[0].id == st.coverage.first_gap_id

    # I09: Reader snapshot isolation while writer prepares next generation
    writer_entered = threading.Event()
    allow_writer_commit = threading.Event()
    writer_finished = threading.Event()

    def writer_worker() -> None:
        with store.write_transaction() as w_conn:
            Store.upsert_gap(
                w_conn,
                repository_id=store.config.repository_id,
                code="GITHUB_BACKFILL_PENDING",
                scope="github:issues",
                message="Issues pending",
                retryable=True,
            )
            writer_entered.set()
            allow_writer_commit.wait(timeout=5.0)
        writer_finished.set()

    t = threading.Thread(target=writer_worker)
    t.start()
    assert writer_entered.wait(timeout=5.0)

    # Reader started while writer has uncommitted changes sees generation 1 and 15 gaps
    with store.read_snapshot() as r_snap:
        assert r_snap.generation == 1
        cnt = int(r_snap.conn.execute("SELECT COUNT(*) FROM coverage_gaps WHERE active = 1").fetchone()[0])
        assert cnt == 15

    allow_writer_commit.set()
    t.join(timeout=5.0)
    assert writer_finished.is_set()

    # Subsequent reader sees generation 2 and 16 gaps
    with store.read_snapshot() as r_snap_after:
        assert r_snap_after.generation == 2
        cnt_after = int(r_snap_after.conn.execute("SELECT COUNT(*) FROM coverage_gaps WHERE active = 1").fetchone()[0])
        assert cnt_after == 16

    # Interrupted write rolls back cleanly and leaves generation 2 intact
    with pytest.raises(RuntimeError, match="abort write"):
        with store.write_transaction() as bad_conn:
            Store.upsert_gap(
                bad_conn,
                repository_id=store.config.repository_id,
                code="STRUCTURE_PENDING",
                scope="blob:temp",
                message="Should roll back",
                retryable=True,
            )
            raise RuntimeError("abort write")

    with store.read_snapshot() as r_snap_final:
        assert r_snap_final.generation == 2
        cnt_final = int(r_snap_final.conn.execute("SELECT COUNT(*) FROM coverage_gaps WHERE active = 1").fetchone()[0])
        assert cnt_final == 16

    store.close()
