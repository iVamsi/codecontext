import hashlib
import sqlite3
from pathlib import Path

import pytest

from codecontext.archive import execute_purge
from codecontext.models import InitRequest, PurgeRequest
from codecontext.store import Store, get_migration_sql_and_hash, split_sql_statements
from tests.fixtures.builders import build_f01_fixture


def _dependency_conn(edges: list[tuple[tuple[str, str], tuple[str, str]]]) -> sqlite3.Connection:
    """In-memory schema with (dependent, input) edges."""
    sql, _ = get_migration_sql_and_hash()
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript(sql)
    for (dk, di), (ik, ii) in edges:
        conn.execute(
            "INSERT INTO dependencies (dependent_kind, dependent_id, input_kind, input_id)"
            " VALUES (?, ?, ?, ?)",
            (dk, di, ik, ii),
        )
    return conn


def test_invalidate_diamond_dependencies_does_not_raise() -> None:
    conn = _dependency_conn(
        [
            (("claim", "c1"), ("source_version", "v")),
            (("claim", "c2"), ("source_version", "v")),
            (("summary", "s"), ("claim", "c1")),
            (("summary", "s"), ("claim", "c2")),
        ]
    )
    stale = Store.invalidate_dependents_recursive(conn, [("source_version", "v")])
    assert stale == {("claim", "c1"), ("claim", "c2"), ("summary", "s")}


def test_invalidate_real_cycle_still_terminates() -> None:
    conn = _dependency_conn(
        [
            (("claim", "c2"), ("claim", "c1")),
            (("claim", "c1"), ("claim", "c2")),
        ]
    )
    stale = Store.invalidate_dependents_recursive(conn, [("claim", "c1")])
    assert stale == {("claim", "c1"), ("claim", "c2")}


def test_split_sql_keeps_trigger_body_whole() -> None:
    sql = "CREATE TABLE t(x);\nCREATE TRIGGER g AFTER INSERT ON t BEGIN SELECT 1; SELECT 2; END;"
    statements = split_sql_statements(sql)
    assert len(statements) == 2
    assert statements[1].endswith("END")


def test_split_sql_ignores_semicolons_in_strings_and_comments() -> None:
    sql = (
        "-- leading; comment\n"
        "INSERT INTO t VALUES ('a;b', \"c;d\"); /* block; comment */\n"
        "INSERT INTO t VALUES ('it''s;');\n"
        "SELECT 1"
    )
    statements = split_sql_statements(sql)
    assert len(statements) == 3
    assert "'a;b'" in statements[0]
    assert "'it''s;'" in statements[1]
    assert statements[2] == "SELECT 1"


def test_split_sql_matches_real_migration_via_executescript() -> None:
    sql, _ = get_migration_sql_and_hash()
    conn = sqlite3.connect(":memory:", isolation_level=None)
    for stmt in split_sql_statements(sql):
        if not stmt.upper().startswith("PRAGMA "):
            conn.execute(stmt)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")}
    assert names


# An over-budget counter forces status() to shrink the gap list all the way down.
def _status_statement_count(tmp_path: Path, gap_count: int, monkeypatch: pytest.MonkeyPatch) -> int:
    fixture = build_f01_fixture(tmp_path / "repo")
    store = Store.initialize(tmp_path / "store", InitRequest(repo_path=fixture.repo_dir))
    with store.write_transaction() as conn:
        for i in range(gap_count):
            Store.upsert_gap(
                conn,
                repository_id=store.config.repository_id,
                code="GIT_HISTORY_PENDING",
                scope=f"git:ref:{i:03d}",
                message="x" * 40,
                retryable=True,
            )
    statements: list[str] = []
    calls: list[int] = []
    original = Store.build_coverage_and_gaps

    def traced(conn: sqlite3.Connection, **kwargs: int):  # type: ignore[no-untyped-def]
        calls.append(1)
        conn.set_trace_callback(statements.append)
        return original(conn, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Store, "build_coverage_and_gaps", staticmethod(traced))
        store.status(token_counter=lambda text: 10_000)
    assert len(calls) == 1
    return len(statements)


def test_status_runs_coverage_queries_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    one = _status_statement_count(tmp_path / "a", 1, monkeypatch)
    fifty = _status_statement_count(tmp_path / "b", 50, monkeypatch)
    assert fifty == one


def test_split_sql_handles_non_ascii_identifiers_and_strings() -> None:
    statements = split_sql_statements("SELECT 'é;ü' AS naïve; SELECT 1")
    assert statements == ["SELECT 'é;ü' AS naïve", "SELECT 1"]


def test_migration_003_indexes_used(tmp_path: Path) -> None:
    fixture = build_f01_fixture(tmp_path / "repo")
    store = Store.initialize(tmp_path / "store", InitRequest(repo_path=fixture.repo_dir))
    assert store.schema_version == 3
    with store.connect_read() as conn:
        plans = {
            "relations_evidence": "DELETE FROM relations WHERE evidence_id = 'e'",
            "coverage_gaps_record": "DELETE FROM coverage_gaps WHERE record_id = 'r'",
            "coverage_gaps_scope_active": "SELECT 1 FROM coverage_gaps WHERE scope = 's' AND active = 1",
        }
        for index, sql in plans.items():
            plan = conn.execute(f"EXPLAIN QUERY PLAN {sql}").fetchall()
            assert index in str(list(map(tuple, plan)))
    store.close()


def test_read_only_open_of_v2_store_does_not_migrate(tmp_path: Path) -> None:
    fixture = build_f01_fixture(tmp_path / "repo")
    store_dir = tmp_path / "store"
    store = Store.initialize(store_dir, InitRequest(repo_path=fixture.repo_dir))
    db_path = store.db_path
    store.close()
    conn = sqlite3.connect(db_path, isolation_level=None)
    for index in ("relations_evidence", "coverage_gaps_record", "coverage_gaps_scope_active"):
        conn.execute(f"DROP INDEX {index}")
    conn.execute("DELETE FROM schema_migrations WHERE version = 3")
    conn.execute("UPDATE store_meta SET schema_version = 2")
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    before = hashlib.sha256(db_path.read_bytes()).hexdigest()

    ro = Store.open(store_dir, read_only=True)
    assert ro.schema_version == 2
    with ro.connect_read() as rconn:
        assert rconn.execute("SELECT COUNT(*) FROM sources").fetchone()[0] >= 0
    ro.close()

    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before
    rw = Store.open(store_dir)
    assert rw.schema_version == 3
    rw.close()


def test_purge_reports_cleanup_pending_when_rmtree_fails(tmp_path: Path) -> None:
    fixture = build_f01_fixture(tmp_path / "repo")
    store_dir = tmp_path / "store"
    store = Store.initialize(store_dir, InitRequest(repo_path=fixture.repo_dir))
    stuck = store.store_path / "tmp"
    (stuck / "inner").mkdir(parents=True, exist_ok=True)
    (stuck / "inner" / "f").write_text("x")
    (stuck / "inner").chmod(0o500)
    try:
        report = execute_purge(store, PurgeRequest(entire_store=True))
        assert report.cleanup_pending is True
    finally:
        (stuck / "inner").chmod(0o700)

