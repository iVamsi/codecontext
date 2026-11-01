import sqlite3
from pathlib import Path

import pytest

from codecontext.models import InitRequest
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
