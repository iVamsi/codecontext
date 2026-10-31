import sqlite3
from pathlib import Path

from codecontext.models import InitRequest
from codecontext.store import (
    SCHEMA_VERSION,
    Store,
    get_all_migrations,
)
from tests.fixtures.builders import build_f01_fixture


def test_migration_v1_to_v2_sequential_upgrade() -> None:
    """Verify that applying 001 and then 002 upgrades the database schema correctly."""
    migrations = get_all_migrations()
    assert len(migrations) == 2
    assert migrations[0][0] == 1
    assert migrations[1][0] == 2

    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")

    # Apply v1
    conn.executescript(migrations[0][1])
    # Check that v1 does not have revision_snapshots yet
    table_names = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]
    assert "revision_snapshots" not in table_names
    assert "document_embeddings" not in table_names

    # Apply v2
    conn.executescript(migrations[1][1])
    v2_tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]
    assert "revision_snapshots" in v2_tables
    assert "revision_ancestors" in v2_tables
    assert "document_embeddings" in v2_tables

    # Check store_meta columns added in v2
    meta_cols = [
        r[1] for r in conn.execute("PRAGMA table_info(store_meta)").fetchall()
    ]
    assert "gitlab_project_id" in meta_cols
    assert "gitlab_name" in meta_cols
    assert "gitlab_base_url" in meta_cols
    assert "workspace_base_revision" in meta_cols

    # Verify key indexes created by 002
    indexes = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
    ]
    expected_indexes = [
        "syntax_occ_ext_kind",
        "syntax_occ_enclosing",
        "snapshot_files_ext",
        "snapshot_files_blob",
        "symbols_qualified_name",
        "symbols_parent",
        "relations_method_state",
        "evidence_git_oid_kind",
        "sources_parent",
        "episodes_commit",
        "episodes_primary_source",
        "commits_diff_state",
        "commit_parents_parent",
        "search_code_path",
        "revision_snapshots_ext",
    ]
    for idx in expected_indexes:
        assert idx in indexes, f"Expected index {idx} was not found"

    conn.close()


def test_store_initialize_creates_v2_store(tmp_path: Path) -> None:
    """Verify Store.initialize creates a v2 database with both migrations recorded."""
    f01 = build_f01_fixture(tmp_path / "repo")
    store_dir = tmp_path / "store"

    req = InitRequest(
        repo_path=f01.repo_dir,
        gitlab_repository="group/project",
        gitlab_base_url="https://gitlab.example.com",
    )
    store = Store.initialize(store_dir, req)
    assert store.schema_version == SCHEMA_VERSION
    assert store.schema_version == 2

    with store.connect_read() as conn:
        mig_rows = conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [r[0] for r in mig_rows] == [1, 2]

        meta_row = conn.execute(
            "SELECT schema_version, gitlab_name, gitlab_base_url FROM store_meta WHERE singleton = 1"
        ).fetchone()
        assert meta_row[0] == 2
        assert meta_row[1] == "group/project"
        assert meta_row[2] == "https://gitlab.example.com"

    store.close()

    # Re-open in read-only mode
    ro_store = Store.open(store_dir, read_only=True)
    assert ro_store.read_only is True
    assert ro_store.schema_version == 2
    ro_store.close()
