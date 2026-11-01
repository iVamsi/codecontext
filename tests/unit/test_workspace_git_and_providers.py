import json
import os
from pathlib import Path

import httpx
import pytest

from codecontext import InitRequest, Memory, SyncRequest
from codecontext.errors import CodeContextError
from codecontext.git import CatFileBatchSession
from codecontext.github import GitHubClient, collect_github_graphql_backfill
from codecontext.identity import validate_gitlab_project_path
from codecontext.lineage import (
    ISSUE_TRAILER_RE,
    PR_TRAILER_RE,
    REVERT_COMMIT_RE,
    _extract_and_persist_commit_message_relations,
)
from codecontext.store import Store
from tests.fixtures.builders import (
    GitCommitSpec,
    build_f01_fixture,
    commit_fixture_spec,
    init_fixture_repo,
)


def test_buffered_cat_file_batch_session(tmp_path: Path) -> None:
    """Verify buffered CatFileBatchSession performs bounded, thread-safe object reads."""
    f01 = build_f01_fixture(tmp_path / "repo")
    empty_hooks = Path(os.devnull)

    with CatFileBatchSession(f01.repo_dir / ".git", empty_hooks) as session:
        # Read HEAD commit by oid
        res = session.read_object(f01.c2_oid)
        assert res is not None
        obj_type, size, content = res
        assert obj_type == "commit"
        assert size > 0
        assert len(content) == size

        # Read nonexistent object returns None
        assert session.read_object("0" * len(f01.c2_oid)) is None

        # Read object with max_bytes bound
        bounded_res = session.read_object_bounded(f01.c2_oid, max_bytes=10)
        assert bounded_res is not None
        bounded_type, bounded_size, bounded_content = bounded_res
        assert bounded_type == "commit"
        assert bounded_content is None or len(bounded_content) <= 10


def test_commit_trailers_parsing_and_reverts(tmp_path: Path) -> None:
    """Verify offline commit message parsing for issues, PR numbers, and reverts."""
    msg = """feat: support semantic search (#123)

Fixes #456
Closes #789
This reverts commit 1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b.
"""
    # Test regexes
    revert_match = REVERT_COMMIT_RE.search(msg)
    assert revert_match is not None
    assert revert_match.group(1).lower() == "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b"

    pr_matches = [
        pm.group("squash_pr") or pm.group("merge_pr") or pm.group("gl_mr")
        for pm in PR_TRAILER_RE.finditer(msg)
    ]
    assert "123" in pr_matches

    issue_matches = [im.group("num") for im in ISSUE_TRAILER_RE.finditer(msg)]
    assert "456" in issue_matches
    assert "789" in issue_matches

    # Test persistence of extracted relations into database
    f01 = build_f01_fixture(tmp_path / "repo")
    store = Store.initialize(
        tmp_path / "store",
        InitRequest(repo_path=f01.repo_dir, github_repository="owner/repo"),
    )
    with store.write_transaction() as conn:
        conn.execute("UPDATE store_meta SET github_repository_id = '12345' WHERE singleton = 1")
        conn.execute(
            """
            INSERT INTO evidence (
                id, kind, source_version_id, git_oid, anchor_json, content_sha256, state
            ) VALUES ('ev_commit_1', 'commit', NULL, ?, '{}', 'hash1', 'available')
            """,
            ("c" * 40,),
        )
        _extract_and_persist_commit_message_relations(
            conn,
            commit_oid="c" * 40,
            message_text=msg,
            commit_ev_id="ev_commit_1",
        )
        rels = conn.execute(
            "SELECT relation_kind, to_id FROM relations WHERE from_id = ?",
            ("c" * 40,),
        ).fetchall()
        kinds = [r[0] for r in rels]
        assert "supersedes" in kinds
        assert "associated_pr" in kinds
        assert "references_issue" in kinds
    store.close()


def test_gitlab_project_path_validation() -> None:
    """Verify GitLab project path validation supports nested groups and rejects invalid patterns."""
    assert validate_gitlab_project_path("group/project") == "group/project"
    assert validate_gitlab_project_path("group/subgroup/project") == "group/subgroup/project"
    assert validate_gitlab_project_path("org/team/subteam/repo.git") == "org/team/subteam/repo.git"

    with pytest.raises(CodeContextError):
        validate_gitlab_project_path("/leading/slash")
    with pytest.raises(CodeContextError):
        validate_gitlab_project_path("trailing/slash/")
    with pytest.raises(CodeContextError):
        validate_gitlab_project_path("no_slash")


def test_docs_ingest_adr_and_markdown(tmp_path: Path) -> None:
    """Verify ADR and Markdown architecture document ingestion into store."""
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(
        repo,
        GitCommitSpec(
            message="Add ADRs and Architecture documentation",
            files={
                "docs/adr/0001-use-sqlite.md": b"# 1. Use SQLite for Storage\n\n## Status\nAccepted\n\n## Context\nWe need embedded fast ACID storage.\n\n## Decision\nWe will use SQLite in WAL mode.\n\n## Consequences\nZero network hops, fast local queries.\n",
                "docs/architecture.md": b"# System Architecture\n\n## Components\n1. Store engine\n2. Graph analysis\n3. Retrieval engine\n",
            },
        ),
    )

    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True, ingest_docs=True))

        with mem._store.connect_read() as conn:
            sources = conn.execute(
                "SELECT kind, provider_id FROM sources WHERE kind IN ('adr', 'design_doc')"
            ).fetchall()
            assert len(sources) >= 1
            src_kinds = [r[0] for r in sources]
            assert "adr" in src_kinds or "design_doc" in src_kinds

            docs = conn.execute(
                "SELECT title, body FROM search_documents WHERE record_kind = 'history'"
            ).fetchall()
            assert len(docs) >= 1
            titles = [r[0] for r in docs]
            assert any("sqlite" in t.lower() or "architecture" in t.lower() for t in titles)


def test_github_graphql_backfill_helper(tmp_path: Path) -> None:
    """collect_github_graphql_backfill sends its query through GitHubClient and stores the PRs."""
    f01 = build_f01_fixture(tmp_path / "repo")
    store = Store.initialize(
        tmp_path / "store",
        InitRequest(repo_path=f01.repo_dir, github_repository="owner/repo"),
    )
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "data": {
                    "repository": {
                        "pullRequests": {
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                            "nodes": [
                                {
                                    "number": 42,
                                    "title": "Add hybrid search",
                                    "body": "Implements BM25 + vector RRF",
                                    "url": "https://github.com/owner/repo/pull/42",
                                    "state": "MERGED",
                                    "createdAt": "2026-01-01T00:00:00Z",
                                    "updatedAt": "2026-01-02T00:00:00Z",
                                    "author": {"login": "alice", "databaseId": 1234},
                                    "headRefOid": "h" * 40,
                                    "mergeCommit": {"oid": "m" * 40},
                                }
                            ],
                        },
                    }
                }
            },
        )

    with GitHubClient(token="fake_token", transport=httpx.MockTransport(handler)) as client:
        used, s_add, v_add, done = collect_github_graphql_backfill(
            store=store,
            client=client,
            repo_uuid=store.config.repository_id,
            repo_prov_id="gh:owner/repo",
            repo_full_name="owner/repo",
            remaining_requests=5,
        )
    assert used == 1
    assert s_add >= 1
    assert done is True
    assert [(r.method, r.url.path) for r in seen] == [("POST", "/graphql")]
    assert seen[0].headers["Authorization"] == "Bearer fake_token"
    assert json.loads(seen[0].content)["variables"] == {"owner": "owner", "name": "repo", "cursor": None}
    store.close()
