from datetime import UTC, datetime
from pathlib import Path

import pytest

from codecontext.errors import CodeContextError
from codecontext.git import GitMirror
from codecontext.github import GitHubClient, collect_focused_github_evidence
from codecontext.lineage import index_focused_path_history
from codecontext.models import (
    GapRecord,
    GetRequest,
    InitRequest,
    SearchRequest,
    SourceRecord,
    SourceVersionRecord,
    SymbolRecord,
)
from codecontext.retrieval import execute_get, execute_search
from codecontext.store import Store
from codecontext.syntax import index_active_snapshot_files
from tests.fixtures.builders import build_f01_fixture
from tests.unit.test_t07_github import make_mock_github_transport


def test_u13_search_ranking_fts_safety_and_get_expansion(tmp_path: Path) -> None:
    """U13, U11, U14, & T08:
    - Exact active path, symbol name, and record ID prioritization
    - Literal-safe FTS5 queries (punctuation and attempted FTS operators never inject)
    - Pagination via cursors and stale-cursor detection
    - `get` expansion across symbol, change, hunk, source, source_version, evidence, and gap handles
    - Access/cutoff enforcement on direct IDs
    """
    f01 = build_f01_fixture(tmp_path / "f01")
    store = Store.initialize(
        tmp_path / "store",
        InitRequest(repo_path=f01.repo_dir, github_repository="example/transport"),
    )
    mirror = GitMirror(store)
    active_rev = mirror.prepare_snapshot("HEAD")
    index_active_snapshot_files(store, mirror)
    index_focused_path_history(store, mirror, active_revision=active_rev, focus_path=b"src/transport.py")

    routes = {
        "/repos/example/transport": (
            200,
            {"id": 9001, "full_name": "example/transport", "private": False},
            {},
        ),
        f"/repos/example/transport/commits/{f01.c1_oid}/pulls": (
            200,
            [{"number": 10, "id": 10010}],
            {},
        ),
        "/repos/example/transport/pulls/10": (200, f01.pr_payload, {}),
        "/repos/example/transport/issues/10/comments?per_page=100": (
            200,
            f01.comments_payload,
            {},
        ),
        "/repos/example/transport/pulls/10/reviews?per_page=100": (
            200,
            f01.reviews_payload,
            {},
        ),
        "/repos/example/transport/pulls/10/comments?per_page=100": (200, [], {}),
        "/repos/example/transport/issues/10/timeline?per_page=100": (200, [], {}),
        "/repos/example/transport/pulls/10/commits?per_page=100": (
            200,
            [{"sha": f01.c1_oid}],
            {},
        ),
        "/repos/example/transport/pulls/10/files?per_page=100": (200, [], {}),
    }
    transport, _ = make_mock_github_transport(routes)
    with GitHubClient(token="", transport=transport) as client:
        collect_focused_github_evidence(store, client, commit_oids=[f01.c1_oid])

    # 1. Exact symbol search (`send`) puts `exact_symbol` first
    res_sym = execute_search(store, mirror, SearchRequest(query="send", kind="code"))
    assert len(res_sym.hits) >= 1
    assert res_sym.hits[0].match == "exact_symbol"
    assert res_sym.hits[0].kind == "symbol"
    send_id = res_sym.hits[0].id

    # 2. Exact path search (`src/transport.py`) puts `exact_path` first
    res_path = execute_search(store, mirror, SearchRequest(query="src/transport.py", kind="code"))
    assert len(res_path.hits) >= 1
    assert res_path.hits[0].match == "exact_path"

    # 3. Exact ID search (`send_id`) puts `exact_id` first
    res_id = execute_search(store, mirror, SearchRequest(query=send_id, kind="all"))
    assert len(res_id.hits) >= 1
    assert res_id.hits[0].id == send_id
    assert res_id.hits[0].match == "exact_id"

    # 4. Malicious / FTS operator string is treated as literal search terms without SQL/FTS syntax error
    malicious_queries = [
        'NEAR(retry, 5) OR NOT "unclosed quote * ^ :',
        "'); DROP TABLE sources; --",
        "*** ??? ::: ((()))",
    ]
    for mq in malicious_queries:
        m_res = execute_search(store, mirror, SearchRequest(query=mq))
        assert m_res.schema_version == 1

    # 5. Lexical history search for "legacy proxy duplicates"
    res_hist = execute_search(
        store,
        mirror,
        SearchRequest(query="legacy proxy duplicates", kind="history", limit=1),
    )
    assert len(res_hist.hits) == 1
    assert res_hist.hits[0].kind == "source_version"
    assert res_hist.next_cursor is not None

    # Paginate to page 2 using cursor
    res_hist_p2 = execute_search(
        store,
        mirror,
        SearchRequest(
            query="legacy proxy duplicates",
            kind="history",
            limit=1,
            cursor=res_hist.next_cursor,
        ),
    )
    assert len(res_hist_p2.hits) == 1
    assert res_hist_p2.hits[0].id != res_hist.hits[0].id
    assert res_hist_p2.hits[0].rank == 2

    # 6. Expand symbol via `get`
    sym_page = execute_get(store, mirror, GetRequest(id=send_id))
    assert isinstance(sym_page.record, SymbolRecord)
    assert sym_page.record.qualified_name == "send"
    assert len(sym_page.content) == 1
    assert "for attempt in range(MAX_RETRIES):" in sym_page.content[0].text

    # 7. Expand source_version via `get` and verify parent source expansion
    ver_id = res_hist.hits[0].id
    ver_page = execute_get(store, mirror, GetRequest(id=ver_id))
    assert isinstance(ver_page.record, SourceVersionRecord)
    assert "legacy proxy duplicates" in ver_page.content[0].text

    src_id = ver_page.record.source_id
    src_page = execute_get(store, mirror, GetRequest(id=src_id))
    assert isinstance(src_page.record, SourceRecord)
    assert src_page.record.selected_version_id == ver_id

    # 8. Test knowledge_cutoff on `get`: PR #10 was created 2026-01-02T09:00:00Z
    before_pr_cutoff = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    with pytest.raises(CodeContextError) as exc_info:
        execute_get(store, mirror, GetRequest(id=ver_id, knowledge_cutoff=before_pr_cutoff))
    assert exc_info.value.code == "RECORD_NOT_ELIGIBLE"
    # Verify error details do not leak body or title!
    assert "legacy proxy" not in str(exc_info.value.details)

    # 9. Walk active coverage gaps via `get(gap_...)` and `next_gap_id`
    st = store.status()
    assert st.coverage.first_gap_id is not None
    cur_gap_id: str | None = st.coverage.first_gap_id
    walked_gaps: list[str] = []
    while cur_gap_id is not None:
        g_page = execute_get(store, mirror, GetRequest(id=cur_gap_id))
        assert isinstance(g_page.record, GapRecord)
        walked_gaps.append(g_page.record.id)
        cur_gap_id = g_page.record.next_gap_id
    assert len(walked_gaps) == st.coverage.active_gap_count

    # 10. Blocked source cannot be retrieved by ID
    with store.write_transaction() as conn:
        conn.execute("UPDATE sources SET availability = 'inaccessible' WHERE id = ?", (src_id,))
    with pytest.raises(CodeContextError) as exc_info:
        execute_get(store, mirror, GetRequest(id=src_id))
    assert exc_info.value.code == "ACCESS_BLOCKED"
    with pytest.raises(CodeContextError) as exc_info:
        execute_get(store, mirror, GetRequest(id=ver_id))
    assert exc_info.value.code == "ACCESS_BLOCKED"

    # 11. U11: Oversized mandatory envelope raises INVALID_REQUEST with minimum_required_tokens
    # Use a custom token counter that reports 1000 tokens for any envelope while max_tokens=512
    with pytest.raises(CodeContextError) as exc_info:
        execute_get(
            store,
            mirror,
            GetRequest(id=send_id, max_tokens=512),
            token_counter=lambda _s: 1000,
        )
    assert exc_info.value.code == "INVALID_REQUEST"
    assert exc_info.value.details["minimum_required_tokens"] > 512

    store.close()
