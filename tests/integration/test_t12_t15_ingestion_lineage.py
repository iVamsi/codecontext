from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from codecontext import (
    ContextRequest,
    EnrichRequest,
    GetRequest,
    InitRequest,
    Memory,
    PathTarget,
    SearchRequest,
    SyncRequest,
)
from codecontext.git import GitMirror
from codecontext.lineage import map_original_anchor_to_target
from examples.fixture_adapter import DeterministicFixtureAdapter
from tests.fixtures.builders import (
    GitCommitSpec,
    build_f01_fixture,
    build_f03_fixture,
    commit_fixture_spec,
    init_fixture_repo,
    run_git_fixture,
)
from tests.unit.test_t07_github import make_mock_github_transport


def test_t12_exhaustive_git_backfill_force_push_and_branch_switching(tmp_path: Path) -> None:
    """T12, I01, I05:
    - Enumerates all configured refs (including divergent branches) and indexes all reachable commits
    - Force-push removing an old branch tip retains previously acquired objects via `refs/codecontext/retained/*`
    - Switching active branches reuses already-parsed symbols without duplicating extractions or symbols.
    """
    f03 = build_f03_fixture(tmp_path / "f03")
    store_dir = tmp_path / "store"

    with Memory.initialize(store_dir, InitRequest(repo_path=f03.repo_dir)) as mem:
        rep1 = mem.sync(SyncRequest(revision="main", offline=True))
        assert rep1.state == "complete"
        assert rep1.revision == f03.backdated_oid

        # Verify all commits across both `main` and `divergent` branches were discovered and diffed
        st1 = mem.status()
        assert st1.commit_count == 6
        assert st1.coverage.git.state == "complete"

        with mem._store.connect_read() as conn:
            ext_before = int(conn.execute("SELECT COUNT(*) FROM extractions").fetchone()[0])
            sym_before = int(conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0])

        # Switch active snapshot to `divergent` branch offline -> reuses existing extractions in prepare_snapshot (0 blobs reparsed)!
        rep2 = mem.sync(SyncRequest(revision="divergent", offline=True))
        assert rep2.revision == f03.divergent_oid
        assert rep2.blobs_parsed == 0

        with mem._store.connect_read() as conn:
            ext_after = int(conn.execute("SELECT COUNT(*) FROM extractions").fetchone()[0])
            sym_after = int(conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0])
            assert ext_after == ext_before
            assert sym_after == sym_before

        # Force-push `divergent` in source repo to `root_oid` and run GC in source repo
        run_git_fixture(f03.repo_dir, ["branch", "-f", "divergent", f03.root_oid])
        run_git_fixture(f03.repo_dir, ["gc", "--prune=now"])

        # Re-sync `main`: old `divergent_oid` remains pinned and readable in owned mirror!
        mem.sync(SyncRequest(revision="main", offline=True))
        with mem._mirror.open_batch_session() as batch:
            old_obj = batch.read_object(f03.divergent_oid)
            assert old_obj is not None and old_obj[0] == "commit"


def test_t13_t14_f06_f08_full_github_enumeration_incremental_edits_and_reconciliation(
    tmp_path: Path,
) -> None:
    """T13, T14, F06, F08, I02, I03:
    - Full `/issues?state=all` enumeration normalizes PR-shaped issue records to one `pull_request` source
    - Closed issues, unmerged PRs, and platform timeline links are indexed
    - Repository rename with unchanged numeric ID preserves store identity
    - Comment edited at T3 with unchanged parent timestamp is detected via incremental `/issues/comments` stream,
      creates a second `source_version`, retains old version for `knowledge_cutoff` search/get, and marks dependent claims `stale`
    - Deleted comment is NOT inferred during an incomplete pass, and IS marked `deleted` only after a complete `reconcile=True` pass.
    """
    f01 = build_f01_fixture(tmp_path / "f01")
    store_dir = tmp_path / "store"

    closed_issue = {
        "id": 50001,
        "number": 5,
        "state": "closed",
        "title": "Worker pool exhaustion during proxy brownouts",
        "body": "Workers hang when proxy times out.",
        "user": {"id": 504, "login": "dave"},
        "created_at": "2026-01-01T08:00:00Z",
        "updated_at": "2026-01-01T09:00:00Z",
        "html_url": "https://github.com/example/transport/issues/5",
    }
    pr_in_issues_list = {
        "id": 10010,
        "number": 10,
        "state": "closed",
        "title": f01.pr_payload["title"],
        "body": f01.pr_payload["body"],
        "user": f01.pr_payload["user"],
        "created_at": f01.pr_payload["created_at"],
        "updated_at": f01.pr_payload["updated_at"],
        "pull_request": {"url": "https://api.github.com/repos/example/transport/pulls/10"},
        "html_url": f01.pr_payload["html_url"],
    }
    unmerged_pr_issue = {
        "id": 10011,
        "number": 11,
        "state": "closed",
        "title": "Experimental 10-retry attempt",
        "body": "Try 10 retries instead.",
        "user": {"id": 502, "login": "bob"},
        "created_at": "2026-01-02T11:00:00Z",
        "updated_at": "2026-01-02T12:00:00Z",
        "pull_request": {"url": "https://api.github.com/repos/example/transport/pulls/11"},
        "html_url": "https://github.com/example/transport/pull/11",
    }
    unmerged_pr_detail = {
        "id": 10011,
        "number": 11,
        "state": "closed",
        "title": "Experimental 10-retry attempt",
        "body": "Try 10 retries instead.",
        "user": {"id": 502, "login": "bob"},
        "created_at": "2026-01-02T11:00:00Z",
        "updated_at": "2026-01-02T12:00:00Z",
        "merged_at": None,
        "merge_commit_sha": None,
        "head": {"sha": "f" * 40, "ref": "exp/ten-retries"},
        "base": {"sha": f01.c0_oid, "ref": "main"},
        "html_url": "https://github.com/example/transport/pull/11",
    }

    routes: dict[str, tuple[int, Any, dict[str, str]]] = {
        "/repos/example/transport": (
            200,
            {"id": 9001, "full_name": "example/transport", "private": False},
            {},
        ),
        f"/repos/example/transport/commits/{f01.c2_oid}/pulls": (200, [], {}),
        f"/repos/example/transport/commits/{f01.c1_oid}/pulls": (
            200,
            [{"number": 10, "id": 10010}],
            {},
        ),
        f"/repos/example/transport/commits/{f01.c0_oid}/pulls": (200, [], {}),
        "/repos/example/transport/issues?state=all&sort=created&direction=asc&per_page=100": (
            200,
            [closed_issue, pr_in_issues_list, unmerged_pr_issue],
            {},
        ),
        "/repos/example/transport/issues/5/comments?per_page=100": (200, [], {}),
        "/repos/example/transport/issues/5/timeline?per_page=100": (
            200,
            [
                {
                    "id": 60001,
                    "event": "cross-referenced",
                    "created_at": "2026-01-02T09:05:00Z",
                    "actor": {"id": 501, "login": "alice"},
                    "source": {"issue": {"number": 10, "pull_request": {}}},
                }
            ],
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
        "/repos/example/transport/pulls/11": (200, unmerged_pr_detail, {}),
        "/repos/example/transport/issues/11/comments?per_page=100": (200, [], {}),
        "/repos/example/transport/pulls/11/reviews?per_page=100": (200, [], {}),
        "/repos/example/transport/pulls/11/comments?per_page=100": (200, [], {}),
        "/repos/example/transport/issues/11/timeline?per_page=100": (200, [], {}),
        "/repos/example/transport/pulls/11/commits?per_page=100": (200, [], {}),
        "/repos/example/transport/pulls/11/files?per_page=100": (200, [], {}),
        "/repos/example/transport/issues/comments?sort=updated&direction=asc&per_page=100": (
            200,
            f01.comments_payload,
            {},
        ),
        "/repos/example/transport/pulls/comments?sort=updated&direction=asc&per_page=100": (
            200,
            [],
            {},
        ),
    }

    transport, _ = make_mock_github_transport(routes)
    with Memory.initialize(
        store_dir,
        InitRequest(repo_path=f01.repo_dir, github_repository="example/transport"),
    ) as mem:
        mem._github_transport = transport
        mem._github_token = ""
        rep = mem.sync(SyncRequest(revision="HEAD"))
        assert rep.state == "complete"

        # Verify PR #10 was normalized to ONE `pull_request` source (never double-counted as an `issue`)
        with mem._store.connect_read() as conn:
            pr10_rows = conn.execute(
                "SELECT kind FROM sources WHERE provider_id = '10'"
            ).fetchall()
            assert len(pr10_rows) == 1
            assert pr10_rows[0]["kind"] == "pull_request"

        # Enrich store so PR #10 has a ready claim
        mem.enrich(
            EnrichRequest(max_input_tokens=100_000, max_output_tokens=20_000, max_calls=10),
            model=DeterministicFixtureAdapter(),
        )
        assert mem.status().claim_count == 1

        # Now simulate F06: comment 20002 is edited at T3 (2026-01-05T12:00:00Z) with new text
        # and repository is renamed to `example/transport-v2` with unchanged numeric ID 9001!
        edited_comment_20002 = {
            **f01.comments_payload[1],
            "body": "Rejected unlimited retries: updated rationale mentioning circuit_breaker_threshold.",
            "updated_at": "2026-01-05T12:00:00Z",
        }
        routes["/repos/example/transport"] = (
            200,
            {"id": 9001, "full_name": "example/transport-v2", "private": False},
            {},
        )
        # Match any incremental issues/comments URL with `since=`
        def dynamic_handler(req: Any) -> Any:
            p_q = req.url.raw_path.decode("ascii")
            if p_q.startswith("/repos/example/transport") and "/issues/comments?sort=updated" in p_q:
                from httpx import Response

                return Response(200, json=[edited_comment_20002])
            if p_q.startswith("/repos/example/transport") and "/pulls/comments?sort=updated" in p_q:
                from httpx import Response

                return Response(200, json=[])
            norm_p = p_q.replace("/repos/example/transport-v2", "/repos/example/transport")
            if norm_p in routes:
                st_c, b_d, hdrs = routes[norm_p]
                from httpx import Response

                return Response(st_c, json=b_d, headers=hdrs)
            from httpx import Response

            return Response(404, json={"message": "Not Found"})

        from httpx import MockTransport

        mem._github_transport = MockTransport(dynamic_handler)
        mem.sync(SyncRequest(revision="HEAD"))

        # Verify:
        # 1. Dependent claim on PR #10 became `stale` when comment 20002 was edited!
        assert mem.status().claim_count == 0
        with mem._store.connect_read() as conn:
            stale_claims = conn.execute(
                "SELECT id, state FROM claims WHERE state = 'stale'"
            ).fetchall()
            assert len(stale_claims) == 1
            stale_claim_id = str(stale_claims[0]["id"])

        # Direct `get` on stale claim returns it with `state='stale'` and `DERIVED_ENRICHMENT_PENDING` gap
        stale_page = mem.get(GetRequest(id=stale_claim_id))
        assert stale_page.record.kind == "claim"
        assert stale_page.record.state == "stale"
        assert any(g.code == "DERIVED_ENRICHMENT_PENDING" for g in stale_page.gaps)

        # 2. Cutoff search before T3 (at 2026-01-03T00:00:00Z) does NOT match `circuit_breaker_threshold` (added at T3),
        #    and DOES match the original T1 version ("under proxy brownouts")!
        cutoff_t2 = datetime(2026, 1, 3, 0, 0, 0, tzinfo=UTC)
        res_future_term = mem.search(
            SearchRequest(
                query="circuit_breaker_threshold",
                kind="history",
                knowledge_cutoff=cutoff_t2,
            )
        )
        assert res_future_term.hits == []

        res_old_term = mem.search(
            SearchRequest(
                query="proxy brownouts",
                kind="history",
                knowledge_cutoff=cutoff_t2,
            )
        )
        assert len(res_old_term.hits) >= 1

        # 3. Now simulate comment 20001 deleted upstream on GitHub:
        # An interrupted reconciliation (max_requests=2) must NOT mark comment 20001 deleted!
        routes["/repos/example/transport/issues/10/comments?per_page=100"] = (
            200,
            [edited_comment_20002],  # comment 20001 is gone
            {},
        )
        mem.sync(SyncRequest(revision="HEAD", reconcile=True, max_requests=2))
        with mem._store.connect_read() as conn:
            avail_mid = conn.execute(
                "SELECT availability FROM sources WHERE provider_id = '20001'"
            ).fetchone()["availability"]
            assert avail_mid == "available"

        # Complete the reconciliation pass -> comment 20001 is now marked `deleted`!
        mem.sync(SyncRequest(revision="HEAD", reconcile=True, max_requests=100))
        with mem._store.connect_read() as conn:
            avail_done = conn.execute(
                "SELECT availability FROM sources WHERE provider_id = '20001'"
            ).fetchone()["availability"]
            assert avail_done == "deleted"


def test_t15_f02_f05_review_to_landed_code_and_scoped_corroboration(tmp_path: Path) -> None:
    """T15, F02, F05, I04, U08:
    - F02: PR head R1 has 5 attempts with inline review comment anchored at R1; R2 changes to 3 attempts
      and lands via squash commit with different OID. Original review anchor opens R1 bytes (`5`),
      while target code displays landed `3` attempts with candidate mapping.
    - F05: Corroboration requires an explicit scoped statement matching the target revision (`active_rev`).
      Unrelated ADR or historical rationale alone stays `historical_only`; scoped ADR upgrades to `corroborated`.
    """
    repo_dir = init_fixture_repo(tmp_path / "f02_repo")
    c0_oid = commit_fixture_spec(
        repo_dir,
        GitCommitSpec(
            message="C0 base",
            files={"src/retry.py": b"def send(x: bytes) -> bool:\n    return len(x) > 0\n"},
            authored_at="2026-01-01T10:00:00Z",
            committed_at="2026-01-01T10:00:00Z",
        ),
    )

    # Create PR head R1 on a side branch with 5 attempts
    run_git_fixture(repo_dir, ["checkout", "-b", "pr-branch", c0_oid])
    r1_code = (
        b"def send(x: bytes) -> bool:\n"
        b"    attempts = 5\n"
        b"    return len(x) <= attempts\n"
    )
    r1_oid = commit_fixture_spec(
        repo_dir,
        GitCommitSpec(
            message="R1: try 5 attempts",
            files={"src/retry.py": r1_code},
            authored_at="2026-01-02T10:00:00Z",
            committed_at="2026-01-02T10:00:00Z",
        ),
    )

    # Switch back to main and land R2 (3 attempts) as a squash commit
    run_git_fixture(repo_dir, ["checkout", "main"])
    r2_code = (
        b"def send(x: bytes) -> bool:\n"
        b"    attempts = 3\n"
        b"    return len(x) <= attempts\n"
    )
    squash_oid = commit_fixture_spec(
        repo_dir,
        GitCommitSpec(
            message="Squash merge PR #20: bound retries to 3",
            files={"src/retry.py": r2_code},
            authored_at="2026-01-03T10:00:00Z",
            committed_at="2026-01-03T10:00:00Z",
        ),
    )

    store_dir = tmp_path / "f02_store"
    with Memory.initialize(
        store_dir,
        InitRequest(repo_path=repo_dir, github_repository="example/retry"),
    ) as mem:
        mem.sync(SyncRequest(revision="main", offline=True))
        mirror = GitMirror(mem._store)

        # Verify R1 original anchor (line 2: `    attempts = 5`) maps as `candidate` to `send` at `squash_oid`
        # while R1 bytes remain inspectable in the owned mirror!
        with mem._store.connect_read() as conn:
            map_res = map_original_anchor_to_target(
                conn,
                mirror,
                original_commit_oid=r1_oid,
                original_path=b"src/retry.py",
                original_line_start=2,
                original_line_end=2,
                target_revision=squash_oid,
            )
            assert map_res.mapping == "candidate"
            assert map_res.method == "same_path_lexical_symbol"

        with mirror.open_batch_session() as batch:
            r1_entries = {e.path: e for e in mirror.list_tree_entries(r1_oid)}
            r1_blob = batch.read_object(r1_entries[b"src/retry.py"].oid)
            assert r1_blob is not None and b"attempts = 5" in r1_blob[2]

        # Test F05 scoped corroboration: add an ADR comment explicitly corroborating `squash_oid`
        adr_comment = {
            "id": 99001,
            "body": f"ADR-001: The 3-attempt retry limit remains required at revision {squash_oid}.",
            "user": {"id": 501, "login": "alice"},
            "created_at": "2026-01-04T10:00:00Z",
            "updated_at": "2026-01-04T10:00:00Z",
            "html_url": "https://github.com/example/retry/pull/20#issuecomment-99001",
        }
        pr20_payload = {
            "id": 20020,
            "number": 20,
            "state": "closed",
            "title": "Bound retries to 3",
            "body": "Limit retries to three because each retry holds a worker and the legacy proxy duplicates requests after its timeout.",
            "user": {"id": 501, "login": "alice"},
            "created_at": "2026-01-02T09:00:00Z",
            "updated_at": "2026-01-04T10:00:00Z",
            "merged_at": "2026-01-03T10:00:00Z",
            "merge_commit_sha": squash_oid,
            "head": {"sha": r1_oid, "ref": "pr-branch"},
            "base": {"sha": c0_oid, "ref": "main"},
            "html_url": "https://github.com/example/retry/pull/20",
        }
        routes = {
            "/repos/example/retry": (
                200,
                {"id": 9002, "full_name": "example/retry", "private": False},
                {},
            ),
            f"/repos/example/retry/commits/{squash_oid}/pulls": (
                200,
                [{"number": 20, "id": 20020}],
                {},
            ),
            f"/repos/example/retry/commits/{c0_oid}/pulls": (200, [], {}),
            "/repos/example/retry/issues?state=all&sort=created&direction=asc&per_page=100": (
                200,
                [],
                {},
            ),
            "/repos/example/retry/pulls/20": (200, pr20_payload, {}),
            "/repos/example/retry/issues/20/comments?per_page=100": (
                200,
                [adr_comment],
                {},
            ),
            "/repos/example/retry/pulls/20/reviews?per_page=100": (200, [], {}),
            "/repos/example/retry/pulls/20/comments?per_page=100": (200, [], {}),
            "/repos/example/retry/issues/20/timeline?per_page=100": (200, [], {}),
            "/repos/example/retry/pulls/20/commits?per_page=100": (
                200,
                [{"sha": r1_oid}],
                {},
            ),
            "/repos/example/retry/pulls/20/files?per_page=100": (200, [], {}),
            "/repos/example/retry/issues/comments?sort=updated&direction=asc&per_page=100": (
                200,
                [],
                {},
            ),
            "/repos/example/retry/pulls/comments?sort=updated&direction=asc&per_page=100": (
                200,
                [],
                {},
            ),
        }
        transport, _ = make_mock_github_transport(routes)
        mem._github_transport = transport
        mem._github_token = ""
        mem.sync(SyncRequest(revision="main", offline=False))
        mem.enrich(
            EnrichRequest(max_input_tokens=100_000, max_output_tokens=20_000, max_calls=10),
            model=DeterministicFixtureAdapter(),
        )

        # At `squash_oid`, the explicit ADR corroboration upgrades applicability to `corroborated`!
        pack_corr = mem.context(
            ContextRequest(
                query="Does the 3-attempt retry limit still apply?",
                target=PathTarget(kind="path", path="src/retry.py", line=1),
            )
        )
        assert pack_corr.rationale_status == "supported"
        assert pack_corr.decisions[0].applicability == "corroborated"
        assert len(pack_corr.decisions[0].applicability_evidence_refs) == 1

        # Now add a new commit C3 (changing README) so active_revision != squash_oid:
        # corroboration does NOT automatically inherit forward to future commits!
        commit_fixture_spec(
            repo_dir,
            GitCommitSpec(message="C3 unrelated doc change", files={"README.md": b"# Retry\n"}),
        )
        mem.sync(SyncRequest(revision="main", offline=True))
        pack_later = mem.context(
            ContextRequest(
                query="Does the 3-attempt retry limit still apply?",
                target=PathTarget(kind="path", path="src/retry.py", line=1),
            )
        )
        assert pack_later.decisions[0].applicability == "historical_only"
