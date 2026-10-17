import shutil
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from codecontext import (
    CodeContextError,
    ContextRequest,
    EnrichRequest,
    ExportRequest,
    GetRequest,
    InitRequest,
    Memory,
    PathTarget,
    PurgeRequest,
    SearchRequest,
    SyncRequest,
)
from codecontext.enrichment import compute_snapshot_groups
from codecontext.retrieval import execute_context
from examples.fixture_adapter import DeterministicFixtureAdapter
from tests.fixtures.builders import (
    GitCommitSpec,
    build_f01_fixture,
    commit_fixture_spec,
    init_fixture_repo,
)
from tests.integration.test_t11_sdk_cli_mcp import _build_f01_routes
from tests.unit.test_t07_github import make_mock_github_transport


def test_t17_t19_summaries_groupings_and_cutoff_eligibility(tmp_path: Path) -> None:
    """T17, T19, U07:
    - Deterministic grouping: deepest non-root `pyproject.toml`/`package.json` directory is a `package`,
      remaining top-level dirs are `directory`, root files are `root_files` (`.`)
    - Enrichment publishes group and repository summaries with exact citations and full dependency links
    - `context` includes stored summary in `overview`, and excludes it when `knowledge_cutoff` precedes any summary dependency!
    """
    repo_dir = init_fixture_repo(tmp_path / "pkg_repo")
    commit_fixture_spec(
        repo_dir,
        GitCommitSpec(
            message="Nested package layout",
            files={
                "README.md": b"# Root Readme\n",
                "pyproject.toml": b"[project]\nname = 'root'\n",
                "packages/core/pyproject.toml": b"[project]\nname = 'core'\n",
                "packages/core/sub/package.json": b'{"name": "sub"}\n',
                "packages/core/sub/index.ts": b"export const x = 1;\n",
                "packages/core/mod.py": b"def core_fn():\n    return 1\n",
                "tools/helper.py": b"def help_fn():\n    return 2\n",
            },
        ),
    )
    store_dir = tmp_path / "pkg_store"
    with Memory.initialize(store_dir, InitRequest(repo_path=repo_dir)) as mem:
        mem.sync(SyncRequest(revision="HEAD", offline=True))
        with mem._store.connect_read() as conn:
            groups = compute_snapshot_groups(conn, repository_id=mem._store.config.repository_id)
        by_disp = {(g.scope_kind, g.path_display): [ p.decode() for p in g.files ] for g in groups}
        # Deepest package root wins for `packages/core/sub/*`
        assert ("package", "packages/core/sub") in by_disp
        assert "packages/core/sub/index.ts" in by_disp[("package", "packages/core/sub")]
        assert ("package", "packages/core") in by_disp
        assert "packages/core/mod.py" in by_disp[("package", "packages/core")]
        assert ("directory", "tools") in by_disp
        assert ("root_files", ".") in by_disp
        assert "README.md" in by_disp[("root_files", ".")]

    # Now test summary creation, citation expansion, and cutoff filtering on F01
    f01 = build_f01_fixture(tmp_path / "f01")
    f01_store_dir = tmp_path / "f01_store"
    routes = _build_f01_routes(f01)
    transport, _ = make_mock_github_transport(routes)

    with Memory.initialize(
        f01_store_dir,
        InitRequest(repo_path=f01.repo_dir, github_repository="example/transport"),
    ) as mem:
        mem._github_transport = transport
        mem._github_token = ""
        mem.sync(SyncRequest(revision="HEAD"))
        rep = mem.enrich(
            EnrichRequest(max_input_tokens=100_000, max_output_tokens=20_000, max_calls=15),
            model=DeterministicFixtureAdapter(),
        )
        assert rep.claims_added == 1
        assert rep.summaries_added >= 1
        assert mem.status().summary_count >= 1

        # Context pack without cutoff includes `stored_summary` overview
        pack = mem.context(
            ContextRequest(
                query="Why is send limited to 3 retries?",
                target=PathTarget(kind="path", path="src/transport.py", line=5),
            )
        )
        assert pack.overview is not None
        assert pack.overview.kind == "stored_summary"
        assert pack.overview.summary_id is not None
        sum_id = pack.overview.summary_id

        # Expand summary via `get`
        sum_page = mem.get(GetRequest(id=sum_id))
        assert sum_page.record.kind == "summary"
        assert len(sum_page.record.statements) >= 1

        # With a cutoff before PR #10 (2026-01-01T12:00:00Z):
        # - `get(sum_id)` raises `RECORD_NOT_ELIGIBLE` because the summary depends on PR #10 (created 2026-01-02)
        # - `context` falls back to `structural` overview and excludes the post-cutoff claim!
        early_cutoff = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        with pytest.raises(CodeContextError) as exc_info:
            mem.get(GetRequest(id=sum_id, knowledge_cutoff=early_cutoff))
        assert exc_info.value.code == "RECORD_NOT_ELIGIBLE"

        pack_cutoff = mem.context(
            ContextRequest(
                query="Why is send limited to 3 retries?",
                target=PathTarget(kind="path", path="src/transport.py", line=5),
                knowledge_cutoff=early_cutoff,
            )
        )
        assert pack_cutoff.overview is not None
        assert pack_cutoff.overview.kind == "structural"
        assert pack_cutoff.decisions == []


def test_t18_t20_f11_i10_i12_purge_race_and_portable_export_sanitization(tmp_path: Path) -> None:
    """T18, T20, F11, I10, I12, I13:
    - Export creates a coherent offline archive, scrubs machine binding paths and runtime credentials
      from `store.json`, `memory.sqlite3`, and `repository.git/config`, and works after deleting source repo
    - Source purge removes the source's versions, evidence, claims, dependent summaries, and FTS documents
      in a foreign-key-safe order; direct `get` on previously known version/evidence/claim/summary IDs returns `RECORD_NOT_FOUND`
    - Purge race barrier: a read paused before its final `access_epoch` authorization check fails with
      `CURSOR_STALE` when a purge commits before authorization; a prior export remains outside purge control.
    """
    f01 = build_f01_fixture(tmp_path / "f01_purge")
    store_dir = tmp_path / "purge_store"
    routes = _build_f01_routes(f01)
    transport, _ = make_mock_github_transport(
        routes, expected_user={"id": 7001, "login": "secret_principal"}
    )

    secret_token = "ghp_SUPER_SECRET_RUNTIME_CREDENTIAL_999"

    with Memory.initialize(
        store_dir,
        InitRequest(repo_path=f01.repo_dir, github_repository="example/transport"),
    ) as mem:
        mem._github_transport = transport
        mem._github_token = secret_token
        mem.sync(SyncRequest(revision="HEAD"))
        mem.enrich(
            EnrichRequest(max_input_tokens=100_000, max_output_tokens=20_000, max_calls=15),
            model=DeterministicFixtureAdapter(),
        )

        # Save IDs of PR #10 source, version, claim, evidence, and summary before purge
        pack_before = mem.context(
            ContextRequest(
                query="Why is send limited to 3 retries?",
                target=PathTarget(kind="path", path="src/transport.py", line=5),
            )
        )
        assert len(pack_before.decisions) == 1
        claim_id = pack_before.decisions[0].id
        ev_id = pack_before.decisions[0].evidence_refs[0].id
        ver_id = pack_before.decisions[0].evidence_refs[0].anchor.source_version_id
        assert ver_id is not None
        ver_page = mem.get(GetRequest(id=ver_id))
        assert ver_page.record.kind == "source_version"
        pr_src_id = ver_page.record.source_id
        assert pack_before.overview is not None and pack_before.overview.summary_id is not None
        sum_id = pack_before.overview.summary_id

        # 1. Export store BEFORE purge (I12)
        export_dir = tmp_path / "prior_export"
        exp_rep = mem.export(ExportRequest(destination=export_dir, include_git_objects=True))
        assert exp_rep.include_git_objects is True
        assert exp_rep.file_count >= 4

        # Existing destination cannot be overwritten
        with pytest.raises(CodeContextError) as exc_info:
            mem.export(ExportRequest(destination=export_dir))
        assert exc_info.value.code == "INVALID_REQUEST"

        # Scan all exported files in `export_dir` for runtime credential and original local binding path: neither appears!
        repo_path_str = str(f01.repo_dir.resolve())
        for root_p, _dirs, files in (export_dir).walk() if hasattr(export_dir, "walk") else []:
            for fn in files:
                raw_b = (Path(root_p) / fn).read_bytes()
                assert secret_token.encode() not in raw_b
                assert b"secret_principal" not in raw_b
                assert repo_path_str.encode() not in raw_b
        # Also explicitly check memory.sqlite3, store.json, and repository.git/config bytes
        for check_rel in ("memory.sqlite3", "store.json", "repository.git/config"):
            raw_b = (export_dir / check_rel).read_bytes()
            assert secret_token.encode() not in raw_b
            assert b"secret_principal" not in raw_b
            assert repo_path_str.encode() not in raw_b

        # 2. Test I10 purge race barrier:
        # Start a context read on `mem`, pause at `pre_authorize_hook` right before final `access_epoch` check,
        # execute source purge in parallel, then resume read and assert it is rejected!
        read_paused = threading.Event()
        purge_done = threading.Event()
        read_errors: list[CodeContextError] = []

        def pause_before_auth() -> None:
            read_paused.set()
            assert purge_done.wait(timeout=10.0)

        def reader_thread() -> None:
            try:
                execute_context(
                    mem._store,
                    mem._mirror,
                    ContextRequest(
                        query="Why is send limited to 3 retries?",
                        target=PathTarget(kind="path", path="src/transport.py", line=5),
                    ),
                    pre_authorize_hook=pause_before_auth,
                )
            except CodeContextError as exc:
                read_errors.append(exc)

        t = threading.Thread(target=reader_thread)
        t.start()
        assert read_paused.wait(timeout=5.0)

        # Execute purge of PR #10 source while reader is paused before authorization check
        purge_rep = mem.purge(PurgeRequest(source_id=pr_src_id))
        assert purge_rep.mode == "source"
        assert purge_rep.removed_source_versions >= 1
        assert purge_rep.removed_claims >= 1
        assert purge_rep.removed_summaries >= 1

        purge_done.set()
        t.join(timeout=5.0)

        # The in-flight read was invalidated at its authorization point!
        assert len(read_errors) == 1
        assert read_errors[0].code == "CURSOR_STALE"

        # 3. Verify all purged IDs (`pr_src_id`, `ver_id`, `ev_id`, `claim_id`, `sum_id`) now return `RECORD_NOT_FOUND`
        for purged_id in (pr_src_id, ver_id, ev_id, claim_id, sum_id):
            with pytest.raises(CodeContextError) as exc_info:
                mem.get(GetRequest(id=purged_id))
            assert exc_info.value.code == "RECORD_NOT_FOUND"

        # FTS search no longer returns any purged text
        s_after = mem.search(SearchRequest(query="legacy proxy duplicates"))
        assert s_after.hits == []

    # 4. Delete the developer's source repository completely and verify the prior export still works offline
    shutil.rmtree(f01.repo_dir)
    with Memory.open(export_dir, read_only=True) as exp_mem:
        exp_pack = exp_mem.context(
            ContextRequest(
                query="Why is send limited to 3 retries?",
                target=PathTarget(kind="path", path="src/transport.py", line=5),
            )
        )
        assert exp_pack.rationale_status == "supported"
        assert len(exp_pack.decisions) == 1
        # Expand code citation offline from exported bare git store
        code_ev_id = exp_pack.code[0].evidence_id
        code_page = exp_mem.get(GetRequest(id=code_ev_id))
        assert "MAX_RETRIES" in code_page.content[0].text
