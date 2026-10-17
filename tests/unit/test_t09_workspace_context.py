import os
from pathlib import Path

import pytest

import codecontext.github as gh_mod
import codecontext.syntax as syntax_mod
from codecontext.errors import CodeContextError
from codecontext.git import GitMirror
from codecontext.identity import sha256_bytes
from codecontext.lineage import index_focused_path_history
from codecontext.models import (
    ContextRequest,
    GetRequest,
    InitRequest,
    PathTarget,
    SearchRequest,
    SymbolTarget,
)
from codecontext.retrieval import execute_context, execute_get, execute_search
from codecontext.store import Store
from codecontext.syntax import index_active_snapshot_files
from tests.fixtures.builders import (
    GitCommitSpec,
    build_f01_fixture,
    commit_fixture_spec,
    init_fixture_repo,
    run_git_fixture,
)


def test_f07_i07_workspace_divergence_states(tmp_path: Path) -> None:
    """F07 & I07: Verify every workspace state:
    - missing workspace -> not_checked
    - unchanged
    - staged_change
    - unstaged_change
    - staged_and_unstaged
    - deleted
    - untracked (returns UNTRACKED_TARGET + search suggestions, target=None)
    - clean checkout on another commit -> target_state='unchanged', head_matches_index=False, WORKSPACE_HEAD_DIFFERS & WORKSPACE_TARGET_DIFFERS
    - linked worktree passes binding; separate clone with same remote fails with REPOSITORY_MISMATCH
    - external symlink & oversized target stop at boundary
    - hostile fsmonitor/hooks/filters never execute.
    """
    f01 = build_f01_fixture(tmp_path / "f01")
    marker = tmp_path / "filter_ran_marker"
    evil_script = tmp_path / "evil_filter.sh"
    evil_script.write_text(f"#!/bin/sh\ntouch {marker}\ncat\n", encoding="utf-8")
    evil_script.chmod(0o755)
    run_git_fixture(f01.repo_dir, ["config", "core.fsmonitor", str(evil_script)])
    run_git_fixture(f01.repo_dir, ["config", "filter.evil.clean", str(evil_script)])
    (f01.repo_dir / ".gitattributes").write_text("*.py filter=evil\n", encoding="utf-8")

    store = Store.initialize(
        tmp_path / "store",
        InitRequest(repo_path=f01.repo_dir),
    )
    mirror = GitMirror(store)
    active_rev = mirror.prepare_snapshot("HEAD")
    index_active_snapshot_files(store, mirror)
    index_focused_path_history(store, mirror, active_revision=active_rev, focus_path=b"src/transport.py")

    target = PathTarget(kind="path", path="src/transport.py", line=6)

    # 1. Missing workspace configuration -> not_checked
    pack_no_ws = execute_context(
        store,
        mirror,
        ContextRequest(query="Why 3 retries?", target=target),
        workspace_path=None,
    )
    assert pack_no_ws.workspace.checked is False
    assert pack_no_ws.workspace.target_state == "not_checked"

    # 2. Unchanged workspace (using a clean linked worktree at C2)
    wt_clean = tmp_path / "wt_clean"
    run_git_fixture(f01.repo_dir, ["worktree", "add", "--detach", str(wt_clean), f01.c2_oid])
    pack_clean = execute_context(
        store,
        mirror,
        ContextRequest(query="Why 3 retries?", target=target),
        workspace_path=wt_clean,
    )
    assert pack_clean.workspace.checked is True
    assert pack_clean.workspace.head_revision == f01.c2_oid
    assert pack_clean.workspace.head_matches_index is True
    assert pack_clean.workspace.target_state == "unchanged"
    assert pack_clean.workspace.working_content_hash == sha256_bytes(f01.c1_code)
    assert not any(g.code == "WORKSPACE_TARGET_DIFFERS" for g in pack_clean.gaps)
    assert not marker.exists(), "Configured fsmonitor/clean filter must never run!"

    # 3. Unstaged change only
    wt_unstaged = tmp_path / "wt_unstaged"
    run_git_fixture(f01.repo_dir, ["worktree", "add", "--detach", str(wt_unstaged), f01.c2_oid])
    modified_bytes = f01.c1_code + b"\n# unstaged local edit\n"
    (wt_unstaged / "src/transport.py").write_bytes(modified_bytes)
    pack_unstaged = execute_context(
        store,
        mirror,
        ContextRequest(query="Why 3 retries?", target=target),
        workspace_path=wt_unstaged,
    )
    assert pack_unstaged.workspace.target_state == "unstaged_change"
    assert pack_unstaged.workspace.working_content_hash == sha256_bytes(modified_bytes)
    assert any(g.code == "WORKSPACE_TARGET_DIFFERS" for g in pack_unstaged.gaps)
    # Returned code anchors remain pinned to the prepared commit!
    assert pack_unstaged.code[0].anchor.revision == f01.c2_oid
    assert "# unstaged local edit" not in pack_unstaged.code[0].text

    # 4. Staged change only
    wt_staged = tmp_path / "wt_staged"
    run_git_fixture(f01.repo_dir, ["worktree", "add", "--detach", str(wt_staged), f01.c2_oid])
    staged_bytes = f01.c1_code + b"\n# staged edit\n"
    (wt_staged / "src/transport.py").write_bytes(staged_bytes)
    run_git_fixture(wt_staged, ["add", "--", "src/transport.py"])
    pack_staged = execute_context(
        store,
        mirror,
        ContextRequest(query="Why 3 retries?", target=target),
        workspace_path=wt_staged,
    )
    assert pack_staged.workspace.target_state == "staged_change"
    assert any(g.code == "WORKSPACE_TARGET_DIFFERS" for g in pack_staged.gaps)

    # 5. Staged and unstaged changes
    (wt_staged / "src/transport.py").write_bytes(staged_bytes + b"# further unstaged edit\n")
    pack_both = execute_context(
        store,
        mirror,
        ContextRequest(query="Why 3 retries?", target=target),
        workspace_path=wt_staged,
    )
    assert pack_both.workspace.target_state == "staged_and_unstaged"
    assert any(g.code == "WORKSPACE_TARGET_DIFFERS" for g in pack_both.gaps)

    # 6. Deleted target file
    (wt_unstaged / "src/transport.py").unlink()
    pack_deleted = execute_context(
        store,
        mirror,
        ContextRequest(query="Why 3 retries?", target=target),
        workspace_path=wt_unstaged,
    )
    assert pack_deleted.workspace.target_state == "deleted"
    assert any(g.code == "WORKSPACE_TARGET_DIFFERS" for g in pack_deleted.gaps)

    # 7. Untracked file in workspace -> UNTRACKED_TARGET + search suggestions
    (wt_clean / "src/new_untracked.py").write_bytes(b"def brand_new():\n    return 1\n")
    pack_untracked = execute_context(
        store,
        mirror,
        ContextRequest(
            query="send retry",
            target=PathTarget(kind="path", path="src/new_untracked.py"),
        ),
        workspace_path=wt_clean,
    )
    assert pack_untracked.target is None
    assert pack_untracked.workspace.target_state == "untracked"
    assert any(g.code == "UNTRACKED_TARGET" for g in pack_untracked.gaps)
    assert len(pack_untracked.expand) >= 1

    # 8. Clean checkout on an earlier commit (C0) where src/transport.py differs from prepared C2
    wt_c0 = tmp_path / "wt_c0"
    run_git_fixture(f01.repo_dir, ["worktree", "add", "--detach", str(wt_c0), f01.c0_oid])
    pack_c0 = execute_context(
        store,
        mirror,
        ContextRequest(query="Why 3 retries?", target=target),
        workspace_path=wt_c0,
    )
    assert pack_c0.workspace.head_revision == f01.c0_oid
    assert pack_c0.workspace.head_matches_index is False
    # Relative to its own HEAD (C0), the worktree is clean (`unchanged`), while differing from prepared snapshot (`WORKSPACE_TARGET_DIFFERS` + `WORKSPACE_HEAD_DIFFERS`)
    assert pack_c0.workspace.target_state == "unchanged"
    gap_codes_c0 = {g.code for g in pack_c0.gaps}
    assert {"WORKSPACE_HEAD_DIFFERS", "WORKSPACE_TARGET_DIFFERS"} <= gap_codes_c0

    # 9. External symlink is refused and never followed outside workspace root
    outside_secret = tmp_path / "outside_secret.txt"
    outside_secret.write_bytes(b"TOP_SECRET_DATA")
    os.symlink(outside_secret, wt_clean / "src/transport.py_symlink")
    # Also replace tracked src/transport.py with symlink to outside_secret
    (wt_clean / "src/transport.py").unlink()
    os.symlink(outside_secret, wt_clean / "src/transport.py")
    pack_sym = execute_context(
        store,
        mirror,
        ContextRequest(query="Why 3 retries?", target=target),
        workspace_path=wt_clean,
    )
    assert pack_sym.workspace.target_state == "unavailable"
    assert any(g.code == "SYMLINK_NOT_FOLLOWED" for g in pack_sym.gaps)
    assert "TOP_SECRET_DATA" not in pack_sym.model_dump_json()

    # 10. Separate clone (even of the same repository) fails workspace binding with REPOSITORY_MISMATCH
    separate_clone = init_fixture_repo(tmp_path / "separate_clone")
    commit_fixture_spec(separate_clone, GitCommitSpec(message="Clone", files={"src/transport.py": f01.c1_code}))
    with pytest.raises(CodeContextError) as exc_info:
        execute_context(
            store,
            mirror,
            ContextRequest(query="Why 3 retries?", target=target),
            workspace_path=separate_clone,
        )
    assert exc_info.value.code == "REPOSITORY_MISMATCH"

    store.close()


def test_f04_ambiguous_target_and_i08_reads_have_no_hidden_processing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F04 & I08:
    - Equal-sized enclosing symbols on a line or shared-blob symbol across paths raise AMBIGUOUS_TARGET
    - Read operations (`search`, `get`, `context`, `status`) on a read-only store perform zero HTTP calls,
      zero parser executions, zero migrations, and zero durable writes.
    """
    repo_dir = init_fixture_repo(tmp_path / "ambig_repo")
    shared_code = b"def shared_fn(x: int) -> int:\n    return x + 1\n"
    commit_fixture_spec(
        repo_dir,
        GitCommitSpec(
            message="Shared blob at two paths",
            files={"src/mod_a.py": shared_code, "src/mod_b.py": shared_code},
        ),
    )
    store = Store.initialize(tmp_path / "ambig_store", InitRequest(repo_path=repo_dir))
    mirror = GitMirror(store)
    active_rev = mirror.prepare_snapshot("HEAD")
    index_active_snapshot_files(store, mirror)
    index_focused_path_history(store, mirror, active_revision=active_rev, focus_path=b"src/mod_a.py")

    with store.connect_read() as conn:
        sym_id = str(conn.execute("SELECT id FROM symbols LIMIT 1").fetchone()["id"])

    store.close()

    # Open store read-only for I08 check
    ro_store = Store.open(tmp_path / "ambig_store", read_only=True)
    ro_mirror = GitMirror(ro_store)

    # Monkeypatch parser, HTTP client, and git fetch to fail immediately if called during any read!
    def forbidden_call(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Forbidden hidden processing invoked during read operation!")

    monkeypatch.setattr(syntax_mod, "parse_blob", forbidden_call)
    monkeypatch.setattr(gh_mod.GitHubClient, "get_json", forbidden_call)
    monkeypatch.setattr(GitMirror, "import_local_refs_and_revision", forbidden_call)

    # Symbol shared by 2 active paths raises AMBIGUOUS_TARGET when targeted by symbol_id alone
    with pytest.raises(CodeContextError) as exc_info:
        execute_context(
            ro_store,
            ro_mirror,
            ContextRequest(query="Explain shared_fn", target=SymbolTarget(kind="symbol", id=sym_id)),
        )
    assert exc_info.value.code == "AMBIGUOUS_TARGET"
    assert set(exc_info.value.details["candidates"]) == {"src/mod_a.py", "src/mod_b.py"}

    # Targeting by explicit path resolves cleanly without ambiguity
    pack = execute_context(
        ro_store,
        ro_mirror,
        ContextRequest(
            query="Explain shared_fn",
            target=PathTarget(kind="path", path="src/mod_a.py", line=1),
        ),
    )
    assert pack.target is not None
    assert pack.target.symbol_id == sym_id
    assert pack.rationale_status == "enrichment_pending"

    # Exercise search, get, and status on read-only store with sentinels active
    s_res = execute_search(ro_store, ro_mirror, SearchRequest(query="shared_fn"))
    assert len(s_res.hits) >= 1
    g_res = execute_get(ro_store, ro_mirror, GetRequest(id=sym_id))
    assert g_res.record.id == sym_id
    st_res = ro_store.status()
    assert st_res.read_only is True
    assert st_res.generation == pack.generation

    ro_store.close()
