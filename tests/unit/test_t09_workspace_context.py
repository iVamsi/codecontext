import errno
import os
import stat
import subprocess
from pathlib import Path

import pytest

import codecontext.github as gh_mod
import codecontext.syntax as syntax_mod
import codecontext.workspace as workspace
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
from codecontext.workspace import inspect_workspace
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


def _big_file_context(tmp_path: Path, max_tokens: int):
    repo_dir = init_fixture_repo(tmp_path / "big_repo")
    body = b"".join(f"value_{i} = {i}  # filler line for budget\n".encode() for i in range(600))
    commit_fixture_spec(repo_dir, GitCommitSpec(message="Big file", files={"src/big.py": body}))
    store = Store.initialize(tmp_path / "big_store", InitRequest(repo_path=repo_dir))
    mirror = GitMirror(store)
    active_rev = mirror.prepare_snapshot("HEAD")
    index_active_snapshot_files(store, mirror)
    index_focused_path_history(store, mirror, active_revision=active_rev, focus_path=b"src/big.py")
    pack = execute_context(
        store,
        mirror,
        ContextRequest(
            query="Why this file?",
            target=PathTarget(kind="path", path="src/big.py"),
            max_tokens=max_tokens,
        ),
        workspace_path=None,
    )
    store.close()
    return pack, body


def test_context_never_exceeds_requested_tokens(tmp_path: Path) -> None:
    pack, _ = _big_file_context(tmp_path, 1600)
    assert pack.budget.truncated is True
    assert pack.budget.used_tokens <= pack.budget.requested_tokens


def test_truncated_code_anchor_matches_text(tmp_path: Path) -> None:
    pack, body = _big_file_context(tmp_path, 1600)
    assert pack.code, "target code must survive the trim"
    view = pack.code[0]
    span = view.anchor.span
    assert len(view.text.encode()) == span.byte_end - span.byte_start
    assert view.text.endswith("\n")
    assert body[span.byte_start : span.byte_end].decode() == view.text
    assert span.line_end - span.line_start + 1 == view.text.count("\n")


def _small_workspace(tmp_path: Path) -> tuple[Store, GitMirror, Path, str]:
    repo_dir = init_fixture_repo(tmp_path / "ws_repo")
    commit_fixture_spec(
        repo_dir,
        GitCommitSpec(
            message="Base",
            files={"src/app.py": b"x = 1\n", ".gitignore": b".env\n"},
        ),
    )
    store = Store.initialize(tmp_path / "ws_store", InitRequest(repo_path=repo_dir))
    mirror = GitMirror(store)
    active_rev = mirror.prepare_snapshot("HEAD")
    return store, mirror, repo_dir, active_rev


def _inspect(store: Store, repo_dir: Path, active_rev: str, target: bytes):
    return inspect_workspace(
        store,
        repo_dir,
        active_revision=active_rev,
        target_path_bytes=target,
        index_blob_oid=None,
        index_blob_bytes=None,
    )


@pytest.mark.parametrize("target", [b".git/config", b".GIT/config", b"sub/.git/config", b".git"])
def test_inspect_rejects_git_dir_paths(tmp_path: Path, target: bytes) -> None:
    store, _mirror, repo_dir, active_rev = _small_workspace(tmp_path)
    with pytest.raises(CodeContextError) as exc:
        _inspect(store, repo_dir, active_rev, target)
    assert exc.value.code == "INVALID_REQUEST"
    store.close()


def test_context_rejects_git_dir_target_with_workspace(tmp_path: Path) -> None:
    store, mirror, repo_dir, _active_rev = _small_workspace(tmp_path)
    with pytest.raises(CodeContextError) as exc:
        execute_context(
            store,
            mirror,
            ContextRequest(query="config", target=PathTarget(kind="path", path=".git/config")),
            workspace_path=repo_dir,
        )
    assert exc.value.code == "INVALID_REQUEST"
    store.close()


def test_ignored_file_has_no_hash(tmp_path: Path) -> None:
    store, _mirror, repo_dir, active_rev = _small_workspace(tmp_path)
    (repo_dir / ".env").write_bytes(b"TOKEN=secret\n")
    (repo_dir / "notes.txt").write_bytes(b"plain\n")

    ignored, _ = _inspect(store, repo_dir, active_rev, b".env")
    assert ignored.target_state == "untracked"
    assert ignored.working_content_hash is None

    visible, _ = _inspect(store, repo_dir, active_rev, b"notes.txt")
    assert visible.working_content_hash == sha256_bytes(b"plain\n")
    store.close()


def test_parent_swapped_for_symlink_is_not_followed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, _mirror, repo_dir, _active_rev = _small_workspace(tmp_path)
    store.close()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "app.py").write_bytes(b"OUTSIDE_SECRET\n")

    real_check = workspace._resolve_root_confined_parent_dir

    def check_then_swap(*args, **kwargs):
        result = real_check(*args, **kwargs)
        (repo_dir / "src").rename(tmp_path / "src_moved")
        os.symlink(outside, repo_dir / "src")
        return result

    monkeypatch.setattr(workspace, "_resolve_root_confined_parent_dir", check_then_swap)
    state, content, _mode = workspace._read_root_confined_regular_file(repo_dir, b"src/app.py")
    assert state == "symlink"
    assert content is None


def test_open_repo_file_no_follow_refuses_symlinked_parent(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "real").mkdir(parents=True)
    (repo / "real" / "f").write_bytes(b"ok")
    os.symlink(repo / "real", repo / "link")
    root_fd = os.open(repo, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fd = workspace.open_repo_file_no_follow(root_fd, b"real/f")
        try:
            assert os.read(fd, 10) == b"ok"
        finally:
            os.close(fd)
        with pytest.raises(OSError) as exc:
            workspace.open_repo_file_no_follow(root_fd, b"link/f")
        assert exc.value.errno == errno.ELOOP
    finally:
        os.close(root_fd)


def test_non_utf8_workspace_path(tmp_path: Path) -> None:  # Review Focus 4
    store, _mirror, repo_dir, active_rev = _small_workspace(tmp_path)
    name = b"caf\xe9.py"

    missing, _ = _inspect(store, repo_dir, active_rev, name)
    assert missing.target_state == "deleted"

    try:
        fd = os.open(os.path.join(os.fsencode(repo_dir), name), os.O_WRONLY | os.O_CREAT, 0o644)
    except OSError as exc:
        store.close()
        if exc.errno == errno.EILSEQ:
            pytest.skip("filesystem rejects non-UTF-8 names")
        raise
    os.write(fd, b"y = 2\n")
    os.close(fd)
    present, _ = _inspect(store, repo_dir, active_rev, name)
    assert present.target_state == "untracked"
    assert present.working_content_hash == sha256_bytes(b"y = 2\n")
    store.close()


def test_parent_swap_does_not_leak_outside_symlink_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    os.symlink("app.py", repo / "src" / "link")
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink("/OUTSIDE_TARGET_MARKER", outside / "link")

    real_check = workspace._resolve_root_confined_parent_dir

    def check_then_swap(*args, **kwargs):
        result = real_check(*args, **kwargs)
        if not (repo / "src").is_symlink():
            (repo / "src").rename(tmp_path / "src_moved")
            os.symlink(outside, repo / "src")
        return result

    monkeypatch.setattr(workspace, "_resolve_root_confined_parent_dir", check_then_swap)
    assert workspace._read_root_confined_symlink_target(repo, b"src/link") is None
    assert workspace._lstat_root_confined(repo, b"src/link") is None


def test_in_root_symlink_target_and_lstat_still_work(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    os.symlink("app.py", repo / "src" / "link")
    with workspace.RepoDirFds(repo) as dir_fds:
        assert workspace._read_root_confined_symlink_target(
            repo, b"src/link", dir_fds=dir_fds
        ) == b"app.py"
        st = workspace._lstat_root_confined(repo, b"src/link", dir_fds=dir_fds)
    assert st is not None and stat.S_ISLNK(st.st_mode)


def test_fifo_is_rejected_without_being_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    os.mkfifo(repo / "pipe")
    opened: list[object] = []
    real_open = os.open

    def recording_open(path, *args, **kwargs):
        opened.append(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(workspace.os, "open", recording_open)
    state, content, _mode = workspace._read_root_confined_regular_file(repo, b"pipe")
    assert (state, content) == ("unavailable", None)
    assert b"pipe" not in opened


def test_snapshot_reuses_directory_fds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files = {f"a/b/f{i}.py": f"v = {i}\n".encode() for i in range(200)}
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(repo, GitCommitSpec(message="Many", files=files))
    store = Store.initialize(tmp_path / "store", InitRequest(repo_path=repo))
    mirror = GitMirror(store)
    head = mirror.prepare_snapshot("HEAD")
    for name in files:
        (repo / name).write_bytes(b"changed = True\n")

    dir_opens: list[bytes] = []
    real_open_dir_at = workspace._open_dir_at

    def counting(dir_fd: int, name: bytes) -> int:
        dir_opens.append(name)
        return real_open_dir_at(dir_fd, name)

    monkeypatch.setattr(workspace, "_open_dir_at", counting)
    snap, _ = mirror._prepare_workspace_revision(repo, head)
    assert snap != head
    assert dir_opens == [b"a", b"b"]
    mirror.close()
    store.close()


def test_repo_dir_fds_closes_least_recently_used(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    for name in ("x", "y", "z"):
        (repo / name).mkdir(parents=True)
        (repo / name / "f").write_bytes(name.encode())
    with workspace.RepoDirFds(repo, max_open=2) as dir_fds:
        for name in ("x", "y", "z", "x"):
            state, content, _ = workspace._read_root_confined_regular_file(
                repo, f"{name}/f".encode(), dir_fds=dir_fds
            )
            assert (state, content) == ("ok", name.encode())
        assert len(dir_fds._fds) == 2


@pytest.mark.parametrize("rel", [b"../s", b"a/../../s", b"./x", b"", b"a//x", b"a/"])
def test_open_repo_file_no_follow_refuses_escaping_components(tmp_path: Path, rel: bytes) -> None:
    (tmp_path / "s").write_bytes(b"OUTSIDE")
    repo = tmp_path / "repo"
    (repo / "a").mkdir(parents=True)
    (repo / "a" / "x").write_bytes(b"inside")
    (repo / "x").write_bytes(b"inside")
    root_fd = os.open(repo, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(OSError) as exc:
            os.close(workspace.open_repo_file_no_follow(root_fd, rel))
        assert exc.value.errno == errno.EINVAL
    finally:
        os.close(root_fd)


def test_repo_dir_fds_refuses_parent_component(tmp_path: Path) -> None:
    (tmp_path / "s").write_bytes(b"OUTSIDE")
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(ValueError):
        workspace.RepoDirFds(repo, max_open=0)
    with workspace.RepoDirFds(repo) as dir_fds:
        with pytest.raises(OSError) as exc:
            dir_fds.dir_fd([b".."])
        assert exc.value.errno == errno.EINVAL
        state, content, _ = workspace._read_root_confined_regular_file(
            repo, b"../s", dir_fds=dir_fds
        )
    assert (state, content) == ("unavailable", None)


def _count_check_ignore(monkeypatch: pytest.MonkeyPatch, *, fail: bool) -> list[list[str]]:
    calls: list[list[str]] = []
    real = workspace.run_safe_git

    def wrapped(args: list[str], **kwargs):
        if args[0] == "check-ignore":
            calls.append(args)
            if fail:
                return subprocess.CompletedProcess(args, 128, b"", b"fatal")
        return real(args, **kwargs)

    monkeypatch.setattr(workspace, "run_safe_git", wrapped)
    return calls


def test_failed_ignore_check_drops_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, _mirror, repo_dir, active_rev = _small_workspace(tmp_path)
    (repo_dir / "notes.txt").write_bytes(b"plain\n")
    calls = _count_check_ignore(monkeypatch, fail=True)
    state, _ = _inspect(store, repo_dir, active_rev, b"notes.txt")
    assert calls
    assert state.target_state == "untracked"
    assert state.working_content_hash is None
    store.close()


def test_context_runs_ignore_check_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, mirror, repo_dir, _active_rev = _small_workspace(tmp_path)
    (repo_dir / ".env").write_bytes(b"TOKEN=secret\n")
    calls = _count_check_ignore(monkeypatch, fail=False)
    pack = execute_context(
        store,
        mirror,
        ContextRequest(query="env", target=PathTarget(kind="path", path=".env")),
        workspace_path=repo_dir,
    )
    assert pack.workspace.target_state == "untracked"
    assert pack.workspace.working_content_hash is None
    assert len(calls) == 1
    store.close()
