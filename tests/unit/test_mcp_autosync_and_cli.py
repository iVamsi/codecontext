import asyncio
import json
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from mcp import Client
from mcp.shared.exceptions import MCPDeprecationWarning
from mcp.types import ListRootsResult, Root

from codecontext.cli import JSONArgumentParser, main
from codecontext.errors import CodeContextError
from codecontext.git import run_safe_git
from codecontext.mcp_server import _DynamicMemoryHolder, _submodule_discovery_key, create_mcp_server
from codecontext.memory import Memory
from codecontext.models import (
    EnrichRequest,
    InitRequest,
    ModelIdentity,
    ModelRequest,
    ModelResponse,
    SyncRequest,
)
from examples.fixture_adapter import DeterministicFixtureAdapter

SECRET = "glpat-test-do-not-leak-0123456789"


def _setup_git_repo(repo_dir: Path, commits: int = 1) -> str:
    repo_dir.mkdir(parents=True, exist_ok=True)
    run_safe_git(["init", "-b", "main"], cwd=repo_dir)
    run_safe_git(["config", "user.name", "Tester"], cwd=repo_dir)
    run_safe_git(["config", "user.email", "tester@example.com"], cwd=repo_dir)
    run_safe_git(["config", "commit.gpgsign", "false"], cwd=repo_dir)
    run_safe_git(["remote", "add", "origin", "https://github.com/example/repo.git"], cwd=repo_dir)

    for i in range(commits):
        name = "app.py" if i == 0 else f"mod_{i}.py"
        (repo_dir / name).write_text(f"def hello_{i}():\n    return 'world {i}'\n", encoding="utf-8")
        run_safe_git(["add", name], cwd=repo_dir)
        run_safe_git(["commit", "-m", f"Add {name} because callers need greeting {i}"], cwd=repo_dir)
    return run_safe_git(["rev-parse", "HEAD"], cwd=repo_dir).stdout.decode().strip()


def _run_cli(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, dict[str, Any]]:
    capsys.readouterr()
    try:
        code = main(argv)
    except SystemExit as exc:
        code = int(exc.code or 0)
    out = capsys.readouterr().out.strip().splitlines()
    return code, json.loads(out[-1]) if out else {}


def _initialized_store(tmp_path: Path, commits: int = 1) -> tuple[Path, Path]:
    repo_dir = tmp_path / "repo"
    _setup_git_repo(repo_dir, commits)
    store_dir = tmp_path / "store"
    with Memory.initialize(store_dir, InitRequest(repo_path=repo_dir)) as mem:
        mem.sync(SyncRequest(revision="HEAD", offline=True))
    return repo_dir, store_dir


_ENRICH_BUDGET = ["--max-input-tokens", "100000", "--max-output-tokens", "20000", "--max-calls", "20"]


def test_mcp_server_tool_registration(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    _setup_git_repo(repo_dir)
    store_dir = tmp_path / "store"

    server, holder = create_mcp_server(store_path=store_dir, workspace_path=repo_dir, auto_init=True)
    tools = {t.name for t in server._tool_manager.list_tools()}
    assert {"search", "get", "context", "list_repositories", "search_graph", "get_architecture", "detect_changes"} <= tools
    assert "sync" not in tools
    assert "enrich" not in tools
    holder.close()

    server2, holder2 = create_mcp_server(
        store_path=store_dir,
        workspace_path=repo_dir,
        include_graph_tools=False,
        auto_sync=True,
        allow_sync=True,
        allow_enrich=True,
        model_adapter=DeterministicFixtureAdapter(),
    )
    tools2 = {t.name for t in server2._tool_manager.list_tools()}
    assert {"search", "sync", "enrich"} <= tools2
    assert "search_graph" not in tools2
    assert "get_architecture" not in tools2
    holder2.close()


def test_mcp_allow_enrich_requires_model(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    _setup_git_repo(repo_dir)
    with pytest.raises(CodeContextError) as exc:
        create_mcp_server(store_path=tmp_path / "store", workspace_path=repo_dir, allow_enrich=True)
    assert exc.value.code == "INVALID_REQUEST"
    assert "--provider" in exc.value.message


def test_auto_sync_runs_once_per_window_and_prepares_new_head(tmp_path: Path) -> None:
    repo_dir, store_dir = _initialized_store(tmp_path)
    (repo_dir / "extra.py").write_text("def extra():\n    return 1\n", encoding="utf-8")
    run_safe_git(["add", "extra.py"], cwd=repo_dir)
    run_safe_git(["commit", "-m", "Add extra"], cwd=repo_dir)
    new_head = run_safe_git(["rev-parse", "HEAD"], cwd=repo_dir).stdout.decode().strip()

    mem = Memory.open(store_dir, read_only=True)
    holder = _DynamicMemoryHolder(mem, auto_sync=True)

    async def trigger_twice() -> tuple[bool, bool]:
        first = holder._maybe_trigger_auto_sync(repo_dir, mem)
        second = holder._maybe_trigger_auto_sync(repo_dir, mem)
        assert first is not None
        await first
        return first is not None, second is not None

    try:
        assert asyncio.run(trigger_twice()) == (True, False)
        assert mem.status().revision == new_head
    finally:
        holder.close()


def test_submodule_cache_refreshes_when_discovery_files_change(tmp_path: Path) -> None:
    repo_dir, store_dir = _initialized_store(tmp_path)
    holder = _DynamicMemoryHolder(Memory.open(store_dir, read_only=True))
    root = repo_dir.resolve()
    stale_sub = root / "vendored"
    stale_sub.mkdir()
    run_safe_git(["init", "-b", "main"], cwd=stale_sub)
    try:
        holder._submodule_cache[root] = (_submodule_discovery_key(root), [stale_sub])
        assert stale_sub in asyncio.run(holder.repositories(None))

        (root / ".gitmodules").write_text("", encoding="utf-8")
        assert stale_sub not in asyncio.run(holder.repositories(None))
        assert holder._submodule_cache[root] == (_submodule_discovery_key(root), [])
    finally:
        holder.close()


def test_json_argument_parser_error_format(capsys: pytest.CaptureFixture[str]) -> None:
    parser = JSONArgumentParser(prog="codecontext")
    parser.add_argument("--required-flag", required=True)

    with pytest.raises(SystemExit) as exc_info:
        parser.error("the following arguments are required: --required-flag")

    assert exc_info.value.code == 2
    data = json.loads(capsys.readouterr().out.strip())
    assert data["error"]["code"] == "INVALID_REQUEST"
    assert "required" in data["error"]["message"]
    assert data["error"]["retryable"] is False


def test_cli_subcommands_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo_dir = tmp_path / "repo"
    _setup_git_repo(repo_dir)
    store_dir = tmp_path / "store"
    store = ["--store", str(store_dir)]

    code, init_out = _run_cli(capsys, [*store, "init", "--repo", str(repo_dir), "--github", "testorg/testrepo"])
    assert code == 0
    assert len(init_out["capabilities"]["syntax_languages"]) > 0

    code, sync_out = _run_cli(capsys, [*store, "sync", "--offline", "--include-untracked"])
    assert code == 0
    assert "revision" in sync_out

    code, status_out = _run_cli(capsys, [*store, "status"])
    assert code == 0
    assert status_out["github_repository"] == "testorg/testrepo"

    code, arch_out = _run_cli(capsys, [*store, "get-architecture"])
    assert code == 0
    assert "modules" in arch_out or "clusters" in arch_out or "summary" in arch_out

    code, dc_out = _run_cli(capsys, [*store, "detect-changes", "--workspace", str(repo_dir)])
    assert code == 0
    assert "affected_symbols" in dc_out and "changed_files" in dc_out

    scip_file = tmp_path / "empty.scip"
    scip_file.write_bytes(b"")
    code, scip_out = _run_cli(capsys, [*store, "import-scip", str(scip_file)])
    assert code == 0
    assert scip_out["documents_processed"] == 0


def test_cli_init_rejects_token_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo_dir = tmp_path / "repo"
    _setup_git_repo(repo_dir)
    store_dir = tmp_path / "store"
    code, out = _run_cli(
        capsys,
        ["--store", str(store_dir), "init", "--repo", str(repo_dir), "--gitlab", "grp/proj", "--token", SECRET],
    )
    assert code == 2
    assert out["error"]["code"] == "INVALID_REQUEST"
    assert not store_dir.exists()


def test_cli_init_gitlab_keeps_env_token_out_of_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITLAB_TOKEN", SECRET)
    repo_dir = tmp_path / "repo"
    _setup_git_repo(repo_dir)
    store_dir = tmp_path / "store"
    code, _ = _run_cli(capsys, ["--store", str(store_dir), "init", "--repo", str(repo_dir), "--gitlab", "grp/proj"])
    assert code == 0
    for path in store_dir.rglob("*"):
        if path.is_file():
            assert SECRET.encode() not in path.read_bytes(), path


@pytest.mark.parametrize(
    ("extra", "message_part"),
    [
        (["--adapter", "examples.fixture_adapter:create_adapter", "--provider", "ollama", "--model", "m"], "not allowed"),
        ([], "one of the arguments"),
        (["--provider", "openai"], "no default model"),
        (["--adapter", "examples.fixture_adapter:create_adapter", "--model", "m"], "--model only works with --provider"),
        (["--provider", "ollama", "--model", "m", "--concurrency", "0"], "1 to 16"),
        (["--provider", "ollama", "--model", "m", "--concurrency", "17"], "1 to 16"),
        (["--provider", "acme", "--model", "m"], "invalid choice"),
        (["--provider", "ollama", "--model", "m", "--embedding-provider", "ollama"], "used together"),
    ],
)
def test_cli_enrich_model_flag_validation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str], message_part: str
) -> None:
    _, store_dir = _initialized_store(tmp_path)
    code, out = _run_cli(capsys, ["--store", str(store_dir), "enrich", *extra, *_ENRICH_BUDGET])
    assert code == 2
    assert out["error"]["code"] == "INVALID_REQUEST"
    assert message_part in out["error"]["message"]


def test_cli_enrich_provider_reaches_builtin_adapter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Port 9 (discard) is closed on test machines, so the adapter fails fast without network access.
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:9")
    _, store_dir = _initialized_store(tmp_path)
    code, out = _run_cli(
        capsys,
        ["--store", str(store_dir), "enrich", "--provider", "ollama", "--model", "llama-test", *_ENRICH_BUDGET],
    )
    assert code == 0
    assert out["usage"]["calls"] >= 1
    assert out["failures"][0]["code"] == "MODEL_FAILED"


def test_cli_enrich_adapter_with_concurrency(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _, store_dir = _initialized_store(tmp_path, commits=3)
    code, out = _run_cli(
        capsys,
        [
            "--store", str(store_dir), "enrich",
            "--adapter", "examples.fixture_adapter:create_adapter",
            "--concurrency", "3",
            *_ENRICH_BUDGET,
        ],
    )
    assert code == 0
    assert out["failed_items"] == 0


def test_cli_search_accepts_embedding_flags(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _, store_dir = _initialized_store(tmp_path)
    store = ["--store", str(store_dir), "search", "hello_0"]
    code, out = _run_cli(capsys, [*store, "--embedding-model", "embed-test"])
    assert code == 2
    assert "used together" in out["error"]["message"]

    code, out = _run_cli(capsys, [*store, "--embedding-provider", "ollama", "--embedding-model", "embed-test"])
    assert code == 0
    assert "hits" in out


class _ConcurrencyProbe:
    """Real fixture adapter that also records how many calls overlap."""

    def __init__(self) -> None:
        self._inner = DeterministicFixtureAdapter(strict_fixture_only=False)
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0
        self.threads: set[int] = set()

    @property
    def identity(self) -> ModelIdentity:
        return self._inner.identity

    def complete(self, request: ModelRequest) -> ModelResponse:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            self.threads.add(threading.get_ident())
        try:
            time.sleep(0.05)
            return self._inner.complete(request)
        finally:
            with self._lock:
                self._active -= 1


def _enrich_with(tmp_path: Path, concurrency: int) -> tuple[_ConcurrencyProbe, dict[str, Any]]:
    _, store_dir = _initialized_store(tmp_path, commits=4)
    probe = _ConcurrencyProbe()
    with Memory.open(store_dir) as mem:
        rep = mem.enrich(
            EnrichRequest(max_input_tokens=100_000, max_output_tokens=20_000, max_calls=20),
            model=probe,
            concurrency=concurrency,
        )
    return probe, rep.model_dump(mode="json")


def test_enrich_concurrency_overlaps_calls_and_matches_sequential(tmp_path: Path) -> None:
    seq_probe, seq = _enrich_with(tmp_path / "seq", 1)
    par_probe, par = _enrich_with(tmp_path / "par", 4)

    assert seq_probe.max_active == 1
    assert seq_probe.threads == {threading.get_ident()}
    assert par_probe.max_active > 1
    for key in ("state", "completed_items", "failed_items", "claims_added", "pending_items"):
        assert par[key] == seq[key], key
    assert par["usage"]["calls"] == seq["usage"]["calls"]


def test_enrich_rejects_out_of_range_concurrency(tmp_path: Path) -> None:
    _, store_dir = _initialized_store(tmp_path)
    with Memory.open(store_dir) as mem, pytest.raises(CodeContextError) as exc:
        mem.enrich(
            EnrichRequest(max_input_tokens=1000, max_output_tokens=1000, max_calls=1),
            model=DeterministicFixtureAdapter(),
            concurrency=0,
        )
    assert exc.value.code == "INVALID_REQUEST"


def test_cli_join_merge_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo_dir = tmp_path / "repo"
    _setup_git_repo(repo_dir)
    store_dir = tmp_path / "store"
    share_dir = tmp_path / "share"

    with Memory.initialize(store_dir, InitRequest(repo_path=repo_dir)) as mem:
        mem.sync(SyncRequest(revision="HEAD", offline=True))
        mem.publish(share_dir)

    code, merge_out = _run_cli(
        capsys,
        [
            "--store", str(tmp_path / "target_store"),
            "join", "--from", str(share_dir), "--repo", str(repo_dir), "--merge",
        ],
    )
    assert code == 0
    assert "repository_id" in merge_out
    assert merge_out["git_objects_fetched"] is True


def test_auto_sync_runs_for_pinned_store(tmp_path: Path) -> None:
    repo_dir, store_dir = _initialized_store(tmp_path)
    (repo_dir / "pinned.py").write_text("def pinned():\n    return 1\n", encoding="utf-8")
    run_safe_git(["add", "pinned.py"], cwd=repo_dir)
    run_safe_git(["commit", "-m", "Add pinned"], cwd=repo_dir)
    new_head = run_safe_git(["rev-parse", "HEAD"], cwd=repo_dir).stdout.decode().strip()

    mem = Memory.open(store_dir, read_only=True)
    holder = _DynamicMemoryHolder(mem, auto_sync=True)

    async def fetch_and_wait() -> None:
        assert await holder.get_memory() is mem
        assert len(holder._sync_tasks) == 1
        await asyncio.gather(*holder._sync_tasks)

    try:
        asyncio.run(fetch_and_wait())
        assert mem.status().revision == new_head
    finally:
        holder.close()


def test_auto_sync_failure_is_logged_without_details(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _, store_dir = _initialized_store(tmp_path)
    holder = _DynamicMemoryHolder(Memory.open(store_dir, read_only=True), auto_sync=True)

    async def failing() -> object:
        raise CodeContextError(code="STORE_BUSY", message=f"busy {SECRET}")

    async def run() -> None:
        task: asyncio.Task[object] = asyncio.get_running_loop().create_task(failing())
        holder._sync_tasks.add(task)
        task.add_done_callback(holder._on_auto_sync_done)
        await asyncio.gather(task, return_exceptions=True)

    try:
        with caplog.at_level("WARNING", logger="codecontext.mcp_server"):
            asyncio.run(run())
        assert "STORE_BUSY" in caplog.text
        assert SECRET not in caplog.text
        assert not holder._sync_tasks
    finally:
        holder.close()


def _roots_ctx(*roots: Path) -> SimpleNamespace:
    async def list_roots() -> SimpleNamespace:
        return SimpleNamespace(roots=[SimpleNamespace(uri=p.as_uri()) for p in roots])

    return SimpleNamespace(session=SimpleNamespace(list_roots=list_roots))


def test_search_on_uninitialized_root_does_not_create_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    _setup_git_repo(root)
    monkeypatch.chdir(tmp_path)
    holder = _DynamicMemoryHolder(None)
    try:
        with pytest.raises(CodeContextError) as exc:
            asyncio.run(holder.get_memory(_roots_ctx(root)))
        assert exc.value.code == "STORE_NOT_INITIALIZED"
        assert "--auto-init" in exc.value.message
        assert not (root / ".codecontext").exists()

        with pytest.raises(CodeContextError) as pinned_exc:
            create_mcp_server(workspace_path=root)
        assert pinned_exc.value.code == "STORE_NOT_INITIALIZED"
        assert not (root / ".codecontext").exists()
    finally:
        holder.close()


def test_unsynced_store_stays_unsynced_without_auto_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    _setup_git_repo(root)
    Memory.initialize(root / ".codecontext", InitRequest(repo_path=root)).close()
    monkeypatch.chdir(tmp_path)
    holder = _DynamicMemoryHolder(None)
    try:
        mem = asyncio.run(holder.get_memory(_roots_ctx(root)))
        assert mem.status().revision is None
    finally:
        holder.close()
    _, pinned = create_mcp_server(workspace_path=root)
    try:
        assert asyncio.run(pinned.get_memory()).status().revision is None
    finally:
        pinned.close()


def test_auto_init_bootstraps_uninitialized_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    head = _setup_git_repo(root)
    monkeypatch.chdir(tmp_path)
    holder = _DynamicMemoryHolder(None, auto_init=True)
    try:
        mem = asyncio.run(holder.get_memory(_roots_ctx(root)))
        assert mem.status().revision == head
    finally:
        holder.close()


def test_cached_repo_not_blocked_by_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codecontext.mcp_server as mcp_server

    cached_root = tmp_path / "cached"
    _setup_git_repo(cached_root)
    slow_root = tmp_path / "slow"
    _setup_git_repo(slow_root)
    for root in (cached_root, slow_root):
        with Memory.initialize(root / ".codecontext", InitRequest(repo_path=root)) as mem:
            mem.sync(SyncRequest(revision="HEAD", offline=True))
    monkeypatch.chdir(tmp_path)
    holder = _DynamicMemoryHolder(None)
    ctx = _roots_ctx(cached_root, slow_root)

    entered = threading.Event()
    release = threading.Event()
    real_prepare = mcp_server._prepare_store

    def slow_prepare(st_path: Path, ws: Path | None, *, auto_init: bool) -> Path | None:
        if ws == slow_root.resolve():
            entered.set()
            release.wait(10)
        return real_prepare(st_path, ws, auto_init=auto_init)

    async def run() -> None:
        cached = await holder.get_memory(ctx, str(cached_root))
        monkeypatch.setattr(mcp_server, "_prepare_store", slow_prepare)
        slow = asyncio.create_task(holder.get_memory(ctx, str(slow_root)))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            assert await asyncio.wait_for(holder.get_memory(ctx, str(cached_root)), 5) is cached
        finally:
            release.set()
        assert (await slow) is not cached

    try:
        asyncio.run(run())
    finally:
        release.set()
        holder.close()


def test_roots_resolved_once_per_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import codecontext.mcp_server as mcp_server

    repo_dir, store_dir = _initialized_store(tmp_path)
    with Memory.initialize(repo_dir / ".codecontext", InitRequest(repo_path=repo_dir)) as mem:
        mem.sync(SyncRequest(revision="HEAD", offline=True))
    monkeypatch.chdir(tmp_path)
    server, holder = create_mcp_server()
    ctx = _roots_ctx(repo_dir)
    calls: list[Path] = []
    real_find = mcp_server._find_git_toplevel

    def counting_find(path: Path) -> Path | None:
        calls.append(path)
        return real_find(path)

    monkeypatch.setattr(mcp_server, "_find_git_toplevel", counting_find)

    async def fetch_ten() -> None:
        for _ in range(10):
            await holder.get_memory(ctx)

    async def list_roots(_ctx: object) -> ListRootsResult:
        return ListRootsResult(roots=[Root(uri=repo_dir.as_uri())])

    async def notify_roots_changed() -> None:
        async with Client(server, mode="legacy", list_roots_callback=list_roots) as client:
            # The SDK deprecates roots on both sides, but current clients still use them.
            with pytest.warns(MCPDeprecationWarning):
                assert not (await client.call_tool("list_repositories", {})).is_error
                assert holder._toplevel_cache
                await client.send_roots_list_changed()
            for _ in range(200):
                if not holder._toplevel_cache:
                    break
                await asyncio.sleep(0.01)
        assert not holder._toplevel_cache

    try:
        asyncio.run(fetch_ten())
        assert len(calls) == 1

        asyncio.run(notify_roots_changed())
        asyncio.run(fetch_ten())
        assert len(calls) == 2
    finally:
        holder.close()


def test_deleted_root_does_not_break_other_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gone = tmp_path / "gone"
    healthy = tmp_path / "healthy"
    for root in (gone, healthy):
        _setup_git_repo(root)
        with Memory.initialize(root / ".codecontext", InitRequest(repo_path=root)) as mem:
            mem.sync(SyncRequest(revision="HEAD", offline=True))
    monkeypatch.chdir(tmp_path)
    holder = _DynamicMemoryHolder(None)
    ctx = _roots_ctx(gone, healthy)

    async def run() -> None:
        healthy_mem = await holder.get_memory(ctx, str(healthy))
        shutil.rmtree(gone)
        assert await holder.repositories(ctx) == [healthy.resolve()]
        assert await holder.get_memory(ctx, str(healthy)) is healthy_mem

    try:
        asyncio.run(run())
        assert gone not in holder._toplevel_cache
    finally:
        holder.close()


def test_root_becomes_repository_without_notification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "later"
    root.mkdir()
    monkeypatch.chdir(tmp_path)
    holder = _DynamicMemoryHolder(None)
    ctx = _roots_ctx(root)
    try:
        assert asyncio.run(holder.repositories(ctx)) == []
        _setup_git_repo(root)
        assert asyncio.run(holder.repositories(ctx)) == [root.resolve()]
    finally:
        holder.close()


def test_serve_error_goes_to_stderr(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "missing"
    capsys.readouterr()
    assert main(["--store", str(missing), "serve"]) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert "STORE_NOT_INITIALIZED" in err


def test_list_repositories_wraps_os_errors(tmp_path: Path) -> None:
    repo_dir, store_dir = _initialized_store(tmp_path)
    server, holder = create_mcp_server(store_path=store_dir, workspace_path=repo_dir)

    async def call() -> Any:
        async with Client(server) as client:
            return await client.call_tool("list_repositories", {})

    repo_dir.chmod(0)
    try:
        result = asyncio.run(call())
    finally:
        repo_dir.chmod(0o755)
        holder.close()
    assert result.is_error
    text = result.content[0].text
    error = json.loads(text[text.index("{"):])["error"]
    assert error["code"] == "IO_ERROR"
    assert error["details"] == {"exception_type": "PermissionError"}
