import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

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

    server, holder = create_mcp_server(store_path=store_dir, workspace_path=repo_dir)
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
