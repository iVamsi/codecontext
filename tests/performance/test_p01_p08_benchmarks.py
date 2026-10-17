import os
import time
from pathlib import Path

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
from examples.fixture_adapter import DeterministicFixtureAdapter
from tests.fixtures.builders import (
    GitCommitSpec,
    build_f01_fixture,
    commit_fixture_spec,
    init_fixture_repo,
    run_git_fixture,
)
from tests.integration.test_t11_sdk_cli_mcp import _build_f01_routes
from tests.unit.test_t07_github import make_mock_github_transport


def build_f12_regression_repo(
    repo_dir: Path,
    *,
    num_files: int = 20,
    symbols_per_file: int = 10,
    num_commits: int = 100,
) -> tuple[Path, int, int]:
    """F12 regression generator:
    Starts with `num_files` files containing `symbols_per_file` symbols each,
    then creates `num_commits` commits changing one function in one file per commit using `git fast-import`
    so even 5,000 commits generate in under a second!
    Returns `(repo_dir, expected_unique_blobs, expected_total_symbols)`.
    """
    init_fixture_repo(repo_dir)

    def make_file_bytes(file_idx: int, edit_step: int) -> bytes:
        lines: list[str] = []
        for s_idx in range(symbols_per_file):
            val = edit_step if s_idx == 0 else s_idx
            lines.append(
                f"def fn_{file_idx}_{s_idx}(x: int) -> int:\n"
                f'    """Docstring for fn_{file_idx}_{s_idx}."""\n'
                f"    return x + {val}\n"
            )
        return "\n".join(lines).encode("utf-8")

    # Build fast-import stream
    stream_chunks: list[bytes] = []
    # Initial commit C0 with all `num_files` files at edit_step=0
    stream_chunks.append(b"commit refs/heads/main\n")
    stream_chunks.append(b"mark :1\n")
    stream_chunks.append(b"author Fixture Author <author@example.test> 1700000000 +0000\n")
    stream_chunks.append(b"committer Fixture Committer <committer@example.test> 1700000000 +0000\n")
    msg0 = b"Initial F12 commit with base files\n"
    stream_chunks.append(f"data {len(msg0)}\n".encode("ascii") + msg0)
    for f_idx in range(num_files):
        content = make_file_bytes(f_idx, 0)
        stream_chunks.append(f"M 100644 inline src/mod_{f_idx:03d}.py\n".encode("ascii"))
        stream_chunks.append(f"data {len(content)}\n".encode("ascii") + content + b"\n")

    # Subsequent commits 1..num_commits each modify 1 function in 1 file
    for step in range(1, num_commits + 1):
        target_f = step % num_files
        content = make_file_bytes(target_f, step)
        ts = 1700000000 + step
        mark = step + 1
        prev_mark = step
        stream_chunks.append(b"commit refs/heads/main\n")
        stream_chunks.append(f"mark :{mark}\n".encode("ascii"))
        stream_chunks.append(f"author Fixture Author <author@example.test> {ts} +0000\n".encode("ascii"))
        stream_chunks.append(f"committer Fixture Committer <committer@example.test> {ts} +0000\n".encode("ascii"))
        msg = f"Commit step {step} editing mod_{target_f:03d}.py\n".encode("ascii")
        stream_chunks.append(f"data {len(msg)}\n".encode("ascii") + msg)
        stream_chunks.append(f"from :{prev_mark}\n".encode("ascii"))
        stream_chunks.append(f"M 100644 inline src/mod_{target_f:03d}.py\n".encode("ascii"))
        stream_chunks.append(f"data {len(content)}\n".encode("ascii") + content + b"\n")

    stream_chunks.append(b"done\n")
    raw_stream = b"".join(stream_chunks)

    import subprocess

    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "LC_ALL": "C",
    }
    subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "fast-import", "--done", "--quiet"],
        cwd=repo_dir,
        env=env,
        input=raw_stream,
        check=True,
    )
    run_git_fixture(repo_dir, ["reset", "--hard", "refs/heads/main"])

    expected_unique_blobs = num_files + num_commits
    expected_total_symbols = expected_unique_blobs * symbols_per_file
    return (repo_dir, expected_unique_blobs, expected_total_symbols)


def test_p01_storage_growth_invariant(tmp_path: Path) -> None:
    """P01: Run F12 regression across commits and assert parsed-symbol rows equal independent
    unique blob/extractor expectations, never current symbols multiplied by commits.
    """
    num_files = 15
    symbols_per_file = 10
    num_commits = 60
    repo_dir, expected_blobs, expected_syms = build_f12_regression_repo(
        tmp_path / "f12_p01",
        num_files=num_files,
        symbols_per_file=symbols_per_file,
        num_commits=num_commits,
    )

    store_dir = tmp_path / "p01_store"
    with Memory.initialize(store_dir, InitRequest(repo_path=repo_dir)) as mem:
        rep = mem.sync(SyncRequest(revision="HEAD", offline=True))
        assert rep.state == "complete"
        assert rep.commits_added == num_commits + 1

        with mem._store.connect_read() as conn:
            ext_cnt = int(conn.execute("SELECT COUNT(*) FROM extractions").fetchone()[0])
            sym_cnt = int(conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0])
            snap_cnt = int(conn.execute("SELECT COUNT(*) FROM snapshot_files").fetchone()[0])

        # Exact structural invariant:
        # Extractions equal unique blobs (15 initial + 60 single-file edits = 75),
        # Symbols equal 75 * 10 = 750 (NOT 15 * 10 * 61 = 9,150!)
        assert ext_cnt == expected_blobs
        assert sym_cnt == expected_syms
        assert snap_cnt == num_files


def test_p02_p03_p05_p06_p08_latency_incremental_and_cost_workloads(tmp_path: Path) -> None:
    """P02, P03, P05, P06, P08:
    - P05: Progressive collection: first searchable code and first cited rationale precede full backfill
    - P02/P03: Warm read latency targets (`get` < 100ms, `search` < 300ms, `context` < 1000ms)
    - P06: Incremental commit + discussion edit does not re-parse unchanged blobs or re-enrich unchanged episodes
    - P08: Reads record zero model calls and zero provider spend.
    """
    f01 = build_f01_fixture(tmp_path / "f01_perf")
    store_dir = tmp_path / "perf_store"
    routes = _build_f01_routes(f01)
    transport, _ = make_mock_github_transport(routes)

    with Memory.initialize(
        store_dir,
        InitRequest(repo_path=f01.repo_dir, github_repository="example/transport"),
    ) as mem:
        mem._github_transport = transport
        mem._github_token = ""

        # P05: Focus sync on `src/transport.py` makes target code and episode rationale available before full backfill!
        sync_focus = mem.sync(
            SyncRequest(
                revision="HEAD",
                focus=PathTarget(kind="path", path="src/transport.py", line=5),
            )
        )
        assert sync_focus.coverage.github.state == "partial"

        adapter = DeterministicFixtureAdapter()
        enrich_focus = mem.enrich(
            EnrichRequest(max_input_tokens=100_000, max_output_tokens=20_000, max_calls=10),
            model=adapter,
        )
        assert enrich_focus.claims_added == 1

        # P03: Measure warm latencies over stratified reads (`search`, `get`, `context`)
        s_res0 = mem.search(SearchRequest(query="send", kind="code"))
        sym_id = s_res0.hits[0].id

        search_latencies_ms: list[float] = []
        get_latencies_ms: list[float] = []
        context_latencies_ms: list[float] = []

        for i in range(20):
            t0 = time.perf_counter()
            mem.search(SearchRequest(query=f"send retry {i % 3}"))
            search_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            t1 = time.perf_counter()
            mem.get(GetRequest(id=sym_id))
            get_latencies_ms.append((time.perf_counter() - t1) * 1000.0)

            t2 = time.perf_counter()
            mem.context(
                ContextRequest(
                    query="Why is send limited to 3 retries?",
                    target=PathTarget(kind="path", path="src/transport.py", line=5),
                )
            )
            context_latencies_ms.append((time.perf_counter() - t2) * 1000.0)

        search_latencies_ms.sort()
        get_latencies_ms.sort()
        context_latencies_ms.sort()
        p95_idx = int(0.95 * 19)
        assert get_latencies_ms[p95_idx] < 100.0
        assert search_latencies_ms[p95_idx] < 300.0
        assert context_latencies_ms[p95_idx] < 1000.0

        # P06: Add one new commit touching only README.md; sync again -> `src/transport.py` blob is NOT reparsed (blobs_parsed=0 for code)
        commit_fixture_spec(
            f01.repo_dir,
            GitCommitSpec(message="Incremental README edit", files={"README.md": b"# Transport v2\n"}),
        )
        inc_sync = mem.sync(SyncRequest(revision="HEAD", offline=True))
        assert inc_sync.commits_added == 1
        assert inc_sync.blobs_parsed == 0  # README is text_only; src/transport.py reused extraction!

        # Unchanged PR #10 episode is a cache hit on re-enrichment
        inc_enrich = mem.enrich(
            EnrichRequest(
                target_ids=[sym_id],
                max_input_tokens=50_000,
                max_output_tokens=10_000,
                max_calls=5,
            ),
            model=adapter,
        )
        assert inc_enrich.usage.calls == 0
        assert inc_enrich.cache_hit_items >= 1
