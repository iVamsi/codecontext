import asyncio
import gc
import sqlite3
import threading
import time
from pathlib import Path

import anyio
import pytest

from codecontext import InitRequest, Memory, SyncRequest, graph
from codecontext.deadline import SyncDeadlineExceeded, sync_deadline
from codecontext.enrichment import SnapshotGroup, compute_snapshot_groups
from codecontext.graph import GetArchitectureRequest, detect_louvain_communities, find_scc_cycles
from codecontext.mcp_server import _run_in_thread
from tests.fixtures.builders import GitCommitSpec, commit_fixture_spec, init_fixture_repo


def _snapshot_conn(paths: list[bytes]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE snapshot_files (path BLOB PRIMARY KEY)")
    conn.executemany("INSERT INTO snapshot_files (path) VALUES (?)", [(p,) for p in paths])
    return conn


def _many_package_paths(count: int) -> list[bytes]:
    paths: list[bytes] = []
    for i in range(count):
        paths.append(b"p%05d/package.json" % i)
        paths.append(b"p%05d/mod.py" % i)
    return paths


def test_compute_snapshot_groups_picks_deepest_package_root() -> None:
    conn = _snapshot_conn(
        [
            b"README.md",
            b"a/pyproject.toml",
            b"a/x.py",
            b"a/b/package.json",
            b"a/b/c/y.py",
            b"a/bb/z.py",
            b"docs/guide.md",
        ]
    )

    groups = compute_snapshot_groups(conn, repository_id="r")

    assert [(g.scope_kind, g.path_bytes, g.files) for g in groups] == [
        ("root_files", b".", [b"README.md"]),
        ("package", b"a", [b"a/bb/z.py", b"a/pyproject.toml", b"a/x.py"]),
        ("package", b"a/b", [b"a/b/c/y.py", b"a/b/package.json"]),
        ("directory", b"docs", [b"docs/guide.md"]),
    ]


def test_compute_snapshot_groups_scales_with_many_package_roots() -> None:
    conn = _snapshot_conn(_many_package_paths(8000))

    started = time.monotonic()
    groups = compute_snapshot_groups(conn, repository_id="r")
    elapsed = time.monotonic() - started

    assert len(groups) == 8000
    assert elapsed < 1.0


def test_compute_snapshot_groups_honours_deadline() -> None:
    conn = _snapshot_conn(_many_package_paths(1000))

    with sync_deadline(0.0), pytest.raises(SyncDeadlineExceeded):
        compute_snapshot_groups(conn, repository_id="r")


def test_find_scc_cycles_honours_deadline() -> None:
    adj = {f"n{i}": {f"n{i + 1}"} for i in range(2000)}

    with sync_deadline(0.0), pytest.raises(SyncDeadlineExceeded):
        find_scc_cycles(adj)


def test_louvain_honours_deadline() -> None:
    nodes = [f"n{i}" for i in range(2000)]
    edges = [(nodes[i], nodes[i + 1], 1.0) for i in range(len(nodes) - 1)]

    with sync_deadline(0.0), pytest.raises(SyncDeadlineExceeded):
        detect_louvain_communities(nodes, edges)


def test_get_architecture_coupling_scales_with_many_clusters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(repo, GitCommitSpec(message="C0", files={"tools/util.py": b"def lonely():\n    return 0\n"}))
    groups = [
        SnapshotGroup(
            scope_kind="package",
            scope_id=f"s{i}",
            path_bytes=b"p%05d" % i,
            path_display=f"p{i:05d}",
            files=[b"p%05d/mod.py" % i],
        )
        for i in range(8000)
    ]
    monkeypatch.setattr(graph, "compute_snapshot_groups", lambda conn, *, repository_id: groups)

    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True))
        started = time.monotonic()
        arch = mem.get_architecture(GetArchitectureRequest(max_clusters=20))
        elapsed = time.monotonic() - started

    assert elapsed < 2.0
    assert len(arch.clusters) == 20
    assert set(arch.coupling) == {c.id for c in arch.clusters}


def test_cancelled_thread_call_keeps_slot_until_worker_finishes() -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def work() -> None:
        started.set()
        release.wait(5)
        finished.set()

    async def scenario() -> None:
        slot = asyncio.Semaphore(1)

        async def handler() -> None:
            async with slot:
                await _run_in_thread(work)

        task = asyncio.create_task(handler())
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0.05)
        assert slot.locked()
        assert not task.done()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
        assert not slot.locked()

    asyncio.run(scenario())


def test_anyio_scope_cancel_waits_for_worker() -> None:
    finished = threading.Event()

    def work() -> str:
        time.sleep(0.6)
        finished.set()
        return "done"

    async def scenario() -> None:
        cpu_started = time.process_time()
        with anyio.move_on_after(0.05):
            await _run_in_thread(work)
        assert finished.is_set()
        # The worker only sleeps, so a busy-waiting event loop would burn a large share of the 0.6 s in CPU.
        assert time.process_time() - cpu_started < 0.05

    anyio.run(scenario)


def test_run_in_thread_returns_result_and_raises_worker_errors() -> None:
    def boom() -> None:
        raise ValueError("bad")

    async def scenario() -> None:
        assert await _run_in_thread(lambda a, b: a + b, 2, 3) == 5
        with pytest.raises(ValueError, match="bad"):
            await _run_in_thread(boom)

    asyncio.run(scenario())


def test_worker_error_after_cancel_is_not_logged_as_unretrieved() -> None:
    started = threading.Event()
    release = threading.Event()

    def work() -> None:
        started.set()
        release.wait(5)
        raise ValueError("late failure")

    async def scenario() -> list[dict]:
        reported: list[dict] = []
        asyncio.get_running_loop().set_exception_handler(lambda _loop, context: reported.append(context))
        task = asyncio.create_task(_run_in_thread(work))
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0.05)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        del task
        gc.collect()
        await asyncio.sleep(0)
        return reported

    assert asyncio.run(scenario()) == []
