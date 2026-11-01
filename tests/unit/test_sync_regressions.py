import time
from pathlib import Path

import httpx
import pytest

from codecontext import InitRequest, Memory, SyncRequest, syntax
from tests.fixtures.builders import (
    GitCommitSpec,
    commit_fixture_spec,
    init_fixture_repo,
    run_git_fixture,
)


def test_kotlin_property_chains(tmp_path: Path) -> None:
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(
        repo,
        GitCommitSpec(
            message="Property chains",
            files={
                "Example.kt": b"""
class Client {
 fun send() {}
 fun send(value: Int) {}
}
class Service(val client: Client?) {
 fun send() {}
}
class PlainService(val client: Client) {}
class Root {
 val service: Service? = null
 fun member() { this.service?.client?.send() }
}
class Other {
 fun send() {}
}
fun direct(service: Service) { service.send() }
fun chained(service: Service) { service.client?.send() }
fun plain(service: PlainService) { service.client.send() }
fun longer(root: Root?) { root?.service?.client?.send() }
fun unknown(service: Service) { service.missing.send() }
"""
            },
        ),
    )
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True))
        for caller, expected in {
            "direct": {"Service.send"},
            "chained": {"Client.send"},
            "plain": {"Client.send"},
            "longer": {"Client.send"},
            "Root.member": {"Client.send"},
            "unknown": set(),
        }.items():
            trace = mem.trace_path({"function_name": caller, "direction": "outbound", "depth": 1})
            assert {e.to_name for e in trace.edges} == expected, caller
        inbound = mem.trace_path(
            {"function_name": "Client.send", "direction": "inbound", "depth": 1}
        )
        assert {e.from_name for e in inbound.edges} == {"chained", "plain", "longer", "Root.member"}
        assert (
            len(
                mem.trace_path(
                    {"function_name": "plain", "direction": "outbound", "depth": 1}
                ).edges
            )
            == 2
        )


def test_sync_preserves_unchanged_rows(tmp_path: Path) -> None:
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(
        repo,
        GitCommitSpec(
            message="Files",
            files={
                "stable.py": b"def stable(): pass\n",
                "changed.py": b"def changed(): stable()\n",
            },
        ),
    )
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True))
        with mem._store.write_transaction() as conn:
            conn.execute("""
                CREATE TRIGGER preserve_snapshot BEFORE DELETE ON snapshot_files
                WHEN OLD.path = CAST('stable.py' AS BLOB)
                BEGIN SELECT RAISE(ABORT, 'unchanged snapshot row deleted'); END;
            """)
            conn.execute("""
                CREATE TRIGGER preserve_document BEFORE DELETE ON search_documents
                WHEN OLD.record_kind = 'code' AND json_extract(OLD.metadata_json, '$.path') = 'stable.py'
                BEGIN SELECT RAISE(ABORT, 'unchanged search document deleted'); END;
            """)
        again = mem.sync(SyncRequest(offline=True))
        assert again.blobs_parsed == 0
        (repo / "changed.py").write_text("def renamed(): stable()\n")
        edited = mem.sync(SyncRequest(offline=True))
        assert edited.blobs_parsed == 1
        assert not mem.search_graph({"name_pattern": "^changed$"}).nodes
        trace = mem.trace_path({"function_name": "stable", "direction": "inbound", "depth": 1})
        assert {e.from_name for e in trace.edges} == {"renamed"}
        run_git_fixture(repo, ["mv", "changed.py", "moved.py"])
        mem.sync(SyncRequest(offline=True))
        assert mem.search_graph({"name_pattern": "^renamed$"}).nodes[0].file_path == "moved.py"
        (repo / "moved.py").unlink()
        mem.sync(SyncRequest(offline=True))
        assert not mem.search_graph({"name_pattern": "^renamed$"}).nodes
        assert not mem.trace_path({"function_name": "stable", "direction": "inbound"}).edges


def test_cached_blob_reads_share_batch_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codecontext.git import CatFileBatchSession

    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(
        repo,
        GitCommitSpec(
            message="Files",
            files={f"old/file{i}.py": f"def function{i}(): pass\n".encode() for i in range(8)},
        ),
    )
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True))
        blob_readers = set()
        original = CatFileBatchSession.read_object

        def record_reader(batch, oid):
            obj = original(batch, oid)
            if obj is not None and obj[0] == "blob":
                blob_readers.add(batch)
            return obj

        monkeypatch.setattr(CatFileBatchSession, "read_object", record_reader)
        mem.sync(SyncRequest(offline=True))
        assert not blob_readers
        run_git_fixture(repo, ["mv", "old", "new"])
        report = mem.sync(SyncRequest(offline=True))
        assert report.blobs_parsed == 0
        assert len(blob_readers) <= 2  # One document reader and one history reader.
        assert {n.file_path for n in mem.search_graph({}).nodes} == {
            f"new/file{i}.py" for i in range(8)
        }


def test_git_subprocess_obeys_remaining_budget() -> None:
    from codecontext.deadline import SyncDeadlineExceeded, sync_deadline
    from codecontext.git import run_safe_git

    start = time.monotonic()
    with pytest.raises(SyncDeadlineExceeded), sync_deadline(0.05):
        run_safe_git(["-c", "alias.wait=!sleep 5", "wait"])
    assert time.monotonic() - start < 1.0


@pytest.mark.parametrize("provider", ["github", "gitlab"])
def test_http_waits_use_remaining_sync_budget(provider: str) -> None:
    from codecontext.deadline import sync_deadline
    from codecontext.github import GitHubClient
    from codecontext.gitlab import GitLabClient

    def handler(request):
        assert 0 < request.extensions["timeout"]["read"] <= 0.1
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    client = (GitHubClient(transport=transport) if provider == "github"
              else GitLabClient(base_url="https://gitlab.com", transport=transport))
    with client, sync_deadline(0.1):
        client.get_json("/test")


def test_cat_file_wait_obeys_remaining_budget(tmp_path: Path) -> None:
    import subprocess
    import sys

    from codecontext.deadline import SyncDeadlineExceeded, sync_deadline
    from codecontext.git import CatFileBatchSession

    start = time.monotonic()
    with CatFileBatchSession(tmp_path, tmp_path) as batch:
        batch._proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
        )
        with pytest.raises(SyncDeadlineExceeded), sync_deadline(0.05):
            batch.read_object("a" * 40)
    assert time.monotonic() - start < 1.0


def test_deadline_during_index_update_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(
        repo, GitCommitSpec(message="First", files={"a.py": b"def target(): pass\n"})
    )
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True))
        (repo / "a.py").write_text("def target(): pass\ndef caller(): target()\n")
        original = syntax._resolve_cross_file_calls_and_imports

        def slow_index(conn):
            conn.execute(
                "WITH RECURSIVE n(x) AS (VALUES(0) UNION ALL SELECT x+1 FROM n WHERE x<100000000) SELECT sum(x) FROM n"
            ).fetchone()

        monkeypatch.setattr(syntax, "_resolve_cross_file_calls_and_imports", slow_index)
        start = time.monotonic()
        partial = mem.sync(SyncRequest(offline=True, max_seconds=1))
        assert time.monotonic() - start < 2.0
        assert partial.stop_reason == "time_budget"
        assert partial.coverage.structure.state == "partial"
        monkeypatch.setattr(syntax, "_resolve_cross_file_calls_and_imports", original)
        complete = mem.sync(SyncRequest(offline=True))
        assert complete.coverage.structure.state == "complete"
        assert mem.search({"query": "caller", "kind": "code"}).hits
        assert {
            e.to_name
            for e in mem.trace_path({"function_name": "caller", "direction": "outbound"}).edges
        } == {"target"}


def test_deadline_during_import_preserves_previous_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codecontext.git import GitMirror, run_safe_git

    repo = init_fixture_repo(tmp_path / "repo")
    first = commit_fixture_spec(
        repo, GitCommitSpec(message="First", files={"a.py": b"def old(): pass\n"})
    )
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True))
        commit_fixture_spec(
            repo, GitCommitSpec(message="Second", files={"a.py": b"def new(): pass\n"})
        )
        original = GitMirror.import_local_refs_and_revision

        def delayed_import(*args, **kwargs):
            run_safe_git(["-c", "alias.wait=!sleep 5", "wait"])
            return original(*args, **kwargs)

        monkeypatch.setattr(GitMirror, "import_local_refs_and_revision", delayed_import)
        start = time.monotonic()
        partial = mem.sync(SyncRequest(offline=True, max_seconds=1))
        assert time.monotonic() - start < 2.0
        assert partial.stop_reason == "time_budget" and partial.revision == first
        assert partial.coverage.structure.state == "partial"
        assert mem.search_graph({"name_pattern": "^old$"}).nodes
        monkeypatch.setattr(GitMirror, "import_local_refs_and_revision", original)
        complete = mem.sync(SyncRequest(offline=True))
        assert complete.coverage.structure.state == "complete"
        assert mem.search_graph({"name_pattern": "^new$"}).nodes


def test_sync_deadline_covers_syntax_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(
        repo,
        GitCommitSpec(
            message="Files",
            files={f"file{i}.py": f"def function{i}(): pass\n".encode() for i in range(60)},
        ),
    )
    original = syntax.parse_blob

    def slow_parse(*args, **kwargs):
        time.sleep(0.3)
        return original(*args, **kwargs)

    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        monkeypatch.setattr(syntax, "parse_blob", slow_parse)
        start = time.monotonic()
        report = mem.sync(SyncRequest(offline=True, max_seconds=1))
        assert time.monotonic() - start < 2.0
        assert report.stop_reason == "time_budget"
        assert report.state == "partial"
        assert report.coverage.structure.state != "complete"
        monkeypatch.setattr(syntax, "parse_blob", original)
        resumed = mem.sync(SyncRequest(offline=True, max_seconds=30))
        assert resumed.coverage.structure.state == "complete"
        assert mem.search_graph({"limit": 100}).total_matched == 60


@pytest.mark.parametrize("merge", [False, True])
def test_shallow_source_import(tmp_path: Path, merge: bool) -> None:
    source = init_fixture_repo(tmp_path / "source")
    commit_fixture_spec(
        source, GitCommitSpec(message="Base", files={"base.py": b"def base(): pass\n"})
    )
    if merge:
        run_git_fixture(source, ["checkout", "-b", "side"])
        commit_fixture_spec(
            source, GitCommitSpec(message="Side", files={"side.py": b"def side(): pass\n"})
        )
        run_git_fixture(source, ["checkout", "main"])
    commit_fixture_spec(
        source, GitCommitSpec(message="Main", files={"main.py": b"def main(): pass\n"})
    )
    if merge:
        run_git_fixture(source, ["merge", "--no-ff", "side", "-m", "Merge"])
    shallow = tmp_path / "shallow"
    run_git_fixture(source, ["clone", "--depth=1", "--no-local", str(source), str(shallow)])
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=shallow)) as mem:
        report = mem.sync(SyncRequest(offline=True))
        assert mem.search_graph({"name_pattern": "^main$"}).nodes
        assert report.coverage.git.state != "complete"
        assert any(g.code == "SHALLOW_HISTORY" for g in report.gaps)

        repeated = mem.sync(SyncRequest(offline=True))
        assert any(g.code == "SHALLOW_HISTORY" for g in repeated.gaps)
        with mem._store.write_transaction() as conn:
            boundary = conn.execute("SELECT oid, diff_state FROM commits").fetchone()
            assert boundary["diff_state"] == "partial"
            # Simulate an older store that incorrectly marked the boundary diff complete.
            conn.execute("UPDATE commits SET diff_state = 'complete'")
        run_git_fixture(shallow, ["fetch", "--unshallow", "origin"])
        recovered = mem.sync(SyncRequest(offline=True))
        assert not any(g.code == "SHALLOW_HISTORY" for g in recovered.gaps)
        assert recovered.coverage.git.state == "complete"
        with mem._store.connect_read() as conn:
            assert conn.execute("SELECT COUNT(*) FROM commits").fetchone()[0] == (4 if merge else 2)
            assert conn.execute("SELECT COUNT(*) FROM commits WHERE diff_state != 'complete'").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM changes WHERE commit_oid = ?", (boundary["oid"],)).fetchone()[0] > 0


def _linear_repo(path: Path, commits: int) -> Path:
    repo = init_fixture_repo(path)
    for i in range(commits):
        commit_fixture_spec(
            repo, GitCommitSpec(message=f"C{i}", files={f"f{i}.py": f"def f{i}(): pass\n".encode()})
        )
    return repo


def test_fetch_uses_sync_budget_not_60s(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from codecontext import git
    from codecontext.deadline import remaining_timeout

    repo = _linear_repo(tmp_path / "repo", 2)
    original = git.run_safe_git
    fetch_budgets: list[float] = []

    def record(args, **kwargs):
        if args and args[0] == "fetch":
            fetch_budgets.append(remaining_timeout(kwargs.get("timeout", 30.0)))
        return original(args, **kwargs)

    monkeypatch.setattr(git, "run_safe_git", record)
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        report = mem.sync(SyncRequest(offline=True, max_seconds=600))
    assert report.state in ("complete", "partial") and report.stop_reason != "error"
    assert fetch_budgets and all(budget > 500 for budget in fetch_budgets)


def test_interrupted_initial_fetch_resumes_in_depth_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codecontext import git
    from codecontext.deadline import SyncDeadlineExceeded

    repo = _linear_repo(tmp_path / "repo", 5)
    monkeypatch.setattr(git, "FETCH_DEPTH_STEP", 2)
    original = git.run_safe_git
    fetches: list[str] = []
    allowed = {"n": 0}

    def limited(args, **kwargs):
        if args and args[0] == "fetch":
            mode = next((a for a in args if a.startswith(("--depth", "--deepen"))), "full")
            if allowed["n"] == 0:
                raise SyncDeadlineExceeded
            allowed["n"] -= 1
            fetches.append(mode)
        return original(args, **kwargs)

    monkeypatch.setattr(git, "run_safe_git", limited)
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        assert mem.sync(SyncRequest(offline=True)).stop_reason == "time_budget"
        allowed["n"] = 1
        assert mem.sync(SyncRequest(offline=True)).stop_reason == "time_budget"
        assert fetches == ["--depth=2"]
        allowed["n"] = 100
        report = mem.sync(SyncRequest(offline=True))
        assert fetches == ["--depth=2", "--deepen=2", "--deepen=2"]
        assert report.coverage.git.state == "complete"
        assert not (mem._mirror.git_dir / "shallow").exists()
        with mem._store.connect_read() as conn:
            assert conn.execute("SELECT COUNT(*) FROM commits").fetchone()[0] == 5
        mem.sync(SyncRequest(offline=True))
        assert fetches[-1] == "full"
        assert not (mem._mirror.git_dir / git.FETCH_RESUME_MARKER).exists()


def test_one_failing_commit_does_not_block_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codecontext import sync
    from codecontext.errors import CodeContextError

    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(repo, GitCommitSpec(message="C0", files={"a.py": b"def a(): pass\n"}))
    bad = commit_fixture_spec(
        repo, GitCommitSpec(message="C1", files={"b.py": b"def b(): pass\n"})
    )
    commit_fixture_spec(repo, GitCommitSpec(message="C2", files={"c.py": b"def c(): pass\n"}))
    original = sync.index_commit_changes_and_hunks

    def failing(store, mirror, commit_oid, **kwargs):
        result = original(store, mirror, commit_oid, **kwargs)
        if commit_oid == bad:
            raise CodeContextError(code="IO_ERROR", message="boom", retryable=True)
        return result

    monkeypatch.setattr(sync, "index_commit_changes_and_hunks", failing)

    def job_rows():
        with mem._store.connect_read() as conn:
            return {
                r["owner_id"]: r
                for r in conn.execute("SELECT * FROM jobs WHERE kind = 'git_commit_diff'")
            }

    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        report = mem.sync(SyncRequest(offline=True))
        assert report.stop_reason != "time_budget"
        jobs = job_rows()
        assert sum(r["state"] == "complete" for r in jobs.values()) == len(jobs) - 1
        assert jobs[bad]["attempts"] == 1 and jobs[bad]["state"] == "pending"
        assert jobs[bad]["last_error_code"] == "IO_ERROR"
        with mem._store.connect_read() as conn:
            assert not conn.execute("SELECT 1 FROM changes WHERE commit_oid = ?", (bad,)).fetchone()

        mem.sync(SyncRequest(offline=True))
        report = mem.sync(SyncRequest(offline=True))
        assert job_rows()[bad]["attempts"] == sync.MAX_DIFF_JOB_ATTEMPTS
        assert job_rows()[bad]["state"] == "retryable_error"
        assert any(g.code == "IO_ERROR" and bad in g.scope for g in report.gaps)

        mem.sync(SyncRequest(offline=True))
        assert job_rows()[bad]["attempts"] == sync.MAX_DIFF_JOB_ATTEMPTS
        assert job_rows()[bad]["state"] == "retryable_error"

        monkeypatch.setattr(sync, "index_commit_changes_and_hunks", original)
        mem.sync(SyncRequest(offline=True))
        assert job_rows()[bad]["state"] == "retryable_error"
        recovered = mem.sync(SyncRequest(offline=True, reconcile=True))
        assert job_rows()[bad]["state"] == "complete"
        assert not any(g.code == "IO_ERROR" for g in recovered.gaps)
        assert recovered.coverage.git.state == "complete"


def test_unexpected_error_in_one_commit_does_not_block_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codecontext import sync

    repo = _linear_repo(tmp_path / "repo", 3)
    bad = run_git_fixture(repo, ["rev-parse", "HEAD~1"]).decode().strip()
    original = sync.index_commit_changes_and_hunks

    def failing(store, mirror, commit_oid, **kwargs):
        if commit_oid == bad:
            raise ValueError("unexpected")
        return original(store, mirror, commit_oid, **kwargs)

    monkeypatch.setattr(sync, "index_commit_changes_and_hunks", failing)
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        report = mem.sync(SyncRequest(offline=True))
        assert report.stop_reason != "time_budget"
        with mem._store.connect_read() as conn:
            jobs = {
                r["owner_id"]: r
                for r in conn.execute("SELECT * FROM jobs WHERE kind = 'git_commit_diff'")
            }
        assert jobs[bad]["state"] == "pending" and jobs[bad]["last_error_code"] == "INTERNAL"
        assert sum(r["state"] == "complete" for r in jobs.values()) == len(jobs) - 1
        mem.sync(SyncRequest(offline=True))
        mem.sync(SyncRequest(offline=True))
        with mem._store.connect_read() as conn:
            message = conn.execute(
                "SELECT message FROM coverage_gaps WHERE code = 'IO_ERROR' AND scope = ?",
                (f"git:commit:{bad}",),
            ).fetchone()["message"]
        assert "INTERNAL: ValueError" in message and "unexpected" not in message


def test_sqlite_error_in_job_propagates_without_burning_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    from codecontext import sync

    repo = _linear_repo(tmp_path / "repo", 3)
    bad = run_git_fixture(repo, ["rev-parse", "HEAD~1"]).decode().strip()
    original = sync.index_commit_changes_and_hunks

    def interrupted(store, mirror, commit_oid, **kwargs):
        if commit_oid == bad:
            raise sqlite3.OperationalError("interrupted")
        return original(store, mirror, commit_oid, **kwargs)

    monkeypatch.setattr(sync, "index_commit_changes_and_hunks", interrupted)
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        with pytest.raises(sqlite3.OperationalError, match="interrupted"):
            mem.sync(SyncRequest(offline=True))
        with mem._store.connect_read() as conn:
            row = conn.execute(
                "SELECT attempts, last_error_code FROM jobs WHERE owner_id = ?", (bad,)
            ).fetchone()
        assert row["attempts"] == 0 and row["last_error_code"] is None


def test_missing_parent_skip_lifts_once_first_parent_exists(tmp_path: Path) -> None:
    from codecontext.store import Store

    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(repo, GitCommitSpec(message="Base", files={"a.py": b"def a(): pass\n"}))
    run_git_fixture(repo, ["checkout", "-b", "side"])
    commit_fixture_spec(repo, GitCommitSpec(message="Side", files={"s.py": b"def s(): pass\n"}))
    run_git_fixture(repo, ["checkout", "main"])
    commit_fixture_spec(repo, GitCommitSpec(message="Main", files={"m.py": b"def m(): pass\n"}))
    run_git_fixture(repo, ["merge", "--no-ff", "side", "-m", "Merge"])
    merge = run_git_fixture(repo, ["rev-parse", "HEAD"]).decode().strip()
    first = run_git_fixture(repo, ["rev-parse", "HEAD^1"]).decode().strip()
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True))
        store = mem._store

        def merge_queued(first_parent: str, second_parent: str) -> bool:
            with store.write_transaction() as conn:
                for index, parent in [(0, first_parent), (1, second_parent)]:
                    conn.execute(
                        "UPDATE commit_parents SET parent_oid = ? WHERE commit_oid = ? AND parent_index = ?",
                        (parent, merge, index),
                    )
                conn.execute("UPDATE commits SET diff_state = 'partial' WHERE oid = ?", (merge,))
                conn.execute("DELETE FROM jobs WHERE owner_id = ?", (merge,))
                Store.upsert_gap(
                    conn,
                    repository_id=store.config.repository_id,
                    code="MISSING_PARENT",
                    scope=f"git:commit:{merge}",
                    message="missing",
                    retryable=True,
                )
                Store.enqueue_pending_diff_jobs(conn)
                return conn.execute("SELECT 1 FROM jobs WHERE owner_id = ?", (merge,)).fetchone() is not None

        # Lineage only diffs against the first parent, so a side parent that never arrives
        # must not hold the merge back forever.
        assert not merge_queued("e" * 40, "f" * 40)
        assert merge_queued(first, "f" * 40)


def test_shallow_boundary_not_rediffed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from codecontext import sync

    source = _linear_repo(tmp_path / "source", 3)
    shallow = tmp_path / "shallow"
    run_git_fixture(source, ["clone", "--depth=1", "--no-local", str(source), str(shallow)])
    original = sync.index_commit_changes_and_hunks
    diffed: list[str] = []

    def counting(store, mirror, commit_oid, **kwargs):
        diffed.append(commit_oid)
        return original(store, mirror, commit_oid, **kwargs)

    monkeypatch.setattr(sync, "index_commit_changes_and_hunks", counting)
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=shallow)) as mem:
        mem.sync(SyncRequest(offline=True))
        n1 = len(diffed)
        assert n1 == 1
        mem.sync(SyncRequest(offline=True))
        assert len(diffed) == n1


def test_diff_blobs_unified_zero_rejects_non_oid(tmp_path: Path) -> None:
    from codecontext.errors import CodeContextError
    from codecontext.git import GitMirror
    from codecontext.store import Store

    repo = _linear_repo(tmp_path / "repo", 1)
    store = Store.initialize(tmp_path / "store", InitRequest(repo_path=repo))
    mirror = GitMirror(store)
    try:
        for bad in ("--output=/tmp/x", "HEAD"):
            with pytest.raises(CodeContextError) as exc:
                mirror.diff_blobs_unified_zero(bad, None)
            assert exc.value.code == "INVALID_REQUEST"
    finally:
        mirror.close()
        store.close()


def test_anchor_mapping_survives_git_diff_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codecontext.errors import CodeContextError
    from codecontext.git import PYTHON_DIFF_MAX_LINES, GitMirror
    from codecontext.lineage import map_original_anchor_to_target
    from codecontext.store import Store

    repo = init_fixture_repo(tmp_path / "repo")
    lines = [f"line {i}\n" for i in range(PYTHON_DIFF_MAX_LINES + 10)]
    first = commit_fixture_spec(
        repo, GitCommitSpec(message="C0", files={"big.txt": "".join(lines).encode()})
    )
    second = commit_fixture_spec(
        repo,
        GitCommitSpec(message="C1", files={"big.txt": ("new\n" + "".join(lines)).encode()}),
    )
    store = Store.initialize(tmp_path / "store", InitRequest(repo_path=repo))
    mirror = GitMirror(store)

    def broken_diff(self, old_oid, new_oid):
        raise CodeContextError(code="IO_ERROR", message="git diff failed")

    monkeypatch.setattr(GitMirror, "diff_blobs_unified_zero", broken_diff)
    try:
        mirror.prepare_snapshot("HEAD")
        with store.connect_read() as conn:
            result = map_original_anchor_to_target(
                conn,
                mirror,
                original_commit_oid=first,
                original_path=b"big.txt",
                original_line_start=5,
                original_line_end=6,
                target_revision=second,
            )
    finally:
        mirror.close()
        store.close()
    assert result.mapping == "unmapped" and result.gap_code == "MAPPING_UNAVAILABLE"
