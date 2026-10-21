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
        time.sleep(0.04)
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
