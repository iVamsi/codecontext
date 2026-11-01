import time
from pathlib import Path

import pytest

from codecontext import InitRequest, Memory, SyncRequest, graph
from codecontext.errors import CodeContextError
from codecontext.graph import (
    DetectChangesRequest,
    GetArchitectureRequest,
    QueryGraphRequest,
    SearchGraphRequest,
    TracePathRequest,
    compile_safe_regex,
    is_test_path,
)
from tests.fixtures.builders import GitCommitSpec, commit_fixture_spec, init_fixture_repo

MOD_V0 = (
    b"def keep(a):\n"
    b"    return a\n\n"
    b"def sig(a):\n"
    b"    return a\n\n"
    b"def body(a):\n"
    b"    return a\n\n"
    b"def gone(a):\n"
    b"    return a\n"
)
MOD_V1 = (
    b"# shifts every line down without touching keep\n"
    b"def keep(a):\n"
    b"    return a\n\n"
    b"def sig(a, b):\n"
    b"    return a\n\n"
    b"def body(a):\n"
    b"    return a + 1\n\n"
    b"def fresh():\n"
    b"    return 0\n"
)


def _build_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = init_fixture_repo(tmp_path / "repo")
    c0 = commit_fixture_spec(
        repo,
        GitCommitSpec(
            message="C0",
            files={
                "src/mod.py": MOD_V0,
                "src/old.py": b"def legacy():\n    return 1\n",
                "src/attestation.py": b"from src.mod import body\n\ndef attest():\n    return body(1)\n",
                "tests/test_mod.py": b"from src.mod import body\n\ndef test_body():\n    assert body(1)\n",
            },
        ),
    )
    commit_fixture_spec(
        repo,
        GitCommitSpec(message="C1", files={"src/mod.py": MOD_V1, "src/old.py": None}),
    )
    return repo, c0


def test_detect_changes_classifies_hunks_and_finds_deleted_symbols(tmp_path: Path) -> None:
    repo, c0 = _build_repo(tmp_path)
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(revision="HEAD", offline=True))
        res = mem.detect_changes(DetectChangesRequest(since=c0, depth=2))

    assert res.changed_files == ["src/mod.py", "src/old.py"]
    assert res.classifications == {
        "sig": "signature_modified",
        "body": "body_modified",
        "fresh": "added",
        "gone": "deleted",
        "legacy": "deleted",
    }
    assert res.deleted_symbols == ["gone", "legacy"]
    assert [s.name for s in res.directly_modified_symbols] == ["sig", "body", "fresh"]
    assert {s.name for s in res.affected_callers} == {"attest", "test_body"}
    assert res.affected_tests == ["test_body"]


def test_detect_changes_worktree_deletion(tmp_path: Path) -> None:
    repo, _ = _build_repo(tmp_path)
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(revision="HEAD", offline=True))
        (repo / "src" / "mod.py").write_bytes(MOD_V1.replace(b"\ndef fresh():\n    return 0\n", b""))
        mem.configure_workspace(repo)
        res = mem.detect_changes(DetectChangesRequest())

    assert res.changed_files == ["src/mod.py"]
    assert res.classifications == {"fresh": "deleted"}
    assert res.deleted_symbols == ["fresh"]
    assert res.directly_modified_symbols == []


def test_get_architecture_uses_louvain_communities_deterministically(tmp_path: Path) -> None:
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(
        repo,
        GitCommitSpec(
            message="C0",
            files={
                "app/api.py": b"from core.db import store_row\n\ndef serve():\n    return store_row(1)\n",
                "core/db.py": b"def store_row(x):\n    return x\n",
                "ui/view.py": b"from ui.widgets import draw_box\n\ndef render():\n    return draw_box()\n",
                "ui/widgets.py": b"def draw_box():\n    return 1\n",
                "tools/util.py": b"def lonely():\n    return 0\n",
            },
        ),
    )
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True))
        first = mem.get_architecture(GetArchitectureRequest())
        second = mem.get_architecture(GetArchitectureRequest())

    assert first == second
    by_id = {c.id: c for c in first.clusters}
    assert set(by_id) == {"community:.", "community:ui", "tools"}
    assert by_id["community:."].file_count == 2
    assert by_id["community:."].scope_kind == "community"
    assert by_id["community:."].internal_calls == 1
    assert by_id["community:ui"].path == "ui"
    assert by_id["tools"].scope_kind == "directory"
    assert first.cycles == []


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/attestation.py", False),
        ("src/contest/latest.py", False),
        ("testing_utils.py", False),
        ("tests/helpers.py", True),
        ("pkg/test/conf.py", True),
        ("test_mod.py", True),
        ("pkg/mod_test.py", True),
        ("cmd/server_test.go", True),
        ("web/app.test.ts", True),
        ("web/app.spec.ts", True),
        ("web/__tests__/app.js", True),
        ("src/main/FooTest.java", True),
        ("src/main/FooTest.kt", True),
    ],
)
def test_is_test_path_matches_segments_not_substrings(path: str, expected: bool) -> None:
    assert is_test_path(path) is expected


def _many_inits_memory(tmp_path: Path) -> Memory:
    repo = init_fixture_repo(tmp_path / "repo")
    files = {
        f"src/m{i}.py": (
            f"class C{i}:\n    def __init__(self):\n        self.load()\n\n"
            f"    def load(self):\n        return {i}\n"
        ).encode()
        for i in range(7)
    }
    files["src/keep.py"] = b"def keep():\n    return 1\n"
    commit_fixture_spec(repo, GitCommitSpec(message="C0", files=files))
    mem = Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo))
    mem.sync(SyncRequest(revision="HEAD", offline=True))
    return mem


@pytest.mark.parametrize(
    "pattern",
    [
        ".*" * 10 + "!",
        "a*" * 8 + "!",
        "(a|a)*b",
        "a{2,}b{3,}c+d",
        "a?" * 28 + "a" * 28,
        ".{0,30}" * 7 + "!",
        r"\w{0,60}" * 4 + "!",
        "a?b?c?",
        "(?:a|aa)" * 24 + "!",
        "(?:a|aa)" * 26 + "!",
        "(a|a)" * 22 + "!",
        "(?:a|a)" * 20 + r"\w*\w+!",
        "a|b|(c|d)|(e|f)",
    ],
)
def test_compile_safe_regex_rejects_backtracking(pattern: str) -> None:
    with pytest.raises(CodeContextError) as exc_info:
        compile_safe_regex(pattern)
    assert exc_info.value.code == "INVALID_REQUEST"


@pytest.mark.parametrize(
    "pattern",
    [
        ".*Handler",
        "^get_",
        "(foo|bar)",
        ".*foo.*",
        "process_.*",
        "(?:ab)+",
        "a*?b+c",
        "[*+]{1,3}x*",
        "x{3}y{2,2}z?",
        "^(get|set)_.*",
        "(foo|bar)Handler",
        "^get_|^set_",
    ],
)
def test_compile_safe_regex_keeps_ordinary_patterns(pattern: str) -> None:
    assert compile_safe_regex(pattern).pattern == pattern


def test_variable_length_query_respects_time_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with _many_inits_memory(tmp_path) as mem:
        monkeypatch.setattr(graph, "GRAPH_TOOL_MAX_SECONDS", 0.0)
        with pytest.raises(CodeContextError) as exc_info:
            mem.query_graph(QueryGraphRequest(query="MATCH (a)-[*1..6]->(b) RETURN count(b)"))
        assert exc_info.value.code == "INVALID_REQUEST"
        assert exc_info.value.message == "Graph query exceeded its time limit"


def test_search_graph_regex_respects_time_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from codecontext import store as store_module

    with _many_inits_memory(tmp_path) as mem:
        assert mem.search_graph(SearchGraphRequest(name_pattern="load")).total_matched == 7
        monkeypatch.setattr(graph, "GRAPH_TOOL_MAX_SECONDS", 0.0)
        with pytest.raises(CodeContextError) as exc_info:
            mem.search_graph(SearchGraphRequest(name_pattern="load"))
        assert exc_info.value.message == "Graph query exceeded its time limit"

        # Let the connection open past the deadline so the SQL itself is what gets interrupted.
        monkeypatch.setattr(store_module, "check_deadline", lambda: None)
        monkeypatch.setattr(store_module, "remaining_timeout", lambda limit: limit)
        with pytest.raises(CodeContextError) as exc_info:
            mem.search_graph(SearchGraphRequest(name_pattern="load"))
        assert exc_info.value.code == "INVALID_REQUEST"
        assert exc_info.value.message == "Graph query exceeded its time limit"
        assert "interrupted" in str(exc_info.value.__cause__.__cause__)


def test_detect_changes_uses_its_own_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert graph.DETECT_CHANGES_MAX_SECONDS == 60.0
    repo, c0 = _build_repo(tmp_path)
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(revision="HEAD", offline=True))
        monkeypatch.setattr(graph, "GRAPH_TOOL_MAX_SECONDS", 0.0)
        monkeypatch.setattr(graph, "DETECT_CHANGES_MAX_SECONDS", 1000.0)
        assert mem.detect_changes(DetectChangesRequest(since=c0)).changed_files

        monkeypatch.setattr(graph, "GRAPH_TOOL_MAX_SECONDS", 1000.0)
        monkeypatch.setattr(graph, "DETECT_CHANGES_MAX_SECONDS", 0.0)
        with pytest.raises(CodeContextError) as exc_info:
            mem.detect_changes(DetectChangesRequest(since=c0))
        assert exc_info.value.message == "Graph query exceeded its time limit"


def test_regex_sees_only_a_bounded_prefix_of_long_text(tmp_path: Path) -> None:
    slow = r"\w*\w+!"
    repo = init_fixture_repo(tmp_path / "repo")
    long_name = "a" * 4000
    commit_fixture_spec(
        repo,
        GitCommitSpec(message="C0", files={"src/m.py": f"def {long_name}():\n    return 1\n".encode()}),
    )
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(revision="HEAD", offline=True))
        started = time.perf_counter()
        assert mem.search_graph(SearchGraphRequest(name_pattern=slow)).total_matched == 0
        # `=~` on a relationship binding reads the raw value, so no GraphNode length cap applies.
        assert not graph._eval_where_atom({"r": {"method": long_name}}, f"r.method =~ '{slow}'")
        assert time.perf_counter() - started < 1.0


def test_graph_time_limit_keeps_tighter_outer_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    from codecontext.deadline import check_deadline, sync_deadline

    @graph._bounded_by_graph_time_limit
    def probe() -> None:
        check_deadline()

    with sync_deadline(0.0), pytest.raises(CodeContextError):
        probe()


def test_order_by_unreturned_column_is_rejected(tmp_path: Path) -> None:
    with _many_inits_memory(tmp_path) as mem:
        with pytest.raises(CodeContextError) as exc_info:
            mem.query_graph(
                QueryGraphRequest(query="MATCH (n) RETURN n.name ORDER BY n.degree DESC LIMIT 10")
            )
        assert exc_info.value.code == "INVALID_REQUEST"
        ordered = mem.query_graph(
            QueryGraphRequest(query="MATCH (n) RETURN n.name AS name ORDER BY name DESC LIMIT 1")
        )
        assert ordered.rows == [{"name": "load"}]


def test_query_without_order_stops_at_cap(tmp_path: Path) -> None:
    with _many_inits_memory(tmp_path) as mem:
        res = mem.query_graph(QueryGraphRequest(query="MATCH (n) RETURN n.name LIMIT 2"))
        assert res.row_count == 2
        assert res.truncated is True
        count = mem.query_graph(QueryGraphRequest(query="MATCH (n:Class) RETURN count(n) LIMIT 2"))
        assert count.rows == [{"count(n)": 7}]


def test_where_keeps_quoted_and(tmp_path: Path) -> None:
    with _many_inits_memory(tmp_path) as mem:
        res = mem.query_graph(
            QueryGraphRequest(query="MATCH (n) WHERE n.name = 'Load AND Save' RETURN n.name")
        )
        assert res.rows == []
        res = mem.query_graph(
            QueryGraphRequest(
                query="MATCH (n) WHERE n.name = 'x OR keep' OR n.name = \"keep\" RETURN n.name"
            )
        )
        assert res.rows == [{"n.name": "keep"}]


def test_trace_path_roots_capped(tmp_path: Path) -> None:
    with _many_inits_memory(tmp_path) as mem:
        everything = mem.search_graph(SearchGraphRequest(name_pattern="^__init__$"))
        assert everything.total_matched == 7
        res = mem.trace_path(TracePathRequest(function_name="__init__", limit=5))
        assert len(res.root_nodes) <= 5
