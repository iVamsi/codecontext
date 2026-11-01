import sqlite3
import time
from pathlib import Path

import pytest

from codecontext import InitRequest, Memory, SyncRequest, graph
from codecontext.errors import CodeContextError
from codecontext.git import PYTHON_DIFF_MAX_LINES
from codecontext.graph import (
    DetectChangesRequest,
    GetArchitectureRequest,
    GetCodeSnippetRequest,
    QueryGraphRequest,
    SearchGraphRequest,
    TracePathRequest,
    compile_safe_regex,
    find_scc_cycles,
    is_test_path,
)
from codecontext.store import Store
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


def test_detect_changes_worktree_file_too_large_to_diff(tmp_path: Path) -> None:
    padding = b"".join(b"PAD_%d = %d\n" % (i, i) for i in range(PYTHON_DIFF_MAX_LINES))
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(repo, GitCommitSpec(message="C0", files={"src/mod.py": MOD_V0 + padding}))
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(revision="HEAD", offline=True))
        (repo / "src" / "mod.py").write_bytes(MOD_V1 + padding)
        mem.configure_workspace(repo)
        res = mem.detect_changes(DetectChangesRequest())

    assert res.classifications == {
        "sig": "signature_modified",
        "body": "body_modified",
        "fresh": "added",
        "gone": "deleted",
    }


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
        "(?#[)" + "a*" * 12 + "!(?#])",
        "(?#[)(a*)*b(?#])",
        "(?#[)" + "(?:a|aa)" * 26 + "!(?#])",
        "(?x)#[\n" + "a*" * 12 + "!\n#]",
        "(?x:#[\n a*a*a* \n#])",
        "(?x-i:#[\n a*a*a* \n#])",
        "(?ax)a",
        "(?i-x:a)",
        "(?#comment)",
        "[]a]a*b*c*",
        "[^]a]a*b*c*",
        r"[\]a]a*b*c*",
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
        "(?i)handler",
        "(?:x)y*",
        "[]a](?:b|c)",
        r"[\]]x*",
        r"\(?x\)",
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
        # Either the progress handler or the regex function's own deadline check aborts it.
        assert str(exc_info.value.__cause__.__cause__) in {
            "interrupted",
            "user-defined function raised exception",
        }


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


def _trace_sql(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    statements: list[str] = []
    original = Store._open_raw_connection

    def traced(self: Store, *, for_write: bool) -> sqlite3.Connection:
        conn = original(self, for_write=for_write)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(Store, "_open_raw_connection", traced)
    return statements


def test_scc_long_chain_no_recursion_error() -> None:
    assert find_scc_cycles({f"c{i:05d}": {f"c{i + 1:05d}"} for i in range(5000)}) == []
    ring = {f"r{i:05d}": {f"r{(i + 1) % 5000:05d}"} for i in range(5000)}
    assert find_scc_cycles(ring) == [sorted(ring)]
    assert find_scc_cycles({"a": {"a", "b"}, "b": {"c"}, "c": {"b"}}) == [["a"], ["b", "c"]]


def test_detect_changes_same_name_in_two_files(tmp_path: Path) -> None:
    repo = init_fixture_repo(tmp_path / "repo")
    base = commit_fixture_spec(
        repo,
        GitCommitSpec(
            message="C0",
            files={
                "a.py": b"def main():\n    return 1\n\ndef other():\n    return 2\n",
                "b.py": b"def main():\n    return 1\n",
            },
        ),
    )
    commit_fixture_spec(
        repo,
        GitCommitSpec(
            message="C1",
            files={"a.py": b"def other():\n    return 2\n", "b.py": b"def main():\n    return 3\n"},
        ),
    )
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(revision="HEAD", offline=True))
        res = mem.detect_changes(DetectChangesRequest(since=base))

    assert ("b.py", "main") in {(s.file_path, s.qualified_name) for s in res.directly_modified_symbols}
    assert res.deleted_symbols == ["main"]


def test_get_code_snippet_does_not_scan_symbols(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, _ = _build_repo(tmp_path)
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(revision="HEAD", offline=True))
        traced_sql = _trace_sql(monkeypatch)
        res = mem.get_code_snippet(GetCodeSnippetRequest(qualified_name="body", include_neighbors=True))

    assert res.node.file_path == "src/mod.py"
    assert res.code == "def body(a):\n    return a + 1"
    assert {n.name for n in res.callers} == {"attest", "test_body"}
    assert res.node.inbound_degree == 2
    assert res.callees == []
    assert traced_sql
    assert not any("FROM symbols" in s and "WHERE" not in s for s in traced_sql)


def test_search_graph_degree_computed_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with _many_inits_memory(tmp_path) as mem:
        traced_sql = _trace_sql(monkeypatch)
        res = mem.search_graph(SearchGraphRequest(name_pattern="load", limit=3, offset=1))
        assert len(res.nodes) == 3
        assert res.total_matched == 7
        assert res.has_more is True
        assert sum("inbound_degree" in s for s in traced_sql) == 1

        past_end = mem.search_graph(SearchGraphRequest(name_pattern="load", offset=50))
        assert past_end.nodes == []
        assert past_end.total_matched == 7
        assert past_end.has_more is False


def test_graph_cache_sees_newly_synced_symbol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(repo, GitCommitSpec(message="C0", files={"m.py": b"def first():\n    return 1\n"}))
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(revision="HEAD", offline=True))
        assert mem.trace_path(TracePathRequest(function_name="first")).root_nodes
        traced_sql = _trace_sql(monkeypatch)
        assert mem.trace_path(TracePathRequest(function_name="first")).root_nodes
        assert not any("FROM symbols" in s for s in traced_sql)

        commit_fixture_spec(
            repo, GitCommitSpec(message="C1", files={"m.py": b"def second():\n    return 2\n"})
        )
        mem.sync(SyncRequest(revision="HEAD", offline=True))
        assert [n.name for n in mem.trace_path(TracePathRequest(function_name="second")).root_nodes] == [
            "second"
        ]
        assert mem.query_graph(QueryGraphRequest(query="MATCH (n) RETURN n.name")).rows == [
            {"n.name": "second"}
        ]


def test_graph_cache_cannot_be_mutated_by_callers(tmp_path: Path) -> None:
    with _many_inits_memory(tmp_path) as mem:
        nodes, edges, _ = graph._load_active_graph(mem._store)
        with pytest.raises(TypeError):
            nodes["x"] = next(iter(nodes.values()))  # type: ignore[index]
        with pytest.raises(AttributeError):
            edges.append(edges[0])  # type: ignore[attr-defined]


def test_long_symbol_names_do_not_break_graph_tools(tmp_path: Path) -> None:
    long_name = "f" * 3000
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(
        repo,
        GitCommitSpec(
            message="C0",
            files={
                "src/m.py": (
                    f"def {long_name}():\n    return 1\n\n"
                    f"def caller():\n    return {long_name}()\n"
                ).encode()
            },
        ),
    )
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(revision="HEAD", offline=True))
        traced = mem.trace_path(TracePathRequest(function_name="caller", direction="outbound"))
        assert {len(n.name) for n in traced.visited_nodes} == {6, 1000}
        rows = mem.query_graph(QueryGraphRequest(query="MATCH (n) RETURN n.name")).rows
        assert sorted(len(r["n.name"]) for r in rows) == [6, 1000]
        assert mem.get_architecture(GetArchitectureRequest()).total_symbols == 2
        found = mem.search_graph(SearchGraphRequest(name_pattern="^fff"))
        assert found.total_matched == 1
        assert len(found.nodes[0].qualified_name) == 2000


def test_search_graph_regex_udf_stops_at_the_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class SlowPattern:
        def search(self, value: str) -> None:
            time.sleep(0.2)
            return None

    with _many_inits_memory(tmp_path) as mem:
        monkeypatch.setattr(graph, "compile_safe_regex", lambda pattern: SlowPattern())
        monkeypatch.setattr(graph, "GRAPH_TOOL_MAX_SECONDS", 0.3)
        started = time.monotonic()
        with pytest.raises(CodeContextError) as exc_info:
            mem.search_graph(SearchGraphRequest(name_pattern="load"))
        elapsed = time.monotonic() - started
    assert exc_info.value.code == "INVALID_REQUEST"
    assert exc_info.value.message == "Graph query exceeded its time limit"
    assert elapsed < 1.0



def test_worktree_read_does_not_follow_a_parent_swapped_for_a_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    (root / "src").mkdir(parents=True)
    (root / "src" / "a.py").write_bytes(b"inside\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.py").write_bytes(b"secret\n")
    assert graph._read_worktree_file(root, b"src/a.py") == b"inside\n"

    real_resolve = Path.resolve

    def resolve_then_swap(self: Path, strict: bool = False) -> Path:
        # Swap the parent right after any path check, as a racing process could.
        result = real_resolve(self, strict)
        if not (root / "src").is_symlink():
            (root / "src").rename(tmp_path / "moved")
            (root / "src").symlink_to(outside)
        return result

    monkeypatch.setattr(Path, "resolve", resolve_then_swap)
    assert graph._read_worktree_file(root, b"src/a.py") is None
