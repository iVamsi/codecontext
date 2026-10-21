import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp import Client

from codecontext import CodeContextError, InitRequest, Memory, SyncRequest
from codecontext.graph import (
    DetectChangesRequest,
    GetArchitectureRequest,
    GetCodeSnippetRequest,
    QueryGraphRequest,
    SearchGraphRequest,
    TracePathRequest,
)
from codecontext.mcp_server import _DynamicMemoryHolder, create_mcp_server
from codecontext.syntax import parse_blob
from tests.fixtures.builders import (
    GitCommitSpec,
    commit_fixture_spec,
    init_fixture_repo,
    run_git_fixture,
)


def test_kotlin_unqualified_calls_use_nearest_scope(tmp_path: Path) -> None:
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(repo, GitCommitSpec(message="Kotlin lexical scopes", files={
        "Main.kt": b"""fun save() {}
fun topLevel() { save() }
fun unrelated() { privateToB() }
class A {
 fun save() {}
 fun save(value: Int) {}
 fun run() { save() }
 fun local() {
  fun save() {}
  save()
 }
}
class B {
 fun save() {}
 fun privateToB() {}
 fun run() { save() }
}
class C {
 fun run() { save() }
}
""",
    }))
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True))
        for caller, expected in {
            "A.run": {"A.save"},
            "B.run": {"B.save"},
            "A.local": {"A.local.save"},
            "C.run": {"save"},
            "topLevel": {"save"},
            "unrelated": set(),
        }.items():
            trace = mem.trace_path(TracePathRequest(function_name=caller, direction="outbound", depth=1))
            assert {e.to_name for e in trace.edges} == expected, caller
        trace = mem.trace_path(TracePathRequest(function_name="A.run", direction="outbound", depth=1))
        assert len(trace.edges) == 2  # Both overloads remain candidates.


def test_kotlin_receiver_does_not_link_unrelated_same_file_method(tmp_path: Path) -> None:
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(repo, GitCommitSpec(message="Kotlin receivers", files={
        "Main.kt": b"class A {\n fun save() {}\n}\nclass B {\n fun save() {}\n}\n"
        b"fun run(b: B) { b.save() }\nfun unknown(x: Any) { x.save() }\n",
    }))
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True))
        trace = mem.trace_path(TracePathRequest(function_name="run", direction="outbound"))
        assert {e.to_name for e in trace.edges} == {"B.save"}
        assert not mem.trace_path(TracePathRequest(function_name="unknown", direction="outbound")).edges


def test_kotlin_local_receiver_shadows_constructor_type(tmp_path: Path) -> None:
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(repo, GitCommitSpec(message="Local receiver", files={
        "Main.kt": b"class A {\n fun save() {}\n}\nclass B {\n fun save() {}\n}\n"
        b"class Worker(val b: A) {\n fun run() {\n val b: B = B()\n b.save()\n }\n}\n",
    }))
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True))
        trace = mem.trace_path(TracePathRequest(function_name="Worker.run", direction="outbound", depth=1))
        assert {e.to_name for e in trace.edges if e.to_name.endswith(".save")} == {"B.save"}


@pytest.mark.parametrize("object_format", ["sha1", "sha256"])
def test_offline_sync_indexes_uncommitted_kotlin_rename(tmp_path: Path, object_format: str) -> None:
    repo = init_fixture_repo(tmp_path / "repo", object_format=object_format)
    committed = commit_fixture_spec(repo, GitCommitSpec(message="Original", files={
        "Main.kt": b"fun oldName() {}\nfun caller() { oldName() }\n",
    }))
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True))
        (repo / "Main.kt").write_text("fun newName() {}\nfun caller() { newName() }\n")
        source_index = (repo / ".git/index").read_bytes()
        mem.sync(SyncRequest(offline=True))
        snapshot = mem.status().revision
        assert snapshot != committed
        assert (repo / ".git/index").read_bytes() == source_index
        assert not mem.search_graph(SearchGraphRequest(name_pattern="^oldName$")).nodes
        assert len(mem.search_graph(SearchGraphRequest(name_pattern="^newName$")).nodes) == 1
        assert "newName()" in mem.get_code_snippet(GetCodeSnippetRequest(qualified_name="caller")).code
        assert run_git_fixture(repo, ["rev-parse", "HEAD"]).decode().strip() == committed
        assert mem.search({"query": "newName", "revision": "HEAD", "kind": "code"}).hits
        mem.sync(SyncRequest(offline=True))
        assert mem.status().revision == snapshot
        mem.sync(SyncRequest(revision=committed, offline=True))
        assert len(mem.search_graph(SearchGraphRequest(name_pattern="^oldName$")).nodes) == 1


async def test_mcp_routes_to_initialized_nested_submodule(tmp_path: Path) -> None:
    library = init_fixture_repo(tmp_path / "library")
    commit_fixture_spec(library, GitCommitSpec(message="Library", files={
        "Library.kt": b"fun libraryOnly() {}\n",
    }))
    app = init_fixture_repo(tmp_path / "app")
    commit_fixture_spec(app, GitCommitSpec(message="App", files={"App.kt": b"fun appOnly() {}\n"}))
    run_git_fixture(app, ["-c", "protocol.file.allow=always", "submodule", "add", str(library), "nested/submodulelib"])
    run_git_fixture(app, ["commit", "-am", "Add library"])
    server, holder = create_mcp_server(workspace_path=app)
    try:
        async with Client(server) as client:
            listed = await client.call_tool("list_repositories", {})
            assert not listed.is_error
            repos = json.loads(listed.content[0].text)
            assert {r["path"] for r in repos["repositories"]} == {
                str(app.resolve()), str((app / "nested/submodulelib").resolve()),
            }
            result = await client.call_tool("search_graph", {
                "request": {"name_pattern": "libraryOnly"}, "repository": "nested/submodulelib",
            })
            assert not result.is_error
            assert result.structured_content["nodes"][0]["file_path"] == "Library.kt"
            root_result = await client.call_tool("search_graph", {"request": {"name_pattern": "appOnly"}})
            assert root_result.structured_content["total_matched"] == 1
            denied = await client.call_tool("search_graph", {
                "request": {}, "repository": str(library),
            })
            assert denied.is_error
            assert not (library / ".codecontext").exists()
    finally:
        holder.close()


async def test_mcp_routes_multiple_client_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    roots = [init_fixture_repo(tmp_path / name) for name in ("one", "two")]
    for i, root in enumerate(roots):
        commit_fixture_spec(root, GitCommitSpec(message="Root", files={
            "main.py": f"def root_{i}(): pass\n".encode(),
        }))
    monkeypatch.chdir(tmp_path)
    holder = _DynamicMemoryHolder(None)

    async def list_roots():
        return SimpleNamespace(roots=[SimpleNamespace(uri=p.as_uri()) for p in roots])

    ctx = SimpleNamespace(session=SimpleNamespace(list_roots=list_roots))
    try:
        two = await holder.get_memory(ctx, str(roots[1]))
        one = await holder.get_memory(None)
        assert one is not two
        assert one.search_graph({"name_pattern": "root_0"}).total_matched == 1
        assert two.search_graph({"name_pattern": "root_1"}).total_matched == 1
        assert await holder.get_memory(ctx, str(roots[0])) is one
        assert await holder.get_memory(None) is one
    finally:
        holder.close()


def test_search_graph_filters_and_pages_without_loading_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from codecontext import graph

    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(repo, GitCommitSpec(message="Graph", files={
        "main.py": b"class A:\n    def one(self): pass\n    def two(self): self.one()\ndef test_skip(): pass\n",
    }))
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True))

        def forbidden(*args):
            pytest.fail("search_graph loaded the full graph")

        monkeypatch.setattr(graph, "_load_active_graph", forbidden)
        contains = mem.search_graph({"relationship": "CONTAINS", "direction": "inbound", "min_degree": 1})
        assert {n.qualified_name for n in contains.nodes} == {"A.one", "A.two"}
        calls = mem.search_graph({"label": "Function", "direction": "inbound", "min_degree": 1})
        assert [n.qualified_name for n in calls.nodes] == ["A.one"]
        page = mem.search_graph({"label": "method", "limit": 1, "offset": 1})
        assert page.total_matched == 2 and not page.has_more
        assert page.nodes[0].qualified_name == "A.two"
        empty = mem.search_graph({"label": "method", "offset": 99})
        assert empty.total_matched == 2 and not empty.nodes and not empty.has_more
        assert not mem.search_graph({"file_pattern": "nomatch"}).nodes
        assert mem.search_graph({"exclude_entry_points": True}).total_matched == 3


def test_sync_invalidates_obsolete_extractor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from codecontext import syntax

    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(repo, GitCommitSpec(message="Kotlin", files={"Main.kt": b"fun example() {}\n"}))
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True))
        old_id = mem.search_graph({}).nodes[0].id
        monkeypatch.setitem(syntax.EXTRACTOR_FINGERPRINTS, "kotlin", "test-next-extractor")
        mem.sync(SyncRequest(offline=True))
        assert mem.search_graph({}).nodes[0].id != old_id


def test_workspace_snapshot_additions_deletions_and_unsafe_files(tmp_path: Path) -> None:
    repo = init_fixture_repo(tmp_path / "repo")
    committed = commit_fixture_spec(repo, GitCommitSpec(message="Original", files={
        "deleted.py": b"def removed(): pass\n",
        "link.py": b"def formerly_regular(): pass\n",
        "large.py": b"def formerly_small(): pass\n",
    }))
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True))
        (repo / "deleted.py").unlink()
        (repo / "link.py").unlink()
        external = tmp_path / "outside.py"
        external.write_text("def outside_secret(): pass\n")
        (repo / "link.py").symlink_to(external)
        (repo / "large.py").write_bytes(b"#" * (2 * 1024 * 1024 + 1))
        (repo / "new.py").write_text("def added(): pass\n")
        (repo / "untracked.py").write_text("def untracked(): pass\n")
        run_git_fixture(repo, ["add", "new.py"])
        staged = (repo / ".git/index").read_bytes()
        mem.sync(SyncRequest(offline=True))
        assert [n.name for n in mem.search_graph({}).nodes] == ["added"]
        assert (repo / ".git/index").read_bytes() == staged
        assert run_git_fixture(repo, ["rev-parse", "HEAD"]).decode().strip() == committed
        with mem._store.connect_read() as conn:
            assert conn.execute("SELECT COUNT(*) FROM coverage_gaps WHERE code = 'BLOB_TOO_LARGE' AND active = 1").fetchone()[0]


def test_go_rust_c_cpp_and_gradle_ast_extraction() -> None:
    """Verify Tree-sitter AST symbol and call extraction for Go, Rust, C, C++, and Gradle Kotlin DSL."""
    # 1. Go
    go_code = (
        b"package main\n\n"
        b'import "fmt"\n\n'
        b"type Server struct {\n    port int\n}\n\n"
        b"func (s *Server) Start() error {\n"
        b"    return initSocket()\n"
        b"}\n\n"
        b"func initSocket() error {\n"
        b"    return nil\n"
        b"}\n"
    )
    go_res = parse_blob(go_code, "go")
    assert go_res.state == "complete"
    assert {"Server", "Server.Start", "initSocket"} <= {s.qualified_name for s in go_res.symbols}
    assert any(r.relation_kind == "calls_candidate" for r in go_res.relations)

    # 2. Rust
    rs_code = (
        b"use std::io;\n\n"
        b"pub struct Worker;\n\n"
        b"impl Worker {\n"
        b"    pub fn execute(&self) {\n"
        b"        run_step();\n"
        b"    }\n"
        b"}\n\n"
        b"pub fn run_step() {}\n"
    )
    rs_res = parse_blob(rs_code, "rust")
    assert rs_res.state == "complete"
    assert {"Worker", "Worker.execute", "run_step"} <= {s.qualified_name for s in rs_res.symbols}
    assert any(r.relation_kind == "calls_candidate" for r in rs_res.relations)

    # 3. C
    c_code = (
        b"#include <stdio.h>\n\n"
        b"struct Packet {\n    int len;\n};\n\n"
        b"int process_packet(int x) {\n"
        b"    return checksum(x);\n"
        b"}\n\n"
        b"int checksum(int x) {\n"
        b"    return x ^ 0xff;\n"
        b"}\n"
    )
    c_res = parse_blob(c_code, "c")
    assert c_res.state == "complete"
    assert {"Packet", "process_packet", "checksum"} <= {s.qualified_name for s in c_res.symbols}
    assert any(r.relation_kind == "calls_candidate" for r in c_res.relations)

    # 4. C++
    cpp_code = (
        b"#include <vector>\n\n"
        b"namespace engine {\n"
        b"class Pipeline {\n"
        b"public:\n"
        b"    int run(int v) {\n"
        b"        return step(v);\n"
        b"    }\n"
        b"    int step(int v) {\n"
        b"        return v + 1;\n"
        b"    }\n"
        b"};\n"
        b"}\n"
    )
    cpp_res = parse_blob(cpp_code, "cpp")
    assert cpp_res.state == "complete"
    assert {"engine", "engine.Pipeline", "engine.Pipeline.run", "engine.Pipeline.step"} <= {
        s.qualified_name for s in cpp_res.symbols
    }

    # 5. Gradle Kotlin DSL (build.gradle.kts)
    gradle_kts = (
        b"plugins {\n"
        b'    id("com.android.application")\n'
        b"}\n\n"
        b"android {\n"
        b'    namespace = "com.example.app"\n'
        b"    defaultConfig {\n"
        b"        minSdk = 26\n"
        b"    }\n"
        b"}\n\n"
        b"dependencies {\n"
        b'    implementation(project(":core:network"))\n'
        b"}\n"
    )
    gradle_res = parse_blob(gradle_kts, "kotlin")
    assert gradle_res.state == "complete"
    assert {"plugins", "android", "android.defaultConfig", "dependencies"} <= {
        s.qualified_name for s in gradle_res.symbols
    }


def test_import_and_type_disambiguated_call_resolution_and_strict_cypher(tmp_path: Path) -> None:
    """Verify both audit findings are resolved:
    1. Importing `a.save` (`from a import save`) produces a call edge ONLY to `a.py`'s `save`,
       NEVER to `b.py`'s unrelated `save`.
    2. Numeric Cypher WHERE conditions (`WHERE f.inbound_degree > 999`) are evaluated accurately,
       and invalid/unsupported WHERE conditions raise `INVALID_REQUEST` instead of being silently ignored.
    """
    repo_dir = init_fixture_repo(tmp_path / "disambig_repo")
    commit_fixture_spec(
        repo_dir,
        GitCommitSpec(
            message="C0: modules a, b, and caller",
            files={
                "src/a.py": b"def save(item: str) -> bool:\n    return True\n",
                "src/b.py": b"def save(item: str) -> bool:\n    return False\n",
                "src/caller.py": (
                    b"from src.a import save\n\n"
                    b"def run_workflow(x: str) -> bool:\n"
                    b"    return save(x)\n"
                ),
            },
        ),
    )

    store_dir = tmp_path / "disambig_store"
    with Memory.initialize(store_dir, InitRequest(repo_path=repo_dir)) as mem:
        mem.sync(SyncRequest(revision="HEAD", offline=True))

        # 1. Verify `run_workflow` links ONLY to `src/a.py`'s `save` and NOT `src/b.py`'s `save`!
        cypher_calls = mem.query_graph(
            QueryGraphRequest(
                query="MATCH (caller)-[r:CALLS]->(callee) WHERE caller.name = 'run_workflow' RETURN callee.name, callee.file_path"
            )
        )
        assert cypher_calls.row_count == 1
        assert cypher_calls.rows[0]["callee.file_path"] == "src/a.py"

        # Verify `src/b.py`'s `save` has inbound_degree == 0
        b_save = mem.query_graph(
            QueryGraphRequest(
                query="MATCH (f:Function) WHERE f.file_path = 'src/b.py' RETURN f.name, f.inbound_degree"
            )
        )
        assert b_save.row_count == 1
        assert b_save.rows[0]["f.inbound_degree"] == 0

        # 2. Verify numeric Cypher WHERE (`WHERE f.inbound_degree > 999`) returns 0 rows!
        high_deg = mem.query_graph(
            QueryGraphRequest(
                query="MATCH (f:Function) WHERE f.inbound_degree > 999 RETURN f.name"
            )
        )
        assert high_deg.row_count == 0
        assert high_deg.rows == []

        # Verify `WHERE f.inbound_degree >= 1` returns only `src/a.py`'s `save`
        called_funcs = mem.query_graph(
            QueryGraphRequest(
                query="MATCH (f:Function) WHERE f.inbound_degree >= 1 RETURN f.name, f.file_path"
            )
        )
        assert called_funcs.row_count == 1
        assert called_funcs.rows[0]["f.file_path"] == "src/a.py"

        # Verify malformed / unknown WHERE operator or property raises INVALID_REQUEST
        with pytest.raises(CodeContextError) as exc_info:
            mem.query_graph(
                QueryGraphRequest(
                    query="MATCH (f:Function) WHERE f.nonexistent_prop > 1 RETURN f.name"
                )
            )
        assert exc_info.value.code == "INVALID_REQUEST"

        for length in ("*", "*1..", "*0..2", "*3..2", "*1..7"):
            with pytest.raises(CodeContextError) as exc_info:
                mem.query_graph({"query": f"MATCH (a)-[r:CALLS{length}]->(b) RETURN b.name"})
            assert exc_info.value.code == "INVALID_REQUEST"

        # 3. Verify ReDoS (nested quantifiers / backreferences) is rejected in search_graph and query_graph
        for redos_pat in ["(a+)+$", "([a-zA-Z]+)*", r"(a+)\1+", "(a|aa)+$", "((a|aa))+$", "(((a|aa)))+$"]:
            with pytest.raises(CodeContextError) as exc_info:
                mem.search_graph(SearchGraphRequest(name_pattern=redos_pat))
            assert exc_info.value.code == "INVALID_REQUEST"

            with pytest.raises(CodeContextError) as exc_info:
                mem.query_graph(
                    QueryGraphRequest(
                        query=f"MATCH (f:Function) WHERE f.name =~ '{redos_pat}' RETURN f.name"
                    )
                )
            assert exc_info.value.code == "INVALID_REQUEST"


async def test_search_graph_trace_path_architecture_detect_changes_and_snippet(
    tmp_path: Path,
) -> None:
    """Verify `search_graph`, `trace_path`, `query_graph` (variable-length paths & COUNT),
    `get_architecture`, `detect_changes`, and `get_code_snippet` across SDK and MCP.
    """
    repo_dir = init_fixture_repo(tmp_path / "graph_repo")
    c0_oid = commit_fixture_spec(
        repo_dir,
        GitCommitSpec(
            message="C0: Multi-file call chain and dead function",
            files={
                "src/controller.py": (
                    b"from src.service import process_order\n\n"
                    b"def handle_request(req_id: int) -> bool:\n"
                    b"    return process_order(req_id)\n\n"
                    b"def unused_dead_helper() -> int:\n"
                    b"    return 0\n"
                ),
                "src/service.py": (
                    b"from src.db import save_record\n\n"
                    b"def process_order(order_id: int) -> bool:\n"
                    b"    return save_record(order_id)\n"
                ),
                "src/db.py": (
                    b"def save_record(rec_id: int) -> bool:\n"
                    b"    return rec_id > 0\n"
                ),
            },
        ),
    )
    # C1 modifies `src/db.py` (`save_record`) so we can test `detect_changes` impact analysis
    commit_fixture_spec(
        repo_dir,
        GitCommitSpec(
            message="C1: Update save_record validation",
            files={
                "src/db.py": (
                    b"def save_record(rec_id: int) -> bool:\n"
                    b"    return rec_id >= 1\n"
                ),
            },
        ),
    )

    store_dir = tmp_path / "graph_store"
    with Memory.initialize(store_dir, InitRequest(repo_path=repo_dir)) as mem:
        mem.sync(SyncRequest(revision="HEAD", offline=True))

        # 1. search_graph by regex pattern
        sg_res = mem.search_graph(SearchGraphRequest(name_pattern="process_.*"))
        assert sg_res.total_matched == 1
        assert sg_res.nodes[0].name == "process_order"
        assert sg_res.nodes[0].inbound_degree == 1
        assert sg_res.nodes[0].outbound_degree == 1

        # 2. Dead code detection
        dead_res = mem.search_graph(
            SearchGraphRequest(
                max_degree=0,
                direction="inbound",
                exclude_entry_points=True,
            )
        )
        assert "unused_dead_helper" in {n.name for n in dead_res.nodes}

        # 3. Multi-hop trace_path (handle_request -> process_order -> save_record)
        tp_res = mem.trace_path(
            TracePathRequest(
                function_name="handle_request",
                direction="outbound",
                depth=2,
            )
        )
        edge_pairs = {(e.from_name, e.to_name, e.depth) for e in tp_res.edges}
        assert ("handle_request", "process_order", 1) in edge_pairs
        assert ("process_order", "save_record", 2) in edge_pairs

        # 4. Cypher variable-length path `*1..2` and `count(f)`
        var_path = mem.query_graph(
            QueryGraphRequest(
                query="MATCH (a)-[r:CALLS*1..2]->(b) WHERE a.name = 'handle_request' RETURN b.name, r.depth ORDER BY r.depth ASC"
            )
        )
        assert var_path.row_count == 2
        assert [r["b.name"] for r in var_path.rows] == ["process_order", "save_record"]

        cnt_res = mem.query_graph(
            QueryGraphRequest(
                query="MATCH (f:Function) RETURN count(f) AS total_funcs"
            )
        )
        assert cnt_res.rows == [{"total_funcs": 4}]

        # 5. get_architecture
        arch = mem.get_architecture(GetArchitectureRequest())
        assert arch.total_files == 3
        assert arch.total_symbols == 4
        assert arch.total_call_edges == 2
        assert len(arch.clusters) == 1

        # 6. detect_changes since C0 -> `save_record` modified, impacts `process_order` (depth 1) and `handle_request` (depth 2)
        dc_res = mem.detect_changes(DetectChangesRequest(since=c0_oid, depth=2))
        assert dc_res.changed_files == ["src/db.py"]
        assert [s.name for s in dc_res.directly_modified_symbols] == ["save_record"]
        assert {s.name for s in dc_res.affected_callers} == {"process_order", "handle_request"}

        same = mem.detect_changes(DetectChangesRequest(since="HEAD"))
        assert same.changed_files == []

        (repo_dir / "src" / "controller.py").write_bytes(
            (repo_dir / "src" / "controller.py").read_bytes() + b"\n"
        )
        (repo_dir / "src" / "service.py").write_bytes(
            (repo_dir / "src" / "service.py").read_bytes() + b"\n"
        )
        run_git_fixture(repo_dir, ["add", "--", "src/service.py"])
        mem.configure_workspace(repo_dir)
        dirty = mem.detect_changes(DetectChangesRequest())
        assert dirty.changed_files == ["src/controller.py", "src/service.py"]

        # 7. get_code_snippet with neighbors
        snip = mem.get_code_snippet(
            GetCodeSnippetRequest(qualified_name="process_order", include_neighbors=True)
        )
        assert "return save_record(order_id)" in snip.code
        assert [c.name for c in snip.callers] == ["handle_request"]
        assert [c.name for c in snip.callees] == ["save_record"]

    # 8. Verify MCP server with `include_graph_tools=True` exposes all 10 tools
    mcp_srv, holder = create_mcp_server(store_dir, include_graph_tools=True)
    try:
        async with Client(mcp_srv) as client:
            tools = await client.list_tools()
            assert {t.name for t in tools.tools} == {
                "search",
                "get",
                "context",
                "search_graph",
                "trace_path",
                "query_graph",
                "get_architecture",
                "detect_changes",
                "get_code_snippet",
                "list_repositories",
            }
    finally:
        holder.close()
