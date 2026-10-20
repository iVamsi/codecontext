from pathlib import Path

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
from codecontext.mcp_server import create_mcp_server
from codecontext.syntax import parse_blob
from tests.fixtures.builders import (
    GitCommitSpec,
    commit_fixture_spec,
    init_fixture_repo,
)


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

        # 3. Verify ReDoS (nested quantifiers / backreferences) is rejected in search_graph and query_graph
        for redos_pat in ["(a+)+$", "([a-zA-Z]+)*", r"(a+)\1+"]:
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

        # 7. get_code_snippet with neighbors
        snip = mem.get_code_snippet(
            GetCodeSnippetRequest(qualified_name="process_order", include_neighbors=True)
        )
        assert "return save_record(order_id)" in snip.code
        assert [c.name for c in snip.callers] == ["handle_request"]
        assert [c.name for c in snip.callees] == ["save_record"]

    # 8. Verify MCP server with `include_graph_tools=True` exposes all 9 tools
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
            }
    finally:
        holder.close()
