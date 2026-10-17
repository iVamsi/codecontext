from pathlib import Path

from mcp import Client

from codecontext import InitRequest, Memory, SyncRequest
from codecontext.graph import (
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


def test_go_rust_c_cpp_ast_extraction() -> None:
    """Verify Tree-sitter AST symbol and call extraction for Go, Rust, C, and C++."""
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


async def test_search_graph_trace_path_and_cypher_query_graph(tmp_path: Path) -> None:
    """Verify cross-file call graph resolution, `search_graph`, `trace_path`, and Cypher `query_graph`."""
    repo_dir = init_fixture_repo(tmp_path / "graph_repo")
    commit_fixture_spec(
        repo_dir,
        GitCommitSpec(
            message="Multi-file call chain and dead function",
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

    store_dir = tmp_path / "graph_store"
    with Memory.initialize(store_dir, InitRequest(repo_path=repo_dir)) as mem:
        mem.sync(SyncRequest(revision="HEAD", offline=True))

        # 1. search_graph by regex pattern
        sg_res = mem.search_graph(SearchGraphRequest(name_pattern="process_.*"))
        assert sg_res.total_matched == 1
        assert sg_res.nodes[0].name == "process_order"
        assert sg_res.nodes[0].inbound_degree == 1
        assert sg_res.nodes[0].outbound_degree == 1

        # 2. Dead code detection via search_graph(max_degree=0, exclude_entry_points=True)
        dead_res = mem.search_graph(
            SearchGraphRequest(
                max_degree=0,
                direction="inbound",
                exclude_entry_points=True,
            )
        )
        dead_names = {n.name for n in dead_res.nodes}
        assert "unused_dead_helper" in dead_names

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

        # 4. Cypher query_graph over edges and nodes
        cypher_edges = mem.query_graph(
            QueryGraphRequest(
                query="MATCH (a)-[r:CALLS]->(b) WHERE a.name = 'handle_request' RETURN a.name, b.name, b.file_path LIMIT 10"
            )
        )
        assert cypher_edges.row_count == 1
        assert cypher_edges.rows[0]["a.name"] == "handle_request"
        assert cypher_edges.rows[0]["b.name"] == "process_order"
        assert cypher_edges.rows[0]["b.file_path"] == "src/service.py"

        cypher_nodes = mem.query_graph(
            QueryGraphRequest(
                query="MATCH (f:Function) WHERE f.name =~ '.*order.*' RETURN f.name, f.file_path"
            )
        )
        assert cypher_nodes.row_count == 1
        assert cypher_nodes.rows[0]["f.name"] == "process_order"

    # 5. Verify MCP server with `include_graph_tools=True` exposes all 6 tools
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
            }
            c_res = await client.call_tool(
                "query_graph",
                {
                    "request": {
                        "query": "MATCH (a)-[r:CALLS]->(b) RETURN a.name, b.name LIMIT 5"
                    }
                },
            )
            assert not c_res.is_error
            assert c_res.structured_content["row_count"] == 2
    finally:
        holder.close()
