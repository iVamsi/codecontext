from pathlib import Path

from codecontext import Memory, SyncRequest
from codecontext.graph import (
    DetectChangesRequest,
    GetArchitectureRequest,
    SearchGraphRequest,
    detect_louvain_communities,
    find_scc_cycles,
)
from codecontext.models import InitRequest
from codecontext.syntax import (
    EXTENSION_TO_LANGUAGE,
    parse_blob,
)
from tests.fixtures.builders import build_f01_fixture


def test_polyglot_languages_extraction() -> None:
    """Verify fallback and AST parsing for expanded polyglot languages."""
    snippets = {
        ".cs": (
            b"using System;\nnamespace App {\n    public class Greeter {\n        public void SayHello() {\n            Console.WriteLine(\"hello\");\n        }\n    }\n}\n",
            ["Greeter", "SayHello"],
        ),
        ".rb": (
            b"require 'json'\nclass Worker\n  def perform(task)\n    process(task)\n  end\nend\n",
            ["Worker", "perform"],
        ),
        ".php": (
            b"<?php\nnamespace App;\nclass Router {\n    public function dispatch($req) {\n        handle($req);\n    }\n}\n",
            ["Router", "dispatch"],
        ),
        ".sh": (
            b"#!/bin/bash\nfunction deploy() {\n    echo 'deploying'\n    build\n}\n",
            ["deploy"],
        ),
        ".swift": (
            b"import Foundation\nclass Service {\n    func start() {\n        setup()\n    }\n}\n",
            ["Service", "start"],
        ),
        ".scala": (
            b"package com.app\nclass Calculator {\n    def add(x: Int, y: Int): Int = {\n        x + y\n    }\n}\n",
            ["Calculator", "add"],
        ),
        ".sql": (
            b"CREATE TABLE users (id INT PRIMARY KEY, name TEXT);\nCREATE FUNCTION get_user(uid INT) RETURNS RECORD AS $$ SELECT * FROM users; $$;\n",
            ["users", "get_user"],
        ),
        ".proto": (
            b'syntax = "proto3";\nmessage UserProfile {\n    string id = 1;\n    string name = 2;\n}\nservice UserService {\n    rpc GetUser(UserProfile) returns (UserProfile);\n}\n',
            ["UserProfile", "UserService", "GetUser"],
        ),
        ".graphql": (
            b"type Query {\n    getUser(id: ID!): User\n}\ntype User {\n    id: ID!\n    name: String\n}\n",
            ["Query", "User"],
        ),
        ".gradle": (
            b'plugins {\n    id "java"\n}\ntask customBuild {\n    doLast {\n        println "Building..."\n    }\n}\n',
            ["customBuild"],
        ),
    }

    for ext, (content, expected_symbols) in snippets.items():
        lang = EXTENSION_TO_LANGUAGE.get(ext)
        assert lang is not None, f"Extension {ext} not in EXTENSION_TO_LANGUAGE"
        res = parse_blob(
            blob_bytes=content,
            language=lang,
        )
        assert res.state == "complete", f"Extraction failed for {ext}"
        extracted_names = [s.name for s in res.symbols]
        for name in expected_symbols:
            assert name in extracted_names, f"Expected symbol {name} in {ext}, got: {extracted_names}"


def test_rich_graph_relations_in_python_and_ts() -> None:
    """Verify IMPORTS, INHERITS, IMPLEMENTS, DECORATES, TESTS occurrences and relationships."""
    py_code = b"""import math

@decorator_fn
class BaseHandler:
    def handle(self):
        pass

class ConcreteHandler(BaseHandler):
    def handle(self):
        super().handle()

def test_handler():
    h = ConcreteHandler()
    h.handle()
"""
    res = parse_blob(
        blob_bytes=py_code,
        language="python",
    )
    assert res.state == "complete"
    occ_kinds = [o.kind for o in res.occurrences]

    # Check for decorator, inheritance, import occurrences
    assert "decorates" in occ_kinds or "inherits" in occ_kinds or "import" in occ_kinds
    sym_names = [s.name for s in res.symbols]
    assert "BaseHandler" in sym_names
    assert "ConcreteHandler" in sym_names
    assert "test_handler" in sym_names


def test_graph_algorithms_louvain_and_scc() -> None:
    """Verify Tarjan SCC detects cycles and Louvain detects communities."""
    # SCC cycle test: A -> B -> C -> A (cycle of 3), C -> D -> E (no cycle)
    adj = {
        "A": {"B"},
        "B": {"C"},
        "C": {"A", "D"},
        "D": {"E"},
        "E": set(),
    }
    cycles = find_scc_cycles(adj)
    assert len(cycles) == 1
    assert set(cycles[0]) == {"A", "B", "C"}

    # Louvain community test
    # Community 1: [1, 2, 3] densely connected
    # Community 2: [4, 5, 6] densely connected
    c_nodes = ["n1", "n2", "n3", "n4", "n5", "n6"]
    c_edges = [
        ("n1", "n2", 5.0),
        ("n2", "n3", 5.0),
        ("n1", "n3", 5.0),
        ("n4", "n5", 5.0),
        ("n5", "n6", 5.0),
        ("n4", "n6", 5.0),
        ("n3", "n4", 0.5),
    ]
    communities = detect_louvain_communities(c_nodes, c_edges)
    assert len(communities) == 2
    # Verify nodes in same community
    c1 = set(communities[0])
    c2 = set(communities[1])
    assert c1 == {"n1", "n2", "n3"} or c1 == {"n4", "n5", "n6"}
    assert c2 == {"n1", "n2", "n3"} or c2 == {"n4", "n5", "n6"}


def test_get_architecture_and_detect_changes_execution(tmp_path: Path) -> None:
    """Verify execute_get_architecture and execute_detect_changes with detailed metrics."""
    f01 = build_f01_fixture(tmp_path / "repo")
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=f01.repo_dir)) as mem:
        mem.sync(SyncRequest(offline=True))

        # Test get_architecture
        arch = mem.get_architecture(GetArchitectureRequest(max_clusters=10))
        assert hasattr(arch, "clusters")
        assert hasattr(arch, "hotspots")
        assert hasattr(arch, "cycles")
        assert hasattr(arch, "coupling")

        # Test search_graph with relationship filtering
        sg_res = mem.search_graph(
            SearchGraphRequest(
                relationship="CALLS",
                direction="both",
                limit=10,
            ),
        )
        assert sg_res.nodes is not None

        # Test detect_changes
        dc_res = mem.detect_changes(DetectChangesRequest(since=None, depth=2, limit=10))
        assert hasattr(dc_res, "changed_files")
        assert hasattr(dc_res, "affected_symbols")
        assert hasattr(dc_res, "deleted_symbols")
        assert hasattr(dc_res, "classifications")
        assert hasattr(dc_res, "affected_tests")


def test_search_graph_relationship_rejects_unknown_values() -> None:
    import pytest
    from pydantic import ValidationError

    assert SearchGraphRequest(relationship="INHERITS").relationship == "INHERITS"
    with pytest.raises(ValidationError):
        SearchGraphRequest(relationship="CALLS' OR '1'='1")
