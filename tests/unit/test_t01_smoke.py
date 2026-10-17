import sqlite3
import subprocess
import sys

import pytest
from pydantic import BaseModel

import codecontext
from codecontext._deps import (
    verify_git_capabilities,
    verify_runtime_dependencies,
    verify_sqlite_capabilities,
    verify_treesitter_capabilities,
)
from codecontext.errors import CodeContextError


def test_package_import_and_cli_help() -> None:
    assert hasattr(codecontext, "__version__")
    proc = subprocess.run(
        [sys.executable, "-m", "codecontext", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "codecontext" in proc.stdout


def test_sqlite_fts5_and_json_capabilities() -> None:
    verify_sqlite_capabilities()


def test_git_version_capability() -> None:
    version = verify_git_capabilities()
    assert version >= (2, 43, 0)
    with pytest.raises(CodeContextError) as exc_info:
        verify_git_capabilities("nonexistent-git-binary-xyz")
    assert exc_info.value.code == "DEPENDENCY_MISSING"


def test_treesitter_grammars_and_queries() -> None:
    info = verify_treesitter_capabilities()
    assert set(info["languages"]) == {"javascript", "python", "tsx", "typescript"}


async def test_mcp_sdk_in_memory_roundtrip() -> None:
    from mcp import Client
    from mcp.server import MCPServer

    server = MCPServer("codecontext-smoke")

    class EchoReq(BaseModel):
        query: str

    class EchoRes(BaseModel):
        echoed: str

    @server.tool()
    def echo(request: EchoReq) -> EchoRes:
        return EchoRes(echoed=request.query)

    async with Client(server) as client:
        tools = await client.list_tools()
        assert [t.name for t in tools.tools] == ["echo"]
        res = await client.call_tool("echo", {"request": {"query": "hello"}})
        assert not res.is_error
        assert res.structured_content == {"echoed": "hello"}


def test_runtime_dependencies_aggregate() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        verify_runtime_dependencies(conn=conn)
    finally:
        conn.close()
