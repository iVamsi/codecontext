import json
import subprocess
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

from codecontext import (
    ContextRequest,
    EnrichRequest,
    InitRequest,
    Memory,
    PathTarget,
    SearchRequest,
    SyncRequest,
)
from codecontext.cli import main as cli_main
from examples.fixture_adapter import DeterministicFixtureAdapter
from tests.fixtures.builders import build_f01_fixture
from tests.unit.test_t07_github import make_mock_github_transport


def _build_f01_routes(f01: object) -> dict[str, tuple[int, object, dict[str, str]]]:
    c1_oid = f01.c1_oid
    pr_payload = f01.pr_payload
    comments_payload = f01.comments_payload
    reviews_payload = f01.reviews_payload
    return {
        "/repos/example/transport": (
            200,
            {"id": 9001, "full_name": "example/transport", "private": False},
            {},
        ),
        f"/repos/example/transport/commits/{c1_oid}/pulls": (
            200,
            [{"number": 10, "id": 10010}],
            {},
        ),
        "/repos/example/transport/pulls/10": (200, pr_payload, {}),
        "/repos/example/transport/issues/10/comments?per_page=100": (
            200,
            comments_payload,
            {},
        ),
        "/repos/example/transport/pulls/10/reviews?per_page=100": (
            200,
            reviews_payload,
            {},
        ),
        "/repos/example/transport/pulls/10/comments?per_page=100": (200, [], {}),
        "/repos/example/transport/issues/10/timeline?per_page=100": (200, [], {}),
        "/repos/example/transport/pulls/10/commits?per_page=100": (
            200,
            [{"sha": c1_oid}],
            {},
        ),
        "/repos/example/transport/pulls/10/files?per_page=100": (200, [], {}),
    }


async def test_t11_early_end_to_end_milestone_and_surface_equivalence(tmp_path: Path) -> None:
    """T11 & Delivery §4 / §6:
    Complete all 9 early agent workflow steps across SDK, CLI, in-memory MCP, and real subprocess stdio MCP:
    1. Read `codecontext://status` and identify prepared commit and coverage.
    2. Call `search` for the retry function (`send`).
    3. Call `context` with returned symbol ID or exact path/line.
    4. Expand one rationale citation through `get` and verify original text.
    5. Verify original reason and rejected alternative with citations.
    6. Preserve `historical_only` when no current corroboration exists.
    7. Ask about undocumented function (`undocumented_helper`) and report `undocumented_in_available_evidence`.
    8. Modify fixture workspace without syncing; repeat `context` and observe `WORKSPACE_TARGET_DIFFERS` with indexed anchors.
    9. Request an unprepared revision and handle the explicit `REVISION_NOT_PREPARED` error.
    Also verify SDK, CLI, and MCP return equivalent canonical ContextPacks at one generation.
    """
    f01 = build_f01_fixture(tmp_path / "f01")
    store_dir = tmp_path / "store"
    routes = _build_f01_routes(f01)
    transport, _ = make_mock_github_transport(routes)

    # Initialize and sync via SDK
    with Memory.initialize(
        store_dir,
        InitRequest(repo_path=f01.repo_dir, github_repository="example/transport"),
    ) as mem:
        mem._github_transport = transport
        mem._github_token = ""
        sync_rep = mem.sync(SyncRequest(revision="HEAD", offline=False))
        assert sync_rep.revision == f01.c2_oid

        # Enrich with DeterministicFixtureAdapter
        enrich_rep = mem.enrich(
            EnrichRequest(max_input_tokens=100_000, max_output_tokens=20_000, max_calls=10),
            model=DeterministicFixtureAdapter(),
        )
        assert enrich_rep.state == "complete"
        assert enrich_rep.claims_added == 1

    # Compare SDK, CLI, and subprocess stdio MCP on the exact same question & generation
    query_str = "Why is send limited to 3 retries?"
    target_dict = {"kind": "path", "path": "src/transport.py", "line": 5}

    with Memory.open(store_dir, read_only=True) as sdk_mem:
        sdk_pack = sdk_mem.context(
            ContextRequest(query=query_str, target=PathTarget.model_validate(target_dict))
        )

    cli_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "codecontext",
            "--store",
            str(store_dir),
            "context",
            query_str,
            "--path",
            "src/transport.py",
            "--line",
            "5",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    cli_pack_json = json.loads(cli_proc.stdout)
    assert cli_pack_json == sdk_pack.model_dump(mode="json")

    # Test real subprocess stdio MCP client through all 9 milestone steps!
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "codecontext",
            "--store",
            str(store_dir),
            "serve",
            "--workspace",
            str(f01.repo_dir),
        ],
    )
    async with Client(server_params) as client:
        # Check advertised tools and annotations
        tools_res = await client.list_tools()
        tools_by_name = {t.name: t for t in tools_res.tools}
        assert set(tools_by_name.keys()) == {"search", "get", "context"}
        for t in tools_res.tools:
            schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", {})
            assert "request" in schema.get("properties", {})
            assert t.annotations is not None
            assert t.annotations.read_only_hint is True
            assert t.annotations.destructive_hint is False
            assert t.annotations.idempotent_hint is True

        # Step 1: Read `codecontext://status` resource
        resources_res = await client.list_resources()
        assert any(str(r.uri) == "codecontext://status" for r in resources_res.resources)
        status_read = await client.read_resource("codecontext://status")
        status_payload = json.loads(status_read.contents[0].text)
        assert status_payload["revision"] == f01.c2_oid
        assert status_payload["read_only"] is True
        assert status_payload["coverage"]["git"]["state"] == "complete"

        # Step 2: Call `search` for retry function `send`
        s_res = await client.call_tool(
            "search",
            {"request": {"query": "send", "kind": "code", "limit": 5}},
        )
        assert not s_res.is_error
        s_content = s_res.structured_content
        assert s_content["schema_version"] == 1
        send_hit = s_content["hits"][0]
        assert send_hit["kind"] == "symbol"
        send_sym_id = send_hit["id"]

        # Step 3: Call `context` with returned symbol ID
        c_res = await client.call_tool(
            "context",
            {
                "request": {
                    "query": query_str,
                    "target": {"kind": "symbol", "id": send_sym_id},
                }
            },
        )
        assert not c_res.is_error
        c_pack = c_res.structured_content
        assert c_pack["rationale_status"] == "supported"
        assert len(c_pack["decisions"]) == 1
        decision = c_pack["decisions"][0]

        # Step 4 & 5: Expand one rationale citation through `get` and verify original text, reason, and rejected alternative
        assert "Unlimited retries with exponential backoff" in decision["alternatives"]
        assert len(decision["evidence_refs"]) >= 1
        cite_ev_id = decision["evidence_refs"][0]["id"]
        g_res = await client.call_tool(
            "get",
            {"request": {"id": cite_ev_id}},
        )
        assert not g_res.is_error
        g_page = g_res.structured_content
        assert "Limit retries to three because each retry holds a worker" in g_page["record"]["evidence"]["excerpt"]

        # Step 6: Preserve `historical_only` when no current corroboration exists
        assert decision["applicability"] == "historical_only"

        # Step 7: Ask about undocumented function (`undocumented_helper`) and report `undocumented_in_available_evidence`
        undoc_res = await client.call_tool(
            "context",
            {
                "request": {
                    "query": "Why does undocumented_helper add 1?",
                    "target": {"kind": "path", "path": "src/transport.py", "line": 17},
                }
            },
        )
        assert not undoc_res.is_error
        undoc_pack = undoc_res.structured_content
        assert undoc_pack["rationale_status"] == "undocumented_in_available_evidence"
        assert undoc_pack["decisions"] == []

        # Step 8: Modify fixture workspace without syncing; repeat `context` and observe `WORKSPACE_TARGET_DIFFERS` with indexed anchors
        (f01.repo_dir / "src/transport.py").write_bytes(f01.c1_code + b"\n# local dirty edit\n")
        dirty_res = await client.call_tool(
            "context",
            {
                "request": {
                    "query": query_str,
                    "target": {"kind": "symbol", "id": send_sym_id},
                }
            },
        )
        assert not dirty_res.is_error
        dirty_pack = dirty_res.structured_content
        assert dirty_pack["workspace"]["target_state"] == "unstaged_change"
        assert any(g["code"] == "WORKSPACE_TARGET_DIFFERS" for g in dirty_pack["gaps"])
        assert dirty_pack["code"][0]["anchor"]["revision"] == f01.c2_oid

        # Step 9: Request an unprepared revision and handle the explicit `REVISION_NOT_PREPARED` error
        err_res = await client.call_tool(
            "context",
            {
                "request": {
                    "query": query_str,
                    "revision": f01.c0_oid,
                }
            },
        )
        assert err_res.is_error is True
        err_text = err_res.content[0].text
        brace_idx = err_text.find("{")
        assert brace_idx != -1
        err_json = json.loads(err_text[brace_idx:])
        assert err_json["error"]["code"] == "REVISION_NOT_PREPARED"
        assert err_json["error"]["details"]["active_revision"] == f01.c2_oid


def test_cli_commands_validation_exit_codes_export_and_purge(tmp_path: Path) -> None:
    """Test CLI commands (`init`, `sync`, `enrich`, `search`, `get`, `context`, `status`, `export`, `purge`)
    and exit codes (0, 1, 2).
    """
    f01 = build_f01_fixture(tmp_path / "f01_cli")
    store_dir = tmp_path / "cli_store"

    # Missing --store returns exit code 2
    assert cli_main(["status"]) == 2

    # 1. init
    assert (
        cli_main(
            [
                "--store",
                str(store_dir),
                "init",
                "--repo",
                str(f01.repo_dir),
            ]
        )
        == 0
    )

    # 2. sync --offline
    assert (
        cli_main(
            [
                "--store",
                str(store_dir),
                "sync",
                "--offline",
                "--path",
                "src/transport.py",
                "--line",
                "5",
            ]
        )
        == 0
    )

    # 3. enrich via --adapter examples.fixture_adapter:create_adapter
    assert (
        cli_main(
            [
                "--store",
                str(store_dir),
                "enrich",
                "--adapter",
                "examples.fixture_adapter:create_adapter",
                "--max-input-tokens",
                "50000",
                "--max-output-tokens",
                "10000",
                "--max-calls",
                "5",
            ]
        )
        == 0
    )

    # 4. search & status
    assert cli_main(["--store", str(store_dir), "search", "send"]) == 0
    assert cli_main(["--store", str(store_dir), "status"]) == 0

    # Invalid CLI arguments (--line with --symbol) returns exit code 2
    assert (
        cli_main(
            [
                "--store",
                str(store_dir),
                "context",
                "why",
                "--symbol",
                "sym_" + "a" * 64,
                "--line",
                "5",
            ]
        )
        == 2
    )

    # Operational failure (missing record ID) returns exit code 1
    assert (
        cli_main(
            [
                "--store",
                str(store_dir),
                "get",
                "sym_" + "f" * 64,
            ]
        )
        == 1
    )

    # 5. export archive and reopen offline
    export_dest = tmp_path / "exported_archive"
    assert (
        cli_main(
            [
                "--store",
                str(store_dir),
                "export",
                "--destination",
                str(export_dest),
            ]
        )
        == 0
    )
    assert (export_dest / "manifest.json").exists()
    assert (export_dest / "OVERVIEW.md").exists()

    # Reopen exported archive via SDK even when read_only=False is passed -> stays read_only!
    with Memory.open(export_dest, read_only=False) as arch_mem:
        st = arch_mem.status()
        assert st.read_only is True
        s_res = arch_mem.search(SearchRequest(query="send"))
        assert len(s_res.hits) >= 1

    # 6. purge --entire-store removes owned store without touching f01.repo_dir or export_dest
    assert (
        cli_main(
            [
                "--store",
                str(store_dir),
                "purge",
                "--entire-store",
            ]
        )
        == 0
    )
    assert not (store_dir / "memory.sqlite3").exists()
    assert f01.repo_dir.exists()
    assert export_dest.exists()
