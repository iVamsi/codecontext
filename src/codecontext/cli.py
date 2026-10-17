import argparse
import importlib
import json
import re
import sys
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from codecontext import __version__
from codecontext.errors import CodeContextError
from codecontext.graph import (
    QueryGraphRequest,
    SearchGraphRequest,
    TracePathRequest,
)
from codecontext.identity import parse_iso_utc
from codecontext.memory import Memory
from codecontext.models import (
    ContextRequest,
    EnrichRequest,
    ExportRequest,
    GetRequest,
    InitRequest,
    ModelAdapter,
    ModelIdentity,
    PathTarget,
    PurgeRequest,
    SearchRequest,
    SymbolTarget,
    SyncRequest,
    Target,
)

ADAPTER_SPEC_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")


def load_cli_adapter(adapter_spec: str) -> ModelAdapter:
    """Load an explicitly supplied `module:factory` from the installed Python environment.
    Never modifies `sys.path` or scans the target repository.
    """
    if not adapter_spec or not ADAPTER_SPEC_RE.match(adapter_spec):
        raise CodeContextError(
            code="INVALID_REQUEST",
            message=f"Invalid --adapter format {adapter_spec!r}. Expected 'module.name:factory_fn'.",
            retryable=False,
        )
    mod_name, factory_name = adapter_spec.split(":", 1)
    try:
        mod = importlib.import_module(mod_name)
        factory = getattr(mod, factory_name)
    except Exception as exc:
        raise CodeContextError(
            code="INVALID_REQUEST",
            message=f"Could not import adapter factory {adapter_spec!r}: {type(exc).__name__}",
            retryable=False,
        ) from exc

    if not callable(factory):
        raise CodeContextError(
            code="INVALID_REQUEST",
            message=f"Adapter factory {adapter_spec!r} is not callable.",
            retryable=False,
        )
    try:
        adapter = factory()
    except Exception as exc:
        raise CodeContextError(
            code="MODEL_FAILED",
            message=f"Adapter factory {adapter_spec!r} raised {type(exc).__name__}.",
            retryable=False,
        ) from exc

    if not isinstance(adapter, ModelAdapter):
        raise CodeContextError(
            code="INVALID_REQUEST",
            message=f"Object returned by {adapter_spec!r} does not satisfy ModelAdapter protocol.",
            retryable=False,
        )
    try:
        ident = adapter.identity
        if not isinstance(ident, ModelIdentity):
            ModelIdentity.model_validate(ident)
    except Exception as exc:
        raise CodeContextError(
            code="INVALID_REQUEST",
            message=f"Adapter {adapter_spec!r} has invalid ModelIdentity: {exc}",
            retryable=False,
        ) from exc

    return adapter


def _parse_cli_target(args: argparse.Namespace) -> Target | None:
    path_val = getattr(args, "path", None)
    b64_val = getattr(args, "path_bytes_b64", None)
    sym_val = getattr(args, "symbol", None)
    line_val = getattr(args, "line", None)

    supplied = sum(1 for x in (path_val, b64_val, sym_val) if x is not None)
    if supplied > 1:
        raise CodeContextError(
            code="INVALID_REQUEST",
            message="--path, --path-bytes-b64, and --symbol are mutually exclusive.",
            retryable=False,
        )
    if line_val is not None and path_val is None and b64_val is None:
        raise CodeContextError(
            code="INVALID_REQUEST",
            message="--line requires either --path or --path-bytes-b64 and cannot be combined with --symbol.",
            retryable=False,
        )
    if sym_val is not None:
        return SymbolTarget(kind="symbol", id=sym_val)
    if path_val is not None or b64_val is not None:
        return PathTarget(
            kind="path",
            path=path_val,
            path_bytes_b64=b64_val,
            line=line_val,
        )
    return None


def _parse_cli_cutoff(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    try:
        return parse_iso_utc(raw)
    except Exception as exc:
        raise CodeContextError(
            code="INVALID_REQUEST",
            message=f"Invalid RFC3339 --knowledge-cutoff {raw!r}: must be timezone-aware UTC.",
            retryable=False,
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codecontext",
        description="Local project memory for coding agents.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--store", type=Path, help="Path to the CodeContext memory store directory.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    subparsers = parser.add_subparsers(dest="subcommand")

    # init
    p_init = subparsers.add_parser("init", help="Initialize a CodeContext store.")
    p_init.add_argument("--repo", required=True, type=Path, help="Path to local Git repository.")
    p_init.add_argument("--github", dest="github", default=None, help="GitHub OWNER/REPO.")
    p_init.add_argument(
        "--allow-issue-repo",
        dest="allow_issue_repos",
        action="append",
        default=[],
        help="Allowlisted external OWNER/REPO for one-hop linked issues.",
    )

    # sync
    p_sync = subparsers.add_parser("sync", help="Synchronize Git history and GitHub discussions.")
    p_sync.add_argument("--revision", default="HEAD")
    p_sync.add_argument("--offline", action="store_true")
    p_sync.add_argument("--reconcile", action="store_true")
    g_sync_target = p_sync.add_mutually_exclusive_group()
    g_sync_target.add_argument("--path", default=None)
    g_sync_target.add_argument("--path-bytes-b64", dest="path_bytes_b64", default=None)
    g_sync_target.add_argument("--symbol", default=None)
    p_sync.add_argument("--line", type=int, default=None)
    p_sync.add_argument("--max-requests", dest="max_requests", type=int, default=1000)
    p_sync.add_argument("--max-seconds", dest="max_seconds", type=int, default=900)

    # enrich
    p_enrich = subparsers.add_parser("enrich", help="Run budgeted model extraction.")
    p_enrich.add_argument("--adapter", required=True, help="Explicit MODULE:FACTORY adapter.")
    p_enrich.add_argument("--max-input-tokens", dest="max_input_tokens", required=True, type=int)
    p_enrich.add_argument("--max-output-tokens", dest="max_output_tokens", required=True, type=int)
    p_enrich.add_argument("--max-calls", dest="max_calls", required=True, type=int)
    p_enrich.add_argument("--max-cost-usd", dest="max_cost_usd", default=None)
    p_enrich.add_argument("--target-id", dest="target_ids", action="append", default=None)

    # search
    p_search = subparsers.add_parser("search", help="Search code, history, decisions, or summaries.")
    p_search.add_argument("query")
    p_search.add_argument(
        "--kind",
        choices=["all", "code", "history", "decision", "summary"],
        default="all",
    )
    p_search.add_argument("--revision", default=None)
    p_search.add_argument("--limit", type=int, default=20)
    p_search.add_argument("--knowledge-cutoff", dest="knowledge_cutoff", default=None)
    p_search.add_argument("--cursor", default=None)

    # get
    p_get = subparsers.add_parser("get", help="Expand a record, citation, or coverage gap by ID.")
    p_get.add_argument("id")
    p_get.add_argument("--max-tokens", dest="max_tokens", type=int, default=6000)
    p_get.add_argument("--knowledge-cutoff", dest="knowledge_cutoff", default=None)
    p_get.add_argument("--cursor", default=None)

    # context
    p_ctx = subparsers.add_parser("context", help="Assemble a cited context pack for a question.")
    p_ctx.add_argument("query")
    g_ctx_target = p_ctx.add_mutually_exclusive_group()
    g_ctx_target.add_argument("--path", default=None)
    g_ctx_target.add_argument("--path-bytes-b64", dest="path_bytes_b64", default=None)
    g_ctx_target.add_argument("--symbol", default=None)
    p_ctx.add_argument("--line", type=int, default=None)
    p_ctx.add_argument("--revision", default=None)
    p_ctx.add_argument("--max-tokens", dest="max_tokens", type=int, default=6000)
    p_ctx.add_argument("--knowledge-cutoff", dest="knowledge_cutoff", default=None)
    p_ctx.add_argument("--workspace", type=Path, default=None)

    # status
    subparsers.add_parser("status", help="Show store status, coverage, and gaps.")

    # export
    p_export = subparsers.add_parser("export", help="Export a portable read-only archive.")
    p_export.add_argument("--destination", required=True, type=Path)
    p_export.add_argument("--metadata-only", dest="metadata_only", action="store_true")

    # purge
    p_purge = subparsers.add_parser("purge", help="Purge a discussion source or the entire store.")
    g_purge = p_purge.add_mutually_exclusive_group(required=True)
    g_purge.add_argument("--source-id", dest="source_id", default=None)
    g_purge.add_argument("--entire-store", dest="entire_store", action="store_true")

    # serve
    p_serve = subparsers.add_parser("serve", help="Run the read-only stdio MCP server.")
    p_serve.add_argument("--workspace", type=Path, default=None)
    p_serve.add_argument(
        "--include-graph-tools",
        dest="include_graph_tools",
        action="store_true",
        help="Also expose search_graph, trace_path, and query_graph (Cypher) MCP tools.",
    )

    # search-graph
    p_sg = subparsers.add_parser("search-graph", help="Search the structural code graph by regex, label, or degree.")
    p_sg.add_argument("--name-pattern", dest="name_pattern", default=None)
    p_sg.add_argument("--qn-pattern", dest="qn_pattern", default=None)
    p_sg.add_argument("--label", default=None)
    p_sg.add_argument("--relationship", choices=["CALLS", "CONTAINS", "ALL"], default="CALLS")
    p_sg.add_argument("--direction", choices=["inbound", "outbound", "both"], default="both")
    p_sg.add_argument("--min-degree", dest="min_degree", type=int, default=None)
    p_sg.add_argument("--max-degree", dest="max_degree", type=int, default=None)
    p_sg.add_argument("--exclude-entry-points", dest="exclude_entry_points", action="store_true")
    p_sg.add_argument("--limit", type=int, default=20)
    p_sg.add_argument("--offset", type=int, default=0)

    # trace-path
    p_tp = subparsers.add_parser("trace-path", help="Trace inbound/outbound call chains in the code graph.")
    p_tp.add_argument("function_name")
    p_tp.add_argument("--direction", choices=["inbound", "outbound", "both"], default="both")
    p_tp.add_argument("--depth", type=int, default=2)
    p_tp.add_argument("--limit", type=int, default=50)

    # query-graph
    p_qg = subparsers.add_parser("query-graph", help="Run a Cypher pattern query over the code graph.")
    p_qg.add_argument("query")
    p_qg.add_argument("--limit", type=int, default=50)

    return parser


def _emit_json(model_obj: Any, *, pretty: bool) -> None:
    data = model_obj.model_dump(mode="json")
    if pretty:
        sys.stdout.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(
            json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        )
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand is None:
        if args.store is None:
            parser.print_help()
            return 0
        parser.error("A subcommand is required when --store is provided.")

    if args.subcommand == "serve":
        try:
            from codecontext.mcp_server import run_stdio_server

            run_stdio_server(
                args.store,
                workspace_path=args.workspace,
                include_graph_tools=bool(args.include_graph_tools),
            )
            return 0
        except KeyboardInterrupt:
            return 130
        except CodeContextError as exc:
            sys.stdout.write(exc.to_json(pretty=bool(args.pretty)) + "\n")
            sys.stdout.flush()
            return 2 if exc.code == "INVALID_REQUEST" else 1

    if args.store is None:
        err = CodeContextError(
            code="INVALID_REQUEST",
            message="Global --store PATH is required before the subcommand.",
            retryable=False,
        )
        sys.stdout.write(err.to_json(pretty=bool(args.pretty)) + "\n")
        return 2

    store_path: Path = args.store
    pretty: bool = bool(args.pretty)

    try:

        if args.subcommand == "init":
            init_req = InitRequest(
                repo_path=args.repo,
                github_repository=args.github,
                allowed_issue_repositories=args.allow_issue_repos,
            )
            with Memory.initialize(store_path, init_req) as mem:
                st = mem.status()
            _emit_json(st, pretty=pretty)
            return 0

        if args.subcommand == "sync":
            target = _parse_cli_target(args)
            sync_req = SyncRequest(
                revision=args.revision,
                offline=bool(args.offline),
                reconcile=bool(args.reconcile),
                focus=target,
                max_requests=args.max_requests,
                max_seconds=args.max_seconds,
            )
            with Memory.open(store_path, read_only=False) as mem:
                rep = mem.sync(sync_req)
            _emit_json(rep, pretty=pretty)
            return 0

        if args.subcommand == "enrich":
            cost_dec: Decimal | None = None
            if args.max_cost_usd is not None:
                try:
                    cost_dec = Decimal(str(args.max_cost_usd))
                except InvalidOperation as exc:
                    raise CodeContextError(
                        code="INVALID_REQUEST",
                        message=f"Invalid --max-cost-usd Decimal {args.max_cost_usd!r}.",
                        retryable=False,
                    ) from exc

            enrich_req = EnrichRequest(
                target_ids=args.target_ids,
                max_input_tokens=args.max_input_tokens,
                max_output_tokens=args.max_output_tokens,
                max_calls=args.max_calls,
                max_cost_usd=cost_dec,
            )
            adapter = load_cli_adapter(args.adapter)
            with Memory.open(store_path, read_only=False) as mem:
                rep = mem.enrich(enrich_req, model=adapter)
            _emit_json(rep, pretty=pretty)
            return 0

        if args.subcommand == "search":
            cutoff = _parse_cli_cutoff(args.knowledge_cutoff)
            s_req = SearchRequest(
                query=args.query,
                kind=args.kind,
                revision=args.revision,
                limit=args.limit,
                knowledge_cutoff=cutoff,
                cursor=args.cursor,
            )
            with Memory.open(store_path, read_only=True) as mem:
                res = mem.search(s_req)
            _emit_json(res, pretty=pretty)
            return 0

        if args.subcommand == "get":
            cutoff = _parse_cli_cutoff(args.knowledge_cutoff)
            g_req = GetRequest(
                id=args.id,
                max_tokens=args.max_tokens,
                knowledge_cutoff=cutoff,
                cursor=args.cursor,
            )
            with Memory.open(store_path, read_only=True) as mem:
                res = mem.get(g_req)
            _emit_json(res, pretty=pretty)
            return 0

        if args.subcommand == "context":
            target = _parse_cli_target(args)
            cutoff = _parse_cli_cutoff(args.knowledge_cutoff)
            c_req = ContextRequest(
                query=args.query,
                target=target,
                revision=args.revision,
                max_tokens=args.max_tokens,
                knowledge_cutoff=cutoff,
            )
            with Memory.open(store_path, read_only=True) as mem:
                if args.workspace is not None:
                    mem.configure_workspace(args.workspace)
                res = mem.context(c_req)
            _emit_json(res, pretty=pretty)
            return 0

        if args.subcommand == "status":
            with Memory.open(store_path, read_only=True) as mem:
                res = mem.status()
            _emit_json(res, pretty=pretty)
            return 0

        if args.subcommand == "export":
            e_req = ExportRequest(
                destination=args.destination,
                include_git_objects=not bool(args.metadata_only),
            )
            with Memory.open(store_path, read_only=True) as mem:
                res = mem.export(e_req)
            _emit_json(res, pretty=pretty)
            return 0

        if args.subcommand == "purge":
            p_req = PurgeRequest(
                source_id=args.source_id,
                entire_store=bool(args.entire_store),
            )
            with Memory.open(store_path, read_only=False) as mem:
                res = mem.purge(p_req)
            _emit_json(res, pretty=pretty)
            return 0

        if args.subcommand == "search-graph":
            sg_req = SearchGraphRequest(
                name_pattern=args.name_pattern,
                qn_pattern=args.qn_pattern,
                label=args.label,
                relationship=args.relationship,
                direction=args.direction,
                min_degree=args.min_degree,
                max_degree=args.max_degree,
                exclude_entry_points=bool(args.exclude_entry_points),
                limit=args.limit,
                offset=args.offset,
            )
            with Memory.open(store_path, read_only=True) as mem:
                res = mem.search_graph(sg_req)
            _emit_json(res, pretty=pretty)
            return 0

        if args.subcommand == "trace-path":
            tp_req = TracePathRequest(
                function_name=args.function_name,
                direction=args.direction,
                depth=args.depth,
                limit=args.limit,
            )
            with Memory.open(store_path, read_only=True) as mem:
                res = mem.trace_path(tp_req)
            _emit_json(res, pretty=pretty)
            return 0

        if args.subcommand == "query-graph":
            qg_req = QueryGraphRequest(
                query=args.query,
                limit=args.limit,
            )
            with Memory.open(store_path, read_only=True) as mem:
                res = mem.query_graph(qg_req)
            _emit_json(res, pretty=pretty)
            return 0

        return 2

    except KeyboardInterrupt:
        return 130
    except CodeContextError as exc:
        sys.stdout.write(exc.to_json(pretty=pretty) + "\n")
        sys.stdout.flush()
        return 2 if exc.code == "INVALID_REQUEST" else 1
    except Exception as exc:
        if "ValidationError" in type(exc).__name__:
            v_err = CodeContextError(
                code="INVALID_REQUEST",
                message=str(exc)[:1000],
                retryable=False,
            )
            sys.stdout.write(v_err.to_json(pretty=pretty) + "\n")
            sys.stdout.flush()
            return 2
        u_err = CodeContextError(
            code="IO_ERROR",
            message=f"Unexpected error: {type(exc).__name__}",
            retryable=False,
        )
        sys.stdout.write(u_err.to_json(pretty=pretty) + "\n")
        sys.stdout.flush()
        return 1
