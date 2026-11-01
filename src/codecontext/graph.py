import functools
import itertools
import os
import posixpath
import re
import sqlite3
import stat
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple

from pydantic import Field

from codecontext.deadline import (
    SyncDeadlineExceeded,
    check_deadline,
    expired,
    remaining_timeout,
    sync_deadline,
)
from codecontext.enrichment import compute_snapshot_groups
from codecontext.errors import CodeContextError
from codecontext.git import (
    CatFileBatchSession,
    GitMirror,
    ParsedHunk,
    compute_unified_zero_hunks,
    parse_diff_tree_raw_z,
    run_safe_git,
)
from codecontext.identity import path_bytes_to_display
from codecontext.models import NonNegative, Positive, PublicModel, RecordID
from codecontext.store import Store
from codecontext.syntax import (
    MAX_PARSE_BYTES,
    ExtractedSymbol,
    detect_language_for_path,
    is_lfs_pointer,
    parse_blob,
)


class GraphNode(PublicModel):
    id: RecordID
    label: Annotated[str, Field(min_length=1, max_length=100)]
    name: Annotated[str, Field(min_length=1, max_length=1000)]
    qualified_name: Annotated[str, Field(min_length=1, max_length=2000)]
    file_path: Annotated[str, Field(min_length=1, max_length=32768)]
    language: Annotated[str, Field(min_length=1, max_length=40)]
    line_start: Positive
    line_end: Positive
    signature: Annotated[str, Field(max_length=4000)]
    inbound_degree: NonNegative = 0
    outbound_degree: NonNegative = 0
    line_count: Positive = 1


class GraphEdge(PublicModel):
    from_id: RecordID
    from_name: str
    to_id: RecordID
    to_name: str
    edge_type: str
    method: str
    depth: Positive = 1


class SearchGraphRequest(PublicModel):
    name_pattern: Annotated[str, Field(min_length=1, max_length=500)] | None = None
    qn_pattern: Annotated[str, Field(min_length=1, max_length=1000)] | None = None
    file_pattern: Annotated[str, Field(min_length=1, max_length=1000)] | None = None
    label: Annotated[str, Field(min_length=1, max_length=100)] | None = None
    relationship: Literal[
        "CALLS",
        "CONTAINS",
        "INHERITS",
        "IMPLEMENTS",
        "OVERRIDES",
        "REFERENCES_TYPE",
        "DECORATES",
        "IMPORTS",
        "TESTS",
        "ALL",
    ] = "CALLS"
    direction: Literal["inbound", "outbound", "both"] = "both"
    min_degree: NonNegative | None = None
    max_degree: NonNegative | None = None
    exclude_entry_points: bool = False
    limit: Annotated[int, Field(ge=1, le=200)] = 20
    offset: NonNegative = 0


class SearchGraphResult(PublicModel):
    nodes: list[GraphNode]
    total_matched: NonNegative
    has_more: bool


class TracePathRequest(PublicModel):
    function_name: Annotated[str, Field(min_length=1, max_length=1000)]
    direction: Literal["inbound", "outbound", "both"] = "both"
    depth: Annotated[int, Field(ge=1, le=6)] = 2
    limit: Annotated[int, Field(ge=1, le=200)] = 50


class TracePathResult(PublicModel):
    root_nodes: list[GraphNode]
    visited_nodes: list[GraphNode]
    edges: list[GraphEdge]


class QueryGraphRequest(PublicModel):
    query: Annotated[str, Field(min_length=1, max_length=4000)]
    limit: Annotated[int, Field(ge=1, le=200)] = 50


class QueryGraphResult(PublicModel):
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: NonNegative
    truncated: bool


class ArchitectureCluster(PublicModel):
    id: str
    scope_kind: str
    path: str
    file_count: NonNegative
    symbol_count: NonNegative
    cohesion: float
    internal_calls: NonNegative
    external_calls: NonNegative
    top_symbols: list[str]
    depends_on_clusters: list[str]


class GetArchitectureRequest(PublicModel):
    max_clusters: Annotated[int, Field(ge=1, le=100)] = 50


class GetArchitectureResult(PublicModel):
    revision: str | None
    total_files: NonNegative
    total_symbols: NonNegative
    total_call_edges: NonNegative
    languages: dict[str, NonNegative]
    clusters: list[ArchitectureCluster]
    hotspots: list[GraphNode]
    cycles: list[list[str]] = Field(default_factory=list)
    coupling: dict[str, dict[str, float]] = Field(default_factory=dict)


class DetectChangesRequest(PublicModel):
    since: Annotated[str, Field(min_length=1, max_length=200)] | None = None
    depth: Annotated[int, Field(ge=1, le=6)] = 2
    limit: Annotated[int, Field(ge=1, le=200)] = 50


class DetectChangesResult(PublicModel):
    changed_files: list[str]
    directly_modified_symbols: list[GraphNode]
    affected_callers: list[GraphNode]
    impact_edges: list[GraphEdge]
    affected_symbols: list[GraphNode] = Field(default_factory=list)
    deleted_symbols: list[str] = Field(default_factory=list)
    classifications: dict[str, str] = Field(default_factory=dict)
    affected_tests: list[str] = Field(default_factory=list)


class GetCodeSnippetRequest(PublicModel):
    qualified_name: Annotated[str, Field(min_length=1, max_length=2000)]
    file_path: Annotated[str, Field(min_length=1, max_length=4096)] | None = None
    include_neighbors: bool = False


class GetCodeSnippetResult(PublicModel):
    node: GraphNode
    code: str
    callers: list[GraphNode]
    callees: list[GraphNode]


ENTRY_POINT_NAMES: frozenset[str] = frozenset(
    {
        "main",
        "__init__",
        "onCreate",
        "onStart",
        "setUp",
        "tearDown",
    }
)


def _kind_to_label(sym_kind: str) -> str:
    if sym_kind in ("function", "async_function"):
        return "Function"
    if sym_kind in ("method", "async_method", "method_signature", "function_signature"):
        return "Method"
    if sym_kind == "class":
        return "Class"
    if sym_kind == "interface":
        return "Interface"
    if sym_kind == "enum":
        return "Enum"
    if sym_kind == "type_alias":
        return "Type"
    if sym_kind == "namespace":
        return "Namespace"
    return "Variable"


GRAPH_TOOL_MAX_SECONDS = 10.0
# Diffs up to a thousand files through git, which needs more room than an in-memory graph walk.
DETECT_CHANGES_MAX_SECONDS = 60.0
DEADLINE_CHECK_EVERY = 256


def _time_limited[**P, R](limit_seconds: Callable[[], float]) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Run a graph tool under `limit_seconds()`, keeping any tighter outer deadline."""

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                with sync_deadline(remaining_timeout(limit_seconds())):
                    try:
                        return fn(*args, **kwargs)
                    except sqlite3.OperationalError as exc:
                        # Store connections abort statements through a progress handler once
                        # the deadline passes; that surfaces here as "interrupted".
                        if expired():
                            raise SyncDeadlineExceeded from exc
                        raise
            except SyncDeadlineExceeded as exc:
                raise CodeContextError(
                    code="INVALID_REQUEST",
                    message="Graph query exceeded its time limit",
                    retryable=False,
                ) from exc

        return wrapper

    return decorate


_bounded_by_graph_time_limit = _time_limited(lambda: GRAPH_TOOL_MAX_SECONDS)


UNSAFE_REGEX_BACKREF_RE = re.compile(r"\\[1-9]|\(\?P=|\(\?<=|\(\?<!|\(\?=|\(\?!")


MAX_VARIABLE_QUANTIFIERS = 2
# Even two variable quantifiers backtrack cubically (`\w*\w+!` takes ~0.1 s on 512 chars and
# ~0.75 s on 1024), and a Python regex inside a SQLite UDF cannot be interrupted, so cap the text.
MAX_REGEX_SUBJECT_CHARS = 512
REGEX_BRACE_QUANTIFIER_RE = re.compile(r"\{(\d*)(,?)(\d*)\}")


def _read_quantifier(pattern: str, i: int) -> tuple[int, bool, bool]:
    """Return `(next_index, is_quantifier, is_variable)` for a quantifier starting at `i`.

    A quantifier is variable when it can match more than one repeat count: `*`, `+`, `?`,
    `{n,}`, and `{n,m}` with m > n. Fixed `{n}` and `{n,n}` cannot backtrack.
    """
    n = len(pattern)
    if i >= n:
        return i, False, False
    if pattern[i] in "*+?":
        variable = True
        i += 1
    else:
        brace = REGEX_BRACE_QUANTIFIER_RE.match(pattern, i)
        if brace is None:
            return i, False, False
        low = int(brace.group(1) or 0)
        variable = bool(brace.group(2)) and (not brace.group(3) or int(brace.group(3)) > low)
        i = brace.end()
    if i < n and pattern[i] in "?+":
        i += 1
    return i, True, variable


def _regex_can_backtrack(pattern: str) -> bool:
    """Reject a quantified group holding `|` or a quantifier, and too many variable quantifiers."""
    stack: list[list[bool]] = []
    variable_count = 0
    i = 0
    n = len(pattern)
    while i < n:
        char = pattern[i]
        if char == "\\":
            i += 2
            continue
        if char == "[":
            i += 1
            if i < n and pattern[i] == "^":
                i += 1
            if i < n and pattern[i] == "]":
                i += 1
            while i < n and pattern[i] != "]":
                if pattern[i] == "\\":
                    i += 2
                else:
                    i += 1
            i += 1
            continue
        if char == "(":
            stack.append([False, False])
            i += 1
            # `(?:`, `(?P<name>` and inline flags start with `?`, which is not a quantifier here.
            if i < n and pattern[i] == "?":
                i += 1
            continue
        if char == "|" and stack:
            stack[-1][0] = True
            i += 1
            continue
        if char == ")" and stack:
            has_alt, has_quant = stack.pop()
            i, quantified, variable = _read_quantifier(pattern, i + 1)
            variable_count += variable
            if quantified and (has_alt or has_quant):
                return True
            if stack and has_alt:
                stack[-1][0] = True
            if stack and (quantified or has_quant):
                stack[-1][1] = True
            continue
        next_i, quantified, variable = _read_quantifier(pattern, i)
        if quantified:
            variable_count += variable
            if stack:
                stack[-1][1] = True
            i = next_i
            continue
        i += 1
    return variable_count > MAX_VARIABLE_QUANTIFIERS


def compile_safe_regex(pattern: str, *, max_length: int = 256) -> re.Pattern[str]:
    """Compile a user regex, rejecting backreferences, lookaround, and quantifier shapes that can backtrack."""
    if len(pattern) > max_length:
        raise CodeContextError(
            code="INVALID_REQUEST",
            message=f"Regex pattern exceeds maximum allowed length ({max_length}).",
            retryable=False,
        )
    if UNSAFE_REGEX_BACKREF_RE.search(pattern) or _regex_can_backtrack(pattern):
        raise CodeContextError(
            code="INVALID_REQUEST",
            message=(
                "Unsafe regex pattern rejected (nested quantifiers, more than "
                f"{MAX_VARIABLE_QUANTIFIERS} variable quantifiers, lookarounds, "
                "or backreferences are forbidden)."
            ),
            retryable=False,
        )
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise CodeContextError(
            code="INVALID_REQUEST",
            message=f"Invalid regex pattern: {exc}",
            retryable=False,
        ) from exc


def _load_active_graph(
    store: Store,
) -> tuple[dict[str, GraphNode], list[tuple[str, str, str, str]], int]:
    """Load active snapshot symbols, active symbol-to-symbol edges, and `access_epoch`."""
    with store.read_snapshot() as snap:
        conn = snap.conn
        expected_epoch = snap.access_epoch
        sym_rows = conn.execute(
            """
            SELECT s.id, s.kind, s.name, s.qualified_name, s.line_start, s.line_end, s.signature,
                   sf.path, e.language
            FROM symbols s
            JOIN extractions e ON e.id = s.extraction_id
            JOIN snapshot_files sf ON sf.extraction_id = s.extraction_id
            ORDER BY sf.path ASC, s.byte_start ASC, s.id ASC
            """
        ).fetchall()

        rel_rows = conn.execute(
            """
            SELECT from_id, to_id, relation_kind, method
            FROM relations
            WHERE from_kind = 'symbol' AND to_kind = 'symbol' AND state = 'active'
            ORDER BY id ASC
            """
        ).fetchall()

    raw_nodes: dict[str, dict[str, Any]] = {}
    for r in sym_rows:
        sid = str(r["id"])
        if sid in raw_nodes:
            continue
        disp_p, _ = path_bytes_to_display(bytes(r["path"]))
        l_start = int(r["line_start"])
        l_end = int(r["line_end"])
        raw_nodes[sid] = {
            "file_path": disp_p,
            "id": sid,
            "inbound": 0,
            "kind": str(r["kind"]),
            "label": _kind_to_label(str(r["kind"])),
            "language": str(r["language"]),
            "line_count": max(1, l_end - l_start + 1),
            "line_end": l_end,
            "line_start": l_start,
            "name": str(r["name"]),
            "outbound": 0,
            "qualified_name": str(r["qualified_name"]),
            "signature": str(r["signature"]),
        }

    edges: list[tuple[str, str, str, str]] = []
    seen_pairs: set[tuple[str, str, str]] = set()
    for rr in rel_rows:
        f_id = str(rr["from_id"])
        t_id = str(rr["to_id"])
        if f_id not in raw_nodes or t_id not in raw_nodes:
            continue
        rkind = str(rr["relation_kind"])
        edge_type = "CALLS" if rkind == "calls_candidate" else ("CONTAINS" if rkind == "contains" else rkind.upper())
        pair_key = (f_id, t_id, edge_type)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        edges.append((f_id, t_id, edge_type, str(rr["method"])))
        if edge_type == "CALLS":
            raw_nodes[f_id]["outbound"] += 1
            raw_nodes[t_id]["inbound"] += 1

    nodes: dict[str, GraphNode] = {
        sid: GraphNode(
            id=d["id"],
            label=d["label"],
            name=d["name"],
            qualified_name=d["qualified_name"],
            file_path=d["file_path"],
            language=d["language"],
            line_start=d["line_start"],
            line_end=d["line_end"],
            signature=d["signature"],
            inbound_degree=d["inbound"],
            outbound_degree=d["outbound"],
            line_count=d["line_count"],
        )
        for sid, d in raw_nodes.items()
    }
    return nodes, edges, expected_epoch


@_bounded_by_graph_time_limit
def execute_search_graph(store: Store, request: SearchGraphRequest) -> SearchGraphResult:
    """Search symbols in the active code knowledge graph by regex pattern, label, and degree."""
    name_re = compile_safe_regex(request.name_pattern) if request.name_pattern else None
    qn_re = compile_safe_regex(request.qn_pattern) if request.qn_pattern else None
    file_re = compile_safe_regex(request.file_pattern) if request.file_pattern else None

    with store.read_snapshot() as snap:
        conn = snap.conn
        conn.create_function("display_path", 1, lambda p: path_bytes_to_display(bytes(p))[0], deterministic=True)
        conn.create_function("symbol_label", 1, _kind_to_label, deterministic=True)
        conditions: list[str] = []
        params: list[Any] = []
        for field, pattern in [("name", name_re), ("qualified_name", qn_re), ("file_path", file_re)]:
            if pattern is not None:
                fn = f"matches_{field}"
                conn.create_function(fn, 1, lambda value, regex=pattern: bool(regex.search(value[:MAX_REGEX_SUBJECT_CHARS])), deterministic=True)
                conditions.append(f"{fn}({field})")
        if request.label:
            if request.label.lower() == "function":
                conditions.append("label IN ('Function', 'Method')")
            else:
                conditions.append("lower(label) = ?")
                params.append(request.label.lower())
        if request.exclude_entry_points:
            placeholders = ",".join("?" for _ in ENTRY_POINT_NAMES)
            conditions.append(f"name NOT IN ({placeholders}) AND substr(name, 1, 5) != 'test_' AND instr(file_path, 'Test') = 0")
            params.extend(sorted(ENTRY_POINT_NAMES))
        where = " AND ".join(conditions) or "1"
        if request.relationship == "ALL":
            relation_filter = ""
        else:
            relation_kind = (
                "calls_candidate"
                if request.relationship == "CALLS"
                else request.relationship.lower()
            )
            relation_filter = f"AND r.relation_kind = '{relation_kind}'"
        degrees = []
        for own, other, alias in [("to_id", "from_id", "inbound_degree"), ("from_id", "to_id", "outbound_degree")]:
            degrees.append(f"""(SELECT COUNT(*) FROM (
                SELECT DISTINCT r.{other}, r.relation_kind
                FROM relations r INDEXED BY relations_{own.removesuffix("_id")}
                WHERE r.{own} = filtered.id AND r.from_kind = 'symbol'
                  AND r.to_kind = 'symbol' AND r.state = 'active' {relation_filter}
                  AND EXISTS (
                    SELECT 1 FROM symbols endpoint
                    JOIN snapshot_files sf ON sf.extraction_id = endpoint.extraction_id
                    WHERE endpoint.id = r.{other}
                  )
            )) AS {alias}""")
        degree_expr = {"inbound": "inbound_degree", "outbound": "outbound_degree", "both": "inbound_degree + outbound_degree"}[request.direction]
        degree_conditions = []
        for op, bound in [(">=", request.min_degree), ("<=", request.max_degree)]:
            if bound is not None:
                degree_conditions.append(f"({degree_expr}) {op} ?")
                params.append(bound)
        degree_where = " AND ".join(degree_conditions) or "1"
        sql = f"""
            WITH paths AS (
                SELECT extraction_id, MIN(path) AS path FROM snapshot_files
                WHERE extraction_id IS NOT NULL GROUP BY extraction_id
            ), nodes AS (
                SELECT s.id, symbol_label(s.kind) AS label, s.name, s.qualified_name,
                       display_path(paths.path) AS file_path, e.language,
                       s.line_start, s.line_end, s.signature,
                       MAX(1, s.line_end - s.line_start + 1) AS line_count
                FROM paths JOIN symbols s ON s.extraction_id = paths.extraction_id
                JOIN extractions e ON e.id = s.extraction_id
            ), filtered AS (
                SELECT * FROM nodes WHERE {where}
            ), degrees AS (
                SELECT filtered.*, {', '.join(degrees)} FROM filtered
            ), matched AS (
                SELECT * FROM degrees WHERE {degree_where}
            )
        """
        total = int(conn.execute(sql + "SELECT COUNT(*) FROM matched", params).fetchone()[0])
        rows = conn.execute(
            sql + "SELECT * FROM matched ORDER BY file_path, line_start, id LIMIT ? OFFSET ?",
            [*params, request.limit, request.offset],
        ).fetchall()
        nodes = [GraphNode(**dict(row)) for row in rows]
        expected_epoch = snap.access_epoch
    store.verify_access_epoch_and_authorize(expected_epoch)
    return SearchGraphResult(
        nodes=nodes, total_matched=total,
        has_more=request.offset + len(nodes) < total,
    )


@_bounded_by_graph_time_limit
def execute_trace_path(store: Store, request: TracePathRequest) -> TracePathResult:
    """Perform multi-hop BFS call-graph traversal (`inbound`, `outbound`, or `both`) up to `request.depth`."""
    nodes, edges, expected_epoch = _load_active_graph(store)

    q_name = request.function_name.strip()
    roots = [
        n
        for n in nodes.values()
        if n.id == q_name or n.qualified_name == q_name or n.name == q_name
    ]
    if not roots:
        roots = [n for n in nodes.values() if q_name.lower() in n.qualified_name.lower()][:5]
    roots = roots[: request.limit]

    outbound_adj: dict[str, list[tuple[str, str, str]]] = {}
    inbound_adj: dict[str, list[tuple[str, str, str]]] = {}
    for f_id, t_id, etype, method in edges:
        if etype == "CALLS":
            outbound_adj.setdefault(f_id, []).append((t_id, etype, method))
            inbound_adj.setdefault(t_id, []).append((f_id, etype, method))

    visited: dict[str, GraphNode] = {r.id: r for r in roots}
    collected_edges: list[GraphEdge] = []
    seen_edges: set[tuple[str, str, str]] = set()

    queue: deque[tuple[str, int]] = deque((r.id, 0) for r in roots)
    steps = 0
    while queue and len(collected_edges) < request.limit:
        if steps % DEADLINE_CHECK_EVERY == 0:
            check_deadline()
        steps += 1
        cur_id, cur_depth = queue.popleft()
        if cur_depth >= request.depth:
            continue
        next_depth = cur_depth + 1

        if request.direction in ("outbound", "both"):
            for tgt_id, etype, method in outbound_adj.get(cur_id, []):
                ekey = (cur_id, tgt_id, etype)
                if ekey not in seen_edges and len(collected_edges) < request.limit:
                    seen_edges.add(ekey)
                    collected_edges.append(
                        GraphEdge(
                            from_id=cur_id,
                            from_name=nodes[cur_id].qualified_name,
                            to_id=tgt_id,
                            to_name=nodes[tgt_id].qualified_name,
                            edge_type=etype,
                            method=method,
                            depth=next_depth,
                        )
                    )
                if tgt_id not in visited:
                    visited[tgt_id] = nodes[tgt_id]
                    queue.append((tgt_id, next_depth))

        if request.direction in ("inbound", "both"):
            for caller_id, etype, method in inbound_adj.get(cur_id, []):
                ekey = (caller_id, cur_id, etype)
                if ekey not in seen_edges and len(collected_edges) < request.limit:
                    seen_edges.add(ekey)
                    collected_edges.append(
                        GraphEdge(
                            from_id=caller_id,
                            from_name=nodes[caller_id].qualified_name,
                            to_id=cur_id,
                            to_name=nodes[cur_id].qualified_name,
                            edge_type=etype,
                            method=method,
                            depth=next_depth,
                        )
                    )
                if caller_id not in visited:
                    visited[caller_id] = nodes[caller_id]
                    queue.append((caller_id, next_depth))

    store.verify_access_epoch_and_authorize(expected_epoch)
    return TracePathResult(
        root_nodes=roots,
        visited_nodes=list(visited.values()),
        edges=collected_edges,
    )


CYPHER_NODE_RE = re.compile(
    r"^\s*MATCH\s*\(\s*(?P<var>[A-Za-z_]\w*)(?::(?P<label>[A-Za-z_]\w*))?\s*(?:\{(?P<props>[^}]*)\})?\s*\)\s*"
    r"(?:WHERE\s+(?P<where>.+?)\s*)?"
    r"RETURN\s+(?P<ret>.+?)"
    r"(?:\s+ORDER\s+BY\s+(?P<order_by>.+?))?"
    r"(?:\s+LIMIT\s+(?P<limit>\d+))?\s*$",
    re.IGNORECASE | re.DOTALL,
)

CYPHER_EDGE_RE = re.compile(
    r"^\s*MATCH\s*\(\s*(?P<v1>[A-Za-z_]\w*)(?::(?P<l1>[A-Za-z_]\w*))?\s*(?:\{(?P<p1>[^}]*)\})?\s*\)\s*"
    r"-\s*\[\s*(?P<relvar>[A-Za-z_]\w*)?(?::(?P<reltype>[A-Za-z_]\w*))?(?:\*(?P<minhop>\d+)(?:\.\.(?P<maxhop>\d+))?)?\s*(?:\{(?P<relprops>[^}]*)\})?\s*\]\s*->\s*"
    r"\(\s*(?P<v2>[A-Za-z_]\w*)(?::(?P<l2>[A-Za-z_]\w*))?\s*(?:\{(?P<p2>[^}]*)\})?\s*\)\s*"
    r"(?:WHERE\s+(?P<where>.+?)\s*)?"
    r"RETURN\s+(?P<ret>.+?)"
    r"(?:\s+ORDER\s+BY\s+(?P<order_by>.+?))?"
    r"(?:\s+LIMIT\s+(?P<limit>\d+))?\s*$",
    re.IGNORECASE | re.DOTALL,
)

PROP_PAIR_RE = re.compile(
    r"([A-Za-z_]\w*)\s*:\s*(?:['\"]([^'\"]*)['\"]|(-?\d+(?:\.\d+)?))"
)
WHERE_ATOM_RE = re.compile(
    r"^\s*(?P<not>NOT\s+)?"
    r"(?P<var>[A-Za-z_]\w*)\.(?P<prop>[A-Za-z_]\w*)\s*"
    r"(?:(?P<null_op>IS\s+NOT\s+NULL|IS\s+NULL)|"
    r"(?P<op>=~|!=|<>|<=|>=|=|<|>|CONTAINS|STARTS\s+WITH|ENDS\s+WITH)\s*"
    r"(?:['\"](?P<sval>[^'\"]*)['\"]|(?P<nval>-?\d+(?:\.\d+)?)|(?P<bval>true|false|null)))\s*$",
    re.IGNORECASE,
)


def _node_attr(node: GraphNode, prop: str) -> Any:
    mapping = {
        "degree": node.inbound_degree + node.outbound_degree,
        "file": node.file_path,
        "file_path": node.file_path,
        "filePath": node.file_path,
        "id": node.id,
        "inbound_degree": node.inbound_degree,
        "kind": node.label,
        "label": node.label,
        "language": node.language,
        "line_count": node.line_count,
        "line_end": node.line_end,
        "line_start": node.line_start,
        "name": node.name,
        "outbound_degree": node.outbound_degree,
        "qualified_name": node.qualified_name,
        "signature": node.signature,
    }
    if prop not in mapping:
        raise CodeContextError(
            code="INVALID_REQUEST",
            message=f"Unknown node property {prop!r} in Cypher query. Supported properties: {sorted(mapping.keys())}",
            retryable=False,
        )
    return mapping[prop]


def _matches_inline_props(node: GraphNode, props_str: str | None) -> bool:
    if not props_str or not props_str.strip():
        return True
    pos = 0
    for m in PROP_PAIR_RE.finditer(props_str):
        k = m.group(1)
        s_val = m.group(2)
        n_val = m.group(3)
        actual = _node_attr(node, k)
        if n_val is not None:
            if float(actual) != float(n_val):
                return False
        else:
            if str(actual or "") != s_val:
                return False
        pos = m.end()
    if pos == 0:
        raise CodeContextError(
            code="INVALID_REQUEST",
            message=f"Malformed inline property filter in Cypher query: {props_str!r}",
            retryable=False,
        )
    return True


def _matches_label(node: GraphNode, label: str | None) -> bool:
    if not label:
        return True
    low = label.lower()
    if low == "function" and node.label in ("Function", "Method"):
        return True
    return node.label.lower() == low


def _eval_where_atom(bindings: dict[str, Any], atom_str: str) -> bool:
    m = WHERE_ATOM_RE.match(atom_str)
    if not m:
        raise CodeContextError(
            code="INVALID_REQUEST",
            message=f"Unsupported or malformed Cypher WHERE condition: {atom_str.strip()!r}",
            retryable=False,
        )
    negate = bool(m.group("not"))
    var = m.group("var")
    prop = m.group("prop")
    if var not in bindings:
        raise CodeContextError(
            code="INVALID_REQUEST",
            message=f"Unknown variable {var!r} in Cypher WHERE clause.",
            retryable=False,
        )
    obj = bindings[var]
    actual = _node_attr(obj, prop) if isinstance(obj, GraphNode) else obj.get(prop)

    null_op = m.group("null_op")
    if null_op:
        is_not_null = "NOT" in null_op.upper()
        res = (actual is not None) if is_not_null else (actual is None)
        return (not res) if negate else res

    op = re.sub(r"\s+", " ", m.group("op").upper())
    s_val = m.group("sval")
    n_val = m.group("nval")
    b_val = m.group("bval")

    if n_val is not None:
        if actual is None or not isinstance(actual, (int, float)):
            res = False
        else:
            lhs = float(actual)
            rhs = float(n_val)
            if op == "=":
                res = lhs == rhs
            elif op in ("!=", "<>"):
                res = lhs != rhs
            elif op == "<":
                res = lhs < rhs
            elif op == "<=":
                res = lhs <= rhs
            elif op == ">":
                res = lhs > rhs
            elif op == ">=":
                res = lhs >= rhs
            else:
                raise CodeContextError(
                    code="INVALID_REQUEST",
                    message=f"Operator {op!r} is not supported for numeric comparisons.",
                    retryable=False,
                )
    elif b_val is not None:
        b_low = b_val.lower()
        target_b = None if b_low == "null" else (b_low == "true")
        if op == "=":
            res = actual == target_b
        elif op in ("!=", "<>"):
            res = actual != target_b
        else:
            raise CodeContextError(
                code="INVALID_REQUEST",
                message=f"Operator {op!r} is not supported for boolean/null literal {b_val!r}.",
                retryable=False,
            )
    else:
        lhs_s = str(actual or "")
        rhs_s = s_val or ""
        if op == "=":
            res = lhs_s == rhs_s
        elif op in ("!=", "<>"):
            res = lhs_s != rhs_s
        elif op == "<":
            res = lhs_s < rhs_s
        elif op == "<=":
            res = lhs_s <= rhs_s
        elif op == ">":
            res = lhs_s > rhs_s
        elif op == ">=":
            res = lhs_s >= rhs_s
        elif op == "CONTAINS":
            res = rhs_s in lhs_s
        elif op == "STARTS WITH":
            res = lhs_s.startswith(rhs_s)
        elif op == "ENDS WITH":
            res = lhs_s.endswith(rhs_s)
        elif op == "=~":
            compiled_re = compile_safe_regex(rhs_s)
            res = bool(compiled_re.search(lhs_s[:MAX_REGEX_SUBJECT_CHARS]))
        else:
            raise CodeContextError(
                code="INVALID_REQUEST",
                message=f"Unsupported string operator {op!r} in Cypher WHERE clause.",
                retryable=False,
            )

    return (not res) if negate else res


def _split_outside_quotes(clause: str, keyword: str) -> list[str]:
    """Split on a whitespace-delimited keyword that is not inside a quoted literal."""
    parts: list[str] = []
    start = 0
    for m in re.finditer(rf"'[^']*'|\"[^\"]*\"|\s+{keyword}\s+", clause, flags=re.IGNORECASE):
        if m.group()[0] not in "'\"":
            parts.append(clause[start : m.start()])
            start = m.end()
    parts.append(clause[start:])
    return parts


def _matches_where(bindings: dict[str, Any], where_clause: str | None) -> bool:
    """Evaluate Cypher WHERE clause with `OR`, `AND`, and `NOT` precedence.
    Strictly validates every condition atom and raises `INVALID_REQUEST` on unknown syntax or properties.
    """
    if not where_clause or not where_clause.strip():
        return True
    for or_grp in _split_outside_quotes(where_clause.strip(), "OR"):
        and_atoms = _split_outside_quotes(or_grp.strip(), "AND")
        if all(_eval_where_atom(bindings, atom) for atom in and_atoms):
            return True
    return False


def _project_return_row(bindings: dict[str, Any], return_exprs: list[str]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for expr in return_exprs:
        clean = expr.strip()
        as_parts = re.split(r"\s+AS\s+", clean, flags=re.IGNORECASE)
        target_expr = as_parts[0].strip()
        col_name = as_parts[1].strip() if len(as_parts) > 1 else target_expr
        if "." in target_expr:
            v_name, p_name = target_expr.split(".", 1)
            v_name = v_name.strip()
            p_name = p_name.strip()
            if v_name not in bindings:
                raise CodeContextError(
                    code="INVALID_REQUEST",
                    message=f"Unknown variable {v_name!r} in RETURN clause.",
                    retryable=False,
                )
            obj = bindings[v_name]
            if isinstance(obj, GraphNode):
                row[col_name] = _node_attr(obj, p_name)
            elif isinstance(obj, dict):
                if p_name not in obj:
                    raise CodeContextError(
                        code="INVALID_REQUEST",
                        message=f"Unknown relationship property {p_name!r} in RETURN clause.",
                        retryable=False,
                    )
                row[col_name] = obj.get(p_name)
            else:
                row[col_name] = None
        else:
            if target_expr not in bindings:
                raise CodeContextError(
                    code="INVALID_REQUEST",
                    message=f"Unknown variable or expression {target_expr!r} in RETURN clause.",
                    retryable=False,
                )
            obj = bindings[target_expr]
            if isinstance(obj, GraphNode):
                row[col_name] = obj.model_dump()
            else:
                row[col_name] = obj
    return row


def _is_count_return(ret_cols: list[str]) -> bool:
    return len(ret_cols) == 1 and bool(
        re.match(r"^count\s*\(.+\)(?:\s+AS\s+\w+)?$", ret_cols[0], flags=re.IGNORECASE)
    )


def _return_column_name(expr: str) -> str:
    as_parts = re.split(r"\s+AS\s+", expr.strip(), flags=re.IGNORECASE)
    return as_parts[1].strip() if len(as_parts) > 1 else as_parts[0].strip()


def _apply_order_by_and_count(
    matches: Iterator[dict[str, Any]],
    ret_cols: list[str],
    order_by_clause: str | None,
    cap: int,
) -> tuple[list[str], list[dict[str, Any]], bool]:
    if _is_count_return(ret_cols):
        col_name = _return_column_name(ret_cols[0])
        return ([col_name], [{col_name: sum(1 for _ in matches)}], False)

    order_parts = [p.strip() for p in (order_by_clause or "").split(",") if p.strip()]
    returned = {_return_column_name(c) for c in ret_cols}
    for opart in order_parts:
        if opart.split()[0] not in returned:
            raise CodeContextError(
                code="INVALID_REQUEST",
                message=f"ORDER BY {opart.split()[0]!r} must name a RETURN column.",
                retryable=False,
            )
    if not order_parts:
        # Rows past `cap + 1` can never be returned or change `truncated`.
        matches = itertools.islice(matches, cap + 1)
    projected = [_project_return_row(b, ret_cols) for b in matches]

    if order_parts:
        for opart in reversed(order_parts):
            tokens = opart.split()
            expr_key = tokens[0]
            desc = len(tokens) > 1 and tokens[1].upper() == "DESC"

            def _sort_key(item: dict[str, Any], key: str = expr_key) -> Any:
                if key in item and item[key] is not None:
                    return item[key]
                return ""

            projected.sort(key=_sort_key, reverse=desc)

    truncated = len(projected) > cap
    return (ret_cols, projected[:cap], truncated)


@_bounded_by_graph_time_limit
def execute_query_graph(store: Store, request: QueryGraphRequest) -> QueryGraphResult:
    """Execute a validated read-only Cypher query over the active snapshot symbol/relation graph."""
    nodes, edges, expected_epoch = _load_active_graph(store)
    q = request.query.strip().rstrip(";")

    m_edge = CYPHER_EDGE_RE.match(q)
    if m_edge:
        v1 = m_edge.group("v1")
        l1 = m_edge.group("l1")
        p1 = m_edge.group("p1")
        relvar = m_edge.group("relvar") or "r"
        reltype = m_edge.group("reltype")
        min_hop = int(m_edge.group("minhop")) if m_edge.group("minhop") else 1
        max_hop = (
            int(m_edge.group("maxhop"))
            if m_edge.group("maxhop")
            else (min_hop if m_edge.group("minhop") else 1)
        )
        if not 1 <= min_hop <= max_hop <= 6:
            raise CodeContextError(
                code="INVALID_REQUEST",
                message="Cypher path bounds must satisfy 1 <= minimum <= maximum <= 6.",
                retryable=False,
            )
        relprops = m_edge.group("relprops")
        v2 = m_edge.group("v2")
        l2 = m_edge.group("l2")
        p2 = m_edge.group("p2")
        where_c = m_edge.group("where")
        ret_c = m_edge.group("ret")
        order_c = m_edge.group("order_by")
        lim_s = m_edge.group("limit")

        cap = min(request.limit, int(lim_s) if lim_s else request.limit, 200)
        ret_cols = [c.strip() for c in ret_c.split(",") if c.strip()]

        req_edge_type: str | None = None
        if reltype and reltype.upper() != "CODERELATION":
            req_edge_type = reltype.upper()
        elif relprops:
            for pm in PROP_PAIR_RE.finditer(relprops):
                if pm.group(1) == "type" and pm.group(2):
                    req_edge_type = pm.group(2).upper()

        def single_hop_matches() -> Iterator[dict[str, Any]]:
            for index, (f_id, t_id, etype, method) in enumerate(edges):
                if index % DEADLINE_CHECK_EVERY == 0:
                    check_deadline()
                if req_edge_type and etype != req_edge_type:
                    continue
                n1 = nodes[f_id]
                n2 = nodes[t_id]
                if not _matches_label(n1, l1) or not _matches_inline_props(n1, p1):
                    continue
                if not _matches_label(n2, l2) or not _matches_inline_props(n2, p2):
                    continue
                bindings = {
                    v1: n1,
                    v2: n2,
                    relvar: {"confidence": 1.0, "depth": 1, "method": method, "type": etype},
                }
                if _matches_where(bindings, where_c):
                    yield bindings

        def variable_length_matches() -> Iterator[dict[str, Any]]:
            adj: dict[str, list[tuple[str, str, str]]] = {}
            for f_id, t_id, etype, method in edges:
                if req_edge_type and etype != req_edge_type:
                    continue
                adj.setdefault(f_id, []).append((t_id, etype, method))

            steps = 0
            for start_id, n1 in nodes.items():
                if not _matches_label(n1, l1) or not _matches_inline_props(n1, p1):
                    continue
                bfs_q: deque[tuple[str, int]] = deque([(start_id, 0)])
                seen_at_start: set[str] = {start_id}
                while bfs_q:
                    cur_id, d = bfs_q.popleft()
                    if d >= max_hop:
                        continue
                    for nxt_id, et, m in adj.get(cur_id, []):
                        if steps % DEADLINE_CHECK_EVERY == 0:
                            check_deadline()
                        steps += 1
                        nd = d + 1
                        if nd >= min_hop:
                            n2 = nodes[nxt_id]
                            if _matches_label(n2, l2) and _matches_inline_props(n2, p2):
                                bindings = {
                                    v1: n1,
                                    v2: n2,
                                    relvar: {"confidence": 1.0, "depth": nd, "method": m, "type": et},
                                }
                                if _matches_where(bindings, where_c):
                                    yield bindings
                        if nxt_id not in seen_at_start:
                            seen_at_start.add(nxt_id)
                            bfs_q.append((nxt_id, nd))

        cols_out, rows_out, truncated = _apply_order_by_and_count(
            single_hop_matches() if max_hop == 1 else variable_length_matches(),
            ret_cols,
            order_c,
            cap,
        )
        store.verify_access_epoch_and_authorize(expected_epoch)
        return QueryGraphResult(
            columns=cols_out,
            rows=rows_out,
            row_count=len(rows_out),
            truncated=truncated,
        )

    m_node = CYPHER_NODE_RE.match(q)
    if m_node:
        var = m_node.group("var")
        label = m_node.group("label")
        props = m_node.group("props")
        where_c = m_node.group("where")
        ret_c = m_node.group("ret")
        order_c = m_node.group("order_by")
        lim_s = m_node.group("limit")

        cap = min(request.limit, int(lim_s) if lim_s else request.limit, 200)
        ret_cols = [c.strip() for c in ret_c.split(",") if c.strip()]

        def node_matches() -> Iterator[dict[str, Any]]:
            for index, node in enumerate(nodes.values()):
                if index % DEADLINE_CHECK_EVERY == 0:
                    check_deadline()
                if not _matches_label(node, label) or not _matches_inline_props(node, props):
                    continue
                bindings = {var: node}
                if _matches_where(bindings, where_c):
                    yield bindings

        cols_out, rows_out, truncated = _apply_order_by_and_count(
            node_matches(), ret_cols, order_c, cap
        )
        store.verify_access_epoch_and_authorize(expected_epoch)
        return QueryGraphResult(
            columns=cols_out,
            rows=rows_out,
            row_count=len(rows_out),
            truncated=truncated,
        )

    raise CodeContextError(
        code="INVALID_REQUEST",
        message="Unsupported Cypher syntax. Expected `MATCH (n[:Label]) [WHERE ...] RETURN ... [ORDER BY ...] [LIMIT N]` or `MATCH (a)-[r[:TYPE][*1..3]]->(b) [WHERE ...] RETURN ...`.",
        retryable=False,
    )


def find_scc_cycles(adj: dict[str, set[str]]) -> list[list[str]]:
    """Find all cyclic Strongly Connected Components using Tarjan's algorithm."""
    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    cycles: list[list[str]] = []

    def strongconnect(v: str) -> None:
        nonlocal index
        indices[v] = index
        lowlinks[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)

        for w in sorted(adj.get(v, set())):
            if w not in indices:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif w in on_stack:
                lowlinks[v] = min(lowlinks[v], indices[w])

        if lowlinks[v] == indices[v]:
            scc: list[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                scc.append(w)
                if w == v:
                    break
            if len(scc) > 1 or (len(scc) == 1 and v in adj.get(v, set())):
                cycles.append(sorted(scc))

    for node in sorted(adj.keys()):
        if node not in indices:
            strongconnect(node)
    return sorted(cycles, key=lambda c: (len(c), c))


def detect_louvain_communities(
    nodes: list[str],
    edges: list[tuple[str, str, float]],
    *,
    resolution: float = 1.0,
) -> list[list[str]]:
    """Partition nodes into cohesive functional communities using modularity optimization."""
    if not nodes:
        return []
    adj: dict[str, dict[str, float]] = {n: {} for n in nodes}
    degrees: dict[str, float] = {n: 0.0 for n in nodes}
    total_weight = 0.0
    for u, v, w in edges:
        if u in adj and v in adj:
            adj[u][v] = adj[u].get(v, 0.0) + w
            adj[v][u] = adj[v].get(u, 0.0) + w
            degrees[u] += w
            degrees[v] += w
            total_weight += w

    if total_weight <= 0:
        return [[n] for n in sorted(nodes)]

    community: dict[str, str] = {n: n for n in nodes}
    comm_nodes: dict[str, set[str]] = {n: {n} for n in nodes}
    comm_degrees: dict[str, float] = {n: degrees[n] for n in nodes}

    m2 = 2.0 * total_weight
    improved = True
    passes = 0
    while improved and passes < 10:
        improved = False
        passes += 1
        for u in sorted(nodes):
            c_u = community[u]
            k_u = degrees[u]
            neigh_comms: dict[str, float] = {}
            for v, w in adj[u].items():
                c_v = community[v]
                neigh_comms[c_v] = neigh_comms.get(c_v, 0.0) + w

            best_comm = c_u
            best_gain = 0.0
            w_u_in = neigh_comms.get(c_u, 0.0)
            deg_c_u = comm_degrees[c_u] - k_u

            # Sorted so ties resolve the same way regardless of edge insertion order.
            for c_cand, w_u_cand in sorted(neigh_comms.items()):
                if c_cand == c_u:
                    continue
                deg_c_cand = comm_degrees[c_cand]
                gain = (w_u_cand - w_u_in) - resolution * (k_u * (deg_c_cand - deg_c_u)) / m2
                if gain > best_gain:
                    best_gain = gain
                    best_comm = c_cand

            if best_comm != c_u and best_gain > 1e-6:
                comm_nodes[c_u].remove(u)
                comm_degrees[c_u] -= k_u
                community[u] = best_comm
                comm_nodes[best_comm].add(u)
                comm_degrees[best_comm] += k_u
                improved = True

    result: list[list[str]] = [sorted(c_set) for c_set in comm_nodes.values() if c_set]
    return sorted(result, key=lambda c: (-len(c), c[0] if c else ""))


def _cluster_files(
    groups: list[Any],
    nodes: dict[str, GraphNode],
    edges: list[tuple[str, str, str, str]],
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, tuple[str, str]]]:
    """Cluster files by Louvain over the cross-file call graph; files without calls stay in their snapshot group.

    Clustering files rather than symbols keeps the graph small on large repositories while still
    following call cohesion. Returns `(file -> cluster id, cluster id -> files, cluster id -> (scope_kind, path))`.
    """
    pair_weights: dict[tuple[str, str], float] = {}
    for f_id, t_id, etype, _ in edges:
        if etype != "CALLS":
            continue
        f_file = nodes[f_id].file_path
        t_file = nodes[t_id].file_path
        if f_file != t_file:
            key = (f_file, t_file) if f_file < t_file else (t_file, f_file)
            pair_weights[key] = pair_weights.get(key, 0.0) + 1.0
    linked_files = sorted({f for pair in pair_weights for f in pair})
    communities = detect_louvain_communities(
        linked_files,
        [(a, b, w) for (a, b), w in sorted(pair_weights.items())],
    )

    file_to_cluster: dict[str, str] = {}
    cluster_files: dict[str, list[str]] = {}
    cluster_meta: dict[str, tuple[str, str]] = {}
    for members in communities:
        if len(members) < 2:
            continue
        common = posixpath.commonpath(members) if all("/" in m for m in members) else ""
        if common in members:
            common = posixpath.dirname(common)
        path = common or "."
        cid = f"community:{path}"
        suffix = 2
        while cid in cluster_meta:
            cid = f"community:{path}#{suffix}"
            suffix += 1
        cluster_meta[cid] = ("community", path)
        cluster_files[cid] = list(members)
        for fs in members:
            file_to_cluster[fs] = cid

    for g in groups:
        rest = [fs for fs in (path_bytes_to_display(fb)[0] for fb in g.files) if fs not in file_to_cluster]
        if not rest:
            continue
        cid = g.path_display
        cluster_meta[cid] = (g.scope_kind, g.path_display)
        cluster_files[cid] = rest
        for fs in rest:
            file_to_cluster[fs] = cid
    return file_to_cluster, cluster_files, cluster_meta


@_bounded_by_graph_time_limit
def execute_get_architecture(
    store: Store,
    request: GetArchitectureRequest,
) -> GetArchitectureResult:
    """Compute structural architecture overview: Louvain call-graph clusters, cohesion scores,
    inter-cluster call dependencies, language breakdown, and top degree hotspots.
    """
    nodes, edges, expected_epoch = _load_active_graph(store)
    with store.read_snapshot() as snap:
        conn = snap.conn
        rev = snap.revision
        total_files = int(conn.execute("SELECT COUNT(*) FROM snapshot_files").fetchone()[0])
        groups = compute_snapshot_groups(conn, repository_id=snap.repository_id)

    lang_counts: dict[str, int] = {}
    file_to_nodes: dict[str, list[GraphNode]] = {}
    for n in nodes.values():
        lang_counts[n.language] = lang_counts.get(n.language, 0) + 1
        file_to_nodes.setdefault(n.file_path, []).append(n)

    file_to_cluster, cluster_files, cluster_meta = _cluster_files(groups, nodes, edges)
    node_to_cluster: dict[str, str] = {
        gn.id: file_to_cluster[fs]
        for fs, f_nodes in file_to_nodes.items()
        if fs in file_to_cluster
        for gn in f_nodes
    }

    internal_calls: dict[str, int] = {cid: 0 for cid in cluster_meta}
    external_calls: dict[str, int] = {cid: 0 for cid in cluster_meta}
    cluster_deps: dict[str, set[str]] = {cid: set() for cid in cluster_meta}
    total_call_edges = 0

    for f_id, t_id, etype, _ in edges:
        if etype != "CALLS":
            continue
        total_call_edges += 1
        c_from = node_to_cluster.get(f_id)
        c_to = node_to_cluster.get(t_id)
        if c_from and c_to:
            if c_from == c_to:
                internal_calls[c_from] += 1
            else:
                external_calls[c_from] += 1
                cluster_deps[c_from].add(c_to)

    clusters_out: list[ArchitectureCluster] = []
    for cid, (s_kind, p_disp) in cluster_meta.items():
        c_nodes = [
            gn
            for fs in cluster_files.get(cid, [])
            for gn in file_to_nodes.get(fs, [])
        ]
        c_nodes.sort(key=lambda x: (-(x.inbound_degree + x.outbound_degree), x.qualified_name))
        int_c = internal_calls[cid]
        ext_c = external_calls[cid]
        tot_c = int_c + ext_c
        cohesion = round(int_c / tot_c, 4) if tot_c > 0 else 1.0
        clusters_out.append(
            ArchitectureCluster(
                id=cid,
                scope_kind=s_kind,
                path=p_disp,
                file_count=len(cluster_files.get(cid, [])),
                symbol_count=len(c_nodes),
                cohesion=cohesion,
                internal_calls=int_c,
                external_calls=ext_c,
                top_symbols=[n.qualified_name for n in c_nodes[:10]],
                depends_on_clusters=sorted(cluster_deps[cid]),
            )
        )

    clusters_out.sort(key=lambda c: (-c.symbol_count, c.path))
    hotspots = sorted(
        nodes.values(),
        key=lambda n: (-(n.inbound_degree + n.outbound_degree), -n.line_count, n.id),
    )[:15]

    scc_cycles = find_scc_cycles(cluster_deps)
    coupling: dict[str, dict[str, float]] = {}
    for cid in cluster_meta:
        ce = float(len(cluster_deps[cid]))
        ca = float(sum(1 for other, deps in cluster_deps.items() if cid in deps))
        instability = round(ce / (ca + ce), 4) if (ca + ce) > 0 else 0.0
        coupling[cid] = {"ca": ca, "ce": ce, "instability": instability}

    store.verify_access_epoch_and_authorize(expected_epoch)
    return GetArchitectureResult(
        revision=rev,
        total_files=total_files,
        total_symbols=len(nodes),
        total_call_edges=total_call_edges,
        languages=lang_counts,
        clusters=clusters_out[: request.max_clusters],
        hotspots=hotspots,
        cycles=scc_cycles,
        coupling=coupling,
    )


MAX_CLASSIFIED_FILES = 1000


class _FileChange(NamedTuple):
    old_path: bytes | None
    new_path: bytes | None
    old_blob_oid: str | None
    new_blob_oid: str | None
    worktree_bytes: bytes | None = None

_TEST_DIR_SEGMENTS = frozenset({"tests", "test", "__tests__"})
_TEST_FILE_RE = re.compile(
    r"(?:^test_.*\.py|.*_test\.(?:py|go)|.*\.(?:test|spec)\.(?:ts|tsx|js|jsx)|.*Test\.(?:java|kt))$"
)


def is_test_path(path: str) -> bool:
    """Match test files by directory segment or filename convention, never by substring."""
    *dirs, leaf = path.split("/")
    return any(d in _TEST_DIR_SEGMENTS for d in dirs) or bool(_TEST_FILE_RE.match(leaf))


def _workspace_changes(
    mirror: GitMirror,
    workspace_path: Path,
    active_rev: str,
) -> list[_FileChange]:
    """Diff the active revision's tree against the worktree, reading new content from disk.

    Falls back to HEAD when the workspace clone lacks the active revision.
    """
    proc = None
    for base in (active_rev, "HEAD"):
        proc = run_safe_git(
            ["diff-index", "-z", "--raw", "--no-abbrev", "--no-renames", "--end-of-options", base],
            cwd=workspace_path,
            empty_hooks_dir=mirror.empty_hooks_dir,
            check=False,
        )
        if proc.returncode == 0:
            break
    if proc is None or proc.returncode != 0:
        return []
    root = workspace_path.resolve()
    out: list[_FileChange] = []
    for rc in parse_diff_tree_raw_z(proc.stdout, parent_index=0, parent_oid=None):
        new_bytes = _read_worktree_file(root, rc.new_path) if rc.new_path is not None else None
        out.append(_FileChange(rc.old_path, rc.new_path, rc.old_blob_oid, None, new_bytes))
    return out


def _read_worktree_file(root: Path, rel: bytes) -> bytes | None:
    """Read a regular file inside `root`; symlinks and oversized files yield None."""
    full = root / os.fsdecode(rel)
    try:
        st = full.lstat()
        if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_PARSE_BYTES:
            return None
        if not full.resolve().is_relative_to(root):
            return None
        return full.read_bytes()
    except OSError:
        return None


def _blob_symbols(
    batch: CatFileBatchSession,
    path: bytes | None,
    blob_oid: str | None,
    content: bytes | None,
) -> tuple[bytes | None, list[ExtractedSymbol]]:
    """Return a text blob's bytes and parsed symbols; binary, LFS, oversized, or unsupported blobs yield no symbols."""
    if path is None:
        return None, []
    if content is None and blob_oid is not None:
        obj = batch.read_object_bounded(blob_oid, max_bytes=MAX_PARSE_BYTES)
        if obj is not None and obj[0] == "blob":
            content = obj[2]
    if content is None or b"\x00" in content or is_lfs_pointer(content):
        return None, []
    lang = detect_language_for_path(path)
    if lang is None:
        return content, []
    return content, parse_blob(content, lang, blob_oid=blob_oid).symbols


def _hunk_side_touches(start: int, count: int, line_start: int, line_end: int) -> bool:
    if count == 0:
        # A pure insertion sits after `start`, so it only lands inside a multi-line symbol.
        return line_start <= start < line_end
    return max(start, line_start) <= min(start + count - 1, line_end)


def _classify_symbol_changes(
    old_syms: list[ExtractedSymbol],
    new_syms: list[ExtractedSymbol],
    hunks: list[ParsedHunk],
) -> list[tuple[str, str]]:
    """Classify symbols by qualified name as added, deleted, signature_modified, or body_modified.

    Symbols present on both sides that no hunk touches are left out.
    """
    # ponytail: overloads sharing a qualified name collapse to the first occurrence.
    old_by_qn: dict[str, ExtractedSymbol] = {}
    for sym in old_syms:
        old_by_qn.setdefault(sym.qualified_name, sym)
    new_by_qn: dict[str, ExtractedSymbol] = {}
    for sym in new_syms:
        new_by_qn.setdefault(sym.qualified_name, sym)

    out: list[tuple[str, str]] = []
    for qn in sorted(old_by_qn.keys() | new_by_qn.keys()):
        old_sym = old_by_qn.get(qn)
        new_sym = new_by_qn.get(qn)
        if old_sym is None:
            out.append((qn, "added"))
            continue
        if new_sym is None:
            out.append((qn, "deleted"))
            continue
        touched = any(
            _hunk_side_touches(h.old_start, h.old_count, old_sym.line_start, old_sym.line_end)
            or _hunk_side_touches(h.new_start, h.new_count, new_sym.line_start, new_sym.line_end)
            for h in hunks
        )
        if not touched:
            continue
        header_touched = old_sym.signature != new_sym.signature or any(
            _hunk_side_touches(h.old_start, h.old_count, old_sym.line_start, old_sym.line_start)
            or _hunk_side_touches(h.new_start, h.new_count, new_sym.line_start, new_sym.line_start)
            for h in hunks
        )
        out.append((qn, "signature_modified" if header_touched else "body_modified"))
    return out


@_time_limited(lambda: DETECT_CHANGES_MAX_SECONDS)
def execute_detect_changes(
    store: Store,
    mirror: GitMirror,
    request: DetectChangesRequest,
    *,
    workspace_path: Path | None = None,
) -> DetectChangesResult:
    """Diff old and new blobs (since `request.since`, against the worktree, or the active commit's
    own changes) to classify touched symbols, then trace inbound callers up to `request.depth`.
    """
    nodes, edges, expected_epoch = _load_active_graph(store)
    with store.read_snapshot() as snap:
        active_rev = snap.revision
        if not active_rev:
            raise CodeContextError(
                code="REVISION_NOT_PREPARED",
                message="No revision prepared.",
                retryable=False,
            )

    file_changes: list[_FileChange] = []
    compared = False
    if request.since:
        compared = True
        src_repo = mirror.verify_source_binding()
        base_oid = mirror.resolve_source_revision(src_repo, request.since)
        for rc in mirror.diff_tree_parent(base_oid, active_rev, parent_index=0):
            file_changes.append(_FileChange(rc.old_path, rc.new_path, rc.old_blob_oid, rc.new_blob_oid))
    elif workspace_path is not None and workspace_path.exists():
        compared = True
        file_changes = _workspace_changes(mirror, workspace_path, active_rev)

    if not file_changes and not compared:
        with store.connect_read() as conn:
            c_rows = conn.execute(
                """
                SELECT old_path, new_path, old_blob_oid, new_blob_oid FROM changes
                WHERE commit_oid = ? ORDER BY parent_index ASC, id ASC
                """,
                (active_rev,),
            ).fetchall()
        seen_change_paths: set[bytes] = set()
        for cr in c_rows:
            old_p = bytes(cr["old_path"]) if cr["old_path"] is not None else None
            new_p = bytes(cr["new_path"]) if cr["new_path"] is not None else None
            key = new_p if new_p is not None else old_p
            if key is None or key in seen_change_paths:
                continue
            seen_change_paths.add(key)
            file_changes.append(_FileChange(old_p, new_p, cr["old_blob_oid"], cr["new_blob_oid"]))

    changed_paths: set[str] = set()
    for old_p, new_p, _, _, _ in file_changes:
        for p in (old_p, new_p):
            if p is not None:
                changed_paths.add(path_bytes_to_display(p)[0])

    classifications: dict[str, str] = {}
    seed_keys: set[tuple[str, str]] = set()
    deleted_symbols: list[str] = []
    with mirror.open_batch_session() as batch:
        # ponytail: only the first MAX_CLASSIFIED_FILES changes are parsed; huge diffs list files but not every symbol.
        for old_p, new_p, old_oid, new_oid, new_bytes_ws in file_changes[:MAX_CLASSIFIED_FILES]:
            old_bytes, old_syms = _blob_symbols(batch, old_p, old_oid, None)
            new_bytes, new_syms = _blob_symbols(batch, new_p, new_oid, new_bytes_ws)
            if not old_syms and not new_syms:
                continue
            hunks = compute_unified_zero_hunks(old_bytes, new_bytes)
            old_disp = path_bytes_to_display(old_p)[0] if old_p is not None else None
            new_disp = path_bytes_to_display(new_p)[0] if new_p is not None else None
            for qn, kind in _classify_symbol_changes(old_syms, new_syms, hunks):
                classifications[qn] = kind
                if kind == "deleted":
                    deleted_symbols.append(qn)
                for disp in (new_disp, old_disp):
                    if disp is not None:
                        seed_keys.add((disp, qn))

    deleted_set = set(deleted_symbols)
    seed_ids: set[str] = set()
    modified_syms: list[GraphNode] = []
    for n in nodes.values():
        if (n.file_path, n.qualified_name) in seed_keys:
            seed_ids.add(n.id)
            if n.qualified_name not in deleted_set:
                modified_syms.append(n)
    modified_ids = {n.id for n in modified_syms}

    inbound_adj: dict[str, list[tuple[str, str, str]]] = {}
    tested_by: dict[str, list[str]] = {}
    for f_id, t_id, etype, method in edges:
        if etype == "CALLS":
            inbound_adj.setdefault(t_id, []).append((f_id, etype, method))
        elif etype == "TESTS":
            tested_by.setdefault(t_id, []).append(f_id)

    affected_callers: dict[str, GraphNode] = {}
    impact_edges: list[GraphEdge] = []
    seen_edges: set[tuple[str, str]] = set()
    queue: deque[tuple[str, int]] = deque((nid, 0) for nid in sorted(seed_ids))
    visited = set(seed_ids)

    while queue and len(impact_edges) < request.limit:
        cur_id, d = queue.popleft()
        if d >= request.depth:
            continue
        nd = d + 1
        for caller_id, etype, method in inbound_adj.get(cur_id, []):
            if (caller_id, cur_id) not in seen_edges and len(impact_edges) < request.limit:
                seen_edges.add((caller_id, cur_id))
                impact_edges.append(
                    GraphEdge(
                        from_id=caller_id,
                        from_name=nodes[caller_id].qualified_name,
                        to_id=cur_id,
                        to_name=nodes[cur_id].qualified_name,
                        edge_type=etype,
                        method=method,
                        depth=nd,
                    )
                )
            if caller_id not in modified_ids:
                affected_callers[caller_id] = nodes[caller_id]
            if caller_id not in visited:
                visited.add(caller_id)
                queue.append((caller_id, nd))

    affected_syms_list = modified_syms[: request.limit] + list(affected_callers.values())[: request.limit]

    affected_tests_set: set[str] = set()
    for sid in seed_ids | affected_callers.keys():
        if is_test_path(nodes[sid].file_path):
            affected_tests_set.add(nodes[sid].qualified_name)
        for test_id in tested_by.get(sid, []):
            affected_tests_set.add(nodes[test_id].qualified_name)

    store.verify_access_epoch_and_authorize(expected_epoch)
    return DetectChangesResult(
        changed_files=sorted(changed_paths),
        directly_modified_symbols=modified_syms[: request.limit],
        affected_callers=list(affected_callers.values())[: request.limit],
        impact_edges=impact_edges,
        affected_symbols=affected_syms_list[: request.limit],
        deleted_symbols=sorted(deleted_set),
        classifications=classifications,
        affected_tests=sorted(affected_tests_set),
    )


@_bounded_by_graph_time_limit
def execute_get_code_snippet(
    store: Store,
    mirror: GitMirror,
    request: GetCodeSnippetRequest,
) -> GetCodeSnippetResult:
    """Retrieve exact symbol source code by `qualified_name` (or short `name`) and optional `file_path`,
    along with direct callers and callees when `include_neighbors=True`.
    """
    nodes, edges, expected_epoch = _load_active_graph(store)
    q = request.qualified_name.strip()

    candidates = [
        n
        for n in nodes.values()
        if (n.id == q or n.qualified_name == q or n.name == q)
        and (request.file_path is None or n.file_path == request.file_path)
    ]
    if not candidates:
        candidates = [
            n
            for n in nodes.values()
            if q.lower() in n.qualified_name.lower()
            and (request.file_path is None or n.file_path == request.file_path)
        ]
    if not candidates:
        raise CodeContextError(
            code="RECORD_NOT_FOUND",
            message=f"Symbol {q!r} not found in active snapshot.",
            retryable=False,
        )

    exact_q = [n for n in candidates if n.qualified_name == q]
    target_node = exact_q[0] if exact_q else candidates[0]

    with store.connect_read() as conn:
        srow = conn.execute(
            """
            SELECT s.byte_start, s.byte_end, e.blob_oid
            FROM symbols s
            JOIN extractions e ON e.id = s.extraction_id
            WHERE s.id = ?
            """,
            (target_node.id,),
        ).fetchone()
    if not srow:
        raise CodeContextError(
            code="RECORD_NOT_FOUND",
            message=f"Symbol {target_node.id} not found.",
            retryable=False,
        )

    with mirror.open_batch_session() as batch:
        b_obj = batch.read_object(str(srow["blob_oid"]))
    code_text = ""
    if b_obj is not None:
        raw_b = b_obj[2]
        code_text = raw_b[int(srow["byte_start"]) : int(srow["byte_end"])].decode(
            "utf-8", errors="replace"
        )[:16384]

    callers: list[GraphNode] = []
    callees: list[GraphNode] = []
    if request.include_neighbors:
        for f_id, t_id, etype, _ in edges:
            if etype != "CALLS":
                continue
            if t_id == target_node.id and f_id in nodes:
                callers.append(nodes[f_id])
            elif f_id == target_node.id and t_id in nodes:
                callees.append(nodes[t_id])

    store.verify_access_epoch_and_authorize(expected_epoch)
    return GetCodeSnippetResult(
        node=target_node,
        code=code_text,
        callers=callers[:25],
        callees=callees[:25],
    )
