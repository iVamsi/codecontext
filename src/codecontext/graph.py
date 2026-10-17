import re
from collections import deque
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field

from codecontext.enrichment import compute_snapshot_groups
from codecontext.errors import CodeContextError
from codecontext.git import GitMirror, run_safe_git
from codecontext.identity import path_bytes_to_display
from codecontext.models import NonNegative, Positive, PublicModel, RecordID
from codecontext.store import Store


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
    relationship: Literal["CALLS", "CONTAINS", "ALL"] = "CALLS"
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


class DetectChangesRequest(PublicModel):
    since: Annotated[str, Field(min_length=1, max_length=200)] | None = None
    depth: Annotated[int, Field(ge=1, le=6)] = 2
    limit: Annotated[int, Field(ge=1, le=200)] = 50


class DetectChangesResult(PublicModel):
    changed_files: list[str]
    directly_modified_symbols: list[GraphNode]
    affected_callers: list[GraphNode]
    impact_edges: list[GraphEdge]


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


def _load_active_graph(
    store: Store,
) -> tuple[dict[str, GraphNode], list[tuple[str, str, str, str]]]:
    """Load active snapshot symbols and active symbol-to-symbol edges (`CALLS`, `CONTAINS`)."""
    with store.read_snapshot() as snap:
        conn = snap.conn
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
    return nodes, edges


def execute_search_graph(store: Store, request: SearchGraphRequest) -> SearchGraphResult:
    """Search symbols in the active code knowledge graph by regex pattern, label, and degree."""
    nodes, edges = _load_active_graph(store)

    try:
        name_re = re.compile(request.name_pattern) if request.name_pattern else None
        qn_re = re.compile(request.qn_pattern) if request.qn_pattern else None
        file_re = re.compile(request.file_pattern) if request.file_pattern else None
    except re.error as exc:
        raise CodeContextError(
            code="INVALID_REQUEST",
            message=f"Invalid regex pattern in search_graph: {exc}",
            retryable=False,
        ) from exc

    in_deg: dict[str, int] = {nid: 0 for nid in nodes}
    out_deg: dict[str, int] = {nid: 0 for nid in nodes}
    for f_id, t_id, etype, _ in edges:
        if request.relationship in ("ALL", etype):
            out_deg[f_id] = out_deg.get(f_id, 0) + 1
            in_deg[t_id] = in_deg.get(t_id, 0) + 1

    matched: list[GraphNode] = []
    for nid, node in nodes.items():
        if not _matches_label(node, request.label):
            continue
        if name_re and not name_re.search(node.name):
            continue
        if qn_re and not qn_re.search(node.qualified_name):
            continue
        if file_re and not file_re.search(node.file_path):
            continue
        if request.exclude_entry_points and (
            node.name in ENTRY_POINT_NAMES
            or node.name.startswith("test_")
            or "Test" in node.file_path
        ):
            continue

        deg = (
            in_deg[nid]
            if request.direction == "inbound"
            else (
                out_deg[nid]
                if request.direction == "outbound"
                else (in_deg[nid] + out_deg[nid])
            )
        )
        if request.min_degree is not None and deg < request.min_degree:
            continue
        if request.max_degree is not None and deg > request.max_degree:
            continue

        matched.append(
            node.model_copy(
                update={
                    "inbound_degree": in_deg[nid],
                    "outbound_degree": out_deg[nid],
                }
            )
        )

    matched.sort(key=lambda n: (n.file_path, n.line_start, n.id))
    total = len(matched)
    sliced = matched[request.offset : request.offset + request.limit]
    has_more = (request.offset + len(sliced)) < total
    return SearchGraphResult(nodes=sliced, total_matched=total, has_more=has_more)


def execute_trace_path(store: Store, request: TracePathRequest) -> TracePathResult:
    """Perform multi-hop BFS call-graph traversal (`inbound`, `outbound`, or `both`) up to `request.depth`."""
    nodes, edges = _load_active_graph(store)

    q_name = request.function_name.strip()
    roots = [
        n
        for n in nodes.values()
        if n.id == q_name or n.qualified_name == q_name or n.name == q_name
    ]
    if not roots:
        roots = [n for n in nodes.values() if q_name.lower() in n.qualified_name.lower()][:5]

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
    while queue and len(collected_edges) < request.limit:
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

    return TracePathResult(
        root_nodes=roots,
        visited_nodes=list(visited.values()),
        edges=collected_edges,
    )


# --- Cypher Query Executor over SQLite symbol/relation graph ---

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
    r"-\s*\[\s*(?P<relvar>[A-Za-z_]\w*)?(?::(?P<reltype>[A-Za-z_]\w*))?(?:\*(?P<minhop>\d+)?(?:\.\.(?P<maxhop>\d+))?)?\s*(?:\{(?P<relprops>[^}]*)\})?\s*\]\s*->\s*"
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
            try:
                res = bool(re.search(rhs_s, lhs_s))
            except re.error as exc:
                raise CodeContextError(
                    code="INVALID_REQUEST",
                    message=f"Invalid regex in Cypher WHERE clause: {exc}",
                    retryable=False,
                ) from exc
        else:
            raise CodeContextError(
                code="INVALID_REQUEST",
                message=f"Unsupported string operator {op!r} in Cypher WHERE clause.",
                retryable=False,
            )

    return (not res) if negate else res


def _matches_where(bindings: dict[str, Any], where_clause: str | None) -> bool:
    """Evaluate Cypher WHERE clause with `OR`, `AND`, and `NOT` precedence.
    Strictly validates every condition atom and raises `INVALID_REQUEST` on unknown syntax or properties.
    """
    if not where_clause or not where_clause.strip():
        return True
    or_groups = re.split(r"\s+OR\s+", where_clause.strip(), flags=re.IGNORECASE)
    for or_grp in or_groups:
        and_atoms = re.split(r"\s+AND\s+", or_grp.strip(), flags=re.IGNORECASE)
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


def _apply_order_by_and_count(
    matched_bindings: list[dict[str, Any]],
    ret_cols: list[str],
    order_by_clause: str | None,
    cap: int,
) -> tuple[list[str], list[dict[str, Any]], bool]:
    # Check if RETURN is a `count(...)` aggregation
    if len(ret_cols) == 1 and re.match(r"^count\s*\(.+\)(?:\s+AS\s+\w+)?$", ret_cols[0], flags=re.IGNORECASE):
        as_parts = re.split(r"\s+AS\s+", ret_cols[0].strip(), flags=re.IGNORECASE)
        col_name = as_parts[1].strip() if len(as_parts) > 1 else ret_cols[0].strip()
        return ([col_name], [{col_name: len(matched_bindings)}], False)

    projected = [_project_return_row(b, ret_cols) for b in matched_bindings]

    if order_by_clause:
        order_parts = [p.strip() for p in order_by_clause.split(",") if p.strip()]
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


def execute_query_graph(store: Store, request: QueryGraphRequest) -> QueryGraphResult:
    """Execute a validated read-only Cypher query over the active snapshot symbol/relation graph."""
    nodes, edges = _load_active_graph(store)
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
        max_hop = min(max_hop, 6)
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

        matched_bindings: list[dict[str, Any]] = []
        if max_hop == 1:
            for f_id, t_id, etype, method in edges:
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
                    matched_bindings.append(bindings)
        else:
            # Variable-length path `-[r:CALLS*min..max]->`
            adj: dict[str, list[tuple[str, str, str]]] = {}
            for f_id, t_id, etype, method in edges:
                if req_edge_type and etype != req_edge_type:
                    continue
                adj.setdefault(f_id, []).append((t_id, etype, method))

            for start_id, n1 in nodes.items():
                if not _matches_label(n1, l1) or not _matches_inline_props(n1, p1):
                    continue
                bfs_q: deque[tuple[str, int, str, str]] = deque([(start_id, 0, "CALLS", "bfs")])
                seen_at_start: set[str] = {start_id}
                while bfs_q:
                    cur_id, d, last_et, last_m = bfs_q.popleft()
                    if d >= max_hop:
                        continue
                    for nxt_id, et, m in adj.get(cur_id, []):
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
                                    matched_bindings.append(bindings)
                        if nxt_id not in seen_at_start:
                            seen_at_start.add(nxt_id)
                            bfs_q.append((nxt_id, nd, et, m))

        cols_out, rows_out, truncated = _apply_order_by_and_count(
            matched_bindings, ret_cols, order_c, cap
        )
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

        matched_bindings = []
        for node in nodes.values():
            if not _matches_label(node, label) or not _matches_inline_props(node, props):
                continue
            bindings = {var: node}
            if _matches_where(bindings, where_c):
                matched_bindings.append(bindings)

        cols_out, rows_out, truncated = _apply_order_by_and_count(
            matched_bindings, ret_cols, order_c, cap
        )
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


def execute_get_architecture(
    store: Store,
    request: GetArchitectureRequest,
) -> GetArchitectureResult:
    """Compute structural architecture overview: package/directory clusters, cohesion scores,
    inter-cluster call dependencies, language breakdown, and top degree hotspots.
    """
    nodes, edges = _load_active_graph(store)
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

    node_to_cluster: dict[str, str] = {}
    cluster_files: dict[str, list[str]] = {}
    cluster_meta: dict[str, tuple[str, str]] = {}

    for g in groups:
        cid = g.path_display
        cluster_meta[cid] = (g.scope_kind, g.path_display)
        f_strs = [path_bytes_to_display(fb)[0] for fb in g.files]
        cluster_files[cid] = f_strs
        for fs in f_strs:
            for gn in file_to_nodes.get(fs, []):
                node_to_cluster[gn.id] = cid

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

    return GetArchitectureResult(
        revision=rev,
        total_files=total_files,
        total_symbols=len(nodes),
        total_call_edges=total_call_edges,
        languages=lang_counts,
        clusters=clusters_out[: request.max_clusters],
        hotspots=hotspots,
    )


def execute_detect_changes(
    store: Store,
    mirror: GitMirror,
    request: DetectChangesRequest,
    *,
    workspace_path: Path | None = None,
) -> DetectChangesResult:
    """Map git changes (either since a ref/commit `request.since` or dirty workspace files)
    to directly modified symbols and trace inbound callers up to `request.depth`.
    """
    nodes, edges = _load_active_graph(store)
    with store.read_snapshot() as snap:
        active_rev = snap.revision
        if not active_rev:
            raise CodeContextError(
                code="REVISION_NOT_PREPARED",
                message="No revision prepared.",
                retryable=False,
            )

    changed_paths: set[str] = set()
    if request.since:
        src_repo = mirror.verify_source_binding()
        base_oid = mirror.resolve_source_revision(src_repo, request.since)
        raw_changes = mirror.diff_tree_parent(base_oid, active_rev, parent_index=0)
        for rc in raw_changes:
            if rc.new_path is not None:
                changed_paths.add(path_bytes_to_display(rc.new_path)[0])
            if rc.old_path is not None:
                changed_paths.add(path_bytes_to_display(rc.old_path)[0])
    elif workspace_path is not None and workspace_path.exists():
        # Check diff between workspace HEAD and active_rev, or staged/unstaged files
        proc = run_safe_git(
            ["diff-index", "--name-only", "-z", "HEAD"],
            cwd=workspace_path,
            empty_hooks_dir=mirror.empty_hooks_dir,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout:
            for p_b in proc.stdout.split(b"\x00"):
                if p_b:
                    changed_paths.add(path_bytes_to_display(p_b)[0])

    if not changed_paths:
        # Fallback: changes in the latest active commit (`active_rev`)
        with store.connect_read() as conn:
            c_rows = conn.execute(
                "SELECT new_path, old_path FROM changes WHERE commit_oid = ?",
                (active_rev,),
            ).fetchall()
            for cr in c_rows:
                if cr["new_path"]:
                    changed_paths.add(path_bytes_to_display(bytes(cr["new_path"]))[0])
                if cr["old_path"]:
                    changed_paths.add(path_bytes_to_display(bytes(cr["old_path"]))[0])

    modified_syms = [
        n for n in nodes.values() if n.file_path in changed_paths
    ]
    modified_ids = {n.id for n in modified_syms}

    inbound_adj: dict[str, list[tuple[str, str, str]]] = {}
    for f_id, t_id, etype, method in edges:
        if etype == "CALLS":
            inbound_adj.setdefault(t_id, []).append((f_id, etype, method))

    affected_callers: dict[str, GraphNode] = {}
    impact_edges: list[GraphEdge] = []
    seen_edges: set[tuple[str, str]] = set()
    queue: deque[tuple[str, int]] = deque((nid, 0) for nid in modified_ids)
    visited = set(modified_ids)

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

    return DetectChangesResult(
        changed_files=sorted(changed_paths),
        directly_modified_symbols=modified_syms[: request.limit],
        affected_callers=list(affected_callers.values())[: request.limit],
        impact_edges=impact_edges,
    )


def execute_get_code_snippet(
    store: Store,
    mirror: GitMirror,
    request: GetCodeSnippetRequest,
) -> GetCodeSnippetResult:
    """Retrieve exact symbol source code by `qualified_name` (or short `name`) and optional `file_path`,
    along with direct callers and callees when `include_neighbors=True`.
    """
    nodes, edges = _load_active_graph(store)
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

    # Prefer exact qualified_name match
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

    return GetCodeSnippetResult(
        node=target_node,
        code=code_text,
        callers=callers[:25],
        callees=callees[:25],
    )
