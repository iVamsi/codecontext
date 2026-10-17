import re
from collections import deque
from typing import Annotated, Any, Literal

from pydantic import Field

from codecontext.errors import CodeContextError
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
        raw_nodes[sid] = {
            "file_path": disp_p,
            "id": sid,
            "inbound": 0,
            "kind": str(r["kind"]),
            "label": _kind_to_label(str(r["kind"])),
            "language": str(r["language"]),
            "line_end": int(r["line_end"]),
            "line_start": int(r["line_start"]),
            "name": str(r["name"]),
            "outbound": 0,
            "qualified_name": str(r["qualified_name"]),
            "signature": str(r["signature"]),
        }

    edges: list[tuple[str, str, str, str]] = []
    for rr in rel_rows:
        f_id = str(rr["from_id"])
        t_id = str(rr["to_id"])
        if f_id not in raw_nodes or t_id not in raw_nodes:
            continue
        rkind = str(rr["relation_kind"])
        edge_type = "CALLS" if rkind == "calls_candidate" else ("CONTAINS" if rkind == "contains" else rkind.upper())
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
    except re.error as exc:
        raise CodeContextError(
            code="INVALID_REQUEST",
            message=f"Invalid regex pattern in search_graph: {exc}",
            retryable=False,
        ) from exc

    # Recompute degrees if relationship != 'CALLS'
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
        # Substring/regex fallback
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
    r"(?:\s+LIMIT\s+(?P<limit>\d+))?\s*$",
    re.IGNORECASE | re.DOTALL,
)

CYPHER_EDGE_RE = re.compile(
    r"^\s*MATCH\s*\(\s*(?P<v1>[A-Za-z_]\w*)(?::(?P<l1>[A-Za-z_]\w*))?\s*(?:\{(?P<p1>[^}]*)\})?\s*\)\s*"
    r"-\s*\[\s*(?P<relvar>[A-Za-z_]\w*)?(?::(?P<reltype>[A-Za-z_]\w*))?\s*(?:\{(?P<relprops>[^}]*)\})?\s*\]\s*->\s*"
    r"\(\s*(?P<v2>[A-Za-z_]\w*)(?::(?P<l2>[A-Za-z_]\w*))?\s*(?:\{(?P<p2>[^}]*)\})?\s*\)\s*"
    r"(?:WHERE\s+(?P<where>.+?)\s*)?"
    r"RETURN\s+(?P<ret>.+?)"
    r"(?:\s+LIMIT\s+(?P<limit>\d+))?\s*$",
    re.IGNORECASE | re.DOTALL,
)

PROP_PAIR_RE = re.compile(r"([A-Za-z_]\w*)\s*:\s*['\"]([^'\"]*)['\"]")
WHERE_COND_RE = re.compile(
    r"(?P<var>[A-Za-z_]\w*)\.(?P<prop>[A-Za-z_]\w*)\s*(?P<op>=~|=|!=|CONTAINS|STARTS\s+WITH)\s*['\"](?P<val>[^'\"]*)['\"]",
    re.IGNORECASE,
)


def _node_attr(node: GraphNode, prop: str) -> Any:
    mapping = {
        "file": node.file_path,
        "file_path": node.file_path,
        "filePath": node.file_path,
        "id": node.id,
        "inbound_degree": node.inbound_degree,
        "kind": node.label,
        "label": node.label,
        "language": node.language,
        "line_end": node.line_end,
        "line_start": node.line_start,
        "name": node.name,
        "outbound_degree": node.outbound_degree,
        "qualified_name": node.qualified_name,
        "signature": node.signature,
    }
    return mapping.get(prop)


def _matches_inline_props(node: GraphNode, props_str: str | None) -> bool:
    if not props_str:
        return True
    for m in PROP_PAIR_RE.finditer(props_str):
        k, v = m.group(1), m.group(2)
        if str(_node_attr(node, k) or "") != v:
            return False
    return True


def _matches_label(node: GraphNode, label: str | None) -> bool:
    if not label:
        return True
    low = label.lower()
    if low == "function" and node.label in ("Function", "Method"):
        return True
    return node.label.lower() == low


def _matches_where(bindings: dict[str, Any], where_clause: str | None) -> bool:
    if not where_clause:
        return True
    parts = re.split(r"\s+AND\s+", where_clause.strip(), flags=re.IGNORECASE)
    for part in parts:
        m = WHERE_COND_RE.search(part)
        if not m:
            continue
        var = m.group("var")
        prop = m.group("prop")
        op = m.group("op").upper()
        val = m.group("val")
        obj = bindings.get(var)
        if obj is None:
            return False
        actual = str(
            _node_attr(obj, prop)
            if isinstance(obj, GraphNode)
            else obj.get(prop, "")
        )
        if op == "=" and actual != val:
            return False
        if op == "!=" and actual == val:
            return False
        if op == "CONTAINS" and val not in actual:
            return False
        if op.startswith("STARTS") and not actual.startswith(val):
            return False
        if op == "=~":
            try:
                if not re.search(val, actual):
                    return False
            except re.error:
                return False
    return True


def _project_return_row(bindings: dict[str, Any], return_exprs: list[str]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for expr in return_exprs:
        clean = expr.strip()
        # Handle optional `AS alias`
        as_parts = re.split(r"\s+AS\s+", clean, flags=re.IGNORECASE)
        target_expr = as_parts[0].strip()
        col_name = as_parts[1].strip() if len(as_parts) > 1 else target_expr
        if "." in target_expr:
            v_name, p_name = target_expr.split(".", 1)
            obj = bindings.get(v_name.strip())
            if isinstance(obj, GraphNode):
                row[col_name] = _node_attr(obj, p_name.strip())
            elif isinstance(obj, dict):
                row[col_name] = obj.get(p_name.strip())
            else:
                row[col_name] = None
        else:
            obj = bindings.get(target_expr)
            if isinstance(obj, GraphNode):
                row[col_name] = obj.model_dump()
            else:
                row[col_name] = obj
    return row


def execute_query_graph(store: Store, request: QueryGraphRequest) -> QueryGraphResult:
    """Execute a read-only Cypher pattern query (`MATCH (n) ... RETURN ...` or `MATCH (a)-[r]->(b) ... RETURN ...`)
    over the active snapshot symbol/relation graph, capped at 200 rows.
    """
    nodes, edges = _load_active_graph(store)
    q = request.query.strip().rstrip(";")

    m_edge = CYPHER_EDGE_RE.match(q)
    if m_edge:
        v1 = m_edge.group("v1")
        l1 = m_edge.group("l1")
        p1 = m_edge.group("p1")
        relvar = m_edge.group("relvar") or "r"
        reltype = m_edge.group("reltype")
        relprops = m_edge.group("relprops")
        v2 = m_edge.group("v2")
        l2 = m_edge.group("l2")
        p2 = m_edge.group("p2")
        where_c = m_edge.group("where")
        ret_c = m_edge.group("ret")
        lim_s = m_edge.group("limit")

        cap = min(request.limit, int(lim_s) if lim_s else request.limit, 200)
        ret_cols = [c.strip() for c in ret_c.split(",") if c.strip()]

        req_edge_type: str | None = None
        if reltype and reltype.upper() != "CODERELATION":
            req_edge_type = reltype.upper()
        elif relprops:
            for pm in PROP_PAIR_RE.finditer(relprops):
                if pm.group(1) == "type":
                    req_edge_type = pm.group(2).upper()

        rows: list[dict[str, Any]] = []
        truncated = False
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
                relvar: {"confidence": 1.0, "method": method, "type": etype},
            }
            if not _matches_where(bindings, where_c):
                continue
            if len(rows) >= cap:
                truncated = True
                break
            rows.append(_project_return_row(bindings, ret_cols))

        return QueryGraphResult(
            columns=ret_cols,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
        )

    m_node = CYPHER_NODE_RE.match(q)
    if m_node:
        var = m_node.group("var")
        label = m_node.group("label")
        props = m_node.group("props")
        where_c = m_node.group("where")
        ret_c = m_node.group("ret")
        lim_s = m_node.group("limit")

        cap = min(request.limit, int(lim_s) if lim_s else request.limit, 200)
        ret_cols = [c.strip() for c in ret_c.split(",") if c.strip()]

        rows = []
        truncated = False
        for node in nodes.values():
            if not _matches_label(node, label) or not _matches_inline_props(node, props):
                continue
            bindings = {var: node}
            if not _matches_where(bindings, where_c):
                continue
            if len(rows) >= cap:
                truncated = True
                break
            rows.append(_project_return_row(bindings, ret_cols))

        return QueryGraphResult(
            columns=ret_cols,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
        )

    raise CodeContextError(
        code="INVALID_REQUEST",
        message="Unsupported Cypher syntax. Expected `MATCH (n[:Label]) [WHERE ...] RETURN ... [LIMIT N]` or `MATCH (a)-[r[:TYPE]]->(b) [WHERE ...] RETURN ... [LIMIT N]`.",
        retryable=False,
    )
