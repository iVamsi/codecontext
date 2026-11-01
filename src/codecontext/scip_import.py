import bisect
import gzip
import heapq
import io
import sqlite3
import zlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codecontext.deadline import check_deadline
from codecontext.errors import CodeContextError
from codecontext.identity import make_relation_id, path_bytes_to_display
from codecontext.models import SCIPImportReport
from codecontext.store import Store

CALLABLE_SYMBOL_KINDS: frozenset[str] = frozenset(
    {
        "function",
        "async_function",
        "method",
        "async_method",
        "function_signature",
        "method_signature",
    }
)
TYPE_SYMBOL_KINDS: frozenset[str] = frozenset(
    {
        "class",
        "interface",
        "enum",
        "type_alias",
    }
)

SCIP_ROLE_DEFINITION = 0x1
# Applies to both the file on disk and its gunzipped payload, so a gzip bomb cannot exhaust memory.
MAX_SCIP_INDEX_BYTES = 512 * 1024 * 1024


def _too_large_error(what: str) -> CodeContextError:
    return CodeContextError(
        code="INVALID_REQUEST",
        message=f"SCIP index {what} exceeds the {MAX_SCIP_INDEX_BYTES // (1024 * 1024)} MiB limit.",
        retryable=False,
    )


@dataclass(frozen=True)
class SCIPRelationship:
    symbol: str
    is_reference: bool = False
    is_implementation: bool = False
    is_type_definition: bool = False
    is_definition: bool = False


@dataclass(frozen=True)
class SCIPSymbolInfo:
    symbol: str
    relationships: list[SCIPRelationship] = field(default_factory=list)


@dataclass(frozen=True)
class SCIPOccurrence:
    range_vals: list[int]
    symbol: str
    symbol_roles: int = 0


@dataclass(frozen=True)
class SCIPDocument:
    relative_path: str
    occurrences: list[SCIPOccurrence] = field(default_factory=list)
    symbols: list[SCIPSymbolInfo] = field(default_factory=list)


@dataclass(frozen=True)
class SCIPIndex:
    documents: list[SCIPDocument] = field(default_factory=list)
    external_symbols: list[SCIPSymbolInfo] = field(default_factory=list)


def _read_varint(buf: memoryview, pos: int, end: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while pos < end:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return result, pos
        shift += 7
        if shift >= 70:
            raise CodeContextError(
                code="INVALID_REQUEST",
                message="Malformed SCIP protobuf: varint exceeds 64 bits.",
                retryable=False,
            )
    raise CodeContextError(
        code="INVALID_REQUEST",
        message="Malformed SCIP protobuf: truncated varint.",
        retryable=False,
    )


def _iter_fields(
    buf: memoryview,
    start: int,
    end: int,
) -> Iterator[tuple[int, int, int | memoryview]]:
    pos = start
    while pos < end:
        tag, pos = _read_varint(buf, pos, end)
        field_num = tag >> 3
        wire_type = tag & 0x07
        if field_num == 0:
            raise CodeContextError(
                code="INVALID_REQUEST",
                message="Malformed SCIP protobuf: field number 0.",
                retryable=False,
            )
        if wire_type == 0:
            val, pos = _read_varint(buf, pos, end)
            yield field_num, 0, val
        elif wire_type == 1:
            if pos + 8 > end:
                raise CodeContextError(
                    code="INVALID_REQUEST",
                    message="Malformed SCIP protobuf: truncated 64-bit field.",
                    retryable=False,
                )
            yield field_num, 1, buf[pos : pos + 8]
            pos += 8
        elif wire_type == 2:
            length, pos = _read_varint(buf, pos, end)
            if pos + length > end:
                raise CodeContextError(
                    code="INVALID_REQUEST",
                    message="Malformed SCIP protobuf: truncated length-delimited field.",
                    retryable=False,
                )
            yield field_num, 2, buf[pos : pos + length]
            pos += length
        elif wire_type == 5:
            if pos + 4 > end:
                raise CodeContextError(
                    code="INVALID_REQUEST",
                    message="Malformed SCIP protobuf: truncated 32-bit field.",
                    retryable=False,
                )
            yield field_num, 5, buf[pos : pos + 4]
            pos += 4
        else:
            raise CodeContextError(
                code="INVALID_REQUEST",
                message=f"Unsupported protobuf wire type {wire_type} in SCIP index.",
                retryable=False,
            )


def _decode_utf8(sub: memoryview) -> str:
    try:
        return bytes(sub).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CodeContextError(
            code="INVALID_REQUEST",
            message="Malformed UTF-8 string in SCIP protobuf.",
            retryable=False,
        ) from exc


def _decode_packed_int32(sub: memoryview) -> list[int]:
    vals: list[int] = []
    pos = 0
    end = len(sub)
    while pos < end:
        v, pos = _read_varint(sub, pos, end)
        if v >= 0x80000000:
            v = int((v & 0xFFFFFFFF) - 0x100000000) if v <= 0xFFFFFFFFFFFFFFFF else v
        vals.append(v)
    return vals


def _parse_relationship(sub: memoryview) -> SCIPRelationship:
    symbol = ""
    is_ref = False
    is_impl = False
    is_type_def = False
    is_def = False
    for fnum, wtype, payload in _iter_fields(sub, 0, len(sub)):
        if fnum == 1 and wtype == 2 and isinstance(payload, memoryview):
            symbol = _decode_utf8(payload)
        elif fnum == 2 and wtype == 0 and isinstance(payload, int):
            is_ref = bool(payload)
        elif fnum == 3 and wtype == 0 and isinstance(payload, int):
            is_impl = bool(payload)
        elif fnum == 4 and wtype == 0 and isinstance(payload, int):
            is_type_def = bool(payload)
        elif fnum == 5 and wtype == 0 and isinstance(payload, int):
            is_def = bool(payload)
    return SCIPRelationship(
        symbol=symbol,
        is_reference=is_ref,
        is_implementation=is_impl,
        is_type_definition=is_type_def,
        is_definition=is_def,
    )


def _parse_symbol_info(sub: memoryview) -> SCIPSymbolInfo:
    symbol = ""
    rels: list[SCIPRelationship] = []
    for fnum, wtype, payload in _iter_fields(sub, 0, len(sub)):
        if fnum == 1 and wtype == 2 and isinstance(payload, memoryview):
            symbol = _decode_utf8(payload)
        elif fnum == 3 and wtype == 2 and isinstance(payload, memoryview):
            rels.append(_parse_relationship(payload))
    return SCIPSymbolInfo(symbol=symbol, relationships=rels)


def _parse_occurrence(sub: memoryview) -> SCIPOccurrence:
    range_vals: list[int] = []
    symbol = ""
    roles = 0
    for fnum, wtype, payload in _iter_fields(sub, 0, len(sub)):
        if fnum == 1:
            if wtype == 2 and isinstance(payload, memoryview):
                range_vals.extend(_decode_packed_int32(payload))
            elif wtype == 0 and isinstance(payload, int):
                range_vals.append(payload)
        elif fnum == 2 and wtype == 2 and isinstance(payload, memoryview):
            symbol = _decode_utf8(payload)
        elif fnum == 3 and wtype == 0 and isinstance(payload, int):
            roles = payload
    return SCIPOccurrence(range_vals=range_vals, symbol=symbol, symbol_roles=roles)


def _normalize_rel_path(rel_path: str) -> str:
    cleaned = rel_path.replace("\\", "/").strip()
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned.lstrip("/")


def _parse_document(sub: memoryview) -> SCIPDocument:
    rel_path = ""
    occs: list[SCIPOccurrence] = []
    syms: list[SCIPSymbolInfo] = []
    for fnum, wtype, payload in _iter_fields(sub, 0, len(sub)):
        if fnum == 1 and wtype == 2 and isinstance(payload, memoryview):
            rel_path = _normalize_rel_path(_decode_utf8(payload))
        elif fnum == 2 and wtype == 2 and isinstance(payload, memoryview):
            occs.append(_parse_occurrence(payload))
        elif fnum == 3 and wtype == 2 and isinstance(payload, memoryview):
            syms.append(_parse_symbol_info(payload))
    return SCIPDocument(relative_path=rel_path, occurrences=occs, symbols=syms)


def decode_scip_index(raw_bytes: bytes) -> SCIPIndex:
    """Decode a SCIP protobuf binary payload (plain or gzip-compressed) without external dependencies."""
    if len(raw_bytes) > MAX_SCIP_INDEX_BYTES:
        raise _too_large_error("file")
    if raw_bytes[:2] == b"\x1f\x8b":
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(raw_bytes)) as gz:
                raw_bytes = gz.read(MAX_SCIP_INDEX_BYTES + 1)
        except (OSError, EOFError, zlib.error) as exc:
            raise CodeContextError(
                code="INVALID_REQUEST",
                message="Invalid gzip-compressed SCIP file.",
                retryable=False,
            ) from exc
        if len(raw_bytes) > MAX_SCIP_INDEX_BYTES:
            raise _too_large_error("decompressed payload")

    mv = memoryview(raw_bytes)
    docs: list[SCIPDocument] = []
    ext_syms: list[SCIPSymbolInfo] = []
    for fnum, wtype, payload in _iter_fields(mv, 0, len(mv)):
        check_deadline()
        if fnum == 3 and wtype == 2 and isinstance(payload, memoryview):
            docs.append(_parse_document(payload))
        elif fnum == 4 and wtype == 2 and isinstance(payload, memoryview):
            ext_syms.append(_parse_symbol_info(payload))
    return SCIPIndex(documents=docs, external_symbols=ext_syms)


@dataclass(frozen=True)
class _ParsedDescriptor:
    short_name: str
    qualified_suffix: str
    is_callable: bool
    is_type: bool
    is_parameter_or_local: bool


def _parse_scip_symbol_descriptor(scip_symbol: str) -> _ParsedDescriptor:
    """Extract descriptor names and kind hints from a SCIP symbol string."""
    s = scip_symbol.strip()
    if not s or s.startswith("local "):
        return _ParsedDescriptor("", "", False, False, True)

    # SCIP format: `<scheme> <manager> <pkg> <version> <descriptors>`
    # Space inside a token is escaped as double-space `  `; split on single spaces up to 4 times.
    idx = 0
    n = len(s)
    spaces_seen = 0
    desc_start = 0
    while idx < n and spaces_seen < 4:
        if s[idx] == " ":
            if idx + 1 < n and s[idx + 1] == " ":
                idx += 2
                continue
            spaces_seen += 1
            desc_start = idx + 1
        idx += 1
    desc = s[desc_start:].strip() if spaces_seen == 4 else s

    # Parameter `(x)` or type parameter `[T]` descriptors are not top-level/member definitions
    if (desc.endswith(")") and not desc.endswith(").")) or desc.endswith("]"):
        return _ParsedDescriptor("", "", False, False, True)

    # Parse descriptor chain: `ns/`, `Type#`, `term.`, `method(disambig).`, `` `escaped` ``
    parts: list[tuple[str, str]] = []  # (name, suffix_char: '/' | '#' | '.' | '().')
    i = 0
    dn = len(desc)
    while i < dn:
        if desc[i] == "`":
            j = i + 1
            buf_chars: list[str] = []
            while j < dn:
                if desc[j] == "`":
                    if j + 1 < dn and desc[j + 1] == "`":
                        buf_chars.append("`")
                        j += 2
                        continue
                    j += 1
                    break
                buf_chars.append(desc[j])
                j += 1
            ident = "".join(buf_chars)
            i = j
        elif desc[i] in ("(", "["):
            close_ch = ")" if desc[i] == "(" else "]"
            j = desc.find(close_ch, i + 1)
            if j == -1:
                break
            i = j + 1
            if i < dn and desc[i] == ".":
                i += 1
            continue
        else:
            j = i
            while j < dn and desc[j] not in ("/", "#", ".", "(", "[", "`"):
                j += 1
            ident = desc[i:j]
            i = j

        if i < dn and desc[i] == "(":
            close_p = desc.find(")", i + 1)
            if close_p != -1 and close_p + 1 < dn and desc[close_p + 1] == ".":
                parts.append((ident, "()."))
                i = close_p + 2
                continue
        if i < dn and desc[i] in ("/", "#", "."):
            parts.append((ident, desc[i]))
            i += 1
        elif ident:
            parts.append((ident, ""))

    meaningful = [(nm, suf) for nm, suf in parts if nm and suf != "/"]
    if not meaningful:
        meaningful = [(nm, suf) for nm, suf in parts if nm]
    if not meaningful:
        return _ParsedDescriptor("", "", False, False, True)

    last_name, last_suf = meaningful[-1]
    is_callable = last_suf == "()."
    is_type = last_suf == "#"
    qual_chain = [nm for nm, suf in parts if nm and suf != "/"]
    if not qual_chain:
        qual_chain = [last_name]
    if last_name in ("<init>", "`<init>`") and len(qual_chain) >= 2:
        owner = qual_chain[-2]
        return _ParsedDescriptor(owner, ".".join(qual_chain[:-1]), True, False, False)
    return _ParsedDescriptor(
        short_name=last_name,
        qualified_suffix=".".join(qual_chain),
        is_callable=is_callable,
        is_type=is_type,
        is_parameter_or_local=False,
    )


def _match_definition_in_file(
    file_syms: list[dict[str, Any]],
    line_1: int,
    parsed: _ParsedDescriptor,
) -> str | None:
    if not file_syms:
        return None
    short = parsed.short_name
    qual_suf = parsed.qualified_suffix
    ctor_names = {short, "__init__", "constructor", "<init>", "init", "new"} if short else set()

    # 1. Exact start line + matching name (or constructor name)
    exact_line_name = [
        s
        for s in file_syms
        if s["line_start"] == line_1
        and (
            s["name"] == short
            or (parsed.is_callable and s["name"] in ctor_names and qual_suf and s["qualified_name"].startswith(qual_suf))
        )
    ]
    if len(exact_line_name) == 1:
        return str(exact_line_name[0]["id"])
    if exact_line_name:
        for s in exact_line_name:
            if qual_suf and (s["qualified_name"] == qual_suf or s["qualified_name"].endswith("." + qual_suf)):
                return str(s["id"])
        return str(exact_line_name[0]["id"])

    # 2. Enclosing span + matching name/qualified_suffix (handles decorated/annotated declarations)
    span_name = [
        s
        for s in file_syms
        if s["line_start"] <= line_1 <= s["line_end"]
        and short
        and (
            s["name"] == short
            or (qual_suf and (s["qualified_name"] == qual_suf or s["qualified_name"].endswith("." + qual_suf)))
        )
    ]
    if span_name:
        span_name.sort(key=lambda s: (s["line_end"] - s["line_start"], abs(s["line_start"] - line_1)))
        return str(span_name[0]["id"])

    # 3. Unambiguous symbol starting on exact line_1 when not a parameter/local
    if not parsed.is_parameter_or_local:
        exact_line = [s for s in file_syms if s["line_start"] == line_1]
        if len(exact_line) == 1:
            return str(exact_line[0]["id"])

    return None


def _build_enclosing_symbol_lookup(file_syms: list[dict[str, Any]]) -> Callable[[int], str | None]:
    """Return `line -> innermost enclosing symbol id`: smallest line span, then deepest line_start,
    then earliest in `file_syms`.

    Every line_start and line_end + 1 is a boundary, so the containing set is constant between two
    boundaries. One sweep with a heap records the winner per segment; lookups bisect the segments.
    """
    order = sorted(range(len(file_syms)), key=lambda i: file_syms[i]["line_start"])
    boundaries = sorted({s["line_start"] for s in file_syms} | {s["line_end"] + 1 for s in file_syms})
    heap: list[tuple[int, int, int]] = []
    winners: list[str | None] = []
    j = 0
    for boundary in boundaries:
        while j < len(order) and file_syms[order[j]]["line_start"] <= boundary:
            sym = file_syms[order[j]]
            heapq.heappush(heap, (sym["line_end"] - sym["line_start"], -sym["line_start"], order[j]))
            j += 1
        while heap and file_syms[heap[0][2]]["line_end"] < boundary:
            heapq.heappop(heap)
        winners.append(str(file_syms[heap[0][2]]["id"]) if heap else None)

    def lookup(line_1: int) -> str | None:
        k = bisect.bisect_right(boundaries, line_1) - 1
        return winners[k] if k >= 0 else None

    return lookup


def retire_dangling_scip_relations(conn: sqlite3.Connection) -> None:
    """Delete SCIP edges whose endpoint is no longer a symbol of the active snapshot."""
    conn.execute(
        """
        DELETE FROM relations
        WHERE method = 'scip'
          AND (from_id NOT IN (SELECT s.id FROM symbols s JOIN snapshot_files sf ON sf.extraction_id = s.extraction_id)
               OR to_id NOT IN (SELECT s.id FROM symbols s JOIN snapshot_files sf ON sf.extraction_id = s.extraction_id))
        """
    )


def execute_import_scip(store: Store, scip_path: Path) -> SCIPImportReport:
    """Import compiler-exact relationships from a SCIP protobuf index into `relations`."""
    resolved = scip_path.expanduser()
    if not resolved.exists() or not resolved.is_file():
        raise CodeContextError(
            code="INVALID_REQUEST",
            message=f"SCIP index file does not exist: {scip_path}",
            retryable=False,
        )
    try:
        with open(resolved, "rb") as fh:
            raw_bytes = fh.read(MAX_SCIP_INDEX_BYTES + 1)
    except OSError as exc:
        raise CodeContextError(
            code="IO_ERROR",
            message=f"Could not read SCIP index file: {exc}",
            retryable=False,
        ) from exc

    scip_index = decode_scip_index(raw_bytes)

    with store.connect_read() as r_conn:
        meta = r_conn.execute(
            "SELECT repository_id, active_revision, generation FROM store_meta WHERE singleton = 1"
        ).fetchone()
        if not meta:
            raise CodeContextError(
                code="STORE_NOT_INITIALIZED",
                message="Store is not initialized.",
                retryable=False,
            )
        repo_id = str(meta["repository_id"])
        active_rev = str(meta["active_revision"]) if meta["active_revision"] else None

        active_rows = r_conn.execute(
            """
            SELECT s.id, s.kind, s.name, s.qualified_name, s.line_start, s.line_end, sf.path
            FROM symbols s
            JOIN snapshot_files sf ON sf.extraction_id = s.extraction_id
            ORDER BY sf.path ASC, s.line_start ASC, s.byte_start ASC, s.id ASC
            """
        ).fetchall()

    syms_by_path: dict[str, list[dict[str, Any]]] = {}
    sym_by_id: dict[str, dict[str, Any]] = {}
    syms_by_qname: dict[str, list[dict[str, Any]]] = {}
    syms_by_name: dict[str, list[dict[str, Any]]] = {}

    for row in active_rows:
        disp_p = _normalize_rel_path(path_bytes_to_display(bytes(row["path"]))[0])
        sid = str(row["id"])
        info = {
            "id": sid,
            "kind": str(row["kind"]),
            "line_end": int(row["line_end"]),
            "line_start": int(row["line_start"]),
            "name": str(row["name"]),
            "path": disp_p,
            "qualified_name": str(row["qualified_name"]),
        }
        syms_by_path.setdefault(disp_p, []).append(info)
        if sid not in sym_by_id:
            sym_by_id[sid] = info
            syms_by_qname.setdefault(info["qualified_name"], []).append(info)
            syms_by_name.setdefault(info["name"], []).append(info)

    scip_to_sql_id: dict[str, str] = {}
    parsed_cache: dict[str, _ParsedDescriptor] = {}

    def get_parsed(scip_sym: str) -> _ParsedDescriptor:
        cached = parsed_cache.get(scip_sym)
        if cached is None:
            cached = _parse_scip_symbol_descriptor(scip_sym)
            parsed_cache[scip_sym] = cached
        return cached

    # Pass 1: Match definition occurrences by (relative_path, line_start) + descriptor
    for doc in scip_index.documents:
        check_deadline()
        file_syms = syms_by_path.get(doc.relative_path, [])
        if not file_syms:
            continue
        for occ in doc.occurrences:
            if not occ.symbol or (occ.symbol_roles & SCIP_ROLE_DEFINITION) == 0:
                continue
            if not occ.range_vals:
                continue
            line_1 = occ.range_vals[0] + 1
            if line_1 < 1:
                continue
            parsed = get_parsed(occ.symbol)
            matched_id = _match_definition_in_file(file_syms, line_1, parsed)
            if matched_id is not None:
                scip_to_sql_id[occ.symbol] = matched_id

        # Match Document.symbols that had no explicit definition occurrence in `occurrences`
        for sinfo in doc.symbols:
            if not sinfo.symbol or sinfo.symbol in scip_to_sql_id:
                continue
            parsed = get_parsed(sinfo.symbol)
            if parsed.is_parameter_or_local or not parsed.short_name:
                continue
            q_matches = [
                s
                for s in file_syms
                if parsed.qualified_suffix
                and (
                    s["qualified_name"] == parsed.qualified_suffix
                    or s["qualified_name"].endswith("." + parsed.qualified_suffix)
                )
            ]
            if len(q_matches) == 1:
                scip_to_sql_id[sinfo.symbol] = str(q_matches[0]["id"])
                continue
            n_matches = [s for s in file_syms if s["name"] == parsed.short_name]
            if len(n_matches) == 1:
                scip_to_sql_id[sinfo.symbol] = str(n_matches[0]["id"])

    def resolve_scip_symbol(scip_sym: str) -> str | None:
        if scip_sym in scip_to_sql_id:
            return scip_to_sql_id[scip_sym]
        parsed = get_parsed(scip_sym)
        if parsed.is_parameter_or_local or not parsed.short_name:
            return None
        if parsed.qualified_suffix and parsed.qualified_suffix in syms_by_qname:
            cands = syms_by_qname[parsed.qualified_suffix]
            if len(cands) == 1:
                scip_to_sql_id[scip_sym] = str(cands[0]["id"])
                return scip_to_sql_id[scip_sym]
        if parsed.qualified_suffix and "." in parsed.qualified_suffix:
            suffix_dot = "." + parsed.qualified_suffix
            cands = [
                s
                for s in syms_by_name.get(parsed.short_name, [])
                if s["qualified_name"].endswith(suffix_dot)
            ]
            if len(cands) == 1:
                scip_to_sql_id[scip_sym] = str(cands[0]["id"])
                return scip_to_sql_id[scip_sym]
        return None

    # Pass 2: Collect compiler-exact edges from occurrences and SymbolInformation.relationships
    edges_to_insert: set[tuple[str, str, str]] = set()  # (from_id, to_id, relation_kind)

    for doc in scip_index.documents:
        check_deadline()
        file_syms = syms_by_path.get(doc.relative_path, [])
        find_enclosing = _build_enclosing_symbol_lookup(file_syms)
        for occ in doc.occurrences:
            if not occ.symbol or (occ.symbol_roles & SCIP_ROLE_DEFINITION) != 0:
                continue
            if not occ.range_vals or not file_syms:
                continue
            tgt_id = resolve_scip_symbol(occ.symbol)
            if tgt_id is None:
                continue
            line_1 = occ.range_vals[0] + 1
            enc_id = find_enclosing(line_1)
            if enc_id is None or enc_id == tgt_id:
                continue
            tgt_info = sym_by_id.get(tgt_id)
            if tgt_info is None:
                continue
            parsed = get_parsed(occ.symbol)
            tkind = tgt_info["kind"]
            if tkind in CALLABLE_SYMBOL_KINDS:
                edges_to_insert.add((enc_id, tgt_id, "calls_candidate"))
            elif tkind in TYPE_SYMBOL_KINDS:
                if parsed.is_callable:
                    edges_to_insert.add((enc_id, tgt_id, "calls_candidate"))
                else:
                    edges_to_insert.add((enc_id, tgt_id, "references_type"))

        for sinfo in doc.symbols:
            src_id = resolve_scip_symbol(sinfo.symbol)
            if src_id is None:
                continue
            src_info = sym_by_id.get(src_id)
            if src_info is None:
                continue
            for rel in sinfo.relationships:
                if not rel.symbol:
                    continue
                tgt_id = resolve_scip_symbol(rel.symbol)
                if tgt_id is None or tgt_id == src_id:
                    continue
                tgt_info = sym_by_id.get(tgt_id)
                if tgt_info is None:
                    continue
                if rel.is_implementation:
                    if src_info["kind"] in CALLABLE_SYMBOL_KINDS or tgt_info["kind"] in CALLABLE_SYMBOL_KINDS:
                        edges_to_insert.add((src_id, tgt_id, "overrides"))
                    else:
                        edges_to_insert.add((src_id, tgt_id, "implements"))
                if rel.is_type_definition:
                    edges_to_insert.add((src_id, tgt_id, "references_type"))
                elif rel.is_reference:
                    if tgt_info["kind"] in CALLABLE_SYMBOL_KINDS:
                        edges_to_insert.add((src_id, tgt_id, "calls_candidate"))
                    elif tgt_info["kind"] in TYPE_SYMBOL_KINDS:
                        edges_to_insert.add((src_id, tgt_id, "references_type"))

    for ext_sinfo in scip_index.external_symbols:
        src_id = resolve_scip_symbol(ext_sinfo.symbol)
        if src_id is None:
            continue
        src_info = sym_by_id.get(src_id)
        if src_info is None:
            continue
        for rel in ext_sinfo.relationships:
            tgt_id = resolve_scip_symbol(rel.symbol)
            if tgt_id is None or tgt_id == src_id:
                continue
            tgt_info = sym_by_id.get(tgt_id)
            if tgt_info is None:
                continue
            if rel.is_implementation:
                if src_info["kind"] in CALLABLE_SYMBOL_KINDS or tgt_info["kind"] in CALLABLE_SYMBOL_KINDS:
                    edges_to_insert.add((src_id, tgt_id, "overrides"))
                else:
                    edges_to_insert.add((src_id, tgt_id, "implements"))

    # Edges from symbols in the imported documents are replaced wholesale; other documents keep theirs.
    replaced_from_ids = {s["id"] for doc in scip_index.documents for s in syms_by_path.get(doc.relative_path, [])}
    relations_added = 0
    with store.write_transaction(bump_generation=True) as conn:
        prior = {
            str(row["id"]): (str(row["from_id"]), str(row["state"]))
            for row in conn.execute("SELECT id, from_id, state FROM relations WHERE method = 'scip'")
        }
        imported_ids: set[str] = set()
        for from_id, to_id, rel_kind in sorted(edges_to_insert):
            check_deadline()
            rel_id = make_relation_id(
                from_kind="symbol",
                from_id=from_id,
                to_kind="symbol",
                to_id=to_id,
                relation_kind=rel_kind,
                method="scip",
            )
            imported_ids.add(rel_id)
            if rel_id not in prior or prior[rel_id][1] != "active":
                relations_added += 1
            conn.execute(
                """
                INSERT INTO relations (
                    id, from_kind, from_id, to_kind, to_id, relation_kind, method, mapping, evidence_id, details_json, state
                ) VALUES (?, 'symbol', ?, 'symbol', ?, ?, 'scip', 'exact', NULL, '{}', 'active')
                ON CONFLICT(id) DO UPDATE SET state = 'active', mapping = 'exact'
                """,
                (rel_id, from_id, to_id, rel_kind),
            )
        conn.executemany(
            "DELETE FROM relations WHERE id = ?",
            [
                (rel_id,)
                for rel_id, (from_id, _state) in prior.items()
                if from_id in replaced_from_ids and rel_id not in imported_ids
            ],
        )
        retire_dangling_scip_relations(conn)

    with store.connect_read() as r_conn:
        gen_row = r_conn.execute("SELECT generation FROM store_meta WHERE singleton = 1").fetchone()
        new_generation = int(gen_row["generation"]) if gen_row else 0

    return SCIPImportReport(
        repository_id=repo_id,
        revision=active_rev,
        generation=new_generation,
        documents_processed=len(scip_index.documents),
        symbols_matched=len(set(scip_to_sql_id.values())),
        relations_added=relations_added,
    )
