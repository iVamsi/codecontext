import importlib.metadata
import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tree_sitter_c
import tree_sitter_cpp
import tree_sitter_go
import tree_sitter_java
import tree_sitter_javascript
import tree_sitter_kotlin
import tree_sitter_python
import tree_sitter_rust
import tree_sitter_typescript
from tree_sitter import Language, Node, Parser, Query

from codecontext.git import GitMirror
from codecontext.identity import (
    byte_span_to_lines,
    canonical_json,
    make_extraction_id,
    make_id,
    make_relation_id,
    make_sym_id,
    path_bytes_to_display,
    sha256_bytes,
)
from codecontext.store import Store

QUERIES_DIR = Path(__file__).parent / "queries"
MAX_PARSE_BYTES = 2 * 1024 * 1024  # 2 MiB
CHUNK_MAX_BYTES = 8192
CHUNK_OVERLAP_BYTES = 512

EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
}

DOC_AND_CONFIG_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".md",
        ".rst",
        ".txt",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".xml",
        ".ini",
        ".cfg",
        ".conf",
        ".properties",
        ".pro",
        ".gradle",
        ".sh",
        ".bash",
        ".zsh",
        ".sql",
        ".css",
        ".html",
        ".proto",
        ".scm",
        ".lock",
    }
)

DOC_AND_CONFIG_FILENAMES: frozenset[str] = frozenset(
    {
        "dockerfile",
        "makefile",
        "gradlew",
        "gradlew.bat",
        "readme",
        "license",
        "copying",
        "changelog",
        "authors",
        "notice",
        ".gitignore",
        ".gitattributes",
        ".editorconfig",
    }
)

CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
WORD_RE = re.compile(r"\w+", re.UNICODE)


def _pkg_ver(dist_name: str) -> str:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


def _load_languages_and_queries() -> tuple[dict[str, Language], dict[str, Query], dict[str, str]]:
    langs = {
        "python": Language(tree_sitter_python.language()),
        "javascript": Language(tree_sitter_javascript.language()),
        "typescript": Language(tree_sitter_typescript.language_typescript()),
        "tsx": Language(tree_sitter_typescript.language_tsx()),
        "kotlin": Language(tree_sitter_kotlin.language()),
        "java": Language(tree_sitter_java.language()),
        "go": Language(tree_sitter_go.language()),
        "rust": Language(tree_sitter_rust.language()),
        "c": Language(tree_sitter_c.language()),
        "cpp": Language(tree_sitter_cpp.language()),
    }
    ts_ver = _pkg_ver("tree-sitter")
    lang_pkg_vers = {
        "python": _pkg_ver("tree-sitter-python"),
        "javascript": _pkg_ver("tree-sitter-javascript"),
        "typescript": _pkg_ver("tree-sitter-typescript"),
        "tsx": _pkg_ver("tree-sitter-typescript"),
        "kotlin": _pkg_ver("tree-sitter-kotlin"),
        "java": _pkg_ver("tree-sitter-java"),
        "go": _pkg_ver("tree-sitter-go"),
        "rust": _pkg_ver("tree-sitter-rust"),
        "c": _pkg_ver("tree-sitter-c"),
        "cpp": _pkg_ver("tree-sitter-cpp"),
    }
    queries: dict[str, Query] = {}
    fingerprints: dict[str, str] = {}
    for name, lang in langs.items():
        q_file = QUERIES_DIR / f"{name}.scm"
        q_bytes = q_file.read_bytes()
        q_hash = sha256_bytes(q_bytes)[:16]
        queries[name] = Query(lang, q_bytes.decode("utf-8"))
        walker_version = 3 if name == "kotlin" else 2
        fingerprints[name] = f"{name}:ts{ts_ver}:g{lang_pkg_vers[name]}:q{q_hash}:walker{walker_version}"
    return langs, queries, fingerprints


LANGUAGES, COMPILED_QUERIES, EXTRACTOR_FINGERPRINTS = _load_languages_and_queries()


def get_extractor_fingerprint(language: str) -> str:
    return EXTRACTOR_FINGERPRINTS[language]


def detect_language_for_path(path_bytes: bytes) -> str | None:
    disp, _ = path_bytes_to_display(path_bytes)
    lower_name = disp.rsplit("/", 1)[-1].lower()
    dot_idx = lower_name.rfind(".")
    if dot_idx != -1:
        ext = lower_name[dot_idx:]
        if ext in EXTENSION_TO_LANGUAGE:
            return EXTENSION_TO_LANGUAGE[ext]
    return None


def is_indexable_text_path(path_bytes: bytes) -> bool:
    disp, _ = path_bytes_to_display(path_bytes)
    lower_name = disp.rsplit("/", 1)[-1].lower()
    if lower_name in DOC_AND_CONFIG_FILENAMES or lower_name.startswith("dockerfile."):
        return True
    dot_idx = lower_name.rfind(".")
    if dot_idx == -1:
        # Extensionless UTF-8 text is indexed per contract §4
        return True
    ext = lower_name[dot_idx:]
    return ext in DOC_AND_CONFIG_EXTENSIONS or ext in EXTENSION_TO_LANGUAGE


def is_lfs_pointer(raw_bytes: bytes) -> bool:
    if len(raw_bytes) > 1024:
        return False
    return raw_bytes.startswith(b"version https://git-lfs.github.com/spec/v1") and b"\noid sha256:" in raw_bytes


def normalize_search_text(text: str) -> str:
    """Normalize search text: Unicode NFKC, casefold, split camelCase and snake_case,
    split path separators and punctuation, retaining original identifier alongside split terms.
    """
    if not text:
        return ""
    nfkc = unicodedata.normalize("NFKC", text)
    raw_words = WORD_RE.findall(nfkc)
    out_tokens: list[str] = []
    for word in raw_words:
        folded = word.casefold()
        if folded:
            out_tokens.append(folded)
        # Split camelCase and snake_case
        camel_parts = CAMEL_SPLIT_RE.split(word)
        sub_parts: list[str] = []
        for cp in camel_parts:
            sub_parts.extend(p for p in cp.split("_") if p)
        if len(sub_parts) > 1:
            for sp in sub_parts:
                sp_f = sp.casefold()
                if sp_f and sp_f != folded:
                    out_tokens.append(sp_f)
    return " ".join(out_tokens)


@dataclass(frozen=True)
class TextChunk:
    ordinal: int
    byte_start: int
    byte_end: int
    text: str


def chunk_utf8_bytes(
    raw_bytes: bytes,
    *,
    max_bytes: int = CHUNK_MAX_BYTES,
    overlap_bytes: int = CHUNK_OVERLAP_BYTES,
) -> list[TextChunk]:
    """Chunk UTF-8 bytes into <= max_bytes slices with overlap_bytes overlap on valid UTF-8 codepoint boundaries."""
    if not raw_bytes:
        return [TextChunk(ordinal=0, byte_start=0, byte_end=0, text="")]
    n = len(raw_bytes)
    if n <= max_bytes:
        return [
            TextChunk(
                ordinal=0,
                byte_start=0,
                byte_end=n,
                text=raw_bytes.decode("utf-8", errors="strict"),
            )
        ]

    chunks: list[TextChunk] = []
    start = 0
    ordinal = 0
    while start < n:
        end = min(start + max_bytes, n)
        if end < n:
            # Back up if inside a UTF-8 continuation byte (0b10xxxxxx)
            while end > start and (raw_bytes[end] & 0xC0) == 0x80:
                end -= 1
        chunk_text = raw_bytes[start:end].decode("utf-8", errors="strict")
        chunks.append(
            TextChunk(
                ordinal=ordinal,
                byte_start=start,
                byte_end=end,
                text=chunk_text,
            )
        )
        ordinal += 1
        if end >= n:
            break
        next_start = max(start + 1, end - overlap_bytes)
        while next_start < end and (raw_bytes[next_start] & 0xC0) == 0x80:
            next_start += 1
        start = next_start
    return chunks


@dataclass(frozen=True)
class ExtractedSymbol:
    id: str
    parent_symbol_id: str | None
    kind: str
    name: str
    qualified_name: str
    byte_start: int
    byte_end: int
    body_byte_start: int | None
    body_byte_end: int | None
    line_start: int
    line_end: int
    signature: str
    docstring: str | None = None


@dataclass(frozen=True)
class ExtractedOccurrence:
    id: str
    enclosing_symbol_id: str | None
    kind: str
    text: str
    byte_start: int
    byte_end: int
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateRelation:
    from_symbol_id: str
    to_symbol_id: str
    relation_kind: str
    method: str


@dataclass(frozen=True)
class ParseResult:
    extraction_id: str
    blob_oid: str
    language: str
    extractor_fingerprint: str
    state: str  # 'complete', 'partial', 'unsupported', 'oversized', 'binary', 'error'
    diagnostics: list[dict[str, Any]]
    symbols: list[ExtractedSymbol]
    occurrences: list[ExtractedOccurrence]
    relations: list[CandidateRelation]


def _collect_error_ranges(node: Node, out: list[tuple[int, int]]) -> None:
    if node.type == "ERROR" or node.is_missing:
        out.append((node.start_byte, node.end_byte))
    for child in node.children:
        _collect_error_ranges(child, out)


def _ranges_intersect(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    if a_start == a_end:
        return b_start <= a_start <= b_end
    if b_start == b_end:
        return a_start <= b_start <= a_end
    return max(a_start, b_start) < min(a_end, b_end)


def _header_intersects_errors(
    decl_start: int,
    header_end: int,
    name_node: Node | None,
    error_ranges: list[tuple[int, int]],
) -> bool:
    if name_node is None or name_node.is_missing or name_node.has_error:
        return True
    for err_s, err_e in error_ranges:
        if _ranges_intersect(decl_start, header_end, err_s, err_e):
            return True
        if _ranges_intersect(name_node.start_byte, name_node.end_byte, err_s, err_e):
            return True
    return False


def _extract_signature(raw_bytes: bytes, decl_start: int, body_start: int | None, decl_end: int) -> str:
    end = body_start if body_start is not None and body_start > decl_start else decl_end
    sig_bytes = raw_bytes[decl_start:end]
    sig_str = sig_bytes.decode("utf-8", errors="replace").strip()
    if sig_str.endswith(":") or sig_str.endswith("{"):
        sig_str = sig_str[:-1].strip()
    lines = [line.strip() for line in sig_str.splitlines() if line.strip()]
    joined = " ".join(lines)
    return joined[:4000]


def _extract_python_docstring(body_node: Node | None, raw_bytes: bytes) -> tuple[str, int, int] | None:
    if body_node is None:
        return None
    for child in body_node.named_children:
        if child.type == "comment":
            continue
        if child.type == "expression_statement" and child.named_Child_count if hasattr(child, "named_Child_count") else len(child.named_children) >= 1:
            expr = child.named_children[0]
            if expr.type == "string":
                raw_doc = raw_bytes[expr.start_byte : expr.end_byte].decode("utf-8", errors="replace")
                return (raw_doc, expr.start_byte, expr.end_byte)
        break
    return None


def parse_blob(
    blob_bytes: bytes,
    language: str,
    extractor_fingerprint: str | None = None,
    *,
    blob_oid: str | None = None,
) -> ParseResult:
    """Parse a unique supported blob once per (blob_oid, language, extractor_fingerprint)."""
    fp = extractor_fingerprint or get_extractor_fingerprint(language)
    actual_oid = blob_oid or sha256_bytes(blob_bytes)
    ext_id = make_extraction_id(blob_oid=actual_oid, language=language, extractor_fingerprint=fp)

    if len(blob_bytes) > MAX_PARSE_BYTES:
        return ParseResult(
            extraction_id=ext_id,
            blob_oid=actual_oid,
            language=language,
            extractor_fingerprint=fp,
            state="oversized",
            diagnostics=[{"code": "BLOB_TOO_LARGE", "size": len(blob_bytes)}],
            symbols=[],
            occurrences=[],
            relations=[],
        )

    if b"\x00" in blob_bytes:
        return ParseResult(
            extraction_id=ext_id,
            blob_oid=actual_oid,
            language=language,
            extractor_fingerprint=fp,
            state="binary",
            diagnostics=[{"code": "BINARY_CONTENT"}],
            symbols=[],
            occurrences=[],
            relations=[],
        )

    try:
        blob_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        return ParseResult(
            extraction_id=ext_id,
            blob_oid=actual_oid,
            language=language,
            extractor_fingerprint=fp,
            state="error",
            diagnostics=[{"code": "TEXT_DECODE_FAILED", "offset": exc.start}],
            symbols=[],
            occurrences=[],
            relations=[],
        )

    if language not in LANGUAGES:
        return ParseResult(
            extraction_id=ext_id,
            blob_oid=actual_oid,
            language=language,
            extractor_fingerprint=fp,
            state="unsupported",
            diagnostics=[{"code": "UNSUPPORTED_SYNTAX", "language": language}],
            symbols=[],
            occurrences=[],
            relations=[],
        )

    ts_lang = LANGUAGES[language]
    parser = Parser(ts_lang)
    tree = parser.parse(blob_bytes)
    if tree is None or tree.root_node is None:
        return ParseResult(
            extraction_id=ext_id,
            blob_oid=actual_oid,
            language=language,
            extractor_fingerprint=fp,
            state="error",
            diagnostics=[{"code": "PARSE_ERROR", "message": "Parser returned no syntax tree"}],
            symbols=[],
            occurrences=[],
            relations=[],
        )

    root = tree.root_node
    error_ranges: list[tuple[int, int]] = []
    _collect_error_ranges(root, error_ranges)
    diagnostics = [
        {"code": "PARSE_ERROR", "byte_start": s, "byte_end": e} for s, e in error_ranges[:50]
    ]

    symbols: list[ExtractedSymbol] = []
    occurrences: list[ExtractedOccurrence] = []
    relations: list[CandidateRelation] = []

    def add_occurrence(
        enclosing_sym_id: str | None,
        occ_kind: str,
        b_start: int,
        b_end: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        text = blob_bytes[b_start:b_end].decode("utf-8", errors="replace").strip()[:1000]
        if not text:
            return
        occ_id = make_id(
            "ext_",
            {
                "byte_end": b_end,
                "byte_start": b_start,
                "extraction_id": ext_id,
                "kind": occ_kind,
                "text": text,
            },
        )
        occurrences.append(
            ExtractedOccurrence(
                id=occ_id,
                enclosing_symbol_id=enclosing_sym_id,
                kind=occ_kind,
                text=text,
                byte_start=b_start,
                byte_end=b_end,
                details=details or {},
            )
        )

    def add_symbol(
        *,
        parent_id: str | None,
        scope_names: list[str],
        sym_kind: str,
        name_node: Node | None,
        decl_start: int,
        decl_end: int,
        body_node: Node | None,
        header_end_override: int | None = None,
    ) -> ExtractedSymbol | None:
        header_end = (
            header_end_override
            if header_end_override is not None
            else (body_node.start_byte if body_node is not None else decl_end)
        )
        if _header_intersects_errors(decl_start, header_end, name_node, error_ranges):
            return None
        assert name_node is not None
        name_str = blob_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8", errors="replace").strip()
        if not name_str:
            return None
        qual_name = ".".join([*scope_names, name_str])
        l_start, l_end = byte_span_to_lines(blob_bytes, decl_start, decl_end)
        if l_start is None or l_end is None:
            l_start, l_end = 1, 1

        body_s = body_node.start_byte if body_node is not None else None
        body_e = body_node.end_byte if body_node is not None else None
        if body_s is not None and body_e is not None:
            if body_s < decl_start or body_e > decl_end or body_e < body_s:
                body_s, body_e = None, None

        sig = _extract_signature(blob_bytes, decl_start, body_s, decl_end)
        sym_id = make_sym_id(
            blob_oid=actual_oid,
            language=language,
            extractor_fingerprint=fp,
            byte_start=decl_start,
            byte_end=decl_end,
            kind=sym_kind,
            qualified_name=qual_name,
        )

        doc_str: str | None = None
        if language == "python" and body_node is not None:
            doc_info = _extract_python_docstring(body_node, blob_bytes)
            if doc_info is not None:
                doc_str, doc_s, doc_e = doc_info
                add_occurrence(sym_id, "docstring", doc_s, doc_e)

        sym = ExtractedSymbol(
            id=sym_id,
            parent_symbol_id=parent_id,
            kind=sym_kind,
            name=name_str,
            qualified_name=qual_name,
            byte_start=decl_start,
            byte_end=decl_end,
            body_byte_start=body_s,
            body_byte_end=body_e,
            line_start=l_start,
            line_end=l_end,
            signature=sig,
            docstring=doc_str,
        )
        symbols.append(sym)
        if parent_id is not None:
            relations.append(
                CandidateRelation(
                    from_symbol_id=parent_id,
                    to_symbol_id=sym_id,
                    relation_kind="contains",
                    method="ast_containment",
                )
            )
        return sym

    def walk_python(node: Node, parent_sym: ExtractedSymbol | None, scope: list[str]) -> None:
        if node.type == "decorated_definition":
            inner = node.child_by_field_name("definition")
            if inner is not None and inner.type in ("function_definition", "class_definition"):
                name_n = inner.child_by_field_name("name")
                body_n = inner.child_by_field_name("body")
                is_async = any(c.type == "async" for c in inner.children)
                if inner.type == "class_definition":
                    skind = "class"
                elif parent_sym is not None and parent_sym.kind == "class":
                    skind = "async_method" if is_async else "method"
                else:
                    skind = "async_function" if is_async else "function"
                sym = add_symbol(
                    parent_id=parent_sym.id if parent_sym else None,
                    scope_names=scope,
                    sym_kind=skind,
                    name_node=name_n,
                    decl_start=node.start_byte,
                    decl_end=node.end_byte,
                    body_node=body_n,
                )
                if sym is not None and body_n is not None:
                    for ch in body_n.children:
                        walk_python(ch, sym, [*scope, sym.name])
                return

        if node.type in ("function_definition", "class_definition"):
            name_n = node.child_by_field_name("name")
            body_n = node.child_by_field_name("body")
            is_async = any(c.type == "async" for c in node.children)
            if node.type == "class_definition":
                skind = "class"
            elif parent_sym is not None and parent_sym.kind == "class":
                skind = "async_method" if is_async else "method"
            else:
                skind = "async_function" if is_async else "function"
            sym = add_symbol(
                parent_id=parent_sym.id if parent_sym else None,
                scope_names=scope,
                sym_kind=skind,
                name_node=name_n,
                decl_start=node.start_byte,
                decl_end=node.end_byte,
                body_node=body_n,
            )
            if sym is not None and body_n is not None:
                for ch in body_n.children:
                    walk_python(ch, sym, [*scope, sym.name])
            return

        if node.type == "expression_statement" and parent_sym is None:
            # Check module-level assignment
            for ch in node.named_children:
                if ch.type == "assignment":
                    left = ch.child_by_field_name("left")
                    right = ch.child_by_field_name("right")
                    if left is not None and left.type == "identifier":
                        add_symbol(
                            parent_id=None,
                            scope_names=scope,
                            sym_kind="variable",
                            name_node=left,
                            decl_start=node.start_byte,
                            decl_end=node.end_byte,
                            body_node=right,
                            header_end_override=left.end_byte,
                        )

        if node.type == "assignment" and parent_sym is None and (node.parent is None or node.parent.type != "expression_statement"):
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if left is not None and left.type == "identifier":
                add_symbol(
                    parent_id=None,
                    scope_names=scope,
                    sym_kind="variable",
                    name_node=left,
                    decl_start=node.start_byte,
                    decl_end=node.end_byte,
                    body_node=right,
                    header_end_override=left.end_byte,
                )

        if node.type == "import_statement":
            add_occurrence(
                parent_sym.id if parent_sym else None,
                "import",
                node.start_byte,
                node.end_byte,
            )
        elif node.type == "import_from_statement":
            has_rel = any(c.type == "relative_import" for c in node.children)
            add_occurrence(
                parent_sym.id if parent_sym else None,
                "relative_import" if has_rel else "import",
                node.start_byte,
                node.end_byte,
            )
        elif node.type == "call":
            fn_node = node.child_by_field_name("function")
            if fn_node is not None:
                raw_callee = blob_bytes[fn_node.start_byte : fn_node.end_byte].decode("utf-8", errors="replace").strip()
                if "." in raw_callee:
                    recv, short_c = raw_callee.rsplit(".", 1)
                    add_occurrence(
                        parent_sym.id if parent_sym else None,
                        "call",
                        node.start_byte,
                        node.end_byte,
                        details={"callee": short_c.strip(), "raw_callee": raw_callee, "receiver": recv.strip()},
                    )
                else:
                    add_occurrence(
                        parent_sym.id if parent_sym else None,
                        "call",
                        node.start_byte,
                        node.end_byte,
                        details={"callee": raw_callee, "raw_callee": raw_callee},
                    )

        for ch in node.children:
            walk_python(ch, parent_sym, scope)

    def walk_js_ts(node: Node, parent_sym: ExtractedSymbol | None, scope: list[str]) -> None:
        # Handle export_statement wrapping a declaration so display range includes export
        if node.type == "export_statement":
            add_occurrence(
                parent_sym.id if parent_sym else None,
                "export",
                node.start_byte,
                node.end_byte,
            )
            decl = node.child_by_field_name("declaration")
            if decl is not None:
                _handle_js_ts_declaration(
                    decl,
                    outer_node=node,
                    parent_sym=parent_sym,
                    scope=scope,
                )
                return

        if _handle_js_ts_declaration(node, outer_node=node, parent_sym=parent_sym, scope=scope):
            return

        if node.type == "import_statement":
            raw_imp = blob_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
            imp_kind = "type_import" if raw_imp.lstrip().startswith("import type") else "import"
            add_occurrence(
                parent_sym.id if parent_sym else None,
                imp_kind,
                node.start_byte,
                node.end_byte,
            )
        elif node.type == "call_expression":
            fn_node = node.child_by_field_name("function")
            if fn_node is not None:
                raw_callee = blob_bytes[fn_node.start_byte : fn_node.end_byte].decode("utf-8", errors="replace").strip()
                if "." in raw_callee:
                    recv, short_c = raw_callee.rsplit(".", 1)
                    add_occurrence(
                        parent_sym.id if parent_sym else None,
                        "call",
                        node.start_byte,
                        node.end_byte,
                        details={"callee": short_c.strip(), "raw_callee": raw_callee, "receiver": recv.strip()},
                    )
                else:
                    add_occurrence(
                        parent_sym.id if parent_sym else None,
                        "call",
                        node.start_byte,
                        node.end_byte,
                        details={"callee": raw_callee, "raw_callee": raw_callee},
                    )
        elif node.type in ("jsx_opening_element", "jsx_self_closing_element"):
            name_n = node.child_by_field_name("name")
            if name_n is not None:
                comp = blob_bytes[name_n.start_byte : name_n.end_byte].decode("utf-8", errors="replace").strip()
                add_occurrence(
                    parent_sym.id if parent_sym else None,
                    "jsx_component",
                    node.start_byte,
                    node.end_byte,
                    details={"component": comp},
                )

        for ch in node.children:
            walk_js_ts(ch, parent_sym, scope)

    def _handle_js_ts_declaration(
        decl: Node,
        *,
        outer_node: Node,
        parent_sym: ExtractedSymbol | None,
        scope: list[str],
    ) -> bool:
        dtype = decl.type
        if dtype in (
            "function_declaration",
            "generator_function_declaration",
            "function_signature",
            "class_declaration",
            "abstract_class_declaration",
            "interface_declaration",
            "type_alias_declaration",
            "enum_declaration",
            "internal_module",
            "module",
            "method_definition",
            "method_signature",
            "abstract_method_signature",
        ):
            name_n = decl.child_by_field_name("name")
            body_n = decl.child_by_field_name("body") or decl.child_by_field_name("value")
            kind_map = {
                "function_declaration": "function",
                "generator_function_declaration": "function",
                "function_signature": "function_signature",
                "class_declaration": "class",
                "abstract_class_declaration": "class",
                "interface_declaration": "interface",
                "type_alias_declaration": "type_alias",
                "enum_declaration": "enum",
                "internal_module": "namespace",
                "module": "namespace",
                "method_definition": "method",
                "method_signature": "method_signature",
                "abstract_method_signature": "method_signature",
            }
            skind = kind_map[dtype]
            if dtype in ("function_signature", "method_signature", "abstract_method_signature", "type_alias_declaration"):
                body_n = None
            sym = add_symbol(
                parent_id=parent_sym.id if parent_sym else None,
                scope_names=scope,
                sym_kind=skind,
                name_node=name_n,
                decl_start=outer_node.start_byte,
                decl_end=outer_node.end_byte,
                body_node=body_n,
            )
            target_walk = body_n if body_n is not None else decl
            next_sym = sym if sym is not None else parent_sym
            next_scope = [*scope, sym.name] if sym is not None else scope
            for ch in target_walk.children:
                walk_js_ts(ch, next_sym, next_scope)
            return True

        if dtype in ("lexical_declaration", "variable_declaration"):
            declarators = [c for c in decl.named_children if c.type == "variable_declarator"]
            for idx, vdecl in enumerate(declarators):
                name_n = vdecl.child_by_field_name("name")
                val_n = vdecl.child_by_field_name("value")
                if name_n is not None and name_n.type == "identifier":
                    is_fn = val_n is not None and val_n.type in (
                        "arrow_function",
                        "function_expression",
                        "function",
                        "generator_function",
                    )
                    skind = "function" if is_fn else "variable"
                    body_n = (
                        val_n.child_by_field_name("body")
                        if is_fn and val_n is not None
                        else val_n
                    )
                    d_start = outer_node.start_byte if len(declarators) == 1 and idx == 0 else vdecl.start_byte
                    d_end = outer_node.end_byte if len(declarators) == 1 and idx == 0 else vdecl.end_byte
                    sym = add_symbol(
                        parent_id=parent_sym.id if parent_sym else None,
                        scope_names=scope,
                        sym_kind=skind,
                        name_node=name_n,
                        decl_start=d_start,
                        decl_end=d_end,
                        body_node=body_n,
                        header_end_override=name_n.end_byte,
                    )
                    if val_n is not None:
                        next_sym = sym if sym is not None else parent_sym
                        next_scope = [*scope, sym.name] if sym is not None else scope
                        for ch in val_n.children:
                            walk_js_ts(ch, next_sym, next_scope)
            return True

        return False

    def walk_kotlin_java(node: Node, parent_sym: ExtractedSymbol | None, scope: list[str]) -> None:
        ntype = node.type
        if ntype in (
            "class_declaration",
            "object_declaration",
            "interface_declaration",
            "enum_declaration",
            "function_declaration",
            "method_declaration",
            "constructor_declaration",
        ):
            name_n = node.child_by_field_name("name")
            if name_n is None:
                for ch in node.named_children:
                    if ch.type in ("identifier", "type_identifier"):
                        name_n = ch
                        break
            body_n = node.child_by_field_name("body")
            if body_n is None:
                for ch in node.named_children:
                    if ch.type in ("class_body", "function_body", "enum_class_body", "block"):
                        body_n = ch
                        break

            kind_map = {
                "class_declaration": "class",
                "object_declaration": "class",
                "interface_declaration": "interface",
                "enum_declaration": "enum",
                "function_declaration": "method" if parent_sym is not None else "function",
                "method_declaration": "method",
                "constructor_declaration": "method",
            }
            sym = add_symbol(
                parent_id=parent_sym.id if parent_sym else None,
                scope_names=scope,
                sym_kind=kind_map[ntype],
                name_node=name_n,
                decl_start=node.start_byte,
                decl_end=node.end_byte,
                body_node=body_n,
            )
            target_walk = body_n if body_n is not None else node
            next_sym = sym if sym is not None else parent_sym
            next_scope = [*scope, sym.name] if sym is not None else scope
            for ch in target_walk.children:
                walk_kotlin_java(ch, next_sym, next_scope)
            return

        if ntype in ("import", "import_declaration"):
            add_occurrence(
                parent_sym.id if parent_sym else None,
                "import",
                node.start_byte,
                node.end_byte,
            )
        elif ntype in ("call_expression", "method_invocation"):
            name_n = node.child_by_field_name("name")
            recv_str: str | None = None
            obj_n = node.child_by_field_name("object")
            if obj_n is not None:
                recv_str = blob_bytes[obj_n.start_byte : obj_n.end_byte].decode("utf-8", errors="replace").strip()
            if name_n is None and node.named_children:
                first_ch = node.named_children[0]
                if first_ch.type == "navigation_expression" and first_ch.named_children:
                    name_n = first_ch.named_children[-1]
                    if len(first_ch.named_children) >= 2:
                        r_n = first_ch.named_children[0]
                        recv_str = blob_bytes[r_n.start_byte : r_n.end_byte].decode("utf-8", errors="replace").strip()
                elif first_ch.type == "identifier":
                    name_n = first_ch
            if name_n is not None:
                callee = blob_bytes[name_n.start_byte : name_n.end_byte].decode("utf-8", errors="replace").strip()
                # Extract top-level Gradle Kotlin DSL blocks (`plugins`, `android`, `dependencies`, `buildTypes`, etc.) as symbols
                gradle_blocks = {
                    "plugins",
                    "android",
                    "dependencies",
                    "defaultConfig",
                    "buildTypes",
                    "buildFeatures",
                    "pluginManagement",
                    "dependencyResolutionManagement",
                }
                lambda_body = next(
                    (c for c in node.named_children if c.type in ("annotated_lambda", "lambda_literal", "call_suffix")),
                    None,
                )
                if language == "kotlin" and callee in gradle_blocks and lambda_body is not None:
                    sym = add_symbol(
                        parent_id=parent_sym.id if parent_sym else None,
                        scope_names=scope,
                        sym_kind="namespace",
                        name_node=name_n,
                        decl_start=node.start_byte,
                        decl_end=node.end_byte,
                        body_node=lambda_body,
                    )
                    next_sym = sym if sym is not None else parent_sym
                    next_scope = [*scope, sym.name] if sym is not None else scope
                    for ch in lambda_body.children:
                        walk_kotlin_java(ch, next_sym, next_scope)
                    return

                det_dict: dict[str, Any] = {"callee": callee}
                if recv_str:
                    det_dict["receiver"] = recv_str
                    if language == "kotlin":
                        ancestor = node.parent
                        while ancestor is not None and ancestor.type != "function_declaration" and "receiver_type" not in det_dict:
                            for declaration in reversed(ancestor.named_children):
                                if declaration.type != "property_declaration":
                                    continue
                                if ancestor.type != "class_body" and declaration.start_byte >= node.start_byte:
                                    continue
                                variable = next((c for c in declaration.named_children if c.type == "variable_declaration"), None)
                                if variable is None or not variable.named_children:
                                    continue
                                var_name = variable.named_children[0].text.decode("utf-8")
                                if var_name != recv_str:
                                    continue
                                type_node = next((c for c in variable.named_children if c.type in ("user_type", "nullable_type")), None)
                                # An untyped local still shadows a typed parameter or field.
                                det_dict["receiver_type"] = type_node.text.decode("utf-8").rstrip("?") if type_node else ""
                                break
                            ancestor = ancestor.parent
                add_occurrence(
                    parent_sym.id if parent_sym else None,
                    "call",
                    node.start_byte,
                    node.end_byte,
                    details=det_dict,
                )

        for ch in node.children:
            walk_kotlin_java(ch, parent_sym, scope)

    def _find_c_declarator_name(decl: Node | None) -> Node | None:
        cur = decl
        while cur is not None:
            if cur.type in ("identifier", "field_identifier", "type_identifier", "destructor_name"):
                return cur
            next_d = cur.child_by_field_name("declarator") or cur.child_by_field_name("name")
            if next_d is not None:
                cur = next_d
                continue
            found = None
            for ch in cur.named_children:
                if ch.type in (
                    "function_declarator",
                    "pointer_declarator",
                    "reference_declarator",
                    "identifier",
                    "field_identifier",
                    "qualified_identifier",
                ):
                    found = ch
                    break
            cur = found
        return None

    def walk_go_rust_c_cpp(node: Node, parent_sym: ExtractedSymbol | None, scope: list[str]) -> None:
        ntype = node.type

        # Go type_spec (`type Foo struct{}` / `type Bar interface{}`)
        if language == "go" and ntype == "type_spec":
            name_n = node.child_by_field_name("name")
            type_n = node.child_by_field_name("type")
            skind = "interface" if (type_n is not None and type_n.type == "interface_type") else "class"
            add_symbol(
                parent_id=parent_sym.id if parent_sym else None,
                scope_names=scope,
                sym_kind=skind,
                name_node=name_n,
                decl_start=node.start_byte,
                decl_end=node.end_byte,
                body_node=type_n,
            )
            return

        # Go method_declaration with receiver type
        if language == "go" and ntype == "method_declaration":
            name_n = node.child_by_field_name("name")
            body_n = node.child_by_field_name("body")
            recv_n = node.child_by_field_name("receiver")
            recv_type_name: str | None = None
            if recv_n is not None:
                for ch in recv_n.named_children:
                    t_node = ch.child_by_field_name("type")
                    if t_node is not None:
                        recv_type_name = (
                            blob_bytes[t_node.start_byte : t_node.end_byte]
                            .decode("utf-8", errors="replace")
                            .lstrip("*")
                            .strip()
                        )
                        break
            m_scope = [*scope, recv_type_name] if recv_type_name else scope
            sym = add_symbol(
                parent_id=parent_sym.id if parent_sym else None,
                scope_names=m_scope,
                sym_kind="method",
                name_node=name_n,
                decl_start=node.start_byte,
                decl_end=node.end_byte,
                body_node=body_n,
            )
            if body_n is not None:
                for ch in body_n.children:
                    walk_go_rust_c_cpp(ch, sym or parent_sym, [*m_scope, sym.name] if sym else m_scope)
            return

        # Rust impl_item (`impl Foo { ... }`)
        if language == "rust" and ntype == "impl_item":
            type_n = node.child_by_field_name("type")
            body_n = node.child_by_field_name("body")
            impl_name = (
                blob_bytes[type_n.start_byte : type_n.end_byte].decode("utf-8", errors="replace").strip()
                if type_n is not None
                else None
            )
            next_scope = [*scope, impl_name] if impl_name else scope
            if body_n is not None:
                for ch in body_n.children:
                    walk_go_rust_c_cpp(ch, parent_sym, next_scope)
            return

        # C/C++ function_definition
        if language in ("c", "cpp") and ntype == "function_definition":
            decl_n = node.child_by_field_name("declarator")
            name_n = _find_c_declarator_name(decl_n)
            body_n = node.child_by_field_name("body")
            skind = "method" if (parent_sym is not None and parent_sym.kind == "class") else "function"
            sym = add_symbol(
                parent_id=parent_sym.id if parent_sym else None,
                scope_names=scope,
                sym_kind=skind,
                name_node=name_n,
                decl_start=node.start_byte,
                decl_end=node.end_byte,
                body_node=body_n,
                header_end_override=body_n.start_byte if body_n is not None else node.end_byte,
            )
            if body_n is not None:
                for ch in body_n.children:
                    walk_go_rust_c_cpp(ch, sym or parent_sym, [*scope, sym.name] if sym else scope)
            return

        # Structs, classes, traits, enums, namespaces, Go/Rust functions
        if ntype in (
            "function_declaration",
            "function_item",
            "struct_item",
            "enum_item",
            "trait_item",
            "struct_specifier",
            "class_specifier",
            "enum_specifier",
            "namespace_definition",
        ):
            name_n = node.child_by_field_name("name")
            body_n = node.child_by_field_name("body")
            if ntype in ("struct_specifier", "class_specifier", "enum_specifier") and body_n is None:
                # Forward declaration without body
                return
            kind_map = {
                "function_declaration": "function",
                "function_item": "method" if scope else "function",
                "struct_item": "class",
                "enum_item": "enum",
                "trait_item": "interface",
                "struct_specifier": "class",
                "class_specifier": "class",
                "enum_specifier": "enum",
                "namespace_definition": "namespace",
            }
            sym = add_symbol(
                parent_id=parent_sym.id if parent_sym else None,
                scope_names=scope,
                sym_kind=kind_map[ntype],
                name_node=name_n,
                decl_start=node.start_byte,
                decl_end=node.end_byte,
                body_node=body_n,
            )
            target_walk = body_n if body_n is not None else node
            next_sym = sym if sym is not None else parent_sym
            next_scope = [*scope, sym.name] if sym is not None else scope
            for ch in target_walk.children:
                walk_go_rust_c_cpp(ch, next_sym, next_scope)
            return

        if ntype in ("import_declaration", "use_declaration", "preproc_include"):
            add_occurrence(
                parent_sym.id if parent_sym else None,
                "import",
                node.start_byte,
                node.end_byte,
            )
        elif ntype == "call_expression":
            fn_n = node.child_by_field_name("function")
            if fn_n is not None:
                raw_callee = blob_bytes[fn_n.start_byte : fn_n.end_byte].decode("utf-8", errors="replace").strip()
                # Extract trailing identifier after `.` or `::` or `->`
                callee = re.split(r"(?:\.|::|->)", raw_callee)[-1].strip()
                if callee:
                    add_occurrence(
                        parent_sym.id if parent_sym else None,
                        "call",
                        node.start_byte,
                        node.end_byte,
                        details={"callee": callee, "raw_callee": raw_callee},
                    )

        for ch in node.children:
            walk_go_rust_c_cpp(ch, parent_sym, scope)

    if language == "python":
        walk_python(root, None, [])
    elif language in ("kotlin", "java"):
        walk_kotlin_java(root, None, [])
    elif language in ("go", "rust", "c", "cpp"):
        walk_go_rust_c_cpp(root, None, [])
    else:
        walk_js_ts(root, None, [])

    # Resolve same-file lexical call candidates where a named definition exists
    symbols_by_id = {s.id: s for s in symbols}
    by_simple_name: dict[str, list[ExtractedSymbol]] = {}
    for s in symbols:
        by_simple_name.setdefault(s.name, []).append(s)

    for occ in occurrences:
        if occ.kind in ("call", "jsx_component") and occ.enclosing_symbol_id is not None:
            # Receiver calls require type/import resolution, even when names exist in this file.
            if occ.details.get("receiver"):
                continue
            callee = occ.details.get("callee") or occ.details.get("component")
            if isinstance(callee, str) and callee in by_simple_name:
                targets = by_simple_name[callee]
                if language == "kotlin":
                    scope_id = occ.enclosing_symbol_id
                    while True:
                        scoped_targets = [s for s in targets if s.parent_symbol_id == scope_id]
                        if scoped_targets or scope_id is None:
                            targets = scoped_targets
                            break
                        scope_id = symbols_by_id[scope_id].parent_symbol_id
                for target_sym in targets:
                    if target_sym.id != occ.enclosing_symbol_id:
                        relations.append(
                            CandidateRelation(
                                from_symbol_id=occ.enclosing_symbol_id,
                                to_symbol_id=target_sym.id,
                                relation_kind="calls_candidate",
                                method="lexical_same_blob",
                            )
                        )

    parse_state = "partial" if error_ranges else "complete"
    return ParseResult(
        extraction_id=ext_id,
        blob_oid=actual_oid,
        language=language,
        extractor_fingerprint=fp,
        state=parse_state,
        diagnostics=diagnostics,
        symbols=symbols,
        occurrences=occurrences,
        relations=relations,
    )


def index_blob_into_store(
    conn: sqlite3.Connection,
    *,
    repository_id: str,
    blob_oid: str,
    blob_bytes: bytes,
    language: str,
) -> tuple[ParseResult, bool]:
    """Ensure `(blob_oid, language, extractor_fingerprint)` is extracted in `extractions` and `symbols`.
    Returns `(parse_result, was_cache_hit)`.
    """
    fp = get_extractor_fingerprint(language)
    ext_id = make_extraction_id(blob_oid=blob_oid, language=language, extractor_fingerprint=fp)
    existing = conn.execute(
        "SELECT id, state, diagnostics_json FROM extractions WHERE id = ?",
        (ext_id,),
    ).fetchone()
    if existing is not None:
        sym_rows = conn.execute(
            """
            SELECT id, parent_symbol_id, kind, name, qualified_name,
                   byte_start, byte_end, body_byte_start, body_byte_end,
                   line_start, line_end, signature
            FROM symbols
            WHERE extraction_id = ?
            ORDER BY byte_start ASC, byte_end DESC, id ASC
            """,
            (ext_id,),
        ).fetchall()
        syms = [
            ExtractedSymbol(
                id=str(r["id"]),
                parent_symbol_id=r["parent_symbol_id"],
                kind=str(r["kind"]),
                name=str(r["name"]),
                qualified_name=str(r["qualified_name"]),
                byte_start=int(r["byte_start"]),
                byte_end=int(r["byte_end"]),
                body_byte_start=r["body_byte_start"],
                body_byte_end=r["body_byte_end"],
                line_start=int(r["line_start"]),
                line_end=int(r["line_end"]),
                signature=str(r["signature"]),
            )
            for r in sym_rows
        ]
        return (
            ParseResult(
                extraction_id=ext_id,
                blob_oid=blob_oid,
                language=language,
                extractor_fingerprint=fp,
                state=str(existing["state"]),
                diagnostics=[],
                symbols=syms,
                occurrences=[],
                relations=[],
            ),
            True,
        )

    res = parse_blob(blob_bytes, language, fp, blob_oid=blob_oid)
    conn.execute(
        """
        INSERT INTO extractions (id, blob_oid, language, extractor_fingerprint, state, diagnostics_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            res.extraction_id,
            res.blob_oid,
            res.language,
            res.extractor_fingerprint,
            res.state,
            canonical_json(res.diagnostics),
        ),
    )
    for sym in res.symbols:
        conn.execute(
            """
            INSERT INTO symbols (
                id, extraction_id, parent_symbol_id, kind, name, qualified_name,
                byte_start, byte_end, body_byte_start, body_byte_end,
                line_start, line_end, signature
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sym.id,
                res.extraction_id,
                sym.parent_symbol_id,
                sym.kind,
                sym.name,
                sym.qualified_name,
                sym.byte_start,
                sym.byte_end,
                sym.body_byte_start,
                sym.body_byte_end,
                sym.line_start,
                sym.line_end,
                sym.signature,
            ),
        )
    for occ in res.occurrences:
        conn.execute(
            """
            INSERT INTO syntax_occurrences (
                id, extraction_id, enclosing_symbol_id, kind, text, byte_start, byte_end, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                occ.id,
                res.extraction_id,
                occ.enclosing_symbol_id,
                occ.kind,
                occ.text,
                occ.byte_start,
                occ.byte_end,
                canonical_json(occ.details),
            ),
        )
    for rel in res.relations:
        rel_id = make_relation_id(
            from_kind="symbol",
            from_id=rel.from_symbol_id,
            to_kind="symbol",
            to_id=rel.to_symbol_id,
            relation_kind=rel.relation_kind,
            method=rel.method,
        )
        mapping = "exact" if rel.relation_kind == "contains" else "candidate"
        conn.execute(
            """
            INSERT INTO relations (
                id, from_kind, from_id, to_kind, to_id, relation_kind, method, mapping, evidence_id, details_json, state
            ) VALUES (?, 'symbol', ?, 'symbol', ?, ?, ?, ?, NULL, '{}', 'active')
            ON CONFLICT(id) DO NOTHING
            """,
            (
                rel_id,
                rel.from_symbol_id,
                rel.to_symbol_id,
                rel.relation_kind,
                rel.method,
                mapping,
            ),
        )

    if res.state == "partial":
        Store.upsert_gap(
            conn,
            repository_id=repository_id,
            code="PARSE_ERROR",
            scope=f"blob:{blob_oid}",
            message=f"Blob {blob_oid[:12]} had syntax errors; extracted valid partial definitions.",
            retryable=False,
        )
    elif res.state == "oversized":
        Store.upsert_gap(
            conn,
            repository_id=repository_id,
            code="BLOB_TOO_LARGE",
            scope=f"blob:{blob_oid}",
            message=f"Blob {blob_oid[:12]} exceeds the 2 MiB parse limit.",
            retryable=False,
        )
    elif res.state == "binary":
        Store.upsert_gap(
            conn,
            repository_id=repository_id,
            code="BINARY_CONTENT",
            scope=f"blob:{blob_oid}",
            message=f"Blob {blob_oid[:12]} contains binary data.",
            retryable=False,
        )
    elif res.state == "error":
        Store.upsert_gap(
            conn,
            repository_id=repository_id,
            code="TEXT_DECODE_FAILED",
            scope=f"blob:{blob_oid}",
            message=f"Blob {blob_oid[:12]} could not be decoded as UTF-8.",
            retryable=False,
        )

    return res, False


def index_active_snapshot_files(
    store: Store,
    mirror: GitMirror,
    *,
    focus_path: bytes | None = None,
    max_blobs: int | None = None,
) -> tuple[int, int]:
    """Process pending `snapshot_files` in the active snapshot, extracting syntax or text chunks
    and populating `search_documents` for active code.
    Returns `(blobs_parsed, cache_hits)`.
    """
    with store.connect_read() as r_conn:
        meta = r_conn.execute(
            "SELECT repository_id, active_revision FROM store_meta WHERE singleton = 1"
        ).fetchone()
        if not meta or not meta["active_revision"]:
            return (0, 0)
        repo_id = str(meta["repository_id"])
        active_rev = str(meta["active_revision"])

        rows = r_conn.execute(
            "SELECT path, blob_oid, mode, content_state FROM snapshot_files ORDER BY path ASC"
        ).fetchall()

    # Prioritize focus_path if supplied
    pending_rows = [r for r in rows if r["content_state"] == "pending"]
    if focus_path is not None:
        pending_rows.sort(key=lambda r: (0 if bytes(r["path"]) == focus_path else 1, bytes(r["path"])))

    if max_blobs is not None:
        pending_rows = pending_rows[:max_blobs]

    blobs_parsed = 0
    cache_hits = 0

    # Read blob contents outside write transaction
    blob_payloads: dict[str, tuple[int, bytes] | None] = {}
    unique_oids = list(dict.fromkeys(str(r["blob_oid"]) for r in pending_rows))
    if unique_oids:
        with mirror.open_batch_session() as batch:
            for oid in unique_oids:
                obj = batch.read_object(oid)
                if obj is None:
                    blob_payloads[oid] = None
                else:
                    blob_payloads[oid] = (obj[1], obj[2])

    with store.write_transaction(bump_generation=True) as conn:
        for row in pending_rows:
            path_bytes = bytes(row["path"])
            blob_oid = str(row["blob_oid"])
            disp_path, path_b64 = path_bytes_to_display(path_bytes)
            payload = blob_payloads.get(blob_oid)

            if payload is None:
                conn.execute(
                    "UPDATE snapshot_files SET content_state = 'metadata_only' WHERE path = ?",
                    (path_bytes,),
                )
                Store.upsert_gap(
                    conn,
                    repository_id=repo_id,
                    code="MISSING_GIT_OBJECT",
                    scope=f"blob:{blob_oid}",
                    message=f"Git blob {blob_oid[:12]} for {disp_path} is missing from object store.",
                    retryable=True,
                )
                continue

            size, raw_bytes = payload
            if is_lfs_pointer(raw_bytes):
                conn.execute(
                    "UPDATE snapshot_files SET content_state = 'metadata_only' WHERE path = ?",
                    (path_bytes,),
                )
                Store.upsert_gap(
                    conn,
                    repository_id=repo_id,
                    code="LFS_CONTENT_NOT_FETCHED",
                    scope=f"blob:{blob_oid}",
                    message=f"File {disp_path} is a Git LFS pointer; external asset content was not fetched.",
                    retryable=False,
                )
                continue

            if size > MAX_PARSE_BYTES:
                conn.execute(
                    "UPDATE snapshot_files SET content_state = 'metadata_only' WHERE path = ?",
                    (path_bytes,),
                )
                Store.upsert_gap(
                    conn,
                    repository_id=repo_id,
                    code="BLOB_TOO_LARGE",
                    scope=f"blob:{blob_oid}",
                    message=f"File {disp_path} ({size} bytes) exceeds the 2 MiB limit.",
                    retryable=False,
                )
                continue

            if b"\x00" in raw_bytes:
                conn.execute(
                    "UPDATE snapshot_files SET content_state = 'metadata_only' WHERE path = ?",
                    (path_bytes,),
                )
                Store.upsert_gap(
                    conn,
                    repository_id=repo_id,
                    code="BINARY_CONTENT",
                    scope=f"blob:{blob_oid}",
                    message=f"File {disp_path} is binary.",
                    retryable=False,
                )
                continue

            try:
                raw_bytes.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                conn.execute(
                    "UPDATE snapshot_files SET content_state = 'metadata_only' WHERE path = ?",
                    (path_bytes,),
                )
                Store.upsert_gap(
                    conn,
                    repository_id=repo_id,
                    code="TEXT_DECODE_FAILED",
                    scope=f"blob:{blob_oid}",
                    message=f"File {disp_path} is not valid UTF-8.",
                    retryable=False,
                )
                continue

            lang = detect_language_for_path(path_bytes)
            if lang is not None:
                parse_res, is_hit = index_blob_into_store(
                    conn,
                    repository_id=repo_id,
                    blob_oid=blob_oid,
                    blob_bytes=raw_bytes,
                    language=lang,
                )
                if is_hit:
                    cache_hits += 1
                else:
                    blobs_parsed += 1
                conn.execute(
                    "UPDATE snapshot_files SET extraction_id = ?, content_state = 'indexed' WHERE path = ?",
                    (parse_res.extraction_id, path_bytes),
                )
            else:
                conn.execute(
                    "UPDATE snapshot_files SET content_state = 'text_only' WHERE path = ?",
                    (path_bytes,),
                )
                if not is_indexable_text_path(path_bytes):
                    Store.upsert_gap(
                        conn,
                        repository_id=repo_id,
                        code="UNSUPPORTED_SYNTAX",
                        scope=f"blob:{blob_oid}",
                        message=f"File {disp_path} uses an unsupported language; indexed as UTF-8 text.",
                        retryable=False,
                    )

        # Rebuild active code search documents so they match the current `snapshot_files`
        conn.execute("DELETE FROM search_documents WHERE record_kind = 'code'")
        indexed_files = conn.execute(
            "SELECT path, blob_oid, extraction_id, content_state FROM snapshot_files WHERE content_state IN ('indexed', 'text_only') ORDER BY path ASC"
        ).fetchall()

        # Group paths by extraction_id so shared blobs index all active paths on their symbols
        ext_to_paths: dict[str, list[tuple[str, str | None, str]]] = {}
        for r in indexed_files:
            p_bytes = bytes(r["path"])
            disp_p, b64_p = path_bytes_to_display(p_bytes)
            b_oid = str(r["blob_oid"])
            e_id = r["extraction_id"]
            if e_id is not None:
                ext_to_paths.setdefault(str(e_id), []).append((disp_p, b64_p, b_oid))

        for ext_id, path_infos in ext_to_paths.items():
            ext_row = conn.execute(
                "SELECT language, blob_oid FROM extractions WHERE id = ?",
                (ext_id,),
            ).fetchone()
            lang_str = str(ext_row["language"]) if ext_row else "unknown"
            b_oid = str(ext_row["blob_oid"]) if ext_row else ""
            sym_rows = conn.execute(
                """
                SELECT id, kind, name, qualified_name, signature, byte_start, byte_end, line_start, line_end
                FROM symbols
                WHERE extraction_id = ?
                ORDER BY byte_start ASC, id ASC
                """,
                (ext_id,),
            ).fetchall()
            doc_rows = conn.execute(
                "SELECT enclosing_symbol_id, text FROM syntax_occurrences WHERE extraction_id = ? AND kind = 'docstring'",
                (ext_id,),
            ).fetchall()
            sym_docs: dict[str, str] = {
                str(dr["enclosing_symbol_id"]): str(dr["text"])
                for dr in doc_rows
                if dr["enclosing_symbol_id"]
            }

            primary_disp, primary_b64, _ = path_infos[0]
            all_path_terms = " ".join(p[0] for p in path_infos)
            raw_blob_b = (blob_payloads.get(b_oid) or (0, b""))[1]
            if not raw_blob_b:
                with mirror.open_batch_session() as batch:
                    b_obj = batch.read_object(b_oid)
                    if b_obj is not None:
                        raw_blob_b = b_obj[2]

            for srow in sym_rows:
                sid = str(srow["id"])
                qname = str(srow["qualified_name"])
                sname = str(srow["name"])
                sig = str(srow["signature"])
                doc_text = sym_docs.get(sid, "")
                b_s = int(srow["byte_start"])
                b_e = int(srow["byte_end"])
                l_s = int(srow["line_start"])
                l_e = int(srow["line_end"])

                if raw_blob_b:
                    from codecontext.identity import make_ev_id

                    sym_slice = raw_blob_b[b_s:b_e]
                    sym_hash = sha256_bytes(sym_slice)
                    for p_disp, p_b64, _ in path_infos:
                        c_anc = {
                            "git_oid": b_oid,
                            "kind": "code",
                            "original_review_commit": None,
                            "path": {"path": p_disp, "path_bytes_b64": p_b64},
                            "revision": active_rev,
                            "source_version_id": None,
                            "span": {
                                "byte_end": b_e,
                                "byte_start": b_s,
                                "line_end": l_e,
                                "line_start": l_s,
                            },
                            "underlying_record_id": sid,
                        }
                        c_ev_id = make_ev_id(
                            repository_id=repo_id,
                            kind="code",
                            source_version_id=None,
                            git_oid=b_oid,
                            anchor=c_anc,
                            content_sha256=sym_hash,
                        )
                        conn.execute(
                            """
                            INSERT INTO evidence (id, kind, source_version_id, git_oid, anchor_json, content_sha256, state)
                            VALUES (?, 'code', NULL, ?, ?, ?, 'available')
                            ON CONFLICT(id) DO NOTHING
                            """,
                            (c_ev_id, b_oid, canonical_json(c_anc), sym_hash),
                        )

                title_norm = normalize_search_text(f"{qname} {primary_disp}")
                ident_norm = normalize_search_text(f"{sname} {qname} {all_path_terms}")
                body_norm = normalize_search_text(f"{sig} {doc_text}")
                meta_json = canonical_json(
                    {
                        "blob_oid": b_oid,
                        "byte_end": b_e,
                        "byte_start": b_s,
                        "display_title": f"{qname} ({primary_disp})",
                        "language": lang_str,
                        "line_end": l_e,
                        "line_start": l_s,
                        "path": primary_disp,
                        "path_bytes_b64": primary_b64,
                        "paths": [p[0] for p in path_infos[:10]],
                        "revision": active_rev,
                        "signature": sig[:320],
                    }
                )
                conn.execute(
                    """
                    INSERT INTO search_documents (record_kind, record_id, chunk_ordinal, title, identifiers, body, metadata_json)
                    VALUES ('code', ?, 0, ?, ?, ?, ?)
                    """,
                    (sid, title_norm, ident_norm, body_norm, meta_json),
                )

        # Also index file/chunk-level documents for text_only files or files without symbols
        with mirror.open_batch_session() as batch:
            for r in indexed_files:
                p_bytes = bytes(r["path"])
                disp_p, b64_p = path_bytes_to_display(p_bytes)
                b_oid = str(r["blob_oid"])
                e_id = r["extraction_id"]
                if e_id is not None:
                    has_syms = int(
                        conn.execute(
                            "SELECT COUNT(*) FROM symbols WHERE extraction_id = ?", (e_id,)
                        ).fetchone()[0]
                    )
                    if has_syms > 0:
                        continue
                obj = batch.read_object(b_oid)
                if obj is None:
                    continue
                raw_b = obj[2]
                try:
                    chunks = chunk_utf8_bytes(raw_b)
                except UnicodeDecodeError:
                    continue
                for ch in chunks:
                    l_s, l_e = byte_span_to_lines(raw_b, ch.byte_start, ch.byte_end)
                    anchor_dict = {
                        "git_oid": b_oid,
                        "kind": "code",
                        "original_review_commit": None,
                        "path": {"path": disp_p, "path_bytes_b64": b64_p},
                        "revision": active_rev,
                        "source_version_id": None,
                        "span": {
                            "byte_end": ch.byte_end,
                            "byte_start": ch.byte_start,
                            "line_end": l_e,
                            "line_start": l_s,
                        },
                        "underlying_record_id": None,
                    }
                    chunk_bytes = raw_b[ch.byte_start : ch.byte_end]
                    c_hash = sha256_bytes(chunk_bytes)
                    from codecontext.identity import make_ev_id

                    ev_id = make_ev_id(
                        repository_id=repo_id,
                        kind="code",
                        source_version_id=None,
                        git_oid=b_oid,
                        anchor=anchor_dict,
                        content_sha256=c_hash,
                    )
                    conn.execute(
                        """
                        INSERT INTO evidence (id, kind, source_version_id, git_oid, anchor_json, content_sha256, state)
                        VALUES (?, 'code', NULL, ?, ?, ?, 'available')
                        ON CONFLICT(id) DO NOTHING
                        """,
                        (ev_id, b_oid, canonical_json(anchor_dict), c_hash),
                    )
                    meta_json = canonical_json(
                        {
                            "blob_oid": b_oid,
                            "byte_end": ch.byte_end,
                            "byte_start": ch.byte_start,
                            "display_title": disp_p,
                            "evidence_id": ev_id,
                            "line_end": l_e,
                            "line_start": l_s,
                            "path": disp_p,
                            "path_bytes_b64": b64_p,
                            "revision": active_rev,
                            "snippet": ch.text[:320],
                        }
                    )
                    conn.execute(
                        """
                        INSERT INTO search_documents (record_kind, record_id, chunk_ordinal, title, identifiers, body, metadata_json)
                        VALUES ('code', ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(record_kind, record_id, chunk_ordinal) DO NOTHING
                        """,
                        (
                            ev_id,
                            ch.ordinal,
                            normalize_search_text(disp_p),
                            normalize_search_text(disp_p),
                            normalize_search_text(ch.text),
                            meta_json,
                        ),
                    )

        rem_pending = int(
            conn.execute(
                "SELECT COUNT(*) FROM snapshot_files WHERE content_state = 'pending'"
            ).fetchone()[0]
        )
        if rem_pending == 0:
            Store.resolve_gaps_in_scope(conn, scope="git:snapshot", code="STRUCTURE_PENDING")

        # Resolve cross-file call candidates using import/module/type-aware disambiguation
        _resolve_cross_file_calls_and_imports(conn)

    return (blobs_parsed, cache_hits)


PY_FROM_IMPORT_RE = re.compile(
    r"^\s*from\s+([A-Za-z0-9_.]+)\s+import\s+(.+)$", re.DOTALL
)
PY_IMPORT_RE = re.compile(r"^\s*import\s+([A-Za-z0-9_.,\s]+)$")
JS_IMPORT_RE = re.compile(
    r"import\s+(?:type\s+)?(.+?)\s+from\s+['\"]([^'\"]+)['\"]", re.DOTALL
)
KT_JAVA_IMPORT_RE = re.compile(r"^\s*import\s+([A-Za-z0-9_.]+)\s*;?\s*$")
PARAM_TYPE_RE = re.compile(
    r"(?:([A-Za-z_]\w*)\s*:\s*([A-Za-z_]\w*)|([A-Z]\w*)\s+([A-Za-z_]\w*))"
)


def _module_spec_matches_path(mod_spec: str, file_path: str) -> bool:
    """Check whether an import module specifier (`a`, `pkg.a`, `./service.js`, `com.example.Repo`)
    matches an indexed repository `file_path` (`a.py`, `src/pkg/a.py`, `src/service.ts`, etc.).
    """
    clean_spec = mod_spec.strip().strip("'\"").lstrip("./")
    for ext in EXTENSION_TO_LANGUAGE:
        if clean_spec.endswith(ext):
            clean_spec = clean_spec[: -len(ext)]
            break
    spec_slash = clean_spec.replace(".", "/").strip("/")
    if not spec_slash:
        return False

    fp_no_ext = file_path
    for ext in EXTENSION_TO_LANGUAGE:
        if fp_no_ext.endswith(ext):
            fp_no_ext = fp_no_ext[: -len(ext)]
            break
    if fp_no_ext.endswith("/__init__") or fp_no_ext.endswith("/index"):
        fp_pkg = fp_no_ext.rsplit("/", 1)[0]
    else:
        fp_pkg = fp_no_ext

    return (
        fp_no_ext == spec_slash
        or fp_no_ext.endswith("/" + spec_slash)
        or fp_pkg == spec_slash
        or fp_pkg.endswith("/" + spec_slash)
    )


def _parse_extraction_imports(
    imp_texts: list[str],
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """Parse import statements in a blob into:
    - `symbol_imports`: `{local_alias_or_name: module_spec}` (e.g. `from a import save` -> `{'save': 'a'}`)
    - `module_aliases`: `{alias: module_spec}` (e.g. `import a` -> `{'a': 'a'}`, `import pkg.a as mod_a` -> `{'mod_a': 'pkg.a'}`)
    - `wildcard_modules`: `[module_spec]`
    """
    symbol_imports: dict[str, str] = {}
    module_aliases: dict[str, str] = {}
    wildcard_modules: list[str] = []

    for raw in imp_texts:
        txt = raw.strip()
        m_from = PY_FROM_IMPORT_RE.match(txt)
        if m_from:
            mod_spec = m_from.group(1).strip()
            names_part = m_from.group(2).strip().strip("()")
            for piece in names_part.split(","):
                p = piece.strip()
                if not p:
                    continue
                if p == "*":
                    wildcard_modules.append(mod_spec)
                elif " as " in p:
                    orig, alias = p.split(" as ", 1)
                    symbol_imports[alias.strip()] = f"{mod_spec}.{orig.strip()}"
                else:
                    symbol_imports[p] = mod_spec
                    module_aliases[p] = f"{mod_spec}.{p}"
            continue

        m_js = JS_IMPORT_RE.search(txt)
        if m_js:
            clause = m_js.group(1).strip()
            mod_spec = m_js.group(2).strip()
            if "{" in clause and "}" in clause:
                inside = clause[clause.find("{") + 1 : clause.rfind("}")]
                for piece in inside.split(","):
                    p = piece.strip()
                    if not p:
                        continue
                    if " as " in p:
                        orig, alias = p.split(" as ", 1)
                        symbol_imports[alias.strip()] = mod_spec
                    else:
                        symbol_imports[p] = mod_spec
            else:
                default_name = clause.split(",")[0].strip()
                if default_name.startswith("* as "):
                    alias = default_name[5:].strip()
                    module_aliases[alias] = mod_spec
                elif default_name:
                    symbol_imports[default_name] = mod_spec
            continue

        m_kt = KT_JAVA_IMPORT_RE.match(txt)
        if m_kt:
            full_imp = m_kt.group(1).strip()
            if full_imp.endswith(".*"):
                wildcard_modules.append(full_imp[:-2])
            elif "." in full_imp:
                mod_part, sym_part = full_imp.rsplit(".", 1)
                symbol_imports[sym_part] = mod_part
            continue

        m_py_imp = PY_IMPORT_RE.match(txt)
        if m_py_imp:
            for piece in m_py_imp.group(1).split(","):
                p = piece.strip()
                if not p:
                    continue
                if " as " in p:
                    mod_s, alias = p.split(" as ", 1)
                    module_aliases[alias.strip()] = mod_s.strip()
                else:
                    short_m = p.rsplit(".", 1)[-1]
                    module_aliases[short_m] = p

    return symbol_imports, module_aliases, wildcard_modules


def _extract_receiver_types_from_signature(signature: str) -> dict[str, str]:
    """Extract parameter/property variable -> TypeName mappings from a function or class signature."""
    var_to_type: dict[str, str] = {}
    for m in PARAM_TYPE_RE.finditer(signature):
        if m.group(1) and m.group(2):
            var_to_type[m.group(1)] = m.group(2)
        elif m.group(3) and m.group(4):
            t_cand, v_cand = m.group(3), m.group(4)
            if t_cand[0].isupper():
                var_to_type[v_cand] = t_cand
    return var_to_type


def _resolve_cross_file_calls_and_imports(conn: sqlite3.Connection) -> None:
    """Resolve cross-file `calls_candidate` and `imports_candidate` relations using:
    1. Explicit import target matching (`from a import save` binds `save()` ONLY to `a.py`'s `save`, never `b.py`'s `save`)
    2. Module alias receiver matching (`a.save()` where `import a` binds to `a.py`'s `save`)
    3. Parameter/field receiver type matching (`repo.save()` where `repo: UserRepository` binds to `UserRepository.save`)
    4. Same-directory/package preference
    5. Global fallback ONLY when the symbol name is globally unique (1 match) across the snapshot.
    """
    conn.execute("DELETE FROM relations WHERE method = 'lexical_snapshot'")

    active_rows = conn.execute(
        """
        SELECT s.id, s.extraction_id, s.parent_symbol_id, s.kind, s.name, s.qualified_name, s.signature,
               sf.path
        FROM symbols s
        JOIN snapshot_files sf ON sf.extraction_id = s.extraction_id
        WHERE s.kind IN ('function', 'async_function', 'method', 'async_method', 'class', 'interface')
        """
    ).fetchall()

    # Map symbol metadata and index by short name
    sym_meta: dict[str, dict[str, Any]] = {}
    by_name: dict[str, list[dict[str, Any]]] = {}
    ext_to_path: dict[str, str] = {}

    for r in active_rows:
        sid = str(r["id"])
        ext_id = str(r["extraction_id"])
        disp_p, _ = path_bytes_to_display(bytes(r["path"]))
        ext_to_path[ext_id] = disp_p
        info = {
            "extraction_id": ext_id,
            "id": sid,
            "kind": str(r["kind"]),
            "name": str(r["name"]),
            "parent_symbol_id": r["parent_symbol_id"],
            "path": disp_p,
            "qualified_name": str(r["qualified_name"]),
            "signature": str(r["signature"]),
        }
        sym_meta[sid] = info
        by_name.setdefault(info["name"], []).append(info)

    # Load imports per active extraction_id
    imp_rows = conn.execute(
        """
        SELECT so.extraction_id, so.text
        FROM syntax_occurrences so
        JOIN snapshot_files sf ON sf.extraction_id = so.extraction_id
        WHERE so.kind IN ('import', 'relative_import', 'type_import')
        """
    ).fetchall()
    ext_raw_imports: dict[str, list[str]] = {}
    for ir in imp_rows:
        ext_raw_imports.setdefault(str(ir["extraction_id"]), []).append(str(ir["text"]))

    ext_parsed_imports: dict[str, tuple[dict[str, str], dict[str, str], list[str]]] = {
        ext_id: _parse_extraction_imports(txts)
        for ext_id, txts in ext_raw_imports.items()
    }

    call_rows = conn.execute(
        """
        SELECT DISTINCT so.extraction_id, so.enclosing_symbol_id, so.details_json
        FROM syntax_occurrences so
        JOIN snapshot_files sf ON sf.extraction_id = so.extraction_id
        WHERE so.kind IN ('call', 'jsx_component') AND so.enclosing_symbol_id IS NOT NULL
        """
    ).fetchall()

    for cr in call_rows:
        enc_id = str(cr["enclosing_symbol_id"])
        ext_id = str(cr["extraction_id"])
        enc_info = sym_meta.get(enc_id)
        if enc_info is None:
            continue

        det = json.loads(str(cr["details_json"])) if cr["details_json"] else {}
        callee = det.get("callee") or det.get("component")
        receiver = det.get("receiver")
        if not isinstance(callee, str) or callee not in by_name:
            continue

        candidates = [c for c in by_name[callee] if c["id"] != enc_id]
        if not candidates:
            continue

        sym_imps, mod_aliases, wildcards = ext_parsed_imports.get(ext_id, ({}, {}, []))
        caller_path = enc_info["path"]
        caller_dir = caller_path.rsplit("/", 1)[0] if "/" in caller_path else ""

        # Gather receiver type map from enclosing function and its parent class (e.g. constructor injection)
        recv_types: dict[str, str] = {}
        parent_id = enc_info.get("parent_symbol_id")
        if parent_id and str(parent_id) in sym_meta:
            parent_info = sym_meta[str(parent_id)]
            recv_types.update(_extract_receiver_types_from_signature(parent_info["signature"]))
        recv_types.update(_extract_receiver_types_from_signature(enc_info["signature"]))
        if isinstance(receiver, str) and "receiver_type" in det:
            recv_types[receiver] = det["receiver_type"]

        resolved_targets: list[dict[str, Any]] = []

        if isinstance(receiver, str) and receiver:
            clean_recv = receiver.replace("this.", "").replace("self.", "").strip()
            # 1A. Receiver is an imported module alias (`a.save()` where `import a` or `from pkg import a`)
            if clean_recv in mod_aliases:
                mod_spec = mod_aliases[clean_recv]
                resolved_targets = [
                    c for c in candidates if _module_spec_matches_path(mod_spec, c["path"])
                ]
            elif clean_recv in sym_imps:
                # Receiver is an imported class or module (`Service.run()`)
                mod_spec = sym_imps[clean_recv]
                resolved_targets = [
                    c
                    for c in candidates
                    if c["qualified_name"] == f"{clean_recv}.{callee}"
                    or _module_spec_matches_path(mod_spec, c["path"])
                ]
            # 1B. Receiver is a typed parameter/field (`api.getUser()` where `api: Api`)
            if not resolved_targets and clean_recv in recv_types:
                t_name = recv_types[clean_recv]
                resolved_targets = [
                    c
                    for c in candidates
                    if c["qualified_name"] == f"{t_name}.{callee}"
                    or c["qualified_name"].endswith(f".{t_name}.{callee}")
                ]
            # 1C. Receiver is `self`/`this` -> same class methods only
            if not resolved_targets and receiver in ("self", "this") and parent_id and str(parent_id) in sym_meta:
                p_qname = sym_meta[str(parent_id)]["qualified_name"]
                resolved_targets = [
                    c for c in candidates if c["qualified_name"] == f"{p_qname}.{callee}"
                ]
        else:
            # Unqualified call `save()`
            # 2A. Same-file definition takes precedence (already linked by `lexical_same_blob`)
            same_file = [c for c in candidates if c["extraction_id"] == ext_id]
            if same_file:
                continue

            # 2B. Explicitly imported symbol (`from a import save` or `import { save } from './a'`)
            if callee in sym_imps:
                mod_spec = sym_imps[callee]
                resolved_targets = [
                    c for c in candidates if _module_spec_matches_path(mod_spec, c["path"])
                ]

            # 2C. Wildcard imports (`from a import *`)
            if not resolved_targets and wildcards:
                resolved_targets = [
                    c
                    for c in candidates
                    if any(_module_spec_matches_path(w, c["path"]) for w in wildcards)
                ]

            # 2D. Same directory/package definition if unique in that directory
            if not resolved_targets and caller_dir:
                same_dir = [
                    c
                    for c in candidates
                    if (c["path"].rsplit("/", 1)[0] if "/" in c["path"] else "") == caller_dir
                ]
                if len(same_dir) == 1:
                    resolved_targets = same_dir

            # 2E. Global fallback ONLY when the symbol name is globally unique across the entire snapshot!
            # (Prevents false edges when multiple modules define common names like `save` or `run`.)
            if not resolved_targets and len(candidates) == 1:
                resolved_targets = candidates

        for tgt in resolved_targets:
            rel_id = make_relation_id(
                from_kind="symbol",
                from_id=enc_id,
                to_kind="symbol",
                to_id=tgt["id"],
                relation_kind="calls_candidate",
                method="lexical_snapshot",
            )
            conn.execute(
                """
                INSERT INTO relations (
                    id, from_kind, from_id, to_kind, to_id, relation_kind, method, mapping, evidence_id, details_json, state
                ) VALUES (?, 'symbol', ?, 'symbol', ?, 'calls_candidate', 'lexical_snapshot', 'candidate', NULL, '{}', 'active')
                ON CONFLICT(id) DO NOTHING
                """,
                (rel_id, enc_id, tgt["id"]),
            )
