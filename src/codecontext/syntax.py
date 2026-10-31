import contextvars
import importlib.metadata
import json
import os
import re
import sqlite3
import unicodedata
from collections.abc import Callable, Iterable
from concurrent.futures import CancelledError, ThreadPoolExecutor
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

from codecontext.deadline import SyncDeadlineExceeded, check_deadline
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
from codecontext.scip_import import retire_dangling_scip_relations
from codecontext.store import Store

QUERIES_DIR = Path(__file__).parent / "queries"
MAX_PARSE_BYTES = 2 * 1024 * 1024  # 2 MiB
CHUNK_MAX_BYTES = 8192
CHUNK_OVERLAP_BYTES = 512
# Checking the clock on every node is measurable on large files; 256 nodes take well under a millisecond.
DEADLINE_CHECK_INTERVAL = 256
# Tree-sitter releases the GIL while parsing, but walking the tree is Python work, so more threads stop helping.
PARSE_WORKERS = min(8, os.cpu_count() or 1)
# Thread start-up costs more than parsing a handful of small files.
PARALLEL_PARSE_MIN_BLOBS = 4
# Bounds how many blobs and parse results are held in memory before they are written.
PARSE_BATCH_SIZE = 64

# Node types that name a referenced type, per grammar. JavaScript has no type annotations.
TYPE_REF_LEAVES: dict[str, frozenset[str]] = {
    "python": frozenset({"identifier", "attribute"}),
    "typescript": frozenset({"type_identifier", "nested_type_identifier"}),
    "tsx": frozenset({"type_identifier", "nested_type_identifier"}),
    "java": frozenset({"type_identifier", "scoped_type_identifier"}),
    "kotlin": frozenset({"user_type"}),
}

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
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".php": "php",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".swift": "swift",
    ".scala": "scala",
    ".sc": "scala",
    ".sql": "sql",
    ".proto": "protobuf",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".gradle": "groovy",
}

STRUCTURAL_LANGUAGES: frozenset[str] = frozenset(
    {
        "c_sharp",
        "ruby",
        "php",
        "bash",
        "swift",
        "scala",
        "sql",
        "protobuf",
        "graphql",
        "groovy",
    }
)

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
        walker_version = 5 if name == "kotlin" else 3
        fingerprints[name] = f"{name}:ts{ts_ver}:g{lang_pkg_vers[name]}:q{q_hash}:walker{walker_version}"
    return langs, queries, fingerprints


LANGUAGES, COMPILED_QUERIES, EXTRACTOR_FINGERPRINTS = _load_languages_and_queries()


def get_extractor_fingerprint(language: str) -> str:
    return EXTRACTOR_FINGERPRINTS.get(language, f"{language}:structural:v1")


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


def _push_children(stack: list[tuple[Node, Any, list[str]]], children: list[Node], parent_sym: Any, scope: list[str]) -> None:
    # Reversed so the stack pops children in source order, matching a recursive pre-order walk.
    stack.extend((ch, parent_sym, scope) for ch in reversed(children))


def _collect_error_ranges(root: Node, out: list[tuple[int, int]]) -> None:
    stack = [root]
    visited = 0
    while stack:
        node = stack.pop()
        visited += 1
        if visited % DEADLINE_CHECK_INTERVAL == 0:
            check_deadline()
        if node.type == "ERROR" or node.is_missing:
            out.append((node.start_byte, node.end_byte))
        stack.extend(reversed(node.children))


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


def _parse_structural_fallback(
    blob_bytes: bytes,
    language: str,
    actual_oid: str,
    fp: str,
    ext_id: str,
) -> ParseResult:
    """Deterministic structural AST extraction for languages without compiled tree-sitter grammars."""
    text = blob_bytes.decode("utf-8", errors="replace")
    symbols: list[ExtractedSymbol] = []
    occurrences: list[ExtractedOccurrence] = []
    relations: list[CandidateRelation] = []

    def add_sym(
        name: str,
        kind: str,
        byte_start: int,
        byte_end: int,
        parent_id: str | None = None,
        qual_name: str | None = None,
        sig: str = "",
        body_start: int | None = None,
        body_end: int | None = None,
    ) -> ExtractedSymbol:
        qname = qual_name or name
        l_start, l_end = byte_span_to_lines(blob_bytes, byte_start, byte_end)
        if l_start is None or l_end is None:
            l_start, l_end = 1, 1
        sid = make_sym_id(
            blob_oid=actual_oid,
            language=language,
            extractor_fingerprint=fp,
            byte_start=byte_start,
            byte_end=byte_end,
            kind=kind,
            qualified_name=qname,
        )
        sym = ExtractedSymbol(
            id=sid,
            parent_symbol_id=parent_id,
            kind=kind,
            name=name,
            qualified_name=qname,
            byte_start=byte_start,
            byte_end=byte_end,
            body_byte_start=body_start,
            body_byte_end=body_end,
            line_start=l_start,
            line_end=l_end,
            signature=sig or name,
            docstring=None,
        )
        symbols.append(sym)
        if parent_id:
            relations.append(
                CandidateRelation(
                    from_symbol_id=parent_id,
                    to_symbol_id=sid,
                    relation_kind="contains",
                    method="structural_containment",
                )
            )
        return sym

    def add_occ(enclosing_id: str | None, occ_kind: str, b_start: int, b_end: int, details: dict[str, Any] | None = None) -> None:
        occ_text = text[b_start:b_end].strip()[:1000]
        if not occ_text:
            return
        occ_id = make_id(
            "ext_",
            {
                "byte_end": b_end,
                "byte_start": b_start,
                "extraction_id": ext_id,
                "kind": occ_kind,
                "text": occ_text,
            },
        )
        occurrences.append(
            ExtractedOccurrence(
                id=occ_id,
                enclosing_symbol_id=enclosing_id,
                kind=occ_kind,
                text=occ_text,
                byte_start=b_start,
                byte_end=b_end,
                details=details or {},
            )
        )

    if language == "c_sharp":
        for m in re.finditer(r"\b(?:public|private|protected|internal|static|abstract|sealed|\s)*\b(class|interface|struct|enum|record)\s+([A-Za-z_]\w*)(?:<[^>]+>)?(?:\s*:\s*([^{]+))?\s*\{", text):
            skind = "interface" if m.group(1) == "interface" else ("enum" if m.group(1) == "enum" else ("struct" if m.group(1) == "struct" else "class"))
            c_name = m.group(2)
            c_sym = add_sym(c_name, skind, m.start(), m.end(), sig=m.group(0).strip(" {"))
            if m.group(3):
                for b_cand in m.group(3).split(","):
                    b_clean = b_cand.strip().split("<")[0]
                    if b_clean:
                        add_occ(c_sym.id, "inherits", m.start(3), m.end(3), {"base_class": b_clean})
        for m in re.finditer(r"\b(?:public|private|protected|internal|static|async|virtual|override|abstract|\s)+\b([A-Za-z0-9_<>[\]?]+)\s+([A-Z]\w*)\s*\(([^)]*)\)\s*\{", text):
            fn_name = m.group(2)
            add_sym(fn_name, "method", m.start(), m.end(), sig=m.group(0).strip(" {"))

    elif language == "ruby":
        for m in re.finditer(r"^\s*class\s+([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)(?:\s*<\s*([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*))?", text, re.MULTILINE):
            c_name = m.group(1)
            c_sym = add_sym(c_name, "class", m.start(), m.end(), sig=m.group(0).strip())
            if m.group(2):
                add_occ(c_sym.id, "inherits", m.start(2), m.end(2), {"base_class": m.group(2)})
        for m in re.finditer(r"^\s*module\s+([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)", text, re.MULTILINE):
            add_sym(m.group(1), "namespace", m.start(), m.end(), sig=m.group(0).strip())
        for m in re.finditer(r"^\s*def\s+(?:self\.)?([a-zA-Z_]\w*[?!=]?)", text, re.MULTILINE):
            add_sym(m.group(1), "method", m.start(), m.end(), sig=m.group(0).strip())

    elif language == "php":
        for m in re.finditer(r"\b(?:abstract|final|\s)*\b(class|interface|trait|enum)\s+([A-Za-z_]\w*)(?:\s+extends\s+([A-Za-z0-9_\\]+))?(?:\s+implements\s+([A-Za-z0-9_\\,\s]+))?\s*\{", text):
            skind = "interface" if m.group(1) == "interface" else "class"
            c_name = m.group(2)
            c_sym = add_sym(c_name, skind, m.start(), m.end(), sig=m.group(0).strip(" {"))
            if m.group(3):
                add_occ(c_sym.id, "inherits", m.start(3), m.end(3), {"base_class": m.group(3)})
        for m in re.finditer(r"\b(?:public|private|protected|static|abstract|final|\s)*\bfunction\s+([A-Za-z_]\w*)\s*\(([^)]*)\)", text):
            add_sym(m.group(1), "function", m.start(), m.end(), sig=m.group(0).strip())

    elif language == "bash":
        for m in re.finditer(r"^\s*(?:function\s+)?([a-zA-Z0-9_.-]+)\s*\(\s*\)\s*\{|^\s*function\s+([a-zA-Z0-9_.-]+)\s*\{", text, re.MULTILINE):
            fn_name = m.group(1) or m.group(2)
            add_sym(fn_name, "function", m.start(), m.end(), sig=f"function {fn_name}()")

    elif language == "swift":
        for m in re.finditer(r"\b(?:public|private|fileprivate|internal|open|final|\s)*\b(class|struct|enum|protocol|actor|extension)\s+([A-Za-z_]\w*)(?:<[^>]+>)?(?:\s*:\s*([^{]+))?\s*\{", text):
            skind = "interface" if m.group(1) == "protocol" else "class"
            c_sym = add_sym(m.group(2), skind, m.start(), m.end(), sig=m.group(0).strip(" {"))
            if m.group(3):
                for b_part in m.group(3).split(","):
                    clean = b_part.strip().split("<")[0]
                    if clean:
                        add_occ(c_sym.id, "inherits", m.start(3), m.end(3), {"base_class": clean})
        for m in re.finditer(r"\b(?:public|private|fileprivate|internal|open|static|override|\s)*\bfunc\s+([a-zA-Z_]\w*)(?:<[^>]+>)?\s*\(([^)]*)\)", text):
            add_sym(m.group(1), "function", m.start(), m.end(), sig=m.group(0).strip())

    elif language == "scala":
        for m in re.finditer(r"\b(?:abstract|sealed|final|\s)*\b(class|object|trait)\s+([A-Za-z_]\w*)(?:<[^>]+>)?(?:\s+extends\s+([A-Za-z0-9_.]+))?(?:\s+with\s+([^{]+))?\s*\{", text):
            skind = "interface" if m.group(1) == "trait" else "class"
            c_sym = add_sym(m.group(2), skind, m.start(), m.end(), sig=m.group(0).strip(" {"))
            if m.group(3):
                add_occ(c_sym.id, "inherits", m.start(3), m.end(3), {"base_class": m.group(3)})
        for m in re.finditer(r"\bdef\s+([a-zA-Z_]\w*)\s*(?:\[[^\]]+\])?(?:\s*\(([^)]*)\))?", text):
            add_sym(m.group(1), "function", m.start(), m.end(), sig=m.group(0).strip())

    elif language == "sql":
        for m in re.finditer(r"\bCREATE\s+(?:OR\s+REPLACE\s+)?(TABLE|VIEW|PROCEDURE|FUNCTION|TRIGGER|INDEX)\s+(?:IF\s+NOT\s+EXISTS\s+)?([`\"\[]?([a-zA-Z0-9_.]+)[`\"\]]?)", text, re.IGNORECASE):
            skind = m.group(1).lower()
            name = m.group(3).strip("`\"[]")
            add_sym(name, skind, m.start(), m.end(), sig=m.group(0).strip())

    elif language == "protobuf":
        for m in re.finditer(r"\b(message|service|enum)\s+([A-Za-z_]\w*)\s*\{", text):
            skind = "interface" if m.group(1) == "service" else ("enum" if m.group(1) == "enum" else "class")
            add_sym(m.group(2), skind, m.start(), m.end(), sig=m.group(0).strip(" {"))
        for m in re.finditer(r"\brpc\s+([A-Za-z_]\w*)\s*\(([^)]+)\)\s*returns\s*\(([^)]+)\)", text):
            add_sym(m.group(1), "method", m.start(), m.end(), sig=m.group(0).strip())

    elif language == "graphql":
        for m in re.finditer(r"\b(type|interface|union|input|enum)\s+([A-Za-z_]\w*)", text):
            skind = "interface" if m.group(1) == "interface" else ("enum" if m.group(1) == "enum" else "class")
            add_sym(m.group(2), skind, m.start(), m.end(), sig=m.group(0).strip())

    elif language == "groovy":
        for m in re.finditer(r"\bclass\s+([A-Za-z_]\w*)\s*\{", text):
            add_sym(m.group(1), "class", m.start(), m.end(), sig=m.group(0).strip(" {"))
        for m in re.finditer(r"\btask\s+([a-zA-Z_]\w*)", text):
            add_sym(m.group(1), "function", m.start(), m.end(), sig=m.group(0).strip())
        for m in re.finditer(r"\bdef\s+([a-zA-Z_]\w*)\s*\(", text):
            add_sym(m.group(1), "function", m.start(), m.end(), sig=m.group(0).strip(" ("))

    # Without a grammar, any `name(` inside a function's span is the best available call signal.
    for sym in symbols:
        if sym.kind in ("function", "method", "async_function", "async_method"):
            body_text = text[sym.byte_start:sym.byte_end]
            for cm in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", body_text):
                cname = cm.group(1)
                if cname not in ("if", "for", "while", "switch", "catch", "return", sym.name):
                    add_occ(
                        sym.id,
                        "call",
                        sym.byte_start + cm.start(1),
                        sym.byte_start + cm.end(1),
                        {"callee": cname, "raw_callee": cname},
                    )

    return ParseResult(
        extraction_id=ext_id,
        blob_oid=actual_oid,
        language=language,
        extractor_fingerprint=fp,
        state="complete",
        diagnostics=[],
        symbols=symbols,
        occurrences=occurrences,
        relations=relations,
    )


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
    check_deadline()
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
        if language in STRUCTURAL_LANGUAGES:
            return _parse_structural_fallback(blob_bytes, language, actual_oid, fp, ext_id)
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
    def read_source(offset: int, _point: Any) -> bytes:
        check_deadline()
        return blob_bytes[offset:offset + 32768]

    # Use blob slices for node text: Node.text re-enters this fallible callback in native code.
    tree = parser.parse(read_source)
    check_deadline()
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

    def text_of(n: Node) -> str:
        return blob_bytes[n.start_byte : n.end_byte].decode("utf-8", errors="replace").strip()

    def ref_name(n: Node) -> str:
        # Generic arguments and call arguments are not part of the referenced name.
        return "".join(re.split(r"[<(\[]", text_of(n), maxsplit=1)[0].split())

    def emit_type_refs(sym_id: str, roots: list[Node]) -> None:
        leaves = TYPE_REF_LEAVES.get(language, frozenset())
        # Python names parameters with the same node type as types, so only names under a `type` count.
        # A call inside an annotation (`Field(...)`) names a function, not a type.
        stack = [(r, language != "python") for r in reversed(roots)]
        while stack:
            n, inside = stack.pop()
            if inside and n.type in leaves:
                if n.type == "user_type":
                    name = ".".join(text_of(c) for c in n.named_children if c.type in ("identifier", "type_identifier"))
                    stack.extend((c, True) for c in reversed(n.named_children) if c.type == "type_arguments")
                else:
                    name = text_of(n)
                if name:
                    add_occurrence(sym_id, "references_type", n.start_byte, n.end_byte, {"type_name": name})
                continue
            if n.type == "call":
                continue
            inside = inside or n.type == "type"
            stack.extend((c, inside) for c in reversed(n.children))

    def emit_declaration_edges(sym_id: str, decl: Node, outer: Node) -> None:
        decorators: list[Node] = []
        supertypes: list[tuple[str, Node]] = []
        type_roots: list[Node | None] = []
        if language == "python":
            if outer.type == "decorated_definition":
                decorators = [d.named_children[0] for d in outer.children if d.type == "decorator" and d.named_children]
            if decl.type == "class_definition":
                supers = decl.child_by_field_name("superclasses")
                if supers is not None:
                    supertypes = [("inherits", c) for c in supers.named_children if c.type in ("identifier", "attribute")]
            else:
                type_roots = [decl.child_by_field_name("parameters"), decl.child_by_field_name("return_type")]
        elif language in ("javascript", "typescript", "tsx"):
            # TypeScript puts member decorators before the member as class-body siblings.
            owners: list[Node] = [outer, decl] if outer != decl else [decl]
            leading: list[Node] = []
            prev = decl.prev_named_sibling if outer == decl else None
            while prev is not None and prev.type == "decorator":
                leading.append(prev)
                prev = prev.prev_named_sibling
            raw_decorators = [*reversed(leading), *(c for o in owners for c in o.children if c.type == "decorator")]
            decorators = [d.named_children[0] for d in raw_decorators if d.named_children]
            for clause in decl.children:
                if clause.type not in ("class_heritage", "extends_type_clause"):
                    continue
                for part in clause.named_children:
                    if part.type == "implements_clause":
                        supertypes += [("implements", t) for t in part.named_children]
                    elif part.type == "extends_clause":
                        supertypes += [("inherits", t) for t in part.named_children if t.type != "type_arguments"]
                    elif part.type != "type_arguments":
                        supertypes.append(("inherits", part))
            type_roots = [decl.child_by_field_name("parameters"), decl.child_by_field_name("return_type")]
        elif language == "java":
            mods = next((c for c in decl.children if c.type == "modifiers"), None)
            if mods is not None:
                decorators = [n for a in mods.named_children if a.type in ("annotation", "marker_annotation")
                              if (n := a.child_by_field_name("name")) is not None]
            for clause in decl.children:
                kind = {"superclass": "inherits", "super_interfaces": "implements", "extends_interfaces": "inherits"}.get(clause.type)
                if kind is None:
                    continue
                parts = clause.named_children
                if len(parts) == 1 and parts[0].type == "type_list":
                    parts = parts[0].named_children
                supertypes += [(kind, t) for t in parts]
            if decl.type in ("method_declaration", "constructor_declaration"):
                type_roots = [decl.child_by_field_name("type"), decl.child_by_field_name("parameters")]
        elif language == "kotlin":
            mods = next((c for c in decl.children if c.type == "modifiers"), None)
            if mods is not None:
                decorators = [n for a in mods.named_children if a.type == "annotation"
                              if (n := next((c for c in a.named_children if c.type in ("user_type", "constructor_invocation")), None)) is not None]
            # Kotlin marks a superclass by its constructor call; a bare supertype on a class is an interface.
            is_interface = any(c.type == "interface" for c in decl.children)
            delegation = next((c for c in decl.children if c.type == "delegation_specifiers"), None)
            for spec in delegation.named_children if delegation is not None else []:
                inner = spec.named_children[0] if spec.named_children else None
                if inner is None:
                    continue
                if inner.type == "constructor_invocation":
                    supertypes.append(("inherits", inner))
                elif inner.type == "explicit_delegation" and inner.named_children:
                    supertypes.append(("implements", inner.named_children[0]))
                elif inner.type == "user_type":
                    supertypes.append(("inherits" if is_interface else "implements", inner))
            type_roots = [c for c in decl.children if c.type in ("function_value_parameters", "user_type", "nullable_type", "primary_constructor")]
        for n in decorators:
            name = ref_name(n)
            if name:
                add_occurrence(sym_id, "decorates", n.start_byte, n.end_byte, {"decorator": name})
        for kind, n in supertypes:
            name = ref_name(n)
            if name:
                add_occurrence(sym_id, kind, n.start_byte, n.end_byte, {"base_class": name})
        emit_type_refs(sym_id, [r for r in type_roots if r is not None])

    def emit_import(enclosing_id: str | None, n: Node, module: str, imported: str, local: str) -> None:
        if module and imported and local:
            add_occurrence(enclosing_id, "imports", n.start_byte, n.end_byte,
                           {"imported": imported, "module": module, "name": local})

    def emit_qualified_import(enclosing_id: str | None, node: Node) -> None:
        if any(c.type in ("asterisk", "*") for c in node.children):
            return
        target = next((c for c in node.named_children if c.type in ("scoped_identifier", "qualified_identifier", "identifier")), None)
        if target is None:
            return
        full = text_of(target)
        if "." not in full:
            return
        module, imported = full.rsplit(".", 1)
        alias = next((c for c in node.named_children if c.type == "identifier" and c != target), None)
        emit_import(enclosing_id, node, module, imported, text_of(alias) if alias is not None else imported)

    def walk_python(root_node: Node) -> None:
        stack: list[tuple[Node, ExtractedSymbol | None, list[str]]] = [(root_node, None, [])]
        visited = 0
        while stack:
            node, parent_sym, scope = stack.pop()
            visited += 1
            if visited % DEADLINE_CHECK_INTERVAL == 0:
                check_deadline()
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
                    if sym is not None:
                        emit_declaration_edges(sym.id, inner, node)
                        if body_n is not None:
                            _push_children(stack, body_n.children, sym, [*scope, sym.name])
                    continue

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
                if sym is not None:
                    emit_declaration_edges(sym.id, node, node)
                    if body_n is not None:
                        _push_children(stack, body_n.children, sym, [*scope, sym.name])
                continue

            if node.type == "expression_statement" and parent_sym is None:
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

            if node.type == "type" and parent_sym is not None:
                # Nested `type` nodes (generics, unions) are covered by the outermost one. The walk still
                # descends because annotations can hold real calls, such as `Annotated[int, Field(...)]`.
                anc = node.parent
                while anc is not None and anc.type not in ("type", "block", "module") and not anc.type.endswith("statement"):
                    anc = anc.parent
                if anc is None or anc.type != "type":
                    emit_type_refs(parent_sym.id, [node])

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
                mod_n = node.child_by_field_name("module_name")
                for name_n in node.children_by_field_name("name") if mod_n is not None else []:
                    orig_n = name_n.child_by_field_name("name") if name_n.type == "aliased_import" else name_n
                    alias_n = name_n.child_by_field_name("alias") if name_n.type == "aliased_import" else name_n
                    if orig_n is not None and alias_n is not None:
                        emit_import(parent_sym.id if parent_sym else None, name_n, text_of(mod_n), text_of(orig_n), text_of(alias_n))
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

            _push_children(stack, node.children, parent_sym, scope)

    def walk_js_ts(root_node: Node) -> None:
        # A non-None fourth item marks a variable declarator and carries the statement that owns its display range.
        stack: list[tuple[Node, ExtractedSymbol | None, list[str], Node | None]] = [(root_node, None, [], None)]
        visited = 0
        while stack:
            node, parent_sym, scope, declarator_outer = stack.pop()
            visited += 1
            if visited % DEADLINE_CHECK_INTERVAL == 0:
                check_deadline()
            if declarator_outer is not None:
                _handle_js_ts_declarator(node, declarator_outer, parent_sym, scope, stack)
                continue

            # The export keyword belongs to the declaration's display range.
            if node.type == "export_statement":
                add_occurrence(
                    parent_sym.id if parent_sym else None,
                    "export",
                    node.start_byte,
                    node.end_byte,
                )
                decl = node.child_by_field_name("declaration")
                if decl is not None:
                    _handle_js_ts_declaration(decl, outer_node=node, parent_sym=parent_sym, scope=scope, stack=stack)
                    continue

            if _handle_js_ts_declaration(node, outer_node=node, parent_sym=parent_sym, scope=scope, stack=stack):
                continue

            if node.type == "import_statement":
                raw_imp = blob_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
                imp_kind = "type_import" if raw_imp.lstrip().startswith("import type") else "import"
                add_occurrence(
                    parent_sym.id if parent_sym else None,
                    imp_kind,
                    node.start_byte,
                    node.end_byte,
                )
                src_n = node.child_by_field_name("source")
                clause = next((c for c in node.named_children if c.type == "import_clause"), None)
                if src_n is not None and clause is not None:
                    module = text_of(src_n).strip("'\"")
                    enc_id = parent_sym.id if parent_sym else None
                    for part in clause.named_children:
                        if part.type == "identifier":
                            emit_import(enc_id, part, module, text_of(part), text_of(part))
                        elif part.type == "named_imports":
                            for spec in part.named_children:
                                orig_n = spec.child_by_field_name("name")
                                if spec.type != "import_specifier" or orig_n is None:
                                    continue
                                alias_n = spec.child_by_field_name("alias") or orig_n
                                emit_import(enc_id, spec, module, text_of(orig_n), text_of(alias_n))
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
            elif node.type == "type_annotation" and parent_sym is not None:
                # Function types nest further annotations; this outermost one already covers them.
                emit_type_refs(parent_sym.id, [node])
                continue

            stack.extend((ch, parent_sym, scope, None) for ch in reversed(node.children))

    def _handle_js_ts_declaration(
        decl: Node,
        *,
        outer_node: Node,
        parent_sym: ExtractedSymbol | None,
        scope: list[str],
        stack: list[tuple[Node, ExtractedSymbol | None, list[str], Node | None]],
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
            if sym is not None:
                emit_declaration_edges(sym.id, decl, outer_node)
            target_walk = body_n if body_n is not None else decl
            next_sym = sym if sym is not None else parent_sym
            next_scope = [*scope, sym.name] if sym is not None else scope
            stack.extend((ch, next_sym, next_scope, None) for ch in reversed(target_walk.children))
            return True

        if dtype in ("lexical_declaration", "variable_declaration"):
            declarators = [c for c in decl.named_children if c.type == "variable_declarator"]
            stack.extend((vdecl, parent_sym, scope, outer_node) for vdecl in reversed(declarators))
            return True

        return False

    def _handle_js_ts_declarator(
        vdecl: Node,
        outer_node: Node,
        parent_sym: ExtractedSymbol | None,
        scope: list[str],
        stack: list[tuple[Node, ExtractedSymbol | None, list[str], Node | None]],
    ) -> None:
        name_n = vdecl.child_by_field_name("name")
        val_n = vdecl.child_by_field_name("value")
        if name_n is None or name_n.type != "identifier":
            return
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
        siblings = [c for c in vdecl.parent.named_children if c.type == "variable_declarator"] if vdecl.parent else [vdecl]
        single = len(siblings) == 1
        d_start = outer_node.start_byte if single else vdecl.start_byte
        d_end = outer_node.end_byte if single else vdecl.end_byte
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
            stack.extend((ch, next_sym, next_scope, None) for ch in reversed(val_n.children))

    def walk_kotlin_java(root_node: Node) -> None:
        stack: list[tuple[Node, ExtractedSymbol | None, list[str]]] = [(root_node, None, [])]
        visited = 0
        while stack:
            node, parent_sym, scope = stack.pop()
            visited += 1
            if visited % DEADLINE_CHECK_INTERVAL == 0:
                check_deadline()
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
                if sym is not None:
                    emit_declaration_edges(sym.id, node, node)
                target_walk = body_n if body_n is not None else node
                next_sym = sym if sym is not None else parent_sym
                next_scope = [*scope, sym.name] if sym is not None else scope
                _push_children(stack, target_walk.children, next_sym, next_scope)
                continue

            if ntype in ("property_declaration", "call_expression"):
                # These branches scan siblings and ancestors, so their cost is not bounded by the node count.
                check_deadline()
            if language == "kotlin" and ntype == "property_declaration" and parent_sym is not None:
                variable = next((c for c in node.named_children if c.type == "variable_declaration"), None)
                if variable is not None and variable.named_children:
                    type_node = next((c for c in variable.named_children if c.type in ("user_type", "nullable_type")), None)
                    if type_node is not None:
                        emit_type_refs(parent_sym.id, [type_node])
                        if parent_sym.kind == "class":
                            name_node = variable.named_children[0]
                            add_occurrence(parent_sym.id, "property_type", node.start_byte, node.end_byte,
                                           details={"name": blob_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8"),
                                                    "type": blob_bytes[type_node.start_byte : type_node.end_byte].decode("utf-8").rstrip("?")})
            if language == "java" and ntype in ("field_declaration", "local_variable_declaration") and parent_sym is not None:
                type_node = node.child_by_field_name("type")
                if type_node is not None:
                    emit_type_refs(parent_sym.id, [type_node])
            if ntype in ("import", "import_declaration"):
                add_occurrence(
                    parent_sym.id if parent_sym else None,
                    "import",
                    node.start_byte,
                    node.end_byte,
                )
                emit_qualified_import(parent_sym.id if parent_sym else None, node)
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
                    # Gradle Kotlin DSL blocks (`plugins`, `android`, ...) act as named sections, so they become symbols.
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
                        _push_children(stack, lambda_body.children, next_sym, next_scope)
                        continue

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
                                    name_node = variable.named_children[0]
                                    var_name = blob_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8")
                                    if var_name != re.split(r"[?.!]", recv_str)[0]:
                                        continue
                                    type_node = next((c for c in variable.named_children if c.type in ("user_type", "nullable_type")), None)
                                    # An untyped local still shadows a typed parameter or field.
                                    det_dict["receiver_type"] = blob_bytes[type_node.start_byte : type_node.end_byte].decode("utf-8").rstrip("?") if type_node else ""
                                    break
                                ancestor = ancestor.parent
                    add_occurrence(
                        parent_sym.id if parent_sym else None,
                        "call",
                        node.start_byte,
                        node.end_byte,
                        details=det_dict,
                    )

            _push_children(stack, node.children, parent_sym, scope)

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

    def walk_go_rust_c_cpp(root_node: Node) -> None:
        stack: list[tuple[Node, ExtractedSymbol | None, list[str]]] = [(root_node, None, [])]
        visited = 0
        while stack:
            node, parent_sym, scope = stack.pop()
            visited += 1
            if visited % DEADLINE_CHECK_INTERVAL == 0:
                check_deadline()
            ntype = node.type

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
                continue

            # Go methods are scoped under their receiver type, which lives outside the method body.
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
                    _push_children(stack, body_n.children, sym or parent_sym, [*m_scope, sym.name] if sym else m_scope)
                continue

            # A Rust `impl` block is not a symbol itself; it only scopes the methods inside it.
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
                    _push_children(stack, body_n.children, parent_sym, next_scope)
                continue

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
                    _push_children(stack, body_n.children, sym or parent_sym, [*scope, sym.name] if sym else scope)
                continue

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
                    # A forward declaration has no body and defines nothing.
                    continue
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
                _push_children(stack, target_walk.children, next_sym, next_scope)
                continue

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
                    # Calls resolve by their last name segment; `.`, `::` and `->` all qualify it.
                    callee = re.split(r"(?:\.|::|->)", raw_callee)[-1].strip()
                    if callee:
                        add_occurrence(
                            parent_sym.id if parent_sym else None,
                            "call",
                            node.start_byte,
                            node.end_byte,
                            details={"callee": callee, "raw_callee": raw_callee},
                        )

            _push_children(stack, node.children, parent_sym, scope)

    if language == "python":
        walk_python(root)
    elif language in ("kotlin", "java"):
        walk_kotlin_java(root)
    elif language in ("go", "rust", "c", "cpp"):
        walk_go_rust_c_cpp(root)
    else:
        walk_js_ts(root)

    # Resolve same-file lexical call candidates where a named definition exists
    symbols_by_id = {s.id: s for s in symbols}
    by_simple_name: dict[str, list[ExtractedSymbol]] = {}
    for s in symbols:
        by_simple_name.setdefault(s.name, []).append(s)

    for occ in occurrences:
        check_deadline()
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
    parsed: ParseResult | None = None,
) -> tuple[ParseResult, bool]:
    """Ensure `(blob_oid, language, extractor_fingerprint)` is extracted in `extractions` and `symbols`.
    `parsed` is a result already computed off this thread; it is used only when the store has no extraction yet.
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

    res = parsed if parsed is not None else parse_blob(blob_bytes, language, fp, blob_oid=blob_oid)
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


def _parse_pending_blobs(
    store: Store,
    rows: list[sqlite3.Row],
    blob_payloads: dict[str, tuple[int, bytes] | None],
) -> tuple[dict[str, ParseResult], set[str]]:
    """Parse the uncached blobs behind `rows` on a thread pool.
    Returns `(parsed, unparsed)`, both keyed by `"<blob_oid>:<language>"`. `unparsed` holds jobs cut off by the
    sync deadline. Small batches return two empty values and the caller parses inline.
    No SQLite connection leaves this thread.
    """
    jobs: dict[str, tuple[str, str, bytes]] = {}
    for row in rows:
        oid = str(row["blob_oid"])
        lang = detect_language_for_path(bytes(row["path"]))
        payload = blob_payloads.get(oid)
        if lang is not None and payload is not None:
            jobs.setdefault(f"{oid}:{lang}", (oid, lang, payload[1]))
    if len(jobs) < PARALLEL_PARSE_MIN_BLOBS:
        return {}, set()
    with store.connect_read() as r_conn:
        for key, (oid, lang, _) in list(jobs.items()):
            ext_id = make_extraction_id(blob_oid=oid, language=lang, extractor_fingerprint=get_extractor_fingerprint(lang))
            if r_conn.execute("SELECT 1 FROM extractions WHERE id = ?", (ext_id,)).fetchone():
                del jobs[key]
    if len(jobs) < PARALLEL_PARSE_MIN_BLOBS:
        return {}, set()
    parsed: dict[str, ParseResult] = {}
    with ThreadPoolExecutor(max_workers=PARSE_WORKERS) as pool:
        # Pool threads do not inherit context variables, and the sync deadline lives in one.
        futures = {
            key: pool.submit(contextvars.copy_context().run, parse_blob, data, lang, blob_oid=oid)
            for key, (oid, lang, data) in jobs.items()
        }
        try:
            for key, future in futures.items():
                try:
                    parsed[key] = future.result()
                except (SyncDeadlineExceeded, CancelledError):
                    # Keep finished results so the caller can checkpoint them; a stuck batch would never progress.
                    for pending in futures.values():
                        pending.cancel()
        except BaseException:
            for pending in futures.values():
                pending.cancel()
            raise
    return parsed, set(jobs) - set(parsed)


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

    # Keep one reader open and checkpoint each file before starting the next.
    blob_payloads: dict[str, tuple[int, bytes] | None] = {}
    with mirror.open_batch_session() as batch:
        for chunk_start in range(0, len(pending_rows), PARSE_BATCH_SIZE):
            chunk = pending_rows[chunk_start : chunk_start + PARSE_BATCH_SIZE]
            for row in chunk:
                check_deadline()
                oid = str(row["blob_oid"])
                if oid not in blob_payloads:
                    obj = batch.read_object(oid)
                    blob_payloads[oid] = None if obj is None else (obj[1], obj[2])
            parsed_ahead, unparsed = _parse_pending_blobs(store, chunk, blob_payloads)
            for row in chunk:
                if f"{row['blob_oid']}:{detect_language_for_path(bytes(row['path']))}" in unparsed:
                    continue
                # After a pooled parse every remaining write is cheap, so finish them before honouring the deadline.
                if not parsed_ahead and not unparsed:
                    check_deadline()
                with store.write_transaction(bump_generation=True) as conn:
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
                            parsed=parsed_ahead.get(f"{blob_oid}:{lang}"),
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
            if unparsed:
                raise SyncDeadlineExceeded

    with mirror.open_batch_session() as cached_batch:
        while True:
            check_deadline()
            with store.write_transaction(bump_generation=True) as conn:
                current_files = conn.execute("SELECT path, blob_oid, extraction_id, content_state FROM snapshot_files ORDER BY path").fetchall()
                manifest = {bytes(r["path"]).hex(): [r["blob_oid"], r["extraction_id"], r["content_state"]] for r in current_files}
                checkpoint = conn.execute("SELECT cursor_json FROM checkpoints WHERE stream_key = 'syntax:documents'").fetchone()
                previous = json.loads(checkpoint[0]) if checkpoint else {}
                if not conn.execute("SELECT 1 FROM checkpoints WHERE stream_key = 'syntax:chunk_dedup:v1'").fetchone():
                    # Repair chunk duplicates created by earlier shared-extraction refreshes.
                    conn.execute("""
                        DELETE FROM search_documents WHERE rowid IN (
                            SELECT rowid FROM (
                                SELECT rowid, ROW_NUMBER() OVER (
                                    PARTITION BY json_extract(metadata_json, '$.path'),
                                                 json_extract(metadata_json, '$.path_bytes_b64'),
                                                 chunk_ordinal
                                    ORDER BY rowid DESC
                                ) AS position
                                FROM search_documents
                                WHERE record_kind = 'code'
                                  AND json_extract(metadata_json, '$.evidence_id') IS NOT NULL
                            ) WHERE position > 1
                        )
                    """)
                    conn.execute("INSERT INTO checkpoints (stream_key, cursor_json) VALUES ('syntax:chunk_dedup:v1', '{}')")
                dirty_paths = {p for p in manifest.keys() | previous.keys() if manifest.get(p) != previous.get(p)}
                if not dirty_paths and checkpoint:
                    Store.resolve_gaps_in_scope(conn, scope="syntax:documents", code="STRUCTURE_PENDING")
                    return blobs_parsed, cache_hits
                remaining_paths = dirty_paths - set(sorted(dirty_paths)[:64])
                dirty_paths -= remaining_paths
                dirty_extractions = {m[p][1] for m in (manifest, previous) for p in dirty_paths if p in m and m[p][1]}
                if checkpoint is None:
                    conn.execute("DELETE FROM search_documents WHERE record_kind = 'code'")
                else:
                    for ext_id in dirty_extractions:
                        conn.execute("DELETE FROM search_documents WHERE record_kind = 'code' AND record_id IN (SELECT id FROM symbols WHERE extraction_id = ?)", (ext_id,))
                    for path_hex in dirty_paths:
                        disp, b64 = path_bytes_to_display(bytes.fromhex(path_hex))
                        conn.execute("DELETE FROM search_documents WHERE record_kind = 'code' AND json_extract(metadata_json, '$.path') = ? AND json_extract(metadata_json, '$.path_bytes_b64') IS ?", (disp, b64))
                indexed_files = [r for r in current_files if r["content_state"] in ("indexed", "text_only")
                                 and (r["extraction_id"] in dirty_extractions or bytes(r["path"]).hex() in dirty_paths)]

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
                    check_deadline()
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
                        b_obj = cached_batch.read_object(b_oid)
                        if b_obj is not None:
                            raw_blob_b = b_obj[2]

                    for srow in sym_rows:
                        check_deadline()
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
                        check_deadline()
                        p_bytes = bytes(r["path"])
                        if p_bytes.hex() not in dirty_paths:
                            continue
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
                if remaining_paths:
                    Store.upsert_gap(conn, repository_id=repo_id, code="STRUCTURE_PENDING", scope="syntax:documents",
                                     message="Code search and call relationships are awaiting refresh.", retryable=True)
                else:
                    _resolve_cross_file_calls_and_imports(conn)
                    Store.resolve_gaps_in_scope(conn, scope="syntax:documents", code="STRUCTURE_PENDING")
                for path_hex in dirty_paths:
                    if path_hex in manifest:
                        previous[path_hex] = manifest[path_hex]
                    else:
                        previous.pop(path_hex, None)
                conn.execute("""
                    INSERT INTO checkpoints (stream_key, cursor_json) VALUES ('syntax:documents', ?)
                    ON CONFLICT(stream_key) DO UPDATE SET cursor_json = excluded.cursor_json
                """, (canonical_json(previous),))

                if not remaining_paths:
                    return blobs_parsed, cache_hits


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


def _module_spec_key(mod_spec: str) -> str:
    """Normalize an import specifier to the slash path that `_module_spec_matches_path` compares."""
    clean_spec = mod_spec.strip().strip("'\"").lstrip("./")
    for ext in EXTENSION_TO_LANGUAGE:
        if clean_spec.endswith(ext):
            clean_spec = clean_spec[: -len(ext)]
            break
    return clean_spec.replace(".", "/").strip("/")


def _path_module_forms(file_path: str) -> tuple[str, str]:
    fp_no_ext = file_path
    for ext in EXTENSION_TO_LANGUAGE:
        if fp_no_ext.endswith(ext):
            fp_no_ext = fp_no_ext[: -len(ext)]
            break
    if fp_no_ext.endswith("/__init__") or fp_no_ext.endswith("/index"):
        return fp_no_ext, fp_no_ext.rsplit("/", 1)[0]
    return fp_no_ext, fp_no_ext


def _segment_suffixes(value: str, sep: str) -> set[str]:
    """Every `t` where `value == t` or `value.endswith(sep + t)`."""
    parts = value.split(sep)
    return {sep.join(parts[i:]) for i in range(len(parts))}


def _path_module_keys(file_path: str) -> set[str]:
    """Every spec key that `_module_spec_matches_path` would accept for `file_path`."""
    fp_no_ext, fp_pkg = _path_module_forms(file_path)
    return _segment_suffixes(fp_no_ext, "/") | _segment_suffixes(fp_pkg, "/")


def _module_spec_matches_path(mod_spec: str, file_path: str) -> bool:
    """Check whether an import module specifier (`a`, `pkg.a`, `./service.js`, `com.example.Repo`)
    matches an indexed repository `file_path` (`a.py`, `src/pkg/a.py`, `src/service.ts`, etc.).
    """
    spec_slash = _module_spec_key(mod_spec)
    if not spec_slash:
        return False
    fp_no_ext, fp_pkg = _path_module_forms(file_path)
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


SNAPSHOT_RELATION_METHODS = (
    "lexical_snapshot",
    "ast_imports",
    "ast_inherits",
    "ast_implements",
    "ast_decorates",
    "ast_references_type",
    "override_resolution",
    "test_convention",
)
TYPE_TARGET_KINDS = frozenset({"class", "interface"})
DECORATOR_TARGET_KINDS = frozenset({"function", "async_function", "class"})


def _root_name(name: str) -> str:
    return re.split(r"[?.!(]", name.removeprefix("this.").removeprefix("self."))[0]


def _import_matches(module: str, imported: str, path: str) -> bool:
    # Python and JS name the defining file; Java and Kotlin name the package, so try `module.imported` as the file.
    return _module_spec_matches_path(module, path) or _module_spec_matches_path(f"{module}.{imported}", path)


def _resolve_cross_file_calls_and_imports(conn: sqlite3.Connection) -> None:
    """Resolve cross-file `calls_candidate` relations using:
    1. Explicit import target matching (`from a import save` binds `save()` ONLY to `a.py`'s `save`, never `b.py`'s `save`)
    2. Module alias receiver matching (`a.save()` where `import a` binds to `a.py`'s `save`)
    3. Parameter/field receiver type matching (`repo.save()` where `repo: UserRepository` binds to `UserRepository.save`)
    4. Same-directory/package preference
    5. Global fallback ONLY when the symbol name is globally unique (1 match) across the snapshot.
    Then resolve `imports`, `inherits`, `implements`, `decorates`, `references_type`, `overrides` and `tests`.
    """
    placeholders = ",".join("?" for _ in SNAPSHOT_RELATION_METHODS)
    previous_relations = {r[0] for r in conn.execute(
        f"SELECT id FROM relations WHERE method IN ({placeholders})", SNAPSHOT_RELATION_METHODS)}
    active_relations: set[str] = set()

    active_rows = conn.execute(
        """
        SELECT s.id, s.extraction_id, s.parent_symbol_id, s.kind, s.name, s.qualified_name, s.signature,
               sf.path
        FROM symbols s
        JOIN snapshot_files sf ON sf.extraction_id = s.extraction_id
        WHERE s.kind IN ('function', 'async_function', 'method', 'async_method', 'class', 'interface')
        """
    ).fetchall()

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

    # Per-name indexes so each call resolves by lookup instead of scanning every same-named definition.
    def index_by(keys_of: Callable[[dict[str, Any]], Iterable[Any]]) -> dict[tuple[str, Any], list[dict[str, Any]]]:
        index: dict[tuple[str, Any], list[dict[str, Any]]] = {}
        for infos in by_name.values():
            for info in infos:
                for key in keys_of(info):
                    index.setdefault((info["name"], key), []).append(info)
        return index

    by_name_ext = index_by(lambda c: (c["extraction_id"],))
    by_name_dir = index_by(lambda c: (c["path"].rsplit("/", 1)[0] if "/" in c["path"] else "",))
    by_name_module = index_by(lambda c: _path_module_keys(c["path"]))
    by_name_qname = index_by(lambda c: (c["qualified_name"],))
    by_name_qsuffix = index_by(lambda c: _segment_suffixes(c["qualified_name"], "."))
    by_name_parent = index_by(lambda c: (c["parent_symbol_id"],))

    def module_matches(name: str, mod_spec: str) -> list[dict[str, Any]]:
        key = _module_spec_key(mod_spec)
        return by_name_module.get((name, key), []) if key else []

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

    property_types: dict[tuple[str, str], set[str]] = {}
    for info in sym_meta.values():
        if info["kind"] == "class":
            for name, type_name in re.findall(r"\b(?:val|var)\s+(\w+)\s*:\s*([\w.]+)", info["signature"]):
                property_types.setdefault((info["id"], name), set()).add(type_name)
    for row in conn.execute("SELECT enclosing_symbol_id, details_json FROM syntax_occurrences WHERE kind = 'property_type'"):
        if row["enclosing_symbol_id"] in sym_meta:
            detail = json.loads(row["details_json"])
            property_types.setdefault((row["enclosing_symbol_id"], detail["name"]), set()).add(detail["type"])

    call_rows = conn.execute(
        """
        SELECT DISTINCT so.extraction_id, so.enclosing_symbol_id, so.details_json
        FROM syntax_occurrences so
        JOIN snapshot_files sf ON sf.extraction_id = so.extraction_id
        WHERE so.kind IN ('call', 'jsx_component') AND so.enclosing_symbol_id IS NOT NULL
        """
    ).fetchall()

    # Module-level imports have no enclosing symbol, so an import is credited to the symbols that use the name.
    name_users: dict[tuple[str, str], set[str]] = {}
    for cr in call_rows:
        check_deadline()
        enc_id = str(cr["enclosing_symbol_id"])
        ext_id = str(cr["extraction_id"])
        det = json.loads(str(cr["details_json"])) if cr["details_json"] else {}
        callee = det.get("callee") or det.get("component")
        receiver = det.get("receiver")
        used = receiver or callee
        if isinstance(used, str) and used:
            name_users.setdefault((ext_id, _root_name(used)), set()).add(enc_id)
        enc_info = sym_meta.get(enc_id)
        if enc_info is None:
            continue

        if not isinstance(callee, str) or callee not in by_name:
            continue

        def others(found: list[dict[str, Any]], enc_id: str = enc_id) -> list[dict[str, Any]]:
            return [c for c in found if c["id"] != enc_id]

        # A symbol id belongs to one extraction, so the caller's own rows all sit in that bucket.
        own_rows = by_name_ext.get((callee, enc_info["extraction_id"]), [])
        candidate_count = len(by_name[callee]) - sum(1 for c in own_rows if c["id"] == enc_id)
        if candidate_count == 0:
            continue

        sym_imps, mod_aliases, wildcards = ext_parsed_imports.get(ext_id, ({}, {}, []))
        caller_path = enc_info["path"]
        caller_dir = caller_path.rsplit("/", 1)[0] if "/" in caller_path else ""

        # Constructor-injected fields live in the parent class signature, so both signatures supply receiver types.
        recv_types: dict[str, str] = {}
        parent_id = enc_info.get("parent_symbol_id")
        if parent_id and str(parent_id) in sym_meta:
            parent_info = sym_meta[str(parent_id)]
            recv_types.update(_extract_receiver_types_from_signature(parent_info["signature"]))
        recv_types.update(_extract_receiver_types_from_signature(enc_info["signature"]))
        if isinstance(receiver, str) and "receiver_type" in det:
            recv_types[re.split(r"[?.!]", receiver)[0]] = det["receiver_type"]

        resolved_targets: list[dict[str, Any]] = []

        if isinstance(receiver, str) and receiver:
            clean_recv = receiver.replace("this.", "").replace("self.", "").strip()
            chain = re.sub(r"[?!]", "", clean_recv).split(".")
            root_types = ({recv_types[chain[0]]} if chain[0] in recv_types
                          else property_types.get((parent_id, chain[0]), set()))
            if len(chain) > 1 and all(part.isidentifier() for part in chain):
                types = root_types
                for property_name in chain[1:]:
                    types = {
                        target_type
                        for type_name in types
                        for owner in by_name.get(type_name.rsplit(".", 1)[-1], [])
                        if owner["kind"] == "class"
                        for target_type in property_types.get((owner["id"], property_name), set())
                    }
                owner_ids = {
                    owner["id"] for type_name in types
                    for owner in by_name.get(type_name.rsplit(".", 1)[-1], []) if owner["kind"] == "class"
                }
                resolved_targets = [
                    c for owner_id in owner_ids for c in others(by_name_parent.get((callee, owner_id), []))
                ]
            # `a.save()` after `import a` or `from pkg import a`.
            if not resolved_targets and clean_recv in mod_aliases:
                resolved_targets = others(module_matches(callee, mod_aliases[clean_recv]))
            elif clean_recv in sym_imps:
                # `Service.run()` where `Service` was imported.
                resolved_targets = others(
                    by_name_qname.get((callee, f"{clean_recv}.{callee}"), [])
                    + module_matches(callee, sym_imps[clean_recv])
                )
            # `api.getUser()` where `api: Api` is a parameter or field.
            if not resolved_targets and len(chain) == 1 and root_types:
                resolved_targets = [
                    c for t_name in root_types
                    for c in others(by_name_qsuffix.get((callee, f"{t_name}.{callee}"), []))
                ]
            # `self` and `this` can only reach methods of the same class.
            if not resolved_targets and receiver in ("self", "this") and parent_id and str(parent_id) in sym_meta:
                p_qname = sym_meta[str(parent_id)]["qualified_name"]
                resolved_targets = others(by_name_qname.get((callee, f"{p_qname}.{callee}"), []))
        else:
            # A same-file definition shadows imports and is already linked by `lexical_same_blob`.
            if others(by_name_ext.get((callee, ext_id), [])):
                continue

            # `from a import save` or `import { save } from './a'`.
            if callee in sym_imps:
                resolved_targets = others(module_matches(callee, sym_imps[callee]))

            # `from a import *`.
            if not resolved_targets and wildcards:
                resolved_targets = [c for w in wildcards for c in others(module_matches(callee, w))]

            # Code often calls a sibling module without importing it (same Go or Java package).
            if not resolved_targets and caller_dir:
                same_dir = others(by_name_dir.get((callee, caller_dir), []))
                if len(same_dir) == 1:
                    resolved_targets = same_dir

            # Only a globally unique name is safe; common names like `save` or `run` would create false edges.
            if not resolved_targets and candidate_count == 1:
                resolved_targets = others(by_name[callee])

        for tgt in resolved_targets:
            rel_id = make_relation_id(
                from_kind="symbol",
                from_id=enc_id,
                to_kind="symbol",
                to_id=tgt["id"],
                relation_kind="calls_candidate",
                method="lexical_snapshot",
            )
            active_relations.add(rel_id)
            if rel_id in previous_relations:
                continue
            conn.execute(
                """
                INSERT INTO relations (
                    id, from_kind, from_id, to_kind, to_id, relation_kind, method, mapping, evidence_id, details_json, state
                ) VALUES (?, 'symbol', ?, 'symbol', ?, 'calls_candidate', 'lexical_snapshot', 'candidate', NULL, '{}', 'active')
                ON CONFLICT(id) DO NOTHING
                """,
                (rel_id, enc_id, tgt["id"]),
            )

    def link(from_id: str, to_id: str, rel_kind: str, method: str) -> None:
        rel_id = make_relation_id(
            from_kind="symbol", from_id=from_id, to_kind="symbol", to_id=to_id, relation_kind=rel_kind, method=method
        )
        active_relations.add(rel_id)
        conn.execute(
            """
            INSERT INTO relations (
                id, from_kind, from_id, to_kind, to_id, relation_kind, method, mapping, evidence_id, details_json, state
            ) VALUES (?, 'symbol', ?, 'symbol', ?, ?, ?, 'candidate', NULL, '{}', 'active')
            ON CONFLICT(id) DO NOTHING
            """,
            (rel_id, from_id, to_id, rel_kind, method),
        )

    occ_rows = conn.execute(
        """
        SELECT DISTINCT o.extraction_id, o.enclosing_symbol_id, o.kind, o.details_json
        FROM syntax_occurrences o
        JOIN snapshot_files sf ON sf.extraction_id = o.extraction_id
        WHERE o.kind IN ('inherits', 'implements', 'decorates', 'references_type')
          AND o.enclosing_symbol_id IS NOT NULL
        """
    ).fetchall()

    class_inheritance: dict[str, set[str]] = {}
    for orow in occ_rows:
        check_deadline()
        enc_id = str(orow["enclosing_symbol_id"])
        ext_id = str(orow["extraction_id"])
        okind = str(orow["kind"])
        try:
            odetails = json.loads(str(orow["details_json"]))
        except json.JSONDecodeError:
            odetails = {}

        target_name = odetails.get("base_class") or odetails.get("decorator") or odetails.get("type_name")
        if not isinstance(target_name, str) or not target_name:
            continue
        name_users.setdefault((ext_id, _root_name(target_name)), set()).add(enc_id)
        short_target = target_name.rsplit(".", 1)[-1].rsplit("::", 1)[-1]
        wanted_kinds = DECORATOR_TARGET_KINDS if okind == "decorates" else TYPE_TARGET_KINDS
        cands = [c for c in by_name.get(short_target, []) if c["id"] != enc_id and c["kind"] in wanted_kinds]
        if not cands:
            continue
        # Same rules as calls: same file, then the file's imports, then a globally unique name.
        targets = [c for c in cands if c["extraction_id"] == ext_id]
        sym_imps = ext_parsed_imports.get(ext_id, ({}, {}, []))[0]
        imported_from = sym_imps.get(_root_name(target_name))
        if not targets and imported_from:
            targets = [c for c in cands if _import_matches(imported_from, short_target, c["path"])]
        if not targets and len(cands) == 1:
            targets = cands
        for tgt in targets:
            link(enc_id, tgt["id"], okind, f"ast_{okind}")
            if okind in ("inherits", "implements"):
                class_inheritance.setdefault(enc_id, set()).add(tgt["id"])

    import_rows = conn.execute(
        """
        SELECT DISTINCT o.extraction_id, o.enclosing_symbol_id, o.details_json
        FROM syntax_occurrences o
        JOIN snapshot_files sf ON sf.extraction_id = o.extraction_id
        WHERE o.kind = 'imports'
        """
    ).fetchall()
    for irow in import_rows:
        check_deadline()
        ext_id = str(irow["extraction_id"])
        idetails = json.loads(str(irow["details_json"]))
        imported, module, local = idetails.get("imported"), idetails.get("module"), idetails.get("name")
        if not (isinstance(imported, str) and isinstance(module, str) and isinstance(local, str)):
            continue
        cands = [c for c in by_name.get(imported, []) if c["extraction_id"] != ext_id]
        targets = [c for c in cands if _import_matches(module, imported, c["path"])]
        if not targets and len(cands) == 1:
            targets = cands
        if not targets:
            continue
        enc = irow["enclosing_symbol_id"]
        sources = [str(enc)] if enc else sorted(name_users.get((ext_id, local), ()))
        for src in sources:
            for tgt in targets:
                if src != tgt["id"]:
                    link(src, tgt["id"], "imports", "ast_imports")

    # A method overrides a same-named method of a resolved superclass or interface.
    methods_by_parent: dict[str, list[dict[str, Any]]] = {}
    for _sid, sinfo in sym_meta.items():
        if sinfo.get("parent_symbol_id"):
            methods_by_parent.setdefault(str(sinfo["parent_symbol_id"]), []).append(sinfo)

    for child_cls_id, parent_cls_ids in class_inheritance.items():
        child_methods = methods_by_parent.get(child_cls_id, [])
        for p_id in parent_cls_ids:
            parent_methods = {m["name"]: m for m in methods_by_parent.get(p_id, [])}
            for cm in child_methods:
                if cm["name"] in parent_methods:
                    pm = parent_methods[cm["name"]]
                    rel_id = make_relation_id(
                        from_kind="symbol",
                        from_id=cm["id"],
                        to_kind="symbol",
                        to_id=pm["id"],
                        relation_kind="overrides",
                        method="override_resolution",
                    )
                    active_relations.add(rel_id)
                    conn.execute(
                        """
                        INSERT INTO relations (
                            id, from_kind, from_id, to_kind, to_id, relation_kind, method, mapping, evidence_id, details_json, state
                        ) VALUES (?, 'symbol', ?, 'symbol', ?, 'overrides', 'override_resolution', 'exact', NULL, '{}', 'active')
                        ON CONFLICT(id) DO NOTHING
                        """,
                        (rel_id, cm["id"], pm["id"]),
                    )

    # Test names follow `test_x`, `XTest` or `x_test` conventions, which point at the tested symbol.
    for sid, sinfo in sym_meta.items():
        sname = sinfo["name"]
        if sname.startswith("test_") or sname.endswith("Test") or sname.endswith("_test"):
            cand_tested: list[str] = []
            if sname.startswith("test_"):
                cand_tested.append(sname[5:])
            if sname.endswith("Test"):
                cand_tested.append(sname[:-4])
            if sname.endswith("_test"):
                cand_tested.append(sname[:-5])
            for ct in cand_tested:
                if ct in by_name:
                    for tgt in by_name[ct]:
                        rel_id = make_relation_id(
                            from_kind="symbol",
                            from_id=sid,
                            to_kind="symbol",
                            to_id=tgt["id"],
                            relation_kind="tests",
                            method="test_convention",
                        )
                        active_relations.add(rel_id)
                        conn.execute(
                            """
                            INSERT INTO relations (
                                id, from_kind, from_id, to_kind, to_id, relation_kind, method, mapping, evidence_id, details_json, state
                            ) VALUES (?, 'symbol', ?, 'symbol', ?, 'tests', 'test_convention', 'candidate', NULL, '{}', 'active')
                            ON CONFLICT(id) DO NOTHING
                            """,
                            (rel_id, sid, tgt["id"]),
                        )

    conn.executemany("DELETE FROM relations WHERE id = ?", [(rid,) for rid in previous_relations - active_relations])
    # SCIP edges come from a separate import, so only retire the ones whose symbols left the snapshot.
    retire_dangling_scip_relations(conn)
