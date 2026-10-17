import shutil
import sqlite3
import subprocess
from typing import Any

from codecontext.errors import CodeContextError


def verify_sqlite_capabilities(conn: sqlite3.Connection | None = None) -> None:
    close_after = False
    if conn is None:
        conn = sqlite3.connect(":memory:")
        close_after = True
    try:
        try:
            conn.execute("CREATE VIRTUAL TABLE _cc_fts_check USING fts5(content)")
            conn.execute("DROP TABLE _cc_fts_check")
        except sqlite3.OperationalError as exc:
            raise CodeContextError(
                code="DEPENDENCY_MISSING",
                message="SQLite FTS5 extension is required but unavailable.",
                retryable=False,
                details={"capability": "sqlite_fts5"},
            ) from exc
        try:
            res = conn.execute("SELECT json_extract('{\"a\": 1}', '$.a')").fetchone()
            if not res or res[0] != 1:
                raise sqlite3.OperationalError("json_extract failed")
        except sqlite3.OperationalError as exc:
            raise CodeContextError(
                code="DEPENDENCY_MISSING",
                message="SQLite JSON functions are required but unavailable.",
                retryable=False,
                details={"capability": "sqlite_json"},
            ) from exc
    finally:
        if close_after:
            conn.close()


def verify_git_capabilities(git_executable: str = "git") -> tuple[int, int, int]:
    git_path = shutil.which(git_executable)
    if not git_path:
        raise CodeContextError(
            code="DEPENDENCY_MISSING",
            message="Git executable (>= 2.43) is required but was not found in PATH.",
            retryable=False,
            details={"dependency": "git", "minimum_version": "2.43.0"},
        )
    try:
        proc = subprocess.run(
            [git_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CodeContextError(
            code="DEPENDENCY_MISSING",
            message="Failed to execute git --version.",
            retryable=False,
            details={"dependency": "git"},
        ) from exc

    raw = proc.stdout.strip()
    # Expected format: "git version 2.43.0..."
    parts = raw.split()
    version_str = parts[2] if len(parts) >= 3 else ""
    num_parts = []
    for piece in version_str.split(".")[:3]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        if digits:
            num_parts.append(int(digits))
    while len(num_parts) < 3:
        num_parts.append(0)
    version_tuple = (num_parts[0], num_parts[1], num_parts[2])
    if version_tuple < (2, 43, 0):
        raise CodeContextError(
            code="DEPENDENCY_MISSING",
            message=f"Git >= 2.43 is required, found {version_str or raw}.",
            retryable=False,
            details={
                "dependency": "git",
                "minimum_version": "2.43.0",
                "found_version": version_str or raw,
            },
        )
    return version_tuple


def verify_treesitter_capabilities() -> dict[str, Any]:
    try:
        import tree_sitter
        import tree_sitter_javascript
        import tree_sitter_python
        import tree_sitter_typescript
        from tree_sitter import Language, Parser, Query

        languages = {
            "python": Language(tree_sitter_python.language()),
            "javascript": Language(tree_sitter_javascript.language()),
            "typescript": Language(tree_sitter_typescript.language_typescript()),
            "tsx": Language(tree_sitter_typescript.language_tsx()),
        }
        minimal_queries = {
            "python": "(function_definition name: (identifier) @name)",
            "javascript": "(function_declaration name: (identifier) @name)",
            "typescript": "(function_declaration name: (identifier) @name)",
            "tsx": "(function_declaration name: (identifier) @name)",
        }
        for name, lang in languages.items():
            parser = Parser(lang)
            tree = parser.parse(b"function f() {}\ndef g(): pass\n")
            if tree is None or tree.root_node is None:
                raise RuntimeError(f"Tree-sitter parser returned no root for {name}")
            Query(lang, minimal_queries[name])
        return {"languages": sorted(languages.keys()), "tree_sitter": getattr(tree_sitter, "__version__", "0.26.0")}
    except Exception as exc:
        if isinstance(exc, CodeContextError):
            raise
        raise CodeContextError(
            code="DEPENDENCY_MISSING",
            message="Tree-sitter grammars failed initialization or query compilation.",
            retryable=False,
            details={"dependency": "tree-sitter", "reason": type(exc).__name__},
        ) from exc


def verify_runtime_dependencies(conn: sqlite3.Connection | None = None, git_executable: str = "git") -> None:
    verify_sqlite_capabilities(conn)
    verify_git_capabilities(git_executable)
    verify_treesitter_capabilities()
