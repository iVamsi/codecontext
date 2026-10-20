# CodeContext

Local project memory for coding agents. CodeContext preserves a repository's code, Git history, GitHub issues and pull request discussions, and documented decisions, then gives coding agents fast, cited access to the evidence relevant to their task.

## Features and scope

- **Three surfaces over one library:** Python SDK (`codecontext.Memory`), CLI (`codecontext` / `python -m codecontext`), and a read-only stdio MCP server (`search`, `get`, `context` tools + `codecontext://status` resource).
- **Blob-owned syntax extraction (11 AST languages + Gradle DSL):** Tree-sitter syntax extraction for Python, JavaScript, TypeScript, JSX, TSX, Kotlin, Gradle Kotlin DSL (`.gradle.kts`), Java, Go, Rust, C, and C++. Unchanged blobs are parsed once per `(blob_oid, language, extractor_fingerprint)` rather than duplicated across commits.
- **Structural code graph & Cypher (`--include-graph-tools`):** Import- and receiver-type-aware call resolution across files, exposed through `search_graph`, `trace_path`, `query_graph` (strict Cypher with numeric comparisons, `OR`/`AND`/`NOT`, variable-length paths `*1..3`, `COUNT(...)`, and `ORDER BY`), `get_architecture`, `detect_changes`, and `get_code_snippet`.
- **GitHub & GitLab history:** Automatic remote detection and ingestion for both GitHub Issues/PRs/Reviews and GitLab Issues/Merge Requests/Discussions.
- **Owned Git and SQLite/FTS5 store:** Preserves raw Git objects in an owned bare repository (`repository.git/`) and metadata, source versions, claims, summaries, and lexical FTS5 indexes in `memory.sqlite3`.
- **Conservative rationale and applicability tracking:** Keeps `support`, `lifecycle`, `mapping`, and `applicability` independent. Historical rationale defaults to `historical_only` and is never upgraded to `corroborated` merely because code bytes are unchanged.
- **Read-only query guarantee:** `search`, `get`, `context`, and `status` never contact GitHub, invoke a model, run a parser, run migrations, or perform durable writes.
- **Workspace divergence detection:** Compares the prepared memory snapshot against the developer's workspace `HEAD`, staging index, and bounded working file bytes without executing Git hooks, `fsmonitor`, or clean/smudge filters.

## Requirements and installation

- **Operating systems:** Linux and macOS (Windows support is deferred).
- **Python:** 3.12 or 3.13.
- **Git:** `>= 2.43` with SQLite FTS5 and JSON extensions enabled.

Install and verify with `uv`:

```bash
uv sync --frozen
uv run pytest -q
uv run ruff check src tests examples evals
```

## Quickstart

### 1. Initialize and synchronize a store

```bash
# Initialize a store outside the indexed repository checkout
codecontext --store /abs/path/to/store init \
  --repo /abs/path/to/project \
  --github owner/repo

# Focus-first synchronization for immediate use on a file/line
codecontext --store /abs/path/to/store sync \
  --path src/retry.py --line 18 \
  --max-requests 100 --max-seconds 120

# Optional budgeted model enrichment using an explicit module:factory adapter
codecontext --store /abs/path/to/store enrich \
  --adapter examples.fixture_adapter:create_adapter \
  --max-input-tokens 100000 \
  --max-output-tokens 20000 \
  --max-calls 10
```

### 2. Query from CLI or Python SDK

```bash
codecontext --store /abs/path/to/store search "retry" --kind code
codecontext --store /abs/path/to/store context "Why is retry count limited?" \
  --path src/retry.py --line 18 \
  --workspace /abs/path/to/project
codecontext --store /abs/path/to/store status
```

```python
from pathlib import Path
from codecontext import ContextRequest, InitRequest, Memory, SyncRequest

store_path = Path("/abs/path/to/store")
repo_path = Path("/abs/path/to/project")

with Memory.initialize(store_path, InitRequest(repo_path=repo_path)) as memory:
    memory.sync(SyncRequest(revision="HEAD", offline=True))

with Memory.open(store_path, read_only=True) as memory:
    memory.configure_workspace(repo_path)
    pack = memory.context(
        ContextRequest(
            query="Explain the constraints behind the retry limit.",
            target={"kind": "path", "path": "src/retry.py", "line": 18},
        )
    )
    print(pack.model_dump_json(indent=2))
```

### 3. Configure MCP for coding agents

Add the server definition (see [`examples/mcp-config.json`](examples/mcp-config.json)) to your MCP host configuration:

```json
{
  "command": "/absolute/path/to/environment/bin/codecontext",
  "args": [
    "--store",
    "/absolute/path/to/project-memory",
    "serve",
    "--workspace",
    "/absolute/path/to/project"
  ]
}
```

Add the guidance snippet in [`examples/agent-instructions.md`](examples/agent-instructions.md) to your project's agent instructions.

## Export, privacy, and purge

- **Untrusted source content:** Repository code and discussions are treated strictly as untrusted data. CodeContext never executes repository hooks, `fsmonitor`, filters, or instructions embedded in comments.
- **Credentials and principal binding:** `GITHUB_TOKEN` is read from the environment during online `sync` only and is never written to `store.json`, `memory.sqlite3`, logs, model prompts, or exports. Private repository evidence is bound to the verified GitHub user principal ID.
- **Portable export (`export`):** Scrubs machine-specific paths and principal bindings using `secure_delete=ON` and `VACUUM`, producing a read-only archive with `manifest.json` and `OVERVIEW.md`. Private code and secrets already committed to Git history remain source evidence in full exports.
- **Team share (`publish` / `join`):** `publish --to DIR` writes that same scrubbed snapshot plus `team.json` (origin identity, generation, and publish time). Teammates run `join --from DIR --repo PATH` into their own store, including `<repo>/.codecontext`. Join succeeds only when `origin` is the same project (HTTPS and SSH match). Publish replaces the share only after you have joined its latest snapshot and then updated your own store. Joining a newer share replaces the local store; run `sync` again to rebuild it from your checkout. This is last-writer by publish time, not a merge. Tokens and machine paths stay out of the share; private Git content does not.
- **Purge (`purge --source-id ID` or `purge --entire-store`):** Removes a discussion source and its entire reverse-dependency closure (versions, evidence, claims, summaries, and FTS documents) or deletes the entire owned store after validating the ownership marker. Copies exported prior to a purge cannot be recalled.

## Known limitations

- One active prepared snapshot is maintained per store (`REVISION_NOT_PREPARED` is returned when querying another branch until `sync --revision REF` is run).
- Unavailable Git objects, shallow history boundaries, and unobserved prior comment edits cannot be reconstructed; they are surfaced explicitly in `coverage` and `gaps`.
