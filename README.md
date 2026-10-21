# CodeContext

Local project memory for coding agents. CodeContext preserves a repository's code, Git history, GitHub issues and pull request discussions, and documented decisions, then gives coding agents fast, cited access to the evidence relevant to their task.

## Features and scope

- **Three surfaces over one library:** Python SDK (`codecontext.Memory`), CLI (`codecontext` / `python -m codecontext`), and a read-only stdio MCP server with evidence tools, graph tools, repository discovery, and a `codecontext://status` resource.
- **Blob-owned syntax extraction (11 AST languages + Gradle DSL):** Tree-sitter syntax extraction for Python, JavaScript, TypeScript, JSX, TSX, Kotlin, Gradle Kotlin DSL (`.gradle.kts`), Java, Go, Rust, C, and C++. Unchanged blobs are parsed once per `(blob_oid, language, extractor_fingerprint)` rather than duplicated across commits.
- **Structural code graph:** Import- and receiver-type-aware candidate call resolution, exposed by default through `search_graph`, `trace_path`, `query_graph`, `get_architecture`, `detect_changes`, and `get_code_snippet`. `query_graph` supports a constrained, read-only Cypher subset. See [graph contracts and measurements](docs/graph-tools.md).
- **GitHub & GitLab history:** Automatic remote detection and ingestion for both GitHub Issues/PRs/Reviews and GitLab Issues/Merge Requests/Discussions.
- **Owned Git and SQLite/FTS5 store:** Preserves raw Git objects in an owned bare repository (`repository.git/`) and metadata, source versions, claims, summaries, and lexical FTS5 indexes in `memory.sqlite3`.
- **Conservative rationale and applicability tracking:** Keeps `support`, `lifecycle`, `mapping`, and `applicability` independent. Historical rationale defaults to `historical_only` and is never upgraded to `corroborated` merely because code bytes are unchanged.
- **Read-only query guarantee:** `search`, `get`, `context`, and `status` never contact GitHub, invoke a model, run a parser, run migrations, or perform durable writes.
- **Workspace divergence detection:** Compares the prepared memory snapshot against the developer's workspace `HEAD`, staging index, and bounded working file bytes without executing Git hooks, `fsmonitor`, or clean/smudge filters.
- **Uncommitted tracked code:** `sync` with the default `HEAD` revision captures tracked working files, including staged additions and deletions, in a synthetic revision inside the owned store. It never changes your checkout, index, or branch. Explicit branch names and commit hashes prepare committed content. Untracked files are excluded; stage new files before syncing them.

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

Use `list_repositories` to discover configured roots and initialized submodules. Pass a returned absolute path in the optional `repository` argument of any query tool to select that store. With an explicit `--store` or `--workspace`, routing is limited to that repository and its initialized submodules. A globally launched server also accepts verified MCP client roots. Each repository has its own index; selecting a submodule bootstraps its `.codecontext` store if needed. Existing stores refresh with `sync`, and submodule calls stay within their selected repository.

Graph tools are enabled by default. `--include-graph-tools` remains accepted; use `--no-include-graph-tools` for only the evidence and repository-discovery tools.

## Export, privacy, and purge

- **Untrusted source content:** Repository code and discussions are treated strictly as untrusted data. CodeContext never executes repository hooks, `fsmonitor`, filters, or instructions embedded in comments.
- **Credentials and principal binding:** `GITHUB_TOKEN` is read from the environment during online `sync` only and is never written to `store.json`, `memory.sqlite3`, logs, model prompts, or exports. Private repository evidence is bound to the verified GitHub user principal ID.
- **Portable export (`export`):** Scrubs machine-specific paths and principal bindings using `secure_delete=ON` and `VACUUM`, producing a read-only archive with `manifest.json` and `OVERVIEW.md`. Full exports include retained Git history and any tracked workspace content captured by earlier syncs, including private code or secrets in those files.
- **Team share (`publish` / `join`):** `publish --to DIR` writes that same scrubbed snapshot plus `team.json` (origin identity, generation, and publish time). Teammates run `join --from DIR --repo PATH` into their own store, including `<repo>/.codecontext`. Join succeeds only when `origin` is the same project (HTTPS and SSH match). Publish replaces the share only after you have joined its latest snapshot and then updated your own store. Joining a newer share replaces the local store; run `sync` again to rebuild it from your checkout. This is last-writer by publish time, not a merge. Tokens and machine paths stay out of the share; private Git content does not.
- **Purge (`purge --source-id ID` or `purge --entire-store`):** Removes a discussion source and its entire reverse-dependency closure (versions, evidence, claims, summaries, and FTS documents) or deletes the entire owned store after validating the ownership marker. Copies exported prior to a purge cannot be recalled.

## Known limitations

- One active prepared snapshot is maintained per store (`REVISION_NOT_PREPARED` is returned when querying another branch until `sync --revision REF` is run).
- Unavailable Git objects, shallow history boundaries, and unobserved prior comment edits cannot be reconstructed; they are surfaced explicitly in `coverage` and `gaps`.
