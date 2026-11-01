# CodeContext

Local project memory for coding agents. CodeContext preserves a repository's code, Git history, GitHub issues and pull request discussions, and documented decisions, then gives coding agents fast, cited access to the evidence relevant to their task.

## Features and scope

- **Three surfaces over one library:** Python SDK (`codecontext.Memory`), CLI (`codecontext` / `python -m codecontext`), and a read-only stdio MCP server with evidence tools, graph tools, repository discovery, and a `codecontext://status` resource. Supports `--auto-sync`, `--allow-sync`, and `--allow-enrich`.
- **Blob-owned syntax extraction (21 AST languages + Gradle DSL):** Tree-sitter syntax extraction for Python, JavaScript, TypeScript, JSX, TSX, Kotlin, Gradle Kotlin/Groovy DSL, Java, Go, Rust, C, C++, C#, Ruby, PHP, Bash/Shell, Swift, Scala, SQL, Protobuf, and GraphQL. Unchanged blobs are parsed once per `(blob_oid, language, extractor_fingerprint)` rather than duplicated across commits.
- **Structural code graph & architecture metrics:** Import- and receiver-type-aware candidate call resolution, exposed through `search_graph`, `trace_path`, `query_graph`, `get_architecture`, `detect_changes`, and `get_code_snippet`. Includes Louvain community clustering, Tarjan SCC cycle detection, module cohesion, and coupling metrics. `query_graph` supports a constrained, read-only Cypher subset.
- **SCIP Protobuf Import:** Ingest industry-standard SCIP index files (`import-scip`) to enrich the code graph with compiler-precise definitions, references, and hover documentation.
- **Architecture & ADR Ingestion:** Automatically ingests Architecture Decision Records (ADRs) and markdown documents from `docs/` and `architecture/` into structured evidence.
- **GitHub & GitLab parity:** Automatic remote detection and full ingestion for both GitHub (Issues, PRs, Reviews, GraphQL backfill) and GitLab (Issues, Merge Requests, Discussions).
- **Zero-dependency LLM & Vector Embedding Adapters:** Built-in adapters for OpenAI-compatible, Anthropic, Google Gemini, and Ollama services, supporting budgeted claim extraction and hybrid lexical/vector semantic search.
- **Owned Git and SQLite/FTS5 store (v2 Schema):** Preserves raw Git objects in an owned bare repository (`repository.git/`) and metadata, source versions, claims, summaries, 15 high-performance indexes, and lexical/vector indexes in `memory.sqlite3`.
- **Conservative rationale and applicability tracking:** Keeps `support`, `lifecycle`, `mapping`, and `applicability` independent. Historical rationale defaults to `historical_only` and is never upgraded to `corroborated` merely because code bytes are unchanged.
- **Read-only query guarantee:** `search`, `get`, `context`, and `status` never contact remote providers, invoke a model, run a parser, run migrations, or perform durable writes.
- **Workspace divergence detection:** Compares the prepared memory snapshot against the developer's workspace `HEAD`, staging index, and bounded working file bytes without executing Git hooks, `fsmonitor`, or clean/smudge filters.
- **Uncommitted tracked code:** `sync` with the default `HEAD` revision captures tracked working files, including staged additions and deletions, in a synthetic revision inside the owned store. It never changes your checkout, index, or branch. Untracked file inclusion is optionally supported with `--include-untracked`.

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

# Optional budgeted model enrichment with a built-in provider.
# The key comes from the environment; --model is required.
export ANTHROPIC_API_KEY=...
codecontext --store /abs/path/to/store enrich \
  --provider anthropic --model YOUR_MODEL --concurrency 4 \
  --max-input-tokens 100000 \
  --max-output-tokens 20000 \
  --max-calls 10
```

`enrich` takes either `--provider` (`openai`, `anthropic`, `gemini`, `ollama`) with `--model`, or your own adapter with `--adapter MODULE:FACTORY`. There is no default model. `--concurrency` runs up to 16 model calls at once; it defaults to 4 with `--provider` and 1 with `--adapter`.

Keys and endpoints come only from environment variables:

| Provider | Key | Endpoint override |
| --- | --- | --- |
| `openai` | `OPENAI_API_KEY` (optional for localhost) | `OPENAI_BASE_URL` |
| `anthropic` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| `gemini` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | none |
| `ollama` | none | `OLLAMA_HOST` |

To turn on hybrid vector search, add `--embedding-provider` (`openai`, `gemini`, `ollama`) and `--embedding-model` to `enrich` to store embeddings, and pass the same pair to `search` or `serve`.

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

Use `list_repositories` to discover configured roots and initialized submodules. Pass a returned absolute path in the optional `repository` argument of any query tool to select that store. With an explicit `--store` or `--workspace`, routing is limited to that repository and its initialized submodules. A globally launched server also accepts verified MCP client roots. Each repository has its own index, and submodule calls stay within their selected repository. Existing stores refresh with `sync`.

The server is read-only by default. If a repository or submodule has no `.codecontext` store, its tools return `STORE_NOT_INITIALIZED` and nothing is written. Run `codecontext init` first, or pass `serve --auto-init` to have the server create and sync missing stores. Startup errors go to stderr, because stdout carries the MCP protocol.

`serve --allow-sync` exposes a `sync` tool. `serve --allow-enrich` exposes an `enrich` tool and needs the same `--provider`/`--model` or `--adapter` flags as the `enrich` command.

Graph tools are enabled by default. `--include-graph-tools` remains accepted; use `--no-include-graph-tools` for only the evidence and repository-discovery tools.

## Export, privacy, and purge

- **Untrusted source content:** Repository code and discussions are treated strictly as untrusted data. CodeContext never executes repository hooks, `fsmonitor`, filters, or instructions embedded in comments.
- **Credentials and principal binding:** Tokens are never accepted as command-line flags. `GITHUB_TOKEN` (GitHub) and `GITLAB_TOKEN` (GitLab) are read from the environment during online `sync` only, and model API keys only when a built-in provider is called. None are written to `store.json`, `memory.sqlite3`, logs, model prompts, error messages, or exports. Private repository evidence is bound to the verified GitHub user principal ID.
- **Portable export (`export`):** Scrubs machine-specific paths and principal bindings using `secure_delete=ON` and `VACUUM`, producing a read-only archive with `manifest.json` and `OVERVIEW.md`. Full exports include retained Git history and any tracked workspace content captured by earlier syncs, including private code or secrets in those files.
- **Team share (`publish` / `join`):** `publish --to DIR` writes that same scrubbed snapshot plus `team.json` (origin identity, generation, and publish time). Teammates run `join --from DIR --repo PATH` into their own store, including `<repo>/.codecontext`. Join succeeds only when `origin` is the same project (HTTPS and SSH match). Pass `--merge` to perform a non-destructive union merge that incorporates remote claims, summaries, and Git objects without overwriting local progress. Tokens and machine paths stay out of the share; private Git content does not.
- **Purge (`purge --source-id ID` or `purge --entire-store`):** Removes a discussion source and its entire reverse-dependency closure (versions, evidence, claims, summaries, and FTS documents) or deletes the entire owned store after validating the ownership marker. Copies exported prior to a purge cannot be recalled.

## Known limitations

- One active prepared snapshot is maintained per store (`REVISION_NOT_PREPARED` is returned when querying another branch until `sync --revision REF` is run).
- Unavailable Git objects, shallow history boundaries, and unobserved prior comment edits cannot be reconstructed; they are surfaced explicitly in `coverage` and `gaps`.
