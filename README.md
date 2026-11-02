# CodeContext

[![CI](https://github.com/iVamsi/codecontext/actions/workflows/ci.yml/badge.svg)](https://github.com/iVamsi/codecontext/actions/workflows/ci.yml)

CodeContext is local project memory for coding agents. It stores a repository's code, Git history, and GitHub or GitLab discussions on your machine, then answers questions with citations. You can use the `codecontext` command, the `codecontext` Python package, or a Model Context Protocol (MCP) server.

## Requirements

- Linux or macOS (CI runs on both)
- Python 3.12 or 3.13
- Git 2.43 or newer
- Python's SQLite with full-text search (FTS5) and JSON functions (CPython includes both)

## Install

```bash
uv tool install git+https://github.com/iVamsi/codecontext
```

To work on this repository:

```bash
git clone https://github.com/iVamsi/codecontext
cd codecontext
uv sync --frozen
uv run ruff check src tests examples evals
uv run pytest tests/unit tests/integration -q
```

CI runs that lint, those tests, and `uv build` on Ubuntu and macOS for Python 3.12 and 3.13. It runs on every push to `main` and every pull request.

## Quickstart

Keep the store outside the repository you index. `sync` reads a GitHub token from `GITHUB_TOKEN`, `GH_TOKEN`, `gh auth token`, or a GitHub MCP config. It reads a GitLab token from `GITLAB_TOKEN`, `GLAB_TOKEN`, `glab`, or a GitLab MCP config on the same host. Pass tokens through those sources. CodeContext keeps them out of the store, logs, and exports.

```bash
codecontext --store /path/to/store init \
  --repo /path/to/project \
  --github owner/repo

codecontext --store /path/to/store sync
codecontext --store /path/to/store search "retry"
codecontext --store /path/to/store context \
  "Why is the retry count limited?" \
  --path src/retry.py --line 18
```

`sync` indexes the revision you name (`HEAD` by default), including tracked files that differ from that commit. Your checkout stays as it is. `--include-untracked` adds untracked files. `--offline` skips the remote. `--use-graphql` backfills GitHub history in bulk.

The Python API matches these commands. `Memory.open(store, read_only=True)` is the query path:

```python
from pathlib import Path
from codecontext import ContextRequest, Memory

with Memory.open(Path("/path/to/store"), read_only=True) as memory:
    memory.configure_workspace(Path("/path/to/project"))
    pack = memory.context(
        ContextRequest(
            query="Explain the constraints behind the retry limit.",
            target={"kind": "path", "path": "src/retry.py", "line": 18},
        )
    )
```

## MCP server

Add the server to your agent host. [`examples/mcp-config.json`](examples/mcp-config.json) is a full sample, and [`examples/agent-instructions.md`](examples/agent-instructions.md) is guidance you can give the agent.

```json
{
  "command": "codecontext",
  "args": ["--store", "/path/to/store", "serve", "--workspace", "/path/to/project"]
}
```

`serve` reads the local store. `search`, `get`, `context`, `status`, and the graph tools do that read and nothing else: no remote calls, no model calls, and no parser runs. `--auto-init` creates a missing `.codecontext` store. `--allow-sync` exposes `sync`. `--allow-enrich` exposes `enrich`. Graph tools stay on unless you pass `--no-include-graph-tools`.

`list_repositories` returns configured roots and initialized submodules. Pass one of those paths as `repository` on any query tool.

## Index

Tree-sitter grammars ship for Python, JavaScript (including JSX), TypeScript, TSX, Kotlin (including `.kts`), Java, Go, Rust, C, and C++. C#, Ruby, PHP, shell, Swift, Scala, SQL, Protobuf, GraphQL, and Gradle Groovy use a structural extractor.

Graph tools search symbols (`search_graph`), trace calls (`trace_path`), run a bounded read-only subset of Cypher (`query_graph`), summarize modules (`get_architecture`), trace the callers of a change (`detect_changes`), and return source (`get_code_snippet`). `import-scip` loads a SCIP index (Sourcegraph's code-intelligence format) when you want compiler-precise references.

`sync` also reads Architecture Decision Records and design docs: `ARCHITECTURE.md`, `DESIGN.md`, `DECISIONS.md`, and Markdown under `adr/`, `docs/adr/`, `decisions/`, and `docs/architecture/`. `--no-ingest-docs` skips that.

`enrich` extracts claims with a budget. Pass `--provider` (`openai`, `anthropic`, `gemini`, or `ollama`) and `--model`, or `--adapter MODULE:FACTORY`. There is no default model. Keys come from `OPENAI_API_KEY` (optional for a localhost endpoint), `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY` / `GOOGLE_API_KEY`. Ollama needs no key. Override endpoints with `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, or `OLLAMA_HOST`.

For hybrid lexical and vector search, pass the same `--embedding-provider` (`openai`, `gemini`, or `ollama`) and `--embedding-model` to `enrich`, then to `search` or `serve`.

## Share or delete a store

`export --destination DIR` writes a portable archive and strips machine paths and account bindings. A full export still contains the Git history that was synced.

`publish --to DIR` writes that snapshot for teammates. `join --from DIR --repo PATH` copies it into a personal store when the checkout's origin is the same project. `--merge` unions the shared claims, summaries, and Git objects into the local store.

`purge --source-id ID` removes one discussion and the records that depend on it. `purge --entire-store` deletes the store after the ownership check.

Report security issues through [GitHub Security Advisories](https://github.com/iVamsi/codecontext/security/advisories/new). Details are in [SECURITY.md](SECURITY.md).

## Limits

Each store keeps one prepared snapshot. Another branch returns `REVISION_NOT_PREPARED` until you run `sync --revision REF`.

Missing Git objects, a shallow clone, and comment edits that were never fetched stay listed as coverage gaps.

## License

[MIT](LICENSE)
