# Graph tool contracts

The MCP server exposes all graph tools by default. Their names resemble tools from other graph servers, but their request and response contracts are CodeContext-specific. MCP calls use a nested `request` object and an optional top-level `repository` path:

```json
{
  "request": {"name_pattern": "save", "limit": 20},
  "repository": "/absolute/path/to/app/nested/submodulelib"
}
```

Call `list_repositories` for available paths. Routing accepts configured repositories, verified client roots when launched globally, and initialized Git submodules beneath those roots. It does not fetch missing submodules or combine call graphs across repositories. Omit `repository` to use the default root. `codecontext://status` describes the default store.

## Workspace snapshots and call accuracy

Run `codecontext --store PATH sync --offline` after editing tracked files. The default `HEAD` sync stores a synthetic commit with the real `HEAD` as its parent, using tracked working file contents. It preserves exact blob citations without modifying the source repository. Repeating a sync with the same content reuses the same synthetic revision. The `WORKSPACE_SNAPSHOT` coverage entry identifies its committed base. Requests using revision `HEAD` read this prepared snapshot; use a branch name or commit hash when syncing committed content only.

Staged additions are included. Untracked files are excluded. Unmerged index entries stop synchronization. Symlinks are not traversed, and working files over 2 MiB are omitted with a coverage gap. Working files must remain stable while being read; this is a per-file snapshot, not an atomic filesystem snapshot. Historical snapshots remain in the owned store and can be included in exports or team shares.

Call edges are lexical candidates, not compiler-resolved proof. Same-file receiver calls use the type/import resolver, so `b.save()` with `b: B` does not acquire an unrelated `A.save()` edge. Kotlin local type declarations override enclosing parameter and constructor types. Unresolved receivers remain unlinked. Dynamic dispatch, inferred local types, extension resolution, and overload selection remain conservative approximations. Sync invalidates older extraction versions automatically.

## Supported Cypher subset

`query_graph` supports one node pattern or one directed edge pattern, optionally with explicit path bounds between 1 and 6 hops. Supported clauses include property filters, numeric and string comparisons, `AND`/`OR`/`NOT`, `COUNT`, `ORDER BY`, and `LIMIT`. Results are bounded by the request limit (at most 200). Examples:

```cypher
MATCH (f:Function) WHERE f.inbound_degree = 0 RETURN f.name, f.file_path
MATCH (a)-[r:CALLS*1..3]->(b) WHERE a.name = 'handle' RETURN b.name, r.depth
MATCH (f:Function) RETURN count(f) AS functions
```

This is not a Neo4j connection or a complete Cypher implementation. Writes, multiple MATCH clauses, arbitrary expressions, and unbounded graph traversal are unsupported. Unrecognized top-level query patterns and invalid path bounds return `INVALID_REQUEST`. `Function` label searches include methods. Regex patterns are validated and reject constructs prone to excessive backtracking.

## Search measurements

`search_graph` filters, counts, and paginates in SQLite. It constructs Python graph nodes only for the returned page and counts only edges whose endpoints belong to the active snapshot. Trace, architecture, and Cypher queries still load the active graph.

A local synthetic Kotlin workload with 200 files and 20,200 symbols produced these median times over five reads:

| Query | Before | After |
| --- | ---: | ---: |
| Exact method-name regex, 200 matches | 248 ms | 22 ms |
| Name regex with no matches | 253 ms | 22 ms |
| Zero-caller search, 20,200 matches | 294 ms | 126 ms |

These measurements cover graph reads, not initial indexing, MCP routing, or a real Android build. Reproduce the workload with:

```bash
uv run pytest -q -s tests/performance/test_graph_search.py
```
