---
name: codecontext
description: Use CodeContext MCP tools (search, context, get, search_graph, trace_path, query_graph, get_architecture, detect_changes, get_code_snippet) when exploring code structure, tracing call chains, running Cypher graph queries, finding dead code, checking architecture clusters, analyzing change impact, explaining why code exists, investigating historical design decisions or constraints, or checking workspace divergence.
---

# CodeContext — Structural Graph + Historical Project Memory

Use the `codecontext` MCP server for both **structural code graph queries** and **cited historical rationale / PR discussion memory**.

## Quick Decision Matrix

| Question | Tool call |
| --- | --- |
| Who calls X / What does X call? | `trace_path(request={"function_name": "X", "direction": "both", "depth": 2})` |
| Find symbols by regex / degree | `search_graph(request={"name_pattern": ".*Pattern.*", "label": "Function"})` |
| Find dead / unused functions | `search_graph(request={"max_degree": 0, "direction": "inbound", "exclude_entry_points": True})` |
| Read symbol source + neighbors | `get_code_snippet(request={"qualified_name": "X", "include_neighbors": True})` |
| Architecture & module cohesion | `get_architecture(request={"max_clusters": 50})` |
| Impact of recent/local changes | `detect_changes(request={"depth": 2})` |
| Run Cypher over call/symbol graph | `query_graph(request={"query": "MATCH (a)-[r:CALLS]->(b) WHERE a.name = 'main' RETURN b.name, b.file_path"})` |
| Locate code, commits, or PRs | `search(request={"query": "...", "kind": "all"})` |
| Why does X exist / what changed? | `context(request={"query": "Why...", "target": {"kind": "path", "path": "src/foo.py", "line": 42}})` |
| Expand full source or citation | `get(request={"id": "ev_..."})` |

## Using Active GitHub / GitLab MCP Servers

- If `context` or `codecontext://status` reports `GITHUB_BACKFILL_PENDING` (local Git history is indexed, remote PRs/Issues have not been synced yet) and a **GitHub MCP** or **GitLab MCP** server is active in the session, you can query that MCP server directly (for example `search_pull_requests`, `pull_request_read`, `issue_read`) for live remote PR/issue details, or run `codecontext --store <repo>/.codecontext sync` (which auto-discovers auth from `gh auth token`, `glab`, or your MCP config without needing manual `GITHUB_TOKEN`/`GITLAB_TOKEN` exports).

## Cypher Examples (for `query_graph`)

```cypher
MATCH (f:Function) WHERE f.name =~ '.*Handler.*' AND f.inbound_degree >= 1 RETURN f.name, f.file_path ORDER BY f.inbound_degree DESC LIMIT 20
MATCH (a)-[r:CALLS*1..3]->(b) WHERE a.name = 'handle_request' RETURN a.name, b.name, r.depth, b.file_path
```

## Historical Rationale Rules

- Check `rationale_status` (`supported`, `enrichment_pending`, `evidence_incomplete`, `undocumented_in_available_evidence`).
- Keep `support`, `lifecycle`, `mapping`, and `applicability` separate. Treat `historical_only` or `inferred` reasons as historical context, not automatically verified current requirements.
- Check `workspace.target_state` and `WORKSPACE_TARGET_DIFFERS` gaps before assuming historical code anchors match uncommitted local edits.
