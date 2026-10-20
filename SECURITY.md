# Security Policy

## Threat model and trust boundaries

CodeContext indexes source repositories, Git histories, and GitHub/GitLab discussions locally for AI coding agents.

1. **Untrusted repository content:** Source code, commit messages, file paths, and issue/PR comments are treated strictly as untrusted data. CodeContext never executes repository code, Git hooks (`core.hooksPath` is set to `/dev/null`), `core.fsmonitor`, external diff tools, or clean/smudge filters (`protocol.allow=never` by default).
2. **Read-only MCP boundary:** The stdio MCP server opens the store in read-only mode (`?mode=ro`) and never runs network requests, parsers, migrations, or model calls during `search`, `get`, `context`, `search_graph`, `trace_path`, `query_graph`, `get_architecture`, `detect_changes`, or `get_code_snippet`.
3. **Credential isolation & principal binding:**
   - Runtime credentials (`GITHUB_TOKEN`, `GITLAB_TOKEN`, or CLI tokens) are never persisted in `store.json`, `memory.sqlite3`, error details, model prompts, or exported archives.
   - HTTPS is strictly required for all remote API and Git transport requests. Cross-host redirects and cross-host `Link` pagination headers are rejected.
   - Private repository evidence is bound to the verified user principal (`store_meta.principal_id`). Attempting to sync or reuse a private store under a different principal fails with `REPOSITORY_MISMATCH`.
4. **Regex & query safety:** User- and agent-supplied regular expressions in `search_graph` and Cypher `query_graph` (`=~`) are bounded and checked against nested quantifiers, backreferences, and lookaround constructs to prevent Regular Expression Denial of Service (ReDoS). All SQLite and FTS5 queries are strictly parameterized.
5. **Export & purge guarantees:**
   - `export` scrubs machine-specific paths and principal bindings using `PRAGMA secure_delete=ON` and `VACUUM`, and validates archive `manifest.json` paths against directory traversal and symlinks on reopen.
   -Note that secrets already committed to the target repository's Git history remain source evidence in full exports (`include_git_objects=True`).
   - `purge` deletes a source and its entire reverse-dependency closure (versions, evidence, claims, summaries, and FTS documents) and increments `access_epoch` so in-flight reads are invalidated before authorization.

## Reporting a vulnerability

Please report security vulnerabilities privately via GitHub Security Advisories on this repository rather than opening a public issue.
