# Source ingestion, syntax, and code lineage

Status: implementation specification, not implemented. Updated 2026-10-16.

Build useful, inspectable memory after the first focused synchronization, then complete the remaining history through resumable work. Preserve the original evidence even when its relationship to current code is uncertain. Read [the shared contracts](00-contracts.md) before this document; those names, limits, public interfaces, and enums are authoritative.

This document specifies internal behavior. Its pseudocode names are implementation responsibilities, not additional public SDK methods. Storage tables and migration ownership belong to the storage specification.

## 1. Work order and completion

A synchronization call performs bounded units of work. Each unit either publishes its records and checkpoint in one database transaction, or leaves its previous checkpoint intact. A second call resumes unfinished work. No model runs during synchronization.

Use this priority order:

1. Validate identity, dependencies, credentials when needed, and the requested revision.
2. Import objects needed for the requested commit. Prepare its file map, commit ancestry membership, and current syntax, prioritizing `focus` if supplied.
3. For a focus target, find at most 100 relevant historical changes, collect up to 25 related episodes, and acquire their original code anchors. These are prioritization limits, not exclusions from eventual backfill.
4. Publish useful collected evidence and pending-enrichment status. The MCP server can already serve this generation.
5. Continue current-snapshot syntax, new Git history, and GitHub enumeration. Alternate a Git work unit and an HTTP work unit when both queues have work.
6. Complete historical syntax and source reconciliation.

A focus never deletes, narrows, or resets the full backfill queue. After every ten focus work units, run one pending ordinary unit to prevent starvation. Within a priority, use persisted insertion order with stable record ID as a tie-breaker. Queue only one job for a given source/object and processing fingerprint.

Persist insertion time in `jobs.created_at_us` and order by `(priority, created_at_us, id)`; updates do not change insertion order. Owner keys include processing fingerprints when the work changes.

`max_requests` counts every attempted HTTP request, including retries and 304 responses. `max_seconds` uses monotonic elapsed time and covers Git, parsing, HTTP, and database work. Do not begin another unit after either budget expires. Use subprocess/HTTP deadlines bounded by the remaining time. A stopped call returns a successful partial `SyncReport`, remaining counts when known, and gaps. A new call must not spend its entire budget repeating completed focus work.

```text
validate(request)
acquire_single_writer_lock()
resolve_and_record_source_ref_snapshot()
resume_or_prepare_active_snapshot(request.revision)
enqueue_missing_jobs(request.focus)
while budget_allows_next_unit:
    job = choose_ready_job_with_fairness()
    result = execute_without_open_database_write_transaction(job)
    transaction:
        validate_result_against_source_identity()
        upsert_immutable_records_and_source_versions(result)
        invalidate_dependent_claims_and_summaries_if_needed()
        checkpoint_job_and_enqueue_discovered_work(result)
        publish_consistent_generation()
return status_with_coverage_and_remaining_work()
```

Preparing a different snapshot publishes its commit, complete file map, and ancestry membership together. Syntax can be partial with explicit coverage. Never combine the new file map with the old revision label. For an unchanged active commit, preserve its usable snapshot while collection progresses.

## 2. Git process boundary

### 2.1 Safe command runner

Use `subprocess` with argument arrays and `shell=False`; consume binary stdout. Never interpolate source text, paths, refs, credentials, or discussion contents into shell commands. Pass only validated full object IDs after revision resolution. Set `GIT_TERMINAL_PROMPT=0`, `GIT_NO_REPLACE_OBJECTS=1`, `GIT_OPTIONAL_LOCKS=0`, and `LC_ALL=C`.

For all Git calls, including source-repository inspection, disable inherited system/global Git configuration, override `core.fsmonitor=false`, and replace inherited `GIT_*` overrides with a known environment. Set an empty hook directory, disable external diff/text conversion, disable automatic maintenance during ingestion, and allow only the transport needed for that call. Do not copy source hooks, templates, aliases, attributes-driven filters, or repository configuration into the owned store. Never request `status`, index refresh, `--filters`, `--textconv`, submodule recursion, or checkout. Source Git config may describe worktree/object layout; its executable hooks and helpers never run.

Resolve a revision with `rev-parse --verify --end-of-options <revision>^{commit}` before further commands. Reject revision strings containing NUL, newline, or a leading hyphen. Resolve `HEAD` against the configured local repository at sync start, then import and use that full object ID. Resolve other refs only within the configured scope; do not treat revision text as a pathspec or arbitrary Git option. Capture the object format with `rev-parse --show-object-format`. [Git revision parsing](https://git-scm.com/docs/git-rev-parse)

Stream stdout and stderr without deadlocking. Keep only a bounded sanitized stderr tail in errors. On timeout or cancellation, terminate the process, reap it, and leave the current unit pending. An interrupted fetch can leave valid Git objects; the next call checks them and continues. Never treat an incomplete subprocess response as a completed record.

### 2.2 Owned object store and ref scope

Initialize a bare `repository.git/` with the source repository's object format and an empty template. Import from the configured local repository using a local file transport into owned namespaces. Do not use shared object alternates or hard-link assumptions: exports and evidence must survive deletion or maintenance of the developer's checkout.

The default scope contains local branches, local remote-tracking branches, tags, and the resolved source `HEAD`; record their original names and observed tips. Exclude stash, replacement refs, reflogs, and unreachable objects from the promised scope. They may explain unavailable history, but are not silently claimed as collected.

Use `refs/codecontext/local/heads/*`, `refs/codecontext/local/remotes/*`, and `refs/codecontext/local/tags/*` for local imports; use `refs/codecontext/github/heads/*`, `refs/codecontext/github/tags/*`, and `refs/codecontext/github/pulls/{number}/head` for GitHub imports. Store the resolved local HEAD under `refs/codecontext/local/HEAD`. Retention refs for evidence use `refs/codecontext/retained/{full_oid}`. These namespaces are implementation details, not user-facing branch names. Never fetch with a blanket refspec that overwrites unrelated owned refs.

For configured GitHub repositories, fetch branches and tags from the validated `https://github.com/{owner}/{repo}.git` endpoint into a separate namespace. Fetch `refs/pull/{number}/head` for enumerated PRs when available. Preserve local and GitHub refs independently so equal branch names cannot overwrite one another. Record deleted or rewritten ref tips as observations and retain already-owned evidence objects; active ancestry membership determines query eligibility.

Authenticated Git fetches use a temporary process environment credential/header configuration scoped to that HTTPS URL, never a token-bearing URL, command argument, log, or saved Git configuration. Disable cross-host redirects. REST uses the same configured principal. Local-only mode performs no network work. `offline=True` imports locally available objects only and reports remote streams as unfinished.

Attempt missing original review commits through the validated primary remote only. If the server will not supply them, preserve available review diff text and report the missing object. Do not crawl fork remotes or arbitrary URLs found in discussions.

Detect shallow history before traversal. Mark the boundary and attempt completion only in the owned mirror when online and the validated remote can provide it. Do not modify or unshallow the developer's checkout. Missing parents remain explicit gaps if recovery fails.

## 3. Git record extraction and sparse indexing

### 3.1 Machine-readable formats

Use the following formats, with fixture tests for their parsers:

| Purpose | Command shape and parser rule |
| --- | --- |
| Reachable commit graph | `rev-list --topo-order --parents --stdin`; feed validated tip IDs, parse each ASCII line as commit then ordered parents. Persist chunks rather than a full in-memory graph. |
| Snapshot files | `ls-tree -r -z --full-tree <commit>`; split NUL records, then the first tab. Metadata precedes the tab; every following byte belongs to the path. |
| Changed paths | `diff-tree --no-commit-id --raw -r -z --no-abbrev --no-renames <parent> <commit>`; parse status metadata and NUL-delimited paths. Process each parent separately. |
| Object bytes | Long-lived `cat-file --batch`; send full hexadecimal IDs and a newline. Read header, exactly its declared byte count, then exactly one framing newline. Never split object content on newlines. |
| Text hunk ranges | For bounded text blobs, `diff --no-ext-diff --no-textconv --no-color --no-renames --diff-algorithm=myers --unified=0 <old_blob> <new_blob> --`; parse hunk headers and counted body lines, never patch display paths as identity. |

The tree format preserves byte paths. Batch object reads are length framed. Raw diff output identifies object changes separately from presentation text. [Git tree listing](https://git-scm.com/docs/git-ls-tree), [Git object batches](https://git-scm.com/docs/git-cat-file), [Git raw differences](https://git-scm.com/docs/git-diff-tree)

Create and retain an empty-tree/empty-blob object using the repository's actual hash format for root commits and missing sides. A root commit compares with the empty tree; a deleted file has no new symbol. Preserve merge parent order and parent-specific hunks; do not use a combined merge diff as the only history record.

Build a byte-offset table from raw line breaks once per text blob. For a hunk side with count greater than zero, map one-based start/count to the bytes covered by those lines. A missing count means one. A zero-count side is an insertion boundary after the stated line, with start zero meaning the beginning; it is not a negative or one-line range. Preserve both native hunk coordinates and converted byte ranges. Assign a nonempty changed range to every overlapping definition, then mark the smallest enclosing symbol as the primary target. For insertion boundaries, associate only an enclosing body whose byte interval contains the boundary; a boundary between definitions remains file-scoped. Ignore the patch's no-final-newline marker when counting body lines, and reject inconsistent hunk counts rather than guessing.

Parse commit objects from their raw bytes. Preserve message bytes and the declared encoding; decode a separate search display with an explicit decoding status. Folded headers such as signatures do not terminate the header section; the first empty line does. Preserve author and committer times separately. A timestamp does not establish ancestry or causation.

Read object type and size before allocation. Source-size limits apply before parsing or diffing. NUL-containing blobs are binary. Non-UTF-8 source retains raw bytes and metadata, with a decoding/structure gap; v1 does not guess character encodings for syntax. Symlinks retain their link text as evidence without traversal. Gitlinks retain the submodule commit reference without fetching submodule contents. LFS pointers remain pointers; report that the asset content was not acquired.

### 3.2 Sparse historical representation

Store each commit once, each parent edge once, each changed-path/parent relation once, each hunk once, and each parsed blob/fingerprint once. The prepared snapshot has one complete path-to-blob map. Historical file lookup reads the relevant owned Git tree; it does not require materializing every historical tree in SQLite.

Index changed paths and old/new blob IDs for history discovery. Associate changed ranges with symbols extracted from the two changed blobs. Unchanged symbols receive no new occurrence row merely because another file changed. Historical references carry commit/path/blob anchors only where a source, change, or query result needs them.

For focus scheduling before full history is indexed, stream path-limited history for the current path and known predecessor paths; include all parents and preserve merge edges. Label it provisional because a rename may be undiscovered. The ordinary whole-history pass fills those gaps. Never declare complete lineage from `git log --follow` alone.

Attempt rename candidate detection separately from factual add/delete records. A unique deleted and added path with the same blob establishes exact byte correspondence. For edited renames, run a bounded Git similarity pass at 50% threshold with a rename candidate limit of 200; store its method and score as `candidate`. If the limit is exceeded, keep add/delete facts and a mapping gap. Do not search every repository file for copies in v1. A copied function creates a separate anchor even when its bytes match.

Store immutable commit lists or per-commit job states as traversal checkpoints, not line offsets into a changing `rev-list` stream. New or rewritten refs enqueue newly reachable commits. Historical Git coverage is complete only after each discovered commit's parents, changed-path records, and required bounded hunks are accounted for; structure coverage remains independent.

## 4. Syntax and documentation extraction

Parse only unique supported blobs below the 2 MiB limit. Choose language by the occurrence's extension: `.py`, `.js`, `.mjs`, `.cjs`, `.jsx`, `.ts`, `.mts`, `.cts`, and `.tsx`. JavaScript grammar handles JSX; TypeScript and TSX use their corresponding grammar entrypoints. Preserve separate cache keys if identical bytes appear under different language selections.

Store a versioned query pack for each language. Include package versions and query-pack hash in `extractor_fingerprint`. During dependency setup, compile every query and run the golden syntax fixtures; a grammar mismatch fails setup rather than silently yielding no symbols. Use the binding's documented `Language`, `Parser`, and query interfaces at the pinned version. [Tree-sitter Python bindings](https://github.com/tree-sitter/py-tree-sitter)

Extract these minimum structures:

| Language | Named definitions and relationships |
| --- | --- |
| Python | Classes, functions, async functions, methods, nested definitions, module assignments, imports, relative imports, call occurrences, and docstrings. Include decorators in the definition's display range while preserving its syntactic body range. |
| JavaScript/JSX | Classes, function declarations, methods, named exports, import declarations, variable bindings, arrow/function expressions assigned to bindings, call occurrences, and JSX component references. |
| TypeScript/TSX | JavaScript structures plus interfaces, type aliases, enums, namespaces, type-only imports/exports, and signature declarations. |

A qualified name contains only lexical named containers. Module paths belong to occurrence anchors, not blob-owned symbol identity. Anonymous expressions remain syntax occurrences unless a binding provides their name. Overloads and repeated names have separate ranges and IDs; never deduplicate by name alone.

Collect exact signatures, definition/body ranges, lexical parent, and reference text. Do not execute imports or invoke a project compiler. Resolve lexical same-scope references only when a single syntactically established binding exists; otherwise retain candidates. Calls through objects, dynamic imports, monkey-patching, and runtime injection remain unresolved. Extraction facts and resolution hypotheses have different relationship kinds.

On syntax errors, preserve the parse error ranges. Emit only definitions whose identifying name and declaration header do not intersect an error/missing node. Mark the blob partially parsed, retain text search, and avoid treating an empty extraction as proof that the file defines nothing.

Index UTF-8 documentation and configuration as bounded text chunks with exact raw byte anchors: Markdown, reStructuredText, plain text, TOML, YAML, JSON, XML, INI, manifests, Dockerfiles, shell files, and extensionless UTF-8 text. These are evidence sources; they are never executed. Chunking uses the shared 8 KiB/512-byte defaults and UTF-8-safe boundaries. Split text and code search projections from preserved raw bytes.

Every tracked file receives metadata. Binary, oversized, unsupported-syntax, decoding-failed, and generated-looking files receive their applicable capability/gap status. V1 applies no silent vendor-directory exclusion. Generated-file heuristics can lower ranking but cannot erase declared collection scope.

## 5. GitHub collection

### 5.1 HTTP contract and endpoint matrix

Use `https://api.github.com`, explicit `Accept: application/vnd.github+json`, a recognizable User-Agent, and `X-GitHub-Api-Version: 2026-03-10`. Freeze this version in fixtures and update it only with a documented compatibility change. [GitHub API versions](https://docs.github.com/en/rest/about-the-rest-api/api-versions)

All paths below are GET requests beneath `/repos/{owner}/{repo}` unless shown otherwise. Use `per_page=100` on paginated endpoints that support it. The full endpoint string, parameters, principal identity, and API version identify a checkpoint.

V1 supports unauthenticated public reads and `GITHUB_TOKEN` user tokens whose identity is verified through `GET /user` before authenticated source persistence/fetch. Store its immutable user ID, not the token. Reject unsupported installation-token authentication with actionable `AUTH_REQUIRED` rather than guessing a principal. Installation-token support is deferred. Verify the primary repository ID through its root endpoint before importing remote evidence; these validation requests count toward the run budget. A change of authenticated principal requires a separate store.

| Stream | Endpoint | Purpose and completion rule |
| --- | --- | --- |
| Repository | Repository root endpoint | Verify immutable repository ID, name, access, and default branch at each online sync. |
| Issues | `/issues?state=all&sort=created&direction=asc` | Enumerate issues including PR issue records. A `pull_request` field identifies PR-shaped records. |
| PR details | `/pulls/{number}` | Collect current base/head, merge fields, state, body, and timestamps. |
| Conversation | `/issues/{number}/comments` | Collect issue and PR conversation bodies and authors. |
| Reviews | `/pulls/{number}/reviews` | Collect review bodies, states, reviewer, submission time, and reviewed commit. |
| Inline review comments | `/pulls/{number}/comments` | Collect original/current code anchors and reply references. |
| Timeline | `/issues/{number}/timeline` | Preserve events; index referenced, cross-referenced, connected, disconnected, closed, reopened, and review-dismissal relationships when supplied. |
| PR commits | `/pulls/{number}/commits` | Provider association evidence; reconcile with local Git reachability. |
| PR files | `/pulls/{number}/files` | Secondary mapping metadata; Git objects remain the patch authority. |
| Focus association | `/commits/{sha}/pulls` | Prioritize provider-associated PRs for relevant commits; preserve all returned associations. |
| Incremental conversations | `/issues/comments?sort=updated&direction=asc&since=...` | Discover edited or new conversation comments independently of parent timestamps. |
| Incremental inline comments | `/pulls/comments?sort=updated&direction=asc&since=...` | Discover edited or new inline comments independently of PR timestamps. |

Issues and PRs share issue numbers and conversation endpoints. Normalize each PR to one `pull_request` source. The issue listing discovers PR identities and schedules their detail fetch; `/pulls/{number}` owns the normalized PR title/body/state. Do not create an additional issue source or duplicate searchable body for the same PR. Conversation comments still use the issues endpoint. [GitHub issues](https://docs.github.com/en/rest/issues/issues), [issue comments](https://docs.github.com/en/rest/issues/comments)

PR commits are capped at 250; PR files at 3,000. Treat either cap or a reported total mismatch as an API coverage gap and recover through available Git objects. Never mark a capped response as full history. [GitHub pull requests](https://docs.github.com/en/rest/pulls/pulls)

Commit-to-PR results establish provider associations, including multiple possible PRs. They do not prove which comment explains a change. [GitHub commit associations](https://docs.github.com/en/rest/commits/commits)

### 5.2 Pagination, source versions, and retries

Follow the response's `Link` relation `next` until absent. Validate that next links remain on `api.github.com` and within the intended endpoint before forwarding authorization. Persist the exact next request only after storing the current page. Deduplicate by immutable provider IDs, including across page boundaries and resumed calls. [GitHub pagination](https://docs.github.com/en/rest/using-the-rest-api/using-pagination-in-the-rest-api)

Store exact decoded body bytes and the original response record separately. Normalize only the fields declared in the shared source-version identity. A changed body, anchor, state, author identity, or source timestamp creates a new version. A new fetch time or ETag alone does not. Delete author identities remain null/attributed as deleted; never invent their former names.

Retain ETag/Last-Modified per request and principal. A 304 validates only that response; it does not prove that every other page or child collection is unchanged. Refresh observation timestamps without creating a version. Preserve all observed versions and never manufacture pre-edit history.

Use connect timeout 10 seconds and read timeout 30 seconds, each capped by the sync deadline. Stream and stop before exceeding the shared 32 MiB response limit. A body over 4 MiB is metadata-only; do not archive its unbounded raw payload elsewhere to bypass the limit.

One HTTP request is in flight per store/principal. On 429 or rate-related 403, respect `Retry-After`, then exhausted-primary reset time; wait at least 60 seconds for a secondary limit when neither is available. Save `retry_at` in the job cursor and return partial work if waiting would exceed the budget. For transport errors and 502/503/504, allow three total attempts with one- and two-second delays before retries, plus jitter in [0,1] second. Record attempts; do not loop on 401, ordinary 403/404, validation errors, or unsupported API versions. [GitHub rate limits](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api), [API practices](https://docs.github.com/en/rest/using-the-rest-api/best-practices-for-using-the-rest-api)

A repository 404 may mean deletion or loss of permission. Mark the scope unavailable and require identity/access recovery rather than asserting deletion. A child 404 with successful repository access becomes a missing/unavailable source observation. The access-control contract determines whether previously stored data is blocked.

### 5.3 Incremental updates and reconciliation

Record an online sync start timestamp. Incremental streams reread from the last completed watermark minus five minutes, with ID/version deduplication. Advance the watermark only after the complete bounded sweep succeeds; cap it at sync start. Refresh issues using `since`, and PR discovery using `sort=updated&direction=desc` until records precede the overlap window. A changed parent schedules all its child streams.

Parent timestamps are insufficient to guarantee every old review edit is noticed. Each ordinary online sync therefore resumes a persistent round-robin child-refresh queue. `reconcile=True` schedules a full primary enumeration and all child streams from their first page, including old closed PRs. It remains budgeted and resumable.

Use a reconciliation pass ID and persist every seen source in `reconciliation_seen(stream_key, pass_id, source_id)`, atomically with each page checkpoint. Keep active pass metadata in `checkpoints.cursor_json`. Only after a complete successful collection sweep may missing previously seen records become unavailable observations. Never infer deletion from an interrupted pass, a filtered incremental response, an API cap, or a permission failure. Remove obsolete seen membership after committing completion. Preserve prior evidence versions with availability and freshness information; dependency invalidation follows the storage contract.

Full listing is not an atomic remote snapshot. After backfill finishes, run an overlapping update sweep and label coverage with the observation interval and reconciliation time. Do not claim all edits occurring during the crawl were observed simultaneously.

Parse explicit GitHub issue references in source text and timeline relationships. Same-repository references are in scope. External issue fetches require an entry in `allowed_issue_repositories`; fetch only the issue and its conversation/timeline, one hop. Do not follow links from that external issue onward, fetch arbitrary web pages, or recursively index the other repository's code.

## 6. Original review anchors and landed code

Keep three distinct anchors: the originally reviewed code, the PR's observed final head, and the prepared target revision. A review's current displayed line is not a replacement for its original anchor.

Preserve `original_commit_id`, `commit_id`, `path`, `side`, `start_side`, original/current line fields, legacy positions, `diff_hunk`, review ID, and `in_reply_to_id`. Reply inheritance links to the parent anchor while retaining the child's identity and body. File-level comments remain file-scoped. Reviews without inline anchors remain episode-scoped. [GitHub review comments](https://docs.github.com/en/rest/pulls/comments), [GitHub reviews](https://docs.github.com/en/rest/pulls/reviews)

For a RIGHT-side comment, resolve the original commit's path and validate its original line range against owned blob bytes and the preserved diff excerpt. For LEFT-side comments, resolve the old side from a known original comparison; do not assume the first parent is the PR base. If the original comparison is unavailable, preserve the diff-hunk anchor and mark code mapping `unmapped`. Legacy patch positions require parsing the original patch; without it, retain the position as evidence only.

Apply this ordered mapping procedure:

1. **Exact original anchor:** original commit, path, blob, and range are available and validated. This establishes correspondence to original code only.
2. **Exact target correspondence:** the target contains the same blob/path/range, or a uniquely verified unchanged range transported across parent-specific hunks and a unique exact-byte move. Set `mapping="exact"` and store every transformation used.
3. **Candidate correspondence:** edited rename, unique lexical symbol/signature association, or bounded patch similarity suggests a match. Set `mapping="candidate"`, preserve alternatives and method, and never upgrade through a chain containing a candidate edge.
4. **No correspondence:** missing source, changed overlapping region, duplicate match, deleted symbol, or ambiguous merge path yields `unmapped` or multiple candidate records with a gap.

For unchanged-range transport, only offset a range when all preceding edits have known line deltas and no edit overlaps the range. Across a merge, require agreement of the applicable parent paths; otherwise keep candidates. Exact byte correspondence still does not establish author intent or current applicability.

Associate the PR with landed code using provider commit associations, recorded merge identity, and owned Git ancestry. Ordinary merge, squash, and rebase have different commit shapes. A squash/rebase patch match is a candidate association even when the patch appears identical. An unmerged PR has no landed anchor. A reverted implementation remains retrievable historical evidence; a revert relationship does not automatically reject every rationale stated in its original PR.

The source-to-code mapping never sets claim lifecycle or applicability. Those are independent enrichment/retrieval decisions under the shared contracts. In particular, “merged,” “bytes still match,” and “no contradiction found” never turn an old rationale into a corroborated current constraint.

## 7. Workspace differences

When trusted local configuration supplies a workspace, compare its `git-common-dir` realpath and filesystem device/inode against the initialization binding before reading target files. Linked worktrees of the original repository pass; unrelated clones fail even if they share a remote URL. Exports have no workspace binding. Resolve workspace HEAD independently of the prepared memory revision. `index_blob_oid` means the prepared memory snapshot blob; `staged_blob_oid` means the Git staging-area blob.

Use `ls-tree` for workspace HEAD and `ls-files --stage -z` for its index, with literal target pathspecs and `core.fsmonitor=false`. Parse mode/object/stage fields and split at the first tab, preserving following path bytes. Compare HEAD entries with stage-zero entries to determine staged changes; compare staged raw blob bytes and mode with root-confined file bytes/mode to determine working changes. An absent HEAD/index entry plus a regular file is untracked. Reject unmerged stages as `unavailable` with a conflict gap. Do not use `git status`, `diff-files`, index refresh, or clean/smudge filters. [Git index listing](https://git-scm.com/docs/git-ls-files)

`target_state` describes normal staged/worktree differences against workspace HEAD. A clean checkout on another commit can therefore report `unchanged` while `head_matches_index=false`. Independently compare the selected target with the prepared memory bytes and add `WORKSPACE_TARGET_DIFFERS` whenever they differ. For mixed states, use `staged_and_unstaged`; if the tracked target is absent from disk, use `deleted` and preserve available stage metadata.

Read regular files with root-confined, no-follow path handling. Never follow a symlink outside the workspace. Bound reads to 2 MiB and compute SHA-256 over raw bytes. Compare working bytes against raw staged bytes without invoking Git clean filters. Line-ending transformations may produce a conservative difference warning; do not execute a filter to suppress it. A file modified while being read yields `unavailable` after one retry. Missing configuration yields `not_checked`.

An untracked target returns `UNTRACKED_TARGET` and bounded search suggestions. Dirty/unsaved content receives no inherited confirmed applicability and no transient parsed overlay in v1. All returned code ranges continue to refer to the prepared commit.

## 8. Required fixtures and expected outcomes

Build fixtures through temporary Git repositories and recorded HTTP responses. Never make correctness tests depend on live GitHub. Keep fixture builders deterministic; use fixed authors/timestamps, explicit branch names, and assertions against raw objects.

| Fixture | Required assertion |
| --- | --- |
| Root commit, ordinary edit, two-parent merge | Root additions use empty-tree semantics; each merge parent has distinct changes; ancestry is preserved. |
| 10,000 commits changing one tiny file each | Symbol rows grow with distinct parsed blobs, not current symbols multiplied by commits; unchanged blobs parse once per fingerprint. |
| Space/tab/newline/non-UTF-8 paths | Exact path bytes round-trip; display paths never replace identity. |
| SHA-256 Git repository | Full object IDs survive import, hashing, records, and exports without truncation. |
| Binary, symlink, gitlink, LFS pointer, oversized blob | Metadata remains available; no traversal/download/execution; gaps distinguish each limitation. |
| Renamed unchanged file versus two identical copies | Unique move yields exact byte correspondence; copies remain separate anchors with ambiguous lineage when necessary. |
| Reviewed code edited before merge | Original quote opens original blob; target mapping becomes candidate/unmapped where the range changed. |
| LEFT-side review with missing comparison | Diff excerpt is available; original base is not invented; mapping remains unmapped. |
| Squash, rebase, cherry-pick, revert | Provider associations persist; patch matches stay candidate; branch ancestry gates target eligibility. |
| Force-push removing review commit | Previously acquired evidence remains; newly missing objects produce gaps; active branch membership changes. |
| Page overlap and interrupted page transaction | Duplicate provider IDs produce one source; resume starts at last committed checkpoint. |
| Comment edited with unchanged parent timestamp | Incremental comment stream or round-robin reconciliation detects it and invalidates dependent derivations. |
| Deleted comment after incomplete reconciliation | No deletion inference until a successful complete sweep; old version remains attributed and availability marked. |
| PR with 251 commits or 3,001 files | REST cap is visible; Git recovery supplies actual changes when objects exist. |
| 304, 429, primary exhaustion, 503, 401 | Correct cache/retry behavior; request budgets include retries; no credential or response-body leak. |
| Clean other branch, staged edit, unstaged edit, deletion, untracked file | HEAD mismatch and dirty state are independent; all code anchors remain pinned to prepared revision. |
| Configured executable fsmonitor, clean/smudge filters, and Git hooks | Marker executables never run during initialization, synchronization, workspace inspection, reads, or export. |
| Parser error, overloads, JSX, TSX, duplicate names | Partial syntax is explicit; distinct range IDs remain; query packs compile against pinned grammars. |
| Stop after each publication boundary | Reopening yields the last consistent snapshot; next sync finishes without duplicate content versions. |

Run a real small GitHub repository pilot after deterministic tests pass. Record time to first focus context, completed coverage, API calls, peak memory, store size, missing sources, and incremental no-change cost. Live pilot numbers are observations, not replacements for repeatable fixtures.

Implementation and live performance checks remain pending. Main limit: unavailable historical objects and unobserved prior discussion edits cannot be reconstructed reliably; preserve the gap instead of inventing continuity.
