# Shared implementation contracts

Status: design specification, not implemented. Updated 2026-10-16.

This document is the source of truth for terminology, identifiers, defaults, public types, and cross-module behavior. Companion documents supply algorithms and tests. If examples conflict with this document, correct the examples before implementing. Do not silently invent a third interpretation.

## 1. Product boundary

Build `codecontext`, a local Python library with a CLI and a read-only MCP server. One store owns one primary Git repository and its GitHub evidence. Explicitly allowed linked GitHub issues may belong to other repositories; those repositories' code is not recursively indexed. One MCP server process opens one store. Multiple projects use multiple stores/server instances.

Release scope: all accessible configured Git history; all primary-repository GitHub issues, PRs, comments, reviews, review comments, and relevant timeline relationships; Python, JavaScript, TypeScript, JSX, and TSX syntax extraction; text/history fallback for other files; optional model enrichment; cited retrieval; export and purge. Initial supported operating systems are Linux and macOS, with Python 3.12 and 3.13. Windows support is deferred until its Git/path/process tests pass.

The first demonstrable milestone is a real MCP client retrieving a function's historical explanation, inspecting original evidence, and handling unknown intent. It precedes exhaustive backfill, global summaries, and export. Release scope still requires those later milestones.

No implementation feature is considered complete solely because this document describes it. The task ledger and test results record implementation progress.

## 2. Fixed engineering choices

- Python `src/` package layout, `argparse` CLI, `sqlite3` storage, standard-library subprocess Git plumbing, `httpx` HTTP, Pydantic v2 public/model-output validation, Tree-sitter parsing, official `mcp` Python SDK. No ORM, separate graph/vector database, agent framework, or background daemon.
- Use `uv` to create the environment and lock dependencies; build with Hatchling. Development tools: pytest, pytest-asyncio, and Ruff. Type annotations are required on public functions; a separate static type checker is not a release requirement.
- Starting direct pins: `mcp==2.3.0`, `tree-sitter==0.26.0`, `tree-sitter-python==0.25.0`, `tree-sitter-javascript==0.25.0`, `tree-sitter-typescript==0.23.2`. These versions were visible in primary package listings during planning. They still require installation and ABI smoke tests in task T01.
- Other direct ranges: `pydantic>=2.10,<3`, `httpx>=0.28,<1`; dev `pytest>=8,<10`, `pytest-asyncio>=1,<2`, `ruff>=0.12,<1`; build `hatchling>=1.26,<2`. Resolve once, commit `uv.lock`, and use `uv sync --frozen` thereafter. If the chosen pins cannot install or interoperate, record the actual failure and change only the affected pin with a smoke test; do not substitute an entire stack.
- Use Git >=2.43. Verify FTS5 and JSON functions at startup. Missing capabilities fail with an actionable dependency error before opening a writable store.
- Public SDK operations are synchronous. Blocking Git/SQLite work in async MCP handlers runs through `asyncio.to_thread`. One request uses one SQLite connection owned by that worker; never share a connection between threads.
- Model enrichment uses an injected Python adapter. No provider credentials, default paid model, or model-specific HTTP integration is shipped in core v1. A CLI adapter is loaded only from an explicit user-supplied `module:factory`; repository files never select executable adapters.
- Runtime storage is outside the source checkout. Require `--store PATH` / `Memory.open(path)` rather than inventing an automatic location. An omitted CLI path is a usage error. Tests use temporary directories.

## 3. Repository and snapshot semantics

A store contains `memory.sqlite3`, `repository.git/` (an owned bare Git object store), `store.json` (non-secret connection/configuration values), and `tmp/`. The database is authoritative; `store.json` contains only initialization configuration and its format version. Tokens are environment inputs, never serialized. `manifest.json` and `OVERVIEW.md` are export products, not independent runtime authorities.

`store.json` has exactly `format_version: 1`, `repository_id: UUID string`, `read_only_archive: bool`, `repo_path: absolute string | null`, `github_repository: OWNER/REPO | null`, and `allowed_issue_repositories: sorted unique string[]`. It is also the ownership marker. Validate its repository UUID against SQLite before deletion or export; never treat a filename alone as proof of ownership. Filesystem common-directory binding remains in SQLite. An archive sets `read_only_archive=true` and `repo_path=null`; every open honors this flag even when `read_only=False` was requested. Do not save tokens, adapters, executable commands, or model options in this file.

Each store has a generated UUID `repository_id`. Initialization records the real path, filesystem device, and inode of the original Git common directory. Linked worktrees sharing that directory are accepted; a separate clone requires a separate store even if its remote matches. Later attempts to bind the store to another repository fail. For GitHub, preserve the immutable repository numeric ID as well as the current owner/name. A rename does not create a new project. Local-only repository identity remains the store UUID; remote names alone never establish identity. Exports remove machine bindings and cannot inspect a workspace until a future explicit rebinding feature is designed.

Maintain exactly one prepared code snapshot per store in v1. It consists of the active commit SHA, its file-to-blob map, and reachable-commit membership. All collected historical Git objects and source records remain available. `sync(revision=...)` prepares another snapshot, atomically replacing the active code map; `offline=True` forbids network activity. Agents can inspect historical evidence without preparing a whole older snapshot. Full code-context queries for a different snapshot require explicit preparation and return `REVISION_NOT_PREPARED` otherwise. Multiple concurrently queried branches use separate stores in v1.

`revision=None` means the active prepared commit, not the live workspace's moving `HEAD`. Explicit refs resolve from recorded refs and must equal the active prepared commit for `search` and `context`. Return the full resolved SHA in every response. `get` may read an immutable historical code/evidence handle already represented in the store, regardless of the active revision.

Changing the active snapshot must not rebuild historical evidence or duplicate unchanged symbol records. Do not create a `(commit, path, symbol)` row for every symbol in every commit.

## 4. Identity, time, paths, and source ranges

- Public record IDs are opaque strings prefixed by type (`sym_`, `chg_`, `hunk_`, `src_`, `ver_`, `ev_`, `claim_`, `sum_`, `gap_`). The suffix is lowercase SHA-256 of canonical UTF-8 JSON identity fields. Canonical JSON uses sorted keys, compact separators, `ensure_ascii=True`, and no floating-point identity fields. Store original input fields so IDs can be audited. Source and gap IDs select mutable status; version and evidence IDs select immutable content subject to access and purge checks.
- Git object IDs retain their actual object format and full width. Do not assume every repository uses 40-character SHA-1. Store `object_format` and reject an incompatible mirror import.
- Source identity uses GitHub repository ID, record kind, and provider numeric/node ID. A source version hashes the normalized relevant payload, including body, title, author identity, anchors, source timestamps, and lifecycle fields. Fetch timestamps do not create a new content version.
- A parsed symbol is owned by `(blob_oid, language, extractor_fingerprint, byte_start, byte_end, kind, qualified_name)`. The fingerprint includes library/grammar versions and query-pack hash. File paths and commits are separate occurrence anchors.
- Store Git paths as bytes internally; use NUL-delimited Git output. Public `path` is UTF-8 when valid; otherwise use a display string with `path_bytes_b64` as authoritative. Never normalize case or Unicode for identity. User-supplied target paths must be relative to the configured repository; absolute paths, `..` traversal, and NUL are rejected.
- Internal source ranges use zero-based, half-open byte offsets. Public human line numbers are one-based; `line_end` is inclusive. Empty insertion/deletion hunks use Git's native count semantics and are handled explicitly, not converted into negative ranges.
- Discussion bodies are indexed as exact UTF-8 bytes of the decoded provider string. Keep the original JSON payload separately. Preserve line endings in evidence; normalize only in a separate search-text field.
- All times are UTC ISO 8601 with `Z`; store sortable integer microseconds internally. Preserve `authored_at`, `updated_at`, `first_observed_at`, and `last_observed_at` separately. Missing provider timestamps are null, never inferred from crawl time.

## 5. Evidence and claim semantics

Every evidence reference resolves to a stored version or an owned Git object plus immutable anchor. A URL alone is insufficient. Exact byte-span presence is mechanically validated; semantic support is evaluated separately. Statements from maintainers remain attributed statements, not automatically verified technical facts.

Use these independent enums without merging their meanings:

| Field | Values | Assignment rule |
| --- | --- | --- |
| `support` | `explicit`, `inferred`, `unknown` | Whether a cited source states the rationale, whether it is an interpretation, or whether no rationale is established. |
| `lifecycle` | `proposed`, `adopted`, `rejected`, `disputed`, `superseded`, `unknown` | Requires evidence for the specific claim; a merged PR alone does not establish adoption of every comment. |
| `mapping` | `exact`, `candidate`, `unmapped` | Describes correspondence from original source/code anchors to the target code. Exact means mechanically established correspondence, not causal proof. |
| `applicability` | `corroborated`, `inferred`, `historical_only`, `unknown` | Describes whether the rationale is supported for the requested target revision. This is computed per response, never globally copied from a claim. |

Default old rationale to `historical_only`. Unchanged function bytes, ancestry, a clean workspace, matching names, and absence of contradictory evidence do not upgrade applicability. `corroborated` requires an explicit corroboration evidence record that identifies the constraint and target scope/revision; it is produced only through validated extraction of an explicit later/current statement or a user-reviewed correction recorded as a source document/ADR. Code/tests can corroborate observable behavior, but cannot independently establish author intent. Model interpretations use `support="inferred"`. The separate `applicability="inferred"` value is reserved and never emitted in v1; there is no runtime reasoning step to assign it. Preserve conflicting sources even when one looks more authoritative.

An explanation with no supported rationale returns `rationale_status="undocumented_in_available_evidence"` only when every required segment of the relevant collected episode has been inspected. If it has not been inspected, use `rationale_status="enrichment_pending"`; if source collection has gaps, use `rationale_status="evidence_incomplete"`. `rationale_status="supported"` requires at least one eligible claim with `support="explicit"`, non-null rationale, and exact correspondence to the requested target. An untargeted query may establish an explicitly scoped historical episode rationale; it must not imply applicability to unspecified current code. Inferred claims can be returned but cannot establish documented intent. Keep gaps visible even with a supported partial answer.

Default knowledge semantics are current collected evidence about the selected code revision. Optional `knowledge_cutoff` excludes source versions authored/edited after the cutoff, and summaries/claims derived from those versions. A current edited body is not retroactively valid at its creation time. Never reconstruct unavailable prior comment bodies. Commit eligibility also requires ancestry; timestamps never replace the graph.

With a cutoff, exclude discussion versions whose relevant source timestamps are unknown and report `SOURCE_TIME_UNKNOWN`; crawl time does not substitute. An earlier version actually captured before an edit may remain usable when its source times satisfy the cutoff. Source edits mark dependent claims/summaries stale. `context` excludes stale derivations even for historical-cutoff queries, returning eligible raw evidence and `DERIVED_ENRICHMENT_PENDING`. `get` can expose a stale claim with its stale status and evidence, but must not present it as a current validated conclusion. Purge is different from an edit: it removes all dependent generated content transitively; previously known purged content IDs return `RECORD_NOT_FOUND`.

## 6. SDK interface

Export `Memory`, request/result Pydantic models, `ModelAdapter`, and `CodeContextError` from `codecontext.__init__`. Keep internal stores/parsers private.

```python
Memory.open(store_path: Path, *, read_only: bool = False) -> Memory
Memory.initialize(store_path: Path, request: InitRequest) -> Memory  # class method
Memory.sync(request: SyncRequest) -> SyncReport
Memory.enrich(request: EnrichRequest, *, model: ModelAdapter) -> EnrichReport
Memory.search(request: SearchRequest) -> SearchResult
Memory.get(request: GetRequest) -> RecordPage
Memory.context(request: ContextRequest) -> ContextPack
Memory.status() -> Status
Memory.export(request: ExportRequest) -> ExportReport
Memory.purge(request: PurgeRequest) -> PurgeReport
Memory.close() -> None
```

`Memory` supports a context manager. The `initialize` class method creates the local store and returns an open instance; it performs local identity validation but no GitHub requests or model calls. Repeating initialization with the same binding and normalized configuration opens the existing compatible store; a different binding/configuration returns `REPOSITORY_MISMATCH`. Normalize repository names and sorted unique allowlists before comparison; do not silently rewrite configuration. `open` does not initialize, migrate, sync, fetch, parse, or call a model. Missing stores fail with `STORE_NOT_INITIALIZED`; initialization is explicit. Until the first snapshot is prepared, `status.revision` is null and code-context reads fail with `REVISION_NOT_PREPARED`. Mutations on a read-only instance fail. Use strict Pydantic models with `extra="forbid"`, bounded strings/arrays, and explicit defaults. Validate before subprocess/network/model calls.

Request fields:

| Request | Fields and defaults |
| --- | --- |
| `InitRequest` | `repo_path: Path`, `github_repository: str | None=None`, `allowed_issue_repositories: list[str]=[]`; GitHub host is `github.com` in v1. Repository existence and identity are checked. |
| `SyncRequest` | `revision: str="HEAD"`, `offline: bool=False`, `reconcile: bool=False`, `focus: Target | None=None`, `max_requests: int=1000`, `max_seconds: int=900`. These budgets checkpoint work; repeated calls eventually finish the accessible scope. |
| `EnrichRequest` | `target_ids: list[str] | None=None`, `max_input_tokens: int`, `max_output_tokens: int`, `max_calls: int`, `max_cost_usd: Decimal | None=None`. Token/call limits are required positive values. A monetary limit requires configured input/output prices. |
| `SearchRequest` | `query: str`, `revision: str | None=None`, `kind: "all"|"code"|"history"|"decision"|"summary"="all"`, `limit: int=20` (1..100), `knowledge_cutoff: datetime | None=None`, `cursor: str | None=None`. |
| `GetRequest` | `id: str`, `max_tokens: int=6000` (512..32000), `cursor: str | None=None`, `knowledge_cutoff: datetime | None=None`. |
| `ContextRequest` | `query: str`, `target: Target | None=None`, `revision: str | None=None`, `max_tokens: int=6000` (512..32000), `knowledge_cutoff: datetime | None=None`. Workspace is configured when opening the server/SDK, not supplied as an arbitrary query path. |
| `ExportRequest` | `destination: Path` (must not exist), `include_git_objects: bool=True`. A metadata-only export explicitly cannot guarantee offline code expansion. |
| `PurgeRequest` | Exactly one of `source_id: str` or `entire_store: bool=True`. Individual Git blob/history erasure is not supported; it requires entire-store purge. |

Enrichment token limits apply to the whole run. Reserve input and maximum output before each call; cap one call at 24,000 estimated input tokens and 4,000 output tokens. Do not retry a model transport failure automatically because its charge state can be unknown. See the enrichment chapter for reservation reconciliation.

Additional request bounds are fixed: refs 1..1,024 characters; `max_requests` 1..100,000; `max_seconds` 1..86,400; each enrichment token total 1..1,000,000,000; `max_calls` 1..1,000,000; `target_ids` at most 1,000 distinct IDs. Optional monetary limits must be positive finite Decimals, at most 1,000,000 USD. Accept at most 64 unique issue-repository allowlist entries, each a validated `OWNER/REPO` without URL syntax, traversal, whitespace, or credentials. Paths follow the byte/path bounds in public models; cursors are at most 8,192 characters. Reject naive cutoff datetimes. Trusted store/repository/workspace paths are normalized absolute local paths before use; a runtime store must not be inside the indexed checkout.

`Target` is a discriminated union: `{kind:"symbol", id:string}` or `{kind:"path", path?:string, path_bytes_b64?:string, line?:positive int}`. A path target requires exactly one of UTF-8 `path` and canonical base64 `path_bytes_b64`. Decode once and reject absolute paths, NUL, and `..` components at the byte boundary; do not use the display path as identity. A path without a line addresses its file. A line selects the smallest containing named symbol; multiple equal candidates return `AMBIGUOUS_TARGET` with bounded candidates. Query text is 1..4000 characters after whitespace trimming. No implicit path/name guessing when a target is explicit.

All read results include `schema_version=1`, `repository_id`, `revision`, `generation`, `access_epoch`, `coverage`, and `gaps`. `SearchResult` adds ranked `hits`, `next_cursor`, and `truncated`. `RecordPage` adds `record`, `evidence`, `relations`, `next_cursor`, and `budget`.

`ContextPack` additionally contains `query`, `target`, `workspace`, `overview`, `code`, `decisions`, `evidence`, `rationale_status`, `budget`, and `expand`. `overview` is a short stored summary or a deterministic structural description. Read operations never synthesize a new natural-language answer with a model. The consuming agent writes its own answer from this pack.

Each decision result includes `id`, `statement`, `rationale`, `alternatives[]`, `constraints[]`, `support`, `lifecycle`, target-specific `mapping` and `applicability`, `evidence_refs[]`, `counterevidence_refs[]`, and `applicability_evidence_refs[]`. Unknown text fields are null; missing alternatives are an empty array, never invented content. Supporting snippets include attribution and exact anchors.

Use `CodeContextError(code, message, retryable, details)` for expected failures. Codes: `INVALID_REQUEST`, `STORE_NOT_INITIALIZED`, `STORE_VERSION_UNSUPPORTED`, `DEPENDENCY_MISSING`, `REPOSITORY_MISMATCH`, `AUTH_REQUIRED`, `ACCESS_BLOCKED`, `RATE_LIMITED`, `STORE_BUSY`, `REVISION_NOT_PREPARED`, `RECORD_NOT_FOUND`, `RECORD_NOT_ELIGIBLE`, `AMBIGUOUS_TARGET`, `CURSOR_STALE`, `MODEL_INVALID_OUTPUT`, `MODEL_FAILED`, `BUDGET_EXHAUSTED`, and `IO_ERROR`. `get` returns `RECORD_NOT_ELIGIBLE` with no body when a known record violates the cutoff. `details` is structured and sanitized. Partial collection is a successful report with gaps, not a fabricated complete result. A missing exact record is an error; an empty search is a successful empty result. Exact nested result schemas are in [public models](07-public-models.md); do not infer additional fields from this prose.

## 7. Workspace contract

`Memory.configure_workspace(path: Path | None)` is an optional local SDK configuration method; CLI/MCP use `--workspace`. It validates the workspace belongs to the store before reading it. This path is trusted local configuration, not a model-controlled path.

Return workspace status with `checked`, `head_revision`, `head_matches_index`, `target_state`, `index_blob_oid`, `staged_blob_oid`, and `working_content_hash` when available. `target_state`: `unchanged`, `staged_change`, `unstaged_change`, `staged_and_unstaged`, `deleted`, `untracked`, `unavailable`, or `not_checked`. Working-content hashes use SHA-256 over raw file bytes and are not presented as stored Git blobs.

V1 detects divergence; it does not build a transient syntax overlay. Staged content is checked separately from working content. Do not follow symlinks outside the root. Bound target reads to the standard source size limit. A changed target keeps all code anchors attached to the indexed commit and adds `WORKSPACE_TARGET_DIFFERS`; current applicability cannot be `corroborated` for the unsaved/dirty content. Untracked targets have no inherited historical lineage; return `UNTRACKED_TARGET` plus search suggestions. With no workspace, state `not_checked` explicitly.

`target_state` describes staged/working changes relative to the workspace's own HEAD. `index_blob_oid` describes the prepared memory snapshot. A clean workspace on a different commit can therefore be `unchanged` while `head_matches_index=false`; compare target bytes to memory independently and emit `WORKSPACE_TARGET_DIFFERS` when they differ. If the target matches but other workspace files differ, record repository divergence and keep rationale applicability conservative; file equality does not prove dependency equality.

## 8. Defaults, budgets, and read guarantees

| Setting | Default |
| --- | --- |
| Search candidates | Top 100 per lexical kind; merged shortlist 100 |
| Context expansion | At most 500 records, 100 relevant changes, and 25 episodes; typed history path length at most 6 |
| Read deadline | 2 seconds soft deadline for database/expansion work; return bounded partial results with `READ_BUDGET_EXHAUSTED` gap; do not kill an in-progress SQLite commit |
| Source chunk | 8 KiB UTF-8-safe chunks with 512-byte overlap for large text; full raw body remains stored when within limit |
| Parse size | 2 MiB per blob; larger files retain metadata/text chunks where applicable and a parse gap |
| Discussion body limit | 4 MiB decoded text per source; larger data is metadata-only with a recorded size gap |
| Single HTTP body limit | 32 MiB, streaming enforced |
| Model input episode limit | 24,000 estimated input tokens per request; chunk larger episodes and preserve dependencies |
| Default token estimate | `ceil(len(text.encode("utf-8"))/3)`; label it `utf8_bytes_div3_v1`, never claim it is exact |
| Read result serialization | Budget the complete canonical JSON response, including citation metadata; preserve the required envelope and uncertainty before adding optional content |
| Fixed read ceilings | `search`: 12,000 estimator tokens; `status`: 6,000. Status includes at most ten detailed gaps, full aggregate counts, and `gap_` expansion handles. Both responses expose budget/truncation/omission metadata. |
| HTTP concurrency | One in-flight request per store/principal in v1 |
| Parser concurrency | One parser worker initially; optimize only after measurements |

A caller can supply a local token-counting callback to `Memory` configuration. MCP uses the default estimator unless its process was explicitly configured otherwise. Deterministic ties sort by stable ID. A response's `budget` includes requested budget, estimated/actual count, estimator ID, truncated flag, and omitted counts.

Preflight the minimal required response envelope. If it alone exceeds `max_tokens`, raise bounded `INVALID_REQUEST` with `details.minimum_required_tokens` and no context/record payload. Do not return an over-budget pack. Aggregate coverage and omitted counts remain present when detailed gap lists are shortened; provide handles for omitted record details. An omitted gap is always reported as an omission, never silently erased.

Read-only means no network, source crawling, parser runs, model calls, migrations, or durable data writes. In-memory bounded caches are permitted. Source expansion reads only stored database records and owned Git objects. If historical parsing is pending, return raw evidence and a gap instead of doing hidden work.

## 9. Coverage and errors

Coverage has independent `git`, `github`, `structure`, and `enrichment` sections, each with `state` (`not_started`, `partial`, `complete`, `blocked`, `unavailable`), completed/known-total counts (unknown totals are null), last successful update, and gap IDs. Include selected refs, active revision, source reconciled-at times, excluded file counts, and parser fingerprints.

Gaps have `code`, `scope`, `message`, `retryable`, and optional `record_id`. They are never silently dropped because an answer has useful evidence. Represent API caps, missing permissions, missing objects, unprocessed history, stale derivations, ambiguous mappings, unsupported syntax, and truncated output separately. `complete` always means complete within declared scope, limits, and observation time.

Every read opens one SQLite read transaction, captures generation/access epoch, and reads only consistent published rows. Mutations increment generation in the same transaction as source/index/dependency updates. Cursors contain query fingerprint, generation, access epoch, and offset; a mismatch returns `CURSOR_STALE`. Do not offer historical-generation reads in v1. The final access-epoch validation is the response authorization point: a purge committed before it invalidates the response. A response already authorized/materialized or delivered cannot be recalled; do not promise stronger cross-process revocation.

## 10. Sources and interpretation

Package versions: [MCP](https://pypi.org/project/mcp/), [Tree-sitter](https://pypi.org/project/tree-sitter/), [Python grammar](https://pypi.org/project/tree-sitter-python/), [JavaScript grammar](https://pypi.org/project/tree-sitter-javascript/), [TypeScript grammar](https://pypi.org/project/tree-sitter-typescript/). Version availability was inspected, not installation-tested.

The official [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) documents typed tools/resources and standard transports. Use its public API and [getting-started guide](https://py.sdk.modelcontextprotocol.io/get-started/), not examples copied from an older SDK major. The record model, conservative applicability rules, budgets, and one-snapshot policy above are this project's design decisions, not claims made by those dependencies.
