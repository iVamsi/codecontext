# Implementation workbook for a lower-cost model

Status: all tasks below are planned, not implemented. This workbook is the execution order for the specification. Do one task at a time, keep its tests meaningful, and record evidence before marking it complete.

## 1. How to use this plan

Start each implementation session with [the master plan](../../PLAN.md), [shared contracts](00-contracts.md), and the current task below. Read only the task's referenced chapters in full. Use the repository knowledge graph for code discovery once source code exists; inspect callers and tests before modifying existing behavior.

The specification's scope and contracts are authoritative. Task descriptions specify incremental delivery, so a capability may be absent in the first slice but still be mandatory before release. Temporary limitations must appear in status/coverage and the progress record. Never quietly relabel a deferred requirement as implemented.

The following rules prevent a smaller model from losing the intent:

1. Work on the first incomplete task whose implementation dependencies and deterministic checks are complete. External evidence gates are tracked separately: T12 may follow T11's passing protocol/code tests while its real-agent pilot is pending; T22 may follow T21's tested harness while human/paid evaluation is pending; T23 packaging work may proceed while a hardware benchmark is pending. These gates still block final release. If a task exposes a bug in an earlier one, repair it with a regression test before advancing.
2. Before editing, state the task ID, expected behavior, touched modules, and relevant fixture/test IDs in the progress record.
3. Implement the specified observable behavior. Do not replace evidence, errors, parsing, or validation with mocked success outside tests.
4. Use real SQLite and temporary Git repositories in integration tests. Mock only external HTTP/model calls and deliberate failure injection.
5. Do not change expected fixture answers merely to make an implementation pass. A legitimate specification correction needs a written explanation and updated affected tests.
6. No embeddings, general plugin framework, extra hosting provider, agent runtime, web UI, or automatic background service is required by this release.
7. All generated claims retain source evidence and uncertainty. Never simplify away branch filtering, original review anchors, missing-history gaps, or dirty-workspace warnings.
8. Read tools remain free of network/model calls, parser execution, migrations, and durable writes. Test this with spies and deny hooks, not only code inspection.
9. Record actual commands/results. Do not claim a test, live pilot, paid-model run, or benchmark passed if it was skipped.
10. Stop and document a genuine external blocker, such as absent model credentials for a live pilot, while continuing independent implementation. Deterministic adapters support local development without credentials.

## 2. Planned repository layout

Create source modules as their tasks need them. Do not fill the tree with empty abstraction layers first.

```text
pyproject.toml
uv.lock
README.md
PLAN.md
docs/plan/                         # this specification
docs/implementation-status.md     # created at implementation start
src/codecontext/
  __init__.py                     # supported public exports only
  __main__.py                     # forwards python -m codecontext to CLI main
  errors.py                       # CodeContextError and transport mapping
  models.py                       # strict requests/results/enums/anchors
  identity.py                     # canonical IDs, hashes, time/range utilities
  store.py                        # connections, transactions, records, invalidation
  migrations/001_initial.sql
  git.py                          # safe Git plumbing and owned object storage
  github.py                       # GET client, normalization, checkpointed streams
  sync.py                         # fair resumable orchestration; never model calls
  syntax.py                       # blob-based parsing and query packs
  queries/python.scm
  queries/javascript.scm
  queries/typescript.scm
  queries/tsx.scm
  lineage.py                      # anchors, hunks, associations, mapping certainty
  workspace.py                    # safe HEAD/index/raw-worktree comparison
  enrichment.py                   # adapters, budgets, extraction, summaries
  retrieval.py                    # FTS ranking, eligibility, traversal, packing
  memory.py                       # public facade; composes the modules above
  cli.py                         # argparse and result/error presentation
  mcp_server.py                  # thin typed stdio adapter
  archive.py                     # consistent export, read-only reopening, purge
tests/
  conftest.py
  fixtures/                      # deterministic builders + small recorded payloads
  unit/
  integration/
  performance/                   # explicit opt-in, not ordinary CI
evals/
  corpus.lock.json
  questions.jsonl
  rubric.schema.json
  run.py
  score.py
  reports/                       # sanitized reports, not private raw archives
examples/
  fixture_adapter.py
  agent-instructions.md
  mcp-config.json
```

Keep SQL inside the storage module/migrations, Git process execution inside `git.py`, HTTP inside `github.py`, and model calls inside `enrichment.py`. A test may inject those dependencies at these seams. The facade validates inputs and orchestrates work; it must not duplicate business rules already owned by a module.

## 3. Task order and dependencies

The early MCP slice finishes at T11. It proves the principal behavior before exhaustive crawling and release tooling.

| Task | Outcome | Depends on |
| --- | --- | --- |
| T01 | Reproducible package and dependency smoke checks | None |
| T02 | Strict records, identities, and deterministic fixture foundation | T01 |
| T03 | SQLite schema, transactions, and generation consistency | T02 |
| T04 | Safe owned Git objects and one prepared snapshot | T03 |
| T05 | Blob-owned syntax and text extraction | T04 |
| T06 | Focused changes and original code evidence | T05 |
| T07 | Focused GitHub PR/review/issue acquisition | T06 |
| T08 | Search and exact source expansion | T07 |
| T09 | Workspace divergence and bounded context assembly | T08 |
| T10 | One explicitly budgeted decision-extraction path | T09 |
| T11 | Real SDK/CLI/MCP end-to-end demonstration | T10 |
| T12 | Exhaustive resumable Git scope | T11 |
| T13 | Full GitHub enumeration and fair scheduling | T12 |
| T14 | Source versions, incremental updates, and reconciliation | T13 |
| T15 | Review-to-landed mapping and difficult histories | T14 |
| T16 | Complete enrichment scheduling and budget recovery | T15 |
| T17 | Knowledge cutoffs and historical-source eligibility | T16 |
| T18 | Complete invalidation, access changes, and source purge | T17 |
| T19 | Evidence-linked module and project summaries | T18 |
| T20 | Portable export, reopen, and entire-store purge | T19 |
| T21 | Human-reviewed explanation evaluation | T20 |
| T22 | Load, first-result, incremental-cost, and latency validation | T21 |
| T23 | Packaging, documentation, and release readiness | T22 |

### T01 — Package and environment

Read contracts §§1–2. Create `pyproject.toml` with the specified dependency pins/ranges, Python support, package layout, and console entry point `codecontext = "codecontext.cli:main"`. The initial CLI can expose only `--help` and `--version`; it must not pretend functional commands exist. Add a minimal empty-package import test.

Run `uv lock`, `uv sync --frozen`, and dependency smoke checks: SQLite FTS5/JSON, supported Git version, all four grammar entrypoints/query compilation, and an official SDK in-memory tool call. Verify the SDK major's actual public API rather than copying an old FastMCP example. Record resolved dependency versions in `uv.lock` and the smoke report.

Done when a fresh environment can import the package and the supported dependency checks pass. Missing binaries or unsupported SQLite capabilities produce actionable errors. Do not initialize a target project's Git repository or install target-project dependencies.

### T02 — Types, identity, and fixtures

Read contracts §§3–9, the complete [public models](07-public-models.md), and validation F01/F03/F09, U01–U03. Implement all public request/result models, enum values, error codes, `Target`, anchor types, record union, and cross-field validators before transport adapters. Provide nullable revision for an initialized but unprepared store.

Implement canonical JSON/ID hashing, byte-path representation, UTC conversion, byte/line span conversion, and redacted error details. Create reusable deterministic Git fixture builders using fixed author/committer identities/times and explicit branch names. Create normalized mock GitHub payload builders, keeping transport IDs separate from project IDs.

Done when malformed requests fail before subprocess/HTTP/model calls, stable IDs remain stable under irrelevant observation changes, and SHA-256 Git IDs are not truncated. Export only the types listed in the contracts.

### T03 — Store and publication

Read the storage chapter. Implement the planned initial migration, migration checksum/version checks, writable/read-only opens, context-manager behavior, and classmethod initialization. Use real SQLite tests for foreign keys, FTS triggers, and generation changes.

Implement short write transactions and snapshot reads. Record sources/versions, symbols, changes, evidence, jobs, and gaps using narrow private methods. Validate polymorphic evidence/relationship references before insertion. Add the store operation lock and transactional update recipes before orchestration depends on them.

Done when readers see the old or new generation, interrupted writes roll back, missing stores do not auto-create, and future schema versions are rejected. U14/U15 and the storage portion of I09 must pass.

### T04 — Safe Git ownership and active snapshot

Read ingestion §§2–3 and storage §§4/7. Implement one binary-safe subprocess runner with explicit environment/configuration, timeouts, output limits, and cancellation cleanup. Build an owned bare mirror of the source object's format with no alternates or source hard-link dependence.

Import the requested commit and declared refs without changing the developer checkout. Resolve HEAD once at synchronization start. Parse `cat-file`, `ls-tree`, commit objects, and parent edges. Prepare the active file map and ancestor set atomically, initially with pending syntax status. Verify repository binding on later opens/syncs.

Done when source checkout/ref/index hashes are unchanged, all selected source bytes remain readable after detaching the source fixture, and an interrupted snapshot switch leaves the prior snapshot usable. Cover F03/F09 and I01/I05. No GitHub network support is required yet.

### T05 — Syntax and text

Read ingestion §4 and storage symbol ownership. Implement `parse_blob(bytes, language, fingerprint)` privately. Add exact extraction/query fixtures for all promised syntax forms; store valid partial results with diagnostics on malformed code. Parse identical blobs once per language/fingerprint, independent of paths and commits.

Index bounded source/document chunks with raw byte anchors and separate normalized search text. Handle binary, oversize, undecodable, symlink, gitlink, LFS-pointer, and unsupported-language cases explicitly. Keep dynamic calls/imports as candidate occurrences.

Done when U04/U05, Unicode/range U02, and the small F12 storage invariant pass. Do not add a compiler, project build, or custom semantic resolver.

### T06 — Focused history and anchors

Read ingestion §§1/3/6. Implement parent-specific raw changes/hunks for a bounded focus history, retaining native empty-side coordinates. Relate changed ranges to existing blob-owned symbols and create exact code/commit/hunk evidence handles.

Implement straightforward exact original anchors and unchanged-blob correspondence first. Retain unsupported/ambiguous mappings as gaps; do not create a heuristic that labels every same-named function an exact match. Persist remaining full-history work so focus processing does not become an accidental permanent scope limit.

Done when F01's function connects to its introducing/modifying changes and exact source bytes. Root and merge changes, insertions/deletions, and an unknown mapping have distinct expected outputs. U06 and the non-GitHub part of I04 pass.

### T07 — Focused GitHub evidence

Read ingestion §5 and storage §6. Implement validated primary repository identity, authenticated GET requests, principal binding before private evidence is persisted, the pinned API version, streaming response limits, and sanitized errors. Reject a changed principal rather than reusing private evidence silently. For focus commits, fetch provider-associated PRs, PR metadata, all paginated conversation/review comments, reviews, and explicitly linked allowed issues. Preserve original inline review anchors.

Implement source normalization/version creation, one PR identity, parent/reply relationships, and immutable local evidence handles. Add deterministic HTTP fixture responses for partial pages, unavailable sources, body edits, and API caps. Store all known coverage gaps.

Done when a focus synchronization can acquire F01/F02 discussions and resume after one interrupted page without duplication. No read operation may perform these requests. The source streams may still be partial until T13/T14.

### T08 — Search and expansion

Read enrichment/retrieval §§5–7. Implement literal-safe normalized FTS queries, record-kind filtering, exact-match prioritization, rank fusion, deterministic ties, bounded shortlist pagination, and source/claim eligibility. Use one FTS table partitioned by record kind as specified by storage.

Implement `get` resolvers for each published ID kind and exact source spans using the exact union in [public models](07-public-models.md). Apply source/store access checks to every read, including direct IDs. Responses include anchors, attribution, direct relationships, coverage, gaps, and budget metadata. Budget the full serialized envelope; reject a budget smaller than the mandatory envelope. Cursors bind request fingerprint, generation, and access epoch.

Done when U11/U13/U14 pass, direct historical source handles remain inspectable, private blocked sources cannot be retrieved by ID, and a malicious FTS string remains literal data. T08 is functional without a model.

### T09 — Workspace and context

Read contracts §7, ingestion §7, and bounded expansion/packing rules. Configure a trusted workspace separately from query arguments. Compare prepared snapshot, workspace HEAD, index entries, and bounded raw working bytes without invoking status hooks/clean filters.

Implement `context` target resolution, typed expansion, required evidence closure, deduplication, gap precedence, deadline, and complete-response packing. Use stored raw evidence while enrichment is absent. Do not call a model or parse old blobs inside context. Return explicit ambiguity/unprepared-revision errors.

Done when F07/U10/U11/I07/I08 pass. Dirty or untracked content has no invented indexed lineage or corroborated applicability. A clean checkout on another commit is distinguished from staged edits.

### T10 — One valid extraction flow

Read enrichment/retrieval §§2–4. Implement the public adapter protocol and strict extraction schema, including supporting/counter/lifecycle citations. Build one complete bounded episode, reserve budget before invoking the adapter, validate exact quote uniqueness, and publish claims/dependencies only after rechecking input fingerprints.

Use the deterministic adapter first. Exercise one explicit rationale, an absent reason, an invented source ID, and a timeout with uncertain usage. Compute conservative applicability in retrieval; unchanged bytes never upgrade it. The CLI must not choose a provider automatically.

Done when a complete F01 episode produces an attributed claim and a completely inspected no-rationale episode produces the correct abstention status. An unprocessed/inaccessible episode must produce a different status. U08–U10/U12/U16 pass for this bounded path.

### T11 — First real agent experience

Read delivery chapter and validation I06/I08. Implement the SDK facade, CLI commands already supported, and the three read-only MCP tools with exact schemas. Expose the status resource. Exercise in-memory and subprocess stdio clients; reserve stdout exclusively for protocol output when serving.

Write the small agent-instruction example and generic MCP configuration. On one public repository selected from the pilot corpus, run focus sync and, only with an explicitly configured adapter/budget, optional enrichment. Have a real agent explain a selected function, expand its original discussion, and answer an undocumented-intent case. It may reason from raw cited sources when enrichment is pending; the server must still report that state honestly.

Done when the saved sanitized transcript proves the agent discovered and used the tools, cited original evidence, and observed workspace/coverage warnings. A protocol echo test alone does not satisfy this task. If a real client is unavailable, record that gate as pending while continuing deterministic implementation.

### T12 — Full Git backfill

Complete configured ref enumeration, every accessible reachable commit, per-parent changes, bounded hunks, historical changed-blob syntax, missing-object recovery, and retained evidence refs. Stream large histories and persist idempotent jobs. Handle force pushes, shallow boundaries, changed ref tips, and source-object garbage collection.

Implement focus/backfill fairness and resume with time limits. Focus prioritization must not reset or starve older history. A metadata-only file remains counted with a content/structure gap. Done when I01/I02 and the Git part of I03 pass, plus P01's structural storage-growth assertion.

### T13 — Full GitHub scope

Finish the endpoint matrix across every primary issue and PR, including closed/unmerged PRs and all paginated child collections. Implement one-hop allowlisted external issues. Validate pagination destinations before sending credentials. Respect request/time budgets, retries, rate reset times, conditional responses, and persistent queue fairness.

Done when full F08 enumeration matches the fixture's independent expected ID inventory. Interrupted/capped/unauthorized streams cannot report complete coverage. I02/I03/I13 pass without live network dependency.

### T14 — Incremental source versions and reconciliation

Implement overlapping incremental windows, repository-wide comment update streams, changed-parent refresh, and persistent round-robin old-child refresh. Full reconciliation uses successful pass membership to distinguish missing records from interrupted enumeration. Preserve observed earlier bodies and their source times.

Retain version documents in the FTS index with eligibility metadata; select one eligible version per source before ranking. This is required for final-release knowledge-cutoff retrieval, even though current-only FTS sufficed for the early slice. Implement source-edit invalidation of directly dependent claims before publishing new source search rows.

Done when edited comments with unchanged parent timestamps are eventually detected, no-change syncs do not duplicate versions, and source search never matches text added after a historical cutoff. F06/F08 and incremental portions of I02/I03 pass.

### T15 — Difficult lineage

Complete original RIGHT/LEFT review anchors, missing comparisons, replies, final-head/landed distinctions, parent-specific range transport, unique exact moves, bounded rename candidates, squash/rebase associations, copy ambiguity, deletion/recreation, and reverts.

Every mapping stores its method and input anchors. One candidate link makes the overall correspondence a candidate. An unmerged PR does not become a landed-code explanation. Done when F02–F05/U06/I04 pass with exact expected mapping statuses and a later constraint change does not retroactively rewrite the original explanation.

### T16 — Enrichment completeness and recovery

Complete multi-segment episode extraction, deterministic deduplication, cross-segment unresolved relationships, full dependency registration, aggregate token/call/spend reservations, cache reuse, bounded schema repair, uncertain-charge handling, and persistent run reports.

No episode becomes fully inspected until every required segment and available source is accounted for. Missing usage preserves conservative reservations. Repeat unchanged enrichment uses the validated cache; source or model fingerprint changes invalidate it. Done when F10/U12/U16/I11 pass and reported model usage matches the adapter fixture ledger.

### T17 — Time and branch eligibility

Complete eligibility for every public read path and every derivation dependency. Use active-snapshot ancestry independently from source authored/edited timestamps. Prefer eligible captured versions; exclude unverifiable source times with explicit gaps. Direct historical `get` remains clearly distinguished from target applicability.

Done when F03/F06/U07/I05 and adversarial future-summary cases pass. Do not implement historical answer synthesis by taking today's edited descriptions and assigning their original creation date.

### T18 — Full invalidation and access retention

Finish recursive reverse-dependency invalidation for sources, mappings, extraction fingerprints, claims, and summaries. Extend T07/T08's principal/access protection with confirmed revocation, the access-epoch authorization check, discussion-source purge, stale cursor behavior, and best-effort physical cleanup reporting. Purge deletes generated bodies in the full dependency closure; retaining stale text is insufficient.

Test races with barriers: a completed revocation before final authorization validation invalidates the pending response; already-authorized returned results cannot be recalled. Done when I09/I10/I13 pass and no current query or direct ID can recover logically purged content. Do not promise secure physical SSD erasure.

### T19 — Project and module views

Add deterministic project orientation from the active snapshot. Follow enrichment §8 exactly: deepest non-root `pyproject.toml`/`package.json` directory, otherwise top-level directory or root-files group, plus one repository summary. Do not infer compiler module boundaries or create every-directory summaries. Optional model summaries use bounded statements with exact citations and register all presented dependencies.

Summaries are navigation aids; every factual statement links to source records, and exact evidence expansion remains available. Missing enrichment yields a structural description. Invalidate only affected summaries on a source/code change, and invalidate revision-specific summaries on snapshot replacement. Done when a changed module's summary becomes stale without invalidating an unrelated module unnecessarily.

### T20 — Export and purge

Read storage §§6–8 and delivery CLI definitions. Implement consistent SQLite backup, owned Git-object copying, sanitized portable metadata, file hashes, atomic publication into a new directory, and read-only archive reopen. Exports retain the store identity/generation while removing credentials and machine-specific paths from both JSON and SQLite.

Implement whole-store purge with ownership-marker validation and closed-instance semantics. A metadata-only export states its expansion limits. Done when I10/I12 pass, the archive works after removing the source checkout, and an existing destination cannot be overwritten.

### T21 — Explanation evaluation

Implement the E01–E08 harness, schemas, run logs, same-source baseline, family-level split, time filtering, and scoring calculations. E08 defines the injected evaluation runner separately from the extraction adapter; use its deterministic runner for local harness checks. Capture the pinned default-branch commits when creating the corpus lock; record exact source versions and collection windows. Do not hard-code moving branch names as benchmark identities.

Prepare 120 questions with the specified 40/80 split and independent expected evidence/required facts. Human review of gold answers is a release requirement, not something a coding model can self-certify. Model-generated drafts may assist but remain unreviewed until checked. Run development evaluation before held-out evaluation; log any subsequent benchmark exposure.

Done when the report includes claim support, required-fact recall, answerable success, unsupported abstention, applicability, counterevidence, and uncertainty intervals. An unfinished human review or absent model budget leaves that gate pending, not passed.

### T22 — Performance and operating cost

Implement explicit performance commands for P01–P08. Record hardware, dependency versions, corpus/fixture identity, warm-up method, process-cold versus cache-cold distinction, query distributions, truncation, memory, disk, HTTP counts, and model usage.

Run the 100k-commit/approximately-1M-current-line workload on the reference-class machine. Measure first searchable source, first useful episode, first cited explanation, completed backfill, and no-change/incremental runs separately. Optimize only measured bottlenecks while preserving correctness fixtures. Do not satisfy latency goals by returning empty/truncated results.

Done when targets are met or a concrete unmet target is recorded as a release blocker. Any deliberate scale/latency-contract change must update the plan and release claims explicitly; a smaller benchmark is not equivalent evidence.

### T23 — Release readiness

Finish documentation for installation, dependencies, read-only agent setup, collection/enrichment budgets, source scope, known limits, privacy/export behavior, and example questions. Add CI for supported Python/OS combinations, deterministic tests, lint, package build, and clean-install smoke tests. Keep large/live/paid evaluation opt-in.

Run the full deterministic suite once after integration changes settle. Validate all public commands' help/error behavior and SDK/CLI/MCP response equivalence. Preserve third-party license notices for grammar queries and dependency code actually reused. Confirm distribution-name availability before publishing; `codecontext` remains the working import/project name until then.

Done when a clean installation reproduces the early user story and every release gate has actual evidence. Publishing a package is a separate external action; this task prepares a reviewable build and release checklist, not an automatic upload.

## 4. Progress record template

At implementation start create `docs/implementation-status.md` using this structure. Update it at the end of every task. Do not use this plan document as evidence that code exists.

```markdown
# Implementation status
Active task: T01
Last completed task: none
Prepared repository/revision: none

| Task | State | Changed files | Validation evidence | Remaining limits |
| --- | --- | --- | --- | --- |
| T01 | not_started | | | |

## Decisions changed from the specification
None.

## External gates
- Real MCP-client pilot: pending
- Live model evaluation budget/adapter: pending
- Human-reviewed gold answers: pending
- Reference-machine load benchmark: pending

## Next handoff
Read PLAN.md, 00-contracts.md, and T01 in the workbook.
```

States are `not_started`, `in_progress`, `implemented_unverified`, `verified`, and `blocked_external`. `verified` requires the relevant checks. If implementation/deterministic checks pass but an external gate is pending, record `blocked_external` plus `implementation_complete=true` and the missing evidence; downstream independent code work may proceed under rule 1. A temporary shortcut gets a code comment of the form `shortcut: <limit>, <when to upgrade>` and an entry in remaining limits. It cannot bypass a release requirement.

## 5. Copyable implementation prompt

```text
Implement task TXX from docs/plan/06-implementation-workbook.md.
Read PLAN.md, docs/plan/00-contracts.md, the task's referenced specification
sections, and docs/implementation-status.md first. Inspect existing code
through the configured knowledge graph before editing. Do not implement
unrelated later tasks or replace required behavior with placeholders.

Preserve the public contracts, conservative evidence/applicability rules,
blob-level deduplication, and read-only query guarantees. Write the relevant
fixture/unit/integration tests described by this task. Use real SQLite and
temporary Git for local integration tests; mock external HTTP/model calls.
Run the targeted checks and report their actual results. If a prior task is
broken, fix it with a regression test. Update implementation-status.md with
changed files, validation, pending gates, and the next task. Never mark a live
pilot, human review, or performance target passed without its evidence.
```

For a model with a small context window, copy only the current task and its linked contract sections into the working context after reading the master plan. Keep full source evidence on disk and inspect it through tools. Do not ask the model to remember the entire specification from an earlier conversation.

## 6. Final release checklist

- [ ] All T01–T23 code tasks are verified; external gates have real evidence.
- [ ] All configured accessible history is accounted for; exclusions and unavailable records remain visible.
- [ ] Exact original sources are expandable from every emitted citation.
- [ ] Historical rationale is not silently promoted to a current requirement.
- [ ] Unchanged blobs reuse extraction and symbols; no per-commit full-symbol snapshots exist.
- [ ] Dirty/workspace branch differences are visible in actual agent responses.
- [ ] `search`, `get`, `context`, and status have no hidden network, parser, model, or durable-write work.
- [ ] All three read surfaces return equivalent content and error semantics.
- [ ] Partial runs resume, source edits invalidate dependencies, and purge prevents future authorized retrieval.
- [ ] Same-source evaluation measures completeness and answerability, not only citation precision.
- [ ] First-result, full-backfill, incremental, storage, and latency costs are measured and published.
- [ ] Package/version/platform limitations and skipped evidence sources are documented.

This workbook enables incremental implementation and review. It does not make historical intent mechanically knowable or eliminate the need for real-client, human-evaluation, and performance checks.
