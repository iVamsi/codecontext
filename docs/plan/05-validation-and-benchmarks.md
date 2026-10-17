# Validation and benchmarks

Status: implementation specification, not executed. Updated 2026-10-16.

This specification tests whether CodeContext gives agents useful historical explanations without overstating what the evidence establishes. It also tests the cost of building that memory. [Shared contracts](00-contracts.md) define public names, limits, and errors; this document defines reproducible fixtures and acceptance checks.

Implement tests alongside the feature they validate. Deterministic tests run offline with temporary repositories, HTTP mocks, a fake model adapter, and a controlled clock. External-repository evaluations and large load tests are explicit commands, not requirements for an ordinary unit-test run. No test executes the indexed project's code, hooks, builds, or package installation.

## 1. Test organization and evidence

Use stable IDs in pytest names, benchmark manifests, and task-completion reports: `Fxx` for fixtures, `Uxx` for unit checks, `Ixx` for integration checks, `Exx` for evaluation procedures, and `Pxx` for performance workloads. Parameterized cases retain their parent ID and add a descriptive suffix. Never reuse an ID for a different requirement.

Each run records the CodeContext commit, lockfile hash, Python/Git/SQLite/parser versions, operating system, fixture version, requested revision, and relevant configuration. Benchmark records also include hardware, corpus hashes, model configuration, budgets, and raw measurements. A passing result links to the exact run. Documentation describing a check is not evidence that it passed.

Run deterministic suites on Linux and macOS with Python 3.12 and 3.13. Test both supported Git object formats where the installed Git supports them. Run real MCP transport tests against the locked SDK. Keep paid model evaluations opt-in with a declared spending limit.

## 2. Deterministic fixtures

### F01: documented retry rationale

Create `transport.py` with a named `send` function at commit C0. C1 adds a three-attempt retry limit. A mock PR contains: “Limit retries to three because each retry holds a worker and the legacy proxy duplicates requests after its timeout.” A second comment proposes unlimited retries; a maintainer explicitly rejects it. C2 changes unrelated documentation.

Store gold byte spans for both constraints, the rejected alternative, author identities, PR linkage, and C1's hunk. The expected rationale is attributed and historically supported. No fixture statement establishes that the legacy proxy remains deployed at C2.

### F02: review comment on replaced code

Create PR head R1 with five attempts, then R2 with three attempts. Anchor an inline comment to the five-attempt line at R1. Add an explicit later discussion selecting three attempts. Land R2 through a squash commit with different object identity.

Provide original/current provider anchors and the actual Git blobs. Include an obsolete suggestion never implemented. Expected evidence expansion displays R1 faithfully, while the target code displays the landed three-attempt implementation. Mapping confidence and adoption require separate evidence.

### F03: branching and integration history

From one root, construct independent ordinary-merge, squash, rebase, cherry-pick, and revert histories using deterministic author identities and times. Include a two-parent merge with meaningful changes against each parent. Create equal-looking changes on divergent branches and a commit whose timestamp predates its parent.

The manifest records each parent edge and intended source association. Ancestry determines eligibility; dates and matching patch text alone do not establish the same logical decision or branch membership.

### F04: symbol identity and ambiguous moves

Rename a file without changing its blob, move a function while editing it, copy it twice into different modules, delete it, and later recreate the same qualified name with different behavior. Include two equal-sized named symbols enclosing a requested line in parser test data.

Record exact file/blob associations and ambiguous candidate associations separately. A copied or recreated name must not inherit an old rationale as an established current constraint. Equal target candidates require `AMBIGUOUS_TARGET`.

### F05: changed constraints and corroboration

Extend F01 without changing `send` bytes. Change a caller, dependency declaration, and deployment configuration. Add conflicting maintainer statements about whether the proxy remains relevant. Finally add an explicit ADR identifying the constraint and the target revision where it still applies.

Provide separate cases with no ADR, an unrelated ADR, and the scoped ADR. Only the last can produce `applicability="corroborated"` for its stated scope. Tests confirming retry behavior establish behavior, not the original author's intent.

### F06: edits, deletion, and historical knowledge

Create a comment at T1, observe it at T2, edit it at T3, and delete it at T4. Preserve both observed bodies. Add another edited comment for which only the latest body is available, plus a record with missing source timestamps. Include future discussion on an older commit.

Gold results identify which versions are eligible at each cutoff. Observation time never fills a missing source timestamp. Missing earlier bodies and unverifiable times produce gaps; later claims and summaries cannot leak through derived records.

### F07: developer workspace states

Use independent worktrees for unchanged, staged-only, unstaged-only, staged-and-unstaged, deleted, untracked, and inaccessible targets. Add a workspace whose HEAD differs from the prepared snapshot and one belonging to another repository, including a separate clone with the same remote. Include an external symlink, an oversized target, executable fsmonitor/hooks/filter fixtures, and permission errors. Assert none of the configured executables run.

Expected hashes are calculated from exact bytes. Staged content comes from the index, working content from the file, and historical content from the prepared revision. Never reinterpret a working-content SHA-256 as a Git blob ID.

### F08: GitHub collection and recovery

Serve paginated mock issues, PRs, conversation comments, reviews, inline comments, and relationships. Include closed/unmerged PRs, a record updated across a pagination boundary, allowed and disallowed cross-repository issue links, deleted content, and repository rename with unchanged provider ID.

Inject rate limits, transient failures, an oversized body, malformed responses, missing permissions, inaccessible PR heads, and endpoint result caps. Provide explicit provider response metadata so retry, checkpoint, and gap behavior can be asserted without real waiting.

### F09: syntax, paths, and size limits

Include Python, JavaScript, TypeScript, JSX, and TSX definitions, nested symbols, imports, unresolved calls, syntax errors, and duplicate names. Add unsupported-language text, Unicode before source ranges, CRLF bodies, files at size boundaries, unusual Git filenames, and a non-UTF-8 path on supported filesystems.

Store expected byte ranges and human line numbers. Assertions distinguish parser facts from candidate semantic relationships. Oversized and unsupported content retains permitted text/history access and explicit structure gaps.

### F10: extraction and hostile source content

The fake adapter returns valid explicit claims, inferred claims, invented source IDs, out-of-range spans, malformed JSON, extra fields, timeouts, usage overruns, conflicting claims, and an empty claims array acknowledging only a subset of required slots. Include repository text instructing an agent to reveal secrets, run commands, or ignore later evidence.

Source text remains quoted data. Invalid model output cannot publish a supported claim; incomplete inspection cannot establish undocumented intent. Mark a fake runtime credential and assert it never enters model requests, logs, database, or export. Separately mark a secret-like string committed in fixture history: it remains source content, demonstrating that v1 does not promise automatic history sanitization. Invoking a cloud adapter explicitly permits transmission of the selected source bytes; local-only adapters avoid that transmission.

### F11: publication, pagination, and purge

Prepare a store containing enough results for multiple pages, claims supported by several sources, counterevidence, and dependent summaries. Add deterministic barriers immediately before publication and before response emission. Maintain an existing export to show that copied artifacts are outside subsequent purge control.

Use this fixture for generation changes, stale cursors, concurrent reads, access-epoch changes, source purge, entire-store purge, and invalidated search indexes. A claim influenced by a purged source is removed entirely even if other support survives; dependent summaries cannot retain its text indirectly.

### F12: storage growth and large history

Provide two generators. The regression generator starts with 200 files containing ten symbols each, then creates 5,000 commits changing one function in one file per commit. The load generator creates approximately one million current source lines and 100,000 reachable commits with deterministic small edits, occasional renames, merges, and discussion episodes.

Record unique blob/extractor combinations, expected symbol records, changed paths, source versions, and discussion bytes. Count records independently from the implementation. Each Git commit still names a complete tree; the generator reuses unchanged blobs/subtrees instead of rewriting every file in every commit.

## 3. Unit checks

Each entry states setup, action, and required result. Prefer public behavior tests; inspect internal rows only for storage invariants that public output cannot demonstrate.

| ID | Setup and action | Required result |
| --- | --- | --- |
| U01 | Hash reordered identity fields, changed observation times, changed payload fields, and both Git object formats. | Canonical identities are stable; relevant source edits change version IDs; observation alone does not; full object IDs survive. |
| U02 | Convert F09 byte ranges, Unicode, CRLF, and empty Git hunks. | Exact spans round-trip; line numbers are one-based; empty ranges never become negative. |
| U03 | Submit empty/oversized queries, unknown fields, invalid budgets, absolute paths, traversal, and malformed targets. | `INVALID_REQUEST` occurs before Git, HTTP, or model calls; error details expose no secrets. |
| U04 | Extract every supported F09 language twice, then alter extractor fingerprint. | Expected syntax records exist; identical cache keys reuse results; changed fingerprints invalidate reuse. |
| U05 | Parse malformed, unsupported, and oversized content. | Usable supported fragments remain correctly labeled; missing structure produces gaps, not fabricated graph edges. |
| U06 | Match F02/F04 original anchors against target code. | Exact and candidate mappings remain distinct; ambiguous candidates retain their method and cannot become causal proof. |
| U07 | Apply F06 cutoff and F03 ancestry filters. | Future, unverifiable-time, and wrong-branch evidence cannot support an eligible claim or derived summary. |
| U08 | Compute F01/F05 applicability for several targets. | Historical rationale defaults to `historical_only`; only scoped explicit corroboration upgrades it; values are computed per response. |
| U09 | Extract a proposal from a merged PR and an explicitly accepted/rejected proposal. | Merge status alone leaves claim lifecycle unestablished; explicit evidence supports the respective lifecycle. |
| U10 | Query an inspected undocumented episode, unprocessed/partly acknowledged episode, incompletely collected episode, and inferred-only rationale. | Return the correct `rationale_status`; only explicit non-null rationale with exact requested-target correspondence establishes supported intent. Useful partial evidence does not hide gaps. |
| U11 | Serialize complete responses with boundary budgets and Unicode. | Entire canonical JSON fits the declared estimator; truncation preserves uncertainty; an oversized mandatory envelope returns bounded `INVALID_REQUEST` with `minimum_required_tokens`. |
| U12 | Reserve adapter tokens/calls and an optional monetary limit with missing prices. | Invalid limits fail before invocation; reservations prevent knowingly exceeding the budget; prices are required for monetary enforcement. |
| U13 | Rank exact symbols, paths, lexical ties, and mixed record kinds. | Exact matches remain discoverable; stable IDs break ties; source-specific scores do not compare as if inherently calibrated. |
| U14 | Change cursor query, cutoff, generation, or access epoch. | Incompatible cursors return `CURSOR_STALE`; compatible paging has stable order without duplicates. |
| U15 | Aggregate partial collectors with unknown totals and truncated gap detail. | Unknown counts remain null; aggregate coverage remains present; omitted gap details have explicit counts/handles. |
| U16 | Validate F10 claims and source spans. | Invalid references cannot publish; attributed statements retain their author; an exact span alone does not prove semantic support. |

## 4. Integration checks

### I01: idempotent collection and ownership

Initialize a temporary store, synchronize F01/F03, and repeat unchanged synchronization. Compare source IDs, symbol counts, dependency records, and Git objects. No duplicate content versions or symbols appear. The developer checkout, refs, index, and configuration remain unchanged. Rebinding the store to another repository returns `REPOSITORY_MISMATCH`.

### I02: useful partial progress and restart

Synchronize F08 with a request budget that interrupts backfill after a useful episode. Query the published memory before completion, then resume repeatedly. Available evidence works immediately with partial coverage. Completed streams are not restarted unnecessarily; eventual accessible-scope completion preserves identities and records every remaining exclusion.

### I03: complete declared source scope

Crawl F08 through all pages and reconciliation. Assert inclusion of closed issues, unmerged PRs, all configured discussion kinds, and explicit allowed issue links. Distinguish platform-confirmed links from textual references. Repository rename preserves identity. API caps, inaccessible heads, and permission gaps prevent a misleading claim of complete coverage.

### I04: review-to-landed-code explanation

Collect and enrich F02/F04 with valid deterministic adapter outputs. Call `context`, then `get` for the cited review and code. Original review bytes and anchors remain inspectable; target code displays its own revision. Obsolete review suggestions cannot explain landed behavior as adopted decisions without supporting evidence.

### I05: one prepared snapshot and branch safety

Prepare one F03 branch, then request another branch through `search` and `context`. Expect `REVISION_NOT_PREPARED`. Explicit offline synchronization prepares that branch atomically; queries then use its full SHA. Historical `get` handles remain inspectable. Switching back reuses unchanged parsed symbols and does not recollect discussion history.

### I06: first real MCP user story

Before exhaustive backfill, connect a real MCP-capable coding agent over the shipped local transport. Use a small licensed real repository pinned to a recorded SHA and containing one independently verified rationale episode. Ask: “Explain why this function has this limit, the rejected alternatives, and what evidence establishes that the reason still applies.”

Record the actual tool transcript. The agent must find the target through `search` or `context`, inspect supporting evidence with `get`, distinguish historical reason from current applicability, and produce usable citations. Repeat against synthetic revert, conflict, and undocumented cases. Missing evidence must produce the appropriate uncertainty. Run a protocol-only client separately to isolate transport failures from agent tool-choice failures. Mock-only success does not satisfy this milestone.

### I07: workspace divergence

Run `context` across every F07 state. Verify `head_revision`, `head_matches_index`, target state, and available blob/hash fields. Changed targets include `WORKSPACE_TARGET_DIFFERS`; their code anchors remain tied to the indexed commit. Untracked targets expose `UNTRACKED_TARGET` and suggestions without invented lineage. Missing workspace configuration returns `not_checked`. External symlinks and oversized reads stop at the configured boundary.

### I08: reads have no hidden processing

Open a fully prepared store read-only. Replace HTTP, adapter invocation, parsing, migration, and Git-fetch entrypoints with sentinels that fail on use. Exercise successful, empty, ambiguous, truncated, stale, and historical reads through SDK and MCP. Compare durable logical data before/after and assert no write SQL. Local reads of owned Git objects are allowed. Pending parsing/enrichment returns raw evidence and gaps; credentials are unnecessary.

### I09: consistent generations

Pause a writer before publication using F11 barriers while a reader builds a response. The response contains one generation only. Publish and start another read; it sees the new generation. Failed publication leaves the last usable generation intact. Existing cursors become stale after a generation change, with no historical-generation fallback.

### I10: purge and access race

Start a read containing a source, pause before its final access-epoch authorization check, then purge that source. Resume and verify that no revoked content is authorized. Search, direct expansion, summaries, counterevidence, caches, and subsequent exports cannot reveal purged text. Query previously saved version/evidence/claim/summary IDs directly and expect `RECORD_NOT_FOUND`, including a claim with additional surviving support. Old cursors fail through their access epoch. Test the separate allowed case of a response authorized before purge; copies cannot be recalled. Entire-store purge removes owned memory without deleting the developer repository. A copied prior export remains outside purge guarantees.

### I11: extraction failure and recovery

Run F10 through `enrich` with each invalid adapter behavior and exhausted budget. Failures are structured and checkpointed; valid prior claims remain available. Retry valid outputs without duplicating claims. Ensure runtime credentials never enter adapter input, and repository instructions never select executable adapters or trigger commands. Verify partial slot inspection remains pending across restart.

### I12: export consistency

Export F11 while publication occurs. The result represents a coherent generation and includes required metadata and owned objects when requested. Reopen it offline and expand citations. An existing destination fails without overwriting it. Metadata-only exports declare their inability to guarantee code expansion. Scan all exported files, including SQLite/Git config, for the fixture runtime credential and original local binding path: neither appears. Source-content secrets are outside this sanitization guarantee and are documented explicitly.

### I13: trust and error boundaries

Attempt arbitrary query filesystem paths, unsupported store versions, missing FTS5/JSON capabilities, Git hooks, forbidden adapters, and mismatched workspace configuration. Fail with actionable canonical errors before unintended activity. Validate request/body limits through streaming mocks. Distinguish transient collection failures from confirmed access revocation; both are visible, but they do not share an invented authorization meaning.

## 5. Human-reviewed explanation evaluation

### E01: fixed pilot corpus and question split

Create 120 human-reviewed questions across pinned Django, Vite, and Requests snapshots: 40 per repository. Choose concrete SHAs and evidence exports before scoring, then freeze their checksums in the evaluation manifest. Synthetic fixtures remain separate deterministic checks. Split 40 development questions and 80 held-out questions as follows:

| Primary question class | Development | Held out | Total |
| --- | ---: | ---: | ---: |
| Original rationale and constraints | 10 | 20 | 30 |
| Later applicability | 8 | 16 | 24 |
| Rejected alternatives | 6 | 12 | 18 |
| Conflicting explanations | 4 | 8 | 12 |
| Original-to-landed mapping | 2 | 4 | 6 |
| Unsupported rationale | 10 | 20 | 30 |
| Total | 40 | 80 | 120 |

Group questions by related PR/change family before splitting. A family includes follow-ups, reverts, linked disputes, and duplicated questions about the same decision. Keep each family entirely in one partition; curate replacement families to satisfy the counts. Never inspect held-out answers to tune prompts, ranking, thresholds, or extraction. Subsequent tuning requires a newly held-out set for a fresh release claim.

### E02: gold answers and required facts

For every question, record target revision, cutoff if any, exact evidence versions/spans, answerability, required reasons, constraints, alternatives, counterevidence, mapping expectation, and acceptable uncertainty. Assign each required fact weight two if omitting it could change an edit decision; otherwise weight one. Mark critical facts explicitly before running systems.

Gold unsupported questions state why available evidence does not establish rationale and distinguish undocumented intent from collection or enrichment gaps. Two reviewers independently verify gold records and adjudicate disagreements. Store the disagreement history. A plausible explanation is insufficient when its source does not establish it.

### E03: same-source comparison

Compare three arms: ordinary agent search over frozen code/Git/GitHub evidence; code plus CodeContext lexical history retrieval without enriched decisions; and the complete memory. Give every arm the same source bytes, revision, cutoff, model/version, system task, and budgets. Ordinary search must have usable file/text/history tools, rather than a deliberately inconvenient dump.

Use at most 20 tool calls, 30,000 aggregate tool-result tokens, 2,000 final-answer tokens, and five minutes per question. Record the common tokenizer and timeout policy. Run each question three times where the approved evaluation budget permits; otherwise run once and label repeatability unmeasured. Include enrichment cost separately and amortize it explicitly when reporting repeated-query cost.

### E04: cutoff and leakage controls

Apply the same evidence filter to every arm before retrieval, including GitHub edits and precomputed summaries. A later-observed version is eligible only when source timestamps establish its cutoff eligibility; missing timestamps create an uncertainty gap. Never infer an unavailable earlier body from the current body. Git ancestry remains an independent condition.

Keep answers and gold spans out of prompts, indexes, enriched claims, and model-adapter examples. Record whether the chosen model may already know public repositories; source isolation controls retrieval leakage but cannot remove pretraining knowledge. Score only evidence-backed assertions.

### E05: scoring and pilot gates

Score individual material factual assertions for citation support and compute weighted required-fact recall for every answerable question. Report macro recall per question and micro recall across facts. Separately measure evidence recall in the first ten results, counterevidence recall, correct applicability, valid abstention, and answerable-question success. Gold evidence is a list of required evidence requirements, each with accepted alternative source versions/spans. A requirement counts once if any accepted alternative is retrieved; repeated chunks from the same source version are one result. Apply recall-at-ten gates to cases satisfiable within ten distinct source versions, and report the attainable ceiling separately for larger cases. Counterevidence recall uses only questions with required counterevidence; a zero denominator is N/A, never 100%.

An answer succeeds only if it includes every critical fact, reaches at least 80% weighted fact recall, makes no unsupported material claim, and handles applicability correctly. Initial held-out pilot gates are:

- All emitted evidence anchors resolve; deterministic branch/cutoff tests have zero leakage.
- At least 95% of audited material assertions have adequate cited support.
- At least 90% macro weighted fact recall on the 60 answerable questions.
- At least 54 of 60 answerable questions meet the complete-success definition.
- At least 19 of 20 unsupported questions abstain correctly with the appropriate reason.
- At least 90% evidence recall at ten and 90% recall of required counterevidence.

These are engineering targets, not claims about arbitrary repositories. High citation precision cannot compensate for failing answerability or completeness. If complete memory performs worse than the same-source baseline on a primary quality measure, investigate and repeat with a fresh holdout before claiming benefit.

### E06: uncertainty and review procedure

Blind reviewers to the system arm, randomize answer order, and independently double-score at least 25% of answers plus every disputed failure. Report agreement and adjudication. Publish numerator/denominator counts, per-repository results, and per-question-class results rather than only aggregate percentages.

Use Wilson 95% intervals for single-run binary proportions. For quality differences and repeated runs, bootstrap whole change families with a fixed seed; repetitions of one question are not independent extra questions. Report paired differences against the baseline and their uncertainty. With only 20 unsupported held-out questions, 19 correct answers does not establish population-level 95% accuracy. Label all results as a 120-question pilot.

### E07: agent efficiency and usability

Use I06 transcripts and evaluation runs to measure useful tool selection, needless expansion, tool-call count, retrieved tokens, elapsed time, and final answer quality. Record cases where the agent ignores relevant tools, misreads an enum, follows hostile source text, or confuses coverage with certainty. Fix tool descriptions or output structure on development examples, then rerun the held-out procedure. A later issue-resolution study is separate from historical-explanation validation.

### E08: evaluation runner and transcript contract

The extraction `ModelAdapter` cannot run an agent with tools. Keep evaluation orchestration outside the installed library, in `evals/run.py`. Define an injected `EvaluationRunner` protocol with `run(request: EvalRequest, tools: EvalTools) -> EvalResult`. Load it only from an explicit evaluation-command `--runner module:factory`. No provider SDK is a required core dependency. Ship a deterministic scripted runner for harness tests; the user supplies a host/provider runner for paid evaluation.

Use strict Pydantic models with these exact fields. All timestamps are UTC; all IDs and file paths refer to the frozen evaluation bundle. No gold answer, rubric, or expected span enters the runner request.

| Model | Required fields |
| --- | --- |
| `EvalRequest` | `run_id: str`, `question_id: str`, `family_id: str`, `arm: Literal["ordinary","lexical","memory"]`, `question: str`, `target_revision: str`, `knowledge_cutoff: datetime | None`, `evidence_manifest_sha256: str`, `model_name: str`, `model_revision: str`, `model_options: dict[str,str|int|bool]`, `system_prompt: str`, `max_tool_calls: int=20`, `max_tool_result_tokens: int=30000`, `max_answer_tokens: int=2000`, `timeout_seconds: int=300`, `tokenizer_id: str`, `repetition: int`, `seed: int | None` |
| `ToolEvent` | `ordinal: int`, `name: str`, `arguments: dict`, `result: dict | None`, `error_code: str | None`, `started_at: datetime`, `duration_ms: int`, `result_tokens: int`, `result_sha256: str | None` |
| `EvalResult` | `run_id: str`, `state: Literal["complete","timeout","budget_exhausted","runner_error","protocol_invalid"]`, `final_answer: str | None`, `events: list[ToolEvent]`, `input_tokens: int | None`, `output_tokens: int | None`, `cost_usd: Decimal | None`, `provider_request_ids: list[str]`, `started_at: datetime`, `finished_at: datetime`, `error_code: str | None`, `runner_fingerprint: str` |

`EvalTools` exposes `list_tools() -> list[ToolSpec]` and `call(name: str, arguments: dict) -> dict`. `ToolSpec` has `name`, `description`, and `input_schema`. The harness owns dispatch, eligibility, shared tokenizer accounting, timestamps, events, and budget checks; the runner cannot supply authoritative event counts. `call` refuses a 21st call, a call after deadline, or a result larger than the remaining token allowance. The tool result includes a bounded budget error when refused. A runner that bypasses this dispatcher or exceeds answer/time limits produces `protocol_invalid`; do not score its answer as compliant success. Test each refusal boundary with the scripted runner. The external runner must support cancellation; if it cannot, stop accepting events at the deadline and record any later incurred provider cost separately.

Every arm receives the same question/system instructions, model identity/options, immutable evidence manifest, and cutoff filter. The common baseline tool set is `list_files`, `read_file`, `search_text`, `git_log`, `git_show`, `list_discussions`, and `read_discussion`. Implement these against frozen bytes and owned Git objects, with bounded results and exact source handles. Do not execute shell commands generated by the runner. The lexical arm adds CodeContext `search/get/context` with enriched claims/summaries disabled. The memory arm adds the full tools. Giving the latter arms additional retrieval methods is the intervention; their underlying available source evidence remains identical. The ordinary arm can still locate the exact gold sources through its baseline tools.

Write one request/result pair per line to `runs.jsonl` in a local evaluation output directory, plus its tool-result blobs and a SHA-256 manifest. Keep prompts/evidence private by default and commit only sanitized reports. Runtime credentials never enter model options or transcripts. Validate imported host transcripts against the same schema and counts before scoring; unsupported imports are rejected, not silently converted. A model that cannot expose usage uses nulls and cannot support a precise cost comparison.

Store gold answers separately in `questions.jsonl` with `id`, `family_id`, `repository`, `revision`, `cutoff`, `partition`, `class`, `question`, `answerability`, `required_facts[]`, `evidence_requirements[]`, `expected_applicability`, and `review_status`. Each required fact has `id`, `text`, `weight` (one or two), `critical`, and accepted evidence-requirement IDs. Each evidence requirement has `id`, `role` (`support`, `counterevidence`, `lifecycle`, or `applicability`), and alternative immutable version/anchor/span sets. Record reviewer IDs and adjudication separately; unresolved gold is excluded from held-out scoring and replaced before the corpus is frozen. Validate exact partition counts and no cross-partition family overlap automatically.

Persist reviewer scoring in `scores.jsonl`: `run_id`, `reviewer_id`, `assertions[]` (text, material, supported, cited evidence IDs), `fact_scores[]` (fact ID, present), `applicability_correct`, `abstention_correct`, `critical_error`, and `notes`. Recompute aggregates from these records; never hand-edit aggregate report percentages. Unit-test perfect, incomplete-but-precise, unsupported, missing-counterevidence, and zero-denominator cases before any paid run.

## 6. Performance, load, and cost

Use an eight-core machine with 16 GB RAM and local SSD for the release load report. Record exact CPU, usable memory, disk, filesystem, OS, power mode, and background load. Freeze corpus hashes, cache state, query selection, and budgets. Separate mocked local measurements from live GitHub wall time; API latency and quota availability are external inputs.

| ID | Workload and action | Measurements and acceptance |
| --- | --- | --- |
| P01 | Run F12 regression at 0, 100, 1,000, and 5,000 commits. | Parsed-symbol rows equal independent unique blob/extractor expectations. Unchanged symbols never gain per-commit copies. Report database/Git/FTS bytes separately; fail the structural invariant even if compression hides it. |
| P02 | Start a fresh process against a prepared store; repeat 30 times. | Report process startup, store-open, first `search`, first `get`, and first `context` latency. Label this process-cold; do not claim OS-cache-cold unless verified. |
| P03 | Warm the fixed corpus, then run 1,000 stratified reads. | Targets: p95 `get` below 100 ms, `search` below 300 ms, and local `context` below one second. Report p50/p95/p99, errors, truncation, and output size; caching one repeated query does not satisfy the test. |
| P04 | Initialize F12 load scope: approximately 1M current lines and 100k commits. | Record total time, peak RSS, disk growth, parse counts, HTTP requests, and checkpoint progress. Complete within machine memory with no whole-history in-memory materialization or out-of-memory failure. Initial indexing duration is measured before setting a release time target. |
| P05 | Begin an empty store with a selected target, then progressively collect/enrich. | Measure time to first searchable code, first inspectable historical episode, first cited rationale, and full accessible-scope completion separately. The first useful result must precede full backfill on the designed workload, with accurate partial coverage. |
| P06 | Add one commit, one edited discussion, and one unrelated issue; synchronize again. | Report incremental time, API calls, bytes, and model usage. Unchanged blobs are not reparsed and unchanged episodes are not reenriched; dependencies invalidate only affected derived records. |
| P07 | Interrupt collection/enrichment at checkpoints; run one writer and four readers. | Report resume time, duplicated requests, lock waits, latency distribution, and consistency failures. Keep published memory usable; bound reads and report `READ_BUDGET_EXHAUSTED` rather than silently unlimited expansion. |
| P08 | Enrich representative small, medium, and large episodes under explicit budgets. | Report input/output tokens, calls, retries, cache hits, estimated versus billed cost when available, and coverage gained. Budget exhaustion preserves completed work. Reads record zero model calls and zero provider spend. |

For P03/P07, include large discussions, exact target queries, broad queries, missing targets, conflicting histories, and maximum supported output budgets. Also report deadline-exhaustion rates: fast truncated responses alone do not establish useful retrieval performance. For live onboarding, publish both wall-clock time and active processing time, including rate-limit delays.

## 7. Release evidence

Require all deterministic checks for implemented release capabilities, I06's real client transcript, a frozen pilot evaluation report, and the specified load report. A failed performance target may justify a documented release-limit change only after measuring the cause and updating the contract; it must not be silently relabeled as passed.

Publish supported languages, tested platforms, source/extraction coverage, known gaps, benchmark machine, pilot sample counts, and measured costs. Preserve raw anonymized results and scripts needed to reproduce them. Keep private evidence and credentials out of committed artifacts.

No library tests, real-client evaluations, or benchmarks have run under this specification. Document/schema checks are recorded in the master plan separately. The principal unresolved product risk is a well-cited but incomplete explanation, or a historical rationale presented as a constraint on changed code.
