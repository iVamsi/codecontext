# Exact public result models

Status: specification only. Implement these declarations in `models.py` at T02, adding the behavioral validators listed below. Request fields and defaults are fixed in [shared contracts](00-contracts.md); model-adapter and extraction shapes are fixed in [enrichment](03-enrichment-and-retrieval.md). These result declarations remove the need to invent nested response fields. They do not require implementing later capabilities at T02.

Use Pydantic v2, `extra="forbid"`, immutable model instances, and JSON-mode serialization. Python construction uses actual enum/literal values, aware datetimes, Paths, and Decimals; validate JSON at transport boundaries with JSON-mode validation. Never enable permissive coercion of arbitrary integers into identifiers or booleans. Serialize UTC datetimes with `Z`, Decimal amounts as strings, optional fields as null, and arrays as arrays. All fields below are required unless a default is explicitly shown. Null means unknown/inapplicable. Empty means known empty within the reported scope.

## 1. Shared primitives

The following code blocks concatenate into one syntactically valid model module. They are a design reference, not application code already present in the repository. Implement additional cross-field validators from section 7 before accepting these as completed models.

```python
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field

class PublicModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

NonNegative = Annotated[int, Field(ge=0)]
Positive = Annotated[int, Field(ge=1)]
RecordID = Annotated[str, Field(min_length=1, max_length=100)]
ObjectID = Annotated[str, Field(min_length=40, max_length=64)]
ShortText = Annotated[str, Field(max_length=1000)]
UTCText = Annotated[str, Field(max_length=40)]
Cursor = Annotated[str, Field(max_length=8192)]
Support = Literal["explicit", "inferred", "unknown"]
Lifecycle = Literal["proposed", "adopted", "rejected", "disputed", "superseded", "unknown"]
Mapping = Literal["exact", "candidate", "unmapped"]
Applicability = Literal["corroborated", "inferred", "historical_only", "unknown"]
DerivedState = Literal["ready", "stale", "retracted"]
RecordKind = Literal["symbol", "change", "hunk", "source", "source_version", "evidence", "claim", "summary", "gap"]

class PathRef(PublicModel):
    path: Annotated[str, Field(max_length=32768)]  # exact UTF-8 or escaped display
    path_bytes_b64: Annotated[str, Field(max_length=65536)] | None

class ByteSpan(PublicModel):
    byte_start: NonNegative
    byte_end: NonNegative
    line_start: Positive | None
    line_end: Positive | None

class Anchor(PublicModel):
    kind: Literal["discussion", "code", "commit", "hunk"]
    source_version_id: RecordID | None
    git_oid: ObjectID | None
    revision: ObjectID | None
    path: PathRef | None
    span: ByteSpan
    underlying_record_id: RecordID | None
    original_review_commit: ObjectID | None

class Attribution(PublicModel):
    provider: Literal["github", "git", "unknown"]
    actor_id: Annotated[str, Field(max_length=200)] | None
    display_name: Annotated[str, Field(max_length=400)] | None
    source_url: Annotated[str, Field(max_length=4096)] | None

class SourceTimes(PublicModel):
    authored_at: UTCText | None
    updated_at: UTCText | None
    first_observed_at: UTCText
    last_observed_at: UTCText

class EvidenceRef(PublicModel):
    id: RecordID
    anchor: Anchor
    content_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

class EvidenceView(PublicModel):
    ref: EvidenceRef
    attribution: Attribution
    times: SourceTimes | None
    excerpt: Annotated[str, Field(max_length=8192)] | None
    availability: Literal["available", "unverified", "inaccessible", "missing_object", "oversized", "binary"]
    historical_scope: Literal["target_ancestor", "nonancestor", "discussion", "unknown"]

class Gap(PublicModel):
    id: RecordID | None
    code: Annotated[str, Field(min_length=1, max_length=100)]
    scope: Annotated[str, Field(max_length=1000)]
    message: ShortText
    retryable: bool
    record_id: RecordID | None

class GapCount(PublicModel):
    code: Annotated[str, Field(max_length=100)]
    count: NonNegative

class CoveragePart(PublicModel):
    state: Literal["not_started", "partial", "complete", "blocked", "unavailable"]
    completed: NonNegative
    known_total: NonNegative | None
    unit: Literal["commits", "records", "blobs", "segments"]
    last_success_at: UTCText | None
    reconciled_at: UTCText | None
    gap_count: NonNegative

class Coverage(PublicModel):
    git: CoveragePart
    github: CoveragePart
    structure: CoveragePart
    enrichment: CoveragePart
    selected_refs: Annotated[list[str], Field(max_length=20)]
    selected_ref_count: NonNegative
    omitted_ref_count: NonNegative
    excluded_file_count: NonNegative
    parser_fingerprints: Annotated[list[str], Field(max_length=10)]
    active_gap_count: NonNegative
    gap_counts: Annotated[list[GapCount], Field(max_length=100)]
    first_gap_id: RecordID | None

class OmittedCounts(PublicModel):
    code: NonNegative = 0
    decisions: NonNegative = 0
    evidence: NonNegative = 0
    relations: NonNegative = 0
    gaps: NonNegative = 0
    hits: NonNegative = 0
    statements: NonNegative = 0
    content_bytes: NonNegative = 0

class Budget(PublicModel):
    requested_tokens: Positive
    used_tokens: NonNegative
    estimator: Annotated[str, Field(min_length=1, max_length=200)]
    count_kind: Literal["estimated", "tokenizer"]
    truncated: bool
    omitted_counts: OmittedCounts

class Expand(PublicModel):
    id: RecordID
    reason: Annotated[str, Field(max_length=300)]

class Envelope(PublicModel):
    schema_version: Literal[1] = 1
    repository_id: Annotated[str, Field(min_length=36, max_length=36)]
    revision: ObjectID | None
    generation: NonNegative
    access_epoch: NonNegative
    coverage: Coverage
    gaps: Annotated[list[Gap], Field(max_length=20)]

class RelationView(PublicModel):
    id: RecordID
    from_kind: str
    from_id: str
    to_kind: str
    to_id: str
    relation_kind: Literal["contains", "imports_candidate", "calls_candidate", "changes", "associated_pr", "references_issue", "review_anchor", "landed_mapping", "targets", "supports", "contradicts", "supersedes", "corroborates"]
    method: Annotated[str, Field(max_length=200)]
    mapping: Mapping
    evidence_id: RecordID | None
    state: Literal["active", "stale", "retracted"]
    explanation: ShortText | None
```

`ObjectID` must additionally validate full lowercase hexadecimal width against the store's actual format, not merely accept any 40–64-character string. `UTCText` must parse as UTC ISO 8601 ending in `Z`. `RecordID` validates the expected public prefix and 64-character lowercase SHA-256 suffix. Internal relation IDs use `rel_` and are not independently get-expandable; the endpoint IDs may be internal Git/episode IDs. Return `Expand` only for public get-expandable IDs.

Discussion anchors require a source version and no Git object. Git anchors require an owned object or an explicit missing-object gap. A commit-message span is measured against the owned raw commit object's bytes, accounting for the header length; a hunk span is measured against stored patch bytes. Code anchors require path, revision, and blob object ID. Empty spans are allowed only when the represented insertion/deletion location is empty. No view may fabricate an excerpt when exact bytes cannot be decoded.

Coverage uses aggregate counts. `selected_refs` is a bounded preview; full ref observations remain internal/export metadata. Parser fingerprints are the active grammar/query-pack fingerprints, not every historical fingerprint. `gap_counts` enumerates the finite implementation gap-code registry, rather than arbitrary provider errors. Add new gap codes to that registry and tests before emission. Request-specific gaps have `id=null` because read operations cannot persist a gap. Persisted coverage gaps have `gap_` IDs and support expansion.

## 2. Search and context results

```python
class SearchHit(PublicModel):
    id: RecordID
    kind: RecordKind
    title: Annotated[str, Field(max_length=1000)]
    snippet: Annotated[str, Field(max_length=320)]
    rank: Positive
    match: Literal["exact_id", "exact_path", "exact_symbol", "lexical"]
    path: PathRef | None
    evidence_id: RecordID | None
    state: DerivedState | None

class SearchResult(Envelope):
    query: Annotated[str, Field(max_length=4000)]
    hits: Annotated[list[SearchHit], Field(max_length=100)]
    next_cursor: Cursor | None
    truncated: bool
    budget: Budget
    expand: Annotated[list[Expand], Field(max_length=20)]

class ResolvedTarget(PublicModel):
    kind: Literal["symbol", "path"]
    symbol_id: RecordID | None
    path: PathRef
    revision: ObjectID
    blob_oid: ObjectID
    line: Positive | None

class WorkspaceState(PublicModel):
    checked: bool
    observed_at: UTCText | None
    head_revision: ObjectID | None
    head_matches_index: bool | None
    repository_diverged: bool | None
    target_state: Literal["unchanged", "staged_change", "unstaged_change", "staged_and_unstaged", "deleted", "untracked", "unavailable", "not_checked"]
    index_blob_oid: ObjectID | None
    staged_blob_oid: ObjectID | None
    working_content_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None

class CodeView(PublicModel):
    evidence_id: RecordID
    symbol_id: RecordID | None
    anchor: Anchor
    language: Annotated[str, Field(max_length=40)] | None
    signature: Annotated[str, Field(max_length=4000)] | None
    text: Annotated[str, Field(max_length=16384)]
    mapping: Mapping

class DecisionView(PublicModel):
    id: RecordID
    statement: Annotated[str, Field(min_length=1, max_length=1000)]
    rationale: Annotated[str, Field(min_length=1, max_length=3000)] | None
    alternatives: Annotated[list[ShortText], Field(max_length=10)]
    constraints: Annotated[list[ShortText], Field(max_length=10)]
    support: Support
    lifecycle: Lifecycle
    state: DerivedState
    mapping: Mapping
    applicability: Applicability
    attributions: Annotated[list[Attribution], Field(max_length=20)]
    evidence_refs: Annotated[list[EvidenceRef], Field(max_length=40)]
    counterevidence_refs: Annotated[list[EvidenceRef], Field(max_length=20)]
    applicability_evidence_refs: Annotated[list[EvidenceRef], Field(max_length=10)]

class SummaryStatement(PublicModel):
    text: Annotated[str, Field(min_length=1, max_length=1000)]
    evidence_refs: Annotated[list[EvidenceRef], Field(min_length=1, max_length=10)]

class Overview(PublicModel):
    kind: Literal["stored_summary", "structural"]
    summary_id: RecordID | None
    text: Annotated[str, Field(max_length=16000)]
    statements: Annotated[list[SummaryStatement], Field(max_length=40)]
    file_count: NonNegative
    group_count: NonNegative
    included_file_count: NonNegative
    included_group_count: NonNegative
    included_claim_count: NonNegative
    omitted_claim_count: NonNegative
    partial: bool

class ContextPack(Envelope):
    query: Annotated[str, Field(max_length=4000)]
    target: ResolvedTarget | None
    workspace: WorkspaceState
    overview: Overview | None
    code: Annotated[list[CodeView], Field(max_length=20)]
    decisions: Annotated[list[DecisionView], Field(max_length=20)]
    evidence: Annotated[list[EvidenceView], Field(max_length=100)]
    rationale_status: Literal["supported", "enrichment_pending", "evidence_incomplete", "undocumented_in_available_evidence"]
    budget: Budget
    expand: Annotated[list[Expand], Field(max_length=20)]
```

No numeric confidence/probability is emitted. Search ordering is the specified rank-fusion algorithm; `rank` is the one-based position in its bounded shortlist, continuing across pages. Exact snippets are search previews, never standalone rationale assertions. For unsupported/oversized signatures, truncate only the preview and expose the full raw code evidence handle.

For untargeted queries `target=null`. A path target absent from the prepared snapshot returns `RECORD_NOT_FOUND`, except a configured workspace's verified untracked file may return `target=null` plus `UNTRACKED_TARGET` and suggestions. A symbol shared by multiple active paths is ambiguous unless one occurrence is uniquely resolved; return `AMBIGUOUS_TARGET`, rather than guessing. `repository_diverged` is null when a bounded workspace check cannot establish it; a clean target alone never establishes a clean repository.

For structural overviews, `summary_id=null`, statements are empty, and text contains only deterministic counts/names/revision information. For stored summaries, text is the newline join of included statements, with evidence included in the pack. Remove a whole statement and update omission counts if its citation bundle cannot fit. `evidence_refs` on decisions includes supporting and lifecycle-role citations; durable `claim_evidence` retains their distinct roles. Applicability refs remain separately labeled.

## 3. Expandable record union

```python
class SymbolRecord(PublicModel):
    kind: Literal["symbol"] = "symbol"
    id: RecordID
    blob_oid: ObjectID
    language: str
    extractor_fingerprint: str
    name: str
    qualified_name: str
    symbol_kind: str
    declaration_span: ByteSpan
    body_span: ByteSpan | None
    parent_symbol_id: RecordID | None
    signature: Annotated[str, Field(max_length=4000)]

class ChangeRecord(PublicModel):
    kind: Literal["change"] = "change"
    id: RecordID
    commit_oid: ObjectID
    parent_index: Annotated[int, Field(ge=-1)]
    parent_oid: ObjectID | None
    old_path: PathRef | None
    new_path: PathRef | None
    old_blob_oid: ObjectID | None
    new_blob_oid: ObjectID | None
    old_mode: NonNegative | None
    new_mode: NonNegative | None
    change_kind: Literal["added", "modified", "deleted", "renamed", "copied", "type_changed", "unmerged", "unknown"]
    rename_score: Annotated[int, Field(ge=0, le=100)] | None
    hunks_state: Literal["pending", "complete", "partial", "unavailable"]

class HunkRecord(PublicModel):
    kind: Literal["hunk"] = "hunk"
    id: RecordID
    change_id: RecordID
    ordinal: NonNegative
    old_start: NonNegative
    old_count: NonNegative
    new_start: NonNegative
    new_count: NonNegative

class SourceRecord(PublicModel):
    kind: Literal["source"] = "source"
    id: RecordID
    source_kind: Literal["issue", "pull_request", "issue_comment", "review", "review_comment", "timeline_event"]
    provider: Literal["github"]
    repository_provider_id: str
    provider_id: str
    parent_source_id: RecordID | None
    selected_version_id: RecordID | None
    availability: Literal["available", "unverified", "inaccessible", "redacted", "deleted"]
    last_seen_at: UTCText
    last_reconciled_at: UTCText | None

class SourceVersionRecord(PublicModel):
    kind: Literal["source_version"] = "source_version"
    id: RecordID
    source_id: RecordID
    title: Annotated[str, Field(max_length=4000)] | None
    attribution: Attribution
    times: SourceTimes
    content_sha256: str
    body_bytes: NonNegative
    current: bool

class EvidenceRecord(PublicModel):
    kind: Literal["evidence"] = "evidence"
    id: RecordID
    evidence: EvidenceView

class ClaimRecord(PublicModel):
    kind: Literal["claim"] = "claim"
    id: RecordID
    episode_id: str
    state: DerivedState
    model_run_id: str
    created_at: UTCText
    decision: DecisionView | None

class SummaryRecord(PublicModel):
    kind: Literal["summary"] = "summary"
    id: RecordID
    scope_kind: Literal["package", "directory", "root_files", "repository"]
    scope_id: str
    revision: ObjectID
    state: DerivedState
    input_fingerprint: str
    partial: bool
    statements: Annotated[list[SummaryStatement], Field(max_length=40)]

class GapRecord(PublicModel):
    kind: Literal["gap"] = "gap"
    id: RecordID
    gap: Gap
    active: bool
    first_seen_at: UTCText
    last_seen_at: UTCText
    next_gap_id: RecordID | None

Record = Annotated[
    SymbolRecord | ChangeRecord | HunkRecord | SourceRecord | SourceVersionRecord |
    EvidenceRecord | ClaimRecord | SummaryRecord | GapRecord,
    Field(discriminator="kind")
]

class ContentSlice(PublicModel):
    field: Literal["body", "code", "patch", "commit_message"]
    span: ByteSpan
    text: Annotated[str, Field(max_length=32768)]
    total_bytes: NonNegative

class RecordPage(Envelope):
    record: Record
    content: Annotated[list[ContentSlice], Field(max_length=20)]
    evidence: Annotated[list[EvidenceView], Field(max_length=100)]
    relations: Annotated[list[RelationView], Field(max_length=100)]
    next_cursor: Cursor | None
    budget: Budget
    expand: Annotated[list[Expand], Field(max_length=20)]
```

Dispatch by prefix, then validate ownership/access/cutoff before projecting a record. `src_` returns metadata and its selected eligible version handle, not a duplicated body. `ver_` returns metadata plus body slices. `sym_` returns blob-owned metadata plus source slices; occurrence relations are bounded and preserve path ambiguity. `chg_` returns metadata and hunk/evidence relationships; `hunk_` returns metadata plus patch slices. `ev_` returns its exact anchor and bounded original content. Commit messages use `ev_` handles; no separate `commit_` public prefix is needed.

Claim expansion has no implicit new target: its mapping/applicability are relative to its stored original targets, with historical-only applicability unless a requested context establishes otherwise. Include `decision` only when its complete required evidence/counterevidence bundle fits. If it does not fit even on an otherwise empty page, return bounded `INVALID_REQUEST` with the minimum required budget; do not return an endless cursor that never admits it. Summary pagination uses whole statement bundles. Source/code/patch pagination uses UTF-8-safe byte slices, then relations. `next_cursor=null` means all remaining content and relations eligible for that record have been served; omitted inaccessible content remains an explicit gap, not a promise of another page.

The root metadata of a record is not repeated as unbounded raw provider JSON. Raw payload is retained internally for audit/version normalization; the public evidence API serves the exact body/anchors. Structural metadata longer than its display cap gets a gap and an original-source handle. Do not truncate identity bytes or canonical stored fields. `get(gap_)` returns bounded status and the next active gap by `(scope,code,id)`, allowing full navigation from `coverage.first_gap_id`. Generation changes require restarting that walk.

## 4. Status and mutation reports

```python
class Capabilities(PublicModel):
    syntax_languages: Annotated[list[str], Field(max_length=10)]
    text_fallback: bool
    git_history: bool
    github_history: bool
    offline_git_objects: bool
    enriched_claims_available: bool

class QueueCounts(PublicModel):
    pending: NonNegative
    running: NonNegative
    complete: NonNegative
    retryable_error: NonNegative
    blocked: NonNegative

class Status(Envelope):
    capabilities: Capabilities
    access_state: Literal["allowed", "blocked"]
    read_only: bool
    github_repository: str | None
    object_format: Literal["sha1", "sha256"]
    jobs: QueueCounts
    snapshot_file_count: NonNegative
    commit_count: NonNegative
    source_count: NonNegative
    claim_count: NonNegative
    summary_count: NonNegative
    budget: Budget
    expand: Annotated[list[Expand], Field(max_length=20)]

class SyncReport(Envelope):
    state: Literal["complete", "partial", "blocked"]
    started_at: UTCText
    finished_at: UTCText
    elapsed_ms: NonNegative
    requests_attempted: NonNegative
    commits_added: NonNegative
    blobs_parsed: NonNegative
    sources_added: NonNegative
    versions_added: NonNegative
    cache_hits: NonNegative
    jobs: QueueCounts
    stop_reason: Literal["scope_complete", "request_budget", "time_budget", "offline", "source_blocked", "cancelled"]

class Usage(PublicModel):
    calls: NonNegative
    estimated_input_tokens: NonNegative
    reserved_output_tokens: NonNegative
    reported_input_tokens: NonNegative | None
    reported_output_tokens: NonNegative | None
    uncertain_input_tokens: NonNegative
    uncertain_output_tokens: NonNegative
    prior_uncertain_input_tokens: NonNegative
    prior_uncertain_output_tokens: NonNegative
    configured_price_cost_usd: Annotated[str, Field(max_length=100)] | None
    cost_is_estimated: bool
    overrun: bool

class ItemFailure(PublicModel):
    item_key: Annotated[str, Field(max_length=300)]
    code: Annotated[str, Field(max_length=100)]
    message: ShortText
    retryable: bool

class EnrichReport(Envelope):
    state: Literal["complete", "partial", "blocked"]
    started_at: UTCText
    finished_at: UTCText
    completed_items: NonNegative
    pending_items: NonNegative
    failed_items: NonNegative
    cache_hit_items: NonNegative
    claims_added: NonNegative
    summaries_added: NonNegative
    usage: Usage
    failures: Annotated[list[ItemFailure], Field(max_length=20)]
    omitted_failure_count: NonNegative
    stop_reason: Literal["scope_complete", "budget", "model_failure", "usage_unknown", "usage_overrun", "cancelled"]

class ExportReport(PublicModel):
    schema_version: Literal[1] = 1
    repository_id: str
    revision: ObjectID | None
    generation: NonNegative
    access_epoch: NonNegative
    destination: str
    include_git_objects: bool
    file_count: NonNegative
    total_bytes: NonNegative
    manifest_sha256: str
    warnings: Annotated[list[ShortText], Field(max_length=20)]

class PurgeReport(PublicModel):
    schema_version: Literal[1] = 1
    repository_id: str
    mode: Literal["source", "entire_store"]
    source_id: RecordID | None
    generation: NonNegative
    access_epoch: NonNegative
    removed_source_versions: NonNegative
    removed_claims: NonNegative
    removed_summaries: NonNegative
    cleanup_pending: bool
    closed: bool
```

Status has a fixed 6,000-token ceiling and at most ten detailed gaps despite the shared envelope's larger maximum. Search has a fixed 12,000-token ceiling. Report aggregate counts even when details are omitted. Do not emit an unbounded expansion handle for every omitted item. The first persisted gap handle and generation-bound navigation provide access to the rest.

Mutation reports are bounded to 12,000 estimator tokens using the same detail-dropping rules, but are reports rather than read-result packs: their aggregate counters remain mandatory and no `budget` object is required. Return at most 20 failures/gaps. Persist full item state internally. Export/Purge reports are small, never include source text, and can be returned after the store is closed. A blocked store returns `ACCESS_BLOCKED` from all read surfaces; its owner can inspect a sanitized failure and perform purge, rather than retrieve private status fields.

## 5. Error object and transport

Use this exact JSON shape for expected errors:

```json
{
  "error": {
    "code": "RECORD_NOT_ELIGIBLE",
    "message": "This record is outside the requested knowledge cutoff.",
    "retryable": false,
    "details": {"id": "ver_example", "reason": "source_edited_after_cutoff"}
  }
}
```

The example ID is illustrative, not a valid fixture ID. Restrict error details to validated scalar/list values with depth at most three and a total serialized size of 4 KiB. Cap the complete error object at 2,000 estimator tokens. Never include provider response bodies, local credentials, raw prompts, or ineligible source titles/snippets. Expected error codes are enumerated in contracts §6. Unknown exceptions become sanitized `IO_ERROR` at the SDK boundary with a local correlation ID; debug diagnostics must also exclude source bodies/credentials by default. Keyboard interrupt and process cancellation are not swallowed as IO failures.

## 6. Identifier recipes

All suffixes use the canonical JSON hash from contracts §4. Include `repository_id` for records whose identity is repository scoped. Define these identity inputs exactly; JSON key names below are part of the recipe:

| Prefix | Canonical identity input |
| --- | --- |
| `sym_` | `blob_oid, language, extractor_fingerprint, byte_start, byte_end, kind, qualified_name` |
| `chg_` | `repository_id, commit_oid, parent_index, old_path_b64, new_path_b64, old_blob_oid, new_blob_oid, change_kind` |
| `hunk_` | `change_id, ordinal, old_start, old_count, new_start, new_count, patch_sha256` |
| `src_` | `provider, repository_provider_id, kind, provider_id` |
| `ver_` | `source_id, content_sha256` |
| `ev_` | `repository_id, kind, source_version_id, git_oid, anchor, content_sha256` |
| `claim_` | `repository_id, episode_id, request_fingerprint, normalized_claim, sorted_evidence_roles` |
| `sum_` | `repository_id, scope_kind, scope_id, revision, input_fingerprint, model_fingerprint, prompt_version` |
| `gap_` | `repository_id, code, scope, record_id` |

Use explicit null for absent identity fields. Canonical path identity always uses base64 of raw bytes, even when valid UTF-8. `normalized_claim` preserves case and punctuation but normalizes line endings and outer whitespace; sort citation/role identities and deduplicate exact repeated entries. Do not use nondeterministic model-run UUIDs in content IDs. Store the full identity input in existing typed fields/validated JSON; no second identity registry is needed.

Internal IDs use the same hash helper with distinct type tags: extraction `(blob,language,fingerprint)`, episode `(repository,pr_source_id)` or `(repository,commit_oid)`, relation `(from,to,kind,method,anchor details)`, segment `(episode,input_fingerprint,ordinal,slot anchors)`, job `(kind,owner key)`, and summary scope `(repository,scope_kind,path_b64)`. Model-run/request IDs are random UUIDs because two dispatched attempts can incur separate costs even when their request fingerprint matches. Public IDs identify records, not permission grants.

## 7. Required validation matrix

- Validate enums, lengths, expected prefixes, full object format, ISO times, and nonnegative counters before persistence or output. Give internal free-string fields above a 1,000-character cap unless a stated larger bound is needed; source sizes are handled by body pagination, not unlimited metadata.
- `PathRef.path_bytes_b64=null` only for exact UTF-8 paths. Otherwise the base64 value is authoritative and the display string escapes invalid bytes unambiguously. Target input differs: it requires exactly one path representation.
- Byte-end must be at least byte-start; line bounds are both null or both positive/in-order. Body spans must lie within declaration spans. Hash an excerpt's exact bytes and compare the evidence locator before returning it.
- All decision, overview, and summary citations must resolve to available included evidence or an explicit accessible expansion. A decision's required support/counterevidence must be in the same pack; an expansion alone cannot justify emitting the positive decision.
- `applicability="inferred"` is reserved; reject it in v1 output. A dirty target cannot be corroborated. `rationale_status="supported"` follows the explicit-evidence/target mapping rule from contracts, not merely a nonempty decisions array.
- `used_tokens` covers the entire canonical serialized pack, including itself. Iterate count/serialization to stability. `truncated` agrees with omissions; `SearchResult.truncated` equals `budget.truncated`. Limits apply after JSON encoding, not just text character counts.
- Counts have declared units; unavailable totals are null, never zero. Report source completeness separately from enrichment and interpretation quality. Read responses never mutate coverage just to persist a request-specific gap.
- IDs in `expand` resolve through `get`; internal episode/job/model-run/relation IDs do not. Resolve public prefixes through a closed dispatch map. A guessed ID is never a filesystem path or SQL fragment.
- Round-trip representative responses through JSON and Pydantic. Snapshot SDK/CLI/MCP schemas; verify discriminated unions, nullable values, fixed budgets, and forbidden extra fields. Test source/gap selectors as mutable, cursors as generation-bound, and source/evidence content as cutoff/access controlled.

No executable public model module has been created or tested yet. These declarations and validators are the required T02 implementation contract.

## 8. Gap-code registry

Use the following closed v1 registry; messages supply bounded context. Persist collection/processing gaps when their underlying state is published. Read-specific omissions/divergence use transient `id=null` gaps. Error codes may also appear as gap codes when a batch made useful partial progress. Never persist a full error body as a gap message.

| Codes | Trigger and scope |
| --- | --- |
| `GIT_HISTORY_PENDING`, `GITHUB_BACKFILL_PENDING`, `STRUCTURE_PENDING`, `ENRICHMENT_PENDING` | Work remains within the corresponding declared stream/episode |
| `SHALLOW_HISTORY`, `MISSING_GIT_OBJECT`, `MISSING_PARENT` | Declared history or original review objects are absent |
| `SOURCE_UNAVAILABLE`, `SOURCE_UNVERIFIED`, `SOURCE_BODY_TOO_LARGE`, `HTTP_BODY_TOO_LARGE` | Provider evidence cannot be fetched/verified/stored within limits |
| `API_RESULT_CAP`, `RECONCILIATION_PENDING`, `HISTORICAL_SEARCH_PARTIAL` | A capped/incomplete pass or early-slice index prevents a completeness claim |
| `AUTH_REQUIRED`, `ACCESS_BLOCKED`, `RATE_LIMITED`, `IO_ERROR` | Canonical operational error affecting one batch scope |
| `UNSUPPORTED_SYNTAX`, `PARSE_ERROR`, `BLOB_TOO_LARGE`, `BINARY_CONTENT`, `TEXT_DECODE_FAILED` | Structure/text capability is limited for this blob |
| `SUBMODULE_NOT_FETCHED`, `LFS_CONTENT_NOT_FETCHED`, `SYMLINK_NOT_FOLLOWED` | The tracked reference is retained; referenced external content is not acquired |
| `MAPPING_AMBIGUOUS`, `MAPPING_UNAVAILABLE`, `REVIEW_COMPARISON_MISSING` | Original/target correspondence cannot be established exactly |
| `SOURCE_TIME_UNKNOWN`, `HISTORICAL_VERSION_UNAVAILABLE`, `DERIVED_ENRICHMENT_PENDING` | Cutoff eligibility or current derivation validity is incomplete |
| `MODEL_FAILED`, `MODEL_INVALID_OUTPUT`, `BUDGET_EXHAUSTED`, `MODEL_USAGE_UNKNOWN`, `MODEL_USAGE_OVERRUN` | Enrichment could not complete the current work item under its budget |
| `EPISODE_INSPECTION_PENDING`, `SUMMARY_INPUTS_OMITTED` | Some required slots or summary inputs were not inspected/presented |
| `READ_BUDGET_EXHAUSTED`, `OUTPUT_TRUNCATED`, `QUERY_TRUNCATED`, `CONFLICTING_EVIDENCE_OMITTED` | A read's time/size limit omitted content; positive claim bundles obey the packing rules |
| `WORKSPACE_TARGET_DIFFERS`, `WORKSPACE_HEAD_DIFFERS`, `WORKSPACE_UNAVAILABLE`, `WORKSPACE_UNMERGED`, `WORKSPACE_UNCHECKED`, `UNTRACKED_TARGET` | Local workspace observation differs or is unavailable; exact indexed anchors remain unchanged |
| `EXPORT_GIT_OBJECTS_OMITTED`, `CLEANUP_PENDING` | An archive omits objects or logical purge completed before physical cleanup |

Each gap identifies the narrowest bounded scope such as `git:refs`, `github:issue:42:comments`, `blob:<oid>`, `episode:<id>`, or `workspace:target`. Store path-sensitive scope bytes in canonical identity fields; use a bounded escaped display in messages. `retryable` means repeating the relevant explicit operation may help; it never instructs the read server to perform that operation. Resolved persistent gaps stay as inactive status records for audit, without purged source text. Status counts active gaps only.
