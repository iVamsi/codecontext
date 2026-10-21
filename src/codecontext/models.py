from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    ValidationInfo,
    model_validator,
)

from codecontext.identity import (
    HEX_64_RE,
    HEX_OID_RE,
    PUBLIC_ID_PREFIXES,
    decode_path_b64,
    format_iso_z,
    parse_iso_utc,
    validate_github_repo_name,
    validate_relative_repo_path_bytes,
)

GAP_CODE_REGISTRY: frozenset[str] = frozenset(
    {
        "GIT_HISTORY_PENDING",
        "GITHUB_BACKFILL_PENDING",
        "STRUCTURE_PENDING",
        "ENRICHMENT_PENDING",
        "SHALLOW_HISTORY",
        "MISSING_GIT_OBJECT",
        "MISSING_PARENT",
        "SOURCE_UNAVAILABLE",
        "SOURCE_UNVERIFIED",
        "SOURCE_BODY_TOO_LARGE",
        "HTTP_BODY_TOO_LARGE",
        "API_RESULT_CAP",
        "RECONCILIATION_PENDING",
        "HISTORICAL_SEARCH_PARTIAL",
        "AUTH_REQUIRED",
        "ACCESS_BLOCKED",
        "RATE_LIMITED",
        "IO_ERROR",
        "UNSUPPORTED_SYNTAX",
        "PARSE_ERROR",
        "BLOB_TOO_LARGE",
        "BINARY_CONTENT",
        "TEXT_DECODE_FAILED",
        "SUBMODULE_NOT_FETCHED",
        "LFS_CONTENT_NOT_FETCHED",
        "SYMLINK_NOT_FOLLOWED",
        "MAPPING_AMBIGUOUS",
        "MAPPING_UNAVAILABLE",
        "REVIEW_COMPARISON_MISSING",
        "SOURCE_TIME_UNKNOWN",
        "HISTORICAL_VERSION_UNAVAILABLE",
        "DERIVED_ENRICHMENT_PENDING",
        "MODEL_FAILED",
        "MODEL_INVALID_OUTPUT",
        "BUDGET_EXHAUSTED",
        "MODEL_USAGE_UNKNOWN",
        "MODEL_USAGE_OVERRUN",
        "EPISODE_INSPECTION_PENDING",
        "SUMMARY_INPUTS_OMITTED",
        "READ_BUDGET_EXHAUSTED",
        "OUTPUT_TRUNCATED",
        "QUERY_TRUNCATED",
        "CONFLICTING_EVIDENCE_OMITTED",
        "WORKSPACE_TARGET_DIFFERS",
        "WORKSPACE_HEAD_DIFFERS",
        "WORKSPACE_UNAVAILABLE",
        "WORKSPACE_UNMERGED",
        "WORKSPACE_UNCHECKED",
        "WORKSPACE_SNAPSHOT",
        "UNTRACKED_TARGET",
        "EXPORT_GIT_OBJECTS_OMITTED",
        "CLEANUP_PENDING",
    }
)


def _validate_public_record_id(val: str) -> str:
    for prefix in PUBLIC_ID_PREFIXES:
        if val.startswith(prefix):
            suffix = val[len(prefix) :]
            if HEX_64_RE.match(suffix):
                return val
    raise ValueError(f"Invalid public RecordID format: {val!r}")


def _validate_relation_or_public_id(val: str) -> str:
    for prefix in PUBLIC_ID_PREFIXES | {"rel_"}:
        if val.startswith(prefix):
            suffix = val[len(prefix) :]
            if HEX_64_RE.match(suffix):
                return val
    raise ValueError(f"Invalid Relation/Record ID format: {val!r}")


def _validate_object_id(val: str, info: ValidationInfo) -> str:
    if not HEX_OID_RE.match(val):
        raise ValueError(f"Invalid Git ObjectID (must be 40 or 64 lowercase hex chars): {val!r}")
    if info.context and isinstance(info.context, dict):
        obj_fmt = info.context.get("object_format")
        if obj_fmt == "sha1" and len(val) != 40:
            raise ValueError(f"Expected 40-char sha1 ObjectID, got {len(val)} chars")
        if obj_fmt == "sha256" and len(val) != 64:
            raise ValueError(f"Expected 64-char sha256 ObjectID, got {len(val)} chars")
    return val


def _validate_utc_text(val: str) -> str:
    if not val.endswith("Z"):
        raise ValueError(f"UTCText must end with 'Z', got {val!r}")
    dt = parse_iso_utc(val)
    if dt.tzinfo is None:
        raise ValueError("UTCText must be timezone-aware UTC")
    return val


def _validate_gap_code(val: str) -> str:
    if val not in GAP_CODE_REGISTRY:
        raise ValueError(f"Unknown gap code {val!r} not in v1 registry")
    return val


def _validate_aware_utc_datetime(dt: datetime) -> datetime:
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError("Naive datetime is forbidden; supply timezone-aware UTC datetime.")
    return dt.astimezone(UTC)


class PublicModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


NonNegative = Annotated[int, Field(ge=0)]
Positive = Annotated[int, Field(ge=1)]
RecordID = Annotated[str, Field(min_length=1, max_length=100), AfterValidator(_validate_public_record_id)]
RelationID = Annotated[
    str, Field(min_length=1, max_length=100), AfterValidator(_validate_relation_or_public_id)
]
ObjectID = Annotated[str, Field(min_length=40, max_length=64), AfterValidator(_validate_object_id)]
ShortText = Annotated[str, Field(max_length=1000)]
UTCText = Annotated[str, Field(max_length=40), AfterValidator(_validate_utc_text)]
Cursor = Annotated[str, Field(min_length=1, max_length=8192)]
AwareDatetime = Annotated[
    datetime,
    AfterValidator(_validate_aware_utc_datetime),
    PlainSerializer(lambda dt: format_iso_z(dt), return_type=str, when_used="json"),
]

Support = Literal["explicit", "inferred", "unknown"]
Lifecycle = Literal["proposed", "adopted", "rejected", "disputed", "superseded", "unknown"]
Mapping = Literal["exact", "candidate", "unmapped"]
Applicability = Literal["corroborated", "inferred", "historical_only", "unknown"]
DerivedState = Literal["ready", "stale", "retracted"]
RecordKind = Literal[
    "symbol",
    "change",
    "hunk",
    "source",
    "source_version",
    "evidence",
    "claim",
    "summary",
    "gap",
]


class PathRef(PublicModel):
    path: Annotated[str, Field(min_length=1, max_length=32768)]
    path_bytes_b64: Annotated[str, Field(min_length=1, max_length=65536)] | None = None

    @model_validator(mode="after")
    def _validate_path_ref(self) -> "PathRef":
        if self.path_bytes_b64 is not None:
            raw = decode_path_b64(self.path_bytes_b64)
            validate_relative_repo_path_bytes(raw)
        else:
            raw = self.path.encode("utf-8")
            validate_relative_repo_path_bytes(raw)
        return self

    def raw_bytes(self) -> bytes:
        if self.path_bytes_b64 is not None:
            return decode_path_b64(self.path_bytes_b64)
        return self.path.encode("utf-8")


class ByteSpan(PublicModel):
    byte_start: NonNegative
    byte_end: NonNegative
    line_start: Positive | None = None
    line_end: Positive | None = None

    @model_validator(mode="after")
    def _validate_span(self) -> "ByteSpan":
        if self.byte_end < self.byte_start:
            raise ValueError(f"byte_end ({self.byte_end}) must be >= byte_start ({self.byte_start})")
        if (self.line_start is None) != (self.line_end is None):
            raise ValueError("line_start and line_end must be both null or both positive integers")
        if self.line_start is not None and self.line_end is not None:
            if self.line_end < self.line_start:
                raise ValueError(f"line_end ({self.line_end}) must be >= line_start ({self.line_start})")
        return self


class Anchor(PublicModel):
    kind: Literal["discussion", "code", "commit", "hunk"]
    source_version_id: RecordID | None = None
    git_oid: ObjectID | None = None
    revision: ObjectID | None = None
    path: PathRef | None = None
    span: ByteSpan
    underlying_record_id: RecordID | None = None
    original_review_commit: ObjectID | None = None

    @model_validator(mode="after")
    def _validate_anchor_kind(self) -> "Anchor":
        if self.kind == "discussion":
            if self.source_version_id is None or not self.source_version_id.startswith("ver_"):
                raise ValueError("Discussion anchor requires a ver_ source_version_id")
            if self.git_oid is not None:
                raise ValueError("Discussion anchor must have git_oid=None")
        elif self.kind == "code":
            if self.source_version_id is not None:
                raise ValueError("Code anchor must have source_version_id=None")
            if self.git_oid is None or self.revision is None or self.path is None:
                raise ValueError("Code anchor requires git_oid, revision, and path")
        elif self.kind in ("commit", "hunk"):
            if self.source_version_id is not None:
                raise ValueError(f"{self.kind} anchor must have source_version_id=None")
            if self.git_oid is None:
                raise ValueError(f"{self.kind} anchor requires git_oid")
        return self


class Attribution(PublicModel):
    provider: Literal["github", "git", "unknown"]
    actor_id: Annotated[str, Field(max_length=200)] | None = None
    display_name: Annotated[str, Field(max_length=400)] | None = None
    source_url: Annotated[str, Field(max_length=4096)] | None = None


class SourceTimes(PublicModel):
    authored_at: UTCText | None = None
    updated_at: UTCText | None = None
    first_observed_at: UTCText
    last_observed_at: UTCText


class EvidenceRef(PublicModel):
    id: RecordID
    anchor: Anchor
    content_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def _validate_ev_prefix(self) -> "EvidenceRef":
        if not self.id.startswith("ev_"):
            raise ValueError("EvidenceRef.id must start with 'ev_'")
        return self


class EvidenceView(PublicModel):
    ref: EvidenceRef
    attribution: Attribution
    times: SourceTimes | None = None
    excerpt: Annotated[str, Field(max_length=8192)] | None = None
    availability: Literal[
        "available",
        "unverified",
        "inaccessible",
        "missing_object",
        "oversized",
        "binary",
    ]
    historical_scope: Literal["target_ancestor", "nonancestor", "discussion", "unknown"]


class Gap(PublicModel):
    id: RecordID | None = None
    code: Annotated[str, Field(min_length=1, max_length=100), AfterValidator(_validate_gap_code)]
    scope: Annotated[str, Field(min_length=1, max_length=1000)]
    message: ShortText
    retryable: bool
    record_id: RecordID | None = None

    @model_validator(mode="after")
    def _validate_gap_id(self) -> "Gap":
        if self.id is not None and not self.id.startswith("gap_"):
            raise ValueError("Gap.id must start with 'gap_' when present")
        return self


class GapCount(PublicModel):
    code: Annotated[str, Field(max_length=100), AfterValidator(_validate_gap_code)]
    count: NonNegative


class CoveragePart(PublicModel):
    state: Literal["not_started", "partial", "complete", "blocked", "unavailable"]
    completed: NonNegative
    known_total: NonNegative | None = None
    unit: Literal["commits", "records", "blobs", "segments"]
    last_success_at: UTCText | None = None
    reconciled_at: UTCText | None = None
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
    first_gap_id: RecordID | None = None

    @model_validator(mode="after")
    def _validate_first_gap(self) -> "Coverage":
        if self.first_gap_id is not None and not self.first_gap_id.startswith("gap_"):
            raise ValueError("Coverage.first_gap_id must start with 'gap_'")
        return self


class OmittedCounts(PublicModel):
    code: NonNegative = 0
    decisions: NonNegative = 0
    evidence: NonNegative = 0
    relations: NonNegative = 0
    gaps: NonNegative = 0
    hits: NonNegative = 0
    statements: NonNegative = 0
    content_bytes: NonNegative = 0

    def has_any_omission(self) -> bool:
        return any(
            getattr(self, field) > 0
            for field in (
                "code",
                "decisions",
                "evidence",
                "relations",
                "gaps",
                "hits",
                "statements",
                "content_bytes",
            )
        )


class Budget(PublicModel):
    requested_tokens: Positive
    used_tokens: NonNegative
    estimator: Annotated[str, Field(min_length=1, max_length=200)]
    count_kind: Literal["estimated", "tokenizer"]
    truncated: bool
    omitted_counts: OmittedCounts


class Expand(PublicModel):
    id: RecordID
    reason: Annotated[str, Field(min_length=1, max_length=300)]


class Envelope(PublicModel):
    schema_version: Literal[1] = 1
    repository_id: Annotated[str, Field(min_length=36, max_length=36)]
    revision: ObjectID | None = None
    generation: NonNegative
    access_epoch: NonNegative
    coverage: Coverage
    gaps: Annotated[list[Gap], Field(max_length=20)]


class RelationView(PublicModel):
    id: RelationID
    from_kind: Annotated[str, Field(min_length=1, max_length=100)]
    from_id: Annotated[str, Field(min_length=1, max_length=100)]
    to_kind: Annotated[str, Field(min_length=1, max_length=100)]
    to_id: Annotated[str, Field(min_length=1, max_length=100)]
    relation_kind: Literal[
        "contains",
        "imports_candidate",
        "calls_candidate",
        "changes",
        "associated_pr",
        "references_issue",
        "review_anchor",
        "landed_mapping",
        "targets",
        "supports",
        "contradicts",
        "supersedes",
        "corroborates",
    ]
    method: Annotated[str, Field(min_length=1, max_length=200)]
    mapping: Mapping
    evidence_id: RecordID | None = None
    state: Literal["active", "stale", "retracted"]
    explanation: ShortText | None = None


class SearchHit(PublicModel):
    id: RecordID
    kind: RecordKind
    title: Annotated[str, Field(max_length=1000)]
    snippet: Annotated[str, Field(max_length=320)]
    rank: Positive
    match: Literal["exact_id", "exact_path", "exact_symbol", "lexical"]
    path: PathRef | None = None
    evidence_id: RecordID | None = None
    state: DerivedState | None = None


class SearchResult(Envelope):
    query: Annotated[str, Field(min_length=1, max_length=4000)]
    hits: Annotated[list[SearchHit], Field(max_length=100)]
    next_cursor: Cursor | None = None
    truncated: bool
    budget: Budget
    expand: Annotated[list[Expand], Field(max_length=20)]

    @model_validator(mode="after")
    def _validate_search_result(self) -> "SearchResult":
        if self.truncated != self.budget.truncated:
            raise ValueError("SearchResult.truncated must equal budget.truncated")
        return self


class ResolvedTarget(PublicModel):
    kind: Literal["symbol", "path"]
    symbol_id: RecordID | None = None
    path: PathRef
    revision: ObjectID
    blob_oid: ObjectID
    line: Positive | None = None

    @model_validator(mode="after")
    def _validate_resolved_target(self) -> "ResolvedTarget":
        if self.symbol_id is not None and not self.symbol_id.startswith("sym_"):
            raise ValueError("ResolvedTarget.symbol_id must start with 'sym_'")
        return self


class WorkspaceState(PublicModel):
    checked: bool
    observed_at: UTCText | None = None
    head_revision: ObjectID | None = None
    head_matches_index: bool | None = None
    repository_diverged: bool | None = None
    target_state: Literal[
        "unchanged",
        "staged_change",
        "unstaged_change",
        "staged_and_unstaged",
        "deleted",
        "untracked",
        "unavailable",
        "not_checked",
    ]
    index_blob_oid: ObjectID | None = None
    staged_blob_oid: ObjectID | None = None
    working_content_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None


class CodeView(PublicModel):
    evidence_id: RecordID
    symbol_id: RecordID | None = None
    anchor: Anchor
    language: Annotated[str, Field(max_length=40)] | None = None
    signature: Annotated[str, Field(max_length=4000)] | None = None
    text: Annotated[str, Field(max_length=16384)]
    mapping: Mapping


class DecisionView(PublicModel):
    id: RecordID
    statement: Annotated[str, Field(min_length=1, max_length=1000)]
    rationale: Annotated[str, Field(min_length=1, max_length=3000)] | None = None
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

    @model_validator(mode="after")
    def _validate_decision_view(self) -> "DecisionView":
        if not self.id.startswith("claim_"):
            raise ValueError("DecisionView.id must start with 'claim_'")
        if self.applicability == "inferred":
            raise ValueError("applicability='inferred' is reserved and cannot be emitted in v1")
        if self.support in ("explicit", "inferred") and not self.evidence_refs:
            raise ValueError("Explicit or inferred decisions require at least one evidence_ref")
        if self.applicability == "corroborated" and not self.applicability_evidence_refs:
            raise ValueError("corroborated applicability requires applicability_evidence_refs")
        return self


class SummaryStatement(PublicModel):
    text: Annotated[str, Field(min_length=1, max_length=1000)]
    evidence_refs: Annotated[list[EvidenceRef], Field(min_length=1, max_length=10)]


class Overview(PublicModel):
    kind: Literal["stored_summary", "structural"]
    summary_id: RecordID | None = None
    text: Annotated[str, Field(max_length=16000)]
    statements: Annotated[list[SummaryStatement], Field(max_length=40)]
    file_count: NonNegative
    group_count: NonNegative
    included_file_count: NonNegative
    included_group_count: NonNegative
    included_claim_count: NonNegative
    omitted_claim_count: NonNegative
    partial: bool

    @model_validator(mode="after")
    def _validate_overview(self) -> "Overview":
        if self.kind == "structural":
            if self.summary_id is not None:
                raise ValueError("Structural overview must have summary_id=None")
            if self.statements:
                raise ValueError("Structural overview must have empty statements")
        elif self.kind == "stored_summary":
            if self.summary_id is None or not self.summary_id.startswith("sum_"):
                raise ValueError("Stored summary overview requires a sum_ summary_id")
        return self


class ContextPack(Envelope):
    query: Annotated[str, Field(min_length=1, max_length=4000)]
    target: ResolvedTarget | None = None
    workspace: WorkspaceState
    overview: Overview | None = None
    code: Annotated[list[CodeView], Field(max_length=20)]
    decisions: Annotated[list[DecisionView], Field(max_length=20)]
    evidence: Annotated[list[EvidenceView], Field(max_length=100)]
    rationale_status: Literal[
        "supported",
        "enrichment_pending",
        "evidence_incomplete",
        "undocumented_in_available_evidence",
    ]
    budget: Budget
    expand: Annotated[list[Expand], Field(max_length=20)]

    @model_validator(mode="after")
    def _validate_context_pack(self) -> "ContextPack":
        dirty_states = {
            "staged_change",
            "unstaged_change",
            "staged_and_unstaged",
            "deleted",
            "untracked",
        }
        has_dirty_target = self.workspace.target_state in dirty_states or any(
            g.code == "WORKSPACE_TARGET_DIFFERS" for g in self.gaps
        )
        if has_dirty_target:
            for d in self.decisions:
                if d.applicability == "corroborated":
                    raise ValueError("Dirty or diverged target cannot have corroborated applicability")

        included_ev_ids = {ev.ref.id for ev in self.evidence}
        for d in self.decisions:
            for ref in d.evidence_refs:
                if ref.id not in included_ev_ids:
                    raise ValueError(
                        f"Decision {d.id} supporting evidence {ref.id} must be included in pack.evidence"
                    )
            for ref in d.counterevidence_refs:
                if ref.id not in included_ev_ids:
                    raise ValueError(
                        f"Decision {d.id} counterevidence {ref.id} must be included in pack.evidence"
                    )
            for ref in d.applicability_evidence_refs:
                if ref.id not in included_ev_ids:
                    raise ValueError(
                        f"Decision {d.id} applicability evidence {ref.id} must be included in pack.evidence"
                    )

        if self.rationale_status == "supported":
            qualifying = [
                d
                for d in self.decisions
                if d.support == "explicit"
                and d.rationale is not None
                and d.state == "ready"
                and (self.target is None or d.mapping == "exact")
            ]
            if not qualifying:
                raise ValueError(
                    "rationale_status='supported' requires at least one eligible explicit decision with non-null rationale and exact target mapping"
                )
        return self


class SymbolRecord(PublicModel):
    kind: Literal["symbol"] = "symbol"
    id: RecordID
    blob_oid: ObjectID
    language: Annotated[str, Field(min_length=1, max_length=40)]
    extractor_fingerprint: Annotated[str, Field(min_length=1, max_length=200)]
    name: Annotated[str, Field(min_length=1, max_length=1000)]
    qualified_name: Annotated[str, Field(min_length=1, max_length=2000)]
    symbol_kind: Annotated[str, Field(min_length=1, max_length=100)]
    declaration_span: ByteSpan
    body_span: ByteSpan | None = None
    parent_symbol_id: RecordID | None = None
    signature: Annotated[str, Field(max_length=4000)]

    @model_validator(mode="after")
    def _validate_symbol_record(self) -> "SymbolRecord":
        if not self.id.startswith("sym_"):
            raise ValueError("SymbolRecord.id must start with 'sym_'")
        if self.body_span is not None:
            if (
                self.body_span.byte_start < self.declaration_span.byte_start
                or self.body_span.byte_end > self.declaration_span.byte_end
            ):
                raise ValueError("Symbol body_span must lie within declaration_span")
        return self


class ChangeRecord(PublicModel):
    kind: Literal["change"] = "change"
    id: RecordID
    commit_oid: ObjectID
    parent_index: Annotated[int, Field(ge=-1)]
    parent_oid: ObjectID | None = None
    old_path: PathRef | None = None
    new_path: PathRef | None = None
    old_blob_oid: ObjectID | None = None
    new_blob_oid: ObjectID | None = None
    old_mode: NonNegative | None = None
    new_mode: NonNegative | None = None
    change_kind: Literal[
        "added",
        "modified",
        "deleted",
        "renamed",
        "copied",
        "type_changed",
        "unmerged",
        "unknown",
    ]
    rename_score: Annotated[int, Field(ge=0, le=100)] | None = None
    hunks_state: Literal["pending", "complete", "partial", "unavailable"]

    @model_validator(mode="after")
    def _validate_change_record(self) -> "ChangeRecord":
        if not self.id.startswith("chg_"):
            raise ValueError("ChangeRecord.id must start with 'chg_'")
        return self


class HunkRecord(PublicModel):
    kind: Literal["hunk"] = "hunk"
    id: RecordID
    change_id: RecordID
    ordinal: NonNegative
    old_start: NonNegative
    old_count: NonNegative
    new_start: NonNegative
    new_count: NonNegative

    @model_validator(mode="after")
    def _validate_hunk_record(self) -> "HunkRecord":
        if not self.id.startswith("hunk_"):
            raise ValueError("HunkRecord.id must start with 'hunk_'")
        if not self.change_id.startswith("chg_"):
            raise ValueError("HunkRecord.change_id must start with 'chg_'")
        return self


class SourceRecord(PublicModel):
    kind: Literal["source"] = "source"
    id: RecordID
    source_kind: Literal[
        "issue",
        "pull_request",
        "issue_comment",
        "review",
        "review_comment",
        "timeline_event",
    ]
    provider: Literal["github"]
    repository_provider_id: Annotated[str, Field(min_length=1, max_length=200)]
    provider_id: Annotated[str, Field(min_length=1, max_length=200)]
    parent_source_id: RecordID | None = None
    selected_version_id: RecordID | None = None
    availability: Literal["available", "unverified", "inaccessible", "redacted", "deleted"]
    last_seen_at: UTCText
    last_reconciled_at: UTCText | None = None

    @model_validator(mode="after")
    def _validate_source_record(self) -> "SourceRecord":
        if not self.id.startswith("src_"):
            raise ValueError("SourceRecord.id must start with 'src_'")
        if self.selected_version_id is not None and not self.selected_version_id.startswith("ver_"):
            raise ValueError("SourceRecord.selected_version_id must start with 'ver_'")
        return self


class SourceVersionRecord(PublicModel):
    kind: Literal["source_version"] = "source_version"
    id: RecordID
    source_id: RecordID
    title: Annotated[str, Field(max_length=4000)] | None = None
    attribution: Attribution
    times: SourceTimes
    content_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    body_bytes: NonNegative
    current: bool

    @model_validator(mode="after")
    def _validate_version_record(self) -> "SourceVersionRecord":
        if not self.id.startswith("ver_"):
            raise ValueError("SourceVersionRecord.id must start with 'ver_'")
        if not self.source_id.startswith("src_"):
            raise ValueError("SourceVersionRecord.source_id must start with 'src_'")
        return self


class EvidenceRecord(PublicModel):
    kind: Literal["evidence"] = "evidence"
    id: RecordID
    evidence: EvidenceView

    @model_validator(mode="after")
    def _validate_evidence_record(self) -> "EvidenceRecord":
        if not self.id.startswith("ev_") or self.evidence.ref.id != self.id:
            raise ValueError("EvidenceRecord.id must match evidence.ref.id and start with 'ev_'")
        return self


class ClaimRecord(PublicModel):
    kind: Literal["claim"] = "claim"
    id: RecordID
    episode_id: Annotated[str, Field(min_length=1, max_length=100)]
    state: DerivedState
    model_run_id: Annotated[str, Field(min_length=1, max_length=100)]
    created_at: UTCText
    decision: DecisionView | None = None

    @model_validator(mode="after")
    def _validate_claim_record(self) -> "ClaimRecord":
        if not self.id.startswith("claim_"):
            raise ValueError("ClaimRecord.id must start with 'claim_'")
        return self


class SummaryRecord(PublicModel):
    kind: Literal["summary"] = "summary"
    id: RecordID
    scope_kind: Literal["package", "directory", "root_files", "repository"]
    scope_id: Annotated[str, Field(min_length=1, max_length=100)]
    revision: ObjectID
    state: DerivedState
    input_fingerprint: Annotated[str, Field(min_length=1, max_length=100)]
    partial: bool
    statements: Annotated[list[SummaryStatement], Field(max_length=40)]

    @model_validator(mode="after")
    def _validate_summary_record(self) -> "SummaryRecord":
        if not self.id.startswith("sum_"):
            raise ValueError("SummaryRecord.id must start with 'sum_'")
        return self


class GapRecord(PublicModel):
    kind: Literal["gap"] = "gap"
    id: RecordID
    gap: Gap
    active: bool
    first_seen_at: UTCText
    last_seen_at: UTCText
    next_gap_id: RecordID | None = None

    @model_validator(mode="after")
    def _validate_gap_record(self) -> "GapRecord":
        if not self.id.startswith("gap_"):
            raise ValueError("GapRecord.id must start with 'gap_'")
        if self.next_gap_id is not None and not self.next_gap_id.startswith("gap_"):
            raise ValueError("GapRecord.next_gap_id must start with 'gap_'")
        return self


Record = Annotated[
    SymbolRecord
    | ChangeRecord
    | HunkRecord
    | SourceRecord
    | SourceVersionRecord
    | EvidenceRecord
    | ClaimRecord
    | SummaryRecord
    | GapRecord,
    Field(discriminator="kind"),
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
    next_cursor: Cursor | None = None
    budget: Budget
    expand: Annotated[list[Expand], Field(max_length=20)]


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
    github_repository: Annotated[str, Field(max_length=200)] | None = None
    object_format: Literal["sha1", "sha256"]
    jobs: QueueCounts
    snapshot_file_count: NonNegative
    commit_count: NonNegative
    source_count: NonNegative
    claim_count: NonNegative
    summary_count: NonNegative
    budget: Budget
    expand: Annotated[list[Expand], Field(max_length=20)]

    @model_validator(mode="after")
    def _validate_status(self) -> "Status":
        if len(self.gaps) > 10:
            raise ValueError("Status includes at most 10 detailed gaps")
        return self


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
    stop_reason: Literal[
        "scope_complete",
        "request_budget",
        "time_budget",
        "offline",
        "source_blocked",
        "cancelled",
    ]


class Usage(PublicModel):
    calls: NonNegative
    estimated_input_tokens: NonNegative
    reserved_output_tokens: NonNegative
    reported_input_tokens: NonNegative | None = None
    reported_output_tokens: NonNegative | None = None
    uncertain_input_tokens: NonNegative
    uncertain_output_tokens: NonNegative
    prior_uncertain_input_tokens: NonNegative
    prior_uncertain_output_tokens: NonNegative
    configured_price_cost_usd: Annotated[str, Field(max_length=100)] | None = None
    cost_is_estimated: bool
    overrun: bool


class ItemFailure(PublicModel):
    item_key: Annotated[str, Field(min_length=1, max_length=300)]
    code: Annotated[str, Field(min_length=1, max_length=100)]
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
    stop_reason: Literal[
        "scope_complete",
        "budget",
        "model_failure",
        "usage_unknown",
        "usage_overrun",
        "cancelled",
    ]


class ExportReport(PublicModel):
    schema_version: Literal[1] = 1
    repository_id: Annotated[str, Field(min_length=36, max_length=36)]
    revision: ObjectID | None = None
    generation: NonNegative
    access_epoch: NonNegative
    destination: Annotated[str, Field(min_length=1, max_length=4096)]
    include_git_objects: bool
    file_count: NonNegative
    total_bytes: NonNegative
    manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    warnings: Annotated[list[ShortText], Field(max_length=20)]


class PurgeReport(PublicModel):
    schema_version: Literal[1] = 1
    repository_id: Annotated[str, Field(min_length=36, max_length=36)]
    mode: Literal["source", "entire_store"]
    source_id: RecordID | None = None
    generation: NonNegative
    access_epoch: NonNegative
    removed_source_versions: NonNegative
    removed_claims: NonNegative
    removed_summaries: NonNegative
    cleanup_pending: bool
    closed: bool


# --- Target and Request Models ---


class SymbolTarget(PublicModel):
    kind: Literal["symbol"] = "symbol"
    id: RecordID

    @model_validator(mode="after")
    def _validate_sym_id(self) -> "SymbolTarget":
        if not self.id.startswith("sym_"):
            raise ValueError("SymbolTarget.id must start with 'sym_'")
        return self


class PathTarget(PublicModel):
    kind: Literal["path"] = "path"
    path: Annotated[str, Field(min_length=1, max_length=32768)] | None = None
    path_bytes_b64: Annotated[str, Field(min_length=1, max_length=65536)] | None = None
    line: Positive | None = None

    @model_validator(mode="after")
    def _validate_path_target(self) -> "PathTarget":
        if (self.path is None) == (self.path_bytes_b64 is None):
            raise ValueError("PathTarget requires exactly one of 'path' or 'path_bytes_b64'")
        raw = self.raw_bytes()
        validate_relative_repo_path_bytes(raw)
        return self

    def raw_bytes(self) -> bytes:
        if self.path_bytes_b64 is not None:
            return decode_path_b64(self.path_bytes_b64)
        assert self.path is not None
        return self.path.encode("utf-8")

    def to_path_ref(self) -> PathRef:
        raw = self.raw_bytes()
        from codecontext.identity import path_bytes_to_display

        disp, b64 = path_bytes_to_display(raw)
        return PathRef(path=disp, path_bytes_b64=b64)


Target = Annotated[SymbolTarget | PathTarget, Field(discriminator="kind")]


def _validate_ref_string(val: str) -> str:
    if not val or len(val) > 1024:
        raise ValueError("Revision/ref must be 1..1024 characters")
    if "\x00" in val or "\n" in val or "\r" in val or val.startswith("-"):
        raise ValueError("Revision/ref must not contain NUL, newline, or a leading hyphen")
    return val


def _validate_trimmed_query(val: str) -> str:
    trimmed = val.strip()
    if not trimmed or len(trimmed) > 4000:
        raise ValueError("Query must be 1..4000 characters after whitespace trimming")
    return trimmed


RefText = Annotated[str, Field(min_length=1, max_length=1024), AfterValidator(_validate_ref_string)]
QueryText = Annotated[str, Field(min_length=1, max_length=4500), AfterValidator(_validate_trimmed_query)]


class InitRequest(PublicModel):
    repo_path: Path
    github_repository: Annotated[str, Field(min_length=3, max_length=200)] | None = None
    allowed_issue_repositories: Annotated[list[str], Field(max_length=64)] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_init(self) -> "InitRequest":
        if self.github_repository is not None:
            validate_github_repo_name(self.github_repository)
        seen: set[str] = set()
        for entry in self.allowed_issue_repositories:
            validate_github_repo_name(entry)
            norm = entry.lower()
            if norm in seen:
                raise ValueError(f"Duplicate repository in allowed_issue_repositories: {entry!r}")
            seen.add(norm)
        return self


class SyncRequest(PublicModel):
    revision: RefText = "HEAD"
    offline: bool = False
    reconcile: bool = False
    focus: Target | None = None
    max_requests: Annotated[int, Field(ge=1, le=100_000)] = 1000
    max_seconds: Annotated[int, Field(ge=1, le=86_400)] = 900


class EnrichRequest(PublicModel):
    target_ids: Annotated[list[RecordID], Field(max_length=1000)] | None = None
    max_input_tokens: Annotated[int, Field(ge=1, le=1_000_000_000)]
    max_output_tokens: Annotated[int, Field(ge=1, le=1_000_000_000)]
    max_calls: Annotated[int, Field(ge=1, le=1_000_000)]
    max_cost_usd: Decimal | None = None

    @model_validator(mode="after")
    def _validate_enrich(self) -> "EnrichRequest":
        if self.target_ids is not None and len(set(self.target_ids)) != len(self.target_ids):
            raise ValueError("target_ids must contain distinct IDs")
        if self.max_cost_usd is not None and (
            not self.max_cost_usd.is_finite()
            or self.max_cost_usd <= Decimal(0)
            or self.max_cost_usd > Decimal(1000000)
        ):
            raise ValueError("max_cost_usd must be a positive finite Decimal <= 1,000,000")
        return self


class SearchRequest(PublicModel):
    query: QueryText
    revision: RefText | None = None
    kind: Literal["all", "code", "history", "decision", "summary"] = "all"
    limit: Annotated[int, Field(ge=1, le=100)] = 20
    knowledge_cutoff: AwareDatetime | None = None
    cursor: Cursor | None = None


class GetRequest(PublicModel):
    id: RecordID
    max_tokens: Annotated[int, Field(ge=512, le=32000)] = 6000
    cursor: Cursor | None = None
    knowledge_cutoff: AwareDatetime | None = None


class ContextRequest(PublicModel):
    query: QueryText
    target: Target | None = None
    revision: RefText | None = None
    max_tokens: Annotated[int, Field(ge=512, le=32000)] = 6000
    knowledge_cutoff: AwareDatetime | None = None


class ExportRequest(PublicModel):
    destination: Path
    include_git_objects: bool = True


class PurgeRequest(PublicModel):
    source_id: RecordID | None = None
    entire_store: bool = False

    @model_validator(mode="after")
    def _validate_purge(self) -> "PurgeRequest":
        if (self.source_id is not None) == bool(self.entire_store):
            raise ValueError("PurgeRequest requires exactly one of source_id or entire_store=True")
        if self.source_id is not None and not self.source_id.startswith("src_"):
            raise ValueError("PurgeRequest.source_id must start with 'src_'")
        return self


# --- Model Adapter & Extraction Schemas ---


class ModelIdentity(PublicModel):
    adapter_name: Annotated[str, Field(min_length=1, max_length=200)]
    model_name: Annotated[str, Field(min_length=1, max_length=200)]
    model_revision: Annotated[str, Field(min_length=1, max_length=200)]
    configuration_fingerprint: Annotated[str, Field(min_length=1, max_length=200)]
    input_price_per_million: Decimal | None = None
    output_price_per_million: Decimal | None = None
    framing_input_tokens: NonNegative = 0

    @model_validator(mode="after")
    def _validate_prices(self) -> "ModelIdentity":
        for field_name in ("input_price_per_million", "output_price_per_million"):
            val = getattr(self, field_name)
            if val is not None and (not val.is_finite() or val < Decimal(0)):
                raise ValueError(f"{field_name} must be a non-negative finite Decimal")
        return self


class ModelRequest(PublicModel):
    request_id: Annotated[str, Field(min_length=1, max_length=100)]
    system_text: str
    input_json: str
    output_schema: dict[str, Any]
    max_output_tokens: Annotated[int, Field(ge=1, le=4000)]
    timeout_seconds: Annotated[int, Field(ge=1, le=3600)] = 120


class ModelResponse(PublicModel):
    output_json: Annotated[str, Field(min_length=1, max_length=262144)]
    input_tokens: NonNegative | None = None
    output_tokens: NonNegative | None = None
    provider_request_id: Annotated[str, Field(max_length=200)] | None = None

    @model_validator(mode="after")
    def _validate_response_bytes(self) -> "ModelResponse":
        if len(self.output_json.encode("utf-8")) > 262144:
            raise ValueError("ModelResponse.output_json exceeds 256 KiB")
        return self


@runtime_checkable
class ModelAdapter(Protocol):
    @property
    def identity(self) -> ModelIdentity: ...

    def complete(self, request: ModelRequest) -> ModelResponse: ...


class ExtractionCitation(PublicModel):
    slot: Annotated[str, Field(min_length=1, max_length=64)]
    quote: Annotated[str, Field(min_length=1, max_length=4000)]


class CorroborationCitation(PublicModel):
    slot: Annotated[str, Field(min_length=1, max_length=64)]
    quote: Annotated[str, Field(min_length=1, max_length=4000)]
    target_anchor_key: Annotated[str, Field(min_length=1, max_length=200)]


class ExtractedClaimOutput(PublicModel):
    statement: Annotated[str, Field(min_length=1, max_length=1000)]
    rationale: Annotated[str, Field(min_length=1, max_length=3000)] | None = None
    alternatives: Annotated[list[ShortText], Field(max_length=10)] = Field(default_factory=list)
    constraints: Annotated[list[ShortText], Field(max_length=10)] = Field(default_factory=list)
    support: Support
    lifecycle: Lifecycle
    target_anchor_keys: Annotated[list[Annotated[str, Field(min_length=1, max_length=200)]], Field(max_length=20)] = Field(
        default_factory=list
    )
    evidence: Annotated[list[ExtractionCitation], Field(max_length=20)] = Field(default_factory=list)
    counterevidence: Annotated[list[ExtractionCitation], Field(max_length=20)] = Field(default_factory=list)
    lifecycle_evidence: Annotated[list[ExtractionCitation], Field(max_length=20)] = Field(default_factory=list)
    corroboration: Annotated[list[CorroborationCitation], Field(max_length=10)] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_claim_output(self) -> "ExtractedClaimOutput":
        if self.support in ("explicit", "inferred") and not self.evidence:
            raise ValueError("Explicit or inferred claims require at least one supporting citation")
        if self.lifecycle != "unknown" and not self.lifecycle_evidence:
            raise ValueError("Non-unknown lifecycle requires at least one lifecycle_evidence citation")
        return self


class EpisodeExtractionOutput(PublicModel):
    claims: Annotated[list[ExtractedClaimOutput], Field(max_length=20)]
    inspected_slots: list[Annotated[str, Field(min_length=1, max_length=64)]]
    unresolved: Annotated[list[Annotated[str, Field(min_length=1, max_length=1000)]], Field(max_length=20)]

    @model_validator(mode="after")
    def _validate_unique_slots(self) -> "EpisodeExtractionOutput":
        if len(set(self.inspected_slots)) != len(self.inspected_slots):
            raise ValueError("inspected_slots must contain distinct slot strings")
        return self


class SummaryStatementOutput(PublicModel):
    text: Annotated[str, Field(min_length=1, max_length=1000)]
    evidence: Annotated[list[ExtractionCitation], Field(min_length=1, max_length=10)]


class SummaryExtractionOutput(PublicModel):
    statements: Annotated[list[SummaryStatementOutput], Field(max_length=40)]
    inspected_slots: list[Annotated[str, Field(min_length=1, max_length=64)]]
    unresolved: Annotated[list[Annotated[str, Field(min_length=1, max_length=1000)]], Field(max_length=20)]

    @model_validator(mode="after")
    def _validate_unique_slots(self) -> "SummaryExtractionOutput":
        if len(set(self.inspected_slots)) != len(self.inspected_slots):
            raise ValueError("inspected_slots must contain distinct slot strings")
        return self
