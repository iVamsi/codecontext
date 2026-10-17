from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from codecontext.errors import CodeContextError
from codecontext.identity import (
    byte_span_to_lines,
    canonical_sha256,
    decode_path_b64,
    encode_path_b64,
    hash_source_version_payload,
    line_range_to_byte_span,
    make_claim_id,
    make_ev_id,
    make_gap_id,
    make_src_id,
    make_sym_id,
    make_ver_id,
    path_bytes_to_display,
    sha256_bytes,
)
from codecontext.models import (
    Anchor,
    Attribution,
    Budget,
    ByteSpan,
    ContextPack,
    ContextRequest,
    Coverage,
    CoveragePart,
    DecisionView,
    EnrichRequest,
    EvidenceRef,
    EvidenceView,
    Gap,
    GapRecord,
    GetRequest,
    InitRequest,
    OmittedCounts,
    PathRef,
    PathTarget,
    Record,
    ResolvedTarget,
    SearchRequest,
    SourceTimes,
    SymbolRecord,
    SyncRequest,
    WorkspaceState,
)
from tests.fixtures.builders import build_f01_fixture, build_f03_fixture, build_f09_samples


def test_u01_canonical_identities_and_git_object_formats(tmp_path: Path) -> None:
    """U01: Hash reordered identity fields, changed observation times, changed payload fields, and both Git object formats."""
    # Reordered dictionary keys produce identical canonical SHA-256
    h1 = canonical_sha256({"b": 2, "a": 1, "nested": {"y": "z", "x": "w"}})
    h2 = canonical_sha256({"a": 1, "nested": {"x": "w", "y": "z"}, "b": 2})
    assert h1 == h2

    # Floating point values in identity are rejected
    with pytest.raises(ValueError, match="Floating-point"):
        canonical_sha256({"score": 0.5})

    # Source version hash: observation times are excluded; body/title/author/source times change version ID
    src_id = make_src_id(
        provider="github",
        repository_provider_id="12345",
        kind="issue_comment",
        provider_id="999",
    )
    v_hash_1 = hash_source_version_payload(
        title=None,
        body="Limit retries to 3",
        author_id="501",
        author_name="alice",
        anchors={},
        authored_at_us=1700000000000000,
        updated_at_us=1700000000000000,
        lifecycle_fields={"state": "open"},
    )
    v_hash_same = hash_source_version_payload(
        title=None,
        body="Limit retries to 3",
        author_id="501",
        author_name="alice",
        anchors={},
        authored_at_us=1700000000000000,
        updated_at_us=1700000000000000,
        lifecycle_fields={"state": "open"},
    )
    assert make_ver_id(source_id=src_id, content_sha256=v_hash_1) == make_ver_id(
        source_id=src_id, content_sha256=v_hash_same
    )

    v_hash_edited = hash_source_version_payload(
        title=None,
        body="Limit retries to 5",
        author_id="501",
        author_name="alice",
        anchors={},
        authored_at_us=1700000000000000,
        updated_at_us=1700000005000000,
        lifecycle_fields={"state": "open"},
    )
    assert v_hash_1 != v_hash_edited

    # Both SHA-1 (40 chars) and SHA-256 (64 chars) Git object IDs survive without truncation
    sha1_oid = "a" * 40
    sha256_oid = "b" * 64
    sym_sha1 = make_sym_id(
        blob_oid=sha1_oid,
        language="python",
        extractor_fingerprint="fp1",
        byte_start=0,
        byte_end=20,
        kind="function",
        qualified_name="send",
    )
    sym_sha256 = make_sym_id(
        blob_oid=sha256_oid,
        language="python",
        extractor_fingerprint="fp1",
        byte_start=0,
        byte_end=20,
        kind="function",
        qualified_name="send",
    )
    rec_1 = SymbolRecord(
        id=sym_sha1,
        blob_oid=sha1_oid,
        language="python",
        extractor_fingerprint="fp1",
        name="send",
        qualified_name="send",
        symbol_kind="function",
        declaration_span=ByteSpan(byte_start=0, byte_end=20, line_start=1, line_end=2),
        signature="def send()",
    )
    rec_256 = SymbolRecord(
        id=sym_sha256,
        blob_oid=sha256_oid,
        language="python",
        extractor_fingerprint="fp1",
        name="send",
        qualified_name="send",
        symbol_kind="function",
        declaration_span=ByteSpan(byte_start=0, byte_end=20, line_start=1, line_end=2),
        signature="def send()",
    )
    assert len(rec_1.blob_oid) == 40
    assert len(rec_256.blob_oid) == 64

    # Validate format-specific ObjectID context check
    SymbolRecord.model_validate(rec_1.model_dump(), context={"object_format": "sha1"})
    SymbolRecord.model_validate(rec_256.model_dump(), context={"object_format": "sha256"})
    with pytest.raises(ValidationError):
        SymbolRecord.model_validate(rec_1.model_dump(), context={"object_format": "sha256"})

    # Test real Git fixture builders in both sha1 and sha256
    f01_sha1 = build_f01_fixture(tmp_path / "s1", object_format="sha1")
    f01_sha256 = build_f01_fixture(tmp_path / "s256", object_format="sha256")
    assert len(f01_sha1.c1_oid) == 40
    assert len(f01_sha256.c1_oid) == 64


def test_u02_byte_ranges_unicode_crlf_and_empty_hunks() -> None:
    """U02: Convert F09 byte ranges, Unicode, CRLF, and empty Git hunks."""
    f09 = build_f09_samples()
    blob = f09.unicode_crlf_source

    # Line 1: "# Café résumé — naïve 🚀\r\n"
    b_start, b_end, l_start, l_end = line_range_to_byte_span(blob, 1, 1)
    assert l_start == 1 and l_end == 1
    assert blob[b_start:b_end].decode("utf-8") == "# Café résumé — naïve 🚀\r\n"
    assert byte_span_to_lines(blob, b_start, b_end) == (1, 1)

    # Multi-line span across CRLF and multi-byte UTF-8
    b_start_2, b_end_2, l_start_2, l_end_2 = line_range_to_byte_span(blob, 2, 3)
    assert (l_start_2, l_end_2) == (2, 4)
    assert byte_span_to_lines(blob, b_start_2, b_end_2) == (2, 4)

    # Empty insertion boundary at start of file (line_start=0, count=0)
    ins0_start, ins0_end, ins0_l1, ins0_l2 = line_range_to_byte_span(blob, 0, 0)
    assert (ins0_start, ins0_end) == (0, 0)
    assert ins0_l1 is None and ins0_l2 is None
    span0 = ByteSpan(byte_start=ins0_start, byte_end=ins0_end, line_start=ins0_l1, line_end=ins0_l2)
    assert span0.byte_start == 0 and span0.byte_end == 0

    # Empty insertion boundary after line 1 (line_start=1, count=0)
    ins1_start, ins1_end, ins1_l1, ins1_l2 = line_range_to_byte_span(blob, 1, 0)
    assert ins1_start == b_end == ins1_end
    assert ins1_l1 is None and ins1_l2 is None

    # Non-UTF-8 path bytes round-trip via base64
    non_utf8 = b"src/weird_\xff\xfe_file.py"
    disp, b64 = path_bytes_to_display(non_utf8)
    assert b64 is not None
    assert "\\xff\\xfe" in disp
    assert decode_path_b64(b64) == non_utf8
    pref = PathRef(path=disp, path_bytes_b64=b64)
    assert pref.raw_bytes() == non_utf8


def test_u03_invalid_requests_and_sanitized_errors() -> None:
    """U03: Submit empty/oversized queries, unknown fields, invalid budgets, absolute paths, traversal, and malformed targets."""
    # Empty or whitespace-only query
    with pytest.raises(ValidationError):
        SearchRequest(query="   ")

    # Oversized query (>4000 chars)
    with pytest.raises(ValidationError):
        SearchRequest(query="x" * 4001)

    # Unknown extra field
    with pytest.raises(ValidationError):
        SearchRequest.model_validate({"query": "retry", "extra_field": 1})

    # Naive cutoff datetime rejected
    with pytest.raises(ValidationError):
        SearchRequest(query="retry", knowledge_cutoff=datetime(2026, 1, 1, 12, 0, 0))

    # Invalid max_tokens bounds
    with pytest.raises(ValidationError):
        ContextRequest(query="retry", max_tokens=100)
    with pytest.raises(ValidationError):
        GetRequest(id="sym_" + "a" * 64, max_tokens=50000)

    # Invalid EnrichRequest budgets (0 or negative or boolean)
    with pytest.raises(ValidationError):
        EnrichRequest(max_input_tokens=0, max_output_tokens=1000, max_calls=1)
    with pytest.raises(ValidationError):
        EnrichRequest(max_input_tokens=True, max_output_tokens=1000, max_calls=1)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        EnrichRequest(
            max_input_tokens=1000,
            max_output_tokens=1000,
            max_calls=1,
            max_cost_usd=Decimal("-1.0"),
        )

    # PathTarget with both or neither of path / path_bytes_b64
    with pytest.raises(ValidationError):
        PathTarget(kind="path", path="src/a.py", path_bytes_b64=encode_path_b64(b"src/a.py"))
    with pytest.raises(ValidationError):
        PathTarget(kind="path")

    # PathTarget with absolute path, '..' traversal, or NUL
    for bad_path in ["/etc/passwd", "../secret.py", "src/../../outside.py", "src/a\x00b.py"]:
        with pytest.raises((ValidationError, CodeContextError)):
            PathTarget(kind="path", path=bad_path)

    for bad_raw in [b"/abs/path.py", b"a/../b.py", b"a/b\x00.py"]:
        with pytest.raises((ValidationError, CodeContextError)):
            PathTarget(kind="path", path_bytes_b64=encode_path_b64(bad_raw))

    # Ref starting with hyphen or containing newline
    with pytest.raises(ValidationError):
        SyncRequest(revision="--upload-pack=evil")
    with pytest.raises(ValidationError):
        SyncRequest(revision="main\nevil")

    # InitRequest with URL or traversal in github_repository
    with pytest.raises((ValidationError, CodeContextError)):
        InitRequest(repo_path=Path("/tmp/repo"), github_repository="https://github.com/a/b")
    with pytest.raises((ValidationError, CodeContextError)):
        InitRequest(repo_path=Path("/tmp/repo"), allowed_issue_repositories=["../evil"])

    # Error details sanitization strips secret keys and caps size
    err = CodeContextError(
        code="INVALID_REQUEST",
        message="Bad input",
        retryable=False,
        details={
            "github_token": "ghp_secret123456",
            "authorization": "Bearer secret",
            "prompt": "raw prompt",
            "safe_field": "allowed_value",
        },
    )
    env = err.to_envelope()
    assert "github_token" not in env["error"]["details"]
    assert "authorization" not in env["error"]["details"]
    assert "prompt" not in env["error"]["details"]
    assert env["error"]["details"]["safe_field"] == "allowed_value"


def test_public_models_cross_field_validation_and_discriminators(tmp_path: Path) -> None:
    """Verify cross-field rules on ContextPack, DecisionView, Record union, and F03 builder."""
    f03 = build_f03_fixture(tmp_path)
    assert len(f03.merge_oid) == 40
    assert f03.root_oid != f03.merge_oid

    repo_id = "12345678-1234-5678-1234-567812345678"
    rev = f03.merge_oid
    cov_part = CoveragePart(state="complete", completed=5, known_total=5, unit="commits", gap_count=0)
    cov = Coverage(
        git=cov_part,
        github=CoveragePart(state="not_started", completed=0, known_total=None, unit="records", gap_count=0),
        structure=CoveragePart(state="complete", completed=3, known_total=3, unit="blobs", gap_count=0),
        enrichment=CoveragePart(state="complete", completed=1, known_total=1, unit="segments", gap_count=0),
        selected_refs=["refs/heads/main"],
        selected_ref_count=1,
        omitted_ref_count=0,
        excluded_file_count=0,
        parser_fingerprints=["py_v1"],
        active_gap_count=0,
        gap_counts=[],
    )

    content_bytes = b"Limit retries to 3"
    content_hash = sha256_bytes(content_bytes)
    src_id = make_src_id(provider="github", repository_provider_id="1", kind="pull_request", provider_id="10")
    ver_id = make_ver_id(source_id=src_id, content_sha256=content_hash)
    anchor = Anchor(
        kind="discussion",
        source_version_id=ver_id,
        span=ByteSpan(byte_start=0, byte_end=len(content_bytes), line_start=1, line_end=1),
    )
    ev_id = make_ev_id(
        repository_id=repo_id,
        kind="discussion",
        source_version_id=ver_id,
        git_oid=None,
        anchor=anchor.model_dump(),
        content_sha256=content_hash,
    )
    ev_ref = EvidenceRef(id=ev_id, anchor=anchor, content_sha256=content_hash)
    ev_view = EvidenceView(
        ref=ev_ref,
        attribution=Attribution(provider="github", actor_id="501", display_name="alice"),
        times=SourceTimes(
            authored_at="2026-01-02T09:00:00Z",
            updated_at="2026-01-02T09:00:00Z",
            first_observed_at="2026-01-02T10:00:00Z",
            last_observed_at="2026-01-02T10:00:00Z",
        ),
        excerpt="Limit retries to 3",
        availability="available",
        historical_scope="discussion",
    )
    claim_id = make_claim_id(
        repository_id=repo_id,
        episode_id="ep_" + "c" * 64,
        request_fingerprint="req_fp",
        statement="Bound retries to 3",
        rationale="Prevent worker pool exhaustion",
        alternatives=[],
        constraints=[],
        support="explicit",
        lifecycle="adopted",
        evidence_roles=[(ev_id, "support")],
    )
    decision = DecisionView(
        id=claim_id,
        statement="Bound retries to 3",
        rationale="Prevent worker pool exhaustion",
        alternatives=[],
        constraints=[],
        support="explicit",
        lifecycle="adopted",
        state="ready",
        mapping="exact",
        applicability="historical_only",
        attributions=[Attribution(provider="github", actor_id="501", display_name="alice")],
        evidence_refs=[ev_ref],
        counterevidence_refs=[],
        applicability_evidence_refs=[],
    )

    # Reserved applicability='inferred' must fail validation
    with pytest.raises(ValidationError):
        DecisionView(
            **{**decision.model_dump(), "applicability": "inferred"}
        )

    # ContextPack with supported rationale
    pack = ContextPack(
        repository_id=repo_id,
        revision=rev,
        generation=1,
        access_epoch=0,
        coverage=cov,
        gaps=[],
        query="Why 3 retries?",
        target=ResolvedTarget(
            kind="path",
            path=PathRef(path="src/transport.py"),
            revision=rev,
            blob_oid="d" * 40,
            line=5,
        ),
        workspace=WorkspaceState(checked=False, target_state="not_checked"),
        overview=None,
        code=[],
        decisions=[decision],
        evidence=[ev_view],
        rationale_status="supported",
        budget=Budget(
            requested_tokens=6000,
            used_tokens=450,
            estimator="utf8_bytes_div3_v1",
            count_kind="estimated",
            truncated=False,
            omitted_counts=OmittedCounts(),
        ),
        expand=[],
    )
    # JSON round-trip validation
    dumped = pack.model_dump_json()
    reloaded = ContextPack.model_validate_json(dumped)
    assert reloaded == pack

    # Discriminated Record union round-trip
    record_adapter = TypeAdapter(Record)
    gap_id = make_gap_id(repository_id=repo_id, code="GIT_HISTORY_PENDING", scope="git:refs", record_id=None)
    gap_rec = GapRecord(
        id=gap_id,
        gap=Gap(
            id=gap_id,
            code="GIT_HISTORY_PENDING",
            scope="git:refs",
            message="Backfill pending",
            retryable=True,
        ),
        active=True,
        first_seen_at="2026-01-02T10:00:00Z",
        last_seen_at="2026-01-02T10:00:00Z",
    )
    parsed_rec = record_adapter.validate_json(gap_rec.model_dump_json())
    assert isinstance(parsed_rec, GapRecord)
    assert parsed_rec.id == gap_id
