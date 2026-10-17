import base64
import hashlib
import json
import math
import re
from datetime import UTC, datetime
from typing import Any

from codecontext.errors import CodeContextError

PUBLIC_ID_PREFIXES: frozenset[str] = frozenset(
    {"sym_", "chg_", "hunk_", "src_", "ver_", "ev_", "claim_", "sum_", "gap_"}
)
INTERNAL_ID_PREFIXES: frozenset[str] = frozenset(
    {"ext_", "ep_", "rel_", "seg_", "job_", "scope_"}
)
ALL_ID_PREFIXES: frozenset[str] = PUBLIC_ID_PREFIXES | INTERNAL_ID_PREFIXES

HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
HEX_OID_RE = re.compile(r"^[0-9a-f]{40}$|^[0-9a-f]{64}$")
GITHUB_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def _check_no_floats(value: Any) -> None:
    if isinstance(value, float):
        raise ValueError("Floating-point values are forbidden in canonical identity JSON.")
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError("Canonical JSON keys must be strings.")
            _check_no_floats(v)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_no_floats(item)


def canonical_json(data: Any) -> str:
    _check_no_floats(data)
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(data: Any) -> str:
    raw = canonical_json(data).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def make_id(prefix: str, payload: dict[str, Any]) -> str:
    if prefix not in ALL_ID_PREFIXES:
        raise ValueError(f"Unknown ID prefix: {prefix}")
    return f"{prefix}{canonical_sha256(payload)}"


def parse_public_id(record_id: str) -> tuple[str, str]:
    for prefix in PUBLIC_ID_PREFIXES:
        if record_id.startswith(prefix):
            suffix = record_id[len(prefix) :]
            if HEX_64_RE.match(suffix):
                return prefix, suffix
    raise CodeContextError(
        code="INVALID_REQUEST",
        message=f"Invalid public record ID: {record_id!r}",
        retryable=False,
        details={"id": record_id[:100]},
    )


def is_valid_record_id(record_id: str, *, allow_rel: bool = False) -> bool:
    prefixes = PUBLIC_ID_PREFIXES | ({"rel_"} if allow_rel else set())
    for prefix in prefixes:
        if record_id.startswith(prefix):
            suffix = record_id[len(prefix) :]
            return bool(HEX_64_RE.match(suffix))
    return False


def make_sym_id(
    *,
    blob_oid: str,
    language: str,
    extractor_fingerprint: str,
    byte_start: int,
    byte_end: int,
    kind: str,
    qualified_name: str,
) -> str:
    return make_id(
        "sym_",
        {
            "blob_oid": blob_oid,
            "byte_end": byte_end,
            "byte_start": byte_start,
            "extractor_fingerprint": extractor_fingerprint,
            "kind": kind,
            "language": language,
            "qualified_name": qualified_name,
        },
    )


def make_chg_id(
    *,
    repository_id: str,
    commit_oid: str,
    parent_index: int,
    old_path_b64: str | None,
    new_path_b64: str | None,
    old_blob_oid: str | None,
    new_blob_oid: str | None,
    change_kind: str,
) -> str:
    return make_id(
        "chg_",
        {
            "change_kind": change_kind,
            "commit_oid": commit_oid,
            "new_blob_oid": new_blob_oid,
            "new_path_b64": new_path_b64,
            "old_blob_oid": old_blob_oid,
            "old_path_b64": old_path_b64,
            "parent_index": parent_index,
            "repository_id": repository_id,
        },
    )


def make_hunk_id(
    *,
    change_id: str,
    ordinal: int,
    old_start: int,
    old_count: int,
    new_start: int,
    new_count: int,
    patch_sha256: str | None,
) -> str:
    return make_id(
        "hunk_",
        {
            "change_id": change_id,
            "new_count": new_count,
            "new_start": new_start,
            "old_count": old_count,
            "old_start": old_start,
            "ordinal": ordinal,
            "patch_sha256": patch_sha256,
        },
    )


def make_src_id(
    *,
    provider: str,
    repository_provider_id: str,
    kind: str,
    provider_id: str,
) -> str:
    return make_id(
        "src_",
        {
            "kind": kind,
            "provider": provider,
            "provider_id": str(provider_id),
            "repository_provider_id": str(repository_provider_id),
        },
    )


def make_ver_id(*, source_id: str, content_sha256: str) -> str:
    return make_id(
        "ver_",
        {
            "content_sha256": content_sha256,
            "source_id": source_id,
        },
    )


def make_ev_id(
    *,
    repository_id: str,
    kind: str,
    source_version_id: str | None,
    git_oid: str | None,
    anchor: dict[str, Any],
    content_sha256: str,
) -> str:
    return make_id(
        "ev_",
        {
            "anchor": anchor,
            "content_sha256": content_sha256,
            "git_oid": git_oid,
            "kind": kind,
            "repository_id": repository_id,
            "source_version_id": source_version_id,
        },
    )


def normalize_claim_text(text: str | None) -> str | None:
    if text is None:
        return None
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def make_claim_id(
    *,
    repository_id: str,
    episode_id: str,
    request_fingerprint: str,
    statement: str,
    rationale: str | None,
    alternatives: list[str],
    constraints: list[str],
    support: str,
    lifecycle: str,
    evidence_roles: list[tuple[str, str]],
) -> str:
    normalized_claim = {
        "alternatives": [normalize_claim_text(a) for a in alternatives],
        "constraints": [normalize_claim_text(c) for c in constraints],
        "lifecycle": lifecycle,
        "rationale": normalize_claim_text(rationale),
        "statement": normalize_claim_text(statement),
        "support": support,
    }
    sorted_roles = sorted({(ev_id, role) for ev_id, role in evidence_roles})
    return make_id(
        "claim_",
        {
            "episode_id": episode_id,
            "normalized_claim": normalized_claim,
            "repository_id": repository_id,
            "request_fingerprint": request_fingerprint,
            "sorted_evidence_roles": [[ev_id, role] for ev_id, role in sorted_roles],
        },
    )


def make_sum_id(
    *,
    repository_id: str,
    scope_kind: str,
    scope_id: str,
    revision: str,
    input_fingerprint: str,
    model_fingerprint: str,
    prompt_version: str,
) -> str:
    return make_id(
        "sum_",
        {
            "input_fingerprint": input_fingerprint,
            "model_fingerprint": model_fingerprint,
            "prompt_version": prompt_version,
            "repository_id": repository_id,
            "revision": revision,
            "scope_id": scope_id,
            "scope_kind": scope_kind,
        },
    )


def make_gap_id(
    *,
    repository_id: str,
    code: str,
    scope: str,
    record_id: str | None,
) -> str:
    return make_id(
        "gap_",
        {
            "code": code,
            "record_id": record_id,
            "repository_id": repository_id,
            "scope": scope,
        },
    )


def make_extraction_id(*, blob_oid: str, language: str, extractor_fingerprint: str) -> str:
    return make_id(
        "ext_",
        {
            "blob_oid": blob_oid,
            "extractor_fingerprint": extractor_fingerprint,
            "language": language,
        },
    )


def make_episode_id(
    *,
    repository_id: str,
    pr_source_id: str | None = None,
    commit_oid: str | None = None,
) -> str:
    if (pr_source_id is None) == (commit_oid is None):
        raise ValueError("Episode ID requires exactly one of pr_source_id or commit_oid")
    if pr_source_id is not None:
        return make_id("ep_", {"pr_source_id": pr_source_id, "repository_id": repository_id})
    return make_id("ep_", {"commit_oid": commit_oid, "repository_id": repository_id})


def make_relation_id(
    *,
    from_kind: str,
    from_id: str,
    to_kind: str,
    to_id: str,
    relation_kind: str,
    method: str,
    anchor_details: dict[str, Any] | None = None,
) -> str:
    return make_id(
        "rel_",
        {
            "anchor_details": anchor_details or {},
            "from_id": from_id,
            "from_kind": from_kind,
            "method": method,
            "relation_kind": relation_kind,
            "to_id": to_id,
            "to_kind": to_kind,
        },
    )


def make_segment_id(
    *,
    episode_id: str,
    input_fingerprint: str,
    ordinal: int,
    slot_anchors: list[str],
) -> str:
    return make_id(
        "seg_",
        {
            "episode_id": episode_id,
            "input_fingerprint": input_fingerprint,
            "ordinal": ordinal,
            "slot_anchors": slot_anchors,
        },
    )


def make_job_id(*, kind: str, owner_id: str) -> str:
    return make_id("job_", {"kind": kind, "owner_id": owner_id})


def make_scope_id(*, repository_id: str, scope_kind: str, path_bytes: bytes) -> str:
    return make_id(
        "scope_",
        {
            "path_b64": encode_path_b64(path_bytes),
            "repository_id": repository_id,
            "scope_kind": scope_kind,
        },
    )


def hash_source_version_payload(
    *,
    title: str | None,
    body: str | None,
    author_id: str | None,
    author_name: str | None,
    anchors: dict[str, Any] | None,
    authored_at_us: int | None,
    updated_at_us: int | None,
    lifecycle_fields: dict[str, Any] | None,
) -> str:
    return canonical_sha256(
        {
            "anchors": anchors or {},
            "author_id": author_id,
            "author_name": author_name,
            "authored_at_us": authored_at_us,
            "body": body,
            "lifecycle_fields": lifecycle_fields or {},
            "title": title,
            "updated_at_us": updated_at_us,
        }
    )


def encode_path_b64(path_bytes: bytes) -> str:
    return base64.b64encode(path_bytes).decode("ascii")


def decode_path_b64(b64_text: str) -> bytes:
    if not b64_text or len(b64_text) > 65536:
        raise CodeContextError(
            code="INVALID_REQUEST",
            message="Invalid base64 path length.",
            retryable=False,
        )
    try:
        decoded = base64.b64decode(b64_text.encode("ascii"), validate=True)
    except Exception as exc:
        raise CodeContextError(
            code="INVALID_REQUEST",
            message="Malformed base64 path bytes.",
            retryable=False,
        ) from exc
    if encode_path_b64(decoded) != b64_text:
        raise CodeContextError(
            code="INVALID_REQUEST",
            message="Non-canonical base64 path encoding.",
            retryable=False,
        )
    return decoded


def validate_relative_repo_path_bytes(path_bytes: bytes) -> bytes:
    if not path_bytes or len(path_bytes) > 32768:
        raise CodeContextError(
            code="INVALID_REQUEST",
            message="Target path must be between 1 and 32768 bytes.",
            retryable=False,
        )
    if b"\x00" in path_bytes:
        raise CodeContextError(
            code="INVALID_REQUEST",
            message="Target path must not contain NUL bytes.",
            retryable=False,
        )
    if path_bytes.startswith(b"/") or path_bytes.startswith(b"\\"):
        raise CodeContextError(
            code="INVALID_REQUEST",
            message="Target path must be relative to the repository root.",
            retryable=False,
        )
    if len(path_bytes) >= 2 and path_bytes[1:2] == b":":
        raise CodeContextError(
            code="INVALID_REQUEST",
            message="Target path must not use drive prefixes.",
            retryable=False,
        )
    parts = path_bytes.replace(b"\\", b"/").split(b"/")
    for part in parts:
        if part in (b"", b".", b".."):
            raise CodeContextError(
                code="INVALID_REQUEST",
                message="Target path contains empty, '.', or '..' traversal components.",
                retryable=False,
            )
    return path_bytes


def escape_non_utf8_path(path_bytes: bytes) -> str:
    out: list[str] = []
    i = 0
    n = len(path_bytes)
    while i < n:
        # Try decoding longest valid UTF-8 chunk or escape single byte
        b = path_bytes[i : i + 1]
        if 0x20 <= path_bytes[i] <= 0x7E and path_bytes[i] != 0x5C:
            out.append(b.decode("ascii"))
            i += 1
        elif path_bytes[i] == 0x5C:
            out.append("\\\\")
            i += 1
        else:
            # Check multi-byte UTF-8 sequence
            decoded_char = None
            for length in (4, 3, 2):
                if i + length <= n:
                    try:
                        ch = path_bytes[i : i + length].decode("utf-8", errors="strict")
                        if ch.isprintable():
                            decoded_char = (ch, length)
                        break
                    except UnicodeDecodeError:
                        pass
            if decoded_char is not None:
                out.append(decoded_char[0])
                i += decoded_char[1]
            else:
                out.append(f"\\x{path_bytes[i]:02x}")
                i += 1
    return "".join(out)


def path_bytes_to_display(path_bytes: bytes) -> tuple[str, str | None]:
    try:
        text = path_bytes.decode("utf-8", errors="strict")
        return text, None
    except UnicodeDecodeError:
        return escape_non_utf8_path(path_bytes), encode_path_b64(path_bytes)


def datetime_to_us(dt: datetime) -> int:
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise CodeContextError(
            code="INVALID_REQUEST",
            message="Datetime must be timezone-aware.",
            retryable=False,
        )
    utc_dt = dt.astimezone(UTC)
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    delta = utc_dt - epoch
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


def us_to_datetime(us: int) -> datetime:
    from datetime import timedelta

    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    return epoch + timedelta(microseconds=us)


def us_to_iso_z(us: int) -> str:
    dt = us_to_datetime(us)
    if dt.microsecond == 0:
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_iso_utc(text: str) -> datetime:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty datetime string.")
    s = text.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError("Naive datetime is not permitted.")
    return dt.astimezone(UTC)


def format_iso_z(dt: datetime) -> str:
    return us_to_iso_z(datetime_to_us(dt))


def default_token_estimator(text: str) -> int:
    if not text:
        return 0
    return math.ceil(len(text.encode("utf-8")) / 3)


def build_line_byte_table(blob_bytes: bytes) -> list[tuple[int, int]]:
    """Return 1-indexed line intervals as zero-based half-open (byte_start, byte_end) pairs.

    Line 1 is at index 0. Each line includes its trailing LF or CRLF if present.
    """
    if not blob_bytes:
        return []
    lines: list[tuple[int, int]] = []
    start = 0
    n = len(blob_bytes)
    while start < n:
        nl = blob_bytes.find(b"\n", start)
        if nl == -1:
            lines.append((start, n))
            break
        end = nl + 1
        lines.append((start, end))
        start = end
    return lines


def line_range_to_byte_span(
    blob_bytes: bytes,
    line_start: int,
    line_count: int,
) -> tuple[int, int, int | None, int | None]:
    """Convert Git hunk 1-based line_start and line_count to (byte_start, byte_end, line_start, line_end).

    For line_count == 0 (insertion/deletion boundary):
    - line_start == 0 means boundary at byte 0 (beginning of file)
    - line_start > 0 means boundary immediately after line `line_start`
    Returns (boundary_byte, boundary_byte, None, None).
    """
    if line_count < 0 or line_start < 0:
        raise ValueError(f"Invalid hunk coordinates: start={line_start}, count={line_count}")
    table = build_line_byte_table(blob_bytes)
    if line_count == 0:
        if line_start == 0:
            return (0, 0, None, None)
        if line_start > len(table):
            raise ValueError(f"Hunk boundary line {line_start} exceeds line count {len(table)}")
        boundary = table[line_start - 1][1]
        return (boundary, boundary, None, None)

    if line_start < 1:
        raise ValueError("Non-empty hunk range must have line_start >= 1")
    line_end = line_start + line_count - 1
    if line_end > len(table):
        raise ValueError(f"Hunk line_end {line_end} exceeds blob line count {len(table)}")
    byte_start = table[line_start - 1][0]
    byte_end = table[line_end - 1][1]
    return (byte_start, byte_end, line_start, line_end)


def byte_span_to_lines(blob_bytes: bytes, byte_start: int, byte_end: int) -> tuple[int | None, int | None]:
    if byte_start < 0 or byte_end < byte_start or byte_end > len(blob_bytes):
        raise ValueError(f"Invalid byte span [{byte_start}, {byte_end}) for blob of {len(blob_bytes)} bytes")
    if byte_start == byte_end:
        return (None, None)
    table = build_line_byte_table(blob_bytes)
    if not table:
        return (None, None)
    first_line: int | None = None
    last_line: int | None = None
    for idx, (l_start, l_end) in enumerate(table, start=1):
        if l_end > byte_start and first_line is None:
            first_line = idx
        if l_start < byte_end:
            last_line = idx
        else:
            break
    return (first_line, last_line)


def validate_github_repo_name(name: str) -> str:
    if not isinstance(name, str) or len(name) > 200 or not GITHUB_REPO_RE.match(name):
        raise CodeContextError(
            code="INVALID_REQUEST",
            message=f"Invalid GitHub repository format: {name!r}. Expected OWNER/REPO.",
            retryable=False,
        )
    owner, repo = name.split("/", 1)
    if owner in (".", "..") or repo in (".", ".."):
        raise CodeContextError(
            code="INVALID_REQUEST",
            message=f"Invalid GitHub repository owner/name: {name!r}.",
            retryable=False,
        )
    return f"{owner}/{repo}"
