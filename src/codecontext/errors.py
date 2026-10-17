import json
from typing import Any, Literal

ErrorCode = Literal[
    "INVALID_REQUEST",
    "STORE_NOT_INITIALIZED",
    "STORE_VERSION_UNSUPPORTED",
    "DEPENDENCY_MISSING",
    "REPOSITORY_MISMATCH",
    "AUTH_REQUIRED",
    "ACCESS_BLOCKED",
    "RATE_LIMITED",
    "STORE_BUSY",
    "REVISION_NOT_PREPARED",
    "RECORD_NOT_FOUND",
    "RECORD_NOT_ELIGIBLE",
    "AMBIGUOUS_TARGET",
    "CURSOR_STALE",
    "MODEL_INVALID_OUTPUT",
    "MODEL_FAILED",
    "BUDGET_EXHAUSTED",
    "IO_ERROR",
]

VALID_ERROR_CODES: frozenset[str] = frozenset(
    {
        "INVALID_REQUEST",
        "STORE_NOT_INITIALIZED",
        "STORE_VERSION_UNSUPPORTED",
        "DEPENDENCY_MISSING",
        "REPOSITORY_MISMATCH",
        "AUTH_REQUIRED",
        "ACCESS_BLOCKED",
        "RATE_LIMITED",
        "STORE_BUSY",
        "REVISION_NOT_PREPARED",
        "RECORD_NOT_FOUND",
        "RECORD_NOT_ELIGIBLE",
        "AMBIGUOUS_TARGET",
        "CURSOR_STALE",
        "MODEL_INVALID_OUTPUT",
        "MODEL_FAILED",
        "BUDGET_EXHAUSTED",
        "IO_ERROR",
    }
)

FORBIDDEN_DETAIL_KEYS = frozenset(
    {
        "token",
        "github_token",
        "authorization",
        "password",
        "secret",
        "credential",
        "api_key",
        "raw_payload",
        "response_body",
        "prompt",
    }
)


def _sanitize_value(val: Any, depth: int = 0) -> Any:
    if depth >= 3:
        if isinstance(val, (str, int, float, bool)) or val is None:
            return val[:256] if isinstance(val, str) else val
        return str(val)[:256]
    if val is None or isinstance(val, (bool, int, float)):
        return val
    if isinstance(val, str):
        return val[:1000]
    if isinstance(val, (list, tuple)):
        return [_sanitize_value(item, depth + 1) for item in val[:50]]
    if isinstance(val, dict):
        out: dict[str, Any] = {}
        for k, v in list(val.items())[:50]:
            key_str = str(k)[:100]
            if key_str.lower() in FORBIDDEN_DETAIL_KEYS:
                continue
            out[key_str] = _sanitize_value(v, depth + 1)
        return out
    return str(val)[:500]


def sanitize_error_details(details: dict[str, Any] | None) -> dict[str, Any]:
    if not details:
        return {}
    cleaned = _sanitize_value(details, depth=0)
    if not isinstance(cleaned, dict):
        return {}
    encoded = json.dumps(cleaned, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    if len(encoded) <= 4096:
        return cleaned
    trimmed: dict[str, Any] = {}
    for k, v in cleaned.items():
        candidate = {**trimmed, k: v}
        if len(json.dumps(candidate, separators=(",", ":"), ensure_ascii=True).encode("utf-8")) <= 4000:
            trimmed[k] = v
        else:
            trimmed["truncated_details"] = True
            break
    return trimmed


class CodeContextError(Exception):
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        if code not in VALID_ERROR_CODES:
            raise ValueError(f"Unsupported error code: {code}")
        super().__init__(message)
        self.code: ErrorCode = code
        self.message: str = message[:1000]
        self.retryable: bool = bool(retryable)
        self.details: dict[str, Any] = sanitize_error_details(details)

    def to_envelope(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
                "details": self.details,
            }
        }

    def to_json(self, *, pretty: bool = False) -> str:
        envelope = self.to_envelope()
        if pretty:
            return json.dumps(envelope, indent=2, sort_keys=True)
        return json.dumps(envelope, separators=(",", ":"), sort_keys=True, ensure_ascii=True)
