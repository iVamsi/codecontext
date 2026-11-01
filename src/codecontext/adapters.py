import json
import math
import os
import re
import struct
import time
from decimal import Decimal
from typing import Any, ClassVar
from urllib.parse import urlparse

import httpx

from codecontext.errors import CodeContextError
from codecontext.identity import canonical_json, sha256_text
from codecontext.models import (
    EmbeddingAdapter,
    ModelAdapter,
    ModelIdentity,
    ModelRequest,
    ModelResponse,
)

MAX_MODEL_RESPONSE_BYTES = 8 * 1024 * 1024
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
_OPEN_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*[ \t]*\r?\n?")
_CLOSE_FENCE_RE = re.compile(r"\r?\n?[ \t]*```$")
_INLINE_FENCE_RE = re.compile(r"```(?:json|JSON)?[ \t]*\r?\n?([\s\S]*?)\r?\n?[ \t]*```")


def strip_markdown_json_fences(text: str) -> str:
    """Strip ```` ```json ... ``` ```` or ```` ``` ... ``` ```` markdown wrappers cleanly."""
    s = text.strip()
    if not s:
        return s
    if s.startswith("```"):
        inner = _OPEN_FENCE_RE.sub("", s, count=1)
        inner = _CLOSE_FENCE_RE.sub("", inner, count=1)
        return inner.strip()
    if not (s.startswith("{") or s.startswith("[")) and "```" in s:
        m = _INLINE_FENCE_RE.search(s)
        if m is not None:
            candidate = m.group(1).strip()
            if candidate:
                return candidate
    return s


def pack_f32_vector(vec: list[float]) -> bytes:
    """Pack a float vector into little-endian IEEE 754 float32 bytes."""
    if not vec:
        raise ValueError("Embedding vector must not be empty")
    for v in vec:
        if not math.isfinite(v):
            raise ValueError("Embedding vector contains non-finite float value")
    return struct.pack(f"<{len(vec)}f", *vec)


def cosine_similarity_f32(vec_a: list[float], blob_b: bytes) -> float:
    """Compute cosine similarity between a Python float list and a packed float32 BLOB."""
    n = len(vec_a)
    if n == 0 or len(blob_b) != n * 4:
        return 0.0
    vec_b = struct.unpack(f"<{n}f", blob_b)
    dot = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0
    for a, b in zip(vec_a, vec_b, strict=True):
        dot += a * b
        norm_a_sq += a * a
        norm_b_sq += b * b
    if norm_a_sq <= 0.0 or norm_b_sq <= 0.0:
        return 0.0
    denom = math.sqrt(norm_a_sq) * math.sqrt(norm_b_sq)
    if denom <= 0.0 or not math.isfinite(dot):
        return 0.0
    res = dot / denom
    return float(res) if math.isfinite(res) else 0.0


def _is_localhost_url(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0")


def _nonneg_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _request_id(value: Any) -> str | None:
    return value[:200] if isinstance(value, str) and value else None


def _invalid_output(provider_label: str, message: str) -> CodeContextError:
    return CodeContextError(
        code="MODEL_INVALID_OUTPUT",
        message=f"{provider_label} {message}",
        retryable=False,
        details={"provider": provider_label},
    )


def _retry_delay(
    attempt: int,
    resp: httpx.Response | None,
    *,
    base_backoff: float,
    max_sleep: float,
    deadline: float,
) -> float | None:
    """Return how long to wait before the next attempt, or None when the deadline leaves no room.

    Exponential backoff, replaced by the server's Retry-After seconds when present, and
    always capped by `max_sleep` and the time left before the deadline.
    """
    room = deadline - time.monotonic() - 0.05
    if room <= 0.0:
        return None
    delay = base_backoff * (2**attempt)
    if resp is not None:
        raw = resp.headers.get("retry-after")
        if raw is not None:
            try:
                parsed = float(raw.strip())
            except ValueError:
                parsed = math.nan
            if math.isfinite(parsed) and parsed >= 0.0:
                delay = parsed
    return min(delay, max_sleep, room)


def _read_capped(resp: httpx.Response, provider_label: str) -> bytes:
    """Read the body, failing as soon as it passes MAX_MODEL_RESPONSE_BYTES."""
    chunks: list[bytes] = []
    size = 0
    for chunk in resp.iter_bytes():
        size += len(chunk)
        if size > MAX_MODEL_RESPONSE_BYTES:
            raise _invalid_output(
                provider_label, f"response exceeds {MAX_MODEL_RESPONSE_BYTES} bytes."
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _post_json_with_retry(
    *,
    url: str,
    headers: dict[str, str],
    json_body: dict[str, Any],
    timeout_seconds: int,
    transport: httpx.BaseTransport | None,
    provider_label: str,
    max_retries: int = 3,
    passthrough_status_codes: frozenset[int] = frozenset(),
) -> httpx.Response:
    """POST JSON, retrying only timeouts, HTTP 429, and HTTP 5xx.

    Error messages carry the provider label and status, never the URL or headers,
    so an API key cannot leak through them.
    """
    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    # An injected transport means a test double; keep its waits short.
    base_backoff = 0.01 if transport is not None else 0.5
    max_sleep = 0.05 if transport is not None else 8.0
    timed_out = TimeoutError(f"{provider_label} request timed out after {timeout_seconds}s.")

    client_kwargs: dict[str, Any] = {"follow_redirects": False}
    if transport is not None:
        client_kwargs["transport"] = transport

    with httpx.Client(**client_kwargs) as client:
        for attempt in range(max_retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise timed_out
            last_try = attempt == max_retries
            try:
                with client.stream(
                    "POST",
                    url,
                    # The response cap counts decoded bytes, so a compressed body could expand
                    # far past it in one chunk before the cap is checked.
                    headers={**headers, "Accept-Encoding": "identity"},
                    json=json_body,
                    timeout=httpx.Timeout(max(0.05, remaining)),
                ) as streamed:
                    resp = streamed
                    body = (
                        _read_capped(streamed, provider_label)
                        if 200 <= streamed.status_code < 300
                        or streamed.status_code in passthrough_status_codes
                        else b""
                    )
            except httpx.TimeoutException:
                delay = None if last_try else _retry_delay(
                    attempt, None, base_backoff=base_backoff, max_sleep=max_sleep, deadline=deadline
                )
                if delay is None:
                    raise timed_out from None
                time.sleep(delay)
                continue
            except httpx.TransportError as exc:
                raise CodeContextError(
                    code="MODEL_FAILED",
                    message=f"{provider_label} could not be reached ({type(exc).__name__}).",
                    retryable=True,
                    details={"provider": provider_label, "error_type": type(exc).__name__},
                ) from None

            status = resp.status_code
            if 200 <= status < 300 or status in passthrough_status_codes:
                return httpx.Response(status, content=body)
            if status in (401, 403):
                raise CodeContextError(
                    code="AUTH_REQUIRED",
                    message=f"{provider_label} rejected the API key (HTTP {status}).",
                    retryable=False,
                    details={"provider": provider_label, "status_code": status},
                )
            if status in _RETRYABLE_STATUS_CODES and not last_try:
                delay = _retry_delay(
                    attempt, resp, base_backoff=base_backoff, max_sleep=max_sleep, deadline=deadline
                )
                if delay is not None:
                    time.sleep(delay)
                    continue
            if status == 429:
                raise CodeContextError(
                    code="RATE_LIMITED",
                    message=f"{provider_label} rate limit exceeded (HTTP 429).",
                    retryable=True,
                    details={"provider": provider_label, "status_code": 429},
                )
            raise CodeContextError(
                code="MODEL_FAILED",
                message=f"{provider_label} request failed with HTTP {status}.",
                retryable=status >= 500,
                details={"provider": provider_label, "status_code": status},
            )
    raise timed_out


def _decode_json_response(resp: httpx.Response, provider_label: str) -> dict[str, Any]:
    try:
        payload = resp.json()
    except Exception:
        raise _invalid_output(provider_label, "returned a response that is not JSON.") from None
    if not isinstance(payload, dict):
        raise _invalid_output(provider_label, "returned an unexpected JSON root type.")
    return payload


class _HttpClient:
    """Shared configuration for built-in HTTP adapters.

    Keys come only from environment variables and are kept out of `repr`, errors,
    and identity fingerprints.
    """

    PROVIDER: ClassVar[str]
    DEFAULT_BASE_URL: ClassVar[str]
    BASE_URL_ENV: ClassVar[str | None] = None
    KEY_ENV: ClassVar[tuple[str, ...]] = ()
    KEY_HEADER: ClassVar[str | None] = None
    KEY_IS_BEARER: ClassVar[bool] = False
    KEY_OPTIONAL_ON_LOCALHOST: ClassVar[bool] = False

    def __init__(
        self,
        model: str | None,
        *,
        base_url: str | None,
        transport: httpx.BaseTransport | None,
    ) -> None:
        name = (model or "").strip()
        if not name:
            raise CodeContextError(
                code="INVALID_REQUEST",
                message=f"A model name is required for the {self.PROVIDER} provider. There is no default model.",
                retryable=False,
                details={"provider": self.PROVIDER},
            )
        self._model = self._normalize_model(name)
        env_url = os.environ.get(self.BASE_URL_ENV, "").strip() if self.BASE_URL_ENV else ""
        url = (base_url or env_url or self.DEFAULT_BASE_URL).strip()
        if "://" not in url:
            url = f"http://{url}"
        self._base_url = url.rstrip("/")
        self._transport = transport

    @staticmethod
    def _normalize_model(name: str) -> str:
        return name

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self._model!r}, base_url={self._base_url!r})"

    def _api_key(self) -> str:
        for var in self.KEY_ENV:
            value = os.environ.get(var, "").strip()
            if value:
                return value
        return ""

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.KEY_HEADER is None:
            return headers
        key = self._api_key()
        local = _is_localhost_url(self._base_url)
        if not key:
            if self.KEY_OPTIONAL_ON_LOCALHOST and local:
                return headers
            raise CodeContextError(
                code="AUTH_REQUIRED",
                message=f"{self.PROVIDER} needs an API key. Set {' or '.join(self.KEY_ENV)}.",
                retryable=False,
                details={"provider": self.PROVIDER},
            )
        if urlparse(self._base_url).scheme != "https" and not local:
            raise CodeContextError(
                code="INVALID_REQUEST",
                message=f"{self.PROVIDER} base URL must use https when an API key is sent to a remote host.",
                retryable=False,
                details={"provider": self.PROVIDER},
            )
        headers[self.KEY_HEADER] = f"Bearer {key}" if self.KEY_IS_BEARER else key
        return headers

    def _post(
        self,
        url: str,
        body: dict[str, Any],
        *,
        timeout_seconds: int,
        label: str,
        passthrough_status_codes: frozenset[int] = frozenset(),
    ) -> httpx.Response:
        return _post_json_with_retry(
            url=url,
            headers=self._headers(),
            json_body=body,
            timeout_seconds=timeout_seconds,
            transport=self._transport,
            provider_label=label,
            passthrough_status_codes=passthrough_status_codes,
        )


class _OpenAIConfig(_HttpClient):
    PROVIDER = "openai_compatible"
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    BASE_URL_ENV = "OPENAI_BASE_URL"
    KEY_ENV = ("OPENAI_API_KEY",)
    KEY_HEADER = "Authorization"
    KEY_IS_BEARER = True
    KEY_OPTIONAL_ON_LOCALHOST = True


class _GeminiConfig(_HttpClient):
    PROVIDER = "gemini"
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
    KEY_ENV = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    KEY_HEADER = "x-goog-api-key"

    @staticmethod
    def _normalize_model(name: str) -> str:
        return name.removeprefix("models/")

    def _method_url(self, method: str) -> str:
        base = self._base_url if self._base_url.endswith("/v1beta") else f"{self._base_url}/v1beta"
        return f"{base}/models/{self._model}:{method}"


class _OllamaConfig(_HttpClient):
    PROVIDER = "ollama"
    DEFAULT_BASE_URL = "http://127.0.0.1:11434"
    BASE_URL_ENV = "OLLAMA_HOST"


class _AnthropicConfig(_HttpClient):
    PROVIDER = "anthropic"
    DEFAULT_BASE_URL = "https://api.anthropic.com"
    BASE_URL_ENV = "ANTHROPIC_BASE_URL"
    KEY_ENV = ("ANTHROPIC_API_KEY",)
    KEY_HEADER = "x-api-key"


class _ModelAdapterBase(_HttpClient):
    """Builds the ModelIdentity shared by every built-in chat adapter."""

    FRAMING_INPUT_TOKENS: ClassVar[int] = 128

    def __init__(
        self,
        model: str | None,
        *,
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
        model_revision: str = "builtin-v1",
        input_price_per_million: Decimal | None = None,
        output_price_per_million: Decimal | None = None,
    ) -> None:
        super().__init__(model, base_url=base_url, transport=transport)
        cfg_fp = sha256_text(
            canonical_json(
                {
                    "adapter": self.PROVIDER,
                    "base_url": self._base_url,
                    "model": self._model,
                    "temperature": 0,
                    **self._extra_config(),
                }
            )
        )
        self._identity = ModelIdentity(
            adapter_name=self.PROVIDER,
            model_name=self._model,
            model_revision=model_revision,
            configuration_fingerprint=cfg_fp,
            input_price_per_million=input_price_per_million,
            output_price_per_million=output_price_per_million,
            framing_input_tokens=self.FRAMING_INPUT_TOKENS,
        )

    def _extra_config(self) -> dict[str, str]:
        return {}

    @property
    def identity(self) -> ModelIdentity:
        return self._identity


class OpenAICompatibleAdapter(_OpenAIConfig, _ModelAdapterBase):
    """Built-in ModelAdapter for OpenAI and OpenAI-compatible chat completions APIs."""

    def complete(self, request: ModelRequest) -> ModelResponse:
        label = self.PROVIDER
        url = f"{self._base_url}/chat/completions"
        common: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": request.system_text},
                {"role": "user", "content": request.input_json},
            ],
            "temperature": 0,
            "max_tokens": request.max_output_tokens,
        }
        strict_format = {
            "type": "json_schema",
            "json_schema": {"name": "codecontext_output", "schema": request.output_schema},
        }
        resp = self._post(
            url,
            {**common, "response_format": strict_format},
            timeout_seconds=request.timeout_seconds,
            label=label,
            passthrough_status_codes=frozenset({400}),
        )
        if resp.status_code == 400:
            # Many compatible servers reject json_schema but accept plain JSON mode.
            resp = self._post(
                url,
                {**common, "response_format": {"type": "json_object"}},
                timeout_seconds=request.timeout_seconds,
                label=label,
            )

        data = _decode_json_response(resp, label)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise _invalid_output(label, "response missing choices.")
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        raw_content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(raw_content, list):
            raw_text = "".join(
                part["text"]
                for part in raw_content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        elif isinstance(raw_content, str):
            raw_text = raw_content
        else:
            raise _invalid_output(label, "response choice missing string content.")

        cleaned = strip_markdown_json_fences(raw_text)
        if not cleaned:
            raise _invalid_output(label, "returned empty output content.")

        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return ModelResponse(
            output_json=cleaned,
            input_tokens=_nonneg_int(usage.get("prompt_tokens")),
            output_tokens=_nonneg_int(usage.get("completion_tokens")),
            provider_request_id=_request_id(data.get("id")),
        )


class AnthropicAdapter(_AnthropicConfig, _ModelAdapterBase):
    """Built-in ModelAdapter for the Anthropic Messages API (`/v1/messages`)."""

    FRAMING_INPUT_TOKENS = 192
    ANTHROPIC_VERSION: ClassVar[str] = "2023-06-01"

    def _extra_config(self) -> dict[str, str]:
        return {"anthropic_version": self.ANTHROPIC_VERSION}

    def _headers(self) -> dict[str, str]:
        return {**super()._headers(), "anthropic-version": self.ANTHROPIC_VERSION}

    def complete(self, request: ModelRequest) -> ModelResponse:
        label = self.PROVIDER
        base = self._base_url
        url = f"{base}/messages" if base.endswith("/v1") else f"{base}/v1/messages"
        body: dict[str, Any] = {
            "model": self._model,
            "max_tokens": request.max_output_tokens,
            "temperature": 0,
            "system": request.system_text,
            "tools": [
                {
                    "name": "codecontext_output",
                    "description": "Emit structured JSON output conforming to the schema.",
                    "input_schema": request.output_schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": "codecontext_output"},
            "messages": [{"role": "user", "content": request.input_json}],
        }
        resp = self._post(url, body, timeout_seconds=request.timeout_seconds, label=label)
        data = _decode_json_response(resp, label)
        content_blocks = data.get("content")
        if not isinstance(content_blocks, list) or not content_blocks:
            raise _invalid_output(label, "response missing content blocks.")

        extracted_json: str | None = None
        text_parts: list[str] = []
        for blk in content_blocks:
            if not isinstance(blk, dict):
                continue
            btype = blk.get("type")
            if btype == "tool_use" and "input" in blk:
                tool_input = blk["input"]
                if isinstance(tool_input, str):
                    extracted_json = strip_markdown_json_fences(tool_input)
                else:
                    extracted_json = json.dumps(tool_input, separators=(",", ":"), ensure_ascii=False)
                break
            if btype == "text" and isinstance(blk.get("text"), str):
                text_parts.append(blk["text"])

        if extracted_json is None and text_parts:
            extracted_json = strip_markdown_json_fences("".join(text_parts))
        if not extracted_json:
            raise _invalid_output(label, "response contained no structured JSON output.")

        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return ModelResponse(
            output_json=extracted_json,
            input_tokens=_nonneg_int(usage.get("input_tokens")),
            output_tokens=_nonneg_int(usage.get("output_tokens")),
            provider_request_id=_request_id(data.get("id")),
        )


class GeminiAdapter(_GeminiConfig, _ModelAdapterBase):
    """Built-in ModelAdapter for the Google Gemini REST API (`:generateContent`)."""

    def complete(self, request: ModelRequest) -> ModelResponse:
        label = self.PROVIDER
        body: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": request.system_text}]},
            "contents": [{"role": "user", "parts": [{"text": request.input_json}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0,
                "maxOutputTokens": request.max_output_tokens,
            },
        }
        resp = self._post(
            self._method_url("generateContent"),
            body,
            timeout_seconds=request.timeout_seconds,
            label=label,
        )
        data = _decode_json_response(resp, label)
        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise _invalid_output(label, "response missing candidates.")
        cand0 = candidates[0] if isinstance(candidates[0], dict) else {}
        content = cand0.get("content") if isinstance(cand0.get("content"), dict) else {}
        parts = content.get("parts") if isinstance(content.get("parts"), list) else []
        cleaned = strip_markdown_json_fences(
            "".join(p["text"] for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str))
        )
        if not cleaned:
            raise _invalid_output(label, "candidate returned empty text output.")

        usage = data.get("usageMetadata") if isinstance(data.get("usageMetadata"), dict) else {}
        in_toks = _nonneg_int(usage.get("promptTokenCount"))
        out_toks = _nonneg_int(usage.get("candidatesTokenCount"))
        # Gemini omits candidatesTokenCount when the candidate produced no tokens.
        if in_toks is not None and usage.get("candidatesTokenCount") is None:
            out_toks = 0
        return ModelResponse(
            output_json=cleaned,
            input_tokens=in_toks,
            output_tokens=out_toks,
            provider_request_id=_request_id(data.get("responseId")),
        )


class OllamaAdapter(_OllamaConfig, _ModelAdapterBase):
    """Built-in ModelAdapter for local Ollama (`/api/chat`)."""

    def complete(self, request: ModelRequest) -> ModelResponse:
        label = self.PROVIDER
        body: dict[str, Any] = {
            "model": self._model,
            "stream": False,
            "format": request.output_schema,
            "options": {"temperature": 0, "num_predict": request.max_output_tokens},
            "messages": [
                {"role": "system", "content": request.system_text},
                {"role": "user", "content": request.input_json},
            ],
        }
        resp = self._post(
            f"{self._base_url}/api/chat", body, timeout_seconds=request.timeout_seconds, label=label
        )
        data = _decode_json_response(resp, label)
        msg = data.get("message")
        raw_content = msg.get("content") if isinstance(msg, dict) else data.get("response")
        if not isinstance(raw_content, str):
            raise _invalid_output(label, "response missing message.content.")
        cleaned = strip_markdown_json_fences(raw_content)
        if not cleaned:
            raise _invalid_output(label, "returned empty message content.")
        return ModelResponse(
            output_json=cleaned,
            input_tokens=_nonneg_int(data.get("prompt_eval_count")),
            output_tokens=_nonneg_int(data.get("eval_count")),
            provider_request_id=None,
        )


def _normalize_embedding_matrix(
    raw_vectors: Any, expected_count: int, provider_label: str
) -> list[list[float]]:
    if not isinstance(raw_vectors, list) or len(raw_vectors) != expected_count:
        got = len(raw_vectors) if isinstance(raw_vectors, list) else "invalid"
        raise _invalid_output(provider_label, f"returned {got} embeddings; expected {expected_count}.")
    matrix: list[list[float]] = []
    dim: int | None = None
    for row in raw_vectors:
        if not isinstance(row, list) or not row:
            raise _invalid_output(provider_label, "returned an empty or non-list embedding vector.")
        if dim is None:
            dim = len(row)
        elif len(row) != dim:
            raise _invalid_output(provider_label, "returned inconsistent embedding dimensions.")
        vec: list[float] = []
        for val in row:
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
                raise _invalid_output(provider_label, "embedding contains non-numeric or non-finite values.")
            vec.append(float(val))
        matrix.append(vec)
    return matrix


class _EmbeddingAdapterBase(_HttpClient):
    """Shared EmbeddingAdapter surface. `dimensions` is 0 until the first response."""

    def __init__(
        self,
        model: str | None,
        *,
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(model, base_url=base_url, transport=transport)
        self._dimensions = 0

    @property
    def model_fingerprint(self) -> str:
        return f"{self.PROVIDER}:{self._model}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: list[str], *, timeout_seconds: int = 60) -> list[list[float]]:
        if not texts:
            return []
        label = f"{self.PROVIDER}_embeddings"
        matrix = _normalize_embedding_matrix(self._fetch(texts, timeout_seconds, label), len(texts), label)
        self._dimensions = len(matrix[0])
        return matrix

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def _fetch(self, texts: list[str], timeout_seconds: int, label: str) -> Any:
        raise NotImplementedError


class OpenAICompatibleEmbeddingAdapter(_OpenAIConfig, _EmbeddingAdapterBase):
    """Built-in EmbeddingAdapter for OpenAI and OpenAI-compatible `/embeddings`."""

    def _fetch(self, texts: list[str], timeout_seconds: int, label: str) -> Any:
        resp = self._post(
            f"{self._base_url}/embeddings",
            {"model": self._model, "input": texts},
            timeout_seconds=timeout_seconds,
            label=label,
        )
        items = _decode_json_response(resp, label).get("data")
        if not isinstance(items, list):
            raise _invalid_output(label, "response missing data array.")
        ordered = sorted(
            (it for it in items if isinstance(it, dict)),
            key=lambda it: it["index"] if isinstance(it.get("index"), int) else 0,
        )
        return [it.get("embedding") for it in ordered]


class GeminiEmbeddingAdapter(_GeminiConfig, _EmbeddingAdapterBase):
    """Built-in EmbeddingAdapter for Google Gemini `:batchEmbedContents`."""

    def _fetch(self, texts: list[str], timeout_seconds: int, label: str) -> Any:
        reqs = [{"model": f"models/{self._model}", "content": {"parts": [{"text": t}]}} for t in texts]
        resp = self._post(
            self._method_url("batchEmbedContents"),
            {"requests": reqs},
            timeout_seconds=timeout_seconds,
            label=label,
        )
        embs = _decode_json_response(resp, label).get("embeddings")
        if not isinstance(embs, list):
            raise _invalid_output(label, "response missing embeddings array.")
        return [e.get("values") if isinstance(e, dict) else None for e in embs]


class OllamaEmbeddingAdapter(_OllamaConfig, _EmbeddingAdapterBase):
    """Built-in EmbeddingAdapter for local Ollama `/api/embed`."""

    def _fetch(self, texts: list[str], timeout_seconds: int, label: str) -> Any:
        resp = self._post(
            f"{self._base_url}/api/embed",
            {"model": self._model, "input": texts},
            timeout_seconds=timeout_seconds,
            label=label,
        )
        return _decode_json_response(resp, label).get("embeddings")


_PROVIDERS: dict[str, tuple[type[_ModelAdapterBase], type[_EmbeddingAdapterBase] | None]] = {
    "openai": (OpenAICompatibleAdapter, OpenAICompatibleEmbeddingAdapter),
    "anthropic": (AnthropicAdapter, None),
    "gemini": (GeminiAdapter, GeminiEmbeddingAdapter),
    "ollama": (OllamaAdapter, OllamaEmbeddingAdapter),
}
_PROVIDER_ALIASES = {"openai_compatible": "openai", "claude": "anthropic", "google": "gemini"}

PROVIDER_NAMES: tuple[str, ...] = tuple(_PROVIDERS)
EMBEDDING_PROVIDER_NAMES: tuple[str, ...] = tuple(k for k, (_, emb) in _PROVIDERS.items() if emb)


def _provider_entry(provider: str, *, embedding: bool) -> Any:
    norm = provider.strip().lower().replace("-", "_")
    entry = _PROVIDERS.get(_PROVIDER_ALIASES.get(norm, norm))
    cls = None if entry is None else entry[1 if embedding else 0]
    if cls is None:
        names = EMBEDDING_PROVIDER_NAMES if embedding else PROVIDER_NAMES
        kind = "embedding provider" if embedding else "model provider"
        raise CodeContextError(
            code="INVALID_REQUEST",
            message=f"Unknown {kind} {provider!r}. Choose one of: {', '.join(names)}.",
            retryable=False,
            details={"provider": provider[:100]},
        )
    return cls


def create_builtin_adapter(
    provider: str,
    model: str | None,
    *,
    base_url: str | None = None,
    transport: httpx.BaseTransport | None = None,
    input_price_per_million: Decimal | None = None,
    output_price_per_million: Decimal | None = None,
) -> ModelAdapter:
    """Create a built-in httpx ModelAdapter. The model is required; keys come from the environment."""
    cls = _provider_entry(provider, embedding=False)
    return cls(
        model,
        base_url=base_url,
        transport=transport,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
    )


def create_builtin_embedding_adapter(
    provider: str,
    model: str | None,
    *,
    base_url: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> EmbeddingAdapter:
    """Create a built-in httpx EmbeddingAdapter. The model is required; keys come from the environment."""
    cls = _provider_entry(provider, embedding=True)
    return cls(model, base_url=base_url, transport=transport)
