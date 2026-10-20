from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from codecontext.archive import execute_export, execute_purge
from codecontext.enrichment import execute_enrich
from codecontext.errors import CodeContextError
from codecontext.git import GitMirror
from codecontext.graph import (
    DetectChangesRequest,
    DetectChangesResult,
    GetArchitectureRequest,
    GetArchitectureResult,
    GetCodeSnippetRequest,
    GetCodeSnippetResult,
    QueryGraphRequest,
    QueryGraphResult,
    SearchGraphRequest,
    SearchGraphResult,
    TracePathRequest,
    TracePathResult,
    execute_detect_changes,
    execute_get_architecture,
    execute_get_code_snippet,
    execute_query_graph,
    execute_search_graph,
    execute_trace_path,
)
from codecontext.identity import default_token_estimator
from codecontext.models import (
    ContextPack,
    ContextRequest,
    EnrichReport,
    EnrichRequest,
    ExportReport,
    ExportRequest,
    GetRequest,
    InitRequest,
    ModelAdapter,
    PurgeReport,
    PurgeRequest,
    RecordPage,
    SearchRequest,
    SearchResult,
    Status,
    SyncReport,
    SyncRequest,
)
from codecontext.retrieval import execute_context, execute_get, execute_search
from codecontext.store import Store
from codecontext.sync import execute_sync
from codecontext.workspace import validate_workspace_binding


def _validate_request_model(model_cls: type[Any], request: Any) -> Any:
    if isinstance(request, model_cls):
        return request
    try:
        if isinstance(request, dict):
            return model_cls.model_validate(request)
        return model_cls.model_validate(request)
    except Exception as exc:
        if isinstance(exc, CodeContextError):
            raise
        raise CodeContextError(
            code="INVALID_REQUEST",
            message=f"Invalid {model_cls.__name__}: {exc}",
            retryable=False,
        ) from exc


class Memory:
    """Public SDK facade for CodeContext project memory."""

    def __init__(
        self,
        store: Store,
        *,
        _private_token: object | None = None,
    ) -> None:
        if _private_token is not _INIT_SENTINEL:
            raise CodeContextError(
                code="INVALID_REQUEST",
                message="Use Memory.initialize(store_path, request) or Memory.open(store_path) instead of direct construction.",
                retryable=False,
            )
        self._store = store
        self._mirror = GitMirror(store)
        self._workspace_path: Path | None = None
        self._token_counter: Callable[[str], int] = default_token_estimator
        self._estimator_id: str = "utf8_bytes_div3_v1"
        self._count_kind: str = "estimated"
        self._github_transport: httpx.BaseTransport | None = None
        self._github_token: str | None = None

    @classmethod
    def initialize(cls, store_path: Path | str, request: InitRequest | dict[str, Any]) -> "Memory":
        validated_req = _validate_request_model(InitRequest, request)
        s_path = Path(store_path)
        store = Store.initialize(s_path, validated_req)
        return cls(store, _private_token=_INIT_SENTINEL)

    @classmethod
    def open(cls, store_path: Path | str, *, read_only: bool = False) -> "Memory":
        s_path = Path(store_path)
        # Validate manifest hashes if opening an exported archive with manifest.json
        manifest_file = s_path.expanduser().resolve() / "manifest.json"
        if manifest_file.exists():
            _validate_archive_manifest(s_path.expanduser().resolve(), manifest_file)
        store = Store.open(s_path, read_only=read_only)
        return cls(store, _private_token=_INIT_SENTINEL)

    def _ensure_open(self) -> None:
        self._store._ensure_open()

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> "Memory":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def configure_workspace(self, path: Path | str | None) -> None:
        self._ensure_open()
        if path is None:
            self._workspace_path = None
            return
        ws_path = Path(path)
        resolved = validate_workspace_binding(self._store, ws_path)
        self._workspace_path = resolved

    def configure_tokenizer(
        self,
        counter: Callable[[str], int],
        estimator_id: str,
        *,
        count_kind: str = "tokenizer",
    ) -> None:
        self._ensure_open()
        if not callable(counter) or not estimator_id or len(estimator_id) > 200:
            raise CodeContextError(
                code="INVALID_REQUEST",
                message="Invalid tokenizer callback or estimator_id.",
                retryable=False,
            )
        self._token_counter = counter
        self._estimator_id = estimator_id
        self._count_kind = "tokenizer" if count_kind == "tokenizer" else "estimated"

    def sync(self, request: SyncRequest | dict[str, Any]) -> SyncReport:
        self._ensure_open()
        validated_req = _validate_request_model(SyncRequest, request)
        return execute_sync(
            self._store,
            self._mirror,
            validated_req,
            github_transport=self._github_transport,
            github_token=self._github_token,
        )

    def enrich(
        self,
        request: EnrichRequest | dict[str, Any],
        *,
        model: ModelAdapter,
    ) -> EnrichReport:
        self._ensure_open()
        validated_req = _validate_request_model(EnrichRequest, request)
        return execute_enrich(
            self._store,
            self._mirror,
            validated_req,
            model=model,
            token_counter=self._token_counter,
        )

    def search(self, request: SearchRequest | dict[str, Any]) -> SearchResult:
        self._ensure_open()
        validated_req = _validate_request_model(SearchRequest, request)
        return execute_search(
            self._store,
            self._mirror,
            validated_req,
            token_counter=self._token_counter,
            estimator_id=self._estimator_id,
            count_kind=self._count_kind,
        )

    def get(self, request: GetRequest | dict[str, Any]) -> RecordPage:
        self._ensure_open()
        validated_req = _validate_request_model(GetRequest, request)
        return execute_get(
            self._store,
            self._mirror,
            validated_req,
            token_counter=self._token_counter,
            estimator_id=self._estimator_id,
            count_kind=self._count_kind,
        )

    def context(self, request: ContextRequest | dict[str, Any]) -> ContextPack:
        self._ensure_open()
        validated_req = _validate_request_model(ContextRequest, request)
        return execute_context(
            self._store,
            self._mirror,
            validated_req,
            workspace_path=self._workspace_path,
            token_counter=self._token_counter,
            estimator_id=self._estimator_id,
            count_kind=self._count_kind,
        )

    def status(self) -> Status:
        self._ensure_open()
        return self._store.status(
            token_counter=self._token_counter,
            estimator_id=self._estimator_id,
            count_kind=self._count_kind,
        )

    def export(self, request: ExportRequest | dict[str, Any]) -> ExportReport:
        self._ensure_open()
        validated_req = _validate_request_model(ExportRequest, request)
        return execute_export(self._store, validated_req)

    def purge(self, request: PurgeRequest | dict[str, Any]) -> PurgeReport:
        self._ensure_open()
        validated_req = _validate_request_model(PurgeRequest, request)
        return execute_purge(self._store, validated_req)

    def search_graph(self, request: SearchGraphRequest | dict[str, Any]) -> SearchGraphResult:
        self._ensure_open()
        validated_req = _validate_request_model(SearchGraphRequest, request)
        return execute_search_graph(self._store, validated_req)

    def trace_path(self, request: TracePathRequest | dict[str, Any]) -> TracePathResult:
        self._ensure_open()
        validated_req = _validate_request_model(TracePathRequest, request)
        return execute_trace_path(self._store, validated_req)

    def query_graph(self, request: QueryGraphRequest | dict[str, Any]) -> QueryGraphResult:
        self._ensure_open()
        validated_req = _validate_request_model(QueryGraphRequest, request)
        return execute_query_graph(self._store, validated_req)

    def get_architecture(
        self,
        request: GetArchitectureRequest | dict[str, Any] | None = None,
    ) -> GetArchitectureResult:
        self._ensure_open()
        validated_req = (
            _validate_request_model(GetArchitectureRequest, request)
            if request is not None
            else GetArchitectureRequest()
        )
        return execute_get_architecture(self._store, validated_req)

    def detect_changes(
        self,
        request: DetectChangesRequest | dict[str, Any] | None = None,
    ) -> DetectChangesResult:
        self._ensure_open()
        validated_req = (
            _validate_request_model(DetectChangesRequest, request)
            if request is not None
            else DetectChangesRequest()
        )
        return execute_detect_changes(
            self._store,
            self._mirror,
            validated_req,
            workspace_path=self._workspace_path,
        )

    def get_code_snippet(
        self,
        request: GetCodeSnippetRequest | dict[str, Any],
    ) -> GetCodeSnippetResult:
        self._ensure_open()
        validated_req = _validate_request_model(GetCodeSnippetRequest, request)
        return execute_get_code_snippet(self._store, self._mirror, validated_req)


_INIT_SENTINEL = object()


def _validate_archive_manifest(store_dir: Path, manifest_file: Path) -> None:
    import json

    from codecontext.identity import sha256_bytes, validate_relative_repo_path_bytes

    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise CodeContextError(
                code="STORE_VERSION_UNSUPPORTED",
                message="Unsupported or invalid archive manifest.json.",
                retryable=False,
            )
        files_map = manifest.get("files")
        if not isinstance(files_map, dict):
            raise ValueError("Missing files map in manifest")
        store_root = store_dir.resolve()
        for rel_path, expected_sha in files_map.items():
            if not isinstance(rel_path, str):
                raise CodeContextError(
                    code="STORE_VERSION_UNSUPPORTED",
                    message="Invalid path entry in archive manifest.json.",
                    retryable=False,
                )
            validate_relative_repo_path_bytes(rel_path.encode("utf-8"))
            fp = store_root / rel_path
            if fp.is_symlink():
                raise CodeContextError(
                    code="STORE_VERSION_UNSUPPORTED",
                    message=f"Archive manifest references a symlink: {rel_path}.",
                    retryable=False,
                )
            resolved_fp = fp.resolve()
            if not resolved_fp.is_relative_to(store_root) or resolved_fp == store_root:
                raise CodeContextError(
                    code="STORE_VERSION_UNSUPPORTED",
                    message=f"Archive manifest path escapes store directory: {rel_path}.",
                    retryable=False,
                )
            if not resolved_fp.is_file():
                raise CodeContextError(
                    code="STORE_NOT_INITIALIZED",
                    message=f"Archive is missing expected file {rel_path}.",
                    retryable=False,
                )
            actual_sha = sha256_bytes(resolved_fp.read_bytes())
            if actual_sha != expected_sha:
                raise CodeContextError(
                    code="STORE_VERSION_UNSUPPORTED",
                    message=f"Archive file checksum mismatch for {rel_path}.",
                    retryable=False,
                )
    except CodeContextError:
        raise
    except Exception as exc:
        raise CodeContextError(
            code="STORE_VERSION_UNSUPPORTED",
            message="Failed to validate archive manifest.json.",
            retryable=False,
        ) from exc
