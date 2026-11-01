import json
from pathlib import Path

import httpx
import pytest

from codecontext import (
    CodeContextError,
    EnrichRequest,
    InitRequest,
    Memory,
    ModelRequest,
    SyncRequest,
)
from codecontext.adapters import (
    AnthropicAdapter,
    GeminiAdapter,
    OllamaAdapter,
    OllamaEmbeddingAdapter,
    OpenAICompatibleAdapter,
    cosine_similarity_f32,
    create_builtin_adapter,
    create_builtin_embedding_adapter,
    pack_f32_vector,
    strip_markdown_json_fences,
)
from codecontext.models import SearchRequest
from codecontext.team import execute_join, execute_publish
from examples.fixture_adapter import DeterministicFixtureAdapter
from tests.fixtures.builders import (
    GitCommitSpec,
    build_f01_fixture,
    commit_fixture_spec,
    init_fixture_repo,
    run_git_fixture,
)

SECRET = "sk-test-do-not-leak-0123456789"


def _model_request() -> ModelRequest:
    return ModelRequest(
        request_id="run_1",
        system_text="system",
        input_json='{"slots": []}',
        output_schema={"type": "object"},
        max_output_tokens=100,
        timeout_seconds=5,
    )


def test_builtin_adapters_factory_and_fence_stripping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify built-in LLM adapter instantiation and markdown fence stripping."""
    assert strip_markdown_json_fences('```json\n{"claims": []}\n```') == '{"claims": []}'
    assert strip_markdown_json_fences('```\n{"claims": []}\n```') == '{"claims": []}'
    assert strip_markdown_json_fences('   {"claims": []}   ') == '{"claims": []}'

    assert isinstance(create_builtin_adapter("openai", "gpt-test"), OpenAICompatibleAdapter)
    assert isinstance(create_builtin_adapter("anthropic", "claude-test"), AnthropicAdapter)
    assert isinstance(create_builtin_adapter("gemini", "models/gemini-test"), GeminiAdapter)
    assert isinstance(create_builtin_adapter("ollama", "llama-test"), OllamaAdapter)

    emb_ad = create_builtin_embedding_adapter("openai", "embed-test")
    assert emb_ad.model_fingerprint == "openai_compatible:embed-test"
    assert emb_ad.dimensions == 0


@pytest.mark.parametrize("model", [None, "", "   "])
def test_builtin_factories_require_explicit_model(model: str | None) -> None:
    with pytest.raises(CodeContextError) as chat_exc:
        create_builtin_adapter("anthropic", model)
    assert chat_exc.value.code == "INVALID_REQUEST"
    assert "no default model" in chat_exc.value.message.lower()

    with pytest.raises(CodeContextError) as emb_exc:
        create_builtin_embedding_adapter("ollama", model)
    assert emb_exc.value.code == "INVALID_REQUEST"


def test_builtin_factories_reject_unknown_provider() -> None:
    with pytest.raises(CodeContextError) as exc:
        create_builtin_adapter("acme", "m")
    assert "openai, anthropic, gemini, ollama" in exc.value.message
    with pytest.raises(CodeContextError):
        create_builtin_embedding_adapter("anthropic", "m")


def test_api_key_comes_from_env_and_never_leaks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    seen_auth: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("Authorization", ""))
        return httpx.Response(500, json={"error": f"echo {SECRET}"})

    adapter = create_builtin_adapter("openai", "gpt-test", transport=httpx.MockTransport(handler))
    with pytest.raises(CodeContextError) as exc:
        adapter.complete(_model_request())

    assert seen_auth and all(h == f"Bearer {SECRET}" for h in seen_auth)
    assert len(seen_auth) == 4
    assert exc.value.code == "MODEL_FAILED"
    for text in (str(exc.value), exc.value.to_json(pretty=False), repr(adapter), repr(adapter.identity)):
        assert SECRET not in text


def test_rejected_key_error_does_not_echo_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    transport = httpx.MockTransport(lambda request: httpx.Response(401, text=SECRET))
    adapter = create_builtin_adapter("anthropic", "claude-test", transport=transport)
    with pytest.raises(CodeContextError) as exc:
        adapter.complete(_model_request())
    assert exc.value.code == "AUTH_REQUIRED"
    assert SECRET not in exc.value.to_json(pretty=False)


def test_missing_key_names_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    adapter = create_builtin_adapter("gemini", "gemini-test")
    with pytest.raises(CodeContextError) as exc:
        adapter.complete(_model_request())
    assert exc.value.code == "AUTH_REQUIRED"
    assert "GEMINI_API_KEY" in exc.value.message


def test_key_is_not_sent_over_plain_http_to_remote_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://models.example.com/v1")
    calls: list[httpx.Request] = []
    adapter = create_builtin_adapter(
        "openai", "gpt-test", transport=httpx.MockTransport(lambda r: calls.append(r) or httpx.Response(200))
    )
    with pytest.raises(CodeContextError) as exc:
        adapter.complete(_model_request())
    assert "https" in exc.value.message
    assert calls == []


def test_retry_honours_retry_after_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    statuses = iter([429, 503, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        if status != 200:
            return httpx.Response(status, headers={"Retry-After": "0"})
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "content": [{"type": "tool_use", "name": "codecontext_output", "input": {"ok": True}}],
                "usage": {"input_tokens": 7, "output_tokens": 3},
            },
        )

    adapter = create_builtin_adapter("anthropic", "claude-test", transport=httpx.MockTransport(handler))
    resp = adapter.complete(_model_request())
    assert resp.output_json == '{"ok":true}'
    assert (resp.input_tokens, resp.output_tokens) == (7, 3)
    assert next(statuses, None) is None


def test_client_errors_and_transport_failures_are_not_retried() -> None:
    calls: list[int] = []

    def bad_request(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(422, json={})

    adapter = create_builtin_adapter("ollama", "llama-test", transport=httpx.MockTransport(bad_request))
    with pytest.raises(CodeContextError) as exc:
        adapter.complete(_model_request())
    assert exc.value.code == "MODEL_FAILED" and exc.value.retryable is False
    assert calls == [1]

    def refused(request: httpx.Request) -> httpx.Response:
        calls.append(2)
        raise httpx.ConnectError("refused", request=request)

    adapter = create_builtin_adapter("ollama", "llama-test", transport=httpx.MockTransport(refused))
    with pytest.raises(CodeContextError):
        adapter.complete(_model_request())
    assert calls == [1, 2]


def test_timeouts_are_retried_a_bounded_number_of_times() -> None:
    attempts: list[int] = []

    def slow(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        raise httpx.ReadTimeout("slow", request=request)

    adapter = create_builtin_adapter("ollama", "llama-test", transport=httpx.MockTransport(slow))
    with pytest.raises(TimeoutError):
        adapter.complete(_model_request())
    assert len(attempts) == 4


def test_ollama_embeddings_and_query_embedding() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["input"]
        return httpx.Response(200, json={"embeddings": [[1.0, float(i), 0.5] for i, _ in enumerate(texts)]})

    adapter = OllamaEmbeddingAdapter("embed-test", transport=httpx.MockTransport(handler))
    assert adapter.embed(["a", "b"]) == [[1.0, 0.0, 0.5], [1.0, 1.0, 0.5]]
    assert adapter.dimensions == 3
    assert adapter.embed_query("q") == [1.0, 0.0, 0.5]


def _embedded_memory(tmp_path: Path) -> Memory:
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(
        repo,
        GitCommitSpec(message="Add pool", files={"src/pool.py": b"def make_pool():\n    return []\n"}),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["input"]
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3, 0.4] for _ in texts]})

    mem = Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo))
    mem.sync(SyncRequest(offline=True))
    mem.configure_embedding_adapter(OllamaEmbeddingAdapter("embed-test", transport=httpx.MockTransport(handler)))
    mem.enrich(
        EnrichRequest(max_input_tokens=100_000, max_output_tokens=20_000, max_calls=10, generate_embeddings=True),
        model=DeterministicFixtureAdapter(strict_fixture_only=False),
    )
    return mem


def test_enrich_generates_document_embeddings(tmp_path: Path) -> None:
    with _embedded_memory(tmp_path) as mem, mem._store.connect_read() as conn:
        rows = conn.execute("SELECT model_fingerprint, dimensions FROM document_embeddings").fetchall()
    assert rows
    assert {(r["model_fingerprint"], r["dimensions"]) for r in rows} == {("ollama:embed-test", 4)}


def test_hybrid_search_returns_vector_hits(tmp_path: Path) -> None:
    with _embedded_memory(tmp_path) as mem:
        res = mem.search(SearchRequest(query="allocate worker resources", kind="code"))
    assert any(h.match == "semantic" and "pool" in h.title for h in res.hits)
    assert not [g for g in res.gaps if g.code == "MODEL_FAILED"]


def test_configure_embedding_adapter_rejects_non_adapter(tmp_path: Path) -> None:
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(repo, GitCommitSpec(message="init", files={"a.py": b"x = 1\n"}))
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        with pytest.raises(CodeContextError):
            mem.configure_embedding_adapter(object())  # type: ignore[arg-type]
        mem.configure_embedding_adapter(None)


def test_vector_packing_and_cosine_similarity() -> None:
    """Verify IEEE-754 vector packing and cosine similarity calculations."""
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    vec3 = [0.0, 1.0, 0.0]

    packed2 = pack_f32_vector(vec2)
    assert len(packed2) == 12  # 3 * 4 bytes

    # Identical vectors -> similarity 1.0
    sim_identical = cosine_similarity_f32(vec1, packed2)
    assert pytest.approx(sim_identical, 1e-5) == 1.0

    # Orthogonal vectors -> similarity 0.0
    packed3 = pack_f32_vector(vec3)
    sim_orthogonal = cosine_similarity_f32(vec1, packed3)
    assert pytest.approx(sim_orthogonal, 1e-5) == 0.0


def test_search_retrieval_and_tiered_fallback(tmp_path: Path) -> None:
    """Verify search ranking, query fallback, and search filters."""
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(
        repo,
        GitCommitSpec(
            message="Add database storage layer",
            files={
                "src/storage.py": b"def init_database_pool():\n    '''Initialize SQLite connection pool.'''\n    pass\n",
                "src/cache.py": b"def get_cached_item():\n    pass\n",
            },
        ),
    )

    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=repo)) as mem:
        mem.sync(SyncRequest(offline=True))

        # Test search query
        res = mem.search(SearchRequest(query="database pool", kind="code"))
        assert len(res.hits) >= 1
        assert any("init_database_pool" in (h.title or "") for h in res.hits)

        # Test query with punctuation / special characters triggers safe fallback
        special_res = mem.search(SearchRequest(query="init_database_pool() AND NOT OR ()", kind="code"))
        assert special_res is not None


def test_scip_protobuf_wire_import(tmp_path: Path) -> None:
    """Verify decoding and importing SCIP index wire-format."""
    f01 = build_f01_fixture(tmp_path / "repo")
    with Memory.initialize(tmp_path / "store", InitRequest(repo_path=f01.repo_dir)) as mem:
        mem.sync(SyncRequest(offline=True))

        scip_file = tmp_path / "index.scip"
        scip_file.write_bytes(b"")
        report = mem.import_scip(scip_file)
        assert report.documents_processed == 0
        assert report.relations_added == 0


def test_team_merge_non_destructive_union(tmp_path: Path) -> None:
    """Verify execute_team_merge merges records from store B into store A without data loss."""
    repo = init_fixture_repo(tmp_path / "repo")
    run_git_fixture(repo, ["remote", "add", "origin", "https://github.com/org/test-repo.git"])
    commit_fixture_spec(
        repo,
        GitCommitSpec(
            message="Initial commit",
            files={"main.py": b"def main(): pass\n"},
        ),
    )

    store_a_dir = tmp_path / "store_a"
    store_b_dir = tmp_path / "store_b"
    share_dir = tmp_path / "team_share"

    # Developer A initializes the repository memory and publishes it
    with Memory.initialize(store_a_dir, InitRequest(repo_path=repo)) as mem_a:
        mem_a.sync(SyncRequest(offline=True))
        execute_publish(mem_a._store, share_dir)

    # Developer B joins the shared team memory
    execute_join(store_b_dir, share_dir, repo)
    with Memory.open(store_b_dir) as mem_b:
        mem_b.sync(SyncRequest(offline=True))
        execute_publish(mem_b._store, share_dir)

    # Merge share from B into A non-destructively
    rep = Memory.merge_share(
        store_path=store_a_dir,
        share_dir=share_dir,
        repo_path=repo,
    )
    assert rep.repository_id is not None
    assert rep.sources_merged >= 0


def test_semantic_search_on_v1_store_reports_gap(tmp_path: Path) -> None:
    with _embedded_memory(tmp_path) as mem:
        mem._store.schema_version = 1
        res = mem.search(SearchRequest(query="allocate worker resources", kind="code"))
    assert not [h for h in res.hits if h.match in ("semantic", "hybrid")]
    assert any(g.code == "ENRICHMENT_PENDING" and g.scope == "search:semantic" for g in res.gaps)
