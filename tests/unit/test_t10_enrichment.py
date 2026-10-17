import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from codecontext.enrichment import execute_enrich
from codecontext.errors import CodeContextError
from codecontext.git import GitMirror
from codecontext.github import GitHubClient, collect_focused_github_evidence
from codecontext.lineage import index_commit_changes_and_hunks, index_focused_path_history
from codecontext.models import (
    ContextRequest,
    EnrichRequest,
    InitRequest,
    ModelIdentity,
    ModelRequest,
    ModelResponse,
    PathTarget,
)
from codecontext.retrieval import execute_context
from codecontext.store import Store
from codecontext.syntax import index_active_snapshot_files
from examples.fixture_adapter import DeterministicFixtureAdapter
from tests.fixtures.builders import (
    GitCommitSpec,
    build_f01_fixture,
    commit_fixture_spec,
    init_fixture_repo,
)
from tests.unit.test_t07_github import make_mock_github_transport


class ScriptedTestAdapter:
    def __init__(
        self,
        responses: list[ModelResponse | Exception],
        *,
        in_price: Decimal | None = Decimal("1.0"),
        out_price: Decimal | None = Decimal("2.0"),
    ) -> None:
        self._responses = list(responses)
        self.calls: list[ModelRequest] = []
        self._identity = ModelIdentity(
            adapter_name="scripted_test",
            model_name="scripted-model",
            model_revision="1.0",
            configuration_fingerprint="script_fp",
            input_price_per_million=in_price,
            output_price_per_million=out_price,
            framing_input_tokens=5,
        )

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        if not self._responses:
            raise RuntimeError("No scripted responses remaining")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _prepare_f01_store(tmp_path: Path) -> tuple[Store, GitMirror, Any]:
    f01 = build_f01_fixture(tmp_path / "f01")
    store = Store.initialize(
        tmp_path / "store",
        InitRequest(repo_path=f01.repo_dir, github_repository="example/transport"),
    )
    mirror = GitMirror(store)
    active_rev = mirror.prepare_snapshot("HEAD")
    index_active_snapshot_files(store, mirror)
    index_focused_path_history(store, mirror, active_revision=active_rev, focus_path=b"src/transport.py")
    index_commit_changes_and_hunks(store, mirror, f01.c2_oid)
    with store.write_transaction(bump_generation=False) as conn:
        Store.resolve_gaps_in_scope(conn, scope="git:history", code="GIT_HISTORY_PENDING")

    routes = {
        "/repos/example/transport": (
            200,
            {"id": 9001, "full_name": "example/transport", "private": False},
            {},
        ),
        f"/repos/example/transport/commits/{f01.c1_oid}/pulls": (
            200,
            [{"number": 10, "id": 10010}],
            {},
        ),
        "/repos/example/transport/pulls/10": (200, f01.pr_payload, {}),
        "/repos/example/transport/issues/10/comments?per_page=100": (
            200,
            f01.comments_payload,
            {},
        ),
        "/repos/example/transport/pulls/10/reviews?per_page=100": (
            200,
            f01.reviews_payload,
            {},
        ),
        "/repos/example/transport/pulls/10/comments?per_page=100": (200, [], {}),
        "/repos/example/transport/issues/10/timeline?per_page=100": (200, [], {}),
        "/repos/example/transport/pulls/10/commits?per_page=100": (
            200,
            [{"sha": f01.c1_oid}],
            {},
        ),
        "/repos/example/transport/pulls/10/files?per_page=100": (200, [], {}),
    }
    transport, _ = make_mock_github_transport(routes)
    with GitHubClient(token="", transport=transport) as client:
        collect_focused_github_evidence(store, client, commit_oids=[f01.c1_oid])

    return store, mirror, f01


def test_f01_u08_u09_u10_explicit_extraction_and_applicability(tmp_path: Path) -> None:
    """U08, U09, U10, & T10:
    - Before enrichment: `rationale_status='enrichment_pending'`
    - Enrich F01 with deterministic fixture adapter -> publishes attributed claim with exact quote spans,
      constraints, rejected alternative, and `applicability='historical_only'` at C2
    - Repeated enrichment is a cache hit (0 model calls)
    - Completely inspected undocumented episode produces `rationale_status='undocumented_in_available_evidence'`.
    """
    store, mirror, _f01 = _prepare_f01_store(tmp_path)

    # Before enrichment: `rationale_status` is `enrichment_pending`
    pack_before = execute_context(
        store,
        mirror,
        ContextRequest(
            query="Why is send limited to 3 retries?",
            target=PathTarget(kind="path", path="src/transport.py", line=5),
        ),
    )
    assert pack_before.rationale_status == "enrichment_pending"
    assert pack_before.decisions == []

    # Run enrichment with DeterministicFixtureAdapter
    adapter = DeterministicFixtureAdapter()
    rep1 = execute_enrich(
        store,
        mirror,
        EnrichRequest(max_input_tokens=100_000, max_output_tokens=20_000, max_calls=10),
        model=adapter,
    )
    assert rep1.state == "complete"
    assert rep1.claims_added == 1
    assert rep1.usage.calls >= 1

    # Query context for `send` after enrichment
    pack_after = execute_context(
        store,
        mirror,
        ContextRequest(
            query="Why is send limited to 3 retries?",
            target=PathTarget(kind="path", path="src/transport.py", line=5),
        ),
    )
    assert pack_after.rationale_status == "supported"
    assert len(pack_after.decisions) == 1
    dec = pack_after.decisions[0]
    assert dec.support == "explicit"
    assert dec.lifecycle == "adopted"
    assert dec.mapping == "exact"
    # Even though `send` bytes are unchanged between C1 and C2, applicability remains `historical_only`!
    assert dec.applicability == "historical_only"
    assert "Unlimited retries with exponential backoff" in dec.alternatives
    assert len(dec.constraints) == 2
    assert len(dec.evidence_refs) >= 1
    assert len(dec.counterevidence_refs) == 1

    # Repeat enrichment without changes -> cache hit, 0 model calls
    rep2 = execute_enrich(
        store,
        mirror,
        EnrichRequest(max_input_tokens=100_000, max_output_tokens=20_000, max_calls=10),
        model=adapter,
    )
    assert rep2.usage.calls == 0
    assert rep2.cache_hit_items >= 1

    store.close()


def test_u10_undocumented_vs_partial_slot_inspection(tmp_path: Path) -> None:
    """U10 & Retrieval §3.3:
    - An episode with an empty claims array where only a subset of required slots are in `inspected_slots`
      remains `enrichment_pending` with `EPISODE_INSPECTION_PENDING` gap.
    - Once all required slots are in `inspected_slots` with empty claims, `rationale_status` becomes
      `undocumented_in_available_evidence`.
    """
    repo_dir = init_fixture_repo(tmp_path / "undoc_repo")
    c_oid = commit_fixture_spec(
        repo_dir,
        GitCommitSpec(
            message="Add mystery_fn without rationale",
            files={"src/mystery.py": b"def mystery_fn(x: int) -> int:\n    return x ^ 42\n"},
        ),
    )
    store = Store.initialize(tmp_path / "undoc_store", InitRequest(repo_path=repo_dir))
    mirror = GitMirror(store)
    mirror.prepare_snapshot("HEAD")
    index_active_snapshot_files(store, mirror)
    index_commit_changes_and_hunks(store, mirror, c_oid)

    # 1. Model returns empty claims and omits S0001 from inspected_slots
    partial_adapter = ScriptedTestAdapter(
        [
            ModelResponse(
                output_json=json.dumps({"claims": [], "inspected_slots": [], "unresolved": []}),
                input_tokens=50,
                output_tokens=20,
            )
        ]
    )
    execute_enrich(
        store,
        mirror,
        EnrichRequest(max_input_tokens=50_000, max_output_tokens=10_000, max_calls=5),
        model=partial_adapter,
    )

    pack_partial = execute_context(
        store,
        mirror,
        ContextRequest(
            query="Why does mystery_fn XOR with 42?",
            target=PathTarget(kind="path", path="src/mystery.py", line=1),
        ),
    )
    assert pack_partial.rationale_status == "enrichment_pending"
    assert any(g.code == "EPISODE_INSPECTION_PENDING" for g in pack_partial.gaps)

    # 2. Model inspects all presented slots (S0001) and returns empty claims
    full_adapter = ScriptedTestAdapter(
        [
            ModelResponse(
                output_json=json.dumps({"claims": [], "inspected_slots": ["S0001"], "unresolved": []}),
                input_tokens=50,
                output_tokens=20,
            )
        ]
    )
    execute_enrich(
        store,
        mirror,
        EnrichRequest(max_input_tokens=50_000, max_output_tokens=10_000, max_calls=5),
        model=full_adapter,
    )

    pack_full = execute_context(
        store,
        mirror,
        ContextRequest(
            query="Why does mystery_fn XOR with 42?",
            target=PathTarget(kind="path", path="src/mystery.py", line=1),
        ),
    )
    assert pack_full.rationale_status == "undocumented_in_available_evidence"
    store.close()


def test_u12_u16_budget_reservation_invalid_quotes_and_timeout(tmp_path: Path) -> None:
    """U12 & U16:
    - Monetary budget with missing prices fails before invocation
    - Invented slot ID or non-matching quote fails validation, triggers at most 1 repair attempt, and never publishes invalid claims
    - Timeout / adapter exception retains reservation as uncertain usage and does not auto-retry.
    """
    repo_dir = init_fixture_repo(tmp_path / "u12_repo")
    c_oid = commit_fixture_spec(
        repo_dir,
        GitCommitSpec(
            message="Commit message with repeated word word and unique phrase",
            files={"src/a.py": b"def f():\n    return 1\n"},
        ),
    )
    store = Store.initialize(tmp_path / "u12_store", InitRequest(repo_path=repo_dir))
    mirror = GitMirror(store)
    mirror.prepare_snapshot("HEAD")
    index_active_snapshot_files(store, mirror)
    index_commit_changes_and_hunks(store, mirror, c_oid)

    # 1. Monetary limit with missing prices fails before any call
    no_price_adapter = ScriptedTestAdapter([], in_price=None, out_price=None)
    with pytest.raises(CodeContextError) as exc_info:
        execute_enrich(
            store,
            mirror,
            EnrichRequest(
                max_input_tokens=50_000,
                max_output_tokens=10_000,
                max_calls=5,
                max_cost_usd=Decimal("5.00"),
            ),
            model=no_price_adapter,
        )
    assert exc_info.value.code == "INVALID_REQUEST"
    assert len(no_price_adapter.calls) == 0

    # 2. First response has ambiguous repeated quote ("word"); repair response has valid unique quote
    repair_adapter = ScriptedTestAdapter(
        [
            ModelResponse(
                output_json=json.dumps(
                    {
                        "claims": [
                            {
                                "alternatives": [],
                                "constraints": [],
                                "corroboration": [],
                                "counterevidence": [],
                                "evidence": [{"quote": "word", "slot": "S0001"}],
                                "lifecycle": "adopted",
                                "lifecycle_evidence": [{"quote": "word", "slot": "S0001"}],
                                "rationale": "Ambiguous quote test",
                                "statement": "Statement 1",
                                "support": "explicit",
                                "target_anchor_keys": [],
                            }
                        ],
                        "inspected_slots": ["S0001"],
                        "unresolved": [],
                    }
                ),
                input_tokens=40,
                output_tokens=30,
            ),
            ModelResponse(
                output_json=json.dumps(
                    {
                        "claims": [
                            {
                                "alternatives": [],
                                "constraints": [],
                                "corroboration": [],
                                "counterevidence": [],
                                "evidence": [{"quote": "unique phrase", "slot": "S0001"}],
                                "lifecycle": "adopted",
                                "lifecycle_evidence": [{"quote": "unique phrase", "slot": "S0001"}],
                                "rationale": "Repaired unique quote",
                                "statement": "Statement 1",
                                "support": "explicit",
                                "target_anchor_keys": [],
                            }
                        ],
                        "inspected_slots": ["S0001"],
                        "unresolved": [],
                    }
                ),
                input_tokens=60,
                output_tokens=30,
            ),
        ]
    )
    rep_repair = execute_enrich(
        store,
        mirror,
        EnrichRequest(max_input_tokens=50_000, max_output_tokens=10_000, max_calls=5),
        model=repair_adapter,
    )
    assert rep_repair.usage.calls == 2
    assert rep_repair.claims_added == 1

    # 3. Adapter timeout/exception on a new commit -> no automatic retry, reservation retained in uncertain usage
    c2_oid = commit_fixture_spec(
        repo_dir,
        GitCommitSpec(
            message="Second commit for timeout test",
            files={"src/a.py": b"def f():\n    return 2\n"},
        ),
    )
    mirror.prepare_snapshot("HEAD")
    index_active_snapshot_files(store, mirror)
    index_commit_changes_and_hunks(store, mirror, c2_oid)

    timeout_adapter = ScriptedTestAdapter([TimeoutError("Provider timed out")])
    rep_timeout = execute_enrich(
        store,
        mirror,
        EnrichRequest(max_input_tokens=50_000, max_output_tokens=10_000, max_calls=5),
        model=timeout_adapter,
    )
    assert len(timeout_adapter.calls) == 1  # No hidden retry!
    assert rep_timeout.stop_reason == "model_failure"
    assert rep_timeout.usage.uncertain_input_tokens > 0
    assert rep_timeout.usage.uncertain_output_tokens > 0

    store.close()
