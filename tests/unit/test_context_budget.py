import json
from pathlib import Path

import pytest

from codecontext.enrichment import execute_enrich
from codecontext.errors import CodeContextError
from codecontext.git import GitMirror
from codecontext.lineage import index_commit_changes_and_hunks, index_focused_path_history
from codecontext.models import (
    ContextPack,
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
from tests.fixtures.builders import GitCommitSpec, commit_fixture_spec, init_fixture_repo

CLAIM_COUNT = 12


class ManyClaimsAdapter:
    """Extracts one claim per distinct phrase, citing every target anchor."""

    identity = ModelIdentity(
        adapter_name="many_claims",
        model_name="many-claims",
        model_revision="1.0",
        configuration_fingerprint="many_claims_fp",
        input_price_per_million=None,
        output_price_per_million=None,
        framing_input_tokens=5,
    )

    def complete(self, request: ModelRequest) -> ModelResponse:
        payload = json.loads(request.input_json)
        if "original_input" in payload:
            payload = payload["original_input"]
        slots = payload["slots"]
        keys = [str(a["key"]) for a in payload.get("anchor_options", [])]
        claims = [
            {
                "alternatives": [],
                "constraints": [],
                "corroboration": [],
                "counterevidence": [],
                "evidence": [{"quote": f"decision phrase number {i:02d}", "slot": "S0001"}],
                "lifecycle": "adopted",
                "lifecycle_evidence": [{"quote": f"decision phrase number {i:02d}", "slot": "S0001"}],
                "rationale": f"Rationale {i:02d} " + "because of reasons " * 8,
                "statement": f"Statement {i:02d}",
                "support": "explicit",
                "target_anchor_keys": keys,
            }
            for i in range(CLAIM_COUNT)
        ]
        out = json.dumps(
            {
                "claims": claims,
                "inspected_slots": [str(s["slot"]) for s in slots],
                "unresolved": [],
            }
        )
        return ModelResponse(output_json=out, input_tokens=40, output_tokens=30)


def _enriched_store(tmp_path: Path) -> tuple[Store, GitMirror]:
    repo_dir = init_fixture_repo(tmp_path / "repo")
    message = "Many decisions\n\n" + "\n".join(
        f"decision phrase number {i:02d}" for i in range(CLAIM_COUNT)
    )
    c_oid = commit_fixture_spec(
        repo_dir,
        GitCommitSpec(message=message, files={"src/a.py": b"def f():\n    return 1\n"}),
    )
    store = Store.initialize(tmp_path / "store", InitRequest(repo_path=repo_dir))
    mirror = GitMirror(store)
    active_rev = mirror.prepare_snapshot("HEAD")
    index_active_snapshot_files(store, mirror)
    index_focused_path_history(store, mirror, active_revision=active_rev, focus_path=b"src/a.py")
    index_commit_changes_and_hunks(store, mirror, c_oid)
    execute_enrich(
        store,
        mirror,
        EnrichRequest(max_input_tokens=100_000, max_output_tokens=20_000, max_calls=5),
        model=ManyClaimsAdapter(),
    )
    return store, mirror


def _context(store: Store, mirror: GitMirror, max_tokens: int) -> ContextPack:
    return execute_context(
        store,
        mirror,
        ContextRequest(
            query="Why does f return 1?",
            target=PathTarget(kind="path", path="src/a.py"),
            max_tokens=max_tokens,
        ),
    )


@pytest.mark.parametrize("max_tokens", [3200, 2400, 1600])
def test_many_decisions_never_exceed_requested_tokens(tmp_path: Path, max_tokens: int) -> None:
    store, mirror = _enriched_store(tmp_path)
    pack = _context(store, mirror, max_tokens)
    store.close()
    assert pack.budget.truncated is True
    assert pack.budget.used_tokens <= max_tokens


def test_trim_drops_lowest_ranked_decisions_first(tmp_path: Path) -> None:
    store, mirror = _enriched_store(tmp_path)
    full = _context(store, mirror, 32000)
    small = _context(store, mirror, 2400)
    store.close()

    full_ids = [d.id for d in full.decisions]
    kept_ids = [d.id for d in small.decisions]
    assert len(full_ids) == CLAIM_COUNT
    assert 1 <= len(kept_ids) < CLAIM_COUNT
    assert kept_ids == full_ids[: len(kept_ids)]
    assert small.budget.used_tokens <= 2400
    assert small.budget.omitted_counts.decisions == CLAIM_COUNT - len(kept_ids)
    expand_ids = {e.id for e in small.expand}
    assert set(full_ids[len(kept_ids) :]) <= expand_ids
    cited = {r.id for d in small.decisions for r in d.evidence_refs}
    assert cited <= {e.ref.id for e in small.evidence}


def test_pack_that_cannot_fit_after_all_drops_is_invalid_request(tmp_path: Path) -> None:
    store, mirror = _enriched_store(tmp_path)
    with pytest.raises(CodeContextError) as exc_info:
        _context(store, mirror, 1250)
    store.close()
    assert exc_info.value.code == "INVALID_REQUEST"
    assert exc_info.value.details["requested_tokens"] == 1250
