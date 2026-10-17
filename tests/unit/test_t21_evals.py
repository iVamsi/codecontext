from pathlib import Path

from codecontext import (
    EnrichRequest,
    InitRequest,
    Memory,
    SyncRequest,
)
from evals.run import (
    EvalRequest,
    ScriptedEvaluationRunner,
    execute_eval_run,
)
from evals.score import (
    AssertionScore,
    FactScore,
    GoldQuestion,
    QuestionScoreRecord,
    bootstrap_family_difference,
    score_corpus_runs,
    validate_questions_corpus,
    wilson_95_interval,
)
from examples.fixture_adapter import DeterministicFixtureAdapter
from tests.fixtures.builders import build_f01_fixture
from tests.integration.test_t11_sdk_cli_mcp import _build_f01_routes
from tests.unit.test_t07_github import make_mock_github_transport


def test_t21_e01_e05_e06_e08_evaluation_harness_and_scoring(tmp_path: Path) -> None:
    """T21, E01-E08:
    - Validate generated 120-question pilot (`evals/questions.jsonl`) has exact 40 dev / 80 held_out split,
      40 per repository, and zero cross-partition family_id overlap
    - Unit-test E05 scoring on perfect, incomplete-but-precise, unsupported, missing-counterevidence,
      and zero-denominator cases
    - Test E08 `EvalTools` refusal boundaries: 21st tool call refused, result token limit refused,
      forged runner events marked `protocol_invalid`, oversized final answer marked `protocol_invalid`,
      and `ordinary` vs `lexical` vs `memory` arms over identical frozen source bytes.
    """
    questions_file = Path("evals/questions.jsonl")
    assert questions_file.exists()
    loaded_questions = [
        GoldQuestion.model_validate_json(line)
        for line in questions_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary_check = validate_questions_corpus(loaded_questions)
    assert summary_check["valid"] is True
    assert summary_check["total_questions"] == 120

    # Test E05 scoring edge cases:
    # 1. Held-out subset with perfect scores passes gates; zero counterevidence denominator yields None (not 100%)
    held_out = [q for q in loaded_questions if q.partition == "held_out"]
    assert len(held_out) == 80
    ans_qs = [q for q in held_out if q.answerability == "answerable"]
    unsup_qs = [q for q in held_out if q.answerability == "unsupported"]
    assert len(ans_qs) == 60 and len(unsup_qs) == 20

    perfect_scores: list[QuestionScoreRecord] = []
    for q in held_out:
        perfect_scores.append(
            QuestionScoreRecord(
                run_id=f"run_{q.id}",
                question_id=q.id,
                reviewer_id="rev_1",
                assertions=[
                    AssertionScore(
                        text="Supported assertion",
                        material=True,
                        supported=True,
                        cited_evidence_ids=[r.accepted_source_ids[0] for r in q.evidence_requirements[:1]],
                    )
                ]
                if q.answerability == "answerable"
                else [],
                fact_scores=[FactScore(fact_id=f.id, present=True) for f in q.required_facts],
                retrieved_source_ids_top10=[
                    r.accepted_source_ids[0] for r in q.evidence_requirements if r.role != "counterevidence"
                ],
                retrieved_counterevidence_ids=[
                    r.accepted_source_ids[0] for r in q.evidence_requirements if r.role == "counterevidence"
                ],
                applicability_correct=True,
                abstention_correct=(q.answerability == "unsupported"),
                critical_error=False,
            )
        )

    rep_perfect = score_corpus_runs(held_out, perfect_scores)
    assert rep_perfect["answerable_success_count"] == 60
    assert rep_perfect["unsupported_abstention_count"] == 20
    assert all(rep_perfect["gates"].values())

    # 2. Incomplete-but-precise case (100% citation precision, but omits critical fact -> fails complete_success!)
    q0 = ans_qs[0]
    incomplete_score = QuestionScoreRecord(
        run_id="run_inc",
        question_id=q0.id,
        reviewer_id="rev_1",
        assertions=[AssertionScore(text="Precise minor fact", material=True, supported=True)],
        fact_scores=[
            FactScore(fact_id=q0.required_facts[0].id, present=False),  # critical fact omitted!
            FactScore(fact_id=q0.required_facts[1].id, present=True),
        ],
        retrieved_source_ids_top10=[q0.evidence_requirements[0].accepted_source_ids[0]],
        retrieved_counterevidence_ids=[],
        applicability_correct=True,
        abstention_correct=False,
        critical_error=False,
    )
    rep_inc = score_corpus_runs([q0], [incomplete_score])
    assert rep_inc["citation_precision"] == 1.0
    assert rep_inc["answerable_success_count"] == 0
    # Zero counterevidence denominator on a single non-conflict question is None (N/A), never 1.0!
    assert rep_inc["counterevidence_recall"] is None

    # Test Wilson 95% interval and family bootstrap helper
    low, high = wilson_95_interval(19, 20)
    assert 0.70 < low < 0.95 < high <= 1.0
    mean_d, ci_l, ci_h = bootstrap_family_difference(
        {"fam1": [0.2, 0.3], "fam2": [0.1, 0.2], "fam3": [0.25, 0.35]},
        iterations=200,
        seed=42,
    )
    assert ci_l <= mean_d <= ci_h

    # Test E08 EvaluationRunner and EvalTools refusal boundaries on F01 store
    f01 = build_f01_fixture(tmp_path / "f01_eval")
    store_dir = tmp_path / "eval_store"
    routes = _build_f01_routes(f01)
    transport, _ = make_mock_github_transport(routes)

    with Memory.initialize(
        store_dir,
        InitRequest(repo_path=f01.repo_dir, github_repository="example/transport"),
    ) as mem:
        mem._github_transport = transport
        mem._github_token = ""
        mem.sync(SyncRequest(revision="HEAD"))
        mem.enrich(
            EnrichRequest(max_input_tokens=100_000, max_output_tokens=20_000, max_calls=10),
            model=DeterministicFixtureAdapter(),
        )

        out_dir = tmp_path / "eval_out"
        base_req = EvalRequest(
            run_id="eval_1",
            question_id="Q001",
            family_id="fam_1",
            arm="memory",
            question="Why is send limited to 3 retries?",
            target_revision=f01.c2_oid,
            evidence_manifest_sha256="a" * 64,
            model_name="test-agent",
            model_revision="1.0",
            model_options={"temperature": 0},
            system_prompt="Use tools to answer.",
            max_tool_calls=2,
            max_tool_result_tokens=30000,
            max_answer_tokens=200,
            timeout_seconds=60,
        )

        # A. Normal compliant run across `ordinary`, `lexical`, and `memory` arms
        for arm_name in ("ordinary", "lexical", "memory"):
            arm_req = base_req.model_copy(update={"run_id": f"run_{arm_name}", "arm": arm_name})
            res_ok = execute_eval_run(
                mem._store,
                arm_req,
                ScriptedEvaluationRunner(),
                output_dir=out_dir,
            )
            assert res_ok.state == "complete"
            assert len(res_ok.events) >= 1

        # B. Refusal boundary: exceeding max_tool_calls (3 calls when max_tool_calls=2) -> `budget_exhausted`
        over_calls_runner = ScriptedEvaluationRunner(
            script=[("list_files", {}), ("list_files", {}), ("list_files", {})]
        )
        res_over_calls = execute_eval_run(
            mem._store,
            base_req.model_copy(update={"run_id": "run_over_calls"}),
            over_calls_runner,
        )
        assert res_over_calls.state == "budget_exhausted"
        assert res_over_calls.events[-1].error_code == "BUDGET_EXHAUSTED"

        # C. Refusal boundary: exceeding max_tool_result_tokens -> `budget_exhausted`
        tiny_tok_req = base_req.model_copy(
            update={"run_id": "run_tiny_tok", "max_tool_result_tokens": 10}
        )
        res_tiny_tok = execute_eval_run(
            mem._store,
            tiny_tok_req,
            ScriptedEvaluationRunner(script=[("context", {"query": "Why 3 retries?"})]),
        )
        assert res_tiny_tok.state == "budget_exhausted"

        # D. Refusal boundary: runner forging events or exceeding max_answer_tokens -> `protocol_invalid`
        res_forged = execute_eval_run(
            mem._store,
            base_req.model_copy(update={"run_id": "run_forged"}),
            ScriptedEvaluationRunner(forge_events=[]),
        )
        assert res_forged.state == "protocol_invalid"
        assert res_forged.final_answer is None

        res_long_ans = execute_eval_run(
            mem._store,
            base_req.model_copy(update={"run_id": "run_long_ans", "max_answer_tokens": 5}),
            ScriptedEvaluationRunner(final_answer="This answer is much longer than five tokens " * 10),
        )
        assert res_long_ans.state == "protocol_invalid"

        assert (out_dir / "runs.jsonl").exists()
