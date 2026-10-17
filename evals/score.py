import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from codecontext.identity import canonical_sha256


class EvalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RequiredFact(EvalModel):
    id: Annotated[str, Field(min_length=1, max_length=100)]
    text: Annotated[str, Field(min_length=1, max_length=1000)]
    weight: Literal[1, 2]
    critical: bool
    accepted_evidence_requirement_ids: Annotated[list[str], Field(min_length=1, max_length=20)]


class EvidenceRequirement(EvalModel):
    id: Annotated[str, Field(min_length=1, max_length=100)]
    role: Literal["support", "counterevidence", "lifecycle", "applicability"]
    accepted_source_ids: Annotated[list[str], Field(min_length=1, max_length=20)]


class GoldQuestion(EvalModel):
    id: Annotated[str, Field(min_length=1, max_length=100)]
    family_id: Annotated[str, Field(min_length=1, max_length=100)]
    repository: Literal["django/django", "vitejs/vite", "psf/requests"]
    revision: Annotated[str, Field(min_length=40, max_length=64)]
    cutoff: str | None = None
    partition: Literal["dev", "held_out"]
    question_class: Literal[
        "original_rationale",
        "later_applicability",
        "rejected_alternatives",
        "conflicting_explanations",
        "original_to_landed_mapping",
        "unsupported_rationale",
    ]
    question: Annotated[str, Field(min_length=1, max_length=2000)]
    answerability: Literal["answerable", "unsupported"]
    required_facts: list[RequiredFact] = Field(default_factory=list)
    evidence_requirements: list[EvidenceRequirement] = Field(default_factory=list)
    expected_applicability: Literal["corroborated", "historical_only", "unknown"]
    expected_abstention_reason: str | None = None
    review_status: Literal["draft_pending_human_review", "reviewed_adjudicated"] = (
        "draft_pending_human_review"
    )


class AssertionScore(EvalModel):
    text: Annotated[str, Field(min_length=1, max_length=1000)]
    material: bool
    supported: bool
    cited_evidence_ids: list[str] = Field(default_factory=list)


class FactScore(EvalModel):
    fact_id: Annotated[str, Field(min_length=1, max_length=100)]
    present: bool


class QuestionScoreRecord(EvalModel):
    run_id: Annotated[str, Field(min_length=1, max_length=100)]
    question_id: Annotated[str, Field(min_length=1, max_length=100)]
    reviewer_id: Annotated[str, Field(min_length=1, max_length=100)]
    assertions: list[AssertionScore]
    fact_scores: list[FactScore]
    retrieved_source_ids_top10: list[str] = Field(default_factory=list)
    retrieved_counterevidence_ids: list[str] = Field(default_factory=list)
    applicability_correct: bool
    abstention_correct: bool
    critical_error: bool
    notes: Annotated[str, Field(max_length=2000)] = ""


def wilson_95_interval(successes: int, total: int) -> tuple[float, float]:
    """Compute Wilson 95% confidence interval for a binomial proportion."""
    if total <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    phat = successes / total
    denom = 1.0 + (z * z) / total
    center = (phat + (z * z) / (2.0 * total)) / denom
    half = (
        z
        * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * total)) / total)
        / denom
    )
    return (max(0.0, round(center - half, 6)), min(1.0, round(center + half, 6)))


def bootstrap_family_difference(
    family_deltas: dict[str, list[float]],
    *,
    iterations: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap whole change families with a fixed seed, returning (mean_diff, ci_low, ci_high)."""
    if not family_deltas:
        return (0.0, 0.0, 0.0)
    fams = sorted(family_deltas.keys())
    all_vals = [v for k in fams for v in family_deltas[k]]
    obs_mean = sum(all_vals) / len(all_vals) if all_vals else 0.0

    rng = random.Random(seed)
    means: list[float] = []
    n_fams = len(fams)
    for _ in range(iterations):
        sampled = [rng.choice(fams) for _ in range(n_fams)]
        vals = [v for fk in sampled for v in family_deltas[fk]]
        means.append(sum(vals) / len(vals) if vals else 0.0)
    means.sort()
    low_idx = max(0, int(0.025 * len(means)))
    high_idx = min(len(means) - 1, int(0.975 * len(means)))
    return (round(obs_mean, 6), round(means[low_idx], 6), round(means[high_idx], 6))


@dataclass(frozen=True)
class SingleQuestionEvaluation:
    question_id: str
    family_id: str
    answerability: str
    weighted_fact_recall: float | None
    all_critical_present: bool
    has_unsupported_material_claim: bool
    complete_success: bool
    abstention_success: bool | None
    evidence_recall_at_10: float | None
    counterevidence_recall: float | None


def evaluate_single_question_score(
    question: GoldQuestion,
    score: QuestionScoreRecord,
) -> SingleQuestionEvaluation:
    """Apply E05 scoring rules to one question and its reviewer score record."""
    has_unsupported_material = any(
        a.material and not a.supported for a in score.assertions
    )

    # Evidence recall at 10 (deduplicated by accepted source requirement)
    top10_set = set(score.retrieved_source_ids_top10[:10])
    support_reqs = [r for r in question.evidence_requirements if r.role != "counterevidence"]
    ev_recall_10: float | None = None
    if support_reqs and len(support_reqs) <= 10:
        satisfied = sum(
            1
            for req in support_reqs
            if any(sid in top10_set for sid in req.accepted_source_ids)
        )
        ev_recall_10 = satisfied / len(support_reqs)

    # Counterevidence recall (zero denominator -> None / N/A, never 100%)
    counter_reqs = [r for r in question.evidence_requirements if r.role == "counterevidence"]
    counter_set = set(score.retrieved_counterevidence_ids) | top10_set
    counter_recall: float | None = None
    if counter_reqs:
        c_sat = sum(
            1
            for req in counter_reqs
            if any(sid in counter_set for sid in req.accepted_source_ids)
        )
        counter_recall = c_sat / len(counter_reqs)

    if question.answerability == "unsupported":
        abst_ok = (
            score.abstention_correct
            and not has_unsupported_material
            and not score.critical_error
        )
        return SingleQuestionEvaluation(
            question_id=question.id,
            family_id=question.family_id,
            answerability="unsupported",
            weighted_fact_recall=None,
            all_critical_present=True,
            has_unsupported_material_claim=has_unsupported_material,
            complete_success=abst_ok,
            abstention_success=abst_ok,
            evidence_recall_at_10=ev_recall_10,
            counterevidence_recall=counter_recall,
        )

    present_ids = {fs.fact_id for fs in score.fact_scores if fs.present}
    total_weight = sum(f.weight for f in question.required_facts)
    earned_weight = sum(
        f.weight for f in question.required_facts if f.id in present_ids
    )
    w_recall = (earned_weight / total_weight) if total_weight > 0 else 1.0
    all_crit = all(
        f.id in present_ids for f in question.required_facts if f.critical
    )

    complete_success = (
        all_crit
        and w_recall >= 0.80
        and not has_unsupported_material
        and score.applicability_correct
        and not score.critical_error
    )
    return SingleQuestionEvaluation(
        question_id=question.id,
        family_id=question.family_id,
        answerability="answerable",
        weighted_fact_recall=w_recall,
        all_critical_present=all_crit,
        has_unsupported_material_claim=has_unsupported_material,
        complete_success=complete_success,
        abstention_success=None,
        evidence_recall_at_10=ev_recall_10,
        counterevidence_recall=counter_recall,
    )


def score_corpus_runs(
    questions: list[GoldQuestion],
    scores: list[QuestionScoreRecord],
) -> dict[str, Any]:
    """Compute E05 aggregate metrics, Wilson 95% intervals, and pilot gate checks from `questions` and `scores`."""
    q_by_id = {q.id: q for q in questions}
    evals: list[SingleQuestionEvaluation] = []

    total_material_assertions = 0
    supported_material_assertions = 0
    total_fact_weight = 0
    earned_fact_weight = 0

    for sc in scores:
        q = q_by_id.get(sc.question_id)
        if q is None:
            continue
        ev = evaluate_single_question_score(q, sc)
        evals.append(ev)

        for a in sc.assertions:
            if a.material:
                total_material_assertions += 1
                if a.supported:
                    supported_material_assertions += 1

        if q.answerability == "answerable":
            present_ids = {fs.fact_id for fs in sc.fact_scores if fs.present}
            for f in q.required_facts:
                total_fact_weight += f.weight
                if f.id in present_ids:
                    earned_fact_weight += f.weight

    answerable_evals = [e for e in evals if e.answerability == "answerable"]
    unsupported_evals = [e for e in evals if e.answerability == "unsupported"]

    macro_fact_recall = (
        sum(e.weighted_fact_recall or 0.0 for e in answerable_evals) / len(answerable_evals)
        if answerable_evals
        else 0.0
    )
    micro_fact_recall = (
        earned_fact_weight / total_fact_weight if total_fact_weight > 0 else 0.0
    )
    citation_precision = (
        supported_material_assertions / total_material_assertions
        if total_material_assertions > 0
        else 1.0
    )

    ans_success_cnt = sum(1 for e in answerable_evals if e.complete_success)
    unsup_abst_cnt = sum(1 for e in unsupported_evals if e.abstention_success)

    ev10_vals = [e.evidence_recall_at_10 for e in evals if e.evidence_recall_at_10 is not None]
    mean_ev10 = sum(ev10_vals) / len(ev10_vals) if ev10_vals else None

    counter_vals = [
        e.counterevidence_recall for e in evals if e.counterevidence_recall is not None
    ]
    mean_counter_recall = (
        sum(counter_vals) / len(counter_vals) if counter_vals else None
    )

    gates = {
        "answerable_complete_success_54_of_60": (
            ans_success_cnt >= 54 if len(answerable_evals) == 60 else False
        ),
        "citation_support_ge_95pct": citation_precision >= 0.95,
        "counterevidence_recall_ge_90pct": (
            mean_counter_recall >= 0.90 if mean_counter_recall is not None else True
        ),
        "evidence_recall_at_10_ge_90pct": (
            mean_ev10 >= 0.90 if mean_ev10 is not None else False
        ),
        "macro_fact_recall_ge_90pct": macro_fact_recall >= 0.90,
        "unsupported_abstention_19_of_20": (
            unsup_abst_cnt >= 19 if len(unsupported_evals) == 20 else False
        ),
    }

    return {
        "answerable_count": len(answerable_evals),
        "answerable_success_count": ans_success_cnt,
        "answerable_success_wilson_95": wilson_95_interval(
            ans_success_cnt, len(answerable_evals)
        ),
        "citation_precision": round(citation_precision, 6),
        "counterevidence_recall": (
            round(mean_counter_recall, 6) if mean_counter_recall is not None else None
        ),
        "evidence_recall_at_10": (
            round(mean_ev10, 6) if mean_ev10 is not None else None
        ),
        "gates": gates,
        "macro_weighted_fact_recall": round(macro_fact_recall, 6),
        "material_assertions_supported": supported_material_assertions,
        "material_assertions_total": total_material_assertions,
        "micro_weighted_fact_recall": round(micro_fact_recall, 6),
        "pilot_label": "120-question pilot",
        "unsupported_abstention_count": unsup_abst_cnt,
        "unsupported_abstention_wilson_95": wilson_95_interval(
            unsup_abst_cnt, len(unsupported_evals)
        ),
        "unsupported_count": len(unsupported_evals),
    }


def validate_questions_corpus(questions: list[GoldQuestion]) -> dict[str, Any]:
    """Validate E01 partition counts (40 dev / 80 held_out across 6 classes, 40 per repo)
    and verify zero family_id overlap between `dev` and `held_out`.
    """
    if len(questions) != 120:
        raise ValueError(f"Expected 120 questions, got {len(questions)}")

    dev_families: set[str] = set()
    held_families: set[str] = set()
    repo_counts: dict[str, int] = {}
    split_class_counts: dict[tuple[str, str], int] = {}

    for q in questions:
        repo_counts[q.repository] = repo_counts.get(q.repository, 0) + 1
        key = (q.partition, q.question_class)
        split_class_counts[key] = split_class_counts.get(key, 0) + 1
        if q.partition == "dev":
            dev_families.add(q.family_id)
        else:
            held_families.add(q.family_id)

    overlap = dev_families & held_families
    if overlap:
        raise ValueError(f"Cross-partition family_id overlap detected: {sorted(overlap)}")

    expected_split_counts = {
        ("dev", "original_rationale"): 10,
        ("held_out", "original_rationale"): 20,
        ("dev", "later_applicability"): 8,
        ("held_out", "later_applicability"): 16,
        ("dev", "rejected_alternatives"): 6,
        ("held_out", "rejected_alternatives"): 12,
        ("dev", "conflicting_explanations"): 4,
        ("held_out", "conflicting_explanations"): 8,
        ("dev", "original_to_landed_mapping"): 2,
        ("held_out", "original_to_landed_mapping"): 4,
        ("dev", "unsupported_rationale"): 10,
        ("held_out", "unsupported_rationale"): 20,
    }
    for k, exp_v in expected_split_counts.items():
        act_v = split_class_counts.get(k, 0)
        if act_v != exp_v:
            raise ValueError(f"Partition/class {k} expected {exp_v}, got {act_v}")

    for repo_name in ("django/django", "vitejs/vite", "psf/requests"):
        if repo_counts.get(repo_name, 0) != 40:
            raise ValueError(f"Repository {repo_name} expected 40 questions, got {repo_counts.get(repo_name, 0)}")

    return {
        "dev_families": len(dev_families),
        "held_out_families": len(held_families),
        "repo_counts": repo_counts,
        "total_questions": len(questions),
        "valid": True,
    }


def generate_pilot_120_questions_jsonl(dest_path: Path) -> str:
    """Generate the 120-question E01 pilot specification across Django, Vite, and Requests
    with exact 40 dev / 80 held_out split and zero cross-partition family overlap.
    Marked `review_status='draft_pending_human_review'` per T21 rule against self-certifying human review.
    """
    repos: list[tuple[str, str]] = [
        ("django/django", "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4"),
        ("vitejs/vite", "b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5"),
        ("psf/requests", "c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f6"),
    ]
    # Exact E01 counts:
    # dev (40): 10 original_rationale, 8 later_applicability, 6 rejected_alternatives, 4 conflicting_explanations, 2 original_to_landed_mapping, 10 unsupported_rationale
    # held_out (80): 20 original_rationale, 16 later_applicability, 12 rejected_alternatives, 8 conflicting_explanations, 4 original_to_landed_mapping, 20 unsupported_rationale
    schedule: list[tuple[str, str, int]] = [
        ("dev", "original_rationale", 10),
        ("dev", "later_applicability", 8),
        ("dev", "rejected_alternatives", 6),
        ("dev", "conflicting_explanations", 4),
        ("dev", "original_to_landed_mapping", 2),
        ("dev", "unsupported_rationale", 10),
        ("held_out", "original_rationale", 20),
        ("held_out", "later_applicability", 16),
        ("held_out", "rejected_alternatives", 12),
        ("held_out", "conflicting_explanations", 8),
        ("held_out", "original_to_landed_mapping", 4),
        ("held_out", "unsupported_rationale", 20),
    ]

    questions: list[GoldQuestion] = []
    q_idx = 0
    for partition, q_class, count in schedule:
        for i in range(count):
            repo_name, rev = repos[q_idx % 3]
            q_idx += 1
            qid = f"Q{q_idx:03d}"
            fam_id = f"fam_{partition}_{q_class}_{i // 2:02d}"
            is_unsup = q_class == "unsupported_rationale"

            if is_unsup:
                q_obj = GoldQuestion(
                    id=qid,
                    family_id=fam_id,
                    repository=repo_name,  # type: ignore[arg-type]
                    revision=rev,
                    cutoff="2026-06-01T00:00:00Z",
                    partition=partition,  # type: ignore[arg-type]
                    question_class=q_class,  # type: ignore[arg-type]
                    question=f"[{repo_name}] What documented rationale explains internal helper constant #{q_idx}?",
                    answerability="unsupported",
                    required_facts=[],
                    evidence_requirements=[],
                    expected_applicability="unknown",
                    expected_abstention_reason="undocumented_in_available_evidence",
                    review_status="draft_pending_human_review",
                )
            else:
                ev_reqs = [
                    EvidenceRequirement(
                        id=f"{qid}_ev1",
                        role="support",
                        accepted_source_ids=[f"src_{qid}_primary"],
                    )
                ]
                if q_class == "conflicting_explanations":
                    ev_reqs.append(
                        EvidenceRequirement(
                            id=f"{qid}_ev_counter",
                            role="counterevidence",
                            accepted_source_ids=[f"src_{qid}_counter"],
                        )
                    )
                q_obj = GoldQuestion(
                    id=qid,
                    family_id=fam_id,
                    repository=repo_name,  # type: ignore[arg-type]
                    revision=rev,
                    cutoff="2026-06-01T00:00:00Z",
                    partition=partition,  # type: ignore[arg-type]
                    question_class=q_class,  # type: ignore[arg-type]
                    question=f"[{repo_name}] Explain the documented {q_class} and constraints for decision #{q_idx}.",
                    answerability="answerable",
                    required_facts=[
                        RequiredFact(
                            id=f"{qid}_f1",
                            text="Primary documented constraint and rationale",
                            weight=2,
                            critical=True,
                            accepted_evidence_requirement_ids=[f"{qid}_ev1"],
                        ),
                        RequiredFact(
                            id=f"{qid}_f2",
                            text="Secondary context or rejected alternative detail",
                            weight=1,
                            critical=False,
                            accepted_evidence_requirement_ids=[f"{qid}_ev1"],
                        ),
                    ],
                    evidence_requirements=ev_reqs,
                    expected_applicability=(
                        "corroborated" if (q_class == "later_applicability" and i % 2 == 0) else "historical_only"
                    ),
                    expected_abstention_reason=None,
                    review_status="draft_pending_human_review",
                )
            questions.append(q_obj)

    validate_questions_corpus(questions)
    lines = [
        json.dumps(q.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        for q in questions
    ]
    content = "\n".join(lines) + "\n"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(content, encoding="utf-8")
    return canonical_sha256(lines)
