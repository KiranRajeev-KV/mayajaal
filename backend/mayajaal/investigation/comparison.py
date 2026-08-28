"""Evaluator-only scoring for bounded investigation model comparisons.

This module deliberately has no dependency on the agent, evidence tools, or
artifact writers.  Hidden synthetic expectations enter only here, after an
investigation run has finished, and its returned objects are comparison
metrics rather than investigation artifacts.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import Field

from mayajaal.schemas.common import SchemaModel

from .models import InvestigationPattern, InvestigationStatus

_ABUSE_PATTERNS = frozenset(
    {
        InvestigationPattern.PROMO_RING,
        InvestigationPattern.REFUND_RING,
        InvestigationPattern.MIXED_ABUSE,
    }
)
_ANALYTICAL_STATUSES = frozenset(
    {
        InvestigationStatus.COMPLETED,
        InvestigationStatus.INSUFFICIENT_EVIDENCE,
    }
)


class EvaluationCase(SchemaModel):
    """Evaluator-only hidden expectation for one fixed comparison case.

    Do not pass this contract to agent construction, evidence retrieval, or
    investigation artifact persistence.
    """

    case_id: str = Field(min_length=1)
    expected_pattern: InvestigationPattern


class ComparisonRunOutcome(SchemaModel):
    """Reliability and conditional analytical-quality outcome for one run."""

    model: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    accepted_analytical_report: bool
    grounding_failure: bool
    budget_failure: bool
    failed_report: bool
    provider_request_failure: bool
    reported_status: InvestigationStatus | None = None
    reported_pattern: InvestigationPattern | None = None
    correct_pattern: bool | None
    false_fraud_accusation: bool | None
    missed_obvious_abuse: bool | None
    appropriate_ambiguity_handling: bool | None
    end_to_end_success: bool


class ConditionalRate(SchemaModel):
    """A rate that always exposes its numerator and denominator."""

    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)
    value: float | None


class ModelComparisonSummary(SchemaModel):
    """Per-model reliability and eligibility-aware quality metrics."""

    total_runs: int = Field(ge=0)
    accepted_report_rate: ConditionalRate
    end_to_end_success_rate: ConditionalRate
    grounding_failure_rate: ConditionalRate
    budget_failure_rate: ConditionalRate
    failed_report_rate: ConditionalRate
    provider_failure_rate: ConditionalRate
    conditional_pattern_accuracy: ConditionalRate
    benign_false_fraud_accusation_rate: ConditionalRate
    benign_system_failure_rate: ConditionalRate
    obvious_abuse_reasoning_miss_rate: ConditionalRate
    obvious_abuse_system_failure_rate: ConditionalRate


def score_comparison_run(
    *,
    model: str,
    case: EvaluationCase,
    status: InvestigationStatus | None,
    pattern: InvestigationPattern | None,
    grounding_failure: bool = False,
    provider_request_failure: bool = False,
) -> ComparisonRunOutcome:
    """Score one completed comparison run without granting system failures credit.

    A quality flag is ``None`` rather than ``False`` when no accepted analytical
    report exists.  This prevents a failed run's fallback ``INCONCLUSIVE``
    pattern from becoming apparent analytical correctness.
    """
    if provider_request_failure and status is not None:
        raise ValueError("provider request failure cannot include a report status")
    if provider_request_failure and grounding_failure:
        raise ValueError("provider request failure cannot be a grounding failure")

    budget_failure = status is InvestigationStatus.BUDGET_EXHAUSTED
    failed_report = status is InvestigationStatus.FAILED
    accepted = (
        not provider_request_failure
        and not grounding_failure
        and status in _ANALYTICAL_STATUSES
    )
    if not accepted:
        return ComparisonRunOutcome(
            model=model,
            case_id=case.case_id,
            accepted_analytical_report=False,
            grounding_failure=grounding_failure,
            budget_failure=budget_failure,
            failed_report=failed_report,
            provider_request_failure=provider_request_failure,
            reported_status=status,
            reported_pattern=pattern,
            correct_pattern=None,
            false_fraud_accusation=None,
            missed_obvious_abuse=None,
            appropriate_ambiguity_handling=None,
            end_to_end_success=False,
        )
    if pattern is None:
        raise ValueError("accepted analytical report requires a reported pattern")

    expected_is_abuse = case.expected_pattern in _ABUSE_PATTERNS
    reported_is_abuse = pattern in _ABUSE_PATTERNS
    correct_pattern = pattern is case.expected_pattern
    appropriate_ambiguity = (
        case.expected_pattern is InvestigationPattern.INCONCLUSIVE
        and (
            pattern is InvestigationPattern.INCONCLUSIVE
            or status is InvestigationStatus.INSUFFICIENT_EVIDENCE
        )
    )
    return ComparisonRunOutcome(
        model=model,
        case_id=case.case_id,
        accepted_analytical_report=True,
        grounding_failure=False,
        budget_failure=False,
        failed_report=False,
        provider_request_failure=False,
        reported_status=status,
        reported_pattern=pattern,
        correct_pattern=correct_pattern,
        false_fraud_accusation=not expected_is_abuse and reported_is_abuse,
        missed_obvious_abuse=expected_is_abuse and not correct_pattern,
        appropriate_ambiguity_handling=appropriate_ambiguity,
        end_to_end_success=correct_pattern or appropriate_ambiguity,
    )


def summarize_model_comparison(
    outcomes: Iterable[ComparisonRunOutcome],
    cases: Iterable[EvaluationCase],
) -> ModelComparisonSummary:
    """Aggregate one model's outcomes with explicit quality denominators."""
    records = tuple(outcomes)
    case_records = tuple(cases)
    expected_by_case = {case.case_id: case.expected_pattern for case in case_records}
    if len(expected_by_case) != len(case_records):
        raise ValueError("evaluation case IDs must be unique")
    if any(record.case_id not in expected_by_case for record in records):
        raise ValueError("comparison outcome references an unknown evaluation case")
    if len({record.case_id for record in records}) != len(records):
        raise ValueError("model comparison contains duplicate case outcomes")
    if len({record.model for record in records}) > 1:
        raise ValueError("model comparison summary accepts outcomes for one model")

    total = len(records)
    accepted = [record for record in records if record.accepted_analytical_report]
    correct_accepted = [record for record in accepted if record.correct_pattern]
    benign_accepted = [
        record
        for record in accepted
        if expected_by_case[record.case_id]
        is InvestigationPattern.BENIGN_SHARED_IDENTITY
    ]
    benign_cases = {
        case.case_id
        for case in case_records
        if case.expected_pattern is InvestigationPattern.BENIGN_SHARED_IDENTITY
    }
    obvious_abuse = [
        record
        for record in records
        if expected_by_case[record.case_id] in _ABUSE_PATTERNS
    ]
    accepted_obvious_abuse = [
        record for record in obvious_abuse if record.accepted_analytical_report
    ]
    return ModelComparisonSummary(
        total_runs=total,
        accepted_report_rate=_rate(len(accepted), total),
        end_to_end_success_rate=_rate(
            sum(record.end_to_end_success for record in records), total
        ),
        grounding_failure_rate=_rate(
            sum(record.grounding_failure for record in records), total
        ),
        budget_failure_rate=_rate(
            sum(record.budget_failure for record in records), total
        ),
        failed_report_rate=_rate(
            sum(record.failed_report for record in records), total
        ),
        provider_failure_rate=_rate(
            sum(record.provider_request_failure for record in records), total
        ),
        conditional_pattern_accuracy=_rate(len(correct_accepted), len(accepted)),
        benign_false_fraud_accusation_rate=_rate(
            sum(record.false_fraud_accusation is True for record in benign_accepted),
            len(benign_accepted),
        ),
        benign_system_failure_rate=_rate(
            sum(
                not record.accepted_analytical_report
                for record in records
                if record.case_id in benign_cases
            ),
            len(benign_cases),
        ),
        obvious_abuse_reasoning_miss_rate=_rate(
            sum(
                record.missed_obvious_abuse is True for record in accepted_obvious_abuse
            ),
            len(accepted_obvious_abuse),
        ),
        obvious_abuse_system_failure_rate=_rate(
            sum(not record.accepted_analytical_report for record in obvious_abuse),
            len(obvious_abuse),
        ),
    )


def _rate(numerator: int, denominator: int) -> ConditionalRate:
    """Build a denominator-explicit rate, preserving no-support cases."""
    return ConditionalRate(
        numerator=numerator,
        denominator=denominator,
        value=None if denominator == 0 else numerator / denominator,
    )
