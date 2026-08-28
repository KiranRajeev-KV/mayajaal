"""Tests for evaluator-only, eligibility-aware comparison scoring."""

import inspect
import unittest

from mayajaal.investigation import (
    EvaluationCase,
    InvestigationPattern,
    InvestigationStatus,
    score_comparison_run,
    summarize_model_comparison,
)


def case(case_id: str, expected_pattern: InvestigationPattern) -> EvaluationCase:
    """Build one hidden evaluator expectation."""
    return EvaluationCase(case_id=case_id, expected_pattern=expected_pattern)


class InvestigationComparisonScoringTests(unittest.TestCase):
    """System reliability must remain separate from analytical quality."""

    def test_failed_inconclusive_receives_no_pattern_credit(self) -> None:
        outcome = score_comparison_run(
            model="fixture-model",
            case=case("promo", InvestigationPattern.PROMO_RING),
            status=InvestigationStatus.FAILED,
            pattern=InvestigationPattern.INCONCLUSIVE,
        )

        self.assertFalse(outcome.accepted_analytical_report)
        self.assertTrue(outcome.failed_report)
        self.assertIsNone(outcome.correct_pattern)
        self.assertIsNone(outcome.missed_obvious_abuse)
        self.assertFalse(outcome.end_to_end_success)

    def test_budget_exhausted_inconclusive_receives_no_pattern_credit(self) -> None:
        outcome = score_comparison_run(
            model="fixture-model",
            case=case("promo", InvestigationPattern.PROMO_RING),
            status=InvestigationStatus.BUDGET_EXHAUSTED,
            pattern=InvestigationPattern.INCONCLUSIVE,
        )

        self.assertTrue(outcome.budget_failure)
        self.assertFalse(outcome.accepted_analytical_report)
        self.assertIsNone(outcome.correct_pattern)
        self.assertIsNone(outcome.false_fraud_accusation)

    def test_grounding_failure_receives_no_pattern_credit(self) -> None:
        outcome = score_comparison_run(
            model="fixture-model",
            case=case("promo", InvestigationPattern.PROMO_RING),
            status=InvestigationStatus.FAILED,
            pattern=InvestigationPattern.INCONCLUSIVE,
            grounding_failure=True,
        )

        self.assertTrue(outcome.grounding_failure)
        self.assertIsNone(outcome.correct_pattern)
        self.assertIsNone(outcome.appropriate_ambiguity_handling)

    def test_valid_correct_fraud_report_scores_as_end_to_end_success(self) -> None:
        outcome = score_comparison_run(
            model="fixture-model",
            case=case("promo", InvestigationPattern.PROMO_RING),
            status=InvestigationStatus.COMPLETED,
            pattern=InvestigationPattern.PROMO_RING,
        )

        self.assertTrue(outcome.accepted_analytical_report)
        self.assertTrue(outcome.correct_pattern)
        self.assertFalse(outcome.missed_obvious_abuse)
        self.assertTrue(outcome.end_to_end_success)

    def test_benign_inconclusive_is_not_a_false_fraud_accusation(self) -> None:
        outcome = score_comparison_run(
            model="fixture-model",
            case=case("household", InvestigationPattern.BENIGN_SHARED_IDENTITY),
            status=InvestigationStatus.COMPLETED,
            pattern=InvestigationPattern.INCONCLUSIVE,
        )

        self.assertFalse(outcome.correct_pattern)
        self.assertFalse(outcome.false_fraud_accusation)
        self.assertFalse(outcome.end_to_end_success)

    def test_valid_fraud_conclusion_on_benign_case_is_false_accusation(self) -> None:
        outcome = score_comparison_run(
            model="fixture-model",
            case=case("household", InvestigationPattern.BENIGN_SHARED_IDENTITY),
            status=InvestigationStatus.COMPLETED,
            pattern=InvestigationPattern.PROMO_RING,
        )

        self.assertTrue(outcome.false_fraud_accusation)
        self.assertFalse(outcome.correct_pattern)

    def test_obvious_abuse_reasoning_miss_differs_from_system_failure(self) -> None:
        expected = case("refund", InvestigationPattern.REFUND_RING)
        reasoning_miss = score_comparison_run(
            model="fixture-model",
            case=expected,
            status=InvestigationStatus.COMPLETED,
            pattern=InvestigationPattern.INCONCLUSIVE,
        )
        system_failure = score_comparison_run(
            model="fixture-model",
            case=expected,
            status=None,
            pattern=None,
            provider_request_failure=True,
        )

        self.assertTrue(reasoning_miss.missed_obvious_abuse)
        self.assertIsNone(system_failure.missed_obvious_abuse)
        self.assertTrue(system_failure.provider_request_failure)

    def test_ambiguous_inconclusive_or_non_abuse_insufficient_evidence_is_appropriate(
        self,
    ) -> None:
        expected = case("ambiguous", InvestigationPattern.INCONCLUSIVE)
        completed = score_comparison_run(
            model="fixture-model",
            case=expected,
            status=InvestigationStatus.COMPLETED,
            pattern=InvestigationPattern.INCONCLUSIVE,
        )
        insufficient = score_comparison_run(
            model="fixture-model",
            case=expected,
            status=InvestigationStatus.INSUFFICIENT_EVIDENCE,
            pattern=InvestigationPattern.BENIGN_SHARED_IDENTITY,
        )

        self.assertTrue(completed.appropriate_ambiguity_handling)
        self.assertTrue(completed.end_to_end_success)
        self.assertTrue(insufficient.appropriate_ambiguity_handling)
        self.assertTrue(insufficient.end_to_end_success)

    def test_ambiguous_insufficient_evidence_with_abuse_pattern_has_no_credit(
        self,
    ) -> None:
        outcome = score_comparison_run(
            model="fixture-model",
            case=case("ambiguous", InvestigationPattern.INCONCLUSIVE),
            status=InvestigationStatus.INSUFFICIENT_EVIDENCE,
            pattern=InvestigationPattern.PROMO_RING,
        )

        self.assertTrue(outcome.accepted_analytical_report)
        self.assertFalse(outcome.appropriate_ambiguity_handling)
        self.assertFalse(outcome.end_to_end_success)

    def test_provider_failure_cannot_carry_a_status_or_pattern(self) -> None:
        expected = case("promo", InvestigationPattern.PROMO_RING)
        with self.assertRaisesRegex(ValueError, "cannot include a report"):
            score_comparison_run(
                model="fixture-model",
                case=expected,
                status=InvestigationStatus.FAILED,
                pattern=None,
                provider_request_failure=True,
            )
        with self.assertRaisesRegex(ValueError, "cannot include a report"):
            score_comparison_run(
                model="fixture-model",
                case=expected,
                status=None,
                pattern=InvestigationPattern.INCONCLUSIVE,
                provider_request_failure=True,
            )

    def test_summary_requires_exactly_one_outcome_for_every_case(self) -> None:
        cases = (
            case("promo", InvestigationPattern.PROMO_RING),
            case("refund", InvestigationPattern.REFUND_RING),
        )
        promo = score_comparison_run(
            model="fixture-model",
            case=cases[0],
            status=InvestigationStatus.COMPLETED,
            pattern=InvestigationPattern.PROMO_RING,
        )
        unknown = promo.model_copy(update={"case_id": "unknown"})

        with self.assertRaisesRegex(ValueError, "exactly one outcome"):
            summarize_model_comparison((promo,), cases)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize_model_comparison((promo, promo), (cases[0],))
        with self.assertRaisesRegex(ValueError, "unknown"):
            summarize_model_comparison((promo, unknown), cases)

    def test_aggregate_rates_keep_quality_and_reliability_denominators_distinct(
        self,
    ) -> None:
        cases = (
            case("promo", InvestigationPattern.PROMO_RING),
            case("household", InvestigationPattern.BENIGN_SHARED_IDENTITY),
            case("ambiguous", InvestigationPattern.INCONCLUSIVE),
        )
        outcomes = (
            score_comparison_run(
                model="fixture-model",
                case=cases[0],
                status=InvestigationStatus.COMPLETED,
                pattern=InvestigationPattern.PROMO_RING,
            ),
            score_comparison_run(
                model="fixture-model",
                case=cases[1],
                status=InvestigationStatus.FAILED,
                pattern=InvestigationPattern.INCONCLUSIVE,
                grounding_failure=True,
            ),
            score_comparison_run(
                model="fixture-model",
                case=cases[2],
                status=None,
                pattern=None,
                provider_request_failure=True,
            ),
        )

        summary = summarize_model_comparison(outcomes, cases)

        self.assertEqual(summary.accepted_report_rate.numerator, 1)
        self.assertEqual(summary.accepted_report_rate.denominator, 3)
        self.assertEqual(summary.conditional_pattern_accuracy.numerator, 1)
        self.assertEqual(summary.conditional_pattern_accuracy.denominator, 1)
        self.assertEqual(summary.grounding_failure_rate.numerator, 1)
        self.assertEqual(summary.failed_report_rate.numerator, 1)
        self.assertEqual(summary.provider_failure_rate.numerator, 1)
        self.assertEqual(summary.benign_false_fraud_accusation_rate.denominator, 0)
        self.assertIsNone(summary.benign_false_fraud_accusation_rate.value)
        self.assertEqual(summary.benign_system_failure_rate.value, 1.0)
        self.assertEqual(summary.obvious_abuse_reasoning_miss_rate.denominator, 1)
        self.assertEqual(summary.obvious_abuse_system_failure_rate.value, 0.0)

    def test_hidden_expectations_are_not_runtime_or_artifact_contracts(self) -> None:
        import mayajaal.investigation.agent as agent
        import mayajaal.investigation.artifacts as artifacts
        import mayajaal.investigation.service as service

        for module in (agent, artifacts, service):
            self.assertNotIn("EvaluationCase", inspect.getsource(module))


if __name__ == "__main__":
    unittest.main()
