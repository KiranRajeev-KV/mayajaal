"""Provider-free assembly tests for the manual three-model comparison command."""

import unittest

from openai import OpenAIError

from mayajaal.investigation import (
    InvestigationConfig,
    InvestigationPattern,
    InvestigationStatus,
)
from scripts.run_investigation_model_comparison import (
    COMPARISON_FIXTURES,
    MODELS,
    CaptureHandler,
    _comparison_status_label,  # pyright: ignore[reportPrivateUsage]
    _failed_run,  # pyright: ignore[reportPrivateUsage]
    _is_provider_request_failure,  # pyright: ignore[reportPrivateUsage]
    build_comparison_summary,
    build_model_config,
    planned_runs,
    render_comparison_summary,
    score_comparison_run,
    token_summary,
)


class InvestigationModelComparisonRunnerTests(unittest.TestCase):
    """The manual command's summaries must be usable without a provider call."""

    def test_frozen_plan_has_six_unique_runtime_cases_and_eighteen_runs(self) -> None:
        self.assertEqual(
            MODELS,
            ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.4-mini-2026-03-17"),
        )
        self.assertEqual(len(COMPARISON_FIXTURES), 6)
        self.assertEqual(len(planned_runs()), 18)
        self.assertEqual(
            tuple(fixture.case.runtime_context_id for fixture in COMPARISON_FIXTURES),
            tuple(f"eval_case_{index:03d}" for index in range(1, 7)),
        )
        self.assertEqual(
            len({fixture.case.runtime_context_id for fixture in COMPARISON_FIXTURES}),
            6,
        )

    def test_model_config_keeps_medium_effort_without_mutating_source_config(
        self,
    ) -> None:
        source = InvestigationConfig(model_name="source-model")
        configured = build_model_config(source, MODELS[0])

        self.assertEqual(configured.model_name, MODELS[0])
        self.assertEqual(configured.reasoning_effort.value, "medium")
        self.assertEqual(source.model_name, "source-model")

    def test_summary_and_markdown_expose_provider_failures_explicitly(self) -> None:
        outcomes = tuple(
            score_comparison_run(
                model=model,
                case=fixture.case,
                status=None,
                pattern=None,
                provider_request_failure=True,
            )
            for fixture, model in planned_runs()
        )
        records = tuple(
            {
                "model": model,
                "case_id": fixture.case.case_id,
                "success": False,
                "token_usage": {},
                "comparison_outcome": outcome.model_dump(mode="json"),
            }
            for (fixture, model), outcome in zip(planned_runs(), outcomes, strict=True)
        )

        summary = build_comparison_summary("fixture-run", records, outcomes)
        markdown = render_comparison_summary(summary)

        self.assertEqual(summary["completed_run_count"], 0)
        self.assertEqual(summary["provider_failure_count"], 18)
        self.assertIn("PROVIDER_FAILED", markdown)
        self.assertIn("Alias refs valid", markdown)
        self.assertIn("Grounding failure", markdown)
        self.assertIn("Context reconciled", markdown)

    def test_summary_requires_all_fixed_case_model_pairs(self) -> None:
        outcome = score_comparison_run(
            model=MODELS[0],
            case=COMPARISON_FIXTURES[0].case,
            status=InvestigationStatus.COMPLETED,
            pattern=InvestigationPattern.PROMO_RING,
        )
        with self.assertRaisesRegex(ValueError, "every fixed case/model pair"):
            build_comparison_summary(
                "fixture-run",
                ({"model": MODELS[0], "case_id": COMPARISON_FIXTURES[0].case.case_id},),
                (outcome,),
            )

    def test_provider_and_harness_failures_are_classified_separately(self) -> None:
        self.assertTrue(_is_provider_request_failure(OpenAIError("provider")))
        self.assertFalse(_is_provider_request_failure(RuntimeError("local")))

        record, outcome = _failed_run(
            record={"model": MODELS[0]},
            model_name=MODELS[0],
            callback=CaptureHandler(),
            started_clock=0.0,
            error=RuntimeError("local artifact verification failed"),
            provider_request_failure=False,
            case=COMPARISON_FIXTURES[0].case,
        )

        self.assertTrue(outcome.harness_failure)
        self.assertFalse(outcome.provider_request_failure)
        self.assertIn("harness_failure", record)
        self.assertNotIn("provider_request_failure", record)

    def test_comparison_status_labels_distinguish_failure_origins(self) -> None:
        self.assertEqual(
            _comparison_status_label({"provider_request_failure": {}}),
            "PROVIDER_FAILED",
        )
        self.assertEqual(
            _comparison_status_label({"harness_failure": {}}), "HARNESS_FAILED"
        )
        self.assertEqual(
            _comparison_status_label(
                {
                    "reported_status": "FAILED",
                    "grounding_failure_code": "INVALID_STRUCTURED_OUTPUT",
                }
            ),
            "FAILED / INVALID_STRUCTURED_OUTPUT",
        )
        self.assertEqual(
            _comparison_status_label({"reported_status": "BUDGET_EXHAUSTED"}),
            "BUDGET_EXHAUSTED",
        )

    def test_token_summary_keeps_provider_cache_and_reasoning_metadata(self) -> None:
        summary = token_summary(
            (
                {
                    "usage_metadata": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                        "input_token_details": {
                            "cache_read": 3,
                            "cache_creation": 4,
                        },
                        "output_token_details": {"reasoning": 2},
                    }
                },
            )
        )

        self.assertEqual(
            summary,
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "reasoning_tokens": 2,
                "cache_read_tokens": 3,
                "cache_creation_tokens": 4,
                "total_tokens": 15,
            },
        )
