"""Provider-free assembly tests for the manual three-model smoke command."""

import unittest

from openai import OpenAIError

from mayajaal.investigation import (
    InvestigationConfig,
    InvestigationPattern,
    InvestigationStatus,
)
from scripts.run_investigation_model_smoke import (
    MODELS,
    SMOKE_CASE,
    CaptureHandler,
    _failed_run,  # pyright: ignore[reportPrivateUsage]
    _is_provider_request_failure,  # pyright: ignore[reportPrivateUsage]
    build_model_config,
    build_smoke_summary,
    render_smoke_summary,
    score_comparison_run,
    token_summary,
)


class InvestigationModelSmokeScriptTests(unittest.TestCase):
    """The manual command's summaries must be usable without a provider call."""

    def test_fixed_models_and_case_are_the_comparison_smoke_contract(self) -> None:
        self.assertEqual(
            MODELS,
            ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.4-mini-2026-03-17"),
        )
        self.assertEqual(SMOKE_CASE.case_id, "clear_promo_ring")
        self.assertIs(SMOKE_CASE.expected_pattern, InvestigationPattern.PROMO_RING)

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
                case=SMOKE_CASE,
                status=None,
                pattern=None,
                provider_request_failure=True,
            )
            for model in MODELS
        )
        records = tuple(
            {
                "model": model,
                "success": False,
                "token_usage": {},
                "comparison_outcome": outcome.model_dump(mode="json"),
            }
            for model, outcome in zip(MODELS, outcomes, strict=True)
        )

        summary = build_smoke_summary("fixture-run", records, outcomes)
        markdown = render_smoke_summary(summary)

        self.assertEqual(summary["completed_run_count"], 0)
        self.assertEqual(summary["provider_failure_count"], 3)
        self.assertIn("REQUEST_FAILED", markdown)
        self.assertIn("Alias refs valid", markdown)
        self.assertIn("Grounding failure", markdown)
        self.assertIn("Context reconciled", markdown)

    def test_summary_requires_all_three_fixed_models(self) -> None:
        outcome = score_comparison_run(
            model=MODELS[0],
            case=SMOKE_CASE,
            status=InvestigationStatus.COMPLETED,
            pattern=InvestigationPattern.PROMO_RING,
        )
        with self.assertRaisesRegex(ValueError, "exactly one run"):
            build_smoke_summary("fixture-run", ({"model": MODELS[0]},), (outcome,))

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
        )

        self.assertTrue(outcome.harness_failure)
        self.assertFalse(outcome.provider_request_failure)
        self.assertIn("harness_failure", record)
        self.assertNotIn("provider_request_failure", record)

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
