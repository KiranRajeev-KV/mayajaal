"""Tests for the file-backed manual generation configuration."""

import unittest
from pathlib import Path

from mayajaal.calibration import CalibrationConfig
from mayajaal.evaluation import EvaluationConfig
from mayajaal.investigation import InvestigationConfig, ReasoningEffort
from mayajaal.policy import PolicyConfig
from mayajaal.synthetic.config import load_generation_config
from mayajaal.synthetic.profile import DiagnosticProfile, PrevalenceProfile


class SyntheticConfigTests(unittest.TestCase):
    def test_checked_in_config_loads(self) -> None:
        config_path = Path(__file__).parents[1] / "config.toml"
        config = load_generation_config(config_path)

        self.assertEqual(config.synthetic_world.seed, 20260824)
        self.assertEqual(config.synthetic_world.difficulty, "standard")
        self.assertEqual(config.synthetic_world.prevalence.preset, "development")
        self.assertEqual(
            config.synthetic_world.prevalence.target_labelled_account_rate, 0.03
        )
        self.assertEqual(config.synthetic_world.validation.multi_seed_count, 5)
        self.assertEqual(config.synthetic_world.validation.shap_sample_count, 1000)
        self.assertEqual(
            config.synthetic_world.active_difficulty.campaign_sharing_multiplier,
            1.0,
        )
        self.assertIsNone(config.synthetic_world.shared_household_count)
        self.assertIsNone(config.synthetic_world.population.benign_network_group_count)
        self.assertEqual(
            config.synthetic_world.population.households_per_thousand_ordinary_accounts,
            35.0,
        )
        self.assertEqual(
            config.synthetic_world.prevalence.ring_sizes, (2, 3, 4, 5, 6, 8)
        )
        self.assertEqual(
            config.synthetic_world.prevalence.minimum_campaigns_per_timeline_bucket,
            2,
        )
        self.assertEqual(
            config.synthetic_world.diagnostics.min_cutoff_positive_samples, 50
        )
        self.assertEqual(
            config.synthetic_world.diagnostics.cutoff_fractions, (0.25, 0.50, 1.00)
        )
        self.assertEqual(config.output.directory, "artifacts/synthetic-world")
        self.assertEqual(config.evaluation.train_end_fraction, 0.25)
        self.assertEqual(config.evaluation.validation_end_fraction, 0.50)
        self.assertEqual(config.evaluation.minimum_positive_samples, 10)
        self.assertEqual(config.calibration.method, "sigmoid")
        self.assertEqual(config.calibration.quantile_bin_count, 10)
        self.assertEqual(config.policy.review_operational_cost_paise, 1500)
        self.assertEqual(config.policy.sensitivity.stressed_odds_multiplier, 2.0)
        self.assertEqual(config.investigation.max_tool_calls, 8)
        self.assertEqual(config.investigation.max_risk_drivers, 5)
        self.assertEqual(config.investigation.max_timeline_events, 20)
        self.assertEqual(config.investigation.model_name, "gpt-5.6-terra")
        self.assertIs(config.investigation.reasoning_effort, ReasoningEffort.MEDIUM)
        self.assertFalse(config.investigation.triggers.investigate_review)
        self.assertFalse(config.investigation.triggers.investigate_block)
        self.assertFalse(config.investigation.triggers.investigate_unstable_allow)

    def test_new_distribution_and_diagnostic_knobs_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            _ = PrevalenceProfile(ring_sizes=(2, 4), ring_size_weights=(0.5, 0.5))
        with self.assertRaises(ValueError):
            _ = DiagnosticProfile(cutoff_fractions=(0.50, 0.25, 1.00))
        with self.assertRaises(ValueError):
            _ = EvaluationConfig(train_end_fraction=0.50, validation_end_fraction=0.50)
        with self.assertRaises(ValueError):
            _ = EvaluationConfig(false_positive_review_cost_paise=1)
        with self.assertRaises(ValueError):
            _ = CalibrationConfig(quantile_bin_count=1)
        with self.assertRaises(ValueError):
            _ = PolicyConfig(block_fraud_residual_loss_fraction=1.1)
        with self.assertRaises(ValueError):
            _ = InvestigationConfig(max_related_accounts=0)
        with self.assertRaises(ValueError):
            _ = InvestigationConfig(max_risk_drivers=0)
        with self.assertRaises(ValueError):
            _ = InvestigationConfig(max_timeline_events=0)
        with self.assertRaises(ValueError):
            _ = InvestigationConfig(model_name="")
        with self.assertRaises(ValueError):
            _ = InvestigationConfig.model_validate({"reasoning_effort": "unsupported"})

    def test_reasoning_effort_known_values_are_validated(self) -> None:
        for value in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
            self.assertEqual(
                InvestigationConfig.model_validate(
                    {"reasoning_effort": value}
                ).reasoning_effort,
                value,
            )


if __name__ == "__main__":
    _ = unittest.main()
