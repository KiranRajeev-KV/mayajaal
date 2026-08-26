"""Tests for the file-backed manual generation configuration."""

import unittest
from pathlib import Path

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
            config.synthetic_world.diagnostics.min_cutoff_positive_samples, 5
        )
        self.assertEqual(
            config.synthetic_world.diagnostics.cutoff_fractions, (0.25, 0.50, 1.00)
        )
        self.assertEqual(config.output.directory, "artifacts/synthetic-world")

    def test_new_distribution_and_diagnostic_knobs_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            _ = PrevalenceProfile(ring_sizes=(2, 4), ring_size_weights=(0.5, 0.5))
        with self.assertRaises(ValueError):
            _ = DiagnosticProfile(cutoff_fractions=(0.50, 0.25, 1.00))


if __name__ == "__main__":
    _ = unittest.main()
