"""Tests for the file-backed manual generation configuration."""

import unittest
from pathlib import Path

from mayajaal.synthetic.config import load_generation_config


class SyntheticConfigTests(unittest.TestCase):
    def test_checked_in_config_loads(self) -> None:
        config_path = Path(__file__).parents[1] / "config.toml"
        config = load_generation_config(config_path)

        self.assertEqual(config.synthetic_world.seed, 20260824)
        self.assertEqual(config.synthetic_world.difficulty, "standard")
        self.assertEqual(
            config.synthetic_world.population.benign_network_group_count, 6
        )
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
        self.assertEqual(
            config.synthetic_world.diagnostics.cutoff_fractions, (0.25, 0.50, 1.00)
        )
        self.assertEqual(config.output.directory, "artifacts/synthetic-world")


if __name__ == "__main__":
    _ = unittest.main()
