"""TOML configuration loading for manual synthetic-world generation."""

import tomllib
from pathlib import Path

from pydantic import Field

from mayajaal.calibration import CalibrationConfig
from mayajaal.evaluation import EvaluationConfig
from mayajaal.schemas.common import SchemaModel

from .profile import GenerationProfile


class OutputConfig(SchemaModel):
    """Non-secret output settings for one generation run."""

    directory: str = Field(min_length=1)


class GenerationConfig(SchemaModel):
    """The file-backed configuration accepted by the generation script."""

    synthetic_world: GenerationProfile
    evaluation: EvaluationConfig
    calibration: CalibrationConfig
    output: OutputConfig


def load_generation_config(path: Path) -> GenerationConfig:
    """Load and validate a UTF-8 TOML generation configuration file."""
    with path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)
    return GenerationConfig.model_validate(raw_config)
