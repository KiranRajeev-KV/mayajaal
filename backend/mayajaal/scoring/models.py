"""Model-neutral contracts for raw scores derived from feature vectors."""

from dataclasses import dataclass
from math import isfinite

from mayajaal.schemas.common import AwareDatetime


@dataclass(frozen=True)
class ScoreObservation:
    """One account score bound to its feature-vector point-in-time input."""

    score_id: str
    base_model_id: str
    subject_id: str
    scoring_cutoff: AwareDatetime
    raw_model_score: float
    feature_vector_id: str

    def __post_init__(self) -> None:
        if not self.score_id or not self.base_model_id:
            raise ValueError("score observation requires non-empty model lineage")
        if not self.subject_id or not self.feature_vector_id:
            raise ValueError(
                "score observation requires subject and feature-vector IDs"
            )
        if (
            self.scoring_cutoff.tzinfo is None
            or self.scoring_cutoff.utcoffset() is None
        ):
            raise ValueError("scoring_cutoff must include a timezone offset")
        if not isfinite(self.raw_model_score):
            raise ValueError("raw model score must be finite")
