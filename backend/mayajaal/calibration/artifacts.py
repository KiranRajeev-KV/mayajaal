"""Artifact writers and plots for model-neutral calibration reports."""

import json
from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from typing import Any

import polars as pl

from .models import CalibrationEvaluation, CalibrationPrediction, SigmoidCalibrator

plt: Any = import_module("matplotlib.pyplot")


def save_calibration_artifacts(
    output_directory: Path,
    predictions: tuple[CalibrationPrediction, ...],
    evaluation: CalibrationEvaluation,
    *,
    metadata: dict[str, object],
) -> dict[str, Path]:
    """Persist a deterministic calibrator, score table, report, and plots."""
    output_directory.mkdir(parents=True, exist_ok=True)
    calibrator_path = output_directory / "sigmoid_calibrator.json"
    predictions_path = output_directory / "calibration_predictions.parquet"
    evaluation_path = output_directory / "calibration_evaluation.json"
    reliability_path = output_directory / "reliability_diagram.png"
    distribution_path = output_directory / "probability_distribution.png"
    calibrator_path.write_text(
        json.dumps(
            _calibrator_json(evaluation.fit.calibrator, metadata),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _prediction_frame(predictions).write_parquet(predictions_path, compression="zstd")
    evaluation_path.write_text(
        json.dumps(
            {
                "protocol": metadata,
                "calibration": asdict(evaluation),
                "artifacts": {
                    "calibrator": str(calibrator_path),
                    "predictions": str(predictions_path),
                    "reliability_diagram": str(reliability_path),
                    "probability_distribution": str(distribution_path),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _save_reliability_diagram(evaluation, reliability_path)
    _save_probability_distribution(predictions, distribution_path)
    return {
        "calibrator": calibrator_path,
        "predictions": predictions_path,
        "evaluation": evaluation_path,
        "reliability_diagram": reliability_path,
        "probability_distribution": distribution_path,
    }


def _prediction_frame(predictions: tuple[CalibrationPrediction, ...]) -> pl.DataFrame:
    return pl.DataFrame(
        [asdict(item) for item in predictions],
        schema={
            "sample_id": pl.String,
            "account_id": pl.String,
            "decision_time": pl.Datetime(time_zone="UTC"),
            "split": pl.String,
            "y_true": pl.Boolean,
            "raw_model_score": pl.Float64,
            "uncalibrated_probability": pl.Float64,
            "calibrated_probability": pl.Float64,
        },
        strict=True,
    )


def _calibrator_json(
    calibrator: SigmoidCalibrator | None, metadata: dict[str, object]
) -> dict[str, object]:
    provenance = {
        "base_model_id": metadata.get("base_model_id"),
        "frozen_provenance": metadata.get("frozen_provenance"),
    }
    if calibrator is None:
        return {"status": "INVALID", "parameters": None, "provenance": provenance}
    return {
        "status": "VALID",
        "parameters": asdict(calibrator),
        "provenance": provenance,
    }


def _save_reliability_diagram(
    evaluation: CalibrationEvaluation, output_path: Path
) -> None:
    figure, axes = plt.subplots()
    axes.plot((0.0, 1.0), (0.0, 1.0), "--", color="black", label="perfect calibration")
    _plot_reliability(axes, evaluation.test_uncalibrated, "uncalibrated")
    if evaluation.test_calibrated is not None:
        _plot_reliability(axes, evaluation.test_calibrated, "sigmoid calibrated")
    axes.set(
        xlabel="Mean predicted probability",
        ylabel="Observed prevalence",
        title="Held-out reliability diagram",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    axes.legend()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _plot_reliability(axes: Any, metrics: Any, label: str) -> None:
    bins = metrics.reliability_bins
    axes.plot(
        [item.mean_predicted_probability for item in bins],
        [item.observed_prevalence for item in bins],
        marker="o",
        label=label,
    )


def _save_probability_distribution(
    predictions: tuple[CalibrationPrediction, ...], output_path: Path
) -> None:
    test = [item for item in predictions if item.split == "test"]
    figure, axes = plt.subplots()
    for probability_field, label, style in (
        ("uncalibrated_probability", "uncalibrated", "solid"),
        ("calibrated_probability", "sigmoid calibrated", "dashed"),
    ):
        values = [getattr(item, probability_field) for item in test]
        if any(value is None for value in values):
            continue
        axes.hist(
            [float(value) for value in values],
            bins=20,
            range=(0.0, 1.0),
            histtype="step",
            linestyle=style,
            label=label,
        )
    axes.set(
        xlabel="Predicted probability",
        ylabel="Held-out sample count",
        title="Held-out probability distribution",
    )
    if axes.get_legend_handles_labels()[0]:
        axes.legend()
    else:
        axes.text(
            0.5, 0.5, "No fitted calibration probabilities", ha="center", va="center"
        )
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
