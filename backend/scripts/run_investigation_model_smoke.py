"""Run one fixed investigation case sequentially through three OpenAI models.

This is a deliberately small, manual live smoke test for the frozen
investigation-comparison setup.  It does not query Admin Usage or Costs APIs,
and it never retries a provider request after a usable result.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from openai import OpenAIError

from mayajaal.calibration import estimate_probability, load_probability_model
from mayajaal.evaluation import load_frozen_full_evaluation
from mayajaal.features import FeatureService
from mayajaal.graph import build_graph_projection
from mayajaal.investigation import (
    ComparisonRunOutcome,
    EvaluationCase,
    EvidenceService,
    InvestigationAgentService,
    InvestigationConfig,
    InvestigationPattern,
    InvestigationRequest,
    InvestigationStatus,
    ModelFacingContextMetrics,
    load_investigation_artifacts,
    save_investigation_artifacts,
    score_comparison_run,
    summarize_model_comparison,
)
from mayajaal.policy import (
    DecisionContext,
    PolicyConfig,
    build_policy_model,
    decide,
)
from mayajaal.resolution import resolve_all
from mayajaal.scoring.service import score_feature_vector
from mayajaal.synthetic import generate_world, profile_for_total_accounts
from mayajaal.synthetic.config import load_generation_config

MODELS = (
    "gpt-5.6-luna",
    "gpt-5.6-terra",
    "gpt-5.4-mini-2026-03-17",
)
SMOKE_CASE = EvaluationCase(
    case_id="clear_promo_ring",
    runtime_context_id="eval_case_001",
    expected_pattern=InvestigationPattern.PROMO_RING,
)
# This is the deterministic representative selected by the prior comparison's
# existing clear_promo_ring fixture for the configured 10k standard world.
SMOKE_ACCOUNT_ID = "2d64b1b1-b27c-5d63-8c44-7e5c2e977ed9"
SMOKE_CASE_DESCRIPTION = "clear promo ring"


class CaptureHandler(BaseCallbackHandler):
    """Capture non-secret provider metadata and token accounting per run."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, object]] = []

    def on_llm_end(self, response: LLMResult, **_: object) -> None:
        for generation_set in response.generations:
            for generation in generation_set:
                message = getattr(generation, "message", None)
                response_metadata = getattr(message, "response_metadata", {})
                usage_metadata = getattr(message, "usage_metadata", {})
                self.calls.append(
                    {
                        "response_id": _mapping_value(response_metadata, "id"),
                        "model_name": _mapping_value(response_metadata, "model_name"),
                        "service_tier": _mapping_value(
                            response_metadata, "service_tier"
                        ),
                        "usage_metadata": _json_safe(usage_metadata),
                    }
                )


def parse_arguments() -> argparse.Namespace:
    """Parse artifact locations without making a provider request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=Path("artifacts/held-out-standard-10k-final"),
        help="Frozen full-evaluation artifact directory, relative to config.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("artifacts/calibration-standard-10k-final"),
        help="Frozen calibration artifact directory, relative to config.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/investigation-model-smoke"),
        help="Timestamped smoke-run directory is created below this root.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the fixed case once per configured model, sequentially."""
    arguments = parse_arguments()
    config_path = arguments.config.resolve()
    backend_directory = config_path.parent
    _load_openai_api_key(backend_directory / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY must be set in the environment or backend/.env"
        )

    config = load_generation_config(config_path)
    profile = profile_for_total_accounts(
        config.synthetic_world,
        config.synthetic_world.validation.full_account_count,
    )
    world = generate_world(profile)
    resolution = resolve_all(
        accounts=world.accounts,
        addresses=world.addresses,
        ip_addresses=world.ip_addresses,
        payment_identities=world.payment_identities,
        devices=world.devices,
    )
    projection = build_graph_projection(world, resolution)
    feature_service = FeatureService(projection)
    evaluation_directory = _resolve_from_config(
        backend_directory, arguments.evaluation_dir
    )
    frozen = load_frozen_full_evaluation(
        evaluation_directory,
        expected_profile=profile,
        expected_evaluation_config=config.evaluation,
    )
    calibration_directory = _resolve_from_config(
        backend_directory, arguments.calibration_dir
    )
    probability_model = load_probability_model(
        calibration_directory / "sigmoid_calibrator.json",
        expected_base_model_id=frozen.base_model_id,
    )
    cutoff = frozen.manifest.test_cutoff
    label_free_events = tuple(
        event.model_copy(update={"synthetic_labels": None}) for event in world.events
    )
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-investigation-model-smoke"
    output_directory = (
        _resolve_from_config(backend_directory, arguments.output_root) / run_id
    )
    output_directory.mkdir(parents=True, exist_ok=False)
    _write_json(
        output_directory / "manifest.json",
        {
            "run_id": run_id,
            "models": list(MODELS),
            "case": {
                "case_id": SMOKE_CASE.case_id,
                "account_id": SMOKE_ACCOUNT_ID,
                "expected_case": SMOKE_CASE_DESCRIPTION,
                "expected_pattern": SMOKE_CASE.expected_pattern.value,
            },
            "reasoning_effort": "medium",
            "provider_max_retries": 0,
            "run_count": len(MODELS),
            "cutoff": cutoff.isoformat(),
            "frozen_evaluation_directory": str(evaluation_directory),
            "calibration_directory": str(calibration_directory),
            "billing_reconciliation": "not queried by this smoke script",
        },
    )

    records: list[dict[str, object]] = []
    outcomes: list[ComparisonRunOutcome] = []
    for model_name in MODELS:
        record, outcome = _run_one_model(
            model_name=model_name,
            investigation_config=config.investigation,
            policy_config=config.policy,
            projection=projection,
            events=label_free_events,
            feature_service=feature_service,
            frozen=frozen,
            probability_model=probability_model,
            world=world,
            cutoff=cutoff,
            output_directory=output_directory,
        )
        records.append(record)
        outcomes.append(outcome)
        _write_json(
            output_directory / "runs" / model_name / "comparison_record.json", record
        )
        print(f"{model_name}: {_smoke_status_label(record)}", flush=True)

    summary = build_smoke_summary(run_id, records, outcomes)
    _write_json(output_directory / "smoke_summary.json", summary)
    (output_directory / "smoke_summary.md").write_text(
        render_smoke_summary(summary), encoding="utf-8"
    )
    print(output_directory)
    return 0


def _run_one_model(
    *,
    model_name: str,
    investigation_config: InvestigationConfig,
    policy_config: PolicyConfig,
    projection: object,
    events: tuple[object, ...],
    feature_service: FeatureService,
    frozen: object,
    probability_model: object,
    world: object,
    cutoff: object,
    output_directory: Path,
) -> tuple[dict[str, object], ComparisonRunOutcome]:
    """Execute exactly one provider attempt and always create an outcome."""
    from mayajaal.calibration import ProbabilityModel
    from mayajaal.evaluation import FrozenFullEvaluation
    from mayajaal.graph import GraphProjection
    from mayajaal.schemas.common import AwareDatetime
    from mayajaal.synthetic import SyntheticWorld

    frozen_evaluation = cast(FrozenFullEvaluation, frozen)
    verified_probability_model = cast(ProbabilityModel, probability_model)
    graph_projection = cast(GraphProjection, projection)
    synthetic_world = cast(SyntheticWorld, world)
    decision_cutoff = cast(AwareDatetime, cutoff)
    run_directory = output_directory / "runs" / model_name
    callback = CaptureHandler()
    started_at = datetime.now(UTC)
    started_clock = time.perf_counter()
    record: dict[str, object] = {
        "model": model_name,
        "case_id": SMOKE_CASE.case_id,
        "hidden_expected_case": SMOKE_CASE_DESCRIPTION,
        "hidden_expected_pattern": SMOKE_CASE.expected_pattern.value,
        "account_id": SMOKE_ACCOUNT_ID,
        "started_at": started_at.isoformat(),
    }
    try:
        run_config = build_model_config(investigation_config, model_name)
        run_directory.mkdir(parents=True, exist_ok=True)
        evidence_service = EvidenceService(
            projection=graph_projection,
            events=cast(tuple[Any, ...], events),
            feature_service=feature_service,
            frozen_evaluation=frozen_evaluation,
            config=run_config,
        )
        vector = feature_service.extract(SMOKE_ACCOUNT_ID, decision_cutoff)
        score = score_feature_vector(frozen_evaluation, vector)
        probability = estimate_probability(
            verified_probability_model,
            score,
            scoring_context_id=SMOKE_CASE.runtime_context_id,
        )
        policy_model = build_policy_model(verified_probability_model, policy_config)
        decision = decide(
            policy_model,
            verified_probability_model,
            score,
            probability,
            DecisionContext(
                exposure_paise=_exposure(
                    synthetic_world, SMOKE_ACCOUNT_ID, decision_cutoff
                ),
                context_id=SMOKE_CASE.runtime_context_id,
            ),
        )
        request = InvestigationRequest.from_policy_decision(
            decision,
            verified_probability_model,
            score,
            probability,
        )
        agent_service = InvestigationAgentService(
            config=run_config,
            callbacks=(callback,),
            max_retries=0,
        )
    except Exception as error:
        return _failed_run(
            record=record,
            model_name=model_name,
            callback=callback,
            started_clock=started_clock,
            error=error,
            provider_request_failure=False,
        )

    try:
        execution = agent_service.run_execution(
            request=request,
            evidence_service=evidence_service,
            score_observation=score,
        )
    except Exception as error:
        return _failed_run(
            record=record,
            model_name=model_name,
            callback=callback,
            started_clock=started_clock,
            error=error,
            provider_request_failure=_is_provider_request_failure(error),
        )

    try:
        paths = save_investigation_artifacts(run_directory, execution)
        # Re-loading without another model call proves the exact artifacts produced
        # by this run remain grounded, cutoff-bound, and provenance-valid.
        loaded = load_investigation_artifacts(
            run_directory,
            request,
            run_config,
            agent_model_id=execution.agent_model_id,
        )
        ended_at = datetime.now(UTC)
        report = execution.report
        outcome = score_comparison_run(
            model=model_name,
            case=SMOKE_CASE,
            status=report.status,
            pattern=report.pattern,
            grounding_failure=execution.grounding_failure is not None,
            model_facing_context_metrics=execution.model_facing_context_metrics,
            model_facing_tool_call_metrics=execution.model_facing_tool_call_metrics,
        )
        record.update(
            _successful_record(
                report_status=report.status,
                report_pattern=report.pattern,
                execution=execution,
                artifact_paths=paths,
                output_directory=output_directory,
                artifacts_verified=loaded.report == execution.report,
                started_clock=started_clock,
                ended_at=ended_at,
                callback=callback,
                decision=decision,
                probability_model_id=probability.probability_model_id,
                probability_estimate_id=probability.probability_estimate_id,
                score_id=score.score_id,
                outcome=outcome,
            )
        )
    except Exception as error:
        return _failed_run(
            record=record,
            model_name=model_name,
            callback=callback,
            started_clock=started_clock,
            error=error,
            provider_request_failure=False,
        )
    return record, outcome


def _failed_run(
    *,
    record: dict[str, object],
    model_name: str,
    callback: CaptureHandler,
    started_clock: float,
    error: Exception,
    provider_request_failure: bool,
) -> tuple[dict[str, object], ComparisonRunOutcome]:
    """Persist one explicit provider or local-harness failure outcome."""
    outcome = score_comparison_run(
        model=model_name,
        case=SMOKE_CASE,
        status=None,
        pattern=None,
        provider_request_failure=provider_request_failure,
        harness_failure=not provider_request_failure,
    )
    failure_key = (
        "provider_request_failure" if provider_request_failure else "harness_failure"
    )
    record.update(
        {
            "success": False,
            "ended_at": datetime.now(UTC).isoformat(),
            "latency_seconds": time.perf_counter() - started_clock,
            failure_key: {
                "type": type(error).__name__,
                "message": str(error),
            },
            "model_responses": callback.calls,
            "token_usage": token_summary(callback.calls),
            "comparison_outcome": outcome.model_dump(mode="json"),
        }
    )
    return record, outcome


def _is_provider_request_failure(error: Exception) -> bool:
    """Classify only an OpenAI SDK error (including a chained cause) as provider-side."""
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, OpenAIError):
            return True
        current = current.__cause__ or current.__context__
    return False


def _successful_record(
    *,
    report_status: InvestigationStatus,
    report_pattern: InvestigationPattern,
    execution: object,
    artifact_paths: Mapping[str, Path],
    output_directory: Path,
    artifacts_verified: bool,
    started_clock: float,
    ended_at: datetime,
    callback: CaptureHandler,
    decision: object,
    probability_model_id: str,
    probability_estimate_id: str,
    score_id: str,
    outcome: ComparisonRunOutcome,
) -> dict[str, object]:
    """Serialize one accepted, bounded execution without exposing credentials."""
    from mayajaal.investigation import InvestigationExecution
    from mayajaal.policy import PolicyDecision

    run = cast(InvestigationExecution, execution)
    policy_decision = cast(PolicyDecision, decision)
    provenance_document = json.loads(
        artifact_paths["provenance"].read_text(encoding="utf-8")
    )
    report_document = json.loads(artifact_paths["report"].read_text(encoding="utf-8"))
    return {
        "success": True,
        "ended_at": ended_at.isoformat(),
        "latency_seconds": time.perf_counter() - started_clock,
        "reported_status": report_status.value,
        "reported_pattern": report_pattern.value,
        "grounding_diagnostic": (
            None
            if run.grounding_failure is None
            else run.grounding_failure.model_dump(mode="json")
        ),
        "accepted_analytical_report": outcome.accepted_analytical_report,
        "end_to_end_success": outcome.end_to_end_success,
        # Alias resolution is a narrower concern than all grounding/protocol
        # validation. For example, a wrong timeline evidence type can fail
        # grounding after every alias resolved correctly.
        "aliases_resolved": _aliases_resolved(run.grounding_failure),
        "grounding_failure_code": (
            None if run.grounding_failure is None else run.grounding_failure.code.value
        ),
        "context_metrics_reconciled": True,
        "comparison_outcome": outcome.model_dump(mode="json"),
        "model_calls": run.report.usage.iterations,
        "tool_calls": run.report.usage.tool_calls,
        "token_usage": token_summary(callback.calls),
        "model_responses": callback.calls,
        "model_facing_context_metrics": _metric_dump(run.model_facing_context_metrics),
        "model_facing_tool_call_metrics": [
            item.model_dump(mode="json") for item in run.model_facing_tool_call_metrics
        ],
        "tool_trace": [
            item.model_dump(mode="json") for item in run.snapshot.tool_trace
        ],
        "investigation_id": provenance_document["provenance"]["investigation_id"],
        "report_id": report_document["report_id"],
        "diagnostic_id": provenance_document.get("diagnostic_id"),
        "decision_id": policy_decision.decision_id,
        "probability_model_id": probability_model_id,
        "probability_estimate_id": probability_estimate_id,
        "score_id": score_id,
        "policy_action": run.report.policy_action.value,
        "policy_action_preserved": run.report.policy_action
        is run.report.request.policy_action,
        "artifacts_verified": artifacts_verified,
        "artifacts": {
            name: str(path.relative_to(output_directory))
            for name, path in artifact_paths.items()
        },
    }


def _aliases_resolved(grounding_failure: object) -> bool:
    """Report only actual malformed/unknown alias failures as unresolved."""
    from mayajaal.investigation import GroundingFailureDiagnostic
    from mayajaal.investigation.errors import GroundingFailureCode

    diagnostic = cast(GroundingFailureDiagnostic | None, grounding_failure)
    return diagnostic is None or diagnostic.code not in {
        GroundingFailureCode.MALFORMED_EVIDENCE_REFERENCE,
        GroundingFailureCode.UNKNOWN_EVIDENCE_REFERENCE,
    }


def _smoke_status_label(run: Mapping[str, object]) -> str:
    """Render provider, harness, budget, and grounded failures distinctly."""
    if "provider_request_failure" in run:
        return "PROVIDER_FAILED"
    if "harness_failure" in run:
        return "HARNESS_FAILED"
    # The persisted per-run record carries the failure object above. This
    # fallback keeps a summary accurate if it is reconstructed from its stable
    # comparison outcome alone.
    comparison = run.get("comparison_outcome")
    if isinstance(comparison, Mapping):
        if comparison.get("provider_request_failure") is True:
            return "PROVIDER_FAILED"
        if comparison.get("harness_failure") is True:
            return "HARNESS_FAILED"
    status = run.get("reported_status")
    if status == InvestigationStatus.FAILED.value:
        failure = run.get("grounding_failure_code")
        return f"FAILED / {failure}" if isinstance(failure, str) else "FAILED"
    if isinstance(status, str):
        return status
    return "HARNESS_FAILED"


def build_smoke_summary(
    run_id: str,
    records: Iterable[Mapping[str, object]],
    outcomes: Iterable[ComparisonRunOutcome],
) -> dict[str, object]:
    """Build an evaluator-only, transparent summary for exactly three runs."""
    records_tuple = tuple(records)
    outcomes_by_model = {outcome.model: outcome for outcome in outcomes}
    if len(records_tuple) != len(MODELS) or len(outcomes_by_model) != len(MODELS):
        raise ValueError("smoke summary requires exactly one run for each fixed model")
    if {str(record["model"]) for record in records_tuple} != set(MODELS):
        raise ValueError("smoke records must cover the fixed model list exactly")
    per_model = {
        model: {
            "comparison": summarize_model_comparison(
                (outcomes_by_model[model],), (SMOKE_CASE,)
            ).model_dump(mode="json"),
            "run": next(record for record in records_tuple if record["model"] == model),
        }
        for model in MODELS
    }
    return {
        "run_id": run_id,
        "models": list(MODELS),
        "case_id": SMOKE_CASE.case_id,
        "expected_pattern": SMOKE_CASE.expected_pattern.value,
        "completed_run_count": sum(
            bool(record.get("success")) for record in records_tuple
        ),
        "provider_failure_count": sum(
            outcome.provider_request_failure for outcome in outcomes_by_model.values()
        ),
        "harness_failure_count": sum(
            outcome.harness_failure for outcome in outcomes_by_model.values()
        ),
        "billing_reconciliation": "not queried by this smoke script",
        "per_model": per_model,
    }


def render_smoke_summary(summary: Mapping[str, object]) -> str:
    """Render a compact human-facing smoke status without token estimates."""
    per_model = cast(dict[str, dict[str, object]], summary["per_model"])
    lines = [
        "# Mayajaal investigation-model smoke test",
        "",
        "Admin Usage and Costs APIs were **not queried**.",
        "",
        "| Model | Status | Pattern | Grounded | Alias refs valid | Grounding failure | Context reconciled | Artifact verified | Expected PROMO_RING | Tool calls | Input | Output | Reasoning | Context bytes |",
        "|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        run = per_model[model]["run"]
        if not isinstance(run, dict):
            raise ValueError("invalid smoke summary run")
        tokens = cast(dict[str, object], run.get("token_usage", {}))
        metrics = cast(dict[str, object], run.get("model_facing_context_metrics", {}))
        expected = run.get("end_to_end_success") is True
        grounded = (
            run.get("grounding_diagnostic") is None and run.get("success") is True
        )
        lines.append(
            "| {model} | {status} | {pattern} | {grounded} | {aliases} | {failure} | {metrics} | "
            "{verified} | {expected} | {tool_calls} | {input_tokens} | {output_tokens} | "
            "{reasoning_tokens} | {bytes} |".format(
                model=model,
                status=_smoke_status_label(run),
                pattern=run.get("reported_pattern", "—"),
                grounded="yes" if grounded else "no",
                aliases="yes" if run.get("aliases_resolved") else "no",
                failure=run.get("grounding_failure_code", "—"),
                metrics="yes" if run.get("context_metrics_reconciled") else "no",
                verified="yes" if run.get("artifacts_verified") else "no",
                expected="yes" if expected else "no",
                tool_calls=run.get("tool_calls", 0),
                input_tokens=tokens.get("input_tokens", 0),
                output_tokens=tokens.get("output_tokens", 0),
                reasoning_tokens=tokens.get("reasoning_tokens", 0),
                bytes=metrics.get("model_facing_serialized_bytes", 0),
            )
        )
    return "\n".join(lines) + "\n"


def token_summary(calls: Iterable[Mapping[str, object]]) -> dict[str, int]:
    """Sum provider-reported token metadata; absent fields remain explicit zeroes."""
    total = {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "total_tokens": 0,
    }
    for call in calls:
        usage = call.get("usage_metadata")
        if not isinstance(usage, Mapping):
            continue
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            total[field] += _as_int(usage.get(field))
        output_details = usage.get("output_token_details")
        if isinstance(output_details, Mapping):
            total["reasoning_tokens"] += _as_int(output_details.get("reasoning"))
        input_details = usage.get("input_token_details")
        if isinstance(input_details, Mapping):
            total["cache_read_tokens"] += _as_int(input_details.get("cache_read"))
            total["cache_creation_tokens"] += _as_int(
                input_details.get("cache_creation")
            )
    return total


def build_model_config(
    config: InvestigationConfig, model_name: str
) -> InvestigationConfig:
    """Validate the fixed smoke model and reasoning effort rather than mutating config."""
    payload = config.model_dump(mode="json")
    payload.update({"model_name": model_name, "reasoning_effort": "medium"})
    return InvestigationConfig.model_validate(payload)


def _exposure(world: object, account_id: str, cutoff: object) -> int:
    """Use the same deterministic order exposure convention as the comparison run."""
    from mayajaal.synthetic import SyntheticWorld

    synthetic_world = cast(SyntheticWorld, world)
    order_ids = {
        str(event.order_id)
        for event in synthetic_world.events
        if str(event.account_id) == account_id
        and event.occurred_at <= cutoff
        and event.order_id is not None
    }
    amounts = [
        order.total_paise
        for order in synthetic_world.orders
        if str(order.id) in order_ids
    ]
    return int(amounts[-1]) if amounts else 100_000


def _load_openai_api_key(path: Path) -> None:
    """Load only the non-persisted runtime API key if the shell did not set it."""
    if os.environ.get("OPENAI_API_KEY") or not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        if name.strip() == "OPENAI_API_KEY":
            os.environ["OPENAI_API_KEY"] = value.strip().strip('"').strip("'")
            return


def _resolve_from_config(config_directory: Path, path: Path) -> Path:
    """Resolve command paths consistently with the existing backend scripts."""
    return path if path.is_absolute() else config_directory / path


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _metric_dump(metrics: ModelFacingContextMetrics | None) -> dict[str, object]:
    return {} if metrics is None else metrics.model_dump(mode="json")


def _mapping_value(value: object, key: str) -> object:
    return value.get(key) if isinstance(value, Mapping) else None


def _json_safe(value: object) -> object:
    """Leave provider metadata structured but make unusual scalar wrappers printable."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return (
        value
        if isinstance(value, str | int | float | bool | type(None))
        else str(value)
    )


def _as_int(value: object) -> int:
    return int(value) if isinstance(value, int | float) else 0


if __name__ == "__main__":
    raise SystemExit(main())
