"""Run deterministic multi-seed synthetic-realism diagnostics.

This is a benchmark validation utility, not a held-out model evaluation. It
inspects generation, deterministic resolution, temporal graph features, and,
when requested, the existing CatBoost/SHAP artifact path.
"""

import argparse
import json
from pathlib import Path

from mayajaal.baseline import (
    global_shap_importance,
    label_vectors,
    save_baseline,
    train_baseline,
)
from mayajaal.features import FeatureService
from mayajaal.graph import build_graph_projection
from mayajaal.resolution import resolve_all
from mayajaal.synthetic import (
    FeatureHealthDiagnostics,
    GenerationProfile,
    SyntheticWorld,
    cutoff_times,
    diagnose_feature_health,
    diagnose_world,
    feature_health_guardrail_failures,
    feature_health_review_warnings,
    generate_world,
    guardrail_failures,
)
from mayajaal.synthetic.config import load_generation_config


def parse_arguments() -> argparse.Namespace:
    """Parse config and benchmark-size options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run one full-size seed including CatBoost and SHAP diagnostics.",
    )
    parser.add_argument(
        "--full-only",
        action="store_true",
        help="Run only the full-size seed; implies --full.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to <output.directory>/validation.",
    )
    return parser.parse_args()


def profile_for_total_accounts(
    profile: GenerationProfile, requested_total: int, seed: int
) -> GenerationProfile:
    """Derive normal-population size so target-rate campaigns reach the total."""
    target_rate = profile.prevalence.resolved_target_rate()
    target_ordinary_accounts = (
        round(requested_total * (1.0 - target_rate))
        if target_rate is not None
        else requested_total
    )

    # Context counts may be population-scaled, so solve their small deterministic
    # fixed point instead of assuming legacy fixed counts.
    def ordinary_accounts(normal_account_count: int) -> int:
        return (
            normal_account_count
            + profile.model_copy(
                update={"normal_account_count": normal_account_count}
            ).resolved_shared_household_count()
            * profile.accounts_per_shared_household
            + profile.population.resolved_benign_network_group_count(
                normal_account_count
            )
            * profile.population.accounts_per_benign_network_group
        )

    candidates = range(max(target_ordinary_accounts + 1, 1))
    normal_account_count = min(
        candidates,
        key=lambda count: (
            abs(ordinary_accounts(count) - target_ordinary_accounts),
            count,
        ),
    )
    return profile.model_copy(
        update={
            "seed": seed,
            "normal_account_count": normal_account_count,
        }
    )


def _resolved_service(
    profile: GenerationProfile,
) -> tuple[SyntheticWorld, FeatureService]:
    world = generate_world(profile)
    resolution = resolve_all(
        accounts=world.accounts,
        addresses=world.addresses,
        ip_addresses=world.ip_addresses,
        payment_identities=world.payment_identities,
        devices=world.devices,
    )
    return world, FeatureService(build_graph_projection(world, resolution))


def validate_profile(
    profile: GenerationProfile,
    *,
    include_baseline: bool,
    baseline_output_directory: Path | None = None,
) -> dict[str, object]:
    """Run temporal diagnostics without using labels to construct features."""
    world, service = _resolved_service(profile)
    health_by_cutoff = {}
    warnings = list(guardrail_failures(diagnose_world(world), profile))
    review_warnings: list[str] = []
    for name, cutoff in cutoff_times(profile).items():
        account_ids = (
            str(account.id)
            for account in world.accounts
            if account.created_at <= cutoff
        )
        vectors = service.extract_many(account_ids, cutoff)
        health = diagnose_feature_health(
            vectors, service.schema, world, cutoff, profile.diagnostics
        )
        health_by_cutoff[name] = health
        review_warnings.extend(
            feature_health_review_warnings(
                health, profile.diagnostics, cutoff_name=name
            )
        )
        warnings.extend(
            feature_health_guardrail_failures(
                health,
                profile,
                cutoff_name=name,
                late=name == "late",
            )
        )
    report: dict[str, object] = {
        "seed": profile.seed,
        "world": diagnose_world(world).to_dict(),
        "feature_health": FeatureHealthDiagnostics(health_by_cutoff).to_dict(),
        "guardrail_failures": sorted(set(warnings)),
        "review_warnings": sorted(set(review_warnings)),
    }
    if include_baseline:
        late_cutoff = cutoff_times(profile)["late"]
        vectors = service.extract_many(
            (str(account.id) for account in world.accounts), late_cutoff
        )
        baseline = train_baseline(
            label_vectors(vectors, world, late_cutoff), service.schema
        )
        shap_vectors = tuple(
            sorted(vectors, key=lambda vector: vector.account_id)[
                : profile.validation.shap_sample_count
            ]
        )
        importance = global_shap_importance(baseline, shap_vectors)
        total_importance = sum(item.mean_absolute_shap for item in importance)
        top_share = (
            importance[0].mean_absolute_shap / total_importance
            if importance and total_importance
            else 0.0
        )
        report["shap"] = {
            "top_feature": importance[0].feature_name if importance else None,
            "top_feature_share": top_share,
            "global_importance": [
                {
                    "feature_name": item.feature_name,
                    "mean_absolute_shap": item.mean_absolute_shap,
                }
                for item in importance
            ],
        }
        if top_share > profile.diagnostics.shap_top_feature_share_warning:
            review_warnings.append(
                "SHAP top-feature share exceeds review threshold; this is not a generator guardrail."
            )
            report["review_warnings"] = sorted(set(review_warnings))
        if baseline_output_directory is not None:
            artifacts = save_baseline(baseline, shap_vectors, baseline_output_directory)
            report["baseline_artifacts"] = {
                "model": str(artifacts.model_path),
                "metadata": str(artifacts.metadata_path),
                "shap_summary": str(artifacts.shap_summary_path),
            }
    return report


def main() -> int:
    """Write a deterministic multi-seed report and optionally a full run report."""
    arguments = parse_arguments()
    config_path = arguments.config.resolve()
    config = load_generation_config(config_path)
    output_directory = (
        arguments.output_dir or Path(config.output.directory) / "validation"
    )
    if not output_directory.is_absolute():
        output_directory = config_path.parent / output_directory
    output_directory.mkdir(parents=True, exist_ok=True)

    profile = config.synthetic_world
    include_full = arguments.full or arguments.full_only
    small_runs = (
        []
        if arguments.full_only
        else [
            validate_profile(
                profile_for_total_accounts(
                    profile,
                    profile.validation.small_account_count,
                    profile.seed + index,
                ),
                include_baseline=False,
            )
            for index in range(profile.validation.multi_seed_count)
        ]
    )
    report: dict[str, object] = {
        "kind": "Mayajaal internal synthetic-realism benchmark validation",
        "difficulty": profile.difficulty.value,
        "prevalence": profile.prevalence.model_dump(mode="json"),
        "multi_seed_runs": small_runs,
    }
    if include_full:
        report["full_run"] = validate_profile(
            profile_for_total_accounts(
                profile, profile.validation.full_account_count, profile.seed
            ),
            include_baseline=True,
            baseline_output_directory=output_directory / "full-baseline",
        )
    output_path = output_directory / "validation.json"
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"validation: {output_path}")
    print(f"multi-seed runs: {len(small_runs)}")
    if include_full:
        full_run = report["full_run"]
        if isinstance(full_run, dict):
            world = full_run["world"]
            if isinstance(world, dict):
                print(
                    "full run: "
                    f"{world['account_count']} accounts, {world['event_count']} events"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
