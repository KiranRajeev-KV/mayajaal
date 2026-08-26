"""Deterministic profile sizing shared by benchmark-oriented entry points."""

from .profile import GenerationProfile


def profile_for_total_accounts(
    profile: GenerationProfile, requested_total: int, seed: int | None = None
) -> GenerationProfile:
    """Derive normal-population size so a target-rate world reaches a total."""
    target_rate = profile.prevalence.resolved_target_rate()
    target_ordinary_accounts = (
        round(requested_total * (1.0 - target_rate))
        if target_rate is not None
        else requested_total
    )

    def ordinary_accounts(normal_account_count: int) -> int:
        candidate = profile.model_copy(
            update={"normal_account_count": normal_account_count}
        )
        return (
            normal_account_count
            + candidate.resolved_shared_household_count()
            * candidate.accounts_per_shared_household
            + candidate.population.resolved_benign_network_group_count(
                normal_account_count
            )
            * candidate.population.accounts_per_benign_network_group
        )

    normal_account_count = min(
        range(max(target_ordinary_accounts + 1, 1)),
        key=lambda count: (
            abs(ordinary_accounts(count) - target_ordinary_accounts),
            count,
        ),
    )
    update: dict[str, int] = {"normal_account_count": normal_account_count}
    if seed is not None:
        update["seed"] = seed
    return profile.model_copy(update=update)
