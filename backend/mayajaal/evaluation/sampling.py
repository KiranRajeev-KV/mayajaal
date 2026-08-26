"""Leakage-safe, one-decision-per-account chronological sample construction."""

from collections import defaultdict
from datetime import datetime

from mayajaal.synthetic import SyntheticWorld

from .models import EvaluationConfig, EvaluationSample, EvaluationSplit, SplitManifest


def build_split_manifest(
    world: SyntheticWorld,
    config: EvaluationConfig,
    *,
    start_at: datetime | None = None,
    end_at: datetime | None = None,
) -> SplitManifest:
    """Create an account-disjoint chronological review benchmark.

    Every account is provisionally placed from the same observable anchor:
    ``Account.created_at``.  Hidden campaign membership is used only *after*
    that label-independent placement to purge a campaign whose members cross
    windows.  It never moves a sample, changes its decision time, or enters
    graph projection, feature extraction, or model inputs.

    A member's score is produced at the *end* of its assigned calendar window,
    so each vector can use only facts already known at that decision time.  The
    three shared decision times keep feature extraction practical at scale.
    """
    span = (
        (start_at, end_at)
        if start_at is not None and end_at is not None
        else world_profile_span(world)
    )
    if span[0] >= span[1]:
        raise ValueError("evaluation time window must increase")
    train_cutoff = span[0] + (span[1] - span[0]) * config.train_end_fraction
    validation_cutoff = span[0] + (span[1] - span[0]) * config.validation_end_fraction
    test_cutoff = span[1]

    provisional_assignments = {
        str(account.id): _window_for(
            account.created_at, train_cutoff, validation_cutoff, test_cutoff
        )
        for account in world.accounts
    }

    group_by_account: dict[str, str] = {}
    for event in world.events:
        account_id = str(event.account_id)
        labels = event.synthetic_labels
        if labels is not None and labels.is_coordinated_abuse:
            group_id = labels.coordination_cluster_id
            if group_id is None:
                raise ValueError("abusive synthetic events require a campaign group")
            existing = group_by_account.setdefault(account_id, group_id)
            if existing != group_id:
                raise ValueError("an account cannot belong to more than one campaign")
    provisional_splits_by_group: defaultdict[str, set[EvaluationSplit]] = defaultdict(
        set
    )
    for account_id, group_id in group_by_account.items():
        provisional_splits_by_group[group_id].add(
            provisional_assignments[account_id][0]
        )
    purged_campaign_group_ids = tuple(
        sorted(
            group_id
            for group_id, splits in provisional_splits_by_group.items()
            if len(splits) > 1
        )
    )
    purged_groups = frozenset(purged_campaign_group_ids)

    samples: list[EvaluationSample] = []
    for account in sorted(world.accounts, key=lambda item: str(item.id)):
        account_id = str(account.id)
        group_id = group_by_account.get(account_id)
        if group_id in purged_groups:
            continue
        split, decision_time = provisional_assignments[account_id]
        if account.created_at > decision_time:
            raise ValueError("an account cannot be scored before it was created")
        samples.append(
            EvaluationSample(
                sample_id=f"account-review:{account_id}:{split.value}",
                account_id=account_id,
                decision_time=decision_time,
                split=split,
                y_true=_label_known_at(world, account_id, decision_time),
                campaign_group_id=group_id,
            )
        )
    return SplitManifest(
        train_cutoff=train_cutoff,
        validation_cutoff=validation_cutoff,
        test_cutoff=test_cutoff,
        samples=tuple(
            sorted(samples, key=lambda item: (item.decision_time, item.sample_id))
        ),
        purged_campaign_group_ids=purged_campaign_group_ids,
    )


def world_profile_span(world: SyntheticWorld) -> tuple[datetime, datetime]:
    """Derive observable world bounds without requiring generator internals."""
    times = [event.occurred_at for event in world.events]
    if not times:
        raise ValueError("evaluation requires at least one event")
    return min(times), max(times)


def _window_for(
    anchor: datetime,
    train_cutoff: datetime,
    validation_cutoff: datetime,
    test_cutoff: datetime,
) -> tuple[EvaluationSplit, datetime]:
    if anchor <= train_cutoff:
        return EvaluationSplit.TRAIN, train_cutoff
    if anchor <= validation_cutoff:
        return EvaluationSplit.VALIDATION, validation_cutoff
    if anchor <= test_cutoff:
        return EvaluationSplit.TEST, test_cutoff
    raise ValueError("sample anchor occurs after test cutoff")


def _label_known_at(world: SyntheticWorld, account_id: str, cutoff: datetime) -> bool:
    """Read synthetic truth only after the cutoff-safe decision was defined."""
    return any(
        str(event.account_id) == account_id
        and event.occurred_at <= cutoff
        and event.synthetic_labels is not None
        and event.synthetic_labels.is_coordinated_abuse
        for event in world.events
    )
