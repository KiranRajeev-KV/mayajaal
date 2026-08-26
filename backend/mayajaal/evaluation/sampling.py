"""Leakage-safe fixed-decision-time chronological sample construction."""

from datetime import datetime

from mayajaal.synthetic import SyntheticWorld

from .models import (
    CampaignPurge,
    EvaluationConfig,
    EvaluationSample,
    EvaluationSplit,
    SplitManifest,
)


def build_split_manifest(
    world: SyntheticWorld,
    config: EvaluationConfig,
    *,
    start_at: datetime | None = None,
    end_at: datetime | None = None,
) -> SplitManifest:
    """Create fixed, rolling review samples for three chronological cutoffs.

    At every decision time, every account created by that point is eligible.
    The interval label is ``True`` only when that account's first labelled abuse
    becomes observable in the current interval.  Thus ordinary accounts recur,
    and an account can be negative before its later abuse becomes observable.

    Hidden campaign membership is evaluation-only.  A campaign with labelled
    events in multiple target intervals is purged entirely; an eligible
    campaign appears through its target interval, then is removed from later
    windows once known.  This prevents a known coordinated campaign from
    contributing positive examples to multiple partitions.  Membership and
    purge data never reach graph construction, feature extraction, or models.
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

    cutoffs = (
        (EvaluationSplit.TRAIN, train_cutoff, None),
        (EvaluationSplit.VALIDATION, validation_cutoff, train_cutoff),
        (EvaluationSplit.TEST, test_cutoff, validation_cutoff),
    )
    group_by_account: dict[str, str] = {}
    labelled_windows_by_group: dict[str, set[EvaluationSplit]] = {}
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
            window = _interval_for(event.occurred_at, cutoffs)
            labelled_windows_by_group.setdefault(group_id, set()).add(window)
    purged_campaign_groups = tuple(
        CampaignPurge(
            campaign_group_id=group_id,
            reason="labelled_abuse_spans_multiple_target_intervals",
        )
        for group_id, windows in sorted(labelled_windows_by_group.items())
        if len(windows) > 1
    )
    purged_campaign_group_ids = tuple(
        item.campaign_group_id for item in purged_campaign_groups
    )
    purged_groups = frozenset(purged_campaign_group_ids)
    target_window_by_group = {
        group_id: next(iter(windows))
        for group_id, windows in labelled_windows_by_group.items()
        if group_id not in purged_groups
    }

    samples: list[EvaluationSample] = []
    for account in sorted(world.accounts, key=lambda item: str(item.id)):
        account_id = str(account.id)
        group_id = group_by_account.get(account_id)
        if group_id in purged_groups:
            continue
        target_window = (
            target_window_by_group.get(group_id) if group_id is not None else None
        )
        for split, decision_time, interval_start in cutoffs:
            if account.created_at > decision_time:
                continue
            if target_window is not None and _split_index(split) > _split_index(
                target_window
            ):
                continue
            samples.append(
                EvaluationSample(
                    sample_id=f"account-review:{account_id}:{split.value}",
                    account_id=account_id,
                    decision_time=decision_time,
                    split=split,
                    y_true=_label_became_known_in_interval(
                        world, account_id, interval_start, decision_time
                    ),
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
        purged_campaign_groups=purged_campaign_groups,
    )


def world_profile_span(world: SyntheticWorld) -> tuple[datetime, datetime]:
    """Derive observable world bounds without requiring generator internals."""
    times = [event.occurred_at for event in world.events]
    if not times:
        raise ValueError("evaluation requires at least one event")
    return min(times), max(times)


def _interval_for(
    observed_at: datetime,
    cutoffs: tuple[tuple[EvaluationSplit, datetime, datetime | None], ...],
) -> EvaluationSplit:
    """Return the one target interval containing an immutable labelled event."""
    for split, cutoff, _ in cutoffs:
        if observed_at <= cutoff:
            return split
    raise ValueError("labelled abuse event occurs after the held-out test cutoff")


def _split_index(split: EvaluationSplit) -> int:
    """Return explicit chronological order without relying on enum string values."""
    return {
        EvaluationSplit.TRAIN: 0,
        EvaluationSplit.VALIDATION: 1,
        EvaluationSplit.TEST: 2,
    }[split]


def _label_became_known_in_interval(
    world: SyntheticWorld,
    account_id: str,
    interval_start: datetime | None,
    cutoff: datetime,
) -> bool:
    """Read synthetic truth only after an interval-safe decision was defined."""
    return any(
        str(event.account_id) == account_id
        and event.occurred_at <= cutoff
        and (interval_start is None or event.occurred_at > interval_start)
        and event.synthetic_labels is not None
        and event.synthetic_labels.is_coordinated_abuse
        for event in world.events
    )
