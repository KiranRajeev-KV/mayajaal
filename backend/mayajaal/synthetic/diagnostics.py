"""Deterministic internal plausibility diagnostics for synthetic worlds.

The reports measure simulator diversity and benchmark shortcuts. They do not
claim calibration to private merchant data. Label-aware sections are explicitly
evaluation-only and never feed generation or feature extraction.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mayajaal.schemas import EventType

if TYPE_CHECKING:
    from mayajaal.features import FeatureSchema, FeatureVector

    from .profile import DiagnosticProfile, GenerationProfile
    from .world import SyntheticWorld


IDENTITY_TYPES = ("device", "ip", "payment", "address")


@dataclass(frozen=True)
class ClassOverlap:
    """Evaluation-only support overlap for one account or feature statistic."""

    positive_min: float
    positive_max: float
    negative_min: float
    negative_max: float
    overlaps: bool


@dataclass(frozen=True)
class NumericFeatureHealth:
    """Cutoff-safe health of one numeric feature column."""

    unique_count: int
    variance: float
    zero_fraction: float
    median: float
    p95: float
    class_histogram_overlap: float | None
    class_auc: float | None


@dataclass(frozen=True)
class CategoricalFeatureHealth:
    """Cutoff-safe health of one categorical feature column."""

    cardinality: int
    missing_fraction: float
    dominant_fraction: float
    best_category_balanced_accuracy: float | None


@dataclass(frozen=True)
class FeatureHealthAtCutoff:
    """Feature-health report at one named temporal reconstruction point."""

    cutoff: datetime
    sample_count: int
    labelled_sample_count: int
    unlabelled_sample_count: int
    class_support_warnings: tuple[str, ...]
    numeric: dict[str, NumericFeatureHealth]
    categorical: dict[str, CategoricalFeatureHealth]
    redundant_numeric_pairs: tuple[tuple[str, str, float], ...]
    inactive_expected_numeric_features: tuple[str, ...]
    intentionally_sparse_numeric_features: tuple[str, ...]

    @property
    def has_sufficient_class_support(self) -> bool:
        """Whether label-separability diagnostics are statistically enforceable."""
        return not self.class_support_warnings

    def to_dict(self) -> dict[str, object]:
        return {
            "cutoff": self.cutoff.isoformat(),
            "sample_count": self.sample_count,
            "labelled_sample_count": self.labelled_sample_count,
            "unlabelled_sample_count": self.unlabelled_sample_count,
            "class_support_warnings": list(self.class_support_warnings),
            "numeric": {name: asdict(value) for name, value in self.numeric.items()},
            "categorical": {
                name: asdict(value) for name, value in self.categorical.items()
            },
            "redundant_numeric_pairs": [
                {"left": left, "right": right, "absolute_correlation": value}
                for left, right, value in self.redundant_numeric_pairs
            ],
            "inactive_expected_numeric_features": list(
                self.inactive_expected_numeric_features
            ),
            "intentionally_sparse_numeric_features": list(
                self.intentionally_sparse_numeric_features
            ),
        }


@dataclass(frozen=True)
class FeatureHealthDiagnostics:
    """Early, middle, and late feature-health snapshots."""

    by_cutoff: dict[str, FeatureHealthAtCutoff]

    def to_dict(self) -> dict[str, object]:
        return {name: value.to_dict() for name, value in self.by_cutoff.items()}


@dataclass(frozen=True)
class SyntheticDiagnostics:
    """Stable internal world summary, independent of a model."""

    account_count: int
    event_count: int
    order_count: int
    labelled_account_count: int
    labelled_event_count: int
    numeric_feature_variation_count: int
    perfect_single_feature_separators: tuple[str, ...]
    distributions: dict[str, float]
    temporal: dict[str, float]
    graph: dict[str, float]
    class_overlap: dict[str, ClassOverlap]

    def to_dict(self) -> dict[str, object]:
        return {
            "account_count": self.account_count,
            "event_count": self.event_count,
            "order_count": self.order_count,
            "labelled_account_count": self.labelled_account_count,
            "labelled_event_count": self.labelled_event_count,
            "numeric_feature_variation_count": self.numeric_feature_variation_count,
            "perfect_single_feature_separators": list(
                self.perfect_single_feature_separators
            ),
            "distributions": self.distributions,
            "temporal": self.temporal,
            "graph": self.graph,
            "class_overlap": {
                name: asdict(value) for name, value in self.class_overlap.items()
            },
        }


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def _p95(values: list[float]) -> float:
    return float(np.percentile(values, 95)) if values else 0.0


def _gini(values: list[float]) -> float:
    if not values or not any(values):
        return 0.0
    ordered = np.sort(np.asarray(values, dtype=float))
    count = len(ordered)
    return float(
        (2.0 * np.dot(np.arange(1, count + 1), ordered))
        / (count * float(np.sum(ordered)))
        - (count + 1.0) / count
    )


def _overlap(positive: list[float], negative: list[float]) -> ClassOverlap:
    if not positive or not negative:
        return ClassOverlap(0.0, 0.0, 0.0, 0.0, False)
    positive_min, positive_max = min(positive), max(positive)
    negative_min, negative_max = min(negative), max(negative)
    return ClassOverlap(
        positive_min=float(positive_min),
        positive_max=float(positive_max),
        negative_min=float(negative_min),
        negative_max=float(negative_max),
        overlaps=not (positive_min > negative_max or negative_min > positive_max),
    )


def _component_nodes(neighbours: dict[str, set[str]]) -> list[set[str]]:
    remaining = set(neighbours)
    components: list[set[str]] = []
    while remaining:
        start = remaining.pop()
        pending: deque[str] = deque([start])
        component: set[str] = set()
        while pending:
            account_id = pending.popleft()
            component.add(account_id)
            discovered = neighbours[account_id] & remaining
            remaining.difference_update(discovered)
            pending.extend(discovered)
        components.append(component)
    return components


def _components(neighbours: dict[str, set[str]]) -> list[int]:
    return [len(component) for component in _component_nodes(neighbours)]


def _projection_statistics(
    neighbours: dict[str, set[str]], prefix: str
) -> dict[str, float]:
    """Return clearly named undirected projection metrics for one node scope."""
    pairs = {
        tuple(sorted((account_id, peer)))
        for account_id, peers in neighbours.items()
        for peer in peers
        if account_id != peer
    }
    if not neighbours:
        return {
            f"{prefix}_account_count": 0.0,
            f"{prefix}_edge_count": 0.0,
            f"{prefix}_component_count": 0.0,
            f"{prefix}_largest_component_account_count": 0.0,
            f"{prefix}_mean_degree": 0.0,
            f"{prefix}_mean_local_clustering": 0.0,
            f"{prefix}_degree_assortativity": 0.0,
        }
    degrees = {account_id: len(peers) for account_id, peers in neighbours.items()}
    clustering: list[float] = []
    for peers in neighbours.values():
        if len(peers) < 2:
            clustering.append(0.0)
            continue
        possible = len(peers) * (len(peers) - 1) / 2
        actual = sum(
            1
            for left, right in combinations(sorted(peers), 2)
            if right in neighbours[left]
        )
        clustering.append(actual / possible)
    endpoints = [
        degree for left, right in pairs for degree in (degrees[left], degrees[right])
    ]
    opposite = [
        degree for left, right in pairs for degree in (degrees[right], degrees[left])
    ]
    assortativity = 0.0
    if len(endpoints) > 1 and len(set(endpoints)) > 1:
        assortativity = float(np.corrcoef(endpoints, opposite)[0, 1])
    components = _components(neighbours)
    return {
        f"{prefix}_account_count": float(len(neighbours)),
        f"{prefix}_edge_count": float(len(pairs)),
        f"{prefix}_component_count": float(len(components)),
        f"{prefix}_largest_component_account_count": float(max(components)),
        f"{prefix}_mean_degree": _mean(list(map(float, degrees.values()))),
        f"{prefix}_mean_local_clustering": _mean(clustering),
        f"{prefix}_degree_assortativity": assortativity,
    }


def graph_statistics(
    account_ids: tuple[str, ...],
    identity_accounts: dict[str, set[str]],
    labelled_accounts: set[str],
) -> dict[str, float]:
    """Measure full, sharing-only, and typed bipartite graph structure.

    Candidate pairs come only from a shared identity; no global all-pairs scan is
    performed.
    """
    full_neighbours: dict[str, set[str]] = {
        account_id: set() for account_id in account_ids
    }
    pair_types: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    pair_identity_count: defaultdict[tuple[str, str], int] = defaultdict(int)
    typed_degrees: defaultdict[str, list[float]] = defaultdict(list)
    typed_reused: defaultdict[str, int] = defaultdict(int)
    for identity_key, accounts in identity_accounts.items():
        identity_type, _ = identity_key.split(":", maxsplit=1)
        typed_degrees[identity_type].append(float(len(accounts)))
        if len(accounts) > 1:
            typed_reused[identity_type] += 1
        for left, right in combinations(sorted(accounts), 2):
            full_neighbours[left].add(right)
            full_neighbours[right].add(left)
            pair_types[(left, right)].add(identity_type)
            pair_identity_count[(left, right)] += 1
    sharing_neighbours = {
        account_id: peers for account_id, peers in full_neighbours.items() if peers
    }
    result = _projection_statistics(full_neighbours, "full_account_projection")
    result.update(
        _projection_statistics(sharing_neighbours, "identity_sharing_subgraph")
    )
    result["full_account_projection_isolated_account_count"] = float(
        sum(not peers for peers in full_neighbours.values())
    )
    result["identity_sharing_subgraph_largest_component_fraction"] = (
        result["identity_sharing_subgraph_largest_component_account_count"]
        / result["identity_sharing_subgraph_account_count"]
        if result["identity_sharing_subgraph_account_count"]
        else 0.0
    )
    result["typed_multi_identity_pair_count"] = float(
        sum(len(types) >= 2 for types in pair_types.values())
    )
    result["typed_two_or_more_identity_pair_fraction"] = (
        result["typed_multi_identity_pair_count"] / float(len(pair_types))
        if pair_types
        else 0.0
    )
    # A butterfly / bipartite four-cycle is K2,2: choose two accounts and two
    # identity nodes that both accounts touch, regardless of identity type.
    result["account_identity_four_cycle_count"] = float(
        sum(
            count * (count - 1) / 2
            for count in pair_identity_count.values()
            if count > 1
        )
    )
    sharing_components = _component_nodes(sharing_neighbours)
    result["identity_sharing_subgraph_largest_component_population_fraction"] = (
        result["identity_sharing_subgraph_largest_component_account_count"]
        / len(account_ids)
        if account_ids
        else 0.0
    )
    result["identity_sharing_subgraph_max_labelled_account_fraction"] = (
        max(
            (len(component & labelled_accounts) for component in sharing_components),
            default=0,
        )
        / len(labelled_accounts)
        if labelled_accounts
        else 0.0
    )
    for identity_type in IDENTITY_TYPES:
        degrees = typed_degrees[identity_type]
        result[f"{identity_type}_identity_count"] = float(len(degrees))
        result[f"{identity_type}_identity_mean_account_degree"] = _mean(degrees)
        result[f"{identity_type}_identity_p95_account_degree"] = _p95(degrees)
        result[f"{identity_type}_identity_degree_gini"] = _gini(degrees)
        result[f"{identity_type}_reused_identity_fraction"] = (
            float(typed_reused[identity_type]) / float(len(degrees)) if degrees else 0.0
        )
    return result


def _typed_peer_jaccard(
    account_ids: tuple[str, ...], identity_accounts: dict[str, set[str]]
) -> dict[str, float]:
    peers: dict[str, dict[str, set[str]]] = {
        account_id: {identity_type: set() for identity_type in IDENTITY_TYPES}
        for account_id in account_ids
    }
    for identity_key, accounts in identity_accounts.items():
        identity_type, _ = identity_key.split(":", maxsplit=1)
        for account_id in accounts:
            peers[account_id][identity_type].update(accounts - {account_id})
    result: dict[str, float] = {}
    for left_type, right_type in combinations(IDENTITY_TYPES, 2):
        values: list[float] = []
        for account_id in account_ids:
            left = peers[account_id][left_type]
            right = peers[account_id][right_type]
            if left or right:
                values.append(float(len(left & right)) / float(len(left | right)))
        result[f"peer_jaccard_{left_type}_{right_type}"] = _mean(values)
    return result


def _labelled_accounts(world: SyntheticWorld) -> set[str]:
    return {
        str(event.account_id)
        for event in world.events
        if event.synthetic_labels is not None
        and event.synthetic_labels.is_coordinated_abuse
    }


def diagnose_world(world: SyntheticWorld) -> SyntheticDiagnostics:
    """Summarize non-model world diversity and evaluation-only label overlap."""
    account_ids = tuple(sorted(str(account.id) for account in world.accounts))
    order_counts: defaultdict[str, int] = defaultdict(int)
    refund_counts: defaultdict[str, int] = defaultdict(int)
    promo_counts: defaultdict[str, int] = defaultdict(int)
    identity_accounts: defaultdict[str, set[str]] = defaultdict(set)
    event_hours: defaultdict[int, int] = defaultdict(int)
    event_days: defaultdict[datetime, int] = defaultdict(int)
    account_times: defaultdict[str, list[datetime]] = defaultdict(list)
    order_account_by_id = {
        str(order.id): str(order.account_id) for order in world.orders
    }
    order_values = [float(order.total_paise) for order in world.orders]
    labelled_accounts = _labelled_accounts(world)
    labelled_events = sum(event.synthetic_labels is not None for event in world.events)

    for event in world.events:
        account_id = str(event.account_id)
        account_times[account_id].append(event.occurred_at)
        event_hours[event.occurred_at.hour] += 1
        event_days[
            event.occurred_at.replace(hour=0, minute=0, second=0, microsecond=0)
        ] += 1
        if event.event_type is EventType.DEVICE_SEEN and event.device_id is not None:
            identity_accounts[f"device:{event.device_id}"].add(account_id)
        elif event.event_type is EventType.IP_SEEN and event.ip_address_id is not None:
            identity_accounts[f"ip:{event.ip_address_id}"].add(account_id)
        elif (
            event.event_type is EventType.PAYMENT_ATTACHED
            and event.payment_identity_id is not None
        ):
            identity_accounts[f"payment:{event.payment_identity_id}"].add(account_id)
        elif event.event_type is EventType.ORDER_PLACED:
            order_counts[account_id] += 1
            if event.address_id is not None:
                identity_accounts[f"address:{event.address_id}"].add(account_id)
        elif event.event_type is EventType.PROMOTION_REDEEMED:
            promo_counts[account_id] += 1
        elif (
            event.event_type is EventType.REFUND_REQUESTED
            and event.order_id is not None
        ):
            refund_counts[order_account_by_id[str(event.order_id)]] += 1

    peer_counts: defaultdict[str, set[str]] = defaultdict(set)
    identity_counts: defaultdict[str, int] = defaultdict(int)
    for accounts in identity_accounts.values():
        for account_id in accounts:
            identity_counts[account_id] += 1
            peer_counts[account_id].update(accounts - {account_id})
    values: dict[str, list[float]] = {
        "order_count": [float(order_counts[account_id]) for account_id in account_ids],
        "refund_requested_count": [
            float(refund_counts[account_id]) for account_id in account_ids
        ],
        "promotion_redemption_count": [
            float(promo_counts[account_id]) for account_id in account_ids
        ],
        "identity_count": [
            float(identity_counts[account_id]) for account_id in account_ids
        ],
        "identity_peer_count": [
            float(len(peer_counts[account_id])) for account_id in account_ids
        ],
    }
    overlaps = {
        name: _overlap(
            [
                feature_values[index]
                for index, account_id in enumerate(account_ids)
                if account_id in labelled_accounts
            ],
            [
                feature_values[index]
                for index, account_id in enumerate(account_ids)
                if account_id not in labelled_accounts
            ],
        )
        for name, feature_values in values.items()
    }
    event_gaps = [
        (right - left).total_seconds() / 3600.0
        for times in account_times.values()
        for left, right in zip(sorted(times), sorted(times)[1:], strict=False)
    ]
    daily_counts = list(map(float, event_days.values()))
    fano = float(np.var(daily_counts) / np.mean(daily_counts)) if daily_counts else 0.0
    gap_mean = _mean(event_gaps)
    gap_std = float(np.std(event_gaps)) if event_gaps else 0.0
    burstiness = (
        (gap_std - gap_mean) / (gap_std + gap_mean) if gap_std + gap_mean else 0.0
    )
    graph = graph_statistics(account_ids, identity_accounts, labelled_accounts)
    graph.update(_typed_peer_jaccard(account_ids, identity_accounts))
    identity_degrees = [float(len(accounts)) for accounts in identity_accounts.values()]
    return SyntheticDiagnostics(
        account_count=len(account_ids),
        event_count=len(world.events),
        order_count=len(world.orders),
        labelled_account_count=len(labelled_accounts),
        labelled_event_count=labelled_events,
        numeric_feature_variation_count=sum(
            len(set(feature_values)) > 1 for feature_values in values.values()
        ),
        perfect_single_feature_separators=tuple(
            sorted(name for name, overlap in overlaps.items() if not overlap.overlaps)
        ),
        distributions={
            "mean_orders_per_account": _mean(values["order_count"]),
            "median_orders_per_account": _median(values["order_count"]),
            "mean_refunds_per_account": _mean(values["refund_requested_count"]),
            "mean_promo_redemptions_per_account": _mean(
                values["promotion_redemption_count"]
            ),
            "mean_identity_count_per_account": _mean(values["identity_count"]),
            "mean_identity_peer_count": _mean(values["identity_peer_count"]),
            "mean_identity_reuse_degree": _mean(identity_degrees),
            "max_identity_reuse_degree": float(max(identity_degrees, default=0.0)),
            "order_value_median_paise": _median(order_values),
            "order_value_p95_paise": _p95(order_values),
            "order_value_gini": _gini(order_values),
            "device_entity_count": float(len(world.devices)),
            "ip_address_entity_count": float(len(world.ip_addresses)),
            "payment_identity_entity_count": float(len(world.payment_identities)),
            "address_entity_count": float(len(world.addresses)),
        },
        temporal={
            "active_day_count": float(len(event_days)),
            "busiest_day_event_count": float(max(event_days.values(), default=0)),
            "busiest_hour_event_count": float(max(event_hours.values(), default=0)),
            "active_hour_count": float(len(event_hours)),
            "daily_event_fano_factor": fano,
            "account_event_gap_median_hours": _median(event_gaps),
            "account_event_gap_p95_hours": _p95(event_gaps),
            "account_event_gap_burstiness": burstiness,
        },
        graph=graph,
        class_overlap=overlaps,
    )


def _labels_at_cutoff(world: SyntheticWorld, cutoff: datetime) -> dict[str, bool]:
    labels = {
        str(account.id): False
        for account in world.accounts
        if account.created_at <= cutoff
    }
    for event in world.events:
        if (
            event.occurred_at <= cutoff
            and str(event.account_id) in labels
            and event.synthetic_labels is not None
            and event.synthetic_labels.is_coordinated_abuse
        ):
            labels[str(event.account_id)] = True
    return labels


def _histogram_overlap(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    low, high = min(*positive, *negative), max(*positive, *negative)
    if low == high:
        return 1.0
    positive_hist, edges = np.histogram(positive, bins=10, range=(low, high))
    negative_hist, _ = np.histogram(negative, bins=edges)
    positive_probability = positive_hist / positive_hist.sum()
    negative_probability = negative_hist / negative_hist.sum()
    return float(np.minimum(positive_probability, negative_probability).sum())


def _auc(values: list[float], labels: list[bool]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(values, labels, strict=True), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        positive_rank_sum += average_rank * sum(
            label for _, label in ordered[index:end]
        )
        index = end
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )
    return float(max(auc, 1.0 - auc))


def _best_category_balanced_accuracy(
    values: list[str], labels: list[bool]
) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    best = 0.5
    for category in sorted(set(values)):
        true_positive = sum(
            value == category and label
            for value, label in zip(values, labels, strict=True)
        )
        false_positive = sum(
            value == category and not label
            for value, label in zip(values, labels, strict=True)
        )
        true_negative = negatives - false_positive
        best = max(best, (true_positive / positives + true_negative / negatives) / 2.0)
    return float(best)


def diagnose_feature_health(
    vectors: tuple[FeatureVector, ...],
    schema: FeatureSchema,
    world: SyntheticWorld,
    cutoff: datetime,
    profile: DiagnosticProfile,
) -> FeatureHealthAtCutoff:
    """Inspect the existing feature schema at one leakage-safe cutoff."""
    ordered = tuple(sorted(vectors, key=lambda vector: vector.account_id))
    labels_by_account = _labels_at_cutoff(world, cutoff)
    labels = [labels_by_account[vector.account_id] for vector in ordered]
    numeric: dict[str, NumericFeatureHealth] = {}
    categorical: dict[str, CategoricalFeatureHealth] = {}
    numeric_values: dict[str, list[float]] = {}
    for name in schema.numeric_names:
        values = [float(vector.values[name]) for vector in ordered]
        numeric_values[name] = values
        positive = [value for value, label in zip(values, labels, strict=True) if label]
        negative = [
            value for value, label in zip(values, labels, strict=True) if not label
        ]
        numeric[name] = NumericFeatureHealth(
            unique_count=len(set(values)),
            variance=float(np.var(values)) if values else 0.0,
            zero_fraction=(
                sum(value == 0.0 for value in values) / len(values) if values else 0.0
            ),
            median=_median(values),
            p95=_p95(values),
            class_histogram_overlap=_histogram_overlap(positive, negative),
            class_auc=_auc(values, labels),
        )
    for name in schema.categorical_names:
        values = [str(vector.values[name]) for vector in ordered]
        counts = {value: values.count(value) for value in sorted(set(values))}
        categorical[name] = CategoricalFeatureHealth(
            cardinality=len(counts),
            missing_fraction=(
                values.count("__missing__") / len(values) if values else 0.0
            ),
            dominant_fraction=(max(counts.values()) / len(values) if values else 0.0),
            best_category_balanced_accuracy=_best_category_balanced_accuracy(
                values, labels
            ),
        )
    redundant: list[tuple[str, str, float]] = []
    for left, right in combinations(schema.numeric_names, 2):
        left_values, right_values = numeric_values[left], numeric_values[right]
        if len(set(left_values)) < 2 or len(set(right_values)) < 2:
            continue
        correlation = float(np.corrcoef(left_values, right_values)[0, 1])
        if abs(correlation) >= 0.98:
            redundant.append((left, right, abs(correlation)))
    inactive = tuple(
        name
        for name in profile.expected_active_numeric_features
        if name not in numeric or numeric[name].unique_count < 2
    )
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    support_warnings: list[str] = []
    if positive_count < profile.min_cutoff_positive_samples:
        support_warnings.append(
            "insufficient positive samples for class metrics: "
            f"{positive_count} < {profile.min_cutoff_positive_samples}"
        )
    if negative_count < profile.min_cutoff_negative_samples:
        support_warnings.append(
            "insufficient negative samples for class metrics: "
            f"{negative_count} < {profile.min_cutoff_negative_samples}"
        )
    return FeatureHealthAtCutoff(
        cutoff=cutoff,
        sample_count=len(ordered),
        labelled_sample_count=positive_count,
        unlabelled_sample_count=negative_count,
        class_support_warnings=tuple(support_warnings),
        numeric=numeric,
        categorical=categorical,
        redundant_numeric_pairs=tuple(sorted(redundant)),
        inactive_expected_numeric_features=inactive,
        intentionally_sparse_numeric_features=profile.intentionally_sparse_numeric_features,
    )


def cutoff_times(profile: GenerationProfile) -> dict[str, datetime]:
    """Return configured early/middle/late reconstruction times."""
    names = ("early", "middle", "late")
    span = profile.end_at - profile.start_at
    return {
        name: profile.start_at + span * fraction
        for name, fraction in zip(
            names, profile.diagnostics.cutoff_fractions, strict=True
        )
    }


def guardrail_failures(
    diagnostics: SyntheticDiagnostics, profile: GenerationProfile
) -> tuple[str, ...]:
    """Return model-independent benchmark guardrail warnings."""
    failures: list[str] = []
    if (
        diagnostics.numeric_feature_variation_count
        < profile.diagnostics.min_variable_numeric_features
    ):
        failures.append("insufficient variable diagnostic features")
    if (
        len(diagnostics.perfect_single_feature_separators)
        > profile.diagnostics.max_single_feature_perfect_separators
    ):
        failures.append("single diagnostic feature perfectly separates labels")
    target = profile.prevalence.resolved_target_rate()
    if target is not None and diagnostics.account_count:
        observed = diagnostics.labelled_account_count / diagnostics.account_count
        if abs(observed - target) > profile.prevalence.target_tolerance:
            failures.append(
                "labelled account prevalence is outside configured tolerance"
            )
    graph = diagnostics.graph
    if (
        graph["identity_sharing_subgraph_largest_component_population_fraction"]
        > profile.diagnostics.max_identity_sharing_component_fraction
    ):
        failures.append("identity-sharing component is disproportionately large")
    if (
        graph["identity_sharing_subgraph_max_labelled_account_fraction"]
        > profile.diagnostics.max_labelled_accounts_in_single_component_fraction
    ):
        failures.append("too many labelled accounts occupy one identity component")
    return tuple(failures)


def _separability_violations(
    health: FeatureHealthAtCutoff, profile: DiagnosticProfile
) -> tuple[str, ...]:
    failures: list[str] = []
    for name, details in health.numeric.items():
        if (
            details.class_auc is not None
            and details.class_auc > profile.max_single_feature_auc
        ):
            failures.append(
                f"numeric feature exceeds single-feature AUC guardrail: {name}"
            )
        if (
            details.class_histogram_overlap is not None
            and details.class_histogram_overlap < profile.min_class_histogram_overlap
        ):
            failures.append(f"numeric feature has low class histogram overlap: {name}")
    for name, details in health.categorical.items():
        if (
            details.best_category_balanced_accuracy is not None
            and details.best_category_balanced_accuracy > profile.max_single_feature_auc
        ):
            failures.append(f"categorical feature exceeds separation guardrail: {name}")
    return tuple(sorted(set(failures)))


def feature_health_guardrail_failures(
    health: FeatureHealthAtCutoff,
    profile: GenerationProfile,
    *,
    cutoff_name: str,
    late: bool,
) -> tuple[str, ...]:
    """Return enforceable, cutoff-named feature-health guardrail failures.

    Class support governs label-aware separability checks only.  Feature
    variation still remains a hard late-cutoff invariant because it does not
    depend on labels.
    """
    failures: list[str] = []
    if late and health.inactive_expected_numeric_features:
        failures.append("expected-active numeric feature is constant at late cutoff")
    if health.has_sufficient_class_support:
        failures.extend(_separability_violations(health, profile.diagnostics))
    return tuple(sorted({f"{cutoff_name}: {failure}" for failure in failures}))


def feature_health_review_warnings(
    health: FeatureHealthAtCutoff,
    profile: DiagnosticProfile,
    *,
    cutoff_name: str,
) -> tuple[str, ...]:
    """Return cutoff-named support and downgraded separability review warnings."""
    warnings = [
        f"{cutoff_name}: {warning}" for warning in health.class_support_warnings
    ]
    if not health.has_sufficient_class_support:
        warnings.extend(
            f"{cutoff_name}: separability guardrail reviewed only due to insufficient "
            f"class support: {violation}"
            for violation in _separability_violations(health, profile)
        )
    return tuple(sorted(set(warnings)))


def write_diagnostics(diagnostics: SyntheticDiagnostics, path: Path) -> Path:
    """Write a stable JSON world report beside generated data."""
    path.write_text(
        json.dumps(diagnostics.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
