"""Internal plausibility diagnostics for a generated commerce world.

These diagnostics describe a simulator's own distributions and topology. They
do not claim similarity to private merchant data and intentionally introduce no
synthetic-data-quality dependency that requires a real reference dataset.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mayajaal.schemas import EventType

if TYPE_CHECKING:
    from .profile import GenerationProfile
    from .world import SyntheticWorld


@dataclass(frozen=True)
class ClassOverlap:
    """The support overlap of one evaluation-only account statistic."""

    positive_min: float
    positive_max: float
    negative_min: float
    negative_max: float
    overlaps: bool


@dataclass(frozen=True)
class SyntheticDiagnostics:
    """Stable internal summary of one deterministic generated world."""

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
        """Return a JSON-compatible deterministic report."""
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


def _overlap(positive: list[float], negative: list[float]) -> ClassOverlap:
    """Measure interval overlap without fitting a classifier."""
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


def _components(neighbours: dict[str, set[str]]) -> list[int]:
    remaining = set(neighbours)
    sizes: list[int] = []
    while remaining:
        start = remaining.pop()
        pending: deque[str] = deque([start])
        size = 0
        while pending:
            account_id = pending.popleft()
            size += 1
            discovered = neighbours[account_id] & remaining
            remaining.difference_update(discovered)
            pending.extend(discovered)
        sizes.append(size)
    return sizes


def _graph_statistics(identity_accounts: dict[str, set[str]]) -> dict[str, float]:
    """Compute lightweight account-projection topology statistics."""
    neighbours: defaultdict[str, set[str]] = defaultdict(set)
    pairs: set[tuple[str, str]] = set()
    for accounts in identity_accounts.values():
        for account_id in accounts:
            peers = accounts - {account_id}
            neighbours[account_id].update(peers)
            for peer in peers:
                if account_id < peer:
                    pairs.add((account_id, peer))
    if not neighbours:
        return {
            "account_projection_edge_count": 0.0,
            "component_count": 0.0,
            "largest_component_account_count": 0.0,
            "mean_account_projection_degree": 0.0,
            "mean_local_clustering": 0.0,
            "degree_assortativity": 0.0,
        }
    degrees = {account_id: len(peers) for account_id, peers in neighbours.items()}
    clustering: list[float] = []
    for _account_id, peers in neighbours.items():
        if len(peers) < 2:
            clustering.append(0.0)
            continue
        possible = len(peers) * (len(peers) - 1) / 2
        actual = sum(
            1
            for peer in peers
            for other in peers
            if peer < other and other in neighbours[peer]
        )
        clustering.append(actual / possible)
    source_degrees = [degrees[left] for left, _ in pairs]
    target_degrees = [degrees[right] for _, right in pairs]
    assortativity = 0.0
    if len(pairs) > 1 and len(set(source_degrees)) > 1 and len(set(target_degrees)) > 1:
        assortativity = float(np.corrcoef(source_degrees, target_degrees)[0, 1])
    components = _components(dict(neighbours))
    return {
        "account_projection_edge_count": float(len(pairs)),
        "component_count": float(len(components)),
        "largest_component_account_count": float(max(components)),
        "mean_account_projection_degree": _mean(list(map(float, degrees.values()))),
        "mean_local_clustering": _mean(clustering),
        "degree_assortativity": assortativity,
    }


def diagnose_world(world: SyntheticWorld) -> SyntheticDiagnostics:
    """Summarize distributions, timing, class overlap, and graph topology.

    Synthetic labels are read only for the evaluation-only class-overlap
    section. The generated entity and event values drive every other metric.
    """
    account_ids = tuple(str(account.id) for account in world.accounts)
    order_counts: defaultdict[str, int] = defaultdict(int)
    refund_counts: defaultdict[str, int] = defaultdict(int)
    promo_counts: defaultdict[str, int] = defaultdict(int)
    identity_accounts: defaultdict[str, set[str]] = defaultdict(set)
    labelled_accounts: set[str] = set()
    labelled_events = 0
    event_hours: defaultdict[int, int] = defaultdict(int)
    event_days: defaultdict[datetime, int] = defaultdict(int)
    order_account_by_id = {
        str(order.id): str(order.account_id) for order in world.orders
    }

    for event in world.events:
        account_id = str(event.account_id)
        event_hours[event.occurred_at.hour] += 1
        event_days[
            event.occurred_at.replace(hour=0, minute=0, second=0, microsecond=0)
        ] += 1
        if event.synthetic_labels is not None:
            labelled_accounts.add(account_id)
            labelled_events += 1
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
        elif event.event_type is EventType.REFUND_REQUESTED:
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
    overlaps: dict[str, ClassOverlap] = {}
    for name, feature_values in values.items():
        positive = [
            feature_values[index]
            for index, account_id in enumerate(account_ids)
            if account_id in labelled_accounts
        ]
        negative = [
            feature_values[index]
            for index, account_id in enumerate(account_ids)
            if account_id not in labelled_accounts
        ]
        overlaps[name] = _overlap(positive, negative)
    graph = _graph_statistics(identity_accounts)
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
            "mean_refunds_per_account": _mean(values["refund_requested_count"]),
            "mean_promo_redemptions_per_account": _mean(
                values["promotion_redemption_count"]
            ),
            "mean_identity_count_per_account": _mean(values["identity_count"]),
            "mean_identity_reuse_degree": _mean(identity_degrees),
            "max_identity_reuse_degree": float(max(identity_degrees, default=0.0)),
        },
        temporal={
            "active_day_count": float(len(event_days)),
            "busiest_day_event_count": float(max(event_days.values(), default=0)),
            "busiest_hour_event_count": float(max(event_hours.values(), default=0)),
            "active_hour_count": float(len(event_hours)),
        },
        graph=graph,
        class_overlap=overlaps,
    )


def guardrail_failures(
    diagnostics: SyntheticDiagnostics, profile: GenerationProfile
) -> tuple[str, ...]:
    """Return internal benchmark warnings without claiming external fidelity."""
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
    return tuple(failures)


def write_diagnostics(diagnostics: SyntheticDiagnostics, path: Path) -> Path:
    """Write a stable JSON report beside generated data."""
    path.write_text(
        json.dumps(diagnostics.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
