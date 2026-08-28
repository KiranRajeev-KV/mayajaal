"""Bounded, deterministic, read-only investigation evidence retrieval."""

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from math import isclose

from pydantic import JsonValue

from mayajaal.baseline import explain_prediction
from mayajaal.evaluation.provenance import FrozenFullEvaluation
from mayajaal.features import FeatureService, FeatureVector
from mayajaal.graph import (
    GraphNodeType,
    GraphProjection,
    GraphRelationship,
    GraphRelationshipType,
)
from mayajaal.schemas import Event, EventType
from mayajaal.scoring import ScoreObservation
from mayajaal.scoring.service import verify_score_from_feature_vector

from .models import (
    EvidenceItem,
    EvidenceSource,
    EvidenceType,
    InvestigationConfig,
    InvestigationRequest,
)
from .provenance import evidence_id


@dataclass(frozen=True)
class _IdentityLink:
    """One cutoff-filtered logical account-to-identity connection."""

    account_id: str
    identity_id: str
    identity_type: GraphNodeType
    first_seen: datetime
    last_seen: datetime
    event_ids: tuple[str, ...]


@dataclass(frozen=True)
class _RankedRelatedAccount:
    """One cutoff-safe peer ranked only by observed identity overlap."""

    account_id: str
    shared_identity_type_count: int
    most_recent_shared_identity_observed_at: datetime


_DIRECT_IDENTITY_TYPES = {
    GraphRelationshipType.USED_DEVICE: GraphNodeType.DEVICE,
    GraphRelationshipType.SEEN_FROM: GraphNodeType.IP_ADDRESS,
    GraphRelationshipType.PAID_WITH: GraphNodeType.PAYMENT_IDENTITY,
}

_EVIDENCE_TYPE_BY_IDENTITY = {
    GraphNodeType.DEVICE: EvidenceType.SHARED_DEVICE,
    GraphNodeType.IP_ADDRESS: EvidenceType.SHARED_IP,
    GraphNodeType.PAYMENT_IDENTITY: EvidenceType.SHARED_PAYMENT_IDENTITY,
    GraphNodeType.ADDRESS: EvidenceType.SHARED_ADDRESS,
}


class EvidenceService:
    """Expose fixed-cutoff evidence through narrow, parameter-free read methods.

    The service receives immutable event records and an already-resolved graph
    projection. It has no driver, query string, write method, or label access.
    Every public method starts from the request's account subject and fixed
    cutoff; callers cannot supply another subject, cutoff, graph query, or
    event filter.
    """

    def __init__(
        self,
        *,
        projection: GraphProjection,
        events: Iterable[Event],
        feature_service: FeatureService,
        frozen_evaluation: FrozenFullEvaluation,
        config: InvestigationConfig,
    ) -> None:
        self._projection = projection
        self._events = tuple(
            sorted(events, key=lambda event: (event.occurred_at, str(event.id)))
        )
        self._feature_service = feature_service
        self._frozen_evaluation = frozen_evaluation
        self._config = config
        self._nodes = {
            (node.node_type, node.canonical_id): node for node in projection.nodes
        }

    @property
    def config(self) -> InvestigationConfig:
        """Return the exact validated limits supplied by trusted application code."""
        return self._config

    def get_risk_explanation(
        self,
        request: InvestigationRequest,
        score_observation: ScoreObservation,
    ) -> tuple[EvidenceItem, ...]:
        """Return bounded TreeSHAP raw-score drivers for the verified score input."""
        vector = self._verified_feature_vector(request, score_observation)
        explanation = explain_prediction(
            self._frozen_evaluation.baseline,
            vector,
            limit=self._config.max_risk_drivers,
        )
        if not isclose(
            explanation.raw_score, score_observation.raw_model_score, abs_tol=1e-12
        ):
            raise ValueError("TreeSHAP explanation does not match verified raw score")
        result: list[EvidenceItem] = []
        for direction, contributions in (
            ("positive", explanation.positive),
            ("negative", explanation.negative),
        ):
            for contribution in contributions:
                facts: dict[str, JsonValue] = {
                    "feature_name": contribution.feature_name,
                    "feature_value": contribution.feature_value,
                    "raw_score_shap_contribution": contribution.shap_value,
                    "direction": direction,
                    "raw_model_score": explanation.raw_score,
                    "base_value": explanation.base_value,
                    "interpretation": (
                        "TreeSHAP model contribution to raw score; not factual proof "
                        "of abuse"
                    ),
                }
                result.append(
                    self._evidence(
                        request,
                        evidence_type=EvidenceType.RISK_DRIVER,
                        source=EvidenceSource.MODEL_EXPLANATION,
                        observed_at=request.cutoff_time,
                        subject_ids=(request.subject_id,),
                        facts=facts,
                    )
                )
        return tuple(result)

    def get_identity_neighborhood(
        self, request: InvestigationRequest
    ) -> tuple[EvidenceItem, ...]:
        """Return the bounded account/identity neighbourhood at the fixed cutoff."""
        links = self._links_at(request.cutoff_time)
        links_by_node = _links_by_node(links)
        root = (GraphNodeType.ACCOUNT, request.subject_id)
        if root not in self._nodes:
            raise ValueError("investigation subject is not present in the graph")
        nodes, edges, truncated = self._bounded_neighborhood(root, links_by_node)
        facts: dict[str, JsonValue] = {
            "root_subject_id": request.subject_id,
            "max_graph_hops": self._config.max_graph_hops,
            "max_graph_nodes": self._config.max_graph_nodes,
            "max_graph_edges": self._config.max_graph_edges,
            "returned_node_count": len(nodes),
            "returned_edge_count": len(edges),
            "truncated": truncated,
            "nodes": [
                {"entity_id": entity_id, "entity_type": node_type.value}
                for node_type, entity_id in nodes
            ],
            "edges": [_link_fact(link) for link in edges],
        }
        return (
            self._evidence(
                request,
                evidence_type=EvidenceType.IDENTITY_NEIGHBORHOOD,
                source=EvidenceSource.TEMPORAL_GRAPH,
                observed_at=request.cutoff_time,
                subject_ids=(request.subject_id,),
                facts=facts,
            ),
        )

    def get_shared_identity_summary(
        self, request: InvestigationRequest
    ) -> tuple[EvidenceItem, ...]:
        """Summarize subject identities shared with other accounts at the cutoff."""
        links = self._links_at(request.cutoff_time)
        account_links = _account_links(links)
        identity_links = _identity_links(links)
        subject_links = account_links.get(request.subject_id, ())
        ranked_related = _rank_related_accounts(
            request.subject_id, subject_links, identity_links
        )
        selected_related = ranked_related[: self._config.max_related_accounts]
        result: list[EvidenceItem] = []
        for subject_link in subject_links:
            same_identity = identity_links[
                (subject_link.identity_type, subject_link.identity_id)
            ]
            all_peers = tuple(
                sorted(
                    link.account_id
                    for link in same_identity
                    if link.account_id != request.subject_id
                )
            )
            if not all_peers:
                continue
            returned_peers = tuple(
                account.account_id
                for account in selected_related
                if account.account_id in all_peers
            )
            first_seen = min(link.first_seen for link in same_identity)
            last_seen = max(link.last_seen for link in same_identity)
            facts: dict[str, JsonValue] = {
                "identity_id": subject_link.identity_id,
                "identity_type": subject_link.identity_type.value,
                "related_account_count": len(all_peers),
                "related_account_ids": list(returned_peers),
                "related_account_ids_truncated": (len(returned_peers) < len(all_peers)),
                "max_related_accounts": self._config.max_related_accounts,
                "first_seen": first_seen.isoformat(),
                "last_seen": last_seen.isoformat(),
            }
            result.append(
                self._evidence(
                    request,
                    evidence_type=_EVIDENCE_TYPE_BY_IDENTITY[
                        subject_link.identity_type
                    ],
                    source=EvidenceSource.IDENTITY_SUMMARY,
                    observed_at=last_seen,
                    subject_ids=(request.subject_id, subject_link.identity_id),
                    facts=facts,
                )
            )
        return tuple(result)

    def get_related_activity(
        self, request: InvestigationRequest
    ) -> tuple[EvidenceItem, ...]:
        """Return bounded, sanitized historical activity for subject and peers."""
        (
            events,
            total_event_count,
            selected_related,
            total_related_count,
        ) = self._activity_events(request)
        metadata: dict[str, JsonValue] = {
            "subject_id": request.subject_id,
            "related_account_ids": [account.account_id for account in selected_related],
            "returned_event_count": len(events),
            "total_event_count": total_event_count,
            "max_events_per_tool": self._config.max_events_per_tool,
            "truncated": total_event_count > len(events),
            "event_selection": "most_recent_n_then_chronological_presentation",
            **_related_account_metadata(
                selected_related,
                total_related_count,
                self._config.max_related_accounts,
            ),
        }
        result = [
            self._evidence(
                request,
                evidence_type=EvidenceType.RELATED_ACCOUNT_ACTIVITY,
                source=EvidenceSource.EVENT_HISTORY,
                observed_at=request.cutoff_time,
                subject_ids=(request.subject_id,),
                facts=metadata,
            )
        ]
        for event in events:
            result.append(
                self._evidence(
                    request,
                    evidence_type=_event_evidence_type(event.event_type),
                    source=EvidenceSource.EVENT_HISTORY,
                    observed_at=event.occurred_at,
                    subject_ids=(str(event.account_id),),
                    facts=_event_facts(
                        event,
                        self._event_canonical_ids(event, request.cutoff_time),
                    ),
                )
            )
        return tuple(result)

    def get_case_timeline(
        self, request: InvestigationRequest
    ) -> tuple[EvidenceItem, ...]:
        """Return one chronologically ordered, cutoff-safe case timeline fact."""
        (
            events,
            total_event_count,
            selected_related,
            total_related_count,
        ) = self._activity_events(request)
        timeline: list[JsonValue] = [
            _event_facts(event, self._event_canonical_ids(event, request.cutoff_time))
            for event in events
        ]
        facts: dict[str, JsonValue] = {
            "subject_id": request.subject_id,
            "related_account_ids": [account.account_id for account in selected_related],
            "returned_event_count": len(timeline),
            "total_event_count": total_event_count,
            "max_events_per_tool": self._config.max_events_per_tool,
            "truncated": total_event_count > len(timeline),
            "event_selection": "most_recent_n_then_chronological_presentation",
            "events": timeline,
            **_related_account_metadata(
                selected_related,
                total_related_count,
                self._config.max_related_accounts,
            ),
        }
        observed_at = events[-1].occurred_at if events else request.cutoff_time
        return (
            self._evidence(
                request,
                evidence_type=EvidenceType.TIMELINE_EVENT,
                source=EvidenceSource.CASE_TIMELINE,
                observed_at=observed_at,
                subject_ids=(request.subject_id,),
                facts=facts,
            ),
        )

    def _verified_feature_vector(
        self,
        request: InvestigationRequest,
        score_observation: ScoreObservation,
    ) -> FeatureVector:
        """Re-score the exact account/cutoff vector before model explanation."""
        if (
            score_observation.score_id != request.score_id
            or score_observation.base_model_id != self._frozen_evaluation.base_model_id
            or score_observation.subject_id != request.subject_id
            or score_observation.scoring_cutoff != request.cutoff_time
            or score_observation.feature_vector_id != request.feature_vector_id
        ):
            raise ValueError("score observation does not match investigation request")
        vector = self._feature_service.extract(request.subject_id, request.cutoff_time)
        verify_score_from_feature_vector(
            score_observation, self._frozen_evaluation, vector
        )
        return vector

    def _links_at(self, cutoff: datetime) -> tuple[_IdentityLink, ...]:
        """Build account/identity links from resolved immutable graph facts only."""
        direct: defaultdict[tuple[str, GraphNodeType, str], list[GraphRelationship]] = (
            defaultdict(list)
        )
        placed: defaultdict[str, list[GraphRelationship]] = defaultdict(list)
        shipped: defaultdict[str, list[GraphRelationship]] = defaultdict(list)
        for relationship in self._projection.relationships:
            if relationship.event_time > cutoff:
                continue
            identity_type = _DIRECT_IDENTITY_TYPES.get(relationship.relationship_type)
            if identity_type is not None:
                direct[
                    (
                        relationship.source_canonical_id,
                        identity_type,
                        relationship.target_canonical_id,
                    )
                ].append(relationship)
            elif relationship.relationship_type is GraphRelationshipType.PLACED:
                placed[relationship.target_canonical_id].append(relationship)
            elif relationship.relationship_type is GraphRelationshipType.SHIPPED_TO:
                shipped[relationship.source_canonical_id].append(relationship)

        result = [_link_from_events(key, values) for key, values in direct.items()]
        for order_id in sorted(set(placed) & set(shipped)):
            for placed_edge in placed[order_id]:
                for shipped_edge in shipped[order_id]:
                    result.append(
                        _IdentityLink(
                            account_id=placed_edge.source_canonical_id,
                            identity_id=shipped_edge.target_canonical_id,
                            identity_type=GraphNodeType.ADDRESS,
                            first_seen=max(
                                placed_edge.event_time, shipped_edge.event_time
                            ),
                            last_seen=max(
                                placed_edge.event_time, shipped_edge.event_time
                            ),
                            event_ids=tuple(
                                sorted({placed_edge.event_id, shipped_edge.event_id})
                            ),
                        )
                    )
        return _merge_links(result)

    def _bounded_neighborhood(
        self,
        root: tuple[GraphNodeType, str],
        links_by_node: dict[tuple[GraphNodeType, str], tuple[_IdentityLink, ...]],
    ) -> tuple[tuple[tuple[GraphNodeType, str], ...], tuple[_IdentityLink, ...], bool]:
        nodes: set[tuple[GraphNodeType, str]] = {root}
        edges: dict[tuple[str, GraphNodeType, str], _IdentityLink] = {}
        pending: deque[tuple[tuple[GraphNodeType, str], int]] = deque([(root, 0)])
        truncated = False
        while pending:
            node, distance = pending.popleft()
            if distance >= self._config.max_graph_hops:
                if links_by_node.get(node):
                    truncated = True
                continue
            for link in links_by_node.get(node, ()):
                edge_key = (link.account_id, link.identity_type, link.identity_id)
                other = (
                    (link.identity_type, link.identity_id)
                    if node[0] is GraphNodeType.ACCOUNT
                    else (GraphNodeType.ACCOUNT, link.account_id)
                )
                if other not in nodes and len(nodes) >= self._config.max_graph_nodes:
                    truncated = True
                    continue
                if edge_key not in edges:
                    if len(edges) >= self._config.max_graph_edges:
                        truncated = True
                        continue
                    edges[edge_key] = link
                if other not in nodes:
                    nodes.add(other)
                    pending.append((other, distance + 1))
        return (
            tuple(sorted(nodes, key=_node_sort_key)),
            tuple(sorted(edges.values(), key=_link_sort_key)),
            truncated,
        )

    def _activity_events(
        self, request: InvestigationRequest
    ) -> tuple[tuple[Event, ...], int, tuple[_RankedRelatedAccount, ...], int]:
        """Return chronological sanitized-source events within account/event budgets."""
        links = self._links_at(request.cutoff_time)
        identity_links = _identity_links(links)
        ranked_related = _rank_related_accounts(
            request.subject_id,
            _account_links(links).get(request.subject_id, ()),
            identity_links,
        )
        selected_related = ranked_related[: self._config.max_related_accounts]
        allowed_accounts = {
            request.subject_id,
            *(account.account_id for account in selected_related),
        }
        all_events = tuple(
            event
            for event in self._events
            if event.occurred_at <= request.cutoff_time
            and str(event.account_id) in allowed_accounts
        )
        selected_events = all_events[-self._config.max_events_per_tool :]
        return selected_events, len(all_events), selected_related, len(ranked_related)

    def _event_canonical_ids(self, event: Event, cutoff: datetime) -> dict[str, str]:
        """Map event-owned graph endpoints to resolved IDs without labels."""
        event_relationships = tuple(
            relationship
            for relationship in self._projection.relationships
            if relationship.event_id == str(event.id)
            and relationship.event_time <= cutoff
        )
        values: dict[str, str] = {}
        for relationship in event_relationships:
            if relationship.target_type is GraphNodeType.DEVICE:
                values["device_id"] = relationship.target_canonical_id
            elif relationship.target_type is GraphNodeType.IP_ADDRESS:
                values["ip_address_id"] = relationship.target_canonical_id
            elif relationship.target_type is GraphNodeType.PAYMENT_IDENTITY:
                values["payment_identity_id"] = relationship.target_canonical_id
            elif relationship.target_type is GraphNodeType.ADDRESS:
                values["address_id"] = relationship.target_canonical_id
        return values

    def _evidence(
        self,
        request: InvestigationRequest,
        *,
        evidence_type: EvidenceType,
        source: EvidenceSource,
        observed_at: datetime,
        subject_ids: tuple[str, ...],
        facts: dict[str, JsonValue],
    ) -> EvidenceItem:
        """Create and verify one deterministic cutoff-bound evidence item."""
        identifier = evidence_id(
            request,
            evidence_type=evidence_type,
            source=source,
            observed_at=observed_at,
            subject_ids=subject_ids,
            facts=facts,
        )
        return EvidenceItem.from_request(
            request,
            evidence_id=identifier,
            evidence_type=evidence_type,
            source=source,
            observed_at=observed_at,
            subject_ids=subject_ids,
            facts=facts,
        ).verify_for_request(request)


def _link_from_events(
    key: tuple[str, GraphNodeType, str], events: list[GraphRelationship]
) -> _IdentityLink:
    account_id, identity_type, identity_id = key
    ordered = sorted(events, key=lambda item: (item.event_time, item.event_id))
    return _IdentityLink(
        account_id=account_id,
        identity_id=identity_id,
        identity_type=identity_type,
        first_seen=ordered[0].event_time,
        last_seen=ordered[-1].event_time,
        event_ids=tuple(item.event_id for item in ordered),
    )


def _rank_related_accounts(
    subject_id: str,
    subject_links: tuple[_IdentityLink, ...],
    identity_links: dict[tuple[GraphNodeType, str], tuple[_IdentityLink, ...]],
) -> tuple[_RankedRelatedAccount, ...]:
    """Rank peers by cutoff-safe shared identity breadth, recency, then ID."""
    shared_types: defaultdict[str, set[GraphNodeType]] = defaultdict(set)
    latest_seen: dict[str, datetime] = {}
    for subject_link in subject_links:
        for peer_link in identity_links[
            (subject_link.identity_type, subject_link.identity_id)
        ]:
            if peer_link.account_id == subject_id:
                continue
            shared_types[peer_link.account_id].add(subject_link.identity_type)
            observed_at = peer_link.last_seen
            latest_seen[peer_link.account_id] = max(
                latest_seen.get(peer_link.account_id, observed_at), observed_at
            )
    return tuple(
        sorted(
            (
                _RankedRelatedAccount(
                    account_id=account_id,
                    shared_identity_type_count=len(identity_types),
                    most_recent_shared_identity_observed_at=latest_seen[account_id],
                )
                for account_id, identity_types in shared_types.items()
            ),
            key=lambda item: (
                -item.shared_identity_type_count,
                -item.most_recent_shared_identity_observed_at.timestamp(),
                item.account_id,
            ),
        )
    )


def _related_account_metadata(
    selected: tuple[_RankedRelatedAccount, ...],
    total_count: int,
    max_related_accounts: int,
) -> dict[str, JsonValue]:
    """Describe bounded peer selection without exposing unselected identities."""
    return {
        "selected_related_account_ids": [item.account_id for item in selected],
        "returned_related_account_count": len(selected),
        "total_related_account_count": total_count,
        "related_accounts_truncated": total_count > len(selected),
        "max_related_accounts": max_related_accounts,
        "related_account_ranking": [
            {
                "rank": index,
                "account_id": item.account_id,
                "shared_identity_type_count": item.shared_identity_type_count,
                "most_recent_shared_identity_observed_at": (
                    item.most_recent_shared_identity_observed_at.isoformat()
                ),
            }
            for index, item in enumerate(selected, start=1)
        ],
    }


def _merge_links(links: Iterable[_IdentityLink]) -> tuple[_IdentityLink, ...]:
    grouped: defaultdict[tuple[str, GraphNodeType, str], list[_IdentityLink]] = (
        defaultdict(list)
    )
    for link in links:
        grouped[(link.account_id, link.identity_type, link.identity_id)].append(link)
    result: list[_IdentityLink] = []
    for key, members in grouped.items():
        result.append(
            _IdentityLink(
                account_id=key[0],
                identity_type=key[1],
                identity_id=key[2],
                first_seen=min(item.first_seen for item in members),
                last_seen=max(item.last_seen for item in members),
                event_ids=tuple(
                    sorted(
                        {event_id for item in members for event_id in item.event_ids}
                    )
                ),
            )
        )
    return tuple(sorted(result, key=_link_sort_key))


def _account_links(
    links: Iterable[_IdentityLink],
) -> dict[str, tuple[_IdentityLink, ...]]:
    result: defaultdict[str, list[_IdentityLink]] = defaultdict(list)
    for link in links:
        result[link.account_id].append(link)
    return {
        account_id: tuple(sorted(values, key=_link_sort_key))
        for account_id, values in result.items()
    }


def _identity_links(
    links: Iterable[_IdentityLink],
) -> dict[tuple[GraphNodeType, str], tuple[_IdentityLink, ...]]:
    result: defaultdict[tuple[GraphNodeType, str], list[_IdentityLink]] = defaultdict(
        list
    )
    for link in links:
        result[(link.identity_type, link.identity_id)].append(link)
    return {
        identity: tuple(sorted(values, key=_link_sort_key))
        for identity, values in result.items()
    }


def _links_by_node(
    links: Iterable[_IdentityLink],
) -> dict[tuple[GraphNodeType, str], tuple[_IdentityLink, ...]]:
    result: defaultdict[tuple[GraphNodeType, str], list[_IdentityLink]] = defaultdict(
        list
    )
    for link in links:
        result[(GraphNodeType.ACCOUNT, link.account_id)].append(link)
        result[(link.identity_type, link.identity_id)].append(link)
    return {
        node: tuple(sorted(values, key=_link_sort_key))
        for node, values in result.items()
    }


def _node_sort_key(node: tuple[GraphNodeType, str]) -> tuple[str, str]:
    return node[0].value, node[1]


def _link_sort_key(link: _IdentityLink) -> tuple[str, str, str]:
    return link.identity_type.value, link.identity_id, link.account_id


def _link_fact(link: _IdentityLink) -> dict[str, JsonValue]:
    return {
        "account_id": link.account_id,
        "identity_id": link.identity_id,
        "identity_type": link.identity_type.value,
        "first_seen": link.first_seen.isoformat(),
        "last_seen": link.last_seen.isoformat(),
        "event_ids": list(link.event_ids),
    }


def _event_evidence_type(event_type: EventType) -> EvidenceType:
    if event_type is EventType.PROMOTION_REDEEMED:
        return EvidenceType.PROMOTION_ACTIVITY
    if event_type in {EventType.REFUND_REQUESTED, EventType.REFUND_RESOLVED}:
        return EvidenceType.REFUND_ACTIVITY
    return EvidenceType.RELATED_ACCOUNT_ACTIVITY


def _event_facts(event: Event, canonical_ids: dict[str, str]) -> dict[str, JsonValue]:
    """Return a deliberately label-free, machine-readable event representation."""
    facts: dict[str, JsonValue] = {
        "event_id": str(event.id),
        "event_type": event.event_type.value,
        "occurred_at": event.occurred_at.isoformat(),
        "account_id": str(event.account_id),
    }
    for field in (
        "device_id",
        "ip_address_id",
        "payment_identity_id",
        "order_id",
        "address_id",
        "promotion_id",
        "refund_id",
    ):
        value = canonical_ids.get(field, getattr(event, field))
        if value is not None:
            facts[field] = str(value)
    return facts
