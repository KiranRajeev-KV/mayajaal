"""Focused tests for bounded, label-free investigation evidence retrieval."""

import inspect
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from uuid import UUID, uuid5

import mayajaal.investigation.service as investigation_service
from mayajaal.baseline import BaselineConfig, train_baseline
from mayajaal.calibration import (
    CalibrationConfig,
    ProbabilityModel,
    SigmoidCalibrator,
    estimate_probability,
)
from mayajaal.evaluation.models import SplitManifest
from mayajaal.evaluation.provenance import FrozenFullEvaluation
from mayajaal.features import (
    FeatureDefinition,
    FeatureKind,
    FeatureSchema,
    FeatureService,
    FeatureVector,
    LabeledFeatureVector,
)
from mayajaal.graph import (
    GraphNode,
    GraphNodeType,
    GraphProjection,
    GraphRelationship,
    GraphRelationshipType,
)
from mayajaal.investigation import (
    EvidenceService,
    EvidenceType,
    InvestigationConfig,
    InvestigationRequest,
    InvestigationSubjectType,
    InvestigationToolContext,
    build_investigation_tools,
)
from mayajaal.policy import (
    DecisionContext,
    PolicyAction,
    PolicyConfig,
    build_policy_model,
    decide,
)
from mayajaal.schemas import Event, EventType
from mayajaal.scoring import ScoreObservation
from mayajaal.scoring.service import score_feature_vector

SUBJECT_ID = str(uuid5(UUID("00000000-0000-0000-0000-000000000001"), "account-subject"))
PEER_ID = str(uuid5(UUID("00000000-0000-0000-0000-000000000001"), "account-peer"))


def at(day: int, hour: int = 0) -> datetime:
    """Return a fixed UTC fixture timestamp."""
    return datetime(2026, 1, day, hour, tzinfo=UTC)


def identifier(name: str) -> UUID:
    """Return a reproducible UUID without a synthetic label or campaign input."""
    return uuid5(UUID("00000000-0000-0000-0000-000000000001"), name)


def event(
    name: str,
    event_type: EventType,
    occurred_at: datetime,
    account_id: UUID,
    **ids: UUID,
) -> Event:
    """Create a public event fact with no synthetic ground-truth field."""
    return Event.model_validate(
        {
            "id": identifier(f"event:{name}"),
            "event_type": event_type,
            "occurred_at": occurred_at,
            "ingested_at": occurred_at + timedelta(seconds=1),
            "account_id": account_id,
            **ids,
        }
    )


def request(subject_id: str = SUBJECT_ID) -> InvestigationRequest:
    """Return a lineage-shaped request for read-only fixture calls."""
    return InvestigationRequest(
        decision_id="decision-fixture",
        policy_id="policy-fixture",
        probability_estimate_id="estimate-fixture",
        score_id="score-fixture",
        feature_vector_id="vector-fixture",
        subject_type=InvestigationSubjectType.ACCOUNT,
        subject_id=subject_id,
        cutoff_time=at(10),
        context_id="order-context",
        policy_action=PolicyAction.REVIEW,
        decision_is_stable_across_scenarios=True,
    )


def relationship(
    relationship_type: GraphRelationshipType,
    source_type: GraphNodeType,
    source_id: str,
    target_type: GraphNodeType,
    target_id: str,
    event_id: str,
    event_time: datetime,
) -> GraphRelationship:
    """Build one immutable event-backed graph edge."""
    return GraphRelationship(
        relationship_type=relationship_type,
        source_type=source_type,
        source_canonical_id=source_id,
        target_type=target_type,
        target_canonical_id=target_id,
        event_id=event_id,
        event_type="fixture_event",
        event_time=event_time,
    )


def evidence_projection(events: tuple[Event, ...]) -> GraphProjection:
    """Build two account identity links with all supported identity types."""
    subject = SUBJECT_ID
    peer = PEER_ID
    identities = (
        (GraphRelationshipType.USED_DEVICE, GraphNodeType.DEVICE, "device-shared"),
        (GraphRelationshipType.SEEN_FROM, GraphNodeType.IP_ADDRESS, "ip-shared"),
        (
            GraphRelationshipType.PAID_WITH,
            GraphNodeType.PAYMENT_IDENTITY,
            "payment-shared",
        ),
    )
    nodes = [
        GraphNode(GraphNodeType.ACCOUNT, subject, {"created_at": at(1)}),
        GraphNode(GraphNodeType.ACCOUNT, peer, {"created_at": at(1)}),
        GraphNode(GraphNodeType.ORDER, "order-subject", {}),
        GraphNode(GraphNodeType.ORDER, "order-peer", {}),
        GraphNode(GraphNodeType.ADDRESS, "address-shared", {}),
    ]
    relationships: list[GraphRelationship] = []
    for relationship_type, node_type, identity_id in identities:
        nodes.append(GraphNode(node_type, identity_id, {}))
        for account_id, when in ((subject, at(2)), (peer, at(3))):
            event_id = str(identifier(f"event:{relationship_type}:{account_id}"))
            relationships.append(
                relationship(
                    relationship_type,
                    GraphNodeType.ACCOUNT,
                    account_id,
                    node_type,
                    identity_id,
                    event_id,
                    when,
                )
            )
    for account_id, order_id, when in (
        (subject, "order-subject", at(4)),
        (peer, "order-peer", at(5)),
    ):
        event_id = str(identifier(f"event:order:{account_id}"))
        relationships.extend(
            (
                relationship(
                    GraphRelationshipType.PLACED,
                    GraphNodeType.ACCOUNT,
                    account_id,
                    GraphNodeType.ORDER,
                    order_id,
                    event_id,
                    when,
                ),
                relationship(
                    GraphRelationshipType.SHIPPED_TO,
                    GraphNodeType.ORDER,
                    order_id,
                    GraphNodeType.ADDRESS,
                    "address-shared",
                    event_id,
                    when,
                ),
            )
        )
    relationships.append(
        relationship(
            GraphRelationshipType.USED_DEVICE,
            GraphNodeType.ACCOUNT,
            subject,
            GraphNodeType.DEVICE,
            "future-device",
            "future-event",
            at(11),
        )
    )
    return GraphProjection(nodes=tuple(nodes), relationships=tuple(relationships))


def ranking_projection() -> GraphProjection:
    """Build overlap/recency/tie fixtures for deterministic peer ranking."""
    subject = SUBJECT_ID
    peers = (
        "account-many",
        "account-recent",
        "account-older",
        "account-a",
        "account-b",
    )
    nodes = [GraphNode(GraphNodeType.ACCOUNT, subject, {"created_at": at(1)})]
    nodes.extend(
        GraphNode(GraphNodeType.ACCOUNT, account_id, {"created_at": at(1)})
        for account_id in peers
    )
    relationships: list[GraphRelationship] = []

    def add_shared(
        identity_type: GraphNodeType,
        identity_id: str,
        relationship_type: GraphRelationshipType,
        peer_id: str,
        peer_time: datetime,
        subject_time: datetime | None = None,
    ) -> None:
        nodes.append(GraphNode(identity_type, identity_id, {}))
        relationships.extend(
            (
                relationship(
                    relationship_type,
                    GraphNodeType.ACCOUNT,
                    subject,
                    identity_type,
                    identity_id,
                    f"subject-{identity_id}",
                    subject_time or at(2),
                ),
                relationship(
                    relationship_type,
                    GraphNodeType.ACCOUNT,
                    peer_id,
                    identity_type,
                    identity_id,
                    f"{peer_id}-{identity_id}",
                    peer_time,
                ),
            )
        )

    add_shared(
        GraphNodeType.DEVICE,
        "device-many",
        GraphRelationshipType.USED_DEVICE,
        "account-many",
        at(3),
    )
    add_shared(
        GraphNodeType.IP_ADDRESS,
        "ip-many",
        GraphRelationshipType.SEEN_FROM,
        "account-many",
        at(4),
    )
    add_shared(
        GraphNodeType.PAYMENT_IDENTITY,
        "payment-recency",
        GraphRelationshipType.PAID_WITH,
        "account-recent",
        at(9),
        at(10),
    )
    add_shared(
        GraphNodeType.PAYMENT_IDENTITY,
        "payment-older",
        GraphRelationshipType.PAID_WITH,
        "account-older",
        at(7),
        at(10),
    )
    add_shared(
        GraphNodeType.DEVICE,
        "device-tie",
        GraphRelationshipType.USED_DEVICE,
        "account-a",
        at(6),
    )
    add_shared(
        GraphNodeType.DEVICE,
        "device-tie",
        GraphRelationshipType.USED_DEVICE,
        "account-b",
        at(6),
    )
    return GraphProjection(nodes=tuple(nodes), relationships=tuple(relationships))


def minimal_frozen_evaluation() -> FrozenFullEvaluation:
    """Create a real, tiny CatBoost model for dependencies not used by graph tools."""
    schema = FeatureSchema(
        (FeatureDefinition("count", FeatureKind.NUMERIC, "Fixture count."),)
    )
    examples = tuple(
        LabeledFeatureVector(
            FeatureVector(f"account-{index}", at(1), {"count": float(index)}),
            index > 1,
        )
        for index in range(4)
    )
    baseline = train_baseline(
        examples, schema, BaselineConfig(iterations=4, depth=2, learning_rate=0.1)
    )
    return FrozenFullEvaluation(
        evaluation_directory=Path("."),
        manifest=SplitManifest(at(1), at(2), at(3), ()),
        records=(),
        raw_scores={},
        baseline=baseline,
        provenance={"base_model_id": "base-model-fixture"},
    )


def fixture_events() -> tuple[Event, ...]:
    """Public activity facts covering promotion and refund retrieval."""
    subject = identifier("account-subject")
    peer = identifier("account-peer")
    device = identifier("device")
    ip_address = identifier("ip")
    payment = identifier("payment")
    address = identifier("address")
    order = identifier("order")
    promotion = identifier("promotion")
    refund = identifier("refund")
    return (
        event("created", EventType.ACCOUNT_CREATED, at(1), subject),
        event("device", EventType.DEVICE_SEEN, at(2), subject, device_id=device),
        event("ip", EventType.IP_SEEN, at(2, 1), subject, ip_address_id=ip_address),
        event(
            "payment",
            EventType.PAYMENT_ATTACHED,
            at(2, 2),
            subject,
            payment_identity_id=payment,
        ),
        event(
            "order",
            EventType.ORDER_PLACED,
            at(4),
            subject,
            order_id=order,
            address_id=address,
        ),
        event(
            "promo",
            EventType.PROMOTION_REDEEMED,
            at(5),
            subject,
            order_id=order,
            promotion_id=promotion,
        ),
        event(
            "refund-request",
            EventType.REFUND_REQUESTED,
            at(6),
            subject,
            order_id=order,
            refund_id=refund,
        ),
        event(
            "refund-resolved",
            EventType.REFUND_RESOLVED,
            at(7),
            subject,
            order_id=order,
            refund_id=refund,
        ),
        event("peer-created", EventType.ACCOUNT_CREATED, at(2), peer),
        event("future", EventType.ACCOUNT_CREATED, at(11), peer),
    )


class EvidenceServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.events = fixture_events()
        cls.projection = evidence_projection(cls.events)
        cls.frozen = minimal_frozen_evaluation()

    def service(self, **config_values: int) -> EvidenceService:
        config = InvestigationConfig.model_validate(config_values)
        return EvidenceService(
            projection=self.projection,
            events=self.events,
            feature_service=FeatureService(self.projection),
            frozen_evaluation=self.frozen,
            config=config,
        )

    def test_identity_summary_is_cutoff_safe_deterministic_and_benign_sharing_ready(
        self,
    ) -> None:
        first = self.service().get_shared_identity_summary(request())
        second = self.service().get_shared_identity_summary(request())
        self.assertEqual(first, second)
        self.assertEqual(
            {item.evidence_type for item in first},
            {
                EvidenceType.SHARED_DEVICE,
                EvidenceType.SHARED_IP,
                EvidenceType.SHARED_PAYMENT_IDENTITY,
                EvidenceType.SHARED_ADDRESS,
            },
        )
        self.assertTrue(all(item.cutoff_time == at(10) for item in first))
        self.assertTrue(all(item.observed_at <= item.cutoff_time for item in first))
        self.assertTrue(
            all(item.facts["related_account_ids"] == [PEER_ID] for item in first)
        )

    def test_service_freezes_external_limits_at_construction(self) -> None:
        config = InvestigationConfig(max_events_per_tool=1, max_related_accounts=1)
        service = EvidenceService(
            projection=self.projection,
            events=self.events,
            feature_service=FeatureService(self.projection),
            frozen_evaluation=self.frozen,
            config=config,
        )
        config.max_events_per_tool = 100
        config.max_related_accounts = 100
        self.assertEqual(service.config.max_events_per_tool, 1)
        self.assertEqual(service.config.max_related_accounts, 1)
        activity = service.get_related_activity(request())
        self.assertEqual(activity[0].facts["detailed_retrieval_event_count"], 1)

    def test_tool_wrapper_exposes_aliases_and_preserves_service_bounds(self) -> None:
        service = self.service(max_related_accounts=1)
        score = ScoreObservation(
            score_id="score-fixture",
            base_model_id=self.frozen.base_model_id,
            subject_id=SUBJECT_ID,
            scoring_cutoff=at(10),
            raw_model_score=0.0,
            feature_vector_id="vector-fixture",
        )
        context = InvestigationToolContext.create(
            request=request(),
            evidence_service=service,
            score_observation=score,
            config=InvestigationConfig(max_tool_calls=1, max_related_accounts=1),
        )
        wrapped = {tool.name: tool for tool in build_investigation_tools(context)}[
            "shared_identity_summary"
        ]
        direct = service.get_shared_identity_summary(request())
        returned = wrapped.invoke({})
        self.assertEqual(len(returned), len(direct))
        self.assertEqual(
            [item["evidence_ref"] for item in returned],
            [f"E{index:03d}" for index in range(1, len(direct) + 1)],
        )
        self.assertTrue(all("evidence_id" not in item for item in returned))
        self.assertEqual(
            [
                context.ledger.resolve_alias(str(item["evidence_ref"]))
                for item in returned
            ],
            [item.evidence_id for item in direct],
        )
        self.assertTrue(
            all(item["facts"]["max_related_accounts"] == 1 for item in returned)
        )

    def test_graph_and_related_account_budgets_truncate_deterministically(self) -> None:
        service = self.service(
            max_graph_hops=2,
            max_graph_nodes=2,
            max_graph_edges=1,
            max_related_accounts=1,
        )
        neighborhood = service.get_identity_neighborhood(request())[0]
        self.assertIsInstance(neighborhood.facts["returned_node_count"], int)
        self.assertIsInstance(neighborhood.facts["returned_edge_count"], int)
        self.assertLessEqual(cast(int, neighborhood.facts["returned_node_count"]), 2)
        self.assertLessEqual(cast(int, neighborhood.facts["returned_edge_count"]), 1)
        self.assertTrue(bool(neighborhood.facts["truncated"]))
        summary = service.get_shared_identity_summary(request())
        self.assertTrue(
            all(
                bool(item.facts["related_account_ids_truncated"]) is False
                for item in summary
            )
        )

        extra_peer = "account-extra-peer"
        augmented = GraphProjection(
            nodes=(
                *self.projection.nodes,
                GraphNode(GraphNodeType.ACCOUNT, extra_peer, {"created_at": at(1)}),
            ),
            relationships=(
                *self.projection.relationships,
                relationship(
                    GraphRelationshipType.USED_DEVICE,
                    GraphNodeType.ACCOUNT,
                    extra_peer,
                    GraphNodeType.DEVICE,
                    "device-shared",
                    "extra-device-event",
                    at(6),
                ),
            ),
        )
        bounded = EvidenceService(
            projection=augmented,
            events=self.events,
            feature_service=FeatureService(augmented),
            frozen_evaluation=self.frozen,
            config=InvestigationConfig(max_related_accounts=1),
        ).get_shared_identity_summary(request())
        device_summary = next(
            item for item in bounded if item.evidence_type is EvidenceType.SHARED_DEVICE
        )
        self.assertEqual(
            len(cast(list[str], device_summary.facts["related_account_ids"])), 1
        )
        self.assertTrue(bool(device_summary.facts["related_account_ids_truncated"]))

    def test_activity_and_timeline_are_label_free_ordered_and_event_bounded(
        self,
    ) -> None:
        service = self.service(max_events_per_tool=3)
        activity = service.get_related_activity(request())
        metadata = activity[0]
        self.assertTrue(bool(metadata.facts["detailed_retrieval_truncated"]))
        self.assertEqual(metadata.facts["detailed_retrieval_event_count"], 3)
        self.assertEqual(metadata.facts["aggregate_scope"], "all_eligible_events")
        self.assertGreater(
            cast(int, metadata.facts["aggregate_event_count"]),
            cast(int, metadata.facts["detailed_retrieval_event_count"]),
        )
        self.assertTrue(all(item.cutoff_time == at(10) for item in activity))
        self.assertTrue(
            all(
                item.observed_at <= at(10) and "synthetic_labels" not in str(item.facts)
                for item in activity
            )
        )
        timeline = service.get_case_timeline(request())[0]
        timeline_events = cast(list[dict[str, object]], timeline.facts["events"])
        self.assertEqual(len(timeline_events), 3)
        self.assertEqual(
            [item["occurred_at"] for item in timeline_events],
            sorted(str(item["occurred_at"]) for item in timeline_events),
        )
        self.assertTrue(bool(timeline.facts["truncated"]))

    def test_activity_is_aggregate_and_timeline_keeps_recent_high_signal_events(
        self,
    ) -> None:
        service = self.service(max_events_per_tool=4, max_timeline_events=2)
        activity = service.get_related_activity(request())
        self.assertEqual(len(activity), 1)
        self.assertNotIn("events", activity[0].facts)
        activity_counts = cast(
            dict[str, int], activity[0].facts["event_counts_by_type"]
        )
        self.assertEqual(activity_counts[EventType.PROMOTION_REDEEMED.value], 1)
        self.assertEqual(activity[0].facts["refund_event_count"], 2)
        self.assertEqual(activity[0].facts["order_event_count"], 1)
        self.assertEqual(activity[0].facts["payment_attachment_count"], 1)
        self.assertTrue(bool(activity[0].facts["detailed_retrieval_truncated"]))
        self.assertEqual(activity[0].facts["aggregate_scope"], "all_eligible_events")
        self.assertGreater(
            cast(int, activity[0].facts["aggregate_event_count"]),
            cast(int, activity[0].facts["detailed_retrieval_event_count"]),
        )
        timeline = service.get_case_timeline(request())[0]
        timeline_events = cast(list[dict[str, object]], timeline.facts["events"])
        self.assertEqual(
            [item["event_id"] for item in timeline_events],
            [
                str(identifier("event:refund-request")),
                str(identifier("event:refund-resolved")),
            ],
        )
        self.assertTrue(bool(timeline.facts["presentation_truncated"]))
        self.assertNotEqual(activity[0].facts, timeline.facts)

    def test_related_accounts_rank_by_overlap_recency_then_identifier(self) -> None:
        projection = ranking_projection()
        service = EvidenceService(
            projection=projection,
            events=self.events,
            feature_service=FeatureService(projection),
            frozen_evaluation=self.frozen,
            config=InvestigationConfig(max_related_accounts=5),
        )
        first = service.get_related_activity(request())[0]
        second = service.get_related_activity(request())[0]
        self.assertEqual(first, second)
        ranking = cast(list[dict[str, object]], first.facts["related_account_ranking"])
        self.assertEqual(
            [item["account_id"] for item in ranking],
            [
                "account-many",
                "account-recent",
                "account-older",
                "account-a",
                "account-b",
            ],
        )
        self.assertEqual(ranking[0]["shared_identity_type_count"], 2)
        self.assertEqual(
            ranking[1]["most_recent_shared_identity_observed_at"], at(9).isoformat()
        )
        self.assertEqual(
            ranking[2]["most_recent_shared_identity_observed_at"], at(7).isoformat()
        )
        self.assertEqual(
            first.facts["selected_related_account_ids"],
            [item["account_id"] for item in ranking],
        )
        self.assertEqual(first.evidence_id, second.evidence_id)

    def test_related_account_limit_uses_the_same_ranked_selection_everywhere(
        self,
    ) -> None:
        projection = ranking_projection()
        service = EvidenceService(
            projection=projection,
            events=self.events,
            feature_service=FeatureService(projection),
            frozen_evaluation=self.frozen,
            config=InvestigationConfig(max_related_accounts=2, max_events_per_tool=4),
        )
        summary = service.get_shared_identity_summary(request())
        activity = service.get_related_activity(request())
        timeline = service.get_case_timeline(request())[0]
        expected = ["account-many", "account-recent"]
        for item in (activity[0], timeline):
            self.assertEqual(item.facts["selected_related_account_ids"], expected)
            self.assertTrue(bool(item.facts["related_accounts_truncated"]))
            self.assertLessEqual(
                cast(int, item.facts["returned_related_account_count"]), 2
            )
        self.assertTrue(
            all(
                "related_account_ranking" not in item.facts
                and "related_account_ids_truncated" in item.facts
                for item in summary
            )
        )

    def test_activity_exposes_promotion_and_refund_without_labels(self) -> None:
        service = self.service(max_events_per_tool=20)
        activity = service.get_related_activity(request())
        self.assertEqual(len(activity), 1)
        self.assertEqual(activity[0].facts["promotion_event_count"], 1)
        self.assertEqual(activity[0].facts["refund_event_count"], 2)
        self.assertEqual(activity[0].facts["order_event_count"], 1)
        self.assertEqual(activity[0].facts["payment_attachment_count"], 1)
        self.assertTrue(
            all("synthetic" not in str(item.facts).casefold() for item in activity)
        )

        labelled_source_events = tuple(
            event.model_copy(
                update={
                    "synthetic_labels": {
                        "is_coordinated_abuse": True,
                        "abuse_types": ["promotion_abuse"],
                    }
                }
            )
            for event in self.events
        )
        with_labels = EvidenceService(
            projection=self.projection,
            events=labelled_source_events,
            feature_service=FeatureService(self.projection),
            frozen_evaluation=self.frozen,
            config=InvestigationConfig(max_events_per_tool=20),
        ).get_related_activity(request())
        self.assertTrue(
            all("synthetic" not in str(item.facts).casefold() for item in with_labels)
        )

    def test_public_methods_offer_no_subject_cutoff_or_query_override(self) -> None:
        for method_name in (
            "get_identity_neighborhood",
            "get_shared_identity_summary",
            "get_related_activity",
            "get_case_timeline",
        ):
            parameters = inspect.signature(
                getattr(EvidenceService, method_name)
            ).parameters
            self.assertEqual(tuple(parameters), ("self", "request"))
        source = inspect.getsource(investigation_service)
        self.assertNotIn("synthetic_labels", source)
        self.assertNotIn("langchain", source.casefold())
        self.assertNotIn("openai", source.casefold())

    def test_risk_explanation_reverifies_actual_feature_vector_and_score(self) -> None:
        schema = FeatureSchema(
            (FeatureDefinition("count", FeatureKind.NUMERIC, "Fixture count."),)
        )
        vectors = tuple(
            FeatureVector(f"account-{index}", at(1), {"count": float(index)})
            for index in range(4)
        )
        baseline = train_baseline(
            tuple(
                LabeledFeatureVector(vector, index > 1)
                for index, vector in enumerate(vectors)
            ),
            schema,
            BaselineConfig(iterations=4, depth=2, learning_rate=0.1),
        )
        frozen = FrozenFullEvaluation(
            evaluation_directory=Path("."),
            manifest=SplitManifest(at(1), at(2), at(3), ()),
            records=(),
            raw_scores={},
            baseline=baseline,
            provenance={"base_model_id": "base-model-fixture"},
        )

        class SingleVectorService:
            def extract(self, account_id: str, cutoff: datetime) -> FeatureVector:
                if account_id != "account-3" or cutoff != at(3):
                    raise ValueError("wrong account or cutoff")
                return FeatureVector("account-3", at(3), {"count": 3.0})

        vector = SingleVectorService().extract("account-3", at(3))
        score = score_feature_vector(frozen, vector)
        probability_model = ProbabilityModel(
            base_model_id=frozen.base_model_id,
            probability_model_id="probability-model-fixture",
            calibration_config=CalibrationConfig(
                minimum_positive_samples=1, minimum_negative_samples=1
            ),
            calibrator=SigmoidCalibrator(coefficient=1.0, intercept=0.0),
            frozen_provenance={"base_model_id": frozen.base_model_id},
        )
        estimate = estimate_probability(probability_model, score)
        policy_model = build_policy_model(probability_model, PolicyConfig())
        decision = decide(
            policy_model,
            probability_model,
            score,
            estimate,
            DecisionContext(exposure_paise=250_000),
        )
        investigation_request = InvestigationRequest.from_policy_decision(
            decision, probability_model, score, estimate
        )
        evidence_service = EvidenceService(
            projection=self.projection,
            events=self.events,
            feature_service=cast(FeatureService, SingleVectorService()),
            frozen_evaluation=frozen,
            config=InvestigationConfig(max_risk_drivers=1),
        )
        explanations = evidence_service.get_risk_explanation(
            investigation_request, score
        )
        self.assertEqual(len(explanations), 1)
        self.assertIs(explanations[0].evidence_type, EvidenceType.RISK_DRIVER)
        self.assertEqual(explanations[0].facts["feature_name"], "count")
        self.assertIn("not factual proof", str(explanations[0].facts))
        with self.assertRaisesRegex(ValueError, "does not match investigation request"):
            _ = evidence_service.get_risk_explanation(request(), score)

    def test_risk_explanation_uses_persisted_snapshot_not_changed_graph_features(
        self,
    ) -> None:
        schema = FeatureSchema(
            (
                FeatureDefinition(
                    "latest_payment_method", FeatureKind.CATEGORICAL, "Fixture."
                ),
            )
        )
        vectors = tuple(
            FeatureVector(
                f"account-{index}",
                at(3),
                {"latest_payment_method": value},
            )
            for index, value in enumerate(
                ("__missing__", "card", "card", "__missing__")
            )
        )
        baseline = train_baseline(
            tuple(
                LabeledFeatureVector(vector, index > 1)
                for index, vector in enumerate(vectors)
            ),
            schema,
            BaselineConfig(iterations=4, depth=2, learning_rate=0.1),
        )
        frozen = FrozenFullEvaluation(
            evaluation_directory=Path("."),
            manifest=SplitManifest(at(1), at(2), at(3), ()),
            records=(),
            raw_scores={},
            baseline=baseline,
            provenance={"base_model_id": "base-model-fixture"},
        )
        persisted_vector = vectors[3]
        score = score_feature_vector(frozen, persisted_vector)
        probability_model = ProbabilityModel(
            base_model_id=frozen.base_model_id,
            probability_model_id="probability-model-fixture",
            calibration_config=CalibrationConfig(
                minimum_positive_samples=1, minimum_negative_samples=1
            ),
            calibrator=SigmoidCalibrator(coefficient=1.0, intercept=0.0),
            frozen_provenance={"base_model_id": frozen.base_model_id},
        )
        estimate = estimate_probability(probability_model, score)
        decision = decide(
            build_policy_model(probability_model, PolicyConfig()),
            probability_model,
            score,
            estimate,
            DecisionContext(exposure_paise=250_000),
        )
        investigation_request = InvestigationRequest.from_policy_decision(
            decision, probability_model, score, estimate
        )

        class ChangedGraphFeatureService:
            def extract(self, *_: object) -> FeatureVector:
                raise AssertionError("mutable graph feature extraction must not run")

        evidence_service = EvidenceService(
            projection=self.projection,
            events=self.events,
            feature_service=cast(FeatureService, ChangedGraphFeatureService()),
            frozen_evaluation=frozen,
            config=InvestigationConfig(max_risk_drivers=1),
            feature_vector=persisted_vector,
        )
        explanations = evidence_service.get_risk_explanation(
            investigation_request, score
        )
        self.assertEqual(
            evidence_service._verified_feature_vector(  # pyright: ignore[reportPrivateUsage]
                investigation_request, score
            ).values["latest_payment_method"],
            "__missing__",
        )
        self.assertTrue(
            all(item.facts["feature_value"] == "__missing__" for item in explanations)
        )


if __name__ == "__main__":
    _ = unittest.main()
