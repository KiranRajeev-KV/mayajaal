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
)
from mayajaal.policy import (
    DecisionContext,
    PolicyAction,
    PolicyConfig,
    build_policy_model,
    decide,
)
from mayajaal.schemas import Event, EventType
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
        self.assertTrue(bool(metadata.facts["truncated"]))
        self.assertEqual(metadata.facts["returned_event_count"], 3)
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

    def test_activity_exposes_promotion_and_refund_without_labels(self) -> None:
        service = self.service(max_events_per_tool=20)
        activity = service.get_related_activity(request())
        evidence_types = {item.evidence_type for item in activity}
        self.assertIn(EvidenceType.PROMOTION_ACTIVITY, evidence_types)
        self.assertIn(EvidenceType.REFUND_ACTIVITY, evidence_types)
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


if __name__ == "__main__":
    _ = unittest.main()
