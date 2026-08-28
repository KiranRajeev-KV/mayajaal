"""Focused tests for fixed, context-bound LangChain investigation tools."""

import inspect
import unittest
from datetime import UTC, datetime
from typing import cast

from pydantic import BaseModel

import mayajaal.investigation.tools as investigation_tools
from mayajaal.investigation import (
    INVESTIGATION_TOOL_NAMES,
    EvidenceItem,
    EvidenceService,
    EvidenceSource,
    EvidenceType,
    InvestigationConfig,
    InvestigationRequest,
    InvestigationSubjectType,
    InvestigationToolBudgetExhausted,
    InvestigationToolContext,
    build_investigation_tools,
)
from mayajaal.policy import PolicyAction
from mayajaal.scoring import ScoreObservation


def cutoff() -> datetime:
    """Return a deterministic, timezone-aware fixture cutoff."""
    return datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def request() -> InvestigationRequest:
    """Return a fixed account-scoped investigation request."""
    return InvestigationRequest(
        decision_id="decision-fixture",
        policy_id="policy-fixture",
        probability_estimate_id="estimate-fixture",
        score_id="score-fixture",
        feature_vector_id="vector-fixture",
        subject_type=InvestigationSubjectType.ACCOUNT,
        subject_id="account-subject",
        cutoff_time=cutoff(),
        context_id="order-context",
        policy_action=PolicyAction.REVIEW,
        decision_is_stable_across_scenarios=True,
    )


def score() -> ScoreObservation:
    """Return the trusted score object held by the runtime context."""
    return ScoreObservation(
        score_id="score-fixture",
        base_model_id="base-model-fixture",
        subject_id="account-subject",
        scoring_cutoff=cutoff(),
        raw_model_score=0.25,
        feature_vector_id="vector-fixture",
    )


def item(
    evidence_type: EvidenceType = EvidenceType.RELATED_ACCOUNT_ACTIVITY,
) -> EvidenceItem:
    """Return one serializable, cutoff-safe evidence item."""
    return EvidenceItem.from_request(
        request(),
        evidence_id="evidence-fixture",
        evidence_type=evidence_type,
        source=EvidenceSource.EVENT_HISTORY,
        observed_at=cutoff(),
        subject_ids=("account-subject",),
        facts={"returned_event_count": 1, "truncated": False},
    )


class RecordingEvidenceService:
    """Minimal read-only service double which records adapter calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def get_risk_explanation(
        self, request: InvestigationRequest, score_observation: ScoreObservation
    ) -> tuple[EvidenceItem, ...]:
        self.calls.append(("risk_explanation", (request, score_observation)))
        return (item(EvidenceType.RISK_DRIVER),)

    def get_identity_neighborhood(
        self, request: InvestigationRequest
    ) -> tuple[EvidenceItem, ...]:
        self.calls.append(("identity_neighborhood", request))
        return (item(EvidenceType.IDENTITY_NEIGHBORHOOD),)

    def get_shared_identity_summary(
        self, request: InvestigationRequest
    ) -> tuple[EvidenceItem, ...]:
        self.calls.append(("shared_identity_summary", request))
        return (item(EvidenceType.SHARED_IP),)

    def get_related_activity(
        self, request: InvestigationRequest
    ) -> tuple[EvidenceItem, ...]:
        self.calls.append(("related_activity", request))
        return (item(),)

    def get_case_timeline(
        self, request: InvestigationRequest
    ) -> tuple[EvidenceItem, ...]:
        self.calls.append(("case_timeline", request))
        return (item(EvidenceType.TIMELINE_EVENT),)


class InvestigationToolTests(unittest.TestCase):
    """Verify the LangChain adapter remains a narrow service wrapper."""

    def context(
        self, *, max_tool_calls: int = 8
    ) -> tuple[InvestigationToolContext, RecordingEvidenceService]:
        """Build a tool context with a minimal read-only service double."""
        service = RecordingEvidenceService()
        return (
            InvestigationToolContext.create(
                request=request(),
                evidence_service=cast(EvidenceService, service),
                score_observation=score(),
                config=InvestigationConfig(max_tool_calls=max_tool_calls),
            ),
            service,
        )

    def test_fixed_allowlist_has_exactly_five_zero_argument_tools(self) -> None:
        context, _ = self.context()
        tools = build_investigation_tools(context)
        self.assertEqual(tuple(tool.name for tool in tools), INVESTIGATION_TOOL_NAMES)
        self.assertEqual(len(tools), 5)
        forbidden_parameters = {
            "account_id",
            "subject_id",
            "cutoff_time",
            "max_graph_hops",
            "max_graph_nodes",
            "max_graph_edges",
            "max_events_per_tool",
            "max_related_accounts",
            "cypher",
            "sql",
            "query",
        }
        for wrapped_tool in tools:
            schema_model = cast(type[BaseModel], wrapped_tool.tool_call_schema)  # pyright: ignore[reportUnknownMemberType]
            schema = schema_model.model_json_schema()
            self.assertEqual(schema.get("properties"), {})
            self.assertTrue(
                forbidden_parameters.isdisjoint(schema.get("properties", {}))
            )

    def test_each_wrapper_returns_existing_evidence_semantics(self) -> None:
        context, service = self.context()
        tools = {
            wrapped_tool.name: wrapped_tool
            for wrapped_tool in build_investigation_tools(context)
        }
        expected = item().model_dump(mode="json")
        actual = tools["related_activity"].invoke({})
        self.assertEqual(actual, [expected])
        self.assertEqual(service.calls, [("related_activity", context.request)])
        self.assertEqual(context.budget.used_tool_calls, 1)
        self.assertEqual(context.request.subject_id, "account-subject")
        self.assertEqual(context.request.cutoff_time, cutoff())

    def test_risk_wrapper_passes_only_the_trusted_score_observation(self) -> None:
        context, service = self.context()
        tools = {
            wrapped_tool.name: wrapped_tool
            for wrapped_tool in build_investigation_tools(context)
        }
        _ = tools["risk_explanation"].invoke({})
        self.assertEqual(
            service.calls,
            [("risk_explanation", (context.request, context.score_observation))],
        )

    def test_global_budget_is_shared_across_tools_and_fails_closed(self) -> None:
        context, service = self.context(max_tool_calls=2)
        tools = {
            wrapped_tool.name: wrapped_tool
            for wrapped_tool in build_investigation_tools(context)
        }
        _ = tools["identity_neighborhood"].invoke({})
        _ = tools["related_activity"].invoke({})
        with self.assertRaisesRegex(
            InvestigationToolBudgetExhausted, "budget exhausted"
        ):
            _ = tools["case_timeline"].invoke({})
        self.assertNotIn("case_timeline", [name for name, _ in service.calls])
        self.assertEqual(context.budget.used_tool_calls, 2)
        self.assertEqual(context.budget.remaining_tool_calls, 0)

    def test_tool_schema_cannot_override_trusted_request_or_service_limits(
        self,
    ) -> None:
        context, service = self.context()
        tools = {
            wrapped_tool.name: wrapped_tool
            for wrapped_tool in build_investigation_tools(context)
        }
        _ = tools["identity_neighborhood"].invoke(
            {
                "subject_id": "other-account",
                "cutoff_time": "2030-01-01T00:00:00+00:00",
                "max_graph_nodes": 100_000,
                "query": "MATCH (n) RETURN n",
            }
        )
        self.assertEqual(service.calls, [("identity_neighborhood", context.request)])

    def test_outputs_are_deterministic_and_preserve_bounded_metadata(self) -> None:
        context, _ = self.context()
        tools = {
            wrapped_tool.name: wrapped_tool
            for wrapped_tool in build_investigation_tools(context)
        }
        first = tools["case_timeline"].invoke({})
        second_context, _ = self.context()
        second_tools = {
            wrapped_tool.name: wrapped_tool
            for wrapped_tool in build_investigation_tools(second_context)
        }
        second = second_tools["case_timeline"].invoke({})
        self.assertEqual(first, second)
        self.assertFalse(first[0]["facts"]["truncated"])
        self.assertNotIn("synthetic", str(first).casefold())

    def test_adapter_exposes_no_openai_or_arbitrary_execution_interfaces(self) -> None:
        source = inspect.getsource(investigation_tools).casefold()
        for forbidden in ("openai", "cypher", "sql", "shell", "requests"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    _ = unittest.main()
