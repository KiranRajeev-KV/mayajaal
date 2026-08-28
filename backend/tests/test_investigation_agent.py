"""Focused tests for the bounded LangChain/OpenAI investigation boundary."""

import inspect
import os
import unittest
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast, override
from unittest.mock import patch

from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool

import mayajaal.investigation.agent as investigation_agent
from mayajaal.investigation import (
    INVESTIGATION_TOOL_NAMES,
    EvidenceItem,
    EvidenceSource,
    EvidenceType,
    InvestigationAgentOutput,
    InvestigationAgentService,
    InvestigationConfig,
    InvestigationPattern,
    InvestigationRequest,
    InvestigationStatus,
    InvestigationSubjectType,
)
from mayajaal.policy import PolicyAction
from mayajaal.scoring import ScoreObservation


def cutoff() -> datetime:
    """Return a deterministic cutoff for agent-bound fixtures."""
    return datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def request() -> InvestigationRequest:
    """Return an account-scoped request with immutable policy context."""
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
    """Return the verified score held by the tool context."""
    return ScoreObservation(
        score_id="score-fixture",
        base_model_id="base-model-fixture",
        subject_id="account-subject",
        scoring_cutoff=cutoff(),
        raw_model_score=0.25,
        feature_vector_id="vector-fixture",
    )


class FakeChatModel(BaseChatModel):
    """A no-network chat model accepted by the real service constructor."""

    @property
    @override
    def _llm_type(self) -> str:
        return "fake-investigation-model"

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])


class RecordingEvidenceService:
    """Read-only service double used by context-bound tool calls."""

    def __init__(self, config: InvestigationConfig) -> None:
        self.config = config
        self.calls: list[str] = []
        self.last_request: InvestigationRequest | None = None

    def _item(self, request: InvestigationRequest) -> EvidenceItem:
        self.last_request = request
        return EvidenceItem.from_request(
            request,
            evidence_id="evidence-fixture",
            evidence_type=EvidenceType.RELATED_ACCOUNT_ACTIVITY,
            source=EvidenceSource.EVENT_HISTORY,
            observed_at=request.cutoff_time,
            subject_ids=(request.subject_id,),
            facts={"returned_event_count": 1, "truncated": False},
        )

    def get_risk_explanation(
        self, request: InvestigationRequest, score_observation: ScoreObservation
    ) -> tuple[EvidenceItem, ...]:
        del score_observation
        self.calls.append("risk_explanation")
        return (self._item(request),)

    def get_identity_neighborhood(
        self, request: InvestigationRequest
    ) -> tuple[EvidenceItem, ...]:
        self.calls.append("identity_neighborhood")
        return (self._item(request),)

    def get_shared_identity_summary(
        self, request: InvestigationRequest
    ) -> tuple[EvidenceItem, ...]:
        self.calls.append("shared_identity_summary")
        return (self._item(request),)

    def get_related_activity(
        self, request: InvestigationRequest
    ) -> tuple[EvidenceItem, ...]:
        self.calls.append("related_activity")
        return (self._item(request),)

    def get_case_timeline(
        self, request: InvestigationRequest
    ) -> tuple[EvidenceItem, ...]:
        self.calls.append("case_timeline")
        return (self._item(request),)


@dataclass
class FakeAgent:
    """Small agent double for service tests; it never calls a provider."""

    state: dict[str, object] | None = None
    error: Exception | None = None
    calls: list[dict[str, object]] = field(default_factory=list)

    def invoke(self, value: dict[str, object]) -> dict[str, object]:
        self.calls.append(value)
        if self.error is not None:
            raise self.error
        if self.state is None:
            raise AssertionError("fake agent requires a structured response state")
        return self.state


class InvestigationAgentTests(unittest.TestCase):
    """Verify the agent keeps trusted fields and hard budgets outside the model."""

    def service_and_evidence(
        self, *, max_tool_calls: int = 8, max_iterations: int = 4
    ) -> tuple[InvestigationAgentService, RecordingEvidenceService]:
        config = InvestigationConfig(
            max_tool_calls=max_tool_calls,
            max_iterations=max_iterations,
        )
        return (
            InvestigationAgentService(config=config, model=FakeChatModel()),
            RecordingEvidenceService(config),
        )

    def test_agent_receives_exact_fixed_tools_and_structured_output(self) -> None:
        service, evidence_service = self.service_and_evidence()
        output = InvestigationAgentOutput(
            status=InvestigationStatus.COMPLETED,
            pattern=InvestigationPattern.INCONCLUSIVE,
            summary="No factual abuse conclusion from the bounded evidence.",
        )
        fake_agent = FakeAgent(
            state={"structured_response": output, "run_model_call_count": 2}
        )
        with patch.object(
            investigation_agent, "create_agent", return_value=fake_agent
        ) as factory:
            report = service.run(
                request=request(),
                evidence_service=evidence_service,  # type: ignore[arg-type]
                score_observation=score(),
            )

        kwargs = factory.call_args.kwargs
        self.assertEqual(
            tuple(tool.name for tool in kwargs["tools"]), INVESTIGATION_TOOL_NAMES
        )
        self.assertEqual(len(kwargs["tools"]), 5)
        self.assertIs(kwargs["model"], service._model)  # pyright: ignore[reportPrivateUsage]
        self.assertIs(kwargs["response_format"], InvestigationAgentOutput)
        middleware = kwargs["middleware"]
        self.assertEqual(len(middleware), 1)
        self.assertIsInstance(middleware[0], ModelCallLimitMiddleware)
        self.assertEqual(middleware[0].run_limit, 4)
        self.assertEqual(middleware[0].exit_behavior, "error")
        self.assertEqual(report.request, request())
        self.assertIs(report.policy_action, PolicyAction.REVIEW)
        self.assertEqual(report.usage.iterations, 2)
        self.assertEqual(len(fake_agent.calls), 1)
        task = str(fake_agent.calls[0]["messages"])
        self.assertIn("account-subject", task)
        self.assertIn("immutable_policy_action: REVIEW", task)
        self.assertNotIn("synthetic", task.casefold())

    def test_run_accepts_no_arbitrary_prompt_or_trusted_field_override(self) -> None:
        service, evidence_service = self.service_and_evidence()
        fake_agent = FakeAgent(
            state={
                "structured_response": {
                    "status": "INSUFFICIENT_EVIDENCE",
                    "pattern": "INCONCLUSIVE",
                    "summary": "Evidence is limited.",
                }
            }
        )
        with patch.object(investigation_agent, "create_agent", return_value=fake_agent):
            report = service.run(
                request=request(),
                evidence_service=evidence_service,  # type: ignore[arg-type]
                score_observation=score(),
            )
        self.assertIs(report.policy_action, PolicyAction.REVIEW)
        self.assertEqual(report.request.subject_id, "account-subject")
        self.assertEqual(
            tuple(inspect.signature(service.run).parameters),
            ("request", "evidence_service", "score_observation"),
        )
        with self.assertRaises(ValueError):
            _ = investigation_agent.InvestigationAgentOutput.model_validate(
                {"status": "COMPLETED", "policy_action": "BLOCK"}
            )

    def test_model_call_limit_returns_typed_budget_exhausted_report(self) -> None:
        service, evidence_service = self.service_and_evidence(max_iterations=2)
        fake_agent = FakeAgent(
            error=ModelCallLimitExceededError(
                thread_count=0,
                run_count=2,
                thread_limit=None,
                run_limit=2,
            )
        )
        with patch.object(investigation_agent, "create_agent", return_value=fake_agent):
            report = service.run(
                request=request(),
                evidence_service=evidence_service,  # type: ignore[arg-type]
                score_observation=score(),
            )
        self.assertIs(report.status, InvestigationStatus.BUDGET_EXHAUSTED)
        self.assertEqual(report.usage.iterations, 2)
        self.assertEqual(report.limitations, ("model-call budget exhausted",))

    def test_shared_tool_budget_remains_enforced_and_fails_closed(self) -> None:
        service, evidence_service = self.service_and_evidence(max_tool_calls=1)

        class ToolCallingAgent(FakeAgent):
            def invoke(self, value: dict[str, object]) -> dict[str, object]:
                del value
                tools = cast_tools(factory.call_args.kwargs["tools"])
                _ = tools[0].invoke({})
                _ = tools[1].invoke({})
                raise AssertionError("second tool call should have failed")

        fake_agent = ToolCallingAgent()
        with patch.object(
            investigation_agent, "create_agent", return_value=fake_agent
        ) as factory:
            report = service.run(
                request=request(),
                evidence_service=evidence_service,  # type: ignore[arg-type]
                score_observation=score(),
            )
        self.assertIs(report.status, InvestigationStatus.BUDGET_EXHAUSTED)
        self.assertEqual(report.usage.tool_calls, 1)
        self.assertEqual(report.limitations, ("tool-call budget exhausted",))
        self.assertEqual(evidence_service.calls, ["risk_explanation"])
        last_request = evidence_service.last_request
        if last_request is None:
            raise AssertionError("tool call did not receive its trusted request")
        self.assertEqual(last_request.cutoff_time, cutoff())

    def test_agent_rejects_config_drift_before_building_tools(self) -> None:
        service, _ = self.service_and_evidence()
        other_service = RecordingEvidenceService(InvestigationConfig())
        with self.assertRaisesRegex(ValueError, "share one config instance"):
            _ = service.run(
                request=request(),
                evidence_service=other_service,  # type: ignore[arg-type]
                score_observation=score(),
            )

    def test_fake_model_needs_no_api_key_and_openai_requires_only_environment_key(
        self,
    ) -> None:
        with patch.dict(os.environ, {}, clear=True):
            _ = InvestigationAgentService(
                config=InvestigationConfig(), model=FakeChatModel()
            )
            with self.assertRaisesRegex(ValueError, "model_name"):
                _ = InvestigationAgentService(config=InvestigationConfig())
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                _ = InvestigationAgentService(
                    config=InvestigationConfig(model_name="approved-model")
                )

    def test_agent_module_has_no_unsafe_capability_or_synthetic_label_access(
        self,
    ) -> None:
        imported_modules = inspect.getmembers(investigation_agent, inspect.ismodule)
        imported_module_names = {module.__name__ for _, module in imported_modules}
        for forbidden in ("requests", "subprocess", "sqlite3", "neo4j"):
            self.assertNotIn(forbidden, imported_module_names)
        self.assertNotIn("SyntheticEventLabels", inspect.getsource(investigation_agent))


def cast_tools(value: object) -> tuple[BaseTool, ...]:
    """Contain the untyped mock call boundary used by the budget test."""
    if not isinstance(value, tuple):
        raise AssertionError("expected fixed tool tuple")
    result: list[BaseTool] = []
    for item in cast(tuple[object, ...], value):
        if not isinstance(item, BaseTool):
            raise AssertionError("expected only LangChain tools")
        result.append(item)
    return tuple(result)


if __name__ == "__main__":
    _ = unittest.main()
