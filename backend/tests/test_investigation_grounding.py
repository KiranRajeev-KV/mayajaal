"""Focused integrity tests for grounded investigation reports and artifacts."""

import inspect
import json
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import JsonValue

import mayajaal.investigation.artifacts as investigation_artifacts
from mayajaal.investigation import (
    AGENT_PROMPT_CONTRACT_VERSION,
    DIAGNOSTIC_PROVENANCE_CONTRACT_VERSION,
    EVIDENCE_CONTRACT_VERSION,
    INVESTIGATION_PROVENANCE_CONTRACT_VERSION,
    EvidenceFinding,
    EvidenceItem,
    EvidenceLedger,
    EvidenceLedgerSnapshot,
    EvidenceSource,
    EvidenceType,
    GroundingFailureCode,
    GroundingFailureDiagnostic,
    InvestigationConfig,
    InvestigationExecution,
    InvestigationGroundingError,
    InvestigationPattern,
    InvestigationReport,
    InvestigationRequest,
    InvestigationStatus,
    InvestigationSubjectType,
    InvestigationToolTrace,
    ReasoningEffort,
    RelatedEntity,
    diagnostic_id,
    evidence_id,
    investigation_id,
    load_investigation_artifacts,
    report_id,
    save_investigation_artifacts,
    validate_report_grounding,
)
from mayajaal.policy import PolicyAction


def cutoff() -> datetime:
    """Return a deterministic point-in-time boundary for all fixtures."""
    return datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def request(*, decision_id: str = "decision-fixture") -> InvestigationRequest:
    """Build a minimal trusted account-scored investigation request."""
    return InvestigationRequest(
        decision_id=decision_id,
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


def evidence(
    request_value: InvestigationRequest,
    *,
    evidence_type: EvidenceType,
    subject_ids: tuple[str, ...] | None = None,
    observed_at: datetime | None = None,
) -> EvidenceItem:
    """Construct one real semantic evidence item without label fields."""
    factual_subjects = subject_ids or (request_value.subject_id,)
    observed = observed_at or request_value.cutoff_time
    facts: dict[str, JsonValue] = {
        "fact": evidence_type.value,
        "truncated": False,
    }
    return EvidenceItem.from_request(
        request_value,
        evidence_id=evidence_id(
            request_value,
            evidence_type=evidence_type,
            source=(
                EvidenceSource.CASE_TIMELINE
                if evidence_type is EvidenceType.TIMELINE_EVENT
                else EvidenceSource.IDENTITY_SUMMARY
            ),
            observed_at=observed,
            subject_ids=factual_subjects,
            facts=facts,
        ),
        evidence_type=evidence_type,
        source=(
            EvidenceSource.CASE_TIMELINE
            if evidence_type is EvidenceType.TIMELINE_EVENT
            else EvidenceSource.IDENTITY_SUMMARY
        ),
        observed_at=observed,
        subject_ids=factual_subjects,
        facts=facts,
    )


class InvestigationGroundingTests(unittest.TestCase):
    """Ensure only returned, request-bound evidence can ground a report."""

    def ledger(self) -> tuple[EvidenceLedger, EvidenceItem, EvidenceItem]:
        case = request()
        shared = evidence(case, evidence_type=EvidenceType.SHARED_DEVICE)
        timeline = evidence(
            case,
            evidence_type=EvidenceType.TIMELINE_EVENT,
            observed_at=cutoff() - timedelta(minutes=1),
        )
        ledger = EvidenceLedger(case)
        ledger.record("shared_identity_summary", (shared,))
        ledger.record("case_timeline", (timeline,))
        return ledger, shared, timeline

    def report(
        self, shared: EvidenceItem, timeline: EvidenceItem
    ) -> InvestigationReport:
        case = request()
        return InvestigationReport(
            request=case,
            policy_action=case.policy_action,
            status=InvestigationStatus.COMPLETED,
            pattern=InvestigationPattern.INCONCLUSIVE,
            key_findings=(
                EvidenceFinding(
                    claim="The account shares one device.",
                    evidence_ids=(shared.evidence_id,),
                ),
            ),
            counterevidence=(
                EvidenceFinding(
                    claim="Shared identities can be benign.",
                    evidence_ids=(shared.evidence_id,),
                ),
            ),
            timeline_evidence_ids=(timeline.evidence_id,),
            related_entities=(
                RelatedEntity(
                    entity_id="device-1",
                    entity_type="DEVICE",
                    evidence_ids=(shared.evidence_id,),
                ),
            ),
            evidence_ids=(shared.evidence_id, timeline.evidence_id),
            summary="Evidence supports no conclusive pattern.",
        )

    def test_ledger_tracks_only_tool_returned_evidence_and_trace_is_deterministic(
        self,
    ) -> None:
        ledger, shared, timeline = self.ledger()
        snapshot = ledger.snapshot()
        self.assertEqual(snapshot.evidence, (shared, timeline))
        self.assertEqual(
            tuple(trace.call_index for trace in snapshot.tool_trace), (1, 2)
        )
        self.assertEqual(
            tuple(trace.tool_name for trace in snapshot.tool_trace),
            ("shared_identity_summary", "case_timeline"),
        )
        self.assertEqual(
            snapshot.tool_trace[0].returned_evidence_ids, (shared.evidence_id,)
        )

    def test_identical_duplicate_is_deduplicated_but_conflict_is_rejected(self) -> None:
        ledger, shared, _ = self.ledger()
        ledger.record("shared_identity_summary", (shared,))
        self.assertEqual(len(ledger.snapshot().evidence), 2)
        conflicting = shared.model_copy(update={"facts": {"fact": "different"}})
        with self.assertRaises(ValueError):
            ledger.record("shared_identity_summary", (conflicting,))

    def test_tool_trace_is_restricted_to_the_fixed_allowlist(self) -> None:
        ledger, shared, _ = self.ledger()
        with self.assertRaisesRegex(ValueError, "fixed allowlist"):
            ledger.record("custom_query", (shared,))

        forged_trace = InvestigationToolTrace.model_construct(
            call_index=1,
            tool_name="custom_query",
            returned_evidence_ids=(shared.evidence_id,),
        )
        forged_snapshot = EvidenceLedgerSnapshot(
            evidence=(shared,), tool_trace=(forged_trace,)
        )
        with self.assertRaisesRegex(ValueError, "fixed allowlist"):
            _ = EvidenceLedger.from_snapshot(request(), forged_snapshot)

        valid_report = InvestigationReport(
            request=request(),
            policy_action=PolicyAction.REVIEW,
            status=InvestigationStatus.INSUFFICIENT_EVIDENCE,
            summary="No factual finding is asserted.",
        )
        forged_execution = InvestigationExecution(
            report=valid_report,
            snapshot=forged_snapshot,
            agent_model_id="fixture-model",
            config=InvestigationConfig(),
        )
        with (
            TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "fixed allowlist"),
        ):
            _ = save_investigation_artifacts(Path(directory), forged_execution)

    def test_grounding_rejects_invented_finding_counterevidence_and_timeline_ids(
        self,
    ) -> None:
        ledger, shared, timeline = self.ledger()
        valid = self.report(shared, timeline)
        self.assertEqual(
            validate_report_grounding(valid, request(), ledger.snapshot()), valid
        )
        for field in ("key_findings", "counterevidence"):
            altered = valid.model_copy(
                update={
                    field: (
                        EvidenceFinding(
                            claim="Invented reference.", evidence_ids=("invented",)
                        ),
                    ),
                    "evidence_ids": ("invented",),
                }
            )
            with self.assertRaises(InvestigationGroundingError):
                validate_report_grounding(altered, request(), ledger.snapshot())
        timeline_altered = valid.model_copy(
            update={"timeline_evidence_ids": (shared.evidence_id,)}
        )
        with self.assertRaisesRegex(InvestigationGroundingError, "TIMELINE_EVENT"):
            validate_report_grounding(timeline_altered, request(), ledger.snapshot())

    def test_grounding_failure_codes_distinguish_reference_failures(self) -> None:
        ledger, shared, timeline = self.ledger()
        valid = self.report(shared, timeline)

        with self.assertRaises(InvestigationGroundingError) as unknown_error:
            ledger.resolve_alias("E999")
        self.assertIs(
            unknown_error.exception.code,
            GroundingFailureCode.UNKNOWN_EVIDENCE_REFERENCE,
        )
        with self.assertRaises(InvestigationGroundingError) as malformed_error:
            ledger.resolve_alias("not-an-alias")
        self.assertIs(
            malformed_error.exception.code,
            GroundingFailureCode.MALFORMED_EVIDENCE_REFERENCE,
        )

        wrong_timeline = valid.model_copy(
            update={"timeline_evidence_ids": (shared.evidence_id,)}
        )
        with self.assertRaises(InvestigationGroundingError) as timeline_error:
            validate_report_grounding(wrong_timeline, request(), ledger.snapshot())
        self.assertIs(
            timeline_error.exception.code,
            GroundingFailureCode.WRONG_TIMELINE_EVIDENCE_TYPE,
        )

        ungrounded_related = valid.model_copy(
            update={
                "related_entities": (
                    RelatedEntity(
                        entity_id="device-1",
                        entity_type="DEVICE",
                        evidence_ids=("invented",),
                    ),
                )
            }
        )
        with self.assertRaises(InvestigationGroundingError) as related_error:
            validate_report_grounding(ungrounded_related, request(), ledger.snapshot())
        self.assertIs(
            related_error.exception.code,
            GroundingFailureCode.UNGROUNDED_RELATED_ENTITY,
        )

        mismatched_request = valid.model_copy(
            update={"request": request(decision_id="other")}
        )
        with self.assertRaises(InvestigationGroundingError) as request_error:
            validate_report_grounding(mismatched_request, request(), ledger.snapshot())
        self.assertIs(
            request_error.exception.code,
            GroundingFailureCode.GROUNDING_VALIDATION_FAILED,
        )

    def test_related_entities_and_evidence_from_another_request_are_rejected(
        self,
    ) -> None:
        ledger, shared, timeline = self.ledger()
        valid = self.report(shared, timeline)
        ungrounded = valid.model_copy(
            update={
                "related_entities": (
                    RelatedEntity(
                        entity_id="device-1",
                        entity_type="DEVICE",
                        evidence_ids=("invented",),
                    ),
                )
            }
        )
        with self.assertRaises(InvestigationGroundingError):
            validate_report_grounding(ungrounded, request(), ledger.snapshot())
        other = evidence(
            request(decision_id="other-decision"),
            evidence_type=EvidenceType.SHARED_DEVICE,
        )
        with self.assertRaisesRegex(ValueError, "evidence cutoff_time|evidence_id"):
            ledger.record("shared_identity_summary", (other,))
        later_request = request().model_copy(
            update={"cutoff_time": cutoff() + timedelta(minutes=1)}
        )
        later = evidence(later_request, evidence_type=EvidenceType.SHARED_DEVICE)
        with self.assertRaisesRegex(ValueError, "cutoff_time"):
            ledger.record("shared_identity_summary", (later,))

    def test_provenance_is_run_deterministic_while_report_identity_binds_prose(
        self,
    ) -> None:
        self.assertEqual(INVESTIGATION_PROVENANCE_CONTRACT_VERSION, 2)
        self.assertEqual(AGENT_PROMPT_CONTRACT_VERSION, 3)
        self.assertEqual(EVIDENCE_CONTRACT_VERSION, 2)
        ledger, shared, timeline = self.ledger()
        snapshot = ledger.snapshot()
        config = InvestigationConfig()
        first = investigation_id(
            request=request(),
            config=config,
            agent_model_id="fixture-model",
            snapshot=snapshot,
        )
        self.assertEqual(
            first,
            investigation_id(
                request=request(),
                config=config,
                agent_model_id="fixture-model",
                snapshot=snapshot,
            ),
        )
        report = self.report(shared, timeline)
        altered_report = report.model_copy(update={"summary": "Different wording."})
        self.assertNotEqual(report_id(first, report), report_id(first, altered_report))
        changed_config = InvestigationConfig(max_tool_calls=9)
        self.assertNotEqual(
            first,
            investigation_id(
                request=request(),
                config=changed_config,
                agent_model_id="fixture-model",
                snapshot=snapshot,
            ),
        )
        changed_reasoning_effort = InvestigationConfig(
            reasoning_effort=ReasoningEffort.HIGH
        )
        self.assertNotEqual(
            first,
            investigation_id(
                request=request(),
                config=changed_reasoning_effort,
                agent_model_id="fixture-model",
                snapshot=snapshot,
            ),
        )
        self.assertNotEqual(
            first,
            investigation_id(
                request=request(),
                config=config,
                agent_model_id="different-model",
                snapshot=snapshot,
            ),
        )
        changed_evidence = evidence(
            request(),
            evidence_type=EvidenceType.SHARED_PAYMENT_IDENTITY,
        )
        changed_ledger = EvidenceLedger(request())
        changed_ledger.record("shared_identity_summary", (changed_evidence,))
        changed_ledger.record("case_timeline", (timeline,))
        self.assertNotEqual(
            first,
            investigation_id(
                request=request(),
                config=config,
                agent_model_id="fixture-model",
                snapshot=changed_ledger.snapshot(),
            ),
        )

    def test_context_metrics_do_not_change_investigation_or_report_identity(
        self,
    ) -> None:
        ledger, shared, timeline = self.ledger()
        snapshot = ledger.snapshot()
        trace = snapshot.tool_trace[0]
        assert trace.model_facing_metrics is not None
        changed_metrics = trace.model_facing_metrics.model_copy(
            update={"model_facing_serialized_bytes": 999_999}
        )
        measured_snapshot = EvidenceLedgerSnapshot(
            evidence=snapshot.evidence,
            tool_trace=(
                trace.model_copy(update={"model_facing_metrics": changed_metrics}),
                snapshot.tool_trace[1],
            ),
        )
        report = self.report(shared, timeline)
        config = InvestigationConfig()
        original_id = investigation_id(
            request=request(),
            config=config,
            agent_model_id="fixture-model",
            snapshot=snapshot,
        )
        measured_id = investigation_id(
            request=request(),
            config=config,
            agent_model_id="fixture-model",
            snapshot=measured_snapshot,
        )

        self.assertEqual(measured_id, original_id)
        self.assertEqual(report_id(measured_id, report), report_id(original_id, report))

    def test_artifacts_round_trip_and_reject_evidence_report_and_provenance_tampering(
        self,
    ) -> None:
        ledger, shared, timeline = self.ledger()
        report = self.report(shared, timeline)
        execution = InvestigationExecution(
            report=report,
            snapshot=ledger.snapshot(),
            agent_model_id="fixture-model",
            config=InvestigationConfig(),
        )
        with TemporaryDirectory() as directory:
            output = Path(directory)
            _ = save_investigation_artifacts(output, execution)
            loaded = load_investigation_artifacts(
                output,
                request(),
                InvestigationConfig(),
                agent_model_id="fixture-model",
            )
            self.assertEqual(loaded, execution)
            with self.assertRaisesRegex(ValueError, "trusted expected configuration"):
                load_investigation_artifacts(
                    output,
                    request(),
                    InvestigationConfig(max_tool_calls=9),
                    agent_model_id="fixture-model",
                )

            with self.assertRaisesRegex(ValueError, "trusted request"):
                load_investigation_artifacts(
                    output,
                    request(decision_id="another-decision"),
                    InvestigationConfig(),
                    agent_model_id="fixture-model",
                )
            evidence_path = output / "evidence.json"
            document = json.loads(evidence_path.read_text(encoding="utf-8"))
            document["evidence"][0]["facts"]["fact"] = "tampered"
            evidence_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_investigation_artifacts(
                    output,
                    request(),
                    InvestigationConfig(),
                    agent_model_id="fixture-model",
                )

            _ = save_investigation_artifacts(output, execution)
            document = json.loads(evidence_path.read_text(encoding="utf-8"))
            document["investigation_provenance_contract_version"] = 1
            evidence_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "investigation lineage"):
                load_investigation_artifacts(
                    output,
                    request(),
                    InvestigationConfig(),
                    agent_model_id="fixture-model",
                )

            _ = save_investigation_artifacts(output, execution)
            trace_document = json.loads(evidence_path.read_text(encoding="utf-8"))
            trace_document["tool_trace"][0]["tool_name"] = "custom_query"
            evidence_path.write_text(json.dumps(trace_document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tool trace"):
                load_investigation_artifacts(
                    output,
                    request(),
                    InvestigationConfig(),
                    agent_model_id="fixture-model",
                )

            _ = save_investigation_artifacts(output, execution)
            report_path = output / "investigation_report.json"
            document = json.loads(report_path.read_text(encoding="utf-8"))
            document["report"]["summary"] = "tampered"
            report_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_investigation_artifacts(
                    output,
                    request(),
                    InvestigationConfig(),
                    agent_model_id="fixture-model",
                )
            _ = save_investigation_artifacts(output, execution)
            for path in output.glob("*.json"):
                self.assertNotIn("OPENAI_API_KEY", path.read_text(encoding="utf-8"))
            provenance_path = output / "investigation_provenance.json"
            document = json.loads(provenance_path.read_text(encoding="utf-8"))
            document["provenance"]["agent_model_id"] = "tampered"
            provenance_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_investigation_artifacts(
                    output,
                    request(),
                    InvestigationConfig(),
                    agent_model_id="fixture-model",
                )

    def test_save_rejects_context_metric_tampering_before_writing(self) -> None:
        ledger, shared, timeline = self.ledger()
        execution = InvestigationExecution(
            report=self.report(shared, timeline),
            snapshot=ledger.snapshot(),
            agent_model_id="fixture-model",
            config=InvestigationConfig(),
        )
        assert execution.model_facing_context_metrics is not None
        assert execution.model_facing_tool_call_metrics
        tampered_total = replace(
            execution,
            model_facing_context_metrics=execution.model_facing_context_metrics.model_copy(
                update={"model_facing_serialized_bytes": 999}
            ),
        )
        tampered_calls = replace(
            execution,
            model_facing_tool_call_metrics=(
                execution.model_facing_tool_call_metrics[0].model_copy(
                    update={"model_facing_serialized_chars": 999}
                ),
                *execution.model_facing_tool_call_metrics[1:],
            ),
        )
        with TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist"
            with self.assertRaisesRegex(ValueError, "total context metrics"):
                save_investigation_artifacts(output, tampered_total)
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError, "tool-call metrics"):
                save_investigation_artifacts(output, tampered_calls)
            self.assertFalse(output.exists())

    def test_rejected_candidate_diagnostic_is_non_authoritative_and_round_trips(
        self,
    ) -> None:
        report = InvestigationReport(
            request=request(),
            policy_action=PolicyAction.REVIEW,
            status=InvestigationStatus.FAILED,
            limitations=("report grounding validation failed",),
        )
        diagnostic = GroundingFailureDiagnostic(
            code=GroundingFailureCode.UNKNOWN_EVIDENCE_REFERENCE,
            detail="evidence reference is not admitted to this investigation",
            rejected_candidate={"status": "COMPLETED", "evidence_refs": ["E999"]},
        )
        execution = InvestigationExecution(
            report=report,
            snapshot=EvidenceLedgerSnapshot(evidence=(), tool_trace=()),
            agent_model_id="fixture-model",
            config=InvestigationConfig(),
            grounding_failure=diagnostic,
        )
        with TemporaryDirectory() as directory:
            output = Path(directory)
            _ = save_investigation_artifacts(output, execution)
            loaded = load_investigation_artifacts(
                output,
                request(),
                InvestigationConfig(),
                agent_model_id="fixture-model",
            )
            self.assertEqual(loaded, execution)
            provenance = json.loads(
                (output / "investigation_provenance.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                provenance["grounding_failure"]["code"], "UNKNOWN_EVIDENCE_REFERENCE"
            )
            self.assertEqual(
                provenance["diagnostic_provenance_contract_version"],
                DIAGNOSTIC_PROVENANCE_CONTRACT_VERSION,
            )
            self.assertEqual(
                provenance["diagnostic_id"],
                diagnostic_id(provenance["provenance"]["investigation_id"], diagnostic),
            )
            self.assertNotIn("grounding_failure", provenance["provenance"])

    def test_grounding_diagnostic_is_deterministic_tamper_evident_and_not_report_identity(
        self,
    ) -> None:
        report = InvestigationReport(
            request=request(),
            policy_action=PolicyAction.REVIEW,
            status=InvestigationStatus.FAILED,
            limitations=("report grounding validation failed",),
        )
        diagnostic = GroundingFailureDiagnostic(
            code=GroundingFailureCode.UNKNOWN_EVIDENCE_REFERENCE,
            detail="evidence reference is not admitted to this investigation",
            rejected_candidate={"status": "COMPLETED", "evidence_refs": ["E999"]},
        )
        execution = InvestigationExecution(
            report=report,
            snapshot=EvidenceLedgerSnapshot(evidence=(), tool_trace=()),
            agent_model_id="fixture-model",
            config=InvestigationConfig(),
            grounding_failure=diagnostic,
        )
        without_diagnostic = InvestigationExecution(
            report=report,
            snapshot=execution.snapshot,
            agent_model_id=execution.agent_model_id,
            config=execution.config,
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            diagnostic_output = root / "with-diagnostic"
            baseline_output = root / "without-diagnostic"
            _ = save_investigation_artifacts(diagnostic_output, execution)
            _ = save_investigation_artifacts(baseline_output, without_diagnostic)
            diagnostic_document = json.loads(
                (diagnostic_output / "investigation_provenance.json").read_text(
                    encoding="utf-8"
                )
            )
            baseline_report = json.loads(
                (baseline_output / "investigation_report.json").read_text(
                    encoding="utf-8"
                )
            )
            diagnostic_report = json.loads(
                (diagnostic_output / "investigation_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                diagnostic_report["report_id"], baseline_report["report_id"]
            )
            self.assertNotIn(
                "diagnostic_id",
                json.loads(
                    (baseline_output / "investigation_provenance.json").read_text(
                        encoding="utf-8"
                    )
                ),
            )
            original_id = diagnostic_document["diagnostic_id"]
            self.assertEqual(
                original_id,
                diagnostic_id(
                    diagnostic_document["provenance"]["investigation_id"], diagnostic
                ),
            )
            for field, value in (
                ("code", "MALFORMED_EVIDENCE_REFERENCE"),
                ("detail", "tampered detail"),
                ("rejected_candidate", {"status": "tampered"}),
            ):
                with self.subTest(field=field):
                    document = json.loads(json.dumps(diagnostic_document))
                    document["grounding_failure"][field] = value
                    (diagnostic_output / "investigation_provenance.json").write_text(
                        json.dumps(document), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(ValueError, "diagnostic"):
                        load_investigation_artifacts(
                            diagnostic_output,
                            request(),
                            InvestigationConfig(),
                            agent_model_id="fixture-model",
                        )
            document = json.loads(json.dumps(diagnostic_document))
            document["diagnostic_id"] = "tampered"
            (diagnostic_output / "investigation_provenance.json").write_text(
                json.dumps(document), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "diagnostic"):
                load_investigation_artifacts(
                    diagnostic_output,
                    request(),
                    InvestigationConfig(),
                    agent_model_id="fixture-model",
                )
            document = json.loads(json.dumps(diagnostic_document))
            document.pop("grounding_failure")
            (diagnostic_output / "investigation_provenance.json").write_text(
                json.dumps(document), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "diagnostic_id"):
                load_investigation_artifacts(
                    diagnostic_output,
                    request(),
                    InvestigationConfig(),
                    agent_model_id="fixture-model",
                )

    def test_only_failed_reports_may_persist_a_grounding_diagnostic(self) -> None:
        valid_report = InvestigationReport(
            request=request(),
            policy_action=PolicyAction.REVIEW,
            status=InvestigationStatus.COMPLETED,
        )
        diagnostic = GroundingFailureDiagnostic(
            code=GroundingFailureCode.INVALID_STRUCTURED_OUTPUT,
            detail="invalid structured output",
        )
        valid_execution = InvestigationExecution(
            report=valid_report,
            snapshot=EvidenceLedgerSnapshot(evidence=(), tool_trace=()),
            agent_model_id="fixture-model",
            config=InvestigationConfig(),
        )
        with TemporaryDirectory() as directory:
            output = Path(directory)
            _ = save_investigation_artifacts(output, valid_execution)
            document = json.loads(
                (output / "investigation_provenance.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("grounding_failure", document)
            self.assertNotIn("diagnostic_id", document)
            self.assertEqual(
                load_investigation_artifacts(
                    output,
                    request(),
                    InvestigationConfig(),
                    agent_model_id="fixture-model",
                ),
                valid_execution,
            )
            with self.assertRaisesRegex(ValueError, "FAILED"):
                _ = save_investigation_artifacts(
                    output,
                    InvestigationExecution(
                        report=valid_report,
                        snapshot=valid_execution.snapshot,
                        agent_model_id=valid_execution.agent_model_id,
                        config=valid_execution.config,
                        grounding_failure=diagnostic,
                    ),
                )

    def test_artifact_verification_has_no_agent_or_model_call_boundary(self) -> None:
        source = inspect.getsource(investigation_artifacts)
        self.assertNotIn("create_agent", source)
        self.assertNotIn("ChatOpenAI", source)

    def test_budget_report_without_claims_remains_grounded(self) -> None:
        case = request()
        report = InvestigationReport(
            request=case,
            policy_action=case.policy_action,
            status=InvestigationStatus.BUDGET_EXHAUSTED,
            limitations=("tool-call budget exhausted",),
        )
        self.assertEqual(
            validate_report_grounding(
                report, case, EvidenceLedgerSnapshot(evidence=(), tool_trace=())
            ),
            report,
        )

    def test_operational_failure_status_cannot_retain_factual_claims(self) -> None:
        _, shared, timeline = self.ledger()
        report = self.report(shared, timeline)
        document = report.model_dump(mode="json")
        document["status"] = InvestigationStatus.FAILED.value
        with self.assertRaisesRegex(ValueError, "cannot retain claims"):
            _ = InvestigationReport.model_validate(document)


if __name__ == "__main__":
    _ = unittest.main()
