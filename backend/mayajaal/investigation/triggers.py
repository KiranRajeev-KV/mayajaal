"""Pure deterministic investigation trigger rules."""

from mayajaal.policy import PolicyAction, PolicyDecision

from .models import (
    InvestigationConfig,
    InvestigationTrigger,
    InvestigationTriggerReason,
)


def should_investigate(
    policy_decision: PolicyDecision,
    config: InvestigationConfig,
) -> InvestigationTrigger:
    """Apply configured read-only trigger rules without changing policy action."""
    if policy_decision.chosen_action is PolicyAction.REVIEW:
        return _configured_trigger(
            config.triggers.investigate_review,
            InvestigationTriggerReason.REVIEW_ACTION,
        )
    if policy_decision.chosen_action is PolicyAction.BLOCK:
        return _configured_trigger(
            config.triggers.investigate_block,
            InvestigationTriggerReason.BLOCK_ACTION,
        )
    if not policy_decision.decision_is_stable_across_scenarios:
        return _configured_trigger(
            config.triggers.investigate_unstable_allow,
            InvestigationTriggerReason.UNSTABLE_ALLOW,
        )
    return InvestigationTrigger(
        should_investigate=False,
        reason=InvestigationTriggerReason.STABLE_ALLOW,
    )


def _configured_trigger(
    enabled: bool,
    reason: InvestigationTriggerReason,
) -> InvestigationTrigger:
    if enabled:
        return InvestigationTrigger(should_investigate=True, reason=reason)
    return InvestigationTrigger(
        should_investigate=False,
        reason=InvestigationTriggerReason.DISABLED_BY_CONFIG,
    )
