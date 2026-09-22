"""Stuck-state detection: when does a hard failure become a human escalation
rather than just a returned result? For this build, every hard_failure
qualifies -- replayer/executor.py already caps recovery attempts and exhausts
the artifact's declared patterns before it ever returns one, so by the time this
fires the recoverable paths are genuinely spent, not just slow.

`should_alert_human`/`AlertNotice` are a deliberately separate, lighter path:
a business outcome the artifact author flagged with `alert_human=True`
(artifact/schema.py's `DeclaredOutcome`) is still a *successful, expected*
result -- the replay completes and returns normally to its caller -- it just
also deserves a person's attention (e.g. a fraud flag, an unusually large
amount). That is not "stuck": there is nothing to take control of, no live
session to hand off, and no reason to pause. Conflating the two would mean
every noteworthy-but-fine outcome blocks a real automated run on a human
responding, which is exactly the "happy-path-only" failure mode Section 1 of
the brief calls out.
"""

from __future__ import annotations

from dataclasses import dataclass

from replayer.classify import ReplayResult, ReplayResultType


@dataclass
class InterventionRequest:
    goal: str
    item: dict
    step: int | None
    reason: str
    evidence_path: str


@dataclass
class AlertNotice:
    goal: str
    item: dict
    outcome_name: str


def should_escalate(result: ReplayResult) -> bool:
    """True for a hard failure a human could actually do something about by
    taking control of the live session.

    `caller_error` is the exception: inputs that don't satisfy the artifact's
    `input_schema` fail before any interaction happens, so there is no stuck
    session to hand over and nothing an operator could fix at the browser -- the
    caller has to fix the call. Escalating those would page a human for a
    problem in the caller's own request, and (with the operator-signal TTL)
    stall an unattended run for minutes on a question no one at the keyboard
    can answer."""
    return result.type == ReplayResultType.HARD_FAILURE and not result.detail.get("caller_error")


def should_alert_human(result: ReplayResult) -> bool:
    """True only for a business outcome the artifact itself declared as
    alert-worthy -- never inferred from the outcome's name or from anything
    else about the result. hard_failure is deliberately excluded here: it
    already goes through should_escalate's heavier path, which supersedes a
    same result then also raising a duplicate, lower-priority alert."""
    return result.type == ReplayResultType.BUSINESS_OUTCOME and bool(result.detail.get("alert_human"))


def build_intervention_request(goal: str, item: dict, result: ReplayResult, run_dir: str) -> InterventionRequest:
    return InterventionRequest(
        goal=goal,
        item=item,
        step=result.detail.get("step"),
        reason=result.detail.get("reason") or result.detail.get("result", {}).get("error", "unspecified"),
        evidence_path=f"{run_dir}/failure/manifest.json",
    )


def build_alert_notice(goal: str, item: dict, result: ReplayResult) -> AlertNotice:
    return AlertNotice(goal=goal, item=item, outcome_name=result.detail["name"])
