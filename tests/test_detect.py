"""Unit tests for escalation/detect.py's should_escalate/should_alert_human
split: a hard_failure escalates (pause + hand off the live session); a
business outcome the artifact flagged alert_human=True only notifies (the
replay already completed and returned normally). These must never overlap.
"""

from __future__ import annotations

from escalation.detect import AlertNotice, build_alert_notice, should_alert_human, should_escalate
from replayer.classify import ReplayResult, ReplayResultType


def test_should_alert_human_true_only_for_flagged_business_outcome():
    flagged = ReplayResult(ReplayResultType.BUSINESS_OUTCOME, {"name": "fraud_hold", "alert_human": True})
    assert should_alert_human(flagged)


def test_should_alert_human_false_for_routine_business_outcome():
    routine = ReplayResult(ReplayResultType.BUSINESS_OUTCOME, {"name": "no_matching_product", "alert_human": False})
    assert not should_alert_human(routine)


def test_should_alert_human_false_when_flag_absent_entirely():
    # A ReplayResult built without classify_outcome (e.g. success) never has
    # this key at all -- must not raise, must not be treated as flagged.
    result = ReplayResult(ReplayResultType.SUCCESS, {"outputs": {}})
    assert not should_alert_human(result)


def test_hard_failure_escalates_but_never_also_alerts():
    # should_escalate and should_alert_human are mutually exclusive by
    # construction (they check disjoint result types) -- this pins that down so
    # a future refactor can't quietly make a hard_failure trigger both paths.
    hard_failure = ReplayResult(ReplayResultType.HARD_FAILURE, {"step": 3, "reason": "no declared pattern matched"})
    assert should_escalate(hard_failure)
    assert not should_alert_human(hard_failure)


def test_build_alert_notice_carries_goal_item_and_outcome_name():
    result = ReplayResult(ReplayResultType.BUSINESS_OUTCOME, {"name": "fraud_hold", "alert_human": True})
    notice = build_alert_notice("reach checkout for a $500 order", {"query": "gold watch"}, result)
    assert notice == AlertNotice(
        goal="reach checkout for a $500 order", item={"query": "gold watch"}, outcome_name="fraud_hold"
    )
