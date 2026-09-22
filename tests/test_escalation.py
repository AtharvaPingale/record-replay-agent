"""Unit tests for the escalation seam -- resume.py, handoff.py and
operator_cli.py's signal handling, none of which had direct coverage.

These are the pieces the brief weighs under "human-in-the-loop escalation," and
two of the behaviours here are safety-relevant enough to be worth pinning down
in a test rather than trusting by inspection: that no non-answer can ever be
mistaken for a human saying "I fixed it," and that resolution is decided by
re-checking the artifact's own checkpoint rather than by anyone's say-so.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from artifact.schema import (
    Artifact,
    Checkpoint,
    DeclaredOutcome,
    LocatorCandidate,
    OutputExtractor,
    Step,
    Target,
)
from escalation.detect import (
    build_alert_notice,
    build_intervention_request,
    should_alert_human,
    should_escalate,
)
from escalation.handoff import (
    ControlOwner,
    describe_operator_changes,
    end_handoff,
    start_handoff,
)
from escalation.operator_cli import wait_for_operator_signal
from escalation.resume import verify_resolution
from replayer.classify import ReplayResult, ReplayResultType
from surfaces.base import EvidenceBundle, Observation


class FakeSurface:
    def __init__(self, url="https://example.com/checkout", dom="", visible=None, extract=None, title="t"):
        self.url, self.dom, self.title = url, dom, title
        self.visible = visible if visible is not None else set()
        self.extract = extract or {}

    def observe(self) -> Observation:
        return Observation(
            url=self.url, title=self.title, screenshot_path=Path("/dev/null"),
            dom_excerpt=self.dom, timestamp=0.0,
        )

    def is_visible(self, text: str) -> bool:
        return text in self.visible

    def is_selector_visible(self, selector: str) -> bool:
        return selector in self.visible

    def current_url(self) -> str:
        return self.url

    def extract_text(self, target: list[dict[str, Any]]) -> str | None:
        return self.extract.get(target[0]["value"])

    def act(self, action):  # pragma: no cover - not exercised by these tests
        return {"ok": True}

    def snapshot_for_evidence(self, out_dir: Path) -> EvidenceBundle:  # pragma: no cover
        raise NotImplementedError


def _artifact(**overrides: Any) -> Artifact:
    base: dict[str, Any] = dict(
        artifact_id="t",
        target=Target(base_url_pattern="https://example.com/*"),
        input_schema={},
        output_schema={},
        checkpoint=Checkpoint(type="element_visible", value="Review Your Order"),
        steps=[Step(step=1, action="wait", params={"ms": 1}, idempotency_note="safe")],
    )
    base.update(overrides)
    return Artifact(**base)


# -- control transfer --------------------------------------------------------


def test_control_formally_transfers_to_the_operator_and_back():
    # "A way to know who is (or should be) in control" -- recorded state, not an
    # inference from which thread happens to be blocked.
    session = start_handoff(9333, FakeSurface())
    assert session.control_owner is ControlOwner.OPERATOR

    end_handoff(session, FakeSurface(url="https://example.com/checkout#done"))
    assert session.control_owner is ControlOwner.AUTOMATION


def test_the_effect_of_the_handoff_on_the_shared_session_is_recorded():
    session = start_handoff(9333, FakeSurface(url="https://example.com/view_cart", title="Cart"))
    end_handoff(session, FakeSurface(url="https://example.com/checkout", title="Checkout"))

    changes = describe_operator_changes(session)
    assert changes["navigated"] is True
    assert changes["url_before"].endswith("/view_cart")
    assert changes["url_after"].endswith("/checkout")
    # The record must not overclaim what it is.
    assert "not the operator's individual actions" in changes["recorded"]


def test_change_record_is_honest_when_no_surface_was_available():
    session = end_handoff(start_handoff(9333), None)
    assert describe_operator_changes(session)["recorded"].startswith("none")


# -- the operator signal -----------------------------------------------------


def test_a_timeout_is_skip_never_done():
    # The load-bearing safety property: timing out must never be
    # indistinguishable from a human affirmatively resolving the issue.
    assert wait_for_operator_signal(timeout_s=0.15) == "skip"


def test_unavailable_stdin_is_skip_not_a_crash(monkeypatch):
    # An unattended run (CI, cron, stdin closed) hits EOF immediately. This used
    # to raise EOFError straight out of the no-timeout path -- crashing the
    # replay *after* it had already escalated.
    def _eof(*_args, **_kwargs):
        raise EOFError

    monkeypatch.setattr("builtins.input", _eof)
    assert wait_for_operator_signal() == "skip"
    assert wait_for_operator_signal(timeout_s=5.0) == "skip"


def test_an_explicit_auto_signal_still_drives_the_same_path():
    assert wait_for_operator_signal(auto_signal="done") == "done"


# -- resolution is re-verified, not trusted ----------------------------------


def test_resolution_is_confirmed_by_the_artifacts_own_checkpoint():
    artifact = _artifact(
        output_schema={"total": "string"},
        output_extractors=[OutputExtractor(name="total", target=LocatorCandidate(kind="dom_selector", value="#t"))],
    )
    surface = FakeSurface(visible={"Review Your Order"}, extract={"#t": "Rs. 400"})

    result = verify_resolution(artifact, surface)
    assert result.type == ReplayResultType.SUCCESS
    # A success via handoff still owes the caller its declared outputs.
    assert result.detail["outputs"] == {"total": "Rs. 400"}


def test_a_handoff_that_did_not_actually_fix_anything_is_still_a_hard_failure():
    surface = FakeSurface(visible=set())
    result = verify_resolution(_artifact(), surface)

    assert result.type == ReplayResultType.HARD_FAILURE
    assert "re-verified live, not assumed" in result.detail["reason"]


def test_a_declared_business_outcome_is_honoured_after_a_handoff_too():
    artifact = _artifact(
        business_outcomes=[DeclaredOutcome(name="no_match", match=Checkpoint(type="text_present", value="no results"))]
    )
    result = verify_resolution(artifact, FakeSurface(dom="<html>no results</html>"))
    assert result.type == ReplayResultType.BUSINESS_OUTCOME
    assert result.detail["name"] == "no_match"


# -- routing -----------------------------------------------------------------


def test_escalation_and_alert_are_mutually_exclusive():
    hard = ReplayResult(ReplayResultType.HARD_FAILURE, {"step": 3})
    flagged = ReplayResult(ReplayResultType.BUSINESS_OUTCOME, {"name": "fraud", "alert_human": True})

    assert should_escalate(hard) and not should_alert_human(hard)
    assert should_alert_human(flagged) and not should_escalate(flagged)


def test_intervention_request_carries_the_context_an_operator_needs():
    result = ReplayResult(ReplayResultType.HARD_FAILURE, {"step": 13, "result": {"error": "locate_failed"}})
    req = build_intervention_request("replay some-capability", {"query": "widget"}, result, "runs/r1")

    assert req.step == 13
    assert req.reason == "locate_failed"
    assert req.item == {"query": "widget"}
    assert req.evidence_path == "runs/r1/failure/manifest.json"


def test_alert_notice_names_the_outcome():
    result = ReplayResult(ReplayResultType.BUSINESS_OUTCOME, {"name": "large_amount", "alert_human": True})
    assert build_alert_notice("g", {"q": "x"}, result).outcome_name == "large_amount"


def test_a_caller_input_error_does_not_page_a_human():
    # Nothing has touched the page when input validation fails, so there is no
    # stuck session to hand over -- escalating would page someone for another
    # component's bug and stall an unattended run on the operator-signal TTL.
    caller_error = ReplayResult(
        ReplayResultType.HARD_FAILURE,
        {"reason": "inputs do not satisfy the artifact's input_schema", "caller_error": True},
    )
    assert not should_escalate(caller_error)
    # ...while an ordinary hard failure mid-session still does.
    assert should_escalate(ReplayResult(ReplayResultType.HARD_FAILURE, {"step": 9}))


def test_a_defaulted_verification_can_never_look_like_a_human_confirmation(monkeypatch):
    # The verification gate's default is a sentinel that is not a typed choice,
    # so "nobody answered" and "a human said confirm" are different values.
    def _eof(*_a, **_k):
        raise EOFError

    monkeypatch.setattr("builtins.input", _eof)
    assert wait_for_operator_signal(choices=("confirm", "reject"), default="unverified") == "unverified"
    assert wait_for_operator_signal(timeout_s=0.1, choices=("confirm", "reject"), default="unverified") == "unverified"


def test_custom_choices_are_honoured_and_unknown_answers_are_re_asked(monkeypatch):
    answers = iter(["maybe", "RESUME"])
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: next(answers))
    assert wait_for_operator_signal(choices=("confirm", "resume"), default="confirm") == "resume"
