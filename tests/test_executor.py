"""Unit tests for replayer/executor.py -- the deterministic production path an
agent actually invokes.

This module had no direct coverage at all: tests/test_classify.py exercises the
pure classifier, but nothing exercised the executor that *dispatches* on it --
the retry ceiling, the declared-recovery path, the final-checkpoint polling, the
input contract, the deadline, or how a policy refusal is reported. Those are the
behaviours the brief weighs under "robustness & error handling," so they are
tested here against a scriptable fake Surface: no browser, no live site, no
model.
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
    RecoverablePattern,
    RecoveryAction,
    Step,
    StepTarget,
    Target,
)
from replayer.classify import ReplayResultType
from replayer.executor import replay_artifact
from surfaces.base import Action, EvidenceBundle, GuardrailBlocked, Observation
from telemetry.log import RunLog


class FakeSurface:
    """Scriptable stand-in for WebSurface, same pattern as test_agent_loop.py's.

    `act_results` is consumed one entry per act() call; an entry that is a
    GuardrailBlocked instance is raised rather than returned, which is how the
    real surface reports a policy refusal. `visible` drives is_visible(), so a
    test can make the checkpoint start failing and then start passing.
    """

    def __init__(
        self,
        act_results: list[Any] | None = None,
        url: str = "https://example.com/products",
        dom: str = "<html>page</html>",
        visible: set[str] | None = None,
        extract: dict[str, str] | None = None,
    ):
        self.act_results = list(act_results or [])
        self.url = url
        self.dom = dom
        self.visible = visible if visible is not None else set()
        self.extract = extract or {}
        self.acted: list[Action] = []
        self.observations = 0

    def observe(self) -> Observation:
        self.observations += 1
        return Observation(
            url=self.url, title="", screenshot_path=Path("/dev/null"),
            dom_excerpt=self.dom, timestamp=0.0,
        )

    def act(self, action: Action) -> dict[str, Any]:
        self.acted.append(action)
        result = self.act_results.pop(0) if self.act_results else {"ok": True}
        if isinstance(result, GuardrailBlocked):
            raise result
        return result

    def snapshot_for_evidence(self, out_dir: Path) -> EvidenceBundle:
        out_dir.mkdir(parents=True, exist_ok=True)
        shot, dom = out_dir / "s.png", out_dir / "d.html"
        shot.write_bytes(b"")
        dom.write_text(self.dom)
        return EvidenceBundle(screenshot_path=shot, dom_snapshot_path=dom, url=self.url)

    def is_visible(self, text: str) -> bool:
        return text in self.visible

    def is_selector_visible(self, selector: str) -> bool:
        return selector in self.visible

    def current_url(self) -> str:
        return self.url

    def extract_text(self, target: list[dict[str, Any]]) -> str | None:
        return self.extract.get(target[0]["value"])


def _artifact(**overrides: Any) -> Artifact:
    base: dict[str, Any] = dict(
        artifact_id="t",
        version=1,
        target=Target(base_url_pattern="https://example.com/*"),
        input_schema={},
        output_schema={},
        checkpoint=Checkpoint(type="element_visible", value="Done"),
        steps=[
            Step(
                step=1,
                action="click",
                target=StepTarget(
                    primary=LocatorCandidate(kind="dom_selector", value="#go"),
                    robustness_reasoning="stable id",
                ),
                idempotency_note="safe",
            )
        ],
    )
    base.update(overrides)
    return Artifact(**base)


def _run(artifact: Artifact, surface: FakeSurface, tmp_path: Path, inputs: dict | None = None, **kw):
    run_log = RunLog(run_id="t", out_dir=tmp_path)
    result = replay_artifact(artifact, inputs or {}, surface, run_log, tmp_path, **kw)
    return result, run_log.read_all()


# -- the happy path and its output contract ----------------------------------


def test_success_verifies_the_checkpoint_and_returns_declared_outputs(tmp_path):
    artifact = _artifact(
        output_schema={"total": "string"},
        output_extractors=[OutputExtractor(name="total", target=LocatorCandidate(kind="dom_selector", value="#t"))],
    )
    surface = FakeSurface(visible={"Done"}, extract={"#t": "Rs. 400"})
    result, _ = _run(artifact, surface, tmp_path)

    assert result.type == ReplayResultType.SUCCESS
    assert result.detail["outputs"] == {"total": "Rs. 400"}


def test_all_steps_ok_but_checkpoint_absent_is_a_hard_failure_not_a_success(tmp_path):
    # The whole point of a checkpoint: "nothing errored" is not evidence the
    # goal was reached.
    surface = FakeSurface(visible=set())
    result, _ = _run(_artifact(), surface, tmp_path)
    assert result.type == ReplayResultType.HARD_FAILURE


# -- the input contract ------------------------------------------------------


def test_missing_required_input_is_a_structured_result_not_a_keyerror(tmp_path):
    artifact = _artifact(
        input_schema={"query": "string"},
        steps=[Step(step=1, action="type", params={"text": "{{query}}"}, idempotency_note="safe")],
    )
    surface = FakeSurface(visible={"Done"})
    result, events = _run(artifact, surface, tmp_path, inputs={})

    assert result.type == ReplayResultType.HARD_FAILURE
    assert "input_schema" in result.detail["reason"]
    assert "query" in result.detail["detail"]
    assert surface.acted == [], "nothing may execute when the inputs are invalid"
    assert any(e["kind"] == "invalid_inputs" for e in events)


def test_wrong_input_type_is_rejected_before_anything_executes(tmp_path):
    artifact = _artifact(
        input_schema={"query": "string"},
        steps=[Step(step=1, action="type", params={"text": "{{query}}"}, idempotency_note="safe")],
    )
    surface = FakeSurface(visible={"Done"})
    result, _ = _run(artifact, surface, tmp_path, inputs={"query": 123})

    assert result.type == ReplayResultType.HARD_FAILURE
    assert "expects string, got int" in result.detail["detail"]
    assert surface.acted == []


# -- the error taxonomy ------------------------------------------------------


def test_declared_business_outcome_is_not_a_failure(tmp_path):
    artifact = _artifact(
        business_outcomes=[
            DeclaredOutcome(name="no_match", match=Checkpoint(type="text_absent", value="add-to-cart"))
        ]
    )
    surface = FakeSurface(act_results=[{"ok": False, "error": "locate_failed"}], dom="<html>empty grid</html>")
    result, _ = _run(artifact, surface, tmp_path)

    assert result.type == ReplayResultType.BUSINESS_OUTCOME
    assert result.detail["name"] == "no_match"


def test_alert_human_fires_for_a_flagged_business_outcome(tmp_path):
    artifact = _artifact(
        business_outcomes=[
            DeclaredOutcome(
                name="fraud_flag",
                match=Checkpoint(type="text_absent", value="add-to-cart"),
                alert_human=True,
            )
        ]
    )
    surface = FakeSurface(act_results=[{"ok": False, "error": "locate_failed"}])
    result, events = _run(artifact, surface, tmp_path)

    assert result.type == ReplayResultType.BUSINESS_OUTCOME
    assert any(e["kind"] == "human_alert" for e in events)


def test_alert_human_also_fires_when_the_outcome_surfaces_at_the_final_checkpoint(tmp_path):
    # The real gap this covers: should_alert_human used to be consulted only on
    # the mid-step path, so a flagged outcome reached after the last step
    # completed was silently swallowed.
    artifact = _artifact(
        business_outcomes=[
            DeclaredOutcome(
                name="fraud_flag",
                match=Checkpoint(type="text_present", value="under review"),
                alert_human=True,
            )
        ]
    )
    surface = FakeSurface(dom="<html>under review</html>", visible=set())
    result, events = _run(artifact, surface, tmp_path)

    assert result.type == ReplayResultType.BUSINESS_OUTCOME
    assert any(e["kind"] == "human_alert" for e in events)


def test_undeclared_failure_is_a_hard_failure_with_an_evidence_bundle(tmp_path):
    surface = FakeSurface(act_results=[{"ok": False, "error": "locate_failed", "detail": "no candidate"}])
    result, events = _run(_artifact(), surface, tmp_path)

    assert result.type == ReplayResultType.HARD_FAILURE
    assert result.detail["step"] == 1
    assert (tmp_path / "failure" / "manifest.json").exists(), "Section 3.5's richer on-failure signal"
    assert (tmp_path / "failure" / "screenshot.png").exists()
    assert any(e["kind"] == "hard_failure" for e in events)


# -- policy refusal ----------------------------------------------------------


def test_guardrail_refusal_mid_replay_is_a_classified_result_not_a_crash(tmp_path):
    # Previously this escaped as an unhandled traceback: no classified result,
    # no evidence bundle, no escalation, and the caller's surface.close() never
    # ran. agent/loop.py always handled it; this path did not.
    surface = FakeSurface(act_results=[GuardrailBlocked("click navigated to a blocked page (/payment)")])
    result, events = _run(_artifact(), surface, tmp_path)

    assert result.type == ReplayResultType.HARD_FAILURE
    assert result.detail["reason"] == "guardrail refused this step"
    assert "blocked page" in result.detail["result"]["detail"]
    assert (tmp_path / "failure" / "manifest.json").exists()
    assert any(e["kind"] == "hard_failure" for e in events)


def test_a_refusal_is_never_retried(tmp_path):
    # A refusal is a policy breach to surface, not transient friction. Even with
    # a declared recoverable pattern that would otherwise match, it must not
    # burn the retry budget re-attempting a blocked action.
    artifact = _artifact(
        recoverable_patterns=[
            RecoverablePattern(
                name="popup", match=Checkpoint(type="text_present", value="page"), recovery="dismiss"
            )
        ]
    )
    surface = FakeSurface(act_results=[GuardrailBlocked("refused"), {"ok": True}])
    result, _ = _run(artifact, surface, tmp_path)

    assert result.type == ReplayResultType.HARD_FAILURE
    assert len(surface.acted) == 1, "the refused action must not be attempted again"


# -- recovery ----------------------------------------------------------------


def test_recoverable_pattern_runs_its_declared_recovery_action_then_retries(tmp_path):
    # Before RecoveryAction existed, `recovery` was prose the executor never
    # read: it retried an unchanged page and presumed the friction had cleared.
    artifact = _artifact(
        recoverable_patterns=[
            RecoverablePattern(
                name="popup",
                match=Checkpoint(type="text_present", value="subscribed"),
                recovery="dismiss the newsletter popup",
                recovery_action=RecoveryAction(action="key", params={"combo": "Escape"}),
            )
        ]
    )
    surface = FakeSurface(
        act_results=[{"ok": False, "error": "intercepted"}, {"ok": True}, {"ok": True}],
        dom="<html>successfully subscribed</html>",
        visible={"Done"},
    )
    result, events = _run(artifact, surface, tmp_path)

    assert result.type == ReplayResultType.SUCCESS
    assert [a.type for a in surface.acted] == ["click", "key", "click"]
    assert surface.acted[1].params == {"combo": "Escape"}
    assert any(e["kind"] == "recovered" for e in events)


def test_a_pattern_with_no_declared_action_still_gets_the_generic_escape(tmp_path):
    # Backwards compatibility: artifacts recorded before recovery_action existed
    # must behave exactly as they did.
    artifact = _artifact(
        recoverable_patterns=[
            RecoverablePattern(
                name="popup", match=Checkpoint(type="text_present", value="subscribed"), recovery="dismiss"
            )
        ]
    )
    surface = FakeSurface(
        act_results=[{"ok": False, "error": "intercepted"}, {"ok": True}, {"ok": True}],
        dom="<html>successfully subscribed</html>",
        visible={"Done"},
    )
    result, _ = _run(artifact, surface, tmp_path)

    assert result.type == ReplayResultType.SUCCESS
    assert surface.acted[1].type == "key" and surface.acted[1].params == {"combo": "Escape"}


def test_a_recoverable_state_that_never_clears_becomes_a_hard_failure(tmp_path):
    artifact = _artifact(
        recoverable_patterns=[
            RecoverablePattern(
                name="popup", match=Checkpoint(type="text_present", value="subscribed"), recovery="dismiss"
            )
        ]
    )
    surface = FakeSurface(
        act_results=[{"ok": False, "error": "intercepted"}] * 20,
        dom="<html>successfully subscribed</html>",
    )
    result, _ = _run(artifact, surface, tmp_path)
    assert result.type == ReplayResultType.HARD_FAILURE


def test_retries_are_capped_by_the_steps_own_declared_ceiling(tmp_path):
    artifact = _artifact(
        steps=[Step(step=1, action="wait", params={"ms": 1}, idempotency_note="safe", max_recovery_attempts=0)],
        recoverable_patterns=[
            RecoverablePattern(
                name="popup", match=Checkpoint(type="text_present", value="page"), recovery="dismiss"
            )
        ],
    )
    surface = FakeSurface(act_results=[{"ok": False, "error": "x"}] * 10)
    result, _ = _run(artifact, surface, tmp_path)

    assert result.type == ReplayResultType.HARD_FAILURE
    assert len(surface.acted) == 1, "max_recovery_attempts=0 means exactly one attempt"


# -- the deadline ------------------------------------------------------------


def test_an_exhausted_deadline_stops_the_replay_with_a_clear_reason(tmp_path):
    artifact = _artifact(
        steps=[Step(step=n, action="wait", params={"ms": 1}, idempotency_note="safe") for n in (1, 2, 3)]
    )
    surface = FakeSurface(visible={"Done"})
    result, _ = _run(artifact, surface, tmp_path, deadline_s=-1.0)

    assert result.type == ReplayResultType.HARD_FAILURE
    assert "deadline" in result.detail["reason"]
    assert surface.acted == [], "an already-blown deadline must not start executing steps"


def test_a_healthy_run_is_unaffected_by_a_generous_deadline(tmp_path):
    surface = FakeSurface(visible={"Done"})
    result, _ = _run(_artifact(), surface, tmp_path, deadline_s=300.0)
    assert result.type == ReplayResultType.SUCCESS


# -- credentials -------------------------------------------------------------


def test_a_resolved_credential_never_reaches_the_run_log(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_SECRET_PW", "hunter2-not-a-real-password")
    artifact = _artifact(
        steps=[Step(step=1, action="type", params={"text": "{{env:TEST_SECRET_PW}}"}, idempotency_note="safe")]
    )
    surface = FakeSurface(visible={"Done"})
    result, events = _run(artifact, surface, tmp_path)

    assert result.type == ReplayResultType.SUCCESS
    # the real secret is what got executed against the page...
    assert surface.acted[0].params["text"] == "hunter2-not-a-real-password"
    # ...and the marker, not the secret, is what got written down
    written = (tmp_path / "log.jsonl").read_text()
    assert "hunter2-not-a-real-password" not in written
    assert "{{env:TEST_SECRET_PW}}" in written


# -- the contract can name its inputs -----------------------------------------


def test_checkpoint_and_extractor_placeholders_are_rendered_from_inputs(tmp_path):
    """Found live on the first parameterized lookup: a checkpoint of 'some row
    is visible' and an extractor of 'the first dollar amount in the table' both
    passed on a page where the search filter had NOT applied, and replay
    reported another customer's balance as the requested member's -- as
    success. A lookup's contract has to be able to say *which* member."""
    artifact = _artifact(
        input_schema={"member_name": "string"},
        checkpoint=Checkpoint(type="element_visible", value="{{member_name}}"),
        output_schema={"balance": "string"},
        output_extractors=[
            OutputExtractor(
                name="balance",
                target=LocatorCandidate(kind="dom_selector", value='tr:has-text("{{member_name}}") .balance'),
            )
        ],
        steps=[Step(step=1, action="type", params={"text": "{{member_name}}"}, idempotency_note="safe")],
    )
    # The page shows Jane's row; Mike's is the first dollar amount but must not be read.
    surface = FakeSurface(
        visible={"Jane Smith"},
        extract={'tr:has-text("Jane Smith") .balance': "$25,000.50", 'tr:has-text("Mike Wilson") .balance': "$8,500.75"},
    )
    result, _ = _run(artifact, surface, tmp_path, inputs={"member_name": "Jane Smith"})

    assert result.type == ReplayResultType.SUCCESS
    assert result.detail["outputs"] == {"balance": "$25,000.50"}


def test_a_rendered_checkpoint_fails_for_a_member_who_is_not_on_the_page(tmp_path):
    artifact = _artifact(
        input_schema={"member_name": "string"},
        checkpoint=Checkpoint(type="element_visible", value="{{member_name}}"),
        business_outcomes=[DeclaredOutcome(name="no_such_member", match=Checkpoint(type="element_visible", value="No users found"))],
        steps=[Step(step=1, action="type", params={"text": "{{member_name}}"}, idempotency_note="safe")],
    )
    surface = FakeSurface(visible={"No users found"})
    result, _ = _run(artifact, surface, tmp_path, inputs={"member_name": "Nobody Real"})

    assert result.type == ReplayResultType.BUSINESS_OUTCOME
    assert result.detail["name"] == "no_such_member"


def test_rendering_the_contract_never_touches_credentials_or_the_original(tmp_path):
    artifact = _artifact(
        input_schema={"q": "string"},
        checkpoint=Checkpoint(type="text_present", value="{{q}} results"),
        steps=[Step(step=1, action="type", params={"text": "{{env:SECRET_PW}}"}, idempotency_note="safe")],
    )
    rendered = artifact.with_inputs_rendered({"q": "shoes"})

    assert rendered.checkpoint.value == "shoes results"
    assert artifact.checkpoint.value == "{{q}} results", "must return a copy, not mutate the original"
    assert rendered.steps[0].params == {"text": "{{env:SECRET_PW}}"}, "steps and credentials are untouched"


def test_a_null_output_is_flagged_on_an_otherwise_successful_replay(tmp_path):
    # Found live: an extractor anchored to `tr` on a div-built table returned
    # None for every member while replay said success three times running.
    artifact = _artifact(
        output_schema={"balance": "string"},
        output_extractors=[OutputExtractor(name="balance", target=LocatorCandidate(kind="dom_selector", value="#nope"))],
    )
    surface = FakeSurface(visible={"Done"}, extract={})  # checkpoint holds, extractor misses
    result, events = _run(artifact, surface, tmp_path)

    assert result.type == ReplayResultType.SUCCESS, "the checkpoint is the success condition"
    assert result.detail["outputs"] == {"balance": None}
    assert result.detail["outputs_missing"] == ["balance"], "but the caller is told which fields are empty"
    assert any(e["kind"] == "output_missing" for e in events)
