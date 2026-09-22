"""Unit tests for artifact/schema.py.

Two things are under test here. First, render_step_params: the two placeholder
syntaxes ({{field}} vs {{env:VAR}}) must never be conflated, since that
distinction is what keeps a real secret out of the artifact and out of every
log derived from it. Second, the parts of the schema that are *enforced* rather
than merely declared -- the input contract, the output contract, and versioning
-- because a contract nothing checks is only a comment.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from artifact.schema import (
    Artifact,
    Checkpoint,
    InputValidationError,
    LocatorCandidate,
    OutputExtractor,
    RecoveryAction,
    Step,
    StepTarget,
    Target,
)


def _artifact(**overrides) -> Artifact:
    """A minimal valid artifact, with any field overridden for the case at hand."""
    base = dict(
        artifact_id="test-artifact",
        target=Target(base_url_pattern="https://example.com/*"),
        input_schema={},
        output_schema={},
        checkpoint=Checkpoint(type="url_contains", value="/done"),
        steps=[Step(step=1, action="wait", params={"ms": 1}, idempotency_note="safe")],
    )
    base.update(overrides)
    return Artifact(**base)


def make_artifact() -> Artifact:
    return Artifact(
        artifact_id="test-artifact",
        target=Target(base_url_pattern="https://example.com/*"),
        input_schema={"query": "string"},
        output_schema={},
        checkpoint=Checkpoint(type="url_contains", value="/done"),
        steps=[],
    )


def make_step(params: dict) -> Step:
    return Step(
        step=1,
        action="type",
        target=StepTarget(
            primary=LocatorCandidate(kind="dom_selector", value="#field"),
            robustness_reasoning="test",
        ),
        params=params,
        idempotency_note="test",
    )


def test_field_placeholder_substitutes_from_caller_inputs():
    artifact = make_artifact()
    step = make_step({"text": "{{query}}"})
    rendered, credential_keys = artifact.render_step_params(step, {"query": "widget"})
    assert rendered == {"text": "widget"}
    assert credential_keys == set()


def test_env_placeholder_resolves_from_environment_not_inputs(monkeypatch):
    monkeypatch.setenv("TEST_ACCOUNT_PASSWORD", "s3cret")
    artifact = make_artifact()
    step = make_step({"text": "{{env:TEST_ACCOUNT_PASSWORD}}"})
    # Deliberately do NOT supply a matching key in inputs -- an env credential
    # must never be satisfiable from caller-supplied inputs.
    rendered, credential_keys = artifact.render_step_params(step, {"query": "widget"})
    assert rendered == {"text": "s3cret"}
    assert credential_keys == {"text"}


def test_missing_field_input_raises_rather_than_silently_using_the_literal_template():
    artifact = make_artifact()
    step = make_step({"text": "{{query}}"})
    with pytest.raises(KeyError):
        artifact.render_step_params(step, {})  # "query" never supplied


def test_plain_literal_params_pass_through_unchanged():
    artifact = make_artifact()
    step = make_step({"ms": 500, "text": "a literal value, not a template"})
    rendered, credential_keys = artifact.render_step_params(step, {})
    assert rendered == {"ms": 500, "text": "a literal value, not a template"}
    assert credential_keys == set()


def test_mixed_field_and_credential_params_are_each_resolved_from_the_right_source(monkeypatch):
    monkeypatch.setenv("TEST_ACCOUNT_EMAIL", "uiagent.test@example.com")
    artifact = make_artifact()
    step = make_step({"email": "{{env:TEST_ACCOUNT_EMAIL}}", "search": "{{query}}"})
    rendered, credential_keys = artifact.render_step_params(step, {"query": "Dress"})
    assert rendered == {"email": "uiagent.test@example.com", "search": "Dress"}
    assert credential_keys == {"email"}


# -- the contract is enforced, not just declared -----------------------------


def test_validate_inputs_reports_every_problem_at_once():
    # One complete answer, not a fix-one-rerun-find-the-next loop: an agent
    # invoking this capability should learn everything it got wrong in one go.
    artifact = _artifact(input_schema={"query": "string", "quantity": "integer"})
    with pytest.raises(InputValidationError) as exc:
        artifact.validate_inputs({"quantity": "two", "bogus": 1})
    message = str(exc.value)
    assert "missing required input 'query'" in message
    assert "'quantity' expects integer, got str" in message
    assert "unknown input(s) ['bogus']" in message


def test_validate_inputs_rejects_bool_for_a_numeric_field():
    # bool is an int subclass in Python; a caller passing True where a quantity
    # belongs is a mistake, not an integer.
    with pytest.raises(InputValidationError):
        _artifact(input_schema={"quantity": "integer"}).validate_inputs({"quantity": True})


def test_validate_inputs_accepts_a_well_formed_call():
    artifact = _artifact(input_schema={"query": "string", "quantity": "integer", "price": "number"})
    artifact.validate_inputs({"query": "widget", "quantity": 2, "price": 9.99})  # must not raise


def test_an_artifact_cannot_promise_an_output_nothing_produces():
    with pytest.raises(ValidationError, match="no output_extractor can produce"):
        _artifact(output_schema={"total": "string"})


def test_an_artifact_cannot_return_an_output_it_never_declared():
    with pytest.raises(ValidationError, match="output_schema never declares"):
        _artifact(
            output_schema={},
            output_extractors=[OutputExtractor(name="ghost", target=LocatorCandidate(kind="dom_selector", value="#x"))],
        )


def test_save_versioned_refuses_to_destroy_a_different_artifact_at_the_same_version(tmp_path):
    # Shape-only comparison used to wave this through: two genuinely different
    # recordings of the same goal routinely share a schema shape (commonly an
    # empty one), and the second silently destroyed the first.
    _artifact(checkpoint=Checkpoint(type="url_contains", value="FIRST")).save_versioned(tmp_path)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        _artifact(checkpoint=Checkpoint(type="url_contains", value="SECOND")).save_versioned(tmp_path)


def test_save_versioned_can_bump_instead_of_refusing(tmp_path):
    first = _artifact(checkpoint=Checkpoint(type="url_contains", value="FIRST")).save_versioned(tmp_path)
    second = _artifact(checkpoint=Checkpoint(type="url_contains", value="SECOND")).save_versioned(
        tmp_path, on_conflict="bump"
    )
    assert first.name.endswith(".v1.json") and second.name.endswith(".v2.json")
    assert "FIRST" in first.read_text() and "SECOND" in second.read_text()


def test_re_saving_identical_content_is_a_no_op_not_a_new_version(tmp_path):
    a = _artifact(checkpoint=Checkpoint(type="url_contains", value="SAME"))
    assert a.save_versioned(tmp_path) == a.save_versioned(tmp_path, on_conflict="bump")
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_a_click_recovery_action_must_say_what_to_click():
    with pytest.raises(ValidationError, match="needs a target to click"):
        RecoveryAction(action="click")
