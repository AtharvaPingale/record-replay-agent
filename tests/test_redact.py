"""Unit tests for guardrails/redact.py: two independent redaction mechanisms.

`redact_value` is shape-based (email/credit-card/SSN/API-key regexes) -- a
second, independent layer for anything that isn't provenance-tracked.
`resolve_credential_placeholders` / `mask_credentials_with_markers` are
provenance-based -- the primary mechanism, since a real credential (e.g. a
random test-account password) doesn't necessarily match any shape pattern at
all. This is the mechanism the whole "never persist secrets" claim rests on,
so it's tested directly rather than only exercised live.
"""

from __future__ import annotations

import json

import os

import pytest

from guardrails.redact import (
    mask_credentials_with_markers,
    credential_env_vars,
    mask_known_credentials_deep,
    mask_known_credential_values,
    redact_value,
    resolve_credential_placeholders,
)


# -- redact_value: shape-based, second layer ----------------------------------


def test_redact_value_masks_an_email():
    assert redact_value("contact me at a@b.com") == "contact me at [REDACTED:email]"


def test_redact_value_masks_a_credit_card_number():
    out = redact_value("card: 4111 1111 1111 1111")
    assert "4111" not in out
    assert "[REDACTED:credit_card]" in out


def test_redact_value_recurses_into_dicts_and_lists():
    out = redact_value({"note": "email me at x@y.com", "tags": ["a@b.com", "safe"]})
    assert out["note"] == "email me at [REDACTED:email]"
    assert out["tags"] == ["[REDACTED:email]", "safe"]


def test_redact_value_passes_through_non_string_types_untouched():
    assert redact_value(42) == 42
    assert redact_value(None) is None
    assert redact_value(True) is True


def test_redact_value_does_not_touch_a_value_with_no_recognizable_shape():
    # This is exactly why provenance-based redaction exists: a plain random
    # password doesn't match any shape pattern here.
    assert redact_value("correcthorsebatterystaple") == "correcthorsebatterystaple"


# -- resolve_credential_placeholders: the primary mechanism -------------------


def test_resolves_env_placeholder_from_the_environment(monkeypatch):
    monkeypatch.setenv("TEST_ACCOUNT_PASSWORD", "s3cret-value")
    resolved, credential_keys = resolve_credential_placeholders({"password": "{{env:TEST_ACCOUNT_PASSWORD}}"})
    assert resolved == {"password": "s3cret-value"}
    assert credential_keys == {"password"}


def test_non_credential_params_pass_through_unchanged():
    resolved, credential_keys = resolve_credential_placeholders({"text": "widget", "ms": 500})
    assert resolved == {"text": "widget", "ms": 500}
    assert credential_keys == set()


def test_mixed_params_only_flag_the_credential_keys():
    os.environ["TEST_ACCOUNT_EMAIL"] = "uiagent.test@example.com"
    try:
        resolved, credential_keys = resolve_credential_placeholders(
            {"text": "{{env:TEST_ACCOUNT_EMAIL}}", "other": "plain value"}
        )
        assert resolved["text"] == "uiagent.test@example.com"
        assert resolved["other"] == "plain value"
        assert credential_keys == {"text"}
    finally:
        del os.environ["TEST_ACCOUNT_EMAIL"]


def test_missing_env_var_raises_rather_than_silently_resolving_to_none():
    with pytest.raises(RuntimeError, match="NOT_A_REAL_VAR_XYZ"):
        resolve_credential_placeholders({"password": "{{env:NOT_A_REAL_VAR_XYZ}}"})


def test_a_field_placeholder_is_not_mistaken_for_a_credential():
    # {{query}} (an ordinary typed input) is resolved elsewhere, in
    # artifact/schema.py's render_step_params -- resolve_credential_placeholders
    # only recognizes the {{env:...}} shape and leaves anything else alone.
    resolved, credential_keys = resolve_credential_placeholders({"text": "{{query}}"})
    assert resolved == {"text": "{{query}}"}
    assert credential_keys == set()


# -- mask_credentials_with_markers: what actually reaches a log line ---------


def test_mask_credentials_swaps_the_resolved_secret_back_for_its_marker():
    rendered = {"text": "s3cret-value", "other": "widget"}
    original = {"text": "{{env:TEST_ACCOUNT_PASSWORD}}", "other": "{{query}}"}
    masked = mask_credentials_with_markers(rendered, original, credential_keys={"text"})
    assert masked["text"] == "{{env:TEST_ACCOUNT_PASSWORD}}"  # the real secret never appears
    assert masked["other"] == "widget"  # non-credential keys keep their resolved value


def test_mask_credentials_with_no_credential_keys_changes_nothing():
    rendered = {"text": "widget"}
    masked = mask_credentials_with_markers(rendered, {"text": "{{query}}"}, credential_keys=set())
    assert masked == {"text": "widget"}


def test_the_real_secret_never_survives_the_round_trip(monkeypatch):
    # End-to-end version of the two functions together: resolve, then mask for
    # logging -- the secret value itself must not appear anywhere in the
    # logged output, only its safe {{env:VAR}} marker.
    monkeypatch.setenv("TEST_ACCOUNT_PASSWORD", "hunter2-but-longer")
    original = {"text": "{{env:TEST_ACCOUNT_PASSWORD}}"}
    resolved, credential_keys = resolve_credential_placeholders(original)
    log_params = mask_credentials_with_markers(resolved, original, credential_keys)
    assert "hunter2-but-longer" not in str(log_params)
    assert log_params["text"] == "{{env:TEST_ACCOUNT_PASSWORD}}"


# -- mask_known_credential_values: the recording-side counterpart -------------
#
# resolve_credential_placeholders protects the replay direction (marker in the
# artifact, real value at the moment of use). Nothing protected the *recording*
# direction: a model typing a literal password into a login form put that
# literal straight into log.jsonl, and from there into the artifact.


def test_a_literal_secret_typed_during_recording_becomes_its_marker(monkeypatch):
    monkeypatch.setenv("TEST_ACCOUNT_PASSWORD", "hunter2-secret-value")
    masked, keys = mask_known_credential_values({"text": "hunter2-secret-value"})
    assert masked == {"text": "{{env:TEST_ACCOUNT_PASSWORD}}"}
    assert keys == {"text"}


def test_the_marker_it_produces_is_what_replay_already_resolves(monkeypatch):
    # The round trip is the point: masking a recording must leave something the
    # replay side understands, not merely something safe to write down.
    monkeypatch.setenv("MY_API_TOKEN", "tok-abcdef123456")
    masked, _ = mask_known_credential_values({"text": "tok-abcdef123456"})
    resolved, keys = resolve_credential_placeholders(masked)
    assert resolved == {"text": "tok-abcdef123456"}
    assert keys == {"text"}


def test_ordinary_values_are_left_alone(monkeypatch):
    monkeypatch.setenv("TEST_ACCOUNT_PASSWORD", "hunter2-secret-value")
    masked, keys = mask_known_credential_values({"text": "widget", "ms": 500})
    assert masked == {"text": "widget", "ms": 500}
    assert keys == set()


def test_a_secret_embedded_in_a_longer_string_is_still_masked(monkeypatch):
    monkeypatch.setenv("DB_PASSWORD", "s3cret-value-here")
    masked, _ = mask_known_credential_values({"text": "login as admin with s3cret-value-here ok"})
    assert "s3cret-value-here" not in masked["text"]
    assert "{{env:DB_PASSWORD}}" in masked["text"]


def test_a_non_credential_shaped_name_can_be_opted_in(monkeypatch):
    # TEST_ACCOUNT_EMAIL is sensitive without being credential-shaped by name;
    # AGENT_CREDENTIAL_VARS is how a deployment says so.
    monkeypatch.setenv("TEST_ACCOUNT_EMAIL", "qa-bot@example.com")
    monkeypatch.setenv("AGENT_CREDENTIAL_VARS", "TEST_ACCOUNT_EMAIL")
    masked, _ = mask_known_credential_values({"text": "qa-bot@example.com"})
    assert masked == {"text": "{{env:TEST_ACCOUNT_EMAIL}}"}


def test_short_env_values_are_not_masked(monkeypatch):
    # Masking a 3-character "secret" everywhere it appears would corrupt a
    # recording far more than it protects anything.
    monkeypatch.setenv("API_KEY", "abc")
    masked, keys = mask_known_credential_values({"text": "abc123 is a product code"})
    assert masked == {"text": "abc123 is a product code"}
    assert keys == set()


def test_a_longer_secret_wins_over_one_that_is_its_substring(monkeypatch):
    monkeypatch.setenv("SHORT_TOKEN", "abcdef1234")
    monkeypatch.setenv("LONG_TOKEN", "abcdef1234567890")
    masked, _ = mask_known_credential_values({"text": "abcdef1234567890"})
    assert masked == {"text": "{{env:LONG_TOKEN}}"}


def test_the_opt_in_config_var_is_not_itself_treated_as_a_secret(monkeypatch):
    # AGENT_CREDENTIAL_VARS holds variable *names*, but its own name matches the
    # "credential" hint -- left in the set it masked the very names it exists to
    # point at, producing nested {{env:{{env:...}}}} markers.
    monkeypatch.setenv("TEST_ACCOUNT_EMAIL", "qa-bot@example.com")
    monkeypatch.setenv("AGENT_CREDENTIAL_VARS", "TEST_ACCOUNT_EMAIL")
    assert "AGENT_CREDENTIAL_VARS" not in credential_env_vars()
    masked, _ = mask_known_credential_values({"text": "see TEST_ACCOUNT_EMAIL for the login"})
    assert masked == {"text": "see TEST_ACCOUNT_EMAIL for the login"}


def test_deep_masking_reaches_a_secret_outside_action_params(monkeypatch):
    # The case that matters for a recording run: the secret travels in the goal
    # string and in the model's own response, not only in an action's params.
    monkeypatch.setenv("TEST_ACCOUNT_PASSWORD", "hunter2-secret-value")
    detail = {
        "prompt": "Goal: log in with hunter2-secret-value then search",
        "response": "I typed hunter2-secret-value into the field",
        "decision": {"input": {"text": "hunter2-secret-value"}},
        "nested": [{"note": "hunter2-secret-value"}],
        "step": 3,
    }
    masked = mask_known_credentials_deep(detail)
    assert "hunter2-secret-value" not in json.dumps(masked)
    assert masked["decision"]["input"]["text"] == "{{env:TEST_ACCOUNT_PASSWORD}}"
    assert masked["nested"][0]["note"] == "{{env:TEST_ACCOUNT_PASSWORD}}"
    assert masked["step"] == 3  # non-strings pass through untouched

