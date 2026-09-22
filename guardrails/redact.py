"""Strip anything matching a configurable secret/PII pattern list before it
reaches a log, artifact, or evidence bundle.

Not a dormant feature: `redact_value` runs on every single event
`telemetry/log.py`'s `RunLog.emit` writes, unconditionally, at the one write
path every caller in this repo goes through -- there is no way to log
something that skips it. A capability whose login needs no typed credential
and whose declared inputs are exempt by design (see below) may trigger this
pattern-matching layer rarely in practice; that's a property of that flow's
shape, not of whether the mechanism runs. The separate layer that redacts
screenshots and OCR text -- `guardrails/pii_redact.py` -- is what actually
has live matches on data-heavy pages; this module doesn't touch either.
"""

from __future__ import annotations

import re

# Deliberately NOT here: a bare-digit-run "account number" pattern. It was
# added once, when the first lookup capability took an account number as its
# input, and removed on review. A declared input is the caller's own value --
# they supplied it, so writing it down is not a new exposure -- and masking it
# defeats the one audit question that matters most for an agent operating
# banking software: did it look up the account it was told to? Incidental PII
# (every *other* member's number, balance and email on the page) is masked by
# guardrails/pii_redact.py before it reaches any prompt, so it arrives here
# already as [REDACTED:...]. That is the layer that protects anything; this one
# is for shapes that are secrets in their own right.
DEFAULT_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "api_key": re.compile(r"\b(sk|pk|key)[-_][A-Za-z0-9]{16,}\b"),
    "password_field": re.compile(r"(\"password\"\s*:\s*)\"[^\"]*\"", re.IGNORECASE),
}


def redact_text(text: str, patterns: dict[str, re.Pattern] = DEFAULT_PATTERNS) -> str:
    out = text
    for name, pattern in patterns.items():
        if name == "password_field":
            out = pattern.sub(r'\1"[REDACTED]"', out)
        else:
            out = pattern.sub(f"[REDACTED:{name}]", out)
    return out


def redact_value(value, patterns: dict[str, re.Pattern] = DEFAULT_PATTERNS):
    """Recursively redact strings inside dicts/lists; pass through other types."""
    if isinstance(value, str):
        return redact_text(value, patterns)
    if isinstance(value, dict):
        return {k: redact_value(v, patterns) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(v, patterns) for v in value]
    return value


ENV_PLACEHOLDER = re.compile(r"^\{\{env:(\w+)\}\}$")


def resolve_credential_placeholders(params: dict) -> tuple[dict, set[str]]:
    """Resolve `{{env:VAR_NAME}}` values from the process environment. Returns
    (resolved_params, credential_keys) -- the second tells a caller which keys
    now hold a real secret, so it can log the *pre-resolution* version instead.

    This exists because pattern-based redaction (redact_value above) is blind
    to a value with no recognizable shape -- a random test-account password
    doesn't match any regex a "credit card" or "api key" pattern would catch.
    Provenance (this value came from an env-credential placeholder) is the only
    reliable signal here, not the value's shape.

    This is the *replay* direction (artifact/schema.py's render_step_params):
    marker in the artifact, real value at the moment of use. The recording
    direction is mask_known_credential_values below, which turns a literal
    secret a model typed during discovery back into the same marker."""
    import os

    resolved: dict = {}
    credential_keys: set[str] = set()
    for k, v in params.items():
        m = ENV_PLACEHOLDER.match(v) if isinstance(v, str) else None
        if m:
            var_name = m.group(1)
            value = os.environ.get(var_name)
            if value is None:
                raise RuntimeError(f"required env var '{var_name}' is not set")
            resolved[k] = value
            credential_keys.add(k)
        else:
            resolved[k] = v
    return resolved, credential_keys


def mask_credentials_with_markers(rendered: dict, original: dict, credential_keys: set[str]) -> dict:
    """For logging/artifact-building: swap a credential key's *resolved secret*
    back out for its original `{{env:VAR_NAME}}` marker -- not a generic
    "[REDACTED]" string, because the marker itself is safe to persist (it names
    an env var, not a value) and is exactly what artifact/from_run.py needs
    verbatim to build a replayable template. Every non-credential key keeps its
    resolved value (e.g. an actual search term), since that's both safe and far
    more useful for audit than showing its own unresolved `{{field}}` template."""
    return {k: (original[k] if k in credential_keys else v) for k, v in rendered.items()}


# Env var names that hold something we must never write down. Matched
# case-insensitively as substrings of the *name*, so TEST_ACCOUNT_PASSWORD and
# STRIPE_API_KEY are both covered without enumerating either.
_CREDENTIAL_NAME_HINTS = (
    "password", "passwd", "passphrase", "secret", "token",
    "api_key", "apikey", "access_key", "private_key", "credential",
)

# Below this length a value is too likely to collide with ordinary page text to
# mask safely -- masking "admin" everywhere it appears would corrupt a recording
# far more than it would protect anything.
_MIN_MASKABLE_CREDENTIAL_LENGTH = 6

# The opt-in list is configuration -- it holds variable *names*, not a secret --
# yet its own name matches "credential" below. Left in, it would treat its own
# contents as a secret and mask the very names it exists to point at.
_CREDENTIAL_CONFIG_VAR = "AGENT_CREDENTIAL_VARS"


def credential_env_vars() -> dict[str, str]:
    """The env vars this process should treat as secrets: those whose *name*
    looks like a credential, plus anything explicitly named in
    AGENT_CREDENTIAL_VARS (comma-separated).

    That second route exists for values that are sensitive without being
    credential-*shaped* by name -- this repo's own TEST_ACCOUNT_EMAIL, say,
    which the canonical artifact genuinely uses as a login credential."""
    import os

    explicit = {v.strip() for v in os.environ.get(_CREDENTIAL_CONFIG_VAR, "").split(",") if v.strip()}
    out: dict[str, str] = {}
    for name, value in os.environ.items():
        if name == _CREDENTIAL_CONFIG_VAR:
            continue
        if not value or len(value) < _MIN_MASKABLE_CREDENTIAL_LENGTH:
            continue
        if name in explicit or any(hint in name.lower() for hint in _CREDENTIAL_NAME_HINTS):
            out[name] = value
    return out


def _swap_secrets_for_markers(text: str, known: list[tuple[str, str]]) -> tuple[str, bool]:
    """Replace every known secret value in `text` with its `{{env:NAME}}` marker.
    Returns (text, whether anything was replaced)."""
    out, hit = text, False
    for var_name, secret in known:
        if secret in out:
            out = out.replace(secret, "{{env:" + var_name + "}}")
            hit = True
    return out, hit


def mask_known_credential_values(params: dict) -> tuple[dict, set[str]]:
    """The recording-side counterpart to resolve_credential_placeholders.

    During a *discovery* run nothing is a placeholder yet: the model types a
    literal value into a login form, and that literal went straight into
    log.jsonl -- and from there, via artifact/from_run.py, into the artifact
    itself. Shape-based redaction (redact_value above) cannot help, because a
    real password matches no pattern.

    So match on the only thing that is reliable here: the value. Anything equal
    to (or containing) a known credential env var's value is swapped for that
    var's `{{env:NAME}}` marker. That protects the log and, because the marker
    is exactly what the replay side already understands, the resulting artifact
    replays correctly instead of carrying a baked-in secret.

    Longest values first, so an env var whose value is a substring of another
    can't mask the shorter one out from under it. Returns (masked_params,
    credential_keys)."""
    known = sorted(credential_env_vars().items(), key=lambda kv: -len(kv[1]))
    if not known:
        return dict(params), set()

    masked: dict = {}
    credential_keys: set[str] = set()
    for key, value in params.items():
        if not isinstance(value, str):
            masked[key] = value
            continue
        new_value, hit = _swap_secrets_for_markers(value, known)
        masked[key] = new_value
        if hit:
            credential_keys.add(key)
    return masked, credential_keys


def mask_known_credentials_deep(value):
    """Recursively swap known secret values for their `{{env:VAR}}` markers
    anywhere inside a nested structure.

    This is the whole-log version of mask_known_credential_values above, and it
    exists because a secret does not only travel through an action's `params`. A
    goal string naming a test account, a model's own response quoting what it
    typed, an error message echoing a failed value -- all of those reach
    telemetry/log.py too, and none of them are dicts of parameters.

    Pattern-based redaction (redact_value) cannot catch any of it: a random
    password matches no regex. Provenance-by-value can, and applying it at the
    single write path means no caller has to remember to."""
    known = sorted(credential_env_vars().items(), key=lambda kv: -len(kv[1]))
    if not known:
        return value

    def _walk(v):
        if isinstance(v, str):
            return _swap_secrets_for_markers(v, known)[0]
        if isinstance(v, dict):
            return {k: _walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_walk(x) for x in v]
        return v

    return _walk(value)
