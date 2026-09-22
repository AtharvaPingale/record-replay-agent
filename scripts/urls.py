"""Shared CLI-input handling: turning what a human types as a target URL into
something the rest of the system can actually use. Not domain logic -- nothing
under agent/, surfaces/, or guardrails/ needs this; it exists because a CLI
argument is a raw string a person typed, and the surface/allowlist layers
below assume a proper URL, not a bare domain.

Lives under scripts/ (a namespace package, no __init__.py -- see run.sh's
PYTHONPATH) rather than in a generic top-level "utils" module: every consumer
of this file is itself a script under scripts/ (currently just
scripts/record_capability.py), and this stays exactly the size of "the one
thing a CLI entry point needs before it can hand a URL to the rest of the
system," not a place new unrelated helpers accumulate.
"""

from __future__ import annotations


def normalize_url(url: str) -> str:
    """A bare domain (`youtube.com`, no scheme) used to reach two different
    real problems at once: `AllowlistConfig.for_url` computes its scoped domain
    via `urlsplit(url).netloc`, which is empty for a schemeless string (the
    whole thing parses as a path, not a host) -- so the allowlist silently
    scoped to nothing, and separately Playwright's `page.goto` rejects a
    schemeless string outright ("Cannot navigate to invalid URL") rather than
    treating it as relative to anything. Assume `https://` for the common
    case (someone just typed a domain) instead of either failing outright or
    letting a broken allowlist scope through unnoticed."""
    if "://" not in url:
        normalized = f"https://{url}"
        print(f"[note] '{url}' has no scheme -- assuming '{normalized}'")
        return normalized
    return url
