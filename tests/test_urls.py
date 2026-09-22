"""Unit test for scripts/urls.py's normalize_url -- pure string logic,
no browser needed. Exists because a schemeless URL ("youtube.com", no
"https://") used to hit two real bugs at once, live: AllowlistConfig.for_url
scoped itself to an empty domain (urlsplit(url).netloc is empty without a
scheme), and Playwright's page.goto rejected the string outright
("Cannot navigate to invalid URL") -- an unhandled crash, not a structured
failure. normalize_url is the fix for the common case (someone just typed a
domain); surfaces/web.py's go_to now also never crashes even when it isn't
(see its own exception handling for that half of the fix).
"""

from __future__ import annotations

from guardrails.allowlist import AllowlistConfig
from scripts.urls import normalize_url


def test_bare_domain_gets_a_scheme_assumed():
    assert normalize_url("youtube.com") == "https://youtube.com"


def test_url_with_a_scheme_passes_through_unchanged():
    assert normalize_url("https://youtube.com") == "https://youtube.com"
    assert normalize_url("http://example.com/path") == "http://example.com/path"


def test_normalized_url_actually_scopes_the_allowlist_correctly():
    # This is the real bug, not just a string-shape assertion: before the fix,
    # AllowlistConfig.for_url("youtube.com") scoped to an empty domain because
    # urlsplit("youtube.com").netloc is "" with no scheme present.
    allowlist = AllowlistConfig.for_url(normalize_url("youtube.com"))
    assert "youtube.com" in allowlist.allowed_domains
