"""Unit tests for guardrails/risk.py: the two-tier trust model (DEFAULT_RULES
vs. REPLAY_RISK_RULES). The distinction that matters most here is *not* which
actions are reversible -- it's that payment/account-deletion stay BLOCKED in
*both* profiles, unconditionally, which is the guarantee the rest of the
system's safety story depends on.
"""

from __future__ import annotations

from guardrails.risk import DEFAULT_RULES, REPLAY_RISK_RULES, RiskLevel, classify


# -- DEFAULT_RULES: the unattended agent, maximally conservative --------------


def test_default_rules_no_longer_block_login_by_themselves():
    # Real decision, not an oversight: login was removed from DEFAULT_RULES'
    # unconditional floor since it costs nothing irreversible by itself
    # (unlike checkout/payment/account-deletion, which stay BLOCKED below).
    # guardrails/allowlist.py's own route-pattern layer still blocks
    # */login* by default regardless -- reaching this classifier at all
    # already required an explicit, named allowlist override.
    rule = classify("click", "https://example.com/login", DEFAULT_RULES)
    assert rule is None


def test_default_rules_block_checkout_payment_and_account_deletion():
    for url in ["https://x.com/checkout", "https://x.com/payment", "https://x.com/delete_account"]:
        rule = classify("type", url, DEFAULT_RULES)
        assert rule is not None and rule.level == RiskLevel.BLOCKED, url


def test_default_rules_treat_cart_actions_as_reversible():
    rule = classify("click", "https://example.com/products", DEFAULT_RULES)
    assert rule is not None
    assert rule.level == RiskLevel.REVERSIBLE


def test_default_rules_have_no_opinion_on_an_unmatched_url():
    # classify() returns None when nothing matches; callers (surfaces/web.py)
    # treat unmatched as reversible, but that's the caller's decision, not
    # something classify() asserts.
    rule = classify("click", "https://example.com/some-neutral-page", DEFAULT_RULES)
    assert rule is None


# -- REPLAY_RISK_RULES: narrower, for a human-reviewed path only -------------


def test_replay_risk_rules_allow_login():
    # A pre-reviewed artifact's own declared login step must be permitted at
    # this layer -- true regardless of DEFAULT_RULES' own login stance, since
    # this is the narrower, human-reviewed profile a replay actually runs
    # under, not DEFAULT_RULES.
    rule = classify("type", "https://example.com/login", REPLAY_RISK_RULES)
    assert rule is None  # no rule matches login here -> not blocked


def test_replay_risk_rules_allow_reaching_checkout_review():
    rule = classify("click", "https://example.com/checkout", REPLAY_RISK_RULES)
    assert rule is not None
    assert rule.level == RiskLevel.REVERSIBLE


def test_replay_risk_rules_still_block_payment_signup_and_account_deletion():
    # The unconditional floor: no declared capability should ever need these,
    # so they stay blocked even in the narrower, human-supervised profile.
    for url in [
        "https://example.com/payment",
        "https://example.com/signup",
        "https://example.com/delete_account",
    ]:
        rule = classify("click", url, REPLAY_RISK_RULES)
        assert rule is not None and rule.level == RiskLevel.BLOCKED, url


def test_default_and_replay_rules_agree_on_the_unconditional_floor():
    # Whatever else differs between the two profiles, payment must be BLOCKED
    # in both -- this is the property the rest of the safety story leans on.
    for rules in (DEFAULT_RULES, REPLAY_RISK_RULES):
        rule = classify("click", "https://example.com/payment", rules)
        assert rule is not None and rule.level == RiskLevel.BLOCKED
