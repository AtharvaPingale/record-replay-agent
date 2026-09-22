"""Risk classification for action types, independent of the allowlist.

The allowlist says *where* the agent may go; this says *how dangerous* an action
is once there. A lookup, a filter, a cart add/remove is reversible and auto-allowed.
Anything that would touch checkout, payment, or account deletion is classified
`blocked` outright -- not confirm-and-proceed -- because we never want this agent
one step from spending real money or destroying an account, even hypothetically,
on a practice site.

`RiskLevel.CONFIRM` is declared for the general case (a real deployment might
want a human nod before an irreversible-but-legitimate action) but is, stated
plainly, not implemented: no rule in DEFAULT_RULES or REPLAY_RISK_RULES ever
produces it, `classify()` doesn't distinguish it from any other non-BLOCKED
level, and `surfaces/web.py`'s `act()` only special-cases `BLOCKED` -- a
CONFIRM-level rule, if one existed, would execute exactly like a REVERSIBLE
one, no pause, no prompt, contradicting what its own name and comment claim.
Found auditing this file for stale code: an earlier version of this docstring
said the mechanism was "real and exercised by tests," which was never true of
the enforcement path and isn't true of the tests either. Left as a declared,
honest gap rather than quietly implemented here, because the real fix touches
a design question this project hasn't needed to answer yet -- where the pause
belongs (inside `WebSurface.act()`, which would couple a low-level surface to
`escalation/operator_cli.py`'s interactive CLI, or surfaced to the caller as a
new result the way `hard_failure` already is) -- not a small patch.

Login is not part of that unconditional floor (it used to be; removed after a
direct decision to allow it, see DEFAULT_RULES below) -- it costs nothing
irreversible by itself, unlike payment or account deletion, and
guardrails/allowlist.py's own route-pattern layer still blocks */login* by
default regardless, so reaching a login action at all already required an
explicit, named override at that layer first.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RiskLevel(str, Enum):
    REVERSIBLE = "reversible"  # auto-allowed
    CONFIRM = "confirm"  # requires explicit confirmation before executing
    BLOCKED = "blocked"  # never executed, no override


@dataclass
class RiskRule:
    # matches if the action type is in `action_types` AND (no url_patterns, or the
    # current url matches one of them) -- lets the same action type (e.g. "click")
    # be reversible on /products but blocked on /payment.
    action_types: list[str]
    level: RiskLevel
    url_patterns: list[str] = None  # fnmatch patterns against the URL; None = any
    reason: str = ""


DEFAULT_RULES: list[RiskRule] = [
    RiskRule(
        action_types=["click", "type", "select"],
        level=RiskLevel.BLOCKED,
        url_patterns=["*checkout*", "*payment*", "*/delete_account*"],
        reason="checkout/payment/account-deletion actions are blocked outright, never confirm-and-proceed",
    ),
    # Login is deliberately NOT in the rule above -- removed after a direct
    # user decision (rather than left as part of the unconditional floor):
    # unlike checkout/payment/account-deletion, a login action costs nothing
    # irreversible by itself, and guardrails/allowlist.py's own
    # DEFAULT_BLOCKED_PATTERNS still blocks */login* by default at the route
    # layer regardless -- reaching this rule at all already required an
    # explicit, named allowlist override (scripts/record_capability.py's
    # --allow-blocked flag) for a target a human chose
    # to point this at. This rule no longer re-blocks it a second time once
    # that first, human-made decision has already been given.
    RiskRule(
        action_types=["click"],
        level=RiskLevel.REVERSIBLE,
        url_patterns=["*/products*", "*/product_details*", "*/view_cart*"],
        reason="adding/removing cart items is reversible",
    ),
]

# A deliberately different, *narrower* trust boundary than DEFAULT_RULES,
# used only where a human has already reviewed every action that will run:
#
#   - A human-supervised recording session -- a human (the operator driving
#     the discovery harness) approves each action as it happens, live.
#   - replayer/executor.py's replay path -- by construction, replay only ever
#     executes an artifact's already-declared, already-reviewed steps. A
#     step being *in the artifact at all* already went through review; the
#     guardrail's job at replay time is to catch drift (an unexpected page,
#     an unexpected element), not to re-litigate a decision already made.
#
# The live, unattended agent loop (agent/loop.py, driven live today via
# scripts/record_capability.py's discovery recording) never uses this -- an
# LLM improvising freely against a site it hasn't been
# reviewed against is exactly the case DEFAULT_RULES' blanket login/checkout
# block exists for. Payment/order-placement and account deletion stay BLOCKED
# even here: reaching a checkout *review* step is the declared goal this
# capability was built and reviewed for; submitting an order or deleting an
# account never is.
REPLAY_RISK_RULES: list[RiskRule] = [
    RiskRule(
        action_types=["click", "type", "select"],
        level=RiskLevel.BLOCKED,
        url_patterns=["*payment*", "*delete_account*", "*/signup*"],
        reason="payment submission and account deletion/signup are blocked outright even in a reviewed capability -- no declared step should ever need them",
    ),
    RiskRule(
        action_types=["click"],
        level=RiskLevel.REVERSIBLE,
        url_patterns=["*/products*", "*/product_details*", "*/view_cart*", "*/checkout*"],
        reason="adding/removing cart items and reaching checkout review is reversible",
    ),
]


def classify(action_type: str, url: str, rules: list[RiskRule] = DEFAULT_RULES) -> RiskRule | None:
    import fnmatch

    for rule in rules:
        if action_type not in rule.action_types:
            continue
        if rule.url_patterns is None or any(fnmatch.fnmatch(url, p) for p in rule.url_patterns):
            return rule
    return None  # no rule matched -> caller decides the default (this build treats unmatched as REVERSIBLE)
