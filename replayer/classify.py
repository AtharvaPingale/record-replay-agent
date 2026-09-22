"""The result taxonomy (REPORT.md Section 3): every replay returns exactly one of
success / business_outcome / recoverable / hard_failure, never conflated.

`business_outcome` and `recoverable` are matched *explicitly* against patterns
declared on the artifact -- they are never inferred from "something didn't work."
This is the mechanism that prevents the "no such member"/"no matching product"
-> treated-as-crash mistake: a business outcome and a hard failure look identical
at the level of "the checkpoint wasn't met" unless you check the declared patterns
first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from artifact.schema import Artifact, Checkpoint
from surfaces.base import Surface


class ReplayResultType(str, Enum):
    SUCCESS = "success"
    BUSINESS_OUTCOME = "business_outcome"
    RECOVERABLE = "recoverable"
    HARD_FAILURE = "hard_failure"


@dataclass
class ReplayResult:
    type: ReplayResultType
    detail: dict[str, Any] = field(default_factory=dict)


def _checkpoint_matches(checkpoint: Checkpoint, page_content: str, url: str, surface: Surface | None = None) -> bool:
    if checkpoint.only_when_url_contains is not None and checkpoint.only_when_url_contains not in url:
        # Page-scoped pattern (business_outcome/recoverable_pattern only --
        # see Checkpoint.only_when_url_contains) being evaluated against a
        # page it was never declared for. Not a match, regardless of what its
        # own type/value would otherwise say -- see the field's docstring for
        # what this prevents.
        return False
    if checkpoint.type == "text_present":
        return checkpoint.value.lower() in page_content.lower()
    if checkpoint.type == "text_absent":
        # For a target that renders no explicit "no results" message at all --
        # an earlier discovery target this repo no longer carries evidence for
        # was exactly this: the only real signal a search matched nothing was
        # that the "add-to-cart" affordance never rendered, so the absence of
        # something else had to stand in for a declared negative state.
        # Modeling that as an explicit absence-check (rather than inferring it
        # from "locate() failed", which is indistinguishable from a genuinely
        # broken page) is what keeps it a declared business outcome instead of
        # a guessed one. A target that *does* render an explicit not-found
        # message doesn't need this type at all -- its own not-found outcome
        # can use text_present instead (see artifact/schema.py's Checkpoint
        # docstring for a related case this type's page-scoping was added to
        # handle).
        return checkpoint.value.lower() not in page_content.lower()
    if checkpoint.type == "url_contains":
        return checkpoint.value in url
    if checkpoint.type == "element_visible":
        # A real *visibility* check via the live surface, not a string search
        # on raw HTML. Added after finding live that text_present is unsafe
        # as a success checkpoint whenever the target
        # markup (e.g. a Bootstrap modal) is always present in the DOM and only
        # CSS-hidden until acted on -- a naive text search would report success
        # on a page nothing was ever done to. Falls back to a (still-honest)
        # False, never to the unsafe text-search, if no surface is available.
        return surface.is_visible(checkpoint.value) if surface is not None else False
    if checkpoint.type == "selector_visible":
        return surface.is_selector_visible(checkpoint.value) if surface is not None else False
    return False


def classify_outcome(artifact: Artifact, page_content: str, url: str, surface: Surface | None = None) -> ReplayResult:
    """Called after a step fails to progress normally (checkpoint not yet met, or
    an unexpected state appeared). Checks declared business_outcomes first, then
    recoverable_patterns, and only falls through to hard_failure if neither the
    artifact's own declared expected-negative states nor its known transient
    friction patterns match."""
    for outcome in artifact.business_outcomes:
        if _checkpoint_matches(outcome.match, page_content, url, surface):
            return ReplayResult(
                ReplayResultType.BUSINESS_OUTCOME,
                {"name": outcome.name, "alert_human": outcome.alert_human},
            )

    for pattern in artifact.recoverable_patterns:
        if _checkpoint_matches(pattern.match, page_content, url, surface):
            return ReplayResult(
                ReplayResultType.RECOVERABLE, {"name": pattern.name, "recovery": pattern.recovery}
            )

    return ReplayResult(ReplayResultType.HARD_FAILURE, {"reason": "no declared pattern matched"})
