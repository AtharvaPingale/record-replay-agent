"""Unit tests for the deterministic result taxonomy (replayer/classify.py).

This is the load-bearing piece of the whole replay path -- it's what makes
"business outcome vs. hard failure" a declared decision instead of a guess.
No browser needed: `_checkpoint_matches` and `classify_outcome` take page
content, a URL, and a surface as plain arguments, so they're testable as pure
functions with a fake `Surface`.
"""

from __future__ import annotations

from artifact.schema import (
    Artifact,
    Checkpoint,
    DeclaredOutcome,
    RecoverablePattern,
    Target,
)
from replayer.classify import ReplayResultType, _checkpoint_matches, classify_outcome


class FakeSurface:
    """Stands in for surfaces.web.WebSurface's is_visible() -- classify_outcome
    only ever calls this one method on a surface."""

    def __init__(self, visible_texts: set[str]):
        self._visible = visible_texts

    def is_visible(self, text: str) -> bool:
        return text in self._visible

    def is_selector_visible(self, selector: str) -> bool:
        return selector in self._visible


def make_artifact(business_outcomes=None, recoverable_patterns=None) -> Artifact:
    return Artifact(
        artifact_id="test-artifact",
        target=Target(base_url_pattern="https://example.com/*"),
        input_schema={},
        output_schema={},
        checkpoint=Checkpoint(type="element_visible", value="Order Confirmed"),
        business_outcomes=business_outcomes or [],
        recoverable_patterns=recoverable_patterns or [],
        steps=[],
    )


# -- _checkpoint_matches: each checkpoint type in isolation --------------------


def test_text_present_matches_case_insensitively():
    assert _checkpoint_matches(Checkpoint(type="text_present", value="Added!"), "<div>added! to cart</div>", "")


def test_text_present_does_not_match_when_absent():
    assert not _checkpoint_matches(Checkpoint(type="text_present", value="Added!"), "<div>nothing here</div>", "")


def test_text_absent_is_the_inverse_of_text_present():
    checkpoint = Checkpoint(type="text_absent", value="add-to-cart")
    assert _checkpoint_matches(checkpoint, "<div>no such button</div>", "")
    assert not _checkpoint_matches(checkpoint, "<button>add-to-cart</button>", "")


def test_url_contains_matches_substring():
    checkpoint = Checkpoint(type="url_contains", value="/checkout")
    assert _checkpoint_matches(checkpoint, "", "https://example.com/checkout?step=review")
    assert not _checkpoint_matches(checkpoint, "", "https://example.com/cart")


def test_element_visible_delegates_to_the_real_surface_not_raw_html():
    # This is the exact bug the discovery run found: a naive text-in-page.content()
    # check reports success on a page nothing was done to, because the confirmation
    # modal's markup is present in the static template, just CSS-hidden. The fix is
    # that element_visible never looks at page_content at all -- only the surface.
    checkpoint = Checkpoint(type="element_visible", value="Added!")
    page_content_with_hidden_modal = "<div class='modal' style='display:none'>Added!</div>"

    assert not _checkpoint_matches(checkpoint, page_content_with_hidden_modal, "", FakeSurface(set()))
    assert _checkpoint_matches(checkpoint, page_content_with_hidden_modal, "", FakeSurface({"Added!"}))


def test_element_visible_is_honestly_false_with_no_surface_available():
    # Never falls back to the unsafe text-search -- an absent surface means
    # "can't verify," not "assume true."
    checkpoint = Checkpoint(type="element_visible", value="Added!")
    assert not _checkpoint_matches(checkpoint, "Added!", "", surface=None)


# -- classify_outcome: the fixed decision order --------------------------------


def test_business_outcome_is_checked_before_recoverable_and_before_hard_failure():
    artifact = make_artifact(
        business_outcomes=[DeclaredOutcome(name="no_matching_product", match=Checkpoint(type="text_absent", value="add-to-cart"))],
        recoverable_patterns=[RecoverablePattern(name="popup", match=Checkpoint(type="text_present", value="subscribe"), recovery="dismiss")],
    )
    # Page content matches the business outcome (no add-to-cart button) and
    # happens to also contain "subscribe" -- business_outcome must win, since
    # it's checked first and a business outcome is never "actually a recoverable
    # popup in disguise."
    result = classify_outcome(artifact, "<div>subscribe now</div>", "https://example.com/search")
    assert result.type == ReplayResultType.BUSINESS_OUTCOME
    assert result.detail["name"] == "no_matching_product"


def test_business_outcome_scoped_to_a_url_does_not_fire_on_a_later_page():
    # Real bug found live: "no_matching_product" means "add-to-cart" is
    # absent right after a *search*, checked on the products page. But
    # classify_outcome runs after ANY step's failure, anywhere in the flow --
    # a later, unrelated hard_failure on the cart/checkout page also finds
    # "add-to-cart" absent there (it's simply the wrong page for that text to
    # ever appear), and without a page scope this misclassified a genuine
    # checkout-button hard_failure as "no_matching_product" instead.
    artifact = make_artifact(
        business_outcomes=[
            DeclaredOutcome(
                name="no_matching_product",
                match=Checkpoint(type="text_absent", value="add-to-cart", only_when_url_contains="/products"),
            )
        ],
    )
    # Same page content (no "add-to-cart" text) but a different URL --
    # already past search, deep in checkout. Must NOT match.
    result = classify_outcome(artifact, "<div>Review Your Order</div>", "https://example.com/view_cart")
    assert result.type == ReplayResultType.HARD_FAILURE

    # The exact same pattern DOES still match on the page it's actually scoped to.
    result_on_products_page = classify_outcome(artifact, "<div>no results</div>", "https://example.com/products")
    assert result_on_products_page.type == ReplayResultType.BUSINESS_OUTCOME


def test_business_outcome_detail_carries_the_declared_alert_human_flag():
    # alert_human defaults to False, and a routine outcome (no_matching_product)
    # should never need to opt in -- most business outcomes need no one's
    # attention. An outcome that does declare alert_human=True must carry that
    # through to the ReplayResult's detail, since escalation/detect.py's
    # should_alert_human() only ever looks at this field, never at the name.
    routine = make_artifact(
        business_outcomes=[DeclaredOutcome(name="no_matching_product", match=Checkpoint(type="text_absent", value="add-to-cart"))]
    )
    result = classify_outcome(routine, "<div>no button here</div>", "https://example.com/search")
    assert result.detail["alert_human"] is False

    flagged = make_artifact(
        business_outcomes=[
            DeclaredOutcome(
                name="fraud_hold",
                match=Checkpoint(type="text_present", value="under review"),
                alert_human=True,
            )
        ]
    )
    result = classify_outcome(flagged, "<div>Your order is under review</div>", "https://example.com/checkout")
    assert result.detail["name"] == "fraud_hold"
    assert result.detail["alert_human"] is True


def test_recoverable_pattern_matches_when_no_business_outcome_does():
    artifact = make_artifact(
        recoverable_patterns=[RecoverablePattern(name="popup", match=Checkpoint(type="text_present", value="subscribe"), recovery="press Escape")]
    )
    result = classify_outcome(artifact, "<div>please subscribe</div>", "https://example.com/products")
    assert result.type == ReplayResultType.RECOVERABLE
    assert result.detail["name"] == "popup"
    assert result.detail["recovery"] == "press Escape"


def test_falls_through_to_hard_failure_when_nothing_declared_matches():
    # This is the safety property that matters most: an *undeclared* state is
    # never silently treated as a business outcome or quietly recovered from --
    # it becomes hard_failure by construction, which is what triggers escalation.
    # Page content here must satisfy neither declared pattern: "add-to-cart" is
    # present (so text_absent doesn't match) and "subscribe" is absent (so the
    # recoverable pattern doesn't match either).
    artifact = make_artifact(
        business_outcomes=[DeclaredOutcome(name="no_matching_product", match=Checkpoint(type="text_absent", value="add-to-cart"))],
        recoverable_patterns=[RecoverablePattern(name="popup", match=Checkpoint(type="text_present", value="subscribe"), recovery="dismiss")],
    )
    result = classify_outcome(artifact, "<button>add-to-cart</button><div>500 internal server error</div>", "https://example.com/products")
    assert result.type == ReplayResultType.HARD_FAILURE


def test_no_declared_patterns_at_all_always_falls_through():
    artifact = make_artifact()
    result = classify_outcome(artifact, "<div>anything</div>", "https://example.com/")
    assert result.type == ReplayResultType.HARD_FAILURE


def test_selector_visible_asks_the_surface_for_a_selector_not_text():
    # Found live: a model declared '.table-row:has-text("1234567890")' as an
    # element_visible checkpoint, which matches TEXT, so it never resolved and
    # a correct recording hard-failed at replay. selector_visible is the
    # explicit home for the structural case.
    from replayer.classify import _checkpoint_matches

    surface = FakeSurface({'.table-row:has-text("1234567890")'})
    assert _checkpoint_matches(Checkpoint(type="selector_visible", value='.table-row:has-text("1234567890")'), "", "", surface)
    assert not _checkpoint_matches(Checkpoint(type="element_visible", value='.table-row:has-text("1234567890")'), "", "", FakeSurface(set()))
    assert not _checkpoint_matches(Checkpoint(type="selector_visible", value=".x"), "", "", None), "no surface -> honest False, never a text search"
