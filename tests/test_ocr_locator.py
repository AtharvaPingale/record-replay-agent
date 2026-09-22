"""Unit tests for locator/ocr.py's pure matching logic (find_best_match,
to_fraction). These never import paddleocr or run real inference -- they
build OcrWord values directly, the same shape run_ocr() would produce, so the
suite stays fast and has no dependency on the optional `ocr` extra being
installed. The one thing genuinely requiring the real engine (does OCR
actually read real UI pixels correctly) is not exercised in this submission --
see REPORT.md Section 4 for why.
"""

from __future__ import annotations

from locator.ocr import OcrWord, find_best_match, to_fraction


def word(text, confidence, bbox):
    return OcrWord(text=text, confidence=confidence, bbox=bbox)


def test_matches_exact_text_case_insensitively():
    words = [word("Add to cart", 0.98, (100, 200, 260, 230))]
    match = find_best_match(words, "add to cart")
    assert match is not None
    assert match.text == "Add to cart"


def test_matches_when_target_is_a_substring_of_the_ocr_line():
    # A target of "Cart" should match an OCR line "Shopping Cart" -- OCR
    # groups words into lines, not always into the exact fragment a caller asks for.
    words = [word("Shopping Cart", 0.95, (139, 138, 248, 163))]
    match = find_best_match(words, "Cart")
    assert match is not None


def test_matches_when_the_ocr_line_is_a_substring_of_the_target():
    # The reverse direction: a target phrase containing extra words around an
    # OCR line that was detected as just the core phrase.
    words = [word("Checkout", 0.99, (50, 50, 150, 80))]
    match = find_best_match(words, "proceed to checkout")
    assert match is not None


def test_no_match_returns_none_rather_than_guessing():
    words = [word("Home", 0.99, (0, 0, 50, 20))]
    assert find_best_match(words, "Delete Account") is None


def test_low_confidence_match_is_discarded():
    # A misread shouldn't drive a real click -- below min_confidence, a
    # textually-matching word is treated as no match at all.
    words = [word("Add to cart", 0.2, (100, 200, 260, 230))]
    assert find_best_match(words, "add to cart", min_confidence=0.5) is None


def test_low_confidence_match_is_kept_when_threshold_is_lowered():
    words = [word("Add to cart", 0.4, (100, 200, 260, 230))]
    assert find_best_match(words, "add to cart", min_confidence=0.3) is not None


def test_single_candidate_wins_even_without_a_hint():
    words = [word("Home", 0.9, (0, 0, 50, 20))]
    match = find_best_match(words, "Home")
    assert match is not None
    assert match.bbox == (0, 0, 50, 20)


def test_multiple_matches_with_no_hint_prefers_highest_confidence():
    # "Home" legitimately appears twice on a real page (top nav + breadcrumb) --
    # this is exactly the disambiguation case the module's docstring describes.
    words = [
        word("Home", 0.80, (69, 137, 126, 163)),
        word("Home", 0.99, (475, 27, 546, 48)),
    ]
    match = find_best_match(words, "Home")
    assert match.confidence == 0.99


def test_multiple_matches_with_a_hint_prefers_the_closest_one():
    words = [
        word("Home", 0.99, (475, 27, 546, 48)),  # top nav, far from the hint
        word("Home", 0.80, (69, 137, 126, 163)),  # breadcrumb, near the hint
    ]
    viewport = (1280, 900)
    hint_frac = (100 / 1280, 150 / 900)  # right next to the breadcrumb "Home"
    match = find_best_match(words, "Home", hint_frac=hint_frac, viewport_size=viewport)
    assert match.bbox == (69, 137, 126, 163)


def test_hint_is_ignored_when_viewport_size_is_missing():
    # A hint without a viewport size can't be converted to pixels -- falls
    # back to highest-confidence rather than crashing or guessing.
    words = [
        word("Home", 0.80, (69, 137, 126, 163)),
        word("Home", 0.99, (475, 27, 546, 48)),
    ]
    match = find_best_match(words, "Home", hint_frac=(0.1, 0.2), viewport_size=None)
    assert match.confidence == 0.99


def test_center_property_is_the_bbox_midpoint():
    w = word("X", 1.0, (100, 200, 300, 240))
    assert w.center == (200, 220)


def test_to_fraction_converts_center_to_a_viewport_fraction():
    w = word("X", 1.0, (100, 200, 300, 240))
    fx, fy = to_fraction(w, (1280, 900))
    assert fx == 200 / 1280
    assert fy == 220 / 900
