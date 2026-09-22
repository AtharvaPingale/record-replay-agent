"""Unit tests for guardrails/pii_redact.py -- the pre-LLM screenshot/prompt-text
redaction pipeline. redact_pii_text is pure regex matching (no OCR engine
needed) and is used for both the DOM excerpt and the OCR excerpt in every
decider's prompt (agent/deciders.py's _redacted_dom_excerpt/_ocr_prompt_section).
redact_screenshot_bytes needs OCR word boxes, but not a real OCR engine --
locator.ocr.run_ocr is monkeypatched with synthetic OcrWord output, the same
pattern tests/test_ocr_locator.py already uses for the matching logic itself.
"""

from __future__ import annotations

import io

import pytest

from PIL import Image

from guardrails.pii_redact import redact_pii_text, redact_screenshot_bytes
from locator.ocr import OcrWord


def test_redacts_a_dashed_ssn():
    assert redact_pii_text("SSN: 123-45-6789") == "SSN: [REDACTED:ssn]"


def test_redacts_a_bank_routing_number():
    assert redact_pii_text("Routing 021000021") == "Routing [REDACTED:bank_routing_number]"


def test_redacts_a_bank_account_number():
    assert redact_pii_text("Account 123456789012") == "Account [REDACTED:bank_account_number]"


def test_redacts_a_currency_amount():
    assert redact_pii_text("Balance: $12,345.67") == "Balance: [REDACTED:currency_amount]"


def test_ordinary_text_passes_through_unchanged():
    # This is exactly the honesty check: nothing shaped like banking/SSN/
    # currency data should ever get touched.
    text = "Submit button, product name, some ordinary label"
    assert redact_pii_text(text) == text


def test_redacts_an_email_address():
    # Real gap found live: the general log-redaction path (guardrails/redact.py)
    # already masked email in what gets *written to disk*, which made it easy
    # to miss that this module -- what actually reaches the model -- had no
    # email pattern of its own until this was added.
    assert redact_pii_text("Contact: jane.smith@example.com") == "Contact: [REDACTED:email]"


def test_exempt_value_is_left_unmasked_but_other_pii_on_the_same_line_still_is():
    # The brief's own "look up a user's balance by account ID" example: the
    # caller already knows the account ID it asked to search for, so showing
    # it back to the model isn't a new exposure -- but an unrelated balance
    # sitting right next to it in the same text still needs to be masked.
    text = "Account 1234567890 Balance $500.00"
    out = redact_pii_text(text, exempt=frozenset({"1234567890"}))
    assert "1234567890" in out
    assert "$500.00" not in out
    assert "[REDACTED:currency_amount]" in out


def test_exempt_only_matches_the_exact_value_not_a_substring():
    # A near-miss (one digit different) must still be redacted -- exemption
    # is exact-match against the caller's own declared value, not fuzzy.
    out = redact_pii_text("Account 1234567891", exempt=frozenset({"1234567890"}))
    assert out == "Account [REDACTED:bank_account_number]"


def _make_png_bytes(size=(100, 40), color="white") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


def test_redact_screenshot_bytes_masks_only_pii_shaped_regions(tmp_path, monkeypatch):
    shot_path = tmp_path / "shot.png"
    shot_path.write_bytes(_make_png_bytes())

    words = [
        OcrWord(text="Add to cart", confidence=0.99, bbox=(0, 0, 40, 10)),
        OcrWord(text="123-45-6789", confidence=0.99, bbox=(50, 20, 90, 30)),
    ]
    monkeypatch.setattr("locator.ocr.run_ocr", lambda path: words)

    png_bytes, redacted_count = redact_screenshot_bytes(str(shot_path))

    assert redacted_count == 1  # only the SSN-shaped word, not "Add to cart"
    result = Image.open(io.BytesIO(png_bytes))
    # The SSN's bbox center is painted black...
    assert result.getpixel((70, 25)) == (0, 0, 0)
    # ...but the non-PII word's region is untouched (still the original white).
    assert result.getpixel((20, 5)) == (255, 255, 255)


def test_redact_screenshot_bytes_with_no_pii_leaves_the_image_untouched(tmp_path, monkeypatch):
    shot_path = tmp_path / "shot.png"
    shot_path.write_bytes(_make_png_bytes())
    monkeypatch.setattr("locator.ocr.run_ocr", lambda path: [OcrWord(text="Add to cart", confidence=0.99, bbox=(0, 0, 40, 10))])

    _png_bytes, redacted_count = redact_screenshot_bytes(str(shot_path))

    assert redacted_count == 0


def test_redact_screenshot_bytes_does_not_mask_an_exempt_value(tmp_path, monkeypatch):
    shot_path = tmp_path / "shot.png"
    shot_path.write_bytes(_make_png_bytes())
    words = [OcrWord(text="Account 1234567890", confidence=0.99, bbox=(0, 0, 40, 10))]
    monkeypatch.setattr("locator.ocr.run_ocr", lambda path: words)

    png_bytes, redacted_count = redact_screenshot_bytes(str(shot_path), exempt=frozenset({"1234567890"}))

    assert redacted_count == 0
    result = Image.open(io.BytesIO(png_bytes))
    assert result.getpixel((20, 5)) == (255, 255, 255)  # left visible, not painted over


# -- persist_redacted_screenshot: the one way a screenshot reaches disk --------


def test_persisted_screenshot_is_the_redacted_image_not_the_raw_capture(tmp_path, monkeypatch):
    from guardrails.pii_redact import persist_redacted_screenshot

    words = [OcrWord(text="$15,000.00", confidence=0.99, bbox=(50, 20, 90, 30))]
    monkeypatch.setattr("locator.ocr.run_ocr", lambda path: words)
    out = tmp_path / "001.png"

    written, count, got_words = persist_redacted_screenshot(_make_png_bytes(), out)

    assert written is True and count == 1 and got_words == words
    saved = Image.open(out)
    assert saved.getpixel((70, 25)) == (0, 0, 0), "the balance region is black in the SAVED file"
    assert saved.getpixel((20, 5)) == (255, 255, 255)


def test_an_exempt_input_stays_visible_in_the_saved_screenshot(tmp_path, monkeypatch):
    # The caller's own account number: they supplied it and need it to verify
    # the run; every other member's data on the same page is still painted out.
    from guardrails.pii_redact import persist_redacted_screenshot

    words = [
        OcrWord(text="1234567890", confidence=0.99, bbox=(0, 0, 40, 10)),
        OcrWord(text="2345678901", confidence=0.99, bbox=(50, 20, 90, 30)),
    ]
    monkeypatch.setattr("locator.ocr.run_ocr", lambda path: words)
    out = tmp_path / "001.png"

    _, count, _ = persist_redacted_screenshot(_make_png_bytes(), out, exempt=frozenset({"1234567890"}))

    assert count == 1
    saved = Image.open(out)
    assert saved.getpixel((20, 5)) == (255, 255, 255), "the requested account stays visible"
    assert saved.getpixel((70, 25)) == (0, 0, 0), "the other member's account is painted out"


def test_when_redaction_cannot_run_nothing_is_written(tmp_path, monkeypatch):
    # Fail-closed: no OCR engine means no screenshot, never a raw one.
    from guardrails.pii_redact import persist_redacted_screenshot

    def _no_ocr(path):
        raise ImportError("No module named 'paddleocr'")

    monkeypatch.setattr("locator.ocr.run_ocr", _no_ocr)
    out = tmp_path / "001.png"

    with pytest.raises(ImportError):
        persist_redacted_screenshot(_make_png_bytes(), out)
    assert not out.exists(), "a raw screenshot must never reach disk"


def test_redact_screenshot_bytes_accepts_bytes_and_precomputed_words(monkeypatch):
    # One OCR pass per observe: the surface computes words once and hands them
    # to both the text excerpt and this redaction. No OCR call may happen here.
    monkeypatch.setattr("locator.ocr.run_ocr", lambda path: (_ for _ in ()).throw(AssertionError("must not OCR again")))
    monkeypatch.setattr("locator.ocr.run_ocr_bytes", lambda b: (_ for _ in ()).throw(AssertionError("must not OCR again")))
    words = [OcrWord(text="123-45-6789", confidence=0.99, bbox=(50, 20, 90, 30))]

    png, count = redact_screenshot_bytes(_make_png_bytes(), words=words)

    assert count == 1
    assert Image.open(io.BytesIO(png)).getpixel((70, 25)) == (0, 0, 0)


# -- redaction by proximity: names are caught by the row they sit on -----------


def _row(y0, y1, *items):
    """OcrWords laid out left to right on one row band."""
    return [OcrWord(text=t, confidence=0.99, bbox=(x, y0, x + 40, y1)) for t, x in zip(items, range(0, 40 * len(items), 40))]


def test_a_name_on_the_same_row_as_pii_is_redacted_without_ner(monkeypatch):
    # Found live: pattern-only redaction left "Mike Wilson" in the clear beside
    # his blacked-out email, account number and balance.
    header = _row(0, 10, "USER", "EMAIL", "BALANCE")
    mike = _row(20, 30, "Mike Wilson", "mike@example.com", "$8,500.75")
    monkeypatch.setattr("locator.ocr.run_ocr", lambda path: header + mike)

    png, count = redact_screenshot_bytes(_make_png_bytes(size=(200, 40)), words=header + mike)

    img = Image.open(io.BytesIO(png))
    assert img.getpixel((20, 25)) == (0, 0, 0), "the name goes with its row"
    assert img.getpixel((60, 25)) == (0, 0, 0) and img.getpixel((100, 25)) == (0, 0, 0)
    assert img.getpixel((20, 5)) == (255, 255, 255), "a header row with no PII is untouched"
    assert count == 3


def test_the_exempt_value_stays_visible_even_on_a_redacted_row(monkeypatch):
    # The requested account number is what lets a reviewer verify the run; the
    # rest of that same customer's row is still painted out.
    row = _row(20, 30, "John Doe", "1234567890", "$15,000.00")
    monkeypatch.setattr("locator.ocr.run_ocr", lambda path: row)

    png, count = redact_screenshot_bytes(_make_png_bytes(size=(200, 40)), words=row, exempt=frozenset({"1234567890"}))

    img = Image.open(io.BytesIO(png))
    assert img.getpixel((60, 25)) == (255, 255, 255), "exempt account number visible"
    assert img.getpixel((20, 25)) == (0, 0, 0), "but the name on that row is still redacted"
    assert img.getpixel((100, 25)) == (0, 0, 0)
    assert count == 2


def test_rows_with_no_pii_are_left_entirely_alone(monkeypatch):
    words = _row(0, 10, "Dashboard", "User Management") + _row(20, 30, "All Users (3)")
    monkeypatch.setattr("locator.ocr.run_ocr", lambda path: words)

    _png, count = redact_screenshot_bytes(_make_png_bytes(size=(200, 40)), words=words)

    assert count == 0
