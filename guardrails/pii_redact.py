"""Pre-LLM PII redaction for screenshots and OCR-derived prompt text -- masks
US banking/SSN/currency/email content before it ever reaches a vision-capable
decider's image payload or any decider's text prompt, local or hosted.
(Email was added after a real gap found live: the general log-redaction path,
guardrails/redact.py's DEFAULT_PATTERNS, already masked email in what gets
*written to disk*, which made it easy to miss that this module -- what
actually reaches the model -- had no email pattern of its own at all.)

Deliberately reuses locator/ocr.py's existing PaddleOCR pass (already run
once per observe() for decider grounding -- see Observation.ocr_excerpt,
agent/deciders.py's _ocr_prompt_section) rather than standing up a second OCR
engine (Tesseract) or an NLP stack (spaCy/presidio): this project already has
OCR word boxes with text and coordinates on every turn, and the actual
redaction need here is pattern matching, not named-entity recognition. A
heavier stack would duplicate a working engine already proven live in this
codebase for no real accuracy gain on patterns that are all regex-shaped
(routing/account numbers, SSNs, currency amounts), not NLP-shaped.

Separate from guardrails/redact.py's DEFAULT_PATTERNS, deliberately: that
dict is applied to EVERY log/evidence-bundle detail (telemetry/log.py,
telemetry/bundle.py) via redact_value, and a broad 8-17-digit "account
number" pattern there would over-redact ordinary log fields (step numbers,
ms values, pixel coordinates) that have nothing to do with a screenshot.
BANKING_PATTERNS here is only ever used by this module's own two entry
points, never by the general log-redaction path.

Honest tradeoff, not hidden: a bank routing number is indistinguishable from
any other bare 9-digit number -- a real limitation of the pattern the
original plan asked for, not one this module invented. On a non-financial
target (an e-commerce or video site, say) this will mask
things that aren't actually PII (an order number, a view count). Failing
toward over-redaction rather than under-redaction is the deliberate choice
for a privacy guardrail.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only for the type hint below -- locator/ocr.py's OcrWord isn't imported
    # at module load (paddleocr is an optional extra; see the deferred import
    # inside redact_screenshot_bytes), and `from __future__ import annotations`
    # above means this file never needed it at runtime -- ruff's F821 caught
    # that the hint was still wrong, not just deferred, since nothing anywhere
    # in the file actually named the type.
    from locator.ocr import OcrWord

BANKING_PATTERNS: dict[str, re.Pattern] = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "bank_routing_number": re.compile(r"\b\d{9}\b"),
    "bank_account_number": re.compile(r"\b\d{8,17}\b"),
    "currency_amount": re.compile(r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b"),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
}

_EMPTY_EXEMPT: frozenset[str] = frozenset()


def _non_exempt_matches(text: str, exempt: frozenset[str]):
    """Yields (name, match) for every BANKING_PATTERNS hit in `text` whose
    exact matched substring isn't in `exempt`. A caller's own declared input
    (the account ID it explicitly asked to search for) isn't a new exposure
    when it's shown back to the model -- the caller already has it. Only
    *other*, incidental PII-shaped content should still be masked. See
    redact_pii_text/redact_screenshot_bytes for how this is used."""
    for name, pattern in BANKING_PATTERNS.items():
        for m in pattern.finditer(text):
            if m.group(0) not in exempt:
                yield name, m


def redact_pii_text(text: str, exempt: frozenset[str] = _EMPTY_EXEMPT) -> str:
    """Masks banking/SSN/currency/email-shaped substrings in any text headed
    into a decider's prompt -- both the raw DOM excerpt and the OCR-derived
    excerpt (Observation.dom_excerpt / ocr_excerpt) go through this before
    agent/deciders.py builds a prompt from them. Found live, verifying the
    original OCR-only version of this module against a real decider call:
    the DOM excerpt carries the exact same text as plain HTML, completely
    unredacted, in every prompt regardless of what the OCR/image side
    masked -- for a real page, that's the bigger and more complete leak of
    the two, not a minor one. redact_screenshot_bytes below is the image
    half of this module.

    `exempt`: literal values to leave unmasked -- e.g. an account ID the
    caller's own goal already named as the thing to search for. Found live:
    a goal shaped like the brief's own "look up a user's balance by account
    ID" example needs the model to actually read the account number back off
    the page to confirm it found the right row; blindly masking it made that
    impossible. Matched by exact substring equality against what a pattern
    found, not a second regex -- this never widens what counts as
    non-PII, it only carves out specific values the caller already knows.
    """
    out = text
    for name, pattern in BANKING_PATTERNS.items():
        out = pattern.sub(lambda m: m.group(0) if m.group(0) in exempt else f"[REDACTED:{name}]", out)
    return out


def redact_screenshot_bytes(
    source: str | Path | bytes, exempt: frozenset[str] = _EMPTY_EXEMPT, words=None,
) -> tuple[bytes, int]:
    """Paints a solid black box over every OCR word/line containing a
    BANKING_PATTERNS match not in `exempt` (see redact_pii_text) -- and over
    every word sharing a row with one, which is how a customer's *name* gets
    caught without entity recognition (see the proximity pass below).
    Returns (redacted PNG bytes, count of regions redacted).

    `source` is a path or the raw PNG bytes. `words`, if given, are OCR results
    already computed for this exact image (locator/ocr.py's OcrWord), so the
    caller can run one OCR pass and use it for both the decider's text excerpt
    and this redaction -- surfaces/web.py does exactly that on every observe(). Always a full rectangle over the OCR-detected box,
    never a partial mask -- a half-covered account number is not safe, and a
    box containing both an exempt value and a genuinely sensitive one (e.g.
    the searched-for account number next to that same row's balance) still
    gets redacted, since only the exempt value itself is cleared, not the
    whole box's PII-ness.

    Raises on any real failure (OCR engine unavailable, corrupt image) --
    deliberately: the caller (agent/deciders.py) must fail closed and drop
    the image for that turn entirely rather than silently sending the raw,
    unredacted one. Fail-safe, not fail-open.
    """
    from PIL import Image, ImageDraw

    from locator.ocr import run_ocr, run_ocr_bytes  # deferred: same optional-dependency pattern as locator/ocr.py itself

    if words is None:
        words = run_ocr_bytes(source) if isinstance(source, bytes) else run_ocr(str(source))
    image = Image.open(io.BytesIO(source) if isinstance(source, bytes) else source).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Pass 1: words that are PII-shaped in their own right.
    flagged = {i for i, w in enumerate(words) if any(True for _ in _non_exempt_matches(w.text, exempt))}

    # Pass 2: redaction by proximity. A person's name is not regex-shaped, so
    # no pattern above will ever catch "Mike Wilson" -- but in every layout that
    # matters here it sits on the same row as the email, account number and
    # balance that *are* caught. Anything sharing a row with a PII match is
    # treated as part of the same record and painted out too, unless the word
    # itself carries an exempt value (the account the caller asked about stays
    # visible so the run can be verified). Found live: with pattern-only
    # redaction, a bank's user list came out with every number and email
    # black and every customer's name and @handle in the clear beside them.
    #
    # This is a heuristic and it errs toward over-redaction on purpose: a
    # "View Details" button or a creation date on a customer's row goes black
    # with the rest of the row. UI chrome on rows with no PII (column headers,
    # page titles, navigation) is untouched, because nothing there triggers
    # pass 1. It does not help a non-tabular layout where a name stands alone
    # -- that is the case that genuinely needs entity recognition.
    def _same_row(a: OcrWord, b: OcrWord) -> bool:
        overlap = min(a.bbox[3], b.bbox[3]) - max(a.bbox[1], b.bbox[1])
        shorter = min(a.bbox[3] - a.bbox[1], b.bbox[3] - b.bbox[1])
        return shorter > 0 and overlap > 0.5 * shorter

    by_proximity = {
        j
        for i in flagged
        for j, w in enumerate(words)
        if j not in flagged
        and _same_row(words[i], w)
        and not any(e in w.text for e in exempt)
    }

    redacted = 0
    for i in flagged | by_proximity:
        draw.rectangle(words[i].bbox, fill="black")
        redacted += 1

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue(), redacted


def persist_redacted_screenshot(raw_png: bytes, out_path: str | Path, exempt: frozenset[str] = _EMPTY_EXEMPT):
    """The one way a screenshot gets written to disk. Returns
    (written: bool, redacted_count: int, words: list[OcrWord] | None).

    Runs a single OCR pass over the in-memory capture, paints every non-exempt
    PII-shaped region black, and writes *that*. The raw capture never reaches
    the filesystem. Fail-closed, deliberately: if OCR is unavailable or the
    engine fails, nothing is written at all -- `written` is False and the
    caller records the screenshot as withheld. A screenshot with a member's
    balance, email and account number in it is exactly the "raw sensitive data"
    Section 3.4 says must not be persisted, and a run's screenshots are the
    most-persisted, most-shared artifact it produces (they become evidence/).
    Saving the raw image "just this once" because redaction could not run would
    be the fail-open path this whole module exists to refuse.

    `words` is returned so the caller can build its OCR text excerpt from the
    same pass rather than paying for a second one."""
    from locator.ocr import run_ocr_bytes

    words = run_ocr_bytes(raw_png)
    png, count = redact_screenshot_bytes(raw_png, exempt=exempt, words=words)
    Path(out_path).write_bytes(png)
    return True, count, words
