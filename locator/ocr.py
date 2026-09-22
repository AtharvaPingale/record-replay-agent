"""OCR-based text location -- the DOM-less fallback described in REPORT.md
Section 4. Where `locator/locate.py`'s "text" candidate resolves against DOM
text nodes, "ocr_text" resolves against words actually rendered to pixels, by
running OCR on a fresh screenshot. This is what makes it usable on a surface
with no usable DOM/accessibility tree at all (a frameset, a canvas-rendered
UI): the resolver doesn't need one.

Engine: PaddleOCR, chosen in REPORT.md Section 4 for its per-box confidence
scores and better accuracy on small/low-contrast UI text than Tesseract. This
module is the only place that imports it -- everything downstream (locate.py,
surfaces/web.py, replayer/executor.py) only ever sees plain (text, confidence,
bbox) tuples, so swapping the engine later (Tesseract, EasyOCR) touches only
this file.

`paddleocr`/`paddlepaddle` are an optional dependency group (`uv sync --extra
ocr`) -- importing this module without them installed raises a clear
ImportError at the point of use, not at import time of anything else in the
codebase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# Found live, in this environment, getting this module working at all: the
# default PaddleOCR config crashes during inference with
# `NotImplementedError: ... ConvertPirAttribute2RuntimeAttribute not support
# ...onednn_instruction.cc` -- an internal Paddle CPU-backend (oneDNN/MKL-DNN)
# incompatibility on this machine, not anything in this codebase. `enable_mkldnn=False`
# avoids the oneDNN code path entirely and inference then works correctly (the
# document-orientation/unwarping/textline-orientation stages are also
# disabled -- this is a screenshot of a UI, not a scanned document, so those
# stages have nothing to correct and only cost latency).
_OCR_KWARGS: dict[str, Any] = dict(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
)

_engine = None  # process-wide singleton: init alone took ~17s live, real enough to matter per-step


def _get_engine():
    global _engine
    if _engine is None:
        from paddleocr import PaddleOCR  # deferred: only this function needs the optional dependency

        _engine = PaddleOCR(**_OCR_KWARGS)
    return _engine


@dataclass
class OcrWord:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1), absolute pixels in the source screenshot

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2, (y0 + y1) / 2)


def run_ocr(screenshot_path: str) -> list[OcrWord]:
    """One OCR pass over a screenshot file. Real, live-verified output shape --
    every discovery run under evidence/ runs this on every turn, both for
    PII redaction (guardrails/pii_redact.py) and the decider's own OCR text
    excerpt: PP-OCRv6 returns `rec_texts` (recognized strings), `rec_scores`
    (confidence 0-1), `rec_boxes` (axis-aligned [x0,y0,x1,y1] per box) -- all
    three the same length, one entry per detected text region."""
    engine = _get_engine()
    result = engine.predict(screenshot_path)
    if not result:
        return []
    page = result[0]
    words = []
    for text, score, box in zip(page["rec_texts"], page["rec_scores"], page["rec_boxes"]):
        x0, y0, x1, y1 = (int(v) for v in box)
        words.append(OcrWord(text=text, confidence=float(score), bbox=(x0, y0, x1, y1)))
    return words


def run_ocr_bytes(png: bytes) -> list[OcrWord]:
    """run_ocr over an in-memory PNG. Exists so a raw screenshot can be OCR'd
    and redacted *before* anything is written to disk -- the unredacted capture
    only ever lives in memory. Goes through a temp file because that is the
    input PaddleOCR is known to handle here; the file is unlinked immediately."""
    import os
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".png")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(png)
        return run_ocr(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def find_best_match(
    words: list[OcrWord],
    target_text: str,
    hint_frac: tuple[float, float] | None = None,
    viewport_size: tuple[int, int] | None = None,
    min_confidence: float = 0.5,
) -> OcrWord | None:
    """Pick the OCR word/line that best matches `target_text` among possibly
    several candidates (a nav label like "Home" can legitimately appear more
    than once on one page).

    Matching is case-insensitive substring, either direction (the OCR line
    might be "Add to cart" for a target of "add to cart", or a target of
    "Cart" should match an OCR line "Shopping Cart"). Below `min_confidence`
    a match is discarded outright -- a low-confidence misread shouldn't drive
    a real click.

    Disambiguation between multiple matches: if the step declared a `hint_frac`
    (a recorded [x_frac, y_frac] point from when the artifact was built), the
    match closest to it wins -- this is the artifact's own memory of *where on
    the page* the target usually is, independent of the exact pixels, which
    tolerates layout drift far better than a stored absolute coordinate would.
    With no hint, the highest-confidence match wins.
    """
    target = target_text.strip().lower()
    candidates = [
        w
        for w in words
        if w.confidence >= min_confidence and (target in w.text.strip().lower() or w.text.strip().lower() in target)
    ]
    if not candidates:
        return None
    if len(candidates) == 1 or hint_frac is None or viewport_size is None:
        return max(candidates, key=lambda w: w.confidence)

    hint_x = hint_frac[0] * viewport_size[0]
    hint_y = hint_frac[1] * viewport_size[1]

    def distance(w: OcrWord) -> float:
        cx, cy = w.center
        return math.hypot(cx - hint_x, cy - hint_y)

    return min(candidates, key=distance)


def to_fraction(word: OcrWord, viewport_size: tuple[int, int]) -> tuple[float, float]:
    """Convert a matched word's center to a viewport fraction -- the same
    representation `relative_coords` already uses, and what an artifact
    should record as a `hint_frac` (never raw pixels, so a different
    resolution still replays sensibly)."""
    cx, cy = word.center
    return (cx / viewport_size[0], cy / viewport_size[1])
