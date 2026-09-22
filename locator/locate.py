"""Multi-strategy element resolution: DOM selector -> visible text -> OCR text
(DOM-less fallback) -> relative coordinates, in that priority order. Coordinates
are always the last resort and are always stored/consumed as a fraction (0-1)
of the viewport, never raw pixels, so a trace/artifact recorded on one screen
resolution still replays correctly on another.

This module only knows about a Playwright `Page` for taking the screenshot an
"ocr_text" candidate resolves against (locator/ocr.py) -- everything else about
that candidate is Playwright-agnostic, which is what would let a desktop
backend reuse it unchanged against its own screen-capture API. The
`Target`/`LocateResult` shapes here are the seam the rest of the system
(executor, replayer, agent loop) already depends on, not on Playwright directly.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playwright.sync_api import Page


class LocateError(Exception):
    def __init__(self, target: list[dict[str, Any]]):
        self.target = target
        super().__init__(f"no candidate locator resolved: {target!r}")


@dataclass
class LocateResult:
    kind: str  # "dom_selector" | "text" | "ocr_text" | "relative_coords"
    handle: Any  # a Playwright Locator, or None for coordinate-based results
    coords: tuple[int, int] | None  # absolute pixel coords, for ocr_text/relative_coords
    # Whether anything was actually confirmed to be *there*. True for the DOM
    # and text kinds (they waited for a visible match) and for ocr_text (it
    # matched real rendered words). False only for relative_coords, which
    # resolves arithmetically and cannot fail: see the note on that branch.
    verified: bool = True


def locate(page: Page, target: list[dict[str, Any]], timeout_ms: int = 3000) -> LocateResult:
    """Try each candidate in `target`'s declared priority order. First visible
    match wins. Raises LocateError if nothing resolves -- callers must not guess."""
    for candidate in target:
        kind = candidate["kind"]
        value = candidate["value"]

        if kind == "dom_selector":
            loc = page.locator(value).first
            try:
                loc.wait_for(state="visible", timeout=timeout_ms)
                return LocateResult(kind="dom_selector", handle=loc, coords=None)
            except Exception:
                continue

        elif kind == "text":
            loc = page.get_by_text(value, exact=False).first
            try:
                loc.wait_for(state="visible", timeout=timeout_ms)
                return LocateResult(kind="text", handle=loc, coords=None)
            except Exception:
                pass
            # get_by_text only matches literal inner text content -- an
            # <input>'s placeholder/aria-label has no text node at all, so a
            # form field identified by its visible label (agent/tools.py's
            # find_field_by_text is exactly this case) can never resolve
            # through get_by_text alone. Found live: YouTube's search box
            # (placeholder="Search") failed here -- get_by_text("Search")
            # does match 3 elements on the page, but .first lands on a
            # visually-hidden a11y label ("Search"/"Search with your voice"),
            # never the visible input, so every attempt timed out waiting
            # for visibility and a local-model discovery run burned 7 steps
            # before giving up and improvising a hardcoded URL instead of a
            # real type action. get_by_placeholder/get_by_label are the two
            # other standard ways a form field exposes an accessible name --
            # still the same "text" kind (find by label), just the matching
            # strategies get_by_text alone doesn't cover.
            for by in (page.get_by_placeholder, page.get_by_label):
                try:
                    alt = by(value, exact=False).first
                    alt.wait_for(state="visible", timeout=timeout_ms)
                    return LocateResult(kind="text", handle=alt, coords=None)
                except Exception:
                    continue
            continue

        elif kind == "ocr_text":
            # The DOM-less fallback (REPORT.md Section 4): resolves against
            # words actually rendered to pixels via a fresh OCR pass on the
            # current page, not a DOM query -- this candidate works the same
            # way whether or not the page has a usable DOM at all. `value` is
            # `{"text": <what to look for>, "hint_frac": [x, y]}` --
            # `hint_frac` is optional and only disambiguates when the text
            # appears more than once on the page.
            from locator.ocr import find_best_match, run_ocr

            vp = page.viewport_size or {"width": 1280, "height": 900}
            hint_frac = tuple(value.get("hint_frac")) if value.get("hint_frac") else None

            def _shoot_and_match():
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    shot_path = f.name
                try:
                    page.screenshot(path=shot_path)
                    words = run_ocr(shot_path)
                finally:
                    Path(shot_path).unlink(missing_ok=True)
                return find_best_match(words, value["text"], hint_frac=hint_frac, viewport_size=(vp["width"], vp["height"]))

            match = _shoot_and_match()
            if match is None:
                continue
            cx, cy = match.center

            # Found live: a real page can push its own content past the fold
            # (here, a promotional banner shifted a real "Add to cart" button
            # down to the last ~17px of a 900px viewport). OCR's bbox for a
            # match clipped by the viewport edge only covers the visible
            # sliver, so its center is computed from a partial box -- close
            # enough to matter for a click. Scroll a near-edge match toward
            # vertical center and re-resolve once with a fresh OCR pass
            # (bounded -- one retry, never open-ended) rather than trust a
            # coordinate that might be off by the clipped fraction of the box.
            edge_margin = 0.1 * vp["height"]
            if cy < edge_margin or cy > vp["height"] - edge_margin:
                page.mouse.wheel(0, cy - vp["height"] / 2)
                page.wait_for_timeout(200)
                rescanned = _shoot_and_match()
                if rescanned is not None:
                    cx, cy = rescanned.center

            return LocateResult(kind="ocr_text", handle=None, coords=(round(cx), round(cy)))

        elif kind == "relative_coords":
            # The one candidate kind that cannot fail, and therefore the one
            # that cannot succeed either: it is arithmetic on the viewport, with
            # nothing on the page consulted. As a *fallback* that makes it
            # actively worse than no candidate at all -- a genuine locate
            # failure (which every kind above reports honestly, letting
            # classify.py decide what it means) silently becomes a click into
            # empty space reported as ok. A wrong success is much harder to
            # debug than a clean hard_failure.
            #
            # Kept, because on a truly DOM-less surface a recorded coordinate is
            # sometimes the only thing there is -- but marked unverified, which
            # surfaces/web.py turns into a review signal on the action's result,
            # and bounds-checked so an off-viewport coordinate is a real
            # LocateError rather than a click at a nonsense position.
            fx, fy = value
            vp = page.viewport_size or {"width": 1280, "height": 900}
            if not (0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0):
                continue  # outside the viewport entirely -- not a usable candidate
            x = round(fx * vp["width"])
            y = round(fy * vp["height"])
            return LocateResult(kind="relative_coords", handle=None, coords=(x, y), verified=False)

        else:
            continue

    raise LocateError(target)
