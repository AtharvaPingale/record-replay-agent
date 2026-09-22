"""The Surface protocol.

Every upstream module (agent loop, replayer, locator, guardrails, evidence,
escalation) talks only to this interface, never to Playwright (or any other
automation library) directly. This build implements `web.py` only; `legacy_web.py`
and `desktop.py` are deliberately not built this pass -- see REPORT.md Section 4
for how they'd plug in here without changing anything above this layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

ActionType = Literal[
    "click", "type", "select", "wait", "assert_text", "go_to", "key", "done", "ask_user"
]


class GuardrailBlocked(Exception):
    """A surface refused an action on policy grounds. Part of the Surface
    contract rather than any one backend's: every surface is expected to refuse
    the same way, and every caller has to handle a refusal the same way, so a
    desktop or legacy-web backend added later raises this too.

    Defined here rather than in surfaces/web.py (which re-exports it for
    backwards compatibility) so that callers which must handle a refusal --
    replayer/executor.py in particular -- can catch it without importing
    Playwright to do so."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass
class Action:
    type: ActionType
    # Locator the action targets, in priority order. Each entry is
    # {"kind": "dom_selector" | "text" | "relative_coords", "value": ...}.
    # Empty for actions that don't target an element (wait, go_to, done, ask_user).
    target: list[dict[str, Any]] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    # Names the keys in `params` whose value came from a {{env:VAR}} credential
    # placeholder (artifact/schema.py's render_step_params second return value),
    # not a literal. Empty by default -- only replayer/executor.py's
    # _step_to_action populates this. Lets a surface flag a secret-shaped field
    # (e.g. an <input type=password>) being filled with a value that did NOT
    # come from a credential reference, without the surface needing to know
    # anything about placeholders itself.
    credential_keys: frozenset[str] = field(default_factory=frozenset)


@dataclass
class Observation:
    url: str
    title: str
    screenshot_path: Path
    dom_excerpt: str  # trimmed HTML/text relevant to the visible viewport
    timestamp: float
    # OCR'd text actually rendered to pixels (locator/ocr.py), alongside
    # dom_excerpt rather than instead of it -- catches text a DOM excerpt
    # can't show at all (canvas, an image, a font icon with no accessible
    # name) without giving up the DOM's much richer selector/structure
    # signal for everything else. None when OCR wasn't run or failed --
    # never required for a decider to act, only supplementary grounding.
    ocr_excerpt: str | None = None
    # Screenshot redaction now happens in the surface, once, before anything is
    # written: the file at screenshot_path is *already* the redacted image, and
    # this is how many PII-shaped regions were painted out of it (0 when none
    # matched). Deciders read this rather than re-redacting -- one OCR pass per
    # observe instead of two, and the on-disk evidence is the same image the
    # model saw. See guardrails/pii_redact.py's persist_redacted_screenshot.
    pii_redacted_count: int = 0
    # False means redaction could not run (OCR unavailable or failed) and the
    # surface therefore wrote NOTHING at screenshot_path -- fail-closed. Every
    # consumer of screenshot_path must treat the file as absent in that case.
    screenshot_redacted: bool = True


@dataclass
class EvidenceBundle:
    screenshot_path: Path
    dom_snapshot_path: Path
    url: str
    extra: dict[str, Any] = field(default_factory=dict)


class Surface(Protocol):
    def observe(self) -> Observation: ...

    def act(self, action: Action) -> dict[str, Any]:
        """Execute one action against the live surface. Returns a small result
        dict (e.g. {"ok": True} or {"ok": False, "error": "..."})."""
        ...

    def snapshot_for_evidence(self, out_dir: Path) -> EvidenceBundle: ...

    def is_visible(self, text: str) -> bool:
        """True only if `text` is on a genuinely rendered (not just present-but-
        hidden) element. Added after a discovery run found live that
        page.content() string search is not a safe success checkpoint: e.g.
        an earlier e-commerce target's "Added!" confirmation modal was *always* present
        in the static page template, just CSS-hidden until an add-to-cart click
        toggles a "show" class -- a naive text-present check on raw HTML would
        report success on a page that was never acted on at all."""
        ...

    def is_selector_visible(self, selector: str) -> bool:
        """True only if the first element matching `selector` (CSS, or a
        Playwright-style chained selector on a web surface) is genuinely
        rendered. The structural counterpart of is_visible: "this row exists"
        rather than "these words are on screen"."""
        ...

    def current_url(self) -> str: ...

    def extract_text(self, target: list[dict[str, Any]]) -> str | None:
        """Read the visible text of the first locator in `target` that
        resolves, or None if none do. Used only after a checkpoint is met, to
        populate an artifact's declared `output_schema` (artifact/schema.py's
        OutputExtractor) -- the read half of the same locate-in-priority-order
        mechanism `act()` uses for the write half (click/type)."""
        ...
