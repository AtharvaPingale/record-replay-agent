"""The real part of escalation: handing an actual human actual control of the
actual live browser session -- not a copy, not a replica, not a re-launched
browser pointed at the same URL (which would lose cart/session state).

Mechanism: the automated WebSurface's Chromium is launched with its CDP port
exposed on localhost (`--remote-debugging-port`). A second client -- a real
person's browser pointed at the DevTools frontend URL, or (as used for this
build's demo, see escalation/operator_cli.py) a second Playwright client calling
`connect_over_cdp` -- attaches to that *same* browser process and *same* page.
Whatever it does happens in the one live session. When the human signals done,
this process's own `page` object reflects their changes immediately, because
it's the same underlying page, not a snapshot of it.

Two things this module owns beyond the connection details:

- **Who holds control.** `ControlOwner` is explicit state, flipped by
  `start_handoff`/`end_handoff` and logged on both transitions. The brief asks
  for "a way to know who is (or should be) in control"; inferring it from the
  fact that some Python thread happens to be blocked is not that.
- **What changed while they held it.** `SessionSnapshot` captures the session
  immediately before control is ceded and immediately after it comes back, and
  `describe_operator_changes` diffs the two. Stated precisely, because the
  distinction matters: this records the *effect* the operator had on the
  session, not a keystroke-level log of their individual actions. Capturing
  those would mean subscribing to CDP input events for the attached client --
  real, and a clean extension of this seam, but not built here.

Full co-browsing UI (screen-sharing, cursors, etc.) is explicitly out of scope --
the operator surface is a CLI. This module is only responsible for the
pause/cede/resume mechanism, and the record of it, being real.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps this module import-light
    from surfaces.base import Surface


class ControlOwner(str, Enum):
    """Who is entitled to act on the live session right now. Automation must
    not act while this says OPERATOR, and the operator is done once it says
    AUTOMATION again."""

    AUTOMATION = "automation"
    OPERATOR = "operator"


@dataclass
class SessionSnapshot:
    """What the shared session looked like at one instant, taken through the
    *automation's* own surface handle -- the same one it will resume on."""

    url: str
    title: str
    screenshot_path: str | None

    @classmethod
    def capture(cls, surface: "Surface") -> "SessionSnapshot":
        obs = surface.observe()
        return cls(url=obs.url, title=obs.title, screenshot_path=str(obs.screenshot_path))


@dataclass
class HandoffSession:
    cdp_url: str
    devtools_frontend_hint: str
    control_owner: ControlOwner = ControlOwner.AUTOMATION
    before: SessionSnapshot | None = None
    after: SessionSnapshot | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def start_handoff(cdp_port: int, surface: "Surface | None" = None) -> HandoffSession:
    """Pause automation and cede control. Called once the browser is already
    running with --remote-debugging-port. Returns the connection info an
    operator (human or, for this demo, a second Playwright client) needs to
    attach to the live session -- with control formally transferred and the
    pre-handoff state captured, so what the operator changes is recoverable
    afterward rather than inferred."""
    cdp_url = f"http://localhost:{cdp_port}"
    return HandoffSession(
        cdp_url=cdp_url,
        devtools_frontend_hint=(
            f"A real operator would open {cdp_url}/json to find the live page's "
            f"devtoolsFrontendUrl and interact with it directly in their browser. "
            f"This build's demo instead attaches a second Playwright client via "
            f"chromium.connect_over_cdp('{cdp_url}') to the same session, to prove "
            f"the mechanism without requiring a human present during the test run."
        ),
        control_owner=ControlOwner.OPERATOR,
        before=SessionSnapshot.capture(surface) if surface is not None else None,
    )


def end_handoff(session: HandoffSession, surface: "Surface | None" = None) -> HandoffSession:
    """Take control back. Always called once the operator has signalled --
    including on 'skip' and on a timeout, so control never silently stays with
    an operator who has gone away."""
    session.control_owner = ControlOwner.AUTOMATION
    if surface is not None:
        session.after = SessionSnapshot.capture(surface)
    return session


def describe_operator_changes(session: HandoffSession) -> dict[str, Any]:
    """A structured, loggable record of what the handoff actually changed on the
    shared session. `recorded` is deliberately explicit about the limits of this
    evidence, so a reader of the log never mistakes it for a keystroke trace."""
    if session.before is None or session.after is None:
        return {
            "recorded": "none -- no surface was supplied to start_handoff/end_handoff",
            "control_owner": session.control_owner.value,
        }
    return {
        "recorded": "before/after state of the shared session, not the operator's individual actions",
        "control_owner": session.control_owner.value,
        "url_before": session.before.url,
        "url_after": session.after.url,
        "navigated": session.before.url != session.after.url,
        "title_before": session.before.title,
        "title_after": session.after.title,
        "screenshot_before": session.before.screenshot_path,
        "screenshot_after": session.after.screenshot_path,
    }
