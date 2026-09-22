"""Real re-verification after a human handoff, through the same pipeline a normal
replay uses -- not a bespoke DOM query on whatever client the operator happened to
attach with.

Found via code review comparing this project against a suggestion list for a
different automation system: this project's own escalation path (a standalone
demo script at the time, since folded into scripts/run_single_replay.py)
originally decided "resolved" by calling
`operator_page.get_by_text("Added!", exact=False).first.is_visible()` on the
*second*, CDP-attached Playwright client -- bypassing replayer/classify.py
entirely, and trusting the operator's own client's view of the page rather than
asking the same question the automated executor would ask. A human operator has
full control of the live session during handoff; resuming without genuinely
re-checking what state that leaves things in is exactly the kind of stale-state
read this project has already had to guard against once for a coordinate-driven
click (surfaces/web.py's wait_for_load_state guard) and now again here.

Scope, stated plainly: this verifies whether the artifact's own declared
checkpoint (or a declared business/recoverable pattern) holds *right now*, via the
shared surface -- it does not add generic mid-artifact continuation of steps
after the one that escalated. This project's escalation model resolves at the
level of one whole replay (scripts/run_single_replay.py's hard_failure branch),
which is how the one escalating step today -- always an artifact's last step --
already works.
Building true mid-artifact step resumption -- re-observing and re-checking a
*specific step's* precondition partway through a longer artifact -- is a bigger
architectural change than this fix, and is not attempted here.
"""

from __future__ import annotations

from artifact.schema import Artifact
from replayer.classify import ReplayResult, ReplayResultType, _checkpoint_matches, classify_outcome
from surfaces.base import Surface


def verify_resolution(artifact: Artifact, surface: Surface, inputs: dict | None = None) -> ReplayResult:
    """Ask whether the artifact's checkpoint now holds, via the live surface the
    automation was already driving -- the same question replayer/executor.py asks
    at the end of every normal replay, asked here again after a human handoff
    instead of trusted on the operator's own say-so."""
    if inputs is not None:
        artifact = artifact.with_inputs_rendered(inputs)  # same concrete contract the replay itself checked
    obs = surface.observe()

    if _checkpoint_matches(artifact.checkpoint, obs.dom_excerpt, obs.url, surface):
        # Same as a normal successful replay (replayer/executor.py): populate the
        # declared outputs now that the checkpoint is genuinely met, rather than
        # handing a caller `{}` just because this success came via a handoff.
        outputs = {}
        for extractor in artifact.output_extractors:
            target = [{"kind": extractor.target.kind, "value": extractor.target.value}]
            outputs[extractor.name] = surface.extract_text(target)
        return ReplayResult(
            ReplayResultType.SUCCESS,
            {"note": "resolved via human handoff, re-verified via checkpoint", "outputs": outputs},
        )

    # Not met -- still check the artifact's own declared patterns before
    # concluding the handoff didn't actually fix anything, same discipline as a
    # normal replay: a business outcome or recoverable pattern is never inferred
    # from "the checkpoint isn't there."
    classified = classify_outcome(artifact, obs.dom_excerpt, obs.url, surface)
    if classified.type in (ReplayResultType.BUSINESS_OUTCOME, ReplayResultType.RECOVERABLE):
        return classified

    return ReplayResult(
        ReplayResultType.HARD_FAILURE,
        {"reason": "handoff did not resolve the artifact's checkpoint (re-verified live, not assumed)"},
    )
