"""Deterministic executor: walk an artifact's steps, locate, act, verify the
checkpoint. Zero LLM calls. Halts and returns a classified result rather than
guessing when something doesn't go as declared -- retries are capped and only
apply to declared recoverable patterns (see classify.py), never open-ended.

Every exit from this function is a `ReplayResult`. Nothing here raises at the
caller: a bad input, a policy refusal, a blown deadline and a genuine app error
are all *results* an agent can act on, because this is the production path an
agent invokes and an exception escaping it would just be an outage with no
diagnosis attached.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from artifact.schema import Artifact, InputValidationError, Step
from escalation.detect import should_alert_human
from telemetry.bundle import write_failure_bundle
from telemetry.log import RunLog
from replayer.classify import ReplayResult, ReplayResultType, classify_outcome
from surfaces.base import Action, GuardrailBlocked, Surface

MAX_RECOVERY_ATTEMPTS_PER_STEP = 2

# Wall-clock ceiling for one whole replay. The per-action timeouts underneath
# this (Playwright's own locate/click/goto bounds) cap a single operation, but
# nothing capped their *sum*: a long artifact whose every step burns its full
# retry budget against a slow or half-broken app could run for many minutes with
# no upper bound at all. Section 3.3 names "transient slowness" and "session
# timeout" as conditions replay has to handle deliberately, and an invocation
# that never returns is the one failure mode a calling agent cannot handle.
# Generous relative to a healthy run (the canonical 13-step capability replays
# in well under a minute) so this only ever fires on something genuinely wrong.
DEFAULT_REPLAY_DEADLINE_S = 300.0


def _step_to_action(step: Step, artifact: Artifact, inputs: dict[str, Any]) -> tuple[Action, set[str]]:
    target = []
    if step.target:
        target.append({"kind": step.target.primary.kind, "value": step.target.primary.value})
        if step.target.fallback:
            target.append({"kind": step.target.fallback.kind, "value": step.target.fallback.value})
    params, credential_keys = artifact.render_step_params(step, inputs)
    action = Action(type=step.action, target=target, params=params, credential_keys=frozenset(credential_keys))
    return action, credential_keys


def _guarded_act(surface: Surface, action: Action) -> dict[str, Any]:
    """`Surface.act` raises GuardrailBlocked on a policy refusal. agent/loop.py
    has always handled that; this path did not, so a refusal mid-replay escaped
    as an unhandled traceback -- no classified result, no evidence bundle, no
    escalation, and the caller's `surface.close()` never ran, leaking the
    browser and its CDP port.

    Turning it into an ordinary failed-action result lets the normal machinery
    carry it, while the distinctive `error` below keeps it from being treated as
    retryable friction: a refused action is a policy breach to surface, never
    something to attempt again."""
    try:
        return surface.act(action)
    except GuardrailBlocked as e:
        return {"ok": False, "error": "guardrail_blocked", "detail": e.reason}


def _perform_recovery(surface: Surface, artifact: Artifact, classified: ReplayResult) -> dict[str, Any]:
    """Execute the matched pattern's *declared* recovery action.

    Previously a recoverable match mid-step just retried the step (nothing had
    actually cleared the interstitial) and a recoverable match at the final
    checkpoint pressed a hardcoded Escape regardless of what the artifact said.
    `RecoverablePattern.recovery` was prose the executor never read. Now the
    declared `recovery_action` is what runs, with the old generic Escape kept as
    the fallback for any pattern that doesn't declare one -- so artifacts
    recorded before that field existed behave exactly as they did."""
    name = classified.detail.get("name")
    pattern = next((p for p in artifact.recoverable_patterns if p.name == name), None)
    declared = pattern.recovery_action if pattern else None

    if declared is None:
        return _guarded_act(surface, Action(type="key", params={"combo": "Escape"}))

    target = [{"kind": declared.target.kind, "value": declared.target.value}] if declared.target else []
    params = dict(declared.params)
    if declared.action == "key":
        params.setdefault("combo", "Escape")
    return _guarded_act(surface, Action(type=declared.action, target=target, params=params))


def _hard_failure(
    run_log: RunLog,
    run_dir: Path,
    surface: Surface,
    trace: list[dict[str, Any]],
    step: int | None,
    detail: dict[str, Any],
) -> ReplayResult:
    """One place that writes the richer on-failure evidence signal (Section 3.5)
    and emits the event, so no hard-failure exit can forget either."""
    bundle = surface.snapshot_for_evidence(run_dir / "failure_tmp")
    write_failure_bundle(run_dir, step, bundle, trace, detail)
    run_log.emit("hard_failure", {"step": step, **detail}, step=step)
    return ReplayResult(ReplayResultType.HARD_FAILURE, {"step": step, **detail})


def _business_outcome(run_log: RunLog, classified: ReplayResult, item_label: str, step: int | None) -> ReplayResult:
    """Emit a business outcome, and page a human if -- and only if -- the
    artifact itself declared this outcome alert-worthy.

    `should_alert_human` used to be consulted on only one of the two paths that
    can return a business outcome (mid-step, not at the final checkpoint), so an
    outcome flagged `alert_human=True` that surfaced after the last step was
    silently swallowed. Both paths come through here now."""
    run_log.emit("business_outcome", {"item": item_label, **classified.detail}, step=step)
    if should_alert_human(classified):
        run_log.emit("human_alert", {"item": item_label, **classified.detail}, step=step)
    return classified


def replay_artifact(
    artifact: Artifact,
    inputs: dict[str, Any],
    surface: Surface,
    run_log: RunLog,
    run_dir: Path,
    item_label: str = "",
    deadline_s: float = DEFAULT_REPLAY_DEADLINE_S,
) -> ReplayResult:
    """Replays one artifact once, with one set of typed inputs. Returns exactly
    one ReplayResult (success / business_outcome / recoverable / hard_failure)."""
    trace: list[dict[str, Any]] = []
    expires_at = time.monotonic() + deadline_s

    # The declared input contract, checked before anything executes. Without
    # this, `input_schema` was documentation: a missing field surfaced as a bare
    # KeyError from inside step rendering, and a wrong-typed one didn't surface
    # at all -- it was typed into the page and failed later as something that
    # looked like an app problem. A calling agent gets one clear answer instead.
    try:
        artifact.validate_inputs(inputs)
    except InputValidationError as e:
        run_log.emit("invalid_inputs", {"item": item_label, "error": str(e)}, step=None)
        return ReplayResult(
            ReplayResultType.HARD_FAILURE,
            {
                "step": None,
                "reason": "inputs do not satisfy the artifact's input_schema",
                "detail": str(e),
                # Not escalatable: nothing has touched the page yet, so there is
                # no live session for an operator to take over -- see
                # escalation/detect.py's should_escalate.
                "caller_error": True,
            },
        )

    # From here on the contract is concrete: a checkpoint of
    # 'element_visible: {{member_name}}' means this member, and an extractor
    # anchored to 'tr:has-text("{{member_name}}")' reads this member's row --
    # see Artifact.with_inputs_rendered for the wrong-answer-as-success this
    # prevents. Steps still render per step, so credentials resolve late.
    artifact = artifact.with_inputs_rendered(inputs)

    def _expired() -> bool:
        return time.monotonic() > expires_at

    for step in artifact.steps:
        if _expired():
            return _hard_failure(
                run_log, run_dir, surface, trace, step.step,
                {"reason": f"replay exceeded its {deadline_s:.0f}s deadline before step {step.step}"},
            )

        action, credential_keys = _step_to_action(step, artifact, inputs)
        # Never let a resolved credential reach a log line -- for a credential
        # key, log the original {{env:VAR_NAME}} marker (safe: a variable name,
        # not a secret) instead of the resolved value, by provenance
        # (credential_keys), independent of whether the secret happens to match
        # any pattern in guardrails/redact.py's shape-based redact_value.
        # `action.params` (unredacted) is still what actually gets executed.
        from guardrails.redact import mask_credentials_with_markers

        log_params = mask_credentials_with_markers(action.params, step.params, credential_keys)
        outcome_recorded = False

        reason = step.target.robustness_reasoning if step.target else "no element target for this action"
        # A step may declare its own retry ceiling (a real business contract --
        # "this specific step needs more/fewer attempts than the default") rather
        # than always falling back to the global constant below.
        max_attempts = step.max_recovery_attempts if step.max_recovery_attempts is not None else MAX_RECOVERY_ATTEMPTS_PER_STEP

        for attempt in range(max_attempts + 1):
            result = _guarded_act(surface, action)
            obs_after = surface.observe()  # screenshot for the record, not used for classification here

            run_log.emit(
                "act",
                {"item": item_label, "step": step.step, "action": step.action, "params": log_params, "result": result},
                step=step.step,
            )
            run_log.emit_replay_step(
                step=step.step,
                screenshot_path=obs_after.screenshot_path,
                decision={"action": step.action, "target": action.target, "params": log_params},
                reason=reason,
                result=result,
                extra={"item": item_label, "attempt": attempt},
            )
            trace.append({"step": step.step, "action": step.action, "result": result})

            if result.get("ok"):
                outcome_recorded = True
                break

            # A policy refusal is never retried and never reclassified against
            # the artifact's declared patterns: the guardrail already did its
            # job (nothing executed), and a declared step being *capable* of
            # tripping it means the artifact is wrong, not just this run.
            if result.get("error") == "guardrail_blocked":
                return _hard_failure(
                    run_log, run_dir, surface, trace, step.step,
                    {"reason": "guardrail refused this step", "result": result},
                )

            # step didn't succeed -- check declared patterns before assuming failure
            obs = surface.observe()
            classified = classify_outcome(artifact, obs.dom_excerpt, obs.url, surface)

            if classified.type == ReplayResultType.RECOVERABLE and attempt < max_attempts and not _expired():
                run_log.emit("recovered", {"item": item_label, "step": step.step, **classified.detail}, step=step.step)
                # Actually clear the declared friction before retrying, rather
                # than retrying an unchanged page and presuming it cleared.
                _perform_recovery(surface, artifact, classified)
                continue

            if classified.type == ReplayResultType.BUSINESS_OUTCOME:
                run_log.emit_replay_step(
                    step=step.step,
                    screenshot_path=obs.screenshot_path,
                    decision={"action": step.action, "target": action.target, "params": log_params},
                    reason=f"declared business_outcome '{classified.detail.get('name')}' matched -- not a failure",
                    result={"ok": False, "classified_as": "business_outcome"},
                    extra={"item": item_label},
                )
                return _business_outcome(run_log, classified, item_label, step.step)

            return _hard_failure(
                run_log, run_dir, surface, trace, step.step, {"result": result, **classified.detail}
            )

        if not outcome_recorded:
            return _hard_failure(
                run_log, run_dir, surface, trace, step.step, {"reason": "exhausted recovery attempts"}
            )

    # all steps executed without early return -- verify the checkpoint explicitly,
    # never assume success just because nothing errored
    from replayer.classify import _checkpoint_matches  # local import: internal helper, not part of the public API

    for attempt in range(MAX_RECOVERY_ATTEMPTS_PER_STEP + 1):
        # Short polling window, not a declared-pattern recovery: found live that
        # the checkpoint element occasionally isn't rendered yet on the very
        # first check after all steps report ok --
        # an async UI update that just needs a moment, not a business/recoverable
        # state and not a hard failure either. This waits for rendering to catch
        # up; it does not change what gets classified once it has.
        checkpoint_met = False
        for _ in range(3):
            obs = surface.observe()
            if _checkpoint_matches(artifact.checkpoint, obs.dom_excerpt, obs.url, surface):
                checkpoint_met = True
                break
            time.sleep(0.4)

        if checkpoint_met:
            # Populate the declared outputs now that the checkpoint is real --
            # output_schema is a promise the artifact makes to its caller
            # (Section 3.2: "typed outputs ... what the agent gets back"); this
            # is what actually keeps it, rather than a caller receiving `{}` on
            # every successful replay.
            outputs: dict[str, Any] = {}
            for extractor in artifact.output_extractors:
                target = [{"kind": extractor.target.kind, "value": extractor.target.value}]
                outputs[extractor.name] = surface.extract_text(target)

            # A declared output that came back empty is not a clean success --
            # the checkpoint held, so the *page* is right, but the extractor
            # missed and the caller is about to receive a null it may treat as
            # a real answer. Found live: an extractor anchored to `tr` on a
            # div-built table returned None for every member while replay
            # reported success three times running. Flagged, not failed: the
            # checkpoint is the success condition, and a partial read should
            # be visible in the result and the log rather than hidden by
            # either downgrading it to a failure or passing it off as whole.
            missing = sorted(k for k, v in outputs.items() if v is None)
            if missing:
                run_log.emit("output_missing", {"item": item_label, "fields": missing,
                    "hint": "checkpoint met but these extractors resolved nothing -- check their targets against the live DOM"}, step=None)

            run_log.emit("success", {"item": item_label, "outputs": outputs}, step=None)
            run_log.emit_replay_step(
                step=None,
                screenshot_path=obs.screenshot_path,
                decision={"checkpoint": artifact.checkpoint.model_dump()},
                reason="all declared steps completed and the artifact's checkpoint was verified visible",
                result={"ok": True, "outputs": outputs},
                extra={"item": item_label},
            )
            detail: dict[str, Any] = {"outputs": outputs}
            if missing:
                detail["outputs_missing"] = missing
            return ReplayResult(ReplayResultType.SUCCESS, detail)

        if _expired():
            return _hard_failure(
                run_log, run_dir, surface, trace, None,
                {"reason": f"replay exceeded its {deadline_s:.0f}s deadline waiting for the final checkpoint"},
            )

        classified = classify_outcome(artifact, obs.dom_excerpt, obs.url, surface)

        if classified.type == ReplayResultType.RECOVERABLE and attempt < MAX_RECOVERY_ATTEMPTS_PER_STEP:
            # Found live: a declared recoverable pattern (an interstitial
            # popup) can match *after* the checkpoint check, not mid-step --
            # meaning something intercepted the final action without raising a
            # locate/act error. Clear it via the pattern's own declared
            # recovery_action (generic Escape if it declares none) rather than
            # leaving a resolvable state exposed as if it were terminal.
            run_log.emit("recovered", {"item": item_label, "phase": "final_checkpoint", **classified.detail}, step=None)
            _perform_recovery(surface, artifact, classified)
            continue

        if classified.type == ReplayResultType.RECOVERABLE:
            # exhausted retries and it's *still* the same declared-recoverable
            # state -- this is no longer "resolved friction," it's a hard failure
            # (the declared recovery didn't clear this pattern).
            return _hard_failure(
                run_log, run_dir, surface, trace, None,
                {"reason": "recoverable pattern did not clear after retries", **classified.detail},
            )

        if classified.type == ReplayResultType.BUSINESS_OUTCOME:
            return _business_outcome(run_log, classified, item_label, None)

        if classified.type == ReplayResultType.HARD_FAILURE and attempt == 0 and artifact.steps:
            # Found live: the terminal click can occasionally report ok=True
            # (Playwright's click genuinely landed) while the page's own click
            # handler silently no-ops -- not a locate failure, not a declared
            # pattern, just an unreliable real-world click. One bounded
            # re-execution of the *same last step* before concluding
            # hard_failure -- not a different action, not a guess at what went
            # wrong, just giving the real click a second try.
            run_log.emit("retry_last_step", {"item": item_label, "reason": "checkpoint unmet, no pattern matched"}, step=None)
            last_action, _ = _step_to_action(artifact.steps[-1], artifact, inputs)
            retry_result = _guarded_act(surface, last_action)
            if retry_result.get("error") == "guardrail_blocked":
                return _hard_failure(
                    run_log, run_dir, surface, trace, None,
                    {"reason": "guardrail refused the final-step retry", "result": retry_result},
                )
            continue

        return _hard_failure(run_log, run_dir, surface, trace, None, classified.detail)

    return _hard_failure(
        run_log, run_dir, surface, trace, None, {"reason": "exhausted recovery attempts at final checkpoint"}
    )
