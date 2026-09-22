"""CLI for replaying a single-flow artifact -- one artifact, one set of typed
inputs, one structured result with outputs. This is the production execution
path: what an AI agent invokes, with no model in the loop.

A genuine `hard_failure` escalates for real, right here: the browser is
always launched with its CDP port exposed, and a hard_failure triggers the
real pause -> handoff -> resume sequence (escalation/detect.py, handoff.py,
operator_cli.py, resume.py). By default `wait_for_operator_signal` blocks on
real stdin input from whoever is running this -- not an auto-signal standing
in for one -- which *is* the real autonomous path's escalation behavior, not
a separate proof of the mechanism.

Usage: ./run.sh python scripts/run_single_replay.py <artifact_path> <inputs_json> [--allow-blocked login,admin,...]

`--allow-blocked` explicitly overrides specific default-blocked route shapes
(guardrails/allowlist.py's DEFAULT_BLOCKED_PATTERNS) at REPLAY time -- same
flag, same reasoning, as scripts/record_capability.py's own at recording time.
Found live: recording a capability with that flag at *recording* time does
not by itself make the resulting artifact replayable -- this script derives
its own, independent allowlist, which knows nothing about what was
overridden during recording. An artifact whose own steps click a
login/admin route needs the same override again here.

--demo-break-step N [--demo-operator fix|skip]: FOR DEMOS/TESTS ONLY, never a
real replay parameter. Corrupts step N's locators in a throwaway copy of the
artifact (written alongside this run's own evidence as `*.BROKEN.json`) before
replaying it, forcing a genuine `hard_failure` at that step so the escalation
path above runs non-interactively, without a person at the keyboard. Nothing
in the escalation mechanism itself is special-cased for this -- it is the
exact same `hard_failure` -> pause -> handoff -> resume sequence a real UI
drift would trigger, just against a deliberately broken artifact instead of a
site that changed. `--demo-operator fix` (the default) spawns
scripts/mock_operator.py as a genuinely *separate process* that attaches to
the paused browser over CDP and performs the real, uncorrupted step by hand,
then signals `done`; `--demo-operator skip` signals `skip` immediately -- the
absent-operator path, which leaves the result a `hard_failure` with the full
evidence bundle.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # e.g. TEST_ACCOUNT_EMAIL/PASSWORD for a {{env:...}} credential step -- see guardrails/redact.py

from artifact.schema import Artifact, InputValidationError
from escalation.detect import build_alert_notice, build_intervention_request, should_alert_human, should_escalate
from escalation.handoff import describe_operator_changes, end_handoff, start_handoff
from escalation.operator_cli import print_alert, print_intervention, print_verification_request, wait_for_operator_signal
from escalation.resume import verify_resolution
from telemetry.log import RunLog
from guardrails.allowlist import AllowlistConfig
from guardrails.risk import REPLAY_RISK_RULES
from replayer.classify import ReplayResult, ReplayResultType
from replayer.executor import replay_artifact
from surfaces.base import Action
from surfaces.web import WebSurface

CDP_PORT = 9333

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("artifact_path")
    parser.add_argument("inputs_json", nargs="?", default="{}")
    parser.add_argument(
        "--allow-blocked", default=None,
        help="comma-separated route shapes to un-block for this replay (e.g. 'login,admin') -- only for a known, reviewed target",
    )
    parser.add_argument(
        "--demo-break-step", type=int, default=None,
        help="FOR DEMOS/TESTS ONLY: corrupt this step's locators in a throwaway copy before replaying, to force a "
             "genuine hard_failure and exercise the real escalation path non-interactively. Never a real replay parameter.",
    )
    parser.add_argument(
        "--demo-operator", choices=["fix", "skip"], default="fix",
        help="with --demo-break-step: 'fix' spawns scripts/mock_operator.py as a separate process to perform the "
             "broken step correctly over CDP before signaling done; 'skip' signals skip immediately, the "
             "absent-operator path. Ignored without --demo-break-step.",
    )
    args = parser.parse_args()

    artifact_path = args.artifact_path
    inputs = json.loads(args.inputs_json)

    artifact = Artifact.load(artifact_path)

    # Check the caller's arguments against the artifact's declared input_schema
    # before launching anything. replayer/executor.py validates too (that is the
    # library contract, and it must hold for every caller), but doing it here as
    # well means a malformed invocation costs no browser launch and no network
    # traffic against the target site.
    try:
        artifact.validate_inputs(inputs)
    except InputValidationError as e:
        print(f"error: {e}")
        print(f"\nthis capability declares: {artifact.input_schema or 'no inputs'}")
        raise SystemExit(2) from e

    run_id = f"replay_{time.strftime('%Y-%m-%d_%H%M%S')}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # --demo-break-step: swap `artifact` for a throwaway corrupted copy before
    # anything downstream (allowlist scoping, replay, evidence) ever sees it.
    # Corrupting after validate_inputs (above) so a malformed --inputs_json is
    # still refused on its own terms, and before everything below so the
    # corrupted copy is what actually replays -- a real hard_failure, not a
    # scripted one. args.artifact_path (the real, uncorrupted file on disk)
    # stays what gets handed to scripts/mock_operator.py further down: the
    # "operator" has to fix the step using the artifact's real declared
    # locator, not the one just broken here.
    if args.demo_break_step is not None:
        original_step = next(s for s in artifact.steps if s.step == args.demo_break_step)
        broken = artifact.model_copy(deep=True)
        broken_step = next(s for s in broken.steps if s.step == args.demo_break_step)
        broken_step.target.primary.value = "#this-control-no-longer-exists"
        if broken_step.target.fallback is not None:
            broken_step.target.fallback.value = "this exact text does not appear on the page"
        broken.revision_note = (
            f"NOT A REAL VERSION -- --demo-break-step corrupted step {args.demo_break_step}'s locators "
            f"(originally {original_step.target.primary.kind} {original_step.target.primary.value!r}) to force a hard_failure."
        )
        broken_path = run_dir / f"{artifact.artifact_id}.v{artifact.version}.BROKEN.json"
        broken_path.write_text(broken.model_dump_json(indent=2))
        print(f"[demo] wrote deliberately broken artifact to {broken_path} -- replaying that copy, not {artifact_path}")
        artifact = broken

    # Derived from the artifact's own declared target, not hardcoded -- found
    # live: this used to always load one config and always navigate to one
    # hardcoded e-commerce site regardless of which artifact was
    # actually passed in, which only ever worked because this repo had a
    # single target until scripts/record_capability.py could produce others.
    allow_patterns = args.allow_blocked.split(",") if args.allow_blocked else None
    allowlist = AllowlistConfig.for_artifact_target(artifact.target.base_url_pattern, allow_patterns=allow_patterns)
    if allow_patterns:
        print(f"[guardrails] --allow-blocked explicitly un-blocked: {allow_patterns} -- your call that this target is safe for it.")
    # The caller's own inputs stay visible in saved screenshots: they supplied
    # them and need to see them to verify the run looked up the right record.
    # Everything else PII-shaped on the page is painted out before any
    # screenshot is written -- see WebSurface.pii_exempt.
    pii_exempt = frozenset(str(v) for v in inputs.values() if isinstance(v, (str, int)) and not isinstance(v, bool))
    surface = WebSurface(
        allowlist=allowlist, screenshot_dir=run_dir / "screenshots", headless=True, risk_rules=REPLAY_RISK_RULES,
        cdp_port=CDP_PORT,  # always exposed -- a real operator can attach the moment a hard_failure escalates
        pii_exempt=pii_exempt,
    )
    run_log = RunLog(run_id=run_id, out_dir=run_dir)

    # Everything from here runs under `finally: surface.close()`. Without it, any
    # exception -- including a guardrail refusal before replayer/executor.py
    # learned to classify one -- left a real Chromium process and an open CDP
    # port behind, on a session that may well be logged in.
    try:
        surface.act(Action(type="go_to", params={"url": artifact.target.resolved_entry_url()}))

        result = replay_artifact(artifact, inputs, surface, run_log, run_dir)

        # ---- the verification gate ------------------------------------------
        # Any classification other than a clean success is shown to a person
        # before it is handed to the caller as the answer. A declared business
        # outcome is a legitimate result -- but the same "No users found" that
        # means "no such account" also appears when a search filter fails to
        # apply or a page errors into its empty state, and the caller would
        # receive a confident, wrong "not found". Likewise a success whose
        # extractor read nothing. The human confirms the classification, or
        # rejects it -- and a rejection is treated as what it is, a hard
        # failure on a live session, which routes into the handoff below.
        needs_verification = result.type == ReplayResultType.BUSINESS_OUTCOME or (
            result.type == ReplayResultType.SUCCESS and result.detail.get("outputs_missing")
        )
        if needs_verification:
            obs = surface.observe()
            print_verification_request(
                f"replay {artifact.artifact_id}", inputs, result.type.value, result.detail,
                str(obs.screenshot_path) if obs.screenshot_redacted else None, obs.url, f"http://localhost:{CDP_PORT}",
            )
            run_log.emit("verification_requested", {"item": inputs, "classified_as": result.type.value, **result.detail})
            # 'unverified' is a sentinel, not a choice a human can type: a
            # non-answer (timeout, closed stdin) can never be mistaken for a
            # confirmation. The result still goes back to the caller -- an
            # unattended replay must be able to return a not-found -- but it
            # carries the truth about whether anyone looked. --demo-break-step
            # auto-confirms this gate defensively: which result type a given
            # (artifact, inputs) pair produces before ever reaching the broken
            # step isn't this flag's business to predict, and the demo must
            # never hang on a prompt it didn't come here to answer.
            verdict = wait_for_operator_signal(
                timeout_s=300.0, choices=("confirm", "reject"), default="unverified",
                auto_signal="confirm" if args.demo_break_step is not None else None,
            )
            result.detail["human_verified"] = {"confirm": True, "unverified": False, "reject": "rejected"}[verdict]
            run_log.emit("verification_resolved", {"item": inputs, "verdict": verdict, "human_verified": result.detail["human_verified"]})
            if verdict == "reject":
                print("[operator] classification rejected -- treating as a hard failure and handing over the session")
                result = ReplayResult(
                    ReplayResultType.HARD_FAILURE,
                    {"step": None, "reason": "human rejected the classification", "rejected_result": result.type.value,
                     "rejected_detail": {k: v for k, v in result.detail.items() if k != "human_verified"}},
                )

        if should_escalate(result):
            req = build_intervention_request(f"replay {artifact.artifact_id}", inputs, result, str(run_dir))
            # Control formally transfers here and the pre-handoff state is
            # captured, so "who is in control" is recorded state rather than an
            # inference from which thread happens to be blocked.
            handoff = start_handoff(CDP_PORT, surface)
            print_intervention(req, handoff)
            run_log.emit(
                "escalation_raised",
                {"item": inputs, "reason": req.reason, "control_owner": handoff.control_owner.value},
            )

            # By default, a real blocking input() -- no auto_signal -- whoever
            # is running this script IS the operator being asked to attach and
            # act. Bounded (unlike a bare input() with no timeout_s) so an
            # unattended run can't hold the browser/CDP port open forever if
            # no one responds -- a timeout is always treated as "skip", never
            # as "done".
            #
            # --demo-break-step supplies the answer itself instead of
            # blocking: 'fix' spawns scripts/mock_operator.py as a genuinely
            # *separate process* -- a second, independent connect_over_cdp
            # client, exactly what a real operator's own browser would be --
            # to perform the real step over CDP before signaling done; 'skip'
            # signals skip immediately, the absent-operator path. Nothing
            # about the escalation mechanism itself branches on this: the same
            # wait_for_operator_signal, start_handoff/end_handoff and
            # verify_resolution calls run either way.
            demo_signal = None
            if args.demo_break_step is not None:
                if args.demo_operator == "fix":
                    fix = subprocess.run(
                        [sys.executable, "scripts/mock_operator.py", args.artifact_path, args.inputs_json,
                         "--step", str(args.demo_break_step), "--cdp", f"http://localhost:{CDP_PORT}"],
                    )
                    demo_signal = "done" if fix.returncode == 0 else "skip"
                else:
                    print("[demo] operator chose not to intervene")
                    demo_signal = "skip"
            signal = wait_for_operator_signal(timeout_s=300.0, auto_signal=demo_signal)

            # Control comes back on every path -- "done", "skip" and timeout
            # alike -- so it never silently stays with an operator who left.
            end_handoff(handoff, surface)
            changes = describe_operator_changes(handoff)
            run_log.emit("operator_actions", {"item": inputs, "signal": signal, **changes})
            if changes.get("navigated"):
                print(f"[operator] session moved: {changes['url_before']} -> {changes['url_after']}")

            resolved = False
            if signal == "done":
                verified = verify_resolution(artifact, surface, inputs)
                resolved = verified.type == ReplayResultType.SUCCESS
                if resolved:
                    result = verified
            run_log.emit(
                "escalation_resolved",
                {
                    "item": inputs,
                    "signal": signal,
                    "resolved": resolved,
                    "control_owner": handoff.control_owner.value,
                },
            )
            print(f"[operator] resolved (re-verified live via the shared surface): {resolved}")

        # A business outcome the artifact itself flagged alert_human still
        # returns normally to the caller -- it is a legitimate result, not a
        # failure -- but a person is told about it here rather than it being
        # findable only by grepping the run log afterward.
        if should_alert_human(result):
            print_alert(build_alert_notice(f"replay {artifact.artifact_id}", inputs, result))

        print(f"run_dir={run_dir}")
        print(f"result={result.type.value}")
        print(f"detail={result.detail}")
    finally:
        surface.close()

    raise SystemExit(0 if result.type in (ReplayResultType.SUCCESS, ReplayResultType.BUSINESS_OUTCOME) else 2)
