"""The mocked operator surface -- deliberately just a console print and a
blocking prompt, per the deliberate choice (REPORT.md §5) not to build a real
co-browsing console. What must be real (the same live session, actually paused, actually
resumable) lives in handoff.py; this file is just how that gets presented to a
person.
"""

from __future__ import annotations

import threading

from escalation.detect import AlertNotice, InterventionRequest
from escalation.handoff import HandoffSession


def print_alert(notice: AlertNotice) -> None:
    """Non-blocking, unlike print_intervention: a business-outcome alert doesn't
    pause the run or need a live session to hand off -- the replay has already
    completed and returned. This just makes sure a person actually sees it
    happened, rather than it being findable only by grepping run_log later."""
    print("-" * 70)
    print("BUSINESS OUTCOME FLAGGED FOR HUMAN ATTENTION (run continued normally)")
    print(f"  goal:    {notice.goal}")
    print(f"  item:    {notice.item}")
    print(f"  outcome: {notice.outcome_name}")
    print("-" * 70)


def print_intervention(req: InterventionRequest, handoff: HandoffSession) -> None:
    print("=" * 70)
    print("HUMAN ESCALATION REQUIRED")
    print(f"  goal:      {req.goal}")
    print(f"  item:      {req.item}")
    print(f"  step:      {req.step}")
    print(f"  reason:    {req.reason}")
    print(f"  evidence:  {req.evidence_path}")
    print(f"  attach at: {handoff.cdp_url}")
    print(f"  {handoff.devtools_frontend_hint}")
    print("Execution is fully paused. The automated loop will not act again")
    print("until this operator signals 'done' or 'skip'.")
    print("=" * 70)


def _prompt_once(choices: tuple[str, ...]) -> str | None:
    """One prompt. Returns the operator's answer (lower-cased), '' for anything
    unrecognised (ask again), or None when stdin is gone entirely.

    That last case is the one worth naming: an unattended run -- CI, a cron
    invocation, anything with stdin closed -- hits EOF immediately. Previously
    that raised EOFError straight out of the default (no-timeout) path, crashing
    the replay *after* it had already escalated; and on the timeout path it
    killed the reader thread with an unhandled traceback printed to stderr, then
    reported "no response within 300s" for something that took no time at all.
    An operator who cannot be asked is simply an operator who did not answer.

    OSError is caught alongside EOFError because "stdin is unreadable" arrives
    under both names depending on how the process was started -- a closed pipe
    raises EOFError, a detached or redirected stdin raises OSError. Both mean
    the same thing here, and neither should reach the caller as a crash."""
    try:
        return input(f"Operator signal ({' / '.join(repr(c) for c in choices)}): ").strip().lower()
    except (EOFError, OSError):
        return None


def wait_for_operator_signal(
    auto_signal: str | None = None,
    timeout_s: float | None = None,
    choices: tuple[str, ...] = ("done", "skip"),
    default: str = "skip",
) -> str:
    """Blocks for the operator's decision -- one of `choices`. The original
    pair is 'done' (they resolved it, resume automation) / 'skip' (this item is
    skipped, not equivalent-actioned); the verification gates use
    'confirm' / 'reject' and 'confirm' / 'resume'. `auto_signal` lets an
    automated demo/test supply the signal non-interactively without changing
    the real control-flow path a human would take.

    `timeout_s`, if given, bounds how long an unattended escalation holds the
    live browser session open. Without it, a human who never responds blocks
    this forever -- there was no TTL here at all before.

    Every non-answer -- a timeout, an unreachable operator, a closed stdin --
    resolves to `default`, which every caller sets to the *conservative*
    option: 'skip', never 'done'; 'confirm the failure', never 'resume'. A
    non-answer must never be indistinguishable from a human affirmatively
    saying they fixed it or that something is correct. `default` may therefore
    be a value that is NOT one of `choices` -- a sentinel like 'unverified' --
    precisely so that a defaulted answer can never be mistaken for a typed one."""
    if auto_signal is not None:
        print(f"[operator signal, non-interactive demo]: {auto_signal}")
        return auto_signal

    if timeout_s is None:
        while True:
            signal = _prompt_once(choices)
            if signal is None:
                print(f"[operator signal] stdin is not available -- treating as {default!r}")
                return default
            if signal in choices:
                return signal

    result: dict[str, str] = {}

    def _read() -> None:
        while True:
            signal = _prompt_once(choices)
            if signal is None:
                result["signal"] = default
                result["reason"] = "stdin unavailable"
                return
            if signal in choices:
                result["signal"] = signal
                return

    reader = threading.Thread(target=_read, daemon=True)
    reader.start()
    reader.join(timeout_s)
    if "signal" in result:
        if reason := result.get("reason"):
            print(f"[operator signal] {reason} -- treating as {default!r}")
        return result["signal"]
    print(f"[operator signal] no response within {timeout_s}s -- treating as {default!r}")
    return default


def read_operator_note(prompt: str = "Optional note for the model (what you changed), or blank: ") -> str | None:
    """A free-text line from the operator, or None if stdin is unavailable."""
    try:
        return input(prompt).strip() or None
    except (EOFError, OSError):
        return None


def print_verification_request(
    goal: str, item: dict, result_type: str, detail: dict, screenshot: str | None, url: str, cdp_url: str
) -> None:
    """The non-success verification gate. A replay has classified its result as
    something other than a clean success -- a declared business outcome, a
    success with an output the extractor could not read -- and that
    classification is about to be handed to a caller as the answer. Before it
    is, a person is shown what the system saw and asked whether the
    classification is right. 'no_such_account' is a legitimate result, but a
    filter that failed to apply or a page that errored into its empty state
    would produce the same words, and the caller would receive a confident,
    wrong 'not found'."""
    print("=" * 70)
    print("HUMAN VERIFICATION REQUIRED -- is this classification correct?")
    print(f"  goal:        {goal}")
    print(f"  inputs:      {item}")
    print(f"  classified:  {result_type}")
    print(f"  detail:      {detail}")
    print(f"  final url:   {url}")
    print(f"  screenshot:  {screenshot}")
    print(f"  live session still open at: {cdp_url}  (look, but do not act -- 'reject' hands you control)")
    print("  'confirm' -> return this result to the caller as verified")
    print("  'reject'  -> the classification is wrong; take control of the session and fix it")
    print("=" * 70)


def print_discovery_stopped(goal: str, status: str, reason: str, detail: dict, log_path: str, cdp_url: str) -> None:
    """A recording did not produce an artifact. Before that is accepted, a
    person is asked to look: sometimes the model genuinely could not finish,
    and sometimes a popup it could not clear, a slow page, or a wrong turn is
    something a human can fix in the live session in ten seconds -- after which
    the model can carry on from where it stopped."""
    print("=" * 70)
    print(f"DISCOVERY STOPPED ({status}) -- no artifact built yet; needs a human to look")
    print(f"  goal:    {goal}")
    print(f"  reason:  {reason}")
    print(f"  detail:  {detail}")
    print(f"  log:     {log_path}")
    print(f"  live session still open at: {cdp_url}")
    print("  'confirm' -> yes, this run failed; discard it")
    print("  'resume'  -> I have intervened in the live session; let the model continue from here")
    print("=" * 70)
