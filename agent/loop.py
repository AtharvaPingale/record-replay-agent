"""The goal-driven agent loop: screenshot+DOM -> decision -> execute ->
re-observe -> repeat, with max-steps/timeout/dead-end stop conditions. This is
the production path for driving a single episode live.

The decision step is pluggable (agent/deciders.py): OllamaDecider (self-hosted,
no API key, no dependency beyond a reachable Ollama daemon -- localhost or
OLLAMA_HOST -- see README's "Using local or self-hosted models" section for
what that trades away) or AnthropicDecider
(hosted, vision-capable, needs ANTHROPIC_API_KEY). With no `decider` passed,
`agent/deciders.py`'s `default_decider()` picks for you: Claude if
ANTHROPIC_API_KEY is set, local Ollama otherwise -- one call works either way,
and the loop itself has no provider-specific code regardless of which runs.

`run_discovery_loop` below is the recording counterpart: the same cycle, plus
the declare_* tools, emitting a trace artifact/from_run.py can collapse into a
reusable capability with no human in between. See README.md's "What actually
produced the committed artifacts" for which committed artifact came from which
path, and why.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.deciders import Decider, default_decider
from telemetry.log import RunLog
from surfaces.base import Action, Observation, Surface
from surfaces.web import GuardrailBlocked


@dataclass
class StopCondition:
    max_steps: int = 25
    timeout_s: float = 300.0
    dead_end_repeats: int = 4  # same (action, target) repeated this many times in a row -> stop


@dataclass
class LoopResult:
    status: str  # "success" | "max_steps" | "timeout" | "dead_end" | "ask_user" | "decider_error"
    # ("blocked" was removed as a terminal status: a GuardrailBlocked action is
    # now logged and fed back to the decider like any other failed action,
    # never an unconditional stop -- see the except clause below.)
    steps_taken: int
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveryState:
    """Everything a recording session has accumulated that must survive a pause.

    Exists so a stuck recording can be *resumed* after a human intervenes in the
    live session, rather than discarded and started over: the steps recorded so
    far, the declarations made so far, and the step counter that keeps the next
    action from colliding with an earlier one in artifact/from_run.py's
    keyed-by-position collapse. `recent_actions` (dead-end detection) is
    deliberately NOT carried across a resume -- the human just changed the page,
    so the model deserves a fresh chance at an action it was repeating.

    `operator_note`, if set, is delivered to the model as its first
    `last_result` after the resume -- what the human did, in their words -- so
    the model is not left inferring why the page suddenly looks different."""

    declared: dict[str, Any] = field(default_factory=lambda: {
        "business_outcomes": [],
        "recoverable_patterns": [],
        "checkpoint": None,
        "output_schema": {},
        "output_extractors": [],
    })
    step_in_template: int = 0
    resumes: int = 0
    operator_note: str | None = None


class MalformedToolCall(Exception):
    """The decider named a real tool but left out a required argument -- seen
    live with a local model confusing 'the text to type' with 'text used to
    locate the field' before agent/tools.py's fields were renamed to disambiguate
    that. Kept as its own exception (distinct from ValueError for an unknown
    tool name entirely) so the loop can treat it as a bad turn to skip and
    retry rather than something that ends the run -- smaller/local models
    won't always follow a tool schema as reliably as a larger hosted one."""


def _timeout_detail(stop: StopCondition, start: float, steps_done: int) -> dict[str, Any]:
    """A bare "timeout" with an empty detail tells whoever reads it nothing --
    in particular, not the thing they actually need to know, which is whether
    the run was progressing and simply needed longer. Found live: a recording
    driven by the claude-cli backend (a subprocess per turn, so several times
    slower than an API call) completed 10 good steps and then stopped on the
    300s default with `detail={}`, which reads like a failure and was not one."""
    elapsed = time.time() - start
    return {
        "timeout_s": stop.timeout_s,
        "elapsed_s": round(elapsed, 1),
        "steps_completed": steps_done,
        "avg_s_per_step": round(elapsed / steps_done, 1) if steps_done else None,
        "hint": (
            f"raise --timeout-s (currently {stop.timeout_s:.0f}s) if the run was still "
            f"making progress -- a slower backend needs more wall clock, not more steps"
        ),
    }


DECIDER_RETRIES = 1  # one bounded retry of the same turn, never an open-ended loop


def _decide_with_retry(decider: Decider, goal: str, obs: Observation, last_result, pii_exempt, run_log: RunLog, step: int):
    """Call the decider; on an exception, log it, retry once, and if it fails
    again return None so the loop can end *gracefully* with a logged reason.

    Found live: a claude-cli turn exceeded its subprocess timeout and the
    RuntimeError went straight out of run_discovery_loop, through the script,
    and onto the terminal -- no log event, no status line, no banner, and a run
    that had completed six good steps simply vanished. A decider is a network
    call to something slow and fallible; one bad turn is exactly the kind of
    condition this loop already recovers from for a malformed tool call or a
    guardrail refusal, and it should be treated the same way."""
    for attempt in range(DECIDER_RETRIES + 1):
        try:
            return decider.step(goal, obs, last_result, exempt=pii_exempt)
        except Exception as e:  # noqa: BLE001 - any decider failure is data for the log, never a crash
            run_log.emit(
                "decider_error",
                {"attempt": attempt, "error": f"{type(e).__name__}: {e}"[:500], "will_retry": attempt < DECIDER_RETRIES},
                step=step,
            )
    return None


def _decider_failed(step: int) -> LoopResult:
    return LoopResult(
        "decider_error", step - 1,
        {"step": step, "hint": "the decider raised on this turn and again on retry -- see the decider_error events in log.jsonl; "
                               "for --backend claude-cli a slow turn usually means the resumed session grew too large"},
    )


def _tool_to_action(name: str, tool_input: dict[str, Any]) -> Action:
    target = []
    if tool_input.get("css_selector"):
        target.append({"kind": "dom_selector", "value": tool_input["css_selector"]})
    if tool_input.get("text") and name == "click":
        target.append({"kind": "text", "value": tool_input["text"]})
    if tool_input.get("find_field_by_text"):
        target.append({"kind": "text", "value": tool_input["find_field_by_text"]})
    # ocr_text always appended last -- locate.py's own priority order is DOM
    # selector -> visible text -> OCR text -> coordinates, and this list is
    # tried candidate by candidate in order, so putting it after
    # css_selector/text here is what actually enforces "DOM-grounded first,
    # pixels-only last resort" for a live decider's own click/type calls, not
    # just for a replayed artifact's declared target.
    ocr_value = tool_input.get("ocr_text") or tool_input.get("find_field_by_ocr_text")
    if ocr_value:
        target.append({"kind": "ocr_text", "value": {"text": ocr_value}})

    try:
        if name == "click":
            return Action(type="click", target=target)
        if name == "type":
            return Action(type="type", target=target, params={"text": tool_input["text"]})
        if name == "select":
            return Action(type="select", target=target, params={"value": tool_input["value"]})
        if name == "wait":
            return Action(type="wait", params={"ms": tool_input.get("ms", 500)})
        if name == "assert_text":
            return Action(type="assert_text", params={"text": tool_input["text"]})
        if name == "go_to":
            return Action(type="go_to", params={"url": tool_input["url"]})
        if name == "done":
            return Action(type="done", params=tool_input)
        if name == "ask_user":
            return Action(type="ask_user", params=tool_input)
    except KeyError as e:
        raise MalformedToolCall(f"tool '{name}' called without required argument {e}") from e
    raise ValueError(f"unknown tool {name}")


def run_agent_loop(
    goal: str,
    surface: Surface,
    run_log: RunLog,
    stop: StopCondition = StopCondition(),
    decider: Decider | None = None,
    model: str | None = None,
    pii_exempt: frozenset[str] | None = None,
) -> LoopResult:
    """`pii_exempt`: literal values (e.g. an account ID the goal itself names
    as the thing to search for) that guardrails/pii_redact.py should leave
    unmasked in the decider's prompt/screenshot -- see that module's
    docstring for why blind redaction breaks a goal shaped like "look up a
    user's balance by account ID"."""
    if decider is None:
        decider = default_decider(model)  # Claude if ANTHROPIC_API_KEY is set, local Ollama otherwise

    start = time.time()
    recent_actions: list[tuple[str, str]] = []
    last_result: dict[str, Any] | None = None

    obs = surface.observe()

    for step in range(1, stop.max_steps + 1):
        if time.time() - start > stop.timeout_s:
            return LoopResult("timeout", step - 1, _timeout_detail(stop, start, step - 1))

        decision = _decide_with_retry(decider, goal, obs, last_result, pii_exempt, run_log, step)
        if decision is None:
            return _decider_failed(step)

        run_log.emit_llm_turn(
            step=step,
            screenshot_path=obs.screenshot_path,
            prompt=decision.prompt,
            response=decision.response,
            decision={"tool": decision.tool_name, "input": decision.tool_input} if decision.tool_name else None,
            reason=decision.tool_input.get("reasoning") if decision.tool_name else None,
            extra={
                "vision_shown_to_model": decider.supports_vision,
                # Every tool call the model proposed this turn, not just the
                # one acted on -- full observability means the log shouldn't
                # silently drop what the model actually said just because
                # this loop's single-action-per-step design didn't use it.
                "all_tool_calls": decision.all_tool_calls,
                # guardrails/pii_redact.py's count of OCR-detected regions
                # masked in this turn's screenshot before it reached the
                # decider -- 0 for a non-vision turn or when nothing matched,
                # logged either way so a human auditing the run can see
                # redaction actually ran, not just assume it did.
                "pii_redacted_count": decision.pii_redacted_count,
                # False = the surface could not redact and therefore wrote no
                # screenshot for this turn -- the path above then names a file
                # that does not exist, on purpose.
                "screenshot_redacted": obs.screenshot_redacted,
            },
        )

        if decision.tool_name is None:
            obs = surface.observe()
            continue  # decider produced only text this turn; re-observe and try again

        if decision.tool_name == "done":
            return LoopResult("success", step, {"summary": decision.tool_input.get("summary")})
        if decision.tool_name == "ask_user":
            return LoopResult("ask_user", step, {"question": decision.tool_input.get("question")})

        try:
            action = _tool_to_action(decision.tool_name, decision.tool_input)
        except MalformedToolCall as e:
            last_result = {"ok": False, "error": str(e)}
            run_log.emit("malformed_tool_call", {"tool": decision.tool_name, "input": decision.tool_input, "error": str(e)}, step=step)
            obs = surface.observe()
            continue  # bad turn from the decider, not a fatal error -- try again next step, now told why it failed

        sig = (action.type, str(action.target) + str(action.params))
        recent_actions.append(sig)
        if recent_actions[-stop.dead_end_repeats :].count(sig) >= stop.dead_end_repeats:
            return LoopResult("dead_end", step, {"repeated": sig})

        try:
            result = surface.act(action)
        except GuardrailBlocked as e:
            # Same recoverability as a MalformedToolCall above, not an unconditional
            # stop: the guardrail already did its actual job here -- the blocked
            # action never executed, nothing unsafe happened -- so there's nothing
            # for terminating the whole run to additionally protect against. Found
            # live: a local model with nothing telling it a URL must be absolute
            # tried a relative go_to ('/products'), got refused (empty domain not
            # in the allowlist), and had no way to learn from that and try again.
            # Every subsequent action still passes through this same guardrail, so
            # feeding the reason back and continuing can't let anything slip past
            # it; dead_end_repeats still catches a model that just retries the
            # identical blocked call forever.
            run_log.emit("guardrail_block", {"action": action.type, "reason": e.reason}, step=step)
            last_result = {"ok": False, "error": f"guardrail_blocked: {e.reason}"}
            obs = surface.observe()
            continue

        run_log.emit("act", {"action": action.type, "params": action.params, "result": result}, step=step)
        last_result = result
        obs = surface.observe()

    return LoopResult("max_steps", stop.max_steps, {"hint": f"raise --max-steps (currently {stop.max_steps}) if the run was still making progress"})


_RECORDING_INSTRUCTIONS = (
    "\n\nYou are RECORDING a reusable capability, not just completing a one-off task -- "
    "this run's actions are being turned into a versioned artifact other callers will "
    "replay later, without any model in the loop. Before calling done: call "
    "declare_checkpoint exactly once, for the condition that proves the goal was really "
    "achieved. Call declare_output once per piece of data the caller should get back "
    "(skip entirely if there's nothing to return). For every click/type/select, always "
    "fill in 'reasoning' -- why this locator will still work next time, not just this "
    "time -- a future reviewer needs to see it, it is not optional narrative.\n\n"
    "The checkpoint must not depend on content that changes between runs. Prefer a "
    "URL pattern (url_contains), stable visible text such as a heading (element_visible), "
    "or a structural element that is always present on the destination page -- a row, a "
    "container, a control -- as a CSS selector (selector_visible; never pass a selector as "
    "element_visible, that type matches text). Do NOT use a value that happens to be "
    "on screen right now -- a specific product name, a view count, the title of "
    "whatever item is currently newest. A goal phrased 'the latest X' is exactly the "
    "trap: the literal title of today's latest X is the one thing guaranteed to be "
    "wrong later, and a checkpoint built on it turns every future replay into a hard "
    "failure. Same rule for a declare_output target: point at WHERE the value lives, "
    "not at the value you can currently read.\n\n"
    "IMPORTANT: if you hit something that isn't guaranteed to appear every time this "
    "runs -- a promo popup, a cookie-consent banner, an ad overlay, a 'sign in to "
    "continue' nag -- dismissing it is still the right thing to do RIGHT NOW, but do "
    "NOT record that dismissal as a required step. A future replay may never see it at "
    "all, and a hardcoded step whose target doesn't exist that time is a guaranteed "
    "hard failure, not a skip. Call declare_recoverable for it instead (the text/element "
    "that identifies the popup, and how you cleared it), then continue with the actual "
    "flow -- never add a step whose only purpose is clearing something session-specific."
)

_DECLARE_TOOL_NAMES = ("declare_checkpoint", "declare_output", "declare_business_outcome", "declare_recoverable")


def _declare_output_target(tool_input: dict[str, Any]) -> dict[str, str]:
    return {"kind": tool_input["target_kind"], "value": tool_input["target_value"]}


def _parameterize(value: Any, declared_inputs: dict[str, str]) -> tuple[Any, str | None]:
    """If `value` contains a declared input's literal, swap it for that input's
    `{{name}}` placeholder. Returns (new_value, field_name or None).

    This is how a recording becomes a *parameterized* capability rather than a
    literal replay of one run: the caller names the inputs up front
    (`record_capability.py --input account_number=1234567890`), and any value
    the model types, or declares as a checkpoint/extractor, that carries that
    literal is written down as the placeholder instead. Same mechanism as
    guardrails/redact.py's credential masking -- provenance by value -- applied
    to the one other class of value that must not be baked into an artifact.

    Three things fall out of one rule. The artifact gets a real `input_schema`
    and `{{field}}` steps, so replay takes typed arguments. The *contract* is
    parameterized too -- a checkpoint the model declared as the account number
    being visible becomes "{{account_number}} is visible", i.e. THE member asked
    about, which is what makes a lookup's success condition honest (see
    Artifact.with_inputs_rendered). And the literal -- here a real financial
    identifier -- never reaches the log or the artifact at all, instead of
    reaching them and then having to be redacted, which would corrupt the
    artifact built from that log. Longest values first so one input that is a
    substring of another cannot shadow it."""
    if not isinstance(value, str) or not declared_inputs:
        return value, None
    matched = None
    for name, literal in sorted(declared_inputs.items(), key=lambda kv: -len(kv[1])):
        if literal and literal in value:
            value = value.replace(literal, "{{" + name + "}}")
            matched = matched or name
    return value, matched


def run_discovery_loop(
    goal: str,
    surface: Surface,
    run_log: RunLog,
    run_dir: Path,
    template_id: str,
    stop: StopCondition = StopCondition(),
    decider: Decider | None = None,
    model: str | None = None,
    pii_exempt: frozenset[str] | None = None,
    declared_inputs: dict[str, str] | None = None,
    state: DiscoveryState | None = None,
) -> LoopResult:
    """`pii_exempt`: see run_agent_loop's own docstring for this param --
    same meaning here.

    `state`: pass the same DiscoveryState object back in to *resume* a session
    that stopped (dead_end, max_steps, timeout, ...) after a human has
    intervened in the live surface -- see DiscoveryState. A fresh recording
    leaves it None.

    `declared_inputs`: {field_name: literal_value} the caller names up front as
    the things a future invocation will vary. Any recorded value carrying one
    of those literals is written as `{{field_name}}` -- see _parameterize. The
    literals are also implicitly PII-exempt for the model's prompt: the caller
    supplied them, so showing them back is not a new exposure, and a lookup
    cannot confirm it found the right row if the identifier it was given is
    masked out of everything it sees.

    The recording loop: same observe -> decide -> act cycle as run_agent_loop,
    but the
    decider's tool schema also includes agent/tools.py's DISCOVERY_TOOLS
    (declare_checkpoint/declare_output/declare_business_outcome/
    declare_recoverable), and every real action gets logged in the exact
    shape artifact/from_run.py already knows how to read (step_template_id,
    item_index, step_in_template, robustness_reasoning) -- so a successful
    run here can go straight into build_artifact_from_run with no human, no
    curl, no Flask server in between. See scripts/record_capability.py for
    the CLI wrapper that actually does that last step.

    Typed inputs come from `declared_inputs` only. Without it this builds a
    single, literal recording of exactly what happened (var_params /
    var_field_names stay empty on every step). With it, every recorded value
    carrying a declared literal is parameterized at record time -- see
    _parameterize. What is still missing, and is a deliberate scope choice
    (REPORT.md §7), is asking the *model* which parts of its own run should
    generalize: today the caller says so up front, the model does not decide.
    """
    from agent.deciders import AnthropicDecider, OllamaDecider
    from agent.tools import DISCOVERY_TOOLS, TOOLS

    if decider is None:
        decider = default_decider(model, tools=TOOLS + DISCOVERY_TOOLS)
    elif isinstance(decider, (AnthropicDecider, OllamaDecider)) and decider._tools_override is None:
        # A caller-supplied decider built with the ordinary tool set still
        # needs the discovery tools added, or declare_checkpoint/declare_output
        # simply never appear as options -- same reasoning default_decider's
        # `tools` kwarg above exists for.
        decider._tools_override = TOOLS + DISCOVERY_TOOLS

    declared_inputs = declared_inputs or {}
    if declared_inputs:
        pii_exempt = frozenset(pii_exempt or ()) | frozenset(declared_inputs.values())
    augmented_goal = goal + _RECORDING_INSTRUCTIONS
    state = state or DiscoveryState()
    declared = state.declared
    step_in_template = state.step_in_template

    start = time.time()  # the wall-clock budget is per invocation: a human's intervention time is not charged to the model
    recent_actions: list[tuple[str, str]] = []
    last_result: dict[str, Any] | None = None
    if state.operator_note is not None:
        # Delivered exactly once, as the outcome of "the previous turn".
        last_result = {"ok": True, "operator_intervened": True, "operator_note": state.operator_note}
        run_log.emit("operator_resume", {"note": state.operator_note, "resume_number": state.resumes}, step=None)
        state.operator_note = None

    obs = surface.observe()

    for step in range(1, stop.max_steps + 1):
        if time.time() - start > stop.timeout_s:
            return LoopResult("timeout", step - 1, _timeout_detail(stop, start, step - 1))

        decision = _decide_with_retry(decider, augmented_goal, obs, last_result, pii_exempt, run_log, step)
        if decision is None:
            return _decider_failed(step)

        run_log.emit_llm_turn(
            step=step,
            screenshot_path=obs.screenshot_path,
            prompt=decision.prompt,
            response=decision.response,
            decision={"tool": decision.tool_name, "input": decision.tool_input} if decision.tool_name else None,
            reason=decision.tool_input.get("reasoning") if decision.tool_name else None,
            extra={
                "vision_shown_to_model": decider.supports_vision,
                # Every tool call the model proposed this turn, not just the
                # one acted on -- full observability means the log shouldn't
                # silently drop what the model actually said just because
                # this loop's single-action-per-step design didn't use it.
                "all_tool_calls": decision.all_tool_calls,
                # guardrails/pii_redact.py's count of OCR-detected regions
                # masked in this turn's screenshot before it reached the
                # decider -- 0 for a non-vision turn or when nothing matched,
                # logged either way so a human auditing the run can see
                # redaction actually ran, not just assume it did.
                "pii_redacted_count": decision.pii_redacted_count,
                # False = the surface could not redact and therefore wrote no
                # screenshot for this turn -- the path above then names a file
                # that does not exist, on purpose.
                "screenshot_redacted": obs.screenshot_redacted,
            },
        )

        if decision.tool_name is None:
            obs = surface.observe()
            continue

        if decision.tool_name == "done":
            if declared["checkpoint"] is None:
                # Never let a caller silently get a checkpoint-less artifact --
                # build_artifact_from_run needs declared["checkpoint"] to exist
                # at all (Checkpoint(**declared["checkpoint"]) has nothing to
                # unpack otherwise). Reported as a distinct outcome, not an
                # exception, since the live goal genuinely was achieved -- only
                # the recording half of it is incomplete.
                run_log.emit("recording_incomplete", {"reason": "done called without declare_checkpoint"}, step=step)
                return LoopResult(
                    "success", step,
                    {"summary": decision.tool_input.get("summary"), "artifact_buildable": False, "reason": "no checkpoint declared"},
                )
            (run_dir / "declared_outcomes.json").write_text(json.dumps(declared, indent=2))
            return LoopResult(
                "success", step,
                {"summary": decision.tool_input.get("summary"), "artifact_buildable": True, "declared_outcomes_path": str(run_dir / "declared_outcomes.json")},
            )
        if decision.tool_name == "ask_user":
            return LoopResult("ask_user", step, {"question": decision.tool_input.get("question")})

        if decision.tool_name in _DECLARE_TOOL_NAMES:
            # A declaration, not a UI action -- never touches the surface,
            # never advances step_in_template (only real actions get a step
            # number; from_run.py's canonical-step sequence is built from
            # those alone). Still needs the same dead_end check real actions
            # get below, though: found live, with a local model, that calling
            # declare_output for the exact same field over and over (each
            # turn re-describing its plan in prose instead of ever calling
            # declare_checkpoint or done) burned through every remaining step
            # silently -- output_extractors' upsert-by-name means the
            # declaration itself is harmless to repeat, but nothing was
            # catching that the model had stopped making progress, so a run
            # that should have ended in a few steps with an honest "dead_end"
            # only ever reported an uninformative "max_steps" instead.
            ti = decision.tool_input
            sig = (decision.tool_name, str(ti))
            recent_actions.append(sig)
            if recent_actions[-stop.dead_end_repeats :].count(sig) >= stop.dead_end_repeats:
                return LoopResult("dead_end", step, {"repeated": sig})
            recorded: dict[str, Any]  # what actually went into the contract (placeholders, not literals)
            if decision.tool_name == "declare_checkpoint":
                recorded = {"type": ti["type"], "value": _parameterize(ti["value"], declared_inputs)[0]}
                declared["checkpoint"] = recorded
            elif decision.tool_name == "declare_output":
                # Found live: a model re-verifying its own work before calling
                # done can genuinely call declare_output for the same field
                # more than once in a run -- output_schema (a dict, keyed by
                # name) already dedupes for free, but output_extractors was a
                # bare list.append, so a real run produced 5 duplicate
                # {name, target} pairs for the same two fields. Upsert by
                # name instead -- the latest declaration for a given name
                # wins, matching from_run.py's own "corrected version wins"
                # discipline for a re-recorded step.
                name = ti["name"]
                declared["output_schema"][name] = ti.get("value_type", "string")
                declared["output_extractors"] = [e for e in declared["output_extractors"] if e["name"] != name]
                target = _declare_output_target(ti)
                target["value"] = _parameterize(target["value"], declared_inputs)[0]
                recorded = {"name": name, "value_type": ti.get("value_type", "string"), "target": target}
                declared["output_extractors"].append({"name": name, "target": target})
            elif decision.tool_name == "declare_business_outcome":
                recorded = {"name": ti["name"], "match": {"type": ti["match_type"], "value": _parameterize(ti["match_value"], declared_inputs)[0]}}
                declared["business_outcomes"].append(recorded)
            elif decision.tool_name == "declare_recoverable":
                recorded = {"name": ti["name"], "match": {"type": ti["match_type"], "value": _parameterize(ti["match_value"], declared_inputs)[0]}, "recovery": ti["recovery"]}
                declared["recoverable_patterns"].append(recorded)
            # Log what was *recorded into the contract*, placeholders included --
            # the raw tool call is already in this turn's llm_turn event.
            run_log.emit(decision.tool_name, recorded, step=step)
            last_result = {"ok": True}
            obs = surface.observe()
            continue

        try:
            action = _tool_to_action(decision.tool_name, decision.tool_input)
        except MalformedToolCall as e:
            last_result = {"ok": False, "error": str(e)}
            run_log.emit("malformed_tool_call", {"tool": decision.tool_name, "input": decision.tool_input, "error": str(e)}, step=step)
            obs = surface.observe()
            continue

        sig = (action.type, str(action.target) + str(action.params))
        recent_actions.append(sig)
        if recent_actions[-stop.dead_end_repeats :].count(sig) >= stop.dead_end_repeats:
            return LoopResult("dead_end", step, {"repeated": sig})

        try:
            result = surface.act(action)
        except GuardrailBlocked as e:
            run_log.emit("guardrail_block", {"action": action.type, "reason": e.reason}, step=step)
            last_result = {"ok": False, "error": f"guardrail_blocked: {e.reason}"}
            obs = surface.observe()
            continue

        # Only a *successful* action advances the canonical step sequence. A
        # failed attempt reuses the same slot, so when the model retries (with a
        # corrected locator, or a different action entirely) the working version
        # overwrites the broken one via from_run.py's keyed-by-position collapse
        # -- rather than both landing in the artifact and the broken one
        # guaranteeing a hard_failure on every replay.
        if result.get("ok"):
            step_in_template += 1
            state.step_in_template = step_in_template
        # The model's own prose gets the same treatment as its typed values: a
        # robustness_reasoning that says "the account number 1234567890 is the
        # identifier the goal pins on" has baked one recording's literal into a
        # reusable artifact's documentation -- found live, this was the one
        # place the literal survived. "{{account_number}} is the identifier..."
        # is both clean and the truer sentence about a parameterized capability.
        reasoning = _parameterize(
            decision.tool_input.get("reasoning") or (
                "no reasoning given during recording" if action.target else "no element target for this action"
            ),
            declared_inputs,
        )[0]
        # A model driving a login form types the real credential as a literal.
        # Swap any known secret back for its {{env:VAR}} marker before this is
        # written down -- otherwise it lands in log.jsonl and, via
        # artifact/from_run.py, in the artifact itself. The marker is exactly
        # what replay already resolves, so the recording stays replayable.
        from guardrails.redact import mask_known_credential_values

        logged_params, _ = mask_known_credential_values(action.params)
        # A param carrying a declared input's literal is recorded as that input's
        # placeholder, in the exact shape artifact/from_run.py already turns into
        # a typed input_schema field -- see _parameterize.
        var_params: list[str] = []
        var_field_names: dict[str, str] = {}
        for key, value in list(logged_params.items()):
            new_value, field = _parameterize(value, declared_inputs)
            if field is not None:
                logged_params[key] = new_value
                var_params.append(key)
                var_field_names[key] = field
        detail: dict[str, Any] = {
            "step_template_id": template_id,
            "item_index": 0,
            "step_in_template": step_in_template,
            "action": action.type,
            "target": action.target,
            "params": logged_params,
            "robustness_reasoning": reasoning,
            "var_params": var_params,
            "var_field_names": var_field_names,
            "result": result,
        }
        if decision.tool_input.get("idempotency_note"):
            detail["idempotency_note"] = _parameterize(decision.tool_input["idempotency_note"], declared_inputs)[0]
        run_log.emit("act", detail, step=step)
        last_result = result
        obs = surface.observe()

    return LoopResult("max_steps", stop.max_steps, {"hint": f"raise --max-steps (currently {stop.max_steps}) if the run was still making progress"})
