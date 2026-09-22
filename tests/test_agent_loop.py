"""Unit tests for agent/loop.py's control flow -- pure logic against fake
Surface/Decider stubs, no browser and no real model, same pattern as
test_classify.py's FakeSurface. Covers the one real bug found live: a
GuardrailBlocked action used to be an unconditional stop; it's now logged and
fed back to the decider like any other failed action.
"""

from __future__ import annotations

from pathlib import Path

from agent.deciders import DecisionResult
from agent.loop import StopCondition, _tool_to_action, run_agent_loop
from surfaces.base import Action, Observation
from surfaces.web import GuardrailBlocked
from telemetry.log import RunLog


def test_tool_to_action_orders_ocr_text_last_after_dom_and_visible_text():
    # locator/locate.py's own priority chain is DOM selector -> visible text
    # -> OCR text -> coordinates, tried candidate by candidate in the order
    # target lists them -- so a live decider's click/type calls only actually
    # get "DOM-grounded first, pixels-only last resort" if _tool_to_action
    # appends ocr_text after css_selector/text here, not before.
    action = _tool_to_action(
        "click", {"css_selector": "#a", "text": "Submit", "ocr_text": "Submit", "reasoning": "x"}
    )
    assert action.target == [
        {"kind": "dom_selector", "value": "#a"},
        {"kind": "text", "value": "Submit"},
        {"kind": "ocr_text", "value": {"text": "Submit"}},
    ]


def test_tool_to_action_ocr_text_alone_still_produces_a_usable_target():
    # The real motivation: canvas-rendered or otherwise DOM-less content has
    # no css_selector/text candidate to give at all.
    action = _tool_to_action("click", {"ocr_text": "Play", "reasoning": "canvas-rendered button"})
    assert action.target == [{"kind": "ocr_text", "value": {"text": "Play"}}]


def test_tool_to_action_find_field_by_ocr_text_for_type():
    action = _tool_to_action("type", {"find_field_by_ocr_text": "Search", "text": "hello"})
    assert action.target == [{"kind": "ocr_text", "value": {"text": "Search"}}]
    assert action.params == {"text": "hello"}


class FakeSurface:
    """Only implements what run_agent_loop actually calls: observe() and
    act(). `act` raises GuardrailBlocked exactly once (on the first go_to),
    then succeeds -- simulating a model that tries a bad URL, gets refused,
    and is given a chance to correct itself."""

    def __init__(self):
        self.blocked_once = False
        self.acted: list[Action] = []

    def observe(self) -> Observation:
        return Observation(url="https://example.com/", title="", screenshot_path=Path("/dev/null"), dom_excerpt="", timestamp=0.0)

    def act(self, action: Action) -> dict:
        self.acted.append(action)
        if action.type == "go_to" and not self.blocked_once:
            self.blocked_once = True
            raise GuardrailBlocked("path '/products' matches no allowed pattern")
        return {"ok": True}


class ScriptedDecider:
    """Replays a fixed sequence of tool calls, one per step() call, ignoring
    goal/obs/last_result -- deterministic and fast, no live model needed."""

    supports_vision = False

    def __init__(self, script: list[tuple[str, dict]]):
        self.script = script
        self.calls = 0

    def step(self, goal, obs, last_result=None, exempt=None) -> DecisionResult:
        name, tool_input = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return DecisionResult(prompt="", response="", tool_name=name, tool_input=tool_input)


def _run_log(tmp_path) -> RunLog:
    return RunLog(run_id="test", out_dir=tmp_path)


def test_guardrail_blocked_action_is_fed_back_not_a_terminal_stop(tmp_path):
    surface = FakeSurface()
    decider = ScriptedDecider(
        [
            ("go_to", {"url": "/products"}),  # refused once by FakeSurface
            ("go_to", {"url": "https://example.com/products"}),  # tries again, succeeds
            ("done", {"summary": "reached products"}),
        ]
    )
    result = run_agent_loop(goal="reach products", surface=surface, run_log=_run_log(tmp_path), decider=decider)

    assert result.status == "success", "a guardrail block must not end the run early"
    assert result.steps_taken == 3
    # the blocked call really was attempted (and really was refused) -- this
    # isn't a test that silently passed because nothing was ever tried
    assert len(surface.acted) == 2
    assert surface.acted[0].params["url"] == "/products"


def test_guardrail_block_is_logged(tmp_path):
    surface = FakeSurface()
    decider = ScriptedDecider([("go_to", {"url": "/products"}), ("done", {})])
    run_log = _run_log(tmp_path)
    run_agent_loop(goal="x", surface=surface, run_log=run_log, decider=decider)

    events = [e for e in run_log.read_all() if e["kind"] == "guardrail_block"]
    assert len(events) == 1
    assert events[0]["detail"]["action"] == "go_to"
    assert events[0]["detail"]["reason"] == "path '/products' matches no allowed pattern"


def test_repeated_identical_blocked_call_still_hits_dead_end(tmp_path):
    """A model that just keeps retrying the exact same refused call must not
    loop forever -- dead_end_repeats is the safety net now that a single
    block is no longer an automatic stop."""

    class AlwaysBlockedSurface:
        def observe(self) -> Observation:
            return Observation(url="https://example.com/", title="", screenshot_path=Path("/dev/null"), dom_excerpt="", timestamp=0.0)

        def act(self, action: Action) -> dict:
            raise GuardrailBlocked("always refused")

    decider = ScriptedDecider([("go_to", {"url": "/nope"})])  # same call every time
    result = run_agent_loop(
        goal="x",
        surface=AlwaysBlockedSurface(),
        run_log=_run_log(tmp_path),
        decider=decider,
        stop=StopCondition(max_steps=25, dead_end_repeats=4),
    )

    assert result.status == "dead_end"
    assert result.steps_taken == 4


def test_a_timeout_reports_what_it_got_through_not_just_the_word_timeout(tmp_path):
    """A bare "timeout" with an empty detail reads like a failure and often
    isn't one. Found live: a recording on the claude-cli backend (a subprocess
    per turn, so several times slower than an API call) completed 10 good steps
    and stopped on the 300s default with `detail={}` -- nothing in that output
    said the run had been progressing and simply needed a longer budget."""
    surface = FakeSurface()
    decider = ScriptedDecider([("wait", {"ms": 1})])
    # timeout_s=0 trips the check on the very first iteration.
    result = run_agent_loop(
        goal="g", surface=surface, run_log=_run_log(tmp_path), decider=decider,
        stop=StopCondition(max_steps=5, timeout_s=0.0),
    )

    assert result.status == "timeout"
    assert result.detail["timeout_s"] == 0.0
    assert "elapsed_s" in result.detail and "steps_completed" in result.detail
    assert "--timeout-s" in result.detail["hint"], "must point at the knob that fixes it"


def test_a_generous_timeout_does_not_interfere(tmp_path):
    surface = FakeSurface()
    decider = ScriptedDecider([("done", {"summary": "ok"})])
    result = run_agent_loop(
        goal="g", surface=surface, run_log=_run_log(tmp_path), decider=decider,
        stop=StopCondition(max_steps=5, timeout_s=300.0),
    )
    assert result.status == "success"


class FlakyDecider:
    """Raises on the first N step() calls, then behaves like ScriptedDecider.
    Models a decider backend that is slow or fallible -- a subprocess timeout,
    a transient API error -- which is what any real one is."""

    supports_vision = False

    def __init__(self, fail_first: int, then: list[tuple[str, dict]]):
        self.fail_first = fail_first
        self.inner = ScriptedDecider(then)
        self.calls = 0

    def step(self, goal, obs, last_result=None, exempt=None) -> DecisionResult:
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RuntimeError(f"claude CLI call failed (TimeoutExpired) on call {self.calls}")
        return self.inner.step(goal, obs, last_result, exempt)


def test_one_decider_failure_is_retried_and_the_run_continues(tmp_path):
    # Found live: a single slow claude-cli turn raised straight out of the loop
    # and a run with six good steps vanished with nothing logged.
    surface = FakeSurface()
    decider = FlakyDecider(fail_first=1, then=[("done", {"summary": "ok"})])
    log = _run_log(tmp_path)
    result = run_agent_loop(goal="g", surface=surface, run_log=log, decider=decider)

    assert result.status == "success"
    errors = [e for e in log.read_all() if e["kind"] == "decider_error"]
    assert len(errors) == 1 and errors[0]["detail"]["will_retry"] is True
    assert "TimeoutExpired" in errors[0]["detail"]["error"]


def test_a_decider_that_fails_twice_ends_the_run_gracefully_not_with_a_traceback(tmp_path):
    surface = FakeSurface()
    decider = FlakyDecider(fail_first=99, then=[("done", {})])
    log = _run_log(tmp_path)
    result = run_agent_loop(goal="g", surface=surface, run_log=log, decider=decider)  # must not raise

    assert result.status == "decider_error"
    assert result.detail["step"] == 1
    assert "claude-cli" in result.detail["hint"]
    errors = [e for e in log.read_all() if e["kind"] == "decider_error"]
    assert [e["detail"]["will_retry"] for e in errors] == [True, False], "exactly one bounded retry"
    assert surface.acted == [], "no action may execute on a turn the decider never decided"


def test_decider_drops_the_image_when_the_surface_withheld_the_screenshot(tmp_path):
    # surfaces/web.py writes nothing when it cannot redact; the decider must
    # then send nothing rather than go looking for a raw file.
    from agent.deciders import _image_b64

    shot = tmp_path / "001.png"
    shot.write_bytes(b"not really written by a surface")
    obs = Observation(url="u", title="t", screenshot_path=shot, dom_excerpt="", timestamp=0.0,
                      pii_redacted_count=3, screenshot_redacted=True)
    b64, count = _image_b64(obs)
    assert b64 is not None and count == 3, "a redacted file is sent, with the surface's own count"

    obs.screenshot_redacted = False
    assert _image_b64(obs) == (None, 0), "withheld by the surface -> nothing sent"

    obs.screenshot_redacted = True
    obs.screenshot_path = tmp_path / "missing.png"
    assert _image_b64(obs) == (None, 0), "missing file -> nothing sent"
