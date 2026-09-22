"""Unit tests for agent/loop.py's run_discovery_loop -- the autonomous
equivalent of the curl-driven harness this repo used to have. Same
FakeSurface/ScriptedDecider pattern as test_agent_loop.py, extended with
agent/tools.py's DISCOVERY_TOOLS (declare_checkpoint/declare_output/etc.).

The load-bearing assertion isn't just "the loop's control flow behaves" --
it's that a successful scripted run produces a log.jsonl + declared_outcomes.json
that artifact/from_run.py's build_artifact_from_run can actually turn into a
real Artifact, with no browser and no live model needed to prove it. That's
the actual claim scripts/record_capability.py makes.
"""

from __future__ import annotations

import json

from pathlib import Path

from agent.deciders import DecisionResult
from agent.loop import StopCondition, run_discovery_loop
from artifact.from_run import build_artifact_from_run
from surfaces.base import Action, Observation
from telemetry.log import RunLog


class FakeSurface:
    def __init__(self):
        self.acted: list[Action] = []

    def observe(self) -> Observation:
        return Observation(url="https://example.com/", title="", screenshot_path=Path("/dev/null"), dom_excerpt="", timestamp=0.0)

    def act(self, action: Action) -> dict:
        self.acted.append(action)
        return {"ok": True}


class ScriptedDecider:
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


def test_full_recording_produces_a_real_buildable_artifact(tmp_path):
    decider = ScriptedDecider(
        [
            ("click", {"css_selector": "#search", "reasoning": "stable id, this site's own search box"}),
            ("type", {"css_selector": "#search", "text": "some query", "reasoning": "same stable id"}),
            ("declare_output", {"name": "view_count", "target_kind": "text", "target_value": "views"}),
            ("declare_checkpoint", {"type": "text_present", "value": "views"}),
            ("done", {"summary": "got the view count"}),
        ]
    )
    surface = FakeSurface()
    run_log = _run_log(tmp_path)

    result = run_discovery_loop(
        goal="get the view count",
        surface=surface,
        run_log=run_log,
        run_dir=tmp_path,
        template_id="get_stats",
        decider=decider,
    )

    assert result.status == "success"
    assert result.detail["artifact_buildable"] is True
    assert (tmp_path / "declared_outcomes.json").exists()

    # Only the two real UI actions (click, type) touched the surface --
    # declare_output/declare_checkpoint/done never call surface.act.
    assert [a.type for a in surface.acted] == ["click", "type"]

    # The actual proof: feed this run's log + declarations through the same
    # builder the real recording path uses, and get a
    # real, valid Artifact back -- no browser, no live model, just this run's
    # own recorded output.
    artifact = build_artifact_from_run(
        run_log_path=str(run_log.path),
        declared_outcomes_path=str(tmp_path / "declared_outcomes.json"),
        artifact_id="get-stats",
        template_id="get_stats",
        base_url_pattern="https://example.com/*",
    )
    assert [s.action for s in artifact.steps] == ["click", "type"]
    assert artifact.steps[0].step == 1
    assert artifact.steps[1].step == 2
    assert artifact.steps[0].target.robustness_reasoning == "stable id, this site's own search box"
    assert artifact.checkpoint.type == "text_present" and artifact.checkpoint.value == "views"
    assert artifact.output_schema == {"view_count": "string"}
    assert artifact.output_extractors[0].name == "view_count"
    assert artifact.output_extractors[0].target.kind == "text"


def test_declare_tools_never_advance_step_in_template(tmp_path):
    decider = ScriptedDecider(
        [
            ("declare_business_outcome", {"name": "not_found", "match_type": "text_present", "match_value": "No results"}),
            ("click", {"text": "Search", "reasoning": "visible label"}),
            ("declare_recoverable", {"name": "popup", "match_type": "text_present", "match_value": "Subscribe", "recovery": "press Escape"}),
            ("declare_checkpoint", {"type": "text_present", "value": "Result"}),
            ("done", {}),
        ]
    )
    surface = FakeSurface()
    run_log = _run_log(tmp_path)

    run_discovery_loop(
        goal="x", surface=surface, run_log=run_log, run_dir=tmp_path, template_id="t", decider=decider,
    )

    act_events = [e for e in run_log.read_all() if e["kind"] == "act"]
    assert len(act_events) == 1  # only the one real click
    assert act_events[0]["detail"]["step_in_template"] == 1

    declared = (tmp_path / "declared_outcomes.json").read_text()
    assert "not_found" in declared
    assert "popup" in declared


def test_done_without_declare_checkpoint_does_not_produce_a_buildable_artifact(tmp_path):
    decider = ScriptedDecider([("click", {"text": "OK", "reasoning": "visible"}), ("done", {"summary": "finished"})])
    surface = FakeSurface()
    run_log = _run_log(tmp_path)

    result = run_discovery_loop(
        goal="x", surface=surface, run_log=run_log, run_dir=tmp_path, template_id="t", decider=decider,
    )

    assert result.status == "success"  # the live goal was still achieved
    assert result.detail["artifact_buildable"] is False
    assert not (tmp_path / "declared_outcomes.json").exists()


def test_repeated_declare_output_for_the_same_name_does_not_duplicate(tmp_path):
    # Real bug found live: a model re-verifying its own work before calling
    # done called declare_output 5 times for the same two field names, and a
    # bare list.append produced 10 duplicate {name, target} entries in a real
    # artifact. Upsert-by-name is the fix.
    decider = ScriptedDecider(
        [
            ("declare_output", {"name": "view_count", "target_kind": "text", "target_value": "496K views"}),
            ("declare_output", {"name": "view_count", "target_kind": "text", "target_value": "496K views"}),
            ("declare_output", {"name": "view_count", "target_kind": "dom_selector", "target_value": "#count"}),  # a later, corrected declaration
            ("declare_checkpoint", {"type": "text_present", "value": "views"}),
            ("done", {}),
        ]
    )
    surface = FakeSurface()
    run_log = _run_log(tmp_path)

    run_discovery_loop(goal="x", surface=surface, run_log=run_log, run_dir=tmp_path, template_id="t", decider=decider)

    import json

    declared = json.loads((tmp_path / "declared_outcomes.json").read_text())
    assert len(declared["output_extractors"]) == 1
    # the LATEST declaration for a repeated name wins, not the first
    assert declared["output_extractors"][0]["target"]["kind"] == "dom_selector"
    assert declared["output_schema"] == {"view_count": "string"}


def test_a_model_stuck_repeating_the_same_declaration_is_caught_as_dead_end(tmp_path):
    # Real bug found live: a local model (qwen2.5:14b-instruct) driving
    # scripts/record_capability.py called declare_output correctly once, then
    # just re-described its own plan in prose every subsequent turn without
    # ever calling declare_checkpoint or done. The identical repeated
    # declare_output call should have been caught the same way a repeated
    # click/type is -- but the dead_end check only ever ran for real UI
    # actions, never for declare_* calls (they `continue` before reaching
    # it), so this silently burned through the entire step budget and
    # reported an uninformative "max_steps" instead of "dead_end".
    same_call = ("declare_output", {"name": "heading", "target_kind": "dom_selector", "target_value": "h1"})
    decider = ScriptedDecider([same_call] * 10)
    surface = FakeSurface()
    run_log = _run_log(tmp_path)

    result = run_discovery_loop(
        goal="x", surface=surface, run_log=run_log, run_dir=tmp_path, template_id="t", decider=decider,
        stop=StopCondition(max_steps=10, dead_end_repeats=4),
    )

    assert result.status == "dead_end"
    assert result.steps_taken == 4  # caught on the 4th identical repeat, not burned to max_steps
    assert not (tmp_path / "declared_outcomes.json").exists()


def test_declared_inputs_become_placeholders_in_steps_and_in_the_contract(tmp_path):
    """The parameter-generalization gap, closed for caller-declared inputs: the
    recorder is told `account_number=1234567890` up front, the model types and
    declares the literal, and everything that carried it -- the typed step, the
    checkpoint, the extractor target -- is written as {{account_number}}. The
    literal (a real financial identifier) never reaches the log or the artifact
    at all, so it never has to be redacted out of the one and then corrupt the
    other."""
    surface = FakeSurface()
    decider = ScriptedDecider([
        ("type", {"css_selector": "#search", "text": "1234567890",
                  "reasoning": "account 1234567890 is the identifier the goal pins on",
                  "idempotency_note": "retyping 1234567890 is a no-op"}),
        ("declare_checkpoint", {"type": "element_visible", "value": "1234567890"}),
        ("declare_output", {"name": "balance", "value_type": "string", "target_kind": "dom_selector",
                            "target_value": '[data-testid^="user-row-"]:has-text("1234567890") .balance'}),
        ("done", {"summary": "ok"}),
    ])
    log = _run_log(tmp_path)
    result = run_discovery_loop(
        goal="look up member 1234567890", surface=surface, run_log=log, run_dir=tmp_path,
        template_id="t", decider=decider, declared_inputs={"account_number": "1234567890"},
    )
    assert result.status == "success" and result.detail["artifact_buildable"]

    # The literal is allowed in the observability fields -- the prompt the model
    # saw and the raw tool call it made -- because a reviewer needs it to verify
    # the run looked up the right record. It must NOT be in anything that
    # becomes the artifact: the recorded action params and the declarations.
    assert "1234567890" not in (tmp_path / "declared_outcomes.json").read_text()
    for line in (tmp_path / "log.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e["kind"] == "act":
            assert "1234567890" not in json.dumps(e["detail"]["params"]), "typed literal must be recorded as a placeholder"
            assert "1234567890" not in e["detail"]["robustness_reasoning"]
        if e["kind"].startswith("declare"):
            assert "1234567890" not in json.dumps(e["detail"]), "declarations are logged as recorded into the contract"

    artifact = build_artifact_from_run(tmp_path / "log.jsonl", tmp_path / "declared_outcomes.json", "a", "t", "https://e.com/*")
    assert artifact.input_schema == {"account_number": "string"}
    assert artifact.steps[0].params == {"text": "{{account_number}}"}
    # the model's prose is parameterized too -- found live, this was the one place the literal survived
    assert artifact.steps[0].target.robustness_reasoning == "account {{account_number}} is the identifier the goal pins on"
    assert artifact.steps[0].idempotency_note == "retyping {{account_number}} is a no-op"
    assert artifact.checkpoint.value == "{{account_number}}"
    assert artifact.output_extractors[0].target.value == '[data-testid^="user-row-"]:has-text("{{account_number}}") .balance'

    # and the round trip: a different caller argument renders a different concrete contract
    rendered = artifact.with_inputs_rendered({"account_number": "2345678901"})
    assert rendered.checkpoint.value == "2345678901"
    assert 'has-text("2345678901")' in rendered.output_extractors[0].target.value


def test_declared_inputs_are_pii_exempt_for_the_model(tmp_path):
    # A lookup cannot confirm it found the right row if the identifier it was
    # given is masked out of everything it sees. The caller supplied it, so
    # showing it back is not a new exposure.
    seen: dict = {}

    class SpyDecider(ScriptedDecider):
        def step(self, goal, obs, last_result=None, exempt=None):
            seen["exempt"] = exempt
            return super().step(goal, obs, last_result, exempt)

    decider = SpyDecider([("done", {"summary": "ok"})])
    run_discovery_loop(goal="g", surface=FakeSurface(), run_log=_run_log(tmp_path), run_dir=tmp_path,
                       template_id="t", decider=decider, declared_inputs={"account_number": "1234567890"})
    assert "1234567890" in seen["exempt"]


# -- resumable recording: a human's intervention is a pause, not a discard ----


def test_a_resumed_recording_keeps_its_steps_and_declarations(tmp_path):
    """The verification gate's 'resume' branch: the model got stuck, a human
    fixed the live page, and the model continues from exactly where it stopped
    -- steps recorded so far kept, checkpoint already declared kept, step
    numbering continuing so nothing collides in from_run's collapse."""
    from agent.loop import DiscoveryState

    surface = FakeSurface()
    state = DiscoveryState()

    # Session 1: two real steps, a checkpoint, then the model wedges on a repeated click -> dead_end.
    first = ScriptedDecider([
        ("click", {"css_selector": "#login", "reasoning": "r"}),
        ("type", {"css_selector": "#q", "text": "x", "reasoning": "r"}),
        ("declare_checkpoint", {"type": "url_contains", "value": "/done"}),
        ("click", {"css_selector": "#stuck", "reasoning": "r"}),  # repeats until dead_end
    ])
    r1 = run_discovery_loop(goal="g", surface=surface, run_log=_run_log(tmp_path), run_dir=tmp_path,
                            template_id="t", decider=first, state=state)
    assert r1.status == "dead_end"
    assert state.step_in_template >= 2 and state.declared["checkpoint"] is not None

    # A human clears the blocker and resumes with a note; session 2 finishes.
    steps_before = state.step_in_template
    state.resumes += 1
    state.operator_note = "dismissed the popup that was covering the button"
    seen: dict = {}

    class NoteSpy(ScriptedDecider):
        def step(self, goal, obs, last_result=None, exempt=None):
            seen.setdefault("first_last_result", last_result)
            return super().step(goal, obs, last_result, exempt)

    second = NoteSpy([
        ("click", {"css_selector": "#continue", "reasoning": "r"}),
        ("done", {"summary": "finished"}),
    ])
    r2 = run_discovery_loop(goal="g", surface=surface, run_log=_run_log(tmp_path), run_dir=tmp_path,
                            template_id="t", decider=second, state=state)

    assert r2.status == "success" and r2.detail["artifact_buildable"]
    assert seen["first_last_result"]["operator_intervened"] is True
    assert "dismissed the popup" in seen["first_last_result"]["operator_note"]
    assert state.operator_note is None, "delivered exactly once"

    artifact = build_artifact_from_run(tmp_path / "log.jsonl", tmp_path / "declared_outcomes.json", "a", "t", "https://e.com/*")
    step_numbers = [s.step for s in artifact.steps]
    assert step_numbers == sorted(step_numbers) and len(step_numbers) == len(set(step_numbers)), "no collisions across the resume"
    assert step_numbers[-1] > steps_before, "the resumed session's steps continue the numbering"
    assert artifact.checkpoint.value == "/done", "the checkpoint declared before the pause survives it"
