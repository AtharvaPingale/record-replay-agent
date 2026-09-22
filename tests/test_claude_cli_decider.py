"""Unit tests for ClaudeCodeCLIDecider -- drives decisions by shelling out to
the `claude` CLI instead of the Anthropic Messages API. subprocess.run is
monkeypatched with a canned CompletedProcess matching the CLI's real
`--output-format json` shape (live-verified against the actual binary while
building this), so these tests need no real `claude` install and cost
nothing.

`_screenshot_file` is exercised against a real temp PNG with
locator.ocr.run_ocr monkeypatched (same pattern as test_pii_redact.py) --
this decider's whole reason to exist is testing without a metered API key,
so it shouldn't need a real OCR engine either.
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

from PIL import Image

from agent.deciders import ClaudeCodeCLIDecider
from surfaces.base import Observation


def _fake_completed_process(stdout_obj: dict, returncode: int = 0, stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(stdout=json.dumps(stdout_obj), stderr=stderr, returncode=returncode)


def _obs(tmp_path, ocr_excerpt=None, dom_excerpt="<html></html>") -> Observation:
    shot_path = tmp_path / "shot.png"
    buf = io.BytesIO()
    Image.new("RGB", (50, 20), color="white").save(buf, format="PNG")
    shot_path.write_bytes(buf.getvalue())
    return Observation(
        url="https://example.com", title="t", screenshot_path=shot_path, dom_excerpt=dom_excerpt,
        timestamp=0.0, ocr_excerpt=ocr_excerpt,
    )


def test_parses_structured_output_into_a_decision(tmp_path, monkeypatch):
    monkeypatch.setattr("locator.ocr.run_ocr", lambda path: [])
    monkeypatch.setattr(
        "subprocess.run",
        lambda cmd, **kw: _fake_completed_process(
            {
                "session_id": "abc-123",
                "structured_output": {"tool_name": "click", "tool_input": {"css_selector": "#go"}, "reasoning": "stable id"},
            }
        ),
    )
    decider = ClaudeCodeCLIDecider()

    result = decider.step("goal", _obs(tmp_path), last_result=None)

    assert result.tool_name == "click"
    assert result.tool_input == {"css_selector": "#go"}
    assert result.all_tool_calls == [{"name": "click", "input": {"css_selector": "#go"}}]


def test_resumes_the_session_on_the_second_call(tmp_path, monkeypatch):
    monkeypatch.setattr("locator.ocr.run_ocr", lambda path: [])
    calls: list[list[str]] = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return _fake_completed_process(
            {"session_id": "abc-123", "structured_output": {"tool_name": "wait", "tool_input": {"ms": 500}}}
        )

    monkeypatch.setattr("subprocess.run", fake_run)
    decider = ClaudeCodeCLIDecider()

    decider.step("goal", _obs(tmp_path), last_result=None)
    decider.step("goal", _obs(tmp_path), last_result={"ok": True})

    assert "--resume" not in calls[0]  # first call: no prior session to resume
    assert "--resume" in calls[1]
    assert calls[1][calls[1].index("--resume") + 1] == "abc-123"


def test_falls_back_to_parsing_the_raw_result_field_when_structured_output_is_missing(tmp_path, monkeypatch):
    # Found live: the CLI's own structured-output extraction can come back
    # empty/wrong-shaped even when the model's text response is itself valid
    # JSON matching the requested schema.
    monkeypatch.setattr("locator.ocr.run_ocr", lambda path: [])
    monkeypatch.setattr(
        "subprocess.run",
        lambda cmd, **kw: _fake_completed_process(
            {"session_id": "abc-123", "result": json.dumps({"tool_name": "done", "tool_input": {"summary": "ok"}})}
        ),
    )
    decider = ClaudeCodeCLIDecider()

    result = decider.step("goal", _obs(tmp_path), last_result=None)

    assert result.tool_name == "done"
    assert result.tool_input == {"summary": "ok"}


def test_a_null_tool_name_means_no_action_this_turn(tmp_path, monkeypatch):
    monkeypatch.setattr("locator.ocr.run_ocr", lambda path: [])
    monkeypatch.setattr(
        "subprocess.run",
        lambda cmd, **kw: _fake_completed_process({"session_id": "abc-123", "structured_output": {"tool_name": None, "tool_input": {}}}),
    )
    decider = ClaudeCodeCLIDecider()

    result = decider.step("goal", _obs(tmp_path), last_result=None)

    assert result.tool_name is None
    assert result.all_tool_calls == []


def test_nonzero_exit_raises_with_stderr_context(tmp_path, monkeypatch):
    monkeypatch.setattr("locator.ocr.run_ocr", lambda path: [])
    monkeypatch.setattr(
        "subprocess.run",
        lambda cmd, **kw: SimpleNamespace(stdout="", stderr="not logged in", returncode=1),
    )
    decider = ClaudeCodeCLIDecider()

    try:
        decider.step("goal", _obs(tmp_path), last_result=None)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "not logged in" in str(e)


def test_cli_reads_the_surfaces_saved_screenshot_which_is_already_redacted(tmp_path, monkeypatch):
    """Inverted from the earlier version of this test: the on-disk file used to
    be raw, so the CLI had to be pointed at a redacted temp copy instead.
    Now surfaces/web.py redacts before it writes anything, so the saved file
    is the redacted one and is exactly what the CLI should read -- a temp copy
    would be a second OCR pass for nothing."""
    captured_prompt = {}

    def fake_run(cmd, **kw):
        captured_prompt["prompt"] = cmd[cmd.index("-p") + 1]
        return _fake_completed_process({"session_id": "s1", "structured_output": {"tool_name": None, "tool_input": {}}})

    monkeypatch.setattr("subprocess.run", fake_run)
    obs = _obs(tmp_path)
    ClaudeCodeCLIDecider().step("goal", obs, last_result=None)

    assert f"Read the screenshot at {obs.screenshot_path}" in captured_prompt["prompt"]


def test_cli_gets_no_screenshot_path_when_the_surface_withheld_it(tmp_path, monkeypatch):
    # screenshot_redacted=False means the surface could not redact and wrote
    # nothing -- the CLI must not be told to Read a file that would be raw.
    captured_prompt = {}

    def fake_run(cmd, **kw):
        captured_prompt["prompt"] = cmd[cmd.index("-p") + 1]
        return _fake_completed_process({"session_id": "s1", "structured_output": {"tool_name": None, "tool_input": {}}})

    monkeypatch.setattr("subprocess.run", fake_run)
    obs = _obs(tmp_path)
    obs.screenshot_redacted = False
    ClaudeCodeCLIDecider().step("goal", obs, last_result=None)

    assert "Read the screenshot at" not in captured_prompt["prompt"]
    assert "No screenshot is available this turn" in captured_prompt["prompt"]


def test_resumed_session_is_dropped_after_the_window(monkeypatch, tmp_path):
    """Found live: --resume carries the whole conversation forward, every prior
    prompt and screenshot included, so per-turn latency climbed 19s -> 114s over
    six turns and the seventh blew the subprocess timeout. Bounding the session
    is the CLI-shaped equivalent of the API deciders' rolling history window."""
    import json
    import subprocess

    from agent.deciders import ClaudeCodeCLIDecider
    from surfaces.base import Observation

    seen_cmds: list[list[str]] = []

    class _Proc:
        returncode = 0
        stderr = ""
        stdout = json.dumps({"session_id": "sess-1", "structured_output": {"tool_name": "wait", "tool_input": {"ms": 1}}})

    def fake_run(cmd, **kwargs):
        seen_cmds.append(cmd)
        return _Proc()

    monkeypatch.setattr(subprocess, "run", fake_run)
    # no OCR engine in tests: make redaction fail closed so no image is involved
    monkeypatch.setattr(
        "guardrails.pii_redact.redact_screenshot_bytes", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no ocr"))
    )

    decider = ClaudeCodeCLIDecider(resume_window=2)  # two turns per session: one fresh, one resumed
    shot = tmp_path / "s.png"; shot.write_bytes(b"")
    obs = Observation(url="https://e.com", title="", screenshot_path=shot, dom_excerpt="<html/>", timestamp=0.0)

    for _ in range(5):
        decider.step("g", obs)

    resumed = ["--resume" in c for c in seen_cmds]
    # fresh, resumed | fresh, resumed | fresh -- no session ever carries more than
    # `resume_window` turns, so context (and latency) is bounded.
    assert resumed == [False, True, False, True, False], resumed


def test_the_cli_subprocess_never_inherits_stdin(tmp_path, monkeypatch):
    # Found live: a piped 'resume' meant for the discovery gate was consumed by
    # the claude CLI child during the previous turn, and the gate saw EOF. With
    # a terminal it would be the operator's keystrokes at risk. `-p` mode has
    # no use for stdin at all.
    import subprocess
    seen = {}

    def fake_run(cmd, **kw):
        seen["stdin"] = kw.get("stdin")
        return _fake_completed_process({"session_id": "s", "structured_output": {"tool_name": None, "tool_input": {}}})

    monkeypatch.setattr("subprocess.run", fake_run)
    ClaudeCodeCLIDecider().step("goal", _obs(tmp_path), last_result=None)
    assert seen["stdin"] is subprocess.DEVNULL
