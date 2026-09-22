"""Regression test for a real bug found live: AnthropicDecider only tracked
a single pending tool_use id between turns, but a Claude response can contain
more than one tool_use block in a single turn. The untracked extra block still
went into `history` (the full, unfiltered resp.content) with no tool_result
ever supplied for it -- a few turns later, once that assistant message aged
into the rolling 4-message window at just the wrong position, the real API
rejected the request outright: "tool_use ids were found without tool_result
blocks immediately after."

No network/API key needed: `step()` only touches `self.client` and
`self.history`/`self._pending_tool_use_ids`, so a fake client stands in for
the real Anthropic SDK, and instances are built without __init__ (which is
the only place that actually imports anthropic / requires an API key).

`_image_b64` (reads the surface's already-redacted screenshot) is
autouse-monkeypatched for every test here: `_obs()`'s `screenshot_path` is
this test file itself, not a real image, so the real pipeline would run a
real PaddleOCR pass and fail closed on every single call -- correct
behavior, but it couples these otherwise pure, no-engine-needed tests to an
optional heavy dependency (`uv sync --extra ocr`) they were never meant to
need. Redaction itself has its own dedicated tests (test_pii_redact.py).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.deciders import AnthropicDecider
from surfaces.base import Observation


@pytest.fixture(autouse=True)
def _fake_redaction(monkeypatch):
    monkeypatch.setattr("agent.deciders._image_b64", lambda obs: ("fake_b64_image", 0))


def _block(type_, **kw):
    return SimpleNamespace(type=type_, **kw)


class FakeAnthropicClient:
    """Returns one canned response per call, in order."""

    def __init__(self, responses: list[list[SimpleNamespace]]):
        self._responses = list(responses)
        self.calls: list[list[dict]] = []  # the `messages` list sent on each call

        class _Messages:
            def create(inner_self, model, max_tokens, tools, messages):
                self.calls.append(messages)
                content = self._responses.pop(0)
                return SimpleNamespace(content=content)

        self.messages = _Messages()


def make_decider(client: FakeAnthropicClient) -> AnthropicDecider:
    decider = AnthropicDecider.__new__(AnthropicDecider)  # bypass __init__ -- no API key/network needed
    decider.client = client
    decider.model = "claude-test"
    decider.history = []
    decider._pending_tool_use_ids = []
    decider._tools_override = None
    return decider


def _obs(ocr_excerpt: str | None = None, dom_excerpt: str = "<html></html>") -> Observation:
    return Observation(
        url="https://example.com", title="t", screenshot_path=__file__, dom_excerpt=dom_excerpt,
        timestamp=0.0, ocr_excerpt=ocr_excerpt,
    )


def _tool_result_ids_in(message: dict) -> list[str]:
    return [b["tool_use_id"] for b in message["content"] if b["type"] == "tool_result"]


def test_every_tool_use_id_gets_a_tool_result_even_when_a_turn_emits_two():
    # Turn 1's response has TWO tool_use blocks -- the real shape that broke
    # this before the fix.
    client = FakeAnthropicClient(
        responses=[
            [_block("tool_use", id="tool_a", name="click", input={}), _block("tool_use", id="tool_b", name="wait", input={})],
            [_block("tool_use", id="tool_c", name="click", input={})],
        ]
    )
    decider = make_decider(client)

    result1 = decider.step("goal", _obs(), last_result=None)
    assert result1.tool_name == "click"  # only the first tool_use is acted on -- unchanged behavior

    # Turn 2: the next user message sent to the API must carry a tool_result
    # for BOTH tool_a and tool_b -- this is the actual bug. Before the fix,
    # only tool_a (the single tracked id) got one; tool_b's tool_use block
    # was still sitting in history with nothing resolving it.
    decider.step("goal", _obs(), last_result={"ok": True})
    sent_messages = client.calls[1]  # what was sent on the second API call
    last_user_message = [m for m in sent_messages if m["role"] == "user"][-1]
    resolved_ids = _tool_result_ids_in(last_user_message)
    assert resolved_ids == ["tool_a", "tool_b"], resolved_ids

    # The real action's outcome goes to the first id; the second (never
    # actually executed) gets an honest synthetic result, not a fabricated
    # "it worked" copy of the first.
    contents = {b["tool_use_id"]: json.loads(b["content"]) for b in last_user_message["content"] if b["type"] == "tool_result"}
    assert contents["tool_a"] == {"ok": True}
    assert "not executed" in contents["tool_b"]["note"]


def test_single_tool_use_per_turn_still_works_as_before():
    client = FakeAnthropicClient(
        responses=[
            [_block("tool_use", id="tool_1", name="click", input={})],
            [_block("tool_use", id="tool_2", name="wait", input={})],
        ]
    )
    decider = make_decider(client)

    decider.step("goal", _obs(), last_result=None)
    decider.step("goal", _obs(), last_result={"ok": True})

    sent_messages = client.calls[1]
    last_user_message = [m for m in sent_messages if m["role"] == "user"][-1]
    assert _tool_result_ids_in(last_user_message) == ["tool_1"]


def test_no_tool_use_in_a_turn_leaves_nothing_pending():
    client = FakeAnthropicClient(responses=[[_block("text", text="thinking out loud, no tool call")]])
    decider = make_decider(client)

    result = decider.step("goal", _obs(), last_result=None)

    assert result.tool_name is None
    assert decider._pending_tool_use_ids == []


def test_ocr_excerpt_is_appended_to_the_prompt_alongside_dom():
    # surfaces/web.py's observe() now runs OCR on every screenshot and hands
    # the result to the decider through Observation.ocr_excerpt -- alongside
    # dom_excerpt, not instead of it. Real motivation: text rendered only to
    # pixels (canvas, an image) has no DOM node for the DOM excerpt to show
    # at all.
    client = FakeAnthropicClient(responses=[[_block("text", text="ok")]])
    decider = make_decider(client)

    result = decider.step("goal", _obs(ocr_excerpt="Add to cart\n$19.99"), last_result=None)

    assert "Add to cart" in result.prompt
    assert "DOM excerpt" in result.prompt  # OCR is additive, not a replacement


def test_all_tool_calls_in_a_turn_are_captured_not_just_the_acted_on_one():
    # Full observability (the run log's job): a turn with two tool_use blocks
    # only ever acts on the first (agent/loop.py's single-action-per-step
    # design), but the model genuinely proposed both -- the log shouldn't
    # silently lose the second one just because the loop didn't use it.
    client = FakeAnthropicClient(
        responses=[[_block("tool_use", id="tool_a", name="click", input={"css_selector": "#a"}), _block("tool_use", id="tool_b", name="wait", input={"ms": 500})]]
    )
    decider = make_decider(client)

    result = decider.step("goal", _obs(), last_result=None)

    assert result.tool_name == "click"  # unchanged: only the first is acted on
    assert result.all_tool_calls == [
        {"name": "click", "input": {"css_selector": "#a"}},
        {"name": "wait", "input": {"ms": 500}},
    ]


def test_dom_excerpt_is_pii_redacted_in_the_prompt_too():
    # Real bug found live: the OCR excerpt and screenshot image were redacted,
    # but the raw DOM excerpt -- plain rendered HTML, included in every
    # prompt regardless of vision support -- still carried an SSN verbatim,
    # since only the OCR/image side of guardrails/pii_redact.py was wired in
    # at first. Caught by actually inspecting what a live decider call would
    # send, not by code review alone.
    client = FakeAnthropicClient(responses=[[_block("text", text="ok")]])
    decider = make_decider(client)

    result = decider.step("goal", _obs(dom_excerpt="<p>SSN: 123-45-6789</p>"), last_result=None)

    assert "123-45-6789" not in result.prompt
    assert "[REDACTED:ssn]" in result.prompt


def test_redacted_screenshot_gets_an_explanatory_note_the_model_can_act_on(monkeypatch):
    # Real bug found live: a vision-capable model shown a screenshot with
    # black-painted regions and no explanation concluded those were the
    # *site's own* privacy masking and spent most of a run trying to reach an
    # "unmasked" view instead of respecting the redaction -- treating this
    # system's own safety control as a UI obstacle to route around.
    monkeypatch.setattr("agent.deciders._image_b64", lambda obs: ("fake_b64_image", 3))
    client = FakeAnthropicClient(responses=[[_block("text", text="ok")]])
    decider = make_decider(client)

    result = decider.step("goal", _obs(), last_result=None)

    assert "3 region(s)" in result.prompt
    assert "THIS SYSTEM's own pre-send redaction" in result.prompt
    assert "not a feature of the site itself" in result.prompt


def test_no_ocr_excerpt_adds_nothing_to_the_prompt():
    # observe() can genuinely produce no OCR text (PaddleOCR not installed,
    # engine crashed, or nothing above min_confidence was found) -- the
    # prompt shouldn't claim an OCR section exists when there's nothing in it.
    client = FakeAnthropicClient(responses=[[_block("text", text="ok")]])
    decider = make_decider(client)

    result = decider.step("goal", _obs(ocr_excerpt=None), last_result=None)

    assert "OCR" not in result.prompt
