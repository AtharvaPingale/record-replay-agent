"""Pluggable decision backends for the agent loop (agent/loop.py).

Two real implementations:
- AnthropicDecider -- vision-capable, hosted API, needs ANTHROPIC_API_KEY.
- OllamaDecider -- self-hosted, DOM-text-grounded. Talks to any Ollama
  daemon: localhost by default, or a remote one via `host=` / the OLLAMA_HOST
  env var (see `ollama_host_default`). Verified against a real local
  Ollama instance in this sandbox running `qwen2.5:14b-instruct` (reports
  native tool-calling support) -- not vision-capable. That's a genuine,
  disclosed tradeoff, not a hidden one: the screenshot is still captured and
  logged for human evidence/audit (`emit_llm_turn`'s `screenshot` field), the
  local model just never sees pixels, only the DOM excerpt. This is a
  legitimate way to run a DOM-grounded agent -- the tool schema already lets
  a model target elements by css_selector/text rather than coordinates, and
  the locator priority this codebase uses (DOM selector -> text -> coords, coords
  last resort) means a text-only model isn't missing the primary anchors
  anyway. If a vision-capable local model is available, pass its name and
  OllamaDecider will include the screenshot too -- see
  `_VISION_MODEL_PATTERNS` below for how that's detected.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from surfaces.base import Observation

# Model-name substrings that indicate an Ollama model accepts an `images`
# field on a chat message -- pattern-matched so a specific version tag
# doesn't need to be pinned here.
_VISION_MODEL_PATTERNS = ("llava", "vl", "vision", "moondream", "bakllava")


@dataclass
class DecisionResult:
    prompt: str
    response: str
    tool_name: str | None
    tool_input: dict[str, Any] = field(default_factory=dict)
    # Every tool call the model proposed this turn, not just the one
    # (tool_name/tool_input) agent/loop.py actually acts on -- a model can
    # genuinely propose more than one in a single turn (this is exactly what
    # AnthropicDecider's tool_use/tool_result pairing below has to account for), and
    # full observability (the brief's Section 3.5) means the run log should
    # show everything the model actually said, not just the part the loop
    # used. [{"name": ..., "input": ...}, ...], same order as proposed.
    all_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # How many OCR-detected regions guardrails/pii_redact.py masked in the
    # screenshot sent this turn (0 for a non-vision turn, or when nothing
    # matched) -- logged by agent/loop.py so a human auditing the run can see
    # redaction actually happened, not just trust that it silently should
    # have.
    pii_redacted_count: int = 0


class Decider(Protocol):
    supports_vision: bool

    def step(
        self, goal: str, obs: Observation, last_result: dict[str, Any] | None = None,
        exempt: frozenset[str] | None = None,
    ) -> DecisionResult:
        """`last_result` is the outcome of whatever action this decider asked
        for last step (None on the very first call) -- both implementations feed
        it back into the model's own context, not just the next screenshot/DOM,
        so the model knows what actually happened to the thing it just tried.

        `exempt`: literal values (e.g. a target account ID the caller's own
        goal already named) to leave unmasked by guardrails/pii_redact.py --
        see that module's own docstring for why blind redaction breaks a
        goal shaped like "look up a user's balance by account ID"."""
        ...


def _redacted_dom_excerpt(obs: Observation, limit: int, exempt: frozenset[str]) -> str:
    """The DOM excerpt slice every decider's prompt includes, with
    banking/SSN/currency/email-shaped substrings masked (guardrails/pii_redact.py)
    -- found live: this is plain rendered HTML, so any PII visible on the
    page appears here as literal text regardless of what the screenshot/OCR
    side redacts. Redaction runs on exactly the slice that's actually sent,
    not the full excerpt, so nothing gets masked in a part of the DOM that
    was going to be truncated away anyway."""
    from guardrails.pii_redact import redact_pii_text

    return redact_pii_text(obs.dom_excerpt[:limit], exempt=exempt)


def _ocr_prompt_section(obs: Observation, exempt: frozenset[str]) -> str:
    """Shared by every decider's prompt building: OCR text (locator/ocr.py,
    run against the same screenshot) alongside the DOM excerpt, not instead
    of it -- text actually rendered to pixels that the DOM excerpt can't show
    at all (canvas, an image, an icon font with no accessible name). Empty
    string, not an empty section header, when observe() didn't run OCR or it
    found nothing -- a decider prompt shouldn't imply a signal exists when it
    doesn't."""
    if not obs.ocr_excerpt:
        return ""
    from guardrails.pii_redact import redact_pii_text

    # This text reaches every decider's prompt, vision or not -- redacted the
    # same way the image itself is (guardrails/pii_redact.py), since a
    # non-vision decider (OllamaDecider with a text-only model) never sees
    # the screenshot at all except through this text.
    redacted_excerpt = redact_pii_text(obs.ocr_excerpt, exempt=exempt)
    return (
        f"\n\nOCR text detected directly in the screenshot's pixels (from "
        f"locator/ocr.py, not the DOM -- includes anything rendered visually "
        f"that has no DOM/accessible-text equivalent, e.g. canvas or image "
        f"content; may also just duplicate ordinary DOM text; banking/SSN/"
        f"email/currency-shaped substrings are masked before this ever "
        f"reaches a prompt):\n{redacted_excerpt[:2000]}"
    )


def _image_redaction_note(pii_redacted_count: int) -> str:
    """Found live: without this, a vision-capable model looking at a
    screenshot with solid-black regions in it reasonably concluded those were
    the *site's own* privacy masking ("the table view has masked amounts for
    privacy overlays") and spent most of a run trying to reach an "unmasked"
    view instead -- treating this system's own pre-send redaction
    (guardrails/pii_redact.py) as a UI obstacle to route around, not a
    boundary to respect. Harmless in this architecture specifically (every
    screenshot is redacted the same way, on every page, so there's no
    actual "unmasked" view to find), but the *intent* is exactly what a
    privacy control shouldn't have to rely on being lucky about. Only
    appended when this turn's screenshot actually has redacted regions in
    it -- an empty string otherwise, so a clean screenshot's prompt doesn't
    imply a signal that isn't there."""
    if pii_redacted_count == 0:
        return ""
    return (
        f"\n\nNOTE: {pii_redacted_count} region(s) in the screenshot above are "
        f"painted solid black. That is THIS SYSTEM's own pre-send redaction, "
        f"applied before the image ever reached you -- not a feature of the "
        f"site itself, and not something to find a way around. Every "
        f"screenshot sent to you goes through the same redaction, so a "
        f"different page/view/details panel will look the same -- do not go "
        f"looking for an 'unmasked' version. If a value you need to report "
        f"lives in a redacted region, declare a dom_selector-based output "
        f"pointing at WHERE it is (declare_output with target_kind "
        f"'dom_selector'), not the literal text you can currently see -- "
        f"that resolves to the real value at replay time regardless of what's "
        f"masked here during recording."
    )


def _image_b64(obs: Observation) -> tuple[str | None, int]:
    """The screenshot to send, as base64, plus how many PII regions were
    painted out of it. The file at obs.screenshot_path is *already* redacted --
    surfaces/web.py redacts before it writes, and the on-disk evidence is the
    same image the model sees -- so this only reads it. Fail-closed is
    inherited: if the surface could not redact it wrote nothing
    (obs.screenshot_redacted is False), and this returns (None, 0) so the
    caller drops the image for the turn. There is no path by which an
    unredacted screenshot can reach a model or the filesystem."""
    path = Path(obs.screenshot_path)
    if not obs.screenshot_redacted or not path.exists():
        print("[pii_redact] no redacted screenshot for this turn -- deciding from DOM/OCR text alone")
        return None, 0
    return base64.b64encode(path.read_bytes()).decode(), obs.pii_redacted_count


DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def ollama_host_default() -> str:
    """Where OllamaDecider talks to when no host is given explicitly: the
    OLLAMA_HOST env var if set (the same variable the `ollama` CLI itself
    honours, so a machine already pointed at a remote GPU box needs no extra
    config here), else localhost. Ollama accepts a bare host:port in that
    variable, so add the scheme if it's missing."""
    import os

    host = os.environ.get("OLLAMA_HOST", "").strip() or DEFAULT_OLLAMA_HOST
    if "://" not in host:
        host = "http://" + host
    return host


def default_decider(
    model: str | None = None, tools: list[dict] | None = None, ollama_host: str | None = None
) -> "Decider":
    """The one place this choice gets made, so agent/loop.py and its callers
    (e.g. scripts/record_capability.py) don't each reimplement it: use Claude if
    ANTHROPIC_API_KEY is set (vision-capable, zero local setup), otherwise
    fall back to a local Ollama model (no key needed, just `ollama serve`
    running) -- local-first, but Claude for free the moment a key exists.
    `model`, if given, is passed to whichever backend gets picked; leave it
    None to use that backend's own default. `ollama_host` only matters if
    Ollama is what gets picked; None means OLLAMA_HOST / localhost (see
    ollama_host_default). `tools`, if given, overrides the
    standard action-only schema -- agent/loop.py's run_discovery_loop passes
    TOOLS + DISCOVERY_TOOLS so a recording session's decider can also declare
    a checkpoint/outputs, without touching the ordinary live-loop schema."""
    import os

    kwargs = {"tools": tools} if tools is not None else {}
    if os.environ.get("ANTHROPIC_API_KEY"):
        if model:
            kwargs["model"] = model
        return AnthropicDecider(**kwargs)
    return OllamaDecider(model=model or "qwen2.5:14b-instruct", host=ollama_host, **kwargs)


class AnthropicDecider:
    supports_vision = True

    def __init__(self, model: str = "claude-sonnet-5", api_key: str | None = None, tools: list[dict] | None = None):
        import os

        import anthropic

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        # None (the ordinary live-loop case) resolves to the standard action
        # tools at call time -- agent/loop.py's run_discovery_loop passes
        # TOOLS + DISCOVERY_TOOLS instead, so the declare_* tools only ever
        # exist for a recording session, never an ordinary live run.
        self._tools_override = tools
        self.history: list[dict] = []
        # A single pending id was wrong (found live, not hypothetical -- see
        # below): Claude can and did return more than one
        # tool_use block in a single response, and only the first one's id was
        # ever tracked here. The extra tool_use(s) still went into `history`
        # (the full `resp.content`, unfiltered), but nothing ever supplied
        # their required tool_result -- so a few turns later, once that
        # assistant message aged into the rolling window at just the wrong
        # position, the API rejected the whole request: "tool_use ids were
        # found without tool_result blocks immediately after." A list, not a
        # single id, is what actually matches Anthropic's real contract (every
        # tool_use in a turn needs a paired tool_result in the very next
        # message, not just the first one this loop happens to act on).
        self._pending_tool_use_ids: list[str] = []

    def step(
        self, goal: str, obs: Observation, last_result: dict[str, Any] | None = None,
        exempt: frozenset[str] | None = None,
    ) -> DecisionResult:
        from agent.tools import TOOLS

        exempt = exempt or frozenset()
        tools = self._tools_override if self._tools_override is not None else TOOLS
        img_b64, pii_redacted_count = _image_b64(obs)
        prompt_text = f"Goal: {goal}\nCurrent URL: {obs.url}\nDOM excerpt:\n{_redacted_dom_excerpt(obs, 4000, exempt)}"
        prompt_text += _ocr_prompt_section(obs, exempt)
        prompt_text += _image_redaction_note(pii_redacted_count)

        content: list[dict[str, Any]] = []
        # Close out EVERY pending tool_use from the previous turn, not just the
        # one this loop actually acted on -- Anthropic's multi-turn tool
        # protocol requires a tool_result for each, immediately after, or the
        # next API call is rejected outright (this is exactly what's handled
        # here). Only the first pending id corresponds to `last_result` (the
        # one action agent/loop.py's single-action-per-step design actually
        # executed); any others are real tool_use blocks the model asked for
        # but this loop never acted on, and get an honest synthetic result
        # instead of a fabricated "it worked."
        for i, tool_use_id in enumerate(self._pending_tool_use_ids):
            result = last_result if i == 0 else {"note": "not executed -- only the first tool call per turn is acted on"}
            content.append({"type": "tool_result", "tool_use_id": tool_use_id, "content": json.dumps(result)})
        # No image block at all when redaction failed (img_b64 is None) --
        # never fall back to the unredacted screenshot. The model still gets
        # the (also-redacted) OCR text and DOM excerpt in prompt_text below.
        if img_b64 is not None:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}})
        content.append({"type": "text", "text": prompt_text})
        self.history.append({"role": "user", "content": content})
        self.history = self.history[-4:]  # rolling context, matches agent/loop.py's original window

        resp = self.client.messages.create(model=self.model, max_tokens=1024, tools=tools, messages=self.history)
        self.history.append({"role": "assistant", "content": resp.content})

        response_text = "\n".join(b.text for b in resp.content if b.type == "text")
        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        tool_use = tool_uses[0] if tool_uses else None
        self._pending_tool_use_ids = [t.id for t in tool_uses]
        return DecisionResult(
            prompt=prompt_text,
            response=response_text or (f"[tool_use: {tool_use.name}]" if tool_use else ""),
            tool_name=tool_use.name if tool_use else None,
            tool_input=tool_use.input if tool_use else {},
            all_tool_calls=[{"name": t.name, "input": t.input} for t in tool_uses],
            pii_redacted_count=pii_redacted_count,
        )


_TOOL_NAMES = (
    "click", "type", "select", "wait", "assert_text", "go_to", "done", "ask_user",
    "declare_checkpoint", "declare_output", "declare_business_outcome", "declare_recoverable",
)


def _extract_inline_tool_call(text: str) -> tuple[str | None, dict[str, Any]]:
    """Fallback for when a local model reasons its way to the right call but
    emits it as JSON prose (e.g. inside a ```json ... ``` block) instead of
    Ollama's structured tool_calls field -- observed live with
    qwen2.5:14b-instruct mid-conversation, not a hypothetical. Best-effort only:
    scans for balanced-brace `{...}` objects and tries each as JSON, looking for
    a recognized tool name. Returns (None, {}) if nothing parses, same as a
    real no-tool-call turn."""
    search_from = 0
    while True:
        start = text.find("{", search_from)
        if start == -1:
            return None, {}
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : i + 1])
                    except (json.JSONDecodeError, ValueError):
                        break
                    name = parsed.get("name")
                    args = parsed.get("arguments") or parsed.get("input")
                    if name in _TOOL_NAMES and isinstance(args, dict):
                        return name, args
                    break
        search_from = start + 1


def _tools_as_openai_schema(tools: list[dict] | None = None) -> list[dict]:
    from agent.tools import TOOLS

    return [
        {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
        for t in (tools if tools is not None else TOOLS)
    ]


class OllamaDecider:
    def __init__(
        self,
        model: str = "qwen2.5:14b-instruct",
        host: str | None = None,
        skip_capability_check: bool = False,
        tools: list[dict] | None = None,
        num_ctx: int = 8192,
    ):
        self.model = model
        # None -> OLLAMA_HOST if set, else localhost; an explicit host wins.
        self.host = (host or ollama_host_default()).rstrip("/")
        self.history: list[dict] = []
        self._tools_override = tools  # see AnthropicDecider's field of the same name
        self.supports_vision = any(p in model.lower() for p in _VISION_MODEL_PATTERNS)
        # Ollama defaults to a 4096-token context regardless of what a model can
        # actually support, unless told otherwise per-request. Found live: one
        # image alone plus a DOM excerpt and a couple of turns of history
        # blew straight through that default (a real "N tokens exceeds 4096"
        # rejection, not a hypothetical). Vision turns need much more headroom
        # than text-only ones; 8192 is a working default for a single-image,
        # few-turn conversation, not a guarantee for every model.
        self.num_ctx = num_ctx

        if not skip_capability_check:
            # Found live: a vision-capable model tested during development
            # reported capabilities ["completion", "vision"] -- no "tools".
            # Ollama silently accepts a `tools` payload for such a model and
            # just never returns a tool_call, which without this check looks
            # identical to "the model chose not to act" every single turn
            # until max_steps. Fail loudly and immediately instead.
            caps = self._model_capabilities()
            if caps is not None and "tools" not in caps:
                raise RuntimeError(
                    f"model '{model}' does not report Ollama tool-calling support "
                    f"(capabilities: {caps}) -- this decider requires native tool "
                    f"calls to produce structured actions. Use a tool-calling model "
                    f"(e.g. qwen2.5:14b-instruct) instead."
                )

    def _model_capabilities(self) -> list[str] | None:
        try:
            req = urllib.request.Request(
                f"{self.host}/api/show",
                data=json.dumps({"model": self.model}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read()).get("capabilities")
        except urllib.error.URLError:
            return None  # can't reach Ollama to check -- let the real call surface the error instead

    def step(
        self, goal: str, obs: Observation, last_result: dict[str, Any] | None = None,
        exempt: frozenset[str] | None = None,
    ) -> DecisionResult:
        exempt = exempt or frozenset()
        if last_result is not None:
            # Ollama's chat API accepts a "tool" role message reporting a tool
            # call's outcome, same shape as OpenAI's. Found live: without this,
            # the model has no way to know whether its last action worked, and
            # (combined with the near-identical prompt each turn) it degenerates
            # into repeating the same call rather than progressing.
            self.history.append({"role": "tool", "content": json.dumps(last_result)})

        pii_redacted_count = 0
        if self.supports_vision:
            prompt_text = f"Goal: {goal}\nCurrent URL: {obs.url}\nDOM excerpt:\n{_redacted_dom_excerpt(obs, 4000, exempt)}"
            prompt_text += _ocr_prompt_section(obs, exempt)
            img_b64, pii_redacted_count = _image_b64(obs)
            prompt_text += _image_redaction_note(pii_redacted_count)
            # No "images" key at all when redaction failed -- never fall back
            # to the raw, unredacted screenshot (see _image_b64).
            user_msg = {"role": "user", "content": prompt_text, "images": [img_b64]} if img_b64 is not None else {"role": "user", "content": prompt_text}
        else:
            prompt_text = (
                f"Goal: {goal}\nCurrent URL: {obs.url}\n"
                f"IMPORTANT: only use a css_selector value that literally appears "
                f"verbatim in the DOM excerpt below (an id= or class= you can see "
                f"in this exact text). Do not guess a plausible-looking selector "
                f"from general e-commerce conventions -- if you're not certain, "
                f"use the visible-text locator field instead (e.g. 'text': 'Add "
                f"to cart' for click, or 'find_field_by_text' for type), since "
                f"that only requires the label to be right, not the markup.\n"
                f"DOM excerpt (no screenshot is shown to you -- decide from this "
                f"structure and text alone):\n{_redacted_dom_excerpt(obs, 6000, exempt)}"
            )
            prompt_text += _ocr_prompt_section(obs, exempt)
            user_msg = {"role": "user", "content": prompt_text}

        self.history.append(user_msg)
        self.history = self.history[-6:]

        if self.supports_vision:
            # Strip images from every *older* turn before sending -- a stale
            # screenshot from 3 turns ago costs the same context budget as the
            # current one but the model can't act on outdated pixels anyway.
            # Multiplying image tokens across a 6-message window is exactly what
            # blew through Ollama's default context window live; this alone
            # roughly halves the steady-state cost of a multi-turn vision run.
            messages = [
                {k: v for k, v in m.items() if k != "images"} if m is not user_msg else m for m in self.history
            ]
        else:
            messages = self.history

        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "tools": _tools_as_openai_schema(self._tools_override),
                "stream": False,
                "options": {"num_ctx": self.num_ctx, "num_predict": 1024},
                # A hybrid-reasoning model tested during development could spend
                # its entire output budget on <think> content before ever
                # reaching a tool call. Tried raising num_predict from 2048 to
                # 4096 live, expecting more headroom to finish thinking:
                # thinking length scaled with the budget instead (8.3k -> 17.2k
                # chars) and it still never converged to an answer, on the same
                # real prompt, twice. That's a real capability ceiling for a
                # small model on a moderately complex page, not a token-budget
                # problem -- so this stays modest rather than burning more
                # latency for no gain. `think: False` is passed regardless
                # since it costs nothing and did shorten thinking length in
                # lighter tests.
                "think": False,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            # HTTPError.read() has the actual error body (Ollama returns a JSON
            # {"error": "..."} on a 4xx) -- swallowing it as a bare "HTTP Error
            # 400: Bad Request" (what `{e}` alone gives you) made an earlier
            # real failure here much harder to diagnose than it needed to be.
            body = e.read().decode(errors="replace")
            raise RuntimeError(f"Ollama at {self.host} rejected the request ({e.code}): {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"could not reach local Ollama at {self.host}: {e}") from e

        message = data["message"]
        # Store the *full* response message, tool_calls included -- not just its
        # (often empty, when the model only calls a tool) text content. Found
        # live: without this, the model has no memory of which action it just
        # took, only the near-identical "Goal/DOM excerpt" prompt repeating each
        # turn, and it degenerates into re-issuing the same tool call forever
        # (caught by the loop's dead_end detection, but the fix is not to lose
        # the model's own turn from its context in the first place).
        self.history.append(message)

        tool_calls = message.get("tool_calls") or []
        content = message.get("content") or ""
        thinking = message.get("thinking") or ""
        all_tool_calls = [{"name": tc["function"]["name"], "input": tc["function"]["arguments"]} for tc in tool_calls]
        if tool_calls:
            tool_name = tool_calls[0]["function"]["name"]
            tool_input = tool_calls[0]["function"]["arguments"]
        else:
            # fall back to parsing an inline JSON tool call out of plain text
            # before concluding the model didn't act this turn -- see
            # _extract_inline_tool_call's docstring for why this is needed.
            tool_name, tool_input = _extract_inline_tool_call(content)
            if tool_name:
                all_tool_calls = [{"name": tool_name, "input": tool_input}]

        # Found live: a turn can produce thousands of characters of `thinking`
        # and nothing else -- logging only `content` made the evidence trail
        # show an empty response for turns that clearly weren't empty. Full
        # `thinking` is kept here, not truncated -- this is the persisted run
        # log, the one place the model's raw output is actually recorded for
        # audit; truncating it here (unlike the in-memory `self.history`,
        # which never leaves this process) would be a real loss, not a
        # convenience.
        if content:
            response_text = content
        elif tool_name:
            response_text = f"[tool_call: {tool_name}]"
        elif thinking:
            response_text = f"[thinking only, no content/tool_call -- {len(thinking)} chars]: {thinking}"
        else:
            response_text = ""

        return DecisionResult(
            prompt=prompt_text,
            response=response_text,
            tool_name=tool_name,
            tool_input=tool_input,
            all_tool_calls=all_tool_calls,
            pii_redacted_count=pii_redacted_count,
        )


class ClaudeCodeCLIDecider:
    """Drives decisions by shelling out to the `claude` CLI (Claude Code)
    instead of calling the Anthropic Messages API directly -- for testing
    without a metered `ANTHROPIC_API_KEY`, using whatever auth `claude`
    itself already has (OAuth/subscription login, not necessarily an API
    key). Not auto-selected by `default_decider()` -- this is an explicit
    choice (`--backend claude-cli`), not a fallback, since its actual cost
    profile is real and non-trivial (see below), not obviously cheaper than
    the API path it's standing in for.

    Real, measured tradeoffs, not guessed:
    - A fresh `claude -p` call reloads Claude Code's own full system
      prompt/tool definitions every time -- ~$0.19-0.40 for a single
      trivial turn, live-measured, before the underlying model call's own
      cost. Prohibitively expensive per agent-loop turn on its own.
    - `--resume <session_id>` after the first turn hits the prompt cache and
      drops that to ~$0.01/turn (also live-measured, an ~18x reduction) --
      this decider always resumes the same CLI session after its first
      turn, for exactly that reason. Changing the schema/tool set/model
      mid-session breaks that cache reuse, so this decider keeps all three
      fixed for the life of one instance.
    - The model does not reliably map free-form reasoning onto exact field
      semantics from a loose schema alone -- an early test asked for
      `{"tool_name": ..., "tool_input": ...}` with no `enum` constraint and
      got back `tool_name: null` with the model's own invented
      `tool_input` shape instead. Fixed here with a strict `enum` on
      `tool_name` and explicit prompt wording naming the exact convention.

    Vision works differently here than the other two deciders: rather than
    base64-encoding the screenshot into an API payload, this decider tells the
    CLI (`--allowedTools Read`, nothing else -- no Bash, no Edit, no wider file
    access) to read the surface's saved screenshot via its own Read tool. That
    file is already redacted -- surfaces/web.py redacts before it writes
    anything -- so there is no unredacted file on disk for the CLI to be
    pointed at, by construction rather than by care.
    """

    supports_vision = True

    def __init__(
        self, model: str | None = None, tools: list[dict] | None = None,
        claude_bin: str = "claude", allowed_tools: str = "Read", timeout_s: float = 300.0,
        resume_window: int = 4,
    ):
        self.model = model
        self._tools_override = tools
        self.claude_bin = claude_bin
        self.allowed_tools = allowed_tools
        self.timeout_s = timeout_s
        self._session_id: str | None = None
        # How many turns to carry on one resumed CLI session before starting a
        # fresh one. Two real, opposing costs, both measured on live recordings:
        #
        #   - A *resumed* turn is fast but the session grows: `--resume` carries
        #     every prior prompt and every screenshot the Read tool ingested. On a
        #     run with no window, per-turn latency went 19s -> 24s -> 22s -> 49s
        #     -> 114s over six turns (the prompt itself flat at ~13k chars), and
        #     turn 7 blew the then-120s subprocess timeout and killed the run.
        #   - A *fresh* turn resets that but is itself slow: 169s on the same
        #     target, against 17-27s for the resumed turns around it -- Claude
        #     Code reloading its full system prompt and tool set with no cache.
        #
        # So the window trades a steep periodic cost for a bounded creeping one.
        # 4 completed a real 8-turn recording that the unbounded version could
        # not; `timeout_s` sits well above the observed fresh-turn cost so that
        # periodic spike is survivable, and agent/loop.py retries a turn that
        # still times out rather than letting it end the run. The API deciders
        # never face this trade -- their rolling 4-turn history window costs
        # nothing to reset -- which is one more reason claude-cli is an explicit
        # choice and never the auto-selected backend.
        self.resume_window = resume_window
        self._turns_on_session = 0

    def _tool_names(self) -> list[str]:
        from agent.tools import TOOLS

        tools = self._tools_override if self._tools_override is not None else TOOLS
        return [t["name"] for t in tools]

    def _tools_description(self) -> str:
        from agent.tools import TOOLS

        tools = self._tools_override if self._tools_override is not None else TOOLS
        return "\n".join(
            f"- {t['name']}: {t['description']} Input schema: {json.dumps(t['input_schema'].get('properties', {}))}"
            for t in tools
        )

    def step(
        self, goal: str, obs: Observation, last_result: dict[str, Any] | None = None,
        exempt: frozenset[str] | None = None,
    ) -> DecisionResult:
        import subprocess

        exempt = exempt or frozenset()

        img_path, pii_redacted_count = self._screenshot_file(obs)
        try:
            prompt_text = self._build_prompt(goal, obs, last_result, exempt, img_path, pii_redacted_count)
            tool_names = self._tool_names()
            schema = json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "tool_name": {"type": ["string", "null"], "enum": [*tool_names, None]},
                        "tool_input": {"type": "object"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["tool_name", "tool_input"],
                }
            )
            cmd = [
                self.claude_bin, "-p", prompt_text, "--output-format", "json",
                "--json-schema", schema, "--allowedTools", self.allowed_tools,
            ]
            if self.model:
                cmd += ["--model", self.model]
            if self._session_id and self._turns_on_session >= self.resume_window:
                self._session_id = None  # window exhausted: next call starts a fresh session
                self._turns_on_session = 0
            if self._session_id:
                cmd += ["--resume", self._session_id]

            try:
                # stdin=DEVNULL: `claude -p` never needs stdin, and a child that
                # inherits it can swallow input meant for *this* process --
                # found live when a piped 'resume' for the discovery gate was
                # consumed by the CLI during the previous turn and the gate saw
                # EOF. With a terminal it is the operator's keystrokes at risk.
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=self.timeout_s, stdin=subprocess.DEVNULL
                )
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                raise RuntimeError(f"claude CLI call failed ({type(e).__name__}: {e})") from e

            if proc.returncode != 0:
                raise RuntimeError(f"claude CLI exited {proc.returncode}: {proc.stderr[:500]}")

            data = json.loads(proc.stdout)
            self._session_id = data.get("session_id") or self._session_id
            self._turns_on_session += 1

            decision = data.get("structured_output")
            if not isinstance(decision, dict):
                # Fallback: parse the raw `result` text as JSON -- the CLI's
                # own structured-output extraction can fail even when the
                # model's text response is itself valid JSON matching the
                # schema (found live: this happened on a small, unusual
                # schema during testing).
                try:
                    decision = json.loads(data.get("result", ""))
                except (json.JSONDecodeError, TypeError):
                    decision = {}

            tool_name = decision.get("tool_name")
            tool_input = decision.get("tool_input") or {}
            all_tool_calls = [{"name": tool_name, "input": tool_input}] if tool_name else []

            return DecisionResult(
                prompt=prompt_text,
                response=json.dumps(decision),
                tool_name=tool_name,
                tool_input=tool_input,
                all_tool_calls=all_tool_calls,
                pii_redacted_count=pii_redacted_count,
            )
        finally:
            pass  # nothing to clean up: the CLI reads the surface's own saved (redacted) screenshot

    def _screenshot_file(self, obs: Observation) -> tuple[str | None, int]:
        """The path the CLI's Read tool is pointed at -- the surface's saved
        screenshot, which is already redacted (see _image_b64). Previously this
        re-ran redaction into a temp copy because the on-disk file was raw; now
        that file IS the redacted one, and the temp copy would be a second OCR
        pass for nothing. (None, 0) when the surface withheld the screenshot."""
        path = Path(obs.screenshot_path)
        if not obs.screenshot_redacted or not path.exists():
            print("[pii_redact] no redacted screenshot for this turn -- the CLI decides from DOM/OCR text alone")
            return None, 0
        return str(path), obs.pii_redacted_count

    def _build_prompt(
        self, goal: str, obs: Observation, last_result: dict[str, Any] | None,
        exempt: frozenset[str], img_path: str | None, pii_redacted_count: int,
    ) -> str:
        image_line = (
            f"Read the screenshot at {img_path} with your Read tool before deciding.\n"
            if img_path else
            "No screenshot is available this turn (redaction failed closed) -- decide from the DOM/OCR text alone.\n"
        )
        last_result_line = f"\nResult of your last action: {json.dumps(last_result)}\n" if last_result is not None else ""
        return (
            f"Goal: {goal}\nCurrent URL: {obs.url}\n{image_line}"
            f"DOM excerpt:\n{_redacted_dom_excerpt(obs, 4000, exempt)}"
            f"{_ocr_prompt_section(obs, exempt)}"
            f"{_image_redaction_note(pii_redacted_count)}"
            f"{last_result_line}\n"
            f"Available tools (choose exactly one per turn):\n{self._tools_description()}\n\n"
            f"Respond with exactly one JSON object: "
            f'{{"tool_name": <the literal tool name from the list above, or null if you are only '
            f'reasoning this turn>, "tool_input": {{...fields matching that tool\'s input schema...}}, '
            f'"reasoning": "why this is a robust choice"}}. tool_name must be one of the exact names '
            f"listed above (e.g. \"click\", not a description of clicking) -- never invent a different "
            f"field layout for tool_input."
        )
