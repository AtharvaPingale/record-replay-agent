"""Structured, append-only run log. One line of JSON per event. Redaction is
applied before anything touches disk -- this is the only write path into a run's
log file, so there's one place to get that right rather than N.

Every step of a live (LLM-driven) run is logged as a single `llm_turn` event
carrying everything needed to audit that decision on its own, without cross-
referencing other lines:
  - screenshot   path to the exact image the model (or, for a discovery run
                 driven via the `claude-cli` backend, the Claude Code session
                 standing in for the API call -- see agent/deciders.py) was shown
  - prompt       the exact text/context sent alongside that screenshot
  - response     the model's raw response (tool call + any accompanying text)
  - decision     the action actually taken, as a plain {type, params} dict
  - reason       why -- the model's own stated rationale, verbatim, not a
                 paraphrase
  - all_tool_calls (in extra) every tool call the model proposed this turn,
                 not just the one `decision` reflects -- a turn can genuinely
                 propose more than one (see agent/deciders.py's
                 AnthropicDecider), and this is the one place that isn't
                 silently dropped
A deterministic replay step has no LLM turn, but is logged just as fully via
`emit_replay_step`: same shape, with `decision`/`reason` sourced from what the
artifact declared instead of a live model response, and `prompt`/`response`
explicitly null so it's never ambiguous which kind of step produced a given line.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from guardrails.redact import mask_known_credentials_deep, redact_value


@dataclass
class LogEvent:
    run_id: str
    step: int | None
    kind: str  # "llm_turn" | "replay_step" | "act" | "guardrail_block" | "escalation_*" | ...
    detail: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class RunLog:
    def __init__(self, run_id: str, out_dir: Path):
        self.run_id = run_id
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / "log.jsonl"

    def emit(self, kind: str, detail: dict[str, Any], step: int | None = None) -> None:
        # Two redaction passes, in this order, at the one write path into a run
        # log so no caller can forget either:
        #   1. by provenance -- any known credential's literal value, anywhere in
        #      the structure, becomes its {{env:VAR}} marker. Catches what shape
        #      matching cannot: a random password appearing in a goal string, a
        #      model's own response, or an error echoing what it tried to type.
        #   2. by shape -- the pattern list (emails, card numbers, SSNs, keys).
        safe = redact_value(mask_known_credentials_deep(detail))
        event = LogEvent(run_id=self.run_id, step=step, kind=kind, detail=safe)
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(event)) + "\n")

    def emit_llm_turn(
        self,
        step: int,
        screenshot_path: str | Path,
        prompt: str,
        response: str,
        decision: dict[str, Any],
        reason: str | None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """One line covering everything about one live decision: what the model
        was shown (screenshot + prompt), what it said (response), what it chose
        to do (decision), and why (reason) -- all in one place, per step."""
        self.emit(
            "llm_turn",
            {
                "screenshot": str(screenshot_path),
                "prompt": prompt,
                "response": response,
                "decision": decision,
                "reason": reason,
                **(extra or {}),
            },
            step=step,
        )

    def emit_replay_step(
        self,
        step: int | None,
        screenshot_path: str | Path | None,
        decision: dict[str, Any],
        reason: str | None,
        result: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> None:
        """The deterministic-replay counterpart to emit_llm_turn -- same shape,
        `prompt`/`response` explicitly None since no model call happened; `reason`
        is the artifact step's own declared robustness_reasoning, not inferred."""
        self.emit(
            "replay_step",
            {
                "screenshot": str(screenshot_path) if screenshot_path else None,
                "prompt": None,
                "response": None,
                "decision": decision,
                "reason": reason,
                "result": result,
                **(extra or {}),
            },
            step=step,
        )

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
