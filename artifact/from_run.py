"""Collapse a discovery run's recorded actions into one reusable, parameterized
artifact.

Design choice: rather than diffing raw actions after the fact to guess which
values varied, the discovery loop (agent/loop.py's run_discovery_loop) requires the
recorder to *declare*, at record time, (a) which template a step belongs to
(a run that repeats the same flow for several items records one template),
(b) each step's locator + robustness_reasoning, and (c) which params are
per-item variables vs. hardcoded. That's the same judgment call a human recorder
makes when naming a selector "stable" -- making it explicit here means
`from_run` collapses structure that was already captured deliberately, instead of
reverse-engineering intent from repeated strings (which is exactly the kind of
guess a deterministic replayer should never make).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from artifact.schema import (
    Artifact,
    Checkpoint,
    DeclaredOutcome,
    LocatorCandidate,
    OutputExtractor,
    RecoverablePattern,
    Step,
    StepTarget,
    Target,
)

_TYPE_HINTS = {"quantity": "integer", "amount": "integer", "price": "number"}

# Last-resort fallback only. A step's idempotency_note comes, in priority order,
# from (1) the recorded action event itself (a discovery run declares it at
# record time, the same way robustness_reasoning must be), then (2) the run's
# declared_outcomes.json `idempotency_notes` map, keyed by step_in_template --
# the same hand-declared file the checkpoint, business outcomes, and output
# extractors already come from, so rebuilding an artifact from its saved trace
# stays byte-for-byte reproducible -- and only then (3) this per-action-type
# default, for a historical log with neither.
_DEFAULT_IDEMPOTENCY_NOTE = {
    "go_to": "navigation only, side-effect-free -- reloading the same URL is a no-op",
    "type": "side-effect-free -- retyping the same value into the same field is a no-op",
    "wait": "side-effect-free",
    "select": "side-effect-free -- reselecting the same value is a no-op",
}


def _infer_type(field_name: str) -> str:
    return _TYPE_HINTS.get(field_name, "string")


def build_artifact_from_run(
    run_log_path: str | Path,
    declared_outcomes_path: str | Path,
    artifact_id: str,
    template_id: str,
    base_url_pattern: str,
    entry_url: str | None = None,
) -> Artifact:
    events = [json.loads(line) for line in Path(run_log_path).read_text().splitlines() if line.strip()]
    act_events = [
        e["detail"]
        for e in events
        if e["kind"] == "act"
        and e["detail"].get("step_template_id") == template_id
        # A recorded action that did NOT succeed is not part of the reusable
        # flow. Without this, an attempt that failed during discovery (a click
        # whose locator never resolved, an assert_text that came back false)
        # became a *required* step in the artifact -- a step known not to work,
        # guaranteeing a hard_failure on every future replay. The raw attempt
        # stays in log.jsonl either way; what changes is that it no longer
        # graduates into the capability. Historical logs predating a `result`
        # field are treated as successful, since nothing recorded otherwise.
        and (e["detail"].get("result", {"ok": True}) or {}).get("ok", True)
    ]
    if not act_events:
        raise ValueError(
            f"no successful recorded actions found for template '{template_id}' in {run_log_path}"
        )

    # canonical step sequence: taken from the first item's repetition (item_index == 0).
    # Keyed (not just filtered+sorted) by step_in_template and overwritten in log
    # order, so if a step is re-recorded after a mid-session correction (a human
    # `resume`d a stuck discovery run -- see agent/loop.py's DiscoveryState),
    # the corrected version wins rather than silently duplicating both the
    # mistake and the fix.
    by_position: dict[int, dict[str, Any]] = {}
    for e in act_events:
        if e["item_index"] == 0:
            by_position[e["step_in_template"]] = e
    first_item_steps = [by_position[k] for k in sorted(by_position)]

    declared = json.loads(Path(declared_outcomes_path).read_text())
    declared_notes: dict[str, str] = declared.get("idempotency_notes", {})

    input_schema: dict[str, str] = {}
    steps: list[Step] = []
    for e in first_item_steps:
        var_params: list[str] = e.get("var_params", [])
        var_field_names: dict[str, str] = e.get("var_field_names", {})
        rendered_params: dict[str, Any] = {}
        for k, v in e["params"].items():
            if k in var_params:
                field_name = var_field_names[k]
                rendered_params[k] = "{{" + field_name + "}}"
                input_schema.setdefault(field_name, _infer_type(field_name))
            else:
                rendered_params[k] = v

        target = None
        if e.get("target"):
            candidates = e["target"]
            target = StepTarget(
                primary=LocatorCandidate(**candidates[0]),
                fallback=LocatorCandidate(**candidates[1]) if len(candidates) > 1 else None,
                robustness_reasoning=e["robustness_reasoning"],
            )

        idempotency_note = (
            e.get("idempotency_note")
            or declared_notes.get(str(e["step_in_template"]))
            or _DEFAULT_IDEMPOTENCY_NOTE.get(
                e["action"],
                "unreviewed: recorded before idempotency_note existed -- do not assume "
                "safe to blindly retry without checking",
            )
        )

        steps.append(
            Step(
                step=e["step_in_template"],
                action=e["action"],
                target=target,
                params=rendered_params,
                idempotency_note=idempotency_note,
                max_recovery_attempts=e.get("max_recovery_attempts"),
            )
        )

    return Artifact(
        artifact_id=artifact_id,
        version=1,
        target=Target(surface="web", base_url_pattern=base_url_pattern, entry_url=entry_url),
        input_schema=input_schema,
        output_schema=declared.get("output_schema", {}),
        checkpoint=Checkpoint(**declared["checkpoint"]),
        business_outcomes=[DeclaredOutcome(**o) for o in declared.get("business_outcomes", [])],
        recoverable_patterns=[RecoverablePattern(**p) for p in declared.get("recoverable_patterns", [])],
        output_extractors=[
            OutputExtractor(name=o["name"], target=LocatorCandidate(**o["target"]))
            for o in declared.get("output_extractors", [])
        ],
        steps=steps,
    )
