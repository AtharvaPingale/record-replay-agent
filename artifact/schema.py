"""The structured capability spec (REPORT.md Section 2): typed, versioned,
serializable. One artifact = one reusable capability against the target app --
here, "log in, find a product, add it to the cart and reach checkout review,"
not "do the shopping." The contract a calling agent programs against:
`input_schema` in, `output_schema` out, `checkpoint` as the proof it worked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

LocatorKind = Literal["dom_selector", "text", "ocr_text", "relative_coords"]
ActionType = Literal["click", "type", "select", "wait", "assert_text", "go_to"]


class LocatorCandidate(BaseModel):
    kind: LocatorKind
    value: Any  # str for dom_selector/text, [x_frac, y_frac] for relative_coords


class StepTarget(BaseModel):
    primary: LocatorCandidate
    fallback: LocatorCandidate | None = None
    robustness_reasoning: str = Field(
        ..., description="Why this locator should survive minor UI drift -- required, not narrative fluff."
    )


class Step(BaseModel):
    step: int
    action: ActionType
    target: StepTarget | None = None  # None for actions with no element target (e.g. "wait")
    params: dict[str, Any] = Field(default_factory=dict)
    idempotency_note: str = Field(
        ...,
        description=(
            "What re-running this step after a partial success does -- e.g. "
            "'side-effect-free, safe to repeat' or 'NOT safe to blindly retry, "
            "would double-add'. Required, same as robustness_reasoning above and "
            "for the same reason: a stated answer a reviewer can check, not an "
            "assumed one baked silently into the executor's bounded-retry loop."
        ),
    )
    max_recovery_attempts: int | None = Field(
        default=None,
        description=(
            "Per-step override of replayer/executor.py's MAX_RECOVERY_ATTEMPTS_PER_STEP "
            "default -- a declared business contract for steps that genuinely need "
            "more (or fewer) retries than the default, not a place to tune polling "
            "cadence (that stays a runtime constant, not schema)."
        ),
    )


class Checkpoint(BaseModel):
    # element_visible matches by visible TEXT; selector_visible by CSS/Playwright
    # selector. Two types rather than one clever one because a model reading
    # "element_visible" naturally hands over a selector -- it did, on a live
    # recording, and the checkpoint silently never matched at replay. The name
    # stays for the artifacts that already use it with text; the new type gives
    # the structural case ("this row exists") an honest home.
    type: Literal["text_present", "text_absent", "url_contains", "element_visible", "selector_visible"]
    value: str
    # Optional page-scoping for a business_outcome/recoverable_pattern match
    # (the artifact's own final checkpoint has no need for this -- it's only
    # ever evaluated once, after the last step). Found live: a
    # `text_absent` business_outcome declared to mean "the search returned no
    # results" (checked right after a search step) can spuriously re-match on
    # a completely different, LATER page in the same flow that also happens
    # not to contain that text -- e.g. "add-to-cart" is legitimately absent
    # from a cart/checkout page too, which has nothing to do with whether the
    # search matched. classify_outcome is called after *any* step's failure,
    # not just the search step's, so without this the pattern isn't scoped to
    # the page it was actually observed and declared against. When set, a
    # match is only even considered while the current URL contains this
    # substring; None (the default) preserves the original unscoped behavior.
    only_when_url_contains: str | None = None


class DeclaredOutcome(BaseModel):
    name: str
    match: Checkpoint
    alert_human: bool = Field(
        default=False,
        description=(
            "Whether hitting this outcome should notify a human, even though it's "
            "not a failure. Most business outcomes are routine and need no one's "
            "attention ('no_matching_product' just tells the caller 'not found'); "
            "some are legitimate results a person should still see -- e.g. a "
            "fraud/compliance flag, a duplicate-record hit, an unusually large "
            "amount. Defaults to False so declaring an outcome never silently "
            "starts paging anyone; set True deliberately per outcome. This is a "
            "notify-only signal, distinct from hard_failure's escalation/handoff -- "
            "the replay still completes and returns normally, a human is just told "
            "about it."
        ),
    )


class RecoveryAction(BaseModel):
    """The *executable* half of a recoverable pattern -- what replay should
    actually do when the pattern matches, rather than only what a human reading
    the artifact is told happened.

    Before this existed, `RecoverablePattern.recovery` was a prose string the
    executor never read: replay retried the step blindly mid-flow and pressed a
    hardcoded Escape at the final checkpoint, whatever the pattern declared.
    That made "dismiss a known interstitial" -- the brief's own example of a
    recoverable condition -- describable but not performable. Optional, and
    `None` preserves exactly the old generic-Escape behaviour, so every artifact
    recorded before this field existed replays identically."""

    action: Literal["key", "click", "wait"] = "key"
    target: LocatorCandidate | None = None  # required for "click"; ignored otherwise
    params: dict[str, Any] = Field(default_factory=dict)  # e.g. {"combo": "Escape"} / {"ms": 500}

    @model_validator(mode="after")
    def _click_needs_a_target(self) -> "RecoveryAction":
        if self.action == "click" and self.target is None:
            raise ValueError("a 'click' recovery_action needs a target to click")
        return self


class RecoverablePattern(BaseModel):
    name: str
    match: Checkpoint
    recovery: str  # human-readable description, for a human reviewing the artifact
    recovery_action: RecoveryAction | None = Field(
        default=None,
        description=(
            "What replay actually executes to clear this pattern. None falls back to "
            "the executor's generic Escape keypress -- see RecoveryAction."
        ),
    )


class Target(BaseModel):
    surface: Literal["web", "legacy_web", "desktop"] = "web"
    # The allowlist *scope*: a glob over URLs this capability may touch.
    base_url_pattern: str
    # Where replay navigates before step 1. Kept separate from the pattern
    # above because one field was doing two jobs and got one of them wrong:
    # the recorder built the pattern as "<entry url>/*" and replay recovered an
    # entry point by stripping the "*" -- leaving a trailing slash that the
    # first autonomously recorded capability's own site answered with a 404
    # before step 1 could run. None means "derive it from base_url_pattern the
    # old way," so every artifact recorded before this field existed replays
    # exactly as it did.
    entry_url: str | None = None

    def resolved_entry_url(self) -> str:
        if self.entry_url:
            return self.entry_url
        return self.base_url_pattern.rstrip("*").rstrip("/") or self.base_url_pattern


class OutputExtractor(BaseModel):
    """How to populate one declared field of `output_schema` once the
    checkpoint is met -- without this, `output_schema` is just a promise the
    artifact never keeps. Kept as its own declared list (not inferred) for the
    same reason locators and business outcomes are declared: a human reviewing
    the artifact should see exactly what data a caller gets back and exactly
    where it comes from, not a black box."""

    name: str  # must match a key in output_schema
    target: LocatorCandidate  # where to read the text from, once the checkpoint is met


class InputValidationError(ValueError):
    """The caller's `inputs` don't satisfy the artifact's declared
    `input_schema`. Raised before any step executes, so a bad invocation is a
    clear, structured result to the caller rather than a KeyError from deep
    inside step rendering -- `input_schema` is a contract an agent calls
    against, and a contract nothing checks is only a comment."""


# Declared type name -> the Python types that satisfy it. `bool` is deliberately
# excluded from "integer"/"number" despite being an int subclass in Python: a
# caller passing True where a quantity is expected is a mistake, not an integer.
_INPUT_TYPE_CHECKS: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
}


class Artifact(BaseModel):
    artifact_id: str
    version: int = 1
    target: Target
    input_schema: dict[str, str]  # field name -> type name ("string" | "integer" | "number")
    output_schema: dict[str, str]
    checkpoint: Checkpoint
    business_outcomes: list[DeclaredOutcome] = Field(default_factory=list)
    recoverable_patterns: list[RecoverablePattern] = Field(default_factory=list)
    output_extractors: list[OutputExtractor] = Field(default_factory=list)
    steps: list[Step]
    revision_note: str | None = Field(
        default=None,
        description=(
            "Why this version differs from the one before it. None for a version "
            "that is exactly what a discovery run produced (rebuildable from its "
            "trace); required by convention for any version a human edited after "
            "review -- a tightened extractor, a newly declared business outcome -- "
            "so the reviewer's reasoning travels with the artifact rather than "
            "living in a commit message."
        ),
    )

    @model_validator(mode="after")
    def _outputs_are_actually_deliverable(self) -> "Artifact":
        """`output_schema` is what a calling agent is promised back;
        `output_extractors` is the only thing that can keep that promise. Left
        unchecked, the two drift silently: a schema key with no extractor means
        a successful replay hands the caller nothing for that field and says
        nothing about it, and an extractor with no schema key means an
        undeclared value appears in the result. Both are mismatches a human
        reviewing the artifact should never have to diff by eye."""
        declared, extracted = set(self.output_schema), {e.name for e in self.output_extractors}
        if missing := declared - extracted:
            raise ValueError(
                f"output_schema declares {sorted(missing)} but no output_extractor can "
                f"produce them -- a caller would silently get nothing back for these"
            )
        if undeclared := extracted - declared:
            raise ValueError(
                f"output_extractors produce {sorted(undeclared)} which output_schema "
                f"never declares -- add them to output_schema or drop the extractor"
            )
        return self

    def validate_inputs(self, inputs: dict[str, Any]) -> None:
        """Check a caller's `inputs` against `input_schema` *before* any step
        runs. Raises InputValidationError listing every problem at once, rather
        than failing on the first one -- an agent invoking this capability
        should get one complete answer about what it got wrong, not a fix-one-
        rerun-find-the-next loop.

        Unknown keys are an error, not ignored: `{"querry": "x"}` against a
        declared `query` is a typo that would otherwise surface much later as a
        missing-field failure deep inside step rendering, with nothing pointing
        at the real cause."""
        problems: list[str] = []
        for field_name, type_name in self.input_schema.items():
            if field_name not in inputs:
                problems.append(f"missing required input '{field_name}' (declared type '{type_name}')")
                continue
            value = inputs[field_name]
            allowed = _INPUT_TYPE_CHECKS.get(type_name)
            if allowed is None:
                problems.append(
                    f"input '{field_name}' declares unknown type '{type_name}' "
                    f"(known: {sorted(_INPUT_TYPE_CHECKS)})"
                )
            elif isinstance(value, bool) and bool not in allowed:
                problems.append(f"input '{field_name}' expects {type_name}, got bool")
            elif not isinstance(value, allowed):
                problems.append(
                    f"input '{field_name}' expects {type_name}, got {type(value).__name__}"
                )
        if unknown := set(inputs) - set(self.input_schema):
            problems.append(
                f"unknown input(s) {sorted(unknown)} -- this artifact declares "
                f"{sorted(self.input_schema) or 'no inputs'}"
            )
        if problems:
            raise InputValidationError("; ".join(problems))

    def with_inputs_rendered(self, inputs: dict[str, Any]) -> "Artifact":
        """A copy with every `{{field}}` placeholder in the *contract* --
        checkpoint value, business-outcome and recoverable-pattern match values,
        output-extractor targets -- substituted from `inputs`. Steps are left
        alone: render_step_params handles them per step, because that is also
        where `{{env:VAR}}` credentials resolve and those must never be rendered
        early or written anywhere.

        Why the contract needs this at all, found live on the first
        parameterized lookup: with a member name as the only input, a checkpoint
        of "some row is visible" and an extractor of "the first dollar amount in
        the table" both passed on a page where the search filter had *not*
        applied -- three rows showing -- and replay reported another customer's
        balance as the requested member's, as `success`. A lookup's success
        condition is "the row for THE member I asked about is showing" and its
        output is "THAT row's balance"; neither is expressible if the contract
        cannot name the input. `{{env:...}}` is deliberately not honoured here."""
        def render(value: Any) -> Any:
            if not isinstance(value, str) or "{{" not in value:
                return value
            out = value
            for field_name, field_value in inputs.items():
                out = out.replace("{{" + field_name + "}}", str(field_value))
            return out

        rendered = self.model_copy(deep=True)
        rendered.checkpoint.value = render(rendered.checkpoint.value)
        for outcome in rendered.business_outcomes:
            outcome.match.value = render(outcome.match.value)
        for pattern in rendered.recoverable_patterns:
            pattern.match.value = render(pattern.match.value)
        for extractor in rendered.output_extractors:
            extractor.target.value = render(extractor.target.value)
        return rendered

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

    def save_versioned(self, dir_path: str | Path, on_conflict: Literal["error", "bump"] = "error") -> Path:
        """Save at a path *derived from* `self.version`, refusing to silently
        destroy a different artifact already sitting at that path.

        The check is on the full serialized content, not just the input/output
        schema shape. Shape alone was not enough: two genuinely different
        recordings of the same goal -- different steps, different checkpoint --
        routinely share a schema shape (commonly an empty one), so the old guard
        waved them through and the second recording silently destroyed the
        first. Since scripts/record_capability.py derives artifact_id from a
        slug of the goal, re-running the same goal was exactly that case.

        `on_conflict`:
          "error" (default) -- refuse, and say to bump `version`. Right for
            rebuilds that are supposed to reproduce what's already there
            (scripts/build_artifact.py): a difference is a real signal.
          "bump" -- save at the next free version instead. Right for a fresh
            recording (scripts/record_capability.py), where a new version is
            the correct outcome and nothing should be lost either way.
        """
        out_path = Path(dir_path) / f"{self.artifact_id}.v{self.version}.json"
        serialized = self.model_dump_json(indent=2)
        if out_path.exists() and out_path.read_text() != serialized:
            if on_conflict == "bump":
                version = self.version
                while (candidate := Path(dir_path) / f"{self.artifact_id}.v{version}.json").exists():
                    if candidate.read_text() == serialized:
                        return candidate  # this exact artifact is already committed at this version
                    version += 1
                bumped = self.model_copy(update={"version": version})
                candidate.write_text(bumped.model_dump_json(indent=2))
                return candidate
            raise ValueError(
                f"refusing to overwrite {out_path}: an artifact with different content already "
                f"exists at this version. Bump `version` (or pass on_conflict='bump') rather "
                f"than landing different behaviour at a path a caller already depends on"
            )
        out_path.write_text(serialized)
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> "Artifact":
        return cls.model_validate_json(Path(path).read_text())

    def render_step_params(self, step: Step, inputs: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
        """Substitute placeholders in a step's params. Two distinct kinds, never
        conflated:

          {{field}}        -- an ordinary typed input, from `inputs` (caller-
                              supplied per invocation, e.g. a search query).
          {{env:VAR_NAME}}  -- a credential, resolved from the *replay
                              process's* environment at the moment of use,
                              never from `inputs`, never written to the
                              artifact JSON as anything but this reference.
                              This is what lets a step like "log in" exist in a
                              reviewable, versioned artifact without the
                              artifact -- or any log derived from it -- ever
                              containing the actual secret.

        Returns (rendered_params, credential_keys): the second names which keys
        now hold a real secret, so a caller building a log line can redact
        exactly those by provenance (guardrails/redact.py's redact_by_keys)
        rather than relying on the value happening to match a pattern.
        """
        from guardrails.redact import resolve_credential_placeholders

        field_substituted = {}
        for k, v in step.params.items():
            if isinstance(v, str) and v.startswith("{{") and v.endswith("}}") and not v.startswith("{{env:"):
                field = v[2:-2].strip()
                field_substituted[k] = inputs[field]
            else:
                field_substituted[k] = v

        return resolve_credential_placeholders(field_substituted)
