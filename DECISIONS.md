# Engineering decisions

The detailed reasoning behind [REPORT.md](REPORT.md), organised under the same
seven headings so each section of the report can point here. The report says
*what* was decided; this says *why*, what was observed live that motivated it,
and where each choice's limits are.

## 1. Architecture

**Why `Surface` is the only thing that knows what a UI is.** Nothing above
`surfaces/` imports Playwright. An artifact records *what to do* and *how to
find the control*; the surface knows *how to perceive and act*. The concrete
proof that this seam is real rather than aspirational: `GuardrailBlocked` lives
in `surfaces/base.py`, not `web.py`, because a policy refusal is part of the
contract every backend owes, not a Playwright detail.

**Why the recorder declares and the builder collapses.** `run_discovery_loop`
gives the model tools to *declare* its checkpoint, outputs, business outcomes
and recoverable patterns in the same conversation as its actions, and requires
a `robustness_reasoning` on every locator and an `idempotency_note` on every
step. `artifact/from_run.py` then turns the trace into an artifact
mechanically. The alternative — diffing raw actions afterwards to guess which
values varied and which selectors were trusted — is exactly the kind of
inference a deterministic system should never make. Typed inputs follow the
same rule: `--input account_number=…` declares the parameter up front, and
every recorded value carrying that literal is written as `{{account_number}}`
at record time, so the literal appears nowhere in the artifact.

**Why single process, no queue.** The brief explicitly doesn't reward scaling
infrastructure. A replay is a synchronous function returning one typed result;
making it a service later is a transport change, not a redesign.

**Why one action per turn.** Simpler to log, reason about and bound — at the
cost of latency on long flows. The 7-turn discovery run took 128 s through
the CLI backend.

**Decider backends.** `agent/deciders.py` is pluggable: `AnthropicDecider`
(hosted, vision), `ClaudeCodeCLIDecider` (shells out to the `claude` CLI with a
strict JSON schema, resuming one session across turns so the prompt cache
holds — this is what recorded the committed run), and `OllamaDecider`
(self-hosted, text-only by default, host from `--ollama-host` / `OLLAMA_HOST`).
The local option exists for privacy, not just as a keyless fallback: for an
agent touching live account sessions, inference on infrastructure you control
is what I'd recommend in production.

## 2. Artifact schema

**Why `robustness_reasoning` and `idempotency_note` are required with no
default.** A reviewer should not have to infer why a locator was trusted or
whether a step is safe to retry. The executor's bounded retry *depends* on the
second answer.

**Why expected-negative states are declared, never inferred.** "No such
member" and "the page is broken" look identical at the level of *the
checkpoint didn't match*. The only thing that separates them is an author
saying in advance which is which — this is the brief's named most-common
mistake, and declaration is what avoids it.

It also names the limit of autonomous recording: a happy-path goal never sees
a not-found state, so has nothing to declare. The committed
`account_not_found` outcome comes from a third, separate recording
(`evidence/discovery-run-notfound-probe/`) whose goal was to search a
nonexistent account and declare what it saw. Its `business_outcomes`
declaration is copied verbatim into the primary run's `declared_outcomes.json`
before the artifact is built;
`tests/test_committed_artifacts.py::test_merged_business_outcomes_are_a_verbatim_copy_of_their_source_run`
checks the copy is exact. Nothing in the outcome's content is hand-authored;
the *splice* across two runs is the one manual step, and it's mechanical, not
creative.

**The v1 → v2 race condition, in full.** v1, given the plain lookup goal,
declared an output extractor of
`.users-card :text-matches("^[$][0-9,]+([.][0-9]{2})?$")` — the first
currency-shaped text in the results card. That was correct on the page load
it happened to see. But the list filters asynchronously: screenshot 004 in
either discovery run shows the requested row visible while two other members'
rows are still rendered, at the moment the checkpoint was already met. On a
slower render, v1's extractor could return a different member's balance as a
confident `success`. Stated precisely: this was never *observed* returning a
wrong balance in a replay — every replay of v1 in this repo's history happened
to run while the right row was first. What's real is that its own evidence
shows the race it depends on not happening.

The fix was not editing v1's JSON. It was re-recording against a goal with one
added sentence — scope the checkpoint and the output to the row containing
this account number, because the list can still be showing other rows for a
moment — and letting the model choose again. v2's extractor is
`xpath=//*[normalize-space(text())='{{account_number}}']/following-sibling::*[1]`:
the cell *next to that exact account number*. It is anchored on the requested
input's own text node, so there is no "other row" it could read; the mistake
is structurally inexpressible, not just less likely.

No reviewer reading a bare selector string would have caught this.
`robustness_reasoning` on a locator is the model's own claim; a race condition
is visible only by watching a run unfold, which is what re-recording (rather
than patching) does. Whether hand-editing an artifact's JSON is *ever* the
right call versus always re-recording is a real open question this project
didn't need to answer, since re-recording worked every time it was tried.

**Why locators are an ordered list.** `primary` then `fallback`, resolved DOM
selector → visible text → OCR text → coordinates. Recorded coordinates are
viewport *fractions*, so a different screen size still replays.

**Versioning, enforced.** `save_versioned` refuses to land different content
at a version a caller already depends on; a re-recording of the same goal
saves alongside rather than overwriting, which is why v1 and v2 both exist on
disk. `revision_note` carries the reasoning for a bump in the author's own
words — the one field a discovery run has no way to write — and a test checks
that every other field in v2 is exactly what its recording produced, so no
quiet rewrite rides along with the note. `validate_inputs` checks a caller's
arguments against `input_schema` before any step runs; a model validator
rejects an artifact whose `output_schema` and `output_extractors` disagree.
A schema nothing checks is a comment.

## 3. Determinism & error handling

**Why nothing escapes as an exception.** A guardrail refusal, invalid inputs,
an exhausted retry budget and a blown wall-clock deadline are all *results*.
Replay is the production path an agent invokes; an exception escaping it is
an outage with no diagnosis attached. A refusal is never retried and never
reclassified — a declared step capable of tripping one means the artifact is
wrong.

**Bounds.** Retries capped per step (`max_recovery_attempts` lets a step
declare its own); a render-settle poll before judging the checkpoint unmet;
one re-execution of the final step for a click the app silently no-ops; a
whole-replay deadline so a slow or half-broken app can't hang a caller.

**Recoverable patterns are executable.** A declared pattern carries a
`recovery_action` (dismiss with Escape, click, wait), so "dismiss the
interstitial" is performed rather than described. None was needed on this
target, which is why `recoverable` is the one result class with no evidence
directory.

**The runtime condition this target actually has.** The user list renders
"No users found" (0 rows) for a moment before data arrives, and the search
filter applies asynchronously — the same text as the genuine not-found state.
The committed `account_not_found` pattern is exactly what the probe run
declared, with no `only_when_url_contains` scoping hand-added afterwards (the
model wasn't asked for one, and nothing here tightens a declaration after the
fact). A transient render on some other page that happens to say those words
would misclassify. That's a named gap, and it's why every `business_outcome`
is put in front of a human for `confirm`/`reject` before it's returned
(Section 5) rather than trusted outright.

**Each mechanism exists because of something observed:**

- `element_visible` is its own checkpoint type because a present-but-CSS-hidden
  modal made a naive `text_present` checkpoint always true.
- `only_when_url_contains` exists because a `text_absent` outcome declared on
  one page spuriously re-matched on a later, unrelated one in the same flow.
- Only *successful* actions graduate into an artifact, because an early
  recorder baked a failed action in as a required step.

**UI drift**, secondarily: fallback locators absorb small changes; anything
larger surfaces as `hard_failure` with a screenshot, DOM snapshot and step
trace rather than being silently worked around. `replay-hard-failure/` is
exactly that — the step, the locators tried, and the page state.

## 4. Heterogeneity & multi-tenant

**The Surface seam.** `observe() → Observation`, `act(Action) → result`,
`is_visible`, `is_selector_visible`, `extract_text`, `snapshot_for_evidence`.
A legacy web app needs no schema change — frameset handling is a `web.py`
concern, and `dom_selector` degrades to `text` then `ocr_text`. A desktop
backend implements the same methods over UIAutomation or AX APIs;
`dom_selector` becomes an accessibility-tree query, and `ocr_text` and
`relative_coords` already work anywhere pixels do. `Target.surface` is
`web | legacy_web | desktop` today.

**The no-clean-DOM case is built.** `locator/ocr.py` resolves a click target
against words actually rendered to pixels, exposed to live deciders as
`ocr_text` and always appended *after* DOM candidates, so pixels are the
fallback and never the default. The same OCR pass drives PII redaction, so it
runs on every observation already. Limits: OCR is not tenant-invariant — font
rendering, DPI and locale change what it reads, unlike a DOM selector — and no
committed artifact declares an `ocr_text` step, because this target has a
usable DOM.

**Overlay model for tenants.** A base artifact per (vendor, capability), plus
a thin per-tenant overlay patching only what differs — `base_url_pattern`, a
branded label, one overridden selector, an extra recoverable pattern for that
tenant's interstitial. Tenants that match the base carry no overlay at all,
which is what keeps hundreds of tenants from becoming hundreds of recordings.
`config/allowlist.json` is already the per-target half of that: one reviewed
allowlist per app instance. An overlay is a merge before load; neither the
schema nor the executor changes.

**Drift detection** falls out of the result taxonomy: a tenant whose
`hard_failure` rate on a shared base artifact rises is drifting. Per-tenant
health from the existing structured log is the signal; a rising rate is the
trigger to re-record and promote a new base version, or pin that tenant to an
overlay. Not built, per the brief.

## 5. Escalation & handoff

**Three gates, in increasing weight.**

1. *Verification* — a replay that classifies as anything other than clean
   success (a declared `business_outcome`; a success whose extractor read
   nothing) shows a person what it saw — classification, final URL, redacted
   screenshot, the still-open session — and asks `confirm` / `reject`. "No
   users found" means *no such account*, but a filter that failed to apply
   produces the same words, and the caller would receive a confident, wrong
   not-found. `reject` is a hard failure on a live session and routes into
   the handoff. Unattended, the result still returns (a production replay
   must be able to say not-found) but marked `human_verified: false` — a
   sentinel no human can type.
2. *Escalation* — `hard_failure`. By the time one is returned, declared
   patterns and bounded retries are genuinely spent; that *is* the definition
   of stuck.
3. *Discovery stop* — a recording that ends without an artifact (`dead_end`,
   `max_steps`, `timeout`, `decider_error`, `ask_user`, or `done` with no
   checkpoint) is not discarded until a person has looked. The session stays
   open on its CDP port; the human either `confirm`s the failure or, having
   cleared whatever blocked the model, `resume`s it from exactly where it
   stopped — steps and declarations intact, with their note delivered to the
   model as the outcome of the previous turn. Unattended defaults to `confirm`.

**Why CDP, and why the same process.** The automation's Chromium runs with
its CDP port exposed. On escalation, execution pauses, an intervention request
prints (goal, inputs, step, reason, evidence path, CDP endpoint) and the
process blocks on real stdin. A human attaches at the printed CDP URL to the
**same browser process and same page** — not a relaunch pointed at the same
URL, which would lose the admin session. In the committed evidence the human
is `scripts/mock_operator.py`: a *separate process* that `connect_over_cdp`s
to the paused browser, performs the step the automation couldn't, and
disconnects. The replay process never knows whether a person or a script was
on the other end.

**Control is recorded state.** `ControlOwner` flips to `OPERATOR` on handoff
and back to `AUTOMATION` on every return path — `done`, `skip`, and timeout
alike — so control never silently stays with an operator who left. The log
has all three events (`escalation_raised`, `operator_actions`,
`escalation_resolved`) with the owner on each.

**What the handoff changed is recorded.** The session is snapshotted before
and after through the *automation's* own handle and diffed into the log. This
records the *effect* the operator had, not a keystroke-level trace; capturing
those means subscribing to CDP input events for the attached client — a clean
extension of this seam, not built.

**Why every non-answer resolves to `skip`, never `done`.** A timeout and an
unreachable operator must never be indistinguishable from a human saying "I
fixed it." Resolution is then re-verified through `escalation/resume.py`,
which re-observes the *shared* surface and checks the artifact's own
checkpoint and extractors — the same question a normal replay asks, never the
operator's own client's view of the page. `evidence/escalation-handoff/` ends
`success` with real outputs that way.

**Scope.** This re-verifies the artifact's *final* checkpoint after a handoff,
not generic continuation from wherever escalation happened — which is why the
escalating step in the evidence is the artifact's last one. Mid-artifact
resumption is a larger change, not attempted.

## 6. Safety

**Why two layers, enforced twice.** Every `act()` goes through both before
touching the page; `surfaces/web.py`'s `act()` is the single choke point a new
caller cannot bypass, and there is no "guardrails off" mode.

1. *Allowlist* — permitted domains, route patterns, and action types. Enforced
   at the network layer via `page.route`, which aborts a blocked navigation
   *before the browser loads a byte* regardless of what triggered it (link
   click, form submit, JS redirect), and again as a post-action URL check for
   client-side routing that makes no request.
2. *Risk classification* — blocks a risky *action type* on a
   payment/signup/account-deletion-shaped URL even if a navigation slipped
   past layer 1.

**Why two trust tiers.** An unattended agent improvising on an unreviewed
site gets the dynamic scope: whole domain allowed, every risky-looking route
shape (login, checkout, payment, admin, delete, signup, …) blocked until a
human names each one to un-block (`--allow-blocked login,admin`, as the
committed recording did). Replay of a *reviewed* artifact on a *reviewed*
target uses `config/allowlist.json` verbatim: routes enumerated positively, so
an unlisted route on the same domain is refused, not merely "not blocked".
Payment, signup and account deletion stay blocked in both tiers. Login is
un-blockable by name because it costs nothing irreversible by itself; a
transfer or a deletion is not.

**Why secrets are handled by provenance, not pattern.** A random password
matches no regex. `{{env:VAR}}` placeholders resolve at the moment of use and
are never written anywhere; on the recording side, any value matching a known
credential env var is swapped back to its marker before it reaches the log.
Both passes happen at the single write path into a run log
(`telemetry/log.py`), across every field, because a secret also travels in a
goal string and a model's response. The committed capability uses the demo's
quick-login button and so carries no credential; the round-trip is exercised
by the tests.

**Why PII redaction happens before anything is written.** `WebSurface.observe()`
captures the screenshot in memory, OCRs it once, paints every non-exempt
account-number/SSN/currency/email-shaped region black — and everything on the
*same row* as a hit, which is how names get caught without entity recognition
— and saves *that*. The file on disk, the image the decider sees, and the
evidence in `evidence/` are the same redacted image; a raw capture never
touches the filesystem. The DOM excerpt in the prompt and the DOM snapshot in
a failure bundle get the same text-side redaction. The caller's own declared
inputs are exempt: they supplied the account number and need to see it to
verify the run looked up the right record. It fails **closed**: if OCR is
unavailable the surface writes *no* screenshot rather than a raw one.

Two findings from building it: an early version redacted the image but leaked
the same values verbatim via the DOM excerpt; and a vision model shown
unexplained black boxes concluded the *site* was hiding data and tried to
route around this system's own privacy control — the prompt now says what the
redaction is, and the committed run's `done` summary shows the model reasoning
correctly about it.

**Known limits, in full.**

- A 9-digit routing number is indistinguishable from any other 9-digit number,
  so this over-redacts on non-financial pages — a deliberate fail-toward-safe
  choice, with `--pii-exempt` as the escape for a value the caller already
  named.
- Row-proximity needs a *row*; a name standing alone in prose survives, and
  the failure bundle's DOM snapshot is pattern-redacted only.
- Risk rules are a blocklist: an action/URL pair matching no rule is
  permitted, so the allowlist is the real gate.
- The password-field tripwire only catches a true `type="password"`.
- The CDP port is open for the whole replay, not only during escalation; it is
  localhost-bound, but any local process could attach.
- `RiskLevel.CONFIRM` — a third tier for irreversible-but-legitimate actions
  that should pause for a human nod — is declared in `guardrails/risk.py` but
  not enforced: `surfaces/web.py`'s `act()` only special-cases `BLOCKED`, so a
  rule at that level would execute unpaused. No rule uses it today, so it isn't
  reachable, but it must be wired before one does. Found by auditing the module
  for stale code, not by exercising it.

## 7. Cuts

**Why the negative-case probe can't happen inside one recording today.**
The schema and the tools support `declare_business_outcome` fine, but
`step_in_template` increments unconditionally on every successful action
(`agent/loop.py`), so an exploratory "search a nonexistent account" taken
mid-session would graduate into a required replay step — permanently typing a
wrong value before the real one on every future replay. The committed
`account_not_found` outcome works around this by running the probe as a
separate recording and splicing its declaration in (Section 2). The real fix
is a way for the discovery loop to mark an action as exploratory so it can
inform a declaration without graduating into `steps`.

**Why v1 wasn't patched.** `revision_note` supports hand-editing an artifact's
JSON after the fact, but the committed v1 → v2 change isn't an example of it:
v2 is a second, independent recording. Every fix found here was resolved by
re-recording with a more specific goal, because a human reading a bare locator
string has no real way to catch a race condition that a re-recording surfaces
directly.

**Why no `ocr_text` step is committed.** The locator kind works and is offered
to the model, but this target has a DOM, so the model never needed it.

**Stretch goals** (capability catalog, multi-run stability scoring, codegen)
were not attempted, in favour of depth on the load-bearing pieces.
