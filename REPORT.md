# record-replay-agent — Design Report

A goal in natural language → one LLM-driven run against a live UI → a typed,
versioned capability artifact → deterministic replay with no model in the loop →
a human handoff on the same live session when replay can't safely finish.

The recorded capability is **`vb-bank-admin-balance`**: log in as an admin,
open User Management, look up a member by account number, return their balance
— the brief's own "look up member 12345" shape, against a demo bank admin
console (no real institution, no real customer data). It was recorded **fully
autonomously** by Claude driving `agent/loop.py`, with `account_number` as a
typed input, and replays with zero model calls. Two versions are committed,
both genuine recordings: `v1` has a real race-condition bug in its extractor;
`v2` is an independent re-recording that structurally can't make that mistake.
A third recording grounds the `account_not_found` outcome. Evidence for all
three runs, three of the four replay outcomes (`recoverable` was never needed
on this target) and the handoff is in [`evidence/`](evidence/).

This report states each decision; **[DECISIONS.md](DECISIONS.md)** carries the
full reasoning, what was observed live that motivated it, and its limits, under
the same seven headings.

## 1. Architecture

Five layers, one process, synchronous. Each talks only to the layer below
through a narrow type:

```
scripts/          CLIs: record a capability · replay an artifact · play the operator
agent/            observe → decide → act loop; pluggable Decider (Anthropic | claude-cli | Ollama)
artifact/         the capability contract: schema + build-from-run
replayer/         deterministic executor + the result taxonomy
surfaces/         the Surface protocol; web.py (Playwright) is the only implementation
locator/ guardrails/ telemetry/ escalation/   cross-cutting, all surface-agnostic
```

- **`Surface` is the only thing that knows what a UI is.** Nothing above it
  imports Playwright. An artifact records *what to do* and *how to find the
  control*; the surface knows *how to perceive and act*. Section 4 rests on
  this seam.
- **The recorder declares; the builder collapses.** The model declares its
  checkpoint, outputs and business outcomes in the same conversation as its
  actions, with a required `robustness_reasoning` per locator and
  `idempotency_note` per step; `artifact/from_run.py` turns the trace into an
  artifact mechanically. Inferring those afterwards from raw actions is
  exactly the guesswork a deterministic system shouldn't do.
- **Single process, no queue; one action per turn.** The brief doesn't reward
  scaling infrastructure. Cost: latency on long flows (7 turns took 128 s).

## 2. Artifact schema

[`artifact/schema.py`](artifact/schema.py). An artifact is a *contract*, not a
script: `input_schema` in, `output_schema` out, `checkpoint` as proof it worked.

```
Artifact
  artifact_id, version, revision_note
  target        {surface: web|legacy_web|desktop, base_url_pattern, entry_url}
  input_schema  {field: "string"|"integer"|"number"|"boolean"}
  output_schema {field: type}          + output_extractors[] (how each is read)
  checkpoint    Checkpoint             the success condition
  business_outcomes[]                  declared expected-negative states (+ alert_human)
  recoverable_patterns[]               declared friction + an executable recovery_action
  steps[]       {action, target{primary, fallback, robustness_reasoning},
                 params, idempotency_note, max_recovery_attempts}
```

- **Expected-negative states are declared, never inferred.** "No such member"
  and "the page is broken" are indistinguishable at the level of *checkpoint
  didn't match*; only an advance declaration separates them. A happy-path
  recording never sees a not-found state, so the committed outcome comes from
  a separate probe recording whose declaration is spliced in verbatim (a test
  enforces the copy is exact).
- **The contract can name its inputs.** `{{account_number}}` appears in the
  checkpoint and extractor, not just step params. This is the v1 → v2 fix:
  v1 read "the first dollar amount in the results card", which its own
  screenshot shows racing an async filter; v2's extractor is anchored on the
  account number's own text node, so there is no other row it could read.
  The fix was re-recording with a more specific goal, not editing JSON.
- **Locators are an ordered list** — DOM selector → visible text → OCR text →
  viewport-fraction coordinates.
- **Versioned and enforced.** `save_versioned` never overwrites a version a
  caller depends on; `revision_note` is the one field a recording can't write,
  and a test checks nothing else changed alongside it. Inputs are validated
  before any step runs; an artifact whose `output_schema` and extractors
  disagree is rejected on load.

## 3. Determinism & error handling

Replay ([`replayer/executor.py`](replayer/executor.py)) makes zero model calls.
Determinism comes from declared locators, declared waits, and an explicitly
verified checkpoint — never "the click returned, so it worked."

**Every replay returns exactly one of four results:**

| Result | Meaning | Caller does | Evidence |
|---|---|---|---|
| `success` | checkpoint verified | reads `outputs` | `replay-success/` |
| `business_outcome` | a *declared* expected state | handles it as a real answer | `replay-business-outcome/` |
| `recoverable` | declared transient friction | nothing — replay runs the declared `recovery_action` and retries | (none needed on this target) |
| `hard_failure` | undeclared, unrecovered | debugs from the evidence bundle | `replay-hard-failure/` |

`business_outcome` and `recoverable` match only against patterns the artifact
declares; anything else is `hard_failure`. **Nothing escapes as an exception** —
a guardrail refusal, invalid inputs, an exhausted retry budget and a blown
deadline are all *results*, because an exception escaping the production path
is an outage with no diagnosis. Retries are capped per step, the checkpoint
gets a render-settle poll before being judged unmet, and a whole-replay
deadline bounds a half-broken app.

This target's real runtime condition is the async filter: the list briefly
says "No users found" before data arrives — the same text as the genuine
not-found. The committed pattern has no URL scoping (the model wasn't asked
for one), so a transient render elsewhere could misclassify. That named gap
is why every `business_outcome` is put in front of a human before it is
returned (Section 5). **UI drift**, secondarily: fallback locators absorb
small changes; anything larger is a `hard_failure` with screenshot, DOM
snapshot and step trace.

## 4. Heterogeneity & multi-tenant

**Surface abstraction.** `Surface` is `observe()`, `act()`, `is_visible`,
`extract_text`, `snapshot_for_evidence`. A legacy web app needs no schema
change — framesets are a `web.py` concern, and `dom_selector` degrades to
`text` then `ocr_text`. A desktop backend implements the same methods over
UIAutomation/AX APIs; `dom_selector` becomes an accessibility-tree query, and
`ocr_text` and `relative_coords` already work anywhere pixels do. The
no-clean-DOM case is built, not just designed: [`locator/ocr.py`](locator/ocr.py)
resolves targets against rendered pixels, always after DOM candidates.

**Multi-tenant reuse.** An artifact describes one *vendor product's*
capability, not one tenant's install: a base artifact per (vendor,
capability) plus a thin per-tenant overlay patching only what differs — a
`base_url_pattern`, a branded label, one selector, an extra recoverable
pattern. Tenants matching the base carry no overlay. An overlay is a merge
before load; neither schema nor executor changes. **Drift detection** falls
out of the result taxonomy: a rising per-tenant `hard_failure` rate on a
shared base is the signal to re-record or pin an overlay. Not built, per the
brief.

## 5. Escalation & handoff

**Nothing short of a clean success is committed on the system's own say-so.**
Three gates: a `business_outcome` or empty-output success is shown to a person
with a redacted screenshot and the still-open session for `confirm`/`reject`
(unattended: returned but marked `human_verified: false`); a `hard_failure`
escalates; a discovery run that stops without an artifact stays open for
`confirm` or `resume` from exactly where it stopped. **"Stuck" is the
`hard_failure` path** — by then declared patterns and bounded retries are spent.

**Taking control of the live session — the part that has to be real.**
Chromium runs with its CDP port exposed. On escalation the process pauses,
prints an intervention request (goal, inputs, step, reason, evidence path,
CDP endpoint) and blocks on stdin. The human attaches to the **same browser
process and same page** — not a relaunch, which would lose the admin session.
In the evidence the human is `scripts/mock_operator.py`, a separate process
that `connect_over_cdp`s, performs the step, and disconnects; replay never
knows whether a person or a script was on the other end.

**Control is recorded state**, not inferred from a blocked thread:
`ControlOwner` flips to `OPERATOR` on handoff and back to `AUTOMATION` on
every return path (`done`, `skip`, timeout), logged with each event. The
session is snapshotted before and after through the automation's own handle
and diffed, recording the operator's *effect* (not keystrokes). **Every
non-answer resolves to `skip`, never `done`**, and resolution is re-verified
through the artifact's own checkpoint via
[`escalation/resume.py`](escalation/resume.py) — the same question a normal
replay asks, never the operator's say-so. `evidence/escalation-handoff/` ends
`success` with real outputs that way.

**Scope:** re-verifies the *final* checkpoint, not mid-artifact continuation.
The operator surface is a console prompt, mocked per the brief; the
pause/cede/resume mechanism under it is not.

## 6. Safety

**Two independent layers, no opt-out**, at the single `act()` choke point:
an **allowlist** of domains, routes and action types, enforced at the network
layer via `page.route` (aborts before a byte loads, whatever triggered the
navigation) and again post-action for client-side routing; and a **risk
classifier** that blocks a risky action type on a payment/signup/deletion-
shaped URL even if navigation slipped past.

**Two trust tiers.** An agent improvising on an unreviewed site gets the
dynamic scope — whole domain, every risky-looking route shape blocked until a
human names it (`--allow-blocked login,admin`). Replay of a reviewed artifact
uses `config/allowlist.json` verbatim, routes enumerated positively. Payment,
signup and deletion stay blocked in both; login is un-blockable because it
costs nothing irreversible by itself.

**Secrets by provenance, not pattern** — `{{env:VAR}}` resolves at the moment
of use and is never written; any recorded value matching a known credential
env var is swapped back to its marker at the single log write path.
**PII redacted before anything is written**: the screenshot is OCR'd in
memory, every account-number/SSN/currency/email-shaped region and its whole
row painted out, and only *that* is saved, shown to the model, or committed.
The DOM excerpt gets the same treatment. Fails closed: no OCR, no screenshot.

**Limits:** over-redacts 9-digit numbers by design; row-proximity needs a
row; risk rules are a blocklist, so the allowlist is the real gate; the CDP
port is localhost-bound but open for the whole replay; `RiskLevel.CONFIRM` is
declared but not yet enforced — no rule uses it, and it must be wired before
one does. Full list in DECISIONS.md §6.

## 7. Cuts

- **Multi-tenant overlays, desktop backend, co-browsing console** — designed,
  not built, per the brief.
- **Mid-artifact step resumption** after a handoff — final-checkpoint
  re-verification only.
- **A negative-case probe *inside* one recording** — an exploratory action
  would graduate into a required replay step today; the probe runs as a
  separate recording instead.
- **Keystroke-level capture** of the operator — before/after effect only.
- **Stretch goals** — not attempted, in favour of depth on the load-bearing
  pieces.

**Next, in order:** (1) the in-recording negative probe, so the model declares
business outcomes itself; (2) per-tenant overlays plus drift logging — purely
additive; (3) mid-artifact resumption after a handoff.
