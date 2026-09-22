# record-replay-agent

Turns one LLM-driven run against a live UI into a typed, versioned, reusable
capability that replays deterministically with **no model in the decision loop**,
inside safety guardrails, with a real human-handoff path on the same live
session when it can't finish.

Design write-up: **[REPORT.md](REPORT.md)** (the decisions, ~3 pages) and
**[DECISIONS.md](DECISIONS.md)** (the full reasoning behind each, same seven
headings). Committed runs: **[evidence/](evidence/)**.

One recorded capability, the brief's own "look up a member and read their
balance" shape, against a demo bank admin console (`vb-bank-demo.vercel.app` —
no real institution, no real customer data):

| Capability | Input | Output | How it was recorded |
|---|---|---|---|
| `vb-bank-admin-balance` — log in as admin → User Management → filter by account number → read that member's balance | `account_number: string` | `account_balance: string` | **fully autonomously**: Claude drove `agent/loop.py` end to end via the `claude` CLI, chose every locator, wrote every `robustness_reasoning`, declared the checkpoint and the output extractor itself |

**Two versions are committed on purpose, both from genuine autonomous
recordings, so the fix between them is a real comparison and not a claim in
prose.**

| | `v1` | `v2` (recommended; used by the demo below) |
|---|---|---|
| Goal given to the recorder | the plain lookup goal | same goal, plus one sentence: scope the checkpoint and the output to the row for this account number, because the list can still be showing other rows for a moment after a search |
| `output_extractors[0].target` | `.users-card :text-matches("^[$][0-9,]+...")` — the first currency-shaped text in the results card | `xpath=//*[normalize-space(text())='{{account_number}}']/following-sibling::*[1]` — the cell *next to that exact account number* |
| The bug | correct on the page load it happened to see; on a slower render, with more than one row still showing, it could return a different member's balance as `success` | not expressible — the locator is anchored on the requested account number's own text, so there's no "other row" it could read from |
| `business_outcomes` | `[]` — the happy-path goal never saw a not-found state | `account_not_found`, grounded in a *third*, separate recording whose goal was to search a nonexistent account and declare what it saw |

Neither version was hand-edited. `v1` is exactly what its recording produced,
rebuildable byte-for-byte from `evidence/discovery-run-v1/` (a test enforces
it). `v2` is a second, independent recording rebuildable from
`evidence/discovery-run-v2/` on every field except `version` and
`revision_note` (a test enforces that too) — those two are the one place a
human's own words are load-bearing, because a discovery run has no way to
write either. See "What the discovery run actually did" below for the full
trace of both, plus the third run that grounds the business outcome.

## Setup

```bash
uv sync --extra ocr                          # first-time only
uv run playwright install --with-deps chromium   # browser + its system libs
./run.sh python -c "print('ok')"             # sanity check
```

`run.sh` is a thin wrapper over `uv run` that puts the repo root on
`PYTHONPATH` (the packages are flat top-level directories, not nested under
`src/`). It also has one sandbox-only accommodation — an `LD_LIBRARY_PATH` for a
hand-extracted `libasound.so.2`, because the machine this was built on couldn't
run `playwright install-deps` — which is skipped automatically when that
directory doesn't exist, so on a normal machine it's a no-op.

**Keys and config**

- **No `.env` is required at all.** `.env.example` documents the three optional
  variables below (copy it to `.env`, gitignored, only if you need one) — the
  demo path further down needs none of them.
- **No API key is required to replay.** Replay makes no model call of any kind.
- **Recording** needs a decider. Three backends, two of them keyless:

  | `--backend` | Auth | Vision | Notes |
  |---|---|---|---|
  | `anthropic` | `ANTHROPIC_API_KEY` (in `.env` or the environment) | yes | auto-selected when the key is set |
  | `claude-cli` | whatever the `claude` CLI is already logged in as | yes | **what recorded the committed run** — shells out to Claude Code with a strict JSON schema and `--allowedTools Read`, resuming one CLI session across turns so the prompt cache holds |
  | `ollama` | none — local daemon | model-dependent | auto-selected when no key is set; `ollama serve` first. Host from `--ollama-host`, else `OLLAMA_HOST`, else `localhost:11434` |

- `uv sync --extra ocr` — PaddleOCR. **Required for any screenshot to be saved
  at all**, on every path including replay: every screenshot is PII-redacted
  *before* it is written, that redaction needs OCR to find the regions, and it
  *fails closed* — without this extra the surface writes no screenshot rather
  than a raw one, and the run log marks each turn `screenshot_redacted: false`.
- `config/allowlist.json` — the hand-curated, reviewed allowlist for the one
  target this build has recorded against: permitted domain, positively
  enumerated routes, blocked route shapes, permitted action types. Replay of an
  artifact whose domain is listed there uses it verbatim. Any other URL gets the
  dynamic default-deny scope (`guardrails/allowlist.py`), where risky-looking
  route shapes — login, checkout, payment, admin, delete, … — are blocked until
  a human names each one to un-block (`--allow-blocked login,admin`).
- `AGENT_CREDENTIAL_VARS` (optional) — extra env var names to treat as secrets
  beyond the ones detected by name (`*PASSWORD*`, `*TOKEN*`, `*SECRET*`,
  `*API_KEY*`). Anything a model types during a recording that matches a known
  secret's value is written down as its `{{env:VAR}}` marker instead of the
  literal, which is exactly what replay resolves. The committed capability
  uses the demo's quick-login button and so carries no credential; the
  mechanism is exercised by the tests.

### Can this run without live services?

- **No LLM is needed for replay.** The demo below makes zero model calls.
- **A live site is still needed for replay**, because replay drives a real
  browser against a real UI. There is no offline/fixture mode.
- **Nothing live is needed** to run the test suite (`pytest`, 200 tests, no
  browser, no network), to rebuild the committed artifact from its saved trace
  (`scripts/build_artifact.py`), or to read any committed run in `evidence/`.

## Demo: run the agent, then replay the resulting artifact

```bash
# 1. Discovery: an LLM records the capability against the live site.
#    (This is what produced evidence/discovery-run-v2/ and artifacts/…v2.json;
#    drop the second sentence of the goal to reproduce v1's row-scoping bug.)
./run.sh python scripts/record_capability.py \
    "https://vb-bank-demo.vercel.app/login" \
    "Log in as the admin, go to User Management, and look up the member with account number 1234567890. The results list can still be showing other members' rows for a moment after you search, before it finishes filtering down to just this one -- so when you declare the checkpoint and the output, scope both to the specific row that contains this account number, not to 'a row is visible' or 'the first balance value on the page' in general. Declare the balance as an output named account_balance." \
    vb-bank-admin-balance --input account_number=1234567890 \
    --backend claude-cli --allow-blocked login,admin --max-steps 25 --timeout-s 900

# 2. Deterministic replay of the recommended version -- zero LLM calls:
./run.sh python scripts/run_single_replay.py \
    artifacts/vb-bank-admin-balance.v2.json '{"account_number": "1234567890"}'
#    -> result=success   detail={'outputs': {'account_balance': '$15,000.00'}}

# 3. A legitimate negative result, not a crash:
./run.sh python scripts/run_single_replay.py \
    artifacts/vb-bank-admin-balance.v2.json '{"account_number": "0000000000"}'
#    -> result=business_outcome  detail={'name': 'account_not_found', ...}
#    (it pauses to ask you to confirm the classification; type `confirm`)

# 4. A bad input is refused before a browser launches:
./run.sh python scripts/run_single_replay.py artifacts/vb-bank-admin-balance.v2.json '{}'

# 5. A hard failure that escalates to a human, who takes over the SAME live
#    browser over CDP, fixes it, and hands control back:
./run.sh python scripts/run_single_replay.py \
    artifacts/vb-bank-admin-balance.v2.json '{"account_number": "1234567890"}' --demo-break-step 3
#    -> hard_failure at step 3 → HUMAN ESCALATION REQUIRED → operator attaches,
#       performs the step → re-verified → result=success with outputs
#    add `--demo-operator skip` for the absent-operator path (stays hard_failure).
```

Step 1 re-records; the recording it produces is saved *alongside* the committed
versions (`save_versioned` never overwrites a different artifact at an existing
version, it bumps to the next free one), and the printed replay command names
the new file. Recording with `--allow-blocked` is the unreviewed-target tier;
replay needs no flag because the target is in `config/allowlist.json`.

Every command writes a timestamped directory under `runs/` (gitignored
scratch). `evidence/` is the curated subset kept on purpose.

## Each phase on its own

```bash
./run.sh python -m pytest tests/ -q                                   # 200 tests, no browser, no network
./run.sh python scripts/build_artifact.py                             # rebuild v1 from evidence/discovery-run-v1, byte-for-byte
./run.sh python scripts/record_capability.py "<url>" "<goal>" [id]    # discovery → artifact
./run.sh python scripts/run_single_replay.py <artifact> '<inputs>'    # deterministic replay (+ real escalation on hard_failure)
./run.sh python scripts/run_single_replay.py <artifact> '<inputs>' --demo-break-step N   # forced hard_failure → handoff → resume
./run.sh python scripts/mock_operator.py <artifact> '<inputs>' --step N        # the operator side, by itself
```

There is no separate guardrails demo, because guardrails aren't behind a flag:
both layers run inside `surfaces/web.py`'s `act()` on every run. Escalation
isn't a demo mode either — a real `hard_failure` exposes the browser's CDP
port, pauses, and blocks on real stdin for an operator by default; nothing in
that mechanism is special-cased for `--demo-break-step`, which only forces a
genuine `hard_failure` (by corrupting one step's locators in a throwaway copy)
and, unless `--demo-operator skip`, supplies the operator: a *separate
process* (`mock_operator.py`) that attaches to the paused browser over CDP,
does the step, and signals `done`.

`scripts/build_artifact.py` with no arguments rebuilds
`artifacts/vb-bank-admin-balance.v1.json` **byte-for-byte** from the committed
trace in `evidence/discovery-run-v1/`. That's the cheapest way to verify the
artifact was derived from a real run rather than hand-written;
`tests/test_committed_artifacts.py` enforces it on every test run (and, for
`v2`, the same check on every field except `version`/`revision_note` — see
below), and refuses a rebuild that comes out different unless `--force` says
the change was deliberate.

## What the discovery run actually did

Stated plainly rather than left to be pieced together. Three runs feed the two
committed artifacts.

**`evidence/discovery-run-v1/`** — the first recording, given the plain goal
with no scoping instruction. Real prompt, real structured response, 7 turns:

1. `click` the admin quick-login button — by `data-testid`.
2. `click` User Management in the sidebar — same.
3. `type` the account number into the search field — recorded as
   `{{account_number}}`, because `--input account_number=1234567890` declared it
   a typed input up front; the literal appears nowhere in the artifact.
4. `wait` 1000ms.
5. `declare_checkpoint` — `selector_visible`, `.users-card :text("{{account_number}}")`.
6. `declare_output` — `dom_selector`,
   `.users-card :text-matches("^[$][0-9,]+([.][0-9]{2})?$")`: the first
   currency-shaped text in the results card. Correct on this run's own page
   load; not scoped to any particular row.
7. `done`.

**`evidence/discovery-run-v2/`** — the same goal, plus one added sentence:
*scope the checkpoint and the output to the specific row for this account
number, because the list can still be showing other members' rows for a
moment after you search.* 7 turns / 128 s via the CLI backend:

1–4. Same login/navigate/search/wait sequence as v1.
5. `declare_checkpoint` — `selector_visible`, `text={{account_number}}`: the
   account number's own text becoming visible, which is inherently row-scoped
   since it's unique.
6. `declare_output` — not the balance's literal text, but
   `xpath=//*[normalize-space(text())='{{account_number}}']/following-sibling::*[1]`:
   the cell *sitting next to that exact account number*, so it can't read a
   different row's balance no matter how the list is still settling.
7. `done`.

**Why v1's approach is a real bug, not a style choice.** The list filters
asynchronously (screenshot 004 in either run shows the requested row visible
while two other members' rows are still rendered). v1's extractor takes
whatever currency-shaped text comes first in the DOM at that moment — on a
slower render, that could be a different member's balance, returned as a
confident `success`. v2's extractor cannot make that mistake structurally: it
starts from the account number's own text node. The fix was not editing v1's
JSON; it was giving the recorder a more specific goal and running it again.
`robustness_reasoning` on a locator is the model's own claim, and a human
reading `.users-card :text-matches("^[$][0-9,]+...")` in review has no way to
see a race condition in a selector's syntax — it's visible only by watching a
run unfold, which is what re-recording (rather than patching) does.

**`evidence/discovery-run-notfound-probe/`** — a third, separate recording,
given a different goal: search an account number that doesn't exist and
declare what the app shows. 7 turns / 82 s:

1–4. Same login/navigate/search/wait sequence, searching `0000000000`.
5. `declare_business_outcome` — `account_not_found`, matching `text_present`
   `"No users found"`.
6. `declare_checkpoint` — the same pattern, for this run's own (very different)
   goal.
7. `done`.

This run's `steps` are never used — its goal isn't the capability being
built. What's used is its `business_outcomes` declaration, copied verbatim
into v2's `declared_outcomes.json` before the artifact is built
(`tests/test_committed_artifacts.py::test_merged_business_outcomes_are_a_verbatim_copy_of_their_source_run`
checks it's an exact copy). Neither v1's nor v2's own goal hits a not-found
state, so neither has a way to declare one; this is the honest way to still
get a declaration grounded in a real observation rather than one typed by
hand. Building the ability to probe a negative case safely *inside* one
recording — without an exploratory action graduating into a required replay
step — is listed as future work in REPORT.md §7 (DECISIONS.md §7 has the mechanics).

All three runs are the committed evidence for **pre-LLM PII redaction**:
`pii_redacted_count` is 21 on the full user list and 7 on the filtered one in
v1 and v2 alike; the prompts contain `[REDACTED:bank_account_number]`,
`[REDACTED:currency_amount]` and `[REDACTED:email]` where the page's real
values were; and screenshot 004 shows every other member's row painted out
while the requested account number stays visible. The screenshots are the
browser's own captures, committed unmodified — there is no unredacted version
on disk anywhere, because the capture is redacted in memory before it is first
written.

## When a human is consulted

Every result short of a clean success is put in front of a person before the
system acts on it — three gates of increasing weight, all with a bounded wait
whose unattended default is the *conservative* option and is recorded as such:

| Trigger | What happens | Choices | Unattended |
|---|---|---|---|
| Replay returns a `business_outcome`, or a `success` with a missing output | classification + redacted screenshot + final URL shown; session left open to look at | `confirm` / `reject` (→ handoff) | returns the result marked `human_verified: false` |
| Replay returns a `hard_failure` | pause; operator takes the **same** live session over CDP; resolution re-verified through the artifact's own checkpoint | `done` / `skip` | `skip`, result stays `hard_failure` |
| Recording stops without an artifact | session left open; reason shown | `confirm` (discard) / `resume` (model continues from where it stopped, with your note) | `confirm` |
| A `business_outcome` the artifact flags `alert_human` | printed and logged; nothing pauses | — | — |

Not gated: a clean `success`, a recoverable pattern that cleared, a guardrail
refusal during discovery (fed back to the model), and invalid inputs (refused
before a browser launches). The operator surface is deliberately a console
print and a stdin prompt, per the brief's scope note; the pause/transfer/resume
mechanism underneath is real.

## Guardrails and escalation, briefly

Full reasoning is in [DECISIONS.md](DECISIONS.md) §5 and §6. In short: two independent
guardrail layers with no opt-out, enforced at a single choke point and at the
network layer; two trust tiers (an unattended agent improvising on an unreviewed
site vs. replay of a reviewed artifact on a reviewed target) with payment,
signup and account deletion blocked in both; secrets handled by `{{env:VAR}}`
provenance rather than pattern-matching; PII redacted from every screenshot and
prompt before it is written or sent; and a handoff that transfers control of
the *same* live browser session over CDP, records who holds control and what
changed while they held it, and re-verifies resolution through the normal
classifier rather than the operator's own say-so.

## Using local or self-hosted models

`agent/deciders.py` is pluggable; `OllamaDecider` talks to an Ollama daemon —
on this machine by default, or anywhere you point it — and is the default
whenever no `ANTHROPIC_API_KEY` is set:

```bash
ollama serve
./run.sh python scripts/record_capability.py "<url>" "<goal>" --backend ollama --model qwen2.5:14b-instruct
# a remote Ollama (a GPU box on the LAN, say): --ollama-host http://gpu-box:11434,
# or set OLLAMA_HOST in the environment / .env, same as the ollama CLI itself.
```

The self-hosted option exists for privacy and security, not just as a
fallback for when there's no API key: nothing about the target page — the
DOM, the screenshots, the account data on screen — ever leaves infrastructure
you control, whether that's this machine or a GPU box behind `OLLAMA_HOST`.
That also opens up a real performance option this build doesn't take: every
screenshot this system writes is PII-redacted first
(guardrails/pii_redact.py), and that redaction pass is pure overhead if the
model reading it never leaves your own infrastructure in the first place.
This build still redacts unconditionally regardless of backend, since it
doesn't try to distinguish "self-hosted means trusted" as a rule — but a
deployment that's fully on self-hosted models could reasonably skip it there
and get real latency back.

Tested with text-only models, passing the DOM excerpt rather than a
screenshot — `qwen2.5:14b-instruct` above is what this was run against, and it
works well for this kind of flow. Not tested with a local vision model reading
screenshots directly; the machine this was built on didn't have the GPU memory
for one at a size worth trusting. In an ideal deployment, a self-hosted model
(a larger local model, or one run on owned/controlled infrastructure rather
than a third-party API) is what I'd actually recommend for this kind of
agent — it's touching live user sessions and account data, which is exactly
the case for keeping inference in-house rather than sending it to an external
provider by default.

## Not built, by design

Multi-tenant plumbing, a desktop backend, queues/clusters, a real co-browsing
console, mid-artifact resumption after a handoff, keystroke-level capture of the
operator's actions, and an artifact-management dashboard.
[REPORT.md](REPORT.md) §4 and §7 (and [DECISIONS.md](DECISIONS.md) for the detail) cover how the current abstractions extend to
those without a rewrite, and what was cut deliberately versus simply left undone.
