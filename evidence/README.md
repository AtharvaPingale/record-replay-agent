# Evidence

The required end-to-end thread: a genuine LLM-driven discovery run → a saved
artifact → deterministic replays covering three of the four result classes
(`recoverable` is declared by the schema; this target never needed one)
→ a real human handoff over the same live browser session.

Every `log.jsonl` here is genuine output from a real run against the live demo
site — nothing in this directory is fabricated or hand-written to look like a
result. Screenshots are the browser's own captures, committed unmodified; every
one of them was PII-redacted in memory before it was first written, so no
unredacted version exists anywhere. `screenshot` paths inside the logs point at
the original `runs/…` directory each run was written to (gitignored scratch);
the files themselves are the `screenshots/` folder next to each log.

## Three discovery runs feed two committed artifacts

Both `vb-bank-admin-balance.v1.json` and `.v2.json` are genuine, independent
autonomous recordings — **neither is a hand edit of the other.** They're both
kept, deliberately, so the fix between them is a comparison a reader can make
directly rather than a claim to take on faith.

### `discovery-run-v1/` — the first recording

```
./run.sh python scripts/record_capability.py "https://vb-bank-demo.vercel.app/login" \
  "Log in as the admin, go to User Management, look up the member with account number 1234567890 and read their current account balance. Declare the balance as an output named account_balance, and declare a checkpoint proving that specific member's row is showing." \
  vb-bank-admin-balance --input account_number=1234567890 --backend claude-cli --allow-blocked login,admin --max-steps 25 --timeout-s 900
```

- `log.jsonl` — 7 turns. `declare_checkpoint`: `selector_visible`,
  `.users-card :text("{{account_number}}")`. `declare_output`: `dom_selector`,
  `.users-card :text-matches("^[$][0-9,]+([.][0-9]{2})?$")` — the first
  currency-shaped text in the results card.
- `screenshots/004.png` — the requested account number visible, every other
  member's row painted out, **and the list still showing three rows** while
  the async filter catches up. This is the race the extractor above doesn't
  account for.
- `declared_outcomes.json` — `business_outcomes: []`: the goal is a pure happy
  path, so nothing prompted the model to consider a not-found case.

`artifacts/vb-bank-admin-balance.v1.json` is this run's output, rebuildable
byte-for-byte with `scripts/build_artifact.py` (no arguments); a test enforces it.

### `discovery-run-v2/` — the same goal, one added sentence

```
./run.sh python scripts/record_capability.py "https://vb-bank-demo.vercel.app/login" \
  "Log in as the admin, go to User Management, and look up the member with account number 1234567890. The results list can still be showing other members' rows for a moment after you search, before it finishes filtering down to just this one -- so when you declare the checkpoint and the output, scope both to the specific row that contains this account number (for example, anchor the selector on an element that itself contains the account number text), not to 'a row is visible' or 'the first balance value on the page' in general, since either of those could match a different member's row while the list is still settling. Declare the balance as an output named account_balance." \
  vb-bank-admin-balance --input account_number=1234567890 --backend claude-cli --allow-blocked login,admin --max-steps 25 --timeout-s 900
```

- `log.jsonl` — 7 turns, 128 s. `declare_checkpoint`: `selector_visible`,
  `text={{account_number}}` — the account number's own text becoming visible,
  inherently row-scoped since it's unique. `declare_output`: `dom_selector`,
  `xpath=//*[normalize-space(text())='{{account_number}}']/following-sibling::*[1]`
  — the cell *next to that exact account number*, wherever it sits.
- `screenshots/004.png` — the same race as v1's (three rows still rendered),
  which is the point: v2's locators don't need the race to not happen.

**What actually changed, precisely.** v1's extractor was never *caught*
returning a wrong balance — every replay of it in this repo's history happened
to run while the requested row was still first in the DOM. What's real and
checkable is the extractor's *target string*: v1's has no reference to
`{{account_number}}` at all; v2's is anchored on it
(`tests/test_committed_artifacts.py::test_v2_extractor_is_scoped_to_the_requested_row_unlike_v1`
asserts exactly this). The fix was not editing v1's locator — it was giving
the recorder a more specific goal and running it again. A human reading
`.users-card :text-matches("^[$][0-9,]+...")` in review has no way to see a
race condition in a selector's syntax; watching a re-recording unfold does.

`artifacts/vb-bank-admin-balance.v2.json` is this run's output. It also
carries a `revision_note` explaining the above in the author's own words — the
one field a discovery run has no way to produce itself, along with `version:
2`. `tests/test_committed_artifacts.py` checks the rest of the artifact
matches the rebuild from this evidence field-for-field, and that those two
fields are present.

### `discovery-run-notfound-probe/` — grounding the business outcome

Neither v1's nor v2's goal is a not-found lookup, so neither run has anything
to declare there. This is a *third*, separate recording whose goal is
specifically to search a nonexistent account and declare what the app shows:

```
./run.sh python scripts/record_capability.py "https://vb-bank-demo.vercel.app/login" \
  "Log in as the admin, go to User Management, and search for the account number 0000000000, which does not belong to any real member. Once you can see what the app shows for a search with no matching results, declare that state as a business outcome named account_not_found (with a checkpoint pattern that proves specifically that the search returned zero results, not just that the page loaded), and declare a checkpoint for the same state. Then call done immediately -- you do not need to look up a real account number or declare any output for this task." \
  vb-bank-admin-notfound-probe --backend claude-cli --allow-blocked login,admin --max-steps 10 --timeout-s 300
```

- `log.jsonl` — 7 turns, 82 s. `declare_business_outcome`:
  `account_not_found`, matching `text_present` `"No users found"`, with no
  `only_when_url_contains` scoping — exactly what the model declared, nothing
  tightened afterward. That's a real, named gap (REPORT.md §3, DECISIONS.md §3): the list also
  renders "No users found" for a moment while loading, which is part of why
  every `business_outcome` is confirmed by a human before being returned as
  the answer, not trusted outright.
- This run's `steps` are never used — its goal isn't the capability being
  built. What's used is only its `business_outcomes` list, copied verbatim
  into v2's `declared_outcomes.json` before that artifact is built
  (`test_merged_business_outcomes_are_a_verbatim_copy_of_their_source_run`
  checks the copy is exact).

**Why a splice instead of one combined recording:** `agent/loop.py`'s
`step_in_template` increments on every successful action with no dedup, so an
exploratory "search a wrong account number" action taken mid-session would
graduate into the artifact as a required replay step — every future replay
would type the wrong number before the right one. Running the probe
separately and copying only its *declaration* (never its steps) avoids that
while keeping the outcome grounded in a real observation rather than typed by
a human. REPORT.md §7 names the real fix (a way to mark an action exploratory
so it can inform a declaration without graduating into `steps`) as not built.

All three runs are the committed evidence for **pre-LLM PII redaction**:
`pii_redacted_count` is 21 on the full user list and 7 on the filtered one in
v1 and v2 alike; the prompts contain `[REDACTED:bank_account_number]`,
`[REDACTED:currency_amount]` and `[REDACTED:email]` where the page's real
values were. The screenshots are the browser's own captures, committed
unmodified — there is no unredacted version on disk anywhere, because the
capture is redacted in memory before it is first written.

## `artifact/` — both versions, in their final form

Copies of `artifacts/*.json` (a test asserts they match).

## The replays: one per result-taxonomy branch, all against v2 (zero LLM calls)

- **`replay-success/`** — `{"account_number": "1234567890"}` → `success`,
  `outputs: {"account_balance": "$15,000.00"}`, read off the live page at the
  checkpoint.
- **`replay-business-outcome/`** — `{"account_number": "0000000000"}` →
  `business_outcome: account_not_found`, `human_verified: true`. The log shows
  `verification_requested` → `verification_resolved` with the operator's
  `confirm`: a not-found is a legitimate answer, but it was put in front of a
  person before being returned as one.
- **`replay-hard-failure/`** — step 3's locators deliberately corrupted
  (`vb-bank-admin-balance.v2.BROKEN.json`, written alongside this run's own
  evidence by `run_single_replay.py --demo-break-step 3 --demo-operator
  skip`). `hard_failure` at step 3, `locate_failed`, with the
  bundle: `failure/screenshot.png`, `failure/dom_snapshot.html`
  (text-redacted), `failure/manifest.json` (step, error, locators tried, full
  trace up to the failure). The escalation was raised and the operator
  answered `skip`, so `escalation_resolved` has `resolved: false` and the
  result stays `hard_failure`.
- **`escalation-handoff/`** — the same broken artifact, but this time the
  operator acts: `hard_failure` → `escalation_raised` (`control_owner:
  operator`) → a **separate process** (`scripts/mock_operator.py`) attaches to
  the paused browser over CDP at `localhost:9333`, types the account number into
  the real search field on the *same* logged-in page, disconnects → the
  operator signals `done` → `operator_actions` (before/after URL,
  `control_owner: automation` again) → `escalation_resolved` `resolved: true`
  → final result `success` with `outputs: {"account_balance": "$15,000.00"}`,
  re-verified through the artifact's own checkpoint on the shared surface, not
  the operator's word.

v1 is not exercised in these replays — it's kept purely as the comparison
point for v2, per the paragraph above. Reproduce any of the replays with the
commands in the [README](../README.md#demo-run-the-agent-then-replay-the-resulting-artifact).
