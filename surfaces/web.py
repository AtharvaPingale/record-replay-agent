"""Playwright implementation of the Surface protocol -- the only concrete surface
this build implements (see surfaces/base.py, and REPORT.md Section 4 for how a
legacy-web or desktop surface would plug in beside this one without changing
anything upstream).

Every `act()` call passes through the allowlist and risk guardrails *before*
touching the page. This is the single choke point all upstream callers (agent
loop, replayer) go through, so guardrails can't be accidentally
bypassed by adding a new caller later.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from playwright.sync_api import Browser, Page, sync_playwright

from guardrails.allowlist import AllowlistConfig, check_action_type, check_navigation
from guardrails.risk import DEFAULT_RULES, RiskLevel, classify
from locator.locate import LocateError, locate
from surfaces.base import Action, EvidenceBundle, GuardrailBlocked, Observation


# Re-exported from surfaces/base.py, where it now lives so that non-Playwright
# callers (replayer/executor.py) can catch it without importing this module.
# Kept importable from here too -- `from surfaces.web import GuardrailBlocked`
# is what every existing caller and test already writes.
__all__ = ["GuardrailBlocked", "WebSurface"]


class WebSurface:
    def __init__(
        self,
        allowlist: AllowlistConfig,
        screenshot_dir: Path,
        headless: bool = True,
        viewport: dict[str, int] | None = None,
        action_delay_s: float = 0.4,
        cdp_port: int | None = None,
        risk_rules: list = None,  # noqa: RUF013 -- None means "use the default rules"
        pii_exempt: frozenset[str] = frozenset(),
    ):
        self.allowlist = allowlist
        self.screenshot_dir = screenshot_dir
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.action_delay_s = action_delay_s  # self-imposed rate limit
        # Every screenshot this surface writes is redacted first
        # (guardrails/pii_redact.py's persist_redacted_screenshot): the raw
        # capture exists only in memory. `pii_exempt` names literal values to
        # leave visible -- the account number the caller asked to look up, say,
        # which they already hold and need to see to verify the run -- while
        # every *other* member's number, balance and email on the same page is
        # painted out. The same OCR pass also feeds Observation.ocr_excerpt, so
        # this costs one pass per observe, not two, and there is no
        # "skip OCR for latency" option any more: a screenshot that skipped
        # redaction would be a screenshot that skipped Section 3.4.
        self.pii_exempt = pii_exempt
        self._redaction_warned = False
        # Overridable because this is the seam for the two deliberately
        # different trust tiers REPORT.md Section 6 describes: DEFAULT_RULES
        # (the default here) governs an unattended agent improvising against
        # an unreviewed site -- discovery via agent/loop.py's run_discovery_loop,
        # driven live via scripts/record_capability.py -- and blocks broadly.
        # REPLAY_RISK_RULES
        # (guardrails/risk.py) is the narrower boundary scripts/run_single_replay.py
        # passes explicitly: replay only ever executes an artifact's own
        # already-declared, already-reviewed steps, so the guardrail's job there
        # is catching drift, not re-litigating a decision a human already made.
        self.risk_rules = risk_rules if risk_rules is not None else DEFAULT_RULES
        self._pw = sync_playwright().start()
        # cdp_port exposes this browser's CDP endpoint over TCP so a *second*
        # client (a real operator's browser, or -- as scripts/mock_operator.py
        # does -- a second Playwright client) can attach to this exact running
        # session for a real human handoff (escalation/handoff.py).
        launch_args = [f"--remote-debugging-port={cdp_port}"] if cdp_port else []
        self.browser: Browser = self._pw.chromium.launch(headless=headless, args=launch_args)
        self.page: Page = self.browser.new_page(viewport=viewport or {"width": 1280, "height": 900})
        self._shot_counter = 0

        # Discovered on an earlier e-commerce discovery run: that site
        # is ad-supported, and third-party ad overlays (non-deterministic copy,
        # different on every impression) intercepted clicks and hijacked
        # navigation. Ad copy can't be a reliable declared recoverable_pattern --
        # there's nothing stable to match. Blocking known ad-serving domains at
        # the network layer is the honest fix: ads aren't part of the product/cart
        # functionality under test, and this keeps replay deterministic instead of
        # gambling on which ad happens to load.
        _AD_DOMAINS = ("doubleclick.net", "googlesyndication.com", "google-analytics.com", "googleadservices.com")

        def _route_handler(route):
            url = route.request.url
            if any(d in url for d in _AD_DOMAINS):
                return route.abort()
            # True prevention, not just after-the-fact detection: a click on
            # an <a href> to a blocked route (this site's own "Place Order" ->
            # /payment, confirmed live) triggers a real top-level navigation
            # request here, before the browser ever renders a byte of the
            # response. Aborting it at the network layer means the blocked
            # page is never loaded at all -- act()'s post-click check below is
            # kept as a second, independent layer for whatever this one
            # doesn't catch (e.g. client-side routing with no new network
            # request), not because this one is expected to miss anything.
            if route.request.is_navigation_request() and route.request.frame == self.page.main_frame:
                decision = check_navigation(self.allowlist, url)
                if not decision.allowed:
                    return route.abort()
            return route.continue_()

        self.page.route("**/*", _route_handler)

    def close(self) -> None:
        self.browser.close()
        self._pw.stop()

    # -- Surface protocol -------------------------------------------------

    def current_url(self) -> str:
        return self.page.url

    def is_visible(self, text: str) -> bool:
        try:
            return self.page.get_by_text(text, exact=False).first.is_visible()
        except Exception:
            return False

    def is_selector_visible(self, selector: str) -> bool:
        try:
            return self.page.locator(selector).first.is_visible()
        except Exception:
            return False

    def extract_text(self, target: list[dict[str, Any]]) -> str | None:
        # Deliberately does NOT go through locate()/its wait_for(state="visible")
        # -- found live that a real, correctly-matching element (a cart line's
        # product-name link) can fail Playwright's strict "visible" heuristic
        # while still being genuinely attached and readable via inner_text().
        # "Visible" is the right bar for an action (you shouldn't click
        # something a real user couldn't see or reach); it's the wrong bar for
        # reading text that's already rendered, so this only waits for
        # "attached," which is what inner_text() does natively.
        #
        # Found live via scripts/record_capability.py's own output: a model
        # recording a capability reasonably declared its outputs with kind
        # "text" (agent/tools.py's declare_output tool explicitly offers it,
        # same locator-robustness reasoning as click/type preferring visible
        # text over a guessed selector) -- but this method silently skipped
        # every "text" candidate and fell through to None, so a "successful"
        # replay would have returned {"view_count": None, "like_count": None}
        # without ever raising anything. Fixed by actually supporting it,
        # not by pretending the tool schema shouldn't offer it.
        for candidate in target:
            try:
                if candidate["kind"] == "dom_selector":
                    text = self.page.locator(candidate["value"]).first.inner_text(timeout=2000)
                elif candidate["kind"] == "text":
                    text = self.page.get_by_text(candidate["value"], exact=False).first.inner_text(timeout=2000)
                else:
                    continue  # a coordinate candidate has nothing reliable to read text from
                return text.strip()
            except Exception:
                continue
        return None

    def _write_redacted_screenshot(self, out_path: Path):
        """Capture in memory, redact, write. Returns (written, count, words).
        On any redaction failure nothing is written -- see
        persist_redacted_screenshot for why that is the only acceptable answer."""
        from guardrails.pii_redact import persist_redacted_screenshot

        raw = self.page.screenshot()
        try:
            return persist_redacted_screenshot(raw, out_path, exempt=self.pii_exempt)
        except Exception as e:  # noqa: BLE001 - any failure here means "withhold", never "write raw"
            if not self._redaction_warned:
                print(
                    f"[pii_redact] screenshot redaction unavailable -- NO screenshots will be saved for this run "
                    f"rather than saving them unredacted ({type(e).__name__}: {e}). Install the `ocr` extra to fix."
                )
                self._redaction_warned = True
            return False, 0, None

    def observe(self) -> Observation:
        self._shot_counter += 1
        shot_path = self.screenshot_dir / f"{self._shot_counter:03d}.png"
        written, redacted_count, words = self._write_redacted_screenshot(shot_path)
        # Kept near-full (not the ~4k slice an LLM prompt uses) because classify.py
        # needs to reliably see a page's own success/business-outcome markers,
        # which on a real page's markup can sit tens of thousands of characters
        # in -- truncating here caused a real false-positive misclassification
        # live, once. Callers building an LLM prompt (agent/loop.py) are
        # responsible for their own token-budget slice.
        dom_excerpt = self.page.content()[:150000]
        ocr_excerpt = "\n".join(w.text for w in words if w.confidence >= 0.5) if words else None
        return Observation(
            url=self.page.url,
            title=self.page.title(),
            screenshot_path=shot_path,
            dom_excerpt=dom_excerpt,
            ocr_excerpt=ocr_excerpt or None,
            timestamp=time.time(),
            pii_redacted_count=redacted_count,
            screenshot_redacted=written,
        )

    def snapshot_for_evidence(self, out_dir: Path) -> EvidenceBundle:
        """The richer on-failure signal (Section 3.5) -- and, being the one
        artifact most likely to be attached to a ticket and passed around,
        the one that most needs to carry no raw PII. Screenshot goes through
        the same redact-or-withhold path as observe(); the DOM snapshot goes
        through the text redaction (guardrails/pii_redact.py's patterns, same
        exempt set), since a page's HTML carries every value its screenshot
        does, in plain text."""
        from guardrails.pii_redact import redact_pii_text

        out_dir.mkdir(parents=True, exist_ok=True)
        shot_path = out_dir / "evidence.png"
        dom_path = out_dir / "evidence_dom.html"
        written, redacted_count, _ = self._write_redacted_screenshot(shot_path)
        dom_path.write_text(redact_pii_text(self.page.content(), exempt=self.pii_exempt))
        return EvidenceBundle(
            screenshot_path=shot_path, dom_snapshot_path=dom_path, url=self.page.url,
            extra={"screenshot_redacted": written, "pii_redacted_count": redacted_count},
        )

    def _is_secret_field_filled_with_literal(self, handle, action: Action) -> bool:
        """Coarse, best-effort signal: the *page's own* markup says this is a
        password field (real DOM `type="password"`, not a name/selector guess),
        and the value just filled into it isn't provenance-tagged as coming from
        a {{env:VAR}} credential reference (action.credential_keys, set by
        replayer/executor.py's _step_to_action). Never blocks the action --
        this is a tripwire for review, not enforcement."""
        try:
            field_type = handle.get_attribute("type")
        except Exception:
            return False
        return field_type == "password" and "text" not in action.credential_keys

    def act(self, action: Action) -> dict[str, Any]:
        decision = check_action_type(self.allowlist, action.type)
        if not decision.allowed:
            raise GuardrailBlocked(decision.reason)

        if action.type == "go_to":
            url = action.params["url"]
            nav_decision = check_navigation(self.allowlist, url)
            if not nav_decision.allowed:
                raise GuardrailBlocked(nav_decision.reason)
            # Found live: go_to was the one action type NOT wrapped like
            # click/type/select below -- a bad URL (missing scheme, a typo, a
            # real DNS/timeout failure) crashed the whole process instead of
            # coming back as a structured, classifiable result the way every
            # other action's failure already does. Same discipline as
            # everywhere else in this codebase: a real runtime error is
            # data for the caller (business_outcome/recoverable/hard_failure
            # via replayer/classify.py), never an unhandled crash.
            try:
                self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            except Exception as e:  # noqa: BLE001 - surfaced as a structured hard_failure upstream, same as click/type/select
                return {"ok": False, "error": "act_exception", "detail": str(e)}
            self.page.wait_for_timeout(self.action_delay_s * 1000)
            return {"ok": True}

        if action.type in ("done", "ask_user"):
            return {"ok": True}  # no page interaction; agent loop handles semantics

        # every element-targeting action is risk-classified against the *current* url
        rule = classify(action.type, self.page.url, self.risk_rules)
        if rule and rule.level == RiskLevel.BLOCKED:
            raise GuardrailBlocked(f"risk guardrail blocked '{action.type}' on {self.page.url}: {rule.reason}")

        if action.type == "wait":
            self.page.wait_for_timeout(action.params.get("ms", 500))
            return {"ok": True}

        if action.type == "key":
            self.page.keyboard.press(action.params["combo"])
            self.page.wait_for_timeout(self.action_delay_s * 1000)
            return {"ok": True}

        if action.type == "assert_text":
            # page-wide visibility check, not a locate() -- there's no single
            # element to target, we're asking "is this text rendered anywhere."
            return {"ok": self.is_visible(action.params["text"])}

        try:
            result = locate(self.page, action.target)
        except LocateError as e:
            return {"ok": False, "error": "locate_failed", "detail": str(e)}

        try:
            if action.type == "click":
                if result.handle is not None:
                    result.handle.click(timeout=5000)
                else:
                    self.page.mouse.click(*result.coords)
            elif action.type == "type":
                text = action.params["text"]
                if result.handle is not None:
                    result.handle.fill(text, timeout=5000)
                    secret_field_literal_value = self._is_secret_field_filled_with_literal(result.handle, action)
                else:
                    self.page.mouse.click(*result.coords)
                    self.page.keyboard.type(text)
                    secret_field_literal_value = False  # no DOM handle to check the field's real `type` attribute against
            elif action.type == "select":
                value = action.params["value"]
                if result.handle is not None:
                    result.handle.select_option(value, timeout=5000)
                else:
                    return {"ok": False, "error": "select_requires_dom_handle"}
            else:
                return {"ok": False, "error": f"unhandled action type {action.type}"}
        except Exception as e:  # noqa: BLE001 - surfaced as a structured hard_failure upstream
            return {"ok": False, "error": "act_exception", "detail": str(e)}

        if action.type == "click":
            # Found live while building the OCR-based locator (locator/ocr.py):
            # a coordinate-driven click (page.mouse.click at a bare x/y, which
            # is what both "ocr_text" and "relative_coords" resolve to, unlike
            # a DOM-handle click) can return before the browser has finished
            # processing the navigation it triggered -- on a real live site,
            # `self.page.url` read immediately afterward sometimes still shows
            # the *previous* page. A DOM-handle click's
            # own `.click()` call was never affected (Playwright's element
            # actionability handling behaves differently there), so this went
            # unnoticed until the OCR locator started exercising the
            # coordinate path far more than `relative_coords` ever had.
            # Concretely reproduced: 3 consecutive real runs where the next
            # line's guardrail check, and the caller's own `current_url()`
            # read, both still reported the pre-click URL. This isn't just an
            # inconvenience -- the guardrail check two lines below reads
            # `self.page.url` to catch a click that reached a blocked route
            # (finding 8, README), so a stale read here would make that check
            # look at the wrong page entirely. Waiting for the load lifecycle
            # to settle first (bounded, and a no-op if the click didn't
            # navigate at all -- an already-loaded page satisfies "load"
            # immediately) fixes both the guardrail check and the caller's own
            # view of where the click actually landed.
            try:
                self.page.wait_for_load_state("load", timeout=3000)
            except Exception:  # noqa: BLE001 - no navigation happened, or it's still in flight; the checks below are honest either way
                pass

            # Found live: a click on an <a href> triggers real browser
            # navigation directly -- unlike "go_to", it never passes through
            # check_navigation *before* acting, because there's no destination
            # URL to check until after the click has already happened. A
            # blocked route reachable only by an in-page link would silently
            # succeed without this. Catch
            # it after the fact instead: if the click landed somewhere the
            # allowlist would have refused to `go_to`, treat it as a breach --
            # a declared artifact step should never be *capable* of doing this
            # in the first place, so reaching this branch at all means
            # something is wrong with the artifact, not just this run.
            post_click_check = check_navigation(self.allowlist, self.page.url)
            if not post_click_check.allowed:
                raise GuardrailBlocked(
                    f"click on {action.target} navigated to a blocked page ({self.page.url}): {post_click_check.reason}"
                )

        self.page.wait_for_timeout(self.action_delay_s * 1000)
        out: dict[str, Any] = {"ok": True, "matched_kind": result.kind}
        if not result.verified:
            # A coordinate-resolved action confirmed nothing was actually there
            # (locator/locate.py's relative_coords branch). Non-blocking, like
            # secret_field_literal_value below -- a signal for whoever reviews
            # the run log that this step's "ok" is weaker than the others'.
            out["unverified_locator"] = True
        if action.type == "type" and secret_field_literal_value:
            # Coarse and non-blocking: never raises, never redacts anything itself
            # -- just a signal for whoever reviews run_log.emit("act", ...) later
            # that a field the *page itself* marks as a password was filled with
            # a value that didn't come from a {{env:VAR}} reference.
            out["secret_field_literal_value"] = True
        return out
