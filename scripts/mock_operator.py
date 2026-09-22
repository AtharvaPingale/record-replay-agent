"""The human operator's side of a handoff, scripted -- a *separate process* that
attaches to the automation's live browser over CDP and performs one step on it.

This is the deliberately-mocked operator surface (REPORT.md §5): a real
operator would open the devtools URL printed by scripts/run_single_replay.py's
intervention notice and act in their own browser. What is real and identical
here is the mechanism -- `chromium.connect_over_cdp` is a second, independent
client connection to the *same* browser process and *same* page the
automation is paused on, not a fresh browser pointed at the same URL. Session
state (the admin login, the filtered list) is exactly what the automation
left, and whatever this process does is what the automation resumes on.

Usage:
    ./run.sh python scripts/mock_operator.py <artifact> <inputs_json> --step N [--cdp http://localhost:9333]

Performs step N of the artifact -- using the artifact's *declared* locator and
the caller's inputs -- via its own Playwright driver, then disconnects. Only
its own client connection closes; the shared browser keeps running. Nothing
here touches the automation's surface, run log or result: the replay process
still re-verifies its own checkpoint through the shared surface after the
operator signals 'done' (escalation/resume.py), rather than trusting anything
this script reports.
"""

from __future__ import annotations

import argparse
import json

from playwright.sync_api import sync_playwright

from artifact.schema import Artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("artifact_path")
    parser.add_argument("inputs_json")
    parser.add_argument("--step", type=int, required=True, help="1-based step of the artifact to perform by hand")
    parser.add_argument("--cdp", default="http://localhost:9333")
    args = parser.parse_args()

    inputs = json.loads(args.inputs_json)
    artifact = Artifact.load(args.artifact_path)
    step = next(s for s in artifact.steps if s.step == args.step)
    params, _ = artifact.render_step_params(step, inputs)
    if step.target is None:
        print(f"step {args.step} has no element target; nothing for an operator to do")
        return 1
    locator = step.target.primary

    with sync_playwright() as pw:
        browser = pw.chromium.connect_over_cdp(args.cdp)
        page = browser.contexts[0].pages[0]
        print(f"[operator] attached over CDP to the live page: {page.url}")
        if locator.kind == "dom_selector":
            handle = page.locator(locator.value).first
        elif locator.kind == "text":
            handle = page.get_by_text(locator.value, exact=False).first
        else:
            print(f"[operator] cannot act on locator kind {locator.kind!r} from here")
            return 1
        if step.action == "click":
            handle.click()
        elif step.action == "type":
            handle.fill(params["text"])
        else:
            print(f"[operator] action {step.action!r} not supported by this mock operator")
            return 1
        page.wait_for_timeout(800)
        print(f"[operator] performed step {args.step} ({step.action} on {locator.kind} {locator.value!r}); now at {page.url}")
        browser.close()  # this client's connection only -- the shared browser stays up
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
