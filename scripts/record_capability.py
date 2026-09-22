"""Autonomous capability recording: no curl, no Flask server, no human
driving each step -- agent/loop.py's run_discovery_loop drives the live LLM
loop itself and declares its own checkpoint/outputs as part of the same
conversation (agent/tools.py's DISCOVERY_TOOLS), then this script hands the
result straight to artifact/from_run.py's build_artifact_from_run. This is
what an earlier curl-driven harness existed as a workaround
for, back when no working API key was available -- now that AnthropicDecider
is proven live (see README), there's no reason left to drive this by hand.

Usage:
    ./run.sh python scripts/record_capability.py "<url>" "<goal>" [artifact_id] \
        [--backend auto|anthropic|ollama|claude-cli] [--model NAME] [--max-steps N] \
        [--ollama-host URL] [--allow-blocked login,admin,...]

`--ollama-host` points the Ollama backend (explicit, or picked by `auto`) at
a daemon other than localhost:11434 -- a GPU box on the LAN, say. Unset, it
falls back to the OLLAMA_HOST env var, the same one the `ollama` CLI reads.

`--allow-blocked` explicitly overrides specific default-blocked route shapes
(guardrails/allowlist.py's DEFAULT_BLOCKED_PATTERNS -- login/checkout/payment/
admin/...) for this one recording, e.g. `--allow-blocked login,admin` for a
goal that genuinely requires logging in or reaching an admin area on a
*known, reviewed* target. Never the default: recording against an arbitrary
site keeps those routes blocked on purpose.

`--backend claude-cli` drives decisions through the `claude` CLI
(ClaudeCodeCLIDecider) instead of the raw Anthropic API -- for testing
without a metered ANTHROPIC_API_KEY, using whatever auth `claude` already
has. Real, measured tradeoff, not assumed: a fresh CLI call reloads Claude
Code's own system prompt every turn (~$0.19-0.40, live-measured) before the
model call's own cost; this decider resumes the same CLI session after its
first turn to hit the prompt cache and cut that to ~$0.01/turn. Needs the
`claude` binary on PATH and already logged in (`claude /login` once,
interactively, outside this script).

`artifact_id` defaults to a slug derived from the goal if omitted.

`--input NAME=VALUE` declares a typed input up front: every recorded value
carrying that literal -- the text the model types, the checkpoint it declares,
an output extractor's target -- is written as `{{NAME}}`, so the artifact takes
it as an argument at replay instead of baking this run's value in. Without it
the recording is a single literal capture of exactly what happened.
"""

from __future__ import annotations

import argparse
import json
from urllib.parse import urlsplit
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # e.g. ANTHROPIC_API_KEY, if using --backend anthropic

from agent.deciders import AnthropicDecider, ClaudeCodeCLIDecider, OllamaDecider, default_decider
from agent.loop import DiscoveryState, StopCondition, run_discovery_loop
from escalation.operator_cli import print_discovery_stopped, read_operator_note, wait_for_operator_signal
from agent.tools import DISCOVERY_TOOLS, TOOLS
from artifact.from_run import build_artifact_from_run
from telemetry.log import RunLog
from guardrails.allowlist import AllowlistConfig
from scripts.urls import normalize_url
from surfaces.base import Action
from surfaces.web import GuardrailBlocked, WebSurface

CDP_PORT = 9334  # replay uses 9333; distinct so the two can run side by side

_DISCOVERY_TOOLSET = TOOLS + DISCOVERY_TOOLS


def _slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].strip("-") or "capability"


def build_decider(backend: str, model: str | None, ollama_host: str | None):
    if backend == "auto":
        decider = default_decider(model, tools=_DISCOVERY_TOOLSET, ollama_host=ollama_host)
        picked = "anthropic" if isinstance(decider, AnthropicDecider) else "ollama"
        print(f"[decider] --backend auto -> {picked} ({'ANTHROPIC_API_KEY is set' if picked == 'anthropic' else 'no ANTHROPIC_API_KEY, using local Ollama'})")
        return decider
    if backend == "anthropic":
        return AnthropicDecider(model=model, tools=_DISCOVERY_TOOLSET) if model else AnthropicDecider(tools=_DISCOVERY_TOOLSET)
    if backend == "ollama":
        return OllamaDecider(model=model or "qwen2.5:14b-instruct", host=ollama_host, tools=_DISCOVERY_TOOLSET)
    if backend == "claude-cli":
        return ClaudeCodeCLIDecider(model=model, tools=_DISCOVERY_TOOLSET)
    raise ValueError(f"unknown backend {backend!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url")
    parser.add_argument("goal")
    parser.add_argument("artifact_id", nargs="?", default=None, help="defaults to a slug derived from the goal")
    parser.add_argument("--template-id", default=None, help="defaults to artifact_id with '-' -> '_'")
    parser.add_argument("--backend", choices=["auto", "anthropic", "ollama", "claude-cli"], default="auto")
    parser.add_argument("--model", default=None)
    parser.add_argument("--ollama-host", default=None,
                        help="Ollama base URL; defaults to $OLLAMA_HOST, else http://localhost:11434")
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument(
        "--timeout-s", type=float, default=300.0,
        help="wall-clock budget for the whole run (default 300). Raise it for a slower backend -- --backend claude-cli shells out per turn and is several times slower than the API path, so a longer flow can run out of time long before it runs out of steps",
    )
    parser.add_argument(
        "--allow-blocked", default=None,
        help="comma-separated route shapes to un-block for this recording (e.g. 'login,admin') -- only for a known, reviewed target",
    )
    parser.add_argument(
        "--input", action="append", default=[], metavar="NAME=VALUE",
        help="declare a typed input this capability will take, with the literal value to use for this recording "
             "(repeatable, e.g. --input account_number=1234567890). Every recorded value carrying that literal -- "
             "typed text, the declared checkpoint, an output extractor's target -- is written as {{NAME}}, so the "
             "artifact takes it as an argument at replay instead of baking this run's value in. Implies --pii-exempt "
             "for the value.",
    )
    parser.add_argument(
        "--pii-exempt", default=None,
        help="comma-separated literal values (e.g. a target account ID) to leave unmasked by guardrails/pii_redact.py -- "
             "for a goal that needs to read back a specific value it already named, see guardrails/pii_redact.py's docstring",
    )
    args = parser.parse_args()

    if args.backend == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Either export it, drop --backend to let --backend auto pick Ollama for you, "
              "or pass --backend ollama explicitly (needs `ollama serve` running).")
        return 1

    url = normalize_url(args.url)
    artifact_id = args.artifact_id or _slugify(args.goal)
    template_id = args.template_id or artifact_id.replace("-", "_")

    try:
        decider = build_decider(args.backend, args.model, args.ollama_host)
    except RuntimeError as e:
        print(str(e))
        return 1

    print(f"[decider] model={getattr(decider, 'model', '?')} vision={decider.supports_vision}")
    if not decider.supports_vision:
        print("[decider] no screenshot is shown to the model -- decisions (and declarations) are made from the DOM excerpt plus OCR text read off each screenshot, not pixels directly.")

    run_id = f"discovery_auto_{time.strftime('%Y-%m-%d_%H%M%S')}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = args.allow_blocked.split(",") if args.allow_blocked else None
    allowlist = AllowlistConfig.for_url(url, allow_patterns=allow_patterns)
    print(f"[guardrails] scoped to {allowlist.allowed_domains}; blocking {len(allowlist.blocked_path_patterns)} risky-looking route patterns by default.")
    if allow_patterns:
        print(f"[guardrails] --allow-blocked explicitly un-blocked: {allow_patterns} -- your call that this target is safe for it.")
    print("[note] automating a site is your call re: its ToS/robots.txt -- not checked here.")

    # Resolve the exempt set before the surface exists: the surface redacts
    # every screenshot it writes, and a declared input (the account number the
    # goal is about) has to stay visible in them or the model cannot find the
    # row -- and neither can whoever reviews the evidence afterward.
    declared_inputs: dict[str, str] = {}
    for spec in args.input:
        name, sep, value = spec.partition("=")
        if not sep or not name.strip() or not value:
            print(f"--input expects NAME=VALUE, got {spec!r}"); return 1
        declared_inputs[name.strip()] = value
    if declared_inputs:
        print(f"[inputs] recording {sorted(declared_inputs)} as typed inputs -- their literals become {{{{name}}}} placeholders")
    pii_exempt = frozenset(args.pii_exempt.split(",")) if args.pii_exempt else frozenset()
    pii_exempt = pii_exempt | frozenset(declared_inputs.values())
    if pii_exempt:
        print(f"[pii_redact] left visible in prompts and saved screenshots: {sorted(pii_exempt)}")

    # cdp_port: the live session stays attachable so that, if the recording
    # stops short, a human can look at -- and fix -- the actual page before
    # deciding whether the run really failed. 9334, not replay's 9333, so a
    # recording and a replay can run side by side.
    surface = WebSurface(allowlist=allowlist, screenshot_dir=run_dir / "screenshots", pii_exempt=pii_exempt, cdp_port=CDP_PORT)
    run_log = RunLog(run_id=run_id, out_dir=run_dir)

    _STUCK_REASONS = {
        "dead_end": "the model repeated the same action/declaration 4 times in a row without progressing",
        "max_steps": f"ran out of steps ({args.max_steps}) before calling done",
        "timeout": "ran out of wall-clock time before calling done",
        "ask_user": "the model explicitly asked for human guidance instead of proceeding",
        "decider_error": "the model backend raised on a turn and again on retry (see decider_error events in the log)",
        "no_checkpoint": "the model called done without ever declaring a checkpoint, so there is nothing to build a success check from",
    }

    try:
        try:
            surface.act(Action(type="go_to", params={"url": url}))
        except GuardrailBlocked as e:
            print(f"blocked before starting: {e.reason}")
            return 1

        # ---- record, and gate every stop on a human ----------------------
        # A run that does not produce an artifact is not discarded on the
        # system's own say-so. The live session stays open, a person is shown
        # why it stopped, and they decide: confirm it genuinely failed, or --
        # having cleared whatever blocked it (a popup the model could not
        # dismiss, a slow page, a wrong turn) -- resume the model from exactly
        # where it stopped, with everything recorded so far intact
        # (agent/loop.py's DiscoveryState). Unattended, this defaults to
        # confirm: a non-answer is never treated as "I fixed it".
        state = DiscoveryState()
        while True:
            result = run_discovery_loop(
                goal=args.goal,
                surface=surface,
                run_log=run_log,
                run_dir=run_dir,
                template_id=template_id,
                stop=StopCondition(max_steps=args.max_steps, timeout_s=args.timeout_s),
                decider=decider,
                pii_exempt=pii_exempt or None,
                declared_inputs=declared_inputs,
                state=state,
            )
            built = result.status == "success" and result.detail.get("artifact_buildable")
            if built:
                break

            status = result.status if result.status != "success" else "no_checkpoint"
            reason = _STUCK_REASONS.get(status, status)
            print_discovery_stopped(args.goal, status, reason, result.detail, f"{run_dir}/log.jsonl", f"http://localhost:{CDP_PORT}")
            run_log.emit("discovery_stopped", {"status": status, "reason": reason, "detail": result.detail, "resumes_so_far": state.resumes})
            verdict = wait_for_operator_signal(timeout_s=args.timeout_s, choices=("confirm", "resume"), default="confirm")
            run_log.emit("discovery_stop_verdict", {"verdict": verdict})
            if verdict != "resume":
                print(f"run_dir={run_dir}")
                print(f"status={status} steps_taken={result.steps_taken} detail={result.detail}")
                print("recording discarded as confirmed-failed" if verdict == "confirm" else "recording discarded")
                return 2 if status != "no_checkpoint" else 3

            state.resumes += 1
            state.operator_note = read_operator_note()
            print(f"[operator] resuming the model from step {state.step_in_template} (resume #{state.resumes})")
            if state.resumes > 5:
                print("too many resumes -- stopping rather than looping forever")
                return 2
    finally:
        # try/finally, not a bare close() after the loop: a crash mid-recording
        # used to leave a real Chromium process running.
        surface.close()

    print(f"run_dir={run_dir}")
    print(f"status={result.status} steps_taken={result.steps_taken} detail={result.detail}")
    if state.resumes:
        print(f"[note] this recording was resumed {state.resumes} time(s) after human intervention -- review the artifact "
              f"for steps the human performed that the model did not: those are NOT in the artifact")

    artifact = build_artifact_from_run(
        run_log_path=str(run_dir / "log.jsonl"),
        declared_outcomes_path=result.detail["declared_outcomes_path"],
        artifact_id=artifact_id,
        template_id=template_id,
        # Scope is the origin (what the allowlist actually enforces on -- a
        # deeper path here only made the pattern lie about what was permitted);
        # the entry point is the exact URL this run started from, recorded as
        # its own field rather than reverse-engineered from the pattern.
        base_url_pattern=f"{urlsplit(url).scheme}://{urlsplit(url).netloc}/*",
        entry_url=url,
    )
    # on_conflict="bump": artifact_id defaults to a slug of the goal, so
    # re-recording the same goal targets the same path. Overwriting there used
    # to silently destroy the earlier recording -- different steps, different
    # checkpoint, same file. A new recording is a new version; nothing is lost
    # either way, and the caller is told which version it landed at.
    out_path = artifact.save_versioned("artifacts", on_conflict="bump")
    if out_path.name != f"{artifact.artifact_id}.v{artifact.version}.json":
        print(f"\nnote: {artifact.artifact_id}.v{artifact.version}.json already held a different "
              f"recording -- this one was saved alongside it, not over it")
    print(f"\nartifact written to {out_path}")
    print(artifact.model_dump_json(indent=2))
    # Show a real, runnable example -- not a blind '{}' that would fail
    # InputValidationError for any artifact that declares required inputs.
    # declared_inputs holds the actual literal used for this recording, so
    # reuse it; anything in input_schema we have no literal for (shouldn't
    # normally happen) falls back to a type-appropriate placeholder.
    example_inputs = {
        name: declared_inputs.get(name, {"integer": 0, "number": 0.0}.get(type_name, "<value>"))
        for name, type_name in artifact.input_schema.items()
    }
    print("\nreplay it deterministically (zero LLM calls) with:")
    print(f"  ./run.sh python scripts/run_single_replay.py {out_path} '{json.dumps(example_inputs)}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
