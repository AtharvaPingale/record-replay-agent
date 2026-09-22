"""Collapse a discovery run into the reusable artifact.

Usage: ./run.sh python scripts/build_artifact.py [run_dir] [artifact_id] [template_id] [--force]

With no arguments this rebuilds artifacts/vb-bank-admin-balance.v1.json from the
committed trace in evidence/discovery-run-v1/ -- byte-for-byte, no browser, no
model call. That reproducibility is the point: it is the cheapest way to verify
the artifact was genuinely derived from that run rather than hand-written.

v2 is not reproducible by this script alone -- it merges a second run's
business_outcomes declaration and carries a revision_note this builder doesn't
know how to write; tests/test_committed_artifacts.py does that construction
explicitly and checks it stays byte-for-byte, and evidence/README.md shows the
exact steps.

The target (base URL scope and entry URL) is read from the run trace itself --
the first `go_to` the recorder logged -- not hardcoded here, so this rebuilds
any recording scripts/record_capability.py produced, against any site.

Which is also why a rebuild that comes out *different* from what is already
committed is refused rather than silently landed -- it means the trace, the
declarations, or the schema changed, and a human should look at that. Pass
`--force` when the change is the deliberate one you just made (e.g. a schema
migration); it overwrites the existing file at the same version.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

from artifact.from_run import build_artifact_from_run
from guardrails.redact import ENV_PLACEHOLDER


def _check_env_vars_available(artifact) -> None:
    """Warn (don't fail) if a {{env:VAR}} this artifact references isn't set in
    *this* environment. Informational only -- the build machine and the replay
    machine can legitimately hold different secrets -- but a typo'd var name is
    far cheaper to catch now than at replay time, which is the only place
    guardrails/redact.py's resolve_credential_placeholders checks today."""
    missing: set[str] = set()
    for step in artifact.steps:
        for v in step.params.values():
            if not isinstance(v, str):
                continue
            m = ENV_PLACEHOLDER.match(v)
            if m and os.environ.get(m.group(1)) is None:
                missing.add(m.group(1))
    if missing:
        print(f"warning: env var(s) not set in this build environment: {sorted(missing)}")
        print("         (this may be fine if the replay environment sets them separately)")


def entry_url_from_trace(run_log_path: str | Path) -> str:
    """The URL the recording started from: the first navigation in the log.
    scripts/record_capability.py records this as the surface's own go_to before
    the model's first turn, so it is the same for every recording."""
    for line in Path(run_log_path).read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event["kind"] == "act" and event["detail"].get("action") == "go_to":
            return event["detail"]["params"]["url"]
        if event["kind"] == "llm_turn":
            prompt = event["detail"].get("prompt") or ""
            marker = "Current URL: "
            if marker in prompt:
                return prompt.split(marker, 1)[1].split()[0]
    raise ValueError(f"no navigation found in {run_log_path}; cannot derive the artifact's target")


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if a != "--force"]
    force = "--force" in sys.argv

    run_dir = argv[0] if len(argv) > 0 else "evidence/discovery-run-v1"
    artifact_id = argv[1] if len(argv) > 1 else "vb-bank-admin-balance"
    template_id = argv[2] if len(argv) > 2 else artifact_id.replace("-", "_")

    entry_url = entry_url_from_trace(f"{run_dir}/log.jsonl")
    artifact = build_artifact_from_run(
        run_log_path=f"{run_dir}/log.jsonl",
        declared_outcomes_path=f"{run_dir}/declared_outcomes.json",
        artifact_id=artifact_id,
        template_id=template_id,
        base_url_pattern=f"{urlsplit(entry_url).scheme}://{urlsplit(entry_url).netloc}/*",
        entry_url=entry_url,
    )
    _check_env_vars_available(artifact)
    try:
        out_path = artifact.save_versioned("artifacts")
    except ValueError as e:
        if not force:
            print(f"error: {e}")
            print(
                "\nThis rebuild does not match what is already committed. Either the run trace, "
                "the declared outcomes, or the schema changed.\nRe-run with --force if that change "
                "is deliberate and you want the committed artifact regenerated."
            )
            raise SystemExit(1) from e
        out_path = Path("artifacts") / f"{artifact.artifact_id}.v{artifact.version}.json"
        artifact.save(out_path)
        print(f"--force: regenerated {out_path} from {run_dir}")
    print(f"wrote {out_path}")
    print(artifact.model_dump_json(indent=2))
