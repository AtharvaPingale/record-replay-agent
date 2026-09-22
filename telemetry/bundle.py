"""On hard_failure only: a self-contained bundle (screenshot + DOM snapshot + the
full step trace up to that point) written to evidence/<run_id>/failure/, so a
failure is debuggable without live access to the app.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from guardrails.redact import redact_value
from surfaces.base import EvidenceBundle


def write_failure_bundle(
    run_dir: Path,
    step: int | None,
    surface_bundle: EvidenceBundle,
    trace_so_far: list[dict[str, Any]],
    error_detail: dict[str, Any],
) -> Path:
    failure_dir = run_dir / "failure"
    failure_dir.mkdir(parents=True, exist_ok=True)

    # screenshot + DOM snapshot are moved alongside the bundle metadata -- moved,
    # not copied, so the surface's staging directory doesn't linger as a second
    # copy of the same evidence next to the bundle.
    screenshot_dest = failure_dir / "screenshot.png"
    dom_dest = failure_dir / "dom_snapshot.html"
    # The surface writes a screenshot only if it could redact it first; a
    # withheld screenshot is recorded as such rather than crashing the bundle.
    shot_src = Path(surface_bundle.screenshot_path)
    shot_written = shot_src.exists()
    if shot_written:
        shot_src.replace(screenshot_dest)
    dom_src = Path(surface_bundle.dom_snapshot_path)
    dom_src.replace(dom_dest)
    for staging in {shot_src.parent, dom_src.parent}:
        if staging != failure_dir and staging.is_dir() and not any(staging.iterdir()):
            staging.rmdir()

    manifest = {
        "step": step,
        "url": surface_bundle.url,
        "error": redact_value(error_detail),
        "trace_so_far": redact_value(trace_so_far),
        "screenshot": str(screenshot_dest.relative_to(run_dir)) if shot_written else None,
        "screenshot_withheld": None if shot_written else "redaction unavailable -- see surfaces/web.py",
        "dom_snapshot": str(dom_dest.relative_to(run_dir)),
        **surface_bundle.extra,
    }
    manifest_path = failure_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path
