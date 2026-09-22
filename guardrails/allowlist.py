"""Configurable allowlist: which domains/routes and which action types this agent
is permitted to touch at all. Consulted before every single `act()` -- not just at
session start -- so a mid-run redirect or an LLM decision that wanders off-target
is caught immediately, not after the fact.
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

# Generic, domain-agnostic default for AllowlistConfig.for_url -- routes shaped
# like these are blocked on *any* site until a human has looked at the specific
# target and decided otherwise (by passing extra_blocked_patterns, or by
# hand-curating a config like config/allowlist.json for a known, trusted target).
DEFAULT_BLOCKED_PATTERNS = [
    "*checkout*",
    "*payment*",
    "*/login*",
    "*/signin*",
    "*/log-in*",
    "*/logout*",
    "*/signup*",
    "*/register*",
    "*/delete*",
    "*/admin*",
    "*/billing*",
    "*/account/security*",
    "*password*",
]


@dataclass
class AllowlistConfig:
    allowed_domains: list[str]
    allowed_path_patterns: list[str]  # glob patterns against the URL path
    blocked_path_patterns: list[str]  # checked first; a match here always wins
    allowed_action_types: list[str]
    source_path: Path | None = field(default=None, repr=False)

    @classmethod
    def load(cls, path: str | Path) -> "AllowlistConfig":
        data = json.loads(Path(path).read_text())
        return cls(
            allowed_domains=data["allowed_domains"],
            allowed_path_patterns=data["allowed_path_patterns"],
            blocked_path_patterns=data.get("blocked_path_patterns", []),
            allowed_action_types=data["allowed_action_types"],
            source_path=Path(path),
        )

    @classmethod
    def for_url(
        cls, url: str, extra_blocked_patterns: list[str] | None = None, allow_patterns: list[str] | None = None
    ) -> "AllowlistConfig":
        """Build an allowlist scoped to whatever URL the caller hands in, for
        an arbitrary recording target (scripts/record_capability.py's own use
        of this, when the target's domain isn't already in the hand-curated
        config) rather than a specific, hand-reviewed one. Whole domain
        allowed by default -- an arbitrary site can't be pre-enumerated into
        specific allowed routes the way config/allowlist.json hand-curates for
        one known target -- so DEFAULT_BLOCKED_PATTERNS below is the actual
        gate: routes shaped like checkout/payment/login/account-management are
        blocked regardless of domain. Combined with guardrails/risk.py's own
        domain-agnostic BLOCKED classification for risky action types, this is
        two independent layers, not one -- a route that slips past one pattern
        list is still caught if the action itself looks like a checkout/payment
        click. Not a ToS or robots.txt check: automating a site you don't
        control is the caller's judgment call, not something this function
        makes for them.

        `allow_patterns`: an explicit, human-typed override removing specific
        default-blocked route shapes for this one run (e.g. `["login"]` lets
        `*/login*` through) -- for a goal that genuinely requires logging into
        a *known, reviewed* target (scripts/record_capability.py's own
        `--allow-blocked` flag is this). Matched by substring against
        DEFAULT_BLOCKED_PATTERNS' own entries, so `"login"` matches the
        `*/login*` pattern without needing the glob spelled out. Never the
        default -- a human has to name exactly which protection they're
        consciously turning off, not "trust this site generally."
        """
        domain = urlsplit(url).netloc.split(":")[0]
        bare = domain[4:] if domain.startswith("www.") else domain
        allow_patterns = allow_patterns or []
        blocked = [
            p for p in DEFAULT_BLOCKED_PATTERNS if not any(allowed in p for allowed in allow_patterns)
        ] + (extra_blocked_patterns or [])
        return cls(
            allowed_domains=[bare, f"www.{bare}"],
            allowed_path_patterns=["*"],
            blocked_path_patterns=blocked,
            allowed_action_types=["click", "type", "select", "wait", "assert_text", "go_to", "key", "done", "ask_user"],
        )

    # The hand-curated, reviewed allowlist for this build's one known target.
    # `for_artifact_target` uses it whenever an artifact's own domain is one the
    # file names; every other domain falls back to the dynamic `for_url` scope.
    HAND_CURATED_PATH = "config/allowlist.json"

    @classmethod
    def for_artifact_target(
        cls, base_url_pattern: str, hand_curated_path: str | Path | None = None,
        allow_patterns: list[str] | None = None,
    ) -> "AllowlistConfig":
        """Pick the allowlist for replaying an artifact against its own declared
        `target.base_url_pattern`.

        Two trust tiers, deliberately different (REPORT.md §6):

          - The artifact's domain is named in the hand-curated config -> that
            config, verbatim. A human has already enumerated exactly which
            routes this target's reviewed capabilities may touch, so there is
            nothing for a per-run override to add; `allow_patterns` is ignored.
          - Any other domain -> `for_url`'s dynamic scope: whole domain
            allowed, risky-looking route shapes blocked by default, and only
            the shapes the caller names in `allow_patterns` (the CLI's
            `--allow-blocked`) un-blocked. Found live: recording a capability
            with `--allow-blocked login,admin` does not by itself make the
            artifact replayable, because replay derives its own allowlist and
            knows nothing about what was overridden at recording time -- the
            override has to be given again, or the target promoted into the
            hand-curated file.

        Found live, the reason this exists at all: scripts/run_single_replay.py
        used to hardcode one config and one entry URL regardless of which
        artifact was loaded -- harmless with a single target in the repo, but
        wrong the moment scripts/record_capability.py could produce artifacts
        against other targets."""
        domain = urlsplit(base_url_pattern).netloc.split(":")[0]
        path = Path(hand_curated_path or cls.HAND_CURATED_PATH)
        if path.exists():
            curated = cls.load(path)
            if domain in curated.allowed_domains:
                return curated
        return cls.for_url(base_url_pattern, allow_patterns=allow_patterns)


@dataclass
class AllowlistDecision:
    allowed: bool
    reason: str


def check_navigation(cfg: AllowlistConfig, url: str) -> AllowlistDecision:
    parts = urlsplit(url)
    domain = parts.netloc.split(":")[0]
    if not any(domain == d or domain.endswith("." + d) for d in cfg.allowed_domains):
        return AllowlistDecision(False, f"domain '{domain}' not in allowlist {cfg.allowed_domains}")

    path = parts.path or "/"
    for pattern in cfg.blocked_path_patterns:
        if fnmatch.fnmatch(path, pattern):
            return AllowlistDecision(False, f"path '{path}' matches blocked pattern '{pattern}'")

    if cfg.allowed_path_patterns and not any(
        fnmatch.fnmatch(path, pattern) for pattern in cfg.allowed_path_patterns
    ):
        return AllowlistDecision(False, f"path '{path}' matches no allowed pattern {cfg.allowed_path_patterns}")

    return AllowlistDecision(True, "domain and path allowed")


def check_action_type(cfg: AllowlistConfig, action_type: str) -> AllowlistDecision:
    if action_type not in cfg.allowed_action_types:
        return AllowlistDecision(False, f"action type '{action_type}' not in allowlist {cfg.allowed_action_types}")
    return AllowlistDecision(True, "action type allowed")
