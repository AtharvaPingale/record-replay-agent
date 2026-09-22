"""Unit tests for guardrails/allowlist.py: domain/route/action-type gating,
checked before every act(), independent of guardrails/risk.py's own layer.
"""

from __future__ import annotations

from pathlib import Path

from guardrails.allowlist import (
    DEFAULT_BLOCKED_PATTERNS,
    AllowlistConfig,
    check_action_type,
    check_navigation,
)


def make_cfg(**overrides) -> AllowlistConfig:
    defaults = dict(
        allowed_domains=["shop.example", "www.shop.example"],
        allowed_path_patterns=["/", "/products*", "/checkout*", "/login*"],
        blocked_path_patterns=["/signup*", "*/payment*"],
        allowed_action_types=["click", "type", "go_to"],
    )
    defaults.update(overrides)
    return AllowlistConfig(**defaults)


def test_off_allowlist_domain_is_blocked():
    decision = check_navigation(make_cfg(), "https://example.com/")
    assert not decision.allowed
    assert "not in allowlist" in decision.reason


def test_subdomain_of_an_allowed_domain_is_allowed():
    # check_navigation matches `domain == d or domain.endswith("." + d)` --
    # a bare listed domain also covers its subdomains.
    decision = check_navigation(make_cfg(allowed_domains=["example.com"]), "https://shop.example.com/products")
    assert decision.allowed


def test_blocked_path_pattern_wins_even_though_domain_is_allowed():
    decision = check_navigation(make_cfg(), "https://shop.example/signup")
    assert not decision.allowed
    assert "blocked pattern" in decision.reason


def test_blocked_pattern_is_checked_before_allowed_pattern():
    # /payment isn't in allowed_path_patterns either, but the reason should name
    # the blocked-pattern match specifically, since blocked_path_patterns is
    # checked first and "always wins" per the module's own docstring.
    decision = check_navigation(make_cfg(), "https://shop.example/payment")
    assert not decision.allowed
    assert "blocked pattern" in decision.reason


def test_path_matching_no_allowed_pattern_is_blocked():
    decision = check_navigation(make_cfg(), "https://shop.example/some-random-path")
    assert not decision.allowed
    assert "no allowed pattern" in decision.reason


def test_allowed_domain_and_allowed_path_succeeds():
    decision = check_navigation(make_cfg(), "https://shop.example/checkout")
    assert decision.allowed


def test_empty_allowed_path_patterns_means_any_path_is_fine():
    cfg = make_cfg(allowed_path_patterns=[])
    decision = check_navigation(cfg, "https://shop.example/anything-at-all")
    assert decision.allowed


def test_check_action_type_blocks_unlisted_action():
    decision = check_action_type(make_cfg(), "select")
    assert not decision.allowed


def test_check_action_type_allows_listed_action():
    decision = check_action_type(make_cfg(), "click")
    assert decision.allowed


# -- AllowlistConfig.for_url: the generic "any link" path --------------------


def test_for_url_scopes_to_the_given_domain_and_its_www_variant():
    cfg = AllowlistConfig.for_url("https://example.com/some/page")
    assert "example.com" in cfg.allowed_domains
    assert "www.example.com" in cfg.allowed_domains


def test_for_url_strips_an_existing_www_prefix_before_deriving_the_pair():
    cfg = AllowlistConfig.for_url("https://www.example.com/some/page")
    assert set(cfg.allowed_domains) == {"example.com", "www.example.com"}


def test_for_url_blocks_generic_risky_route_shapes_on_any_domain():
    cfg = AllowlistConfig.for_url("https://example.com/")
    for risky_url, pattern in [
        ("https://example.com/checkout", "*checkout*"),
        ("https://example.com/payment", "*payment*"),
        ("https://example.com/admin/users", "*/admin*"),
    ]:
        decision = check_navigation(cfg, risky_url)
        assert not decision.allowed, f"{risky_url} should have been blocked by {pattern}"


def test_for_url_allows_everything_else_on_the_scoped_domain():
    cfg = AllowlistConfig.for_url("https://example.com/")
    decision = check_navigation(cfg, "https://example.com/some/arbitrary/page")
    assert decision.allowed


def test_for_url_merges_extra_blocked_patterns_without_losing_the_defaults():
    cfg = AllowlistConfig.for_url("https://example.com/", extra_blocked_patterns=["*/vip-lounge*"])
    assert set(DEFAULT_BLOCKED_PATTERNS).issubset(set(cfg.blocked_path_patterns))
    assert not check_navigation(cfg, "https://example.com/vip-lounge").allowed


def test_for_url_allow_patterns_unblocks_only_the_named_route_shape():
    # scripts/record_capability.py's --allow-blocked flag: an
    # explicit, human-typed override for a known, reviewed target that
    # genuinely needs to log in -- everything else stays blocked.
    cfg = AllowlistConfig.for_url("https://example.com/", allow_patterns=["login"])
    assert check_navigation(cfg, "https://example.com/login").allowed
    assert not check_navigation(cfg, "https://example.com/checkout").allowed
    assert not check_navigation(cfg, "https://example.com/admin/users").allowed


def test_for_url_with_no_allow_patterns_blocks_everything_as_before():
    cfg = AllowlistConfig.for_url("https://example.com/")
    assert set(cfg.blocked_path_patterns) == set(DEFAULT_BLOCKED_PATTERNS)


# -- AllowlistConfig.for_artifact_target: replaying a recorded capability ----


def test_for_artifact_target_forwards_allow_patterns_for_a_non_hand_curated_domain(tmp_path):
    # Real gap found live: recording a capability with --allow-blocked at
    # *recording* time (scripts/record_capability.py) does not by itself
    # make the resulting artifact replayable -- run_single_replay.py derives
    # its own, independent allowlist at replay time, which previously had no
    # way to receive the same override at all.
    cfg = AllowlistConfig.for_artifact_target(
        "https://some-other-vendor.example/login/*", hand_curated_path=tmp_path / "absent.json",
        allow_patterns=["login", "admin"],
    )
    assert check_navigation(cfg, "https://some-other-vendor.example/login").allowed
    assert check_navigation(cfg, "https://some-other-vendor.example/admin/users").allowed
    assert not check_navigation(cfg, "https://some-other-vendor.example/checkout").allowed


def test_for_artifact_target_uses_the_committed_curated_config_for_the_bank_target():
    # The one target this build has a reviewed allowlist for: replaying its
    # artifacts needs no --allow-blocked, and the config's own positive route
    # enumeration is the gate -- an unlisted route on the same domain is
    # refused, not merely "not blocked".
    cfg = AllowlistConfig.for_artifact_target("https://vb-bank-demo.vercel.app/*")
    assert cfg.source_path == Path("config/allowlist.json")
    assert check_navigation(cfg, "https://vb-bank-demo.vercel.app/login").allowed
    assert check_navigation(cfg, "https://vb-bank-demo.vercel.app/admin/users").allowed
    assert not check_navigation(cfg, "https://vb-bank-demo.vercel.app/admin/transfer").allowed
    assert not check_navigation(cfg, "https://vb-bank-demo.vercel.app/admin/users/delete").allowed
    assert not check_navigation(cfg, "https://vb-bank-demo.vercel.app/some/unlisted/page").allowed


def test_for_artifact_target_ignores_allow_patterns_for_a_hand_curated_domain(tmp_path):
    # The hand-curated config already declares its own routes explicitly --
    # allow_patterns has nothing to add there and shouldn't change anything.
    hand_curated = tmp_path / "allowlist.json"
    hand_curated.write_text(
        '{"allowed_domains": ["reviewed-vendor.example"], '
        '"allowed_path_patterns": ["/login*"], "blocked_path_patterns": [], "allowed_action_types": ["click"]}'
    )
    cfg = AllowlistConfig.for_artifact_target(
        "https://reviewed-vendor.example/*", hand_curated_path=str(hand_curated), allow_patterns=["admin"]
    )
    assert cfg.allowed_path_patterns == ["/login*"]  # loaded from the file, untouched by allow_patterns
