from __future__ import annotations

from quote_anchor_safety import (
    QuoteAnchorSafetyConfig,
    apply_quote_anchor_safety,
)


def test_quote_anchor_safety_disabled_preserves_ticks() -> None:
    result = apply_quote_anchor_safety(
        cfg=QuoteAnchorSafetyConfig(enabled=False),
        target_bid_tick=1002,
        target_ask_tick=1003,
        target_bid_price=100.29,
        target_ask_price=100.21,
        tick_size=0.1,
        fast_bid_tick=1000,
        fast_ask_tick=1004,
        fast_anchor_age_ms=1.0,
    )

    assert result.safe_bid_tick == 1002
    assert result.safe_ask_tick == 1003
    assert result.anchor_source == "disabled"
    assert not result.suppress_buy
    assert not result.suppress_sell


def test_quote_anchor_safety_uses_bookticker_clamp_and_recheck() -> None:
    result = apply_quote_anchor_safety(
        cfg=QuoteAnchorSafetyConfig(enabled=True),
        target_bid_tick=1003,
        target_ask_tick=1000,
        target_bid_price=100.39,
        target_ask_price=100.01,
        tick_size=0.1,
        fast_bid_tick=1001,
        fast_ask_tick=1002,
        fast_anchor_age_ms=3.0,
        depth_bid_tick=1000,
        depth_ask_tick=1004,
        depth_anchor_age_ms=2.0,
    )

    assert result.anchor_source == "bookticker"
    assert result.safe_bid_tick == 1001
    assert result.safe_ask_tick == 1002
    assert result.bid_clamped
    assert result.ask_clamped
    assert not result.depth_fallback_used
    assert not result.post_only_risk_after_recheck


def test_quote_anchor_safety_guarded_depth_fallback_when_fast_stale() -> None:
    result = apply_quote_anchor_safety(
        cfg=QuoteAnchorSafetyConfig(enabled=True, max_fast_anchor_age_ms=10.0, max_depth_fallback_age_ms=10.0),
        target_bid_tick=1003,
        target_ask_tick=1004,
        tick_size=0.1,
        fast_bid_tick=1001,
        fast_ask_tick=1002,
        fast_anchor_age_ms=250.0,
        depth_bid_tick=1000,
        depth_ask_tick=1005,
        depth_anchor_age_ms=2.0,
    )

    assert result.anchor_source == "depth_guarded_fallback"
    assert result.depth_fallback_used
    assert result.safe_bid_tick == 1000
    assert result.safe_ask_tick == 1005
    assert not result.suppress_buy
    assert not result.suppress_sell


def test_quote_anchor_safety_suppresses_when_no_fresh_anchor() -> None:
    result = apply_quote_anchor_safety(
        cfg=QuoteAnchorSafetyConfig(enabled=True, max_fast_anchor_age_ms=10.0, max_depth_fallback_age_ms=10.0),
        target_bid_tick=1000,
        target_ask_tick=1002,
        tick_size=0.1,
        fast_bid_tick=1000,
        fast_ask_tick=1002,
        fast_anchor_age_ms=500.0,
        depth_bid_tick=999,
        depth_ask_tick=1003,
        depth_anchor_age_ms=500.0,
    )

    assert result.anchor_source == "stale_anchor"
    assert result.stale_anchor
    assert result.suppress_buy
    assert result.suppress_sell
    assert not result.post_only_risk_after_recheck
