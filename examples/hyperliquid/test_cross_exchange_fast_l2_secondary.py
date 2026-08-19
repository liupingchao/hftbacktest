import numpy as np
import polars as pl

from examples.hyperliquid.cross_exchange_fast_l2_secondary import (
    PRICE_SCALE,
    _depth_and_replenishment,
    _forward_window_extreme,
    _secondary_eligible_column,
    attach_secondary_labels,
)


def test_forward_window_extreme_uses_strict_open_left_boundary():
    query = np.asarray([100, 200], dtype=np.int64)
    events = np.asarray([100, 150, 250], dtype=np.int64)
    values = np.asarray([10, 20, 30], dtype=np.int64)
    maximum = _forward_window_extreme(query, events, values, 100, True)
    minimum = _forward_window_extreme(query, events, values, 100, False)
    assert maximum.tolist() == [20.0, 30.0]
    assert minimum.tolist() == [20.0, 30.0]


def test_secondary_eligibility_column_matches_label_contract():
    assert (
        _secondary_eligible_column("d_bh", 100, "depth_depletion_fraction")
        == "d_bh_h100_depth_depletion_eligible"
    )
    assert (
        _secondary_eligible_column("d_hb", 500, "trade_arrival")
        == "d_hb_h500_trade_arrival_eligible"
    )


def test_depth_zero_and_replenishment_failure_are_observable():
    decision = np.asarray([100], dtype=np.int64)
    anchor = np.asarray([100 * PRICE_SCALE], dtype=np.int64)
    snapshot_ts = np.asarray([50, 150, 250], dtype=np.int64)
    prices = np.asarray(
        [
            [100, 101, 102, 103, 104],
            [101, 102, 103, 104, 105],
            [101, 102, 103, 104, 105],
        ],
        dtype=np.int64,
    ) * PRICE_SCALE
    quantities = np.ones((3, 5), dtype=np.float64)
    depth, replenishment, anchor_ok, depth_ok, replenishment_ok = (
        _depth_and_replenishment(
            decision,
            anchor,
            snapshot_ts,
            prices,
            quantities,
            200,
            True,
        )
    )
    assert anchor_ok.tolist() == [True]
    assert depth_ok.tolist() == [True]
    assert depth.tolist() == [1.0]
    assert replenishment_ok.tolist() == [True]
    assert replenishment.tolist() == [1.0]


def test_anchor_beyond_retained_depth_is_unobserved():
    decision = np.asarray([100], dtype=np.int64)
    anchor = np.asarray([110 * PRICE_SCALE], dtype=np.int64)
    snapshot_ts = np.asarray([50, 150], dtype=np.int64)
    prices = np.asarray(
        [[100, 101, 102, 103, 104], [100, 101, 102, 103, 104]],
        dtype=np.int64,
    ) * PRICE_SCALE
    quantities = np.ones((2, 5), dtype=np.float64)
    depth, _, anchor_ok, depth_ok, _ = _depth_and_replenishment(
        decision,
        anchor,
        snapshot_ts,
        prices,
        quantities,
        100,
        True,
    )
    assert anchor_ok.tolist() == [False]
    assert depth_ok.tolist() == [False]
    assert np.isnan(depth[0])


def test_attach_secondary_labels_respects_trade_side_and_price():
    frame = pl.DataFrame(
        {
            "decision_ts_ns": [100],
            "hyperliquid_ask_q": [100.0],
            "hyperliquid_bid_q": [99.0],
        }
    )
    fast = {
        "snapshot_ts": np.asarray([50, 150], dtype=np.int64),
        "bid_px": np.asarray([[99, 98, 97, 96, 95], [99, 98, 97, 96, 95]])
        * PRICE_SCALE,
        "bid_qty": np.ones((2, 5)),
        "ask_px": np.asarray(
            [[100, 101, 102, 103, 104], [100, 101, 102, 103, 104]]
        )
        * PRICE_SCALE,
        "ask_qty": np.ones((2, 5)),
        "trade_ts": np.asarray([120, 130], dtype=np.int64),
        "trade_px": np.asarray([100, 99], dtype=np.int64) * PRICE_SCALE,
        "trade_qty": np.asarray([1.0, 1.0]),
        "trade_side": np.asarray([1, -1], dtype=np.int8),
    }
    result = attach_secondary_labels(frame, fast)
    assert result["d_bh_h100_trade_arrival"][0] == 1.0
    assert result["d_hb_h100_trade_arrival"][0] == 1.0
