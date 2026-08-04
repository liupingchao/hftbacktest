from __future__ import annotations

import ast
import inspect
import subprocess
from pathlib import Path

import numpy as np
from hftbacktest import BUY_EVENT, DEPTH_EVENT, SELL_EVENT

from examples.tutorial_reproduction import notebook_support
from examples.tutorial_reproduction.market_queue_experiments import (
    MARKET_SPECS,
    PreparedMarket,
    _align_equity_series,
    _causal_queue_signals,
    _equity_curve,
    _initial_mid_price_from_data,
    _normalize_flow_quantities,
    _order_qty,
    _prepare_external_market,
    _price_bounds,
    _queue_signal_strategy,
    _queue_signal_values,
    _short_horizon_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_NOTEBOOKS = (
    "Making Multiple Markets.ipynb",
    "Making Multiple Markets - Introduction.ipynb",
    "Probability Queue Models.ipynb",
    "Queue-Based Market Making in Large Tick Size Assets.ipynb",
    "High-Frequency Grid Trading - Comparison Across Other Exchanges.ipynb",
)


def test_market_queue_notebooks_are_copies_assigned_to_0804t005() -> None:
    slugs = [slug for slug, _, _ in notebook_support.EXPERIMENTS[16:]]

    assert len(slugs) == 5
    assert {notebook_support._TASK_BY_SLUG[slug] for slug in slugs} == {"0804T005"}
    for slug in slugs:
        target = notebook_support._NOTEBOOK_PATH_BY_SLUG[slug]
        assert target.parts[:3] == (
            "tutorial_reproduction",
            "notebooks",
            "0804T005",
        )


def test_original_notebooks_have_no_worktree_or_index_diff() -> None:
    for cached in (False, True):
        command = ["git", "diff", "--quiet"]
        if cached:
            command.append("--cached")
        command.extend(["--", *[f"examples/{name}" for name in ORIGINAL_NOTEBOOKS]])
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def test_market_specs_cover_five_distinct_real_mounts() -> None:
    assert len(MARKET_SPECS) == 5
    assert len({spec.venue for spec in MARKET_SPECS}) == 5
    assert {spec.venue for spec in MARKET_SPECS} == {
        "binance-futures",
        "bybit",
        "okex-swap",
        "bitget-futures",
        "gate-io-futures",
    }
    assert all(spec.tick_size > 0 for spec in MARKET_SPECS)
    assert all(spec.lot_size > 0 for spec in MARKET_SPECS)
    assert all(spec.contract_multiplier > 0 for spec in MARKET_SPECS)


def test_external_market_conversion_uses_supported_convert_fuse_arguments() -> None:
    tree = ast.parse(inspect.getsource(_prepare_external_market))
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "convert_fuse"
    )
    keywords = {keyword.arg for keyword in call.keywords}

    assert "buffer_size" not in keywords
    assert "ss_buffer_size" in keywords


def test_queue_signal_helpers_are_point_in_time() -> None:
    bid = np.arange(100.0, 120.0)
    ask = bid + 1.0
    bid_qty = np.arange(1.0, 21.0)
    ask_qty = np.arange(21.0, 1.0, -1.0)
    trades = np.sin(np.arange(20))

    changed_bid_qty = bid_qty.copy()
    changed_ask_qty = ask_qty.copy()
    changed_trades = trades.copy()
    changed_bid_qty[10:] *= 1_000
    changed_ask_qty[10:] *= 2_000
    changed_trades[10:] *= 3_000

    original = _causal_queue_signals(
        bid,
        ask,
        bid_qty,
        ask_qty,
        trades,
        5,
    )
    modified = _causal_queue_signals(
        bid,
        ask,
        changed_bid_qty,
        changed_ask_qty,
        changed_trades,
        5,
    )

    np.testing.assert_allclose(original[:10], modified[:10])
    assert not np.allclose(original[10:], modified[10:])


def test_strategy_and_future_mutation_share_queue_signal_core() -> None:
    for function in (_queue_signal_strategy, _causal_queue_signals):
        source = inspect.getsource(function.py_func if hasattr(function, "py_func") else function)
        tree = ast.parse(source)
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert _queue_signal_values.py_func.__name__ in calls


def test_initial_price_order_qty_and_roi_ignore_future_events() -> None:
    dtype = [("ev", "i8"), ("px", "f8"), ("qty", "f8")]
    data = np.array(
        [
            (DEPTH_EVENT | SELL_EVENT, 101.0, 2.0),
            (DEPTH_EVENT | BUY_EVENT, 100.0, 3.0),
            (DEPTH_EVENT | SELL_EVENT, 102.0, 1.0),
            (DEPTH_EVENT | BUY_EVENT, 99.0, 1.0),
        ],
        dtype=dtype,
    )
    changed = data.copy()
    changed[2:]["px"] *= 100
    original_mid = _initial_mid_price_from_data(data)
    changed_mid = _initial_mid_price_from_data(changed)
    assert original_mid == changed_mid == 100.5

    def market(mid: float) -> PreparedMarket:
        return PreparedMarket(
            venue="test",
            symbol="TEST",
            tick_size=0.1,
            lot_size=0.001,
            contract_multiplier=1.0,
            initial_mid_price=mid,
            fused_npz="unused",
            raw_files={},
            staged_sha256={},
            fused_rows=0,
        )

    assert _order_qty(market(original_mid)) == _order_qty(market(changed_mid))
    assert _price_bounds(market(original_mid)) == _price_bounds(market(changed_mid))


def test_initial_price_ignores_trades_before_complete_depth() -> None:
    dtype = [("ev", "i8"), ("px", "f8"), ("qty", "f8")]
    data = np.array(
        [
            (SELL_EVENT, 100.0, 1.0),
            (BUY_EVENT, 99.0, 1.0),
            (DEPTH_EVENT | SELL_EVENT, 102.0, 2.0),
            (DEPTH_EVENT | BUY_EVENT, 100.0, 3.0),
        ],
        dtype=dtype,
    )

    assert _initial_mid_price_from_data(data) == 101.0


def test_initial_price_applies_depth_deletions() -> None:
    dtype = [("ev", "i8"), ("px", "f8"), ("qty", "f8")]
    data = np.array(
        [
            (DEPTH_EVENT | SELL_EVENT, 101.0, 2.0),
            (DEPTH_EVENT | SELL_EVENT, 101.0, 0.0),
            (DEPTH_EVENT | BUY_EVENT, 100.0, 3.0),
            (DEPTH_EVENT | SELL_EVENT, 102.0, 4.0),
        ],
        dtype=dtype,
    )

    assert _initial_mid_price_from_data(data) == 101.0


def test_equity_alignment_uses_backward_wall_clock_asof() -> None:
    timestamps, aligned = _align_equity_series(
        [
            (
                np.asarray([100, 200, 300, 400], dtype=np.int64),
                np.asarray([0.0, 1.0, 2.0, 3.0]),
            ),
            (
                np.asarray([150, 250, 350, 450], dtype=np.int64),
                np.asarray([10.0, 11.0, 12.0, 13.0]),
            ),
        ],
        interval_ns=100,
    )

    np.testing.assert_array_equal(timestamps, [200, 300, 400])
    np.testing.assert_allclose(
        aligned,
        [[1.0, 10.0], [2.0, 11.0], [3.0, 12.0]],
    )


def test_flow_quantities_are_normalized_to_base_asset() -> None:
    flow = np.zeros((1, 8))
    flow[0, 3:7] = [100_000.0, 50_000.0, 20_000.0, 10_000.0]

    normalized = _normalize_flow_quantities(flow, 0.0001)

    np.testing.assert_allclose(normalized[0, 3:7], [10.0, 5.0, 2.0, 1.0])
    np.testing.assert_allclose(flow[0, 3:7], [100_000.0, 50_000.0, 20_000.0, 10_000.0])


def test_short_horizon_metrics_are_finite_for_nonconstant_curve() -> None:
    metrics = _short_horizon_metrics(np.arange(1_000, dtype=np.float64), 100)

    assert metrics["changes"] > 0
    assert metrics["mean"] > 0
    assert metrics["std"] == 0
    assert metrics["sharpe"] is None


def test_equity_curve_drops_uninitialized_prices() -> None:
    records = np.zeros(
        3,
        dtype=[
            ("timestamp", "i8"),
            ("balance", "f8"),
            ("position", "f8"),
            ("price", "f8"),
            ("fee", "f8"),
        ],
    )
    records["timestamp"] = [100, 200, 300]
    records["price"] = [np.nan, 100.0, 101.0]
    records["position"] = 1.0
    market = PreparedMarket(
        venue="test",
        symbol="TEST",
        tick_size=0.1,
        lot_size=1.0,
        contract_multiplier=1.0,
        initial_mid_price=100.0,
        fused_npz="unused",
        raw_files={},
        staged_sha256={},
        fused_rows=0,
    )

    np.testing.assert_allclose(_equity_curve(records, market), [100.0, 101.0])
