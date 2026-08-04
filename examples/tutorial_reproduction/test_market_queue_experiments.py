from __future__ import annotations

import ast
import inspect
import subprocess
from pathlib import Path

import numpy as np

from examples.tutorial_reproduction import notebook_support
from examples.tutorial_reproduction.market_queue_experiments import (
    MARKET_SPECS,
    _prepare_external_market,
    _causal_queue_signals,
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


def test_short_horizon_metrics_are_finite_for_nonconstant_curve() -> None:
    metrics = _short_horizon_metrics(np.arange(1_000, dtype=np.float64), 100)

    assert metrics["changes"] > 0
    assert metrics["mean"] > 0
    assert metrics["std"] == 0
    assert metrics["sharpe"] is None
