from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("cross_exchange_sample_expansion.py")
SPEC = importlib.util.spec_from_file_location("cross_exchange_sample_expansion", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _raw(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as fh:
        fh.write(payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_sample(
    root: Path,
    sample_id: str,
    start_minute: int,
    rv: float,
    *,
    effective_future_age_ms: int = 1000,
) -> None:
    public = root / f"cross_exchange_public_sample_{sample_id}"
    join = root / f"cross_exchange_lead_lag_join_{sample_id}"
    pricing = root / f"binance_led_hyperliquid_pricing_signal_{sample_id}"
    binance_raw = public / "binance_public_raw" / "raw.gz"
    hl_raw = public / "hyperliquid_public_sample" / "raw.gz"
    binance_sha = _raw(binance_raw, f"binance-{sample_id}".encode())
    hl_sha = _raw(hl_raw, f"hl-{sample_id}".encode())
    start = f"2026-06-25T{14 + start_minute // 60:02d}:{start_minute % 60:02d}:00+00:00"
    _write_json(
        public / "sample_manifest.json",
        {
            "requested_duration_seconds": 1800,
            "git_commit": "fixture",
            "overlap": {"overlap_seconds": 1800},
        },
    )
    _write_json(
        public / "binance_public_raw" / "collection_manifest.json",
        {
            "actual_duration_seconds": 1800,
            "local_start_time": start,
            "message_count_by_event_type": {
                "bookTicker": 100,
                "depthUpdate": 100,
                "trade": 100 + int(rv * 10),
            },
            "reconnect_count": 0,
            "raw_sha256": binance_sha,
        },
    )
    _write_json(
        public / "hyperliquid_public_sample" / "collection_manifest.json",
        {
            "actual_duration_seconds": 1800,
            "message_count_by_channel": {"l2Book": 100, "trades": 100},
            "reconnect_count": 0,
            "raw_sha256": hl_sha,
        },
    )
    _write_json(
        public / "binance_alignment" / "metrics.json",
        {"top5_row_count": 100},
    )
    _write_json(
        public / "hyperliquid_public_sample" / "alignment" / "metrics.json",
        {
            "topn_row_count": 100,
            "decision_row_count": 30,
            "trade_event_count": 100,
        },
    )
    joined = []
    for index in range(50):
        joined.append(
            {
                "joined_row_quality": "primary_usable",
                "cross_exchange_future_join": "0",
                "cross_exchange_missing_binance_join": "0",
                "hyperliquid_decision_ts": 1_000_000_000 + index * 500_000_000,
                "hyperliquid_l2book_local_ts": 1_000_000_000 + index * 500_000_000,
                "hyperliquid_l2book_event_ts": 999_000_000 + index * 500_000_000,
                "binance_local_ts": 990_000_000 + index * 500_000_000,
                "binance_exch_ts": 980_000_000 + index * 500_000_000,
                "binance_source_age_ms": 10,
                "hyperliquid_join_age_ms": 0,
                "hyperliquid_best_bid_px": 100,
                "hyperliquid_best_ask_px": 100.1,
                "hyperliquid_mid_px": 100.05 + index * 0.01,
                "hyperliquid_top5_microprice_px": 100.04 + index * 0.01,
                "hyperliquid_spread_ticks": 1,
                "hyperliquid_bid_topn_px": "100;99.9;99.8;99.7;99.6",
                "hyperliquid_ask_topn_px": "100.1;100.2;100.3;100.4;100.5",
                "hyperliquid_bid_topn_qtys": "1;1;1;1;1",
                "hyperliquid_ask_topn_qtys": "1;1;1;1;1",
                "hyperliquid_top5_bid_qty": 5,
                "hyperliquid_top5_ask_qty": 5,
                "binance_mid_px": 100,
                "binance_top5_microprice_px": 100.01,
                "binance_bid_top5_px": "99.9;99.8;99.7;99.6;99.5",
                "binance_ask_top5_px": "100.1;100.2;100.3;100.4;100.5",
                "binance_bid_top5_qtys": "1;1;1;1;1",
                "binance_ask_top5_qtys": "1;1;1;1;1",
                "binance_rolling_rv_ticks_20": rv,
                "basis_mid_ticks": 0.5,
                "hyperliquid_top5_imbalance": 0.1,
                "hyperliquid_microprice_minus_mid_ticks": -0.1,
                "hyperliquid_context_quality": "primary_usable",
            }
        )
    _write_csv(join / "cross_exchange_joined_features.csv", joined)
    _write_json(
        join / "join_quality_summary.json",
        {
            "cross_exchange_join": {
                "future_join_count": 0,
                "missing_binance_join_count": 0,
                "stale_binance_source_count": 0,
            }
        },
    )
    pricing_rows = []
    for index in range(40):
        pricing_rows.append(
            {
                "horizon_ms": 1000,
                "source_row_index": index,
                "future_row_index": index + 2,
                "future_hyperliquid_decision_ts": 2_000_000_000 + index * 500_000_000,
                "effective_future_age_ms": effective_future_age_ms,
                "input_binance_top5_imbalance": 0.1,
                "input_binance_microprice_minus_mid_ticks": 0.2,
                "input_binance_mid_move_ticks_from_prev": 0,
                "input_binance_top5_bid_qty": 5,
                "hyperliquid_future_mid_move_ticks": 1,
                "hyperliquid_future_microprice_minus_mid_change_ticks": 0.1,
                "label_row_quality": "primary_label_available",
            }
        )
    _write_csv(pricing / "pricing_signal_rows.csv", pricing_rows)
    _write_json(pricing / "run_manifest.json", {"row_counts": {"pricing_signal_rows": 25}})


def test_build_artifacts_preserves_symmetric_touches_and_unlocks_t003(
    tmp_path: Path,
) -> None:
    root = tmp_path / "analysis"
    ids = ["xemm_0625_t002_a", "xemm_0625_t002_b", "xemm_0625_t002_c"]
    _fixture_sample(root, ids[0], 0, 1)
    _fixture_sample(root, ids[1], 31, 2)
    _fixture_sample(root, ids[2], 62, 3)
    output = tmp_path / "output"
    result = MODULE.build_artifacts(
        analysis_root=root, sample_ids=ids, output_dir=output
    )

    assert result["manifest"]["recommendation"] == (
        "sample_contract_ready_for_signal_acceptance"
    )
    assert result["manifest"]["t003_creation_unlocked"] is True
    assert result["manifest"]["complete_symmetric_context_row_count"] == 120
    assert result["manifest"]["valid_for_1000ms_signal_acceptance_row_count"] == 120
    assert result["manifest"]["effective_horizon_valid_for_1000ms_signal_acceptance"] is True
    assert len({row["observed_regime"] for row in result["regime_rows"]}) == 3
    assert all(
        row["hyperliquid_buy_touch_quote_px"] == row["hyperliquid_current_bid_px"]
        and row["hyperliquid_sell_touch_quote_px"] == row["hyperliquid_current_ask_px"]
        for row in result["context_rows"]
    )
    assert all(row["complete_context"] for row in result["context_rows"])
    assert all(row["valid_for_1000ms_signal_acceptance"] for row in result["context_rows"])
    assert (output / "sample_expansion_manifest.json").exists()
    assert (output / "boundary_manifest.json").exists()
    assert (output / "effective_horizon_validity_matrix.csv").exists()


def test_build_artifacts_blocks_t003_when_effective_horizon_is_late(
    tmp_path: Path,
) -> None:
    root = tmp_path / "analysis"
    ids = ["xemm_0625_t002_a", "xemm_0625_t002_b", "xemm_0625_t002_c"]
    _fixture_sample(root, ids[0], 0, 1, effective_future_age_ms=5000)
    _fixture_sample(root, ids[1], 31, 2, effective_future_age_ms=5000)
    _fixture_sample(root, ids[2], 62, 3, effective_future_age_ms=5000)
    result = MODULE.build_artifacts(
        analysis_root=root, sample_ids=ids, output_dir=tmp_path / "output"
    )

    assert result["manifest"]["recommendation"] == "needs_more_public_samples"
    assert result["manifest"]["t003_creation_unlocked"] is False
    assert result["manifest"]["complete_symmetric_context_row_count"] == 120
    assert result["manifest"]["valid_for_1000ms_signal_acceptance_row_count"] == 0
    assert result["manifest"]["effective_horizon_gate_reason"] == (
        "1000ms_near_target_label_coverage_insufficient"
    )
    assert all(row["complete_context"] for row in result["context_rows"])
    assert not any(
        row["valid_for_1000ms_signal_acceptance"] for row in result["context_rows"]
    )
