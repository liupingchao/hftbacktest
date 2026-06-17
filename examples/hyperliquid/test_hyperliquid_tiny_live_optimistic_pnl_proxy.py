from __future__ import annotations

import csv
import json
from pathlib import Path

from examples.hyperliquid.hyperliquid_tiny_live_optimistic_pnl_proxy import FINAL_RECOMMENDATION, run


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_optimistic_pnl_proxy_outputs_reconciled_upper_bound(tmp_path: Path) -> None:
    primary_rows = tmp_path / "primary_pricing_signal_rows.csv"
    canonical_rows = tmp_path / "canonical_pricing_signal_rows.csv"
    replay_manifest = tmp_path / "replay_manifest.json"
    source_manifest = tmp_path / "source_artifact_manifest.csv"
    output_dir = tmp_path / "out"

    fields = [
        "sample_id",
        "source_row_index",
        "hyperliquid_decision_ts",
        "joined_row_quality",
        "label_row_quality",
        "context_hyperliquid_context_quality",
        "context_hyperliquid_join_age_bucket",
        "binance_source_age_ms",
        "context_basis_mid_ticks",
        "context_hyperliquid_mid_px",
        "context_hyperliquid_spread_ticks",
        "horizon_ms",
        "hyperliquid_future_mid_move_ticks",
        "basis_future_mid_response_ticks",
    ]
    _write_csv(
        primary_rows,
        [
            {
                "sample_id": "primary_sample",
                "source_row_index": "1",
                "hyperliquid_decision_ts": "100",
                "joined_row_quality": "primary_usable",
                "label_row_quality": "primary_label_available",
                "context_hyperliquid_context_quality": "primary_usable",
                "context_hyperliquid_join_age_bucket": "fresh_0_50ms",
                "binance_source_age_ms": "10",
                "context_basis_mid_ticks": "80",
                "context_hyperliquid_mid_px": "100",
                "context_hyperliquid_spread_ticks": "10",
                "horizon_ms": "100",
                "hyperliquid_future_mid_move_ticks": "2",
                "basis_future_mid_response_ticks": "-3",
            },
            {
                "sample_id": "primary_sample",
                "source_row_index": "1",
                "hyperliquid_decision_ts": "100",
                "joined_row_quality": "primary_usable",
                "label_row_quality": "primary_label_available",
                "context_hyperliquid_context_quality": "primary_usable",
                "context_hyperliquid_join_age_bucket": "fresh_0_50ms",
                "binance_source_age_ms": "10",
                "context_basis_mid_ticks": "80",
                "context_hyperliquid_mid_px": "100",
                "context_hyperliquid_spread_ticks": "10",
                "horizon_ms": "1000",
                "hyperliquid_future_mid_move_ticks": "1",
                "basis_future_mid_response_ticks": "-2",
            },
        ],
        fields,
    )
    _write_csv(
        canonical_rows,
        [
            {
                "sample_id": "canonical_sample",
                "source_row_index": "1",
                "hyperliquid_decision_ts": "100",
                "joined_row_quality": "primary_usable",
                "label_row_quality": "primary_label_available",
                "context_hyperliquid_context_quality": "primary_usable",
                "context_hyperliquid_join_age_bucket": "fresh_0_50ms",
                "binance_source_age_ms": "10",
                "context_basis_mid_ticks": "-80",
                "context_hyperliquid_mid_px": "100",
                "context_hyperliquid_spread_ticks": "10",
                "horizon_ms": "1000",
                "hyperliquid_future_mid_move_ticks": "-3",
                "basis_future_mid_response_ticks": "2",
            }
        ],
        fields,
    )
    replay_manifest.write_text(
        json.dumps(
            {
                "pricing_rows_inputs": [str(primary_rows), str(canonical_rows)],
                "pricing_rows_input_count": 2,
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        source_manifest,
        [{"sample_id": "canonical_sample", "source_artifact_path": str(canonical_rows), "source_row_count": "1"}],
        ["sample_id", "source_artifact_path", "source_row_count"],
    )

    manifest = run(replay_manifest, source_manifest, output_dir)

    assert manifest["final_recommendation"] == FINAL_RECOMMENDATION
    assert manifest["boundary_flags"]["order_placement_called"] is False
    assert manifest["boundary_flags"]["real_pnl_claimed"] is False
    assert manifest["sample_sets"]["requested_six"]["compute_status"] == "not_computed"
    assert manifest["sample_sets"]["0617T005_8_input"]["compute_status"] == "computed"

    reconciliation = list(csv.DictReader((output_dir / "sample_set_reconciliation.csv").open(newline="", encoding="utf-8")))
    assert any(row["sample_set_id"] == "requested_six" and row["reconciliation_status"] == "needs_input_clarification" for row in reconciliation)

    fixed = list(csv.DictReader((output_dir / "fixed_horizon_pnl_summary.csv").open(newline="", encoding="utf-8")))
    buy_row = next(
        row
        for row in fixed
        if row["sample_set_id"] == "0617T005_8_input"
        and row["sample_id"] == "primary_sample"
        and row["quote_side"] == "buy"
        and row["threshold_ticks"] == "75"
        and row["persistence_count"] == "1"
        and row["horizon_ms"] == "100"
    )
    assert buy_row["mean_optimistic_mid_pnl_ticks"] == "7"
    assert buy_row["total_optimistic_mid_pnl_usdc"] == "0.007"

    oracle = list(csv.DictReader((output_dir / "oracle_best_horizon_summary.csv").open(newline="", encoding="utf-8")))
    assert any(row["oracle_label"] == "non_tradeable_oracle_upper_bound" for row in oracle)

    audit = list(csv.DictReader((output_dir / "row_level_audit_sample.csv").open(newline="", encoding="utf-8")))
    assert audit[0]["proof_boundary"] == "optimistic_proxy_not_real_pnl_not_real_fill"
