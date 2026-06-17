from __future__ import annotations

import csv
import json
from pathlib import Path

from examples.hyperliquid.hyperliquid_tiny_live_signal_quote_replay import run


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_signal_quote_replay_outputs_expected_artifacts(tmp_path: Path) -> None:
    pricing_path = tmp_path / "pricing_signal_rows.csv"
    row_level_path = tmp_path / "row_level_read_only_cases.csv"
    source_manifest_path = tmp_path / "source_artifact_manifest.csv"
    output_dir = tmp_path / "out"

    pricing_fields = [
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
    ]
    _write_csv(
        pricing_path,
        [
            {
                "sample_id": "s1",
                "source_row_index": "1",
                "hyperliquid_decision_ts": "100",
                "joined_row_quality": "primary_usable",
                "label_row_quality": "primary_label_available",
                "context_hyperliquid_context_quality": "primary_usable",
                "context_hyperliquid_join_age_bucket": "fresh_0_50ms",
                "binance_source_age_ms": "10",
                "context_basis_mid_ticks": "25",
                "context_hyperliquid_mid_px": "100",
                "context_hyperliquid_spread_ticks": "10",
                "horizon_ms": "1000",
            },
            {
                "sample_id": "s1",
                "source_row_index": "2",
                "hyperliquid_decision_ts": "200",
                "joined_row_quality": "primary_usable",
                "label_row_quality": "primary_label_available",
                "context_hyperliquid_context_quality": "primary_usable",
                "context_hyperliquid_join_age_bucket": "fresh_0_50ms",
                "binance_source_age_ms": "10",
                "context_basis_mid_ticks": "-25",
                "context_hyperliquid_mid_px": "101",
                "context_hyperliquid_spread_ticks": "10",
                "horizon_ms": "1000",
            },
        ],
        pricing_fields,
    )
    _write_csv(
        row_level_path,
        [
            {"source_sample_id": "s_hist", "context_basis_mid_ticks": "25"},
            {"source_sample_id": "s_hist", "context_basis_mid_ticks": "50"},
        ],
        ["source_sample_id", "context_basis_mid_ticks"],
    )
    _write_csv(
        source_manifest_path,
        [
            {
                "sample_id": "s_hist",
                "source_artifact_path": "/missing/pricing_signal_rows.csv",
                "source_row_count": "2",
            }
        ],
        ["sample_id", "source_artifact_path", "source_row_count"],
    )

    run(pricing_path, row_level_path, source_manifest_path, output_dir)

    manifest = json.loads((output_dir / "replay_manifest.json").read_text(encoding="utf-8"))
    assert manifest["boundary_flags"]["order_placement_called"] is False
    assert manifest["final_recommendation"] == "hyperliquid_tiny_live_signal_quote_replay_needs_threshold_calibration"

    threshold_rows = list(csv.DictReader((output_dir / "threshold_sensitivity.csv").open(newline="", encoding="utf-8")))
    assert any(row["intent_buy"] == "1" for row in threshold_rows)
    assert any(row["intent_sell"] == "1" for row in threshold_rows)

    row_level_rows = list(csv.DictReader((output_dir / "row_level_basis_distribution_by_sample.csv").open(newline="", encoding="utf-8")))
    assert row_level_rows[0]["sample_id"] == "s_hist"
