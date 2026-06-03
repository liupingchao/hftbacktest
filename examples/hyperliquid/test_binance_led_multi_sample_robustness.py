from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import binance_led_multi_sample_robustness as runner


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _make_pricing_signal_dir(base: Path, sample_id: str, sign: float) -> Path:
    out = base / sample_id
    rows: list[dict[str, object]] = []
    for index in range(80):
        z = -1.5 if index < 40 else 1.5
        label = sign * z * 2.0
        for horizon in [100, 250]:
            row: dict[str, object] = {
                "sample_id": sample_id,
                "source_row_index": index,
                "future_row_index": index + 1,
                "hyperliquid_decision_ts": 1_000_000_000 + index * 500_000_000,
                "binance_local_ts": 1_000_000_000 + index * 500_000_000 - 1_000_000,
                "binance_source_age_ms": 1,
                "joined_row_quality": "primary_usable",
                "horizon_ms": horizon,
                "future_hyperliquid_decision_ts": 1_000_000_000 + (index + 1) * 500_000_000,
                "effective_future_age_ms": 500 if horizon in {100, 250} else horizon,
                "label_row_quality": "primary_label_available",
                "timestamp_policy": "inputs_at_decision_ts_future_labels_at_or_after_target",
                "trade_pressure_policy": "disabled_unverified_side_semantics",
                "basis_contract_caveat": "diagnostic_only",
                "context_hyperliquid_context_quality": "primary_usable",
                "context_hyperliquid_join_age_bucket": "fresh_0_50ms",
                "context_hyperliquid_spread_ticks": 10,
                "hyperliquid_future_mid_move_ticks": label,
                "hyperliquid_future_microprice_minus_mid_change_ticks": label / 2,
                "hyperliquid_future_top5_imbalance_change": label / 100,
                "basis_future_mid_response_ticks": label,
                "basis_future_microprice_response_ticks": label,
            }
            for feature in runner.ALLOWLIST:
                row[f"input_{feature}"] = z
                row[f"input_{feature}_z"] = z
            rows.append(row)
    fieldnames = list(rows[0])
    _write_csv(out / "pricing_signal_rows.csv", rows, fieldnames)
    _write_csv(
        out / "horizon_label_summary.csv",
        [
            {
                "horizon_ms": 100,
                "label": "hyperliquid_future_mid_move_ticks",
                "row_count": 80,
                "effective_future_age_ms_min": 500,
                "effective_future_age_ms_mean": 500,
                "effective_future_age_ms_max": 500,
            }
        ],
        [
            "horizon_ms",
            "label",
            "row_count",
            "effective_future_age_ms_min",
            "effective_future_age_ms_mean",
            "effective_future_age_ms_max",
        ],
    )
    _write_csv(
        out / "venue_state_conditioning_summary.csv",
        [
            {
                "horizon_ms": 100,
                "hyperliquid_context_quality": "primary_usable",
                "hyperliquid_join_age_bucket": "fresh_0_50ms",
                "hyperliquid_spread_bucket": "spread_0_10_ticks",
                "row_count": 80,
                "mean_future_mid_move_ticks": sign,
            }
        ],
        [
            "horizon_ms",
            "hyperliquid_context_quality",
            "hyperliquid_join_age_bucket",
            "hyperliquid_spread_bucket",
            "row_count",
            "mean_future_mid_move_ticks",
        ],
    )
    manifest = {
        "schema_version": "binance_led_hyperliquid_pricing_signal_v1",
        "task_id": "test",
        "input_dirs": {
            "join_dir": str(base / f"{sample_id}_join"),
            "analysis_dir": str(base / f"{sample_id}_analysis"),
            "contract_dir": str(base / "contract"),
        },
        "row_counts": {
            "input_rows": 80,
            "primary_rows": 80,
            "excluded_rows": 0,
            "pricing_signal_rows": len(rows),
        },
        "quality": {
            "recommendation": "keep_for_read_only_research",
            "single_public_sample_caveat": True,
            "source_join_quality": {
                "future_join_count": 0,
                "missing_binance_join_count": 0,
                "primary_usable_row_count": 80,
            },
        },
        "boundary_flags": runner.BOUNDARY_FLAGS,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


def test_build_from_existing_single_sample_marks_need_more_public_samples(tmp_path: Path) -> None:
    output_dir = tmp_path / "aggregate"
    result = runner.build_multi_sample_robustness_artifacts(
        pricing_signal_dirs=[runner.DEFAULT_PRICING_SIGNAL_DIRS[0]],
        output_dir=output_dir,
    )
    manifest = result["run_manifest"]

    assert manifest["sample_count"] == 1
    assert manifest["quality"]["recommendation"] == "needs_more_public_samples"
    assert manifest["quality"]["needs_additional_public_samples"] is True
    assert manifest["boundary_flags"]["no_order_endpoints"] is True
    assert manifest["boundary_flags"]["no_strategy_implementation"] is True
    for name in [
        "multi_sample_manifest.json",
        "sample_quality_matrix.csv",
        "feature_horizon_stability_across_samples.csv",
        "effective_horizon_aliasing_by_sample.csv",
        "venue_state_conditioning_across_samples.csv",
        "pricing_signal_robustness_recommendation.md",
    ]:
        assert (output_dir / name).exists()

    sample_quality = _read_csv(output_dir / "sample_quality_matrix.csv")
    assert sample_quality[0]["sample_id"] == "cross_exchange_public_sample_0602T001"


def test_synthetic_three_sample_stable_recommends_read_only_refinement(tmp_path: Path) -> None:
    dirs = [_make_pricing_signal_dir(tmp_path, f"sample_{idx}", sign=1.0) for idx in range(3)]
    result = runner.build_multi_sample_robustness_artifacts(
        pricing_signal_dirs=dirs,
        output_dir=tmp_path / "aggregate",
    )

    assert result["run_manifest"]["quality"]["recommendation"] == "continue_read_only_runner_refinement"
    stability = _read_csv(tmp_path / "aggregate" / "feature_horizon_stability_across_samples.csv")
    core = [
        row
        for row in stability
        if row["feature"] == "binance_top5_imbalance"
        and row["horizon_ms"] == "100"
        and row["label"] == "hyperliquid_future_mid_move_ticks"
    ][0]
    assert core["stability_verdict"] == "stable_across_samples"
    assert core["eligible_sample_count"] == "3"


def test_synthetic_mixed_direction_rejects_runner_design(tmp_path: Path) -> None:
    dirs = [
        _make_pricing_signal_dir(tmp_path, "sample_pos", sign=1.0),
        _make_pricing_signal_dir(tmp_path, "sample_neg_a", sign=-1.0),
        _make_pricing_signal_dir(tmp_path, "sample_neg_b", sign=-1.0),
    ]
    result = runner.build_multi_sample_robustness_artifacts(
        pricing_signal_dirs=dirs,
        output_dir=tmp_path / "aggregate",
    )

    assert result["run_manifest"]["quality"]["recommendation"] == "reject_for_runner_design"
