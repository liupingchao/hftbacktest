from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import canonical_event_mode_evidence as canonical_loader
import canonical_signal_quality_ranking as ranking


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _sample(sample_id: str, *, canonical: bool = True) -> dict[str, object]:
    decision_mode = "event" if canonical else "synthetic_fixed_grid"
    canonical_status = (
        canonical_loader.CANONICAL_EVENT_STATUS
        if canonical
        else canonical_loader.DIAGNOSTIC_SYNTHETIC_STATUS
    )
    return {
        "sample_id": sample_id,
        "decision_mode": decision_mode,
        "canonical_status": canonical_status,
        "pricing_signal_dir": f"/tmp/{sample_id}/pricing",
        "source_sample_dir": f"/tmp/{sample_id}/source",
        "source_join_dir": f"/tmp/{sample_id}/join",
        "source_analysis_dir": f"/tmp/{sample_id}/analysis",
        "alignment_run_manifest": f"/tmp/{sample_id}/alignment/run_manifest.json",
        "alignment_metrics": f"/tmp/{sample_id}/alignment/metrics.json",
        "event_decision_count": 10 if canonical else 0,
        "synthetic_decision_count": 0 if canonical else 10,
        "run_manifest": f"/tmp/{sample_id}/pricing/run_manifest.json",
        "pricing_signal_rows": f"/tmp/{sample_id}/pricing/pricing_signal_rows.csv",
    }


def _quality_row(sample: dict[str, object]) -> dict[str, object]:
    return {
        "sample_id": sample["sample_id"],
        "pricing_signal_dir": sample["pricing_signal_dir"],
        "source_join_dir": sample["source_join_dir"],
        "source_analysis_dir": sample["source_analysis_dir"],
        "source_sample_dir": sample["source_sample_dir"],
        "decision_mode": sample["decision_mode"],
        "canonical_status": sample["canonical_status"],
        "alignment_run_manifest": sample["alignment_run_manifest"],
        "alignment_metrics": sample["alignment_metrics"],
        "event_decision_count": sample["event_decision_count"],
        "synthetic_decision_count": sample["synthetic_decision_count"],
        "input_rows": 10,
        "primary_rows": 10,
        "excluded_rows": 0,
        "pricing_signal_rows": 20,
        "future_join_count": 0,
        "missing_binance_join_count": 0,
        "primary_usable_row_count": 10,
        "effective_future_age_ms_min": 100,
        "effective_future_age_ms_mean": 500,
        "effective_future_age_ms_max": 1000,
        "horizon_count": 4,
        "independent_future_row_delta_count": 4,
        "horizon_future_row_delta_groups": "100:1|250:1|500:2,3|1000:3,4",
        "aliased_horizon_signature_count": 0,
        "pricing_signal_recommendation": "keep_for_read_only_research",
        "single_public_sample_caveat": "True",
        "quality_status": sample["canonical_status"],
    }


FEATURE_FIELDS = [
    "feature",
    "horizon_ms",
    "label",
    "sample_count",
    "eligible_sample_count",
    "canonical_eligible_sample_count",
    "canonical_independent_future_row_delta_count",
    "majority_direction",
    "direction_consistency_ratio",
    "positive_sample_count",
    "negative_sample_count",
    "zero_sample_count",
    "diagnostic_synthetic_sample_count",
    "total_row_count",
    "mean_high_minus_low_effect",
    "std_high_minus_low_effect",
    "mean_abs_corr",
    "sample_effects",
    "stability_verdict",
]


def _feature_row(
    feature: str,
    horizon_ms: int,
    *,
    stable: bool = True,
    direction_consistency: float = 1.0,
    effect: float = 5.0,
    corr: float = 0.1,
    sample_effects: str = "s1:canonical_event_mode:positive:5|s2:canonical_event_mode:positive:4|s3:canonical_event_mode:positive:6",
    diagnostic_count: int = 0,
) -> dict[str, object]:
    return {
        "feature": feature,
        "horizon_ms": horizon_ms,
        "label": "hyperliquid_future_mid_move_ticks",
        "sample_count": 3,
        "eligible_sample_count": 3,
        "canonical_eligible_sample_count": 3 if diagnostic_count == 0 else 0,
        "canonical_independent_future_row_delta_count": 3 if horizon_ms >= 500 else 1,
        "majority_direction": "positive",
        "direction_consistency_ratio": direction_consistency,
        "positive_sample_count": 3 if stable else 2,
        "negative_sample_count": 0 if stable else 1,
        "zero_sample_count": 0,
        "diagnostic_synthetic_sample_count": diagnostic_count,
        "total_row_count": 30,
        "mean_high_minus_low_effect": effect,
        "std_high_minus_low_effect": 1,
        "mean_abs_corr": corr,
        "sample_effects": sample_effects,
        "stability_verdict": "stable_across_samples" if stable else "unstable_across_samples",
    }


def _make_aggregate(base: Path, *, canonical: bool = True, feature_rows: list[dict[str, object]]) -> Path:
    out = base / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    samples = [_sample("s1", canonical=canonical), _sample("s2", canonical=canonical), _sample("s3", canonical=canonical)]
    canonical_count = 3 if canonical else 0
    diagnostic_count = 0 if canonical else 3
    manifest = {
        "schema_version": "binance_led_hyperliquid_multisample_robustness_v1",
        "task_id": "test",
        "sample_count": len(samples),
        "canonical_sample_count": canonical_count,
        "diagnostic_synthetic_sample_count": diagnostic_count,
        "samples": samples,
    }
    (out / "multi_sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_csv(
        out / "sample_quality_matrix.csv",
        [_quality_row(sample) for sample in samples],
        [
            "sample_id",
            "pricing_signal_dir",
            "source_join_dir",
            "source_analysis_dir",
            "source_sample_dir",
            "decision_mode",
            "canonical_status",
            "alignment_run_manifest",
            "alignment_metrics",
            "event_decision_count",
            "synthetic_decision_count",
            "input_rows",
            "primary_rows",
            "excluded_rows",
            "pricing_signal_rows",
            "future_join_count",
            "missing_binance_join_count",
            "primary_usable_row_count",
            "effective_future_age_ms_min",
            "effective_future_age_ms_mean",
            "effective_future_age_ms_max",
            "horizon_count",
            "independent_future_row_delta_count",
            "horizon_future_row_delta_groups",
            "aliased_horizon_signature_count",
            "pricing_signal_recommendation",
            "single_public_sample_caveat",
            "quality_status",
        ],
    )
    _write_csv(out / "feature_horizon_stability_across_samples.csv", feature_rows, FEATURE_FIELDS)
    aliasing_rows = []
    for sample in samples:
        for horizon in [100, 250, 500, 1000]:
            aliasing_rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "decision_mode": sample["decision_mode"],
                    "canonical_status": sample["canonical_status"],
                    "horizon_ms": horizon,
                    "row_count": 10,
                    "effective_future_age_ms_min": horizon,
                    "effective_future_age_ms_mean": horizon + 10,
                    "effective_future_age_ms_max": horizon + 20,
                    "effective_future_row_delta_min": 1,
                    "effective_future_row_delta_mean": 1 if horizon < 500 else 3,
                    "effective_future_row_delta_max": 1 if horizon < 500 else 4,
                    "distinct_future_row_delta_count": 1 if horizon < 500 else 3,
                    "distinct_effective_age_count": 10,
                    "mean_age_alias_group": horizon,
                    "aliasing_status": "distinct_mean_effective_age",
                }
            )
    _write_csv(
        out / "effective_horizon_aliasing_by_sample.csv",
        aliasing_rows,
        [
            "sample_id",
            "decision_mode",
            "canonical_status",
            "horizon_ms",
            "row_count",
            "effective_future_age_ms_min",
            "effective_future_age_ms_mean",
            "effective_future_age_ms_max",
            "effective_future_row_delta_min",
            "effective_future_row_delta_mean",
            "effective_future_row_delta_max",
            "distinct_future_row_delta_count",
            "distinct_effective_age_count",
            "mean_age_alias_group",
            "aliasing_status",
        ],
    )
    _write_csv(
        out / "venue_state_conditioning_across_samples.csv",
        [
            {
                "horizon_ms": 500,
                "hyperliquid_context_quality": "primary_usable",
                "hyperliquid_join_age_bucket": "fresh_0_50ms",
                "hyperliquid_spread_bucket": "spread_0_10_ticks",
                "sample_count": 3,
                "row_count": 30,
                "conditioning_status": "multi_sample_regime",
            }
        ],
        [
            "horizon_ms",
            "hyperliquid_context_quality",
            "hyperliquid_join_age_bucket",
            "hyperliquid_spread_bucket",
            "sample_count",
            "row_count",
            "conditioning_status",
        ],
    )
    return out


def _four_feature_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for horizon in [100, 250, 500, 1000]:
        rows.append(_feature_row("binance_mid_move_ticks_from_prev", horizon, effect=9, corr=0.12))
        rows.append(_feature_row("binance_top5_imbalance", horizon, effect=6, corr=0.10))
        rows.append(_feature_row("binance_top5_bid_qty", horizon, effect=5, corr=0.11))
        rows.append(
            _feature_row(
                "binance_microprice_minus_mid_ticks",
                horizon,
                stable=horizon != 1000,
                direction_consistency=0.67 if horizon == 1000 else 1.0,
                effect=7,
                corr=0.10,
            )
        )
    return rows


def test_build_ranking_outputs_all_allowlist_features_only(tmp_path: Path) -> None:
    input_dir = _make_aggregate(tmp_path, feature_rows=_four_feature_rows())
    output_dir = tmp_path / "ranking"

    result = ranking.build_signal_quality_ranking_artifacts(
        input_dir=input_dir,
        contract_dir=ranking.DEFAULT_CONTRACT_DIR,
        output_dir=output_dir,
    )

    features = [row["feature"] for row in result["ranking_rows"]]
    assert set(features) == set(ranking.EXPECTED_PRIMARY_ALLOWLIST)
    assert features[0] == "binance_mid_move_ticks_from_prev"
    assert all(row["bucket"] in ranking.ALLOWED_BUCKETS for row in result["ranking_rows"])
    assert (output_dir / "signal_quality_ranking.csv").exists()
    assert (output_dir / "signal_quality_reject_watch_list.csv").exists()
    assert (output_dir / "signal_quality_ranking_manifest.json").exists()
    assert (output_dir / "signal_quality_ranking_report.md").exists()

    manifest = json.loads((output_dir / "signal_quality_ranking_manifest.json").read_text(encoding="utf-8"))
    assert manifest["boundary_flags"]["no_order_endpoints"] is True
    assert manifest["boundary_flags"]["no_strategy_implementation"] is True
    assert manifest["diagnostic_rejection_count"] == 0
    assert manifest["primary_allowlist"] == ranking.EXPECTED_PRIMARY_ALLOWLIST


def test_classification_penalizes_short_horizon_reliance_and_concentration() -> None:
    rows: list[dict[str, object]] = []
    for horizon in [100, 250, 500, 1000]:
        rows.append(_feature_row("binance_mid_move_ticks_from_prev", horizon, effect=8, corr=0.12))
        rows.append(
            _feature_row(
                "binance_top5_imbalance",
                horizon,
                stable=horizon != 1000,
                direction_consistency=0.67 if horizon == 1000 else 1.0,
                effect=5,
                corr=0.12,
            )
        )
        rows.append(
            _feature_row(
                "binance_top5_bid_qty",
                horizon,
                effect=5,
                corr=0.10,
                sample_effects="s1:canonical_event_mode:positive:19|s2:canonical_event_mode:positive:1|s3:canonical_event_mode:positive:1",
            )
        )
        rows.append(
            _feature_row(
                "binance_microprice_minus_mid_ticks",
                horizon,
                stable=False,
                direction_consistency=0.34,
                effect=1,
                corr=0.02,
            )
        )

    summaries = ranking._summarize_feature_rows(
        feature_rows=[{key: str(value) for key, value in row.items()} for row in rows],
        allowlist=ranking.EXPECTED_PRIMARY_ALLOWLIST,
        canonical_sample_count=3,
        usable_primary_row_count=30,
    )
    by_feature = {summary.feature: summary for summary in summaries}

    assert by_feature["binance_top5_imbalance"].bucket == "watch_regime_dependent"
    assert by_feature["binance_top5_imbalance"].short_horizon_reliance_penalty > 0
    assert by_feature["binance_top5_bid_qty"].bucket == "watch_regime_dependent"
    assert by_feature["binance_top5_bid_qty"].single_sample_concentration_penalty > 0
    assert by_feature["binance_microprice_minus_mid_ticks"].bucket == "reject_for_canonical_signal_ranking"


def test_refuses_diagnostic_only_synthetic_input(tmp_path: Path) -> None:
    diagnostic_rows = [
        _feature_row(
            "binance_top5_imbalance",
            100,
            diagnostic_count=3,
            sample_effects="",
        )
    ]
    input_dir = _make_aggregate(tmp_path, canonical=False, feature_rows=diagnostic_rows)

    with pytest.raises(ranking.SignalQualityRankingError, match="non-canonical|zero canonical"):
        ranking.build_signal_quality_ranking_artifacts(
            input_dir=input_dir,
            contract_dir=ranking.DEFAULT_CONTRACT_DIR,
            output_dir=tmp_path / "ranking",
        )


def test_real_canonical_input_ranking_matches_controller_interpretation(tmp_path: Path) -> None:
    output_dir = tmp_path / "ranking"
    result = ranking.build_signal_quality_ranking_artifacts(
        input_dir=ranking.DEFAULT_INPUT_DIR,
        contract_dir=ranking.DEFAULT_CONTRACT_DIR,
        output_dir=output_dir,
    )
    rows = _read_csv(output_dir / "signal_quality_ranking.csv")
    by_feature = {row["feature"]: row for row in rows}

    assert len(rows) == 4
    assert rows[0]["feature"] == "binance_mid_move_ticks_from_prev"
    assert by_feature["binance_mid_move_ticks_from_prev"]["bucket"] == "keep_for_read_only_research"
    assert by_feature["binance_top5_imbalance"]["bucket"] == "keep_for_read_only_research"
    assert by_feature["binance_top5_bid_qty"]["bucket"] == "watch_regime_dependent"
    assert by_feature["binance_microprice_minus_mid_ticks"]["bucket"] == "watch_regime_dependent"
    assert all(row["controller_alignment"] == "matches_current_controller_interpretation" for row in rows)
