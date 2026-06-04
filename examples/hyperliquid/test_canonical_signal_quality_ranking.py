from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import canonical_event_mode_evidence as loader
import canonical_signal_quality_ranking as ranking
from test_canonical_event_mode_evidence import _make_aggregate, _sample


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _allowlist_csv(path: Path) -> Path:
    rows = [
        {"feature": feature, "decision": "allow", "status": "primary_allowlist"}
        for feature in ranking.DEFAULT_FEATURES
    ]
    _write_csv(path, rows, ["feature", "decision", "status"])
    return path


def _replace_feature_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
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
    _write_csv(path / "feature_horizon_stability_across_samples.csv", rows, fieldnames)


def _feature_row(
    feature: str,
    horizon_ms: int,
    *,
    consistency: float,
    effect: float,
    corr: float,
    independent_delta: int,
    stable: bool = True,
    sample_effects: str = "a:canonical_event_mode:positive:1|b:canonical_event_mode:positive:1|c:canonical_event_mode:positive:1",
) -> dict[str, object]:
    return {
        "feature": feature,
        "horizon_ms": horizon_ms,
        "label": "hyperliquid_future_mid_move_ticks",
        "sample_count": 3,
        "eligible_sample_count": 3,
        "canonical_eligible_sample_count": 3,
        "canonical_independent_future_row_delta_count": independent_delta,
        "majority_direction": "positive" if consistency >= 0.5 else "negative",
        "direction_consistency_ratio": consistency,
        "positive_sample_count": 3 if consistency >= 1 else 2,
        "negative_sample_count": 0 if consistency >= 1 else 1,
        "zero_sample_count": 0,
        "diagnostic_synthetic_sample_count": 0,
        "total_row_count": 300,
        "mean_high_minus_low_effect": effect,
        "std_high_minus_low_effect": 0,
        "mean_abs_corr": corr,
        "sample_effects": sample_effects,
        "stability_verdict": "stable_across_samples" if stable else "unstable_across_samples",
    }


def _canonical_input(tmp_path: Path) -> Path:
    return _make_aggregate(
        tmp_path,
        [
            _sample("sample_a_event", decision_mode="event", canonical_status=loader.CANONICAL_EVENT_STATUS),
            _sample("sample_b_event", decision_mode="event", canonical_status=loader.CANONICAL_EVENT_STATUS),
            _sample("sample_c_event", decision_mode="event", canonical_status=loader.CANONICAL_EVENT_STATUS),
        ],
    )


def test_ranking_orders_stable_1000ms_features(tmp_path: Path) -> None:
    input_dir = _canonical_input(tmp_path)
    rows: list[dict[str, object]] = []
    for feature, effect, corr in [
        ("binance_mid_move_ticks_from_prev", 100, 0.1),
        ("binance_top5_imbalance", 60, 0.2),
        ("binance_top5_bid_qty", 40, 0.1),
        ("binance_microprice_minus_mid_ticks", 20, 0.1),
    ]:
        rows.append(_feature_row(feature, 1000, consistency=1.0, effect=effect, corr=corr, independent_delta=4))
        rows.append(_feature_row(feature, 500, consistency=1.0, effect=effect / 2, corr=corr, independent_delta=3))
    _replace_feature_rows(input_dir, rows)

    result = ranking.build_signal_quality_ranking(
        input_dir=input_dir,
        output_dir=tmp_path / "ranking",
        allowlist_csv=_allowlist_csv(tmp_path / "allowlist.csv"),
    )

    ordered = [row["feature"] for row in result["ranking_rows"]]
    assert ordered[0] == "binance_mid_move_ticks_from_prev"
    assert set(ordered) == set(ranking.DEFAULT_FEATURES)
    assert {row["final_bucket"] for row in result["ranking_rows"]} == {"keep_for_read_only_research"}


def test_watch_classification_for_regime_dependent_signal(tmp_path: Path) -> None:
    input_dir = _canonical_input(tmp_path)
    rows = []
    for feature in ranking.DEFAULT_FEATURES:
        stable = feature != "binance_microprice_minus_mid_ticks"
        rows.append(
            _feature_row(
                feature,
                1000,
                consistency=1.0 if stable else 0.8,
                effect=80,
                corr=0.1,
                independent_delta=4,
                stable=stable,
            )
        )
        rows.append(
            _feature_row(
                feature,
                500,
                consistency=1.0 if stable else 0.9,
                effect=40,
                corr=0.1,
                independent_delta=3,
                stable=True,
            )
        )
    _replace_feature_rows(input_dir, rows)

    result = ranking.build_signal_quality_ranking(
        input_dir=input_dir,
        output_dir=tmp_path / "ranking",
        allowlist_csv=_allowlist_csv(tmp_path / "allowlist.csv"),
    )
    micro = next(row for row in result["ranking_rows"] if row["feature"] == "binance_microprice_minus_mid_ticks")
    assert micro["final_bucket"] == "watch_regime_dependent"


def test_sample_concentration_penalizes_score(tmp_path: Path) -> None:
    balanced = _feature_row(
        "binance_top5_imbalance",
        1000,
        consistency=1.0,
        effect=100,
        corr=0.2,
        independent_delta=4,
        sample_effects="a:canonical_event_mode:positive:10|b:canonical_event_mode:positive:10|c:canonical_event_mode:positive:10",
    )
    concentrated = _feature_row(
        "binance_top5_imbalance",
        1000,
        consistency=1.0,
        effect=100,
        corr=0.2,
        independent_delta=4,
        sample_effects="a:canonical_event_mode:positive:28|b:canonical_event_mode:positive:1|c:canonical_event_mode:positive:1",
    )
    balanced_score = ranking._aggregate_feature_rows([balanced], features=["binance_top5_imbalance"])[0]
    concentrated_score = ranking._aggregate_feature_rows([concentrated], features=["binance_top5_imbalance"])[0]
    assert float(concentrated_score["ranking_score"]) < float(balanced_score["ranking_score"])
    assert "sample_concentration_watch" in concentrated_score["reason"]


def test_short_horizon_only_signal_is_rejected(tmp_path: Path) -> None:
    input_dir = _canonical_input(tmp_path)
    rows = [
        _feature_row(feature, 100, consistency=1.0, effect=100, corr=0.2, independent_delta=1)
        for feature in ranking.DEFAULT_FEATURES
    ]
    _replace_feature_rows(input_dir, rows)

    result = ranking.build_signal_quality_ranking(
        input_dir=input_dir,
        output_dir=tmp_path / "ranking",
        allowlist_csv=_allowlist_csv(tmp_path / "allowlist.csv"),
    )
    assert {row["final_bucket"] for row in result["ranking_rows"]} == {"reject_for_canonical_signal_ranking"}


def test_refuses_diagnostic_only_input(tmp_path: Path) -> None:
    input_dir = _make_aggregate(
        tmp_path,
        [
            _sample(
                "sample_synth",
                decision_mode="synthetic_fixed_grid",
                canonical_status=loader.DIAGNOSTIC_SYNTHETIC_STATUS,
            )
        ],
    )
    with pytest.raises((loader.EvidenceValidationError, ranking.RankingInputError)):
        ranking.build_signal_quality_ranking(
            input_dir=input_dir,
            output_dir=tmp_path / "ranking",
            allowlist_csv=_allowlist_csv(tmp_path / "allowlist.csv"),
        )


def test_writes_manifest_and_report(tmp_path: Path) -> None:
    input_dir = _canonical_input(tmp_path)
    output_dir = tmp_path / "ranking"
    ranking.build_signal_quality_ranking(
        input_dir=input_dir,
        output_dir=output_dir,
        allowlist_csv=_allowlist_csv(tmp_path / "allowlist.csv"),
    )

    manifest = json.loads((output_dir / "signal_quality_ranking_manifest.json").read_text(encoding="utf-8"))
    assert manifest["canonical_sample_count"] == 3
    assert manifest["ranked_feature_count"] == 4
    assert (output_dir / "signal_quality_ranking.csv").exists()
    assert (output_dir / "signal_quality_reject_watch_list.csv").exists()
    assert (output_dir / "signal_quality_ranking_report.md").exists()


def test_report_interpretation_matches_actual_buckets(tmp_path: Path) -> None:
    input_dir = _canonical_input(tmp_path)
    rows = []
    for feature in ranking.DEFAULT_FEATURES:
        is_mid_move = feature == "binance_mid_move_ticks_from_prev"
        rows.append(
            _feature_row(
                feature,
                1000,
                consistency=1.0 if is_mid_move else 0.8,
                effect=100 if is_mid_move else 40,
                corr=0.2 if is_mid_move else 0.1,
                independent_delta=4,
                stable=is_mid_move,
            )
        )
        rows.append(
            _feature_row(
                feature,
                500,
                consistency=1.0 if is_mid_move else 0.9,
                effect=50 if is_mid_move else 20,
                corr=0.2 if is_mid_move else 0.1,
                independent_delta=3,
                stable=True,
            )
        )
    _replace_feature_rows(input_dir, rows)
    output_dir = tmp_path / "ranking"

    result = ranking.build_signal_quality_ranking(
        input_dir=input_dir,
        output_dir=output_dir,
        allowlist_csv=_allowlist_csv(tmp_path / "allowlist.csv"),
    )

    report = (output_dir / "signal_quality_ranking_report.md").read_text(encoding="utf-8")
    for row in result["ranking_rows"]:
        assert f"`{row['feature']}` is `{row['final_bucket']}`" in report
    top5 = next(row for row in result["ranking_rows"] if row["feature"] == "binance_top5_imbalance")
    assert top5["final_bucket"] == "watch_regime_dependent"
    assert "`binance_top5_imbalance` is `watch_regime_dependent`" in report
    assert "binance_top5_imbalance` remains a strong book-pressure candidate and is kept" not in report
