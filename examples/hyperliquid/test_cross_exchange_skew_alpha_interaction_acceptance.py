from __future__ import annotations

import json
from pathlib import Path

from examples.hyperliquid import cross_exchange_skew_alpha_interaction_acceptance as acceptance


def test_acceptance_uses_same_universe_and_keeps_skew_disabled(tmp_path: Path) -> None:
    result = acceptance.build_acceptance_artifacts(output_dir=tmp_path)
    manifest = result["manifest"]
    rows = result["rows"]

    zero_ids = [row["decision_id"] for row in rows if row["case"] == "alpha_zero_skew"]
    bounded_ids = [row["decision_id"] for row in rows if row["case"] == "alpha_bounded_skew"]
    assert zero_ids == bounded_ids
    assert manifest["same_universe"] is True
    assert manifest["final_recommendation"] == "offline_c12_structural_acceptance_pass_keep_skew_disabled"
    assert manifest["skew_enablement_recommendation"] == "remain_disabled_pending_real_c12_evidence"
    assert manifest["boundary"]["no_live_orders"] is True


def test_acceptance_separates_observed_proxy_and_censored_evidence(tmp_path: Path) -> None:
    result = acceptance.build_acceptance_artifacts(output_dir=tmp_path)
    manifest = result["manifest"]
    rows = result["rows"]

    assert manifest["structural_gates"]["observed_and_proxy_evidence_separate"] is True
    assert manifest["structural_gates"]["censored_rows_explicit"] is True
    assert all(row["censored"] is True for row in rows if row["observed_fill_status"] == "censored_no_fill")
    assert all(row["observed_fill_status"] != row["proxy_fill_status"] for row in rows)
    for case in ("alpha_zero_skew", "alpha_bounded_skew"):
        summary = manifest["summaries"][case]
        assert summary["observed_fill_count"] == 3
        assert summary["censored_no_fill_count"] == 3
        assert "proxy_fill_coverage" in summary
        assert "mean_observed_markout_1s_ticks" in summary
        assert "mean_observed_markout_5s_ticks" in summary
        assert "spread_retention" in summary
        assert "peak_abs_position_btc" in summary
        assert "mean_recovery_duration_ms" in summary


def test_acceptance_artifacts_are_deterministic_and_post_only(tmp_path: Path) -> None:
    first = acceptance.build_acceptance_artifacts(output_dir=tmp_path / "first")
    second = acceptance.build_acceptance_artifacts(output_dir=tmp_path / "second")

    assert first["rows"] == second["rows"]
    assert first["manifest"] == second["manifest"]
    assert all(row["post_only_invariant"] is True for row in first["rows"])
    assert (tmp_path / "first" / "skew_alpha_interaction_rows.csv").exists()
    persisted = json.loads(
        (tmp_path / "first" / "skew_alpha_interaction_summary.json").read_text(encoding="utf-8")
    )
    assert persisted["structural_gates"]["reduce_side_preserved_near_cap"] is True
