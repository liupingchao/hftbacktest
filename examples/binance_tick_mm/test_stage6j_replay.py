from __future__ import annotations

from pathlib import Path

from stage6j_replay import (
    RULE_CANDIDATES,
    _attach_baseline_action_path_diffs,
    _prepare_config,
    _scan_action_path_coverage,
    _summarize,
)


def _base_cfg(tmp_path: Path) -> dict[str, object]:
    return {
        "paths": {"output_root": str(tmp_path / "old")},
        "summary": {"enabled": False, "output_json": "old.json", "daily_csv": "old.csv"},
        "audit": {"output_csv": "old.csv", "mode": "off"},
        "backtest_cadence": {
            "mode": "audit_replay",
            "market_state_overlay": "audit",
            "strategy_position_overlay": "audit",
            "working_order_overlay": "audit",
        },
        "risk": {
            "inventory_inflight_exposure_enabled": False,
            "inventory_add_side_cancel_cooldown_ms": 999.0,
        },
    }


def test_rule_candidates_include_required_controls() -> None:
    names = {str(candidate["candidate"]) for candidate in RULE_CANDIDATES}

    assert "baseline_inflight_only" in names
    assert "add_side_guard_only" in names
    assert "add_side_toxic_timing_50ms" in names
    assert "add_side_toxic_timing_100ms" in names
    assert "add_side_toxic_timing_200ms" in names
    assert "add_side_guard_plus_toxic_timing_100ms" in names
    assert "broad_add_side_cooldown_200ms_control" in names


def test_prepare_config_forces_optimization_replay_overlay_contract(tmp_path: Path) -> None:
    candidate = {
        "candidate": "add_side_guard_plus_toxic_timing_100ms",
        "cancel_race_guard_enabled": True,
        "cancel_race_guard_pending_cancel_block": True,
        "cancel_race_guard_post_fill_cooldown_ms": 0.0,
        "adverse_timing_guard_enabled": True,
        "adverse_timing_guard_target_deterioration_enabled": True,
        "adverse_timing_guard_pending_cancel_enabled": True,
        "adverse_timing_guard_post_cancel_fill_enabled": True,
        "adverse_timing_guard_cooldown_ms": 100.0,
        "adverse_timing_guard_min_target_move_ticks": 2,
        "adverse_timing_guard_block_mode": "add_side_only",
        "add_side_toxic_timing_guard_enabled": True,
        "add_side_toxic_timing_guard_pending_cancel_enabled": True,
        "add_side_toxic_timing_guard_post_cancel_fill_enabled": True,
        "add_side_toxic_timing_guard_target_move_enabled": True,
        "add_side_toxic_timing_guard_window_ms": 100.0,
        "add_side_toxic_timing_guard_min_target_move_ticks": 2,
        "add_side_toxic_timing_guard_latency_threshold_ms": 0.0,
        "add_side_toxic_timing_guard_block_mode": "add_side_submit_only",
        "inventory_add_side_cancel_cooldown_ms": 0.0,
    }

    cfg = _prepare_config(_base_cfg(tmp_path), tmp_path / "run", candidate)

    assert cfg["paths"]["output_root"] == str(tmp_path / "run")
    assert cfg["summary"]["enabled"] is True
    assert cfg["summary"]["output_json"] == "summary.json"
    assert cfg["audit"]["mode"] == "full"
    assert cfg["audit"]["output_csv"] == "audit_bt_stage6j.csv"
    assert cfg["backtest_cadence"]["market_state_overlay"] == "off"
    assert cfg["backtest_cadence"]["strategy_position_overlay"] == "off"
    assert cfg["backtest_cadence"]["working_order_overlay"] == "off"
    assert cfg["risk"]["inventory_inflight_exposure_enabled"] is True
    assert cfg["risk"]["cancel_race_guard_enabled"] is True
    assert cfg["risk"]["cancel_race_guard_post_fill_cooldown_ms"] == 0.0
    assert cfg["risk"]["adverse_timing_guard_enabled"] is True
    assert cfg["risk"]["adverse_timing_guard_cooldown_ms"] == 100.0
    assert cfg["risk"]["adverse_timing_guard_min_target_move_ticks"] == 2
    assert cfg["risk"]["adverse_timing_guard_block_mode"] == "add_side_only"
    assert cfg["risk"]["add_side_toxic_timing_guard_enabled"] is True
    assert cfg["risk"]["add_side_toxic_timing_guard_window_ms"] == 100.0
    assert cfg["risk"]["add_side_toxic_timing_guard_min_target_move_ticks"] == 2
    assert cfg["risk"]["add_side_toxic_timing_guard_block_mode"] == "add_side_submit_only"
    assert cfg["risk"]["inventory_add_side_cancel_cooldown_ms"] == 0.0


def test_scan_action_path_coverage_counts_submit_blocks_and_reduce_side(tmp_path: Path) -> None:
    audit_csv = tmp_path / "audit.csv"
    audit_csv.write_text(
        "\n".join(
            [
                "add_side_submit_eligible_buy,add_side_submit_eligible_sell,add_side_submit_blocked_buy,add_side_submit_blocked_sell,add_side_submit_reduce_side_allowed_buy,add_side_submit_reduce_side_allowed_sell",
                "1,0,1,0,0,1",
                "0,1,0,1,1,0",
                "1,1,0,1,0,1",
                "",
            ]
        )
    )

    coverage = _scan_action_path_coverage(audit_csv)

    assert coverage["add_side_submit_eligible_buy_count"] == 2
    assert coverage["add_side_submit_eligible_sell_count"] == 2
    assert coverage["add_side_submit_blocked_buy_count"] == 1
    assert coverage["add_side_submit_blocked_sell_count"] == 2
    assert coverage["add_side_submit_blocked_total_count"] == 3
    assert coverage["add_side_submit_blocked_reduce_side_count"] == 1


def test_attach_baseline_action_path_diffs_counts_suppressed_submit(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    baseline.write_text(
        "\n".join(
            [
                "ts_local,action,planned_action,add_side_submit_blocked_buy,add_side_submit_blocked_sell",
                "1,submit_buy,submit_buy,0,0",
                "2,submit_sell,submit_sell,0,0",
                "",
            ]
        )
    )
    candidate.write_text(
        "\n".join(
            [
                "ts_local,action,planned_action,add_side_submit_blocked_buy,add_side_submit_blocked_sell",
                "1,keep,keep,1,0",
                "2,submit_sell,submit_sell,0,0",
                "",
            ]
        )
    )
    rows = [
        {
            "run_id": "r1",
            "candidate": "baseline_inflight_only",
            "status": "ok",
            "audit_csv": str(baseline),
            "rows": "2",
        },
        {
            "run_id": "r1",
            "candidate": "add_side_toxic_timing_100ms",
            "status": "ok",
            "audit_csv": str(candidate),
        },
    ]

    _attach_baseline_action_path_diffs(rows)

    assert rows[0]["baseline_row_compare_count"] == 2
    assert rows[1]["baseline_action_or_planned_diff_count"] == 1
    assert rows[1]["baseline_submit_overlap_blocked_count"] == 1
    assert rows[1]["submit_removed_by_guard_count"] == 1


def test_summarize_marks_enabled_overlays_as_hard_failure() -> None:
    rows = [
        {
            "run_id": "r1",
            "candidate": "baseline_inflight_only",
            "status": "ok",
            "pnl_mtm": "1.0",
            "max_abs_position_notional": "200.0",
            "cancel_fill_count": "2",
            "source_inventory_worsening_no_readd_count": "1",
            "source_same_side_readd_inventory_worsening_count": "0",
            "add_side_submit_blocked_total_count": "0",
            "add_side_submit_blocked_reduce_side_count": "0",
            "baseline_action_or_planned_diff_count": "0",
            "baseline_submit_overlap_blocked_count": "0",
            "submit_removed_by_guard_count": "0",
            "audit_replay_lag_gate_passed": "True",
            "audit_replay_market_state_overlay_mode": "off",
            "audit_replay_strategy_position_overlay_mode": "off",
            "audit_replay_working_order_overlay_mode": "off",
        },
        {
            "run_id": "r1",
            "candidate": "bad_overlay",
            "status": "ok",
            "pnl_mtm": "2.0",
            "max_abs_position_notional": "210.0",
            "cancel_fill_count": "1",
            "source_inventory_worsening_no_readd_count": "0",
            "source_same_side_readd_inventory_worsening_count": "0",
            "add_side_submit_blocked_total_count": "3",
            "add_side_submit_blocked_reduce_side_count": "0",
            "baseline_action_or_planned_diff_count": "2",
            "baseline_submit_overlap_blocked_count": "2",
            "submit_removed_by_guard_count": "2",
            "audit_replay_lag_gate_passed": "True",
            "audit_replay_market_state_overlay_mode": "audit",
            "audit_replay_strategy_position_overlay_mode": "off",
            "audit_replay_working_order_overlay_mode": "off",
        },
    ]

    summary = _summarize(rows)

    assert summary["sample_count"] == 1
    assert summary["hard_failure_count"] == 1
    assert summary["candidates"]["baseline_inflight_only"]["cancel_fill_count_sum"] == 2
