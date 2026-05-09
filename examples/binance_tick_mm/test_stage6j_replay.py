from __future__ import annotations

from pathlib import Path

from stage6j_replay import RULE_CANDIDATES, _prepare_config, _summarize


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
    assert "add_side_guard_post_fill_50ms" in names
    assert "add_side_guard_post_fill_100ms" in names
    assert "add_side_guard_post_fill_200ms" in names
    assert "broad_add_side_cooldown_200ms_control" in names


def test_prepare_config_forces_optimization_replay_overlay_contract(tmp_path: Path) -> None:
    candidate = {
        "candidate": "add_side_guard_post_fill_100ms",
        "cancel_race_guard_enabled": True,
        "cancel_race_guard_pending_cancel_block": True,
        "cancel_race_guard_post_fill_cooldown_ms": 100.0,
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
    assert cfg["risk"]["cancel_race_guard_post_fill_cooldown_ms"] == 100.0
    assert cfg["risk"]["inventory_add_side_cancel_cooldown_ms"] == 0.0


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
