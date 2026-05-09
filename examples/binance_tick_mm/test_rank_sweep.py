from __future__ import annotations

from pathlib import Path

import pytest

from rank_sweep import rank_sweep, resolve_position_limit


def _control() -> dict[str, object]:
    return {
        "rows": 100,
        "pnl_mtm": -1.0,
        "max_drawdown_mtm": 1.0,
        "avg_abs_position_notional": 100.0,
        "max_abs_position_notional": 200.0,
        "drop_api_rate": 0.10,
        "drop_latency_rate": 0.20,
        "actions": {"keep": 80, "submit_buy": 10, "cancel_buy": 10},
        "reject_reasons": {},
    }


def _row(**overrides: object) -> dict[str, str]:
    row = {
        "run_id": "1",
        "status": "ok",
        "pnl_mtm": "1.0",
        "max_drawdown_mtm": "1.0",
        "avg_abs_position_notional": "100.0",
        "max_abs_position_notional": "200.0",
        "drop_api_rate": "0.10",
        "drop_latency_rate": "0.20",
        "actions_keep": "80",
        "actions_submit_buy": "10",
        "actions_cancel_buy": "10",
        "audit_replay_lag_gate_passed": "True",
        "audit_replay_market_state_overlay_mode": "off",
        "audit_replay_strategy_position_overlay_mode": "off",
        "audit_replay_working_order_overlay_mode": "off",
    }
    row.update({key: str(value) for key, value in overrides.items()})
    return row


def test_rank_sweep_keeps_safe_candidates_and_scores_descending() -> None:
    result = rank_sweep(
        [
            _row(run_id=1, pnl_mtm="1.0"),
            _row(run_id=2, pnl_mtm="2.0"),
        ],
        _control(),
        max_abs_position_notional=250.0,
        max_drop_api_delta=0.02,
        max_drop_latency_delta=0.02,
        min_passed_candidates=1,
        churn_weight=0.0001,
        max_abs_position_notional_source="test",
    )

    assert result["passed"] is True
    assert [row["run_id"] for row in result["ranked"]] == ["2", "1"]
    assert result["rejected"] == []
    assert result["summary"]["max_abs_position_notional_limit"] == 250.0
    assert result["summary"]["max_abs_position_notional_limit_source"] == "test"


def test_rank_sweep_rejects_enabled_overlays_and_risk_breaches() -> None:
    result = rank_sweep(
        [
            _row(run_id=1, audit_replay_market_state_overlay_mode="audit"),
            _row(run_id=2, max_abs_position_notional="300"),
            _row(run_id=3, drop_api_rate="0.13"),
            _row(run_id=4, audit_replay_strategy_position_overlay_mode="audit"),
            _row(run_id=5, audit_replay_working_order_overlay_mode="audit"),
        ],
        _control(),
        max_abs_position_notional=250.0,
        max_drop_api_delta=0.02,
        max_drop_latency_delta=0.02,
        min_passed_candidates=1,
        churn_weight=0.0001,
    )

    assert result["passed"] is False
    reasons = "|".join(row["rejection_reasons"] for row in result["rejected"])
    assert "market_state_overlay_enabled" in reasons
    assert "strategy_position_overlay_enabled" in reasons
    assert "working_order_overlay_enabled" in reasons
    assert "max_abs_position_notional" in reasons
    assert "drop_api_rate" in reasons


def test_resolve_position_limit_prefers_explicit_cli_value(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text("[risk]\nmax_notional_pos = 250.0\nmax_position_qty = 0.003\n")

    limit = resolve_position_limit(
        [{"config_path": str(cfg)}],
        explicit_max_abs_position_notional=123.0,
        position_reference_price=80000.0,
    )

    assert limit.max_abs_position_notional == pytest.approx(123.0)
    assert limit.source == "cli:--max-abs-position-notional"


def test_resolve_position_limit_reads_config_path_and_takes_tighter_risk_limit(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text("[risk]\nmax_notional_pos = 250.0\nmax_position_qty = 0.003\n")

    limit = resolve_position_limit(
        [{"config_path": str(cfg)}],
        position_reference_price=90000.0,
    )

    assert limit.max_abs_position_notional == pytest.approx(250.0)
    assert "risk.max_notional_pos" in limit.source


def test_resolve_position_limit_can_use_qty_with_reference_price(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text("[risk]\nmax_notional_pos = 500.0\nmax_position_qty = 0.003\n")

    limit = resolve_position_limit(
        [{"config_path": str(cfg)}],
        position_reference_price=80000.0,
    )

    assert limit.max_abs_position_notional == pytest.approx(240.0)
    assert "risk.max_position_qty*position_reference_price" in limit.source


def test_resolve_position_limit_uses_base_config_before_sweep_rows(tmp_path: Path) -> None:
    base = tmp_path / "base.toml"
    row_cfg = tmp_path / "row.toml"
    base.write_text("[risk]\nmax_notional_pos = 200.0\n")
    row_cfg.write_text("[risk]\nmax_notional_pos = 250.0\n")

    limit = resolve_position_limit(
        [{"config_path": str(row_cfg)}],
        base_config=base,
    )

    assert limit.max_abs_position_notional == pytest.approx(200.0)
    assert str(base) in limit.source
