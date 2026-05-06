from __future__ import annotations

from pathlib import Path

import pytest

from align_live_run import (
    LiveInitialState,
    build_backtest_config,
    find_audit_csv,
    read_live_audit_run_id,
    read_live_first_feed_ts_local,
    read_live_initial_state,
    read_live_window,
)


def test_find_audit_csv_prefers_run_id_match(tmp_path: Path) -> None:
    old = tmp_path / "audit_live.csv"
    matched = tmp_path / "audit_live_live_btcusdt_123.csv"
    old.write_text("x\n")
    matched.write_text("x\n")

    assert find_audit_csv(tmp_path, "live_btcusdt_123") == matched


def test_read_live_window_preserves_nanosecond_timestamps(tmp_path: Path) -> None:
    audit = tmp_path / "audit_live.csv"
    audit.write_text(
        "ts_local,action\n"
        "1777342117432305664,keep\n"
        "1777347796789512192,keep\n"
    )

    window = read_live_window(audit)

    assert window.first_ts_local == 1777342117432305664
    assert window.last_ts_local == 1777347796789512192
    assert window.rows == 2
    assert window.start_day == "2026-04-28"
    assert window.end_day == "2026-04-28"


def test_read_live_audit_run_id_uses_single_audit_run_id(tmp_path: Path) -> None:
    audit = tmp_path / "audit_live.csv"
    audit.write_text(
        "run_id,ts_local,action\n"
        "base_btcusdt_123,100,keep\n"
        "base_btcusdt_123,200,keep\n"
    )

    identity = read_live_audit_run_id(audit, "base")

    assert identity.requested_run_id == "base"
    assert identity.audit_run_id == "base_btcusdt_123"


def test_read_live_audit_run_id_prefers_unique_prefix_match(tmp_path: Path) -> None:
    audit = tmp_path / "audit_live.csv"
    audit.write_text(
        "run_id,ts_local,action\n"
        "other,100,keep\n"
        "base_btcusdt_123,200,keep\n"
    )

    identity = read_live_audit_run_id(audit, "base")

    assert identity.audit_run_id == "base_btcusdt_123"


def test_read_live_audit_run_id_raises_for_ambiguous_prefix(tmp_path: Path) -> None:
    audit = tmp_path / "audit_live.csv"
    audit.write_text(
        "run_id,ts_local,action\n"
        "base_btcusdt_123,100,keep\n"
        "base_btcusdt_456,200,keep\n"
    )

    with pytest.raises(ValueError, match="multiple live audit run_id"):
        read_live_audit_run_id(audit, "base")


def test_read_live_initial_state_prefers_rest_position(tmp_path: Path) -> None:
    audit = tmp_path / "audit_live.csv"
    audit.write_text(
        "ts_local,position,rest_position,action\n"
        "1777342117432305664,0.002,-0.001,keep\n"
    )

    state = read_live_initial_state(audit)

    assert state.position == pytest.approx(-0.001)
    assert state.rest_position == pytest.approx(-0.001)
    assert state.ts_local == 1777342117432305664
    assert state.source_field == "rest_position"


def test_read_live_initial_state_falls_back_to_position(tmp_path: Path) -> None:
    audit = tmp_path / "audit_live.csv"
    audit.write_text("ts_local,position,action\n100,-0.001,keep\n")

    state = read_live_initial_state(audit)

    assert state.position == pytest.approx(-0.001)
    assert state.rest_position is None
    assert state.source_field == "position"


def test_read_live_first_feed_ts_local_prefers_feed_latency_anchor(tmp_path: Path) -> None:
    audit = tmp_path / "audit_live.csv"
    audit.write_text(
        "run_id,event_type,ts_local,ts_exch,feed_latency_ns,action\n"
        "live,decision,1000,900,25,keep\n"
    )

    assert read_live_first_feed_ts_local(audit, "live") == 925


def test_build_backtest_config_injects_audit_replay_fields(tmp_path: Path) -> None:
    base_cfg = {
        "symbol": {"name": "BTCUSDT"},
        "market": {"tick_size": 0.1, "lot_size": 0.001, "contract_size": 1.0},
            "risk": {"order_notional": 100.0},
            "latency": {"latency_guard_ms": 5.0},
            "strategy": {"two_phase_replace_enabled": True},
        }
    audit = tmp_path / "audit_live.csv"
    latency = tmp_path / "latency.npz"

    cfg, path = build_backtest_config(
        base_cfg=base_cfg,
        local_dir=tmp_path,
        latency_npz=latency,
        audit_csv=audit,
        run_id="live_btcusdt_123",
        mode="audit_replay",
        initial_state=LiveInitialState(
            position=-0.001,
            rest_position=-0.001,
            ts_local=1777342117432305664,
            source_field="rest_position",
        ),
    )

    assert path == tmp_path / "config_backtest_audit_replay.toml"
    assert cfg["paths"]["output_root"] == str(tmp_path / "out" / "backtest_audit_replay")
    assert cfg["latency"]["order_latency_npz"] == str(latency)
    assert "guard_mode" not in cfg["latency"]
    assert cfg["feed_latency"]["audit_csv"] == str(audit)
    assert cfg["feed_latency"]["run_id"] == "live_btcusdt_123"
    assert cfg["strategy"]["two_phase_replace_enabled"] is True
    assert cfg["queue"]["power_prob_n"] == 5.0
    assert cfg["audit"]["output_csv"] == "audit_bt_audit_replay.csv"
    assert cfg["market_data_replay"]["live_local_feed_compat"] is True
    assert cfg["backtest_cadence"]["mode"] == "audit_replay"
    assert cfg["backtest_cadence"]["audit_csv"] == str(audit)
    assert cfg["backtest_cadence"]["run_id"] == "live_btcusdt_123"
    assert cfg["backtest_cadence"]["replay_mode"] == "emit_due"
    assert cfg["backtest_cadence"]["trigger_ts_source"] == "feed_local"
    assert cfg["backtest_cadence"]["feed_latency_column"] == "feed_latency_ns"
    assert cfg["backtest_cadence"]["max_lag_ms"] == 250.0
    assert cfg["backtest_cadence"]["max_exch_lag_ms"] == 250.0
    assert cfg["backtest_cadence"]["strict_lag_gate"] is True
    assert cfg["backtest_cadence"]["lag_gate_action"] == "fail"
    assert cfg["backtest_cadence"]["market_state_overlay"] == "audit"
    assert cfg["alignment_init"]["enabled"] is True
    assert cfg["alignment_init"]["position_mode"] == "synthetic_fill"
    assert cfg["alignment_init"]["position"] == pytest.approx(-0.001)
    assert cfg["alignment_init"]["source"] == "rest_position"
    assert cfg["alignment_init"]["ts_local"] == 1777342117432305664
    assert path.exists()
