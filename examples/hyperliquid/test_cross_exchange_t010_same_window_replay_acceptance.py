from __future__ import annotations

import csv
import json
from pathlib import Path

from examples.hyperliquid import cross_exchange_t010_same_window_replay_acceptance as acceptance


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _make_artifact(root: Path, *, fill_count: int = 0) -> Path:
    event = root / "event_driven_edge_gate_live"
    _write_json(
        event / "event_driven_watcher_manifest.json",
        {
            "hyperliquid_l2book_fast": True,
            "watcher_seconds_elapsed": 436.0,
            "post_open_orders_public_state_pass_count": 1,
            "post_open_orders_public_state_block_count": 0,
            "trigger_found": True,
            "trigger_count": 1,
            "event_driven_guard_status": "pass",
            "event_driven_guard": {"handoff_phase": "post_open_orders_inline_reprice"},
            "live_submissions_count": 1,
            "post_only_reject_count": 0,
            "fill_count": fill_count,
            "maker_fill_count": fill_count,
        },
    )
    _write_json(
        event / "public_stream_summary.json",
        {
            "message_count_by_channel": {"l2Book": 10, "trades": 5},
            "total_trade_event_count": 20,
            "reconnect_count": 0,
            "subscription_options": {"hyperliquid_l2book_fast": True},
        },
    )
    _write_json(
        event / "inline_reprice_manifest.json",
        {
            "real_order_endpoint_called": True,
            "real_cancel_endpoint_called": True,
            "shutdown_proof_status": "pass",
            "order_status_types": ["resting"],
            "final_open_orders_count": 0,
        },
    )
    _write_json(event / "cancel_shutdown_proof.json", {"proof_status": "pass", "final_open_orders": []})
    _write_json(
        event / "private_order_response_audit.json",
        {
            "order_submission_attempted": True,
            "order_status_rows": [{"status_type": "resting", "payload": {"oid": "<redacted>"}}],
        },
    )
    _write_json(root / "independent_remote_open_orders_check.json", {"final_open_orders_count": 0})
    _write_json(event / "account_inventory_snapshots.json", {"post_state": {"assetPositions": []}})
    _write_json(event / "market_markout_snapshot.json", {"pre_submit_current_l2": {"levels": []}, "post_submit_current_l2": {"levels": []}})
    _write_json(event / "max_loss_monitor_summary.json", {"status": "pass"})
    _write_csv(
        event / "inline_reprice_attempt_matrix.csv",
        [
            {
                "guard_status": "pass",
                "edge_gate_status": "pass",
                "side": "buy",
                "limit_px": "63889.0",
                "size_btc": "0.002",
                "order_endpoint_called": "True",
            }
        ],
        ["guard_status", "edge_gate_status", "side", "limit_px", "size_btc", "order_endpoint_called"],
    )
    _write_csv(
        event / "inline_reprice_guard_matrix.csv",
        [{"status": "pass", "post_only_non_crossing": "True"}],
        ["status", "post_only_non_crossing"],
    )
    _write_csv(
        event / "public_state_freshness_matrix.csv",
        [{"state_observed_after_open_orders_end": "True"}],
        ["state_observed_after_open_orders_end"],
    )
    _write_csv(
        event / "order_intent_audit.csv",
        [{"side": "buy", "limit_px": "63889.0", "size_btc": "0.002", "time_in_force": "Alo"}],
        ["side", "limit_px", "size_btc", "time_in_force"],
    )
    _write_csv(
        event / "live_fill_ledger.csv",
        [],
        ["source_window", "fill_id", "side", "qty_btc", "price_usdc", "fee_usdc", "rebate_usdc", "liquidity"],
    )
    return root


def test_same_window_acceptance_passes_supported_no_fill_lifecycle(tmp_path: Path) -> None:
    input_root = _make_artifact(tmp_path / "input")

    manifest = acceptance.run_acceptance(input_root=input_root, output_dir=tmp_path / "out")

    assert manifest["final_recommendation"] == acceptance.PASSED_RECOMMENDATION
    assert manifest["market_view_acceptance"] == "pass"
    assert manifest["decision_path_acceptance"] == "pass"
    assert manifest["lifecycle_acceptance"] == "pass"
    assert manifest["economics_no_fill_acceptance"] == "pass"
    assert manifest["optimism_check_acceptance"] == "pass"
    assert manifest["boundary_status"] == "pass"


def test_same_window_acceptance_fails_if_no_fill_fact_is_violated(tmp_path: Path) -> None:
    input_root = _make_artifact(tmp_path / "input", fill_count=1)

    manifest = acceptance.run_acceptance(input_root=input_root, output_dir=tmp_path / "out")

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert manifest["lifecycle_acceptance"] == "fail"
    assert manifest["economics_no_fill_acceptance"] == "fail"
