from __future__ import annotations

import csv
import json
from pathlib import Path

from examples.hyperliquid import hyperliquid_tiny_live_m2_pnl_ledger as ledger


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_m1_like_window(root: Path) -> Path:
    window = root / "window_1" / "pulled_back_awsserver1"
    _write_json(
        window / "executor_manifest.json",
        {
            "order_submission_attempted": True,
            "order_status_types": ["resting"],
            "private_endpoint_called": True,
            "real_order_endpoint_called": True,
            "real_cancel_endpoint_called": True,
            "shutdown_proof_status": "pass",
        },
    )
    _write_json(
        window / "private_order_response_audit.json",
        {"order_status_rows": [{"status_type": "resting", "payload": {"oid": "0x" + "1" * 64}}]},
    )
    _write_json(
        window / "private_preflight_summary.json",
        {
            "preflight_summary": {
                "client_preflight": {"user_fill_count": 0},
                "asset_position_count_before": 0,
            }
        },
    )
    _write_json(window / "cancel_shutdown_proof.json", {"final_open_orders": []})
    _write_csv(
        window / "order_intent_audit.csv",
        [
            {
                "symbol": "BTC",
                "side": "buy",
                "size_btc": "0.01",
                "limit_px": "62634.0",
                "time_in_force": "Alo",
                "endpoint_called": "true",
            }
        ],
        ["symbol", "side", "size_btc", "limit_px", "time_in_force", "endpoint_called"],
    )
    return window


def test_m1_no_fill_artifacts_fail_closed_for_realized_pnl(tmp_path: Path) -> None:
    input_root = tmp_path / "m1"
    make_m1_like_window(input_root)

    manifest = ledger.run_ledger(input_root=input_root, output_dir=tmp_path / "out")

    assert manifest["final_recommendation"] == ledger.READY_RECOMMENDATION
    assert manifest["windows_found"] == 1
    assert manifest["live_realized_pnl_proof"] is False
    assert manifest["realized_pnl_proof_status"] == "fail_closed_no_realized_live_pnl"
    with (tmp_path / "out" / "source_completeness_matrix.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["fills_available"] == "False"
    assert rows[0]["realized_pnl_proof_status"] == "unavailable_no_fill_or_settlement"


def test_complete_fixture_computes_pnl_fee_inventory_and_slippage(tmp_path: Path) -> None:
    input_root = tmp_path / "m1"
    make_m1_like_window(input_root)
    fixture = tmp_path / "fills.csv"
    ledger.write_fixture(fixture)

    manifest = ledger.run_ledger(input_root=input_root, output_dir=tmp_path / "out", fixture_fills=fixture)

    summary = manifest["ledger_summary"]
    assert summary["fill_count"] == 1
    assert summary["maker_fill_count"] == 1
    assert summary["gross_pnl_usdc"] == 0.26
    assert summary["fee_usdc"] == 0.125268
    assert summary["net_pnl_usdc"] == 0.134732
    assert summary["inventory_delta_btc"] == 0.01
    assert summary["slippage_usdc"] == 0.0


def test_missing_input_windows_blocks(tmp_path: Path) -> None:
    manifest = ledger.run_ledger(input_root=tmp_path / "missing", output_dir=tmp_path / "out")

    assert manifest["final_recommendation"] == ledger.BLOCKED_RECOMMENDATION
    assert manifest["blocking_reasons"] == ["no_input_windows_found"]


def test_redaction_masks_secret_shaped_values() -> None:
    redacted = ledger.redact({"oid": "0x" + "1" * 64, "address": "0x" + "2" * 40, "ok": "kept"})

    assert redacted["oid"] == "<redacted>"
    assert redacted["address"] == "<redacted>"
    assert redacted["ok"] == "kept"
