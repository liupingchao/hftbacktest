from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from examples.hyperliquid import cross_exchange_price_math
from examples.hyperliquid import hyperliquid_tiny_live_real_order_executor as executor


def test_default_config_validation_passes() -> None:
    config = executor.TinyLiveConfig()
    precision = executor.mock_precision()

    rows = executor.validate_config(config, precision)

    assert {row["status"] for row in rows} == {"pass"}


def test_cap_validation_rejects_duration_symbol_and_tif() -> None:
    precision = executor.mock_precision()
    config = executor.TinyLiveConfig(symbol="ETH", duration_seconds=601, time_in_force="Gtc")

    rows = executor.validate_config(config, precision)
    failed = {row["check"] for row in rows if row["status"] == "fail_closed"}

    assert {"symbol_is_btc", "duration_lte_600s", "time_in_force_is_alo"} <= failed
    with pytest.raises(executor.ValidationError):
        executor.assert_config_valid(config, precision)


def test_order_intent_rejects_non_alo_and_oversize() -> None:
    config = executor.TinyLiveConfig()
    precision = executor.mock_precision()
    bad_tif = executor.OrderIntent(symbol="BTC", is_buy=True, size_btc=0.01, limit_px=65000, time_in_force="Ioc")
    bad_size = executor.OrderIntent(symbol="BTC", is_buy=True, size_btc=0.02, limit_px=65000)

    with pytest.raises(executor.ValidationError, match="Alo"):
        executor.validate_order_intent(config, precision, bad_tif)
    with pytest.raises(executor.ValidationError, match="size"):
        executor.validate_order_intent(config, precision, bad_size)


def test_max_loss_missing_or_reached_fails_closed() -> None:
    config = executor.TinyLiveConfig()

    missing = executor.loss_status(config, None)
    reached = executor.loss_status(config, executor.LossSnapshot(entry_px=65000, mark_px=61000, position_btc=0.01))

    assert missing["status"] == "fail_closed"
    assert missing["reason"] == "missing_loss_snapshot"
    assert reached["status"] == "fail_closed"
    assert reached["reason"] == "max_loss_reached"


def test_live_order_path_requires_explicit_live_mode() -> None:
    config = executor.TinyLiveConfig(live_mode=False)
    precision = executor.mock_precision()
    intent = executor.OrderIntent(symbol="BTC", is_buy=True, size_btc=0.01, limit_px=65000)
    client = executor.MockHyperliquidClient()

    with pytest.raises(executor.ValidationError, match="live_mode=false"):
        executor.run_order_once(
            config=config,
            precision=precision,
            intent=intent,
            loss_snapshot=executor.LossSnapshot(entry_px=65000, mark_px=65000, position_btc=0.01),
            client=client,
        )


def test_live_mode_requires_operator_ack() -> None:
    config = executor.TinyLiveConfig(live_mode=True, operator_ack="")

    rows = executor.validate_config(config, executor.mock_precision())

    assert any(row["check"] == "live_mode_has_operator_ack" and row["status"] == "fail_closed" for row in rows)


def test_cancel_all_shutdown_flow_passes_when_tracked_refs_absent() -> None:
    client = executor.MockHyperliquidClient()
    intent = executor.OrderIntent(symbol="BTC", is_buy=True, size_btc=0.01, limit_px=65000)
    refs = executor.extract_tracked_refs(client.order(intent))

    evidence = executor.shutdown_cancel_all(client=client, symbol="BTC", tracked_refs=refs)

    assert evidence.proof_status == "pass"
    assert evidence.requested_refs
    assert len(evidence.cancel_results) == 1


def test_cancel_all_shutdown_fails_closed_when_tracked_ref_remains_open() -> None:
    client = executor.MockHyperliquidClient(final_open_orders=[{"oid": 618001000, "coin": "BTC"}])
    refs = [{"oid": 618001000, "cloid": "still_open"}]

    evidence = executor.shutdown_cancel_all(client=client, symbol="BTC", tracked_refs=refs)

    assert evidence.proof_status == "fail_closed"
    assert evidence.fail_closed_reason == "tracked_order_still_open_or_ambiguous"


def test_redaction_masks_sensitive_fields() -> None:
    payload = {
        "signature": "0xabc",
        "nested": {"private_key": "secret", "value": 1},
        "items": [{"nonce": 123}],
    }

    redacted = executor.redact(payload)

    assert redacted["signature"] == "<redacted>"
    assert redacted["nested"]["private_key"] == "<redacted>"
    assert redacted["items"][0]["nonce"] == "<redacted>"
    assert redacted["nested"]["value"] == 1


def test_generate_cloid_is_sdk_compatible_hex() -> None:
    cloid = executor.generate_cloid("0618T004")

    assert cloid.startswith("0x")
    assert len(cloid) == 34
    int(cloid[2:], 16)


def test_build_canary_intent_stays_post_only_and_under_notional_cap() -> None:
    precision = executor.PrecisionFacts(
        symbol="BTC",
        sz_decimals=5,
        tick_size=0.1,
        lot_size=0.00001,
        mid_px=110000.0,
        source="unit_test",
    )

    intent = executor.build_canary_intent(precision=precision)

    assert intent.symbol == "BTC"
    assert intent.is_buy is True
    assert intent.time_in_force == "Alo"
    assert intent.limit_px < precision.mid_px
    assert intent.notional_usdc <= executor.MAX_ORDER_NOTIONAL_USDC
    executor.validate_order_intent(executor.TinyLiveConfig(), precision, intent)


def test_executor_rounding_delegates_to_authoritative_price_math() -> None:
    assert executor.round_hyperliquid_perp_price(12.34567, 0) == cross_exchange_price_math.normalize_hl_perp_price(
        12.34567,
        sz_decimals=0,
        side="nearest",
    )


def test_validate_order_intent_rejects_invalid_price_precision() -> None:
    config = executor.TinyLiveConfig()
    precision = executor.PrecisionFacts(symbol="BTC", sz_decimals=5, tick_size=1.0, lot_size=0.00001, mid_px=65000.0, source="unit")
    intent = executor.OrderIntent(symbol="BTC", is_buy=True, size_btc=0.001, limit_px=65000.12)

    with pytest.raises(executor.ValidationError, match="invalid_limit_price"):
        executor.validate_order_intent(config, precision, intent)


def test_generate_self_test_artifacts(tmp_path: Path) -> None:
    manifest = executor.generate_self_test_artifacts(tmp_path)

    assert manifest["task_id"] == "0618T001"
    assert manifest["final_recommendation"] == executor.FINAL_RECOMMENDATION_READY
    assert manifest["executor_ready"] is True
    assert manifest["real_order_endpoint_called"] is False
    assert manifest["private_endpoint_called"] is False
    assert manifest["shutdown_proof_status"] == "pass"
    assert "live_mode=false" in manifest["live_mode_blocked_in_self_test_reason"]

    manifest_path = tmp_path / "executor_manifest.json"
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest

    with (tmp_path / "order_intent_audit.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["time_in_force"] == "Alo"
    assert rows[0]["endpoint_called"] == "false"

    required = [
        "run_intent_marker.json",
        "approved_config_snapshot.json",
        "environment_dependency_snapshot.json",
        "official_doc_recheck_summary.csv",
        "precision_tick_lot_snapshot.csv",
        "preflight_validation_summary.csv",
        "order_intent_audit.csv",
        "private_order_response_audit.json",
        "cancel_shutdown_proof.json",
        "max_loss_monitor_summary.json",
        "final_safety_summary.json",
        "sha256_manifest.csv",
    ]
    for name in required:
        assert (tmp_path / name).exists()
        assert (tmp_path / name).stat().st_size > 0


def test_generate_real_order_canary_can_disable_schedule_cancel(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(executor, "build_live_client_from_env", lambda: executor.MockHyperliquidClient())
    monkeypatch.setattr(executor, "fetch_live_precision", lambda client: executor.mock_precision())

    manifest = executor.generate_real_order_canary_artifacts(
        output_dir=tmp_path,
        use_schedule_cancel=False,
        canary_task_id="0618T007",
    )

    assert manifest["task_id"] == "0618T007"
    assert manifest["final_recommendation"] == executor.FINAL_RECOMMENDATION_CANARY_READY
    assert manifest["schedule_cancel_endpoint_called"] is False
    assert manifest["schedule_cancel_required"] is False
    assert manifest["use_schedule_cancel"] is False
    assert manifest["real_cancel_endpoint_called"] is True
    assert manifest["shutdown_proof_status"] == "pass"

    cancel_proof = json.loads((tmp_path / "cancel_shutdown_proof.json").read_text(encoding="utf-8"))
    assert cancel_proof["schedule_cancel_endpoint_called"] is False
    assert cancel_proof["proof_status"] == "pass"
