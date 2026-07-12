from __future__ import annotations

import csv
import json
from pathlib import Path

from examples.hyperliquid import cross_exchange_quote_fill_probability_evidence as qfp


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _make_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    t011 = tmp_path / "t011"
    t002 = tmp_path / "t002"
    t003 = tmp_path / "t003"
    _write_json(t002 / "aggregate_replay_acceptance_summary.json", {"final_recommendation": "batch_same_window_replay_acceptance_passed"})
    _write_csv(
        t002 / "batch_replay_acceptance_matrix.csv",
        [
            {
                "window_id": "0708T001",
                "source_kind": "prior_accepted_replay_reference",
                "source_path": ".workflow/reports/0708T002-qa.md",
                "live_classification": "submitted_resting_no_fill",
                "order_status_types": "resting",
                "max_order_size_btc": "0.002",
                "real_order_endpoint_called": "true",
            },
            {
                "window_id": "0709T001_window_02",
                "source_kind": "t011_live_window_artifact",
                "source_path": "window_02",
                "live_classification": "submitted_resting_no_fill",
                "order_status_types": "resting",
                "max_order_size_btc": "0.005",
                "real_order_endpoint_called": "true",
            },
        ],
    )
    _write_json(t003 / "multi_window_synthesis_manifest.json", {"final_recommendation": "route_to_quote_fill_probability_evidence"})
    window = t011 / "window_02"
    _write_csv(
        window / "quote_attempt_matrix.csv",
        [
            {
                "attempt": "1",
                "event_sequence": "701",
                "source_channel": "trades",
                "source_event_exchange_time_ms": "1783580109291",
                "guard_status": "pass",
                "quote_px": "62851.0",
                "side": "buy",
                "limit_px": "62851.0",
                "size_btc": "0.005",
                "post_only_tif": "Alo",
                "order_endpoint_called": "True",
                "order_status_types": "resting",
                "post_only_reject": "False",
                "fill_count_after_attempt": "0",
                "maker_fill_count_after_attempt": "0",
            }
        ],
    )
    _write_csv(
        window / "inline_reprice_guard_matrix.csv",
        [
            {
                "attempt": "1",
                "status": "pass",
                "selected_quote_px": "62851.0",
                "selected_side": "buy",
                "selected_size_btc": "0.005",
                "current_bid": "62851.0",
                "current_ask": "62852.0",
                "current_same_side_top_qty_btc": "0.00351",
                "current_same_side_top_order_count": "1",
                "current_top_depth_multiple_of_order": "0.702",
                "post_only_non_crossing": "True",
                "current_touch_match": "True",
            }
        ],
    )
    _write_csv(
        window / "current_candidate_audit.csv",
        [
            {
                "event_sequence": "701",
                "rolling_trade_count_last_3s": "13",
                "touch_trade_qty_btc": "0.02801",
                "strict_trade_through_qty_btc": "0.04634",
                "at_or_through_trade_qty_btc": "0.07435",
                "required_depletion_qty_btc": "0.02642",
                "queue_depletion_multiple": "2.814",
                "public_depletion_status": "depleted_top_plus_order_proxy",
                "inference_scope": "rolling_proxy_not_exact_queue_or_fill_probability",
            }
        ],
    )
    _write_csv(
        window / "quote_aging_guard_matrix.csv",
        [
            {
                "attempt": "1",
                "status": "pass",
                "reason": "",
                "hold_elapsed_seconds": "3.008385",
            }
        ],
    )
    _write_json(
        window / "market_markout_snapshot.json",
        {
            "pre_submit_current_l2": {"levels": [[{"px": "62851", "sz": "0.00351", "n": 1}], [{"px": "62852", "sz": "29.0", "n": 128}]]},
            "post_submit_current_l2": {"levels": [[{"px": "62851", "sz": "0.00351", "n": 1}], [{"px": "62852", "sz": "29.0", "n": 128}]]},
        },
    )
    _write_csv(window / "inline_reprice_post_only_reject_matrix.csv", [], ["attempt", "reject_reason"])
    return t011, t002, t003


def test_analysis_routes_to_public_flow_artifact_repair_for_interval_gap(tmp_path: Path) -> None:
    t011, t002, t003 = _make_inputs(tmp_path)

    manifest = qfp.run_analysis(t011_root=t011, t002_dir=t002, t003_dir=t003, output_dir=tmp_path / "out")

    assert manifest["attempt_count"] == 2
    assert manifest["accepted_source_row_count"] == 2
    assert manifest["prior_reference_count"] == 1
    assert manifest["live_artifact_attempt_count"] == 1
    assert manifest["final_recommendation"] == "route_to_public_flow_artifact_repair"
    assert manifest["trade_through_status_counts"]["public_flow_artifact_missing"] == 1
    assert manifest["trade_through_status_counts"]["rolling_proxy_present_resting_interval_missing"] == 1


def test_post_only_reject_is_not_a_fill_probability_sample() -> None:
    quote_row = {"attempt": "1", "post_only_reject": "True"}
    reject_rows = [{"attempt": "1", "reject_reason": "Post only order would have immediately matched, bbo was 1@2"}]

    assert qfp.reject_interpretation(quote_row, reject_rows) == "consistent_post_only_protection"
    assert qfp.no_fill_state({"post_only_reject": "True"}) == "not_resting_rejected"
