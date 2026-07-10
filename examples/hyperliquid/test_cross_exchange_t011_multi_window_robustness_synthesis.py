from __future__ import annotations

import csv
import json
from pathlib import Path

from examples.hyperliquid import cross_exchange_t011_multi_window_robustness_synthesis as synth


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _matrix_rows(*, replay_fail: bool = False) -> list[dict]:
    base = {
        "source_kind": "t011_live_window_artifact",
        "market_view_acceptance": "pass",
        "decision_path_acceptance": "pass",
        "lifecycle_acceptance": "pass",
        "economics_acceptance": "pass",
        "optimism_acceptance": "pass",
        "boundary_acceptance": "pass",
        "overall_acceptance": "pass",
        "l2book_messages": "10",
        "trades_messages": "5",
        "reconnect_count": "0",
        "current_candidate_count": "10",
        "trigger_count": "1",
        "anti_drift_pass_count": "2",
        "anti_drift_block_count": "0",
        "edge_gate_pass_count": "1",
        "edge_gate_block_count": "0",
        "live_submissions_count": "1",
        "max_real_order_submissions": "2",
        "max_order_size_btc": "0.005",
        "post_only_reject_count": "0",
        "fill_count": "0",
        "maker_fill_count": "0",
        "ledger_fill_rows": "0",
        "final_open_orders_count": "0",
        "independent_final_open_orders_count": "0",
        "shutdown_proof_status": "pass",
    }
    rows = []
    for idx, classification in enumerate(["submitted_resting_no_fill", "submitted_resting_no_fill", "submitted_rejected"], start=1):
        row = dict(base)
        row.update({"window_id": f"w{idx}", "live_classification": classification, "order_status_types": "resting" if classification.endswith("no_fill") else "error"})
        rows.append(row)
    if replay_fail:
        rows[0]["overall_acceptance"] = "fail"
    return rows


def _make_inputs(tmp_path: Path, *, replay_fail: bool = False) -> tuple[Path, Path]:
    t001 = tmp_path / "t001.json"
    _write_json(t001, {"boundary_pass": True})
    t002 = tmp_path / "t002"
    _write_json(t002 / "aggregate_replay_acceptance_summary.json", {"final_recommendation": "batch_same_window_replay_acceptance_passed"})
    rows = _matrix_rows(replay_fail=replay_fail)
    _write_csv(t002 / "batch_replay_acceptance_matrix.csv", rows, list(rows[0].keys()))
    return t001, t002


def test_synthesis_routes_multiple_no_fill_windows_to_quote_fill_probability(tmp_path: Path) -> None:
    t001, t002 = _make_inputs(tmp_path)

    manifest = synth.run_synthesis(t001_summary=t001, t002_dir=t002, output_dir=tmp_path / "out")

    assert manifest["final_recommendation"] == "route_to_quote_fill_probability_evidence"
    assert manifest["accepted_window_count"] == 3


def test_synthesis_routes_replay_failure_to_repair(tmp_path: Path) -> None:
    t001, t002 = _make_inputs(tmp_path, replay_fail=True)

    manifest = synth.run_synthesis(t001_summary=t001, t002_dir=t002, output_dir=tmp_path / "out")

    assert manifest["final_recommendation"] == "route_to_replay_repair"
