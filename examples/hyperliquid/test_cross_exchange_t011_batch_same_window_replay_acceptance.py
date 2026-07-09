from __future__ import annotations

import csv
import json
from pathlib import Path

from examples.hyperliquid import cross_exchange_t011_batch_same_window_replay_acceptance as batch


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_qa(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("任务ID：\n- 0708T002\n\n状态：\n- 已通过\n", encoding="utf-8")


def _make_window(root: Path, idx: int, *, status: str = "resting", submissions: int = 1, independent_open: int = 0) -> Path:
    w = root / f"window_{idx:02d}"
    _write_json(
        w / "event_driven_watcher_manifest.json",
        {
            "hyperliquid_l2book_fast": True,
            "trigger_found": True,
            "trigger_count": 1,
            "edge_gate_live_compatible_source_available": True,
            "current_candidate_count": 10,
            "anti_drift_pass_count": 2,
            "anti_drift_block_count": 0,
            "edge_gate_pass_count": 1,
            "edge_gate_block_count": 0,
            "live_submissions_count": submissions,
            "max_real_order_submissions": 2,
            "max_order_size_btc": 0.005,
        },
    )
    _write_json(
        w / "inline_reprice_manifest.json",
        {
            "real_order_endpoint_called": submissions > 0,
            "real_cancel_endpoint_called": submissions > 0,
            "shutdown_proof_status": "pass",
            "order_status_types": [status] if submissions else [],
            "post_only_reject_count": 1 if status == "error" else 0,
            "fill_count": 0,
            "maker_fill_count": 0,
            "final_open_orders_count": 0,
            "max_order_size_btc": 0.005,
        },
    )
    _write_json(
        w / "private_order_response_audit.json",
        {"order_status_rows": [{"status_type": status}] if submissions else [], "order_submission_attempted": submissions > 0},
    )
    _write_json(w / "cancel_shutdown_proof.json", {"proof_status": "pass", "final_open_orders": []})
    _write_json(
        w / "public_stream_summary.json",
        {"message_count_by_channel": {"l2Book": 3, "trades": 2}, "reconnect_count": 0},
    )
    _write_json(w / "independent_final_open_orders_check.json", {"final_open_orders_count": independent_open})
    _write_csv(
        w / "order_intent_audit.csv",
        [{"time_in_force": "Alo", "size_btc": "0.005"}] if submissions else [],
        ["time_in_force", "size_btc"],
    )
    _write_csv(w / "live_fill_ledger.csv", [], ["source_window", "fill_id"])
    return w


def _make_root(root: Path, *, independent_open: int = 0) -> Path:
    for idx, status in [(1, "error"), (2, "resting"), (3, "resting")]:
        _make_window(root, idx, status=status, independent_open=independent_open if idx == 2 else 0)
    return root


def test_batch_acceptance_accepts_rejected_and_resting_no_fill_windows(tmp_path: Path) -> None:
    t011 = _make_root(tmp_path / "t011")
    qa = tmp_path / "0708T002-qa.md"
    _write_qa(qa)

    manifest = batch.run_batch_acceptance(t011_root=t011, qa_0708t002=qa, output_dir=tmp_path / "out")

    assert manifest["final_recommendation"] == batch.PASSED_RECOMMENDATION
    assert manifest["window_count"] == 4
    assert manifest["classification_counts"]["submitted_rejected"] == 1
    assert manifest["classification_counts"]["submitted_resting_no_fill"] == 3
    assert manifest["overall_acceptance_counts"] == {"pass": 4}


def test_batch_acceptance_blocks_nonzero_independent_open_orders(tmp_path: Path) -> None:
    t011 = _make_root(tmp_path / "t011", independent_open=1)
    qa = tmp_path / "0708T002-qa.md"
    _write_qa(qa)

    manifest = batch.run_batch_acceptance(t011_root=t011, qa_0708t002=qa, output_dir=tmp_path / "out")

    assert manifest["final_recommendation"] == batch.BLOCKED_RECOMMENDATION
    assert "0709T001_window_02" in manifest["blocked_window_ids"]
