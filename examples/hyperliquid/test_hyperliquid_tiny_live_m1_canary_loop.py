from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.hyperliquid import hyperliquid_tiny_live_m1_canary_loop as loop


def _write_window_artifacts(path: Path, *, schedule_cancel_called: bool = False) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "executor_manifest.json").write_text(
        json.dumps(
            {
                "final_recommendation": "hyperliquid_tiny_live_real_order_canary_ready_for_qa",
                "order_submission_attempted": True,
                "order_status_types": ["resting"],
                "private_endpoint_called": True,
                "real_order_endpoint_called": True,
                "real_cancel_endpoint_called": True,
                "schedule_cancel_endpoint_called": schedule_cancel_called,
                "schedule_cancel_required": False,
                "use_schedule_cancel": schedule_cancel_called,
                "shutdown_proof_status": "pass",
                "credentials_written": False,
                "secret_values_written": False,
                "raw_signatures_written": False,
                "max_loss_status": "pass",
                "git_commit": "abc1234",
            }
        ),
        encoding="utf-8",
    )
    (path / "cancel_shutdown_proof.json").write_text(
        json.dumps({"final_open_orders": [], "proof_status": "pass"}),
        encoding="utf-8",
    )


def test_validate_window_accepts_tracked_cancel_without_schedule_cancel(tmp_path: Path) -> None:
    _write_window_artifacts(tmp_path)

    row = loop.validate_window(tmp_path, 1)

    assert row["gate_status"] == "pass"
    assert row["schedule_cancel_endpoint_called"] is False
    assert row["final_open_orders_count"] == 0


def test_validate_window_rejects_schedule_cancel_usage(tmp_path: Path) -> None:
    _write_window_artifacts(tmp_path, schedule_cancel_called=True)

    with pytest.raises(loop.LoopError, match="schedule_cancel_was_used"):
        loop.validate_window(tmp_path, 1)
