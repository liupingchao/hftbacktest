from __future__ import annotations

import json
from pathlib import Path

from examples.hyperliquid import cross_exchange_online_estimators as estimators
from examples.hyperliquid import hyperliquid_tiny_live_m2_public_watcher as watcher


def _attempt(
    attempt: int,
    *,
    side: str = "buy",
    quote_px: float = 100.0,
    size_btc: float = 0.01,
    status: str = "resting",
    order_called: bool = True,
    rejected: bool = False,
    cancel_called: bool = True,
    fair_mid_px: float = 101.0,
    reason: str = "",
) -> dict[str, object]:
    return {
        "attempt": attempt,
        "side": side,
        "limit_px": quote_px,
        "size_btc": size_btc,
        "fair_mid_px": fair_mid_px,
        "order_endpoint_called": order_called,
        "order_status_types": status,
        "post_only_reject": rejected,
        "cancel_endpoint_called": cancel_called,
        "quote_aging_guard_reason": reason,
    }


def _resting(attempt: int, *, end_ms: int = 8_000) -> dict[str, object]:
    return {
        "attempt": attempt,
        "side": "buy",
        "quote_px": 100.0,
        "size_btc": 0.01,
        "interval_start_ms": 1_000,
        "interval_end_ms": end_ms,
        "lifecycle_completeness_status": "proxy_bounded_interval_available",
    }


def _coverage(attempt: int) -> dict[str, object]:
    return {
        "attempt": attempt,
        "coverage_status": "complete_interval_trade_stream_coverage",
    }


def _fill(attempt: int, fill_id: str, qty: float, *, fingerprint: str | None = None) -> dict[str, object]:
    return {
        "attempt": attempt,
        "fill_id": fill_id,
        "qty_btc": qty,
        "fill_time_ms": 4_000,
        "attribution_status": "matched_tracked_oid",
        "fill_payload_fingerprint": fingerprint or f"fp-{fill_id}",
    }


def test_lifecycle_transition_censoring_and_partial_ratio() -> None:
    attempts = [
        _attempt(1),
        _attempt(2, status="error", rejected=True),
        _attempt(3, status="resting", cancel_called=False),
        _attempt(4, status="resting", reason="duration_elapsed"),
        _attempt(5, status="resting"),
    ]
    resting = [
        _resting(1),
        _resting(3, end_ms=3_000),
        _resting(4),
        _resting(5),
    ]
    rows, quarantine = estimators.normalize_fill_feedback_lifecycles(
        attempt_rows=attempts,
        resting_lifecycle_rows=resting,
        fill_rows=[_fill(1, "full-1", 0.01), _fill(5, "partial-1", 0.004)],
        public_coverage_rows=[_coverage(1), _coverage(3), _coverage(4), _coverage(5)],
        artifact_task_id="T020",
    )

    assert quarantine == []
    by_attempt = {int(row["attempt_id"]): row for row in rows}
    assert by_attempt[1]["observation_status"] == "included_observed_full_fill"
    assert by_attempt[1]["full_fill"] is True
    assert by_attempt[2]["observation_status"] == "excluded_rejected"
    assert by_attempt[3]["observation_status"] == "censored_short_hold"
    assert by_attempt[4]["observation_status"] == "censored_run_end"
    assert by_attempt[5]["observation_status"] == "included_observed_partial_fill"
    assert by_attempt[5]["fill_ratio"] == 0.4


def test_submitted_lifecycle_row_wins_over_no_submit_candidates() -> None:
    candidate = _attempt(
        1,
        status="skipped",
        order_called=False,
        rejected=False,
        cancel_called=False,
    )
    candidate["event_sequence"] = 10
    submitted = _attempt(
        1,
        status="error",
        order_called=True,
        rejected=False,
        cancel_called=False,
    )
    submitted["event_sequence"] = 20

    rows, quarantine = estimators.normalize_fill_feedback_lifecycles(
        attempt_rows=[candidate, submitted],
        resting_lifecycle_rows=[],
        fill_rows=[],
        artifact_task_id="0720T026",
    )

    assert quarantine == []
    assert len(rows) == 1
    assert rows[0]["submitted"] is True
    assert rows[0]["rejected"] is True
    assert rows[0]["terminal_status"] == "rejected"
    assert rows[0]["observation_status"] == "excluded_rejected"
    assert rows[0]["included_in_feedback"] is False
    assert rows[0]["integrity_status"] == "pass"


def test_conflicting_submitted_lifecycle_rows_remain_quarantined() -> None:
    first = _attempt(1, quote_px=100.0)
    second = _attempt(1, quote_px=101.0)

    rows, quarantine = estimators.normalize_fill_feedback_lifecycles(
        attempt_rows=[first, second],
        resting_lifecycle_rows=[_resting(1)],
        fill_rows=[],
        public_coverage_rows=[_coverage(1)],
        artifact_task_id="T020",
    )

    assert any(
        row["reason"]
        == "conflicting_duplicate_submitted_attempt_lifecycle"
        for row in quarantine
    )
    assert len(rows) == 1
    assert rows[0]["integrity_status"] == "fail_closed"
    assert rows[0]["observation_status"] == (
        "censored_integrity_conflict"
    )
    assert rows[0]["included_in_feedback"] is False


def test_duplicate_fill_is_idempotent_and_conflict_fails_closed() -> None:
    rows, quarantine = estimators.normalize_fill_feedback_lifecycles(
        attempt_rows=[_attempt(1)],
        resting_lifecycle_rows=[_resting(1)],
        fill_rows=[
            _fill(1, "same", 0.004),
            _fill(1, "same", 0.004),
            _fill(1, "same", 0.005, fingerprint="different"),
        ],
        public_coverage_rows=[_coverage(1)],
        artifact_task_id="T020",
    )

    assert any(row["reason"] == "conflicting_same_fill_identity" for row in quarantine)
    assert len(rows) == 1
    assert rows[0]["fill_identity_count"] == 0
    assert rows[0]["observation_status"] == "censored_integrity_conflict"
    assert rows[0]["included_in_feedback"] is False

    deduped_rows, deduped_quarantine = estimators.normalize_fill_feedback_lifecycles(
        attempt_rows=[_attempt(1)],
        resting_lifecycle_rows=[_resting(1)],
        fill_rows=[_fill(1, "same", 0.004), _fill(1, "same", 0.004)],
        public_coverage_rows=[_coverage(1)],
        artifact_task_id="T020",
    )
    assert deduped_quarantine == []
    assert deduped_rows[0]["filled_qty_btc"] == 0.004
    assert deduped_rows[0]["duplicate_fill_count"] == 1


def test_exposure_weighted_aggregation_and_neutral_until_target() -> None:
    rows, _ = estimators.normalize_fill_feedback_lifecycles(
        attempt_rows=[_attempt(1), _attempt(2)],
        resting_lifecycle_rows=[_resting(1), _resting(2)],
        fill_rows=[_fill(1, "full", 0.01)],
        public_coverage_rows=[_coverage(1), _coverage(2)],
        artifact_task_id="T020",
    )
    controller = estimators.ExposureWeightedFillFeedback(
        config=estimators.FillFeedbackConfig(
            target_fill_ratio=None,
            min_observations=1,
            min_exposure_seconds=0,
        )
    )
    controller.ingest_lifecycles(rows)
    aggregate = controller.aggregate()
    assert aggregate["included_observation_count"] == 2
    assert aggregate["exposure_weighted_fill_ratio"] == 0.5
    assert aggregate["public_arrival_rate_per_second"] == 0
    candidate = controller.snapshot(as_of_ms=8_000)["candidate"]
    assert candidate["status"] == "unavailable_neutral"
    assert candidate["reason"] == "target_fill_ratio_not_configured_from_live_evidence"
    assert candidate["activation_enabled"] is False


def test_feedback_hysteresis_rate_limit_anti_windup_and_restart_checksum() -> None:
    rows, _ = estimators.normalize_fill_feedback_lifecycles(
        attempt_rows=[_attempt(1)],
        resting_lifecycle_rows=[_resting(1)],
        fill_rows=[_fill(1, "full", 0.01)],
        public_coverage_rows=[_coverage(1)],
        artifact_task_id="T020",
    )
    config = estimators.FillFeedbackConfig(
        target_fill_ratio=0.5,
        min_observations=1,
        min_exposure_seconds=0,
        max_abs_offset_ticks=0.2,
        max_rate_ticks_per_second=0.1,
        hysteresis_ratio=0.01,
        proportional_gain=10.0,
        integral_gain=1.0,
        integral_limit=0.1,
    )
    controller = estimators.ExposureWeightedFillFeedback(config=config)
    controller.ingest_lifecycles(rows)
    first = controller.snapshot(as_of_ms=8_000)
    assert first["candidate"]["status"] == "pass_observe_only"
    assert first["candidate"]["bounded_offset_ticks"] == 0.2
    assert first["candidate"]["anti_windup_applied"] is True
    state = controller.state_envelope()

    restored = estimators.ExposureWeightedFillFeedback(config=config)
    assert restored.restore_state(state) is True
    assert restored.state_envelope() == state
    corrupted = dict(state)
    corrupted["checksum_sha256"] = "bad"
    rejected = estimators.ExposureWeightedFillFeedback(config=config)
    assert rejected.restore_state(corrupted) is False
    assert rejected.restore_status == "neutral"
    assert rejected.restore_reason == "state_checksum_mismatch"

    hysteresis_config = estimators.FillFeedbackConfig(
        target_fill_ratio=1.0,
        min_observations=1,
        min_exposure_seconds=0,
        hysteresis_ratio=0.01,
    )
    stable = estimators.ExposureWeightedFillFeedback(config=hysteresis_config)
    stable.ingest_lifecycles(rows)
    candidate = stable.snapshot(as_of_ms=8_000)["candidate"]
    assert candidate["hysteresis_applied"] is True
    assert candidate["bounded_offset_ticks"] == 0


def test_public_watcher_feedback_artifacts_and_fill_feedback_replay(tmp_path: Path) -> None:
    watcher.write_csv(
        tmp_path / "quote_attempt_matrix.csv",
        [_attempt(1)],
        watcher.inline_attempt_fieldnames(),
    )
    watcher.write_csv(
        tmp_path / "resting_interval_lifecycle_matrix.csv",
        [_resting(1)],
        watcher.resting_interval_lifecycle_fieldnames(),
    )
    watcher.write_csv(
        tmp_path / "public_stream_coverage.csv",
        [_coverage(1)],
        watcher.public_stream_coverage_fieldnames(),
    )
    watcher.write_csv(
        tmp_path / "live_fill_ledger.csv",
        [_fill(1, "full", 0.01)],
        [
            "attempt",
            "attempt_key",
            "fill_id",
            "side",
            "qty_btc",
            "attribution_status",
            "fill_payload_fingerprint",
            "fill_time_ms",
        ],
    )
    artifacts = watcher.write_fill_feedback_artifacts(
        output_dir=tmp_path,
        artifact_task_id="T020",
        target_fill_ratio=None,
    )
    assert artifacts["manifest"]["private_endpoint_called"] is False
    assert artifacts["snapshot"]["candidate"]["activation_enabled"] is False
    assert Path(artifacts["output_files"]["fill_feedback_snapshot"]).exists()

    replay_dir = tmp_path / "replay"
    manifest = estimators.build_fill_feedback_replay_artifacts(
        input_dir=tmp_path,
        output_dir=replay_dir,
    )
    assert manifest["snapshot_match"] is True
    replay_snapshot = json.loads(
        (replay_dir / "replay_fill_feedback_snapshot.json").read_text(encoding="utf-8")
    )
    source_snapshot = json.loads(
        (tmp_path / "fill_feedback_snapshot.json").read_text(encoding="utf-8")
    )
    assert replay_snapshot == source_snapshot
