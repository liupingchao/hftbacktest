from __future__ import annotations

from pathlib import Path
import sys

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import cross_exchange_candidate_episode_merging as merging  # noqa: E402
import cross_exchange_liquidity_response_case_hierarchy as accepted_hierarchy  # noqa: E402


def _candidate(
    *,
    segment_id: str = "segment_0001",
    connection_epoch_id: str = "epoch_001",
    segment_end_ts_ns: int = 5_000_000_000,
    connection_epoch_end_ts_ns: int = 5_000_000_000,
    shock_ts_ns: int,
    direction_sign: int = 1,
    binance_pre_spread_px: float = 0.01,
    binance_pre_mid_px: float = 100.0,
    top5_impacted_qty: float = 15.0,
    top5_opposite_qty: float = 17.0,
    candidate_id: str | None = None,
) -> dict[str, object]:
    row = {
        "segment_id": segment_id,
        "connection_epoch_id": connection_epoch_id,
        "segment_end_ts_ns": segment_end_ts_ns,
        "connection_epoch_end_ts_ns": connection_epoch_end_ts_ns,
        "shock_ts_ns": shock_ts_ns,
        "direction_sign": direction_sign,
        "binance_pre_spread_px": binance_pre_spread_px,
        "binance_pre_mid_px": binance_pre_mid_px,
        "binance_pre_top5_impacted_qty": top5_impacted_qty,
        "binance_pre_top5_opposite_qty": top5_opposite_qty,
    }
    if candidate_id is not None:
        row["candidate_id"] = candidate_id
    return row


def _timeline_row(ts_ns: int, *, bid_px: float = 99.99, ask_px: float = 100.0, qty: float = 4.0) -> dict[str, str]:
    row = {
        "common_ts_ns": str(ts_ns),
        "binance_bid_1_px": str(bid_px),
        "binance_ask_1_px": str(ask_px),
    }
    for level in range(1, 6):
        row[f"binance_bid_{level}_qty"] = str(qty)
        row[f"binance_ask_{level}_qty"] = str(qty)
        if level > 1:
            row[f"binance_bid_{level}_px"] = str(bid_px - (level - 1) * 0.01)
            row[f"binance_ask_{level}_px"] = str(ask_px + (level - 1) * 0.01)
    return row


def test_recovery_wrapper_matches_accepted_semantics() -> None:
    timeline_rows = [
        _timeline_row(1_050_000_000, ask_px=100.0, qty=4.0),
        _timeline_row(1_110_000_000, ask_px=100.0, qty=4.0),
    ]
    normalized = merging._normalize_timeline_states(timeline_rows)
    expected = accepted_hierarchy._recovery_checkpoint(
        timeline=normalized,
        start_ns=1_000_000_000,
        end_ns=1_130_000_000,
        pre_spread_px=0.01,
        pre_depth=32.0,
        direction_sign=1,
        extreme_mid_px=99.995,
        recovery_span_ms=50,
        depth_recovery_ratio=0.8,
        spread_allowance_ticks=1,
    )
    observed = merging.candidate_recovery_checkpoint_v1(
        timeline=timeline_rows,
        start_ns=1_000_000_000,
        end_ns=1_130_000_000,
        pre_spread_px=0.01,
        pre_depth=32.0,
        direction_sign=1,
        extreme_mid_px=99.995,
    )
    assert observed == expected


def test_episode_merging_builds_cluster_flow_overlap_membership() -> None:
    candidates = [
        _candidate(shock_ts_ns=1_000_000_000),
        _candidate(shock_ts_ns=1_050_000_000),
        _candidate(shock_ts_ns=1_180_000_000),
        _candidate(shock_ts_ns=3_400_000_000),
    ]
    timeline_rows = [
        _timeline_row(1_070_000_000, ask_px=100.08, qty=1.0),
        _timeline_row(1_130_000_000, ask_px=100.08, qty=1.0),
    ]
    result = merging.episode_merging_v1(
        candidates,
        timelines_by_segment={"segment_0001": timeline_rows},
    )
    assert result["summary"] == {
        "version": "episode_merging_v1",
        "candidate_count": 4,
        "cluster_count": 3,
        "continuous_flow_episode_count": 2,
        "overlap_block_count_2000ms": 2,
        "segment_connection_epoch_count": 1,
    }
    assert len(result["membership"]) == 4
    assert len({row["cluster_id"] for row in result["membership"][:2]}) == 1
    assert len({row["continuous_flow_episode_id"] for row in result["membership"][:3]}) == 1
    assert len({row["overlap_block_id"] for row in result["membership"][:3]}) == 1
    assert result["membership"][-1]["overlap_block_id"] != result["membership"][0]["overlap_block_id"]


def test_segment_and_epoch_boundaries_terminate_cluster_episode_and_overlap_blocks() -> None:
    candidates = [
        _candidate(shock_ts_ns=1_000_000_000, candidate_id="a"),
        _candidate(shock_ts_ns=1_050_000_000, connection_epoch_id="epoch_002", candidate_id="b"),
        _candidate(shock_ts_ns=1_060_000_000, segment_id="segment_0002", connection_epoch_id="epoch_001", candidate_id="c"),
    ]
    result = merging.episode_merging_v1(candidates)
    membership = {row["candidate_id"]: row for row in result["membership"]}
    assert membership["a"]["cluster_id"] != membership["b"]["cluster_id"]
    assert membership["a"]["continuous_flow_episode_id"] != membership["b"]["continuous_flow_episode_id"]
    assert membership["a"]["overlap_block_id"] != membership["b"]["overlap_block_id"]
    assert membership["b"]["cluster_id"] != membership["c"]["cluster_id"]


def test_recovery_lookahead_after_right_cluster_does_not_bridge() -> None:
    candidates = [
        _candidate(shock_ts_ns=1_000_000_000, candidate_id="left"),
        _candidate(shock_ts_ns=1_130_000_000, candidate_id="right"),
    ]
    timeline_rows = [
        _timeline_row(1_090_000_000, ask_px=100.03, qty=1.0),
        _timeline_row(1_200_000_000, ask_px=100.0, qty=4.0),
        _timeline_row(1_260_000_000, ask_px=100.0, qty=4.0),
    ]
    result = merging.episode_merging_v1(
        candidates,
        timelines_by_segment={"segment_0001": timeline_rows},
    )
    assert result["boundary_audit"][0]["recovery_status"] == "missing_recovery_evidence"
    assert result["boundary_audit"][0]["merged"] is False
    assert len({row["continuous_flow_episode_id"] for row in result["membership"]}) == 2


def test_overlap_blocks_clip_to_epoch_end() -> None:
    candidates = [
        _candidate(
            shock_ts_ns=1_000_000_000,
            connection_epoch_end_ts_ns=1_500_000_000,
            candidate_id="left",
        ),
        _candidate(
            shock_ts_ns=1_600_000_000,
            connection_epoch_id="epoch_002",
            connection_epoch_end_ts_ns=5_000_000_000,
            candidate_id="right",
        ),
    ]
    result = merging.episode_merging_v1(candidates)
    membership = {row["candidate_id"]: row for row in result["membership"]}
    assert membership["left"]["window_end_ts_ns"] == 1_500_000_000
    assert membership["left"]["overlap_block_id"] != membership["right"]["overlap_block_id"]


def test_duplicate_candidate_ids_fail_closed() -> None:
    candidates = [
        _candidate(shock_ts_ns=1_000_000_000, candidate_id="dup"),
        _candidate(shock_ts_ns=1_050_000_000, candidate_id="dup"),
    ]
    with pytest.raises(merging.CandidateEpisodeMergingError, match="duplicate candidate id"):
        merging.episode_merging_v1(candidates)


def test_out_of_order_candidates_fail_closed() -> None:
    candidates = [
        _candidate(shock_ts_ns=1_050_000_000, candidate_id="later"),
        _candidate(shock_ts_ns=1_000_000_000, candidate_id="earlier"),
    ]
    with pytest.raises(merging.CandidateEpisodeMergingError, match="sorted by shock timestamp"):
        merging.episode_merging_v1(candidates)


def test_missing_required_contract_field_fails_closed() -> None:
    candidate = _candidate(shock_ts_ns=1_000_000_000)
    candidate.pop("connection_epoch_end_ts_ns")
    with pytest.raises(merging.CandidateEpisodeMergingError, match="candidate field is required"):
        merging.episode_merging_v1([candidate])


def test_conservation_helper_rejects_dropped_or_duplicated_membership() -> None:
    candidates = [
        _candidate(shock_ts_ns=1_000_000_000, candidate_id="a"),
        _candidate(shock_ts_ns=1_050_000_000, candidate_id="b"),
    ]
    result = merging.episode_merging_v1(candidates)
    dropped = result["membership"][:1]
    duplicated = result["membership"] + [result["membership"][0]]
    with pytest.raises(merging.CandidateEpisodeMergingError, match="does not conserve"):
        merging.assert_all_candidates_conserved(candidates, dropped)
    with pytest.raises(merging.CandidateEpisodeMergingError, match="does not conserve"):
        merging.assert_all_candidates_conserved(candidates, duplicated)


def test_same_timestamp_candidates_preserve_validated_source_order() -> None:
    candidates = [
        _candidate(shock_ts_ns=1_000_000_000, candidate_id="b"),
        _candidate(shock_ts_ns=1_000_000_000, candidate_id="a"),
    ]
    result = merging.episode_merging_v1(candidates)
    assert [row["candidate_id"] for row in result["membership"]] == ["b", "a"]


def test_generator_input_is_materialized_once_for_conservation() -> None:
    candidates = (
        _candidate(shock_ts_ns=shock_ts_ns, candidate_id=candidate_id)
        for shock_ts_ns, candidate_id in (
            (1_000_000_000, "a"),
            (1_050_000_000, "b"),
        )
    )
    result = merging.episode_merging_v1(candidates)
    assert [row["candidate_id"] for row in result["membership"]] == ["a", "b"]
