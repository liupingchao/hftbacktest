from __future__ import annotations

import gzip
import os
from array import array
from dataclasses import replace
from pathlib import Path

import pytest

import cross_exchange_trigger_aligned_episodes as episode_v3


TREE_BOUNDARIES = (
    ("directory_inventory", episode_v3._directory_inventory),
    ("artifact_records", episode_v3._artifact_records),
    ("artifact_closure", episode_v3._verify_artifact_closure),
)


def candidate(*, primary: bool = True, segment_end_delta_ns: int = 5_000_000_000):
    shock = 10_000_000_000
    decision = shock + 50_000_000 if primary else shock + 60_000_000
    return episode_v3.Candidate(
        candidate_id="jul30:segment_0001:1",
        episode_id="jul30:segment_0001:1",
        segment_id="segment_0001",
        candidate_seq=1,
        aggressor_side="buy",
        direction_sign=1,
        burst_start_ts_ns=shock - 1_000_000,
        burst_end_ts_ns=shock + 1_000_000,
        burst_duration_ms=2.0,
        burst_trade_count=2,
        burst_trade_qty=2.0,
        touch_trade_qty=2.0,
        touch_trade_qty_at_shock=1.0,
        touch_trade_qty_through_decision=2.0,
        post_decision_burst_trade_count=0,
        pre_state_ts_ns=shock - 2_000_000,
        pre_best_px=100.0,
        pre_best_qty=10.0,
        shock_ts_ns=shock,
        impact_ratio=0.5,
        shock_impact_ratio=0.3,
        detector_decision_ts_ns=decision,
        confirmation_lag_ms=50.0,
        confirmed_best_px=101.0,
        confirmed_best_qty=5.0,
        price_level_depleted=True,
        queue_drop_ratio=0.5,
        confirmed_removed_qty=5.0,
        trade_explained_ratio=1.0,
        attribution="trade_driven",
        pre_hl_bbo_ts_ns=shock - 3_000_000,
        pre_hl_bbo_age_ms=3.0,
        pre_hl_fast_source_ts_ns=shock - 3_000_000,
        pre_hl_fast_age_ms=3.0,
        primary_episode=primary,
        rejection_reason="" if primary else "same_direction_dedup_50ms",
        connection_epoch_id="0",
        segment_start_ts_ns=shock - 5_000_000_000,
        segment_end_ts_ns=shock + segment_end_delta_ns,
        cluster_id="segment_0001-0-C000001",
        continuous_flow_episode_id="segment_0001-0-E000001",
        overlap_block_id="segment_0001-0-B000001",
        window_end_ts_ns=shock + min(segment_end_delta_ns, 2_000_000_000),
    )


def numeric_store(channel: str, rows: list[tuple[int, float, float, float, float, int]]):
    store = episode_v3.NumericEventStore("segment_0001", channel)
    for seq, (ts, x1, x2, x3, x4, side) in enumerate(rows, start=1):
        store.ts.append(ts)
        store.seq.append(seq)
        store.raw_seq.append(seq + 100)
        store.item_index.append(0)
        store.exchange_ts.append(ts - 1)
        store.x1.append(x1)
        store.x2.append(x2)
        store.x3.append(x3)
        store.x4.append(x4)
        store.side.append(side)
    return store


def timeline() -> episode_v3.TimelineStore:
    store = episode_v3.TimelineStore("segment_0001")
    for seq, ts in enumerate(
        (9_997_000_000, 10_000_000_000, 10_100_000_000, 12_000_000_000),
        start=1,
    ):
        store.ts.append(ts)
        store.common_seq.append(seq)
        store.binance_observed_at.append(ts)
        store.binance_bid.append(100.0 + seq)
        store.binance_ask.append(101.0 + seq)
        store.binance_bid_qty.append(10.0)
        store.binance_ask_qty.append(9.0)
        store.binance_bid_top5.append(50.0)
        store.binance_ask_top5.append(45.0)
        store.hl_fast_observed_at.append(ts - 1000)
        store.hl_fast_bid.append(99.0 + seq)
        store.hl_fast_ask.append(100.0 + seq)
        store.hl_fast_bid_qty.append(20.0)
        store.hl_fast_ask_qty.append(18.0)
        store.hl_fast_bid_top5.append(100.0)
        store.hl_fast_ask_top5.append(90.0)
        store.trigger_track.append("hyperliquid_fast")
        store.fast_event_ts.append(ts - 1000)
        store.fast_event_version.append(ts - 1000)
        store.midpoint_sq_return_prefix.append(
            store.midpoint_sq_return_prefix[-1] + 0.0001
        )
    return store


def segment_data(cand, *, adverse: bool = True):
    hl_bbo_rows = [
        (9_999_000_000, 99.0, 10.0, 100.0, 10.0, 0),
        (
            10_100_000_000,
            100.0 if adverse else 99.0,
            10.0,
            101.0 if adverse else 100.0,
            10.0,
            0,
        ),
        (12_000_000_000, 101.0, 10.0, 102.0, 10.0, 0),
    ]
    hl_trade_rows = [
        (10_050_000_000, 99.5, 1.0, 0.0, 0.0, 1),
        (10_150_000_000, 100.5, 1.0, 0.0, 0.0, 1),
    ]
    bin_bbo = numeric_store(
        "binance_book_ticker",
        [(10_000_000_000, 101.0, 10.0, 102.0, 9.0, 0)],
    )
    bin_trades = numeric_store(
        "binance_trade",
        [
            (9_999_000_000, 100.0, 1.0, 0.0, 0.0, 1),
            (10_000_000_000, 100.0, 1.0, 0.0, 0.0, 1),
        ],
    )
    return episode_v3.SegmentData(
        spec=episode_v3.SegmentSpec(
            segment_id="segment_0001",
            manifest_path=Path("manifest"),
            binance_hot_path=Path("bin"),
            hyperliquid_hot_path=Path("hl"),
            auxiliary_path=Path("aux"),
            timeline_path=Path("timeline"),
            decision_label_path=Path("labels"),
            segment_start_ts_ns=cand.segment_start_ts_ns,
            segment_end_ts_ns=cand.segment_end_ts_ns,
            expected_rows={},
            expected_shas={},
        ),
        timeline=timeline(),
        binance_bbo=bin_bbo,
        binance_trades=bin_trades,
        hl_bbo=numeric_store("hyperliquid_bbo", hl_bbo_rows),
        hl_trades=numeric_store("hyperliquid_trade", hl_trade_rows),
        auxiliary_intervals=[],
        binance_signed_qty_prefix=array("d", [0.0, 1.0, 2.0]),
        binance_ofi_prefix=array("d", [0.0, 0.0]),
        candidate_ts=array("q", [cand.shock_ts_ns]),
        candidate_direction=array("b", [cand.direction_sign]),
    )


def test_rejected_anchor_has_null_confirm():
    cand = candidate(primary=False)
    row = episode_v3._anchor_row(
        cand, r0_manifest_sha256="a" * 64, auxiliary_ids=()
    )
    assert row["t_confirm_ns"] == ""
    assert row["detector_decision_ts_ns"] != ""
    assert row["family_a_available"] == "true"
    assert row["family_b_available"] == "false"


def test_feature_observation_after_landmark_fails_closed():
    cand = candidate()
    with pytest.raises(episode_v3.EpisodeV3Error, match="future observation"):
        episode_v3._feature_row(
            cand,
            family_view="family_a",
            decision_landmark="t_candidate",
            landmark_ts_ns=cand.shock_ts_ns,
            feature_name="leak",
            value=1.0,
            observed_at_ns=cand.shock_ts_ns + 1,
            source_event_id="event",
            source_book_version="1",
            availability_reason="available",
        )


def test_unavailable_feature_cannot_be_zero_filled():
    cand = candidate()
    row = episode_v3._feature_row(
        cand,
        family_view="family_a",
        decision_landmark="t_candidate",
        landmark_ts_ns=cand.shock_ts_ns,
        feature_name="basis",
        value=None,
        observed_at_ns=None,
        source_event_id="",
        source_book_version="",
        availability_reason="basis_state_not_admitted_in_stage4",
    )
    assert row["value"] == ""
    assert row["availability_reason"] != "available"


def test_grid_marks_forward_fill_as_no_new_information():
    cand = candidate()
    data = segment_data(cand)
    row = episode_v3._grid_row(
        cand,
        data,
        family_view="family_a",
        decision_landmark="t_candidate",
        landmark_ts_ns=cand.shock_ts_ns,
        relative_ms=50,
    )
    assert row["availability_reason"] == "available"
    assert row["binance_no_new_information"] == "true"
    assert row["hyperliquid_bbo_no_new_information"] == "true"
    assert row["risk_gap_bps"] == row["d_bh_bps"]


def test_first_event_is_interval_not_point():
    cand = candidate()
    row, counts = episode_v3._outcome_row(cand, segment_data(cand))
    name = "time_to_first_adverse_target_bbo_event"
    assert row[f"{name}_status"] == "interval_censored"
    assert int(row[f"{name}_interval_lower_ns"]) == cand.shock_ts_ns
    assert int(row[f"{name}_interval_lower_ns"]) < int(
        row[f"{name}_interval_upper_ns"]
    )
    assert counts["point_coerced"] == 0


def test_same_receipt_trade_items_share_one_observation_interval():
    cand = candidate()
    data = segment_data(cand)
    data.hl_trades = numeric_store(
        "hyperliquid_trade",
        [
            (10_050_000_000, 99.0, 1.0, 0.0, 0.0, -1),
            (10_050_000_000, 100.5, 1.0, 0.0, 0.0, 1),
        ],
    )
    row, _ = episode_v3._outcome_row(cand, data)
    name = "time_to_first_target_trade_at_or_through_vulnerable_pretrigger_quote"
    assert row[f"{name}_status"] == "interval_censored"
    assert int(row[f"{name}_interval_lower_ns"]) == cand.shock_ts_ns
    assert int(row[f"{name}_interval_lower_ns"]) < int(
        row[f"{name}_interval_upper_ns"]
    )


def test_outcome_baseline_is_strict_pre_at_equal_receipt_timestamp():
    cand = candidate()
    data = segment_data(cand)
    data.hl_bbo = numeric_store(
        "hyperliquid_bbo",
        [
            (9_999_000_000, 99.0, 10.0, 100.0, 10.0, 0),
            (10_000_000_000, 109.0, 10.0, 110.0, 10.0, 0),
            (10_100_000_000, 100.0, 10.0, 101.0, 10.0, 0),
            (12_000_000_000, 101.0, 10.0, 102.0, 10.0, 0),
        ],
    )
    row, _ = episode_v3._outcome_row(cand, data)
    assert float(row["target_midpoint_markout_250ms_bps"]) > 0
    name = "time_to_first_adverse_target_bbo_event"
    assert row[f"{name}_status"] == "interval_censored"
    assert int(row[f"{name}_interval_lower_ns"]) == cand.shock_ts_ns


def test_confirmation_features_use_detector_state_and_frozen_burst():
    cand = replace(
        candidate(),
        burst_start_ts_ns=9_999_000_000,
        burst_end_ts_ns=10_000_000_000,
        burst_duration_ms=1.0,
        burst_trade_count=2,
        burst_trade_qty=2.0,
        touch_trade_qty=2.0,
        touch_trade_qty_at_shock=2.0,
        touch_trade_qty_through_decision=2.0,
        pre_state_ts_ns=9_998_000_000,
        pre_best_px=100.0,
        pre_best_qty=10.0,
        shock_impact_ratio=0.2,
    )
    data = segment_data(cand)
    data.binance_trades = numeric_store(
        "binance_trade",
        [
            (9_999_000_000, 100.0, 1.0, 0.0, 0.0, 1),
            (10_000_000_000, 100.0, 1.0, 0.0, 0.0, 1),
            (10_002_000_000, 99.0, 1.0, 0.0, 0.0, -1),
            (10_003_000_000, 100.0, 1.0, 0.0, 0.0, 1),
        ],
    )
    detector_state = episode_v3.TimelineStore("segment_0001")
    for seq, (ts_ns, ask_px, ask_qty) in enumerate(
        (
            (9_998_000_000, 100.0, 10.0),
            (10_000_000_000, 100.0, 10.0),
            (10_050_000_000, 101.0, 5.0),
        ),
        start=1,
    ):
        detector_state.ts.append(ts_ns)
        detector_state.common_seq.append(seq)
        detector_state.binance_observed_at.append(ts_ns)
        detector_state.binance_bid.append(99.0)
        detector_state.binance_ask.append(ask_px)
        detector_state.binance_bid_qty.append(10.0)
        detector_state.binance_ask_qty.append(ask_qty)
        detector_state.binance_bid_top5.append(50.0)
        detector_state.binance_ask_top5.append(50.0)
        detector_state.hl_fast_observed_at.append(ts_ns - 1)
        detector_state.hl_fast_bid.append(99.0)
        detector_state.hl_fast_ask.append(100.0)
        detector_state.hl_fast_bid_qty.append(10.0)
        detector_state.hl_fast_ask_qty.append(10.0)
        detector_state.hl_fast_bid_top5.append(50.0)
        detector_state.hl_fast_ask_top5.append(50.0)
        detector_state.trigger_track.append("binance_depth")
        detector_state.fast_event_ts.append(ts_ns - 1)
        detector_state.fast_event_version.append(ts_ns - 1)
        detector_state.midpoint_sq_return_prefix.append(0.0)
    data.timeline = detector_state
    rows = {
        row["feature_name"]: row
        for row in episode_v3._confirmation_feature_rows(cand, data)
    }
    count = rows["confirmed_burst_trade_count_through_decision"]
    assert count["value"] == "2"
    assert count["observed_at_ns"] == str(cand.t_confirm_ns)
    assert count["source_event_id"].endswith("common_l2_timeline:3")
    assert count["source_book_version"] == "3"
    assert rows["post_candidate_trade_continuation_count"]["value"] == "0"


def test_segment_end_censors_unobserved_event():
    cand = candidate(segment_end_delta_ns=80_000_000)
    row, counts = episode_v3._outcome_row(
        cand, segment_data(cand, adverse=False)
    )
    assert row["outcome_horizon_status"] == "segment_censored"
    assert counts["segment_censored"] >= 1
    assert row["public_quote_risk_availability"] == "segment_censored"
    assert row["public_quote_survives_horizon"] == ""


def test_event_count_uses_distinct_source_events():
    cand = candidate()
    rows = list(
        episode_v3._event_count_rows(
            cand,
            segment_data(cand),
            family_view="family_a",
            decision_landmark="t_candidate",
            landmark_ts_ns=cand.shock_ts_ns,
        )
    )
    bbo = next(row for row in rows if row["channel"] == "hyperliquid_bbo")
    assert bbo["event_1_source_event_id"]
    assert bbo["event_2_source_event_id"]
    assert bbo["event_1_source_event_id"] != bbo["event_2_source_event_id"]


def test_deterministic_gzip_has_zero_mtime(tmp_path):
    path = tmp_path / "rows.csv.gz"
    episode_v3._write_csv(path, [{"a": "1"}], ("a",))
    raw = path.read_bytes()
    assert raw[4:8] == b"\0\0\0\0"
    with gzip.open(path, "rt") as fh:
        assert fh.read() == "a\n1\n"


def _package_tree_with_unsupported_entry(tmp_path, entry_kind):
    package = tmp_path / "package"
    package.mkdir()
    if entry_kind == "dangling_symlink":
        (package / "unexpected-link").symlink_to(tmp_path / "missing-target")
    elif entry_kind == "file_symlink":
        target = tmp_path / "existing-target.txt"
        target.write_text("target\n", encoding="utf-8")
        (package / "unexpected-link").symlink_to(target)
    elif entry_kind == "directory_symlink":
        target = tmp_path / "existing-target-directory"
        target.mkdir()
        (package / "unexpected-link").symlink_to(
            target,
            target_is_directory=True,
        )
    elif entry_kind == "fifo":
        os.mkfifo(package / "unexpected-fifo")
    else:  # pragma: no cover - parameter list is exhaustive
        raise AssertionError(entry_kind)
    return package


@pytest.mark.parametrize("boundary_name,boundary", TREE_BOUNDARIES)
@pytest.mark.parametrize(
    "entry_kind",
    (
        "dangling_symlink",
        "file_symlink",
        "directory_symlink",
        "fifo",
    ),
)
def test_package_tree_boundaries_reject_unsupported_descendant_entry(
    tmp_path,
    boundary_name,
    boundary,
    entry_kind,
):
    package = _package_tree_with_unsupported_entry(tmp_path, entry_kind)
    with pytest.raises(
        episode_v3.EpisodeV3Error,
        match="package tree entry type forbidden",
    ):
        boundary(package)


@pytest.mark.parametrize("boundary_name,boundary", TREE_BOUNDARIES)
def test_package_tree_boundaries_reject_root_symlink(
    tmp_path,
    boundary_name,
    boundary,
):
    real_package = tmp_path / "real-package"
    real_package.mkdir()
    package_link = tmp_path / "package-link"
    package_link.symlink_to(real_package, target_is_directory=True)
    with pytest.raises(
        episode_v3.EpisodeV3Error,
        match="tree root must be a real directory",
    ):
        boundary(package_link)


def test_exact_tree_inventory_accepts_regular_file_directory_tree(tmp_path):
    root = tmp_path / "package"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "root.txt").write_text("root\n", encoding="utf-8")
    (nested / "child.txt").write_text("child\n", encoding="utf-8")

    assert episode_v3._tree_entry_type_contract() == {
        "root_type": "real_directory",
        "descendant_allowed_types": ["regular_file", "directory"],
        "classification": "lstat with stat.S_ISREG/stat.S_ISDIR",
        "symlink_target_following": False,
        "forbidden_types": [
            "symlink",
            "fifo",
            "socket",
            "character_device",
            "block_device",
            "other_special",
        ],
        "closure_before_manifest_or_identity": True,
    }
    entries = episode_v3._exact_tree_entries(root)
    assert [
        (entry.relative_path, entry.entry_type)
        for entry in entries
    ] == [
        ("nested", "directory"),
        ("nested/child.txt", "regular_file"),
        ("root.txt", "regular_file"),
    ]
    inventory = episode_v3._directory_inventory(root)
    assert [row["path"] for row in inventory] == [
        "nested/child.txt",
        "root.txt",
    ]
    assert [row["bytes"] for row in inventory] == [6, 5]


def test_forbidden_later_session_path_fails_closed():
    with pytest.raises(episode_v3.EpisodeV3Error, match="forbidden"):
        episode_v3._guard_allowed_path(
            Path("/tmp/0807T001_skhynix_4h_continuous/raw.gz")
        )


def test_source_semantic_digest_rejects_coherent_payload_rehash():
    expected = episode_v3.RowStreamDigest()
    observed = episode_v3.RowStreamDigest()
    expected.update({"candidate_id": "c1", "classification": "trade_driven"})
    observed.update({"candidate_id": "c1", "classification": "arbitrary_relabel"})
    with pytest.raises(episode_v3.EpisodeV3Error, match="source-semantic drift"):
        episode_v3._assert_row_stream_identity(
            label="anchors",
            expected=expected,
            observed=observed,
        )


@pytest.mark.parametrize("field_name", episode_v3.FEATURE_FIELDS)
def test_exact_feature_projection_compares_every_field(field_name):
    expected = {
        name: f"expected-{name}" for name in episode_v3.FEATURE_FIELDS
    }
    observed = dict(expected)
    observed[field_name] = f"mutated-{field_name}"
    with pytest.raises(
        episode_v3.EpisodeV3Error,
        match=r"features_family_a source-semantic drift.*fields=",
    ):
        episode_v3._verify_exact_row_stream(
            label="features_family_a",
            expected_rows=[expected],
            observed_rows=[observed],
            fields=episode_v3.FEATURE_FIELDS,
        )


@pytest.mark.parametrize("field_name", episode_v3.VIEW_FIELDS)
def test_exact_view_projection_compares_every_field(field_name):
    expected = {
        name: f"expected-{name}" for name in episode_v3.VIEW_FIELDS
    }
    observed = dict(expected)
    observed[field_name] = f"mutated-{field_name}"
    with pytest.raises(
        episode_v3.EpisodeV3Error,
        match=r"views_family_b source-semantic drift.*fields=",
    ):
        episode_v3._verify_exact_row_stream(
            label="views_family_b",
            expected_rows=[expected],
            observed_rows=[observed],
            fields=episode_v3.VIEW_FIELDS,
        )


def test_exact_projection_rejects_missing_or_extra_rows():
    row = {name: f"value-{name}" for name in episode_v3.FEATURE_FIELDS}
    with pytest.raises(
        episode_v3.EpisodeV3Error,
        match="observed_row_missing=True",
    ):
        episode_v3._verify_exact_row_stream(
            label="features_family_a",
            expected_rows=[row],
            observed_rows=[],
            fields=episode_v3.FEATURE_FIELDS,
        )
    with pytest.raises(
        episode_v3.EpisodeV3Error,
        match="expected_row_missing=True",
    ):
        episode_v3._verify_exact_row_stream(
            label="features_family_a",
            expected_rows=[],
            observed_rows=[row],
            fields=episode_v3.FEATURE_FIELDS,
        )


def test_source_semantic_contract_covers_all_feature_and_view_rows():
    projection = episode_v3._source_semantic_projection_contract()
    feature = projection["exact_feature_projection"]
    view = projection["exact_view_projection"]
    assert feature["fields"] == list(episode_v3.FEATURE_FIELDS)
    assert view["fields"] == list(episode_v3.VIEW_FIELDS)
    assert feature["families"]["family_a"]["row_count"] == 23_092_892
    assert feature["families"]["family_b"]["row_count"] == 15_310_944
    assert view["families"]["family_a"]["row_count"] == 268_522
    assert view["families"]["family_b"]["row_count"] == 141_768
    assert feature["families"]["family_a"]["feature_names"] == list(
        episode_v3._feature_name_universe("family_a")
    )
    assert feature["families"]["family_b"]["feature_names"] == list(
        episode_v3._feature_name_universe("family_b")
    )


def test_source_semantic_aggregate_contract_freezes_all_projections():
    contract = episode_v3._source_semantic_aggregate_contract()
    projections = contract["projections"]
    assert tuple(projections) == (
        "anchors",
        "views_family_a",
        "views_family_b",
        "features_family_a",
        "features_family_b",
        "sparse_range_family_a",
        "sparse_range_family_b",
        "fixed_grid_family_a",
        "fixed_grid_family_b",
        "event_count_family_a",
        "event_count_family_b",
        "outcomes",
    )
    assert contract["entry_keys"] == list(
        episode_v3.SOURCE_SEMANTIC_AGGREGATE_ENTRY_KEYS
    )
    assert projections["anchors"]["fields"] == list(episode_v3.ANCHOR_FIELDS)
    assert projections["anchors"]["expected_rows"] == 268_522
    assert projections["views_family_a"]["fields"] == list(episode_v3.VIEW_FIELDS)
    assert projections["views_family_b"]["expected_rows"] == 141_768
    assert projections["features_family_a"]["fields"] == list(
        episode_v3.FEATURE_FIELDS
    )
    assert projections["features_family_a"]["expected_rows"] == 23_092_892
    assert projections["features_family_b"]["expected_rows"] == 15_310_944
    assert projections["sparse_range_family_a"]["fields"] == list(
        episode_v3.RANGE_FIELDS
    )
    assert projections["fixed_grid_family_a"]["fields"] == list(
        episode_v3.GRID_FIELDS
    )
    assert projections["event_count_family_a"]["fields"] == list(
        episode_v3.EVENT_COUNT_FIELDS
    )
    assert projections["outcomes"]["fields"] == list(episode_v3.OUTCOME_FIELDS)
    assert (
        episode_v3._source_semantic_projection_contract()[
            "aggregate_evidence_contract"
        ]
        == contract
    )


def test_exact_projection_reports_full_row_count_and_digest():
    rows = [
        {name: f"row-{index}-{name}" for name in episode_v3.FEATURE_FIELDS}
        for index in range(2)
    ]
    result = episode_v3._verify_exact_row_stream(
        label="features_family_a",
        expected_rows=rows,
        observed_rows=[dict(row) for row in rows],
        fields=episode_v3.FEATURE_FIELDS,
    )
    assert result["expected_rows"] == result["observed_rows"] == 2
    assert result["expected_sha256"] == result["observed_sha256"]
    assert result["mismatch_rows"] == 0
    assert result["fields"] == list(episode_v3.FEATURE_FIELDS)
