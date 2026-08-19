#!/usr/bin/env python3
"""Build and verify Jul30 linked Family A/B trigger-aligned Episode v3 truth.

The package produced here is a public-market research fact layer. It reuses
the accepted Jul30 structured R0/timeline event stores through immutable
catalog and range references, and materializes candidate anchors, landmark
views, fixed grids, event-count views, feature observation ledgers, and
interval/right-censored market outcomes. It does not read legacy episode
response rows, later-session event rows, or private/order data.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import ctypes
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import stat
import sys
import uuid
from array import array
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, TextIO
from zoneinfo import ZoneInfo


sys.dont_write_bytecode = True

TASK_ID = "0815T003"
SCHEMA_VERSION = "episode_v3_jul30_v1"
CONTRACT_VERSION = "skhynix_jul30_episode_v3_contract_v3"
FROZEN_DATE = "2026-08-15"
SESSION_ID = "jul30"
CAMPAIGN_ID = "0729T010-skhynix-4h-8x30m"
PROFILE_ID = "skhynix"
SEGMENT_IDS = tuple(f"segment_{index:04d}" for index in range(1, 9))
GRID_MS = (-2000, -1000, -500, -250, -100, -50, -25, -10, 0, 10, 25, 50, 100, 250, 500, 1000, 2000)
EVENT_COUNT_POINTS = (1, 2, 3, 5, 10, 20)
OUTCOME_HORIZONS_MS = (100, 250, 500, 1000, 2000)
MARKOUT_HORIZONS_MS = (250, 500, 1000, 2000)
OUTCOME_HORIZON_NS = 2_000_000_000
EVIDENCE_LABELS = {
    **{f"segment_{index:04d}": "historical_discovery" for index in range(1, 5)},
    **{
        f"segment_{index:04d}": "historical_internal_validation"
        for index in range(5, 9)
    },
}

WORKTREE_ROOT = Path(
    "/Users/liu/Documents/hftbacktest-0814t001-skhynix-episode-research"
)
SOURCE_REPO_ROOT = Path("/Users/liu/Documents/hftbacktest")
DEFAULT_OUTPUT_DIR = (
    WORKTREE_ROOT
    / "local_live_analysis"
    / "skhynix_trigger_aligned_episode_research_v1_stage04_jul30_episode_v3"
)
STAGE1_DIR = (
    WORKTREE_ROOT / "local_live_analysis/skhynix_trigger_aligned_episode_research_v1"
)
STAGE2_DIR = (
    WORKTREE_ROOT
    / "local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage02_density"
)
STAGE3_DIR = (
    WORKTREE_ROOT
    / "local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage03_detector_parity"
)
JUL30_CAMPAIGN_DIR = (
    SOURCE_REPO_ROOT
    / "local_live_analysis/cross_exchange_collection_campaign_0730T011_skhynix_4h_8x30m"
)
JUL30_R0_DIR = (
    SOURCE_REPO_ROOT
    / "local_live_analysis/skhynix_cross_exchange_research_0730T013"
)

EXPECTED_STAGE1_CORE = (
    "9c2756f966a1a05816f9e6a91d7361d76cfea10fc3c587944ca2f0c1d2eb7c96"
)
EXPECTED_STAGE1_FULL = (
    "c540cc056313716b3bdd2b9c0fe076cda15a7f152b399ae6a1283b3aa8aa6590"
)
EXPECTED_STAGE2_CORE = (
    "7b3d06c3225f77c9929cdd3fa40d69c0866d83dbb75dd584dcb7db6db22ad3f8"
)
EXPECTED_STAGE2_FULL = (
    "bdade16c53aed7bba54fdb9408ba3a8a03036d183720f589bda2a762448a3833"
)
EXPECTED_STAGE3_CORE = (
    "4939d1c1addce493edb2f368297d56b37edd0b123de01497dcdee6e77637eb9b"
)
EXPECTED_STAGE3_FULL = (
    "ff8e3434672226371051151cea838503877dca79ac7860cf179256362d75e404"
)
EXPECTED_R0_MANIFEST_SHA256 = (
    "c46c735d7933587af6eece4a9dd1bce241b3c093866efce976b1c3f952e72ce0"
)
EXPECTED_STAGE2_MEMBERSHIP_SHA256 = (
    "b656ccd5442a80743b35e13e8913c1f7b7c4bef46d47283e86d8ec80a21b2afc"
)
EXPECTED_STAGE3_PROJECTION_SHA256 = (
    "68e65a612d7bce2911a4e3c96fd10528914290ca045c8912e6be45045fd0415f"
)
EXPECTED_COUNTS = {
    "family_a": 268522,
    "family_b": 141768,
    "feature_family_a": 23092892,
    "feature_family_b": 15310944,
    "rejected": 126754,
    "clusters": 39928,
    "flows": 10536,
    "overlap_blocks": 9,
}
SOURCE_SEMANTIC_AGGREGATE_CONTRACT_VERSION = (
    "skhynix_jul30_source_semantic_aggregate_evidence_v2"
)
SOURCE_SEMANTIC_AGGREGATE_ENTRY_KEYS = (
    "expected_rows",
    "observed_rows",
    "expected_sha256",
    "observed_sha256",
    "mismatch_rows",
    "fields",
)

BINANCE_HOT_FIELDS_V1 = (
    "segment_id",
    "event_seq",
    "source_raw_seq",
    "local_ts_ns",
    "exchange_ts_ns",
    "event_type",
    "symbol",
    "update_id",
    "bid_px",
    "bid_qty",
    "ask_px",
    "ask_qty",
    "buyer_is_maker",
    "aggressor_side",
    "trade_px",
    "trade_qty",
    "trade_id",
)
HYPERLIQUID_HOT_FIELDS_V1 = (
    "segment_id",
    "event_seq",
    "source_raw_seq",
    "source_item_index",
    "local_ts_ns",
    "exchange_ts_ns",
    "event_type",
    "coin",
    "bid_px",
    "bid_qty",
    "bid_n",
    "ask_px",
    "ask_qty",
    "ask_n",
    "trade_side",
    "trade_px",
    "trade_qty",
    "trade_id",
    "trade_hash",
    "trade_users_json",
)
AUXILIARY_FIELDS_V1 = (
    "segment_id",
    "event_seq",
    "track_id",
    "source_raw_seq",
    "local_ts_ns",
    "exchange_ts_ns",
    "event_type",
    "coin",
    "degraded",
    "degraded_interval_id",
    "mark_px",
    "oracle_px",
    "mid_px",
    "funding",
    "premium",
    "open_interest",
    "impact_bid_px",
    "impact_ask_px",
    "target_mid_px",
    "candle_open",
    "candle_high",
    "candle_low",
    "candle_close",
    "candle_volume",
    "candle_trade_count",
    "payload_json",
)

ANCHOR_FIELDS = (
    "campaign_id",
    "session_id",
    "segment_id",
    "episode_id",
    "candidate_id",
    "shock_cluster_id",
    "flow_cluster_id",
    "overlap_block_id",
    "candidate_seq",
    "direction_sign",
    "aggressor_side",
    "t_burst_start_ns",
    "t_candidate_ns",
    "t_confirm_ns",
    "detector_decision_ts_ns",
    "confirmation_lag_ns",
    "family_a_available",
    "family_b_available",
    "confirmed",
    "classification",
    "rejection_reason",
    "censor_time_ns",
    "censor_reason",
    "quality_flags_json",
    "connection_epoch_id",
    "segment_start_ts_ns",
    "segment_end_ts_ns",
    "evidence_label",
    "source_manifest_sha256",
)
VIEW_FIELDS = (
    "candidate_id",
    "episode_id",
    "family_view",
    "decision_landmark",
    "landmark_ts_ns",
    "anchor_artifact",
    "sparse_range_artifact",
    "fixed_grid_artifact",
    "event_count_artifact",
    "feature_ledger_artifact",
    "market_outcome_artifact",
    "population_eligible",
    "quality_subset_eligible",
    "quality_subset_reason",
)
RANGE_FIELDS = (
    "candidate_id",
    "episode_id",
    "family_view",
    "landmark_ts_ns",
    "window_start_ts_ns",
    "window_end_ts_ns",
    "connection_epoch_id",
    "binance_book_ticker_first_event_seq",
    "binance_book_ticker_last_event_seq",
    "binance_book_ticker_count",
    "binance_trade_first_event_seq",
    "binance_trade_last_event_seq",
    "binance_trade_count",
    "hyperliquid_bbo_first_event_seq",
    "hyperliquid_bbo_last_event_seq",
    "hyperliquid_bbo_count",
    "hyperliquid_trade_first_event_seq",
    "hyperliquid_trade_last_event_seq",
    "hyperliquid_trade_count",
    "hyperliquid_fast_l2_first_book_version",
    "hyperliquid_fast_l2_last_book_version",
    "hyperliquid_fast_l2_count",
    "common_l2_first_common_seq",
    "common_l2_last_common_seq",
    "common_l2_count",
    "inside_segment_start",
    "inside_segment_end",
)
GRID_FIELDS = (
    "candidate_id",
    "episode_id",
    "family_view",
    "decision_landmark",
    "landmark_ts_ns",
    "requested_relative_ms",
    "requested_ts_ns",
    "inside_segment",
    "availability_reason",
    "connection_epoch_id",
    "core_degraded_mask",
    "auxiliary_degraded_mask",
    "binance_bid_px",
    "binance_ask_px",
    "binance_mid_px",
    "binance_impacted_best_qty",
    "binance_opposite_best_qty",
    "binance_impacted_top5_qty",
    "binance_opposite_top5_qty",
    "binance_observed_at_ns",
    "binance_source_event_id",
    "binance_source_book_version",
    "binance_strict_asof_ts_ns",
    "binance_effective_relative_ns",
    "binance_source_age_ns",
    "binance_no_new_information",
    "hyperliquid_bbo_bid_px",
    "hyperliquid_bbo_ask_px",
    "hyperliquid_bbo_mid_px",
    "hyperliquid_bbo_impacted_qty",
    "hyperliquid_bbo_opposite_qty",
    "hyperliquid_bbo_observed_at_ns",
    "hyperliquid_bbo_source_event_id",
    "hyperliquid_bbo_source_book_version",
    "hyperliquid_bbo_strict_asof_ts_ns",
    "hyperliquid_bbo_effective_relative_ns",
    "hyperliquid_bbo_source_age_ns",
    "hyperliquid_bbo_no_new_information",
    "hyperliquid_fast_bid_px",
    "hyperliquid_fast_ask_px",
    "hyperliquid_fast_mid_px",
    "hyperliquid_fast_impacted_top5_qty",
    "hyperliquid_fast_opposite_top5_qty",
    "hyperliquid_fast_observed_at_ns",
    "hyperliquid_fast_source_event_id",
    "hyperliquid_fast_source_book_version",
    "hyperliquid_fast_strict_asof_ts_ns",
    "hyperliquid_fast_effective_relative_ns",
    "hyperliquid_fast_source_age_ns",
    "hyperliquid_fast_no_new_information",
    "d_bh_bps",
    "d_hb_bps",
    "risk_gap_bps",
)
EVENT_COUNT_FIELDS = (
    "candidate_id",
    "episode_id",
    "family_view",
    "decision_landmark",
    "landmark_ts_ns",
    "channel",
    *tuple(
        field
        for count in EVENT_COUNT_POINTS
        for field in (
            f"event_{count}_source_event_id",
            f"event_{count}_observed_at_ns",
            f"event_{count}_relative_ns",
            f"event_{count}_inside_segment",
            f"event_{count}_availability_reason",
        )
    ),
)
FEATURE_FIELDS = (
    "episode_id",
    "candidate_id",
    "family_view",
    "decision_landmark",
    "feature_name",
    "value",
    "observed_at_ns",
    "source_event_id",
    "source_book_version",
    "calculation_version",
    "availability_reason",
)
FIRST_EVENT_NAMES = (
    "time_to_first_adverse_target_bbo_event",
    "time_to_first_target_trade_at_or_through_vulnerable_pretrigger_quote",
    "time_to_first_target_impacted_side_price_retreat",
)
OUTCOME_FIELDS = (
    "candidate_id",
    "episode_id",
    "t_candidate_ns",
    "t_confirm_ns",
    "outcome_horizon_end_ns",
    "outcome_horizon_status",
    "outcome_censor_time_ns",
    "outcome_censor_reason",
    *tuple(
        field
        for name in FIRST_EVENT_NAMES
        for field in (
            f"{name}_status",
            f"{name}_interval_lower_ns",
            f"{name}_interval_upper_ns",
            f"{name}_censor_time_ns",
            f"{name}_censor_reason",
            f"{name}_source_event_id",
        )
    ),
    *tuple(
        field
        for horizon in OUTCOME_HORIZONS_MS
        for field in (
            f"gap_survival_{horizon}ms",
            f"gap_survival_{horizon}ms_availability",
            f"risk_gap_{horizon}ms_bps",
        )
    ),
    *tuple(
        field
        for horizon in MARKOUT_HORIZONS_MS
        for field in (
            f"target_midpoint_markout_{horizon}ms_bps",
            f"target_midpoint_markout_{horizon}ms_availability",
        )
    ),
    "maximum_adverse_excursion_0_2000ms_bps",
    "maximum_favorable_excursion_0_2000ms_bps",
    "excursion_availability",
    "source_leg_gap_closure_contribution_bps",
    "target_leg_gap_closure_contribution_bps",
    "gap_closure_contribution_availability",
    "adverse_event_before_confirmed",
    "adverse_event_before_confirmed_status",
    "public_trade_reaches_quote",
    "public_bbo_moves_through_quote",
    "public_quote_survives_horizon",
    "public_adverse_exposure",
    "public_quote_risk_availability",
)
SEGMENT_SUMMARY_FIELDS = (
    "segment_id",
    "evidence_label",
    "candidate_rows",
    "family_a_rows",
    "family_b_rows",
    "rejected_rows",
    "anchor_rows",
    "outcome_rows",
    "sparse_range_family_a_rows",
    "sparse_range_family_b_rows",
    "fixed_grid_family_a_rows",
    "fixed_grid_family_b_rows",
    "event_count_family_a_rows",
    "event_count_family_b_rows",
    "feature_ledger_family_a_rows",
    "feature_ledger_family_b_rows",
    "feature_unavailable_rows",
    "future_feature_observation_mismatch_count",
    "anchor_ordering_mismatch_count",
    "cross_segment_path_mismatch_count",
    "cross_epoch_path_mismatch_count",
    "synthetic_rejected_confirm_count",
    "interval_censored_outcome_count",
    "right_censored_outcome_count",
    "segment_censored_outcome_count",
    "epoch_censored_outcome_count",
    "quality_censored_outcome_count",
    "point_coerced_outcome_count",
    "auxiliary_degraded_grid_rows",
    "core_degraded_grid_rows",
)
SOURCE_CATALOG_FIELDS = (
    "session_id",
    "segment_id",
    "source_store_id",
    "venue",
    "channel",
    "path",
    "row_count",
    "bytes",
    "sha256",
    "schema_fields_json",
    "source_event_identity",
    "connection_epoch_contract",
    "degraded_contract",
    "usage",
)
QUALITY_INTERVAL_FIELDS = (
    "session_id",
    "segment_id",
    "interval_id",
    "mask_type",
    "track_id",
    "start_ts_ns",
    "end_ts_ns",
    "policy",
    "reason",
    "core_outcome_censor",
    "auxiliary_feature_degraded",
)
INPUT_BINDING_FIELDS = (
    "snapshot_phase",
    "scope",
    "role",
    "session_id",
    "segment_id",
    "root",
    "path",
    "relative_path",
    "bytes",
    "sha256",
)

BOUNDARY = {
    "jul30_legacy_episode_rows_read": False,
    "aug03_aug04_future_event_rows_read": False,
    "aug07_event_rows_read": False,
    "model_or_score_run": False,
    "case_retrieval_run": False,
    "actionability_run": False,
    "own_order_fill_pnl_fields": False,
    "network_accessed": False,
    "private_or_order_endpoint_accessed": False,
    "new_collection": False,
}

FORBIDDEN_PATH_TOKENS = (
    "skhynix_liquidity_response_0730t017/episodes/",
    "skhynix_liquidity_response_0803t002/",
    "skhynix_liquidity_response_0804t008/",
    "0807t001_skhynix_4h_continuous/",
    "0807t002_skhynix_basis_postprocess/",
    "0807t002_skhynix_postprocess_compact/",
)
FORBIDDEN_FIELD_TOKENS = (
    "own_order",
    "own_fill",
    "fill_probability",
    "filled",
    "execution_pnl",
    "pnl",
    "inventory",
    "fee",
    "queue_position",
)

CALC_VERSION = "episode_v3_jul30_feature_calculation_v2"
TRADE_QUANTITY_SOURCE = "binance_trade_q"
RPI_ADJUSTMENT_STATUS = "unavailable_historical_feed"
DEPTH_STREAM = "depth@0ms"
UNDERLYING_MARKET_STATE = "unknown_calendar_state"
CLASSIFICATION_SOURCE = (
    "accepted_stage3_candidate_audit_projection.attribution"
)
CLASSIFICATION_ALLOWED_VALUES = (
    "cancel_driven",
    "mixed",
    "trade_driven",
    "uncertain",
)
CONFIRMATION_FEATURE_NAMES = (
    "confirmed_burst_trade_count_through_decision",
    "confirmed_burst_trade_qty_through_decision",
    "confirmed_burst_duration_through_decision_ms",
    "confirmed_touch_trade_qty_through_decision",
    "confirmed_impact_ratio_legacy",
    "confirmed_queue_drop_ratio",
    "confirmed_price_level_depleted",
    "confirmed_removed_qty",
    "confirmed_trade_explained_ratio",
    "confirmed_attribution",
    "confirmation_lag_ms",
    "post_candidate_trade_continuation_count",
    "post_candidate_trade_continuation_qty",
    "local_binance_source_event_count_candidate_to_confirm",
)


class EpisodeV3Error(RuntimeError):
    """Raised when Stage 4 cannot satisfy its frozen contract."""


@dataclass
class RowStreamDigest:
    count: int = 0
    digest: Any = field(default_factory=hashlib.sha256)

    def update(
        self,
        row: Mapping[str, Any],
        fields: Sequence[str] | None = None,
    ) -> None:
        payload = (
            dict(row)
            if fields is None
            else {
                field_name: (
                    ""
                    if row.get(field_name) is None
                    else str(row.get(field_name, ""))
                )
                for field_name in fields
            }
        )
        self.digest.update(canonical_json_bytes(payload) + b"\n")
        self.count += 1

    @property
    def sha256(self) -> str:
        return self.digest.hexdigest()


def _assert_row_stream_identity(
    *,
    label: str,
    expected: RowStreamDigest,
    observed: RowStreamDigest,
) -> None:
    if expected.count != observed.count or expected.sha256 != observed.sha256:
        raise EpisodeV3Error(
            f"{label} source-semantic drift "
            f"expected={expected.count}/{expected.sha256} "
            f"observed={observed.count}/{observed.sha256}"
        )


def _normalized_row_values(
    row: Mapping[str, Any],
    fields: Sequence[str],
) -> tuple[str, ...]:
    return tuple(
        "" if row.get(field_name) is None else str(row.get(field_name, ""))
        for field_name in fields
    )


def _update_exact_row_digest(
    digest: Any,
    values: Sequence[str],
) -> None:
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)


def _verify_exact_row_stream(
    *,
    label: str,
    expected_rows: Iterable[Mapping[str, Any]],
    observed_rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
    aggregate_digest: Any | None = None,
) -> dict[str, Any]:
    missing = object()
    digest = hashlib.sha256()
    count = 0
    for row_number, (expected_row, observed_row) in enumerate(
        zip_longest(expected_rows, observed_rows, fillvalue=missing),
        start=1,
    ):
        if expected_row is missing or observed_row is missing:
            raise EpisodeV3Error(
                f"{label} source-semantic drift row={row_number} "
                f"expected_row_missing={expected_row is missing} "
                f"observed_row_missing={observed_row is missing}"
            )
        expected_values = _normalized_row_values(expected_row, fields)
        observed_values = _normalized_row_values(observed_row, fields)
        if expected_values != observed_values:
            mismatched_fields = [
                field_name
                for field_name, expected_value, observed_value in zip(
                    fields,
                    expected_values,
                    observed_values,
                    strict=True,
                )
                if expected_value != observed_value
            ]
            candidate_id = (
                expected_row.get("candidate_id")
                or expected_row.get("episode_id")
                or ""
            )
            raise EpisodeV3Error(
                f"{label} source-semantic drift row={row_number} "
                f"candidate_id={candidate_id} "
                f"fields={','.join(mismatched_fields)}"
            )
        _update_exact_row_digest(digest, expected_values)
        if aggregate_digest is not None:
            _update_exact_row_digest(aggregate_digest, expected_values)
        count += 1
    sha256 = digest.hexdigest()
    return {
        "expected_rows": count,
        "observed_rows": count,
        "expected_sha256": sha256,
        "observed_sha256": sha256,
        "mismatch_rows": 0,
        "fields": list(fields),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def canonical_json_line_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload) + b"\n").hexdigest()


def canonical_pretty_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_bytes())
    except Exception as exc:
        raise EpisodeV3Error(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise EpisodeV3Error(f"{path}: expected JSON object")
    return payload


def _read_canonical_json(path: Path, *, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = _read_json(path)
    if raw != canonical_pretty_json_bytes(payload):
        raise EpisodeV3Error(f"{label} canonical bytes drift")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_pretty_json_bytes(payload))


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _float_text(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    if abs(value) < 5e-16:
        value = 0.0
    return format(value, ".12g")


def _int_text(value: int | None) -> str:
    return "" if value is None else str(value)


def _parse_bool(value: Any, *, label: str) -> bool:
    if value is True or str(value).lower() == "true":
        return True
    if value is False or str(value).lower() == "false":
        return False
    raise EpisodeV3Error(f"{label}: invalid boolean {value!r}")


def _required_int(row: Mapping[str, Any], field_name: str) -> int:
    value = row.get(field_name)
    if value in (None, ""):
        raise EpisodeV3Error(f"required integer is empty: {field_name}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise EpisodeV3Error(f"invalid integer {field_name}: {value!r}") from exc


def _optional_int(row: Mapping[str, Any], field_name: str) -> int | None:
    value = row.get(field_name)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise EpisodeV3Error(f"invalid integer {field_name}: {value!r}") from exc


def _required_float(row: Mapping[str, Any], field_name: str) -> float:
    value = row.get(field_name)
    if value in (None, ""):
        raise EpisodeV3Error(f"required float is empty: {field_name}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise EpisodeV3Error(f"invalid float {field_name}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise EpisodeV3Error(f"non-finite float {field_name}")
    return parsed


def _optional_float(row: Mapping[str, Any], field_name: str) -> float | None:
    value = row.get(field_name)
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise EpisodeV3Error(f"invalid float {field_name}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise EpisodeV3Error(f"non-finite float {field_name}")
    return parsed


@contextmanager
def deterministic_gzip_text_writer(path: Path) -> Iterator[TextIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=1, mtime=0
        ) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                yield text


def _open_csv_writer(
    stack: ExitStack, path: Path, fields: Sequence[str]
) -> csv.DictWriter[str]:
    if path.suffix == ".gz":
        fh = stack.enter_context(deterministic_gzip_text_writer(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = stack.enter_context(path.open("w", encoding="utf-8", newline=""))
    writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    return writer


def _write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]
) -> int:
    count = 0
    with ExitStack() as stack:
        writer = _open_csv_writer(stack, path, fields)
        for row in rows:
            writer.writerow({field_name: row.get(field_name, "") for field_name in fields})
            count += 1
    return count


def _strict_csv_rows(
    path: Path, fields: Sequence[str] | None = None
) -> Iterator[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        observed_fields = tuple(reader.fieldnames or ())
        if fields is not None and observed_fields != tuple(fields):
            raise EpisodeV3Error(f"{path}: CSV schema drift")
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(
                value is None or isinstance(value, list) for value in row.values()
            ):
                raise EpisodeV3Error(f"{path}: row {row_number} cell-width drift")
            yield {str(key): str(value) for key, value in row.items()}


def _gzip_row_count(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        return max(0, sum(1 for _ in fh) - 1)


@dataclass(frozen=True)
class ExactTreeEntry:
    path: Path
    relative_path: str
    entry_type: str
    mode: int
    bytes: int


def _tree_entry_type_contract() -> dict[str, Any]:
    return {
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


def _lstat_entry_type(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "regular_file"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISCHR(mode):
        return "character_device"
    if stat.S_ISBLK(mode):
        return "block_device"
    return "other_special"


def _exact_tree_entries(root: Path) -> tuple[ExactTreeEntry, ...]:
    root = Path(root)
    try:
        root_lstat = root.lstat()
    except FileNotFoundError as exc:
        raise EpisodeV3Error(f"missing directory: {root}") from exc
    root_type = _lstat_entry_type(root_lstat.st_mode)
    if root_type != "directory":
        raise EpisodeV3Error(
            f"tree root must be a real directory: {root} type={root_type}"
        )

    entries: list[ExactTreeEntry] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as iterator:
            names = sorted(entry.name for entry in iterator)
        child_directories: list[Path] = []
        for name in names:
            path = directory / name
            try:
                entry_lstat = path.lstat()
            except FileNotFoundError as exc:
                raise EpisodeV3Error(
                    f"package tree entry disappeared during scan: {path}"
                ) from exc
            entry_type = _lstat_entry_type(entry_lstat.st_mode)
            relative_path = path.relative_to(root).as_posix()
            if entry_type not in {"regular_file", "directory"}:
                raise EpisodeV3Error(
                    "package tree entry type forbidden: "
                    f"{relative_path} type={entry_type}"
                )
            entries.append(
                ExactTreeEntry(
                    path=path,
                    relative_path=relative_path,
                    entry_type=entry_type,
                    mode=entry_lstat.st_mode,
                    bytes=entry_lstat.st_size,
                )
            )
            if entry_type == "directory":
                child_directories.append(path)
        pending.extend(reversed(child_directories))
    return tuple(sorted(entries, key=lambda entry: entry.relative_path))


def _directory_inventory(root: Path) -> list[dict[str, Any]]:
    entries = _exact_tree_entries(root)
    return [
        {
            "path": entry.relative_path,
            "bytes": entry.bytes,
            "sha256": sha256_file(entry.path),
        }
        for entry in entries
        if entry.entry_type == "regular_file"
    ]


def _stage1_full_inventory_sha(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "path": row["path"],
            "size": row["bytes"],
            "sha256": row["sha256"],
            "role": "accepted_stage1_package",
            "session_id": "",
            "segment_id": "",
        }
        for row in rows
    ]
    return canonical_json_line_sha256(payload)


def _package_full_inventory_sha(root: Path) -> str:
    return canonical_json_sha256(_directory_inventory(root))


def _guard_allowed_path(path: Path) -> None:
    raw = path.as_posix().lower()
    if any(token in raw for token in FORBIDDEN_PATH_TOKENS):
        raise EpisodeV3Error(f"forbidden Stage 4 input path: {path}")


def _verify_dependency_packages() -> dict[str, dict[str, Any]]:
    specs = (
        ("stage1", STAGE1_DIR, "research_manifest.json", EXPECTED_STAGE1_CORE, EXPECTED_STAGE1_FULL),
        ("stage2", STAGE2_DIR, "density_manifest.json", EXPECTED_STAGE2_CORE, EXPECTED_STAGE2_FULL),
        ("stage3", STAGE3_DIR, "parity_manifest.json", EXPECTED_STAGE3_CORE, EXPECTED_STAGE3_FULL),
    )
    result: dict[str, dict[str, Any]] = {}
    for name, root, manifest_name, expected_core, expected_full in specs:
        resolved = root.resolve()
        if resolved != root:
            raise EpisodeV3Error(f"{name}: noncanonical dependency root")
        manifest = _read_json(root / manifest_name)
        if manifest.get("core_package_sha256") != expected_core:
            raise EpisodeV3Error(f"{name}: accepted core identity drift")
        rows = _directory_inventory(root)
        full = (
            _stage1_full_inventory_sha(rows)
            if name == "stage1"
            else canonical_json_sha256(rows)
        )
        if full != expected_full:
            raise EpisodeV3Error(f"{name}: accepted full inventory drift")
        result[name] = {
            "root": str(root),
            "core_sha256": expected_core,
            "full_inventory_sha256": expected_full,
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
        }
    return result


@dataclass(frozen=True)
class SegmentSpec:
    segment_id: str
    manifest_path: Path
    binance_hot_path: Path
    hyperliquid_hot_path: Path
    auxiliary_path: Path
    timeline_path: Path
    decision_label_path: Path
    segment_start_ts_ns: int
    segment_end_ts_ns: int
    expected_rows: Mapping[str, int]
    expected_shas: Mapping[str, str]


@dataclass
class Candidate:
    candidate_id: str
    episode_id: str
    segment_id: str
    candidate_seq: int
    aggressor_side: str
    direction_sign: int
    burst_start_ts_ns: int
    burst_end_ts_ns: int
    burst_duration_ms: float
    burst_trade_count: int
    burst_trade_qty: float
    touch_trade_qty: float
    touch_trade_qty_at_shock: float
    touch_trade_qty_through_decision: float | None
    post_decision_burst_trade_count: int | None
    pre_state_ts_ns: int
    pre_best_px: float
    pre_best_qty: float
    shock_ts_ns: int
    impact_ratio: float
    shock_impact_ratio: float
    detector_decision_ts_ns: int | None
    confirmation_lag_ms: float | None
    confirmed_best_px: float | None
    confirmed_best_qty: float | None
    price_level_depleted: bool | None
    queue_drop_ratio: float | None
    confirmed_removed_qty: float | None
    trade_explained_ratio: float | None
    attribution: str
    pre_hl_bbo_ts_ns: int | None
    pre_hl_bbo_age_ms: float | None
    pre_hl_fast_source_ts_ns: int | None
    pre_hl_fast_age_ms: float | None
    primary_episode: bool
    rejection_reason: str
    connection_epoch_id: str
    segment_start_ts_ns: int
    segment_end_ts_ns: int
    cluster_id: str
    continuous_flow_episode_id: str
    overlap_block_id: str
    window_end_ts_ns: int

    @property
    def family_b_available(self) -> bool:
        return self.primary_episode

    @property
    def t_confirm_ns(self) -> int | None:
        return self.detector_decision_ts_ns if self.primary_episode else None

    @property
    def vulnerable_side(self) -> str:
        return "ask" if self.direction_sign == 1 else "bid"


@dataclass
class NumericEventStore:
    segment_id: str
    channel: str
    ts: array = field(default_factory=lambda: array("q"))
    seq: array = field(default_factory=lambda: array("q"))
    raw_seq: array = field(default_factory=lambda: array("q"))
    item_index: array = field(default_factory=lambda: array("q"))
    exchange_ts: array = field(default_factory=lambda: array("q"))
    x1: array = field(default_factory=lambda: array("d"))
    x2: array = field(default_factory=lambda: array("d"))
    x3: array = field(default_factory=lambda: array("d"))
    x4: array = field(default_factory=lambda: array("d"))
    side: array = field(default_factory=lambda: array("b"))

    def source_event_id(self, index: int) -> str:
        return (
            f"{SESSION_ID}:{self.segment_id}:{self.channel}:"
            f"{self.seq[index]}:{self.raw_seq[index]}:{self.item_index[index]}"
        )

    def range_indices(self, start_ns: int, end_ns: int) -> tuple[int, int]:
        return bisect.bisect_left(self.ts, start_ns), bisect.bisect_right(self.ts, end_ns)

    def asof_index(self, ts_ns: int) -> int:
        return bisect.bisect_right(self.ts, ts_ns) - 1

    def strict_pre_index(self, ts_ns: int) -> int:
        return bisect.bisect_left(self.ts, ts_ns) - 1


@dataclass
class TimelineStore:
    segment_id: str
    ts: array = field(default_factory=lambda: array("q"))
    common_seq: array = field(default_factory=lambda: array("q"))
    binance_observed_at: array = field(default_factory=lambda: array("q"))
    binance_bid: array = field(default_factory=lambda: array("d"))
    binance_ask: array = field(default_factory=lambda: array("d"))
    binance_bid_qty: array = field(default_factory=lambda: array("d"))
    binance_ask_qty: array = field(default_factory=lambda: array("d"))
    binance_bid_top5: array = field(default_factory=lambda: array("d"))
    binance_ask_top5: array = field(default_factory=lambda: array("d"))
    hl_fast_observed_at: array = field(default_factory=lambda: array("q"))
    hl_fast_bid: array = field(default_factory=lambda: array("d"))
    hl_fast_ask: array = field(default_factory=lambda: array("d"))
    hl_fast_bid_qty: array = field(default_factory=lambda: array("d"))
    hl_fast_ask_qty: array = field(default_factory=lambda: array("d"))
    hl_fast_bid_top5: array = field(default_factory=lambda: array("d"))
    hl_fast_ask_top5: array = field(default_factory=lambda: array("d"))
    trigger_track: list[str] = field(default_factory=list)
    fast_event_ts: array = field(default_factory=lambda: array("q"))
    fast_event_version: array = field(default_factory=lambda: array("q"))
    midpoint_sq_return_prefix: array = field(default_factory=lambda: array("d", [0.0]))

    def asof_index(self, ts_ns: int) -> int:
        return bisect.bisect_right(self.ts, ts_ns) - 1

    def strict_pre_index(self, ts_ns: int) -> int:
        return bisect.bisect_left(self.ts, ts_ns) - 1

    def source_event_id(self, index: int) -> str:
        return f"{SESSION_ID}:{self.segment_id}:common_l2_timeline:{self.common_seq[index]}"

    def fast_source_event_id(self, index: int) -> str:
        return (
            f"{SESSION_ID}:{self.segment_id}:hyperliquid_fast_l2:"
            f"{self.hl_fast_observed_at[index]}"
        )


@dataclass
class SegmentData:
    spec: SegmentSpec
    timeline: TimelineStore
    binance_bbo: NumericEventStore
    binance_trades: NumericEventStore
    hl_bbo: NumericEventStore
    hl_trades: NumericEventStore
    auxiliary_intervals: list[dict[str, Any]]
    binance_signed_qty_prefix: array
    binance_ofi_prefix: array
    candidate_ts: array
    candidate_direction: array


def _validate_header(path: Path, expected: Sequence[str]) -> None:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        observed = tuple(next(reader, ()))
    if observed != tuple(expected):
        raise EpisodeV3Error(f"{path}: structured source schema drift")


def _load_segment_specs() -> tuple[dict[str, Any], dict[str, Any], list[SegmentSpec]]:
    r0_path = JUL30_R0_DIR / "research_input_manifest.json"
    if sha256_file(r0_path) != EXPECTED_R0_MANIFEST_SHA256:
        raise EpisodeV3Error("Jul30 R0 manifest SHA drift")
    r0 = _read_json(r0_path)
    if (
        r0.get("passes") is not True
        or r0.get("campaign_id") != CAMPAIGN_ID
        or r0.get("profile_id") != PROFILE_ID
        or int(r0.get("segment_count", -1)) != len(SEGMENT_IDS)
    ):
        raise EpisodeV3Error("Jul30 R0 manifest identity drift")
    alignment_path = JUL30_R0_DIR / "alignment/alignment_manifest.json"
    alignment = _read_json(alignment_path)
    if (
        alignment.get("passes") is not True
        or alignment.get("campaign_id") != CAMPAIGN_ID
        or alignment.get("profile_id") != PROFILE_ID
        or alignment.get("join_clock") != "same_host_local_receipt_time_time_ns"
        or alignment.get("source_manifest", {}).get("sha256")
        != EXPECTED_R0_MANIFEST_SHA256
    ):
        raise EpisodeV3Error("Jul30 alignment manifest identity drift")
    if any(
        int(alignment.get(field_name, -1)) != 0
        for field_name in (
            "timestamp_regression_count",
            "future_decision_join_count",
            "cross_segment_label_count",
        )
    ):
        raise EpisodeV3Error("Jul30 alignment zero-error gate drift")
    timeline_index: dict[str, tuple[int, int]] = {}
    for row in _strict_csv_rows(JUL30_CAMPAIGN_DIR / "timeline_index.csv"):
        segment_id = row["segment_id"]
        timeline_index[segment_id] = (
            int(row["first_common_ts_ns"]),
            int(row["last_common_ts_ns"]),
        )
    specs: list[SegmentSpec] = []
    descriptors = r0.get("segments")
    if not isinstance(descriptors, list):
        raise EpisodeV3Error("R0 segment descriptors missing")
    for descriptor in descriptors:
        segment_id = str(descriptor.get("segment_id", ""))
        if segment_id not in SEGMENT_IDS:
            raise EpisodeV3Error(f"unexpected Jul30 segment: {segment_id}")
        segment_manifest_path = JUL30_R0_DIR / str(descriptor["manifest_path"])
        if sha256_file(segment_manifest_path) != descriptor["manifest_sha256"]:
            raise EpisodeV3Error(f"{segment_id}: segment manifest SHA drift")
        segment_manifest = _read_json(segment_manifest_path)
        outputs = descriptor["outputs"]
        if segment_manifest.get("outputs") != outputs:
            raise EpisodeV3Error(f"{segment_id}: R0 descriptor/output drift")
        timeline_entry = segment_manifest["source_files"]["timeline"]
        timeline_path = Path(str(timeline_entry["path"]))
        _guard_allowed_path(timeline_path)
        if sha256_file(timeline_path) != timeline_entry["sha256"]:
            raise EpisodeV3Error(f"{segment_id}: timeline SHA drift")
        decision_entry = alignment["decision_label_outputs"][segment_id]
        decision_path = JUL30_R0_DIR / "alignment" / str(decision_entry["path"])
        if sha256_file(decision_path) != decision_entry["sha256"]:
            raise EpisodeV3Error(f"{segment_id}: decision-label SHA drift")
        start_ns, end_ns = timeline_index[segment_id]
        specs.append(
            SegmentSpec(
                segment_id=segment_id,
                manifest_path=segment_manifest_path,
                binance_hot_path=JUL30_R0_DIR
                / str(outputs["binance_hot_events"]["path"]),
                hyperliquid_hot_path=JUL30_R0_DIR
                / str(outputs["hyperliquid_hot_events"]["path"]),
                auxiliary_path=JUL30_R0_DIR
                / str(outputs["hyperliquid_auxiliary_events"]["path"]),
                timeline_path=timeline_path,
                decision_label_path=decision_path,
                segment_start_ts_ns=start_ns,
                segment_end_ts_ns=end_ns,
                expected_rows={
                    "binance_hot_events": int(
                        outputs["binance_hot_events"]["row_count"]
                    ),
                    "hyperliquid_hot_events": int(
                        outputs["hyperliquid_hot_events"]["row_count"]
                    ),
                    "hyperliquid_auxiliary_events": int(
                        outputs["hyperliquid_auxiliary_events"]["row_count"]
                    ),
                    "timeline": int(timeline_entry["row_count"]),
                    "decision_labels": int(decision_entry["row_count"]),
                },
                expected_shas={
                    "segment_manifest": str(descriptor["manifest_sha256"]),
                    "binance_hot_events": str(
                        outputs["binance_hot_events"]["sha256"]
                    ),
                    "hyperliquid_hot_events": str(
                        outputs["hyperliquid_hot_events"]["sha256"]
                    ),
                    "hyperliquid_auxiliary_events": str(
                        outputs["hyperliquid_auxiliary_events"]["sha256"]
                    ),
                    "timeline": str(timeline_entry["sha256"]),
                    "decision_labels": str(decision_entry["sha256"]),
                },
            )
        )
    specs.sort(key=lambda item: item.segment_id)
    if tuple(spec.segment_id for spec in specs) != SEGMENT_IDS:
        raise EpisodeV3Error("Jul30 segment order drift")
    return r0, alignment, specs


def _validate_decision_labels(spec: SegmentSpec) -> None:
    count = 0
    previous_ts = -1
    required = {
        "campaign_id",
        "segment_id",
        "decision_local_ts_ns",
        "timeline_asof_ts_ns",
        "hyperliquid_bbo_asof_ts_ns",
        "hyperliquid_bbo_age_ms",
        "auxiliary_degraded",
        "auxiliary_degraded_interval_ids",
    }
    opener = gzip.open
    with opener(spec.decision_label_path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not required.issubset(reader.fieldnames or ()):
            raise EpisodeV3Error(
                f"{spec.segment_id}: decision-label required schema drift"
            )
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise EpisodeV3Error(
                    f"{spec.segment_id}: decision-label row width drift {row_number}"
                )
            if (
                row["campaign_id"] != CAMPAIGN_ID
                or row["segment_id"] != spec.segment_id
            ):
                raise EpisodeV3Error(
                    f"{spec.segment_id}: decision-label identity drift"
                )
            ts_ns = int(row["decision_local_ts_ns"])
            if ts_ns < previous_ts:
                raise EpisodeV3Error(
                    f"{spec.segment_id}: decision-label timestamp regression"
                )
            if row["timeline_asof_ts_ns"]:
                if int(row["timeline_asof_ts_ns"]) > ts_ns:
                    raise EpisodeV3Error(
                        f"{spec.segment_id}: future timeline decision join"
                    )
            elif row["eligible"] != "false" or not row["warmup_reason"]:
                raise EpisodeV3Error(
                    f"{spec.segment_id}: unexplained missing timeline asof"
                )
            if row["hyperliquid_bbo_asof_ts_ns"] and int(
                row["hyperliquid_bbo_asof_ts_ns"]
            ) > ts_ns:
                raise EpisodeV3Error(
                    f"{spec.segment_id}: future Hyperliquid BBO decision join"
                )
            if (
                not row["hyperliquid_bbo_asof_ts_ns"]
                and (row["eligible"] != "false" or not row["warmup_reason"])
            ):
                raise EpisodeV3Error(
                    f"{spec.segment_id}: unexplained missing Hyperliquid BBO asof"
                )
            previous_ts = ts_ns
            count += 1
    if count != spec.expected_rows["decision_labels"]:
        raise EpisodeV3Error(f"{spec.segment_id}: decision-label row-count drift")


def _load_quality_intervals() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_path = JUL30_R0_DIR / "segment_and_mask_index.csv"
    for index, row in enumerate(_strict_csv_rows(source_path), start=1):
        if row["campaign_id"] != CAMPAIGN_ID or row["segment_id"] not in SEGMENT_IDS:
            raise EpisodeV3Error("quality interval identity drift")
        mask_type = row["mask_type"]
        if mask_type == "segment_epoch":
            start_ns = int(row["first_common_ts_ns"])
            end_ns = int(row["last_common_ts_ns"])
            interval_id = f"{row['segment_id']}:segment_epoch:0"
            core_censor = "false"
            aux_degraded = "false"
        elif mask_type == "auxiliary_degraded_interval":
            start_ns = int(row["mask_start_ts_ns"])
            end_ns = int(row["mask_end_ts_ns"])
            interval_id = f"{row['segment_id']}:{row['track_id']}:{index}"
            core_censor = "false"
            aux_degraded = "true"
        else:
            raise EpisodeV3Error(f"unknown quality mask type: {mask_type}")
        rows.append(
            {
                "session_id": SESSION_ID,
                "segment_id": row["segment_id"],
                "interval_id": interval_id,
                "mask_type": mask_type,
                "track_id": row["track_id"],
                "start_ts_ns": str(start_ns),
                "end_ts_ns": str(end_ns),
                "policy": row["policy"],
                "reason": row["reason"],
                "core_outcome_censor": core_censor,
                "auxiliary_feature_degraded": aux_degraded,
            }
        )
    if sum(row["mask_type"] == "segment_epoch" for row in rows) != 8:
        raise EpisodeV3Error("segment epoch interval count drift")
    auxiliary = [row for row in rows if row["mask_type"] == "auxiliary_degraded_interval"]
    if len(auxiliary) != 2 or {row["track_id"] for row in auxiliary} != {
        "asset_context",
        "main_all_mids",
    }:
        raise EpisodeV3Error("Jul30 auxiliary degraded interval drift")
    return rows


def _candidate_iter() -> Iterator[Candidate]:
    projection_path = STAGE3_DIR / "candidate_audit_projection.csv.gz"
    membership_path = STAGE2_DIR / "candidate_episode_membership.csv.gz"
    if sha256_file(projection_path) != EXPECTED_STAGE3_PROJECTION_SHA256:
        raise EpisodeV3Error("Stage 3 candidate projection SHA drift")
    if sha256_file(membership_path) != EXPECTED_STAGE2_MEMBERSHIP_SHA256:
        raise EpisodeV3Error("Stage 2 membership SHA drift")
    projection_iter = _strict_csv_rows(projection_path)
    membership_iter = _strict_csv_rows(membership_path)
    seen = 0
    for projection, membership in zip(projection_iter, membership_iter, strict=True):
        if projection["session_id"] != membership["session_id"]:
            raise EpisodeV3Error("candidate projection/membership session drift")
        if projection["session_id"] != SESSION_ID:
            break
        candidate_id = (
            f"{SESSION_ID}:{projection['segment_id']}:{projection['candidate_seq']}"
        )
        if membership["candidate_id"] != candidate_id:
            raise EpisodeV3Error("candidate projection/membership ID drift")
        exact_pairs = (
            ("segment_id", "segment_id"),
            ("aggressor_side", "aggressor_side"),
            ("direction_sign", "direction_sign"),
            ("shock_ts_ns", "shock_ts_ns"),
            ("decision_ts_ns", "decision_ts_ns"),
            ("impact_ratio", "impact_ratio"),
            ("pre_state_ts_ns", "pre_state_ts_ns"),
            ("primary_episode", "primary_episode"),
            ("rejection_reason", "rejection_reason"),
        )
        if any(projection[left] != membership[right] for left, right in exact_pairs):
            raise EpisodeV3Error(f"{candidate_id}: Stage 2/3 join drift")
        primary = _parse_bool(
            projection["primary_episode"], label=f"{candidate_id}:primary_episode"
        )
        detector_decision = _optional_int(projection, "decision_ts_ns")
        if primary and detector_decision is None:
            raise EpisodeV3Error(f"{candidate_id}: confirmed candidate lacks decision")
        if not primary and not projection["rejection_reason"]:
            raise EpisodeV3Error(f"{candidate_id}: rejected candidate lacks reason")
        if projection["attribution"] not in CLASSIFICATION_ALLOWED_VALUES:
            raise EpisodeV3Error(
                f"{candidate_id}: Stage 3 attribution/classification value drift"
            )
        yield Candidate(
            candidate_id=candidate_id,
            episode_id=candidate_id,
            segment_id=projection["segment_id"],
            candidate_seq=int(projection["candidate_seq"]),
            aggressor_side=projection["aggressor_side"],
            direction_sign=int(projection["direction_sign"]),
            burst_start_ts_ns=int(projection["burst_start_ts_ns"]),
            burst_end_ts_ns=int(projection["burst_end_ts_ns"]),
            burst_duration_ms=float(projection["burst_duration_ms"]),
            burst_trade_count=int(projection["burst_trade_count"]),
            burst_trade_qty=float(projection["burst_trade_qty"]),
            touch_trade_qty=float(projection["touch_trade_qty"]),
            touch_trade_qty_at_shock=float(projection["touch_trade_qty_at_shock"]),
            touch_trade_qty_through_decision=_optional_float(
                projection, "touch_trade_qty_through_decision"
            ),
            post_decision_burst_trade_count=_optional_int(
                projection, "post_decision_burst_trade_count"
            ),
            pre_state_ts_ns=int(projection["pre_state_ts_ns"]),
            pre_best_px=float(projection["pre_best_px"]),
            pre_best_qty=float(projection["pre_best_qty"]),
            shock_ts_ns=int(projection["shock_ts_ns"]),
            impact_ratio=float(projection["impact_ratio"]),
            shock_impact_ratio=float(projection["shock_impact_ratio"]),
            detector_decision_ts_ns=detector_decision,
            confirmation_lag_ms=_optional_float(projection, "confirmation_lag_ms"),
            confirmed_best_px=_optional_float(projection, "confirmed_best_px"),
            confirmed_best_qty=_optional_float(projection, "confirmed_best_qty"),
            price_level_depleted=(
                None
                if projection["price_level_depleted"] == ""
                else _parse_bool(
                    projection["price_level_depleted"],
                    label=f"{candidate_id}:price_level_depleted",
                )
            ),
            queue_drop_ratio=_optional_float(projection, "queue_drop_ratio"),
            confirmed_removed_qty=_optional_float(
                projection, "confirmed_removed_qty"
            ),
            trade_explained_ratio=_optional_float(
                projection, "trade_explained_ratio"
            ),
            attribution=projection["attribution"],
            pre_hl_bbo_ts_ns=_optional_int(projection, "pre_hl_bbo_ts_ns"),
            pre_hl_bbo_age_ms=_optional_float(projection, "pre_hl_bbo_age_ms"),
            pre_hl_fast_source_ts_ns=_optional_int(
                projection, "pre_hl_fast_source_ts_ns"
            ),
            pre_hl_fast_age_ms=_optional_float(projection, "pre_hl_fast_age_ms"),
            primary_episode=primary,
            rejection_reason=projection["rejection_reason"],
            connection_epoch_id=membership["connection_epoch_id"],
            segment_start_ts_ns=int(membership["segment_start_ts_ns"]),
            segment_end_ts_ns=int(membership["segment_end_ts_ns"]),
            cluster_id=membership["cluster_id"],
            continuous_flow_episode_id=membership[
                "continuous_flow_episode_id"
            ],
            overlap_block_id=membership["overlap_block_id"],
            window_end_ts_ns=int(membership["window_end_ts_ns"]),
        )
        seen += 1
    if seen != EXPECTED_COUNTS["family_a"]:
        raise EpisodeV3Error(
            f"Jul30 candidate count drift: expected={EXPECTED_COUNTS['family_a']} "
            f"observed={seen}"
        )


def _group_candidates_by_segment() -> Iterator[tuple[str, list[Candidate]]]:
    current_segment = ""
    rows: list[Candidate] = []
    for candidate in _candidate_iter():
        if current_segment and candidate.segment_id != current_segment:
            yield current_segment, rows
            rows = []
        current_segment = candidate.segment_id
        rows.append(candidate)
    if rows:
        yield current_segment, rows


def _load_timeline(spec: SegmentSpec) -> TimelineStore:
    store = TimelineStore(segment_id=spec.segment_id)
    previous_ts = -1
    previous_mid: float | None = None
    previous_fast_observed_at = -1
    count = 0
    with gzip.open(spec.timeline_path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or ())
        required = {
            "campaign_id",
            "segment_id",
            "profile_id",
            "common_seq",
            "common_ts_ns",
            "trigger_track",
            "binance_local_ts_ns",
            "binance_bid_1_px",
            "binance_bid_1_qty",
            "binance_ask_1_px",
            "binance_ask_1_qty",
            "hyperliquid_fast_local_ts_ns",
            "hyperliquid_fast_bid_1_px",
            "hyperliquid_fast_bid_1_qty",
            "hyperliquid_fast_ask_1_px",
            "hyperliquid_fast_ask_1_qty",
        }
        required.update(
            f"{venue}_{side}_{level}_{suffix}"
            for venue in ("binance", "hyperliquid_fast")
            for side in ("bid", "ask")
            for level in range(1, 6)
            for suffix in ("px", "qty")
        )
        if not required.issubset(fields):
            raise EpisodeV3Error(f"{spec.segment_id}: common timeline schema drift")
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise EpisodeV3Error(
                    f"{spec.segment_id}: common timeline row-width drift {row_number}"
                )
            if (
                row["campaign_id"] != CAMPAIGN_ID
                or row["segment_id"] != spec.segment_id
                or row["profile_id"] != PROFILE_ID
            ):
                raise EpisodeV3Error(
                    f"{spec.segment_id}: common timeline identity drift"
                )
            ts_ns = int(row["common_ts_ns"])
            if ts_ns < previous_ts:
                raise EpisodeV3Error(
                    f"{spec.segment_id}: common timeline timestamp regression"
                )
            previous_ts = ts_ns
            binance_bid = float(row["binance_bid_1_px"])
            binance_ask = float(row["binance_ask_1_px"])
            fast_bid = float(row["hyperliquid_fast_bid_1_px"])
            fast_ask = float(row["hyperliquid_fast_ask_1_px"])
            if binance_ask <= binance_bid or fast_ask <= fast_bid:
                raise EpisodeV3Error(f"{spec.segment_id}: crossed timeline state")
            binance_mid = (binance_bid + binance_ask) / 2.0
            squared_return = (
                0.0
                if previous_mid in (None, 0.0)
                else ((binance_mid - previous_mid) / previous_mid) ** 2
            )
            previous_mid = binance_mid
            store.ts.append(ts_ns)
            store.common_seq.append(int(row["common_seq"]))
            store.binance_observed_at.append(int(row["binance_local_ts_ns"]))
            store.binance_bid.append(binance_bid)
            store.binance_ask.append(binance_ask)
            store.binance_bid_qty.append(float(row["binance_bid_1_qty"]))
            store.binance_ask_qty.append(float(row["binance_ask_1_qty"]))
            store.binance_bid_top5.append(
                sum(float(row[f"binance_bid_{level}_qty"]) for level in range(1, 6))
            )
            store.binance_ask_top5.append(
                sum(float(row[f"binance_ask_{level}_qty"]) for level in range(1, 6))
            )
            fast_observed_at = int(row["hyperliquid_fast_local_ts_ns"])
            store.hl_fast_observed_at.append(fast_observed_at)
            store.hl_fast_bid.append(fast_bid)
            store.hl_fast_ask.append(fast_ask)
            store.hl_fast_bid_qty.append(float(row["hyperliquid_fast_bid_1_qty"]))
            store.hl_fast_ask_qty.append(float(row["hyperliquid_fast_ask_1_qty"]))
            store.hl_fast_bid_top5.append(
                sum(
                    float(row[f"hyperliquid_fast_bid_{level}_qty"])
                    for level in range(1, 6)
                )
            )
            store.hl_fast_ask_top5.append(
                sum(
                    float(row[f"hyperliquid_fast_ask_{level}_qty"])
                    for level in range(1, 6)
                )
            )
            store.trigger_track.append(row["trigger_track"])
            if fast_observed_at != previous_fast_observed_at:
                if fast_observed_at > ts_ns:
                    raise EpisodeV3Error(
                        f"{spec.segment_id}: future fast-L2 state in timeline"
                    )
                store.fast_event_ts.append(fast_observed_at)
                store.fast_event_version.append(fast_observed_at)
                previous_fast_observed_at = fast_observed_at
            store.midpoint_sq_return_prefix.append(
                store.midpoint_sq_return_prefix[-1] + squared_return
            )
            count += 1
    if count != spec.expected_rows["timeline"]:
        raise EpisodeV3Error(f"{spec.segment_id}: timeline row-count drift")
    if not store.ts:
        raise EpisodeV3Error(f"{spec.segment_id}: empty timeline")
    return store


def _load_binance_hot(
    spec: SegmentSpec,
) -> tuple[NumericEventStore, NumericEventStore, array, array]:
    _validate_header(spec.binance_hot_path, BINANCE_HOT_FIELDS_V1)
    bbo = NumericEventStore(segment_id=spec.segment_id, channel="binance_book_ticker")
    trades = NumericEventStore(segment_id=spec.segment_id, channel="binance_trade")
    signed_prefix = array("d", [0.0])
    ofi_prefix = array("d", [0.0])
    previous_bbo: tuple[float, float, float, float] | None = None
    previous_ts = -1
    count = 0
    with gzip.open(
        spec.binance_hot_path, "rt", encoding="utf-8", newline=""
    ) as fh:
        for row_number, row in enumerate(csv.DictReader(fh), start=2):
            if None in row:
                raise EpisodeV3Error(
                    f"{spec.segment_id}: Binance hot row-width drift {row_number}"
                )
            if row["segment_id"] != spec.segment_id or row["symbol"] != "SKHYNIXUSDT":
                raise EpisodeV3Error(f"{spec.segment_id}: Binance hot identity drift")
            ts_ns = int(row["local_ts_ns"])
            if ts_ns < previous_ts:
                raise EpisodeV3Error(
                    f"{spec.segment_id}: Binance hot timestamp regression"
                )
            previous_ts = ts_ns
            event_type = row["event_type"]
            if event_type == "bookTicker":
                bid = float(row["bid_px"])
                bid_qty = float(row["bid_qty"])
                ask = float(row["ask_px"])
                ask_qty = float(row["ask_qty"])
                if ask <= bid:
                    raise EpisodeV3Error(f"{spec.segment_id}: crossed Binance BBO")
                bbo.ts.append(ts_ns)
                bbo.seq.append(int(row["event_seq"]))
                bbo.raw_seq.append(int(row["source_raw_seq"]))
                bbo.item_index.append(0)
                bbo.exchange_ts.append(int(row["exchange_ts_ns"] or 0))
                bbo.x1.append(bid)
                bbo.x2.append(bid_qty)
                bbo.x3.append(ask)
                bbo.x4.append(ask_qty)
                bbo.side.append(0)
                if previous_bbo is None:
                    ofi = 0.0
                else:
                    prev_bid, prev_bid_qty, prev_ask, prev_ask_qty = previous_bbo
                    bid_contribution = (
                        bid_qty
                        if bid > prev_bid
                        else -prev_bid_qty
                        if bid < prev_bid
                        else bid_qty - prev_bid_qty
                    )
                    ask_contribution = (
                        -ask_qty
                        if ask < prev_ask
                        else prev_ask_qty
                        if ask > prev_ask
                        else prev_ask_qty - ask_qty
                    )
                    ofi = bid_contribution + ask_contribution
                previous_bbo = (bid, bid_qty, ask, ask_qty)
                ofi_prefix.append(ofi_prefix[-1] + ofi)
            elif event_type == "trade":
                side = 1 if row["aggressor_side"] == "buy" else -1
                qty = float(row["trade_qty"])
                trades.ts.append(ts_ns)
                trades.seq.append(int(row["event_seq"]))
                trades.raw_seq.append(int(row["source_raw_seq"]))
                trades.item_index.append(0)
                trades.exchange_ts.append(int(row["exchange_ts_ns"] or 0))
                trades.x1.append(float(row["trade_px"]))
                trades.x2.append(qty)
                trades.x3.append(0.0)
                trades.x4.append(0.0)
                trades.side.append(side)
                signed_prefix.append(signed_prefix[-1] + side * qty)
            else:
                raise EpisodeV3Error(
                    f"{spec.segment_id}: unexpected Binance event type {event_type}"
                )
            count += 1
    if count != spec.expected_rows["binance_hot_events"]:
        raise EpisodeV3Error(f"{spec.segment_id}: Binance hot row-count drift")
    if len(ofi_prefix) != len(bbo.ts) + 1:
        raise EpisodeV3Error(f"{spec.segment_id}: Binance OFI prefix drift")
    if len(signed_prefix) != len(trades.ts) + 1:
        raise EpisodeV3Error(f"{spec.segment_id}: Binance trade prefix drift")
    return bbo, trades, signed_prefix, ofi_prefix


def _load_hyperliquid_hot(
    spec: SegmentSpec,
) -> tuple[NumericEventStore, NumericEventStore]:
    _validate_header(spec.hyperliquid_hot_path, HYPERLIQUID_HOT_FIELDS_V1)
    bbo = NumericEventStore(segment_id=spec.segment_id, channel="hyperliquid_bbo")
    trades = NumericEventStore(segment_id=spec.segment_id, channel="hyperliquid_trade")
    previous_ts = -1
    count = 0
    with gzip.open(
        spec.hyperliquid_hot_path, "rt", encoding="utf-8", newline=""
    ) as fh:
        for row_number, row in enumerate(csv.DictReader(fh), start=2):
            if None in row:
                raise EpisodeV3Error(
                    f"{spec.segment_id}: Hyperliquid hot row-width drift {row_number}"
                )
            if row["segment_id"] != spec.segment_id or row["coin"] != "xyz:SKHX":
                raise EpisodeV3Error(
                    f"{spec.segment_id}: Hyperliquid hot identity drift"
                )
            ts_ns = int(row["local_ts_ns"])
            if ts_ns < previous_ts:
                raise EpisodeV3Error(
                    f"{spec.segment_id}: Hyperliquid hot timestamp regression"
                )
            previous_ts = ts_ns
            target = bbo if row["event_type"] == "bbo" else trades
            if row["event_type"] not in {"bbo", "trade"}:
                raise EpisodeV3Error(
                    f"{spec.segment_id}: unexpected Hyperliquid event type"
                )
            target.ts.append(ts_ns)
            target.seq.append(int(row["event_seq"]))
            target.raw_seq.append(int(row["source_raw_seq"]))
            target.item_index.append(int(row["source_item_index"] or 0))
            target.exchange_ts.append(int(row["exchange_ts_ns"] or 0))
            if row["event_type"] == "bbo":
                bid = float(row["bid_px"])
                ask = float(row["ask_px"])
                if ask <= bid:
                    raise EpisodeV3Error(
                        f"{spec.segment_id}: crossed Hyperliquid BBO"
                    )
                target.x1.append(bid)
                target.x2.append(float(row["bid_qty"]))
                target.x3.append(ask)
                target.x4.append(float(row["ask_qty"]))
                target.side.append(0)
            else:
                target.x1.append(float(row["trade_px"]))
                target.x2.append(float(row["trade_qty"]))
                target.x3.append(0.0)
                target.x4.append(0.0)
                target.side.append(1 if row["trade_side"] == "B" else -1)
            count += 1
    if count != spec.expected_rows["hyperliquid_hot_events"]:
        raise EpisodeV3Error(f"{spec.segment_id}: Hyperliquid hot row-count drift")
    return bbo, trades


def _load_auxiliary_intervals(
    spec: SegmentSpec, quality_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    _validate_header(spec.auxiliary_path, AUXILIARY_FIELDS_V1)
    count = sum(1 for _ in _strict_csv_rows(spec.auxiliary_path, AUXILIARY_FIELDS_V1))
    if count != spec.expected_rows["hyperliquid_auxiliary_events"]:
        raise EpisodeV3Error(f"{spec.segment_id}: auxiliary row-count drift")
    return [
        {
            "interval_id": str(row["interval_id"]),
            "track_id": str(row["track_id"]),
            "start_ts_ns": int(row["start_ts_ns"]),
            "end_ts_ns": int(row["end_ts_ns"]),
        }
        for row in quality_rows
        if row["segment_id"] == spec.segment_id
        and row["mask_type"] == "auxiliary_degraded_interval"
    ]


def _load_segment_data(
    spec: SegmentSpec,
    quality_rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[Candidate],
) -> SegmentData:
    _validate_decision_labels(spec)
    timeline = _load_timeline(spec)
    binance_bbo, binance_trades, signed_prefix, ofi_prefix = _load_binance_hot(spec)
    hl_bbo, hl_trades = _load_hyperliquid_hot(spec)
    auxiliary_intervals = _load_auxiliary_intervals(spec, quality_rows)
    candidate_ts = array("q", (candidate.shock_ts_ns for candidate in candidates))
    candidate_direction = array("b", (candidate.direction_sign for candidate in candidates))
    if any(
        candidate.segment_start_ts_ns != candidates[0].segment_start_ts_ns
        or candidate.segment_end_ts_ns != candidates[0].segment_end_ts_ns
        or candidate.connection_epoch_id != "0"
        for candidate in candidates
    ):
        raise EpisodeV3Error(f"{spec.segment_id}: accepted boundary/epoch drift")
    if spec.segment_start_ts_ns < candidates[0].segment_start_ts_ns:
        raise EpisodeV3Error(f"{spec.segment_id}: common state predates structural segment")
    return SegmentData(
        spec=spec,
        timeline=timeline,
        binance_bbo=binance_bbo,
        binance_trades=binance_trades,
        hl_bbo=hl_bbo,
        hl_trades=hl_trades,
        auxiliary_intervals=auxiliary_intervals,
        binance_signed_qty_prefix=signed_prefix,
        binance_ofi_prefix=ofi_prefix,
        candidate_ts=candidate_ts,
        candidate_direction=candidate_direction,
    )


def _auxiliary_degraded_ids(data: SegmentData, ts_ns: int) -> tuple[str, ...]:
    return tuple(
        interval["interval_id"]
        for interval in data.auxiliary_intervals
        if interval["start_ts_ns"] <= ts_ns <= interval["end_ts_ns"]
    )


def _nominal_underlying_clock_state(ts_ns: int) -> str:
    dt = datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=ZoneInfo("Asia/Seoul"))
    minute = dt.hour * 60 + dt.minute
    if 8 * 60 + 30 <= minute < 9 * 60:
        return "pre_open_or_auction"
    if 9 * 60 <= minute < 15 * 60 + 20:
        return "continuous_trading"
    if 15 * 60 + 20 <= minute < 18 * 60:
        return "closing_or_post_close"
    return "closed"


def _mid(bid: float, ask: float) -> float:
    return (bid + ask) / 2.0


def _spread_metrics(
    binance_bid: float,
    binance_ask: float,
    hl_bid: float,
    hl_ask: float,
    direction_sign: int,
) -> tuple[float, float, float]:
    reference = (_mid(binance_bid, binance_ask) + _mid(hl_bid, hl_ask)) / 2.0
    if reference <= 0:
        raise EpisodeV3Error("non-positive cross-venue reference midpoint")
    d_bh = (binance_bid - hl_ask) / reference * 10_000.0
    d_hb = (hl_bid - binance_ask) / reference * 10_000.0
    return d_bh, d_hb, d_bh if direction_sign == 1 else d_hb


def _range_triplet(store: NumericEventStore, start_ns: int, end_ns: int) -> tuple[str, str, str]:
    left, right = store.range_indices(start_ns, end_ns)
    if left >= right:
        return "", "", "0"
    return str(store.seq[left]), str(store.seq[right - 1]), str(right - left)


def _timeline_range_triplet(
    store: TimelineStore, start_ns: int, end_ns: int
) -> tuple[str, str, str]:
    left = bisect.bisect_left(store.ts, start_ns)
    right = bisect.bisect_right(store.ts, end_ns)
    if left >= right:
        return "", "", "0"
    return (
        str(store.common_seq[left]),
        str(store.common_seq[right - 1]),
        str(right - left),
    )


def _fast_range_triplet(
    store: TimelineStore, start_ns: int, end_ns: int
) -> tuple[str, str, str]:
    left = bisect.bisect_left(store.fast_event_ts, start_ns)
    right = bisect.bisect_right(store.fast_event_ts, end_ns)
    if left >= right:
        return "", "", "0"
    return (
        str(store.fast_event_version[left]),
        str(store.fast_event_version[right - 1]),
        str(right - left),
    )


def _range_row(
    candidate: Candidate,
    data: SegmentData,
    *,
    family_view: str,
    landmark_ts_ns: int,
) -> dict[str, str]:
    window_start = max(
        candidate.segment_start_ts_ns, landmark_ts_ns - OUTCOME_HORIZON_NS
    )
    window_end = min(
        candidate.segment_end_ts_ns, landmark_ts_ns + OUTCOME_HORIZON_NS
    )
    binance_bbo = _range_triplet(data.binance_bbo, window_start, window_end)
    binance_trade = _range_triplet(data.binance_trades, window_start, window_end)
    hl_bbo = _range_triplet(data.hl_bbo, window_start, window_end)
    hl_trade = _range_triplet(data.hl_trades, window_start, window_end)
    fast = _fast_range_triplet(data.timeline, window_start, window_end)
    timeline = _timeline_range_triplet(data.timeline, window_start, window_end)
    return {
        "candidate_id": candidate.candidate_id,
        "episode_id": candidate.episode_id,
        "family_view": family_view,
        "landmark_ts_ns": str(landmark_ts_ns),
        "window_start_ts_ns": str(window_start),
        "window_end_ts_ns": str(window_end),
        "connection_epoch_id": candidate.connection_epoch_id,
        "binance_book_ticker_first_event_seq": binance_bbo[0],
        "binance_book_ticker_last_event_seq": binance_bbo[1],
        "binance_book_ticker_count": binance_bbo[2],
        "binance_trade_first_event_seq": binance_trade[0],
        "binance_trade_last_event_seq": binance_trade[1],
        "binance_trade_count": binance_trade[2],
        "hyperliquid_bbo_first_event_seq": hl_bbo[0],
        "hyperliquid_bbo_last_event_seq": hl_bbo[1],
        "hyperliquid_bbo_count": hl_bbo[2],
        "hyperliquid_trade_first_event_seq": hl_trade[0],
        "hyperliquid_trade_last_event_seq": hl_trade[1],
        "hyperliquid_trade_count": hl_trade[2],
        "hyperliquid_fast_l2_first_book_version": fast[0],
        "hyperliquid_fast_l2_last_book_version": fast[1],
        "hyperliquid_fast_l2_count": fast[2],
        "common_l2_first_common_seq": timeline[0],
        "common_l2_last_common_seq": timeline[1],
        "common_l2_count": timeline[2],
        "inside_segment_start": _bool_text(
            landmark_ts_ns - OUTCOME_HORIZON_NS >= candidate.segment_start_ts_ns
        ),
        "inside_segment_end": _bool_text(
            landmark_ts_ns + OUTCOME_HORIZON_NS <= candidate.segment_end_ts_ns
        ),
    }


def _grid_row(
    candidate: Candidate,
    data: SegmentData,
    *,
    family_view: str,
    decision_landmark: str,
    landmark_ts_ns: int,
    relative_ms: int,
) -> dict[str, str]:
    requested_ts = landmark_ts_ns + relative_ms * 1_000_000
    base = {
        "candidate_id": candidate.candidate_id,
        "episode_id": candidate.episode_id,
        "family_view": family_view,
        "decision_landmark": decision_landmark,
        "landmark_ts_ns": str(landmark_ts_ns),
        "requested_relative_ms": str(relative_ms),
        "requested_ts_ns": str(requested_ts),
        "inside_segment": "false",
        "availability_reason": "outside_segment",
        "connection_epoch_id": candidate.connection_epoch_id,
        "core_degraded_mask": "",
        "auxiliary_degraded_mask": "",
    }
    if not (
        candidate.segment_start_ts_ns <= requested_ts <= candidate.segment_end_ts_ns
    ):
        return base
    timeline_index = data.timeline.asof_index(requested_ts)
    bbo_index = data.hl_bbo.asof_index(requested_ts)
    if timeline_index < 0 or bbo_index < 0:
        base.update(
            {
                "inside_segment": "true",
                "availability_reason": "no_asof_state",
            }
        )
        return base
    binance_observed = int(data.timeline.binance_observed_at[timeline_index])
    fast_observed = int(data.timeline.hl_fast_observed_at[timeline_index])
    bbo_observed = int(data.hl_bbo.ts[bbo_index])
    if max(binance_observed, fast_observed, bbo_observed) > requested_ts:
        raise EpisodeV3Error(f"{candidate.candidate_id}: future fixed-grid join")
    sign = candidate.direction_sign
    binance_bid = float(data.timeline.binance_bid[timeline_index])
    binance_ask = float(data.timeline.binance_ask[timeline_index])
    hl_fast_bid = float(data.timeline.hl_fast_bid[timeline_index])
    hl_fast_ask = float(data.timeline.hl_fast_ask[timeline_index])
    hl_bbo_bid = float(data.hl_bbo.x1[bbo_index])
    hl_bbo_ask = float(data.hl_bbo.x3[bbo_index])
    d_bh, d_hb, risk_gap = _spread_metrics(
        binance_bid, binance_ask, hl_bbo_bid, hl_bbo_ask, sign
    )
    auxiliary_ids = _auxiliary_degraded_ids(data, requested_ts)
    bin_impacted = (
        data.timeline.binance_ask_qty[timeline_index]
        if sign == 1
        else data.timeline.binance_bid_qty[timeline_index]
    )
    bin_opposite = (
        data.timeline.binance_bid_qty[timeline_index]
        if sign == 1
        else data.timeline.binance_ask_qty[timeline_index]
    )
    bin_top5_impacted = (
        data.timeline.binance_ask_top5[timeline_index]
        if sign == 1
        else data.timeline.binance_bid_top5[timeline_index]
    )
    bin_top5_opposite = (
        data.timeline.binance_bid_top5[timeline_index]
        if sign == 1
        else data.timeline.binance_ask_top5[timeline_index]
    )
    hl_bbo_impacted = data.hl_bbo.x4[bbo_index] if sign == 1 else data.hl_bbo.x2[bbo_index]
    hl_bbo_opposite = data.hl_bbo.x2[bbo_index] if sign == 1 else data.hl_bbo.x4[bbo_index]
    hl_fast_impacted = (
        data.timeline.hl_fast_ask_top5[timeline_index]
        if sign == 1
        else data.timeline.hl_fast_bid_top5[timeline_index]
    )
    hl_fast_opposite = (
        data.timeline.hl_fast_bid_top5[timeline_index]
        if sign == 1
        else data.timeline.hl_fast_ask_top5[timeline_index]
    )
    base.update(
        {
            "inside_segment": "true",
            "availability_reason": "available",
            "core_degraded_mask": "",
            "auxiliary_degraded_mask": "|".join(auxiliary_ids),
            "binance_bid_px": _float_text(binance_bid),
            "binance_ask_px": _float_text(binance_ask),
            "binance_mid_px": _float_text(_mid(binance_bid, binance_ask)),
            "binance_impacted_best_qty": _float_text(float(bin_impacted)),
            "binance_opposite_best_qty": _float_text(float(bin_opposite)),
            "binance_impacted_top5_qty": _float_text(float(bin_top5_impacted)),
            "binance_opposite_top5_qty": _float_text(float(bin_top5_opposite)),
            "binance_observed_at_ns": str(binance_observed),
            "binance_source_event_id": data.timeline.source_event_id(timeline_index),
            "binance_source_book_version": str(
                data.timeline.common_seq[timeline_index]
            ),
            "binance_strict_asof_ts_ns": str(requested_ts),
            "binance_effective_relative_ns": str(
                binance_observed - landmark_ts_ns
            ),
            "binance_source_age_ns": str(requested_ts - binance_observed),
            "binance_no_new_information": _bool_text(binance_observed < requested_ts),
            "hyperliquid_bbo_bid_px": _float_text(hl_bbo_bid),
            "hyperliquid_bbo_ask_px": _float_text(hl_bbo_ask),
            "hyperliquid_bbo_mid_px": _float_text(_mid(hl_bbo_bid, hl_bbo_ask)),
            "hyperliquid_bbo_impacted_qty": _float_text(float(hl_bbo_impacted)),
            "hyperliquid_bbo_opposite_qty": _float_text(float(hl_bbo_opposite)),
            "hyperliquid_bbo_observed_at_ns": str(bbo_observed),
            "hyperliquid_bbo_source_event_id": data.hl_bbo.source_event_id(bbo_index),
            "hyperliquid_bbo_source_book_version": str(data.hl_bbo.seq[bbo_index]),
            "hyperliquid_bbo_strict_asof_ts_ns": str(requested_ts),
            "hyperliquid_bbo_effective_relative_ns": str(
                bbo_observed - landmark_ts_ns
            ),
            "hyperliquid_bbo_source_age_ns": str(requested_ts - bbo_observed),
            "hyperliquid_bbo_no_new_information": _bool_text(
                bbo_observed < requested_ts
            ),
            "hyperliquid_fast_bid_px": _float_text(hl_fast_bid),
            "hyperliquid_fast_ask_px": _float_text(hl_fast_ask),
            "hyperliquid_fast_mid_px": _float_text(_mid(hl_fast_bid, hl_fast_ask)),
            "hyperliquid_fast_impacted_top5_qty": _float_text(
                float(hl_fast_impacted)
            ),
            "hyperliquid_fast_opposite_top5_qty": _float_text(
                float(hl_fast_opposite)
            ),
            "hyperliquid_fast_observed_at_ns": str(fast_observed),
            "hyperliquid_fast_source_event_id": data.timeline.fast_source_event_id(
                timeline_index
            ),
            "hyperliquid_fast_source_book_version": str(fast_observed),
            "hyperliquid_fast_strict_asof_ts_ns": str(requested_ts),
            "hyperliquid_fast_effective_relative_ns": str(
                fast_observed - landmark_ts_ns
            ),
            "hyperliquid_fast_source_age_ns": str(requested_ts - fast_observed),
            "hyperliquid_fast_no_new_information": _bool_text(
                fast_observed < requested_ts
            ),
            "d_bh_bps": _float_text(d_bh),
            "d_hb_bps": _float_text(d_hb),
            "risk_gap_bps": _float_text(risk_gap),
        }
    )
    return base


def _event_count_rows(
    candidate: Candidate,
    data: SegmentData,
    *,
    family_view: str,
    decision_landmark: str,
    landmark_ts_ns: int,
) -> Iterator[dict[str, str]]:
    channel_specs: tuple[
        tuple[str, Sequence[int], Any, Any], ...
    ] = (
        (
            "hyperliquid_bbo",
            data.hl_bbo.ts,
            lambda index: data.hl_bbo.source_event_id(index),
            lambda index: int(data.hl_bbo.ts[index]),
        ),
        (
            "hyperliquid_trade",
            data.hl_trades.ts,
            lambda index: data.hl_trades.source_event_id(index),
            lambda index: int(data.hl_trades.ts[index]),
        ),
        (
            "hyperliquid_fast_l2",
            data.timeline.fast_event_ts,
            lambda index: (
                f"{SESSION_ID}:{candidate.segment_id}:hyperliquid_fast_l2:"
                f"{data.timeline.fast_event_version[index]}"
            ),
            lambda index: int(data.timeline.fast_event_ts[index]),
        ),
    )
    for channel, timestamps, event_id, observed_at in channel_specs:
        start_index = bisect.bisect_right(timestamps, landmark_ts_ns)
        row: dict[str, str] = {
            "candidate_id": candidate.candidate_id,
            "episode_id": candidate.episode_id,
            "family_view": family_view,
            "decision_landmark": decision_landmark,
            "landmark_ts_ns": str(landmark_ts_ns),
            "channel": channel,
        }
        for count in EVENT_COUNT_POINTS:
            index = start_index + count - 1
            prefix = f"event_{count}_"
            if index >= len(timestamps):
                row[prefix + "source_event_id"] = ""
                row[prefix + "observed_at_ns"] = ""
                row[prefix + "relative_ns"] = ""
                row[prefix + "inside_segment"] = "false"
                row[prefix + "availability_reason"] = "source_exhausted"
                continue
            ts_ns = observed_at(index)
            if ts_ns > candidate.segment_end_ts_ns:
                row[prefix + "source_event_id"] = ""
                row[prefix + "observed_at_ns"] = ""
                row[prefix + "relative_ns"] = ""
                row[prefix + "inside_segment"] = "false"
                row[prefix + "availability_reason"] = "segment_boundary"
                continue
            row[prefix + "source_event_id"] = event_id(index)
            row[prefix + "observed_at_ns"] = str(ts_ns)
            row[prefix + "relative_ns"] = str(ts_ns - landmark_ts_ns)
            row[prefix + "inside_segment"] = "true"
            row[prefix + "availability_reason"] = "available"
        yield row


def _prefix_value(
    timestamps: Sequence[int],
    prefix: Sequence[float],
    start_ns: int,
    end_ns: int,
) -> float:
    left = bisect.bisect_left(timestamps, start_ns)
    right = bisect.bisect_right(timestamps, end_ns)
    return float(prefix[right] - prefix[left])


def _count_window(timestamps: Sequence[int], start_ns: int, end_ns: int) -> int:
    return bisect.bisect_right(timestamps, end_ns) - bisect.bisect_left(
        timestamps, start_ns
    )


def _candidate_direction_counts(
    data: SegmentData,
    candidate: Candidate,
    window_ms: int,
) -> tuple[int, int]:
    start_ns = candidate.shock_ts_ns - window_ms * 1_000_000
    left = bisect.bisect_left(data.candidate_ts, start_ns)
    right = bisect.bisect_left(data.candidate_ts, candidate.shock_ts_ns)
    same = sum(
        1
        for index in range(left, right)
        if int(data.candidate_direction[index]) == candidate.direction_sign
    )
    return same, right - left - same


def _frozen_burst_indices(
    candidate: Candidate,
    data: SegmentData,
) -> tuple[int, ...]:
    start = bisect.bisect_left(
        data.binance_trades.ts, candidate.burst_start_ts_ns
    )
    stop = bisect.bisect_right(
        data.binance_trades.ts, candidate.burst_start_ts_ns
    )
    matches: list[tuple[int, ...]] = []
    for origin_index in range(start, stop):
        if int(data.binance_trades.side[origin_index]) != candidate.direction_sign:
            continue
        if (
            float(data.binance_trades.x1[origin_index]) <= 0
            or float(data.binance_trades.x2[origin_index]) <= 0
        ):
            continue
        origin_ts = int(data.binance_trades.ts[origin_index])
        indices: list[int] = []
        for index in range(origin_index, len(data.binance_trades.ts)):
            ts_ns = int(data.binance_trades.ts[index])
            px = float(data.binance_trades.x1[index])
            qty = float(data.binance_trades.x2[index])
            side = int(data.binance_trades.side[index])
            if px == 0 and qty == 0:
                break
            if (
                side != candidate.direction_sign
                or ts_ns - origin_ts > 10_000_000
            ):
                break
            indices.append(index)
        if not indices:
            continue
        if (
            len(indices) == candidate.burst_trade_count
            and int(data.binance_trades.ts[indices[-1]])
            == candidate.burst_end_ts_ns
            and math.isclose(
                math.fsum(
                    float(data.binance_trades.x2[index]) for index in indices
                ),
                candidate.burst_trade_qty,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            matches.append(tuple(indices))
    if len(matches) != 1:
        raise EpisodeV3Error(
            f"{candidate.candidate_id}: frozen detector burst reconstruction drift"
        )
    return matches[0]


def _confirmation_timeline_index(
    candidate: Candidate,
    data: SegmentData,
) -> int:
    if candidate.t_confirm_ns is None:
        raise EpisodeV3Error(f"{candidate.candidate_id}: confirmation missing")
    confirmation_end = candidate.shock_ts_ns + 100_000_000
    decision_index = -1
    for index in range(
        bisect.bisect_left(data.timeline.ts, candidate.shock_ts_ns),
        len(data.timeline.ts),
    ):
        ts_ns = int(data.timeline.ts[index])
        if ts_ns > confirmation_end:
            break
        current_px = (
            float(data.timeline.binance_ask[index])
            if candidate.direction_sign == 1
            else float(data.timeline.binance_bid[index])
        )
        current_qty = (
            float(data.timeline.binance_ask_qty[index])
            if candidate.direction_sign == 1
            else float(data.timeline.binance_bid_qty[index])
        )
        depleted = (
            current_px > candidate.pre_best_px
            if candidate.direction_sign == 1
            else current_px < candidate.pre_best_px
        )
        dropped = (
            current_px == candidate.pre_best_px
            and current_qty <= candidate.pre_best_qty * 0.70
        )
        if depleted or dropped:
            decision_index = index
            break
    if (
        decision_index < 0
        or int(data.timeline.ts[decision_index]) != candidate.t_confirm_ns
    ):
        raise EpisodeV3Error(
            f"{candidate.candidate_id}: shared detector confirmation state drift"
        )
    return decision_index


def _confirmation_feature_context(
    candidate: Candidate,
    data: SegmentData,
) -> tuple[int, str, str, tuple[tuple[str, Any], ...]]:
    confirmation_index = _confirmation_timeline_index(candidate, data)
    frozen_burst = _frozen_burst_indices(candidate, data)
    prefix = tuple(
        index
        for index in frozen_burst
        if int(data.binance_trades.ts[index]) <= candidate.t_confirm_ns
    )
    if not prefix:
        raise EpisodeV3Error(
            f"{candidate.candidate_id}: empty frozen burst confirmation prefix"
        )
    confirmation_trade_qty = math.fsum(
        float(data.binance_trades.x2[index]) for index in prefix
    )
    confirmation_burst_duration_ms = (
        int(data.binance_trades.ts[prefix[-1]]) - candidate.burst_start_ts_ns
    ) / 1_000_000.0
    continuation = tuple(
        index
        for index in prefix
        if int(data.binance_trades.ts[index]) > candidate.shock_ts_ns
    )
    touch_through_decision = math.fsum(
        float(data.binance_trades.x2[index])
        for index in prefix
        if (
            float(data.binance_trades.x1[index]) >= candidate.pre_best_px
            if candidate.direction_sign == 1
            else float(data.binance_trades.x1[index]) <= candidate.pre_best_px
        )
    )
    if candidate.touch_trade_qty_through_decision is None or not math.isclose(
        touch_through_decision,
        candidate.touch_trade_qty_through_decision,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise EpisodeV3Error(
            f"{candidate.candidate_id}: detector burst touch prefix drift"
        )
    source_event_id = data.timeline.source_event_id(confirmation_index)
    source_book_version = str(data.timeline.common_seq[confirmation_index])
    values = (
        (
            "confirmed_burst_trade_count_through_decision",
            len(prefix),
        ),
        (
            "confirmed_burst_trade_qty_through_decision",
            confirmation_trade_qty,
        ),
        (
            "confirmed_burst_duration_through_decision_ms",
            confirmation_burst_duration_ms,
        ),
        (
            "confirmed_touch_trade_qty_through_decision",
            candidate.touch_trade_qty_through_decision,
        ),
        ("confirmed_impact_ratio_legacy", candidate.impact_ratio),
        ("confirmed_queue_drop_ratio", candidate.queue_drop_ratio),
        ("confirmed_price_level_depleted", candidate.price_level_depleted),
        ("confirmed_removed_qty", candidate.confirmed_removed_qty),
        ("confirmed_trade_explained_ratio", candidate.trade_explained_ratio),
        ("confirmed_attribution", candidate.attribution),
        ("confirmation_lag_ms", candidate.confirmation_lag_ms),
        ("post_candidate_trade_continuation_count", len(continuation)),
        (
            "post_candidate_trade_continuation_qty",
            math.fsum(
                float(data.binance_trades.x2[index])
                for index in continuation
            ),
        ),
        (
            "local_binance_source_event_count_candidate_to_confirm",
            _count_window(
                data.binance_bbo.ts,
                candidate.shock_ts_ns,
                candidate.t_confirm_ns,
            )
            + _count_window(
                data.binance_trades.ts,
                candidate.shock_ts_ns,
                candidate.t_confirm_ns,
            ),
        ),
    )
    if tuple(name for name, _value in values) != CONFIRMATION_FEATURE_NAMES:
        raise EpisodeV3Error("confirmation feature contract drift")
    return (
        int(data.timeline.ts[confirmation_index]),
        source_event_id,
        source_book_version,
        values,
    )


def _confirmation_feature_rows(
    candidate: Candidate,
    data: SegmentData,
) -> tuple[dict[str, str], ...]:
    if candidate.t_confirm_ns is None:
        raise EpisodeV3Error(f"{candidate.candidate_id}: confirmation missing")
    observed_at, source_event_id, source_book_version, values = (
        _confirmation_feature_context(candidate, data)
    )
    return tuple(
        _feature_row(
            candidate,
            family_view="family_b",
            decision_landmark="t_confirm",
            landmark_ts_ns=candidate.t_confirm_ns,
            feature_name=name,
            value=value,
            observed_at_ns=observed_at if value is not None else None,
            source_event_id=source_event_id if value is not None else "",
            source_book_version=(
                source_book_version if value is not None else ""
            ),
            availability_reason=(
                "available"
                if value is not None
                else "detector_field_unavailable"
            ),
        )
        for name, value in values
    )


def _derived_source_id(*source_ids: str) -> str:
    material = "|".join(value for value in source_ids if value)
    if not material:
        return ""
    return f"derived:{hashlib.sha256(material.encode()).hexdigest()}"


def _feature_row(
    candidate: Candidate,
    *,
    family_view: str,
    decision_landmark: str,
    landmark_ts_ns: int,
    feature_name: str,
    value: Any,
    observed_at_ns: int | None,
    source_event_id: str,
    source_book_version: str,
    availability_reason: str,
) -> dict[str, str]:
    unavailable = value is None or value == ""
    if unavailable:
        if availability_reason in ("", "available"):
            raise EpisodeV3Error(
                f"{candidate.candidate_id}:{feature_name}: unavailable reason missing"
            )
        value_text = ""
        observed_text = ""
        source_event_id = ""
        source_book_version = ""
    else:
        if observed_at_ns is None:
            raise EpisodeV3Error(
                f"{candidate.candidate_id}:{feature_name}: observed_at missing"
            )
        if observed_at_ns > landmark_ts_ns:
            raise EpisodeV3Error(
                f"{candidate.candidate_id}:{feature_name}: future observation"
            )
        if not source_event_id or not source_book_version:
            raise EpisodeV3Error(
                f"{candidate.candidate_id}:{feature_name}: source identity missing"
            )
        if isinstance(value, bool):
            value_text = _bool_text(value)
        elif isinstance(value, float):
            value_text = _float_text(value)
        else:
            value_text = str(value)
        observed_text = str(observed_at_ns)
        availability_reason = "available"
    return {
        "episode_id": candidate.episode_id,
        "candidate_id": candidate.candidate_id,
        "family_view": family_view,
        "decision_landmark": decision_landmark,
        "feature_name": feature_name,
        "value": value_text,
        "observed_at_ns": observed_text,
        "source_event_id": source_event_id,
        "source_book_version": source_book_version,
        "calculation_version": CALC_VERSION,
        "availability_reason": availability_reason,
    }


def _feature_name_universe(family_view: str) -> tuple[str, ...]:
    names = [
        "direction_sign",
        "underlying_market_state",
        "nominal_underlying_clock_state",
        "collection_topology_fingerprint",
        "hyperliquid_feed_cadence_regime",
        "connection_epoch_id",
        "trade_quantity_source",
        "rpi_adjustment_status",
        "depth_stream",
        "auxiliary_degraded",
        "time_since_segment_start_ms",
        "time_to_segment_end_ms",
        "pre_d_bh_bps",
        "pre_d_hb_bps",
        "pre_risk_gap_bps",
        "pre_binance_bid_px",
        "pre_binance_ask_px",
        "pre_binance_mid_px",
        "pre_binance_spread_px",
        "pre_binance_tick_normalized_spread",
        "pre_hyperliquid_bbo_bid_px",
        "pre_hyperliquid_bbo_ask_px",
        "pre_hyperliquid_bbo_mid_px",
        "pre_hyperliquid_bbo_spread_px",
        "pre_binance_bbo_age_ms",
        "pre_hyperliquid_bbo_age_ms",
        "pre_hyperliquid_fast_age_ms",
        "pre_binance_impacted_best_qty",
        "pre_binance_opposite_best_qty",
        "pre_binance_impacted_top5_qty",
        "pre_binance_opposite_top5_qty",
        "pre_binance_top5_imbalance",
        "pre_hyperliquid_impacted_best_qty",
        "pre_hyperliquid_opposite_best_qty",
        "pre_hyperliquid_fast_impacted_top5_qty",
        "pre_hyperliquid_fast_opposite_top5_qty",
        "pre_hyperliquid_fast_top5_imbalance",
        *(
            f"pre_risk_gap_change_{window_ms}ms_bps"
            for window_ms in (10, 25, 50, 100, 250, 500)
        ),
        "pre_binance_gap_formation_contribution_100ms_bps",
        "pre_hyperliquid_gap_formation_contribution_100ms_bps",
        *(
            f"pre_binance_signed_trade_flow_{window_ms}ms"
            for window_ms in (10, 50, 100, 250, 500)
        ),
    ]
    for window_ms in (100, 500):
        names.extend(
            (
                f"pre_binance_ofi_proxy_{window_ms}ms",
                f"pre_binance_bbo_update_intensity_{window_ms}ms",
                f"pre_hyperliquid_bbo_event_intensity_{window_ms}ms",
                f"pre_hyperliquid_trade_event_intensity_{window_ms}ms",
                f"pre_hyperliquid_fast_l2_event_intensity_{window_ms}ms",
                f"pre_same_direction_candidate_count_{window_ms}ms",
                f"pre_opposite_direction_candidate_count_{window_ms}ms",
            )
        )
    for window_ms in (100, 500):
        names.extend(
            (
                f"pre_binance_realized_volatility_{window_ms}ms",
                f"pre_binance_impacted_queue_change_{window_ms}ms",
                f"pre_hyperliquid_midpoint_change_{window_ms}ms_bps",
            )
        )
    names.extend(
        f"recent_continuous_flow_density_{window_ms}ms"
        for window_ms in (100, 500, 1000)
    )
    names.extend(
        (
            "trailing_basis_residual",
            "trailing_basis_robust_zscore",
            "time_since_authoritative_underlying_state_transition",
            "trailing_volatility_bucket",
            "trailing_spread_bucket",
            "trailing_liquidity_bucket",
            "trigger_aggressor_side",
            "trigger_pre_visible_queue_qty",
            "trigger_observed_trade_qty_at_candidate",
            "trigger_observed_trade_qty_to_visible_prequeue_ratio_at_candidate",
            "trigger_candidate_prefix_trade_qty",
            "trigger_candidate_prefix_trade_count",
            "trigger_candidate_prefix_duration_ms",
        )
    )
    if family_view == "family_b":
        names.extend(
            (
                "confirmed_burst_trade_count_through_decision",
                "confirmed_burst_trade_qty_through_decision",
                "confirmed_burst_duration_through_decision_ms",
                "confirmed_touch_trade_qty_through_decision",
                "confirmed_impact_ratio_legacy",
                "confirmed_queue_drop_ratio",
                "confirmed_price_level_depleted",
                "confirmed_removed_qty",
                "confirmed_trade_explained_ratio",
                "confirmed_attribution",
                "confirmation_lag_ms",
                "post_candidate_trade_continuation_count",
                "post_candidate_trade_continuation_qty",
                "local_binance_source_event_count_candidate_to_confirm",
                "early_r_hyperliquid_bbo_event_count_to_confirm",
                "early_r_hyperliquid_trade_event_count_to_confirm",
                "early_r_hyperliquid_fast_l2_event_count_to_confirm",
                "early_r_hyperliquid_midpoint_change_to_confirm_bps",
                "early_r_risk_gap_change_to_confirm_bps",
                "early_r_candidate_plus_25ms_hl_midpoint_change_bps",
                "early_r_candidate_plus_50ms_hl_midpoint_change_bps",
                "early_r_candidate_plus_100ms_hl_midpoint_change_bps",
            )
        )
    if len(names) != len(set(names)):
        raise EpisodeV3Error("feature name universe contains duplicates")
    return tuple(names)


def _quality_censored_feature_rows(
    candidate: Candidate,
    data: SegmentData,
    *,
    family_view: str,
    decision_landmark: str,
    landmark_ts_ns: int,
    topology_fingerprint: str,
) -> Iterator[dict[str, str]]:
    if family_view != "family_a":
        raise EpisodeV3Error("missing pre-HL-BBO quality censor is Family A only")
    contract_source = f"contract:{CONTRACT_VERSION}"
    constant_values: dict[str, Any] = {
        "direction_sign": candidate.direction_sign,
        "underlying_market_state": UNDERLYING_MARKET_STATE,
        "nominal_underlying_clock_state": _nominal_underlying_clock_state(
            landmark_ts_ns
        ),
        "collection_topology_fingerprint": topology_fingerprint,
        "hyperliquid_feed_cadence_regime": "jul30_observed_public_feed",
        "connection_epoch_id": candidate.connection_epoch_id,
        "trade_quantity_source": TRADE_QUANTITY_SOURCE,
        "rpi_adjustment_status": RPI_ADJUSTMENT_STATUS,
        "depth_stream": DEPTH_STREAM,
        "auxiliary_degraded": bool(
            _auxiliary_degraded_ids(data, landmark_ts_ns)
        ),
        "time_since_segment_start_ms": (
            landmark_ts_ns - candidate.segment_start_ts_ns
        )
        / 1_000_000.0,
        "time_to_segment_end_ms": (
            candidate.segment_end_ts_ns - landmark_ts_ns
        )
        / 1_000_000.0,
        "trigger_aggressor_side": candidate.aggressor_side,
        "trigger_pre_visible_queue_qty": candidate.pre_best_qty,
        "trigger_observed_trade_qty_at_candidate": candidate.touch_trade_qty_at_shock,
        "trigger_observed_trade_qty_to_visible_prequeue_ratio_at_candidate": (
            candidate.shock_impact_ratio
        ),
    }
    for feature_name in _feature_name_universe(family_view):
        if feature_name in constant_values:
            yield _feature_row(
                candidate,
                family_view=family_view,
                decision_landmark=decision_landmark,
                landmark_ts_ns=landmark_ts_ns,
                feature_name=feature_name,
                value=constant_values[feature_name],
                observed_at_ns=(
                    candidate.shock_ts_ns
                    if feature_name.startswith("trigger_")
                    or feature_name == "direction_sign"
                    else candidate.segment_start_ts_ns
                ),
                source_event_id=contract_source,
                source_book_version=CONTRACT_VERSION,
                availability_reason="available",
            )
        else:
            yield _feature_row(
                candidate,
                family_view=family_view,
                decision_landmark=decision_landmark,
                landmark_ts_ns=landmark_ts_ns,
                feature_name=feature_name,
                value=None,
                observed_at_ns=None,
                source_event_id="",
                source_book_version="",
                availability_reason=(
                    "quality_censored_missing_prior_hyperliquid_bbo"
                ),
            )


def _feature_rows(
    candidate: Candidate,
    data: SegmentData,
    *,
    family_view: str,
    decision_landmark: str,
    landmark_ts_ns: int,
    topology_fingerprint: str,
) -> Iterator[dict[str, str]]:
    pre_index = data.timeline.asof_index(candidate.pre_state_ts_ns)
    pre_bbo_index = data.hl_bbo.asof_index(candidate.pre_state_ts_ns)
    landmark_index = data.timeline.asof_index(landmark_ts_ns)
    landmark_bbo_index = data.hl_bbo.asof_index(landmark_ts_ns)
    if min(pre_index, landmark_index) < 0:
        raise EpisodeV3Error(f"{candidate.candidate_id}: missing decision state")
    if min(pre_bbo_index, landmark_bbo_index) < 0:
        yield from _quality_censored_feature_rows(
            candidate,
            data,
            family_view=family_view,
            decision_landmark=decision_landmark,
            landmark_ts_ns=landmark_ts_ns,
            topology_fingerprint=topology_fingerprint,
        )
        return
    pre_binance_observed = int(data.timeline.binance_observed_at[pre_index])
    pre_fast_observed = int(data.timeline.hl_fast_observed_at[pre_index])
    pre_bbo_observed = int(data.hl_bbo.ts[pre_bbo_index])
    if max(pre_binance_observed, pre_fast_observed, pre_bbo_observed) >= candidate.shock_ts_ns:
        raise EpisodeV3Error(f"{candidate.candidate_id}: pre-state is not strict")
    landmark_binance_observed = int(
        data.timeline.binance_observed_at[landmark_index]
    )
    landmark_fast_observed = int(data.timeline.hl_fast_observed_at[landmark_index])
    landmark_bbo_observed = int(data.hl_bbo.ts[landmark_bbo_index])
    if max(
        landmark_binance_observed, landmark_fast_observed, landmark_bbo_observed
    ) > landmark_ts_ns:
        raise EpisodeV3Error(f"{candidate.candidate_id}: future landmark state")

    pre_timeline_id = data.timeline.source_event_id(pre_index)
    pre_fast_id = data.timeline.fast_source_event_id(pre_index)
    pre_bbo_id = data.hl_bbo.source_event_id(pre_bbo_index)
    landmark_timeline_id = data.timeline.source_event_id(landmark_index)
    landmark_fast_id = data.timeline.fast_source_event_id(landmark_index)
    landmark_bbo_id = data.hl_bbo.source_event_id(landmark_bbo_index)
    pre_source_id = _derived_source_id(pre_timeline_id, pre_fast_id, pre_bbo_id)
    landmark_source_id = _derived_source_id(
        landmark_timeline_id, landmark_fast_id, landmark_bbo_id
    )
    pre_observed = max(pre_binance_observed, pre_fast_observed, pre_bbo_observed)
    landmark_observed = max(
        landmark_binance_observed, landmark_fast_observed, landmark_bbo_observed
    )
    pre_version = (
        f"{data.timeline.common_seq[pre_index]}|"
        f"{data.timeline.hl_fast_observed_at[pre_index]}|"
        f"{data.hl_bbo.seq[pre_bbo_index]}"
    )
    landmark_version = (
        f"{data.timeline.common_seq[landmark_index]}|"
        f"{data.timeline.hl_fast_observed_at[landmark_index]}|"
        f"{data.hl_bbo.seq[landmark_bbo_index]}"
    )

    def emit(
        feature_name: str,
        value: Any,
        observed_at_ns: int | None = pre_observed,
        source_event_id: str = pre_source_id,
        source_book_version: str = pre_version,
        availability_reason: str = "available",
    ) -> dict[str, str]:
        return _feature_row(
            candidate,
            family_view=family_view,
            decision_landmark=decision_landmark,
            landmark_ts_ns=landmark_ts_ns,
            feature_name=feature_name,
            value=value,
            observed_at_ns=observed_at_ns,
            source_event_id=source_event_id,
            source_book_version=source_book_version,
            availability_reason=availability_reason,
        )

    contract_source = f"contract:{CONTRACT_VERSION}"
    contract_observed = candidate.segment_start_ts_ns
    constant_features = (
        ("direction_sign", candidate.direction_sign, candidate.shock_ts_ns),
        ("underlying_market_state", UNDERLYING_MARKET_STATE, contract_observed),
        (
            "nominal_underlying_clock_state",
            _nominal_underlying_clock_state(landmark_ts_ns),
            contract_observed,
        ),
        ("collection_topology_fingerprint", topology_fingerprint, contract_observed),
        ("hyperliquid_feed_cadence_regime", "jul30_observed_public_feed", contract_observed),
        ("connection_epoch_id", candidate.connection_epoch_id, contract_observed),
        ("trade_quantity_source", TRADE_QUANTITY_SOURCE, contract_observed),
        ("rpi_adjustment_status", RPI_ADJUSTMENT_STATUS, contract_observed),
        ("depth_stream", DEPTH_STREAM, contract_observed),
    )
    for name, value, observed in constant_features:
        yield emit(
            name,
            value,
            int(observed),
            contract_source,
            CONTRACT_VERSION,
        )
    degraded_ids = _auxiliary_degraded_ids(data, landmark_ts_ns)
    yield emit(
        "auxiliary_degraded",
        bool(degraded_ids),
        landmark_ts_ns,
        f"quality-mask:{candidate.segment_id}",
        "|".join(degraded_ids) or "none",
    )
    yield emit(
        "time_since_segment_start_ms",
        (landmark_ts_ns - candidate.segment_start_ts_ns) / 1_000_000.0,
        landmark_ts_ns,
        contract_source,
        CONTRACT_VERSION,
    )
    yield emit(
        "time_to_segment_end_ms",
        (candidate.segment_end_ts_ns - landmark_ts_ns) / 1_000_000.0,
        landmark_ts_ns,
        contract_source,
        CONTRACT_VERSION,
    )

    pre_bin_bid = float(data.timeline.binance_bid[pre_index])
    pre_bin_ask = float(data.timeline.binance_ask[pre_index])
    pre_hl_bid = float(data.hl_bbo.x1[pre_bbo_index])
    pre_hl_ask = float(data.hl_bbo.x3[pre_bbo_index])
    pre_d_bh, pre_d_hb, pre_risk = _spread_metrics(
        pre_bin_bid,
        pre_bin_ask,
        pre_hl_bid,
        pre_hl_ask,
        candidate.direction_sign,
    )
    for name, value in (
        ("pre_d_bh_bps", pre_d_bh),
        ("pre_d_hb_bps", pre_d_hb),
        ("pre_risk_gap_bps", pre_risk),
        ("pre_binance_bid_px", pre_bin_bid),
        ("pre_binance_ask_px", pre_bin_ask),
        ("pre_binance_mid_px", _mid(pre_bin_bid, pre_bin_ask)),
        ("pre_binance_spread_px", pre_bin_ask - pre_bin_bid),
        (
            "pre_binance_tick_normalized_spread",
            (pre_bin_ask - pre_bin_bid) / 0.01,
        ),
        ("pre_hyperliquid_bbo_bid_px", pre_hl_bid),
        ("pre_hyperliquid_bbo_ask_px", pre_hl_ask),
        ("pre_hyperliquid_bbo_mid_px", _mid(pre_hl_bid, pre_hl_ask)),
        ("pre_hyperliquid_bbo_spread_px", pre_hl_ask - pre_hl_bid),
        (
            "pre_binance_bbo_age_ms",
            (candidate.shock_ts_ns - pre_binance_observed) / 1_000_000.0,
        ),
        (
            "pre_hyperliquid_bbo_age_ms",
            (candidate.shock_ts_ns - pre_bbo_observed) / 1_000_000.0,
        ),
        (
            "pre_hyperliquid_fast_age_ms",
            (candidate.shock_ts_ns - pre_fast_observed) / 1_000_000.0,
        ),
    ):
        yield emit(name, value)

    sign = candidate.direction_sign
    bin_impacted_best = (
        data.timeline.binance_ask_qty[pre_index]
        if sign == 1
        else data.timeline.binance_bid_qty[pre_index]
    )
    bin_opposite_best = (
        data.timeline.binance_bid_qty[pre_index]
        if sign == 1
        else data.timeline.binance_ask_qty[pre_index]
    )
    bin_impacted_top5 = (
        data.timeline.binance_ask_top5[pre_index]
        if sign == 1
        else data.timeline.binance_bid_top5[pre_index]
    )
    bin_opposite_top5 = (
        data.timeline.binance_bid_top5[pre_index]
        if sign == 1
        else data.timeline.binance_ask_top5[pre_index]
    )
    hl_impacted_best = (
        data.hl_bbo.x4[pre_bbo_index] if sign == 1 else data.hl_bbo.x2[pre_bbo_index]
    )
    hl_opposite_best = (
        data.hl_bbo.x2[pre_bbo_index] if sign == 1 else data.hl_bbo.x4[pre_bbo_index]
    )
    hl_fast_impacted_top5 = (
        data.timeline.hl_fast_ask_top5[pre_index]
        if sign == 1
        else data.timeline.hl_fast_bid_top5[pre_index]
    )
    hl_fast_opposite_top5 = (
        data.timeline.hl_fast_bid_top5[pre_index]
        if sign == 1
        else data.timeline.hl_fast_ask_top5[pre_index]
    )
    for name, value in (
        ("pre_binance_impacted_best_qty", float(bin_impacted_best)),
        ("pre_binance_opposite_best_qty", float(bin_opposite_best)),
        ("pre_binance_impacted_top5_qty", float(bin_impacted_top5)),
        ("pre_binance_opposite_top5_qty", float(bin_opposite_top5)),
        (
            "pre_binance_top5_imbalance",
            (float(bin_impacted_top5) - float(bin_opposite_top5))
            / max(float(bin_impacted_top5) + float(bin_opposite_top5), 1e-12),
        ),
        ("pre_hyperliquid_impacted_best_qty", float(hl_impacted_best)),
        ("pre_hyperliquid_opposite_best_qty", float(hl_opposite_best)),
        ("pre_hyperliquid_fast_impacted_top5_qty", float(hl_fast_impacted_top5)),
        ("pre_hyperliquid_fast_opposite_top5_qty", float(hl_fast_opposite_top5)),
        (
            "pre_hyperliquid_fast_top5_imbalance",
            (float(hl_fast_impacted_top5) - float(hl_fast_opposite_top5))
            / max(
                float(hl_fast_impacted_top5) + float(hl_fast_opposite_top5),
                1e-12,
            ),
        ),
    ):
        yield emit(name, value)

    for window_ms in (10, 25, 50, 100, 250, 500):
        prior_ts = candidate.pre_state_ts_ns - window_ms * 1_000_000
        prior_timeline = data.timeline.asof_index(prior_ts)
        prior_bbo = data.hl_bbo.asof_index(prior_ts)
        if min(prior_timeline, prior_bbo) < 0:
            yield emit(
                f"pre_risk_gap_change_{window_ms}ms_bps",
                None,
                None,
                "",
                "",
                "insufficient_pre_segment_history",
            )
            continue
        _, _, prior_risk = _spread_metrics(
            float(data.timeline.binance_bid[prior_timeline]),
            float(data.timeline.binance_ask[prior_timeline]),
            float(data.hl_bbo.x1[prior_bbo]),
            float(data.hl_bbo.x3[prior_bbo]),
            sign,
        )
        yield emit(
            f"pre_risk_gap_change_{window_ms}ms_bps",
            pre_risk - prior_risk,
            pre_observed,
            _derived_source_id(
                pre_source_id,
                data.timeline.source_event_id(prior_timeline),
                data.hl_bbo.source_event_id(prior_bbo),
            ),
            f"{pre_version}|{data.timeline.common_seq[prior_timeline]}|{data.hl_bbo.seq[prior_bbo]}",
        )
    prior_100_timeline = data.timeline.asof_index(
        candidate.pre_state_ts_ns - 100_000_000
    )
    prior_100_bbo = data.hl_bbo.asof_index(candidate.pre_state_ts_ns - 100_000_000)
    if min(prior_100_timeline, prior_100_bbo) >= 0:
        reference = (
            _mid(pre_bin_bid, pre_bin_ask) + _mid(pre_hl_bid, pre_hl_ask)
        ) / 2.0
        if sign == 1:
            source_contribution = (
                pre_bin_bid - float(data.timeline.binance_bid[prior_100_timeline])
            ) / reference * 10_000.0
            target_contribution = (
                float(data.hl_bbo.x3[prior_100_bbo]) - pre_hl_ask
            ) / reference * 10_000.0
        else:
            source_contribution = (
                float(data.timeline.binance_ask[prior_100_timeline]) - pre_bin_ask
            ) / reference * 10_000.0
            target_contribution = (
                pre_hl_bid - float(data.hl_bbo.x1[prior_100_bbo])
            ) / reference * 10_000.0
        yield emit("pre_binance_gap_formation_contribution_100ms_bps", source_contribution)
        yield emit("pre_hyperliquid_gap_formation_contribution_100ms_bps", target_contribution)
    else:
        yield emit(
            "pre_binance_gap_formation_contribution_100ms_bps",
            None,
            None,
            "",
            "",
            "insufficient_pre_segment_history",
        )
        yield emit(
            "pre_hyperliquid_gap_formation_contribution_100ms_bps",
            None,
            None,
            "",
            "",
            "insufficient_pre_segment_history",
        )

    for window_ms in (10, 50, 100, 250, 500):
        start_ns = candidate.shock_ts_ns - window_ms * 1_000_000
        signed_flow = _prefix_value(
            data.binance_trades.ts,
            data.binance_signed_qty_prefix,
            start_ns,
            candidate.shock_ts_ns,
        )
        yield emit(
            f"pre_binance_signed_trade_flow_{window_ms}ms",
            signed_flow,
            candidate.shock_ts_ns,
            f"range:{candidate.segment_id}:binance_trade:{start_ns}:{candidate.shock_ts_ns}",
            f"{start_ns}:{candidate.shock_ts_ns}",
        )
    for window_ms in (100, 500):
        start_ns = candidate.shock_ts_ns - window_ms * 1_000_000
        ofi = _prefix_value(
            data.binance_bbo.ts,
            data.binance_ofi_prefix,
            start_ns,
            candidate.shock_ts_ns,
        )
        yield emit(
            f"pre_binance_ofi_proxy_{window_ms}ms",
            ofi,
            candidate.shock_ts_ns,
            f"range:{candidate.segment_id}:binance_book_ticker:{start_ns}:{candidate.shock_ts_ns}",
            f"{start_ns}:{candidate.shock_ts_ns}",
        )
        yield emit(
            f"pre_binance_bbo_update_intensity_{window_ms}ms",
            _count_window(data.binance_bbo.ts, start_ns, candidate.shock_ts_ns),
            candidate.shock_ts_ns,
            f"range:{candidate.segment_id}:binance_book_ticker:{start_ns}:{candidate.shock_ts_ns}",
            f"{start_ns}:{candidate.shock_ts_ns}",
        )
        yield emit(
            f"pre_hyperliquid_bbo_event_intensity_{window_ms}ms",
            _count_window(data.hl_bbo.ts, start_ns, candidate.shock_ts_ns),
            candidate.shock_ts_ns,
            f"range:{candidate.segment_id}:hyperliquid_bbo:{start_ns}:{candidate.shock_ts_ns}",
            f"{start_ns}:{candidate.shock_ts_ns}",
        )
        yield emit(
            f"pre_hyperliquid_trade_event_intensity_{window_ms}ms",
            _count_window(data.hl_trades.ts, start_ns, candidate.shock_ts_ns),
            candidate.shock_ts_ns,
            f"range:{candidate.segment_id}:hyperliquid_trade:{start_ns}:{candidate.shock_ts_ns}",
            f"{start_ns}:{candidate.shock_ts_ns}",
        )
        yield emit(
            f"pre_hyperliquid_fast_l2_event_intensity_{window_ms}ms",
            _count_window(
                data.timeline.fast_event_ts, start_ns, candidate.shock_ts_ns
            ),
            candidate.shock_ts_ns,
            f"range:{candidate.segment_id}:hyperliquid_fast_l2:{start_ns}:{candidate.shock_ts_ns}",
            f"{start_ns}:{candidate.shock_ts_ns}",
        )
        same, opposite = _candidate_direction_counts(data, candidate, window_ms)
        yield emit(
            f"pre_same_direction_candidate_count_{window_ms}ms",
            same,
            candidate.shock_ts_ns,
            f"range:{candidate.segment_id}:candidate_atoms:{start_ns}:{candidate.shock_ts_ns}",
            f"{start_ns}:{candidate.shock_ts_ns}",
        )
        yield emit(
            f"pre_opposite_direction_candidate_count_{window_ms}ms",
            opposite,
            candidate.shock_ts_ns,
            f"range:{candidate.segment_id}:candidate_atoms:{start_ns}:{candidate.shock_ts_ns}",
            f"{start_ns}:{candidate.shock_ts_ns}",
        )
    for window_ms in (100, 500):
        start_ns = candidate.pre_state_ts_ns - window_ms * 1_000_000
        left = bisect.bisect_left(data.timeline.ts, start_ns)
        right = bisect.bisect_right(data.timeline.ts, candidate.pre_state_ts_ns)
        if right - left < 2:
            yield emit(
                f"pre_binance_realized_volatility_{window_ms}ms",
                None,
                None,
                "",
                "",
                "insufficient_timeline_updates",
            )
        else:
            variance = float(
                data.timeline.midpoint_sq_return_prefix[right]
                - data.timeline.midpoint_sq_return_prefix[left]
            )
            yield emit(
                f"pre_binance_realized_volatility_{window_ms}ms",
                math.sqrt(max(variance, 0.0)),
            )
        prior_index = data.timeline.asof_index(start_ns)
        if prior_index < 0:
            yield emit(
                f"pre_binance_impacted_queue_change_{window_ms}ms",
                None,
                None,
                "",
                "",
                "insufficient_pre_segment_history",
            )
            yield emit(
                f"pre_hyperliquid_midpoint_change_{window_ms}ms_bps",
                None,
                None,
                "",
                "",
                "insufficient_pre_segment_history",
            )
        else:
            prior_bin_impacted = (
                data.timeline.binance_ask_qty[prior_index]
                if sign == 1
                else data.timeline.binance_bid_qty[prior_index]
            )
            yield emit(
                f"pre_binance_impacted_queue_change_{window_ms}ms",
                float(bin_impacted_best) - float(prior_bin_impacted),
            )
            prior_fast_mid = _mid(
                float(data.timeline.hl_fast_bid[prior_index]),
                float(data.timeline.hl_fast_ask[prior_index]),
            )
            current_fast_mid = _mid(
                float(data.timeline.hl_fast_bid[pre_index]),
                float(data.timeline.hl_fast_ask[pre_index]),
            )
            yield emit(
                f"pre_hyperliquid_midpoint_change_{window_ms}ms_bps",
                sign
                * (current_fast_mid - prior_fast_mid)
                / max(prior_fast_mid, 1e-12)
                * 10_000.0,
            )
    for window_ms in (100, 500, 1000):
        start_ns = candidate.shock_ts_ns - window_ms * 1_000_000
        yield emit(
            f"recent_continuous_flow_density_{window_ms}ms",
            _count_window(data.candidate_ts, start_ns, candidate.shock_ts_ns),
            candidate.shock_ts_ns,
            f"range:{candidate.segment_id}:candidate_atoms:{start_ns}:{candidate.shock_ts_ns}",
            f"{start_ns}:{candidate.shock_ts_ns}",
        )

    unavailable = (
        ("trailing_basis_residual", "basis_state_not_admitted_in_stage4"),
        ("trailing_basis_robust_zscore", "basis_state_not_admitted_in_stage4"),
        (
            "time_since_authoritative_underlying_state_transition",
            "authoritative_holiday_calendar_unavailable",
        ),
        ("trailing_volatility_bucket", "bucket_freeze_deferred_to_stage6"),
        ("trailing_spread_bucket", "bucket_freeze_deferred_to_stage6"),
        ("trailing_liquidity_bucket", "bucket_freeze_deferred_to_stage6"),
    )
    for name, reason in unavailable:
        yield emit(name, None, None, "", "", reason)

    candidate_trade_index = data.binance_trades.asof_index(candidate.shock_ts_ns)
    if candidate_trade_index < 0:
        raise EpisodeV3Error(f"{candidate.candidate_id}: candidate trade missing")
    candidate_trade_id = data.binance_trades.source_event_id(candidate_trade_index)
    candidate_trade_observed = int(data.binance_trades.ts[candidate_trade_index])
    burst_left = bisect.bisect_left(
        data.binance_trades.ts, candidate.burst_start_ts_ns
    )
    burst_right = bisect.bisect_right(
        data.binance_trades.ts, candidate.shock_ts_ns
    )
    candidate_prefix_qty = sum(
        float(data.binance_trades.x2[index])
        for index in range(burst_left, burst_right)
        if int(data.binance_trades.side[index]) == sign
    )
    for name, value in (
        ("trigger_aggressor_side", candidate.aggressor_side),
        ("trigger_pre_visible_queue_qty", candidate.pre_best_qty),
        ("trigger_observed_trade_qty_at_candidate", candidate.touch_trade_qty_at_shock),
        (
            "trigger_observed_trade_qty_to_visible_prequeue_ratio_at_candidate",
            candidate.shock_impact_ratio,
        ),
        ("trigger_candidate_prefix_trade_qty", candidate_prefix_qty),
        ("trigger_candidate_prefix_trade_count", burst_right - burst_left),
        (
            "trigger_candidate_prefix_duration_ms",
            (candidate.shock_ts_ns - candidate.burst_start_ts_ns) / 1_000_000.0,
        ),
    ):
        yield emit(
            name,
            value,
            candidate_trade_observed,
            candidate_trade_id,
            str(data.binance_trades.seq[candidate_trade_index]),
        )

    if family_view == "family_b":
        if candidate.t_confirm_ns is None:
            raise EpisodeV3Error(f"{candidate.candidate_id}: Family B lacks confirm")
        confirm_ts = candidate.t_confirm_ns
        yield from _confirmation_feature_rows(candidate, data)
        landmark_bin_bid = float(data.timeline.binance_bid[landmark_index])
        landmark_bin_ask = float(data.timeline.binance_ask[landmark_index])
        landmark_hl_bid = float(data.hl_bbo.x1[landmark_bbo_index])
        landmark_hl_ask = float(data.hl_bbo.x3[landmark_bbo_index])
        _, _, landmark_risk = _spread_metrics(
            landmark_bin_bid,
            landmark_bin_ask,
            landmark_hl_bid,
            landmark_hl_ask,
            sign,
        )
        candidate_bbo_index = data.hl_bbo.asof_index(candidate.shock_ts_ns)
        candidate_timeline_index = data.timeline.asof_index(candidate.shock_ts_ns)
        _, _, candidate_risk = _spread_metrics(
            float(data.timeline.binance_bid[candidate_timeline_index]),
            float(data.timeline.binance_ask[candidate_timeline_index]),
            float(data.hl_bbo.x1[candidate_bbo_index]),
            float(data.hl_bbo.x3[candidate_bbo_index]),
            sign,
        )
        candidate_hl_mid = _mid(
            float(data.hl_bbo.x1[candidate_bbo_index]),
            float(data.hl_bbo.x3[candidate_bbo_index]),
        )
        landmark_hl_mid = _mid(landmark_hl_bid, landmark_hl_ask)
        for name, value in (
            (
                "early_r_hyperliquid_bbo_event_count_to_confirm",
                _count_window(data.hl_bbo.ts, candidate.shock_ts_ns, confirm_ts),
            ),
            (
                "early_r_hyperliquid_trade_event_count_to_confirm",
                _count_window(data.hl_trades.ts, candidate.shock_ts_ns, confirm_ts),
            ),
            (
                "early_r_hyperliquid_fast_l2_event_count_to_confirm",
                _count_window(
                    data.timeline.fast_event_ts, candidate.shock_ts_ns, confirm_ts
                ),
            ),
            (
                "early_r_hyperliquid_midpoint_change_to_confirm_bps",
                sign
                * (landmark_hl_mid - candidate_hl_mid)
                / max(candidate_hl_mid, 1e-12)
                * 10_000.0,
            ),
            (
                "early_r_risk_gap_change_to_confirm_bps",
                landmark_risk - candidate_risk,
            ),
        ):
            yield emit(
                name,
                value,
                landmark_observed,
                landmark_source_id,
                landmark_version,
            )
        for horizon_ms in (25, 50, 100):
            endpoint = candidate.shock_ts_ns + horizon_ms * 1_000_000
            if endpoint > confirm_ts:
                yield emit(
                    f"early_r_candidate_plus_{horizon_ms}ms_hl_midpoint_change_bps",
                    None,
                    None,
                    "",
                    "",
                    "prefix_endpoint_after_confirm_forbidden",
                )
                continue
            endpoint_index = data.hl_bbo.asof_index(endpoint)
            if endpoint_index < 0:
                yield emit(
                    f"early_r_candidate_plus_{horizon_ms}ms_hl_midpoint_change_bps",
                    None,
                    None,
                    "",
                    "",
                    "no_asof_target_bbo",
                )
                continue
            endpoint_mid = _mid(
                float(data.hl_bbo.x1[endpoint_index]),
                float(data.hl_bbo.x3[endpoint_index]),
            )
            yield emit(
                f"early_r_candidate_plus_{horizon_ms}ms_hl_midpoint_change_bps",
                sign
                * (endpoint_mid - candidate_hl_mid)
                / max(candidate_hl_mid, 1e-12)
                * 10_000.0,
                int(data.hl_bbo.ts[endpoint_index]),
                data.hl_bbo.source_event_id(endpoint_index),
                str(data.hl_bbo.seq[endpoint_index]),
            )


def _first_event_interval(
    store: NumericEventStore,
    *,
    start_ns: int,
    end_ns: int,
    predicate: Any,
    full_horizon: bool,
    segment_end_ns: int,
) -> dict[str, str]:
    last_non_event = start_ns
    index = bisect.bisect_right(store.ts, start_ns)
    while index < len(store.ts) and int(store.ts[index]) <= end_ns:
        observation_ts = int(store.ts[index])
        group_end = bisect.bisect_right(store.ts, observation_ts, lo=index)
        event_index = next(
            (
                candidate_index
                for candidate_index in range(index, group_end)
                if predicate(candidate_index)
            ),
            None,
        )
        if event_index is not None:
            return {
                "status": "interval_censored",
                "interval_lower_ns": str(last_non_event),
                "interval_upper_ns": str(observation_ts),
                "censor_time_ns": "",
                "censor_reason": "",
                "source_event_id": store.source_event_id(event_index),
            }
        last_non_event = observation_ts
        index = group_end
    return {
        "status": "right_censored" if full_horizon else "segment_censored",
        "interval_lower_ns": str(last_non_event),
        "interval_upper_ns": "",
        "censor_time_ns": str(end_ns),
        "censor_reason": ""
        if full_horizon
        else (
            "segment_boundary"
            if end_ns == segment_end_ns
            else "connection_epoch_boundary"
        ),
        "source_event_id": "",
    }


def _outcome_row(
    candidate: Candidate, data: SegmentData
) -> tuple[dict[str, str], Counter[str]]:
    horizon_target = candidate.shock_ts_ns + OUTCOME_HORIZON_NS
    horizon_end = min(horizon_target, candidate.segment_end_ts_ns)
    full_horizon = horizon_end == horizon_target
    horizon_status = "complete" if full_horizon else "segment_censored"
    horizon_reason = "" if full_horizon else "segment_boundary"
    baseline_bbo_index = data.hl_bbo.strict_pre_index(candidate.shock_ts_ns)
    baseline_timeline_index = data.timeline.strict_pre_index(
        candidate.shock_ts_ns
    )
    if baseline_timeline_index < 0:
        raise EpisodeV3Error(f"{candidate.candidate_id}: outcome baseline missing")
    if baseline_bbo_index < 0:
        row = {
            "candidate_id": candidate.candidate_id,
            "episode_id": candidate.episode_id,
            "t_candidate_ns": str(candidate.shock_ts_ns),
            "t_confirm_ns": _int_text(candidate.t_confirm_ns),
            "outcome_horizon_end_ns": str(candidate.shock_ts_ns),
            "outcome_horizon_status": "quality_censored",
            "outcome_censor_time_ns": str(candidate.shock_ts_ns),
            "outcome_censor_reason": "missing_prior_hyperliquid_bbo",
        }
        for name in FIRST_EVENT_NAMES:
            row[f"{name}_status"] = "quality_censored"
            row[f"{name}_interval_lower_ns"] = ""
            row[f"{name}_interval_upper_ns"] = ""
            row[f"{name}_censor_time_ns"] = str(candidate.shock_ts_ns)
            row[f"{name}_censor_reason"] = "missing_prior_hyperliquid_bbo"
            row[f"{name}_source_event_id"] = ""
        for horizon in OUTCOME_HORIZONS_MS:
            row[f"gap_survival_{horizon}ms"] = ""
            row[f"gap_survival_{horizon}ms_availability"] = "quality_censored"
            row[f"risk_gap_{horizon}ms_bps"] = ""
        for horizon in MARKOUT_HORIZONS_MS:
            row[f"target_midpoint_markout_{horizon}ms_bps"] = ""
            row[
                f"target_midpoint_markout_{horizon}ms_availability"
            ] = "quality_censored"
        row["maximum_adverse_excursion_0_2000ms_bps"] = ""
        row["maximum_favorable_excursion_0_2000ms_bps"] = ""
        row["excursion_availability"] = "quality_censored"
        row["source_leg_gap_closure_contribution_bps"] = ""
        row["target_leg_gap_closure_contribution_bps"] = ""
        row["gap_closure_contribution_availability"] = "quality_censored"
        row["adverse_event_before_confirmed"] = ""
        row["adverse_event_before_confirmed_status"] = "quality_censored"
        row["public_trade_reaches_quote"] = ""
        row["public_bbo_moves_through_quote"] = ""
        row["public_quote_survives_horizon"] = ""
        row["public_adverse_exposure"] = ""
        row["public_quote_risk_availability"] = "quality_censored"
        return row, Counter(
            {
                "quality_censored": len(FIRST_EVENT_NAMES),
                "point_coerced": 0,
                "epoch_censored": 0,
            }
        )
    baseline_hl_bid = float(data.hl_bbo.x1[baseline_bbo_index])
    baseline_hl_ask = float(data.hl_bbo.x3[baseline_bbo_index])
    baseline_hl_mid = _mid(baseline_hl_bid, baseline_hl_ask)
    baseline_bin_bid = float(data.timeline.binance_bid[baseline_timeline_index])
    baseline_bin_ask = float(data.timeline.binance_ask[baseline_timeline_index])
    sign = candidate.direction_sign

    adverse = _first_event_interval(
        data.hl_bbo,
        start_ns=candidate.shock_ts_ns,
        end_ns=horizon_end,
        predicate=lambda index: sign
        * (
            _mid(float(data.hl_bbo.x1[index]), float(data.hl_bbo.x3[index]))
            - baseline_hl_mid
        )
        > 0,
        full_horizon=full_horizon,
        segment_end_ns=candidate.segment_end_ts_ns,
    )
    trade_reaches = _first_event_interval(
        data.hl_trades,
        start_ns=candidate.shock_ts_ns,
        end_ns=horizon_end,
        predicate=lambda index: (
            int(data.hl_trades.side[index]) == sign
            and (
                float(data.hl_trades.x1[index]) >= baseline_hl_ask
                if sign == 1
                else float(data.hl_trades.x1[index]) <= baseline_hl_bid
            )
        ),
        full_horizon=full_horizon,
        segment_end_ns=candidate.segment_end_ts_ns,
    )
    retreat = _first_event_interval(
        data.hl_bbo,
        start_ns=candidate.shock_ts_ns,
        end_ns=horizon_end,
        predicate=lambda index: (
            float(data.hl_bbo.x3[index]) > baseline_hl_ask
            if sign == 1
            else float(data.hl_bbo.x1[index]) < baseline_hl_bid
        ),
        full_horizon=full_horizon,
        segment_end_ns=candidate.segment_end_ts_ns,
    )
    bbo_through = _first_event_interval(
        data.hl_bbo,
        start_ns=candidate.shock_ts_ns,
        end_ns=horizon_end,
        predicate=lambda index: (
            float(data.hl_bbo.x1[index]) >= baseline_hl_ask
            if sign == 1
            else float(data.hl_bbo.x3[index]) <= baseline_hl_bid
        ),
        full_horizon=full_horizon,
        segment_end_ns=candidate.segment_end_ts_ns,
    )
    row: dict[str, str] = {
        "candidate_id": candidate.candidate_id,
        "episode_id": candidate.episode_id,
        "t_candidate_ns": str(candidate.shock_ts_ns),
        "t_confirm_ns": _int_text(candidate.t_confirm_ns),
        "outcome_horizon_end_ns": str(horizon_end),
        "outcome_horizon_status": horizon_status,
        "outcome_censor_time_ns": "" if full_horizon else str(horizon_end),
        "outcome_censor_reason": horizon_reason,
    }
    for name, result in zip(
        FIRST_EVENT_NAMES, (adverse, trade_reaches, retreat), strict=True
    ):
        for suffix in (
            "status",
            "interval_lower_ns",
            "interval_upper_ns",
            "censor_time_ns",
            "censor_reason",
            "source_event_id",
        ):
            row[f"{name}_{suffix}"] = result[suffix]

    for horizon_ms in OUTCOME_HORIZONS_MS:
        target_ts = candidate.shock_ts_ns + horizon_ms * 1_000_000
        prefix = f"gap_survival_{horizon_ms}ms"
        if target_ts > candidate.segment_end_ts_ns:
            row[prefix] = ""
            row[prefix + "_availability"] = "segment_censored"
            row[f"risk_gap_{horizon_ms}ms_bps"] = ""
            continue
        timeline_index = data.timeline.asof_index(target_ts)
        bbo_index = data.hl_bbo.asof_index(target_ts)
        if min(timeline_index, bbo_index) < 0:
            row[prefix] = ""
            row[prefix + "_availability"] = "no_asof_state"
            row[f"risk_gap_{horizon_ms}ms_bps"] = ""
            continue
        _, _, risk_gap = _spread_metrics(
            float(data.timeline.binance_bid[timeline_index]),
            float(data.timeline.binance_ask[timeline_index]),
            float(data.hl_bbo.x1[bbo_index]),
            float(data.hl_bbo.x3[bbo_index]),
            sign,
        )
        row[prefix] = _bool_text(risk_gap > 0)
        row[prefix + "_availability"] = "available"
        row[f"risk_gap_{horizon_ms}ms_bps"] = _float_text(risk_gap)

    for horizon_ms in MARKOUT_HORIZONS_MS:
        target_ts = candidate.shock_ts_ns + horizon_ms * 1_000_000
        value_name = f"target_midpoint_markout_{horizon_ms}ms_bps"
        availability_name = (
            f"target_midpoint_markout_{horizon_ms}ms_availability"
        )
        if target_ts > candidate.segment_end_ts_ns:
            row[value_name] = ""
            row[availability_name] = "segment_censored"
            continue
        bbo_index = data.hl_bbo.asof_index(target_ts)
        if bbo_index < 0:
            row[value_name] = ""
            row[availability_name] = "no_asof_target_bbo"
            continue
        target_mid = _mid(
            float(data.hl_bbo.x1[bbo_index]), float(data.hl_bbo.x3[bbo_index])
        )
        row[value_name] = _float_text(
            sign
            * (target_mid - baseline_hl_mid)
            / max(baseline_hl_mid, 1e-12)
            * 10_000.0
        )
        row[availability_name] = "available"

    start_index = bisect.bisect_right(data.hl_bbo.ts, candidate.shock_ts_ns)
    end_index = bisect.bisect_right(data.hl_bbo.ts, horizon_end)
    movements = [0.0]
    movements.extend(
        sign
        * (
            _mid(float(data.hl_bbo.x1[index]), float(data.hl_bbo.x3[index]))
            - baseline_hl_mid
        )
        / max(baseline_hl_mid, 1e-12)
        * 10_000.0
        for index in range(start_index, end_index)
    )
    row["maximum_adverse_excursion_0_2000ms_bps"] = _float_text(max(movements))
    row["maximum_favorable_excursion_0_2000ms_bps"] = _float_text(
        max(-value for value in movements)
    )
    row["excursion_availability"] = (
        "available" if full_horizon else "segment_censored_partial_path"
    )

    if full_horizon:
        target_timeline_index = data.timeline.asof_index(horizon_target)
        target_bbo_index = data.hl_bbo.asof_index(horizon_target)
        if min(target_timeline_index, target_bbo_index) < 0:
            row["source_leg_gap_closure_contribution_bps"] = ""
            row["target_leg_gap_closure_contribution_bps"] = ""
            row["gap_closure_contribution_availability"] = "no_asof_state"
        else:
            target_bin_bid = float(data.timeline.binance_bid[target_timeline_index])
            target_bin_ask = float(data.timeline.binance_ask[target_timeline_index])
            target_hl_bid = float(data.hl_bbo.x1[target_bbo_index])
            target_hl_ask = float(data.hl_bbo.x3[target_bbo_index])
            reference = (
                _mid(baseline_bin_bid, baseline_bin_ask) + baseline_hl_mid
            ) / 2.0
            if sign == 1:
                source_contribution = (
                    baseline_bin_bid - target_bin_bid
                ) / reference * 10_000.0
                target_contribution = (
                    target_hl_ask - baseline_hl_ask
                ) / reference * 10_000.0
            else:
                source_contribution = (
                    target_bin_ask - baseline_bin_ask
                ) / reference * 10_000.0
                target_contribution = (
                    baseline_hl_bid - target_hl_bid
                ) / reference * 10_000.0
            row["source_leg_gap_closure_contribution_bps"] = _float_text(
                source_contribution
            )
            row["target_leg_gap_closure_contribution_bps"] = _float_text(
                target_contribution
            )
            row["gap_closure_contribution_availability"] = "available"
    else:
        row["source_leg_gap_closure_contribution_bps"] = ""
        row["target_leg_gap_closure_contribution_bps"] = ""
        row["gap_closure_contribution_availability"] = "segment_censored"

    if candidate.t_confirm_ns is None:
        row["adverse_event_before_confirmed"] = ""
        row["adverse_event_before_confirmed_status"] = "no_confirm_landmark"
    elif adverse["status"] == "interval_censored":
        lower = int(adverse["interval_lower_ns"])
        upper = int(adverse["interval_upper_ns"])
        if upper <= candidate.t_confirm_ns:
            row["adverse_event_before_confirmed"] = "true"
            row["adverse_event_before_confirmed_status"] = "identified_true"
        elif lower >= candidate.t_confirm_ns:
            row["adverse_event_before_confirmed"] = "false"
            row["adverse_event_before_confirmed_status"] = "identified_false"
        else:
            row["adverse_event_before_confirmed"] = ""
            row["adverse_event_before_confirmed_status"] = (
                "ambiguous_interval_straddles_confirm"
            )
    elif horizon_end >= candidate.t_confirm_ns:
        row["adverse_event_before_confirmed"] = "false"
        row["adverse_event_before_confirmed_status"] = (
            "identified_not_observed_before_confirm"
        )
    else:
        row["adverse_event_before_confirmed"] = ""
        row["adverse_event_before_confirmed_status"] = (
            "censored_before_confirm"
        )

    trade_observed = trade_reaches["status"] == "interval_censored"
    through_observed = bbo_through["status"] == "interval_censored"
    if full_horizon:
        row["public_trade_reaches_quote"] = _bool_text(trade_observed)
        row["public_bbo_moves_through_quote"] = _bool_text(through_observed)
        row["public_quote_survives_horizon"] = _bool_text(
            not trade_observed and not through_observed
        )
        row["public_adverse_exposure"] = _bool_text(max(movements) > 0)
        row["public_quote_risk_availability"] = "available"
    else:
        row["public_trade_reaches_quote"] = ""
        row["public_bbo_moves_through_quote"] = ""
        row["public_quote_survives_horizon"] = ""
        row["public_adverse_exposure"] = ""
        row["public_quote_risk_availability"] = "segment_censored"

    counts: Counter[str] = Counter()
    for result in (adverse, trade_reaches, retreat):
        counts[result["status"]] += 1
    counts["point_coerced"] = 0
    counts["quality_censored"] = 0
    counts["epoch_censored"] = 0
    return row, counts


def _anchor_row(
    candidate: Candidate,
    *,
    r0_manifest_sha256: str,
    auxiliary_ids: Sequence[str],
) -> dict[str, str]:
    if candidate.attribution not in CLASSIFICATION_ALLOWED_VALUES:
        raise EpisodeV3Error(
            f"{candidate.candidate_id}: classification semantic mapping drift"
        )
    if candidate.burst_start_ts_ns > candidate.shock_ts_ns:
        raise EpisodeV3Error(f"{candidate.candidate_id}: burst/candidate ordering")
    if candidate.t_confirm_ns is not None and candidate.t_confirm_ns <= candidate.shock_ts_ns:
        raise EpisodeV3Error(f"{candidate.candidate_id}: confirm ordering")
    if candidate.pre_hl_bbo_ts_ns is None:
        censor_time = candidate.shock_ts_ns
        censor_reason = "quality_missing_prior_hyperliquid_bbo"
    else:
        censor_time = min(
            candidate.shock_ts_ns + OUTCOME_HORIZON_NS,
            candidate.segment_end_ts_ns,
        )
        censor_reason = (
            ""
            if censor_time == candidate.shock_ts_ns + OUTCOME_HORIZON_NS
            else "segment_boundary"
        )
    quality_flags = {
        "auxiliary_degraded_at_candidate": bool(auxiliary_ids),
        "auxiliary_degraded_interval_ids": list(auxiliary_ids),
        "core_l2_degraded": False,
        "depth_stream": DEPTH_STREAM,
        "rpi_adjustment_status": RPI_ADJUSTMENT_STATUS,
        "trade_quantity_source": TRADE_QUANTITY_SOURCE,
        "underlying_market_state": UNDERLYING_MARKET_STATE,
        "missing_prior_hyperliquid_bbo": candidate.pre_hl_bbo_ts_ns is None,
    }
    return {
        "campaign_id": CAMPAIGN_ID,
        "session_id": SESSION_ID,
        "segment_id": candidate.segment_id,
        "episode_id": candidate.episode_id,
        "candidate_id": candidate.candidate_id,
        "shock_cluster_id": candidate.cluster_id,
        "flow_cluster_id": candidate.continuous_flow_episode_id,
        "overlap_block_id": candidate.overlap_block_id,
        "candidate_seq": str(candidate.candidate_seq),
        "direction_sign": str(candidate.direction_sign),
        "aggressor_side": candidate.aggressor_side,
        "t_burst_start_ns": str(candidate.burst_start_ts_ns),
        "t_candidate_ns": str(candidate.shock_ts_ns),
        "t_confirm_ns": _int_text(candidate.t_confirm_ns),
        "detector_decision_ts_ns": _int_text(candidate.detector_decision_ts_ns),
        "confirmation_lag_ns": (
            ""
            if candidate.t_confirm_ns is None
            else str(candidate.t_confirm_ns - candidate.shock_ts_ns)
        ),
        "family_a_available": "true",
        "family_b_available": _bool_text(candidate.family_b_available),
        "confirmed": _bool_text(candidate.primary_episode),
        "classification": candidate.attribution,
        "rejection_reason": candidate.rejection_reason,
        "censor_time_ns": str(censor_time),
        "censor_reason": censor_reason,
        "quality_flags_json": json.dumps(
            quality_flags, sort_keys=True, separators=(",", ":")
        ),
        "connection_epoch_id": candidate.connection_epoch_id,
        "segment_start_ts_ns": str(candidate.segment_start_ts_ns),
        "segment_end_ts_ns": str(candidate.segment_end_ts_ns),
        "evidence_label": EVIDENCE_LABELS[candidate.segment_id],
        "source_manifest_sha256": r0_manifest_sha256,
    }


def _view_row(
    candidate: Candidate,
    *,
    family_view: str,
    decision_landmark: str,
    landmark_ts_ns: int,
) -> dict[str, str]:
    segment = candidate.segment_id
    quality_eligible = (
        landmark_ts_ns + OUTCOME_HORIZON_NS <= candidate.segment_end_ts_ns
    )
    return {
        "candidate_id": candidate.candidate_id,
        "episode_id": candidate.episode_id,
        "family_view": family_view,
        "decision_landmark": decision_landmark,
        "landmark_ts_ns": str(landmark_ts_ns),
        "anchor_artifact": f"anchors/{segment}.csv.gz",
        "sparse_range_artifact": (
            f"paths/sparse_range_index/{family_view}/{segment}.csv.gz"
        ),
        "fixed_grid_artifact": f"paths/fixed_grid/{family_view}/{segment}.csv.gz",
        "event_count_artifact": (
            f"paths/event_count/{family_view}/{segment}.csv.gz"
        ),
        "feature_ledger_artifact": f"features/{family_view}/{segment}.csv.gz",
        "market_outcome_artifact": f"outcomes/{segment}.csv.gz",
        "population_eligible": "true",
        "quality_subset_eligible": _bool_text(quality_eligible),
        "quality_subset_reason": ""
        if quality_eligible
        else "insufficient_same_segment_2000ms_room",
    }


def _process_segment(
    output_dir: Path,
    *,
    spec: SegmentSpec,
    candidates: Sequence[Candidate],
    quality_rows: Sequence[Mapping[str, Any]],
    topology_fingerprint: str,
    r0_manifest_sha256: str,
) -> tuple[dict[str, Any], set[str]]:
    if not candidates or any(candidate.segment_id != spec.segment_id for candidate in candidates):
        raise EpisodeV3Error(f"{spec.segment_id}: candidate partition drift")
    data = _load_segment_data(spec, quality_rows, candidates)
    paths = {
        "anchor": output_dir / f"anchors/{spec.segment_id}.csv.gz",
        "family_a": output_dir / f"views/family_a/{spec.segment_id}.csv.gz",
        "family_b": output_dir / f"views/family_b/{spec.segment_id}.csv.gz",
        "range_a": output_dir
        / f"paths/sparse_range_index/family_a/{spec.segment_id}.csv.gz",
        "range_b": output_dir
        / f"paths/sparse_range_index/family_b/{spec.segment_id}.csv.gz",
        "grid_a": output_dir
        / f"paths/fixed_grid/family_a/{spec.segment_id}.csv.gz",
        "grid_b": output_dir
        / f"paths/fixed_grid/family_b/{spec.segment_id}.csv.gz",
        "event_a": output_dir
        / f"paths/event_count/family_a/{spec.segment_id}.csv.gz",
        "event_b": output_dir
        / f"paths/event_count/family_b/{spec.segment_id}.csv.gz",
        "feature_a": output_dir / f"features/family_a/{spec.segment_id}.csv.gz",
        "feature_b": output_dir / f"features/family_b/{spec.segment_id}.csv.gz",
        "outcome": output_dir / f"outcomes/{spec.segment_id}.csv.gz",
    }
    counters: Counter[str] = Counter()
    feature_names: set[str] = set()
    with ExitStack() as stack:
        writers = {
            "anchor": _open_csv_writer(stack, paths["anchor"], ANCHOR_FIELDS),
            "family_a": _open_csv_writer(stack, paths["family_a"], VIEW_FIELDS),
            "family_b": _open_csv_writer(stack, paths["family_b"], VIEW_FIELDS),
            "range_a": _open_csv_writer(stack, paths["range_a"], RANGE_FIELDS),
            "range_b": _open_csv_writer(stack, paths["range_b"], RANGE_FIELDS),
            "grid_a": _open_csv_writer(stack, paths["grid_a"], GRID_FIELDS),
            "grid_b": _open_csv_writer(stack, paths["grid_b"], GRID_FIELDS),
            "event_a": _open_csv_writer(
                stack, paths["event_a"], EVENT_COUNT_FIELDS
            ),
            "event_b": _open_csv_writer(
                stack, paths["event_b"], EVENT_COUNT_FIELDS
            ),
            "feature_a": _open_csv_writer(
                stack, paths["feature_a"], FEATURE_FIELDS
            ),
            "feature_b": _open_csv_writer(
                stack, paths["feature_b"], FEATURE_FIELDS
            ),
            "outcome": _open_csv_writer(stack, paths["outcome"], OUTCOME_FIELDS),
        }
        for candidate in candidates:
            auxiliary_ids = _auxiliary_degraded_ids(data, candidate.shock_ts_ns)
            anchor = _anchor_row(
                candidate,
                r0_manifest_sha256=r0_manifest_sha256,
                auxiliary_ids=auxiliary_ids,
            )
            writers["anchor"].writerow(anchor)
            counters["anchor_rows"] += 1
            counters["candidate_rows"] += 1
            if not candidate.primary_episode:
                counters["rejected_rows"] += 1
                if anchor["t_confirm_ns"]:
                    counters["synthetic_rejected_confirm_count"] += 1

            writers["family_a"].writerow(
                _view_row(
                    candidate,
                    family_view="family_a",
                    decision_landmark="t_candidate",
                    landmark_ts_ns=candidate.shock_ts_ns,
                )
            )
            counters["family_a_rows"] += 1
            writers["range_a"].writerow(
                _range_row(
                    candidate,
                    data,
                    family_view="family_a",
                    landmark_ts_ns=candidate.shock_ts_ns,
                )
            )
            counters["sparse_range_family_a_rows"] += 1
            for relative_ms in GRID_MS:
                grid_row = _grid_row(
                    candidate,
                    data,
                    family_view="family_a",
                    decision_landmark="t_candidate",
                    landmark_ts_ns=candidate.shock_ts_ns,
                    relative_ms=relative_ms,
                )
                writers["grid_a"].writerow(grid_row)
                counters["fixed_grid_family_a_rows"] += 1
                if grid_row["auxiliary_degraded_mask"]:
                    counters["auxiliary_degraded_grid_rows"] += 1
                if grid_row["core_degraded_mask"]:
                    counters["core_degraded_grid_rows"] += 1
            for event_row in _event_count_rows(
                candidate,
                data,
                family_view="family_a",
                decision_landmark="t_candidate",
                landmark_ts_ns=candidate.shock_ts_ns,
            ):
                writers["event_a"].writerow(event_row)
                counters["event_count_family_a_rows"] += 1
            family_a_features = list(
                _feature_rows(
                    candidate,
                    data,
                    family_view="family_a",
                    decision_landmark="t_candidate",
                    landmark_ts_ns=candidate.shock_ts_ns,
                    topology_fingerprint=topology_fingerprint,
                )
            )
            if tuple(row["feature_name"] for row in family_a_features) != (
                _feature_name_universe("family_a")
            ):
                raise EpisodeV3Error(
                    f"{candidate.candidate_id}: Family A feature universe drift"
                )
            for feature_row in family_a_features:
                writers["feature_a"].writerow(feature_row)
                feature_names.add(feature_row["feature_name"])
                counters["feature_ledger_family_a_rows"] += 1
                if feature_row["availability_reason"] != "available":
                    counters["feature_unavailable_rows"] += 1
                if feature_row["observed_at_ns"] and int(
                    feature_row["observed_at_ns"]
                ) > candidate.shock_ts_ns:
                    counters["future_feature_observation_mismatch_count"] += 1

            outcome, outcome_counts = _outcome_row(candidate, data)
            writers["outcome"].writerow(outcome)
            counters["outcome_rows"] += 1
            counters["interval_censored_outcome_count"] += outcome_counts[
                "interval_censored"
            ]
            counters["right_censored_outcome_count"] += outcome_counts[
                "right_censored"
            ]
            counters["segment_censored_outcome_count"] += outcome_counts[
                "segment_censored"
            ]
            counters["epoch_censored_outcome_count"] += outcome_counts[
                "epoch_censored"
            ]
            counters["quality_censored_outcome_count"] += outcome_counts[
                "quality_censored"
            ]
            counters["point_coerced_outcome_count"] += outcome_counts[
                "point_coerced"
            ]

            if candidate.family_b_available:
                if candidate.t_confirm_ns is None:
                    raise EpisodeV3Error(f"{candidate.candidate_id}: missing Family B landmark")
                writers["family_b"].writerow(
                    _view_row(
                        candidate,
                        family_view="family_b",
                        decision_landmark="t_confirm",
                        landmark_ts_ns=candidate.t_confirm_ns,
                    )
                )
                counters["family_b_rows"] += 1
                writers["range_b"].writerow(
                    _range_row(
                        candidate,
                        data,
                        family_view="family_b",
                        landmark_ts_ns=candidate.t_confirm_ns,
                    )
                )
                counters["sparse_range_family_b_rows"] += 1
                for relative_ms in GRID_MS:
                    grid_row = _grid_row(
                        candidate,
                        data,
                        family_view="family_b",
                        decision_landmark="t_confirm",
                        landmark_ts_ns=candidate.t_confirm_ns,
                        relative_ms=relative_ms,
                    )
                    writers["grid_b"].writerow(grid_row)
                    counters["fixed_grid_family_b_rows"] += 1
                    if grid_row["auxiliary_degraded_mask"]:
                        counters["auxiliary_degraded_grid_rows"] += 1
                    if grid_row["core_degraded_mask"]:
                        counters["core_degraded_grid_rows"] += 1
                for event_row in _event_count_rows(
                    candidate,
                    data,
                    family_view="family_b",
                    decision_landmark="t_confirm",
                    landmark_ts_ns=candidate.t_confirm_ns,
                ):
                    writers["event_b"].writerow(event_row)
                    counters["event_count_family_b_rows"] += 1
                family_b_features = list(
                    _feature_rows(
                        candidate,
                        data,
                        family_view="family_b",
                        decision_landmark="t_confirm",
                        landmark_ts_ns=candidate.t_confirm_ns,
                        topology_fingerprint=topology_fingerprint,
                    )
                )
                if tuple(row["feature_name"] for row in family_b_features) != (
                    _feature_name_universe("family_b")
                ):
                    raise EpisodeV3Error(
                        f"{candidate.candidate_id}: Family B feature universe drift"
                    )
                for feature_row in family_b_features:
                    writers["feature_b"].writerow(feature_row)
                    feature_names.add(feature_row["feature_name"])
                    counters["feature_ledger_family_b_rows"] += 1
                    if feature_row["availability_reason"] != "available":
                        counters["feature_unavailable_rows"] += 1
                    if feature_row["observed_at_ns"] and int(
                        feature_row["observed_at_ns"]
                    ) > candidate.t_confirm_ns:
                        counters["future_feature_observation_mismatch_count"] += 1

    counters["anchor_ordering_mismatch_count"] = 0
    counters["cross_segment_path_mismatch_count"] = 0
    counters["cross_epoch_path_mismatch_count"] = 0
    return {
        field_name: (
            spec.segment_id
            if field_name == "segment_id"
            else EVIDENCE_LABELS[spec.segment_id]
            if field_name == "evidence_label"
            else str(counters[field_name])
        )
        for field_name in SEGMENT_SUMMARY_FIELDS
    }, feature_names


def _csv_header(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        return list(next(csv.reader(fh), ()))


def _source_catalog_rows(
    specs: Sequence[SegmentSpec],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec in specs:
        sources = (
            (
                "binance_hot_events",
                "binance",
                "book_ticker_and_trade",
                spec.binance_hot_path,
                spec.expected_rows["binance_hot_events"],
                (
                    "jul30:{segment_id}:binance_{event_type}:"
                    "{event_seq}:{source_raw_seq}:0"
                ),
                "exact_sparse_range_and_feature_source",
            ),
            (
                "hyperliquid_hot_events",
                "hyperliquid",
                "bbo_and_trade",
                spec.hyperliquid_hot_path,
                spec.expected_rows["hyperliquid_hot_events"],
                (
                    "jul30:{segment_id}:hyperliquid_{event_type}:"
                    "{event_seq}:{source_raw_seq}:{source_item_index}"
                ),
                "exact_sparse_range_event_count_grid_and_outcome_source",
            ),
            (
                "hyperliquid_auxiliary_events",
                "hyperliquid",
                "auxiliary",
                spec.auxiliary_path,
                spec.expected_rows["hyperliquid_auxiliary_events"],
                (
                    "jul30:{segment_id}:hyperliquid_auxiliary:"
                    "{event_seq}:{source_raw_seq}:0"
                ),
                "auxiliary_feature_availability_only",
            ),
            (
                "common_l2_timeline",
                "cross_venue",
                "replayed_l2_state",
                spec.timeline_path,
                spec.expected_rows["timeline"],
                "jul30:{segment_id}:common_l2_timeline:{common_seq}",
                "shared_state_store_grid_feature_and_outcome_source",
            ),
            (
                "alignment_decision_labels",
                "cross_venue",
                "accepted_strict_asof_evidence",
                spec.decision_label_path,
                spec.expected_rows["decision_labels"],
                "jul30:{segment_id}:decision_label:{decision_seq}",
                "schema_and_strict_asof_admission_evidence",
            ),
        )
        for (
            source_store_id,
            venue,
            channel,
            path,
            row_count,
            identity,
            usage,
        ) in sources:
            _guard_allowed_path(path)
            rows.append(
                {
                    "session_id": SESSION_ID,
                    "segment_id": spec.segment_id,
                    "source_store_id": source_store_id,
                    "venue": venue,
                    "channel": channel,
                    "path": str(path.resolve()),
                    "row_count": str(row_count),
                    "bytes": str(path.stat().st_size),
                    "sha256": sha256_file(path),
                    "schema_fields_json": json.dumps(
                        _csv_header(path), separators=(",", ":")
                    ),
                    "source_event_identity": identity,
                    "connection_epoch_contract": (
                        "epoch=0;accepted_collector_single_epoch_proof;"
                        "segment_boundary_terminates"
                    ),
                    "degraded_contract": (
                        "core_hot_and_timeline_not_globally_degraded;"
                        "segment_0002_auxiliary_masks_are_track_specific"
                    ),
                    "usage": usage,
                }
            )
    return rows


def _runtime_source_test_paths() -> dict[str, Path]:
    return {
        "runtime_source/cross_exchange_trigger_aligned_episodes.py": (
            WORKTREE_ROOT
            / "examples/hyperliquid/cross_exchange_trigger_aligned_episodes.py"
        ),
        "runtime_source/cross_exchange_jul30_episode_v3_admission.py": (
            WORKTREE_ROOT
            / "examples/hyperliquid/cross_exchange_jul30_episode_v3_admission.py"
        ),
        "runtime_tests/test_cross_exchange_trigger_aligned_episodes.py": (
            WORKTREE_ROOT
            / "examples/hyperliquid/test_cross_exchange_trigger_aligned_episodes.py"
        ),
        "runtime_tests/test_cross_exchange_jul30_episode_v3_admission.py": (
            WORKTREE_ROOT
            / "examples/hyperliquid/test_cross_exchange_jul30_episode_v3_admission.py"
        ),
    }


def _artifact_relative_paths() -> tuple[str, ...]:
    paths = [
        "frozen_episode_v3_contract.json",
        "input_bindings.csv",
        "quality_intervals.csv",
        "reports/jul30_episode_v3.md",
        "segment_summary.csv",
        "source_event_store_catalog.csv",
        *_runtime_source_test_paths().keys(),
    ]
    for segment_id in SEGMENT_IDS:
        paths.extend(
            (
                f"anchors/{segment_id}.csv.gz",
                f"views/family_a/{segment_id}.csv.gz",
                f"views/family_b/{segment_id}.csv.gz",
                f"paths/sparse_range_index/family_a/{segment_id}.csv.gz",
                f"paths/sparse_range_index/family_b/{segment_id}.csv.gz",
                f"paths/fixed_grid/family_a/{segment_id}.csv.gz",
                f"paths/fixed_grid/family_b/{segment_id}.csv.gz",
                f"paths/event_count/family_a/{segment_id}.csv.gz",
                f"paths/event_count/family_b/{segment_id}.csv.gz",
                f"features/family_a/{segment_id}.csv.gz",
                f"features/family_b/{segment_id}.csv.gz",
                f"outcomes/{segment_id}.csv.gz",
            )
        )
    return tuple(sorted(paths))


def _artifact_directory_paths() -> tuple[str, ...]:
    directories: set[str] = set()
    for relative_path in _artifact_relative_paths():
        parent = Path(relative_path).parent
        while parent.as_posix() != ".":
            directories.add(parent.as_posix())
            parent = parent.parent
    return tuple(sorted(directories))


def _input_inventory_rows(snapshot_phase: str) -> list[dict[str, str]]:
    roots = (
        ("accepted_stage1_package", "dependency", STAGE1_DIR),
        ("accepted_stage2_package", "dependency", STAGE2_DIR),
        ("accepted_stage3_package", "dependency", STAGE3_DIR),
        ("jul30_campaign", "read_only_jul30_raw_and_timeline", JUL30_CAMPAIGN_DIR),
        ("jul30_r0_r1", "read_only_jul30_structured", JUL30_R0_DIR),
    )
    rows: list[dict[str, str]] = []
    for scope, role, root in roots:
        root = root.resolve()
        for record in _directory_inventory(root):
            path = root / str(record["path"])
            _guard_allowed_path(path)
            rows.append(
                {
                    "snapshot_phase": snapshot_phase,
                    "scope": scope,
                    "role": role,
                    "session_id": SESSION_ID
                    if scope.startswith("jul30_")
                    else "",
                    "segment_id": next(
                        (
                            segment_id
                            for segment_id in SEGMENT_IDS
                            if f"/{segment_id}/" in f"/{record['path']}/"
                        ),
                        "",
                    ),
                    "root": str(root),
                    "path": str(path),
                    "relative_path": str(record["path"]),
                    "bytes": str(record["bytes"]),
                    "sha256": str(record["sha256"]),
                }
            )
    for relative_path, path in sorted(_runtime_source_test_paths().items()):
        if not path.is_file():
            raise EpisodeV3Error(f"missing Stage 4 source/test: {path}")
        rows.append(
            {
                "snapshot_phase": snapshot_phase,
                "scope": "stage4_fixed_source_test",
                "role": "runtime_reproducibility",
                "session_id": "",
                "segment_id": "",
                "root": str(WORKTREE_ROOT),
                "path": str(path.resolve()),
                "relative_path": relative_path,
                "bytes": str(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    rows.sort(
        key=lambda row: (
            row["scope"],
            row["segment_id"],
            row["relative_path"],
            row["path"],
        )
    )
    return rows


def _inventory_identity(rows: Sequence[Mapping[str, str]]) -> str:
    return canonical_json_sha256(
        [
            {key: value for key, value in row.items() if key != "snapshot_phase"}
            for row in rows
        ]
    )


def _topology_fingerprint() -> str:
    rows = list(
        _strict_csv_rows(
            STAGE1_DIR / "data_admission/session_topology.csv"
        )
    )
    matching = [row for row in rows if row["session_id"] == SESSION_ID]
    if len(matching) != 1:
        raise EpisodeV3Error("Jul30 topology row cardinality drift")
    value = matching[0]["collection_topology_fingerprint"]
    if len(value) != 64:
        raise EpisodeV3Error("Jul30 topology fingerprint drift")
    return value


def _expected_manifest_exact_counts() -> dict[str, int]:
    return {
        "family_a_rows": EXPECTED_COUNTS["family_a"],
        "family_b_rows": EXPECTED_COUNTS["family_b"],
        "rejected_rows": EXPECTED_COUNTS["rejected"],
        "anchor_rows": EXPECTED_COUNTS["family_a"],
        "outcome_rows": EXPECTED_COUNTS["family_a"],
        "segment_rows": len(SEGMENT_IDS),
        "cluster_count": EXPECTED_COUNTS["clusters"],
        "flow_count": EXPECTED_COUNTS["flows"],
        "overlap_block_count_2000ms": EXPECTED_COUNTS["overlap_blocks"],
    }


def _source_semantic_aggregate_contract() -> dict[str, Any]:
    def binding(
        section: str,
        field: str,
        expected_value: int,
    ) -> dict[str, Any]:
        return {
            "section": section,
            "field": field,
            "expected_value": expected_value,
        }

    projection_specs = (
        (
            "anchors",
            ANCHOR_FIELDS,
            EXPECTED_COUNTS["family_a"],
            (
                binding("exact_counts", "anchor_rows", EXPECTED_COUNTS["family_a"]),
                binding(
                    "aggregate_output_counts",
                    "candidate_rows",
                    EXPECTED_COUNTS["family_a"],
                ),
                binding(
                    "aggregate_output_counts",
                    "rejected_rows",
                    EXPECTED_COUNTS["rejected"],
                ),
                binding(
                    "aggregate_output_counts",
                    "anchor_rows",
                    EXPECTED_COUNTS["family_a"],
                ),
                binding(
                    "aggregate_output_counts",
                    "synthetic_rejected_confirm_count",
                    0,
                ),
                binding(
                    "aggregate_output_counts",
                    "anchor_ordering_mismatch_count",
                    0,
                ),
            ),
        ),
        (
            "views_family_a",
            VIEW_FIELDS,
            EXPECTED_COUNTS["family_a"],
            (
                binding(
                    "exact_counts",
                    "family_a_rows",
                    EXPECTED_COUNTS["family_a"],
                ),
                binding(
                    "aggregate_output_counts",
                    "family_a_rows",
                    EXPECTED_COUNTS["family_a"],
                ),
            ),
        ),
        (
            "views_family_b",
            VIEW_FIELDS,
            EXPECTED_COUNTS["family_b"],
            (
                binding(
                    "exact_counts",
                    "family_b_rows",
                    EXPECTED_COUNTS["family_b"],
                ),
                binding(
                    "aggregate_output_counts",
                    "family_b_rows",
                    EXPECTED_COUNTS["family_b"],
                ),
            ),
        ),
        (
            "features_family_a",
            FEATURE_FIELDS,
            EXPECTED_COUNTS["feature_family_a"],
            (
                binding(
                    "aggregate_output_counts",
                    "feature_ledger_family_a_rows",
                    EXPECTED_COUNTS["feature_family_a"],
                ),
                binding(
                    "aggregate_output_counts",
                    "feature_unavailable_rows",
                    2_863_855,
                ),
                binding(
                    "aggregate_output_counts",
                    "future_feature_observation_mismatch_count",
                    0,
                ),
            ),
        ),
        (
            "features_family_b",
            FEATURE_FIELDS,
            EXPECTED_COUNTS["feature_family_b"],
            (
                binding(
                    "aggregate_output_counts",
                    "feature_ledger_family_b_rows",
                    EXPECTED_COUNTS["feature_family_b"],
                ),
            ),
        ),
        (
            "sparse_range_family_a",
            RANGE_FIELDS,
            EXPECTED_COUNTS["family_a"],
            (
                binding(
                    "aggregate_output_counts",
                    "sparse_range_family_a_rows",
                    EXPECTED_COUNTS["family_a"],
                ),
                binding(
                    "aggregate_output_counts",
                    "cross_segment_path_mismatch_count",
                    0,
                ),
                binding(
                    "aggregate_output_counts",
                    "cross_epoch_path_mismatch_count",
                    0,
                ),
            ),
        ),
        (
            "sparse_range_family_b",
            RANGE_FIELDS,
            EXPECTED_COUNTS["family_b"],
            (
                binding(
                    "aggregate_output_counts",
                    "sparse_range_family_b_rows",
                    EXPECTED_COUNTS["family_b"],
                ),
            ),
        ),
        (
            "fixed_grid_family_a",
            GRID_FIELDS,
            EXPECTED_COUNTS["family_a"] * len(GRID_MS),
            (
                binding(
                    "aggregate_output_counts",
                    "fixed_grid_family_a_rows",
                    EXPECTED_COUNTS["family_a"] * len(GRID_MS),
                ),
                binding(
                    "aggregate_output_counts",
                    "auxiliary_degraded_grid_rows",
                    4_316,
                ),
                binding(
                    "aggregate_output_counts",
                    "core_degraded_grid_rows",
                    0,
                ),
            ),
        ),
        (
            "fixed_grid_family_b",
            GRID_FIELDS,
            EXPECTED_COUNTS["family_b"] * len(GRID_MS),
            (
                binding(
                    "aggregate_output_counts",
                    "fixed_grid_family_b_rows",
                    EXPECTED_COUNTS["family_b"] * len(GRID_MS),
                ),
            ),
        ),
        (
            "event_count_family_a",
            EVENT_COUNT_FIELDS,
            EXPECTED_COUNTS["family_a"] * 3,
            (
                binding(
                    "aggregate_output_counts",
                    "event_count_family_a_rows",
                    EXPECTED_COUNTS["family_a"] * 3,
                ),
            ),
        ),
        (
            "event_count_family_b",
            EVENT_COUNT_FIELDS,
            EXPECTED_COUNTS["family_b"] * 3,
            (
                binding(
                    "aggregate_output_counts",
                    "event_count_family_b_rows",
                    EXPECTED_COUNTS["family_b"] * 3,
                ),
            ),
        ),
        (
            "outcomes",
            OUTCOME_FIELDS,
            EXPECTED_COUNTS["family_a"],
            (
                binding(
                    "exact_counts",
                    "outcome_rows",
                    EXPECTED_COUNTS["family_a"],
                ),
                binding(
                    "aggregate_output_counts",
                    "outcome_rows",
                    EXPECTED_COUNTS["family_a"],
                ),
                binding(
                    "aggregate_output_counts",
                    "interval_censored_outcome_count",
                    627_406,
                ),
                binding(
                    "aggregate_output_counts",
                    "right_censored_outcome_count",
                    177_826,
                ),
                binding(
                    "aggregate_output_counts",
                    "segment_censored_outcome_count",
                    316,
                ),
                binding(
                    "aggregate_output_counts",
                    "epoch_censored_outcome_count",
                    0,
                ),
                binding(
                    "aggregate_output_counts",
                    "quality_censored_outcome_count",
                    18,
                ),
                binding(
                    "aggregate_output_counts",
                    "point_coerced_outcome_count",
                    0,
                ),
            ),
        ),
    )
    projections = {
        name: {
            "fields": list(fields),
            "expected_rows": expected_rows,
            "manifest_count_bindings": list(manifest_count_bindings),
        }
        for name, fields, expected_rows, manifest_count_bindings in projection_specs
    }
    return {
        "contract_version": SOURCE_SEMANTIC_AGGREGATE_CONTRACT_VERSION,
        "projection_key_policy": "exact",
        "entry_key_policy": "exact",
        "entry_keys": list(SOURCE_SEMANTIC_AGGREGATE_ENTRY_KEYS),
        "manifest_aggregate_output_count_key_policy": {
            "policy": "exact",
            "derivation": (
                "unique binding.field values from "
                "projections[*].manifest_count_bindings where "
                "binding.section=aggregate_output_counts"
            ),
        },
        "canonical_integer_policy": (
            "type(value) is int; bool and string encodings are forbidden"
        ),
        "digest_policy": (
            "lowercase canonical 64-hex sha256; expected_sha256 must equal "
            "observed_sha256"
        ),
        "mismatch_policy": "mismatch_rows is canonical int zero",
        "projections": projections,
    }


def _source_semantic_projection_contract() -> dict[str, Any]:
    return {
        "aggregate_evidence_contract": _source_semantic_aggregate_contract(),
        "exact_feature_projection": {
            "comparison": (
                "source-derived exact whole-row ordered stream; every "
                "FEATURE_FIELDS value is compared"
            ),
            "fields": list(FEATURE_FIELDS),
            "families": {
                "family_a": {
                    "feature_names": list(_feature_name_universe("family_a")),
                    "row_count": EXPECTED_COUNTS["feature_family_a"],
                },
                "family_b": {
                    "feature_names": list(_feature_name_universe("family_b")),
                    "row_count": EXPECTED_COUNTS["feature_family_b"],
                },
            },
            "coverage": "all feature rows in both families",
        },
        "exact_view_projection": {
            "comparison": (
                "source-derived exact whole-row ordered stream; every "
                "VIEW_FIELDS value is compared"
            ),
            "fields": list(VIEW_FIELDS),
            "families": {
                "family_a": {"row_count": EXPECTED_COUNTS["family_a"]},
                "family_b": {"row_count": EXPECTED_COUNTS["family_b"]},
            },
            "coverage": (
                "all population, linkage, eligibility, quality, and artifact "
                "reference fields in both families"
            ),
        },
    }


def _require_canonical_evidence_int(value: Any, *, label: str) -> int:
    if type(value) is not int:
        raise EpisodeV3Error(f"{label}: canonical integer required")
    return value


def _require_canonical_sha256(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EpisodeV3Error(f"{label}: canonical lowercase SHA256 required")
    return value


def _manifest_count_binding_expected_values(
    projection_contracts: Mapping[str, Mapping[str, Any]],
    *,
    section: str,
) -> dict[str, int]:
    expected_values: dict[str, int] = {}
    expected_binding_keys = {"section", "field", "expected_value"}
    for projection_name, projection_contract in projection_contracts.items():
        for binding in projection_contract["manifest_count_bindings"]:
            if type(binding) is not dict or set(binding) != expected_binding_keys:
                raise EpisodeV3Error(
                    f"{projection_name}: manifest count binding schema drift"
                )
            if binding["section"] != section:
                continue
            field_name = binding["field"]
            if type(field_name) is not str or not field_name:
                raise EpisodeV3Error(
                    f"{projection_name}: manifest count binding field drift"
                )
            if field_name in expected_values:
                raise EpisodeV3Error(
                    f"{projection_name}: duplicate manifest count binding "
                    f"{section}.{field_name}"
                )
            expected_values[field_name] = _require_canonical_evidence_int(
                binding["expected_value"],
                label=(
                    f"{projection_name}: manifest count binding expected value "
                    f"{section}.{field_name}"
                ),
            )
    return expected_values


def _validate_source_semantic_verification_evidence(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    semantic_verification = manifest.get("_source_semantic_verification")
    expected_semantic_keys = {
        "source_semantic_verified",
        "scope",
        "aggregate",
    }
    if (
        type(semantic_verification) is not dict
        or set(semantic_verification) != expected_semantic_keys
        or semantic_verification["source_semantic_verified"] is not True
    ):
        raise EpisodeV3Error(
            "complete source-semantic verification evidence missing"
        )

    expected_scope = _source_semantic_projection_contract()
    scope = semantic_verification["scope"]
    if type(scope) is not dict or scope != expected_scope:
        raise EpisodeV3Error("source-semantic verification scope drift")

    aggregate_contract = expected_scope["aggregate_evidence_contract"]
    projection_contracts = aggregate_contract["projections"]
    aggregate = semantic_verification["aggregate"]
    if type(aggregate) is not dict:
        raise EpisodeV3Error("source-semantic aggregate must be a dict")
    if set(aggregate) != set(projection_contracts):
        raise EpisodeV3Error(
            "source-semantic aggregate projection universe drift"
        )

    exact_counts = manifest.get("exact_counts")
    expected_exact_counts = _expected_manifest_exact_counts()
    if type(exact_counts) is not dict or set(exact_counts) != set(
        expected_exact_counts
    ):
        raise EpisodeV3Error("manifest exact count evidence drift")
    for key, expected_value in expected_exact_counts.items():
        actual_value = _require_canonical_evidence_int(
            exact_counts[key], label=f"manifest exact_counts.{key}"
        )
        if actual_value != expected_value:
            raise EpisodeV3Error(f"manifest exact_counts.{key} contradiction")

    aggregate_output_counts = manifest.get("aggregate_output_counts")
    expected_aggregate_output_counts = _manifest_count_binding_expected_values(
        projection_contracts,
        section="aggregate_output_counts",
    )
    if (
        type(aggregate_output_counts) is not dict
        or set(aggregate_output_counts) != set(expected_aggregate_output_counts)
    ):
        raise EpisodeV3Error(
            "manifest aggregate_output_counts key universe drift"
        )
    for field_name, expected_value in expected_aggregate_output_counts.items():
        observed_value = _require_canonical_evidence_int(
            aggregate_output_counts[field_name],
            label=f"manifest aggregate_output_counts.{field_name}",
        )
        if observed_value != expected_value:
            raise EpisodeV3Error(
                "manifest aggregate_output_counts."
                f"{field_name} contradiction"
            )

    expected_entry_keys = set(aggregate_contract["entry_keys"])
    for projection_name, projection_contract in projection_contracts.items():
        entry = aggregate[projection_name]
        if type(entry) is not dict:
            raise EpisodeV3Error(
                f"{projection_name}: aggregate entry must be a dict"
            )
        if set(entry) != expected_entry_keys:
            raise EpisodeV3Error(
                f"{projection_name}: aggregate entry key set drift"
            )
        if type(entry["fields"]) is not list or entry["fields"] != (
            projection_contract["fields"]
        ):
            raise EpisodeV3Error(f"{projection_name}: aggregate fields drift")

        expected_rows = projection_contract["expected_rows"]
        for count_key in ("expected_rows", "observed_rows"):
            count = _require_canonical_evidence_int(
                entry[count_key],
                label=f"{projection_name}.{count_key}",
            )
            if count != expected_rows:
                raise EpisodeV3Error(
                    f"{projection_name}.{count_key} contradiction"
                )

        expected_sha256 = _require_canonical_sha256(
            entry["expected_sha256"],
            label=f"{projection_name}.expected_sha256",
        )
        observed_sha256 = _require_canonical_sha256(
            entry["observed_sha256"],
            label=f"{projection_name}.observed_sha256",
        )
        if expected_sha256 != observed_sha256:
            raise EpisodeV3Error(f"{projection_name}: aggregate digest mismatch")

        mismatch_rows = _require_canonical_evidence_int(
            entry["mismatch_rows"],
            label=f"{projection_name}.mismatch_rows",
        )
        if mismatch_rows != 0:
            raise EpisodeV3Error(
                f"{projection_name}: aggregate mismatch_rows must be zero"
            )

        for binding in projection_contract["manifest_count_bindings"]:
            section_name = binding["section"]
            field_name = binding["field"]
            expected_bound_count = binding["expected_value"]
            section = (
                exact_counts
                if section_name == "exact_counts"
                else aggregate_output_counts
            )
            binding_label = f"{section_name}.{field_name}"
            if field_name not in section:
                raise EpisodeV3Error(
                    f"{projection_name}: manifest count binding missing "
                    f"{binding_label}"
                )
            bound_count = _require_canonical_evidence_int(
                section[field_name],
                label=f"manifest {binding_label}",
            )
            if bound_count != expected_bound_count:
                raise EpisodeV3Error(
                    f"{projection_name}: manifest count contradiction "
                    f"{binding_label}"
                )
    return semantic_verification


def _canonical_contract(
    *,
    feature_names: Sequence[str],
    source_catalog_rows: Sequence[Mapping[str, str]],
    dependency_identities: Mapping[str, Mapping[str, Any]],
    input_inventory_sha256: str,
) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "frozen_date": FROZEN_DATE,
        "population": {
            "family_a": {
                "count": EXPECTED_COUNTS["family_a"],
                "landmark": "t_candidate=shock_ts_ns",
                "includes_all_rejected": True,
                "rejected_t_confirm": None,
            },
            "family_b": {
                "count": EXPECTED_COUNTS["family_b"],
                "landmark": "t_confirm=decision_ts_ns",
                "linked_subset": "same candidate IDs where primary_episode=true",
            },
            "rejected_count": EXPECTED_COUNTS["rejected"],
            "shared_append_only_candidate_record": True,
        },
        "anchor_schema": list(ANCHOR_FIELDS),
        "classification": {
            "field": "classification",
            "semantic_mapping": "classification := detector attribution",
            "source": CLASSIFICATION_SOURCE,
            "allowed_values": list(CLASSIFICATION_ALLOWED_VALUES),
            "reconstruction": (
                "exact candidate_id join to accepted Stage 3 "
                "candidate_audit_projection.csv.gz attribution"
            ),
        },
        "view_schema": list(VIEW_FIELDS),
        "sparse_range_schema": list(RANGE_FIELDS),
        "fixed_grid": {
            "points_ms": list(GRID_MS),
            "schema": list(GRID_FIELDS),
            "strict_asof": True,
            "no_new_information_required_for_forward_fill": True,
        },
        "event_count": {
            "points": list(EVENT_COUNT_POINTS),
            "channels": [
                "hyperliquid_bbo",
                "hyperliquid_trade",
                "hyperliquid_fast_l2",
            ],
            "schema": list(EVENT_COUNT_FIELDS),
            "fast_l2_event_identity": (
                "unique hyperliquid_fast_local_ts_ns changes in common timeline;"
                "forward-filled repeats are not events"
            ),
        },
        "feature_observation_ledger": {
            "schema": list(FEATURE_FIELDS),
            "feature_names": list(sorted(feature_names)),
            "invariant": "every non-null observed_at_ns <= family landmark",
            "unavailable_value": "",
            "unavailable_requires_reason": True,
            "trade_quantity_source": TRADE_QUANTITY_SOURCE,
            "rpi_adjustment_status": RPI_ADJUSTMENT_STATUS,
            "depth_stream": DEPTH_STREAM,
            "underlying_market_state": UNDERLYING_MARKET_STATE,
            "confirmation_feature_names": list(CONFIRMATION_FEATURE_NAMES),
            "confirmation_provenance": (
                "common L2 timeline first state at/after Candidate satisfying "
                "the frozen detector depletion/drop predicate"
            ),
            "confirmation_observed_at": "t_confirm_ns exactly",
            "confirmation_source_event_identity": (
                "jul30:{segment_id}:common_l2_timeline:{common_seq}"
            ),
            "confirmed_burst_membership": (
                "accepted Stage 3 fixed-origin 10ms burst, terminated by "
                "opposite side, zero-economic reset, or origin-relative gap"
            ),
        },
        "response_outcome": {
            "schema": list(OUTCOME_FIELDS),
            "first_event_names": list(FIRST_EVENT_NAMES),
            "first_event_interval": (
                "(max(t_candidate,last_non_event_observed_at),"
                "first_event_observed_at]"
            ),
            "first_event_left_truncation": (
                "interval_lower_ns >= t_candidate_ns for every post-trigger "
                "first-event outcome"
            ),
            "vulnerable_quote_baseline": (
                "last Hyperliquid BBO with local receipt timestamp "
                "strictly less than t_candidate_ns"
            ),
            "cross_venue_baseline": (
                "last common L2 timeline state with common timestamp "
                "strictly less than t_candidate_ns"
            ),
            "equal_receipt_timestamp_policy": (
                "events at t_candidate_ns are excluded from the baseline and "
                "from post-trigger first-event observation"
            ),
            "first_event_point_coercion_forbidden": True,
            "right_censor": "full 2000ms core path with event not observed",
            "segment_censor": "structural segment end before 2000ms",
            "epoch_censor": (
                "epoch=0 equals structural segment; segment takes precedence"
            ),
            "quality_censor": (
                "core quality failure only; Jul30 auxiliary intervals do not "
                "censor BBO/trade/fast-L2 outcomes"
            ),
            "gap_survival_horizons_ms": list(OUTCOME_HORIZONS_MS),
            "markout_horizons_ms": list(MARKOUT_HORIZONS_MS),
            "public_quote_risk_fields": [
                "public_trade_reaches_quote",
                "public_bbo_moves_through_quote",
                "public_quote_survives_horizon",
                "public_adverse_exposure",
            ],
        },
        "direction_normalization": {
            "aggressive_buy": {
                "direction_sign": 1,
                "vulnerable_side": "ask",
                "risk_gap": "d_bh=binance_bid-hyperliquid_ask",
            },
            "aggressive_sell": {
                "direction_sign": -1,
                "vulnerable_side": "bid",
                "risk_gap": "d_hb=hyperliquid_bid-binance_ask",
            },
            "bps_reference": (
                "mean(binance_midpoint,hyperliquid_bbo_midpoint)"
            ),
        },
        "source_event_store": {
            "storage": (
                "shared immutable accepted Jul30 structured partitions; "
                "candidate windows use deterministic range indexes"
            ),
            "physical_event_duplication_per_episode": False,
            "catalog_row_count": len(source_catalog_rows),
            "catalog_row_stream_sha256": canonical_json_sha256(
                list(source_catalog_rows)
            ),
            "r0_aggregate_counts": {
                "binance_hot_rows": 10744733,
                "hyperliquid_hot_rows": 541122,
                "hyperliquid_auxiliary_rows": 49169,
                "timeline_rows": 556861,
            },
        },
        "quality": {
            "connection_epoch": (
                "accepted Stage 2 single epoch id 0 per segment"
            ),
            "segment_0002_auxiliary_intervals": [
                "asset_context",
                "main_all_mids",
            ],
            "auxiliary_degradation_is_not_core_outcome_censor": True,
            "core_degraded_grid_rows_expected": 0,
        },
        "source_semantic_admission": {
            "oracle": (
                "accepted Stage 2/3 candidate truth plus accepted Jul30 "
                "R0/common-timeline/hot-event/quality sources"
            ),
            "package_self_hash_is_not_an_oracle": True,
            "exact_full_row_projections": list(
                _source_semantic_aggregate_contract()["projections"]
            ),
            **_source_semantic_projection_contract(),
            "source_semantic_verified_scope": (
                "all fields and all ordered rows in every named exact "
                "projection, including both Family A/B feature and view "
                "populations"
            ),
            "standalone_attestation": (
                "before source_semantic_verified=true, require exact scope, "
                "exact aggregate projection and entry key sets, frozen field "
                "order and row counts, canonical equal SHA256 digests, zero "
                "mismatch rows, and exact manifest count bindings"
            ),
            "coherent_manifest_rehash_must_fail_closed": True,
            "archived_verifier_bound_to_current_runtime_source": True,
        },
        "episode_membership": {
            "contract": "episode_merging_v1 accepted Stage 2 exact join",
            "cluster_count": EXPECTED_COUNTS["clusters"],
            "flow_count": EXPECTED_COUNTS["flows"],
            "overlap_block_count_2000ms": EXPECTED_COUNTS["overlap_blocks"],
        },
        "evidence_labels": EVIDENCE_LABELS,
        "canonical_serialization": {
            "json": "indent=2,sort_keys=true,trailing_newline",
            "csv": "utf-8,LF,exact_header_and_cell_width",
            "gzip": "filename_empty,mtime=0,compresslevel=1",
        },
        "tree_entry_type_contract": _tree_entry_type_contract(),
        "artifact_path_allowlist": list(_artifact_relative_paths()),
        "artifact_directory_allowlist": list(_artifact_directory_paths()),
        "atomic_publication": {
            "staging_build": True,
            "fsync_before_publish": True,
            "atomic_directory_swap_or_rename": True,
            "partial_output_visible": False,
        },
        "dependency_identities": dependency_identities,
        "input_inventory_sha256": input_inventory_sha256,
        "boundary": BOUNDARY,
        "forbidden_field_tokens": list(FORBIDDEN_FIELD_TOKENS),
        "forbidden_input_path_tokens": list(FORBIDDEN_PATH_TOKENS),
    }


def _canonical_report(
    segment_rows: Sequence[Mapping[str, str]],
    aggregate_counts: Mapping[str, Any],
    input_inventory_sha256: str,
) -> str:
    lines = [
        "# Jul30 Episode v3 Build",
        "",
        f"- schema: `{SCHEMA_VERSION}`",
        f"- Family A: `{aggregate_counts['family_a_rows']}`",
        f"- Family B: `{aggregate_counts['family_b_rows']}`",
        f"- rejected Family A: `{aggregate_counts['rejected_rows']}`",
        f"- clusters/flows/overlap blocks: `{EXPECTED_COUNTS['clusters']}` / "
        f"`{EXPECTED_COUNTS['flows']}` / `{EXPECTED_COUNTS['overlap_blocks']}`",
        f"- input inventory SHA256: `{input_inventory_sha256}`",
        "- source storage: accepted shared Jul30 R0/timeline partitions plus "
        "candidate range indexes",
        "- outcome timing: interval/right/segment censored public-market evidence",
        "- auxiliary masks: segment_0002 track-specific; core outcomes remain separate",
        "- boundary: no later-session event rows, models, actionability, orders, or PnL",
        "",
        "## Segments",
        "",
        "| segment | candidates | Family B | feature rows A/B | interval/right/segment censor |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in segment_rows:
        lines.append(
            f"| {row['segment_id']} | {row['candidate_rows']} | "
            f"{row['family_b_rows']} | "
            f"{row['feature_ledger_family_a_rows']}/"
            f"{row['feature_ledger_family_b_rows']} | "
            f"{row['interval_censored_outcome_count']}/"
            f"{row['right_censored_outcome_count']}/"
            f"{row['segment_censored_outcome_count']} |"
        )
    return "\n".join(lines) + "\n"


def _fsync_tree(root: Path) -> None:
    entries = _exact_tree_entries(root)
    for entry in entries:
        if entry.entry_type != "regular_file":
            continue
        with entry.path.open("rb") as fh:
            os.fsync(fh.fileno())
    directories = sorted(
        (
            entry.path
            for entry in entries
            if entry.entry_type == "directory"
        ),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in directories:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_exchange_directories(left: Path, right: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renamex_np = getattr(libc, "renamex_np", None)
    if renamex_np is None:
        raise EpisodeV3Error("atomic directory exchange is unavailable")
    renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    renamex_np.restype = ctypes.c_int
    if renamex_np(os.fsencode(left), os.fsencode(right), 0x00000002) != 0:
        error_number = ctypes.get_errno()
        raise EpisodeV3Error(
            f"atomic directory exchange failed: errno={error_number}"
        )


def _publish_staging(staging: Path, output_dir: Path) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if not output_dir.exists():
        os.rename(staging, output_dir)
    else:
        _atomic_exchange_directories(staging, output_dir)
        shutil.rmtree(staging)
    parent_fd = os.open(output_dir.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _artifact_records(root: Path) -> list[dict[str, Any]]:
    expected = set(_artifact_relative_paths())
    expected_directories = set(_artifact_directory_paths())
    entries = _exact_tree_entries(root)
    observed = {
        entry.relative_path
        for entry in entries
        if entry.entry_type == "regular_file"
        and entry.relative_path != "episode_v3_manifest.json"
    }
    if observed != expected:
        raise EpisodeV3Error(
            f"artifact universe drift missing={sorted(expected-observed)} "
            f"extra={sorted(observed-expected)}"
        )
    observed_directories = {
        entry.relative_path
        for entry in entries
        if entry.entry_type == "directory"
    }
    if observed_directories != expected_directories:
        raise EpisodeV3Error(
            "artifact directory universe drift "
            f"missing={sorted(expected_directories-observed_directories)} "
            f"extra={sorted(observed_directories-expected_directories)}"
        )
    entry_by_path = {
        entry.relative_path: entry
        for entry in entries
        if entry.entry_type == "regular_file"
    }
    return [
        {
            "path": relative_path,
            "bytes": entry_by_path[relative_path].bytes,
            "sha256": sha256_file(entry_by_path[relative_path].path),
        }
        for relative_path in sorted(expected)
    ]


def _copy_runtime_source_tests(output_dir: Path) -> dict[str, str]:
    shas: dict[str, str] = {}
    for relative_path, source in sorted(_runtime_source_test_paths().items()):
        target = output_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        shas[relative_path] = sha256_file(target)
    return shas


def build_package(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir = Path(os.path.abspath(output_dir))
    if os.path.lexists(output_dir):
        _exact_tree_entries(output_dir)
    dependencies = _verify_dependency_packages()
    r0, _alignment, specs = _load_segment_specs()
    quality_rows = _load_quality_intervals()
    before_rows = _input_inventory_rows("before")
    before_identity = _inventory_identity(before_rows)
    topology_fingerprint = _topology_fingerprint()
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        source_catalog_rows = _source_catalog_rows(specs)
        _write_csv(
            staging / "source_event_store_catalog.csv",
            source_catalog_rows,
            SOURCE_CATALOG_FIELDS,
        )
        _write_csv(
            staging / "quality_intervals.csv",
            quality_rows,
            QUALITY_INTERVAL_FIELDS,
        )
        segment_specs = {spec.segment_id: spec for spec in specs}
        segment_rows: list[dict[str, str]] = []
        feature_names: set[str] = set()
        observed_segment_order: list[str] = []
        for segment_id, candidates in _group_candidates_by_segment():
            observed_segment_order.append(segment_id)
            if segment_id not in segment_specs:
                raise EpisodeV3Error(f"unexpected candidate segment {segment_id}")
            summary, names = _process_segment(
                staging,
                spec=segment_specs[segment_id],
                candidates=candidates,
                quality_rows=quality_rows,
                topology_fingerprint=topology_fingerprint,
                r0_manifest_sha256=EXPECTED_R0_MANIFEST_SHA256,
            )
            segment_rows.append(summary)
            feature_names.update(names)
            print(
                json.dumps(
                    {
                        "stage4_segment_complete": segment_id,
                        "candidates": len(candidates),
                        "family_b": int(summary["family_b_rows"]),
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
        if tuple(observed_segment_order) != SEGMENT_IDS:
            raise EpisodeV3Error("candidate segment order/cardinality drift")
        _write_csv(
            staging / "segment_summary.csv",
            segment_rows,
            SEGMENT_SUMMARY_FIELDS,
        )
        aggregate: Counter[str] = Counter()
        for row in segment_rows:
            for field_name in SEGMENT_SUMMARY_FIELDS[2:]:
                aggregate[field_name] += int(row[field_name])
        if (
            aggregate["family_a_rows"] != EXPECTED_COUNTS["family_a"]
            or aggregate["anchor_rows"] != EXPECTED_COUNTS["family_a"]
            or aggregate["outcome_rows"] != EXPECTED_COUNTS["family_a"]
            or aggregate["family_b_rows"] != EXPECTED_COUNTS["family_b"]
            or aggregate["rejected_rows"] != EXPECTED_COUNTS["rejected"]
            or aggregate["synthetic_rejected_confirm_count"] != 0
            or aggregate["future_feature_observation_mismatch_count"] != 0
            or aggregate["anchor_ordering_mismatch_count"] != 0
            or aggregate["cross_segment_path_mismatch_count"] != 0
            or aggregate["cross_epoch_path_mismatch_count"] != 0
            or aggregate["point_coerced_outcome_count"] != 0
            or aggregate["core_degraded_grid_rows"] != 0
        ):
            raise EpisodeV3Error("Stage 4 aggregate acceptance counts drift")

        runtime_shas = _copy_runtime_source_tests(staging)
        after_rows = _input_inventory_rows("after")
        after_identity = _inventory_identity(after_rows)
        if before_identity != after_identity:
            raise EpisodeV3Error("Stage 4 input inventory changed during build")
        _write_csv(
            staging / "input_bindings.csv",
            [*before_rows, *after_rows],
            INPUT_BINDING_FIELDS,
        )
        contract = _canonical_contract(
            feature_names=sorted(feature_names),
            source_catalog_rows=source_catalog_rows,
            dependency_identities=dependencies,
            input_inventory_sha256=before_identity,
        )
        _write_json(staging / "frozen_episode_v3_contract.json", contract)
        report = _canonical_report(segment_rows, aggregate, before_identity)
        report_path = staging / "reports/jul30_episode_v3.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        artifacts = _artifact_records(staging)
        core_sha = canonical_json_sha256(artifacts)
        manifest = {
            "task_id": TASK_ID,
            "schema_version": SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "frozen_date": FROZEN_DATE,
            "manifest_path": "episode_v3_manifest.json",
            "artifact_path_allowlist": list(_artifact_relative_paths()),
            "artifact_directory_allowlist": list(_artifact_directory_paths()),
            "artifacts": artifacts,
            "core_package_sha256": core_sha,
            "contract_sha256": sha256_file(
                staging / "frozen_episode_v3_contract.json"
            ),
            "runtime_source_test_sha256_by_path": runtime_shas,
            "dependencies": dependencies,
            "source_manifest_sha256": EXPECTED_R0_MANIFEST_SHA256,
            "input_inventory_sha256_before": before_identity,
            "input_inventory_sha256_after": after_identity,
            "input_inventory_unchanged": True,
            "exact_counts": _expected_manifest_exact_counts(),
            "aggregate_output_counts": dict(sorted(aggregate.items())),
            "feature_names": sorted(feature_names),
            "source_catalog_rows": len(source_catalog_rows),
            "quality_interval_rows": len(quality_rows),
            "boundary": BOUNDARY,
        }
        _write_json(staging / "episode_v3_manifest.json", manifest)
        _fsync_tree(staging)
        verified_manifest = dict(verify_package(staging))
        expected_published_inventory = _directory_inventory(staging)
        _publish_staging(staging, output_dir)
        if _directory_inventory(output_dir) != expected_published_inventory:
            raise EpisodeV3Error("atomic publication byte identity drift")
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return verified_manifest


def _assert_canonical_int_text(value: str, *, label: str, nullable: bool = False) -> None:
    if nullable and value == "":
        return
    try:
        parsed = int(value)
    except ValueError as exc:
        raise EpisodeV3Error(f"{label}: invalid integer text") from exc
    if str(parsed) != value:
        raise EpisodeV3Error(f"{label}: noncanonical integer text")


def _assert_canonical_float_text(
    value: str, *, label: str, nullable: bool = False
) -> None:
    if nullable and value == "":
        return
    try:
        parsed = float(value)
    except ValueError as exc:
        raise EpisodeV3Error(f"{label}: invalid float text") from exc
    if _float_text(parsed) != value:
        raise EpisodeV3Error(f"{label}: noncanonical float text")


def _verify_gzip_headers(output_dir: Path) -> None:
    for path in sorted(output_dir.rglob("*.gz")):
        header = path.read_bytes()[:10]
        if len(header) != 10 or header[:3] != b"\x1f\x8b\x08":
            raise EpisodeV3Error(f"{path}: invalid gzip header")
        if header[3] & 0x08 or header[4:8] != b"\x00\x00\x00\x00":
            raise EpisodeV3Error(f"{path}: nondeterministic gzip metadata")


def _verify_runtime_archive(
    output_dir: Path, manifest: Mapping[str, Any]
) -> None:
    observed: dict[str, str] = {}
    for relative_path, fixed_path in sorted(_runtime_source_test_paths().items()):
        archived = output_dir / relative_path
        if not archived.is_file():
            raise EpisodeV3Error(f"missing runtime archive: {relative_path}")
        if archived.read_bytes() != fixed_path.read_bytes():
            raise EpisodeV3Error(
                f"fixed worktree source/test archive identity drift: {relative_path}"
            )
        observed[relative_path] = sha256_file(archived)
    if observed != manifest["runtime_source_test_sha256_by_path"]:
        raise EpisodeV3Error("runtime source/test SHA map drift")


def _verify_input_bindings(
    output_dir: Path, manifest: Mapping[str, Any]
) -> None:
    published = list(
        _strict_csv_rows(output_dir / "input_bindings.csv", INPUT_BINDING_FIELDS)
    )
    before = [row for row in published if row["snapshot_phase"] == "before"]
    after = [row for row in published if row["snapshot_phase"] == "after"]
    if len(before) + len(after) != len(published) or not before or not after:
        raise EpisodeV3Error("input binding phase universe drift")
    before_payload = [
        {key: value for key, value in row.items() if key != "snapshot_phase"}
        for row in before
    ]
    after_payload = [
        {key: value for key, value in row.items() if key != "snapshot_phase"}
        for row in after
    ]
    if before_payload != after_payload:
        raise EpisodeV3Error("input before/after binding drift")
    actual = _input_inventory_rows("before")
    if actual != before:
        raise EpisodeV3Error("actual Stage 4 input inventory drift")
    identity = _inventory_identity(before)
    if (
        identity != manifest["input_inventory_sha256_before"]
        or identity != manifest["input_inventory_sha256_after"]
        or manifest["input_inventory_unchanged"] is not True
    ):
        raise EpisodeV3Error("manifest input inventory identity drift")
    for row in published:
        path = Path(row["path"])
        if row["path"] != str(path) or row["path"] != str(path.resolve()):
            raise EpisodeV3Error("noncanonical input path text")
        _guard_allowed_path(path)


def _verify_artifact_closure(output_dir: Path) -> dict[str, Any]:
    entries = _exact_tree_entries(output_dir)
    all_files = {
        entry.relative_path
        for entry in entries
        if entry.entry_type == "regular_file"
    }
    expected_files = set(_artifact_relative_paths()) | {"episode_v3_manifest.json"}
    if all_files != expected_files:
        raise EpisodeV3Error(
            f"package file universe drift missing={sorted(expected_files-all_files)} "
            f"extra={sorted(all_files-expected_files)}"
        )
    observed_directories = {
        entry.relative_path
        for entry in entries
        if entry.entry_type == "directory"
    }
    expected_directories = set(_artifact_directory_paths())
    if observed_directories != expected_directories:
        raise EpisodeV3Error(
            "package directory universe drift "
            f"missing={sorted(expected_directories-observed_directories)} "
            f"extra={sorted(observed_directories-expected_directories)}"
        )
    for entry in entries:
        path = entry.path
        if entry.entry_type == "directory" and path.name == "__pycache__":
            raise EpisodeV3Error("package-local bytecode directory is forbidden")
        lowered = entry.relative_path.lower()
        if any(token in lowered for token in ("aug03", "aug04", "aug07", "0807")):
            raise EpisodeV3Error(f"later-session package artifact is forbidden: {path}")
        if any(
            token in path.name.lower()
            for token in ("own_order", "fill", "pnl", "actionability", "model", "score")
        ):
            raise EpisodeV3Error(f"forbidden package artifact name: {path}")
    manifest = _read_canonical_json(
        output_dir / "episode_v3_manifest.json", label="Episode v3 manifest"
    )
    expected_keys = {
        "task_id",
        "schema_version",
        "contract_version",
        "frozen_date",
        "manifest_path",
        "artifact_path_allowlist",
        "artifact_directory_allowlist",
        "artifacts",
        "core_package_sha256",
        "contract_sha256",
        "runtime_source_test_sha256_by_path",
        "dependencies",
        "source_manifest_sha256",
        "input_inventory_sha256_before",
        "input_inventory_sha256_after",
        "input_inventory_unchanged",
        "exact_counts",
        "aggregate_output_counts",
        "feature_names",
        "source_catalog_rows",
        "quality_interval_rows",
        "boundary",
    }
    if set(manifest) != expected_keys:
        raise EpisodeV3Error("Episode v3 manifest schema drift")
    if (
        manifest["task_id"] != TASK_ID
        or manifest["schema_version"] != SCHEMA_VERSION
        or manifest["contract_version"] != CONTRACT_VERSION
        or manifest["frozen_date"] != FROZEN_DATE
        or manifest["manifest_path"] != "episode_v3_manifest.json"
        or manifest["artifact_path_allowlist"] != list(_artifact_relative_paths())
        or manifest["artifact_directory_allowlist"]
        != list(_artifact_directory_paths())
        or manifest["boundary"] != BOUNDARY
        or manifest["source_manifest_sha256"] != EXPECTED_R0_MANIFEST_SHA256
    ):
        raise EpisodeV3Error("Episode v3 manifest identity drift")
    artifacts = _artifact_records(output_dir)
    if artifacts != manifest["artifacts"]:
        raise EpisodeV3Error("artifact record drift")
    if canonical_json_sha256(artifacts) != manifest["core_package_sha256"]:
        raise EpisodeV3Error("core package SHA drift")
    return manifest


def _verify_segment_outputs(
    output_dir: Path,
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, str]], Counter[str], set[str]]:
    summaries: list[dict[str, str]] = []
    aggregate: Counter[str] = Counter()
    feature_names: set[str] = set()
    global_ids: set[str] = set()
    global_family_b_ids: set[str] = set()
    for segment_id in SEGMENT_IDS:
        anchor_path = output_dir / f"anchors/{segment_id}.csv.gz"
        family_a_path = output_dir / f"views/family_a/{segment_id}.csv.gz"
        family_b_path = output_dir / f"views/family_b/{segment_id}.csv.gz"
        outcome_path = output_dir / f"outcomes/{segment_id}.csv.gz"
        anchors: dict[str, dict[str, str]] = {}
        expected_b_ids: list[str] = []
        counters: Counter[str] = Counter()
        family_b_iter = iter(_strict_csv_rows(family_b_path, VIEW_FIELDS))
        for anchor, family_a, outcome in zip(
            _strict_csv_rows(anchor_path, ANCHOR_FIELDS),
            _strict_csv_rows(family_a_path, VIEW_FIELDS),
            _strict_csv_rows(outcome_path, OUTCOME_FIELDS),
            strict=True,
        ):
            candidate_id = anchor["candidate_id"]
            if (
                candidate_id != anchor["episode_id"]
                or candidate_id != family_a["candidate_id"]
                or candidate_id != outcome["candidate_id"]
                or family_a["episode_id"] != candidate_id
                or outcome["episode_id"] != candidate_id
            ):
                raise EpisodeV3Error(f"{segment_id}: linked candidate identity drift")
            if candidate_id in global_ids:
                raise EpisodeV3Error(f"duplicate candidate ID: {candidate_id}")
            global_ids.add(candidate_id)
            anchors[candidate_id] = anchor
            _assert_canonical_int_text(
                anchor["candidate_seq"], label=f"{candidate_id}:candidate_seq"
            )
            for field_name in (
                "t_burst_start_ns",
                "t_candidate_ns",
                "censor_time_ns",
                "segment_start_ts_ns",
                "segment_end_ts_ns",
            ):
                _assert_canonical_int_text(
                    anchor[field_name], label=f"{candidate_id}:{field_name}"
                )
            _assert_canonical_int_text(
                anchor["t_confirm_ns"],
                label=f"{candidate_id}:t_confirm_ns",
                nullable=True,
            )
            t_burst = int(anchor["t_burst_start_ns"])
            t_candidate = int(anchor["t_candidate_ns"])
            t_confirm = int(anchor["t_confirm_ns"]) if anchor["t_confirm_ns"] else None
            if t_burst > t_candidate or (t_confirm is not None and t_confirm <= t_candidate):
                counters["anchor_ordering_mismatch_count"] += 1
            confirmed = anchor["confirmed"] == "true"
            if anchor["classification"] not in CLASSIFICATION_ALLOWED_VALUES:
                raise EpisodeV3Error(
                    f"{candidate_id}: classification value drift"
                )
            if confirmed != (anchor["family_b_available"] == "true"):
                raise EpisodeV3Error(f"{candidate_id}: Family B availability drift")
            if confirmed:
                if t_confirm is None or anchor["rejection_reason"]:
                    raise EpisodeV3Error(f"{candidate_id}: confirmed anchor drift")
                expected_b_ids.append(candidate_id)
                family_b = next(family_b_iter, None)
                if family_b is None or family_b["candidate_id"] != candidate_id:
                    raise EpisodeV3Error(f"{candidate_id}: Family B linkage drift")
                global_family_b_ids.add(candidate_id)
                counters["family_b_rows"] += 1
            else:
                counters["rejected_rows"] += 1
                if t_confirm is not None:
                    counters["synthetic_rejected_confirm_count"] += 1
                if not anchor["rejection_reason"]:
                    raise EpisodeV3Error(f"{candidate_id}: rejection reason missing")
            if (
                family_a["family_view"] != "family_a"
                or family_a["decision_landmark"] != "t_candidate"
                or family_a["landmark_ts_ns"] != anchor["t_candidate_ns"]
            ):
                raise EpisodeV3Error(f"{candidate_id}: Family A landmark drift")
            counters["candidate_rows"] += 1
            counters["family_a_rows"] += 1
            counters["anchor_rows"] += 1
            counters["outcome_rows"] += 1
            if outcome["t_candidate_ns"] != anchor["t_candidate_ns"]:
                raise EpisodeV3Error(f"{candidate_id}: outcome landmark drift")
            if outcome["t_confirm_ns"] != anchor["t_confirm_ns"]:
                raise EpisodeV3Error(f"{candidate_id}: outcome confirm drift")
            for name in FIRST_EVENT_NAMES:
                status = outcome[f"{name}_status"]
                lower = outcome[f"{name}_interval_lower_ns"]
                upper = outcome[f"{name}_interval_upper_ns"]
                if status == "interval_censored":
                    _assert_canonical_int_text(lower, label=f"{candidate_id}:{name}:lower")
                    _assert_canonical_int_text(upper, label=f"{candidate_id}:{name}:upper")
                    if int(lower) < t_candidate:
                        raise EpisodeV3Error(
                            f"{candidate_id}:{name}: pre-Candidate interval support"
                        )
                    if int(upper) <= int(lower):
                        raise EpisodeV3Error(f"{candidate_id}:{name}: point/empty interval")
                    if not outcome[f"{name}_source_event_id"]:
                        raise EpisodeV3Error(f"{candidate_id}:{name}: source missing")
                    counters["interval_censored_outcome_count"] += 1
                elif status in {"right_censored", "segment_censored"}:
                    if upper or outcome[f"{name}_source_event_id"]:
                        raise EpisodeV3Error(f"{candidate_id}:{name}: censored event populated")
                    _assert_canonical_int_text(
                        lower, label=f"{candidate_id}:{name}:censored_lower"
                    )
                    if int(lower) < t_candidate:
                        raise EpisodeV3Error(
                            f"{candidate_id}:{name}: pre-Candidate censor support"
                        )
                    counters[f"{status}_outcome_count"] += 1
                elif status == "quality_censored":
                    if lower or upper or outcome[f"{name}_source_event_id"]:
                        raise EpisodeV3Error(
                            f"{candidate_id}:{name}: quality censor populated"
                        )
                    counters["quality_censored_outcome_count"] += 1
                else:
                    raise EpisodeV3Error(f"{candidate_id}:{name}: status drift")
            for horizon in OUTCOME_HORIZONS_MS:
                availability = outcome[
                    f"gap_survival_{horizon}ms_availability"
                ]
                value = outcome[f"gap_survival_{horizon}ms"]
                risk = outcome[f"risk_gap_{horizon}ms_bps"]
                if availability == "available":
                    if value not in {"true", "false"} or not risk:
                        raise EpisodeV3Error(
                            f"{candidate_id}: gap outcome availability drift"
                        )
                    _assert_canonical_float_text(
                        risk, label=f"{candidate_id}:risk_gap_{horizon}"
                    )
                elif value or risk:
                    raise EpisodeV3Error(
                        f"{candidate_id}: unavailable gap silently populated"
                    )
            for horizon in MARKOUT_HORIZONS_MS:
                availability = outcome[
                    f"target_midpoint_markout_{horizon}ms_availability"
                ]
                value = outcome[f"target_midpoint_markout_{horizon}ms_bps"]
                if availability == "available":
                    _assert_canonical_float_text(
                        value, label=f"{candidate_id}:markout_{horizon}"
                    )
                elif value:
                    raise EpisodeV3Error(
                        f"{candidate_id}: unavailable markout silently populated"
                    )
            if outcome["public_quote_risk_availability"] != "available":
                if any(
                    outcome[field_name]
                    for field_name in (
                        "public_trade_reaches_quote",
                        "public_bbo_moves_through_quote",
                        "public_quote_survives_horizon",
                        "public_adverse_exposure",
                    )
                ):
                    raise EpisodeV3Error(
                        f"{candidate_id}: unavailable quote risk silently populated"
                    )
        if next(family_b_iter, None) is not None:
            raise EpisodeV3Error(f"{segment_id}: extra Family B row")

        for family_view, expected_ids in (
            ("family_a", list(anchors)),
            ("family_b", expected_b_ids),
        ):
            range_path = (
                output_dir
                / f"paths/sparse_range_index/{family_view}/{segment_id}.csv.gz"
            )
            observed_ids = []
            for row in _strict_csv_rows(range_path, RANGE_FIELDS):
                candidate_id = row["candidate_id"]
                observed_ids.append(candidate_id)
                anchor = anchors.get(candidate_id)
                if anchor is None or row["family_view"] != family_view:
                    raise EpisodeV3Error(f"{candidate_id}: sparse range identity drift")
                if row["connection_epoch_id"] != "0":
                    counters["cross_epoch_path_mismatch_count"] += 1
                if not (
                    int(anchor["segment_start_ts_ns"])
                    <= int(row["window_start_ts_ns"])
                    <= int(row["window_end_ts_ns"])
                    <= int(anchor["segment_end_ts_ns"])
                ):
                    counters["cross_segment_path_mismatch_count"] += 1
                counters[f"sparse_range_{family_view}_rows"] += 1
            if observed_ids != expected_ids:
                raise EpisodeV3Error(f"{segment_id}: {family_view} sparse IDs drift")

            grid_path = (
                output_dir / f"paths/fixed_grid/{family_view}/{segment_id}.csv.gz"
            )
            grid_counts: Counter[str] = Counter()
            for row in _strict_csv_rows(grid_path, GRID_FIELDS):
                candidate_id = row["candidate_id"]
                anchor = anchors.get(candidate_id)
                if anchor is None or row["family_view"] != family_view:
                    raise EpisodeV3Error(f"{candidate_id}: fixed-grid identity drift")
                requested_relative = int(row["requested_relative_ms"])
                if requested_relative not in GRID_MS:
                    raise EpisodeV3Error(f"{candidate_id}: unknown grid point")
                landmark_field = (
                    "t_candidate_ns" if family_view == "family_a" else "t_confirm_ns"
                )
                landmark = int(anchor[landmark_field])
                if (
                    int(row["landmark_ts_ns"]) != landmark
                    or int(row["requested_ts_ns"])
                    != landmark + requested_relative * 1_000_000
                ):
                    raise EpisodeV3Error(f"{candidate_id}: grid time drift")
                grid_counts[candidate_id] += 1
                if row["inside_segment"] == "true" and row["availability_reason"] == "available":
                    for prefix in ("binance", "hyperliquid_bbo", "hyperliquid_fast"):
                        observed = int(row[f"{prefix}_observed_at_ns"])
                        requested = int(row["requested_ts_ns"])
                        if observed > requested:
                            raise EpisodeV3Error(f"{candidate_id}: future grid value")
                        expected_no_new = observed < requested
                        if row[f"{prefix}_no_new_information"] != _bool_text(
                            expected_no_new
                        ):
                            raise EpisodeV3Error(
                                f"{candidate_id}: no-new-information drift"
                            )
                        if int(row[f"{prefix}_source_age_ns"]) != requested - observed:
                            raise EpisodeV3Error(f"{candidate_id}: source-age drift")
                        if not row[f"{prefix}_source_event_id"]:
                            raise EpisodeV3Error(
                                f"{candidate_id}: grid source identity missing"
                            )
                    expected_risk = (
                        row["d_bh_bps"]
                        if anchor["direction_sign"] == "1"
                        else row["d_hb_bps"]
                    )
                    if row["risk_gap_bps"] != expected_risk:
                        raise EpisodeV3Error(f"{candidate_id}: risk-gap direction swap")
                    if row["core_degraded_mask"]:
                        counters["core_degraded_grid_rows"] += 1
                    if row["auxiliary_degraded_mask"]:
                        if segment_id != "segment_0002":
                            raise EpisodeV3Error(
                                f"{candidate_id}: auxiliary mask outside segment_0002"
                            )
                        counters["auxiliary_degraded_grid_rows"] += 1
                elif row["availability_reason"] == "outside_segment":
                    if any(
                        row[field_name]
                        for field_name in (
                            "binance_bid_px",
                            "hyperliquid_bbo_bid_px",
                            "hyperliquid_fast_bid_px",
                            "risk_gap_bps",
                        )
                    ):
                        raise EpisodeV3Error(
                            f"{candidate_id}: outside-segment grid populated"
                        )
                counters[f"fixed_grid_{family_view}_rows"] += 1
            if (
                list(grid_counts) != expected_ids
                or any(count != len(GRID_MS) for count in grid_counts.values())
            ):
                raise EpisodeV3Error(f"{segment_id}: {family_view} grid cardinality drift")

            event_path = (
                output_dir / f"paths/event_count/{family_view}/{segment_id}.csv.gz"
            )
            event_counts: Counter[str] = Counter()
            for row in _strict_csv_rows(event_path, EVENT_COUNT_FIELDS):
                candidate_id = row["candidate_id"]
                anchor = anchors.get(candidate_id)
                if anchor is None or row["family_view"] != family_view:
                    raise EpisodeV3Error(f"{candidate_id}: event-count identity drift")
                landmark_field = (
                    "t_candidate_ns" if family_view == "family_a" else "t_confirm_ns"
                )
                landmark = int(anchor[landmark_field])
                seen_event_ids: list[str] = []
                seen_timestamps: list[int] = []
                for point in EVENT_COUNT_POINTS:
                    prefix = f"event_{point}_"
                    if row[prefix + "availability_reason"] == "available":
                        event_id = row[prefix + "source_event_id"]
                        observed = int(row[prefix + "observed_at_ns"])
                        if not event_id or observed <= landmark:
                            raise EpisodeV3Error(
                                f"{candidate_id}: event-count causal drift"
                            )
                        seen_event_ids.append(event_id)
                        seen_timestamps.append(observed)
                if len(seen_event_ids) != len(set(seen_event_ids)):
                    raise EpisodeV3Error(
                        f"{candidate_id}: forward-filled state counted as event"
                    )
                if seen_timestamps != sorted(seen_timestamps):
                    raise EpisodeV3Error(f"{candidate_id}: event-count ordering drift")
                event_counts[candidate_id] += 1
                counters[f"event_count_{family_view}_rows"] += 1
            if (
                list(event_counts) != expected_ids
                or any(count != 3 for count in event_counts.values())
            ):
                raise EpisodeV3Error(
                    f"{segment_id}: {family_view} event-count cardinality drift"
                )

            feature_path = (
                output_dir / f"features/{family_view}/{segment_id}.csv.gz"
            )
            feature_candidate_counts: Counter[str] = Counter()
            expected_feature_names = _feature_name_universe(family_view)
            current_feature_candidate = ""
            current_feature_names: list[str] = []
            for row in _strict_csv_rows(feature_path, FEATURE_FIELDS):
                candidate_id = row["candidate_id"]
                anchor = anchors.get(candidate_id)
                if anchor is None or row["family_view"] != family_view:
                    raise EpisodeV3Error(f"{candidate_id}: feature identity drift")
                if current_feature_candidate and candidate_id != current_feature_candidate:
                    if tuple(current_feature_names) != expected_feature_names:
                        raise EpisodeV3Error(
                            f"{current_feature_candidate}: feature universe drift"
                        )
                    current_feature_names = []
                current_feature_candidate = candidate_id
                current_feature_names.append(row["feature_name"])
                landmark_field = (
                    "t_candidate_ns" if family_view == "family_a" else "t_confirm_ns"
                )
                landmark = int(anchor[landmark_field])
                feature_names.add(row["feature_name"])
                feature_candidate_counts[candidate_id] += 1
                if row["value"]:
                    if (
                        not row["observed_at_ns"]
                        or not row["source_event_id"]
                        or not row["source_book_version"]
                        or row["availability_reason"] != "available"
                    ):
                        raise EpisodeV3Error(
                            f"{candidate_id}: feature observation ledger incomplete"
                        )
                    if int(row["observed_at_ns"]) > landmark:
                        counters["future_feature_observation_mismatch_count"] += 1
                    if (
                        family_view == "family_b"
                        and row["feature_name"]
                        in CONFIRMATION_FEATURE_NAMES
                        and (
                            int(row["observed_at_ns"]) != landmark
                            or not row["source_event_id"].startswith(
                                f"{SESSION_ID}:{segment_id}:"
                                "common_l2_timeline:"
                            )
                        )
                    ):
                        raise EpisodeV3Error(
                            f"{candidate_id}: confirmation feature provenance drift"
                        )
                else:
                    if (
                        row["observed_at_ns"]
                        or row["source_event_id"]
                        or row["source_book_version"]
                        or row["availability_reason"] in {"", "available"}
                    ):
                        raise EpisodeV3Error(
                            f"{candidate_id}: unavailable feature zero/source fill"
                        )
                    counters["feature_unavailable_rows"] += 1
                counters[f"feature_ledger_{family_view}_rows"] += 1
            if current_feature_candidate and tuple(current_feature_names) != (
                expected_feature_names
            ):
                raise EpisodeV3Error(
                    f"{current_feature_candidate}: feature universe drift"
                )
            if list(feature_candidate_counts) != expected_ids:
                raise EpisodeV3Error(
                    f"{segment_id}: {family_view} feature candidate coverage drift"
                )
        counters["epoch_censored_outcome_count"] = 0
        counters["point_coerced_outcome_count"] = 0
        summary = {
            field_name: (
                segment_id
                if field_name == "segment_id"
                else EVIDENCE_LABELS[segment_id]
                if field_name == "evidence_label"
                else str(counters[field_name])
            )
            for field_name in SEGMENT_SUMMARY_FIELDS
        }
        summaries.append(summary)
        for field_name in SEGMENT_SUMMARY_FIELDS[2:]:
            aggregate[field_name] += int(summary[field_name])
    if len(global_ids) != EXPECTED_COUNTS["family_a"]:
        raise EpisodeV3Error("global Family A candidate coverage drift")
    if len(global_family_b_ids) != EXPECTED_COUNTS["family_b"]:
        raise EpisodeV3Error("global Family B candidate coverage drift")
    published = list(
        _strict_csv_rows(output_dir / "segment_summary.csv", SEGMENT_SUMMARY_FIELDS)
    )
    if published != summaries:
        raise EpisodeV3Error("segment summary canonical value drift")
    if aggregate["auxiliary_degraded_grid_rows"] <= 0:
        raise EpisodeV3Error("auxiliary degraded grid evidence was deleted")
    return summaries, aggregate, feature_names


def _verify_source_semantics(
    output_dir: Path,
    *,
    specs: Sequence[SegmentSpec],
    quality_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    spec_by_segment = {spec.segment_id: spec for spec in specs}
    results: dict[str, Any] = {}
    observed_segments: list[str] = []
    topology_fingerprint = _topology_fingerprint()
    aggregate_contract = _source_semantic_aggregate_contract()
    projection_contracts = aggregate_contract["projections"]
    projection_names = tuple(projection_contracts)
    aggregate_counts: Counter[str] = Counter()
    aggregate_digests = {
        name: hashlib.sha256() for name in projection_names
    }
    for segment_id, candidates in _group_candidates_by_segment():
        observed_segments.append(segment_id)
        spec = spec_by_segment.get(segment_id)
        if spec is None:
            raise EpisodeV3Error(
                f"{segment_id}: source-semantic candidate segment drift"
            )
        data = _load_segment_data(spec, quality_rows, candidates)

        def expected_anchors() -> Iterator[dict[str, str]]:
            for candidate in candidates:
                yield _anchor_row(
                    candidate,
                    r0_manifest_sha256=EXPECTED_R0_MANIFEST_SHA256,
                    auxiliary_ids=_auxiliary_degraded_ids(
                        data, candidate.shock_ts_ns
                    ),
                )

        def expected_views_family_a() -> Iterator[dict[str, str]]:
            for candidate in candidates:
                yield _view_row(
                    candidate,
                    family_view="family_a",
                    decision_landmark="t_candidate",
                    landmark_ts_ns=candidate.shock_ts_ns,
                )

        def expected_views_family_b() -> Iterator[dict[str, str]]:
            for candidate in candidates:
                if not candidate.family_b_available:
                    continue
                if candidate.t_confirm_ns is None:
                    raise EpisodeV3Error(
                        f"{candidate.candidate_id}: source-semantic confirm missing"
                    )
                yield _view_row(
                    candidate,
                    family_view="family_b",
                    decision_landmark="t_confirm",
                    landmark_ts_ns=candidate.t_confirm_ns,
                )

        def expected_features_family_a() -> Iterator[dict[str, str]]:
            for candidate in candidates:
                yield from _feature_rows(
                    candidate,
                    data,
                    family_view="family_a",
                    decision_landmark="t_candidate",
                    landmark_ts_ns=candidate.shock_ts_ns,
                    topology_fingerprint=topology_fingerprint,
                )

        def expected_features_family_b() -> Iterator[dict[str, str]]:
            for candidate in candidates:
                if not candidate.family_b_available:
                    continue
                if candidate.t_confirm_ns is None:
                    raise EpisodeV3Error(
                        f"{candidate.candidate_id}: source-semantic confirm missing"
                    )
                yield from _feature_rows(
                    candidate,
                    data,
                    family_view="family_b",
                    decision_landmark="t_confirm",
                    landmark_ts_ns=candidate.t_confirm_ns,
                    topology_fingerprint=topology_fingerprint,
                )

        def expected_sparse_family_a() -> Iterator[dict[str, str]]:
            for candidate in candidates:
                yield _range_row(
                    candidate,
                    data,
                    family_view="family_a",
                    landmark_ts_ns=candidate.shock_ts_ns,
                )

        def expected_sparse_family_b() -> Iterator[dict[str, str]]:
            for candidate in candidates:
                if not candidate.family_b_available:
                    continue
                if candidate.t_confirm_ns is None:
                    raise EpisodeV3Error(
                        f"{candidate.candidate_id}: source-semantic confirm missing"
                    )
                yield _range_row(
                    candidate,
                    data,
                    family_view="family_b",
                    landmark_ts_ns=candidate.t_confirm_ns,
                )

        def expected_grid_family_a() -> Iterator[dict[str, str]]:
            for candidate in candidates:
                for relative_ms in GRID_MS:
                    yield _grid_row(
                        candidate,
                        data,
                        family_view="family_a",
                        decision_landmark="t_candidate",
                        landmark_ts_ns=candidate.shock_ts_ns,
                        relative_ms=relative_ms,
                    )

        def expected_grid_family_b() -> Iterator[dict[str, str]]:
            for candidate in candidates:
                if not candidate.family_b_available:
                    continue
                if candidate.t_confirm_ns is None:
                    raise EpisodeV3Error(
                        f"{candidate.candidate_id}: source-semantic confirm missing"
                    )
                for relative_ms in GRID_MS:
                    yield _grid_row(
                        candidate,
                        data,
                        family_view="family_b",
                        decision_landmark="t_confirm",
                        landmark_ts_ns=candidate.t_confirm_ns,
                        relative_ms=relative_ms,
                    )

        def expected_event_count_family_a() -> Iterator[dict[str, str]]:
            for candidate in candidates:
                yield from _event_count_rows(
                    candidate,
                    data,
                    family_view="family_a",
                    decision_landmark="t_candidate",
                    landmark_ts_ns=candidate.shock_ts_ns,
                )

        def expected_event_count_family_b() -> Iterator[dict[str, str]]:
            for candidate in candidates:
                if not candidate.family_b_available:
                    continue
                if candidate.t_confirm_ns is None:
                    raise EpisodeV3Error(
                        f"{candidate.candidate_id}: source-semantic confirm missing"
                    )
                yield from _event_count_rows(
                    candidate,
                    data,
                    family_view="family_b",
                    decision_landmark="t_confirm",
                    landmark_ts_ns=candidate.t_confirm_ns,
                )

        def expected_outcomes() -> Iterator[dict[str, str]]:
            for candidate in candidates:
                outcome, _counts = _outcome_row(candidate, data)
                yield outcome

        projection_specs = (
            (
                "anchors",
                expected_anchors(),
                f"anchors/{segment_id}.csv.gz",
                ANCHOR_FIELDS,
            ),
            (
                "views_family_a",
                expected_views_family_a(),
                f"views/family_a/{segment_id}.csv.gz",
                VIEW_FIELDS,
            ),
            (
                "views_family_b",
                expected_views_family_b(),
                f"views/family_b/{segment_id}.csv.gz",
                VIEW_FIELDS,
            ),
            (
                "features_family_a",
                expected_features_family_a(),
                f"features/family_a/{segment_id}.csv.gz",
                FEATURE_FIELDS,
            ),
            (
                "features_family_b",
                expected_features_family_b(),
                f"features/family_b/{segment_id}.csv.gz",
                FEATURE_FIELDS,
            ),
            (
                "sparse_range_family_a",
                expected_sparse_family_a(),
                f"paths/sparse_range_index/family_a/{segment_id}.csv.gz",
                RANGE_FIELDS,
            ),
            (
                "sparse_range_family_b",
                expected_sparse_family_b(),
                f"paths/sparse_range_index/family_b/{segment_id}.csv.gz",
                RANGE_FIELDS,
            ),
            (
                "fixed_grid_family_a",
                expected_grid_family_a(),
                f"paths/fixed_grid/family_a/{segment_id}.csv.gz",
                GRID_FIELDS,
            ),
            (
                "fixed_grid_family_b",
                expected_grid_family_b(),
                f"paths/fixed_grid/family_b/{segment_id}.csv.gz",
                GRID_FIELDS,
            ),
            (
                "event_count_family_a",
                expected_event_count_family_a(),
                f"paths/event_count/family_a/{segment_id}.csv.gz",
                EVENT_COUNT_FIELDS,
            ),
            (
                "event_count_family_b",
                expected_event_count_family_b(),
                f"paths/event_count/family_b/{segment_id}.csv.gz",
                EVENT_COUNT_FIELDS,
            ),
            (
                "outcomes",
                expected_outcomes(),
                f"outcomes/{segment_id}.csv.gz",
                OUTCOME_FIELDS,
            ),
        )
        if tuple(name for name, *_rest in projection_specs) != projection_names:
            raise EpisodeV3Error(
                "source-semantic projection implementation/contract drift"
            )
        segment_result: dict[str, Any] = {}
        for name, expected_rows, relative_path, fields in projection_specs:
            projection_result = _verify_exact_row_stream(
                label=f"{segment_id}: {name}",
                expected_rows=expected_rows,
                observed_rows=_strict_csv_rows(
                    output_dir / relative_path, fields
                ),
                fields=fields,
                aggregate_digest=aggregate_digests[name],
            )
            aggregate_counts[name] += projection_result["expected_rows"]
            segment_result[name] = projection_result
        results[segment_id] = segment_result

    if tuple(observed_segments) != SEGMENT_IDS:
        raise EpisodeV3Error("source-semantic segment order drift")
    exact_global_counts = {
        name: projection_contract["expected_rows"]
        for name, projection_contract in projection_contracts.items()
    }
    if dict(aggregate_counts) != exact_global_counts:
        raise EpisodeV3Error(
            "source-semantic exact projection cardinality drift "
            f"expected={exact_global_counts} observed={dict(aggregate_counts)}"
        )
    results["aggregate"] = {
        name: {
            "expected_rows": exact_global_counts[name],
            "observed_rows": exact_global_counts[name],
            "expected_sha256": aggregate_digests[name].hexdigest(),
            "observed_sha256": aggregate_digests[name].hexdigest(),
            "mismatch_rows": 0,
            "fields": projection_contracts[name]["fields"],
        }
        for name in projection_names
    }
    return results


def verify_package(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Mapping[str, Any]:
    output_dir = Path(os.path.abspath(output_dir))
    manifest = _verify_artifact_closure(output_dir)
    dependencies = _verify_dependency_packages()
    if dependencies != manifest["dependencies"]:
        raise EpisodeV3Error("manifest/actual dependency identity drift")
    _verify_runtime_archive(output_dir, manifest)
    _verify_input_bindings(output_dir, manifest)
    _verify_gzip_headers(output_dir)
    _r0, _alignment, specs = _load_segment_specs()
    for spec in specs:
        _validate_decision_labels(spec)
    quality_rows = _load_quality_intervals()
    published_quality = list(
        _strict_csv_rows(output_dir / "quality_intervals.csv", QUALITY_INTERVAL_FIELDS)
    )
    if published_quality != quality_rows:
        raise EpisodeV3Error("quality interval publication drift")
    source_catalog_rows = _source_catalog_rows(specs)
    published_catalog = list(
        _strict_csv_rows(
            output_dir / "source_event_store_catalog.csv", SOURCE_CATALOG_FIELDS
        )
    )
    if published_catalog != source_catalog_rows:
        raise EpisodeV3Error("source event-store catalog drift")
    source_semantic_results = _verify_source_semantics(
        output_dir,
        specs=specs,
        quality_rows=quality_rows,
    )
    summaries, aggregate, feature_names = _verify_segment_outputs(
        output_dir, manifest
    )
    if sorted(feature_names) != manifest["feature_names"]:
        raise EpisodeV3Error("manifest feature-name universe drift")
    contract = _read_canonical_json(
        output_dir / "frozen_episode_v3_contract.json",
        label="frozen Episode v3 contract",
    )
    expected_contract = _canonical_contract(
        feature_names=sorted(feature_names),
        source_catalog_rows=source_catalog_rows,
        dependency_identities=dependencies,
        input_inventory_sha256=manifest["input_inventory_sha256_before"],
    )
    if contract != expected_contract:
        raise EpisodeV3Error("frozen Episode v3 contract drift")
    if sha256_file(output_dir / "frozen_episode_v3_contract.json") != manifest[
        "contract_sha256"
    ]:
        raise EpisodeV3Error("frozen Episode v3 contract SHA drift")
    expected_exact = _expected_manifest_exact_counts()
    if manifest["exact_counts"] != expected_exact:
        raise EpisodeV3Error("manifest exact count drift")
    if manifest["aggregate_output_counts"] != dict(sorted(aggregate.items())):
        raise EpisodeV3Error("manifest aggregate output count drift")
    if (
        manifest["source_catalog_rows"] != len(source_catalog_rows)
        or manifest["quality_interval_rows"] != len(quality_rows)
    ):
        raise EpisodeV3Error("manifest catalog/quality count drift")
    report = (output_dir / "reports/jul30_episode_v3.md").read_text(
        encoding="utf-8"
    )
    if report != _canonical_report(
        summaries, aggregate, manifest["input_inventory_sha256_before"]
    ):
        raise EpisodeV3Error("canonical Episode v3 report drift")
    all_research_fields = (
        ANCHOR_FIELDS
        + VIEW_FIELDS
        + RANGE_FIELDS
        + GRID_FIELDS
        + EVENT_COUNT_FIELDS
        + FEATURE_FIELDS
        + OUTCOME_FIELDS
    )
    for field_name in all_research_fields:
        lowered = field_name.lower()
        if any(token in lowered for token in FORBIDDEN_FIELD_TOKENS):
            raise EpisodeV3Error(f"forbidden research field: {field_name}")
    verified = dict(manifest)
    verified["_source_semantic_verification"] = {
        "source_semantic_verified": True,
        "scope": _source_semantic_projection_contract(),
        "aggregate": source_semantic_results["aggregate"],
    }
    return verified


def compare_packages(left: Path, right: Path) -> dict[str, Any]:
    verify_package(left)
    verify_package(right)
    left_inventory = _directory_inventory(left)
    right_inventory = _directory_inventory(right)
    return {
        "identical": left_inventory == right_inventory,
        "file_count": len(left_inventory),
        "total_bytes": sum(int(row["bytes"]) for row in left_inventory),
        "full_inventory_sha256": canonical_json_sha256(left_inventory),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--compare-to")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    try:
        manifest = (
            verify_package(output_dir)
            if args.verify_only
            else build_package(output_dir)
        )
        inventory = _directory_inventory(output_dir.resolve())
        result: dict[str, Any] = {
            "verified": True,
            "artifact_count": len(manifest["artifacts"]),
            "file_count": len(inventory),
            "total_bytes": sum(int(row["bytes"]) for row in inventory),
            "core_package_sha256": manifest["core_package_sha256"],
            "full_inventory_sha256": canonical_json_sha256(inventory),
            "family_a_rows": manifest["exact_counts"]["family_a_rows"],
            "family_b_rows": manifest["exact_counts"]["family_b_rows"],
            "rejected_rows": manifest["exact_counts"]["rejected_rows"],
        }
        if args.compare_to:
            result["comparison"] = compare_packages(
                output_dir, Path(args.compare_to)
            )
    except (EpisodeV3Error, OSError, KeyError, ValueError) as exc:
        print(
            json.dumps(
                {"verified": False, "error": str(exc)}, indent=2, sort_keys=True
            )
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
