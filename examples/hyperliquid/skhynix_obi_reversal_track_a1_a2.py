#!/usr/bin/env python3
"""A1 state ledger and A2 first-passage targets for OBI_REVERSAL_V1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from examples.hyperliquid.skhynix_obi_reversal_track_a0 import (
        CONTROL_OBI_BIN_WIDTH,
        GRID_MS,
        PRIMARY_THRESHOLD,
        CaptureBinding,
        discover_bindings,
    )
except ModuleNotFoundError:
    from skhynix_obi_reversal_track_a0 import (  # type: ignore[no-redef]
        CONTROL_OBI_BIN_WIDTH,
        GRID_MS,
        PRIMARY_THRESHOLD,
        CaptureBinding,
        discover_bindings,
    )


TASK_ID = "0828T005"
SCHEMA_VERSION = "skhynix_obi_reversal_track_a1_a2_v1"
A0_SCHEMA_VERSION = "skhynix_obi_reversal_track_a0_v1"
TAU_MAX_MS = 120_000
BARRIER_TICKS = 1.0
TAU_STEPS = TAU_MAX_MS // GRID_MS

MIN_CAUSE_EVENTS = 200
MIN_CAUSE_DATES = 8
MAX_CAUSE_DATE_SHARE = 0.35
MAX_CENSORING_FRACTION = 0.50
MAX_AMBIGUITY_FRACTION = 0.01

DEFAULT_FEATURE_ROOT = Path(
    "local_live_analysis/skhynix_phase_alignment_track_a_0827T004"
)
DEFAULT_A0_ROOT = Path(
    "local_live_analysis/skhynix_obi_reversal_track_a0_0828T004"
)
DEFAULT_OUTPUT = Path(
    "local_live_analysis/skhynix_obi_reversal_track_a1_a2_0828T005"
)

QTY_FIELDS = tuple(
    [f"bid_qty_log_l{level}" for level in range(1, 6)]
    + [f"ask_qty_log_l{level}" for level in range(1, 6)]
)
ADD_CANCEL_FIELDS = tuple(
    [f"bid_add_log_l{level}" for level in range(1, 6)]
    + [f"bid_cancel_log_l{level}" for level in range(1, 6)]
    + [f"ask_add_log_l{level}" for level in range(1, 6)]
    + [f"ask_cancel_log_l{level}" for level in range(1, 6)]
)
DECISION_FIELDS = (
    *QTY_FIELDS,
    *ADD_CANCEL_FIELDS,
    "trade_buy_qty_log",
    "trade_sell_qty_log",
    "spread_ticks",
    "midpoint_delta_ticks",
    "bid_depth_concentration",
    "ask_depth_concentration",
    "source_age_ms_log",
    "no_new_information",
    "depth_update_count_log",
)
FUTURE_TARGET_FIELD = "midpoint_delta_ticks"


class A1A2Error(RuntimeError):
    """Fail-closed A1/A2 materialization error."""


@dataclass(frozen=True)
class CaptureData:
    binding: CaptureBinding
    ts_ns: np.ndarray
    valid: np.ndarray
    feature_names: tuple[str, ...]
    base_features: np.ndarray

    @property
    def feature_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.feature_names)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_artifact_manifest(root: Path, manifest_name: str) -> dict[str, Any]:
    path = root / manifest_name
    payload = json.loads(path.read_text(encoding="utf-8"))
    bad: list[str] = []
    for item in payload["artifacts"]:
        artifact = root / item["path"]
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(item["size_bytes"])
            or _sha256(artifact) != item["sha256"]
        ):
            bad.append(item["path"])
    if bad:
        raise A1A2Error(f"artifact_manifest_mismatch:{bad}")
    return payload


def verify_a0_contracts(a0_root: Path) -> dict[str, Any]:
    manifest = verify_artifact_manifest(a0_root, "run_manifest.json")
    classification = json.loads(
        (a0_root / "classification.json").read_text(encoding="utf-8")
    )
    target = json.loads(
        (a0_root / "contracts/target_contract.json").read_text(encoding="utf-8")
    )
    state = json.loads(
        (a0_root / "contracts/reversal_state_machine.json").read_text(
            encoding="utf-8"
        )
    )
    control = json.loads(
        (a0_root / "contracts/control_entry_contract.json").read_text(
            encoding="utf-8"
        )
    )
    if classification["status"] != "passed":
        raise A1A2Error("a0_not_passed")
    if not all(classification["gates"].values()):
        raise A1A2Error("a0_gate_not_closed")
    expected = {
        "schema_version": A0_SCHEMA_VERSION,
        "quote_convention": "midpoint",
        "barrier_ticks": BARRIER_TICKS,
        "tau_max_ms": TAU_MAX_MS,
        "clock_start": "decision_timestamp",
        "simultaneous_hit": "interval_ambiguous_censor",
        "post_barrier_return_or_markout_allowed": False,
    }
    for key, value in expected.items():
        if target.get(key) != value:
            raise A1A2Error(f"frozen_target_contract_mismatch:{key}")
    if target.get("materialized") is not False:
        raise A1A2Error("a0_target_already_materialized")
    if state.get("old_band_abs_OBI") != PRIMARY_THRESHOLD:
        raise A1A2Error("frozen_primary_threshold_mismatch")
    if state.get("primary_alignment") != "reversal_detected_at":
        raise A1A2Error("frozen_alignment_mismatch")
    if control.get("primary_model_uses_full_common_risk_set") is not True:
        raise A1A2Error("frozen_common_risk_set_mismatch")
    return {
        "a0_manifest_sha256": _sha256(a0_root / "run_manifest.json"),
        "a0_artifact_count": manifest["artifact_count"],
        "a0_classification": classification["classification"],
        "target_contract_sha256": _sha256(
            a0_root / "contracts/target_contract.json"
        ),
        "state_machine_contract_sha256": _sha256(
            a0_root / "contracts/reversal_state_machine.json"
        ),
        "control_contract_sha256": _sha256(
            a0_root / "contracts/control_entry_contract.json"
        ),
    }


def load_capture_data(binding: CaptureBinding) -> CaptureData:
    with np.load(binding.cache_path, allow_pickle=False) as payload:
        names = tuple(str(value) for value in payload["feature_names"])
        missing = [name for name in DECISION_FIELDS if name not in names]
        if missing:
            raise A1A2Error(
                f"missing_materialization_fields:{binding.capture_id}:{missing}"
            )
        if str(payload["capture_id"]) != binding.capture_id:
            raise A1A2Error(f"capture_identity_mismatch:{binding.capture_id}")
        if str(payload["research_date"]) != binding.research_date:
            raise A1A2Error(f"date_identity_mismatch:{binding.capture_id}")
        if str(payload["role"]) != binding.role:
            raise A1A2Error(f"role_identity_mismatch:{binding.capture_id}")
        ts_ns = np.asarray(payload["ts_ns"], dtype=np.int64)
        valid = np.asarray(payload["valid"], dtype=np.bool_)
        features = np.asarray(payload["base_features"], dtype=np.float64)
    if len(ts_ns) != len(valid) or len(ts_ns) != len(features):
        raise A1A2Error(f"cache_length_mismatch:{binding.capture_id}")
    if len(ts_ns) and not np.all(np.diff(ts_ns) == GRID_MS * 1_000_000):
        raise A1A2Error(f"nonuniform_grid:{binding.capture_id}")
    return CaptureData(
        binding=binding,
        ts_ns=ts_ns,
        valid=valid,
        feature_names=names,
        base_features=features,
    )


def _feature(data: CaptureData, name: str) -> np.ndarray:
    return data.base_features[:, data.feature_index[name]]


def _quantities(data: CaptureData, index: int) -> tuple[np.ndarray, np.ndarray]:
    mapping = data.feature_index
    bid = np.expm1(
        data.base_features[
            index,
            [mapping[f"bid_qty_log_l{level}"] for level in range(1, 6)],
        ]
    )
    ask = np.expm1(
        data.base_features[
            index,
            [mapping[f"ask_qty_log_l{level}"] for level in range(1, 6)],
        ]
    )
    return bid, ask


def _normalized_difference(positive: float, negative: float) -> float:
    denominator = positive + negative
    return (positive - negative) / denominator if denominator > 0 else 0.0


def _window_sum(values: np.ndarray, index: int, steps: int) -> float:
    start = max(0, index - steps + 1)
    window = values[start : index + 1]
    return float(np.sum(window)) if np.all(np.isfinite(window)) else math.nan


def extract_decision_features(data: CaptureData, index: int) -> dict[str, float]:
    if index < 0 or index >= len(data.ts_ns) or not data.valid[index]:
        raise A1A2Error(f"invalid_decision_index:{data.binding.capture_id}:{index}")
    bid, ask = _quantities(data, index)
    total_bid = float(np.sum(bid))
    total_ask = float(np.sum(ask))
    total_depth = total_bid + total_ask
    per_level = np.divide(
        bid - ask,
        bid + ask,
        out=np.zeros_like(bid),
        where=(bid + ask) > 0,
    )
    midpoint_delta = _feature(data, "midpoint_delta_ticks")

    mapping = data.feature_index
    bid_add = np.expm1(
        data.base_features[
            index,
            [mapping[f"bid_add_log_l{level}"] for level in range(1, 6)],
        ]
    )
    bid_cancel = np.expm1(
        data.base_features[
            index,
            [mapping[f"bid_cancel_log_l{level}"] for level in range(1, 6)],
        ]
    )
    ask_add = np.expm1(
        data.base_features[
            index,
            [mapping[f"ask_add_log_l{level}"] for level in range(1, 6)],
        ]
    )
    ask_cancel = np.expm1(
        data.base_features[
            index,
            [mapping[f"ask_cancel_log_l{level}"] for level in range(1, 6)],
        ]
    )
    positive_book_flow = float(np.sum(bid_add) + np.sum(ask_cancel))
    negative_book_flow = float(np.sum(ask_add) + np.sum(bid_cancel))
    trade_buy = float(
        np.expm1(data.base_features[index, mapping["trade_buy_qty_log"]])
    )
    trade_sell = float(
        np.expm1(data.base_features[index, mapping["trade_sell_qty_log"]])
    )
    seconds = int(data.ts_ns[index] // 1_000_000_000) % 86_400
    angle = 2 * math.pi * seconds / 86_400

    values = {
        "current_obi": _normalized_difference(total_bid, total_ask),
        "current_l1_obi": float(per_level[0]),
        "current_l1_l3_obi": _normalized_difference(
            float(np.sum(bid[:3])),
            float(np.sum(ask[:3])),
        ),
        "level_imbalance_dispersion": float(np.std(per_level)),
        "log_total_depth": math.log1p(total_depth),
        "bid_depth_concentration": float(
            data.base_features[index, mapping["bid_depth_concentration"]]
        ),
        "ask_depth_concentration": float(
            data.base_features[index, mapping["ask_depth_concentration"]]
        ),
        "spread_ticks": float(
            data.base_features[index, mapping["spread_ticks"]]
        ),
        "signed_book_flow": _normalized_difference(
            positive_book_flow,
            negative_book_flow,
        ),
        "signed_trade_flow": _normalized_difference(trade_buy, trade_sell),
        "trailing_midpoint_1s_ticks": _window_sum(midpoint_delta, index, 10),
        "trailing_midpoint_5s_ticks": _window_sum(midpoint_delta, index, 50),
        "trailing_abs_midpoint_5s_ticks": _window_sum(
            np.abs(midpoint_delta),
            index,
            50,
        ),
        "source_age_ms": float(
            np.expm1(data.base_features[index, mapping["source_age_ms_log"]])
        ),
        "no_new_information": float(
            data.base_features[index, mapping["no_new_information"]]
        ),
        "depth_update_count": float(
            np.expm1(
                data.base_features[index, mapping["depth_update_count_log"]]
            )
        ),
        "utc_time_sin": math.sin(angle),
        "utc_time_cos": math.cos(angle),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise A1A2Error(
            f"nonfinite_decision_feature:{data.binding.capture_id}:{index}"
        )
    return values


def _find_index(data: CaptureData, ts_ns: int) -> int:
    index = int(np.searchsorted(data.ts_ns, ts_ns))
    if index >= len(data.ts_ns) or int(data.ts_ns[index]) != ts_ns:
        raise A1A2Error(
            f"decision_timestamp_not_on_grid:{data.binding.capture_id}:{ts_ns}"
        )
    return index


def _obi_bin(value: float) -> int:
    return int(
        math.floor(
            max(abs(value) - PRIMARY_THRESHOLD, 0.0)
            / CONTROL_OBI_BIN_WIDTH
        )
    )


def build_state_entries(
    captures: dict[str, CaptureData],
    reversal_rows: Sequence[dict[str, str]],
    control_rows: Sequence[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reversal_keys = {
        (
            row["research_date"],
            int(row["side"]),
            _obi_bin(float(row["obi_at_detection"])),
        )
        for row in reversal_rows
    }
    control_keys = {
        (
            row["research_date"],
            int(row["side"]),
            int(row["obi_bin"]),
        )
        for row in control_rows
    }
    common_keys = reversal_keys & control_keys
    entries: list[dict[str, Any]] = []

    def append_entry(
        row: dict[str, str],
        entry_type: str,
        ordinal: int,
    ) -> None:
        capture = captures[row["capture_id"]]
        detect_ts_ns = int(row["detect_ts_ns"])
        index = _find_index(capture, detect_ts_ns)
        side = int(row["side"])
        declared_obi = float(row["obi_at_detection"])
        obi_bin = (
            int(row["obi_bin"])
            if entry_type == "control"
            else _obi_bin(declared_obi)
        )
        key = (row["research_date"], side, obi_bin)
        features = extract_decision_features(capture, index)
        if not math.isclose(
            features["current_obi"],
            declared_obi,
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise A1A2Error(
                f"frozen_obi_mismatch:{row['capture_id']}:{detect_ts_ns}"
            )
        prefix = "rev" if entry_type == "reversal" else "ctl"
        entries.append(
            {
                "entry_id": f"{prefix}_{ordinal:06d}",
                "entry_type": entry_type,
                "reversal_indicator_R": int(entry_type == "reversal"),
                "capture_id": row["capture_id"],
                "research_date": row["research_date"],
                "role": row["role"],
                "side": side,
                "decision_index": index,
                "decision_ts_ns": detect_ts_ns,
                "cross_ts_ns": (
                    int(row["cross_ts_ns"])
                    if entry_type == "reversal"
                    else ""
                ),
                "detection_delay_ms": (
                    int(row["detection_delay_ms"])
                    if entry_type == "reversal"
                    else ""
                ),
                "obi_bin": obi_bin,
                "risk_set_key": f"{row['research_date']}|{side}|{obi_bin}",
                "primary_common_support": str(key in common_keys).lower(),
                **features,
            }
        )

    for ordinal, row in enumerate(reversal_rows):
        append_entry(row, "reversal", ordinal)
    for ordinal, row in enumerate(control_rows):
        append_entry(row, "control", ordinal)

    support_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int, int], Counter[str]] = defaultdict(Counter)
    for entry in entries:
        key = (
            entry["research_date"],
            int(entry["side"]),
            int(entry["obi_bin"]),
        )
        grouped[key][entry["entry_type"]] += 1
    for key in sorted(grouped):
        counts = grouped[key]
        support_rows.append(
            {
                "research_date": key[0],
                "side": key[1],
                "obi_bin": key[2],
                "reversal_count": counts["reversal"],
                "control_count": counts["control"],
                "common_support": str(key in common_keys).lower(),
            }
        )
    return entries, support_rows


def first_passage(
    midpoint_delta_ticks: np.ndarray,
    valid: np.ndarray,
    start_index: int,
    side: int,
    *,
    tau_steps: int = TAU_STEPS,
    barrier_ticks: float = BARRIER_TICKS,
) -> dict[str, Any]:
    if side not in (-1, 1):
        raise A1A2Error(f"invalid_side:{side}")
    cumulative = 0.0
    available = min(tau_steps, len(midpoint_delta_ticks) - start_index - 1)
    for step in range(1, available + 1):
        index = start_index + step
        if not valid[index] or not math.isfinite(float(midpoint_delta_ticks[index])):
            return {
                "event_type": "censored",
                "cause_code": 0,
                "event_time_ms": step * GRID_MS,
                "censor_reason": "quality_failure",
                "oriented_displacement_at_exit_ticks": cumulative * side,
                "interval_ambiguous": False,
            }
        cumulative += float(midpoint_delta_ticks[index])
        oriented = cumulative * side
        follow = oriented >= barrier_ticks
        fail = oriented <= -barrier_ticks
        if follow and fail:
            return {
                "event_type": "censored",
                "cause_code": 0,
                "event_time_ms": step * GRID_MS,
                "censor_reason": "interval_ambiguous",
                "oriented_displacement_at_exit_ticks": oriented,
                "interval_ambiguous": True,
            }
        if follow:
            return {
                "event_type": "follow",
                "cause_code": 1,
                "event_time_ms": step * GRID_MS,
                "censor_reason": "",
                "oriented_displacement_at_exit_ticks": oriented,
                "interval_ambiguous": False,
            }
        if fail:
            return {
                "event_type": "fail",
                "cause_code": 2,
                "event_time_ms": step * GRID_MS,
                "censor_reason": "",
                "oriented_displacement_at_exit_ticks": oriented,
                "interval_ambiguous": False,
            }
    reason = "tau_max" if available == tau_steps else "capture_end"
    return {
        "event_type": "censored",
        "cause_code": 0,
        "event_time_ms": available * GRID_MS,
        "censor_reason": reason,
        "oriented_displacement_at_exit_ticks": cumulative * side,
        "interval_ambiguous": False,
    }


def pre_detection_transition(
    midpoint_delta_ticks: np.ndarray,
    valid: np.ndarray,
    cross_index: int,
    detect_index: int,
    side: int,
) -> dict[str, Any]:
    if detect_index <= cross_index:
        raise A1A2Error("nonpositive_cross_detection_interval")
    interval_valid = valid[cross_index + 1 : detect_index + 1]
    interval_delta = midpoint_delta_ticks[cross_index + 1 : detect_index + 1]
    oriented_at_detection = (
        float(np.sum(interval_delta)) * side
        if np.all(interval_valid) and np.all(np.isfinite(interval_delta))
        else math.nan
    )
    result = first_passage(
        midpoint_delta_ticks[: detect_index + 1],
        valid[: detect_index + 1],
        cross_index,
        side,
        tau_steps=detect_index - cross_index,
    )
    return {
        "pre_detection_transition": (
            result["event_type"]
            if result["event_type"] in {"follow", "fail"}
            else "none"
        ),
        "pre_detection_transition_time_ms": (
            result["event_time_ms"]
            if result["event_type"] in {"follow", "fail"}
            else ""
        ),
        "pre_detection_censor_reason": (
            result["censor_reason"]
            if result["censor_reason"] == "quality_failure"
            else ""
        ),
        "pre_detection_oriented_displacement_at_detection_ticks": (
            oriented_at_detection
        ),
    }


def materialize_targets(
    entries: Sequence[dict[str, Any]],
    captures: dict[str, CaptureData],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_rows: list[dict[str, Any]] = []
    pre_detection_rows: list[dict[str, Any]] = []
    for entry in entries:
        if entry["primary_common_support"] != "true":
            continue
        capture = captures[entry["capture_id"]]
        midpoint_delta = _feature(capture, FUTURE_TARGET_FIELD)
        result = first_passage(
            midpoint_delta,
            capture.valid,
            int(entry["decision_index"]),
            int(entry["side"]),
        )
        target_rows.append(
            {
                "entry_id": entry["entry_id"],
                "entry_type": entry["entry_type"],
                "reversal_indicator_R": entry["reversal_indicator_R"],
                "capture_id": entry["capture_id"],
                "research_date": entry["research_date"],
                "role": entry["role"],
                "side": entry["side"],
                "decision_ts_ns": entry["decision_ts_ns"],
                "event_type": result["event_type"],
                "cause_code": result["cause_code"],
                "event_time_ms": result["event_time_ms"],
                "censor_reason": result["censor_reason"],
                "interval_ambiguous": str(
                    result["interval_ambiguous"]
                ).lower(),
            }
        )
        if entry["entry_type"] == "reversal":
            cross_index = _find_index(capture, int(entry["cross_ts_ns"]))
            diagnostic = pre_detection_transition(
                midpoint_delta,
                capture.valid,
                cross_index,
                int(entry["decision_index"]),
                int(entry["side"]),
            )
            pre_detection_rows.append(
                {
                    "entry_id": entry["entry_id"],
                    "capture_id": entry["capture_id"],
                    "research_date": entry["research_date"],
                    "role": entry["role"],
                    "side": entry["side"],
                    "cross_ts_ns": entry["cross_ts_ns"],
                    "decision_ts_ns": entry["decision_ts_ns"],
                    "detection_delay_ms": entry["detection_delay_ms"],
                    **diagnostic,
                }
            )
    return target_rows, pre_detection_rows


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def target_diagnostics(
    target_rows: Sequence[dict[str, Any]],
    pre_detection_rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    event_counts = Counter(row["event_type"] for row in target_rows)
    type_events: dict[str, Counter[str]] = defaultdict(Counter)
    by_date: dict[str, Counter[str]] = defaultdict(Counter)
    censor_counts = Counter(
        row["censor_reason"]
        for row in target_rows
        if row["event_type"] == "censored"
    )
    for row in target_rows:
        type_events[row["entry_type"]][row["event_type"]] += 1
        by_date[row["research_date"]][row["event_type"]] += 1

    variation_rows = []
    for date in sorted(by_date):
        counts = by_date[date]
        variation_rows.append(
            {
                "research_date": date,
                "entry_count": sum(counts.values()),
                "follow_count": counts["follow"],
                "fail_count": counts["fail"],
                "censored_count": counts["censored"],
            }
        )
    censor_rows = [
        {
            "censor_reason": reason,
            "count": count,
            "fraction_of_all_entries": _fraction(count, len(target_rows)),
        }
        for reason, count in sorted(censor_counts.items())
    ]

    cause_metrics: dict[str, dict[str, Any]] = {}
    for cause in ("follow", "fail"):
        rows = [row for row in target_rows if row["event_type"] == cause]
        date_counts = Counter(row["research_date"] for row in rows)
        cause_metrics[cause] = {
            "event_count": len(rows),
            "date_count": len(date_counts),
            "maximum_single_date_share": (
                max(date_counts.values()) / len(rows) if rows else 1.0
            ),
            "event_time_ms_p10": (
                float(np.percentile([row["event_time_ms"] for row in rows], 10))
                if rows
                else None
            ),
            "event_time_ms_p50": (
                float(np.percentile([row["event_time_ms"] for row in rows], 50))
                if rows
                else None
            ),
            "event_time_ms_p90": (
                float(np.percentile([row["event_time_ms"] for row in rows], 90))
                if rows
                else None
            ),
        }

    pre_counts = Counter(
        row["pre_detection_transition"] for row in pre_detection_rows
    )
    event_times = np.asarray(
        [
            int(row["event_time_ms"])
            for row in target_rows
            if row["event_type"] in {"follow", "fail"}
        ],
        dtype=np.int64,
    )
    event_time_cdf = {
        str(horizon): (
            float(np.mean(event_times <= horizon)) if len(event_times) else 0.0
        )
        for horizon in (
            100,
            200,
            300,
            500,
            1_000,
            2_000,
            5_000,
            10_000,
            30_000,
            120_000,
        )
    }
    pre_detection_fraction = _fraction(
        pre_counts["follow"] + pre_counts["fail"],
        len(pre_detection_rows),
    )
    warnings = []
    if event_time_cdf["100"] >= 0.20:
        warnings.append("one_tick_barrier_near_100ms_grid_resolution")
    if pre_detection_fraction >= 0.50:
        warnings.append("high_pre_detection_transition_fraction")
    diagnostics = {
        "entry_count": len(target_rows),
        "entry_type_counts": {
            entry_type: sum(counts.values())
            for entry_type, counts in sorted(type_events.items())
        },
        "event_counts": dict(sorted(event_counts.items())),
        "event_fractions": {
            event: _fraction(count, len(target_rows))
            for event, count in sorted(event_counts.items())
        },
        "cause_metrics": cause_metrics,
        "entry_type_event_counts": {
            entry_type: dict(sorted(counts.items()))
            for entry_type, counts in sorted(type_events.items())
        },
        "censor_reason_counts": dict(sorted(censor_counts.items())),
        "interval_ambiguity_fraction": _fraction(
            sum(row["interval_ambiguous"] == "true" for row in target_rows),
            len(target_rows),
        ),
        "event_time_cdf": event_time_cdf,
        "pre_detection_transition_counts": dict(sorted(pre_counts.items())),
        "pre_detection_transition_fraction": pre_detection_fraction,
        "diagnostic_warnings": warnings,
    }
    return diagnostics, variation_rows, censor_rows


def _artifact_manifest(out_dir: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file() or path.name == "run_manifest.json":
            continue
        artifacts.append(
            {
                "path": str(path.relative_to(out_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def run_a1_a2(
    feature_root: Path,
    a0_root: Path,
    out_dir: Path,
    *,
    verify_cache_hashes: bool = True,
) -> dict[str, Any]:
    upstream = verify_a0_contracts(a0_root)
    bindings = discover_bindings(
        feature_root,
        verify_raw_hashes=False,
        verify_cache_hashes=verify_cache_hashes,
    )
    captures = {
        binding.capture_id: load_capture_data(binding) for binding in bindings
    }
    reversal_rows = _read_csv(a0_root / "states/reversal_entries.csv")
    control_rows = _read_csv(a0_root / "states/control_candidates.csv")
    entries, support_rows = build_state_entries(
        captures,
        reversal_rows,
        control_rows,
    )

    primary_entries = [
        row for row in entries if row["primary_common_support"] == "true"
    ]
    state_feature_fields = [
        field
        for field in primary_entries[0]
        if field
        not in {
            "entry_id",
            "entry_type",
            "reversal_indicator_R",
            "capture_id",
            "research_date",
            "role",
            "side",
            "decision_index",
            "decision_ts_ns",
            "cross_ts_ns",
            "detection_delay_ms",
            "obi_bin",
            "risk_set_key",
            "primary_common_support",
        }
    ]
    all_state_features_finite = all(
        math.isfinite(float(row[field]))
        for row in primary_entries
        for field in state_feature_fields
    )
    date_entry_counts = Counter(row["research_date"] for row in primary_entries)
    max_date_entry_share = (
        max(date_entry_counts.values()) / len(primary_entries)
        if primary_entries
        else 1.0
    )
    primary_type_counts = Counter(row["entry_type"] for row in primary_entries)
    a1_gates = {
        "a0_contract_closure": True,
        "state_feature_finiteness": all_state_features_finite,
        "common_risk_set_has_both_types": (
            primary_type_counts["reversal"] >= 1_000
            and primary_type_counts["control"] >= 1_000
        ),
        "all_dates_represented": len(date_entry_counts) == 9,
        "maximum_date_entry_share": max_date_entry_share <= 0.35,
        "frozen_reversal_count_reproduced": len(reversal_rows) == 4_021,
        "frozen_control_count_reproduced": len(control_rows) == 3_162,
    }

    target_rows, pre_detection_rows = materialize_targets(entries, captures)
    diagnostics, variation_rows, censor_rows = target_diagnostics(
        target_rows,
        pre_detection_rows,
    )
    cause_metrics = diagnostics["cause_metrics"]
    type_events = diagnostics["entry_type_event_counts"]
    a2_gates = {
        "minimum_follow_events": (
            cause_metrics["follow"]["event_count"] >= MIN_CAUSE_EVENTS
        ),
        "minimum_fail_events": (
            cause_metrics["fail"]["event_count"] >= MIN_CAUSE_EVENTS
        ),
        "follow_date_support": (
            cause_metrics["follow"]["date_count"] >= MIN_CAUSE_DATES
        ),
        "fail_date_support": (
            cause_metrics["fail"]["date_count"] >= MIN_CAUSE_DATES
        ),
        "follow_date_concentration": (
            cause_metrics["follow"]["maximum_single_date_share"]
            <= MAX_CAUSE_DATE_SHARE
        ),
        "fail_date_concentration": (
            cause_metrics["fail"]["maximum_single_date_share"]
            <= MAX_CAUSE_DATE_SHARE
        ),
        "censoring_fraction": (
            diagnostics["event_fractions"].get("censored", 0.0)
            <= MAX_CENSORING_FRACTION
        ),
        "interval_ambiguity_fraction": (
            diagnostics["interval_ambiguity_fraction"]
            <= MAX_AMBIGUITY_FRACTION
        ),
        "both_causes_in_reversals": (
            type_events.get("reversal", {}).get("follow", 0) > 0
            and type_events.get("reversal", {}).get("fail", 0) > 0
        ),
        "both_causes_in_controls": (
            type_events.get("control", {}).get("follow", 0) > 0
            and type_events.get("control", {}).get("fail", 0) > 0
        ),
    }
    status = (
        "passed"
        if all(a1_gates.values()) and all(a2_gates.values())
        else "failed"
    )
    classification_name = (
        "A1_A2_state_and_targets_materialized"
        if status == "passed"
        else "A1_or_A2_identification_gate_failed"
    )

    _write_json(
        out_dir / "contracts/upstream_closure.json",
        {
            "schema_version": SCHEMA_VERSION,
            **upstream,
            "feature_cache_count": len(bindings),
            "feature_cache_hashes_verified_now": verify_cache_hashes,
        },
    )
    _write_json(
        out_dir / "contracts/state_feature_contract.json",
        {
            "schema_version": SCHEMA_VERSION,
            "decision_timestamp": "reversal_detected_at_or_control_detected_at",
            "primary_risk_set": (
                "full reusable date_side_absolute_OBI_bin common support"
            ),
            "state_feature_fields": state_feature_fields,
            "stronger_H0_only_not_primary": [
                "generic_OBI_slope",
                "generic_OBI_lags",
            ],
            "future_information_allowed": False,
        },
    )
    _write_json(
        out_dir / "contracts/target_materialization_contract.json",
        {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": "OBI_REVERSAL_V1",
            "quote_convention": "midpoint",
            "input_future_field": FUTURE_TARGET_FIELD,
            "input_semantics": (
                "100ms grid-close midpoint increment versus previous grid"
            ),
            "clock_start": "decision_timestamp",
            "future_accumulation_starts": "next_grid_after_decision",
            "orientation": "side_times_future_midpoint_displacement_ticks",
            "barrier_ticks": BARRIER_TICKS,
            "tau_max_ms": TAU_MAX_MS,
            "causes": {"1": "follow", "2": "fail", "0": "censored"},
            "post_first_passage_values_retained": False,
            "post_barrier_return_or_markout_allowed": False,
            "sampling_limitation": (
                "first passage is observed at 100ms grid closes; "
                "within-grid dual barrier hits are not observable"
            ),
        },
    )
    _write_json(
        out_dir / "contracts/gate_contract.json",
        {
            "schema_version": SCHEMA_VERSION,
            "a1_gates": a1_gates,
            "a2_gates": a2_gates,
            "a2_thresholds_frozen_before_target_materialization": {
                "minimum_events_per_cause": MIN_CAUSE_EVENTS,
                "minimum_dates_per_cause": MIN_CAUSE_DATES,
                "maximum_single_date_share_per_cause": MAX_CAUSE_DATE_SHARE,
                "maximum_censoring_fraction": MAX_CENSORING_FRACTION,
                "maximum_interval_ambiguity_fraction": MAX_AMBIGUITY_FRACTION,
                "both_causes_required_in_each_entry_type": True,
            },
        },
    )
    _write_json(
        out_dir / "contracts/outcome_access_ledger.json",
        {
            "schema_version": SCHEMA_VERSION,
            "A1_decision_time_fields_read": list(DECISION_FIELDS),
            "A1_future_fields_read": [],
            "A2_future_fields_read": [FUTURE_TARGET_FIELD],
            "future_values_retained": [
                "first_transition_type",
                "first_transition_time",
                "censoring_status",
            ],
            "post_first_passage_path_read_for_features": False,
            "post_first_passage_return_or_markout_materialized": False,
            "H0_H1_model_fitted": False,
            "private_API_access": False,
            "orders": 0,
            "new_collection": False,
        },
    )

    _write_csv(
        out_dir / "states/directional_entry_ledger.csv",
        entries,
        list(entries[0]),
    )
    reversal_ledger_rows = [
        row for row in entries if row["entry_type"] == "reversal"
    ]
    _write_csv(
        out_dir / "states/reversal_ledger.csv",
        reversal_ledger_rows,
        list(reversal_ledger_rows[0]),
    )
    _write_csv(
        out_dir / "states/common_risk_set_support.csv",
        support_rows,
        list(support_rows[0]),
    )
    _write_csv(
        out_dir / "targets/first_passage_ledger.csv",
        target_rows,
        list(target_rows[0]),
    )
    _write_csv(
        out_dir / "targets/pre_detection_transition.csv",
        pre_detection_rows,
        list(pre_detection_rows[0]),
    )
    _write_csv(
        out_dir / "targets/target_variation_by_date.csv",
        variation_rows,
        list(variation_rows[0]),
    )
    _write_csv(
        out_dir / "targets/censoring_summary.csv",
        censor_rows,
        list(censor_rows[0]) if censor_rows else [
            "censor_reason",
            "count",
            "fraction_of_all_entries",
        ],
    )
    _write_json(
        out_dir / "support/target_diagnostics.json",
        diagnostics,
    )
    _write_json(
        out_dir / "reports/A1_A2_summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "classification": classification_name,
            "all_entry_count": len(entries),
            "primary_entry_count": len(primary_entries),
            "primary_reversal_count": primary_type_counts["reversal"],
            "primary_control_count": primary_type_counts["control"],
            "target_event_counts": diagnostics["event_counts"],
            "target_event_fractions": diagnostics["event_fractions"],
            "cause_metrics": diagnostics["cause_metrics"],
            "pre_detection_transition_fraction": diagnostics[
                "pre_detection_transition_fraction"
            ],
            "diagnostic_warnings": diagnostics["diagnostic_warnings"],
            "a1_gates": a1_gates,
            "a2_gates": a2_gates,
        },
    )
    _write_json(
        out_dir / "classification.json",
        {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "stage": "A1_state_ledger_and_A2_target_materialization",
            "status": status,
            "classification": classification_name,
            "a1_gates": a1_gates,
            "a2_gates": a2_gates,
            "H0_H1_increment_tested": False,
            "predictive_value_claim_allowed": False,
            "prospective_claim_prohibited": True,
            "diagnostic_warnings": diagnostics["diagnostic_warnings"],
            "next_stage_authorized": status == "passed",
            "next_stage": (
                "A3_H0_baseline_and_H1_incremental_competing_risk_test"
                if status == "passed"
                else "stop_and_classify_identification_failure"
            ),
        },
    )
    manifest = _artifact_manifest(out_dir)
    _write_json(out_dir / "run_manifest.json", manifest)
    return {
        "status": status,
        "classification": classification_name,
        "artifact_count": manifest["artifact_count"],
        "all_entry_count": len(entries),
        "primary_entry_count": len(primary_entries),
        "event_counts": diagnostics["event_counts"],
        "pre_detection_transition_fraction": diagnostics[
            "pre_detection_transition_fraction"
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--a0-root", type=Path, default=DEFAULT_A0_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-cache-hash-verification",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = run_a1_a2(
        args.feature_root,
        args.a0_root,
        args.output,
        verify_cache_hashes=not args.skip_cache_hash_verification,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
