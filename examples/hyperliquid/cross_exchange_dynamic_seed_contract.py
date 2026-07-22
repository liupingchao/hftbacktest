"""Strict loader for the accepted public dynamic-spread seed contract."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from examples.hyperliquid import cross_exchange_online_estimators as estimators


SEED_TASK_ID = "0722T066"
SEED_SCHEMA_VERSION = "cross_exchange_dynamic_spread_seed_v1"
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
SEED_CONTRACT_FIELDS = {
    "schema_version",
    "task_id",
    "source_commit",
    "runner_sha256",
    "symbol",
    "tick_size",
    "interval_ms",
    "distance_ticks",
    "source_event_rows_sha256",
    "quote_exposure_rows_sha256",
    "intensity_fit_rows_sha256",
    "source_event_row_count",
    "quote_exposure_row_count",
    "per_side",
    "seed_eligible",
    "inference_scope",
    "seed_contract_sha256",
}


class DynamicSeedContractError(ValueError):
    """Raised when a dynamic-spread seed fails strict validation."""


def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_hash(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_seed_into_estimator(
    *,
    estimator: estimators.EventTimeOnlineEstimator,
    contract_path: Path,
    exposure_path: Path,
    expected_seed_contract_sha256: str,
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict) or set(contract) != SEED_CONTRACT_FIELDS:
        raise DynamicSeedContractError("seed_contract_fields_mismatch")
    if contract.get("schema_version") != SEED_SCHEMA_VERSION:
        raise DynamicSeedContractError("seed_contract_schema_mismatch")
    if contract.get("task_id") != SEED_TASK_ID:
        raise DynamicSeedContractError("seed_contract_task_id_mismatch")
    if not HEX_SHA256.fullmatch(expected_seed_contract_sha256):
        raise DynamicSeedContractError("expected_seed_contract_sha256_invalid")
    claimed_hash = str(contract.get("seed_contract_sha256") or "")
    payload = dict(contract)
    payload.pop("seed_contract_sha256", None)
    computed_hash = _sha256_bytes(_canonical(payload))
    if claimed_hash != computed_hash:
        raise DynamicSeedContractError("seed_contract_self_hash_mismatch")
    if claimed_hash != expected_seed_contract_sha256:
        raise DynamicSeedContractError("seed_contract_expected_hash_mismatch")
    if _file_hash(exposure_path) != contract["quote_exposure_rows_sha256"]:
        raise DynamicSeedContractError("seed_exposure_file_hash_mismatch")
    if contract.get("seed_eligible") is not True:
        raise DynamicSeedContractError("seed_contract_not_eligible")
    rows = _read_csv(exposure_path)
    if len(rows) != int(contract["quote_exposure_row_count"]):
        raise DynamicSeedContractError("seed_exposure_row_count_mismatch")

    event_count_before = len(estimator.event_rows())
    bucket_count_before = len(estimator.bucket_rows())
    exposure_count_before = len(estimator.quote_exposure_rows())
    for row in rows:
        if (
            str(row.get("resting_confirmed")).lower() not in {"false", "0"}
            or row.get("source")
            != "public_multi_distance_counterfactual_non_resting_seed"
        ):
            raise DynamicSeedContractError("seed_exposure_boundary_mismatch")
        estimator.observe_quote_exposure(
            exposure_id=f"seed:{row['exposure_id']}",
            side=str(row["side"]),
            quote_px=float(row["quote_px"]),
            reference_mid_px=float(row["reference_mid_px"]),
            start_exchange_time_ms=int(row["start_exchange_time_ms"]),
            end_exchange_time_ms=int(row["end_exchange_time_ms"]),
            arrival_count=int(row["arrival_count"]),
            arrival_volume_btc=float(row["arrival_volume_btc"]),
            pre_trade_side_depth_btc=(
                None
                if row.get("pre_trade_side_depth_btc") in {"", None}
                else float(row["pre_trade_side_depth_btc"])
            ),
            max_sweep_depth_penetration=(
                None
                if row.get("max_sweep_depth_penetration") in {"", None}
                else float(row["max_sweep_depth_penetration"])
            ),
            arrival_evidence_source=str(row["arrival_evidence_source"]),
            resting_confirmed=False,
            source=str(row["source"]),
        )
    if (
        len(estimator.event_rows()) != event_count_before
        or len(estimator.bucket_rows()) != bucket_count_before
    ):
        raise DynamicSeedContractError("seed_contaminated_current_market_state")
    fits = {side: estimator.fit_intensity(side) for side in ("buy", "sell")}
    if any(fits[side]["status"] != "pass" for side in fits):
        raise DynamicSeedContractError("loaded_seed_intensity_fit_not_pass")
    return {
        "status": "pass",
        "loaded_row_count": (
            len(estimator.quote_exposure_rows()) - exposure_count_before
        ),
        "seed_contract_sha256": claimed_hash,
        "contract_path": str(contract_path),
        "exposure_path": str(exposure_path),
        "current_market_event_count_before": event_count_before,
        "current_market_event_count_after": len(estimator.event_rows()),
        "current_market_bucket_count_before": bucket_count_before,
        "current_market_bucket_count_after": len(estimator.bucket_rows()),
        "current_market_state_contaminated": False,
        "fits": fits,
        "actual_quote_behavior_changed": False,
        "inference_scope": (
            "counterfactual_intensity_seed_only_current_market_state_unseeded"
        ),
    }
