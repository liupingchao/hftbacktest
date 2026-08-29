#!/usr/bin/env python3
"""Execute the frozen FRESH_CHANNEL_CONSENSUS_MSTATE_V1 A-1 audit."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


TASK_ID = "0829T002"
HYPOTHESIS_ID = "FRESH_CHANNEL_CONSENSUS_MSTATE_V1"
AUDIT_ID = "FRESH_CHANNEL_CONSENSUS_MSTATE_V1_A_MINUS1"
SCHEMA_VERSION = "skhynix_fresh_channel_consensus_mstate_a_minus1_v1"
PLAN_PATH = Path(
    "docs/"
    "skhynix_binance_precision_first_fresh_channel_consensus_mstate_v1_"
    "a_minus1_audit_plan_20260829.md"
)
PLAN_SHA256 = "9f214893d7d4ef0431efaf6641c52c84d2fe4f0b516897818b92671355343de5"
PREDECESSOR_PATH = Path(
    "examples/hyperliquid/skhynix_flow_coherence_a_minus1_audit.py"
)
PREDECESSOR_COMMIT = "45544ecc"
PREDECESSOR_BLOB_OID = "494c203e7195f292e057f7708c99f52096259a02"
PREDECESSOR_SHA256 = (
    "f7dc1565bf0a45363dadf3204d827e0d13687f6cc3307c2e7c5e77aeb321400c"
)
PREDECESSOR_AST_SHA256 = {
    "base_masks": (
        "bc2155a38bd1707fcdb77bdebea611da3889934a47d415bdfd0d5a95842d7114"
    ),
    "build_features": (
        "e5cca6c2b7627ef8e3719e4fdecb5a540a42a2028e2fb141ea9fff4f7c246933"
    ),
    "fixed_opposite_orientation_pairs": (
        "04cef064fdaf5cba94421d6d3250760ccf623b2d0531a883154d2a3bdb4b293d"
    ),
    "null_layout": (
        "2def320606fa9caf45e5878845e91026b1284b57b1dd3ff8265038de92c8dcf5"
    ),
    "permute_trade_direction_paths": (
        "b870945f3a079f34337912776001c8bbe8af41c2644e2c0b4acf76277e7637ce"
    ),
    "prior_count": (
        "abbd39099a876c70713b05871152fdfa56d1ee8d142611debe849f1afcc018a8"
    ),
    "run_length": (
        "074d37d93e2b66de5d94c9112494fd3f615a5f24aa3ce77a2f5da64a5b37e37d"
    ),
}
CONTROLLER_COMMIT = "af0d38f432a343ec3fb5d89c7567f93745565f91"
SOURCE_AUTHORITY_INVENTORY_PATH = Path(
    "local_live_analysis/"
    "skhynix_precision_first_flow_coherence_a_minus1_0829T001/"
    "support/source_cache_inventory.csv"
)
SOURCE_AUTHORITY_INVENTORY_BLOB = (
    "c1c877b65a25e8976f972883b311c5a086ed4536"
)
SOURCE_AUTHORITY_INVENTORY_SHA256 = (
    "e6f8f3fedb76eeed6d99cb8cb5306732af54f20bcb0882b56983dc61273a39e1"
)
SOURCE_AUTHORITY_INVENTORY_PAYLOAD_SHA256 = (
    "e554793e98d9a000b1b8c0049897ed42a14167d07f16b1602acfcbb2f048b12c"
)
SOURCE_AUTHORITY_CONTRACT_PATH = Path(
    "local_live_analysis/"
    "skhynix_precision_first_flow_coherence_a_minus1_0829T001/"
    "contracts/source_cache_contract.json"
)
SOURCE_AUTHORITY_CONTRACT_BLOB = (
    "e72e34b2f9fe78eb4fed6515f7dcac8e14fc2c59"
)
SOURCE_AUTHORITY_CONTRACT_SHA256 = (
    "4b6b2c570093725a2f83a356d453b0c022bb3c47c23e8daf441c51ded55de494"
)

DEFAULT_SOURCE_CACHE_ROOT = Path(
    "/Users/liu/Documents/"
    "hftbacktest-0829t001-precision-first-flow-coherence-audit/"
    "local_live_analysis/"
    "skhynix_precision_first_flow_coherence_a_minus1_0829T001/cache"
)
DEFAULT_OUT_DIR = Path(
    "local_live_analysis/"
    "skhynix_fresh_channel_consensus_mstate_a_minus1_0829T002"
)

CHECKPOINT_MS = 20
CHECKPOINT_NS = 20_000_000
COOLDOWN_NS = 30_000_000_000
REFRACTORY_NS = 30_000_000_000
DEPENDENCE_NS = 30_000_000_000
RATIO_HISTORY_MS = 500
PRESTATE_MS = 120
PRESTATE_COUNT = PRESTATE_MS // CHECKPOINT_MS
NULL_REPLICATES = 199
NULL_SEED = 20260829
NULL_DURATIONS_MS = (10_000, 30_000, 60_000)
SELECTION_DURATION_MS = 30_000
ACTIVITY_Q60 = 44.0
RAW_RATE_LIMIT_PER_HOUR = 5.0
RAW_BURST_LIMIT = 2
SELECTION_NULL_RATE_LIMIT = 0.10
REQUIRED_NON_CACHE_ARTIFACTS = frozenset(
    {
        "classification.json",
        "contracts/cross_fit_selection_contract.json",
        "contracts/execution_evidence_contract.json",
        "contracts/gate_contract.json",
        "contracts/outcome_access_ledger.json",
        "contracts/precision_filter_family_contract.json",
        "contracts/source_cache_contract.json",
        "contracts/structural_null_contract.json",
        "contracts/mstate_detector_contract.json",
        "reports/A_minus1_summary.json",
        "run_manifest.json",
        "support/candidate_ledger.csv",
        "support/channel_state_support_by_date.csv",
        "support/cross_fitted_null_summary.csv",
        "support/cross_fitted_signal_ledger.csv",
        "support/filter_support_by_date.csv",
        "support/fold_selection_ledger.csv",
        "support/mstate_support_by_date.csv",
        "support/orphan_strict_onset_by_date.csv",
        "support/parameter_monotonicity.csv",
        "support/slice_invariance.csv",
        "support/source_cache_inventory.csv",
        "support/structural_false_fire_summary.csv",
    }
)


class AuditError(RuntimeError):
    """Fail-closed audit error."""


@dataclass(frozen=True)
class PrecisionFilter:
    filter_id: str
    ttl_ms: int
    persistence_ms: int
    margin: float

    @property
    def persistence_count(self) -> int:
        return self.persistence_ms // CHECKPOINT_MS


FILTERS = tuple(
    PrecisionFilter(f"F{ti}{pi}{mi}", ttl, persistence, margin)
    for ti, ttl in enumerate((100, 60, 40))
    for pi, persistence in enumerate((200, 400, 800))
    for mi, margin in enumerate((0.00, 0.10, 0.20))
)
FILTER_INDEX = {item.filter_id: index for index, item in enumerate(FILTERS)}
CHANNELS = ("trade", "depletion", "ofi")
MARGINS = (0.0, 0.1, 0.2)
TTLS_MS = (100, 60, 40)

GLOBAL_INVALID = 0
NEW_INVALID = 1
NEW_POS = 2
NEW_NEG = 3
NEW_NEUTRAL = 4
NO_UPDATE = 5
ACTION_NAMES = (
    "GLOBAL_INVALID",
    "NEW_INVALID",
    "NEW_POS",
    "NEW_NEG",
    "NEW_NEUTRAL",
    "NO_UPDATE",
)

M_SIGNAL_NEG = -1
M_BACKGROUND = 0
M_SIGNAL_POS = 1
M_ABSTAIN = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def function_ast_hashes(source: str) -> dict[str, str]:
    tree = ast.parse(source)
    result: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name not in PREDECESSOR_AST_SHA256:
                continue
            dumped = ast.dump(
                node, annotate_fields=True, include_attributes=False
            ).encode("ascii")
            result[node.name] = hashlib.sha256(dumped).hexdigest()
    return result


def load_bound_predecessor(repo_root: Path) -> Any:
    path = (repo_root / PREDECESSOR_PATH).resolve()
    if sha256_file(path) != PREDECESSOR_SHA256:
        raise AuditError("predecessor_blob_sha256_mismatch")
    source = path.read_text(encoding="ascii")
    if function_ast_hashes(source) != PREDECESSOR_AST_SHA256:
        raise AuditError("predecessor_callable_ast_mismatch")

    import subprocess

    blob_oid = subprocess.check_output(
        ["git", "rev-parse", f"{PREDECESSOR_COMMIT}:{PREDECESSOR_PATH}"],
        cwd=repo_root,
        text=True,
    ).strip()
    if blob_oid != PREDECESSOR_BLOB_OID:
        raise AuditError("predecessor_git_blob_oid_mismatch")

    module_name = "skhynix_bound_flow_coherence_a_minus1"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AuditError("predecessor_import_spec_missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    for name in PREDECESSOR_AST_SHA256:
        value = getattr(module, name, None)
        if (
            not callable(value)
            or value.__module__ != module_name
            or value.__name__ != name
            or value.__code__.co_filename != str(path)
        ):
            raise AuditError(f"predecessor_callable_binding:{name}")
    return module


def verify_controller_and_cache_authority(
    *,
    repo_root: Path,
    source_cache_root: Path,
    output_cache_root: Path,
    predecessor: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    subprocess.run(
        ["git", "cat-file", "-e", f"{CONTROLLER_COMMIT}^{{commit}}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", CONTROLLER_COMMIT, "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if ancestry.returncode:
        raise AuditError("controller_commit_not_ancestor")

    authority_rows = (
        (
            SOURCE_AUTHORITY_INVENTORY_PATH,
            SOURCE_AUTHORITY_INVENTORY_BLOB,
            SOURCE_AUTHORITY_INVENTORY_SHA256,
        ),
        (
            SOURCE_AUTHORITY_CONTRACT_PATH,
            SOURCE_AUTHORITY_CONTRACT_BLOB,
            SOURCE_AUTHORITY_CONTRACT_SHA256,
        ),
    )
    bound = []
    for path, blob_oid, expected_sha in authority_rows:
        current = repo_root / path
        if sha256_file(current) != expected_sha:
            raise AuditError(f"source_authority_sha:{path}")
        actual_blob = subprocess.check_output(
            ["git", "rev-parse", f"{CONTROLLER_COMMIT}:{path.as_posix()}"],
            cwd=repo_root,
            text=True,
        ).strip()
        if actual_blob != blob_oid:
            raise AuditError(f"source_authority_blob:{path}")
        bound.append(
            {
                "path": path.as_posix(),
                "blob_oid": blob_oid,
                "sha256": expected_sha,
            }
        )

    raw_rows = predecessor.read_csv(
        repo_root / SOURCE_AUTHORITY_INVENTORY_PATH
    )
    inventory = [
        {
            "cache_name": row["cache_name"],
            "size_bytes": int(row["size_bytes"]),
            "row_count": int(row["row_count"]),
            "cache_schema_version": int(row["cache_schema_version"]),
            "cache_sha256": row["cache_sha256"],
            "paired_determinism_verified": row[
                "paired_determinism_verified"
            ]
            == "True",
            "cache_field_schema_verified": row[
                "cache_field_schema_verified"
            ]
            == "True",
        }
        for row in raw_rows
    ]
    if (
        len(inventory) != 29
        or canonical_sha(inventory)
        != SOURCE_AUTHORITY_INVENTORY_PAYLOAD_SHA256
    ):
        raise AuditError("source_inventory_payload_mismatch")

    output_cache_root.mkdir(parents=True, exist_ok=True)
    for row in inventory:
        name = row["cache_name"]
        source = source_cache_root / name
        if not source.is_file():
            raise AuditError(f"source_cache_missing:{name}")
        if (
            source.stat().st_size != row["size_bytes"]
            or sha256_file(source) != row["cache_sha256"]
        ):
            raise AuditError(f"source_cache_identity:{name}")
        with np.load(source, allow_pickle=False) as values:
            predecessor.validate_cache_field_names(values.files, name)
            if (
                len(values["ts_ns"]) != row["row_count"]
                or int(values["cache_schema_version"][0])
                != row["cache_schema_version"]
            ):
                raise AuditError(f"source_cache_schema:{name}")
        target = output_cache_root / name
        if target.exists():
            if sha256_file(target) != row["cache_sha256"]:
                raise AuditError(f"task_cache_drift:{name}")
        else:
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
    return (
        {
            "controller_commit": CONTROLLER_COMMIT,
            "controller_commit_is_ancestor": True,
            "authority_artifacts": bound,
            "source_blob_closure": True,
        },
        inventory,
    )


def filter_contract_rows() -> list[dict[str, Any]]:
    return [
        {
            "filter_id": item.filter_id,
            "filter_index": index,
            "ttl_ms": item.ttl_ms,
            "persistence_ms": item.persistence_ms,
            "margin": item.margin,
        }
        for index, item in enumerate(FILTERS)
    ]


def source_preflight(
    features: dict[str, np.ndarray],
) -> tuple[int, dict[str, np.ndarray]]:
    unsigned = (
        "trade_total",
        "bid_depletion",
        "ask_depletion",
        "ofi_abs",
    )
    signed = ("trade_signed", "ofi")
    invalid = sum(
        int(np.count_nonzero(~np.isfinite(features[name]) | (features[name] < 0)))
        for name in unsigned
    )
    invalid += sum(
        int(np.count_nonzero(~np.isfinite(features[name])))
        for name in signed
    )
    event_masks = {
        "trade": features["trade_total"] > 0,
        "depletion": (
            features["bid_depletion"] + features["ask_depletion"] > 0
        ),
        "ofi": features["ofi_abs"] > 0,
    }
    return invalid, event_masks


def base_eligibility(
    features: dict[str, np.ndarray], predecessor: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    detector_ready, activity_supported, _ = predecessor.base_masks(
        features, cooldown_ns=COOLDOWN_NS
    )
    base = (
        detector_ready
        & activity_supported
        & features["valid_book"].astype(bool)
    )
    return detector_ready, activity_supported, base


def channel_actions(
    *,
    features: dict[str, np.ndarray],
    base_eligible: np.ndarray,
    event_masks: dict[str, np.ndarray],
    margin: float,
) -> np.ndarray:
    n = len(base_eligible)
    fast = features["ratios_100"]
    medium = features["ratios_500"]
    result = np.full((n, len(CHANNELS)), GLOBAL_INVALID, dtype=np.int8)
    for channel_index, channel in enumerate(CHANNELS):
        active = base_eligible
        new = active & event_masks[channel]
        result[active, channel_index] = NO_UPDATE
        finite = np.isfinite(fast[:, channel_index]) & np.isfinite(
            medium[:, channel_index]
        )
        result[new & ~finite, channel_index] = NEW_INVALID
        valid_new = new & finite
        pos = (
            valid_new
            & (fast[:, channel_index] >= 0.50 + margin)
            & (medium[:, channel_index] >= 0.25 + margin)
        )
        neg = (
            valid_new
            & (fast[:, channel_index] <= -0.50 - margin)
            & (medium[:, channel_index] <= -0.25 - margin)
        )
        if np.any(pos & neg):
            raise AuditError("channel_sign_overlap")
        result[valid_new & ~(pos | neg), channel_index] = NEW_NEUTRAL
        result[pos, channel_index] = NEW_POS
        result[neg, channel_index] = NEW_NEG
    if np.any((result < GLOBAL_INVALID) | (result > NO_UPDATE)):
        raise AuditError("channel_action_partition")
    return result


def channel_memories(
    *,
    actions: np.ndarray,
    ts_ns: np.ndarray,
    segments: np.ndarray,
    ttl_ms: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    family = channel_memory_family(
        actions=actions,
        ts_ns=ts_ns,
        segments=segments,
        ttl_values_ms=(ttl_ms,),
    )
    return family[ttl_ms]


def channel_memory_family(
    *,
    actions: np.ndarray,
    ts_ns: np.ndarray,
    segments: np.ndarray,
    ttl_values_ms: Sequence[int],
) -> dict[
    int, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]
]:
    n = len(ts_ns)
    result = {
        ttl_ms: (
            np.full((n, len(CHANNELS)), 9, dtype=np.int8),
            np.full((n, len(CHANNELS)), -1, dtype=np.int32),
            {
                "expiry": np.zeros(len(CHANNELS), dtype=np.int64),
                "neutral_overwrite": np.zeros(
                    len(CHANNELS), dtype=np.int64
                ),
                "unauthorized_refresh": np.zeros(
                    len(CHANNELS), dtype=np.int64
                ),
                "cross_segment_carry": np.zeros(
                    len(CHANNELS), dtype=np.int64
                ),
            },
        )
        for ttl_ms in ttl_values_ms
    }
    indices = np.arange(n, dtype=np.int64)
    segment_boundary = np.ones(n, dtype=bool)
    segment_boundary[1:] = segments[1:] != segments[:-1]

    for channel_index in range(len(CHANNELS)):
        action = actions[:, channel_index]
        update = np.isin(action, (NEW_POS, NEW_NEG, NEW_NEUTRAL))
        clear = (
            segment_boundary
            | (action == GLOBAL_INVALID)
            | (action == NEW_INVALID)
        )
        last_update = np.maximum.accumulate(
            np.where(update, indices, -1)
        )
        last_clear = np.maximum.accumulate(np.where(clear, indices, -1))
        has_memory = (last_update >= 0) & (last_update >= last_clear)
        safe_index = np.maximum(last_update, 0)
        age_ns = ts_ns - ts_ns[safe_index]
        update_values = np.zeros(n, dtype=np.int8)
        update_values[action == NEW_POS] = 1
        update_values[action == NEW_NEG] = -1
        for ttl_ms, (states, ages_ms, diagnostics) in result.items():
            fresh = (
                has_memory
                & (age_ns >= 0)
                & (age_ns <= ttl_ms * 1_000_000)
            )
            states[fresh, channel_index] = update_values[
                safe_index[fresh]
            ]
            ages_ms[fresh, channel_index] = (
                age_ns[fresh] // 1_000_000
            ).astype(np.int32)
            previous_fresh = np.zeros(n, dtype=bool)
            previous_fresh[1:] = (
                fresh[:-1] & ~segment_boundary[1:]
            )
            diagnostics["expiry"][channel_index] = np.count_nonzero(
                previous_fresh
                & ~fresh
                & (action == NO_UPDATE)
                & ~clear
            )
            previous_directional = np.zeros(n, dtype=bool)
            previous_directional[1:] = (
                np.isin(states[:-1, channel_index], (-1, 1))
                & ~segment_boundary[1:]
            )
            diagnostics["neutral_overwrite"][
                channel_index
            ] = np.count_nonzero(
                (action == NEW_NEUTRAL) & previous_directional
            )
            diagnostics["cross_segment_carry"][
                channel_index
            ] = np.count_nonzero(
                segment_boundary
                & (action == NO_UPDATE)
                & (states[:, channel_index] != 9)
            )
    return result


def aggregate_mstate(memories: np.ndarray) -> np.ndarray:
    unknown = np.any(memories == 9, axis=1)
    positive = np.all(memories == 1, axis=1)
    negative = np.all(memories == -1, axis=1)
    if np.any(positive & negative):
        raise AuditError("mstate_signal_overlap")
    result = np.full(len(memories), M_BACKGROUND, dtype=np.int8)
    result[positive] = M_SIGNAL_POS
    result[negative] = M_SIGNAL_NEG
    result[unknown] = M_ABSTAIN
    return result


def build_mstate_family(
    *,
    features: dict[str, np.ndarray],
    predecessor: Any,
    event_masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    detector_ready, activity_supported, base = base_eligibility(
        features, predecessor
    )
    actions_by_margin = {}
    states_by_key = {}
    ages_by_key = {}
    memory_diagnostics = {}
    for margin in MARGINS:
        actions = channel_actions(
            features=features,
            base_eligible=base,
            event_masks=event_masks,
            margin=margin,
        )
        actions_by_margin[margin] = actions
        memory_family = channel_memory_family(
            actions=actions,
            ts_ns=features["ts_ns"],
            segments=features["segment_id"],
            ttl_values_ms=TTLS_MS,
        )
        for ttl_ms, (states, ages, diagnostics) in memory_family.items():
            key = (ttl_ms, margin)
            states_by_key[key] = aggregate_mstate(states)
            ages_by_key[key] = ages
            memory_diagnostics[key] = diagnostics
    return {
        "detector_ready": detector_ready,
        "activity_supported": activity_supported,
        "base_eligible": base,
        "actions_by_margin": actions_by_margin,
        "states_by_key": states_by_key,
        "ages_by_key": ages_by_key,
        "memory_diagnostics": memory_diagnostics,
    }


def natural_onset_mask(
    state: np.ndarray, segments: np.ndarray, direction: int
) -> np.ndarray:
    signal = state == direction
    prior_background = np.zeros(len(state), dtype=bool)
    for segment in np.unique(segments):
        idx = np.flatnonzero(segments == segment)
        if len(idx) <= PRESTATE_COUNT:
            continue
        background = (state[idx] == M_BACKGROUND).astype(np.int16)
        prefix = np.concatenate(
            ([0], np.cumsum(background, dtype=np.int64))
        )
        positions = np.arange(PRESTATE_COUNT, len(idx))
        totals = prefix[positions] - prefix[positions - PRESTATE_COUNT]
        prior_background[idx[positions]] = totals == PRESTATE_COUNT
    return signal & prior_background


def common_candidate_ledger(
    *,
    capture_id: str,
    research_date: str,
    features: dict[str, np.ndarray],
    family: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    ts = features["ts_ns"]
    events = features["event_seq"]
    segments = features["segment_id"]
    common_state = family["states_by_key"][(100, 0.0)]
    common_ages = family["ages_by_key"][(100, 0.0)]
    candidates: list[tuple[int, int]] = []
    for direction in (-1, 1):
        indices = np.flatnonzero(
            natural_onset_mask(common_state, segments, direction)
        )
        candidates.extend((int(index), direction) for index in indices)
    candidates.sort(key=lambda item: (item[0], item[1]))

    counts = Counter(raw_base_candidates=len(candidates))
    last_by_direction = {-1: -10**30, 1: -10**30}
    admitted: list[dict[str, Any]] = []
    for index, direction in candidates:
        candidate_ts = int(ts[index])
        if candidate_ts < last_by_direction[direction] + REFRACTORY_NS:
            counts["common_refractory_suppressed"] += 1
            continue
        last_by_direction[direction] = candidate_ts
        identity = {
            "capture_id": capture_id,
            "candidate_event_seq": int(events[index]),
            "candidate_ts_ns": candidate_ts,
            "direction": direction,
            "segment_id": int(segments[index]),
        }
        admitted.append(
            {
                **identity,
                "candidate_id": canonical_sha(identity),
                "candidate_index": index,
                "research_date": research_date,
                "channel_last_observation_ages_ms": {
                    channel: int(common_ages[index, channel_index])
                    for channel_index, channel in enumerate(CHANNELS)
                },
                "channel_last_observation_ts_ns": {
                    channel: (
                        candidate_ts
                        - int(common_ages[index, channel_index])
                        * 1_000_000
                    )
                    for channel_index, channel in enumerate(CHANNELS)
                },
                "common_prestate_background_count": PRESTATE_COUNT,
                "common_prestate_abstain_count": 0,
                "common_prestate_signal_count": 0,
            }
        )
    counts["common_refractory_admitted"] = len(admitted)

    previous: dict[str, Any] | None = None
    cluster_start_ts = -1
    for candidate in admitted:
        same_cluster = (
            previous is not None
            and candidate["segment_id"] == previous["segment_id"]
            and candidate["candidate_ts_ns"] - previous["candidate_ts_ns"]
            <= DEPENDENCE_NS
        )
        if not same_cluster:
            cluster_start_ts = int(candidate["candidate_ts_ns"])
        candidate["dependence_cluster_id"] = (
            f"{capture_id}:{candidate['segment_id']}:{cluster_start_ts}"
        )
        previous = candidate
    counts["fixed_cluster_count"] = len(
        {row["dependence_cluster_id"] for row in admitted}
    )
    return admitted, dict(sorted(counts.items()))


def evaluate_filter_family(
    *,
    candidates: list[dict[str, Any]],
    features: dict[str, np.ndarray],
    family: dict[str, Any],
) -> None:
    segments = features["segment_id"]
    events = features["event_seq"]
    ts = features["ts_ns"]
    for candidate in candidates:
        index = int(candidate["candidate_index"])
        direction = int(candidate["direction"])
        admissions: list[str] = []
        confirmations: dict[str, dict[str, int]] = {}
        cancellations: dict[str, str] = {}
        for item in FILTERS:
            state = family["states_by_key"][(item.ttl_ms, item.margin)]
            prestate_start = index - PRESTATE_COUNT
            persistence_end = index + item.persistence_count
            if prestate_start < 0 or persistence_end >= len(state):
                cancellations[item.filter_id] = "insufficient_history"
                continue
            prestate_slice = slice(prestate_start, index)
            persistence_slice = slice(index + 1, persistence_end + 1)
            same_segment = (
                int(segments[prestate_start]) == int(segments[index])
                and int(segments[persistence_end]) == int(segments[index])
            )
            if not same_segment:
                cancellations[item.filter_id] = "segment_boundary"
                continue
            prestate = state[prestate_slice]
            if np.any(prestate == M_ABSTAIN):
                cancellations[item.filter_id] = "prestate_abstain"
                continue
            if not np.all(prestate == M_BACKGROUND):
                cancellations[item.filter_id] = "prestate_not_background"
                continue
            if state[index] == M_ABSTAIN:
                cancellations[item.filter_id] = "anchor_abstain"
                continue
            if state[index] != direction:
                cancellations[item.filter_id] = "anchor_not_consensus"
                continue
            persistence = state[persistence_slice]
            if np.any(persistence == M_ABSTAIN):
                cancellations[item.filter_id] = "persistence_abstain"
                continue
            if np.any(persistence == -direction):
                cancellations[item.filter_id] = "opposite_consensus"
                continue
            if not np.all(persistence == direction):
                cancellations[item.filter_id] = "consensus_lost"
                continue
            admissions.append(item.filter_id)
            confirmations[item.filter_id] = {
                "confirmation_event_seq": int(events[persistence_end]),
                "confirmation_ts_ns": int(ts[persistence_end]),
                "persistence_exposure_ms": item.persistence_ms,
            }
        candidate["admitted_filter_ids"] = admissions
        candidate["confirmations"] = confirmations
        candidate["filter_cancel_reasons"] = cancellations


def analyze_features(
    *,
    capture_id: str,
    research_date: str,
    features: dict[str, np.ndarray],
    predecessor: Any,
    event_masks: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    invalid_source, derived_event_masks = source_preflight(features)
    if invalid_source:
        raise AuditError("invalid_source_contribution")
    if event_masks is None:
        event_masks = derived_event_masks
    else:
        for channel in CHANNELS:
            if not np.array_equal(event_masks[channel], derived_event_masks[channel]):
                raise AuditError(f"new_evidence_mask_mismatch:{channel}")
    mstate_family = build_mstate_family(
        features=features,
        predecessor=predecessor,
        event_masks=event_masks,
    )
    candidates, counts = common_candidate_ledger(
        capture_id=capture_id,
        research_date=research_date,
        features=features,
        family=mstate_family,
    )
    evaluate_filter_family(
        candidates=candidates,
        features=features,
        family=mstate_family,
    )
    signal_counts = Counter()
    cluster_sets: dict[str, set[str]] = {
        item.filter_id: set() for item in FILTERS
    }
    for candidate in candidates:
        for filter_id in candidate["admitted_filter_ids"]:
            signal_counts[filter_id] += 1
            cluster_sets[filter_id].add(candidate["dependence_cluster_id"])
    mstate_counts = {}
    for item in FILTERS:
        state = mstate_family["states_by_key"][(item.ttl_ms, item.margin)]
        mstate_counts[item.filter_id] = {
            "SIGNAL_POS": int(np.count_nonzero(state == M_SIGNAL_POS)),
            "SIGNAL_NEG": int(np.count_nonzero(state == M_SIGNAL_NEG)),
            "BACKGROUND": int(np.count_nonzero(state == M_BACKGROUND)),
            "ABSTAIN": int(np.count_nonzero(state == M_ABSTAIN)),
        }
    return {
        **mstate_family,
        "event_masks": event_masks,
        "ts_ns": features["ts_ns"],
        "segments": features["segment_id"],
        "candidates": candidates,
        "counts": counts,
        "raw_cluster_counts": {
            key: len(value) for key, value in cluster_sets.items()
        },
        "mstate_counts": mstate_counts,
    }


def interval_all_mask(
    mask: np.ndarray,
    segments: np.ndarray,
    *,
    left_count: int,
    right_count: int,
) -> np.ndarray:
    result = np.zeros(len(mask), dtype=bool)
    width = left_count + right_count + 1
    for segment in np.unique(segments):
        idx = np.flatnonzero(segments == segment)
        if len(idx) < width:
            continue
        values = mask[idx].astype(np.int64)
        prefix = np.concatenate(([0], np.cumsum(values, dtype=np.int64)))
        centers = np.arange(left_count, len(idx) - right_count)
        totals = prefix[centers + right_count + 1] - prefix[
            centers - left_count
        ]
        result[idx[centers]] = totals == width
    return result


def exposure_masks(
    *,
    analysis: dict[str, Any],
    comparison: np.ndarray,
    segments: np.ndarray,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for item in FILTERS:
        state = analysis["states_by_key"][(item.ttl_ms, item.margin)]
        supported = state != M_ABSTAIN
        causal = interval_all_mask(
            supported,
            segments,
            left_count=PRESTATE_COUNT,
            right_count=item.persistence_count,
        )
        comparison_path = interval_all_mask(
            comparison,
            segments,
            left_count=(
                PRESTATE_MS + RATIO_HISTORY_MS + item.ttl_ms
            ) // CHECKPOINT_MS,
            right_count=item.persistence_count,
        )
        result[item.filter_id] = causal & comparison_path
    return result


def raw_support_masks(
    analysis: dict[str, Any], segments: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        item.filter_id: interval_all_mask(
            analysis["states_by_key"][(item.ttl_ms, item.margin)]
            != M_ABSTAIN,
            segments,
            left_count=PRESTATE_COUNT,
            right_count=item.persistence_count,
        )
        for item in FILTERS
    }


def filter_cluster_counts(
    analysis: dict[str, Any],
    masks: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, dict[str, list[dict[str, Any]]]]:
    cluster_sets = [set() for _ in FILTERS]
    rows: dict[str, list[dict[str, Any]]] = {
        item.filter_id: [] for item in FILTERS
    }
    for candidate in analysis["candidates"]:
        index = int(candidate["candidate_index"])
        for filter_id in candidate["admitted_filter_ids"]:
            item = FILTERS[FILTER_INDEX[filter_id]]
            if masks is not None and not masks[filter_id][index]:
                continue
            cluster_sets[FILTER_INDEX[filter_id]].add(
                candidate["dependence_cluster_id"]
            )
            confirmation = candidate["confirmations"][filter_id]
            rows[filter_id].append(
                {
                    "capture_id": candidate["capture_id"],
                    "research_date": candidate["research_date"],
                    "segment_id": candidate["segment_id"],
                    "direction": candidate["direction"],
                    "candidate_id": candidate["candidate_id"],
                    "candidate_ts_ns": candidate["candidate_ts_ns"],
                    "candidate_event_seq": candidate["candidate_event_seq"],
                    "confirmation_ts_ns": confirmation[
                        "confirmation_ts_ns"
                    ],
                    "confirmation_event_seq": confirmation[
                        "confirmation_event_seq"
                    ],
                    "dependence_cluster_id": candidate[
                        "dependence_cluster_id"
                    ],
                    "filter_id": filter_id,
                    "ttl_ms": item.ttl_ms,
                    "persistence_ms": item.persistence_ms,
                    "margin": item.margin,
                }
            )
    return (
        np.asarray([len(value) for value in cluster_sets], dtype=np.int64),
        rows,
    )


def candidate_diagnostics(
    candidate_batches: Sequence[Sequence[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, str]:
    counters: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    clusters: dict[tuple[str, str], set[str]] = defaultdict(set)
    hash_rows = []
    csv_rows = []
    for candidates in candidate_batches:
        for candidate in candidates:
            hash_row = (
                {
                    "capture_id": candidate["capture_id"],
                    "candidate_ts_ns": candidate["candidate_ts_ns"],
                    "candidate_event_seq": candidate["candidate_event_seq"],
                    "direction": candidate["direction"],
                    "segment_id": candidate["segment_id"],
                    "dependence_cluster_id": candidate[
                        "dependence_cluster_id"
                    ],
                    "admitted_filter_ids": candidate[
                        "admitted_filter_ids"
                    ],
                    "filter_cancel_reasons": candidate[
                        "filter_cancel_reasons"
                    ],
                }
            )
            hash_rows.append(hash_row)
            csv_rows.append(
                {
                    "capture_id": candidate["capture_id"],
                    "research_date": candidate["research_date"],
                    "segment_id": candidate["segment_id"],
                    "direction": candidate["direction"],
                    "candidate_id": candidate["candidate_id"],
                    "candidate_ts_ns": candidate["candidate_ts_ns"],
                    "candidate_event_seq": candidate[
                        "candidate_event_seq"
                    ],
                    "dependence_cluster_id": candidate[
                        "dependence_cluster_id"
                    ],
                    "channel_last_observation_ts_json": json.dumps(
                        candidate["channel_last_observation_ts_ns"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "channel_last_observation_age_ms_json": json.dumps(
                        candidate["channel_last_observation_ages_ms"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "common_prestate_background_count": candidate[
                        "common_prestate_background_count"
                    ],
                    "common_prestate_abstain_count": candidate[
                        "common_prestate_abstain_count"
                    ],
                    "common_prestate_signal_count": candidate[
                        "common_prestate_signal_count"
                    ],
                    "admitted_filter_ids": ";".join(
                        candidate["admitted_filter_ids"]
                    ),
                    "confirmation_map_json": json.dumps(
                        candidate["confirmations"],
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ),
                    "cancel_reason_map_json": json.dumps(
                        candidate["filter_cancel_reasons"],
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ),
                }
            )
            date = str(candidate["research_date"])
            for item in FILTERS:
                key = (date, item.filter_id)
                counters[key]["common_candidate_count"] += 1
                if item.filter_id in candidate["admitted_filter_ids"]:
                    counters[key]["admitted_candidate_count"] += 1
                    clusters[key].add(candidate["dependence_cluster_id"])
                else:
                    reason = candidate["filter_cancel_reasons"][
                        item.filter_id
                    ]
                    counters[key][f"cancel_{reason}"] += 1
    rows = []
    fields = (
        "cancel_insufficient_history",
        "cancel_segment_boundary",
        "cancel_prestate_abstain",
        "cancel_prestate_not_background",
        "cancel_anchor_abstain",
        "cancel_anchor_not_consensus",
        "cancel_persistence_abstain",
        "cancel_opposite_consensus",
        "cancel_consensus_lost",
    )
    for key in sorted(counters):
        date, filter_id = key
        values = counters[key]
        rows.append(
            {
                "research_date": date,
                "filter_id": filter_id,
                "common_candidate_count": values[
                    "common_candidate_count"
                ],
                "admitted_candidate_count": values[
                    "admitted_candidate_count"
                ],
                "admitted_cluster_count": len(clusters[key]),
                **{field: values[field] for field in fields},
            }
        )
    hash_rows.sort(
        key=lambda row: (
            row["capture_id"],
            row["candidate_ts_ns"],
            row["candidate_event_seq"],
            row["direction"],
        )
    )
    csv_rows.sort(
        key=lambda row: (
            row["capture_id"],
            row["candidate_ts_ns"],
            row["candidate_event_seq"],
            row["direction"],
        )
    )
    return rows, csv_rows, len(hash_rows), canonical_sha(hash_rows)


def mstate_support_rows(
    dated_analyses: Sequence[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    totals: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for date, analysis in dated_analyses:
        for item in FILTERS:
            states = analysis["mstate_counts"][item.filter_id]
            totals[(date, item.filter_id)].update(states)
            totals[(date, item.filter_id)]["EXPECTED"] += (
                len(analysis["base_eligible"])
            )
    rows = []
    for (date, filter_id), values in sorted(totals.items()):
        total = (
            values["SIGNAL_POS"]
            + values["SIGNAL_NEG"]
            + values["BACKGROUND"]
            + values["ABSTAIN"]
        )
        rows.append(
            {
                "research_date": date,
                "filter_id": filter_id,
                "signal_pos_checkpoint_count": values["SIGNAL_POS"],
                "signal_neg_checkpoint_count": values["SIGNAL_NEG"],
                "background_checkpoint_count": values["BACKGROUND"],
                "abstain_checkpoint_count": values["ABSTAIN"],
                "total_checkpoint_count": total,
                "partition_exact": total == values["EXPECTED"],
            }
        )
    return rows


def channel_state_support_rows(
    dated_analyses: Sequence[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    totals: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    maximum_age: dict[tuple[str, str, str], int] = defaultdict(lambda: -1)
    for date, analysis in dated_analyses:
        for item in FILTERS:
            actions = analysis["actions_by_margin"][item.margin]
            ages = analysis["ages_by_key"][(item.ttl_ms, item.margin)]
            diagnostics = analysis["memory_diagnostics"][
                (item.ttl_ms, item.margin)
            ]
            for channel_index, channel in enumerate(CHANNELS):
                key = (date, item.filter_id, channel)
                action = actions[:, channel_index]
                for code, name in enumerate(ACTION_NAMES):
                    totals[key][name] += int(np.count_nonzero(action == code))
                totals[key]["EXPECTED"] += len(action)
                totals[key]["NEW_EVIDENCE"] += int(
                    np.count_nonzero(analysis["event_masks"][channel])
                )
                totals[key]["EXPIRY"] += int(
                    diagnostics["expiry"][channel_index]
                )
                totals[key]["NEUTRAL_OVERWRITE"] += int(
                    diagnostics["neutral_overwrite"][channel_index]
                )
                totals[key]["UNAUTHORIZED_REFRESH"] += int(
                    diagnostics["unauthorized_refresh"][channel_index]
                )
                finite_ages = ages[:, channel_index][
                    ages[:, channel_index] >= 0
                ]
                if len(finite_ages):
                    maximum_age[key] = max(
                        maximum_age[key], int(np.max(finite_ages))
                    )
    rows = []
    for (date, filter_id, channel), values in sorted(totals.items()):
        total = sum(values[name] for name in ACTION_NAMES)
        rows.append(
            {
                "research_date": date,
                "filter_id": filter_id,
                "channel": channel,
                "global_invalid_action_count": values["GLOBAL_INVALID"],
                "new_invalid_action_count": values["NEW_INVALID"],
                "new_pos_action_count": values["NEW_POS"],
                "new_neg_action_count": values["NEW_NEG"],
                "new_neutral_action_count": values["NEW_NEUTRAL"],
                "no_update_action_count": values["NO_UPDATE"],
                "total_action_count": total,
                "action_partition_exact": total == values["EXPECTED"],
                "observed_new_evidence_count": values["NEW_EVIDENCE"],
                "invalid_source_contribution_count": 0,
                "expiry_count": values["EXPIRY"],
                "neutral_overwrite_count": values["NEUTRAL_OVERWRITE"],
                "unauthorized_ttl_refresh_count": values[
                    "UNAUTHORIZED_REFRESH"
                ],
                "maximum_memory_age_ms": maximum_age[
                    (date, filter_id, channel)
                ],
                "selection_null_new_evidence_mask_mismatch_count": 0,
                "evaluation_10s_new_evidence_mask_mismatch_count": 0,
                "evaluation_30s_new_evidence_mask_mismatch_count": 0,
                "evaluation_60s_new_evidence_mask_mismatch_count": 0,
            }
        )
    return rows


def orphan_strict_onset_rows(
    dated_analyses: Sequence[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    totals: Counter[tuple[str, str]] = Counter()
    for date, analysis in dated_analyses:
        segments = analysis["segments"]
        common = {
            (candidate["candidate_ts_ns"], candidate["direction"])
            for candidate in analysis["candidates"]
        }
        ts = analysis["ts_ns"]
        for item in FILTERS:
            state = analysis["states_by_key"][(item.ttl_ms, item.margin)]
            natural = set()
            for direction in (-1, 1):
                natural.update(
                    (int(ts[index]), direction)
                    for index in np.flatnonzero(
                        natural_onset_mask(state, segments, direction)
                    )
                )
            totals[(date, item.filter_id)] += len(natural - common)
    return [
        {
            "research_date": date,
            "filter_id": filter_id,
            "orphan_strict_onset_count": count,
        }
        for (date, filter_id), count in sorted(totals.items())
    ]


def maximum_five_second_burst(rows: Sequence[dict[str, Any]]) -> int:
    by_capture: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_capture[str(row["capture_id"])].append(
            int(row["confirmation_ts_ns"])
        )
    maximum = 0
    width = 5_000_000_000
    for timestamps in by_capture.values():
        timestamps.sort()
        left = 0
        for right, value in enumerate(timestamps):
            while value - timestamps[left] > width:
                left += 1
            maximum = max(maximum, right - left + 1)
    return maximum


def finite_type7(values: Iterable[float], q: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not np.all(np.isfinite(array)):
        raise AuditError("nonfinite_quantile_input")
    return float(np.quantile(array, q, method="linear"))


def pair_identities(
    layout: tuple[np.ndarray, list[dict[str, Any]]]
) -> list[tuple[int, int, int]]:
    result = []
    for parent in layout[1]:
        segment, parent_number = parent["parent_key"]
        for pair_id, _ in enumerate(parent["pairs"]):
            result.append((int(segment), int(parent_number), pair_id))
    return result


def pair_distances(
    layout: tuple[np.ndarray, list[dict[str, Any]]]
) -> list[float]:
    return [
        float(pair["distance"])
        for parent in layout[1]
        for pair in parent["pairs"]
    ]


def stream_root(
    bank_code: int,
    duration_ms: int,
    replicate: int,
    capture_ordinal: int,
) -> tuple[int, int, int, int, int]:
    return (
        NULL_SEED,
        bank_code,
        duration_ms,
        replicate,
        capture_ordinal,
    )


def rng_for(
    bank_code: int,
    duration_ms: int,
    replicate: int,
    capture_ordinal: int,
) -> np.random.Generator:
    return np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence(
                stream_root(
                    bank_code, duration_ms, replicate, capture_ordinal
                )
            )
        )
    )


def select_filters(
    *,
    dates: Sequence[str],
    selection_counts: np.ndarray,
    selection_exposure_checkpoint_counts: np.ndarray,
) -> list[dict[str, Any]]:
    if selection_counts.shape != (
        NULL_REPLICATES,
        len(dates),
        len(FILTERS),
    ):
        raise AuditError("selection_count_shape")
    if selection_exposure_checkpoint_counts.shape != (
        len(dates),
        len(FILTERS),
    ):
        raise AuditError("selection_exposure_shape")
    if (
        not np.issubdtype(
            selection_exposure_checkpoint_counts.dtype, np.integer
        )
        or np.any(selection_exposure_checkpoint_counts < 0)
    ):
        raise AuditError("selection_exposure_corrupt")
    rows = []
    for held_out_index, held_out_date in enumerate(dates):
        train_mask = np.ones(len(dates), dtype=bool)
        train_mask[held_out_index] = False
        selected: PrecisionFilter | None = None
        selected_p95 = math.nan
        for filter_index, item in enumerate(FILTERS):
            exposure_checkpoint_count = int(
                np.sum(
                    selection_exposure_checkpoint_counts[
                        train_mask, filter_index
                    ]
                )
            )
            if exposure_checkpoint_count < 0:
                raise AuditError("selection_exposure_corrupt")
            if exposure_checkpoint_count == 0:
                continue
            exposure = float(
                exposure_units(exposure_checkpoint_count)["exposure_hours"]
            )
            replicate_counts = np.sum(
                selection_counts[:, train_mask, filter_index], axis=1
            )
            p95_rate = finite_type7(
                replicate_counts.astype(np.float64) / exposure, 0.95
            )
            if p95_rate <= SELECTION_NULL_RATE_LIMIT:
                selected = item
                selected_p95 = p95_rate
                break
        rows.append(
            {
                "held_out_date": held_out_date,
                "fold_index": held_out_index,
                "fold_state": (
                    "SELECTED" if selected is not None else "META_ABSTAIN"
                ),
                "selected_filter_id": (
                    selected.filter_id if selected is not None else ""
                ),
                "selection_null_rate_p95_per_hour": selected_p95,
            }
        )
    return rows


def monotonicity_rows(
    candidate_sets: dict[str, set[str]],
    cluster_sets: dict[str, set[str]],
) -> list[dict[str, Any]]:
    rows = []
    for left in FILTERS:
        for right in FILTERS:
            stricter = (
                right.persistence_ms >= left.persistence_ms
                and right.margin >= left.margin
                and right.ttl_ms <= left.ttl_ms
                and (
                    right.persistence_ms > left.persistence_ms
                    or right.margin > left.margin
                    or right.ttl_ms < left.ttl_ms
                )
            )
            if not stricter:
                continue
            violations = len(
                candidate_sets[right.filter_id]
                - candidate_sets[left.filter_id]
            )
            cluster_violations = len(
                cluster_sets[right.filter_id]
                - cluster_sets[left.filter_id]
            )
            rows.append(
                {
                    "looser_filter_id": left.filter_id,
                    "stricter_filter_id": right.filter_id,
                    "candidate_subset_violations": violations,
                    "cluster_subset_violations": cluster_violations,
                }
            )
    return rows


def slice_invariance_rows(
    *,
    capture_id: str,
    research_date: str,
    features: dict[str, np.ndarray],
    predecessor: Any,
    full_analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    ts = features["ts_ns"]
    segments = features["segment_id"]
    full_identity = {
        (
            candidate["candidate_ts_ns"],
            candidate["direction"],
            filter_id,
            candidate["dependence_cluster_id"],
        )
        for candidate in full_analysis["candidates"]
        for filter_id in candidate["admitted_filter_ids"]
    }
    rows = []
    stride = 600_000_000_000
    guard = 62_000_000_000
    for segment in np.unique(segments):
        idx = np.flatnonzero(segments == segment)
        if not len(idx) or int(ts[idx[-1]]) - int(ts[idx[0]]) < stride:
            continue
        start_ts = int(ts[idx[0]]) + stride
        while start_ts + guard <= int(ts[idx[-1]]):
            start = int(np.searchsorted(ts, start_ts, side="left"))
            if start >= len(ts) or int(segments[start]) != int(segment):
                start_ts += stride
                continue
            sliced = {
                key: value[start:].copy()
                for key, value in features.items()
                if isinstance(value, np.ndarray) and len(value) == len(ts)
            }
            sliced_analysis = analyze_features(
                capture_id=capture_id,
                research_date=research_date,
                features=sliced,
                predecessor=predecessor,
            )
            comparison_ts = start_ts + guard
            expected = {
                item
                for item in full_identity
                if item[0] >= comparison_ts
                and int(segment)
                == int(
                    segments[
                        int(np.searchsorted(ts, item[0], side="left"))
                    ]
                )
            }
            actual = {
                (
                    candidate["candidate_ts_ns"],
                    candidate["direction"],
                    filter_id,
                    candidate["dependence_cluster_id"],
                )
                for candidate in sliced_analysis["candidates"]
                for filter_id in candidate["admitted_filter_ids"]
                if candidate["candidate_ts_ns"] >= comparison_ts
            }
            full_after_guard = (
                (ts >= comparison_ts) & (segments == int(segment))
            )
            sliced_after_guard = (
                (sliced["ts_ns"] >= comparison_ts)
                & (sliced["segment_id"] == int(segment))
            )
            expected_support_count = int(
                np.count_nonzero(
                    (
                        full_analysis["states_by_key"][(100, 0.0)]
                        != M_ABSTAIN
                    )
                    & full_after_guard
                )
            )
            actual_support_count = int(
                np.count_nonzero(
                    (
                        sliced_analysis["states_by_key"][(100, 0.0)]
                        != M_ABSTAIN
                    )
                    & sliced_after_guard
                )
            )
            rows.append(
                {
                    "capture_id": capture_id,
                    "research_date": research_date,
                    "segment_id": int(segment),
                    "artificial_start_ts_ns": start_ts,
                    "comparison_guard_ts_ns": comparison_ts,
                    "expected_count": len(expected),
                    "actual_count": len(actual),
                    "identity_exact": expected == actual,
                    "expected_support_count": expected_support_count,
                    "actual_support_count": actual_support_count,
                    "support_count_exact": (
                        expected_support_count == actual_support_count
                    ),
                }
            )
            start_ts += stride
    return rows


def gate(
    gate_id: str, conditions: Sequence[tuple[str, bool, Any, str]]
) -> dict[str, Any]:
    rows = [
        {
            "condition": name,
            "passed": bool(passed),
            "actual": actual,
            "required": required,
        }
        for name, passed, actual, required in conditions
    ]
    return {
        "gate_id": gate_id,
        "passed": all(row["passed"] for row in rows),
        "conditions": rows,
    }


def finite_le(value: Any, limit: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) <= limit
    )


def finite_ge(value: Any, limit: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= limit
    )


def exposure_units(checkpoint_count: int) -> dict[str, Any]:
    if not isinstance(checkpoint_count, (int, np.integer)):
        raise AuditError("exposure_checkpoint_count_type")
    seconds = int(checkpoint_count) * CHECKPOINT_MS / 1_000
    return {
        "exposure_checkpoint_count": int(checkpoint_count),
        "exposure_seconds": seconds,
        "exposure_hours": seconds / 3_600,
    }


def numeric_integrity_violations(summary: dict[str, Any]) -> int:
    violations = 0

    def nonnegative_finite(value: Any) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0
        )

    def exact_units(item: dict[str, Any]) -> bool:
        count = item.get("exposure_checkpoint_count")
        seconds = item.get("exposure_seconds")
        hours = item.get("exposure_hours")
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            or not nonnegative_finite(seconds)
            or not nonnegative_finite(hours)
        ):
            return False
        expected_seconds = count * CHECKPOINT_MS / 1_000
        return math.isclose(
            float(seconds), expected_seconds, rel_tol=0, abs_tol=1e-12
        ) and math.isclose(
            float(hours),
            expected_seconds / 3_600,
            rel_tol=0,
            abs_tol=1e-15,
        )

    for item in summary.get("estimators", {}).values():
        violations += int(not exact_units(item))
        observed = item.get("observed_cluster_count")
        null_p95 = item.get("null_cluster_count_p95")
        violations += int(
            not isinstance(observed, int)
            or isinstance(observed, bool)
            or observed < 0
        )
        violations += int(not nonnegative_finite(null_p95))
        exposure_count = item.get("exposure_checkpoint_count")
        rate = item.get("null_false_cluster_rate_p95_per_hour")
        burden = item.get("structural_null_burden_ratio_p95")
        share = item.get("maximum_single_date_share")
        if isinstance(exposure_count, int) and exposure_count == 0:
            violations += int(rate is not None)
        else:
            violations += int(not nonnegative_finite(rate))
        if isinstance(observed, int) and observed == 0:
            violations += int(burden is not None or share is not None)
        else:
            violations += int(not nonnegative_finite(burden))
            violations += int(
                not nonnegative_finite(share)
                or (nonnegative_finite(share) and float(share) > 1)
            )
        for name in ("count_tail_p",):
            value = item.get(name)
            violations += int(
                not nonnegative_finite(value)
                or (nonnegative_finite(value) and float(value) > 1)
            )
        for name in (
            "represented_date_count",
            "dates_above_date_null_p90",
        ):
            value = item.get(name)
            violations += int(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            )

    raw = summary.get("raw", {})
    violations += int(not exact_units(raw))
    raw_observed = raw.get("observed_cluster_count")
    violations += int(
        not isinstance(raw_observed, int)
        or isinstance(raw_observed, bool)
        or raw_observed < 0
    )
    raw_exposure = raw.get("exposure_checkpoint_count")
    raw_rate = raw.get("cluster_rate_per_hour")
    if isinstance(raw_exposure, int) and raw_exposure == 0:
        violations += int(raw_rate is not None)
    else:
        violations += int(not nonnegative_finite(raw_rate))
    raw_burst = raw.get("maximum_5s_burst")
    violations += int(
        not isinstance(raw_burst, int)
        or isinstance(raw_burst, bool)
        or raw_burst < 0
    )
    return violations


def classify(gates: Sequence[dict[str, Any]]) -> str:
    mapping = {
        "A-1-0": "Aminus1_source_not_admissible",
        "A-1-1": "Aminus1_zero_outcome_boundary_violated",
        "A-1-2": "Aminus1_mstate_integrity_failed",
        "A-1-3": "Aminus1_structural_null_not_admissible",
        "A-1-4": "Aminus1_selection_integrity_failed",
        "A-1-5": "Aminus1_structural_support_not_estimable",
        "A-1-6": "Aminus1_structural_false_fire_control_failed",
        "A-1-7": "Aminus1_signal_not_sparse",
    }
    for item in gates:
        if not item["passed"]:
            return mapping[item["gate_id"]]
    return "Aminus1_historical_structural_false_fire_control_candidate"


def build_gates(
    summary: dict[str, Any], *, deterministic_build: bool
) -> list[dict[str, Any]]:
    null = summary["null_admissibility"]
    integrity = summary["integrity"]
    primary = summary["estimators"]["30000"]
    sensitivity_10 = summary["estimators"]["10000"]
    sensitivity_60 = summary["estimators"]["60000"]
    raw = summary["raw"]
    determinism = summary.get("determinism_evidence", {})
    derived_numeric_violations = numeric_integrity_violations(summary)
    stored_numeric_value = summary["integrity"].get(
        "numeric_integrity_violations", 0
    )
    stored_numeric_violations = (
        int(stored_numeric_value)
        if isinstance(stored_numeric_value, int)
        and not isinstance(stored_numeric_value, bool)
        and stored_numeric_value >= 0
        else 1
    )
    total_numeric_violations = (
        derived_numeric_violations + stored_numeric_violations
    )
    gates = [
        gate(
            "A-1-0",
            (
                ("plan_sha", summary["plan_sha_verified"], True, "true"),
                (
                    "predecessor_binding",
                    summary["predecessor_binding_verified"],
                    True,
                    "true",
                ),
                (
                    "source_cache_closure",
                    summary["source_cache_closure"],
                    True,
                    "true",
                ),
                (
                    "deterministic_build",
                    deterministic_build,
                    deterministic_build,
                    "true",
                ),
                (
                    "preseal_difference_count",
                    determinism.get("preseal_difference_count") == 0,
                    determinism.get("preseal_difference_count"),
                    "0",
                ),
                (
                    "pending_difference_count",
                    determinism.get("pending_difference_count") == 0,
                    determinism.get("pending_difference_count"),
                    "0",
                ),
                (
                    "final_difference_count",
                    determinism.get("final_difference_count") == 0,
                    determinism.get("final_difference_count"),
                    "0",
                ),
            ),
        ),
        gate(
            "A-1-1",
            (
                (
                    "zero_outcome_boundary",
                    summary["zero_outcome_boundary"],
                    summary["zero_outcome_boundary"],
                    "true",
                ),
                (
                    "unexpected_fields",
                    summary["unexpected_field_count"] == 0,
                    summary["unexpected_field_count"],
                    "0",
                ),
            ),
        ),
        gate(
            "A-1-2",
            (
                (
                    "mstate_partition_violations",
                    integrity["mstate_partition_violations"] == 0,
                    integrity["mstate_partition_violations"],
                    "0",
                ),
                (
                    "action_partition_violations",
                    integrity["action_partition_violations"] == 0,
                    integrity["action_partition_violations"],
                    "0",
                ),
                (
                    "new_invalid_action_count",
                    integrity["new_invalid_action_count"] == 0,
                    integrity["new_invalid_action_count"],
                    "0",
                ),
                (
                    "ttl_refresh_violations",
                    integrity["ttl_refresh_violations"] == 0,
                    integrity["ttl_refresh_violations"],
                    "0",
                ),
                (
                    "cross_segment_memory_carry",
                    integrity["cross_segment_memory_carry"] == 0,
                    integrity["cross_segment_memory_carry"],
                    "0",
                ),
                (
                    "anchor_contract_violations",
                    integrity["anchor_contract_violations"] == 0,
                    integrity["anchor_contract_violations"],
                    "0",
                ),
                (
                    "feature_boundary_violations",
                    integrity["feature_boundary_violations"] == 0,
                    integrity["feature_boundary_violations"],
                    "0",
                ),
                (
                    "monotonicity_violations",
                    integrity["monotonicity_violations"] == 0,
                    integrity["monotonicity_violations"],
                    "0",
                ),
                (
                    "slice_invariance_mismatches",
                    integrity["slice_invariance_mismatches"] == 0,
                    integrity["slice_invariance_mismatches"],
                    "0",
                ),
            ),
        ),
        gate(
            "A-1-3",
            (
                (
                    "replicate_count",
                    null["replicate_count_exact"],
                    null["replicate_count_exact"],
                    "true",
                ),
                (
                    "fingerprints",
                    null["minimum_distinct_fingerprints"] >= 190,
                    null["minimum_distinct_fingerprints"],
                    ">=190",
                ),
                (
                    "stream_overlap",
                    null["stream_identity_overlap"] == 0,
                    null["stream_identity_overlap"],
                    "0",
                ),
                (
                    "invariant_mismatches",
                    null["invariant_mismatches"] == 0,
                    null["invariant_mismatches"],
                    "0",
                ),
                (
                    "minimum_date_pairs",
                    null["minimum_date_pair_count"] >= 3,
                    null["minimum_date_pair_count"],
                    ">=3",
                ),
                (
                    "p95_joint_distance",
                    finite_le(
                        null["maximum_date_p95_joint_distance"], 0.60
                    ),
                    null["maximum_date_p95_joint_distance"],
                    "<=0.60",
                ),
            ),
        ),
        gate(
            "A-1-4",
            (
                (
                    "fold_count",
                    integrity["fold_count"] == 9,
                    integrity["fold_count"],
                    "9",
                ),
                (
                    "observed_selection_access",
                    integrity["observed_selection_access"] == 0,
                    integrity["observed_selection_access"],
                    "0",
                ),
                (
                    "null_bank_overlap",
                    integrity["null_bank_overlap"] == 0,
                    integrity["null_bank_overlap"],
                    "0",
                ),
                (
                    "numeric_integrity_violations",
                    total_numeric_violations == 0,
                    total_numeric_violations,
                    "0",
                ),
            ),
        ),
        gate(
            "A-1-5",
            (
                (
                    "primary_exposure_positive",
                    finite_ge(primary["exposure_hours"], 0)
                    and primary["exposure_hours"] > 0,
                    primary["exposure_hours"],
                    ">0",
                ),
                (
                    "primary_clusters_ge_30",
                    finite_ge(primary["observed_cluster_count"], 30),
                    primary["observed_cluster_count"],
                    ">=30",
                ),
                (
                    "represented_dates_ge_4",
                    finite_ge(primary["represented_date_count"], 4),
                    primary["represented_date_count"],
                    ">=4",
                ),
                (
                    "single_date_share_le_0_50",
                    finite_le(primary["maximum_single_date_share"], 0.50),
                    primary["maximum_single_date_share"],
                    "<=0.50",
                ),
            ),
        ),
        gate(
            "A-1-6",
            (
                (
                    "primary_null_rate",
                    finite_le(
                        primary[
                            "null_false_cluster_rate_p95_per_hour"
                        ],
                        0.10,
                    ),
                    primary["null_false_cluster_rate_p95_per_hour"],
                    "<=0.10",
                ),
                (
                    "primary_burden",
                    finite_le(
                        primary["structural_null_burden_ratio_p95"],
                        0.10,
                    ),
                    primary["structural_null_burden_ratio_p95"],
                    "<=0.10",
                ),
                (
                    "primary_tail",
                    finite_le(primary["count_tail_p"], 0.01),
                    primary["count_tail_p"],
                    "<=0.01",
                ),
                (
                    "dates_above_null_p90",
                    finite_ge(primary["dates_above_date_null_p90"], 4),
                    primary["dates_above_date_null_p90"],
                    ">=4",
                ),
                (
                    "sensitivity_10_estimable",
                    sensitivity_10["estimable"] is True,
                    sensitivity_10["estimable"],
                    "true",
                ),
                (
                    "sensitivity_60_estimable",
                    sensitivity_60["estimable"] is True,
                    sensitivity_60["estimable"],
                    "true",
                ),
                (
                    "sensitivity_10_rate",
                    finite_le(
                        sensitivity_10[
                            "null_false_cluster_rate_p95_per_hour"
                        ],
                        0.20,
                    ),
                    sensitivity_10[
                        "null_false_cluster_rate_p95_per_hour"
                    ],
                    "<=0.20",
                ),
                (
                    "sensitivity_60_rate",
                    finite_le(
                        sensitivity_60[
                            "null_false_cluster_rate_p95_per_hour"
                        ],
                        0.20,
                    ),
                    sensitivity_60[
                        "null_false_cluster_rate_p95_per_hour"
                    ],
                    "<=0.20",
                ),
                (
                    "sensitivity_10_burden",
                    finite_le(
                        sensitivity_10[
                            "structural_null_burden_ratio_p95"
                        ],
                        0.20,
                    ),
                    sensitivity_10[
                        "structural_null_burden_ratio_p95"
                    ],
                    "<=0.20",
                ),
                (
                    "sensitivity_60_burden",
                    finite_le(
                        sensitivity_60[
                            "structural_null_burden_ratio_p95"
                        ],
                        0.20,
                    ),
                    sensitivity_60[
                        "structural_null_burden_ratio_p95"
                    ],
                    "<=0.20",
                ),
                (
                    "sensitivity_10_tail",
                    finite_le(sensitivity_10["count_tail_p"], 0.05),
                    sensitivity_10["count_tail_p"],
                    "<=0.05",
                ),
                (
                    "sensitivity_60_tail",
                    finite_le(sensitivity_60["count_tail_p"], 0.05),
                    sensitivity_60["count_tail_p"],
                    "<=0.05",
                ),
            ),
        ),
        gate(
            "A-1-7",
            (
                (
                    "raw_exposure_positive",
                    finite_ge(raw["exposure_hours"], 0)
                    and raw["exposure_hours"] > 0,
                    raw["exposure_hours"],
                    ">0",
                ),
                (
                    "raw_rate_le_5",
                    finite_le(
                        raw["cluster_rate_per_hour"],
                        RAW_RATE_LIMIT_PER_HOUR,
                    ),
                    raw["cluster_rate_per_hour"],
                    "<=5",
                ),
                (
                    "raw_burst_le_2",
                    finite_le(
                        raw["maximum_5s_burst"], RAW_BURST_LIMIT
                    ),
                    raw["maximum_5s_burst"],
                    "<=2",
                ),
            ),
        ),
    ]
    return gates


def estimator(
    *,
    observed_by_date: np.ndarray,
    null_by_replicate_date: np.ndarray,
    exposure_checkpoint_count: int,
) -> dict[str, Any]:
    if (
        np.any(observed_by_date < 0)
        or np.any(null_by_replicate_date < 0)
        or exposure_checkpoint_count < 0
    ):
        raise AuditError("negative_estimator_input")
    units = exposure_units(exposure_checkpoint_count)
    exposure_hours = float(units["exposure_hours"])
    observed = int(np.sum(observed_by_date))
    null_totals = np.sum(null_by_replicate_date, axis=1)
    estimable = exposure_hours > 0 and observed > 0
    if exposure_hours <= 0:
        rate = None
    else:
        rate = finite_type7(null_totals, 0.95) / exposure_hours
    burden = (
        finite_type7(null_totals, 0.95) / observed
        if observed > 0
        else None
    )
    tail = (1 + int(np.count_nonzero(null_totals >= observed))) / 200
    represented = int(np.count_nonzero(observed_by_date))
    maximum_share = (
        float(np.max(observed_by_date) / observed)
        if observed > 0
        else None
    )
    dates_above = 0
    for date_index, value in enumerate(observed_by_date):
        p90 = finite_type7(null_by_replicate_date[:, date_index], 0.90)
        dates_above += int(int(value) > p90)
    return {
        "estimable": bool(estimable),
        "observed_cluster_count": observed,
        "null_cluster_count_p95": finite_type7(null_totals, 0.95),
        "null_false_cluster_rate_p95_per_hour": rate,
        "structural_null_burden_ratio_p95": burden,
        "count_tail_p": tail,
        "represented_date_count": represented,
        "maximum_single_date_share": maximum_share,
        "dates_above_date_null_p90": dates_above,
        **units,
    }


def strip_dynamic_summary(summary: dict[str, Any]) -> dict[str, Any]:
    result = dict(summary)
    for key in (
        "gates",
        "classification",
        "deterministic_build",
        "future_target_access_authorized",
        "exploratory_a0_execution_authorized",
        "confirmatory_a0_authorized",
        "prospective_precision_validation_required",
        "determinism_evidence",
    ):
        result.pop(key, None)
    return result


def nonfinite_paths(value: Any, prefix: str = "root") -> list[str]:
    if isinstance(value, float) and not math.isfinite(value):
        return [prefix]
    if isinstance(value, dict):
        return [
            path
            for key, item in value.items()
            for path in nonfinite_paths(item, f"{prefix}.{key}")
        ]
    if isinstance(value, (list, tuple)):
        return [
            path
            for index, item in enumerate(value)
            for path in nonfinite_paths(item, f"{prefix}[{index}]")
        ]
    return []


def sanitize_nonfinite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: sanitize_nonfinite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_nonfinite(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_nonfinite(item) for item in value]
    return value


def non_cache_artifact_paths(output_root: Path) -> set[str]:
    return {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file() and "cache" not in path.parts
    }


def seal_dynamic_outputs(
    *,
    output_root: Path,
    predecessor: Any,
    summary: dict[str, Any],
    deterministic_build: bool,
    determinism_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contracts = output_root / "contracts"
    reports = output_root / "reports"
    nonfinite = nonfinite_paths(summary)
    clean_summary = sanitize_nonfinite(summary)
    clean_summary["integrity"]["numeric_integrity_violations"] = (
        int(
            clean_summary["integrity"].get(
                "numeric_integrity_violations", 0
            )
        )
        + len(nonfinite)
    )
    evidence = determinism_evidence or {
        "stage": "preseal",
        "preseal_difference_count": None,
        "pending_difference_count": None,
        "final_difference_count": None,
    }
    clean_summary["determinism_evidence"] = evidence
    gates = build_gates(
        clean_summary, deterministic_build=deterministic_build
    )
    classification = classify(gates)
    payload = dict(clean_summary)
    payload["gates"] = gates
    payload["classification"] = classification
    payload["deterministic_build"] = deterministic_build
    payload["future_target_access_authorized"] = False
    payload["exploratory_a0_execution_authorized"] = False
    payload["confirmatory_a0_authorized"] = False
    payload["prospective_precision_validation_required"] = (
        classification
        == "Aminus1_historical_structural_false_fire_control_candidate"
    )
    predecessor.write_json(
        contracts / "gate_contract.json", {"gates": gates}
    )
    predecessor.write_json(
        contracts / "execution_evidence_contract.json",
        {
            "deterministic_build": deterministic_build,
            "determinism_evidence": evidence,
            "plan_sha_verified": clean_summary["plan_sha_verified"],
            "predecessor_binding_verified": clean_summary[
                "predecessor_binding_verified"
            ],
            "predecessor_binding": {
                "path": PREDECESSOR_PATH.as_posix(),
                "commit": PREDECESSOR_COMMIT,
                "blob_oid": PREDECESSOR_BLOB_OID,
                "blob_sha256": PREDECESSOR_SHA256,
                "callable_ast_sha256": PREDECESSOR_AST_SHA256,
            },
            "source_cache_closure": clean_summary[
                "source_cache_closure"
            ],
        },
    )
    predecessor.write_json(reports / "A_minus1_summary.json", payload)
    predecessor.write_json(
        output_root / "classification.json",
        {
            "task_id": TASK_ID,
            "hypothesis_id": HYPOTHESIS_ID,
            "audit_id": AUDIT_ID,
            "classification": classification,
            "gates": gates,
            "future_target_access_authorized": False,
            "exploratory_a0_execution_authorized": False,
            "confirmatory_a0_authorized": False,
        },
    )
    manifest = predecessor.artifact_manifest(output_root)
    predecessor.write_json(
        output_root / "run_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "hypothesis_id": HYPOTHESIS_ID,
            "artifact_count": len(manifest["artifacts"]),
            "artifacts": manifest["artifacts"],
        },
    )
    produced = non_cache_artifact_paths(output_root)
    if produced != REQUIRED_NON_CACHE_ARTIFACTS:
        missing = sorted(REQUIRED_NON_CACHE_ARTIFACTS - produced)
        extra = sorted(produced - REQUIRED_NON_CACHE_ARTIFACTS)
        raise AuditError(
            f"required_output_set_mismatch:missing={missing}:extra={extra}"
        )
    return payload


def write_contracts_and_summary(
    *,
    output_root: Path,
    predecessor: Any,
    summary: dict[str, Any],
    source_binding: dict[str, Any],
    cache_inventory: Sequence[dict[str, Any]],
    fold_rows: Sequence[dict[str, Any]],
    signal_rows: Sequence[dict[str, Any]],
    null_summary_rows: Sequence[dict[str, Any]],
    structural_rows: Sequence[dict[str, Any]],
    monotonic_rows: Sequence[dict[str, Any]],
    slice_rows: Sequence[dict[str, Any]],
    admission_rows: Sequence[dict[str, Any]],
    candidate_ledger_rows: Sequence[dict[str, Any]],
    mstate_rows: Sequence[dict[str, Any]],
    channel_rows: Sequence[dict[str, Any]],
    orphan_rows: Sequence[dict[str, Any]],
    deterministic_build: bool,
) -> None:
    contracts = output_root / "contracts"
    support_dir = output_root / "support"
    reports = output_root / "reports"
    contracts.mkdir(parents=True, exist_ok=True)
    support_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    predecessor.write_json(
        contracts / "source_cache_contract.json",
        {
            "source_binding": source_binding,
            "cache_count": len(cache_inventory),
            "cache_inventory_sha256": canonical_sha(cache_inventory),
            "allowed_cache_fields": sorted(
                predecessor.ALLOWED_CACHE_FIELDS
            ),
            "consumed_cache_fields": sorted(
                predecessor.CONSUMED_CACHE_FIELDS
            ),
            "plan_path": PLAN_PATH.as_posix(),
            "plan_sha256": PLAN_SHA256,
        },
    )
    predecessor.write_json(
        contracts / "outcome_access_ledger.json",
        {
            "future_target_accessed": False,
            "future_price_accessed": False,
            "fill_fee_pnl_accessed": False,
            "consumed_cache_fields": sorted(
                predecessor.CONSUMED_CACHE_FIELDS
            ),
            "poisoned_unconsumed_fields_change_output": False,
        },
    )
    for obsolete in (
        contracts / "filter_family_contract.json",
        contracts / "predecessor_binding.json",
        contracts / "selection_contract.json",
        support_dir / "filter_admission_by_date.csv",
    ):
        obsolete.unlink(missing_ok=True)
    predecessor.write_json(
        contracts / "mstate_detector_contract.json",
        {
            "states": [
                "SIGNAL_POS",
                "SIGNAL_NEG",
                "BACKGROUND",
                "ABSTAIN",
            ],
            "channel_actions": list(ACTION_NAMES),
            "channels": list(CHANNELS),
            "new_evidence_masks": {
                "trade": "trade_total > 0",
                "depletion": "bid_depletion + ask_depletion > 0",
                "ofi": "ofi_abs > 0",
            },
            "no_update_refreshes_ttl": False,
            "common_anchor": {
                "ttl_ms": 100,
                "margin": 0.0,
                "prestate_background_ms": PRESTATE_MS,
            },
            "abstain_contributes_to_signal": False,
            "abstain_contributes_to_exposure": False,
            "comparison_mask_is_external": True,
            "mstate_support_rows": len(mstate_rows),
            "channel_support_rows": len(channel_rows),
        },
    )
    predecessor.write_json(
        contracts / "precision_filter_family_contract.json",
        {
            "filter_count": len(FILTERS),
            "filters": filter_contract_rows(),
            "common_refractory_ms": REFRACTORY_NS // 1_000_000,
            "fixed_cluster_ms": DEPENDENCE_NS // 1_000_000,
        },
    )
    predecessor.write_json(
        contracts / "structural_null_contract.json",
        {
            "selection_bank": {
                "bank_code": 3,
                "duration_ms": SELECTION_DURATION_MS,
                "replicates": NULL_REPLICATES,
            },
            "evaluation_bank": {
                "bank_code": 4,
                "durations_ms": list(NULL_DURATIONS_MS),
                "replicates_per_duration": NULL_REPLICATES,
            },
            "seed_root": (
                "[20260829,bank_code,microblock_ms,"
                "replicate_id,capture_ordinal]"
            ),
            "admissibility": summary["null_admissibility"],
        },
    )
    predecessor.write_json(
        contracts / "cross_fit_selection_contract.json",
        {
            "observed_selection_access": False,
            "selection_duration_ms": SELECTION_DURATION_MS,
            "threshold_per_hour": SELECTION_NULL_RATE_LIMIT,
            "none_sentinel": "META_ABSTAIN",
            "folds": list(fold_rows),
        },
    )
    predecessor.write_csv(
        support_dir / "source_cache_inventory.csv",
        list(cache_inventory),
        (
            "cache_name",
            "size_bytes",
            "row_count",
            "cache_schema_version",
            "cache_sha256",
            "paired_determinism_verified",
            "cache_field_schema_verified",
        ),
    )
    predecessor.write_csv(
        support_dir / "fold_selection_ledger.csv",
        list(fold_rows),
        (
            "held_out_date",
            "fold_index",
            "fold_state",
            "selected_filter_id",
            "selection_null_rate_p95_per_hour",
        ),
    )
    predecessor.write_csv(
        support_dir / "cross_fitted_signal_ledger.csv",
        list(signal_rows),
        (
            "capture_id",
            "research_date",
            "segment_id",
            "direction",
            "candidate_id",
            "candidate_ts_ns",
            "candidate_event_seq",
            "confirmation_ts_ns",
            "confirmation_event_seq",
            "dependence_cluster_id",
            "filter_id",
            "persistence_ms",
            "margin",
            "ttl_ms",
            "duration_ms",
        ),
    )
    predecessor.write_csv(
        support_dir / "cross_fitted_null_summary.csv",
        list(null_summary_rows),
        (
            "duration_ms",
            "replicate",
            "cluster_count",
        ),
    )
    predecessor.write_csv(
        support_dir / "structural_false_fire_summary.csv",
        list(structural_rows),
        (
            "duration_ms",
            "estimable",
            "observed_cluster_count",
            "null_cluster_count_p95",
            "exposure_checkpoint_count",
            "exposure_seconds",
            "exposure_hours",
            "null_false_cluster_rate_p95_per_hour",
            "structural_null_burden_ratio_p95",
            "count_tail_p",
            "represented_date_count",
            "maximum_single_date_share",
            "dates_above_date_null_p90",
        ),
    )
    predecessor.write_csv(
        support_dir / "parameter_monotonicity.csv",
        list(monotonic_rows),
        (
            "looser_filter_id",
            "stricter_filter_id",
            "candidate_subset_violations",
            "cluster_subset_violations",
        ),
    )
    predecessor.write_csv(
        support_dir / "slice_invariance.csv",
        list(slice_rows),
        (
            "capture_id",
            "research_date",
            "segment_id",
            "artificial_start_ts_ns",
            "comparison_guard_ts_ns",
            "expected_count",
            "actual_count",
            "identity_exact",
            "expected_support_count",
            "actual_support_count",
            "support_count_exact",
        ),
    )
    predecessor.write_csv(
        support_dir / "mstate_support_by_date.csv",
        list(mstate_rows),
        (
            "research_date",
            "filter_id",
            "signal_pos_checkpoint_count",
            "signal_neg_checkpoint_count",
            "background_checkpoint_count",
            "abstain_checkpoint_count",
            "total_checkpoint_count",
            "partition_exact",
        ),
    )
    predecessor.write_csv(
        support_dir / "channel_state_support_by_date.csv",
        list(channel_rows),
        (
            "research_date",
            "filter_id",
            "channel",
            "global_invalid_action_count",
            "new_invalid_action_count",
            "new_pos_action_count",
            "new_neg_action_count",
            "new_neutral_action_count",
            "no_update_action_count",
            "total_action_count",
            "action_partition_exact",
            "observed_new_evidence_count",
            "invalid_source_contribution_count",
            "expiry_count",
            "neutral_overwrite_count",
            "unauthorized_ttl_refresh_count",
            "maximum_memory_age_ms",
            "selection_null_new_evidence_mask_mismatch_count",
            "evaluation_10s_new_evidence_mask_mismatch_count",
            "evaluation_30s_new_evidence_mask_mismatch_count",
            "evaluation_60s_new_evidence_mask_mismatch_count",
        ),
    )
    predecessor.write_csv(
        support_dir / "orphan_strict_onset_by_date.csv",
        list(orphan_rows),
        (
            "research_date",
            "filter_id",
            "orphan_strict_onset_count",
        ),
    )
    predecessor.write_csv(
        support_dir / "candidate_ledger.csv",
        list(candidate_ledger_rows),
        (
            "capture_id",
            "research_date",
            "segment_id",
            "direction",
            "candidate_id",
            "candidate_ts_ns",
            "candidate_event_seq",
            "dependence_cluster_id",
            "channel_last_observation_ts_json",
            "channel_last_observation_age_ms_json",
            "common_prestate_background_count",
            "common_prestate_abstain_count",
            "common_prestate_signal_count",
            "admitted_filter_ids",
            "confirmation_map_json",
            "cancel_reason_map_json",
        ),
    )
    predecessor.write_csv(
        support_dir / "filter_support_by_date.csv",
        list(admission_rows),
        (
            "research_date",
            "filter_id",
            "common_candidate_count",
            "admitted_candidate_count",
            "admitted_cluster_count",
            "cancel_insufficient_history",
            "cancel_segment_boundary",
            "cancel_prestate_abstain",
            "cancel_prestate_not_background",
            "cancel_anchor_abstain",
            "cancel_anchor_not_consensus",
            "cancel_persistence_abstain",
            "cancel_opposite_consensus",
            "cancel_consensus_lost",
        ),
    )
    seal_dynamic_outputs(
        output_root=output_root,
        predecessor=predecessor,
        summary=summary,
        deterministic_build=deterministic_build,
    )


def execute_audit(
    repo_root: Path,
    source_cache_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    if sha256_file(repo_root / PLAN_PATH) != PLAN_SHA256:
        raise AuditError("plan_sha_mismatch")
    predecessor = load_bound_predecessor(repo_root)
    output_root.mkdir(parents=True, exist_ok=True)
    source_binding, cache_inventory = verify_controller_and_cache_authority(
        repo_root=repo_root,
        source_cache_root=source_cache_root,
        output_cache_root=output_root / "cache",
        predecessor=predecessor,
    )
    cache_inventory.sort(key=lambda row: row["cache_name"].encode("ascii"))
    dates = sorted(
        {
            predecessor.date_from_cache_name(row["cache_name"])
            for row in cache_inventory
        }
    )
    if len(dates) != 9:
        raise AuditError("research_date_count")
    date_index = {value: index for index, value in enumerate(dates)}

    selection_counts = np.zeros(
        (NULL_REPLICATES, len(dates), len(FILTERS)), dtype=np.int64
    )
    evaluation_counts = {
        duration: np.zeros(
            (NULL_REPLICATES, len(dates), len(FILTERS)), dtype=np.int64
        )
        for duration in NULL_DURATIONS_MS
    }
    selection_exposure = np.zeros(
        (len(dates), len(FILTERS)), dtype=np.int64
    )
    evaluation_exposure = {
        duration: np.zeros(
            (len(dates), len(FILTERS)), dtype=np.int64
        )
        for duration in NULL_DURATIONS_MS
    }
    observed_counts = {
        duration: np.zeros((len(dates), len(FILTERS)), dtype=np.int64)
        for duration in NULL_DURATIONS_MS
    }
    observed_rows: dict[
        int, dict[str, dict[str, list[dict[str, Any]]]]
    ] = {
        duration: {
            date: {item.filter_id: [] for item in FILTERS}
            for date in dates
        }
        for duration in NULL_DURATIONS_MS
    }
    raw_counts = np.zeros((len(dates), len(FILTERS)), dtype=np.int64)
    raw_exposure = np.zeros(
        (len(dates), len(FILTERS)), dtype=np.int64
    )
    raw_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        date: {item.filter_id: [] for item in FILTERS} for date in dates
    }
    all_candidate_sets = {
        item.filter_id: set() for item in FILTERS
    }
    all_cluster_sets = {item.filter_id: set() for item in FILTERS}
    slice_rows: list[dict[str, Any]] = []
    observed_candidate_batches: list[list[dict[str, Any]]] = []
    dated_analyses: list[tuple[str, dict[str, Any]]] = []
    mstate_partition_violations = 0
    action_partition_violations = 0
    new_invalid_action_count = 0
    ttl_refresh_violations = 0
    cross_segment_memory_carry = 0
    anchor_contract_violations = 0
    feature_boundary_violations = 0
    null_invariant_mismatches = 0
    pair_counts_by_duration_date: dict[int, Counter[str]] = {
        duration: Counter() for duration in NULL_DURATIONS_MS
    }
    pair_distances_by_duration_date: dict[
        int, dict[str, list[float]]
    ] = {
        duration: defaultdict(list) for duration in NULL_DURATIONS_MS
    }
    fingerprint_hashers: dict[
        tuple[int, int], list[hashlib._Hash]
    ] = {}
    for bank_code, durations in (
        (3, (SELECTION_DURATION_MS,)),
        (4, NULL_DURATIONS_MS),
    ):
        for duration in durations:
            fingerprint_hashers[(bank_code, duration)] = [
                hashlib.sha256() for _ in range(NULL_REPLICATES)
            ]
    stream_ids: dict[int, set[tuple[int, int, int, int, int]]] = {
        3: set(),
        4: set(),
    }

    for capture_ordinal, cache_row in enumerate(cache_inventory):
        cache_name = cache_row["cache_name"]
        capture_id = cache_name[:-4]
        research_date = predecessor.date_from_cache_name(cache_name)
        d_index = date_index[research_date]
        features = predecessor.build_features(
            output_root / "cache" / cache_name
        )
        invalid_source_count, event_masks = source_preflight(features)
        if invalid_source_count:
            raise AuditError(
                f"invalid_source_contribution:{cache_name}:"
                f"{invalid_source_count}"
            )
        feature_boundary_violations += predecessor.feature_window_boundary_violations(
            features
        )
        analysis = analyze_features(
            capture_id=capture_id,
            research_date=research_date,
            features=features,
            predecessor=predecessor,
            event_masks=event_masks,
        )
        observed_candidate_batches.append(analysis["candidates"])
        dated_analyses.append((research_date, analysis))
        support = analysis["base_eligible"]
        segments = features["segment_id"]
        raw_masks = raw_support_masks(analysis, segments)
        raw_count_values, capture_raw_rows = filter_cluster_counts(
            analysis, None
        )
        raw_counts[d_index] += raw_count_values
        for filter_index, item in enumerate(FILTERS):
            raw_exposure[d_index, filter_index] += np.count_nonzero(
                raw_masks[item.filter_id]
            )
            raw_rows[research_date][item.filter_id].extend(
                capture_raw_rows[item.filter_id]
            )
            all_candidate_sets[item.filter_id].update(
                row["candidate_id"]
                for row in capture_raw_rows[item.filter_id]
            )
            all_cluster_sets[item.filter_id].update(
                row["dependence_cluster_id"]
                for row in capture_raw_rows[item.filter_id]
            )
            states = analysis["mstate_counts"][item.filter_id]
            mstate_partition_violations += int(
                states["SIGNAL_POS"]
                + states["SIGNAL_NEG"]
                + states["BACKGROUND"]
                + states["ABSTAIN"]
                != len(support)
            )
            actions = analysis["actions_by_margin"][item.margin]
            action_partition_violations += int(
                np.any((actions < GLOBAL_INVALID) | (actions > NO_UPDATE))
            )
            new_invalid_action_count += int(
                np.count_nonzero(actions == NEW_INVALID)
            )
            diagnostics = analysis["memory_diagnostics"][
                (item.ttl_ms, item.margin)
            ]
            ttl_refresh_violations += int(
                np.sum(diagnostics["unauthorized_refresh"])
            )
            cross_segment_memory_carry += int(
                np.sum(diagnostics["cross_segment_carry"])
            )
        common_state = analysis["states_by_key"][(100, 0.0)]
        for candidate in analysis["candidates"]:
            index = int(candidate["candidate_index"])
            anchor_contract_violations += int(
                common_state[index] != int(candidate["direction"])
                or index < PRESTATE_COUNT
                or not np.all(
                    common_state[index - PRESTATE_COUNT : index]
                    == M_BACKGROUND
                )
            )

        slice_rows.extend(
            slice_invariance_rows(
                capture_id=capture_id,
                research_date=research_date,
                features=features,
                predecessor=predecessor,
                full_analysis=analysis,
            )
        )

        layouts = {
            duration: predecessor.null_layout(
                features, support, duration * 1_000_000
            )
            for duration in NULL_DURATIONS_MS
        }
        exposure_by_duration = {}
        for duration, layout in layouts.items():
            comparison = layout[0]
            masks = exposure_masks(
                analysis=analysis,
                comparison=comparison,
                segments=segments,
            )
            exposure_by_duration[duration] = masks
            count_values, rows_by_filter = filter_cluster_counts(
                analysis, masks
            )
            observed_counts[duration][d_index] += count_values
            for filter_index, item in enumerate(FILTERS):
                checkpoint_count = np.count_nonzero(
                    masks[item.filter_id]
                )
                evaluation_exposure[duration][
                    d_index, filter_index
                ] += checkpoint_count
                if duration == SELECTION_DURATION_MS:
                    selection_exposure[
                        d_index, filter_index
                    ] += checkpoint_count
                observed_rows[duration][research_date][
                    item.filter_id
                ].extend(rows_by_filter[item.filter_id])
            pair_count = len(pair_identities(layout))
            pair_counts_by_duration_date[duration][research_date] += pair_count
            pair_distances_by_duration_date[duration][research_date].extend(
                pair_distances(layout)
            )

        for bank_code, durations in (
            (3, (SELECTION_DURATION_MS,)),
            (4, NULL_DURATIONS_MS),
        ):
            for duration in durations:
                layout = layouts[duration]
                identities = pair_identities(layout)
                comparison = layout[0]
                masks = exposure_by_duration[duration]
                for replicate in range(NULL_REPLICATES):
                    root = stream_root(
                        bank_code, duration, replicate, capture_ordinal
                    )
                    if root in stream_ids[bank_code]:
                        raise AuditError("rng_stream_identity_duplicate")
                    stream_ids[bank_code].add(root)
                    null_features, null_comparison, diagnostics = (
                        predecessor.permute_trade_direction_paths(
                            features,
                            support,
                            rng_for(
                                bank_code,
                                duration,
                                replicate,
                                capture_ordinal,
                            ),
                            duration * 1_000_000,
                            layout=layout,
                        )
                    )
                    if not np.array_equal(comparison, null_comparison):
                        raise AuditError("null_comparison_mask_drift")
                    swap_bits = diagnostics.pop("swap_bits")
                    if len(swap_bits) != len(identities):
                        raise AuditError("null_swap_identity_count")
                    hasher = fingerprint_hashers[(bank_code, duration)][
                        replicate
                    ]
                    for identity, swap in zip(identities, swap_bits):
                        hasher.update(
                            (
                                f"{capture_ordinal}:{identity[0]}:"
                                f"{identity[1]}:{identity[2]}:{int(swap)};"
                            ).encode("ascii")
                        )
                    null_invariant_mismatches += sum(
                        int(diagnostics[name])
                        for name in (
                            "maximum_pair_label_count_difference",
                            "magnitude_mismatches",
                            "zero_mask_mismatches",
                            "missingness_mismatches",
                            "denominator_mismatches",
                        )
                    )
                    null_event_masks = source_preflight(null_features)[1]
                    for channel in CHANNELS:
                        null_invariant_mismatches += int(
                            not np.array_equal(
                                event_masks[channel],
                                null_event_masks[channel],
                            )
                        )
                    null_analysis = analyze_features(
                        capture_id=capture_id,
                        research_date=research_date,
                        features=null_features,
                        predecessor=predecessor,
                        event_masks=event_masks,
                    )
                    values, _ = filter_cluster_counts(
                        null_analysis, masks
                    )
                    if bank_code == 3:
                        selection_counts[replicate, d_index] += values
                    else:
                        evaluation_counts[duration][
                            replicate, d_index
                        ] += values

    fold_rows = select_filters(
        dates=dates,
        selection_counts=selection_counts,
        selection_exposure_checkpoint_counts=selection_exposure,
    )
    selected_indices = [
        (
            FILTER_INDEX[row["selected_filter_id"]]
            if row["selected_filter_id"]
            else None
        )
        for row in fold_rows
    ]
    cross_observed: dict[int, np.ndarray] = {}
    cross_null: dict[int, np.ndarray] = {}
    cross_exposure: dict[int, int] = {}
    signal_rows = []
    null_summary_rows = []
    structural_rows = []
    estimators = {}
    for duration in NULL_DURATIONS_MS:
        by_date = np.zeros(len(dates), dtype=np.int64)
        null_by_rep_date = np.zeros(
            (NULL_REPLICATES, len(dates)), dtype=np.int64
        )
        exposure_checkpoint_count = 0
        for d_index, selected in enumerate(selected_indices):
            if selected is None:
                continue
            by_date[d_index] = observed_counts[duration][d_index, selected]
            null_by_rep_date[:, d_index] = evaluation_counts[duration][
                :, d_index, selected
            ]
            exposure_checkpoint_count += int(
                evaluation_exposure[duration][d_index, selected]
            )
            filter_id = FILTERS[selected].filter_id
            for row in observed_rows[duration][dates[d_index]][filter_id]:
                signal_rows.append({**row, "duration_ms": duration})
        cross_observed[duration] = by_date
        cross_null[duration] = null_by_rep_date
        cross_exposure[duration] = exposure_checkpoint_count
        item = estimator(
            observed_by_date=by_date,
            null_by_replicate_date=null_by_rep_date,
            exposure_checkpoint_count=exposure_checkpoint_count,
        )
        estimators[str(duration)] = item
        structural_rows.append({"duration_ms": duration, **item})
        totals = np.sum(null_by_rep_date, axis=1)
        null_summary_rows.extend(
            {
                "duration_ms": duration,
                "replicate": replicate,
                "cluster_count": int(totals[replicate]),
            }
            for replicate in range(NULL_REPLICATES)
        )

    raw_observed = 0
    raw_checkpoint_count = 0
    selected_raw_rows = []
    for d_index, selected in enumerate(selected_indices):
        if selected is None:
            continue
        raw_observed += int(raw_counts[d_index, selected])
        raw_checkpoint_count += int(raw_exposure[d_index, selected])
        selected_raw_rows.extend(
            raw_rows[dates[d_index]][FILTERS[selected].filter_id]
        )
    raw_units = exposure_units(raw_checkpoint_count)
    raw_hours = float(raw_units["exposure_hours"])
    raw_rate = raw_observed / raw_hours if raw_hours > 0 else None

    monotonic_rows = monotonicity_rows(
        all_candidate_sets, all_cluster_sets
    )
    (
        admission_rows,
        candidate_ledger_rows,
        candidate_count,
        candidate_ledger_sha256,
    ) = (
        candidate_diagnostics(observed_candidate_batches)
    )
    mstate_rows = mstate_support_rows(dated_analyses)
    channel_rows = channel_state_support_rows(dated_analyses)
    orphan_rows = orphan_strict_onset_rows(dated_analyses)
    monotonicity_violations = sum(
        int(row["candidate_subset_violations"])
        + int(row["cluster_subset_violations"])
        for row in monotonic_rows
    )
    slice_mismatches = sum(
        int(not bool(row["identity_exact"]))
        + int(not bool(row["support_count_exact"]))
        for row in slice_rows
    )
    fingerprints = {
        key: len({hasher.hexdigest() for hasher in hashers})
        for key, hashers in fingerprint_hashers.items()
    }
    maximum_date_p95: float | None = 0.0
    minimum_date_pairs = math.inf
    for duration in NULL_DURATIONS_MS:
        for date in dates:
            minimum_date_pairs = min(
                minimum_date_pairs,
                pair_counts_by_duration_date[duration][date],
            )
            values = pair_distances_by_duration_date[duration][date]
            if not values:
                maximum_date_p95 = None
            elif maximum_date_p95 is not None:
                maximum_date_p95 = max(
                    maximum_date_p95,
                    finite_type7(values, 0.95),
                )
    stream_overlap = len(stream_ids[3] & stream_ids[4])
    summary = {
        "task_id": TASK_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "audit_id": AUDIT_ID,
        "plan_sha256": PLAN_SHA256,
        "plan_sha_verified": True,
        "predecessor_binding_verified": True,
        "source_cache_closure": True,
        "zero_outcome_boundary": True,
        "unexpected_field_count": 0,
        "research_dates": dates,
        "cache_count": len(cache_inventory),
        "common_candidate_count": candidate_count,
        "candidate_ledger_sha256": candidate_ledger_sha256,
        "fold_selection": fold_rows,
        "estimators": estimators,
        "raw": {
            "observed_cluster_count": raw_observed,
            **raw_units,
            "cluster_rate_per_hour": raw_rate,
            "maximum_5s_burst": maximum_five_second_burst(
                selected_raw_rows
            ),
        },
        "null_admissibility": {
            "replicate_count_exact": True,
            "minimum_distinct_fingerprints": min(fingerprints.values()),
            "distinct_fingerprints": {
                f"{key[0]}:{key[1]}": value
                for key, value in sorted(fingerprints.items())
            },
            "stream_identity_overlap": stream_overlap,
            "invariant_mismatches": null_invariant_mismatches,
            "minimum_date_pair_count": int(minimum_date_pairs),
            "maximum_date_p95_joint_distance": maximum_date_p95,
        },
        "integrity": {
            "fold_count": len(fold_rows),
            "observed_selection_access": 0,
            "null_bank_overlap": stream_overlap,
            "monotonicity_violations": monotonicity_violations,
            "slice_invariance_mismatches": slice_mismatches,
            "numeric_integrity_violations": 0,
            "mstate_partition_violations": mstate_partition_violations,
            "action_partition_violations": action_partition_violations,
            "new_invalid_action_count": new_invalid_action_count,
            "ttl_refresh_violations": ttl_refresh_violations,
            "cross_segment_memory_carry": cross_segment_memory_carry,
            "anchor_contract_violations": anchor_contract_violations,
            "feature_boundary_violations": feature_boundary_violations,
        },
    }
    write_contracts_and_summary(
        output_root=output_root,
        predecessor=predecessor,
        summary=summary,
        source_binding=source_binding,
        cache_inventory=cache_inventory,
        fold_rows=fold_rows,
        signal_rows=signal_rows,
        null_summary_rows=null_summary_rows,
        structural_rows=structural_rows,
        monotonic_rows=monotonic_rows,
        slice_rows=slice_rows,
        admission_rows=admission_rows,
        candidate_ledger_rows=candidate_ledger_rows,
        mstate_rows=mstate_rows,
        channel_rows=channel_rows,
        orphan_rows=orphan_rows,
        deterministic_build=False,
    )
    return summary


def compare_outputs(
    predecessor: Any, left: Path, right: Path
) -> list[str]:
    if left.resolve() == right.resolve():
        raise AuditError("determinism_pair_roots_not_distinct")
    return predecessor.compare_outputs(left.resolve(), right.resolve())


def repair_existing(repo_root: Path, output_root: Path) -> dict[str, Any]:
    raise AuditError(
        "repair_existing_not_supported_for_frozen_mstate_audit:"
        "run_a_fresh_build"
    )


def finalize_pair(
    repo_root: Path, left: Path, right: Path
) -> dict[str, Any]:
    predecessor = load_bound_predecessor(repo_root)
    differences = compare_outputs(predecessor, left, right)
    if differences:
        raise AuditError(f"preseal_output_mismatch:{differences[:5]}")
    summaries = []
    for root in (left, right):
        summary = strip_dynamic_summary(
            json.loads(
                (root / "reports/A_minus1_summary.json").read_text(
                    encoding="ascii"
                )
            )
        )
        summaries.append(summary)

    pending_evidence = {
        "stage": "pending",
        "preseal_difference_count": 0,
        "pending_difference_count": None,
        "final_difference_count": None,
    }
    for root, summary in zip((left, right), summaries):
        seal_dynamic_outputs(
            output_root=root,
            predecessor=predecessor,
            summary=summary,
            deterministic_build=False,
            determinism_evidence=pending_evidence,
        )
    pending_differences = compare_outputs(predecessor, left, right)
    if pending_differences:
        raise AuditError(
            f"pending_output_mismatch:{pending_differences[:5]}"
        )

    final_pending_evidence = {
        "stage": "final_pending",
        "preseal_difference_count": 0,
        "pending_difference_count": 0,
        "final_difference_count": None,
    }
    for root, summary in zip((left, right), summaries):
        seal_dynamic_outputs(
            output_root=root,
            predecessor=predecessor,
            summary=summary,
            deterministic_build=True,
            determinism_evidence=final_pending_evidence,
        )
    final_differences = compare_outputs(predecessor, left, right)
    if final_differences:
        raise AuditError(
            f"final_pending_output_mismatch:{final_differences[:5]}"
        )

    final_evidence = {
        "stage": "final",
        "preseal_difference_count": 0,
        "pending_difference_count": 0,
        "final_difference_count": 0,
    }
    payloads = []
    for root, summary in zip((left, right), summaries):
        payloads.append(
            seal_dynamic_outputs(
                output_root=root,
                predecessor=predecessor,
                summary=summary,
                deterministic_build=True,
                determinism_evidence=final_evidence,
            )
        )
    differences = compare_outputs(predecessor, left, right)
    if differences:
        raise AuditError(f"final_output_mismatch:{differences[:5]}")
    return payloads[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--source-cache-root",
        type=Path,
        default=DEFAULT_SOURCE_CACHE_ROOT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--finalize-pair",
        type=Path,
        nargs=2,
        metavar=("BUILD_A", "BUILD_B"),
    )
    parser.add_argument("--repair-existing", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if args.finalize_pair and args.repair_existing:
        raise AuditError("multiple_execution_modes")
    if args.finalize_pair:
        summary = finalize_pair(
            repo_root,
            args.finalize_pair[0].resolve(),
            args.finalize_pair[1].resolve(),
        )
    elif args.repair_existing:
        summary = repair_existing(
            repo_root,
            args.repair_existing.resolve(),
        )
    else:
        summary = execute_audit(
            repo_root,
            args.source_cache_root.resolve(),
            args.output_root.resolve(),
        )
    print(json.dumps(summary, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()
