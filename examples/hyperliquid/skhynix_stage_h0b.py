#!/usr/bin/env python3
"""Execute and verify the frozen Stage H0-B conditional-risk audit."""

from __future__ import annotations

import argparse
import csv
import ctypes
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

import research_package_trust as trust
import skhynix_stage_h0b_contracts as contracts
import skhynix_stage_h0a_support as h0a_support


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_SOURCE_ROOT = Path("/Users/liu/Documents/hftbacktest")
TASK_PATH = REPO_ROOT / ".workflow/tasks/0823T002.md"
MATRIX_PATH = REPO_ROOT / ".workflow/contracts/0823T002-surface-matrix.json"
PRIMARY_PLAN_PATH = (
    REPO_ROOT
    / "docs/skhynix_stage_h0b_conditional_risk_audit_plan_20260823.md"
)
DIAGNOSTIC_PLAN_PATH = (
    REPO_ROOT
    / "docs/skhynix_stage_h0b_conditional_risk_audit_plan_v2_20260823.md"
)
FRAMEWORK_PATH = (
    REPO_ROOT
    / "docs/skhynix_continuous_hazard_maker_research_framework_v2.md"
)
PRIMARY_PLAN_REVIEW_PATH = (
    REPO_ROOT / ".workflow/reports/0823T002-plan-review.md"
)
DIAGNOSTIC_PLAN_REVIEW_PATH = (
    REPO_ROOT / ".workflow/reports/0823T002-plan-v2-review.md"
)
SEMANTIC_INVENTORY_PATH = (
    REPO_ROOT
    / ".workflow/contracts/0823T002-semantic-source-inventory.csv"
)
SOURCE_INVENTORY_CONTRACT_PATH = (
    REPO_ROOT
    / ".workflow/contracts/0823T002-source-inventory-contract.json"
)
H0A_ROOT = (
    REPO_ROOT
    / "local_live_analysis/"
    "skhynix_continuous_conditional_risk_v2_stage_h0a_support_only"
)
LATENCY_ROOT = (
    REPO_ROOT
    / "local_live_analysis/"
    "skhynix_c6in_hyperliquid_execution_latency_0822T002"
)
TUPLE_ROOT = (
    REPO_ROOT
    / "local_live_analysis/"
    "skhynix_h0b_primary_tuple_supersession_0823T001"
)
STAGE2_ROOT = (
    REPO_ROOT
    / "local_live_analysis/"
    "skhynix_trigger_aligned_episode_research_v1_stage02_density"
)
STAGE3_ROOT = (
    REPO_ROOT
    / "local_live_analysis/"
    "skhynix_trigger_aligned_episode_research_v1_stage03_detector_parity"
)
STAGE4_ROOT = (
    REPO_ROOT
    / "local_live_analysis/"
    "skhynix_trigger_aligned_episode_research_v1_stage04_jul30_episode_v3"
)
DEFAULT_PACKAGE = (
    REPO_ROOT
    / "local_live_analysis/"
    "skhynix_continuous_conditional_risk_v2_stage_h0b_0823T002"
)
PRIMARY_PLAN_SHA256 = (
    "c1be0fdbd58f19c201c2faa7251621402486e6ebabf259af316b98bcf4c92b10"
)
PRIMARY_REVIEW_SHA256 = (
    "c81112cc215d9b847e1691b9fce6be391cd2604736337ffa7fe35ab89cb22158"
)
DIAGNOSTIC_PLAN_SHA256 = (
    "12b09677c0bcf2e921900f04e424ae7977e967ace70f391ab28c26fc4fb98a63"
)
DIAGNOSTIC_REVIEW_SHA256 = (
    "dfb23069a0d6f057c36b9eec380c66228b30667cbcc98b126c0be1c838171d4d"
)
FRAMEWORK_SHA256 = (
    "20c711fc056004d36ddf671766d04a1bbc2df58116234b4be285a859b55ceeec"
)
EXPECTED_SEMANTIC_INVENTORY_SHA256 = (
    "0a7bcb7c46817ce7189468880a4d19d54edd8c80a3fa66768dffd33597e51da9"
)
SOURCE_INVENTORY_CONTRACT_SHA256 = (
    "c57fce590d62e6d0576fa0ffb186c60372a64b42d1af4e3523651ea5d7cb7686"
)
MATRIX_SHA256 = (
    "478cb177fe5834bfd359d4d4e56004c3fa95c69cf2b4da0a67446ceeec22745b"
)
H0A_TUPLE_SHA256 = (
    "e5d1b132248ff1a6933678c32a54e6b4147c1c6f47dab25103011ecbd7a68eca"
)
H0A_COMPOSITE = (
    "2682c32eefac427eed1899a3d492b3fc7545520f72021723e8fef7a0d4d8d9d0"
)
LATENCY_COMPOSITE = (
    "7d851ab161ec02c621dffff63ef2f3e962a2aa0c7ed8b4382b285df9534bb0df"
)
TUPLE_SHA256 = (
    "e3badf4c179a9e717ea49ff6617b637ec7e4c967c3dec365bebe78ef9399457c"
)
TUPLE_COMPOSITE = (
    "5ec515e00ab2765a281084a64fbe0e1962727059e42734be33b7362258a22f76"
)
KERNEL_SOURCE_TREE_SHA256 = (
    "cee2395afad9420c38235ba195bf030e92330015e1a15937ebc22fa707c80203"
)
STAGE4_HEADER_SHA256 = (
    "e7af9e9e84973ed074a09957435f1a10e146bb03ffa0e3365d238f8d79d01239"
)
STAGE4_OUTCOMES = {
    "outcomes/segment_0001.csv.gz": (
        "669817ba04cdde44d087607d28218aaeb7bf05d4faee74c7809ef612afbb0eee"
    ),
    "outcomes/segment_0002.csv.gz": (
        "4c88223823fc3af73c036bc96494c674cfd2d4368ca3393b0ae133273aee6e6d"
    ),
    "outcomes/segment_0003.csv.gz": (
        "ff699a28b055028e04a2aff861446ff19685ca36684d7913b1678cd8a0fd6547"
    ),
    "outcomes/segment_0004.csv.gz": (
        "88b024fb615b86e7de467911c1ad37ee91197791e0d0c276420fbd88ec0e82f2"
    ),
    "outcomes/segment_0005.csv.gz": (
        "d348bc1d20477efa12da34d7bc957935a312e3a8217367fb9a30b6370ab6b3aa"
    ),
    "outcomes/segment_0006.csv.gz": (
        "c6a255900e6e76dcad950bb7e43e6d1bf5f723beb34c9c494999f3b4082138ab"
    ),
    "outcomes/segment_0007.csv.gz": (
        "b419533bb1a06f7cb7edaa131e66f83e4f95a73112802f3f256a8756402e73ae"
    ),
    "outcomes/segment_0008.csv.gz": (
        "4c6e89b26eafb3999d70f4f10c7501302f5bc2296e198ab615b4580be941777d"
    ),
}
STAGE4_PROJECTED_FIELDS = (
    "candidate_id",
    "episode_id",
    "t_candidate_ns",
    "outcome_horizon_status",
    "time_to_first_adverse_target_bbo_event_status",
    "time_to_first_adverse_target_bbo_event_interval_lower_ns",
    "time_to_first_adverse_target_bbo_event_interval_upper_ns",
    "time_to_first_adverse_target_bbo_event_censor_time_ns",
    "time_to_first_adverse_target_bbo_event_censor_reason",
    "public_bbo_moves_through_quote",
    "public_quote_risk_availability",
)
HYPERLIQUID_HEADER = (
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
BINANCE_HEADER = (
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

PRIMARY_NULL_SEED = 8232001
TIME_BOOTSTRAP_SEED = 8232002
FLOW_BOOTSTRAP_SEED = 8232003
RQ3_BOOTSTRAP_SEED = 8232004
BOOTSTRAP_REPLICATES = 2000
OPTIMIZER_MAX_ITERATIONS = 500
OPTIMIZER_GRADIENT_TOLERANCE = 1e-8
OPTIMIZER_PARAMETER_TOLERANCE = 1e-10
RIDGE_LAMBDA = 1.0
MODEL_CHUNK_ROWS = 250_000
HOSTILE_FAIL_OPEN_SENTINEL = "H0B_HOSTILE_MUTATION_FAILED_OPEN"


def production_contract_state() -> dict[str, Any]:
    feature_sources = (
        "r0_binance_bookticker",
        "r0_hyperliquid_bbo",
        "accepted_stage2_primary",
        "accepted_stage3_primary",
    )
    return {
        "kernel_pin": KERNEL_SOURCE_TREE_SHA256,
        "master_framework_pin": (
            PRIMARY_PLAN_SHA256,
            PRIMARY_REVIEW_SHA256,
            DIAGNOSTIC_PLAN_SHA256,
            DIAGNOSTIC_REVIEW_SHA256,
            FRAMEWORK_SHA256,
            MATRIX_SHA256,
        ),
        "accepted_h0a_binding": (H0A_TUPLE_SHA256, H0A_COMPOSITE),
        "accepted_latency_binding": (
            LATENCY_COMPOSITE,
            contracts.PRIMARY_LATENCY_MS,
        ),
        "accepted_tuple_binding": (TUPLE_SHA256, TUPLE_COMPOSITE),
        "accepted_stage1_4_binding": tuple(sorted(STAGE4_OUTCOMES.items())),
        "session_roles": {
            "formal": tuple(contracts.FORMAL_SESSIONS),
            "aug03": (
                "diagnostic_only",
                False,
                "historical_transfer",
            ),
        },
        "underlying_state_boundary": tuple(
            (*contracts.H0_RAW_FEATURES, *contracts.H1_ADDED_RAW_FEATURES)
        ),
        "semantic_source_inventory": (
            EXPECTED_SEMANTIC_INVENTORY_SHA256,
            SOURCE_INVENTORY_CONTRACT_SHA256,
        ),
        "build_envelope": ("A", "B"),
        "source_schema": (
            HYPERLIQUID_HEADER,
            BINANCE_HEADER,
            STAGE4_PROJECTED_FIELDS,
        ),
        "source_ordering": (
            "event_seq_strictly_increasing",
            "local_ts_ns_nondecreasing",
            "same_timestamp_ordered_by_event_seq",
        ),
        "guarded_opener": (
            "semantic_inventory_only_before_outcome_permit",
            "stage4_only_after_primary_seal_and_diagnostic_permit",
        ),
        "feature_source_boundary": feature_sources,
        "two_envelope_boundary": (
            "H0B0",
            "fresh_H0B1",
            "fresh_H0B1_DIAGNOSTIC",
        ),
        "outcome_access_permit": (
            "skhynix_stage_h0b_outcome_access_permit_v2",
            "A",
            "B",
            "admitted",
            True,
        ),
        "support_replay": (
            "a266184403a830fc422764e90c6dd48a5c1900af13333246ca7e220d664728df"
        ),
        "calendar_grid": (contracts.GRID_NS, contracts.HORIZON_NS, 0),
        "side_expansion": tuple(contracts.SIDES),
        "event_definition": (
            "maker_ask_risk:target_bid_gte_reference_ask",
            "maker_bid_risk:target_ask_lte_reference_bid",
        ),
        "support_class_mapping": tuple(
            sorted(contracts.SUPPORT_DISPOSITIONS.items())
        ),
        "observation_bounds": (
            "event:(L,U]",
            "right_censor:S(50ms)",
            "straddle:S(L)_for_L_lt_50ms_lt_U",
        ),
        "interval_likelihood": (
            "S_exact(L)-S_exact(U)",
            contracts.GRID_NS,
            contracts.HORIZON_NS,
        ),
        "right_censor_likelihood": "S_exact(50ms)",
        "horizon_straddle": "S_exact(L)_without_rounding_or_dropping",
        "risk_score": "1-product_of_all_five_bin_survivals",
        "binary_subset": ("binary_identification_supported",),
        "h0_features": tuple(contracts.H0_RAW_FEATURES),
        "h1_features": tuple(contracts.H1_ADDED_RAW_FEATURES),
        "design_matrix": (
            tuple(contracts.H0_DESIGN_COLUMNS),
            tuple(contracts.H1_DESIGN_COLUMNS),
        ),
        "basis_residual": "prior_only_ewma",
        "missing_value_policy": (
            "training_fold_nearest_rank_median",
            "training_fold_iqr_floor_1",
            "explicit_missing_indicator",
        ),
        "dose_definition": (
            "confirmed_decision_ts",
            "trailing_500ms_queue_drop_ratio",
        ),
        "walk_forward": (
            60,
            20,
            10,
            contracts.PURGE_NS,
            contracts.EMBARGO_NS,
        ),
        "estimator": (
            "piecewise_exponential_discrete_hazard",
            RIDGE_LAMBDA,
            OPTIMIZER_MAX_ITERATIONS,
            OPTIMIZER_GRADIENT_TOLERANCE,
            OPTIMIZER_PARAMETER_TOLERANCE,
        ),
        "numeric_seed_conventions": (
            "nearest_rank",
            "NumPy Generator(PCG64)",
            PRIMARY_NULL_SEED,
            TIME_BOOTSTRAP_SEED,
            FLOW_BOOTSTRAP_SEED,
            RQ3_BOOTSTRAP_SEED,
            BOOTSTRAP_REPLICATES,
        ),
        "rq1_statistic": "equal_weight_side_block_rate_dispersion",
        "rq1_stationary_null": (
            "cadence_conditioned_stationary_bootstrap",
            "session_side_calendar_blocks",
        ),
        "rq2_score": (
            "interval_log_loss_primary",
            "positive_improvement_contribution",
        ),
        "rq2_concentration": "max_positive_cell_share_le_0.25",
        "time_bootstrap": (
            "calendar_block_multiplier",
            "row_weighted_estimand",
        ),
        "flow_component_assignment": (
            "one_endpoint_one_component",
            "background_2s_units",
        ),
        "flow_bootstrap": (
            "flow_component_multiplier",
            "row_weighted_component_estimand",
        ),
        "rq3_threshold_source": "training_fold_only",
        "rq3_regime": (
            "debounced_entry_exit",
            "no_posthoc_threshold",
        ),
        "rq3_km_ties": "events_before_censors_at_exact_time",
        "rq3_cluster_bootstrap": (
            "detection_block_multiplier",
            BOOTSTRAP_REPLICATES,
        ),
        "rq3_side_aggregation": "equal_side_bonferroni90",
        "latency_roles": tuple(sorted(contracts.LATENCY_ROLES.items())),
        "classification_precedence": (
            tuple(contracts.ALLOWED_CLASSIFICATIONS),
            (
                "data_quality",
                "rq1",
                "rq2",
                "rq3_6600ms",
                "positive_candidate",
            ),
            contracts.CLAIM_LIMIT,
        ),
        "primary_result_seal": tuple(contracts.PRIMARY_RESULT_FILES),
        "stage4_projection": (
            tuple(sorted(STAGE4_OUTCOMES.items())),
            STAGE4_HEADER_SHA256,
            STAGE4_PROJECTED_FIELDS,
            (
                "grid_boundary",
                "accepted_support_class",
                "identified_event_or_no_event",
            ),
        ),
        "stage4_crosscheck": (
            "primary_seal",
            "build_specific_diagnostic_permit",
            "eight_exact_stage4_reads",
        ),
        "aug07_nonaccess": False,
        "deterministic_build": (
            "byte_identical_primary",
            "byte_identical_diagnostic",
            "primary_bytes_unchanged_post_stage4",
        ),
        "output_schema": (
            tuple(sorted(contracts.CSV_HEADERS)),
            17,
            10,
            14,
            42,
        ),
        "package_tree": (
            tuple(sorted(contracts.EXACT_PACKAGE_FILES)),
            tuple(sorted(contracts.PACKAGE_DIRECTORIES)),
        ),
        "layered_identity": (
            "R_from_17_files",
            "C_from_10_files",
            "E_from_14_files",
            "composite_from_R_C_E",
        ),
        "manifest_self_exclusion": (
            contracts.MANIFEST_FILE,
            len(contracts.EXACT_PACKAGE_FILES) - 1,
        ),
        "atomic_publication": "final_must_not_exist",
        "zero_external_action": False,
    }


def validate_production_contract_state(state: Mapping[str, Any]) -> None:
    expected_keys = set(production_contract_state())
    contracts.require(
        set(state) == expected_keys,
        "H0B_MASTER_FRAMEWORK_MISMATCH",
        "$.production_contract",
        f"missing={sorted(expected_keys - set(state))} "
        f"extra={sorted(set(state) - expected_keys)}",
    )

    def exact(surface: str, expected: Any, code: str) -> None:
        contracts.require(
            state[surface] == expected,
            code,
            f"$.production_contract.{surface}",
            f"expected={expected!r} observed={state[surface]!r}",
        )

    exact("kernel_pin", KERNEL_SOURCE_TREE_SHA256, "H0B_KERNEL_PIN_MISMATCH")
    exact(
        "master_framework_pin",
        (
            PRIMARY_PLAN_SHA256,
            PRIMARY_REVIEW_SHA256,
            DIAGNOSTIC_PLAN_SHA256,
            DIAGNOSTIC_REVIEW_SHA256,
            FRAMEWORK_SHA256,
            MATRIX_SHA256,
        ),
        "H0B_MASTER_FRAMEWORK_MISMATCH",
    )
    exact(
        "accepted_h0a_binding",
        (H0A_TUPLE_SHA256, H0A_COMPOSITE),
        "H0B_H0A_IDENTITY_MISMATCH",
    )
    exact(
        "accepted_latency_binding",
        (LATENCY_COMPOSITE, 6600),
        "H0B_LATENCY_IDENTITY_MISMATCH",
    )
    exact(
        "accepted_tuple_binding",
        (TUPLE_SHA256, TUPLE_COMPOSITE),
        "H0B_TUPLE_IDENTITY_MISMATCH",
    )
    exact(
        "accepted_stage1_4_binding",
        tuple(sorted(STAGE4_OUTCOMES.items())),
        "H0B_DEPENDENCY_IDENTITY_MISMATCH",
    )
    exact(
        "session_roles",
        {
            "formal": ("jul30", "aug04"),
            "aug03": (
                "diagnostic_only",
                False,
                "historical_transfer",
            ),
        },
        "H0B_SESSION_ROLE_MISMATCH",
    )
    exact(
        "underlying_state_boundary",
        (
            "elapsed_session_fraction",
            "elapsed_session_fraction_squared",
            "elapsed_segment_fraction",
            "target_bbo_update_count_1s",
            "target_bbo_no_new_information_fraction_1s",
            "risk_gap_bps",
            "risk_gap_change_50ms_bps",
            "binance_bbo_age_ms",
            "hyperliquid_bbo_age_ms",
            "trailing_basis_residual",
        ),
        "H0B_UNDERLYING_STATE_INFERENCE_FORBIDDEN",
    )
    exact(
        "semantic_source_inventory",
        (
            EXPECTED_SEMANTIC_INVENTORY_SHA256,
            SOURCE_INVENTORY_CONTRACT_SHA256,
        ),
        "H0B_SEMANTIC_INVENTORY_MISMATCH",
    )
    exact(
        "build_envelope",
        ("A", "B"),
        "H0B_BUILD_ENVELOPE_MISMATCH",
    )
    exact(
        "source_schema",
        (
            HYPERLIQUID_HEADER,
            BINANCE_HEADER,
            STAGE4_PROJECTED_FIELDS,
        ),
        "H0B_SOURCE_SCHEMA_MISMATCH",
    )
    exact(
        "source_ordering",
        (
            "event_seq_strictly_increasing",
            "local_ts_ns_nondecreasing",
            "same_timestamp_ordered_by_event_seq",
        ),
        "H0B_SOURCE_ORDERING_MISMATCH",
    )
    exact(
        "guarded_opener",
        (
            "semantic_inventory_only_before_outcome_permit",
            "stage4_only_after_primary_seal_and_diagnostic_permit",
        ),
        "H0B_FORBIDDEN_PATH_ACCESS",
    )
    exact(
        "feature_source_boundary",
        (
            "r0_binance_bookticker",
            "r0_hyperliquid_bbo",
            "accepted_stage2_primary",
            "accepted_stage3_primary",
        ),
        "H0B_FEATURE_SOURCE_BOUNDARY_MISMATCH",
    )
    exact(
        "two_envelope_boundary",
        ("H0B0", "fresh_H0B1", "fresh_H0B1_DIAGNOSTIC"),
        "H0B_OUTCOME_ACCESS_BEFORE_PERMIT",
    )
    exact(
        "outcome_access_permit",
        (
            "skhynix_stage_h0b_outcome_access_permit_v2",
            "A",
            "B",
            "admitted",
            True,
        ),
        "H0B_OUTCOME_PERMIT_MISMATCH",
    )
    exact(
        "support_replay",
        (
            "a266184403a830fc422764e90c6dd48a5c1900af13333246ca7e220d664728df"
        ),
        "H0B_SUPPORT_COMMITMENT_MISMATCH",
    )
    exact(
        "calendar_grid",
        (10_000_000, 50_000_000, 0),
        "H0B_CALENDAR_GRID_MISMATCH",
    )
    exact(
        "side_expansion",
        ("maker_ask_risk", "maker_bid_risk"),
        "H0B_SIDE_PAIR_MISMATCH",
    )
    exact(
        "event_definition",
        (
            "maker_ask_risk:target_bid_gte_reference_ask",
            "maker_bid_risk:target_ask_lte_reference_bid",
        ),
        "H0B_EVENT_DEFINITION_MISMATCH",
    )
    exact(
        "support_class_mapping",
        tuple(sorted(contracts.SUPPORT_DISPOSITIONS.items())),
        "H0B_SUPPORT_CLASS_DISPOSITION_MISMATCH",
    )
    exact(
        "observation_bounds",
        (
            "event:(L,U]",
            "right_censor:S(50ms)",
            "straddle:S(L)_for_L_lt_50ms_lt_U",
        ),
        "H0B_OBSERVATION_BOUND_MISMATCH",
    )
    exact(
        "interval_likelihood",
        ("S_exact(L)-S_exact(U)", 10_000_000, 50_000_000),
        "H0B_INTERVAL_LIKELIHOOD_MISMATCH",
    )
    exact(
        "right_censor_likelihood",
        "S_exact(50ms)",
        "H0B_RIGHT_CENSOR_MISMATCH",
    )
    exact(
        "horizon_straddle",
        "S_exact(L)_without_rounding_or_dropping",
        "H0B_HORIZON_STRADDLE_MISMATCH",
    )
    exact(
        "risk_score",
        "1-product_of_all_five_bin_survivals",
        "H0B_RISK_SCORE_MISMATCH",
    )
    exact(
        "binary_subset",
        ("binary_identification_supported",),
        "H0B_BINARY_SUBSET_MISMATCH",
    )
    exact(
        "h0_features",
        (
            "elapsed_session_fraction",
            "elapsed_session_fraction_squared",
            "elapsed_segment_fraction",
            "target_bbo_update_count_1s",
            "target_bbo_no_new_information_fraction_1s",
        ),
        "H0B_H0_FEATURE_ALLOWLIST_MISMATCH",
    )
    exact(
        "h1_features",
        (
            "risk_gap_bps",
            "risk_gap_change_50ms_bps",
            "binance_bbo_age_ms",
            "hyperliquid_bbo_age_ms",
            "trailing_basis_residual",
        ),
        "H0B_H1_FEATURE_ALLOWLIST_MISMATCH",
    )
    expected_h0_design = (
        "side_maker_ask",
        "z_elapsed_session_fraction",
        "z_elapsed_session_fraction_squared",
        "z_elapsed_segment_fraction",
        "z_target_bbo_update_count_1s",
        "z_target_bbo_no_new_information_fraction_1s",
        "is_missing_elapsed_session_fraction",
        "is_missing_elapsed_session_fraction_squared",
        "is_missing_elapsed_segment_fraction",
        "is_missing_target_bbo_update_count_1s",
        "is_missing_target_bbo_no_new_information_fraction_1s",
    )
    expected_h1_design = (
        *expected_h0_design,
        "z_risk_gap_bps",
        "z_risk_gap_change_50ms_bps",
        "z_binance_bbo_age_ms",
        "z_hyperliquid_bbo_age_ms",
        "z_trailing_basis_residual",
        "is_missing_risk_gap_bps",
        "is_missing_risk_gap_change_50ms_bps",
        "is_missing_binance_bbo_age_ms",
        "is_missing_hyperliquid_bbo_age_ms",
        "is_missing_trailing_basis_residual",
    )
    exact(
        "design_matrix",
        (expected_h0_design, expected_h1_design),
        "H0B_DESIGN_MATRIX_MISMATCH",
    )
    exact(
        "basis_residual",
        "prior_only_ewma",
        "H0B_BASIS_RESIDUAL_MISMATCH",
    )
    exact(
        "missing_value_policy",
        (
            "training_fold_nearest_rank_median",
            "training_fold_iqr_floor_1",
            "explicit_missing_indicator",
        ),
        "H0B_MISSING_VALUE_POLICY_MISMATCH",
    )
    exact(
        "dose_definition",
        ("confirmed_decision_ts", "trailing_500ms_queue_drop_ratio"),
        "H0B_DOSE_RECONSTRUCTION_MISMATCH",
    )
    exact(
        "walk_forward",
        (60, 20, 10, 500_000_000, 500_000_000),
        "H0B_WALK_FORWARD_MISMATCH",
    )
    exact(
        "estimator",
        (
            "piecewise_exponential_discrete_hazard",
            1.0,
            500,
            1e-8,
            1e-10,
        ),
        "H0B_ESTIMATOR_CONTRACT_MISMATCH",
    )
    exact(
        "numeric_seed_conventions",
        (
            "nearest_rank",
            "NumPy Generator(PCG64)",
            8232001,
            8232002,
            8232003,
            8232004,
            2000,
        ),
        "H0B_NUMERIC_CONVENTION_MISMATCH",
    )
    exact(
        "rq1_statistic",
        "equal_weight_side_block_rate_dispersion",
        "H0B_RQ1_STATISTIC_MISMATCH",
    )
    exact(
        "rq1_stationary_null",
        (
            "cadence_conditioned_stationary_bootstrap",
            "session_side_calendar_blocks",
        ),
        "H0B_RQ1_NULL_MISMATCH",
    )
    exact(
        "rq2_score",
        (
            "interval_log_loss_primary",
            "positive_improvement_contribution",
        ),
        "H0B_RQ2_SCORE_MISMATCH",
    )
    exact(
        "rq2_concentration",
        "max_positive_cell_share_le_0.25",
        "H0B_RQ2_CONCENTRATION_MISMATCH",
    )
    exact(
        "time_bootstrap",
        ("calendar_block_multiplier", "row_weighted_estimand"),
        "H0B_TIME_BOOTSTRAP_MISMATCH",
    )
    exact(
        "flow_component_assignment",
        ("one_endpoint_one_component", "background_2s_units"),
        "H0B_FLOW_COMPONENT_MISMATCH",
    )
    exact(
        "flow_bootstrap",
        (
            "flow_component_multiplier",
            "row_weighted_component_estimand",
        ),
        "H0B_FLOW_BOOTSTRAP_MISMATCH",
    )
    exact(
        "rq3_threshold_source",
        "training_fold_only",
        "H0B_RQ3_THRESHOLD_MISMATCH",
    )
    exact(
        "rq3_regime",
        ("debounced_entry_exit", "no_posthoc_threshold"),
        "H0B_RQ3_REGIME_MISMATCH",
    )
    exact(
        "rq3_km_ties",
        "events_before_censors_at_exact_time",
        "H0B_RQ3_KM_MISMATCH",
    )
    exact(
        "rq3_cluster_bootstrap",
        ("detection_block_multiplier", 2000),
        "H0B_RQ3_BOOTSTRAP_MISMATCH",
    )
    exact(
        "rq3_side_aggregation",
        "equal_side_bonferroni90",
        "H0B_RQ3_SIDE_AGGREGATION_MISMATCH",
    )
    exact(
        "latency_roles",
        (
            (25, "legacy_sensitivity"),
            (50, "legacy_sensitivity"),
            (100, "historical_optimistic_sensitivity"),
            (250, "legacy_sensitivity"),
            (500, "legacy_sensitivity"),
            (850, "terminal_observability_normal_path_diagnostic_only"),
            (6600, "measurement_selected_primary"),
        ),
        "H0B_PRIMARY_LATENCY_MISMATCH",
    )
    exact(
        "classification_precedence",
        (
            tuple(contracts.ALLOWED_CLASSIFICATIONS),
            (
                "data_quality",
                "rq1",
                "rq2",
                "rq3_6600ms",
                "positive_candidate",
            ),
            "screening_audit_not_final_signal_or_strategy",
        ),
        "H0B_CLASSIFICATION_MISMATCH",
    )
    exact(
        "primary_result_seal",
        tuple(contracts.PRIMARY_RESULT_FILES),
        "H0B_PRIMARY_SEAL_MISMATCH",
    )
    exact(
        "stage4_projection",
        (
            tuple(sorted(STAGE4_OUTCOMES.items())),
            STAGE4_HEADER_SHA256,
            STAGE4_PROJECTED_FIELDS,
            (
                "grid_boundary",
                "accepted_support_class",
                "identified_event_or_no_event",
            ),
        ),
        "H0B_STAGE4_PROJECTION_MISMATCH",
    )
    exact(
        "stage4_crosscheck",
        (
            "primary_seal",
            "build_specific_diagnostic_permit",
            "eight_exact_stage4_reads",
        ),
        "H0B_STAGE4_OPEN_BEFORE_PRIMARY_SEAL",
    )
    exact(
        "aug07_nonaccess",
        False,
        "H0B_AUG07_ACCESS_FORBIDDEN",
    )
    exact(
        "deterministic_build",
        (
            "byte_identical_primary",
            "byte_identical_diagnostic",
            "primary_bytes_unchanged_post_stage4",
        ),
        "H0B_BUILD_MISMATCH",
    )
    exact(
        "output_schema",
        (tuple(sorted(contracts.CSV_HEADERS)), 17, 10, 14, 42),
        "H0B_OUTPUT_SCHEMA_MISMATCH",
    )
    exact(
        "package_tree",
        (
            tuple(sorted(contracts.EXACT_PACKAGE_FILES)),
            tuple(sorted(contracts.PACKAGE_DIRECTORIES)),
        ),
        "H0B_PACKAGE_TREE_MISMATCH",
    )
    exact(
        "layered_identity",
        (
            "R_from_17_files",
            "C_from_10_files",
            "E_from_14_files",
            "composite_from_R_C_E",
        ),
        "H0B_IDENTITY_BINDING_MISMATCH",
    )
    exact(
        "manifest_self_exclusion",
        (contracts.MANIFEST_FILE, 41),
        "H0B_MANIFEST_SELF_REFERENCE_MISMATCH",
    )
    exact(
        "atomic_publication",
        "final_must_not_exist",
        "PUBLICATION_FINAL_EXISTS",
    )
    exact(
        "zero_external_action",
        False,
        "H0B_EXTERNAL_ACTION_FORBIDDEN",
    )


def validate_session_role_contract(
    formal_sessions: Sequence[str],
    diagnostic_session: tuple[str, str, bool, str],
) -> None:
    contracts.require(
        tuple(formal_sessions) == ("jul30", "aug04")
        and diagnostic_session
        == ("aug03", "diagnostic_only", False, "historical_transfer"),
        "H0B_SESSION_ROLE_MISMATCH",
        "$.session_roles",
        f"formal={tuple(formal_sessions)!r} "
        f"diagnostic={diagnostic_session!r}",
    )


def validate_underlying_state_boundary(features: Sequence[str]) -> None:
    allowed = (
        "elapsed_session_fraction",
        "elapsed_session_fraction_squared",
        "elapsed_segment_fraction",
        "target_bbo_update_count_1s",
        "target_bbo_no_new_information_fraction_1s",
        "risk_gap_bps",
        "risk_gap_change_50ms_bps",
        "binance_bbo_age_ms",
        "hyperliquid_bbo_age_ms",
        "trailing_basis_residual",
    )
    contracts.require(
        tuple(features) == allowed,
        "H0B_UNDERLYING_STATE_INFERENCE_FORBIDDEN",
        "$.underlying_state_boundary",
        f"observed={tuple(features)!r}",
    )


def validate_h0_feature_allowlist(features: Sequence[str]) -> None:
    contracts.require(
        tuple(features)
        == (
            "elapsed_session_fraction",
            "elapsed_session_fraction_squared",
            "elapsed_segment_fraction",
            "target_bbo_update_count_1s",
            "target_bbo_no_new_information_fraction_1s",
        ),
        "H0B_H0_FEATURE_ALLOWLIST_MISMATCH",
        "$.h0_features",
        f"observed={tuple(features)!r}",
    )


def validate_h1_feature_allowlist(features: Sequence[str]) -> None:
    contracts.require(
        tuple(features)
        == (
            "risk_gap_bps",
            "risk_gap_change_50ms_bps",
            "binance_bbo_age_ms",
            "hyperliquid_bbo_age_ms",
            "trailing_basis_residual",
        ),
        "H0B_H1_FEATURE_ALLOWLIST_MISMATCH",
        "$.h1_features",
        f"observed={tuple(features)!r}",
    )


def validate_latency_role_contract(
    roles: Mapping[int, str],
    *,
    primary_latency_ms: int,
    diagnostic_latency_ms: int,
) -> None:
    expected = {
        25: "legacy_sensitivity",
        50: "legacy_sensitivity",
        100: "historical_optimistic_sensitivity",
        250: "legacy_sensitivity",
        500: "legacy_sensitivity",
        850: "terminal_observability_normal_path_diagnostic_only",
        6600: "measurement_selected_primary",
    }
    contracts.require(
        dict(roles) == expected
        and primary_latency_ms == 6600
        and diagnostic_latency_ms == 850,
        "H0B_PRIMARY_LATENCY_MISMATCH",
        "$.latency_roles",
        f"roles={dict(roles)!r} primary={primary_latency_ms} "
        f"diagnostic={diagnostic_latency_ms}",
    )


def validate_source_access(
    relative_path: str,
    *,
    source_role: str,
    phase: str,
) -> None:
    relative = Path(relative_path)
    contracts.require(
        not relative.is_absolute() and ".." not in relative.parts,
        "H0B_FORBIDDEN_PATH_ACCESS",
        relative_path,
        "source path must be repository-relative and confined",
    )
    lowered = relative_path.lower()
    if phase == "feature_read":
        contracts.require(
            "aug07" not in lowered and "0807" not in lowered,
            "H0B_AUG07_ACCESS_FORBIDDEN",
            relative_path,
            "Aug07 event rows are diagnostic metadata only",
        )
        contracts.require(
            source_role
            in {
                "r0_binance_bookticker",
                "r0_hyperliquid_bbo",
                "accepted_stage2_primary",
                "accepted_stage3_primary",
            },
            "H0B_FEATURE_SOURCE_BOUNDARY_MISMATCH",
            relative_path,
            f"source_role={source_role}",
        )
        contracts.require(
            "decision_labels" not in lowered
            and "/outcomes/" not in lowered
            and "stage04" not in lowered,
            "H0B_FEATURE_SOURCE_BOUNDARY_MISMATCH",
            relative_path,
            "future/outcome-bearing feature source is forbidden",
        )
        return
    if phase == "preoutcome_inventory":
        contracts.require(
            "stage04" not in lowered and "/outcomes/" not in lowered,
            "H0B_STAGE4_OPEN_BEFORE_PRIMARY_SEAL",
            relative_path,
            "Stage 4 is forbidden before the primary seal",
        )
        return
    contracts.require(
        phase == "post_primary_seal_stage4"
        and source_role == "accepted_stage4_outcome"
        and relative_path.startswith(
            (
                STAGE4_ROOT.relative_to(REPO_ROOT)
                / "outcomes"
            ).as_posix()
            + "/"
        ),
        "H0B_FORBIDDEN_PATH_ACCESS",
        relative_path,
        f"invalid phase/source_role={phase}/{source_role}",
    )


def validate_external_action_boundary(
    attempted_actions: Sequence[str],
) -> None:
    contracts.require(
        tuple(attempted_actions) == (),
        "H0B_EXTERNAL_ACTION_FORBIDDEN",
        "$.external_actions",
        f"attempted={tuple(attempted_actions)!r}",
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    contracts.require(
        type(value) is dict,
        "H0B_OUTPUT_SCHEMA_MISMATCH",
        str(path),
        "root must be an object",
    )
    return value


def write_json(path: Path, value: Any, *, fsync: bool = False) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(contracts.canonical_json_bytes(value))
    if fsync:
        contracts.fsync_file(path)


def relative_source_path(path: str) -> Path:
    relative = Path(path)
    current = REPO_ROOT / relative
    if current.exists():
        return current
    canonical = CANONICAL_SOURCE_ROOT / relative
    if canonical.exists():
        return canonical
    raise contracts.H0BError(
        "H0B_SEMANTIC_INVENTORY_MISMATCH",
        path,
        "accepted source path is missing",
    )


def exact_csv_header_sha256(path: Path) -> str:
    if path.name.endswith(".csv.gz"):
        opener = gzip.open
    elif path.suffix == ".csv":
        opener = open
    else:
        return ""
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n")
    return hashlib.sha256((header + "\n").encode("utf-8")).hexdigest()


def read_exact_csv(
    path: Path,
    fields: Sequence[str],
    *,
    compressed: bool = False,
) -> Iterable[list[str]]:
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        contracts.require(
            header == list(fields),
            "H0B_SOURCE_SCHEMA_MISMATCH",
            str(path),
            f"expected={list(fields)!r} observed={header!r}",
        )
        for line_number, row in enumerate(reader, 2):
            contracts.require(
                len(row) == len(fields),
                "H0B_SOURCE_SCHEMA_MISMATCH",
                f"{path}:{line_number}",
                f"expected={len(fields)} observed={len(row)}",
            )
            yield row


def validate_dispatch(task_path: Path, matrix_path: Path) -> dict[str, Any]:
    validate_production_contract_state(production_contract_state())
    validate_session_role_contract(
        contracts.FORMAL_SESSIONS,
        ("aug03", "diagnostic_only", False, "historical_transfer"),
    )
    validate_underlying_state_boundary(
        (*contracts.H0_RAW_FEATURES, *contracts.H1_ADDED_RAW_FEATURES)
    )
    validate_h0_feature_allowlist(contracts.H0_RAW_FEATURES)
    validate_h1_feature_allowlist(contracts.H1_ADDED_RAW_FEATURES)
    validate_latency_role_contract(
        contracts.LATENCY_ROLES,
        primary_latency_ms=contracts.PRIMARY_LATENCY_MS,
        diagnostic_latency_ms=contracts.DIAGNOSTIC_LATENCY_MS,
    )
    validate_external_action_boundary(())
    for path, expected, code in (
        (
            PRIMARY_PLAN_PATH,
            PRIMARY_PLAN_SHA256,
            "H0B_MASTER_FRAMEWORK_MISMATCH",
        ),
        (
            PRIMARY_PLAN_REVIEW_PATH,
            PRIMARY_REVIEW_SHA256,
            "H0B_MASTER_FRAMEWORK_MISMATCH",
        ),
        (
            DIAGNOSTIC_PLAN_PATH,
            DIAGNOSTIC_PLAN_SHA256,
            "H0B_MASTER_FRAMEWORK_MISMATCH",
        ),
        (
            DIAGNOSTIC_PLAN_REVIEW_PATH,
            DIAGNOSTIC_REVIEW_SHA256,
            "H0B_MASTER_FRAMEWORK_MISMATCH",
        ),
        (FRAMEWORK_PATH, FRAMEWORK_SHA256, "H0B_MASTER_FRAMEWORK_MISMATCH"),
        (
            SEMANTIC_INVENTORY_PATH,
            EXPECTED_SEMANTIC_INVENTORY_SHA256,
            "H0B_SEMANTIC_INVENTORY_MISMATCH",
        ),
        (
            SOURCE_INVENTORY_CONTRACT_PATH,
            SOURCE_INVENTORY_CONTRACT_SHA256,
            "H0B_SEMANTIC_INVENTORY_MISMATCH",
        ),
        (matrix_path, MATRIX_SHA256, "H0B_MASTER_FRAMEWORK_MISMATCH"),
    ):
        observed = contracts.sha256_file(path)
        contracts.require(
            observed == expected,
            code,
            str(path),
            f"expected={expected} observed={observed}",
        )
    command = [
        sys.executable,
        str(
            REPO_ROOT
            / ".workflow/workflow-kit/validate_research_package_task.py"
        ),
        "--task",
        str(task_path),
        "--matrix",
        str(matrix_path),
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    contracts.require(
        result.returncode == 0,
        "H0B_MASTER_FRAMEWORK_MISMATCH",
        "$.dispatch",
        result.stdout + result.stderr,
    )
    return json.loads(result.stdout)


def runtime_source_inventory(
    root: Path = REPO_ROOT,
) -> list[dict[str, Any]]:
    paths = (
        "examples/hyperliquid/skhynix_stage_h0b.py",
        "examples/hyperliquid/skhynix_stage_h0b_contracts.py",
        "examples/hyperliquid/test_skhynix_stage_h0b.py",
        "examples/hyperliquid/test_skhynix_stage_h0b_package.py",
    )
    return contracts.file_inventory(root, paths)


def runtime_source_tree_sha256(root: Path = REPO_ROOT) -> str:
    return contracts.canonical_json_sha256(runtime_source_inventory(root))


def validate_semantic_inventory() -> tuple[list[dict[str, str]], str]:
    with SEMANTIC_INVENTORY_PATH.open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    fields = contracts.CSV_HEADERS["preoutcome_source_inventory.csv"]
    contracts.require(
        rows and tuple(rows[0]) == fields,
        "H0B_SEMANTIC_INVENTORY_MISMATCH",
        str(SEMANTIC_INVENTORY_PATH),
        "header mismatch",
    )
    observed_order = [
        tuple(row[field] for field in fields)
        for row in rows
    ]
    contracts.require(
        observed_order == sorted(observed_order),
        "H0B_SEMANTIC_INVENTORY_MISMATCH",
        str(SEMANTIC_INVENTORY_PATH),
        "row order mismatch",
    )
    for row in rows:
        validate_source_access(
            row["relative_path"],
            source_role=row["source_role"],
            phase="preoutcome_inventory",
        )
        path = relative_source_path(row["relative_path"])
        contracts.require(
            path.stat().st_size == int(row["bytes"]),
            "H0B_SEMANTIC_INVENTORY_MISMATCH",
            row["relative_path"],
            "byte count mismatch",
        )
        contracts.require(
            contracts.sha256_file(path) == row["sha256"],
            "H0B_SEMANTIC_INVENTORY_MISMATCH",
            row["relative_path"],
            "raw SHA256 mismatch",
        )
        if row["header_sha256"]:
            contracts.require(
                exact_csv_header_sha256(path) == row["header_sha256"],
                "H0B_SOURCE_SCHEMA_MISMATCH",
                row["relative_path"],
                "header SHA256 mismatch",
            )
    identity = contracts.sha256_file(SEMANTIC_INVENTORY_PATH)
    contracts.require(
        identity == EXPECTED_SEMANTIC_INVENTORY_SHA256,
        "H0B_SEMANTIC_INVENTORY_MISMATCH",
        str(SEMANTIC_INVENTORY_PATH),
        f"observed={identity}",
    )
    return rows, identity


def upstream_identity_payloads() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    return (
        read_json(H0A_ROOT / "h0a_manifest.json"),
        read_json(LATENCY_ROOT / "measurement_manifest.json"),
        read_json(TUPLE_ROOT / "supersession_manifest.json"),
        read_json(TUPLE_ROOT / "superseding_primary_tuple.json"),
    )


def validate_upstream_identity_payloads(
    h0a_manifest: Mapping[str, Any],
    latency_manifest: Mapping[str, Any],
    tuple_manifest: Mapping[str, Any],
    tuple_payload: Mapping[str, Any],
) -> None:
    contracts.require(
        h0a_manifest.get("task_id") == "0821T001"
        and h0a_manifest.get("primary_tuple_sha256") == H0A_TUPLE_SHA256
        and h0a_manifest.get("composite_identity") == H0A_COMPOSITE
        and h0a_manifest.get("input_inventory_unchanged") is True,
        "H0B_H0A_IDENTITY_MISMATCH",
        str(H0A_ROOT / "h0a_manifest.json"),
        "accepted H0-A manifest identity mismatch",
    )
    contracts.require(
        latency_manifest.get("task_id") == "0822T002"
        and latency_manifest.get("composite_identity") == LATENCY_COMPOSITE
        and latency_manifest.get("p95_cancel_effective_latency_us") == 6561052
        and latency_manifest.get("recommended_gate_latency_ms") == 6600
        and latency_manifest.get("h0a_tuple_mutated") is False
        and latency_manifest.get("h0b_outcome_accessed") is False,
        "H0B_LATENCY_IDENTITY_MISMATCH",
        str(LATENCY_ROOT / "measurement_manifest.json"),
        "accepted latency identity or recommendation mismatch",
    )
    contracts.require(
        tuple_manifest.get("task_id") == "0823T001"
        and tuple_manifest.get("superseding_tuple_sha256") == TUPLE_SHA256
        and tuple_manifest.get("composite_identity") == TUPLE_COMPOSITE
        and tuple_manifest.get("primary_latency_ms") == 6600
        and tuple_manifest.get("diagnostic_latency_ms") == 850
        and tuple_manifest.get("historical_optimistic_sensitivity_ms") == 100
        and tuple_manifest.get("h0b_outcome_accessed") is False
        and tuple_manifest.get("research_contains_outcome_value") is False
        and tuple_manifest.get("undeclared_change_count") == 0
        and tuple_payload.get("task_id") == "0823T001"
        and tuple_payload.get("gate_latency_ms") == 6600
        and tuple_payload.get("latency_diagnostic_ms") == [850]
        and tuple_payload.get("formal_session_ids") == ["jul30", "aug04"]
        and tuple_payload.get("diagnostic_session_ids") == ["aug03"]
        and tuple_payload.get("supersession_status")
        == "supersedes_latency_only"
        and tuple_payload.get("selection_status") == "selected",
        "H0B_TUPLE_IDENTITY_MISMATCH",
        str(TUPLE_ROOT / "supersession_manifest.json"),
        "accepted tuple supersession identity or role mismatch",
    )


def accepted_binding_rows() -> list[dict[str, Any]]:
    validate_upstream_identity_payloads(*upstream_identity_payloads())
    authorities = (
        (
            "primary_plan_v1",
            PRIMARY_PLAN_PATH,
            PRIMARY_PLAN_SHA256,
        ),
        (
            "primary_plan_review",
            PRIMARY_PLAN_REVIEW_PATH,
            PRIMARY_REVIEW_SHA256,
        ),
        (
            "diagnostic_plan_v2",
            DIAGNOSTIC_PLAN_PATH,
            DIAGNOSTIC_PLAN_SHA256,
        ),
        (
            "diagnostic_plan_review",
            DIAGNOSTIC_PLAN_REVIEW_PATH,
            DIAGNOSTIC_REVIEW_SHA256,
        ),
        (
            "master_framework",
            FRAMEWORK_PATH,
            FRAMEWORK_SHA256,
        ),
        (
            "accepted_h0a_tuple",
            H0A_ROOT / "primary_tuple_freeze.json",
            H0A_TUPLE_SHA256,
        ),
        (
            "accepted_h0a_manifest",
            H0A_ROOT / "h0a_manifest.json",
            None,
        ),
        (
            "accepted_latency_manifest",
            LATENCY_ROOT / "measurement_manifest.json",
            None,
        ),
        (
            "accepted_tuple",
            TUPLE_ROOT / "superseding_primary_tuple.json",
            TUPLE_SHA256,
        ),
        (
            "accepted_tuple_manifest",
            TUPLE_ROOT / "supersession_manifest.json",
            None,
        ),
        (
            "semantic_source_inventory",
            SEMANTIC_INVENTORY_PATH,
            EXPECTED_SEMANTIC_INVENTORY_SHA256,
        ),
        (
            "source_inventory_contract",
            SOURCE_INVENTORY_CONTRACT_PATH,
            SOURCE_INVENTORY_CONTRACT_SHA256,
        ),
    )
    rows = []
    for binding_id, path, expected in authorities:
        observed = contracts.sha256_file(path)
        if expected is not None:
            contracts.require(
                observed == expected,
                "H0B_H0A_IDENTITY_MISMATCH",
                str(path),
                f"expected={expected} observed={observed}",
            )
        rows.append(
            {
                "binding_id": binding_id,
                "authority_path": path.relative_to(REPO_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": observed,
                "status": "accepted",
            }
        )
    return rows


def accepted_input_bindings_payload() -> dict[str, Any]:
    return {
        "schema_version": "skhynix_stage_h0b_accepted_input_bindings_v2",
        "task_id": contracts.TASK_ID,
        "primary_plan_sha256": PRIMARY_PLAN_SHA256,
        "diagnostic_plan_sha256": DIAGNOSTIC_PLAN_SHA256,
        "diagnostic_review_sha256": DIAGNOSTIC_REVIEW_SHA256,
        "surface_matrix_sha256": MATRIX_SHA256,
        "expected_semantic_source_inventory_sha256": (
            EXPECTED_SEMANTIC_INVENTORY_SHA256
        ),
        "bindings": accepted_binding_rows(),
    }


def preoutcome_contract_payload() -> dict[str, Any]:
    likelihood = {
        "bins_ns": [10_000_000] * 5,
        "event": "S_exact(L)-S_exact(U)",
        "right_censor": "S_5",
        "horizon_straddle": "S_exact(L)",
        "floor": contracts.LIKELIHOOD_FLOOR,
    }
    design = {
        "H0": list(contracts.H0_DESIGN_COLUMNS),
        "H1": list(contracts.H1_DESIGN_COLUMNS),
        "side": {"maker_ask_risk": 1.0, "maker_bid_risk": 0.0},
        "ridge_penalty_weight": 1.0,
    }
    walk_forward = {
        "initial_train_blocks": 60,
        "test_blocks": 20,
        "minimum_remainder_blocks": 10,
        "purge_ns": contracts.PURGE_NS,
        "embargo_ns": contracts.EMBARGO_NS,
        "accepted_complete_blocks": {"jul30": 232, "aug04": 119},
    }
    resampling = {
        "replicates": BOOTSTRAP_REPLICATES,
        "seeds": {
            "rq1": PRIMARY_NULL_SEED,
            "time": TIME_BOOTSTRAP_SEED,
            "flow": FLOW_BOOTSTRAP_SEED,
            "rq3": RQ3_BOOTSTRAP_SEED,
        },
        "generator": "NumPy Generator(PCG64)",
    }
    classification = {
        "allowed": list(contracts.ALLOWED_CLASSIFICATIONS),
        "precedence": [
            "data_quality",
            "rq1",
            "rq2",
            "rq3_6600ms",
            "positive_candidate",
        ],
    }
    output = {
        "csv_headers": {
            path: list(header)
            for path, header in sorted(contracts.CSV_HEADERS.items())
        },
        "R": list(contracts.R_FILES),
        "C": list(contracts.C_FILES),
        "E": list(contracts.E_FILES),
        "manifest": contracts.MANIFEST_FILE,
        "directories": list(contracts.PACKAGE_DIRECTORIES),
        "maximum_package_bytes": 134_217_728,
    }
    return {
        "schema_version": "skhynix_stage_h0b_preoutcome_contract_v2",
        "task_id": contracts.TASK_ID,
        "primary_plan_sha256": PRIMARY_PLAN_SHA256,
        "diagnostic_plan_sha256": DIAGNOSTIC_PLAN_SHA256,
        "diagnostic_review_sha256": DIAGNOSTIC_REVIEW_SHA256,
        "surface_matrix_sha256": MATRIX_SHA256,
        "source_inventory_contract_sha256": (
            SOURCE_INVENTORY_CONTRACT_SHA256
        ),
        "likelihood_contract_sha256": contracts.canonical_json_sha256(
            likelihood
        ),
        "design_matrix_contract_sha256": contracts.canonical_json_sha256(
            design
        ),
        "walk_forward_contract_sha256": contracts.canonical_json_sha256(
            walk_forward
        ),
        "resampling_contract_sha256": contracts.canonical_json_sha256(
            resampling
        ),
        "classification_contract_sha256": contracts.canonical_json_sha256(
            classification
        ),
        "output_contract_sha256": contracts.canonical_json_sha256(output),
        "runtime_source_tree_sha256": runtime_source_tree_sha256(),
    }


def replay_support(output_root: Path) -> dict[str, Any]:
    replay = Path(output_root) / "support_replay"
    result = h0a_support.project_support(
        repo_root=REPO_ROOT,
        output_root=replay,
        source_root=CANONICAL_SOURCE_ROOT,
    )
    accepted = H0A_ROOT / "support_projection_commitments.csv"
    observed = replay / "support_projection_commitments.csv"
    accepted_raw = accepted.read_bytes()
    observed_raw = observed.read_bytes()
    contracts.require(
        observed_raw == accepted_raw,
        "H0B_SUPPORT_COMMITMENT_MISMATCH",
        str(observed),
        "support projection commitments differ from accepted H0-A",
    )
    with accepted.open(newline="", encoding="utf-8") as handle:
        row_count = sum(1 for _ in csv.DictReader(handle))
    return {
        "accepted_sha256": contracts.sha256_file(accepted),
        "observed_sha256": contracts.sha256_file(observed),
        "exact_match": True,
        "row_count": row_count,
        "projection": result,
    }


def build_envelope(
    build_root: Path,
    build_label: str,
    semantic_inventory_sha256: str,
    preoutcome_contract_sha256: str,
) -> dict[str, Any]:
    return {
        "build_label": build_label,
        "resolved_build_root": str(Path(build_root).resolve()),
        "runtime_pid": os.getpid(),
        "runtime_source_tree_sha256": runtime_source_tree_sha256(),
        "semantic_source_inventory_sha256": semantic_inventory_sha256,
        "preoutcome_contract_sha256": preoutcome_contract_sha256,
    }


def run_h0b0(
    *,
    task_path: Path,
    matrix_path: Path,
    build_root: Path,
    build_label: str,
) -> dict[str, Any]:
    contracts.require(
        build_label in {"A", "B"},
        "H0B_BUILD_ENVELOPE_MISMATCH",
        "$.build_label",
        build_label,
    )
    root = Path(build_root)
    root.mkdir(parents=True, exist_ok=False)
    validate_dispatch(task_path, matrix_path)
    _, semantic_identity = validate_semantic_inventory()
    bindings = accepted_input_bindings_payload()
    bindings_path = root / "accepted_input_bindings.json"
    write_json(bindings_path, bindings, fsync=True)
    preoutcome = preoutcome_contract_payload()
    preoutcome_path = root / "preoutcome_contract.json"
    write_json(preoutcome_path, preoutcome, fsync=True)
    preoutcome_identity = contracts.sha256_file(preoutcome_path)
    replay = replay_support(root)
    receipt = {
        "schema_version": "skhynix_stage_h0b_support_replay_receipt_v1",
        "task_id": contracts.TASK_ID,
        "build_label": build_label,
        "accepted_h0a_commitments_sha256": replay["accepted_sha256"],
        "observed_h0a_commitments_sha256": replay["observed_sha256"],
        "exact_commitment_match": replay["exact_match"],
        "forbidden_outcome_access_count": 0,
        "replay_row_count": replay["row_count"],
    }
    receipt_path = root / "support_replay_receipt.json"
    write_json(receipt_path, receipt, fsync=True)
    inventory_path = root / "preoutcome_source_inventory.csv"
    inventory_path.write_bytes(SEMANTIC_INVENTORY_PATH.read_bytes())
    contracts.fsync_file(inventory_path)
    envelope = build_envelope(
        root,
        build_label,
        semantic_identity,
        preoutcome_identity,
    )
    envelope_identity = contracts.canonical_json_sha256(envelope)
    permit = {
        "schema_version": "skhynix_stage_h0b_outcome_access_permit_v2",
        "task_id": contracts.TASK_ID,
        "build_label": build_label,
        "status": "admitted",
        "fsynced": True,
        "primary_plan_sha256": PRIMARY_PLAN_SHA256,
        "diagnostic_plan_sha256": DIAGNOSTIC_PLAN_SHA256,
        "diagnostic_review_sha256": DIAGNOSTIC_REVIEW_SHA256,
        "surface_matrix_sha256": MATRIX_SHA256,
        "runtime_source_tree_sha256": runtime_source_tree_sha256(),
        "preoutcome_contract_sha256": preoutcome_identity,
        "source_inventory_contract_sha256": (
            SOURCE_INVENTORY_CONTRACT_SHA256
        ),
        "semantic_source_inventory_sha256": semantic_identity,
        "build_envelope": envelope,
        "build_envelope_sha256": envelope_identity,
        "support_replay_receipt_sha256": contracts.sha256_file(receipt_path),
        "accepted_input_bindings_sha256": contracts.sha256_file(bindings_path),
    }
    permit_path = root / "outcome_access_permit.json"
    write_json(permit_path, permit, fsync=True)
    ledger = {
        "schema_version": "skhynix_stage_h0b_outcome_access_ledger_v1",
        "task_id": contracts.TASK_ID,
        "build_label": build_label,
        "events": [
            {
                "sequence": 1,
                "process_role": "H0B0",
                "phase": "preoutcome",
                "relative_path": "preoutcome_source_inventory.csv",
                "access_kind": "identity_and_header_validation",
                "bytes_read": inventory_path.stat().st_size,
                "permit_sha256": "",
                "admitted": True,
            },
            {
                "sequence": 2,
                "process_role": "H0B0",
                "phase": "permit",
                "relative_path": "outcome_access_permit.json",
                "access_kind": "fsync_write",
                "bytes_read": 0,
                "permit_sha256": contracts.sha256_file(permit_path),
                "admitted": True,
            },
        ],
    }
    ledger_path = root / "outcome_access_ledger.json"
    write_json(ledger_path, ledger, fsync=True)
    contracts.fsync_directory(root)
    return {
        "verified": True,
        "task_id": contracts.TASK_ID,
        "build_label": build_label,
        "build_root": str(root.resolve()),
        "semantic_source_inventory_sha256": semantic_identity,
        "preoutcome_contract_sha256": preoutcome_identity,
        "build_envelope_sha256": envelope_identity,
        "permit_sha256": contracts.sha256_file(permit_path),
        "support_replay_exact": True,
        "outcome_predicate_evaluated": False,
        "stage4_bytes_opened": False,
    }


def validate_permit(build_root: Path) -> dict[str, Any]:
    root = Path(build_root)
    permit_path = root / "outcome_access_permit.json"
    permit = read_json(permit_path)
    expected_keys = {
        "schema_version",
        "task_id",
        "build_label",
        "status",
        "fsynced",
        "primary_plan_sha256",
        "diagnostic_plan_sha256",
        "diagnostic_review_sha256",
        "surface_matrix_sha256",
        "runtime_source_tree_sha256",
        "preoutcome_contract_sha256",
        "source_inventory_contract_sha256",
        "semantic_source_inventory_sha256",
        "build_envelope",
        "build_envelope_sha256",
        "support_replay_receipt_sha256",
        "accepted_input_bindings_sha256",
    }
    contracts.require(
        set(permit) == expected_keys,
        "H0B_OUTCOME_PERMIT_MISMATCH",
        str(permit_path),
        f"keys={sorted(permit)}",
    )
    contracts.require(
        permit["status"] == "admitted"
        and permit["fsynced"] is True
        and permit["task_id"] == contracts.TASK_ID
        and permit["primary_plan_sha256"] == PRIMARY_PLAN_SHA256
        and permit["diagnostic_plan_sha256"] == DIAGNOSTIC_PLAN_SHA256
        and permit["diagnostic_review_sha256"]
        == DIAGNOSTIC_REVIEW_SHA256
        and permit["surface_matrix_sha256"] == MATRIX_SHA256
        and permit["runtime_source_tree_sha256"] == runtime_source_tree_sha256()
        and permit["source_inventory_contract_sha256"]
        == SOURCE_INVENTORY_CONTRACT_SHA256
        and permit["semantic_source_inventory_sha256"]
        == EXPECTED_SEMANTIC_INVENTORY_SHA256,
        "H0B_OUTCOME_PERMIT_MISMATCH",
        str(permit_path),
        "frozen permit scalar mismatch",
    )
    envelope = permit["build_envelope"]
    contracts.require(
        contracts.canonical_json_sha256(envelope)
        == permit["build_envelope_sha256"]
        and Path(envelope["resolved_build_root"]).resolve() == root.resolve()
        and envelope["build_label"] == permit["build_label"]
        and envelope["runtime_source_tree_sha256"]
        == runtime_source_tree_sha256()
        and envelope["semantic_source_inventory_sha256"]
        == EXPECTED_SEMANTIC_INVENTORY_SHA256,
        "H0B_OUTCOME_PERMIT_MISMATCH",
        "$.build_envelope",
        "build-specific envelope mismatch",
    )
    receipt = root / "support_replay_receipt.json"
    bindings = root / "accepted_input_bindings.json"
    preoutcome = root / "preoutcome_contract.json"
    contracts.require(
        contracts.sha256_file(receipt)
        == permit["support_replay_receipt_sha256"]
        and contracts.sha256_file(bindings)
        == permit["accepted_input_bindings_sha256"]
        and contracts.sha256_file(preoutcome)
        == permit["preoutcome_contract_sha256"],
        "H0B_OUTCOME_PERMIT_MISMATCH",
        "$.permit.bindings",
        "permit-bound file mismatch",
    )
    return permit


def hostile_stage4_ledger_mutation() -> None:
    permit_sha = "a" * 64
    events = [
        {
            "sequence": 1,
            "process_role": "H0B1_DIAGNOSTIC_PERMIT",
            "phase": "post_primary_seal_permit",
            "relative_path": "stage4_diagnostic_permit.json",
            "access_kind": "fsync_write",
            "bytes_read": 0,
            "permit_sha256": permit_sha,
            "admitted": True,
        },
        {
            "sequence": 2,
            "process_role": "H0B1_DIAGNOSTIC_PERMIT",
            "phase": "diagnostic_shadow",
            "relative_path": "shadow_permit.json",
            "access_kind": "fsync_write",
            "bytes_read": 0,
            "permit_sha256": permit_sha,
            "admitted": True,
        },
    ]
    for sequence, relative in enumerate(sorted(STAGE4_OUTCOMES), start=3):
        events.append(
            {
                "sequence": sequence,
                "process_role": "H0B1_DIAGNOSTIC",
                "phase": "post_primary_seal_stage4",
                "relative_path": (
                    STAGE4_ROOT.relative_to(REPO_ROOT)
                    / relative
                ).as_posix(),
                "access_kind": "exact_11_field_projection",
                "bytes_read": 1,
                "permit_sha256": permit_sha,
                "admitted": True,
            }
        )
    validate_stage4_access_ledger(
        {
            "schema_version": (
                "skhynix_stage_h0b_outcome_access_ledger_v1"
            ),
            "task_id": contracts.TASK_ID,
            "build_label": "A",
            "events": events,
        },
        build_label="A",
        diagnostic_permit_sha256=permit_sha,
    )


def hostile_package_tree_mutation() -> None:
    with tempfile.TemporaryDirectory(prefix="0823T002-package-tree-") as raw:
        root = Path(raw)
        for directory in contracts.PACKAGE_DIRECTORIES:
            (root / directory).mkdir()
        for relative in contracts.EXACT_PACKAGE_FILES:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
        (root / "unexpected.txt").write_text("mutation\n", encoding="ascii")
        contracts.validate_exact_package_tree(root)


def hostile_publication_overwrite_mutation() -> None:
    with tempfile.TemporaryDirectory(prefix="0823T002-publication-") as raw:
        final = Path(raw) / "final"
        final.mkdir()
        require_publication_target_absent(final)


def hostile_contract_state_mutation(surface_id: str, observed: Any) -> None:
    state = production_contract_state()
    state[surface_id] = observed
    validate_production_contract_state(state)


def hostile_upstream_identity_mutation(surface_id: str) -> None:
    h0a_manifest, latency_manifest, tuple_manifest, tuple_payload = (
        upstream_identity_payloads()
    )
    if surface_id == "accepted_h0a_binding":
        h0a_manifest["composite_identity"] = "0" * 64
    elif surface_id == "accepted_latency_binding":
        latency_manifest["recommended_gate_latency_ms"] = 850
    elif surface_id == "accepted_tuple_binding":
        tuple_manifest["primary_latency_ms"] = 850
    else:
        raise AssertionError(surface_id)
    validate_upstream_identity_payloads(
        h0a_manifest,
        latency_manifest,
        tuple_manifest,
        tuple_payload,
    )


def hostile_source_schema_mutation() -> None:
    with tempfile.TemporaryDirectory(prefix="0823T002-source-schema-") as raw:
        path = Path(raw) / "mutated.csv.gz"
        header = (*HYPERLIQUID_HEADER[:-1], "mutated_column")
        path.write_bytes(
            contracts.deterministic_gzip(
                (",".join(header) + "\n").encode("ascii")
            )
        )
        list(
            read_exact_csv(
                path,
                HYPERLIQUID_HEADER,
                compressed=True,
            )
        )


def hostile_source_ordering_mutation() -> None:
    with tempfile.TemporaryDirectory(prefix="0823T002-source-order-") as raw:
        path = Path(raw) / "mutated.csv.gz"
        rows = []
        for sequence, timestamp in ((2, 100), (1, 100)):
            row = {field: "" for field in HYPERLIQUID_HEADER}
            row.update(
                {
                    "segment_id": "segment_0001",
                    "event_seq": str(sequence),
                    "local_ts_ns": str(timestamp),
                    "event_type": "bbo",
                    "bid_px": "100",
                    "ask_px": "101",
                }
            )
            rows.append(row)
        path.write_bytes(
            contracts.deterministic_gzip(
                contracts.csv_bytes(rows, HYPERLIQUID_HEADER)
            )
        )
        load_quote_events(
            path,
            header=HYPERLIQUID_HEADER,
            event_type="bbo",
            segment_id="segment_0001",
        )


def hostile_design_matrix_mutation() -> None:
    contracts.transform_design(
        np.zeros((1, len(contracts.H0_RAW_FEATURES)), dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        (),
        model="H2",
    )


def hostile_missing_value_mutation() -> None:
    contracts.fit_feature_scales(
        np.full((1, 1), np.nan, dtype=np.float64),
        ("missing_feature",),
    )


def hostile_walk_forward_mutation() -> None:
    contracts.build_walk_forward_folds(tuple(range(69, -1, -1)))


def hostile_flow_component_mutation() -> None:
    component_a = FlowComponent("a", 100, 200)
    component_b = FlowComponent("b", 150, 250)
    flow_units_for_grid(
        session="jul30",
        segment_id="segment_0001",
        epoch="segment_0001:epoch_0",
        grid=np.asarray([175], dtype=np.int64),
        components={
            ("jul30", "segment_0001", "segment_0001:epoch_0"): (
                component_a,
                component_b,
            )
        },
    )


def hostile_classification_mutation() -> None:
    contracts.classify_primary(
        {
            "jul30": {
                "data_quality": True,
                "rq1": True,
                "rq2": True,
                "rq3": True,
            },
            "aug03": {
                "data_quality": True,
                "rq1": True,
                "rq2": True,
                "rq3": True,
            },
            "aug04": {
                "data_quality": True,
                "rq1": True,
                "rq2": True,
                "rq3": True,
            },
        }
    )


def hostile_deterministic_build_mutation() -> None:
    with tempfile.TemporaryDirectory(prefix="0823T002-build-compare-") as raw:
        left = Path(raw) / "a"
        right = Path(raw) / "b"
        left.mkdir()
        right.mkdir()
        (left / "result.csv").write_bytes(b"build-a\n")
        (right / "result.csv").write_bytes(b"build-b\n")
        compare_build_files(left, right, ("result.csv",))


def negative_case(surface_id: str, expected_code: str) -> None:
    matrix = read_json(MATRIX_PATH)
    surfaces = {
        surface["surface_id"]: surface
        for surface in matrix["surfaces"]
    }
    contracts.require(
        surface_id in surfaces,
        "H0B_MASTER_FRAMEWORK_MISMATCH",
        "$.negative_case",
        f"unknown surface={surface_id}",
    )
    surface = surfaces[surface_id]
    declared_code = surface["negative_mutations"][0][
        "expected_error_code"
    ]
    declared_codes = {
        item["negative_mutations"][0]["expected_error_code"]
        for item in surfaces.values()
    }
    contracts.require(
        declared_code == expected_code,
        "H0B_MASTER_FRAMEWORK_MISMATCH",
        "$.negative_case",
        f"surface={surface_id} expected={expected_code}",
    )
    contracts.require(
        HOSTILE_FAIL_OPEN_SENTINEL not in declared_codes,
        "H0B_MASTER_FRAMEWORK_MISMATCH",
        "$.negative_case.fail_open_sentinel",
        HOSTILE_FAIL_OPEN_SENTINEL,
    )

    mutation_checks = {
        "kernel_pin": lambda: hostile_contract_state_mutation(
            "kernel_pin", "0" * 64
        ),
        "master_framework_pin": lambda: hostile_contract_state_mutation(
            "master_framework_pin",
            (
                PRIMARY_PLAN_SHA256,
                PRIMARY_REVIEW_SHA256,
                PRIMARY_PLAN_SHA256,
                PRIMARY_REVIEW_SHA256,
                FRAMEWORK_SHA256,
                MATRIX_SHA256,
            ),
        ),
        "accepted_h0a_binding": lambda: hostile_upstream_identity_mutation(
            "accepted_h0a_binding"
        ),
        "accepted_latency_binding": lambda: (
            hostile_upstream_identity_mutation("accepted_latency_binding")
        ),
        "accepted_tuple_binding": lambda: hostile_upstream_identity_mutation(
            "accepted_tuple_binding"
        ),
        "accepted_stage1_4_binding": lambda: hostile_contract_state_mutation(
            "accepted_stage1_4_binding",
            tuple(sorted(STAGE4_OUTCOMES.items()))[:-1],
        ),
        "session_roles": lambda: validate_session_role_contract(
            ("jul30", "aug03", "aug04"),
            ("aug03", "formal", True, "historical_transfer"),
        ),
        "underlying_state_boundary": lambda: (
            validate_underlying_state_boundary(
            (
                *contracts.H0_RAW_FEATURES,
                *contracts.H1_ADDED_RAW_FEATURES,
                "krx_underlying_state",
                )
            )
        ),
        "semantic_source_inventory": lambda: hostile_contract_state_mutation(
            "semantic_source_inventory",
            (str(REPO_ROOT), SOURCE_INVENTORY_CONTRACT_SHA256),
        ),
        "build_envelope": lambda: hostile_contract_state_mutation(
            "build_envelope", ("A", "A")
        ),
        "source_schema": hostile_source_schema_mutation,
        "source_ordering": hostile_source_ordering_mutation,
        "guarded_opener": lambda: validate_source_access(
            "../forbidden.csv",
            source_role="accepted_stage2_primary",
            phase="feature_read",
        ),
        "feature_source_boundary": lambda: validate_source_access(
            "local_live_analysis/alignment/decision_labels.csv",
            source_role="alignment_decision_labels",
            phase="feature_read",
        ),
        "two_envelope_boundary": lambda: hostile_contract_state_mutation(
            "two_envelope_boundary",
            ("H0B1", "H0B1", "H0B1_DIAGNOSTIC"),
        ),
        "outcome_access_permit": lambda: hostile_contract_state_mutation(
            "outcome_access_permit",
            (
                "skhynix_stage_h0b_outcome_access_permit_v2",
                "A",
                "A",
                "admitted",
                True,
            ),
        ),
        "support_replay": lambda: hostile_contract_state_mutation(
            "support_replay", "0" * 64
        ),
        "calendar_grid": lambda: hostile_contract_state_mutation(
            "calendar_grid", (contracts.GRID_NS, contracts.HORIZON_NS, 1)
        ),
        "side_expansion": lambda: hostile_contract_state_mutation(
            "side_expansion", contracts.SIDES[:1]
        ),
        "event_definition": lambda: hostile_contract_state_mutation(
            "event_definition",
            (
                "maker_ask_risk:target_ask_lte_reference_bid",
                "maker_bid_risk:target_bid_gte_reference_ask",
            ),
        ),
        "support_class_mapping": lambda: hostile_contract_state_mutation(
            "support_class_mapping",
            tuple(
                sorted(
                    {
                        **contracts.SUPPORT_DISPOSITIONS,
                        "interval_likelihood_only_supported": (
                            "include_primary_and_binary"
                        ),
                    }.items()
                )
            ),
        ),
        "observation_bounds": lambda: hostile_contract_state_mutation(
            "observation_bounds",
            (
                "event:[L,U]",
                "right_censor:event_at_zero",
                "straddle:dropped",
            ),
        ),
        "interval_likelihood": lambda: contracts.likelihood_and_loss(
            np.full((1, 5), 0.1),
            np.asarray([9], dtype=np.int8),
            np.asarray([0.0]),
            np.asarray([contracts.HORIZON_NS], dtype=np.float64),
        ),
        "right_censor_likelihood": lambda: hostile_contract_state_mutation(
            "right_censor_likelihood", "binary_no_event_at_0ms"
        ),
        "horizon_straddle": lambda: contracts.likelihood_and_loss(
            np.full((1, 5), 0.1),
            np.asarray([3], dtype=np.int8),
            np.asarray([50_000_000.0]),
            np.asarray([60_000_000.0]),
        ),
        "risk_score": lambda: hostile_contract_state_mutation(
            "risk_score", "single_bin_hazard"
        ),
        "binary_subset": lambda: hostile_contract_state_mutation(
            "binary_subset",
            (
                "binary_identification_supported",
                "interval_likelihood_only_supported",
            ),
        ),
        "h0_features": lambda: validate_h0_feature_allowlist(
            (*contracts.H0_RAW_FEATURES, "future_return"),
        ),
        "h1_features": lambda: validate_h1_feature_allowlist(
            (*contracts.H1_ADDED_RAW_FEATURES, "post_horizon_outcome"),
        ),
        "design_matrix": hostile_design_matrix_mutation,
        "basis_residual": lambda: hostile_contract_state_mutation(
            "basis_residual", "full_session_mean"
        ),
        "missing_value_policy": hostile_missing_value_mutation,
        "dose_definition": lambda: hostile_contract_state_mutation(
            "dose_definition", ("shock_ts", "forward_500ms_queue_drop_ratio")
        ),
        "walk_forward": hostile_walk_forward_mutation,
        "estimator": lambda: hostile_contract_state_mutation(
            "estimator",
            (
                "piecewise_exponential_discrete_hazard",
                0.1,
                OPTIMIZER_MAX_ITERATIONS,
                OPTIMIZER_GRADIENT_TOLERANCE,
                OPTIMIZER_PARAMETER_TOLERANCE,
            ),
        ),
        "numeric_seed_conventions": lambda: contracts.nearest_rank(
            np.asarray([], dtype=np.float64), 0.5
        ),
        "rq1_statistic": lambda: hostile_contract_state_mutation(
            "rq1_statistic", "pooled_side_rows"
        ),
        "rq1_stationary_null": lambda: hostile_contract_state_mutation(
            "rq1_stationary_null",
            (
                "fixed_disjoint_microblock_shuffle",
                "support_source_only_stratum",
            ),
        ),
        "rq2_score": lambda: hostile_contract_state_mutation(
            "rq2_score",
            ("best_metric_posthoc", "net_improvement_contribution"),
        ),
        "rq2_concentration": lambda: hostile_contract_state_mutation(
            "rq2_concentration", "net_absolute_cell_share_le_0.50"
        ),
        "time_bootstrap": lambda: hostile_contract_state_mutation(
            "time_bootstrap", ("row_iid_bootstrap", "equal_unit_estimand")
        ),
        "flow_component_assignment": hostile_flow_component_mutation,
        "flow_bootstrap": lambda: hostile_contract_state_mutation(
            "flow_bootstrap",
            ("flow_component_multiplier", "equal_component_estimand"),
        ),
        "rq3_threshold_source": lambda: hostile_contract_state_mutation(
            "rq3_threshold_source", "oof_pooled_side"
        ),
        "rq3_regime": lambda: hostile_contract_state_mutation(
            "rq3_regime", ("one_tick_switch", "posthoc_threshold")
        ),
        "rq3_km_ties": lambda: contracts.kaplan_meier_median(
            np.asarray([-1.0]),
            np.asarray([False]),
        ),
        "rq3_cluster_bootstrap": lambda: hostile_contract_state_mutation(
            "rq3_cluster_bootstrap", ("greenwood_independent_regime", 2000)
        ),
        "rq3_side_aggregation": lambda: hostile_contract_state_mutation(
            "rq3_side_aggregation", "pooled_regimes"
        ),
        "latency_roles": lambda: validate_latency_role_contract(
            {
                **contracts.LATENCY_ROLES,
                850: "measurement_selected_primary",
                6600: "diagnostic_rescuable",
            },
            primary_latency_ms=850,
            diagnostic_latency_ms=6600,
        ),
        "classification_precedence": hostile_classification_mutation,
        "primary_result_seal": lambda: hostile_contract_state_mutation(
            "primary_result_seal",
            contracts.PRIMARY_RESULT_FILES[:-1],
        ),
        "stage4_projection": lambda: landmark_status_with_precedence(
            outside_grid=False,
            support_class="unknown_support_class",
        ),
        "stage4_crosscheck": hostile_stage4_ledger_mutation,
        "aug07_nonaccess": lambda: validate_source_access(
            "local_live_analysis/aug07/events.csv.gz",
            source_role="r0_hyperliquid_bbo",
            phase="feature_read",
        ),
        "deterministic_build": hostile_deterministic_build_mutation,
        "output_schema": lambda: contracts.csv_bytes(
            [{"field": "value", "unexpected": "mutation"}],
            ("field",),
        ),
        "package_tree": hostile_package_tree_mutation,
        "layered_identity": lambda: hostile_contract_state_mutation(
            "layered_identity",
            (
                "R_from_17_files",
                "reuse_old_C",
                "reuse_old_E",
                "composite_from_mixed_layers",
            ),
        ),
        "manifest_self_exclusion": lambda: hostile_contract_state_mutation(
            "manifest_self_exclusion", ("", 42)
        ),
        "atomic_publication": hostile_publication_overwrite_mutation,
        "zero_external_action": lambda: validate_external_action_boundary(
            ("network",)
        ),
    }
    contracts.require(
        set(mutation_checks) == set(surfaces),
        "H0B_MASTER_FRAMEWORK_MISMATCH",
        "$.negative_case.universe",
        f"missing={sorted(set(surfaces) - set(mutation_checks))} "
        f"extra={sorted(set(mutation_checks) - set(surfaces))}",
    )
    mutation_checks[surface_id]()
    raise contracts.H0BError(
        HOSTILE_FAIL_OPEN_SENTINEL,
        surface_id,
        "targeted mutation failed open",
    )


def hostile_preflight(
    *,
    task_path: Path,
    matrix_path: Path,
    output: Path,
    write_surface_evidence: bool = True,
) -> dict[str, Any]:
    dispatch = validate_dispatch(task_path, matrix_path)
    matrix = read_json(matrix_path)
    current_rows = []
    frozen_rows = []
    with tempfile.TemporaryDirectory(prefix="0823T002-frozen-runtime-") as raw:
        frozen_repo = Path(raw) / "repo"
        frozen = frozen_repo / "examples/hyperliquid"
        frozen.mkdir(parents=True)
        frozen_matrix = (
            frozen_repo
            / ".workflow/contracts/0823T002-surface-matrix.json"
        )
        frozen_matrix.parent.mkdir(parents=True)
        shutil.copy2(matrix_path, frozen_matrix)
        for source in (
            REPO_ROOT / "examples/hyperliquid/skhynix_stage_h0b.py",
            REPO_ROOT / "examples/hyperliquid/skhynix_stage_h0b_contracts.py",
            REPO_ROOT / "examples/hyperliquid/test_skhynix_stage_h0b.py",
            REPO_ROOT
            / "examples/hyperliquid/test_skhynix_stage_h0b_package.py",
        ):
            shutil.copy2(source, frozen / source.name)
        for source in (
            H0A_ROOT / "h0a_manifest.json",
            LATENCY_ROOT / "measurement_manifest.json",
            TUPLE_ROOT / "supersession_manifest.json",
            TUPLE_ROOT / "superseding_primary_tuple.json",
        ):
            destination = frozen_repo / source.relative_to(REPO_ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        for surface in matrix["surfaces"]:
            mutation = surface["negative_mutations"][0]
            surface_id = surface["surface_id"]
            expected = mutation["expected_error_code"]
            try:
                negative_case(surface_id, expected)
            except contracts.H0BError as exc:
                observed = exc.code
            current_rows.append(
                {
                    "mutation_id": mutation["mutation_id"],
                    "expected_error_code": expected,
                    "error_code": observed,
                }
            )
            command = [
                sys.executable,
                str(frozen / "skhynix_stage_h0b.py"),
                "negative-case",
                "--surface",
                surface_id,
                "--expected-code",
                expected,
            ]
            result = subprocess.run(
                command,
                cwd=frozen,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={
                    **os.environ,
                    "PYTHONPATH": os.pathsep.join(
                        (
                            str(frozen),
                            str(REPO_ROOT / "examples/hyperliquid"),
                        )
                    ),
                },
            )
            contracts.require(
                result.stdout.strip() != "",
                "H0B_BUILD_MISMATCH",
                f"$.frozen_negative.{surface_id}",
                result.stderr,
            )
            payload = json.loads(result.stdout)
            frozen_rows.append(
                {
                    "mutation_id": mutation["mutation_id"],
                    "expected_error_code": expected,
                    "error_code": payload["error"]["code"],
                }
            )
        frozen_runtime_source_tree_sha256 = runtime_source_tree_sha256(
            frozen_repo
        )
    fail_open_count = sum(
        row["expected_error_code"] != row["error_code"]
        for row in (*current_rows, *frozen_rows)
    )
    receipt = {
        "schema_version": "skhynix_stage_h0b_hostile_preflight_v2",
        "task_id": contracts.TASK_ID,
        "dispatch": dispatch,
        "runtime_source_tree_sha256": runtime_source_tree_sha256(),
        "frozen_runtime_source_tree_sha256": (
            frozen_runtime_source_tree_sha256
        ),
        "surface_contract": current_rows,
        "frozen_surface_contract": frozen_rows,
        "current_negative_mutation_count": len(current_rows),
        "frozen_negative_mutation_count": len(frozen_rows),
        "fail_open_count": fail_open_count,
        "outcome_predicate_evaluated": False,
        "stage4_bytes_opened": False,
        "network_private_order_cancel_live_access": False,
    }
    contracts.require(
        fail_open_count == 0,
        "H0B_EXTERNAL_ACTION_FORBIDDEN",
        "$.hostile_preflight",
        f"fail_open_count={fail_open_count}",
    )
    write_json(output, receipt)
    if write_surface_evidence:
        for surface, current, frozen in zip(
            matrix["surfaces"],
            current_rows,
            frozen_rows,
        ):
            for artifact in surface["artifacts"]:
                if not artifact["path"].startswith(
                    ".workflow/reports/0823T002-surface-"
                ):
                    continue
                write_json(
                    REPO_ROOT / artifact["path"],
                    {
                        "schema_version": (
                            "skhynix_stage_h0b_surface_evidence_v1"
                        ),
                        "task_id": contracts.TASK_ID,
                        "surface_id": surface["surface_id"],
                        "mutation_id": current["mutation_id"],
                        "current_error_code": current["error_code"],
                        "frozen_error_code": frozen["error_code"],
                        "expected_error_code": current[
                            "expected_error_code"
                        ],
                        "current_and_frozen_rejected": True,
                        "outcome_predicate_evaluated": False,
                        "stage4_bytes_opened": False,
                    },
                )
    return receipt


def validate_hostile_preflight_receipt(
    path: Path,
    *,
    matrix_path: Path,
    expected_dispatch: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = read_json(path)
    expected_keys = {
        "schema_version",
        "task_id",
        "dispatch",
        "runtime_source_tree_sha256",
        "frozen_runtime_source_tree_sha256",
        "surface_contract",
        "frozen_surface_contract",
        "current_negative_mutation_count",
        "frozen_negative_mutation_count",
        "fail_open_count",
        "outcome_predicate_evaluated",
        "stage4_bytes_opened",
        "network_private_order_cancel_live_access",
    }
    contracts.require(
        set(receipt) == expected_keys,
        "H0B_OUTCOME_ACCESS_BEFORE_PERMIT",
        str(path),
        f"missing={sorted(expected_keys - set(receipt))} "
        f"extra={sorted(set(receipt) - expected_keys)}",
    )
    contracts.require(
        receipt["schema_version"]
        == "skhynix_stage_h0b_hostile_preflight_v2"
        and receipt["task_id"] == contracts.TASK_ID
        and receipt["dispatch"] == dict(expected_dispatch)
        and receipt["runtime_source_tree_sha256"]
        == runtime_source_tree_sha256(),
        "H0B_OUTCOME_ACCESS_BEFORE_PERMIT",
        str(path),
        "hostile receipt identity or runtime binding mismatch",
    )
    contracts.require(
        receipt["frozen_runtime_source_tree_sha256"]
        == receipt["runtime_source_tree_sha256"],
        "H0B_OUTCOME_ACCESS_BEFORE_PERMIT",
        str(path),
        "frozen hostile runtime does not match current runtime inventory",
    )
    matrix = read_json(matrix_path)
    declared = [
        {
            "mutation_id": surface["negative_mutations"][0]["mutation_id"],
            "expected_error_code": surface["negative_mutations"][0][
                "expected_error_code"
            ],
            "error_code": surface["negative_mutations"][0][
                "expected_error_code"
            ],
        }
        for surface in matrix["surfaces"]
    ]
    contracts.require(
        receipt["surface_contract"] == declared
        and receipt["frozen_surface_contract"] == declared,
        "H0B_OUTCOME_ACCESS_BEFORE_PERMIT",
        str(path),
        "current and frozen hostile contracts must exactly match dispatch order",
    )
    count = len(declared)
    fail_open_count = sum(
        row["expected_error_code"] != row["error_code"]
        for row in (
            *receipt["surface_contract"],
            *receipt["frozen_surface_contract"],
        )
    )
    contracts.require(
        receipt["current_negative_mutation_count"] == count
        and receipt["frozen_negative_mutation_count"] == count
        and receipt["fail_open_count"] == fail_open_count
        and fail_open_count == 0,
        "H0B_OUTCOME_ACCESS_BEFORE_PERMIT",
        str(path),
        "hostile mutation counts or fail-open count mismatch",
    )
    contracts.require(
        receipt["outcome_predicate_evaluated"] is False
        and receipt["stage4_bytes_opened"] is False
        and receipt["network_private_order_cancel_live_access"] is False,
        "H0B_EXTERNAL_ACTION_FORBIDDEN",
        str(path),
        "hostile preflight crossed a prohibited execution boundary",
    )
    return receipt


@dataclass(frozen=True)
class QuoteEvents:
    local_ts_ns: np.ndarray
    event_seq: np.ndarray
    bid_px: np.ndarray
    ask_px: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class FlowComponent:
    component_id: str
    start_ns: int
    end_ns: int


@dataclass
class SessionDataset:
    session: str
    segment_ids: tuple[str, ...]
    grid_ts_ns: np.ndarray
    segment_code: np.ndarray
    block_start_ns: np.ndarray
    h0_raw: np.ndarray
    h1_added_raw: np.ndarray
    dose: np.ndarray
    branch: np.ndarray
    lower_elapsed_ns: np.ndarray
    upper_elapsed_ns: np.ndarray
    binary_identified: np.ndarray
    binary_event: np.ndarray
    flow_unit: np.ndarray
    projection_rows: list[dict[str, Any]]
    censoring_rows: list[dict[str, Any]]
    exclusion_rows: list[dict[str, Any]]


def parse_positive_quote(bid_text: str, ask_text: str) -> tuple[float, float, bool]:
    try:
        bid = float(bid_text)
        ask = float(ask_text)
    except ValueError:
        return math.nan, math.nan, False
    valid = (
        math.isfinite(bid)
        and math.isfinite(ask)
        and bid > 0.0
        and ask > 0.0
        and bid <= ask
    )
    return bid, ask, valid


def load_quote_events(
    path: Path,
    *,
    header: Sequence[str],
    event_type: str,
    segment_id: str,
) -> QuoteEvents:
    indexes = {name: header.index(name) for name in (
        "segment_id",
        "event_seq",
        "local_ts_ns",
        "event_type",
        "bid_px",
        "ask_px",
    )}
    timestamps: list[int] = []
    sequences: list[int] = []
    bids: list[float] = []
    asks: list[float] = []
    valid_rows: list[bool] = []
    previous_sequence = -1
    previous_timestamp = -1
    for row in read_exact_csv(path, header, compressed=True):
        contracts.require(
            row[indexes["segment_id"]] == segment_id,
            "H0B_SOURCE_ORDERING_MISMATCH",
            str(path),
            f"segment={row[indexes['segment_id']]} expected={segment_id}",
        )
        sequence = int(row[indexes["event_seq"]])
        timestamp = int(row[indexes["local_ts_ns"]])
        contracts.require(
            sequence > previous_sequence and timestamp >= previous_timestamp,
            "H0B_SOURCE_ORDERING_MISMATCH",
            str(path),
            f"event_seq={sequence} local_ts_ns={timestamp}",
        )
        previous_sequence = sequence
        previous_timestamp = timestamp
        if row[indexes["event_type"]] != event_type:
            continue
        bid, ask, valid = parse_positive_quote(
            row[indexes["bid_px"]],
            row[indexes["ask_px"]],
        )
        timestamps.append(timestamp)
        sequences.append(sequence)
        bids.append(bid)
        asks.append(ask)
        valid_rows.append(valid)
    contracts.require(
        bool(timestamps),
        "H0B_SOURCE_SCHEMA_MISMATCH",
        str(path),
        f"no {event_type} rows",
    )
    return QuoteEvents(
        local_ts_ns=np.asarray(timestamps, dtype=np.int64),
        event_seq=np.asarray(sequences, dtype=np.int64),
        bid_px=np.asarray(bids, dtype=np.float64),
        ask_px=np.asarray(asks, dtype=np.float64),
        valid=np.asarray(valid_rows, dtype=bool),
    )


def stage2_components() -> dict[tuple[str, str, str], tuple[FlowComponent, ...]]:
    path = STAGE2_ROOT / "candidate_episode_membership.csv.gz"
    validate_source_access(
        path.relative_to(REPO_ROOT).as_posix(),
        source_role="accepted_stage2_primary",
        phase="feature_read",
    )
    expected = (
        "session_id",
        "candidate_id",
        "primary_episode",
        "rejection_reason",
        "aggressor_side",
        "direction_sign",
        "shock_ts_ns",
        "decision_ts_ns",
        "impact_ratio",
        "pre_state_ts_ns",
        "pre_state_age_ms",
        "segment_id",
        "connection_epoch_id",
        "segment_start_ts_ns",
        "segment_end_ts_ns",
        "cluster_id",
        "continuous_flow_episode_id",
        "overlap_block_id",
        "window_end_ts_ns",
    )
    index = {name: expected.index(name) for name in expected}
    grouped: dict[tuple[str, str, str, str], list[tuple[int, int]]] = defaultdict(list)
    for row in read_exact_csv(path, expected, compressed=True):
        if row[index["primary_episode"]] != "true":
            continue
        overlap = row[index["overlap_block_id"]]
        if not overlap:
            continue
        key = (
            row[index["session_id"]],
            row[index["segment_id"]],
            row[index["connection_epoch_id"]],
            overlap,
        )
        grouped[key].append(
            (
                int(row[index["shock_ts_ns"]]),
                int(row[index["window_end_ts_ns"]]),
            )
        )
    result: dict[tuple[str, str, str], list[FlowComponent]] = defaultdict(list)
    for (session, segment, epoch, overlap), intervals in grouped.items():
        result[(session, segment, epoch)].append(
            FlowComponent(
                component_id=f"component:{overlap}",
                start_ns=min(value[0] for value in intervals),
                end_ns=max(value[1] for value in intervals),
            )
        )
    frozen_counts = {"jul30": 9, "aug04": 6}
    final: dict[tuple[str, str, str], tuple[FlowComponent, ...]] = {}
    observed_counts = Counter()
    for key, components in result.items():
        ordered = sorted(components, key=lambda value: (
            value.start_ns,
            value.end_ns,
            value.component_id,
        ))
        for left, right in zip(ordered, ordered[1:]):
            contracts.require(
                left.end_ns < right.start_ns,
                "H0B_FLOW_COMPONENT_MISMATCH",
                f"{key}",
                f"overlap {left} {right}",
            )
        final[key] = tuple(ordered)
        observed_counts[key[0]] += len(ordered)
    for session, expected_count in frozen_counts.items():
        contracts.require(
            observed_counts[session] == expected_count,
            "H0B_FLOW_COMPONENT_MISMATCH",
            f"$.components.{session}",
            f"expected={expected_count} observed={observed_counts[session]}",
        )
    return final


def stage3_dose_events() -> dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]]:
    path = STAGE3_ROOT / "candidate_audit_projection.csv.gz"
    validate_source_access(
        path.relative_to(REPO_ROOT).as_posix(),
        source_role="accepted_stage3_primary",
        phase="feature_read",
    )
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_header = (
            "session_id,campaign_id,segment_id,profile_id,candidate_seq,"
            "aggressor_side,direction_sign,burst_start_ts_ns,burst_end_ts_ns,"
            "burst_duration_ms,burst_trade_count,burst_trade_qty,touch_trade_qty,"
            "touch_trade_qty_at_shock,touch_trade_qty_through_decision,"
            "post_decision_burst_trade_count,pre_state_ts_ns,pre_best_px,"
            "pre_best_qty,shock_ts_ns,impact_ratio,shock_impact_ratio,"
            "decision_ts_ns,confirmation_lag_ms,confirmed_best_px,"
            "confirmed_best_qty,price_level_depleted,queue_drop_ratio,"
            "confirmed_removed_qty,trade_explained_ratio,attribution,"
            "pre_hl_bbo_ts_ns,pre_hl_bbo_age_ms,pre_hl_fast_source_ts_ns,"
            "pre_hl_fast_age_ms,primary_episode,rejection_reason"
        ).split(",")
        contracts.require(
            reader.fieldnames == expected_header,
            "H0B_SOURCE_SCHEMA_MISMATCH",
            str(path),
            "Stage 3 projection header mismatch",
        )
        grouped: dict[tuple[str, str, int], list[tuple[int, float]]] = defaultdict(list)
        for row in reader:
            if row["primary_episode"] != "true":
                continue
            direction = int(row["direction_sign"])
            dose = float(row["queue_drop_ratio"])
            contracts.require(
                direction in {-1, 1}
                and math.isfinite(dose)
                and 0.0 <= dose <= 1.0,
                "H0B_DOSE_RECONSTRUCTION_MISMATCH",
                str(path),
                f"direction={direction} dose={dose}",
            )
            grouped[
                (row["session_id"], row["segment_id"], direction)
            ].append((int(row["decision_ts_ns"]), dose))
    result = {}
    for key, values in grouped.items():
        values.sort()
        result[key] = (
            np.asarray([value[0] for value in values], dtype=np.int64),
            np.asarray([value[1] for value in values], dtype=np.float64),
        )
    return result


def trailing_dose(
    grid: np.ndarray,
    events: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    if events is None or events[0].size == 0:
        return np.zeros(grid.size, dtype=np.float64)
    timestamps, values = events
    cumulative = np.concatenate(
        (np.asarray([0.0]), np.cumsum(values, dtype=np.float64))
    )
    right = np.searchsorted(timestamps, grid, side="right")
    left = np.searchsorted(timestamps, grid - 500_000_000, side="right")
    return cumulative[right] - cumulative[left]


def flow_units_for_grid(
    *,
    session: str,
    segment_id: str,
    epoch: str,
    grid: np.ndarray,
    components: Mapping[
        tuple[str, str, str],
        tuple[FlowComponent, ...],
    ],
) -> np.ndarray:
    result = np.asarray(
        [
            f"background:{session}:{segment_id}:{epoch}:{int(ts // 2_000_000_000)}"
            for ts in grid
        ],
        dtype=object,
    )
    for component in components.get((session, segment_id, epoch), ()):
        mask = np.logical_and(
            grid >= component.start_ns,
            grid <= component.end_ns,
        )
        contracts.require(
            not np.any(
                np.char.startswith(
                    result[mask].astype(str),
                    "component:",
                )
            ),
            "H0B_FLOW_COMPONENT_MISMATCH",
            f"{session}/{segment_id}/{component.component_id}",
            "row assigned to two components",
        )
        result[mask] = (
            f"component:{session}:{segment_id}:{epoch}:"
            f"{component.component_id}"
        )
    return result


def event_geometry(
    *,
    grid: np.ndarray,
    reference_indexes: np.ndarray,
    events: QuoteEvents,
    side: str,
    identification_class: str = "binary_identification_supported",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    branch = np.full(grid.size, 2, dtype=np.int8)
    lower = np.zeros(grid.size, dtype=np.int64)
    upper = np.full(grid.size, contracts.HORIZON_NS, dtype=np.int64)
    binary_event = np.zeros(grid.size, dtype=bool)
    for row_index, (timestamp, reference_index) in enumerate(
        zip(grid.tolist(), reference_indexes.tolist())
    ):
        vulnerable = (
            events.ask_px[reference_index]
            if side == "maker_ask_risk"
            else events.bid_px[reference_index]
        )
        endpoint = timestamp + contracts.HORIZON_NS
        start = int(
            np.searchsorted(events.local_ts_ns, timestamp, side="right")
        )
        scan_end = int(
            np.searchsorted(events.local_ts_ns, endpoint, side="right")
        )
        if identification_class == "interval_likelihood_only_supported":
            after = int(
                np.searchsorted(events.local_ts_ns, endpoint, side="right")
            )
            while after < events.local_ts_ns.size and not events.valid[after]:
                after += 1
            scan_end = min(events.local_ts_ns.size, after + 1)
        last_non_adverse_ts = timestamp
        cursor = start
        while cursor < scan_end:
            group_ts = int(events.local_ts_ns[cursor])
            group_end = cursor + 1
            while (
                group_end < scan_end
                and events.local_ts_ns[group_end] == group_ts
            ):
                group_end += 1
            adverse_found = False
            qualifying_non_adverse = False
            for event_index in range(cursor, group_end):
                if not events.valid[event_index]:
                    continue
                adverse = (
                    events.bid_px[event_index] >= vulnerable
                    if side == "maker_ask_risk"
                    else events.ask_px[event_index] <= vulnerable
                )
                if adverse:
                    adverse_found = True
                    branch[row_index] = 1
                    lower[row_index] = last_non_adverse_ts - timestamp
                    upper[row_index] = group_ts - timestamp
                    binary_event[row_index] = group_ts <= endpoint
                    break
                qualifying_non_adverse = True
            if adverse_found:
                break
            if qualifying_non_adverse and group_ts < endpoint:
                last_non_adverse_ts = group_ts
            cursor = group_end
        if (
            branch[row_index] == 1
            and upper[row_index] > contracts.HORIZON_NS
        ):
            contracts.require(
                identification_class
                == "interval_likelihood_only_supported"
                and lower[row_index] < contracts.HORIZON_NS,
                "H0B_HORIZON_STRADDLE_MISMATCH",
                f"$.geometry[{row_index}]",
                f"L={lower[row_index]} U={upper[row_index]}",
            )
            branch[row_index] = 3
            binary_event[row_index] = False
        if branch[row_index] == 1:
            contracts.require(
                0 <= lower[row_index] < upper[row_index] <= contracts.HORIZON_NS,
                "H0B_OBSERVATION_BOUND_MISMATCH",
                f"$.geometry[{row_index}]",
                f"L={lower[row_index]} U={upper[row_index]}",
            )
    return branch, lower, upper, binary_event


def cadence_features(
    *,
    grid: np.ndarray,
    segment_start_ns: int,
    valid_receive_ts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    count = np.full(grid.size, np.nan, dtype=np.float64)
    no_information = np.full(grid.size, np.nan, dtype=np.float64)
    full = grid - 1_000_000_000 >= segment_start_ns
    if not full.any():
        return count, no_information
    selected = grid[full]
    right = np.searchsorted(valid_receive_ts, selected, side="right")
    left = np.searchsorted(
        valid_receive_ts,
        selected - 1_000_000_000,
        side="right",
    )
    count[full] = right - left
    receive_bin_end = (
        (valid_receive_ts + contracts.GRID_NS - 1)
        // contracts.GRID_NS
        * contracts.GRID_NS
    )
    unique_bin_end = np.unique(receive_bin_end)
    occupied_right = np.searchsorted(unique_bin_end, selected, side="right")
    occupied_left = np.searchsorted(
        unique_bin_end,
        selected - 1_000_000_000,
        side="right",
    )
    occupied = occupied_right - occupied_left
    no_information[full] = (100.0 - occupied) / 100.0
    return count, no_information


def quote_state(
    events: QuoteEvents,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indexes = np.searchsorted(events.local_ts_ns, grid, side="right") - 1
    available = indexes >= 0
    safe = np.maximum(indexes, 0)
    valid = np.logical_and(available, events.valid[safe])
    bid = np.where(valid, events.bid_px[safe], np.nan)
    ask = np.where(valid, events.ask_px[safe], np.nan)
    timestamp = np.where(valid, events.local_ts_ns[safe], -1)
    return indexes, bid, ask, timestamp


def basis_residual(
    basis_level: np.ndarray,
) -> np.ndarray:
    result = np.full(basis_level.size, np.nan, dtype=np.float64)
    alpha = 1.0 - math.exp(
        -math.log(2.0) * contracts.GRID_NS / 60_000_000_000
    )
    ewma: float | None = None
    for index, value in enumerate(basis_level):
        if not math.isfinite(float(value)):
            continue
        if ewma is not None:
            result[index] = float(value) - ewma
            ewma += alpha * (float(value) - ewma)
        else:
            ewma = float(value)
    return result


def risk_gap_features(
    *,
    grid: np.ndarray,
    hyperliquid: QuoteEvents,
    binance: QuoteEvents,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, hl_bid, hl_ask, hl_timestamp = quote_state(hyperliquid, grid)
    _, bn_bid, bn_ask, bn_timestamp = quote_state(binance, grid)
    bn_mid = 0.5 * (bn_bid + bn_ask)
    hl_mid = 0.5 * (hl_bid + hl_ask)
    reference_mid = 0.5 * (bn_mid + hl_mid)
    valid = np.logical_and.reduce(
        (
            np.isfinite(reference_mid),
            reference_mid > 0.0,
            np.isfinite(bn_bid),
            np.isfinite(bn_ask),
            np.isfinite(hl_bid),
            np.isfinite(hl_ask),
        )
    )
    ask_gap = np.where(
        valid,
        10_000.0 * (bn_bid - hl_ask) / reference_mid,
        np.nan,
    )
    bid_gap = np.where(
        valid,
        10_000.0 * (hl_bid - bn_ask) / reference_mid,
        np.nan,
    )
    past = grid - contracts.HORIZON_NS
    _, past_hl_bid, past_hl_ask, _ = quote_state(hyperliquid, past)
    _, past_bn_bid, past_bn_ask, _ = quote_state(binance, past)
    past_bn_mid = 0.5 * (past_bn_bid + past_bn_ask)
    past_hl_mid = 0.5 * (past_hl_bid + past_hl_ask)
    past_reference = 0.5 * (past_bn_mid + past_hl_mid)
    past_valid = np.logical_and.reduce(
        (
            np.isfinite(past_reference),
            past_reference > 0.0,
            np.isfinite(past_bn_bid),
            np.isfinite(past_bn_ask),
            np.isfinite(past_hl_bid),
            np.isfinite(past_hl_ask),
        )
    )
    past_ask_gap = np.where(
        past_valid,
        10_000.0 * (past_bn_bid - past_hl_ask) / past_reference,
        np.nan,
    )
    past_bid_gap = np.where(
        past_valid,
        10_000.0 * (past_hl_bid - past_bn_ask) / past_reference,
        np.nan,
    )
    ask_change = ask_gap - past_ask_gap
    bid_change = bid_gap - past_bid_gap
    binance_age_ms = np.where(
        bn_timestamp >= 0,
        (grid - bn_timestamp) / 1_000_000.0,
        np.nan,
    )
    hyperliquid_age_ms = np.where(
        hl_timestamp >= 0,
        (grid - hl_timestamp) / 1_000_000.0,
        np.nan,
    )
    basis_level = np.where(
        valid,
        10_000.0 * (bn_mid - hl_mid) / reference_mid,
        np.nan,
    )
    residual = basis_residual(basis_level)
    risk_gap = np.column_stack((ask_gap, bid_gap))
    risk_gap_change = np.column_stack((ask_change, bid_change))
    return (
        risk_gap,
        risk_gap_change,
        binance_age_ms,
        hyperliquid_age_ms,
        residual,
    )


def projection_identity_row(
    *,
    session: str,
    segment_id: str,
    epoch: str,
    grid_ts_ns: int,
    side: str,
    block_start_ns: int,
    h0_values: np.ndarray,
    h1_values: np.ndarray,
    dose: float,
    branch: int,
    lower_ns: int,
    upper_ns: int,
    binary_identified: bool,
    binary_event: bool,
    flow_unit: str,
) -> dict[str, Any]:
    def nullable(value: float) -> float | None:
        observed = float(value)
        return observed if math.isfinite(observed) else None

    return {
        "schema_version": "skhynix_stage_h0b_projection_tuple_v1",
        "session": session,
        "segment_id": segment_id,
        "connection_epoch_id": epoch,
        "grid_ts_ns": int(grid_ts_ns),
        "side": side,
        "absolute_block_id": (
            str(int(block_start_ns)) if block_start_ns >= 0 else ""
        ),
        "h0_raw": [nullable(value) for value in h0_values],
        "h1_added_raw": [nullable(value) for value in h1_values],
        "trailing_queue_shock_dose_500ms": nullable(dose),
        "likelihood_branch": int(branch),
        "lower_elapsed_ns": int(lower_ns),
        "upper_elapsed_ns": int(upper_ns),
        "binary_identified": bool(binary_identified),
        "binary_event": bool(binary_event),
        "flow_unit": flow_unit,
    }


def build_session_dataset(
    *,
    spec: Any,
    h0a_segments: Sequence[Any],
    dose_events: Mapping[
        tuple[str, str, int],
        tuple[np.ndarray, np.ndarray],
    ],
    components: Mapping[
        tuple[str, str, str],
        tuple[FlowComponent, ...],
    ],
) -> SessionDataset:
    session_start = min(segment.start_ns for segment in h0a_segments)
    session_end = max(segment.end_ns for segment in h0a_segments)
    all_grid: list[np.ndarray] = []
    all_segment_codes: list[np.ndarray] = []
    all_blocks: list[np.ndarray] = []
    all_h0: list[np.ndarray] = []
    all_h1_added: list[np.ndarray] = []
    all_dose: list[np.ndarray] = []
    all_branch: list[np.ndarray] = []
    all_lower: list[np.ndarray] = []
    all_upper: list[np.ndarray] = []
    all_binary_identified: list[np.ndarray] = []
    all_binary_event: list[np.ndarray] = []
    all_flow: list[np.ndarray] = []
    projection_rows: list[dict[str, Any]] = []
    censoring_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    segment_ids = tuple(segment.segment_id for segment in h0a_segments)

    for segment_code, segment in enumerate(h0a_segments):
        r0_segment = (
            CANONICAL_SOURCE_ROOT
            / spec.r0_relative_path
            / "segments"
            / segment.segment_id
        )
        hyperliquid_path = r0_segment / "hyperliquid_hot_events.csv.gz"
        binance_path = r0_segment / "binance_hot_events.csv.gz"
        validate_source_access(
            hyperliquid_path.relative_to(CANONICAL_SOURCE_ROOT).as_posix(),
            source_role="r0_hyperliquid_bbo",
            phase="feature_read",
        )
        validate_source_access(
            binance_path.relative_to(CANONICAL_SOURCE_ROOT).as_posix(),
            source_role="r0_binance_bookticker",
            phase="feature_read",
        )
        hyperliquid = load_quote_events(
            hyperliquid_path,
            header=HYPERLIQUID_HEADER,
            event_type="bbo",
            segment_id=segment.segment_id,
        )
        binance = load_quote_events(
            binance_path,
            header=BINANCE_HEADER,
            event_type="bookTicker",
            segment_id=segment.segment_id,
        )
        first, last, nominal_count = h0a_support.segment_grid_bounds(
            segment.start_ns,
            segment.end_ns,
        )
        nominal_grid = np.arange(
            first,
            last + contracts.GRID_NS,
            contracts.GRID_NS,
            dtype=np.int64,
        )
        reference_indexes = (
            np.searchsorted(
                hyperliquid.local_ts_ns,
                nominal_grid,
                side="right",
            )
            - 1
        )
        available = reference_indexes >= 0
        safe_reference = np.maximum(reference_indexes, 0)
        reference_valid = np.logical_and(
            available,
            hyperliquid.valid[safe_reference],
        )
        endpoint_inside = (
            nominal_grid + contracts.HORIZON_NS < segment.end_ns
        )
        eligible = np.logical_and(reference_valid, endpoint_inside)
        support_class = np.full(
            nominal_grid.size,
            "binary_identification_supported",
            dtype=object,
        )
        support_class[~available] = "reference_quote_unavailable"
        support_class[np.logical_and(available, ~reference_valid)] = (
            "invalid_quote_state"
        )
        support_class[
            np.logical_and(reference_valid, ~endpoint_inside)
        ] = "right_censored_segment"
        class_counts = Counter(support_class.tolist())
        for side in contracts.SIDES:
            for identification_class in contracts.IDENTIFICATION_CLASSES:
                count = int(class_counts[identification_class])
                censoring_rows.append(
                    {
                        "session": spec.session_id,
                        "segment_id": segment.segment_id,
                        "side": side,
                        "identification_class": identification_class,
                        "disposition": contracts.SUPPORT_DISPOSITIONS[
                            identification_class
                        ],
                        "row_count": count,
                        "reason_code": (
                            ""
                            if identification_class
                            in {
                                "binary_identification_supported",
                                "interval_likelihood_only_supported",
                            }
                            else identification_class
                        ),
                    }
                )
                if count and identification_class not in {
                    "binary_identification_supported",
                    "interval_likelihood_only_supported",
                }:
                    exclusion_rows.append(
                        {
                            "session": spec.session_id,
                            "segment_id": segment.segment_id,
                            "side": side,
                            "stage": "support",
                            "reason_code": identification_class,
                            "row_count": count,
                        }
                    )
        grid = nominal_grid[eligible]
        reference = reference_indexes[eligible]
        block_start = (grid // contracts.BLOCK_NS) * contracts.BLOCK_NS
        complete = np.logical_and.reduce(
            (
                block_start >= segment.start_ns,
                block_start + contracts.BLOCK_NS <= segment.end_ns,
                block_start
                + contracts.BLOCK_NS
                - contracts.GRID_NS
                + contracts.HORIZON_NS
                < segment.end_ns,
            )
        )
        block_start = np.where(complete, block_start, -1)
        elapsed_session = (
            (grid - session_start) / (session_end - session_start)
        )
        elapsed_segment = (
            (grid - segment.start_ns) / (segment.end_ns - segment.start_ns)
        )
        valid_hl_receive = hyperliquid.local_ts_ns[hyperliquid.valid]
        cadence_count, no_information = cadence_features(
            grid=grid,
            segment_start_ns=segment.start_ns,
            valid_receive_ts=valid_hl_receive,
        )
        h0_raw = np.column_stack(
            (
                elapsed_session,
                elapsed_session**2,
                elapsed_segment,
                cadence_count,
                no_information,
            )
        )
        (
            risk_gap,
            risk_gap_change,
            binance_age,
            hyperliquid_age,
            residual,
        ) = risk_gap_features(
            grid=grid,
            hyperliquid=hyperliquid,
            binance=binance,
        )
        h1_added = np.empty((grid.size, 2, 5), dtype=np.float64)
        h1_added[:, :, 0] = risk_gap
        h1_added[:, :, 1] = risk_gap_change
        h1_added[:, :, 2] = binance_age[:, None]
        h1_added[:, :, 3] = hyperliquid_age[:, None]
        h1_added[:, :, 4] = residual[:, None]
        dose = np.column_stack(
            (
                trailing_dose(
                    grid,
                    dose_events.get(
                        (spec.session_id, segment.segment_id, 1)
                    ),
                ),
                trailing_dose(
                    grid,
                    dose_events.get(
                        (spec.session_id, segment.segment_id, -1)
                    ),
                ),
            )
        )
        epoch = f"{segment.segment_id}:epoch_0"
        flow = flow_units_for_grid(
            session=spec.session_id,
            segment_id=segment.segment_id,
            epoch=epoch,
            grid=grid,
            components=components,
        )
        branch = np.empty((grid.size, 2), dtype=np.int8)
        lower = np.empty((grid.size, 2), dtype=np.int64)
        upper = np.empty((grid.size, 2), dtype=np.int64)
        binary_event = np.empty((grid.size, 2), dtype=bool)
        for side_index, side in enumerate(contracts.SIDES):
            (
                branch[:, side_index],
                lower[:, side_index],
                upper[:, side_index],
                binary_event[:, side_index],
            ) = event_geometry(
                grid=grid,
                reference_indexes=reference,
                events=hyperliquid,
                side=side,
            )
        binary_identified = np.ones((grid.size, 2), dtype=bool)

        for side_index, side in enumerate(contracts.SIDES):
            hasher = hashlib.sha256()
            for row_index in range(grid.size):
                hasher.update(
                    contracts.canonical_json_bytes(
                        projection_identity_row(
                            session=spec.session_id,
                            segment_id=segment.segment_id,
                            epoch=epoch,
                            grid_ts_ns=int(grid[row_index]),
                            side=side,
                            block_start_ns=int(block_start[row_index]),
                            h0_values=h0_raw[row_index],
                            h1_values=h1_added[row_index, side_index],
                            dose=float(dose[row_index, side_index]),
                            branch=int(branch[row_index, side_index]),
                            lower_ns=int(lower[row_index, side_index]),
                            upper_ns=int(upper[row_index, side_index]),
                            binary_identified=True,
                            binary_event=bool(
                                binary_event[row_index, side_index]
                            ),
                            flow_unit=str(flow[row_index]),
                        )
                    )
                )
            event_count = int(binary_event[:, side_index].sum())
            censor_count = int(
                (branch[:, side_index] == 2).sum()
            )
            straddle_count = int(
                (branch[:, side_index] == 3).sum()
            )
            projection_rows.append(
                {
                    "session": spec.session_id,
                    "segment_id": segment.segment_id,
                    "side": side,
                    "support_row_count": nominal_count,
                    "interval_likelihood_row_count": int(grid.size),
                    "binary_row_count": int(grid.size),
                    "event_observed_count": event_count,
                    "full_horizon_right_censor_count": censor_count,
                    "horizon_straddle_count": straddle_count,
                    "geometric_exclusion_count": int(
                        nominal_count - grid.size
                    ),
                    "canonical_projection_sha256": hasher.hexdigest(),
                    "first_grid_ts_ns": int(first),
                    "last_grid_ts_ns": int(last),
                }
            )

        all_grid.append(grid)
        all_segment_codes.append(
            np.full(grid.size, segment_code, dtype=np.int16)
        )
        all_blocks.append(block_start.astype(np.int64))
        all_h0.append(h0_raw)
        all_h1_added.append(h1_added)
        all_dose.append(dose)
        all_branch.append(branch)
        all_lower.append(lower)
        all_upper.append(upper)
        all_binary_identified.append(binary_identified)
        all_binary_event.append(binary_event)
        all_flow.append(flow)

    return SessionDataset(
        session=spec.session_id,
        segment_ids=segment_ids,
        grid_ts_ns=np.concatenate(all_grid),
        segment_code=np.concatenate(all_segment_codes),
        block_start_ns=np.concatenate(all_blocks),
        h0_raw=np.concatenate(all_h0),
        h1_added_raw=np.concatenate(all_h1_added),
        dose=np.concatenate(all_dose),
        branch=np.concatenate(all_branch),
        lower_elapsed_ns=np.concatenate(all_lower),
        upper_elapsed_ns=np.concatenate(all_upper),
        binary_identified=np.concatenate(all_binary_identified),
        binary_event=np.concatenate(all_binary_event),
        flow_unit=np.concatenate(all_flow),
        projection_rows=projection_rows,
        censoring_rows=censoring_rows,
        exclusion_rows=exclusion_rows,
    )


def paired_raw_features(
    dataset: SessionDataset,
    anchor_indexes: np.ndarray,
    model: str,
) -> tuple[np.ndarray, np.ndarray]:
    indexes = np.asarray(anchor_indexes, dtype=np.int64)
    side = np.tile(np.asarray([1.0, 0.0]), indexes.size)
    h0 = np.repeat(dataset.h0_raw[indexes], 2, axis=0)
    if model == "H0":
        return h0, side
    added = dataset.h1_added_raw[indexes].reshape(-1, 5)
    return np.column_stack((h0, added)), side


def paired_outcomes(
    dataset: SessionDataset,
    anchor_indexes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indexes = np.asarray(anchor_indexes, dtype=np.int64)
    return (
        dataset.branch[indexes].reshape(-1),
        dataset.lower_elapsed_ns[indexes].reshape(-1),
        dataset.upper_elapsed_ns[indexes].reshape(-1),
        dataset.binary_identified[indexes].reshape(-1),
        dataset.binary_event[indexes].reshape(-1),
    )


@dataclass
class FittedHazardModel:
    model: str
    scales: tuple[contracts.FeatureScale, ...]
    alpha: np.ndarray
    beta: np.ndarray
    converged: bool
    iterations: int
    objective: float
    gradient_max_abs: float

    def probabilities(self, raw: np.ndarray, side: np.ndarray) -> np.ndarray:
        design = contracts.transform_design(
            raw,
            side,
            self.scales,
            model=self.model,
        )
        eta = self.alpha[None, :] + design @ self.beta[:, None]
        return contracts.sigmoid(eta)


def fit_hazard_model(
    *,
    model: str,
    raw: np.ndarray,
    side: np.ndarray,
    branch: np.ndarray,
    lower_ns: np.ndarray,
    upper_ns: np.ndarray,
) -> FittedHazardModel:
    features = (
        contracts.H0_RAW_FEATURES
        if model == "H0"
        else contracts.H1_RAW_FEATURES
    )
    scales = contracts.fit_feature_scales(raw, features)
    design = contracts.transform_design(
        raw,
        side,
        scales,
        model=model,
    )
    parameter_count = 5 + design.shape[1]
    initial = np.concatenate(
        (
            np.full(5, -4.0, dtype=np.float64),
            np.zeros(design.shape[1], dtype=np.float64),
        )
    )

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        alpha = theta[:5]
        beta = theta[5:]
        total_loss = 0.0
        alpha_gradient = np.zeros(5, dtype=np.float64)
        beta_gradient = np.zeros(beta.size, dtype=np.float64)
        for start in range(0, design.shape[0], MODEL_CHUNK_ROWS):
            stop = min(design.shape[0], start + MODEL_CHUNK_ROWS)
            selected = design[start:stop]
            eta = alpha[None, :] + selected @ beta[:, None]
            q = contracts.sigmoid(eta)
            _, loss, eta_gradient = (
                contracts.interval_objective_eta_gradient(
                    q,
                    branch[start:stop],
                    lower_ns[start:stop],
                    upper_ns[start:stop],
                )
            )
            total_loss += float(loss.sum())
            alpha_gradient += eta_gradient.sum(axis=0)
            beta_gradient += selected.T @ eta_gradient.sum(axis=1)
        total_loss += 0.5 * RIDGE_LAMBDA * float(beta @ beta)
        beta_gradient += RIDGE_LAMBDA * beta
        gradient = np.concatenate((alpha_gradient, beta_gradient))
        contracts.require(
            math.isfinite(total_loss) and np.isfinite(gradient).all(),
            "H0B_ESTIMATOR_CONTRACT_MISMATCH",
            "$.optimizer",
            "non-finite objective or gradient",
        )
        return total_loss, gradient

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": OPTIMIZER_MAX_ITERATIONS,
            "gtol": OPTIMIZER_GRADIENT_TOLERANCE,
            "ftol": OPTIMIZER_PARAMETER_TOLERANCE,
            "maxls": 50,
        },
    )
    contracts.require(
        result.x.shape == (parameter_count,)
        and np.isfinite(result.x).all()
        and math.isfinite(float(result.fun)),
        "H0B_ESTIMATOR_CONTRACT_MISMATCH",
        "$.optimizer.result",
        repr(result),
    )
    gradient_max = float(np.max(np.abs(result.jac)))
    converged = bool(result.success)
    return FittedHazardModel(
        model=model,
        scales=scales,
        alpha=np.asarray(result.x[:5], dtype=np.float64),
        beta=np.asarray(result.x[5:], dtype=np.float64),
        converged=converged,
        iterations=int(result.nit),
        objective=float(result.fun),
        gradient_max_abs=gradient_max,
    )


def predict_model(
    *,
    fitted: FittedHazardModel,
    raw: np.ndarray,
    side: np.ndarray,
    branch: np.ndarray,
    lower_ns: np.ndarray,
    upper_ns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q = fitted.probabilities(raw, side)
    _, loss = contracts.likelihood_and_loss(
        q,
        branch,
        lower_ns,
        upper_ns,
    )
    risk = 1.0 - np.prod(1.0 - q, axis=1)
    return loss, risk


def binary_metrics(
    risk: np.ndarray,
    event: np.ndarray,
) -> tuple[float, float]:
    probability = np.clip(
        np.asarray(risk, dtype=np.float64),
        contracts.PROBABILITY_MIN,
        contracts.PROBABILITY_MAX,
    )
    target = np.asarray(event, dtype=np.float64)
    brier = float(np.mean((probability - target) ** 2))
    log_loss = float(
        np.mean(
            -target * np.log(probability)
            - (1.0 - target) * np.log1p(-probability)
        )
    )
    return brier, log_loss


def bin_diagnostic_rows(
    *,
    session: str,
    fold_id: int,
    model: str,
    side: str,
    training_risk: np.ndarray,
    test_risk: np.ndarray,
    test_event: np.ndarray,
    path: str,
) -> list[dict[str, Any]]:
    edges = contracts.nearest_rank_edges(
        training_risk,
        [index / 10.0 for index in range(1, 10)],
    )
    assigned = contracts.assign_right_closed_bins(test_risk, edges)
    rows = []
    bin_field = "reliability_bin" if path == "reliability" else "decile"
    for bin_index in range(10):
        mask = assigned == bin_index
        count = int(mask.sum())
        rows.append(
            {
                "session": session,
                "fold_id": fold_id,
                "model": model,
                "side": side,
                bin_field: bin_index + 1,
                "training_lower_edge": (
                    None if bin_index == 0 else edges[bin_index - 1]
                ),
                "training_upper_edge": (
                    None if bin_index == 9 else edges[bin_index]
                ),
                "test_count": count,
                "event_count": int(test_event[mask].sum()) if count else 0,
                "mean_predicted_risk": (
                    float(test_risk[mask].mean()) if count else None
                ),
                "realized_rate": (
                    float(test_event[mask].mean()) if count else None
                ),
            }
        )
    return rows


def detect_regimes(
    *,
    dataset: SessionDataset,
    fold_id: int,
    anchor_indexes: np.ndarray,
    side_index: int,
    risk: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
) -> list[dict[str, Any]]:
    indexes = np.asarray(anchor_indexes, dtype=np.int64)
    timestamps = dataset.grid_ts_ns[indexes]
    segments = dataset.segment_code[indexes]
    result: list[dict[str, Any]] = []
    regime_id = 1
    run_start = 0
    while run_start < indexes.size:
        run_stop = run_start + 1
        while (
            run_stop < indexes.size
            and segments[run_stop] == segments[run_stop - 1]
            and timestamps[run_stop]
            == timestamps[run_stop - 1] + contracts.GRID_NS
        ):
            run_stop += 1
        cursor = run_start
        entry_count = 0
        in_regime = False
        detect_index = -1
        exit_count = 0
        while cursor < run_stop:
            value = float(risk[cursor])
            if not in_regime:
                entry_count = entry_count + 1 if value >= entry_threshold else 0
                if entry_count == 3:
                    in_regime = True
                    detect_index = cursor
                    exit_count = 0
            else:
                exit_count = exit_count + 1 if value <= exit_threshold else 0
                if exit_count == 5:
                    detect_time = int(timestamps[detect_index])
                    exit_time = int(timestamps[cursor])
                    result.append(
                        {
                            "session": dataset.session,
                            "fold_id": fold_id,
                            "side": contracts.SIDES[side_index],
                            "regime_id": regime_id,
                            "t_detect_ns": detect_time,
                            "t_exit_ns": exit_time,
                            "censor_time_ns": None,
                            "censored": False,
                            "total_dwell_ns": exit_time - detect_time,
                            "detection_block_id": str(
                                (detect_time // contracts.BLOCK_NS)
                                * contracts.BLOCK_NS
                            ),
                            "entry_threshold": entry_threshold,
                            "exit_threshold": exit_threshold,
                        }
                    )
                    regime_id += 1
                    in_regime = False
                    entry_count = 0
                    exit_count = 0
                    detect_index = -1
            cursor += 1
        if in_regime:
            detect_time = int(timestamps[detect_index])
            censor_time = int(timestamps[run_stop - 1] + contracts.GRID_NS)
            result.append(
                {
                    "session": dataset.session,
                    "fold_id": fold_id,
                    "side": contracts.SIDES[side_index],
                    "regime_id": regime_id,
                    "t_detect_ns": detect_time,
                    "t_exit_ns": None,
                    "censor_time_ns": censor_time,
                    "censored": True,
                    "total_dwell_ns": censor_time - detect_time,
                    "detection_block_id": str(
                        (detect_time // contracts.BLOCK_NS)
                        * contracts.BLOCK_NS
                    ),
                    "entry_threshold": entry_threshold,
                    "exit_threshold": exit_threshold,
                }
            )
            regime_id += 1
        run_start = run_stop
    return result


@dataclass
class SessionEvaluation:
    fold_score_rows: list[dict[str, Any]]
    feature_rows: list[dict[str, Any]]
    reliability_rows: list[dict[str, Any]]
    decile_rows: list[dict[str, Any]]
    coarse_rows: list[dict[str, Any]]
    regime_rows: list[dict[str, Any]]
    oof_fold: np.ndarray
    oof_loss_h0: np.ndarray
    oof_loss_h1: np.ndarray
    oof_risk_h0: np.ndarray
    oof_risk_h1: np.ndarray
    oof_h1_decile: np.ndarray
    data_quality: bool
    gate_reasons: list[str]


def feature_availability_rows(
    *,
    session: str,
    fold_id: int,
    model: str,
    scales: Sequence[contracts.FeatureScale],
) -> list[dict[str, Any]]:
    rows = []
    for scale in scales:
        missing_fraction = scale.missing_count / scale.row_count
        rows.append(
            {
                "session": session,
                "fold_id": fold_id,
                "model": model,
                "feature": scale.feature,
                "row_count": scale.row_count,
                "missing_count": scale.missing_count,
                "missing_fraction": missing_fraction,
                "training_median": scale.median,
                "training_q25": scale.q25,
                "training_q75": scale.q75,
                "scale": scale.scale,
                "availability_gate_pass": missing_fraction <= 0.05,
            }
        )
    return rows


def coarse_conditional_rows(
    *,
    dataset: SessionDataset,
    fold_id: int,
    train_indexes: np.ndarray,
    test_indexes: np.ndarray,
    loss_h0: np.ndarray,
    loss_h1: np.ndarray,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    by_side: dict[int, list[dict[str, Any]]] = {}
    side_totals: dict[int, float] = {}
    for side_index, side in enumerate(contracts.SIDES):
        train_gap = dataset.h1_added_raw[train_indexes, side_index, 0]
        finite_gap = train_gap[np.isfinite(train_gap)]
        contracts.require(
            finite_gap.size > 0,
            "H0B_H1_FEATURE_ALLOWLIST_MISMATCH",
            f"$.coarse.{dataset.session}.{fold_id}.{side}.risk_gap",
            "no finite training cross-spread value",
        )
        gap_edges = contracts.nearest_rank_edges(
            finite_gap,
            (0.2, 0.4, 0.6, 0.8),
        )
        train_positive_dose = dataset.dose[train_indexes, side_index]
        train_positive_dose = train_positive_dose[train_positive_dose > 0.0]
        dose_edges = (
            contracts.nearest_rank_edges(
                train_positive_dose,
                (1.0 / 3.0, 2.0 / 3.0),
            )
            if train_positive_dose.size
            else (0.0, 0.0)
        )
        gap = dataset.h1_added_raw[test_indexes, side_index, 0]
        dose = dataset.dose[test_indexes, side_index]
        gap_bin = np.where(
            np.isfinite(gap),
            contracts.assign_right_closed_bins(gap, gap_edges) + 1,
            0,
        )
        dose_bin = np.where(
            dose == 0.0,
            0,
            contracts.assign_right_closed_bins(dose, dose_edges) + 1,
        )
        side_loss_h0 = loss_h0[:, side_index]
        side_loss_h1 = loss_h1[:, side_index]
        improvement = side_loss_h0 - side_loss_h1
        rows: list[dict[str, Any]] = []
        positive_total = 0.0
        for cross_bin in range(0, 6):
            for dose_index in range(0, 4):
                mask = np.logical_and(
                    gap_bin == cross_bin,
                    dose_bin == dose_index,
                )
                count = int(mask.sum())
                if count == 0:
                    positive = 0.0
                    event_count = 0
                    realized = None
                    mean_h0 = None
                    mean_h1 = None
                else:
                    positive = max(0.0, float(improvement[mask].sum()))
                    event_count = int(
                        dataset.binary_event[
                            test_indexes[mask],
                            side_index,
                        ].sum()
                    )
                    realized = event_count / count
                    mean_h0 = float(side_loss_h0[mask].mean())
                    mean_h1 = float(side_loss_h1[mask].mean())
                positive_total += positive
                rows.append(
                    {
                        "session": dataset.session,
                        "fold_id": fold_id,
                        "side": side,
                        "cross_spread_bin": cross_bin,
                        "dose_bin": dose_index,
                        "row_count": count,
                        "binary_identified_count": count,
                        "event_count": event_count,
                        "realized_rate": realized,
                        "mean_loss_h0": mean_h0,
                        "mean_loss_h1": mean_h1,
                        "positive_improvement": positive,
                        "cell_share": None,
                    }
                )
        by_side[side_index] = rows
        side_totals[side_index] = positive_total
    for side_index in range(2):
        other_index = 1 - side_index
        for row, other in zip(by_side[side_index], by_side[other_index]):
            own_total = side_totals[side_index]
            other_total = side_totals[other_index]
            if own_total > 0.0 and other_total > 0.0:
                row["cell_share"] = 0.5 * (
                    row["positive_improvement"] / own_total
                ) + 0.5 * (
                    other["positive_improvement"] / other_total
                )
            result.append(row)
    return result


def evaluate_session(
    dataset: SessionDataset,
) -> SessionEvaluation:
    complete_blocks = np.unique(
        dataset.block_start_ns[dataset.block_start_ns >= 0]
    )
    expected_count = {"jul30": 232, "aug04": 119}[dataset.session]
    contracts.require(
        complete_blocks.size == expected_count,
        "H0B_WALK_FORWARD_MISMATCH",
        f"$.complete_blocks.{dataset.session}",
        f"expected={expected_count} observed={complete_blocks.size}",
    )
    folds = contracts.build_walk_forward_folds(complete_blocks.tolist())
    expected_folds = {"jul30": 9, "aug04": 3}[dataset.session]
    contracts.require(
        len(folds) == expected_folds,
        "H0B_WALK_FORWARD_MISMATCH",
        f"$.folds.{dataset.session}",
        f"expected={expected_folds} observed={len(folds)}",
    )
    row_count = dataset.grid_ts_ns.size
    oof_fold = np.full(row_count, -1, dtype=np.int16)
    oof_loss_h0 = np.full((row_count, 2), np.nan, dtype=np.float64)
    oof_loss_h1 = np.full((row_count, 2), np.nan, dtype=np.float64)
    oof_risk_h0 = np.full((row_count, 2), np.nan, dtype=np.float64)
    oof_risk_h1 = np.full((row_count, 2), np.nan, dtype=np.float64)
    oof_h1_decile = np.full((row_count, 2), -1, dtype=np.int8)
    fold_score_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []
    coarse_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    gate_reasons: list[str] = []
    valid_fold_count = 0

    for fold in folds:
        train_mask = np.logical_and(
            np.isin(dataset.block_start_ns, fold.train_blocks),
            dataset.grid_ts_ns <= fold.purge_boundary_ns,
        )
        test_mask = np.logical_and(
            np.isin(dataset.block_start_ns, fold.test_blocks),
            dataset.grid_ts_ns >= fold.embargo_boundary_ns,
        )
        train_indexes = np.flatnonzero(train_mask)
        test_indexes = np.flatnonzero(test_mask)
        contracts.require(
            train_indexes.size > 0 and test_indexes.size > 0,
            "H0B_WALK_FORWARD_MISMATCH",
            f"$.folds.{dataset.session}.{fold.fold_id}",
            "empty train or test rows",
        )
        fitted_models: dict[str, FittedHazardModel] = {}
        test_losses: dict[str, np.ndarray] = {}
        test_risks: dict[str, np.ndarray] = {}
        training_risks: dict[str, np.ndarray] = {}
        fold_valid = True
        for model in contracts.MODELS:
            train_raw, train_side = paired_raw_features(
                dataset,
                train_indexes,
                model,
            )
            (
                train_branch,
                train_lower,
                train_upper,
                _,
                _,
            ) = paired_outcomes(dataset, train_indexes)
            fitted = fit_hazard_model(
                model=model,
                raw=train_raw,
                side=train_side,
                branch=train_branch,
                lower_ns=train_lower,
                upper_ns=train_upper,
            )
            fitted_models[model] = fitted
            feature_rows.extend(
                feature_availability_rows(
                    session=dataset.session,
                    fold_id=fold.fold_id,
                    model=model,
                    scales=fitted.scales,
                )
            )
            if model == "H1":
                h0_scales = fitted_models["H0"].scales
                for left, right in zip(
                    h0_scales,
                    fitted.scales[: len(contracts.H0_RAW_FEATURES)],
                ):
                    contracts.require(
                        left == right,
                        "H0B_MISSING_VALUE_POLICY_MISMATCH",
                        f"$.folds.{dataset.session}.{fold.fold_id}",
                        f"H0 scale differs inside H1 for {left.feature}",
                    )
            if not fitted.converged:
                fold_valid = False
                gate_reasons.append(
                    f"{dataset.session}:fold_{fold.fold_id}:{model}:optimizer"
                )
            test_raw, test_side = paired_raw_features(
                dataset,
                test_indexes,
                model,
            )
            (
                test_branch,
                test_lower,
                test_upper,
                _,
                test_event,
            ) = paired_outcomes(dataset, test_indexes)
            test_loss, test_risk = predict_model(
                fitted=fitted,
                raw=test_raw,
                side=test_side,
                branch=test_branch,
                lower_ns=test_lower,
                upper_ns=test_upper,
            )
            training_loss, training_risk = predict_model(
                fitted=fitted,
                raw=train_raw,
                side=train_side,
                branch=train_branch,
                lower_ns=train_lower,
                upper_ns=train_upper,
            )
            del training_loss
            test_losses[model] = test_loss.reshape(-1, 2)
            test_risks[model] = test_risk.reshape(-1, 2)
            training_risks[model] = training_risk.reshape(-1, 2)
            side_metrics = []
            for side_index, side_name in enumerate(contracts.SIDES):
                brier, binary_loss = binary_metrics(
                    test_risks[model][:, side_index],
                    test_event.reshape(-1, 2)[:, side_index],
                )
                side_metrics.append(
                    (
                        float(test_losses[model][:, side_index].mean()),
                        brier,
                        binary_loss,
                    )
                )
                reliability_rows.extend(
                    bin_diagnostic_rows(
                        session=dataset.session,
                        fold_id=fold.fold_id,
                        model=model,
                        side=side_name,
                        training_risk=training_risks[model][
                            :, side_index
                        ],
                        test_risk=test_risks[model][:, side_index],
                        test_event=test_event.reshape(-1, 2)[
                            :, side_index
                        ],
                        path="reliability",
                    )
                )
                decile_rows.extend(
                    bin_diagnostic_rows(
                        session=dataset.session,
                        fold_id=fold.fold_id,
                        model=model,
                        side=side_name,
                        training_risk=training_risks[model][
                            :, side_index
                        ],
                        test_risk=test_risks[model][:, side_index],
                        test_event=test_event.reshape(-1, 2)[
                            :, side_index
                        ],
                        path="decile",
                    )
                )
            fold_score_rows.append(
                {
                    "session": dataset.session,
                    "fold_id": fold.fold_id,
                    "model": model,
                    "train_first_block": str(fold.train_blocks[0]),
                    "train_last_block": str(fold.train_blocks[-1]),
                    "test_first_block": str(fold.test_blocks[0]),
                    "test_last_block": str(fold.test_blocks[-1]),
                    "purge_ns": contracts.PURGE_NS,
                    "embargo_ns": contracts.EMBARGO_NS,
                    "train_row_count": int(train_indexes.size * 2),
                    "test_row_count": int(test_indexes.size * 2),
                    "ask_test_rows": int(test_indexes.size),
                    "bid_test_rows": int(test_indexes.size),
                    "converged": fitted.converged,
                    "iterations": fitted.iterations,
                    "objective": fitted.objective,
                    "ask_interval_log_loss": side_metrics[0][0],
                    "bid_interval_log_loss": side_metrics[1][0],
                    "session_interval_log_loss": 0.5
                    * (side_metrics[0][0] + side_metrics[1][0]),
                    "ask_brier": side_metrics[0][1],
                    "bid_brier": side_metrics[1][1],
                    "session_brier": 0.5
                    * (side_metrics[0][1] + side_metrics[1][1]),
                    "ask_binary_log_loss": side_metrics[0][2],
                    "bid_binary_log_loss": side_metrics[1][2],
                    "session_binary_log_loss": 0.5
                    * (side_metrics[0][2] + side_metrics[1][2]),
                }
            )
        if fold_valid:
            valid_fold_count += 1
        oof_fold[test_indexes] = fold.fold_id
        oof_loss_h0[test_indexes] = test_losses["H0"]
        oof_loss_h1[test_indexes] = test_losses["H1"]
        oof_risk_h0[test_indexes] = test_risks["H0"]
        oof_risk_h1[test_indexes] = test_risks["H1"]
        for side_index in range(2):
            edges = contracts.nearest_rank_edges(
                training_risks["H1"][:, side_index],
                [index / 10.0 for index in range(1, 10)],
            )
            oof_h1_decile[test_indexes, side_index] = (
                contracts.assign_right_closed_bins(
                    test_risks["H1"][:, side_index],
                    edges,
                )
            )
            entry = contracts.nearest_rank(
                training_risks["H1"][:, side_index],
                0.90,
            )
            exit_threshold = contracts.nearest_rank(
                training_risks["H1"][:, side_index],
                0.70,
            )
            regime_rows.extend(
                detect_regimes(
                    dataset=dataset,
                    fold_id=fold.fold_id,
                    anchor_indexes=test_indexes,
                    side_index=side_index,
                    risk=test_risks["H1"][:, side_index],
                    entry_threshold=entry,
                    exit_threshold=exit_threshold,
                )
            )
        coarse_rows.extend(
            coarse_conditional_rows(
                dataset=dataset,
                fold_id=fold.fold_id,
                train_indexes=train_indexes,
                test_indexes=test_indexes,
                loss_h0=test_losses["H0"],
                loss_h1=test_losses["H1"],
            )
        )

    h1_missing_gate = True
    complete_anchor = dataset.block_start_ns >= 0
    for feature_index, feature in enumerate(contracts.H1_ADDED_RAW_FEATURES):
        missing_fraction = float(
            (
                ~np.isfinite(
                    dataset.h1_added_raw[
                        complete_anchor,
                        :,
                        feature_index,
                    ]
                )
            ).mean()
        )
        if missing_fraction > 0.05:
            h1_missing_gate = False
            gate_reasons.append(
                f"{dataset.session}:{feature}:missing_fraction={missing_fraction}"
            )
    data_quality = (
        valid_fold_count >= 3
        and h1_missing_gate
        and np.isfinite(oof_loss_h0[oof_fold >= 0]).all()
        and np.isfinite(oof_loss_h1[oof_fold >= 0]).all()
    )
    if valid_fold_count < 3:
        gate_reasons.append(
            f"{dataset.session}:valid_fold_count={valid_fold_count}"
        )
    return SessionEvaluation(
        fold_score_rows=fold_score_rows,
        feature_rows=feature_rows,
        reliability_rows=reliability_rows,
        decile_rows=decile_rows,
        coarse_rows=coarse_rows,
        regime_rows=regime_rows,
        oof_fold=oof_fold,
        oof_loss_h0=oof_loss_h0,
        oof_loss_h1=oof_loss_h1,
        oof_risk_h0=oof_risk_h0,
        oof_risk_h1=oof_risk_h1,
        oof_h1_decile=oof_h1_decile,
        data_quality=data_quality,
        gate_reasons=gate_reasons,
    )


RQ1_C_SOURCE = r"""
#include <stdint.h>
#include <stddef.h>

int h0b_rq1_replicate(
    const int64_t *timestamps,
    const int16_t *segments,
    const int8_t *strata,
    const int8_t *outcomes,
    const int32_t *block_codes,
    const int32_t *pool,
    const int32_t *pool_offsets,
    const double *restart_uniform,
    const uint64_t *source_uniform,
    int64_t row_count,
    int32_t block_count,
    double restart_probability,
    int64_t *event_counts
) {
    for (int64_t i = 0; i < (int64_t)block_count * 2; ++i) {
        event_counts[i] = 0;
    }
    int32_t source = -1;
    for (int64_t i = 0; i < row_count; ++i) {
        int restart = (i == 0) || (restart_uniform[i] < restart_probability);
        if (i > 0 && (
            segments[i] != segments[i - 1] ||
            timestamps[i] != timestamps[i - 1] + 10000000LL
        )) {
            restart = 1;
        }
        if (!restart) {
            int32_t next = source + 1;
            if (
                next >= row_count ||
                segments[next] != segments[i] ||
                timestamps[next] != timestamps[source] + 10000000LL ||
                strata[next] != strata[i]
            ) {
                restart = 1;
            } else {
                source = next;
            }
        }
        if (restart) {
            int32_t group = (int32_t)segments[i] * 5 + (int32_t)strata[i];
            int32_t begin = pool_offsets[group];
            int32_t end = pool_offsets[group + 1];
            if (end <= begin) {
                return 2;
            }
            uint64_t width = (uint64_t)(end - begin);
            source = pool[begin + (int32_t)(source_uniform[i] % width)];
        }
        int32_t block = block_codes[i];
        if (block < 0 || block >= block_count) {
            return 3;
        }
        event_counts[(int64_t)block * 2] += outcomes[(int64_t)source * 2];
        event_counts[(int64_t)block * 2 + 1] += outcomes[(int64_t)source * 2 + 1];
    }
    return 0;
}
"""


def load_rq1_kernel(work_root: Path) -> Any:
    source = Path(work_root) / "rq1_stationary_kernel.c"
    library = Path(work_root) / "rq1_stationary_kernel.dylib"
    source.write_text(RQ1_C_SOURCE, encoding="ascii")
    command = [
        "/usr/bin/clang",
        "-O3",
        "-dynamiclib",
        str(source),
        "-o",
        str(library),
    ]
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    contracts.require(
        result.returncode == 0,
        "H0B_RQ1_NULL_MISMATCH",
        str(source),
        result.stdout + result.stderr,
    )
    loaded = ctypes.CDLL(str(library))
    function = loaded.h0b_rq1_replicate
    function.argtypes = [
        np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.int16, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.int8, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.int8, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.uint64, flags="C_CONTIGUOUS"),
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_double,
        np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS"),
    ]
    function.restype = ctypes.c_int
    return function


def cadence_strata(dataset: SessionDataset, indexes: np.ndarray) -> np.ndarray:
    selected = np.asarray(indexes, dtype=np.int64)
    result = np.zeros(selected.size, dtype=np.int8)
    cadence = dataset.h0_raw[selected, 3]
    segments = dataset.segment_code[selected]
    for segment_code in np.unique(segments):
        mask = segments == segment_code
        values = cadence[mask]
        finite = values[np.isfinite(values)]
        contracts.require(
            finite.size > 0,
            "H0B_RQ1_NULL_MISMATCH",
            f"$.cadence.{dataset.session}.{segment_code}",
            "no finite cadence anchors",
        )
        edges = contracts.nearest_rank_edges(
            finite,
            (0.25, 0.50, 0.75),
        )
        assigned = np.zeros(values.size, dtype=np.int8)
        finite_mask = np.isfinite(values)
        assigned[finite_mask] = (
            contracts.assign_right_closed_bins(
                values[finite_mask],
                edges,
            )
            + 1
        )
        result[mask] = assigned
    return result


def rq1_block_rows(
    dataset: SessionDataset,
) -> tuple[list[dict[str, Any]], float, float, float]:
    complete = dataset.block_start_ns >= 0
    blocks = np.unique(dataset.block_start_ns[complete])
    rows = []
    variances = []
    for side_index, side in enumerate(contracts.SIDES):
        rates = []
        for block in blocks:
            mask = dataset.block_start_ns == block
            denominator = int(mask.sum())
            events = int(dataset.binary_event[mask, side_index].sum())
            rate = events / denominator
            rates.append(rate)
            rows.append(
                {
                    "session": dataset.session,
                    "side": side,
                    "absolute_block_id": str(int(block)),
                    "block_start_ns": int(block),
                    "block_end_ns": int(block + contracts.BLOCK_NS),
                    "binary_identified_count": denominator,
                    "event_count": events,
                    "block_rate": rate,
                }
            )
        variances.append(float(np.var(rates, ddof=1)))
    return rows, variances[0], variances[1], 0.5 * sum(variances)


def rq1_stationary_null(
    *,
    dataset: SessionDataset,
    mean_rows: int,
    work_root: Path,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> np.ndarray:
    indexes = np.flatnonzero(dataset.block_start_ns >= 0)
    timestamps = np.ascontiguousarray(
        dataset.grid_ts_ns[indexes],
        dtype=np.int64,
    )
    segments = np.ascontiguousarray(
        dataset.segment_code[indexes],
        dtype=np.int16,
    )
    strata = np.ascontiguousarray(
        cadence_strata(dataset, indexes),
        dtype=np.int8,
    )
    outcomes = np.ascontiguousarray(
        dataset.binary_event[indexes].astype(np.int8),
    )
    unique_blocks, block_codes = np.unique(
        dataset.block_start_ns[indexes],
        return_inverse=True,
    )
    block_codes = np.ascontiguousarray(block_codes, dtype=np.int32)
    group_count = len(dataset.segment_ids) * 5
    pools = []
    offsets = [0]
    for group in range(group_count):
        segment_code = group // 5
        stratum = group % 5
        pool = np.flatnonzero(
            np.logical_and(
                segments == segment_code,
                strata == stratum,
            )
        )
        if np.any(
            np.logical_and(
                segments == segment_code,
                strata == stratum,
            )
        ):
            contracts.require(
                pool.size > 0,
                "H0B_RQ1_NULL_MISMATCH",
                f"$.pool.{group}",
                "observed target stratum has no source anchors",
            )
        pools.append(pool.astype(np.int32))
        offsets.append(offsets[-1] + pool.size)
    pool_values = np.ascontiguousarray(
        np.concatenate(pools) if pools else np.asarray([], dtype=np.int32),
        dtype=np.int32,
    )
    pool_offsets = np.ascontiguousarray(offsets, dtype=np.int32)
    denominators = np.bincount(
        block_codes,
        minlength=unique_blocks.size,
    ).astype(np.float64)
    kernel = load_rq1_kernel(work_root)
    namespace = f"rq1|{dataset.session}|mean_rows={mean_rows}"
    generator = np.random.Generator(
        np.random.PCG64(
            contracts.derived_seed(PRIMARY_NULL_SEED, namespace)
        )
    )
    output = np.empty(replicates, dtype=np.float64)
    event_counts = np.empty((unique_blocks.size, 2), dtype=np.int64)
    restart_probability = 1.0 / mean_rows
    max_uint64 = np.iinfo(np.uint64).max
    for replicate in range(replicates):
        restart_uniform = np.ascontiguousarray(
            generator.random(indexes.size),
            dtype=np.float64,
        )
        source_uniform = np.ascontiguousarray(
            generator.integers(
                0,
                max_uint64,
                size=indexes.size,
                dtype=np.uint64,
            ),
            dtype=np.uint64,
        )
        return_code = kernel(
            timestamps,
            segments,
            strata,
            outcomes,
            block_codes,
            pool_values,
            pool_offsets,
            restart_uniform,
            source_uniform,
            indexes.size,
            unique_blocks.size,
            restart_probability,
            event_counts,
        )
        contracts.require(
            return_code == 0,
            "H0B_RQ1_NULL_MISMATCH",
            f"$.replicate.{replicate}",
            f"kernel_return_code={return_code}",
        )
        ask_rates = event_counts[:, 0] / denominators
        bid_rates = event_counts[:, 1] / denominators
        output[replicate] = 0.5 * (
            np.var(ask_rates, ddof=1)
            + np.var(bid_rates, ddof=1)
        )
    return output


def cluster_multiplier_ratios(
    *,
    loss_h0: np.ndarray,
    loss_h1: np.ndarray,
    cluster_ids: np.ndarray,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> np.ndarray:
    clusters, inverse = np.unique(cluster_ids, return_inverse=True)
    cluster_count = clusters.size
    counts = np.bincount(inverse, minlength=cluster_count).astype(np.float64)
    sums_h0 = np.column_stack(
        (
            np.bincount(
                inverse,
                weights=loss_h0[:, 0],
                minlength=cluster_count,
            ),
            np.bincount(
                inverse,
                weights=loss_h0[:, 1],
                minlength=cluster_count,
            ),
        )
    )
    sums_h1 = np.column_stack(
        (
            np.bincount(
                inverse,
                weights=loss_h1[:, 0],
                minlength=cluster_count,
            ),
            np.bincount(
                inverse,
                weights=loss_h1[:, 1],
                minlength=cluster_count,
            ),
        )
    )
    generator = np.random.Generator(np.random.PCG64(seed))
    ratios = []
    for _ in range(replicates):
        weights = generator.exponential(1.0, cluster_count)
        denominator = float(weights @ counts)
        if denominator <= 0.0:
            continue
        h0_side = (weights @ sums_h0) / denominator
        h1_side = (weights @ sums_h1) / denominator
        h0_session = 0.5 * float(h0_side.sum())
        h1_session = 0.5 * float(h1_side.sum())
        if h0_session > 0.0 and math.isfinite(h1_session):
            ratios.append(h1_session / h0_session)
    return np.asarray(ratios, dtype=np.float64)


def rq2_session_score(
    *,
    dataset: SessionDataset,
    evaluation: SessionEvaluation,
) -> tuple[dict[str, Any], dict[str, Any]]:
    valid = evaluation.oof_fold >= 0
    contracts.require(
        valid.any(),
        "H0B_RQ2_SCORE_MISMATCH",
        f"$.oof.{dataset.session}",
        "no OOF rows",
    )
    h0_loss = evaluation.oof_loss_h0[valid]
    h1_loss = evaluation.oof_loss_h1[valid]
    h0_risk = evaluation.oof_risk_h0[valid]
    h1_risk = evaluation.oof_risk_h1[valid]
    event = dataset.binary_event[valid]
    side_h0_loss = h0_loss.mean(axis=0)
    side_h1_loss = h1_loss.mean(axis=0)
    session_h0_loss = 0.5 * float(side_h0_loss.sum())
    session_h1_loss = 0.5 * float(side_h1_loss.sum())
    normalized = session_h1_loss / session_h0_loss
    h0_binary = [binary_metrics(h0_risk[:, side], event[:, side]) for side in range(2)]
    h1_binary = [binary_metrics(h1_risk[:, side], event[:, side]) for side in range(2)]
    h0_brier = 0.5 * (h0_binary[0][0] + h0_binary[1][0])
    h1_brier = 0.5 * (h1_binary[0][0] + h1_binary[1][0])
    h0_binary_loss = 0.5 * (h0_binary[0][1] + h0_binary[1][1])
    h1_binary_loss = 0.5 * (h1_binary[0][1] + h1_binary[1][1])
    spreads = []
    for side in range(2):
        decile = evaluation.oof_h1_decile[valid, side]
        bottom = event[:, side][decile == 0]
        top = event[:, side][decile == 9]
        contracts.require(
            bottom.size > 0 and top.size > 0,
            "H0B_RQ2_SCORE_MISMATCH",
            f"$.deciles.{dataset.session}.{side}",
            "empty top or bottom decile",
        )
        spreads.append(float(top.mean() - bottom.mean()))
    top_bottom_spread = 0.5 * sum(spreads)
    finite_shares = [
        float(row["cell_share"])
        for row in evaluation.coarse_rows
        if row["cell_share"] is not None
    ]
    concentration_valid = bool(finite_shares)
    max_share = max(finite_shares) if finite_shares else None
    block_ids = dataset.block_start_ns[valid]
    time_ratios = cluster_multiplier_ratios(
        loss_h0=h0_loss,
        loss_h1=h1_loss,
        cluster_ids=block_ids,
        seed=contracts.derived_seed(
            TIME_BOOTSTRAP_SEED,
            f"rq2_time|{dataset.session}",
        ),
    )
    flow_ids = dataset.flow_unit[valid]
    unique_flow, flow_counts = np.unique(flow_ids, return_counts=True)
    largest_flow_share = float(flow_counts.max() / flow_counts.sum())
    flow_ratios = cluster_multiplier_ratios(
        loss_h0=h0_loss,
        loss_h1=h1_loss,
        cluster_ids=flow_ids,
        seed=contracts.derived_seed(
            FLOW_BOOTSTRAP_SEED,
            f"rq2_flow|{dataset.session}",
        ),
    )
    time_valid = (
        np.unique(block_ids).size >= 20
        and time_ratios.size >= 1900
    )
    flow_valid = (
        unique_flow.size >= 6
        and largest_flow_share <= 0.50
        and flow_ratios.size >= 1900
    )
    time_lower = (
        contracts.nearest_rank(time_ratios, 0.05) if time_valid else None
    )
    time_upper = (
        contracts.nearest_rank(time_ratios, 0.95) if time_valid else None
    )
    flow_lower = (
        contracts.nearest_rank(flow_ratios, 0.05) if flow_valid else None
    )
    flow_upper = (
        contracts.nearest_rank(flow_ratios, 0.95) if flow_valid else None
    )
    valid_gate = (
        evaluation.data_quality
        and concentration_valid
        and time_valid
        and flow_valid
    )
    passed = (
        normalized <= 0.99
        and h1_brier / h0_brier <= 1.0
        and h1_binary_loss / h0_binary_loss <= 1.0
        and top_bottom_spread > 0.0
        and max_share is not None
        and max_share <= 0.50
        and time_upper is not None
        and time_upper < 1.0
        and flow_upper is not None
        and flow_upper < 1.0
    )
    gate_reason = (
        ""
        if valid_gate
        else "inconclusive_data_quality_or_coverage"
    )
    row = {
        "session": dataset.session,
        "h0_interval_log_loss": session_h0_loss,
        "h1_interval_log_loss": session_h1_loss,
        "normalized_interval_log_loss_h1_h0": normalized,
        "h0_brier": h0_brier,
        "h1_brier": h1_brier,
        "brier_ratio_h1_h0": h1_brier / h0_brier,
        "h0_binary_log_loss": h0_binary_loss,
        "h1_binary_log_loss": h1_binary_loss,
        "binary_log_loss_ratio_h1_h0": h1_binary_loss / h0_binary_loss,
        "top_bottom_realized_rate_spread": top_bottom_spread,
        "max_positive_cell_share": max_share,
        "time_ci_lower": time_lower,
        "time_ci_upper": time_upper,
        "flow_ci_lower": flow_lower,
        "flow_ci_upper": flow_upper,
        "rq2_pass": passed if valid_gate else None,
        "gate_reason": gate_reason,
    }
    diagnostics = {
        "time_distinct_blocks": int(np.unique(block_ids).size),
        "time_finite_replicates": int(time_ratios.size),
        "flow_distinct_units": int(unique_flow.size),
        "flow_largest_unit_row_share": largest_flow_share,
        "flow_finite_replicates": int(flow_ratios.size),
    }
    return row, diagnostics


def rq3_rows(
    *,
    session: str,
    regimes: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    summary_rows: list[dict[str, Any]] = []
    side_facts: dict[int, dict[str, Any]] = {}
    for side_index, side in enumerate(contracts.SIDES):
        selected = [row for row in regimes if row["side"] == side]
        durations = np.asarray(
            [int(row["total_dwell_ns"]) for row in selected],
            dtype=np.float64,
        )
        censored = np.asarray(
            [bool(row["censored"]) for row in selected],
            dtype=bool,
        )
        blocks = np.asarray(
            [str(row["detection_block_id"]) for row in selected],
            dtype=object,
        )
        observed = durations[~censored]
        point_median = (
            contracts.kaplan_meier_median(durations, censored)
            if durations.size
            else None
        )
        distinct_blocks = int(np.unique(blocks).size) if blocks.size else 0
        bootstrap = np.asarray([], dtype=np.float64)
        if durations.size and distinct_blocks >= 20 and point_median is not None:
            bootstrap = contracts.weighted_cluster_km_bootstrap(
                durations,
                censored,
                blocks,
                seed=contracts.derived_seed(
                    RQ3_BOOTSTRAP_SEED,
                    f"rq3|{session}|{side}",
                ),
                replicates=BOOTSTRAP_REPLICATES,
            )
        valid = (
            durations.size > 0
            and distinct_blocks >= 20
            and point_median is not None
            and bootstrap.size >= 1900
        )
        lower_total = (
            contracts.nearest_rank(bootstrap, 0.05) if valid else None
        )
        side_facts[side_index] = {
            "regime_count": int(durations.size),
            "observed_count": int((~censored).sum()),
            "censored_count": int(censored.sum()),
            "identified_fraction": (
                float((~censored).mean()) if durations.size else None
            ),
            "dwell_p10": (
                contracts.nearest_rank(observed, 0.10)
                if observed.size
                else None
            ),
            "dwell_p50": (
                contracts.nearest_rank(observed, 0.50)
                if observed.size
                else None
            ),
            "dwell_p90": (
                contracts.nearest_rank(observed, 0.90)
                if observed.size
                else None
            ),
            "km_median": point_median,
            "lower_total": lower_total,
            "distinct_blocks": distinct_blocks,
            "bootstrap_count": int(bootstrap.size),
            "valid": valid,
        }
        for latency_ms, role in sorted(contracts.LATENCY_ROLES.items()):
            latency_ns = latency_ms * 1_000_000
            fact = side_facts[side_index]
            summary_rows.append(
                {
                    "session": session,
                    "side": side,
                    "latency_ms": latency_ms,
                    "latency_role": role,
                    "regime_count": fact["regime_count"],
                    "observed_exit_count": fact["observed_count"],
                    "right_censored_count": fact["censored_count"],
                    "identified_fraction": fact["identified_fraction"],
                    "dwell_p10_ms": (
                        fact["dwell_p10"] / 1_000_000.0
                        if fact["dwell_p10"] is not None
                        else None
                    ),
                    "dwell_p50_ms": (
                        fact["dwell_p50"] / 1_000_000.0
                        if fact["dwell_p50"] is not None
                        else None
                    ),
                    "dwell_p90_ms": (
                        fact["dwell_p90"] / 1_000_000.0
                        if fact["dwell_p90"] is not None
                        else None
                    ),
                    "km_total_dwell_p50_ms": (
                        fact["km_median"] / 1_000_000.0
                        if fact["km_median"] is not None
                        else None
                    ),
                    "residual_dwell_p50_ms": (
                        (fact["km_median"] - latency_ns) / 1_000_000.0
                        if fact["km_median"] is not None
                        else None
                    ),
                    "residual_dwell_lower95_ms": (
                        (fact["lower_total"] - latency_ns) / 1_000_000.0
                        if fact["lower_total"] is not None
                        else None
                    ),
                    "switching_rate_per_minute": (
                        fact["regime_count"]
                        / max(1, fact["distinct_blocks"])
                    ),
                    "distinct_detection_blocks": fact["distinct_blocks"],
                    "identifiable_bootstrap_replicates": fact[
                        "bootstrap_count"
                    ],
                }
            )
    actionability_rows = []
    for latency_ms, role in sorted(contracts.LATENCY_ROLES.items()):
        latency_ns = latency_ms * 1_000_000
        ask = side_facts[0]
        bid = side_facts[1]
        valid = bool(ask["valid"] and bid["valid"])
        ask_residual = (
            (ask["km_median"] - latency_ns) / 1_000_000.0
            if ask["km_median"] is not None
            else None
        )
        bid_residual = (
            (bid["km_median"] - latency_ns) / 1_000_000.0
            if bid["km_median"] is not None
            else None
        )
        ask_lower = (
            (ask["lower_total"] - latency_ns) / 1_000_000.0
            if ask["lower_total"] is not None
            else None
        )
        bid_lower = (
            (bid["lower_total"] - latency_ns) / 1_000_000.0
            if bid["lower_total"] is not None
            else None
        )
        equal_fraction = (
            0.5
            * (
                ask["identified_fraction"]
                + bid["identified_fraction"]
            )
            if ask["identified_fraction"] is not None
            and bid["identified_fraction"] is not None
            else None
        )
        equal_residual = (
            0.5 * (ask_residual + bid_residual)
            if ask_residual is not None and bid_residual is not None
            else None
        )
        equal_lower = (
            0.5 * (ask_lower + bid_lower)
            if ask_lower is not None and bid_lower is not None
            else None
        )
        passed = (
            equal_fraction is not None
            and equal_fraction >= 0.90
            and equal_lower is not None
            and equal_lower > 0.0
        )
        actionability_rows.append(
            {
                "session": session,
                "latency_ms": latency_ms,
                "latency_role": role,
                "primary": latency_ms == contracts.PRIMARY_LATENCY_MS,
                "ask_identified_fraction": ask["identified_fraction"],
                "bid_identified_fraction": bid["identified_fraction"],
                "equal_weight_identified_fraction": equal_fraction,
                "ask_residual_p50_ms": ask_residual,
                "bid_residual_p50_ms": bid_residual,
                "equal_weight_residual_p50_ms": equal_residual,
                "ask_lower95_ms": ask_lower,
                "bid_lower95_ms": bid_lower,
                "bonferroni90_equal_weight_lower_ms": equal_lower,
                "session_pass": passed if valid else None,
                "can_rescue_primary": False,
            }
        )
    primary_row = next(
        row
        for row in actionability_rows
        if row["latency_ms"] == contracts.PRIMARY_LATENCY_MS
    )
    primary_pass = bool(primary_row["session_pass"] is True)
    return summary_rows, actionability_rows, primary_pass


def latency_role_rows() -> list[dict[str, Any]]:
    validate_latency_role_contract(
        contracts.LATENCY_ROLES,
        primary_latency_ms=contracts.PRIMARY_LATENCY_MS,
        diagnostic_latency_ms=contracts.DIAGNOSTIC_LATENCY_MS,
    )
    rows = []
    for latency_ms, role in sorted(contracts.LATENCY_ROLES.items()):
        rows.append(
            {
                "latency_ms": latency_ms,
                "latency_role": role,
                "primary": latency_ms == contracts.PRIMARY_LATENCY_MS,
                "diagnostic": latency_ms == contracts.DIAGNOSTIC_LATENCY_MS,
                "can_rescue_primary": False,
                "source_authority": (
                    "accepted_0822T002_measurement"
                    if latency_ms
                    in {
                        contracts.PRIMARY_LATENCY_MS,
                        contracts.DIAGNOSTIC_LATENCY_MS,
                    }
                    else "accepted_superseding_tuple_sensitivity"
                ),
            }
        )
    return rows


def sort_output_rows(path: str, rows: list[dict[str, Any]]) -> None:
    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        session = contracts.SESSION_ORDER.get(str(row.get("session", "")), -1)
        side = contracts.SIDE_ORDER.get(str(row.get("side", "")), -1)
        model = contracts.MODEL_ORDER.get(str(row.get("model", "")), -1)
        scope = 1 if row.get("scope") == "session" else 0
        segment = str(row.get("segment_id", ""))
        fold = int(row.get("fold_id", 0) or 0)
        latency = int(row.get("latency_ms", 0) or 0)
        block = str(row.get("absolute_block_id", ""))
        bin_value = int(
            row.get(
                "reliability_bin",
                row.get(
                    "decile",
                    row.get("cross_spread_bin", 0),
                ),
            )
            or 0
        )
        dose_bin = int(row.get("dose_bin", 0) or 0)
        regime = int(row.get("regime_id", 0) or 0)
        return (
            session,
            scope,
            segment,
            fold,
            model,
            side,
            latency,
            block,
            bin_value,
            dose_bin,
            regime,
        )

    rows.sort(key=key)


def write_primary_outputs(
    root: Path,
    outputs: Mapping[str, list[dict[str, Any]]],
    primary_classification: Mapping[str, Any],
) -> None:
    for relative, rows in outputs.items():
        sort_output_rows(relative, rows)
        fields = contracts.CSV_HEADERS[relative]
        target = root / relative
        if relative.endswith(".csv.gz"):
            contracts.write_gzip_csv_exact(target, rows, fields)
        else:
            contracts.write_csv_exact(target, rows, fields)
    write_json(root / "primary_classification.json", primary_classification)


def primary_results_identity(root: Path) -> str:
    return contracts.inventory_sha256(root, contracts.PRIMARY_RESULT_FILES)


def run_h0b1(
    *,
    build_root: Path,
) -> dict[str, Any]:
    root = Path(build_root)
    permit = validate_permit(root)
    h0a_sessions = h0a_support.load_sessions(CANONICAL_SOURCE_ROOT)
    spec_by_session = {
        spec.session_id: spec for spec in h0a_support.SESSION_SPECS
    }
    dose_events = stage3_dose_events()
    components = stage2_components()
    outputs: dict[str, list[dict[str, Any]]] = {
        path: []
        for path in contracts.CSV_HEADERS
        if path
        not in {
            "preoutcome_source_inventory.csv",
            "diagnostics/stage4_landmark_crosscheck.csv",
        }
    }
    formal_facts: dict[str, dict[str, Any]] = {}
    gate_reasons: list[str] = []
    diagnostics: dict[str, Any] = {}
    for session in ("jul30", "aug03", "aug04"):
        dataset = build_session_dataset(
            spec=spec_by_session[session],
            h0a_segments=h0a_sessions[session],
            dose_events=dose_events,
            components=components,
        )
        outputs["censoring_disposition.csv"].extend(
            dataset.censoring_rows
        )
        outputs["exclusion_counts.csv"].extend(dataset.exclusion_rows)
        outputs["support_outcome_projection_commitments.csv"].extend(
            dataset.projection_rows
        )
        if session not in contracts.FORMAL_SESSIONS:
            continue
        rq1_rows, ask_variance, bid_variance, observed_d = rq1_block_rows(
            dataset
        )
        outputs["rq1_block_rates.csv"].extend(rq1_rows)
        rq1_work = root / f"rq1_work_{session}"
        rq1_work.mkdir(parents=True, exist_ok=False)
        primary_null = rq1_stationary_null(
            dataset=dataset,
            mean_rows=500,
            work_root=rq1_work,
        )
        robustness_250 = rq1_stationary_null(
            dataset=dataset,
            mean_rows=250,
            work_root=rq1_work,
        )
        robustness_1000 = rq1_stationary_null(
            dataset=dataset,
            mean_rows=1000,
            work_root=rq1_work,
        )
        null_p95 = contracts.nearest_rank(primary_null, 0.95)
        rq1_pass = observed_d > null_p95
        outputs["rq1_dispersion_tests.csv"].append(
            {
                "session": session,
                "ask_variance": ask_variance,
                "bid_variance": bid_variance,
                "observed_d_session": observed_d,
                "primary_mean_run_rows": 500,
                "null_replicates": BOOTSTRAP_REPLICATES,
                "null_p95": null_p95,
                "primary_pass": rq1_pass,
                "robustness_250_p95": contracts.nearest_rank(
                    robustness_250,
                    0.95,
                ),
                "robustness_1000_p95": contracts.nearest_rank(
                    robustness_1000,
                    0.95,
                ),
            }
        )
        evaluation = evaluate_session(dataset)
        outputs["rq2_oof_fold_scores.csv"].extend(
            evaluation.fold_score_rows
        )
        outputs["rq2_feature_availability.csv"].extend(
            evaluation.feature_rows
        )
        outputs["rq2_reliability.csv"].extend(
            evaluation.reliability_rows
        )
        outputs["rq2_risk_deciles.csv"].extend(evaluation.decile_rows)
        outputs["rq2_coarse_conditional_risk.csv"].extend(
            evaluation.coarse_rows
        )
        outputs["diagnostics/regime_intervals.csv.gz"].extend(
            evaluation.regime_rows
        )
        rq2_row, rq2_diagnostics = rq2_session_score(
            dataset=dataset,
            evaluation=evaluation,
        )
        outputs["rq2_session_scores.csv"].append(rq2_row)
        (
            rq3_summary,
            rq3_actionability,
            rq3_primary_pass,
        ) = rq3_rows(
            session=session,
            regimes=evaluation.regime_rows,
        )
        outputs["rq3_regime_summary.csv"].extend(rq3_summary)
        outputs["rq3_latency_actionability.csv"].extend(
            rq3_actionability
        )
        rq3_primary_row = next(
            row
            for row in rq3_actionability
            if row["latency_ms"] == contracts.PRIMARY_LATENCY_MS
        )
        rq3_valid = rq3_primary_row["session_pass"] is not None
        rq2_valid = rq2_row["rq2_pass"] is not None
        data_quality = (
            evaluation.data_quality and rq2_valid and rq3_valid
        )
        formal_facts[session] = {
            "rq1": rq1_pass if data_quality else None,
            "rq2": rq2_row["rq2_pass"] if data_quality else None,
            "rq3": rq3_primary_pass if data_quality else None,
            "data_quality": data_quality,
        }
        gate_reasons.extend(evaluation.gate_reasons)
        if not rq2_valid:
            gate_reasons.append(f"{session}:rq2_bootstrap_or_concentration")
        if not rq3_valid:
            gate_reasons.append(f"{session}:rq3_identification")
        diagnostics[session] = {
            "rq2": rq2_diagnostics,
            "rq1_primary_null_replicates": int(primary_null.size),
            "rq3_regime_count": len(evaluation.regime_rows),
        }
        del dataset
        del evaluation
    outputs["diagnostics/latency_scenario_roles.csv"] = (
        latency_role_rows()
    )
    classification, precedence, classification_reasons = (
        contracts.classify_primary(formal_facts)
    )
    gate_reasons.extend(classification_reasons)
    primary_classification = {
        "schema_version": "skhynix_stage_h0b_primary_classification_v1",
        "task_id": contracts.TASK_ID,
        "classification": classification,
        "precedence_path": precedence,
        "gate_reasons": gate_reasons,
        "formal_session_facts": formal_facts,
        "latency_roles": {
            "850": {
                "role": contracts.LATENCY_ROLES[850],
                "can_rescue_primary": False,
            },
            "6600": {
                "role": contracts.LATENCY_ROLES[6600],
                "primary": True,
            },
        },
        "claim_limit": contracts.CLAIM_LIMIT,
    }
    write_primary_outputs(root, outputs, primary_classification)
    ledger_path = root / "outcome_access_ledger.json"
    ledger = read_json(ledger_path)
    permit_sha = contracts.sha256_file(
        root / "outcome_access_permit.json"
    )
    sequence = len(ledger["events"]) + 1
    with SEMANTIC_INVENTORY_PATH.open(
        newline="", encoding="utf-8"
    ) as handle:
        inventory_rows = list(csv.DictReader(handle))
    for row in inventory_rows:
        if row["source_role"] not in {
            "r0_binance_bookticker",
            "r0_hyperliquid_bbo",
            "accepted_stage2_primary",
            "accepted_stage3_primary",
        }:
            continue
        ledger["events"].append(
            {
                "sequence": sequence,
                "process_role": "H0B1",
                "phase": "primary_outcome",
                "relative_path": row["relative_path"],
                "access_kind": (
                    "public_bbo_outcome_scan"
                    if row["source_role"]
                    in {
                        "r0_binance_bookticker",
                        "r0_hyperliquid_bbo",
                    }
                    else "accepted_public_feature_metadata"
                ),
                "bytes_read": int(row["bytes"]),
                "permit_sha256": permit_sha,
                "admitted": True,
            }
        )
        sequence += 1
    write_json(ledger_path, ledger, fsync=True)
    primary_identity = primary_results_identity(root)
    classification_identity = contracts.sha256_file(
        root / "primary_classification.json"
    )
    receipt = {
        "schema_version": "skhynix_stage_h0b_outcome_build_v1",
        "task_id": contracts.TASK_ID,
        "build_label": permit["build_label"],
        "primary_results_sha256": primary_identity,
        "primary_classification_sha256": classification_identity,
        "classification": classification,
        "diagnostics": diagnostics,
        "stage4_crosscheck_opened": False,
        "aug07_event_rows_opened": False,
        "network_private_order_cancel_live_access": False,
    }
    write_json(root / "outcome_build_receipt.json", receipt, fsync=True)
    return receipt


def write_primary_seal(
    *,
    build_a: Path,
    build_b: Path,
) -> dict[str, Any]:
    identity_a = primary_results_identity(build_a)
    identity_b = primary_results_identity(build_b)
    classification_a = contracts.sha256_file(
        build_a / "primary_classification.json"
    )
    classification_b = contracts.sha256_file(
        build_b / "primary_classification.json"
    )
    contracts.require(
        identity_a == identity_b
        and classification_a == classification_b,
        "H0B_BUILD_MISMATCH",
        "$.primary_results",
        f"A={identity_a}/{classification_a} "
        f"B={identity_b}/{classification_b}",
    )
    seal = {
        "schema_version": "skhynix_stage_h0b_primary_result_seal_v2",
        "task_id": contracts.TASK_ID,
        "primary_plan_sha256": PRIMARY_PLAN_SHA256,
        "diagnostic_plan_sha256": DIAGNOSTIC_PLAN_SHA256,
        "diagnostic_review_sha256": DIAGNOSTIC_REVIEW_SHA256,
        "surface_matrix_sha256": MATRIX_SHA256,
        "semantic_source_inventory_sha256": (
            EXPECTED_SEMANTIC_INVENTORY_SHA256
        ),
        "build_a_primary_results_sha256": identity_a,
        "build_b_primary_results_sha256": identity_b,
        "primary_results_sha256": identity_a,
        "primary_classification_sha256": classification_a,
        "stage4_crosscheck_opened": False,
        "sealed_fsynced": True,
    }
    raw = contracts.canonical_json_bytes(seal)
    for root in (build_a, build_b):
        path = root / "primary_result_seal.json"
        path.write_bytes(raw)
        contracts.fsync_file(path)
        contracts.fsync_directory(root)
    return seal


def stage4_projection_contract_sha256() -> str:
    return contracts.canonical_json_sha256(
        {
            "paths": [
                {"relative_path": path, "sha256": sha256}
                for path, sha256 in sorted(STAGE4_OUTCOMES.items())
            ],
            "full_header_sha256": STAGE4_HEADER_SHA256,
            "projected_fields": list(STAGE4_PROJECTED_FIELDS),
            "grid_ns": contracts.GRID_NS,
            "horizon_ns": contracts.HORIZON_NS,
            "censor_precedence": [
                "grid_boundary",
                "accepted_support_class",
                "identified_event_or_no_event",
            ],
            "support_classes": list(contracts.IDENTIFICATION_CLASSES),
            "eligible_rule": "both_h0b_and_stage4_50ms_endpoints_identified",
        }
    )


def write_stage4_diagnostic_permit(build_root: Path) -> dict[str, Any]:
    root = Path(build_root)
    outcome_permit = validate_permit(root)
    seal_path = root / "primary_result_seal.json"
    seal = read_json(seal_path)
    expected_seal_keys = {
        "schema_version",
        "task_id",
        "primary_plan_sha256",
        "diagnostic_plan_sha256",
        "diagnostic_review_sha256",
        "surface_matrix_sha256",
        "semantic_source_inventory_sha256",
        "build_a_primary_results_sha256",
        "build_b_primary_results_sha256",
        "primary_results_sha256",
        "primary_classification_sha256",
        "stage4_crosscheck_opened",
        "sealed_fsynced",
    }
    contracts.require(
        set(seal) == expected_seal_keys
        and seal["task_id"] == contracts.TASK_ID
        and seal["primary_plan_sha256"] == PRIMARY_PLAN_SHA256
        and seal["diagnostic_plan_sha256"] == DIAGNOSTIC_PLAN_SHA256
        and seal["diagnostic_review_sha256"] == DIAGNOSTIC_REVIEW_SHA256
        and seal["surface_matrix_sha256"] == MATRIX_SHA256
        and seal["semantic_source_inventory_sha256"]
        == EXPECTED_SEMANTIC_INVENTORY_SHA256
        and seal["sealed_fsynced"] is True
        and seal["stage4_crosscheck_opened"] is False
        and primary_results_identity(root)
        == seal["primary_results_sha256"]
        and contracts.sha256_file(root / "primary_classification.json")
        == seal["primary_classification_sha256"],
        "H0B_STAGE4_OPEN_BEFORE_PRIMARY_SEAL",
        str(seal_path),
        "primary seal is absent, stale or invalid",
    )
    permit = {
        "schema_version": "skhynix_stage_h0b_stage4_diagnostic_permit_v1",
        "task_id": contracts.TASK_ID,
        "build_label": outcome_permit["build_label"],
        "status": "admitted",
        "fsynced": True,
        "primary_plan_sha256": PRIMARY_PLAN_SHA256,
        "diagnostic_plan_sha256": DIAGNOSTIC_PLAN_SHA256,
        "diagnostic_review_sha256": DIAGNOSTIC_REVIEW_SHA256,
        "surface_matrix_sha256": MATRIX_SHA256,
        "runtime_source_tree_sha256": runtime_source_tree_sha256(),
        "primary_seal_sha256": contracts.sha256_file(seal_path),
        "primary_results_sha256": seal["primary_results_sha256"],
        "primary_classification_sha256": seal[
            "primary_classification_sha256"
        ],
        "stage4_projection_contract_sha256": (
            stage4_projection_contract_sha256()
        ),
    }
    path = root / "stage4_diagnostic_permit.json"
    contracts.require(
        not path.exists(),
        "H0B_STAGE4_OPEN_BEFORE_PRIMARY_SEAL",
        str(path),
        "diagnostic permit already exists",
    )
    write_json(path, permit, fsync=True)
    contracts.fsync_directory(root)
    permit_sha = contracts.sha256_file(path)
    ledger_path = root / "outcome_access_ledger.json"
    ledger = read_json(ledger_path)
    ledger["events"].append(
        {
            "sequence": len(ledger["events"]) + 1,
            "process_role": "H0B1_DIAGNOSTIC_PERMIT",
            "phase": "post_primary_seal_permit",
            "relative_path": "stage4_diagnostic_permit.json",
            "access_kind": "fsync_write",
            "bytes_read": 0,
            "permit_sha256": permit_sha,
            "admitted": True,
        }
    )
    write_json(ledger_path, ledger, fsync=True)
    return {
        "verified": True,
        "task_id": contracts.TASK_ID,
        "build_label": permit["build_label"],
        "diagnostic_permit_sha256": permit_sha,
        "primary_seal_sha256": permit["primary_seal_sha256"],
        "stage4_bytes_opened": False,
    }


def validate_stage4_diagnostic_permit(build_root: Path) -> dict[str, Any]:
    root = Path(build_root)
    outcome_permit = validate_permit(root)
    path = root / "stage4_diagnostic_permit.json"
    permit = read_json(path)
    expected_keys = {
        "schema_version",
        "task_id",
        "build_label",
        "status",
        "fsynced",
        "primary_plan_sha256",
        "diagnostic_plan_sha256",
        "diagnostic_review_sha256",
        "surface_matrix_sha256",
        "runtime_source_tree_sha256",
        "primary_seal_sha256",
        "primary_results_sha256",
        "primary_classification_sha256",
        "stage4_projection_contract_sha256",
    }
    seal_path = root / "primary_result_seal.json"
    seal = read_json(seal_path)
    contracts.require(
        set(permit) == expected_keys
        and permit["schema_version"]
        == "skhynix_stage_h0b_stage4_diagnostic_permit_v1"
        and permit["task_id"] == contracts.TASK_ID
        and permit["build_label"] == outcome_permit["build_label"]
        and permit["status"] == "admitted"
        and permit["fsynced"] is True
        and permit["primary_plan_sha256"] == PRIMARY_PLAN_SHA256
        and permit["diagnostic_plan_sha256"] == DIAGNOSTIC_PLAN_SHA256
        and permit["diagnostic_review_sha256"]
        == DIAGNOSTIC_REVIEW_SHA256
        and permit["surface_matrix_sha256"] == MATRIX_SHA256
        and permit["runtime_source_tree_sha256"]
        == runtime_source_tree_sha256()
        and permit["primary_seal_sha256"]
        == contracts.sha256_file(seal_path)
        and permit["primary_results_sha256"]
        == seal["primary_results_sha256"]
        and permit["primary_classification_sha256"]
        == seal["primary_classification_sha256"]
        and permit["stage4_projection_contract_sha256"]
        == stage4_projection_contract_sha256(),
        "H0B_STAGE4_OPEN_BEFORE_PRIMARY_SEAL",
        str(path),
        "diagnostic permit mismatch",
    )
    return permit


def stage2_jul30_membership() -> dict[str, dict[str, Any]]:
    path = STAGE2_ROOT / "candidate_episode_membership.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        result = {}
        for row in reader:
            if row["session_id"] != "jul30":
                continue
            candidate = row["candidate_id"]
            contracts.require(
                candidate not in result,
                "H0B_STAGE4_PROJECTION_MISMATCH",
                candidate,
                "duplicate Stage 2 candidate",
            )
            result[candidate] = {
                "candidate_id": candidate,
                "segment_id": row["segment_id"],
                "connection_epoch_id": row["connection_epoch_id"],
                "shock_ts_ns": int(row["shock_ts_ns"]),
                "direction_sign": int(row["direction_sign"]),
                "primary_episode": row["primary_episode"] == "true",
            }
    return result


def read_stage4_projection(path: Path, expected_sha256: str) -> list[dict[str, str]]:
    validate_source_access(
        path.relative_to(REPO_ROOT).as_posix(),
        source_role="accepted_stage4_outcome",
        phase="post_primary_seal_stage4",
    )
    contracts.require(
        contracts.sha256_file(path) == expected_sha256,
        "H0B_STAGE4_PROJECTION_MISMATCH",
        str(path),
        "accepted Stage 4 raw SHA mismatch",
    )
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        contracts.require(
            header is not None
            and len(header) == 62
            and hashlib.sha256(
                (",".join(header) + "\n").encode("utf-8")
            ).hexdigest()
            == STAGE4_HEADER_SHA256,
            "H0B_SOURCE_SCHEMA_MISMATCH",
            str(path),
            "Stage 4 full header mismatch",
        )
        indexes = []
        for field in STAGE4_PROJECTED_FIELDS:
            contracts.require(
                field in header,
                "H0B_STAGE4_PROJECTION_MISMATCH",
                str(path),
                f"missing projected field {field}",
            )
            indexes.append(header.index(field))
        result = []
        for raw in reader:
            contracts.require(
                len(raw) == 62,
                "H0B_SOURCE_SCHEMA_MISMATCH",
                str(path),
                f"row width={len(raw)}",
            )
            result.append(
                {
                    field: raw[index]
                    for field, index in zip(
                        STAGE4_PROJECTED_FIELDS,
                        indexes,
                    )
                }
            )
    return result


LANDMARK_CENSOR_STATUS_BY_SUPPORT_CLASS = {
    "interval_likelihood_only_supported": (
        "diagnostic_censored_interval_likelihood_only"
    ),
    "right_censored_segment": (
        "diagnostic_censored_right_censored_segment"
    ),
    "right_censored_source_end": (
        "diagnostic_censored_right_censored_source_end"
    ),
    "epoch_censored": "diagnostic_censored_epoch_censored",
    "core_quality_censored": (
        "diagnostic_censored_core_quality_censored"
    ),
    "source_gap_censored": (
        "diagnostic_censored_source_gap_censored"
    ),
    "reference_quote_unavailable": (
        "diagnostic_censored_reference_quote_unavailable"
    ),
    "invalid_quote_state": "diagnostic_censored_invalid_quote_state",
}


def landmark_status_from_support_class(
    support_class: str,
    *,
    event: bool | None = None,
) -> str:
    contracts.require(
        support_class in contracts.IDENTIFICATION_CLASSES,
        "H0B_STAGE4_PROJECTION_MISMATCH",
        "$.landmark_support_class",
        support_class,
    )
    if support_class == "binary_identification_supported":
        contracts.require(
            event is not None,
            "H0B_STAGE4_PROJECTION_MISMATCH",
            "$.landmark_event",
            "binary-identified landmark requires event/no-event",
        )
        return "identified_event" if event else "identified_no_event"
    contracts.require(
        event is None,
        "H0B_STAGE4_PROJECTION_MISMATCH",
        "$.landmark_event",
        f"censored support class {support_class} cannot carry an event",
    )
    return LANDMARK_CENSOR_STATUS_BY_SUPPORT_CLASS[support_class]


def accepted_h0a_support_class(
    *,
    reference_available: bool,
    reference_valid: bool,
    target_inside_segment: bool,
    same_epoch: bool,
    core_quality_eligible: bool,
    source_gap: bool,
    source_ended: bool,
    endpoint_closed: bool,
    interval_bounds_supported: bool,
) -> str:
    facts = h0a_support.classify_support(
        reference_available=reference_available,
        reference_valid=reference_valid,
        target_inside_segment=target_inside_segment,
        same_epoch=same_epoch,
        core_quality_eligible=core_quality_eligible,
        source_gap=source_gap,
        source_ended=source_ended,
        endpoint_closed=endpoint_closed,
        interval_bounds_supported=interval_bounds_supported,
    )
    contracts.require(
        facts.identification_class in contracts.IDENTIFICATION_CLASSES,
        "H0B_STAGE4_PROJECTION_MISMATCH",
        "$.accepted_h0a_support_class",
        facts.identification_class,
    )
    return facts.identification_class


@dataclass(frozen=True)
class AcceptedH0ASupportState:
    reference_available: bool
    reference_valid: bool
    target_inside_segment: bool
    same_epoch: bool
    core_quality_eligible: bool
    source_gap: bool
    source_ended: bool
    endpoint_closed: bool
    interval_bounds_supported: bool


def accepted_h0a_support_class_from_state(
    state: AcceptedH0ASupportState,
) -> str:
    return accepted_h0a_support_class(**state.__dict__)


def accepted_h0a_support_state(
    *,
    reference_available: bool,
    reference_valid: bool,
    target_inside_segment: bool,
) -> AcceptedH0ASupportState:
    return AcceptedH0ASupportState(
        reference_available=reference_available,
        reference_valid=reference_valid,
        target_inside_segment=target_inside_segment,
        same_epoch=target_inside_segment,
        core_quality_eligible=True,
        source_gap=False,
        source_ended=False,
        endpoint_closed=target_inside_segment,
        interval_bounds_supported=True,
    )


def accepted_h0a_support_by_grid(
    segment: Any,
    *,
    accepted_commitment: Mapping[str, str],
    requested_grid_ns: set[int],
) -> tuple[dict[int, str], Counter[str]]:
    first, last, nominal_count = h0a_support.segment_grid_bounds(
        segment.start_ns,
        segment.end_ns,
    )
    timestamps = np.asarray(
        [row.local_ts_ns for row in segment.bbo],
        dtype=np.int64,
    )
    validity = np.asarray(
        [row.valid for row in segment.bbo],
        dtype=bool,
    )
    grid = np.arange(
        first,
        last + contracts.GRID_NS,
        contracts.GRID_NS,
        dtype=np.int64,
    )
    contracts.require(
        grid.size == nominal_count,
        "H0B_STAGE4_PROJECTION_MISMATCH",
        segment.segment_id,
        "accepted H0-A nominal grid count mismatch",
    )
    reference = np.searchsorted(timestamps, grid, side="right") - 1
    result: dict[int, str] = {}
    counts: Counter[str] = Counter(
        {name: 0 for name in contracts.IDENTIFICATION_CLASSES}
    )
    projection_hasher = hashlib.sha256()
    for grid_ts_ns, reference_index in zip(
        grid.tolist(),
        reference.tolist(),
    ):
        available = reference_index >= 0
        valid = available and bool(validity[reference_index])
        state = accepted_h0a_support_state(
            reference_available=available,
            reference_valid=valid,
            target_inside_segment=(
                grid_ts_ns + contracts.HORIZON_NS < segment.end_ns
            ),
        )
        support_class = accepted_h0a_support_class_from_state(state)
        if grid_ts_ns in requested_grid_ns:
            result[grid_ts_ns] = support_class
        counts[support_class] += 1
        block_start = (grid_ts_ns // contracts.BLOCK_NS) * (
            contracts.BLOCK_NS
        )
        last_start = (
            block_start + contracts.BLOCK_NS - contracts.GRID_NS
        )
        block_id = (
            f"{segment.session_id}:{block_start}"
            if block_start >= segment.start_ns
            and block_start + contracts.BLOCK_NS <= segment.end_ns
            and last_start + contracts.HORIZON_NS < segment.end_ns
            else ""
        )
        projection_hasher.update(
            h0a_support.canonical_json_bytes(
                {
                    "binary_endpoint_identification_supported": (
                        support_class
                        == "binary_identification_supported"
                    ),
                    "complete_60s_block_id_or_empty": block_id,
                    "connection_epoch_id": segment.connection_epoch_id,
                    "grid_ts_ns": grid_ts_ns,
                    "horizon_ms": 50,
                    "identification_class": support_class,
                    "interval_likelihood_eligible": support_class
                    in {
                        "binary_identification_supported",
                        "interval_likelihood_only_supported",
                    },
                    "observation_bound_contract_id": (
                        h0a_support.OBSERVATION_BOUND_CONTRACT_ID
                    ),
                    "quality_eligible": support_class
                    not in {
                        "core_quality_censored",
                        "source_gap_censored",
                    },
                    "schema_version": "h0a_support_projection_tuple_v1",
                    "segment_id": segment.segment_id,
                    "session_id": segment.session_id,
                }
            )
            + b"\n"
        )
    contracts.require(
        accepted_commitment["connection_epoch_id"]
        == segment.connection_epoch_id
        and int(accepted_commitment["support_projection_row_count"])
        == nominal_count
        and int(accepted_commitment["first_grid_ts_ns"]) == first
        and int(accepted_commitment["last_grid_ts_ns"]) == last
        and accepted_commitment["support_projection_sha256"]
        == projection_hasher.hexdigest(),
        "H0B_SUPPORT_COMMITMENT_MISMATCH",
        segment.segment_id,
        "accepted H0-A 50ms per-grid support projection mismatch",
    )
    contracts.require(
        set(result) == requested_grid_ns,
        "H0B_SUPPORT_CLASS_DISPOSITION_MISMATCH",
        segment.segment_id,
        "requested accepted support landmarks were not resolved",
    )
    return result, counts


def accepted_h0a_50ms_commitments() -> dict[str, dict[str, str]]:
    path = H0A_ROOT / "support_projection_commitments.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        contracts.require(
            tuple(reader.fieldnames or ()) == h0a_support.COMMITMENT_FIELDS,
            "H0B_SUPPORT_COMMITMENT_MISMATCH",
            str(path),
            "accepted H0-A commitment header mismatch",
        )
        rows = {
            row["segment_id"]: row
            for row in reader
            if row["session_id"] == "jul30"
            and row["horizon_ms"] == "50"
        }
    contracts.require(
        set(rows)
        == {f"segment_{index:04d}" for index in range(1, 9)},
        "H0B_SUPPORT_COMMITMENT_MISMATCH",
        str(path),
        "Jul30 50ms segment commitment set mismatch",
    )
    return rows


def accepted_h0a_50ms_class_counts() -> Counter[str]:
    path = H0A_ROOT / "censoring_identification_by_horizon.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        contracts.require(
            tuple(reader.fieldnames or ()) == h0a_support.CENSOR_FIELDS,
            "H0B_SUPPORT_CLASS_DISPOSITION_MISMATCH",
            str(path),
            "accepted H0-A censor census header mismatch",
        )
        rows = [
            row
            for row in reader
            if row["session_id"] == "jul30"
            and row["horizon_ms"] == "50"
        ]
    counts = Counter(
        {
            row["identification_class"]: int(row["grid_count"])
            for row in rows
        }
    )
    contracts.require(
        len(rows) == len(contracts.IDENTIFICATION_CLASSES)
        and set(counts) == set(contracts.IDENTIFICATION_CLASSES),
        "H0B_SUPPORT_CLASS_DISPOSITION_MISMATCH",
        str(path),
        "accepted Jul30 50ms support class universe mismatch",
    )
    return counts


def landmark_status_with_precedence(
    *,
    outside_grid: bool,
    support_class: str,
    event: bool | None = None,
) -> str:
    if outside_grid:
        return "diagnostic_censored_grid_boundary"
    return landmark_status_from_support_class(
        support_class,
        event=event,
    )


def h0b_landmark_statuses(
    landmarks: Mapping[str, list[tuple[str, int, str]]],
) -> dict[str, str]:
    sessions = h0a_support.load_sessions(CANONICAL_SOURCE_ROOT)
    spec = next(
        item
        for item in h0a_support.SESSION_SPECS
        if item.session_id == "jul30"
    )
    segment_map = {
        segment.segment_id: segment
        for segment in sessions["jul30"]
    }
    result: dict[str, str] = {}
    accepted_support_maps: dict[str, dict[int, str]] = {}
    accepted_support_counts: Counter[str] = Counter()
    accepted_commitments = accepted_h0a_50ms_commitments()
    for segment_id, segment in segment_map.items():
        first, last, _ = h0a_support.segment_grid_bounds(
            segment.start_ns,
            segment.end_ns,
        )
        requested_grid_ns = {
            int(value[1])
            for value in landmarks.get(segment_id, [])
            if first <= int(value[1]) <= last
        }
        support_map, support_counts = accepted_h0a_support_by_grid(
            segment,
            accepted_commitment=accepted_commitments[segment_id],
            requested_grid_ns=requested_grid_ns,
        )
        accepted_support_maps[segment_id] = support_map
        accepted_support_counts.update(support_counts)
    contracts.require(
        sum(accepted_support_counts.values())
        == sum(
            int(row["support_projection_row_count"])
            for row in accepted_commitments.values()
        )
        and accepted_support_counts == accepted_h0a_50ms_class_counts()
        and set(accepted_support_counts)
        == set(contracts.IDENTIFICATION_CLASSES),
        "H0B_SUPPORT_CLASS_DISPOSITION_MISMATCH",
        "$.jul30.accepted_support",
        "accepted support universe or count mismatch",
    )
    for segment_id, values in landmarks.items():
        segment = segment_map[segment_id]
        path = (
            CANONICAL_SOURCE_ROOT
            / spec.r0_relative_path
            / "segments"
            / segment_id
            / "hyperliquid_hot_events.csv.gz"
        )
        events = load_quote_events(
            path,
            header=HYPERLIQUID_HEADER,
            event_type="bbo",
            segment_id=segment_id,
        )
        grid = np.asarray([value[1] for value in values], dtype=np.int64)
        first, last, _ = h0a_support.segment_grid_bounds(
            segment.start_ns,
            segment.end_ns,
        )
        reference = (
            np.searchsorted(events.local_ts_ns, grid, side="right") - 1
        )
        outside_grid = np.logical_or(grid < first, grid > last)
        support_classes = np.asarray(
            [
                (
                    "binary_identification_supported"
                    if outside_grid[index]
                    else accepted_support_maps[segment_id][int(timestamp)]
                )
                for index, timestamp in enumerate(grid.tolist())
            ],
            dtype=object,
        )
        for position in np.flatnonzero(outside_grid).tolist():
            result[values[position][0]] = landmark_status_with_precedence(
                outside_grid=True,
                support_class=str(support_classes[position]),
            )
        for side in contracts.SIDES:
            side_positions = [
                index
                for index, value in enumerate(values)
                if value[2] == side
                and not outside_grid[index]
                and support_classes[index]
                == "binary_identification_supported"
            ]
            if side_positions:
                selected = np.asarray(side_positions, dtype=np.int64)
                _, _, _, event = event_geometry(
                    grid=grid[selected],
                    reference_indexes=reference[selected],
                    events=events,
                    side=side,
                )
                for position, event_value in zip(
                    selected.tolist(),
                    event.tolist(),
                ):
                    result[values[position][0]] = (
                        landmark_status_with_precedence(
                            outside_grid=False,
                            support_class=(
                                "binary_identification_supported"
                            ),
                            event=bool(event_value),
                        )
                    )
        for position, support_class in enumerate(support_classes.tolist()):
            candidate = values[position][0]
            if candidate in result:
                continue
            result[candidate] = landmark_status_with_precedence(
                outside_grid=False,
                support_class=str(support_class),
            )
    contracts.require(
        len(result) == sum(len(values) for values in landmarks.values()),
        "H0B_STAGE4_PROJECTION_MISMATCH",
        "$.h0b_landmark_statuses",
        "candidate status conservation failure",
    )
    return result


LANDMARK_CENSOR_COUNT_FIELD = {
    "diagnostic_censored_grid_boundary": (
        "h0b_grid_boundary_censored_count"
    ),
    "diagnostic_censored_interval_likelihood_only": (
        "h0b_interval_likelihood_only_censored_count"
    ),
    "diagnostic_censored_right_censored_segment": (
        "h0b_right_censored_segment_count"
    ),
    "diagnostic_censored_right_censored_source_end": (
        "h0b_right_censored_source_end_count"
    ),
    "diagnostic_censored_epoch_censored": "h0b_epoch_censored_count",
    "diagnostic_censored_core_quality_censored": (
        "h0b_core_quality_censored_count"
    ),
    "diagnostic_censored_source_gap_censored": (
        "h0b_source_gap_censored_count"
    ),
    "diagnostic_censored_reference_quote_unavailable": (
        "h0b_reference_quote_unavailable_count"
    ),
    "diagnostic_censored_invalid_quote_state": (
        "h0b_invalid_quote_state_count"
    ),
}


def stage4_endpoint_status(
    row: Mapping[str, str],
    *,
    endpoint_ns: int,
) -> str:
    status = row["time_to_first_adverse_target_bbo_event_status"]
    upper_text = row[
        "time_to_first_adverse_target_bbo_event_interval_upper_ns"
    ]
    censor_text = row[
        "time_to_first_adverse_target_bbo_event_censor_time_ns"
    ]
    if status == "interval_censored" and upper_text:
        if int(upper_text) <= endpoint_ns:
            return "identified_event"
        return "diagnostic_censored"
    if (
        status == "right_censored"
        and censor_text
        and int(censor_text) >= endpoint_ns
    ):
        return "identified_no_event"
    return "diagnostic_censored"


def update_stage4_crosscheck_counter(
    counter: Counter[str],
    *,
    h0b_status: str,
    stage4_status: str,
) -> None:
    allowed_h0b = {
        "identified_event",
        "identified_no_event",
        *LANDMARK_CENSOR_COUNT_FIELD,
    }
    allowed_stage4 = {
        "identified_event",
        "identified_no_event",
        "diagnostic_censored",
    }
    contracts.require(
        h0b_status in allowed_h0b and stage4_status in allowed_stage4,
        "H0B_STAGE4_PROJECTION_MISMATCH",
        "$.crosscheck.status",
        f"h0b={h0b_status} stage4={stage4_status}",
    )
    counter["joined_count"] += 1
    h0b_identified = h0b_status.startswith("identified_")
    stage4_identified = stage4_status.startswith("identified_")
    if h0b_identified:
        counter["h0b_identified_count"] += 1
    else:
        counter["h0b_censored_count"] += 1
        counter[LANDMARK_CENSOR_COUNT_FIELD[h0b_status]] += 1
    if stage4_identified:
        counter["stage4_identified_count"] += 1
    else:
        counter["stage4_censored_count"] += 1
    if not h0b_identified and not stage4_identified:
        counter["both_censored_count"] += 1
        counter["censored_count"] += 1
        return
    if not h0b_identified:
        counter["h0b_only_censored_count"] += 1
        counter["censored_count"] += 1
        return
    if not stage4_identified:
        counter["stage4_only_censored_count"] += 1
        counter["censored_count"] += 1
        return
    counter["eligible_count"] += 1
    h0b_event = h0b_status == "identified_event"
    stage4_event = stage4_status == "identified_event"
    counter["h0b_event_count"] += int(h0b_event)
    counter["stage4_event_count"] += int(stage4_event)
    if h0b_event and stage4_event:
        counter["both_event_count"] += 1
    elif h0b_event:
        counter["h0b_only_count"] += 1
    elif stage4_event:
        counter["stage4_only_count"] += 1
    else:
        counter["neither_count"] += 1


def stage4_crosscheck_row(
    *,
    scope: str,
    segment_id: str,
    side: str,
    counter: Counter[str],
    direction_match: bool,
    naming_match: bool,
) -> dict[str, Any]:
    h0b_censor_sum = sum(
        counter[field]
        for field in LANDMARK_CENSOR_COUNT_FIELD.values()
    )
    contracts.require(
        counter["joined_count"]
        == counter["h0b_identified_count"] + counter["h0b_censored_count"]
        and counter["h0b_censored_count"] == h0b_censor_sum
        and counter["joined_count"]
        == counter["stage4_identified_count"]
        + counter["stage4_censored_count"]
        and counter["h0b_censored_count"]
        == counter["h0b_only_censored_count"]
        + counter["both_censored_count"]
        and counter["stage4_censored_count"]
        == counter["stage4_only_censored_count"]
        + counter["both_censored_count"]
        and counter["censored_count"]
        == counter["h0b_only_censored_count"]
        + counter["stage4_only_censored_count"]
        + counter["both_censored_count"]
        and counter["joined_count"]
        == counter["eligible_count"] + counter["censored_count"]
        and counter["eligible_count"]
        == counter["both_event_count"]
        + counter["h0b_only_count"]
        + counter["stage4_only_count"]
        + counter["neither_count"]
        and counter["h0b_event_count"]
        == counter["both_event_count"] + counter["h0b_only_count"]
        and counter["stage4_event_count"]
        == counter["both_event_count"] + counter["stage4_only_count"]
        and all(value >= 0 for value in counter.values()),
        "H0B_STAGE4_PROJECTION_MISMATCH",
        f"$.crosscheck.{scope}.{segment_id}.{side}",
        "count conservation failure",
    )
    eligible = counter["eligible_count"]
    return {
        "scope": scope,
        "session": "jul30",
        "segment_id": segment_id,
        "side": side,
        **{
            field: counter[field]
            for field in contracts.CSV_HEADERS[
                "diagnostics/stage4_landmark_crosscheck.csv"
            ]
            if field.endswith("_count")
        },
        "h0b_event_rate": (
            counter["h0b_event_count"] / eligible if eligible else None
        ),
        "stage4_event_rate": (
            counter["stage4_event_count"] / eligible if eligible else None
        ),
        "agreement_fraction": (
            (
                counter["both_event_count"]
                + counter["neither_count"]
            )
            / eligible
            if eligible
            else None
        ),
        "direction_mapping_match": direction_match,
        "quote_risk_naming_match": naming_match,
        "primary_seal_unchanged": True,
    }


def validate_diagnostic_permit_pair(
    permit_a: Mapping[str, Any],
    permit_b: Mapping[str, Any],
    *,
    primary_seal_sha256: str,
) -> None:
    contracts.require(
        permit_a["build_label"] == "A"
        and permit_b["build_label"] == "B"
        and permit_a["primary_seal_sha256"]
        == permit_b["primary_seal_sha256"]
        == primary_seal_sha256
        and permit_a["diagnostic_permit_sha256"]
        != permit_b["diagnostic_permit_sha256"]
        and permit_a["stage4_bytes_opened"] is False
        and permit_b["stage4_bytes_opened"] is False,
        "H0B_STAGE4_OPEN_BEFORE_PRIMARY_SEAL",
        "$.diagnostic_permits",
        "A/B diagnostic permit contract mismatch",
    )


def validate_stage4_access_ledger(
    ledger: Mapping[str, Any],
    *,
    build_label: str,
    diagnostic_permit_sha256: str,
) -> None:
    events = ledger["events"]
    event_keys = {
        "sequence",
        "process_role",
        "phase",
        "relative_path",
        "access_kind",
        "bytes_read",
        "permit_sha256",
        "admitted",
    }
    contracts.require(
        isinstance(events, list)
        and all(
            isinstance(event, Mapping) and set(event) == event_keys
            for event in events
        ),
        "H0B_STAGE4_OPEN_BEFORE_PRIMARY_SEAL",
        f"$.ledger.{build_label}.events",
        "ledger event schema mismatch",
    )
    expected_stage4_paths = {
        (
            STAGE4_ROOT.relative_to(REPO_ROOT)
            / relative
        ).as_posix()
        for relative in STAGE4_OUTCOMES
    }
    permit_events = [
        event
        for event in events
        if event["phase"] == "post_primary_seal_permit"
    ]
    stage4_events = [
        event
        for event in events
        if event["phase"] == "post_primary_seal_stage4"
    ]
    diagnostic_like_events = [
        event
        for event in events
        if event["process_role"]
        in {"H0B1_DIAGNOSTIC_PERMIT", "H0B1_DIAGNOSTIC"}
        or event["relative_path"] == "stage4_diagnostic_permit.json"
        or event["relative_path"] in expected_stage4_paths
        or event["permit_sha256"] == diagnostic_permit_sha256
    ]
    contracts.require(
        ledger["schema_version"]
        == "skhynix_stage_h0b_outcome_access_ledger_v1"
        and ledger["task_id"] == contracts.TASK_ID
        and ledger["build_label"] == build_label
        and [event["sequence"] for event in events]
        == list(range(1, len(events) + 1))
        and len(permit_events) == 1
        and permit_events[0]["process_role"]
        == "H0B1_DIAGNOSTIC_PERMIT"
        and permit_events[0]["relative_path"]
        == "stage4_diagnostic_permit.json"
        and permit_events[0]["access_kind"] == "fsync_write"
        and permit_events[0]["bytes_read"] == 0
        and permit_events[0]["permit_sha256"]
        == diagnostic_permit_sha256
        and permit_events[0]["admitted"] is True
        and len(stage4_events) == len(STAGE4_OUTCOMES)
        and diagnostic_like_events == [
            permit_events[0],
            *stage4_events,
        ]
        and events[permit_events[0]["sequence"] - 1 :]
        == [permit_events[0], *stage4_events]
        and {
            event["relative_path"]
            for event in stage4_events
        }
        == expected_stage4_paths
        and all(
            event["sequence"] > permit_events[0]["sequence"]
            and event["process_role"] == "H0B1_DIAGNOSTIC"
            and event["access_kind"] == "exact_11_field_projection"
            and event["bytes_read"] > 0
            and event["permit_sha256"] == diagnostic_permit_sha256
            and event["admitted"] is True
            for event in stage4_events
        ),
        "H0B_STAGE4_OPEN_BEFORE_PRIMARY_SEAL",
        f"$.ledger.{build_label}",
        "diagnostic permit or Stage 4 access ordering mismatch",
    )


def run_stage4_diagnostic(
    *,
    build_root: Path,
) -> dict[str, Any]:
    root = Path(build_root)
    diagnostic_permit = validate_stage4_diagnostic_permit(root)
    seal_path = root / "primary_result_seal.json"
    seal_raw = seal_path.read_bytes()
    seal = read_json(seal_path)
    contracts.require(
        seal["sealed_fsynced"] is True
        and seal["stage4_crosscheck_opened"] is False
        and seal["primary_plan_sha256"] == PRIMARY_PLAN_SHA256
        and seal["diagnostic_plan_sha256"] == DIAGNOSTIC_PLAN_SHA256
        and seal["diagnostic_review_sha256"] == DIAGNOSTIC_REVIEW_SHA256
        and primary_results_identity(root)
        == seal["primary_results_sha256"]
        and contracts.sha256_file(root / "primary_classification.json")
        == seal["primary_classification_sha256"],
        "H0B_STAGE4_OPEN_BEFORE_PRIMARY_SEAL",
        str(seal_path),
        "primary seal is absent, stale or already opened",
    )
    membership = stage2_jul30_membership()
    projected_rows: list[dict[str, str]] = []
    for relative, expected_sha in STAGE4_OUTCOMES.items():
        segment_id = Path(relative).name.removesuffix(".csv.gz")
        for row in read_stage4_projection(
            STAGE4_ROOT / relative,
            expected_sha,
        ):
            row["_stage4_segment_id"] = segment_id
            projected_rows.append(row)
    joined = []
    observed_candidates = set()
    landmarks: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for row in projected_rows:
        candidate = row["candidate_id"]
        contracts.require(
            candidate in membership and candidate not in observed_candidates,
            "H0B_STAGE4_PROJECTION_MISMATCH",
            candidate,
            "missing or duplicate Stage 2 candidate join",
        )
        observed_candidates.add(candidate)
        source = membership[candidate]
        candidate_ts = int(row["t_candidate_ns"])
        contracts.require(
            candidate_ts == source["shock_ts_ns"]
            and row["_stage4_segment_id"] == source["segment_id"]
            and str(source["connection_epoch_id"]) == "0",
            "H0B_STAGE4_PROJECTION_MISMATCH",
            candidate,
            "timestamp, path segment or accepted epoch drift",
        )
        direction = source["direction_sign"]
        contracts.require(
            direction in {-1, 1},
            "H0B_STAGE4_PROJECTION_MISMATCH",
            candidate,
            f"direction_sign={direction}",
        )
        side = "maker_ask_risk" if direction == 1 else "maker_bid_risk"
        grid = candidate_ts // contracts.GRID_NS * contracts.GRID_NS
        landmarks[source["segment_id"]].append((candidate, grid, side))
        joined.append((row, source, side))
    contracts.require(
        observed_candidates == set(membership),
        "H0B_STAGE4_PROJECTION_MISMATCH",
        "$.candidate_conservation",
        f"stage4={len(observed_candidates)} stage2={len(membership)}",
    )
    h0b_statuses = h0b_landmark_statuses(landmarks)
    aggregate: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    direction_match = True
    naming_match = True
    for row, source, side in joined:
        key = (source["segment_id"], side)
        endpoint = source["shock_ts_ns"] + contracts.HORIZON_NS
        h0b_status = h0b_statuses[row["candidate_id"]]
        stage4_status = stage4_endpoint_status(row, endpoint_ns=endpoint)
        available = row["public_quote_risk_availability"] == "available"
        if available:
            expected_boolean = (
                row["time_to_first_adverse_target_bbo_event_status"]
                == "interval_censored"
            )
            naming_match = naming_match and (
                (row["public_bbo_moves_through_quote"] == "true")
                == expected_boolean
            )
        direction_match = direction_match and (
            (source["direction_sign"] == 1 and side == "maker_ask_risk")
            or (
                source["direction_sign"] == -1
                and side == "maker_bid_risk"
            )
        )
        counter = aggregate[key]
        update_stage4_crosscheck_counter(
            counter,
            h0b_status=h0b_status,
            stage4_status=stage4_status,
        )
    rows = []
    for (segment_id, side), counter in sorted(
        aggregate.items(),
        key=lambda item: (
            item[0][0],
            contracts.SIDE_ORDER[item[0][1]],
        ),
    ):
        rows.append(
            stage4_crosscheck_row(
                scope="segment",
                segment_id=segment_id,
                side=side,
                counter=counter,
                direction_match=direction_match,
                naming_match=naming_match,
            )
        )
    count_fields = [
        field
        for field in contracts.CSV_HEADERS[
            "diagnostics/stage4_landmark_crosscheck.csv"
        ]
        if field.endswith("_count")
    ]
    for side in contracts.SIDES:
        totals: Counter[str] = Counter()
        for (_, observed_side), counter in aggregate.items():
            if observed_side != side:
                continue
            for field in count_fields:
                totals[field] += counter[field]
        rows.append(
            stage4_crosscheck_row(
                scope="session",
                segment_id="ALL",
                side=side,
                counter=totals,
                direction_match=direction_match,
                naming_match=naming_match,
            )
        )
    sort_output_rows("diagnostics/stage4_landmark_crosscheck.csv", rows)
    output_path = root / "diagnostics/stage4_landmark_crosscheck.csv"
    contracts.write_csv_exact(
        output_path,
        rows,
        contracts.CSV_HEADERS[
            "diagnostics/stage4_landmark_crosscheck.csv"
        ],
    )
    contracts.require(
        seal_path.read_bytes() == seal_raw
        and primary_results_identity(root)
        == seal["primary_results_sha256"]
        and contracts.sha256_file(root / "primary_classification.json")
        == seal["primary_classification_sha256"],
        "H0B_PRIMARY_SEAL_MISMATCH",
        str(seal_path),
        "Stage 4 diagnostic changed primary bytes or seal",
    )
    ledger_path = root / "outcome_access_ledger.json"
    ledger = read_json(ledger_path)
    diagnostic_permit_sha = contracts.sha256_file(
        root / "stage4_diagnostic_permit.json"
    )
    sequence = len(ledger["events"]) + 1
    for relative in sorted(STAGE4_OUTCOMES):
        ledger["events"].append(
            {
                "sequence": sequence,
                "process_role": "H0B1_DIAGNOSTIC",
                "phase": "post_primary_seal_stage4",
                "relative_path": (
                    STAGE4_ROOT.relative_to(REPO_ROOT)
                    / relative
                ).as_posix(),
                "access_kind": "exact_11_field_projection",
                "bytes_read": (STAGE4_ROOT / relative).stat().st_size,
                "permit_sha256": diagnostic_permit_sha,
                "admitted": True,
            }
        )
        sequence += 1
    write_json(ledger_path, ledger, fsync=True)
    session_rows = [row for row in rows if row["scope"] == "session"]
    receipt = {
        "schema_version": "skhynix_stage_h0b_stage4_diagnostic_v2",
        "task_id": contracts.TASK_ID,
        "build_label": diagnostic_permit["build_label"],
        "primary_plan_sha256": PRIMARY_PLAN_SHA256,
        "diagnostic_plan_sha256": DIAGNOSTIC_PLAN_SHA256,
        "diagnostic_review_sha256": DIAGNOSTIC_REVIEW_SHA256,
        "diagnostic_permit_sha256": diagnostic_permit_sha,
        "stage4_crosscheck_sha256": contracts.sha256_file(output_path),
        "primary_results_sha256": seal["primary_results_sha256"],
        "primary_classification_sha256": seal[
            "primary_classification_sha256"
        ],
        "primary_seal_sha256": contracts.sha256_file(seal_path),
        "stage4_path_count": len(STAGE4_OUTCOMES),
        "stage4_projected_field_count": len(STAGE4_PROJECTED_FIELDS),
        "joined_count": sum(row["joined_count"] for row in session_rows),
        "eligible_count": sum(
            row["eligible_count"] for row in session_rows
        ),
        "censored_count": sum(
            row["censored_count"] for row in session_rows
        ),
        "primary_seal_unchanged": True,
    }
    write_json(root / "stage4_diagnostic_receipt.json", receipt, fsync=True)
    return receipt


def surface_assignment_projection() -> list[dict[str, Any]]:
    matrix = read_json(MATRIX_PATH)
    rows = []
    prefix = (
        "local_live_analysis/"
        "skhynix_continuous_conditional_risk_v2_stage_h0b_0823T002/"
    )
    package_files: dict[str, set[str]] = {"R": set(), "C": set(), "E": set()}
    package_directories: set[str] = set()
    observed_package_paths: set[str] = set()
    for surface in matrix["surfaces"]:
        for artifact in surface["artifacts"]:
            rows.append(
                {
                    "surface_id": surface["surface_id"],
                    "path": artifact["path"],
                    "entry_type": artifact["entry_type"],
                    "required": artifact["required"],
                    "identity_layer": surface["identity_layer"],
                    "exact_contract_sha256": (
                        contracts.canonical_json_sha256(
                            surface["exact_contract"]
                        )
                    ),
                }
            )
            if not artifact["path"].startswith(prefix):
                continue
            normalized = artifact["path"][len(prefix) :]
            contracts.require(
                normalized
                and not normalized.startswith("/")
                and ".." not in Path(normalized).parts
                and normalized not in observed_package_paths
                and artifact["required"] is True,
                "H0B_IDENTITY_BINDING_MISMATCH",
                artifact["path"],
                "invalid package-owned assignment",
            )
            observed_package_paths.add(normalized)
            if artifact["entry_type"] == "directory":
                contracts.require(
                    surface["identity_layer"] == "E",
                    "H0B_IDENTITY_BINDING_MISMATCH",
                    artifact["path"],
                    "package directory must be layer E",
                )
                package_directories.add(normalized)
            else:
                package_files[surface["identity_layer"]].add(normalized)
    contracts.require(
        package_files["R"] == set(contracts.R_FILES)
        and package_files["C"] == set(contracts.C_FILES)
        and package_files["E"]
        == set((*contracts.E_FILES, contracts.MANIFEST_FILE))
        and package_directories == set(contracts.PACKAGE_DIRECTORIES),
        "H0B_IDENTITY_BINDING_MISMATCH",
        "$.surface_assignments",
        "package-owned R/C/E set oracle mismatch",
    )
    rows.sort(
        key=lambda row: (
            row["path"],
            row["surface_id"],
            row["entry_type"],
        )
    )
    return rows


def runtime_contract_bridge(root: Path) -> dict[str, Any]:
    preoutcome = read_json(root / "preoutcome_contract.json")
    return {
        "schema_version": "skhynix_stage_h0b_runtime_contract_bridge_v2",
        "task_id": contracts.TASK_ID,
        "kernel_source_tree_sha256": KERNEL_SOURCE_TREE_SHA256,
        "primary_plan_sha256": PRIMARY_PLAN_SHA256,
        "diagnostic_plan_sha256": DIAGNOSTIC_PLAN_SHA256,
        "diagnostic_review_sha256": DIAGNOSTIC_REVIEW_SHA256,
        "surface_matrix_sha256": MATRIX_SHA256,
        "files": contracts.file_inventory(root, contracts.C_FILES),
        "research_surface_assignments_sha256": (
            contracts.canonical_json_sha256(
                surface_assignment_projection()
            )
        ),
        "output_schema_contract_sha256": preoutcome[
            "output_contract_sha256"
        ],
        "contract_versions": {
            "observation_bounds": (
                "h0a_hyperliquid_bbo_receive_interval_v1"
            ),
            "interval_likelihood": (
                "h0b_piecewise_constant_hazard_v1"
            ),
            "design_matrix": "h0b_h0_h1_design_matrix_v1",
            "walk_forward": "h0b_expanding_60_20_v1",
            "resampling": "h0b_dependency_resampling_v1",
            "classification": "h0b_screening_classification_v1",
            "output_schema": "h0b_output_schema_v1",
            "identity_bridge": "h0b_layered_identity_bridge_v1",
        },
        "hard_boundary": {
            "aug07_event_rows_read": False,
            "stage4_before_primary_seal": False,
            "r1_decision_labels_read": False,
            "network_accessed": False,
            "private_endpoint_accessed": False,
            "order_or_cancel_accessed": False,
            "live_action_executed": False,
        },
    }


def publication_envelope_bridge(root: Path) -> dict[str, Any]:
    expected_files = sorted(
        contracts.EXACT_PACKAGE_FILES,
        key=lambda value: value.encode("utf-8"),
    )
    return {
        "schema_version": (
            "skhynix_stage_h0b_publication_envelope_bridge_v1"
        ),
        "task_id": contracts.TASK_ID,
        "files": contracts.file_inventory(root, contracts.E_FILES),
        "expected_files": expected_files,
        "expected_directories": list(contracts.PACKAGE_DIRECTORIES),
        "manifest_excluded_path": contracts.MANIFEST_FILE,
        "manifest_self_binding_normalization": (
            "manifest_excluded_from_E_inventory_and_package_count_bytes"
        ),
        "package_file_count_rule": (
            "exact_regular_file_count_excluding_h0b_manifest"
        ),
        "package_total_bytes_rule": (
            "exact_regular_file_bytes_excluding_h0b_manifest"
        ),
        "atomic_publication": {
            "no_overwrite": True,
            "fsync_tree_before_rename": True,
            "atomic_rename": True,
        },
        "verify_only_zero_write": True,
        "archive_scope": "package_only",
        "kernel_package_admission_portable": True,
        "full_source_semantic_replay_portable": False,
        "outcome_values_present": True,
    }


def package_identities(root: Path) -> dict[str, str]:
    research = trust.compute_research_data_identity(
        contracts.file_inventory(root, contracts.R_FILES)
    )
    runtime = trust.compute_runtime_contract_identity(
        research,
        runtime_contract_bridge(root),
    )
    publication = trust.compute_publication_envelope_identity(
        research,
        runtime,
        publication_envelope_bridge(root),
    )
    composite = trust.compute_composite_package_identity(
        research,
        runtime,
        publication,
    )
    return {
        "research_data_identity": research,
        "runtime_contract_identity": runtime,
        "publication_envelope_identity": publication,
        "composite_package_identity": composite,
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def report_value(value: str | None) -> str:
    return value if value not in {None, ""} else "NA"


def package_report_text(
    *,
    root: Path,
    research_identity: str,
    runtime_identity: str,
) -> str:
    classification = read_json(root / "primary_classification.json")
    rq1 = {
        row["session"]: row
        for row in read_csv_rows(root / "rq1_dispersion_tests.csv")
    }
    rq2 = {
        row["session"]: row
        for row in read_csv_rows(root / "rq2_session_scores.csv")
    }
    rq3 = {
        row["session"]: row
        for row in read_csv_rows(root / "rq3_latency_actionability.csv")
        if row["latency_ms"] == "6600"
    }
    primary_seal = read_json(root / "primary_result_seal.json")
    stage4_sha = contracts.sha256_file(
        root / "diagnostics/stage4_landmark_crosscheck.csv"
    )
    lines = [
        "# Stage H0-B Conditional-Risk Audit",
        "",
        "- task: `0823T002`",
        "- status: `待验收`",
        f"- classification: `{classification['classification']}`",
        f"- primary_plan_sha256: `{PRIMARY_PLAN_SHA256}`",
        f"- diagnostic_plan_sha256: `{DIAGNOSTIC_PLAN_SHA256}`",
        f"- diagnostic_review_sha256: `{DIAGNOSTIC_REVIEW_SHA256}`",
        "- formal_sessions: `jul30,aug04`",
        "- primary_tuple: `public_bbo_moves_through_quote/delta=0/horizon=50ms/latency=6600ms/equal_weight_bid_ask_session_scores`",
        f"- rq1_jul30_pass: `{report_value(rq1['jul30']['primary_pass'])}`",
        f"- rq1_aug04_pass: `{report_value(rq1['aug04']['primary_pass'])}`",
        "- rq2_jul30_ratio_time_upper_flow_upper_pass: "
        f"`{report_value(rq2['jul30']['normalized_interval_log_loss_h1_h0'])}/"
        f"{report_value(rq2['jul30']['time_ci_upper'])}/"
        f"{report_value(rq2['jul30']['flow_ci_upper'])}/"
        f"{report_value(rq2['jul30']['rq2_pass'])}`",
        "- rq2_aug04_ratio_time_upper_flow_upper_pass: "
        f"`{report_value(rq2['aug04']['normalized_interval_log_loss_h1_h0'])}/"
        f"{report_value(rq2['aug04']['time_ci_upper'])}/"
        f"{report_value(rq2['aug04']['flow_ci_upper'])}/"
        f"{report_value(rq2['aug04']['rq2_pass'])}`",
        "- rq3_6600_jul30_lower_ms_pass: "
        f"`{report_value(rq3['jul30']['bonferroni90_equal_weight_lower_ms'])}/"
        f"{report_value(rq3['jul30']['session_pass'])}`",
        "- rq3_6600_aug04_lower_ms_pass: "
        f"`{report_value(rq3['aug04']['bonferroni90_equal_weight_lower_ms'])}/"
        f"{report_value(rq3['aug04']['session_pass'])}`",
        "- rq3_850_role: `terminal_observability_normal_path_diagnostic_only/non_rescue`",
        f"- primary_results_sha256: `{primary_seal['primary_results_sha256']}`",
        "- primary_classification_sha256: "
        f"`{primary_seal['primary_classification_sha256']}`",
        f"- stage4_crosscheck_sha256: `{stage4_sha}`",
        f"- research_data_identity: `{research_identity}`",
        f"- code_contract_identity: `{runtime_identity}`",
        "- evidence_identity/composite_identity: bound by `h0b_manifest.json` to avoid report self-reference",
        "- outcome_access: `public_only_after_build_specific_admitted_permits`",
        "- stage4_access: `post_primary_seal_build_specific_diagnostic_permits_exact_projection_only`",
        "- aug07_access: `false`",
        "- network/private/order/cancel/live_access: `false`",
        "- claim_limit: `screening_audit_not_final_signal_or_strategy`",
    ]
    return "\n".join(lines) + "\n"


def copy_contract_surface(staging: Path) -> None:
    copies = {
        "contracts/execution_plan.md": DIAGNOSTIC_PLAN_PATH,
        "contracts/surface_matrix.json": MATRIX_PATH,
        "contracts/task.md": TASK_PATH,
        "contracts/v2_framework.md": FRAMEWORK_PATH,
        "runtime_source/skhynix_stage_h0b.py": (
            REPO_ROOT / "examples/hyperliquid/skhynix_stage_h0b.py"
        ),
        "runtime_source/skhynix_stage_h0b_contracts.py": (
            REPO_ROOT
            / "examples/hyperliquid/skhynix_stage_h0b_contracts.py"
        ),
        "runtime_tests/test_skhynix_stage_h0b.py": (
            REPO_ROOT / "examples/hyperliquid/test_skhynix_stage_h0b.py"
        ),
        "runtime_tests/test_skhynix_stage_h0b_package.py": (
            REPO_ROOT
            / "examples/hyperliquid/test_skhynix_stage_h0b_package.py"
        ),
    }
    for relative, source in copies.items():
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    matrix = read_json(MATRIX_PATH)
    write_json(
        staging / "contracts/accepted_kernel_pin.json",
        matrix["kernel_pin"],
    )


def require_publication_target_absent(final: Path) -> None:
    contracts.require(
        not Path(final).exists(),
        "PUBLICATION_FINAL_EXISTS",
        str(final),
        "final package root already exists",
    )


def assemble_package(
    *,
    build_a: Path,
    build_b: Path,
    final_root: Path,
) -> dict[str, Any]:
    final = Path(final_root)
    require_publication_target_absent(final)
    staging = final.with_name(
        f".{final.name}.staging-{os.getpid()}"
    )
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for directory in contracts.PACKAGE_DIRECTORIES:
        (staging / directory).mkdir()
    for relative in contracts.R_FILES:
        source = build_a / relative
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(
        build_a / "preoutcome_contract.json",
        staging / "preoutcome_contract.json",
    )
    copy_contract_surface(staging)
    evidence_copies = {
        "accepted_input_bindings.json": build_a / "accepted_input_bindings.json",
        "outcome_access_ledger_build_a.json": build_a / "outcome_access_ledger.json",
        "outcome_access_ledger_build_b.json": build_b / "outcome_access_ledger.json",
        "outcome_access_permit_build_a.json": build_a / "outcome_access_permit.json",
        "outcome_access_permit_build_b.json": build_b / "outcome_access_permit.json",
        "preoutcome_source_inventory.csv": build_a / "preoutcome_source_inventory.csv",
        "primary_result_seal.json": build_a / "primary_result_seal.json",
        "support_replay_receipt_build_a.json": build_a / "support_replay_receipt.json",
        "support_replay_receipt_build_b.json": build_b / "support_replay_receipt.json",
        "stage4_diagnostic_permit_build_a.json": build_a / "stage4_diagnostic_permit.json",
        "stage4_diagnostic_permit_build_b.json": build_b / "stage4_diagnostic_permit.json",
        "stage4_diagnostic_receipt_build_a.json": build_a / "stage4_diagnostic_receipt.json",
        "stage4_diagnostic_receipt_build_b.json": build_b / "stage4_diagnostic_receipt.json",
    }
    for relative, source in evidence_copies.items():
        shutil.copyfile(source, staging / relative)
    research = trust.compute_research_data_identity(
        contracts.file_inventory(staging, contracts.R_FILES)
    )
    runtime = trust.compute_runtime_contract_identity(
        research,
        runtime_contract_bridge(staging),
    )
    (staging / "reports/h0b_conditional_risk_audit.md").write_text(
        package_report_text(
            root=staging,
            research_identity=research,
            runtime_identity=runtime,
        ),
        encoding="utf-8",
        newline="\n",
    )
    identities = package_identities(staging)
    contracts.require(
        identities["research_data_identity"] == research
        and identities["runtime_contract_identity"] == runtime,
        "H0B_IDENTITY_BINDING_MISMATCH",
        "$.package_identity",
        "R/C changed while constructing report",
    )
    package_file_count = len(contracts.EXACT_PACKAGE_FILES) - 1
    package_total_bytes = sum(
        (staging / relative).stat().st_size
        for relative in contracts.EXACT_PACKAGE_FILES
        if relative != contracts.MANIFEST_FILE
    )
    manifest = {
        "schema_version": "skhynix_stage_h0b_manifest_v2",
        "task_id": contracts.TASK_ID,
        "status": "待验收",
        "primary_plan_sha256": PRIMARY_PLAN_SHA256,
        "diagnostic_plan_sha256": DIAGNOSTIC_PLAN_SHA256,
        "diagnostic_review_sha256": DIAGNOSTIC_REVIEW_SHA256,
        "primary_results_sha256": read_json(
            staging / "primary_result_seal.json"
        )["primary_results_sha256"],
        "primary_classification_sha256": contracts.sha256_file(
            staging / "primary_classification.json"
        ),
        "stage4_crosscheck_sha256": contracts.sha256_file(
            staging / "diagnostics/stage4_landmark_crosscheck.csv"
        ),
        **identities,
        "package_file_count": package_file_count,
        "package_total_bytes": package_total_bytes,
    }
    write_json(staging / contracts.MANIFEST_FILE, manifest)
    contracts.validate_exact_package_tree(staging)
    contracts.require(
        sum(
            path.stat().st_size
            for path in staging.rglob("*")
            if path.is_file()
        )
        <= 134_217_728,
        "H0B_PACKAGE_TREE_MISMATCH",
        str(staging),
        "package exceeds 128 MiB",
    )
    for path in sorted(staging.rglob("*"), reverse=True):
        if path.is_file():
            contracts.fsync_file(path)
    for path in sorted(
        [item for item in staging.rglob("*") if item.is_dir()],
        reverse=True,
    ):
        contracts.fsync_directory(path)
    contracts.fsync_directory(staging)
    os.replace(staging, final)
    contracts.fsync_directory(final.parent)
    return {
        "package_root": str(final),
        **identities,
        "package_file_count": package_file_count,
        "package_total_bytes": package_total_bytes,
    }


def validate_json_key_universes(root: Path) -> None:
    expected = {
        "accepted_input_bindings.json": {
            "schema_version",
            "task_id",
            "primary_plan_sha256",
            "diagnostic_plan_sha256",
            "diagnostic_review_sha256",
            "surface_matrix_sha256",
            "expected_semantic_source_inventory_sha256",
            "bindings",
        },
        "h0b_manifest.json": {
            "schema_version",
            "task_id",
            "status",
            "primary_plan_sha256",
            "diagnostic_plan_sha256",
            "diagnostic_review_sha256",
            "primary_results_sha256",
            "primary_classification_sha256",
            "stage4_crosscheck_sha256",
            "research_data_identity",
            "runtime_contract_identity",
            "publication_envelope_identity",
            "composite_package_identity",
            "package_file_count",
            "package_total_bytes",
        },
        "outcome_access_ledger_build_a.json": {
            "schema_version",
            "task_id",
            "build_label",
            "events",
        },
        "outcome_access_ledger_build_b.json": {
            "schema_version",
            "task_id",
            "build_label",
            "events",
        },
        "outcome_access_permit_build_a.json": {
            "schema_version",
            "task_id",
            "build_label",
            "status",
            "fsynced",
            "primary_plan_sha256",
            "diagnostic_plan_sha256",
            "diagnostic_review_sha256",
            "surface_matrix_sha256",
            "runtime_source_tree_sha256",
            "preoutcome_contract_sha256",
            "source_inventory_contract_sha256",
            "semantic_source_inventory_sha256",
            "build_envelope",
            "build_envelope_sha256",
            "support_replay_receipt_sha256",
            "accepted_input_bindings_sha256",
        },
        "outcome_access_permit_build_b.json": {
            "schema_version",
            "task_id",
            "build_label",
            "status",
            "fsynced",
            "primary_plan_sha256",
            "diagnostic_plan_sha256",
            "diagnostic_review_sha256",
            "surface_matrix_sha256",
            "runtime_source_tree_sha256",
            "preoutcome_contract_sha256",
            "source_inventory_contract_sha256",
            "semantic_source_inventory_sha256",
            "build_envelope",
            "build_envelope_sha256",
            "support_replay_receipt_sha256",
            "accepted_input_bindings_sha256",
        },
        "preoutcome_contract.json": {
            "schema_version",
            "task_id",
            "primary_plan_sha256",
            "diagnostic_plan_sha256",
            "diagnostic_review_sha256",
            "surface_matrix_sha256",
            "source_inventory_contract_sha256",
            "likelihood_contract_sha256",
            "design_matrix_contract_sha256",
            "walk_forward_contract_sha256",
            "resampling_contract_sha256",
            "classification_contract_sha256",
            "output_contract_sha256",
            "runtime_source_tree_sha256",
        },
        "primary_classification.json": {
            "schema_version",
            "task_id",
            "classification",
            "precedence_path",
            "gate_reasons",
            "formal_session_facts",
            "latency_roles",
            "claim_limit",
        },
        "primary_result_seal.json": {
            "schema_version",
            "task_id",
            "primary_plan_sha256",
            "diagnostic_plan_sha256",
            "diagnostic_review_sha256",
            "surface_matrix_sha256",
            "semantic_source_inventory_sha256",
            "build_a_primary_results_sha256",
            "build_b_primary_results_sha256",
            "primary_results_sha256",
            "primary_classification_sha256",
            "stage4_crosscheck_opened",
            "sealed_fsynced",
        },
        "stage4_diagnostic_permit_build_a.json": {
            "schema_version",
            "task_id",
            "build_label",
            "status",
            "fsynced",
            "primary_plan_sha256",
            "diagnostic_plan_sha256",
            "diagnostic_review_sha256",
            "surface_matrix_sha256",
            "runtime_source_tree_sha256",
            "primary_seal_sha256",
            "primary_results_sha256",
            "primary_classification_sha256",
            "stage4_projection_contract_sha256",
        },
        "stage4_diagnostic_permit_build_b.json": {
            "schema_version",
            "task_id",
            "build_label",
            "status",
            "fsynced",
            "primary_plan_sha256",
            "diagnostic_plan_sha256",
            "diagnostic_review_sha256",
            "surface_matrix_sha256",
            "runtime_source_tree_sha256",
            "primary_seal_sha256",
            "primary_results_sha256",
            "primary_classification_sha256",
            "stage4_projection_contract_sha256",
        },
        "stage4_diagnostic_receipt_build_a.json": {
            "schema_version",
            "task_id",
            "build_label",
            "primary_plan_sha256",
            "diagnostic_plan_sha256",
            "diagnostic_review_sha256",
            "diagnostic_permit_sha256",
            "stage4_crosscheck_sha256",
            "primary_results_sha256",
            "primary_classification_sha256",
            "primary_seal_sha256",
            "stage4_path_count",
            "stage4_projected_field_count",
            "joined_count",
            "eligible_count",
            "censored_count",
            "primary_seal_unchanged",
        },
        "stage4_diagnostic_receipt_build_b.json": {
            "schema_version",
            "task_id",
            "build_label",
            "primary_plan_sha256",
            "diagnostic_plan_sha256",
            "diagnostic_review_sha256",
            "diagnostic_permit_sha256",
            "stage4_crosscheck_sha256",
            "primary_results_sha256",
            "primary_classification_sha256",
            "primary_seal_sha256",
            "stage4_path_count",
            "stage4_projected_field_count",
            "joined_count",
            "eligible_count",
            "censored_count",
            "primary_seal_unchanged",
        },
        "support_replay_receipt_build_a.json": {
            "schema_version",
            "task_id",
            "build_label",
            "accepted_h0a_commitments_sha256",
            "observed_h0a_commitments_sha256",
            "exact_commitment_match",
            "forbidden_outcome_access_count",
            "replay_row_count",
        },
        "support_replay_receipt_build_b.json": {
            "schema_version",
            "task_id",
            "build_label",
            "accepted_h0a_commitments_sha256",
            "observed_h0a_commitments_sha256",
            "exact_commitment_match",
            "forbidden_outcome_access_count",
            "replay_row_count",
        },
    }
    for relative, keys in expected.items():
        observed = read_json(root / relative)
        contracts.require(
            set(observed) == keys,
            "H0B_OUTPUT_SCHEMA_MISMATCH",
            relative,
            f"missing={sorted(keys - set(observed))} "
            f"extra={sorted(set(observed) - keys)}",
        )


def verify_package(
    *,
    package: Path,
    report: Path | None = None,
) -> dict[str, Any]:
    root = Path(package)
    before = trust.metadata_snapshot(root)
    contracts.validate_exact_package_tree(root)
    for relative, fields in contracts.CSV_HEADERS.items():
        path = root / relative
        opener = gzip.open if relative.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), None)
        contracts.require(
            header == list(fields),
            "H0B_OUTPUT_SCHEMA_MISMATCH",
            relative,
            f"expected={list(fields)} observed={header}",
        )
    validate_json_key_universes(root)
    manifest = read_json(root / contracts.MANIFEST_FILE)
    identities = package_identities(root)
    contracts.require(
        manifest["primary_plan_sha256"] == PRIMARY_PLAN_SHA256
        and manifest["diagnostic_plan_sha256"] == DIAGNOSTIC_PLAN_SHA256
        and manifest["diagnostic_review_sha256"]
        == DIAGNOSTIC_REVIEW_SHA256,
        "H0B_IDENTITY_BINDING_MISMATCH",
        "$.manifest.plan_identities",
        "primary or diagnostic plan identity mismatch",
    )
    for field, observed in identities.items():
        contracts.require(
            manifest[field] == observed,
            "H0B_IDENTITY_BINDING_MISMATCH",
            f"$.manifest.{field}",
            f"declared={manifest[field]} observed={observed}",
        )
    seal = read_json(root / "primary_result_seal.json")
    contracts.require(
        primary_results_identity(root) == seal["primary_results_sha256"]
        and contracts.sha256_file(root / "primary_classification.json")
        == seal["primary_classification_sha256"]
        and seal["stage4_crosscheck_opened"] is False
        and seal["sealed_fsynced"] is True,
        "H0B_PRIMARY_SEAL_MISMATCH",
        "$.primary_seal",
        "primary seal mismatch",
    )
    contracts.require(
        contracts.sha256_file(
            root / "diagnostics/stage4_landmark_crosscheck.csv"
        )
        == manifest["stage4_crosscheck_sha256"],
        "H0B_STAGE4_PROJECTION_MISMATCH",
        "$.stage4_crosscheck",
        "manifest Stage 4 SHA mismatch",
    )
    permit_a = read_json(root / "outcome_access_permit_build_a.json")
    permit_b = read_json(root / "outcome_access_permit_build_b.json")
    contracts.require(
        permit_a["semantic_source_inventory_sha256"]
        == permit_b["semantic_source_inventory_sha256"]
        == EXPECTED_SEMANTIC_INVENTORY_SHA256
        and permit_a["build_envelope_sha256"]
        != permit_b["build_envelope_sha256"]
        and permit_a["build_label"] == "A"
        and permit_b["build_label"] == "B",
        "H0B_BUILD_ENVELOPE_MISMATCH",
        "$.permits",
        "A/B semantic or envelope contract mismatch",
    )
    receipt_a = read_json(root / "support_replay_receipt_build_a.json")
    receipt_b = read_json(root / "support_replay_receipt_build_b.json")
    contracts.require(
        receipt_a["exact_commitment_match"] is True
        and receipt_b["exact_commitment_match"] is True
        and receipt_a["forbidden_outcome_access_count"] == 0
        and receipt_b["forbidden_outcome_access_count"] == 0,
        "H0B_SUPPORT_COMMITMENT_MISMATCH",
        "$.support_replay",
        "support replay receipt mismatch",
    )
    diagnostic_permit_a = read_json(
        root / "stage4_diagnostic_permit_build_a.json"
    )
    diagnostic_permit_b = read_json(
        root / "stage4_diagnostic_permit_build_b.json"
    )
    diagnostic_receipt_a = read_json(
        root / "stage4_diagnostic_receipt_build_a.json"
    )
    diagnostic_receipt_b = read_json(
        root / "stage4_diagnostic_receipt_build_b.json"
    )
    diagnostic_crosscheck_sha = contracts.sha256_file(
        root / "diagnostics/stage4_landmark_crosscheck.csv"
    )
    contracts.require(
        diagnostic_permit_a["build_label"] == "A"
        and diagnostic_permit_b["build_label"] == "B"
        and diagnostic_permit_a["primary_seal_sha256"]
        == diagnostic_permit_b["primary_seal_sha256"]
        == contracts.sha256_file(root / "primary_result_seal.json")
        and diagnostic_receipt_a["diagnostic_permit_sha256"]
        == contracts.sha256_file(
            root / "stage4_diagnostic_permit_build_a.json"
        )
        and diagnostic_receipt_b["diagnostic_permit_sha256"]
        == contracts.sha256_file(
            root / "stage4_diagnostic_permit_build_b.json"
        )
        and diagnostic_receipt_a["stage4_crosscheck_sha256"]
        == diagnostic_receipt_b["stage4_crosscheck_sha256"]
        == diagnostic_crosscheck_sha
        and diagnostic_receipt_a["primary_seal_unchanged"] is True
        and diagnostic_receipt_b["primary_seal_unchanged"] is True,
        "H0B_STAGE4_PROJECTION_MISMATCH",
        "$.stage4_diagnostic",
        "diagnostic permit or receipt mismatch",
    )
    diagnostic_permit_shas = {
        "a": contracts.sha256_file(
            root / "stage4_diagnostic_permit_build_a.json"
        ),
        "b": contracts.sha256_file(
            root / "stage4_diagnostic_permit_build_b.json"
        ),
    }
    for label in ("a", "b"):
        ledger = read_json(root / f"outcome_access_ledger_build_{label}.json")
        validate_stage4_access_ledger(
            ledger,
            build_label=label.upper(),
            diagnostic_permit_sha256=diagnostic_permit_shas[label],
        )
        for event in ledger["events"]:
            relative = str(event["relative_path"]).lower()
            contracts.require(
                "aug07" not in relative
                and "decision_labels" not in relative
                and event["admitted"] is True,
                "H0B_AUG07_ACCESS_FORBIDDEN",
                relative,
                "forbidden or unadmitted ledger event",
            )
    expected_count = len(contracts.EXACT_PACKAGE_FILES) - 1
    expected_bytes = sum(
        (root / relative).stat().st_size
        for relative in contracts.EXACT_PACKAGE_FILES
        if relative != contracts.MANIFEST_FILE
    )
    contracts.require(
        manifest["package_file_count"] == expected_count
        and manifest["package_total_bytes"] == expected_bytes,
        "H0B_MANIFEST_SELF_REFERENCE_MISMATCH",
        "$.manifest.counts",
        f"expected={expected_count}/{expected_bytes}",
    )
    after = trust.metadata_snapshot(root)
    trust.assert_zero_write_snapshot(before, after, location=str(root))
    result = {
        "schema_version": "skhynix_stage_h0b_package_admission_v1",
        "task_id": contracts.TASK_ID,
        "package_root": str(root.resolve()),
        "verified": True,
        "file_count": len(contracts.EXACT_PACKAGE_FILES),
        "directory_count": len(contracts.PACKAGE_DIRECTORIES),
        "package_total_bytes_excluding_manifest": expected_bytes,
        **identities,
        "primary_results_sha256": seal["primary_results_sha256"],
        "primary_classification_sha256": seal[
            "primary_classification_sha256"
        ],
        "stage4_crosscheck_sha256": manifest[
            "stage4_crosscheck_sha256"
        ],
        "zero_write": True,
    }
    if report is not None:
        write_json(report, result)
    return result


def run_subprocess(command: Sequence[str]) -> dict[str, Any]:
    result = subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    contracts.require(
        result.returncode == 0,
        "H0B_BUILD_MISMATCH",
        "$.subprocess",
        f"command={command!r}\nstdout={result.stdout}\nstderr={result.stderr}",
    )
    return json.loads(result.stdout)


def compare_build_files(
    left: Path,
    right: Path,
    paths: Iterable[str],
) -> None:
    for relative in paths:
        contracts.require(
            (left / relative).read_bytes() == (right / relative).read_bytes(),
            "H0B_BUILD_MISMATCH",
            relative,
            "Build A/B bytes differ",
        )


def build_formal(
    *,
    task_path: Path,
    matrix_path: Path,
    output: Path,
    build_a: Path,
    build_b: Path,
    receipt: Path,
) -> dict[str, Any]:
    dispatch = validate_dispatch(task_path, matrix_path)
    hostile_path = (
        REPO_ROOT / ".workflow/reports/0823T002-hostile-preflight.json"
    )
    contracts.require(
        hostile_path.is_file(),
        "H0B_OUTCOME_ACCESS_BEFORE_PERMIT",
        str(hostile_path),
        "hostile preflight receipt is required",
    )
    validate_hostile_preflight_receipt(
        hostile_path,
        matrix_path=matrix_path,
        expected_dispatch=dispatch,
    )
    with tempfile.TemporaryDirectory(
        prefix="0823T002-formal-hostile-replay-"
    ) as raw:
        replay_path = Path(raw) / "hostile-preflight.json"
        hostile_preflight(
            task_path=task_path,
            matrix_path=matrix_path,
            output=replay_path,
            write_surface_evidence=False,
        )
        contracts.require(
            replay_path.read_bytes() == hostile_path.read_bytes(),
            "H0B_OUTCOME_ACCESS_BEFORE_PERMIT",
            str(hostile_path),
            "formal hostile replay differs from submitted receipt",
        )
    validator = [
        sys.executable,
        str(
            REPO_ROOT
            / ".workflow/workflow-kit/validate_research_package_task.py"
        ),
        "--task",
        str(task_path),
        "--matrix",
        str(matrix_path),
        "--negative-evidence",
        str(hostile_path),
    ]
    run_subprocess(validator)
    runner = str(REPO_ROOT / "examples/hyperliquid/skhynix_stage_h0b.py")
    for root, label in ((build_a, "A"), (build_b, "B")):
        run_subprocess(
            [
                sys.executable,
                runner,
                "h0b0",
                "--task",
                str(task_path),
                "--matrix",
                str(matrix_path),
                "--build-root",
                str(root),
                "--build-label",
                label,
            ]
        )
        run_subprocess(
            [
                sys.executable,
                runner,
                "outcome",
                "--build-root",
                str(root),
            ]
        )
    compare_build_files(
        build_a,
        build_b,
        contracts.PRIMARY_RESULT_FILES,
    )
    seal = write_primary_seal(build_a=build_a, build_b=build_b)
    diagnostic_permit_a = run_subprocess(
        [
            sys.executable,
            runner,
            "diagnostic-permit",
            "--build-root",
            str(build_a),
        ]
    )
    diagnostic_permit_b = run_subprocess(
        [
            sys.executable,
            runner,
            "diagnostic-permit",
            "--build-root",
            str(build_b),
        ]
    )
    validate_diagnostic_permit_pair(
        diagnostic_permit_a,
        diagnostic_permit_b,
        primary_seal_sha256=contracts.sha256_file(
            build_a / "primary_result_seal.json"
        ),
    )
    diagnostic_a = run_subprocess(
        [
            sys.executable,
            runner,
            "diagnostic",
            "--build-root",
            str(build_a),
        ]
    )
    diagnostic_b = run_subprocess(
        [
            sys.executable,
            runner,
            "diagnostic",
            "--build-root",
            str(build_b),
        ]
    )
    compare_build_files(
        build_a,
        build_b,
        ("diagnostics/stage4_landmark_crosscheck.csv",),
    )
    contracts.require(
        diagnostic_a["stage4_crosscheck_sha256"]
        == diagnostic_b["stage4_crosscheck_sha256"]
        and diagnostic_a["primary_results_sha256"]
        == diagnostic_b["primary_results_sha256"]
        == seal["primary_results_sha256"]
        and diagnostic_a["primary_seal_unchanged"] is True
        and diagnostic_b["primary_seal_unchanged"] is True,
        "H0B_BUILD_MISMATCH",
        "$.stage4_diagnostics",
        "A/B Stage 4 diagnostic mismatch",
    )
    package = assemble_package(
        build_a=build_a,
        build_b=build_b,
        final_root=output,
    )
    admission = verify_package(package=output)
    result = {
        "schema_version": "skhynix_stage_h0b_build_receipt_v1",
        "task_id": contracts.TASK_ID,
        "build_a": str(Path(build_a).resolve()),
        "build_b": str(Path(build_b).resolve()),
        "semantic_source_inventory_sha256": (
            EXPECTED_SEMANTIC_INVENTORY_SHA256
        ),
        "build_envelopes_distinct": True,
        "primary_result_seal": seal,
        "stage4_permit_build_a": diagnostic_permit_a,
        "stage4_permit_build_b": diagnostic_permit_b,
        "stage4_build_a": diagnostic_a,
        "stage4_build_b": diagnostic_b,
        "package": package,
        "admission": admission,
        "aug07_event_rows_opened": False,
        "network_private_order_cancel_live_access": False,
    }
    write_json(receipt, result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    hostile = subparsers.add_parser("hostile-preflight")
    hostile.add_argument("--task", type=Path, required=True)
    hostile.add_argument("--matrix", type=Path, required=True)
    hostile.add_argument("--output", type=Path, required=True)

    negative = subparsers.add_parser("negative-case")
    negative.add_argument("--surface", required=True)
    negative.add_argument("--expected-code", required=True)

    h0b0 = subparsers.add_parser("h0b0")
    h0b0.add_argument("--task", type=Path, required=True)
    h0b0.add_argument("--matrix", type=Path, required=True)
    h0b0.add_argument("--build-root", type=Path, required=True)
    h0b0.add_argument("--build-label", required=True)

    outcome = subparsers.add_parser("outcome")
    outcome.add_argument("--build-root", type=Path, required=True)

    diagnostic_permit = subparsers.add_parser("diagnostic-permit")
    diagnostic_permit.add_argument(
        "--build-root",
        type=Path,
        required=True,
    )

    diagnostic = subparsers.add_parser("diagnostic")
    diagnostic.add_argument("--build-root", type=Path, required=True)

    formal = subparsers.add_parser("build-formal")
    formal.add_argument("--task", type=Path, required=True)
    formal.add_argument("--matrix", type=Path, required=True)
    formal.add_argument("--output", type=Path, required=True)
    formal.add_argument("--build-a", type=Path, required=True)
    formal.add_argument("--build-b", type=Path, required=True)
    formal.add_argument("--receipt", type=Path, required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--package", type=Path, required=True)
    verify.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "hostile-preflight":
            result = hostile_preflight(
                task_path=args.task,
                matrix_path=args.matrix,
                output=args.output,
            )
        elif args.command == "negative-case":
            try:
                negative_case(args.surface, args.expected_code)
            except contracts.H0BError as exc:
                print(
                    json.dumps(
                        {"verified": False, "error": exc.as_dict()},
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 2
            raise contracts.H0BError(
                HOSTILE_FAIL_OPEN_SENTINEL,
                "$.negative_case",
                "mutation failed open",
            )
        elif args.command == "h0b0":
            result = run_h0b0(
                task_path=args.task,
                matrix_path=args.matrix,
                build_root=args.build_root,
                build_label=args.build_label,
            )
        elif args.command == "outcome":
            result = run_h0b1(build_root=args.build_root)
        elif args.command == "diagnostic-permit":
            result = write_stage4_diagnostic_permit(args.build_root)
        elif args.command == "diagnostic":
            result = run_stage4_diagnostic(build_root=args.build_root)
        elif args.command == "build-formal":
            result = build_formal(
                task_path=args.task,
                matrix_path=args.matrix,
                output=args.output,
                build_a=args.build_a,
                build_b=args.build_b,
                receipt=args.receipt,
            )
        else:
            result = verify_package(
                package=args.package,
                report=args.report,
            )
    except (OSError, ValueError, contracts.H0BError) as exc:
        error = (
            exc.as_dict()
            if isinstance(exc, contracts.H0BError)
            else {
                "code": "H0B_RUNTIME_ERROR",
                "location": "$",
                "detail": str(exc),
            }
        )
        print(
            json.dumps(
                {"verified": False, "error": error},
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
