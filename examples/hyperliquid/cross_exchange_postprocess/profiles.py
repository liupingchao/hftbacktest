"""Versioned pipeline profile registry."""

from __future__ import annotations

from typing import Any


PROFILE_REGISTRY_SCHEMA_VERSION = "cross_exchange_postprocess_profiles_v1"

PROFILES: dict[str, dict[str, Any]] = {
    "dataset": {
        "version": "1",
        "executable": True,
        "stages": ["raw_audit", "r0", "r1"],
        "description": "Auditable campaign input to replay-ready R0/R1 dataset.",
        "capabilities": {
            "raw_integrity": "required",
            "epoch_aware_l2_replay": "required",
            "exact_reconnect_masks": "required",
            "alignment_labels": "required",
            "basis_dislocation": "not_run",
            "lead_lag": "not_run",
            "maker_diagnostics": "not_run",
            "liquidity_response_hierarchy": "not_run",
        },
    },
    "basis-research": {
        "version": "1",
        "executable": True,
        "stages": ["raw_audit", "r0", "r1", "basis_dislocation"],
        "description": (
            "Replay-ready R0/R1 dataset plus point-in-time basis and "
            "directional BBO dislocation state."
        ),
        "capabilities": {
            "raw_integrity": "required",
            "epoch_aware_l2_replay": "required",
            "exact_reconnect_masks": "required",
            "alignment_labels": "required",
            "basis_dislocation": "required",
            "lead_lag": "not_run",
            "maker_diagnostics": "not_run",
            "liquidity_response_hierarchy": "not_run",
        },
    },
    "signal-research": {
        "version": "1",
        "executable": False,
        "stages": [
            "raw_audit",
            "r0",
            "r1",
            "basis_dislocation",
            "lead_lag",
            "maker_diagnostics",
        ],
        "description": "Dataset plus directional basis, lead-lag and maker diagnostics.",
        "not_implemented_stages": [
            "lead_lag",
            "maker_diagnostics",
        ],
    },
    "full-research": {
        "version": "1",
        "executable": False,
        "stages": [
            "raw_audit",
            "r0",
            "r1",
            "basis_dislocation",
            "lead_lag",
            "maker_diagnostics",
            "shock_atom",
            "flow_episode",
            "motif_prototype",
            "temporary_regime",
        ],
        "description": "Signal research plus liquidity-response hierarchy.",
        "not_implemented_stages": [
            "lead_lag",
            "maker_diagnostics",
            "shock_atom",
            "flow_episode",
            "motif_prototype",
            "temporary_regime",
        ],
    },
}


def get_profile(profile_id: str) -> dict[str, Any]:
    try:
        return PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(
            f"unknown pipeline profile {profile_id!r}; available={sorted(PROFILES)}"
        ) from exc
