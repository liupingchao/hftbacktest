#!/usr/bin/env python3
"""Execute the zero-target SAFE_REENTRY_AFTER_FLOW_EXCURSION_V1 A0 audit."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from examples.hyperliquid import skhynix_liquidity_break_onset_a0 as upstream
from examples.hyperliquid.skhynix_phase_alignment_track_a import (
    DEFAULT_BINDINGS,
    Capture,
    _save_npz_deterministic,
    _sha256,
    _write_csv,
    _write_json,
    discover_captures,
)


TASK_ID = "0828T011"
SCHEMA_VERSION = "skhynix_safe_reentry_after_flow_excursion_a0_v1"
HYPOTHESIS_ID = "SAFE_REENTRY_AFTER_FLOW_EXCURSION_V1"
CONTRACT_PATH = Path(
    "docs/skhynix_binance_safe_reentry_after_flow_excursion_v1_a0_plan_20260828.md"
)
CONTRACT_SHA256 = "05c082e38bfbc5c7292fa1c36bb8886888e1aecb0430a20d374688de9cae079a"
UPSTREAM_DIR = Path(
    "local_live_analysis/skhynix_liquidity_break_onset_a0_0828T008"
)
UPSTREAM_NORMALIZATION_SHA256 = (
    "ce21bd913fd236431b97f89dc6d707b4fd08a9b92027a3a5fbd87d6b8ffd3b89"
)
UPSTREAM_PRESSURE_SHA256 = (
    "982bbadae02b92808047970a3bb1c6adf23792135f6a3147260ad54bbfac4435"
)
DEFAULT_OUT_DIR = Path(
    "local_live_analysis/skhynix_safe_reentry_after_flow_excursion_a0_0828T011"
)

CHECKPOINT_NS = 20_000_000
NOVELTY_NS = 1_000_000_000
NOVELTY_CHECKPOINTS = NOVELTY_NS // CHECKPOINT_NS
PERSISTENCE_WINDOW_NS = 250_000_000
PERSISTENCE_EXPOSURE_NS = 100_000_000
PERSISTENCE_CHECKPOINTS = PERSISTENCE_EXPOSURE_NS // CHECKPOINT_NS
REFRACTORY_NS = 1_000_000_000
CONTROL_STRIDE_NS = 250_000_000
CONTROL_HISTORY_EXCLUSION_NS = 5_000_000_000
DEPTH_RECOVERY_FRACTION = 0.80
SAFE_SPREAD_TICKS = 2.0
MAX_ABS_OBI = 0.50
RELEASE_SCORE = 1.5
TAU_CANDIDATES_MS = (500, 1_000, 2_000, 5_000, 10_000)

BACKGROUND_BUILDING = "BACKGROUND_BUILDING"
BACKGROUND_READY = "BACKGROUND_READY"
EXCURSION_CANDIDATE = "EXCURSION_CANDIDATE"
EXCURSION_ACTIVE = "EXCURSION_ACTIVE"
RECOVERY_CANDIDATE = "RECOVERY_CANDIDATE"
REFRACTORY = "REFRACTORY"


class A0Error(RuntimeError):
    """Fail-closed A0 error."""


@dataclass
class CacheRef:
    capture: Capture
    final_path: Path
    event_key_path: Path
    segment_end_by_id: dict[int, int]
    first_ts_ns: int
    last_ts_ns: int


@dataclass
class CaptureDetection:
    candidates: list[dict[str, Any]]
    episodes: list[dict[str, Any]]
    anchors: list[dict[str, Any]]
    process_intervals: list[dict[str, Any]]
    diagnostics: dict[str, Any]


def _progress(message: str) -> None:
    print(f"[SAFE-REENTRY-A0] {message}", file=sys.stderr, flush=True)


def _safe_percentile(values: Sequence[float] | np.ndarray, q: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, q)) if len(array) else math.nan


def _component_pair(z_values: np.ndarray) -> str:
    return upstream._component_pair(z_values)


def _orientation_indices(direction: int) -> tuple[int, int, int]:
    return upstream._orientation_indices(direction)


def _structural_arrays(
    x: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    plus_x = x[:, [0, 2, 4]]
    minus_x = x[:, [1, 3, 5]]
    plus_z = z[:, [0, 2, 4]]
    minus_z = z[:, [1, 3, 5]]
    plus_score = np.sum(np.maximum(plus_z, 0.0), axis=1)
    minus_score = np.sum(np.maximum(minus_z, 0.0), axis=1)
    plus = (
        (np.sum(plus_z >= upstream.Z_STAR, axis=1) >= 2)
        & (plus_score >= upstream.PRESSURE_STAR)
        & (plus_x[:, 0] > 0)
    )
    minus = (
        (np.sum(minus_z >= upstream.Z_STAR, axis=1) >= 2)
        & (minus_score >= upstream.PRESSURE_STAR)
        & (minus_x[:, 0] > 0)
    )
    direction = np.zeros(len(x), dtype=np.int8)
    direction[plus & ~minus] = 1
    direction[minus & ~plus] = -1
    both = plus & minus
    ambiguous = both & (
        np.abs(plus_score - minus_score) < upstream.CONFLICT_GAP
    )
    direction[both & ~ambiguous & (plus_score > minus_score)] = 1
    direction[both & ~ambiguous & (minus_score > plus_score)] = -1
    return plus, minus, plus_score, minus_score, direction


def _edge_crossings(
    predicate: np.ndarray,
    eligible: np.ndarray,
    ts: np.ndarray,
    segments: np.ndarray,
) -> np.ndarray:
    prior = np.zeros(len(predicate), dtype=bool)
    contiguous = (
        (segments[1:] == segments[:-1])
        & (ts[1:] - ts[:-1] == CHECKPOINT_NS)
        & eligible[:-1]
    )
    prior[1:] = predicate[:-1] & contiguous
    return predicate & eligible & ~prior


def build_event_key_cache(
    capture: Capture,
    final_path: Path,
    cache_dir: Path,
) -> Path:
    path = cache_dir / f"{capture.capture_id}.npz"
    with np.load(final_path, allow_pickle=False) as final:
        final_ts = final["ts_ns"].copy()
        final_segments = final["segment_id"].copy()
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            if not np.array_equal(data["ts_ns"], final_ts):
                raise A0Error(f"event_key_ts_mismatch:{capture.capture_id}")
            if not np.array_equal(data["segment_id"], final_segments):
                raise A0Error(f"event_key_segment_mismatch:{capture.capture_id}")
        return path

    ts_values: list[int] = []
    event_seq_values: list[int] = []
    segment_values: list[int] = []

    def collect(snapshot: upstream.ReplaySnapshot) -> None:
        ts_values.append(snapshot.ts_ns)
        event_seq_values.append(snapshot.event_seq)
        segment_values.append(snapshot.segment_id)

    engine = upstream.ReplayEngine(capture)
    engine.run(on_checkpoint=collect)
    replay_ts = np.asarray(ts_values, dtype=np.int64)
    replay_segments = np.asarray(segment_values, dtype=np.int64)
    if not np.array_equal(replay_ts, final_ts):
        raise A0Error(f"replay_checkpoint_ts_mismatch:{capture.capture_id}")
    if not np.array_equal(replay_segments, final_segments):
        raise A0Error(f"replay_checkpoint_segment_mismatch:{capture.capture_id}")
    _save_npz_deterministic(
        path,
        ts_ns=replay_ts,
        event_seq=np.asarray(event_seq_values, dtype=np.int64),
        segment_id=replay_segments,
    )
    return path


def _episode_id(
    capture_id: str,
    segment_id: int,
    candidate_ts: int,
    candidate_seq: int,
    confirmation_ts: int,
    confirmation_seq: int,
) -> str:
    payload = "|".join(
        (
            HYPOTHESIS_ID,
            capture_id,
            str(segment_id),
            str(candidate_ts),
            str(candidate_seq),
            str(confirmation_ts),
            str(confirmation_seq),
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _baseline(
    ts: np.ndarray,
    segment: np.ndarray,
    eligible: np.ndarray,
    bid_depth: np.ndarray,
    ask_depth: np.ndarray,
    spread: np.ndarray,
    obi: np.ndarray,
    activity: np.ndarray,
    candidate_index: int,
) -> dict[str, float] | None:
    candidate_ts = int(ts[candidate_index])
    lower = int(np.searchsorted(ts, candidate_ts - NOVELTY_NS, side="left"))
    upper = int(
        np.searchsorted(ts, candidate_ts - PERSISTENCE_EXPOSURE_NS, side="left")
    )
    indices = np.arange(lower, upper, dtype=np.int64)
    if len(indices) != 45:
        return None
    if (
        indices[0] < 0
        or indices[-1] >= candidate_index
        or not np.all(segment[indices] == segment[candidate_index])
        or not np.all(eligible[indices])
        or not np.all(np.diff(ts[indices]) == CHECKPOINT_NS)
    ):
        return None
    return {
        "bid_depth_baseline": float(np.median(bid_depth[indices])),
        "ask_depth_baseline": float(np.median(ask_depth[indices])),
        "spread_baseline": float(np.median(spread[indices])),
        "obi_baseline": float(np.median(obi[indices])),
        "activity_baseline": float(np.median(activity[indices])),
        "baseline_checkpoint_count": int(len(indices)),
    }


def run_state_machine(
    capture: Capture,
    arrays: dict[str, np.ndarray],
    event_seq: np.ndarray,
    global_scale_floors: np.ndarray,
) -> CaptureDetection:
    ts = arrays["ts_ns"].astype(np.int64, copy=False)
    segments = arrays["segment_id"].astype(np.int64, copy=False)
    valid = arrays["valid"].astype(bool, copy=False)
    x = arrays["x"].astype(np.float64, copy=False)
    z = arrays["z"].astype(np.float64, copy=False)
    local_scale = arrays["local_scale"].astype(np.float64, copy=False)
    bid_depth = arrays["bid_depth_current"].astype(np.float64, copy=False)
    ask_depth = arrays["ask_depth_current"].astype(np.float64, copy=False)
    obi = arrays["obi_current"].astype(np.float64, copy=False)
    spread = arrays["spread_ticks"].astype(np.float64, copy=False)
    activity = arrays["activity_count"].astype(np.int64, copy=False)

    finite = (
        np.all(np.isfinite(x), axis=1)
        & np.all(np.isfinite(z), axis=1)
        & np.all(np.isfinite(local_scale), axis=1)
        & np.isfinite(bid_depth)
        & np.isfinite(ask_depth)
        & np.isfinite(obi)
        & np.isfinite(spread)
    )
    eligible = valid & finite
    plus, minus, plus_score, minus_score, direction = _structural_arrays(x, z)
    plus &= eligible
    minus &= eligible
    quiet = (
        eligible
        & ~plus
        & ~minus
        & (plus_score < RELEASE_SCORE)
        & (minus_score < RELEASE_SCORE)
    )
    plus_edges = _edge_crossings(plus, eligible, ts, segments)
    minus_edges = _edge_crossings(minus, eligible, ts, segments)

    candidates: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    anchors: list[dict[str, Any]] = []
    process_intervals: list[dict[str, Any]] = []
    state = BACKGROUND_BUILDING
    quiet_count = 0
    candidate: dict[str, Any] | None = None
    episode: dict[str, Any] | None = None
    refractory_started_at: int | None = None
    last_segment: int | None = None
    raw_edges = int(np.sum(plus_edges) + np.sum(minus_edges))

    counters = Counter()

    def reset_background() -> None:
        nonlocal state, quiet_count, candidate, episode, refractory_started_at
        state = BACKGROUND_BUILDING
        quiet_count = 0
        candidate = None
        episode = None
        refractory_started_at = None

    def close_candidate(index: int, status: str) -> None:
        nonlocal candidate
        if candidate is None:
            return
        row = {
            **candidate,
            "candidate_status": status,
            "candidate_end_ts_ns": int(ts[index]),
            "candidate_end_event_seq": int(event_seq[index]),
            "qualification_elapsed_ms": (
                int(ts[index]) - int(candidate["candidate_ts_ns"])
            )
            / 1e6,
        }
        candidates.append(row)
        process_intervals.append(
            {
                "capture_id": capture.capture_id,
                "start_ts_ns": candidate["candidate_ts_ns"],
                "end_ts_ns": int(ts[index]),
                "kind": "candidate",
            }
        )
        counters[status] += 1
        candidate = None

    def update_path_metrics(index: int) -> None:
        target = episode if episode is not None else candidate
        if target is None:
            return
        target["peak_pressure_up"] = max(
            float(target["peak_pressure_up"]), float(plus_score[index])
        )
        target["peak_pressure_down"] = max(
            float(target["peak_pressure_down"]), float(minus_score[index])
        )
        target["integrated_pressure_up_score_ms"] += (
            float(plus_score[index]) * CHECKPOINT_NS / 1e6
        )
        target["integrated_pressure_down_score_ms"] += (
            float(minus_score[index]) * CHECKPOINT_NS / 1e6
        )
        target["raw_micro_crossing_count"] += int(plus_edges[index])
        target["raw_micro_crossing_count"] += int(minus_edges[index])
        if episode is not None:
            episode["maximum_bid_depth_deficit"] = max(
                float(episode["maximum_bid_depth_deficit"]),
                max(
                    0.0,
                    1.0
                    - float(bid_depth[index])
                    / max(float(episode["bid_depth_baseline"]), 1e-12),
                ),
            )
            episode["maximum_ask_depth_deficit"] = max(
                float(episode["maximum_ask_depth_deficit"]),
                max(
                    0.0,
                    1.0
                    - float(ask_depth[index])
                    / max(float(episode["ask_depth_baseline"]), 1e-12),
                ),
            )
            episode["minimum_spread_ticks"] = min(
                float(episode["minimum_spread_ticks"]), float(spread[index])
            )
            episode["maximum_spread_ticks"] = max(
                float(episode["maximum_spread_ticks"]), float(spread[index])
            )

    def close_episode(index: int, terminal_status: str) -> None:
        nonlocal episode
        if episode is None:
            return
        row = {
            **episode,
            "terminal_ts_ns": int(ts[index]),
            "terminal_event_seq": int(event_seq[index]),
            "terminal_status": terminal_status,
            "episode_duration_ms": (
                int(ts[index]) - int(episode["candidate_ts_ns"])
            )
            / 1e6,
            "confirmed_duration_ms": (
                int(ts[index]) - int(episode["confirmation_ts_ns"])
            )
            / 1e6,
            "recovery_duration_ms": (
                int(ts[index]) - int(episode["first_recovery_candidate_ts_ns"])
            )
            / 1e6
            if episode["first_recovery_candidate_ts_ns"] != ""
            else math.nan,
            "terminal_spread_ticks": float(spread[index]),
            "terminal_obi": float(obi[index]),
            "terminal_bid_depth": float(bid_depth[index]),
            "terminal_ask_depth": float(ask_depth[index]),
            "terminal_bid_recovered": bool(
                bid_depth[index]
                >= DEPTH_RECOVERY_FRACTION * episode["bid_depth_baseline"]
            ),
            "terminal_ask_recovered": bool(
                ask_depth[index]
                >= DEPTH_RECOVERY_FRACTION * episode["ask_depth_baseline"]
            ),
        }
        episodes.append(row)
        process_intervals.append(
            {
                "capture_id": capture.capture_id,
                "start_ts_ns": episode["candidate_ts_ns"],
                "end_ts_ns": int(ts[index]),
                "kind": "confirmed_episode",
            }
        )
        counters[terminal_status] += 1
        episode = None

    def start_candidate(index: int, initial_direction: int) -> bool:
        nonlocal candidate, state
        base = _baseline(
            ts,
            segments,
            eligible,
            bid_depth,
            ask_depth,
            spread,
            obi,
            activity,
            index,
        )
        common = {
            "capture_id": capture.capture_id,
            "research_date": capture.research_date,
            "role": capture.role,
            "segment_id": int(segments[index]),
            "candidate_ts_ns": int(ts[index]),
            "candidate_event_seq": int(event_seq[index]),
            "candidate_checkpoint_index": int(index),
            "initial_direction": int(initial_direction),
            "prequiet_checkpoint_count": int(quiet_count),
            "qualifying_exposure_ms": 0.0,
            "peak_pressure_up": float(plus_score[index]),
            "peak_pressure_down": float(minus_score[index]),
            "integrated_pressure_up_score_ms": 0.0,
            "integrated_pressure_down_score_ms": 0.0,
            "raw_micro_crossing_count": 0,
        }
        if base is None:
            candidate = common
            close_candidate(index, "baseline_incomplete")
            reset_background()
            return False
        candidate = {**common, **base}
        state = EXCURSION_CANDIDATE
        update_path_metrics(index)
        return True

    def confirm_candidate(index: int) -> None:
        nonlocal candidate, episode, state
        assert candidate is not None
        initial_direction = int(candidate["initial_direction"])
        oriented = list(_orientation_indices(initial_direction))
        nonfloor_count = int(
            np.sum(local_scale[index, oriented] > global_scale_floors[oriented])
        )
        oriented_z = z[index, oriented]
        episode = {
            **candidate,
            "episode_id": _episode_id(
                capture.capture_id,
                int(candidate["segment_id"]),
                int(candidate["candidate_ts_ns"]),
                int(candidate["candidate_event_seq"]),
                int(ts[index]),
                int(event_seq[index]),
            ),
            "confirmation_ts_ns": int(ts[index]),
            "confirmation_event_seq": int(event_seq[index]),
            "confirmation_checkpoint_index": int(index),
            "confirmation_delay_ms": (
                int(ts[index]) - int(candidate["candidate_ts_ns"])
            )
            / 1e6,
            "confirmation_component_pair": _component_pair(oriented_z),
            "confirmation_nonfloor_component_count": nonfloor_count,
            "current_dominant_direction": initial_direction,
            "direction_switch_count": 0,
            "recovery_reset_count": 0,
            "refractory_reset_count": 0,
            "first_recovery_candidate_ts_ns": "",
            "first_refractory_ts_ns": "",
            "maximum_bid_depth_deficit": 0.0,
            "maximum_ask_depth_deficit": 0.0,
            "minimum_spread_ticks": float(spread[index]),
            "maximum_spread_ticks": float(spread[index]),
        }
        candidates.append(
            {
                **candidate,
                "candidate_status": "confirmed",
                "candidate_end_ts_ns": int(ts[index]),
                "candidate_end_event_seq": int(event_seq[index]),
                "qualification_elapsed_ms": (
                    int(ts[index]) - int(candidate["candidate_ts_ns"])
                )
                / 1e6,
            }
        )
        counters["confirmed"] += 1
        candidate = None
        state = EXCURSION_ACTIVE

    def maybe_switch_direction(index: int) -> None:
        if episode is None:
            return
        chosen = int(direction[index])
        if chosen and chosen != int(episode["current_dominant_direction"]):
            episode["direction_switch_count"] += 1
            episode["current_dominant_direction"] = chosen

    def depth_recovered(index: int) -> bool:
        assert episode is not None
        return bool(
            bid_depth[index]
            >= DEPTH_RECOVERY_FRACTION * episode["bid_depth_baseline"]
            and ask_depth[index]
            >= DEPTH_RECOVERY_FRACTION * episode["ask_depth_baseline"]
        )

    def begin_recovery(index: int) -> None:
        nonlocal state, refractory_started_at
        assert episode is not None
        state = RECOVERY_CANDIDATE
        if episode["first_recovery_candidate_ts_ns"] == "":
            episode["first_recovery_candidate_ts_ns"] = int(ts[index])
        if depth_recovered(index):
            state = REFRACTORY
            refractory_started_at = int(ts[index])
            if episode["first_refractory_ts_ns"] == "":
                episode["first_refractory_ts_ns"] = int(ts[index])

    for index in range(len(ts)):
        segment_changed = last_segment is not None and int(segments[index]) != last_segment
        if segment_changed or not eligible[index]:
            if candidate is not None:
                close_candidate(index, "reset_or_quality_censored")
            if episode is not None:
                close_episode(index, "reset_or_quality_censored")
            reset_background()
            last_segment = int(segments[index])
            continue
        last_segment = int(segments[index])

        if state in {BACKGROUND_BUILDING, BACKGROUND_READY}:
            if quiet[index]:
                quiet_count += 1
                state = (
                    BACKGROUND_READY
                    if quiet_count >= NOVELTY_CHECKPOINTS
                    else BACKGROUND_BUILDING
                )
                continue
            if state == BACKGROUND_READY and (plus[index] or minus[index]):
                if plus[index] and minus[index] and direction[index] == 0:
                    candidate = {
                        "capture_id": capture.capture_id,
                        "research_date": capture.research_date,
                        "role": capture.role,
                        "segment_id": int(segments[index]),
                        "candidate_ts_ns": int(ts[index]),
                        "candidate_event_seq": int(event_seq[index]),
                        "candidate_checkpoint_index": int(index),
                        "initial_direction": 0,
                        "prequiet_checkpoint_count": int(quiet_count),
                        "qualifying_exposure_ms": 0.0,
                        "peak_pressure_up": float(plus_score[index]),
                        "peak_pressure_down": float(minus_score[index]),
                        "integrated_pressure_up_score_ms": 0.0,
                        "integrated_pressure_down_score_ms": 0.0,
                        "raw_micro_crossing_count": int(plus_edges[index])
                        + int(minus_edges[index]),
                    }
                    close_candidate(index, "direction_ambiguous")
                    reset_background()
                    continue
                start_candidate(index, int(direction[index]))
                quiet_count = 0
                continue
            quiet_count = 0
            state = BACKGROUND_BUILDING
            continue

        if state == EXCURSION_CANDIDATE:
            assert candidate is not None
            elapsed = int(ts[index]) - int(candidate["candidate_ts_ns"])
            if elapsed > PERSISTENCE_WINDOW_NS:
                close_candidate(index, "transient_rejected")
                reset_background()
                if quiet[index]:
                    quiet_count = 1
                continue
            initial_direction = int(candidate["initial_direction"])
            if int(direction[index]) == -initial_direction:
                close_candidate(index, "pre_confirmation_direction_switch")
                reset_background()
                continue
            update_path_metrics(index)
            qualifies = plus[index] if initial_direction == 1 else minus[index]
            if qualifies:
                candidate["qualifying_exposure_ms"] += CHECKPOINT_NS / 1e6
            if (
                candidate["qualifying_exposure_ms"]
                >= PERSISTENCE_EXPOSURE_NS / 1e6
            ):
                confirm_candidate(index)
                maybe_switch_direction(index)
                if quiet[index]:
                    begin_recovery(index)
            continue

        assert episode is not None
        update_path_metrics(index)
        maybe_switch_direction(index)

        if state == EXCURSION_ACTIVE:
            if quiet[index]:
                begin_recovery(index)
            continue

        if state == RECOVERY_CANDIDATE:
            if not quiet[index]:
                state = EXCURSION_ACTIVE
                episode["recovery_reset_count"] += 1
                refractory_started_at = None
                continue
            if depth_recovered(index):
                state = REFRACTORY
                refractory_started_at = int(ts[index])
                if episode["first_refractory_ts_ns"] == "":
                    episode["first_refractory_ts_ns"] = int(ts[index])
            continue

        if state == REFRACTORY:
            if not quiet[index]:
                state = EXCURSION_ACTIVE
                episode["refractory_reset_count"] += 1
                refractory_started_at = None
                continue
            if not depth_recovered(index):
                state = RECOVERY_CANDIDATE
                episode["refractory_reset_count"] += 1
                refractory_started_at = None
                continue
            assert refractory_started_at is not None
            if int(ts[index]) - refractory_started_at < REFRACTORY_NS:
                continue
            episode["refractory_completed_ts_ns"] = int(ts[index])
            episode["refractory_completed_event_seq"] = int(event_seq[index])
            episode["refractory_duration_ms"] = (
                int(ts[index]) - refractory_started_at
            ) / 1e6
            spread_ok = float(spread[index]) >= SAFE_SPREAD_TICKS - 1e-9
            obi_ok = abs(float(obi[index])) <= MAX_ABS_OBI
            if spread_ok and obi_ok:
                oriented = list(
                    _orientation_indices(int(episode["initial_direction"]))
                )
                anchor = {
                    "episode_id": episode["episode_id"],
                    "capture_id": capture.capture_id,
                    "research_date": capture.research_date,
                    "role": capture.role,
                    "segment_id": int(segments[index]),
                    "anchor_ts_ns": int(ts[index]),
                    "anchor_event_seq": int(event_seq[index]),
                    "direction": int(episode["initial_direction"]),
                    "spread_ticks": float(spread[index]),
                    "obi_current": float(obi[index]),
                    "bid_depth_current": float(bid_depth[index]),
                    "ask_depth_current": float(ask_depth[index]),
                    "total_depth_current": float(
                        bid_depth[index] + ask_depth[index]
                    ),
                    "activity_count": int(activity[index]),
                    "bid_depth_baseline": float(
                        episode["bid_depth_baseline"]
                    ),
                    "ask_depth_baseline": float(
                        episode["ask_depth_baseline"]
                    ),
                    "nonfloor_component_count": int(
                        np.sum(
                            local_scale[index, oriented]
                            > global_scale_floors[oriented]
                        )
                    ),
                }
                anchors.append(anchor)
                close_episode(index, "safe_reentry_available")
            elif not spread_ok:
                close_episode(index, "recovered_without_wide_spread")
            else:
                close_episode(index, "recovered_but_imbalanced")
            reset_background()

    if len(ts):
        last = len(ts) - 1
        if candidate is not None:
            close_candidate(last, "capture_end_censored")
        if episode is not None:
            close_episode(last, "never_recovered")

    diagnostics = {
        "capture_id": capture.capture_id,
        "research_date": capture.research_date,
        "checkpoint_count": len(ts),
        "eligible_checkpoint_count": int(np.sum(eligible)),
        "quiet_checkpoint_count": int(np.sum(quiet)),
        "structural_checkpoint_count": int(np.sum(plus | minus)),
        "raw_micro_crossing_count": raw_edges,
        "candidate_count": len(candidates),
        "confirmed_excursion_count": len(episodes),
        "safe_reentry_anchor_count": len(anchors),
        **dict(counters),
    }
    return CaptureDetection(
        candidates=candidates,
        episodes=episodes,
        anchors=anchors,
        process_intervals=process_intervals,
        diagnostics=diagnostics,
    )


def _inside_interval(
    intervals: Sequence[tuple[int, int]], ts_ns: int
) -> bool:
    if not intervals:
        return False
    starts = [row[0] for row in intervals]
    index = bisect.bisect_right(starts, ts_ns) - 1
    return bool(
        index >= 0 and intervals[index][0] <= ts_ns <= intervals[index][1]
    )


def build_controls(
    caches: Sequence[CacheRef],
    detections: dict[str, CaptureDetection],
    depth_floor: float,
) -> list[dict[str, Any]]:
    controls: list[dict[str, Any]] = []
    control_id = 0
    for cache in caches:
        detection = detections[cache.capture.capture_id]
        intervals = sorted(
            (
                int(row["start_ts_ns"]),
                int(row["end_ts_ns"]),
            )
            for row in detection.process_intervals
        )
        recent_excursions = sorted(
            int(row["confirmation_ts_ns"]) for row in detection.episodes
        )
        recent_anchors = sorted(
            int(row["anchor_ts_ns"]) for row in detection.anchors
        )
        with np.load(cache.final_path, allow_pickle=False) as data:
            ts = data["ts_ns"].copy()
            segments = data["segment_id"].copy()
            valid = data["valid"].astype(bool)
            x = data["x"].astype(np.float64)
            z = data["z"].astype(np.float64)
            bid_depth = data["bid_depth_current"].astype(np.float64)
            ask_depth = data["ask_depth_current"].astype(np.float64)
            obi = data["obi_current"].astype(np.float64)
            spread = data["spread_ticks"].astype(np.float64)
            activity = data["activity_count"].astype(np.int64)
        plus, minus, plus_score, minus_score, _ = _structural_arrays(x, z)
        eligible = (
            valid
            & np.all(np.isfinite(z), axis=1)
            & np.isfinite(bid_depth)
            & np.isfinite(ask_depth)
            & np.isfinite(obi)
            & np.isfinite(spread)
        )
        quiet = (
            eligible
            & ~plus
            & ~minus
            & (plus_score < RELEASE_SCORE)
            & (minus_score < RELEASE_SCORE)
        )
        candidate_ts = (
            (int(ts[0]) // CONTROL_STRIDE_NS) + 1
        ) * CONTROL_STRIDE_NS
        while candidate_ts <= int(ts[-1]):
            index = int(np.searchsorted(ts, candidate_ts, side="right") - 1)
            if index < 0 or not quiet[index]:
                candidate_ts += CONTROL_STRIDE_NS
                continue
            if _inside_interval(intervals, candidate_ts):
                candidate_ts += CONTROL_STRIDE_NS
                continue
            last_excursion_index = bisect.bisect_right(
                recent_excursions, candidate_ts
            ) - 1
            if (
                last_excursion_index >= 0
                and candidate_ts - recent_excursions[last_excursion_index]
                < CONTROL_HISTORY_EXCLUSION_NS
            ):
                candidate_ts += CONTROL_STRIDE_NS
                continue
            last_anchor_index = bisect.bisect_right(recent_anchors, candidate_ts) - 1
            if (
                last_anchor_index >= 0
                and candidate_ts - recent_anchors[last_anchor_index]
                < CONTROL_HISTORY_EXCLUSION_NS
            ):
                candidate_ts += CONTROL_STRIDE_NS
                continue
            if (
                spread[index] < SAFE_SPREAD_TICKS - 1e-9
                or abs(obi[index]) > MAX_ABS_OBI
                or bid_depth[index] <= depth_floor
                or ask_depth[index] <= depth_floor
            ):
                candidate_ts += CONTROL_STRIDE_NS
                continue
            for direction in (1, -1):
                controls.append(
                    {
                        "control_id": control_id,
                        "capture_id": cache.capture.capture_id,
                        "research_date": cache.capture.research_date,
                        "role": cache.capture.role,
                        "segment_id": int(segments[index]),
                        "ts_ns": int(candidate_ts),
                        "checkpoint_ts_ns": int(ts[index]),
                        "direction": direction,
                        "spread_ticks": float(spread[index]),
                        "obi_current": float(obi[index]),
                        "bid_depth_current": float(bid_depth[index]),
                        "ask_depth_current": float(ask_depth[index]),
                        "total_depth_current": float(
                            bid_depth[index] + ask_depth[index]
                        ),
                        "activity_count": int(activity[index]),
                        "time_block": int(candidate_ts // 1_000_000_000 // 1_800),
                    }
                )
                control_id += 1
            candidate_ts += CONTROL_STRIDE_NS
    return controls


def _quintiles(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if not len(reference):
        return np.zeros(len(values), dtype=np.int8)
    cuts = np.quantile(reference, [0.2, 0.4, 0.6, 0.8])
    return np.searchsorted(cuts, values, side="right").astype(np.int8)


def add_matching_bins(
    anchors: list[dict[str, Any]], controls: list[dict[str, Any]]
) -> None:
    dates = {
        row["research_date"] for row in anchors
    } | {
        row["research_date"] for row in controls
    }
    for date in sorted(dates):
        arows = [row for row in anchors if row["research_date"] == date]
        crows = [row for row in controls if row["research_date"] == date]
        if not crows:
            continue
        references = {
            "bid": np.asarray([row["bid_depth_current"] for row in crows]),
            "ask": np.asarray([row["ask_depth_current"] for row in crows]),
            "activity": np.asarray([row["activity_count"] for row in crows]),
        }
        for rows in (arows, crows):
            bid_q = _quintiles(
                np.asarray([row["bid_depth_current"] for row in rows]),
                references["bid"],
            )
            ask_q = _quintiles(
                np.asarray([row["ask_depth_current"] for row in rows]),
                references["ask"],
            )
            activity_q = _quintiles(
                np.asarray([row["activity_count"] for row in rows]),
                references["activity"],
            )
            for row, bq, aq, actq in zip(rows, bid_q, ask_q, activity_q):
                row["obi_bin"] = int(abs(float(row["obi_current"])) // 0.10)
                row["spread_bin"] = int(round(float(row["spread_ticks"])))
                row["bid_depth_quintile"] = int(bq)
                row["ask_depth_quintile"] = int(aq)
                row["activity_quintile"] = int(actq)
                anchor_ts = int(row.get("anchor_ts_ns", row.get("ts_ns")))
                row["time_block"] = int(anchor_ts // 1_000_000_000 // 1_800)


def match_controls(
    anchors: list[dict[str, Any]], controls: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    add_matching_bins(anchors, controls)
    by_id = {int(row["control_id"]): row for row in controls}
    exact: dict[tuple[Any, ...], list[tuple[int, int]]] = defaultdict(list)
    adjacent: dict[tuple[Any, ...], list[tuple[int, int]]] = defaultdict(list)
    for row in controls:
        key = (
            row["research_date"],
            row["direction"],
            row["spread_bin"],
            row["obi_bin"],
            row["bid_depth_quintile"],
            row["ask_depth_quintile"],
            row["activity_quintile"],
            row["time_block"],
        )
        exact[key].append((int(row["ts_ns"]), int(row["control_id"])))
        adjacent[key[:2] + key[3:]].append(
            (int(row["ts_ns"]), int(row["control_id"]))
        )
    for mapping in (exact, adjacent):
        for values in mapping.values():
            values.sort()

    used: set[int] = set()
    matched: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    ordered = sorted(
        anchors,
        key=lambda row: (
            row["research_date"],
            int(row["anchor_ts_ns"]),
            int(row["direction"]),
        ),
    )
    for anchor_index, anchor in enumerate(ordered):
        if "spread_bin" not in anchor:
            unmatched.append(
                {
                    "anchor_index": anchor_index,
                    "episode_id": anchor["episode_id"],
                    "capture_id": anchor["capture_id"],
                    "research_date": anchor["research_date"],
                    "anchor_ts_ns": anchor["anchor_ts_ns"],
                    "direction": anchor["direction"],
                    "reason": "no_same_date_control_reference",
                }
            )
            continue
        key = (
            anchor["research_date"],
            anchor["direction"],
            anchor["spread_bin"],
            anchor["obi_bin"],
            anchor["bid_depth_quintile"],
            anchor["ask_depth_quintile"],
            anchor["activity_quintile"],
            anchor["time_block"],
        )
        choices = exact.get(key, [])
        relaxation = "exact"
        target = int(anchor["anchor_ts_ns"])
        available = [row for row in choices if row[1] not in used]
        if not available:
            base = key[:2] + key[3:]
            available = [
                row
                for row in adjacent.get(base, [])
                if row[1] not in used
                and abs(int(by_id[row[1]]["spread_bin"]) - int(key[2])) == 1
            ]
            relaxation = "adjacent_spread"
        if not available:
            unmatched.append(
                {
                    "anchor_index": anchor_index,
                    "episode_id": anchor["episode_id"],
                    "capture_id": anchor["capture_id"],
                    "research_date": anchor["research_date"],
                    "anchor_ts_ns": anchor["anchor_ts_ns"],
                    "direction": anchor["direction"],
                    "reason": "no_unused_matched_control",
                }
            )
            continue
        chosen_ts, control_id = min(
            available,
            key=lambda row: (abs(row[0] - target), row[0], row[1]),
        )
        used.add(control_id)
        control = by_id[control_id]
        matched.append(
            {
                "anchor_index": anchor_index,
                "episode_id": anchor["episode_id"],
                "anchor_capture_id": anchor["capture_id"],
                "anchor_ts_ns": anchor["anchor_ts_ns"],
                "control_id": control_id,
                "control_capture_id": control["capture_id"],
                "control_ts_ns": chosen_ts,
                "research_date": anchor["research_date"],
                "direction": anchor["direction"],
                "relaxation": relaxation,
                "time_distance_ms": abs(chosen_ts - target) / 1e6,
            }
        )
    return matched, unmatched


def _inter_episode_ms(episodes: Sequence[dict[str, Any]]) -> np.ndarray:
    output: list[float] = []
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in episodes:
        grouped[row["capture_id"]].append(int(row["confirmation_ts_ns"]))
    for values in grouped.values():
        values.sort()
        output.extend((b - a) / 1e6 for a, b in zip(values, values[1:]))
    return np.asarray(output, dtype=np.float64)


def _coverage_geometry(
    anchors: Sequence[dict[str, Any]],
    cache_by_capture: dict[str, CacheRef],
) -> tuple[dict[int, tuple[float, float]], int | None]:
    output: dict[int, tuple[float, float]] = {}
    for tau_ms in TAU_CANDIDATES_MS:
        complete: list[bool] = []
        by_date: dict[str, list[bool]] = defaultdict(list)
        for row in anchors:
            cache = cache_by_capture[row["capture_id"]]
            end_ts = cache.segment_end_by_id[int(row["segment_id"])]
            value = int(row["anchor_ts_ns"]) + tau_ms * 1_000_000 <= end_ts
            complete.append(value)
            by_date[row["research_date"]].append(value)
        overall = float(np.mean(complete)) if complete else 0.0
        minimum = min(
            (float(np.mean(values)) for values in by_date.values()),
            default=0.0,
        )
        output[tau_ms] = (overall, minimum)
    eligible = [
        tau
        for tau, (overall, minimum) in output.items()
        if overall >= 0.95 and minimum >= 0.80
    ]
    return output, max(eligible) if eligible else None


def classify_a0(
    gates: dict[str, bool],
    *,
    excursion_rate: float,
    median_inter_excursion_ms: float,
    violation_counts: dict[str, int],
    anchor_state_invalid: bool,
) -> str:
    if not gates["A0_0_source_closure"]:
        return "A0_source_not_admissible"
    if not gates["A0_1_zero_outcome_boundary"]:
        return "A0_zero_outcome_boundary_violated"
    if not gates["A0_2_normalization_support"]:
        return "A0_normalization_support_failed"
    if not gates["A0_3_excursion_support"]:
        if excursion_rate > 100 or (
            math.isfinite(median_inter_excursion_ms)
            and median_inter_excursion_ms < 2_000
        ):
            return "A0_excursion_still_near_continuous"
        return "A0_excursion_support_insufficient"
    if not gates["A0_4_novelty_persistence_compression"]:
        if violation_counts.get("novelty", 0):
            return "A0_novelty_contract_failed"
        if violation_counts.get("persistence", 0) or violation_counts.get(
            "backdated", 0
        ):
            return "A0_persistence_contract_failed"
        if violation_counts.get("overlap", 0) or violation_counts.get(
            "new_episode_while_active", 0
        ):
            return "A0_refractory_contract_failed"
        return "A0_excursion_still_near_continuous"
    if not gates["A0_5_safe_reentry_support"]:
        if anchor_state_invalid:
            return "A0_safe_reentry_current_state_invalid"
        return "A0_safe_reentry_support_insufficient"
    if not gates["A0_6_control_common_support"]:
        return "A0_control_common_support_insufficient"
    if not gates["A0_7_followup_geometry"]:
        return "A0_followup_geometry_insufficient"
    return "A0_safe_reentry_contract_supported"


def evaluate_gates(
    captures: Sequence[Capture],
    candidates: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    matched: list[dict[str, Any]],
    global_scale_floors: np.ndarray,
    selected_tau_ms: int | None,
) -> tuple[dict[str, bool], str, dict[str, Any]]:
    duration_hours = sum(row.duration_seconds for row in captures) / 3600
    admitted_dates = {row.research_date for row in captures}
    excursion_by_date = Counter(row["research_date"] for row in episodes)
    anchor_by_date = Counter(row["research_date"] for row in anchors)
    matched_by_date = Counter(row["research_date"] for row in matched)
    direction_counts = Counter(int(row["initial_direction"]) for row in episodes)
    inter_excursion = _inter_episode_ms(episodes)
    raw_crossings = sum(
        int(row["raw_micro_crossing_count"]) for row in diagnostics
    )
    excursion_count = len(episodes)
    anchor_count = len(anchors)
    excursion_rate = excursion_count / max(duration_hours, 1e-12)
    anchor_rate = anchor_count / max(duration_hours, 1e-12)
    median_inter = _safe_percentile(inter_excursion, 50)

    novelty_violations = sum(
        int(row["prequiet_checkpoint_count"]) < NOVELTY_CHECKPOINTS
        for row in episodes
    )
    persistence_violations = sum(
        float(row["qualifying_exposure_ms"])
        < PERSISTENCE_EXPOSURE_NS / 1e6
        or float(row["confirmation_delay_ms"])
        < PERSISTENCE_EXPOSURE_NS / 1e6
        for row in episodes
    )
    backdated = sum(
        int(row["confirmation_ts_ns"]) < int(row["candidate_ts_ns"])
        for row in episodes
    )
    ordered_episodes = sorted(
        episodes,
        key=lambda row: (
            row["capture_id"],
            int(row["candidate_ts_ns"]),
            int(row["terminal_ts_ns"]),
        ),
    )
    overlaps = 0
    prior_by_capture: dict[str, int] = {}
    for row in ordered_episodes:
        capture_id = row["capture_id"]
        if int(row["candidate_ts_ns"]) <= prior_by_capture.get(capture_id, -1):
            overlaps += 1
        prior_by_capture[capture_id] = int(row["terminal_ts_ns"])
    ids = [row["episode_id"] for row in episodes]
    duplicate_ids = len(ids) - len(set(ids))
    new_episode_while_active = overlaps + duplicate_ids
    pair_presence = {
        pair: any(row["confirmation_component_pair"] in {pair, "dep_trade_ofi"}
                  for row in episodes)
        for pair in upstream.PAIR_NAMES
    }
    directions_present = all(direction_counts[value] > 0 for value in (1, -1))

    episode_nonfloor = [
        int(row["confirmation_nonfloor_component_count"]) >= 2
        for row in episodes
    ]
    anchor_nonfloor = [
        int(row["nonfloor_component_count"]) >= 2 for row in anchors
    ]
    episode_nonfloor_share = (
        float(np.mean(episode_nonfloor)) if episode_nonfloor else 0.0
    )
    anchor_nonfloor_share = (
        float(np.mean(anchor_nonfloor)) if anchor_nonfloor else 0.0
    )
    anchor_state_invalid = any(
        float(row["spread_ticks"]) < SAFE_SPREAD_TICKS - 1e-9
        or abs(float(row["obi_current"])) > MAX_ABS_OBI
        or float(row["bid_depth_current"])
        < DEPTH_RECOVERY_FRACTION * float(row["bid_depth_baseline"])
        or float(row["ask_depth_current"])
        < DEPTH_RECOVERY_FRACTION * float(row["ask_depth_baseline"])
        for row in anchors
    )
    common_support_by_date = {
        date: matched_by_date[date] / max(anchor_by_date[date], 1)
        for date in anchor_by_date
    }
    represented_excursion_dates = {
        date for date, count in excursion_by_date.items() if count > 0
    }
    represented_anchor_dates = {
        date for date, count in anchor_by_date.items() if count > 0
    }
    violation_counts = {
        "novelty": novelty_violations,
        "persistence": persistence_violations,
        "backdated": backdated,
        "overlap": overlaps,
        "new_episode_while_active": new_episode_while_active,
    }

    gates = {
        "A0_0_source_closure": bool(
            len(captures) == 29
            and len(admitted_dates) == 9
            and all(row.depth_gap_count == 0 for row in captures)
        ),
        "A0_1_zero_outcome_boundary": True,
        "A0_2_normalization_support": bool(
            np.all(np.isfinite(global_scale_floors))
            and np.all(global_scale_floors > 0)
            and episode_nonfloor_share >= 0.95
            and anchor_nonfloor_share >= 0.95
        ),
        "A0_3_excursion_support": bool(
            excursion_count >= 300
            and len(represented_excursion_dates) >= 8
            and all(
                excursion_by_date[date] >= 15
                for date in represented_excursion_dates
            )
            and 2 <= excursion_rate <= 100
            and max(excursion_by_date.values(), default=excursion_count)
            / max(excursion_count, 1)
            <= 0.35
            and min(direction_counts.values(), default=0)
            / max(excursion_count, 1)
            >= 0.20
        ),
        "A0_4_novelty_persistence_compression": bool(
            novelty_violations == 0
            and persistence_violations == 0
            and backdated == 0
            and overlaps == 0
            and new_episode_while_active == 0
            and raw_crossings / max(excursion_count, 1) >= 5
            and median_inter >= 2_000
            and directions_present
            and all(pair_presence.values())
        ),
        "A0_5_safe_reentry_support": bool(
            anchor_count >= 200
            and len(represented_anchor_dates) >= 8
            and all(
                anchor_by_date[date] >= 10 for date in represented_anchor_dates
            )
            and 1 <= anchor_rate <= 50
            and max(anchor_by_date.values(), default=anchor_count)
            / max(anchor_count, 1)
            <= 0.35
            and not anchor_state_invalid
        ),
        "A0_6_control_common_support": bool(
            len(matched) >= 200
            and len(matched) / max(anchor_count, 1) >= 0.90
            and min(common_support_by_date.values(), default=0.0) >= 0.75
            and max(matched_by_date.values(), default=len(matched))
            / max(len(matched), 1)
            <= 0.35
            and len({int(row["control_id"]) for row in matched}) == len(matched)
        ),
        "A0_7_followup_geometry": selected_tau_ms is not None,
    }
    classification = classify_a0(
        gates,
        excursion_rate=excursion_rate,
        median_inter_excursion_ms=median_inter,
        violation_counts=violation_counts,
        anchor_state_invalid=anchor_state_invalid,
    )
    metrics = {
        "duration_hours": duration_hours,
        "raw_micro_crossing_count": raw_crossings,
        "candidate_count": len(candidates),
        "excursion_count": excursion_count,
        "excursion_rate_per_hour": excursion_rate,
        "anchor_count": anchor_count,
        "anchor_rate_per_hour": anchor_rate,
        "crossing_to_excursion_ratio": raw_crossings / max(excursion_count, 1),
        "median_inter_excursion_ms": median_inter,
        "direction_counts": dict(direction_counts),
        "pair_presence": pair_presence,
        "excursion_nonfloor_ge2_share": episode_nonfloor_share,
        "safe_reentry_nonfloor_ge2_share": anchor_nonfloor_share,
        "violation_counts": violation_counts,
        "anchor_state_invalid": anchor_state_invalid,
        "common_support": len(matched) / max(anchor_count, 1),
        "minimum_date_common_support": min(
            common_support_by_date.values(), default=0.0
        ),
    }
    return gates, classification, metrics


def _write_source_and_cache_manifests(
    captures: Sequence[Capture],
    caches: Sequence[CacheRef],
    out_dir: Path,
    *,
    verify_hashes: bool,
) -> dict[str, Any]:
    source_rows: list[dict[str, Any]] = []
    for capture in captures:
        if not capture.raw_path.is_file():
            raise A0Error(f"raw_missing:{capture.raw_path}")
        size_ok = capture.raw_path.stat().st_size == capture.raw_size_bytes
        if not size_ok:
            raise A0Error(f"raw_size_mismatch:{capture.capture_id}")
        hash_ok = True
        if verify_hashes:
            hash_ok = _sha256(capture.raw_path) == capture.raw_sha256
        if not hash_ok:
            raise A0Error(f"raw_hash_mismatch:{capture.capture_id}")
        source_rows.append(
            {
                "capture_id": capture.capture_id,
                "research_date": capture.research_date,
                "role": capture.role,
                "start_utc": capture.start_utc,
                "end_utc": capture.end_utc,
                "duration_seconds": capture.duration_seconds,
                "raw_path": str(capture.raw_path),
                "raw_size_bytes": capture.raw_size_bytes,
                "raw_sha256": capture.raw_sha256,
                "depth_gap_count": capture.depth_gap_count,
                "raw_hash_verified": str(hash_ok).lower(),
            }
        )
    _write_csv(
        out_dir / "support/source_inventory.csv",
        source_rows,
        list(source_rows[0]),
    )

    cache_rows = []
    for cache in caches:
        cache_rows.append(
            {
                "capture_id": cache.capture.capture_id,
                "final_cache_path": str(cache.final_path),
                "final_cache_size_bytes": cache.final_path.stat().st_size,
                "final_cache_sha256": _sha256(cache.final_path),
                "event_key_cache_path": str(cache.event_key_path),
                "event_key_cache_size_bytes": cache.event_key_path.stat().st_size,
                "event_key_cache_sha256": _sha256(cache.event_key_path),
            }
        )
    _write_csv(
        out_dir / "contracts/upstream_cache_manifest.csv",
        cache_rows,
        list(cache_rows[0]),
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "contract_sha256": CONTRACT_SHA256,
        "capture_count": len(captures),
        "research_dates": sorted({row.research_date for row in captures}),
        "duration_hours": sum(row.duration_seconds for row in captures) / 3600,
        "raw_hashes_verified_now": verify_hashes,
        "upstream_task_id": "0828T008",
        "upstream_qa_commit": "ae3048a3",
        "upstream_normalization_contract_sha256": (
            UPSTREAM_NORMALIZATION_SHA256
        ),
        "upstream_pressure_contract_sha256": UPSTREAM_PRESSURE_SHA256,
        "future_target_fields_read": [],
    }
    _write_json(out_dir / "contracts/source_manifest.json", payload)
    return payload


def _write_contracts(
    out_dir: Path,
    denominator_floors: dict[str, float],
    global_scale_floors: np.ndarray,
    gates: dict[str, bool],
) -> None:
    contracts: dict[str, dict[str, Any]] = {
        "event_ordering_contract.json": {
            "event_key": ["checkpoint_local_receive_ts_ns", "event_seq_in_file"],
            "checkpoint_before_same_timestamp_message": True,
            "event_seq_meaning": "last_file_event_causally_visible_at_checkpoint",
            "confirmation_backdating": False,
        },
        "micro_pressure_contract.json": {
            "upstream_hypothesis": "LIQUIDITY_BREAK_ONSET_V1",
            "window_ms": 50,
            "levels": 5,
            "weights": upstream.LEVEL_WEIGHTS.tolist(),
            "components": [
                "vulnerable_net_depletion",
                "aggressive_trade_pressure",
                "whole_book_flow_pressure",
            ],
            "z_star": upstream.Z_STAR,
            "pressure_star": upstream.PRESSURE_STAR,
            "coherence": "at_least_two_of_three",
            "role": "micro_observation_not_anchor",
        },
        "normalization_contract.json": {
            "upstream_task_id": "0828T008",
            "checkpoint_ms": 20,
            "history_ms": 60_000,
            "guard_ms": 500,
            "center": "rolling_median",
            "scale": "rolling_IQR_div_1.349",
            "calibration_only": True,
            "denominator_floors": denominator_floors,
            "global_scale_floors": dict(
                zip(upstream.COMPONENT_NAMES, global_scale_floors.tolist())
            ),
        },
        "novelty_contract.json": {
            "prequiet_ms": 1_000,
            "required_checkpoints": int(NOVELTY_CHECKPOINTS),
            "bilateral": True,
            "micro_predicate_false": True,
            "both_pressure_scores_below": RELEASE_SCORE,
            "reset_invalidates_history": True,
        },
        "persistence_contract.json": {
            "qualification_window_ms": 250,
            "required_exposure_ms": 100,
            "checkpoint_ms": 20,
            "candidate_checkpoint_counts": True,
            "backdating": False,
            "opposite_dominant_preconfirmation": "reject_candidate",
        },
        "excursion_state_machine.json": {
            "states": [
                BACKGROUND_BUILDING,
                BACKGROUND_READY,
                EXCURSION_CANDIDATE,
                EXCURSION_ACTIVE,
                RECOVERY_CANDIDATE,
                REFRACTORY,
                "TERMINAL",
            ],
            "opposite_crossing_while_active": "same_episode_direction_switch",
            "new_episode_while_active": False,
            "baseline_window": "[candidate-1000ms,candidate-100ms)",
            "baseline_aggregation": "componentwise_median",
        },
        "recovery_refractory_contract.json": {
            "bilateral_depth_recovery_fraction": DEPTH_RECOVERY_FRACTION,
            "pressure_score_below": RELEASE_SCORE,
            "refractory_ms": 1_000,
            "renewed_pressure": "return_same_episode_to_active",
            "depth_lapse": "return_same_episode_to_recovery_candidate",
        },
        "safe_reentry_contract.json": {
            "anchor": "refractory_completed_at",
            "current_spread_ticks_min": SAFE_SPREAD_TICKS,
            "current_abs_obi_max": MAX_ABS_OBI,
            "bilateral_depth_recovery_fraction": DEPTH_RECOVERY_FRACTION,
            "wait_for_future_opportunity": False,
        },
        "control_support_contract.json": {
            "stride_ms": 250,
            "recent_excursion_exclusion_ms": 5_000,
            "current_spread_ticks_min": SAFE_SPREAD_TICKS,
            "current_abs_obi_max": MAX_ABS_OBI,
            "reuse": False,
            "match": [
                "same_date",
                "same_direction",
                "same_spread_ticks",
                "same_abs_obi_bin_0.10",
                "same_bid_depth_quintile",
                "same_ask_depth_quintile",
                "same_activity_quintile",
                "same_30_minute_block",
            ],
            "only_relaxation": "adjacent_spread_tick",
            "future_outcome_exclusion": False,
        },
        "downstream_target_stub.json": {
            "materialized": False,
            "causes": [
                "n_contact",
                "n_adverse",
                "n_spread_collapse",
                "n_timeout",
            ],
            "queue_fill_bound_materialized": False,
            "public_contact_is_real_fill": False,
            "tau_candidates_ms": list(TAU_CANDIDATES_MS),
        },
        "H0_H1_contract.json": {
            "H0": "current_static_safe_state",
            "H1_adds": "recent_completed_excursion_path",
            "fitted_in_A0": False,
        },
        "gate_contract.json": {
            "gates": gates,
            "thresholds": {
                "minimum_excursions": 300,
                "minimum_excursion_dates": 8,
                "minimum_excursions_per_represented_date": 15,
                "excursion_rate_per_hour": [2, 100],
                "maximum_excursion_date_share": 0.35,
                "minority_direction_share": 0.20,
                "minimum_crossing_to_excursion_ratio": 5,
                "minimum_median_inter_excursion_ms": 2_000,
                "minimum_safe_reentry_anchors": 200,
                "minimum_safe_reentry_dates": 8,
                "minimum_anchors_per_represented_date": 10,
                "safe_reentry_rate_per_hour": [1, 50],
                "minimum_matched_pairs": 200,
                "minimum_common_support": 0.90,
                "minimum_per_date_common_support": 0.75,
            },
        },
        "outcome_access_ledger.json": {
            "future_midpoint_fields_read": [],
            "future_best_price_fields_read": [],
            "future_contact_fields_read": [],
            "queue_fill_targets_materialized": False,
            "markout_PnL_fields_read": [],
            "H0_H1_fitted": False,
            "new_collection": False,
            "private_order_access": False,
        },
    }
    for name, payload in contracts.items():
        _write_json(
            out_dir / "contracts" / name,
            {"schema_version": SCHEMA_VERSION, **payload},
        )


def _write_rows(
    path: Path,
    rows: list[dict[str, Any]],
    fallback_fields: list[str],
) -> None:
    fields = list(rows[0]) if rows else fallback_fields
    _write_csv(path, rows, fields)


def _manifest(out_dir: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file() or "/cache/" in f"/{path.relative_to(out_dir)}/":
            continue
        relative = str(path.relative_to(out_dir))
        if relative == "run_manifest.json":
            continue
        artifacts.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    _write_json(out_dir / "run_manifest.json", payload)
    return payload


def run_a0(
    *,
    bindings: Path,
    upstream_dir: Path,
    out_dir: Path,
    event_key_cache_dir: Path | None,
    verify_hashes: bool,
) -> dict[str, Any]:
    if not CONTRACT_PATH.is_file() or _sha256(CONTRACT_PATH) != CONTRACT_SHA256:
        raise A0Error("contract_sha_mismatch")
    normalization_contract = upstream_dir / "contracts/normalization_contract.json"
    pressure_contract = upstream_dir / "contracts/pressure_component_contract.json"
    if _sha256(normalization_contract) != UPSTREAM_NORMALIZATION_SHA256:
        raise A0Error("upstream_normalization_contract_sha_mismatch")
    if _sha256(pressure_contract) != UPSTREAM_PRESSURE_SHA256:
        raise A0Error("upstream_pressure_contract_sha_mismatch")
    with normalization_contract.open(encoding="utf-8") as handle:
        normalization_payload = json.load(handle)
    with pressure_contract.open(encoding="utf-8") as handle:
        pressure_payload = json.load(handle)
    denominator_floors = {
        key: float(value)
        for key, value in pressure_payload["denominator_floors"].items()
    }
    global_scale_floors = np.asarray(
        [
            normalization_payload["global_scale_floors"][name]
            for name in upstream.COMPONENT_NAMES
        ],
        dtype=np.float64,
    )

    captures = discover_captures(bindings)
    if len(captures) != 29:
        raise A0Error(f"capture_count_mismatch:{len(captures)}")
    key_cache_dir = event_key_cache_dir or out_dir / "cache/event_key"
    caches: list[CacheRef] = []
    for index, capture in enumerate(captures, start=1):
        final_path = upstream_dir / "cache/final" / f"{capture.capture_id}.npz"
        if not final_path.is_file():
            raise A0Error(f"upstream_final_cache_missing:{capture.capture_id}")
        event_key_path = build_event_key_cache(
            capture, final_path, key_cache_dir
        )
        with np.load(final_path, allow_pickle=False) as data:
            segment_end_by_id = {
                int(key): int(value)
                for key, value in zip(
                    data["segment_end_ids"], data["segment_end_ts"]
                )
            }
            caches.append(
                CacheRef(
                    capture=capture,
                    final_path=final_path,
                    event_key_path=event_key_path,
                    segment_end_by_id=segment_end_by_id,
                    first_ts_ns=int(data["ts_ns"][0]),
                    last_ts_ns=int(data["ts_ns"][-1]),
                )
            )
        _progress(
            f"bound cache {index}/{len(captures)} {capture.capture_id}"
        )

    source = _write_source_and_cache_manifests(
        captures, caches, out_dir, verify_hashes=verify_hashes
    )

    detections: dict[str, CaptureDetection] = {}
    candidates: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    anchors: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for index, cache in enumerate(caches, start=1):
        with np.load(cache.final_path, allow_pickle=False) as data:
            arrays = {name: data[name].copy() for name in data.files}
        with np.load(cache.event_key_path, allow_pickle=False) as keys:
            event_seq = keys["event_seq"].copy()
        detection = run_state_machine(
            cache.capture, arrays, event_seq, global_scale_floors
        )
        detections[cache.capture.capture_id] = detection
        candidates.extend(detection.candidates)
        episodes.extend(detection.episodes)
        anchors.extend(detection.anchors)
        diagnostics.append(detection.diagnostics)
        _progress(
            f"detected {index}/{len(caches)} {cache.capture.capture_id}: "
            f"{len(detection.episodes)} excursions, "
            f"{len(detection.anchors)} safe anchors"
        )

    controls = build_controls(
        caches, detections, denominator_floors["depth_scale_floor"]
    )
    _progress(f"built {len(controls)} outcome-blind controls")
    matched, unmatched = match_controls(anchors, controls)
    _progress(f"matched {len(matched)}/{len(anchors)} safe anchors")
    cache_by_capture = {row.capture.capture_id: row for row in caches}
    coverage, selected_tau_ms = _coverage_geometry(anchors, cache_by_capture)
    gates, classification, metrics = evaluate_gates(
        captures,
        candidates,
        episodes,
        anchors,
        diagnostics,
        matched,
        global_scale_floors,
        selected_tau_ms,
    )
    _write_contracts(
        out_dir, denominator_floors, global_scale_floors, gates
    )

    session_rows = []
    grouped_sessions: dict[tuple[str, str], list[Capture]] = defaultdict(list)
    for capture in captures:
        grouped_sessions[(capture.research_date, capture.role)].append(capture)
    for (date, role), rows in sorted(grouped_sessions.items()):
        session_rows.append(
            {
                "research_date": date,
                "role": role,
                "capture_count": len(rows),
                "duration_hours": sum(row.duration_seconds for row in rows)
                / 3600,
                "capture_ids": "|".join(row.capture_id for row in rows),
            }
        )
    _write_csv(
        out_dir / "contracts/session_role_ledger.csv",
        session_rows,
        list(session_rows[0]),
    )

    _write_rows(
        out_dir / "support/candidate_ledger.csv",
        candidates,
        ["capture_id", "candidate_ts_ns", "candidate_status"],
    )
    _write_rows(
        out_dir / "support/excursion_ledger.csv",
        episodes,
        ["episode_id", "capture_id", "confirmation_ts_ns", "terminal_status"],
    )
    _write_rows(
        out_dir / "support/safe_reentry_anchor_ledger.csv",
        anchors,
        ["episode_id", "capture_id", "anchor_ts_ns"],
    )
    _write_rows(
        out_dir / "support/control_candidates.csv",
        controls,
        ["control_id", "capture_id", "ts_ns"],
    )
    _write_rows(
        out_dir / "support/matched_control_pairs.csv",
        matched,
        ["anchor_index", "control_id"],
    )
    _write_rows(
        out_dir / "support/unmatched_safe_reentry_anchors.csv",
        unmatched,
        ["anchor_index", "episode_id"],
    )
    _write_rows(
        out_dir / "support/capture_diagnostics.csv",
        diagnostics,
        ["capture_id", "checkpoint_count"],
    )

    duration_by_date: dict[str, float] = defaultdict(float)
    for capture in captures:
        duration_by_date[capture.research_date] += capture.duration_seconds
    candidate_by_date = Counter(row["research_date"] for row in candidates)
    confirmed_by_date = Counter(
        row["research_date"]
        for row in candidates
        if row["candidate_status"] == "confirmed"
    )
    transient_by_date_status = Counter(
        (row["research_date"], row["candidate_status"])
        for row in candidates
        if row["candidate_status"] != "confirmed"
    )
    episode_by_date = Counter(row["research_date"] for row in episodes)
    anchor_by_date = Counter(row["research_date"] for row in anchors)
    matched_by_date = Counter(row["research_date"] for row in matched)
    micro_by_date = Counter()
    structural_by_date = Counter()
    for row in diagnostics:
        micro_by_date[row["research_date"]] += int(
            row["raw_micro_crossing_count"]
        )
        structural_by_date[row["research_date"]] += int(
            row["structural_checkpoint_count"]
        )

    micro_rows = []
    candidate_rows = []
    excursion_rows = []
    anchor_rows = []
    compression_rows = []
    control_overlap_rows = []
    for date in sorted(duration_by_date):
        hours = duration_by_date[date] / 3600
        micro_rows.append(
            {
                "research_date": date,
                "raw_micro_crossing_count": micro_by_date[date],
                "structural_checkpoint_count": structural_by_date[date],
                "duration_hours": hours,
                "raw_micro_crossing_rate_per_hour": micro_by_date[date]
                / max(hours, 1e-12),
            }
        )
        candidate_rows.append(
            {
                "research_date": date,
                "candidate_count": candidate_by_date[date],
                "confirmed_candidate_count": confirmed_by_date[date],
                "confirmation_share": confirmed_by_date[date]
                / max(candidate_by_date[date], 1),
            }
        )
        excursion_rows.append(
            {
                "research_date": date,
                "excursion_count": episode_by_date[date],
                "duration_hours": hours,
                "excursion_rate_per_hour": episode_by_date[date]
                / max(hours, 1e-12),
            }
        )
        anchor_rows.append(
            {
                "research_date": date,
                "safe_reentry_anchor_count": anchor_by_date[date],
                "duration_hours": hours,
                "safe_reentry_rate_per_hour": anchor_by_date[date]
                / max(hours, 1e-12),
                "matched_count": matched_by_date[date],
                "common_support": matched_by_date[date]
                / max(anchor_by_date[date], 1),
            }
        )
        compression_rows.append(
            {
                "research_date": date,
                "raw_micro_crossing_count": micro_by_date[date],
                "confirmed_excursion_count": episode_by_date[date],
                "crossing_to_excursion_ratio": micro_by_date[date]
                / max(episode_by_date[date], 1),
            }
        )
        control_overlap_rows.append(
            {
                "research_date": date,
                "anchor_count": anchor_by_date[date],
                "matched_count": matched_by_date[date],
                "common_support": matched_by_date[date]
                / max(anchor_by_date[date], 1),
            }
        )
    _write_csv(
        out_dir / "support/micro_crossing_support_by_date.csv",
        micro_rows,
        list(micro_rows[0]),
    )
    _write_csv(
        out_dir / "support/candidate_support_by_date.csv",
        candidate_rows,
        list(candidate_rows[0]),
    )
    transient_rows = [
        {
            "research_date": date,
            "candidate_status": status,
            "count": count,
        }
        for (date, status), count in sorted(transient_by_date_status.items())
    ]
    _write_rows(
        out_dir / "support/transient_rejection_support.csv",
        transient_rows,
        ["research_date", "candidate_status", "count"],
    )
    _write_csv(
        out_dir / "support/excursion_support_by_date.csv",
        excursion_rows,
        list(excursion_rows[0]),
    )
    _write_csv(
        out_dir / "support/safe_reentry_support_by_date.csv",
        anchor_rows,
        list(anchor_rows[0]),
    )
    _write_csv(
        out_dir / "support/crossing_to_episode_compression.csv",
        compression_rows,
        list(compression_rows[0]),
    )
    _write_csv(
        out_dir / "support/control_overlap_by_date.csv",
        control_overlap_rows,
        list(control_overlap_rows[0]),
    )

    duration_values = np.asarray(
        [float(row["episode_duration_ms"]) for row in episodes]
    )
    confirmed_duration_values = np.asarray(
        [float(row["confirmed_duration_ms"]) for row in episodes]
    )
    duration_rows = [
        {
            "quantile": label,
            "episode_duration_ms": _safe_percentile(duration_values, q),
            "confirmed_duration_ms": _safe_percentile(
                confirmed_duration_values, q
            ),
        }
        for label, q in (("p10", 10), ("p50", 50), ("p90", 90), ("p99", 99))
    ]
    _write_csv(
        out_dir / "support/excursion_duration_distribution.csv",
        duration_rows,
        list(duration_rows[0]),
    )
    direction_switch_counts = Counter(
        int(row["direction_switch_count"]) for row in episodes
    )
    direction_switch_rows = [
        {
            "direction_switch_count": value,
            "episode_count": count,
            "episode_share": count / max(len(episodes), 1),
        }
        for value, count in sorted(direction_switch_counts.items())
    ]
    _write_rows(
        out_dir / "support/direction_switch_composition.csv",
        direction_switch_rows,
        ["direction_switch_count", "episode_count", "episode_share"],
    )
    reset_counts = Counter(
        (
            int(row["recovery_reset_count"]),
            int(row["refractory_reset_count"]),
        )
        for row in episodes
    )
    reset_rows = [
        {
            "recovery_reset_count": recovery,
            "refractory_reset_count": refractory,
            "episode_count": count,
            "episode_share": count / max(len(episodes), 1),
        }
        for (recovery, refractory), count in sorted(reset_counts.items())
    ]
    _write_rows(
        out_dir / "support/recovery_reset_composition.csv",
        reset_rows,
        [
            "recovery_reset_count",
            "refractory_reset_count",
            "episode_count",
            "episode_share",
        ],
    )
    terminal_counts = Counter(row["terminal_status"] for row in episodes)
    terminal_rows = [
        {
            "terminal_status": status,
            "episode_count": count,
            "episode_share": count / max(len(episodes), 1),
        }
        for status, count in sorted(terminal_counts.items())
    ]
    _write_rows(
        out_dir / "support/terminal_episode_composition.csv",
        terminal_rows,
        ["terminal_status", "episode_count", "episode_share"],
    )
    refractory_rows = []
    for date in sorted(duration_by_date):
        subset = [row for row in episodes if row["research_date"] == date]
        completed = [
            row
            for row in subset
            if row["terminal_status"]
            in {
                "safe_reentry_available",
                "recovered_without_wide_spread",
                "recovered_but_imbalanced",
            }
        ]
        refractory_rows.append(
            {
                "research_date": date,
                "episode_count": len(subset),
                "refractory_completed_count": len(completed),
                "refractory_completion_share": len(completed)
                / max(len(subset), 1),
            }
        )
    _write_csv(
        out_dir / "support/refractory_completion_support.csv",
        refractory_rows,
        list(refractory_rows[0]),
    )
    spread_values = np.asarray(
        [float(row["spread_ticks"]) for row in anchors]
    )
    spread_rows = [
        {
            "quantile": label,
            "spread_ticks": _safe_percentile(spread_values, q),
        }
        for label, q in (("p10", 10), ("p50", 50), ("p90", 90), ("p99", 99))
    ]
    _write_csv(
        out_dir / "support/safe_reentry_spread_distribution.csv",
        spread_rows,
        list(spread_rows[0]),
    )
    coverage_rows = [
        {
            "tau_ms": tau,
            "overall_complete_fraction": values[0],
            "minimum_date_complete_fraction": values[1],
            "selected": str(tau == selected_tau_ms).lower(),
        }
        for tau, values in coverage.items()
    ]
    _write_csv(
        out_dir / "support/followup_geometry.csv",
        coverage_rows,
        list(coverage_rows[0]),
    )

    terminal_counts = Counter(row["terminal_status"] for row in episodes)
    candidate_status_counts = Counter(
        row["candidate_status"] for row in candidates
    )
    direction_switch_values = np.asarray(
        [int(row["direction_switch_count"]) for row in episodes]
    )
    recovery_reset_values = np.asarray(
        [int(row["recovery_reset_count"]) for row in episodes]
    )
    refractory_reset_values = np.asarray(
        [int(row["refractory_reset_count"]) for row in episodes]
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "passed" if all(gates.values()) else "failed",
        "classification": classification,
        "capture_count": len(captures),
        "research_date_count": len({row.research_date for row in captures}),
        **metrics,
        "candidate_status_counts": dict(candidate_status_counts),
        "terminal_status_counts": dict(terminal_counts),
        "episode_duration_ms_p50": _safe_percentile(duration_values, 50),
        "episode_duration_ms_p90": _safe_percentile(duration_values, 90),
        "direction_switch_count_p50": _safe_percentile(
            direction_switch_values, 50
        ),
        "direction_switch_count_p90": _safe_percentile(
            direction_switch_values, 90
        ),
        "recovery_reset_count_p50": _safe_percentile(
            recovery_reset_values, 50
        ),
        "refractory_reset_count_p50": _safe_percentile(
            refractory_reset_values, 50
        ),
        "control_candidate_count": len(controls),
        "matched_pair_count": len(matched),
        "selected_tau_max_ms": selected_tau_ms,
        "denominator_floors": denominator_floors,
        "global_scale_floors": dict(
            zip(upstream.COMPONENT_NAMES, global_scale_floors.tolist())
        ),
        "gates": gates,
        "A1_authorized": (
            classification == "A0_safe_reentry_contract_supported"
        ),
        "future_target_fields_read": [],
    }
    _write_json(out_dir / "reports/A0_summary.json", summary)
    _write_json(
        out_dir / "classification.json",
        {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "status": summary["status"],
            "classification": classification,
            "gates": gates,
            "A1_authorized": summary["A1_authorized"],
            "future_target_fields_read": [],
        },
    )
    _manifest(out_dir)
    _progress(
        f"classification={classification}, excursions={len(episodes)}, "
        f"anchors={len(anchors)}"
    )
    del source
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings", type=Path, default=DEFAULT_BINDINGS)
    parser.add_argument("--upstream-dir", type=Path, default=UPSTREAM_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--event-key-cache-dir", type=Path)
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()
    summary = run_a0(
        bindings=args.bindings,
        upstream_dir=args.upstream_dir,
        out_dir=args.out_dir,
        event_key_cache_dir=args.event_key_cache_dir,
        verify_hashes=args.verify_hashes,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
