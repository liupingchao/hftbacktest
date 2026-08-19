#!/usr/bin/env python3
"""Pure structural trigger-density helpers for task 0815T001 substage A."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


TRIGGER_AUDIT_SCHEMA = (
    "campaign_id",
    "segment_id",
    "profile_id",
    "candidate_seq",
    "aggressor_side",
    "direction_sign",
    "burst_start_ts_ns",
    "burst_end_ts_ns",
    "burst_duration_ms",
    "burst_trade_count",
    "burst_trade_qty",
    "touch_trade_qty",
    "touch_trade_qty_at_shock",
    "touch_trade_qty_through_decision",
    "post_decision_burst_trade_count",
    "pre_state_ts_ns",
    "pre_best_px",
    "pre_best_qty",
    "shock_ts_ns",
    "impact_ratio",
    "shock_impact_ratio",
    "decision_ts_ns",
    "confirmation_lag_ms",
    "confirmed_best_px",
    "confirmed_best_qty",
    "price_level_depleted",
    "queue_drop_ratio",
    "confirmed_removed_qty",
    "trade_explained_ratio",
    "attribution",
    "pre_hl_bbo_ts_ns",
    "pre_hl_bbo_age_ms",
    "pre_hl_fast_source_ts_ns",
    "pre_hl_fast_age_ms",
    "primary_episode",
    "rejection_reason",
)
FAMILY_A = "family_a_all_candidates"
FAMILY_B = "family_b_confirmed_only"
EPISODE_MERGING_VERSION = "episode_merging_v1"
SENSITIVITY_VERSION = "trigger_density_sensitivity_v1"
WINDOW_MS = 2000
TIME_BLOCK_SECONDS = 60
ESS_MAX_LAG_SECONDS = 60
OUTCOME_LIKE_FIELD_TOKENS = (
    "response",
    "outcome",
    "markout",
    "pnl",
    "brier",
    "crps",
    "adverse",
    "favorable",
    "score",
)
QUANTILES = (0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99)


class DensityContractError(RuntimeError):
    """Raised when trigger-density structural contracts fail closed."""


@dataclass(frozen=True)
class TriggerCandidate:
    session_id: str
    campaign_id: str
    segment_id: str
    profile_id: str
    candidate_seq: int
    candidate_id: str
    aggressor_side: str
    direction_sign: int
    shock_ts_ns: int
    decision_ts_ns: int | None
    impact_ratio: float
    primary_episode: bool
    rejection_reason: str


@dataclass(frozen=True)
class StructuralSpan:
    session_id: str
    segment_id: str
    start_ts_ns: int
    end_ts_ns: int

    def __post_init__(self) -> None:
        if self.end_ts_ns <= self.start_ts_ns:
            raise DensityContractError(
                f"structural span must have positive duration: {self.segment_id}"
            )


def _normalize_bool_text(value: str, field: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise DensityContractError(f"{field} must be literal true/false")


def _int_text(value: Any, field: str) -> int:
    text = str(value).strip()
    if not text:
        raise DensityContractError(f"{field} must be present")
    try:
        return int(text)
    except ValueError as exc:
        raise DensityContractError(f"{field} must be an integer") from exc


def _optional_int_text(value: Any, field: str) -> int | None:
    text = str(value).strip()
    if not text:
        return None
    return _int_text(text, field)


def _float_text(value: Any, field: str) -> float:
    text = str(value).strip()
    if not text:
        raise DensityContractError(f"{field} must be present")
    try:
        return float(text)
    except ValueError as exc:
        raise DensityContractError(f"{field} must be numeric") from exc


def _validate_field_names(field_names: Sequence[str]) -> None:
    if tuple(field_names) != TRIGGER_AUDIT_SCHEMA:
        extra = [name for name in field_names if name not in TRIGGER_AUDIT_SCHEMA]
        missing = [name for name in TRIGGER_AUDIT_SCHEMA if name not in field_names]
        if extra:
            lowered = [name.lower() for name in extra]
            if any(
                token in name
                for name in lowered
                for token in OUTCOME_LIKE_FIELD_TOKENS
            ):
                raise DensityContractError(
                    f"outcome-like fields are forbidden in trigger audit: {extra}"
                )
        raise DensityContractError(
            f"trigger audit schema drift: missing={missing} extra={extra}"
        )


def parse_trigger_audit_rows(
    session_id: str, rows: Sequence[Mapping[str, Any]]
) -> list[TriggerCandidate]:
    if not rows:
        raise DensityContractError("trigger audit rows must not be empty")
    _validate_field_names(tuple(rows[0].keys()))
    by_segment_expected: dict[str, int] = {}
    closed_segments: set[str] = set()
    seen_candidate_ids: set[str] = set()
    last_shock_by_segment: dict[str, int] = {}
    parsed: list[TriggerCandidate] = []
    current_segment: str | None = None
    for row in rows:
        if tuple(row.keys()) != TRIGGER_AUDIT_SCHEMA:
            raise DensityContractError("trigger audit row order or schema drift")
        segment_id = str(row["segment_id"])
        if segment_id in closed_segments:
            raise DensityContractError(f"segment rows are reordered: {segment_id}")
        if current_segment is None:
            current_segment = segment_id
        elif segment_id != current_segment:
            closed_segments.add(current_segment)
            current_segment = segment_id
        candidate_seq = _int_text(row["candidate_seq"], "candidate_seq")
        expected_seq = by_segment_expected.get(segment_id, 1)
        if candidate_seq != expected_seq:
            if candidate_seq < expected_seq:
                raise DensityContractError(
                    f"duplicate or reordered candidate_seq for {segment_id}"
                )
            raise DensityContractError(
                f"missing candidate_seq for {segment_id}: expected {expected_seq}"
            )
        by_segment_expected[segment_id] = expected_seq + 1
        candidate_id = f"{session_id}:{segment_id}:{candidate_seq}"
        if candidate_id in seen_candidate_ids:
            raise DensityContractError(f"duplicate candidate_id: {candidate_id}")
        seen_candidate_ids.add(candidate_id)
        aggressor_side = str(row["aggressor_side"])
        direction_sign = _int_text(row["direction_sign"], "direction_sign")
        if aggressor_side not in {"buy", "sell"}:
            raise DensityContractError("aggressor_side must be buy or sell")
        if direction_sign not in {-1, 1}:
            raise DensityContractError("direction_sign must be -1 or 1")
        if (aggressor_side == "buy" and direction_sign != 1) or (
            aggressor_side == "sell" and direction_sign != -1
        ):
            raise DensityContractError("aggressor_side / direction_sign drift")
        shock_ts_ns = _int_text(row["shock_ts_ns"], "shock_ts_ns")
        previous_shock_ts = last_shock_by_segment.get(segment_id)
        if previous_shock_ts is not None and shock_ts_ns < previous_shock_ts:
            raise DensityContractError(
                f"reordered shock_ts_ns within segment {segment_id}"
            )
        last_shock_by_segment[segment_id] = shock_ts_ns
        primary_episode = _normalize_bool_text(
            str(row["primary_episode"]).strip(), "primary_episode"
        )
        decision_ts_ns = _optional_int_text(row["decision_ts_ns"], "decision_ts_ns")
        rejection_reason = str(row["rejection_reason"]).strip()
        if primary_episode and (decision_ts_ns is None or decision_ts_ns <= shock_ts_ns):
            raise DensityContractError(
                f"confirmed candidate requires valid decision_ts_ns: {candidate_id}"
            )
        if primary_episode and rejection_reason:
            raise DensityContractError(
                f"confirmed candidate cannot carry rejection_reason: {candidate_id}"
            )
        if (not primary_episode) and not rejection_reason:
            raise DensityContractError(
                f"rejected candidate requires rejection_reason: {candidate_id}"
            )
        if decision_ts_ns is not None and decision_ts_ns <= 0:
            raise DensityContractError(f"decision_ts_ns must be positive: {candidate_id}")
        parsed.append(
            TriggerCandidate(
                session_id=session_id,
                campaign_id=str(row["campaign_id"]),
                segment_id=segment_id,
                profile_id=str(row["profile_id"]),
                candidate_seq=candidate_seq,
                candidate_id=candidate_id,
                aggressor_side=aggressor_side,
                direction_sign=direction_sign,
                shock_ts_ns=shock_ts_ns,
                decision_ts_ns=decision_ts_ns,
                impact_ratio=_float_text(row["impact_ratio"], "impact_ratio"),
                primary_episode=primary_episode,
                rejection_reason=rejection_reason,
            )
        )
    return parsed


def validate_family_population(
    candidates: Sequence[TriggerCandidate],
    expected_candidate_count: int | None = None,
    expected_confirmed_count: int | None = None,
) -> dict[str, int]:
    candidate_count = len(candidates)
    confirmed_count = sum(1 for candidate in candidates if candidate.primary_episode)
    if expected_candidate_count is not None and candidate_count != expected_candidate_count:
        raise DensityContractError(
            f"Family A count drift: expected {expected_candidate_count}, got {candidate_count}"
        )
    if expected_confirmed_count is not None and confirmed_count != expected_confirmed_count:
        raise DensityContractError(
            f"Family B count drift: expected {expected_confirmed_count}, got {confirmed_count}"
        )
    return {
        "candidate_count": candidate_count,
        "confirmed_count": confirmed_count,
    }


def _session_duration_seconds(spans: Sequence[StructuralSpan]) -> float:
    return sum((span.end_ts_ns - span.start_ts_ns) for span in spans) / 1_000_000_000.0


def build_count_rate_summary(
    candidates: Sequence[TriggerCandidate], spans: Sequence[StructuralSpan]
) -> list[dict[str, Any]]:
    if not spans:
        raise DensityContractError("at least one structural span is required")
    duration_seconds = _session_duration_seconds(spans)
    if duration_seconds <= 0:
        raise DensityContractError("session duration must be positive")
    counts = validate_family_population(candidates)
    rows = []
    for population, count in (
        (FAMILY_A, counts["candidate_count"]),
        (FAMILY_B, counts["confirmed_count"]),
    ):
        rows.append(
            {
                "population": population,
                "count": count,
                "duration_seconds": duration_seconds,
                "rate_per_second": count / duration_seconds,
                "rate_per_minute": count * 60.0 / duration_seconds,
                "rate_per_hour": count * 3600.0 / duration_seconds,
            }
        )
    return rows


def _quantile_linear(values: Sequence[float], probability: float) -> float:
    if not values:
        raise DensityContractError("quantiles require at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _gap_quantiles_ms(values_ms: Sequence[float]) -> dict[str, float] | None:
    if not values_ms:
        return None
    return {
        f"p{int(probability * 100):02d}_ms": _quantile_linear(values_ms, probability)
        for probability in QUANTILES
    }


def _landmark_ts_ns(candidate: TriggerCandidate, population: str) -> int:
    if population == FAMILY_A:
        return candidate.shock_ts_ns
    if population == FAMILY_B:
        if not candidate.primary_episode or candidate.decision_ts_ns is None:
            raise DensityContractError(
                f"Family B requires confirmed decision landmark: {candidate.candidate_id}"
            )
        return candidate.decision_ts_ns
    raise DensityContractError(f"unknown landmark population: {population}")


def build_inter_trigger_distribution(
    candidates: Sequence[TriggerCandidate],
) -> list[dict[str, Any]]:
    populations = (
        (FAMILY_A, list(candidates)),
        (
            FAMILY_B,
            [candidate for candidate in candidates if candidate.primary_episode],
        ),
    )
    rows: list[dict[str, Any]] = []
    for population, population_candidates in populations:
        population_candidates = sorted(
            population_candidates,
            key=lambda candidate: (
                candidate.segment_id,
                _landmark_ts_ns(candidate, population),
                candidate.candidate_id,
            ),
        )
        consecutive_pairs = [
            (previous, current)
            for previous, current in zip(
                population_candidates, population_candidates[1:]
            )
            if previous.segment_id == current.segment_id
        ]
        all_gaps = [
            (
                _landmark_ts_ns(current, population)
                - _landmark_ts_ns(previous, population)
            )
            / 1_000_000.0
            for previous, current in consecutive_pairs
        ]
        same_side_gaps = [
            (
                _landmark_ts_ns(current, population)
                - _landmark_ts_ns(previous, population)
            )
            / 1_000_000.0
            for previous, current in consecutive_pairs
            if previous.direction_sign == current.direction_sign
        ]
        opposite_side_gaps = [
            (
                _landmark_ts_ns(current, population)
                - _landmark_ts_ns(previous, population)
            )
            / 1_000_000.0
            for previous, current in consecutive_pairs
            if previous.direction_sign != current.direction_sign
        ]
        for side_relation, values_ms in (
            ("all", all_gaps),
            ("same_side", same_side_gaps),
            ("opposite_side", opposite_side_gaps),
        ):
            quantiles = _gap_quantiles_ms(values_ms)
            rows.append(
                {
                    "population": population,
                    "side_relation": side_relation,
                    "pair_count": len(values_ms),
                    "quantiles_available": quantiles is not None,
                    **(
                        quantiles
                        if quantiles is not None
                        else {f"p{int(probability * 100):02d}_ms": None for probability in QUANTILES}
                    ),
                }
            )
    return rows


def _span_map(spans: Sequence[StructuralSpan]) -> dict[str, StructuralSpan]:
    mapping = {span.segment_id: span for span in spans}
    if len(mapping) != len(spans):
        raise DensityContractError("duplicate structural span segment_id")
    return mapping


def build_window_union_summary(
    candidates: Sequence[TriggerCandidate],
    spans: Sequence[StructuralSpan],
    window_ms: int = WINDOW_MS,
) -> list[dict[str, Any]]:
    span_by_segment = _span_map(spans)
    window_ns = window_ms * 1_000_000
    structural_duration_ms = _session_duration_seconds(spans) * 1_000.0
    rows: list[dict[str, Any]] = []
    for population, selected in (
        (FAMILY_A, list(candidates)),
        (FAMILY_B, [candidate for candidate in candidates if candidate.primary_episode]),
    ):
        merged_ranges: list[tuple[int, int]] = []
        longest_run_ns = 0
        by_segment: dict[str, list[TriggerCandidate]] = defaultdict(list)
        for candidate in selected:
            if candidate.segment_id not in span_by_segment:
                raise DensityContractError(
                    f"candidate references unknown structural span: {candidate.segment_id}"
                )
            by_segment[candidate.segment_id].append(candidate)
        coverage_ns = 0
        for segment_id, segment_candidates in sorted(by_segment.items()):
            span = span_by_segment[segment_id]
            ranges = sorted(
                (
                    _landmark_ts_ns(candidate, population),
                    min(
                        _landmark_ts_ns(candidate, population) + window_ns,
                        span.end_ts_ns,
                    ),
                )
                for candidate in segment_candidates
            )
            if not ranges:
                continue
            if ranges[0][0] < span.start_ts_ns or ranges[-1][0] >= span.end_ts_ns:
                raise DensityContractError(
                    f"candidate window lies outside structural span: {segment_id}"
                )
            current_start, current_end = ranges[0]
            for start_ns, end_ns in ranges[1:]:
                if start_ns <= current_end:
                    current_end = max(current_end, end_ns)
                    continue
                coverage_ns += current_end - current_start
                longest_run_ns = max(longest_run_ns, current_end - current_start)
                merged_ranges.append((current_start, current_end))
                current_start, current_end = start_ns, end_ns
            coverage_ns += current_end - current_start
            longest_run_ns = max(longest_run_ns, current_end - current_start)
            merged_ranges.append((current_start, current_end))
        rows.append(
            {
                "population": population,
                "window_ms": window_ms,
                "overlap_block_count_2000ms": len(merged_ranges),
                "structural_duration_ms": structural_duration_ms,
                "window_union_coverage_ms": coverage_ns / 1_000_000.0,
                "window_union_coverage_fraction": min(
                    1.0,
                    (coverage_ns / 1_000_000.0) / structural_duration_ms,
                ),
                "longest_continuous_trigger_run_ms": longest_run_ns / 1_000_000.0,
            }
        )
    return rows


def build_sensitivity_membership(
    candidates: Sequence[TriggerCandidate],
    primary_flow_first_candidate_ids: Iterable[str],
) -> list[dict[str, Any]]:
    first_candidate_ids = set(primary_flow_first_candidate_ids)
    refractory_thresholds_ms = (100, 250, 500)
    last_selected_by_threshold: dict[
        int, dict[tuple[str, int], int]
    ] = {
        threshold_ms: {} for threshold_ms in refractory_thresholds_ms
    }
    memberships: list[dict[str, Any]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.segment_id,
            item.shock_ts_ns,
            item.candidate_id,
        ),
    ):
        same_side_key = (candidate.segment_id, candidate.direction_sign)
        refractory_membership: dict[str, bool] = {}
        for threshold_ms in refractory_thresholds_ms:
            previous_selected_ts_ns = last_selected_by_threshold[
                threshold_ms
            ].get(same_side_key)
            selected = (
                previous_selected_ts_ns is None
                or (
                    candidate.shock_ts_ns - previous_selected_ts_ns
                )
                / 1_000_000.0
                > threshold_ms
            )
            refractory_membership[
                f"same_side_refractory_{threshold_ms}ms"
            ] = selected
            if selected:
                last_selected_by_threshold[threshold_ms][
                    same_side_key
                ] = candidate.shock_ts_ns
        memberships.append(
            {
                "candidate_id": candidate.candidate_id,
                "session_id": candidate.session_id,
                "population": FAMILY_A,
                "family_a_eligible": True,
                "family_b_eligible": candidate.primary_episode,
                "version": SENSITIVITY_VERSION,
                "impact_ge_050": candidate.impact_ratio >= 0.50,
                "impact_ge_070": candidate.impact_ratio >= 0.70,
                **refractory_membership,
                "first_per_primary_flow_episode": candidate.candidate_id
                in first_candidate_ids,
            }
        )
    unknown = first_candidate_ids - {candidate.candidate_id for candidate in candidates}
    if unknown:
        raise DensityContractError(f"unknown first-per-flow candidate ids: {sorted(unknown)}")
    return memberships


def build_time_block_membership(
    candidates: Sequence[TriggerCandidate],
    spans: Sequence[StructuralSpan],
    block_seconds: int = TIME_BLOCK_SECONDS,
) -> list[dict[str, Any]]:
    span_by_segment = _span_map(spans)
    memberships = []
    for candidate in candidates:
        span = span_by_segment.get(candidate.segment_id)
        if span is None:
            raise DensityContractError(
                f"candidate references unknown structural span: {candidate.segment_id}"
            )
        candidate_offset_ns = candidate.shock_ts_ns - span.start_ts_ns
        if candidate_offset_ns < 0 or candidate.shock_ts_ns >= span.end_ts_ns:
            raise DensityContractError(
                f"candidate lies outside structural span: {candidate.candidate_id}"
            )
        for population in (
            (FAMILY_A, FAMILY_B) if candidate.primary_episode else (FAMILY_A,)
        ):
            landmark_ts_ns = _landmark_ts_ns(candidate, population)
            offset_ns = landmark_ts_ns - span.start_ts_ns
            if offset_ns < 0 or landmark_ts_ns >= span.end_ts_ns:
                raise DensityContractError(
                    f"{population} landmark lies outside structural span: "
                    f"{candidate.candidate_id}"
                )
            block_index = offset_ns // (block_seconds * 1_000_000_000)
            block_start_ns = (
                span.start_ts_ns
                + block_index * block_seconds * 1_000_000_000
            )
            block_end_ns = min(
                block_start_ns + block_seconds * 1_000_000_000,
                span.end_ts_ns,
            )
            memberships.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "population": population,
                    "landmark_ts_ns": landmark_ts_ns,
                    "segment_id": candidate.segment_id,
                    "block_index": int(block_index),
                    "block_id": (
                        f"{candidate.segment_id}:block_{int(block_index):04d}"
                    ),
                    "block_start_ts_ns": block_start_ns,
                    "block_end_ts_ns": block_end_ns,
                }
            )
    return memberships


def build_time_block_catalog(
    spans: Sequence[StructuralSpan],
    block_seconds: int = TIME_BLOCK_SECONDS,
) -> list[dict[str, Any]]:
    if block_seconds <= 0:
        raise DensityContractError("time block seconds must be positive")
    rows = []
    block_ns = block_seconds * 1_000_000_000
    for span in spans:
        block_count = math.ceil(
            (span.end_ts_ns - span.start_ts_ns) / block_ns
        )
        for block_index in range(block_count):
            block_start_ns = span.start_ts_ns + block_index * block_ns
            block_end_ns = min(block_start_ns + block_ns, span.end_ts_ns)
            rows.append(
                {
                    "session_id": span.session_id,
                    "segment_id": span.segment_id,
                    "block_index": block_index,
                    "block_id": (
                        f"{span.segment_id}:block_{block_index:04d}"
                    ),
                    "block_start_ts_ns": block_start_ns,
                    "block_end_ts_ns": block_end_ns,
                    "duration_seconds": (
                        block_end_ns - block_start_ns
                    )
                    / 1_000_000_000.0,
                    "partial_block": block_end_ns - block_start_ns < block_ns,
                }
            )
    return rows


def summarize_time_blocks(
    memberships: Sequence[Mapping[str, Any]],
    spans: Sequence[StructuralSpan] | None = None,
) -> list[dict[str, Any]]:
    if spans is not None:
        structural_count = len(build_time_block_catalog(spans))
        return [
            {
                "population": population,
                "time_block_count_60s": structural_count,
                "count_semantics": "all_segment_contained_time_blocks",
            }
            for population in (FAMILY_A, FAMILY_B)
        ]
    block_ids_by_population: dict[str, set[str]] = defaultdict(set)
    for row in memberships:
        population = str(row["population"])
        if population not in {FAMILY_A, FAMILY_B}:
            raise DensityContractError(f"unknown population in time block row: {population}")
        block_ids_by_population[population].add(str(row["block_id"]))
    return [
        {
            "population": population,
            "time_block_count_60s": len(block_ids_by_population.get(population, set())),
            "count_semantics": "occupied_time_blocks_only",
        }
        for population in (FAMILY_A, FAMILY_B)
    ]


def _autocorrelation(series: Sequence[int], lag: int) -> float:
    mean = sum(series) / len(series)
    centered = [value - mean for value in series]
    denominator = sum(value * value for value in centered)
    if denominator == 0.0:
        raise DensityContractError("constant ESS series is unavailable")
    numerator = sum(
        centered[index] * centered[index + lag]
        for index in range(len(series) - lag)
    )
    return numerator / denominator


def estimate_segment_bartlett_ess(series: Sequence[int]) -> dict[str, Any]:
    if len(series) < 2:
        return {
            "available": False,
            "unavailable_reason": "too_short_for_autocorrelation",
            "sample_size_n": len(series),
            "bartlett_tau": None,
            "effective_sample_size": None,
        }
    if len(set(series)) == 1:
        return {
            "available": False,
            "unavailable_reason": "constant_series",
            "sample_size_n": len(series),
            "bartlett_tau": None,
            "effective_sample_size": None,
        }
    max_lag = min(ESS_MAX_LAG_SECONDS, len(series) - 1)
    pair_sum_total = 0.0
    used_lags = 0
    lag = 1
    while lag <= max_lag:
        rho_odd = _autocorrelation(series, lag)
        rho_even = _autocorrelation(series, lag + 1) if lag + 1 <= max_lag else 0.0
        pair_sum = rho_odd + rho_even
        if pair_sum <= 0.0:
            break
        pair_sum_total += pair_sum
        used_lags = lag + 1 if lag + 1 <= max_lag else lag
        lag += 2
    tau = 1.0 + 2.0 * pair_sum_total
    n_value = len(series)
    effective_n = min(float(n_value), max(1.0, n_value / tau))
    return {
        "available": True,
        "unavailable_reason": "",
        "sample_size_n": n_value,
        "lags_used": used_lags,
        "bartlett_tau": tau,
        "effective_sample_size": effective_n,
    }


def build_segment_second_count_series(
    candidates: Sequence[TriggerCandidate],
    spans: Sequence[StructuralSpan],
    population: str,
) -> dict[str, list[int]]:
    if population not in {FAMILY_A, FAMILY_B}:
        raise DensityContractError(f"unknown ESS population: {population}")
    filtered = [
        candidate
        for candidate in candidates
        if population == FAMILY_A or candidate.primary_episode
    ]
    span_by_segment = _span_map(spans)
    counts: dict[str, list[int]] = {}
    counts_by_segment_second: dict[str, Counter[int]] = defaultdict(Counter)
    for candidate in filtered:
        span = span_by_segment.get(candidate.segment_id)
        if span is None:
            raise DensityContractError(
                f"candidate references unknown structural span: {candidate.segment_id}"
            )
        landmark_ts_ns = _landmark_ts_ns(candidate, population)
        second_index = (landmark_ts_ns - span.start_ts_ns) // 1_000_000_000
        if second_index < 0:
            raise DensityContractError(
                f"{population} landmark precedes segment start: "
                f"{candidate.candidate_id}"
            )
        if landmark_ts_ns >= span.end_ts_ns:
            raise DensityContractError(
                f"{population} landmark reaches segment end: "
                f"{candidate.candidate_id}"
            )
        counts_by_segment_second[candidate.segment_id][int(second_index)] += 1
    for span in spans:
        span_seconds = math.ceil((span.end_ts_ns - span.start_ts_ns) / 1_000_000_000.0)
        counts[span.segment_id] = [
            counts_by_segment_second[span.segment_id].get(second_index, 0)
            for second_index in range(span_seconds)
        ]
    return counts


def build_effective_sample_size_rows(
    candidates: Sequence[TriggerCandidate],
    spans: Sequence[StructuralSpan],
) -> list[dict[str, Any]]:
    rows = []
    for population in (FAMILY_A, FAMILY_B):
        per_segment = build_segment_second_count_series(candidates, spans, population)
        segment_effective_n = 0.0
        segment_sample_n = 0
        unavailable_reasons: list[str] = []
        for span in spans:
            result = estimate_segment_bartlett_ess(per_segment[span.segment_id])
            rows.append(
                {
                    "population": population,
                    "segment_id": span.segment_id,
                    "estimator_name": "bartlett_ess_1s_geyer_ipps",
                    **result,
                }
            )
            if result["available"]:
                segment_effective_n += float(result["effective_sample_size"])
                segment_sample_n += int(result["sample_size_n"])
            else:
                unavailable_reasons.append(
                    f"{span.segment_id}:{result['unavailable_reason']}"
                )
        rows.append(
            {
                "population": population,
                "segment_id": "session_total",
                "estimator_name": "bartlett_ess_1s_geyer_ipps",
                "available": not unavailable_reasons,
                "unavailable_reason": "|".join(unavailable_reasons),
                "sample_size_n": segment_sample_n if not unavailable_reasons else None,
                "bartlett_tau": None,
                "effective_sample_size": (
                    segment_effective_n if not unavailable_reasons else None
                ),
            }
        )
    return rows


def validate_effective_sample_size_rows(
    rows: Sequence[Mapping[str, Any]]
) -> None:
    for row in rows:
        estimator_name = str(row.get("estimator_name", ""))
        if estimator_name != "bartlett_ess_1s_geyer_ipps":
            raise DensityContractError(f"unexpected ESS estimator: {estimator_name}")
        effective_n = row.get("effective_sample_size")
        sample_size = row.get("sample_size_n")
        if row.get("available") is False:
            if effective_n is not None:
                raise DensityContractError("unavailable ESS must not invent effective_sample_size")
            continue
        if effective_n is None or sample_size is None:
            raise DensityContractError("available ESS must include sample_size_n and effective_sample_size")
        if float(effective_n) == float(sample_size):
            tau = row.get("bartlett_tau")
            if tau is None:
                raise DensityContractError("row_count-as-N_eff is forbidden")
            if float(tau) < 1.0:
                raise DensityContractError("Bartlett tau must be at least one")
