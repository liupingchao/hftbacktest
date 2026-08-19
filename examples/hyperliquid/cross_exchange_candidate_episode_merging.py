from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import cross_exchange_liquidity_response_case_hierarchy as accepted_hierarchy


EPISODE_MERGING_VERSION = "episode_merging_v1"
PRIMARY_OUTCOME_HORIZON_MS = 2000
PRIMARY_CLUSTER_GAP_MS = 100
PRIMARY_BRIDGE_GAP_MS = 250
PRIMARY_RECOVERY_SPAN_MS = 50
PRIMARY_DEPTH_RECOVERY_RATIO = 0.80
PRIMARY_SPREAD_ALLOWANCE_TICKS = 1


class CandidateEpisodeMergingError(ValueError):
    """Raised when candidate episode merging inputs or outputs drift."""


@dataclass(frozen=True)
class _Candidate:
    candidate_id: str
    segment_id: str
    connection_epoch_id: str
    segment_end_ts_ns: int
    connection_epoch_end_ts_ns: int
    shock_ts_ns: int
    direction_sign: int
    binance_pre_spread_px: float
    binance_pre_mid_px: float
    binance_pre_top5_impacted_qty: float
    binance_pre_top5_opposite_qty: float
    original: Mapping[str, Any]

    @property
    def partition_key(self) -> tuple[str, str]:
        return self.segment_id, self.connection_epoch_id

    @property
    def top5_depth(self) -> float:
        return self.binance_pre_top5_impacted_qty + self.binance_pre_top5_opposite_qty

    @property
    def overlap_window_end_ts_ns(self) -> int:
        return min(
            self.shock_ts_ns + PRIMARY_OUTCOME_HORIZON_MS * 1_000_000,
            self.segment_end_ts_ns,
            self.connection_epoch_end_ts_ns,
        )


def _require_text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if value is None or str(value) == "":
        raise CandidateEpisodeMergingError(f"candidate field is required: {field}")
    return str(value)


def _require_int(row: Mapping[str, Any], field: str) -> int:
    value = row.get(field)
    if value is None or str(value) == "":
        raise CandidateEpisodeMergingError(f"candidate field is required: {field}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise CandidateEpisodeMergingError(f"candidate field must be int: {field}") from exc


def _require_float(row: Mapping[str, Any], field: str) -> float:
    value = row.get(field)
    if value is None or str(value) == "":
        raise CandidateEpisodeMergingError(f"candidate field is required: {field}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise CandidateEpisodeMergingError(f"candidate field must be float: {field}") from exc


def _normalize_candidates(candidates: Iterable[Mapping[str, Any]]) -> list[_Candidate]:
    normalized: list[_Candidate] = []
    seen_keys: set[str] = set()
    previous_partition: tuple[str, str] | None = None
    previous_shock_ts_ns: int | None = None
    closed_partitions: set[tuple[str, str]] = set()
    partition_metadata: dict[tuple[str, str], tuple[int, int]] = {}
    for index, row in enumerate(candidates, start=1):
        segment_id = _require_text(row, "segment_id")
        connection_epoch_id = _require_text(row, "connection_epoch_id")
        shock_ts_ns = _require_int(row, "shock_ts_ns")
        segment_end_ts_ns = _require_int(row, "segment_end_ts_ns")
        connection_epoch_end_ts_ns = _require_int(row, "connection_epoch_end_ts_ns")
        if segment_end_ts_ns < shock_ts_ns:
            raise CandidateEpisodeMergingError("segment boundary precedes candidate shock")
        if connection_epoch_end_ts_ns < shock_ts_ns:
            raise CandidateEpisodeMergingError("connection epoch boundary precedes candidate shock")
        partition_key = (segment_id, connection_epoch_id)
        metadata = (segment_end_ts_ns, connection_epoch_end_ts_ns)
        if partition_key in partition_metadata and partition_metadata[partition_key] != metadata:
            raise CandidateEpisodeMergingError(
                "segment/epoch boundary metadata drift within one partition"
            )
        partition_metadata.setdefault(partition_key, metadata)
        if previous_partition is None:
            previous_partition = partition_key
        elif partition_key != previous_partition:
            closed_partitions.add(previous_partition)
            if partition_key in closed_partitions:
                raise CandidateEpisodeMergingError(
                    "candidate partitions must be contiguous in input order"
                )
            previous_partition = partition_key
            previous_shock_ts_ns = None
        candidate_id = str(row.get("candidate_id") or f"{segment_id}-{connection_epoch_id}-A{index:06d}")
        if candidate_id in seen_keys:
            raise CandidateEpisodeMergingError(f"duplicate candidate id: {candidate_id}")
        if previous_shock_ts_ns is not None and shock_ts_ns < previous_shock_ts_ns:
            raise CandidateEpisodeMergingError("candidate rows must be sorted by shock timestamp")
        previous_shock_ts_ns = shock_ts_ns
        seen_keys.add(candidate_id)
        normalized.append(
            _Candidate(
                candidate_id=candidate_id,
                segment_id=segment_id,
                connection_epoch_id=connection_epoch_id,
                segment_end_ts_ns=segment_end_ts_ns,
                connection_epoch_end_ts_ns=connection_epoch_end_ts_ns,
                shock_ts_ns=shock_ts_ns,
                direction_sign=_require_int(row, "direction_sign"),
                binance_pre_spread_px=_require_float(row, "binance_pre_spread_px"),
                binance_pre_mid_px=_require_float(row, "binance_pre_mid_px"),
                binance_pre_top5_impacted_qty=_require_float(
                    row, "binance_pre_top5_impacted_qty"
                ),
                binance_pre_top5_opposite_qty=_require_float(
                    row, "binance_pre_top5_opposite_qty"
                ),
                original=row,
            )
        )
    return normalized


def _normalize_timeline_states(
    states: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    accepted_states = []
    for row in states:
        bid_px = float(row["binance_bid_1_px"])
        ask_px = float(row["binance_ask_1_px"])
        bid_depth = sum(float(row[f"binance_bid_{level}_qty"]) for level in range(1, 6))
        ask_depth = sum(float(row[f"binance_ask_{level}_qty"]) for level in range(1, 6))
        accepted_states.append(
            {
                "ts_ns": int(row["common_ts_ns"]),
                "mid_px": (bid_px + ask_px) / 2.0,
                "spread_px": ask_px - bid_px,
                "top5_depth": bid_depth + ask_depth,
            }
        )
    accepted_states.sort(key=lambda row: int(row["ts_ns"]))
    return {
        "states": accepted_states,
        "ts_ns": [int(row["ts_ns"]) for row in accepted_states],
    }


def candidate_recovery_checkpoint_v1(
    *,
    timeline: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None,
    start_ns: int,
    end_ns: int,
    pre_spread_px: float,
    pre_depth: float,
    direction_sign: int,
    extreme_mid_px: float,
    recovery_span_ms: int = PRIMARY_RECOVERY_SPAN_MS,
    depth_recovery_ratio: float = PRIMARY_DEPTH_RECOVERY_RATIO,
    spread_allowance_ticks: int = PRIMARY_SPREAD_ALLOWANCE_TICKS,
) -> dict[str, Any]:
    normalized_timeline: dict[str, Any] | None
    if timeline is None:
        normalized_timeline = None
    elif isinstance(timeline, Mapping) and {"states", "ts_ns"} <= set(timeline):
        normalized_timeline = {
            "states": list(timeline["states"]),
            "ts_ns": list(timeline["ts_ns"]),
        }
    else:
        normalized_timeline = _normalize_timeline_states(timeline)  # type: ignore[arg-type]
    return accepted_hierarchy._recovery_checkpoint(
        timeline=normalized_timeline,
        start_ns=start_ns,
        end_ns=end_ns,
        pre_spread_px=pre_spread_px,
        pre_depth=pre_depth,
        direction_sign=direction_sign,
        extreme_mid_px=extreme_mid_px,
        recovery_span_ms=recovery_span_ms,
        depth_recovery_ratio=depth_recovery_ratio,
        spread_allowance_ticks=spread_allowance_ticks,
    )


def _partition_rows(candidates: list[_Candidate]) -> dict[tuple[str, str], list[_Candidate]]:
    rows: dict[tuple[str, str], list[_Candidate]] = {}
    for candidate in candidates:
        rows.setdefault(candidate.partition_key, []).append(candidate)
    return rows


def _dominant_direction(candidates: list[_Candidate]) -> int:
    total = sum(candidate.direction_sign for candidate in candidates)
    if total > 0:
        return 1
    if total < 0:
        return -1
    return candidates[0].direction_sign


def _summarize_cluster(
    segment_id: str,
    connection_epoch_id: str,
    cluster_seq: int,
    members: list[_Candidate],
) -> dict[str, Any]:
    return {
        "cluster_id": f"{segment_id}-{connection_epoch_id}-C{cluster_seq:06d}",
        "segment_id": segment_id,
        "connection_epoch_id": connection_epoch_id,
        "first_candidate_id": members[0].candidate_id,
        "last_candidate_id": members[-1].candidate_id,
        "first_shock_ts_ns": members[0].shock_ts_ns,
        "last_shock_ts_ns": members[-1].shock_ts_ns,
        "candidate_count": len(members),
        "duration_ms": (members[-1].shock_ts_ns - members[0].shock_ts_ns) / 1_000_000.0,
    }


def _build_clusters(partitioned: dict[tuple[str, str], list[_Candidate]]) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, list[_Candidate]]]:
    clusters: list[dict[str, Any]] = []
    candidate_to_cluster: dict[str, str] = {}
    cluster_members: dict[str, list[_Candidate]] = {}
    for (segment_id, connection_epoch_id), rows in partitioned.items():
        current: list[_Candidate] = []
        cluster_seq = 0
        previous_shock_ts_ns: int | None = None
        for candidate in rows:
            if (
                current
                and previous_shock_ts_ns is not None
                and (candidate.shock_ts_ns - previous_shock_ts_ns) / 1_000_000.0
                > PRIMARY_CLUSTER_GAP_MS
            ):
                cluster_seq += 1
                cluster = _summarize_cluster(segment_id, connection_epoch_id, cluster_seq, current)
                clusters.append(cluster)
                cluster_members[cluster["cluster_id"]] = list(current)
                for member in current:
                    candidate_to_cluster[member.candidate_id] = cluster["cluster_id"]
                current = []
            current.append(candidate)
            previous_shock_ts_ns = candidate.shock_ts_ns
        if current:
            cluster_seq += 1
            cluster = _summarize_cluster(segment_id, connection_epoch_id, cluster_seq, current)
            clusters.append(cluster)
            cluster_members[cluster["cluster_id"]] = list(current)
            for member in current:
                candidate_to_cluster[member.candidate_id] = cluster["cluster_id"]
    return clusters, candidate_to_cluster, cluster_members


def _summarize_episode(
    segment_id: str,
    connection_epoch_id: str,
    episode_seq: int,
    clusters: list[dict[str, Any]],
    members: list[_Candidate],
) -> dict[str, Any]:
    return {
        "continuous_flow_episode_id": f"{segment_id}-{connection_epoch_id}-E{episode_seq:06d}",
        "segment_id": segment_id,
        "connection_epoch_id": connection_epoch_id,
        "first_cluster_id": clusters[0]["cluster_id"],
        "last_cluster_id": clusters[-1]["cluster_id"],
        "first_candidate_id": members[0].candidate_id,
        "last_candidate_id": members[-1].candidate_id,
        "start_ts_ns": members[0].shock_ts_ns,
        "end_ts_ns": members[-1].shock_ts_ns,
        "cluster_count": len(clusters),
        "candidate_count": len(members),
        "duration_ms": (members[-1].shock_ts_ns - members[0].shock_ts_ns) / 1_000_000.0,
    }


def _build_episodes(
    *,
    clusters: list[dict[str, Any]],
    cluster_members: dict[str, list[_Candidate]],
    timelines_by_segment: Mapping[str, Mapping[str, Any] | Iterable[Mapping[str, Any]]] | None,
) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    cluster_to_episode: dict[str, str] = {}
    boundary_audit: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    clusters_by_partition: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for cluster in clusters:
        key = (str(cluster["segment_id"]), str(cluster["connection_epoch_id"]))
        clusters_by_partition.setdefault(key, []).append(cluster)
    normalized_timelines = {
        segment_id: (
            timeline
            if isinstance(timeline, Mapping) and {"states", "ts_ns"} <= set(timeline)
            else _normalize_timeline_states(timeline)  # type: ignore[arg-type]
        )
        for segment_id, timeline in (timelines_by_segment or {}).items()
    }
    for (segment_id, connection_epoch_id), partition_clusters in clusters_by_partition.items():
        episode_seq = 1
        current_clusters = [partition_clusters[0]]
        current_members = list(cluster_members[partition_clusters[0]["cluster_id"]])
        direction_sign = _dominant_direction(current_members)
        pre_spread_px = current_members[0].binance_pre_spread_px
        pre_depth = current_members[0].top5_depth
        extreme_mid_px = current_members[0].binance_pre_mid_px
        timeline = normalized_timelines.get(segment_id)
        for left_cluster, right_cluster in zip(partition_clusters, partition_clusters[1:]):
            gap_ms = (
                int(right_cluster["first_shock_ts_ns"]) - int(left_cluster["last_shock_ts_ns"])
            ) / 1_000_000.0
            merged = False
            recovery = {"status": "gap_exceeds_bridge"}
            reason = "gap_exceeds_bridge"
            if gap_ms <= PRIMARY_BRIDGE_GAP_MS:
                recovery = candidate_recovery_checkpoint_v1(
                    timeline=timeline,
                    start_ns=int(left_cluster["last_shock_ts_ns"]),
                    end_ns=int(right_cluster["first_shock_ts_ns"]),
                    pre_spread_px=pre_spread_px,
                    pre_depth=pre_depth,
                    direction_sign=direction_sign,
                    extreme_mid_px=extreme_mid_px,
                )
                if recovery["status"] == "no_recovery_checkpoint":
                    merged = True
                    reason = "bridge_without_recovery_checkpoint"
                else:
                    reason = str(recovery["status"])
            boundary_audit.append(
                {
                    "segment_id": segment_id,
                    "connection_epoch_id": connection_epoch_id,
                    "left_cluster_id": left_cluster["cluster_id"],
                    "right_cluster_id": right_cluster["cluster_id"],
                    "cluster_gap_ms": gap_ms,
                    "bridge_gap_ms": PRIMARY_BRIDGE_GAP_MS,
                    "recovery_status": recovery["status"],
                    "merged": merged,
                    "decision_reason": reason,
                }
            )
            if merged:
                next_members = cluster_members[right_cluster["cluster_id"]]
                current_clusters.append(right_cluster)
                current_members.extend(next_members)
                for member in next_members:
                    if direction_sign >= 0:
                        extreme_mid_px = max(extreme_mid_px, member.binance_pre_mid_px)
                    else:
                        extreme_mid_px = min(extreme_mid_px, member.binance_pre_mid_px)
                continue
            episode = _summarize_episode(
                segment_id, connection_epoch_id, episode_seq, current_clusters, current_members
            )
            episodes.append(episode)
            for cluster in current_clusters:
                cluster_to_episode[cluster["cluster_id"]] = episode["continuous_flow_episode_id"]
            episode_seq += 1
            current_clusters = [right_cluster]
            current_members = list(cluster_members[right_cluster["cluster_id"]])
            direction_sign = _dominant_direction(current_members)
            pre_spread_px = current_members[0].binance_pre_spread_px
            pre_depth = current_members[0].top5_depth
            extreme_mid_px = current_members[0].binance_pre_mid_px
        episode = _summarize_episode(
            segment_id, connection_epoch_id, episode_seq, current_clusters, current_members
        )
        episodes.append(episode)
        for cluster in current_clusters:
            cluster_to_episode[cluster["cluster_id"]] = episode["continuous_flow_episode_id"]
    return episodes, cluster_to_episode, boundary_audit


def _build_overlap_blocks(
    partitioned: dict[tuple[str, str], list[_Candidate]]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    overlap_blocks: list[dict[str, Any]] = []
    candidate_to_overlap_block: dict[str, str] = {}
    for (segment_id, connection_epoch_id), rows in partitioned.items():
        block_seq = 0
        current_members: list[_Candidate] = []
        current_start_ts_ns: int | None = None
        current_end_ts_ns: int | None = None
        for candidate in rows:
            window_start = candidate.shock_ts_ns
            window_end = candidate.overlap_window_end_ts_ns
            if current_start_ts_ns is None or current_end_ts_ns is None:
                block_seq += 1
                current_members = [candidate]
                current_start_ts_ns = window_start
                current_end_ts_ns = window_end
                continue
            if window_start <= current_end_ts_ns:
                current_members.append(candidate)
                current_end_ts_ns = max(current_end_ts_ns, window_end)
                continue
            block_id = f"{segment_id}-{connection_epoch_id}-B{block_seq:06d}"
            for member in current_members:
                candidate_to_overlap_block[member.candidate_id] = block_id
            overlap_blocks.append(
                {
                    "overlap_block_id": block_id,
                    "segment_id": segment_id,
                    "connection_epoch_id": connection_epoch_id,
                    "start_ts_ns": current_start_ts_ns,
                    "end_ts_ns": current_end_ts_ns,
                    "candidate_count": len(current_members),
                }
            )
            block_seq += 1
            current_members = [candidate]
            current_start_ts_ns = window_start
            current_end_ts_ns = window_end
        if current_members and current_start_ts_ns is not None and current_end_ts_ns is not None:
            block_id = f"{segment_id}-{connection_epoch_id}-B{block_seq:06d}"
            for member in current_members:
                candidate_to_overlap_block[member.candidate_id] = block_id
            overlap_blocks.append(
                {
                    "overlap_block_id": block_id,
                    "segment_id": segment_id,
                    "connection_epoch_id": connection_epoch_id,
                    "start_ts_ns": current_start_ts_ns,
                    "end_ts_ns": current_end_ts_ns,
                    "candidate_count": len(current_members),
                }
            )
    return overlap_blocks, candidate_to_overlap_block


def assert_all_candidates_conserved(
    candidates: Iterable[Mapping[str, Any]],
    membership: Iterable[Mapping[str, Any]],
) -> None:
    normalized = _normalize_candidates(candidates)
    expected_ids = [candidate.candidate_id for candidate in normalized]
    actual_ids = [str(row["candidate_id"]) for row in membership]
    if Counter(actual_ids) != Counter(expected_ids):
        raise CandidateEpisodeMergingError("candidate membership does not conserve all candidates")
    for row in membership:
        for field in (
            "cluster_id",
            "continuous_flow_episode_id",
            "overlap_block_id",
            "segment_id",
            "connection_epoch_id",
        ):
            if not row.get(field):
                raise CandidateEpisodeMergingError(f"membership field is required: {field}")


def summarize_episode_merging_v1(result: Mapping[str, Any]) -> dict[str, Any]:
    membership = list(result["membership"])
    return {
        "version": result["version"],
        "candidate_count": len(membership),
        "cluster_count": len(result["clusters"]),
        "continuous_flow_episode_count": len(result["continuous_flow_episodes"]),
        "overlap_block_count_2000ms": len(result["overlap_blocks"]),
        "segment_connection_epoch_count": len(
            {
                (row["segment_id"], row["connection_epoch_id"])
                for row in membership
            }
        ),
    }


def episode_merging_v1(
    candidates: Iterable[Mapping[str, Any]],
    *,
    timelines_by_segment: Mapping[str, Mapping[str, Any] | Iterable[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    candidate_rows = list(candidates)
    normalized = _normalize_candidates(candidate_rows)
    partitioned = _partition_rows(normalized)
    clusters, candidate_to_cluster, cluster_members = _build_clusters(partitioned)
    episodes, cluster_to_episode, boundary_audit = _build_episodes(
        clusters=clusters,
        cluster_members=cluster_members,
        timelines_by_segment=timelines_by_segment,
    )
    overlap_blocks, candidate_to_overlap_block = _build_overlap_blocks(partitioned)
    episode_by_id = {
        row["continuous_flow_episode_id"]: row for row in episodes
    }
    membership = []
    for candidate in normalized:
        cluster_id = candidate_to_cluster[candidate.candidate_id]
        episode_id = cluster_to_episode[cluster_id]
        overlap_block_id = candidate_to_overlap_block[candidate.candidate_id]
        membership.append(
            {
                "candidate_id": candidate.candidate_id,
                "segment_id": candidate.segment_id,
                "connection_epoch_id": candidate.connection_epoch_id,
                "shock_ts_ns": candidate.shock_ts_ns,
                "cluster_id": cluster_id,
                "continuous_flow_episode_id": episode_id,
                "overlap_block_id": overlap_block_id,
                "window_end_ts_ns": candidate.overlap_window_end_ts_ns,
                "episode_candidate_count": episode_by_id[episode_id]["candidate_count"],
            }
        )
    result = {
        "version": EPISODE_MERGING_VERSION,
        "parameters": {
            "cluster_gap_ms": PRIMARY_CLUSTER_GAP_MS,
            "bridge_gap_ms": PRIMARY_BRIDGE_GAP_MS,
            "recovery_span_ms": PRIMARY_RECOVERY_SPAN_MS,
            "depth_recovery_ratio": PRIMARY_DEPTH_RECOVERY_RATIO,
            "spread_allowance_ticks": PRIMARY_SPREAD_ALLOWANCE_TICKS,
            "outcome_horizon_ms": PRIMARY_OUTCOME_HORIZON_MS,
        },
        "membership": membership,
        "clusters": clusters,
        "continuous_flow_episodes": episodes,
        "overlap_blocks": overlap_blocks,
        "boundary_audit": boundary_audit,
        "summary": {},
    }
    assert_all_candidates_conserved(candidate_rows, membership)
    result["summary"] = summarize_episode_merging_v1(result)
    return result
