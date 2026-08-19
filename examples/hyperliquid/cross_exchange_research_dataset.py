#!/usr/bin/env python3
"""Build an auditable cross-exchange research event store from a local campaign."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import io
import json
import math
import os
import shutil
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

try:
    from cross_exchange_symbol_registry import available_profile_ids, get_symbol_profile
except ModuleNotFoundError:  # pragma: no cover - package import path
    from examples.hyperliquid.cross_exchange_symbol_registry import (
        available_profile_ids,
        get_symbol_profile,
    )


TASK_ID = "0730T013"
SCHEMA_VERSION = "cross_exchange_research_dataset_v2"
BINANCE_HOT_FIELDS = [
    "segment_id",
    "event_seq",
    "source_raw_seq",
    "local_ts_ns",
    "exchange_ts_ns",
    "event_type",
    "symbol",
    "connection_epoch_id",
    "degraded",
    "degraded_interval_ids",
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
]
HYPERLIQUID_HOT_FIELDS = [
    "segment_id",
    "event_seq",
    "source_raw_seq",
    "source_item_index",
    "local_ts_ns",
    "exchange_ts_ns",
    "event_type",
    "coin",
    "connection_epoch_id",
    "degraded",
    "degraded_interval_ids",
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
]
AUXILIARY_FIELDS = [
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
]
MASK_INDEX_FIELDS = [
    "campaign_id",
    "segment_id",
    "profile_id",
    "first_common_ts_ns",
    "last_common_ts_ns",
    "previous_segment_gap_ms",
    "cross_segment_continuity_claimed",
    "mask_type",
    "track_id",
    "mask_start_ts_ns",
    "mask_end_ts_ns",
    "duration_ms",
    "policy",
    "reason",
]
AUXILIARY_TRACKS = ("asset_context", "main_all_mids", "target_dex_all_mids")


class ResearchDatasetError(RuntimeError):
    """Raised when the local campaign cannot satisfy the R0 dataset contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ResearchDatasetError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResearchDatasetError(f"{path}: expected a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def _deterministic_gzip_text_writer(path: Path) -> Iterator[TextIO]:
    with path.open("wb") as raw_fh:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_fh,
            compresslevel=1,
            mtime=0,
        ) as gzip_fh:
            with io.TextIOWrapper(
                gzip_fh,
                encoding="utf-8",
                newline="",
            ) as text_fh:
                yield text_fh


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    row_count = 0
    try:
        if path.suffix == ".gz":
            fh_context = _deterministic_gzip_text_writer(temporary)
        else:
            fh_context = temporary.open("w", encoding="utf-8", newline="")
        with fh_context as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fieldnames})
                row_count += 1
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return row_count


def _parse_raw_line(line: str, *, path: Path, raw_seq: int) -> tuple[int, dict[str, Any]]:
    try:
        local_ts_text, payload_text = line.split(" ", 1)
        local_ts_ns = int(local_ts_text)
        payload = json.loads(payload_text)
    except Exception as exc:
        raise ResearchDatasetError(f"{path}: invalid raw row {raw_seq}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResearchDatasetError(f"{path}: raw row {raw_seq} is not a JSON object")
    return local_ts_ns, payload


def _compact_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _bool_text(value: Any) -> str:
    return "true" if value is True else "false"


def _int_ns_from_ms(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(value) * 1_000_000


def _require_hash(path: Path, expected: Any, *, label: str) -> str:
    if not path.is_file():
        raise ResearchDatasetError(f"{label}: missing file {path}")
    computed = sha256_file(path)
    if str(expected or "") != computed:
        raise ResearchDatasetError(
            f"{label}: SHA mismatch for {path}: expected={expected}, computed={computed}"
        )
    return computed


def _require_nonnegative_int(value: Any, *, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ResearchDatasetError(f"{label}: invalid integer {value!r}") from exc
    if parsed < 0:
        raise ResearchDatasetError(f"{label}: negative integer {parsed}")
    return parsed


def _require_bool(value: Any, *, label: str) -> bool:
    if value is True or str(value).lower() == "true":
        return True
    if value is False or str(value).lower() == "false":
        return False
    raise ResearchDatasetError(f"{label}: invalid boolean {value!r}")


def _combined_channel_counts(manifest: dict[str, Any], *, label: str) -> dict[str, int]:
    combined: Counter[str] = Counter()
    for field in ("message_count_by_channel", "control_message_count_by_channel"):
        counts = manifest.get(field, {})
        if not isinstance(counts, dict):
            raise ResearchDatasetError(f"{label}: invalid {field}")
        for channel, value in counts.items():
            combined[str(channel)] += _require_nonnegative_int(
                value,
                label=f"{label}:{field}:{channel}",
            )
    return dict(sorted(combined.items()))


def _timeline_csv_summary(
    path: Path,
    *,
    expected_campaign_id: str,
    expected_segment_id: str,
    expected_profile_id: str,
) -> dict[str, int]:
    row_count = 0
    first_common_ts_ns: int | None = None
    last_common_ts_ns: int | None = None
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required_fields = {
            "campaign_id",
            "segment_id",
            "profile_id",
            "common_ts_ns",
        }
        missing_fields = required_fields.difference(reader.fieldnames or [])
        if missing_fields:
            raise ResearchDatasetError(
                f"{path}: missing timeline fields {sorted(missing_fields)}"
            )
        for row_count, row in enumerate(reader, start=1):
            observed_identity = (
                row["campaign_id"],
                row["segment_id"],
                row["profile_id"],
            )
            expected_identity = (
                expected_campaign_id,
                expected_segment_id,
                expected_profile_id,
            )
            if observed_identity != expected_identity:
                raise ResearchDatasetError(
                    f"{path}: timeline row identity mismatch at row {row_count}: "
                    f"expected={expected_identity}, observed={observed_identity}"
                )
            try:
                common_ts_ns = int(row["common_ts_ns"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ResearchDatasetError(
                    f"{path}: invalid common_ts_ns at row {row_count}"
                ) from exc
            if last_common_ts_ns is not None and common_ts_ns < last_common_ts_ns:
                raise ResearchDatasetError(
                    f"{path}: common timestamp regressed at row {row_count}"
                )
            if first_common_ts_ns is None:
                first_common_ts_ns = common_ts_ns
            last_common_ts_ns = common_ts_ns
    if row_count == 0 or first_common_ts_ns is None or last_common_ts_ns is None:
        raise ResearchDatasetError(f"{path}: empty common timeline")
    return {
        "row_count": row_count,
        "first_common_ts_ns": first_common_ts_ns,
        "last_common_ts_ns": last_common_ts_ns,
    }


def _track_raw_path(sample_dir: Path, track: dict[str, Any]) -> Path:
    relative = str(track.get("relative_output_dir", "."))
    return sample_dir / "hyperliquid_public_sample" / relative / "raw.gz"


def _expected_dex(profile_coin: str) -> str:
    return profile_coin.split(":", 1)[0] if ":" in profile_coin else ""


def _degraded_interval_id(interval: dict[str, Any], index: int) -> str:
    return (
        f"{interval.get('segment_id', '')}:"
        f"{interval.get('track_id', '')}:"
        f"{index + 1}"
    )


def _matching_degraded_interval(
    *,
    intervals: list[dict[str, Any]],
    segment_id: str,
    track_id: str,
    local_ts_ns: int,
) -> tuple[bool, str]:
    for index, interval in enumerate(intervals):
        if interval.get("segment_id") != segment_id or interval.get("track_id") != track_id:
            continue
        start = int(interval["degraded_start_local_ts_ns"])
        end = int(interval["recovered_local_ts_ns"])
        if start <= local_ts_ns <= end:
            return True, _degraded_interval_id(interval, index)
    return False, ""


def _matching_degraded_interval_ids(
    *,
    intervals: list[dict[str, Any]],
    segment_id: str,
    track_id: str,
    local_ts_ns: int,
) -> list[str]:
    return [
        _degraded_interval_id(interval, index)
        for index, interval in enumerate(intervals)
        if interval.get("segment_id") == segment_id
        and interval.get("track_id") == track_id
        and int(interval["degraded_start_local_ts_ns"])
        <= local_ts_ns
        <= int(interval["recovered_local_ts_ns"])
    ]


def _connection_epoch_id(
    *,
    intervals: list[dict[str, Any]],
    segment_id: str,
    track_id: str,
    local_ts_ns: int,
) -> int:
    return sum(
        1
        for interval in intervals
        if interval.get("segment_id") == segment_id
        and interval.get("track_id") == track_id
        and interval.get("reason") == "websocket_reconnect"
        and int(interval["disconnect_local_ts_ns"]) <= local_ts_ns
    )


def _binance_rows(
    *,
    path: Path,
    segment_id: str,
    expected_symbol: str,
    expected_counts: dict[str, Any],
    expected_snapshot_count: Any,
    intervals: list[dict[str, Any]],
    observed: dict[str, Any],
) -> Iterable[dict[str, Any]]:
    counts: Counter[str] = Counter()
    previous_local_ts = -1
    event_seq = 0
    connection_epochs_seen: set[int] = set()
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for raw_seq, line in enumerate(fh, start=1):
            local_ts_ns, payload = _parse_raw_line(line, path=path, raw_seq=raw_seq)
            if local_ts_ns < previous_local_ts:
                raise ResearchDatasetError(
                    f"{path}: local timestamp regressed at raw row {raw_seq}"
                )
            previous_local_ts = local_ts_ns
            data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
            if not isinstance(data, dict):
                raise ResearchDatasetError(f"{path}: invalid Binance payload at raw row {raw_seq}")
            event_type = str(data.get("e") or "")
            if not event_type:
                if data.get("lastUpdateId") is not None:
                    event_type = "snapshot"
                else:
                    event_type = "unknown"
            counts[event_type] += 1
            degraded_ids = _matching_degraded_interval_ids(
                intervals=intervals,
                segment_id=segment_id,
                track_id="binance",
                local_ts_ns=local_ts_ns,
            )
            connection_epoch_id = _connection_epoch_id(
                intervals=intervals,
                segment_id=segment_id,
                track_id="binance",
                local_ts_ns=local_ts_ns,
            )
            connection_epochs_seen.add(connection_epoch_id)
            if event_type in {"bookTicker", "trade", "depthUpdate"}:
                symbol = str(data.get("s") or data.get("ps") or "").upper()
                if symbol != expected_symbol.upper():
                    raise ResearchDatasetError(
                        f"{path}: expected symbol {expected_symbol}, got {symbol} "
                        f"at raw row {raw_seq}"
                    )
            elif event_type == "snapshot":
                if (
                    data.get("lastUpdateId") is None
                    or not isinstance(data.get("bids"), list)
                    or not isinstance(data.get("asks"), list)
                ):
                    raise ResearchDatasetError(
                        f"{path}: invalid Binance snapshot at raw row {raw_seq}"
                    )
            else:
                raise ResearchDatasetError(
                    f"{path}: unexpected Binance event type {event_type!r} "
                    f"at raw row {raw_seq}"
                )
            if event_type not in {"bookTicker", "trade"}:
                continue
            event_seq += 1
            row = {
                "segment_id": segment_id,
                "event_seq": event_seq,
                "source_raw_seq": raw_seq,
                "local_ts_ns": local_ts_ns,
                "exchange_ts_ns": _int_ns_from_ms(data.get("T") or data.get("E")),
                "event_type": event_type,
                "symbol": symbol,
                "connection_epoch_id": connection_epoch_id,
                "degraded": _bool_text(bool(degraded_ids)),
                "degraded_interval_ids": "|".join(degraded_ids),
                "update_id": data.get("u", ""),
            }
            if event_type == "bookTicker":
                row.update(
                    {
                        "bid_px": data.get("b", ""),
                        "bid_qty": data.get("B", ""),
                        "ask_px": data.get("a", ""),
                        "ask_qty": data.get("A", ""),
                    }
                )
            else:
                buyer_is_maker = data.get("m")
                if buyer_is_maker not in {True, False}:
                    raise ResearchDatasetError(
                        f"{path}: missing trade buyer-maker flag at raw row {raw_seq}"
                    )
                row.update(
                    {
                        "buyer_is_maker": _bool_text(buyer_is_maker),
                        "aggressor_side": "sell" if buyer_is_maker else "buy",
                        "trade_px": data.get("p", ""),
                        "trade_qty": data.get("q", ""),
                        "trade_id": data.get("t", ""),
                    }
                )
            yield row

    expected_exact = {
        str(event_type): _require_nonnegative_int(
            value,
            label=f"{path}:{event_type}:expected_count",
        )
        for event_type, value in expected_counts.items()
    }
    expected_exact["snapshot"] = _require_nonnegative_int(
        expected_snapshot_count,
        label=f"{path}:snapshot:expected_count",
    )
    if dict(sorted(counts.items())) != dict(sorted(expected_exact.items())):
        raise ResearchDatasetError(
            f"{path}: Binance event counts mismatch: "
            f"expected={dict(sorted(expected_exact.items()))}, "
            f"observed={dict(sorted(counts.items()))}"
        )
    observed.update(
        {
            "source_message_count_by_event_type": dict(sorted(counts.items())),
            "normalized_row_count": event_seq,
            "last_local_ts_ns": previous_local_ts,
            "connection_epoch_count": (
                max(connection_epochs_seen) + 1
                if connection_epochs_seen
                else 0
            ),
        }
    )


def _hyperliquid_hot_rows(
    *,
    path: Path,
    segment_id: str,
    expected_coin: str,
    expected_counts: dict[str, Any],
    expected_raw_row_count: Any,
    intervals: list[dict[str, Any]],
    observed: dict[str, Any],
) -> Iterable[dict[str, Any]]:
    counts: Counter[str] = Counter()
    previous_local_ts = -1
    event_seq = 0
    trade_item_count = 0
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for raw_seq, line in enumerate(fh, start=1):
            local_ts_ns, payload = _parse_raw_line(line, path=path, raw_seq=raw_seq)
            if local_ts_ns < previous_local_ts:
                raise ResearchDatasetError(
                    f"{path}: local timestamp regressed at raw row {raw_seq}"
                )
            previous_local_ts = local_ts_ns
            channel = str(payload.get("channel") or "")
            counts[channel] += 1
            data = payload.get("data")
            degraded_ids = _matching_degraded_interval_ids(
                intervals=intervals,
                segment_id=segment_id,
                track_id="hyperliquid_fast",
                local_ts_ns=local_ts_ns,
            )
            connection_epoch_id = _connection_epoch_id(
                intervals=intervals,
                segment_id=segment_id,
                track_id="hyperliquid_fast",
                local_ts_ns=local_ts_ns,
            )
            if channel == "bbo":
                if not isinstance(data, dict) or str(data.get("coin") or "") != expected_coin:
                    raise ResearchDatasetError(
                        f"{path}: invalid bbo coin at raw row {raw_seq}"
                    )
                bbo = data.get("bbo")
                if not isinstance(bbo, list) or len(bbo) != 2:
                    raise ResearchDatasetError(f"{path}: invalid bbo at raw row {raw_seq}")
                bid = bbo[0] if isinstance(bbo[0], dict) else {}
                ask = bbo[1] if isinstance(bbo[1], dict) else {}
                event_seq += 1
                yield {
                    "segment_id": segment_id,
                    "event_seq": event_seq,
                    "source_raw_seq": raw_seq,
                    "source_item_index": 0,
                    "local_ts_ns": local_ts_ns,
                    "exchange_ts_ns": _int_ns_from_ms(data.get("time")),
                    "event_type": "bbo",
                    "coin": expected_coin,
                    "connection_epoch_id": connection_epoch_id,
                    "degraded": _bool_text(bool(degraded_ids)),
                    "degraded_interval_ids": "|".join(degraded_ids),
                    "bid_px": bid.get("px", ""),
                    "bid_qty": bid.get("sz", ""),
                    "bid_n": bid.get("n", ""),
                    "ask_px": ask.get("px", ""),
                    "ask_qty": ask.get("sz", ""),
                    "ask_n": ask.get("n", ""),
                }
            elif channel == "trades":
                if not isinstance(data, list):
                    raise ResearchDatasetError(
                        f"{path}: invalid trades payload at raw row {raw_seq}"
                    )
                for item_index, trade in enumerate(data):
                    if not isinstance(trade, dict) or str(trade.get("coin") or "") != expected_coin:
                        raise ResearchDatasetError(
                            f"{path}: invalid trade coin at raw row {raw_seq}, item {item_index}"
                        )
                    event_seq += 1
                    trade_item_count += 1
                    yield {
                        "segment_id": segment_id,
                        "event_seq": event_seq,
                        "source_raw_seq": raw_seq,
                        "source_item_index": item_index,
                        "local_ts_ns": local_ts_ns,
                        "exchange_ts_ns": _int_ns_from_ms(trade.get("time")),
                        "event_type": "trade",
                        "coin": expected_coin,
                        "connection_epoch_id": connection_epoch_id,
                        "degraded": _bool_text(bool(degraded_ids)),
                        "degraded_interval_ids": "|".join(degraded_ids),
                        "trade_side": trade.get("side", ""),
                        "trade_px": trade.get("px", ""),
                        "trade_qty": trade.get("sz", ""),
                        "trade_id": trade.get("tid", ""),
                        "trade_hash": trade.get("hash", ""),
                        "trade_users_json": _compact_json(trade.get("users", [])),
                    }
            elif channel == "l2Book":
                if not isinstance(data, dict) or str(data.get("coin") or "") != expected_coin:
                    raise ResearchDatasetError(
                        f"{path}: invalid l2Book coin at raw row {raw_seq}"
                    )

    expected_exact = {
        str(channel): _require_nonnegative_int(
            value,
            label=f"{path}:{channel}:expected_count",
        )
        for channel, value in expected_counts.items()
    }
    if dict(sorted(counts.items())) != dict(sorted(expected_exact.items())):
        raise ResearchDatasetError(
            f"{path}: Hyperliquid channel counts mismatch: "
            f"expected={dict(sorted(expected_exact.items()))}, "
            f"observed={dict(sorted(counts.items()))}"
        )
    raw_row_count = sum(counts.values())
    expected_rows = _require_nonnegative_int(
        expected_raw_row_count,
        label=f"{path}:expected_raw_row_count",
    )
    if raw_row_count != expected_rows:
        raise ResearchDatasetError(
            f"{path}: raw row count mismatch: expected={expected_rows}, observed={raw_row_count}"
        )
    observed.update(
        {
            "source_message_count_by_channel": dict(sorted(counts.items())),
            "normalized_row_count": event_seq,
            "trade_item_count": trade_item_count,
            "last_local_ts_ns": previous_local_ts,
        }
    )


def _auxiliary_rows(
    *,
    sample_dir: Path,
    segment_id: str,
    expected_coin: str,
    tracks: dict[str, Any],
    track_manifests: dict[str, dict[str, Any]],
    intervals: list[dict[str, Any]],
    observed: dict[str, Any],
) -> Iterable[dict[str, Any]]:
    track_results: dict[str, Any] = {}
    target_dex = _expected_dex(expected_coin)

    def iter_track(track_id: str) -> Iterable[dict[str, Any]]:
        track = tracks.get(track_id)
        if not isinstance(track, dict):
            raise ResearchDatasetError(f"{segment_id}: missing research track {track_id}")
        track_manifest = track_manifests.get(track_id)
        if not isinstance(track_manifest, dict):
            raise ResearchDatasetError(
                f"{segment_id}: missing collection manifest for {track_id}"
            )
        path = _track_raw_path(sample_dir, track)
        counts: Counter[str] = Counter()
        previous_local_ts = -1
        normalized_rows = 0
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for raw_seq, line in enumerate(fh, start=1):
                local_ts_ns, payload = _parse_raw_line(line, path=path, raw_seq=raw_seq)
                if local_ts_ns < previous_local_ts:
                    raise ResearchDatasetError(
                        f"{path}: local timestamp regressed at raw row {raw_seq}"
                    )
                previous_local_ts = local_ts_ns
                channel = str(payload.get("channel") or "")
                counts[channel] += 1
                if channel in {
                    "subscriptionResponse",
                    "pong",
                    "ping",
                    "parse_error",
                    "transport_close",
                }:
                    continue
                if channel not in {"activeAssetCtx", "candle", "allMids"}:
                    raise ResearchDatasetError(
                        f"{path}: unexpected auxiliary channel {channel!r} "
                        f"at raw row {raw_seq}"
                    )
                data = payload.get("data")
                if not isinstance(data, dict):
                    raise ResearchDatasetError(
                        f"{path}: invalid {channel} payload at raw row {raw_seq}"
                    )
                degraded, interval_id = _matching_degraded_interval(
                    intervals=intervals,
                    segment_id=segment_id,
                    track_id=track_id,
                    local_ts_ns=local_ts_ns,
                )
                normalized_rows += 1
                row: dict[str, Any] = {
                    "segment_id": segment_id,
                    "track_id": track_id,
                    "source_raw_seq": raw_seq,
                    "local_ts_ns": local_ts_ns,
                    "event_type": channel,
                    "degraded": _bool_text(degraded),
                    "degraded_interval_id": interval_id,
                    "payload_json": _compact_json(payload),
                }
                if channel == "activeAssetCtx":
                    coin = str(data.get("coin") or "")
                    if coin != expected_coin:
                        raise ResearchDatasetError(
                            f"{path}: expected coin {expected_coin}, got {coin}"
                        )
                    ctx = data.get("ctx")
                    if not isinstance(ctx, dict):
                        raise ResearchDatasetError(f"{path}: invalid activeAssetCtx ctx")
                    impact = ctx.get("impactPxs")
                    impact = impact if isinstance(impact, list) else []
                    row.update(
                        {
                            "coin": coin,
                            "mark_px": ctx.get("markPx", ""),
                            "oracle_px": ctx.get("oraclePx", ""),
                            "mid_px": ctx.get("midPx", ""),
                            "funding": ctx.get("funding", ""),
                            "premium": ctx.get("premium", ""),
                            "open_interest": ctx.get("openInterest", ""),
                            "impact_bid_px": impact[0] if len(impact) > 0 else "",
                            "impact_ask_px": impact[1] if len(impact) > 1 else "",
                        }
                    )
                elif channel == "candle":
                    coin = str(data.get("s") or "")
                    if coin != expected_coin:
                        raise ResearchDatasetError(
                            f"{path}: expected candle coin {expected_coin}, got {coin}"
                        )
                    row.update(
                        {
                            "coin": coin,
                            "exchange_ts_ns": _int_ns_from_ms(data.get("T") or data.get("t")),
                            "candle_open": data.get("o", ""),
                            "candle_high": data.get("h", ""),
                            "candle_low": data.get("l", ""),
                            "candle_close": data.get("c", ""),
                            "candle_volume": data.get("v", ""),
                            "candle_trade_count": data.get("n", ""),
                        }
                    )
                elif channel == "allMids":
                    dex = str(data.get("dex") or "")
                    if track_id == "main_all_mids" and dex:
                        raise ResearchDatasetError(
                            f"{path}: main allMids must not carry named dex {dex}"
                        )
                    if track_id == "target_dex_all_mids" and dex != target_dex:
                        raise ResearchDatasetError(
                            f"{path}: expected target dex {target_dex}, got {dex}"
                        )
                    mids = data.get("mids")
                    if not isinstance(mids, dict):
                        raise ResearchDatasetError(f"{path}: invalid allMids map")
                    row.update(
                        {
                            "coin": expected_coin if track_id == "target_dex_all_mids" else "",
                            "target_mid_px": mids.get(expected_coin, ""),
                        }
                    )
                yield row

        expected_counts = _combined_channel_counts(
            track_manifest,
            label=f"{segment_id}:{track_id}:collection_manifest",
        )
        if dict(sorted(counts.items())) != expected_counts:
            raise ResearchDatasetError(
                f"{path}: Hyperliquid channel counts mismatch: "
                f"expected={expected_counts}, observed={dict(sorted(counts.items()))}"
            )
        expected_raw_rows = _require_nonnegative_int(
            track_manifest.get("raw_row_count"),
            label=f"{path}:expected_raw_row_count",
        )
        if sum(counts.values()) != expected_raw_rows:
            raise ResearchDatasetError(
                f"{path}: raw row count mismatch: "
                f"expected={expected_raw_rows}, observed={sum(counts.values())}"
            )
        track_results[track_id] = {
            "source_message_count_by_channel": dict(sorted(counts.items())),
            "normalized_row_count": normalized_rows,
            "last_local_ts_ns": previous_local_ts,
        }

    iterators = {
        track_id: iter(iter_track(track_id))
        for track_id in AUXILIARY_TRACKS
    }
    heap: list[tuple[int, int, int, dict[str, Any]]] = []
    track_priority = {
        track_id: index
        for index, track_id in enumerate(AUXILIARY_TRACKS)
    }
    for track_id, iterator in iterators.items():
        try:
            row = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(
            heap,
            (
                int(row["local_ts_ns"]),
                track_priority[track_id],
                int(row["source_raw_seq"]),
                row,
            ),
        )

    event_seq = 0
    while heap:
        _, _, _, row = heapq.heappop(heap)
        event_seq += 1
        row["event_seq"] = event_seq
        yield row
        track_id = str(row["track_id"])
        try:
            next_row = next(iterators[track_id])
        except StopIteration:
            continue
        heapq.heappush(
            heap,
            (
                int(next_row["local_ts_ns"]),
                track_priority[track_id],
                int(next_row["source_raw_seq"]),
                next_row,
            ),
        )

    observed.update(
        {
            "tracks": track_results,
            "normalized_row_count": event_seq,
        }
    )


def _validate_hyperliquid_l2_raw(
    *,
    path: Path,
    expected_coin: str,
    expected_counts: dict[str, Any],
    expected_raw_row_count: Any,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    previous_local_ts = -1
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for raw_seq, line in enumerate(fh, start=1):
            local_ts_ns, payload = _parse_raw_line(line, path=path, raw_seq=raw_seq)
            if local_ts_ns < previous_local_ts:
                raise ResearchDatasetError(
                    f"{path}: local timestamp regressed at raw row {raw_seq}"
                )
            previous_local_ts = local_ts_ns
            channel = str(payload.get("channel") or "")
            counts[channel] += 1
            if channel in {
                "subscriptionResponse",
                "pong",
                "ping",
                "parse_error",
                "transport_close",
            }:
                continue
            if channel != "l2Book":
                raise ResearchDatasetError(
                    f"{path}: unexpected standard-L2 channel {channel!r} "
                    f"at raw row {raw_seq}"
                )
            data = payload.get("data")
            if not isinstance(data, dict) or str(data.get("coin") or "") != expected_coin:
                raise ResearchDatasetError(
                    f"{path}: invalid standard-L2 coin at raw row {raw_seq}"
                )
            levels = data.get("levels")
            if not isinstance(levels, list) or len(levels) != 2:
                raise ResearchDatasetError(
                    f"{path}: invalid standard-L2 levels at raw row {raw_seq}"
                )

    expected_exact = {
        str(channel): _require_nonnegative_int(
            value,
            label=f"{path}:{channel}:expected_count",
        )
        for channel, value in expected_counts.items()
    }
    observed_exact = dict(sorted(counts.items()))
    if observed_exact != dict(sorted(expected_exact.items())):
        raise ResearchDatasetError(
            f"{path}: Hyperliquid channel counts mismatch: "
            f"expected={dict(sorted(expected_exact.items()))}, observed={observed_exact}"
        )
    expected_rows = _require_nonnegative_int(
        expected_raw_row_count,
        label=f"{path}:expected_raw_row_count",
    )
    if sum(counts.values()) != expected_rows:
        raise ResearchDatasetError(
            f"{path}: raw row count mismatch: "
            f"expected={expected_rows}, observed={sum(counts.values())}"
        )
    return {
        "source_message_count_by_channel": observed_exact,
        "raw_row_count": sum(counts.values()),
        "last_local_ts_ns": previous_local_ts,
    }


def _segment_mask_rows(
    *,
    campaign_id: str,
    profile_id: str,
    index_row: dict[str, str],
    intervals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    segment_id = index_row["segment_id"]
    common = {
        "campaign_id": campaign_id,
        "segment_id": segment_id,
        "profile_id": profile_id,
        "first_common_ts_ns": index_row["first_common_ts_ns"],
        "last_common_ts_ns": index_row["last_common_ts_ns"],
        "previous_segment_gap_ms": index_row["previous_segment_gap_ms"],
        "cross_segment_continuity_claimed": str(
            index_row["cross_segment_continuity_claimed"]
        ).lower(),
    }
    rows = [
        {
            **common,
            "mask_type": "segment_epoch",
            "policy": "never_compute_across_segment_boundary",
            "reason": "fresh_exchange_snapshots",
        }
    ]
    for index, interval in enumerate(intervals):
        if interval.get("segment_id") != segment_id:
            continue
        is_fast_staleness = (
            interval.get("track_id") == "hyperliquid_fast"
            and interval.get("reason") == "source_age_exceeds_limit"
        )
        is_core_reconnect = (
            interval.get("track_id")
            in {"binance", "hyperliquid_fast", "hyperliquid_standard"}
            and interval.get("reason") == "websocket_reconnect"
        )
        rows.append(
            {
                **common,
                "mask_type": (
                    "core_l2_reconnect_interval"
                    if is_core_reconnect
                    else (
                        "fast_l2_staleness_interval"
                        if is_fast_staleness
                        else "auxiliary_degraded_interval"
                    )
                ),
                "track_id": interval.get("track_id", ""),
                "mask_start_ts_ns": interval.get("degraded_start_local_ts_ns", ""),
                "mask_end_ts_ns": interval.get("recovered_local_ts_ns", ""),
                "duration_ms": interval.get("duration_ms", ""),
                "policy": interval.get("policy", ""),
                "reason": (
                    f"{interval.get('reason', '')}:"
                    f"{_degraded_interval_id(interval, index)}"
                ),
            }
        )
    return rows


def _prepare_output(output_dir: Path, *, clean_output: bool) -> Path:
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    if output_dir.exists():
        if any(output_dir.iterdir()) and not clean_output:
            raise ResearchDatasetError(
                f"output directory is nonempty: {output_dir}; pass --clean-output to replace it"
            )
    temporary.mkdir(parents=True)
    return temporary


def _publish_output(temporary_output: Path, output_dir: Path) -> None:
    if not output_dir.exists():
        os.replace(temporary_output, output_dir)
        return
    backup = output_dir.with_name(f"{output_dir.name}.backup-{os.getpid()}")
    if backup.exists():
        shutil.rmtree(backup)
    os.replace(output_dir, backup)
    try:
        os.replace(temporary_output, output_dir)
    except Exception:
        os.replace(backup, output_dir)
        raise
    shutil.rmtree(backup)


def _read_timeline_index(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ResearchDatasetError(f"{path}: empty timeline index")
    return rows


def _validate_segment_sources(
    *,
    campaign_dir: Path,
    expected_campaign_id: str,
    profile_id: str,
    index_row: dict[str, str],
    expected_binance_symbol: str,
    expected_coin: str,
) -> dict[str, Any]:
    segment_id = index_row["segment_id"]
    if index_row.get("campaign_id") != expected_campaign_id:
        raise ResearchDatasetError(f"{segment_id}: timeline index campaign mismatch")
    if _require_bool(
        index_row.get("cross_segment_continuity_claimed"),
        label=f"{segment_id}:timeline_index:cross_segment_continuity_claimed",
    ):
        raise ResearchDatasetError(f"{segment_id}: timeline index claims continuity")
    profile_dir = campaign_dir / "segments" / segment_id / profile_id
    sample_dir = profile_dir / "sample"
    strict_quality = _read_json(profile_dir / "strict_quality.json")
    if strict_quality.get("passes") is not True:
        raise ResearchDatasetError(f"{segment_id}: strict quality did not pass")
    strict_intervals = strict_quality.get("degraded_intervals", [])
    if not isinstance(strict_intervals, list):
        raise ResearchDatasetError(f"{segment_id}: invalid strict degraded intervals")
    timeline_manifest = _read_json(profile_dir / "common_l2_timeline_manifest.json")
    if timeline_manifest.get("passes") is not True:
        raise ResearchDatasetError(f"{segment_id}: common timeline did not pass")
    if timeline_manifest.get("campaign_id") != expected_campaign_id:
        raise ResearchDatasetError(f"{segment_id}: timeline campaign mismatch")
    if timeline_manifest.get("segment_id") != segment_id:
        raise ResearchDatasetError(f"{segment_id}: timeline segment mismatch")
    if timeline_manifest.get("profile_id") != profile_id:
        raise ResearchDatasetError(f"{segment_id}: timeline profile mismatch")
    if timeline_manifest.get("binance_symbol") != expected_binance_symbol:
        raise ResearchDatasetError(f"{segment_id}: timeline Binance symbol mismatch")
    if timeline_manifest.get("hyperliquid_coin") != expected_coin:
        raise ResearchDatasetError(f"{segment_id}: timeline coin mismatch")
    segment_boundary = timeline_manifest.get("segment_boundary")
    if not isinstance(segment_boundary, dict) or _require_bool(
        segment_boundary.get("cross_segment_continuity_claimed"),
        label=f"{segment_id}:timeline_manifest:cross_segment_continuity_claimed",
    ):
        raise ResearchDatasetError(f"{segment_id}: timeline manifest claims continuity")
    timeline_path = profile_dir / "common_l2_timeline.csv.gz"
    timeline_sha = _require_hash(
        timeline_path,
        timeline_manifest.get("timeline_sha256"),
        label=f"{segment_id}:timeline_manifest",
    )
    if timeline_sha != index_row.get("timeline_sha256"):
        raise ResearchDatasetError(f"{segment_id}: timeline index SHA mismatch")
    if int(index_row["row_count"]) != int(timeline_manifest["timeline_row_count"]):
        raise ResearchDatasetError(f"{segment_id}: timeline row-count mismatch")
    timeline_summary = _timeline_csv_summary(
        timeline_path,
        expected_campaign_id=expected_campaign_id,
        expected_segment_id=segment_id,
        expected_profile_id=profile_id,
    )
    expected_timeline_summary = {
        "row_count": _require_nonnegative_int(
            timeline_manifest.get("timeline_row_count"),
            label=f"{segment_id}:timeline_manifest:timeline_row_count",
        ),
        "first_common_ts_ns": _require_nonnegative_int(
            timeline_manifest.get("first_common_ts_ns"),
            label=f"{segment_id}:timeline_manifest:first_common_ts_ns",
        ),
        "last_common_ts_ns": _require_nonnegative_int(
            timeline_manifest.get("last_common_ts_ns"),
            label=f"{segment_id}:timeline_manifest:last_common_ts_ns",
        ),
    }
    if timeline_summary != expected_timeline_summary:
        raise ResearchDatasetError(
            f"{segment_id}: timeline CSV/manifest boundary mismatch: "
            f"manifest={expected_timeline_summary}, csv={timeline_summary}"
        )
    index_timeline_summary = {
        "row_count": _require_nonnegative_int(
            index_row.get("row_count"),
            label=f"{segment_id}:timeline_index:row_count",
        ),
        "first_common_ts_ns": _require_nonnegative_int(
            index_row.get("first_common_ts_ns"),
            label=f"{segment_id}:timeline_index:first_common_ts_ns",
        ),
        "last_common_ts_ns": _require_nonnegative_int(
            index_row.get("last_common_ts_ns"),
            label=f"{segment_id}:timeline_index:last_common_ts_ns",
        ),
    }
    if timeline_summary != index_timeline_summary:
        raise ResearchDatasetError(
            f"{segment_id}: timeline CSV/index boundary mismatch: "
            f"index={index_timeline_summary}, csv={timeline_summary}"
        )

    binance_manifest = _read_json(sample_dir / "binance_public_raw" / "collection_manifest.json")
    hyperliquid_manifest = _read_json(
        sample_dir / "hyperliquid_public_sample" / "collection_manifest.json"
    )
    bundle = _read_json(
        sample_dir / "hyperliquid_public_sample" / "research_bundle_manifest.json"
    )
    if (
        bundle.get("quality", {}).get("all_tracks_pass") is not True
        and strict_quality.get("segmented_replay_eligible") is not True
    ):
        raise ResearchDatasetError(
            f"{segment_id}: research bundle failed without recovered replay eligibility"
        )
    tracks = bundle.get("tracks")
    if not isinstance(tracks, dict):
        raise ResearchDatasetError(f"{segment_id}: invalid research tracks")

    sources: dict[str, Any] = {
        "timeline": {
            "path": timeline_path,
            "sha256": timeline_sha,
            "row_count": int(timeline_manifest["timeline_row_count"]),
        }
    }
    binance_path = sample_dir / "binance_public_raw" / "raw.gz"
    sources["binance"] = {
        "path": binance_path,
        "sha256": _require_hash(
            binance_path,
            binance_manifest.get("raw_sha256"),
            label=f"{segment_id}:binance",
        ),
    }
    fast_path = sample_dir / "hyperliquid_public_sample" / "raw.gz"
    sources["hyperliquid_fast"] = {
        "path": fast_path,
        "sha256": _require_hash(
            fast_path,
            hyperliquid_manifest.get("raw_sha256"),
            label=f"{segment_id}:hyperliquid_fast",
        ),
    }
    track_manifests: dict[str, dict[str, Any]] = {}
    for track_id, track in tracks.items():
        if not isinstance(track, dict):
            raise ResearchDatasetError(f"{segment_id}:{track_id}: invalid track")
        raw_path = _track_raw_path(sample_dir, track)
        computed = _require_hash(
            raw_path,
            track.get("raw_sha256"),
            label=f"{segment_id}:{track_id}",
        )
        manifest_path = raw_path.parent / "collection_manifest.json"
        collection_manifest = _read_json(manifest_path)
        if collection_manifest.get("raw_row_count_reconciled") is not True:
            raise ResearchDatasetError(
                f"{segment_id}:{track_id}: raw row count was not reconciled"
            )
        parse_error_count = _require_nonnegative_int(
            collection_manifest.get("parse_error_count"),
            label=f"{segment_id}:{track_id}:parse_error_count",
        )
        effective_quality = strict_quality.get("effective_track_quality", {}).get(
            track_id, {}
        )
        if effective_quality and effective_quality.get("passes") is not True:
            raise ResearchDatasetError(
                f"{segment_id}:{track_id}: effective track quality did not pass"
            )
        if parse_error_count != 0:
            matching_evidence = [
                interval.get("transport_marker_evidence", {})
                for interval in strict_intervals
                if interval.get("source_track_id") == track_id
            ]
            if not matching_evidence or any(
                int(evidence.get("recorded_parse_error_count", -1))
                != parse_error_count
                or int(evidence.get("effective_parse_error_count", -1)) != 0
                or evidence.get("passes") is not True
                for evidence in matching_evidence
            ):
                raise ResearchDatasetError(
                    f"{segment_id}:{track_id}: unreconciled parse errors present"
                )
        if computed != str(collection_manifest.get("raw_sha256") or ""):
            raise ResearchDatasetError(
                f"{segment_id}:{track_id}: bundle/collection SHA mismatch"
            )
        bundle_counts = track.get("message_count_by_channel")
        manifest_counts = collection_manifest.get("message_count_by_channel")
        if bundle_counts != manifest_counts:
            raise ResearchDatasetError(
                f"{segment_id}:{track_id}: bundle/collection channel-count mismatch"
            )
        track_manifests[track_id] = collection_manifest
        if track_id == "fast_market":
            if computed != sources["hyperliquid_fast"]["sha256"]:
                raise ResearchDatasetError(f"{segment_id}: fast-market SHA mismatch")
            continue
        sources[track_id] = {"path": raw_path, "sha256": computed}
    for required in ("standard_l2", *AUXILIARY_TRACKS):
        if required not in sources:
            raise ResearchDatasetError(f"{segment_id}: missing source track {required}")
    source_raw = timeline_manifest.get("source_raw")
    if not isinstance(source_raw, dict):
        raise ResearchDatasetError(f"{segment_id}: missing timeline source_raw")
    timeline_source_map = {
        "binance": "binance",
        "hyperliquid_fast": "hyperliquid_fast",
        "hyperliquid_standard": "standard_l2",
    }
    for timeline_source_id, source_id in timeline_source_map.items():
        timeline_source = source_raw.get(timeline_source_id)
        if not isinstance(timeline_source, dict):
            raise ResearchDatasetError(
                f"{segment_id}: missing timeline source_raw {timeline_source_id}"
            )
        if timeline_source.get("sha256") != sources[source_id]["sha256"]:
            raise ResearchDatasetError(
                f"{segment_id}: timeline source-raw SHA mismatch for {timeline_source_id}"
            )
    strict_core_intervals = [
        interval
        for interval in strict_intervals
        if interval.get("track_class") == "core"
    ]
    timeline_core_intervals = timeline_manifest.get("reconnect_intervals", [])
    if timeline_core_intervals != strict_core_intervals:
        raise ResearchDatasetError(
            f"{segment_id}: timeline/core reconnect interval mismatch"
        )

    standard_manifest = track_manifests["standard_l2"]
    standard_l2_validation = _validate_hyperliquid_l2_raw(
        path=sources["standard_l2"]["path"],
        expected_coin=expected_coin,
        expected_counts=_combined_channel_counts(
            standard_manifest,
            label=f"{segment_id}:standard_l2:collection_manifest",
        ),
        expected_raw_row_count=standard_manifest.get("raw_row_count"),
    )

    return {
        "segment_id": segment_id,
        "profile_dir": profile_dir,
        "sample_dir": sample_dir,
        "strict_quality": strict_quality,
        "timeline_manifest": timeline_manifest,
        "binance_manifest": binance_manifest,
        "hyperliquid_manifest": hyperliquid_manifest,
        "bundle": bundle,
        "tracks": tracks,
        "track_manifests": track_manifests,
        "standard_l2_validation": standard_l2_validation,
        "sources": sources,
    }


def build_research_dataset(
    *,
    campaign_dir: Path,
    output_dir: Path,
    profile_id: str,
    task_id: str = TASK_ID,
    clean_output: bool = False,
) -> dict[str, Any]:
    campaign_dir = campaign_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    campaign_manifest_path = campaign_dir / "campaign_manifest.json"
    timeline_index_path = campaign_dir / "timeline_index.csv"
    campaign_manifest = _read_json(campaign_manifest_path)
    if campaign_manifest.get("passes") is not True:
        raise ResearchDatasetError("campaign manifest did not pass")
    if profile_id not in campaign_manifest.get("profiles", []):
        raise ResearchDatasetError(f"campaign does not contain profile {profile_id}")
    if campaign_manifest.get("cross_segment_continuity_claimed") is not False:
        raise ResearchDatasetError("campaign must not claim cross-segment continuity")

    profile = get_symbol_profile(profile_id)
    index_rows = _read_timeline_index(timeline_index_path)
    expected_segments = [
        str(item.get("segment_id"))
        for item in campaign_manifest.get("segments", [])
        if isinstance(item, dict)
    ]
    observed_segments = [row["segment_id"] for row in index_rows]
    if observed_segments != expected_segments:
        raise ResearchDatasetError(
            f"timeline segment order mismatch: expected={expected_segments}, observed={observed_segments}"
        )
    if any(row.get("profile_id") != profile_id for row in index_rows):
        raise ResearchDatasetError("timeline index profile mismatch")
    intervals = campaign_manifest.get("degraded_intervals", [])
    if not isinstance(intervals, list):
        raise ResearchDatasetError("invalid degraded intervals")

    validated_segments = [
        _validate_segment_sources(
            campaign_dir=campaign_dir,
            expected_campaign_id=str(campaign_manifest["campaign_id"]),
            profile_id=profile_id,
            index_row=index_row,
            expected_binance_symbol=profile.binance_symbol,
            expected_coin=profile.hyperliquid_coin,
        )
        for index_row in index_rows
    ]
    previous_last_common_ts_ns: int | None = None
    for index_row in index_rows:
        first_common_ts_ns = int(index_row["first_common_ts_ns"])
        if previous_last_common_ts_ns is None:
            if index_row.get("previous_segment_gap_ms") not in {"", None}:
                raise ResearchDatasetError(
                    f"{index_row['segment_id']}: first segment must not have a previous gap"
                )
        else:
            expected_gap_ms = (
                first_common_ts_ns - previous_last_common_ts_ns
            ) / 1_000_000
            try:
                indexed_gap_ms = float(index_row["previous_segment_gap_ms"])
            except (TypeError, ValueError) as exc:
                raise ResearchDatasetError(
                    f"{index_row['segment_id']}: invalid previous segment gap"
                ) from exc
            if not math.isclose(
                expected_gap_ms,
                indexed_gap_ms,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ResearchDatasetError(
                    f"{index_row['segment_id']}: previous segment gap mismatch: "
                    f"expected={expected_gap_ms}, indexed={indexed_gap_ms}"
                )
        previous_last_common_ts_ns = int(index_row["last_common_ts_ns"])
    initial_source_hashes = {
        f"{segment['segment_id']}:{source_id}": source["sha256"]
        for segment in validated_segments
        for source_id, source in segment["sources"].items()
    }

    temporary_output = _prepare_output(output_dir, clean_output=clean_output)
    segment_manifests: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    aggregate: Counter[str] = Counter()
    try:
        for index_row, segment in zip(index_rows, validated_segments):
            segment_id = segment["segment_id"]
            segment_output = temporary_output / "segments" / segment_id
            segment_output.mkdir(parents=True)

            binance_observed: dict[str, Any] = {}
            binance_output = segment_output / "binance_hot_events.csv.gz"
            binance_rows = _write_csv(
                binance_output,
                _binance_rows(
                    path=segment["sources"]["binance"]["path"],
                    segment_id=segment_id,
                    expected_symbol=profile.binance_symbol,
                    expected_counts=segment["binance_manifest"].get(
                        "message_count_by_event_type", {}
                    ),
                    expected_snapshot_count=segment["binance_manifest"].get(
                        "depth_snapshot_bridge_count"
                    ),
                    intervals=intervals,
                    observed=binance_observed,
                ),
                BINANCE_HOT_FIELDS,
            )
            if binance_rows != binance_observed.get("normalized_row_count"):
                raise ResearchDatasetError(f"{segment_id}: Binance normalized count mismatch")
            expected_binance_epoch_count = int(
                segment["timeline_manifest"]
                .get("connection_epoch_count_by_track", {})
                .get("binance", 0)
            )
            if (
                binance_observed.get("connection_epoch_count")
                != expected_binance_epoch_count
            ):
                raise ResearchDatasetError(
                    f"{segment_id}: Binance connection epoch count mismatch: "
                    f"expected={expected_binance_epoch_count}, "
                    f"observed={binance_observed.get('connection_epoch_count')}"
                )

            hyperliquid_observed: dict[str, Any] = {}
            hyperliquid_output = segment_output / "hyperliquid_hot_events.csv.gz"
            hyperliquid_rows = _write_csv(
                hyperliquid_output,
                _hyperliquid_hot_rows(
                    path=segment["sources"]["hyperliquid_fast"]["path"],
                    segment_id=segment_id,
                    expected_coin=profile.hyperliquid_coin,
                    expected_counts=_combined_channel_counts(
                        segment["hyperliquid_manifest"],
                        label=f"{segment_id}:fast_market:collection_manifest",
                    ),
                    expected_raw_row_count=segment["hyperliquid_manifest"].get(
                        "raw_row_count"
                    ),
                    intervals=intervals,
                    observed=hyperliquid_observed,
                ),
                HYPERLIQUID_HOT_FIELDS,
            )
            if hyperliquid_rows != hyperliquid_observed.get("normalized_row_count"):
                raise ResearchDatasetError(
                    f"{segment_id}: Hyperliquid normalized count mismatch"
                )

            auxiliary_observed: dict[str, Any] = {}
            auxiliary_output = segment_output / "hyperliquid_auxiliary_events.csv.gz"
            auxiliary_rows = _write_csv(
                auxiliary_output,
                _auxiliary_rows(
                    sample_dir=segment["sample_dir"],
                    segment_id=segment_id,
                    expected_coin=profile.hyperliquid_coin,
                    tracks=segment["tracks"],
                    track_manifests=segment["track_manifests"],
                    intervals=intervals,
                    observed=auxiliary_observed,
                ),
                AUXILIARY_FIELDS,
            )
            if auxiliary_rows != auxiliary_observed.get("normalized_row_count"):
                raise ResearchDatasetError(
                    f"{segment_id}: auxiliary normalized count mismatch"
                )

            current_masks = _segment_mask_rows(
                campaign_id=str(campaign_manifest["campaign_id"]),
                profile_id=profile_id,
                index_row=index_row,
                intervals=intervals,
            )
            mask_rows.extend(current_masks)
            segment_manifest = {
                "schema_version": SCHEMA_VERSION,
                "task_id": task_id,
                "campaign_id": campaign_manifest["campaign_id"],
                "segment_id": segment_id,
                "profile_id": profile_id,
                "symbols": {
                    "binance": profile.binance_symbol,
                    "hyperliquid": profile.hyperliquid_coin,
                },
                "clock_policy": {
                    "join_clock": "same_host_local_receipt_time_time_ns",
                    "exchange_timestamps": "diagnostic_only",
                    "future_decision_joins_allowed": False,
                },
                "segment_boundary": {
                    "first_common_ts_ns": int(index_row["first_common_ts_ns"]),
                    "last_common_ts_ns": int(index_row["last_common_ts_ns"]),
                    "previous_segment_gap_ms": (
                        float(index_row["previous_segment_gap_ms"])
                        if index_row["previous_segment_gap_ms"]
                        else None
                    ),
                    "cross_segment_continuity_claimed": False,
                },
                "source_files": {
                    source_id: {
                        "path": str(source["path"]),
                        "sha256": source["sha256"],
                        **(
                            {"row_count": source["row_count"]}
                            if source.get("row_count") is not None
                            else {}
                        ),
                    }
                    for source_id, source in segment["sources"].items()
                },
                "outputs": {
                    "binance_hot_events": {
                        "path": str(binance_output.relative_to(temporary_output)),
                        "row_count": binance_rows,
                        "sha256": sha256_file(binance_output),
                    },
                    "hyperliquid_hot_events": {
                        "path": str(hyperliquid_output.relative_to(temporary_output)),
                        "row_count": hyperliquid_rows,
                        "sha256": sha256_file(hyperliquid_output),
                    },
                    "hyperliquid_auxiliary_events": {
                        "path": str(auxiliary_output.relative_to(temporary_output)),
                        "row_count": auxiliary_rows,
                        "sha256": sha256_file(auxiliary_output),
                    },
                },
                "reconciliation": {
                    "binance": binance_observed,
                    "hyperliquid_fast": hyperliquid_observed,
                    "hyperliquid_standard_l2": segment["standard_l2_validation"],
                    "hyperliquid_auxiliary": auxiliary_observed,
                    "mask_row_count": len(current_masks),
                    "passes": True,
                },
                "connection_epochs": segment["timeline_manifest"].get(
                    "connection_epoch_count_by_track", {}
                ),
                "capability_boundary": {
                    "common_l2_is_replayed_state_source": True,
                    "continuous_exact_replay": segment["timeline_manifest"].get(
                        "capability_boundary", {}
                    ).get("continuous_exact_replay"),
                    "segmented_replay_eligible": True,
                    "old_l2_state_forward_filled_across_reconnect": False,
                    "normalized_hot_events_are_source_sidecars": True,
                    "original_raw_is_information_complete_source": True,
                    "l3_l4_queue_reconstruction": False,
                    "exact_fill_simulation": False,
                    "signal_fitted": False,
                    "new_collection_performed": False,
                },
                "passes": True,
            }
            segment_manifest_path = segment_output / "segment_event_store_manifest.json"
            _write_json(segment_manifest_path, segment_manifest)
            segment_manifest["manifest_path"] = str(
                segment_manifest_path.relative_to(temporary_output)
            )
            segment_manifest["manifest_sha256"] = sha256_file(segment_manifest_path)
            segment_manifests.append(segment_manifest)
            aggregate["timeline_rows"] += int(index_row["row_count"])
            aggregate["binance_hot_rows"] += binance_rows
            aggregate["hyperliquid_hot_rows"] += hyperliquid_rows
            aggregate["hyperliquid_trade_items"] += int(
                hyperliquid_observed["trade_item_count"]
            )
            aggregate["hyperliquid_auxiliary_rows"] += auxiliary_rows

        mask_index_path = temporary_output / "segment_and_mask_index.csv"
        mask_row_count = _write_csv(mask_index_path, mask_rows, MASK_INDEX_FIELDS)
        aggregate["mask_rows"] = mask_row_count
        final_source_hashes = {
            f"{segment['segment_id']}:{source_id}": sha256_file(source["path"])
            for segment in validated_segments
            for source_id, source in segment["sources"].items()
        }
        if final_source_hashes != initial_source_hashes:
            raise ResearchDatasetError("source files changed while building the research dataset")

        runtime_source_path = Path(__file__).resolve()
        runtime_archive_path = (
            temporary_output / "runtime_source" / runtime_source_path.name
        )
        runtime_archive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(runtime_source_path, runtime_archive_path)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "task_id": task_id,
            "source_tasks": {
                "collection": "0729T010",
                "postprocess": "0730T011",
                "research_plan": "0730T012",
            },
            "source_status": {
                "collection_postprocess_business_complete": True,
                "postprocess_independent_qa_status": "pending",
                "research_plan_qa_status": "passed",
            },
            "campaign_id": campaign_manifest["campaign_id"],
            "campaign_dir": str(campaign_dir),
            "campaign_manifest": {
                "path": str(campaign_manifest_path),
                "sha256": sha256_file(campaign_manifest_path),
            },
            "timeline_index": {
                "path": str(timeline_index_path),
                "sha256": sha256_file(timeline_index_path),
                "row_count": len(index_rows),
            },
            "profile_id": profile_id,
            "symbols": {
                "binance": profile.binance_symbol,
                "hyperliquid": profile.hyperliquid_coin,
            },
            "segment_count": len(segment_manifests),
            "degraded_interval_count": len(intervals),
            "degraded_intervals": intervals,
            "segment_and_mask_index": {
                "path": str(mask_index_path.relative_to(temporary_output)),
                "row_count": mask_row_count,
                "sha256": sha256_file(mask_index_path),
            },
            "aggregate_counts": dict(sorted(aggregate.items())),
            "source_hash_count": len(initial_source_hashes),
            "source_hashes_unchanged": True,
            "runtime_source": {
                "builder": {
                    "path": str(runtime_source_path),
                    "sha256": sha256_file(runtime_source_path),
                    "bytes": runtime_source_path.stat().st_size,
                    "archive_path": str(
                        runtime_archive_path.relative_to(temporary_output)
                    ),
                    "archive_sha256": sha256_file(runtime_archive_path),
                }
            },
            "segments": [
                {
                    "segment_id": item["segment_id"],
                    "manifest_path": item["manifest_path"],
                    "manifest_sha256": item["manifest_sha256"],
                    "outputs": item["outputs"],
                }
                for item in segment_manifests
            ],
            "boundary": {
                "local_existing_data_only": True,
                "network_accessed": False,
                "aws_accessed": False,
                "ssh_accessed": False,
                "new_collection_performed": False,
                "private_endpoints_accessed": False,
                "order_endpoints_accessed": False,
                "feature_fitting_performed": False,
                "parameter_search_performed": False,
                "signal_conclusion_allowed": False,
                "arbitrage_conclusion_allowed": False,
                "exact_fill_claim_allowed": False,
                "additional_collection_requires_explicit_user_authorization": True,
                "additional_collection_requires_user_confirmed_active_trading_window": True,
            },
            "passes": True,
        }
        manifest_path = temporary_output / "research_input_manifest.json"
        _write_json(manifest_path, manifest)
        _publish_output(temporary_output, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary_output, ignore_errors=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--symbol-profile", choices=available_profile_ids(), default="skhynix")
    parser.add_argument("--task-id", default=TASK_ID)
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build_research_dataset(
            campaign_dir=Path(args.campaign_dir),
            output_dir=Path(args.output_dir),
            profile_id=args.symbol_profile,
            task_id=args.task_id,
            clean_output=args.clean_output,
        )
    except (ResearchDatasetError, OSError, ValueError) as exc:
        print(json.dumps({"passes": False, "error": str(exc)}, indent=2, sort_keys=True))
        return 4
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
