# Unified Latency Observability Contract

## Purpose

This contract gives Binance and Hyperliquid one versioned latency vocabulary.
It is the measurement foundation for live/backtest alignment, tick-to-order
optimization, instance benchmarking, and later order-lifecycle analysis.

The first implementation is a public-data Rust probe. It does not submit or
cancel orders and does not claim exchange ACK, resting, fill, or cancel latency.

## Clock Domains

Each trace may contain two timestamp arrays:

- `unix_ns`: Unix epoch nanoseconds. Use only for exchange timestamp to local
  arrival comparisons.
- `monotonic_ns`: process-local monotonic nanoseconds. Use for all local
  pipeline and request-lifecycle intervals.

The rules are strict:

1. Never subtract a monotonic timestamp from a Unix timestamp.
2. Never use wall-clock timestamps for process-local stage durations.
3. `feed_network` is an observed clock-offset-sensitive metric, not a pure
   one-way network measurement.
4. NTP/PTP state and host identity belong in run metadata for production
   benchmarks.

## Stages

Schema version `latency-trace-v1` reserves these ordered stages:

1. `exchange_event`
2. `socket_receive`
3. `ws_frame_receive`
4. `parse_done`
5. `book_apply_done`
6. `signal_start`
7. `signal_end`
8. `decision`
9. `order_encode_done`
10. `socket_send`
11. `exchange_ack`
12. `resting`
13. `fill`
14. `cancel_request`
15. `cancel_ack`

A missing observation is represented by a missing/sentinel timestamp and is
counted as `missing`. A negative duration within one valid clock domain is
counted as `invalid`; it is never folded into a percentile.

## Intervals

| Interval | Start | End | Clock |
| --- | --- | --- | --- |
| `feed_network` | exchange_event | ws_frame_receive | Unix |
| `socket_to_frame` | socket_receive | ws_frame_receive | monotonic |
| `frame_to_parse` | ws_frame_receive | parse_done | monotonic |
| `parse_to_book` | parse_done | book_apply_done | monotonic |
| `book_to_signal` | book_apply_done | signal_start | monotonic |
| `signal_compute` | signal_start | signal_end | monotonic |
| `signal_to_decision` | signal_end | decision | monotonic |
| `decision_to_encode` | decision | order_encode_done | monotonic |
| `encode_to_send` | order_encode_done | socket_send | monotonic |
| `tick_to_wire` | ws_frame_receive | socket_send | monotonic |
| `send_to_ack` | socket_send | exchange_ack | monotonic |
| `send_to_resting` | socket_send | resting | monotonic |
| `resting_to_fill` | resting | fill | monotonic |
| `cancel_round_trip` | cancel_request | cancel_ack | monotonic |

The public probe records `ws_frame_receive` through `socket_send`.
`socket_receive` requires connector-level socket instrumentation. Exchange ACK
and later stages require a separately authorized private/order integration.

## Raw And Summary Outputs

Raw traces are newline-delimited JSON. Every row includes schema version,
venue, channel, symbol, trace id, optional exchange sequence, and fixed stage
timestamp arrays. Raw rows are the source of truth.

Summary JSON contains, per interval:

- `count`
- `missing`
- `invalid`
- `mean_ns`
- `p50_ns`
- `p90_ns`
- `p99_ns`
- `p999_ns`
- `max_ns`

It also contains trace-level integrity counters:

- `traces_seen`
- `unique_trace_ids`
- `duplicate_trace_ids`
- `duplicate_stage_marks`
- `complete_required`
- `incomplete_required`
- `out_of_order`
- `dropped_traces`

The required public-probe chain is `ws_frame_receive`, `parse_done`,
`book_apply_done`, `signal_start`, `signal_end`, `decision`,
`order_encode_done`, and `socket_send`. A completed trace has each stage
exactly once and in monotonic order.

Quantiles use deterministic R-7 linear interpolation over sorted valid integer
nanosecond samples. Re-running `summarize` against the same raw file must
produce byte-identical pretty JSON.

## Hot-Path And Loss Boundary

Stage marking uses fixed-size timestamp arrays. Completed traces enter a
bounded handoff with non-blocking `try_send`. Queue saturation increments
`dropped_traces`; it must never be hidden by otherwise favorable percentiles.

The release acceptance benchmark measures fixed stage marking plus bounded
handoff. Its P99 budget is `5us` per trace. This budget validates measurement
overhead only; it is not a production tick-to-wire claim.

The public command accepts optional `--connect-ip` and `--tls12-only`
diagnostics. The former connects TCP to a selected address while retaining the
logical WebSocket URL for TLS SNI and the HTTP Host header. The latter pins the
TLS protocol for proxy compatibility. Both are run metadata, not permanent
exchange defaults.

## Initial Release Baseline

On 2026-07-28, the release binary captured 100 public BBO traces per venue:

| Venue | Complete | Dropped | tick_to_wire P50 | tick_to_wire P99 |
| --- | ---: | ---: | ---: | ---: |
| Binance Futures | 100/100 | 0 | 7.271 us | 66.089 us |
| Hyperliquid | 100/100 | 0 | 42.521 us | 109.273 us |

The accepted release run measured each recorder construction plus bounded
handoff operation individually (`batch_size=1`) over 1,000,000 traces at P50
`334 ns` and P99 `500 ns`, with zero drops.

These are probe/loopback baselines on the local development host, not live
order-path claims. Binance `feed_network` produced 45 invalid negative samples
out of 100 because the local and exchange wall clocks were not aligned. That
result is retained as evidence that clock-offset-sensitive metrics fail visibly
instead of contaminating percentiles.

The accepted raw traces, summaries, benchmark result, commands, host metadata,
DNS answers, source hashes, and artifact hashes are stored under
`docs/evidence/latency_probe_20260728/`. A public probe that reaches its
deadline without one valid BBO fails instead of accepting an empty summary.

## Goal 1 Acceptance Ladder

1. Contract and standalone probe: this task.
2. Connector receive/parse/book hooks for Binance and Hyperliquid.
3. Strategy signal/decision/encode/send hooks with allocation profiling.
4. Authorized ACK/resting/fill/cancel lifecycle measurement.
5. Stable 8-hour capture plus gap/recovery accounting.
6. AWS instance hunting and tuning using the same probe artifact and fixed
   benchmark protocol.

Target claims such as `tick_to_wire P99 <= 1 ms` are accepted only after the
corresponding real execution path emits raw traces. Synthetic and loopback
results validate the measurement system, not production trading performance.
