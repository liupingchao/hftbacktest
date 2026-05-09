# Iter1 Priority 2 Replay Lag and Working-Order Plan

Date: 2026-05-04

## Goal

Before maker strategy optimization, make audit replay timing and order-state divergence explicit enough that bad replay rows cannot be treated as strategy signal.

## Scope

Priority 2 covers:

- strict audit replay lag gate;
- audit replay decision rows preserving actual backtest feed timestamps;
- compare report lag buckets and gate-filtered metrics;
- working-order lifecycle mismatch attribution, including semantic vs identity-only split;
- explicit decision-row working-order semantics for bid/ask qty, status, request state, and pending-cancel state;
- validation that lifecycle rows do not pollute decision-row audit checks.

Not in scope:

- maker parameter search;
- queue/fill model tuning;
- PnL attribution;
- live canary rerun.

## Implementation Plan

1. Add replay timing fields to audit schema:
   - `replay_scheduled_ts_local`
   - `bt_feed_ts_local`
   - `bt_feed_ts_exch`
   - `replay_lag_ns`
   - `replay_lag_abs_ns`
2. In audit replay mode, keep `ts_local` as the scheduled live decision timestamp and write actual backtest feed timestamps separately.
3. Add replay lag gate config:
   - `max_lag_ms`
   - `strict_lag_gate`
   - `lag_gate_action = report|drop|fail`
4. Make generated alignment configs use `max_lag_ms = 250.0`, `strict_lag_gate = true`, and `lag_gate_action = "fail"`.
5. Extend compare reports with:
   - `replay_lag`
   - `api_throttle.replay_lag_abs_diff_le_gate`
   - `api_throttle.replay_lag_abs_diff_gt_gate`
   - `working_order_lifecycle`
6. Split working-order lifecycle mismatch into:
   - semantic state mismatch: side, price tick, qty, active count, state, pending cancel;
   - identity-only mismatch: raw order id / local order detail differences after semantic state matches;
   - REST/local divergence evidence from safety status or explicit open-order diff.
7. Add first-divergence report with surrounding decision/lifecycle context.
8. Add explicit bid/ask working-order fields:
   - `working_bid_qty`, `working_ask_qty`
   - `working_bid_status`, `working_ask_status`
   - `working_bid_req`, `working_ask_req`
   - `working_bid_pending_cancel`, `working_ask_pending_cancel`
9. Make compare prefer explicit working-order semantic fields and fall back to parsing `local_open_orders` for historical CSVs.
10. Write backtest decision-row local open-order diagnostics so working-order comparison has symmetric local-state evidence.
11. Make audit validation apply formula checks only to decision rows.

## Acceptance

Unit and static acceptance:

```bash
python -m pytest \
  examples/binance_tick_mm/test_backtest_tick_mm.py \
  examples/binance_tick_mm/test_compare_audit.py \
  examples/binance_tick_mm/test_validate_audit.py \
  examples/binance_tick_mm/test_latency_from_audit.py \
  examples/binance_tick_mm/test_pipeline_live_raw.py \
  examples/binance_tick_mm/test_align_live_run.py
```

Artifact acceptance:

- Backtest audit CSV header contains all replay timing fields.
- `validate_audit.py --strict` passes on generated audit replay CSV.
- `alignment_report_audit_replay_report.json` contains replay lag gate and working-order lifecycle breakdown.
- `alignment_report_audit_replay_report_v2.json` contains `semantic_mismatch_rows`, `identity_only_mismatch_rows`, and `first_semantic_divergence_context`.
- Fresh live/backtest audit CSV headers contain explicit working-order semantic fields, and compare uses them instead of raw order ids for semantic parity.
- If `strict_lag_gate = true` and `lag_gate_action = "fail"`, any lag breach fails the replay run.
- For historical diagnostic samples, use `lag_gate_action = "report"` and treat `replay_lag.in_gate_rows` as the usable optimization subset.

Decision gate for maker optimization:

- `replay_lag.in_gate_rate` must be high enough for the target live window.
- `replay_lag.abs_ns.p99 <= 250ms` is required for a full-window maker optimization baseline.
- If the full window fails but the gate subset is clean, the sample remains useful for diagnosis only.
