# 0714T004 Quote/Fill Probability Evidence Rerun

Final recommendation: `route_to_public_flow_artifact_repair`

- Attempt rows: `71`
- Resting attempt rows: `1`
- Public-trade summary rows: `1`
- Coverage evidence rows: `1`
- Censoring rows: `71`
- Depth proxy rows: `71`
- Order status counts: `{'skipped': 70, 'resting': 1}`
- No-fill state counts: `{'no_order_submitted': 70, 'resting_no_fill_observed': 1}`
- Trade-through counts: `{'not_applicable_no_order_submitted': 70, 'coverage_not_proven_complete': 1}`
- Censoring counts: `{'not_applicable_no_order_submitted': 70, 'short_hold_censored': 1}`

The v2 live package contains one submitted/resting/no-fill lifecycle, but `public_stream_coverage.csv` reports incomplete interval coverage and zero rows are marked as `artifact_gap_not_no_exchange_trades`. This does not support fill probability, queue priority, fee/rebate, quote policy design, or realized PnL claims.
