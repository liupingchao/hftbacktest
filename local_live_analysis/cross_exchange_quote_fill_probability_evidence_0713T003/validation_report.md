# 0713T003 Quote/Fill Probability Evidence Rerun

Final recommendation: `route_to_public_flow_artifact_repair`

- Attempt rows: `18`
- Resting attempt rows: `1`
- Public-trade summary rows: `1`
- Censoring rows: `18`
- Depth proxy rows: `18`
- Order status counts: `{'skipped': 17, 'resting': 1}`
- No-fill state counts: `{'no_order_submitted': 17, 'resting_no_fill_observed': 1}`
- Trade-through counts: `{'not_applicable_no_order_submitted': 17, 'rolling_proxy_present_resting_interval_missing': 1}`
- Censoring counts: `{'not_applicable_no_order_submitted': 17, 'short_hold_censored': 1}`

No fill is explained by the combination of: no captured attempt-keyed interval public trades, exact interval public-flow/depth reconstruction still being proxy-only, and a short-horizon censored resting window. This does not support fill probability, queue priority, fee/rebate, or realized PnL claims.
