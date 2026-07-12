# 0710T001 Quote/Fill Probability Evidence

Final recommendation: `route_to_public_flow_artifact_repair`

- Attempt count: `5`
- Order status counts: `{'resting': 3, 'error': 2}`
- No-fill state counts: `{'resting_no_fill_observed': 3, 'not_resting_rejected': 2}`
- Depth proxy counts: `{'depth_proxy_missing': 1, 'depth_proxy_present': 4}`
- Trade-through status counts: `{'public_flow_artifact_missing': 1, 'rolling_proxy_strict_trade_through_present': 2, 'rolling_proxy_present_resting_interval_missing': 2}`
- Censoring counts: `{'horizon_missing': 1, 'not_applicable_rejected': 2, 'short_hold_censored': 2}`

The available evidence supports post-only reject and censored no-fill classification only. It does not support a fitted fill-probability model, exact queue priority, fee/rebate, realized PnL, maker viability, T012, promotion, or final MVP claims.
