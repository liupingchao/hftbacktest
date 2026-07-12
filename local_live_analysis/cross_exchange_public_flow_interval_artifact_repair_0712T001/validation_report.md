# 0712T001 Public-Flow Interval Artifact Repair

Final route: `route_to_controlled_same_envelope_live_evidence_with_resting_interval_public_flow_artifacts`

- Accepted resting/no-fill attempts: `3`
- Reconstruction statuses: `{'not_reconstructable_from_current_artifact': 1, 'partial_proxy_only': 2}`
- Public-trades statuses: `{'not_reconstructable_from_current_artifact': 3}`
- Depletion statuses: `{'not_reconstructable_from_current_artifact': 3}`
- Gap fields: `{'resting_start_ts': 3, 'cancel_or_shutdown_ts': 3, 'same_side_visible_depth_at_resting_start': 3, 'public_trades_during_resting_interval': 3, 'depletion_trade_through_estimate': 3}`

Current artifacts can bind live resting attempts to response-time and hold-duration proxies plus pre-submit depth proxy, but they do not reconstruct individual public trades or actual same-side depletion during the resting interval.

No fill probability, queue priority, fee/rebate, realized PnL, maker viability, T012, promotion, or final MVP claim is made.
