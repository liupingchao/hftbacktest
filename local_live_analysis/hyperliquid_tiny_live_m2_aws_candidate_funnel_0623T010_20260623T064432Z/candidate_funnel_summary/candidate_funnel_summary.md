# 0623T010 Candidate Funnel Summary

Watcher seconds elapsed: `600.000648`
Public stream counts: `{'l2Book': 112, 'pong': 19, 'subscriptionResponse': 2, 'trades': 1147}`
Current candidates: `1259`
Fresh-touch evidence pass: `2`
Fresh-touch gate allowed: `0`
Fair-mid source evaluations: `0`
Edge gate evaluations: `0`
Shadow would-submit: `0`

## First Blocking Stage

No candidate passed the accepted fresh-touch/dynamic-size gate. Because that gate never allowed a candidate, anti-drift, Binance freshness, fair-mid source, and edge gate were not reached.

## Dominant Skip Reasons

- `missing_same_side_strict_through_support;missing_touch_freshness_or_queue_reset_evidence`: `753`
- `missing_touch_freshness_or_queue_reset_evidence`: `299`
- `missing_same_side_strict_through_support;missing_touch_freshness_or_queue_reset_evidence;missing_recent_same_side_at_or_through_throughput`: `205`
- `missing_same_side_strict_through_support`: `2`

## Reason Atoms

- `missing_touch_freshness_or_queue_reset_evidence`: `1257`
- `missing_same_side_strict_through_support`: `960`
- `missing_recent_same_side_at_or_through_throughput`: `205`

## Boundary

No credentials, private/account/order/cancel endpoint, live client, real order, or final gate path was used.
