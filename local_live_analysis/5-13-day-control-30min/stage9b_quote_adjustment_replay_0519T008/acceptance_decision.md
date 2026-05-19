# 0519T008 Step 9B Quote-Adjustment Replay Decision

## Boundary

- Mode: `default_off_offline_diagnostic`
- Dataset: `5-13-day-control-30min`
- Default-off only: yes
- Offline only: yes
- Live / promotion authorized: no
- Production strategy behavior changed: no

## Classification

- result: `needs_more_instrumentation`
- missing T006 audit fields in existing sample: `15`

## Reasons

- 15 T006 quote-update audit fields are missing in the existing sample; runner used proxy fields for validation

## Candidate Metrics

- `baseline_control`: decision rows `47499`, submit orders `2516`, fill rate `0.02106518282988871`, fill-after-cancel rate `0.006359300476947536`
- `fair_reservation_shift_edge_25`: decision rows `41920`, submit orders `2040`, fill rate `0.02107843137254902`, fill-after-cancel rate `0.0058823529411764705`
- `inventory_reservation_shift_band`: decision rows `37029`, submit orders `1903`, fill rate `0.019968470835522858`, fill-after-cancel rate `0.007882291119285338`
- `spread_widening_stale_latency`: decision rows `16528`, submit orders `0`, fill rate ``, fill-after-cancel rate ``
- `size_reduction_or_add_side_suppression_pressure`: decision rows `34478`, submit orders `851`, fill rate `0.023501762632197415`, fill-after-cancel rate `0.011750881316098707`
- `stale_latency_no_fresh_add`: decision rows `0`, submit orders `0`, fill rate ``, fill-after-cancel rate ``
- `min_move_quote_age_churn_guard`: decision rows `46961`, submit orders `2468`, fill rate `0.020664505672609402`, fill-after-cancel rate `0.006482982171799027`
- `post_only_safety_interaction`: decision rows `11913`, submit orders `827`, fill rate `0.019347037484885126`, fill-after-cancel rate `0.009673518742442563`

## Next-Step Interpretation

This output validates runner, metric, artifact, and diagnostic-classification mechanics. A single sample cannot establish live readiness or generalized profitability. If the result is `needs_more_instrumentation`, collect or replay with T006 audit fields before promotion-style claims. If later candidates become `promising_but_single_sample`, open a separate multi-sample validation planning task.
