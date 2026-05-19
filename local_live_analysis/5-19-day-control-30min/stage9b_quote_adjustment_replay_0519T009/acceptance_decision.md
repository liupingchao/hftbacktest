# 0519T008 Step 9B Quote-Adjustment Replay Decision

## Boundary

- Mode: `default_off_offline_diagnostic`
- Dataset: `5-19-day-control-30min`
- Default-off only: yes
- Offline only: yes
- Live / promotion authorized: no
- Production strategy behavior changed: no

## Classification

- result: `promising_but_single_sample`
- missing T006 audit fields in existing sample: `0`

## Reasons

- active candidates have nonzero coverage, but only one current-format sample is available

## Candidate Metrics

- `baseline_control`: decision rows `96341`, submit orders `4098`, fill rate `0.026842362127867253`, fill-after-cancel rate `0.012933138116154222`
- `fair_reservation_shift_edge_25`: decision rows `86098`, submit orders `3484`, fill rate `0.022101033295063147`, fill-after-cancel rate `0.010332950631458095`
- `inventory_reservation_shift_band`: decision rows `86695`, submit orders `3621`, fill rate `0.02540734603700635`, fill-after-cancel rate `0.012979839823253245`
- `spread_widening_stale_latency`: decision rows `14314`, submit orders `0`, fill rate ``, fill-after-cancel rate ``
- `size_reduction_or_add_side_suppression_pressure`: decision rows `75109`, submit orders `2402`, fill rate `0.022481265611990008`, fill-after-cancel rate `0.009159034138218152`
- `stale_latency_no_fresh_add`: decision rows `0`, submit orders `0`, fill rate ``, fill-after-cancel rate ``
- `min_move_quote_age_churn_guard`: decision rows `93897`, submit orders `3832`, fill rate `0.028444676409185805`, fill-after-cancel rate `0.013569937369519834`
- `post_only_safety_interaction`: decision rows `24700`, submit orders `1077`, fill rate `0.022284122562674095`, fill-after-cancel rate `0.011142061281337047`

## Next-Step Interpretation

This output validates runner, metric, artifact, and diagnostic-classification mechanics. A single sample cannot establish live readiness or generalized profitability. If the result is `needs_more_instrumentation`, collect or replay with T006 audit fields before promotion-style claims. If later candidates become `promising_but_single_sample`, open a separate multi-sample validation planning task.
