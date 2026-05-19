# 0514T005 execution outcome labels

## Dataset
- run_dir: `/home/molly/project/hftbacktest/local_live_analysis/5-19-day-control-30min`
- output_dir: `/home/molly/project/hftbacktest/local_live_analysis/5-19-day-control-30min/stage5_execution_outcome_labels_0514T005`
- stage3 classification: `unknown`
- submit_orders: `4098`
- filled_orders: `110`
- fill_after_cancel_orders: `53`

## Coverage
- `fill_probability`: `available` - Observed directly from submit to first fill over 100/500/1000/5000ms horizons
- `time_to_fill`: `available` - Observed first-fill elapsed time is available for every filled order
- `adverse_selection_after_fill`: `available` - Observed side-adjusted future-mid markout after fill
- `spread_capture`: `available` - Observed realized spread proxy from fill price vs fill-time mid, plus future-mid retained spread
- `queue_priority_proxy`: `observed_only_proxy` - Only top1/top5 size, join age, stale age, and latency proxies are available; exact queue position is not observable
- `cancel_to_fill_race`: `available` - Observed cancel-request and fill timestamps allow direct fill-after-cancel labels
- `post_only_reject_throttle_churn`: `observed_only_proxy` - Actual submit orders expose churn directly; reject/throttle are only available as nearby decision-level context counts
- `inventory_impact`: `available` - Position before/after fill and inventory cycle fields are observed from audit rows
- `quote_placement_distance`: `available` - Distance-to-BBO ticks, placement bucket, post-only risk, and target-offset bucket are derived directly from submit context
- `missed_fill_opportunity_cost`: `observed_only_proxy` - Only future-mid observed opportunity-cost proxies are available; no counterfactual queue/fill proof
- `realized_pnl_decomposition`: `observed_only_proxy` - Only observed spread/markout decomposition proxies are available; fees are parameterized, inventory MTM is horizon-based
- `tail_risk`: `available` - 110 filled orders available for tail quantiles
- `partial_fill_lifecycle`: `available` - Lifecycle state, fill ratio, fill count, remaining qty, and terminal state are observed directly
- `inventory_cycle`: `available` - Observed from subsequent decision positions after fill, with explicit censoring when flat is not seen
- `sample_validity_censoring`: `available` - Each horizon includes observable/right-censored/tail-truncated flags and missing-lifecycle flags

## Lifecycle
- `canceled`: count=3959, rate=0.9660810151293314
- `expired`: count=27, rate=0.006588579795021962
- `filled`: count=110, rate=0.026842362127867253
- `open_or_missing`: count=2, rate=0.0004880429477794046
- `full_fill`: count=110, rate=0.026842362127867253
- `partial_fill`: count=0, rate=0.0
- `no_fill`: count=3988, rate=0.9731576378721327
- `fill_after_cancel_request`: count=53, rate=0.012933138116154222
- `cancel_ack_before_fill`: count=0, rate=0.0
- `cancel_ack_after_fill`: count=0, rate=0.0
- `fast_cancel_churn`: count=3531, rate=0.8616398243045388

## Censoring
- `fill_probability` horizon `100`: observable=4097, right_censored=1, tail_truncated=1
- `fill_markout` horizon `100`: observable=102, right_censored=8, tail_truncated=0
- `fill_probability` horizon `500`: observable=4097, right_censored=1, tail_truncated=1
- `fill_markout` horizon `500`: observable=93, right_censored=17, tail_truncated=0
- `fill_probability` horizon `1000`: observable=4096, right_censored=2, tail_truncated=2
- `fill_markout` horizon `1000`: observable=88, right_censored=22, tail_truncated=0
- `fill_probability` horizon `5000`: observable=4096, right_censored=2, tail_truncated=2
- `fill_markout` horizon `5000`: observable=88, right_censored=22, tail_truncated=1

## Method Notes
- queue / priority, missed opportunity, and realized PnL decomposition remain observed-only proxy labels.
- hazard summary is discrete horizon-based survival reporting; Cox-style modeling remains later work.
- observed correlations and proxies are not counterfactual queue/fill proof and are not strategy PnL proof.
