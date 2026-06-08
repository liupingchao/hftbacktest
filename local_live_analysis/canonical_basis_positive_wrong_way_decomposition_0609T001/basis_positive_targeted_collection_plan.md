# Basis-Positive Targeted Collection Plan

Task: `0609T001`

## Boundary

- This is collection design only. No new data was collected in this task.
- Any collection must be a later separately dispatched task with its own QA acceptance.

## Target Regimes

- active/high-vol: validate whether basis-positive signal survives larger visible Binance movement.
- normal: reduce sample concentration and test baseline persistence.
- quiet: retest the current quiet-sample tail concentration.
- wide-spread: add rows outside the narrow current spread support.
- basis-large: validate positive-basis magnitude monotonicity and tail loss.
- HL-book-conflict: collect states where Hyperliquid imbalance or microprice conflicts with positive basis.

## Candidate Visible Tail Hypotheses

- `basis_magnitude_bucket=basis_positive_small`: 42 wrong-way rows, classification `promising_visible_filter`.
- `hl_top5_imbalance_bucket=hl_top5_imbalance_negative_small`: 31 wrong-way rows, classification `promising_visible_filter`.
- `hl_microprice_minus_mid_bucket=hl_microprice_minus_mid_negative_small`: 30 wrong-way rows, classification `promising_visible_filter`.
- `sample_id=cross_exchange_public_sample_xemm_0603_quiet_a_event`: 60 wrong-way rows, classification `needs_more_samples`.
- `coarse_time_window=cross_exchange_public_sample_xemm_0603_quiet_a_event:window_middle`: 45 wrong-way rows, classification `needs_more_samples`.

## Evidence Gates For A Later Task

- Add enough independent samples so max sample row share is below `0.40` for basis-positive rows and wrong-way rows.
- Require at least `5` samples and at least `2` active/high-vol, `2` normal, and `1` quiet windows before changing the recommendation.
- For a proposed filter, require wrong-way concentration to repeat in at least `3` samples and controlled basis effect to remain positive in Binance momentum and HL book-state buckets.
- Reject the filter hypothesis if wrong-way rows spread evenly across visible states or if controlled basis effect collapses to a Binance momentum or HL book-state proxy.

## Current Recommendation

- Final recommendation: `targeted_collection_ready`.
- This recommendation does not authorize collection inside T001.
