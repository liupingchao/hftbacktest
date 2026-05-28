# 0528T001 Business Report

执行线程：
- 业务线程-python

任务状态：
- 待验收

## Scope

Implemented the read-only/default-off inventory-aware quote placement runner defined by the accepted `0526T006` `inventory_aware_quote_placement_request` contract.

Boundary preserved:
- no live run
- no strategy behavior change
- no default-on behavior
- no parameter search
- no guard relaxation
- no promotion claim

## Implementation

Runner:
- `examples/binance_tick_mm/inventory_aware_quote_placement.py`

Focused tests:
- `examples/binance_tick_mm/test_inventory_aware_quote_placement.py`

Output directory:
- `local_live_analysis/stage9i_inventory_aware_quote_placement_0528T001/`

Generated artifacts:
- `run_manifest.json`
- `candidate_decision_rows.csv`
- `bucket_metrics_by_inventory_side_edge_distance.csv`
- `clean_only_stability_summary.csv`
- `caveated_sample_sensitivity.csv`
- `participation_and_fill_loss.csv`
- `inventory_recovery_quality.csv`
- `quote_mechanics_safety.csv`
- `candidate_recommendation.md`

## Sample Set

Accepted current-format samples used:
- `5-19-day-control-30min`
- `5-19-night-active-30min-a` caveated
- `5-19-night-active-30min-b`
- `5-19-night-active-30min-c`
- `5-21-day-control-60min`
- `5-26-active-minmove-control-30min-a`
- `5-26-active-minmove-control-60min-a` caveated
- `5-26-active-makeredge-control-180min-a`
- `5-26-active-minmove-control-30min-b`

All nine samples were usable. Audit seq join coverage and Stage 5C seq join coverage are `1.0` for each sample.

## Result

Clean-only verdict:
- `reject`

Caveated sensitivity verdict:
- `reject`

Key clean-only metrics:
- rows: `35,266`
- fills: `994`
- candidate request rows: `14,363`
- candidate request fills: `528`
- request fill rate: `0.03676`
- no-change fill rate: `0.02229`
- request 5s markout mean: `-85.21` ticks
- no-change 5s markout mean: `-70.20` ticks
- request spread capture mean: `6.85` ticks
- no-change spread capture mean: `16.78` ticks

Interpretation:
- The fixed policy skeleton has enough clean request fill mass to evaluate.
- The requested buckets are not merely under-sampled; they are worse than no-change on both 5s markout and spread capture.
- Participation/fill-loss, queue effects, opportunity cost, and PnL decomposition remain observed-only proxies from submitted orders, not counterfactual simulation.

## Recommendation

Do not proceed from this fixed `inventory_aware_quote_placement_request` skeleton to parameter design or strategy implementation as-is.

A later task may design a revised inventory-aware policy, but `0528T001` does not authorize that work. It also does not authorize live, default-on, guard relaxation, parameter search, or promotion.

## Verification

Passed:
- `python examples/binance_tick_mm/inventory_aware_quote_placement.py --help`
- `python -m pytest examples/binance_tick_mm/test_inventory_aware_quote_placement.py -q`
- `python examples/binance_tick_mm/inventory_aware_quote_placement.py --output-dir local_live_analysis/stage9i_inventory_aware_quote_placement_0528T001`
- `python3 .workflow/build_dashboard.py`
- `git diff --check`
