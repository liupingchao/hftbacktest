# 0530T002 Command Summary

This summary records the task-scoped Stage 9M chain and the resume-time checks.
It is a read-only evidence collection and rerun summary; no strategy behavior,
guard, parameter, default-on, tiny-live, promotion, replay semantic, connector,
core API, or Hyperliquid change was made.

## Collection And Pull

- Existing-sample scan output: `local_live_analysis/stage9m_targeted_clean_fill_0530T002/existing_sample_scan.json`
- New control sample: `5-31-stage9m-cleanfill-control-120min-a`
- Start marker UTC: `2026-05-30T16:16:06Z`
- Stop marker UTC: `2026-05-30T18:16:06Z`
- Stop exit code: `0`
- Deployed commit: `4760d481da3a06021ce25f9de4f2f0914662c5e0`
- Archive sha256: `f25ff59f0dc67bfc5a1ac99d43612ac1acdfdcb451ff7d26a20feb0eba3234f7`

## Derived Chain

The accepted chain artifacts are present at:

- maker acceptance: `local_live_analysis/5-31-stage9m-cleanfill-control-120min-a/maker_acceptance_with_market_view.json`
- T009 sidecar/join: `local_live_analysis/5-31-stage9m-cleanfill-control-120min-a/t009_fixed_sidecar/`
- Stage 5 labels: `local_live_analysis/5-31-stage9m-cleanfill-control-120min-a/stage5_execution_outcome_labels_0514T005/`
- Stage 5C safety: `local_live_analysis/5-31-stage9m-cleanfill-control-120min-a/stage5c_quote_anchor_safety_0530T002/`
- Stage 6 calibration: `local_live_analysis/5-31-stage9m-cleanfill-control-120min-a/stage6_execution_calibration_0530T002/`
- Stage 9K rerun: `local_live_analysis/stage9m_targeted_clean_fill_0530T002/stage9k_with_120min_sample/`
- Stage 9L rerun: `local_live_analysis/stage9m_targeted_clean_fill_0530T002/stage9l_with_120min_sample/`

## Resume-Time Checks

- `python examples/binance_tick_mm/maker_acceptance.py --help` -> passed
- `python examples/binance_tick_mm/binance_top5_provenance.py --help` -> passed
- `python examples/binance_tick_mm/execution_outcome_labels.py --help` -> passed
- `python examples/binance_tick_mm/quote_anchor_safety.py --help` -> passed
- `python examples/binance_tick_mm/execution_outcome_calibration.py --help` -> passed
- `python examples/binance_tick_mm/fill_quality_bucket_synthesis.py --help` -> passed
- `python examples/binance_tick_mm/fill_quality_rejection_decomposition.py --help` -> passed
- JSON sanity over 7 key artifacts -> passed
- CSV header sanity over 3 key Stage 9M artifacts -> passed

Focused pytest was not run during this resume because no Python source or test
files were changed. The existing Stage 9K/9L rerun artifacts were generated
before the interruption and were checked by artifact parse/schema sanity here.
