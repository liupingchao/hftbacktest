# 0625T005 Business Report

## Status

- 任务状态: `待验收`
- 业务线程: `业务线程-shadow`
- 最终建议: `production_shadow_accepted_for_replay_contract`
- Commit: `80b0e72`

## Scope

- 使用 `0627T001` QA-accepted public sample rows:
  - `local_live_analysis/cross_exchange_mvp_hl_fast_sample_expansion_0627T001/symmetric_edge_context_coverage.csv`
- 使用 `0625T003` QA-accepted signal contract:
  - `local_live_analysis/cross_exchange_mvp_signal_acceptance_0625T003/accepted_signal_contract.json`
- 使用 `0625T004` QA-accepted shared kernel:
  - `examples/hyperliquid/cross_exchange_shared_signal_kernel.py`
  - `local_live_analysis/cross_exchange_mvp_shared_kernel_0625T004/shared_kernel_manifest.json`
- 执行 offline/public-only/no-submit production shadow acceptance。
- 未采集新数据，未执行 AWS/remote alignment，未读取 credential，未调用 private/account/order/cancel endpoint，未初始化 live client，未下单，未修改 watcher 或 production config。

## Implementation

- 新增 runner:
  - `examples/hyperliquid/cross_exchange_production_shadow.py`
- 新增 focused tests:
  - `examples/hyperliquid/test_cross_exchange_production_shadow.py`
- 新增 output package:
  - `local_live_analysis/cross_exchange_mvp_production_shadow_0625T005/`

Runner 行为:

- 只保留 `valid_for_1000ms_signal_acceptance=true` 且 `1000ms <= effective_future_age_ms <= 1250ms` 的 rows。
- 调用 T004 shared kernel 生成 signal、side、fair-mid、touch quote intent、edge、action 和 block reason。
- Normalization stats 明确冻结自 `0627T001` valid public rows 的 feature distribution，且不使用 future labels:
  - `normalization_stats_source=all_valid_0627T001_public_rows_feature_distribution_no_future_labels`
- Future labels 只用于 no-submit counterfactual markout，不作为决策输入。
- T003 warning bucket 保留在 diagnostics 中。

## Results

- Raw input rows: `10747`
- Valid shadow rows: `10704`
- Would-submit rows: `1098`
- Would-submit by window:
  - `xemm_0627_t001_hlfast_utc16_a`: `429`
  - `xemm_0627_t001_hlfast_utc17_b`: `291`
  - `xemm_0627_t001_hlfast_utc17_c`: `378`
- Per-window mean adjusted counterfactual edge ticks:
  - `13.54662005`
  - `2.02233677`
  - `4.33333333`
- Aggregate mean adjusted counterfactual edge ticks: `7.32058288`
- Max window contribution: `0.39071038`
- Warning bucket visibility:
  - warning-bucket decisions: `1231`
  - warning-bucket would-submit rows: `91`
- Final recommendation:
  - `production_shadow_accepted_for_replay_contract`

Caveat:

- Median adjusted counterfactual edge is `-1.5` ticks because many 1000ms rows have zero mid move and the no-submit proxy subtracts the `1.5` tick buffer. The acceptance recommendation relies on per-window and aggregate mean adjusted edge, sufficient would-submit counts, and non-dominant window contribution; QA should keep this caveat visible before moving beyond replay-contract work.

## Output Package

- `production_shadow_manifest.json`
- `shadow_decision_rows.csv`
- `funnel_summary.csv`
- `would_submit_rows.csv`
- `counterfactual_markout_rows.csv`
- `edge_proxy_summary.csv`
- `per_window_edge_summary.csv`
- `regime_stability.csv`
- `source_age_stability.csv`
- `basis_stability.csv`
- `boundary_manifest.json`
- `recommendation.md`

## Verification

- `python -m pytest examples/hyperliquid/test_cross_exchange_production_shadow.py -q`
  - Result: `3 passed`
- `python -m py_compile examples/hyperliquid/cross_exchange_production_shadow.py examples/hyperliquid/test_cross_exchange_production_shadow.py`
  - Result: passed
- `python examples/hyperliquid/cross_exchange_production_shadow.py --help`
  - Result: passed
- Formal runner:
  - `python examples/hyperliquid/cross_exchange_production_shadow.py --output-dir local_live_analysis/cross_exchange_mvp_production_shadow_0625T005`
  - Result: generated final package with `production_shadow_accepted_for_replay_contract`
- JSON parse:
  - `python -m json.tool` passed for `production_shadow_manifest.json` and `boundary_manifest.json`.
- CSV/artifact validation:
  - required artifacts all present and non-empty
  - `shadow_decision_rows.csv`: `10704` rows
  - `would_submit_rows.csv`: `1098` rows
  - per-window would-submit counts exceed `20/window`
  - per-window mean adjusted counterfactual edge is positive
  - boundary flags are true
- Deterministic reproduction:
  - reran output to `/tmp/0625T005_repro.oIMiHJ`
  - 2 JSON, 9 CSV and 1 markdown artifact matched after excluding output file paths.
- Combined focused regression:
  - `python -m pytest examples/hyperliquid/test_cross_exchange_signal_acceptance.py examples/hyperliquid/test_cross_exchange_shared_signal_kernel.py examples/hyperliquid/test_cross_exchange_production_shadow.py -q` -> `9 passed`
- `python -m py_compile examples/hyperliquid/cross_exchange_signal_acceptance.py examples/hyperliquid/cross_exchange_shared_signal_kernel.py examples/hyperliquid/cross_exchange_production_shadow.py examples/hyperliquid/test_cross_exchange_production_shadow.py`
  - Result: passed

## Boundaries

- No network collection.
- No AWS execution.
- No remote alignment.
- No credential read.
- No private/account/order/cancel endpoint.
- No user stream.
- No live client initialization.
- No live order.
- No watcher strategy change.
- No production config change.
- No signal feature search, threshold tuning, side mapping change, or horizon change.
- No canary or promotion authorization.

## Done

- Final recommendation enum:
  - `production_shadow_accepted_for_replay_contract`
- Status:
  - `待验收`
- No watcher/live strategy behavior changed.
- No private/order endpoints were used.
- No orders were placed.
- No canary/promotion was authorized.

## Commit

- commit: `80b0e72`
- 提交信息: `0705 run cross exchange production shadow t005`
