# 0625T007 Business Report

## Status

- 任务状态: `待验收`
- 业务线程: `业务线程-replay`
- 最终建议: `public_market_view_replay_alignment_ready_for_qa`
- Commit: `12c81ee`

## Scope

- 使用 `0627T001` QA-accepted aligned public context rows:
  - `local_live_analysis/cross_exchange_mvp_hl_fast_sample_expansion_0627T001/symmetric_edge_context_coverage.csv`
- 使用 `0625T004` QA-accepted shared kernel:
  - `examples/hyperliquid/cross_exchange_shared_signal_kernel.py`
  - `local_live_analysis/cross_exchange_mvp_shared_kernel_0625T004/shared_kernel_manifest.json`
- 使用 `0625T005` QA-accepted production shadow reference:
  - `local_live_analysis/cross_exchange_mvp_production_shadow_0625T005/production_shadow_manifest.json`
  - `local_live_analysis/cross_exchange_mvp_production_shadow_0625T005/shadow_decision_rows.csv`
- 使用 `0625T006` QA-accepted replay contract:
  - `local_live_analysis/cross_exchange_mvp_audit_replay_contract_0625T006/replay_input_contract.json`
- 未采集新数据，未执行 AWS/remote alignment，未读取 credential，未调用 private/account/order/cancel endpoint，未初始化 live client，未下单，未修改 watcher 或 production config。

Scope caveat:

- 本地仓库不包含 `0627T001` raw WebSocket 文件；本轮 replay source 是 QA-accepted `0627T001` aligned public context package。T007 不声称重复 raw-file reconstruction，也未执行新 public collection。

## Implementation

- 新增 runner:
  - `examples/hyperliquid/cross_exchange_public_replay_alignment.py`
- 新增 focused tests:
  - `examples/hyperliquid/test_cross_exchange_public_replay_alignment.py`
- 新增 output package:
  - `local_live_analysis/cross_exchange_mvp_public_replay_alignment_0625T007/`

Runner 行为:

- 从 `0627T001` public context rows 重建 replay market-view rows。
- 使用 T005 frozen normalization stats 和 kernel parameters 调用 T004 shared kernel。
- 将 replay decision rows 与 T005 production-shadow `shadow_decision_rows.csv` 按 `decision_id` 对齐。
- 比较 action、block reason、signal status/score/abs z、side、fair-mid、quote intent、edge、source-age/basis/warning bucket 和 boundary flags。
- 输出 market-view/source-age/cadence gate、future-label exclusion gate、action-path comparison 和 mismatch attribution。

## Results

- Final recommendation:
  - `public_market_view_replay_alignment_ready_for_qa`
- Raw source rows: `10747`
- Replay rows: `10704`
- Reference decision rows: `10704`
- Comparison rows: `10704`
- Matched decision rows: `10704`
- Mismatched decision rows: `0`
- Mismatch row count: `0`
- Action mismatch count: `0`
- Missing reference count: `0`
- Missing replay count: `0`
- Market-view fail-closed count: `0`
- Cadence/source-age gate fail count: `0`
- Future join count: `0`
- Final blocking reasons: none
- T006 schema hash consumed:
  - `0a899c61d63cf5326e16fa8b2d95ae7dc965b04ada72f3ba99811abfca0b9ab5`

Cadence/source-age summary:

- `xemm_0627_t001_hlfast_utc16_a`: `3587` rows, median decision delta `500ms`, max decision delta `3000ms`, gate `pass`.
- `xemm_0627_t001_hlfast_utc17_b`: `3567` rows, median decision delta `500ms`, max decision delta `1000ms`, gate `pass`.
- `xemm_0627_t001_hlfast_utc17_c`: `3550` rows, median decision delta `500ms`, max decision delta `1000ms`, gate `pass`.

Future-label exclusion:

- Decision input future field count: `0`.
- Future label rows present but excluded: `10704`.
- Future timestamp not-after-decision count: `0`.

## Output Package

- `replay_alignment_manifest.json`
- `replay_market_view_rows.csv`
- `replay_decision_rows.csv`
- `action_path_comparison.csv`
- `mismatch_attribution.csv`
- `cadence_source_age_report.csv`
- `future_join_report.csv`
- `boundary_manifest.json`
- `validation_report.md`

## Verification

- `python -m pytest examples/hyperliquid/test_cross_exchange_public_replay_alignment.py -q`
  - Result: `3 passed`
- `python -m py_compile examples/hyperliquid/cross_exchange_public_replay_alignment.py examples/hyperliquid/test_cross_exchange_public_replay_alignment.py`
  - Result: passed
- `python examples/hyperliquid/cross_exchange_public_replay_alignment.py --help`
  - Result: passed
- Formal runner:
  - `python examples/hyperliquid/cross_exchange_public_replay_alignment.py --output-dir local_live_analysis/cross_exchange_mvp_public_replay_alignment_0625T007`
  - Result: generated final package with `public_market_view_replay_alignment_ready_for_qa`
- JSON parse:
  - passed for `replay_alignment_manifest.json` and `boundary_manifest.json`.
- CSV/artifact validation:
  - required artifacts all present and non-empty
  - `replay_market_view_rows.csv`: `10704` rows
  - `replay_decision_rows.csv`: `10704` rows
  - `action_path_comparison.csv`: `10704` rows
  - `future_join_report.csv`: all checks pass
  - `mismatch_attribution.csv`: `none/none/0/pass`
  - boundary flags are true
- Deterministic reproduction:
  - reran output to `/tmp/0625T007_repro.Ps7NIK`
  - 2 JSON, 6 CSV and 1 markdown artifact matched after excluding manifest output paths.
- Combined focused regression:
  - `python -m pytest examples/hyperliquid/test_cross_exchange_signal_acceptance.py examples/hyperliquid/test_cross_exchange_shared_signal_kernel.py examples/hyperliquid/test_cross_exchange_production_shadow.py examples/hyperliquid/test_cross_exchange_mvp_audit_replay_contract.py examples/hyperliquid/test_cross_exchange_public_replay_alignment.py -q` -> `18 passed`
- Combined `py_compile`
  - Result: passed
- `git diff --check`
  - Result: passed

## Boundaries

- No new public collection.
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
  - `public_market_view_replay_alignment_ready_for_qa`
- Status:
  - `待验收`
- No watcher/live strategy behavior changed.
- No private/order endpoints were used.
- No orders were placed.
- No canary/promotion was authorized.

## Commit

- commit: `12c81ee`
- 提交信息: `0706 run cross exchange public replay alignment t007`
