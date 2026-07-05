# 0625T004 Business Report

## Status

- 任务状态: `待验收`
- 业务线程: `业务线程-kernel`
- 最终建议: `shared_signal_quote_intent_kernel_ready_for_qa`
- Commit: 无
- 未提交原因: 当前工作区已有 `0702T001/0702T002` 相关未提交改动和本轮 QA/事实源更新；为避免把 unrelated 改动混入提交，本轮未执行 git commit。

## Scope

- 使用 `0625T003` QA-accepted signal contract:
  - `local_live_analysis/cross_exchange_mvp_signal_acceptance_0625T003/accepted_signal_contract.json`
- 创建 offline/public-only/read-only shared pure decision kernel。
- 生成固定 fixture inputs / outputs / manifest / boundary manifest。
- 未改 watcher 生产路径，未运行 shadow，未执行网络采集，未读 credential，未调用 private/account/order/cancel endpoint，未下单，未授权 canary 或 promotion。

## Implementation

- 新增 shared kernel:
  - `examples/hyperliquid/cross_exchange_shared_signal_kernel.py`
- 新增 focused tests:
  - `examples/hyperliquid/test_cross_exchange_shared_signal_kernel.py`
- 新增 output package:
  - `local_live_analysis/cross_exchange_mvp_shared_kernel_0625T004/`

Kernel 行为:

- 读取并验证 `cross_exchange_signal_contract_v1`：
  - candidate: `binance_lead_composite`
  - feature schema: `input_binance_top5_imbalance`, `input_binance_microprice_minus_mid_ticks`, `input_binance_mid_move_ticks_from_prev`
  - horizon: `1000ms`
  - threshold: `abs(z) >= 1.0`
  - side mapping: `positive_signal_buy_negative_signal_sell`
- 内核输入要求显式传入 normalization stats；fixture 使用 deterministic identity stats (`mean=0`, `std=1`)。这避免在 shadow/replay 中隐式回看或重新调参。
- 输出 deterministic decision record:
  - signal components / score / threshold status
  - side
  - signed expected move ticks
  - fair mid
  - single-layer touch quote intent
  - post-only `Alo`
  - edge ticks
  - action `would_submit` or `block`
  - block reason
  - boundary flags showing no endpoint/client/order behavior
- Fixed fixture cases:
  - `fixture_buy_pass`
  - `fixture_sell_pass`
  - `fixture_signal_below_threshold`
  - `fixture_missing_feature_block`
  - `fixture_warning_bucket_visible`

## Artifacts

- `shared_kernel_manifest.json`
  - `final_recommendation=shared_signal_quote_intent_kernel_ready_for_qa`
  - `fixture_input_count=5`
  - `fixture_output_count=5`
  - `would_submit_fixture_count=3`
  - `block_fixture_count=2`
  - `warning_bucket_visible_in_fixture=true`
- `fixture_inputs.json`
- `fixture_outputs.json`
- `boundary_manifest.json`

T003 warning bucket remains visible:

- `some_source_age_or_basis_buckets_have_negative_adjusted_proxy`
- fixture case `fixture_warning_bucket_visible`

## Verification

- `python -m pytest examples/hyperliquid/test_cross_exchange_shared_signal_kernel.py -q`
  - Result: `3 passed`
- `python -m py_compile examples/hyperliquid/cross_exchange_shared_signal_kernel.py examples/hyperliquid/test_cross_exchange_shared_signal_kernel.py`
  - Result: passed
- `python examples/hyperliquid/cross_exchange_shared_signal_kernel.py --help`
  - Result: passed
- `python examples/hyperliquid/cross_exchange_shared_signal_kernel.py --generate-fixtures --output-dir local_live_analysis/cross_exchange_mvp_shared_kernel_0625T004`
  - Result: generated `shared_signal_quote_intent_kernel_ready_for_qa`
- JSON parse:
  - `python -m json.tool` passed for `shared_kernel_manifest.json`, `fixture_inputs.json`, `fixture_outputs.json`, and `boundary_manifest.json`.
- Deterministic reproduction:
  - reran output to `/tmp/0625T004_qa_repro.m4LD6q`
  - 4 JSON artifacts matched after excluding output file paths.
- Combined focused regression:
  - `python -m pytest examples/hyperliquid/test_cross_exchange_signal_acceptance.py examples/hyperliquid/test_cross_exchange_shared_signal_kernel.py -q` -> `6 passed`
- `git diff --check`
  - Result: passed.

## Boundaries

- No network collection.
- No AWS execution.
- No remote alignment.
- No credential read.
- No private/account/order/cancel endpoint.
- No user stream.
- No live client initialization.
- No live order.
- No shadow execution.
- No watcher strategy change.
- No production config change.
- No new signal feature search, threshold tuning, side mapping change, or horizon change.
- No canary or promotion authorization.

## Remaining Risk

- The kernel now exists as the shared callable contract for T005/T007, but the existing watcher public-shadow path was not changed in this task. T005 should consume this module when implementing production-equivalent public shadow.
- `expected_move_ticks_per_signal_z=4.0` is an explicit fixture/kernel parameter for deterministic quote-intent evidence, not a production-tuned profitability claim. T005 must validate counterfactual edge over fresh public windows.

## Done

- Exact T003 contract consumed:
  - `local_live_analysis/cross_exchange_mvp_signal_acceptance_0625T003/accepted_signal_contract.json`
- Final recommendation enum:
  - `shared_signal_quote_intent_kernel_ready_for_qa`
- Status:
  - `待验收`
- No watcher/live strategy behavior changed.
- No private/order endpoints were used.
- No orders were placed.
- No shadow/canary/promotion was authorized.

## Commit

- commit: 无
- 提交信息: 无
