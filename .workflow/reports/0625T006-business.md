# 0625T006 Business Report

## Status

- 任务状态: `待验收`
- 业务线程: `业务线程-audit`
- 最终建议: `audit_replay_contract_ready_for_qa`
- Commit: `5c7e94b`

## Scope

- 使用 `0625T003` QA-accepted signal contract:
  - `local_live_analysis/cross_exchange_mvp_signal_acceptance_0625T003/accepted_signal_contract.json`
- 使用 `0625T004` QA-accepted shared kernel manifest/boundary:
  - `local_live_analysis/cross_exchange_mvp_shared_kernel_0625T004/shared_kernel_manifest.json`
  - `local_live_analysis/cross_exchange_mvp_shared_kernel_0625T004/boundary_manifest.json`
- 使用 `0625T005` QA-accepted production shadow manifest/boundary:
  - `local_live_analysis/cross_exchange_mvp_production_shadow_0625T005/production_shadow_manifest.json`
  - `local_live_analysis/cross_exchange_mvp_production_shadow_0625T005/boundary_manifest.json`
- 兼容性只读参考已有 tiny-live / ledger artifacts:
  - `local_live_analysis/hyperliquid_tiny_live_m1_canary_loop_0618T007/`
  - `local_live_analysis/hyperliquid_tiny_live_m2_pnl_ledger_0618T008/`
- 未采集新数据，未执行 AWS/remote alignment，未读取 credential，未调用 private/account/order/cancel endpoint，未初始化 live client，未下单，未修改 watcher 或 production config。

## Implementation

- 新增 validator:
  - `examples/hyperliquid/cross_exchange_mvp_audit_replay_contract.py`
- 新增 focused tests:
  - `examples/hyperliquid/test_cross_exchange_mvp_audit_replay_contract.py`
- 新增 output package:
  - `local_live_analysis/cross_exchange_mvp_audit_replay_contract_0625T006/`

Validator 行为:

- 定义 MVP audit/replay contract schema，覆盖:
  - run/event/decision/order identifiers
  - Binance/Hyperliquid timestamps and source ages
  - dual-market top5 market view
  - signal/fair-mid/side/quote intent
  - block/would-submit decision fields
  - submit/resting/reject/cancel/fill lifecycle
  - fee/inventory/PnL fields
  - local boundary flags
- 生成稳定 schema hash。
- 验证 synthetic lifecycle fixtures。
- 对已有 T005/M1/M2 artifacts 做兼容性分类；缺少 fill/fee/inventory/PnL 的历史 artifacts 只作为 partial/fail-closed reference，不升级为 MVP live success 证据。
- 保留 T005 median edge caveat 和 T003 warning bucket。

## Results

- Final recommendation:
  - `audit_replay_contract_ready_for_qa`
- Schema version:
  - `cross_exchange_mvp_audit_replay_contract_v1`
- Schema hash:
  - `0a899c61d63cf5326e16fa8b2d95ae7dc965b04ada72f3ba99811abfca0b9ab5`
- Field count:
  - `63`
- Required field count:
  - `44`
- Schema categories:
  - `boundary`
  - `decision`
  - `economics`
  - `identifier`
  - `lifecycle`
  - `market_view`
  - `metadata`
  - `quote_intent`
  - `signal`
  - `source_age`
  - `timestamp`
  - `venue`

Synthetic lifecycle validation:

- `accepted_lifecycle`: `9` rows, expected `pass`, actual `pass`, issue count `0`.
- `fail_closed_lifecycle`: `2` rows, expected `fail_closed`, actual `fail_closed`, issue count `5`.
- Fail-closed reason codes:
  - `boundary_violation`
  - `missing_or_invalid_fill_px`
  - `missing_or_invalid_fill_qty`
  - `missing_order_identity_for_lifecycle`
  - `missing_required_field`

Existing artifact compatibility:

- `0625T005_production_shadow`:
  - `accepted_partial_contract`
  - accepted scope: public market, signal, fair-mid, side, quote intent, would-submit, counterfactual markout, boundary.
  - unsupported fields: submit/resting/reject/cancel/fill, fee/inventory/realized PnL.
- `0618T007_m1_repeated_canary`:
  - `accepted_lifecycle_reference_no_fill_pnl`
  - accepted scope: submit attempt, resting status, tracked cancel, final open-orders empty.
  - unsupported fields: fill/fee/inventory/realized PnL.
  - conservative reason: `no_fill_or_settlement_evidence`.
- `0618T008_m2_pnl_ledger`:
  - `accepted_fail_closed_ledger_reference`
  - accepted scope: ledger summary, source completeness, realized PnL proof status, boundary.
  - unsupported fields: realized PnL proof.
  - conservative reason: `fail_closed_no_realized_live_pnl`.

Caveats preserved:

- T005 median adjusted counterfactual edge caveat remains visible: median is `-1.5` ticks.
- T003 warning bucket remains preserved for replay diagnostics.
- This T006 result defines a contract/validator only; it does not prove public replay alignment and does not authorize T008 live submit.

## Output Package

- `audit_schema_manifest.json`
- `replay_input_contract.json`
- `schema_hash.json`
- `synthetic_lifecycle_fixtures.csv`
- `synthetic_lifecycle_validation.csv`
- `validation_issues.csv`
- `existing_artifact_compatibility.csv`
- `boundary_manifest.json`
- `validation_report.md`
- `audit_replay_contract_manifest.json`

## Verification

- `python -m pytest examples/hyperliquid/test_cross_exchange_mvp_audit_replay_contract.py -q`
  - Result: `6 passed`
- `python -m py_compile examples/hyperliquid/cross_exchange_mvp_audit_replay_contract.py examples/hyperliquid/test_cross_exchange_mvp_audit_replay_contract.py`
  - Result: passed
- `python examples/hyperliquid/cross_exchange_mvp_audit_replay_contract.py --help`
  - Result: passed
- Formal runner:
  - `python examples/hyperliquid/cross_exchange_mvp_audit_replay_contract.py --output-dir local_live_analysis/cross_exchange_mvp_audit_replay_contract_0625T006`
  - Result: generated final package with `audit_replay_contract_ready_for_qa`
- JSON parse:
  - passed for `audit_schema_manifest.json`, `replay_input_contract.json`, and `boundary_manifest.json`.
- CSV/artifact validation:
  - required artifacts all present and non-empty
  - `synthetic_lifecycle_validation.csv`: `2` validation rows
  - `existing_artifact_compatibility.csv`: `3` compatibility rows
  - boundary flags are true
- Deterministic reproduction:
  - reran output to `/tmp/0625T006_repro.ddaJCK`
  - 5 JSON, 4 CSV and 1 markdown artifact matched after excluding output file paths.
- Combined focused regression:
  - `python -m pytest examples/hyperliquid/test_cross_exchange_signal_acceptance.py examples/hyperliquid/test_cross_exchange_shared_signal_kernel.py examples/hyperliquid/test_cross_exchange_production_shadow.py examples/hyperliquid/test_cross_exchange_mvp_audit_replay_contract.py -q` -> `15 passed`
- Combined `py_compile`:
  - Result: passed
- `git diff --check`
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
  - `audit_replay_contract_ready_for_qa`
- Status:
  - `待验收`
- No watcher/live strategy behavior changed.
- No private/order endpoints were used.
- No orders were placed.
- No canary/promotion was authorized.

## Commit

- commit: `5c7e94b`
- 提交信息: `0705 define cross exchange audit replay contract t006`
