# 0625T003 Business Report

## Status

- 任务状态: `待验收`
- 业务线程: `业务线程-research`
- 最终建议: `signal_contract_accepted_for_shadow`
- QA: 待 QA 验收线程复核；本报告不是 QA 通过结论。

## Scope

- 使用 `0627T001` QA-accepted package:
  - `local_live_analysis/cross_exchange_mvp_hl_fast_sample_expansion_0627T001/`
- 执行 offline/public-only/read-only out-of-sample signal acceptance。
- 未采集新数据，未执行 AWS/remote alignment，未读取 credential，未调用 private/account/order/cancel endpoint，未下单，未运行 shadow/canary，未修改 watcher 或 production config。

## Implementation

- 新增 runner:
  - `examples/hyperliquid/cross_exchange_signal_acceptance.py`
- 新增 focused tests:
  - `examples/hyperliquid/test_cross_exchange_signal_acceptance.py`
- 新增 output package:
  - `local_live_analysis/cross_exchange_mvp_signal_acceptance_0625T003/`

Runner 行为:

- 读取 `sample_expansion_manifest.json`、独立 `boundary_manifest.json` 和 `symmetric_edge_context_coverage.csv`。
- Fail-closed input gate 要求:
  - source recommendation 为 `sample_contract_ready_for_signal_acceptance`
  - `t003_creation_unlocked=true`
  - source boundary flags 确认 public-only/offline/no-private/no-order/no-live/no-promotion。
- 只保留:
  - `valid_for_1000ms_signal_acceptance=true`
  - `nominal_horizon_ms=1000`
  - `1000ms <= effective_future_age_ms <= 1250ms`
- 使用 leave-one-window-out split；每折只用两个 train windows 选择 normalization、threshold、candidate 和 side mapping，再在 held-out window 验证。
- 预注册 threshold grid: `0.0/0.5/1.0` absolute train z-score。
- 冻结合约只允许 Binance lead eligible candidates；context fields 仅作为 diagnostic candidates / stability buckets。

## Input Gate

- Input package: `local_live_analysis/cross_exchange_mvp_hl_fast_sample_expansion_0627T001/`
- Required artifacts present and non-empty.
- `input_gate_report.json` result: `gate_passed=true`.
- Effective-horizon rows:
  - `xemm_0627_t001_hlfast_utc16_a`: context `3592`, complete `3591`, valid `3587`
  - `xemm_0627_t001_hlfast_utc17_b`: context `3581`, complete `3581`, valid `3567`
  - `xemm_0627_t001_hlfast_utc17_c`: context `3574`, complete `3573`, valid `3550`
  - aggregate valid rows: `10704`

## Accepted Signal Contract

- Contract file: `accepted_signal_contract.json`
- Candidate: `binance_lead_composite`
- Feature schema:
  - `input_binance_top5_imbalance`
  - `input_binance_microprice_minus_mid_ticks`
  - `input_binance_mid_move_ticks_from_prev`
- Horizon: `1000ms`
- Effective-horizon row condition:
  - `valid_for_1000ms_signal_acceptance=true`
  - `1000ms <= effective_future_age_ms <= 1250ms`
- Normalization: train-fold z-score mean/std.
- Threshold: `abs(z) >= 1.0`
- Side mapping:
  - `positive_signal_buy_negative_signal_sell`
- Edge formula:
  - `fair_mid_px = current_hyperliquid_mid + signed_expected_move_ticks * tick_size`
  - `buy_edge_ticks = (fair_mid_px - quote_px) / tick_size`
  - `sell_edge_ticks = (quote_px - fair_mid_px) / tick_size`
- Acceptance buffer used for offline proxy:
  - `fee_adverse_buffer_ticks=1.5`

## Held-Out Performance

| Held-out window | Active rows | Coverage | Direction hit nonzero | Mean signed label ticks | Adjusted edge proxy ticks | Wrong-way nonzero | Contribution |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `xemm_0627_t001_hlfast_utc16_a` | 433 | 0.12071369 | 0.76530612 | 15.02309469 | 13.52309469 | 0.23469388 | 0.38660714 |
| `xemm_0627_t001_hlfast_utc17_b` | 284 | 0.07961873 | 0.87234043 | 3.39788732 | 1.89788732 | 0.12765957 | 0.25357143 |
| `xemm_0627_t001_hlfast_utc17_c` | 403 | 0.11352113 | 0.96428571 | 5.52109181 | 4.02109181 | 0.03571429 | 0.35982143 |

Notes:

- Same candidate, threshold and side mapping were selected in all three folds.
- Aggregate held-out active rows were not dominated by a single window; max contribution was `0.38660714`.
- Direction hit rate including zero labels is low because most 1000ms rows have no mid move; nonzero direction hit is reported separately and is the acceptance stability metric.
- `regime_stability.csv` has one warning bucket with negative adjusted proxy: `xemm_0627_t001_hlfast_utc17_b / binance_source_age_mid`, active rows `89`, adjusted proxy `-0.93820225`. This is recorded as a warning in `signal_acceptance_manifest.json`; it should be reviewed before any step beyond public shadow.

## Output Package

Required artifacts:

- `signal_acceptance_manifest.json`
- `input_gate_report.json`
- `train_eval_split_manifest.json`
- `effective_horizon_acceptance.csv`
- `candidate_signal_matrix.csv`
- `heldout_performance_by_window.csv`
- `side_mapping_evidence.csv`
- `threshold_sensitivity.csv`
- `regime_stability.csv`
- `accepted_signal_contract.json`
- `boundary_manifest.json`
- `recommendation.md`

Final recommendation:

- `signal_contract_accepted_for_shadow`

This recommendation only unlocks QA consideration for the next production-equivalent public shadow task after QA acceptance. It does not authorize watcher/live strategy behavior changes, private/order endpoints, live orders, shadow execution, canary, promotion, or Hyperliquid replay/alignment claims.

## Verification

- `python -m pytest examples/hyperliquid/test_cross_exchange_signal_acceptance.py -q`
  - Result: `3 passed`
- `python -m py_compile examples/hyperliquid/cross_exchange_signal_acceptance.py`
  - Result: passed
- `python examples/hyperliquid/cross_exchange_signal_acceptance.py --help`
  - Result: passed
- Formal runner:
  - `python examples/hyperliquid/cross_exchange_signal_acceptance.py --input-dir local_live_analysis/cross_exchange_mvp_hl_fast_sample_expansion_0627T001 --output-dir local_live_analysis/cross_exchange_mvp_signal_acceptance_0625T003`
  - Result: generated final package with `signal_contract_accepted_for_shadow`
- JSON parse:
  - `python -m json.tool` over all output JSON files passed.
- CSV/artifact validation:
  - required artifacts all present and non-empty
  - `effective_horizon_acceptance.csv`: 3 rows
  - `candidate_signal_matrix.csv`: 10704 rows
  - `heldout_performance_by_window.csv`: 3 rows
  - `side_mapping_evidence.csv`: 3 rows
  - `threshold_sensitivity.csv`: 126 rows
  - `regime_stability.csv`: 21 rows
- Deterministic reproduction:
  - reran output to `/tmp/0625T003_repro`
  - exact deterministic artifacts matched; `signal_acceptance_manifest.json` matched after excluding generated timestamp and absolute artifact/output paths.
- `git diff --check`
  - Result: passed

## Done

- Exact input package consumed:
  - `local_live_analysis/cross_exchange_mvp_hl_fast_sample_expansion_0627T001/`
- Source manifests consumed:
  - `sample_expansion_manifest.json`
  - `boundary_manifest.json`
- Candidate allowlist actually used:
  - Binance lead eligible: `binance_top5_imbalance`, `binance_microprice_minus_mid_ticks`, `binance_mid_move_ticks_from_prev`, `binance_lead_composite`
  - Diagnostic context-only: `context_basis_mid_ticks`, `context_hyperliquid_top5_imbalance`, `context_hyperliquid_microprice_minus_mid_ticks`
- Final recommendation enum:
  - `signal_contract_accepted_for_shadow`
- No watcher/live strategy behavior changed.
- No private/order endpoints were used.
- No orders were placed.
- No shadow/canary/promotion was authorized.

## Commit

- commit: pending final commit
- 提交信息: pending final commit
