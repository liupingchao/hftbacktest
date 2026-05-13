```md
执行线程：
- 测试线程

任务ID：
- 0512T003

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 正常验收。重点检查 `5-11-night-active` 是否已完成本地 audit replay、maker acceptance、cancel-fill 风险诊断和归档刷新。

files：
- local_live_analysis/5-11-night-active/audit_live_5-11-night-active.csv
- local_live_analysis/5-11-night-active/raw_market_data/btcusdt_20260511.gz
- local_live_analysis/5-11-night-active/out/backtest_audit_replay/audit_bt_audit_replay.csv
- local_live_analysis/5-11-night-active/backtest_audit_replay_result.json
- local_live_analysis/5-11-night-active/alignment_report_audit_replay.json
- local_live_analysis/5-11-night-active/maker_acceptance.json
- local_live_analysis/5-11-night-active/out/cancel_fill_risk/
- local_live_analysis/stage6j_regime_control_cancel_fill_risk_5-11/
- local_live_analysis/archive/5-11-night-active.tar.gz
- local_live_analysis/archive/5-11-night-active.tar.gz.sha256
- .workflow/tasks/0512T003.md
- .workflow/reports/0512T003-business.md
- task_plan.md
- progress.md
- findings.md

action：
- 确认 `5-11-night-active` 已拉回本地，live audit rows `1,018,503`，decision rows `802,999`。
- 使用 conda 环境 `/home/molly/anaconda3/envs/hftbacktest/bin/python`，并通过 `PYTHONPATH=py-hftbacktest:examples/binance_tick_mm` 指向当前仓库本地 `py-hftbacktest`。
- 补跑 audit replay，并生成 `backtest_audit_replay_result.json`、`alignment_report_audit_replay.json`、`maker_acceptance.json`。
- 跑单样本 cancel-fill 风险诊断，再把 `5-9-noon`、`5-10-day-control-1h-06`、`5-11-night-active` 合并成 current-format 跨样本风险对比。
- 刷新 `FILE_MANIFEST.txt`、`SHA256SUMS.txt`、`local_live_analysis/archive/5-11-night-active.tar.gz` 和 archive sha256。
- 未修改策略代码，未启动新的 live test。

verify：
- `PYTHONPATH=py-hftbacktest:examples/binance_tick_mm /home/molly/anaconda3/envs/hftbacktest/bin/python examples/binance_tick_mm/compare_audit.py --bt local_live_analysis/5-11-night-active/out/backtest_audit_replay/audit_bt_audit_replay.csv --live local_live_analysis/5-11-night-active/audit_live_5-11-night-active.csv --align-mode seq --out local_live_analysis/5-11-night-active/alignment_report_audit_replay.json`
  - exit 0，输出 `alignment_report_audit_replay.json`。
- `PYTHONPATH=py-hftbacktest:examples/binance_tick_mm /home/molly/anaconda3/envs/hftbacktest/bin/python examples/binance_tick_mm/maker_acceptance.py --alignment-report local_live_analysis/5-11-night-active/alignment_report_audit_replay.json --backtest-result local_live_analysis/5-11-night-active/backtest_audit_replay_result.json --out local_live_analysis/5-11-night-active/maker_acceptance.json`
  - exit 0，`passed=true`。
- `PYTHONPATH=py-hftbacktest:examples/binance_tick_mm /home/molly/anaconda3/envs/hftbacktest/bin/python examples/binance_tick_mm/analyze_cancel_fill_risk.py --run-id 5-9-noon --run-id 5-10-day-control-1h-06 --run-id 5-11-night-active --local-root local_live_analysis --out-dir local_live_analysis/stage6j_regime_control_cancel_fill_risk_5-11`
  - exit 0，decision `proceed_to_stage6j_narrow_rule`，current-format sample count `3`。
- `tar -czf local_live_analysis/archive/5-11-night-active.tar.gz -C local_live_analysis 5-11-night-active`
  - exit 0。
- `sha256sum local_live_analysis/archive/5-11-night-active.tar.gz`
  - archive sha256 `6c6b36833aedcedb01f47c8aa0ff1b91c0344870255cf34d49c023f55e55f389`。

done：
- `maker_acceptance` 通过：
  - `action_match_rate=1.0`
  - `planned_action_match_rate=1.0`
  - `reject_reason_match_rate=1.0`
  - `throttle_reason_match_rate=1.0`
  - `working_order_lifecycle.semantic_mismatch_rows=0`
  - `working_order_lifecycle.blocking_mismatch_rows=0`
  - `api_throttle.mismatch_attribution.mismatch_rows=0`
  - `api_throttle.mismatch_attribution.target_tick_mismatch_rows=0`
  - `replay_lag.missing_lag_rows=0`
  - `replay_lag.missing_exchange_lag_rows=0`
  - strict replay gate enabled/strict/passed = `true/true/true`
  - strict replay gate breach/drop/fail = `0/0/0`
  - post-startup outside dual gate rows = `0`
- Audit replay summary:
  - common rows `802,996`
  - scheduled/consumed `802,999 / 802,996`
  - unconsumed `3`
  - post-startup lag breaches `0`
  - `pnl_mtm=-7.6108`
  - `max_abs_position_notional=246.29655`
  - `drop_latency_rate=0.22386786484615118`
  - `drop_api_rate=0.1674167741806933`
- 诊断项：
  - non-blocking/diagnostic working-order mismatch rows `712,579`，但 hard gate 的 semantic/blocking mismatch 为 `0`。
  - top5 feed-state parity 仍不是完全一致：top5 tick match rate about `0.9044`，top5 qty match rate about `0.8898`；当前不阻塞 action-path acceptance。
- `5-11-night-active` cancel-fill 风险：
  - fills `915`
  - fill-after-cancel-request `391`
  - notional rate `0.427300`
  - same-side readd while cancel-requested `1436`
  - same-side readd then cancel-fill `13`
  - worsening cancel-fill `191`
  - add-side candidate `201`
  - adverse-selection / inventory-reducing candidate `190`
  - p90 cancel-to-fill latency `36.228ms`
  - max abs position after cancel-requested fill `0.003`
- 跨样本 current-format 风险对比：

| run | fills | cancel-fill | notional rate | add-side candidate | adverse-selection candidate |
|---|---:|---:|---:|---:|---:|
| `5-9-noon` | 75 | 28 | 0.373264 | 15 | 13 |
| `5-10-day-control-1h-06` | 46 | 21 | 0.456445 | 7 | 14 |
| `5-11-night-active` | 915 | 391 | 0.427300 | 201 | 190 |

- 跨样本 decision：`proceed_to_stage6j_narrow_rule`。
- 结论：
  - `5-11-night-active` 是可用的 current-format 4H night-active 验收样本：本地 archive 已刷新，audit replay 已完成，`maker_acceptance` hard gates 通过。
  - live/backtest hard gates 已对齐：action/planned/reject/throttle match rate 均为 `1.0`，working semantic/blocking mismatch 为 `0/0`，API/throttle mismatch 为 `0`，strict replay lag breach/drop/fail 为 `0/0/0`，post-startup outside dual gate rows 为 `0`。
  - 这个样本适合作为后续主开发/诊断样本，因为 4H 窗口提供了更大的事件量：fills `915`，fill-after-cancel-request `391`，notional rate `0.427300`，add-side candidates `201`，adverse-selection / inventory-reducing candidates `190`。
  - cancel-requested fill 风险在更长 night-active 窗口里明显重复并放大，且同时包含 add-side inventory-worsening path 与 inventory-reducing/adverse-selection path；这支持进入 narrow rule 设计/离线验证，而不是 maker 参数搜索。
  - 不建议直接进入 maker 参数优化、live micro test 或默认开启任何规则。后续必须先结合 `0512T001` 的 Stage 6J source-path observability 限制和 `0512T004` 的观测门禁，再推进 `0512T002` 的 add-side submit/re-add toxic timing rule 设计。

blockers：
- 无执行阻塞。
- 业务阻塞：Stage 6J 仍不能单独证明 live adverse-selection source-path 改善，需要后续规则设计合同补充 action-path coverage / live-event observability gate。

commit：
- 无

提交信息：
- 无
```
