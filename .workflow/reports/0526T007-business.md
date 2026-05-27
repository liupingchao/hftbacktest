```md
执行线程：
- 测试线程

任务ID：
- 0526T007

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0526T007.md`
- `.workflow/reports/0526T007-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `local_live_analysis/5-26-active-makeredge-control-180min-a/**`

action：
- 180min current-format no-rule/default-off live control 样本已完成采集、拉回、归档和后处理。
- 补齐了 `t009_fixed_sidecar`、Stage 5、Step 5C、Stage 6、Step 9B、Step 9D 的派生物。
- 仍保持 no-rule / default-off control 边界，没有启用 candidate、没有放宽 guard、没有修改策略、没有做 tiny live 或 promotion。

verify：
- `python examples/binance_tick_mm/binance_top5_provenance.py build-sidecars --input-gz local_live_analysis/5-26-active-makeredge-control-180min-a/raw_market_data/btcusdt_20260526.gz --out-dir local_live_analysis/5-26-active-makeredge-control-180min-a/t009_fixed_sidecar --sample-id 5-26-active-makeredge-control-180min-a --symbol BTCUSDT --tick-size 0.1 --buffer-size 25000000`
- `python examples/binance_tick_mm/binance_top5_provenance.py join-decisions --audit-csv local_live_analysis/5-26-active-makeredge-control-180min-a/audit_live_5-26-active-makeredge-control-180min-a.csv --top5-csv local_live_analysis/5-26-active-makeredge-control-180min-a/t009_fixed_sidecar/top5_sidecar.csv --out-csv local_live_analysis/5-26-active-makeredge-control-180min-a/t009_fixed_sidecar/joined_decisions.csv`
- `python examples/binance_tick_mm/maker_acceptance.py --alignment-report local_live_analysis/5-26-active-makeredge-control-180min-a/alignment_report_audit_replay.json --backtest-result local_live_analysis/5-26-active-makeredge-control-180min-a/backtest_audit_replay_result.json --sidecar-metrics local_live_analysis/5-26-active-makeredge-control-180min-a/t009_fixed_sidecar/metrics.json --joined-decision-metrics local_live_analysis/5-26-active-makeredge-control-180min-a/t009_fixed_sidecar/joined_decisions.metrics.json --top5-sidecar-csv local_live_analysis/5-26-active-makeredge-control-180min-a/t009_fixed_sidecar/top5_sidecar.csv --joined-decisions-csv local_live_analysis/5-26-active-makeredge-control-180min-a/t009_fixed_sidecar/joined_decisions.csv --out local_live_analysis/5-26-active-makeredge-control-180min-a/maker_acceptance.json`
- `python examples/binance_tick_mm/execution_outcome_labels.py --run-dir local_live_analysis/5-26-active-makeredge-control-180min-a --output-dir local_live_analysis/5-26-active-makeredge-control-180min-a/stage5_execution_outcome_labels_0514T005`
- `python examples/binance_tick_mm/quote_anchor_safety.py --run-dir local_live_analysis/5-26-active-makeredge-control-180min-a --output-dir local_live_analysis/5-26-active-makeredge-control-180min-a/stage5c_quote_anchor_safety_0518T004`
- `python examples/binance_tick_mm/execution_outcome_calibration.py --run-dir local_live_analysis/5-26-active-makeredge-control-180min-a --output-dir local_live_analysis/5-26-active-makeredge-control-180min-a/stage6_final_calibration_0519T001`
- `python examples/binance_tick_mm/quote_adjustment_replay.py --run-dir local_live_analysis/5-26-active-makeredge-control-180min-a --output-dir local_live_analysis/5-26-active-makeredge-control-180min-a/stage9b_quote_adjustment_replay_0519T008`
- `python examples/binance_tick_mm/candidate_bucket_refinement.py --run-dir local_live_analysis/5-26-active-makeredge-control-180min-a --output-dir local_live_analysis/5-26-active-makeredge-control-180min-a/stage9d_candidate_bucket_refinement_0526T007`

done：
- 180min 样本采集完成并已归档。
- T009 sidecar/join、Stage 5 labels、Step 5C diagnostics、Stage 6 calibration、Step 9B runner、Step 9D fine-bucket refinement 都已落盘。
- 当前样本仍然只是 current-format no-rule/default-off control data，不支持 live/default-on/promotion。

blockers：
- 无

commit：
- 无

提交信息：
- 无
```
