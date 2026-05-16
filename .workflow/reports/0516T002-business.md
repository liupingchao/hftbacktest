```md
执行线程：
- 业务线程-python

任务ID：
- 0516T002

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0516T002.md`
- `.workflow/reports/0516T002-business.md`
- `examples/binance_tick_mm/replay_lifecycle_mismatch_diagnosis.py`
- `examples/binance_tick_mm/test_replay_lifecycle_mismatch_diagnosis.py`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 在现有 read-only replay lifecycle diagnosis runner 上新增 `0516T002` repeatability mode：
  - 新增 CLI 参数 `--queue-ahead-repeatability`
  - 在现有 `5-13-day-control-30min` 内扫描 live no-fill / later-canceled touch candidates
  - 区分 replay-fill candidate 与 proxy-only fill-candidate
  - 计算 queue-ahead proxy metrics
- 新增输出：
  - `queue_ahead_repeatability_summary.md`
  - `queue_ahead_candidate_cases.csv`
  - `queue_ahead_proxy_metrics.csv`
  - `queue_ahead_depth_trade_windows.csv`
  - `queue_ahead_bucket_summary.csv`
  - `run_manifest.json`
- 本轮保持 read-only 边界：
  - 未修改 replay fill model
  - 未修改 queue / priority / touch fill 逻辑
  - 未修改策略
  - 未补样本
  - 未启动 live

verify：
- `python -m pytest examples/binance_tick_mm/test_replay_lifecycle_mismatch_diagnosis.py`
  - 结果：`7 passed`
- `python examples/binance_tick_mm/replay_lifecycle_mismatch_diagnosis.py --help`
  - 结果：通过，新增 `--queue-ahead-repeatability`
- `python examples/binance_tick_mm/replay_lifecycle_mismatch_diagnosis.py --run-dir local_live_analysis/5-13-day-control-30min --output-dir local_live_analysis/5-13-day-control-30min/stage6i_queue_ahead_repeatability_0516T002 --queue-ahead-repeatability`
  - 结果：通过

done：
- 真实样本产物：
  - `local_live_analysis/5-13-day-control-30min/stage6i_queue_ahead_repeatability_0516T002/`
- coverage：
  - matched submit rows：`2516 / 2516`
  - price tick equal rows：`2516 / 2516`
  - qty equal rows：`2516 / 2516`
- repeatability counts：
  - candidate_cases：`366`
  - replay_fill_candidate_cases：`1`
  - proxy_only_candidate_cases：`365`
  - queue_ahead_mismatch_cases：`366`
  - strong_queue_ahead_mismatch_cases：`325`
  - replay_fill_queue_ahead_mismatch_cases：`1`
  - target_4948_cases：`1`
- `4948` 结果：
  - `4948` 是唯一 replay-fill queue-ahead mismatch case
  - same-price trade qty：`8.884`
  - submit visible qty：`21.143`
  - anchor/replay-fill visible qty：`15.633`
  - same-price qty / submit visible qty：`0.4202`
  - order-at-touch share：`1.0`
- 结论：
  - queue-ahead proxy no-fill pattern 在当前样本里重复出现，不是只有 `4948`
  - 但 replay-fill 版本仍然只有 `4948` 一个
  - 因此 `4948` 不是 queue-ahead 现象的孤例，但仍是 replay false-positive 的孤例
  - `repeatability_decision = repeated_proxy_queue_ahead_mismatch`
  - `repair_design_readiness = not_ready_proxy_only`
- 是否足够开 repair：
  - 不足够直接进入 generalized queue/touch repair design 或 implementation
  - 如果后续要推进，应先设计更保守的 proxy-only research / calibration gate，而不是直接改 replay fill model
  - 仍缺 exact queue position、order-id-level queue depletion、以及更多 replay-fill false-positive cases

blockers：
- 无

commit：
- 待提交

提交信息：
- 待提交
```
