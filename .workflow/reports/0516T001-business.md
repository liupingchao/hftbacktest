```md
执行线程：
- 业务线程-python

任务ID：
- 0516T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0516T001.md`
- `.workflow/reports/0516T001-business.md`
- `examples/binance_tick_mm/replay_lifecycle_mismatch_diagnosis.py`
- `examples/binance_tick_mm/test_replay_lifecycle_mismatch_diagnosis.py`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 在现有 read-only replay lifecycle diagnosis runner 上新增 `0516T001` queue/priority evidence mode：
  - 新增 CLI 参数 `--queue-priority-order-id`
  - 只分析单一目标 order id，本轮为 `4948`
  - 复用 matched submit comparison、live/replay audit、T009 top5 sidecar 和 raw market gzip
- 新增输出：
  - `queue_priority_evidence.csv`
  - `queue_priority_window_trade_qty.csv`
  - `queue_priority_depth_timeline.csv`
  - `queue_priority_supportive_trades.csv`
  - `QUEUE_PRIORITY_EVIDENCE_DIAGNOSIS_SUMMARY.md`
  - `run_manifest.json`
- 本轮保持 read-only 边界：
  - 未修改 replay fill model
  - 未修改 queue / priority / touch fill 逻辑
  - 未修改策略
  - 未补样本
  - 未启动 live

verify：
- `python -m pytest examples/binance_tick_mm/test_replay_lifecycle_mismatch_diagnosis.py`
  - 结果：`6 passed`
- `python examples/binance_tick_mm/replay_lifecycle_mismatch_diagnosis.py --help`
  - 结果：通过，新增 `--queue-priority-order-id`
- `python examples/binance_tick_mm/replay_lifecycle_mismatch_diagnosis.py --run-dir local_live_analysis/5-13-day-control-30min --output-dir local_live_analysis/5-13-day-control-30min/stage6h_queue_priority_evidence_0516T001 --queue-priority-order-id 4948`
  - 结果：通过

done：
- 真实样本产物：
  - `local_live_analysis/5-13-day-control-30min/stage6h_queue_priority_evidence_0516T001/`
- row counts：
  - `queue_priority_evidence_rows = 1`
  - `window_rows = 8`
  - `depth_timeline_rows = 195`
  - `supportive_trade_rows = 32`
- 单 case 结论：
  - target order: `4948`
  - submit_key: `28940|sell`
  - case_label: `live_canceled_replay_filled`
  - classification: `queue_ahead_depth_can_absorb_observed_trades`
- 关键证据：
  - order price tick: `811327`
  - submit -> replay fill 期间，order 一直在 touch：
    - `order_at_touch_share_submit_to_replay_fill = 1.0`
  - submit -> replay fill 期间，同价 supportive aggressive trade：
    - count: `31`
    - qty: `8.884`
  - submit 时 order price 可见 ask qty：`21.143`
  - replay fill 时 order price 可见 ask qty：`15.633`
  - 同价成交量 / submit visible qty：`0.4202`
  - 同价成交量 / replay-fill visible qty：`0.5683`
  - replay fill 前 `10ms` 同价成交量：`8.0`，仍低于 submit visible qty `21.143`
- 解释：
  - `4948` 的 replay fill 不是 hidden trigger；market 确实打到同价 touch。
  - 但从 submit 到 replay fill 的累计同价成交量小于可见 touch 队列量。
  - 因此 live 没 fill、replay fill 的更合理解释是：replay 缺少 queue-ahead / priority / exposure 状态，把“touch 有成交”近似成“我方订单可成交”。
- 是否足够开 repair：
  - 当前仍为 `evidence_sufficient_for_repair = 0`
  - 缺的是 exact queue position、order-id-level queue depletion、以及重复同类 case
  - 本任务不建议直接进入 generalized queue/touch repair implementation

blockers：
- 无

commit：
- 待提交

提交信息：
- 待提交
```
