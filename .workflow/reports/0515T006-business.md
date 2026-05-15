```md
执行线程：
- 业务线程-python

任务ID：
- 0515T006

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0515T006.md`
- `.workflow/reports/0515T006-business.md`
- `examples/binance_tick_mm/replay_lifecycle_mismatch_diagnosis.py`
- `examples/binance_tick_mm/test_replay_lifecycle_mismatch_diagnosis.py`

action：
- 在现有 read-only diagnosis runner 上增加 `0515T006` single-case mode：
  - 新增 `--single-order-id`
  - 用于只分析单一 residual replay fill case
- 这轮只分析 `28940|sell` / `4948`：
  - 不修改 replay fill model
  - 不修改 touch fill / queue / strategy 逻辑
- 新增 single-case diagnosis 输出：
  - tick-normalized supportive trade counting，避免浮点价格比较把 `81132.7` 误判成不支持成交
  - replay fill 前 `10/25/50/100/250/500/1000/5000ms` supportive trade counts
  - submit / replay fill / live cancel 的 top1 sidecar context
  - single-case trigger classification
  - evidence sufficiency / missing evidence 判断
- 真实样本运行：
  - `python examples/binance_tick_mm/replay_lifecycle_mismatch_diagnosis.py --run-dir local_live_analysis/5-13-day-control-30min --output-dir local_live_analysis/5-13-day-control-30min/stage6g_single_residual_fill_diagnosis_0515T006 --single-order-id 4948`

verify：
- `python -m pytest examples/binance_tick_mm/test_replay_lifecycle_mismatch_diagnosis.py`
- `python examples/binance_tick_mm/replay_lifecycle_mismatch_diagnosis.py --help`
- `python examples/binance_tick_mm/replay_lifecycle_mismatch_diagnosis.py --run-dir local_live_analysis/5-13-day-control-30min --output-dir local_live_analysis/5-13-day-control-30min/stage6g_single_residual_fill_diagnosis_0515T006 --single-order-id 4948`

done：
- 真实样本产物：
  - `local_live_analysis/5-13-day-control-30min/stage6g_single_residual_fill_diagnosis_0515T006/`
- 单 case 结论：
  - target order: `4948`
  - submit_key: `28940|sell`
  - case_label: `live_canceled_replay_filled`
  - classification: `queue_exposure_proxy_bias_possible`
- 关键证据：
  - replay fill 发生在 live cancel request 前约 `262.284544ms`
  - replay fill 前 `10ms / 25ms / 50ms / 100ms` supportive trades 计数：
    - `13 / 13 / 13 / 14`
  - 最近一笔 supportive trade 距 replay fill 仅约 `0.503ms`
  - replay fill 时订单仍在 touch：
    - replay fill top1 bid/ask = `81132.6 / 81132.7`
    - order price = `81132.7`
  - submit 时 top1 ask qty 约 `21.143`
  - replay fill 时 top1 ask qty 约 `15.633`
  - live cancel 时 top1 ask 已经下移到 `81124.6`
- 解释：
  - `4948` 不再像“没有可观测的 fill trigger”
  - raw trade 和 top1 sidecar 都支持：replay fill 前后确实存在打到该价位的主动买成交
  - 但 live 没有 fill，且更晚才 cancel，这更像 queue-exposure / priority approximation 偏差，而不是 hidden trigger path
- 是否足够直接开 repair：
  - 当前 runner 标成 `evidence_sufficient_for_repair = 0`
  - 原因不是没有方向，而是：
    - 缺少 exact queue position 证据
    - 仍只有单 case
    - 不足以安全推出 generalized touch/queue repair
- 因此更准确的结论是：
  - 已经足以把 `4948` 从“完全未知”收窄成 `queue_exposure_proxy_bias_possible`
  - 但还不够支持直接开一个 repair implementation task
  - 如果后续继续，应优先做更强的 queue / priority proxy diagnosis 或跨 case repeatability 检查，而不是立刻改 fill model

blockers：
- 无

commit：
- 待提交

提交信息：
- 待提交
```
