```md
执行线程：
- 业务线程-python

任务ID：
- 0518T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0518T001.md`
- `.workflow/reports/0518T001-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`

action：
- 基于 `0516T001` / `0516T002` 的 queue-ahead proxy evidence，产出 conservative queue proxy gate repair design contract。
- 本任务保持 design-only 边界：
  - 未实现 gate
  - 未修改 replay fill model
  - 未修改 queue / priority / touch fill 逻辑
  - 未修改策略
  - 未补样本
  - 未启动 live

4948 case evidence summary：
- target order：`4948`
- submit_key：`28940|sell`
- case_label：`live_canceled_replay_filled`
- side：`sell`
- order price tick：`811327`
- replay fill 发生在 live cancel request 前约 `262.28ms`
- replay fill 前同价 supportive aggressive trade qty：`8.884`
- replay fill 前同价 supportive aggressive trade count：`31`
- submit visible ask qty at order price：`21.143`
- replay-fill visible ask qty at order price：`15.633`
- same-price qty / submit visible qty：`0.4202`
- same-price qty / replay-fill visible qty：`0.5683`
- order-at-touch share submit -> replay fill：`1.0`
- diagnosis class：`queue_ahead_depth_can_absorb_observed_trades`
- interpretation：
  - replay fill 不是 hidden trigger；市场确实有同价 aggressive trades 打到 touch。
  - 但同价成交量小于 submit/replay-fill visible queue proxy。
  - live no-fill 合理，因为可见 queue ahead 足以吸收观察到的同价成交量。
  - replay fill 可疑，因为 replay 可能把“touch 有成交”近似成“我方订单可成交”，缺少 queue-ahead / priority / order-exposure 状态。

conservative queue proxy gate design：
- 适用对象：
  - 只用于 replay-fill candidate gating。
  - 只处理 live/replay residual diagnosis 或未来 repair candidate 中的 touch-level fill optimism。
  - 不用于策略实时 quote decision，不用于 live promotion 判断。
- gate 输入字段：
  - `submit_key`
  - `order_id`
  - `side`
  - `order_price_tick`
  - `submit_ts`
  - `replay_fill_ts`
  - `live_cancel_request_ts`
  - `same_price_trade_qty_submit_to_fill`
  - `same_price_trade_count_submit_to_fill`
  - `same_price_trade_qty_window_10ms/25ms/50ms/100ms`
  - `submit_visible_qty_at_order_price`
  - `replay_fill_visible_qty_at_order_price`
  - `top1_visible_qty_decay`
  - `unexplained_depth_shrink`
  - `order_at_touch_share`
  - `order_at_touch_duration_ms`
  - `quote_age_ms`
  - `join_age_ms`
  - `latency_bucket`
  - `gap_crossed`
  - `join_stale`
- candidate conservative thresholds：
  - Require `order_at_touch_share >= 0.95` before applying this gate. If the order was not mostly at touch, this is not a queue-ahead touch-fill case.
  - Treat replay fill as suspicious when:
    - `same_price_trade_qty / submit_visible_qty < 1.0`
    - and `same_price_trade_qty / replay_fill_visible_qty < 1.0`
    - and there is no strong unexplained top1 shrink that would imply the queue ahead was depleted off-trade.
  - Treat cases as stronger suspicious candidates when both ratios are `< 0.75`; `4948` falls here with `0.4202` and `0.5683`.
  - Do not gate on thin evidence if `submit_visible_qty` or `replay_fill_visible_qty` is missing/zero; classify as insufficient evidence instead.
  - Do not apply when `join_stale = true`, `gap_crossed = true`, or join age exceeds the accepted market-view threshold; classify as data-quality-blocked.
- top1 visible qty decay / unexplained shrink handling：
  - If top1 visible qty decays below same-price trade qty, the gate should not suppress fill without additional evidence.
  - If visible qty remains above same-price trade qty, the gate can mark replay fill as queue-ahead-suspicious.
  - If visible qty shrinks materially without matching trades, classify separately as `unexplained_depth_shrink_present`; this needs a more careful rule because cancels ahead can improve our queue position.
- latency / join-age / stale guards：
  - The gate must only run on market-view rows that pass Stage 3-style join quality.
  - High latency buckets should be reported as strata, not used alone as suppress conditions.
  - Stale joins, future joins, missing joins, and gap-crossed joins should block the gate from making a repair decision.
- replay-fill candidate gating semantics：
  - Proposed future behavior is not "always cancel replay fill".
  - Future implementation should first output diagnostics:
    - `queue_proxy_gate_candidate`
    - `queue_proxy_gate_reason`
    - `queue_proxy_gate_evidence_strength`
    - `queue_proxy_gate_data_quality`
  - Only after multi-sample validation should it be allowed to convert a replay fill into non-fill / keep-working / cancel-aligned lifecycle.
  - Any conversion must be default-off or diagnostic-only before becoming an acceptance gate.
- failure modes and false-negative risk：
  - False suppress risk: visible queue can shrink from cancels ahead, so same-price trade qty below visible qty does not prove no fill.
  - False keep-fill risk: hidden liquidity, partial queue depletion, and missing order-id-level position can still make live fill possible.
  - Overfit risk: current replay-fill false-positive evidence is still only `4948`.
  - Data-quality risk: top5 sidecar / join age / stale rows can create bad queue proxies.
  - Scope risk: this gate must not be described as exact queue position or MBO reconstruction.

implementation prerequisites：
- 更多 current-format samples。
- 更多 replay-fill false-positive cases，或者跨样本 repeatability evidence。
- 至少一个 QA-accepted diagnostic-only run 证明：
  - gate 能识别 `4948` 类 case
  - gate 不会大量误伤正常 replay fills
  - gate 不只是 overfit `4948`
- 明确的数据质量过滤：
  - accepted market-view rows only
  - no stale/future/missing/gap-crossed joins
- 明确默认状态：
  - first implementation should be diagnostic-only / default-off
  - no live promotion authority

current decision：
- 当前不实施修复。
- 原因：
  - `0516T002` 证明 queue-ahead proxy no-fill pattern 重复出现，但 replay-fill false-positive 仍只有 `4948` 一条。
  - 缺 exact queue position 和 order-id-level queue depletion。
  - 直接改 replay fill model 会有过拟合和误杀真实 fill 的风险。
- 下一步应等待更多 current-format samples / replay false-positive cases，再开单独 repair implementation task。

verify：
- 人工检查 `.workflow/tasks/0518T001.md` 和 `.workflow/reports/0518T001-business.md`
- `python3 .workflow/build_dashboard.py`
- `git diff --check`

done：
- 已明确 `4948` case 的 evidence summary。
- 已设计 conservative queue proxy gate 的候选输入、阈值、guard、gating semantics 和 failure modes。
- 已明确当前不实施修复。
- 已明确后续等更多数据 / 更多 case 后再修复。

blockers：
- 无

commit：
- 待提交

提交信息：
- 待提交
```
