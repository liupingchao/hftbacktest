# 线程回报

执行线程：
- 总控 auto-loop / 业务实现线程

任务ID：
- 0718T019

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/cross_exchange_online_estimators.py`
- `examples/hyperliquid/cross_exchange_shared_signal_kernel.py`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/test_cross_exchange_online_estimators.py`
- `local_live_analysis/principal_alignment_task8_0718T019/`
- `local_live_analysis/principal_alignment_task8_0718T019-replay/`

action：
- 建立固定 1s event-time bucket 和 state dedupe；同一状态不会按 websocket 消息数重复加权。
- out-of-order/future event 进入 quarantine；每个 bucket 输出 mid return/realized volatility、spread/depth/liquidity、trade direction、toxicity、adverse volume、sweep/depth penetration。
- quote exposure 支持以 side/price/distance/resting start-end/arrival evidence 记录；public-flow helper 使用成交前可见 L2、方向性 at-or-through 成交和 exposure interval。
- side-specific A/k 使用 arrival rate 对 distance 的 log-linear fit，输出样本、有效 bucket、RMSE、confidence 和 A/k confidence bounds。
- `compute_dynamic_half_spread()` 使用 A、k、volatility、risk aversion、inventory、liquidity、toxicity，包含 fixed fallback、hard bounds 和 rate limit；shared kernel overlay 明确 fixed quote authoritative。
- watcher/status/artifacts 记录 estimator snapshot、event rows、bucket/quarantine/exposure/intensity 文件；Task 7 fixed quote path 未读取 dynamic candidate。

verify：
- new estimator/replay/integration tests：`8 passed`。
- watcher + estimator focused：`66 passed`。
- shared-kernel/public-replay/shadow/price regression：`30 passed`。
- executor/fill/kill-switch regression：`103 passed`。
- remote clean isolated clone `cd513f6` py_compile/diff check passed。
- public-only remote window：
  - `60.058079s`；
  - `146` event-driven evaluations；
  - `113` L2 book events、`161` trades、无 reconnect/disconnect；
  - `268` accepted estimator events、`68` event-time buckets、quarantine `0`；
  - shadow would-submit `0`，dynamic candidate fallback fixed `0.5 tick`；
  - credentials/private/account/order/cancel 全 false。
- terminal checksum remote/local：`22/22` pass，missing/mismatch `0`。
- replay：记录 `274` event rows，quote exposure rows `0`，snapshot hash exact match。

done：
- Task 8 observe-only estimator、fixed quote boundary、public-only evidence 和 deterministic replay 已完成。

blockers：
- 当前 live window 没有真实 resting exposure；A/k 不能作为已校准的 live 参数，dynamic-spread activation 不能在本任务内发生。

commit：
- `cd513f6`

提交信息：
- `Add observe-only event-time estimators`
