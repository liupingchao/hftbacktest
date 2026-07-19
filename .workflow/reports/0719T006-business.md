# 线程回报

执行线程：
- 总控 auto-loop / 业务实现线程

任务ID：
- 0719T006

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0719T006.md`
- `examples/hyperliquid/cross_exchange_live_remote_orchestrator.py`
- `examples/hyperliquid/cross_exchange_t024_same_window_acceptance.py`
- `examples/hyperliquid/hyperliquid_maker_order_manager.py`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/test_cross_exchange_live_integrated_offline_acceptance.py`
- `examples/hyperliquid/test_cross_exchange_live_remote_orchestrator.py`
- `examples/hyperliquid/test_cross_exchange_t024_same_window_acceptance.py`
- `examples/hyperliquid/test_hyperliquid_maker_order_manager.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`

action：
- Orchestrator 新增显式 exact profile：
  - `legacy-single-order` 和 `two-sided-manager`；
  - exact 运行必须显式选 profile，非 exact 运行不得携带 profile；
  - two-sided profile 固定 one window、`<=900s`、`0.005 BTC/order`、`0.01 BTC position`、`1 USDC loss`、两次 submissions、edge-gate mode、manager、requote `2`、fast L2、`live_open_orders` private proof 和 Binance public lead。
- Watcher CLI 将 `requote_attempts` 和 `max_real_order_submissions` 作为独立参数传播。
- Manager submit action 记录真实脱敏 `client.order()` response 和实际 endpoint-call fact；rejected/unknown 也计入 submission cap/evidence。
- Two-sided manager artifact 固定 buy 后 sell：
  - 两条 intent、attempt、status 和 actual order result；
  - 独立 attempt `1/2` 和 task/window-scoped keys；
  - 每侧一条 tracked reference 和 target-bound cancel proof；
  - 移除聚合 `side=buy+sell` 主证据。
- Approved config、watcher、edge/anti-drift manifest 补齐 task/window identity。
- Task 12 acceptance 升级到双边 manager v3 contract：
  - exact profile/command/edge/Binance source；
  - exactly two intents/attempts/results/resting statuses/references；
  - per-side price/size/status identity join；
  - producer raw cancel proof 独立重建。
- 增加 actual writer nominal integration，以及 one-sided、duplicate-side、aggregate-side、stale key、missing manager、wrong mode/budget、forged lifecycle 反例。
- 未修改 quote formula、edge threshold、strategy activation、risk caps、multi-level activation 或 actual quote behavior。
- 本任务未调用 live/private/account/order/cancel/network/remote/service。

verify：
- Exact profile preflight/negative regressions、orchestrator timeout/provenance integration：通过。
- Manager nominal/reject endpoint accounting、actual response persistence、per-side artifact regressions：通过。
- Task 12 acceptance：`65 passed`，包括 actual writer nominal 和全部双边反例。
- 相关 orchestrator/acceptance focused：`85 passed`。
- 相关 manager/watcher focused：通过。
- Full `python -m pytest examples/hyperliquid -q`：`605 passed in 32.60s`。
- Modified Python `py_compile`、三个 CLI `--help`、`git diff --check` 通过。

done：
- 下一 formal live task 已有唯一、显式、fail-closed 的 two-sided manager exact profile。
- Producer-written buy/sell lifecycle evidence 可被独立 acceptance 精确重建。
- 单边、聚合、重复、stale、错 command/budget 和伪造 lifecycle 均 fail closed。
- 离线实现完成，等待独立 QA。

blockers：
- 独立 QA 通过前不得启动新的 live task。
- Principal Task 12 和 Task 10 multi-level prerequisite 仍需下一 formal live lifecycle 和后续 acceptance 裁决。

commit：
- `338975f53ccdb351912b03a54cb58e246c01f8a1`

提交信息：
- `Enforce exact two-sided manager evidence`
