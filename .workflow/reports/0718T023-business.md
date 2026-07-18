# 0718T023 P3-CUMULATIVE-TINY-LIVE-SAME-WINDOW-ACCEPTANCE

执行线程：
- 总控 auto-loop / 业务实现线程

任务ID：
- 0718T023

状态：
- 待验收

更新时间：
- 2026-07-18 23:58 CST

## 执行摘要

本任务按 `event-driven-live` 执行了一个 BTC 单层、fixed-spread、post-only `Alo` tiny-live 窗口。Live 作为主要数据源；未激活 dynamic spread、fill feedback、inventory skew、multi-level 或 manager quote behavior。

远端运行完成，child 正常退出，终止和独立 open-orders proof 均通过。窗口产生了一个真实 `resting -> cancel -> final open orders empty` 生命周期，但在回收证据中发现 task-scoped envelope 没有被实际配置快照执行，且内层 watcher/fill artifacts 使用了旧 task identity。因此本任务按 stop condition 阻断，不启动第二个 live 窗口，也不把该窗口升级为 T023 acceptance。

## 预期 Envelope

- symbol：`BTC`
- route：Binance lead / Hyperliquid maker
- max order size：`0.005 BTC`
- aggregate position delta：`0.01 BTC`
- max loss：`1 USDC`
- max submissions：`2`
- bounded window：`1800s`
- time in force：`Alo`

## 实际 Live 证据

- remote source marker：`c8e606095ba7b1f32226c91fb1e903db631c4469`
- remote/local SHA-256：`62/62` verified，missing `0`，mismatch `0`
- orchestrator：`complete`
- child return code：`0`
- child reaped：`true`
- termination escalation：未发生
- submission count：`1`
- order：BTC buy `0.002`，limit `64105.0`，notional `128.21 USDC`
- TIF：`Alo`
- exchange status：`resting`
- cancel endpoint：已调用
- final owned open orders：`0`
- independent private open-orders proof：`0`
- post account BTC asset position：空
- fill count：`0`
- maker fill count：`0`
- live fill ledger rows：`0`
- liquidity-role evidence rows：`0`
- writer：`cross_exchange_live_status_v2`，healthy，failure count `0`
- activation flags：dynamic spread、fill feedback、multi-level、actual quote behavior 均为 `false`

Public window 记录了 `566` 个 L2 book events、`262` 个 trade events，reconnect count 为 `0`。这些数据只作为 live market-view/decision-path 输入，不被解释为 fill probability、queue priority、fee、PnL 或 maker viability 证明。

## 阻断事实

`approved_config_snapshot.json` 实际记录：

- `max_loss_usdc=30.0`，超出任务要求的 `1.0`
- `max_position_btc=0.04`，超出任务要求的 `0.01`
- `duration_seconds=600`，虽然更紧，但不能抵消前述上限偏离

Artifact identity 也不一致：

- root/orchestrator task：`0718T023`
- watcher manifest task：`0623T007`
- fill-window/executor manifest task：`0622T004`
- attempt key：`0622T004:window_01:attempt_1`

这同时触发：

1. exact authorization envelope mismatch；
2. window/attempt identity mismatch；
3. same-window replay 输入不能被视为 T023 的 exact config/control reproduction。

窗口本身没有观察到 position、fill 或 loss breach；但“实际未发生 breach”不能替代“运行时配置满足 envelope”。

## 代码修复

提交：

- `a57c7da Enforce task-scoped live envelope and artifact identity`

修复内容：

- orchestrator 显式透传 `--max-loss-usdc` 和 `--max-position-btc`；
- event-driven watcher 将 task/window identity 和 scoped caps 传入 fill-window；
- fill ledger、cloid、attempt key、manifest 使用传入 task/window identity；
- executor 接受严格小于全局上限的 task-scoped loss/position caps，并对非正值 fail closed；
- fake watcher/integrated offline fixtures 增加新参数覆盖。

## Verification

- focused executor/watcher/orchestrator：`106 passed`
- fill-loop related tests：`23 passed`
- full `python -m pytest examples/hyperliquid -q`：`455 passed`
- py_compile：通过
- `git diff --check`：通过

## Unsupported Claims

本任务不支持：

- stable profitability；
- fill rate 或 fill probability；
- fee/rebate calibration；
- realized/unrealized PnL；
- exact queue priority；
- maker viability；
- multi-level activation；
- promotion 或 final MVP acceptance。

## 下一步边界

T023 在当前窗口按 stop condition 结束。要继续 live，必须新建正式 repair/re-run task，先在 preflight 中验证 `approved_config_snapshot`、task/window/attempt identity 和 source/config hash，再取得新的 clean live window；不得复用本窗口作为 T023 的 accepted lifecycle 或 replay baseline。
