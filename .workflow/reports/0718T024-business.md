# 线程回报

执行线程：
- 总控 auto-loop / 业务实现线程

任务ID：
- 0718T024

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 验收机制、执行安全、证据完整性和 same-window config/decision/control reproduction。
- 不验收 stable PnL、fill-rate、fee/rebate calibration、queue priority、maker viability、multi-level activation、promotion 或 final MVP pass。

files：
- `.workflow/tasks/0718T024.md`
- `examples/hyperliquid/cross_exchange_live_remote_orchestrator.py`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py`
- `examples/hyperliquid/cross_exchange_t024_same_window_acceptance.py`
- `examples/hyperliquid/test_cross_exchange_live_remote_orchestrator.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_loop.py`
- `examples/hyperliquid/test_cross_exchange_t024_same_window_acceptance.py`
- `local_live_analysis/principal_alignment_task12_repair_0718T024/`

action：
- 修复 preorder-blocked artifact path，确保 run intent、config snapshot、fill-window manifest 和 executor manifest 使用调用方 task/window identity。
- config snapshot 现在显式记录 `task_id`、`window_id`、`artifact_window_id`、task-scoped max loss/position/order size/submission caps。
- orchestrator 增加 `--require-exact-envelope` 和 `--preflight-only`；mismatch 会在 watcher、credential/private/order/cancel 之前 fail。
- 增加 event-driven live artifact same-window acceptance，独立核对 provenance/identity、config/control、decision、lifecycle、risk/fill reconciliation 和 anti-optimism boundary。
- 使用 commit `4c32d991062ad77af9d271cfcb0a8c6c7cf154e1` 创建独立 remote source archive。
- no-network preflight 精确证明：
  - task/window：`0718T024 / 1`
  - BTC、post-only `Alo`
  - `0.005 BTC` max order size
  - `0.01 BTC` max position
  - `1 USDC` max loss
  - `2` max submissions
  - `900s` single window
  - dynamic spread、fill feedback、inventory skew、multi-level 和 actual quote behavior change 全 false
  - watcher/credential/private/account/order/cancel 全未调用。
- private read-only account/service preflight 证明 source/import/python/env 精确，kill-switch clear，open orders `0`，BTC position `0.0`，冲突进程 `0`。
- 第一次 systemd carrier 因 shell path interpolation 为空而在 orchestrator 启动前失败；没有 run status、heartbeat、watcher、credential/private/order/cancel。该事实记录在 `preflight/launch_attempt_01.json`，不计为 live window。
- 改用 systemd native working directory/direct argv 后启动唯一 T024 live window。
- live window 实际结果：
  - source commit：`4c32d991062ad77af9d271cfcb0a8c6c7cf154e1`
  - watcher task：`0718T024`
  - exact runtime caps：`0.005 / 1 / 0.01 / 2`
  - trigger found，event guard pass
  - one real BTC buy post-only order：`0.005 @ 64770`
  - order status reached `resting`
  - tracked cancel called，shutdown proof pass
  - final owned open orders `0`
  - independent private open-orders proof `0`
  - post account BTC position `0.0`
  - estimated loss `0.0 USDC`
  - fill/maker fill/ledger/attribution/liquidity-role rows all `0`
  - child return code `0`、reaped、no SIGKILL
  - status writer healthy，failure count `0`
  - dynamic/fill/multi-level/actual behavior activation all off。
- remote run checksum `61/61` pass；本地 `sha256sum -c` 全部通过。
- preflight/account/post-live 四个 proof 文件 remote/local SHA-256 逐项一致。
- same-window acceptance：
  - provenance/identity `24/24 pass`
  - config/control `27/27 pass`
  - decision replay `10/10 pass`
  - lifecycle/evidence `24/24 pass`
  - economics boundary `6/6 pass`
  - optimism checks `6/6 pass`
  - final recommendation：`principal_task12_mechanism_and_evidence_integrity_passed`

verify：
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_loop.py -q`：`23 passed`。
- orchestrator/watcher/executor/integrated acceptance related regression：`132 passed`。
- T024 acceptance focused regression：`4 passed`。
- `python -m pytest examples/hyperliquid -q`：`461 passed`。
- modified Python `py_compile`、orchestrator/acceptance CLI `--help`、`git diff --check` 通过。
- remote terminal SHA-256 `61/61`，missing `0`，mismatch `0`；local manifest verification 全通过。

done：
- T023 的 exact envelope 和 stale inner identity blocker 已由新 source commit 和新 live artifact 修复。
- T024 获得 accepted-candidate single-level submit/resting/cancel/final-open-orders/account/checksum lifecycle。
- live 作为主要事实来源，same-window offline acceptance 只复现 source/config/decision/control 并保持 live fill/order facts authoritative。

blockers：
- 无 T024 机制/证据完整性 blocker。
- zero fill 继续阻止 stable economics、fill-rate、fee/rebate calibration、queue priority 和 maker viability 结论。
- multi-level 未激活；是否重新评估必须由后续独立 formal task 决定。

commit：
- `4c32d99`
- `d4fdc9d`

提交信息：
- `Repair exact-envelope live evidence rerun`
- `Link T024 acceptance to preflight source`
