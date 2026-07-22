# 0721T047 Business Report

执行线程：
- 总控 auto-loop / 业务执行线程

任务ID：
- 0721T047

状态：
- 待验收

更新时间：
- 2026-07-22 09:08 Asia/Shanghai

是否进行QA验收：
- 是

QA说明：
- 无

files：
- Implementation commit：`1412afead9da637bdeb22ad10d7950ee702c8f7f`
- Live evidence：`local_live_analysis/principal_alignment_dynamic_spread_0721T047/`
- Acceptance repair：`0722T048`

action：
- 在 T046 accepted fixed-quote two-sided manager 上增加独立 bounded dynamic-spread activation path。
- 仅当 estimator candidate `pass`、finite、bounded 且在 `[0.5, 10.0]` hard bounds 内时才允许动态 half-spread；否则固定回退并记录 reason。
- 保持 fill feedback、inventory skew、multi-level 和其它报价变量关闭。
- 增加 exact profile `two-sided-dynamic-manager` 和显式 `--enable-dynamic-spread`。
- 完成一次 exact-envelope retry1 tiny-live；首轮因短 source marker 被 pre-start fail-closed，watcher、private、order、cancel 均未启动/调用，该失败证据保留在 `prestart_short_marker_failure/`。

verify：
- 本地 focused tests：shared kernel `11 passed`；online estimators `24 passed`；orchestrator `37 passed`；watcher `134 passed`；acceptance `244 passed`。
- Full `python -m pytest examples/hyperliquid -q`：`1217 passed in 51.69s`。
- `py_compile` 和 `git diff --check`：通过。
- Live preflight：source `63/63`、account/open-orders/kill-switch/service/conflicting-process gate 全部 pass。
- retry1 live：2026-07-21 23:49:08 UTC 至 2026-07-22 00:03:36 UTC；1 window；child returncode `0`；child reaped；无 SIGKILL。
- Envelope：Alo、Binance public lead / Hyperliquid lag、`1800s`、`0.005 BTC/order`、`0.01 BTC position`、`1 USDC loss`、`2 submissions`、`2` manager attempts、`3s hold`、`10s wait`、fast L2。
- Live facts：提交 `2`、fills `0`、final open orders `0`、BTC position `0.0`、estimated loss `0.0 USDC`。
- Dynamic facts：activation `true`；quote input count `1`；candidate `fallback_fixed`；reason `cold_start_or_invalid_side_intensity_fit`；authoritative quote behavior changed `false`。
- 远端 terminal checksum：`110/110` pass；runtime source `63/63` pass；archive SHA-256：`47a16a42862ad3781562029a940cba054cbac70546c5f39655b98f711a518f4a`。
- Estimator replay：event rows `5074`，confirmed exposure `8`，censor `2`，quarantine `0`，source/replay snapshot match。
- Fill-feedback replay：lifecycle `2`，source/replay snapshot match；activation remains false。
- Independent post-live account proof：source exact，open orders `0`，BTC position `0.0`，service inactive，kill switch clear，credentials/raw account address not written，order/cancel endpoint false。该 proof 保存在 run root 外，不进入 sealed terminal manifest。
- T048 修复 acceptance verifier 后，同窗 acceptance exit `0`：provenance `113/113`、config `76/76`、decision `43/43`、lifecycle `78/78`、economics `6/6`、optimism `6/6`。

done：
- 动态 spread 的 bounded activation、fixed fallback、候选 provenance、执行安全和 live evidence integrity 已有完整证据。
- T047 只支持机制/证据完整性通过，不支持 stable PnL、fill-rate、fee/rebate calibration、queue priority、maker viability、promotion、multi-level 或 final MVP claim。

blockers：
- 独立 QA 验收。

commit：
- `1412afead9da637bdeb22ad10d7950ee702c8f7f`

提交信息：
- `Enable bounded dynamic spread activation`
