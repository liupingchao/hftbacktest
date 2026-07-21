# 0721T044 Business Report

执行线程：
- 总控 auto-loop / 业务执行线程 / live-awsserver1

任务ID：
- 0721T044

状态：
- 待验收

更新时间：
- 2026-07-21 16:44:10 UTC

是否进行QA验收：
- 是

QA说明：
- Same-window acceptance fail-closed；需要独立QA确认是否为 verifier 的 batch-level freshness projection 绑定缺陷。
- T044 只允许一个 live window；不得在本任务启动第二窗。

source：
- Exact source commit：`260f8964812eb20506eb221333e6aca17da20b4d`。
- Remote source：`/home/admin/hftbacktest-cross-exchange-0721T044`。
- Source archive local/remote SHA-256：
  `b89676d8fffa078090f28a4129728df0d57b9e88964499121dc5ab7fdae1a942`。
- Runtime source start/post 均为 `63/63 pass`；无 missing、unexpected 或 mismatched file。

preflight：
- No-start orchestrator preflight通过，未启动watcher或调用private/order/cancel endpoint。
- 首次account proof因沿用helper内硬编码的`62`个runtime files而fail-closed；本任务exact source实际为`63`个。该失败发生在live前，且未调用order/cancel。
- 修正helper为显式要求`63`后，两次pre-live proof均通过：open orders `0`、BTC position `0.0`、kill-switch clear、account scope一致、无冲突service/process。
- 首次systemd启动因缺少显式working directory，在watcher、private read和order endpoint前失败；无run dir、lock或live process产生。
- 使用相同exact source和显式`--working-directory`修正启动；该pre-start failure不构成第二live window。

live：
- Host：`awsserver1`；Hyperliquid `BTC` lag，Binance public lead。
- 一个且仅一个window：`2026-07-21T16:27:18Z`至`2026-07-21T16:34:45Z`。
- Profile：single-level two-sided manager；post-only `Alo`；`0.005 BTC/order`；最多`2` submissions；hold `3s`。
- Dynamic spread、fill feedback、inventory skew、multi-level和actual quote behavior change均为false。
- Event sequence `1896`授权双边manager：
  - buy：`0.005 BTC @ 66532.0`，exchange response `rejected`，reference-bound terminal proof通过。
  - sell：`0.005 BTC @ 66558.0`，exchange-confirmed resting，随后reference-bound cancel success。
- Submission `2`；fill `0`；maker fill `0`；final owned/account open orders `0`；post BTC position `0.0`；estimated loss `0.0 USDC`。
- Historical fallback未实际调用：direct query attempt `0`，history attempt `0`；未声称T040以外的history机制事实。
- Manager exposure：confirmed interval `1`、confirmed exposure `3`、leading censor `1`、quarantine `0`。
- Hold shutdown v2通过：source close required/closed、stop acknowledged、thread exited、stop wait后无in-flight read、无disconnect/reconnect。
- Child PID `33236` return code `0`、已reap；无termination或SIGKILL；service回到inactive。
- Post-live独立account proof通过：open orders `0`、BTC position `0.0`、kill-switch clear、无冲突process/service。

integrity：
- Remote artifact SHA manifest `108/108 pass`。
- 完整artifact archive remote/local SHA-256：
  `7823d74bb5b3eeaae83400732ffe5fd75bd4a920935b3ae4b175df8374b92d78`。
- Status writer健康；完整run、preflight、window、replay和acceptance evidence已pullback。
- Live evidence保存在
  `local_live_analysis/principal_alignment_estimator_observe_only_0721T044/`，
  本任务未重写历史证据。

replay：
- Estimator replay exit `0`。
- Event rows `3261`；source/replay snapshot SHA-256均为
  `3b70839b2474ea75b25741634a44ba9efe7a9e956173633334130b49dcf34fef`。
- Interval/exposure/censor/quarantine exact match：`1/3/1/0`。
- Buy fit unavailable：`insufficient_observations`；sell fit unavailable：
  `insufficient_distance_variation`。
- Dynamic candidate observe-only；activation false；actual quote behavior unchanged。

same-window acceptance：
- Command使用exact task/source/root和`1800s` expectation，exit `2`并fail-closed。
- Provenance `113/113`、config `72/72`、lifecycle `78/78`、economics `6/6`、optimism `6/6`。
- Decision `39 pass / 4 fail`；四个summary failure来自同一个validation reason：
  `attempt_public_state_freshness_projection_unbound:16:1896`。
- Producer persisted summary的`validation_reasons=[]`。
- Raw event `1896`只有一个pre-submit public-state refresh row，attempt `1`，
  projected seq为`1920/1920`且observed-after-end为true。
- 同一manager batch产生attempt `1`和`2`；两侧attempt均投影完全相同的
  `1920/1920/true`，但acceptance只按`(event_sequence, attempt)`精确join，
  因而把attempt `2`判为unbound。
- Offline acceptance boundary保持
  `offline_only=true`；network/private/order/cancel/remote/new-live/credentials-read均为false。

done：
- 唯一current-source fixed-quote tiny-live window已安全完成，并保留exact source、执行、terminal、exposure、replay和账户终态证据。
- 首个formal stop condition为same-window decision evidence verifier fail-closed；本任务没有启动第二window。
- Live事实不支持stable A/k、fill-rate、profitability、multi-level、promotion或最终MVP pass。

blockers：
- 独立QA acceptance。
- Same-window verifier对two-sided manager batch共享freshness projection的绑定规则需要下一正式离线任务修复；不得修改T044 immutable evidence。

commit：
- Source/dispatch commit：`6142b5e`。
- Business report commit：待提交。

提交信息：
- `Report T044 fixed-quote live evidence`
