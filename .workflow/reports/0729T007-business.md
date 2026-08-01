# 线程回报

执行线程：
- 业务线程-remote-public-data

任务ID：
- 0729T007

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0729T007.md`
- `.workflow/reports/0729T007-business.md`
- `local_live_analysis/cross_exchange_public_sample_skhynix_0729T007_60s_20260729T082454Z/`
- `local_live_analysis/cross_exchange_public_sample_skhynix_0729T007_30m_20260729T082727Z/`

action：
- 将 T006 修复部署到独立 runtime
  `/home/admin/0729T007-hftbacktest`。
- 本地和远端 collector SHA256：
  `3e1335c4a7560ed850d6cfc99c09934f407f886828cd7f4f434f0a4441dcb310`。
- 在 termination-protected c6in winner
  `i-0a962e47210528526 / c6in.xlarge / ap-northeast-1c` 上依次运行：
  - 60 秒 public-only canary
  - canary 通过后的 1800 秒 public-only live validation
- 两个窗口均使用 Binance `SKHYNIXUSDT trade/depth@0ms/bookTicker` 和
  Hyperliquid `xyz:SKHX l2Book/trades fast=true`。
- 两个窗口通过 transient systemd cgroup 托管，raw-only，跳过远端
  alignment。
- 远端生成逐文件 SHA256 清单后拉回本地并独立重算。

verify：
- 60 秒 canary：
  - UTC `2026-07-29 08:24:54` 至 `08:25:55`
  - CST `2026-07-29 16:24:54` 至 `16:25:55`
  - unit `Result=success / ExecMainStatus=0`
  - overlap `60.049549339s`
  - Binance `60.063614086s`，单连接，reconnect `0`
  - 首次 snapshot 过旧，第二次同连接 snapshot 成功 bridge
  - depth `2,111`，trade `5,817`，bookTicker `12,149`
  - `depth_replay_ready=true`，continuity gap `0`
  - Hyperliquid `60.057706861s`，L2 `113`，trades `223`
  - ACK 原文包含 `l2Book / xyz:SKHX / fast=true`
  - Hyperliquid reconnect/disconnect `0`
- 30 分钟窗口：
  - UTC `2026-07-29 08:27:28` 至 `08:57:28`
  - CST `2026-07-29 16:27:28` 至 `16:57:28`
  - unit `Result=success / ExecMainStatus=0`
  - overlap `1799.999199135s`
  - Binance `1800.074023972s`，单连接，reconnect `0`
  - 首次 snapshot `11160480899871` 过旧
  - 第二次 snapshot `11160480916253` 在同一 buffer bridge 到
    `U=11160480914464 / u=11160480916398`
  - depth `58,845`，trade `130,366`，bookTicker `287,858`
  - 独立 raw 检查 continuity gap `0`、invalid depth `0`、
    symbol mismatch `0`
  - `depth_replay_ready=true`
  - Hyperliquid `1800.00748635s`，L2 `3,317`，trades `5,570`
  - 29 个完整分钟 L2 min/p50/p99/max：
    `107 / 111 / 111.72 / 112`
  - L2 gap p50/p99/max：
    `540.474 / 674.715 / 1058.075 ms`
  - invalid L2 `0`、wrong coin `0`、reconnect/disconnect `0`
- 完整性：
  - 两个窗口各有 `16` 个远端校验文件，本地全部匹配
  - 两个 venue raw gzip integrity 通过
  - JSON errors `0`
  - non-monotonic local timestamp `0`
  - manifest counts 与本地 raw 全量计数一致
- 30 分钟 raw：
  - Binance `16,908,119` bytes
  - SHA256 `e78afce215a194d5f1c6618644cff9cf9333991202df9555aec769078f0b2ce4`
  - Hyperliquid `1,134,234` bytes
  - SHA256 `791b6470d319a74b9cb2544836ee59964f4de0b12144b7bf20b4890c56f823b8`

done：
- 目标要求的 60 秒 canary 和 30 分钟双交易所 live validation 均完成。
- T005 live 反例在 T006 修复后被真实关闭：
  - 两个成功窗口都需要一次同连接 snapshot refresh。
  - 两个窗口都在第二次 snapshot 建立 bridge，且没有 WebSocket reconnect。
- 本轮支持 Binance 捕获窗口从 snapshot bridge 开始的 replay-ready 结论。
- 本轮不声称跨断线 exact dual-venue replay、订单成交模拟、PnL 或策略提升。

blockers：
- 无。
- 非阻塞可观测性短板：成功 `run_manifest.json` 未单独持久化两个 child
  return code；当前由 parent `ExecMainStatus=0`、两个成功 manifest 和日志
  共同证明终态。

commit：
- 无

提交信息：
- 无
