# 线程回报

执行线程：
- 业务线程-remote-public-data

任务ID：
- 0729T003

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0729T003.md`
- `.workflow/reports/0729T003-business.md`
- `local_live_analysis/cross_exchange_public_sample_skhynix_0729T003_2h_20260729T022724Z/`

action：
- 在 termination-protected c6in 胜者
  `i-0a962e47210528526 / c6in.xlarge / ap-northeast-1c` 上执行 public-only
  SKHYNIX 双交易所采集。
- 使用 Binance `SKHYNIXUSDT`：
  - `trade`
  - `depth@0ms`
  - `bookTicker`
  - REST depth snapshot limit `100`
- 使用 Hyperliquid `xyz:SKHX`：
  - `l2Book`
  - `trades`
  - `fast=true`
- 先执行 20 秒 canary，再执行冻结的 `7200` 秒正式窗口。
- 正式窗口通过 systemd transient service 运行：
  - start: `2026-07-29 02:27:24 UTC / 10:27:24 CST`
  - end: `2026-07-29 04:27:25 UTC / 12:27:25 CST`
  - service result: `success`
  - exit status: `0`
- 远端完成 gzip、raw SHA、逐文件 SHA 和 tar SHA 检查后，通过 SSM SSH
  tunnel 回拉本地。
- 临时 SSH key 只用于本任务传输，任务完成后移除。

verify：
- Canary:
  - Binance snapshot `HTTP 200`, all three streams non-empty, reconnect `0`.
  - Hyperliquid ACK confirms
    `{"type":"l2Book","coin":"xyz:SKHX","fast":true}`.
  - 20 seconds: Hyperliquid L2 `38`, trades `139`, reconnect `0`.
- Runtime:
  - main and two child PIDs remained unchanged.
  - systemd `Result=success`, `ExecMainStatus=0`.
  - no reconnect or disconnect on either venue.
- Transfer:
  - remote tar SHA256:
    `5f116f9466815322257c6244074de98d7151d12330a49e00146d48e74a0a0cc2`
  - local tar SHA exactly matched.
  - all 14 remote-tracked files passed local SHA256 verification.
  - both raw gzip files passed local integrity checks.
- Full local parse:
  - Binance raw lines: `3,873,345`
  - Binance JSON errors: `0`
  - Binance non-monotonic local timestamps: `0`
  - Binance manifest count reconciliation: pass
  - Hyperliquid raw lines: `55,151`
  - Hyperliquid JSON errors: `0`
  - Hyperliquid non-monotonic local timestamps: `0`
  - Hyperliquid manifest count reconciliation: pass
- Hyperliquid fast L2:
  - L2 rows: `13,265`
  - trades rows: `41,645`
  - full-minute L2 min/p50/p99/max: `108 / 111 / 112 / 112`
  - overall L2 rate: `110.54/min`
  - L2 gap p50/p99/max: `540.06 / 695.51 / 1945.52 ms`
  - invalid L2 shapes: `0`
  - wrong coin rows: `0`
- Binance stream:
  - bookTicker: `2,003,786`
  - depthUpdate: `255,775`
  - trade: `1,613,783`
  - invalid depth shapes: `0`
  - symbol mismatch: `0`
  - continuity gaps from the first captured depth event: `0`
- Cross venue overlap:
  - `7200.037246452` seconds
- `git diff --check -- .workflow/tasks/0729T003.md .workflow/reports/0729T003-business.md`
  passed.

done：
- The requested two-hour public collection completed and the full data package
  is present locally at:
  `local_live_analysis/cross_exchange_public_sample_skhynix_0729T003_2h_20260729T022724Z/`
- Stable two-hour collection is proven for both venues:
  - requested and actual duration met.
  - zero reconnects and disconnects.
  - hashes, gzip integrity, JSON parsing and manifest counts pass.
- Hyperliquid high-frequency L2 is proven:
  - ACK contains `fast=true`.
  - all 120 full minutes contain `108-112` L2 snapshots.
  - every L2 row has two non-empty sides.
- Binance exact startup book replay is not proven:
  - REST snapshot `lastUpdateId=11158119028898`.
  - first captured depth event starts at `U=11158119040568`.
  - startup gap is `11,670` update IDs and about `142.727 ms`.
  - all captured depth events are continuous after the first event, but the
    snapshot cannot bridge into that sequence.
- Therefore this package is valid for stable raw-feed analysis and
  Hyperliquid full-L2 replay, but it must not be claimed as an exact
  Binance+Hyperliquid dual-book replay dataset.
- Required next repair:
  - buffer Binance depth events concurrently while fetching the REST snapshot.
  - enforce `U <= lastUpdateId <= u` bridge acceptance.
  - fail closed and refetch/reconnect when no bridge is present.

blockers：
- Binance startup REST snapshot does not bridge to the first captured
  `depthUpdate`; exact Binance L2 reconstruction from the beginning of this
  sample is blocked.

commit：
- 无

提交信息：
- 无
