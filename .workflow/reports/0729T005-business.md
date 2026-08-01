# 线程回报

执行线程：
- 业务线程-remote-public-data

任务ID：
- 0729T005

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0729T005.md`
- `.workflow/reports/0729T005-business.md`
- `local_live_analysis/cross_exchange_public_sample_skhynix_0729T005_60s_20260729T081312Z/`

action：
- 将 `0729T004` 修复后的 collector 部署到独立远端目录
  `/home/admin/0729T005-hftbacktest`。
- 本地与远端 `synchronized_public_collection.py` SHA256 均为
  `bf30c60be14b8cb6d5839a8c4d791c194fa867d5c2958ce656a213e66628a7ac`。
- 在 termination-protected c6in winner
  `i-0a962e47210528526 / c6in.xlarge / ap-northeast-1c` 上通过 transient
  systemd unit 运行 60 秒 SKHYNIX public-data canary。
- Binance 使用 `SKHYNIXUSDT trade/depth@0ms/bookTicker`；Hyperliquid
  使用 `xyz:SKHX l2Book/trades fast=true`。
- Canary gate 未通过，因此没有启动 30 分钟窗口。
- 失败产物已完整拉回本地。

verify：
- 本地 preflight：focused collector tests `21 passed`。
- AWS：实例 running，API termination protection `True`。
- Remote：clock synchronized，关键源码 SHA 与本地一致，py_compile 与
  collector help 通过。
- systemd terminal result：
  - `Result=exit-code`
  - `ExecMainStatus=2`
- Binance：
  - 实际运行 `2.619629298s`
  - connection attempts `4`
  - reconnects `3`
  - 每次 REST snapshot 均有效，但均落后于第一条已捕获 diff
  - bridge count `0`
  - `depth_replay_ready=false`
  - child return code `1`
- Hyperliquid：
  - 实际运行 `60.063177919s`
  - L2 `112`
  - trades `334`
  - `l2Book`/`trades` ACK 各 `1`
  - reconnect/disconnect `0`
  - child return code `0`
- 本地产物包含两个 venue manifest、raw SHA、日志、runtime source hash
  和失败 run manifest。

done：
- T004 修复源码已被真实部署并执行，不存在远端仍运行旧源码的问题。
- Live canary 暴露新的 Binance bootstrap 缺陷：
  - 当有效 snapshot 的 `lastUpdateId` 小于第一条可用 diff 的 `U` 时，
    当前实现关闭 WebSocket 并重新连接。
  - 四次连接均重复同一模式，最终达到 reconnect 上限。
  - 正确恢复边界应保留当前 WebSocket buffer，并从官方流程的 REST
    snapshot 步骤重新获取 snapshot，而不是立即丢弃 buffer。
- 30 分钟窗口按冻结门禁未启动。

blockers：
- Binance snapshot 过旧时缺少同连接 snapshot refresh，导致
  snapshot bridge 无法建立。

commit：
- 无

提交信息：
- 无
