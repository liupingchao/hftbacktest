# 线程回报

执行线程：
- 业务线程-python/public-data-infra

任务ID：
- 0807T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0807T001.md`
- `.workflow/runners/0807T001_transfer_to_amdserver.sh`
- `.workflow/runners/com.hftbacktest.0807T001.transfer.plist`
- c6in runtime：`/home/admin/0807T001-hftbacktest`
- c6in campaign：`/home/admin/0807T001_skhynix_4h_continuous`
- planned AMD destination：
  `/home/molly/project/hftbacktest/local_live_analysis/0807T001_skhynix_4h_continuous`
- local raw campaign：
  `/Users/liu/Documents/hftbacktest/local_live_analysis/0807T001_skhynix_4h_continuous`
- transfer evidence：
  `/home/molly/project/hftbacktest/local_live_analysis/0807T001_skhynix_4h_continuous.transfer.json`

action：
- 冻结当前 collector/supervisor/timeline/registry 源码到独立 c6in
  runtime，逐文件 SHA 与本地源码一致。
- 在 `c6in-winner` 启动连续 `14400s`、collection-only、
  Hyperliquid research-max 的 SKHYNIX 双交易所 public-data 采集。
- 启用 bounded recovered core-L2 reconnect admission：
  单次 `15s`、全段累计 `30s`。
- 启动 one-shot LaunchAgent transfer watcher；成功终态后经 AMD 临时
  目录传输、完整 inventory 对账并原子发布。
- LaunchAgent 后续因 SSM banner timeout 以 `255` 退出，未产生 AMD
  temp/final 目录。
- 恢复流程在 c6in 生成 39 文件 source inventory 和只读 tar，将 tar
  拆成 8 个独立 SHA 分片拉回本机，重组后核对 archive SHA，再经临时
  目录发布本机 raw campaign。
- 本机 raw campaign 通过完整 inventory 后传至 AMD temp，第二次核对
  inventory 完全一致后 rename 为 final。

verify：
- 本机 conda focused tests：`81 passed in 3.62s`。
- 远端 runtime `py_compile` 和 supervisor CLI 参数检查通过。
- 本地与 c6in 五个 runtime 源文件 SHA 完全一致。
- systemd unit：
  `hftbacktest-0807t001-skhynix-4h.service`，
  `active/running`，MainPID `924456`。
- unit 于 `2026-08-07 00:59:50 UTC`
  （`2026-08-07 08:59:50 CST`）进入 active。
- `run_status.json` 为 `running/collecting_segment`，heartbeat 持续更新。
- supervisor、synchronized collector、Binance collector 和 Hyperliquid
  research-max collector 均已启动，raw bytes 持续增长。
- LaunchAgent `com.hftbacktest.0807T001.transfer` 为 `state=running`，
  PID `3750`，stderr 为空并已成功轮询远端 unit。
- AMD final/temp 目标在传输开始前均不存在。
- collection 于 `2026-08-07 04:59:52 UTC` 完成，run state
  `complete/collection_only_complete`，campaign manifest
  `network_collection_complete=true`、`passes=true`。
- 远端 source inventory 共 `39` 文件；本机与 AMD inventory 均逐行
  相同，inventory SHA256 为
  `a30654f32cea375c49e4b26e39ed3c0de436e6d1f08de959601db749f262a003`。
- raw tar 为 `162,979,840` bytes，SHA256 为
  `9528907baebf33c188ba87066383a548c53f5e5aab69179d955ab73801deea91`。
- AMD transfer evidence `passes=true`、`atomic_publish=true`、
  `relative_path_size_sha256_match=true`。

done：
- 4H public collection、三端 raw campaign 归档、完整 inventory
  对账和 AMD 原子发布均已完成。
- 原始 campaign 保持 collection-only，timeline/R0/R1/basis 后处理由
  独立任务 `0807T002` 使用 working copy 执行。

blockers：
- 无。

commit：
- 无

提交信息：
- 无
