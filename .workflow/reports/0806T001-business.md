# 线程回报

执行线程：
- 业务线程-python/public-data-infra

任务ID：
- 0806T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0806T001.md`
- `.workflow/runners/0806T001_transfer_to_amdserver.sh`
- `.workflow/runners/com.hftbacktest.0806T001.transfer.plist`
- c6in runtime：`/home/admin/0806T001-hftbacktest`
- c6in campaign：`/home/admin/0806T001_skhynix_2h_continuous`
- planned AMD destination：
  `/home/molly/project/hftbacktest/local_live_analysis/0806T001_skhynix_2h_continuous`

action：
- 冻结当前 collector/supervisor/timeline/registry 源码到独立 c6in
  runtime，逐文件 SHA 与本地源码一致。
- 在 `c6in-winner` 启动连续 `7200s`、collection-only、research-max
  SKHYNIX 双交易所 public-data 采集。
- 启用 bounded recovered core-L2 reconnect admission：
  单次 `15s`、全段累计 `30s`，且必须有完整恢复证据。
- 增加一次性 macOS LaunchAgent 托管 transfer watcher。watcher 在
  systemd/campaign 双门禁通过后，经 AMD 临时目录传输、逐文件
  path/size/SHA256 对账，再原子发布最终目录。

verify：
- 本机 conda focused tests：`81 passed in 4.01s`。
- 本地和远端 Python runtime `py_compile` 通过。
- 远端 supervisor CLI 包含 recovered reconnect、continuous collection
  和 collection-only 参数。
- 远端 venv 未安装 `pytest`，因此未在 c6in 重复运行 pytest；采集不
  依赖 pytest。
- systemd unit：
  `hftbacktest-0806t001-skhynix-2h.service`，
  `active/running`，MainPID `818092`。
- unit 于 `2026-08-06 02:15:59 UTC`
  （`2026-08-06 10:15:59 CST`）进入 active。
- `run_status.json` 为 `running/collecting_segment`，heartbeat 持续更新。
- supervisor、synchronized collector、Binance collector 和 Hyperliquid
  research-max collector 均已启动；raw gzip 文件持续增长。
- LaunchAgent `com.hftbacktest.0806T001.transfer` 为 `state=running`，
  watcher PID `8873`，已成功进入远端轮询。
- AMD final/temp 目标在传输开始前均不存在。

done：
- 2H 采集已启动并由远端 systemd 托管。
- 自动传输已由本机 launchd 托管，不依赖当前 Codex/Terminal 会话。
- nominal collection end 为 `2026-08-06 04:15:59 UTC`
  （`2026-08-06 12:15:59 CST`）；结束后仍需完成 manifest 门禁、
  传输和 SHA 对账。
- 采集最终 `campaign_manifest.json` 为 `passes=true`，
  `run_status.json` 为 `complete/collection_only_complete`，且无
  `abort_manifest.json`。
- transient systemd unit 完成后被回收；watcher 已修复为接受
  `LoadState=not-found`，同时以持久化 run status、campaign manifest
  和 abort absence 作为终态事实源。
- 2026-08-07 已完成补传，AMD 发布 `39` 个文件、
  `54,153,592` bytes；两端 relative path/size/SHA256 inventory
  完全一致，inventory SHA256 为
  `a85e314ce9e2cb2f3867289ba9108c19129402c8614c02f9e1be43fba695af52`。

blockers：
- 无。

commit：
- 无

提交信息：
- 无
