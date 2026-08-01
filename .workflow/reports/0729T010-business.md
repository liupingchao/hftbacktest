# 线程回报

执行线程：
- 业务线程-python/public-data-infra

任务ID：
- 0729T010

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0729T010.md`
- `.workflow/reports/0729T010-business.md`
- remote output:
  `/home/admin/0729T010_skhynix_4h_8x30m`

action：
- 已完成 c6in winner 的资源、冲突进程、目标 unit 和 runtime source
  预检。
- 已使用 systemd 启动 `14400s`、`8 x 1800s` 的 SKHYNIX 双交易所
  public-data campaign。
- Unit:
  `hftbacktest-0729t010-skhynix-4h.service`.
- Output:
  `/home/admin/0729T010_skhynix_4h_8x30m`.
- 八个采集 segment 均完成后，supervisor 开始严格后处理。
- `segment_0001` 完成严格质量检查和 common L2 timeline。
- `segment_0002` 因两个 Hyperliquid 辅助研究轨道各发生一次重连而
  fail closed；后续 segment 未展开 timeline。

verify：
- Host: `ip-172-31-6-10`.
- Available disk: approximately `28G`.
- Available memory: approximately `6.9Gi`.
- Passwordless sudo: available.
- Existing collector/supervisor process: none.
- Target systemd unit before start: inactive.
- Runtime source SHA matches accepted T009 collector, supervisor, registry and
  timeline source.
- systemd start request: accepted.
- Immediate unit state: `active`.
- Main PID: `58650`.
- Start time: `2026-07-29 23:54:49 UTC`.
- End time: `2026-07-30 03:59:31 UTC`.
- systemd result: `exit-code`, status `2`.
- Campaign state: `failed`.
- Collected segments: `8/8`.
- Strictly completed segments: `1/8`.
- Failure:
  `hyperliquid_track_reconnect_nonzero:asset_context` and
  `hyperliquid_track_reconnect_nonzero:main_all_mids` in `segment_0002`.
- Core L2 evidence across all eight segments:
  - Binance `depth_replay_ready=true`: `8/8`.
  - Binance reconnects: `0/8`; depth continuity gaps: `0/8`.
  - Hyperliquid fast L2 reconnects: `0/8`.
  - Hyperliquid standard L2 reconnects: `0/8`.
  - Binance depth updates: `527,652`.
  - Hyperliquid fast L2 snapshots: `26,528`.
  - Hyperliquid standard L2 snapshots: `2,690`.
- Auxiliary reconnects:
  - `segment_0002 / asset_context`: `1`.
  - `segment_0002 / main_all_mids`: `1`.
  - all other tracks and segments: `0`.
- Remote artifact checks:
  - package size: approximately `389M`.
  - gzip test: `49/49` passed.
  - raw SHA-256 versus manifest: `48/48` passed.

done：
- 原始采集窗口已经覆盖完整 4H，但 campaign 没有顺利结束。
- 失败来自第 2 段两个辅助研究轨道的重连严格门禁，不是 Binance、
  Hyperliquid fast L2 或 standard L2 的核心采集失败。
- 用户要求仅在成功时拉取，因此本轮没有把远端数据拉回本地。

blockers：
- 必须先决定是否：
  - 维持全轨道零重连标准并重新采集；或
  - 将辅助轨道重连降为带覆盖率/新鲜度约束的 warning，再对现有数据
    重新执行严格后处理和本地拉取。

commit：
- 无

提交信息：
- 无
