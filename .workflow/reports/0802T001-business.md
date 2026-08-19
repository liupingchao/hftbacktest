# 线程回报

执行线程：
- 业务线程-python/public-data-infra

任务ID：
- 0802T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无。

files：
- `.workflow/tasks/0802T001.md`
- `.workflow/reports/0802T001-business.md`
- remote output:
  `/home/admin/0802T001_skhynix_5h_10x30m`

action：
- 检查原定自动化是否执行，确认此前没有启动任何采集。
- 在用户既有明确授权下，直接补启动 5 小时 SKHYNIX public-data
  campaign。
- 使用 c6in winner 上已部署且与本地 SHA 一致的 information-max
  supervisor、collector、registry 和 timeline source。
- 使用 transient systemd 托管：
  `hftbacktest-0802t001-skhynix-5h.service`。
- Campaign:
  `0802T001-skhynix-5h-10x30m`。
- Runtime:
  `18000s / 10 x 1800s`。
- 启动确认后停止检查，不监控后续五小时进度。
- 十个 30 分钟 raw segment 均以 collector return code `0` 完成。
- 全部 raw collection 完成后，supervisor 从 `segment_0001` 开始 strict
  timeline postprocess，并 fail closed。

verify：
- Host:
  `c6in-winner / i-0a962e47210528526 / ip-172-31-6-10`。
- Start request:
  accepted。
- Invocation ID:
  `df969ec751454cc4a74fd93edd059259`。
- Start time:
  `2026-08-02 23:33:10 UTC`。
- Unit:
  `ActiveState=active`, `SubState=running`。
- Main PID:
  `481145`。
- Campaign status:
  `state=running`, `phase=collecting_segment`,
  `current_segment=segment_0001`。
- Collector PID:
  `481148`。
- Binance child PID:
  `481149`, symbol `SKHYNIXUSDT`。
- Hyperliquid child PID:
  `481150`, coin `xyz:SKHX`, `--research-max`。
- Output directory、runtime source、run status、heartbeat 和首个 segment
  目录均已创建。
- Terminal unit:
  `ActiveState=failed`, `Result=exit-code`, `ExecMainStatus=2`。
- All collection segments:
  `10/10` recorded in `collected_segments`。
- Campaign terminal state:
  `failed`, `phase=campaign_aborted`。
- Failure:
  `timeline_quality_failed:segment_0001:skhynix:`
  `hyperliquid_fast_source_age_exceeds_limit`。
- Segment 1 strict collection quality passes:
  duration, venue overlap, market coverage, reconnect policy, head/tail
  freshness and arrival-gap checks are all within their frozen gates.
- Segment 1 common L2 timeline:
  `60,394` rows, `future_join_count=0`, timestamp regressions `0`。
- Segment 1 Hyperliquid fast source age:
  p50 `256.744892ms`, p99 `588.801743ms`, max `2034.006653ms`。
- Frozen fast source-age limit:
  `2000ms`; maximum breach is `34.006653ms`。
- Remote campaign size:
  approximately `202M`; all discovered gzip files pass `gzip -t`。

done：
- Five-hour raw collection completed across all ten segments, but the strict
  campaign did not complete successfully.
- The failed strict campaign is not pulled to local storage and is not used
  to build R0/R1 alignment artifacts.
- 本轮未访问 private/account/order endpoints。

blockers：
- R0 dataset builder requires a campaign that passes strict quality and
  timeline closure. This campaign is ineligible until a separately scoped
  repair defines and validates the fast-L2 age treatment.

commit：
- 无

提交信息：
- 无
