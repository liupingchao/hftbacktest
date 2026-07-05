```md
执行线程：
- 业务线程-scheduled-aws-amdserver

任务ID：
- 0702T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0702T001.md`
- `.workflow/runners/0702T001_aws_collect.sh`
- `.workflow/runners/0702T001_local_postprocess.sh`
- `.workflow/reports/0702T001-business.md`
- `task_plan.md`
- `progress.md`

action：
- 新建 `0702T001` 正式任务。
- 准备 AWS raw-only 三段顺序采集脚本。
- 准备本地 amdserver copyback、checksum、alignment、join、analysis、pricing signal、sample expansion 后处理脚本。
- 明确北京时间 / 东京时间 / UTC 换算：北京时间 `2026-07-02 19:45:00` 等价于东京时间 `2026-07-02 20:45:00 JST`，等价于 UTC `2026-07-02 11:45:00Z`。
- 已将 AWS 采集脚本复制到 `awsserver1:/home/admin/hft_live/hftbacktest_0627T001/.workflow/runners/0702T001_aws_collect.sh`。
- 已安装并启用 AWS user systemd timer：`0702T001-aws-collect.timer` -> `0702T001-aws-collect.service`。
- 已设置 `admin` 用户 linger：`Linger=yes`。
- 已安装本地 amdserver user systemd transient timer：`0702T001-local-postprocess.timer` -> `0702T001-local-postprocess.service`。

verify：
- 本地 `date`：`2026-07-02 18:19:59 CST +0800`。
- `awsserver1 date`：`2026-07-02 19:20:59 JST +0900`。
- 本地 `git rev-parse --short HEAD`：`b538621`。
- 远端 `/home/admin/hft_live/hftbacktest_0627T001` commit：`bf3bc82`，包含 `0627T001` 已验证的 `--skip-alignment` runner。
- AWS timer verified:
  - `ActiveState=active`
  - `UnitFileState=enabled`
  - `NextElapseUSecRealtime=Thu 2026-07-02 20:45:00 JST`
  - `systemctl --user list-timers` shows `0702T001-aws-collect.timer` triggers at `Thu 2026-07-02 20:45:00 JST`
- Local amdserver timer verified:
  - `0702T001-local-postprocess.timer` is `active (waiting)`
  - trigger is `Thu 2026-07-02 21:25:00 CST`
- `bash -n .workflow/runners/0702T001_aws_collect.sh` passed.
- `bash -n .workflow/runners/0702T001_local_postprocess.sh` passed.
- `git diff --check` passed.

done：
- 任务、脚本、AWS 定时采集 timer、本地后处理 timer 均已准备并验证。
- 当前等待 AWS timer 到点触发第一段采集。
- 运行中更新：
  - AWS timer 已按计划触发，第一段 `xemm_0702_t001_hlfast_bjt1945_a` 于 UTC `2026-07-02T11:45:55Z` / 北京时间 `2026-07-02 19:45:55` / 东京时间 `2026-07-02 20:45:55 JST` 开始。
  - 第一段采集于 UTC `2026-07-02T12:15:55Z` 完成，采集命令返回 `rc=0`。
  - 第一段 evidence：`alignment_status=skipped`，`binance_alignment.status=skipped`，`hyperliquid_alignment.status=skipped`，`alignment_execution_host=macmini_or_amdserver`，overlap `1800.005196229s`，HL `l2Book=3335`，Binance `bookTicker/depthUpdate/trade=734874/67066/115017`。
  - 原 AWS service 在第一段后因额外校验要求旧远端 runner 写出 `raw_collection_only=true` 而失败；第一段采集本身有效，且无 AWS alignment 运行。
  - 已修正 runner 校验逻辑为以 `alignment_status=skipped` 和两个 alignment 子对象 `status=skipped` 为硬证据；若 `raw_collection_only` 字段存在，仍要求其为 `true`。
  - 已启动恢复 service `0702T001-aws-collect-remaining.service`，仅运行剩余两段 `xemm_0702_t001_hlfast_bjt2015_b` 和 `xemm_0702_t001_hlfast_bjt2045_c`。
  - 恢复 service 于 UTC `2026-07-02T12:19:53Z` / 北京时间 `2026-07-02 20:19:53` / 东京时间 `2026-07-02 21:19:53 JST` 开始第二段，当前 `active (running)`。
  - 本地后处理 timer 仍为 `active (waiting)`，触发时间 `Thu 2026-07-02 21:25:00 CST`。
- `2026-07-02 21:40 CST` 状态更新：
  - 第一段、第二段 raw collection 已完成；第三段尚未完成。
  - 第二段 `xemm_0702_t001_hlfast_bjt2015_b` 于 UTC `2026-07-02T12:19:53Z` / 北京时间 `20:19:53` 开始，于 UTC `2026-07-02T12:49:53Z` / 北京时间 `20:49:53` 完成，采集命令返回 `rc=0`。
  - 第二段 evidence：`alignment_status=skipped`，`binance_alignment.status=skipped`，`hyperliquid_alignment.status=skipped`，`alignment_execution_host=macmini_or_amdserver`，overlap 采集日志显示 `1800.030s`，HL `l2Book=3339`，Binance `bookTicker/depthUpdate/trade=1393557/67447/219749`。
  - `0702T001-aws-collect-remaining.service` 在第二段后因 runner 内嵌 Python 缩进错误失败，未启动第三段；第二段采集本身有效，且无 AWS alignment 运行。
  - 原 `0702T001-local-postprocess.service` 于北京时间 `21:25` 触发并拉回第一段，但在脚本环境中退出；尚未开始 alignment。
  - 已修复 AWS runner 缩进，并启动第三段恢复 service `0702T001-aws-collect-c.service`。
  - 第三段 `xemm_0702_t001_hlfast_bjt2045_c` 于东京时间 `2026-07-02 22:42:57 JST` / 北京时间 `2026-07-02 21:42:57` 启动，当前 `active (running)`。
  - 已修复本地后处理脚本 PATH / absolute binary 兼容性，并重新安排 `0702T001-local-postprocess-rerun.timer` 于北京时间 `2026-07-02 22:18:00` 触发。
- `2026-07-02 22:43 CST` alignment / package 结果：
  - 第三段 `xemm_0702_t001_hlfast_bjt2045_c` 已完成，AWS raw-only status log 记录 `status=completed`。
  - 本地后处理已完成 copyback、checksum、Binance alignment、Hyperliquid alignment、join、analysis、pricing signal 和 sample expansion。
  - 本地 package 路径：`local_live_analysis/cross_exchange_mvp_hl_fast_sample_expansion_0702T001/`。
  - Binance alignment metrics 已生成：
    - `xemm_0702_t001_hlfast_bjt1945_a`: `top5_row_count=67066`, `snapshot_alignment_status=missing`, `first_valid_update_aligned=""`
    - `xemm_0702_t001_hlfast_bjt2015_b`: `top5_row_count=67447`, `snapshot_alignment_status=missing`, `first_valid_update_aligned=""`
    - `xemm_0702_t001_hlfast_bjt2045_c`: `top5_row_count=67507`, `snapshot_alignment_status=missing`, `first_valid_update_aligned=""`
  - Hyperliquid alignment metrics 已生成且全部通过 market view:
    - `xemm_0702_t001_hlfast_bjt1945_a`: `topn_row_count=3335`, `decision_row_count=3600`, `trade_event_count=11749`, `sample_classification=passes_pricing_research_market_view`
    - `xemm_0702_t001_hlfast_bjt2015_b`: `topn_row_count=3339`, `decision_row_count=3600`, `trade_event_count=19350`, `sample_classification=passes_pricing_research_market_view`
    - `xemm_0702_t001_hlfast_bjt2045_c`: `topn_row_count=3341`, `decision_row_count=3599`, `trade_event_count=27712`, `sample_classification=passes_pricing_research_market_view`
  - Sample quality matrix: raw checksum 全部 match，reconnect 全部 `0`，future/stale join 为 `0`；第一段有 `missing_join_count=1`。
  - Final recommendation: `sample_collection_invalid`; `t003_creation_unlocked=false`。
  - 直接原因：三个样本的 Binance top5 字段覆盖率为 `0`，缺字段为 `binance_top5_microprice_px`, `binance_bid_top5_px`, `binance_ask_top5_px`, `binance_bid_top5_qtys`, `binance_ask_top5_qtys`, `input_binance_top5_imbalance`, `input_binance_microprice_minus_mid_ticks`, `input_binance_top5_bid_qty`。
  - 因此虽然 nominal `1000ms` near-target label rows 足够（`3590/3598/3492`），但 complete symmetric context rows 为 `0/0/0`，valid for `1000ms` signal acceptance rows 为 `0/0/0`。

blockers：
- 无

commit：
- 无

提交信息：
- 无
```
