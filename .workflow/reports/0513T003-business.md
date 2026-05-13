```md
执行线程：
- 测试线程

任务ID：
- 0513T003

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 正常验收。重点检查 `5-13-day-control-15min` 是否完成拉回、align replay、maker acceptance，以及 T002 MarketView provenance 字段是否在 live / normal replay / audit replay overlay 中按预期标源。本任务不是 live promotion。

files：
- .workflow/tasks/0513T003.md
- .workflow/reports/0513T003-business.md
- task_plan.md
- progress.md
- local_live_analysis/5-13-day-control-15min/
- local_live_analysis/archive/5-13-day-control-15min.tar.gz
- local_live_analysis/archive/5-13-day-control-15min.tar.gz.sha256

action：
- 按用户要求更新 `awsserver1` 并采集 15 分钟 `5-13-day-control-15min`。
- 代码同步改用 git 管理：
  - 本地创建并推送分支 `0513T003-t002-live-validation`。
  - 同步 commit：`192470f Add market view provenance live validation`。
  - 远端执行 `git fetch origin 0513T003-t002-live-validation` 后，用 `git restore --source=192470f -- <files>` 只更新验证所需文件，没有 reset 远端脏工作树。
- 第一次启动在 `5-13-day-control-15min-failed-schema-20260513T0821` 失败并保留目录；原因是远端 `audit_schema.py` 滞后于当前策略代码，CSV writer fieldnames 缺少 0511/0512 已有 adverse/toxic audit 字段。随后通过 git 同步当前 `audit_schema.py` 后重新正式采集。
- 正式 run：
  - start local：`2026-05-13T08:35:45+0900`
  - start UTC：`2026-05-12T23:35:45Z`
  - stop requested local：`2026-05-13T08:50:45+0900`
  - stopped local：`2026-05-13T08:50:58+0900`
  - deployed commit：`192470f`
- 运行 `align_live_run.py` 拉回远端 artifacts、转换 raw gzip、生成 normal replay / audit replay、alignment reports 和 archive。
- 运行 `maker_acceptance.py` 生成 `local_live_analysis/5-13-day-control-15min/maker_acceptance.json`。
- 执行 provenance 只读检查，覆盖 live audit、normal replay audit、audit replay audit。

verify：
- `python examples/binance_tick_mm/align_live_run.py --run-id 5-13-day-control-15min --local-root local_live_analysis --remote-host admin@awsserver1 --remote-root /home/admin/hft_live` -> exit 0。
  - optional warning：旧兼容路径 `/home/admin/hft_live/hftbacktest/examples/binance_tick_mm/audit_live*.csv` 不存在；run-local output audit 已成功拉回。
  - 输出：
    - `local_live_analysis/5-13-day-control-15min/audit_live_5-13-day-control-15min.csv`
    - `local_live_analysis/5-13-day-control-15min/out/live_raw/btcusdt/manifest_2026-05-12_to_2026-05-12.json`
    - `local_live_analysis/5-13-day-control-15min/alignment_report_audit_replay.json`
    - `local_live_analysis/5-13-day-control-15min/live_alignment_summary.md`
    - `local_live_analysis/archive/5-13-day-control-15min.tar.gz`
- `python examples/binance_tick_mm/maker_acceptance.py --alignment-report local_live_analysis/5-13-day-control-15min/alignment_report_audit_replay.json --backtest-result local_live_analysis/5-13-day-control-15min/backtest_audit_replay_result.json --out local_live_analysis/5-13-day-control-15min/maker_acceptance.json` -> exit 0，`passed=true`。
- archive sha256：
  - `4603f525cfca615e9c0f446a35b0de7aa29793569bda5dbf5b7d33e2ff672555  5-13-day-control-15min.tar.gz`

主要结果：
- Live alignment summary：
  - requested run ID：`5-13-day-control-15min`
  - live audit run ID：`5-13-day-control-15min_btcusdt_1778628945`
  - live rows：`33714`
  - decision rows：`28020`
  - audit replay rows：`28019`
  - audit replay consumed/scheduled：`28019 / 28020`
- maker acceptance：
  - `passed=true`
  - hard failures：`[]`
  - common rows：`28019`
  - action match rate：`1.0`
  - planned action match rate：`1.0`
  - reject reason match rate：`1.0`
  - throttle reason match rate：`1.0`
  - working-order semantic/blocking mismatch：`0 / 0`
  - API throttle mismatch / target tick mismatch：`0 / 0`
  - strict replay lag gate passed：`true`
  - strict replay lag breach/drop/fail：`0 / 0 / 0`

T002 provenance 验证：
- live audit：
  - rows：`33714`
  - decision rows：`28020`
  - audit fields：`144`
  - missing provenance fields：`[]`
  - decision `market_view_source`：`live_depth = 28020`
  - decision `top5_source`：`live_depth = 28020`
- normal replay audit：
  - rows：`56676`
  - decision rows：`56667`
  - missing provenance fields：`[]`
  - decision `market_view_source`：`replay_depth = 56667`
  - decision `top5_source`：`replay_depth = 56667`
- audit replay overlay：
  - rows：`32950`
  - decision rows：`28019`
  - missing provenance fields：`[]`
  - decision `market_view_source`：`audit_overlay = 28019`
  - decision `top5_source`：`replay_depth = 28019`
  - decision `market_overlay_source`：`audit = 28019`
  - decision `top5_overlay_source`：empty `28019`

done：
- `5-13-day-control-15min` 已拉回并完成 `align_live_run.py` / `maker_acceptance.py`。
- T002 的核心目标通过真实 live 输出验证：新增 provenance 字段可以区分 live depth、replay depth、audit overlay，并明确 audit overlay 后 top5 仍来自 replay depth，不能伪装成 full L2 对齐证明。
- 这次验证只说明字段链路和 action-path acceptance 正常，不证明策略收益改善，不授权 live promotion。

blockers：
- 无。
- 远端首次失败暴露了部署层 schema 滞后风险；后续 live 更新应优先使用 git 同步，并在启动前检查 `audit_schema.py` 与策略代码兼容。

commit：
- 192470f

提交信息：
- Add market view provenance live validation
```
