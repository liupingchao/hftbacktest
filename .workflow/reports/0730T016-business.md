# 线程回报

执行线程：
- 业务线程-python/cross-exchange-research

任务ID：
- 0730T016

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0730T016.md`
- `.workflow/reports/0730T016-business.md`
- `examples/hyperliquid/cross_exchange_alignment_acceptance.py`
- `examples/hyperliquid/test_cross_exchange_alignment_acceptance.py`
- `local_live_analysis/skhynix_cross_exchange_research_0730T013/alignment/`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 将 R1 builder 升级为 schema v2，默认 task id `0730T016`。
- 生成八个 segment 分区的冻结 decision-label 宽表：
  - 每个 Binance bid/ask price-change decision 一行
  - `10/25/50/100/250/500/1000/2000ms`
  - primary response label
  - diagnostic wall-clock label
  - label source timestamp、effective horizon、BBO age、
    no-new-information、price-changed 和 segment-boundary flags
- Complete-feature warmup 要求 prior timeline 和 prior Hyperliquid BBO
  同时存在；缺任一项即不可进入 eligible denominator。
- R0 provenance 硬门禁覆盖：
  - `56` source SHA/row count
  - `24` normalized output SHA/row count
  - segment manifest SHA/identity
  - aggregate output counts
  - build 前后 source SHA 不变
  - build 前后 `24` 个 normalized R0 input SHA 不变
- Exact mask 硬门禁逐行核对 campaign/profile/segment/boundary/track/
  time/duration/policy/reason；degraded reason 必须完全相等。
- Reconciliation 增加 source age、near/stale、mismatch duration、
  maximum contiguous duration 和 frozen distance/missing-asof gates。
- Acceptance `passes` 现在是 provenance、mask、labels、
  reconciliation、timestamp、future join、coverage 和 source stability
  的合取。
- Warmup decision 仍保留所有 horizon target/primary/wall 标签；只是不进入
  complete-feature denominator。
- 每个 horizon 增加真正的 intrahorizon price-update flag，可区分
  change-then-revert 与 no-update。
- 标签文件在 publication 前执行两次实际 schema/row-count/SHA 重扫。
- Clean rebuild 保持 temporary + backup/rollback publication。

verify：
- Unit/failure-injection: `8 passed`。
- Alignment + R0 + timeline + supervisor regression: `67 passed in 0.41s`。
- `py_compile`, CLI `--help`, `git diff --check`: pass。
- Synthetic counterexamples:
  - duplicate decision 去重
  - incomplete BBO/timeline warmup 排除
  - wrong source/output SHA/row count fail closed
  - wrong exact mask identity fail closed
  - `0%` exact match + bounded asynchronous distance 可通过
  - extreme reconciliation distance 发布 `passes=false`
  - failed clean build 保留旧输出
  - post-build label truncation fail closed
  - build 中 normalized R0 output mutation fail closed
  - forged degraded reason suffix fail closed
  - change-then-revert intrahorizon flag
- Real eight-segment build:
  - status: `passes=true`
  - alignment size: `557M`
  - files: `15`
  - frozen decision-label rows: `2,366,631`
  - complete-feature eligible rows: `2,366,574`
  - warmup excluded: `57`
  - source/output provenance rows: `80`
  - source hashes: `56/56` unchanged
  - normalized R0 output hashes: `24/24` unchanged
  - masks: `8` segment epoch + `2` degraded interval
  - timestamp regression/future join/cross-segment label: `0/0/0`
  - accepted primary horizons: `1000ms`, `2000ms`
  - all `24` reconciliation segment/comparison gates pass
- Independent full semantic scan:
  - checked label rows: `2,366,631`
  - decision sequence/dedupe/time order problems: `0`
  - eligible/as-of/warmup problems: `0`
  - target/primary/wall/effective/no-new/boundary problems: `0`
  - output SHA/row-count problems: `0`
  - warmup rows with all horizon targets: `57/57`
  - intrahorizon true flags: `12,149,677`
  - change-then-revert horizons explicitly represented: `114,616`
  - intrahorizon semantic mismatches: `0`

done：
- T015 QA 和 T016 首轮 QA 的全部 P1/P2 已形成实现和反例保护。
- R1 v2 真实产物已生成并独立复算。
- T016 进入 `待验收`；R2/R3 继续等待 QA。

blockers：
- 无。

boundary：
- 仅使用现有本地数据。
- 未访问 network、AWS、SSH 或交易端点。
- 未新增采集、未拟合信号。
- 新增采集仍需独立任务、用户明确授权和活跃交易时段确认。

commit：
- 无

提交信息：
- 无
