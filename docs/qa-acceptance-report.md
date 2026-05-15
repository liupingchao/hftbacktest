# QA Acceptance Report

# QA 验收结果

执行线程：
- QA验收线程

任务ID：
- 0515T002

状态：
- 已通过

更新时间：
- 2026-05-15 11:41 Asia/Shanghai

验收线程：
- QA验收线程

验收对象：
- 业务线程-python + 0515T002

验收方式：
- 正常验收

验收范围：
- 基于 `0515T001` 的只读 mismatch 诊断结果，制定一个 replay lifecycle repair 的 planning-only 合同，目标是把 cancel-request / cancel-ack / terminal-state / long-horizon persistence 这些候选 repair 方向收敛成最小可实施边界，并明确后续真正的 repair implementation task 该怎么切分。本任务只做设计，不修改 replay、不新跑 replay sweep、不补采样本、不改策略。

验收步骤：
1. 读取 `.workflow/tasks/0515T002.md`。
2. 读取 `.workflow/reports/0515T002-business.md`。
3. 检查任务状态、业务回报、verify 证据、done 结论和阻塞项。
4. 写入 `.workflow/reports/0515T002-qa.md` 并刷新看板。

实际结果：
- 任务文件存在：.workflow/tasks/0515T002.md
- 业务回报状态：待验收
- 业务回报存在：.workflow/reports/0515T002-business.md
- 已形成 replay lifecycle repair design contract。
- 已明确 repair hypotheses、优先级、最小 repair scope、成功/失败标准、验证 artifacts 与后续 implementation 边界。
- 已明确本任务未实现 replay 修复、未运行新 replay、未补采样本、未改策略、未启动 live。

验收结论：
- 已通过
- 结论说明：
  -  计划里的 repair 优先级

  plan only task, target is: 1. P0
- cancel-requested fill eligibility window 太宽
  - terminal-state cutoff semantics 和 live 不一致
2. P1
- long-horizon persistence 过于乐观
  - cancel-ack application timing 落后于后续 lifecycle 更新
3. P2
- remaining qty / partial lifecycle bookkeeping
  - 当前样本 partial_fill=0，所以这不是第一优先级

通过项：
1. 业务回报已进入待验收状态
2. 任务声明需要 QA 验收
3. 业务回报包含 verify 证据
4.  计划里的 repair 优先级

  plan only task, target is: 1. P0
- cancel-requested fill eligibility window 太宽
  - terminal-state cutoff semantics 和 live 不一致
2. P1
- long-horizon persistence 过于乐观
  - cancel-ack application timing 落后于后续 lifecycle 更新
3. P2
- remaining qty / partial lifecycle bookkeeping
  - 当前样本 partial_fill=0，所以这不是第一优先级

不通过项：
1. 无

缺陷清单：
1. 无

阻塞项：
- 无

业务回报阻塞项：
- 无

建议总控下一步：
1. 总控可以将该任务视为验收通过。
2. 如存在后续任务，可按前置条件派发下一任务。

提交信息：
- commit：待提交
# QA 验收结果

执行线程：
- QA验收线程

任务ID：
- 0515T004

状态：
- 已通过

更新时间：
- 2026-05-15 14:55 CST

验收线程：
- QA验收线程

验收对象：
- 业务线程-python + 0515T004

验收范围：
- 验收 `0515T004` 是否保持 read-only residual diagnosis 边界，并且是否基于真实样本对 `0515T003` 后剩余的 2 个 residual matched-submit cases 给出了足够清楚、证据驱动的 case-level trigger classification。

验收步骤：
1. 检查任务单与业务回报，确认本任务没有扩展成 replay repair、策略修改、sample expansion 或 live 判断。
2. 检查 `stage6e_residual_replay_fill_diagnosis_0515T004/` 的产物、manifest 与 residual case csv。
3. 复核 focused tests 与 CLI 证据，确认结论与产物一致。

实际结果：
- 任务保持了 read-only 边界，只修改了 `replay_lifecycle_mismatch_diagnosis.py` 与对应测试，没有修改 replay engine / strategy files。
- 真实样本产物齐全：
  - `RESIDUAL_REPLAY_FILL_DIAGNOSIS_SUMMARY.md`
  - `residual_case_diagnosis.csv`
  - `run_manifest.json`
- residual diagnosis 给出了清楚的非对称结论：
  - `3879|buy` / `572` 明确归类为 `cancel_race_window_too_short`
  - `28940|sell` / `4948` 保守归类为 `residual_replay_fill_trigger_uncertain`
- focused verification 通过：
  - `python -m pytest examples/binance_tick_mm/test_replay_lifecycle_mismatch_diagnosis.py`
  - `python examples/binance_tick_mm/replay_lifecycle_mismatch_diagnosis.py --help`

验收结论：
- 已通过
- 结论说明：
  - `0515T004` 已经足够作为后续 very narrow cancel-race residual repair 的事实基础，但不足以支持对 `4948` 这类 uncertain residual 开 generalized repair。

通过项：
1. 保持了 read-only residual diagnosis 边界，没有越界成 repair。
2. 真实样本上给出了两类 residual 的 case-level timeline 与 trigger classification。
3. 对第二条 residual case 保持了证据不足时的保守判断，没有过度宣称 root cause。

不通过项：
1. 无

缺陷清单：
1. 无

阻塞项：
- 无

建议总控下一步：
1. 将 `0515T005` 作为 narrow implementation task，仅修 `cancel_race_window_too_short` residual。
2. 不要把 `4948` / `residual_replay_fill_trigger_uncertain` 并入同一修复任务。

提交信息：
- commit：无
# QA 验收结果

执行线程：
- QA验收线程

任务ID：
- 0515T005

状态：
- 已通过

更新时间：
- 2026-05-15 23:10 CST

验收线程：
- QA验收线程

验收对象：
- 业务线程-python + 0515T005

验收范围：
- 验收 `0515T005` 是否保持 narrow cancel-race residual repair 边界，是否只修复了 `cancel_race_window_too_short` residual，是否没有把 `4948` uncertain residual 纳入本轮改动，以及 same-sample regression 是否支撑这个结论。

验收步骤：
1. 检查任务单与业务回报，确认本任务边界只限 `cancel_race_window_too_short`。
2. 检查 stage6f calibration / mismatch / residual diagnosis 产物。
3. 复核运行的 focused tests 与回归结果。

实际结果：
- 任务边界符合要求：实现只在 live-confirmed short cancel-race fill 上把 replay `cancel_ack` 窄改写成 replay `fill`，没有扩大到 generalized touch/queue repair。
- same-sample regression 结果符合业务回报：
  - `live_filled_replay_canceled` residual 已消失
  - Stage 6E residual case count `2 -> 1`
  - remaining residual case 只有 `28940|sell` / `4948`
- `4948` 未被误处理：
  - residual diagnosis 仍为 `residual_replay_fill_trigger_uncertain`
  - terminal state diff rows 只剩这 1 条
- aggregate 指标只有极小变化，仍在对齐阈值内：
  - filled gap `0.000397`
  - canceled gap `0.000397`
  - open_or_missing gap `0.0`
- focused verification 通过：
  - `python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py`
  - `python -m pytest examples/binance_tick_mm/test_replay_lifecycle_mismatch_diagnosis.py`

验收结论：
- 已通过
- 结论说明：
  - `0515T005` 按要求完成了 narrow cancel-race residual repair，消除了 `572` 这一类 short cancel-race miss，同时没有把 `4948` uncertain residual 混进本轮修复。

通过项：
1. 任务保持了 very narrow repair scope。
2. `572` residual 被消除。
3. `4948` residual 保持独立，未被误修。
4. Aggregate 指标未发生结构性回退。

不通过项：
1. 无

缺陷清单：
1. 无

阻塞项：
- 无

建议总控下一步：
1. 将 `4948` 作为单独只读诊断任务 `0515T006` 继续分析。
2. 不要把 `0515T005` retroactively broaden 成 generalized replay fill repair。

提交信息：
- commit：无
