```md
执行线程：
- 业务线程-python

任务ID：
- 0514T008

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0514T008.md`
- `.workflow/reports/0514T008-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`

action：
- 读取 `0514T007` task / business / QA、Stage 6B calibration summary 与 gap tables。
- 重新判断下一步优先级：先做 replay fill/cancel lifecycle mismatch diagnosis / repair planning，而不是 sample-first expansion。
- 输出 planning-only 诊断合同，明确：
  - root-cause hypotheses
  - priority-ranked diagnosis questions
  - required future artifacts
  - recommended implementation task boundary
  - why not sample-first
- 未实现 runner，未修改 replay / fill model，未新跑 replay sweep，未补采样本，未改策略，未启动 live。

why not sample-first：
- `0514T007` 已证明 comparison unit 稳定：
  - matched submit coverage `2516/2516`
  - matched price tick equality `2516/2516`
  - matched qty equality `2516/2516`
- 这意味着当前主问题不是 sample coverage 或对齐口径缺失，而是 replay lifecycle generation 本身：
  - replay fills `172` vs live `53`
  - replay fill-after-cancel `133` vs live `16`
  - final state gaps：`canceled` `0.0906`、`filled` `0.0473`、`open_or_missing` `0.0433`
  - cancel-to-fill delay aggregate gap 极大
- 如果不先解释和缩小这些 lifecycle mismatches，多补 current-format samples 的高概率结果只是重复确认 replay 偏差是系统性的，不能推进 quote-adjustment promotion。
- 因此 next task 应先是 mismatch diagnosis / repair planning，再决定是否需要更多 current-format samples 作为扩展验证。

problem framing：
- 当前 Stage 6B 已经证明：
  - submit opportunity matching 不是主 blocker
  - short-horizon fill probability 与 fast-cancel churn 大体可比
  - 真正不可信的是 replay 在 lifecycle 后半段的状态生成：
    - final order state
    - fill-after-cancel-request
    - cancel-to-fill delay
    - long-horizon fill accumulation
    - markout observability coverage
- 所以 diagnosis 重点不是“怎么再找更多样本”，而是“replay 是如何把同一批 matched submit opportunities 推成更高 fill / more cancel-race / different terminal-state path 的”。

priority-ranked diagnosis questions：

1. Priority P0: replay-only fill creation path
   - 对每个 matched submit key，分成：
     - live no-fill / replay fill
     - live canceled / replay filled
     - live open_or_missing / replay filled
   - 问题：
     - replay-only fills 主要集中在哪些 horizons、placement buckets、inventory buckets？
     - replay-only fills 是否主要来自长期挂着最终成交，而不是短时撮合？
   - 目的：
     - 先确认 replay 是否系统性“挂得更久就更容易成交”，还是在 cancel-request window 里“过度留单”。

2. Priority P0: cancel-request / cancel-ack / fill sequencing mismatch
   - 对 replay 和 live 分别重建 timeline：
     - submit_ts
     - cancel_request_ts
     - cancel_ack_ts
     - fill_ts
     - terminal_ts
   - 问题：
     - replay 是否系统性缺失 / 延迟 cancel-ack？
     - replay 是否在 cancel-request 后继续保留 same order 为可成交状态过久？
     - fill-after-cancel-request 放大主要来自：
       - fill logic 过 aggressive
       - cancel-ack 太晚
       - terminal cutoff 规则不同
   - 目的：
     - 解释 `16` vs `133` 的 fill-after-cancel gap 和巨大 cancel-to-fill delay gap。

3. Priority P1: terminal-state transition mismatch
   - 分析 `final_state_gap.csv` 中：
     - `canceled`
     - `filled`
     - `open_or_missing`
   - 问题：
     - replay 为什么少 canceled、多 filled、多 open_or_missing？
     - 是 terminal condition 漏判、remaining qty semantics 问题，还是 lifecycle event ordering 问题？
   - 目的：
     - 判断 diagnosis 后续更像需要修 fill logic、state machine，还是 cancel terminal semantics。

4. Priority P1: markout observability coverage mismatch
   - `coverage_gap.csv` 显示 fill_markout observable rows 差异显著，而 fill_probability observable coverage 一致。
   - 问题：
     - 为什么 matched submit opportunities 一致，但 replay 产生更多可观测 fills / markouts？
     - 这是“more fills”直接导致的，还是 replay future-decision / lifecycle persistence 额外增加了可观测窗口？
   - 目的：
     - 区分 lifecycle mismatch 与 observability mismatch，避免把后者误解为纯 price-path 问题。

5. Priority P1: strata hot-spot localization
   - `0514T007` 已显示差异集中在：
     - deeper `step_back_gt1`
     - `edge_vs_fair q2/q3`
     - higher `inventory_score`
     - larger same-side size buckets
     - some higher latency / join-age buckets
   - 问题：
     - replay mismatch 是全局性的，还是只在某些 placement / inventory / latency path 特别严重？
   - 目的：
     - 为后续实现任务确定最小必要对比表，避免一上来改全局 fill model。

6. Priority P2: sample policy after diagnosis
   - 问题：
     - 在 diagnosis 完成前，补样本能新增什么信息？
     - diagnosis 完成后，什么条件下才值得扩 current-format samples？
   - 结论：
     - sample expansion 只能放在 diagnosis / repair 之后，用于验证修正后是否跨样本稳定，而不是用来替代 root-cause diagnosis。

root-cause hypothesis list：
- H1：replay fill generation 对 resting order 的 long-horizon persistence 过于乐观，导致 matched submit 在 live no-fill 的情况下，replay 更容易 eventually fill。
- H2：replay cancel-request / cancel-ack lifecycle 比 live 慢，或 terminal cutoff 逻辑不同，导致 cancel-requested order 在 replay 中保持“可成交”状态过久。
- H3：replay terminal-state transition（filled / canceled / open_or_missing）语义与 live 不同，尤其在 cancel-requested 或 low-fill orders 上。
- H4：remaining qty / partial lifecycle semantics 处理不一致，虽然当前样本 `partial_fill=0`，但 terminal bookkeeping 仍可能影响 filled vs canceled classification。
- H5：某些 placement / latency / inventory bucket 上，replay 的 effective time priority 或 order persistence 被系统性放大。
- H6：markout observability mismatch 主要是 lifecycle-induced side effect，而不是独立 price sampling bug。

required future artifacts for the implementation task：
- submit-key mismatch tables：
  - `matched_submit_state_diff.csv`
  - `replay_only_fill_cases.csv`
  - `live_cancel_replay_fill_cases.csv`
- lifecycle timeline tables：
  - `cancel_fill_timeline_diff.csv`
  - `terminal_state_transition_diff.csv`
  - `cancel_ack_delay_diff.csv`
- aggregation tables：
  - `state_diff_by_placement.csv`
  - `state_diff_by_inventory.csv`
  - `state_diff_by_latency.csv`
  - `replay_only_fill_by_horizon.csv`
  - `cancel_race_gap_by_bucket.csv`
- summary markdown：
  - `REPLAY_LIFECYCLE_MISMATCH_DIAGNOSIS_SUMMARY.md`

recommended implementation-task boundary：
- next implementation task should be read-only diagnosis, not repair:
  - read existing live audit / audit replay / Stage 6B outputs
  - reconstruct matched submit state-diff tables
  - produce timeline and bucket attribution
  - no replay rerun
  - no strategy changes
- only after that diagnosis passes QA should a separate repair task be created:
  - scope may include replay fill model / lifecycle state machine / cancel-ack handling adjustments
  - repair task must have its own acceptance and must not be folded into the diagnosis task
- sample expansion task should be last:
  - only after diagnosis and any repair task
  - used to confirm whether replay/live lifecycle gap shrinks on more than one current-format sample

decision recommendation：
- immediate next task after `0514T008` should be:
  - a read-only mismatch diagnosis implementation task
- do not create a sample-collection task first
- do not create a quote-adjustment / inventory redesign task first
- do not create a replay repair task before the diagnosis tables exist

verify：
- 人工检查 `.workflow/tasks/0514T008.md` 和 `.workflow/reports/0514T008-business.md` 是否完整。
- `python3 .workflow/build_dashboard.py`
- `git diff --check`

done：
- 已明确为什么当前应先做 replay fill/cancel lifecycle mismatch diagnosis，而不是先补样本。
- 已给出 diagnosis scope、优先级、root-cause hypotheses、required artifacts 和后续实现任务边界。
- 已明确本任务未实现代码、未运行新 replay、未采样、未改策略、未启动 live。

blockers：
- 无

commit：
- 待提交

提交信息：
- 待提交
```
