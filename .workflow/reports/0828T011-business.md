# 业务执行回报

执行线程：
- SKHYNIX Safe Reentry After Flow Excursion A0 Execution 业务线程

任务ID：
- 0828T011

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0828T011.md`
- `examples/hyperliquid/skhynix_safe_reentry_after_flow_excursion_a0.py`
- `examples/hyperliquid/test_skhynix_safe_reentry_after_flow_excursion_a0.py`
- `local_live_analysis/skhynix_safe_reentry_after_flow_excursion_a0_0828T011/`
- `.workflow/reports/0828T011-business.md`

action：
- 绑定 0828T010 frozen plan SHA
  `05c082e38bfbc5c7292fa1c36bb8886888e1aecb0430a20d374688de9cae079a`。
- 绑定 predecessor 0828T008 accepted normalization 和 micro-pressure
  contracts。
- 重放 29 个 raw captures，生成 checkpoint
  `(local_receive_ts_ns,event_seq_in_file)` sidecar；逐 capture 核对
  checkpoint timestamp 与 segment identity。
- 实现 1000ms bilateral novelty、250ms qualification window 内完整
  100ms causal persistence、single-episode direction switching、
  bilateral 80% depth recovery 和 1000ms refractory。
- 使用 `[candidate-1000ms,candidate-100ms)` componentwise median 作为
  causal depth/context baseline。
- 仅在 refractory completion 当前时点检查
  `spread>=2 ticks`、`abs(OBI)<=0.50` 和 bilateral depth。
- 构造 250ms stride outcome-blind controls，排除最近 5000ms confirmed
  excursion/safe anchor，执行无复用 current-state matching。
- 生成全部 frozen contracts、support ledgers、summary、classification
  和 run manifest。

canonical result：
- classification：`A0_normalization_support_failed`
- status：`failed`
- `A1_authorized=false`
- source：29 captures、9 dates、35.9172008142 hours
- raw micro crossings：490,307
- candidates：253
  - confirmed：42
  - pre-confirmation direction switch rejected：82
  - transient rejected：129
- confirmed excursion rate：1.1693561594/hour
- represented excursion dates：6/9
- crossing/excursion compression：11,673.9762
- median inter-excursion interval：464,140ms
- episode duration：
  - p50：172,060ms
  - p90：962,582ms
  - maximum：4,839,820ms
- direction switches：
  - p50：410.5
  - p90：3,170.6
  - maximum：13,859
- recovery reset p50：510.5
- refractory reset p50：61.5
- terminal composition：
  - `recovered_without_wide_spread`：34
  - `never_recovered`：5
  - `recovered_but_imbalanced`：2
  - `safe_reentry_available`：1
- safe-reentry anchors：1
- safe-reentry rate：0.0278418133/hour
- controls：30,286
- matched pairs：1/1；该数不能通过 minimum 200-pair gate。
- excursion non-floor `>=2` share：0.1190476190
- safe-reentry non-floor `>=2` share：0.0
- selected geometry-only follow-up horizon：10,000ms

gates：
- `A0_0_source_closure=true`
- `A0_1_zero_outcome_boundary=true`
- `A0_2_normalization_support=false`
- `A0_3_excursion_support=false`
- `A0_4_novelty_persistence_compression=true`
- `A0_5_safe_reentry_support=false`
- `A0_6_control_common_support=false`
- `A0_7_followup_geometry=true`

interpretation：
- novelty、persistence 和 refractory 成功把 near-continuous pressure
  crossings 压缩为非重叠 episode，且没有 backdating、overlap 或
  persistence violation。
- 压缩后的对象不是稳定、可重复的短 flow excursion，而更接近持续数十秒
  到数十分钟、内部反复翻向的 long-lived flow regime。
- 42 个 excursions 只分布于 6 个日期，且 27 个集中在 2026-08-26；
  历史支持不足且日期集中。
- 绝大部分 episode 在完整恢复和 refractory 完成时 spread 已收窄；
  34/42 为 `recovered_without_wide_spread`。
- 唯一 safe-reentry anchor 仍依赖 directional normalization floor，
  因此不能作为稳定 landmark。
- V1 不得进入 A1 target materialization。不能通过缩短 novelty、
  persistence、refractory 或等待 future wide spread 来救结果。

verify：
- raw size/SHA closure：通过。
- 29/29 replay checkpoint timestamp/segment closure：通过。
- upstream cache size/SHA inventory：通过。
- focused regression：`24 passed in 1.51s`。
- Python compile：通过。
- ruff：通过。
- canonical/build-B 非 cache inventory：41/41 文件逐字节一致。
- run manifest：40 artifacts，size/SHA closure 通过。
- future midpoint/best price/contact fields：`[]`。
- queue-fill targets materialized：`false`。
- H0/H1 fitted：`false`。
- `git diff --check`：通过。

done：
- SAFE_REENTRY_AFTER_FLOW_EXCURSION_V1 A0 已完整执行。
- 研究 hypothesis 在当前 frozen tuple 和数据集上未通过。
- A1 不获授权。

blockers：
- 研究链路阻塞于 A0；不是实现阻塞。

commit：
- 待提交

提交信息：
- research: execute safe reentry excursion A0
