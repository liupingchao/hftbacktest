# 业务执行回报

执行线程：
- SKHYNIX Flow Internal Directional Alpha A0 Execution 业务线程

任务ID：
- 0828T013

状态：
- 待验收

是否进行QA验收：
- 是

files：
- `.workflow/tasks/0828T013.md`
- `examples/hyperliquid/skhynix_flow_internal_directional_alpha_a0.py`
- `examples/hyperliquid/test_skhynix_flow_internal_directional_alpha_a0.py`
- `local_live_analysis/skhynix_flow_internal_directional_alpha_a0_0828T013/`
- `.workflow/reports/0828T013-business.md`

action：
- 从 commit `91cc0770` 验证四个 frozen authority blobs 和 exact 29-row
  source inventory。
- 对全部 raw 文件执行 size/SHA256 closure。
- 实现 checkpoint-before-same-timestamp-message 的 20ms right-open replay、
  atomic trade/depletion/OFI/activity bins 和 100/500/2000ms bounded ratios。
- 对无法从 snapshot 桥接首个 depthUpdate 的 capture fail closed，不继续
  使用伪静态 book。
- 仅用有效的 Jul 29 calibration segment 冻结
  `Q_activity_60=44 messages/500ms`。
- 实现 active mixed history、dominance candidate、120ms persistence、
  300ms refractory、persistent flip、release dwell 和 total precedence。
- 实现 deterministic dual-direction controls、chronological no-reuse
  matching、30s epoch clusters 和 geometry-only follow-up audit。
- 生成全部 required contracts/support tables、唯一 classification 和
  artifact manifest。

result：
- classification：
  `A0_directional_anchor_date_concentrated`。
- `A1_authorized=false`，`primary_tau=null`。
- detector-ready `29.695294h`，active-flow `12.613944h`。
- raw qualifying checkpoint-direction pairs `798,594`。
- confirmed anchors `3`，全部位于 `2026-08-24`：
  `1 mixed_onset + 2 persistent_flip`。
- directions：`2 down + 1 up`。
- matched pairs `1`，overall common support `0.333333`。
- failed gates：`A0-3, A0-5, A0-6, A0-7`。
- A0-0、A0-1、A0-2、A0-4 通过。
- 7 captures 出现 initial snapshot-depth bridge failure；其中 6 个无有效
  detector-ready segment，1 个在后续 snapshot 后恢复。
- 没有 future midpoint/BBO/markout、fill、fee、PnL 或 H0/H1/H2 fit。

verify：
- focused pytest：`14 passed`。
- `ruff check`：通过。
- Python compile：通过。
- `git diff --check`：通过。
- Markdown fence parity：通过。
- required 45 outputs：全部存在。
- canonical manifest：46 artifacts，size/SHA closure 零错误。
- fresh build B 与 canonical：47 个非-cache 文件逐字节完全一致。
- raw hash verification：29/29 通过。

done：
- A0 已完整执行并冻结 failure classification。
- A1 未获授权。
- 大型 cache 和 `control_candidates.csv` 留在本地；Git 仅跟踪紧凑
  contracts、summaries、support tables 和 exact manifest hashes。

blockers：
- 无执行阻塞。
- 科学路径被 A0 support gates 阻止，禁止用调参 rescue V1。

commit：
- 待提交

提交信息：
- `research: execute flow internal directional alpha A0`
