# 0617T004 Business Report

执行线程：
- 业务线程-research

任务ID：
- 0617T004

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0617T004.md`
- `.workflow/reports/0617T004-business.md`
- `docs/hyperliquid_tiny_live_signal_quote_policy_protocol.md`
- `local_live_analysis/hyperliquid_tiny_live_signal_quote_policy_protocol_0617T004/protocol_manifest.json`
- `local_live_analysis/hyperliquid_tiny_live_signal_quote_policy_protocol_0617T004/threshold_evidence_matrix.csv`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Defined the Hyperliquid tiny-live signal / quote policy protocol for the Binance-lead / Hyperliquid-lag path.
- Consumed accepted local read-only evidence from `0601T004`, `0601T005`, `0604T003`, and the approved `0617T003` caps snapshot.
- Locked the signal formula to `basis_mid = binance_mid - hyperliquid_mid` and `basis_mid_ticks = basis_mid / hyperliquid_tick_size`.
- Defined persistence, side mapping, maker-only/post-only quote behavior, cancel/stop reasons, and required runtime audit fields.
- Preserved all `0617T003` caps as applying only to separately dispatched `0616T008`.
- Failed closed on live-threshold calibration: accepted artifacts support directional structure, but not a defensible absolute live cutoff, so the protocol recommends threshold calibration before any live execution.

final recommendation：
- `hyperliquid_tiny_live_signal_quote_policy_needs_threshold_calibration`

verify：
- Checked accepted evidence artifacts and local summaries.
- Wrote protocol manifest and evidence matrix.
- Boundary review passed: no private endpoint, no account query, no order placement, no cancellation, no amendment, no live bot.
- `git diff --check` passed.

done：
- Signal formula, side mapping, quote policy, size/cap policy, cancel/stop policy, and required audit fields are defined.
- Threshold source/status is explicitly `blocked_for_live_execution`.
- The next required step is a separate read-only replay / threshold calibration task, not `0616T008`.
- No credentials were read; no private API was called; no account query occurred; no orders were placed/cancelled/amended; no live bot was started.

blockers：
- No defensible absolute live threshold could be derived from accepted artifacts alone.

commit：
- 无

提交信息：
- 无
