# 0706T005 Business Report

执行线程：
- 业务线程-scoped-replay-acceptance

任务ID：
- 0706T005

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0706T005.md`
- `.workflow/reports/0706T005-business.md`
- `local_live_analysis/cross_exchange_mvp_t010_scoped_replay_acceptance_0706T005/scoped_replay_acceptance_manifest.json`
- `local_live_analysis/cross_exchange_mvp_t010_scoped_replay_acceptance_0706T005/supported_fact_replay_comparison.csv`
- `local_live_analysis/cross_exchange_mvp_t010_scoped_replay_acceptance_0706T005/unsupported_fail_closed_matrix.csv`
- `local_live_analysis/cross_exchange_mvp_t010_scoped_replay_acceptance_0706T005/optimism_check_matrix.csv`
- `local_live_analysis/cross_exchange_mvp_t010_scoped_replay_acceptance_0706T005/boundary_manifest.json`
- `local_live_analysis/cross_exchange_mvp_t010_scoped_replay_acceptance_0706T005/validation_report.md`
- `local_live_analysis/cross_exchange_mvp_t010_scoped_replay_acceptance_0706T005/sha256_manifest.csv`

action：
- 新建并执行 `0706T005 / 0625T010-SCOPED Supported-Fact Same-Window Replay Acceptance`。
- 只读取已验收的本地 artifact：
  - `0706T002 / 0625T008` live-submit pulled-back artifact
  - `0706T003 / 0625T009` execution outcome calibration artifact
- 生成 scoped replay acceptance artifact。
- 对齐 supported facts：
  - order intent: `BTC buy 0.01 @ 62146.0`
  - notional: `621.46 USDC`
  - post-only `Alo`
  - order submission attempted
  - real order endpoint reached in source live artifact
  - order status `resting`
  - primary tracked cancel success
  - shutdown proof pass
  - independent final open-orders count `0`
- 检查 unsupported/fail-closed：
  - submit/ack latency
  - resting duration
  - cancel latency
  - cancel-fill race
  - fill horizon
  - fill probability
  - fee/rebate
  - inventory transition
  - realized PnL
  - stable PnL
  - maker viability
- 检查 replay optimism：不得推断 fill probability、fill horizon、fee/rebate、inventory、realized PnL、reject-rate-zero、zero latency、maker viability。

result：
- Final recommendation: `scoped_same_window_replay_acceptance_passed`.
- Supported fact comparison: `12/12 pass`.
- Unsupported fail-closed matrix: `11/11 pass`.
- Optimism checks: `8/8 pass`.
- Boundary status: `pass`.

verify：
- `python -m json.tool` passed for generated manifest and boundary JSON.
- CSV schema/row checks passed:
  - supported fact comparison rows: `12`
  - unsupported fail-closed rows: `11`
  - optimism check rows: `8`
  - sha256 manifest rows: `6`
- All acceptance rows are `pass`.
- `git diff --check` passed.

done：
- Scoped same-window replay acceptance is ready for QA.
- This result does not unlock `0625T011`, full `0625T010`, `0625T012`, another live-submit, repeated-window run, fill-seeking run, stable PnL claim, maker viability claim, promotion, or final MVP pass.

blockers：
- 无 for scoped replay acceptance.
- Full replay/live acceptance remains blocked until a separately authorized task captures complete live lifecycle/economics/PnL evidence.

commit：
- 无

提交信息：
- 无
