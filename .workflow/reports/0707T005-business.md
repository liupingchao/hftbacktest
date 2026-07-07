执行线程：
- 测试线程-local-analysis

任务ID：
- 0707T005

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0707T005.md`
- `.workflow/reports/0707T005-business.md`
- `local_live_analysis/cross_exchange_t010_inline_reprice_handoff_diagnosis_0707T005/`
- `task_plan.md`
- `progress.md`
- `findings.md`

source artifact：
- `local_live_analysis/cross_exchange_t010_repaired_controlled_live_evidence_0707T004_20260707T060126Z/event_driven_edge_gate_live/`

output artifact：
- `local_live_analysis/cross_exchange_t010_inline_reprice_handoff_diagnosis_0707T005/`

analysis summary：
- post-open-orders public-state resync rows: `8`; all passed and observed L2 after `open_orders_end_ns`.
- inline reprice attempt rows: `8`; all failed closed.
- immediate pre-submit guard rows: `8`; all failed closed.
- candidate age at guard: min `4.899`s, median `5.3149999999999995`s, max `5.577`s, versus guard max `1.0`s.
- source event to post-open-orders L2 delta: min `4600.0`ms, median `5025.0`ms, max `5254.0`ms.
- intent present after reprice: `1/8`.
- missing intent after reprice: `7/8`.

root cause classification：
- primary: `post_open_orders_handoff_latency_exceeds_immediate_age_guard`.
- secondary: `inline_reprice_recomputed_candidate_often_no_longer_submit_ready_so_intent_fields_disappear`.
- not primary: post-open-orders public-state freshness failure, live edge source missing, private order endpoint failure, post-only reject/lifecycle failure, or a required threshold/quote/size change.

action：
- Parsed accepted `0707T004` manifest and matrices.
- Compared selected trigger candidate context, post-open-orders public-state resync rows, inline reprice attempt rows, quote attempt rows, immediate pre-submit guard rows, and current candidate audit.
- Generated `handoff_timeline.csv`, `reason_taxonomy.csv`, `handoff_diagnosis_manifest.json`, and `README.md` under the 0707T005 diagnosis artifact.
- Identified that all 8 rows pass the repaired post-open-orders L2 resync gate, then fail the final guard because the candidate is already 4.899s-5.577s old; 7/8 rows also lose submit-ready intent fields after current-candidate recomputation.

verify：
- JSON parse passed for source manifest and generated diagnosis manifest.
- CSV parse passed for source matrices and generated diagnosis CSV files.
- Artifact sanity reload passed.
- `git diff --check` pending final QA commit.

done：
- 0707T005 diagnosis completed.
- Recommendation: create a narrow repair task `T010-INLINE-REPRICE-HANDOFF-CONTRACT-REPAIR` before rerunning controlled live evidence.
- The repair should make trigger candidate audit input vs current reprice candidate output explicit, preserve enough fields to distinguish stale-with-intent from recompute-no-intent, and emit a specific fail-closed latency/handoff reason.
- Do not change anti-drift thresholds, touch-stability thresholds, quote envelope, size, or max submissions in that repair.

blockers：
- Full `0625T010` remains blocked.
- No submitted lifecycle, fill/no-fill economics, fee/rebate, inventory transition, realized PnL, or replay-ready live order evidence was produced by this diagnosis.

commit：
- pending

提交信息：
- pending
