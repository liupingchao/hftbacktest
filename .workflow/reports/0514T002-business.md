```md
执行线程：
- 业务线程-python

任务ID：
- 0514T002

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0514T002.md`
- `.workflow/reports/0514T002-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`

action：
- 制定 Stage 4 read-only pricing-model research 计划合同。
- 确认 Stage 4 可开始：
  - `0514T001` QA 已通过。
  - `5-13-day-control-30min` 在 Stage 3 中分类为 `passes_pricing_research_market_view`。
  - required gates 已过：`future_join_count=0`、`join_missing_count=0`、`gap_crossed_join_count=0`、`first_valid_update_aligned=true`、`depth_pu_mismatch_count=0`、`decision_join_coverage=1.0`。
  - quality gates 已过：top5 tick match `0.8232`、top5 qty match `0.8014`、stale join rate `0.00909`、BBO mismatch rate `0.000163`。
- 明确 Stage 4 目标：
  - 只做 fair-value / reservation adjustment 的 read-only 研究。
  - 找出候选定价信号是否对未来 mid / side-adjusted markout 有稳定解释力。
  - 产出候选信号排序、拒绝原因和后续是否值得实现的建议。
- 明确非目标：
  - 不改策略，不改 fair/target 公式，不改配置默认值。
  - 不启动 live，不重新采集。
  - 不做 full L2 equivalence、exact queue position 或 queue/fill model calibration。
  - 不声称策略 PnL 或 live promotion readiness。

Stage 4 primary inputs：
- Primary sample：`local_live_analysis/5-13-day-control-30min/`
- Required files：
  - `audit_live_5-13-day-control-30min.csv`
  - `alignment_report_audit_replay.json`
  - `maker_acceptance_stage3.json`
  - `t009_fixed_sidecar/top5_sidecar.csv`
  - `t009_fixed_sidecar/joined_decisions.csv`
  - `t009_fixed_sidecar/metrics.json`
  - `t009_fixed_sidecar/joined_decisions.metrics.json`
- Optional sanity sample：
  - `5-13-day-control-15min` 可用于 compressed BBO/mid sanity，但没有 T009 fixed sidecar 级别的 top5 market-view acceptance，不应作为主 top5 signal 结论源。

Candidate signal plan：
1. Baselines：
   - current mid
   - BBO/bookTicker mid
   - spread and short-horizon volatility buckets
2. Microprice family：
   - top1 weighted mid / microprice
   - top5 microprice using top5 tick/qty sidecar
   - top5 imbalance-adjusted mid
3. Imbalance family：
   - top1 imbalance
   - top5 imbalance
   - top5 depth slope / liquidity concentration proxy
4. OFI proxy family：
   - top1/top5 quantity delta over as-of sidecar rows
   - signed depth size change buckets
   - bookTicker-vs-depth movement buckets
5. Lead-lag / fresh-price family：
   - bookTicker update freshness
   - depth update freshness
   - recent best bid/ask tick move
   - stale-age buckets from Stage 3 join metrics

Evaluation plan：
- Universe:
  - accepted decision rows from `5-13-day-control-30min`
  - exclude `join_used_future=true`, `join_missing=true`, `join_gap_crossed=true`
  - bucket or exclude `join_stale=true`; default report both all accepted rows and non-stale subset
- Markout horizons:
  - `100ms`
  - `500ms`
  - `1s`
  - `5s`
- Metrics:
  - raw future mid change
  - side-adjusted markout
  - correlation / rank correlation
  - monotonic bucket spread by signal quantile
  - top-vs-bottom quantile markout
  - sample count per bucket and stale-age bucket
  - stability across halves / time buckets
- Candidate selection:
  - A signal is a candidate only if it has enough rows, monotonic bucket behavior in the main horizon set, and does not depend on stale/gap/future joins.
  - A signal is rejected if effect is only in stale rows, has unstable sign across halves, has too few rows, or duplicates a simpler baseline.

Output plan for later implementation task：
- Output directory:
  - `local_live_analysis/5-13-day-control-30min/stage4_pricing_research_0514T003/`
- Files:
  - `pricing_research_summary.md`
  - `candidate_signal_metrics.csv`
  - `candidate_signal_metrics.json`
  - `bucket_tables/<signal>.csv`
  - `markout_by_horizon.csv`
  - `rejected_signals.csv`
  - `run_manifest.json`
- Summary must include:
  - sample and artifact hashes / source paths
  - row filters and row counts
  - Stage 3 market-view classification used
  - accepted/rejected signal list
  - next-step recommendation

Acceptance criteria for later implementation task：
- Must be read-only and deterministic.
- Must run only over existing local artifacts.
- Must not modify strategy behavior, configs, live scripts, connector/core, or standard npz schema.
- Must preserve Stage 3 filters and explicitly report stale/gap/future rows.
- Must output enough evidence to decide whether a later design task should implement a fair/reservation adjustment.
- Passing Stage 4 research does not authorize live or strategy implementation by itself.

Recommended next task：
- `0514T003` - Stage 4 pricing-model research runner implementation.
- Thread：业务线程-python.
- Scope：implement read-only research runner and run it on `5-13-day-control-30min`.
- Excluded：strategy implementation, live, core/connector/schema changes, queue/fill calibration.

verify：
- 人工检查 `.workflow/tasks/0514T002.md` 和 `.workflow/reports/0514T002-business.md`。
- `python3 .workflow/build_dashboard.py` -> dashboard refreshed.
- `git diff --check` -> passed before commit.

done：
- Stage 4 plan 已制定。
- Stage 4 可开始，但只能以 read-only research runner 形式开始。
- 推荐后续创建 `0514T003` 实现并运行 read-only pricing-model research。
- 本任务未实现代码、未运行实验、未改策略、未启动 live。

blockers：
- 无

commit：
- 35ae865

提交信息：
- docs(workflow): plan stage4 pricing research
```
