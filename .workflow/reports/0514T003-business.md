```md
执行线程：
- 业务线程-python

任务ID：
- 0514T003

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0514T002.md`
- `.workflow/tasks/0514T003.md`
- `.workflow/reports/0514T003-business.md`
- `examples/binance_tick_mm/pricing_research.py`
- `examples/binance_tick_mm/test_pricing_research.py`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`
- `docs/qa-acceptance-report.md`

action：
- 已通过 QA runner 验收 `0514T002` plan，并将 `.workflow/tasks/0514T002.md` 状态更新为 `已通过`。
- 新增并执行 `0514T003` Stage 4 read-only pricing-model research runner implementation。
- 新增 `examples/binance_tick_mm/pricing_research.py`：
  - 读取 accepted sample 下的 live audit decision rows、T009 `joined_decisions.csv`、T009 `top5_sidecar.csv`、sidecar metrics 和 Stage 3 acceptance JSON。
  - 生成 top5/bookTicker/freshness 特征。
  - 计算 `100ms / 500ms / 1s / 5s` future top5 mid markout 和 side-adjusted markout。
  - 输出 candidate metrics、bucket tables、markout-by-horizon、rejected signal list、summary 和 run manifest。
  - manifest 固定 `generated_at_utc=1970-01-01T00:00:00+00:00`，同输入可稳定复跑。
- 新增 `examples/binance_tick_mm/test_pricing_research.py`：
  - 覆盖 top5 snapshot parsing。
  - 覆盖 decision feature filters 和 markout 计算。
  - 覆盖 signal metrics / bucket output。
  - 覆盖 end-to-end artifact writing。
- 在 `5-13-day-control-30min` 上全量运行：
  - output dir：`local_live_analysis/5-13-day-control-30min/stage4_pricing_research_0514T003/`
  - 输出文件：
    - `pricing_research_summary.md`
    - `candidate_signal_metrics.csv`
    - `candidate_signal_metrics.json`
    - `bucket_tables/*.csv`
    - `markout_by_horizon.csv`
    - `rejected_signals.csv`
    - `run_manifest.json`

run result：
- Stage 3 classification used：`passes_pricing_research_market_view`
- audit decision rows：`47499`
- feature rows：`47499`
- accepted-with-stale rows：`47499`
- primary non-stale rows：`47067`
- stale rows excluded from primary：`432`
- join_used_future rows：`0`
- join_missing rows：`0`
- join_gap_crossed rows：`0`
- startup_excluded rows：`0`
- usable top5 snapshot rows：`67318`
- markout rows：
  - `100ms`: `47499`
  - `500ms`: `47497`
  - `1000ms`: `47499`
  - `5000ms`: `47499`
- metric rows：`224`
- signal summary rows：`28`
- candidate_for_followup signals：`14`

top candidate signals on primary non-stale universe：
- `top5_imbalance`: best horizon `500ms`, abs spearman `0.712595`, top-bottom spread `69.3943` ticks.
- `top1_imbalance`: best horizon `500ms`, abs spearman `0.707535`, top-bottom spread `68.8754` ticks.
- `top1_microprice_edge_ticks`: best horizon `500ms`, abs spearman `0.703850`, top-bottom spread `68.7688` ticks.
- `top5_microprice_edge_ticks`: best horizon `500ms`, abs spearman `0.682520`, top-bottom spread `63.0323` ticks.
- `top5_depth_imbalance_qty`: best horizon `500ms`, abs spearman `0.662366`, top-bottom spread `52.7500` ticks.
- `top5_ask_qty`: best horizon `500ms`, abs spearman `0.662147`, top-bottom spread `-62.0165` ticks.
- `top5_bid_qty`: best horizon `500ms`, abs spearman `0.651660`, top-bottom spread `56.6778` ticks.
- `top5_ofi_proxy`: best horizon `500ms`, abs spearman `0.233971`, top-bottom spread `28.2709` ticks.
- `top1_ofi_proxy`: best horizon `100ms`, abs spearman `0.157919`, top-bottom spread `18.1582` ticks.

rejected / downgraded signals：
- `reservation_edge_ticks`: duplicates `fair_edge_ticks` in this sample.
- `audit_mid_edge_ticks`: duplicates `audit_bbo_mid_edge_ticks` in this sample.
- `spread_ticks`: unstable split sign / weak rank correlation / weak top-bottom spread.
- `book_view_stale_ms` and `latency_signal_ms`: duplicates `audit_feed_latency_ms` in this sample.
- `bookticker_mid_edge_ticks`: weak rank correlation under the default candidate threshold.
- `bookticker_depth_age_ms`, `vol_bps`, `bookticker_join_age_ms`, `max_join_age_ms`, `inventory_score`, `spread_bps`, `depth_join_age_ms`, `top5_join_age_ms`: unstable split sign and/or weak rank/top-bottom evidence.

boundary：
- 未修改 strategy behavior、fair/target 公式、配置默认值、risk guards 或 quote placement。
- 未启动 live、未重新采集、未改 AWS/remote state。
- 未修改 core Rust、connector、py `event_dtype` 或标准 npz schema。
- 结果只支持后续 fair/reservation adjustment 设计任务的候选输入，不证明策略 PnL、full L2 equivalence、exact queue position、queue/fill model correctness 或 live readiness。

verify：
- `python -m pytest examples/binance_tick_mm/test_pricing_research.py` -> `4 passed`
- `python examples/binance_tick_mm/pricing_research.py --help` -> passed
- `python examples/binance_tick_mm/pricing_research.py --sample-dir local_live_analysis/5-13-day-control-30min --output-dir local_live_analysis/5-13-day-control-30min/stage4_pricing_research_0514T003` -> passed
- `python3 .workflow/build_dashboard.py` -> passed
- `git diff --check` -> passed

done：
- `0514T002` plan 已由 QA runner 验收通过。
- `0514T003` read-only pricing research runner 已实现并在 `5-13-day-control-30min` 全量运行。
- 研究产物已生成到 `local_live_analysis/5-13-day-control-30min/stage4_pricing_research_0514T003/`。
- 当前任务进入 QA。

blockers：
- 无

commit：
- 065b7e5

提交信息：
- feat(binance): add stage4 pricing research runner
```
