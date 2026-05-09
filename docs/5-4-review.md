# 2026-05-04 Live/Backtest Code Review Packet

本文档给第三方 reviewer 使用，范围是当前 Binance tick market-making live/backtest 对齐代码、iter0 baseline、iter1 lifecycle candidate 资料。

本次只整理 review 索引，不修改代码实现。review 时请看当前工作区文件树，而不是只看 `HEAD`；当前 `HEAD` 为 `f6ff495`，相关实现文件处于 dirty/untracked 状态。

## Review Goal

核心问题：

- 当前 live/backtest 对齐逻辑是否能正确复现实盘窗口。
- `audit_replay` cadence 是否实现正确，尤其是 `single` replay mode 的消费、跳过、lag 统计。
- iter1 的 order lifecycle audit 是否足以定位 maker fill/cancel race。
- 对比指标是否只使用 `decision` 行，生命周期诊断行是否没有污染 action/reject/fair/reservation/position 指标。
- 当前仍未完全解决的 API/throttle guard alignment 问题，应由 reviewer 判断是实现问题、模型问题，还是实盘机制差异。

## Primary Code Paths

### Strategy, Live, Backtest

- `examples/binance_tick_mm/strategy_core.py`
  - shared strategy logic.
  - audit row construction.
  - lifecycle event row construction.
  - quote/cancel/submit decision state used by live and backtest.
- `examples/binance_tick_mm/live_tick_mm.py`
  - live bot entry point.
  - Binance connector interaction.
  - REST/local open-order safety checks.
  - live audit CSV writing.
  - lifecycle events from submit/cancel/ORDER_TRADE_UPDATE.
- `examples/binance_tick_mm/backtest_tick_mm.py`
  - backtest entry point.
  - audit replay cadence implementation.
  - `replay_mode = "single"` and `replay_mode = "drain_due"`.
  - initial position injection.
  - latency/API guard simulation.
  - backtest audit CSV and summary JSON outputs.
- `examples/binance_tick_mm/audit_schema.py`
  - shared audit schema for live and backtest.
  - decision, safety, order lifecycle fields.
  - backwards-compatible CSV column contract.

### Alignment, Metrics, Latency

- `examples/binance_tick_mm/compare_audit.py`
  - live vs backtest audit comparison.
  - decision-row filtering.
  - action/reject/planned/throttle match rates.
  - fair/reservation/position/spread/vol MAE.
  - drop-rate metrics.
  - legacy iter0 column compatibility.
- `examples/binance_tick_mm/latency_from_audit.py`
  - observed latency extraction from live audit.
  - order latency NPZ generation.
  - latency stats JSON generation.
- `examples/binance_tick_mm/pipeline_live_raw.py`
  - collector gzip to hftbacktest NPZ conversion.
  - live raw manifest generation.
  - truncated gzip/incomplete trailing line tolerance.
- `examples/binance_tick_mm/pipeline.py`
  - historical Tardis pipeline; useful as reference for normal backtest data flow.
- `examples/binance_tick_mm/align_live_run.py`
  - local orchestration for fetch/convert/latency/backtest/compare/archive.
  - generates normal and audit-replay configs/results.
- `examples/binance_tick_mm/spike_initial_state.py`
  - helper for inspecting/injecting initial state scenarios.

### Config, Deploy, Tests

- `examples/binance_tick_mm/config.example.toml`
  - config template and available knobs.
- `examples/binance_tick_mm/deploy/run_live.sh`
  - live deployment wrapper.
- `examples/binance_tick_mm/deploy/binancefutures.toml`
  - connector config template.
- `examples/binance_tick_mm/test_backtest_tick_mm.py`
  - backtest cadence, replay, initial-state tests.
- `examples/binance_tick_mm/test_compare_audit.py`
  - compare metric and schema compatibility tests.
- `examples/binance_tick_mm/test_latency_from_audit.py`
  - observed latency extraction tests.
- `examples/binance_tick_mm/test_pipeline_live_raw.py`
  - live raw conversion and truncated gzip tests.
- `examples/binance_tick_mm/test_align_live_run.py`
  - orchestration tests.

### Reference Docs

- `docs/binance_tick_mm.md`
  - current live/backtest workflow.
  - audit schema field groups.
  - lifecycle event types.
  - `audit_replay` semantics and result counters.
- `docs/binance_tick_mm_alignment_execution_plan.md`
  - broader alignment plan and historical context.

## iter0 Materials

Use the recomputed iter0 metrics from 2026-05-03 for comparison. The original `BASELINE.md` is still useful as the frozen historical record, but its old metric口径 is not the current comparison baseline.

Root:

- `baselines/iter0/align_lowfill_btcusdt_20260429_143354/`

Key docs:

- `baselines/iter0/align_lowfill_btcusdt_20260429_143354/BASELINE.md`
  - original frozen iter0 record.
  - source run, artifact inventory, old metrics, known gaps.
- `baselines/iter0/align_lowfill_btcusdt_20260429_143354/RECOMPUTED_METRICS_2026-05-03.md`
  - current comparison baseline after updated audit replay and compare logic.
  - reviewer should use this file for iter0 vs iter1 judgments.
- `baselines/iter0/align_lowfill_btcusdt_20260429_143354/live_alignment_summary.md`
  - run-level live/backtest alignment summary.
- `baselines/iter0/align_lowfill_btcusdt_20260429_143354/FILE_MANIFEST.txt`
  - artifact inventory.
- `baselines/iter0/align_lowfill_btcusdt_20260429_143354/SHA256SUMS.txt`
  - per-file checksums.

Key inputs/artifacts:

- `baselines/iter0/align_lowfill_btcusdt_20260429_143354/audit_live_align_lowfill_btcusdt_20260429_143354.csv`
- `baselines/iter0/align_lowfill_btcusdt_20260429_143354/raw_market_data/btcusdt_20260429.gz`
- `baselines/iter0/align_lowfill_btcusdt_20260429_143354/live_order_latency.npz`
- `baselines/iter0/align_lowfill_btcusdt_20260429_143354/live_order_latency_stats.json`
- `baselines/iter0/align_lowfill_btcusdt_20260429_143354/config_live.toml`
- `baselines/iter0/align_lowfill_btcusdt_20260429_143354/config_backtest_normal.toml`
- `baselines/iter0/align_lowfill_btcusdt_20260429_143354/config_backtest_audit_replay.toml`
- `baselines/iter0/align_lowfill_btcusdt_20260429_143354/alignment_report_normal.json`
- `baselines/iter0/align_lowfill_btcusdt_20260429_143354/alignment_report_audit_replay.json`
- `baselines/iter0/align_lowfill_btcusdt_20260429_143354/backtest_normal_result.json`
- `baselines/iter0/align_lowfill_btcusdt_20260429_143354/backtest_audit_replay_result.json`
- `baselines/iter0/align_lowfill_btcusdt_20260429_143354/out/backtest_normal/audit_bt_normal.csv`
- `baselines/iter0/align_lowfill_btcusdt_20260429_143354/out/backtest_audit_replay/audit_bt_audit_replay.csv`
- `baselines/iter0/align_lowfill_btcusdt_20260429_143354/logs/`

iter0 recomputed audit-replay headline metrics:

| Metric | iter0 recomputed |
| --- | ---: |
| Consumed/scheduled | `53,873 / 53,895` |
| Skipped due rows | `0` |
| Max lag breaches | `0` |
| Action match | `0.9729` |
| Reject reason match | `0.7471` |
| Planned action match | `0.7198` |
| Throttle reason match | `0.7561` |
| Fair MAE | `3.6749` |
| Reservation MAE | `4.1013` |
| Position MAE | `0.000890 BTC` |
| BT/live latency drop | `0.1966 / 0.2068` |
| BT/live API drop | `0.1899 / 0.1218` |

Known iter0 context:

- old live audit lacks some later lifecycle/safety diagnostic columns.
- no terminal `open_order_mismatch` in audit rows, but live ended with open-order divergence in the original baseline notes.
- API/throttle alignment was already the main unresolved area after recompute.

## iter1 Materials

Root:

- `local_live_analysis_iter1/iter1_lifecycle_btcusdt_20260503_1800/`

Archive:

- `local_live_analysis_iter1/archive/INDEX.md`
- `local_live_analysis_iter1/archive/iter1_lifecycle_btcusdt_20260503_1800.tar.gz`
- `local_live_analysis_iter1/archive/iter1_lifecycle_btcusdt_20260503_1800.tar.gz.sha256`

Key docs:

- `local_live_analysis_iter1/iter1_lifecycle_btcusdt_20260503_1800/ITER1_ACCEPTANCE.md`
  - lifecycle gate result.
  - H3 status.
  - archive completeness.
  - audit replay notes.
  - lifecycle audit schema notes.
- `local_live_analysis_iter1/iter1_lifecycle_btcusdt_20260503_1800/live_alignment_summary.md`
  - run summary and archive references.
- `local_live_analysis_iter1/iter1_lifecycle_btcusdt_20260503_1800/FILE_MANIFEST.txt`
  - artifact inventory.
- `local_live_analysis_iter1/iter1_lifecycle_btcusdt_20260503_1800/SHA256SUMS.txt`
  - per-file checksums.

Key inputs/artifacts:

- `local_live_analysis_iter1/iter1_lifecycle_btcusdt_20260503_1800/audit_live_iter1_lifecycle_btcusdt_20260503_1800.csv`
- `local_live_analysis_iter1/iter1_lifecycle_btcusdt_20260503_1800/raw_market_data/btcusdt_20260503.gz`
- `local_live_analysis_iter1/iter1_lifecycle_btcusdt_20260503_1800/live_order_latency.npz`
- `local_live_analysis_iter1/iter1_lifecycle_btcusdt_20260503_1800/live_order_latency_stats.json`
- `local_live_analysis_iter1/iter1_lifecycle_btcusdt_20260503_1800/config_live.toml`
- `local_live_analysis_iter1/iter1_lifecycle_btcusdt_20260503_1800/config_backtest_normal.toml`
- `local_live_analysis_iter1/iter1_lifecycle_btcusdt_20260503_1800/config_backtest_audit_replay.toml`
- `local_live_analysis_iter1/iter1_lifecycle_btcusdt_20260503_1800/alignment_report_normal.json`
- `local_live_analysis_iter1/iter1_lifecycle_btcusdt_20260503_1800/alignment_report_audit_replay.json`
- `local_live_analysis_iter1/iter1_lifecycle_btcusdt_20260503_1800/backtest_normal_result.json`
- `local_live_analysis_iter1/iter1_lifecycle_btcusdt_20260503_1800/backtest_audit_replay_result.json`
- `local_live_analysis_iter1/iter1_lifecycle_btcusdt_20260503_1800/out/backtest_normal/audit_bt_normal.csv`
- `local_live_analysis_iter1/iter1_lifecycle_btcusdt_20260503_1800/out/backtest_audit_replay/audit_bt_audit_replay.csv`
- `local_live_analysis_iter1/iter1_lifecycle_btcusdt_20260503_1800/out/backtest_normal/summary_normal.json`
- `local_live_analysis_iter1/iter1_lifecycle_btcusdt_20260503_1800/out/backtest_audit_replay/summary_audit_replay.json`
- `local_live_analysis_iter1/iter1_lifecycle_btcusdt_20260503_1800/logs/`

iter1 audit-replay headline metrics:

| Metric | iter1 |
| --- | ---: |
| Consumed/scheduled | `62,619 / 62,620` |
| Replay mode | `single` |
| Skipped due rows | `0` |
| Action match | `0.9773` |
| Reject reason match | `0.7919` |
| Planned action match | `0.7684` |
| Throttle reason match | `0.7992` |
| Fair MAE | `4.2498` |
| Reservation MAE | `4.5925` |
| Position MAE | `0.000690 BTC` |
| BT/live latency drop | `0.1864 / 0.1946` |
| BT/live API drop | `0.1617 / 0.0817` |

iter1 lifecycle/safety notes:

- live audit rows: `67,715`.
- live window: `2026-05-03T09:00:44Z` to `2026-05-03T10:01:52Z`.
- final bot position and REST position both `0.002 BTC`.
- final REST open orders after stop: `0`.
- terminal `open_order_mismatch`: `0`.
- non-ok safety checks were transient and recovered on the next safety check.
- lifecycle event families are present: `decision`, `safety_check`, `order_submit_sent`, `cancel_sent`, `order_new`, `order_update`, `cancel_ack`, `fill`, `expired`.

## iter0 vs iter1 Current Reading

Use this as context, not as a substitute for reviewing the JSON/CSV evidence.

| Metric | iter0 recomputed | iter1 | Reading |
| --- | ---: | ---: | --- |
| Action match | `0.9729` | `0.9773` | iter1 better |
| Reject reason match | `0.7471` | `0.7919` | iter1 better |
| Planned action match | `0.7198` | `0.7684` | iter1 better |
| Throttle reason match | `0.7561` | `0.7992` | iter1 better |
| Fair MAE | `3.6749` | `4.2498` | iter1 worse, still under gate |
| Reservation MAE | `4.1013` | `4.5925` | iter1 worse, still under gate |
| Position MAE | `0.000890 BTC` | `0.000690 BTC` | iter1 better |
| Latency drop abs diff | `0.0102` | `0.0082` | iter1 slightly better |
| API drop abs diff | `0.0681` | `0.0801` | iter1 worse |

Current conclusion before third-party review:

- iter1 passes lifecycle/P2 and audit replay cadence/fair/latency gate.
- iter1 is not universally better than recomputed iter0 on every metric.
- remaining H3 concern is API/throttle guard alignment.
- fair/reservation MAE is acceptable by gate but not better than recomputed iter0.

## Suggested Review Checklist

1. Verify `audit_schema.py` is complete and stable:
   - decision rows and lifecycle rows share a compatible schema.
   - added lifecycle fields are populated consistently in live and backtest.
   - legacy iter0 audits remain comparable.

2. Verify live lifecycle semantics:
   - submit/cancel/fill/update events are emitted with correct `linked_strategy_seq`, order IDs, side, price, qty, and timestamps.
   - `fill_after_cancel_request` correctly identifies maker fill/cancel races.
   - REST/local/WS open-order seen flags are meaningful and not misleading.

3. Verify `audit_replay` implementation:
   - `single` mode consumes at most one scheduled live decision per feed event.
   - lag/skipped/unconsumed counters are correct.
   - replay does not silently drop decisions except where counters expose it.
   - normal cadence behavior is unaffected.

4. Verify comparison logic:
   - `compare_audit.py` filters to `event_type == "decision"` for primary metrics.
   - lifecycle rows are preserved for diagnosis but excluded from action/reject/fair/reservation/position metrics.
   - drop-rate definitions match the acceptance docs.
   - legacy iter0 missing columns are handled explicitly and do not hide real mismatches.

5. Verify latency/API guard modeling:
   - observed latency extraction matches live audit timestamps.
   - API interval guard and quote throttle semantics match live behavior as closely as possible.
   - current API drop mismatch is explainable from evidence, or clearly flagged as an implementation gap.

6. Verify archive reproducibility:
   - iter0 uses `RECOMPUTED_METRICS_2026-05-03.md` as the current comparison baseline.
   - iter1 archive checksum and per-file checksums pass.
   - `ITER1_ACCEPTANCE.md` and `docs/binance_tick_mm.md` accurately describe replay mode and lifecycle fields.

Suggested local tests for reviewer:

```bash
python -m pytest \
  examples/binance_tick_mm/test_backtest_tick_mm.py \
  examples/binance_tick_mm/test_compare_audit.py \
  examples/binance_tick_mm/test_latency_from_audit.py \
  examples/binance_tick_mm/test_pipeline_live_raw.py \
  examples/binance_tick_mm/test_align_live_run.py
```

Suggested artifact checks:

```bash
cd baselines/iter0/align_lowfill_btcusdt_20260429_143354
sha256sum -c SHA256SUMS.txt

cd /home/molly/project/hftbacktest/local_live_analysis_iter1/iter1_lifecycle_btcusdt_20260503_1800
sha256sum -c SHA256SUMS.txt

cd /home/molly/project/hftbacktest/local_live_analysis_iter1/archive
sha256sum -c iter1_lifecycle_btcusdt_20260503_1800.tar.gz.sha256
```
