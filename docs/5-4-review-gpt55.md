# 2026-05-04 Live/Backtest Alignment Review - GPT-5.5

## Review Scope

本次阅读了 `docs/5-4-review.md`、`docs/5-4-plan.md`，并按文档索引检查了当前工作区的 dirty/untracked 实现与归档材料，重点覆盖：

- `examples/binance_tick_mm/audit_schema.py`
- `examples/binance_tick_mm/strategy_core.py`
- `examples/binance_tick_mm/live_tick_mm.py`
- `examples/binance_tick_mm/backtest_tick_mm.py`
- `examples/binance_tick_mm/compare_audit.py`
- `examples/binance_tick_mm/latency_from_audit.py`
- `examples/binance_tick_mm/pipeline_live_raw.py`
- `examples/binance_tick_mm/pipeline.py`
- `examples/binance_tick_mm/align_live_run.py`
- `examples/binance_tick_mm/spike_initial_state.py`
- 文档中列出的 config/deploy/test 文件
- iter0 归档：`baselines/iter0/align_lowfill_btcusdt_20260429_143354/`
- iter1 归档：`local_live_analysis_iter1/iter1_lifecycle_btcusdt_20260503_1800/`

## Executive Verdict

当前实现已经解决了几个关键基础问题：shared audit schema 是兼容追加式设计，`compare_audit.py` 的主指标确实过滤到 `decision` 行，`latency_from_audit.py` 也过滤 lifecycle rows，iter1 归档校验通过，建议测试集全部通过。

但现在还不适合冻结 backtest research contract 或进入大规模 maker 参数优化。主要原因不是单一的交易所 residual risk，而是仍存在几个可定位的实现/口径问题：

- `audit_replay single` 的 consumed ratio 很好，但 lag gate 实际被 `max_lag_ms = 0.0` 关闭；iter1 的 replay lag 分布很大，当前验收容易把“用旧 live timestamp 标记较晚 backtest feed state”的情况隐藏掉。
- live 和 backtest 的 quote throttle state 更新语义不一致，尤其在 `two_phase_replace_enabled = true` 的 cancel-only 阶段，会直接影响 API/throttle alignment。
- `fill_after_cancel_request` 目前大量 false positive，不能直接作为 maker fill/cancel race 结论。
- live 会在发单后用 observed order latency 反标 `latency_guard`，backtest 不做同一件事，且这和“前瞻 guard 只用 feed latency”的设计不一致。
- iter0 recomputed metrics 没有对应的校验归档 JSON/CSV，baseline 目录内的 checksum 通过的是旧 audit-replay 产物。

## Findings

### 1. Audit replay lag gate is effectively disabled

Severity: High

`backtest_tick_mm.py` 的 `_audit_replay_decision_due()` 只有在 `max_lag_ns > 0` 时才统计 breach：

- `examples/binance_tick_mm/backtest_tick_mm.py:318`
- `examples/binance_tick_mm/backtest_tick_mm.py:335`
- `examples/binance_tick_mm/backtest_tick_mm.py:348`

而 `align_live_run.py` 生成 iter1 audit replay config 时写入：

- `examples/binance_tick_mm/align_live_run.py:399` `tolerance_ms = 0.0`
- `examples/binance_tick_mm/align_live_run.py:400` `replay_mode = "single"`
- `examples/binance_tick_mm/align_live_run.py:401` `max_lag_ms = 0.0`

因此 `audit_replay_max_lag_breaches = 0` 不是一个有效 gate，只代表没有启用阈值。iter1 `backtest_audit_replay_result.json` 的实际 lag 分布是：

- consumed/scheduled: `62619 / 62620`
- skipped due rows: `0`
- lag mean: `1.9046s`
- lag p50: `636ms`
- lag p90: `6.033s`
- lag p99: `11.477s`
- lag max: `18.102s`

这会影响对“同一实盘窗口复现”的判断。当前 backtest 在 later feed event 上消费 live schedule，但把 audit row 的 `ts_local` 写成 consumed live timestamp；`compare_audit.py` 再按 `strategy_seq` 或 row timestamp 对齐时，看起来 cadence 很接近，但市场状态其实可能来自晚几秒的 backtest feed。

建议：

- 对 audit replay 设置非零 `max_lag_ms`，例如先用 `250ms` 或 `1000ms` 做硬报告。
- 在 audit row/result 中同时记录 `decision_ts` 和实际 `feed_ts_local`，避免 compare 只看到重写后的 live schedule timestamp。
- H3+ gate 不应只看 consumed ratio，还应看 lag p99/max 或 breach count。

### 2. Live/backtest quote throttle state updates differ

Severity: High

live 每次实际发送 API action 后都会更新 quote throttle state：

- `examples/binance_tick_mm/live_tick_mm.py:492` cancel action
- `examples/binance_tick_mm/live_tick_mm.py:495` submit buy
- `examples/binance_tick_mm/live_tick_mm.py:497` submit sell
- `examples/binance_tick_mm/live_tick_mm.py:503` `throttle_state.mark_sent(...)`

backtest 对应路径则调用 `update_quote_throttle_state()`：

- `examples/binance_tick_mm/backtest_tick_mm.py:896`
- `examples/binance_tick_mm/strategy_core.py:805`
- `examples/binance_tick_mm/strategy_core.py:813` 只在 `any(action.kind == "submit")` 时 mark

这意味着 cancel-only API action 在 live 中会刷新 quote throttle baseline，在 backtest 中不会。iter1 config 开启了 `two_phase_replace_enabled = true`，替换报价时经常先 cancel、下一轮再 submit；这个差异会直接制造 quote throttle/API interval mismatch。

iter1 audit-replay top mismatch 也符合这个方向：

- BT `api_interval_guard` vs live empty: `6380`
- BT empty vs live `quote_throttle`: `2707`
- BT `quote_throttle` vs live empty: `1983`
- BT empty vs live `api_interval_guard`: `703`

建议：

- live 和 backtest 必须使用同一个 helper 更新 `QuoteThrottleState`。
- 明确定义 quote throttle 是“submit quote throttle”还是“any quote API action throttle”。如果 cancel 也算 quote update，两边都 mark；如果 cancel 不算，live 也不要 mark。
- 增加 cancel-only + two-phase replace 的单测，覆盖后续 submit 是否被同样 throttle。

### 3. `fill_after_cancel_request` currently over-flags races

Severity: High

live/backtest 当前只要 order 曾经 `mark_cancel_requested()`，后续观察到 fill 就标记 `fill_after_cancel_request`：

- `examples/binance_tick_mm/live_tick_mm.py:663`
- `examples/binance_tick_mm/backtest_tick_mm.py:1042`

这个逻辑没有校验 `fill_ts >= cancel_request_ts`。iter1 live CSV 中：

- live fill rows: `172`
- live `fill_after_cancel_request = 1`: `27`
- 其中 `24` 个的 `fill_ts < cancel_request_ts`

样例：

```text
strategy_seq=1723 order_id=143 fill_ts=1777798925312000000
cancel_request_ts=1777798925312851487 lifecycle_detail=fill_after_cancel_request
```

这更像是“本地请求 cancel 前，交易所已经成交，但本地稍后才观察到 fill”，而不是严格意义上的 fill-after-cancel race。backtest 也有同类情况，但少得多：`32` 个标记里 `3` 个 `fill_ts < cancel_request_ts`。

建议：

- 把当前字段改成严格语义：只有 `fill_ts >= cancel_request_ts > 0` 才标 `fill_after_cancel_request`。
- 如需保留“本地观察在 cancel request 之后”的诊断，另加字段，例如 `fill_observed_after_cancel_request`。
- 增加测试覆盖 `fill_ts < cancel_request_ts` 不应标记严格 race。

### 4. Live post-send observed latency contaminates `latency_guard`

Severity: Medium-High

live loop 先用 `feed_latency_ns` 做前瞻 latency guard，但发单后又检查 observed order entry latency：

- `examples/binance_tick_mm/live_tick_mm.py:291`
- `examples/binance_tick_mm/live_tick_mm.py:296`
- `examples/binance_tick_mm/live_tick_mm.py:541`

如果已经 `sent_api`，且 `entry_latency_ns > latency_guard_ns`，live 会把该 decision 标成 `dropped_by_latency = 1`，并在没有 reject reason 时写 `latency_guard`。backtest 对应路径不做这一步：

- `examples/binance_tick_mm/backtest_tick_mm.py:772`
- `examples/binance_tick_mm/backtest_tick_mm.py:774`
- `examples/binance_tick_mm/backtest_tick_mm.py:914`

这会让 live 的已执行 action 同时看起来像 latency drop，污染 `drop_latency_rate` 和 reject reason alignment。`test_backtest_tick_mm.py` 已经确认 backtest latency guard signal 只用 feed latency，不用 predicted entry latency；因此 live 的 post-send 标记至少应该独立命名，不能混入 guard drop。

文档也有过期描述：`docs/binance_tick_mm.md:174` 仍写着 backtest latency guard 使用 `max(feed_latency_ns, predicted_entry_ns)`，但当前代码和测试都已经是 feed-only。

建议：

- 删除 live post-send 对 `dropped_by_latency/reject_reason` 的反标，或改为单独字段如 `observed_entry_latency_over_guard`。
- 更新 `docs/binance_tick_mm.md` 的 latency guard 描述。
- 增加 live/backtest guard parity 测试，至少覆盖“已发单后 observed latency 超阈值不应改变 reject_reason”。

### 5. Iter0 recomputed metrics are not backed by archived JSON/CSV in the baseline directory

Severity: Medium-High

`docs/5-4-review.md` 明确要求使用 `RECOMPUTED_METRICS_2026-05-03.md` 作为 iter0 current baseline。该文档中的 audit replay 指标是：

- rows: `53873`
- consumed/scheduled: `53873 / 53895`
- skipped due rows: `0`
- action match: `0.9729`
- API drop: `0.1899 / 0.1218`

但 `baselines/iter0/align_lowfill_btcusdt_20260429_143354/` 中 checksum 覆盖的 `backtest_audit_replay_result.json` 和 `alignment_report_audit_replay.json` 仍是旧产物：

- rows: `11177`
- consumed/scheduled: `53873 / 53895`
- skipped due rows: `42696`
- action match: `0.9111`
- BT/live API drop: `0.00456 / 0.12179`

`SHA256SUMS.txt` 和 `FILE_MANIFEST.txt` 也没有包含 `RECOMPUTED_METRICS_2026-05-03.md`。因此 iter0 的“当前比较口径”现在依赖一个未校验的 markdown 摘要，而不是可复现的 archived JSON/CSV。

建议：

- 把 recompute 后的 `backtest_audit_replay_result.json`、`alignment_report_audit_replay.json`、`audit_bt_audit_replay.csv` 一起归档或放入单独 `recomputed_2026-05-03/` 目录。
- 将 recomputed docs 和产物加入 `FILE_MANIFEST.txt` / `SHA256SUMS.txt`。
- `docs/5-4-review.md` 中列出的 iter0 JSON 若继续是旧口径，应明确标注“historical old metrics，不用于 current comparison”。

### 6. Audit replay schedule loader does not filter `decision` rows

Severity: Medium

`compare_audit.py` 对主指标有明确 decision filter：

- `examples/binance_tick_mm/compare_audit.py:19`
- `examples/binance_tick_mm/compare_audit.py:87`

但 audit replay schedule 加载只按 `run_id` 和 timestamp 列读取，并用 set 去重：

- `examples/binance_tick_mm/backtest_tick_mm.py:294`
- `examples/binance_tick_mm/backtest_tick_mm.py:299`
- `examples/binance_tick_mm/backtest_tick_mm.py:311`

iter1 当前没有被这个问题击穿：live CSV 有 `67715` 行，其中 `62620` 条 `decision`，schedule 也是 `62620`，因为 lifecycle/safety 行大多共享 decision timestamp。但这个实现依赖隐含不变量：所有 lifecycle/diagnostic rows 都不会引入额外唯一 `ts_local`，且 decision rows 不会有重复 `ts_local`。

建议：

- `_load_audit_cadence_schedule()` 应默认只读取 `event_type in {"", "0", "decision"}`。
- 如需要兼容旧 CSV，可在缺失 `event_type` 时视为 decision。
- 保留 dedupe 也可以，但 result 中应报告 raw decision rows、unique schedule rows、deduped count。

## Positive Checks

### Schema and compare

- `AUDIT_FIELDS` 将 lifecycle 字段追加在主决策字段之后，`REQUIRED_ALIGNMENT_FIELDS` 不要求 lifecycle 字段，legacy iter0 CSV 可以继续比较。
- `compare_audit.py` 的 `_decision_rows()` 正确排除 lifecycle rows；测试 `test_compare_ignores_lifecycle_rows_for_alignment_and_summary` 覆盖了这一点。
- iter1 live audit 抽查：`67715` total rows，`62620` decision rows，event families 包含 `safety_check`, `order_submit_sent`, `decision`, `order_new`, `order_update`, `cancel_sent`, `cancel_ack`, `fill`, `expired`。

### Latency extraction and raw pipeline

- `latency_from_audit.py` 只从 decision rows 提取 `req_ts/exch_ts/resp_ts`，不会把 lifecycle rows 混入 latency NPZ。
- `pipeline_live_raw.py` 对 truncated gzip 的容错路径有测试覆盖，manifest 结构和 backtest 入口兼容。

### Lifecycle evidence

iter1 lifecycle row families 足够支持基本 submit/cancel/fill/update/expired 追踪，也能定位“某个 order 是否在 cancel request 后才被本地观察为 fill”。但在修正 `fill_after_cancel_request` 严格语义之前，它还不足以直接证明 maker cancel/fill race 数量。

### Safety/archive

- iter1 per-file checksum: pass。
- iter1 archive checksum: pass。
- iter0 archived historical checksum: pass。
- iter1 final bot position 与 REST position 一致，final REST open orders 为 0，terminal `open_order_mismatch` 为 0。

## API/Throttle Alignment Reading

当前 API/throttle mismatch 不应完全归因于 Binance 实盘机制差异。至少有两类可修实现问题：

- quote throttle state live/backtest 更新规则不同。
- live 的 post-send observed latency 标记会改变 reject/drop 口径。

此外，audit replay lag gate 关闭会让部分 guard mismatch 的时间归因变得不可靠：decision row timestamp 与实际使用的 backtest feed state 可能相差几秒。

因此下一步应先做 API/throttle mismatch breakdown，但 breakdown 前建议先修正或隔离上述实现差异，否则 top mismatch cases 会混合真实交易所机制、模型缺口和 instrumentation 口径问题。

## PnL Attribution Gap

`docs/5-4-plan.md` 中提出的 PnL attribution 仍是必要前置条件。当前代码可以比较 action/reject/fair/reservation/position/drop rate，但还不能解释 live/backtest PnL 差异来自：

- spread capture
- fees/rebate
- inventory mark-to-market
- adverse selection after fill
- latency/API missed opportunity
- cancel/fill race impact
- unfilled quote opportunity

在上述 guard/lifecycle 口径修正前，不建议把当前 backtest PnL 或参数扫描结果作为可迁移 maker 策略证据。

## Verification Run

本次运行的测试：

```bash
python -m pytest \
  examples/binance_tick_mm/test_backtest_tick_mm.py \
  examples/binance_tick_mm/test_compare_audit.py \
  examples/binance_tick_mm/test_latency_from_audit.py \
  examples/binance_tick_mm/test_pipeline_live_raw.py \
  examples/binance_tick_mm/test_align_live_run.py
```

结果：`78 passed in 1.12s`。

本次运行的 checksum：

- `local_live_analysis_iter1/iter1_lifecycle_btcusdt_20260503_1800/SHA256SUMS.txt`: pass
- `local_live_analysis_iter1/archive/iter1_lifecycle_btcusdt_20260503_1800.tar.gz.sha256`: pass
- `baselines/iter0/align_lowfill_btcusdt_20260429_143354/SHA256SUMS.txt`: pass for archived historical files

注意：iter0 checksum pass 不代表 recomputed metrics JSON/CSV 已归档；见 finding 5。

## Recommended Next Steps

1. 先修 audit replay lag reporting：设置非零 `max_lag_ms`，记录实际 feed timestamp，并把 lag p99/max 纳入 gate。
2. 统一 live/backtest quote throttle state 更新语义，并加 cancel-only/two-phase replace 测试。
3. 修正 `fill_after_cancel_request` 严格语义，另设字段保留本地观察顺序诊断。
4. 移除或重命名 live post-send observed latency 对 `latency_guard` 的反标。
5. 重新归档 iter0 recomputed 产物，避免 baseline markdown 与 checksum JSON 不一致。
6. 在上述修正后再做 API/throttle mismatch breakdown 和 PnL attribution。
7. 等 API/throttle 和 PnL attribution 闭环后，再冻结 backtest research contract 并进入 maker 参数 walk-forward。
