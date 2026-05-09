# Iter1 Checkpoint and Priority 1 Acceptance Plan

Date: 2026-05-04

## Baseline Checkpoint

当前代码存档点命名为 `iter1`。

本地归档目录：

```text
local_live_analysis_iter1/archive/iter1_code_snapshot/
```

归档内容应包含：

- `head_commit.txt`: 当前 `HEAD`。
- `git_status_short.txt`: 当前 dirty/untracked 状态。
- `tracked_changes.patch`: tracked 文件相对 `HEAD` 的 binary diff。
- `tracked_changes.stat`: tracked diff 统计。
- `code_file_list.txt`: 本次代码/文档源码归档文件列表。
- `iter1_code_snapshot.tar.gz`: `examples/binance_tick_mm/` 和 `docs/` 下的代码、配置、测试、文档快照。
- `iter1_code_snapshot.tar.gz.sha256`: 快照 checksum。

这个 checkpoint 不做 git stash/tag/commit，原因是当前工作区包含大量 dirty 和 untracked 文件；用本地归档保留当前状态，不改变工作树。

## Priority 1 Goal

目标不是一次性解决所有 review 问题，而是先消除会污染 maker 策略优化输入的口径/实现差异。

完成 Priority 1 后，应达到：

- live/backtest 对 API/quote throttle 的判定语义一致。
- live 的 post-send observed latency 不再污染 `dropped_by_latency` 和 `reject_reason`。
- audit replay schedule 明确只消费 live `decision` 行。
- 现有测试通过，并新增覆盖关键 parity case。
- 可以基于修正后的代码重跑 iter1 backtest alignment，得到可解释的 API/throttle mismatch breakdown。

不纳入 Priority 1 的事项：

- 完整 PnL attribution。
- 严格 `fill_after_cancel_request` 语义修正。
- iter0 recomputed artifact 重归档。
- maker 参数扫描。
- queue/fill model 调优。

## Workstream 1: Quote Throttle State Parity

### Problem

当前 live 和 backtest 对 `QuoteThrottleState` 的更新时间不一致：

- live 在每个实际 API action 后调用 `throttle_state.mark_sent(...)`。
- backtest 只在 executed actions 中包含 `submit` 时调用 `update_quote_throttle_state(...)`。

iter1 config 开启了 `two_phase_replace_enabled = true`，所以 cancel-only 阶段很常见。这个差异会直接影响后续 submit 是否被 `quote_throttle` 或 `api_interval_guard` 拦截。

### Implementation Plan

1. 在 `strategy_core.py` 中定义唯一的 throttle state update helper。
2. 明确语义：
   - submit quote action 应更新 throttle state。
   - normal-side cancel action 作为 two-phase quote replace 的第一阶段，也应更新 throttle state。
   - pure extra-order cleanup cancel 不应更新 throttle state，因为 `should_throttle_quote_update()` 已将它排除在 quote throttle 之外。
3. live 和 backtest 都只调用这个 shared helper，不直接调用 `mark_sent(...)`。
4. 保持 `should_throttle_quote_update()` 的 bypass 语义一致：pos limit 和 extra cleanup 不触发 quote throttle。

### Unit Acceptance

新增或更新 `examples/binance_tick_mm/test_backtest_tick_mm.py`：

- `submit_buy` 会更新 `last_sent_api_ts` 和 last target ticks。
- `cancel_buy` / `cancel_sell` 会更新 throttle state。
- `cancel_extra` 不更新 throttle state。
- live/backtest 使用同一个 helper，禁止 live 直接调用 `QuoteThrottleState.mark_sent(...)`，除 helper 内部外。

### Artifact Acceptance

用修正代码重跑 iter1 audit replay backtest，对比旧 iter1：

- action match 不低于 `0.97`。
- API drop abs diff 应低于旧 iter1 `0.0801`，或 top mismatch breakdown 能证明剩余差异不是 throttle state parity 导致。
- `throttle_reason_mismatch_top` 中由 cancel-only state divergence 导致的模式应明显下降。

## Workstream 2: Live Latency Guard Semantics

### Problem

live 先用 `feed_latency_ns` 做前瞻 latency guard，但发单后如果 observed `entry_latency_ns > latency_guard_ns`，会把同一 decision 反标成：

- `dropped_by_latency = 1`
- `reject_reason = latency_guard`，当原本没有 reject reason 时

backtest 不做这个 post-send 反标。这个行为会污染 drop rate 和 reject reason alignment，也不符合“latency guard 是发单前保护信号”的口径。

### Implementation Plan

1. live 保留 `entry_latency_ns` / `resp_latency_ns` 原始观测值。
2. 删除 post-send observed latency 对 `dropped_by_latency` 和 `reject_reason` 的修改。
3. 如需要诊断，可追加独立字段，例如 `observed_entry_latency_over_guard`。如果不追加字段，先通过 `entry_latency_ns` 分布做诊断。
4. 更新 `docs/binance_tick_mm.md`：live 和 backtest 当前 guard signal 都应是 feed-latency based，observed order latency 是 audit/diagnostic，不是前瞻 reject reason。

### Unit Acceptance

建议把 decision finalization 中的 latency/drop 口径抽成可测 helper，或至少新增针对逻辑函数的测试：

- 当 pre-send `feed_latency_ns <= latency_guard_ns` 且 action 已发送，即使 observed `entry_latency_ns > latency_guard_ns`，`reject_reason` 也不应变成 `latency_guard`。
- 当 pre-send `feed_latency_ns > latency_guard_ns`，仍应标记 `dropped_by_latency = 1` 和 `reject_reason = latency_guard`。
- backtest 现有 feed-only latency guard 测试保持通过。

### Live Acceptance

旧 iter1 live CSV 已经写入旧口径，不能完整验收这个改动。需要下一次小仓位 live canary 验收：

- decision rows 中，`action != keep` 且 `reject_reason == latency_guard` 的行应为 `0`，除非该 action 字段表示计划动作而非实际动作。
- `dropped_by_latency = 1` 的行必须对应 pre-send `latency_signal_ms > latency_guard_ms`。
- observed `entry_latency_ns` 长尾仍可在 audit summary 中看到，但不改变 reject/drop 口径。

## Workstream 3: Audit Replay Schedule Filters Decision Rows

### Problem

`compare_audit.py` 主指标只使用 `event_type == decision` 行，但 `_load_audit_cadence_schedule()` 当前只按 `run_id` 和 timestamp 读取，不过滤 lifecycle/diagnostic rows。

iter1 当前没有明显击穿，因为 lifecycle rows 大多共享 decision timestamp；但这个隐含不变量不应成为 cadence contract。

### Implementation Plan

1. `_load_audit_cadence_schedule()` 默认只加载：
   - missing/empty `event_type`
   - `event_type == "0"`
   - `event_type == "decision"`
2. 保持 legacy CSV 兼容：如果 CSV 没有 `event_type` 列，则按旧行为处理。
3. 在 result JSON 中增加 schedule diagnostics：
   - raw decision row count
   - unique schedule count
   - deduped decision timestamp count
   - ignored non-decision row count

### Unit Acceptance

新增测试：

- mixed CSV 中 lifecycle row 带唯一 `ts_local` 时，不进入 replay schedule。
- old CSV 没有 `event_type` 时仍能加载 schedule。
- decision rows 有重复 timestamp 时，schedule 去重，并报告 deduped count。

### Artifact Acceptance

重跑 iter1 audit replay：

- `audit_replay_scheduled_count == live decision rows`，iter1 预期为 `62620`。
- `audit_replay_consumed_count / audit_replay_scheduled_count >= 0.999`。
- `audit_replay_skipped_due_count == 0`。
- `audit_replay_unconsumed_count <= 1`，或有明确解释。

## Combined Acceptance Process

### Step 1: Static Checks

运行：

```bash
rg -n "mark_sent\\(" examples/binance_tick_mm
```

验收：

- 除 `strategy_core.py` 中 shared helper 或 `QuoteThrottleState.mark_sent()` 定义外，live/backtest 不直接调用 `mark_sent(...)`。

运行：

```bash
rg -n "entry_latency_ns > latency_guard|reject_reason = \"latency_guard\"" examples/binance_tick_mm/live_tick_mm.py
```

验收：

- live 中不存在 post-send observed entry latency 反标 `reject_reason` 的逻辑。
- pre-send latency guard 逻辑仍存在。

### Step 2: Unit Tests

运行：

```bash
python -m pytest \
  examples/binance_tick_mm/test_backtest_tick_mm.py \
  examples/binance_tick_mm/test_compare_audit.py \
  examples/binance_tick_mm/test_latency_from_audit.py \
  examples/binance_tick_mm/test_pipeline_live_raw.py \
  examples/binance_tick_mm/test_align_live_run.py
```

验收：

- 全部通过。
- 新增 tests 覆盖 throttle parity、decision-only schedule、live latency guard semantics。

### Step 3: Iter1 Offline Replay

用修正后的代码重跑 iter1 audit replay，输出到新的验收目录，例如：

```text
local_live_analysis_iter1/iter1_priority1_acceptance/
```

建议保留：

- `config_backtest_audit_replay.toml`
- `backtest_audit_replay_result.json`
- `alignment_report_audit_replay.json`
- `out/backtest_audit_replay/audit_bt_audit_replay.csv`
- mismatch breakdown 摘要

验收指标：

| Metric | Target |
| --- | ---: |
| scheduled count | equals live decision rows |
| consumed ratio | `>= 0.999` |
| skipped due rows | `0` |
| unconsumed rows | `<= 1` or explained |
| action match | `>= 0.97` |
| latency drop abs diff | `<= 0.02` |
| API drop abs diff | `< 0.0801` preferred; otherwise explain top cases |
| throttle mismatch top | no obvious live/backtest state-update parity pattern |

说明：旧 iter1 live CSV 仍带旧 live latency口径，所以 offline replay 不能完全验收 Workstream 2，只能确认 backtest side 和 compare side 没有回归。

### Step 4: Next Live Canary

下一次小仓位 live canary 后，用同窗口 backtest replay 验收：

- final bot position 与 REST position 一致。
- final REST open orders 为 `0`。
- terminal `open_order_mismatch = 0`。
- `action_match >= 0.97`。
- `latency drop abs diff <= 0.02`。
- `API drop abs diff <= 0.05`，或 top mismatch 有明确 residual-risk 解释。
- no post-send latency guard contamination：
  - actual sent action rows 不应因 observed entry latency 被标成 `latency_guard` reject。

## Exit Criteria For Starting Maker Optimization

Priority 1 完成后，可以进入“最小策略优化前置阶段”，但还不是完整 research contract。

允许开始的优化范围：

- base spread
- inventory skew
- quote size
- quote refresh cadence
- quote throttle 参数

仍不建议优化的范围：

- cancel/fill race specific policy
- adverse-selection filter
- queue/fill model 参数
- 任何只看 net PnL、没有 attribution 的大规模参数扫描

进入 maker 参数优化前，还需要完成最小 PnL attribution。Priority 1 的作用是确保后续 attribution 和参数优化不会先被 API/throttle/latency 口径污染。
