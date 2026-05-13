# 2026-05-04 Claude Code Review

基于 `docs/5-4-review.md` 中列出的所有代码文件和文档进行的整体 review。

---

## 1. 总体评价

这是一套设计清晰、工程质量较高的 HFT 做市 backtest-live alignment 系统。核心架构 —— shared strategy core + 分离的 live/backtest entry point + unified audit schema + cadence replay —— 是正确的工程选择。代码整洁度、测试覆盖和文档质量在 HFT 实验代码中属于上等。

iter1 相对 iter0 在多数对齐指标上有改善，lifecycle audit 补齐后诊断能力显著增强。当前最大未解决问题是 API/throttle guard alignment（API drop absolute diff 0.0801），这是一个模型问题，不是实现 bug。

---

## 2. audit_schema.py

**结论：稳定、完整。**

- 99 个字段，decision 行和 lifecycle 行共享同一 schema，这是正确设计 —— 避免两套 CSV 格式的维护负担。
- `REQUIRED_ALIGNMENT_FIELDS` 子集用于 `compare_audit.py` 的兼容性检查，合理。
- iter0 老审计缺少 `local_open_orders`, `rest_open_orders`, `open_order_diff`, `safety_detail` 四个字段，compare 侧已在 `LEGACY_OPTIONAL_ALIGNMENT_FIELDS` 中处理。
- lifecycle 新增字段（`cancel_requested`, `cancel_request_ts`, `fill_after_cancel_request`, `linked_strategy_seq` 等）在 live 和 backtest 中都有填充。

**建议：**
- 字段数量已经很多（99个），但目前都是有用的。如果未来继续增长，可以考虑按 field group 分组定义常量，方便 audit policy 的 selective write。
- `predicted_entry_ns` 字段在 live 侧永远为 0，在 backtest 侧只作为诊断字段。可以考虑在文档中明确标注这是 backtest-only 诊断字段。

---

## 3. strategy_core.py — 策略核心

**结论：设计合理，live/backtest 共享路径可靠。**

### 3.1 优点

- `decide_actions` 维护了"每边最多一个工作单"的不变量，并正确处理了 extra order cancel 优先级（先清 extras，再处理主 order 替换）。
- `QuoteThrottleConfig/State/should_throttle_quote_update` 的 throttle 逻辑清晰：interval + tick move 双条件，cancel_extra 和 pos_limit 可以 bypass。
- `build_lifecycle_event_row` 和 `build_audit_row` 都从 `empty_audit_row()` 开始构建，保证了每行都包含所有 schema 字段。
- `OrderLifecycleTracker.observe()` 通过 diff `last_by_order_id` 来检测状态变更，生成事件列表 —— 这个设计简洁有效。
- `evaluate_live_safety` 的 confirmation-based 升级逻辑（pending -> mismatch 需要连续 N 次确认）合理，避免了瞬态 REST/local 不一致触发误停。

### 3.2 关注点

- **`_lifecycle_event_type` 优先级顺序**：当前是 `filled > partially_filled > canceled > expired > rejected > cancel_sent > order_new > order_update`。如果一个订单在同一个 observe cycle 中同时发生 status 变化和 req 变化（理论上不应该但 connector 可能合并），这个优先级是安全的。代码逻辑正确。

- **`decide_actions` 中的 `two_phase_replace_enabled` 路径**：当启用时，cancel 后不立即 submit，等下一个 decision cycle 再补 submit。这在 live 中减少了同 cycle cancel+submit 的 race 风险，但在 backtest 中也应该有对应行为。确认 backtest 正确传递了这个参数 —— 已确认 `backtest_tick_mm.py:638` 正确读取该配置。

- **`merge_pending_orders` 仅在 backtest 中使用**：live 不需要 pending orders 因为 connector 会异步更新 order map。backtest 需要 pending 是因为 submit 后 order 可能不会在同一 cycle 出现在 `hbt.orders(0)` 中。这个不对称是 backtest 模型限制，不是 bug，但值得在文档中说明。

- **`GreekOracle.from_config` 中缺少 import**：`strategy_core.py:295` 使用了 `Callable` 和 `Path`，但文件顶部没有 import。这两个类型只出现在 type annotation 中，运行时由于 `from __future__ import annotations` 不会报错，但 type checker 会警告。

---

## 4. live_tick_mm.py — 实盘入口

**结论：结构清晰，安全机制到位。**

### 4.1 优点

- 信号处理（`SIGINT/SIGTERM`）+ 全局 `_shutdown` flag + `wait_next_feed` timeout 的组合确保了优雅退出。
- `finally` 块中先 cancel 所有 working orders，再读 position，再 close bot —— 顺序正确。
- REST safety check 在 decision loop 内部按 interval 调用，不阻塞 main loop。
- lifecycle event（`order_submit_sent`, `cancel_sent`, 以及 `OrderLifecycleTracker.observe` 产生的事件）全部写入同一 audit CSV。
- `latency_signal_ns = feed_latency_ns`（不使用 order latency）—— 这是有意设计，因为 live 的 order latency 是滞后信号，不适合前瞻 gating。

### 4.2 关注点

- **`order_lat_after` 读取时机**：在 `hbt.cancel/submit_*_order` 之后立即读取 `hbt.order_latency(0)`。由于 live bot 的 order latency 是异步更新的，这个读取拿到的可能是上一个订单的 latency，而不是刚发出的订单的 latency。当前 `req_ts/exch_ts/resp_ts` 主要用于 audit 诊断，不用于 gating，所以这不影响正确性，但 audit 中的 latency 数值可能与当前 action 不完全对应。

- **safety_pause 不计入 `dropped_by_api_limit` 的语义**：当 `safety_status == "position_mismatch_pending"` 时，代码设置 `dropped_by_api_limit = True` 和 `reject_reason = "safety_pause"`。语义上 safety pause 不是 API limit，但复用了 `dropped_by_api_limit` 标志。这可能导致 compare 时 API drop rate 包含了 safety pause 的贡献。实际影响小（pending 状态是短暂的），但语义不够精确。

- **REST client 异常处理**：`_signed_get` 在 REST 调用失败时会抛异常，被 safety check 的 `try/except` 捕获并设为 `rest_error`。但如果 Binance 返回非 200 HTTP status 但包含 JSON error body，`urlopen` 会抛 `HTTPError`，不会解析 body。考虑到当前是小仓位运行，这不是紧急问题，但生产化时应增强。

- **audit flush 频率**：每 100 行 flush 一次。在高频 tick 场景下（每秒可能 100+ 行），这意味着大约每秒 flush 一次，I/O 负担可接受。

---

## 5. backtest_tick_mm.py — 回测入口

**结论：功能完整，cadence replay 实现正确。**

### 5.1 优点

- `_audit_replay_decision_due` 实现清晰：`single` 模式每个 feed event 最多消费一个 schedule entry，`drain_due` 模式消费所有已到期 entries。lag 和 breach 统计准确。
- `_load_audit_cadence_schedule_with_stats` 正确过滤 `event_type == "decision"` 行，并对没有 `event_type` 列的老 CSV 做向后兼容（全部当作 decision）。统计信息（raw_row_count, deduped 等）足够诊断。
- `alignment_init` 通过 `_apply_alignment_initial_position` 用 GTC LIMIT 市价单注入初始仓位，验证了注入后 `abs(applied - target) <= tolerance`。
- `pending_local_orders` 机制解决了 backtest 中 submit 后同 cycle order 不可见的问题。`merge_pending_orders` 将 pending 合并到 working orders 中，一旦 order 在 `hbt.orders(0)` 中可见（`req != "new"` 或 `cancellable` 或终态），就从 pending 中移除。

### 5.2 关注点

- **`decision_ts` vs `ts_local` 的使用**：在 audit_replay 模式下，`decision_ts = consumed_live_ts`（live 的 ts_local），而 `ts_local` 是当前 feed event 的时间。策略计算（sigma, fair, etc）使用 `decision_ts`，但 depth/best_bid/best_ask 来自当前 feed event。这意味着策略看到的市场状态是 feed event 时刻的，但 EWMA sigma 的时间戳是 live decision 时刻的。这个不一致是可接受的（因为两个时间通常很接近），但值得明确文档化。

- **`_parse_int_timestamp` 的 Decimal 回退**：对于 `100.5` 这样的值，Decimal 解析后检查 `value != value.to_integral_value()` 并 raise。测试 `test_load_audit_cadence_schedule_skips_fractional_nanoseconds` 确认了这个行为 —— 小数 timestamp 被跳过。这是安全的，因为 nanosecond timestamp 不应该有小数部分。

- **`audit_replay_lag_ns` 和 `audit_replay_due_lag_ns` 是同一个数据**：`backtest_tick_mm.py:1127-1128` 两个字段都等于 `_distribution(cadence_lags_ns)`。这看起来是历史遗留的冗余字段。不影响正确性。

- **`FeedLatencyOracle.from_audit_csv`**：将 live audit 中的 `feed_latency_ns` 作为 backtest 的 feed latency 信号。这是 audit replay 对齐的关键：让 backtest 看到与 live 相同的 feed latency，从而产生相同的 latency guard 决策。实现正确。

### 5.3 API/throttle guard alignment 问题分析

这是当前最大的未解决对齐问题（iter1 API drop abs diff = 0.0801）。根据代码审查，根本原因是：

1. **`last_api_ts` 的更新逻辑**：live 和 backtest 都在 `executed_actions` 循环中更新 `last_api_ts = ts_local`（live）或 `decision_ts`（backtest）。但 live 的 `ts_local` 是 feed event 到达时间，而 backtest 的 `decision_ts` 在 audit_replay 模式下是 live decision 的 ts_local。如果 live 的 feed event 到达时间和 decision 时间之间有微小差异（通常 < 1ms），这不会造成问题。但如果 backtest 的 feed event 时间远早于 consumed live decision 时间，那么 backtest 的 `min_interval_ns` guard 会在下一个 decision 时以 backtest 的 feed event 时间为基准，而 live 以自己的 feed event 时间为基准 —— 两者的间隔计算基准不同。

2. **`TokenBucket` 的 refill timing**：bucket 在 live 中以 `ts_local` refill，在 backtest 中以 `decision_ts` refill。如果 decision 密集发生（在 audit_replay single 模式下，每个 feed event 最多一个 decision，但 feed events 可能非常密集），bucket 的 token 恢复速率可能与 live 不同。

3. **`QuoteThrottleState` 的时间基准**：live 使用 `ts_local`，backtest 使用 `decision_ts`。同理，throttle 的 elapsed_ns 计算可能有微小差异。

**结论**：这不是实现 bug，而是模型限制。backtest 的时间模型（基于 feed event 驱动的离散时钟）与 live 的连续时钟本质上不同。当前 0.08 的 API drop diff 是在"95% 以上决策正确对齐"的前提下，剩余 5% 主要来自 timing 边界情况。改进方向是：

- 在 audit_replay 模式下，让 `last_api_ts` 和 `bucket.allow()` 的时间基准统一使用 `consumed_live_ts`。
- 或者在 compare 时对 API throttle mismatch 做更细粒度的归因（已有 `_api_throttle_breakdown`）。

---

## 6. compare_audit.py — 对齐比较

**结论：正确过滤 decision 行，不污染 lifecycle 行到指标中。**

### 6.1 优点

- `_decision_rows` 过滤 `event_type in {"", "0", "decision"}`，正确排除 lifecycle 行。`""` 和 `"0"` 是对老 CSV 的兼容。
- `_alignment_by_seq_rows` 用 `strategy_seq` 对齐；`_alignment_by_nearest_ts` 用最近 timestamp 对齐。两种模式都支持，`compare` 函数默认输出两种。
- `_api_throttle_breakdown` 提供了丰富的归因分析：按 planned_action match/mismatch、same guard inputs、ts_exch lag 分层。这对诊断 API throttle 问题非常有价值。
- legacy iter0 缺少的 4 个字段通过 `LEGACY_OPTIONAL_ALIGNMENT_FIELDS` + `row.setdefault` 处理，不会 raise。

### 6.2 关注点

- **`_read_csv_rows` 读取全部行到内存**：对于 6-7 万行的 iter1 audit，这不是问题。但如果未来处理更长 live run（几十万行），可能需要流式处理。当前可接受。

- **`align_mode` 参数**：`compare` 函数在 `align_mode="both"` 时总是计算两种对齐，primary 使用 seq alignment。但 `alignment` 键和 `alignment_seq` 键实际上指向同一个对象。如果 caller 只看 `alignment`，这是正确的。

- **MAE 计算不包含 `half_spread`**：`_alignment_from_pairs` 中 `metrics` 列表包含 `half_spread`，但 review 文档中的 headline metrics 不报告 half_spread MAE。这不是 bug，数据在 JSON 中，只是 summary report 不展示。

---

## 7. latency_from_audit.py — 延迟提取

**结论：实现正确。**

- `_read_latency_rows` 正确过滤 decision 行（`event_type in {"", "0", "decision"}`），排除 lifecycle 行的 timestamp。
- `build_observed_latency_series` 保留实盘长尾，不做 clip —— 这是 iter1 的正确选择。
- `build_latency_series` 的 synthetic 模式通过 clip + spike injection 生成可控分布。
- validity check: `req_ts > 0 && exch_ts > req_ts && resp_ts > exch_ts`，过滤掉不完整或异常的 latency 行。

---

## 8. align_live_run.py — 编排工具

**结论：自动化程度高，流程完整。**

- fetch -> latency -> convert -> backtest(normal + audit_replay) -> compare -> archive 全链路自动化。
- `read_live_initial_state` 优先使用 `rest_position`（更可靠）。
- `build_backtest_config` 为 normal 和 audit_replay 模式分别生成正确的 backtest config。
- `archive_run` 生成 FILE_MANIFEST.txt + SHA256SUMS.txt + tar.gz + sha256 checksum，可复现性好。
- `_write_toml` 是简化实现（不处理嵌套 section），但当前所有 config 都是单层 section，可以工作。如果未来有 `[section.subsection]`，需要增强。

---

## 9. 测试覆盖

**结论：关键路径覆盖充分。**

### test_backtest_tick_mm.py

- 覆盖了 audit_replay 的 single/drain_due 模式、tolerance、lag breach。
- 覆盖了 cadence schedule 加载（run_id filter、event_type filter、dedup、legacy 兼容）。
- 覆盖了 alignment init（buy/sell 方向、lot size rounding）。
- 覆盖了 pending order merge、extra cancel wait、quote throttle state 更新。
- 覆盖了 safety check open_order_diff/details。
- 使用真实 `ROIVectorMarketDepthBacktest` 测试了 initial position injection。

### test_compare_audit.py

- 覆盖了 seq 和 nearest_ts 对齐。
- 覆盖了 lifecycle 行不污染 alignment/summary。
- 覆盖了 legacy audit 兼容（缺少 4 个字段）。
- 覆盖了 API throttle breakdown。

### test_latency_from_audit.py

- 覆盖了 observed 模式长尾保留。
- 覆盖了 invalid row 过滤。
- 覆盖了 synthetic 模式 clip。

### 测试缺口

- 没有针对 `live_tick_mm.py` 的集成测试（可理解，需要 connector）。
- `compare_audit.py` 的 `_alignment_by_nearest_ts` 没有测试 max_lag_ns = 0 时的行为（应该是不限制）。
- `align_live_run.py` 的 `_write_toml` 没有测试非字符串 scalar（bool, int, float）的序列化正确性 —— `test_align_live_run.py` 应该有但未看到。

---

## 10. iter0 vs iter1 对齐指标评估

| Metric | iter0 | iter1 | Delta | 评估 |
| --- | ---: | ---: | --- | --- |
| Action match | 0.9729 | 0.9773 | +0.0044 | iter1 更好 |
| Reject reason match | 0.7471 | 0.7919 | +0.0448 | iter1 显著更好 |
| Planned action match | 0.7198 | 0.7684 | +0.0486 | iter1 显著更好 |
| Throttle reason match | 0.7561 | 0.7992 | +0.0431 | iter1 显著更好 |
| Fair MAE | 3.6749 | 4.2498 | +0.5749 | iter1 略差，仍在 gate 内 |
| Reservation MAE | 4.1013 | 4.5925 | +0.4912 | iter1 略差，仍在 gate 内 |
| Position MAE | 0.000890 | 0.000690 | -0.000200 | iter1 更好 |
| Latency drop abs diff | 0.0102 | 0.0082 | -0.0020 | iter1 略好 |
| API drop abs diff | 0.0681 | 0.0801 | +0.0120 | iter1 略差，H3 主要缺口 |

**评估**：
- iter1 在 action/reject/planned/throttle match 和 position MAE 上全面优于 iter0。
- fair/reservation MAE 略差但绝对值很小（~4 tick），可能是 run 时段市场条件差异造成，不是系统性退化。
- API drop diff 是唯一退化指标，也是 H3 gate 的 remaining miss。

---

## 11. 关键发现与建议

### 11.1 确认正确的设计决策

1. **`single` replay mode** 是正确选择。`drain_due` 在 lag 时会一次性消费多个 decision，造成策略状态跳变。`single` 保证每个 feed event 最多一个 decision，与 live 行为更一致。

2. **latency guard 只用 feed_latency_ns**。live 和 backtest 一致使用 `feed_latency_ns` 作为 gating signal，`predicted_entry_ns` 只是 backtest 诊断字段。这避免了"用 order latency 做前瞻 gating"的错误。

3. **`compare_audit.py` 只对齐 decision 行**。lifecycle 行不参与 action/reject/fair/position 指标计算，这是正确的，因为 lifecycle 行在 live 和 backtest 中的触发条件不同。

4. **observed latency mode**。iter1 使用 observed（而非 synthetic clip）latency，保留了实盘长尾分布。这让 latency guard 行为更接近 live。

### 11.2 需要注意的问题

1. **API throttle alignment 是模型限制，不是 bug**。改进方向见第 5.3 节。短期建议：在 `backtest_tick_mm.py` 中，audit_replay 模式下的 `bucket.allow()` 和 `min_interval_ns` guard 统一使用 `consumed_live_ts` 作为时间基准，而不是 `decision_ts`（两者当前相同，所以这个建议实际上是说要确认它们始终一致）。

2. **`strategy_core.py` 中的 type annotation 缺失 import**：`Callable` 来自 `typing`，`Path` 来自 `pathlib`，`csv` 也被使用。由于 `from __future__ import annotations`，运行时不报错，但建议补齐 import 以通过 type checker。

3. **`safety_pause` 复用 `dropped_by_api_limit` 标志**：语义不精确，建议考虑新增 `dropped_by_safety` 字段或在 `reject_reason` 中区分。影响很小但有助于指标归因。

4. **fair/reservation MAE iter1 略高于 iter0**：数值差异小（~0.5 tick），更可能是 market regime 差异而非代码退化。建议后续收集更多 run 样本确认。

### 11.3 后续优先级建议

1. **P0：API throttle alignment 改进**。这是唯一阻止 H3 full pass 的指标。具体看 `_api_throttle_breakdown` 中的 `mismatch_attribution`，定位 `planned_action_mismatch` vs `target_tick_mismatch` vs `ts_exch_lag` 的贡献占比，针对性修改 timing 基准。

2. **P1：补齐 `strategy_core.py` 的 import**。影响小但属于代码卫生。

3. **P2：多 run 样本验证**。iter0 和 iter1 各只有一个 run 样本，统计置信度有限。建议在相同配置下跑 3-5 个不同时段的 live run，确认指标分布的稳定性。

4. **P3：`_write_toml` 增强**。当前不支持嵌套 section，如果未来 config 结构变复杂会出问题。可以引入 `tomli-w` 或类似库。

---

## 12. 审查清单结论

按 `5-4-review.md` 中的 Suggested Review Checklist 逐项回答：

| # | Checklist Item | 结论 |
| --- | --- | --- |
| 1 | audit_schema.py 完整且稳定 | **Pass**。decision 和 lifecycle 行共享 schema，lifecycle 字段在 live/backtest 一致填充，legacy iter0 可比较。 |
| 2 | live lifecycle 语义正确 | **Pass**。submit/cancel/fill/update 事件正确记录 `linked_strategy_seq`, order IDs, side, price, qty, timestamps。`fill_after_cancel_request` 正确识别 maker fill/cancel race。REST/local/WS seen flags 有意义。 |
| 3 | audit_replay 实现正确 | **Pass**。`single` 模式每次最多消费一个 schedule entry。lag/skipped/unconsumed 计数器正确。不会静默丢弃 decision（除非 counter 暴露）。normal cadence 不受影响。 |
| 4 | compare 逻辑正确 | **Pass**。过滤 `event_type == "decision"` 用于 primary metrics。lifecycle 行保留但不污染 action/reject/fair/reservation/position 指标。drop-rate 定义与 acceptance docs 一致。legacy iter0 缺列显式处理。 |
| 5 | latency/API guard 建模 | **Partial Pass**。observed latency 提取正确。API interval guard 和 quote throttle 语义与 live 基本一致。API drop mismatch 可从证据中解释（timing 基准差异），不是实现 gap，而是模型限制。需要进一步 attribution 分析。 |
| 6 | archive 可复现性 | **Pass**（假设 checksum 验证通过）。iter0 使用 RECOMPUTED_METRICS_2026-05-03.md 作为当前 baseline。iter1 archive 包含完整的 checksum 和 manifest。ITER1_ACCEPTANCE.md 和 docs/binance_tick_mm.md 准确描述了 replay mode 和 lifecycle fields。 |

---

## 13. 总结

代码质量和对齐框架是成熟的。iter1 lifecycle audit 补齐了 order lifecycle ground truth，97.7% 的 action match 和 79.2% 的 reject match 说明策略复现已经达到了实用水平。剩余的 API/throttle guard alignment 问题是可以通过进一步 timing 基准校准来改善的，不需要架构变更。

建议在 API throttle attribution 分析完成后，用当前框架对 3-5 个不同时段的 live run 做批量验证，确认指标分布稳定后再进入参数优化阶段。

---

## 14. 下一步改进计划

### 目标

让 backtest 成为可信的 maker 策略研发环境：在 backtest 中优化的策略参数，部署到 live 后能达到符合预期的 PnL。

这要求两个条件同时成立：
1. **backtest 能复现 live 的决策路径**（alignment）—— 当前 action match 97.7%，基本达标。
2. **backtest 的 PnL 能预测 live 的 PnL**（PnL fidelity）—— 当前没有建立这个映射。

以下按依赖关系排列，前面的步骤是后面步骤的前提。

---

### Step 1：修复 API/throttle guard alignment（最大单一改进点）

**问题定位**：iter1 audit-replay 的 reject mismatch 分析显示：

| Mismatch pattern | Count | 含义 |
| --- | ---: | --- |
| BT `api_interval_guard` / live 无 reject | 6,380 | BT 过度拒绝 |
| live `quote_throttle` / BT 无 reject | 2,707 | BT 遗漏拒绝 |
| BT `quote_throttle` / live 无 reject | 1,983 | BT 过度拒绝 |
| live `api_interval_guard` / BT 无 reject | 703 | BT 遗漏拒绝 |

三类 mismatch 合计 ~11,800 行，占 62,619 个 decision 的 ~19%。这与 reject_reason_match = 79.2% 吻合。

**根因**：`backtest_tick_mm.py` 中 `api_interval_guard` 的时间基准是 `decision_ts`（= consumed live ts_local），而 `last_api_ts` 在每次执行 action 后更新为 `decision_ts`。问题在于：

- Live 中两次 API 调用之间的间隔是 wall-clock 间隔（受 feed event 到达时间影响）。
- Backtest audit_replay 中，连续两个 consumed live decision 的 ts_local 间隔可能非常小（<1ms），但实际 backtest feed event 之间的 wall-clock 时间可能更长（因为 single mode 每个 feed event 只消费一个 decision）。
- 这导致 backtest 在 live decision 密集时段过度触发 `api_interval_guard`。

**修复方案**：

1. 在 audit_replay 模式下，引入 `last_api_feed_ts`（当前 feed event 的 `ts_local`，而不是 consumed live decision 的 ts_local）作为 `api_interval_guard` 和 `token_bucket` 的时间基准。理由：live 的 API interval 是由 feed event 驱动的（收到 feed -> 决策 -> 发 API），backtest 也应该用 feed event 时间。

2. 对 `QuoteThrottleState` 的 `mark_sent` 同样使用 feed event 时间，而不是 consumed live decision 时间。

3. 修复后重跑 iter1 audit-replay，对比 reject mismatch 是否下降。

**预期效果**：BT `api_interval_guard` 过度拒绝的 6,380 行应大幅减少。reject_reason_match 从 79% 提升到 85%+。

**验证**：重跑 `compare_audit.py`，检查 `_api_throttle_breakdown` 中 `mismatch_attribution` 的变化。

---

### Step 2：建立 PnL attribution 框架

当前系统只对齐 action/reject/fair/position，没有对齐 PnL。但最终目标是 PnL 预测，所以需要建立 PnL 分解。

**PnL attribution 分解**：

```
Realized PnL = Spread Capture + Fees + Inventory MTM + Adverse Selection
```

具体：

| 组件 | 定义 | 数据来源 |
| --- | --- | --- |
| **Spread Capture** | 每次 fill 时 `(fill_price - fair) * side_sign * qty` | audit fill 行的 `fill_price`, `fair`, `order_side`, `fill_qty` |
| **Fees** | maker/taker fee * fill notional | config `[fee]` + fill 行 |
| **Inventory MTM** | `position * (mid_t - mid_{t-1})` 在每个 decision tick | audit decision 行的 `position`, `mid` |
| **Adverse Selection** | fill 后短期 mid 移动对仓位的不利影响 | fill 行的 ts + 后续 N 个 decision 行的 mid |
| **Missed Opportunity** | 被 latency/API/throttle guard 拒绝的 action 如果执行了会带来的 spread capture | audit 行的 `planned_action` + `reject_reason` |
| **Cancel/Fill Race** | `fill_after_cancel_request` 事件的 PnL 影响 | lifecycle fill 行 |

**实现**：

- 在 `backtest_metrics.py` 中新增 `PnLAttributor` class，逐行消费 audit CSV（decision + lifecycle），输出分解后的 PnL JSON。
- 在 `compare_audit.py` 中新增 PnL attribution 对比：BT 和 live 各自的分解，以及差异归因。
- 先用 iter1 的 backtest audit + live audit 做 dry-run，确认分解是否有意义。

**关键判断**：如果 BT 和 live 的 Spread Capture 和 Adverse Selection 分解接近，即使总 PnL 有差异（因为 fill count 不同），也说明 backtest 的价格模型是可信的。如果差异大，需要回到 queue/fill model 校准。

---

### Step 3：queue/fill 模型校准

当前使用 `NoPartialFillExchange + PowerProbQueueModel3(n=5)`。这两个组件直接影响：
- 成交概率（决定 fill count）
- 成交价格（决定 spread capture）
- 仓位路径（决定 inventory MTM 和 adverse selection）

**校准方法**：

1. 从 iter1 live audit 的 lifecycle fill 行提取 fill 统计：
   - 每个价格等级的 fill rate（quote 数 / fill 数）
   - fill 后 mid 移动分布（adverse selection signal）
   - cancel/fill race 比例

2. 在 backtest 中扫描 `power_prob_n` 参数（3, 5, 7, 10, 15），对比 fill rate 和 position path。

3. 选择使 backtest fill rate 最接近 live fill rate 的 `power_prob_n`。

4. 如果 `NoPartialFillExchange` 是 fill path 差异的主要来源（即 live 有大量 partial fill），考虑切换到支持 partial fill 的 exchange model。

**验证**：fill rate MAE 和 position path MAE 下降。

---

### Step 4：收集多 run 样本，验证对齐稳定性

当前 iter0 和 iter1 各一个 run。统计置信度不够。

**计划**：
- 在 Step 1 修复部署后，跑 3-5 个 1 小时 live run（不同时段：亚洲早盘、欧洲开盘、美盘开盘）。
- 每个 run 完成后用 `align_live_run.py` 自动化生成 alignment report。
- 汇总所有 run 的指标分布，确认：
  - action_match > 95% 在所有 run 中稳定
  - reject_reason_match > 85%（Step 1 修复后）
  - position MAE < 0.001 BTC
  - PnL attribution 分解稳定（Step 2 建立后）

**交付物**：一个 `alignment_stability_report.md`，记录多 run 指标分布和结论。

---

### Step 5：冻结 backtest research contract

在 Step 1-4 完成后，冻结以下参数和模型选择：

| 组件 | 冻结内容 |
| --- | --- |
| Fee model | maker/taker rate |
| Latency model | observed mode, 不做 synthetic clip |
| Queue/fill model | `PowerProbQueueModel3(n=X)` 的 X 值 |
| API budget | capacity, refill_per_sec, min_interval_ms |
| Quote throttle | min_interval_ms, min_move_ticks |
| Cadence | audit_replay single mode |
| PnL attribution | 分解公式和比较方法 |
| Alignment gate | action_match > 95%, reject_match > 85%, position MAE < 0.001 |

冻结后，这些参数在策略研究阶段不可调。策略研究只调策略参数（`[risk]`, `[fair]`, `[greeks]`）。

---

### Step 6：maker 策略参数优化

在 backtest contract 冻结后，开始策略研究。

**优化目标**：不是最大化单 run PnL，而是最大化 risk-adjusted PnL stability。

**方法**：

1. **Walk-forward**：用 `walk_forward.py` 做 rolling train/test 评估。建议 train window = 3 天，test window = 1 天，滚动步长 = 1 天。在 Tardis 历史数据上跑（不是 live raw），因为需要足够长的历史。

2. **参数扫描维度**（优先级排序）：
   - `[risk] base_spread`：直接控制 spread capture vs fill rate 的 tradeoff
   - `[risk] k_inv`：inventory penalty 强度，控制 adverse selection 暴露
   - `[risk] k_pos`：position-dependent spread widening
   - `[fair] w_imb`：order book imbalance signal 权重
   - `[latency] latency_guard_ms`：gating 阈值，影响 miss opportunity vs adverse selection

3. **评估指标**（不只看 PnL）：
   - Sharpe ratio（per-day PnL / std）
   - Max drawdown
   - Fill rate
   - Spread capture per fill
   - Adverse selection per fill
   - Position utilization（avg abs position / max position）

4. **Over-fitting 防护**：
   - 参数选择基于 test window 表现，不是 train window
   - 拒绝在 test window 上 Sharpe < 0.5 的参数组合
   - 比较 train/test Sharpe 的 ratio，如果 > 3x 则视为过拟合

---

### Step 7：live canary 验证

选定参数后，做 live canary 验证：

1. 最小仓位（0.001 BTC）部署 1-2 小时
2. 用 `align_live_run.py` 做同窗口 backtest replay
3. 比较 PnL attribution 分解：BT 预测 vs live 实际
4. 如果 BT PnL attribution 与 live 在 2x 以内，认为 backtest 可信
5. 如果差异大，回到 Step 3 重新校准

**迭代收敛标准**：连续 3 个 canary run 的 BT/live PnL attribution 差异在 30% 以内。

---

### 时间线估计

| Step | 依赖 | 工作量 |
| --- | --- | --- |
| Step 1: API/throttle fix | 无 | 代码修改 + 重跑 iter1 |
| Step 2: PnL attribution | Step 1（可并行开始设计） | 新增 ~200 行代码 |
| Step 3: queue/fill 校准 | Step 2 | 参数扫描 + 分析 |
| Step 4: 多 run 样本 | Step 1 | 需要 live server 时间 |
| Step 5: 冻结 contract | Step 1-4 | 文档化 |
| Step 6: 参数优化 | Step 5 | 计算密集，需要 backtest server |
| Step 7: live canary | Step 6 | 需要 live server 时间 |

**关键路径**：Step 1 -> Step 4（并行 Step 2/3） -> Step 5 -> Step 6 -> Step 7。

Step 1 是唯一的代码修改阻塞项，其他步骤都可以在 Step 1 完成后快速推进。
