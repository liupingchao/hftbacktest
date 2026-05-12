# Task Plan

## Workflow

This repository uses `workflow-kit` as the persistent workflow layer for continued hftbacktest development.

The active project focus is the Binance tick-level maker market-making research and live/backtest alignment work.

Reference docs:

- `.workflow/workflow-kit/workflow-manual.md`
- `.workflow/workflow-kit/task-dispatch-template.md`
- `.workflow/workflow-kit/thread-report-template.md`
- `.workflow/workflow-kit/qa-acceptance-template.md`
- `.workflow/workflow-kit/workflow-web-field-mapping.md`

## Threads

- 总控：main Codex/Claude planning session.
- 业务线程-core：Rust core implementation and crate-level changes.
- 业务线程-python：Python binding and packaging changes.
- 业务线程-docs：project docs, workflow docs, and operation notes.
- 测试线程：test discovery, regression runs, failure summaries, and evidence collection.
- QA验收线程：final acceptance result for each task.

## Current Task Pool

| Task ID | Title | Thread | Status | QA |
|---|---|---|---|---|
| `0510T001` | 建立 binance_tick_mm live/backtest 闭环任务模板 | 测试线程 | 待验收 | 正常验收 |
| `0510T002` | 自动执行跨样本 Stage 6J 验证矩阵 | 测试线程 | 待验收 | 正常验收 |
| `0511T001` | adverse-selection timing rule 设计规格与验收合同 | 业务线程-python | 待执行 | 正常验收 |
| `0511T002` | adverse-selection timing guard default-off 实现与 Stage 6J replay | 业务线程-python | 待执行 | 正常验收 |
| `0511T003` | 升级 workflow dashboard 为实验决策看板 | 业务线程-docs | 待执行 | 正常验收 |
| `0511T004` | adverse timing trigger 未命中原因诊断 | 测试线程 | 待验收 | 正常验收 |
| `0512T001` | 对齐 Stage 6J replay 与 live adverse-selection source-path | 测试线程 | 已通过 | 正常验收 |
| `0512T002` | add-side submit/re-add toxic timing rule 设计合同 | 业务线程-python | 已通过 | 正常验收 |
| `0512T003` | 5-11-night-active live 样本 replay/acceptance/cancel-fill 分析 | 测试线程 | 已通过 | 正常验收 |
| `0512T004` | Stage 6J / live adverse-selection 观测门禁改进合同 | 测试线程 | 已通过 | 正常验收 |
| `0512T005` | add-side toxic timing guard default-off 实现与离线 replay | 业务线程-python | 已通过 | 正常验收 |
| `0512T006` | T005 blocked-row attribution 分析计划 | 测试线程 | 已通过 | 正常验收 |
| `0512T007` | T005 toxic timing attribution 实验 | 测试线程 | 已通过 | 正常验收 |
| `0512T008` | hbt.depth live/replay view 确认与数据层质量门禁 | 测试线程 | 待验收 | 正常验收 |
| `0513T001` | MarketView provenance / top5 audit transparency 规划合同 | 业务线程-python | 已通过 | 正常验收 |
| `0513T002` | MarketView provenance / top5 audit transparency 最小实现 | 业务线程-python | 待验收 | 正常验收 |
| `0513T003` | 5-13-day-control-15min T002 live-data 验证 | 测试线程 | 执行中 | 正常验收 |

## Current Project Sources

Primary code:

- `examples/binance_tick_mm/`

Primary planning and acceptance docs:

- `docs/5-8-future-plan.md`
- `docs/binance_tick_mm_alignment_execution_plan.md`
- `docs/maker_optimization_acceptance.md`
- `docs/5-4-plan.md`
- `docs/5-8-terminal-reconcile-plan.md`
- `docs/stage6f-5-8-night-plan.md`
- `docs/stage6i-cancel-requested-fill-risk-plan.md`
- `docs/dry-run-plan-5-8.md`

Current accepted baseline from existing docs:

- `5-9-small` / Stage 6G is the current action-path acceptance baseline.
- Maker optimization replay must keep audit overlays off.
- `maker_acceptance.py` is the hard-gate script before optimization.
- Next work should preserve live/backtest action, planned action, reject, throttle, replay lag, and working-order semantic parity.

## Post-T007 Research Direction Backlog

Status:

- These are strategic research directions, not formal task IDs yet.
- Do not treat this section as authorization to implement, replay new candidates, or run live.
- Each direction below should later be split into a narrow workflow task with its own task file, acceptance criteria, and QA.
- `0512T007` ruled out continuing the same `pending_cancel+target_move` pure toxic timing rule with only 50/100/200ms window tuning. Future work should not extend that line without a new design contract.

### Direction A: Fair Price / Microprice / OFI Predictive Power

Motivation:

- The current maker strategy uses a simple fair-price / greeks model and does not yet prove that its short-horizon fair value is competitive with order-book microstructure signals.
- Literature and market-making practice suggest that microprice, order-book imbalance, and order-flow imbalance may be more directly tied to short-horizon adverse selection than a binary submit timing guard.

Future task shape:

- Build a read-only feature study over accepted live/replay samples.
- Compute mid, weighted mid, microprice, top-of-book imbalance, multi-level imbalance, order-flow imbalance, and simple book-pressure features.
- Evaluate 100ms / 500ms / 1s / 5s side-adjusted markout and directional prediction power.
- Report predictive power separately from strategy PnL.
- Do not modify live strategy or quote placement in the first task.

Expected outputs:

- Feature table by timestamp / decision row.
- Markout correlation / bucket table.
- Stability comparison across `5-11-night-active`, `5-10-day-control-1h-06`, `5-9-noon`, and `5-9-small`.
- Recommendation on whether fair-price adjustment is worth a later design task.

Reference starting points:

- Stoikov, `The Micro-Price`: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694
- Deep Order Flow Imbalance: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3900141

### Direction B: Queue Position / Fill Quality Attribution

Motivation:

- Maker edge is strongly affected by queue position, queue age, and whether cancel/re-add loses valuable queue priority.
- T007 showed that suppressing some submits did not hit replay risk orders. The next useful question is whether risky fills are better explained by queue state and order value than by submit timing windows.

Future task shape:

- Build a read-only queue/fill attribution study.
- Estimate queue-ahead or proxy queue position at submit time where data permits.
- Track queue age, cancel request age, fill-after-cancel-request, same-side re-add, missed fill, and post-fill markout.
- Separate good fills, adverse fills, cancel-requested fills, missed fills after cancel, and stale queue retention.
- Do not add a rule in the first task.

Expected outputs:

- Per-order fill quality table.
- Queue-age / queue-position bucket markout.
- Cancel/re-add queue-loss attribution.
- Decision on whether future optimization should preserve queue, step back, widen, reduce size, or cancel faster.

Reference starting point:

- Queue position valuation in a limit order book: https://business.columbia.edu/faculty/research/model-queue-position-valuation-limit-order-book

### Direction C: Quote Adjustment Instead Of Binary Submit Suppression

Motivation:

- T005/T007 indicate that a binary `suppress submit` timing rule can hit the decision path without filtering the risk-source orders.
- A more natural maker strategy lever is quote adjustment: skew, spread widening, size reduction, join/step-back choice, or inventory-aware reservation price.

Future task shape:

- Only after Direction A or B produces evidence, design a default-off quote adjustment contract.
- Candidate controls may include fair-price shift, spread widening, inventory skew, size throttle, or queue-aware join/step-back.
- The first design task must define action-path coverage, replay-model regression, and live-derived source-path proof separately.
- Do not promote to live without replay, acceptance, risk diagnostics, and QA.

Expected outputs:

- Design contract for quote adjustment candidates.
- Explicit no-live boundary.
- Candidate matrix that includes baseline and conservative controls.
- Acceptance criteria for PnL, position, drop/API, churn, cancel-fill source-path, and action-path changes.

Reference starting point:

- Prediction-Based Limit Order Trading: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4320775

### Direction D: Market Data / Local Book Quality Gate

Motivation:

- Microprice, OFI, and queue attribution are only meaningful if the local order book is correctly reconstructed and latency-stamped.
- Binance depth streams and local order book synchronization should be treated as a data-quality gate before trusting microstructure features.
- `hbt.depth(0)` exists in both live and backtest, but the view may still differ because live reads the connector-maintained book at decision time while backtest reads a replay-reconstructed book. The shared API does not prove identical market state.
- The current audit alignment mainly proves the compressed decision surface, such as best bid/ask, mid, top5 sizes, target ticks, actions, reject/throttle state, and working-order semantics. It does not prove full L2 book, queue state, OFI, or microprice equivalence.
- Existing audit replay overlays can force live-compressed market state into replay, which is useful for action-path alignment but can hide whether replay depth reconstruction itself matches the live book.

Future task shape:

- Validate local book reconstruction against Binance depth stream semantics.
- Check sequence gaps, update-id continuity, bookTicker / depth consistency, top-of-book drift, and timestamp latency.
- Report whether existing live samples are good enough for microprice / OFI / queue studies.
- Confirm exactly what view the strategy receives in live and backtest: which fields are read from `hbt.depth(0)`, where they are compressed into fair/target/audit fields, and whether `strategy_core` ever sees full depth.
- Compare live audit market fields with replay-reconstructed depth on matched decision timestamps, including best bid/ask, top5 ticks/qtys, target ticks, and stale/missing book rows.
- If the comparison is insufficient for queue / microprice / OFI, define a later modification task to extend audit capture with top-N book, update ids, exchange timestamps, local receipt timestamps, and bookTicker-vs-depth consistency fields.

Expected outputs:

- Local book quality report.
- Gap / resync / latency summary.
- Decision on whether current samples can support Direction A/B, or whether a new no-rule data collection run is needed.
- Explicit recommendation on whether the current audit fields are enough, or whether the data layer and audit schema must be modified before microstructure-signal work.

T008 execution note:

- `0512T008` has been executed and is waiting for QA. It confirmed that current audit alignment is sufficient for compressed action-path alignment of the existing simple strategy, but not sufficient to prove full L2 / queue / OFI / microprice equivalence.
- `market_state_overlay=audit` forces live compressed market/fair/target fields into audit replay, but does not overlay top5 tick/qty strings; those remain replay-depth derived.
- Stage 6J no-overlay matched-decision comparisons still show material live/replay view differences, especially on `5-11-night-active`.
- Converted npz files do not retain Binance `U/u/pu` or `lastUpdateId`; raw gzip has them, but decision-row audit does not expose per-decision book provenance.
- `0513T001` is the date-updated planning-only replacement for the previously named T009. It writes the implementation plan and acceptance scheme only; it does not implement code, change `hbt.depth(0)`, run replay, or start live.
- If QA accepts `0513T001`, the next formal implementation task should be a separate `0513T002` focused on MarketView provenance / top5 audit transparency minimal implementation before microprice / OFI / queue feature research.

0513T001 planning decision:

- Do not modify the core `hbt.depth(0)` API first. Improve transparency at the Binance maker strategy layer with an explicit `MarketView` / `BookViewSnapshot` wrapper.
- Both live and backtest should eventually build decision market view through one helper, likely `build_market_view_from_depth(...)`.
- top5 alignment is necessary for current compressed decision-view transparency, but it is not sufficient for full L2 / queue / OFI / microprice proof.
- Any top5 overlay must include provenance fields and must not be interpreted as replay reconstructed book alignment.
- If Python strategy/backtest layers cannot expose update ids, bookTicker provenance, or per-decision book provenance, a later core/data task may be needed.

Reference starting point:

- Binance WebSocket Streams / local order book management: https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams

### Suggested Decomposition Order

1. First formal follow-up should be a read-only data/feature task, likely Direction A plus the minimum necessary Direction D checks.
2. Second formal follow-up should be queue/fill attribution, Direction B, if current samples contain enough book/order lifecycle detail.
3. Only after A/B evidence exists should we create a quote adjustment design task, Direction C.
4. No live micro test should be considered until a later implementation has passed replay, acceptance, risk diagnostics, and QA.

## Standard Binance Maker MM Loop

Every serious live/backtest iteration should follow this loop unless the task explicitly says otherwise:

1. 采集 live 样本：固定策略参数，不开新规则，使用清楚 run id，保存 live audit、raw gzip、connector/bot 日志。
2. 拉回并归档：拉回到 `local_live_analysis/<run_id>/`，生成 `local_live_analysis/archive/<run_id>.tar.gz`，记录 start/stop 时间。
3. replay/acceptance 验收：跑 normal replay、audit replay 和 `maker_acceptance.py`；先确认 action/planned/reject/throttle、working-order、API/throttle、strict replay lag gates。
4. 风险诊断：跑 `analyze_cancel_fill_risk.py`，检查 cancel-fill source-path、markout 和 latency bucket。
5. 判断问题类型：框架不对齐先修框架；对齐通过但风险高才进入规则设计。
6. 离线规则 replay：用同一批样本跑 Stage 6J replay，比较 baseline、add-side guard、cooldown 和 adverse-selection timing rule。
7. 跨样本验证：至少覆盖 daytime 和 night-active，检查 PnL、max position、drop rate、churn、cancel-fill source-path 是否稳定改善。
8. live micro test 决策：多样本 replay 通过后才开小 notional live，然后回到第 1 步。

## Stable Rules

- One formal task per task ID.
- One task should stay narrow enough to verify.
- QA acceptance is the final task result source of truth.
- `progress.md` records the current operating state.
- `findings.md` records durable risks, failures, and lessons.
- `.workflow/dashboard.html` and `.workflow/dispatch_suggestions.md` are generated by `.workflow/build_dashboard.py`.
