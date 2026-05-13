```md
执行线程：
- 业务线程-python

任务ID：
- 0513T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 正常验收。重点检查本任务是否只做 planning-only 的实现计划和验收方案，是否把原 T009 改为日期更新后的 `0513T001`，是否没有改策略代码、没有改 `hbt.depth(0)` 底层 API、没有运行 replay/live。

files：
- .workflow/tasks/0513T001.md
- .workflow/reports/0513T001-business.md
- task_plan.md
- progress.md
- findings.md

action：
- 基于 `0512T008` 结论制定 `MarketView provenance / top5 audit transparency` 实现计划。
- 明确本轮不实现代码，不修改 `hbt.depth(0)` 底层 API，不运行 replay/sweep，不启动 live。
- 把原计划中的 T009 调整为 `0513T001`，日期变更后作为 planning-only 任务。
- 设计后续实现任务的 scope、non-scope、文件归属、audit fields、测试和验收 gate。

verify：
- `.workflow/tasks/0513T001.md` 存在。
- `.workflow/reports/0513T001-business.md` 存在。
- `python3 .workflow/build_dashboard.py` 待本报告写入后执行。
- `git diff -- examples/binance_tick_mm` 待最终检查，预期无策略代码改动。

done：
- 结论：
  - 需要改透明度，但不建议第一步直接改 `hbt.depth(0)` 底层 API。
  - 后续实现应先在 Binance maker 策略层新增显式 `MarketView / BookViewSnapshot` 包装层。
  - top5 对齐是当前策略 decision-view 透明度的必要门槛，但不充分，不能替代 full L2 / queue / OFI / microprice proof。
  - overlay top5 可以作为 audit replay decision view 对齐手段，但必须带 provenance，不能把 overlay 后对齐解释成 replay depth reconstruction 已修复。

## 后续实现任务建议

建议后续真正实现另建 `0513T002`：

标题：
- MarketView provenance / top5 audit transparency 最小实现

任务性质：
- 实现 + focused tests + 最小 smoke verification。

不授权内容：
- 不改交易策略规则。
- 不改 `hbt.depth(0)` core API。
- 不做 microprice / OFI / queue 研究。
- 不运行 live。
- 不把 overlay 对齐当成 replay book reconstruction proof。

## 0513T002 Scope

### 0. 文件边界和禁止范围

Allowed files：
- `examples/binance_tick_mm/strategy_core.py`
- `examples/binance_tick_mm/live_tick_mm.py`
- `examples/binance_tick_mm/backtest_tick_mm.py`
- `examples/binance_tick_mm/test_backtest_tick_mm.py`

Allowed generated / workflow files：
- `.workflow/tasks/0513T002.md`
- `.workflow/reports/0513T002-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`

Constraints inside allowed files：
- `strategy_core.py`：
  - 可以新增 `MarketView` / `BookViewSnapshot`、`build_market_view_from_depth(...)`、audit row 字段和纯函数支撑。
  - 不能改变 `decide_actions()` 的交易语义、submit/cancel/reduce-side 规则、guard 判定结果、order id 生成规则或默认参数。
- `live_tick_mm.py` / `backtest_tick_mm.py`：
  - 可以把现有 `hbt.depth(0)` 后的 market view 构造切到 shared helper。
  - 可以写入 provenance 字段和 overlay source 字段。
  - 不能修改 fair/target 公式、throttle 逻辑、latency guard、inventory guard、submit/cancel 规则、position limit、risk 参数默认值。
- `test_backtest_tick_mm.py`：
  - 可以增加 focused tests 覆盖 helper、audit schema、overlay provenance。
  - 不能通过放宽断言或删除现有行为检查来掩盖行为变化。

Forbidden files / areas：
- `hftbacktest/`
- `py-hftbacktest/`
- Rust crates / workspace core files
- connector / live deployment scripts
- `examples/binance_tick_mm/config*.toml` 默认策略配置
- `examples/binance_tick_mm/stage6j_replay.py` candidate/replay ranking 逻辑
- live run scripts、AWS/remote deployment state、archive tarballs

Requires separate task：
- 修改 `hbt.depth(0)` 或 core depth API。
- 修改 Binance raw converter / npz schema。
- 修改 connector 本地 order book 管理。
- 增加 `U/u/pu`、`lastUpdateId`、bookTicker provenance，如果当前 Python 策略层拿不到这些字段。
- 实现 microprice / OFI / queue feature study。
- 修改交易规则、fair price 公式、quote placement、risk guard 或 live 参数。

### 1. 策略层 MarketView 包装

目标：
- 在策略层显式定义 decision market view，减少 `hbt.depth(0)` 调用后的隐式字段流。

建议文件归属：
- `examples/binance_tick_mm/strategy_core.py`
- `examples/binance_tick_mm/live_tick_mm.py`
- `examples/binance_tick_mm/backtest_tick_mm.py`
- `examples/binance_tick_mm/test_backtest_tick_mm.py`

计划：
- 新增轻量数据结构，命名可为 `MarketView` 或 `BookViewSnapshot`。
- 新增 helper：`build_market_view_from_depth(depth, *, source, ts_local, ts_exch, feed_latency_ns, overlay_source="")`。
- helper 负责：
  - 读取 `best_bid`、`best_ask`、`best_bid_tick`、`best_ask_tick`。
  - 计算 `mid`、`spread`。
  - 计算 top5 ticks/qtys 和 `bid_size` / `ask_size`。
  - 写入 source/provenance 字段。
  - 对 invalid book 返回清晰状态或沿用现有 skip 逻辑。
- live/backtest 外层都使用该 helper；`decide_actions()` 保持不接收 full depth。

验收重点：
- helper 输出与现有 `compute_top5_size()` / `format_top5_levels()` 在 mock depth 上一致。
- live/backtest 行为不因 helper 抽取而变化。

### 2. Audit provenance 字段

目标：
- 让每个 decision row 说明 market view 和 top5 的来源。

建议新增字段：
- `market_view_source`: `live_depth` / `replay_depth` / `audit_overlay`
- `top5_source`: `live_depth` / `replay_depth` / `audit_overlay`
- `market_overlay_source`
- `top5_overlay_source`
- `book_view_ts_local`
- `book_view_ts_exch`
- `book_view_feed_latency_ns`
- `book_view_stale_ms`
- `top5_depth_best_bid_tick`
- `top5_depth_best_ask_tick`

条件字段：
- `depth_update_u`
- `depth_update_pu`
- `last_update_id`
- `bookticker_bid`
- `bookticker_ask`

说明：
- 条件字段只有在当前 Python/live/backtest 层能可靠拿到时才实现。
- 如果 hftbacktest depth object 不暴露 update ids 或 bookTicker provenance，0513T002 应把它们写成明确 blocker，并建议另建 core/data task，不应伪造字段。

验收重点：
- audit schema 测试覆盖新增字段。
- 默认值稳定，旧 audit 读取逻辑不崩。
- provenance 字段在 live、normal backtest、audit replay、Stage 6J no-overlay 中语义清楚。

### 3. Overlay 语义显式化

目标：
- 消除 T008 发现的半透明状态：compressed market/fair/target 被 overlay，但 top5 仍来自 replay depth。

可选设计 A：top5 跟随 market overlay
- 当 `market_state_overlay=audit` 时，top5 ticks/qtys 也从 live audit overlay。
- `market_view_source=audit_overlay`
- `top5_source=audit_overlay`
- 优点：audit replay decision view 完整对齐，便于 action-path alignment。
- 风险：容易被误读为 replay depth 本身已对齐；必须在报告和字段里标明 overlay。

可选设计 B：top5 不 overlay，但显式标源
- 保持当前 top5 来自 replay depth。
- `market_view_source=audit_overlay`
- `top5_source=replay_depth`
- 优点：暴露 replay depth 与 live top5 差异。
- 风险：audit row 中 compressed market 与 top5 来源不同，使用者必须理解 mixed source。

推荐：
- 0513T002 优先实现 B 的显式标源，除非验收要求 audit replay top5 完整对齐。
- 若实现 A，必须同时保留 no-overlay comparison 工具，防止隐藏 replay reconstruction mismatch。

验收重点：
- focused test 必须证明 overlay 时 `market_view_source` 与 `top5_source` 不会混淆。
- 报告必须明确 overlay top5 只是 decision-view 对齐，不是 replay book reconstruction proof。

### 4. 最小验证矩阵

必须跑：
- `python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py -k "top5 or market_state_overlay or build_audit_row"`
- `python examples/binance_tick_mm/maker_acceptance.py --help`
- `python examples/binance_tick_mm/align_live_run.py --help`
- `python3 .workflow/build_dashboard.py`

建议跑：
- 用一个小样本或现有 audit replay 输出做 no-trade / no-live smoke。
- 比较改动前后 baseline action/planned/reject/throttle 行为，不应改变。

禁止：
- 不跑新的 Stage 6J candidate replay。
- 不跑 live。
- 不改策略参数默认值。

## QA 验收方案

QA 应检查：
1. 是否只实现 transparency / provenance，不改变交易策略决策。
2. 新增 helper 是否覆盖 live/backtest 共同路径，避免继续扩散隐式 `hbt.depth(0)` 字段处理。
3. audit provenance 字段是否能解释：
   - live depth
   - replay depth
   - audit overlay
   - top5 是否 overlay
4. top5 对齐结论是否被正确限制为 decision-view/audit-view 层，不被写成 full L2 proof。
5. tests 是否覆盖 helper、audit schema、overlay provenance。
6. `git diff -- examples/binance_tick_mm` 是否只包含透明化和测试改动，不包含策略规则改动。

## 何时才考虑改 `hbt.depth(0)` 底层 API

只有在 0513T002 证明 Python 策略层无法取得必要 provenance 时，才另建 core/data task。触发条件包括：
- depth object 无法暴露当前 book 的 update id / lastUpdateId。
- replay pipeline 无法把 Binance `U/u/pu` 映射到 decision row。
- bookTicker 与 depth 的一致性无法在策略层记录。
- top-N snapshot 无法通过现有 depth object 稳定生成或验证。

在这些触发条件出现前，不应先动 core API。

blockers：
- 无规划执行阻塞。
- 实现阻塞待 0513T002 处理：当前 Python 策略层是否能拿到 depth update id / bookTicker provenance 尚不确定。

commit：
- 无

提交信息：
- 无
```
