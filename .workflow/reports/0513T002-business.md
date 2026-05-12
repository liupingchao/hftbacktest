```md
执行线程：
- 业务线程-python

任务ID：
- 0513T002

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 正常验收。重点检查实现是否严格遵守 `0513T001` 文件边界，是否只做 strategy-layer MarketView provenance / top5 audit transparency，是否没有改变交易语义、fair/target 公式、risk guards、configs、core API、converter/npz schema、replay candidates 或 live 状态。

files：
- .workflow/tasks/0513T002.md
- .workflow/reports/0513T002-business.md
- examples/binance_tick_mm/strategy_core.py
- examples/binance_tick_mm/live_tick_mm.py
- examples/binance_tick_mm/backtest_tick_mm.py
- examples/binance_tick_mm/test_backtest_tick_mm.py
- task_plan.md
- progress.md
- findings.md

action：
- 在 `strategy_core.py` 新增 `MarketView` dataclass 和 `build_market_view_from_depth(...)` helper。
- 在 `strategy_core.py` 注册新增 audit provenance 字段到共享 `AUDIT_FIELDS`，没有修改 `audit_schema.py` 文件。
- live loop 改为通过 `build_market_view_from_depth(..., source="live_depth")` 构造 best bid/ask、mid、spread、top5 ticks/qtys、top5 sizes。
- backtest loop 改为通过 `build_market_view_from_depth(..., source="replay_depth")` 构造 replay depth view。
- audit replay overlay 保持 T001 推荐 B：compressed market state overlay 后写 `market_view_source=audit_overlay` / `market_overlay_source=audit`，但 top5 仍保留 `top5_source=replay_depth`，不伪装成 replay book 已对齐。
- `build_audit_row(...)` 写入 provenance 字段：
  - `market_view_source`
  - `top5_source`
  - `market_overlay_source`
  - `top5_overlay_source`
  - `book_view_ts_local`
  - `book_view_ts_exch`
  - `book_view_feed_latency_ns`
  - `book_view_stale_ms`
  - `top5_depth_best_bid_tick`
  - `top5_depth_best_ask_tick`
- 增加 focused tests 覆盖 audit schema 字段、MarketView helper 输出和 build-audit-row mixed provenance。

verify：
- `python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py -k "top5 or market_state_overlay or build_audit_row or market_view"` -> 7 passed, 121 deselected。
- `python examples/binance_tick_mm/maker_acceptance.py --help` -> exit 0。
- `python examples/binance_tick_mm/align_live_run.py --help` -> exit 0。
- `python -m py_compile examples/binance_tick_mm/strategy_core.py examples/binance_tick_mm/live_tick_mm.py examples/binance_tick_mm/backtest_tick_mm.py examples/binance_tick_mm/test_backtest_tick_mm.py` -> exit 0。
- `git diff --name-only` -> only allowed implementation files:
  - `examples/binance_tick_mm/backtest_tick_mm.py`
  - `examples/binance_tick_mm/live_tick_mm.py`
  - `examples/binance_tick_mm/strategy_core.py`
  - `examples/binance_tick_mm/test_backtest_tick_mm.py`
- `git diff -- examples/binance_tick_mm` reviewed: no fair/target formula change, no submit/cancel decision semantic change, no risk guard/default config change.

done：
- T002 完成了 T001 允许范围内的最小实现。
- 新 audit provenance 可以区分 live depth、replay depth、audit overlay，以及 overlay 后 top5 仍来自 replay depth 的 mixed-source 状态。
- 未实现 `U/u/pu`、`lastUpdateId`、bookTicker provenance；原因是当前 Python strategy/backtest 层不能可靠拿到这些字段。若需要这些字段，应另建 core/data task。
- 未修改 `hbt.depth(0)` core API。
- 未修改 `audit_schema.py`、`hftbacktest/`、`py-hftbacktest/`、Rust crates、connector、configs、`stage6j_replay.py`、live scripts、AWS 状态或 archive tarballs。
- 未运行 live，未运行新的 Stage 6J replay/sweep。
- 未改变交易规则、fair price 公式、quote placement、risk guards、submit/cancel/reduce-side 规则或默认参数。

blockers：
- 无执行阻塞。
- 后续如要记录 Binance update ids / bookTicker provenance，需要另建 core/data task。

commit：
- 无

提交信息：
- 无
```
