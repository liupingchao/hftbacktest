```md
执行线程：
- 业务线程-python

任务ID：
- 0519T006

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0519T006.md`
- `.workflow/reports/0519T006-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`
- `examples/binance_tick_mm/audit_schema.py`
- `examples/binance_tick_mm/strategy_core.py`
- `examples/binance_tick_mm/backtest_tick_mm.py`
- `examples/binance_tick_mm/live_tick_mm.py`
- `examples/binance_tick_mm/test_backtest_tick_mm.py`

action：
- 将 `0519T006` 状态切到执行中后完成 Step 8C default-off quote-update helper / instrumentation implementation。
- 在共享策略层新增 quote-update audit helper，统一记录 intent/action/reason、min move、quote/join/anchor age、latency bucket、throttle/token state、cancel-readd bucket、drop cause、post-only pre/post-check 和 inventory request placeholder。
- 扩展 `AUDIT_FIELDS`，让 live/backtest audit row 都能写出新增字段，并保持缺省值稳定。
- 将 helper 接入 backtest/live audit row 构建路径，但不改变 action、planned_action、reject_reason、throttle_reason 或实际 submit/cancel 行为。
- 增加 focused tests 覆盖新增 schema 字段、helper 稳定输出、默认 row 缺省值、live/backtest helper wiring 和 preflight/schema compatibility。

新增 audit/helper 字段：
- `quote_update_intent`
- `quote_update_action`
- `quote_update_reason`
- `min_move_passed`
- `quote_age_ms`
- `join_age_ms`
- `anchor_age_ms`
- `latency_bucket`
- `throttle_state`
- `token_bucket_state`
- `cancel_readd_bucket`
- `reject_throttle_drop_cause`
- `post_only_pre_check`
- `post_only_post_check`
- `inventory_request_id`

默认行为说明：
- 默认 quote placement / cancel / submit 行为保持不变。
- 现有 throttle/API/latency suppression semantics 保持不变。
- helper 只读取已有 decision/action 状态和快照后写 audit 字段，不作为控制流入口。
- `python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py` 覆盖了共享 helper、schema、新字段缺省值和既有 action/guard 单测；schema/preflight 测试覆盖新增字段兼容性。

placeholder / diagnostic-only：
- `inventory_request_id` 当前是 passive placeholder。
- `quote_age_ms`、`join_age_ms`、`anchor_age_ms`、`latency_bucket`、`cancel_readd_bucket`、`throttle_state`、`token_bucket_state`、`post_only_pre_check`、`post_only_post_check` 是 instrumentation / diagnostic-only，不表示 live readiness 或 promotion。

explicit non-goals：
- no Step 9
- no replay sweep
- no live
- no default-on
- no Step 5C promotion
- no inventory-control implementation

verify：
- `python -m py_compile examples/binance_tick_mm/strategy_core.py examples/binance_tick_mm/backtest_tick_mm.py examples/binance_tick_mm/live_tick_mm.py examples/binance_tick_mm/audit_schema.py examples/binance_tick_mm/test_backtest_tick_mm.py examples/binance_tick_mm/deploy/preflight_live_run.py`
- `python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py`
- `python -m pytest examples/binance_tick_mm/test_validate_audit.py examples/binance_tick_mm/test_deploy_preflight.py`
- `python -m pytest examples/binance_tick_mm/test_quote_anchor_safety.py`
- `python examples/binance_tick_mm/maker_acceptance.py --help`
- `python3 .workflow/build_dashboard.py`
- `git diff --check`

done：
- Step 8C helper / instrumentation implementation 已完成。
- live/backtest 使用同一套 helper / field names。
- audit schema 和 preflight/schema compatibility 已通过 focused verification。
- 默认行为保持不变；本任务没有进入 Step 9、没有 replay sweep、没有 live、没有 default-on、没有 promotion。

blockers：
- 无。

commit：
- 待提交

提交信息：
- 待提交
```
