# 线程回报

执行线程：
- 业务线程-python

任务ID：
- 0604T014

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0604T014.md`
- `.workflow/reports/0604T014-business.md`
- `examples/binance_tick_mm/test_backtest_tick_mm.py`

action：
- 只读检查 `examples/binance_tick_mm/live_tick_mm.py` 和 `examples/binance_tick_mm/strategy_core.py`，确认 review 指向的 live safety 状态迁移和 pause/fatal 条件。
- 在 `examples/binance_tick_mm/test_backtest_tick_mm.py` 新增 focused 诊断测试，不修改 production 行为。
- 新增测试 helper 复刻当前 live 代码的 fatal break 条件和 position mismatch pause 条件，用于稳定复现当前实际语义。

review 判断：
- `[P2] confirmed position mismatch 在 non-fatal 模式下仍可能继续交易` 为事实。

根本原因：
- `strategy_core.py` 的 `evaluate_live_safety()` 会在仓位 mismatch 达到 `position_mismatch_confirmations` 后把状态从 `position_mismatch_pending` 推进到 `position_mismatch`。
- `live_tick_mm.py` 的 fatal path 在 `fail_on_mismatch=true` 时会对 confirmed `position_mismatch` break。
- 但 non-fatal pause path 只检查 `safety_state.safety_status == "position_mismatch_pending"`，没有包含 confirmed `position_mismatch`。
- 因此 `fail_on_mismatch=false` 且 `position_mismatch_pause_trading=true` 时，pending 会暂停，confirmed 反而会进入 `decide_actions(...)`。

case matrix：
- `fail_on_mismatch=true + confirmations=2`：第一次 `position_mismatch_pending` 不 fatal；第二次 `position_mismatch` fatal break。符合当前设计。
- `fail_on_mismatch=false + position_mismatch_pause_trading=true + confirmations=2`：第一次 pending 暂停；第二次 confirmed 不 fatal 且不 pause，会继续交易。复现 review bug。
- `fail_on_mismatch=false + position_mismatch_pause_trading=true + confirmations=1`：第一次直接 `position_mismatch`，没有 pending 阶段，因此当前 pause 条件完全不命中，会继续交易。确认存在直接 confirmed 漏暂停 case。
- `fail_on_mismatch=false + position_mismatch_pause_trading=false + confirmations=2`：pending/confirmed 都不 pause。该行为符合显式关闭 pause 的配置语义。
- mismatch 恢复为 ok：`evaluate_live_safety()` 返回 `ok` / `position_mismatch=0.0`；live loop 的计数重置分支会在非 pending/confirmed 状态下重置。
- `rest_error`：fatal 模式会 break；non-fatal 模式下没有 position pause path。不是同一个 confirmed/pending 漏洞，但语义上 non-fatal 会继续交易，后续可按 live safety 策略单独评估是否需要 REST error pause。
- `open_order_mismatch_pending/open_order_mismatch`：fatal 模式 pending 不 break、confirmed break；non-fatal 模式没有 open-order pause path。它不是 position pause 的同类 confirmed 后漏暂停，因为当前没有定义 open-order pause 开关；若总控希望 non-fatal open-order mismatch 也暂停，需要另开设计任务。

建议修复方案：
- 最小修复：把 `live_tick_mm.py` 的 position mismatch pause 条件从只匹配 `position_mismatch_pending` 改为匹配 `{"position_mismatch_pending", "position_mismatch"}`。
- 回归测试建议：
  - `fail_on_mismatch=false + position_mismatch_pause_trading=true + confirmations=2`：pending 和 confirmed 都应 pause。
  - `fail_on_mismatch=false + position_mismatch_pause_trading=true + confirmations=1`：第一次 confirmed 也应 pause。
  - `fail_on_mismatch=false + position_mismatch_pause_trading=false`：仍明确允许 pending/confirmed 不 pause。
  - `fail_on_mismatch=true`：confirmed 仍 fatal break，不被 pause 逻辑改变。
- 可选后续设计：评估 non-fatal `rest_error` 和 `open_order_mismatch` 是否也需要独立 pause 开关；这不是本轮修复的必要范围。

verify：
- `/home/molly/anaconda3/envs/hftbacktest/bin/python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py -k "live_safety or position_mismatch or safety_pause"`：`24 passed, 150 deselected`
- `/home/molly/anaconda3/envs/hftbacktest/bin/python -m pytest examples/binance_tick_mm/test_*.py`：`289 passed`
- `rg -n "position_mismatch_pause_trading|safety_status == \"position_mismatch_pending\"|open_order_mismatch_pending|open_order_mismatch" examples/binance_tick_mm/live_tick_mm.py`：确认 pause 条件唯一匹配 `position_mismatch_pending`。
- `git diff --check`：通过，无输出。

done：
- 已复现 review bug 和 `confirmations=1` 直接 confirmed 漏暂停 case。
- 已明确根因和建议最小修复。
- 本任务未修改 `live_tick_mm.py`、`strategy_core.py` 或其他 production 行为。

blockers：
- 无。

commit：
- `d478086`

提交信息：
- `diagnose nonfatal position mismatch pause gap`
