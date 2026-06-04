```md
执行线程：
- 业务线程-python

任务ID：
- 0604T011

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0604T011.md`
- `.workflow/reports/0604T011-business.md`
- `examples/binance_tick_mm/strategy_core.py`
- `examples/binance_tick_mm/test_backtest_tick_mm.py`

action：
- 修复 `evaluate_live_safety()` open-order mismatch gate：
  - 原逻辑只在 `rest_open_order_count != local_open_order_count` 时进入 open-order mismatch。
  - 新逻辑在数量不一致或 `open_order_diff` 非空时进入同一套 open-order mismatch / grace / confirmation 流程。
- 补充 `0604T010` same-count open-order drift case matrix 的 focused regression tests：
  - 7 个 `open_order_diff` 非空 case 覆盖 pending 与 confirmed mismatch。
  - 2 个边界 case 保持 `ok`，不扩大 `open_order_diff()` 元数据语义。
- 未修改策略 quote/order 行为、connector、生产配置、audit schema、compare 语义或 `open_order_diff()` key 语义。

case matrix verification：
- `single_side_drift_bid_vs_ask`: pending=`open_order_mismatch_pending`, confirmed=`open_order_mismatch`
- `single_price_tick_drift`: pending=`open_order_mismatch_pending`, confirmed=`open_order_mismatch`
- `single_qty_drift`: pending=`open_order_mismatch_pending`, confirmed=`open_order_mismatch`
- `single_price_and_qty_drift`: pending=`open_order_mismatch_pending`, confirmed=`open_order_mismatch`
- `two_orders_one_leg_price_drift`: pending=`open_order_mismatch_pending`, confirmed=`open_order_mismatch`
- `two_orders_same_count_disjoint_set`: pending=`open_order_mismatch_pending`, confirmed=`open_order_mismatch`
- `single_random_client_quote_key_different`: pending=`open_order_mismatch_pending`, confirmed=`open_order_mismatch`
- `boundary_quote_key_same_id_different_expected_no_diff`: pending=`ok`, confirmed=`ok`
- `boundary_status_exec_tif_different_same_quote_key`: pending=`ok`, confirmed=`ok`

acceptance notes：
- `0604T010` 中 7 个 `open_order_diff` 非空的 same-count drift case 不再返回 `ok`。
- confirmation 逻辑已覆盖：首次为 `open_order_mismatch_pending`，确认次数达到阈值后为 `open_order_mismatch`。
- `safety_detail` 保留对应非空 `open_order_diff`。
- live 主循环无需改动：它已经在 `open_order_mismatch_pending/open_order_mismatch` 时累计 `open_order_mismatch_count`，本修复让 same-count diff 能进入这些状态。
- quote key 相同但 id/client id 不同仍不误报。
- status/exec/tif/cancel-request 等元数据差异仍保持 `0604T010` 记录的 diff 表达能力边界，未在本任务扩展。

verify：
- `python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py -k "evaluate_live_safety or open_order_diff"`
  - 结果：`19 passed, 138 deselected`
- 一次性 Python 片段复跑 `0604T010` case matrix：
  - 结果：前 7 个 same-count diff case 均从旧行为 `ok` 变为 pending/confirmed mismatch；后 2 个边界 case 仍为 `ok/ok`。
- `python -m pytest examples/binance_tick_mm/test_*.py`
  - 结果：`272 passed`
- `git diff --check`
  - 结果：通过。
- `python -m pytest examples/binance_tick_mm`
  - 结果：阻塞于环境/收集阶段，不是本任务断言失败。
  - 具体错误：collect `examples/binance_tick_mm/run_env_test.py` 时 import `hftbacktest.data.utils.tardis`，numba cache 报 `RuntimeError: cannot cache function '_convert_depth': no locator available for file '/home/molly/anaconda3/lib/python3.13/site-packages/hftbacktest/data/utils/tardis.py'`。
  - 已补跑常规 `test_*.py` 测试集合并通过 `272 passed`。

done：
- bug 已修复：same-count side/price/qty/order-set drift 不再被 `evaluate_live_safety()` 判为 `ok`。
- `0604T010` 的复现 case 已转为回归测试和一次性矩阵验证。
- 本任务未扩大 `open_order_diff()` 元数据语义，未改 live 配置或策略行为。

blockers：
- `python -m pytest examples/binance_tick_mm` 全目录命令受当前 Python/numba cache 环境问题阻塞；常规 `test_*.py` 回归已通过。

commit：
- 待提交

提交信息：
- 待提交
```
