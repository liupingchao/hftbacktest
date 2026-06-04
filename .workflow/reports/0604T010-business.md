```md
执行线程：
- 业务线程-research

任务ID：
- 0604T010

状态：
- 待验收

是否进行QA验收：
- 否

QA说明：
- 当前任务结果暂不进入QA验收，待总控确认后再决定是否派发QA验收。

files：
- `.workflow/tasks/0604T010.md`
- `.workflow/reports/0604T010-business.md`
- 只读检查：
  - `examples/binance_tick_mm/strategy_core.py`
  - `examples/binance_tick_mm/live_tick_mm.py`
  - `examples/binance_tick_mm/test_backtest_tick_mm.py`
  - `examples/binance_tick_mm/audit_schema.py`
  - `examples/binance_tick_mm/compare_audit.py`

action：
- 已按 case matrix 检查 same-count open-order drift 漏检问题。
- 已用一次性 Python 片段批量复现主要 case；未落盘复现脚本，未修改策略代码，未新增/修改测试。
- 已定位共同根因和 `open_order_diff()` 表达能力边界。

结论：
- review 属实。
- 只要 `rest_open_order_count == local_open_order_count`，当前 `evaluate_live_safety()` 不会进入 open-order mismatch 分支，即使 `open_order_diff` 非空，也会返回 `safety_status=ok`。
- 这会导致同数量但 side/price/qty 内容不同的订单漂移被 live safety 静默放过。

关键代码事实：
- `evaluate_live_safety()` 的 open-order mismatch gate 是 `cfg.open_order_check and rest_open_order_count != local_open_order_count`，见 `examples/binance_tick_mm/strategy_core.py:260`。
- `open_order_diff` 只在数量不一致分支里被写入 `safety_detail`，见 `examples/binance_tick_mm/strategy_core.py:273`；数量相同时走 `status = "ok"`，见 `examples/binance_tick_mm/strategy_core.py:274-275`。
- `open_order_diff()` 使用 quote key 集合比较：`side:price_tick:normalized_qty`，见 `examples/binance_tick_mm/strategy_core.py:749-774`。
- live 主循环先计算 `open_order_diff_value`，再调用 `evaluate_live_safety()`，见 `examples/binance_tick_mm/live_tick_mm.py:399-414`。
- `open_order_mismatch_count` 只在 `safety_status` 为 `open_order_mismatch_pending/open_order_mismatch` 时累计；同数量 drift 返回 `ok` 时会重置/不累计，见 `examples/binance_tick_mm/live_tick_mm.py:416-419`。
- `fail_on_mismatch` 对 `ok` 不触发停止路径，见 `examples/binance_tick_mm/live_tick_mm.py:455-461`。
- audit schema 会记录 `open_order_diff` / `safety_status`，见 `examples/binance_tick_mm/audit_schema.py:119-127`；compare 侧也能把非空 `open_order_diff` 视为 divergence，见 `examples/binance_tick_mm/compare_audit.py:596-601`。因此这是“可被 audit 观察到，但不被 live safety 阻断”的 silent failure。

case matrix：

| case | count | diff 非空 | safety_status | mismatch count 累计 | fail_on_mismatch | 判定 |
|---|---:|---:|---|---:|---:|---|
| single_side_drift_bid_vs_ask | 1/1 | 是 | ok | 否 | 否 | 漏检 |
| single_price_tick_drift | 1/1 | 是 | ok | 否 | 否 | 漏检 |
| single_qty_drift | 1/1 | 是 | ok | 否 | 否 | 漏检 |
| single_price_and_qty_drift | 1/1 | 是 | ok | 否 | 否 | 漏检 |
| two_orders_one_leg_price_drift | 2/2 | 是 | ok | 否 | 否 | 漏检 |
| two_orders_same_count_disjoint_set | 2/2 | 是 | ok | 否 | 否 | 漏检 |
| single_random_client_quote_key_different | 1/1 | 是 | ok | 否 | 否 | 漏检 |
| boundary_quote_key_same_id_different_expected_no_diff | 1/1 | 否 | ok | 否 | 否 | 非漏检；正常 random/id 差异边界 |
| boundary_status_exec_tif_different_same_quote_key | 1/1 | 否 | ok | 否 | 否 | `open_order_diff()` 表达能力边界 |

复现输出摘要：
- `single_side_drift_bid_vs_ask`: `local_only=buy:770001:0.001;rest_only=sell:770001:0.001`，返回 `ok`。
- `single_price_tick_drift`: `local_only=buy:770001:0.001;rest_only=buy:770002:0.001`，返回 `ok`。
- `single_qty_drift`: `local_only=buy:770001:0.001;rest_only=buy:770001:0.002`，返回 `ok`。
- `two_orders_same_count_disjoint_set`: local/rest 两边都是 2 张但集合完全不同，返回 `ok`。
- `boundary_status_exec_tif_different_same_quote_key`: status/exec/tif/cancel-request 等元数据差异不进入当前 quote key，因此 `open_order_diff` 为空，返回 `ok`；这不是 safety 消费 diff 的漏检，而是 diff 表达能力边界。

共同根本原因：
- `evaluate_live_safety()` 把 open-order mismatch 的判定入口绑定到“数量不一致”，没有把 `open_order_diff` 非空作为 mismatch 条件。
- 因为 status 返回 `ok`，live 主循环不会累计 `open_order_mismatch_count`，confirmation 机制失效。
- 因为 status 返回 `ok`，`fail_on_mismatch` 不会触发停止路径。
- `open_order_diff` 被写入 `LiveSafetyState.open_order_diff`，audit/lifecycle row 可以保留证据，但该证据没有参与 live safety 状态机。

分支根因：
- 已能识别但未被 safety 消费：
  - side 不同。
  - price tick 不同。
  - qty 不同。
  - 单边或双边集合不同。
  - random client id 场景下 quote key 不同。
- `open_order_diff()` 自身暂不能表达：
  - status / timeInForce / executedQty / cancel requested / cancellable / update timestamp 等元数据差异。
  - 当前 diff key 是 `side:price_tick:qty`，这些字段不参与 key。后续是否要把这些差异纳入 live safety，需要单独定义业务语义，避免把正常部分成交/REST 状态滞后误判为硬 mismatch。

测试覆盖缺口：
- 现有 `open_order_diff` 测试覆盖了 quote key 相同但 id 不同不误报，以及 local-only 订单 diff。
- 现有 `evaluate_live_safety` 测试只覆盖了数量不一致时保留 open-order detail 并进入 mismatch。
- 缺少 count 相同但 `open_order_diff` 非空时必须进入 `open_order_mismatch_pending/open_order_mismatch` 的回归测试。
- 缺少 live 主循环确认 `open_order_mismatch_count` 对 same-count diff 累计的测试。

建议后续修复任务边界：
- 修复入口应只覆盖 live safety open-order mismatch 判定，不扩大到策略 quote 逻辑、connector、生产配置或 audit schema。
- 建议把 open-order mismatch 条件改为：`cfg.open_order_check and (rest_open_order_count != local_open_order_count or bool(open_order_diff))`，并沿用现有 grace / confirmation / `safety_detail` 流程。
- 回归测试至少覆盖：
  - count 相同、side drift，第二次确认后 `open_order_mismatch`。
  - count 相同、price drift。
  - count 相同、qty drift。
  - 双订单同 count、单腿 drift。
  - 双订单同 count、集合不同。
  - quote key 相同但 client/order id 不同仍不误报。
  - status/exec/tif 元数据差异当前不进入 `open_order_diff` 的边界，除非后续任务明确扩展 diff 语义。

verify：
- `python - <<'PY' ... PY` 一次性批量复现 case matrix。
  - 结果：9 个 case 已覆盖；其中 7 个 `open_order_diff` 非空的 same-count 内容 drift 全部返回 `safety_status=ok`，均不累计 mismatch count，不触发 `fail_on_mismatch`。
- `python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py -k "evaluate_live_safety or open_order_diff"`
  - 结果：`3 passed, 138 deselected`。
- 只读代码检查：
  - `strategy_core.py`
  - `live_tick_mm.py`
  - `test_backtest_tick_mm.py`
  - `audit_schema.py`
  - `compare_audit.py`

done：
- 已定位所有主要 same-count side/price/qty/order-set drift case。
- 已复现这些 case 当前均被判为 `ok`。
- 已确认共同根因是 `evaluate_live_safety()` 只用 count mismatch 作为 open-order safety gate，未消费非空 `open_order_diff`。
- 已区分 `open_order_diff` 已能识别但 safety 未消费的漏检，与 `open_order_diff()` 自身不能表达的元数据边界。

blockers：
- 无。

commit：
- 无

提交信息：
- 无
```
