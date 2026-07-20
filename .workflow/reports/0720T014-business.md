# 业务线程执行回报

执行线程：
- 总控 auto-loop / 业务执行线程

任务ID：
- 0720T014

状态：
- 待验收

更新时间：
- 2026-07-20 09:16 CST

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0720T014.md`
- `examples/hyperliquid/cross_exchange_t024_same_window_acceptance.py`
- `examples/hyperliquid/test_cross_exchange_t024_same_window_acceptance.py`

action：
- 将 attempt-key suffix 的直接 `int()` 转换替换为现有 `raw_strict_positive_attempt()` bounded parser。
- Suffix 现在与 standalone attempt identity 使用同一契约：
  - 最大位数受限；
  - 数值必须为正；
  - 数值不得超过 `RAW_MAX_CANCEL_REFERENCE_ATTEMPT`。
- 保留 exact task、`window_01`、row window 和 attempt-id equality 检查。
- 新增 synchronized/resealed 回归：
  - 5000 位数字 suffix；
  - `RAW_MAX_CANCEL_REFERENCE_ATTEMPT + 1` suffix。
- 两种攻击均不再抛异常，而是返回 blocked manifest，并包含 `attempt_key_mismatch` validation reason。

verify：
- Exact implementation commit：
  - `33975476d629e1abc1f62324276785ac0908093f`
- Focused acceptance/watcher/orchestrator/manager：
  - `245 passed in 16.96s`
- Full Hyperliquid：
  - `708 passed in 35.56s`
- Modified modules/tests `py_compile`：pass。
- Acceptance CLI `--help`：pass。
- Implementation commit `git show --check`：pass。
- T011 byte-exact offline replay：
  - legacy bridge authorized for exact T011 task/source；
  - provenance `112 pass / 0 fail`；
  - config `70 pass / 1 fail`；
  - decision `16 pass / 27 fail`；
  - lifecycle `34 pass / 27 fail`；
  - independent summary validation reasons `[]`；
  - candidate/manager/submission `21 / 1 / 0`；
  - final recommendation 继续 blocked。
- 未执行 live、private、account、order、cancel、network、remote 或 service。

done：
- 超长和越界 attempt-key suffix 稳定转换为 fail-closed acceptance 结果，不再导致 verifier 异常退出。
- T013 已接受的 identity、authorization、causal join、legacy bridge、source/checksum/fill/cancel/path 行为未改变。
- Strategy formula、threshold、risk cap、activation 和 multi-level 状态未改变。

blockers：
- 当前唯一流程节点是独立 QA 验收 T014。
- T014 QA 通过前不得启动新的 bounded live window。

commit：
- `33975476d629e1abc1f62324276785ac0908093f`

提交信息：
- `Bound attempt key suffix parsing`
