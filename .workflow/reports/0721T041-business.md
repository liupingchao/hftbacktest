# 0721T041 Business Report

执行线程：
- 总控 auto-loop / 业务执行线程

任务ID：
- 0721T041

状态：
- 待验收

更新时间：
- 2026-07-21 22:45:58 CST

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/cross_exchange_delayed_history_probe_acceptance.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `examples/hyperliquid/test_cross_exchange_delayed_history_probe_acceptance.py`

action：
- Producer direct result改为exact object `{"status":"unknownOid"}`；`error`、`order`、`orders`及任何未知字段均在history调用前fail closed。
- Verifier direct rows要求strict integer `attempt=1`、canonical/alias target exact等于deterministic synthetic token，并使用同一exact result allowlist。
- History verifier要求strict integer `attempt=1`、query target canonical/alias token exact、result top-level keys exact为`status/orders`。
- `historical_row_classifications`不再是信任根；verifier从每条raw history result row独立重建`exact_synthetic/foreign/malformed`并与producer字段exact比较。
- Non-dict、invalid status/reference、conflicting identity、synthetic exact match和classification forgery均写出blocked manifest，不抛出未处理异常。
- Producer在持久化前移除空identity alias，使有效OID-only、`cloid=None`和`cloid=""`保持可验证foreign semantics。
- Producer拒绝raw endpoint中的redaction marker、token/alias/conflict/invalid evidence字段，以及history row外层任何第二oid/cloid identity surface。
- Verifier独立拒绝producer不可能生成的外层identity、marker存在、无有效alias的null/empty token/alias-token及其他nested evidence形态。
- T040 QA六个hostile artifacts全部纳入回归；immutable T040 live artifact未修改。

verify：
- Acceptance focused tests：`54 passed`。
- Producer delayed-history focused tests：`19 passed`。
- Combined verifier/watcher/orchestrator regression：`223 passed in 22.78s`。
- Full Hyperliquid regression：`1197 passed in 51.45s`。
- Python compile、`git diff --check`、cached diff check通过。
- Immutable T040 live artifact：acceptance exit `0`，`14/14 pass`，`blocking_checks=[]`。
- T040 QA hostile artifacts：
  - `direct_extra_orders`、`direct_error`：blocked by `direct_query_contract`。
  - `foreign_history_target`、`forged_history_classification`：blocked by `history_result_envelope_contract`。
  - `false_attempt`、`conflicting_alias`：blocked；不再获得empty blocking checks。
- Independent hostile review经历三轮反例修复后最终结论：无P0/P1/P2 findings。
- T037 exact replay：exit `2`，decision `39/4`、lifecycle `78/0`、blocked、offline-only。
- T031 exact replay：exit `2`，decision `43/0`、lifecycle `55/23`、blocked、offline-only。
- T026 exact `1800s` replay：exit `0`，decision `43/0`、lifecycle `78/0`、pass、offline-only。
- T022/T016 exact replay：exit `2`，decision `43/0`、lifecycle `66/12`、blocked、offline-only。
- 本任务未执行live、private/account、order、cancel、network、remote或service操作。

done：
- T040 QA的direct exact envelope和history metadata binding两个P1均由producer、independent verifier和hostile regression关闭。
- Immutable T040 endpoint evidence无需第二live window即可由修复后的verifier接受。
- Delayed-history observe-only mechanism evidence已准备进入独立QA；只有QA通过后才能关闭Task 8 evidence gate。

blockers：
- 独立QA acceptance。

commit：
- `2717d1c9a720dc6e8b399091f8008b840204e8eb`

提交信息：
- `Repair delayed history evidence binding`
