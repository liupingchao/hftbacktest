# 0721T042 Business Report

执行线程：
- 总控 auto-loop / 业务执行线程

任务ID：
- 0721T042

状态：
- 待验收

更新时间：
- 2026-07-21 23:14:18 CST

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/cross_exchange_delayed_history_probe_acceptance.py`
- `examples/hyperliquid/test_cross_exchange_delayed_history_probe_acceptance.py`

action：
- `persisted_history_order_evidence_issues()`现在对全部六种nested identity aliases执行exact representation检查。
- 任一present `oid/orderId/order_id/cloid/clientOrderId/client_order_id`必须exact等于`<redacted>`；raw numeric/string、null、empty和partial marker均fail closed。
- Representation通过后才允许`historical_reference_tokens`重建opaque token semantics。
- Existing marker presence、token-without-alias、outer identity和classification contracts保持严格。
- 新增canonical raw OID、raw cloid、raw dual identity及null/empty/partial marker regressions。
- 新增四种alternative alias的raw+matching-token blocked regressions。
- 新增四种alternative alias及dual identity经`redact_with_reference_tokens`后的positive round-trip regressions。

verify：
- Acceptance focused tests：`69 passed`。
- Combined verifier/watcher/orchestrator regression：`238 passed in 22.73s`。
- Full Hyperliquid regression：`1212 passed in 51.11s`。
- Python compile、`git diff --check`、cached diff check通过。
- T041 QA exact `raw_nested_identity` artifact：exit `2`，blocked by `history_result_envelope_contract`和`history_unknown_contract`。
- Immutable T040 live artifact：`14/14 pass`。
- 六个T040 QA hostile artifacts全部blocked。
- Independent hostile review最终无P0/P1/P2 findings。
- T037：decision `39/4`、lifecycle `78/0`、blocked、offline-only。
- T031：decision `43/0`、lifecycle `55/23`、blocked、offline-only。
- T026 exact `1800s`：decision `43/0`、lifecycle `78/0`、pass、offline-only。
- T022/T016：decision `43/0`、lifecycle `66/12`、blocked、offline-only。
- 本任务未执行live、private、account、order、cancel、network、remote或service操作。

done：
- T041 QA的raw nested identity representation P1和替代alias测试P2均关闭。
- Persisted nested identity现在同时要求exact `<redacted>` representation和exact token/alias semantics。
- Immutable T040 evidence无需修改或第二live window即可通过。

blockers：
- 独立QA acceptance。

commit：
- `21e0d2b120f100981aa7bd48e9c7bea2942bfff1`

提交信息：
- `Enforce persisted history identity redaction`
