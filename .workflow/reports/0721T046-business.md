# 0721T046 Business Report

执行线程：
- 总控 auto-loop / 业务执行线程

任务ID：
- 0721T046

状态：
- 待验收

更新时间：
- 2026-07-21 23:14:31 UTC

是否进行QA验收：
- 是

QA说明：
- 本任务完全offline-only；未执行live、private/account、network、remote、service、order或cancel操作。
- T044 immutable evidence未修改、未重新seal；新acceptance输出写入`/tmp`。

implementation：
- Commit：`fca64596892093248345ecafe5337db0c640bddc`
- 提交信息：`Prioritize exact manager freshness joins`
- 修改：
  - `cross_exchange_t024_same_window_acceptance.py`
  - `test_cross_exchange_t024_same_window_acceptance.py`
  - `test_hyperliquid_tiny_live_m2_event_driven_watcher.py`

action：
- 对每个canonical manager batch先检查每个attempt是否已有exact freshness identity。
- 若两个attempt均有exact freshness row，跳过event-level batch bridge cardinality和fallback precondition，继续逐attempt exact semantic/projection validation。
- 仅在至少一个canonical submitted attempt缺少exact row时启用shared freshness bridge。
- Bridge identity现在要求`order_endpoint_called=true`、`order_status_types`非空且不为`skipped`、`window_id`显式存在且exact匹配。
- 保留unique freshness、首attempt、status/time/projection exact checks。
- 增加双exact rows positive、真实duplicate identity、skipped/empty status和missing window hostile regression。

verify：
- `test_manager_batch_attempt_bridge_requires_exact_two_sided_identity`：`1 passed`。
- `test_task7_explicit_manager_mode_uses_two_sided_path`：`1 passed`。
- `python -m pytest examples/hyperliquid/test_cross_exchange_t024_same_window_acceptance.py -q`：`244 passed`。
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py -q`：`133 passed`。
- `python -m pytest examples/hyperliquid -q`：`1214 passed in 51.16s`。
- `py_compile`和`git diff --check`：通过。

immutable T044 replay：
- Exact task/source/remote-root和`1800s` expectation，exit `0`。
- Final recommendation：`principal_task12_mechanism_and_evidence_integrity_passed`。
- Decision `43/43 pass`；lifecycle `78/78 pass`；config `72/72 pass`；provenance `113/113 pass`；economics `6/6 pass`；optimism `6/6 pass`。
- Independent decision summary `validation_reasons=[]`；manager attempt identities `2`；submitted attempts `2`；candidate evaluations `1896`。
- Live facts unchanged：submissions `2`、fills `0`、final owned open orders `0`、BTC position `0.0`、estimated loss `0.0 USDC`。
- Acceptance boundary为`offline_only=true`；network/private/order/cancel/remote/new-live/credentials-read均为false。
- T044 source snapshot、runtime source、terminal checksum、account proof和estimator replay未被重写。

done：
- T045 QA指出的exact-join优先级和manager submitted identity两个缺陷已修复。
- Single shared-row和双exact-row两种合法freshness合同均可通过；malformed/ambiguous evidence继续fail-closed。
- Dynamic spread、fill feedback、inventory skew、multi-level和actual quote behavior change仍未启用。
- 本任务没有新增live window。

blockers：
- 独立QA验收。
- QA通过前不得进入bounded dynamic-spread live。

提交信息：
- Implementation：`fca64596892093248345ecafe5337db0c640bddc`
- Business/report：待提交。
