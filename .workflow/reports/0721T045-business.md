# 0721T045 Business Report

执行线程：
- 总控 auto-loop / 业务执行线程

任务ID：
- 0721T045

状态：
- 待验收

更新时间：
- 2026-07-21 22:55:36 UTC

是否进行QA验收：
- 是

QA说明：
- 本任务完全offline-only；未执行live、private/account、network、remote、service、order或cancel操作。
- T044 immutable evidence未修改、未重新seal；所有新acceptance输出写入`/tmp`。

implementation：
- Commit：`3fcb883a0557c507648908fbcfddf5946d35ffbb`
- 提交信息：`Repair manager batch freshness verifier`
- 修改：
  - `cross_exchange_t024_same_window_acceptance.py`
  - `test_cross_exchange_t024_same_window_acceptance.py`
  - `test_hyperliquid_tiny_live_m2_event_driven_watcher.py`

action：
- 保留原有`(event_sequence, attempt)` exact freshness join为首选。
- 新增canonical manager batch识别：同一event必须恰好两个已提交attempt、连续attempt id、唯一buy/sell、task/window/attempt key完整且一致。
- 只对`canonical_primary_outcome_required()`的strict路径传入manager-enabled事实；旧`0719T001`兼容路径不启用bridge。
- 同一event只有一条freshness row、且该row属于batch首个attempt、状态/时间顺序/三字段projection有效时，才允许第二侧复用该row。
- duplicate、missing、cross-event、wrong task/window、wrong side、non-submitted、non-consecutive identity、empty或mismatched projection均fail-closed。
- 提取并复用统一的raw bool和public-state projection helper，避免producer/rebuild字段语义漂移。

focused verify：
- `test_manager_batch_attempt_bridge_requires_exact_two_sided_identity`：
  `1 passed`；覆盖valid pair、extra row、non-submitted、duplicate side、wrong task和wrong window。
- `test_task7_explicit_manager_mode_uses_two_sided_path`：
  `1 passed`；覆盖strict `0721T038` positive path和duplicate freshness、manager-disabled、projection mismatch、identity mismatch、duplicate side、cross-event hostile。
- `python -m pytest examples/hyperliquid/test_cross_exchange_t024_same_window_acceptance.py -q`：
  `244 passed`。
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py -q`：
  `133 passed`。
- `python -m py_compile ...`和`git diff --check`：通过。

immutable T044 replay：
- Command使用exact task/source/remote-root和`1800s` expectation，结果exit `0`。
- Final recommendation：
  `principal_task12_mechanism_and_evidence_integrity_passed`。
- Decision `43/43 pass`；lifecycle `78/78 pass`；config `72/72 pass`；
  provenance `113/113 pass`；economics `6/6 pass`；optimism `6/6 pass`。
- Independent decision summary：
  - `validation_reasons=[]`
  - candidate evaluations `1896`
  - manager attempt identities `2`
  - submitted attempts `2`
  - exposure/censor/quarantine remains `3/1/0`
- Live facts unchanged：submissions `2`、fills `0`、final owned open orders `0`、
  BTC position `0.0`、estimated loss `0.0 USDC`。
- Acceptance boundary仍为`offline_only=true`；network/private/order/cancel/remote/new-live/credentials-read均为false。
- T044 source snapshot、runtime source、terminal checksum、account proof和estimator replay未被重写。

regression：
- `python -m pytest examples/hyperliquid -q`：
  `1214 passed in 51.34s`。
- Historical rollout compatibility tests remain green；旧路径没有被manager bridge强制要求freshness matrix。

done：
- T044 same-window verifier false-negative已修复，并在同一immutable evidence上通过。
- 修复没有合成fill、PnL、queue priority或稳定经济性证据。
- Dynamic spread、fill feedback、inventory skew、multi-level和actual quote behavior change仍未启用。
- 本任务没有新增live window。

blockers：
- 独立QA验收。
- Principal Task 8后续bounded dynamic-spread live仍需单独formal task；本任务只关闭observe-only same-window evidence gate。

提交信息：
- Implementation：`3fcb883a0557c507648908fbcfddf5946d35ffbb`
- Business/report：待提交。
