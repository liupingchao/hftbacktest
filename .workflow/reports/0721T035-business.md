# 0721T035 Business Report

执行线程：
- 总控 auto-loop / 业务执行线程

任务ID：
- 0721T035

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py`
- `examples/hyperliquid/cross_exchange_t024_same_window_acceptance.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_attribution.py`
- `examples/hyperliquid/test_cross_exchange_t024_same_window_acceptance.py`

action：
- Producer audit 和 Task 12 independent helper 在 attempt id、method schema、canonical coverage 和 valid-count normalization 之前扫描所有 raw attempt/canonical result rows。
- Top-level `method=historical_orders` 与 nested `result.status=historical_orders` 均独立触发 delayed-history protocol requirement。
- Direct method 携带 historical result，或 historical method 携带 non-historical/missing result，均显式产生 `terminal_audit_query_method_result_mismatch`。
- Malformed historical attempt id 即使不会进入 valid historical count，也仍要求 exact marker/timing、post-history final snapshot 和 history call timing evidence。
- Canonical-only nested historical status 在 attempt rows 正常或完全缺失时都不能被归一化掉。
- Non-dict attempt/result rows 保持结构性 fail closed，但不会凭空产生 historical semantics。
- Exact production historical contract 和 consistent legacy direct-only v4 contract 保持通过。
- Live executor、endpoint timing、quote、risk、size、submission、side-set 和所有 adaptive activation 未改变。

verify：
- Two-file producer/acceptance regression：`414 passed in 8.87s`。
- Combined four-file focused regression：`537 passed in 27.78s`。
- Full Hyperliquid regression：`1040 passed in 47.97s`。
- Python compile、`git diff --check` 通过。
- T031 estimator replay exit `1`：persisted/rebuilt exposure `6/6`，censor `0/2`，quarantine `1/0`，`snapshot_match=false`。
- T031 acceptance exit `2`：provenance `112/112`、config `72/72`、decision `43/43`、lifecycle/evidence `55 pass / 23 fail`、economics `6/6`。
- T026 exact acceptance exit `0`，lifecycle/evidence `78/78 pass`。
- T016/T022 exact acceptance 均 exit `2`，lifecycle/evidence 均为 `66 pass / 12 fail`。
- 全程 offline；未触发 live、private/account、order、cancel、network、remote 或 service 操作。

done：
- T034 QA 的 nested historical-result 和 malformed historical-attempt P1 均在 producer 与 independent helper 中一致 fail closed。
- Task 12 T035+ 对 structurally valid but semantically forged evidence 通过 independent rebuild comparison 阻断。
- 修复 source 已准备进入独立 QA；通过前不允许新 private read、live 或 adaptive/multi-level activation。

blockers：
- 独立 QA acceptance。

commit：
- 180d24bd520e68487cdf843385b3940f426b0dc8

提交信息：
- Repair raw historical semantics trigger
