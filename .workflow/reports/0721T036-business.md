# 0721T036 Business Report

执行线程：
- 总控 auto-loop / 业务执行线程

任务ID：
- 0721T036

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
- Producer 与 independent verifier 新增 strict raw method accessor；只有 exact string 才进入 method interpretation，list/dict/nested container 等 hostile JSON 不再参与 set membership。
- Audit attempt loop、terminal reconciliation loop 和 Task 12 structural summary 均使用 total method parsing/type guard。
- Historical envelope coherence 现在要求 result dict、exact `historical_orders` status、orders list，以及每行合法 status、order dict 和无 malformed identity token/alias shape。
- Missing/null/bool/number/string/dict orders、non-dict rows、missing order/status、invalid status、empty identity 和 container identity/alias-token payload 均显式产生 `terminal_audit_query_method_result_mismatch`。
- `orders=[]` 保持结构合法并独立分类为 `unknown`，不会被误写成 terminal proof。
- Valid exact/foreign rows继续由 T024 strict reference classifier 决定 exact/foreign/conflicting/malformed，envelope coherence 不替代 terminal proof。
- Task 12 T036+ container method 和 malformed history envelope 均返回 blocked manifest，不再异常退出。
- Live executor、watcher、manager、orchestrator、endpoint timing、quote、risk、size、submission 和所有 adaptive activation 未改变。

verify：
- Two-file producer/acceptance regression：`444 passed in 9.11s`。
- Combined four-file focused regression：`567 passed in 27.81s`。
- Full Hyperliquid regression：`1070 passed in 48.08s`。
- Python compile、`git diff --check` 通过。
- T031 estimator replay exit `1`：persisted/rebuilt exposure `6/6`，censor `0/2`，quarantine `1/0`，`snapshot_match=false`。
- T031 acceptance exit `2`：provenance `112/112`、config `72/72`、decision `43/43`、lifecycle/evidence `55 pass / 23 fail`、economics `6/6`。
- T026 exact acceptance exit `0`，lifecycle/evidence `78/78 pass`。
- T016/T022 exact acceptance 均 exit `2`，lifecycle/evidence 均为 `66 pass / 12 fail`。
- 全程 offline；未触发 live、private/account、order、cancel、network、remote 或 service 操作。

done：
- T035 QA 的 unhashable method crash 和 malformed historical envelope false-pass 均在 producer、independent helper 和 Task 12 层 fail closed。
- Parser 对本任务覆盖的 hostile JSON method/result/order identity domain 是 total 的。
- 修复 source 已准备进入独立 QA；通过前不允许新 private read、live 或 adaptive/multi-level activation。

blockers：
- 独立 QA acceptance。

commit：
- ba379819964a31d3e763586440a16c66185ec9d2

提交信息：
- Make history envelope parsing fail closed
